"""``cuda_flat_pwa_gen_s0g`` — the s0f fixed-wave cache, free block UN-STAGED.

This is the kernel half.  It is IMMUTABLE per mask, like every other kernel
backend: the constructor takes the columns AND their values
(``fixed_cols=`` + ``fixed_values=``), builds the wave permutation, the free-only
``gamma``/CSR and the context for it, and never changes afterwards.  This is
the RAW kernel form (what these tests and direct users call); the
``fixed_coeff`` wrapper always supplies both; there is no lazy fill and no
snapshot — the values ARE the declaration.
The values are what ``load_data`` fills the ``Afix`` cache from, so a mask
without them is refused.  The mask LIFECYCLE — resolving a mask from
names/columns, swapping the kernel when the caller changes it, retaining the
samples across that swap — lives in the ``fixed_coeff`` wrapper backend
(:mod:`tabpwa.backends._fixed_coeff`).  Construct this class directly only when
the mask is known up front (or through ``fixed_coeff`` otherwise).

The masking itself: a wave whose every CSR column is fixed has an
iteration-invariant denominator, so its whole contribution
``Afix = Amp·conj(D)/|D|²`` is cached once per handle; the wave axis is
permuted (fixed first) and ``gamma``/the CSR are built for the FREE waves.
Unlike s0f — which runs F (``A1 = Σ ck·Afix``), the tile over the free waves
and B (``dck = Σ dA·Afix``) as three passes — s0g sums BOTH blocks inside one
per-element loop:

    per (e, p):  A = Σ_{k<nfix} ck_k·Afix[e,k,p]
                   + Σ_{k<N} ck_k·Amp[e,slot(k),p]·conj(D_k)/|D_k|²
                 with D_k = Σ_m coeff_m·Z_m(e)   (walked here, not staged)

so there is no F, no B, no ``A1``, no per-event ``dA`` buffer and no scratch of
any kind; a block carries hundreds of events instead of one log-sum group.

``n_fix == 0`` is the parent's path (no cache, no fixed loops).  A multi-BW
chain (``n_res != 1``) raises naming ``cuda_flat_pwa_gen2``;
``resolution_size > 1`` whose group does not fit one block raises naming
``cuda_flat_pwa_gen_s0f``.  ``grads["coeff"][m] == 0`` for a fixed wave's
columns is the accepted contract (no ``D``, no ``pv``).  The guard raises
``ValueError`` when a declared-fixed column moves; it compares against the
DECLARED VALUES on every compute, the first included.

Host contract::

    params = {"ck": complex (n_wave,), "coeff": real (n_basis,)}
    grads  = {"ck", "coeff", "norm"}      # norm only on a normed call
"""
import numpy as np

from tabpwa.backends.core import ComputeBackend, register_backend
from tabpwa.fixed_coeff_spec import (resolve_mask_and_values,
                                    warn_unfixed_mask)

_ffi = None
_lib = None

#: raised when a declared-fixed coefficient moves after the cache exists
FIXED_CHANGED = (
    "cuda_flat_pwa_gen_s0g: a coefficient declared fixed changed after the "
    "cache was built — fixed columns only; use a free-column backend when "
    "they float")


def uslot_map(basis_table, basis_m_index):
    """Dedup the basis interpolation on ``(row bytes, mass column)``.

    Returns ``(uslot_row, uslot_col, slot_of_basis)``: per surviving slot the
    table row and the mass column it reads, and the ``(n_basis,)`` map from a
    basis row onto its slot.

    The key is the row's **own bytes** plus the event column it is sampled
    at.  Both are byte-exact table data, so the fold is lossless.  It is NOT
    ``basis_kind``: that is the factor-local slot index (0 = s0, 1 = "one",
    2 = running width, …), so a kind-keyed fold merges resonances that share
    a slot index but not a row — measured 16 "slots" on the real config where
    the true answer is 43, i.e. silently wrong denominators.
    """
    bas = np.ascontiguousarray(np.asarray(basis_table, np.complex128))
    bmi = np.ascontiguousarray(np.asarray(basis_m_index, np.int64))
    n = int(bas.shape[0])
    if bmi.size != n:
        raise ValueError(
            f"cuda_flat_pwa_gen_s0g: basis_m_index has {bmi.size} entries, "
            f"expected n_basis = {n}")
    rows = np.ascontiguousarray(bas.real), np.ascontiguousarray(bas.imag)
    seen, row_of, col_of = {}, [], []
    slot = np.zeros(n, np.int32)
    for m in range(n):
        key = (rows[0][m].tobytes(), rows[1][m].tobytes(), int(bmi[m]))
        u = seen.get(key)
        if u is None:
            u = len(row_of)
            seen[key] = u
            row_of.append(m)
            col_of.append(int(bmi[m]))
        slot[m] = u
    return (np.asarray(row_of, np.int32), np.asarray(col_of, np.int32), slot)


def amp_slot_map(matrix_angle, fl_table, fl_q_index, num_table, num_m_index,
                 n_wave, n_proj):
    """Dedup the ck/coeff-independent numerator ``Amp = fa·fl·num`` per wave.

    A wave's whole numerator is fixed by its table rows, so two waves with
    byte-identical rows produce byte-identical amplitudes: the fold is exact
    and never looks at a sampled value.

    The key covers **every projection** — ``matrix_angle`` is p-major, so
    column ``p·n_wave + k`` is projection ``p`` of wave ``k`` and a wave's
    slot must carry all ``n_proj`` of them (a key that took one column would
    merge waves whose other projection differs).

    Returns ``(slot_of_wave, rep_of_slot)``.
    """
    am = np.ascontiguousarray(np.asarray(matrix_angle, np.complex128))
    fl = np.ascontiguousarray(np.asarray(fl_table, np.float64))
    fq = np.ascontiguousarray(np.asarray(fl_q_index, np.int64))
    nt = np.ascontiguousarray(np.asarray(num_table, np.complex128))
    nmi = np.ascontiguousarray(np.asarray(num_m_index, np.int64))
    nw, P = int(n_wave), int(n_proj)
    if am.shape[1] != P * nw:
        raise ValueError(
            f"cuda_flat_pwa_gen_s0g: matrix_angle has {am.shape[1]} columns, "
            f"expected n_proj·n_wave = {P}·{nw}")
    seen, rep = {}, []
    slot = np.zeros(nw, np.int32)
    for k in range(nw):
        key = []
        for p in range(P):
            col = am[:, p * nw + k]
            key.append(col.real.tobytes())
            key.append(col.imag.tobytes())
        for d in range(fl.shape[1]):
            key.append(fl[k, d].tobytes())
            key.append(int(fq[k, d]))
        for r in range(nt.shape[1]):
            key.append(nt[k, r].real.tobytes())
            key.append(nt[k, r].imag.tobytes())
            key.append(int(nmi[k, r]))
        key = tuple(key)
        s = seen.get(key)
        if s is None:
            s = len(rep)
            seen[key] = s
            rep.append(k)
        slot[k] = s
    return slot, np.asarray(rep, np.int32)


def index_runs(gamma_matrix, n_basis=None):
    """Per-wave CSR of the ``n_res == 1`` gamma's TRUE non-zeros.

    Returns ``(gs_lo, gs_hi, nz_k, nz_m)``.  ``[gs_lo[k], gs_hi[k])`` indexes
    the flat ``(wave, column)`` lists.  Interior holes are simply absent, so
    a holed row costs its non-zeros rather than its span — the real config has
    holes in 23 of its 26 waves.  An all-zero row raises: ``D`` would be 0.
    """
    g = np.ascontiguousarray(np.asarray(gamma_matrix, np.float64))
    if g.ndim == 3:
        if g.shape[1] != 1:
            raise ValueError(
                "cuda_flat_pwa_gen_s0g needs exactly ONE BW per chain "
                f"(top -> R + X), got n_res = {g.shape[1]} — use "
                "cuda_flat_pwa_gen2 for multi-BW chains")
        g = g[:, 0, :]
    if g.ndim != 2:
        raise ValueError(
            f"cuda_flat_pwa_gen_s0g: gamma_matrix must be (n_wave, n_basis) "
            f"at n_res == 1, got shape {g.shape}")
    nw = int(g.shape[0])
    if n_basis is not None and int(n_basis) != int(g.shape[1]):
        raise ValueError(
            f"cuda_flat_pwa_gen_s0g: gamma_matrix has {g.shape[1]} columns, "
            f"expected n_basis = {int(n_basis)}")
    gs_lo = np.zeros(nw, np.int32)
    gs_hi = np.zeros(nw, np.int32)
    nz_k, nz_m = [], []
    for k in range(nw):
        idx = np.nonzero(g[k])[0]
        if idx.size == 0:
            raise ValueError(
                f"cuda_flat_pwa_gen_s0g: wave {k} selects no basis column "
                f"(gamma row is all zero) — the denominator would be 0")
        gs_lo[k], gs_hi[k] = len(nz_m), len(nz_m) + idx.size
        nz_k.extend([k] * idx.size)
        nz_m.extend(int(m) for m in idx)
    return (gs_lo, gs_hi,
            np.asarray(nz_k, np.int32), np.asarray(nz_m, np.int32))


def wave_plan(mask, gs_lo, gs_hi, nz_m):
    """``(perm, inv_perm, fixed_wave)`` from the declared fixed-column mask.

    ``fixed_wave[k] = all(mask[nz_m[gs_lo[k]:gs_hi[k]]])`` — a wave is
    cached only when ALL of its CSR columns are declared fixed (a partly-fixed
    wave keeps a ``D`` that moves, so it must stay on the free path).  (Read
    as "no column of the wave is free"; the mask is 1 = fixed.)

    ``perm`` puts the fixed waves first and keeps the original order inside
    each group; it is the IDENTITY when nothing is fixed.  ``inv_perm`` is its
    inverse (both index spaces are the model wave axis).  The unmasked path is
    the PARENT's arithmetic, not its bit pattern: the summation order differs
    (measured ~2e-15, i.e. ten orders below the fp32 cache floor).
    """
    m = np.asarray(mask, np.int8).ravel()
    n_wave = int(np.asarray(gs_lo).size)
    nz_m = np.asarray(nz_m, np.int64)
    fixed_wave = np.zeros(n_wave, bool)
    for k in range(n_wave):
        lo, hi = int(gs_lo[k]), int(gs_hi[k])
        fixed_wave[k] = bool(m[nz_m[lo:hi]].all())
    perm = np.concatenate([np.nonzero(fixed_wave)[0],
                           np.nonzero(~fixed_wave)[0]]).astype(np.int32)
    inv_perm = np.empty(n_wave, np.int32)
    inv_perm[perm] = np.arange(n_wave, dtype=np.int32)
    return perm, inv_perm, fixed_wave


def split_waves(gs_lo, gs_hi, nz_k, nz_m, nz_u, nz_pair, fixed_wave):
    """The free/fixed CSR split (the free one renumbered ``0..n_free-1``).

    ``nz_k``/``nz_m``/``nz_u``/``nz_pair`` are the ORIGINAL full CSR; the
    return is a dict with the free arrays (``gs_lo``/``gs_hi``/``nz_*``, wave
    indices local to the free block) and the fixed arrays (``fix_lo``/
    ``fix_hi``/``fix_pair``, each wave's ORIGINAL entries).
    """
    gs_lo = np.asarray(gs_lo, np.int64)
    gs_hi = np.asarray(gs_hi, np.int64)
    nz_k = np.asarray(nz_k, np.int64)
    nz_m = np.asarray(nz_m, np.int64)
    nz_u = np.asarray(nz_u, np.int64)
    nz_pair = np.asarray(nz_pair, np.int64)
    fw = np.asarray(fixed_wave, bool)
    n_wave = int(gs_lo.size)
    free_waves = np.nonzero(~fw)[0]
    fixed_waves = np.nonzero(fw)[0]
    local = np.full(n_wave, -1, np.int64)
    local[free_waves] = np.arange(free_waves.size, dtype=np.int64)

    keep_free = ~fw[nz_k]
    keep_fix = fw[nz_k]
    counts = gs_hi - gs_lo

    def _starts(waves):
        c = counts[waves]
        out = np.zeros(c.size, np.int64)
        if c.size > 1:
            out[1:] = np.cumsum(c)[:-1]
        return out

    lo_f = _starts(free_waves)
    lo_x = _starts(fixed_waves)
    return {
        "free_waves": free_waves,
        "fixed_waves": fixed_waves,
        "n_free": int(free_waves.size),
        "n_fix": int(fixed_waves.size),
        "gs_lo": lo_f.astype(np.int32),
        "gs_hi": (lo_f + counts[free_waves]).astype(np.int32),
        "nz_k": local[nz_k[keep_free]].astype(np.int32),
        "nz_m": nz_m[keep_free].astype(np.int32),
        "nz_u": nz_u[keep_free].astype(np.int32),
        "nz_pair": np.ascontiguousarray(nz_pair[keep_free]),
        "fix_lo": lo_x.astype(np.int32),
        "fix_hi": (lo_x + counts[fixed_waves]).astype(np.int32),
        "fix_pair": np.ascontiguousarray(nz_pair[keep_fix]),
    }


def _load_lib():
    from cffi import FFI
    from tabpwa.cuda.build import ensure
    global _ffi
    if _ffi is None:
        _ffi = FFI()
        _ffi.cdef("""
void* cuda_create_context_flat_pwa_gen_s0g(
    const double*,int, const int*,int,
    const double*,const double*,int, const int*,int,
    const double*,const double*,int, const int*,int,
    const int*,int, const int*,int,
    const int*,int, const int*,int, int,int,
    const int*,int, const int*,int, const int*,int,
    const long long*,int,
    const int*,int, const int*,int,
    const int*,int, const int*,int, const int*,int,
    const long long*,int,
    int,int,
    const double*,int,
    const int*,int,
    const double*,int, const double*,int,
    const double*,const double*,int,
    double,double,int, double,double,int,
    int,int,int,int,int, int,int,int,int,int, int,int,int,
    const double*,int);
void* cuda_load_data_flat_pwa_gen_s0g(void*,const double*,const double*,
    const double*,const double*,const double*,int);
void cuda_free_data_flat_pwa_gen_s0g(void*);
void cuda_free_context_flat_pwa_gen_s0g(void*);
void cuda_compute_flat_pwa_gen_s0g(void*,void*,const double*,const double*,
    const double*,double,int, double*,double*,double*,double*,double*,double*);
int cuda_get_device_count_flat_pwa_gen_s0g(void);
int cuda_handle_afix_ready_flat_pwa_gen_s0g(void*);
int cuda_has_fixed_values_flat_pwa_gen_s0g(void*);
long long cuda_tile_bytes_flat_pwa_gen_s0g(int,int,int,int,int);
long long cuda_fix_bytes_flat_pwa_gen_s0g(void*,int);
int cuda_n_fix_flat_pwa_gen_s0g(void*);
int cuda_n_free_flat_pwa_gen_s0g(void*);
int cuda_n_nz_free_flat_pwa_gen_s0g(void*);
int cuda_n_nz_fix_flat_pwa_gen_s0g(void*);
int cuda_tile_count_flat_pwa_gen_s0g(void*);
int cuda_te_for_flat_pwa_gen_s0g(int,int,int,int,int);
void cuda_chunk_plan_flat_pwa_gen_s0g(void*,int,int*);
int cuda_smem_bytes_flat_pwa_gen_s0g(void*);
int cuda_cap_flat_pwa_gen_s0g(void*);
void cuda_stats_flat_pwa_gen_s0g(int, int*);
int cuda_ngroups_flat_pwa_gen_s0g(void*);
int cuda_te_flat_pwa_gen_s0g(void*);
void cuda_get_device_name_flat_pwa_gen_s0g(char*,int);
""")
    lib_path = ensure("kernels_flat_pwa_gen_s0g.cu",
                      "libcuda_kernels_flat_pwa_gen_s0g.so")
    if not lib_path:
        raise RuntimeError("cuda_flat_pwa_gen_s0g: library not found")
    return _ffi.dlopen(lib_path)


class FlatPwaGenS0GHandle:
    """One loaded sample for THIS kernel (one mask)."""

    __slots__ = ("ptr", "_lib", "ne", "_keep", "__weakref__")

    def __init__(self, ptr, lib, ne):
        self.ptr = ptr
        self._lib = lib
        self.ne = int(ne)
        self._keep = []

    def free(self):
        """Idempotent: the second call (and a NULL handle) is a no-op."""
        if self.ptr is not None:
            self._lib.cuda_free_data_flat_pwa_gen_s0g(self.ptr)
            self.ptr = None
            self._keep.clear()

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class CUDAKernelFlatPwaGenS0G:
    """The flat ``pwa_gen`` tile (n_res == 1) with the free block un-staged,
    built for ONE coefficient mask.

    One kernel per chunk: the fixed cache's block and the free block are summed
    in the same per-``(e, p)`` loop, so there is no F/B pass and no scratch.
    """

    name = "cuda_flat_pwa_gen_s0g"

    def __init__(self, config, batch_size=50000, resolution_size=1,
                 tile_events=None, te=0, fixed_cols=None,
                 fixed_values=None):
        global _lib
        if "fl_index" in config or "num_index" in config:
            raise ValueError(
                "cuda_flat_pwa_gen_s0g: pass the FULL (non-deduped) config — "
                "this backend reads every row where the builder emits it, and "
                "a config that already carries fl_index/num_index has a "
                "different row layout")
        resolution_size = int(resolution_size)
        if resolution_size < 1:
            raise ValueError(
                "cuda_flat_pwa_gen_s0g: resolution_size must be >= 1, got "
                f"{resolution_size}")
        if _lib is None:
            _lib = _load_lib()
        self._lib = _lib
        self.config = config
        if _lib.cuda_get_device_count_flat_pwa_gen_s0g() <= 0:
            raise RuntimeError("cuda_flat_pwa_gen_s0g: no CUDA devices")
        buf = _ffi.new("char[256]")
        _lib.cuda_get_device_name_flat_pwa_gen_s0g(buf, 256)
        self.device_name = _ffi.string(buf).decode()

        c = config
        #: ``n_wave`` / ``n_proj`` / ``n_res`` / ``n_decay`` from the shapes the
        #: builder emits — never a stored count (that is the flat config's rule)
        self.n_wave = int(np.asarray(c["fl_table"]).shape[0])
        self.n_proj = max(1, int(np.asarray(c["matrix_angle"]).shape[1])
                          // self.n_wave)
        self.n_res = int(np.asarray(c["num_table"]).shape[1])
        if self.n_res != 1:
            raise ValueError(
                "cuda_flat_pwa_gen_s0g needs exactly ONE BW per chain "
                f"(top -> R + X), got n_res = {self.n_res} — use "
                "cuda_flat_pwa_gen2 for multi-BW chains")
        self.n_decay = int(np.asarray(c["fl_q_index"]).shape[1])
        self.n_ab = int(np.asarray(c["angle_k"]).shape[0])
        self.n_comp = int(np.asarray(c["angle_k"]).shape[1])
        self.n_mass = int(np.max(c["num_m_index"])) + 1
        self.n_momentum = int(np.max(c["fl_q_index"])) + 1
        self.n_angle_total = int(np.max(c["angle_index"])) + 1
        self.n_basis = int(np.asarray(c["basis_table"]).shape[0])
        self.batch_size = int(batch_size)
        self.resolution_size = resolution_size
        self._te_hint = int(te) if te else (
            0 if tile_events is None else int(tile_events))
        self._last_dnorm = None

        # ── the mask this kernel is built for (one, immutable) ───────────
        # ONE declaration: ``fixed_cols`` AND ``fixed_values``.  A mask
        # without values is an error — the cache is filled in ``load_data``,
        # so there is no later coefficient read — and values without a mask
        # too.  Names go through ``fixed_coeff_spec.resolve_fixed_cols``.
        m, self.fixed_values = resolve_mask_and_values(
            fixed_cols, fixed_values, self.n_basis, name="cuda_flat_pwa_gen_s0g")
        self.mask = None if m is None else np.ascontiguousarray(m).copy()
        cf = (np.zeros(self.n_basis, np.int8) if self.mask is None
              else self.mask)

        bas = np.asarray(c["basis_table"], np.complex128)
        gam_full = np.asarray(c["gamma_matrix"], np.float64)

        # ── the two structural dedups ─────────────────────────────────────
        (self.uslot_row, self.uslot_col,
         self.slot_of_basis) = uslot_map(bas, c["basis_m_index"])
        self.n_uslot = int(self.uslot_row.size)

        self.slot_of_wave_orig, self.rep_of_slot = amp_slot_map(
            c["matrix_angle"], c["fl_table"], c["fl_q_index"],
            c["num_table"], c["num_m_index"], self.n_wave, self.n_proj)
        self.n_uniq = int(self.rep_of_slot.size)

        # ── the full CSR of the gamma's TRUE non-zeros ────────────────────
        (self.gs_lo_full, self.gs_hi_full,
         self.nz_k_full, self.nz_m_full) = index_runs(gam_full, self.n_basis)
        self.n_nz_full = int(self.nz_m_full.size)
        self.nz_u_full = np.ascontiguousarray(
            self.slot_of_basis[self.nz_m_full], np.int32)
        self.nz_pair_full = np.ascontiguousarray(
            (self.nz_u_full.astype(np.int64) << 32)
            | self.nz_m_full.astype(np.int64))

        # ── the mask's plan: permutation, free-only CSR/gamma, slot order ─
        (self.perm, self.inv_perm,
         self.fixed_wave) = wave_plan(cf, self.gs_lo_full, self.gs_hi_full,
                                      self.nz_m_full)
        self.n_fix = int(self.fixed_wave.sum())
        self.n_free = self.n_wave - self.n_fix
        sp = split_waves(self.gs_lo_full, self.gs_hi_full, self.nz_k_full,
                         self.nz_m_full, self.nz_u_full, self.nz_pair_full,
                         self.fixed_wave)
        self.free_waves = sp["free_waves"]
        self.fixed_waves = sp["fixed_waves"]
        self.gs_lo, self.gs_hi = sp["gs_lo"], sp["gs_hi"]
        self.nz_k, self.nz_m = sp["nz_k"], sp["nz_m"]
        self.nz_u, self.nz_pair = sp["nz_u"], sp["nz_pair"]
        self.fix_lo, self.fix_hi = sp["fix_lo"], sp["fix_hi"]
        self.fix_pair = sp["fix_pair"]
        self.n_nz = int(self.nz_m.size)
        self.n_nz_fix = int(self.fix_pair.size)
        self.slot_of_wave = np.ascontiguousarray(
            self.slot_of_wave_orig[self.perm], np.int32)
        self.fix_slot_of_wave = np.ascontiguousarray(
            self.slot_of_wave_orig[self.fixed_waves], np.int32)
        gam2 = gam_full.reshape(self.n_wave * self.n_res, self.n_basis)
        self.gamma = np.ascontiguousarray(gam2[self.free_waves])

        # a non-empty mask that caches nothing is a silent no-op: say so once
        # (the `fixed_coeff` wrapper suppresses this duplicate and emits its
        # own, so a wrapper-driven mask warns exactly once)
        warn_unfixed_mask(config["gamma_matrix"], self.mask,
                          config.get("den_names"),
                          name="cuda_flat_pwa_gen_s0g", stacklevel=3)

        #: THE DECLARATION, as the int8 mask and as column indices.  `mask`
        #: is what the kernel was built for; `declared_fixed` is the same
        #: thing as indices.  NOT to be confused with `watched_cols`: the
        #: declaration may include columns no cached wave reads.
        self.mask = self.mask            # (int8 [n_basis] or None)
        self.declared_fixed = np.nonzero(cf)[0].astype(np.int64)
        #: the invalidation WATCH LIST — only the columns the CACHED waves
        #: read.  The guard compares THESE, because the others do not enter
        #: the cache.  It is NOT the declaration: `CUDAKernel(kc,
        #: fixed_cols=k.watched_cols)` is a DIFFERENT (usually empty) mask.
        if self.fixed_waves.size:
            cols = [self.nz_m_full[self.gs_lo_full[k]:self.gs_hi_full[k]]
                    for k in self.fixed_waves]
            self.watched_cols = np.unique(np.concatenate(cols)).astype(np.int64)
        else:
            self.watched_cols = np.zeros(0, np.int64)

        #: the mask-INDEPENDENT inputs kept for the create call
        self._bas = bas
        self._fl_tab = np.asarray(c["fl_table"], np.float64).reshape(
            self.n_wave * self.n_decay, -1)
        self._num = np.asarray(c["num_table"], np.complex128).reshape(
            self.n_wave * self.n_res, -1)
        self._fl_q = np.asarray(c["fl_q_index"], np.int64)
        self._num_m = np.asarray(c["num_m_index"], np.int64)

        # ── a log-sum group must fit ONE block ────────────────────────────
        # s0g's group close happens inside a block (that is what removes the
        # staged `common`), so `rsize` events have to fit `te`.  When they do
        # not, s0f is the backend: it walks a group inside one block.
        te_max = int(self._lib.cuda_te_for_flat_pwa_gen_s0g(
            resolution_size, self.n_proj, self.n_free, self.n_wave,
            self.n_basis))
        if te_max <= 0:
            raise ValueError(
                "cuda_flat_pwa_gen_s0g: resolution_size "
                f"{resolution_size} does not fit one block for this shape "
                f"(n_wave {self.n_wave}, n_proj {self.n_proj}, n_basis "
                f"{self.n_basis}) — use cuda_flat_pwa_gen_s0f, which walks a "
                "group inside one block")
        self.te_max = te_max

        self._ctx = None
        self._keep = []
        self._ctx = self._create_context()
        self.loads = 0

    def _create_context(self):
        """Build the C context for THIS mask."""
        lib = self._lib
        c = self.config
        keep_alive = []

        def D(a):
            arr = np.ascontiguousarray(a, np.float64)
            keep_alive.append(arr)
            return _ffi.cast("double*", _ffi.from_buffer(arr))

        def I(a):
            arr = np.ascontiguousarray(a, np.int32)
            keep_alive.append(arr)
            return _ffi.cast("int*", _ffi.from_buffer(arr))

        def L(a):
            arr = np.ascontiguousarray(a, np.int64)
            keep_alive.append(arr)
            return _ffi.cast("long long*", _ffi.from_buffer(arr))

        fl_tab, num = self._fl_tab, self._num
        ctx = lib.cuda_create_context_flat_pwa_gen_s0g(
            D(fl_tab), fl_tab.size, I(self._fl_q), self._fl_q.size,
            D(num.real), D(num.imag), num.size, I(self._num_m),
            self._num_m.size,
            D(self._bas.real), D(self._bas.imag), self._bas.size,
            I(c["basis_m_index"]), self.n_basis,
            I(self.uslot_row), int(self.uslot_row.size),
            I(self.uslot_col), int(self.uslot_col.size),
            I(self.slot_of_wave), int(self.slot_of_wave.size),
            I(self.rep_of_slot), int(self.rep_of_slot.size),
            self.n_uniq, self.n_uslot,
            I(self.nz_k), int(self.nz_k.size),
            I(self.nz_m), int(self.nz_m.size),
            I(self.nz_u), int(self.nz_u.size),
            L(self.nz_pair), int(self.nz_pair.size),
            I(self.gs_lo), int(self.gs_lo.size),
            I(self.gs_hi), int(self.gs_hi.size),
            I(self.fix_lo), int(self.fix_lo.size),
            I(self.fix_hi), int(self.fix_hi.size),
            I(self.fix_slot_of_wave), int(self.fix_slot_of_wave.size),
            L(self.fix_pair), int(self.fix_pair.size),
            self.n_fix, self.n_free,
            D(self.gamma), int(self.gamma.size),
            I(np.asarray(c["angle_index"])), self.n_ab,
            D(np.asarray(c["angle_k"], np.float64)),
            int(np.asarray(c["angle_k"]).size),
            D(np.asarray(c["angle_b"], np.float64)),
            int(np.asarray(c["angle_b"]).size),
            D(np.asarray(c["matrix_angle"], np.complex128).real),
            D(np.asarray(c["matrix_angle"], np.complex128).imag),
            int(np.asarray(c["matrix_angle"]).size),
            float(c["m_min"]), float(c["m_delta"]), int(self._bas.shape[1]),
            float(c["q_min"]), float(c["q_delta"]), int(fl_tab.shape[1]),
            self.n_wave, self.n_proj, self.n_res, self.n_decay,
            self.n_basis, self.n_ab, self.n_comp,
            self.n_mass, self.n_momentum, self.n_angle_total,
            self.batch_size, self.resolution_size,
            int(self._te_hint),
            (D(self.fixed_values) if self.fixed_values is not None
             else _ffi.NULL),
            (self.n_basis if self.fixed_values is not None else 0))
        if not ctx:
            raise RuntimeError(
                "cuda_flat_pwa_gen_s0g: context creation failed")
        self._keep = keep_alive
        self.smem_bytes = int(lib.cuda_smem_bytes_flat_pwa_gen_s0g(ctx))
        self.cap = int(lib.cuda_cap_flat_pwa_gen_s0g(ctx))
        self.ngroups = int(lib.cuda_ngroups_flat_pwa_gen_s0g(ctx))
        self.te = int(lib.cuda_te_flat_pwa_gen_s0g(ctx))
        assert int(lib.cuda_n_fix_flat_pwa_gen_s0g(ctx)) == self.n_fix
        assert int(lib.cuda_n_free_flat_pwa_gen_s0g(ctx)) == self.n_free
        return ctx

    # ── data ────────────────────────────────────────────────────────────
    def load_data(self, data):
        """Build a FRESH handle for this kernel's mask."""
        mass = np.ascontiguousarray(np.asarray(data["mass"], np.float64))
        mom = np.ascontiguousarray(np.asarray(data["q"], np.float64))
        ang = np.ascontiguousarray(np.asarray(data["angle"], np.float64))
        if ang.ndim == 2:
            ang = ang.reshape(ang.shape[0], self.n_angle_total, self.n_comp)
        ne = mass.shape[0]
        w = np.ascontiguousarray(np.asarray(
            data.get("weight", np.ones(ne)), np.float64).ravel())
        b = data.get("bkg", np.zeros(ne))
        b = np.ascontiguousarray(np.asarray(
            np.full(ne, float(b)) if np.isscalar(b) else b,
            np.float64).ravel())
        keep = [mass, mom, ang, w, b]
        ptr = self._lib.cuda_load_data_flat_pwa_gen_s0g(
            self._ctx,
            _ffi.cast("double*", _ffi.from_buffer(mass)),
            _ffi.cast("double*", _ffi.from_buffer(mom)),
            _ffi.cast("double*", _ffi.from_buffer(ang)),
            _ffi.cast("double*", _ffi.from_buffer(w)),
            _ffi.cast("double*", _ffi.from_buffer(b)), ne)
        if not ptr:
            raise RuntimeError(
                "cuda_flat_pwa_gen_s0g: load_data failed (device allocation) —"
                " with a mask on, the fixed-wave cache is "
                f"{self.fix_bytes(ne)} B for this sample "
                f"(ne {ne}, n_fix {self.n_fix}, n_proj {self.n_proj})")
        h = FlatPwaGenS0GHandle(ptr, self._lib, ne)
        h._keep = keep
        self.loads += 1
        return h

    def free(self):
        """Context only.  Handles are owned by whoever called ``load_data``."""
        if getattr(self, "_ctx", None):
            self._lib.cuda_free_context_flat_pwa_gen_s0g(self._ctx)
            self._ctx = None
            self._keep = []

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    # ── compute ─────────────────────────────────────────────────────────
    def compute(self, params, data_handle, norm=None, return_p=True):
        if data_handle.ptr is None:
            raise ValueError(
                "cuda_flat_pwa_gen_s0g: this data handle was already freed")
        ne = data_handle.ne
        ck = np.asarray(params["ck"], np.complex128).ravel()
        coeff = np.asarray(params["coeff"], np.float64).ravel()
        if ck.size != self.n_wave:
            raise ValueError(
                f"cuda_flat_pwa_gen_s0g: ck has {ck.size} entries, expected "
                f"n_wave = {self.n_wave}")
        if coeff.size != self.n_basis:
            raise ValueError(
                f"cuda_flat_pwa_gen_s0g: coeff has {coeff.size} entries, "
                f"expected n_basis = {self.n_basis}")
        coeff = np.ascontiguousarray(coeff)

        # ── the invalidation guard (before anything is launched) ─────────
        # The values are DECLARED with the mask and the cache was filled from
        # them at `load_data`, so the check is strict on EVERY call, the first
        # included — the comparison is against the declared values, not a
        # snapshot (the snapshot/lazy branch is gone).
        if self.watched_cols.size:
            vals = coeff[self.watched_cols]
            if not np.array_equal(vals,
                                  self.fixed_values[self.watched_cols]):
                raise ValueError(FIXED_CHANGED)

        # model wave order in, model wave order out
        ckp = ck[self.perm]
        ckr = np.ascontiguousarray(ckp.real)
        cki = np.ascontiguousarray(ckp.imag)

        P = np.zeros(ne) if return_p else None
        P_c = (_ffi.cast("double*", _ffi.from_buffer(P)) if return_p
               else _ffi.NULL)
        gckr = np.zeros(self.n_wave)
        gcki = np.zeros(self.n_wave)
        gco = np.zeros(self.n_basis)
        oQ = _ffi.new("double*")
        oDn = _ffi.new("double*")
        self._lib.cuda_compute_flat_pwa_gen_s0g(
            self._ctx, data_handle.ptr,
            _ffi.cast("double*", _ffi.from_buffer(ckr)),
            _ffi.cast("double*", _ffi.from_buffer(cki)),
            _ffi.cast("double*", _ffi.from_buffer(coeff)),
            float(norm) if norm is not None else 0.0,
            1 if norm is not None else 0,
            oQ, oDn, P_c,
            _ffi.cast("double*", _ffi.from_buffer(gckr)),
            _ffi.cast("double*", _ffi.from_buffer(gcki)),
            _ffi.cast("double*", _ffi.from_buffer(gco)))
        Q = float(oQ[0])

        grads = {"ck": (gckr + 1j * gcki)[self.inv_perm], "coeff": gco}
        if norm is not None:
            self._last_dnorm = float(oDn[0])
            grads["norm"] = self._last_dnorm
        else:
            self._last_dnorm = None
        return Q, grads, (P if return_p else None)

    # ── reporting, for the fixture ──────────────────────────────────────
    @property
    def run_widths(self):
        """Free-wave gamma non-zeros (``<= n_basis``), in free-wave order."""
        return (self.gs_hi - self.gs_lo).astype(np.int64)

    def gamma_iters(self):
        """``(ref, run)`` basis-row iterations per event of phase 1."""
        return (self.n_free * self.n_basis, int(self.n_nz))

    def afix_ready(self, handle):
        """Whether this handle's `Afix` cache is already filled.

        It is true right after ``load_data`` — the load fills the cache from
        the declared values — and stays true.
        """
        return bool(self._lib.cuda_handle_afix_ready_flat_pwa_gen_s0g(handle.ptr))

    def has_fixed_values(self):
        """Whether this context carries the mask's DECLARED VALUES."""
        return bool(self._lib.cuda_has_fixed_values_flat_pwa_gen_s0g(self._ctx))

    def fix_bytes(self, ne):
        """The fixed-wave cache's size for a sample of ``ne`` events, in B."""
        return int(self._lib.cuda_fix_bytes_flat_pwa_gen_s0g(
            self._ctx, int(ne)))

    def chunk_plan(self, ne):
        """``(cap, n_chunks, aligned)`` for a sample of ``ne`` events.

        ``aligned`` is 1 when every chunk start and every non-final chunk
        length is a multiple of ``rsize`` — the invariant that stops a log-sum
        group straddling two chunks (the F/T/B interleave runs per chunk).
        """
        out = np.zeros(3, np.int32)
        self._lib.cuda_chunk_plan_flat_pwa_gen_s0g(
            self._ctx, int(ne), _ffi.cast("int*", _ffi.from_buffer(out)))
        return int(out[0]), int(out[1]), int(out[2])

    def tile_count(self):
        """Tile launches since construction — ONE per `cap` chunk."""
        return int(self._lib.cuda_tile_count_flat_pwa_gen_s0g(self._ctx))

    def groups_per_block(self):
        """``te`` in whole log-sum groups of ``rsize`` events."""
        return 1 if self.resolution_size <= 1 else self.te // self.resolution_size

    def kernel_stats(self, smem=None):
        """The compiled resource usage of the tile kernel (no launch)."""
        if smem is None:
            smem = int(self.tile_bytes())
        out = np.zeros(6, np.int32)
        self._lib.cuda_stats_flat_pwa_gen_s0g(
            int(smem), _ffi.cast("int*", _ffi.from_buffer(out)))
        return dict(regs=int(out[0]), local_bytes=int(out[1]),
                    blocks_per_sm=int(out[2]), warps_per_sm=int(out[3]),
                    max_threads=int(out[4]), static_smem=int(out[5]))

    def tile_bytes(self, te=None, n_wave=None, n_proj=None, n_basis=None,
                   n_free=None):
        """The tile size ``gps0g_layout`` gives — the kernel's own arithmetic."""
        return int(self._lib.cuda_tile_bytes_flat_pwa_gen_s0g(
            self.te if te is None else int(te),
            self.n_wave if n_wave is None else int(n_wave),
            self.n_proj if n_proj is None else int(n_proj),
            self.n_basis if n_basis is None else int(n_basis),
            self.n_free if n_free is None else int(n_free)))


@register_backend("cuda_flat_pwa_gen_s0g", model="pwa_gen")
class CUDAFlatPwaGenS0GBackend(ComputeBackend):

    """Backend wrapper for :class:`CUDAKernelFlatPwaGenS0G`.

    The mask is declared as ``fixed_cols`` + ``fixed_values`` (what the
    ``fixed_coeff`` wrapper passes); neither is the parent path.  Names are a
    helper, not an API: ``fixed_coeff_spec.resolve_fixed_cols`` turns them into
    columns.
    Changing it between fits is the ``fixed_coeff`` wrapper's job.
    """

    def __init__(self, kernel_config, batch_size=50000, resolution_size=1,
                 tile_events=None, te=0, fixed_cols=None,
                 fixed_values=None):
        self.kernel = CUDAKernelFlatPwaGenS0G(
            kernel_config, batch_size=batch_size,
            resolution_size=resolution_size, tile_events=tile_events, te=te,
            fixed_cols=fixed_cols, fixed_values=fixed_values)
        self.resolution_size = self.kernel.resolution_size

    def load_data(self, data_np):
        return self.kernel.load_data(data_np)

    def compute(self, params, data_handle, norm=None, return_p=True):
        return self.kernel.compute(params, data_handle, norm=norm,
                                   return_p=return_p)

    def free(self):
        if getattr(self, "kernel", None) is not None:
            self.kernel.free()
            self.kernel = None

    def __del__(self):
        self.free()
