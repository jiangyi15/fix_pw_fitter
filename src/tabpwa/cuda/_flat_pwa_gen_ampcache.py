"""``cuda_flat_pwa_gen_ampcache`` — the flat ``pwa_gen`` config with EVERY
coefficient pinned, so the whole per-wave amplitude is a per-handle cache.

This is the flat family's analogue of :mod:`tabpwa.cuda._v4_pwa_cache`: one
kernel, one cache, one contraction per iteration.  It is a case-1 copy of
``kernels_flat_pwa_gen_s0f.cu`` SPECIALISED to "every coefficient is pinned",
so — unlike s0f — it carries no mask, no wave permutation, no F/T/B split, no
``split_counts`` / ``aux_bytes`` / ``fused_count`` / ``chunk_plan`` plan
report and no ``bwd_mode``.  Every wave's whole amplitude contribution

    Afix[e, k, p] = Amp[e, slot(k), p] · conj(D_k) / |D_k|²,
    D_k           = Π_{l<n_res} Σ_m gamma[k,l,m]·coeff_m·Z_m(e)

is cached once per handle in ``float2`` (the flat family's cache dtype), and a
compute is the ck contraction — the whole per-iteration work:

    A[e,p] = Σ_k ck_k·Afix[e,k,p]
    P_e    = Σ_p |A[e,p]|²
    dck_k  = Σ_{e,p} dA[e,p]·Afix[e,k,p]
    dcoeff = 0 for EVERY column

NO conjugation on ``Afix`` in the backward: ``dA`` is already
``dQ/dP·conj(A)`` — the reference's ``grad_ck = Σ dQ_dA·common·fa`` with
``dQ_dA = dQ_dP·conj(A)`` — so ``dA·Afix`` is the correct Wirtinger half and
a second conjugate would turn it into ``|A|²``.

── the addressing: MEASURED, not assumed ────────────────────────────────

``Afix[e,k,p]`` (``p`` innermost, the ``k`` stride is ``n_proj``) is THE stored
layout.  The ``k``-innermost variant ``Afix[e,p·n_wave+k]`` — the same numbers,
a permutation inside each event's block — was A/B'd behind a runtime
``layout=`` flag and is a WASH: 0.999× at 0.6 M, 0.998× at 6 M, parity ≤ 5e-16,
the same 48 registers.  The tile is FP64-throughput bound (`ncu`: FP64 pipe
85 % of peak, DRAM 11 %), so the byte pattern is not the limiter.  The flag and
the losing index map were removed; the `.cu` header states the measurement.

The denominator is the reference's own form — a PRODUCT over the factor axis
(``n_res`` Breit-Wigners per chain, in the flat config's
``(n_wave, n_res, n_basis)`` gamma), each factor one sparse dot over the basis
rows that factor selects.  ``n_res == 1`` is the single-row special case and
takes the same one dot and one reciprocal, so it is unchanged.

``grads["coeff"] == 0.0`` for every column is the CONTRACT, not an
approximation: a pinned coefficient has no direction in ``x``, and the
constraint layer drops those entries.

── the fill, and the guard: ONE lifecycle ────────────────────────────────

``cuda_v4_pwa_cache``'s, exactly: the cache is filled ONCE PER HANDLE on that
handle's FIRST ``compute``, from ``params["coeff"]``.  ``load_data`` never
fills — it hands back a handle with ``_filled = False`` and ``_coeff = None`` —
and nothing about the pinned values lives in the constructor, the kernel config
or a module-level place.  There is no ``coeff=`` argument: one way to reach the
state, so the state cannot be reached twice, differently.

After that first call the handle is strict on EVERY call
(``compute_gram`` included, which is why it takes ``coeff``): a ``coeff`` that
differs from ``_coeff`` raises ``ValueError`` instead of re-filling or zeroing
the gradient, because a stale cache would silently return the objective of a
different point.  The snapshot is PER HANDLE, so the fitter's data and
phase-space handles each fill once from the same vector and each guard their
own, exactly as ``cuda_v4_pwa_cache`` keeps a per-handle ``_fixed_mg``.

Host contract::

    params = {"ck": complex (n_wave,), "coeff": real (n_basis,)}  # pinned
    grads  = {"ck", "coeff", "norm"}      # coeff exactly 0, norm on a normed call

The grouped log-sum (``resolution_size``) is s0f's fused branch: a block owns
whole groups (``rsize <= te``, ``rsize == 1`` being the per-event objective),
and a group too large for one block's tile is walked by one block in
``te``-sized inner chunks (s0f's path-B semantics, where its F/T/B fallback
ends up).

── the Gram ──────────────────────────────────────────────────────────────

``compute_gram(handle, weight=None, coeff=None)`` reduces the SAME cache into
the wave Gram ``D[k,j] = Σ_e w_e Σ_p conj(Afix[e,k,p])·Afix[e,j,p]`` — the
phase-space pre-integration ``integrated_pwa_gen`` uses for its O(N²) norm.
It goes through the same ``_fill_once``: an unfilled handle needs the pinned
``coeff`` to fill (so the delegate passes ``params["coeff"]``), and a filled one
is guarded against the vector it was filled from.  There is no second way to
fill.
"""
import numpy as np

from tabpwa.backends.core import ComputeBackend, register_backend

_ffi = None
_lib = None

#: raised when a pinned coefficient moves after the cache exists (the CONTRACT
#: message — a test asserts it verbatim)
GUARD = (
    "cuda_flat_pwa_gen_ampcache: coeff moved after the cache was built — the "
    "basis weights must be pinned; use cuda_flat_pwa_gen_s0e2 when they float")


def uslot_map(basis_table, basis_m_index):
    """Dedup the basis interpolation on ``(row bytes, mass column)``.

    Returns ``(uslot_row, uslot_col, slot_of_basis)``: per surviving slot the
    table row and the mass column it reads, and the ``(n_basis,)`` map from a
    basis row onto its slot.

    The key is the row's **own bytes** plus the event column it is sampled
    at.  Both are byte-exact table data, so the fold is lossless.  It is NOT
    ``basis_kind``: that is the factor-local slot index (0 = s0, 1 = "one",
    2 = running width, …), so a kind-keyed fold merges resonances that share a
    slot index but not a row — measured 16 "slots" on the real config where
    the true answer is 43, i.e. silently wrong denominators.
    """
    bas = np.ascontiguousarray(np.asarray(basis_table, np.complex128))
    bmi = np.ascontiguousarray(np.asarray(basis_m_index, np.int64))
    n = int(bas.shape[0])
    if bmi.size != n:
        raise ValueError(
            f"cuda_flat_pwa_gen_ampcache: basis_m_index has {bmi.size} "
            f"entries, expected n_basis = {n}")
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
            f"cuda_flat_pwa_gen_ampcache: matrix_angle has {am.shape[1]} "
            f"columns, expected n_proj·n_wave = {P}·{nw}")
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


def den_runs(gamma_matrix, n_basis=None, n_res=None):
    """Per-``(wave, res)`` CSR of the denominator's TRUE non-zeros.

    The reference's own form is a PRODUCT over the factor axis::

        D_k = Π_{l<n_res}  s_{k,l},
        s_{k,l} = Σ_m gamma_matrix[k,l,m]·coeff[m]·basis_m(m_e)

    so the CSR is indexed by the FLAT ``(wave, res)`` row ``k·n_res + l``, not
    by the wave: ``[row_start[r], row_start[r+1])`` indexes the ``col``/``val``
    lists of that one row, and the wave's ``D`` is the product of its ``n_res``
    row dots.  At ``n_res == 1`` the single row IS the wave's denominator, i.e.
    the ``n_res == 1`` arithmetic unchanged (one dot, one reciprocal).

    ``gamma_matrix`` is accepted in either shape: ``(n_wave, n_basis)`` — the
    one-factor shorthand, which therefore requires ``n_res == 1`` — or
    ``(n_wave, n_res, n_basis)``, whose factor axis must match the config's
    ``n_res`` (the numerator's factor count, from ``num_table``).

    Returns ``(row_start, nz_row, nz_col, nz_val)``; ``nz_val`` carries the
    gamma weight itself (the coefficient is folded in at the fill, since it is
    the cache's parameter).  Interior holes are simply absent, so a holed row
    costs its non-zeros rather than its span — the real config has holes in 23
    of its 26 waves.  An all-zero row raises: its factor would be 0, hence
    ``D`` 0.
    """
    g = np.ascontiguousarray(np.asarray(gamma_matrix, np.float64))
    if g.ndim == 2:
        if n_res is not None and int(n_res) != 1:
            raise ValueError(
                "cuda_flat_pwa_gen_ampcache: gamma_matrix is "
                f"(n_wave, n_basis) — the ONE-factor shorthand — but the "
                f"config has n_res = {int(n_res)}; a multi-BW chain needs the "
                "(n_wave, n_res, n_basis) form")
        g = g[:, None, :]
    if g.ndim != 3:
        raise ValueError(
            "cuda_flat_pwa_gen_ampcache: gamma_matrix must be "
            "(n_wave, n_res, n_basis) or (n_wave, n_basis), got shape "
            f"{g.shape}")
    nw, nr, nb = (int(v) for v in g.shape)
    if n_basis is not None and int(n_basis) != nb:
        raise ValueError(
            f"cuda_flat_pwa_gen_ampcache: gamma_matrix has {nb} columns, "
            f"expected n_basis = {int(n_basis)}")
    if n_res is not None and int(n_res) != nr:
        raise ValueError(
            f"cuda_flat_pwa_gen_ampcache: gamma_matrix has {nr} factors per "
            f"wave, but the numerator has n_res = {int(n_res)} — the "
            "denominator product and the numerator chain must agree")
    row_start = np.zeros(nw * nr + 1, np.int32)
    nz_row, nz_col, nz_val = [], [], []
    for k in range(nw):
        for l in range(nr):
            row = k * nr + l
            idx = np.nonzero(g[k, l])[0]
            if idx.size == 0:
                raise ValueError(
                    f"cuda_flat_pwa_gen_ampcache: wave {k} factor {l} selects "
                    "no basis column (gamma row is all zero) — that "
                    "denominator factor would be 0")
            nz_row.extend([row] * idx.size)
            nz_col.extend(int(m) for m in idx)
            nz_val.extend(float(v) for v in g[k, l, idx])
            row_start[row + 1] = len(nz_col)
    return (row_start, np.asarray(nz_row, np.int32),
            np.asarray(nz_col, np.int32), np.asarray(nz_val, np.float64))


def _load_lib():
    from cffi import FFI
    from tabpwa.cuda.build import ensure
    global _ffi
    if _ffi is None:
        _ffi = FFI()
        _ffi.cdef("""
void* cuda_create_context_flat_pwa_gen_ampcache(
    const double*,int, const int*,int,
    const double*,const double*,int, const int*,int,
    const double*,const double*,int, const int*,int,
    const int*,int, const int*,int,
    const int*,int, const int*,int, int,int,
    const long long*,int, const double*,int,
    const int*,int,
    const int*,int,
    const double*,int, const double*,int,
    const double*,const double*,int,
    double,double,int, double,double,int,
    int,int,int,int,int, int,int,int,int,int, int,int,int);
void* cuda_load_data_flat_pwa_gen_ampcache(void*,const double*,const double*,
    const double*,const double*,const double*,int);
void cuda_free_data_flat_pwa_gen_ampcache(void*);
void cuda_free_context_flat_pwa_gen_ampcache(void*);
int cuda_fill_afix_flat_pwa_gen_ampcache(void*,void*,const double*);
void cuda_compute_flat_pwa_gen_ampcache(void*,void*,const double*,
    const double*,double,int, double*,double*,double*,double*,double*);
int cuda_gram_flat_pwa_gen_ampcache(void*,void*,const double*,double*);
long long cuda_gram_bytes_flat_pwa_gen_ampcache(int,int);
int cuda_gram_ok_flat_pwa_gen_ampcache(void*);
void cuda_gram_plan_flat_pwa_gen_ampcache(int,int*);
void cuda_gram_stats_flat_pwa_gen_ampcache(int,int*);
int cuda_get_device_count_flat_pwa_gen_ampcache(void);
int cuda_handle_afix_ready_flat_pwa_gen_ampcache(void*);
long long cuda_cache_bytes_flat_pwa_gen_ampcache(void*,int);
long long cuda_tile_bytes_flat_pwa_gen_ampcache(int,int,int);
int cuda_tile_count_flat_pwa_gen_ampcache(void*);
int cuda_smem_bytes_flat_pwa_gen_ampcache(void*);
int cuda_cap_flat_pwa_gen_ampcache(void*);
int cuda_te_flat_pwa_gen_ampcache(void*);
int cuda_ngroups_flat_pwa_gen_ampcache(void*);
void cuda_chunk_plan_flat_pwa_gen_ampcache(void*,int,int*);
void cuda_block_plan_flat_pwa_gen_ampcache(void*,int*);
void cuda_stats_flat_pwa_gen_ampcache(int, int*);
void cuda_fill_stats_flat_pwa_gen_ampcache(int*);
void cuda_get_device_name_flat_pwa_gen_ampcache(char*,int);
""")
    lib_path = ensure("kernels_flat_pwa_gen_ampcache.cu",
                      "libcuda_kernels_flat_pwa_gen_ampcache.so")
    if not lib_path:
        raise RuntimeError(
            "cuda_flat_pwa_gen_ampcache: library not found")
    return _ffi.dlopen(lib_path)


class FlatPwaGenAmpCacheHandle:
    """One loaded sample for THIS kernel.

    ``_filled`` / ``_coeff`` are the PER-HANDLE fill state (``cuda_v4_pwa_cache``'s
    ``_filled`` / ``_fixed_mg``): the first ``compute`` fills the cache from the
    coefficients and snapshots them here, and every later call is guarded
    against exactly that vector.
    """

    __slots__ = ("ptr", "_lib", "ne", "_keep", "_filled", "_coeff",
                 "__weakref__")

    def __init__(self, ptr, lib, ne):
        self.ptr = ptr
        self._lib = lib
        self.ne = int(ne)
        self._keep = []
        self._filled = False
        self._coeff = None

    def free(self):
        """Idempotent: the second call (and a NULL handle) is a no-op."""
        if self.ptr is not None:
            self._lib.cuda_free_data_flat_pwa_gen_ampcache(self.ptr)
            self.ptr = None
            self._keep.clear()

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class CUDAKernelFlatPwaGenAmpCache:
    """The flat ``pwa_gen`` config with every coefficient pinned.

    The coefficients are the CACHE's parameter, not the fit's: they are fixed
    for the lifetime of the cache and the guard raises when they move.
    """

    name = "cuda_flat_pwa_gen_ampcache"

    def __init__(self, config, batch_size=50000, resolution_size=1,
                 tile_events=None, te=0):
        global _lib
        if "fl_index" in config or "num_index" in config:
            raise ValueError(
                "cuda_flat_pwa_gen_ampcache: pass the FULL (non-deduped) "
                "config — this backend reads every row where the builder "
                "emits it, and a config that already carries "
                "fl_index/num_index has a different row layout")
        resolution_size = int(resolution_size)
        if resolution_size < 1:
            raise ValueError(
                "cuda_flat_pwa_gen_ampcache: resolution_size must be >= 1, "
                f"got {resolution_size}")
        if _lib is None:
            _lib = _load_lib()
        self._lib = _lib
        self.config = config
        if _lib.cuda_get_device_count_flat_pwa_gen_ampcache() <= 0:
            raise RuntimeError("cuda_flat_pwa_gen_ampcache: no CUDA devices")
        buf = _ffi.new("char[256]")
        _lib.cuda_get_device_name_flat_pwa_gen_ampcache(buf, 256)
        self.device_name = _ffi.string(buf).decode()

        c = config
        #: ``n_wave`` / ``n_proj`` / ``n_res`` / ``n_decay`` from the shapes the
        #: builder emits — never a stored count (that is the flat config's rule)
        self.n_wave = int(np.asarray(c["fl_table"]).shape[0])
        self.n_proj = max(1, int(np.asarray(c["matrix_angle"]).shape[1])
                          // self.n_wave)
        self.n_res = int(np.asarray(c["num_table"]).shape[1])
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
        #: how many handle fills this kernel has run — the ONE lifecycle's own
        #: observability (a test asserts two handles cost two fills, not four)
        self.fills = 0
        self.loads = 0

        bas = np.asarray(c["basis_table"], np.complex128)
        gam_full = np.asarray(c["gamma_matrix"], np.float64)

        # ── the two structural dedups (s0f's, verbatim in effect) ──────────
        (self.uslot_row, self.uslot_col,
         self.slot_of_basis) = uslot_map(bas, c["basis_m_index"])
        self.n_uslot = int(self.uslot_row.size)

        self.slot_of_wave, self.rep_of_slot = amp_slot_map(
            c["matrix_angle"], c["fl_table"], c["fl_q_index"],
            c["num_table"], c["num_m_index"], self.n_wave, self.n_proj)
        self.n_uniq = int(self.rep_of_slot.size)

        # ── the CSR of every (wave, factor) gamma row's TRUE non-zeros ─────
        # `nz_pair` packs (uslot << 32) | column, `nz_val` carries the gamma
        # weight, and `row_start` is the FLAT (wave·n_res + factor) row index —
        # the wave's denominator is the product over its `n_res` rows.
        (self.row_start, self.nz_row,
         self.nz_col, self.nz_val) = den_runs(
            gam_full, self.n_basis, self.n_res)
        self.n_nz = int(self.nz_col.size)
        self.n_row = int(self.row_start.size) - 1
        self.nz_u = np.ascontiguousarray(
            self.slot_of_basis[self.nz_col], np.int32)
        self.nz_pair = np.ascontiguousarray(
            (self.nz_u.astype(np.int64) << 32)
            | self.nz_col.astype(np.int64))

        #: the mask-independent inputs kept for the create call
        self._bas = bas
        self._fl_tab = np.asarray(c["fl_table"], np.float64).reshape(
            self.n_wave * self.n_decay, -1)
        self._num = np.asarray(c["num_table"], np.complex128).reshape(
            self.n_wave * self.n_res, -1)
        self._fl_q = np.asarray(c["fl_q_index"], np.int64)
        self._num_m = np.asarray(c["num_m_index"], np.int64)

        self._ctx = None
        self._keep = []
        self._ctx = self._create_context()
        self.loads = 0

    def _check_coeff_len(self, coeff):
        coeff = np.ascontiguousarray(np.asarray(coeff, np.float64).ravel())
        if coeff.size != self.n_basis:
            raise ValueError(
                f"cuda_flat_pwa_gen_ampcache: coeff has {coeff.size} entries, "
                f"expected n_basis = {self.n_basis}")
        return coeff.copy()

    def _create_context(self):
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
        ctx = lib.cuda_create_context_flat_pwa_gen_ampcache(
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
            L(self.nz_pair), int(self.nz_pair.size),
            D(self.nz_val), int(self.nz_val.size),
            I(self.row_start), int(self.row_start.size),
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
            int(self._te_hint))
        if not ctx:
            raise RuntimeError(
                "cuda_flat_pwa_gen_ampcache: context creation failed")
        self._keep = keep_alive
        self.smem_bytes = int(
            lib.cuda_smem_bytes_flat_pwa_gen_ampcache(ctx))
        self.cap = int(lib.cuda_cap_flat_pwa_gen_ampcache(ctx))
        self.ngroups = int(lib.cuda_ngroups_flat_pwa_gen_ampcache(ctx))
        self.te = int(lib.cuda_te_flat_pwa_gen_ampcache(ctx))
        return ctx

    # ── data ────────────────────────────────────────────────────────────
    def load_data(self, data):
        """Build a FRESH handle.  It NEVER fills — its first ``compute`` does."""
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
        ptr = self._lib.cuda_load_data_flat_pwa_gen_ampcache(
            self._ctx,
            _ffi.cast("double*", _ffi.from_buffer(mass)),
            _ffi.cast("double*", _ffi.from_buffer(mom)),
            _ffi.cast("double*", _ffi.from_buffer(ang)),
            _ffi.cast("double*", _ffi.from_buffer(w)),
            _ffi.cast("double*", _ffi.from_buffer(b)), ne)
        if not ptr:
            raise RuntimeError(
                "cuda_flat_pwa_gen_ampcache: load_data failed (device "
                "allocation) — the cache is "
                f"{self.cache_bytes(ne)} B for this sample "
                f"(ne {ne}, n_wave {self.n_wave}, n_proj {self.n_proj})")
        h = FlatPwaGenAmpCacheHandle(ptr, self._lib, ne)
        h._keep = keep
        self.loads += 1
        return h

    def _fill(self, handle, coeff):
        """Fill one handle's cache — the ONE place the cache is built."""
        coeff = np.ascontiguousarray(coeff, np.float64)
        ok = self._lib.cuda_fill_afix_flat_pwa_gen_ampcache(
            self._ctx, handle.ptr,
            _ffi.cast("double*", _ffi.from_buffer(coeff)))
        if not ok:
            raise RuntimeError(
                "cuda_flat_pwa_gen_ampcache: the Afix fill failed")
        self.fills += 1

    def _fill_once(self, handle, coeff):
        """``cuda_v4_pwa_cache._fill_once``'s lifecycle, per HANDLE.

        The first call is the FILL and snapshots the vector; every later call
        is strict (it re-fills nothing and raises when the coefficients moved).
        Nothing about the pinned values lives in the constructor, the config or
        a module-level place — only here, on the handle that was filled.
        """
        coeff = self._check_coeff_len(coeff)
        if handle._filled:
            if not np.array_equal(coeff, handle._coeff):
                raise ValueError(GUARD)
            return
        self._fill(handle, coeff)
        handle._filled = True
        handle._coeff = coeff.copy()

    def free(self):
        """Context only.  Handles are owned by whoever called ``load_data``."""
        if getattr(self, "_ctx", None):
            self._lib.cuda_free_context_flat_pwa_gen_ampcache(self._ctx)
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
                "cuda_flat_pwa_gen_ampcache: this data handle was already "
                "freed")
        ne = data_handle.ne
        ck = np.asarray(params["ck"], np.complex128).ravel()
        coeff = np.ascontiguousarray(
            np.asarray(params["coeff"], np.float64).ravel())
        if ck.size != self.n_wave:
            raise ValueError(
                f"cuda_flat_pwa_gen_ampcache: ck has {ck.size} entries, "
                f"expected n_wave = {self.n_wave}")
        if coeff.size != self.n_basis:
            raise ValueError(
                f"cuda_flat_pwa_gen_ampcache: coeff has {coeff.size} "
                f"entries, expected n_basis = {self.n_basis}")

        # ── the ONE lifecycle: fill this handle, or guard it (before any
        #    launch).  The first call fills and snapshots; the second and every
        #    later call is strict — zeroing the coefficient gradient and
        #    carrying on would return the objective of a DIFFERENT point.
        self._fill_once(data_handle, coeff)

        ckr = np.ascontiguousarray(ck.real)
        cki = np.ascontiguousarray(ck.imag)
        P = np.zeros(ne) if return_p else None
        P_c = (_ffi.cast("double*", _ffi.from_buffer(P)) if return_p
               else _ffi.NULL)
        gckr = np.zeros(self.n_wave)
        gcki = np.zeros(self.n_wave)
        oQ = _ffi.new("double*")
        oDn = _ffi.new("double*")
        self._lib.cuda_compute_flat_pwa_gen_ampcache(
            self._ctx, data_handle.ptr,
            _ffi.cast("double*", _ffi.from_buffer(ckr)),
            _ffi.cast("double*", _ffi.from_buffer(cki)),
            float(norm) if norm is not None else 0.0,
            1 if norm is not None else 0,
            oQ, oDn, P_c,
            _ffi.cast("double*", _ffi.from_buffer(gckr)),
            _ffi.cast("double*", _ffi.from_buffer(gcki)))
        Q = float(oQ[0])

        # `dcoeff` is EXACTLY 0 for every column, by contract: the cache holds
        # no `D`, so no coefficient has a direction here.
        grads = {"ck": gckr + 1j * gcki,
                 "coeff": np.zeros(self.n_basis)}
        if norm is not None:
            self._last_dnorm = float(oDn[0])
            grads["norm"] = self._last_dnorm
        else:
            self._last_dnorm = None
        return Q, grads, (P if return_p else None)

    # ── the Gram (the phase-space pre-integration) ───────────────────────
    def compute_gram(self, handle, weight=None, coeff=None):
        """Wave Gram ``D`` (``n_wave × n_wave`` complex) from this handle's cache.

        ``D[k,j] = Σ_e w_e Σ_p conj(Afix[e,k,p])·Afix[e,j,p]`` — conjugation on
        the FIRST index, the convention ``integrated_pwa_gen.gram`` uses
        (``einsum("e,ekp,ejp->kj", w, c.conj(), c)``), so
        ``Σ_e w_e P_e = Re(ck† D ck)`` and ``∂/∂ck = conj(D ck)`` follow
        directly.  The kernel computes the UPPER TRIANGLE and writes the mirror
        from the same product, so the Hermitian symmetry is structural; only
        the cross-block atomic order can separate the two triangles
        (``|D − D†|/max|D|`` was measured at 1.9e-16).

        ``weight`` is the per-event phase-space weight (``None`` = 1.0
        everywhere); it is UPLOADED per call, so the handle's own copy is not
        consulted.  ``coeff`` is required for an unfilled handle and is the
        vector the cache is filled from — the ONE lifecycle's ``_fill_once``,
        not a third way in.  On a filled handle it only has to MATCH.

        Peak device memory is the handle's own cache plus the N×N accumulator;
        a streamed caller (``integrated_pwa_gen``) loads one batch at a time.
        """
        if handle.ptr is None:
            raise ValueError(
                "cuda_flat_pwa_gen_ampcache: this data handle was already "
                "freed")
        if coeff is None:
            if not handle._filled:
                raise ValueError(
                    "cuda_flat_pwa_gen_ampcache: the Gram needs the pinned "
                    "coeff — pass it (`coeff=`) so the handle's cache can be "
                    "filled, or compute once on this handle first")
        else:
            self._fill_once(handle, coeff)
        ne = int(handle.ne)
        n = self.n_wave
        if weight is not None:
            w = np.ascontiguousarray(np.asarray(weight, np.float64).ravel())
            if w.size != ne:
                raise ValueError(
                    f"cuda_flat_pwa_gen_ampcache: weight has {w.size} "
                    f"entries, expected one per event ({ne})")
            wp = _ffi.cast("double*", _ffi.from_buffer(w))
        else:
            w, wp = None, _ffi.NULL
        D = np.zeros(n * n, np.complex128)
        ok = self._lib.cuda_gram_flat_pwa_gen_ampcache(
            self._ctx, handle.ptr, wp,
            _ffi.cast("double*", _ffi.from_buffer(D)))
        if not ok:
            raise RuntimeError(
                "cuda_flat_pwa_gen_ampcache: the GPU Gram build failed for "
                f"(n_wave {self.n_wave}, n_proj {self.n_proj}, ne {handle.ne}, "
                f"gram_bytes {self.gram_bytes()}, gram_ok {self.gram_ok}) — "
                "the C side printed its reason above (shared tile / unfilled "
                "Afix / launch); a shape change here takes "
                "integrated_pwa_gen's numpy builder")
        return D.reshape(n, n)

    def gram_bytes(self, n_wave=None, n_proj=None):
        """The Gram kernel's shared tile for this (or a given) shape, in B."""
        return int(self._lib.cuda_gram_bytes_flat_pwa_gen_ampcache(
            self.n_wave if n_wave is None else int(n_wave),
            self.n_proj if n_proj is None else int(n_proj)))

    @property
    def gram_ok(self):
        """Whether THIS context can build a Gram at all.

        The Gram's shared tile is ``2·N² + 8·N·P`` doubles (22 848 B at
        ``N = 34, P = 2``), which is past the 48 KiB default for a wide config
        and past every device's opt-in ceiling sooner or later (``N = 76``
        needs 102 144 B and this box offers 101 376).  The Gram is optional
        here — the pinned-cache compute path has no dependency on it — so a
        shape above the ceiling reports ``False`` and a caller WITH a fallback
        (``integrated_pwa_gen``'s numpy builder) takes it, instead of paying a
        cache fill to discover the refusal from a failed call.
        """
        return bool(self._lib.cuda_gram_ok_flat_pwa_gen_ampcache(self._ctx))

    def gram_plan(self, ne):
        """``(events_per_block, blocks, n_sm)`` for a sample of ``ne`` events."""
        out = np.zeros(3, np.int32)
        self._lib.cuda_gram_plan_flat_pwa_gen_ampcache(
            int(ne), _ffi.cast("int*", _ffi.from_buffer(out)))
        return int(out[0]), int(out[1]), int(out[2])

    def gram_kernel_stats(self, smem=None):
        """Compiled resource usage of the GRAM kernel (regs / local / blocks)."""
        if smem is None:
            smem = self.gram_bytes()
        out = np.zeros(6, np.int32)
        self._lib.cuda_gram_stats_flat_pwa_gen_ampcache(
            int(smem), _ffi.cast("int*", _ffi.from_buffer(out)))
        return dict(regs=int(out[0]), local_bytes=int(out[1]),
                    blocks_per_sm=int(out[2]), warps_per_sm=int(out[3]),
                    max_threads=int(out[4]), static_smem=int(out[5]))

    # ── reporting, for the fixture ──────────────────────────────────────
    def afix_ready(self, handle):
        """Whether this handle's DEVICE ``Afix`` flag is set (``_filled`` is
        the lifecycle's own flag; the two are set by the same fill)."""
        return bool(
            self._lib.cuda_handle_afix_ready_flat_pwa_gen_ampcache(handle.ptr))

    def cache_bytes(self, ne):
        """The cache's size for a sample of ``ne`` events, in B."""
        return int(self._lib.cuda_cache_bytes_flat_pwa_gen_ampcache(
            self._ctx, int(ne)))

    def tile_count(self):
        """Tile launches since ``create`` (ONE per chunk, nothing else)."""
        return int(self._lib.cuda_tile_count_flat_pwa_gen_ampcache(self._ctx))

    def block_plan(self):
        """``(te, per_blk, walks_group)`` for this context's tile."""
        out = np.zeros(3, np.int32)
        self._lib.cuda_block_plan_flat_pwa_gen_ampcache(
            self._ctx, _ffi.cast("int*", _ffi.from_buffer(out)))
        return int(out[0]), int(out[1]), int(out[2])

    def chunk_plan(self, ne):
        """``(cap, n_chunks, aligned)`` for a sample of ``ne`` events."""
        out = np.zeros(3, np.int32)
        self._lib.cuda_chunk_plan_flat_pwa_gen_ampcache(
            self._ctx, int(ne), _ffi.cast("int*", _ffi.from_buffer(out)))
        return int(out[0]), int(out[1]), int(out[2])

    def kernel_stats(self, smem=None):
        """The compiled resource usage of the tile kernel (no launch)."""
        if smem is None:
            smem = int(self.tile_bytes())
        out = np.zeros(6, np.int32)
        self._lib.cuda_stats_flat_pwa_gen_ampcache(
            int(smem), _ffi.cast("int*", _ffi.from_buffer(out)))
        return dict(regs=int(out[0]), local_bytes=int(out[1]),
                    blocks_per_sm=int(out[2]), warps_per_sm=int(out[3]),
                    max_threads=int(out[4]), static_smem=int(out[5]))

    def fill_kernel_stats(self):
        """The compiled resource usage of the ``Afix`` fill kernel."""
        out = np.zeros(6, np.int32)
        self._lib.cuda_fill_stats_flat_pwa_gen_ampcache(
            _ffi.cast("int*", _ffi.from_buffer(out)))
        return dict(regs=int(out[0]), local_bytes=int(out[1]),
                    blocks_per_sm=int(out[2]), warps_per_sm=int(out[3]),
                    max_threads=int(out[4]), static_smem=int(out[5]))

    def tile_bytes(self, te=None, n_wave=None, n_proj=None):
        """The tile size ``gps0a_layout`` gives — the kernel's own arithmetic."""
        return int(self._lib.cuda_tile_bytes_flat_pwa_gen_ampcache(
            self.te if te is None else int(te),
            self.n_wave if n_wave is None else int(n_wave),
            self.n_proj if n_proj is None else int(n_proj)))


@register_backend("cuda_flat_pwa_gen_ampcache", model="pwa_gen")
class CUDAFlatPwaGenAmpCacheBackend(ComputeBackend):

    """Backend wrapper for :class:`CUDAKernelFlatPwaGenAmpCache`.

    ONE lifecycle (``cuda_v4_pwa_cache``'s): no ``coeff=`` here either — each
    handle's cache is filled on that handle's FIRST ``compute`` from
    ``params["coeff"]`` and guarded per handle thereafter.

    The capability tag is declared by the ALIAS registration in
    ``tabpwa.backends.cuda_flat_pwa_gen_ampcache_backend`` (the same split
    s0f/s0g use), so the tag has exactly ONE owner: a name registered twice
    keeps only the winning module's tags, and a second declaration here could
    be silently overridden or silently win.
    """

    def __init__(self, kernel_config, batch_size=50000, resolution_size=1,
                 tile_events=None, te=0):
        self.kernel = CUDAKernelFlatPwaGenAmpCache(
            kernel_config, batch_size=batch_size,
            resolution_size=resolution_size, tile_events=tile_events, te=te)
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
