"""``cuda_flat_pwa_gen_s0e4`` — the s0e FUSED TILE on the flat ``pwa_gen`` config.

The FOURTH case-1 copy of the fused tile, and the SECOND structural change
after s0e2's packed CSR.  It is ``cuda_flat_pwa_gen_s0e`` plus **exactly two
changes**:

1. **the fitted coefficients are PRE-GATHERED into CSR order on the host**,
   ``cw[i] = coeff[nz_m[i]]`` (one ``cudaMemcpyAsync`` of ``n_nz`` doubles per
   call), so the phase-A inner loop has no coefficient indirection;
2. **a row with ``nnz > split_nnz`` is split into TWO contiguous halves
   processed by TWO threads**, so the phase barrier waits for ``~nnz/2``
   instead of ``nnz``.  Phase A becomes "accumulate ``D`` only" and the
   reciprocal + ``common`` move to a new phase A2 after a barrier (which also
   isolates the fp64 divide).

Why: on the PRODUCTION config (``config_kn4_plot.yml``: n_wave 26, n_basis 64,
n_proj 2, 130 non-zeros, nnz per wave ``{1: 5, 3: 11, 9: 8, 10: 2}``) at
``te = 8`` there are 208 ``(e,k)`` items on 256 threads — ONE item per thread —
so phase A's barrier waits for the 10-nnz row: max 10 vs mean 4.1, a **2.46x
tail**.  Sorting by nnz does not help (measured: identical); only splitting the
heavy row shortens the critical path.

``split_nnz`` is a CREATE-TIME knob (``0`` = the compiled default, 6): the
segmentation is built on the host in ``create`` and the kernel only reads
``n_seg``/``seg_*``, so the threshold never needs a rebuild.

The rest of the header (the basis rotation, the two byte-exact dedups, the
chunk rule, path A/B for the grouped log-sum) is s0e's, verbatim — see
``_flat_pwa_gen_s0e.py``.  Like every flat backend it takes the **same config
input and conversion** and adds the one requirement

    n_res == 1     (top -> R + X: every chain carries exactly one BW factor)

``n_res`` is ``num_table.shape[1]``.  A multi-BW chain raises a ``ValueError``
naming ``cuda_flat_pwa_gen2``.

Host contract, identical to the other flat backends::

    params = {"ck": complex (n_wave,), "coeff": real (n_basis,)}
    grads  = {"ck", "coeff", "norm"}      # norm only on a normed call

``grad_ck`` is the **raw Wirtinger half** (``CKProduct.backprop_grad`` applies
the 2·Re convention downstream).  Analytic gradients only; finite differences
are used in the fixture, never in the fit.
"""
import numpy as np

from tabpwa.backends.core import ComputeBackend, register_backend

_ffi = None
_lib = None


# ── the two structural dedups, in one place so a test can ask ──────────

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
            f"cuda_flat_pwa_gen_s0e4: basis_m_index has {bmi.size} entries, "
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
            f"cuda_flat_pwa_gen_s0e4: matrix_angle has {am.shape[1]} columns, "
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
                "cuda_flat_pwa_gen_s0e4 needs exactly ONE BW per chain "
                f"(top -> R + X), got n_res = {g.shape[1]} — use "
                "cuda_flat_pwa_gen2 for multi-BW chains")
        g = g[:, 0, :]
    if g.ndim != 2:
        raise ValueError(
            f"cuda_flat_pwa_gen_s0e4: gamma_matrix must be (n_wave, n_basis) "
            f"at n_res == 1, got shape {g.shape}")
    nw = int(g.shape[0])
    if n_basis is not None and int(n_basis) != int(g.shape[1]):
        raise ValueError(
            f"cuda_flat_pwa_gen_s0e4: gamma_matrix has {g.shape[1]} columns, "
            f"expected n_basis = {int(n_basis)}")
    gs_lo = np.zeros(nw, np.int32)
    gs_hi = np.zeros(nw, np.int32)
    nz_k, nz_m = [], []
    for k in range(nw):
        idx = np.nonzero(g[k])[0]
        if idx.size == 0:
            raise ValueError(
                f"cuda_flat_pwa_gen_s0e4: wave {k} selects no basis column "
                f"(gamma row is all zero) — the denominator would be 0")
        gs_lo[k], gs_hi[k] = len(nz_m), len(nz_m) + idx.size
        nz_k.extend([k] * idx.size)
        nz_m.extend(int(m) for m in idx)
    return (gs_lo, gs_hi,
            np.asarray(nz_k, np.int32), np.asarray(nz_m, np.int32))


def _te_hint(te, resolution_size, tile_events=None):
    """Validate / combine the two spellings of the tile hint.

    ``te`` (the s0e4 sweep knob) wins over ``tile_events`` (the s0e-compatible
    name).  ``0`` means "the kernel picks"; otherwise the tile must be a
    positive int that does not split a log-sum group (``rsize > 1`` requires
    a whole multiple), which is exactly what the kernel itself refuses.
    """
    te = int(te)
    if te < 0:
        raise ValueError(
            f"cuda_flat_pwa_gen_s0e4: te must be 0 (kernel's own choice) or a "
            f"positive int, got {te}")
    if te == 0:
        return 0 if tile_events is None else int(tile_events)
    rsize = int(resolution_size)
    if rsize > 1 and te >= rsize and te % rsize != 0:
        raise ValueError(
            f"cuda_flat_pwa_gen_s0e4: te={te} is not a whole multiple of "
            f"resolution_size={rsize} — a tile would split a log-sum group")
    return te


def _load_lib():
    from cffi import FFI
    from tabpwa.cuda.build import ensure
    global _ffi
    if _ffi is None:
        _ffi = FFI()
        _ffi.cdef("""
void* cuda_create_context_flat_pwa_gen_s0e4(
    const double*,int, const int*,int,
    const double*,const double*,int, const int*,int,
    const double*,const double*,int, const int*,int,
    const int*,int, const int*,int,
    const int*,int, const int*,int, int,int,
    const int*,int, const int*,int, const int*,int,
    const int*,int, const int*,int,
    const double*,int,
    const int*,int,
    const double*,int, const double*,int,
    const double*,const double*,int,
    double,double,int, double,double,int,
    int,int,int,int,int, int,int,int,int,int, int,int,int,int);
void* cuda_load_data_flat_pwa_gen_s0e4(void*,const double*,const double*,
    const double*,const double*,const double*,int);
void cuda_free_data_flat_pwa_gen_s0e4(void*);
void cuda_free_context_flat_pwa_gen_s0e4(void*);
void cuda_compute_flat_pwa_gen_s0e4(void*,void*,const double*,const double*,
    const double*,double,int, double*,double*,double*,double*,double*,double*);
int cuda_get_device_count_flat_pwa_gen_s0e4(void);
long long cuda_tile_bytes_flat_pwa_gen_s0e4(int,int,int,int);
int cuda_smem_bytes_flat_pwa_gen_s0e4(void*);
int cuda_cap_flat_pwa_gen_s0e4(void*);
void cuda_stats_flat_pwa_gen_s0e4(int, int*);
int cuda_ngroups_flat_pwa_gen_s0e4(void*);
int cuda_te_flat_pwa_gen_s0e4(void*);
int cuda_split_nnz_flat_pwa_gen_s0e4(void*);
int cuda_nseg_flat_pwa_gen_s0e4(void*);
void cuda_segments_flat_pwa_gen_s0e4(void*, int*);
void cuda_get_device_name_flat_pwa_gen_s0e4(char*,int);
""")
    lib_path = ensure("kernels_flat_pwa_gen_s0e4.cu",
                      "libcuda_kernels_flat_pwa_gen_s0e4.so")
    if not lib_path:
        raise RuntimeError("cuda_flat_pwa_gen_s0e4: library not found")
    return _ffi.dlopen(lib_path)


class FlatPwaGenS0E4DataHandle:
    """One loaded sample — a per-load C handle, never the context.

    The fitter holds the phase-space handle and the data handle at the same
    time, so each ``load_data`` owns its arrays.

    The handle holds per-EVENT data only: the two load-time caches
    (``us`` = the rotated basis interpolation, ``amp`` = ``fa·fl·num``),
    ``weight``/``bkg``, and ``P_dev`` — the caller-visible per-event OUTPUT,
    which is not scratch (it survives the whole call and is copied out once).
    """

    __slots__ = ("ptr", "ne", "_lib", "_keep", "__weakref__")

    def __init__(self, ptr, lib, ne):
        self.ptr = ptr
        self._lib = lib
        self.ne = int(ne)
        self._keep = []

    def free(self):
        """Idempotent: the second call (and a NULL handle) is a no-op."""
        if self.ptr is not None:
            self._lib.cuda_free_data_flat_pwa_gen_s0e4(self.ptr)
            self.ptr = None
            self._keep.clear()

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class CUDAKernelFlatPwaGenS0E4:
    """The s0e fused tile over the flat ``pwa_gen`` config (n_res == 1)."""

    def __init__(self, config, batch_size=50000, resolution_size=1,
                 tile_events=None, te=0, split_nnz=0):
        global _lib
        if "fl_index" in config or "num_index" in config:
            raise ValueError(
                "cuda_flat_pwa_gen_s0e4: pass the FULL (non-deduped) config — "
                "this backend reads every row where the builder emits it, and "
                "a config that already carries fl_index/num_index has a "
                "different row layout")
        resolution_size = int(resolution_size)
        if resolution_size < 1:
            raise ValueError(
                "cuda_flat_pwa_gen_s0e4: resolution_size must be >= 1, got "
                f"{resolution_size}")
        # `te` (events per tile) is the s0e4 sweep knob: 0 = the kernel's own
        # choice, otherwise an explicit tile.  A tile must never split a
        # log-sum group (the kernel refuses such a hint too — this is the
        # early, clear error).  `tile_events` is the s0e-compatible spelling
        # of the same hint, kept so the copied fixture works unchanged.
        te_hint = _te_hint(te, resolution_size, tile_events)
        # `split_nnz` (phase-A rows longer than this are split into two halves)
        # is a CREATE-TIME knob: 0 = the kernel's compiled default.  The
        # segmentation is built in `create`, so the threshold costs a create,
        # never a rebuild.
        split_nnz = int(split_nnz)
        if split_nnz < 0:
            raise ValueError(
                "cuda_flat_pwa_gen_s0e4: split_nnz must be >= 0 (0 = the "
                f"kernel's built-in default), got {split_nnz}")
        if _lib is None:
            _lib = _load_lib()
        self._lib = _lib
        self.config = config
        if _lib.cuda_get_device_count_flat_pwa_gen_s0e4() <= 0:
            raise RuntimeError("cuda_flat_pwa_gen_s0e4: no CUDA devices")
        buf = _ffi.new("char[256]")
        _lib.cuda_get_device_name_flat_pwa_gen_s0e4(buf, 256)
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
                "cuda_flat_pwa_gen_s0e4 needs exactly ONE BW per chain "
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
        self._last_dnorm = None

        bas = np.asarray(c["basis_table"], np.complex128)
        gam_full = np.asarray(c["gamma_matrix"], np.float64)

        # ── the two structural dedups ─────────────────────────────────────
        (self.uslot_row, self.uslot_col,
         self.slot_of_basis) = uslot_map(bas, c["basis_m_index"])
        self.n_uslot = int(self.uslot_row.size)

        self.slot_of_wave, self.rep_of_slot = amp_slot_map(
            c["matrix_angle"], c["fl_table"], c["fl_q_index"],
            c["num_table"], c["num_m_index"], self.n_wave, self.n_proj)
        self.n_uniq = int(self.rep_of_slot.size)

        # ── the CSR of the gamma's TRUE non-zeros ─────────────────────────
        self.gs_lo, self.gs_hi, self.nz_k, self.nz_m = index_runs(
            gam_full, self.n_basis)
        self.n_nz = int(self.nz_m.size)
        #: the uslot of each run entry, taken through the (k, m) pair
        self.nz_u = np.ascontiguousarray(
            self.slot_of_basis[self.nz_m], np.int32)

        gam = np.ascontiguousarray(
            gam_full.reshape(self.n_wave * self.n_res, self.n_basis))

        keep_alive = []

        def D(a):
            arr = np.ascontiguousarray(a, np.float64)
            keep_alive.append(arr)
            return _ffi.cast("double*", _ffi.from_buffer(arr))

        def I(a):
            arr = np.ascontiguousarray(a, np.int32)
            keep_alive.append(arr)
            return _ffi.cast("int*", _ffi.from_buffer(arr))

        fl_tab = np.asarray(c["fl_table"], np.float64).reshape(
            self.n_wave * self.n_decay, -1)
        num = np.asarray(c["num_table"], np.complex128).reshape(
            self.n_wave * self.n_res, -1)
        fl_q = np.asarray(c["fl_q_index"], np.int64)
        num_m = np.asarray(c["num_m_index"], np.int64)

        self._ctx = _lib.cuda_create_context_flat_pwa_gen_s0e4(
            D(fl_tab), fl_tab.size, I(fl_q), fl_q.size,
            D(num.real), D(num.imag), num.size, I(num_m), num_m.size,
            D(bas.real), D(bas.imag), bas.size, I(c["basis_m_index"]),
            self.n_basis,
            I(self.uslot_row), int(self.uslot_row.size),
            I(self.uslot_col), int(self.uslot_col.size),
            I(self.slot_of_wave), int(self.slot_of_wave.size),
            I(self.rep_of_slot), int(self.rep_of_slot.size),
            self.n_uniq, self.n_uslot,
            I(self.nz_k), int(self.nz_k.size),
            I(self.nz_m), int(self.nz_m.size),
            I(self.nz_u), int(self.nz_u.size),
            I(self.gs_lo), int(self.gs_lo.size),
            I(self.gs_hi), int(self.gs_hi.size),
            D(gam), int(gam.size),
            I(np.asarray(c["angle_index"])), self.n_ab,
            D(np.asarray(c["angle_k"], np.float64)),
            int(np.asarray(c["angle_k"]).size),
            D(np.asarray(c["angle_b"], np.float64)),
            int(np.asarray(c["angle_b"]).size),
            D(np.asarray(c["matrix_angle"], np.complex128).real),
            D(np.asarray(c["matrix_angle"], np.complex128).imag),
            int(np.asarray(c["matrix_angle"]).size),
            float(c["m_min"]), float(c["m_delta"]), int(bas.shape[1]),
            float(c["q_min"]), float(c["q_delta"]), int(fl_tab.shape[1]),
            self.n_wave, self.n_proj, self.n_res, self.n_decay,
            self.n_basis, self.n_ab, self.n_comp,
            self.n_mass, self.n_momentum, self.n_angle_total,
            self.batch_size, self.resolution_size,
            te_hint, split_nnz)
        if not self._ctx:
            raise RuntimeError(
                "cuda_flat_pwa_gen_s0e4: context creation failed")
        self._keep = keep_alive
        # the ONE definition of each of these lives in C; asking the context
        # is what stops a Python copy from drifting against it
        self.smem_bytes = int(
            _lib.cuda_smem_bytes_flat_pwa_gen_s0e4(self._ctx))
        self.cap = int(_lib.cuda_cap_flat_pwa_gen_s0e4(self._ctx))
        self.ngroups = int(_lib.cuda_ngroups_flat_pwa_gen_s0e4(self._ctx))
        self.te = int(_lib.cuda_te_flat_pwa_gen_s0e4(self._ctx))
        #: the EFFECTIVE segmentation threshold (0 at the call = the compiled
        #: default, so the context is what says which one was used)
        self.split_nnz = int(
            _lib.cuda_split_nnz_flat_pwa_gen_s0e4(self._ctx))
        #: phase-A segments: ``Σ_k (nnz(k) > split_nnz ? 2 : 1)``
        self.n_seg = int(_lib.cuda_nseg_flat_pwa_gen_s0e4(self._ctx))

    # ── reporting, for the fixture ──────────────────────────────────────
    @property
    def run_widths(self):
        """Gamma non-zeros selected by each wave (``<= n_basis``)."""
        return (self.gs_hi - self.gs_lo).astype(np.int64)

    def segments(self):
        """The phase-A segmentation as ``(k, lo, hi, part)`` arrays.

        Read back from the CONTEXT — the segmentation has one definition, in
        ``create`` — so a test can assert the partition rather than re-derive
        it in Python.
        """
        out = np.zeros(4 * self.n_seg, np.int32)
        self._lib.cuda_segments_flat_pwa_gen_s0e4(
            self._ctx, _ffi.cast("int*", _ffi.from_buffer(out)))
        return (out[0::4].copy(), out[1::4].copy(),
                out[2::4].copy(), out[3::4].copy())

    def gamma_iters(self):
        """``(ref, run)`` basis-row iterations per event of phase 1."""
        return (self.n_wave * self.n_basis, int(self.n_nz))

    def kernel_stats(self, smem=None):
        """The compiled resource usage of the tile kernel (no launch).

        Returns ``dict(regs=..., local_bytes=..., blocks_per_sm=...,
        warps_per_sm=..., max_threads=..., static_smem=...)``.  ``local_bytes
        > 0`` means the kernel SPILLS — the failure mode of an over-tight
        ``__launch_bounds__`` / ``-D..._MIN_BLOCKS`` — and ``blocks_per_sm`` is
        computed for THIS shape's dynamic shared memory, so it is the
        occupancy the fit really gets.
        """
        import numpy as np
        if smem is None:
            smem = int(self.tile_bytes())
        out = np.zeros(6, np.int32)
        self._lib.cuda_stats_flat_pwa_gen_s0e4(
            int(smem), _ffi.cast("int*", _ffi.from_buffer(out)))
        return dict(regs=int(out[0]), local_bytes=int(out[1]),
                    blocks_per_sm=int(out[2]), warps_per_sm=int(out[3]),
                    max_threads=int(out[4]), static_smem=int(out[5]))

    def tile_bytes(self, te=None, n_wave=None, n_proj=None, n_basis=None):
        """The tile size ``gps04_layout`` gives — the kernel's own arithmetic."""
        return int(self._lib.cuda_tile_bytes_flat_pwa_gen_s0e4(
            self.te if te is None else int(te),
            self.n_wave if n_wave is None else int(n_wave),
            self.n_proj if n_proj is None else int(n_proj),
            self.n_basis if n_basis is None else int(n_basis)))

    # ── data ────────────────────────────────────────────────────────────
    def load_data(self, data):
        """Build a FRESH handle; nothing is freed or overwritten here."""
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
        ptr = self._lib.cuda_load_data_flat_pwa_gen_s0e4(
            self._ctx,
            _ffi.cast("double*", _ffi.from_buffer(mass)),
            _ffi.cast("double*", _ffi.from_buffer(mom)),
            _ffi.cast("double*", _ffi.from_buffer(ang)),
            _ffi.cast("double*", _ffi.from_buffer(w)),
            _ffi.cast("double*", _ffi.from_buffer(b)), ne)
        if not ptr:
            raise RuntimeError(
                "cuda_flat_pwa_gen_s0e4: load_data failed (device allocation)")
        h = FlatPwaGenS0E4DataHandle(ptr, self._lib, ne)
        h._keep = keep
        return h

    def free(self):
        """Context only.  Handles are owned by whoever called ``load_data``."""
        if getattr(self, "_ctx", None):
            self._lib.cuda_free_context_flat_pwa_gen_s0e4(self._ctx)
            self._ctx = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    # ── compute ─────────────────────────────────────────────────────────
    def compute(self, params, data_handle, norm=None, return_p=True):
        ne = data_handle.ne
        hp = data_handle.ptr
        if hp is None:
            raise RuntimeError(
                "cuda_flat_pwa_gen_s0e4: this data handle was already freed")
        ck = np.asarray(params["ck"], np.complex128).ravel()
        coeff = np.asarray(params["coeff"], np.float64).ravel()
        if ck.size != self.n_wave:
            raise ValueError(
                f"cuda_flat_pwa_gen_s0e4: ck has {ck.size} entries, expected "
                f"n_wave = {self.n_wave}")
        if coeff.size != self.n_basis:
            raise ValueError(
                f"cuda_flat_pwa_gen_s0e4: coeff has {coeff.size} entries, "
                f"expected n_basis = {self.n_basis}")
        coeff = np.ascontiguousarray(coeff)
        ckr = np.ascontiguousarray(ck.real)
        cki = np.ascontiguousarray(ck.imag)

        P = np.zeros(ne) if return_p else None
        P_c = (_ffi.cast("double*", _ffi.from_buffer(P)) if return_p
               else _ffi.NULL)
        gckr = np.zeros(self.n_wave)
        gcki = np.zeros(self.n_wave)
        gco = np.zeros(self.n_basis)
        oQ = _ffi.new("double*")
        oDn = _ffi.new("double*")
        self._lib.cuda_compute_flat_pwa_gen_s0e4(
            self._ctx, hp,
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

        grads = {"ck": gckr + 1j * gcki, "coeff": gco}
        if norm is not None:
            self._last_dnorm = float(oDn[0])
            grads["norm"] = self._last_dnorm
        else:
            self._last_dnorm = None
        return Q, grads, (P if return_p else None)


@register_backend("cuda_flat_pwa_gen_s0e4", model="pwa_gen")
class CUDAFlatPwaGenS0E4Backend(ComputeBackend):
    """Backend wrapper for :class:`CUDAKernelFlatPwaGenS0E4`."""

    def __init__(self, kernel_config, batch_size=50000, resolution_size=1,
                 tile_events=None, split_nnz=0, **kw):
        # `**kw` is the pass-through contract of the backend factory: the
        # caller's spec kwargs arrive here unchanged (``te`` included).
        # `split_nnz` is the phase-A segmentation threshold (0 = the kernel's
        # compiled default) and travels with the create call, so it is a
        # runtime knob.
        self.kernel = CUDAKernelFlatPwaGenS0E4(
            kernel_config, batch_size=batch_size,
            resolution_size=resolution_size, tile_events=tile_events,
            split_nnz=split_nnz, **kw)
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
