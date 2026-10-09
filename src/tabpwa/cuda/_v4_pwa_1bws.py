"""
CUDA kernel v4 PWA, ONE BW PER CHAIN + ck_order (mass-sorted waves).

Sibling of ``_v4_pwa_1bw`` (same one-BW-per-chain constraint, SAME kernel
binary — the sorted layout is purely a wrapper-level data transform, so
the two backends are directly comparable).  On top of the 1bw per-wave
BW expansion, the base waves are reordered by their DATA MASS column
(stable — waves sharing a mass keep their wave order):

    ck_order = [waves of mass 0, waves of mass 1, ...]

so every per-wave BW array is a non-decreasing run in wave order
(contiguous device reads/writes) and the host-side m0/g0 gradient
reductions become vectorized bincounts over sorted runs.  The transform
is outside the model layer: ck is permuted on the way in, ck gradients
and the Gram matrix are un-permuted on the way out.

Original cuda_v4_pwa documentation:

==========================================================================
CUDA kernel v4 PWA — projection-sum PWA (derived from cuda_v3_ampcache).

No time evolution / no D0-D0bar mixing / no scalar (Gamma, DeltaGamma, ...)
parameters.  The event probability is an incoherent sum over *projections*:

    P(e) = Σ_p  | A_p(e) |² ,     A_p(e) = Σ_k  ck_k · a_{p,k}(e)

* p = 0..P-1  — projection index (helicity / spin projection etc.), count is
  a fixed extra dimension of the wave space: the per-wave spatial amplitude
  entries are stored **p-major**, entry w = p·N + k with N = n_wave / P.
* k = 0..N-1  — base partial wave; **all projections share the same ck_k**.
  The projection only changes the *angular* part of a_{p,k} (the matrix_angle
  column / fl_order row of the entry), never the BW propagator or ck.

The angular amplitude cache is inherited from ``cuda_v3_ampcache``: the
per-event per-entry angular factor ``Amp = fa·fl`` is pure kinematics, filled
once at load_data (fp64 compute, float2 storage — constant over the fit) and
reused every compute() call, while ``1/bw_p`` is recomputed per iteration so
the m0/g0 gradients keep flowing.

The kernel config must provide ``n_proj`` and the p-major-duplicated arrays
(matrix_angle with P·N columns, bw_order/fl_order/mass_index/... repeated P
times).  ``params["ck"]`` has length N = n_wave / n_proj.
"""

import os
import numpy as np
from cffi import FFI

_ffi = FFI()
_ffi.cdef("""
void* cuda_create_context_v4(
    const int* m0_i,int n1, const int* g0_i,int n2,
    const int* g0_m,int n3, const int* mass_i,int n4,
    const int* fl_t,int n5, const int* fl_q,int n6,
    const int* bw_o,int n7, const int* fl_o,int n8,
    const int* ang_i,int n9,
    const double* ak,int n10, const double* ab,int n11,
    const double* mar,int n12, const double* mai,int n13,
    const double* gtr,int n14, const double* gti,int n15,
    double gmin,double gdel,int gbins,
    const double* mg,int n16,
    const int* gci,int ngci,
    const double* ft,int n17, double flmin,double fldel,int fbins,
    int nw,int nr,int nd,int nub,int ngr,
    int nm,int nmom,int nak_,int nat,int nac,
    int n_m0p,int n_g0p,
    int batch_size,
    const int* slot_of_wave,int n_slot,
    const int* rep_of_slot,int n_rep,
    int n_uniq,int n_proj,
    const int* row_start,int n_rs);
void cuda_free_context_v4(void*);
void* cuda_load_data_v4(void*,const double*,int,const double*,int,
    const double*,int,const double*,const double*,int);
void cuda_free_data_v4(void*);
void cuda_compute_v4(void*,void*,
    const double*,const double*,const double*,const double*,
    double,int,
    double*,double*,double*,double*,double*,double*,double*);
void cuda_gram_matrix_v4(void*,void*,
    const double*,const double*,double*,double*);
int cuda_get_device_count();
int cuda_get_device_name(char*,int);
""")


def _load_lib():
    from tabpwa.cuda.build import ensure
    lib_path = ensure("kernels_v4_pwa_1bw.cu", "libcuda_kernels_v4_pwa_1bw.so")
    if not lib_path:
        raise RuntimeError(
            f"v4 PWA CUDA library not found at {lib_path}.")
    return _ffi.dlopen(lib_path)


class DataHandle:
    def __init__(self, ptr, lib, ne):
        self.ptr = ptr
        self._lib = lib
        self.ne = ne
        self._keep = []

    def free(self):
        if self.ptr is not None:
            self._lib.cuda_free_data_v4(self.ptr)
            self.ptr = None
            self._keep.clear()

    def __del__(self):
        self.free()


class CUDAKernelV4PWA1BWS:
    """Projection-sum PWA kernel on the ampcache infrastructure.

    No time/mixing: ``params`` needs only ``ck`` (length N = n_wave/n_proj),
    ``m0`` and ``g0``.  Gradients returned for ``ck``/``m0``/``g0`` only.
    """

    def __init__(self, config, batch_size=50000, lib_path=None):
        self._lib = _load_lib()
        n_dev = self._lib.cuda_get_device_count()
        if n_dev > 0:
            name_buf = _ffi.new("char[256]")
            self._lib.cuda_get_device_name(name_buf, 256)
            print(f"v4 PWA GPU: {_ffi.string(name_buf).decode()}")
        else:
            raise RuntimeError("No CUDA devices found")

        c = config
        self.config = c
        self.n_wave = c["matrix_angle"].shape[1]     # P·N entries per event
        self.n_proj = int(c.get("n_proj", 1))
        if self.n_wave % self.n_proj != 0:
            raise ValueError(
                f"v4 PWA: n_wave={self.n_wave} not divisible by "
                f"n_proj={self.n_proj}")
        self.n_wave_base = self.n_wave // self.n_proj  # N = ck length
        self.n_res = c["bw_order"].size // self.n_wave
        if self.n_res != 1:
            raise ValueError(
                "cuda_v4_pwa_1bw needs exactly ONE BW per chain "
                f"(top -> R + X), got n_res = {self.n_res} — use "
                "cuda_v4_pwa for multi-BW chains")

        # ── ck_order: sort BASE waves by the DATA MASS column ───────
        # stable argsort -> per-wave mass arrays become non-decreasing
        # runs ([0,0,…,1,1,…]); waves sharing a mass keep their wave
        # order; p-major blocks all get the SAME base permutation.
        # Pure outside transform: ck in, grads/Gram out.
        n_base = self.n_wave_base
        bw_flat = np.asarray(c["bw_order"]).reshape(-1)
        base_bw = bw_flat[:n_base]
        self.ck_order = np.argsort(
            np.asarray(c["mass_index"])[base_bw],
            kind="stable").astype(np.int64)
        self._perm = np.concatenate(
            [p * n_base + self.ck_order for p in range(self.n_proj)])
        # per-entry arrays only — angle_index/angle_k/angle_b are indexed
        # by unique ANGLE BASIS (amp-cache representatives), not by wave
        c = dict(c)                      # permuted view for everything below
        c["matrix_angle"] = np.asarray(c["matrix_angle"])[:, self._perm]
        # fl_order is flat per (entry, decay vertex): permute the entry rows
        c["fl_order"] = (np.asarray(c["fl_order"])
                         .reshape(self.n_wave, -1)[self._perm]).ravel()
        c["bw_order"] = bw_flat[self._perm]

        # ── per-wave BW expansion: NO bw_order indirection ──────────
        # Every wave becomes its own BW slot: the kernel indexes the BW
        # arrays by the wave index directly (contiguous, no gather, no
        # atomics).  matrix_gamma stays the config-side row->slot source;
        # its gamma rows are duplicated per wave here (col = wave).
        bw = np.asarray(c["bw_order"]).reshape(-1)
        if bw.size != self.n_wave:
            raise ValueError("bw_order must be flat (n_wave,) at n_res=1")
        m0_index_w = np.asarray(c["m0_index"])[bw]
        mass_index_w = np.asarray(c["mass_index"])[bw]
        # merge the gamma matrix (gamma -> bw) with bw_order (bw -> wave):
        # gamma rows are duplicated per wave, col = the owning wave
        mg_w = np.asarray(c["matrix_gamma"])[:, bw]     # (n_gamma, n_wave)
        gamma_col_idx_w, rows_w = np.nonzero(mg_w.T)    # wave-major
        gamma_col_idx_w = gamma_col_idx_w.astype(np.int32)
        rows_w = rows_w.astype(np.int64)
        g0_index_w = np.asarray(c["g0_index"])[rows_w]
        g0_mass_index_w = np.asarray(c["g0_mass_index"])[rows_w]
        self.n_unique_bw = self.n_wave          # wave index == BW slot
        self.n_gamma_rows = int(rows_w.size)
        # rows are wave-major (nonzero over mg_w.T): per-wave row ranges
        self.row_start = np.searchsorted(
            gamma_col_idx_w, np.arange(self.n_wave),
            side="left").astype(np.int32)
        self._m0_index_w = m0_index_w           # host-side grad reduction
        self._g0_index_w = g0_index_w
        self._mass_index_w = mass_index_w       # sorted runs (ck_order)
        self.n_decay = c["fl_order"].size // self.n_wave
        self.n_mass = int(np.max(c["mass_index"])) + 1
        self.n_momentum = int(np.max(c["fl_q_index"])) + 1
        self.n_angle_k = c["angle_k"].shape[0]
        self.n_angle_total = (int(np.max(c["angle_index"])) + 1
                              if len(c["angle_index"]) > 0 else 0)
        self.n_angle_comp = c["angle_k"].shape[-1]
        self.n_m0_params = int(np.max(c["m0_index"])) + 1
        self.n_g0_params = int(np.max(c["g0_index"])) + 1

        from tabpwa.amp_cache import build_amp_cache_layout
        self.n_uniq, self.slot_of_wave, self.rep_of_slot = \
            build_amp_cache_layout(c)
        self._ka = []
        self._last_dnorm = None

        def _db(a):
            arr = np.ascontiguousarray(a, np.float64)
            buf = _ffi.from_buffer(arr)
            self._ka.append(buf)
            return _ffi.cast("double*", buf)

        def _ib(a):
            arr = np.ascontiguousarray(a, np.int32)
            buf = _ffi.from_buffer(arr)
            self._ka.append(buf)
            return _ffi.cast("int*", buf)

        ma = c["matrix_angle"]
        gt = c["gamma_table"]
        mg = c["matrix_gamma"]
        gamma_col_idx = np.argmax(mg, axis=1).astype(np.int32)

        self._ctx = self._lib.cuda_create_context_v4(
            _ib(m0_index_w), len(m0_index_w),
            _ib(g0_index_w), len(g0_index_w),
            _ib(g0_mass_index_w), len(g0_mass_index_w),
            _ib(mass_index_w), len(mass_index_w),
            _ib(c["fl_type"]), len(c["fl_type"]),
            _ib(c["fl_q_index"]), len(c["fl_q_index"]),
            _ffi.NULL, 0,                                       # no bw_order
            _ib(c["fl_order"]), len(c["fl_order"]),
            _ib(c["angle_index"]), len(c["angle_index"]),
            _db(c["angle_k"].flatten()), c["angle_k"].size,
            _db(c["angle_b"].flatten()), c["angle_b"].size,
            _db(np.real(ma).flatten()), ma.size,
            _db(np.imag(ma).flatten()), ma.size,
            _db(np.real(gt).flatten()), gt.size,
            _db(np.imag(gt).flatten()), gt.size,
            float(c["gamma_min"]), float(c["gamma_delta"]), gt.shape[-1],
            _ffi.NULL, 0,                    # matrix_gamma: wrapper-side only
            _ib(gamma_col_idx_w), len(gamma_col_idx_w),
            _db(c["fl_table"].flatten()), c["fl_table"].size,
            float(c["fl_min"]), float(c["fl_delta"]), c["fl_table"].shape[-1],
            self.n_wave, self.n_res, self.n_decay,
            self.n_unique_bw, self.n_gamma_rows,
            self.n_mass, self.n_momentum,
            self.n_angle_k, self.n_angle_total, self.n_angle_comp,
            self.n_m0_params, self.n_g0_params, batch_size,
            _ib(self.slot_of_wave), len(self.slot_of_wave),
            _ib(self.rep_of_slot), len(self.rep_of_slot),
            self.n_uniq, self.n_proj,
            _ib(self.row_start), len(self.row_start))

    def load_data(self, data):
        ka = []

        def _db(a):
            arr = np.ascontiguousarray(a, np.float64)
            buf = _ffi.from_buffer(arr)
            ka.append(buf)
            return _ffi.cast("double*", buf)

        ne = data["mass"].shape[0]
        mass = data["mass"].reshape(ne, -1)[:, :self.n_mass]
        mom = data["q"].reshape(ne, -1)[:, :self.n_momentum]
        ang = data["angle"].reshape(ne, -1)[:, :self.n_angle_total
                                            * self.n_angle_comp]
        nang = self.n_angle_total
        bkg_key = "bkg_raw" if "bkg_raw" in data else "bkg"

        dh = DataHandle(self._lib.cuda_load_data_v4(
            self._ctx, _db(mass), mass.shape[1],
            _db(mom), mom.shape[1], _db(ang), nang,
            _db(data["weight"]), _db(data[bkg_key]), ne), self._lib, ne)
        dh._keep = ka
        return dh

    def compute(self, params, data_handle, norm=None, return_p=True):
        """Forward + gradients for the projection-sum PWA model.

        Args:
            params: dict with 'ck' (complex length N), 'm0', 'g0'.
                A 'scalar' key (if present, e.g. from a legacy config) is
                ignored — this model has no scalar parameters.
            norm: optional float normalization; None → unnormalized.
            return_p: if True return per-event P (always returned here).

        Returns:
            (Q, grads, P) with grads keys ck/m0/g0 (no 'scalar').
        """
        ck = params["ck"]
        nw = self.n_wave
        nbase = self.n_wave_base           # N — ck / gradient buffer size
        nu_ = self.n_unique_bw             # m0 gradient buffer size
        ng_ = self.n_gamma_rows            # g0 gradient buffer size
        nm_ = self.n_m0_params
        gg_ = self.n_g0_params
        use_norm = 0 if norm is None else 1
        norm_val = norm if norm is not None else 0.0

        if len(ck) != nbase:
            raise ValueError(
                f"v4 PWA: ck length {len(ck)} != n_wave/n_proj = {nbase} "
                f"({self.n_wave}/{self.n_proj})")

        ck = ck[self.ck_order]          # ck_order transform (model -> kernel)
        ck_r = np.real(ck).astype(np.float64)
        ck_i = np.imag(ck).astype(np.float64)
        m0 = np.zeros(nm_, np.float64)
        m0[:len(params["m0"])] = np.asarray(params["m0"])
        g0 = np.zeros(gg_, np.float64)
        g0[:len(params["g0"])] = np.asarray(params["g0"])

        oQ = _ffi.new("double*")
        oDn = _ffi.new("double*")
        oP = (np.zeros(data_handle.ne, np.float64)
              if return_p else None)
        ogck_r = np.zeros(nbase, np.float64)
        ogck_i = np.zeros(nbase, np.float64)
        ogm0 = np.zeros(nu_, np.float64)
        ogg0 = np.zeros(ng_, np.float64)

        ka = []

        def _db(a):
            arr = np.ascontiguousarray(a, np.float64)
            buf = _ffi.from_buffer(arr)
            ka.append(buf)
            return _ffi.cast("double*", buf)

        self._lib.cuda_compute_v4(
            self._ctx, data_handle.ptr,
            _db(ck_r), _db(ck_i), _db(m0), _db(g0),
            norm_val, use_norm,
            oQ, oDn, (_db(oP) if oP is not None else _ffi.NULL), _db(ogck_r), _db(ogck_i),
            _db(ogm0), _db(ogg0))
        self._last_dnorm = float(oDn[0]) if use_norm else None

        # vectorized segmented sums over the sorted per-wave runs
        grad_m0 = np.bincount(self._m0_index_w, weights=ogm0,
                              minlength=self.n_m0_params)
        grad_g0 = np.bincount(self._g0_index_w, weights=ogg0,
                              minlength=self.n_g0_params)
        gck = np.empty(self.n_wave_base, np.complex128)
        gck[self.ck_order] = ogck_r + 1j * ogck_i   # back to model order
        grads = {"ck": gck, "m0": grad_m0, "g0": grad_g0}
        return oQ[0], grads, (oP if return_p else None)

    def compute_gram(self, phsp_handle, m0, g0):
        """Wave Gram matrix D (N, N) of a loaded phsp handle at m0/g0.

        ``D[k,j] = Σ_e w_e Σ_p conj(a_{p,k})·a_{p,j}`` — the pre-integrated
        phase-space norm matrix for the shared-ck PWA model.  Callers stream
        the phsp in batches (load → compute_gram → free) to bound memory.
        """
        n = self.n_wave_base
        nu_ = self.n_unique_bw
        ng_ = self.n_gamma_rows
        m0_arr = np.zeros(self.n_m0_params, np.float64)
        m0_arr[:len(m0)] = np.asarray(m0)
        g0_arr = np.zeros(self.n_g0_params, np.float64)
        g0_arr[:len(g0)] = np.asarray(g0)
        oDr = np.zeros(n * n, np.float64)
        oDi = np.zeros(n * n, np.float64)
        ka = []

        def _db(a):
            arr = np.ascontiguousarray(a, np.float64)
            buf = _ffi.from_buffer(arr)
            ka.append(buf)
            return _ffi.cast("double*", buf)

        self._lib.cuda_gram_matrix_v4(
            self._ctx, phsp_handle.ptr,
            _db(m0_arr), _db(g0_arr), _db(oDr), _db(oDi))
        D = (oDr + 1j * oDi).reshape(n, n)
        D = (D + D.conj().T) / 2.0
        # un-permute: kernel wave i is model wave ck_order[i]
        Dm = np.empty_like(D)
        Dm[np.ix_(self.ck_order, self.ck_order)] = D
        return Dm

    def free(self):
        if self._ctx is not None:
            self._lib.cuda_free_context_v4(self._ctx)
            self._ctx = None
        self._ka.clear()

    def __del__(self):
        self.free()
