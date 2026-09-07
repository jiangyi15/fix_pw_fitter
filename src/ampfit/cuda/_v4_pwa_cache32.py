"""
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
    int n_uniq,int n_proj);
void cuda_free_context_v4(void*);
void* cuda_load_data_v4(void*,const double*,int,const double*,int,
    const double*,int,const double*,const double*,int);
void cuda_free_data_v4(void*);
void cuda_compute_v4(void*,void*,
    const double*,const double*,const double*,const double*,
    double,int,
    double*,double*,double*,double*,double*,double*);
void cuda_compute_v4_cache(void*,void*,
    const double*,const double*,const double*,const double*,
    double,int,
    double*,double*,double*,double*,double*,double*,double*);
int cuda_fill_common_v4_cache(void*,void*,
    const double*,int,const double*,int,
    const double*,int,const double*,int,const double*,int);
void cuda_gram_from_cache_v4(void*,void*,double*,double*);
int cuda_get_device_count();
int cuda_get_device_name(char*,int);
""")


def _load_lib():
    from ampfit.cuda.build import ensure
    ensure("kernels_v4_pwa_cache32.cu", "libcuda_kernels_v4_pwa_cache32.so")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(script_dir, "libcuda_kernels_v4_pwa_cache32.so")
    if not os.path.exists(lib_path):
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



class CUDAKernelV4PWACache32:
    """FP32-STORAGE fixed-m0/g0 full-amplitude cache.

    Identical to CUDAKernelV4PWACache but the cached amplitude matrix is
    stored as float2 (half the steady-state memory traffic of the fp64
    cache, ~2x per-call speed at large wave counts).  All arithmetic stays
    fp64: only the cached F amplitudes are fp32 (storage precision ~1e-7
    per amplitude).  Same API / same fixed-m0/g0 contract.
    """
    """Projection-sum PWA kernel, fixed m0/g0 FULL-amplitude cache.

    Standard kernel interface (``load_data(data)`` / ``compute(params, h)``):
    the device handle keeps only weight/bkg + the cached spatial amplitude
    ``common[e, p·N+k] = Amp/bw`` at the fixed m0/g0.  The one-time cache is
    built lazily inside the FIRST ``compute()`` from the host arrays retained
    at load (mass/momentum/angle uploaded per batch, transient).  ``params``
    m0/g0 seen then must stay constant — changing them raises ValueError
    (floating fits must use the plain ``cuda_v4_pwa`` kernel instead).
    """

    def __init__(self, config, batch_size=50000, lib_path=None):
        self._lib = _load_lib()
        n_dev = self._lib.cuda_get_device_count()
        if n_dev > 0:
            name_buf = _ffi.new("char[256]")
            self._lib.cuda_get_device_name(name_buf, 256)
            print(f"v4 PWA cache32 GPU: {_ffi.string(name_buf).decode()}")
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
        self.n_decay = c["fl_order"].size // self.n_wave
        self.n_unique_bw = len(c["m0_index"])
        self.n_gamma_rows = c["matrix_gamma"].shape[0]
        self.n_mass = int(np.max(c["mass_index"])) + 1
        self.n_momentum = int(np.max(c["fl_q_index"])) + 1
        self.n_angle_k = c["angle_k"].shape[0]
        self.n_angle_total = (int(np.max(c["angle_index"])) + 1
                              if len(c["angle_index"]) > 0 else 0)
        self.n_angle_comp = c["angle_k"].shape[-1]
        self.n_m0_params = int(np.max(c["m0_index"])) + 1
        self.n_g0_params = int(np.max(c["g0_index"])) + 1

        from ampfit.amp_cache import build_amp_cache_layout
        self.n_uniq, self.slot_of_wave, self.rep_of_slot = \
            build_amp_cache_layout(c)
        self._ka = []
        self._dnorm = None            # d(NLL)/d(norm) from the last norm compute
        self.dnorm_on_gpu = True

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
            _ib(c["m0_index"]), len(c["m0_index"]),
            _ib(c["g0_index"]), len(c["g0_index"]),
            _ib(c["g0_mass_index"]), len(c["g0_mass_index"]),
            _ib(c["mass_index"]), len(c["mass_index"]),
            _ib(c["fl_type"]), len(c["fl_type"]),
            _ib(c["fl_q_index"]), len(c["fl_q_index"]),
            _ib(c["bw_order"]), len(c["bw_order"]),
            _ib(c["fl_order"]), len(c["fl_order"]),
            _ib(c["angle_index"]), len(c["angle_index"]),
            _db(c["angle_k"].flatten()), c["angle_k"].size,
            _db(c["angle_b"].flatten()), c["angle_b"].size,
            _db(np.real(ma).flatten()), ma.size,
            _db(np.imag(ma).flatten()), ma.size,
            _db(np.real(gt).flatten()), gt.size,
            _db(np.imag(gt).flatten()), gt.size,
            float(c["gamma_min"]), float(c["gamma_delta"]), gt.shape[-1],
            _db(mg.flatten()), mg.size,
            _ib(gamma_col_idx), len(gamma_col_idx),
            _db(c["fl_table"].flatten()), c["fl_table"].size,
            float(c["fl_min"]), float(c["fl_delta"]), c["fl_table"].shape[-1],
            self.n_wave, self.n_res, self.n_decay,
            self.n_unique_bw, self.n_gamma_rows,
            self.n_mass, self.n_momentum,
            self.n_angle_k, self.n_angle_total, self.n_angle_comp,
            self.n_m0_params, self.n_g0_params, batch_size,
            _ib(self.slot_of_wave), len(self.slot_of_wave),
            _ib(self.rep_of_slot), len(self.rep_of_slot),
            self.n_uniq, self.n_proj)

    def _padded_mg(self, params):
        nm_ = self.n_m0_params
        gg_ = self.n_g0_params
        m0 = np.zeros(nm_, np.float64)
        m0[:len(params["m0"])] = np.asarray(params["m0"])
        g0 = np.zeros(gg_, np.float64)
        g0[:len(params["g0"])] = np.asarray(params["g0"])
        return m0, g0

    def load_data(self, data):
        """Standard load — keep weight/bkg on device and the kernel-array
        momenta on the CPU side for the lazy one-time cache fill."""
        ka = []

        def _db(a):
            arr = np.ascontiguousarray(a, np.float64)
            buf = _ffi.from_buffer(arr)
            ka.append(buf)
            return _ffi.cast("double*", buf)

        ne = data["mass"].shape[0]
        mass = np.ascontiguousarray(data["mass"].reshape(ne, -1))
        mom = np.ascontiguousarray(data["q"].reshape(ne, -1))
        ang = np.ascontiguousarray(data["angle"].reshape(ne, -1))
        nang = self.n_angle_total
        bkg_key = "bkg_raw" if "bkg_raw" in data else "bkg"

        dh = DataHandle(self._lib.cuda_load_data_v4(
            self._ctx, _db(mass), mass.shape[1],
            _db(mom), mom.shape[1], _db(ang), nang,
            _db(data["weight"]), _db(data[bkg_key]), ne), self._lib, ne)
        dh._keep = ka
        dh._arrays = (mass, mom, ang)
        dh._filled = False
        dh._fixed_mg = None
        return dh

    def _fill_once(self, data_handle, m0, g0):
        """Build the fixed-m0/g0 full-amplitude cache once (Python flag on
        the data handle).  C keeps only weight/bkg + the cached common."""
        if data_handle._filled:
            fm, fg = data_handle._fixed_mg
            if not (np.array_equal(fm, m0) and np.array_equal(fg, g0)):
                raise ValueError(
                    "cuda_v4_pwa_cache: m0/g0 changed after the cache was "
                    "built — fixed m0/g0 only; use cuda_v4_pwa when they "
                    "float")
            return
        mass, mom, ang = data_handle._arrays
        ka2 = []

        def _db2(a):
            arr = np.ascontiguousarray(a, np.float64)
            buf = _ffi.from_buffer(arr)
            ka2.append(buf)
            return _ffi.cast("double*", buf)

        ok = self._lib.cuda_fill_common_v4_cache(
            self._ctx, data_handle.ptr,
            _db2(m0), len(m0), _db2(g0), len(g0),
            _db2(mass), mass.shape[1],
            _db2(mom), mom.shape[1],
            _db2(ang), ang.shape[1])
        data_handle._keep += ka2
        if not ok:
            raise RuntimeError("cuda_fill_common_v4_cache failed")
        data_handle._filled = True
        data_handle._fixed_mg = (m0.copy(), g0.copy())

    def compute_gram(self, phsp_handle, m0, g0):
        """Wave Gram D (N, N) from the CACHED full amplitude at fixed m0/g0.

        The cache is built lazily on first use (same Python flag), then D is
        reduced directly from the device common_cache — no BW recompute.
        """
        n = self.n_wave_base
        m0p, g0p = self._padded_mg({"m0": m0, "g0": g0})
        self._fill_once(phsp_handle, m0p, g0p)
        oDr = np.zeros(n * n, np.float64)
        oDi = np.zeros(n * n, np.float64)
        ka = []

        def _db(a):
            arr = np.ascontiguousarray(a, np.float64)
            buf = _ffi.from_buffer(arr)
            ka.append(buf)
            return _ffi.cast("double*", buf)

        self._lib.cuda_gram_from_cache_v4(
            self._ctx, phsp_handle.ptr, _db(oDr), _db(oDi))
        D = (oDr + 1j * oDi).reshape(n, n)
        return (D + D.conj().T) / 2.0

    def compute(self, params, data_handle, norm=None, return_p=True):
        """Forward + gradients; builds the fixed-m0/g0 amplitude cache lazily
        on the first call (from the m0/g0 of *params*), then only contracts
        ck.  Gradients for m0/g0 are zero.

        Args:
            params: dict with 'ck' (complex length N), 'm0', 'g0'.  The
                m0/g0 are FIXED for the handle lifetime; a later call with
                different values raises ValueError (use cuda_v4_pwa).
            norm: optional float normalization; None → unnormalized.
            return_p: if True return per-event P (always returned here).

        Returns:
            (Q, grads, P) with grads keys ck/m0/g0 (m0/g0 all zero).
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
                f"v4 PWA cache: ck length {len(ck)} != n_wave/n_proj = "
                f"{nbase} ({self.n_wave}/{self.n_proj})")

        ck_r = np.real(ck).astype(np.float64)
        ck_i = np.imag(ck).astype(np.float64)
        m0, g0 = self._padded_mg(params)

        self._fill_once(data_handle, m0, g0)

        oQ = _ffi.new("double*")
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

        odn = _ffi.new("double*")
        self._lib.cuda_compute_v4_cache(
            self._ctx, data_handle.ptr,
            _db(ck_r), _db(ck_i), _db(m0), _db(g0),
            norm_val, use_norm,
            oQ, (_db(oP) if oP is not None else _ffi.NULL),
            _db(ogck_r), _db(ogck_i),
            _db(ogm0), _db(ogg0), odn)
        data_handle._keep += ka
        # d(NLL)/d(norm) accumulated on device; None when unnormalized
        self._dnorm = odn[0] if use_norm else None

        m0_idx = self.config["m0_index"]
        g0_idx = self.config["g0_index"]
        grad_m0 = np.zeros(self.n_m0_params, np.float64)
        grad_g0 = np.zeros(self.n_g0_params, np.float64)
        for i, idx in enumerate(m0_idx):
            grad_m0[idx] += ogm0[i]
        for i, idx in enumerate(g0_idx):
            grad_g0[idx] += ogg0[i]

        grads = {"ck": ogck_r + 1j * ogck_i,
                 "m0": grad_m0, "g0": grad_g0}
        return oQ[0], grads, oP

    def free(self):
        if self._ctx is not None:
            self._lib.cuda_free_context_v4(self._ctx)
            self._ctx = None
        self._ka.clear()

    def __del__(self):
        self.free()
