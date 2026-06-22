"""
CUDA kernel v3 — float32 version with Catmull-Rom interpolation backend.

Based on v2_f32, but uses Catmull-Rom (cubic Hermite) interpolation for the
gamma table (complex) and FL table (real) instead of linear interpolation.
All other logic is identical to v2_f32.

GPU memory managed in C (cudaMalloc/cudaFree inside kernels_v3_f32.cu).
Python only handles: config → context, data upload, compute dispatch,
gradient reduction (n_unique_bw→n_m0, n_gamma_rows→n_g0).

Three C calls:
    cuda_create_context_v3_f32  — one-time config + scratch allocation
    cuda_load_data_v3_f32        — per-dataset data upload
    cuda_compute_v3_f32          — forward + backward + reductions in one call
"""

import os
import numpy as np
from cffi import FFI

_ffi = FFI()
_ffi.cdef("""
void* cuda_create_context_v3_f32(
    const int* m0_i,int n1, const int* g0_i,int n2,
    const int* g0_m,int n3, const int* mass_i,int n4,
    const int* fl_t,int n5, const int* fl_q,int n6,
    const int* bw_o,int n7, const int* fl_o,int n8,
    const int* ang_i,int n9,
    const float* ak,int n10, const float* ab,int n11,
    const float* mar,int n12, const float* mai,int n13,
    const float* gtr,int n14, const float* gti,int n15,
    float gmin,float gdel,int gbins,
    const float* mg,int n16,
    const float* ft,int n17, float flmin,float fldel,int fbins,
    int nw,int nr,int nd,int nub,int ngr,
    int nm,int nmom,int nak_,int nat,int nac,
    int n_m0p,int n_g0p,
    int batch_size);
void cuda_free_context_v3_f32(void*);
void* cuda_load_data_v3_f32(void*,const float*,int,const float*,int,
    const float*,int,const float*,const float*,const float*,
    const float*,int);
void cuda_free_data_v3_f32(void*);
void cuda_compute_v3_f32(void*,void*,
    const float*,const float*,const float*,const float*,
    float,float,float,float,float,float,float,int,
    double*,double*,float*,float*,float*,float*,float*);
void cuda_gram_matrix_v3_f32(void*,void*,
    const float*,const float*,
    double*,double*,double*,double*,double*,double*);
int cuda_get_device_count();
int cuda_get_device_name(char*,int);
""")


def _load_lib():
    """Load the v3 float32 shared library."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(script_dir, "cuda", "libcuda_kernels_v3_f32.so")
    if not os.path.exists(lib_path):
        raise RuntimeError(
            f"v3 float32 CUDA library not found at {lib_path}. "
            "Build with: nvcc -shared --compiler-options '-fPIC' -arch=sm_86 -o <path> kernels_v3_f32.cu -lcudart"
        )
    return _ffi.dlopen(lib_path)


# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------

class DataHandleF32:
    """Python-side wrapper around a C DataHandle pointer (float32)."""

    def __init__(self, ptr, lib, ne):
        self.ptr = ptr
        self._lib = lib
        self.ne = ne
        self._keep = []

    def free(self):
        if self.ptr is not None:
            self._lib.cuda_free_data_v3_f32(self.ptr)
            self.ptr = None
            self._keep.clear()


class CUDAKernelV3F32:
    """High-level v3 float32 CUDA kernel backend (Catmull-Rom interpolation).

    Usage::

        kernel = CUDAKernelV3F32(config)
        data = kernel.load_data(data_dict)
        Q, grads, P = kernel.compute(params, data)

    Compatible return format with v2 CUDAKernelV2 for drop-in replacement.
    """

    def __init__(self, config, batch_size=50000, lib_path=None):
        # ---- load library ----
        self._lib = _load_lib()

        # ---- device info ----
        n_dev = self._lib.cuda_get_device_count()
        if n_dev > 0:
            name_buf = _ffi.new("char[256]")
            self._lib.cuda_get_device_name(name_buf, 256)
            print(f"✓ v3 f32 GPU: {_ffi.string(name_buf).decode()}")
        else:
            raise RuntimeError("No CUDA devices found")

        # ---- config dimensions ----
        c = config  # shorthand
        self.config = c
        self.n_wave = c["matrix_angle"].shape[1]
        self.n_res = c["bw_order"].size // self.n_wave
        self.n_decay = c["fl_order"].size // self.n_wave
        self.n_unique_bw = len(c["m0_index"])
        self.n_gamma_rows = c["matrix_gamma"].shape[0]
        self.n_mass = int(np.max(c["mass_index"])) + 1
        self.n_momentum = int(np.max(c["fl_q_index"])) + 1
        self.n_angle_k = c["angle_k"].shape[0]
        self.n_angle_total = (
            int(np.max(c["angle_index"])) + 1
            if len(c["angle_index"]) > 0 else 0
        )
        self.n_angle_comp = c["angle_k"].shape[-1]
        self.n_m0_params = int(np.max(c["m0_index"])) + 1
        self.n_g0_params = int(np.max(c["g0_index"])) + 1

        # CFFI buffer keepalive (must stay alive for duration of context)
        self._ka = []

        def _fb(a):
            arr = np.ascontiguousarray(a, np.float32)
            buf = _ffi.from_buffer(arr)
            self._ka.append(buf)
            return _ffi.cast("float*", buf)

        def _ib(a):
            arr = np.ascontiguousarray(a, np.int32)
            buf = _ffi.from_buffer(arr)
            self._ka.append(buf)
            return _ffi.cast("int*", buf)

        ma = c["matrix_angle"]
        gt = c["gamma_table"]

        # ---- create GPU context ----
        self._ctx = self._lib.cuda_create_context_v3_f32(
            _ib(c["m0_index"]), len(c["m0_index"]),
            _ib(c["g0_index"]), len(c["g0_index"]),
            _ib(c["g0_mass_index"]), len(c["g0_mass_index"]),
            _ib(c["mass_index"]), len(c["mass_index"]),
            _ib(c["fl_type"]), len(c["fl_type"]),
            _ib(c["fl_q_index"]), len(c["fl_q_index"]),
            _ib(c["bw_order"]), len(c["bw_order"]),
            _ib(c["fl_order"]), len(c["fl_order"]),
            _ib(c["angle_index"]), len(c["angle_index"]),
            _fb(c["angle_k"].flatten()), c["angle_k"].size,
            _fb(c["angle_b"].flatten()), c["angle_b"].size,
            _fb(np.real(ma).flatten().astype(np.float32)), ma.size,
            _fb(np.imag(ma).flatten().astype(np.float32)), ma.size,
            _fb(np.real(gt).flatten().astype(np.float32)), gt.size,
            _fb(np.imag(gt).flatten().astype(np.float32)), gt.size,
            float(c["gamma_min"]), float(c["gamma_delta"]),
            gt.shape[-1],
            _fb(c["matrix_gamma"].flatten().astype(np.float32)), c["matrix_gamma"].size,
            _fb(c["fl_table"].flatten().astype(np.float32)), c["fl_table"].size,
            float(c["fl_min"]), float(c["fl_delta"]),
            c["fl_table"].shape[-1],
            self.n_wave, self.n_res, self.n_decay,
            self.n_unique_bw, self.n_gamma_rows,
            self.n_mass, self.n_momentum,
            self.n_angle_k, self.n_angle_total, self.n_angle_comp,
            self.n_m0_params, self.n_g0_params,
            batch_size,
        )

    # -- data lifecycle ------------------------------------------------

    def load_data(self, data):
        """Upload a dataset to GPU.

        Args:
            data: dict with keys 'mass', 'q', 'angle', 'frac', 'time',
                  'weight', 'bkg_raw'.  Arrays are reshaped to flat
                  (N, n_features) before upload.
        Returns:
            DataHandleF32 (opaque pointer, pass to compute()).
        """
        # Keep buffers alive during upload
        ka = []
        def _fb(a):
            arr = np.ascontiguousarray(a, np.float32)
            buf = _ffi.from_buffer(arr)
            ka.append(buf)
            return _ffi.cast("float*", buf)

        ne = data["mass"].shape[0]

        # Reshape from (N, 24, 2) -> (N, n_mass) etc.
        mass = data["mass"].reshape(ne, -1)
        mom = data["q"].reshape(ne, -1)
        ang = data["angle"].reshape(ne, -1)

        # Use n_angle_total for per-event angle stride
        nang = self.n_angle_total

        # Accept 'bkg' or 'bkg_raw' (Fitter uses 'bkg')
        bkg_key = "bkg_raw" if "bkg_raw" in data else "bkg"

        dh = DataHandleF32(self._lib.cuda_load_data_v3_f32(
            self._ctx,
            _fb(mass), mass.shape[1],
            _fb(mom), mom.shape[1],
            _fb(ang), nang,
            _fb(data["frac"]),
            _fb(data["time"]),
            _fb(data["weight"]),
            _fb(data[bkg_key]),
            ne,
        ), self._lib, ne)
        dh._keep = ka
        return dh

    # -- compute -------------------------------------------------------

    def compute(self, params, data_handle, norm=None, return_p=True):
        """Compute forward + backward pass.

        Args:
            params: dict with keys 'ck' (complex, shape=(n_wave,)),
                    'm0' (float, shape=(n_unique_bw,)),
                    'g0' (float, shape=(n_gamma_rows,)),
                    'scalar' (list: Gamma, DGamma, DM, Ap, pr, pp).
            data_handle: DataHandleF32 from load_data().
            norm: optional float normalization factor.
        Returns:
            (Q, grads, P) where:
                Q: float (total negative log-likelihood)
                grads: dict with keys 'ck', 'm0', 'g0', 'scalar'
                P: ndarray of per-event P values
        """
        ck = params["ck"]
        nw = self.n_wave
        nu_ = self.n_unique_bw
        ng_ = self.n_gamma_rows

        use_norm = 0 if norm is None else 1
        norm_val = norm if norm is not None else 0.0

        ck_r = np.real(ck).astype(np.float32)
        ck_i = np.imag(ck).astype(np.float32)

        # Pad m0/g0 to the sizes expected by the C kernel
        m0 = np.zeros(nu_, np.float32)
        m0[:len(params["m0"])] = np.asarray(params["m0"])
        g0 = np.zeros(ng_, np.float32)
        g0[:len(params["g0"])] = np.asarray(params["g0"])
        G, DG, DM, Ap_, pr_, pp_ = params["scalar"]

        # CPU output buffers (double precision)
        oQ = _ffi.new("double*")
        oP = np.zeros(data_handle.ne, np.float64)
        ogck_r = np.zeros(nw, np.float32)
        ogck_i = np.zeros(nw, np.float32)
        ogm0 = np.zeros(nu_, np.float32)
        ogg0 = np.zeros(ng_, np.float32)
        ogsc = np.zeros(6, np.float32)

        # Keep CFFI buffer refs alive during call
        ka = []
        def _fb(a):
            arr = np.ascontiguousarray(a, np.float32)
            buf = _ffi.from_buffer(arr)
            ka.append(buf)
            return _ffi.cast("float*", buf)

        # For double output buffers: use float64 numpy array and a raw pointer
        oP_buf = _ffi.from_buffer(np.ascontiguousarray(oP))
        ka.append(oP_buf)

        ne = data_handle.ne

        self._lib.cuda_compute_v3_f32(
            self._ctx, data_handle.ptr,
            _fb(ck_r), _fb(ck_i),
            _fb(m0), _fb(g0),
            G, DG, DM, Ap_, pr_, pp_, norm_val, use_norm,
            oQ, _ffi.cast("double*", oP_buf),
            _fb(ogck_r), _fb(ogck_i),
            _fb(ogm0), _fb(ogg0),
            _fb(ogsc),
        )

        # Reduce gradients from (n_unique_bw,) -> (n_m0_params,)
        # Use float64 for accumulation to avoid overflow
        m0_idx = self.config["m0_index"]
        g0_idx = self.config["g0_index"]
        grad_m0 = np.zeros(self.n_m0_params, np.float64)
        grad_g0 = np.zeros(self.n_g0_params, np.float64)
        for i, idx in enumerate(m0_idx):
            grad_m0[idx] += float(ogm0[i])
        for i, idx in enumerate(g0_idx):
            grad_g0[idx] += float(ogg0[i])

        grads = {
            "ck": ogck_r + 1j * ogck_i,
            "m0": grad_m0,
            "g0": grad_g0,
            "scalar": ogsc.copy(),
        }

        return oQ[0], grads, oP

    # -- gram matrix (phsp pre-integration) -----------------------------------

    def compute_gram(self, phsp_handle, m0, g0):
        """Compute reduced Gram matrices from phsp data on GPU (v3 f32).

        Same interface as :meth:`CUDAKernelV2F32.compute_gram`.
        Uses Catmull-Rom interpolation for exact numpy match.
        """
        ng2 = self.n_wave // 8
        sz = ng2 * ng2

        oMpp_r = np.zeros(sz, np.float64)
        oMpp_i = np.zeros(sz, np.float64)
        oMmm_r = np.zeros(sz, np.float64)
        oMmm_i = np.zeros(sz, np.float64)
        oMpm_r = np.zeros(sz, np.float64)
        oMpm_i = np.zeros(sz, np.float64)

        m0_arr = np.zeros(self.n_unique_bw, np.float32)
        m0_arr[:len(m0)] = np.asarray(m0, np.float32)
        g0_arr = np.zeros(self.n_gamma_rows, np.float32)
        g0_arr[:len(g0)] = np.asarray(g0, np.float32)

        ka = []
        def _fb(a):
            arr = np.ascontiguousarray(a, np.float32)
            buf = _ffi.from_buffer(arr)
            ka.append(buf)
            return _ffi.cast("float*", buf)

        self._lib.cuda_gram_matrix_v3_f32(
            self._ctx, phsp_handle.ptr,
            _fb(m0_arr), _fb(g0_arr),
            _ffi.cast("double*", _ffi.from_buffer(oMpp_r)),
            _ffi.cast("double*", _ffi.from_buffer(oMpp_i)),
            _ffi.cast("double*", _ffi.from_buffer(oMmm_r)),
            _ffi.cast("double*", _ffi.from_buffer(oMmm_i)),
            _ffi.cast("double*", _ffi.from_buffer(oMpm_r)),
            _ffi.cast("double*", _ffi.from_buffer(oMpm_i)),
        )

        Mpp = oMpp_r.reshape(ng2, ng2) + 1j * oMpp_i.reshape(ng2, ng2)
        Mmm = oMmm_r.reshape(ng2, ng2) + 1j * oMmm_i.reshape(ng2, ng2)
        Mpm = oMpm_r.reshape(ng2, ng2) + 1j * oMpm_i.reshape(ng2, ng2)

        return Mpp, Mmm, Mpm

    # -- cleanup -------------------------------------------------------

    def free(self):
        """Free GPU context (config arrays + scratch)."""
        if self._ctx is not None:
            self._lib.cuda_free_context_v3_f32(self._ctx)
            self._ctx = None
        self._ka.clear()

    def __del__(self):
        self.free()
