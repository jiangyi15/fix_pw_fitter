"""
CUDA merged-index kernel — wraps libcuda_kernels_merged.so.

This variant pre-merges m0_index/mass_index with bw_order, eliminating
the scatter step. BW is computed at all 896 positions directly.

Usage:
    kernel = CUDAMergedKernel(config)
    kernel.load_data(data)
    Q, grads, P = kernel.compute(params)
"""
import numpy as np
from cffi import FFI
import os

ffi = FFI()

CDEF = """
/* Memory management */
int cuda_alloc(void** ptr, unsigned long size);
int cuda_free(void* ptr);
int cuda_memcpy_to_device(void* dst, const void* src, unsigned long size);
int cuda_memcpy_to_host(void* dst, const void* src, unsigned long size);
int cuda_memset(void* ptr, int value, unsigned long size);

int cuda_get_device_count();
int cuda_get_device_name(char* name, int len);

/* g_bw computation (same as original) */
void launch_compute_g_bw(
    const double* mass, const double* g0,
    const int* g0_index, const int* g0_mass_index,
    const double* matrix_gamma,
    const double* gamma_table_real, const double* gamma_table_imag,
    double gamma_min, double gamma_delta,
    int n_gamma_rows, int n_unique_bw, int n_mass, int gamma_table_bins,
    double* g_interp_real, double* g_interp_imag,
    double* g_bw_real, double* g_bw_imag,
    int n_events);

/* Merged-index main forward kernel */
void launch_compute_main_merged(
    const double* mass, const double* momentum, const double* angle,
    const double* frac, const double* time, const double* weight, const double* bkg,
    const int* bw_m0_index, const int* fl_type,
    const int* bw_mass_index, const int* fl_q_index,
    const int* fl_order, const int* angle_index,
    const double* angle_k, const double* angle_b,
    const double* matrix_angle_real, const double* matrix_angle_imag,
    const double* g_bw_real, const double* g_bw_imag,
    const double* fl_table,
    double fl_min, double fl_delta,
    int n_wave, int n_res, int n_decay, int n_total_positions,
    int n_mass, int n_momentum, int n_angle_k, int n_angle_total,
    int fl_table_bins,
    const double* ck_real, const double* ck_imag, const double* m0,
    double Gamma, double Delta_Gamma, double Delta_m,
    double A_p, double poq_rho, double pop_phi,
    double* Q_out, double* P_out,
    double* pap_real, double* pap_imag, double* pam_real, double* pam_imag,
    double* gp_real, double* gp_imag, double* gm_real, double* gm_imag,
    double* poq_real, double* poq_imag,
    double* bw_p_real, double* bw_p_imag,
    double* common_amp_factor_real, double* common_amp_factor_imag,
    double* ap_real, double* ap_imag, double* am_real, double* am_imag,
    double* dQ_dP,
    double* bw_dom_real, double* bw_dom_imag,
    int n_events, int use_norm, double norm);

/* Merged-index gradient kernel */
void launch_gradient_merged(
    const double* P,
    const double* pap_real, const double* pap_imag,
    const double* pam_real, const double* pam_imag,
    const double* gp_real, const double* gp_imag,
    const double* gm_real, const double* gm_imag,
    const double* poq_real, const double* poq_imag,
    const double* bw_p_real, const double* bw_p_imag,
    const double* common_amp_factor_real,
    const double* common_amp_factor_imag,
    const double* ap_real, const double* ap_imag,
    const double* am_real, const double* am_imag,
    const double* dQ_dP,
    const double* bw_dom_real, const double* bw_dom_imag,
    const double* g_interp_real, const double* g_interp_imag,
    const double* g_bw_real, const double* g_bw_imag,
    const double* frac, const double* time,
    const int* m0_index, const int* g0_index,
    const int* bw_order,
    const double* matrix_gamma,
    const double* m0, const double* g0,
    const double* ck_real, const double* ck_imag,
    double Gamma, double Delta_Gamma, double Delta_m,
    double A_p, double poq_rho, double pop_phi,
    int n_wave, int n_res, int n_unique_bw, int n_total_positions,
    int n_gamma_rows, int n_mass,
    double* grad_ck_real_partial,
    double* grad_ck_imag_partial,
    double* grad_m0_partial,
    double* grad_g0_partial,
    double* grad_Gamma_partial,
    double* grad_DeltaGamma_partial,
    double* grad_DeltaM_partial,
    double* grad_Ap_partial,
    double* grad_poq_rho_partial,
    double* grad_pop_phi_partial,
    int n_events);

/* Scatter-indexed g_bw: interpolates gamma + scatters to (N, n_total_positions) */
void launch_compute_g_bw_scatter(
    const double* mass, const double* g0,
    const int* g0_index, const int* g0_mass_index,
    const double* gamma_table_real, const double* gamma_table_imag,
    double gamma_min, double gamma_delta,
    int n_gamma_rows, int n_total_positions, int n_mass, int gamma_table_bins,
    const int* bw_pos_gamma_idx, const int* bw_pos_gamma_off,
    double* g_interp_real, double* g_interp_imag,
    double* g_bw_real, double* g_bw_imag,
    int n_events);
"""

ffi.cdef(CDEF)


def _load_lib():
    lib_path = os.path.join(os.path.dirname(__file__), "cuda", "libcuda_kernels_merged.so")
    if not os.path.exists(lib_path):
        raise RuntimeError(f"Merged CUDA library not found at {lib_path}. Run build first.")
    return ffi.dlopen(lib_path)


class GPUDataBuffer:
    """Manages a set of GPU buffers that can be uploaded in bulk."""
    def __init__(self, lib, fields):
        self.lib = lib
        self._ptrs = {}
        self._sizes = {}
        for name, (shape, dtype) in fields:
            nelem = int(np.prod(shape))
            nbytes = nelem * np.dtype(dtype).itemsize
            ptr = ffi.new("void**")
            lib.cuda_alloc(ptr, nbytes)
            self._ptrs[name] = ptr[0]
            self._sizes[name] = nbytes

    def set(self, name, arr):
        arr = np.ascontiguousarray(arr)
        assert arr.nbytes <= self._sizes[name], f"{name}: {arr.nbytes} > {self._sizes[name]}"
        self.lib.cuda_memcpy_to_device(self._ptrs[name], ffi.from_buffer(arr), arr.nbytes)

    def ptr(self, name):
        return self._ptrs[name]

    def free(self):
        for p in self._ptrs.values():
            self.lib.cuda_free(p)
        self._ptrs.clear()


class GPUDataHolder:
    """Holds pointers for a single dataset loaded on GPU, plus scratch space."""
    def __init__(self, lib, n_wave, n_unique_bw, n_gamma_rows, n_total_positions):
        self.lib = lib
        self.n_wave = n_wave
        self.n_unique_bw = n_unique_bw
        self.n_gamma_rows = n_gamma_rows
        self.n_total_positions = n_total_positions  # n_wave * n_res

        # Global device pointers (set by load_data)
        self.d_mass = None
        self.d_momentum = None
        self.d_angle = None
        self.d_frac = None
        self.d_time = None
        self.d_weight = None
        self.d_bkg = None
        self.d_bw_m0_index = None   # merged index
        self.d_bw_mass_index = None  # merged index
        self.d_bw_order = None       # still needed for backward scatter
        self.d_m0_index = None       # still needed for backward
        self.d_n_events = 0
        self._scratch = None

    def alloc_intermediates(self, n_events):
        lib = self.lib
        def _alloc(shape, dtype):
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            ptr = ffi.new("void**")
            lib.cuda_alloc(ptr, nbytes)
            return ptr[0]

        nw = self.n_wave
        nu = self.n_unique_bw
        ng = self.n_gamma_rows
        nt = self.n_total_positions

        self._scratch = {
            # g_bw intermediates (unique positions)
            "g_interp_real": _alloc((n_events, ng), np.float64),
            "g_interp_imag": _alloc((n_events, ng), np.float64),
            "g_bw_real": _alloc((n_events, nt), np.float64),   # per-position layout
            "g_bw_imag": _alloc((n_events, nt), np.float64),

            # Forward outputs
            "Q_out": _alloc((n_events,), np.float64),
            "P_out": _alloc((n_events,), np.float64),
            "pap_real": _alloc((n_events,), np.float64),
            "pap_imag": _alloc((n_events,), np.float64),
            "pam_real": _alloc((n_events,), np.float64),
            "pam_imag": _alloc((n_events,), np.float64),
            "gp_real": _alloc((n_events,), np.float64),
            "gp_imag": _alloc((n_events,), np.float64),
            "gm_real": _alloc((n_events,), np.float64),
            "gm_imag": _alloc((n_events,), np.float64),
            "poq_real": _alloc((n_events,), np.float64),
            "poq_imag": _alloc((n_events,), np.float64),
            "bw_p_real": _alloc((n_events, nw), np.float64),
            "bw_p_imag": _alloc((n_events, nw), np.float64),
            "common_amp_factor_real": _alloc((n_events, nw), np.float64),
            "common_amp_factor_imag": _alloc((n_events, nw), np.float64),
            "ap_real": _alloc((n_events,), np.float64),
            "ap_imag": _alloc((n_events,), np.float64),
            "am_real": _alloc((n_events,), np.float64),
            "am_imag": _alloc((n_events,), np.float64),
            "dQ_dP": _alloc((n_events,), np.float64),
            # Per-position bw_dom (n_total_positions per event)
            "bw_dom_real": _alloc((n_events, nt), np.float64),
            "bw_dom_imag": _alloc((n_events, nt), np.float64),
            # Gradient partials
            "grad_ck_real_partial": _alloc((n_events, nw), np.float64),
            "grad_ck_imag_partial": _alloc((n_events, nw), np.float64),
            "grad_m0_partial": _alloc((n_events, nu), np.float64),
            "grad_g0_partial": _alloc((n_events, ng), np.float64),
            "grad_Gamma_partial": _alloc((n_events,), np.float64),
            "grad_DeltaGamma_partial": _alloc((n_events,), np.float64),
            "grad_DeltaM_partial": _alloc((n_events,), np.float64),
            "grad_Ap_partial": _alloc((n_events,), np.float64),
            "grad_poq_rho_partial": _alloc((n_events,), np.float64),
            "grad_pop_phi_partial": _alloc((n_events,), np.float64),
        }

    def free(self):
        for p in self._scratch.values():
            self.lib.cuda_free(p)
        self._scratch = None
        for attr in ['d_mass', 'd_momentum', 'd_angle', 'd_frac',
                      'd_time', 'd_weight', 'd_bkg',
                      'd_bw_m0_index', 'd_bw_mass_index', 'd_bw_order',
                      'd_m0_index']:
            p = getattr(self, attr, None)
            if p is not None:
                self.lib.cuda_free(p)
                setattr(self, attr, None)


class CUDAMergedKernel:
    """CUDA kernel with merged-index BW computation."""

    def __init__(self, config):
        self.lib = _load_lib()

        # Detect GPU
        name_buf = ffi.new("char[256]")
        self.lib.cuda_get_device_name(name_buf, 256)
        gpu_name = ffi.string(name_buf).decode()
        print(f"✓ Using GPU: {gpu_name}")

        # Config dimensions
        self.n_wave = config["matrix_angle"].shape[1]
        self.n_res = config["bw_order"].size // self.n_wave
        self.n_decay = config["fl_order"].size // self.n_wave
        self.n_unique_bw = len(config["m0_index"])
        self.n_gamma_rows = len(config["g0_index"])
        self.n_angle_k = config["angle_k"].shape[0]
        self.n_angle_total = int(np.max(config["angle_index"])) + 1 if len(config["angle_index"]) > 0 else 0
        self.n_total_positions = self.n_wave * self.n_res  # 896
        self.n_m0_unique = int(np.max(config["m0_index"])) + 1
        self.n_g0_unique = int(np.max(config["g0_index"])) + 1
        self.scalar_n = 6

        # Pre-compute merged indices (CPU)
        bo = config["bw_order"]
        self._bw_m0_index = np.ascontiguousarray(
            config["m0_index"][bo].astype(np.int32))
        self._bw_mass_index = np.ascontiguousarray(
            config["mass_index"][bo].astype(np.int32))
        self._bw_order = np.ascontiguousarray(bo.astype(np.int32))
        self._m0_index = np.ascontiguousarray(config["m0_index"].astype(np.int32))

        # Pre-compute per-position gamma scatter indices
        mg = config["matrix_gamma"]  # (288, 216)
        bw_pos_gamma_idx = []
        bw_pos_gamma_off = [0]
        for pos in range(len(bo)):
            rows = np.where(mg[:, bo[pos]] != 0)[0]
            bw_pos_gamma_idx.extend(rows.tolist())
            bw_pos_gamma_off.append(len(bw_pos_gamma_idx))
        self._bw_pos_gamma_idx = np.ascontiguousarray(
            np.array(bw_pos_gamma_idx, dtype=np.int32))
        self._bw_pos_gamma_off = np.ascontiguousarray(
            np.array(bw_pos_gamma_off, dtype=np.int32))

        # Upload constant index arrays to GPU
        def _upload(arr):
            nbytes = arr.nbytes
            ptr = ffi.new("void**")
            self.lib.cuda_alloc(ptr, nbytes)
            self.lib.cuda_memcpy_to_device(ptr[0], ffi.from_buffer(arr), nbytes)
            return ptr[0]

        self.d_bw_m0_index = _upload(self._bw_m0_index)
        self.d_bw_mass_index = _upload(self._bw_mass_index)
        self.d_bw_order = _upload(self._bw_order)
        self.d_m0_index = _upload(self._m0_index)
        self.d_bw_pos_gamma_idx = _upload(self._bw_pos_gamma_idx)
        self.d_bw_pos_gamma_off = _upload(self._bw_pos_gamma_off)

        # Upload constant physical arrays
        self._matrix_gamma = np.ascontiguousarray(config["matrix_gamma"].astype(np.float64))
        self._matrix_angle_real = np.ascontiguousarray(np.real(config["matrix_angle"]).astype(np.float64))
        self._matrix_angle_imag = np.ascontiguousarray(np.imag(config["matrix_angle"]).astype(np.float64))
        self._gamma_table_real = np.ascontiguousarray(np.real(config["gamma_table"]).astype(np.float64))
        self._gamma_table_imag = np.ascontiguousarray(np.imag(config["gamma_table"]).astype(np.float64))
        self._fl_table = np.ascontiguousarray(config["fl_table"].astype(np.float64))
        self._angle_k = np.ascontiguousarray(config["angle_k"].astype(np.float64))
        self._angle_b = np.ascontiguousarray(config["angle_b"].astype(np.float64))
        self._g0_index = np.ascontiguousarray(config["g0_index"].astype(np.int32))
        self._g0_mass_index = np.ascontiguousarray(config["g0_mass_index"].astype(np.int32))
        self._fl_type = np.ascontiguousarray(config["fl_type"].astype(np.int32))
        self._fl_q_index = np.ascontiguousarray(config["fl_q_index"].astype(np.int32))
        self._mass_index = np.ascontiguousarray(config["mass_index"].astype(np.int32))
        self._angle_index = np.ascontiguousarray(config["angle_index"].astype(np.int32))
        self._fl_order = np.ascontiguousarray(config["fl_order"].astype(np.int32))

        def _upload_const(arr):
            nbytes = arr.nbytes
            ptr = ffi.new("void**")
            self.lib.cuda_alloc(ptr, nbytes)
            self.lib.cuda_memcpy_to_device(ptr[0], ffi.from_buffer(arr), nbytes)
            return ptr[0]

        self.d_matrix_gamma = _upload_const(self._matrix_gamma)
        self.d_matrix_angle_real = _upload_const(self._matrix_angle_real)
        self.d_matrix_angle_imag = _upload_const(self._matrix_angle_imag)
        self.d_gamma_table_real = _upload_const(self._gamma_table_real)
        self.d_gamma_table_imag = _upload_const(self._gamma_table_imag)
        self.d_fl_table = _upload_const(self._fl_table)
        self.d_angle_k = _upload_const(self._angle_k)
        self.d_angle_b = _upload_const(self._angle_b)
        self.d_g0_index = _upload_const(self._g0_index)
        self.d_g0_mass_index = _upload_const(self._g0_mass_index)
        self.d_fl_type = _upload_const(self._fl_type)
        self.d_fl_q_index = _upload_const(self._fl_q_index)
        self.d_mass_index = _upload_const(self._mass_index)
        self.d_angle_index = _upload_const(self._angle_index)
        self.d_fl_order = _upload_const(self._fl_order)

        self.gamma_min = float(config["gamma_min"])
        self.gamma_delta = float(config["gamma_delta"])
        self.fl_min = float(config["fl_min"])
        self.fl_delta = float(config["fl_delta"])
        self._n_bins_gamma = config["gamma_table"].shape[-1]
        self._n_bins_fl = config["fl_table"].shape[-1]

        self._gpu_data = None  # set by load_data

    def load_data(self, data_np):
        """Upload event data to GPU, create scratch buffers."""
        lib = self.lib
        n = data_np["mass"].shape[0]
        gpu = GPUDataHolder(lib, self.n_wave, self.n_unique_bw,
                            self.n_gamma_rows, self.n_total_positions)
        gpu.d_n_events = n
        gpu.alloc_intermediates(n)

        def _upload(arr):
            arr = np.ascontiguousarray(arr.astype(np.float64))
            nbytes = arr.nbytes
            ptr = ffi.new("void**")
            lib.cuda_alloc(ptr, nbytes)
            lib.cuda_memcpy_to_device(ptr[0], ffi.from_buffer(arr), nbytes)
            return ptr[0]

        gpu.d_mass = _upload(data_np["mass"])
        gpu.d_momentum = _upload(data_np["q"])
        gpu.d_angle = _upload(data_np["angle"])
        gpu.d_frac = _upload(data_np["frac"])
        gpu.d_time = _upload(data_np["time"])
        gpu.d_weight = _upload(data_np["weight"])
        bkg = data_np.get("bkg", np.zeros(n, dtype=np.float64))
        gpu.d_bkg = _upload(np.asarray(bkg))

        self._gpu_data = gpu
        return gpu

    def free(self):
        if self._gpu_data is not None:
            self._gpu_data.free()
            self._gpu_data = None

    def compute(self, params, data_handle, norm=None):
        """Run forward + backward pass on GPU."""
        lib = self.lib
        gpu = data_handle
        n = gpu.d_n_events

        scratch = gpu._scratch
        nw = self.n_wave
        nu = self.n_unique_bw
        ng = self.n_gamma_rows
        nt = self.n_total_positions

        def _upload(arr):
            arr = np.ascontiguousarray(arr.astype(np.float64))
            nbytes = arr.nbytes
            ptr = ffi.new("void**")
            lib.cuda_alloc(ptr, nbytes)
            lib.cuda_memcpy_to_device(ptr[0], ffi.from_buffer(arr), nbytes)
            return ptr[0]

        d_g0 = _upload(np.asarray(params["g0"]))

        # 1. Compute g_bw at per-position layout via scatter-indexed kernel
        #    (interpolates gamma + scatters directly to 896 positions)
        lib.launch_compute_g_bw_scatter(
            gpu.d_mass, d_g0,
            self.d_g0_index, self.d_g0_mass_index,
            self.d_gamma_table_real, self.d_gamma_table_imag,
            self.gamma_min, self.gamma_delta,
            ng, nt, 48, self._n_bins_gamma,
            self.d_bw_pos_gamma_idx, self.d_bw_pos_gamma_off,
            scratch["g_interp_real"], scratch["g_interp_imag"],
            scratch["g_bw_real"], scratch["g_bw_imag"],
            n)

        # 2. Merged-index main forward
        use_norm = 0 if norm is None else 1
        norm_val = norm if norm is not None else 1.0

        ck_real = np.ascontiguousarray(np.real(params["ck"]).astype(np.float64))
        ck_imag = np.ascontiguousarray(np.imag(params["ck"]).astype(np.float64))
        m0_arr = np.ascontiguousarray(params["m0"].astype(np.float64))
        scalar = params["scalar"]
        Gamma, Delta_Gamma, Delta_m, A_p, poq_rho, pop_phi = scalar

        d_ck_real = _upload(ck_real)
        d_ck_imag = _upload(ck_imag)
        d_m0 = _upload(m0_arr)
        d_g0 = _upload(np.asarray(params["g0"]))

        lib.launch_compute_main_merged(
            gpu.d_mass, gpu.d_momentum, gpu.d_angle,
            gpu.d_frac, gpu.d_time, gpu.d_weight, gpu.d_bkg,
            self.d_bw_m0_index, self.d_fl_type,
            self.d_bw_mass_index, self.d_fl_q_index,
            self.d_fl_order, self.d_angle_index,
            self.d_angle_k, self.d_angle_b,
            self.d_matrix_angle_real, self.d_matrix_angle_imag,
            scratch["g_bw_real"], scratch["g_bw_imag"],
            self.d_fl_table,
            self.fl_min, self.fl_delta,
            nw, self.n_res, self.n_decay, nt,
            48, 72, self.n_angle_k, self.n_angle_total,
            self._n_bins_fl,
            d_ck_real, d_ck_imag, d_m0,
            Gamma, Delta_Gamma, Delta_m, A_p, poq_rho, pop_phi,
            scratch["Q_out"], scratch["P_out"],
            scratch["pap_real"], scratch["pap_imag"],
            scratch["pam_real"], scratch["pam_imag"],
            scratch["gp_real"], scratch["gp_imag"],
            scratch["gm_real"], scratch["gm_imag"],
            scratch["poq_real"], scratch["poq_imag"],
            scratch["bw_p_real"], scratch["bw_p_imag"],
            scratch["common_amp_factor_real"], scratch["common_amp_factor_imag"],
            scratch["ap_real"], scratch["ap_imag"],
            scratch["am_real"], scratch["am_imag"],
            scratch["dQ_dP"],
            scratch["bw_dom_real"], scratch["bw_dom_imag"],
            n, use_norm, norm_val)

        lib.cuda_free(d_ck_real)
        lib.cuda_free(d_ck_imag)
        lib.cuda_free(d_m0)
        lib.cuda_free(d_g0)

        # 4. Merged-index gradient
        lib.launch_gradient_merged(
            scratch["P_out"],
            scratch["pap_real"], scratch["pap_imag"],
            scratch["pam_real"], scratch["pam_imag"],
            scratch["gp_real"], scratch["gp_imag"],
            scratch["gm_real"], scratch["gm_imag"],
            scratch["poq_real"], scratch["poq_imag"],
            scratch["bw_p_real"], scratch["bw_p_imag"],
            scratch["common_amp_factor_real"], scratch["common_amp_factor_imag"],
            scratch["ap_real"], scratch["ap_imag"],
            scratch["am_real"], scratch["am_imag"],
            scratch["dQ_dP"],
            scratch["bw_dom_real"], scratch["bw_dom_imag"],
            scratch["g_interp_real"], scratch["g_interp_imag"],
            scratch["g_bw_real"], scratch["g_bw_imag"],
            gpu.d_frac, gpu.d_time,
            self.d_m0_index, self.d_g0_index,
            self.d_bw_order,
            self.d_matrix_gamma,
            d_m0, ffi.NULL,  # m0 and g0 — g0 is not used in merged gradient
            d_ck_real, d_ck_imag,
            Gamma, Delta_Gamma, Delta_m, A_p, poq_rho, pop_phi,
            nw, self.n_res, nu, nt, ng, 48,
            scratch["grad_ck_real_partial"], scratch["grad_ck_imag_partial"],
            scratch["grad_m0_partial"],
            scratch["grad_g0_partial"],
            scratch["grad_Gamma_partial"],
            scratch["grad_DeltaGamma_partial"],
            scratch["grad_DeltaM_partial"],
            scratch["grad_Ap_partial"],
            scratch["grad_poq_rho_partial"],
            scratch["grad_pop_phi_partial"],
            n)

        # 5. Download results
        def _download(ptr, shape, dtype):
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            buf = ffi.new(f"char[{nbytes}]")
            lib.cuda_memcpy_to_host(buf, ptr, nbytes)
            return np.frombuffer(ffi.buffer(buf), dtype=dtype).reshape(shape)

        Q_val = float(np.sum(_download(scratch["Q_out"], (n,), np.float64)))
        P_arr = _download(scratch["P_out"], (n,), np.float64)

        # Accumulate gradients
        grad_ck_real = _download(scratch["grad_ck_real_partial"], (n, nw), np.float64).sum(axis=0)
        grad_ck_imag = _download(scratch["grad_ck_imag_partial"], (n, nw), np.float64).sum(axis=0)
        grad_m0 = _download(scratch["grad_m0_partial"], (n, nu), np.float64).sum(axis=0)
        grad_g0 = _download(scratch["grad_g0_partial"], (n, ng), np.float64).sum(axis=0)

        # Scatter m0 gradient by m0_index
        m0_scatter = np.zeros((nu, self.n_m0_unique), dtype=np.float64)
        for i, m_idx in enumerate(self._m0_index):
            m0_scatter[i, m_idx] = 1.0
        grad_m0 = grad_m0 @ m0_scatter

        # Scatter g0 gradient by g0_index
        g0_scatter = np.zeros((ng, self.n_g0_unique), dtype=np.float64)
        for i, g_idx in enumerate(self._g0_index):
            g0_scatter[i, g_idx] = 1.0
        grad_g0 = grad_g0 @ g0_scatter

        # Scalar grads
        g_Gamma = float(_download(scratch["grad_Gamma_partial"], (n,), np.float64).sum())
        g_DGamma = float(_download(scratch["grad_DeltaGamma_partial"], (n,), np.float64).sum())
        g_DM = float(_download(scratch["grad_DeltaM_partial"], (n,), np.float64).sum())
        g_Ap = float(_download(scratch["grad_Ap_partial"], (n,), np.float64).sum())
        g_poq_rho = float(_download(scratch["grad_poq_rho_partial"], (n,), np.float64).sum())
        g_pop_phi = float(_download(scratch["grad_pop_phi_partial"], (n,), np.float64).sum())

        grads = {
            "ck": grad_ck_real.astype(np.complex128) + 1j * grad_ck_imag.astype(np.complex128),
            "m0": grad_m0,
            "g0": grad_g0,
            "scalar": np.array([g_Gamma, g_DGamma, g_DM, g_Ap, g_poq_rho, g_pop_phi]),
        }

        # Handle norm case properly
        if norm is not None:
            Q_val = Q_val  # already NLL
        else:
            # Q = sum(P * weight) for norm computation
            w = data_handle.d_weight
            w_host = _download(w, (n,), np.float64)
            Q_val = float(np.sum(w_host * P_arr))

        return Q_val, grads, P_arr
