"""
CUDA merged-index kernel — struct-based unified API.

This variant uses pre-merged m0_index/mass_index with bw_order, eliminating
the scatter step. BW is computed at all 896 positions directly.

Usage:
    kernel = CUDAMergedKernel(config)
    dh = kernel.load_data(data_np)
    Q, grads, P = kernel.compute(params, dh, norm=None)
    dh.free()
    kernel.free()
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

/* Struct-based unified compute API */
typedef struct {
    const double* mass; const double* momentum; const double* angle;
    const double* frac; const double* time; const double* weight; const double* bkg;
    const double* ck_real; const double* ck_imag; const double* m0; const double* g0;
    double Gamma; double Delta_Gamma; double Delta_m;
    double A_prod; double poq_rho; double pop_phi;
    double norm_val; int use_norm;
} ComputeData;

typedef struct {
    const int* bw_m0_index; const int* bw_mass_index;
    const int* fl_type; const int* fl_q_index;
    const int* fl_order; const int* angle_index;
    const int* g0_index; const int* g0_mass_index;
    const int* bw_pos_gamma_idx; const int* bw_pos_gamma_off;
    const int* m0_index; const int* bw_order;
} ComputeIndices;

typedef struct {
    const double* angle_k; const double* angle_b;
    const double* matrix_angle_real; const double* matrix_angle_imag;
    const double* gamma_table_real; const double* gamma_table_imag;
    double gamma_min; double gamma_delta; int gamma_table_bins;
    const double* matrix_gamma;
    const double* fl_table; double fl_min; double fl_delta; int fl_table_bins;
} ComputeConstants;

typedef struct {
    int n_events; int n_wave; int n_res; int n_decay;
    int n_total_positions; int n_unique_bw; int n_gamma_rows;
    int n_mass; int n_momentum; int n_angle_k; int n_angle_total;
} ComputeDims;

typedef struct {
    double* g_interp_real; double* g_interp_imag;
    double* g_bw_real; double* g_bw_imag;
    double* Q_out; double* P_out;
    double* pap_real; double* pap_imag;
    double* pam_real; double* pam_imag;
    double* gp_real; double* gp_imag;
    double* gm_real; double* gm_imag;
    double* poq_real; double* poq_imag;
    double* bw_p_real; double* bw_p_imag;
    double* common_amp_factor_real; double* common_amp_factor_imag;
    double* ap_real; double* ap_imag;
    double* am_real; double* am_imag;
    double* dQ_dP;
    double* bw_dom_real; double* bw_dom_imag;
    double* grad_ck_real_partial; double* grad_ck_imag_partial;
    double* grad_m0_partial; double* grad_g0_partial;
    double* grad_Gamma_partial; double* grad_DeltaGamma_partial;
    double* grad_DeltaM_partial; double* grad_Ap_partial;
    double* grad_poq_rho_partial; double* grad_pop_phi_partial;
} ComputeScratch;

void launch_compute_all(const ComputeData* d, const ComputeIndices* idx,
                        const ComputeConstants* c, const ComputeDims* dim,
                        ComputeScratch* s);
"""

ffi.cdef(CDEF)


def _load_lib():
    lib_path = os.path.join(os.path.dirname(__file__), "cuda", "libcuda_kernels_merged.so")
    if not os.path.exists(lib_path):
        raise RuntimeError(f"Merged CUDA library not found at {lib_path}. Run build first.")
    return ffi.dlopen(lib_path)


def _upload_arr(lib, arr):
    """Upload a numpy array to GPU, return GPU pointer."""
    arr = np.ascontiguousarray(arr)
    nbytes = arr.nbytes
    ptr = ffi.new("void**")
    lib.cuda_alloc(ptr, nbytes)
    lib.cuda_memcpy_to_device(ptr[0], ffi.from_buffer(arr), nbytes)
    return ptr[0]


def _download_arr(lib, ptr, shape, dtype):
    """Download GPU array to host numpy array."""
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    buf = ffi.new(f"char[{nbytes}]")
    lib.cuda_memcpy_to_host(buf, ptr, nbytes)
    return np.frombuffer(ffi.buffer(buf), dtype=dtype).reshape(shape).copy()


class _DataHandle:
    """Holds GPU data and scratch for a single dataset."""
    
    def __init__(self, lib, data_ptr, scratch_ptr, n_events):
        self.lib = lib
        self.data = data_ptr
        self.scratch = scratch_ptr
        self.n_events = n_events
        # Store pointers for cleanup (filled by load_data)
        self._event_ptrs = []
        self._scratch_ptrs = []
    
    def free(self):
        for p in self._event_ptrs:
            self.lib.cuda_free(p)
        for p in self._scratch_ptrs:
            self.lib.cuda_free(p)
        self._event_ptrs.clear()
        self._scratch_ptrs.clear()


class CUDAMergedKernel:
    """CUDA kernel with struct-based unified API."""

    def __init__(self, config):
        self.lib = _load_lib()
        
        # Detect GPU
        name_buf = ffi.new("char[256]")
        self.lib.cuda_get_device_name(name_buf, 256)
        gpu_name = ffi.string(name_buf).decode()
        print(f"✓ Using GPU: {gpu_name}")
        
        # Extract dimensions from config
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
        
        # Compute n_mass and n_momentum from indices
        self.n_mass = int(np.max(config["mass_index"])) + 1 if len(config["mass_index"]) > 0 else 0
        self.n_momentum = int(np.max(config["fl_q_index"])) + 1 if len(config["fl_q_index"]) > 0 else 0
        
        # Pre-compute merged indices (CPU)
        bo = config["bw_order"]
        self._bw_m0_index = np.ascontiguousarray(config["m0_index"][bo].astype(np.int32))
        self._bw_mass_index = np.ascontiguousarray(config["mass_index"][bo].astype(np.int32))
        self._bw_order = np.ascontiguousarray(bo.astype(np.int32))
        self._m0_index = np.ascontiguousarray(config["m0_index"].astype(np.int32))
        
        # Pre-compute per-position gamma scatter indices
        mg = config["matrix_gamma"]
        bw_pos_gamma_idx = []
        bw_pos_gamma_off = [0]
        for pos in range(len(bo)):
            rows = np.where(mg[:, bo[pos]] != 0)[0]
            bw_pos_gamma_idx.extend(rows.tolist())
            bw_pos_gamma_off.append(len(bw_pos_gamma_idx))
        self._bw_pos_gamma_idx = np.ascontiguousarray(np.array(bw_pos_gamma_idx, dtype=np.int32))
        self._bw_pos_gamma_off = np.ascontiguousarray(np.array(bw_pos_gamma_off, dtype=np.int32))
        
        # Upload constant index arrays to GPU
        self.d_bw_m0_index = _upload_arr(self.lib, self._bw_m0_index)
        self.d_bw_mass_index = _upload_arr(self.lib, self._bw_mass_index)
        self.d_bw_order = _upload_arr(self.lib, self._bw_order)
        self.d_m0_index = _upload_arr(self.lib, self._m0_index)
        self.d_bw_pos_gamma_idx = _upload_arr(self.lib, self._bw_pos_gamma_idx)
        self.d_bw_pos_gamma_off = _upload_arr(self.lib, self._bw_pos_gamma_off)
        
        # Upload other index arrays
        self.d_fl_type = _upload_arr(self.lib, np.ascontiguousarray(config["fl_type"].astype(np.int32)))
        self.d_fl_q_index = _upload_arr(self.lib, np.ascontiguousarray(config["fl_q_index"].astype(np.int32)))
        self.d_fl_order = _upload_arr(self.lib, np.ascontiguousarray(config["fl_order"].astype(np.int32)))
        self.d_angle_index = _upload_arr(self.lib, np.ascontiguousarray(config["angle_index"].astype(np.int32)))
        self._g0_index = np.ascontiguousarray(config["g0_index"].astype(np.int32))
        self.d_g0_index = _upload_arr(self.lib, self._g0_index)
        self.d_g0_mass_index = _upload_arr(self.lib, np.ascontiguousarray(config["g0_mass_index"].astype(np.int32)))
        
        # Upload constant physical arrays
        self.d_matrix_gamma = _upload_arr(self.lib, np.ascontiguousarray(config["matrix_gamma"].astype(np.float64)))
        self.d_matrix_angle_real = _upload_arr(self.lib, np.ascontiguousarray(np.real(config["matrix_angle"]).astype(np.float64)))
        self.d_matrix_angle_imag = _upload_arr(self.lib, np.ascontiguousarray(np.imag(config["matrix_angle"]).astype(np.float64)))
        self.d_gamma_table_real = _upload_arr(self.lib, np.ascontiguousarray(np.real(config["gamma_table"]).astype(np.float64)))
        self.d_gamma_table_imag = _upload_arr(self.lib, np.ascontiguousarray(np.imag(config["gamma_table"]).astype(np.float64)))
        self.d_fl_table = _upload_arr(self.lib, np.ascontiguousarray(config["fl_table"].astype(np.float64)))
        self.d_angle_k = _upload_arr(self.lib, np.ascontiguousarray(config["angle_k"].astype(np.float64)))
        self.d_angle_b = _upload_arr(self.lib, np.ascontiguousarray(config["angle_b"].astype(np.float64)))
        
        # Store scalar constants
        self.gamma_min = float(config["gamma_min"])
        self.gamma_delta = float(config["gamma_delta"])
        self.fl_min = float(config["fl_min"])
        self.fl_delta = float(config["fl_delta"])
        self._n_bins_gamma = config["gamma_table"].shape[-1]
        self._n_bins_fl = config["fl_table"].shape[-1]
        
        # Build persistent structs with GPU pointers
        self._ctx_idx = ffi.new("ComputeIndices*")
        self._ctx_idx.bw_m0_index = self.d_bw_m0_index
        self._ctx_idx.bw_mass_index = self.d_bw_mass_index
        self._ctx_idx.fl_type = self.d_fl_type
        self._ctx_idx.fl_q_index = self.d_fl_q_index
        self._ctx_idx.fl_order = self.d_fl_order
        self._ctx_idx.angle_index = self.d_angle_index
        self._ctx_idx.g0_index = self.d_g0_index
        self._ctx_idx.g0_mass_index = self.d_g0_mass_index
        self._ctx_idx.bw_pos_gamma_idx = self.d_bw_pos_gamma_idx
        self._ctx_idx.bw_pos_gamma_off = self.d_bw_pos_gamma_off
        self._ctx_idx.m0_index = self.d_m0_index
        self._ctx_idx.bw_order = self.d_bw_order
        
        self._ctx_c = ffi.new("ComputeConstants*")
        self._ctx_c.angle_k = self.d_angle_k
        self._ctx_c.angle_b = self.d_angle_b
        self._ctx_c.matrix_angle_real = self.d_matrix_angle_real
        self._ctx_c.matrix_angle_imag = self.d_matrix_angle_imag
        self._ctx_c.gamma_table_real = self.d_gamma_table_real
        self._ctx_c.gamma_table_imag = self.d_gamma_table_imag
        self._ctx_c.gamma_min = self.gamma_min
        self._ctx_c.gamma_delta = self.gamma_delta
        self._ctx_c.gamma_table_bins = self._n_bins_gamma
        self._ctx_c.matrix_gamma = self.d_matrix_gamma
        self._ctx_c.fl_table = self.d_fl_table
        self._ctx_c.fl_min = self.fl_min
        self._ctx_c.fl_delta = self.fl_delta
        self._ctx_c.fl_table_bins = self._n_bins_fl
        
        self._ctx_dim = ffi.new("ComputeDims*")
        self._ctx_dim.n_events = 0  # Set per-dataset
        self._ctx_dim.n_wave = self.n_wave
        self._ctx_dim.n_res = self.n_res
        self._ctx_dim.n_decay = self.n_decay
        self._ctx_dim.n_total_positions = self.n_total_positions
        self._ctx_dim.n_unique_bw = self.n_unique_bw
        self._ctx_dim.n_gamma_rows = self.n_gamma_rows
        self._ctx_dim.n_mass = self.n_mass
        self._ctx_dim.n_momentum = self.n_momentum
        self._ctx_dim.n_angle_k = self.n_angle_k
        self._ctx_dim.n_angle_total = self.n_angle_total
        
        self._handles = []
    
    def load_data(self, data_np):
        """Upload event data to GPU, create scratch buffers, return DataHandle."""
        lib = self.lib
        n = data_np["mass"].shape[0]
        
        # Create ComputeData struct
        data = ffi.new("ComputeData*")
        
        # Upload event data
        d_mass = _upload_arr(lib, np.ascontiguousarray(data_np["mass"].astype(np.float64)))
        d_momentum = _upload_arr(lib, np.ascontiguousarray(data_np["q"].astype(np.float64)))
        d_angle = _upload_arr(lib, np.ascontiguousarray(data_np["angle"].astype(np.float64)))
        d_frac = _upload_arr(lib, np.ascontiguousarray(data_np["frac"].astype(np.float64)))
        d_time = _upload_arr(lib, np.ascontiguousarray(data_np["time"].astype(np.float64)))
        d_weight = _upload_arr(lib, np.ascontiguousarray(data_np["weight"].astype(np.float64)))
        bkg = data_np.get("bkg", np.zeros(n, dtype=np.float64))
        d_bkg = _upload_arr(lib, np.ascontiguousarray(bkg.astype(np.float64)))
        
        data.mass = d_mass
        data.momentum = d_momentum
        data.angle = d_angle
        data.frac = d_frac
        data.time = d_time
        data.weight = d_weight
        data.bkg = d_bkg
        
        # Create ComputeScratch struct with GPU buffers
        scratch = ffi.new("ComputeScratch*")
        scratch_ptrs = []
        
        def _alloc_scratch(shape, dtype):
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            ptr = ffi.new("void**")
            lib.cuda_alloc(ptr, nbytes)
            scratch_ptrs.append(ptr[0])
            return ptr[0]
        
        nw = self.n_wave
        nu = self.n_unique_bw
        ng = self.n_gamma_rows
        nt = self.n_total_positions
        
        scratch.g_interp_real = _alloc_scratch((n, ng), np.float64)
        scratch.g_interp_imag = _alloc_scratch((n, ng), np.float64)
        scratch.g_bw_real = _alloc_scratch((n, nt), np.float64)
        scratch.g_bw_imag = _alloc_scratch((n, nt), np.float64)
        
        scratch.Q_out = _alloc_scratch((n,), np.float64)
        scratch.P_out = _alloc_scratch((n,), np.float64)
        scratch.pap_real = _alloc_scratch((n,), np.float64)
        scratch.pap_imag = _alloc_scratch((n,), np.float64)
        scratch.pam_real = _alloc_scratch((n,), np.float64)
        scratch.pam_imag = _alloc_scratch((n,), np.float64)
        scratch.gp_real = _alloc_scratch((n,), np.float64)
        scratch.gp_imag = _alloc_scratch((n,), np.float64)
        scratch.gm_real = _alloc_scratch((n,), np.float64)
        scratch.gm_imag = _alloc_scratch((n,), np.float64)
        scratch.poq_real = _alloc_scratch((n,), np.float64)
        scratch.poq_imag = _alloc_scratch((n,), np.float64)
        scratch.bw_p_real = _alloc_scratch((n, nw), np.float64)
        scratch.bw_p_imag = _alloc_scratch((n, nw), np.float64)
        scratch.common_amp_factor_real = _alloc_scratch((n, nw), np.float64)
        scratch.common_amp_factor_imag = _alloc_scratch((n, nw), np.float64)
        scratch.ap_real = _alloc_scratch((n,), np.float64)
        scratch.ap_imag = _alloc_scratch((n,), np.float64)
        scratch.am_real = _alloc_scratch((n,), np.float64)
        scratch.am_imag = _alloc_scratch((n,), np.float64)
        scratch.dQ_dP = _alloc_scratch((n,), np.float64)
        scratch.bw_dom_real = _alloc_scratch((n, nt), np.float64)
        scratch.bw_dom_imag = _alloc_scratch((n, nt), np.float64)
        
        scratch.grad_ck_real_partial = _alloc_scratch((n, nw), np.float64)
        scratch.grad_ck_imag_partial = _alloc_scratch((n, nw), np.float64)
        scratch.grad_m0_partial = _alloc_scratch((n, nu), np.float64)
        scratch.grad_g0_partial = _alloc_scratch((n, ng), np.float64)
        scratch.grad_Gamma_partial = _alloc_scratch((n,), np.float64)
        scratch.grad_DeltaGamma_partial = _alloc_scratch((n,), np.float64)
        scratch.grad_DeltaM_partial = _alloc_scratch((n,), np.float64)
        scratch.grad_Ap_partial = _alloc_scratch((n,), np.float64)
        scratch.grad_poq_rho_partial = _alloc_scratch((n,), np.float64)
        scratch.grad_pop_phi_partial = _alloc_scratch((n,), np.float64)
        
        # Create handle
        handle = _DataHandle(lib, data, scratch, n)
        handle._event_ptrs = [d_mass, d_momentum, d_angle, d_frac, d_time, d_weight, d_bkg]
        handle._scratch_ptrs = scratch_ptrs
        handle._weight_ptr = d_weight  # Store for norm computation
        
        self._handles.append(handle)
        return handle
    
    def compute(self, params, dh, norm=None):
        """Run forward + backward pass on GPU using single unified call."""
        lib = self.lib
        n = dh.n_events
        data = dh.data
        scratch = dh.scratch
        
        nw = self.n_wave
        nu = self.n_unique_bw
        ng = self.n_gamma_rows
        
        # Upload per-call params
        ck_real = np.ascontiguousarray(np.real(params["ck"]).astype(np.float64))
        ck_imag = np.ascontiguousarray(np.imag(params["ck"]).astype(np.float64))
        m0_arr = np.ascontiguousarray(params["m0"].astype(np.float64))
        g0_arr = np.ascontiguousarray(params["g0"].astype(np.float64))
        scalar = params["scalar"]
        Gamma, Delta_Gamma, Delta_m, A_p, poq_rho, pop_phi = scalar
        
        d_ck_real = _upload_arr(lib, ck_real)
        d_ck_imag = _upload_arr(lib, ck_imag)
        d_m0 = _upload_arr(lib, m0_arr)
        d_g0 = _upload_arr(lib, g0_arr)
        
        # Fill ComputeData with params
        data.ck_real = d_ck_real
        data.ck_imag = d_ck_imag
        data.m0 = d_m0
        data.g0 = d_g0
        data.Gamma = Gamma
        data.Delta_Gamma = Delta_Gamma
        data.Delta_m = Delta_m
        data.A_prod = A_p
        data.poq_rho = poq_rho
        data.pop_phi = pop_phi
        data.use_norm = 0 if norm is None else 1
        data.norm_val = norm if norm is not None else -1e100
        
        # Update n_events in dims
        self._ctx_dim.n_events = n
        
        # Single unified call
        lib.launch_compute_all(data, self._ctx_idx, self._ctx_c, self._ctx_dim, scratch)
        
        # Free per-call params
        lib.cuda_free(d_ck_real)
        lib.cuda_free(d_ck_imag)
        lib.cuda_free(d_m0)
        lib.cuda_free(d_g0)
        
        # Download results
        Q_val = float(np.sum(_download_arr(lib, scratch.Q_out, (n,), np.float64)))
        P_arr = _download_arr(lib, scratch.P_out, (n,), np.float64)
        
        # Accumulate gradients
        grad_ck_real = _download_arr(lib, scratch.grad_ck_real_partial, (n, nw), np.float64).sum(axis=0)
        grad_ck_imag = _download_arr(lib, scratch.grad_ck_imag_partial, (n, nw), np.float64).sum(axis=0)
        grad_m0 = _download_arr(lib, scratch.grad_m0_partial, (n, nu), np.float64).sum(axis=0)
        grad_g0 = _download_arr(lib, scratch.grad_g0_partial, (n, ng), np.float64).sum(axis=0)
        
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
        g_Gamma = float(_download_arr(lib, scratch.grad_Gamma_partial, (n,), np.float64).sum())
        g_DGamma = float(_download_arr(lib, scratch.grad_DeltaGamma_partial, (n,), np.float64).sum())
        g_DM = float(_download_arr(lib, scratch.grad_DeltaM_partial, (n,), np.float64).sum())
        g_Ap = float(_download_arr(lib, scratch.grad_Ap_partial, (n,), np.float64).sum())
        g_poq_rho = float(_download_arr(lib, scratch.grad_poq_rho_partial, (n,), np.float64).sum())
        g_pop_phi = float(_download_arr(lib, scratch.grad_pop_phi_partial, (n,), np.float64).sum())
        
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
            w_host = _download_arr(lib, dh._weight_ptr, (n,), np.float64)
            Q_val = float(np.sum(w_host * P_arr))
        
        return Q_val, grads, P_arr
    
    def free(self):
        """Free all GPU memory."""
        # Free per-handle data
        for h in self._handles:
            h.free()
        self._handles.clear()
        
        # Free constant/index arrays
        ptrs = [
            self.d_bw_m0_index, self.d_bw_mass_index, self.d_bw_order,
            self.d_m0_index, self.d_bw_pos_gamma_idx, self.d_bw_pos_gamma_off,
            self.d_fl_type, self.d_fl_q_index, self.d_fl_order,
            self.d_angle_index, self.d_g0_index, self.d_g0_mass_index,
            self.d_matrix_gamma, self.d_matrix_angle_real, self.d_matrix_angle_imag,
            self.d_gamma_table_real, self.d_gamma_table_imag,
            self.d_fl_table, self.d_angle_k, self.d_angle_b,
        ]
        for p in ptrs:
            if p is not None:
                self.lib.cuda_free(p)
