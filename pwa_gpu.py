"""
PWA GPU - CUDA accelerated partial wave analysis

Architecture:
  PWAConfig - holds indices, tables, matrixes (reused across datasets)
  PWAData   - holds per-event arrays on GPU (one per dataset)
  PWAGPU    - high-level fitter that combines a config with a data object

Usage:
  fitter = PWAGPU(config)
  data1 = fitter.load_data(data_tuple1)
  data2 = fitter.load_data(data_tuple2)

  fitter.compute(params, data1, N)
  fitter.grad(params, data2, N)
"""

import numpy as np
from cffi import FFI
import os
import subprocess
import time

ffi = FFI()

# Define C interface - matches pwa_gpu_kernels.cu
ffi.cdef("""
    typedef struct { double x; double y; } cuDoubleComplex;

    void* pwa_create_config(
        const int* bw_index, const int* gamma_index, const int* bw_order,
        const int* bf_index, const int* bf_order, const int* ang_index,
        const double* ang_k, const double* ang_b,
        const double* matrix_gamma, const cuDoubleComplex* matrix_ang,
        const cuDoubleComplex* gamma_table, const double* bf_table,
        int n_waves, int n_m0, int n_g0,
        int n_res_per_wave, int n_decays_per_wave,
        int n_bf_types, int n_basis, int n_ang_per_basis,
        int n_gamma_points, int n_bf_points
    );
    void pwa_destroy_config(void* cfg);

    void* pwa_create_data(
        const double* mass_flat, const double* q_flat, const double* angles_flat,
        const double* time_arr, const double* frac_arr,
        int n_events, int mass_stride, int q_stride, int ang_stride
    );
    void pwa_destroy_data(void* data);

    void pwa_compute(
        void* cfg, void* data,
        double* p_out, cuDoubleComplex* amp_p_out, cuDoubleComplex* amp_m_out,
        double* grad_ck_re, double* grad_ck_im,
        double* grad_m0_out, double* grad_g0_out,
        double* grad_scalar_out,
        const cuDoubleComplex* ck, const double* m0, const double* g0,
        double delta_m, double delta_g, double g_val,
        double ap, double lam, double phi,
        double N_val,
        const double* weights, const double* bkg_arr,
        int n_waves, int n_m0, int n_g0,
        int n_res_per_wave, int n_decays_per_wave,
        int n_bf_types, int n_basis, int n_ang_per_basis,
        int n_gamma_points, int n_bf_points,
        double g_min, double g_delta, double q_min, double q_delta
    );
""")

# Compile CUDA code
def compile_cuda():
    """Compile CUDA code to shared library"""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    cu_file = os.path.join(src_dir, 'pwa_gpu_kernels.cu')
    so_file = os.path.join(src_dir, 'libpwa_gpu.so')

    if os.path.exists(so_file):
        if os.path.getmtime(cu_file) <= os.path.getmtime(so_file):
            return so_file

    cmd = [
        'nvcc', '-shared', '-Xcompiler', '-fPIC',
        '-o', so_file, cu_file,
        '-lcudart', '--ptxas-options=-v'
    ]

    print("Compiling CUDA code...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}")
        raise RuntimeError("CUDA compilation failed")

    print(f"Compiled to {so_file}")
    return so_file


class PWAData:
    """
    Per-event data on GPU.

    Usage:
        data = PWAData(fitter, mass, q, angles, time, frac, weights, bkg)
        fitter.compute(params, data, N)
    """

    def __init__(self, fitter, mass, q, angles, time_arr, frac, weights, bkg):
        """
        Upload event data to GPU.

        Args:
            fitter: PWAGPU instance (provides lib for GPU ops)
            mass:     (n_events, n_topo, n_res) or (n_events, n_topo*n_res)
            q:        (n_events, n_topo, n_decays) or (n_events, n_topo*n_decays)
            angles:   (n_events, n_topo, n_ang) or (n_events, n_topo*n_ang)
            time_arr: (n_events,)
            frac:     (n_events,)
            weights:  (n_events,)
            bkg:      (n_events,) or scalar
        """
        self.lib = fitter.lib
        n_events = mass.shape[0]

        # Flatten arrays
        mass_flat = np.ascontiguousarray(mass.reshape(n_events, -1), dtype=np.float64)
        q_flat = np.ascontiguousarray(q.reshape(n_events, -1), dtype=np.float64)
        angles_flat = np.ascontiguousarray(angles.reshape(n_events, -1), dtype=np.float64)
        time_arr = np.ascontiguousarray(time_arr, dtype=np.float64)
        frac_arr = np.ascontiguousarray(frac, dtype=np.float64)

        self.n_events = n_events
        self.mass_stride = mass_flat.shape[1]
        self.q_stride = q_flat.shape[1]
        self.ang_stride = angles_flat.shape[1]

        # Store CPU-side arrays
        self.weights = np.ascontiguousarray(weights, dtype=np.float64)
        if isinstance(bkg, np.ndarray):
            self.bkg = np.ascontiguousarray(bkg, dtype=np.float64)
        else:
            self.bkg = float(bkg)

        # Upload to GPU
        self._ptr = self.lib.pwa_create_data(
            ffi.cast("double*", mass_flat.ctypes.data),
            ffi.cast("double*", q_flat.ctypes.data),
            ffi.cast("double*", angles_flat.ctypes.data),
            ffi.cast("double*", time_arr.ctypes.data),
            ffi.cast("double*", frac_arr.ctypes.data),
            n_events, self.mass_stride, self.q_stride, self.ang_stride
        )
        if not self._ptr:
            raise RuntimeError("pwa_create_data failed")

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr is not None:
            try:
                self.lib.pwa_destroy_data(self._ptr)
            except Exception:
                pass
            self._ptr = None


class PWAGPU:
    """
    GPU-accelerated PWA fitter.

    Holds configuration (indices, tables, matrixes) on GPU.
    Data is passed in via PWAData objects for each compute call.
    Forward pass and gradients are always computed together in one GPU call.

    Usage:
        fitter = PWAGPU(config)
        data1 = PWAData(fitter, mass1, q1, angles1, time1, frac1, w1, b1)
        data2 = PWAData(fitter, mass2, q2, angles2, time2, frac2, w2, b2)

        p, q, grads = fitter.compute(params, data1, N)
    """

    def __init__(self, config, device_id=0):
        """Initialize GPU and create config (indices, tables on GPU)."""
        so_path = compile_cuda()
        self.lib = ffi.dlopen(so_path)
        self.config = config

        # Compute dimensions from config
        n_waves_from_br = len(config.get('bw_order', [])) // 2
        if n_waves_from_br < 2:
            n_waves_from_br = 2
        self.n_waves = n_waves_from_br
        self.n_m0 = len(config['bw_index'])
        self.n_g0 = len(config['gamma_index'])
        self.n_res_per_wave = len(config['bw_order']) // n_waves_from_br
        self.n_decays_per_wave = len(config['bf_order']) // n_waves_from_br
        self.n_bf_types = len(config['bf_index'])
        self.n_basis = config['ang_index'].shape[0]
        self.n_ang_per_basis = config['ang_index'].shape[1]
        self.n_gamma_points = config['gamma_table'].shape[1]
        self.n_bf_points = config['bf_table'].shape[1]
        self.g_min = config['g_min']
        self.g_delta = config['g_delta']
        self.q_min = config['q_min']
        self.q_delta = config['q_delta']

        # Prepare config arrays
        bw_index = np.ascontiguousarray(config['bw_index'], dtype=np.int32)
        gamma_index = np.ascontiguousarray(config['gamma_index'], dtype=np.int32)
        bw_order = np.ascontiguousarray(config['bw_order'], dtype=np.int32)
        bf_index = np.ascontiguousarray(config['bf_index'], dtype=np.int32)
        bf_order = np.ascontiguousarray(config['bf_order'], dtype=np.int32)
        ang_index = np.ascontiguousarray(config['ang_index'].flatten(), dtype=np.int32)
        ang_k = np.ascontiguousarray(config['ang_k'].flatten())
        ang_b = np.ascontiguousarray(config['ang_b'].flatten())
        matrix_gamma = np.ascontiguousarray(config['matrix_gamma'])

        # Complex matrix_ang
        matrix_ang = config['matrix_ang']
        matrix_ang_c = np.zeros(len(matrix_ang.flat), dtype=np.complex128)
        matrix_ang_c.real = matrix_ang.real.flatten()
        matrix_ang_c.imag = matrix_ang.imag.flatten()
        matrix_ang_c = np.ascontiguousarray(matrix_ang_c)

        # gamma_table - handle broadcasting
        gamma_table = np.ascontiguousarray(config['gamma_table'])
        if gamma_table.ndim == 2 and gamma_table.shape[0] == 1 and self.n_g0 > 1:
            gamma_table = np.repeat(gamma_table, self.n_g0, axis=0)
            gamma_table = np.ascontiguousarray(gamma_table)
        if gamma_table.dtype != np.complex128:
            gamma_table = gamma_table.astype(np.complex128)
        gamma_table = np.ascontiguousarray(gamma_table.reshape(-1))
        self._gamma_table = gamma_table  # prevent GC

        # bf_table - handle broadcasting
        bf_table = np.ascontiguousarray(config['bf_table'])
        if bf_table.ndim == 2 and bf_table.shape[0] == 1 and self.n_bf_types > 1:
            bf_table = np.repeat(bf_table, self.n_bf_types, axis=0)
            bf_table = np.ascontiguousarray(bf_table)
        bf_table = bf_table.reshape(-1)
        bf_table = np.ascontiguousarray(bf_table)

        # Create CUDA config (indices, tables, scratch on GPU)
        self._cfg = self.lib.pwa_create_config(
            ffi.cast("int*", bw_index.ctypes.data),
            ffi.cast("int*", gamma_index.ctypes.data),
            ffi.cast("int*", bw_order.ctypes.data),
            ffi.cast("int*", bf_index.ctypes.data),
            ffi.cast("int*", bf_order.ctypes.data),
            ffi.cast("int*", ang_index.ctypes.data),
            ffi.cast("double*", ang_k.ctypes.data),
            ffi.cast("double*", ang_b.ctypes.data),
            ffi.cast("double*", matrix_gamma.ctypes.data),
            ffi.cast("cuDoubleComplex*", matrix_ang_c.ctypes.data),
            ffi.cast("cuDoubleComplex*", gamma_table.ctypes.data),
            ffi.cast("double*", bf_table.ctypes.data),
            self.n_waves, self.n_m0, self.n_g0,
            self.n_res_per_wave, self.n_decays_per_wave,
            self.n_bf_types, self.n_basis, self.n_ang_per_basis,
            self.n_gamma_points, self.n_bf_points
        )
        if not self._cfg:
            raise RuntimeError("pwa_create_config failed")

        print("PWA GPU initialized")

    def load_data(self, data):
        """
        Convenience: create a PWAData from a data tuple.

        Args:
            data: tuple of (mass, q, angles, time, frac, weights, bkg)

        Returns:
            PWAData object
        """
        return PWAData(self, *data)

    @property
    def last_p(self):
        """Per-event probabilities from the most recent compute() call."""
        return getattr(self, '_last_p', None)

    def compute(self, params, data, N=None):
        """
        Compute q_val + gradients in one GPU call.

        Per-event probabilities are cached in ``self.last_p``.

        Args:
            params: (ck, m0, g0, delta_m, delta_g, g, ap, lam, phi)
            data: PWAData object
            N: normalization (None for chi-square mode, gradients always computed)

        Returns:
            q_val: likelihood scalar
            grads: dict with keys:
                'ck'      — gradient w.r.t. complex couplings (n_waves,)
                'm0'      — gradient w.r.t. resonance masses (n_m0,)
                'g0'      — gradient w.r.t. widths (n_g0,)
                'N'       — gradient w.r.t. norm, or None for chi-square
                'delta_m' — gradient w.r.t. mass diff
                'delta_g' — gradient w.r.t. width diff
                'g'       — gradient w.r.t. avg width
                'ap'      — gradient w.r.t. CP asymmetry
                'lam'     — gradient w.r.t. |lambda|
                'phi'     — gradient w.r.t. CP phase
        """
        ck, m0, g0, delta_m, delta_g, g, ap, lam, phi = params

        ck_c = np.ascontiguousarray(ck.astype(np.complex128))
        m0 = np.ascontiguousarray(m0)
        g0 = np.ascontiguousarray(g0)

        if isinstance(data.bkg, np.ndarray):
            bkg_arr = np.ascontiguousarray(data.bkg, dtype=np.float64)
        else:
            bkg_arr = np.full(data.n_events, data.bkg, dtype=np.float64)

        p_out = np.zeros(data.n_events, dtype=np.float64)
        amp_p_out = np.zeros(data.n_events, dtype=np.complex128)
        amp_m_out = np.zeros(data.n_events, dtype=np.complex128)
        grad_ck_re = np.zeros(self.n_waves, dtype=np.float64)
        grad_ck_im = np.zeros(self.n_waves, dtype=np.float64)
        grad_m0 = np.zeros(self.n_m0, dtype=np.float64)
        grad_g0 = np.zeros(self.n_g0, dtype=np.float64)
        grad_scalar = np.zeros(7, dtype=np.float64)

        self.lib.pwa_compute(
            self._cfg, data._ptr,
            ffi.cast("double*", p_out.ctypes.data),
            ffi.cast("cuDoubleComplex*", amp_p_out.ctypes.data),
            ffi.cast("cuDoubleComplex*", amp_m_out.ctypes.data),
            ffi.cast("double*", grad_ck_re.ctypes.data),
            ffi.cast("double*", grad_ck_im.ctypes.data),
            ffi.cast("double*", grad_m0.ctypes.data),
            ffi.cast("double*", grad_g0.ctypes.data),
            ffi.cast("double*", grad_scalar.ctypes.data),
            ffi.cast("cuDoubleComplex*", ck_c.ctypes.data),
            ffi.cast("double*", m0.ctypes.data),
            ffi.cast("double*", g0.ctypes.data),
            delta_m, delta_g, g, ap, lam, phi,
            N if N is not None else -1.0,
            ffi.cast("double*", data.weights.ctypes.data),
            ffi.cast("double*", bkg_arr.ctypes.data),
            self.n_waves, self.n_m0, self.n_g0,
            self.n_res_per_wave, self.n_decays_per_wave,
            self.n_bf_types, self.n_basis, self.n_ang_per_basis,
            self.n_gamma_points, self.n_bf_points,
            self.g_min, self.g_delta, self.q_min, self.q_delta
        )

        if N is not None and N > 0:
            bkg_vals = data.bkg if isinstance(data.bkg, np.ndarray) else np.full(data.n_events, data.bkg)
            q = p_out / N + bkg_vals
            q_val = np.sum(data.weights * np.log(q))
        else:
            q_val = np.sum(data.weights * p_out)

        grad_ck = grad_ck_re + 1j * grad_ck_im

        grads = {
            'ck':      grad_ck,
            'm0':      grad_m0,
            'g0':      grad_g0,
            'N':       grad_scalar[6] if N is not None else None,
            'delta_m': grad_scalar[0],
            'delta_g': grad_scalar[1],
            'g':       grad_scalar[2],
            'ap':      grad_scalar[3],
            'lam':     grad_scalar[4],
            'phi':     grad_scalar[5],
        }

        self._last_p = p_out
        return q_val, grads

    def __del__(self):
        """Cleanup GPU config"""
        if hasattr(self, '_cfg') and self._cfg is not None:
            try:
                self.lib.pwa_destroy_config(self._cfg)
            except Exception:
                pass
            self._cfg = None


# ============================================================
# Convenience function to compare with numpy
# ============================================================

def compare_with_numpy(fitter_gpu, fitter_numpy, params, data, N=None):
    """Compare GPU and numpy results"""
    data_gpu = fitter_gpu.load_data(data)
    t0 = time.time()
    q_gpu, _ = fitter_gpu.compute(params, data_gpu, N)
    t_gpu = time.time() - t0

    t0 = time.time()
    q_numpy, grad_numpy = fitter_numpy.compute(params, data, N)
    t_numpy = time.time() - t0

    max_diff = np.max(np.abs(fitter_gpu.last_p - fitter_numpy._last_p))

    print(f"GPU time: {t_gpu*1000:.2f} ms")
    print(f"Numpy time: {t_numpy*1000:.2f} ms")
    print(f"Speedup: {t_numpy/t_gpu:.1f}x")
    print(f"Max diff: {max_diff:.2e}")

    return max_diff


if __name__ == "__main__":
    print("PWA GPU module loaded successfully")
