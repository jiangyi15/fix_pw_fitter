"""
CUDA kernel with persistent GPU memory using CFFI.

This provides direct CUDA access without CuPy or PyCUDA dependencies.

Architecture:
1. CUDA kernels compiled to shared library (libcuda_kernels.so)
2. CFFI bindings provide Python interface
3. GPUData class manages persistent GPU memory
4. CUDAKernel provides high-level Python API

Usage:
    # Build the CUDA library first
    python build_cuda.py

    # Then use from Python
    kernel = CUDAKernel(config)
    kernel.load_data(data)  # Load once
    Q, grads, P = kernel.compute(params)  # Compute many times
"""

import numpy as np
from cffi import FFI
import os
import sys

# Initialize FFI
ffi = FFI()

# Define C interface
CDEF = """
/* Memory management */
int cuda_alloc(void** ptr, unsigned long size);
int cuda_free(void* ptr);
int cuda_memcpy_to_device(void* dst, const void* src, unsigned long size);
int cuda_memcpy_to_host(void* dst, const void* src, unsigned long size);

/* Device info */
int cuda_get_device_count();
int cuda_get_device_name(char* name, int len);

/* Kernel launch */
void launch_forward_kernel(
    const double* mass,
    const double* momentum,
    const double* angle,
    const double* frac,
    const double* time,
    const double* weight,
    const double* bkg,
    const int* m0_index,
    const int* g0_index,
    const int* fl_type,
    const int* mass_index,
    const int* g0_mass_index,
    const int* fl_q_index,
    const int* bw_order,
    const int* fl_order,
    const int* angle_index,
    const double* angle_k,
    const double* angle_b,
    const double* matrix_angle,
    const double* matrix_gamma,
    const double* gamma_table,
    const double* fl_table,
    double gamma_min,
    double gamma_delta,
    double fl_min,
    double fl_delta,
    int n_wave,
    int n_res,
    int n_decay,
    int n_unique_bw,
    int n_gamma_rows,
    int n_mass,
    int n_momentum,
    int n_angle_total,
    const double* ck_real,
    const double* ck_imag,
    const double* m0,
    const double* g0,
    double Gamma,
    double Delta_Gamma,
    double Delta_m,
    double A_p,
    double poq_rho,
    double pop_phi,
    double* Q_out,
    double* P_out,
    double* pap_real,
    double* pap_imag,
    double* pam_real,
    double* pam_imag,
    double* gp_real,
    double* gp_imag,
    double* gm_real,
    double* gm_imag,
    double* poq_real,
    double* poq_imag,
    double* bw_p_real,
    double* bw_p_imag,
    double* common_amp_factor_real,
    double* common_amp_factor_imag,
    int n_events
);
"""

ffi.cdef(CDEF)


class CUDALibrary:
    """
    Loads and manages the CUDA shared library via CFFI.
    """

    def __init__(self, lib_path="libcuda_kernels.so"):
        """Load CUDA library"""
        if not os.path.exists(lib_path):
            raise RuntimeError(
                f"CUDA library not found: {lib_path}\n"
                "Please build it first:\n"
                "  python build_cuda.py"
            )

        try:
            self.lib = ffi.dlopen(lib_path)
            print(f"✓ Loaded CUDA library: {lib_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load CUDA library: {e}")

    def alloc(self, size):
        """Allocate GPU memory"""
        ptr = ffi.new("void**")
        err = self.lib.cuda_alloc(ptr, size)
        if err != 0:
            raise RuntimeError(f"CUDA allocation failed with error code {err}")
        return ptr[0]

    def free(self, ptr):
        """Free GPU memory"""
        err = self.lib.cuda_free(ptr)
        if err != 0:
            raise RuntimeError(f"CUDA free failed with error code {err}")

    def copy_to_device(self, dst, src, size):
        """Copy data to GPU"""
        err = self.lib.cuda_memcpy_to_device(dst, src, size)
        if err != 0:
            raise RuntimeError(f"CUDA memcpy to device failed with error code {err}")

    def copy_to_host(self, dst, src, size):
        """Copy data from GPU"""
        err = self.lib.cuda_memcpy_to_host(dst, src, size)
        if err != 0:
            raise RuntimeError(f"CUDA memcpy to host failed with error code {err}")

    def get_device_count(self):
        """Get number of CUDA devices"""
        return self.lib.cuda_get_device_count()

    def get_device_name(self):
        """Get CUDA device name"""
        name = ffi.new("char[256]")
        self.lib.cuda_get_device_name(name, 256)
        return ffi.string(name).decode('utf-8')


class GPUArray:
    """
    Manages a GPU array with automatic memory management.
    """

    def __init__(self, lib, shape, dtype=np.float64):
        """
        Allocate GPU array.

        Args:
            lib: CUDALibrary instance
            shape: array shape
            dtype: numpy dtype
        """
        self.lib = lib
        self.shape = shape
        self.dtype = dtype
        self.nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        self.ptr = lib.alloc(self.nbytes)

    def set(self, data):
        """Copy data to GPU"""
        if data.shape != self.shape:
            raise ValueError(f"Shape mismatch: {data.shape} vs {self.shape}")

        # Ensure contiguous array
        data = np.ascontiguousarray(data, dtype=self.dtype)

        # Get pointer to data
        ptr = ffi.cast("void*", data.ctypes.data)

        # Copy to GPU
        self.lib.copy_to_device(self.ptr, ptr, self.nbytes)

    def get(self):
        """Copy data from GPU"""
        data = np.empty(self.shape, dtype=self.dtype)
        ptr = ffi.cast("void*", data.ctypes.data)
        self.lib.copy_to_host(ptr, self.ptr, self.nbytes)
        return data

    def free(self):
        """Free GPU memory"""
        if self.ptr is not None:
            self.lib.free(self.ptr)
            self.ptr = None


class GPUData:
    """
    Manages persistent GPU data for amplitude analysis.

    Data is loaded once and stays on GPU for multiple computations.
    """

    def __init__(self, config, lib):
        """Initialize GPU memory"""
        self.config = config
        self.lib = lib

        # Allocate config arrays on GPU
        self._alloc_config_gpu()

        # Data arrays (allocated when load_data is called)
        self.mass_gpu = None
        self.momentum_gpu = None
        self.angle_gpu = None
        self.frac_gpu = None
        self.time_gpu = None
        self.weight_gpu = None
        self.bkg_gpu = None

        self.n_events = 0
        self.data_loaded = False

    def _alloc_config_gpu(self):
        """Allocate config arrays on GPU (persistent)"""
        # Index arrays
        self.m0_index_gpu = GPUArray(self.lib, (len(self.config["m0_index"]),), np.int32)
        self.m0_index_gpu.set(self.config["m0_index"])

        self.g0_index_gpu = GPUArray(self.lib, (len(self.config["g0_index"]),), np.int32)
        self.g0_index_gpu.set(self.config["g0_index"])

        self.fl_type_gpu = GPUArray(self.lib, (len(self.config["fl_type"]),), np.int32)
        self.fl_type_gpu.set(self.config["fl_type"])

        # ... allocate other config arrays ...

    def load_data(self, data):
        """Load data to GPU (call once)"""
        self.n_events = data["mass"].shape[0]

        # Allocate and copy data arrays
        self.mass_gpu = GPUArray(self.lib, data["mass"].shape, np.float64)
        self.mass_gpu.set(data["mass"])

        self.momentum_gpu = GPUArray(self.lib, data["q"].shape, np.float64)
        self.momentum_gpu.set(data["q"])

        # ... allocate other data arrays ...

        self.data_loaded = True
        print(f"✓ Loaded {self.n_events} events to GPU")

    def compute(self, params, norm=None):
        """Compute with current parameters"""
        if not self.data_loaded:
            raise RuntimeError("Must call load_data() before compute()")

        # For now, fall back to NumPy
        # Full CUDA implementation requires complete kernel code
        raise NotImplementedError(
            "Full CUDA kernel implementation required.\n"
            "Current version uses NumPy backend with CFFI interface structure."
        )

    def free(self):
        """Free all GPU memory"""
        # Free config arrays
        self.m0_index_gpu.free()
        self.g0_index_gpu.free()
        self.fl_type_gpu.free()

        # Free data arrays
        if self.mass_gpu:
            self.mass_gpu.free()
        if self.momentum_gpu:
            self.momentum_gpu.free()
        # ... free other arrays ...

        self.data_loaded = False


class CUDAKernel:
    """
    High-level CUDA kernel interface with persistent GPU data.

    Usage:
        kernel = CUDAKernel(config)
        kernel.load_data(data)  # Load once
        Q, grads, P = kernel.compute(params, norm=None)  # Compute many times
    """

    def __init__(self, config):
        """Initialize CUDA kernel"""
        try:
            self.lib = CUDALibrary()
        except RuntimeError as e:
            print(f"Warning: {e}")
            print("Falling back to NumPy kernel")
            self.cuda_available = False
            from numpy_kernel import NumpyKernelCorrect
            self.numpy_kernel = NumpyKernelCorrect(config)
        else:
            self.cuda_available = True
            self.gpu_data = GPUData(config, self.lib)

            # Get device info
            n_devices = self.lib.get_device_count()
            if n_devices > 0:
                device_name = self.lib.get_device_name()
                print(f"Using GPU: {device_name}")

        self.config = config
        self.data = None

    def load_data(self, data):
        """Load data to GPU"""
        self.data = data

        if self.cuda_available:
            self.gpu_data.load_data(data)
        else:
            print("CUDA not available, data will be used with NumPy kernel")

    def compute(self, params, norm=None):
        """Compute with current parameters"""
        if self.cuda_available:
            return self.gpu_data.compute(params, norm)
        else:
            return self.numpy_kernel._compute(params, self.data, norm)

    def free_data(self):
        """Free GPU data"""
        if self.cuda_available:
            self.gpu_data.free()
        self.data = None


if __name__ == "__main__":
    print("="*70)
    print("CUDA KERNEL TEST (CFFI)")
    print("="*70)

    # Check if library exists
    if not os.path.exists("libcuda_kernels.so"):
        print("\nCUDA library not found. Building...")
        import subprocess
        subprocess.run([sys.executable, "build_cuda.py"])

    # Test CUDA library loading
    try:
        lib = CUDALibrary()
        print(f"✓ CUDA library loaded")

        n_devices = lib.get_device_count()
        print(f"✓ Found {n_devices} CUDA device(s)")

        if n_devices > 0:
            device_name = lib.get_device_name()
            print(f"✓ Device: {device_name}")

        # Test memory allocation
        arr = GPUArray(lib, (1000,), np.float64)
        print(f"✓ Allocated GPU array: 1000 elements")

        # Test data transfer
        data = np.random.random(1000)
        arr.set(data)
        result = arr.get()
        print(f"✓ Data transfer test: max error = {np.max(np.abs(data - result)):.2e}")

        arr.free()
        print(f"✓ GPU memory freed")

        print("\n" + "="*70)
        print("✓ CFFI CUDA interface working!")
        print("="*70)
        print("\nNote: Full kernel implementation requires completing CUDA kernel code.")
        print("Current version demonstrates memory management and CFFI bindings.")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure:")
        print("1. CUDA Toolkit is installed")
        print("2. nvcc is in PATH")
        print("3. Run: python build_cuda.py")
