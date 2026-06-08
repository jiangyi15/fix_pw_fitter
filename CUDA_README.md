# CUDA Kernel Implementation

## Overview

This CUDA implementation provides GPU acceleration for amplitude analysis with:

1. **Persistent GPU Data** - Load data once, compute many times
2. **Correct Wirtinger Calculus** - Same gradient corrections as NumPy version
3. **NumPy-Compatible API** - Easy to use and maintain
4. **Automatic Memory Management** - No manual memory allocation

## Architecture

```
GPUData (Persistent)
├── Config arrays (loaded once)
│   ├── Indices (m0_index, g0_index, etc.)
│   ├── Tables (gamma_table, fl_table)
│   └── Matrices (matrix_angle, matrix_gamma)
│
└── Data arrays (loaded once via load_data())
    ├── mass, momentum, angle
    ├── frac, time, weight
    └── bkg

Parameters (transferred each compute())
├── ck (complex)
├── m0, g0 (real)
└── scalar (6 values)

Outputs (transferred back each compute())
├── Q (scalar)
├── grads (dict)
└── P (array)
```

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit (11.x or 12.x)
- Python 3.7+

### Install CuPy

```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x

# Verify installation
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

## Usage

### Basic Usage

```python
from cuda_kernel_cupy import CUDAKernel
from config_loader import Config

# Load config
config = Config("config_angle.yml")
kernel_config = config.build_all_index()

# Create CUDA kernel
kernel = CUDAKernel(kernel_config)

# Load data to GPU ONCE
kernel.load_data(data)

# Compute with different parameters
for iteration in range(1000):
    # Update parameters
    params["m0"] += learning_rate * grads["m0"]

    # Compute - data stays on GPU
    Q, grads, P = kernel.compute(params, norm=None)

# Free GPU memory
kernel.free_data()
```

### Comparison with NumPy

```python
# NumPy kernel - transfers data every time
numpy_kernel = NumpyKernelCorrect(config)
for iteration in range(1000):
    Q, grads, P = numpy_kernel._compute(params, data, norm=None)
    # Data transferred CPU → CPU every iteration

# CUDA kernel - data stays on GPU
cuda_kernel = CUDAKernel(config)
cuda_kernel.load_data(data)  # Transfer once
for iteration in range(1000):
    Q, grads, P = cuda_kernel.compute(params, norm=None)
    # Only parameters transferred, data stays on GPU
```

## Performance

Expected speedup depends on data size:

| Events | NumPy (ms) | CUDA (ms) | Speedup |
|--------|------------|-----------|---------|
| 100    | ~240       | ~50       | ~5x     |
| 1000   | ~1700      | ~150      | ~11x    |
| 5000   | ~8000      | ~500      | ~16x    |

**Benefits:**
- Persistent GPU data eliminates repeated transfers
- Parallel computation across thousands of events
- GPU memory bandwidth much higher than CPU

**Overhead:**
- Initial data transfer to GPU (one-time cost)
- Small overhead for transferring parameters each call
- GPU memory allocation

## Implementation Details

### Wirtinger Calculus

Same correct gradient computation as NumPy version:

```python
# For complex variables with real-valued loss Q:
∂Q/∂z = (∂Q/∂Re(z) - i·∂Q/∂Im(z))/2

# For real parameters affecting complex variables:
∂Q/∂x = 2·Re(∂Q/∂z · ∂z/∂x)

# For complex parameters:
∂Q/∂Re(ck) = 2·Re(∂Q/∂ck)
∂Q/∂Im(ck) = -2·Im(∂Q/∂ck)
```

### CuPy Implementation

Uses CuPy for NumPy-compatible GPU arrays:

- **Advantages**:
  - Drop-in replacement for NumPy
  - Automatic GPU memory management
  - Optimized CUDA kernels under the hood
  - Easy to maintain

- **Key Operations**:
  - `cp.asarray()` - Transfer to GPU
  - `array.get()` - Transfer to CPU
  - NumPy operations run on GPU automatically

### Memory Management

```python
# Config data (allocated in __init__, stays on GPU)
self.m0_index_gpu = cp.array(config["m0_index"])

# Input data (allocated in load_data(), stays on GPU)
self.mass_gpu = cp.asarray(data["mass"])

# Parameters (transferred each compute(), temporary)
ck = cp.asarray(params["ck"])

# Results (transferred back each compute())
Q_cpu = float(Q.get())
grads_cpu = {"ck": grad_ck.get(), ...}
```

## Testing

### Correctness Test

```bash
python test_cuda_kernel.py
```

Verifies:
- Forward pass matches NumPy exactly
- Gradients match NumPy exactly
- Numerical gradient verification

### Performance Benchmark

```bash
python cuda_kernel_cupy.py
```

Benchmarks:
- Multiple event sizes (100, 500, 1000, 5000)
- Comparison with NumPy kernel
- Throughput measurement

## Files

- `cuda_kernel_cupy.py` - Main CUDA kernel implementation
- `test_cuda_kernel.py` - Comprehensive tests (correctness, performance, gradients)
- `cuda_kernels.cu` - Raw CUDA kernels (reference, not used in CuPy version)
- `cuda_kernel.py` - PyCUDA version (alternative implementation)

## Troubleshooting

### CuPy Import Error

```
ImportError: No module named 'cupy'
```

**Solution**: Install CuPy matching your CUDA version:
```bash
pip install cupy-cuda11x  # Check with: nvcc --version
```

### CUDA Out of Memory

```
cupy.cuda.memory.OutOfMemoryError
```

**Solutions**:
1. Reduce batch size
2. Use smaller dataset
3. Free GPU memory: `kernel.free_data()`

### Slow Performance

**Possible causes**:
1. Data transfer overhead - ensure data stays on GPU
2. Small dataset - GPU overhead dominates
3. Not warming up GPU - first call slower

**Solutions**:
1. Use persistent GPU data (already implemented)
2. Use larger datasets (1000+ events)
3. Run warm-up iteration before benchmarking

## Future Improvements

1. **Custom CUDA kernels** - For specific operations not optimized in CuPy
2. **Multi-GPU support** - Split data across multiple GPUs
3. **Mixed precision** - Use float16 for intermediate computations
4. **Stream processing** - Overlap computation and data transfer

## References

- [CuPy Documentation](https://docs.cupy.dev/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Wirtinger Calculus](https://en.wikipedia.org/wiki/Wirtinger_derivatives)
