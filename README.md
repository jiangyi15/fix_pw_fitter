# ampfit — Amplitude Analysis Fitting Framework

A Python package for partial wave amplitude analysis with NumPy and CUDA backends,
full Wirtinger-calculus gradients, parameter constraints, and a global Fitter class.

## Package Structure

```
├── pyproject.toml
├── config_angle.yml              # Example configuration
├── src/ampfit/
│   ├── __init__.py               # Exports: Config, Fitter, NumpyKernel, CUDAKernel, ParameterConstraint
│   ├── config_loader.py          # YAML config → index arrays for kernels
│   ├── particle_model.py         # Particle property definitions
│   ├── angular_formula.py        # Angular distribution formulas
│   ├── numpy_kernel.py           # NumPy reference kernel (correct Wirtinger gradients)
│   ├── param_constraint.py       # Parameter constraints: fixed, same, scale
│   ├── fitter.py                 # Global Fitter: config → NLL with norm
│   ├── _cuda.py                  # CUDA Python bindings (CFFI)
│   └── cuda/
│       ├── kernels.cu            # CUDA C kernels (optimized)
│       └── build.py              # Build script for CUDA .so
├── tests/
│   ├── test_cuda_kernel.py
│   └── ...
└── archive/                      # Historical files
```

## Installation

```bash
pip install -e .                  # Install in development mode
python -m ampfit.cuda.build       # Build CUDA library (requires nvcc)
```

## Usage

### Low-level: Direct kernel usage

```python
from ampfit import Config, NumpyKernel, CUDAKernel

config = Config("config_angle.yml")
kernel_config = config.build_all_index()

# NumPy
nk = NumpyKernel(kernel_config)
Q, grads, P = nk._compute(params, data)

# CUDA
ck = CUDAKernel(kernel_config)
dh = ck.load_data(data)
Q, grads, P = ck.compute(params, dh)
```

### High-level: Fitter with norm and constraints

```python
from ampfit import Fitter

fitter = Fitter("config_angle.yml")
fitter.set_data(data)
fitter.set_phsp(phsp)
fitter.set_default_params(m0=m0_arr, g0=g0_arr, scalar=[...])

# Parameter constraints (optional)
fitter.set_fixed({"name": 1+0j, ...})
fitter.set_same([["a", "b"], ...])

# Optimizer-ready NLL
x = fitter.initial_values()
nll, grad_x = fitter.get_nll(x)  # handles norm internally
```

## Performance

| Events | NumPy (ms) | CUDA (ms) | Speedup |
|--------|-----------|-----------|---------|
| 100    | 21        | 1.4       | 14×     |
| 1000   | 170       | 7.3       | 23×     |
| 10000  | 1500      | 53        | 28×     |
| 100000 | —         | 543       | 185K ev/s |

All gradients verified to machine precision (< 1e-12) using 3-point numerical method.

## Requirements

- Python ≥ 3.10
- numpy, pyyaml, cffi
- CUDA Toolkit (for GPU acceleration)
