# ampfit — Amplitude Analysis Fitting Framework

A Python package for partial wave amplitude analysis with NumPy and CUDA backends,
full Wirtinger-calculus gradients, parameter constraints, and a global Fitter class.

## Package Structure

```
├── pyproject.toml
├── fit.sh                          # One-command fit pipeline
├── run_fit.py                      # Full fit script with archive-compatible constraints
├── config_angle.yml                # Example configuration
├── src/ampfit/
│   ├── __init__.py                 # Exports: Config, Fitter, NumpyKernel, CUDAKernel, ...
│   ├── config_loader.py            # YAML config → index arrays for kernels
│   ├── particle_model.py           # Particle property definitions + get_gamma_defaults()
│   ├── angular_formula.py          # Angular distribution formulas
│   ├── numpy_kernel.py             # NumPy reference kernel (Wirtinger gradients)
│   ├── param_constraint.py         # Parameter constraints: fixed, same, scale
│   ├── boundary.py                 # BoundTransform (arctan-based, bijective)
│   ├── fitter.py                   # Fitter: NLL → BFGS fit → save_params → plot
│   ├── _cuda.py                    # CUDA Python bindings (CFFI) + GPUDataBuffer
│   └── cuda/
│       ├── kernels.cu              # Optimized CUDA C kernels (forward + backward)
│       └── build.py                # Auto-build on first import
├── tests/
│   ├── test_cuda_kernel.py
│   └── ...
└── archive/                        # Historical files (TensorFlow reference)
```

## Installation

```bash
pip install -e .                  # Install in development mode
# CUDA builds automatically on first use (requires nvcc)
```

## Usage

### One-command fit (shell script)

```bash
./fit.sh                                    # Full fit with defaults
./fit.sh config_angle.yml data.npz phsp.npz results/
```

### High-level: Fitter with constraints

```bash
python run_fit.py                            # Compute NLL
python run_fit.py --fit --maxiter 200 --plot results/
python run_fit.py --init results.json --fit --maxiter 500  # Restart
python run_fit.py --fix-mass-width --fit     # Fix masses/widths
```

### Python API

```python
from ampfit import Fitter

fitter = Fitter("config_angle.yml")
fitter.set_data(data)
fitter.set_phsp(phsp)

# Constraints: fixed, same, scale
fitter.set_fixed({"B->..._g_ls_0r": 1.0, "B->..._g_ls_0i": 0.0})
fitter.set_same([["a_par", "b_par"]])
fitter.set_scale({"a_par": -1})

# Boundary ranges (arctan transform, bijective)
fitter.set_range("gamma", -0.3, 0.3)

# Flat parameter vector
x = fitter.initial_values()
nll, grad = fitter.get_nll(x)

# BFGS fit with Hessian
result = fitter.fit(x)
uncertainties = fitter.get_uncertainties(result)  # {name: (val, err)}

# Save/restart
fitter.save_params(result, "results.json")
x_restart = fitter.values_from_dict(json.load("results.json"))

# Plot distributions
fitter.plot(result, prefix="plots/")
```

### Low-level: Direct kernel

```python
from ampfit import Config, NumpyKernel, CUDAKernel

config = Config("config_angle.yml")
kernel_config = config.build_all_index()

# NumPy reference
nk = NumpyKernel(kernel_config)
Q, grads, P = nk._compute(params, data)

# CUDA accelerated
ck = CUDAKernel(kernel_config)
dh = ck.load_data(data)
Q, grads, P = ck.compute(params, dh)
```

## Performance

| Backend | 1000 ev (fwd+bwd) | Notes |
|---------|------------------:|-------|
| **NumPy** | 170.9 ms | reference, float64 |
| **ONNX Runtime** | 19.1 ms | **9× vs NumPy**, float32, CPU only |
| **CUDA** (RTX 3070 Ti) | 6.9 ms | 2.8× vs ONNX, native GPU |

The ONNX model (`pwa_forward.onnx`) has 637 nodes, opset 11, float32, and outputs all 7 gradients (Q, P, grad_ck, grad_m0, grad_g0, grad_scalar) matching the numpy kernel to ~1e-05. Build with `python build_onnx_model.py --batch-size 1000`. Runs anywhere without CUDA Toolkit — just `pip install onnxruntime`.

## Architecture

```
x (flat vector)
  ↓
├─ VariableRegistry: names → indices
├─ ParameterConstraint: ck = build_ck(x_ck)  (combination products)
├─ apply_bounds: arctan transform for bounded params
├─ kernel.compute: forward + backward pass (CUDA or NumPy)
│   ├─ g_bw, bw_p, angular factors, amplitudes
│   ├─ time evolution, probability, NLL
│   └─ Wirtinger gradients for all params
├─ norm from phsp (batched if phsp > GPU memory)
├─ purity-based likelihood: -log(purity·P/norm + (1-purity)·bkg/Nb)
├─ gradient combination: direct + norm chain
└─ BFGS fit → Hessian → uncertainties → JSON export
```

All gradients verified to machine precision (< 1e-10) using 3-point numerical method.
Bound transforms use bijective arctan (not sin) for exact save/restore roundtrip.

## Requirements

- Python ≥ 3.10
- numpy, pyyaml, cffi
- CUDA Toolkit (for GPU acceleration, auto-builds on first use)
- scipy (for BFGS fit)
- matplotlib (for plot)
