# ampfit — Amplitude Analysis Fitting Framework

A Python package for partial wave amplitude analysis with NumPy, CUDA, and ONNX Runtime backends,
full Wirtinger-calculus gradients, parameter constraints, and a global Fitter class.

## Package Structure

```
├── pyproject.toml
├── fit.sh                          # One-command fit pipeline
├── run_fit.py                      # Full fit script with archive-compatible constraints
├── config_angle.yml                # Example configuration
├── build_onnx_model.py             # CLI to export ONNX models to file (optional)
├── src/ampfit/
│   ├── __init__.py                 # Exports: Config, Fitter, backends, ...
│   ├── config_loader.py            # YAML config → index arrays for kernels
│   ├── particle_model.py           # Particle property definitions + get_gamma_defaults()
│   ├── angular_formula.py          # Angular distribution formulas
│   ├── numpy_kernel.py             # NumPy reference kernel (Wirtinger gradients)
│   ├── backends.py                 # ComputeBackend base + 4 backends
│   ├── _cuda.py                    # CUDA Python bindings (CFFI), float64 kernel
│   ├── _cuda_f32.py                # CUDA float32 kernel (2-4× faster)
│   ├── _onnx_builder.py            # In-memory ONNX graph builder (no PyTorch)
│   ├── cuda/
│   │   ├── kernels.cu              # Optimized CUDA C kernels (forward + backward)
│   │   └── build.py                # Auto-build on first import
│   ├── fitter.py                   # Fitter: NLL → BFGS fit → save_params → plot
│   ├── param_constraint.py         # Parameter constraints: fixed, same, scale
│   └── boundary.py                 # BoundTransform (arctan-based, bijective)
├── tests/
│   ├── test_cuda_kernel.py
│   ├── test_onnx_cuda.py           # ONNX CUDA smoke tests
│   ├── test_fitter_constraints.py
│   ├── benchmark_backends.py       # Cross-backend performance benchmark
│   └── validate_gradients.py       # 3-point numerical gradient verification
└── archive/                        # Historical files (TensorFlow reference)
```

## Installation

```bash
pip install -e .                  # Install in development mode
# CUDA builds automatically on first use (requires nvcc)
# ONNX Runtime: pip install onnxruntime onnx  (GPU: onnxruntime-gpu)
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

# Backend selection via string shortcut
fitter = Fitter("config_angle.yml")              # default: CUDA f64
fitter = Fitter("config_angle.yml", backend="cuda32")   # CUDA f32
fitter = Fitter("config_angle.yml", backend="numpy")    # NumPy f64
fitter = Fitter("config_angle.yml", backend="onnx_cpu") # ONNX CPU
fitter = Fitter("config_angle.yml", backend="onnx_cuda")# ONNX CUDA

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
from ampfit import Config
from ampfit.backends import NumpyBackend, CUDABackend, ONNXBackend

config = Config("config_angle.yml")
kc = config.build_all_index()

# NumPy reference
nk = NumpyBackend(kc)
Q, grads, P = nk.compute(params, data)

# CUDA accelerated (f64 or f32)
ck = CUDABackend(kc, dtype="float64")
dh = ck.load_data(data)
Q, grads, P = ck.compute(params, dh)

# ONNX Runtime (CPU or CUDA) — builds model in-memory
onnx = ONNXBackend(kernel_config=kc, providers=["CUDAExecutionProvider"])
dh = onnx.load_data(data)
Q, grads, P = onnx.compute(params, dh)
```

## Performance

All benchmarks on **NVIDIA GeForce RTX 3070 Ti Laptop GPU** (events/sec, higher is better):

| Backend | 64 | 256 | 1024 | 8192 | vs NumPy (max) |
|---------|:---:|:---:|:----:|:----:|:--------------:|
| **NumPy f64** CPU | 5.5K | 5.6K | 5.3K | 5.2K | 1× |
| **CUDA f64** GPU | 66K | 110K | 162K | 183K | **33×** |
| **CUDA f32** GPU | 119K | 336K | 681K | 798K | **145×** |
| **ONNX f32** CPU | 3K | 11K | 40K | 32K | 8× |
| **ONNX f32** CUDA | 21K | 79K | 355K | 338K | **64×** |

**Latency** (forward + backward pass, 1024 events):

| Backend | Time | Speedup |
|---------|:----:|:-------:|
| **NumPy f64** CPU | 193 ms | 1× |
| **ONNX** CPU | 26 ms | 7.4× |
| **ONNX** CUDA | **2.9 ms** | **67×** |
| **CUDA f64** GPU | 6.3 ms | 31× |
| **CUDA f32** GPU | **1.5 ms** | **129×** |

### Key observations

- **CUDA f32** is the fastest overall: **145× vs NumPy**, **4× faster than CUDA f64**
- **ONNX CUDA** offers 64× speedup without requiring CUDA Toolkit at build time
- **ONNX CPU** is 8× vs NumPy — useful on machines without GPU
- Custom CUDA kernels outperform ONNX because they are purpose-built for this computation
- ONNX model is built **in-memory** from kernel config — no pre-exported `.onnx` file needed

## Backend Architecture

```
x (flat vector)
  ↓
├─ VariableRegistry: names → indices
├─ ParameterConstraint: ck = build_ck(x_ck)  (combination products)
├─ apply_bounds: arctan transform for bounded params
├─ backend.compute: forward + backward pass
│   │
│   ├─ NumPyBackend   — pure NumPy f64, reference implementation
│   ├─ CUDABackend    — custom CUDA C kernels (f64 or f32)
│   └─ ONNXBackend    — ONNX Runtime (CPU/CUDA), dual-model:
│       ├─ norm model: Q = sum(P·weight), norm gradients
│       └─ forward model: NLL + NLL gradients
│
├─ norm from phsp (batched for large datasets)
├─ purity-based likelihood: -log(purity·P/norm + (1-purity)·bkg/Nb)
├─ gradient combination: direct + norm chain
└─ BFGS fit → Hessian → uncertainties → JSON export
```

### ONNXBackend Design

The ONNX backend uses **two ONNX models** built in-memory:

- **Norm model** (`norm_model=True`): computes `Q = sum(P·weight)` with correct norm gradients — used internally for normalisation integral computation
- **Forward model** (`norm_model=False`): computes full NLL `Q = -Σw·log(P/norm + bkg)` with NLL gradients — used for likelihood evaluation during fitting

Both models are built from `kernel_config` at a moderate fixed batch size (default 1024). The `compute()` method handles arbitrarily large datasets by **splitting into fixed-size batches** with weight-0 masking on the final partial batch.

**Data transfer note:** Unlike `CUDABackend.load_data()` which uploads data to GPU
persistently via `GPUDataHolder`, `ONNXBackend.load_data()` stores data as numpy arrays
in system memory. ONNX Runtime transfers data CPU→GPU inside every `sess.run()` call,
so data upload happens on each batch of every `compute()` invocation. This adds overhead
vs the custom CUDA backend which keeps data resident on GPU. A future optimization could
use ONNX Runtime's `IOBinding` to pre-allocate GPU buffers.

## Gradient Validation

All backends validated against a **3-point central-difference numerical reference**:

| Backend | norm=None | norm=NLL |
|---------|:---------:|:--------:|
| NumPy f64 | 1e-9 to 1e-11 | 1e-7 to 1e-8 |
| CUDA f64 | 1e-9 to 1e-11 | 1e-7 to 1e-8 |
| CUDA f32 | 1e-7 to 3e-8 | 1e-7 to 5e-8 |
| ONNX f32 | 1e-4 to 3e-4 | 2e-4 to 5e-4 |

ONNX f32 precision (~1e-4) is limited by float32 vs the float64 numerical reference.
Bound transforms use bijective arctan (not sin) for exact save/restore roundtrip.

## Requirements

- Python ≥ 3.10
- numpy, pyyaml, cffi
- onnx, onnxruntime or onnxruntime-gpu (for ONNX backend)
- CUDA Toolkit (for GPU acceleration, auto-builds on first use)
- scipy (for BFGS fit)
- matplotlib (for plot)
