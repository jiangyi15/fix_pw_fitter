# ampfit — Amplitude Analysis Fitting Framework

A Python package for partial wave amplitude analysis with NumPy, CUDA, and ONNX Runtime backends,
full Wirtinger-calculus gradients, parameter constraints, and a global Fitter class.

## Package Structure

```
├── pyproject.toml
├── fit.sh                          # One-command fit pipeline
├── run_fit.py                      # Full fit script with archive-compatible constraints
├── config_amp.yml                  # Example configuration
├── scripts/
│   └── calc_fractions.py           # Amplitude fraction calculator
├── src/ampfit/
│   ├── __init__.py                 # Exports: Config, Fitter, backends, ...
│   ├── config_loader.py            # YAML config → index arrays for kernels
│   │                               #   + get_ck_indices() / get_decay_ck_indices()
│   ├── particle_model.py           # Particle property definitions + BW parameter models
│   ├── particle_model/             # Lineshape models (GS_rho, BWR, Flatte, Bugg, ...)
│   ├── angular_formula.py          # Angular distribution formulas
│   ├── numpy_kernel.py             # NumPy reference kernel (Wirtinger gradients)
│   ├── amp_frac.py                 # AmplitudeFractions — ratio + uncertainty propagation
│   ├── backends/                   # ComputeBackend base + backends
│   │   ├── core.py                 #   Base class + registry
│   │   ├── numpy_backend.py        #   NumPy f64 reference
│   │   ├── cuda_backends.py        #   CUDA v2/v3 (f64/f32)
│   │   └── onnx_backend.py         #   ONNX Runtime (CPU/CUDA)
│   ├── _cuda_v2.py                 # CUDA float64 v2 kernel (CFFI)
│   ├── _cuda_v2_f32.py             # CUDA float32 v2 kernel
│   ├── _cuda_v3.py                 # CUDA float64 v3 kernel (Catmull-Rom)
│   ├── _cuda_v3_f32.py             # CUDA float32 v3 kernel (Catmull-Rom)
│   ├── _onnx_builder.py            # In-memory ONNX graph builder (no PyTorch)
│   ├── cuda/
│   │   ├── build.py                # Auto-build on first import
│   │   ├── kernels_v2.cu           # CUDA C kernels v2 (linear interpolation)
│   │   ├── kernels_v2_f32.cu       # Float32 v2 kernel
│   │   ├── kernels_v3.cu           # CUDA C kernels v3 (Catmull-Rom interpolation)
│   │   └── kernels_v3_f32.cu       # Float32 v3 kernel
│   ├── fitter.py                   # Fitter: NLL → BFGS → save_params → cal_uncertainties
│   ├── param_constraint.py         # Parameter constraints: fixed, same, scale
│   ├── boundary.py                 # BoundTransform (arctan-based, bijective)
│   └── config_loader.py            # YAML → index arrays
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
# Install from remote (interp_bw_ftime branch)
pip install git+ssh://git@github.com/jiangyi15/fix_pw_fitter.git@interp_bw_ftime

# Or with optional backends
pip install "ampfit[onnx] @ git+ssh://git@github.com/jiangyi15/fix_pw_fitter.git@interp_bw_ftime"
pip install "ampfit[onnx-gpu] @ git+ssh://git@github.com/jiangyi15/fix_pw_fitter.git@interp_bw_ftime"

# Install in development mode (local)
pip install -e .
# CUDA builds automatically on first use (requires nvcc)
# ONNX Runtime: pip install onnxruntime onnx  (GPU: onnxruntime-gpu)
```

## Usage

### One-command fit (shell script)

```bash
./fit.sh                                    # Full fit with defaults
./fit.sh config_amp.yml data.npz phsp.npz results/
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

# Backend selection via string shortcut (default: cuda → v3)
fitter = Fitter("config_amp.yml")                         # CUDA f64 v3
fitter = Fitter("config_amp.yml", backend="cuda32_v3")    # CUDA f32 v3
fitter = Fitter("config_amp.yml", backend="numpy")        # NumPy f64
fitter = Fitter("config_amp.yml", backend="onnx_cuda")    # ONNX CUDA

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

# Save / load results (auto-saves error_matrix.npy alongside JSON)
fitter.save_params(result, "results.json")
loaded = fitter.load_results("results.json")   # auto-loads error_matrix.npy
# → loaded.x, loaded.hess_inv

# Uncertainty propagation for derived observables
def my_observable(phys_dict):
    return value
err = fitter.cal_uncertainties(my_observable, param_names, result)

# Plot distributions
fitter.plot(result, prefix="plots/")
```

### Amplitude Fractions

```python
from ampfit.amp_frac import AmplitudeFractions

af = AmplitudeFractions(fitter, fit_result)

# Fractions for two subsets of partial waves (single 3‑point or analytical sweep)
vals, errs = af.fractions([[0,1,2], [3,4,5]])

# Custom denominator
vals, errs = af.fractions([range(224)], denominator=range(224, 448))

# Look up ck indices by resonance name
idx_f0 = fitter.config.get_ck_indices("f0(500)")
idx = fitter.config.get_decay_ck_indices([("a1(1260)p", "f0(500)")])

# Scripted batch computation
python scripts/calc_fractions.py fit_results.json -o fractions.csv
```

### Low-level: Direct kernel

```python
from ampfit import Config
from ampfit.backends import NumpyBackend, create_backend

config = Config("config_amp.yml")
kc = config.build_all_index()

# NumPy reference
nk = NumpyBackend(kc)
Q, grads, P = nk.compute(params, data)

# CUDA v3 (or use string "cuda" / "cuda_v3")
ck = create_backend("cuda_v3", kc)
dh = ck.load_data(data)
Q, grads, P = ck.compute(params, dh)
```

## Performance

All benchmarks on **NVIDIA GeForce RTX 3070 Ti Laptop GPU** (events/sec, higher is better).
Mean ± 1σ over 5 runs:

| Backend | 64 | 256 | 1024 | 10000 | vs NumPy (max) |
|---------|:---:|:---:|:----:|:-----:|:--------------:|
| **NumPy f64** CPU | 6.9±0.1K | 6.6±0.1K | 6.1±0.2K | 4.4±0.0K | 1× |
| **CUDA f64 v2** GPU | 84±1K | 128±14K | 175±1K | 187±0.4K | **29×** |
| **CUDA f64 v3** GPU | 74±6K | 114±1K | 147±5K | 186±0.3K | **24×** |
| **CUDA f32 v2** GPU | 137±6K | 376±1K | 680±10K | 786±3K | **111×** |
| **CUDA f32 v3** GPU | 124±8K | 290±3K | 484±4K | 612±2K | **79×** |
| **ONNX f32** CPU | 3.0±0.2K | 12±1K | 48±3K | 47±2K | 8× |
| **ONNX f32** CUDA | 22.2±0.1K | 88±1K | 360±2K | 343±3K | **59×** |

**Latency** (forward + backward pass, 1024 events):

| Backend | Time | Speedup |
|---------|:----:|:-------:|
| **NumPy f64** CPU | 169 ms | 1× |
| **ONNX** CPU | 21 ms | 8.0× |
| **ONNX** CUDA | **2.8 ms** | **60×** |
| **CUDA f64 v2** GPU | 5.9 ms | 29× |
| **CUDA f64 v3** GPU | 7.0 ms | 24× |
| **CUDA f32 v2** GPU | **1.5 ms** | **112×** |
| **CUDA f32 v3** GPU | 2.1 ms | 80× |

### Key observations

- **CUDA f32 v2** is the fastest overall: **111× vs NumPy**, **~4× faster than CUDA f64**
- **CUDA f32 v3** has better numerical stability for large datasets but slightly slower than v2
- **CUDA v2 (f64)** is slightly faster than v3 at large batches (per-event norm vs Gram matrix)
- **ONNX CUDA** offers 59× speedup without requiring CUDA Toolkit at build time
- **ONNX CPU** is 8× vs NumPy — useful on machines without GPU
- Custom CUDA kernels outperform ONNX because they are purpose-built for this computation
- ONNX model is built **in-memory** from kernel config — no pre-exported `.onnx` file needed
- `cuda` / `cuda64` are aliases for `cuda_v3` / `cuda64_v3` (the latest stable v3 backend)

## Backend Architecture

```
x (flat vector)
  ↓
├─ VariableRegistry: names → indices
├─ ParameterConstraint: ck = build_ck(x_ck)  (combination products)
├─ apply_bounds (arctan transform for bounded params)
├─ backend.compute: forward + backward pass
│   │
│   ├─ NumPyBackend   — pure NumPy f64, reference implementation
│   ├─ CUDABackendV3  — CUDA C kernels f64/f32 v3 (Catmull-Rom, default)
│   ├─ CUDABackendV2  — CUDA C kernels f64/f32 v2 (linear interpolation)
│   └─ ONNXBackend    — ONNX Runtime (CPU/CUDA)
│
├─ norm from phsp (batched for large datasets)
├─ purity-based likelihood: -log(purity·P/norm + (1-purity)·bkg/Nb)
├─ gradient combination: direct + norm chain
└─ BFGS fit → Hessian → uncertainties → JSON + error_matrix.npy
```

### Result persistence

`save_params()` writes two files:

| File | Format | Content |
|------|--------|---------|
| `results.json` | JSON | Best-fit parameter values, errors, NLL, status |
| `results_error_matrix.npy` | NumPy .npy | Inverse Hessian (covariance) matrix |

`load_results()` auto-detects and loads the `_error_matrix.npy` when present.

## Gradient Validation

All backends validated against a **3-point central-difference numerical reference**:

| Backend | norm=None | norm=NLL |
|---------|:---------:|:--------:|
| NumPy f64 | 1e-9 to 1e-11 | 1e-7 to 1e-8 |
| CUDA f64 v2 | 1e-9 to 1e-11 | 1e-7 to 1e-8 |
| CUDA f64 v3 | 1e-9 to 1e-11 | 1e-7 to 1e-8 |
| CUDA f32 v2 | 1e-7 to 3e-8 | 1e-7 to 5e-8 |
| CUDA f32 v3 | 1e-7 to 3e-8 | 1e-7 to 5e-8 |
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
