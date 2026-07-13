# ampfit — Amplitude Analysis Fitting Framework

A Python package for partial wave amplitude analysis with NumPy, CUDA, and ONNX Runtime backends,
full Wirtinger-calculus gradients, parameter constraints, and a global Fitter class.

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

| Backend | 64 | 256 | 1024 | 10000 | vs NumPy (max) |
|---------|:---:|:---:|:----:|:-----:|:--------------:|
| **NumPy f64** CPU | 6.1K | 8.2K | 6.3K | 4.5K | 1× |
| **CUDA f64 v2** GPU | 72K | 114K | 176K | 188K | **42×** |
| **CUDA f64 v3** GPU | 84K | 140K | 172K | 186K | **41×** |
| **CUDA f64 v3_split** GPU | 104K | 220K | 267K | 322K | **72×** |
| **CUDA f64 v3_sparse** GPU | **115K** | **363K** | **559K** | **715K** | **159×** |
| **CUDA f32 v3** GPU | 138K | 349K | 504K | 625K | 139× |
| **CUDA f32 v2** GPU | 140K | 370K | 691K | 801K | **178×** |

**Latency** (forward + backward pass, 1024 events):

| Backend | Time | Speedup |
|---------|:----:|:-------:|
| **NumPy f64** CPU | 163 ms | 1× |
| **CUDA f64 v2** GPU | 7.4 ms | 22× |
| **CUDA f64 v3** GPU | 6.1 ms | 27× |
| **CUDA f64 v3_split** GPU | 3.7 ms | 44× |
| **CUDA f64 v3_sparse** GPU | **1.8 ms** | **91×** |
| **CUDA f32 v3** GPU | 2.1 ms | 78× |
| **CUDA f32 v2** GPU | **1.5 ms** | **109×** |

### Key observations

- **CUDA f32 v2** is the fastest overall: **178× vs NumPy**, but uses float32 precision throughout
- **CUDA f64 v3_sparse** is the fastest FP64 backend: **159× vs NumPy**, **3.8× faster than v3** at scale
  - Sparse scatter/gather replaces 99.5%-sparse matrix_gamma matmul (62208 FMAs → 288 ops)
  - FP32 FA (angular factor) + float momentum/angle storage for bandwidth savings
  - Full FP64 precision for g0/m0/BW computation
- **CUDA f64 v3_split** is the baseline split-kernel: 72× vs NumPy
- **CUDA v3_sparse** outperforms **CUDA f32 v3** at large batch sizes (715K vs 625K eve/s at 10K)
- Custom CUDA kernels outperform ONNX because they are purpose-built for this computation
- ONNX model is built **in-memory** from kernel config — no pre-exported `.onnx` file needed
- `cuda` / `cuda64` are aliases for `cuda_v3` / `cuda64_v3`
- `cuda_v3_sparse` can also be used as the base backend for IntegratedBackend

## Architecture: Three Layers

### 1. Kernel — raw compute engine

The lowest level. Pure number crunching with no knowledge of constraints, normalization, or data management.

```python
class CUDAKernelV3:        # cuda/_v3.py
    def __init__(self, config)           # config dict from build_all_index()
    def load_data(self, data) → Handle  # upload to GPU
    def compute(self, params, handle, norm=None) → (Q, grads, P)
    def compute_gram(self, phsp_handle, m0, g0) → Gram matrices
```

Implementations: **NumpyKernel** (reference), **CUDAKernelV3/V2** (f64/f32), **ONNXKernel**.

### 2. Backend — standard interface around a kernel

Wraps a kernel with a uniform API for the Fitter. Handles batching, `return_p` semantics, and optional pre‑computation (e.g. Gram matrices for IntegratedBackend).

```python
class _CUDABackend(ComputeBackend):   # backends/cuda_backends.py
    def load_data(self, data_np) → Handle
    def compute(self, params, handle, norm=None, return_p=True)
    def free(self)
```

All CUDA backends share `_CUDABackend` — each subclass just selects which kernel to wrap.
**IntegratedBackend** adds Gram‑matrix based norm for O(n²) instead of O(N_phsp · n_wave).

### 3. Fitter — orchestrator with constraint pipeline

Owns data, constraints, and the full parameter transform chain. The only layer that stores numpy arrays.

```python
class Fitter:
    set_data(data), set_phsp(phsp)     # store data → backend.load_data()
    set_fixed(slots), set_same(pairs)  # constraint transforms
    get_nll(x) → (nll, grad)           # full pipeline
    fit(x0, ...) → result              # BFGS optimizer
```

### Data ownership

```
Storage:        Fitter owns numpy arrays (_data_np, _phsp_np)
                    ↓  backend.load_data()
Handle:         Backend returns a handle (numpy dict or CUDA DataHandle)
                    ↓  passed to compute()
Compute:        Kernel reads arrays/handle, produces Q, grads, P
```

- **NumPy**: `load_data` is a no‑op — returns the dict itself (zero copy).
- **CUDA**: `load_data` uploads to GPU, returns a `DataHandle` (GPU pointer).
- **IntegratedBackend**: passes the phsp `handle` through to `_load_phsp_matrices`,
  avoiding a redundant GPU re‑upload for Gram computation.

### Pipeline

```
x (flat vector, 115 params)
  ↓ _build_params
apply_bounds → to_dict → resolve → build_ck  (constraint chain)
  ↓ params dict {ck, m0, g0, scalar}
  ├── Fitter._compute_norm_batched:
  │     backend.compute(params, phsp_handle, norm=None, return_p=False) → norm
  ├── Fitter.get_nll_raw:
  │     backend.compute(params, data_handle, norm=norm) → (nll, grads)
  │     total_grad = kernel_grad + dNLL_dnorm · norm_grad
  └── Fitter._flat_gradient:
        chain_gradient(mass_width → scale → fixed) → bound_grad
  ↓
(nll, grad_x)  → BFGS optimizer
```

### Registered backends

| Name | Backend class | Precision |
|------|:-------------|:---------:|
| `numpy` | NumpyBackend | f64 |
| `cuda_v3` / `cuda64_v3` / `cuda` | CUDABackendV3 | f64 |
| `cuda32_v3` | CUDABackendV3F32 | f32 |
| `cuda_mixed_v3` | CUDABackendV3Mixed | f32+f64 |
| `cuda_v2` / `cuda64_v2` | CUDABackendV2 | f64 |
| `cuda32_v2` | CUDABackendV2F32 | f32 |
| `integrated` | IntegratedBackend | base‑dependent |
| `onnx_cpu` / `onnx_cuda` | ONNXBackend | f32 |

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
- **Core**: numpy, pyyaml, scipy, sympy, matplotlib, cffi
- **ONNX backend**: onnx, onnxruntime or onnxruntime-gpu (optional)
- **CUDA**: NVIDIA GPU + CUDA Toolkit (auto-builds kernels on first use, optional)
- **tests**: pytest
