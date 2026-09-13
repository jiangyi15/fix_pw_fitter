# ampfit — Amplitude Analysis Fitting Framework

A Python package for partial wave amplitude analysis with NumPy, CUDA, and ONNX Runtime backends,
full Wirtinger-calculus gradients, parameter constraints, and a global Fitter class.

## Installation

```bash
# Install from remote (interp_bw_pwa branch)
pip install git+ssh://git@github.com/jiangyi15/fix_pw_fitter.git@interp_bw_pwa

# Or with optional backends
pip install "ampfit[onnx] @ git+ssh://git@github.com/jiangyi15/fix_pw_fitter.git@interp_bw_pwa"
pip install "ampfit[onnx-gpu] @ git+ssh://git@github.com/jiangyi15/fix_pw_fitter.git@interp_bw_pwa"

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
fitter = Fitter("config_amp.yml")                         # backend from config, else cuda64
fitter = Fitter("config_amp.yml", backend="numpy")        # explicit override
fitter = Fitter("config_amp.yml", backend="cuda32_v3")    # CUDA f32 v3
# the config may also select it:  config: {backend: numpy}
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
from ampfit.backends import create_backend

config = Config("config_amp.yml")
kc = config.build_all_index()

# NumPy reference
nk = create_backend("numpy", kc)
nh = nk.load_data(data)
Q, grads, P = nk.compute(params, nh)

# CUDA v3 (or use string "cuda" / "cuda_v3")
ck = create_backend("cuda_v3", kc)
dh = ck.load_data(data)
Q, grads, P = ck.compute(params, dh)
```

### Amplitude models (what the config builds)

The config's ``amp_model`` selects the amplitude model, which owns the
kernel-config content and the parameter transform (so there is no runtime
"has scalars?" branching):

```yaml
amp_model: pwa               # default: ck/m0/g0 only (no time/mixing/scalars)
# amp_model: flavour_tag_mix # legacy time/mixing model: adds 6 scalars
#                            # (aliases: flour_tag_mix, p4_directly)
```

Legacy ``data.amp_model:`` is still honoured.  Custom models subclass
``ampfit.amp_model.AmplitudeModel(config)``, register with
``@register_amplitude_model("name")``, set ``default_backend``, and override
``build_kernel_config()`` / ``build_params_transform()``.  Backends declare
the model they serve at registration —
``@register_backend("integrated_pwa", amp_model="pwa")`` (``amp_model=None``
= universal) — and the model derives its valid backend set from that registry.
Choosing a backend the model does not register (e.g. ``integrated`` for the
PWA model, which uses ``integrated_pwa``) is rejected by ``Fitter`` with the
registered list.

### Pure-PWA (projection-sum) kernels

The pure-PWA family implements the scalar-free projection-sum model

```
P(e) = Σ_p |A_p(e)|² ,     A_p(e) = Σ_k ck_k · a_{p,k}(e)
```

— no time evolution / D-mixing / scalar (Γ, ΔΓ, …) parameters.  All
projections share the same ``ck``; the projection changes only the angular
part of each wave.  Wave entries are stored **p-major**
(``n_wave = n_proj · N``, ``ck`` length ``N``); the projection count is the
config key ``n_proj`` (default 1).

| Backend | Note |
|---------|------|
| ``numpy_pwa`` | CPU f64 reference for the projection-sum model |
| ``cuda_v4_pwa`` | GPU f64: one-time angular-amplitude cache + per-iteration BW propagator, so ``m0``/``g0`` keep flowing |
| ``cuda_v4_pwa_cache`` / ``cuda32_v4_pwa_cache`` | **full-amplitude** cache for fits with every ``m0``/``g0`` FIXED (bit-equal to v4 there; returns zero m0/g0 gradients; use ``cuda_v4_pwa`` when masses/widths float). The 32 variant stores the cache as float2 |
| ``cuda_v5_pwa`` | Same model/forward as v4, but the data NLL **does not log per event**: events are grouped into ``resolution_size``-sized chunks and one log is taken per group, ``Q = -Σ_groups log Σ_{e∈g} w_e·(P_e/norm + bkg_e)``. ``resolution_size`` defaults to 1 (≡ per-event v4 for unit weights). Groups align to the event index, independent of the GPU ``batch_size``. For NLL comparisons against v4 at ``resolution_size > 1`` the copy rows of each group must carry group-normalised weights (``w = 1/resolution_size`` each) |
| ``integrated_pwa`` | Gram-matrix norm (O(n²)) with the data NLL delegated to a base backend (default ``cuda_v4_pwa``) |

```python
from ampfit.backends import create_backend

be = create_backend("cuda_v4_pwa", kc)                 # fitted m0/g0
be = create_backend({"name": "cuda_v5_pwa",             # group-log NLL
                     "resolution_size": 20}, kc)
be = create_backend({"name": "integrated_pwa",          # Gram norm
                     "base": "cuda_v4_pwa"}, kc)
```

NumPy backends (``numpy`` / ``numpy_pwa``) are CPU reference
implementations, useful for tests and cross-checks; production fits
normally use a CUDA/CPU backend.

**dNLL/dnorm contract**: every backend whose ``compute(..., norm=<float>)``
runs returns ``grads["norm"]`` — the native d(NLL)/d(norm) of its own
objective (per-event backends use the closed-form per-event derivative;
caches / ``cuda_v5_pwa`` supply the value from their kernel).  The fitter
consumes it from the returned gradient dict, and shard/integrated wrappers
forward it automatically.

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

Implementations: **NumpyKernel** (reference), **CUDAKernelV3/V2** (f64/f32),
**ONNXKernel**, and the pure-PWA kernels **NumpyPWA**, **CUDAKernelV4PWA**,
**CUDAKernelV4PWACache / …Cache32**, **CUDAKernelV5PWA**.

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
| `numpy` | NumpyBackend | f64 CPU reference |
| `cuda_v3` / `cuda64_v3` / `cuda` | CUDABackendV3 | f64 |
| `cuda32_v3` | CUDABackendV3F32 | f32 |
| `cuda_mixed_v3` | CUDABackendV3Mixed | f32+f64 |
| `cuda_v2` / `cuda64_v2` | CUDABackendV2 | f64 |
| `cuda32_v2` | CUDABackendV2F32 | f32 |
| `integrated` | IntegratedBackend | base‑dependent |
| `onnx_cpu` / `onnx_cuda` | ONNXBackend | f32 |
| `numpy_pwa` | NumpyPWABackend | f64 CPU reference |
| `cuda_v4_pwa` | CUDABackendV4PWA | f64 |
| `cuda_v4_pwa_cache` / `cuda32_v4_pwa_cache` | CUDABackendV4PWACache / …Cache32 | f64 (f32 cache) |
| `cuda_v5_pwa` | CUDABackendV5PWA | f64 (group-log NLL) |
| `integrated_pwa` | IntegratedPWABackend | base‑dependent |
| `cpu_v3` / `cpu64_v3` | CPUBackendV3 | f64 (C + OpenMP + AVX2) |
| `shard` | ShardBackend | multi‑process |

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
