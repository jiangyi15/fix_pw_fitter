# TabPWA — Tabulated Partial Wave Analysis

A Python package for partial wave amplitude analysis with NumPy, C/CUDA, and ONNX Runtime
backends, analytic (Wirtinger) gradients, parameter constraints, and a global `Fitter`.

## The name: “Tab” = **tabulated**

**TabPWA = Tabulated Partial Wave Analysis** — TensorFlow-free, and compatible with the
physics conventions of [TFPWA](https://github.com/jiangyi15/tf-pwa).

The “**Tab**” is the core design choice, not a decoration. Instead of evaluating the
momentum-dependent factors on every event, `tabpwa` **samples them once at build time onto
fixed grids**:

- `fl_table` — Blatt–Weisskopf barrier / form factors `F_L(q)`;
- `gamma_table` — running widths `Γ(m)`;
- the angular basis and per-handle amplitude pieces.

At fit time the per-event values are obtained by fast **Catmull–Rom interpolation** of those
tables, so the kernels only recompute what actually changes each iteration (e.g. the
Breit–Wigner propagator). This is what makes it fast without an autodiff framework: TFPWA
computes the same physics directly through TensorFlow autodiff, whereas `tabpwa` precomputes
the tables and propagates **exact analytic (Wirtinger) gradients** through the interpolation.

The tables are built once per configuration by the model layer, then consumed by the pure
**Kernel** layer; batching/normalisation live in the **Backend** layer and constraints + NLL +
optimisation in the **Fitter** (the three-layer architecture). Sampling density is
configurable through `build_defaults` (`n_interp`, `d`, `barrier`, `complex_tail`).

Distribution name `TabPWA`, import name `tabpwa`.

## Installation

```bash
# Install from remote (interp_bw_pwa branch)
pip install git+ssh://git@github.com/jiangyi15/fix_pw_fitter.git@interp_bw_pwa

# Or with optional backends
pip install "tabpwa[onnx] @ git+ssh://git@github.com/jiangyi15/fix_pw_fitter.git@interp_bw_pwa"
pip install "tabpwa[onnx-gpu] @ git+ssh://git@github.com/jiangyi15/fix_pw_fitter.git@interp_bw_pwa"

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
python run_fit.py --save-init starts/init.json             # Dump the start
python run_fit.py --init starts/init.json --fit            # Replay that start
python run_fit.py --fix-mass-width --fit     # Fix masses/widths
```

### Python API

```python
from tabpwa import Fitter

# Backend selection via string shortcut (default: model's registered "default")
fitter = Fitter("config_amp.yml")                         # config backend, else the model's "default" (TD → cuda64/cuda_v3, PWA → cuda_v4_pwa)
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
from tabpwa.amp_frac import AmplitudeFractions

af = AmplitudeFractions(fitter, fit_result)

# Fractions for two subsets of partial waves (single 3‑point or analytical sweep)
vals, errs = af.fractions([[0,1,2], [3,4,5]])

# Custom denominator
vals, errs = af.fractions([range(224)], denominator=range(224, 448))

# Look up ck indices by resonance name (physics/kc live on fitter.model)
idx_f0 = fitter.model.get_ck_indices("f0(500)")
idx = fitter.model.get_decay_ck_indices([("a1(1260)p", "f0(500)")])

# Scripted batch computation
python scripts/calc_fractions.py fit_results.json -o fractions.csv
```

### Low-level: Direct kernel

```python
from tabpwa.amp_model import build_amplitude_model
from tabpwa.backends import create_backend

kc = build_amplitude_model("config_amp.yml").build_kernel_config()

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
# amp_model: flavour_tag_mix # time-dependent (TD) mixing model: adds 6 scalars
#                            # (aliases: flour_tag_mix, p4_directly)
```

Legacy ``data.amp_model:`` is still honoured.  ``build_amplitude_model(source)``
(source = file path, config dict, or ``RawConfig``) returns the model;
``Config(path)`` is a thin legacy alias for it, and ``RawConfig`` is the raw
input (``dic`` / ``_config_path`` / ``backend_spec``).  The model derives the
interpreted physics (decay tree, index/tables, base kernel config
``build_base_kernel_config()``) and owns the policy; ``Fitter`` builds it from
a ``RawConfig`` and keeps it as ``fitter.model``.  Custom models subclass
``tabpwa.amp_model.AmplitudeModel``, register with
``@register_amplitude_model("name")``, and override ``build_kernel_config()``
/ ``build_params_transform()`` / ``build_event_data()``.

Backends register one name per decorator, scoped by amplitude-model **name**:
``@register_backend("integrated_pwa", model="pwa")`` (``model=None`` =
universal).  Stack decorators for aliases/defaults — e.g. ``cuda_v4_pwa`` is
also ``"default"`` for ``pwa``.  The model knows nothing about backends;
``Fitter`` asks ``backends_for_model(model.name)`` and
``create_backend(..., model=model.name)`` resolves ``"default"`` per model.
Choosing a backend not registered for the model (e.g. ``integrated`` for PWA)
is rejected with the registered list.

### Barrier factors & build defaults

The Blatt–Weisskopf form factor of every decay vertex is baked into the kernel
config's ``fl_table`` (every backend only interpolates it), so barrier forms
are pluggable in pure Python: subclass ``tabpwa.BarrierFactor``, register with
``@register_barrier("name")``, and override ``factor(q)`` (extra parameters go
in ``get_params()``).  Built-ins: ``"bw"`` (default, Blatt–Weisskopf) and
``"exp"`` (``exp(-(q·d)²/2)``).

```python
import numpy as np
from tabpwa import BarrierFactor, register_barrier

@register_barrier("myform")
class MyBarrier(BarrierFactor):
    def factor(self, q):
        return np.exp(-0.5 * (q * self.d) ** 2)
```

Select it per decay with the existing decay-entry kwargs (defaults ``bw`` /
``d = 3.0``):

```yaml
decay:
  B:
  - [rho, pi, barrier: {type: exp, d: 1.5}]   # or `barrier: bw` + sibling `d:`
```

Build-time defaults come from the global build context, not the config.  The
module ``tabpwa.build_defaults`` exposes ``scope``, a ``with`` scope:
the overrides apply only inside the scope, then it is back to the global
defaults.

```python
from tabpwa.build_defaults import scope

with scope(n_interp=4000, d=1.5):
    kc = build_amplitude_model(cfg).build_kernel_config()
# here n_interp / d are back to the global defaults
```

Keys: ``n_interp`` (``fl``/``gamma`` table sampling), ``d`` (barrier radius),
``barrier`` (default type), ``complex_tail`` (``(magnitude, phase)`` suffixes
for a complex ck parameter's two slots — default ``("r", "i")``, or
``("rho", "phi")``; set before constructing the ``Fitter``).  A config may also set
``defaults: {d: 1.5, n_interp: 4000}``; the model loads it as a *temporary*
context for its own build, so config files stay physics-only.  The barrier
forms themselves are model metadata (``fitter.model.fl_forms``); the kernel
config keeps only the arrays the kernels index (``fl_table``, ``fl_type``,
``fl_q_index``, ``fl_order``, ``fl_min``, ``fl_delta``).

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
from tabpwa.backends import create_backend

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
- **IntegratedBackend**: wraps the phsp `handle` in a `_PhspBundle` and builds
  Gram matrices lazily via `ensure_gram`, avoiding a redundant GPU re‑upload.

### Pipeline

```
x (flat vector, 115 params)
  ↓ build_params
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

| Name | Backend class | Precision | Model scope |
|------|:-------------|:---------:|:-----------:|
| `numpy` | NumpyBackend | f64 CPU reference | `flavour_tag_mix` |
| `cuda_v3` / `cuda64_v3` / `cuda` | CUDABackendV3 | f64 | `flavour_tag_mix` (`"default"`) |
| `cuda_v3_cache` | CUDABackendV3Cache | f64 | `flavour_tag_mix` (integrated base default) |
| `cuda_v3_sparse` | CUDABackendV3Sparse | f64 (fp32 FA) | `flavour_tag_mix` |
| `cuda_v3_ampcache` | CUDABackendV3AmpCache | f64 | `flavour_tag_mix` |
| `cuda32_v3` | CUDABackendV3F32 | f32 | `flavour_tag_mix` |
| `cuda_mixed_v3` | CUDABackendV3Mixed | f32+f64 | `flavour_tag_mix` |
| `cuda_v2` / `cuda64_v2` | CUDABackendV2 | f64 | `flavour_tag_mix` |
| `cuda32_v2` | CUDABackendV2F32 | f32 | `flavour_tag_mix` |
| `integrated` | IntegratedBackend | base‑dependent | `flavour_tag_mix` |
| `onnx` / `onnx_cpu` / `onnx_cuda` | ONNXBackend | f32 | `flavour_tag_mix` |
| `numpy_pwa` | NumpyPWABackend | f64 CPU reference | `pwa` |
| `cuda_v4_pwa` | CUDABackendV4PWA | f64 | `pwa` (`"default"`) |
| `cuda_v4_pwa_cache` / `cuda32_v4_pwa_cache` | CUDABackendV4PWACache / …Cache32 | f64 (f32 cache) | `pwa` |
| `cuda_v5_pwa` | CUDABackendV5PWA | f64 (group-log NLL) | `pwa` |
| `integrated_pwa` | IntegratedPWABackend | base‑dependent | `pwa` |
| `cpu_v3` / `cpu64_v3` | CPUBackendV3 | f64 (C + OpenMP + AVX2) | `flavour_tag_mix` |
| `shard` | ShardBackend | multi‑process | universal (`model=None`) |

Each model resolves a `"default"` when no backend is given: `cuda_v3` for
`flavour_tag_mix`, `cuda_v4_pwa` for `pwa`.

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
