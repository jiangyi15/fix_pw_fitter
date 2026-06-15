# AGENTS.md — ampfit

## Quick Start

```bash
# Install in dev mode
pip install -e .

# One-command fit (uses cuda32_v2 backend by default)
./fit.sh

# Or manually
python run_fit.py --fit --maxiter 1000 --backend cuda32_v2 --save results.json --plot plots/
```

## Key Commands

| Task | Command |
|------|---------|
| Install | `pip install -e .` |
| Full fit | `./fit.sh [config] [data.npz] [phsp.npz] [output/]` |
| Compute NLL only | `python run_fit.py` |
| Fit with options | `python run_fit.py --fit --maxiter 200 --backend numpy --plot results/` |
| Restart from checkpoint | `python run_fit.py --init results.json --fit` |
| Fix masses/widths | `python run_fit.py --fix-mass-width --fit` |

## Backend Selection

Backend is set via `--backend` flag or `Fitter(..., backend=...)`:

| Backend | Flag | Requires | Speed |
|---------|------|----------|-------|
| CUDA f32 v2 | `cuda32_v2` | CUDA Toolkit | **4× (fastest)** |
| CUDA f32 v3 | `cuda32_v3` | CUDA Toolkit | **4× (fastest)** |
| CUDA f64 v2 | `cuda64_v2`, `cuda_v2` | CUDA Toolkit | 2.7× |
| CUDA f64 v3 | `cuda64_v3`, `cuda_v3` | CUDA Toolkit | 2.7× |
| CUDA (alias) | `cuda`, `cuda32`, `cuda64` | CUDA Toolkit | maps to v2/v3 |
| ONNX CUDA | `onnx_cuda` | onnxruntime-gpu | 64× |
| ONNX CPU | `onnx_cpu` | onnxruntime | 8× |
| NumPy | `numpy` | — | 1× (reference) |

**v2 vs v3**: v2 is the original kernel, v3 has improved numerical stability for large datasets.

CUDA auto-builds on first import (needs `nvcc`). ONNX builds model in-memory at runtime.

## Architecture

```
x (flat param vector)
  → VariableRegistry (names ↔ indices)
  → ParameterConstraint (ck = product combos)
  → apply_bounds (arctan transform)
  → backend.compute (forward + backward)
      ├─ NumPyBackend   — pure NumPy f64
      ├─ CUDABackend    — custom CUDA C kernels (f64/f32)
      └─ ONNXBackend    — dual ONNX models (norm + forward)
  → norm from phsp (batched)
  → NLL: -log(purity·P/norm + (1-purity)·bkg)
  → BFGS fit → Hessian → uncertainties
```

## Package Layout

- `src/ampfit/` — main package
  - `config_loader.py` — YAML config → index arrays
  - `numpy_kernel.py` — reference kernel with Wirtinger gradients
  - `backends.py` — compute backends (NumPy, CUDA, ONNX)
  - `fitter.py` — Fitter class (NLL, fit, save, plot)
  - `param_constraint.py` — fixed/same/scale constraints
  - `boundary.py` — arctan bijective bound transform
  - `cuda/kernels.cu` — CUDA C source (auto-compiled)

## Gotchas

1. **Gradient calculus**: Uses Wirtinger calculus for complex parameters. Real params (m₀, Γ) use `2·Re(∂Q/∂z · ∂z/∂x)`, complex params (ck) use `∂Q/∂z*` directly.

2. **ONNX data transfer**: Unlike CUDA (persistent GPU memory), ONNX uploads data to GPU every `sess.run()` call. Large datasets are batched automatically (default 1024).

3. **Config files**: YAML format with particle models and decay chains. See `config_angle.yml` for example.

4. **NPZ data format**: Input data is `{mass, q, angle, weight, ...}` arrays in `.npz` files.

5. **Gradient validation**: Run `python tests/validate_gradients.py` to check 3-point numerical vs analytical gradients.

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_cuda_kernel.py

# Validate gradients
python tests/validate_gradients.py
```

## Dependencies

- **Required**: numpy, pyyaml, cffi, scipy (for BFGS)
- **Optional**: onnxruntime/onnxruntime-gpu, matplotlib (for plots)
- **CUDA**: nvcc in PATH for GPU acceleration
