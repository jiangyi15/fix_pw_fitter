# AGENTS.md — ampfit

Amplitude analysis fitting framework.  NumPy / CUDA / ONNX backends, Wirtinger gradients, BFGS fit with constraints.

## Quick Start

```bash
pip install -e .
./fit.sh                                          # full pipeline
python run_fit.py --fit --maxiter 1000 --backend integrated  # or explicit
```

## Backend Selection

Backend via `--backend` or `Fitter(..., backend=...)`.

| Backend | Flag | Note |
|---------|------|------|
| **Integrated** | `{"name": "integrated", "base": "cuda_v3"}` | Hyper backend — Gram matrix O(n²) norm + base for data NLL. Default in `fit.sh`. |
| CUDA f64 v3 | `cuda_v3`, `cuda`, `cuda64` | Catmull-Rom interpolation (default standalone). |
| CUDA f32 v3 | `cuda32_v3` | Faster, ~1e-7 precision. |
| CUDA f64 v2 | `cuda_v2`, `cuda64_v2` | Linear interpolation. |
| CUDA f32 v2 | `cuda32_v2` | Fastest raw kernel. |
| NumPy | `numpy` | Reference f64 implementation. |
| ONNX | `onnx_cpu` / `onnx_cuda` | In-memory ONNX graph (no export needed). |

`cuda` / `cuda64` alias to `cuda_v3`.  v3 uses Catmull-Rom (matches NumPy exactly).  v2 uses linear.

Backend dict spec with nested resolution via `eval_backend_spec()`:
`{"name": "integrated", "base": "cuda_v3"}`  — `base` is resolved recursively.

### IntegratedBackend (Gram matrix norm)

Pre-computes 56×56 Gram matrices from phsp.  Norm is O(n²) per iteration instead of O(N_phsp × 448).
Base backend handles data NLL separately.  Accessible via:
```python
create_backend({"name": "integrated", "base": "cuda_v3"}, kernel_config)
```

If the base backend's kernel has `compute_gram()`, it auto-uses GPU for Gram matrix computation
(instead of the slow NumPy batched path).  CUDA v2/v3 kernels all have `compute_gram()`.

### CUDA Build

All 4 `.cu` files (v2/v3 × f64/f32) compiled via:
```bash
python -m ampfit.cuda.build
```
Auto-detects `nvcc`, needed `-arch=sm_86`, `-ccbin`, `-allow-unsupported-compiler` flags.

## Architecture

```
x (flat free-param vector)
  → VariableRegistry (names ↔ indices)
  → ParameterConstraint (ck = product of slot parameters per comb)
  → apply_bounds (arctan bijective transform)
  → backend.compute(params, data_handle, norm=None, return_p=True)
      ├─ NumPyBackend   — pure NumPy f64
      ├─ CUDABackend    — CFFI to CUDA C kernels (v2/v3 f64/f32)
      └─ ONNXBackend    — in-memory ONNX Runtime (CPU/CUDA)
  → norm from phsp (batched)
  → NLL: -log(purity·P/norm + (1-purity)·bkg)
  → BFGS fit → Hessian → uncertainties
```

### `compute(params, data_handle, norm=None, return_p=True)`

Unified across all backends.  `return_p=False` → skips per-event P allocation (fast norm).
`return_p=True` (default) → returns per-event P array.

### CK array layout

448 entries total = 8 blocks × 56 base waves.
- Blocks 0–3 (indices   0–223): B0  (g_ls)
- Blocks 4–7 (indices 224–447): B0bar (g_lsbar)

Each block has 4 identical-particle permutations of the 56 base waves.
Parameter names from `get_partial_waves_params()` encode the full chain:
`B->a1(1260)p.pim2_g_ls_0` → first-decay stem `a1(1260)p.pim2`.
`get_ck_map()` returns tuples `(total_name, g_ls_name, sub_g_ls_name, subsub_g_ls_name)`.

## Key Files

| File | Role |
|------|------|
| `run_fit.py` | Full fit with `build_constraints()`. |
| `fit.sh` | One-command pipeline (integrated backend, archive init). |
| `src/ampfit/fitter.py` | `Fitter` class: NLL → BFGS → save/load, uncertainty propagation. |
| `src/ampfit/amp_frac.py` | `AmplitudeFractions.fractions(masks, denominator)` — ratio + gradient uncertainty. |
| `src/ampfit/param_constraint.py` | Constraints: `FixedOverride`, `ScaleTransform`, `NameResolution`. |
| `src/ampfit/backends/` | Backends directory (core + numpy/cuda/onnx/integrated). |
| `src/ampfit/numpy_kernel.py` | Reference kernel, Wirtinger gradients, `compute_Mpp()`. |
| `src/ampfit/backends/integrated_backend.py` | Gram matrix hyper backend. |
| `scripts/calc_fractions.py` | Rx→3π and B→R1R2 amplitude fractions. |
| `scripts/calc_first_decay_fractions.py` | Fractions by first decay (terminal + LaTeX + CSV + PDF). |
| `scripts/calc_sub_decay_ratios.py` | Sub-decay ratios Rx→Xπ / Rx→3π (terminal + LaTeX + PDF, LS labels). |
| `scripts/sub_decay_pdg.py` | PDG comparison table (LaTeX). |

## Key Gotchas

1. **Wirtinger gradients**: ∂Q/∂ck[i] (complex), then `backprop_grad` distributes to slot names.
   For real params (m₀, Γ): `dQ/dx = 2·Re(∂Q/∂z · ∂z/∂x)`.

2. **Masked CK gradients**: `_compute_total_and_grad` zeros `grad_ck[i]` for masked-out entries.
   Without this, disconnected parameters contribute spurious gradients through `backprop_grad`.
   `fun_jac` iterates over ALL keys in `grad_num`/`grad_den`, not just `param_names` — alias
   gradients from `backprop_grad` are propagated through `chain_gradient` to optimizer space.

3. **Constraint pipeline**: `resolve()` = `name_res → fixed_tr → scale_transforms[]`.  `inverse()` = reverse order.
   Each scale is an independent :class:`ScaleTransform` (name + factor) in a list applied in-order.
   Scale is the outermost layer: it multiplies ALL params including fixed ones, matching TFPWA convention
   where scale is at the amplitude product level.  `FixedOverride.chain_grad` REMOVES gradient entries
   for fixed params (so they contribute zero uncertainty).  `ScaleTransform.backward` multiplies
   gradient by the same factor.  The base :class:`Transform` class provides ``forward()``, ``backward()``,
   and optional ``inverse()`` interface for custom parameter-space transforms.

4. **Amplitude fractions with denominator**: `fractions(masks, denominator=)` computes
   `R_i = Σ w·|A_mask_i|² / Σ w·|A_denom|²` with gradient chain accounting for covariance
   between numerator and denominator.  B-level g_ls cancels in the ratio WHEN denominator
   is the same resonance's total (flavor-specific).  Merged denominators (both charges) add
   cross-terms.

5. **identical particle permutations**: 56 base waves × 8 blocks.  Blocks 0–3 are B0, 4–7 are
   B0bar.  Within each flavor, 4 blocks correspond to permutations of identical particles
   (pip1↔pip2, pim1↔pim2).  `bw_order` and `fl_order` differ across blocks.

6. **`rsplit` in `_chain_ranges()`**: Use `config._chain_ranges()` not `config.full_decay._chain_ranges()`.

7. **LS wave labels**: `Decay.get_ls_list()` returns `(L, S)` tuples in order corresponding to
   `_g_ls_N` suffix.  Parity constraint: `(-1)^L = P_parent × P_d1 × P_d2`.  If = +1, L even (0,2,4…);
   if = -1, L odd (1,3,5…).

8. **MergeCP in sub-decay script**: When `merge_cp=True`, ratio uses pi_p (first-charge) base
   indices only to avoid cross-terms from mixing both charges in the denominator.  Values are
   CP-symmetric, so one charge is sufficient.

9. **`save_params()` output**: Saves JSON + `_error_matrix.npy` alongside.  `load_results()`
   auto-detects error matrix.  Constraints saved as `_constraints.json` alongside results.

10. **Fit restart**: `--init results.json` reconstructs x from saved values, allows continuing
    BFGS from checkpoint.

## Testing

```bash
pytest tests/ -v
pytest tests/test_fitter_constraints.py -v
tests/validate_gradients.py    # 3-point numerical gradient verification
```

All 6 backends (numpy, cuda_v2, cuda32_v2, cuda_v3, cuda32_v3, onnx_cpu) pass gradient validation.
ONNX tolerance relaxed to 5e-3.

## Constraints File Format

JSON with keys `fixed`, `same`, `scale`, `bounds`.
- `fixed`: `{slot_name: value}`  (injects constant, param removed from optimizer)
- `same`: `[[alias, canonical, ...]]`  (first element is canonical, rest are aliases)
- `scale`: `{slot_name: factor}`  (multiply resolved value)
- `bounds`: `{name: [lo, hi]}`  (arctan transform applied)
