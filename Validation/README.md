# Validation — ampfit vs TFPWA Reference

Backend validation for the B → 4π amplitude analysis fitting framework.

## Reference

TFPWA fit result: `pw_cfit5_td6_fix29/final_params_0.json`
- **NLL:** -29656.14
- **Norm:** 34717.11

## NLL Reproduction

Each backend tested in an isolated process (no GPU context cross-contamination).

| Backend | Norm | NLL | Diff from Ref |
|---------|------|-----|---------------|
| TFPWA ref | 34717.11 | -29656.14 | — |
| **cuda_v3** (f64, Catmull-Rom) | 17359.01 | -29624.51 | +31.64 |
| **cuda_v2** (f64, linear) | 17358.93 | -29624.47 | +31.67 |
| **cuda32_v3** (f32, Catmull-Rom) | 17359.02 | -29624.51 | +31.64 |
| **cuda32_v2** (f32, linear) | 17358.94 | -29624.47 | +31.67 |
| **onnx_cuda** (GPU) | 17352.20 | -29624.59 | +31.55 |
| **integrated** (base=cuda_v3) | 17343.17 | -29654.57 | +1.57 |
| **integrated** (base=cuda32_v3) | 17342.81 | -29655.27 | **+0.88** |

### Key observations

- All four CUDA kernels agree within 0.04 on NLL and within 0.5 of TFPWA/2 norm.
- **Integrated backend** norm uses Gram matrices and is time-independent, which gives a different norm and a closer NLL (+1.6 for f64 base, **+0.88** for f32 base).
- Always run in separate processes when switching backends — `__del__` alone does not fully reset GPU context between different kernel versions.

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `repro_nll_ampfit.py` | NLL reproduction vs TFPWA reference for any backend | `python repro_nll_ampfit.py --backend <name>` |
| `compare_final.py` | Per-wave K factor, ck, A/Abar, P comparison (112 waves) | Needs REF_DIR reference data |
| `check_load.py` | Verify data loading pipeline | `python check_load.py` |
| `repro_nll.py` | Older NLL reproduction (deprecated) | — |
| `roundtrip.py` | Parameter round-trip test | — |
