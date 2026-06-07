# interp_fitter

Reference NumPy implementation of a partial wave analysis fitter with energy-dependent width interpolation, Breit-Wigner propagators, Blatt-Weisskopf form factors, angular basis expansion, and time-dependent mixing (B-meson style).

## Amplitude

```
A_wave = ck · ∏ BW_i(m_i) · ∏ F_j(q_j) · ∑ basis_n(angle) · M_ang[n,w]
```

The full amplitude is split into two CP-conjugate groups for time-dependent analysis.

## Waveform

```
PB    = |ep·A₀ + poq·em·A₁|²
PBbar = |em/poq·A₀ + ep·A₁|²
P     = (1-frac)·(1-ap)·PB + frac·(1+ap)·PBbar
```

## Objective

| Mode | Q |
|------|---|
| No norm | `Q = Σ w·P` |
| With norm | `Q = -Σ w·log(P/norm + bkg)` |

## Installation

```bash
pip install -e .
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```


## Config structure

The `Kernel.__init__(config)` expects a dict with the following keys:

### Interpolation tables

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `gamma_table` | `(n_types, n_int)` | complex | Energy-dependent width lookup table |
| `fl_table` | `(n_types, n_int)` | float | Form factor lookup table |

### Mapping matrices

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `matrix_gamma` | `(n_m0, n_gamma)` | float | Maps gamma components → mass parameters |
| `matrix_ang` | `(nbasis, nwaves)` | complex | Projects angular basis → waves |

### Gamma indexers

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `g0_index` | `(n_gamma,)` | int | Maps each gamma term → g0 parameter |
| `gamma_index` | `(n_gamma,)` | int | Selects mass columns for gamma interpolation |
| `gamma_type` | `(n_gamma,)` | int | Selects row in `gamma_table` per gamma term |
| `gamma_min` | scalar | float | Lower bound for gamma interpolation |
| `gamma_delta` | scalar | float | Bin width for gamma interpolation |

### BW indexers

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `m0_index` | `(n_bw,)` | int | Maps each BW term → m0 parameter |
| `bw_index` | `(n_bw,)` | int | Selects mass columns for BW denominator |
| `bw_gamma_index` | `(n_bw,)` | int | Selects gamma_for_mass column per BW term |
| `bw_order` | `(nwaves × nres,)` | int | Orders BW terms per wave and resonance |

### Form factor indexers

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `q_index` | `(n_fl,)` | int | Selects q columns for form factor interpolation |
| `fl_type` | `(n_fl,)` | int | Selects row in `fl_table` per form factor term |
| `fl_min` | scalar | float | Lower bound for FL interpolation |
| `fl_delta` | scalar | float | Bin width for FL interpolation |
| `fl_order` | `(nwaves × ndecays,)` | int | Orders form factor terms per wave and decay |

### Angular indexers

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `angle_index` | `(n_ang,)` | int | Selects angle columns |
| `angle_k` | `(n_ang,)` | float | Scale per angular term |
| `angle_b` | `(n_ang,)` | float | Offset per angular term |
| `ang_order` | `(nbasis, n_per)` | int | Groups angular terms into basis products |

## ONNX model

Built via `build_onnx_model(config, ...)` in `interp_fitter.onnx_model`.

### Inputs

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `ck_re` | `(nwaves,)` | float32 | Real part of coupling constants |
| `ck_im` | `(nwaves,)` | float32 | Imag part of coupling constants |
| `m0` | `(n_m0,)` | float32 | Mass parameters |
| `g0` | `(n_g0,)` | float32 | Width parameters |
| `time_params` | `(6,)` | float32 | `[γ, Δγ, Δm, poqr, poqi, ap]` |
| `mass` | `(nevt, ndim_mass)` | float32 | Invariant masses |
| `q` | `(nevt, ndim_q)` | float32 | Momentum transfer for form factors |
| `angle` | `(nevt, ndim_angle)` | float32 | Helicity angles |
| `time` | `(nevt,)` | float32 | Decay time |
| `weight` | `(nevt,)` | float32 | Event weights |
| `frac` | `(nevt,)` | float32 | Mistag fraction |
| `bkg` | `(nevt,)` | float32 | Background fraction |
| `norm` | `()` | float32 | Normalisation (only with `with_norm=True`) |

### Outputs

| Name | Shape | Description |
|------|-------|-------------|
| `P` | `(nevt,)` | Per-event probability |
| `Q` | `()` | Objective value |
| `grad_ck_re` | `(nwaves,)` | ∂Q/∂Re(ck) |
| `grad_ck_im` | `(nwaves,)` | ∂Q/∂Im(ck) |
| `grad_m0` | `(n_m0,)` | ∂Q/∂m0 |
| `grad_g0` | `(n_g0,)` | ∂Q/∂g0 |
| `grad_time_params` | `(6,)` | ∂Q/∂[γ, Δγ, Δm, poqr, poqi, ap] |
| `grad_norm` | `()` | ∂Q/∂norm (only with `with_norm=True`) |

### Derived dimensions

| Dim | Source |
|-----|--------|
| `nwaves` | `matrix_ang.shape[1]` |
| `n_m0` | `matrix_gamma.shape[0]` |
| `n_g0` | `max(g0_index) + 1` |
| `n_gamma` | `len(g0_index)` |
| `n_bw` | `len(m0_index)` |
| `nres` | `len(bw_order) // nwaves` |
| `n_fl` | `len(q_index)` |
| `ndec` | `len(fl_order) // nwaves` |
| `nbasis` | `ang_order.shape[0]` |
| `n_ang` | `len(angle_index)` |
| `ndim_mass` | `max(max(gamma_index), max(bw_index)) + 1` |
| `ndim_q` | `max(q_index) + 1` |
| `ndim_angle` | `max(angle_index) + 1` |
