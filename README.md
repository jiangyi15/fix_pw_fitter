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
