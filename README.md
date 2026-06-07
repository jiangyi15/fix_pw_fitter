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


## config

```
# config.yml
decay:
    # a -> r + c or y + d, with perporities {p_break: True}
    A: [
        [R, C, {p_break: True}],
        [Y, D, {p_break: True}, {model: modelname}],
    ]
    R: [[B, D]] # R -> B +D
    Y: [B, C] # when only one decay, we can using single list

particle:
    $top: A
    $finals: [B, C, D]
    A:
        J: 0
        P: -1
        mass: 5.
    R: [R1, R2] # R can be replace to R1 and R2
    Y: {J: 12, P: -1, mass: 3, model: BW} # or direcly with perproiteis
    R1: {J: 0, P: 1, mass: 3, model: BW }
    R2: {J: 1, P: -1, mass: 3, model: BW}
    B: {J: 0, P: -1, mass: 0.1}
    C: {J: 0, P: -1, mass: 0.1}
    D: {J: 0, P: -1, mass: 0.1}
```

based on it we can buit the strucure as
```
DecayGroup: [[A -> R1 + C, R1 -> B +D], ...]
    DecayChain: [A -> R1 + C, R1 -> B +D]
        Decay: A -> R1 + C
        Decay: R1 -> B +D
    ...
```


