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

## Helicity amplitude formula

For a single decay vertex ``A → B + C`` with orbital angular momentum *L* and total
spin *S*, the angular part of the helicity amplitude is:

```
T^{L,S}_{λ_A,λ_B,λ_C}(φ,θ) = √((2L+1)/(2J_A+1))
    ×  ⟨J_B, λ_B, J_C, -λ_C | S, δ⟩         CG coefficient
    ×  ⟨L, 0, S, δ | J_A, δ⟩                CG coefficient
    ×  e^{i·λ_A·φ}                           azimuthal phase
    ×  d^{J_A}_{λ_A,δ}(θ)                    Wigner d-function
```

where ``δ = λ_B − λ_C``.

### Wigner-d half-angle expansion

```
d^{J}_{λ,δ}(θ) = Σ_k  C_k · sin(θ/2)^{sp_k} · cos(θ/2)^{cp_k}
```

Each term coefficient ``C_k = sign × p × √r / d`` (exact integers, gcd-reduced)
comes from the factorial ratio:

```
C_k = sign · √[(J+λ)!(J-λ)!(J+δ)!(J-δ)!]  /  [k!(J-λ-k)!(J+δ-k)!(λ-δ+k)!]
```

### Azimuthal phase

```
e^{i·λ_A·φ} =
    cos(λ_A·φ)                     if λ_A = 0
    cos(|λ_A|·φ) + i·sin(|λ_A|·φ)  if λ_A > 0
    cos(|λ_A|·φ) − i·sin(|λ_A|·φ)  if λ_A < 0
```

Imaginary parts are encoded directly in the coefficient: ``coeff = i·√2/2``.
SymPy handles ``I·I = −1`` automatically in cascade multiplication.

### Cascade

For a decay chain, vertex amplitudes are **multiplied**:

```
T_{chain} = T_{v₀} × T_{v₁} × T_{v₂} × …
```

Intermediate-resonance helicities are **summed over** (matched via
``λ_B(vᵢ) = λ_A(vᵢ₊₁)``). The output is organized by:

```
helicity key = "λ_root, λ_final₁, λ_final₂, …"
LS key       = "L₁,S₁; L₂,S₂; …"
```

### Mapping to Kernel config

Each (helicity, LS) combination produces ``AmpTerm``\s that fill the
``matrix_ang`` coefficients in the Kernel:

```
AmpTerm(coeff, factors=[
    Factor("theta_i", "cos"|"sin", k),     →  cos(k·θᵢ/2) or sin(k·θᵢ/2)
    Factor("phi_j",   "cos"|"sin", k),     →  cos(k·φⱼ/2) or sin(k·φⱼ/2)
])
```

| Kernel field | Source |
|-------------|--------|
| ``angle_k``, ``angle_b`` | ``k`` from ``Factor.k``, ``b=0`` or ``b=−π/2`` for sin |
| ``ang_order`` | Groups ``Factor``\s into basis products per wave |
| ``matrix_ang`` | ``AmpTerm.coeff`` times ``CG`` × ``Wigner-d`` × ``LS-factor`` |

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
    Y: {J: 1, P: -1, mass: 3, model: BW} # or direcly with perproiteis
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


