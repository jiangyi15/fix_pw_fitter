# PWACG benchmark — ψ(3686) → φ K⁺K⁻

A like-for-like comparison with

> **PWACG: Partial Wave Analysis Code Generator supporting Newton-conjugate
> gradient method** — X. Dong, Y.-C. Sun, C.-C. Pan, A.-Y. Cheng, A.-B. Wang,
> H. Cai, K. Zhu, arXiv:[2403.09225](https://arxiv.org/abs/2403.09225) (v2,
> 2025-11-17); code **[github.com/caihao/PWACG](https://github.com/caihao/PWACG)**
> (MIT, JAX).

Both sides are in this directory: **ours** (`config_gen.yml` + `bench.py`), and
**theirs** (`pwacg_ref/`, their generated JIT module driven by a small runner,
with the vendored clone kept byte-identical behind a `jax.config` shim).

The model is the paper's: **ψ(3686) → φ + R**, `φ → K⁺K⁻`, `R → K⁺K⁻`, i.e. four
kaons in two pairs, with **six** resonances — the paper's Table 1 (f₀(980) is the
one entry that cannot be used as printed; see the note under it):

| R | mass (GeV) | width (GeV) | F_i (%) | components |
|---|---|---|---|---|
| f₀(980) † | 1.022 | 0.209 | 42.89 | 2 |
| **f₂(2550)** | 2.545 | 0.303 | 30.95 | 5 |
| f₂′(1525) | 1.522 | 0.091 | 30.27 | 5 |
| f₀(1710) | 1.672 | 0.169 | 8.97 | 2 |
| f₂(1270) | 1.302 | 0.189 | 3.98 | 5 |
| f₂(2150) | 2.150 | 0.143 | 2.52 | 5 |

† **f₀(980) is the one row not used as printed.**  The table lists an *effective*
mass/width, which is how a Flatté is often summarised; a Flatté has no single
width, so the configs carry instead the reference's own **fitted Flatté
parameters** — `mass 0.9794812`, `g_ππ 0.0124622`, `g_KK 0.1067862` (`rg =
8.570188`) from its `config/pwa_info_kk.json` and its generated arg slicing —
mapped into our basis as `g_i = g_i^their/m₀`.  Every other row is the table's
value verbatim.

24 amplitude components in total, and our wave axis carries exactly the same 24
(`2, 5, 2, 5, 5, 5` per chain — `make_data.py` asserts it).  The reference's repo
config calls the sixth module `phif2_2340` with its *fitted* 2.548/0.324; the
paper's table says f₂(2550) at 2.545/0.303, which is what is used here.

The top declares **`spins: [-1, 1]`**, i.e. **two** polarisations, matching the
paper's `½ Σ_{μ=1,2} A^μ A*^μ`; a bare `J: 1` gave `n_proj = 3` and put a third
polarisation into our cost that the reference does not compute.

**What the paper benchmarks is not speed.**  Its §3 measures *convergence*:
10⁴ MC events, 300 random starting points, and the fraction reaching the global
optimum — **NTCG+AutoDiff 6.67 %**, iMinuit+AutoDiff 1.33 %, iMinuit+FDM 0.33 %
— plus a memory wall (3×10⁶ integration events exceed the 16 GB V100).  There is
no "N× faster than X" claim in the paper at all.  Reproducing that table would
compare optimisers, not frameworks, so this directory does the thing the paper
does *not*: it times the **per-iteration value + gradient** both sides compute,
at the paper's sample size and at the integration sizes it warns about.

**The headline — ours against PWACG, one per-iteration value + gradient.**  One
GPU, the same events, the same model content on both sides (six resonances, 24
amplitude components, 2 polarisations, **59 free parameters** each), the
per-event objective, and none of their Newton-CG/Hessian machinery in the
timing.  Ours is one `Fitter.get_nll`; theirs is the four jitted calls their
optimiser makes per step (`thread_likelihood` + `thread_grad_likelihood`), XLA
compilation excluded.  **ms — Mev/s**, both over data *and* phase space and both
value + gradient:

| data + phsp | ours `s0e2` | ours `s0e5` | theirs, jac path | ours vs theirs |
|---|---|---|---|---|
| 10 k / 10 k | 1.0 — 19.2 | 1.0 — 20.5 | **1.059 — 18.9** | parity |
| 100 k / 100 k | 4.4 — 45.1 | **3.8 — 52.7** | **2.811 — 71.1** | **1.35× slower** (`s0e5`) |
| 1 M / 1 M | 38.9 — 51.4 | **32.6 — 61.3** | 39.5–40.5 — 50.6 | parity → **1.2× faster** (`s0e5`) |

Read it as: **at the paper's own 10⁴-event size the two are level; at 100 k
theirs is ahead; at 10⁶ ours is** — and their advantage sits where their
per-event cost happens to dip, a dip whose cause is not established
(§*Against the reference*).

These numbers are repeated section by section (our-side sweep, the reference
comparison, the one-call bound, the CPU family) because each carries a different
caveat — but they come from **two sweeps**: min-of-5→10 for the comparison
tables and min-of-3 for the per-kernel sweep, which is why `s0e2`/`s0e5` read
19.2/20.5 here and 19.3/20.7 in §*The rest of our kernels*. That is the ±1 %
run-to-run spread measured below, not a disagreement; a re-measurement means
editing all of them together.  The same ordering holds against a stricter bound, the **one-call**
form that differentiates their combined objective instead of summing two
per-sample gradients (its gradient checked against their own assembly, rtol
1e-10):

| one-call bound | 10 k | 100 k | 1 M |
|---|---|---|---|
| theirs (combined `L`) | **0.691 ms** | **2.422 ms** | **34.388 ms** |
| their four-call path, same run | 0.965 | 2.789 | 39.524 |
| ours `s0e2` / `s0e5` | 1.0 / 1.0 | 4.4 / 3.8 | 38.9 / 32.6 |

Memory is the other half of the trade: ours peaks at **0.17 / 0.60 / 1.46 GiB**
at 10 k / 1 M / 3 M (data **and** phase space), while their own §3 wall is 3×10⁶
integration events on a 16 GB V100.

The **CPU family** (`cpu_flat_pwa_gen` and its two cached copies) is in
§*The rest of our kernels*: 2.2 / 7.7–8.5 ms at 10 k / 100 k, and **not
rankable at 1 M** on this box (32.6–84.2 ms across seven identical runs) — so
it contributes no ratio, only those two points.

Everything below is the evidence: the model and where it does *not* match
theirs, the per-kernel tables for our whole family, what exactly is timed on
each side, their own `Control.run`, and the memory series.

## The model, and why it is ONE config after all

The reference's amplitude is

```
A(ξ) = Σ_w  c_w · [ BW_φ(m₁₂) · BW_R(m₃₄) ] · U_w(ξ)
```

with the per-event `U_w` tensors precomputed and resident (`phif0` 2 components,
`phif2` 5, shared by the six modules — 24 amplitude components) and the **φ
mass/width hard-coded** at 1.02 / 0.004.

That φ factor looks like a model component, and keeping it costs the whole fast
kernel family: a chain with two resonances is `n_res = 2`, and `s0e*`, `s0c*`,
`1bw`, `s0f`, `s0g` all refuse it.  It is not a model component, though — it is
a **weight**:

```
log|φ·F|² = log|φ|² + log|F|²
```

φ is *parameter-free*, so in the DATA term its log is a constant (`−Σ_e log|φ_e|²`,
independent of every fitted parameter).  In the **normalisation** it is not a
constant but a fixed per-event *weight* on `|F(c)|²`.  `set_phsp` normalises the
weights to `Σw = 1`, so the norm is the **weighted mean**

```
norm(c) = Σ_e w_e·|F_e(c)|²         with  w_e = |φ_e|² / Σ|φ|²
```

Folding `|φ(m₁₂)|²` into the phase-space weights and dropping the φ from the
amplitude therefore leaves the same likelihood surface **up to a parameter-free
constant**, and it leaves the chain at `n_res = 1`, where the fast tiles apply.
`make_data.py` writes that sample (`data/phsp_wphi.npz`) and `config_gen.yml`
reads it; both assert the config's φ equals the one the weights were built with.

Verified through the real backends, not argued (φ kept vs φ as weights, same
events, same 59-parameter point, 10 k + 10 k):

| | `config_pwacg.yml` (φ in the amplitude) | `config_gen.yml` (φ as weights) |
|---|---|---|
| kernel | `cuda_flat_pwa_gen2`, `n_res = 2` | `cuda_flat_pwa_gen_s0e2`, `n_res = 1` |
| NLL (`numpy_flat`, fp64) | **+40 583.0517** | **−9 956.2436** |
| NLL (the two kernels) | +40 588.838 | −9 950.452 |
| ΔNLL | — | **+50 539.30 = 5.0539 per event** |
| gradient | — | max\|Δg\|/\|g\| = **6.5e-13** (fp64, uncached) / 3.1e-6 (fp32 cache) — see below |

**The constant is not `−Σ log|φ|²`** — that is only 1.7 % of it.  With the norm
normalised to `Σw = 1`:

```
ΔNLL = N·[ log⟨|φ|²⟩_phsp − ⟨log|φ|²⟩_data ]
     = 10 000·[ log(143.847925) − (−0.085173) ] = +50 539.30    (measured +50 539.30)
```

both terms parameter-free and both measured on the shipped 10 k sample
(`⟨|φ|²⟩_phsp` is the flat phase space's mean of `|BW_φ(m₁₂)|²`, `⟨log|φ|²⟩_data`
the data's), and the right-hand side closes to **~1e-8 relative** — 2.7e-8
against these 7-digit values, 6.7e-10 against the unrounded sample means, i.e.
the decomposition is the identity.  The norm ratio says the same thing:
`norm_pwacg/norm_gen = 143.847925`, matching `⟨|φ|²⟩_phsp` to seven digits.  (Both
figures were re-measured after the reference backend moved to the kernels'
Catmull-Rom interpolation; the earlier `143.8479` / `−0.088` pair was the fp32-cache
backend's and did not close — it evaluates to +50 567.6, not +50 539.3.)  The
constant is a property of the **sample**, not a universal number: three sample
generations gave 146.8, 157.2 and 143.8.

**The identity is exact** — and the two floors above it are backend properties,
not the model.  Measured on the same events and the same `x`, by backend:

| backend | what it is | `max|Δg|/|g|` | median component |
|---|---|---|
| `cuda_flat_pwa_gen` | fp64, no cache, Catmull-Rom | **6.5e-13** | 8.6e-13 |
| `cuda_flat_pwa_gen2` / `s0e2` | fp64 compute, **fp32 `float2` cache** | 3.1e-6 | 4.2e-6 |
| `numpy_flat` | fp64, Catmull-Rom (like the kernels) | **9.4e-13** | 8.4e-13 |

So on a kernel with no fp32 cache the substitution is exact to machine
precision — both fp64 backends land at ~1e-12 whatever their provenance — and
the φ's own line shape is not a source of error: the
kernels interpolate with **Catmull-Rom** (`gps02_cr1d`, 4-point Hermite with
central-difference tangents), which reproduces a quadratic *exactly* in the
interior — verified on the shared `−m²` row at `s = 0, 0.13, 0.5, 0.87`,
`1e-16` relative.  (The first and last grid bins are exact for neither scheme:
the edge clamp duplicates a neighbour instead of taking a one-sided tangent,
which costs `8e-5` / `2.6e-4` in `m²` there — the kernels' own behaviour.  No
event in this benchmark is within 70 bins of either edge: the φ peak sits at
`m₁₂ ≈ 1.02` on a grid starting at `0.9774`.)

The remaining floor is the **fp32 cache** (`float2`, the flat family's documented
floor): it appears as 3.1e-6 whenever either side runs on `gen2`/`s0e2`, and it is
the same floor the family's own gates use.  `numpy_flat` used to be the loosest of
the three because it evaluated rows with two-point **linear** interpolation while
every kernel uses Catmull-Rom (`gps02_cr1d`, `gps0_cr1d`, `numpy_pwa` too); it now
uses Catmull-Rom as well — the same four-point Hermite with the same edge
clamping — so the fp64 reference and the fp64 kernel agree to 9.4e-13 and 6.5e-13
respectively.  On a quadratic row the chord error that used to show up here is
`Δ²/4 = 7.7e-8` in `m²`, which was worth up to 1.9e-5 relative on `|BW_φ|²` for
events sitting on a narrow peak (`|D|² = 3.85e-5`, since `m₀² − m² ≈ m₀Γ`).

`config_pwacg.yml` stays in the directory as that cross-check: it keeps the φ in
the amplitude, refuses the fast kernels, and exists to prove the weight is not an
approximation.  Dropping the φ *without* the weight is what would change the
model: measured on the same 10 k sample, `cos(g, g_φ)` falls to **+0.75** instead
of 1 (6.5e-13 on the uncached fp64 kernel) — the normalisation is then
unweighted, so the φ stops acting as
a weight at all.

The wave count is matched deliberately: **24 on both sides**.  The reference
formalism has no identical-particle / CP block expansion, so neither config
declares symmetry; declaring it would quadruple the wave axis (`n_perm = 2!·2!`)
and time 96 waves against their 24 — strictly more physics, and it would flatter
neither side.  The symmetrised variant exists only as the appendix
`config_gen_sym.yml` (96 waves), which states what the symmetry costs.

### How closely the model matches — and where it does not

**Matched:** the topology; the six resonances with the paper's Table 1
masses/widths; **24 amplitude components with the same per-resonance split**
(2/5/2/5/5/5); **2 polarisations**; and the per-event objective.

**The parameter decomposition is the SAME, and it was checked in their GENERATED
code**, not inferred from the templates.  `rendered_scripts/fit_object_kk.py`
slices the 59-vector explicitly:

```python
phi_mass, phi_width = 1.02, 0.004          # FIXED, not arguments
kk_f980_mass, kk_f980_g_kk, kk_f980_rg     # 3 lineshape args
kk_f980_const  = np.array([0.1, args[3]])  # 2 components, the FIRST fixed at 0.1
kk_f0_const    = np.array([args[7], args[8]])
kk_f1270_const = np.array([args[13] ... args[17]])            # 5
kk_f2_const    = np.array([args[29] ... args[43]]).reshape(-1, 5)   # 3 modules x 5
```

so theirs is **23 free complex couplings (46 real — the 24th, `f980_c1`, is the
fixed gauge) + 13 lineshape parameters = 59**.  Ours is **23 complex couplings
(46 real: a head per chain times its per-(L,S) weights) + 13 lineshape
parameters = 59**.  Same split, same total, and both carry 24 amplitude
components with one degenerate overall scale.

**Line shapes.**  `config_gen.yml` and `config_pwacg.yml` use the reference's
own, from `extra_amp.py`, and `check_model.py` compares each denominator with
their generated formula (model-driven: it reads the registered model, so editing
`extra_amp.py` or mistyping a coupling moves the check).  The only difference
between the two configs is where the φ lives:

| resonance | theirs | ours (both configs) |
|---|---|---|
| f₂(2550), f₂′(1525), f₀(1710), f₂(2150) | `BW`: `1/(m² − s − i·m·w)` | **`BW`**: `gamma(m) = 1` → `m₀² − m² − i·m₀·g₀` — the same function |
| f₀(980) | `flatte980`: `1/(m² − s − i(g_ππ·ρ_ππ + g_KK·ρ_KK))`, `ρ = √\|1 − 4m²/s\|`, `g_KK = rg·g_ππ` | **`PWACG_flatte980`**: `γ = [ρ_ππ, ρ_KK]` (real on both sides of threshold), couplings `g_i = g_i^their/m₀` — verbatim from `extra_amp.py` |
| f₂(1270) | `flatte1270`: `gr/((s−m²)+i·g₁)`, `g₁ = g₁(m²)·(q²)^{2.5}/b₂(s)`, `q² = s/4 − 0.0194792`, `b₂ = q²(q²+0.1825)+0.033306` | **`PWACG_flatte1270`** — their `b₂` form, verbatim |

**The appendix (`config_gen_sym.yml`) is the one that substitutes built-ins** —
`FlatteC` with the complex breakup momentum (imaginary below K-K threshold, where
theirs stays real) and `BWR` L=2 instead of their `b₂` polynomial — because it
exists only to price the symmetry.  That
substitution is a real amplitude difference (the ratio flips from `+2` above
threshold to `−2i` below it), which is one more reason nothing from that file is
comparable.

**The denominator sum has a different SHAPE, and that is not cosmetic.**  A BW
contributes ONE channel term and a Flatté TWO, on both sides — but where the sum
lives differs.  Theirs evaluates one closed-form complex propagator per
(wave, event) — `flatte980` costs one extra `ρ` inside it — while ours puts each
channel on the basis axis and contracts it:

```
den_names = [f0980_s0, one, f0980_g0_t, f0980_g1_t,      # the Flatté: 2 channels
             f22550_s0, f22550_width_t, ...  ]            # each BW: 1 channel
n_basis 14 (6 deduped interpolation slots), n_wave 24
```

so the two-channel resonance is 2 of our 14 basis columns and 2 of their `ρ`
evaluations, and the per-event denominator work is a basis contraction on our side
against 24 closed-form propagators on theirs.  Neither count alone says who does
more; both are stated here so the reader is not invited to infer it.

**Masses and widths: free on both sides, with one setup difference.**  The φ is
fixed — hard-coded `np.array([1.02])` / `np.array([0.004])` in their generated
module, pinned here by `constrains.fix_var` — so it is not a parameter on either
side.  Every other resonance floats, mass and width both, and f₀(980) through the
Flatté (mass, g_ππ, g_KK) rather than a plain width: **13 lineshape parameters,
all free**, on both sides, and they are exactly the 13 in `x = 59`.

Where it differs is **bounds**: we apply them (`mass_width_bounds` from
`apply_constrains` gives masses ±1 GeV around their defaults and
widths ∈ [0.01, ≈0.2–0.28]), while their `Control.run` defines an `mw_range`
table and then *never uses it* — `minimize(...)` is called without `bounds`, and
the penalty that would implement it is hard-wired off:

```python
mass_index  = onp.array([0, 5, 11, 23, 24, 25])      # defined,
width_index = onp.array([6, 12, 26, 27, 28])         #  ...
mw_range    = onp.array([[0.98, 10.0], ...])         #  never referenced
...
step_function = 0.0
return -np.sum(...) + 100.0 * step_function
```

So their fit is formally unbounded and ours is not.  That changes a *fit*, not
the per-iteration cost the tables measure.

**Also not matched**, and no event-by-event intensity comparison is claimed:

| | theirs | ours | why |
|---|---|---|---|
| φ propagator | `self.BW(1.02, 0.004, Sbc_phi)` multiplied into every wave (`einsum("j,ij->ij", phi, f)`) | **as an MC weight**, exactly (see above): `config_gen.yml` drops it from the amplitude with `|φ|²` in the weights; `config_pwacg.yml` keeps it in the amplitude as the cross-check | — |
| barrier radius | Blatt–Weisskopf `Q0 = 0.197321/R`, `R = 1` | `fl_table` with `d = 3.0` | different radius and parameterisation |
| amplitude basis | covariant tensors (Zou–Bugg) | helicity / (L,S) | equivalent spaces, different bases: no coefficient-by-coefficient map, so the counts match and the individual amplitudes do not |

So: same resonances, same component count, same parameter count, the same line
shape on four of six and the same channel axis on the fifth.  The intensities are
still not equal (φ, f₂(1270)'s width form, the barrier radius, the basis), and
the timing claim is a **cost comparison at matched wave content** — not a
validation of either amplitude.

## Files

| file | purpose |
|---|---|
| `config_gen.yml` | the **benchmark** model: six resonances / 24 waves, their line shapes, φ folded into the phase-space weights |
| `config.yml` | the same model on the plain `pwa` family (`amp_model: pwa`) |
| `config_gen_sym.yml` | **appendix**: `identical_particles` declared (96 waves = 4 × 24); also keeps the built-in line shapes and the unweighted phase space, so it is the *other* variant too — never comparable |
| `config_pwacg.yml` | the **cross-check**: φ kept in the amplitude (`n_res = 2`), proving the weight identity |
| `extra_amp.py` | the two line shapes this framework does not ship (`PWACG_flatte980`, `PWACG_flatte1270`), registered on import |
| `check_model.py` | verifies every denominator against the reference's GENERATED formulas |
| `make_data.py` | flat four-body phase space, kernel arrays, and the reference's `Momentum_kk*.npz` contract |
| `make_data_from_ref.py` | the same conversion starting from the **reference's published sample** instead of a generated one |
| `bench.py` | ms/iteration, Mev/s, NLL, `x`, kernel shape for one backend |
| `mem.py` | peak device memory vs sample size, one child process per size |
| `pwacg_ref/` | **their side**: `BUILD.md` (the recipe and every trap), `jax_legacy_config.py`, `stubs/`, `bench_ref.py` (per-evaluation timings), `fit_ref.py` (their own `Control.run`) |

## Running

```bash
python make_data.py --n-data 10000 --n-phsp 10000      # 10k / 100k / 1M / 3M all work
python bench.py --backend cuda_flat_pwa_gen_s0e2 --max-events 10000 --repeat 10 --vary
python bench.py --backend fixed_coeff --base cuda_flat_pwa_gen_s0f --mask --fix-mw --max-events 10000
python bench.py --backend cuda_flat_pwa_gen_s0e2 --rsize 17        # the GROUPED objective (not comparable)
python mem.py --backend cuda_flat_pwa_gen_s0e2 --sizes "10000,1000000,3000000"

# their side (needs the clone; see pwacg_ref/BUILD.md)
python pwacg_ref/bench_ref.py --pwacg <clone> --npz data/Momentum_kk.npz \
    --mc data/Momentum_kk_phsp.npz --slices 20          # per-evaluation, jac path
python pwacg_ref/fit_ref.py   --pwacg <clone> --maxiter 3   # their own Control.run
```

`make_data.py` writes, from the **same** four-momenta:

* `data/data.npz`, `data/phsp.npz` — kernel format (`mass`/`q`/`angle`);
* `data/Momentum_kk.npz`, `data/Momentum_kk_phsp.npz` — the reference's
  contract: keys `Kp`, `Km`, `Pip`, `Pim`, each `(n, 4)` with columns
  **(px, py, pz, E)**, float64, same event order.  `Kp`/`Km` are the φ pair,
  `Pip`/`Pim` the R pair (the π names are the reference's inherited misnomer).

The phase space is the exact flat four-body density of the 2+2 cascade
(`dΦ₄ ∝ p_top(M; m₁₂, m₃₄)·p₁₂(m₁₂)·p₃₄(m₃₄)`, rejection at ≈8 % acceptance,
isotropic two-body decays boosted into the top frame).  Grids and sampled ranges
are asserted and printed: **mass [0.9774, 3.2024], q [0, 1.7322]** at
`n_interp = 4000`; the 3 M sample spans mass [0.987, 2.693], q [0.002, 1.553].
The mapping to the reference contract is pinned by exact equalities
(`M(Kp+Km) ≡ mass[:,0]`, `M(Pip+Pim) ≡ mass[:,1]`, `Σp ≡ M_ψ`) rather than a peak
position — a **flat** sample has no φ peak by construction.

## Measured — our side

Min-of-5 to 10, `--max-events N` = N data **and** N phase-space events, matched
config (24 waves, **2** polarisations, `n_res = 1`), `--vary` (every free slot
moves each iteration, so no value-keyed cache can be flattered).  `x = 59` — 23 free complex couplings (46 real) + 13 lineshape
parameters, the paper's count exactly; `x = 48` with `--fix-mw` (the Flatté's two
couplings are not mass/width and stay free).

The objective is the **per-event** one (`resolution_size = 1`), because that is
what the reference computes: `−Σ ln(P/norm)`.  The grouped log-sum
(`--rsize 17`) is a *different objective* and is reported separately below, never
in the comparison.

| kernel | x | 10k/10k | 100k/100k | 1M/1M | 3M/3M |
|---|---|---|---|---|---|
| `cuda_flat_pwa_gen_s0e2` (production tile) | 59 | **1.0 ms** — 19.2 | **4.4 ms** — 45.1 | **38.9 ms** — 51.4 | **115.9 ms** — 51.8 |
| `cuda_flat_pwa_gen_s0e5` (denominator dedup) | 59 | **1.0 ms** — 20.5 | **3.8 ms** — 52.7 | **32.6 ms** — 61.3 | **97.0 ms** — 61.8 |
| `cuda_flat_pwa_gen_s0g` | 59 | 1.2 ms | — | — | — |
| `cuda_flat_pwa_gen2` (general leaf, for contrast) | 59 | 1.1 ms | — | — | — |
| `fixed_coeff --base s0f --mask --fix-mw` (22/24 cached — the Flatté's two couplings are not mass/width) | 48 | 0.9 ms | — | — | — |
| `cuda_flat_pwa_gen_ampcache` (**coeff pinned — NOT a fit**) | 48 | 0.5 ms | — | — | — |

The sample was regenerated once while these numbers were being corrected (same
generator, same seed): the 10 k cells were re-measured on the current file, the
100 k+ cells on its predecessor.  Per-event work is unchanged, and the two
generations agree at 10 k to 3 % — but the two are not the same events, so an
absolute NLL is only comparable within one row's run.

Run-to-run spread on an otherwise idle box, min-of-10 per call: the GPU rows
above reproduce to **±1 %** (3 M: 115.9 and 116.0 in two runs; 1 M `s0e5`: 32.6
repeatedly).  The **CPU rows do not** — see the note under the kernel table.

Mev/s over data + phase space (`s0e2`): 19.2, 45.1, 51.4, 51.8; `s0e5`:
20.5, 52.7, 61.3, 61.8.  The 10 k cell is per-call-overhead bound.  `ampcache` is
labelled because `grads["coeff"] == 0` by contract: its 0.5 ms is a lower bound
and never a fit iteration.

**The cross-check (`config_pwacg.yml`, φ kept in the amplitude → `n_res = 2`, `gen2`):**

| kernel | x | 10k/10k | 100k/100k | 1M/1M | 3M/3M |
|---|---|---|---|---|---|
| `cuda_flat_pwa_gen2` | 59 | 1.6 ms | 10.5 ms | 99.2 ms | 296.7 ms |

20.2 Mev/s at 3 M, against 51.8 for the matched config's tile — the whole
difference is that `gen2` walks the general two-factor denominator with no
per-wave cache.  Peak memory 0.18 / 1.26 / **3.42 GiB** at 10 k / 1 M / 3 M
(the tiles need 1.46 GiB at 3 M).

With the **grouped** objective (`--rsize 17`, a different NLL — not comparable
with the reference) the same tile gives 1.0 / 4.5 / 37.5 ms at 10 k / 100 k / 1 M.

## Against the reference

The comparable row is **free couplings, per-event objective, one data pass + one
integration pass, value + gradient, the same events on both sides**.  That is one
`Fitter.get_nll` against their **jac path** — `thread_likelihood` +
`thread_grad_likelihood`, i.e. the value and the gradient of the data term and of
the MC term.  Nothing of their Newton-CG/Hessian machinery is in this row.

**The timed quantity is the same on both sides, and it is checked, not assumed:**

| | ours (`bench.py`) | theirs (`bench_ref.py`) |
|---|---|---|
| call | `Fitter.get_nll(x)` once per repeat — one call returns the value and the gradient | their **jac path**: `jit_data_likelihood_kk` + `jit_grad_data_likelihood_kk` + `jit_mc_likelihood_kk` + `jit_grad_mc_likelihood_kk`, which is exactly what `thread_likelihood` + `thread_grad_likelihood` invoke |
| returns | `(scalar, gradient)`; asserted `ndim(value) == 0` and `gradient.shape == (len(x),)` = **(59,)** | value and gradient from separate jitted calls; gradient asserted `shape == args.shape` = **(59,)** |
| covers | the phase-space normalisation **and** the data NLL | the MC integral term **and** the data term |
| Hessian / HVP | never computed | never called — the HVP path is their *Newton-CG* optimiser, a different algorithm, and is exercised separately (below) |
| excluded | — | XLA compilation, first call only |

Their value and gradient are **separate jitted calls**, so each sample's forward
pass runs twice per iteration — that is their code, not a harness choice:

```python
# their thread_likelihood            # their thread_grad_likelihood
Σ data_lh + 100·log(Σ mc_lh/1000)    Σ data_grad + 100·Σ mc_grad / Σ mc
```

**A one-call path has to differentiate the COMBINED scalar, not sum two
per-sample gradients.**  Two `value_and_grad`s (one per sample), added up, give
`Σdata_grad + Σmc_grad` — which is **not** `∇L`, because the `100·log(Σmc)` chain
factor `100/Σmc` is missing — that sum is the gradient of a different scalar.
The correct one-call form is

```python
def combined(args):  return data_likelihood_kk(args) + 100.0*log(mc_likelihood_kk(args)/1000.0)
jax.jit(jax.value_and_grad(combined))
```

and `bench_ref.py` **checks it is their gradient**, not merely a cheaper one: the
taken gradient is compared with their own assembly `Σdata_grad +
100·Σmc_grad/Σmc` (rtol 1e-10), and the value it returns is their `Control.run`
starting FCN to the digits they print (`−89182.155945` against `−89182.1559446709`).

Measured (min-of-5→7, same runs): **0.691 ms at 10 k**, against 0.965–1.002 ms for
their four-call path — the bound is 1.4× cheaper than their code, because it
replaces two forward passes per sample with one forward + one backward.

The dimension checks are in the code: `bench_ref.py` asserts the scalar/gradient
shapes of each call, and `get_nll`'s return is asserted in `bench.py`'s contract
(`(float, ndarray of len(x))`).  "Value + gradient, not value alone, no Hessian"
is therefore a property of the harness rather than a claim in prose.

| size | ours `s0e2` | ours `s0e5` | theirs, **jac path** | best ratio |
|---|---|---|---|---|
| 10 k / 10 k | 1.0 ms — 19.2 | 1.0 ms — 20.5 | **1.059 ms** — 18.9 | parity |
| 100 k / 100 k | 4.4 ms — 45.1 | 3.8 ms — 52.7 | **2.811 ms** — 71.1 | **1.35× slower** (`s0e5`) |
| 1 M / 1 M | 38.9 ms — 51.4 | 32.6 ms — 61.3 | **39.5–40.5 ms** — 50.6 | parity to **1.2× faster** (`s0e5`) |

Their four calls at 10 k summed to 1.002 ms (the per-call split is not quoted —
it was measured in an earlier run whose total was 1.001, i.e. the same to 0.1 %);
the 100 k and 1 M columns are the same four calls summed.  Run-to-run: their 10 k and 100 k cells
reproduced exactly across runs (1.002 and 2.811 ms), the 1 M cell gave 39.512 and
40.454 ms, i.e. **±2.4 %** — which is why that row is quoted as a range.  Against
`s0e2` the 1 M row is parity; the `s0e5` column is the one to quote there.

The four columns of "theirs" are the data value, data gradient, MC value and MC
gradient — the calls their optimiser makes per step, and the figures to quote
because they are what runs.  The correct **one-call** bound (differentiating the
combined objective, with its gradient checked against their assembly) is
measured at all three sizes:

| one-call bound (combined L) | 10 k | 100 k | 1 M |
|---|---|---|---|
| theirs | **0.691 ms** | **2.422 ms** | **34.388 ms** |
| their four-call path, same run | 0.965 | 2.789 | 39.524 |
| ours `s0e2` / `s0e5` | 1.0 / 1.0 | 4.4 / 3.8 | 38.9 / 32.6 |
| ratio, ours vs the one-call bound | 1.45× slower | 1.82× (`s0e2`) / 1.57× (`s0e5`) slower | 1.13× / **1.05× faster** |

So the one-call path narrows their cost by 1.15–1.4× but does not change the
picture: they are ahead where their per-event cost dips (100 k), and `s0e5` is at
or slightly ahead of even that bound at 10⁶.

Their terms are `pwacg_ref/bench_ref.py` on this box, min-of-5 after two
warm-ups, XLA compilation excluded (0.3–1.1 s, and 12 s at 1 M).  **Every column
is real events, not tiled arrays**: our sample is sliced to 10 k / 100 k / 1 M,
their `Tensor/CacheTensor.py` computes the tensors for exactly that many events,
and `PWAFunc` is built on those.  (Tiling the 10 k tensors ×10/×100 instead — legal for timing, since the
per-event work is unchanged — agrees with the real-event figures to 0.5 % at
100 k and 2.4 % at 1 M.)

Their tensor precompute needs `slices` above ~10⁵ events on a 12 GB card: with
`slices = 1` it allocates the whole sample at once and dies at 100 k
(`RESOURCE_EXHAUSTED … allocate 12.22GiB`).  `--slices 20` (100 k) and `--slices
200` (1 M) use their own chunking parameter; the precompute (6.5 s / 10 s) is
outside every timed number.

**Throughput, in the benchmark's own unit** (Mev/s over both samples):

| size | ours `s0e2` | ours `s0e5` | theirs, jac path |
|---|---|---|---|
| 10 k / 10 k | 19.2 | 20.5 | 18.9 |
| 100 k / 100 k | 45.1 | 52.7 | **71.1** |
| 1 M / 1 M | 51.4 | **61.3** | 50.6 |

Their 100 k point is a genuine dip in cost and **its cause is not established**.
It is *not* bandwidth or capacity: 128 B/event at 71.1 Mev/s is ~18 GB/s, about
3 % of this card's DRAM rate and further still from its L2, and our own 160 B at
51.4 Mev/s is just as far from any memory limit — our tile is fp64-throughput
bound (`AGENTS.md`'s `ncu`: FP64 pipe 85 % of peak, DRAM 11 %).  Locating where
the dip sits would need a size sweep; `bench_ref.py --reuse` must not be used for
one (it keeps the previous tensors while the arrays shrink), which is why no sweep
is quoted.

### Their full pipeline (`Control.run`) — and what a step really costs

`pwacg_ref/fit_ref.py` runs their own fit (`Control.run` →
`scipy.minimize`, with `method`, `jac` and `hessp` chosen by `--method`) rather
than individual calls.  **Five** shims are needed, each labelled where it is
installed: `config = jax.config` (they call `config.update(...)` without importing
it), a `pynvml` no-op (their memory log returns 0 — a placeholder, not a
measurement), a `minimize` wrapper (injects `maxiter`, which they never pass, and
records `nit/nfev/njev/nhev`), a `save_in_json` no-op (their post-fit writer
reads a template the pipeline is supposed to have produced), and the
GPU/threading attributes on `mod.args()` (one process, one thread, one chunk).

**It does not complete a step on this box.**  With `--maxiter 3`, 10 k data + 10 k
MC:

```
nit=0  nfev=1  njev=1  nhev=1180     optimiser wall 5.066 s
Exception in thread … jit_hvp_in_sigle_device
  self.kk_mc_hvp_g[device_rank, _, :] = g_dot_v
IndexError: only integers, slices … are valid indices
```

Every `hessp` call raises inside their worker thread (the exception is swallowed
by `Thread`), so scipy's CG receives a zero/invalid product, and
`thread_hvp` also divides by a never-assigned `kk_mc_hvp_v` (RuntimeWarning:
divide by zero → NaN).  The run therefore terminates at `nit=0` without moving.

**The gradient path runs, and that is the one to compare with ours.**  The
paper fits three ways — iMinuit+FDM, iMinuit+AutoDiff, NTCG+AutoDiff — and their
shipped `Control.run` hard-codes the third (Newton-CG *with* `hessp`).  `fit_ref.py
--method` selects the path instead:

| `--method` | what it drives | result at 10 k / 10 k, `--maxiter 10` |
|---|---|---|
| `newton-cg` (their default) | `jac` + `hessp` (HVP) | **`nit=0`**, 1180 failed `hessp` calls, 5.07 s — never takes a step |
| `bfgs` | **`jac` only** — no Hessian | **`nit=10`**, 23 value + 23 gradient calls, **47.0 ms** (4.7 ms/iteration), NLL −224 252.69 |
| `fd` | no `jac` at all (their iMinuit+FDM shape) | available, same harness |

`bfgs` is the path an iMinuit+AutoDiff run would drive — iMinuit calls the same
`thread_likelihood`/`thread_grad_likelihood` pair — so **4.7 ms per iteration** is
their gradient-path cost at the paper's sample size.  (iMinuit itself is not
installable here: this host reaches `github.com` and `arxiv.org` but not PyPI, so
the row is measured through scipy's BFGS with their functions, not through
iMinuit's driver.)

A full-fit wall-clock race is **not** published: our own `Fitter.fit` costs
4.55 ms per `get_nll` call *inside* a fit against 0.89 ms for the same call
measured standalone in the same process (`fit(maxiter=10)`: 63.7 ms for 14
calls), and that ~3.6 ms/call of optimiser-layer overhead is not accounted for
here.  Until it is, the per-evaluation table above is the honest comparison and a
fit-time race would be comparing unexplained overheads.

**What that tells us about "per iteration", measured directly instead**
(`jit_*_kk` on the same objects, 10 k data, min-of-5):

| their evaluation | ms | note |
|---|---|---|
| `jit_data_likelihood_kk` (value) | 0.127 | |
| `jit_grad_data_likelihood_kk` (gradient) | 0.393 | |
| `jit_hvp_data_fwdrev_kk` (Hessian-vector) | 0.534 | the CG inner step |

One `hessp` call costs ~4.3 ms inside `Control.run` (5066/1182) (their wrapper → 2 jvp + a
`value_and_grad` + a value call, plus thread overhead), and their one attempt
spent **1180** of them.  So their optimiser step is **HVP-dominated**, and the
0.78 ms value+gradient figure in the table above is their *evaluation* cost, not
a step.  The two are different units and are reported as such: we compare
evaluations, because that is the only quantity both frameworks can measure
without one of the two optimisers being the subject.

**What is *not* the comparison**, so it is not read as one:

* `ampcache` (0.5 ms): coefficients pinned, `grads["coeff"] == 0` — a lower
  bound, not a fit iteration;
* the masked `s0f` row (0.9 ms, 22 of 24 cached): every mass/width fixed, but the
  Flatté's two couplings are not mass/width, so its waves stay live and the cache
  buys nothing at this size.  Their masses/widths float (59 parameters), so this is a
  *different scenario* — it shows what the constraint cache is worth, not who is
  faster on the paper's model;
* `--rsize 17`: a grouped log-sum, a different objective.

**What is re-done each iteration** (measured from the sources, not inferred —
`kernels_flat_pwa_gen_s0e2.cu`):

| | ours (`s0e2`) | theirs |
|---|---|---|
| interpolation | **load time only**: 2 fill kernels build `us` (basis rows at each event's mass, deduped by row bytes) and `amp` (`fa·fl·num` at each event's angles/q/mass) | precomputed once by `Tensor/CacheTensor.py`, one `.npy` per tensor |
| streamed per event per iteration | `us` 48 B (6 slots) + `amp` 112 B (7 slots × 2 projections) = **160 B**, fp32 (`float2`) | `phif0` 32 B + `phif2` 80 B + φ/f 16 B = **128 B**, fp64 (the tensors are shared by the six modules) |
| per-event work | `D = Σ_m coeff_m·us_m` over 13 basis columns (one factor, `n_res = 1`), then 24 waves × 2 projections | 24 couplings × tensor components, 2 propagators, 2 polarisations |
| launches per `compute` | 3 small H2D + 4 memsets + **one tile launch per `cap`-event chunk** (50 k) | a few fused XLA kernels |

So nothing is re-interpolated per iteration on either side.  We stream **1.25×
their bytes** because we keep the per-event basis rows that `coeff` re-contracts
each iteration while they keep only parameter-free numerator tensors — and we
stream them in fp32, which is a real asymmetry in our favour and is why the
byte ratio is not larger.

That byte ratio is the reason the comparison is run on this config at all:
keeping φ as a `BWR` factor makes the chain `n_res = 2`, and every fast tile is
`n_res1`, so the only leaf that can run is the general `gen2` — 1.6–1.7 ms on the
φ-kept config against 1.0 ms for the tile here at 10 k, i.e. the kernel choice,
not the model, decides that number.

## The rest of our kernels, the CPU family included

Same config, same samples, same `--vary`, min-of-3 — every `pwa_gen` leaf that
accepts this model, plus the three C/OpenMP CPU kernels.  Values are
`ms — Mev/s` (Mev/s is over data + phase space, i.e. comparable across rows):

| backend | 10 k / 10 k | 100 k / 100 k | 1 M / 1 M |
|---|---|---|---|
| `cuda_flat_pwa_gen_s0e5` | 1.0 — 20.7 | **3.8 — 52.7** | **32.6 — 61.3** |
| `cuda_flat_pwa_gen_s0e2` | 1.0 — 19.3 | 4.4 — 45.1 | 38.9 — 51.4 |
| `cuda_flat_pwa_gen_s0e3` | 1.0 — 19.6 | 4.5 — 44.7 | 39.1 — 51.1 |
| `cuda_flat_pwa_gen_s0e4` | 1.0 — 19.7 | 4.5 — 44.7 | 39.3 — 50.9 |
| `cuda_flat_pwa_gen_s0e` | 1.1 — 18.8 | 4.8 — 41.3 | 38.4 — 52.0 |
| `cuda_flat_pwa_gen_s0g` | 1.2 — 16.5 | 4.8 — 41.9 | 42.2 — 47.4 |
| `cuda_flat_pwa_gen2` (general) | 1.1 — 17.1 | 5.9 — 34.0 | 53.3 — 37.5 |
| **`cpu_flat_pwa_gen`** (C/OpenMP, uncached) | 2.2 — 9.0 | 7.7–8.5 | **32.6–84.2 (unstable)** |
| `cpu_flat_pwa_gen_cached` | 1.7 — 11.7 | 4.9–5.9 | 29.9–54.0 (unstable) |
| `cpu_flat_pwa_gen_cached2` | 1.5 — 13.3 | 5.0–6.5 | 34.8–46.3 (unstable) |

**The CPU rows are not reproducible at 1 M on this box**, and that has to be said
before any of them is read: `cpu_flat_pwa_gen` measured 32.4, 38.1, 39.2, 46.2,
47.9, 49.4 and 84.2 ms in seven runs of the same command (`--repeat 3`, same
sample, load average < 1).  The cached siblings spread 29.9–54.0 and 34.8–46.3.
A 2.6× spread makes a ranking unsupportable: **no claim is made about the CPU at
1 M**, and none of those cells appears in a ratio.  At 100 k the CPU rows are
stable to ~10 % and are quoted as points.  (The likely cause is package-level
throttling — the 84.2 ms reading followed back-to-back GPU runs — but that is a
hypothesis, not a measurement.)

What the table *does* support, on the rows that reproduce to ±1 %:

* **`s0e5` is our fastest kernel at every size ≥ 100 k** — 3.8 ms at 100 k and
  32.6 ms at 1 M, against 4.4 / 38.9 for `s0e2`, and 97.0 ms at 3 M (61.8 Mev/s)
  against 115.9;
* its 1.2–1.3× over `s0e2` is the denominator dedup (waves keyed by
  `(gamma pattern, mass columns)`), and it is the one variant that is ahead of
  their jac path at 1 M (32.6 against 39.5 ms);
* the cached CPU siblings are the wrong trade at 1 M even inside their own
  spread — their median (~42 / ~40 ms) sits above the uncached run but inside the
  same noise, so the byte argument below is arithmetic, not a measurement;

The reason the CPU caches *should* lose at 1 M is still arithmetic, and it is
  bytes, measured on THIS model: the uncached kernel streams only its inputs —
  `mass` 2 + `q` 3 + `angle` 1×6 = 88 B/event — and re-does the Catmull-Rom
  basis/barrier/numerator interpolation per event and wave on a 14-column basis;
  `cpu_flat_pwa_gen_cached` stores the interpolated `us` (6 slots) + `fa`
  (14 deduped angle columns) + `fn` (24 waves) = **704 B/event**, and
  `…_cached2` stores the finished `fa·fl·num` product, **864 B/event**.  At 1 M
  events that is 1.4–1.7 GB per iteration against 176 MB, i.e. host-RAM bound
  rather than compute bound.  The caches pay below 100 k (1.7 / 1.5 ms against
  2.3) and cost above it — the opposite of the GPU, where `s0e5`'s dedup helps
  everywhere;
* the GPU spread across `s0e`…`s0e5` is 1.0–1.2 ms at 10 k and 32.6–39.3 at 1 M,
  i.e. the variants are within 20 % of each other on this 24-wave model.

Against the reference's jac path at 1 M (**39.512 ms — 50.6 Mev/s**), `s0e5` is
**faster by 1.21×** (32.6 ms — 61.3 Mev/s) and `s0e2` is at parity (38.9 ms — 51.4
Mev/s).  No such claim is made for the CPU: its 1 M spread covers both sides of
their number.

**They all compute the same objective** — the NLL printed by each row agrees to
~1e-8 relative between the GPU kernels and the CPU ones (`-12410.113269` vs
`-12410.113073` at 10 k, `-553842.003669` vs `-553841.999083` at 1 M): the
difference is fp64 summation order, not a different model, which is what makes
the ms column readable against the others.

`s0e5`'s 1.2× over `s0e2` is its denominator dedup (waves keyed by
`(gamma pattern, mass columns)`) — it pays on this 24-wave model, and it is the
one variant that is ahead of both `s0e2` and their jac path at 1 M.

## What their cached data actually is

Read from the generated module and their `Tensor/` code, and checked numerically
on our own events:

| file | shape | contents |
|---|---|---|
| `phi_kk.npy` | `(ne,)` | **s_φ = M²(K⁺K⁻)_φ**, byte-identical to their `invm(Kp+Km)` (verified: max abs diff 0) |
| `f_kk.npy` | `(ne,)` | **s_f = M²(K⁺K⁻)_f** (same check, diff 0) |
| `phif0_kk.npy` | `(2, ne, 2)` | 2 parameter-free **covariant tensor amplitudes** × events × 2 polarisations, compact support |
| `phif2_kk.npy` | `(5, ne, 2)` | the same for the five J=2 components |

`phif0` is `{phif001, phif021}` and `phif2` is
`{phif201, phif221, phif222, phif223, phif243}` — the names in their
`Tensor/CacheTensor.fill_func`, each `_phifXXX` being a Zou–Bugg covariant tensor
built from the four-momenta by `Tensor/FUNC.py` (`T1_cov`, `T2_cov`, the metric
contractions; the source marks them "# Eq.(37)" / "# Eq.(41)" of the paper).
**No fitted parameter enters the cached arrays** — that is the whole design.

What the generated code computes from them, per event (`fit_object_kk.py`):

```
const_ph = dconstruct(const, theta)                    # c_i·e^{iθ_i}
phif     = einsum_ord("ijk,li->ljk", amp, const_ph)    # per component
phif     = einsum("ljk,lj->jk", phif, bw)              # × BW_φ·BW_R
data     = -Σ_e log( Σ_μ |Σ_modules phif|² )           # μ = 1,2  (dabs + axis=1)
mc       =  Σ_e     Σ_μ |Σ_modules phif|²
```

with `bw = BW_flatte980/BW_BW/BW_flatte1270` built from the two cached `s`
arrays and the fitted masses/widths.  So the per-iteration division of labour is
fixed: **tensors and kinematics are read, couplings and propagators are applied.**
That is the 128 B/event in the comparison above (7 components × 2 polarisations ×
8 B + 2 s × 8 B), against our 160 B of fp32 interpolation caches.

## The published sample

The reference publishes its demo sample as a GitHub release asset: **`v1.0.0`,
"Data", `data.zip`, 788 MB** (`releases/download/v1.0.0/data.zip`).  It contains
the per-event four-momenta in exactly the contract described above.

`make_data_from_ref.py` converts it into this framework's event arrays, so both
sides can be timed on **their** events:

```bash
python make_data_from_ref.py --zip /path/to/data.zip         # or --ref-dir <extracted>/
```

**Not used for the measurements above**, and the reason is environmental, not
methodological: from this host the release download is throttled to ~30 kB/s
(≈7 h for 751 MB; `pypi`/`jsdelivr` do not answer at all while `github.com` and
`api.github.com` do).  The bridge is verified instead by a round trip — the
generated `Momentum_kk.npz` is read back through `make_data_from_ref.py`, and the
converted events land inside the tables (mass [0.987, 2.693], q [0.002, 1.553])
with the same 3×10⁶ events.  When the archive is available, `--zip` re-runs the
whole table on their data with no other change.

## Memory

`mem.py` uses one child process per size.  Each row below was taken with the
size invoked **alone** (`--sizes <n>`), because several sizes in one invocation
share the sampler's baseline — the previous child's allocation is still counted
at the next child's start (measured: a 1 M cell reads 0.77 GiB inside a combined
run against 0.60 GiB alone).  Sampling is out-of-process (`tabpwa.monitor`,
10 ms).

| data + phsp | PEAK device memory | host RSS |
|---|---|---|
| 10 000 / 10 000 | 0.17 GiB | 0.81 GiB |
| 1 000 000 / 1 000 000 | 0.60 GiB | 0.86 GiB |
| 3 000 000 / 3 000 000 | **1.46 GiB** | 0.99 GiB |

The reference at 3×10⁶ integration events exceeds the 16 GB V100 it was measured
on (its own §3 wall), while its per-event tensors are resident.  Ours at 3×10⁶
data **and** 3×10⁶ phase space peaks at 1.46 GiB — linear in the sample, inside
this box's 12 GB with room to spare.  Their measured peak at 1×10⁶ MC events is
745 MB, so at equal sample size the two are comparable; the difference is what
happens as the sample grows.

The 3 M row needs `make_data.py --n-data 3000000 --n-phsp 3000000` first; `mem.py`
prints a note instead of silently timing a smaller file when the sample on disk
is short.

## Free parameters vs the paper's 59

`x = 59` — the same count as the paper, by construction: 23 free complex
couplings (46 real) plus 13 lineshape parameters (four `(mass, width)` BW pairs, the f₂(1270)
`(mass, width)`, and the f₀(980)'s `(mass, g_ππ, g_KK)`).  It matches because the
amplitude content matches; with a bare `BWR` everywhere it was 58, the Flatté's
second channel being the one parameter that was missing.

## Revisions

| side | revision |
|---|---|
| this framework | commit `682d671` on `interp_bw_pwa` — on top of `8bb3b9e` (the pinned-coefficient family), and the one that moves `numpy_flat` to the kernels' Catmull-Rom interpolation. **Every `numpy_flat` figure in this document is post-`682d671`**: with the old linear reference they were 8.8e-6 in the identity's gradient residual and 143.8479 (not 143.847925) in the norm ratio |
| reference | `caihao/PWACG` at `f516d21` (2026-05-18), `Tensor/` + golden `rendered_scripts/fit_object_kk.py` |

Numerics on the reference side are its own: x64, its `dplex` real/imag layout,
its fp64 BJ/Flatté propagators.  No absolute likelihood is compared across the
two frameworks — the objectives differ by construction (their MC mean versus our
batched phase-space integral), so only per-iteration cost is compared, and both
sides print their value so a call that computed nothing cannot look fast.
