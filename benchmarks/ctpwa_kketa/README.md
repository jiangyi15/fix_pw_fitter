# CTPWA benchmark — J/ψ → K⁺K⁻η

Reproduces the *setup* of

> B. Xiang, S. Fang, B. Liu, "A high performance partial wave analysis
> framework for hadron spectroscopy", Chin. Phys. C **51**, 013003 (2027),
> doi:[10.1088/1674-1137/aeb254](https://doi.org/10.1088/1674-1137/aeb254)
> (framework: <https://github.com/BHXiang/ctpwa>, arXiv:2608.25963)

so the same sample size can be timed with this framework.

## The paper's numbers (their Table 1), N_data = 10⁶, N_phsp = 5×10⁶

| stage | fixed resonances | floating resonances |
|---|---|---|
| amplitude initialisation | ~94 s | ~94 s |
| NLL + gradient per iteration | ~15 ms | ~100 ms |
| full Hessian | ~0.35 s | ~1.6 s |
| device memory | ~20 GB | ~32 GB |
| complete fit | ~7 s (~500 iters) | ~100 s (~1000 iters) |

## The model — taken from the paper's fit figure, not from prose

The text states only counts ("16 intermediate resonances ... 33 complex
couplings") and the repo ships a 6-resonance demo, so the model comes from the
component **legend of the fit projection** (`results_plot.pdf` in the arXiv
source), which lists every amplitude:

| K⁺K⁻η sector: J/ψ → Xη, X → K⁺K⁻ | Kη sector: J/ψ → K\*K̄, K\* → Kη (per charge) |
|---|---|
| φ(1020), ω(1420), φ(1680), X(1750), ω(2220) — 5×`1⁻⁻` | K\*(1410), K\*(1680), K\*(2380) — 3×`1⁻` |
| ω₃(1670), φ₃(1850), ρ₃(1690), ρ₃(1990), ρ₃(2250) — 5×`3⁻⁻` | K₂\*(1430), K₂\*(1980) — 2×`2⁺` |
| ρ₅(2350) — 1×`5⁻⁻` | K₃\*(1780) `3⁻`, K₄\*(2045) `4⁺`, K₅\*(2380) `5⁻` |
| + non-resonant NR₁, NR₃, NR₅(K K̄) | + non-resonant NR₁, NR₃(Kη) |

That is 19 distinct resonances (11 + 8) plus 7 non-resonant partial waves, and
the framework reproduces the paper's **dimension exactly**:

```
34 chains -> 34 partial waves
φ(1020) ck fixed to 1 (gauge: overall scale+phase) -> 33 free complex couplings
                                                     = 66 real slots  <- the paper's 33
+ 38 mass/width parameters (19 resonances x 2), 1 real slot each
    all fixed -> x = 66    |   10 free -> x = 76    |   all free -> x = 104
```

The prose's "16" is consistent with 11 K⁺K⁻ states + 5 Kη `J^P` classes;
"33 couplings" is reproduced as an exact count.

**Physics notes that decide the list.** Every state carries one `(L, S)`
combination with `L = J` — the natural-parity set that `J/ψ → Xη` and
`K* → Kη` admit.  Consequently the `0⁺⁺`/`2⁺⁺` K K̄ states have no wave
(`C = +1` is forbidden with an η), and a `1⁺` has **no** Kη wave at all (two
pseudoscalars force `P = (−1)^J`), which is why the legend's `K₁*(1410/1680/2380)`
are the J=1 natural-parity K\* states (`K*(1410)` …), not the `K₁(1270/1400)`
1⁺ ones.  `K*(892)` cannot couple to Kη (0.8955 < m_K + m_η = 1.0415 → q₀ = 0),
so it is absent from the Kη sector.

## This directory

```
config.yml       the model (`amp_model: pwa`)
config_gen.yml   the same model as `amp_model: pwa_gen` (basis/flat config)
make_data.py     samples built by the framework: generate_pwa_phsp + build_event_data
bench.py         times one `Fitter.get_nll` (norm + data NLL + gradients)
check_one.py     one backend, one evaluation: dumps Q/P/nll (cross-backend compare)
data/            generated samples (gitignored)
```

```bash
python make_data.py --n-data 1000000 --n-phsp 5000000      # 27 s
python bench.py --backend integrated_pwa --base cuda_v4_pwa_cache --repeat 3 --fix-mw
python bench.py --backend cuda_v4_pwa_s0e --repeat 3 --free-mw phi1680_mass,phi1680_width
```

`--fix-mw` fixes every mass/width at its config value; `--free-mw a,b,…` leaves
the named ones free.  This matters because **`BWR` masses/widths are free by
default** in this framework (`x` = 104 without the flags), which is neither of
the paper's scenarios.

## Measured (RTX 5070 Ti Laptop, Ryzen 9 8945HX)

10⁶ data + 5×10⁶ phase-space events, one `get_nll` per iteration, min of 3,
one backend per process:

| scenario | backend | ms/iter | vs the paper |
|---|---|---|---|
| masses/widths fixed (x = 66) | `integrated_pwa` + `cuda_v4_pwa_cache` | **6.3** | **2.4× faster** than 15 ms |
| 10 resonance params free (x = 76) | `cuda_v4_pwa` | 657.7 | 6.6× slower than 100 ms |
| | `cuda_v4_pwa_s0c` | 532.7 | 5.3× slower |
| | `cuda_v4_pwa_s0e` | **235.1** | **2.4× slower** |
| `pwa_gen`, `rsize=1` (x = 66 or 76) | `cuda_flat_pwa_gen_s0e2` | 138.4 | 1.4× slower |

So: **faster on the fixed-parameter column, slower on the floating one.**  The
fixed column is like-for-like (`integrated_pwa` + amplitude cache ≈ their
`integrated_pwa` + `cuda_v4_pwa_ampcache` with the resonance parameters held);
the floating column cannot use a Gram precomputation here, so it recomputes the
phase-space integral every iteration — the same structural penalty the paper
describes, larger for us.  The `pwa_gen` row is a different model family and is
not their algorithm; it is included because it is the per-event kernel.

## Verification gates (all re-run on the final sample)

```
2000 events, masses/widths fixed, vs numpy_pwa (reference):
  numpy_flat              rel dNLL 5.3e-07   rel|dP| 6.1e-06
  cuda_v4_pwa             rel dNLL 1.3e-08   rel|dP| 1.1e-07
  cuda_v4_pwa_s0c / s0e   rel dNLL 1.3e-08   rel|dP| 1.1e-07
  cuda_flat_pwa_gen_s0e2  rel dNLL 8.7e-07   rel|dP| 1.4e-05   (fp32 cache floor)
10 resonance params free, gradients vs cuda_v4_pwa:
  s0b 1.5e-14 | s0c 1.8e-14 | s0e 2.9e-16
full size, all pwa backends: NLL = +1472670.012285 (flat: +1472670.010809)
```

**A trap worth recording**: a table-based backend (`numpy_flat`, the flat CUDA
family, `s0c/s0d/s0e`) extrapolates silently outside its `mass`/`q` grids while
an analytic one (`numpy_pwa`, `cuda_v4_pwa`, `s0b`) does not, so an
out-of-range sample makes the two families disagree by orders of magnitude on
the *same* parameters.  `make_data.py` therefore asserts both grids and both
arrays against them.

## What is deliberately different

* The Hessian row of the paper's table is not reproduced (no analytic Hessian
  path here); `get_nll` (value + gradient) is the comparable quantity.
* The sample is **flat 3-body phase space** (unit weights), built by the
  framework's own generator and event builder — not a model toy.  The *timing*
  is comparable (same size, topology, per-iteration work); the NLL value is
  not physical.  `toy_pwa_generate` is the model-toy route, but its rejection
  acceptance on this model is ~0.2 % (~5×10⁸ candidates for 10⁶ events).
* Masses/widths are PDG values; the paper's fit results differ slightly and do
  not affect the timing.
