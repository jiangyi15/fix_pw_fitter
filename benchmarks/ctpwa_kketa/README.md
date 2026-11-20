# CTPWA benchmark — J/ψ → K⁺K⁻η

Reproduces the *setup* of

> B. Xiang, S. Fang, B. Liu, "A high performance partial wave analysis
> framework for hadron spectroscopy", Chin. Phys. C **51**, 013003 (2027),
> doi:[10.1088/1674-1137/aeb254](https://doi.org/10.1088/1674-1137/aeb254)
> (framework: <https://github.com/BHXiang/ctpwa>)

so the same sample size can be timed with this framework.

## The paper's numbers (their Table 1), N_data = 10⁶, N_phsp = 5×10⁶

| stage | fixed resonances | floating resonances |
|---|---|---|
| amplitude initialisation | ~94 s | ~94 s |
| NLL + gradient per iteration | ~15 ms | ~100 ms |
| full Hessian | ~0.35 s | ~1.6 s |
| device memory | ~20 GB | ~32 GB |
| complete fit | ~7 s (~500 iters) | ~100 s (~1000 iters) |

and (their Fig. 2/3, generic 10-channel benchmark): NLL+grad ≈ 4 ms and the
20×20 analytic Hessian ≈ 13 ms at 10⁶ events, against 45 ms / 1.4 s for GPU
autograd and 2.5 s / 81 s on the CPU.

Their model: 16 intermediate resonances, 33 complex couplings, both the
K⁺K⁻ and the Kη subsystems.

## This directory

```
config.yml      the model, in this framework's config format
make_data.py    3-body phase-space generator (K+ K- eta), flat in m²_KK
bench.py        times one `Fitter.get_nll` (norm + data NLL + gradients)
data/           generated samples (gitignored)
```

```bash
python make_data.py --n-signal 1000000 --n-phsp 5000000 --out data
python bench.py --backend cuda_flat_pwa_gen_s0e2 --rsize 17
```

## Measured (this box: RTX 5070 Ti Laptop, Ryzen 9 8945HX)

10⁶ data + 5×10⁶ phase-space events, masses and widths fixed, one `get_nll`
(norm + data NLL + gradients) per iteration, min of 3, one backend per process:

| backend | ms / iteration | vs the paper's 15 ms |
|---|---|---|
| `integrated_pwa` + `cuda_v4_pwa_cache` (Gram norm + amplitude cache, m/w fixed) | **3.2** | **4.7x faster** |
| `cuda_flat_pwa_gen_s0e2` (per-event tile, not the paper's algorithm) | 71.9 | — |

The Gram path is the like-for-like row: the phase-space norm is precomputed
once, so an iteration is only the parameter update — which is exactly what the
paper's 15 ms measures for its own `integrated_pwa` + `cuda_v4_pwa_ampcache`
with the resonance parameters held.

**Sample construction**: the timing uses `make_flat_data.py`, which writes the
kernel arrays directly in the config's valid ranges (`mass`, `q`, `angle`,
unit weights).  Cost depends on shapes and sizes, not on the values, so this is
the right sample for a *timing* comparison — and it is why the NLL values above
are not physical.

## What is deliberately different

* The paper's **exact** 16-resonance list is not in its public repository
  (`example/config.yml` there ships a 6-resonance demo), so the states here
  are the PDG ones that populate the two subsystems; the topology, the event
  counts and the coupling dimension (33) follow the paper.
* The sample is generated phase space, not a model toy, so the *timing* is
  comparable (same size, same topology) while the NLL value is not.
* This framework has no analytic Hessian path, so the Hessian row of the
  table is not reproduced; `get_nll` (value + gradient) is the comparable
  quantity.
* **A physics sample is currently blocked by a framework bug**, not by this
  benchmark: `tabpwa.pwa_build.generate_pwa_phsp` returns four-vectors with
  negative energies that sum to zero on a plain `jpsi -> R + c` chain
  (measured: Sigma E = 0.0000, |Sigma p| = 3.0968, m^2 = -9.59), including on
  `tests/config_pwa.yml`, so `toy_pwa_generate` cannot produce a 3-body toy.
  The hand-written generator here (`make_data.py`'s earlier revision) was
  verified correct (mirror subsystems, sqrt(s) = 3.0969 exactly) and is the
  fallback once the framework one is fixed.
