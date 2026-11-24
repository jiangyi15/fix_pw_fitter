# CTPWA benchmark — J/ψ → K⁺K⁻η

Reproduces the *setup* of

> B. Xiang, S. Fang, B. Liu, "A high performance partial wave analysis
> framework for hadron spectroscopy", Chin. Phys. C **51**, 013003 (2027),
> doi:[10.1088/1674-1137/aeb254](https://doi.org/10.1088/1674-1137/aeb254)
> (CTPWA, arXiv:2608.25963)

so the same sample can be timed with this framework.  The sample is 10⁶ signal
plus 5×10⁶ phase-space events of `J/ψ → K⁺K⁻η`, the model is described in §2,
and §6–§9 are the measurements.  Those are reproducible from `make_data.py`
plus one `bench.py` call per backend; §10 additionally needs `ctpwa_ref/` (their
framework, built per `ctpwa_ref/BUILD.md`).

**The headline — this framework against CTPWA, by sample size (§10).**  Their
sources were built here (`ctpwa_ref/BUILD.md`) and run on the same GPU, the same
model — 34 partial waves, 2 polarizations, per-wave map verified to <1e-6
(`ctpwa_ref/coupling_map.md`) — and the same data:phase-space ratio.  One
`get_nll` (value + gradient), parameters moving, min of 3 for ours and 4 for
theirs (`ctpwa_ref/BUILD.md`):

| data + phsp | CTPWA, x = 76 | ours, x = 76 (`s0g`) | CTPWA, x = 66 | ours, x = 66 (Gram) | CTPWA, x = 104 | ours, x = 104 (`s0f`) |
|---|---|---|---|---|---|---|
| 0.6×10⁶ | 26.61 | **9.3** (2.9×) | **1.16** | 1.3 | 92.55 | **13.8** (6.7×) |
| 1.2×10⁶ | 52.12 | **16.6** (3.1×) | 2.04 | **1.7** | 182.2 | **25.6** (7.1×) |
| 2.4×10⁶ | — | **31.1** | 3.31 | **2.7** | OOM | **49.0** |
| 6×10⁶ (the paper's) | — | **74.8** | — | **6.2** | — | 119.0 |

`x` is what floats, not which kernel runs: **66** = every mass/width pinned,
**76** = the ten named resonance parameters free, **104** = every mass/width
free.  The kernels are named in parentheses because the choice follows from `x` —
the Gram path needs `x` = 66 (it zeroes the `m0`/`g0` blocks), and the
kernels `s0f`/`s0g` cache the waves whose coefficient row `x` cannot move.

The `x` = 104 column quotes `s0f` because it is the kernel the
all-free shape favours, NOT because it flatters the comparison: there the
cache engages on only
7 of 34 waves, so `s0f` is 1.17× the plain tile (13.8 against 16.1 ms) and the
gap to CTPWA's 92.55 is **6.7× with the cache, 5.7× without**.  The plain tile
is the conservative reading of that column and the conclusion holds either way.

Every cell is measured: the four `—` and the one `OOM` are all their memory
wall (their free paths exceed the card above 1.2×10⁶), never a missing run.
Their all-free column is `vary.py config_freeall.yml` — 38 free resonance
parameters = our `x` = 104 — re-measured on a fresh `sm_120` build of their
sources here: 79.84 ms with `theta` repeating, **92.55 ms** with it moving at
0.6×10⁶, and **182.23 ms** at 1.2×10⁶.  At the paper's size neither of their
columns fits on this GPU, so no comparison is claimed there — only our
measured absolute numbers.

The last two columns are the extreme the paper does not run.  At `x` = 104 all
104 of our parameters float: 54 of the 60 basis columns move and the **7 waves
whose denominator is a constant** (the non-resonant `ONE` waves) stay
iteration-invariant, so the cache is not empty and `s0f` gains 1.17–1.19× — while
theirs costs 92.55 ms against our 13.8 ms, a 6.7× gap that is the largest in the
table.  The gain tracks the CACHED FRACTION, which is the number to carry away:
7/34 → 1.19×, 27/34 → 1.49× (`s0f`) / 1.92× (`s0g`), 34/34 → 4.2×
(141.8 → 33.7).  `s0g` is only 1.07× at `x` = 104 and loses to `s0f` at both
extremes — its un-staged double `D_k` walk pays only in the mixed middle, which
is where production sits.

The free column is what the coefficient cache changed: caching the **27 of 34
waves no free parameter can move** (derived from the constraint state, not
written by hand; 432 B/event per handle) takes 6×10⁶ from 235.7 ms — the
previous best, `cuda_v4_pwa_s0e` — to 74.8 ms, 1.9× over the plain per-event
tile (142.4 ms) and 1.1× over this box's GPU+CPU hybrid (84.0 ms).

What the table shows is the trade: **their fixed column wins the smallest size**
(1.16 against our 1.3 ms — per-call overhead, ours leads from 1.2×10⁶), while
**the free column is ours at every size it is measured** (2.9–3.1×).

**Revisions the numbers belong to** — a timing is only meaningful with the code
it was taken from:

| component | revision |
|---|---|
| this framework (`tabpwa`, the `src/` tree) | commit `12288736cba6` on `interp_bw_pwa`; `src/` clean when the runs were made.  The `s0f`/`s0g` rows in §6.1/§6.2 were measured at `c98ed5b` (the commit that adds `cuda_flat_pwa_gen_s0f`/`_s0g`), everything else at `12288736cba6` |
| CTPWA (`ctpwa_ref/BUILD.md`) | clone at `bad50bd3fd38` (2026-11-20) |

So: `git checkout 12288736cba6` for the framework side, and the pinned clone in
`ctpwa_ref/BUILD.md` for theirs, before re-running anything here.  For the
`s0f`/`s0g` rows, `git checkout c98ed5b` instead (their harness flag `--mask` is
in `bench.py`; the two kernels are additive, so nothing else in the table
changes).

---

## 1. What the paper reports (its Table 1)

| stage | fixed resonance params | floating resonance params |
|---|---|---|
| amplitude initialisation | ~94 s | ~94 s |
| NLL + gradient per iteration | ~15 ms | ~100 ms |
| full Hessian | ~0.35 s | ~1.6 s |
| device memory | ~20 GB | ~32 GB |
| complete single fit | ~7 s (~500 iters) | ~100 s (~1000 iters) |

Their model: "16 intermediate resonances ... 33 complex couplings", both the
K⁺K⁻ and the Kη subsystems.

The table is quoted for **provenance, not as a yardstick**: the paper does not
name its device, and §10 shows its two NLL rows cannot both come from one
hardware factor.  Every comparison in this document is against their code run on
this box.

## 2. The model — from the paper's fit figure, not from prose

The text gives only counts and the repo ships a 6-resonance demo, so the model
is taken from the **component legend of the fit projection**
(`results_plot.pdf` in the arXiv source), which lists every amplitude:

| K⁺K⁻η sector: J/ψ → Xη, X → K⁺K⁻ | Kη sector: J/ψ → K\*K̄, K\* → Kη (per charge) |
|---|---|
| φ(1020), ω(1420), φ(1680), X(1750), ω(2220) — 5×`1⁻⁻` | K\*(1410), K\*(1680), K\*(2380) — 3×`1⁻` |
| ω₃(1670), φ₃(1850), ρ₃(1690), ρ₃(1990), ρ₃(2250) — 5×`3⁻⁻` | K₂\*(1430), K₂\*(1980) — 2×`2⁺` |
| ρ₅(2350) — 1×`5⁻⁻` | K₃\*(1780) `3⁻`, K₄\*(2045) `4⁺`, K₅\*(2380) `5⁻` |
| + non-resonant NR₁, NR₃, NR₅(K K̄) | + non-resonant NR₁, NR₃(Kη) |

Every state carries one `(L, S)` with **`L = J`** — the natural-parity set that
`J/ψ → Xη` and `K* → Kη` admit.  Consequences that fix the list: the `0⁺⁺`/`2⁺⁺`
K K̄ states have no wave (`C = +1` is forbidden opposite an η), a `1⁺` has **no**
Kη wave at all (two pseudoscalars force `P = (−1)^J` — so the legend's
`K₁*(1410/1680/2380)` are the J=1 natural-parity K\* states, not the
`K₁(1270/1400)` 1⁺ ones), and `K*(892)` cannot couple to Kη
(0.8955 < m_K + m_η = 1.0415 → q₀ = 0).

## 3. Dimension (the paper's 33 is reproduced exactly)

```
34 chains -> 34 partial waves
φ(1020) ck fixed to 1 (gauge: overall scale+phase) -> 33 free complex couplings
                                                     = 66 real slots   (= their 33)
+ 38 mass/width parameters (19 resonances x 2), one real slot each
   all fixed -> x = 66   |   10 free -> x = 76   |   all free -> x = 104
```

The figure lists 19 distinct resonances (11 + 8, charges folded) and 7
non-resonant partial waves; the prose's "16" is consistent with 11 K⁺K⁻ states
plus the 5 Kη `J^P` classes, while the coupling count matches exactly.

**All 34 are live**: a one-hot scan (`check_waves.py`, one coupling at a time)
returns a non-zero per-event intensity for every wave (34/34, weakest 9.2e-1
against 2.5e3), with the topology `jψ → K⁻ + R(K⁺η)` — the resonance
must recoil against the *opposite*-charge kaon; `config.yml` carries a comment
at the decay block explaining why the same-charge variant is silently wrong.

## 4. Files and usage

```
config.yml       the model (`amp_model: pwa`)
config_gen.yml   the same model as `amp_model: pwa_gen` (flat/basis config)
make_data.py     samples, built by the framework (generate_pwa_phsp + build_event_data)
bench.py         one Fitter.get_nll: time, Mev/s, prep breakdown, --shard
mem.py           peak device memory of one backend (out-of-process nvidia-smi
                 sampler, 10 ms) + host RSS
mem_shard.py     peak device VRAM (out-of-process nvidia-smi) + per-process RAM
                 of a mixed `shard` (psutil VmHWM)
check_one.py     one backend, one evaluation: dumps Q/P/nll/grad for comparison
check_waves.py   one-hot scan: every partial wave alive in the sample (34/34)
ctpwa_ref/       §10 only: their config (fixed/free), their-sample generator,
                 a timing harness and the build recipe — see ctpwa_ref/BUILD.md
data/            generated samples (gitignored)
```

```bash
python make_data.py --n-data 1000000 --n-phsp 5000000        # 27 s
python bench.py --backend integrated_pwa --base cuda_v4_pwa_cache --fix-mw
python bench.py --backend cuda_v4_pwa_s0e --free-mw \
    phi1680_mass,phi1680_width,X1750_mass,X1750_width,omega2220_mass,omega2220_width,\
    K3st1780p_mass,K3st1780p_width,K4st2045p_mass,K4st2045p_width   # x = 76
python bench.py --config config_gen.yml --shard cuda_flat_pwa_gen_s0e2,cpu_flat_pwa_gen_cached \
                --weights 57,43

# the fixed-coefficient kernels: `fixed_coeff` is the mask's lifecycle, `--base`
# names the kernel.  `--mask` derives the columns from the constraint state
# (the same x0 the timing uses), so --fix-mw / --free-mw decide what is cached:
python bench.py --config config_gen.yml --backend fixed_coeff \
                --base cuda_flat_pwa_gen_s0g --mask --free-mw \
    phi1680_mass,phi1680_width,X1750_mass,X1750_width,omega2220_mass,omega2220_width,\
    K3st1780p_mass,K3st1780p_width,K4st2045p_mass,K4st2045p_width   # 27 fixed / 7 free

# peak memory (GPU: the monitor's out-of-process nvidia-smi sampler at 10 ms,
# MiB-quantised; CPU: exact VmHWM).  That sampler IS the measurement -- an
# external nvidia-smi log would be the same instrument, not a cross-check:
python mem.py --backend cuda_v4_pwa_s0e --repeat 2
```

`--fix-mw` pins every mass/width; `--free-mw a,b,…` leaves the named ones free.
This matters because **`BWR` masses/widths are free by default here** (`x` = 104
without either flag), which is neither of the paper's scenarios.

## 5. Verification — are the numbers trustworthy?

Two questions: does `pwa_gen` reproduce the `pwa` model, and do the kernels
reproduce the analytic reference?

```
2000 events, masses/widths fixed, vs numpy_pwa (the `pwa` reference):
  numpy_flat              rel dNLL 4.6e-07   rel|dP| 6.0e-06    (its own interp floor)
  cuda_v4_pwa             rel dNLL 1.3e-08   rel|dP| 1.1e-07
  cuda_v4_pwa_s0c / s0e   rel dNLL 1.3e-08   rel|dP| 1.1e-07
  cuda_flat_pwa_gen_s0e2  rel dNLL 8.7e-07   rel|dP| 1.4e-05    (fp32 cache floor)

2000 events, MASKED (a wave's whole coeff row declared fixed), vs numpy_flat:
  cuda_flat_pwa_gen_s0f / s0g   Q 9.2e-07  P 3.3e-06  dck 1.3e-06  dcoeff 7.5e-06
                                (tests/_flat_masked.py: 5 fixtures x rsize 1/17,
                                 STRICT _rel, the same floor as the unmasked anchor)
  full size (27/34 waves cached): both agree with s0e2 to 3.2e-11 relative
                                (NLL +1040064.591552 against +1040064.591585)

10 resonance params free, gradients vs cuda_v4_pwa:
  2000 events: s0b 1.5e-14 | s0c 1.8e-14 | s0e 2.9e-16
  full sample (x=76): s0c rel dNLL 3.0e-15, gradient 7.0e-13, P 5.9e-14
                      s0e rel dNLL 4.7e-16, gradient 4.8e-14, P 5.9e-14

CPU flat kernels reproduce numpy_pwa at 2000 events
(+2.0498764897e+03 both; relative difference 8e-14);
at full size +1040064.591055 against numpy_pwa's +1040064.591398 (rel 3.3e-10).

all pwa backends, full size: NLL = +1040064.591398
pwa_gen tile:                NLL = +1040064.591585
mixed shard 57/43:           NLL = +1040064.597434   (1e-8: each worker sums its own slice)
```

## 6. Time per iteration

10⁶ data + 5×10⁶ phase-space events, one `get_nll` (norm + data NLL +
gradients), min of 3, one backend per process.  The paper's 15 ms / 100 ms are
on a device it does not name, so they are not used as a yardstick: §10 runs
their implementation on this box instead, and that is the comparison.  The split
into two configurations follows the paper's:
`integrated_pwa` is the **fixed-mass/width** configuration (its Gram norm is
parameter-independent), and **every other kernel is quoted in the free
configuration**, where it is the only option.  Our per-event kernels cost the
same either way (freeing masses changes neither the caches nor the per-event
work), but the free column is where they are used — with one exception:
`s0f`/`s0g` exist to exploit the mask, so they are quoted
in BOTH columns, and its two columns differ by 2.8x (`s0f`) and 1.8x (`s0g`).

**Method note — the parameters are held between repeats** (`get_nll(x0)` with
the same `x`), which *is* the fixed column.  `bench.py --vary` perturbs every
free slot by `1e-3(1+|x|)` per iteration, as a real fit does; with it the quoted
numbers barely move: `s0e` ten free 235.4 → 235.7 ms, CPU 174.1 → 175.5 (both in
one process), hybrid
81.9 → 84.0, and `integrated_pwa` (masses pinned, so only the couplings move)
6.2 → 6.6 ms.  No result depends on the parameters being still, and no quoted
backend is hiding a value-keyed cache.  `s0f`/`s0g` are the ones whose
cache IS keyed by the parameters, so `--vary` is its sharpest check: `s0f`
94.7 → 96.0 ms, `s0g` 73.5 → 74.8 (both in one process).  It cannot hide a stale
cache by construction — the cached columns are the ones the constraint state
declares fixed, and every call compares their `coeff` against the declared
values, raising `FIXED_CHANGED` rather than reusing them (a free column that
moves is recomputed, which is the point).  The same check against their
implementation is **not** clean — its free column is 9–16 % faster when `theta`
repeats than when it moves, which is why §10 quotes their numbers at moving
parameters (`ctpwa_ref/vary.py`).

### 6.1 Masses/widths fixed (x = 66)

| backend | ms/iter | Mev/s |
|---|---|---|
| `integrated_pwa` + `cuda_v4_pwa_cache` (Gram norm + amp cache) | **6.2** | 967 |
| `cuda_flat_pwa_gen_s0f` — per-event, all 34 waves cached | 33.7 | 178.0 |
| `cuda_flat_pwa_gen_s0g` — per-event, un-staged, all 34 cached | 40.8 | 147.0 |

The like-for-like row: our `integrated_pwa` + `cuda_v4_pwa_cache` with the
resonance parameters held.  Its per-iteration cost depends on the **data**
sample only — measured 1.7 ms at 2x10^5 data events whether the phase space is
10^6 or 5x10^6, and 6.2 ms at 10^6 data — because the phase-space side is a
Gram matrix.  With the parameters held, that makes every per-event kernel
(33.7-605 ms) the wrong tool.

The two cached-wave rows are what the mask buys the per-event column: with
`--fix-mw` every mass/width is pinned, so all 34 waves are iteration-invariant
and `s0f` caches their whole contribution at load (`Afix = Amp·conj(D)/|D|²`,
432 B/event per handle).  That takes the per-event cost from 234.5/605 ms to
33.7 ms — 5.4x away from the Gram path rather than 38x, but still losing, and
at 178 Mev/s it is the fastest per-event kernel here.  `s0g` (the un-staged
variant, 40.8 ms) is slower precisely because it stages no per-event scratch:
it is the MIXED case's kernel, not this one's.

### 6.2 Resonance parameters free (x = 76)

| backend | ms/iter | Mev/s |
|---|---|---|
| `shard` GPU + CPU 57/43 (§9) | **82.0** | 73.2 |
| `cuda_flat_pwa_gen_s0g` (un-staged, 27/34 waves cached) | **73.5** | 81.6 |
| `cuda_flat_pwa_gen_s0f` (split F/T/B, 27/34 waves cached) | 94.7 | 63.4 |
| `cuda_flat_pwa_gen_s0e2` (per-event tile, rsize=1) | 141.2 | 42.5 |
| `cpu_flat_pwa_gen_cached` (per-event, CPU 32 thr, rsize=1) | 179.3 | 33.5 |
| `cpu_flat_pwa_gen_cached2` | 192.3 | 31.2 |
| `cpu_flat_pwa_gen` | 234.2 | 25.6 |
| `cuda_v4_pwa_s0e` (best per-event `pwa` kernel) | 234.5 | 25.6 |
| `cuda_v4_pwa_s0c` | 473.5 | 12.7 |
| `cuda_v4_pwa` | 605.1 | 9.9 |

The `s0f`/`s0g` rows are the same objective as `s0e2` with the mask derived from
this column's constraint state: the ten named masses/widths are free, which
moves 14 of the 60 basis columns, so **27 of the 34 waves have an
iteration-invariant denominator and are cached at load** (67/88 non-zeros; 432
B/event per handle).  That is 1.49x (`s0f`) and 1.92x (`s0g`) over the parent at
`rsize=1`; at `rsize=17` the same comparison is 144.6 -> 92.9 / 72.1 ms, because
the grouped objective gives the tile more room.  **Convention**: these two rows
are measured at the **same `x`** (`--vary` off), whereas §10's free-column
entries — including the header's 74.8 ms at 6×10⁶ — are measured with the
parameters moving.  The difference is inside the run-to-run spread and barely
moves the ratios: 1.49× → 1.48× (`s0f`) and 1.92× → 1.90× (`s0g`).  They SUPPLEMENT `s0e2` rather
than replace it: `s0e2` is the right kernel when few waves are fixed (`s0f`
mask-off is within ~1 % of it, `s0g` +3 %), and `s0g` is the mixed-case one —
all-fixed it is 21 % slower than `s0f` (40.8 vs 33.7), which is the §6.1 row.

**The all-free extreme (x = 104)** is the one to keep in mind before quoting
these as a general speedup: freeing every mass/width moves 54 of the 60 columns
and leaves **7 waves** invariant (constant denominator — the non-resonant
`ONE` waves), which is measured at 141.8 -> **119.0** ms for `s0f` (1.19x) and
132.2 for `s0g` (1.07x).  The cache scales with the FIXED fraction: 7/34 cached
buys 1.19x, 27/34 buys 1.49x (`s0f`) and 1.92x (`s0g`, where the un-staged pass
stops paying for the split), 34/34 buys 4.2x (141.8 -> 33.7, the §6.1 row).  `s0g` therefore loses to
`s0f` at both extremes and wins only in the mixed middle, which is where the
production configuration sits.

The product cache (`cached2`) is 7 % slower here; its documented
`n_fa > N·(P−1)` rule makes this shape a tie, so treat `cached` as the default
and `cached2` as shape-dependent.

`integrated_pwa` cannot appear here — its Gram norm assumes the resonance
parameters are fixed, and that is enforced rather than merely slow: the Gram
backends zero the `m0`/`g0` gradient blocks, so `bench.py` refuses the
combination outright, and underneath it the cache base raises
`ValueError: cuda_v4_pwa_cache: m0/g0 changed after the cache was built —
fixed m0/g0 only; use cuda_v4_pwa when they float`.  That is why the framework's
own best number moves 6.2 ms → 234.5 ms between the two columns.

**Takeaway**: at fixed masses the Gram path is **5.4× cheaper than the best
per-event kernel** (6.2 against 33.7 ms, `s0f` with every wave
cached), 38× cheaper than the per-event `pwa` kernels it replaces
(234.5–605 ms), and 11.9× cheaper than the best free-column configuration
(6.2 against 73.5 ms).  Where the masses float, the free column is 73.5 ms on
`s0g` (27 of 34 waves cached), 94.7 on `s0f`, 141.2 ms on the
plain tile, 234.5 ms pure GPU and 82.0 ms for the hybrid; §10
puts these against their implementation on the same box, where ours leads at the
paper's size.

## 7. Time to prepare

Phases: `model build` (Fitter + kernel/.so + tables), `file i/o` (npz read —
not a precompute), `upload+caches` (the two device handles), `constrain`
(`apply_constrains` + `initial_values`, 0.0 s here but printed and included in
`total`), and `first NLL` (anything lazy, e.g. the Gram matrices).

| backend | build | i/o | upload+caches | first NLL | total |
|---|---|---|---|---|---|
| `integrated_pwa` + `cuda_v4_pwa_cache` | 1.1 | 0.5 | 0.0 | **1.4** | **3.0 s** |
| `cuda_v4_pwa` | 0.3 | 0.5 | 0.4 | 0.7 | 1.8 s |
| `cuda_v4_pwa_s0c` | 0.3 | 0.5 | 0.4 | 0.5 | 1.7 s |
| `cuda_v4_pwa_s0e` | 0.3 | 0.5 | 0.8 | 0.2 | 1.8 s |
| `cuda_flat_pwa_gen_s0e2` | 0.3 | 0.5 | 0.6 | 0.1 | 1.5 s |
| `cpu_flat_pwa_gen_cached` | 0.1 | 0.5 | 0.8 | 0.2 | 1.6 s |
| `cpu_flat_pwa_gen` | 0.1 | 0.5 | 0.3 | 0.2 | 1.0 s |

Columns are independently rounded, so a row need not sum to its printed
`total` (which comes from the same run).

The comparable column is `model build + first NLL` = **2.5 s** for
`integrated_pwa`, because nothing here builds their
`nEv × nSL × nFree × nSigma` kinematic-tensor table; their measured
initialisation on this box is 151–185 s (§10).

**The 6.2 ms does not include the Gram build**: `ensure_gram` runs inside the
first `get_nll`, which `bench.py` excludes as warm-up.  Charged to the fit, a
500-iteration fit is `1.4 s / 500 + 6.2 ms = 9.0 ms` per iteration (their
measured 151–185 s of initialisation, §10, is 0.3–0.37 s per iteration at the
same 500 steps).  The Gram is rebuilt per process (`cache_file`
defaults to `None`; `set_cache_file` would make it a file read).

## 8. Memory

Peak over the whole run, one backend per process, on an **idle device** — the
instrument is device-wide `nvidia-smi`, sampled OUT OF PROCESS at 10 ms (MiB
granularity), so any other GPU process would be included.  It starts before
`Fitter(...)`, so the baseline is the card's idle level: `context` is the CUDA
context this backend creates and `PEAK` includes it.  That is the honest
reading of the card's use, and it is 0.13-0.16 GiB higher than the historical
"peak (in-process)" numbers below, whose 1 ms in-process `cudaMemGetInfo`
thread had already created the context before taking its baseline and
therefore excluded it — a re-run today reproduces the "external max" column,
not "peak (in-process)".  The current peak is also coarser (10 ms,
MiB-quantised) than that historical 1 ms probe, so a re-run may differ at the
sampler's resolution.  Same split as §6: `integrated_pwa` is the fixed
configuration, every other kernel is listed under the free one.  A footprint is
a property of the backend's cache, not of the parameter set, so the free rows
hold at x=66 too.

Two things hold the "peak" number honest:

* **host RSS is `VmHWM`**, the kernel's own high-water mark — exact, no sampling.
* **the device number is the driver's own**: `nvidia-smi` reads what the card
  reports, so it cannot be optimistic about this process.  The migration is
  consistent with the old instrument, not a change of measurement: the
  pre-migration 1 ms in-process `cudaMemGetInfo` peak and a 100 ms
  `nvidia-smi` log taken alongside the largest row (`cuda_v4_pwa_s0e`, below)
  agreed — **11.09 GiB absolute** against `10.93 GiB + 0.16 GiB` of CUDA
  context, with 2 MiB held before the run — so the 10 ms sampler reports the
  same footprint at its own resolution and is not picking up another process.
  Its limit is granularity, not accuracy: at 10 ms and 1 MiB a transient
  allocation shorter than one interval can be under-reported, which is why
  every B/event figure below is also checked against the allocations.

### 8.1 Masses/widths fixed (x = 66)

| backend | peak GPU | host RSS |
|---|---|---|
| `integrated_pwa` + `cuda_v4_pwa_cache` | **2.36 GiB** | — |

### 8.2 Resonance parameters free (x = 76)

| backend | peak GPU | host RSS |
|---|---|---|
| `cuda_v4_pwa_s0e` | **10.93 GiB** | — |
| `cuda_v4_pwa_s0c` | 1.74 GiB | — |
| `cuda_v4_pwa` | 1.67 GiB | — |
| `cuda_flat_pwa_gen_s0e2` | 3.11 GiB | — |
| `cpu_flat_pwa_gen` | — | 2.15 GiB |
| `cpu_flat_pwa_gen_cached` | — | 9.64 GiB |
| `cpu_flat_pwa_gen_cached2` | — | 10.36 GiB |

**Per-kernel footprint, with the external check.**  The rows span 1.7-10.9 GiB,
and that spread is each kernel's own cache rather than a measurement artefact —
every row was re-run at full size with an `nvidia-smi` log at 100 ms taken
alongside, and the external maximum is the in-process peak plus the CUDA context
(0.13-0.16 GiB) in all five.  **This table predates the migration**: "peak
(in-process)" excluded the context and "external max" is that peak plus it; the
current sampler's `PEAK` lands in the "external max" column.

| backend | peak (in-process) | B/event | external max | what the cache is |
|---|---|---|---|---|
| `cuda_v4_pwa` | 1.67 GiB | 299 | 1.83 GiB | data columns (48) + one `float2` per deduped wave (208) + workspace |
| `cuda_v4_pwa_s0c` | 1.74 GiB | 311 | 1.89 GiB | the same set in the uniform-coefficient layout |
| `cuda_flat_pwa_gen_s0e2` | 3.11 GiB | 557 (499 resident) | 3.27 GiB | flat `us` 264 + amp 208 + 24, plus a per-chunk fp64 staging |
| `cuda_v4_pwa_s0e` | 10.93 GiB | 1956 | 11.09 GiB | `gi` 1088 (gamma interpolation) + `us` 608 + `amp` 208 + data 48 |
| `integrated_pwa` + cache | 2.36 GiB | 2.53 kB per **data** event | 2.49 GiB | data side only; the phase space enters as the 34×34 Gram |

So "1.67 GiB looks small" is the design: a kernel with no gamma-interpolation
cache pays 299 B/event where `s0e`'s 1088 B/event of `gi` alone makes 1956.

A `cuda_v4_pwa` handle costs ~256 B/event: `mass` (3 columns), `weight`, `bkg` and
`P` are 48 B, and the kernel also keeps its `float2` amplitude cache over
`n_uniq = 26` (208 B) — 1.54 GB at 6×10⁶, and the measured 1.67 GiB is that plus
the workspace.  So 1.67 GiB is the floor for a backend with no *Gram*
precomputation (the amplitude cache is what a per-event kernel always pays).  `s0e`'s 10.93 GiB is its load-time caches
(`n_wave` 68 flat = `P·N`, `n_uniq` 26, `n_uslot` 38, `n_gamma_rows` 68):
`amp 208 + us 608 + gi 1088 + data 48 = 1952 B/event` → 10.91 GiB predicted,
10.93 measured (100 %); the `gi` term dominates and scales with the flat wave
count.  The handle keeps only `mass` (3 columns), `weight`, `bkg` and `P`;
`q`/`angle` are transient in the fill (`kernels_v4_pwa_s0e.cu:755`).

Their own precheck needs **17.6 GB (16.4 GiB)** at its 3-polarization default and
**11.96 GB (11.14 GiB)** at two, so the memory verdict splits by configuration
rather than being one win: the **fixed** row is 7× below it (2.36 GiB), the flat
tile 4× (2.77 GiB), and the free column is **only 2 % below** (10.93 GiB) — and
their code then dies in the amplitude pass (§10) while both of ours run at the
paper's size.
The reason is *not* that nothing here caches: `s0e` does hold a per-event
amplitude cache (1.95 kB/event = 10.93 GiB, i.e. not far from their table), and
the CPU kernels 1.60 kB/event.  It is that

* their amplitude table `A` is dense over **all** events and all SL x
  polarizations — `34·n_pol·16 B = 1.09 kB/event` at 2 pol, 6.53 GB (6.08 GiB) at 6×10⁶ —
  and their precheck adds an upcast double copy plus workspaces (17.6 GB for the
  3-pol config);
* this framework's fixed path (`integrated_pwa`) needs **no phase-space table at
  all**.  Measured: its footprint is `0.26 GiB + 2.2 kB` per **data** event and
  is independent of the phsp size (10⁶ data gives 2.36 GiB with either 10⁶ or
  5×10⁶ phase-space events); the 5×10⁶ events enter only through the 34×34 Gram.
* where a full per-event cache *is* used (`s0e`, the CPU variants), its budget
  is 1.6-1.95 kB/event over **all** events — 10.93 GiB at 6×10⁶, the only row
  directly comparable with their table.  For the CPU kernels the measured host
  RSS also carries the input sample (~1 GiB at 6×10⁶), so `cached`'s 9.64 GiB is
  an **upper bound** on its 1.60 kB/event cache; comparing two CPU backends is
  unaffected (the sample cancels), the absolute per-event figure is not.

So the per-amplitude-entry costs are **8 B** (one complex per deduped wave, this
framework) against **16 B** per (SL × polarization) entry — 208 B/event against
1088 B/event at this shape, a factor 5.2 — and the fixed row additionally skips
the phase-space table.  Their precheck adds the `dF` table and
its workspaces on top of the `A` table.

**Why the flat tile is another ~3.9x smaller** (`cuda_flat_pwa_gen_s0e2`, 496 B
against the `pwa` s0e's 1952 B per event).  From the allocations and the
runtime constants, not from a guess:

```
flat s0e2  w+b 16 + us 8*n_uslot(33)=264 + amp 8*n_uniq(13)*n_proj(2)=208 + P 8
           =  496 B/event -> 2.77 GiB at 6e6     (measured resident 2.79 GiB)
pwa  s0e   amp 208 + us 16*n_uslot(38)=608 + gi 16*n_gamma_rows(68)=1088
           + data 48 = 1952 B/event -> 10.91 GiB (measured 10.93 GiB)
```

Both tile kernels cache a per-event amplitude (208 B) and a `us` table; the
difference is almost entirely the **gamma-interp table `gi`** (1088 B/event,
more than half the `pwa` budget), which the flat config does not need because
its denominator is a basis sum over shared rows.  `us` is also narrower
(`float2`, 8 B/slot, versus double real+imag, 16 B), and the flat config dedups
to 13 unique wave amplitudes against the `pwa` model's 26.  Its peak is
**3.11 GiB**: the transient fp64 staging of mass/q/angle (168 B/event) is freed
chunk by chunk as the caches fill, so it does not add to the 2.79 GiB resident
footprint — a 100 ms `nvidia-smi` log peaks at 3.27 GiB, which confirms the two
do not overlap.

## 9. The mixed CPU + GPU configuration

`shard` splits the events across a GPU worker and a CPU worker (both speak the
same flat config; `start_method="spawn"`):

| configuration | ms/iter | Mev/s |
|---|---|---|
| GPU tile alone (`cuda_flat_pwa_gen_s0e2`) | 141.0 | 42.6 |
| CPU cached alone (32 thr) | 179.3 | 33.5 |
| `shard` 40/60, OMP 16 | 117.0 | 51.3 |
| `shard` 50/50, OMP 16 | 91.5 | 65.6 |
| `shard` 57/43, OMP 8 | 99.2 | 60.5 |
| `shard` 57/43, OMP 12 | 84.1 | 71.4 |
| `shard` 57/43, OMP 16 | 86.7 | 69.2 |
| **`shard` 57/43, OMP 24** | **82.0** | **73.2** |
| `shard` 57/43, OMP 32 | 92.5 | 64.8 |

(The weight and thread sweep above was measured on the earlier sample; the
corrected point at 57/43, OMP 24 is **82.0 ms / 73.2 Mev/s**, i.e. within 2 %.)

The split optimum is what the single rates predict,
`f* = T_cpu/(T_gpu + T_cpu) ≈ 0.56` at the 32-thread rates (0.57 for the
~187 ms that the 24-thread optimum implies); the **thread count is a second
knob** —
all 32 cores starve the GPU worker's host-side feeding (92.5 vs 82.0 ms), 8
cores make the CPU worker too slow (99.2).  An uncached CPU partner is worse at
every split (125.6 at 50/50, 138.0 at 40/60).

Result: **82.0 ms = 1.72× faster than the GPU tile alone** — at the time the
fastest per-event configuration measured here, and 3.0× their implementation's
fitted 6×10⁶ number (§10).  It is no longer the fastest free-column number: the
`s0g` reaches **73.5 ms** (§6.2) by removing 27 waves' per-iteration
work outright instead of splitting the remaining work across two devices.  It
costs both memories at once:

| split | VRAM | RAM parent | RAM GPU worker | RAM CPU worker | RAM total |
|---|---|---|---|---|---|
| 57/43 | 1.93 GiB | 3.07 | 1.68 | 4.31 | 9.07 GiB |
| 70/30 | 2.34 GiB | 3.00 | 1.90 | 3.17 | 8.09 GiB |

Each side scales roughly with its share — `1.93/0.57 = 3.39` GiB against the
tile's 3.11, `4.31/0.43 = 10.0` GiB against the CPU's 9.64, both within 10 % —
so the split does buy VRAM, at the price of a third body (the parent holding the
full sample).

## 10. Same-box comparison with the CTPWA implementation

The paper never names its device, so its 15 / 100 ms cannot be put against
numbers measured here.  Their sources were therefore built here
(`ctpwa_ref/BUILD.md`) and run on the same GPU, the same
model — their framework confirms it: **"Number of partial waves: 34"**,
`nvec=34`, `ntheta=0` / `10` — the same sample sizes (each framework's own
    generator), and the same **2
polarizations** (their default is 3, which inflates the fixed column ~1.5x, so
`ctpwa_ref/config.yml` sets `polarization: ["1","-1"]` to match our
`spins: [-1, 1]`).

**Why the comparison is meaningful**: a code-level comparison of the two
lineshapes finds the same function up to per-wave constants — identical
propagator `m0² − m² − i·m0·Γ(m)`, identical running width
`Γ = g₀ (q/q₀)^{2L+1} (m₀/m) B'²`, identical Blatt-Weisskopf polynomials `P_L`
(`N_L` in their source; the normalised form factor is `B'_L` ours, `Bf` theirs)
for L = 0…5 with the same radius 3.0, and the same breakup momentum with the
same below-threshold clamp.  The differences are per-wave constants — the `q₀`
reference (1 GeV here, physical `q₀(m₀)` there), the angular/tensor basis, and a
**global** `C = 100/(4π)` (`C·π = 25.0000000164`, spin-independent, so it is the
`1/√(4π)` per vertex and cancels in every ratio) — i.e. a rescaling absorbed by
the coupling.  **The map is `ctpwa_ref/coupling_map.md`**: the closed form
`alpha_w = C·(−1)^(L+1)·B'_L(q0_ref, q0_prod, d)·B'_L(q0_ref, q0_dec, d)` with
`d = 3.0` reproduces all **27 BWR waves** to <1e-6 relative; the **7
non-resonant (`ONE`) waves** have no closed form and their constants are
measured only.  Verified at 2 000 events on identical momenta: one-hot per wave,
all 34 amplitudes match with Pearson correlation 1.000000 and a ratio constant to
a median 9.4e-11; a random complex vector through the map gives a median ratio
0.9999999987 with a robust scatter 1.98e-10 (this framework's interpolation
floor).

**They could not complete the paper's sample here.**  With their 3-polarization
default the precheck refuses (17 560 MB needed).  With 2 polarizations the
precheck passes — 11 960 MB of 12 137, i.e. 98.5 % of the card — and the run then
dies in the amplitude pass
(`CUDA error in computeSLAmp: an illegal memory access`, then
`slamp table needs 192 MiB, only 0 MiB available` — their runtime prints this
line in Chinese).

### Reproduction by sample size

Every entry below is **measured at that size** on this box (RTX 5070 Ti Laptop 12 GB,
Ryzen 9 8945HX), except the values marked `fit` (their 6x10^6 numbers, which
cannot be run here).  Data:phsp is 1:5 as in the paper.
`x` = 66 is every mass/width pinned, 76 is the ten named resonance parameters
floating, 104 is every mass/width floating (38 free resonance parameters on
their side too); their CPU entry is the **x** = 66 configuration.
`theirs, CUDA, x=104` was REBUILT and re-measured here per
`ctpwa_ref/BUILD.md` (all-free = 38 resonance parameters, `ntheta` 38):
92.55 ms at 0.6×10⁶ — `BUILD.md` had recorded 92.69, i.e. −0.2 % — and
182.23 ms at 1.2×10⁶, which is new coverage.  The build is validated, not
assumed: `vary.py config_free.yml` reproduces the documented 26.61 ms as
26.63.  Their PyPI wheel in the `ctpwa312` env is NOT a substitute — it is
`cp312` and that env's torch is `2.12.0+cu126`, which has no `sm_120` kernels
(`no kernel image is available for execution`), so it silently falls back to
CPU (1338 ms at 0.6×10⁶).  The 2.4×10⁶ OOM is the 38-parameter gradient path,
not the sample: the same 2.4×10⁶ sample runs all-fixed at 3.31 ms.

| data + phsp | theirs, CUDA, x=66 | ours, CUDA, x=66 | theirs, CUDA, x=76 | ours, CUDA, x=76 | theirs, CUDA, x=104 | ours, CUDA, x=104 (`s0f`) | ours, CPU | theirs, CPU | ours, hybrid |
|---|---|---|---|---|---|---|---|---|---|
| 1e5 + 5e5 (0.6×10⁶) | 1.16 | 1.3 | 26.61 | 24.2 | 92.55 | **13.8** | 18.8 | 19.17 | **10.2** |
| 2e5 + 1e6 (1.2×10⁶) | 2.04 | 1.7 | 52.12 | 47.5 | 182.2 | **25.6** | 36.7 | — | **18.4** |
| 4e5 + 2e6 (2.4×10⁶) | 3.31 | 2.7 | OOM | 94.5 | OOM | **49.0** | 70.5 | — | **34.0** |
| 6e5 + 3e6 (3.6×10⁶) | OOM | 3.6 | — | 141.1 | — | **72.5** | — | — | — |
| 1e6 + 5e6 (6×10⁶, the paper's) | OOM (fit 7.6) | **6.2** | — (fit 256) | 235.7 | — | **119.0** | 177.2 | — | **84.0** |

All columns in ms per `get_nll` (value + gradient); "ours, CPU" and "ours,
hybrid" are the flat `pwa_gen` family at `rsize=1`, "ours, CUDA, free" is
`cuda_v4_pwa_s0e`, "ours, CUDA, fixed" is `integrated_pwa` + `cuda_v4_pwa_cache`.
The free column is the plain per-event kernel: with the coefficient cache (§6.2)
it becomes **9.3 / 16.6 / 31.1 / 74.8 ms** at 0.6 / 1.2 / 2.4 / 6×10⁶ (27 of 34
waves cached, varying parameters — the numbers the header quotes).

**All resonance parameters free (`ours, CUDA, x=104`).**  The
extreme the paper does not run, measured because the coefficient cache is easy to
dismiss as inert here — it is not.  With no `--fix-mw` / `--free-mw`, `BWR`
masses and widths are all free, 54 of the 60 basis columns move, and the **7
waves whose denominator is a constant** (the non-resonant `ONE` waves) stay
iteration-invariant: 13.8 / 25.6 / 49.0 / 72.5 / 119.0 ms at 0.6 / 1.2 / 2.4 /
3.6 / 6×10⁶, i.e. 1.17-1.19x over the plain tile.  Theirs costs 92.55 ms at 0.6×10⁶ and 182.23 at 1.2×10⁶ — 6.7× and 7.1× ours —
and runs out of memory at 2.4×10⁶.  The gain tracks the CACHED FRACTION:
7/34 → 1.19x, 27/34 → 1.49x (`s0f`) and 1.92x (`s0g`), 34/34 → 4.2x.  It is also
why `s0g` — which pays an un-staged double `D_k` walk — is only 1.07x at
x = 104 and loses to `s0f` at both extremes, winning only in the mixed middle
that production sits in.

Every free-column entry is measured with the **parameters moving** (each call
gets a different `x`), because that is what a fit does; on our side the same-`x`
numbers are inside the run-to-run spread (235.4 → 235.7, 81.9 → 84.0 at 6×10⁶;
CPU 174.1 → 175.5 measured as one pair, the table's 179.3 being a separate run).
Their fixed-θ values are given in the scaling block below and are up to 14 %
lower (their varying-θ cost up to 16 % higher) — a fast path a real fit never
reaches.  Stability check: their free-10 points reproduce across sessions to 0.2 %
(23.11 vs 23.22 ms at 0.6×10⁶, 45.75 vs 45.70 at 1.2×10⁶), and the θ effect
reproduces to the same accuracy.  The fixed-column entries were re-taken with the
harness that synchronises and reports the minimum, which moved the 1.2×10⁶ point
from 1.78 to 2.04 ms; that one entry we could not reproduce and quote the
re-measured value.  Their free column cannot go past 1.2×10⁶: the `dF` table is
`ne · nSL · nFree · 16 B`, but the gradient pass STREAMS it, so that product is
the per-chunk working set, not the resident footprint — MEASURED peak 4.38 GB at
0.6×10⁶ and 7.52 GB at 1.2×10⁶ with 38 free, and an OOM at 2.4×10⁶ (peak 11.7 GB
of 11.5 GB free, in `torch.autograd.grad`).  An earlier version of this file
quoted ≈33 GB at 6×10⁶ from the product, which is wrong for that reason.  Their fixed column could not
complete 3.6×10⁶ or 6×10⁶ either — same amplitude-pass failure as §10's full-size
2-polarization run, not a precheck refusal.

**Reading it**

* **fixed column**: theirs is faster at the smallest rung (1.16 vs 1.3 ms) and
  ours from 1.2×10⁶ up (~17 % at 1.2×10⁶, ~18 % at 2.4×10⁶), reaching 6.2 ms
  (measured) against their fitted 7.6 ms at the paper's size — their fit is
  linear through the three measured fixed-column rungs, since they cannot run
  the paper's size at all.  Their fixed path is a
  matrix-vector product and was verified to be value-independent (1.09 ms with
  the same 1.1 ms either way: 1.09 with one coupling set against 1.08 with all
  34 nonzero and re-drawn per call, and 1.16 / 1.01 on a re-run — a 1 ms call is
  noise-dominated, so there is no fast path to find at zero parameters).
* **free column, pure GPU**: ours is faster at every rung — 24.2 vs 26.6 ms at
  0.6×10⁶, 47.5 vs 52.1 at 1.2×10⁶, and 235.7 against a fitted 256 ms at the
  paper's size (~8 %).  This is the opposite of what two earlier versions of this
  table said (first "2x faster", from their config freeing 20 parameters instead
  of ten; then "a tie", from measuring them at fixed θ).
* **CPU**: at 0.6×10⁶ our CPU kernel (18.8 ms) beats *both* GPUs' free column,
  and theirs (19.17 ms, fixed configuration — their free path on CPU was not
  run) is equal to ours — but its initialisation is 56 s against our 1.0 s
  (model build + first NLL).  Ours stays ahead of the per-event `pwa` GPU kernel
  at every rung; what overtakes it is the **flat tile** (141.0 vs 177.2 ms at
  6×10⁶), whose own ladder was not measured.
* **hybrid** (`shard` 57/43, GPU + CPU): the fastest **free-column**
  configuration at every size *when measured*, and still the only one that uses
  both memories — 10.2 ms against their 26.6 at 0.6×10⁶ (2.6x)
  and **3.0x their fitted 6×10⁶ number** (84 vs 256 ms).  Since then the
  `s0g` reaches 73.5 ms at 6×10⁶ (§6.2) by caching the 27
  iteration-invariant waves; the two are not exclusive (a `s0g` GPU worker plus
  a CPU worker is the obvious next measurement, and is not claimed here).
  The fixed column is a different
  algorithm and stays far faster (6.2-40.8 ms).
* **initialisation**: 3.0 s (ours, `integrated_pwa`) against 253 s (theirs, a
  linear fit through the measured 151.3/163.2/185.2 s) — 84x, and paid once per
  fit; against the 1.8 s `s0e` initialisation it is 141x.

**The free column scales differently in the two designs — and their quoted
numbers carry a fast path.**  Their cost grows with the number of floating
parameters (the `dF` table is `ne · nSL · nFree · 16 B`, streamed by the gradient
pass, so the resident set is a chunk of it and not the product), *and* it depends on whether the parameters actually move: a call that
repeats the previous `theta` skips work a real fit cannot skip.  Both are
measured here, at 0.6x10^6, on their CUDA backend:

```
floating params      theirs, fixed theta / varying theta   theirs, 6x10^6 (scaled)   ours s0e, 6x10^6
    0                        1.09  /  1.08                         7.6                    235.5
    4                        7.63  /  8.35                        ~80                    235.4
   10                       23.23  /  26.63                        256                   235.5
   20                       46.07  /  53.33                       ~514                   235.4
   38 (all)                 79.84  /  92.55                       ~893                   235.4
```

Rows 10 and 38 were RE-MEASURED on a fresh `sm_120` build (see §10); rows 4
and 20 are from the original run and were not re-taken.  The gap between the two
theta columns is **+9 % at four parameters and +16 % at
thirty-eight**, i.e. it is not a constant overhead but part of the per-parameter
work; the 0 row (their fixed algorithm, no `dF` table) is the control and shows
none of it at the 1 ms level (1.09 / 1.08 here; 1.16 / 1.01 on a re-run, i.e.
noise either way).  The
6x10^6 column is scaled from the 0.6x10^6 one by the factor the ten-parameter row
gives from two measured rungs (0.6 and 1.2x10^6: 26.61 → 52.12 ms, hence
`1.1 + 42.5 ns/event`, i.e. x9.63); the other rows assume the same factor, which
is why they are marked `~`.

Over the free points their varying-theta cost is `~1.0 + 2.46*n` ms at 0.6x10^6
(fixed theta: `~1.4 + 2.10*n`), i.e. linear with a near-zero intercept — the
intercept is the fixed-column algorithm, the slope is the per-parameter `dF`
stream.  Ours is **flat** across 0/4/10/38, because the kernel computes analytic
gradients for every parameter regardless of how many float; the CPU and hybrid
columns are flat too (CPU 174.1 same-`x` vs 175.5 varying, both in one process —
the table's 179.3 is the same kernel in a separate run, a ~3 % spread; hybrid
82.0 vs 84.0).

So the crossover is at **≈9.3 floating parameters** (`235.7` against their
interpolated 4 → 10 marginal of 29.3 ms per parameter): below it theirs is
faster (3x at four, ~80 vs 235 ms), above it this framework is (2.2x at twenty,
235 vs ~514 ms; 3.8x with every parameter free, 235 vs ~893 ms).  The paper's ten
therefore sits just *past* the crossover, not on it — at their stated ten this
framework is ~8 % faster, and the hybrid (84 ms) is three times faster.

**Why the paper's own numbers are not a yardstick**: a single hardware factor
would preserve the ratio between the two columns, but the paper's free/fixed
ratio is 6.7 (100/15 ms) against **33.7 measured here** (256/7.6) — a 5× change
in the ratio, which no device scaling produces.  Their fixed row is
host/launch-bound and this laptop is *faster* than their device there (7.6 vs
15 ms), while the free row is fp64-bound and their device is 2.6× faster at it
(256 vs 100 ms).  That is a statement about their measurement setup, and it is
why this section compares implementations run the same way rather than table
entries.

What this does **not** say: their `dF`-table design is the memory-cheap route in
the *floating* column (they stream a table where this framework recomputes), and
their analytic Hessian (0.35 s / 1.6 s) has no counterpart here.

## 11. Caveats

* **The paper does not state its device**, so its timings are not used for any
  comparison here; §1–§9 report our numbers, and §10 runs their implementation
  on this box.  Their precheck needs 17.6 GB at their 3-polarization default,
  and with 2 polarizations (11.96 GB) it passes the precheck but dies in
  amplitude computation — so their code cannot complete the paper's sample on
  this 12 GB card either way.  Every number measured here is from
  an **RTX 5070 Ti Laptop (12 GB, sm_120)**.
* **The sample is flat 3-body phase space** (unit weights), built by the
  framework's own generator and event builder — the *timing* is comparable
  (same size, topology, per-iteration work), the NLL value is not physical.  A
  model toy would need `Fitter.generate_toy` (engine `toy_pwa_generate`), whose
  rejection acceptance here is ~0.2 % (~5×10⁸ candidates for 10⁶ events).
* **No analytic Hessian**, so the Hessian row of their table is not reproduced;
  `get_nll` (value + gradient) is the comparable quantity.
* Masses/widths are PDG values; the paper's fit results differ slightly and do
  not affect the timing.

## 12. Three traps this harness encodes

1. **Table-based backends extrapolate silently.**  `numpy_flat`, the flat CUDA
   family and the `s0c/s0d/s0e` kernels read `mass`/`q` from interpolation
   tables; an out-of-range sample makes them disagree with the analytic
   backends by orders of magnitude on the same parameters, with no error.
   `make_data.py` asserts both arrays against the grids of the config it loads
   (here the two configs' grids coincide).
2. **Device caches are per handle.**  `s0e`'s ~2 kB/event caches are allocated
   on every `load_data`, so loading the same data a third time asks for
   ~13 GiB on a 12 GiB card and the backend dies without a message.
   `check_one.py` reuses `Fitter._data_holder`.
3. **Four-vectors are `(E, pₓ, p_y, p_z)`** (`generate_pwa_phsp` writes energy
   first; the loaders read `m4[..., 0]` as E).  Passing `(pₓ, p_y, p_z, E)`
   produces all-NaN event data rather than an error.
