# AGENTS.md — tabpwa

Amplitude analysis fitting framework.  NumPy / CUDA / ONNX backends, Wirtinger gradients, BFGS fit with constraints.

## Quick Start

```bash
pip install -e .
./fit.sh                                          # full pipeline
python run_fit.py --fit --maxiter 1000 --backend integrated  # or explicit
```

## Commit hygiene

Never chain a commit with the test/check commands.  Run the tests as a
separate step, verify green, then commit as a **single standalone git
command**.  A red or incomplete verification must never be committed,
and a commit must never ride on the tail of a test pipeline.

**Never commit unfinished work.**  A commit is a self-contained unit that
must stand on its own: the feature it names is implemented, tested and
verified *in that commit*, not "will be fixed in the next one".  Concretely,
do not commit when any of these is true:

- a test for the change is written but failing (or not written yet);
- a guard/validation the change obviously needs is missing, or a known
  defect in the new code path is still open;
- the change only makes sense together with an uncommitted edit elsewhere
  (a half-migrated API, a doc that contradicts the code);
- the verification had to be narrowed (skipped tests, a smaller suite, an
  environment workaround) and the reason is not part of the commit message.

If the work is in that state, keep it in the working tree (or stage it),
say what is missing, and wait — do not "checkpoint" it as a commit.  A
follow-up fix commit is for problems *discovered later*, never for work
the author already knew was incomplete at commit time.

## Backend Selection

| Backend | Flag | Note |
|---------|------|------|
| **Integrated** | `{"name": "integrated", "base": "cuda_v3_cache"}` | Gram matrix O(n²) norm + base backend for data NLL. TD model only (`n_blocks=8`); pure PWA uses `integrated_pwa`. Default base `cuda_v3_cache`. |
| **v3 amp-cache** | `cuda_v3_ampcache` | v3 sparse + per-handle cache of the minimal-set angular amplitudes (`Amp = fa·fl`, 272 unique/event). Forward per wave = `ck·Amp_slot[w]/bw_p`, so only the BW propagator is recomputed each iteration (m0/g0 stay fitted, grads flow). momentum/angle are uploaded only transiently for the fill and **not kept** on the GPU. fp64 fill → float2 store (constant over fit). ~1.7× faster than sparse per iteration. |
| **v4 PWA** | `cuda_v4_pwa` | No time/mixing/scalars: `P = Σ_p |Σ_k ck_k·a_{p,k}|²`. Derived from ampcache (same angular cache + per-iteration BW). All projections share one `ck`; the projection only changes the angular part. Wave entries stored p-major: `n_wave = n_proj·N`, `ck` length `N = n_wave/n_proj`. Config key `n_proj` (default 1). |
| **v5 PWA s0cw** | `cuda_v5_pwa_s0cw` | `cuda_v5_pwa_s0c` with the **group-weighted log NLL**: `Q = −Σ_g W_g·log[(Pb_g/norm + Bb_g)/W_g]`, `W_g = Σ_{e∈g} w_e` (the group total weight, a third grp_sum output). dnorm `Σ W_g·Pb_g/(norm·(Pb_g + norm·Bb_g))`; gradient scale `c_g = −W_g/(Pb + norm·gB)`. Zero-weight groups inert (w > 0 guard). With group-normalised weights (Σw = 1 per group) identical to the s0c. Fixture `tests/test_pwa_v5_s0cw.py` (run alone). |
| **v4 PWA cache** | `cuda_v4_pwa_cache` | Same pure-PWA model, **full-amplitude cache for fixed m0/g0** (fpwfitter-style): C keeps only weight/bkg + the cached `common[e, p·N+k]`; mass/momentum/angle never live in C — the Python DataHandle retains them and builds the cache on the FIRST `compute()` from the params m0/g0 (Python-level `_filled` flag; no C-side state). Verified bit-equal to `cuda_v4_pwa` at fixed m0/g0 (Q/P/ck-grad; m0/g0 grads zero). **Strictly fixed-m0/g0**: no auto refill/switch; later m0/g0 changes raise ValueError — float masses/widths ⇒ use `cuda_v4_pwa`. |
| **v4 PWA 1bw** | `cuda_v4_pwa_1bw` | One BW **per chain** specialisation of `cuda_v4_pwa` (`top → R + X`, `n_res == 1` enforced by the wrapper — multi-BW chains raise). Same config input and conversion; the kernel drops the per-wave BW product loop (the propagator IS the single denominator) and gives every wave its OWN BW slot — the wrapper merges `matrix_gamma` (gamma→bw) with `bw_order` (bw→wave) into per-wave `m0_index`/`mass_index`/gamma rows (`row_start` ranges): no `bw_order` indirection, contiguous BW
  arrays, no atomics, and the running-width interpolation is FUSED into the BW kernel (separate g_bw launch removed). NLL bit-exact vs `cuda_v4_pwa`, gradients ≤1 ulp (separate compilation) — e.g. `config_pwa.yml`-style models; `tests/config_pwa_bw.yml` is the free-m0/g0 fixture. Sibling
  **`cuda_v4_pwa_1bws`**: same kernel binary + mass-sorted `ck_order` — the wrapper reorders base waves by
  their data mass column (stable) so per-wave BW arrays are contiguous runs; pure outside transform (ck in, gradients/Gram
  un-permuted out), vectorized bincount m0/g0 reductions — for A/B comparison against `cuda_v4_pwa_1bw`. Sibling
  **`cuda_v4_pwa_s0`**: same one-BW-per-wave structure with DIRECT pole parameters — the wrapper passes
  `s0 = m0^2` (per wave) and `mult = m0*g0` (per gamma row) so the device formula is `bw = s0 - m^2 - i*mult*G(m)`;
  the device outputs dQ/ds0 and dQ/dmult (from the stored g_interp rows) and the wrapper chains them to the fitted
  m0/g0 on the host — no m0/g0 gradient algebra on the device, no m0_index on the GPU. |
| **pwa_gen (flat config)** | `numpy_flat` | The `pwa_gen` model's own kernel vocabulary, built by `tabpwa/pwa_gen_config.build_from_groups` from the decay groups.  Per (expanded wave `k`, projection `p`, event `i`): `F_ipk = fa_pk · Π_d fl_table[k,d](q_{i,fl_q_index[k,d]}) · Π_r num_table[k,r](m_{i,num_m_index[k,r]})`, `fa_{p·n_wave+k} = Σ_a matrix_angle[a, p·n_wave+k] · Π_j cos(angle_{i,angle_index[a],j}·angle_k[a,j] + angle_b[a,j])` — i.e. `fa = ka @ matrix_angle` with `matrix_angle` ``(n_angle_basis, n_proj·n_wave)`` **p-major**, the long-standing layout, so `n_proj = matrix_angle.shape[1] // n_wave` and no separate projection axis exists, `D_ik = Π_l Σ_m gamma_matrix[k,l,m]·basis_table[m](m_{i,basis_m_index[m]})`.  `fl` and `num` are **fully expanded** (one table row per use — no `fl_type`/`n_index` indirection); the angular basis is keyed on `(topo, k, b)` so the waves of one topology share rows and `angle_index[a]` is the topology's angle-column base; the denominator basis is **one flat appended list** (a row per (factor, basis row), duplicates kept) selected by the 0/1 `gamma_matrix`.  `den_names` is the parallel coefficient-name list, so the fitted block is `coeff` of length `n_basis` with no padding and no copy-sum.  **No counts are stored** — `n_wave` / `n_proj` / `n_res` / `n_decay` / `n_basis` are all `shape[...]` and the backend asserts their agreement once at create time.  `params = {"ck", "coeff"}`: `ck` is flat over `g_ls ⊗ n` (one slot per expanded wave, `len == n_wave`) via `FlatKernelParams`; `coeff` is `(n_basis,)`.  `dedup(config)` is a separate pass that folds rows equal **and** sharing their mass column, emitting `fl_index`/`num_index` and a remapped `gamma_matrix`; the backend accepts either layout.  Fixture: `tests/test_pwa_gen.py`. |
| **pwa_gen (flat config), CUDA** | `cuda_flat_pwa_gen` | Case-1 copy of the flat config on the GPU (own `.cu`/`.so`/wrapper/symbols, `kernels_flat_pwa_gen.cu` + `cuda/_flat_pwa_gen.py`).  One fused kernel over a tile of `FLAT_TE=4` events: `ka` is evaluated once per (event, angle basis) and reused by every (wave, projection), the barrier and numerator products once per (event, wave), the per-factor denominators `s_{e,k,l}` are kept in shared memory (the coeff gradient needs both `s_l` and their prefix·suffix product, never a division by a factor that can vanish).  Reads the **full, non-deduped** config only — a `dedup`ed config is rejected with a clear error, because the folded form indexes through `fl_index`/`num_index` which this kernel does not read.  `params = {"ck", "coeff"}`, `grad_ck` is the raw Wirtinger half (the 2·Re convention is applied by `CKProduct.backprop_grad`, same as `numpy_pwa`/`numpy_flat`).  Anchors vs `numpy_flat` on both fixtures at `norm=None` and scalar norm: `Q`/`P`/`ck`/`coeff`/`dnorm` **1e-8** (the float2 amplitude-cache floor); `dcoeff` is exact against central differences (ratio 1.0) and `dck` matches `2·Re`/`2·Im` of it.  **Performance ~1.3 M/s** on the 100-wave/2-projection benchmark shape (49.9 ms at 65k events) — it recomputes every interpolation per iteration by design (no caches), so it is the correctness reference rather than the tuned path; `cuda_flat_cache_pwa_gen` is the fast sibling.  Fixture: `tests/test_cuda_flat_pwa_gen.py`.  Test data must lie INSIDE both interpolation grids (outside, the tables clamp and CPU/GPU diverge). |
| **pwa_gen (flat config), CUDA cached** | `cuda_flat_cache_pwa_gen` | Case-1 copy with two changes over `cuda_flat_pwa_gen`: (1) **cached F** — `F_{e,k} = fl·Π num` does not depend on `ck` or `coeff`, so it is filled once in `load_data` into a `float2` array and reused every iteration; only the denominator (which carries `coeff`) and the projection sum run per compute; (2) **backend-internal dedup** of the basis rows on `(coeff name, basis kind, mass column)` — the row's physical identity.  The fold happens in the backend on its own copy, so `pwa_gen_config` and every other backend are untouched: the builder still emits every row and this backend folds 10→5 / 32→8 locally.  Folding on the structural key rather than sampled values cannot merge rows that merely look alike numerically (two identical `kmatrix2` models at different mass columns stay apart).  The caller keeps the FULL-form `coeff`; the wrapper gathers it onto the survivors and scatters the gradient back so each weight's derivative appears **once** (folded copies zero), which is what the naming transform consumes — replicating it onto the copies would multiply the gradient by the copy count.  **The cached `F` alone gives ~1.00×** against `cuda_flat_pwa_gen` (measured interleaved at 20k/65k on all three fixtures: 0.96–1.02×), because `F` only removes the `fl·num` products — `n_wave·n_res` interpolations — while the dominating cost is the denominator (`n_wave·n_res·n_basis` basis interpolations, which depend on `coeff` and cannot be cached) and the `fa` reconstruction.  The **superseded** 1.8× figure compared against the unfolded basis, i.e. it was measuring the dedup, not the cache.  **The two hot spots, both fixed and measured** on the 100-wave/2-projection shape (65k events, interleaved): (1) phase 2 was one thread per `(e,p)` — only `evn·n_proj` = 16 of 256 threads — each serialising `n_wave` complex divisions; it is now a lane-split reduction (threads divided into `nlane = ceil(nt/(evn·n_proj))` groups, lane `L` taking `k ≡ L mod nlane`, closed with one shared `atomicAdd`); (2) phase 5 looped `for (k = tid; k < n_wave)`, so again only `n_wave` threads worked, each running `n_res·n_basis·evn` serial iterations — it is now two fully parallel passes (pass A over `(e,k)` writing `dQ/ds` to its own shared buffer `s_pf`, pass B over `(k,l,m)` reducing into shared `s_gc`, flushed once per block).  Together: **13.5 → 8.6 ms (1.57×)**.  `s_pf` must be a SEPARATE buffer from `s_s`: `Π_{l'≠l} s` reads every factor, so writing results back into `s_s` corrupts the later factors (that aliasing bug cost one debugging round).  Beyond `FC_MAXRES = 16` factors the kernel falls back to the per-wave shape rather than truncating.  (3) **`1/|D|²` is cached once in phase 1** — the biggest single win.  `D = Π_l Σ_m γ·basis_m` is COMPLEX (the basis is complex), but `|D|²` is a REAL scalar, so the shared divisor of `F/D` and of `dQ/dD` is one real reciprocal.  Phases 2, 4 and 5 each redid the fp64 complex DIVISION — `ne·n_wave·(n_proj+2)` = 26 M divisions per call at the benchmark shape — against a consumer GPU's ~1/64-rate unpipelined fp64 divide.  Caching the scalar turns them all into multiplies: **8.6 → 6.5 ms (10.0 M/s)**, so this kernel is now FASTER than `cuda_v5_pwa_s0e` (7.3 ms) and 1.8× its own starting point.  (4) **phase 2 folds `ck` in before the projection loop.**  `D` depends on the EVENT (via `basis_m(m_e)`), so `w_{e,k} = ck_k·F_{e,k}/D_{e,k}` is shared by every projection: thread `(e, lane)` now owns all `n_proj` outputs of its `k` slice and forms `w` once, instead of rebuilding `F/D` and re-multiplying by `ck` per projection (28 → 18 multiplies per `(e,k)` at `n_proj = 2`).  Measured gain is only ~0.5 % (6.53 → 6.48 ms) because phase 2 is no longer the bottleneck — worth keeping for the smaller op count, not as a speedup.  `FC_MAXPROJ = 4` bounds the per-thread accumulator array before it would spill; past that the one-thread-per-`(e,p)` shape returns.

**Shared memory is the binding constraint, and it is why caching `common = F/D` was REJECTED**: an extra `float2[FC_TE·n_wave]` pushes the block from 44.8 to 57.3 KiB, dropping occupancy from 3 to 2 blocks/SM at 164 KiB/SM, and measured 6.48 → 6.75 ms.  Any future buffer here must be checked against that trade, not just against the SM's opt-in limit.  Note the launch also needs `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` on the FUNCTION — a `smem_bytes <= optin` check alone is not enough.  Without it a tile above the 48 KiB DEFAULT is rejected at launch with "invalid argument", and because `CUDA_CHECK` only PRINTS (it does not raise) and the post-launch check is never reached, `compute` returns `Q = 0` at exit status 0.  Verified on the pre-fix binary: `n_wave` 56 (25.5 KiB, no opt-in) → 13515.78, `n_wave` 110 (49.2 KiB) → **0.0**, `n_wave` 200 (88.5 KiB) → **0.0**.  The supported ceiling is set by the device: `MaxSharedMemoryPerBlockOptin` is 99 KiB here, i.e. `n_wave <= 223` for this shape; past that `create` refuses with a clear error rather than returning 0.  **The missing opt-in was latent at `n_wave = 100` and only became reachable here:** the ORIGINAL tile was 26.0 KiB at `n_wave = 100` (first failure at 188, measured: 188 → ok, 200 → 0.0), while the buffers added in this session ([`s_pf`](`2·FC_TE·n_wave·n_res`), `s_rd` (`FC_TE·n_wave`)) took it to 44.8 KiB and the threshold down to 108 — so the work here made a pre-existing hole reachable rather than creating it.  Gated by `test_wide_config_crosses_the_shared_memory_optin` (every other fixture is `n_wave ≤ 2`, which is why it survived this long).

Benchmark: `tests/benchmark_flat_vs_pwa.py` (interleaved, all kernels live at once, `--rsize` for the grouped objective).  Anchors vs `numpy_flat`: `Q`/`P`/`ck` 1e-8; the per-name coefficient total 3e-8; all 7 free parameters 5.9e-8 against central differences through `Fitter.get_nll`.  Fixture: `tests/test_cuda_flat_cache_pwa_gen.py`. |
| **v5 PWA log-sum** | `cuda_v5_pwa` | Same projection-sum PWA as v4, but the **data NLL does not log per event**: events are grouped into `resolution_size`-sized chunks (backend kwarg, default 1) and one log is taken per group: `Q = -Σ_groups log Σ_{e∈g} w_e·(P_e/norm + bkg_e)`. Groups align to the event index (independent of the GPU `batch_size`). Per-event returned `P` stays `P_sig = Σ_p|A_p|²`; the phsp path (`norm=None`) and `compute_gram` are unchanged/linear. `dNLL/dnorm` is accumulated per group on the C path and exposed via `dnorm_on_gpu`/`_last_dnorm`. `resolution_size=1` with unit weights reproduces v4 exactly. For NLL comparisons against v4 at `resolution_size > 1`, the copy rows of each group must carry group-normalised weights (`w = 1/resolution_size` each, group total 1) — otherwise the group sum `Σ w·pdf` carries an extra `resolution_size` factor.  With that normalisation `Σw == n_groups`, so the fitter's purity constant `−log(p)·Σw` is exact for the group objective too; with unit per-row weights a spurious `(Σw−n_groups)·log p` remains when purity<1. |
| **v5 PWA log-sum s0** | `cuda_v5_pwa_s0` | `cuda_v4_pwa_s0` (one BW per wave, direct pole parameters `s0 = m0²`/`mult = m0·g0`, device outputs dQ/ds0 + dQ/dmult, host chains to m0/g0) + the v5 grouped log-sum data NLL (`resolution_size` kwarg, default 1; `dNLL/dnorm` per group via `dnorm_on_gpu`/`_last_dnorm`). `resolution_size=1` with unit weights is **bit-identical** to `cuda_v4_pwa_s0` (NLL + grads); at `rsize > 1` the NLL matches `cuda_v5_pwa` to ≤1 ulp (the s0 BW form rounds differently from the generic `m0·g_interp` path). Fixture tests `tests/test_pwa_v5_s0.py`. |
| **v4 PWA s0b** | `cuda_v4_pwa_s0b` | `cuda_v4_pwa_s0` with device-side output accumulation: Q, dNLL/dnorm and the gradient partials tree-reduce into persistent context accumulators (one small blocking D2H per call instead of 7 per batch), and the per-call parameter upload uses pinned host staging + `cudaMemcpyAsync` (pageable sync H2D costs ~0.9 ms/call on some drivers). Identical interface/math; summation order differs at ~1e-16. Fitter-level NLL+grad 5.2× faster at 20k events, 2.4× at 100k. Profiling note: the GPU kernels are only 10–30% of a compute call — the per-iteration cost lives in host-side copies/overhead; a per-handle g_interp cache showed NO gain (the gamma interpolation is 0.8–7%). |
| **v4 PWA s0c** | `cuda_v4_pwa_s0c` | s0b with the **uniform-coefficient format**: the denominator is the basis sum `BW_w = Σ_i M_wi·F_i` (`F_0 = s0 = m0²`, `F_1 = −1`, `F_l = −i·G_l`), so every fitted coefficient shares one gradient formula (K5+K6 fused; K4/grad_coeff on the K3 multi-event scheduling). Tile sibling `cuda_v5_pwa_s0c` adds the grouped log-sum NLL. |
| **v4 PWA s0d (u-cache)** | `cuda_v4_pwa_s0d` | s0c with the **iteration-invariant γ-interp cached at load**: the interp `(gamma-table row, mass column)` depends only on fixed inputs, so a `us_cache_fill_kernel` fills `us/real|imag [ne·n_uslot]` (+ the per-row `g_interp` for the grad kernel) ONCE per data load, and `amp_cache_amp_kernel` becomes a pure matvec — no interp, no shared staging, zero `__syncthreads()` per iteration. Results identical to s0c. Gain = f(interp share): 1.64× on the narrow-wave template (6 waves / 5 uslots, 11.39→6.93 ms @10⁶), ~1.0× on wave-dense synthetic shapes (there the matvec + `[ne·n_wave]` write streams bind, identical in both). Costs ~`ne·n_uslot`·16 B extra per handle (fp64 kept for the bit-identical contract). Sibling `cuda_v5_pwa_s0d` ports the same u-cache onto `cuda_v5_pwa_s0c` (log-sum group kernels unchanged; identical results at every `resolution_size`). |
| **v3 sparse** | `cuda_v3_sparse` | Split-kernel + sparse scatter/gather for matrix_gamma (99.5% sparse). FP32 FA + float momentum/angle. **3.8× faster than v3** (271→70 ms). FP64 for g0/m0/BW. |
| **CPU (C+OMP)** | `cpu_v3`, `cpu64_v3` | Pure C + OpenMP + AVX2. Sparse scatter/gather. **6.3× faster than NumPy** (2459→391 ms at 10K). No GPU required. Auto-built via gcc. |
| CUDA f64 v3 | `cuda_v3`, `cuda`, `cuda64` | Catmull-Rom (default standalone). |
| CUDA f32 v3 | `cuda32_v3` | Faster, ~1e-7 precision. |
| CUDA f64 v2 | `cuda_v2`, `cuda64_v2` | Linear interpolation. |
| CUDA mixed v3 | `cuda_mixed_v3` | Mixed fp32/fp64 v3 variant. |
| CUDA f32 v2 | `cuda32_v2` | Fastest raw kernel. |
| NumPy | `numpy` | CPU reference implementation (f64). |
| NumPy PWA | `numpy_pwa` | CPU projection-sum reference implementation (plain `pwa` only — see `numpy_flat`). |
| NumPy PWA gen (flat) | `numpy_flat` | `pwa_gen` reference backend (`NumpyFlatBackend`) over the flat config: plain loops, no index pre-folding, no cache — it reads each table at the index the config names and derives every count from `shape[...]`, asserting their agreement once.  Accepts the full **and** the `dedup`ed layout (`fl_index`/`num_index`).  Gradients are the same algebra `numpy_pwa` uses (`grad_ck` is the raw Wirtinger half — `CKProduct.backprop_grad` applies the 2·Re convention), verified against central differences at 1e-9. |
| Integrated PWA | `integrated_pwa` | Gram-matrix norm for the projection-sum PWA + base (default `cuda_v4_pwa`). |
| v4 PWA cache f32 | `cuda32_v4_pwa_cache` | fp32 variant of `cuda_v4_pwa_cache`. |
| Shard | `shard` | Multi-process wrapper sharding data across workers (`backends`, `weights`, `align`, `start_method`). Universal (`model=None`). |
| ONNX | `onnx_cpu` / `onnx_cuda` | In-memory graph, built with batch_size=1024. |

`cuda` / `cuda64` alias to `cuda_v3`.  GPU `__del__` auto-frees memory — no manual `.free()` needed.

### Forward-mode duals (`tabpwa.fwd`)

`Fwd(value, grad)` is a scalar with a value and a dense gradient over a
**fixed** parameter list; `+ - * / pow sqrt log` propagate the gradient.  No
layout object: the parameter space is the gradient vector's own index space,
fixed once by the fitter —

```python
f.fwd_names()                       # the free parameters, x-slot order
f.fwd_parameter("rho_mass")         # unit tangent: grad[i] = 1 at that slot
f.fwd_of(total, grads, x=x)         # from a compute()'s (value, {name: d}) pair
f.fwd_backward(kernel_grads, resolved)   # ck/m0/g0 blocks -> parameter names
f.sum_stream(fn, batch=50_000)      # sum a chunked contribution
f.sum_stream(params, resolved, x=x, batch=50_000)  # or param dicts
```

`sum_stream` is a **sum**, not a forward-mode feature: a contribution may be a
number, an array or a `Fwd`, or any pytree of those, and the chunks are added
up leaf by leaf (`fwd.batch_sum`: the `tfpwa.data.data_merge` recursion with
`+` as the leaf operation, which also carries plain values — that is how the
finite-difference route streams).  `fwd.data_split` does the chunking, chunks
**outer** so one `load_data` serves every output, and only one accumulated tree
is ever alive.

so every `Fwd` in one fit shares the ordering and the operators raise (not
broadcast) when two different spaces meet.  `sigma(cov)` gives
`sqrt(g C gᵀ)`; `a.cov(b, cov)` / `a.correlation(b, cov)` (alias `corr`) the
pairwise covariance and correlation (`0` when either side is a constant), and
`corr_matrix(duals, cov)` the full matrix for a list of observables.

Elementary functions, usable on duals and plain numbers alike: `sin`, `cos`,
`exp`, `log`, `sqrt`, `erf`, `gamma`, `loggamma` (via
`scipy.special`, derivatives `2/√π·e^{−x²}`, `Γ·ψ`, `ψ`), `atan2(y, x)`,
`mod(a, b)` (floor semantics, gradient `g_a − floor(a/b)·g_b`), `abs` (`abs(x)` = real modulus with the sign in the
gradient; `abs(re, im)` = the complex modulus of a pair), and the
polar/rectangular pair — `reim(magnitude,
phase)` gives the two real parts of `m·e^{iφ}` (`re² + im²` recovers `|z|²`,
gradient included), `polar(re, im)` and `magnitude(re, im)` the inverse.  So
a complex model quantity is carried as two real duals and a real observable
(e.g. `abs(lineshape)**2`) is formed from them.

Each of those is written **once** and decorated with `@dual` (exported),
which simply converts every argument to a `Fwd`: a number is just
`Fwd(value, 0)`, i.e. a **scalar** tangent, and numpy broadcasts it against a
real parameter vector — so a mixed expression like `2*sin(x) + 1` keeps the
vector gradient.  The call always returns a dual (tuples elementwise); take
`.value` for the plain number, and `.sigma` of a constant is 0.  There is no
second implementation to keep in sync:

```python
from tabpwa.fwd import dual, Fwd

@dual
def lorentz(m, m0, w):
    return w * w / ((m * m - m0 * m0) ** 2 + w * w)   # number or dual
```

`AmplitudeFractions` (fraction = `Σ w·|A_mask|² / Σ w·|A_all|²`) provides only
the **per-handle integrand**; chunking, loading and accumulation are the
fitter's (`sum_stream`), never the fraction class's:

```python
af  = AmplitudeFractions(fitter, fit_result)
out = f.sum_stream(lambda h: af.fwd_integrate([[1], [2], [1, 2]], h),
                   data=arrays, batch=50_000)      # [Fwd, Fwd, Fwd]
R   = out[0] / out[2]                              # value + exact gradient
band = R.sigma(fit_result.hess_inv)
```

The sample is given as **arrays** (`data=` a numpy dict) or left `None` for the
fitter's phase space — never a handle.  The handle resolves lazily: the given
arrays (loaded here, freed by `close()`), else `fitter._phsp_holder` from
`set_phsp` (used as-is, never freed), else a load of `fitter._phsp_np`.  A
batched run slices whichever arrays are in play, so it integrates the sample
you asked for, not the fitter's.  The load is **lazy and never whole-sample** in
that mode: `batch=` streams one chunk per `load_data` (a 120-event sample at
`batch=32` gives loads of 32/32/32/24 and leaves `af._handle is None`), so the
peak memory really is one chunk.

A `Fwd`'s gradient is **`x`-space** — the space `hess_inv` lives in — and the
only reduction from the kernel's resolved-name gradient to it is
`cm.full_gradient(grad_dict, resolved, x)`.  Resolved is *not* that space: it
holds aliases (`rhoB_width` is `rhoA_width`), bounded values
(`resolved = bounds(x)`, so `d resolved/dx` is 0.011 for `rhoA_width` in the
`config_amp` fixture) and unpacked names.  Indexing resolved names literally
into the `fwd_names()` vector drops 198 of 349 gradient names and gives
`sigma` 0.00759 instead of the true 0.003813.

`Fitter.fwd_of(value, grad_dict, resolved=None, x=None)` is that single
assembly point, and **requires** the point: no fallback to `_last_xk` (a stale
`x` chains to the wrong gradient while the value stays identical, so nothing
downstream can notice).  `sum_stream(fn, resolved, x=…, batch=…)`'s param-dict
form demands `x` for the same reason.

`af.fwd_integrate(masks, handle, phys=None, x=None)` delegates to it and
returns one `Fwd` per mask; it never loads or frees, which is what makes it a
`sum_stream` chunk callback.  A mask over a *fixed* coupling legitimately has
a zero gradient.

`fractions(masks, denominator=None, param_names=None, jac=True, batch=None)`
— `batch=None` keeps `cal_uncertainties_multi` (analytic gradients by resolved
name, reduced to `x` inside it), `batch=N` streams
through `sum_stream` and takes `R.sigma(hess_inv)` on the dual that already
spans `x`.  All three agree to 1e-9, blinded or not.

**Which frame they evaluate at is not a detail.**  Blinding is display-only:
the program always works with the real physics (`build_params(x)[1]`, the
model frame — what the kernel, the likelihood and the fit use), and
`report_params(x)` returns those real values *minus* the seed-derived offset,
i.e. the real value hidden behind an unknown bias, for printing and saving.
Nothing internal consumes the report frame.  So every derivative here is taken
around the **model** frame (`fd_jacobian`, `cal_uncertainties_multi`,
`fwd_of`'s default).  Evaluating at the report frame — which is what
`report_params` used to be used for — gives the fraction of a parameter point
the fit never fitted: measured 3.29 vs 3.33 on `pwa_gen_mb`, and the two
routes then disagree by 1 %.

`jac=False` sweeps finite differences over the flat slots — aliases move
together and the bounds Jacobian is included, so it cannot be wrong — and
`batch=` streams it like the analytic route (at `2·len(x)` evaluations, so the
loads multiply; that is the price of finite differences).

`Fitter.fd_jacobian(fn, …, names=None)` is the one FD entry point: `names=None`
is that flat-slot sweep, `names=[…]` steps those resolved values and chains to
`x` analytically (`2·len(names)` evaluations instead of `2·len(x)`).  The named
form is for an observable with a small **genuinely complete** set (a
Breit-Wigner mass/width — `get_bw_params` passes exactly its 2 names); it is
not a shortcut for a fraction, which depends on every parameter the model
consumes (more names than `x` on the `pwa_gen` fixtures), so `fractions` only
offers the `x` sweep.  A requested name that is not resolved warns rather than
being silently dropped.

The mask memo is cleared per evaluation — it used to go stale across
perturbations, and `_compute_total_and_grad` re-derived `ck` from the best-fit
params instead of the perturbed ones, silently returning zero errors.

`scripts/calc_fractions.py` / `calc_first_decay_fractions.py` expose this as
`--batch N`, and every fraction script takes its sample from
`Fitter.load_dataset(prefix, path=…, max_events=…)` — the config-aware loader,
whose angle layout comes from `kernel_config["angle_k"]`.  The static
`Fitter.load_npz` must not be used for a sample: its `n_angle_comp` defaults
to 3 (legacy B→4π), so a pure-PWA config loads into the wrong shape.

For a **curve or grid** (many outputs, few parameters) don't build one dual
per point by hand — take the FD Jacobian and map it:

```python
def intensity(phys, grid):          # a REAL observable
    return abs(lineshape(phys, grid))**2

y, J = f.fd_jacobian(intensity, (m_grid,), fit_result=result)   # no lambda
duals = rows(y, J)                  # one Fwd per grid point
band  = [d.sigma(cov) for d in duals]
```

`fd_jacobian(fn, args=(), kwargs=None, *, fit_result=…, x=…, rel=…)` calls
`fn(phys, *args, **kwargs)` with the extra arguments given **explicitly**
(not collected variadically), so the fitter's own keywords can never be
swallowed by the callable's `**kwargs`.  It uses central differences in
**`x` space** — step `j` moves flat slot `j` and re-resolves
(`build_params(x ± h·e_j)[1]`, the model frame — *not* the blinded
`report_params`), so `J[i, j] = ∂values[i]/∂x[j]` and
`rows(y, J).sigma(hinv)` is the correct error.  (An earlier version perturbed
the *resolved* dict and its docstring still claimed `x`; the two differ by the
bounds Jacobian — `f0(980)_mass` 1.0 vs 0.548 — so `sigma` was wrong on every
bounded parameter.)  It costs `2·n_var` evaluations and returns
`(values, J)` with `J[i, j] = ∂ value_i / ∂ x_j` — the row layout `rows()`
turns into duals.  That is the cheap form for a mass scan: one vectorised
evaluation of the grid plus `2·n_var` tangent sweeps, not `n_points` scalar
dual chains.
The observable must be **real** (`Re`, `Im`, `|y|²`, a ratio …): `Fwd` is
real-valued, so a complex return raises rather than being silently truncated
(`abs(lineshape(...))**2` above, not the complex `lineshape` itself).

Scope: linear accumulation per sample (what the kernel already returns as
value + `grads`) plus closed-form post-processing on the accumulated
scalars.  It is **not** a general autodiff — no per-event arrays, no graph,
no model re-evaluation; a quantity that needs the model at changed
parameters stays a kernel call (analytic) or finite differences.  Cost is
O(n_params) per operation and in practice numpy-overhead bound: ~1–2.6 µs
per op at `n_var` 50 *and* 1000 (`tests/benchmark_fwd.py`), i.e. ~20 µs for
a mask set — against a kernel `compute` of 0.5–30 ms.

### IntegratedBackend (Gram matrix norm)

Pre-computes 56×56 Gram matrices from phsp.  Norm is O(n²) per iteration instead of O(N_phsp × 448).
Base backend handles data NLL.  Default base is `cuda_v3_cache`.
**TD model only** (`n_blocks=8` + scalars): a pure-PWA config uses the
sibling `integrated_pwa` backend.  Backends register a single name per
decorator scoped by amplitude-model NAME (a string) — e.g.
`@register_backend("integrated", model="flavour_tag_mix")`.  `Fitter` asks
`backends_for_model(model.name)` for the allowed set and
`create_backend(..., model=model.name)` resolves the per-model `"default"`;
picking `integrated` for a PWA config fails fast at backend selection with
the registered list.

### CUDA Build

Kernels auto-build on import — `.so` is rebuilt automatically when its build
key changes (source, **target arch**, tag, nvcc version, schema — SHA-256).
The CLI is **incremental by default** and takes name globs:

```bash
python -m tabpwa.cuda.build                        # build what is stale (all)
python -m tabpwa.cuda.build --list                 # what exists / what is stale
python -m tabpwa.cuda.build pwa_gen                # substring: only pwa_gen kernels
python -m tabpwa.cuda.build 'v5_*'                 # glob on the source name
python -m tabpwa.cuda.build pwa_gen --force        # rebuild that subset
python -m tabpwa.cuda.build --arch sm_86           # single arch
python -m tabpwa.cuda.build --arch sm_75,sm_86     # fat binary (multi-arch)
```

Selection is `fnmatch` on the source name, the stem and both library names; a
**bare word is a substring match** (`pwa_gen` → `kernels_v5_pwa_gen_bw.cu` …
four kernels), so no wildcards are needed for the common case.  An empty
selection means *everything*, and a pattern that matches nothing is a **hard
error** with the available list — a typo must not look like a successful
no-op.  `TABPWA_HASH_DIR` relocates the `.hash` bookkeeping for a read-only
source tree (the binaries still land next to the sources).

Auto-detection priority:
1. `nvidia-smi` → exact `-arch=sm_XY` for the installed GPU
2. `nvcc --version` → fat binary: `sm_70+sm_86` (CUDA < 13) or `sm_86` only (CUDA ≥ 13)

`--arch` is **validated** against `nvcc --list-gpu-arch`: CUDA 13 dropped
`compute_70`, and asking for it raises instead of failing every kernel
quietly.  The arch is part of the build key, so a native `sm_120` binary is
never mistaken for a fat `sm_70,sm_86` one (that used to ship locally-only
cubins into the AppImage).

**Toolkit selection** — `TABPWA_NVCC` > `CUDA_HOME`/`CUDA_PATH` > `PATH` >
well-known prefixes.  When `--arch` names an SM the preferred toolkit cannot
emit, the builder picks the next one that can, so `sm_70` on a CUDA-13 host
resolves to the CUDA 12 toolkit instead of failing::

    CUDA_HOME=/path/to/cuda-12.9 python -m tabpwa.cuda.build --arch sm_70,sm_86

That is how the AppImage (which bundles CUDA 12.8) gets `sm_70`; on a CUDA 13
toolkit without a cu12 fallback, `set_arch('sm_70')` raises with the supported
list.  The host compiler follows the same idea: `TABPWA_CC`/`CC`, then
`gcc-14`…`gcc-11` (CUDA rejects *future* gcc majors, so gcc-13 often beats the
default gcc 15), then the conda-forge `x86_64-conda-linux-gnu-g++`.

One `ensure()` costs **two** nvcc invocations: a shared compiler-compat probe
(cached per process, invalidated by `set_arch`) and the kernel itself.  It
used to be four — the probe was re-run per kernel *and* compiled a real,
unrelated kernel as its target.

## Kernel policy

**Never modify a working kernel.**  A kernel that works (tests green, in
use) is immutable: no edits, no signature changes, no "small" tweaks.
The only allowed actions are:

1. **New copy** — implement a new standalone kernel: add a new C source
   (`.cu`) by copying the existing kernel's source into it and
   **adapting** the content to the new variant under its own new symbol
   names; mirror the Python wrapper module (`_<name>.py`) the same way.
   The copy links beside the old kernel — every existing kernel file
   and its tests stay byte-identical.
2. **Fix a broken kernel** — a kernel whose tests fail / that does not
   work may be repaired in place.
3. **Replace a working kernel** — take a green case-1 copy and make it
   **replace** the old kernel: **relink** the copied symbols to the old
   kernel's names and swap the files in wholesale (same API), so the
   new implementation takes over.  The old kernel is superseded, never
   edited incrementally, and its previous copy stays available for
   rollback.

If a task seems to require touching a working kernel, the design is
wrong — state that instead of editing.

## Amplitude models

`amp_model` in the config (top-level, or legacy `data.amp_model`) selects an
`AmplitudeModel` (`tabpwa/amp_model.py`) via `build_amplitude_model(source)`,
where *source* is a file path, a config dict, or a `RawConfig`.  Registered
models: `pwa` (default) and `flavour_tag_mix` — the **time-dependent (TD)**
mixing model (aliases `flour_tag_mix`, `p4_directly`); add your own with
`@register_amplitude_model("name")`.

Object roles:
- **`RawConfig`** (`tabpwa/config_loader.py`) — the raw YAML input only:
  `dic`, `_config_path`, `backend_spec` (`load_config`).  No derived physics.
- **`Config(path)`** — a thin **legacy alias** → `build_amplitude_model(path)`
  (returns the model).  Prefer `build_amplitude_model(...)` / `RawConfig`.
- **`BaseModel`** (`tabpwa/base_model.py`) — the interpreted physics: decay
  tree, index/tables, the predefined base kernel config
  (`build_base_kernel_config()`; `build_all_index()` is a compatibility
  shim).  Constructible without a model: `BaseModel(config_or_dict)`.
- **`AmplitudeModel(BaseModel)`** — physics + model policy (`name`, scalar
  policy, `params_transform_cls`, the `build_kernel_config` seam, and
  `build_event_data`); it derives its own physics from the raw config.
- **`Fitter`** — composition root: builds the model (`self.model`) from a
  `RawConfig` (`self.config`) and reads policy/kc through the model.

`build_params_transform()` → `BuildKernelParams` (`tabpwa/kernel_params.py`):
`pwa` = ck/m0/g0 only, `flavour_tag_mix` adds the six time/mixing scalars.

Backends register a **single name per decorator**, scoped by amplitude-model
**name** (a string): `@register_backend("integrated_pwa", model="pwa")`
(`model=None` = universal, e.g. `shard`).  Stack decorators for aliases and
per-model defaults — e.g. `cuda_v4_pwa` is also registered as `"default"` for
`pwa`.  The model knows nothing about backends: `Fitter` asks
`backends_for_model(model.name)` for the allowed set and
`create_backend(..., model=model.name)` resolves `"default"` per model.

### JointFitter — simultaneous fit over several configs (`tabpwa/joint_fit.py`)

Simultaneous fits need ONE parameter space: a config pair that both
speak about `rho770_mass` must optimise one variable.  Instead of
merging constraint managers, the shared `ConstraintManager` is created
once and **injected** into every sub-`Fitter`
(`Fitter(cfg, cm=shared)`): each model registers its names into it
(`cm.register_names`), so a name present in two configs lands in the
SAME flat-`x` slot by construction — no slicing/aliasing/remap layer:

```python
from tabpwa.joint_fit import JointFitter
joint = JointFitter(["channelA.yml", "channelB.yml"],
                    backend="cuda_v5_pwa_s0c")   # or add_config later
x0 = joint.initial_values(seed=1)
nll, grad = joint.get_nll(x0)      # Σ_i NLL_i, index-aligned gradients
result = joint.fit(x0, maxiter=2000)
```

`get_nll` sums per-model NLL and gradients: each `Fitter.get_nll(x)`
returns a flat gradient over the *shared* registry (names another model
does not own are zero there via `cm.full_gradient`'s name lookup), so
the sum is a plain element-wise add.  `values_from_dict` /
`load_fixed_from_dict` / `values_dict` operate on the union; one blind
frame (`joint.set_blind`) covers the whole fit.

Same model, several datasets: keep the variant expansion **outside** the
joint fit — `tabpwa/config_broadcast.py` (`ConfigBroadcast`) turns a
template config into N configs from its own `broadcast:` list, then just
add each one (they share the model/cm, hence one x-space):

```python
from tabpwa.config_broadcast import ConfigBroadcast, scan

joint = JointFitter()
for label, cfg in ConfigBroadcast.items("multi.yml"):
    joint.add_config(cfg, backend="cuda_v5_pwa_s0c", label=label)
```

```yaml
data:
  dat_order: [pip, pim, eta]
  phsp_arr: shared_phsp.npz
broadcast:
  - {data: {data_arr: run1.npz}}
  - {data: {data_arr: run2.npz}, purity: 0.93, label: run2}
```

`ConfigBroadcast(template)` is a standalone helper: a container
(`len(cb)`, `for cfg in cb`, `cb[i]`) with `labels()`, `pairs()`,
`expand()`, plus the class-level consumer entry
(`ConfigBroadcast.items(source)` → `[(label, config), …]`) and
`has_broadcast(source)` (the form test).  One concept — a diff list in
the template; for a scan, pre-compute the list and put it there:
`cfg["broadcast"] = scan({"data.purity": [0.9, 0.95]})` (cartesian
product, auto `key=value` labels).  Relative `.npy`/`.npz` paths inside
`data:` resolve against the config's directory (variants are handed to
`Fitter` as dicts).  `JointFitter.add_config` always builds **one**
Fitter and takes an optional `label`.  A stray `broadcast:` key is
ignored by `Fitter` itself.  Diffs must not touch *forbidden* top-level
keys — default `("decay",)` (all variants share the decay topology);
configurable via `ConfigBroadcast(source, forbidden=(…))` (`()` disables;
`SameTopoJointFitter` keeps the default and keeps its deep decay-equality
safety net).

#### SameTopoJointFitter and `load_fit`

`SameTopoJointFitter(template, …)` (`joint_fit.py`) owns the broadcast
for the common case: the variants must share the **`decay` section only**
(particle content, `dat_order`, constraints and backends are free — the
check is `strict=True` by default).  The shared decay guarantees equal
observables, so it adds `dataset_labels()`, `nll_report(x)`,
`wave_report()` and `stack_plotter(result, groups=…)`.

`load_fit(source, …)` (`from tabpwa import load_fit`) is the auto-loader:
a template carrying `broadcast:` → `SameTopoJointFitter`, anything else →
a plain `Fitter` (constraints applied, datasets loaded):

```python
joint = load_fit("multi.yml", backend="cuda_v5_pwa_s0c")
res = joint.fit(); print(joint.nll_report(res.x))
joint.stack_plotter(res).plot_common(stacked=True)
```

#### Stacked plots (`tabpwa/plot_pw_stack.py`, `plot_pw_draw.py`)

`PWGroupPlotter.var_samples(...)` exposes the **pre-binning plot data**
(weighted samples for data/model/background/groups); `PWStackPlotter`
merges those arrays over datasets and **bins once**, so nothing forces
the datasets to share axes (`stacked=True` = Σ data, `√Σw²` errors, Σ
model/background; `stacked=False` = one colour per dataset with its own
data points and model curve).  All drawing lives in
`plot_pw_draw.py` (`bin_edges`, `make_var_axes`, `draw_var_panel`,
`draw_var_figure`, `draw_overlay_figure`, `save_figure`), so
`PWGroupPlotter.plot_var` and `PWStackPlotter.plot_var` are thin callers
and no plot code is tied to a fitter.

#### Polarised PWA (`amp_model: polarised_pwa`)

Projection-sum PWA with a fitted spin-density matrix.  The intensity
couples the top's helicity projections through
`rho = I + sum_l p_l * G_l` — the default basis is the P²−1 traceless
Hermitian generators over the top's helicity states
(`Tr(G_a G_b) = 2 delta_ab`; P = `len(spins)` of the top).  The
polarisation parameters are declared as model scalars, auto-named
`{top}_px`, `{top}_py`, `{top}_pz` (+ `{top}_p4`, ... beyond two
states) — nothing to configure beyond `amp_model: polarised_pwa`;
fixing/freeing them is the constraint system's job.  Zero parameters =>
`rho = I` => exactly the plain `pwa` intensity (validation anchor).
Positivity is not enforced (bounds via `var_range` if needed).

* `tabpwa/spin_density.py` — the generator basis + `rho_matrix`.
* `PolarisedKernelParams` (`kernel_params.py`) — the spin params travel
  through the params dict as `scalar` (the legacy scalar mechanism —
  the norm chain and constraint combination treat them like the
  existing scalars; no `Fitter` changes needed).
* Backend: `cuda_polarised` is the **default** (with `numpy_polarised`
  explicit-only, never a silent fallback) — a standalone kernel
  (`kernels_polarised.cu` + `_polarised.py`, a case-1 copy of the v4
  ampcache machinery, symbols relinked `..._polarised`; existing
  kernels untouched) whose reduce computes the ρ-contraction directly:
  `P_e = Re Σ_{p,p'} ρ[p,p']·A_p·conj(A_p')`, `dQ_dA =
  dQ_dP·(ρ·conj(A))`, and the generator basis (uploaded at context
  creation) yields finished `grads["scalar"]` in-kernel (no raw dQ/dρ
  in the API).  ρ is built host-side from `scalar` per call; no Gram
  path (the norm runs via the full-forward phsp pass).
  `numpy_polarised` — the CPU reference: `NumpyPolarisedPWA`
  (`numpy_pwa.py`; `NumpyPWA` itself is rho-free), same
  contraction/finished-scalar-grads contract.  Why a standalone kernel:
  the shared-ck projection-sum interface applies ONE `ck` vector to
  every projection, while the ρ-contraction's per-projection mixing
  weights differ per projection and change every iteration —
  inexpressible without kernel-side changes (kernel policy case 1;
  existing kernels stay untouched).
* Identifiability note: the params are measurable only where the
  helicity projections differ per event; fixtures whose projections are
  identical (e.g. identical angle columns) leave the spin params
  unidentifiable (the FD test documents this).

Same-name collisions auto-deal deterministically (latest registration
wins, `Warning` lines in `joint.log`): fixed values merge (later value
replaces), a name fixed by ANY config stays fixed (`free_var` in a
config explicitly unfreezes), **bounds are intersected**, scale and
mass/width transforms are replaced per name (never stacked), alias
groups (`var_equal`) union-merge, defaults merge.  The per-name
transform replace lives in `ConstraintManager.set_mass_width_transforms`
/`set_scale`: identity is the transform's **output names** (two
transforms producing the same parameter are alternatives; a transform
consuming another's output is a legitimate chain), and only transforms
registered by *earlier* calls are eligible (a model may legitimately
list several transforms producing the same family within one call — TD
models), so single-config behaviour is unchanged.
`setup_mass_width_transforms` therefore always calls with `reset=False`;
a shared cm must never be reset by a joining config.

#### Basis PWA (`amp_model: pwa_gen`, alias `pwa_basis`)

Projection-sum PWA on the particle `BasisModel` basis: for a chain whose
resonance is a `BasisModel` the amplitude is
`A_w = ck_w·fa·fl·N_{i(w)}(m) / Σ_j g_j·G_j(m)`, expressed in the existing
kernel vocabulary as an **opaque basis sum** (`BW = Σ_j coeff_j·F_j`) plus a
tabulated **numerator** factor.  Everything is additive — a config where the
resonances use the *default* `BasisModel` basis (`model: basis`) produces
byte-identical `pwa` kernel config / NLL / gradients.

Kernel-config keys (only emitted when a model actually needs them):

* `n_table` (complex `(n_rows, n_bins)`, row 0 ≡ 1), `n_min` / `n_delta`, and
  `n_index` (int32 `[n_fact]`, one entry per **(wave, factor)**
  `f = w·n_res + r`, `0` = constant row) — the per-event numerator factor
  `Π_r N_{n_index[w·n_res+r]}(m_e)`, applied only when some `n_index != 0`.
  Each factor reads **its own BW's mass column**, `mass_index[bw_order[f]]`
  — the same one its denominator propagator uses, so `B → R1 x, R1 → R2 y`
  gives `N_R1(m_R1)·N_R2(m_R2)`.  There is no separate `n_mass_index`: a
  second copy of that mapping could only drift out of sync with it.
* `den_row` (int32 `(n_unique_bw, n_basis)`: `0` = F 1, `1` = F −m²,
  `2+r` = `den_table[r]`, `-1` = 0), `den_table` (complex `(n_extra, n_bins)`,
  the finished extra rows `m⁴`, `−iρ`, `+i·s·ρ`, …), `den_min`/`den_delta`.
  Present when a resonance has a basis transform or extra basis rows; the
  backends then take an explicit `params["coeff"]` `(n_fact, n_basis)` and
  return `grads["coeff"]` instead of deriving `coeff` from `m0`/`g0`.

ck rule: one chain expands into one wave per free numerator row, with
`ck_w = g_ls[chain]·n_i` (the heads go into the `ck_map` tuple, so
`CKProduct` builds the outer product).  `ck_map`/`get_ck_map()` carry the
expanded combinations; `FlatKernelParams` builds the flat `coeff` and chains
`grads["coeff"]` back to the basis *weight* names, and
`PWAgenWeightsTransform` (`PWAGen.extra_transforms()`, registered by
`Fitter.setup_mass_width_transforms` as a mass/width-style transform) maps the
weights to/from the physical parameters (FD VJP, the `make_basis_transform`
contract).

Backends: `numpy_flat` (`default` for `pwa_gen`) — the flat config's
reference implementation.  The GPU siblings `cuda_flat_pwa_gen` and
`cuda_flat_cache_pwa_gen` (cached `F`/basis/`fa`) read the same config; the
former `cpu_pwa_gen` / `numpy_pwa_gen` / `cuda_v5_pwa_gen_{s0e,flat,bw,bwt}`
backends read the old `den_row`/`n_index`/`bw_order`/`matrix_angle` config and
were **removed** together with their kernels: a `pwa_gen` config no longer
carries any of those keys, so a kernel that reads them has nothing to read.
The plain-PWA backends (`numpy_pwa`, `cpu_v5_pwa`, `cpu_pwa_s0c`,
`cuda_v4_pwa`) stay registered for `model="pwa"` alone.

#### `resolution_size` — the grouped log-sum objective

Every pwa_gen backend takes `resolution_size` (constructor kwarg, also via
`backend={"name": …, "resolution_size": N}`, default 1).  Groups are
consecutive events aligned to the **global** event index (`g = e // N`),
independent of the tile or `batch_size`:

```
W_g = Σ_{e∈g} w_e                 S_g = Σ_{e∈g} w_e·(P_e/norm + bkg_e)
Q      = -Σ_g W_g·log(S_g/W_g)    (W_g = 0 ⇒ the group is inert)
dQ/dP_e = -(W_g/S_g)·(w_e/norm)
dnorm  = Σ_g W_g·Pb_g/(S_g·norm²)      Pb_g = Σ_{e∈g} w_e·P_e
```

`resolution_size = 1` is exactly the per-event NLL `-Σ w·log(P/norm + bkg)`.
At `> 1` with **group-normalised** weights (Σw = 1 per group) the log argument
becomes the group's mean pdf — that is a *different* objective from the
per-event sum (a Jensen shift), not an approximation of it; normalising this
way is what keeps `Σw` equal to the group count so the fitter's purity
constant `-log(p)·Σw` stays exact.  `P` returned to the caller is always the
per-event intensity — grouping changes the objective, never `P`.

Real use is **10–100**, i.e. larger than a CUDA tile, so a group straddles
tiles: the kernels accumulate per-group partials in global `gS`/`gPb`/`gW`,
close them in a small per-group kernel that also publishes
`scale_g = -W_g/S_g`, and then the gradient pass reads `dqp[ge] = scale·w/norm`
instead of deriving `dQ/dP` locally.  That costs one extra forward pass only
when `rsize > 1`; `rsize == 1` and `norm=None` keep the single-pass path and
launch no extra kernel.  Cost measured on the 100-wave/2-projection shape at
65k events: 8.6 ms → 11.9 ms, flat across `rsize` 10…100 (the extra pass is
per-tile, not per-group).  A short final group (`ne % rsize != 0`) is handled
everywhere; the CUDA group buffers are sized from the real `ne` at
`load_data`.  Fixture: `tests/test_pwa_gen_resolution_size.py`.

Basis mixing: a config may put a `BasisModel` resonance and an ordinary
resonance side by side — the generic path then tabulates *every* resonance's
rows past the two built-ins, and a plain resonance contributes
`[m0², −1, m0·g_l]` with the same numbers as the non-generic path.

**Identical-particle / CP blocks (`n_blocks > 1`) ARE supported, and it is
purely a build-time index rebasing** — no backend knows blocks exist.
`build_from_groups` emits the wave axis as `n_blocks · n_base` and rebases
every per-use index onto its block's column band; the backends already read
`fl_q_index` / `num_m_index` / `angle_index` as plain column numbers and
iterate `k` over `fl_table.shape[0]`, so `n_wave == len(ck)` holds by
construction.  This is why no kernel change was needed: the earlier version
of this paragraph claimed a backend change was required, and claimed the
already-removed `bw_order` handled it — both wrong.

The block bands come from `build_tree_event_data`, which concatenates one band
per block: `mass` → `n_blocks·n_topo·n_res`, `q` → `n_blocks·n_topo·n_decay`,
`angle` → `n_blocks·n_topo`.  So the strides are exactly those three
quantities, and block `b` owns `[b·stride, (b+1)·stride)`.

The block ORDER must match `get_ck_map()` (cp-outer, perm-inner, combo
innermost), which is what `DecayTree.block_orders()` already yields (all
permutations, then their CP copies), so block `b` is ck block `b` — the two
orderings were checked rather than assumed.

The angle formula is block-**invariant** (`_wave_angle_terms` depends only on
`chain, ls, projection`); a CP block is the same chain evaluated on conjugated
momenta, and the CP/parity transform is applied to the DATA.  The angle basis
KEY, however, uses the block-rebased column, so a CP block gets its own rows —
its columns are a different function of the event.

Fixtures: `tests/config_pwa_gen_ident.yml` (identical only, 2 blocks) and
`tests/config_pwa_gen_cp.yml` (identical AND CP on the same pair, 4 blocks —
the hardest case).  Gates in `tests/test_pwa_gen_blocks.py`, including two
that do not trust the backends: block 0 must be byte-identical to the
`n_blocks == 1` config once un-rebased, and the objective must equal the
quadratic form its own unit responses imply (with a non-zero off-diagonal
block, i.e. the blocks interfere as amplitudes rather than summing as
independent intensities).

In the generic path the fitted `coeff` block replaces `m0`/`g0`, so the
`{name}_mass` / `{name}_width` parameters are **not** free there unless they
are also basis-transform inputs (otherwise they would be identically-zero
gradient directions).

Fixtures: `tests/config_pwa_gen.yml` (default basis, gate A),
`tests/config_pwa_gen_km.yml` (`kmatrix2`, gate B/C), `tests/test_pwa_gen.py`
(includes mixed-basis, two-K-matrix, CP-block and numpy/cpu parity cases),
`tests/test_pwa_gen_s0e.py`, `tests/benchmark_pwa_gen.py`,
`tests/benchmark_pwa_gen_s0e.py`.

### Barrier factors & build-time defaults

The Blatt-Weisskopf form factor `F_L(q)` of every decay vertex is baked into
the kernel config's `fl_table` (all backends / `onnx_cpu` / `onnx_cuda` only
interpolate it), so barrier **forms** are pluggable in pure Python:

- `tabpwa.bw_form_factor.BarrierFactor` — base class (`L`, `q0_ref`, `d`);
  override `factor(q)` and `get_params()` (extra parameters go in
  `get_params`, which feeds `get_id()` and the model's `fl_forms`).
  `register_barrier(name)` registers a subclass; built-ins: `"bw"` (default,
  Blatt-Weisskopf, unchanged) and `"exp"` (`exp(-(q·d)²/2)`).
- Selected **per decay** with the existing decay-entry kwargs:
  `barrier: <name>` or `barrier: {type: exp, d: 1.5, alpha: 2.0}` (extra keys
  go to the class constructor); a sibling `d:` sets the radius.  Defaults are
  `bw` / `3.0`.

**Build-time defaults come from the global build context, not the config.**
`tabpwa.build_defaults` is the module; its `scope(**kw)` is a
`contextvars`-backed `with` scope:
`with build_defaults.scope(n_interp=4000, d=1.5): …`
(`from tabpwa.build_defaults import scope`).  Inside the scope the
given overrides apply; on exit everything is back to the global defaults
(`n_interp=2000`, `d=3.0`, `barrier="bw"`, in `build_defaults.DEFAULTS`).
Recognised keys: `n_interp` (sampling of `fl_table` / `gamma_table`), `d`
(Blatt-Weisskopf radius), `barrier` (default type), `complex_tail`
(`(magnitude, phase)` suffixes naming a complex ck parameter's two slots —
default `("r", "i")`, alternatively `("rho", "phi")`; captured when the model/
transform is built, so set it before constructing the `Fitter`).
Build code reads it when a particle/decay omits the value (`BaseModel._barrier`,
`build_fl_table`, `build_gamma_table`, `particle_model.build_particle`), so
config files stay physics-only.  A config may also carry a `defaults:` mapping
(`defaults: {d: 1.5, n_interp: 4000}`); `BaseModel` loads it as a **temporary**
context for its own build — it does not leak into the global context.

`BaseModel.fl_forms` is the ordered list of unique `BarrierFactor` objects
(model metadata for the `fl_table` rows); the kernel config keeps only the
arrays the kernels index (see gotcha 13).

#### `fl_*` arrays: kernel inputs vs model metadata

`fl_table` is `(n_forms, n_interp)` and the kernels pick a row per
(wave, decay-vertex) with **two independent indices**:

```c
int fl_idx = fl_order[w * n_decay + d];                          // (wave, vertex) -> pair
double q   = momentum[event * n_momentum + fl_q_index[fl_idx]];   // INPUT: which q
fl *= interp(fl_table, fl_type[fl_idx], q, …);                    // FUNCTION: which row
```

- `fl_type` — the `fl_table` **row** (which form / function),
- `fl_q_index` — the momentum **slot** (which input `q`),
- `fl_order` — per (wave, vertex) → the dedup pair `(row, q-slot)`,
- `fl_min` / `fl_delta` — the table grid.

Those five are genuine kernel inputs and stay in the kernel config.  The
*meaning* of each row (`L`, radius, type, params) is not — it lives on the
model as `BaseModel.fl_forms` (a list of `BarrierFactor`); `fl_type` indexes
it.

### Dispersive line-shape base (design — not yet implemented)

General base for lineshape models whose propagator **real part** comes
from the dispersion relation of an absorptive part (the GS construction,
generalised).  Intended for new dispersive models; `GS_rho` keeps its
exact closed form and serves as the validation oracle.

* **Subclass hook**: `im_pi_over_gamma0(s)` — the absorptive self-energy
  `Im Π(s)` at `Γ0 = 1`, including the kinematic loop factor (for the ρ:
  `M_ref²·q³·B'²/(√s·q0³)`); threshold `s_th = (dm2 + dm3)²` from the
  prepared daughters.
* **Base, once per instance**: tabulates `Im Π` on an internal s-grid and
  computes `Re Π` at each node as the **exact per-segment PV integral of
  the linear interpolant**:

      Re Π(u) − Re Π(M_ref²) = ((u − M_ref²)/π) · Σ_seg ∫ (a + b·s')·[1/(u−s') − 1/(M_ref²−s')] ds'

  (the subtraction factor `(M_ref² − u)` pulled out via `1/a − 1/b`; each
  segment integral is closed-form — logs; PV regularised by
  subtract-and-add at `s' = u`; the difference kernel makes the tail
  convergent even for `Γ(m) ~ √s`).
* **`gamma(m)`** returns the packed complex unit row
  `[Im Π/Γ0](m) + i·[Re Π/(Γ0·M_ref)](m)` — Re and Im carry the **same**
  factor (both pieces are dimensionless shapes; the `M_ref` in the
  Re-piece's denominator belongs to the unit's `(q0, m0_ref)` scale
  definition).  Im is exact from the hook, Re interpolated from the
  precomputed table (the kernel re-interpolates onto its own grid as
  usual).
* **Validation oracle**: with the ππ p-wave `Im Π` (the GS integrand),
  the numeric `Re Π` must reproduce the closed form
  `Γ0·M_ref²/k0³·[k²·Δh + (M_ref²−u)·k0²·h′]` at grid accuracy.

## Three-Layer Architecture

```
Fitter (orchestrator) — owns constraints + numpy data (_data_np, _phsp_np)
  │  set_data / set_phsp → backend.load_data() → handle
  │  get_nll(x) → build_params → _compute_norm_batched → get_nll_raw → _flat_gradient
  │
  ├── Backend (standard interface) — wraps kernel, handles batching
  │     load_data(data_np) → Handle
  │     compute(params, handle, norm=None, return_p=?) → (Q, grads, P)
  │     free()
  │     │
  │     ├── _CUDABackend (base for CUDA v2/v3/f32/mixed)
  │     ├── NumpyBackend     — load_data returns dict itself (zero-copy)
  │     ├── IntegratedBackend — Gram matrix norm + delegates data NLL to base
  │     └── ONNXBackend       — ONNX Runtime with batching
  │
  └── Kernel (raw compute) — no constraints, no norm, just math
        __init__(config) → set up indices/tables
        load_data(data) → upload to device
        compute(params, handle, norm=None) → (Q, grads, P)
        └── NumpyKernel / CUDAKernelV3/V2 / ONNXKernel
```

**Layering / decay tree**: `decay:` / `particle:` are interpreted by the
standalone value object `tabpwa.decay_tree.DecayTree` — structure, chains,
stable topology index, and the `data.identical_particles` / `cp_particles`
symmetry declarations (`n_perm` / `n_cp` / `n_blocks` + `block_orders()`).
`BaseModel` holds the interpreted physics (DecayTree + index/tables + base
kernel config).  Layer direction is `config_loader -> base_model ->
decay_tree`, plus `amp_model -> base_model`; `base_model` is a leaf w.r.t.
`config_loader`/`amp_model`/`backends`.  Legacy attribute names
(`full_decay`, `decay_struct`, `topo_index`, `n_topo`, `n_decay`, `n_res`,
`n_angles`, `finals`) are aliases of `model.decay_tree`; the
`Particle`/`Decay`/`DecayChain`/`DecayGroup` classes +
`symmetry_factors`/`block_column_orders`/`row_block_factors` +
`_projection_duplicate` are re-exported from `config_loader`.

**AmplitudeModel**: subclasses `BaseModel` — the model *is* the interpreted
physical model plus fitting policy.  It derives its own physics from the raw
config (`AmplitudeModel(raw_config_or_dict)`), so it shares no mutable state
with a Config.  `build_kernel_config()` is the override seam: `PWA` adds the
pure-PWA meta keys on top of the base config; `FlavourTagMix` (the
**time-dependent / TD** model) keeps the base.  `Fitter` keeps the model as `fitter.model` and reads its
policy/kernel config through it.

### Event data (model-driven)

`Fitter.load_momenta_conf` only reads the file, reorders columns by
`data.dat_order`, and delegates to `model.build_event_data(momenta, kc)`:

- **PWA** (default): `build_tree_event_data(cfg, kc, chains_by_topo, momenta)`
  — a generic tree fill with **identical-particle / CP block expansion**
  (`n_blocks = n_perm · n_cp` blocks; each block reorders the final columns,
  and a CP block additionally reverses the 3-momentum `p → −p` **in the CM
  frame**).  `n_blocks == 1` is the plain single-block fill.
- **`flavour_tag_mix` (TD)**: `momenta_to_data` — the time-dependent 24-row
  layout that also emits the `frac`/`time` mixing inputs.  The override is
  structure-routed: the legacy 24-row fill runs only for the 4π structure
  (four pion finals in two identical pairs); any other declared structure
  (e.g. K*K̄* — no identical particles, CP blocks only) takes the generic
  tree walk with `frac`/`time` (defaults 0.5/0.0, overridable) appended —
  the fill and the kernel config share the tree's row enumeration by
  construction (`n_angle_total = max(angle_index)+1`), so the TD kernel's
  config-driven contract (`reshape` + slice to `n_mass` /
  `n_angle_total·n_angle_comp`) is met for any declared topology.

Block enumeration lives on the tree: `DecayTree.block_orders()` →
`[(column order, is_cp), ...]` (order is arbitrary — all blocks share one
`ck` and are summed).  Geometry-only consumers construct `DecayTree` directly.

### Data flow

```
Fitter._data_np / _phsp_np  (numpy arrays, CPU)
  → backend.load_data(data)
      → NumPy: returns dict (zero copy)
      → CUDA:  uploads to GPU, returns DataHandle
  → backend.compute(params, handle, ...)
      → kernel reads compute-relevant data from handle
      → returns (Q, grads, P)

Fitter._compute_norm_batched(params):
  backend.compute(params, phsp_handle, norm=None, return_p=False)
    → Integrated: uses Gram matrices (cached, O(n²))
    → CUDA/NumPy: full forward pass (batched internally)

Fitter.get_nll_raw(params):
  norm, norm_grads = _compute_norm_batched(params)
  nll, grads, P = backend.compute(params, data_handle, norm=norm)
  total_grad = kernel_grad + dNLL_dnorm * norm_grad
```

- **Fitter** is the only layer that stores numpy arrays.
- **Backend** adds `return_p` semantics, optional pre‑computation (Gram).
- **Kernel** is pure compute — knows nothing about constraints, normalization, or the fitter.
- All batching is handled inside the backend (CUDA C code or numpy backend split loop).
- `_compute_norm_batched` always calls `backend.compute(norm=None, return_p=False)`:
  - IntegratedBackend → Gram path (O(n²) matmul from cached matrices)
  - Other backends → full forward pass (batched internally, `return_p=False` just skips P).

### Likelihood layer (design — not implemented)

The `norm → Q` pipeline above is currently hard-coded in
`Fitter.get_nll_raw`.  `docs/likelihood_layer.md` specifies a standalone
`tabpwa/likelihood.py` as a **pipeline of `Step` objects** threaded by a plain
loop (`run(steps, params, datasets)`).  Every step type shares **one general
constructor** `Step(spec, context)` and is built through a common
registry: `@register_step("norm")` / `STEP_TYPES` / `build_step(spec,
context)`.  A spec is plain data (`{"type": "data", "needs": "data",
"name": "nll_data"}`), `spec` carries the type configuration and `context` the
build-time dependencies (backend, `purity`, `sumw`, `purity_const`).  Each
step declares `needs` (the dataset **name** it reads) and `outputs` (value
names it publishes) and **stores no data**: the caller keeps a `datasets`
mapping (`name -> handle`, handles from `backend.load_data()`) and the driver
passes `datasets[step.needs]` at call time.  The mapping holds **both forms**
of each dataset as separate entries — `"data"`/`"phsp"` (backend handles, for
`compute`) and `"data_np"`/`"phsp_np"` (the numpy arrays) — so a step declares
which one it reads.  The observed count comes from the numpy entry:
`SumWStep(needs="data_np")` publishes `"@<key>:sumw"` / `"@<key>:n"`
(constants, empty gradient), so extended terms such as the Poisson term read
`S` without any hand-passed constant, backend change, or driver pre-seeding,
and the pipeline stays static.
Two **shared updateable dicts** carry the state: `values` (`name -> scalar`) and `grads` (`name -> ∂value/∂params`), so
one value (e.g. the normalisation) is reused by every later step.
`cur_grads[name] = {source: derivative}` with the reserved source `"params"`
holding the direct `∂v/∂params`; every other source is a scalar cotangent.
**The driver applies the chain**
`grads[name] = ∂name/∂params + Σ_src (∂name/∂src)·grads[src]`, so a step
cannot drop the factor.  Value names are unique per producer and **`"nll"` is
reserved for the objective**, produced only by the final `"sum"` step (terms
publish `nll_data`, `nll_poisson`, `nll_penalty_<tag>`, …), so no term reads
or aliases the objective.  The default spec list (`norm`, `data`, `sum`) is a
bit-exact port of `Fitter.get_nll_raw`.  Named objectives are a registry of
**builders** (`register_nll(name)` / `build_pipeline(fitter)`) returning spec
lists; resolution `Fitter(nll=…)` → `config["nll"]` → `"standard"`, unknown
names fail with the registered list, the chosen name is saved in the fit
metadata.  Different objectives (shared-normalisation multi-dataset,
extended/Poisson yields, external measurement constraints) are **different
spec lists**; there is no objective/NLL class hierarchy and no backend
capability metadata.  `ConstraintManager` transforms (fixed/same/scale/
bounds/blind) are *not* NLL terms and are untouched; priors remain
resolved-frame additions merged before `_kernel_builder.backward`.

### `compute(params, data_handle, norm=None, return_p=True)`

`return_p=False` → skips per-event P (fast norm).  `return_p=True` (default) → returns P array.
Gradients are dict: `{"ck": ..., "m0": ..., "g0": ..., "scalar": ...}`.

### CK array layout

448 total = 8 blocks × 56 base waves.  Blocks 0–3: B0 (g_ls), Blocks 4–7: B0bar (g_lsbar).
Each block has 4 identical-particle permutations of the 56 base waves.

## Parameter sizing for kernel params

When building random test params, sizes come from kernel_config indices:

```python
# CORRECT:
n_m0 = int(np.max(kc["m0_index"])) + 1   # = 20 (unique BW masses)
n_g0 = int(np.max(kc["g0_index"])) + 1   # = 23 (unique gamma params)
n_ck = len(config.get_ck_map())           # = 448

# WRONG (too large):
n_m0 = len(kc["m0_index"])               # = 216
n_g0 = len(kc["g0_index"])               # = 288
```

`np.take(m0, m0_index)` scatters unique → full internally.

## Key Gotchas

1. **g0/m0 sizing**: Params use unique size (`max(index)+1`), not the expanded `len(index)`. The kernel scatters via `np.take()`. ONNX backend reduces automatically; benchmarks must pass correct sizes.

2. **`LinearTransform`** (formerly `ScaleTransform`): `d[name] = factor * d[name] + bias`.  `ScaleTransform` is an alias.  `set_scale()` accepts `{name: factor}` or `{name: (factor, bias)}`.

3. **`get_defaults()`** on particle models returns only parameters that need defaults (the ones the fitter optimises directly).  `OneModel.get_defaults()` returns `{}` (mass/width fixed by transform).  CK matrix returns `{name_mass}` only — width and ck are transform inputs, their defaults come from the model config internally.

4. **`save_params()`** accepts either a `fit_result` object or a **flat `x` vector** directly:
   ```python
   fitter.save_params(x_flat, "checkpoint.json")  # no result object needed
   ```
   Missing hess_inv → errors dict is empty.

5. **Interrupt checkpoint**: Ctrl+C during `fit()` saves the last BFGS iterate to `checkpoint.json` via `fitter._last_xk` (stored by `get_nll()`). Resume with `--init checkpoint.json`.

6. **`get_decay_ck_indices(decay_pairs, wave_idx=)`: `wave_idx` picks a specific LS combination (0=S-wave, 1=D-wave, etc.) within each matching chain.  `_get()` helper in scripts uses OR semantics (each pair resolved independently, then unioned).

7. **`fmt_meas(v, e, pct=False)`** in `tabpwa.utils`: error-threshold decimal formatting.  `e=0 → no ±.  Thresholds: <0.355 → 2dp, <0.950 → 1dp, else 0dp (after 3‑sig‑fig rounding).`

8. **Constraint file format**: JSON with `fixed`, `same`, `scale`, `bounds`.  `scale` values can be `float` or `[factor, bias]`.

9. **Fit restart**: `--init results.json` or `--init checkpoint.json` reconstructs x from saved values. `--save-init [path]` writes the starting vector in the same format (default `<config>_init.json` next to the config; loop mode writes `<config>_init{run}.json`), so a random start can be replayed exactly.

10. **Wirtinger gradients**: ∂Q/∂ck[i] (complex), `backprop_grad` distributes to slot names.  For real params: `dQ/dx = 2·Re(∂Q/∂z · ∂z/∂x)`.

11. **`g0` is not the direct nominal width**: the fitted `g0` parameter is the width at the reference defined by the resonance mass FIXED in the config's particle table (the mass used to build the running-width `gamma_table`/`fl_table`).  It is NOT tied to the `m0` fit parameter — when `m0` is fitted away from that fixed reference mass, the physical Breit–Wigner width changes through the running-width term `g0·γ(m)`.  So treat fitted `g0` values as “width at the config reference mass”, not the width at the fitted mass.

12. **`Config` vs `RawConfig`**: `Config(path)` is a legacy alias for `build_amplitude_model(path)` (it returns the model).  The raw input object is `RawConfig` (`dic` / `_config_path` / `backend_spec`); `Fitter.config` is a `RawConfig` and `Fitter.model` is the `AmplitudeModel`.  Physics/kc members (`full_decay`, `build_kernel_config()`, `m0_phys_name`, …) live on `BaseModel`/`AmplitudeModel`, not on `RawConfig`.

13. **Barrier metadata lives on `BaseModel`, not the kernel config**: the kernel config carries only what the kernels index — `fl_table` / `fl_type` / `fl_q_index` / `fl_order` / `fl_min` / `fl_delta`.  The barrier *forms* (`L` / radius / type / params) are `BaseModel.fl_forms` (a list of `BarrierFactor`); `fl_type` indexes it.  Consumers (`lineshape_common.b_barrier_factor`, `scripts/calc_3pi_lineshape.py`) read `f.model.fl_forms`.  The two `fl` indices are independent: `fl_type[fl_idx]` selects the function (table row), `fl_q_index[fl_idx]` selects the input momentum slot, and `fl_order` maps each (wave, vertex) to the dedup pair.

14. **Build-time defaults via `build_defaults`**: `n_interp`, `d`, `barrier` are read from the `tabpwa.build_defaults` module (a `contextvars` scope), not the config — so configs stay physics-only.  A config's top-level `defaults:` is loaded as a *temporary* context while that model is built (`BaseModel._default_context`).

15. **`load_data` must return a per-load handle, never the context**: `Fitter.set_phsp` loads phase space and `set_data` loads data, and **both handles stay alive for the whole fit** (`_compute_norm_batched` integrates over the phase-space one while `get_nll_raw` evaluates the data one).  A kernel whose `cuda_load_data_*` returns its own context (or whose wrapper keeps one mutable `self._handle` and frees it at the top of `load_data`) makes the two the same object, so the second load silently overwrites the first and the norm is integrated over the **data** — while every kernel-level test still passes, because those call `backend.compute` with an explicit handle.  The established kernels `calloc` a `DataHandle2` per load; `tests/test_fitter_norm_handles.py` pins the contract through the `Fitter` (`norm` through the fitter must equal a fresh instance's phase-space integral, and the two handle identities must differ).

## Testing

Test data is **self-contained under `tests/`**: the test configs
(`config_angle.yml`, `config_amp.yml`, `config_pwa.yml`, …) live in
`tests/`, and `tests/conftest.py` (`pytest_configure`) generates
`tests/lineshapes/` — deterministic synthetic gamma/partial-width tables
in the exact formats the models load (`width_linear_npy` (N,3)
`[m, Im Π, Re Π]`; `ck_matrix_v2` gamma grid + `(N, n_ck, n_ck)` complex
partial matrix + channel-order JSON) — whenever missing.  Tests never
reference files outside `tests/`; generated data is never committed.
Example configs remain at the repo root as copies of the `tests/`
versions (examples only — the suite never reads them).

```bash
pytest tests/ -v                        # 1032 tests
pytest tests/test_fitter_constraints.py  # constraint pipeline
pytest tests/test_new_apis.py            # LinearTransform, fmt_meas, get_defaults, …
tests/validate_gradients.py              # 3-point gradient validation (all 6 backends)
```

Benchmarks:
```bash
python tests/benchmark_backends.py       # cross-backend (8 backends)
python tests/benchmark_cuda_v2_vs_v3.py  # v2 linear vs v3 Catmull-Rom
python tests/benchmark_integrated.py     # Integrated vs CUDA via Fitter.get_nll()
python tests/benchmark_pwa_kernels.py    # pure-PWA family on a synthetic kernel config:
                                         # raw compute (P / NLL+grad) + full NLL
                                         # (integrated_pwa base=cuda_v4_pwa_cache)
                                         # --n-comp N --n-proj 2 --preset large -> 10^5-10^6
```
