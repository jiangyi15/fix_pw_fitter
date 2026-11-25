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

## Versioning & changelog

The version has ONE source: `src/tabpwa/_version.py` (`__version__`).
`pyproject.toml` reads it (`[tool.setuptools.dynamic]`), and
`tabpwa.__version__` / `python -m tabpwa --version` /
`python run_fit.py --version` report it.  Do not state the current number
anywhere else, docs included — the changelog's dated `## [X.Y.Z]` headings are
the historical record, not a source.  `tests/test_version.py` fails if a surface
drifts or if the `[project]` table grows a second copy.

`CHANGELOG.md` (Keep a Changelog; SemVer, and at 0.x a minor bump may break the
API) is where a change is announced:

- a user-visible change — behaviour, API, CLI, a numerical result, a file
  format — adds a bullet under `## [Unreleased]` in the SAME commit;
- a release moves that block under `## [X.Y.Z] - YYYY-MM-DD`, bumps
  `_version.py` and tags `vX.Y.Z`, all in one commit;
- build it from a CLEAN tree (`rm -rf build src/*.egg-info`): setuptools reuses
  `build/lib`, so a package-data glob removed from `pyproject.toml` still ships
  files an earlier build copied there (measured: 42 stale `.hash` files in the
  wheel until `build/` was removed);
- internal refactors and test-only changes need no entry, but a doc that names
  a version does (it is a surface).

`tests/test_version.py` also requires a dated heading for the current version,
so a bump without a release note fails the suite.

Dependency declarations are audited by `tests/test_packaging.py`: every
third-party top-level import in `src/tabpwa` must be declared in
`pyproject.toml` (base or an extra), `onnx`/`onnxruntime` must stay in the
`onnx` extra, and `import tabpwa` must work with those extras absent.  A new
import therefore needs a declaration, not just a working environment.

## Documentation

`docs/README.md` indexes every document as a **design** (what should exist and
why — normative, may describe something unbuilt) or an **implementation** (what
the tree does: files, API, measurements, gates, limits).  Keep them as SEPARATE
files: a design change goes in the design first, and the implementation file
records what was built and any deviation.  If the two disagree, the design is
the intent and the implementation file is the bug — fix the code or the design,
not the implementation prose alone.

Live split: `docs/mpi_backend_design.md` / `docs/mpi_backend.md` (+
`docs/slurm.md` for deployment); `docs/mpi_service_design.md` /
`docs/mpi_service.md` for the subprocess-held MPI job (the mode that keeps every
MPI fact out of the package) — DESIGN and implementation, both live;
`docs/remote_protocol_design.md` is a draft standard with NO implementation.  `docs/likelihood_layer.md` is a design that
was never built.  `AGENTS.md` stays the architecture reference; `docs/` is for
the topics that would make it unwieldy.

## Backend Selection

| Backend | Flag | Note |
|---------|------|------|
| **Integrated** | `{"name": "integrated", "base": "cuda_v3_cache"}` | Gram-matrix norm + base backend for the data NLL. TD model only (`n_blocks=8`); pure PWA uses `integrated_pwa`. Default base `cuda_v3_cache`. |
| **v3 amp-cache** | `cuda_v3_ampcache` | `cuda_v3_sparse` plus a per-handle cache of the angular amplitudes, so only the BW propagator is recomputed per iteration. ~1.7× faster than sparse. |
| **v4 PWA** | `cuda_v4_pwa` | Projection-sum PWA, no time/mixing/scalars; all projections share one `ck`. Config key `n_proj` (default 1). |
| **v5 PWA s0cw** | `cuda_v5_pwa_s0cw` | `cuda_v5_pwa_s0c` with the group-weighted log NLL; zero-weight groups inert. Fixture `tests/test_pwa_v5_s0cw.py` (run alone). |
| **v4 PWA cache** | `cuda_v4_pwa_cache` | Full-amplitude cache for fixed m0/g0; bit-equal to `cuda_v4_pwa` there. **Strictly fixed m0/g0** — later changes raise ValueError; float masses/widths ⇒ use `cuda_v4_pwa`. |
| **v4 PWA 1bw family** | `cuda_v4_pwa_1bw`, `cuda_v4_pwa_1bws`, `cuda_v4_pwa_s0` | One BW per chain (`n_res == 1`, multi-BW chains raise); NLL bit-exact vs `cuda_v4_pwa`, gradients ≤1 ulp. `1bws` = same kernel with mass-sorted `ck_order` (A/B only); `s0` = direct pole parameters `s0 = m0²` / `mult = m0·g0`. Fixture `tests/config_pwa_bw.yml` (free m0/g0). |
| **pwa_gen (flat config)** | `numpy_flat` | `pwa_gen` reference backend (`NumpyFlatBackend`) over the flat config: plain loops, no cache; accepts the full and the `dedup`ed layout. Fixture `tests/test_pwa_gen.py`. |
| **pwa_gen (flat config), CUDA** | `cuda_flat_pwa_gen` | GPU port of the flat config that recomputes every interpolation per iteration (no caches) — the correctness reference. Reads the full, non-deduped config only; a `dedup`ed one is rejected. Test data must lie inside both interpolation grids. Fixtures `tests/_flat_compute.py` (the shared compute suite) and `tests/test_flat_dedup.py` (the config contract and the dedup-layout parity). |
| **pwa_gen (flat config), CUDA cached** | `cuda_flat_cache_pwa_gen` | `cuda_flat_pwa_gen` with a cached `F` and backend-internal dedup of the basis rows on `(coeff name, basis kind, mass column)`; the caller keeps the full-form `coeff`. Fixtures `tests/_flat_compute.py` (the shared compute suite) and `tests/test_flat_dedup.py` (the config contract: the layout pair, the reused-handle cache). |
| **pwa_gen (flat config), CUDA cached v2** | `cuda_flat_pwa_gen2` | `cuda_flat_cache_pwa_gen` with four exact algebraic reductions (u-merge, `n_res == 1` fork, gated `n_slot` `dcoeff` collapse, `fa` folded into the cache). Working memory bounded by `batch_size`, not the sample; `n_wave` ceiling 298. Fixtures `tests/_flat_compute.py` (the shared compute suite) and `tests/test_flat_dedup.py` (the config contract, the `slot_collapse` on/off A/B, the wide-basis fallback). |
| **pwa_gen (flat config), CUDA 1bw** | `cuda_flat_pwa_gen_1bw` | `cuda_flat_pwa_gen2` specialised to one BW per chain (`n_res == 1` only, else `ValueError`); sparse gamma rows, working memory bounded by `batch_size`. Fixtures `tests/_flat_compute.py` (the shared compute suite) and `tests/test_flat_dedup.py` (the config contract, the sparse gamma rows, the `slot_collapse` A/B). |
| **pwa_gen (flat config), CUDA s0e tile** | `cuda_flat_pwa_gen_s0e` | Flat-config s0e fused tile: one pass for the grouped log-sum at `resolution_size > 1`. `n_res == 1` only. Fixtures `tests/_flat_compute.py` (the shared compute suite), `tests/test_flat_tile.py` (the tile family's config contract, the over-large-tile and tile-hint refusals) and `tests/test_flat_isolation.py` (the case-1 table). |
| **pwa_gen (flat config), CUDA s0c compute** | `cuda_flat_pwa_gen_s0c` | `cuda_v5_pwa_s0c`'s compute on the flat config with the fill replaced; stores no per-event interpolation cache — the memory choice, not the speed one. `n_res == 1` only; gamma weights must be exactly 0 or 1. Fixtures `tests/_flat_compute.py` (the shared compute suite) and `tests/test_flat_dedup.py` (the config contract; s0c2's bit-identical fused-reduction parity). |
| **pwa_gen (flat config), CUDA fixed-coefficient split** | `cuda_flat_pwa_gen_s0f` | Case-1 copy of `cuda_flat_pwa_gen_s0e2` that SPLITS the wave axis: a wave whose entire `coeff` row is declared fixed is iteration-invariant (`D` too), so it is cached once per handle as `Afix[e,k,p] = Amp·conj(D)/\|D\|²` (`float2`, `8·n_proj·n_fix` B/event) and costs nothing per iteration inside the tile: the fixed block is LINEAR in `ck`, so it lives OUTSIDE the tile — kernel F (`A1[e,p] = Σ ck_k·Afix`, no smem, no barriers), the tile T (its parent's phases over `n_free`, seeded from `A1` once and writing its finished `dA` to a global buffer) and kernel B (`dck_k = Σ dA·Afix`, a GEMV into the parent's `g_ck`). `gamma`/the CSR are built for the FREE waves only. `n_fix == 0` launches neither F nor B and allocates neither buffer (the parent's path); `split_fixed=False` keeps the old in-tile fixed loops as a benchmark lever. The split is a PERMUTATION of the wave axis (fixed first) applied to the host arrays, with `ck[perm]` in and `dck[perm⁻¹]` out per call — `n_fix == 0` is the identity, hence the parent's path. **`grads["coeff"] == 0` at a CACHED wave's columns by contract** (no `D`, no `pv`; a declared-but-uncached column of a partly-fixed wave is still differentiated) — the constraint layer drops those entries. The kernel is built for ONE mask and is immutable, declared as **`fixed_cols` + `fixed_values`** (the ONE spelling; nothing declared is the parent path). The values live in the CONTEXT (per mask) and `load_data` fills `Afix` from them, so `compute` never fills and the guard compares `params["coeff"]` at the cached columns against them on EVERY call, the first included; a mask without values (or values without a mask) is an error at both levels. Names are a helper, not an API: `fixed_coeff_spec.resolve_fixed_cols(den_names, names=, param_map=)` returns COLUMN INDICES. The snapshot/lazy-fill branch and the `coeff_fixed` / `coeff_fixed_cols` config keys are GONE. The LIFECYCLE lives in the `fixed_coeff` wrapper (`tabpwa/backends/_fixed_coeff.py`, tag `delegate`): it starts mask-less (so a `Fitter` starts on the parent path), `set_fixed_cols(cols, values)` takes the declaration (the name helper is `tabpwa/fixed_coeff_spec.resolve_fixed_cols`; the constraint state's one-call form is `set_fixed_coeffs(fitter)`, which derives BOTH), frees the base and rebuilds it passing `fixed_cols=` / `fixed_values=` as constructor kwargs (a base that does not take them is mask-agnostic — a documented no-op), and every retained sample re-loads lazily on its next compute; a failed rebuild rolls back to the previous mask. Values are checked per handle (`ValueError` when a declared-fixed column moves); a name is a COLUMN, not a parameter — freeing one charge's mass moves both sectors' columns (`K3st1780p_mass` → `K3st1780p_s0/width_t` **and** `K3st1780m_s0/width_t`), so `fixed_cols` is the production form. MEASURED `rsize=17`, min-of-5 per backend with each handle loaded ONCE. The `Afix` fill happens inside `load_data` (the mask carries its values), so it is in the LOAD, not in a compute: `tests/benchmark_s0f.py` / `benchmark_s0g.py` time the load explicitly and print it as `load+fill`, and the `first` row is the first COMPUTE (fill already done). A harness that reloads per repetition measures the load+fill and inflates every ratio. `kketa`, 6 M events: 34 waves, 27 fixed / 7 free, 67/88 fixed nnz — `s0e2` 144.4 ms steady → **92.8 ms = 1.56×** (the first COMPUTE is ~93 ms: the fill is in the load); all 34 fixed → **33.7 ms = 4.29×** (0.6 M at rsize 1: 4.0 ms = 6.7 ns/event; the old F+T+B path 63.3 ms and the in-tile fixed loop 88.0 ms at 6 M). The all-fixed case runs a FUSED branch inside the tile (`n_free == 0`: A from `Afix`, the objective and the dck back-propagation in one launch — F and B are not launched and A1/dA are not allocated; a group must fit one block, `rsize <= te`, else the call falls back to F+T+B). The mixed case's F, tile and B are INTERLEAVED PER `cap` CHUNK (F(c) -> T(c) -> B(c)), and the split's scratch (`A1`+`dA`+`dscale`) is ONE `cap`-sized allocation per CONTEXT — batch-sized, so it does not depend on `ne`, and NOT duplicated per handle (measured 2.80 MB at 6 M with `cap` 50 k / P 2: `cap·P·16 + cap·P·8 + cap·8`; before this the buffers were `ne`-sized in every handle, 336 MB each and 672 MB across the data + phase-space handles, hence `cuda_aux_bytes` reports the per-context figure and `handle_has_aux` is gone); it is mask-gated (none on the fused or mask-less paths) and a mask change recreates the context, so it is reallocated with it; the chunk rule keeps `cap` a multiple of `rsize`, so a group never straddles a chunk (`chunk_plan`). Measured on the mixed wall time at 92.8 vs 90.3 ms (6 M, rsize 17) — +2.8 %, kept for the memory and the chunk-alignment invariant — because the free-wave tile dominates, and the fixed-block slope is flat: 33/32/27 fixed waves give 90.5/89.8/93.1 ms; B stays at ~72-83 GB/s (the chunk's Afix does not survive the tile's `us`/`amp` stream in the 36 MiB L2), and smaller batches make it worse (20 k 99.8, 8 k 119.5 ms). The mixed case's kernel B is EVENT-MAJOR (one thread per `(e,p)`, contiguous k, hierarchical (wave,slice) shared partials); `bwd_mode='wave'` keeps the old mapping for an A/B. Production `jpsi_eta_pi_pi` shape (26 waves, 2 M events): the config's own free `x` (8 fixed / 18 free) 45.2 → **41.8 ms = 1.08×**; every mass/width pinned (25 fixed / 1 free) 45.2 → **28.6 ms = 1.58×**. Mask-off (`s0f/off` vs `s0e2`) is within ~1 % (throttle noise: +0.8 % at 400 k, −0.7 % at 40 k). All-fixed now MATCHES the full-amplitude cache the pwa family has (`cuda_v4_pwa_cache` 6.8 ns/event at 0.6 M): 6.7 ns/event there and 5.6 at 6 M, on a cache half its size (432 vs 2176 B/event), because the fused branch is a single pass over `Afix` with no barriers. It is the MIXED case that remains barrier-bound (~15 ns/event): the tile over the free waves still runs its chunk pipeline, and that is the gap to the all-fixed figure. Registered `pwa_gen` with the `n_res1` and `fixed_coeff` tags (a multi-BW chain is refused with `cuda_flat_pwa_gen2` named; the mask contract is the shared `tests/_flat_masked.py`, selected by the `fixed_coeff` tag, so a third fixed-cache kernel is covered on registration). Fixtures `tests/_flat_masked.py` (the shared fixed-coefficient contract, including the per-context scratch memory gate), `tests/test_fixed_coeff.py` (the wrapper lifecycle) and `tests/test_flat_isolation.py` (the case-1 table); benchmark `tests/benchmark_s0f.py`. |
| **pwa_gen (flat config), CUDA un-staged fixed cache** | `cuda_flat_pwa_gen_s0g` | Case-1 copy of `cuda_flat_pwa_gen_s0f` that removes the SPLIT: the fixed block and the free block are summed in the SAME per-element loop, one thread per `(e, p)`, so there is no F, no B, no `A1`, no per-event `dA` buffer and no scratch of any kind — `cuda_aux_bytes` / `split_counts` / `fused_count` do not exist here. Per element: `A = Σ_{k<nfix} ck_k·Afix[e,k,p] + Σ_{k<N} ck_k·Amp[e,slot(k),p]·conj(D_k)/|D_k|²` with `D_k = Σ_m coeff_m·Z_m(e)` walked IN PLACE; then `P_e` from the shared `|A_{e,p}|²`, the objective, `dA = dQ/dP·conj(A)` (in place, fp64 shared), the fixed `dck` as a (wave, slice) reduction over it, the free `u`/`pv` (one thread per `(e,k)`, `D_k` rebuilt) and `dcoeff` (one thread per CSR entry, register reduction, ONE global atomic). Barriers: **3 per BLOCK** at `rsize == 1` (after A, after dA, after the backward — the last publishes the shared `dck`/`pv` partials; the per-event scale is derived redundantly by every projection's thread), **5** at `rsize > 1` (plus the group close `q -> sums -> scale`); s0f's mixed tile paid 6 per 8-event chunk. `te` is NOT a tuning knob: the largest whole-group tile inside the 48 KiB budget, capped at `GPS0G_MAX_TE` 256 (measured `te` 255 at rsize 17 on the `kketa` shape = 15 groups per block, 46.4 KiB shared), because nothing is staged. **A log-sum group must fit ONE block**: `rsize > 1` with `te < rsize` is refused at construction naming `cuda_flat_pwa_gen_s0f` (which walks a group inside one block) — the documented fallback. Accepted cost: `D_k` is recomputed once per projection forward and once per `(e,k)` backward (`P+1` CSR walks instead of 1 + a staged `common`); the per-free-wave slope is 0.49 ns/event against s0f's 0.32, and the mask-off path is therefore +2.4 % over `s0e2` (149.1 vs 145.6 ms at 6 M, the double walk). MEASURED `kketa`, 6 M, rsize 17, handles loaded once, min-of-3 after two warm-ups: mixed 27/7 (14 moving columns) `s0e2` 144.5 → s0f **92.8** → **s0g 72.1 ms = 12.0 ns/event, 1.29× over s0f** (first call 101.9); the fixed-wave ladder 33/32/27 fixed → **60.2 / 56.8 / 71.9 ms** (1.50× / 1.59× / 1.29× over s0f) — the win scales with the FIXED fraction, which is the barrier pipeline it removes; at 0.6 M the n_free sweep 1/2/4/7/14/20/27/34 free gives 1.35/1.52/1.40/1.27/1.10/1.02/1.00/0.97×. **All-fixed is s0f's**: 33.7 (5.62 ns/event, its fused branch) vs **39.0 (6.49)** = s0g 1.16× SLOWER (s0g still pays the `|A|²` staging and a projection-redundant `P_e`), so s0g is the MIXED-case alternative, never the all-fixed one. `ncu` (0.5 M, 50 k chunks): the ONLY per-iteration kernel is `gps0g_tile_kernel`, 1.084 ms per 50 k-event chunk (197 blocks), with `gps0g_fill_afix` 4.56 ms once — F/B do not exist. Registered `pwa_gen` with `n_res1`; the `fixed_coeff` wrapper drives it unchanged (`base: cuda_flat_pwa_gen_s0g`; the mask arrives as `fixed_cols` + `fixed_values`, and `load_data` fills `Afix` from them), and `grads["coeff"] == 0` at a CACHED wave's columns is the same contract. Fixed-wave `dck` differs from s0f at 1.8e-8 (s0f stages its finished `dA` through a float2 buffer for kernel B; s0g keeps it fp64, so s0g is the one nearer `numpy_flat`), while `Q`/`P`/`dcoeff`/`dnorm` agree to 2e-15. Fixtures `tests/_flat_masked.py` (the shared fixed-coefficient contract) and `tests/test_flat_isolation.py` (the case-1 table, which now also carries the `kernels_flat_pwa_gen_s0f.cu` sha256 pin); benchmark `tests/benchmark_s0g.py`. |
| **pwa_gen (flat config), CUDA ampcache** | `cuda_flat_pwa_gen_ampcache` | Case-1 copy of `cuda_flat_pwa_gen_s0f` specialised to "EVERY coefficient is pinned": no mask, no wave permutation, no F/T/B, no `split_counts`/`aux_bytes`/`fused_count`/`chunk_plan`. The whole per-wave denominator is a PER-HANDLE cache — `Afix[e,k,p] = Amp·conj(D)/|D|²` as `float2` (the flat family's only cache dtype; "store fp32" is not a variant here, it is the mode) — so an iteration is `A = Σ_k ck_k·Afix`, `P_e = Σ_p |A|²`, `dck = Σ dA·Afix`, and `dcoeff = 0` at EVERY column. 544 B/event at the `kketa` shape (34 waves × 2 projections), i.e. the same bytes as `s0f` all-fixed, and MEASURED the same speed (7.6 ms at 0.6 M, 1.00× — the script says so; the value is the CONTRACT, not a speedup: no `fixed_cols` to derive and keep right, and a moved weight raises instead of silently reusing the cache). `dck_k = Σ_{e,p} dA[e,p]·Afix[e,k,p]` with **NO conjugation on `Afix`**: `dA` is already `dQ/dP·conj(A)` (`numpy_flat`'s `dQ_dA = dQ_dP·conj(A)`, then `grad_ck = Σ dQ_dA·common·fa`), so a second conjugate would be `|A|²` and the fit would descend in the wrong direction — the one mistake a copy of this kernel can make silently, which is why the .cu header, the wrapper docstring and the test header each state it. MEASURED against the uncached `s0e2` (min-of-5 interleaved, `tests/benchmark_ampcache.py`): **3.88×** at the equal-size sample (0.6 M data + 0.6 M phsp) and **3.25×** at a 2 k data / 0.3 M phsp mix — the ratio is SAMPLE-DEPENDENT (it is the per-iteration `D` walk the cache removes, so the mix that spends relatively more time in the data pass gains less), never a shape-independent constant. The LIFECYCLE is `cuda_v4_pwa_cache`'s, and there is exactly ONE: no `coeff=` constructor argument, `load_data` NEVER fills (a handle comes back `_filled = False`, `_coeff = None`), and a handle's FIRST `compute` calls `_fill_once(handle, coeff)` from `params["coeff"]` BEFORE any launch. From the handle's SECOND call on the snapshot is strict per HANDLE (`_filled`/`_coeff`, v4's `_fixed_mg` shape) — the fitter's data and phase-space handles each fill once and each guard their own — and a `coeff` that differs raises `cuda_flat_pwa_gen_ampcache: coeff moved after the cache was built — the basis weights must be pinned; use cuda_flat_pwa_gen_s0e2 when they float`. Nothing about the pinned values lives in the constructor, the kernel config or a module level, so the state cannot be reached twice, differently. (Consequence for timing: the fill sits in `first`, not in the load — `tests/benchmark_ampcache.py` prints `load` and `first` separately for exactly that reason.) Tagged `coeff_pinned`: a pinned kernel cannot satisfy the free-coefficient contract (FD on `coeff`, the mask-off anchor), so `tests/_flat_compute.backends_to_test()` subtracts that capability exactly as it subtracts `reference`/`delegate`, and `test_pwa_gen_resolution_size`'s registry sweep skips it for the same reason — its own file pins what it does. `rsize > te` keeps s0f's PATH-B semantics (one block per group, `te`-sized inner chunks, a second gather sweeps the cache) because the F/T/B fallback does not exist here; `te` is `clamp(8·rsize, 64, 256)` floored to whole groups — the "largest tile that fits smem" rule was MEASURED 22 % slower, and neither `te` nor the spill count is a TEST here — they are HOW the kernel works, so they belong to `tests/benchmark_ampcache.py`. `n_res == 1` is the single-row case of the GENERAL denominator — a PRODUCT over the factor axis (`D_k = Π_l Σ_m gamma[k,l,m]·coeff_m·Z_m`), one CSR row per `(wave, factor)` — so a multi-BW chain is computed like any other leaf (MEASURED `_mb`/`n_res == 2` at its own fp32 floor, and a 3-iteration BFGS through the `Fitter` matches `cuda_flat_pwa_gen2` to 2e-7 in the objective); what is still refused is a CSR that cannot describe that product (`n_wave·n_res + 1` rows, one weight per non-zero, and a factor axis agreeing with the numerator's chain). The GRAM: `compute_gram(handle, weight=None, coeff=None)` reduces the SAME cache into `D[k,j] = Σ_e w_e Σ_p conj(Afix[e,k,p])·Afix[e,j,p]` (conjugation on the FIRST index — `integrated_pwa_gen`'s `einsum("e,ekp,ejp->kj", w, c.conj(), c)` and `cuda_gram_from_cache_v4`'s `conj(c_k)·c_j`), so `Σ_e w_e P_e = Re(ck† D ck)` with no transpose or sign. It goes through the same `_fill_once` (an unfilled handle needs the `coeff` to fill; a filled one is guarded) — no third way in; `weight` is UPLOADED per call into a context buffer and the handle's own copy is not consulted. MAPPING: one block owns `ev` events (host rule: `8·SMcount` blocks, rounded up to whole 8-event tiles), stages 8 events at a time into SHARED as `float2 sA[8][N][P]`, and each thread owns a FIXED partition of the upper triangle `k<=j` (closed-form row-major inversion) which it accumulates in REGISTERS over the tile — so the inner loop has no atomics at all (one owner per element) and ONE shared read-modify-write per element per tile, and the whole Gram (`2·N²` doubles, 18.5 KB at N=34) lives in shared with ONE global `atomicAdd` per element per block at the end. `D[j,k] = conj(D[k,j])` is written from the same product (the asymmetry is then only the cross-block atomic order, MEASURED 1.9e-16). RE-ASSOCIATION: the total is the sum of per-block partials in retirement order, so it is order-dependent at ~1e-15 (fp64 partials of fp32 cache entries) — the family's accepted floor; the Gram is built once per pinned `coeff`, so nothing per-iteration rides on it. `gram_kernel_stats()` reports regs/local/blocks (MEASURED 40 regs, 0 local, 6 blocks/SM, 48 warps/SM at the `kketa` shape; 448 B shared at the km fixture, 22.8 KB at 34×2). `integrated_pwa_gen` PREFERS it through `getattr(base.kernel, "compute_gram", None)` and STREAMS the phase space (one `mc_batch` batch loaded, Gram-reduced and freed at a time, `integrated_pwa`'s pattern), with the numpy builder kept as fallback AND oracle. MEASURED 0.6 M phsp events (`kketa`, `mc_batch` 5000, one process, both builders on the same sample back to back): numpy 5.5-5.7 s vs GPU 137-160 ms = **35-41×** (numpy was 4088 ms at 26× with the linear oracle, i.e. the scheme is what these milliseconds buy), peak `gpu_used` 180 MiB for the build (ONE batch = 2.7 MB of cache, not the sample's 326 MB), `|Dgpu − Dnumpy|/|Dnumpy|` **6.6e-8** max/max (3.1e-8 Frobenius) — one fp32 ulp of the cache, `2^-24 = 6.0e-8`; it read **2.0e-5** while the numpy builder interpolated the tables linearly, so that figure was the INTERPOLATION, not the cache (A/B here: 6.6e-8 with the kernels' Catmull-Rom, 1.96e-5 with the old two-point scheme) — **and the NORM, which is what a fit consumes, is 1.7e-11 against `s0e2`'s own `Σ w P` where the numpy builder is 3.5e-9 from that same norm**, because the GPU reduces the very cache the production kernels use while the numpy path is a separate fp64 evaluation of the same tables; read that gap as 3×-of-nothing rather than the five orders it was: the numpy builder sat 1.1e-6 from that same norm (2.3e-6 from `s0e2`'s `Σ w P`) only while IT was the thing that did not match — `numpy_flat` read the tables two-point linearly then and reads them with the kernels' own Catmull-Rom now, so the remaining 3.5e-9 is fp64 association, not interpolation (SAMPLE-DEPENDENT: the same comparison on `tests/benchmark_ampcache.py`'s own synthetic sample — `_events(kc, ne, seed+1)`, not the shipped `kketa` file — gives 2.9e-8 at the same 0.6 M events, so quote the sample with the number); break-even 41-48 iterations against `ampcache`'s own per-iteration phsp pass (137-160 ms ÷ 3.26-3.34 ms), or ~10-12 against an `s0e2` pass (13.8 ms) — the DATA pass is not saved by a Gram, so it is in no break-even here. At 5 M phsp (6 M total, data 1 M) two runs give numpy 45.9-55.7 s (34.1-40.0 s linear) vs GPU **1.21-1.22 s = 38-46×**, peak 180 MiB, and the two builders agree to <=2e-10 in the norm (1.2e-6 linear), break-even 43 / 9 iterations. THE CACHE ADDRESSING is `Afix[e,k,p]` (p innermost, the k stride is `P`) and that is MEASURED, not assumed: the k-innermost alternative `Afix[e,p·N+k]` — a pure permutation inside each event's block, the same numbers through the same kernel, so a runtime `layout=` flag was legitimate to A/B — is a WASH, min-of-5 interleaved on the `kketa` shape (`pk` / `kp`): 0.6 M (0.3 M + 0.3 M) 4.2 / 4.2 ms (6.96 / 6.96 ns/event, 156 / 156 GB/s of the 2·544 B/event the two sweeps move) and 6 M (1 M + 5 M) **33.8 / 33.9 ms = 0.998×** (5.64 / 5.65 ns/event, 193 / 193 GB/s), Gram build 358 / 362 ms, tile 48 regs and 0 local bytes EITHER WAY; the winner alone is 12.2 ms (6.11 ns/event, 178 GB/s) at 1 M + 1 M. Parity first: both layouts agree on `Q`/`P`/`dck`/`D` to ≤ 5e-16 relative (P and dck bit-identical — every element's k-run is summed in the same order), so the switch is addressing only; a deliberately mismatched fill breaks it to O(1), which is what makes the equality a gate rather than a no-op. `ncu` on the tile says why the layout cannot matter: **FP64 pipe 85 % of peak, DRAM 11 %, L2 13 %**, 40 % of the warp cycles are CTA-barrier stalls and the scheduler issues every 16.4 cycles — the kernel is fp64-THROUGHPUT bound (939 k non-fused vs 445 k fused fp64 instructions, i.e. up to ~34 % available from FMA fusion), so the cache's byte pattern is not the limiter. The `kp` path and the flag are DELETED (no `layout=` argument survives); the addressing is not a lever here. RESIDUAL vs `cuda_v4_pwa_cache`, and where the premise's gap actually comes from: at 1 M data + 1 M phsp (34 waves × 2 proj, min-of-3/5, same Fitter) ampcache alone is **12.2 ms = 6.12 ns/event**, `integrated_pwa_gen` over it (Gram norm, one 332 ms streamed build) is **6.8 ms = 3.41 ns/event**, and `integrated_pwa(base=cuda_v4_pwa_cache)` — fp64, 2176 B/event — is **7.42 ms = 3.55 ns/event** (its own norm is the Gram, 0.0066 ms). So under the SAME norm ampcache is 1.09× AHEAD at a quarter of the bytes, and the "4.2 vs 3.9 ms" that motivated the layout A/B compared ampcache's FULL per-call phase-space pass against v4 behind the Gram: the gap was the norm, not the cache. Parent source pinned by sha256 (same digest as s0g's pin). Fixtures `tests/_flat_pinned.py` (the shared pinned-coefficient contract), `tests/test_flat_ampcache.py` (the dtype/Gram claims) and `tests/test_flat_isolation.py` (the case-1 table); harness `tests/benchmark_ampcache.py`. |
| **pwa_gen (flat config), CUDA ampcache fp32** | `cuda_flat_pwa_gen_ampcache32` | Case-1 copy of `cuda_flat_pwa_gen_ampcache` with ONLY the per-iteration contraction moved to fp32.  The split is per QUANTITY, not per file: the fill (`us`, `amp`, `Afix`, i.e. `D_k = Π_l Σ_m gamma·coeff·Z`) stays **fp64** — it is the delicate interpolated sum of the flat model, it runs ONCE per handle, and the stored `Afix` is byte-identical to the parent's; the staged `A`/`dA` (`float2` shared), the contraction `A = Σ ck·Afix` and `dck = Σ dA·Afix` run **fp32** (~68 complex MACs/event, no cancellation-prone sum); every REDUCTION — `Q`/NLL, the group totals, `P_e`'s `Σ_p`, `dnorm`, the global `g_ck` (its float block partials widen to fp64 before the one atomic) — and the Gram stay **fp64**.  Why fp32 is defensible HERE and not in `s0e2`: the denominator walk that fp64 protects is inside the cache, evaluated once.  MEASURED vs the fp64 parent on the SAME cache (the second rounding alone): Q 1.05e-6 worst (`_mb`, norm 1.3), typically ~1e-8, P 2.1e-7, dck 1.4e-7, dnorm 4.4e-9 — at or below the cache's own floor (vs `numpy_flat`, at the `_flat_pinned` ONE_BW anchor: Q 7.6e-8 / P 3.8e-7 / dck 2.9e-7, i.e. the parent's own figures to ~10 %; both read 8.9e-7 / 3.3e-6 / 1.3e-6 against the LINEAR-interpolating reference, which is no longer what `numpy_flat` is) — and a 3-iteration BFGS on `_mb` gives ΔNLL 1.4e-5 at NLL 1598 (8.7e-9 rel) with **max|Δx| 2.3e-6**, i.e. no fit degradation.  SPEED (min-of-5 interleaved, `kketa` 34 waves × 2 projections, 544 B/event): 0.3 M + 0.3 M **6.94 → 3.87 ns/event = 1.79×**, 1 M + 5 M **5.64 → 2.39 ns/event = 2.36×** (cache stream 113 → 266 GB/s; kernel_stats both 48 regs / 0 local, tile smem 4760 vs 7960 B).  It is NOT memory-bound even now: `ncu` on a 600 k-event chunk gives 3.98× with **DRAM 34 %** (parent 8.6 %) and the **fp64 pipe still 56 %** — the residual is the SCALAR EDGE (`dqp = weight/norm·scale` per element plus the fp64 `P_e` accumulation the contract keeps): 88 k fp64 thread-ops against 4.3 M fp32, yet on a 1/64-rate unit they cost about as much pipe time as the whole contraction did.  So the win is chunk-dependent (2.36× at the production 50 k chunking, 3.98× in one huge chunk) and the fp64 pipe cannot idle without moving the reductions, which the precision policy forbids.  Two gate lessons recorded in the fixture: a first ILP unroll SPILLED 56 B at 48 registers (the register budget, not the arithmetic; `tests/benchmark_ampcache32.py` prints `kernel_stats()` so a spill is visible where it matters), and the central-difference step must be **h = 1e-2, not the parent's 1e-6**, because `Q` carries a fixed fp32 offset that varies with `ck` (at 1e-6 the check appeared to condemn a gradient that matches the parent to 1e-7).  Same lifecycle (`_fill_once` per handle, strict from the second call), same guard text with the new name, `grads["coeff"] == 0`, `n_res` general, same `te` rule and `rsize > te` path B; tagged `coeff_pinned` (the only declaration).  Parent source pinned by sha256 (`a2b53fa4…`).  Fixtures `tests/_flat_pinned.py` (the shared pinned-coefficient contract), `tests/test_flat_ampcache.py` (the second-rounding claim) and `tests/test_flat_isolation.py` (the case-1 table); harness `tests/benchmark_ampcache32.py`. |
| **pwa_gen (flat config), CUDA factor dedup** | `cuda_flat_pwa_gen_fslot` | Case-1 copy of `cuda_flat_pwa_gen2` (SAME C identifiers — a copy is made WITHOUT RENAMING) that runs the coefficient-gradient sweep over distinct FACTOR rows instead of wave blocks: `(wave, factor)` gamma rows are grouped by the ROW BYTES across waves *and* across the factor axis (the basis mass column is a property of the basis row, not of a `(wave, factor)` row, so it is not part of the key), so two topologies that share a resonance compute it once — `1 + n_topo` dots against `2·n_topo` (`_mb`: 2 → 2, one topology; `mb2`: 6 → 4, three sharing `pipi`).  BACKWARD ONLY: phase 1 and `Q`/`P`/`dck`/`norm` are the parent's; the `(f, m)` sweep is fed by a CSR pass `dqds_{e,f} = Σ_{r∈f} dQ/ds_{e,r}` that runs INSIDE the existing `dsum` window (between the phase-4 barrier and the next), so **no `__syncthreads()` is added** — the tile is barrier-bound.  `use_fslot` forces the wave collapse OFF (the sweep needs the per-row `s_pv`); that can never cost dots, `n_fslot <= n_slot·n_res` always.  Gate `n_res > 1 and n_fslot < n_slot·n_res` — a REAL saving against the parent's own dot count on the path it would take, not a ratio: at equality the CSR accumulation buys no dot and measured **4 % SLOWER** (interleaved min-of-5: `_mb` gen2 0.72 vs fslot 0.75 ms), so `_mb` (2 factor rows against its collapsed 1×2) takes the PARENT path and only `_mb2` (4 vs 6) and `4pi` (12 vs 16) engage; `n_res == 1` never engages (`fslot ≡ slot`).  Host map is a VERBATIM copy of `slot_map` applied to `gamma.reshape(n_wave*n_res, n_basis)` (standalone rule, pinned in `tests/_flat_isolation.HELPER_COPIES`).  MEMORY: the tile is the parent's **+128 B** on the shapes that matter (`pv` uncollapsed, `fs` replaces the `dd`/`w` the collapse would have needed); only a contrived structure where the collapsed parent keeps a SMALL `n_slot` while the factor rows stay many can make `fs` exceed `dd`+`w`.  NOT bit-comparable to `gen2` by construction, and this is a property of the PARENT: gen2 is not reproducible against ITSELF (phase 2 splits the wave axis over lanes and each lane `atomicAdd`s) — measured gen2-vs-gen2 `Q` 1.0–2.3e-15, `P` bit-equal only on `_mb`, `dcoeff` 1–35 ulp; fslot's deviations sit at or below that (`Q` 8.1e-16, `dck` 4e-16, `dcoeff` 5 ulp `_mb` / 26 ulp `mb2`, `dnorm` 0–3 ulp), gated at `1e-13` relative with the baseline table in the fixture.  MEASURED `get_nll` (`benchmark_mb_factor.py --n 20000`, interleaved min-of-5): `_mb2` 1.20 → 1.09 ms = **1.10×**, and `_mb` — which the gate now sends down the parent path — the same as gen2 (it was 0.75 vs 0.72 = 0.96× while the gate let it through). Small on 4/12-wave fixtures where the dot count barely moves; the win is structural and grows with `1 + n_topo`, and the forward is untouched.  Fixtures `tests/config_pwa_gen_mb2.yml` (three topologies sharing `pipi`), `tests/config_pwa_gen_4pi_mb.yml` (`n_blocks > 1` × `n_res > 1`: 32 waves = 8 identical-particle blocks × 4, 12 factor rows against 16 dots) + `tests/benchmark_mb_factor.py`; contract `tests/test_flat_dedup.py` (the redundancy counts; the floor gate; the multi-BW anchor and its FD check, both parametrised over the three multi-BW shapes so `n_blocks > 1` × `n_res > 1` is covered by the CONTRACT and not by one bespoke test; and that test's 4pi case, which is evaluated at the model-default point AND at a second, independently drawn one and asserts at each that both kernels repeat to 1e-13 before comparing, so the parity cannot be launch order) and the case-1 row in `tests/test_flat_isolation.py` (own `.so`/module, `symbols: "parent"`). |

| **pwa_gen (flat config), Gram norm** | `integrated_pwa_gen` | The flat config's pre-integrated norm, the sibling of `integrated_pwa`: with `coeff` PINNED the per-event amplitude `c = common·fa` is linear in `ck`, so `Σ_e w_e P_e = ck†·D·ck` with `D[k,k'] = Σ_e w_e Σ_p conj(c)·c` — an `N×N` Hermitian Gram (`N = n_wave`; the `Σ_p` collapses INSIDE it, so no projection axis is needed) built once per pinned `coeff` and then evaluated in O(N²). Delegates the data NLL to `base` (default `cuda_flat_pwa_gen_s0e2`) and supplies `norm` itself. The algebra is NOT re-derived: `numpy_flat` grew `_parse` / `_coeff_block` / `_wave_amplitudes` and a public `wave_amplitudes(data, coeff)`, and the Gram calls that — the reference and the norm cannot drift (value vs the reference's own `Σ w P` to 1e-12, `∂norm/∂ck` to 1e-10). MEASURED at 0.6×10⁶: build 5.5-5.7 s (one numpy phsp pass; 4.1 s while the reference interpolated the tables two-point LINEARLY — a Catmull-Rom read is 4 gathers against 2, so matching the kernels' scheme costs this builder 1.36×), then 0.002-0.008 ms per iteration (Python overhead dominates at N=34) against the 13.8 ms phase-space pass it replaces — **break-even ≈ 397-410** (three runs: 5477/13.80, 5576/13.80, 5681/13.85), and it is scale-free because both sides are linear in events: 5 M measures 45.9-55.7 s ÷ 114.9-119.4 ms = 400-470. **THE BUILD IS NO LONGER THE NUMPY PASS when the base has a GPU Gram**: `cuda_flat_pwa_gen_ampcache.compute_gram` reduces its own `Afix` cache on the device, and this backend PREFERS it (`getattr(base.kernel, "compute_gram", None)` AND `getattr(kernel, "gram_ok", True)`) while keeping the numpy builder as the fallback for every base without one AND as the oracle it is checked against. The GPU build STREAMS the phase space exactly as `integrated_pwa._ensure_gram` does — `mc_batch` events loaded into the base, Gram-reduced, freed, `h.free()` before the next, weight sliced with the same bounds — so the peak device memory is ONE batch's cache (2.7 MB at 5000 events on the `kketa` shape) plus the `N×N` accumulator, never `ne·n_wave·n_proj·8` (326 MB at 0.6 M). MEASURED (0.6 M, `mc_batch` 5000, both builders on the same sample in one process): numpy 5576 ms → GPU **160 ms = 35×** (numpy 4088 ms / 26× with the linear oracle), peak `gpu_used` 180 MiB for the build, **break-even ≈ 41-48 iterations** against the per-iteration norm the Gram replaces (137-160 ms ÷ 3.26-3.34 ms); at 5 M phsp (6 M total, data 1 M) two runs give numpy 45.9-55.7 s (34.1-40.0 s linear) → **1.21-1.22 s = 38-46×** (a ~20 % run spread on the numpy side at this size; the CR-vs-linear ratio is 1.35-1.39 either way), break-even 43. The GPU norm also agrees with `s0e2`'s `Σ w P` BETTER than the numpy builder — but no longer by five orders: 1.7e-11 (0.6 M) / ~1e-10 (5 M) for the GPU against a numpy builder that is 3.5e-9 / <=2e-10 from that same norm, having been 1.1e-6 / 1.2e-6 while it read the tables two-point linearly. The GPU reduces the cache the kernels compute `P` from; the numpy path is now an independent fp64 evaluation that agrees to association order, which is what an oracle should be. A batch boundary changes nothing (`mc_batch` 5000 vs 137 agree to 1e-12 relative, gated). The GRAM IS A FUNCTION OF `coeff` and that is enforced, not assumed: a moved weight raises `integrated_pwa_gen: coeff moved after the Gram was built — the basis weights must stay pinned` from the SECOND call on and on BOTH branches — the norm branch (`_ensure_gram`, up to 1e-12) and the data/plot branch (`return_p`), which compares `params["coeff"]` against the BACKEND-level snapshot with a cheap `array_equal` and never builds a Gram from a data handle (a data bundle has no `D`: the Gram integrates the PHASE SPACE). The FIRST call of a fresh backend DEFINES the point, so it cannot raise — that is the fitter starting up, not a weight that moved — where `integrated_pwa` would have silently zeroed `m0/g0` and left a dead parameter in the fit. WHICH BUILDER runs: the GPU one only when the base's kernel has `compute_gram` AND its own `gram_ok` — the Gram's tile is `2·N² + 8·N·P` DOUBLES of shared (22 848 B at `N=34, P=2` for `kketa`), so a wide wave axis exceeds the device's opt-in ceiling (`N=76` needs 102 144 B and this box offers 101 376; `N=200` exceeds any device) and `gram_ok` is False; otherwise the numpy oracle builds `D`, and a GPU call that fails at RUN time falls back the same way with a `RuntimeWarning` naming `tabpwa.integrated_pwa_gen.IntegratedPWAFlat` (a shape can be refused between the check and the call, so the fallback is not dead code). Built on the PHSP bundle only. `resolution_size` is accepted and FORWARDED to the base: grouping changes the data NLL, not the normalisation (`norm` is the same integral; a test pins rsize 1 == 17 to 1e-12). Registered `delegate`, so the flat contract suite does not sweep it. The streaming means the GPU figure includes a load+fill per batch — that is the honest cost of never holding the sample on the device, and it is what the 35× and the break-even are computed from. Fixture `tests/test_integrated_pwa_gen.py`. |
| **v5 PWA log-sum** | `cuda_v5_pwa` | Projection-sum PWA with the grouped log-sum data NLL (`resolution_size`, default 1); per-event `P` unchanged. Comparisons against v4 at `rsize > 1` need group-normalised weights. |
| **v5 PWA log-sum s0** | `cuda_v5_pwa_s0` | `cuda_v4_pwa_s0` plus the grouped log-sum; `rsize=1` bit-identical to `cuda_v4_pwa_s0`. Fixture `tests/test_pwa_v5_s0.py`. |
| **v4 PWA s0b** | `cuda_v4_pwa_s0b` | `cuda_v4_pwa_s0` with device-side output accumulation and pinned/async parameter upload; identical math (summation order ~1e-16). Pick it over `s0` when per-call overhead dominates (small samples). |
| **v4 PWA s0c** | `cuda_v4_pwa_s0c` | `s0b` in the uniform-coefficient format (denominator is the basis sum, so every fitted coefficient shares one gradient formula). Tile sibling `cuda_v5_pwa_s0c` adds the grouped log-sum NLL. |
| **v4 PWA s0d (u-cache)** | `cuda_v4_pwa_s0d` | `s0c` with the iteration-invariant γ-interp cached at load; results identical to `s0c`. Wins on narrow-wave configs, ~1.0× on wave-dense shapes. Sibling `cuda_v5_pwa_s0d` adds the grouped log-sum NLL. |
| **v3 sparse** | `cuda_v3_sparse` | Split-kernel sparse v3 (matrix_gamma 99.5 % sparse): fp32 FA, fp64 g0/m0/BW. 3.8× faster than `cuda_v3`. |
| **CPU (C+OMP)** | `cpu_v3`, `cpu64_v3` | Pure C + OpenMP + AVX2, auto-built via gcc. 6.3× faster than `numpy`; no GPU required. |
| CUDA f64 v3 | `cuda_v3`, `cuda`, `cuda64` | Catmull-Rom (default standalone). |
| CUDA f32 v3 | `cuda32_v3` | Faster, ~1e-7 precision. |
| CUDA f64 v2 | `cuda_v2`, `cuda64_v2` | Linear interpolation. |
| CUDA mixed v3 | `cuda_mixed_v3` | Mixed fp32/fp64 v3 variant. |
| CUDA f32 v2 | `cuda32_v2` | Fastest raw kernel. |
| flat CPU (C+OpenMP, AVX-512 / AVX2) | `cpu_flat_pwa_gen` | The flat `pwa_gen` config on the CPU: C + OpenMP, events as lanes, `numpy_flat`'s semantics and gradients.  The vector WIDTH is a compile-time abstraction with **8 LOGICAL lanes in both builds** (one `zmm` on AVX-512, two `ymm` halves on AVX2+FMA) — a 4-lane tile would move the tile partition and every reduction grouping, so the builds would differ at 1e-16; as written they are **bit-identical** (`np.array_equal` on Q/P/dck/dcoeff/dnorm, 3 fixtures x rsize x norm).  `-ffp-contract=off` is required because GCC's FMA contraction differs per ISA cost model, and `log` must come from the 4-wide glibc entry in both (its 8-wide form is not bit-equal).  **21.3 Mev/s** (data pass, full sample) vs the GPU tile's 36.8 and AVX2's 19.2 (1.11x, gather-bound); ~160x the only other CPU path (`numpy_flat`, 0.13).  No load-time cache: 34.6 % is the per-event Catmull-Rom gather the GPU buys with 5.9 GB (`us`); the lever is a loop swap, not a cache. |
| flat CPU cached | `cpu_flat_pwa_gen_cached` | Case-1 copy of `cpu_flat_pwa_gen` with the TILE's load-time caches: `us` (basis rows at the deduped slots), `fa` (deduped angle columns) and `fn` = `num·fl` per wave, so a compute only streams them and reduces.  **Bit-identical** to the uncached kernel (the caches are the same interpolation code) — that identity is what forced `fn` to stay `num·fl` rather than the combined `fa·num·fl` (pre-combining rounds differently).  **1600 B/event** fp64 (us 768 + fa 416 + fn 416) = 16.4 GB per fit at 8.7 M, fill ~1 s; **31.6 Mev/s** data pass (1.49x the uncached 21.3; GPU `s0e2` 37.7) and BANDWIDTH-bound — the 14 GB stream is 57.6 % of the pass at 87.7 GB/s, i.e. this box's read ceiling.  In a hybrid `shard` with `s0e2` it lifts the ceiling to **61.6 Mev/s** at `[64,36]` (48.5 with the uncached CPU). |
| flat CPU cached, product | `cpu_flat_pwa_gen_cached2` | The split cache's sibling with the amplitude cached as the FINISHED product `pfa = fa·fl·num` per (event, wave, projection), folded in the GPU tile's order (`F = fl·num`, then `F·fa`), so the reduce keeps no `fa` array and does no `fa` multiply.  Bytes: `(n_uslot + N·P)·16` vs the split's `(n_uslot + n_fa + N)·16`, so the product wins iff `n_fa > N·(P−1)`.  MEASURED on the real sample: `P = 2`, `n_fa = N = 26` is a TIE (1600 B either way) and gains 1.4 % from the missing multiply (272.9 ms / 32.05 Mev/s vs the split's 276.6 / 31.63); **`P = 1` drops 15 % of the bytes and gains 15 % (214.3 ms / 40.81 Mev/s) — faster than the GPU tile's 37.7 on the same sample**.  Streams at 87 GB/s (58.1 % of the pass), i.e. still memory-bound.  NOT bit-identical to the split cache (the association moved): worst 28.8 ulp (`dcoeff`, km fixture), gated at ≤ 64 ulp with the exact-entry count reported, and ≤ 5.4e-15 relative against the uncached kernel; the AVX-512 and AVX2 builds remain bit-identical to each other. |
| NumPy | `numpy` | CPU reference implementation (f64). |
| NumPy PWA | `numpy_pwa` | CPU projection-sum reference implementation (plain `pwa` only; the `pwa_gen` reference is `numpy_flat`). |
| Integrated PWA | `integrated_pwa` | Gram-matrix norm for the projection-sum PWA + base (default `cuda_v4_pwa`). |
| v4 PWA cache f32 | `cuda32_v4_pwa_cache` | fp32 variant of `cuda_v4_pwa_cache`. |
| Shard | `shard` | Multi-process wrapper sharding data across workers (`backends`, `weights`, `align`, `start_method`). Universal (`model=None`).  `spawn`/`forkserver` re-EXECUTE the calling script in every worker (`runpy.run_path(…, run_name="__mp_main__")`, because the `Process` object is pickled and its target may live in `__main__`), so the program's top-level code must be under `if __name__ == '__main__':` — an unguarded program recurses until Python's bootstrap guard kills the child (a bare `multiprocessing` program then exits **0** with a dead worker).  The backend reports both of ITS failure modes instead of hanging on a queue that can never produce a reply: `shard worker i could not be started (…)` when `p.start()` fails (the parent sees `BrokenPipeError`), `shard worker i died (exitcode=…) without returning a result` when a worker disappears after starting (bootstrap crash, segfault, OOM) — detected by polling the process, since `worker_timeout=None` means "no deadline", not "no diagnosis".  One limit is CPYTHON's, not the backend's: a pickled `Process` or kernel config larger than the 64 KiB spawn pipe, with a child that dies while importing `__main__`, makes the PARENT block inside `p.start()` (measured on 3.12; 3.13 raises `BrokenPipeError`) before any backend code — so a spawn-guard gate must use a tiny config, a worker backend that can actually be BUILT from it, and a `compute` after the load (a `{}` config plus a real backend dies in the backend's constructor, which looks exactly like a bootstrap death).  And a failure of ANY command (load/free included) replies ONCE as `(_ERROR, phase, traceback)`, so it is reported where it happened and the pool then REFUSES further work (the failed worker's later replies are still queued, so one reply per compute no longer holds).  Workers are DAEMON processes the backend owns (a worker never starts a child), so an unhandled error in the parent exits instead of joining a worker blocked on `task_queue.get()` forever.   NO DEFAULT DEADLINE: `worker_timeout` is `None` unless the caller asks, and a `load_data` PREPROCESSING has no deadline at all — nothing may assume how long a kernel's load or compute takes.  The only intervals are liveness SLICES (0.5 s parent poll, 5 s idle worker poll); a worker exiting with its `kill -9`ed driver is the one thing not assumed.  `free()` is COOPERATIVE (the `("exit",)` command, then wait for the worker's current task, no deadline); `kill()` is the immediate form and is what `__del__` uses.  The kill pass runs in a `finally`, so an interrupt of that wait (or a command that could not be delivered) still ends the pool — a worker that never reads the command keeps `free()` waiting BY DESIGN.  The join AFTER a SIGKILL is bounded (`_REAP_WAIT`, one budget for the whole pool) because a reap is not a wait on anyone's work.  The worker's parent check is `os.getppid()` PLUS the driver's pid, NOT a multiprocessing sentinel: under `fork` a worker inherits its siblings' sentinels, so a sentinel-based pool stopped one worker per slice in creation order, and under `forkserver` the worker's DIRECT parent is the forkserver, which outlives the driver — only the driver's own pid sees a `kill -9`ed driver in that mode.  The worker protocol is exactly the commands it receives — CREATE (its own bootstrap), LOAD, COMPUTE, EXIT (an explicit `("exit",)` command — no `None` magic value) — and only a COMPUTE has a NORMAL reply (a failure of any command replies once, naming its phase, and the pool then refuses further work rather than shift the stream); ordering is the queue's, so a worker runs its load before the compute that follows it. |
| ONNX | `onnx_cpu` / `onnx_cuda` | In-memory graph, built with batch_size=1024. |

`cuda` / `cuda64` alias to `cuda_v3`.  GPU `__del__` auto-frees memory — no manual `.free()` needed.

**Backend capability TAGS** — `register_backend(name, model=None, *, tags=())` stores a frozenset per `(model, name)`; `backend_tags(name, model)` reads it back and `backends_for_model(model, tag=...)` filters (a collection means ALL-OF; `tag=None` returns exactly the old frozenset).  The vocabulary is small and DECLARED at the registration, never guessed from a name: `reference` (the NumPy oracle the suite compares against), `delegate` (wraps another backend instead of computing: `auto`, `shard`, `integrated`, `integrated_pwa`), `external` (needs a process the test did not start: e.g. the `mpi` ranks), `n_res1` (REFUSES a multi-BW chain — each one verified against its wrapper's guard).  Test code selects by capability, so `tests/_flat_compute.py` derives the `pwa_gen` leaves from the registry and `tag=` instead of a name list, and a backend registered later is covered without editing the suite.

**`backend: auto`** is a real, PER-MODEL registered backend and each model's class OWNS its rule — `AutoFlatPwaGenBackend` (`pwa_gen`) and `AutoPWABackend` (`pwa`), both `AutoBackend` subclasses that override `choose(kernel_config)`; there is NO shared selection table and no generic dispatcher (class policy: the policy lives with the model).  A model with no `auto` registration is not eligible and the registry's own `allowed: [...]` error is the answer.  `pwa_gen` checks SHAPE first (`flat_shape(kc)`, mirrored from the s0e wrapper): `n_res != 1` → `cuda_flat_pwa_gen2` (s0e/s0c raise on multi-BW chains), a `tile_bytes(n_wave, n_basis, n_proj)` above the device's shared-memory opt-in → `cuda_flat_pwa_gen_s0c`; then availability: no CUDA → `numpy_flat`, otherwise `cuda_flat_pwa_gen_s0e`.  `pwa` has NO shape fork and no preference order: no CUDA → `numpy_pwa`, otherwise `cuda_v4_pwa` — the model's registered default and the only UNCONDITIONAL member (the cache variant needs fixed m0/g0 and raises otherwise; s0d/s0e/1bw need `n_res == 1`, which `tests/config_pwa.yml` HAS — the kernels divide by `n_wave = P·N`, not by `len(all_comb)` — so those do construct there, and what rules them out is the sample-size ordering `auto` cannot see, plus `cuda_v5_pwa_s0e`'s 8.7 M segfault); `resolution_size > 1` → `cuda_v5_pwa` (`cpu_v5_pwa` without CUDA), because the v4 backend and the NumPy reference have no grouped objective and raise `TypeError` on that kwarg (it is a CONSTRUCTOR kwarg, read from `**kw`), while `pwa_gen` needs no such fork — its whole family takes `resolution_size`.  The choice is AVAILABILITY ONLY — no memory rule and no shared constant, because the sample size is unknown when the backend is built: `cuda_state()` (ONE cached `nvidia-smi --query-gpu=memory.free,memory.total,compute_cap` call, **no deadline**, and a failure is not read as "no GPU": with a GPU device node present (`/dev/nvidia*`) it RAISES so `auto` cannot downshift a GPU box to a CPU backend, while no device node makes the same failure the FACT `has_cuda=False`; the same rule and the same helper are used by `cuda/build`'s arch probe, so the arch flags and the build KEY cannot disagree, `nounits` is **MiB** so the values are converted, plus `find_nvcc()` and the `smem_optin_bytes` CC table — sm_70/75 96 KiB, sm_80/86/89/120 99 KiB, sm_90 227 KiB, unknown 96 KiB; a wrong value only costs performance because the s0e `create` still refuses loudly), `flat_shape()` and `tile_bytes()` (`gps0_layout`'s `te·(2W + 5N + 4P + 3) + 4N + 2NB + 3` doubles, `te` from `gps0_pick_te`, an upper bound since the C side may shrink further) are shared FACTS, not a rule; `free_bytes`/`total_bytes` are reported by `describe()` and decide nothing.  Resolution is EAGER in `__init__` from the host state + the kernel config — never from the data: a config may load several datasets and `ne` is unknown when the backend is built — and every call forwards to `self._delegate`, the concrete backend named in `self.chosen`; `describe()` prints `auto -> <chosen> (free …, cc …, n_wave/n_res/n_basis)` and `__getattr__` forwards the delegate's attributes (`resolution_size`, `kernel_stats`, …).  A `choose` result is validated against `backends_for_model(MODEL)` at construction — a stale name raises instead of degrading to another backend.  `config.dic["backend"]` keeps `"auto"`; the concrete delegate is what the wrapper holds, named in `fitter.backend.chosen` and printed by `describe()`.  An optional `n_events` (kwarg, or the config's `data.n_events`) switches the flat rule from availability to a MEASURED ordering: above `LARGE_SAMPLE` (2e5 — where the per-item rate is flat within 2 %, measured 3.1 Mev/s at 2 k rising to 38.9 at 8.7 M) the leaves that already passed the shape guards are ordered by `FLAT_RATE_MEV_S`, keyed by `(cc, n_wave, n_basis, n_proj)` — the rate is a property of the KERNEL CONFIG and the device, not of the event count, so a shape or device that was never measured gets NO ordering (availability stands) and a small or unknown sample keeps today's answer.  Measured on the full 8.7 M sample, Mev/s: `s0e2` 34.0, `s0e` 33.2, `s0e3` 32.7, `1bw` 24.7, `s0c` 18.3, `s0c2` 17.8, `gen2` 6.55 — with their memory side (854, 854, 854, 1146, 446, 446, 1106 B/event): `1bw`/`gen2` are the memory hogs and `s0c` the light one, so speed alone is not enough at larger samples.

**`cuda_flat_pwa_gen_s0c2`** (case-1 copy of `s0c`) fuses the THREE per-chunk feature reductions (`gck_r`, `gck_i`, `gco`) into ONE launch (`grid = dim3(n_chunks, 3)`, buffer by `blockIdx.y`), so a compute is **8 launches → 6** (10 → 8 with the grouped objective).  Bit-identical to s0c (`np.array_equal` on Q/P/dck/dcoeff/dnorm, and a C-level `memcmp` of fused vs the three separate launches) — each slice keeps s0c's block shape, chunking, fold order and atomic, and the slices never share a destination.  MEASURED (idle GPU, interleaved min-of-5): **6.9 % at ne 4000** on the km fixture (129.5 → 121.1 µs at `norm=None`, 225.6 → 210.7 with `norm`), 1.3 % at 20 k, 0.1 % at 50 k, and **0.0 % on the production shape** — `n_basis = 64 > 32` takes the *acc* fallback, which stays as s0c's three launches BY DESIGN (fusing it needs a second fused kernel with `grid = dim3(max_nf, 3)`; that is the scoped follow-up, and the one that would move production).  Two corrections this measurement forced: ncu's 16.25 µs/launch was ~4× the wall-clock cost (~4.2 µs), so the "21 % of the iteration" the profile suggested is really ~8 µs; and s0c is not bit-reproducible against ITSELF at ne 4000 (cross-block `atomicAdd` in the Q reduce and `grad_ck` gives a 3.7e-9 Q spread), so the bitwise gate is at 64 events.  sm_70/sm_120 compile with 0 spills (amp 76/50 registers).  Harness: `tests/benchmark_s0c_variants.py`.

Selection rules for the s0 family: `s0e` is fastest below `ne ≈ 2.7 M` and `1bw` above it, but 1bw's fixed per-call cost is paid every BFGS iteration, so it is a handicap on small samples.  `cuda_v5_pwa_s0e` SEGFAULTS on the 8.7 M-event sample and is not usable for production fits at that size.  Benchmark: `tests/benchmark_flat_vs_pwa.py` (`--rsize` for the grouped objective).

### MPI backend (`mpi`) — co-located ranks, one slice each

`mpi` (`tabpwa/backends/mpi_backend.py`, tag `delegate`, universal) turns N
processes launched by `mpiexec` into one backend: every rank is a full tabpwa
process holding ONE contiguous slice (`shard_backend.split_offsets`, weighted
and `align`-aware), `compute` runs on that slice and the result is reduced —
`Q`, every gradient block and `grads["norm"]` by `allreduce(SUM)`, the
per-event `P` by `allgather` ONLY when `return_p` (the fitter never asks; a
gather instead of an allgather desynchronises a `generate_toy` driven through
the MPI fitter — measured hang).

Launch/binding: node-local rank (`COMM_TYPE_SHARED`) selects
`CUDA_VISIBLE_DEVICES` before the local backend is built.  A LAUNCHER-set value
is respected: empty = hide every device, one entry = this task's own
(`--gpus-per-task=1`), several entries = a node-level list (`--gres=gpu:N`)
indexed by the node-local rank (otherwise every rank would sit on device 0).
`device=None` disables binding, an int binds that device everywhere; a rank
whose node-local index exceeds the visible GPU count raises naming `-n` and the
count (a host with no NVIDIA device is left unbound).  Slurm deployment
(`srun` vs `mpiexec`, PMI/PMIx, `--gpus-per-task`, signals):
`docs/slurm.md`.  The FIRST build is serialised (rank 0 builds, barrier)
so N ranks sharing an install do not compile one kernel concurrently.  `shard`
inside a rank is refused (fork after `MPI_Init` + CUDA), and an `mpi` backend
inside a `shard` worker is refused through the marker the worker sets.  The
FIRST `compute` of a process `bcast`s the parameter dict from rank 0 —
`initial_values()` is unseeded per process, so a file-driven `run_fit --fit`
would otherwise fit a different point on every rank.  Every rank must call
`compute` the same number of times (`allreduce` is collective), which is why all
ranks run the same deterministic fit.  EVERY rank-local failure site (device
check, build/barrier, `load_data`, the reduction) replays its exception through
`comm.Abort`, because an unguarded raise hangs the peers in a collective
(measured 45–60 s versus 1.2 s); an empty slice is refused by name (the `pwa`
kernels cannot load one, and `split_offsets` can produce one via `align` or
skewed weights).  The PACKAGE stays transport-agnostic: no module under
`tabpwa/` knows a rank, a launcher variable or a writer role — the MPI facts
(rank, size) live inside `mpi_backend` only.  Output ownership is therefore the
PROGRAM's: `run_fit.py` (a CLI, not the library) asks `fitter.backend.rank` and
writes on rank 0 only, while a LIBRARY program under `mpiexec` writes once per
rank and prints N times.  A program that should not deal with rank at all needs
the driver/worker launch (`docs/mpi_backend_design.md` §11.6), not a
package-level rank concept.

`local` MUST be ONE concrete kernel (a leaf): a delegate/composer (`auto`,
`integrated*`, `fixed_coeff`, `shard`, `mpi`) is refused by the `delegate` tag,
because `auto` may resolve differently per rank and a wrapper's per-handle state
is not forwarded.  `local=None` resolves the model's `default`.

Sample GENERATION is single-process: `Fitter.generate_toy`/`generate_phsp` drive
the backend per batch, so on an MPI fitter they are a collective (lockstep, N×
work).  Generate once with a serial fitter / a file, then `set_phsp`/`set_data`
the arrays.  Optional extra `tabpwa[mpi]` (mpi4py), imported lazily with the
extra named.  No rank-aware loader yet: every rank holds the full buffer before
slicing (`N×` peak) — the tracked follow-up.  Design
`docs/mpi_backend_design.md`, as built and measured `docs/mpi_backend.md`;
fixture `tests/test_mpi_backend.py` (14 gates; the SEVEN `mpiexec`
ones skip without a launcher).

### MPI service (`mpi_service`) — the program runs once, MPI in a child

`mpi_service` (`tabpwa/backends/mpi_service_backend.py`, tags `delegate` +
`external`) is the DRIVER half of `docs/mpi_service_design.md`: it imports NO
`mpi4py` and reads no rank.  On the first `load_data` it starts
`mpiexec -n N python -m tabpwa.mpi_worker --local … --addr <unix-socket>`,
accepts rank 0 on that socket and talks the frames of `tabpwa/_ipc.py`.
Stdio cannot carry the protocol (Hydra muxes the child's stdin — measured), and
a `socketpair` fd is not inherited through the launcher, hence the socket PATH in
argv.  The worker's loop bcasts each op and calls the ORDINARY `mpi` backend on
every rank, so slicing, device binding, the build barrier and the reduction exist
once.  `n_procs` is REQUIRED; `local` is a leaf; the constructor is offline and
REFUSES to start inside an MPI/scheduler step (that case is `srun` + the
all-ranks `mpi`).  Lifecycle: lazy start, `shutdown` → `terminate` → `kill`,
`atexit` reaper, bounded stderr tail appended to errors, socket dir removed.
`ship` loading only in v1 (the reference path is open); the `N×` transient is
the same follow-up as the all-ranks mode.  As built: `docs/mpi_service.md`;
fixture `tests/test_mpi_service.py` (8 gates, incl. "the parent imports no
mpi4py and has no rank"; only the gates that START a child need a launcher, so
the boundary rules run on a box without MPI).

### Remote compute (`remote`) — DRAFT STANDARD, not implemented

**STATUS: draft.**  The normative text is `docs/remote_protocol_design.md` (transport,
content key, endpoints, distribution, open questions).  NO implementation is in
this tree: a prototype was built and measured during the design review and is
kept out (stashed) so the standard can be decided first — see §16 of that
document.  Do not build on it before §14 is decided (the encoding is the
main question).  For the CO-LOCATED multi-GPU case use the implemented `mpi` backend
above (`docs/mpi_backend.md`).

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

**Bounds come in three SHAPES, and the shape is what the transform is.**
`set_range(name, lo, hi)` (config `var_range: {name: [lo, hi]}`) is the
two-sided sine; `set_range(name, lo, None)` / `[lo, null]` is a one-sided LOWER
bound, `set_range(name, None, hi)` the upper mirror, both with the UNIT-SCALED
map `y = lo + sqrt(x²+1) − 1` (branch `x ≥ 0`; upper: `y = hi − sqrt(x²+1) + 1`
on `x ≤ 0`).  No per-parameter `scale` exists and none may be added: a scale
would move the point where the map becomes linear, i.e. make the optimizer's
step unit parameter-dependent, which is what `gtol`/`eps` assume is fixed —
without it `|x| ≈ 1` is the quadratic region for every parameter and
`dy/dx → 1` in the bulk, so a parameter that wants to be large is not compressed
at all (the two-sided sine cannot do that: its branch is `±kπ/2`, half the range
width).  The open side is carried as `±inf` in `a`/`b`, so the code that clips a
value or a step inside a range needs no special case, and the saved JSON writes
`null`.  `cm.initial_values` draws a bounded slot ON the branch (rescaled from
`U(-π,π)`; `|u|/π` for a half-line), and a `default` outside its range WARNS: the
requested value is unreachable (the default and the range disagree — a config
error), so the start is placed just INSIDE the edge instead, at the nearest
point where `|dy/dx| = START_GRAD = 0.1` (`boundary.py`).  That inset is what
keeps a start usable: exactly on a wall `dNLL/dx = 0` with `F'' = |g_y|/k > 0`,
so a minimizer reads "converged" at iteration 0 — a wrong answer when the
optimum is INTERIOR.  Every shape is quadratically flat at its wall, so the
inset costs only 0.25 % of a range's width in `y` (5.0e-3 for a half-line)
while keeping 10 % of the y-gradient in `x`, far above any `gtol`.  The wall
itself is untouched — `forward`/`grad`/`inverse` keep their exact meaning — so
a fit genuinely pulled outwards still converges ONTO the edge.

A BINDING soft range is REPORTED, not repaired: the edge of the sine IS the
bound and it sits at a FINITE `x = ±kπ/2` (the arctan needed `1e15`), so
"at the boundary" is `value ≈ a|b` within a share of the width —
`Fitter.WALL_TOL = 1e-3`, since the delta σ is already ~7× wrong there —
`Fitter.boundary_report(result)` lists those slots with their edge, and
`fit()` / `get_uncertainties()` raise a `RuntimeWarning` naming them.  No sigma
exists at a wall — the transform's limit, not the model's: `dy/dx = 0` at the
edge, so the delta σ collapses and `H_xx` there (`= |g_y|/k`) is set by the pull,
not by the likelihood — while the LOCATION and every OTHER parameter stay
correct, because the bound map is per-slot diagonal.  The map is the PLAIN sine
(`x` is never clamped): at the edge `y' = 0`, `y'' = −1/k`, so `F = NLL(y(x))`
has `F'' = |g_y|/k > 0` and the edge is a proper, unique, positively curved
minimum in `x` — MEASURED on a binding range: `x* = 7.4539` against the branch
end `7.4524`, `y* − b = −2.3e-7`, `H_xx = +0.129`, converged.  A clamp would pin
everything past the edge to one value (`dNLL/dx = 0` on a half-line, a
non-unique minimiser, `H_xx = 0`) — the failure this shape avoids, and the reason
the fit is not "pinned" at the edge.  The remedy for a binding range is to
widen/remove it and refit, or to read a one-sided interval from the profile
likelihood; the code warns rather than masking a sub-block.

**Two curvature frames, and only one of them is `hess_inv`.**
`compute_numerical_hessian(x)` defaults to `space='x'` (the optimizer frame:
`inv` of it is the `hess_inv` that `fwd`/`cal_uncertainties`/`profile` chain
through `dx`), and `space='bounded'` differences the AFTER-BOUNDS value
`y = bound(x)` of every free slot instead — `d²NLL/dy²`, with the step taken in
`y` units and clipped inside the range.  It is not a transformed matrix: the
pipeline takes a START CUT, `cm.resolve(..., start_after='bounds')` /
`cm.full_gradient(..., start_after='bounds')` (the mirror of `stop_after`), so
`Fitter.get_nll(y, start_after='bounds')` returns `dNLL/dy` directly and the
shared `fd_hess` (per-slot steps) produces the matrix.  The `d²x/dy²` term comes
out of that by construction; `start_after` is part of the `build_params` cache
key, `get_nll` does not let a `y` call overwrite `_last_xk` (the interrupt
checkpoint is an x vector), and an out-of-range `y` raises instead of being
extrapolated.
`get_uncertainties(space='bounded', use_cached=False)` inverts THAT and reports
the physical sigmas directly (no `|dy/dx|` scaling), which is the only route
that survives a parameter near or at a range edge: there `dy/dx → 0`, so the
x-space route is mis-scaled and drops the `g·d²x/dy²` term.  Measured on the
sine bound (range width 9.49, branch `±7.45`): at mid-range the two frames agree
exactly (ratio 1.000); 1 % from the edge the delta σ is 0.41× the bounded one,
0.1 % → 0.15×, 0.01 % → 0.048× (0.056 against 1.17).  The sine also keeps
`cond(H_x)` sane (1.8e2 there, against 1.1e11 for the arctan it replaced).  A
BINDING soft range converges ONTO the edge (`x* = 7.4539` against the branch end
`7.4524`, `y* − b = −2.3e-7`, `H_xx = +0.129` — PD, because `F'' = |g_y|/k`), but
no σ is defined there: `dy/dx = 0`, so the delta σ collapses and the bounded
route refuses within `_HESS_WALL_TOL = 1e-6` of the width, naming the profile
likelihood as the one-sided answer.  A
bounded-space matrix is returned under its OWN name —
`unc, res.hess_inv_bounded = get_uncertainties(..., return_hess_inv=True,
space='bounded')`, the attribute that `space='bounded'` reads back (assigning it
to `res.hess_inv` is wrong: that one is chained through `dx`); it is also
persisted as its OWN file — `save_params(..., bounded_errors=True)` writes
`*_error_matrix_bounded.npy` BESIDE the x-space `*_error_matrix.npy` (a file name
is the tag a reader sees), `load_results` restores it as
`res.hess_inv_bounded`, and `get_uncertainties(space='bounded')` reuses it
instead of recomputing (`use_cached=False` forces the `2·N` evaluations).  The
JSON states the frame of every block — `error_frame: "value_bounded"`,
`hess_inv_frame: "x"`, `hess_inv_bounded_frame` when the second file exists —
because the `error` dict was ALWAYS after-bounds while the matrix was x, and
nothing said so.  A parameter sitting exactly ON a bound has no finite step and
raises — a one-sided interval is the profile likelihood's job.

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
Nothing internal consumes the report frame.  So every derivative here is taken around the **model** frame (`fd_jacobian`, `cal_uncertainties_multi`, `fwd_of`'s default).

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

The mask memo is cleared per evaluation: a stale memo silently returns zero errors.

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
`rows(y, J).sigma(hinv)` is the correct error.  It costs `2·n_var` evaluations and returns
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

**Packaging: sources, never binaries.**  The wheel and the sdist carry
`cuda/*.cu` / `cpu/*.c` and no `.so` and no `.hash`.  The build key hashes the
SOURCE and the target arch + nvcc identity, so a prebuilt binary cannot be
validated on another machine (`ensure` refuses without a matching hash) and the
wheel is `py3-none-any`, which cannot legally hold an arch-specific library; a
`.hash` without its `.so` is never consulted.  Prebuilt binaries are the
AppImage's job (`appimag/build.sh` fat-builds `sm_70,sm_75,sm_86,sm_89`); a pip
prebuilt channel would need separate platform wheels and a key that does not
include the builder's nvcc.

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

The probe has **no deadline**, and a failure is not read as "no GPU": with a
device node present (`/dev/nvidia*`) an unusable `nvidia-smi` RAISES naming
`--arch`, because falling through to the default archs would build for an SM
the box may not have — and `_arch_signature` (part of the build key) calls the
SAME probe, so a native binary cannot be accepted under `auto:unknown`. A
missing binary with no device node is a CPU-only box: the default stands. Of
several devices the FIRST is targeted (a newline from a multi-device query used
to land inside the `-arch` flag).

`--arch` is **validated** against `nvcc --list-gpu-arch`: CUDA 13 dropped
`compute_70`, and asking for it raises instead of failing every kernel
quietly.  The arch is part of the build key, so a native `sm_120` binary is
never mistaken for a fat `sm_70,sm_86` one.

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

**Per-arch extra flags (env, arch-keyed).**  `TABPWA_NVCC_FLAGS` is appended to every kernel build; `TABPWA_NVCC_FLAGS_SM120` (or `_SM_120`) only when that SM is targeted, so one tree can carry a V100 recipe beside a consumer one WITHOUT invalidating the other's binaries (a non-targeted arch var does not change the build key).  `TABPWA_CUDA_PTX=1` emits `code=[sm_X,compute_X]` instead of SASS only, so one binary survives an SM bump by JIT.  All of it is mixed into the build key ONLY when set, so changing a flag REBUILDS while the default build keeps the historical key and no existing binary is invalidated.  There is no magic flag: the useful ones are register budgeting (`-maxrregcount` / `__launch_bounds__`) and `-Xptxas -O3` (the default) — pick them from a profile.  Measured on the s0e tile: the 2-chain ILP split took it 48 → 56 registers and 80 → 63 % occupancy (2–4 % slower); `__launch_bounds__(GPS02_BLOCK, GPS02_MIN_BLOCKS)` restored 48 and parity.  That second argument is a MACRO (`-DGPS02_MIN_BLOCKS=N`, or an `#if __CUDA_ARCH__` block when one fat binary must differ per cubin) because the register budget is per arch: measured on sm_120 with 0 spills, N = 4/5/6 → 64/48/40 registers → 32/40/48 warps per SM (67/83/100 %).  Combine it with the arch-keyed env hook, e.g. `TABPWA_NVCC_FLAGS_SM120=-DGPS02_MIN_BLOCKS=6`, to A/B a build.  A V100 (`sm_70`) needs a CUDA 12 toolkit — CUDA 13 lists only `compute_75+` — and compiles every kernel in this tree when given one.

**Reading the register/occupancy numbers at RUNTIME.**  CUDA exposes the compiled resource usage per FUNCTION — no device-code intrinsic exists, but `cudaFuncGetAttributes` gives `numRegs`, `localSizeBytes` (non-zero ⇒ the kernel SPILLS, which is exactly how an over-tight `__launch_bounds__` fails) and `maxThreadsPerBlock`, and `cudaOccupancyMaxActiveBlocksPerMultiprocessor` gives blocks/SM for the real block size and dynamic shared memory; `-Xptxas -v` / `cuobjdump -res-usage` are the build-time equivalents.  `kernel_stats()` on the s0e2/s0e3 wrappers wraps all of it, and `tests/benchmark_s0e_variants.py` prints it for every variant it builds (`regs`, `local B`, `blocks/SM`, `warps/SM`, `SPILLS`), so the A/B needs no profiler to explain a difference.  Measured this way: s0e2 tile 48 regs / 0 local / 5 blocks / 40 warps, s0e3 the same; the per-arch spill table in `kernels_flat_pwa_gen_s0e2.cu` (N=4 clean everywhere, N=5 only on sm_75/sm_120) was produced with `-Xptxas -v`.

**s0e optimisation experiments (measured, IDLE GPU).**  Production shape (`config_kn4_plot.yml`: n_wave 26, n_basis 64, n_proj 2, 130 CSR non-zeros = 5/wave `{1:5, 3:11, 9:8, 10:2}`, te 8), ne 50 k, rsize 17, interleaved min-of-5 in one process, ns/event: **s0e** 25.70 (norm None) / 27.91 (norm 1.3); **`s0e2`** (packed 64-bit CSR load, 2-chain fp64 dot, arch-aware `__launch_bounds__`) **25.20 / 27.45 = ~2 %, the only non-regression**; **`s0e3`** (fp32 seed + 2 fp64 Newton steps for `1/|D|²`, numerically ≤8.3e-16 vs s0e) 26.17 / 28.44 = ~2 % slower; **`s0e4`** (host-gathered `cw` in CSR order + rows longer than `split_nnz` split across two threads, phase A → A2) 26.96 / 29.33 = ~5 % slower, because `8 events x 36 segments = 288 > 256` threads keeps the critical path at 10 non-zeros while every `(e,k)` pays an extra shared atomic + the A2 pass (the threshold sweep T6 ≈ T9 confirms the segment count was not the cost).  CONCLUSION: the tile is fp64-throughput bound (~80 % occupancy, fp64 = 57 % of pipe cycles) and none of the three structural levers moves it beyond noise; all four are registered as case-1 alternatives and NONE is a default.  All four tile kernels compile for `sm_70` (cu12) with 0 spills — 128 / 64 / 128 / 128 registers — so the V100 path is open.  Harness: `tests/benchmark_s0e_variants.py` (it also prints `kernel_stats()` per variant).

**`s0e5`** (case-1 copy of `s0e2`: explicit FMA, `|ck|²` hoisted to n_wave scalars per iteration, ONE `1/(a²+b²)` shared by the forward and backward path, and the **DENOMINATOR DEDUP** — waves keyed by `(gamma pattern, mass columns)`: **26 → 18 keys**, 130 → 74 non-zero pairs/event on the production config) is **NEUTRAL**: 232.9 ms / 37.55 Mev/s vs `s0e2`'s 232.0 / 37.70 (data pass), `get_nll` 256.9 / 39.83 vs 257.2 / 39.80.  Not a default; kept because the dedup is EXACT (a key's dot is the same fp64 sequence) and pays on wave-heavy configs where the key count is a small fraction.  MEASURED reason it does not pay here: the tile is NOT fp64-instruction-count bound — the FMA rewrite cut fp64 instructions 4.4 % and removed 56 of 130 FMAs/event with no wall-time change.  What dominates is the **barrier (straggler) wait**: 50.8 % of the 74–78 cycles between issues (52.1 % in s0e5), and dropping one of the six `__syncthreads()` (the `s0e6` experiment, 242.2 ms / 36.11 Mev/s, +4 %, not kept) moved it only to 51.3 % — the wait is spread over the per-chunk pipeline (`te` 8, 256 threads, 26 waves).  `DADD` is 79–85 % of the fp64 thread instructions (the 32-lane shuffle reductions), which no fusion removes; 18–19 of 32 lanes are active.  A `one warp per wave` phase C was measured and REJECTED (449.8 ms / 19.44 Mev/s, 1.94× worse) — 8 of 32 lanes own an event at `te = 8`.

One `ensure()` costs **two** nvcc invocations: a shared compiler-compat probe (cached per process, invalidated by `set_arch`) and the kernel itself.

## Class policy

**A feature belongs in the subclass that needs it.  Do not modify a working
parent class to support one subclass.**

A base class is shared by every subclass, so a branch, flag or `Optional` hook
added for one of them is read by all of them, makes the base's own behaviour
invisible from its code, and becomes permanent — removing it later breaks the
subclass that relied on it.  This is the kernel policy below, applied one level
up to the Python side of the same hierarchy.

1. **Override, don't flag.**  `PWAGen` needs a different transform list, so it
   writes its own `particle_transforms()`; the base keeps `pwa`'s.
2. **A hook is a contract, not a placeholder.**  `extra_transforms()` /
   `constant_weight_values()` live on `PWAGen`, beside the method that calls
   them.
3. **The parent changes only** for (a) a defect affecting every subclass, or
   (b) a refactor that leaves every subclass's behaviour identical —
   *verified*, not asserted: same free set / inputs / outputs and `dQ = 0` on
   every fixture before it lands.
4. **No `isinstance`/name test on your own subclasses** in a shared path.  If a
   shared caller must know which kind it has, the distinction belongs in the
   interface (e.g. `BasisModel.make_basis_transform() -> None`), not in the
   caller.  Asking which of two SEPARATE interfaces an object implements
   (`isinstance(mdl, ParticleBaseModel)`) is not this case.
5. **No model-specific knowledge in the `Fitter`.**  `"one"`, the m² constant,
   is `pwa_gen`'s constant and lives in `PWAGen` together with the transform
   that produces it.

## Kernel policy

**Never modify a working kernel.**  A kernel that works (tests green, in
use) is immutable: no edits, no signature changes, no "small" tweaks.
The only allowed actions are:

1. **New copy** — implement a new standalone kernel: add a new C source
   (`.cu`) by copying the existing kernel's source into it and
   **adapting** the content to the new variant; mirror the Python wrapper
   module (`_<name>.py`) the same way.
   What must be NEW is the artefact's IDENTITY: its own source path,
   its own `.so` name in `ensure(...)` (a shared name would overwrite the
   original binary), its own module/class/registry name, and its own build
   entry.  **A copy is made WITHOUT RENAMING**: the new `.cu` is the
   parent's source verbatim apart from the behavioural change, keeping every
   identifier (`gp2_*`, `GP_*`, `GPLayout`, and the exported C API names).
   That is what keeps the copy reviewable — its diff is the algorithm and
   nothing else.  No rename is needed for it: the kernels are loaded with
   `cffi.ffi.dlopen` (`RTLD_NOW | RTLD_LOCAL`) and each is its own `.so`, so
   an identical `gp2_*` / `cuda_*_flat_pwa_gen2` name in a second library
   cannot be interposed.  The distinct prefixes the existing kernels carry
   (`gp2_`, `gps0_`, …) are history, not a convention — do not imitate them
   in a copy (a kernel written from scratch is free to name its identifiers
   however it likes; the rule is about COPIES).  If two kernels ever had to
   live in ONE object or be loaded `RTLD_GLOBAL` (neither happens today; an
   explicit flag, `LD_PRELOAD` or a plugin would be the cause), fix THAT —
   separate the object, drop the global load — never rename.
   The copy links beside the old kernel — every existing kernel file
   and its tests stay byte-identical.  **A copy is STANDALONE: it must not
   import another backend's module.**  The build-time helpers it needs
   (`uslot_map`, `amp_slot_map`, `index_runs`) travel with it as verbatim
   copies (extracted with `ast`, pinned by an `np.array_equal` test against
   the originals).  `cuda_flat_pwa_gen_s0c` predates this rule and still
   imports them from `s0e`/`1bw` — flagged, untouched; fixing it is a case-1
   replacement's job, never an edit.
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

A comment inside a pinned kernel is not an exception, and some of them now name
a test module the suite reorganisation deleted (a `.cu` header citing its old
per-kernel fixture), or claim the copy is made "under its own new symbol names"
(the wording this policy used before it was corrected to "a copy is made WITHOUT
RENAMING").  Editing such a comment would change the source digest the case-1
table pins, i.e. it would break the very guard that proves the kernel was not
touched, so the stale text stays: read it as provenance, not as a rule.
Every file that carried it has now been corrected as its digest moved anyway
(`s0c`/`s0c2` with the zero-weight repair, `s0f`/`cache`/`gen2`/`1bw`/`fslot`
with the group-weight mask, and `kernels_flat_pwa_gen.cu`, which has no pin at
all — it is the parent, not a copy).  No stale text is left in a frozen kernel;
`kernels_v5_pwa_s0cw.cu` is edited for the same repair and is pinned by NO
digest — its contract is gated numerically by `tests/test_pwa_v5_s0cw.py`.

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

## Plugins (`plugin:` in the config)

A plugin is a Python MODULE whose import registers what a config names — a
backend, an amplitude model, a particle model.  The config declares them at the
top level::

    plugin:
      - tabpwa_mpi        # would register `mpi` / `mpi_service`
      - my_lab.models     # registers @register_model / @register_amplitude_model

`Fitter.__init__` imports every entry (also accepted as `plugins:`, and as a
bare string) immediately after `RawConfig` — BEFORE `build_amplitude_model` and
before `resolve_backend_spec` — so the registrations are visible to both the
model builder and the backend resolver, and the calling script needs no
`import`.  There is no entry-point discovery and none is needed: an explicit
import is the mechanism.  `tabpwa.plugins.load_plugins(spec, where=…)` is the
same function for a caller that builds a `BaseModel` directly.  The import is
idempotent (`sys.modules`); a failure raises `ImportError` naming the plugin and
the config.  A config is trusted input: importing a plugin executes code,
exactly as a Python entry point would.

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
  ampcache machinery, written before the no-rename rule so its symbols carry the
  `..._polarised` prefix — history, not a convention; existing
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
`grads["coeff"]` back to the basis *weight* names, and each resonance's own
basis transform — `PWAGen.extra_transforms()` returns exactly the
`_pg_bws[i]["transform"]` list, registered by
`Fitter.setup_mass_width_transforms` as mass/width-style transforms — maps the
weights to/from the physical parameters (the `make_basis_transform` contract;
`PlainBasisWeightsTransform` carries its own EXACT VJP — `s0 = m0²` and
`{g}_t = m0·g` are differentiated in closed form, not by finite differences).  A **combined** wrapper is unnecessary: the constraint manager
already unions the list (inputs added, outputs removed, constants pinned),
de-duplicates by structural signature and chains in registration order, so
`extra_transforms` hands the per-model transforms over as-is.  A model whose
`make_basis_transform()` returns `None` contributes nothing — its weights ARE
the parameters, which is a legitimate model, not an error.

Backends: `numpy_flat` (`default` for `pwa_gen`) — the flat config's
reference implementation.  The GPU siblings `cuda_flat_pwa_gen` and
`cuda_flat_cache_pwa_gen` (cached `F`/basis/`fa`) read the same config.  A
`pwa_gen` config carries none of the `den_row`/`n_index`/`bw_order`/`matrix_angle`
keys, so only backends that read the flat config are registered.
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
A group is AVAILABLE when `|W_g| > 1e-12` — an ABSOLUTE epsilon, because a
group whose weights cancel can sum to a few 1e-17 that is not exactly `0` — and
the mask covers the ratio `S_g/W_g` — the only division by the TOTAL — so a
numerically-zero total never divides (the scale `-W_g/S_g` divides by the group
SUM `S_g` instead: `S_g = 0` is a separate degenerate case, ±inf, pre-existing
and reachable only with `P = 0` and `bkg = 0`).  The SIGN of `W_g` is part of the
formula, so a negative total is computed rather than dropped (`Q(-w) = -Q(w)`,
`dQ/dP(-w) = -dQ/dP(w)`), and the weights need not share a sign — a signed
(sWeight) bin is a legitimate group — but its MEAN PDF must come out positive:
a group whose `S_g/W_g <= 0` has no log, and the kernels return NaN for it with
no diagnostic (the reference warns once, through numpy).
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

Every registered particle model must subclass `BasisModel` (`register_model` raises otherwise), so the basis protocol is always available.  **Two PARALLEL particle interfaces, `BaseModel(BasisModel)`.  `BasisModel` is the INTERFACE** — `num_basis` / `den_basis` / `get_num_name` / `get_den_name` / `num_slot_names` / `make_basis_transform` (default `None`) / `get_defaults` (default `{}`) / `amp_raw` / `amp` — and it knows nothing about running widths, gamma rows, mass or width.  `den_basis` and `get_den_name` RAISE: a basis cannot invent its own denominator; the denominator WEIGHTS are simply the resolved dict's entries read by name (the model's transform outputs, or its parameters when it has no transform) and the numerator likewise — the transform writes `head{mag}`/`head{phase}`, the amplitude reads them.  There is no weight function on either side.  **`BaseModel(BasisModel)`** adds the other interface (`gamma` / `get_gamma_name` / `get_mass_name` / `get_gamma_count` / `get_defaults` / `get_bw_params` / `get_bw_param_names` / `make_mass_width_transform`) which the plain `pwa` family reads, and is also the interface's DEFAULT implementation: its `den_basis` composes `[1, −m², −i·γ…]` from its OWN `gamma`, so the delegation is the subclass's business, never the interface reaching into a sibling.  The two have SEPARATE evaluation entries — `BasisModel.amp_raw` is the basis ratio at the parameters as given and `amp` is that composed with its BASIS transform; `BaseModel.amplitude_raw` states the denominator through the gamma names at the parameters as given and `amplitude` is that composed with its MASS/WIDTH transform — so neither calls the other's path.  `amp` must be a real method, NOT `amp = amp_raw`: a class-level alias binds the function, so a subclass overriding `amp_raw` silently keeps the stale base implementation.  **`kmatrix_n`** is the K-matrix family (its only classes: `kmatrix_n`, `kmatrix_n_gs`).  Its `ρ` is `bwr_model.phase_space` = `(q/m)·q^{2L}·B'²`, M-FREE (`phase_space(m)/phase_space(M) ≡ running_width(m, M)` exactly, so the coupling carries the scale; no `m_π`-based threshold anywhere).  ONE class whose pole content is a LIST — `mass_list: [m1, m2, …]` / `width_list: [Γ1, Γ2, …]`, so `n_pole = len(mass_list)`; BOTH lists are required and equal length, `n_pole` is not a key at all, and there is no scalar `width` shorthand (poles are different resonances — there is no total to split).  The optional `mass` is only the GS reference `M`, default `mass_list[0]`.  The fitted parameters are `{name}_mass{i}` / `{name}_width{i}` with `i = 0 … N−1` (the `mass_list`/`width_list` entries are their defaults) plus the `{name}_beta{i}{mag,phase}` numerator pairs with `i = 1 … N−1`, which have NO config key and NO default — a coupling is linear in the amplitude and zero at the origin, so like `g_ls` it starts random (`U(-π, π)`) and is set by `--init` / the constraint file (magnitude/phase suffixes from `complex_tail`).  `beta0 ≡ 1` is the GAUGE head: the numerator's overall complex scale is unobservable (`P → |c|²P`, `norm → |c|²norm`, so `P/norm` is invariant), so fixing it leaves `2N + 2(N−1)` free parameters — the same count as N Breit-Wigners with one overall reference fixed (at N = 1 there is no numerator parameter); `L` defaults to `BasisModel.default_L()` — the lowest partial wave of the particle's own decay, `min(l for l, s in decay[0].get_ls_list())` — one rule shared by `BWR`/`BWRS` and both `kmatrix_n` classes (the class fallback is 0, the GS variants 1, and it applies only when no tree is attached; an explicit `L:` always wins); the scalar `mass` is build-time only (the reference `M`) and is NOT a parameter.  There is NO `g{i}` parameter — the basis transform derives `g_i²` from the width — and an unknown per-pole key is simply ignored, so only `beta{i}` is read for the numerator.  `widths` means WIDTHS — the convention is `m·Γ(m) = m_i·Γ_i·ρ(m)/Re ρ(m_i)` (the FULL, possibly complex `ρ(m)`; only the normalisation at the pole is `Re`), the `−i` coefficient of the propagator, so `Γ(m_i) = Γ_i`; the transform computes that coefficient once, `g_i² = m_i·Γ_i/Re ρ(m_i)` (the `−i` already sits in the basis row, so the absorptive part is `Re ρ`; `Re ρ_gs(m_i) = 1` exactly, so the GS subclass overrides only `ρ`), and feeds it to the polynomial weights without a sqrt round trip.  `mass_list: [m0]` / `width_list: [Γ0]` alone therefore reproduces `BWRS` / `GS_rho` with no manual coupling arithmetic.  `width_norm: false` switches that pole normalisation OFF (`g_i² = m_i·Γ_i`) — for a unit already normalised at the pole, or to drop the `Re ρ(m0)` dependence entirely; the declared width is then reproduced only where `Re ρ(m_i) = 1` (`width_norm: true` is the default; anything that is not a bool raises).  It also makes the basis transform's VJP ANALYTIC (`KMatrixNWeightsTransform.backward` is closed form — every weight is a polynomial in `m_i²` and `Γ_i`, linear in `β` — with no extra `forward` calls).  `width_norm: true` is analytic too: its one non-polynomial piece, `dRe ρ/dm`, is the `KMatrixNModel.drho_dm` hook — a centred difference of `ρ` (two evaluations) by default, or whatever a subclass/instance sets (an exact expression) — not a difference of the whole transform.  Its phase space is the METHOD `_rho` (`bwr_model.phase_space`) — THERE IS NO `rho:` CONFIG OPTION — and the GS variant is the SUBCLASS `kmatrix_n_gs`, which overrides that one method with the complex `gs_unit`; daughters come from the decay tree, else `daug2Mass`/`daug3Mass`.  It emits `2N+1` denominator rows (`[s^k]` + `[−iρ·s^k]`) and `N` numerator heads; the weights are the polynomial coefficients of `Π(m_i²−s)` / `Σ g_i² Π_{j≠i}`.  At N=1 it IS `BWRS` / `GS_rho` (gated) — that anchor is what let the fixed-arity two- and three-pole models be deleted once `kmatrix_n` reproduced them bit for bit; N ≥ 2 are plain instances of the same class.  A pure `BasisModel` — both classes.  **`kmatrix_n_ieps` / `kmatrix_n_gs_ieps`** are case-1 copies (the parent is untouched) that add the `m² − iε` prescription to EVERY pole factor, linearised: `Π_i(m_i²−s+iε) = Π + iε·D + O(ε²)` with `D = −Π' = Σ_i Π_{j≠i}`, so the extra rows are `i·s^k` (k = 0 … N−1) with the DERIVED weights `−ieps·(k+1)·w_{k+1}` (no new parameter — only `ieps`, whose weight slots are transform outputs, never fitted).  It turns a simple real-axis pole (below-threshold mass, zero width) into a finite amplitude; a DEGENERATE pair is a double root where `D = 0`, so the linear term cannot lift it, and `gs_unit` is `nan` below threshold, so the GS variant cannot regulate there.  `ieps: 0` (the default) is bit-identical to `kmatrix_n`.  **`ExpCheb`** is the exponential lineshape `exp(-k(m²-m0²))` as a basis model: rows `T_n(t)` (Chebyshev, `t = ((m_lo²+m_hi²)/2 - m²)/((m_hi²-m_lo²)/2)`) and ONE denominator row (1), with the fitted `(m0, k)` entering ONLY the weights `e^{k(m0²-s_c)}·(1 or 2)·I_n(k·half)` — closed-form (Bessel), analytic in both, no `k`-grid / spline / SVD, and no growing envelope in the rows (which is what forces `gamma_clip` in `ExpSpline`/SVD).  11 rows → 7e-9, 15 → 2e-13, UNIFORM in the fitted `m0` (a drift only changes the scalar prefactor); the end-to-end floor is then the kernel's table interpolation (~`1.4e-3/n_interp²`, 2.8e-7 at the default 2000), so raise `n_interp` through `build_defaults.scope` rather than adding rows.  Config keys: `mass`, `k`, `n_cheb`, `m_range` (default `[2·M_PION, M_B - M_PION]`).  **A basis-only model is refused by the plain-`pwa` path** with a message naming `amp_model: pwa_gen` — the interface consequence, not a flag: `kmatrix_n` has no `gamma` / `get_gamma_name` / `make_mass_width_transform`, and it brings its own basis transform (`KMatrixNWeightsTransform`).  `BaseModel`'s default basis is `[1, -m², -i·γ_j(m)]` with weights `[m0², +1, m0·g_j]` — the m² weight is the SHARED literal `"one"`, fixed to 1, so the NAME READS AS ITS VALUE; `poly` is deliberately NOT used, because it reads like a FITTED coefficient while this weight is a fixed -1.  The pair is self-consistent and reproduces the plain BW denominator `m0² - m² - i·m0·Σ g_j γ_j`, so a plain model, a BasisModel and the pwa_gen kernel share ONE implementation instead of three copies that can drift.  `BaseModel.make_basis_transform()` returns `PlainBasisWeightsTransform` (in `particle_model/base.py`, NOT in `pwa_gen_transform` — it reads `get_gamma_name()`, which only `BaseModel` has), which publishes those weights as transform OUTPUTS.  Models with a non-uniform basis override the transform (`kmatrix_n` does).  `RhoOmegaGS` (10 basis rows) needs no override: the default basis transform holds its coupling stage, so the two paths agree.  `_pg_den_basis` calls `model.den_basis(grid)` and does NOT restate the default: a second copy can drift out of sync with `_den_weights` and silently flip the m² term.

**Pole positions (`solve_pole`)** — a K-matrix pole is NOT a Breit-Wigner peak, so `get_bw_params` (a real running-width peak) is meaningless here: the pole is the zero of `D(m) = P(m²) − i·ρ(m; sheet)·Q(m²)` in the complex MASS plane, where `P`/`Q` are the two real-row weight polynomials (the amplitude's own denominator — `β` cancels, it only normalises the numerator).  `KMatrixNModel.den(m, sheet=1)` evaluates it at a complex mass, and `solve_pole(inits=None, sheet=1, tol=1e-9)` runs a capped Newton from each `mass_list` entry (or from the given complex-`m` `inits`) returning one record per init: `m` (=√s), `s`, `mass = Re m`, `width = −2·Im m` (i.e. `√s = M − iΓ/2`; a physical resonance has `Im √s < 0`), `residual = |D|`, `converged`, `n_iter`.  `sheet` is the INT `1` / `−1` and it is NOTHING BUT THE SIGN OF THE PHASE-SPACE SQUARE ROOT — `q = ±sqrt(p²)/(2m)`, threaded through `_two_body_cm_mom` / `phase_space` / `gs_unit` / `_rho(m, sheet)`.  It lives on the ROOT, not on ρ as a whole, because a multi-channel model (Flatté-like) has one `sqrt` per channel and can put them on different branches.  For this one-channel family ρ is odd in `q`, so `ρ(m, −1) = −ρ(m, 1)` exactly and `den(m, 1) + den(m, −1) = 2·P(m²)`; the GS unit additionally moves `fs` through the `log` of `h(s)`, so there the two branches are NOT exact negatives (measured 0.8 % apart at `0.775−0.05i`, 5.5 % at `0.70−0.10i`, pole shifted by 1e-5).  Measured: zero widths reproduce `mass_list` exactly (residual 0, one iteration); a constant ρ (sheet-blind) gives `s = m₀² − i·m₀Γ₀` to 1e-10; ππ P-wave `m₀ = 0.775, Γ₀ = 0.149` → `0.76996 − 0.07501i`; the production `a2(1320)` 4-pole set converges on all four poles (residual ~1e-14) on every sheet.  A COMPLEX mass needed the analytic continuation of ρ: `_two_body_cm_mom` (both `bwr_model` and `gs_rho_model`) keeps its below-threshold clamp on the real path and takes `±sqrt(p²)/(2m)` (no clamp) for a complex one, and `gs_unit` / `_rho` no longer force `dtype=float` — real inputs are unchanged bit for bit.  `kmatrix_n_gs` degenerates to the `GS_rho` propagator at `width_norm: false` with `width_list = [g0]`, so its pole IS the GS pole; `kmatrix_n_ieps` inherits `solve_pole` and reports the ε→0 pole (`ieps` is a real-axis regulator, not part of the pole condition).  `Fitter.get_bw_params` / `particle_bw_params` still refuse a basis model — the report is a 2-number BW peak — so `solve_pole` is the model-level answer.

**Which models are dispersive at all.**  PLAIN `kmatrix_n` contains NO dispersion: its ρ is the real M-free phase space (0 below threshold, real above), measured `Re ρ` only — nothing in it is a dispersion integral.  The complex unit, and with it any Re/Im structure, belongs to `kmatrix_n_gs` / `GS_rho` alone (`ρ = Γ/g0 + i·fs/M`).
**The GS unit's h is the analytic function, for EQUAL daughters.**  With `m1 == m2 == m` the h function has the closed analytic form `h(s) = (2/π)(q/√s)·arccosh(√s/2m)`, `q = sqrt(s−4m²)/2` — single-valued on the s-plane cut along `[0, 4m²]`, so `Re h` and `Im h` are a Cauchy (dispersion) pair on that cut.  The implementation is the log form `(2/π)(q/√s)·ln((√s+2q)/sm)`; the two agree to 4e-16 (ππ) and 2e-15 (KK̄), gated in `tests/test_rho_omega_gs.py::test_h_is_the_analytic_arccosh_form_for_equal_daughters`.  That is a statement about h's OWN cut — NOT that `fs` equals a Γ-dispersion integral over `[4m², ∞)`: no code integrates anything (`fs` IS the closed form), and such an integral is only conditionally convergent for a linearly growing Γ.  On the cut h is real, so the discontinuity of the unit there comes from the `q³` factor and is purely imaginary.  With UNEQUAL daughters the same log form uses `sm = m1 + m2`, which is the equal-mass formula, NOT the general unequal-mass loop: the difference is nearly degenerate with using the mean mass (measured ratios 1.00…1.05 for `|m2−m1|` = 0.02…0.20) but it remains an approximation.  It is not exercised in production: the GS-unit users are `GS_rho` (ρ770/r1450/r1900) and `RhoOmegaGS`, both ππ, i.e. equal-mass; the unequal-mass channels (πη) go through `BWR` / `one` / `ExpSplineSVD`, which have no h at all.

**Wave indexing**: a basis resonance expands ONE geometry wave into `count` FLAT ck slots (one per free numerator row), contiguous and in base order — `AmplitudeModel.wave_expansion() -> [(base_index, offset, count)]`, whose base default is the IDENTITY (plain `pwa`) and which `PWAGen` overrides; `sum(count)` is the FLAT base wave count (equal to `len(get_partial_waves_params())`, i.e. `get_ck_map()` pre-block) and therefore the block stride of the flat ck vector.  `full_decay.get_partial_waves_params()` is GEOMETRY-only (one entry per wave, no heads) and must never be used as the stride or as a flat ck index — that is a silent mis-mask, not an error: measured on the `kmatrix_n` production config (20 geometry waves, 26 flat slots) the geometry-based grouping claimed 20 of 26 slots, pointing every wave after the first expanded one at a neighbour's slot.  Consumers route through the expansion: `scripts/calc_first_decay_fractions.py` / `calc_sub_decay_ratios.py` (the stem regex still reads the geometry list — the numerator head is appended LAST, so `pw[1]` is unchanged) and `plot_pwa_groups.discover_pwa_groups`, whose block expansion is `block · sum(count) + slot`.

**The `Fitter` never touches particle models.**  `AmplitudeModel` owns the tree, so it owns every sweep over it: `particle_transforms()`, `particle_defaults()`, `particle_model(name)` and `particle_bw_params(name, resolved)` (the pole report: the model names the parameters it depends on — `get_bw_param_names`, overridden by `ck_matrix_v2` for its standalone width — and computes the pole).  `Fitter.setup_mass_width_transforms` just registers the list — it holds no model-specific knowledge.

The walk lives in `_particle_mw_transforms()` and `AmplitudeModel.particle_transforms()` returns exactly it — the `pwa` list.  **`PWAGen` overrides `particle_transforms()` outright** with its own: per-resonance basis transforms (`extra_transforms()`, which also drops a transform declaring neither inputs nor outputs — a no-op cannot change a value or a gradient) plus one `FixedValueTransform` built from its own `constant_weight_values()`.  No particle transforms: its coefficients ARE the basis weights and each basis transform is self-contained, so a particle transform would feed nothing while running on every `resolve`/`chain_gradient` and striking its outputs off the fitted surface.  The base class has no switch and no `constant_weight_values()` / `extra_transforms()` hook at all.

**The default basis transform is SELF-CONTAINED.**  `PlainBasisWeightsTransform` is every model's one answer to "physical parameters → basis weights": it holds the model's own `make_mass_width_transform()` stage in `self._mw` whenever a coupling it reads is DERIVED (`RhoOmegaGS`: `w0..w7` from `mass/width/mass2/width2`; `BWRS`: `{p}_width_t` from mass and width), and declares THAT stage's inputs — the physical parameters — as its own.  The mass name comes from `get_mass_name()` and is declared only when the stage does not produce it (`KToSVDWeightsTransform` returns it); declaring it anyway puts a direction on the fitted surface that the stage then overwrites.  A model whose couplings ARE its parameters (the default plain BW) declares them directly and holds no stage.  Measured `input_names`: `RhoOmegaGS` → `[mass, width, mass2, width2]`; plain basis → `[mass, width]`; `OneModel` (constant basis) → nothing.  So no transform has to be registered before it for it to be correct.

`amplitude` is `BaseModel`'s evaluation and states the denominator through the GAMMA names (`m0² − m² − i·m0·Σ g_l·γ_l(m)`); `amplitude_raw` is its own basis ratio and `amp_raw` is `BasisModel`'s.  `PlainBasisWeightsTransform` is that one conversion WITH the gradient: `forward` computes `s0 = m0²` and each `{g}_t = m0·g` and publishes every `get_den_name()` slot except `one`, and `backward` is the exact VJP of it.  The names are the layout: `{name}_s0` carries m0² and each `{g}_t` carries `m0·g`, so `backward` states the derivative of that same conversion against those exact names — no stored classification, no index into a list.  `_resolved()` MERGES (`{**d, **self._mw.forward(d)}`), the VJP chains the coupling gradients through `self._mw.backward`, and a missing declared input raises rather than resolving to 0.  A transform's `forward` returns ONLY its outputs, so the mass must be carried through explicitly.

**A transform must NOT emit one of its own input names.**  `apply_forward` writes `output_names` into the resolved dict, so an overlapping name OVERWRITES its input and the physical value is lost.  The kernel-facing `m0·g` slot is therefore named `{gamma}_t` (`bwr_model.BWRSModel.get_gamma_name` uses this convention), and `PlainBasisWeightsTransform` reads `{gamma}` while producing `{gamma}_t`.

`"one"` is NOT a per-resonance transform output: it is the constant m² coefficient (+1, the row is `-m²`), independent of every parameter.  It is produced ONCE for the whole fit by a `FixedValueTransform` (`param_constraint.py`) built from `PWAGen.constant_weight_values()` — `{"one": 1.0}` when its `den_names` contains it — and appended **last** in `PWAGen.particle_transforms()`, after every per-model transform.  One producer per name matters: two transforms declaring the same output are treated as alternatives by the constraint layer, so with a second producer the surviving value would depend on registration order.  `PlainBasisWeightsTransform` therefore excludes it from `output_names`, and no consumer patches another transform's output.

**`PWAGen.build_kernel_config` must NOT call `build_single_index()`.**  That method is the PLAIN-PWA index builder: it emits `g0_index` / `g0_mass_index` / `gamma_table` for a kernel whose denominator is `s0 − m² − i·mult·γ(m)`.  The basis path has no gamma table — it reads each resonance's BASIS rows — so `_pg_host_records` derives its BW list straight from the decay tree (`bw_names = [dec.core.name for dec in ch.decays[1:]]`, first occurrence, chain order) and never touches `unique_bw`.  Deriving from the tree keeps the basis path independent of the plain-PWA index structures.  The loops `unique_bw` / `unique_gamma` / `unique_fl` are then only exercised by the `pwa` path, where they belong.

**Consequence for the fp32 gates:** the `float2` cache floor tracks `P`'s DYNAMIC RANGE, not `n_basis`.  Measured: `_km`/`_same` agree with `numpy_flat` to ~1e-7, while `_ident`/`_cp` (P spans 7.5e-4 .. 7.2e2) reach ~6e-6 in `Q`, ~3e-5 in `dck`/`dcoeff`.  `_rel` divides the worst ABSOLUTE error by the global max, so the near-zero events dominate it.  The kernels agree with EACH OTHER to 1e-16 and the per-event MEDIAN relative error stays ~1e-7; the tests use per-quantity bounds (`CACHE_FLOOR`, `CACHE_FLOOR_P`, `CACHE_FLOOR_COEFF`).

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
construction.

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
`tests/config_pwa_gen_km.yml` (`kmatrix_n`, gate B/C), `tests/test_pwa_gen.py`
(includes mixed-basis, two-K-matrix, CP-block and numpy/cpu parity cases),
`tests/test_flat_tile.py` (the s0e compute family) and
`tests/benchmark_flat_pwa_gen.py`.

**`skip`** (`model: skip`) is a particle model whose denominator basis is the single constant row `[1]` —
the way a config STATES "this factor is 1".  It carries no flag: the decision is a DICT-ONLY optimisation
of the built flat config, `tabpwa/pwa_gen_config.drop_unit_factors(kc)`, applied once inside `build_from_groups`
(`return drop_unit_factors(config)`), so every `pwa_gen` backend — `numpy_flat` included — receives it.  `drop_unit_factors`
drops every factor `r` whose DENOMINATOR is the constant 1 in EVERY wave — the check is the whole
chain `gamma_matrix[:, r]` (which rows it selects) → `basis_table` (the row really is 1, not merely named
`one`) → the weight (`den_names == "one"`, the fixed parameter-free coefficient) → the NUMERATOR
(`num_table[:, r]` must be the constant 1 in every wave).  The numerator check is what keeps
`spline_c_idx` — a unit DENOMINATOR carrying the whole lineshape in its numerator — from being dropped
(measured: dropping it silently deleted the expansion); `skip`'s numerator IS 1, so `skip` still drops.
A numerator axis that is PRESENT but unreadable (not 3-D, or the wrong `n_res`) **raises** instead of
falling back to the denominator-only rule — `dedup()` emits a 2-D `num_table`, so a deduped config
cannot answer the question at all and must be passed in its full form.
It shrinks ONLY the factor axis: `num_table`, `gamma_matrix`, and the basis rows only that factor selected
(`basis_table`, `basis_m_index` ROWS, `basis_m_key`, `basis_kind`, `den_names`, `basis_key`); the shared
`one` row survives.  `num_m_index`
and `basis_m_index` keep their STRUCTURAL values — they ARE the mass column (`n_res·topo + r`), so nothing
is remapped and the event buffer keeps a column per structural factor (`pwa_build` and `read_var` stay
structural; the kernel simply never reads the dropped column).  A factor that is 1 in only SOME waves is
LEFT ALONE (the axis is shared), and `drop_unit_factors` returns `kc` ITSELF when nothing is droppable, so
every existing config is bit-identical.  `model.n_res` / `decay_tree.n_res` stay the STRUCTURAL `n_decay − 1`.
A config switches it off with `drop_unit: false`, which keeps every factor — the reference the drop has to
agree with (same P, Q within 1e-13).  `skip` is `pwa_gen` only: on the plain `pwa` path it declares no
running width, so that guard is the error.  `one` is NOT this — numerically 1 but occupying a slot, so its
users' configs are untouched.

#### Complex cubic spline in the numerator (`model: spline_c_idx`)

`SplineCIdx` (`particle_model/spline_c_idx_model.py`) is TFPWA's
`Interp1DSplineIdx` (`tf_pwa/amp/interpolation.py`) ported onto a
`BasisModel`: a **complex cubic spline in the daughter invariant mass, carried
by the NUMERATOR**.  The rows are solved ONCE at construction by
`spline_k_model.spline_basis_matrix` — the VERBATIM port of TFPWA's
`spline_xi_matrix` (piecewise cubic monomials `a + b·m + c·m² + d·m³` per
interval, one column per knot) — with **`bc_type: not-a-knot`** (the default;
`natural` differs by up to 0.449 in a cardinal row, so the choice is not
cosmetic; TFPWA's `clamped` has no port and is refused by name).
`with_bound` (default `False`) pins the two END knots to 0 by DROPPING columns
0 and N−1 (`n_free = N − 2`); `True` keeps all `N` (`n_free = N`) — column
omission is exact because the matrix is linear in the knot values.  The rows
are REAL; the coefficient is COMPLEX, one per FITTED knot, so `c_i = A(
points[i])` and `A` is the complex cubic spline through `(points[i], c_i)`.
`points` (or `min_m` / `max_m` / `interp_N`) needs `N > 2`, finite, unique,
ascending (not-a-knot is SINGULAR at exactly three knots and says so — use
`natural` there).  The denominator is the single constant row `[1]` (the
shared fixed `"one"` — the `SkipModel` shape), so `|A|² ∝ |Σ_i c_i·B_i(m)|²`
and the normalisation is the fit's.  `pwa_gen` expands ONE FLAT `ck` SLOT PER
ROW, so a geometry wave owns `n_free` contiguous slots
`ck_{w,i} = g_ls(w)·c_i` and `len(ck) == n_wave·n_free`.  **Nothing but the
coefficients is fitted**: the knots are config constants, so
`make_basis_transform() -> None` (a coefficient IS its knot's value) and
`get_defaults() == {}`; the heads are plain ck parameters, which
`_pg_chain_factors` accepts WITHOUT a transform (a DERIVED weight — `kmatrix_n`'s
β, `ExpCheb`'s Bessel weights — still needs one).  The spelling is TFPWA's:
`{name}_point_{i}r` / `{name}_point_{i}i` (`complex_tail`; the pair
`CKProduct` already builds) — no doubled row axis, and `polar: false` has no
tabpwa counterpart (refused).  GAUGE: TFPWA pins `point_value[fix_idx] = 1`
(default `fix_idx = N//2 − 1`); tabpwa's `ck_redundancy` constrain removes the
same redundancy but pins the FIRST fitted coefficient, so the model only
VALIDATES and EXPOSES `fix_idx` (`fixed_values()` returns the slot dict) — to
reproduce TFPWA's middle knot, free the default pin and `set_fixed(...)`.
`outside` (default `"spline"`, TFPWA's `spline_c_idx`): the interval index is
clipped to `[0, N−2]` and the end interval's CUBIC is evaluated — cubic
continuation, which GROWS (measured on the 6-knot fixture with random
coefficients: `|A|` = 4.8 / 10.5 / 142 at h/2, h, 4h BELOW the support and
5.9 / 15.1 / 236 ABOVE); `outside="zero"` is TFPWA's twin `spline_c` (exactly
0 outside; the two agree inside).  Rows MEASURED on the 6-knot fixture: vs
`scipy.CubicSpline` 5.8e-14 (not-a-knot) / 1.0e-13 (natural), vs an
independent tridiagonal solve 1.1e-13, partition of unity 9.5e-14,
cardinality 2.8e-14 (4.6e-12 over 18 knots); `num_basis` costs 0.14 ms on the
2000-point build grid (0.31 ms at 18 knots; one memoised evaluation per build,
the builder asks once per row on the same object).  TABULATION: TFPWA
evaluates the cubic exactly per event; here `num_basis` is sampled on
`gamma_grid` (`n_interp`, default 2000) and the kernels re-interpolate with
Catmull-Rom, so the amplitude is CR∘spline — keep `points` inside the grid
window (`m_min − 0.01 … m_max + 0.01`) and raise `n_interp` via
`build_defaults.scope` to shrink that extra error, which is ≈2.1e-7 at the
default and falls as `n_interp⁻³` (`~1.6e3/n³`; measured 1.07e-4, 1.33e-5,
1.61e-6, 2.05e-7, 2.46e-8 at n = 250/500/1000/2000/4000).  `drop_unit_factors` checks the NUMERATOR too: a unit
denominator does not make a factor the identity (`skip` still drops, this model
must not).  Fixtures `tests/config_pwa_gen_cidx.yml` (6 points, `with_bound`
default ⇒ 4 slots) and `tests/config_pwa_gen_cidx_2f.yml` (a two-factor chain:
the spline beside an ordinary BW, where the denominator-only rule would have
dropped the expansion); gates `tests/test_spline_c_idx.py` for the model and
`tests/test_skip_model.py` for `drop_unit_factors` (the owner of that function).

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

### Toy data (`Fitter.generate_toy`) — the sample's law, its ceiling, its norm

THREE entry points, each producing only its own sample:
`generate_toy(x0, n, *, seed, top_p, maxp, fbg, stats)`,
`generate_toy_bg(x0, n, *, purity, bg, …)` (the mixture) and
`generate_phsp(n, *, seed, top_p, bkg, momenta, stats)` (the norm).  There is no
`n_phsp` inside the toy: accept–reject needs only `P` and a ceiling (a constant
factor cancels), so the normalisation integral is a separate call.  The toy's
rows are the MODEL's whole event buffer (mass/q/angle/weight plus whatever the
model adds), never a fixed key set.

WHICH generator runs is the model's: `AmplitudeModel.toy_generator()` returns
`None` for the generic accept–reject (`tabpwa/toy_pwa.py`, which needs only
`build_event_data`, the parameter layer and a backend), and `FlavourTagMix`
returns its own for `frac`/`time` (it refuses `top_p`/`maxp`, building its own
phase space).  A hook is called with the FLAT `x0` and must go through
`build_params`; handing it an already-RESOLVED dict re-resolves it through the
constraint manager and moves the generating point (~2× on the masses) while
`ToyStats.truth` still reports `x0`.

The per-event law lives in the CONFIG, once: `data.extra_var.<name>.random` is a
NUMPY call `{dist: <RandomState method>, **kwargs}` (`uniform`, `choice`,
`exponential`, …), and `AmplitudeModel.mixing_inputs(n, rng)` fills the columns
the model declares in `extra_columns`, so the toy signal and a background drawn
from the same declaration carry the same law.  `default:` alone is a constant,
and an undeclared column falls back to `time ~ U(0, TIME_MAX)` /
`frac ∈ {0, 1}`.  Both sides evaluate it with a `RandomState` (a numpy call on a
`Generator` would accept a different method set).

The ceiling is in/out: `stats.maxp` comes back and goes in as `maxp=` (it is a
property of the parameter point and the topology), `maxp_scale < 1` is refused
(a raise cannot lower a maximum) and the `guess/new` thinning is exact because a
raise precedes its own batch's acceptance — `raise_ceiling` is the ONE
implementation, shared with the background draw and with the time-dependent
engine (whose `p_max_margin` is a MARGIN, not a fixed bound: a later batch above
it raises and re-thins, the same rule).

PROGRESS AND PERSISTENCE ARE OPT-IN KNOBS on both engines:
`progress=True` prints ONE throttled line per second (`ToyProgress`: accepted/
target, batches, flat tried, efficiency, rate, elapsed, ceiling; `None`, the
default, is AUTO — on only for a terminal, so a log or a test stays quiet), and
`cache_dir=` checkpoints to disk and RESUMES: every accepted batch is appended as
`chunk_NNNNNN.npz` and `state.json` (counters, the ceiling, the RNG state) is
replaced atomically after it, so a `max_loops`/wall-clock kill loses nothing and
the next call with the same directory continues the SAME sub-seed stream — the
resumed sample is bit-identical to an uninterrupted run at that seed (gated).
A cache belongs to a SIGNATURE (`toy_cache_signature`: engine, target, seed,
resolved-parameter digest, model name, kernel-config digest), so another point in
the same directory raises instead of blending two samples; a finished one starts
over; and `cache_dir` needs an explicit seed because the seed is part of that
signature.  `ToyCache` is ONE implementation, used by both engines.

THE RUNAWAY GUARD IS A NO-PROGRESS WINDOW, never a total candidate budget:
`stall_loops` (default 1000 batches) raises when that many CONSECUTIVE batches
accept nothing, and `max_loops` defaults to `None` (an optional hard cap).  A
peaked, low-efficiency amplitude runs as long as it keeps accepting — a fixed
candidate cap (the old 20 M) is wrong for exactly that model — while a dead
density (a ceiling far above the true maximum, or a zero support) still fails
in bounded time.  Both engines share the rule: `toy_pwa_generate` and
`toy_generator.toy_generate`.

THE BACKGROUND COLUMN IS A SHAPE, never a normalisation: `bkg = fbg(events)` in
the toy (`None` → ones; `fbg=lambda x: np.zeros(...)` for a strict signal-only
column), `bg(...)` in the mixture and `bkg(...)` in the norm — one evaluator,
`toy_pwa.bkg_values` (uniform for `None`; refuses a wrong length, non-finite or
negative values).  The AMOUNT is `set_data`'s: `bkg_scaled = (1-p)/p · bkg / N_b`
with `N_b` the norm sample's mean, so the norm must be built with the SAME
function.  Counts are two Poisson draws (`n·p`, `n·(1-p)`); an EMPTY draw raises
rather than returning a buffer with no `mass`; `purity` defaults to
`config_purity()` (top-level → `data.purity` → `data.bg_frac`), and `p = 0` is
generation-only because `set_data` divides by it.

Tests: `tests/test_gen_toy.py` (every model shape, the TD law, the ceiling, the
refusals) and `tests/test_toy_generator.py` (the `draw_extra` vocabulary).

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

## System monitor (`tabpwa.monitor`)

ONE sampling entry point, used by `benchmarks/ctpwa_kketa/mem.py` /
`mem_shard.py`, the device-memory leak gates and any harness that wants a peak
or a series:

```python
with create_monitor(what=("gpu_used", "gpu_total", "hwm"), interval=0.05) as mon:
    mon.mark("load"); ...; mon.mark("nll")
res = mon.close()                       # idempotent; get_results() works mid-run too
res.peak("gpu_used", phase="nll"); res.series("hwm"); res.summary()
```

**The sampling runs in a SUBPROCESS** — one `nvidia-smi --query-gpu=… -lms N` child
for the whole run, whose stdout a single parent thread only drains.  A sampling
THREAD would share the GIL and the CUDA context with the timed region and perturb
the numbers it reports, which is why it is not one.  The SAMPLER therefore
**creates no CUDA context** — it reads the driver — and cannot inflate the
memory it measures.  The separate `free_bytes()` point probe DOES initialise the primary
context on its first call — which is exactly why it is not part of the sampler
and is not used for a peak.  `nvidia-smi` reads the driver, so MiB is the
granularity — a gate that needs exact BYTES uses `monitor.free_bytes()` /
`monitor.sync()`, the `libcudart`-via-`ctypes` point probe (the single copy,
in `src/`; there is no test-side copy).

**`what=` selects the FIELDS and the MECHANISM.**  Any `gpu_*` field starts the
child and only those columns are requested; a host-only request (`rss`, `hwm`)
starts no child, no pipe and no thread, so a `cpu_*` benchmark is structurally
GPU-free and a GPU-less box behaves identically.  `what=()` is legal (timing
only).  An **unknown** field name raises, listing the valid ones; `peak`/`series`
of a field that was **not requested** raises with what this monitor does sample —
which is a different answer from a requested field the box cannot provide
(`None`).  Collapsing the two is the silent failure this API exists to avoid.

**A memory GATE is written on `free_bytes()` (bytes), never on the sampler.**
The sampler is MiB-granular by construction (`nvidia-smi`), and a phase peak can
also catch the neighbouring step's frees because a `mark()` is not a barrier.
MEASURED: the mask-change leak gate migrated onto `gpu_used` with its 2 MiB bound
produced a 4 MiB spread and failed inside the suite while passing in isolation —
a bound at the sampler's quantisation measures the box's quietness, not a leak.
The rule is: the sampler for a PEAK or a SERIES (a benchmark's resident memory),
`free_bytes()` for a before/after assertion (a leak gate).

`close()` is safe on the normal path, on an exception and twice over (`terminate`
→ `wait` → `kill`, plus an `atexit` reaper), because the recipe it replaces —
`nvidia-smi -lms 100 … & kill %1` — leaves an orphan sampler behind whenever the
run dies.

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
pytest tests/ -v                        # 2831 tests, 11 skipped
pytest tests/test_fitter_constraints.py  # constraint pipeline
pytest tests/test_new_apis.py            # LinearTransform, fmt_meas, get_defaults, …
tests/validate_gradients.py              # 3-point gradient validation (all 6 backends)
```

**The flat family's suite is grouped by CONTRACT; there are no kernel-named test
files.**  Eight files, each the owner of one contract (no other file may hold a
gate on the same backend/sample/config — a duplicate is deleted rather than kept
in two copies that can drift):

| file | owns |
|---|---|
| `tests/_flat_compute.py` | the unmasked `load_data`/`compute` contract, over every leaf — **axis A**: a kc STRUCTURE x every leaf — `_flat_models.CONFIGS` declares each fixture's `needs` (the capability tag it must tolerate) and the mechanism it covers; a leaf is swept on every fixture whose `needs` its tags do not refuse (`REFUSES`), minus the multi-BW pair whose anchor `test_flat_dedup.py` owns, so `FLOORS` is the measured RECORD and no longer the selector |
| `tests/test_flat_models.py` (`tests/_flat_models.py`) | **axis B**: every registered particle MODEL as a UNIT OF A CONFIG — its declaration (row names, heads, `ck` count, `wave_expansion`) with no backend, then its numbers and its FD gradient on `pwa_gen`'s own default (the `numpy_flat` PATH is `isinstance`-pinned, no absolute value is pinned per model — a self-consistent wrong forward passes, a wrong VJP does not); no per-leaf matrix and no fixture — each entry renders into `_flat_models.TEMPLATE`, a minimal in-memory config, pinned key-by-key against `config_pwa_gen.yml` for the default basis.  `MODELS`/`EXCLUDED` (reason + owner) and `OWNED_ELSEWHERE` (a model-only fixture's owner) are live gates |
| `tests/_flat_masked.py` | the fixed-coefficient contract (`fixed_coeff`), incl. the split's per-context scratch gate |
| `tests/_flat_pinned.py` | the pinned-coefficient contract (`coeff_pinned`) |
| `tests/test_flat_tile.py` | the s0e family's config contract (five kernels as parameters) |
| `tests/test_flat_dedup.py` | the dedup/index family, incl. the deduped-layout bit parity |
| `tests/test_flat_ampcache.py` | the two cache variants' dtype and Gram claims |
| `tests/test_flat_isolation.py` | the ONE case-1 table (source digests, own module/`.so`; the C symbols are NOT renamed) |

Selection is by CAPABILITY TAG, never by name: each contract derives its backend
list from `backends_for_model("pwa_gen", tag=…)` (`reference` / `delegate` /
`n_res1` / `fixed_coeff` / `coeff_pinned` / `probe`), so a backend registered
later is covered without editing any file.  The selection is LIVE: `probe` is a
declared tag (test stubs were model-less and landed in every model), the rule is
a pure `_select(registry, tags_of)`, and `_missing_from_suite` reports a backend
registered after import instead of ignoring it.  `CONFIG_OWNER` (in
`tests/_flat_compute.py`) maps every flat leaf to the file that owns its config contract and each owner
asserts the mapping, so a new leaf fails at import rather than going silently
unswept.  `pytest tests/<file>.py` collects from the repo root without
`PYTHONPATH` (`tests/conftest.py` adds the directory).

**A test is `(config, params, sample) → numbers`, plus a refusal, plus the guard
on a pinned value.  An assertion about HOW a kernel works is not a test**: no
`te` ladders, no shared-memory formulas, no launch counters, no CSR internals, no
symbol lists, no register/spill counts (~160 such assertions were removed
deliberately).  Those levers are BENCHMARK-only — `split_fixed`, `fused`,
`bwd_mode` are A/B knobs for `tests/benchmark_s0f.py` / `benchmark_s0g.py`, and
`te`/spills are printed by `tests/benchmark_ampcache*.py` — and one measured
consequence is recorded: `[s0e4-split_nnz]` on the production shape sweeps
`norm=1.3` only, because at `norm=None` its `dcoeff` measures 1.1e-14 against a
1e-14 bound (true in the deleted per-kernel file too).


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
