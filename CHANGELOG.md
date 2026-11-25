# Changelog

All notable changes to TabPWA are documented here.  The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) — at `0.x` a minor
bump may change the API, a patch bump may not.

**Where the number lives.** `src/tabpwa/_version.py` is the ONE source;
`pyproject.toml` reads it, and `tabpwa.__version__`,
`python -m tabpwa --version` and `python run_fit.py --version` all report it
(`tests/test_version.py` fails if a surface drifts).

**How to change it.** Every user-visible change (behaviour, API, CLI, numerical
result, file format) gets a bullet under `[Unreleased]`, in the same commit.
A release moves that block under a new dated `## [X.Y.Z]` heading, bumps
`_version.py`, and tags `vX.Y.Z`.

## [Unreleased]

### Added

- `spline_c_idx` particle model (`particle_model/spline_c_idx_model.py`): a
  COMPLEX cubic spline in the daughter invariant mass, carried by the
  NUMERATOR as a `BasisModel` — TFPWA's `Interp1DSplineIdx` construction (the
  `spline_xi_matrix` port; `bc_type: not-a-knot`), with the knot values as the
  fitted complex coefficients (`{name}_point_{i}r` / `{name}_point_{i}i`;
  `with_bound: false` by default pins the end knots to 0 and fits the N−2
  inner ones).  The denominator is the constant row `[1]`, so `pwa_gen`
  expands one flat `ck` slot per fitted knot and
  `|A|^2 ∝ |Σ_i c_i·B_i(m)|^2`; `outside: spline` keeps TFPWA's cubic
  continuation outside the support (`outside: zero` returns 0 there, TFPWA's
  `spline_c` twin), and the plain `pwa` path refuses the model naming
  `amp_model: pwa_gen`.  A numerator model whose weights ARE its parameters is
  now accepted by `pwa_gen` without a basis transform.  `drop_unit_factors`
  additionally requires a factor's NUMERATOR to be 1 before dropping it, so a
  unit denominator no longer hides a numerator expansion (`skip` unchanged),
  and it now raises on a present-but-unreadable numerator axis (a deduped
  config's 2-D `num_table`) instead of silently restoring the old rule.
- Config-declared plugins: a top-level `plugin: [module, …]` (also `plugins:`, or
  a bare string) makes `Fitter` import those modules right after loading the
  config — before the amplitude model is built and before the backend is
  resolved — so a package that registers a backend/model needs no `import` in
  the calling script and no entry-point machinery.  The import is idempotent; a
  failure names the plugin and the config.  `tabpwa.plugins.load_plugins` is the
  same entry for a caller building a `BaseModel` directly.  A config is trusted
  input (importing a plugin runs code).

- `mpi_service` backend (`tabpwa/backends/mpi_service_backend.py`, tags
  `delegate`+`external`): the PROGRAM runs once and the MPI job is a CHILD —
  `mpiexec -n N python -m tabpwa.mpi_worker` started lazily on the first
  `load_data`, rank 0 driven over a Unix socket (stdio cannot carry it: the
  launcher muxes the child's stdin).  The worker loop bcasts each op and calls
  the ORDINARY `mpi` backend on every rank, so slicing, device binding, the
  build barrier and the reduction exist once.  The driver imports no `mpi4py`
  and has no rank, so a program writes output once; `n_procs` is required,
  `local` is a leaf, nesting inside an MPI/Slurm step is refused, and lifecycle
  is `shutdown`→`terminate`→`kill` with an `atexit` reaper.  `ship` loading in
  v1.  Design `docs/mpi_service_design.md`, as built `docs/mpi_service.md`.

- `mpi` backend (`tabpwa/backends/mpi_backend.py`, tags `delegate`+`external`): N ranks
  under `mpiexec`, one contiguous slice each (`split_offsets`, weighted/align-
  aware), with `Q`, every gradient block and `grads["norm"]` reduced by
  `allreduce(SUM)`; the per-event `P` is allgathered only when
  `return_p=True`.  Rank→device binding by node-local rank (a launcher-set
  `CUDA_VISIBLE_DEVICES` wins; a rank with no GPU raises), the first kernel
  build serialised across ranks, and `shard` nested inside a rank refused.
  The package stays transport-agnostic: no rank or launcher logic outside the
  backend; the CLI (`run_fit.py`) asks the backend's `rank` and writes on
  rank 0 only.
  `local` must be one concrete kernel per rank (a delegate/composer is
  refused by tag); a launcher-set `CUDA_VISIBLE_DEVICES` is respected — one
  entry is this task's GPU, a node-level list is indexed by the node-local rank
  (Slurm `--gres=gpu:N`), and an empty value hides every device.  The first compute broadcasts rank 0's parameters (an unseeded
  `initial_values()` differs per process), a rank with an empty slice or no
  device aborts the job with its message instead of hanging the peers, and an
  `mpi` backend inside a `shard` worker is refused.
  Optional extra `tabpwa[mpi]` (mpi4py, lazy import).  Design:
  `docs/mpi_backend_design.md`; as built and measured: `docs/mpi_backend.md`;
  deployment: `docs/slurm.md`.

### Added (pwa_gen factor-level dedup)

- `cuda_flat_pwa_gen_fslot` (`pwa_gen`): a case-1 copy of
  `cuda_flat_pwa_gen2` that runs the coefficient-gradient sweep over distinct
  FACTOR rows — `(wave, factor)` gamma rows grouped across waves and across the
  factor axis — instead of the parent's wave blocks, so two topologies that
  share a resonance compute it once.  Backward only (phase 1, `Q`, `P`, `dck`
  and `norm` are the parent's, and `dcoeff` is the same sum re-associated:
  measured 5 ulp (`_mb`) / 26 ulp (`mb2`) against the parent, at or below gen2's OWN run-to-run spread), and
  gated: it engages only when `n_res > 1` and `n_fslot < n_slot·n_res` — a real
  saving against the parent's own dot count (at equality the extra reduction
  buys nothing and measured 4 % slower), otherwise the parent's path runs
  unchanged (`n_res == 1` always).  No `n_res1`
  tag — it takes a multi-BW chain.

### Added (toy generation)

- `generate_toy(progress=True)` prints one throttled progress line per second
  (accepted/target, batches, flat tried, efficiency, rate, elapsed, ceiling);
  the default `None` is AUTO — on only for a terminal, so scripts and logs stay
  quiet.  Both engines report through the one `toy_pwa.ToyProgress`.
- `generate_toy(cache_dir=…)` checkpoints the accepted pool to a directory and
  RESUMES from it: each accepted batch is appended as `chunk_NNNNNN.npz`, and
  `state.json` (counters, the ceiling, the RNG state) is replaced atomically
  after it, so a wall-clock kill or a crash does not throw a long generation
  away — the next call with the same directory continues the same RNG stream,
  and the resumed sample is bit-identical to an uninterrupted run at that seed.
  A cache belongs to a signature (engine, target, seed, generating point, model
  and kernel config), so another run in the same directory raises instead of
  blending two samples; a finished cache starts over.  `cache_dir` requires an
  explicit seed.  `ToyCache` is one implementation for both engines.
  `scripts/gen_toy_pwa.py` exposes it (`--cache-dir`, `--stall-loops`,
  `--verbose`) and `--max-count` now defaults to `0` (no cap), so the CLI
  follows the no-progress rule instead of stopping a slow but healthy model.

### Added (bounds diagnostics)

- `Fitter.boundary_report(result)` lists the free slots sitting ON a range edge,
  and `fit()` / `get_uncertainties()` emit a `RuntimeWarning` naming them: a
  binding soft range has no sigma (the transform's limit, not the model's:
  `dy/dx = 0` at the wall, so the delta sigma collapses and `H_xx` there is set
  by the pull), while the location and every other parameter stay correct
  because the bound map is per-slot diagonal.  "At the boundary" is
  `value ≈ a|b` within `1e-3` of the width (`Fitter.WALL_TOL`) — the sine puts
  the edge at a finite `x` and the fit converges onto it, where the arctan
  needed `x ~ 1e15` and its `1e15`-scale comparison.  Nothing is
  repaired silently: widen/remove the range and refit, or use the profile.

### Added (uncertainties)

- The constraint pipeline takes a START CUT, the mirror of `stop_after`:
  `cm.resolve` / `cm.flat_resolve` / `cm.full_gradient` and
  `Fitter.build_params` / `get_nll` accept `start_after`, where
  `start_after="bounds"` means the input IS the bounded value — so the bounded
  frame is a first-class variable and `get_nll(y, start_after="bounds")` returns
  `dNLL/dy`.  `compute_numerical_hessian(space="bounded")` is now that ordinary
  engine (one `fd_hess` with per-slot steps) instead of a hand-rolled
  `dNLL/dx·dx/dy` loop; `start_after` is part of the `build_params` cache key,
  a `y` call does not overwrite `_last_xk`, and an out-of-range `y` raises.

- `get_uncertainties(..., return_hess_inv=True)` also returns the inverse
  Hessian that was USED, as `(uncertainties, hess_inv)` — the `(N, N)` matrix in
  the same `x` space `fit_result.x` lives in, so a result that carries none (a
  constrained SLSQP fit, a JSON saved without its error matrix) can be given one
  back with `res.hess_inv = …` and then work with `save_params`,
  `cal_uncertainties_multi`, `fwd` and `profile_scan`.
- `compute_numerical_hessian(space="bounded")` / `get_uncertainties(
  space="bounded", use_cached=False)` take the curvature in the AFTER-BOUNDS
  value of each free slot instead of the optimizer variable, which is what makes
  a parameter near its range edge report a physical sigma: the arctan bound has
  `dy/dx -> 0` at the edge, so the x-space route is mis-scaled and singular
  (measured `cond(H_x) = 1.1e11` and a 154x error at a converged optimum 0.025 %
  inside the range, against a physical sigma that does not move with the
  distance to the edge; measured 21× at 0.01 % inside a soft range, 1.000 at
  mid-range).  `return_hess_inv=True` follows the SPACE: with
  `space='bounded'` it returns the after-bounds covariance, which belongs in
  `res.hess_inv_bounded` (the attribute the bounded route reads back), never in
  `res.hess_inv` — that one is chained through `dx` and is a different matrix.
  It is stored as its OWN file —
  `save_params(..., bounded_errors=True)` writes
  `*_error_matrix_bounded.npy` beside the x-space `*_error_matrix.npy`,
  `load_results` restores it as `res.hess_inv_bounded`, and
  `get_uncertainties(space="bounded")` then reuses it instead of paying the
  `2·N` gradient evaluations.  The JSON now records the frame of every block
  (`error_frame: "value_bounded"`, `hess_inv_frame: "x"`,
  `hess_inv_bounded_frame` when the second file exists), because the `error`
  dict and the matrix were always in different frames and nothing said so.
  `save_hessian(..., space="bounded")` saves the physical one standalone.

### Changed

- Bounds are now the SINE transform, `y = bias + k·sin(x/k)` on
  `|x| ≤ k·π/2` (the map MINUIT and every stepping minimizer of that family
  uses), replacing the arctan: the edge is the FINITE `x = ±k·π/2` instead of
  `tan(π/2)/c ≈ 1e15`, `dy/dx = cos(x/k)` decays linearly there instead of like
  `1/x²` (measured `9.9e-12` → `2.0e-3` at `y = b − 1e-6·w`, `w = 0.5`), and a
  `default` sitting on an edge no longer inverts to a start where the gradient is
  `1e-32`.  `x` is NEVER clamped: the plain sine is periodic, and at the edge
  `y' = 0`, `y'' = -1/k`, so `F = NLL(y(x))` has `F'' = |g_y|/k > 0` — a binding
  soft range converges to a PROPER, unique minimum at the edge (measured
  `x* = 7.4539` vs branch end `7.4524`, `y* - b = -2.3e-7`, `H_xx = +0.129`)
  instead of being pinned by a flat region (`dNLL/dx = 0` on a half-line,
  non-unique minimiser, `H_xx = 0`).  `inverse` is the principal branch, so the
  PHYSICAL value stays single-valued while the optimizer coordinate does not; a
  σ still does not exist at a binding edge (the profile likelihood is the answer),
  which is what `fit()` now reports instead of letting `success=True` speak.
  `initial_values` draws a bounded slot's random start ON the branch (rescaled
  from `U(-π,π)`), so narrowing `var_range` no longer pushes every start onto the
  wall (measured before: `|x|`-spread covering 0.003 % of a width-0.1 range).
  The old `boundary.BoundTransform` behaviour is gone, not selectable.
- ONE-SIDED bounds: `set_range(name, lo, None)` / `var_range: {name: [lo, null]}`
  (and the `None, hi` mirror) use the same `boundary.BoundTransform` with
  `kind="lower"`/`"upper"`, the unit-scaled `y = lo + sqrt(x²+1) − 1` on `x ≥ 0`
  (mirror on `x ≤ 0`).  No
  per-parameter scale — that would make the optimizer's step unit depend on the
  parameter — so `|x| ≈ 1` is the quadratic region everywhere and `dy/dx → 1` in
  the bulk.  The open side is `±inf` internally (the step/clip code is unchanged)
  and `null` in the saved JSON.
- A `default` that maps onto a transform's WALL now starts just INSIDE it:
  `initial_values` places the start at the nearest point with
  `|dy/dx| = START_GRAD = 0.1` (`boundary.BoundTransform.start_inside`) instead
  of exactly on the edge, where `dNLL/dx = 0` and `F'' = |g_y|/k > 0` — a
  stationary, PD point, so a minimizer reported convergence at iteration 0 and
  a default outside its range silently fitted the WRONG (wall) value (the
  `pi1(1600)` case: range `[1.4, 1.9]`, default `1.3544`, interior optimum
  `0.22` away).  The inset is a fraction of the WIDTH, not of the range's
  position, because every shape is quadratically flat at its wall: `0.25 %` of
  a range's width in `y` (`5.0e-3` for a half-line) buys 10 % of the y-gradient
  in `x`, far above any `gtol`.  The warning stays (the default and the range
  disagree — a config error) and the transforms keep their exact meaning, so a
  fit genuinely pulled outwards still converges ONTO the edge.

- `generate_toy`'s runaway guard is now a NO-PROGRESS window, not a total
  candidate budget: `stall_loops` (default 1000 batches) raises when that many
  CONSECUTIVE proposal batches accept nothing, so a low-efficiency (peaked)
  amplitude runs as long as it keeps accepting — the old `max_loops=1000` cut it
  off after 20 M candidates however healthy the acceptance was.
  `max_loops` now defaults to `None` (an optional hard cap) and stays
  available for a caller who wants one.  The time-dependent engine
  (`toy_generator.toy_generate`) gets the same guard, and its `P_max` estimate
  is RAISED to `p_max_margin·max(batch P)` when a later batch exceeds it — the
  accepted pool re-thinned by `old/new` through the shared
  `toy_pwa.raise_ceiling` — instead of a fixed 1.3-margin ceiling that silently
  over-accepted events above it.  `p_max_margin < 1` and a non-positive
  `batch_size`/`compute_batch`/`stall_loops` are refused.

### Fixed

- HALF-INTEGER spins (`J: "1/2"`, `J: 0.5`) were unusable: `_spin_state_count`
  did `int(2*J)` on the raw config value (`ValueError: invalid literal for
  int()`), `_helicity_top_states` did `int(J)`, and `Decay.get_ls_list`
  subtracted the raw values (`TypeError: unsupported operand type(s)`).  All
  three read the spin through `helicity_angle.to_spin`, so a fractional `J`
  works: `s` steps in halves from `|jb-jc|`, the ORBITAL `l` stays an integer
  in `[ceil|ja-s|, floor(ja+s)]`, and the helicity list is `2J+1` values for
  the top AND for every final (one helper, `_helicity_states`; the per-final
  list used to enumerate `-2J … +2J`, i.e. `4J+1`, which agreed with `2J+1`
  only while every final was spin-0).  A config in which NO chain admits a
  wave — a spin-1/2 core decaying to two scalars needs `l = 1/2`, which does
  not exist — is now refused by name at that vertex instead of surfacing as
  `ValueError: need at least one array to stack` from `build_from_groups`; a
  single dead chain beside live ones still builds.
- A HOPELESS toy background was diagnosed after ~33 s: the all-zero check
  samples a 1e6-event budget in batches, the batch was pinned at
  `max(64, 4·(n_bg−have))`, and `build_event_data` costs 55 µs/event at 64 vs
  5.4 at 1e5 — 15 625 rebuilds.  The batch now GROWS while the answer is still
  "zero" (64 → 8192, cap); the 1e6 budget and the message are unchanged and the
  cost is 4.1 s (MEASURED).  A NARROW support costs the same (its first small
  batch already accepts) but a background that is zero on an EARLY batch now
  proposes larger batches, so it draws a DIFFERENT RNG sequence at a fixed
  seed — a different sample from the previous release.  The false "zero over
  1e6" probability is unchanged: it depends on the total sampled events, not on
  how they are chunked.
- `shard` reported nothing when a worker could not be started or died: the
  parent saw `BrokenPipeError` from `p.start()` (spawn's preparation write) or
  blocked forever in `queue.get()` when a worker had already exited
  (`worker_timeout=None` is the default and means "no deadline", not "no
  diagnosis") — on a batch system, a burnt allocation.  Both now raise with the
  worker index and, for `spawn`/`forkserver`, the cause: those methods RE-EXECUTE
  the calling script in every worker (`runpy.run_path(…, run_name="__mp_main__")`,
  since the `Process` object is pickled and its target may live in `__main__`), so
  a program whose top-level code is not under `if __name__ == "__main__":`
  re-runs itself in the child (measured: a bare `multiprocessing` program then
  exits 0 with a dead worker).  Workers are now DAEMON processes owned by the
  backend, and a non-daemon worker is joined by the interpreter at exit, so an
  unhandled error in the parent hung the program on a worker blocked in
  `task_queue.get()`.  One consequence: a NESTED `shard` (a shard worker
  hosting another shard pool, which used to run) is now REFUSED by name, since
  daemonic processes may not start children — use one flat `backends` list.
- A zero-weight group returned `NaN` gradients on `cuda_flat_pwa_gen_s0c`,
  `cuda_flat_pwa_gen_s0c2` and `cuda_v5_pwa_s0cw` (`dck`, `dm0`, `dg0`, and
  `dcoeff` on the flat pair): the objective declares such a group inert
  (`W_g = 0` contributes no log, no gradient), and the NLL kernel skipped it —
  but the group-scale kernel divided `-W_g` by the group sum, which is 0 as
  well, so `0/0` was multiplied into every `dQ/dP` of that group (`dck` came
  back all-NaN).  Both kernels now publish a scale of 0 for a zero-weight
  group, exactly as their siblings already did.  Nothing changes for a group
  with positive weight (measured bit-identical on `Q`/`P`/every gradient/
  `dnorm`), and `s0c`/`s0c2` remain bit-identical to each other.  The
  `cuda_v5_pwa` / `_s0` / `_s0c` / `_s0d` siblings keep their own objective —
  `Q = -Σ log(pdf)` with no group weight — so they have no such division and
  are unchanged.
- `fd_hess` divided a Hessian COLUMN by the step of its ROW
  (`/ (2*eps)` with a per-slot `eps` array), so `compute_numerical_hessian(
  space="bounded")` was wrong on the off-diagonals once the bounded values had
  different magnitudes (measured `max|H−A| = 0.79`, ~6 % on the sigmas;
  `2.5e-10` after the fix).  Invisible with the scalar `eps` every other caller
  uses.  Gated by a NON-DIAGONAL SPD reference with two bounded slots of
  different magnitude — a diagonal reference cannot see it.
- `JointFitter` crashed (`float(None)`) on a config with a one-sided bound: it
  intersected the SERIALISED bounds dict; it now uses the live transforms
  (`±inf` for an open side), and an intersection with no bound unsets it.
- `load_results` loaded `*_error_matrix_bounded.npy` by file existence alone, so
  a stale matrix from an earlier save (or another run's shape) was adopted
  silently; it now requires the JSON frame tag AND the `(N,N)` shape, and
  `save_params` removes a bounded file it is not writing.
- A non-PSD curvature matrix no longer produces silent `NaN` sigmas:
  `_params_from_fit` uses `sqrt(|diag|)` and warns, naming a negative slot.
- `profile_scan` warns and clips the first step when it exceeds a bounded
  parameter's branch — the signature of a Hessian passed where the covariance is
  expected (or of a wall), both of which fold the scan.

- `bounds:` in a config raised `KeyError: 'height'` — it read `sb["height"]`
  instead of `sb["high"]`; both `bounds:` and `var_range:` now also accept
  `null` for an open side.

- `get_uncertainties(use_cached=False)` ALWAYS computes the numerical Hessian
  now.  The guard was `not use_cached and hasattr(fit_result, 'hess_inv')`, so a
  result without the attribute (a plain scipy `OptimizeResult`, exactly the
  SLSQP case) fell back to the cached route and reported 0.0 for every
  parameter — measured 4/4 zeros before, the true matrix now.
- A scipy result from `method="L-BFGS-B"` is usable: its `hess_inv` is a
  `LbfgsInvHessProduct` LinearOperator (it has no `toarray`), which made
  `FitResult`'s `np.asarray` raise `TypeError` AFTER the fit had converged.
  `tabpwa.fit_result.as_matrix` densifies it by the operator's own action on the
  identity, and every consumer (`save_params`, `save_hessian`, `profile`,
  `cal_uncertainties*`) goes through it.

## [0.1.0] - 2026-11-24

First version recorded by this changelog.  The framework's earlier development
(backends and CUDA kernels, particle/basis models, the constraint pipeline,
fractions, plots) is in `git log`; the entries below are what landed immediately
before this file was introduced.

### Added

- `tabpwa.phasespace` — a mass-tree phase-space sampler for **any decay
  topology** (`generate_phsp`, `PhaseSpace`, `fix_beam`, `two_body_momentum`,
  `boost`): a slack-simplex proposal per vertex (weight exactly `Π q`, no window
  factor), a rejection ceiling SOLVED per vertex (closed form for 2 daughters, a
  stationary quartic for 3, a nested bracket/ternary table for 4+), and a mass
  pass before angles and boosts.
- `Fitter.generate_toy`, `Fitter.generate_toy_bg`, `Fitter.generate_phsp` —
  model-general toy data: accept–reject on `|A|²` with the ceiling returned for
  reuse (`stats.maxp` / `maxp=`), Poisson signal/background counts
  (`n·p`, `n·(1-p)`), the background sampled flat and accepted on `bg(x)`, and
  the normalisation as a separate phase-space sample (no `n_phsp` inside the
  toy).  The toy's rows are the model's whole event buffer, so every model shape
  works.
- `AmplitudeModel.toy_generator()` and `AmplitudeModel.mixing_inputs()` — a
  model supplies its own generator (the time-dependent model, for
  `frac`/`time`) and declares the extra per-event columns it reads.
- `data.extra_var.<name>.random` — the ONE definition of a sample's per-event
  law, as a numpy call: `{dist: uniform, low: 0, high: 10}`,
  `{dist: choice, a: [0, 1]}`, `{dist: exponential, scale: 1.519}`, …;
  `default:` alone is a constant, and an undeclared column falls back to
  `time ~ U(0, 10)` / `frac ∈ {0, 1}`.
- `Fitter.config_purity()` — the single purity reader shared by `set_data` and
  the toy (top-level → `data.purity` → `data.bg_frac`).
- `tabpwa.__version__`, `python -m tabpwa --version`,
  `python run_fit.py --version`, an installed `tabpwa` console script
  (`[project.scripts]`, the same entry point as `python -m tabpwa`), and this
  changelog.
- `tests/test_packaging.py` — every third-party top-level import in `src/tabpwa`
  must be a declared dependency (or live in an extra), the package must import
  with the optional backends absent, and the console entry point must report
  the version.

### Changed

- `pwa_build.generate_pwa_phsp` dispatches on topology: the frozen 3-body closed
  form is kept bit-identical (pinned by digest) and every other shape goes
  through the mass-tree sampler, compiled once per config.
- Dependency floors for the APIs the tree actually uses: `numpy>=1.17`
  (`default_rng`, `SeedSequence`), `scipy>=1.5` (`root_scalar`,
  `PchipInterpolator`, `CubicSpline`), `sympy>=1.6`, `matplotlib>=3.3`,
  `PyYAML>=5.1`, `cffi>=1.12` — previously every dependency was unversioned.
- `[project] readme = "README.md"` (the wheel metadata carried no long
  description) and a `MANIFEST.in`, so the sdist ships `CHANGELOG.md` and
  `README.md`.
- Kernel BINARIES are not packaged: the wheel/sdist carry the `.cu`/`.c`
  sources (the build key hashes the source, so a binary could not be validated
  on another machine anyway) and the per-machine `.hash` companions are no
  longer shipped.  Prebuilt binaries stay the AppImage's job.
- The `cuda` extra is gone: it only repeated `cffi`, which the base install
  already declares (the CPU kernels load their `.so` through cffi too).  CUDA
  needs `nvcc`, not a pip package, so `pip install tabpwa` is what a GPU host
  installs.

### Fixed

- `FlavourTagMix.build_event_data`'s legacy 4π branch accepted `frac`/`time` and
  silently dropped them (the mixing law was replaced by the defaults).
- `generate_phsp` could write a signal-MC builder's `bkg = 0` into the norm
  sample, leaving `N_b = 0` and making `set_data` refuse; the column is now the
  background SHAPE (`fbg`/`bg`/`bkg`, uniform by default).
- `phasespace.two_body_momentum` returned a spurious momentum for an impossible
  decay (it clamped the Källén product instead of gating on `M ≥ m₁ + m₂`).
- The time-dependent toy generated at a re-resolved parameter point (moving the
  masses by ~2× away from the requested `x0`) and its norm sample was integrated
  over a point measure in (`frac`, `time`) — measured 4.7 % off the law-integrated
  integral.  Both now use the sample's own point and law.
- An empty Poisson split in `generate_toy_bg` (both counts 0) now raises naming
  the draw instead of returning a buffer with no `mass` column.
- Selecting an ONNX backend without `onnxruntime` raises with the extra named
  (`pip install 'tabpwa[onnx]'`) instead of a bare `ModuleNotFoundError`.

### Docs

- README "Toy data" example (executed end to end), an AGENTS.md section for the
  toy layer, and the docstrings that still described deleted string modes
  (`time="uniform"`, `frac="random"`).
