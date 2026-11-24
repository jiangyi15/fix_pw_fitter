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
  v1.  Design `docs/mpi_service_design.md`, as built `docs/mpi_service.md`;
  fixture `tests/test_mpi_service.py`.

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
  deployment: `docs/slurm.md`; fixture `tests/test_mpi_backend.py`.

### Changed

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
