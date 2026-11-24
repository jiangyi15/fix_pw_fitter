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
