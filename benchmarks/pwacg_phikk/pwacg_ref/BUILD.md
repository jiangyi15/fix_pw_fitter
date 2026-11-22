# Building and running the reference (PWACG)

The reference is `caihao/PWACG` (MIT).  Its sources are **not vendored** here —
clone them, since the benchmark must time *their* code rather than a
reimplementation, and a copy would be one more thing to keep honest:

```bash
git clone git@github.com:caihao/PWACG.git      # SSH: plain HTTPS hangs on this host
git -C PWACG checkout f516d21                  # 2026-05-18, the revision measured
```

## What the environment needs

**A Python with JAX and a working CUDA device.  Nothing else.**  On the box these
numbers were taken with it was a conda env holding jax 0.11.0 / CUDA 13, where
`jax.devices()` returns `[CudaDevice(id=0)]` (RTX 5070 Ti).  Any interpreter that
satisfies `python -c "import jax; print(jax.devices())"` and shows a CudaDevice
works.

Despite what `pyproject.toml` declares (`jax, numpy, pandas, iminuit, pynvml,
matplotlib, tabulate, root`), the timed path needs **no ROOT**, **no `iminuit`**,
**no `pynvml`**, **no network** and **no API key**.  Installing them would add
surface that can move numbers we are not measuring; the two that their generated
module imports at *module scope* are stubbed instead (`stubs/`), and both raise
if anything actually calls them.

## Two shims, and why they are outside the clone

1. **`jax.config`** — `Tensor/CacheTensor.py:7` (and `dmultiprocess`) do
   `from jax.config import config`.  Modern JAX has no importable `jax.config`
   *module*; the modern call is `jax.config.update(...)`.  `jax_legacy_config.py`
   registers a module alias in `sys.modules`, so the legacy import resolves and
   the clone stays byte-identical (`git status` clean) — a shim you can delete
   the day they update, instead of a fork you have to re-apply.
2. **`iminuit` / `pynvml`** — imported at module scope by the generated fit
   object, used only by its Migrad driver and its multi-GPU bookkeeping.  The
   stubs raise on use, so a fit that reached them fails loudly rather than
   getting a fake number.

Import `jax_legacy_config` **first**, and put `stubs/` on `sys.path`.

## The pipeline, in the order that works

```
Momentum_kk.npz  ──their CacheTensor──▶  data/real_data/{phi,f,phif0,phif2}_kk.npy
                                          │
                                          ▼
                              create_all_scripts.py  ──▶  rendered_scripts/fit_object_kk.py
                                          │
                                          ▼
                                       PWAFunc(cdl=…)  ──▶  data_likelihood_kk / mc_likelihood_kk
```

`bench_ref.py` drives all of it:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python pwacg_ref/bench_ref.py \
    --pwacg /path/to/PWACG \
    --npz ../data/Momentum_kk.npz --mc ../data/Momentum_kk_phsp.npz \
    [--reuse] [--mc-sizes 10000,100000,1000000] [--repeat 10]
```

It copies the two `.npz` files into `<clone>/data/{real_data,mc_truth}/`, runs
their `Tensor/CacheTensor.py` to produce the per-event tensors, runs
`create_all_scripts.py` **if** `rendered_scripts/` is missing, imports the
generated module, and times eager / jitted / gradient / `value_and_grad` on the
data term and the MC term.  `--reuse` skips the tensor step for a clone that
already holds them.

## Traps, each measured

* **Order.**  `create_all_scripts.py` fails at `prepare_all` with
  `FileNotFoundError: data/real_data/phi_kk.npy` — the *tensors* are a
  precondition of code generation, and only `phi_kk.npy` / `f_kk.npy` are
  strictly needed (`phif0`/`phif2` can be absent and it still succeeds).  So:
  tensors → codegen → import.  `bench_ref.py` does exactly that.
* **Codegen is offline, and only for the shipped model.**  With no
  `DEEPSEEK_API_KEY` the generator logs `LLM client unavailable` and copies
  `.llm_cache/golden/<artifact>/latest/file_content.py` verbatim — the emitted
  `fit_object_kk.py` is byte-identical to the golden artefact.  That means the
  current tree cannot render a *different* model without a key and the LLM path;
  the jinja2 templates are gone.
* **Their multiprocess driver does not work with their own tensors.**
  `run/fit_kk.py`'s loader does `array_split(..., axis=1)` on a 1-D array and
  indexes `axis=2`, and `data/weight/weight_kk.npy` is never produced;
  `Control.run` also references an unimported `config`.  `PWAFunc` is fine — it
  wants 1-D `Sbc` exactly as `CacheTensor` writes it, which is how their own
  `tests/` feed it.
* **XLA preallocation.**  With JAX's default the sampler reads ~9.1 GiB of
  preallocated memory that is not the model's.  Set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` and read
  `jax.devices()[0].memory_stats()`.
* **Peak memory includes the precompute** when the tensors are built in the same
  process (measured 2.7 GB there vs 138 MB at 10 k and 745 MB at 1 M for the
  model alone).  Use `--reuse` in a second run when the number is meant to be
  the model's resident footprint.
* **First call ≠ steady state.**  One XLA compile per signature: 0.32 s (data
  likelihood), 1.1 s (its gradient), 0.27 s (MC), 7–12 s at 1 M events.  Every
  number in `../README.md` is steady state, min-of-N, after two warm-ups.

## The array contract (verified against their sources)

`Momentum_kk.npz` holds exactly four keys, each `(n_events, 4)`, with columns
**(px, py, pz, E)** — spatial first, energy last: `Tensor/CacheTensor.py`
flips the spatial part and dots with the metric `diag(-1,-1,-1,1)`.

| key | meaning | our `dat_order` |
|---|---|---|
| `Kp`, `Km` | the φ pair | `Kp1`, `Km1` |
| `Pip`, `Pim` | the R pair | `Kp2`, `Km2` |

`Pip`/`Pim` are a ππ misnomer inherited from an earlier model.  `data_info`'s
`size` is unused; `begain`/`end` slice rows (`end` exclusive) and `slices` is the
`array_split` chunk count.

## The published sample

The repo's **`v1.0.0` "Data" release** carries `data.zip` (788 MB, the demo
sample).  `../make_data_from_ref.py` converts it to this framework's arrays, so
both sides run on their events:

```bash
python make_data_from_ref.py --zip /path/to/data.zip
```

Getting that archive from this host is the one thing no recipe here can fix: the
release download is throttled to ~30 kB/s (≈7 h), while `github.com` and
`api.github.com` answer normally and `pypi`/`jsdelivr` do not answer at all.
The bridge is verified by a round trip instead — the benchmark's own generated
`Momentum_kk.npz` is read back through it and lands inside the tables.
