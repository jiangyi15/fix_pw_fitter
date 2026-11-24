# MPI backend — IMPLEMENTATION (as built)

**Kind: IMPLEMENTATION.**  The contract is **`docs/mpi_backend_design.md`**;
this file records what the tree actually does, where it lives, and the
measurements.  If the two disagree, the design is the intent and this file is
the bug.

```
src/tabpwa/backends/mpi_backend.py      the backend (registered `mpi`)
tests/test_mpi_backend.py               14 gates (7 of them under `mpiexec`)
pyproject.toml                          optional extra: mpi = ["mpi4py"]
docs/slurm.md                           deployment: srun vs mpiexec, GPUs, signals
docs/mpi_backend_design.md              the design
```

## 1. Registration and the spec

`MPIBackend`, registered `mpi` with tags `delegate` + `external` (universal: it
serves every amplitude model).  `external` is a capability, not a name: the
backend needs a process the test did not start, so the capability sweeps skip
it.

```yaml
backend:
  name: mpi
  local: cuda_v4_pwa       # ONE concrete kernel (a leaf); default = model's default
  device: auto             # auto | None | int   (default: auto)
  align: null              # null = derive from the local backend's resolution_size
  weights: null            # null = equal; else one non-negative weight per rank
  build_first: true        # rank-0-first build barrier (default: true)
```

`local` may be a name or a spec dict, never a backend instance and never a
`delegate` (see §3).

## 2. What one compute does

```python
# first call of a process:
params = comm.bcast(params, root=0)          # one collective, then never again
q, grads, p = local.compute(params, slice_handle, norm, return_p)
# agree on null blocks (rank-invariant participation), then reduce:
q            = allreduce(q, SUM)
grads[k]     = allreduce(grads[k], SUM)      # incl. grads["norm"]
p            = allgather(p)                  # only when return_p
```

`load_data` slices `[bounds[rank], bounds[rank+1])` with
`shard_backend.split_offsets(n, weights, align)` and hands the slice to the
local backend.  `free` frees the local handle; `__del__` calls it.

## 3. Behaviour the design requires, as implemented

| rule | implementation |
|---|---|
| `local` is a leaf | `_require_leaf` checks the `delegate` tag (refuses `auto`, `integrated*`, `fixed_coeff`, `shard`, `mpi`); `local=None` → the model's `default` |
| no nesting | `shard` inside a rank refused; `mpi` inside a `shard` worker refused through the marker `shard_backend._worker_main` sets |
| device binding | `device_index(node-local rank, device, gpu_count, CUDA_VISIBLE_DEVICES)`; launcher env wins — empty = hide all, one entry = the task's device, several = a node-level list indexed by the node-local rank; no launcher value → node-local rank, validated against the count; no GPU → leave unbound |
| no concurrent build | rank 0 builds, `Barrier`, the others build, `Barrier` |
| `align` | derived from the local backend in this order: public `resolution_size`, private `_resolution_size`, the spec's `resolution_size`; explicit `align=` wins |
| empty slice | refused in `load_data` with `n`/`align`/`weights` in the message (the `pwa` kernels cannot load one) |
| failure | every rank-local site (device, build/barrier, `load_data`, reduction) replays its exception through `comm.Abort(1)`; `compute` reports `rank i/N failed: …` |
| output ownership | the PACKAGE is transport-agnostic (no rank/launcher logic outside `mpi_backend`). The CLI (`run_fit.py`) asks `fitter.backend.rank` and writes on rank 0 only; a LIBRARY program under `mpiexec` writes once per rank — the driver/worker launch (design §11.6) is what would remove the distinction |

## 4. Measured evidence

Two ranks, `numpy_pwa`, 400 events, one slice per rank:

```
events/rank=[200, 200]
Q  mpi=170.422779474070  serial=170.422779474070  d=0.000e+00
grads max|d|: ck 1.4e-14, m0 1.0e-12, g0 1.7e-13
```

Two ranks with the real CUDA backend on ONE host (the box has one GPU, so the
ranks are oversubscribed — correctness only, no speed claim):

```
rows/rank=200
Q  mpi=170.4227792440  serial=170.4227792440  d=1.99e-13
grad max|d|: ck 3.6e-15, m0 1.1e-13, g0 5.7e-14
```

Grouped objective (`cpu_v5_pwa`, `resolution_size=4`): with `align` derived
correctly, the gate's 30-event sample (12/18 split) matches the serial grouped
fit to <1e-9 (asserted); a review probe at 27 events read 3.6e-15.  Before the
fix the derivation missed the v5 family's private `_resolution_size`, silently
used `align=1`, and that probe read 1.6 % in `Q` and 9.2 % in a gradient with no
exception.

`generate_toy` driven through the MPI fitter: the accepted sample is identical
on both ranks and in a serial run (SHA-256 of the masses) — the lockstep the P
allgather buys.

A refused rank used to leave the peers in a `Barrier` for the launcher's whole
timeout (measured 45 s for the device refusal, 60 s for a failing `load_data`);
with `Abort` the job ends immediately — the gates require < 30 s (observed
~1–2 s).

The development box: `mpi4py 4.1.2`, **MPICH 4.3.2** (`ch4:ofi`, Hydra) — note
this is not a Slurm-aware MPICH build; see `docs/slurm.md`.

## 5. Gates

`tests/test_mpi_backend.py` — 14 tests; the SEVEN that launch `mpiexec` skip
when no launcher is on `PATH` (the pure rules still run):

| gate | pins |
|---|---|
| `device_index` rules | None / auto / int / empty / one entry / node-level list / past-the-list / no-GPU |
| node-level list binds | `CUDA_VISIBLE_DEVICES="3,5"` → `"3"` on local rank 0 |
| one rank bit-identical | every block and `P` `array_equal` against the local backend |
| `align` derivation | `numpy_flat` (public), `cpu_v5_pwa` (private), `numpy_pwa` (none → 1) |
| composer refused | `auto`, `integrated_pwa`, `fixed_coeff` |
| `shard` inside a rank | refused (fork after `MPI_Init`, CUDA in the parent) |
| `mpi` inside a `shard` worker | refused via the worker's `TABPWA_SHARD_WORKER` marker |
| 2-rank parity | `Q`, grads and `P` vs a serial backend, `P` concatenated in order |
| grouped 2-rank parity | `align == 4` **and** the full `Fitter.get_nll` vs a serial grouped fit |
| toy lockstep | both ranks and a serial run produce one SHA-256 |
| unseeded start | `initial_values(seed=None)` on two ranks → identical `Q` |
| no-GPU refusal ends fast | patched GPU count, `< 30 s`, message in stderr |
| empty slice refused | `weights=[1,1000]`, `< 30 s`, "owns no rows" |
| fitter path | `set_phsp`/`set_data`/`get_nll` incl. `grads["norm"]` vs serial |

## 6. Packaging

`pip install 'tabpwa[mpi]'` (`mpi4py`); the import is lazy — `import tabpwa`
does not require MPI, and `mpi4py` is not a base dependency
(`tests/test_packaging.py`).

## 7. Limitations / follow-ups

* **`N×` memory**: v1 keeps design option A — every rank loads the full sample
  before slicing.  The rank-aware loader (option B) is the open design item and
  the limit to watch on multi-node jobs.
* **No launch helper**: the user runs `mpiexec`/`srun`; `docs/slurm.md` gives
  both variants.  An internal launch (`MPI_Comm_spawn`) is deliberately out of
  scope.
* **Output is the program's job, not the library's**: the package knows no
  rank.  Under `mpiexec`, a library program writes once per rank and prints N
  times; the CLI guards its writes, and making the program run exactly once
  needs the driver/worker launch — an open design question
  (`docs/mpi_backend_design.md` §11.6), deliberately not built.
* **No mixed backends per rank** (a CPU rank among GPUs) and no benchmark
  against `shard` yet.
* **Result determinism** across ranks is assumed; if a kernel ever disagrees
  bitwise between ranks, the design says broadcast the reduced result too.

## 8. What the review changed

The implementation was reviewed before landing; the must-fix items and their
gates are in §3/§5: `align` silently 1 for the v5 family (silent 1.6 %),
rank failures outside `compute` hanging the job, an empty slice documented as
supported while the `pwa` kernels reject it, and a file-driven fit starting
from a different `x` on every rank.  The should-fix set added rank-invariant
participation, the node-level `CUDA_VISIBLE_DEVICES` rule, the `shard`-direction
refusal, rank-0 writes in `run_fit.py`, and the `external` tag in the closed
vocabulary.