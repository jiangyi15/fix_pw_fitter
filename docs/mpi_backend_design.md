# MPI backend — DESIGN

**Kind: DESIGN.**  This is the contract, not a report on the code.  What shipped,
where, and the measurements live in **`docs/mpi_backend.md`**; where the two
disagree, this document is the intent and the implementation document records
the deviation.

Companion of `docs/remote_protocol_design.md`: that one is the **remote,
heterogeneous** deployment (a thin server, a wire protocol).  This one is the
**co-located** deployment — N GPUs in one host, or a homogeneous cluster —
launched by an MPI launcher, with no server process and no wire format.

They are alternatives, not layers: MPI ranks are the full framework, so they do
not deliver the "server without the analysis framework" property; the HTTP
protocol does not give you an MPI launcher's device binding or the
shared-filesystem data path.

---

## 1. When this is the right tool

| use MPI ranks | use the HTTP protocol (design: `docs/remote_protocol_design.md`) |
|---|---|
| one host with N GPUs, or a homogeneous cluster | hosts you do not administer, heterogeneous installs |
| the sample is on a shared filesystem | the sample lives on the client only |
| batch job at the launcher, no notebook | interactive client, servers started independently |
| same package/build on every rank | a thin server that need not carry tabpwa |

## 2. Programme shape

* **Every rank runs the same script.**  `mpirun`/`srun`/`mpiexec -n N python
  run_fit.py …`; there is no command channel and no rank-0 driver.  The
  optimisation is deterministic, so all ranks hold the same parameter vector and
  the only communication is the per-iteration reduction.
* The backend is selected as `mpi` (`backend: {name: mpi, …}` or
  `--backend mpi`) and wraps ONE **local** backend per rank: the transport is
  replaced, the compute is not.
* `local` MUST be a LEAF — a delegate/composer is refused, because (a) `auto`
  may resolve to a different kernel on each rank (availability differs) and
  (b) a wrapper's per-handle state (a fixed-coefficient mask, a Gram cache) is
  not forwarded to the ranks.  The `delegate` capability tag is the rule; the
  default (`local=None`) is the model's `default`, which must itself be a leaf.
* **The package MUST stay transport-agnostic.**  No module under ``tabpwa/``
  knows a rank, a launcher variable or a writer role: ``Fitter`` and every
  backend are transport-agnostic, and the MPI facts (rank, size) live inside
  ``mpi_backend`` only.  Consequence: OWNERSHIP OF OUTPUT belongs to the
  PROGRAM, not the library — a library user who saves or plots under
  ``mpiexec`` gets one write per rank.  The CLI (``run_fit.py``) is that owner
  today and asks the backend for its ``rank``; a program that should not deal
  with rank at all needs the driver/worker launch (§11.6), not a package-level
  rank concept.  Ordinary printing is not marshalled either: a merged launcher
  log repeats each line once per rank (use a per-task log).
* Every rank MUST construct the backend (`Split_type`/`Barrier` are collective)
  and MUST call `compute` the same number of times.

## 3. Rank → device

* The **node-local** rank is the device index (two nodes with 4 GPUs each both
  bind devices 0–3, not 0–7), obtained from
  `comm.Split_type(MPI.COMM_TYPE_SHARED)` — launcher-agnostic, no reliance on
  `SLURM_LOCALID` or `OMPI_COMM_WORLD_LOCAL_RANK`.
* `CUDA_VISIBLE_DEVICES` MUST be set **before the rank's CUDA context exists**
  (i.e. before the local backend is built, which is where a kernel may compile
  and initialise CUDA).  A value set by the LAUNCHER takes precedence, with
  these meanings:
  * empty                    — "hide every device"; leave it;
  * one entry               — this task's own device (Slurm
                              `--gpus-per-task=1`); leave it;
  * several entries         — a node-level list (Slurm `--gres=gpu:N`); select
                              `entries[node-local rank]`, so the ranks do not
                              all land on the first visible device.  A rank past
                              the list MUST fail with a message naming the list
                              length and the launch knobs.
* With no launcher value, the binding is `node-local rank`, validated against
  the visible GPU count: a rank with no device MUST fail with a message naming
  the rank count and the GPU count, never fall back silently to a CPU backend.
  A host with **no** NVIDIA device leaves the binding alone, so a CPU local
  backend stays usable.
* `n_workers`-style composition is refused in both directions: an MPI rank may
  not contain a `shard` (fork after `MPI_Init`, and CUDA in the parent), and an
  `mpi` backend may not live inside a `shard` worker.

## 4. Data path

The sample reaches each rank as a CONTIGUOUS slice, and the slice boundaries
are the backend's own (§5), shared by loader and backend.  Three ways to get
there:

| option | peak RAM | requires |
|---|---|---|
| **A** every rank loads the full sample, keeps its slice | N × sample, then 1/N live | nothing |
| **B** rank-aware load (`np.load(mmap_mode="r")` then slice) | ≈ 1 × sample per node | a `rank=`/`size=` (or `slice=`) hook on the loader |
| **C** rank 0 loads and scatters chunks | ≈ 2 × sample transiently | pickling a command channel — this is `shard`'s design and what MPI should avoid |

The DESIGN target is **B**; **A** is acceptable only as a temporary fallback
and MUST be documented where users see it, because `N ×` the sample per node is
a real limit for large samples.  The choice belongs to the loader layer, not to
the backend: the backend always keeps `[bounds[r], bounds[r+1])`.

## 5. Slicing

Reuse `shard_backend.split_offsets(ne, weights, align)` — ONE implementation,
shared with `shard`:

* contiguous rows, **weighted** (equal event count is not equal work for a
  non-uniform sample);
* `align = resolution_size` when a grouped objective is in use, so a log-sum
  group never straddles two ranks; the value SHOULD be derived from the local
  backend (checking, in order, a public `resolution_size`, a private
  `_resolution_size`, and the constructor spec the caller wrote), and may be
  overridden explicitly;
* weight lists MUST be non-negative with a positive sum, and their length MUST
  equal the rank count;
* a slice with **no events** MUST be REFUSED with a message naming `n`, `align`
  and the weights: the numpy/flat leaves can compute a zero contribution, but
  the `pwa` kernels cannot load an empty sample, and `split_offsets` can produce
  an empty slice without user error (`align`/`resolution_size` above `ne/2`, or
  a skewed weight list).  The refusal MUST be identical on every rank (it is a
  function of the global split), so it can abort the job cleanly.

## 6. Per-iteration protocol

All ranks hold the same parameters, compute their slice, and reduce:

```
Q_r, grads_r, P_r = local.compute(params, slice_handle, norm, return_p)
Q      = allreduce(Q_r, SUM)
grads  = {k: allreduce(grads_r[k], SUM)}          # incl. grads["norm"]
P      = allgather(P_r)                           # ONLY when return_p
```

* `Q`, every gradient block and `grads["norm"]` are sums over **disjoint**
  events, so the reduced value is the full-sample objective up to association
  (§9).  Only host arrays cross ranks (parameter blocks, one scalar per rank);
  CUDA-aware MPI is not required.
* `P` is per-event and is never requested by a fit (`get_nll_raw` and
  `_compute_norm_batched` pass `return_p=False`).  When it IS requested it MUST
  be **allgathered**, not gathered to rank 0: every rank must see the same full
  array, or a caller that drives the backend per batch (the toy generators, a
  plot on each rank) desynchronises the ranks.
* **Participation must be rank-invariant.**  A null gradient block, a missing
  `P`, or any other rank-local condition MUST NOT decide whether a collective
  runs; the ranks agree on such a condition first (one small broadcast) and a
  disagreement ends the job.
* **The parameters are broadcast once**, on the first `compute` of a process,
  from rank 0.  `initial_values()` is unseeded per process, so a file-driven fit
  would otherwise evaluate a different point on every rank and the summed
  gradient would be meaningless.  After that no broadcast is needed: identical
  inputs and identical reduced results keep the ranks in lockstep.  An
  objective that is non-deterministic across ranks (wall-clock line search, RNG
  inside an evaluation) MUST broadcast every iteration instead — a documented
  switch, not a heuristic.
* **Sample generation is a single-process operation.**  The toy/phase-space
  generators drive the backend per batch, so on an MPI fitter they become a
  collective (lockstep, correct, but the same work on every rank).  The
  documented workflow is: generate once (serial process or a file), then
  `set_phsp`/`set_data` the arrays, which are sliced per rank.

## 7. Failure semantics

* Any rank-local failure MUST end the JOB (`comm.Abort(code)`) with the rank id
  and the message, at EVERY site that can fail before a collective — device
  selection, the first build and its barrier, `load_data`, and the reduction
  (not only `compute`).  An unguarded raise leaves the peers blocked in a
  collective until the launcher kills them.
* There is no retry and no degraded mode: a lost rank means an incomplete sum,
  which would silently change the likelihood.  There is no graceful checkpoint
  after `Abort`; use the caller's interrupt path before the deadline.
* The first kernel build MUST be serialised across ranks (rank 0 builds, then a
  barrier) so N ranks sharing one install never compile the same library
  concurrently.

## 8. Dependencies and launch

* `mpi4py` is an OPTIONAL dependency (`tabpwa[mpi]`), imported lazily, and a
  missing runtime MUST name the extra, never raise a bare `ModuleNotFoundError`.
* The MPI library must match the launcher (`mpirun`/`mpiexec`/`srun`); no
  CUDA-aware build is needed.
* Dynamic process management (`MPI_Comm_spawn`, starting ranks from a notebook)
  is NOT part of the design: support is uneven across MPI implementations, and
  the architecture wants every rank to run the same program.  A notebook runs
  the fit through a shell/launcher call.
* A workstation with fewer GPUs than ranks is a test configuration
  (`--oversubscribe` plus a CPU local backend), not a tuning target.

## 9. Precision

* The protocol guarantees **structural** equality with a serial run: the same
  gradient block names, shapes, dtypes, `grads["norm"]` iff `norm`, `P` iff
  requested.
* The values are sums of per-slice values, so they differ from a serial
  evaluation only by floating-point association.  Conformance tests SHOULD use
  `rtol = 1e-12`; a single rank MUST be bit-identical to the local backend.
* Cross-rank result determinism is assumed by the lockstep model.  If an MPI
  implementation ever returns different bits on different ranks, the model
  breaks — then rank 0 MUST broadcast the reduced result too.

## 10. What to reuse (not copy)

* `shard_backend.split_offsets` — the one weighted, `align`-aware split;
* the reduction of shard results (`Q + Σ`, gradients `+`) — the same arithmetic
  as the remote pool;
* the per-worker `device=` pattern of `shard` (it also sets
  `CUDA_VISIBLE_DEVICES`);
* `backends/core.py` (`ComputeBackend`, `register_backend`, capability tags) as
  the seam, and the `delegate` tag as the "not a leaf" rule.

## 11. Open questions

1. **Rank-aware loader API**: a `rank=`/`size=` pair on `Fitter.load_dataset`,
   or a `slice=` on `set_data`/`set_phsp`?  The former also avoids parsing the
   file N times.  This is the `N×` memory fix and the main remaining design
   item.
2. **Mixed backends per rank** (a CPU rank among GPUs): refused, or allowed when
   the caller says so?
3. **Threads**: should `shard` eventually offer a thread mode so one node can
   fill more than one GPU per MPI rank without forking?
4. **Launch helper**: a `python -m tabpwa.mpi -n N -- …` wrapper (resolving
   `mpiexec`/`srun`) versus keeping the launcher the user's business.  See
   `docs/slurm.md` for the deployment variants.
5. **Result broadcast**: if a future kernel is not deterministic across ranks,
   broadcast the reduced result each iteration (one more collective) — decide
   whether to make that the default before it is needed.
6. **Internal launch / "shard-like" mode**: let the backend start the MPI job
   itself (a driver process + workers) so the program runs ONCE and no launcher
   is typed.  Two forms: ``MPI_Comm_spawn`` (MPICH-only) or a subprocess
   ``mpiexec`` plus a command channel.  Either needs a command channel and
   either data shipping or a prepared file, and it loses the "every rank runs
   the same deterministic program" simplicity that makes the current reduction
   and failure model defensible — it becomes shard-over-MPI.  Against that: the
   the only package-level difference left is the launcher line
   is one launcher line.  Recommendation: do NOT build the driver/worker mode
   unless a user with no scheduler has to start fits from a notebook; the
   prepared-artifact workflow (§4 option B) plus ``srun``/``mpiexec`` covers the
   batch case.