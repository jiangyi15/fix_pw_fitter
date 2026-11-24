# MPI service backend — DESIGN (driver + subprocess workers)

**Kind: DESIGN.  Implemented (v1): `docs/mpi_service.md`.**  Normative for the
backend that keeps ALL MPI state inside a child process tree.  Companions: `docs/mpi_backend_design.md`
(the all-ranks mode, implemented), `docs/remote_protocol_design.md` (a network
server, draft).

## 1. The rule this design satisfies

No module under `tabpwa/` may know a rank, a launcher variable or a writer role
(the all-ranks mode leaks the last one to the program: a library user saving
under `mpiexec` writes once per rank).  The only way to keep that rule AND make
the program run once is to put the whole MPI job in a CHILD process:

```
driver process                          child process tree
  Fitter + package (transport-agnostic)       mpiexec -n N  python -m tabpwa.mpi_worker
  mpi_service backend  ── pipe/socket ──▶     rank 0: command loop ─┐
  no mpi4py, no rank, no env probe                                  │ bcast(op)
                                              every rank: the ORDINARY `mpi` backend
                                              (slice, device, build barrier, allreduce)
```

The child is deliberately NOT a second implementation: every rank builds
`create_backend({"name": "mpi", "local": …})` — the all-ranks backend of
`docs/mpi_backend_design.md` — and the worker loop only adds the driver channel
on rank 0.  Slicing (`split_offsets`), the rank→device rule, the serialised first
build, the `allreduce`/`allgather` reduction and the abort policy are that
backend's, once.

Consequences: the program writes output once because there is only one program
process; the package stays transport-agnostic; `mpi4py` is imported only in the
child; the child's ranks never run user code (they compute).

## 2. Process model and launcher

* The driver starts the child LAZILY on the first `load_data` (not in the
  constructor: a `Fitter` must be constructible offline).
* The child is `mpiexec -n N python -m tabpwa.mpi_worker <spec>` (or `mpirun`),
  resolved from `TABPWA_MPI_LAUNCHER`, then `mpiexec`, then `mpirun`.
* **Nesting is refused, not attempted.**  If the driver is already inside an MPI
  or scheduler step (`PMI_RANK`/`PMIX_RANK`/`OMPI_COMM_WORLD_RANK`/
  `SLURM_STEP_ID` present), starting another launcher is wrong; the backend MUST
  raise and name the all-ranks mode (`backend: mpi`, the idiomatic `srun` form)
  as the alternative.  A service that is its own job is the network protocol's
  case, not this one.
* Device binding is the child's business: `mpi_worker` builds the local backend
  with the same `device`/`align` rules as the all-ranks backend, from its own
  `COMM_WORLD`.  The driver never sets `CUDA_VISIBLE_DEVICES`.

## 3. Channel and framing

* One full-duplex channel to rank 0 (the gateway): a **Unix-domain socket** whose
  PATH is passed in argv.  stdio cannot be used — Hydra/orted multiplex the
  child's stdin (measured: `mpiexec` aborts with "process reading stdin too
  slowly"), and a `socketpair` fd is not reliably inherited through the
  launcher.  No network in v1.
* Framing is the envelope already specified for the remote protocol: a
  length-prefixed zip of `.npy` members plus one JSON `__meta__` (tuples tagged,
  `{"__array__": …}` references), so the two designs share one encoding rather
  than inventing a second.  Requests are serialised: one in flight.
* Every message carries `"protocol": 1`; the handshake checks it.

## 4. Handshake

On start, the worker answers:

```json
{"protocol": 1, "backend": "cuda_v4_pwa", "tabpwa": "0.1.0",
 "n_ranks": 4, "local": "cuda_v4_pwa", "device_plan": ["0","1","2","3"]}
```

The driver validates `protocol` and that `backend` matches the requested local
spec; a mismatch aborts the child and raises.  `device_plan` is informational.

## 5. Operations

| op | driver → gateway | gateway → driver |
|---|---|---|
| `load` | `{"kind": "data"\|"phsp", "path"?: str, "arrays"?: {…}, "hash"?: str, "model"?: str}` | `{"n": int}` |
| `compute` | `{"params": …, "norm": float\|null, "return_p": bool}` | `{"Q": float, "grads": …, "none": [...]}` (+ `P`) |
| `free` | `{"kind": …}` | `{"ok": true}` |
| `shutdown` | — | the child finalises and exits 0 |

The loop is: rank 0 receives the op, `bcast`s it to the ranks (the op payload
includes the kernel config on the first `load`), and EVERY rank calls the same
method on its own `mpi` backend — `load_data` / `compute` / `free`.  The gateway
does not slice, bind devices or reduce anything itself; it forwards the
returned value from rank 0's backend.  The compute contract is therefore the
all-ranks one by construction (`docs/mpi_backend_design.md` §6/§9): `Q`, every
gradient block and `grads["norm"]` are sums; `P` only when `return_p`.

## 6. Data paths

Two ways to get the sample to the child; both feed the same `load_data(full_sample)` on every rank; the backend keeps the
rank's slice and drops the rest.  That is the all-ranks data path (option A) and
its `N×` transient is inherited here too — the rank-aware loader (option B) is
the single follow-up that removes it for BOTH modes.

* **ship** — rank 0 receives the arrays once and `bcast`s them; required for an
  in-memory sample (a generated toy).  One channel transfer, then one broadcast.
* **reference** — the driver sends `path` + content `hash`; rank 0 `bcast`s the
  reference and every rank opens the file (`np.load(..., mmap_mode="r")`).  No
  sample bytes on the channel; the file must be visible to the child.

The backend MUST verify a `reference` load: same hash on the driver's file and
in the child (the child can hash what it opens, or the driver sends the hash of
what it has and the child re-hashes).  Mismatch is an error, not a re-read.

## 7. Slicing and the grouped objective

Slicing is NOT re-implemented here: the inner `mpi` backend calls
`split_offsets(n, weights, align)` in every rank, derives `align` from the local
objective and refuses an empty slice — the rules of
`docs/mpi_backend_design.md` §5.  The worker must therefore pass the same
`align`/`weights` through to the inner backend, and nothing else.

## 8. Lifecycle and failure

* The child lives as long as the backend.  `free()` sends `shutdown`, waits
  (bounded), then `terminate`, then `kill`; `__del__` and an `atexit` reaper do
  the same, so a crashed driver cannot orphan an `mpiexec` tree.
* `SIGINT` in the driver is forwarded to the child (`KeyboardInterrupt` in the
  fit still checkpoints).
* If the child exits, every later call raises with the tail of its stderr; the
  backend does NOT restart silently (a restart would hide a compute failure).
* A rank-local failure calls `comm.Abort`, the gateway dies, the pipe closes,
  and the driver sees an exited child — the same information as the all-ranks
  mode, now at the driver.
* No retry, no degraded mode.

## 9. Schedulers

* **Inside a Slurm allocation**: start the PROGRAM with `srun` (the all-ranks
  `mpi` backend) — this service must not spawn a launcher inside a step
  (§2 refuses).  Alternatively, the service is its own job and the driver
  connects over a network protocol (`docs/remote_protocol_design.md`).
* **Workstation / server / login node**: the service mode is the convenient
  one; no `mpiexec` on the user's command line.
* **Notebook**: allowed (the child is a subprocess), unlike `MPI_Comm_spawn`
  (MPICH-only and needs MPI in the driver).

## 10. Performance

* Per iteration: one pipe hop + one `bcast` + compute + one `allreduce` + one
  pipe hop.  Messages are KB; the fit's `return_p=False` path sends no P.
* The sample crosses ONCE (ship) or not at all (reference).
* Startup: `mpiexec` plus the child's import and (first time) kernel build; the
  child can stay alive across fits if the backend is reused.

## 11. Driver-side API

```yaml
backend:
  name: mpi_service
  local: cuda_v4_pwa          # ONE concrete kernel (a leaf), as in the all-ranks mode
  n_procs: 4                  # mpiexec -n
  launcher: auto              # auto | mpiexec | mpirun
  data: auto                  # auto | ship | reference
  timeout: 600
```

`n_procs` is REQUIRED (the driver cannot infer the GPUs it may take).  The
backend registers as `mpi_service` with tags `delegate` + `external`, is a
`ComputeBackend`, and has the same lifecycle (`load_data`/`compute`/`free`).
It does not expose a rank.

## 12. Contrast

| | all-ranks `mpi` | `mpi_service` (this) | `shard` | remote protocol |
|---|---|---|---|---|
| program processes | N (every rank) | **1** | 1 (+ children that never run the program) | 1 |
| rank in the package | no | no | no | no |
| output once | only via the CLI guard | **yes** | yes | yes |
| MPI state | in every rank | **only in the child** | none | none (server side) |
| scheduler | idiomatic (`srun`) | not inside a step | one node | anywhere |
| sample transfer | per rank re-load | once, or by reference | pickled chunks | upload once |

## 13. Open questions

1. **Framing reuse**: adopt the remote protocol's envelope verbatim (recommended)
   or a simpler pickle/length protocol for the pipe?
2. **Data default**: `ship` (works for generated samples) or `reference` (best
   for large files) when both are possible — auto-detect by a size threshold?
3. **Persistence**: should the child survive `free()` so a second `Fitter` can
   reuse it (a per-named-service registry), or die with the backend?  (v1: dies.)
4. **Multiple clients**: one gateway per child, one client per gateway in v1.
   A shared service is the network protocol's job.
5. **`P` on demand**: gather to the gateway only (the driver is the only reader)
   — simpler than the all-ranks allgather.
6. **Launcher detection**: exact list of "already inside a step" variables, and
   whether `srun` (not `mpiexec`) should ever be launched by the service.
7. **Windows**: no `socketpair`/`fork` assumptions; v1 is POSIX.

## 14. Implementation (shipped vs open)

```
src/tabpwa/mpi_worker.py            worker entry: handshake + command loop; builds
                                    the ORDINARY `mpi` backend on every rank and
                                    forwards load/compute/free to it
src/tabpwa/backends/mpi_service_backend.py
                                    driver side: lazy Popen, envelope over the pipe,
                                    lifecycle/reaper, no mpi4py import
tests/test_mpi_service.py           handshake; ship vs reference parity against the
                                    local backend; a fit's NLL; child death raises;
                                    nesting refused; no mpi4py/rank in the parent
```

Shipped in v1: the socket channel, the handshake, `ship` loading, `compute`,
`free`, `shutdown`, the nesting refusal, the atexit reaper, and the gates above
except the reference-load one (that path is still open).  As built:
`docs/mpi_service.md`.