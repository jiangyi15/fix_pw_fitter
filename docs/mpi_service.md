# MPI service backend — IMPLEMENTATION (as built)

**Kind: IMPLEMENTATION.**  The contract is
`docs/mpi_service_design.md`; this file records what the tree does, with the
measurements and gates.

```
src/tabpwa/_ipc.py                       framed messages (length + npz, JSON meta,
                                         array skeleton, tuples tagged)
src/tabpwa/mpi_worker.py                 child entry: handshake + command loop;
                                         builds the ORDINARY `mpi` backend per rank
src/tabpwa/backends/mpi_service_backend.py
                                         driver: lazy `mpiexec` child, Unix-socket
                                         channel, lifecycle; NO mpi4py, NO rank
tests/test_mpi_service.py                8 gates
```

## 1. Registration and the spec

`mpi_service`, tags `delegate` + `external`, universal.  `local` must be a LEAF
(the same `delegate`-tag rule as the all-ranks backend — the check is that
backend's own static method, called from here, so the two cannot drift).

```yaml
backend:
  name: mpi_service
  local: cuda_v4_pwa       # ONE concrete kernel
  n_procs: 4               # REQUIRED — the driver cannot infer the GPU budget
  device: auto             # auto | None | int (passed to the worker)
  align: null              # null = the local objective's resolution_size
  weights: null            # one non-negative weight per rank
  launcher: auto           # auto | mpiexec | mpirun | PATH; TABPWA_MPI_LAUNCHER
  timeout: 600
```

The constructor is OFFLINE (no child): `Fitter(...)` works with no `mpiexec` on
the box; the child starts on the first `load_data`.  The constructor REFUSES
when the process is already inside an MPI/scheduler step (`PMI_RANK`,
`PMIX_RANK`, `OMPI_COMM_WORLD_RANK`, `MV2_COMM_WORLD_RANK`, `SLURM_STEP_ID`) and
names the all-ranks `mpi` backend as the alternative.

## 2. The channel is a Unix socket, not stdio

Hydra/orted own the child's stdin and multiplex it (measured: using `mpiexec`'s
stdin for the protocol makes Hydra abort with *"process reading stdin too
slowly; can't keep up"*).  So the driver:

1. creates a private directory (`tempfile.mkdtemp`) and a `AF_UNIX` socket in it;
2. listens, then starts `mpiexec -n N python -m tabpwa.mpi_worker --local … --addr <path>`;
3. `accept()`s the connection from rank 0 and talks the frames of
   `tabpwa/_ipc.py` over `conn.makefile("rwb")` with the child's socket timeout.

The child's stdin/stdout are `DEVNULL`; every rank redirects fd 1 to stderr, and
the driver drains stderr into a bounded tail that is appended to every raised
error.  The socket path must fit the `AF_UNIX` limit (the driver refuses a
`TMPDIR` that makes it too long).

## 3. What the child does

```
rank 0:  connect → handshake {protocol, n_ranks, tabpwa, local}
         loop: read one frame → comm.bcast((meta, arrays)) → execute → reply
ranks 1..N-1: bcast the same op, execute, never touch the socket
```

* `load`: the first one carries the kernel config (JSON skeleton + arrays); every
  rank builds `create_backend({"name": "mpi", "local": …, "device": …}, kc,
  model=…)` — the all-ranks backend's device binding, first-build barrier and
  slicing, once.  Then `inner.load_data(data)` on every rank; rank 0 replies `n`.
* `compute`: `inner.compute(params, handle, norm, return_p)`; rank 0 packs `Q`,
  the gradient blocks (arrays) and the scalar/`None` blocks.
* `free`: drops the handle on every rank.
* `shutdown`: rank 0 replies, the loop ends, MPI finalises.
* A rank-local failure prints its traceback, rank 0 sends `{"error": …}` when it
  can, and the group `comm.Abort`s.  The READ is inside that guard too: a
  truncated frame (a driver killed mid-`write`) would otherwise kill rank 0
  while its peers wait in the `bcast` — an N-rank job holding N GPUs with no
  error anywhere (measured before the fix: `mpiexec` still running after 45 s,
  `EOFError: ipc: truncated message`; after it: `Abort(1)`, exit 1, 0.0 s).
  So the driver sees the reply, or an error, or the EOF plus the stderr tail —
  and the socket channel carries a frame-length cap (`_ipc._MAX_FRAME`) so a
  desynchronised stream fails instead of allocating from a garbage length.

## 4. Measured (numpy_pwa, 240 phase-space + 160 data events, 2 ranks)

```
serial    Q = -127.18889717360435
service   Q = -127.18889717360435          dQ = 0.0
max |grad_service - grad_serial| = 4.5e-14
load (child start + 2 samples) = 1.14 s     get_nll = 8 ms (serial 1 ms)
child pid stable across evaluations         describe(): "running, program is a single process"
```

The 8 ms is the pipe plus two collectives on the *numpy* backend; for a GPU
backend the compute dominates.  The parent process imports no `mpi4py` and has
no `rank` attribute (gated).

## 5. Gates (`tests/test_mpi_service.py`, 8)

| gate | pins |
|---|---|
| program runs once + parity | `Q` at `rtol 1e-12`, grads `allclose`, child pid unchanged on the next evaluation, `describe()` says running |
| the parent never imports `mpi4py` | a subprocess imports `tabpwa`, constructs the backend, asserts `mpi4py` is not in `sys.modules` and `backend` has no `rank` |
| a killed child raises | kill the process, `compute` raises `RuntimeError` (no hang) |
| nesting refused | `PMI_RANK=1` → `RuntimeError` naming the all-ranks mode |
| `n_procs` required | constructor raises |
| leaf-only `local` | `auto` refused by the `delegate` tag |
| `P` and a reload | `return_p=True` is bit-equal to serial; `handle.free()` drops ONE sample and the next load reuses the child; after `free()` a call raises "shut down" instead of restarting |
| one model per service | a second `load` with a different kernel config aborts with "DIFFERENT kernel config" |

Only the gates that START a child are marked `needs_launcher`; the boundary
rules run on a box without MPI.

## 5a. Lifecycle rules the code enforces

* **`free()` is final.**  A later `load_data`/`compute` raises
  "this backend was shut down" rather than spawning a new `mpiexec` and then
  failing on a stale handle (a silent restart would hide the failure — design §8).
* **A broken channel stays broken.**  A timeout or an EOF mid-frame sets a dead
  flag: the partial body cannot be resynchronised, so every later call raises
  "the channel to the worker is broken" instead of reading the remainder as a
  length prefix.  `_ipc` also caps one frame at 16 GiB, so a desynchronised
  stream fails instead of allocating from a garbage length.
* **One model per service.**  The worker digests the kernel config of the first
  `load` and refuses a later one that differs.
* **Shutdown is short.**  `shutdown` → 2 s wait → `terminate` → 5 s → `kill` →
  reap, then the private directory is removed even when no child ever started
  (a failed `Popen` used to leak it).  A healthy `free()` costs 0.06 s; a FROZEN
  child costs ~7 s, not the 40 s the first version paid at `__del__`/Ctrl+C.
* **The child cleans up its own socket** on a clean exit, so a driver killed
  without `atexit` still leaves nothing behind.

## 6. Limits (v1)

* **Ship only**: the driver sends the sample arrays over the socket once; the
  `reference` path (path + hash, every rank opens the file) is designed and not
  implemented.
* **POSIX only** (`AF_UNIX`); no Windows.
* **One child per backend**, one client per child; no persistence across
  backends (the child dies with `free()`/`__del__`, with an `atexit` reaper).
* **Inherited `N×` transient**: the worker calls the ordinary `mpi` backend's
  `load_data(full_sample)` on every rank, so the rank-aware loader (design §4
  option B) is still the one follow-up that fixes both modes.
* **Not usable inside a step** by construction (nesting is refused); that case
  is the all-ranks backend under `srun`.
* Per-call overhead is one pipe round trip; with a `numpy` local backend that
  dominates (8 ms vs 1 ms), with a GPU kernel it does not.