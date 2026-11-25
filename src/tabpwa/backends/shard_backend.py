"""
ShardBackend — multi-process backend that shards data across workers.

Each worker runs its own backend in a separate process (GPU contexts are
isolated).  Data is split evenly; compute results are combined in the
parent process.

Intended use: multi-GPU.  Shard N GPUs and get ~N× throughput.
Currently testable with multiple workers on the same GPU.

Usage::

    # Two workers, same GPU, same backend
    create_backend({
        "name": "shard",
        "backends": ["cuda_v3_sparse", "cuda_v3_sparse"],
    }, kc)

    # Shorthand — N copies of the same backend:
    create_backend({
        "name": "shard",
        "backends": "cuda_v3_sparse",
        "n_workers": 2,
    }, kc)

     # Per-worker device assignment:
    create_backend({
        "name": "shard",
        "backends": [
            {"name": "cuda_v3_sparse", "device": 0},
            {"name": "cuda_v3_sparse", "device": 1},
        ],
    }, kc)

    # Uneven split (e.g. CPU 1× slower → give it less data):
    create_backend({
        "name": "shard",
        "backends": ["cpu_v3", "cuda_v3_sparse"],
        "weights": [1, 3],     # CPU gets 25%, GPU gets 75%
    }, kc)

    # Aligned split for cuda_v5_pwa resolution groups (chunks start/end on
    # resolution_size-aligned rows so the group log never straddles a cut):
    create_backend({
        "name": "shard",
        "align": 20,
        "backends": [
            {"name": "cuda_v5_pwa", "resolution_size": 20},
            {"name": "cuda_v5_pwa", "resolution_size": 20},
        ],
    }, kc)
"""
import os
import time
import traceback
from queue import Empty

import numpy as np
import multiprocessing
from .core import ComputeBackend, register_backend


def split_offsets(ne, weights, align=None):
    """Row boundaries of the weighted data split (len(weights)+1 entries).

    ``bounds[0] = 0``, ``bounds[-1] = ne``, worker *i* gets the contiguous
    rows ``[bounds[i], bounds[i+1])``.

    With ``align > 1`` every interior boundary is rounded down to a multiple
    of *align* (so each worker except possibly the last starts and ends on an
    ``align``-aligned row boundary — exactly what ``cuda_v5_pwa`` needs when
    its log-sum groups of ``resolution_size == align`` must never straddle a
    worker cut); the ``ne % align`` leftover rows stay with the last worker
    as the partial tail group.
    """
    weights = np.asarray(list(weights), dtype=float)
    n = len(weights)
    if n == 0:
        return [0, ne]
    wsum = float(weights.sum())
    if wsum <= 0:
        raise ValueError("weights must be positive")
    # cumulative fraction after each of the FIRST n-1 workers (the last
    # worker gets whatever is left, up to ne)
    cum = np.cumsum(weights / wsum)[:-1]

    if align is not None and int(align) > 1:
        a = int(align)
        blocks = ne // a                      # full aligned rows to distribute
        pos = [i for i, c in enumerate(cum) if weights[i] > 0]
        if blocks < len(pos):
            raise ValueError(
                f"align={a}: {ne} rows give only {blocks} aligned blocks for "
                f"{len(pos) + (1 if weights[-1] > 0 else 0)} workers — "
                f"reduce n_workers/align or use fewer, larger weights")
        edges = (cum * blocks).astype(np.int64)      # cumulative blocks
        for j in range(1, len(pos)):                 # floor can collide
            if edges[pos[j]] <= edges[pos[j - 1]]:
                raise ValueError(
                    f"align={a}: worker split cannot give every worker >=1 "
                    f"aligned block; use a smaller weights spread")
        return [0] + [int(e) * a for e in edges] + [ne]

    offs = (cum * ne).astype(np.int64)
    offs = np.maximum.accumulate(offs).clip(0, ne)
    return [0] + [int(o) for o in offs] + [ne]


# First element of a result-queue message that carries a worker traceback.
_ERROR = "__shard_worker_error__"

#: How often the parent looks at a SILENT worker to see whether it is still
#: there.  These bound no work: ``worker_timeout=None`` (the default) means
#: "as long as the work takes", and a worker that is ALIVE is simply waited
#: for.  The poll exists only because a worker that has EXITED needs nothing
#: and its queue would stay empty forever.  Once the worker is GONE, what it
#: produced is already in the pipe (its feeder flushes at exit), so the result
#: is DRAINED, not waited for: no grace period is scheduled for anyone's work.
_DEATH_POLL = 0.5

#: How long an IDLE worker waits on its task queue before looking at its
#: parent.  This bounds NO work: a task wakes ``get`` the moment it arrives and
#: is then executed without any limit (``worker_timeout`` is the caller's knob,
#: ``None`` by default).  It is a slice only for the parent check — the one
#: thing that must not be assumed is that a driver stays alive: a `kill -9`ed
#: driver runs no `atexit`, so nothing else would ever stop the worker.  The
#: parent is identified by ``os.getppid()`` PLUS the driver's pid, NOT by a
#: multiprocessing sentinel: under ``fork`` a worker inherits the sentinels of
#: the workers started before it, so a sentinel reports its EOF only when the
#: LATER workers are gone and the pool would stop one worker per slice, in
#: creation order.  ``getppid()`` is reparenting, which the kernel performs at
#: the instant the parent dies — but under ``forkserver`` the worker's DIRECT
#: parent is the forkserver, which outlives the driver, so the driver's own pid
#: is checked as well.
_WORKER_POLL = 5.0

#: How long a REAP pass waits after SIGKILL — ONE budget for the whole pool,
#: not per process, so eight wedged workers cannot turn a `__del__` into eight
#: waits.  Reaping is not waiting on the worker's work (the process is already
#: dead or in the kernel's hands), but a process in uninterruptible sleep (a
#: hung GPU driver call is the canonical case here) would otherwise block the
#: interpreter with nothing left to do.  After this the process is left to the
#: kernel.
_REAP_WAIT = 5.0


def _reap_all(procs):
    """Join processes that have ALREADY been SIGKILLed, within one budget."""
    deadline = time.monotonic() + _REAP_WAIT
    for p in procs:
        if p.pid is None:             # start() never completed: not joinable
            continue
        try:
            p.join(timeout=max(0.0, deadline - time.monotonic()))
        except Exception:
            pass


_POSIX = os.name == "posix"


def _pid_alive(pid):
    """Whether *pid* names a live process (a ZOMBIE still answers here).

    POSIX ONLY, deliberately: on Windows ``os.kill(pid, 0)`` is
    ``TerminateProcess``, so this probe would KILL the driver instead of asking
    about it.  Off POSIX it answers "alive" and the parent watchdog is NOT
    IMPLEMENTED: Windows never reparents a child, so ``getppid()`` is frozen at
    creation and cannot see a dead parent either.  (The shard backend is POSIX
    in practice — as are the kernels it shards.)

    It answers about a NUMBER, not an identity: a driver pid recycled before the
    next idle slice would keep a worker waiting until that pid is released.
    The error is in the conservative direction — a DELAYED exit, never a
    premature one.
    """
    if not _POSIX:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:          # alive, just not signalable by us
        return True
    return True


def _worker_main(kernel_config, backend_spec, task_queue, result_queue,
                 device=None, model=None, driver_pid=None):
    """Persistent worker: one backend, several datasets keyed by id.

    Tasks are tuples ``("load", data_id, chunk)`` /
    ``("compute", data_id, params, norm, return_p)`` / ``("free", data_id)`` /
    ``("exit",)``.  Every command is an explicit tuple: the worker stops on
    ``("exit",)`` and on nothing else (a malformed task is an error, not a
    shutdown).  Only a COMPUTE has a normal reply; a failure of ANY command
    replies once as ``(_ERROR, phase, traceback)``.
    """
    # the driver this worker belongs to, read ONCE: `getppid` changes when the
    # parent dies (the kernel reparents), and unlike the multiprocessing
    # sentinel it cannot be inherited from a sibling under `fork`.  Under
    # `forkserver` the direct parent is the FORKSERVER, so the driver's pid is
    # checked too — otherwise a `kill -9`ed driver would never be noticed.
    ppid0 = os.getppid()
    # an MPI rank inside this forked/spawned child would use a broken MPI state
    os.environ["TABPWA_SHARD_WORKER"] = "1"
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    from tabpwa.backends import create_backend

    if isinstance(backend_spec, dict):
        spec = {k: v for k, v in backend_spec.items()
                if k not in ("device",)}
    else:
        spec = backend_spec

    be = create_backend(spec, kernel_config, model=model)
    # a worker is DAEMON (see ``_ensure_pool``), and daemonic processes may not
    # start children — so a nested shard pool, which WORKED before the workers
    # became daemonic, must be refused by name instead of dying later inside
    # ``Process.start`` with CPython's bare ``AssertionError``.
    if isinstance(be, ShardBackend):
        raise RuntimeError(
            "a shard worker cannot host another shard pool: the workers are "
            "daemonic, and daemonic processes may not start children.  Use "
            "one flat ``backends`` list instead of nesting `shard`")
    handles = {}
    while True:
        try:
            task = task_queue.get(timeout=_WORKER_POLL)
        except Empty:
            # The driver may be GONE without having sent `("exit",)`: a
            # `kill -9` (or a scheduler's SIGKILL on a timeout) runs no
            # `atexit`, so nothing terminates a daemonic child and no command
            # ever arrives — the worker would block in `get()` forever, holding
            # its GPU context.  Exit with the parent instead.
            # (`_WORKER_POLL` is only a slice: a task wakes it at once.)
            if (os.getppid() != ppid0
                    or (driver_pid is not None and not _pid_alive(driver_pid))):
                break
            continue
        op = task[0] if isinstance(task, (tuple, list)) and task else None
        if op == "exit":
            break
        try:
            if op == "load":
                _, data_id, chunk = task
                handles[data_id] = be.load_data(chunk)
            elif op == "compute":
                _, data_id, params, norm, return_p = task
                Q, grads, P = be.compute(params, handles[data_id], norm=norm,
                                         return_p=return_p)
                if norm is not None and "norm" not in grads:
                    raise RuntimeError(
                        f"shard worker ({type(be).__name__}) did not return "
                        f"grads['norm'] for a normed compute")
                result_queue.put((Q, grads, P))
            elif op == "free":
                h = handles.pop(task[1], None)
                if h is not None and hasattr(h, "free"):
                    h.free()
            else:
                raise RuntimeError(f"unknown shard task {task!r}")
        except Exception:
            # Never die silently: the parent waits on this queue, so an
            # unhandled error must come back as a message (else it hangs).
            # The PHASE is part of the message: a failing `load`/`free` replies
            # too, and the parent must say which command failed.
            result_queue.put((_ERROR, op or "task", traceback.format_exc()))

    # explicit release of every remaining dataset before the backend context
    for h in handles.values():
        if hasattr(h, "free"):
            h.free()
    handles.clear()
    be.free()


class ShardDataHandle:
    """Dataset identifier inside the shared worker pool."""

    def __init__(self, data_id, n_events, backend=None):
        self.data_id = data_id
        self.n_events = n_events
        self._backend = backend

    def free(self):
        """Drop this dataset from every worker (frees its device memory).

        The worker pool itself stays alive; ``ShardBackend.free()`` shuts
        the pool down.  Idempotent.
        """
        if self.data_id < 0 or self._backend is None:
            return
        for tq in self._backend._task_queues:
            try:
                tq.put(("free", self.data_id))
            except Exception:
                break                     # a broken pool: its own free() kills it
        self.data_id = -1

    def __del__(self):
        """Drop the dataset when the handle is collected.

        ``ShardDataHandle`` is not a ``DataHandle`` subclass, so it did not
        inherit ``DataHandle.__del__``: a caller that loads many datasets into
        one long-lived pool and drops them kept every worker-side copy.  The
        pool is still the lifetime unit; this only stops a dataset from
        outliving its handle.
        """
        try:
            self.free()
        except Exception:
            pass


@register_backend("shard", tags=('delegate',))
class ShardBackend(ComputeBackend):
    """Multi-process backend — shards data across worker backends.

    Nested spec: ``backends`` (one spec per worker).

    Parameters
    ----------
    kernel_config : dict
        Config from ``Config.build_all_index()``.
    backends : str, list of str, or list of dict
        Backend spec(s).  A single string creates ``n_workers`` copies.
        A list of strings/dicts creates one worker per entry.
        Dict form supports ``"device"`` key for CUDA_VISIBLE_DEVICES.
    n_workers : int, optional
        Number of workers when *backends* is a single string (default 2).
    weights : list of float, optional
        Split ratio per worker.  Default equal.  E.g. ``[1, 3]`` gives
        worker 1 one quarter and worker 2 three quarters of the data.
        Useful when mixing fast (GPU) and slow (CPU) backends.
    start_method : str, optional
        ``multiprocessing`` start method for the worker pool
        (``"fork"`` / ``"spawn"`` / ``"forkserver"``; default: the system
        default).  Prefer ``"spawn"``/``"forkserver"`` in multi-threaded or
        CUDA applications where ``fork`` is unsafe.
    worker_timeout : float, optional
        Seconds to wait for each worker COMPUTE result before raising.
        ``None`` (default) = no DEADLINE, not "no diagnosis": a worker that has
        EXITED (bootstrap crash, segfault, OOM) is detected by polling its
        process and raises with the exit code, and worker exceptions always
        come back as a message.  A positive value adds a wall-clock bound on a
        worker that is alive but stuck in ``compute``.
        Only a COMPUTE has a NORMAL reply.  ``load_data`` queues its command
        and returns (its duration — a complex config's kernel build or cache
        fill — is not something this backend may assume).  A failure of any
        command, ``load``/``free`` included, replies ONCE and is reported with
        its phase; the pool is then refused further work, because the failed
        worker's later replies are still queued.
    align : int, optional
        Round every interior split boundary down to a multiple of *align*
        (worker chunks then start/end on ``align``-aligned rows, except the
        final partial-tail worker).  Combine with a worker backend of
        ``cuda_v5_pwa`` whose ``resolution_size == align`` so the log-sum
        groups never straddle a worker cut.
    """

    def __init__(self, kernel_config, backends=None, n_workers=None,
                 weights=None, align=None, start_method=None,
                 worker_timeout=None, model=None):
        # set first: free()/__del__ must be safe if spawn fails
        self._procs = []
        self._task_queues = []
        self._result_queues = []
        self.kernel_config = kernel_config
        self._model = model
        self._align = int(align) if align else None
        self._worker_timeout = (float(worker_timeout)
                                if worker_timeout else None)
        if self._worker_timeout is not None and self._worker_timeout <= 0:
            raise ValueError("worker_timeout must be > 0 (or None)")
        if self._align is not None and self._align < 2:
            raise ValueError("align must be >= 2 (or None)")
        if start_method is not None and start_method not in (
                "fork", "spawn", "forkserver"):
            raise ValueError(
                f"start_method must be fork/spawn/forkserver or None, "
                f"got {start_method!r}")
        self._mp = (multiprocessing.get_context(start_method)
                    if start_method else multiprocessing.get_context())
        self._specs = []
        self._weights = []
        # persistent worker pool (created lazily) + dataset id counter
        self._task_queues = []
        self._result_queues = []
        self._procs = []
        self._next_id = 0
        #: set when a worker replied with a failure: its later replies are still
        #: queued, so one reply per compute no longer holds
        self._broken = None

        if backends is None:
            backends = []
        if isinstance(backends, str):
            n = n_workers or 2
            self._specs = [(backends, None)] * n
        elif isinstance(backends, (list, tuple)):
            for spec in backends:
                if isinstance(spec, str):
                    self._specs.append((spec, None))
                elif isinstance(spec, dict):
                    # Keep the FULL worker spec (resolution_size, batch_size,
                    # ...) minus ``device``; the worker builds from this dict.
                    worker = {k: v for k, v in spec.items() if k != "device"}
                    if "name" not in worker and worker:
                        worker["name"] = next(iter(worker))
                    self._specs.append((worker, spec.get("device")))
        self._n_workers = len(self._specs)

        # Default: equal weights
        if weights is not None:
            if len(weights) != self._n_workers:
                raise ValueError(
                    f"len(weights)={len(weights)} != n_workers={self._n_workers}")
            self._weights = list(weights)
        else:
            self._weights = [1.0] * self._n_workers

    # -- data lifecycle ------------------------------------------------

    def _start_hint(self):
        """Why a spawn/forkserver worker cannot start, and the fix.

        Both methods re-IMPORT the calling program in the child, so its
        top-level code runs there too: an unguarded program builds the backend
        again, starts another pool, and the bootstrap aborts (or recurses)
        before the worker ever reads a task.
        """
        if self._mp.get_start_method() in ("spawn", "forkserver"):
            return (
                f"; with start_method={self._mp.get_start_method()!r} every "
                "worker re-imports the calling program, so its top-level code "
                "must be under `if __name__ == '__main__':`")
        return ""

    def _shutdown_started(self, failed=None):
        """Kill the workers of a PARTLY started pool (a failed ``start``).

        ``failed`` is the process whose ``start()`` raised: it is NOT in
        ``_procs`` (it is appended only on success) but it may have left a
        child behind, so it is killed too.  Each join follows a SIGKILL, so it
        does not depend on the worker cooperating, and it is BOUNDED: a process
        wedged in uninterruptible sleep must not hang an error path.
        """
        procs = list(self._procs) + ([failed] if failed is not None else [])
        self._procs.clear()
        # the queues this pool created are abandoned too; close them so no
        # feeder thread is left writing into a pipe nobody reads
        for q in list(self._task_queues) + list(self._result_queues):
            try:
                q.close()
            except Exception:
                pass
        self._task_queues = []
        self._result_queues = []
        for p in procs:
            if p.pid is None:          # start() never completed: not joinable
                continue
            try:
                if p.is_alive():
                    p.kill()
            except Exception:
                pass
        _reap_all(procs)

    def _ensure_pool(self):
        if self._procs:
            return
        nw = self._n_workers
        self._task_queues = [self._mp.Queue() for _ in range(nw)]
        self._result_queues = [self._mp.Queue() for _ in range(nw)]
        for i in range(nw):
            spec, device = self._specs[i]
            p = self._mp.Process(
                target=_worker_main,
                args=(self.kernel_config, spec, self._task_queues[i],
                      self._result_queues[i], device, self._model,
                      os.getpid()),
            )
            # the backend OWNS these processes (`free()` shuts them down); a
            # non-daemon child is joined by the interpreter at exit, so an
            # unhandled error in the parent would hang the program forever on
            # a worker blocked in ``task_queue.get()``.  A worker never starts
            # a child of its own (shard-in-shard and mpi-in-worker are
            # refused), which is the only thing daemon mode forbids.
            p.daemon = True
            try:
                p.start()
            except (RuntimeError, OSError) as exc:
                # BrokenPipeError (an OSError) is what the parent sees when the
                # child aborts before reading its preparation data; the raw
                # RuntimeError is what the CHILD sees on the same failure.
                self._shutdown_started(failed=p)
                raise RuntimeError(
                    f"shard worker {i} could not be started "
                    f"({type(exc).__name__}: {str(exc).strip()})"
                    f"{self._start_hint()}") from exc
            self._procs.append(p)

    def load_data(self, data_np):
        if self._broken is not None:
            raise RuntimeError(
                f"{self._broken}\nthe pool is desynchronised; build a new "
                f"backend instead of reusing this one")
        ne = data_np["mass"].shape[0]
        nw = self._n_workers
        if nw == 0:
            return ShardDataHandle(-1, ne, self)
        self._ensure_pool()

        # Compute weighted split offsets (aligned to multiples of --align)
        bounds = split_offsets(ne, self._weights, self._align)
        if len(bounds) != nw + 1:
            raise ValueError(
                f"internal split error: got {len(bounds) - 1} chunks for "
                f"{nw} workers")

        chunks = []
        for i in range(nw):
            st = int(bounds[i])
            en = int(bounds[i + 1])
            chunk = {k: (v[st:en] if isinstance(v, np.ndarray) else v)
                     for k, v in data_np.items()}
            chunks.append(chunk)

        data_id = self._next_id
        self._next_id += 1
        for i in range(nw):
            self._task_queues[i].put(("load", data_id, chunks[i]))
        # NO normal reply for a load: the worker protocol is exactly the
        # commands it receives — create, load, compute, free, exit — and only
        # a compute answers.  Ordering is the queue's: a worker runs its load
        # before the compute that follows it on the same queue, so one reply
        # per compute stays attributed.  A FAILURE of the load replies once as
        # `(_ERROR, "load", traceback)` and marks the pool desynchronised, so
        # it is reported as a load failure rather than as the next compute's.
        return ShardDataHandle(data_id, ne, self)

    # -- result handling -----------------------------------------------

    def _dead_message(self, i, proc):
        """Why a silent worker is silent, and what to do about the usual cause."""
        return (f"shard worker {i} died (exitcode={proc.exitcode}) without "
                f"returning a result; its own traceback is on stderr"
                f"{self._start_hint()}")

    def _get_alive(self, i):
        """``result_queues[i].get()`` that also fails when the worker DIES.

        ``worker_timeout=None`` means "wait as long as the worker needs" — but
        a worker that is GONE needs nothing, and its queue stays empty forever
        (a hang, on a batch system a burnt allocation).  A bootstrap crash (e.g.
        ``spawn`` with a top-level program that is not under
        ``if __name__ == '__main__'``), a segfault and an OOM kill all land
        here.  Poll the process; honour a result put just before the exit, then
        raise with the exit code.
        """
        rq = self._result_queues[i]
        if i >= len(self._procs):
            raise RuntimeError(
                f"shard worker {i} is gone: the pool was freed or never "
                f"started")
        proc = self._procs[i]
        while True:
            try:
                return rq.get(timeout=_DEATH_POLL)
            except Empty:
                if proc.is_alive():
                    continue
                # the worker is GONE and its feeder thread flushes at exit, so
                # whatever it produced is already in the pipe: DRAIN it, never
                # wait a scheduled grace for it
                try:
                    return rq.get_nowait()
                except Empty:
                    raise RuntimeError(self._dead_message(i, proc)) from None

    def _recv_raw(self, i, timeout=None):
        """Next message from worker *i*: a COMPUTE result, or a failure.

        The normal message is a compute result; a failure of any command is
        ``(_ERROR, phase, traceback)`` and latches the pool broken.

        ``timeout=None`` means "as long as the worker needs" — a worker that
        has EXITED is still detected (``_get_alive``), because its queue then
        stays empty forever.  With a ``timeout`` the death is still attributed:
        a worker that died during the wait is reported with its exit code, not
        as "did not respond".
        """
        rq = self._result_queues[i]
        proc = self._procs[i] if i < len(self._procs) else None
        try:
            msg = (rq.get(timeout=timeout) if timeout is not None
                   else self._get_alive(i))
        except Empty:
            if proc is not None and not proc.is_alive():
                raise RuntimeError(self._dead_message(i, proc)) from None
            alive = proc.is_alive() if proc is not None else False
            raise RuntimeError(
                f"shard worker {i} did not respond within {timeout}s "
                f"(alive={alive})") from None
        except (EOFError, OSError) as exc:
            # a worker killed mid-`send_bytes` tears its message
            if proc is not None and not proc.is_alive():
                raise RuntimeError(self._dead_message(i, proc)) from None
            raise RuntimeError(
                f"shard worker {i} result stream is broken "
                f"({type(exc).__name__}: {exc})") from exc
        if isinstance(msg, tuple) and len(msg) >= 2 and msg[0] == _ERROR:
            phase = msg[1] if len(msg) > 1 else "?"
            detail = msg[2] if len(msg) > 2 else msg[-1]
            # one failure leaves the worker's remaining replies in the queue,
            # so the stream no longer matches one reply per compute: refuse
            # further work instead of returning a previous call's numbers
            self._broken = (f"shard worker {i} failed during {phase}:\n"
                            f"{detail}")
            raise RuntimeError(self._broken)
        return msg

    def _recv(self, i):
        """Receive worker *i*'s COMPUTE result; an error becomes a raise."""
        return self._recv_raw(i, self._worker_timeout)

    # -- compute -------------------------------------------------------

    def compute(self, params, data_handle, norm=None, return_p=True):
        nw = self._n_workers
        if nw == 0:
            if return_p:
                return 0.0, {}, np.array([])
            return 0.0, {}, None
        if self._broken is not None:
            raise RuntimeError(
                f"{self._broken}\nthe pool is desynchronised; build a new "
                f"backend instead of reusing this one")
        if not self._procs:
            raise RuntimeError(
                "shard pool is not running: free()/kill() was called, or "
                "load_data() was never reached")

        did = data_handle.data_id
        for tq in self._task_queues:
            tq.put(("compute", did, params, norm, return_p))

        Q_total = 0.0
        grads_total = None
        P_list = []

        for i in range(nw):
            Q_i, grads_i, P_i = self._recv(i)
            Q_total += Q_i

            if grads_total is None:
                grads_total = {k: np.asarray(v).copy()
                               for k, v in grads_i.items()}
            else:
                if set(grads_i) != set(grads_total):
                    raise RuntimeError(
                        f"shard worker {i} returned gradient keys "
                        f"{sorted(grads_i)} != {sorted(grads_total)}")
                for k in grads_i:
                    if grads_i[k] is not None:
                        g1 = np.asarray(grads_i[k])
                        g0 = grads_total[k]
                        if isinstance(g0, tuple):
                            grads_total[k] = tuple(
                                np.asarray(a) + np.asarray(b)
                                for a, b in zip(g0, g1))
                        else:
                            grads_total[k] = g0 + g1

            if return_p and P_i is not None:
                P_list.append(P_i)

        P = np.concatenate(P_list, axis=0) if (return_p and P_list) else None
        return Q_total, grads_total, P

    # -- cleanup -------------------------------------------------------

    def free(self):
        """Stop the pool COOPERATIVELY: the ``("exit",)`` command, then wait.

        No deadline is put on a worker's current task — that is the user's
        work, and this backend may not decide how long it takes.  An IDLE
        worker exits at once; a BUSY one finishes what it is doing and then
        exits.  :meth:`kill` is the immediate form (and what ``__del__`` uses,
        because an interpreter shutdown must not block).

        The KILL pass sits in a ``finally``: an interrupt during the
        cooperative wait (or a worker that never reads the command, because it
        is stuck inside a compute) still takes the pool down.  That reap is
        BOUNDED — the process has already been killed, so waiting on it is not
        waiting on anyone's work, and a process in uninterruptible sleep is
        left to the kernel.

        Idempotent: the state is detached before anything can raise.
        """
        tqs, self._task_queues = self._task_queues, []
        procs, self._procs = self._procs, []
        rqs, self._result_queues = self._result_queues, []
        delivered = True
        try:
            for tq in tqs:
                try:
                    tq.put(("exit",))
                except Exception:
                    delivered = False
                    break     # the command cannot arrive: go straight to KILL
            if delivered:
                for p in procs:
                    if p.pid is None:     # start() never completed
                        continue
                    # COOPERATIVE and deliberately unbounded: a busy worker
                    # finishes its task.  This is the user's work, so no clock
                    # is put on it; the `finally` is what guarantees an exit.
                    try:
                        p.join()
                    except Exception:
                        pass
        finally:
            for p in procs:
                if p.pid is None:
                    continue
                try:
                    if p.is_alive():
                        p.kill()
                except Exception:
                    pass
            _reap_all(procs)
            for q in list(tqs) + list(rqs):
                try:
                    q.close()             # no feeder left writing into a pipe
                except Exception:
                    pass

    def kill(self):
        """Stop the pool NOW: SIGKILL every worker, no exit command.

        For an abort (an exception mid-compute, an interrupted job) and for
        interpreter shutdown — never a deadline on a worker's work, just no
        cooperation.  The join after a SIGKILL only REAPS the process; it does
        not wait for anything the worker was doing.
        """
        procs, self._procs = self._procs, []
        tqs, self._task_queues = self._task_queues, []
        rqs, self._result_queues = self._result_queues, []
        for p in procs:
            if p.pid is None:
                continue
            try:
                if p.is_alive():
                    p.kill()
            except Exception:
                pass
        _reap_all(procs)
        for q in list(tqs) + list(rqs):
            try:
                q.close()
            except Exception:
                pass

    def __del__(self):
        # KILL, never free(): at interpreter shutdown nothing may block on a
        # worker's current task, and a failed ``_ensure_pool`` can leave the
        # object being collected where no thread may be created
        try:
            self.kill()
        except Exception:
            pass
