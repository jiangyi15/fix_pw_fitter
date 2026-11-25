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

#: A worker that DIED (bootstrap crash, segfault, OOM kill) never replies, so a
#: blocking ``queue.get()`` waits forever.  The default ``worker_timeout=None``
#: means "as long as the worker needs", which cannot cover a worker that needs
#: nothing: poll its process instead, then give the queue one ``_DEATH_GRACE``
#: second so a result flushed just before the exit is not lost.
_DEATH_POLL = 0.5
_DEATH_GRACE = 2.0


def _worker_main(kernel_config, backend_spec, task_queue, result_queue,
                 device=None, model=None):
    """Persistent worker: one backend, several datasets keyed by id.

    Tasks are tuples ``("load", data_id, chunk)`` /
    ``("compute", data_id, params, norm, return_p)`` / ``("free", data_id)``;
    ``None`` shuts the worker down.
    """
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
    for task in iter(task_queue.get, None):
        try:
            op = task[0]
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
                raise RuntimeError(f"unknown shard task {op!r}")
        except Exception:
            # Never die silently: the parent waits on this queue, so an
            # unhandled error must come back as a message (else it hangs).
            result_queue.put((_ERROR, traceback.format_exc()))

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
            tq.put(("free", self.data_id))
        self.data_id = -1


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
        Seconds to wait for each worker result before raising.  ``None``
        (default) = no DEADLINE, not "no diagnosis": a worker that has EXITED
        (bootstrap crash, segfault, OOM) is detected by polling its process and
        raises with the exit code, and worker exceptions always come back as a
        message.  A positive value adds a wall-clock bound on a worker that is
        alive but stuck.
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
        child behind, so it is killed too.  The joins are bounded like
        ``free()``'s — an error path must not hang on a worker in ``D`` state.
        """
        procs = list(self._procs) + ([failed] if failed is not None else [])
        self._procs.clear()
        for p in procs:
            if p.pid is None:          # start() never completed: not joinable
                continue
            if p.is_alive():
                p.kill()
            p.join(timeout=5)

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
                      self._result_queues[i], device, self._model),
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
        proc = self._procs[i] if i < len(self._procs) else None
        while True:
            try:
                return rq.get(timeout=_DEATH_POLL)
            except Empty:
                if proc is None or proc.is_alive():
                    continue
                try:
                    return rq.get(timeout=_DEATH_GRACE)
                except Empty:
                    raise RuntimeError(self._dead_message(i, proc)) from None

    def _recv(self, i):
        """Receive worker *i*'s result; turn an error/timeout into a raise."""
        rq = self._result_queues[i]
        try:
            msg = (rq.get(timeout=self._worker_timeout)
                   if self._worker_timeout else self._get_alive(i))
        except Empty:
            alive = self._procs[i].is_alive() if i < len(self._procs) else False
            raise RuntimeError(
                f"shard worker {i} did not respond within "
                f"{self._worker_timeout}s (alive={alive})") from None
        if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == _ERROR:
            raise RuntimeError(f"shard worker {i} failed:\n{msg[1]}")
        return msg

    # -- compute -------------------------------------------------------

    def compute(self, params, data_handle, norm=None, return_p=True):
        nw = self._n_workers
        if nw == 0:
            if return_p:
                return 0.0, {}, np.array([])
            return 0.0, {}, None

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
        for tq in self._task_queues:
            tq.put(None)
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join()
        self._procs.clear()
        self._task_queues.clear()
        self._result_queues.clear()

    def __del__(self):
        # best effort: a failed ``_ensure_pool`` can leave the object being
        # collected at interpreter shutdown, where no thread may be created
        try:
            self.free()
        except Exception:
            pass
