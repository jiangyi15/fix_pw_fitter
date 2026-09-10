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


def _worker_main(kernel_config, backend_spec, task_queue, result_queue,
                 device=None):
    """Persistent worker: one backend, several datasets keyed by id.

    Tasks are tuples ``("load", data_id, chunk)`` /
    ``("compute", data_id, params, norm, return_p)`` / ``("free", data_id)``;
    ``None`` shuts the worker down.
    """
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    from ampfit.backends import create_backend

    if isinstance(backend_spec, dict):
        spec = {k: v for k, v in backend_spec.items()
                if k not in ("device",)}
    else:
        spec = backend_spec

    be = create_backend(spec, kernel_config)
    handles = {}
    for task in iter(task_queue.get, None):
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
            handles.pop(task[1], None)
        else:
            raise RuntimeError(f"unknown shard task {op!r}")

    be.free()


class ShardDataHandle:
    """Dataset identifier inside the shared worker pool."""

    def __init__(self, data_id, n_events):
        self.data_id = data_id
        self.n_events = n_events

    def free(self):
        pass


@register_backend("shard")
class ShardBackend(ComputeBackend):
    """Multi-process backend — shards data across worker backends.

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
    align : int, optional
        Round every interior split boundary down to a multiple of *align*
        (worker chunks then start/end on ``align``-aligned rows, except the
        final partial-tail worker).  Combine with a worker backend of
        ``cuda_v5_pwa`` whose ``resolution_size == align`` so the log-sum
        groups never straddle a worker cut.
    """

    def __init__(self, kernel_config, backends=None, n_workers=None,
                 weights=None, align=None):
        self.kernel_config = kernel_config
        self._align = int(align) if align else None
        if self._align is not None and self._align < 2:
            raise ValueError("align must be >= 2 (or None)")
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
                    name = spec.get("name", list(spec.keys())[0])
                    dev = spec.get("device")
                    self._specs.append((name, dev))
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

    def _ensure_pool(self):
        if self._procs:
            return
        nw = self._n_workers
        self._task_queues = [multiprocessing.Queue() for _ in range(nw)]
        self._result_queues = [multiprocessing.Queue() for _ in range(nw)]
        for i in range(nw):
            name, device = self._specs[i]
            p = multiprocessing.Process(
                target=_worker_main,
                args=(self.kernel_config, name, self._task_queues[i],
                      self._result_queues[i], device),
            )
            p.start()
            self._procs.append(p)

    def load_data(self, data_np):
        ne = data_np["mass"].shape[0]
        nw = self._n_workers
        if nw == 0:
            return ShardDataHandle(-1, ne)
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
        return ShardDataHandle(data_id, ne)

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
            Q_i, grads_i, P_i = self._result_queues[i].get()
            Q_total += Q_i

            if grads_total is None:
                grads_total = {k: np.asarray(v).copy()
                               for k, v in grads_i.items()}
            else:
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
        self.free()
