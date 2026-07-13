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
"""
import os
import numpy as np
import multiprocessing
from .core import ComputeBackend, register_backend


def _worker_main(kernel_config, backend_spec, data_chunk,
                 task_queue, result_queue, device=None):
    """Worker process: create backend, load data, loop on compute tasks."""
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    from ampfit.backends import create_backend

    if isinstance(backend_spec, dict):
        spec = {k: v for k, v in backend_spec.items()
                if k not in ("device",)}
    else:
        spec = backend_spec

    be = create_backend(spec, kernel_config)
    dh = be.load_data(data_chunk)

    for task in iter(task_queue.get, None):
        params, norm = task
        Q, grads, P = be.compute(params, dh, norm=norm)
        result_queue.put((Q, grads, P))

    be.free()


class ShardDataHandle:
    """Opaque handle — tracks total event count."""
    def __init__(self, n_events):
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
    """

    def __init__(self, kernel_config, backends=None, n_workers=None,
                 weights=None):
        self.kernel_config = kernel_config
        self._workers = []
        self._task_queues = []
        self._result_queues = []
        self._specs = []
        self._weights = []

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

    def load_data(self, data_np):
        ne = data_np["mass"].shape[0]
        nw = self._n_workers
        if nw == 0:
            return ShardDataHandle(ne)

        # Compute weighted split offsets
        wsum = sum(self._weights)
        frac = np.cumsum([0.0] + [w / wsum for w in self._weights])
        frac[-1] = 1.0  # pin to exact end
        offsets = (frac * ne).astype(np.intp)

        chunks = []
        for i in range(nw):
            st = int(offsets[i])
            en = int(offsets[i + 1])
            chunk = {k: (v[st:en] if isinstance(v, np.ndarray) else v)
                     for k, v in data_np.items()}
            chunks.append(chunk)

        self._task_queues = [multiprocessing.Queue() for _ in range(nw)]
        self._result_queues = [multiprocessing.Queue() for _ in range(nw)]
        self._workers = []

        for i in range(nw):
            name, device = self._specs[i]
            p = multiprocessing.Process(
                target=_worker_main,
                args=(self.kernel_config, name, chunks[i],
                      self._task_queues[i], self._result_queues[i], device),
            )
            p.start()
            self._workers.append(p)

        return ShardDataHandle(ne)

    # -- compute -------------------------------------------------------

    def compute(self, params, data_handle, norm=None, return_p=True):
        nw = self._n_workers
        if nw == 0:
            if return_p:
                return 0.0, {}, np.array([])
            return 0.0, {}, None

        for tq in self._task_queues:
            tq.put((params, norm))

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
        for p in self._workers:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join()
        self._workers.clear()
        self._task_queues.clear()
        self._result_queues.clear()

    def __del__(self):
        self.free()
