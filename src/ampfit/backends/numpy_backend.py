"""NumPy backend — pure CPU computation (float64, reference)."""
import numpy as np
from .core import ComputeBackend, per_event_dnorm, register_backend


@register_backend("numpy")
class NumpyBackend(ComputeBackend):
    """Pure NumPy computation (f64, CPU) with optional batching.

    Reports its native dNLL/dnorm (per-event NLL) in ``_last_dnorm``.
    """
    def __init__(self, kernel_config, batch_size=50000):
        from ampfit.numpy_kernel import NumpyKernel
        self.kernel = NumpyKernel(kernel_config)
        self._batch_size = batch_size

    def load_data(self, data_np):
        return data_np  # numpy dict is its own handle

    def compute(self, params, data_handle, norm=None, return_p=True):
        data = data_handle
        ne = data["mass"].shape[0]
        need_p = bool(return_p) or norm is not None  # P needed for dnorm
        if ne <= self._batch_size:
            Q, grads, P = self.kernel._compute(params, data, norm=norm,
                                               return_p=need_p)
            if norm is not None:
                grads["norm"] = per_event_dnorm(
                    norm, P, data["weight"], data.get("bkg"))
            return Q, grads, (P if return_p else None)

        # Batched: split into chunks, accumulate results
        bs = self._batch_size
        nbat = (ne + bs - 1) // bs

        Q_total = 0.0
        dnorm = 0.0
        grads_total = None
        P_list = []

        for b in range(nbat):
            st = b * bs
            en = min(st + bs, ne)
            chunk = {k: v[st:en] if isinstance(v, np.ndarray) else v
                     for k, v in data.items()}
            Qb, gb, Pb = self.kernel._compute(
                params, chunk, norm=norm, return_p=need_p)

            Q_total += Qb
            if norm is not None:
                dnorm += per_event_dnorm(
                    norm, Pb, chunk["weight"], chunk.get("bkg"))
            P_list.append(Pb)

            # Accumulate gradients (same keys, element-wise sum)
            if grads_total is None:
                grads_total = {k: np.asarray(v).copy() for k, v in gb.items()}
            else:
                for k in gb:
                    if gb[k] is not None:
                        grads_total[k] += np.asarray(gb[k])

        if norm is not None:
            grads_total["norm"] = dnorm
        P = np.concatenate(P_list, axis=0) if return_p else None
        return Q_total, grads_total, P
