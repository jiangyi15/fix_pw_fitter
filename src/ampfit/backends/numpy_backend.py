"""NumPy backend — pure CPU computation (float64, reference)."""
import numpy as np
from .core import ComputeBackend, register_backend


@register_backend("numpy")
class NumpyBackend(ComputeBackend):
    """Pure NumPy computation (f64, CPU) with optional batching."""
    def __init__(self, kernel_config, batch_size=50000):
        from ampfit.numpy_kernel import NumpyKernel
        self.kernel = NumpyKernel(kernel_config)
        self._batch_size = batch_size

    def load_data(self, data_np):
        return data_np  # numpy dict is its own handle

    def compute(self, params, data_handle, norm=None, return_p=True):
        data = data_handle
        ne = data["mass"].shape[0]
        if ne <= self._batch_size:
            return self.kernel._compute(params, data, norm=norm,
                                        return_p=return_p)

        # Batched: split into chunks, accumulate results
        bs = self._batch_size
        nbat = (ne + bs - 1) // bs

        Q_total = 0.0
        grads_total = None
        P_list = []

        for b in range(nbat):
            st = b * bs
            en = min(st + bs, ne)
            chunk = {k: v[st:en] if isinstance(v, np.ndarray) else v
                     for k, v in data.items()}
            Qb, gb, Pb = self.kernel._compute(
                params, chunk, norm=norm, return_p=return_p)

            Q_total += Qb
            P_list.append(Pb)

            # Accumulate gradients (same keys, element-wise sum)
            if grads_total is None:
                grads_total = {k: np.asarray(v).copy() for k, v in gb.items()}
            else:
                for k in gb:
                    if gb[k] is not None:
                        grads_total[k] += np.asarray(gb[k])

        P = np.concatenate(P_list, axis=0) if return_p else None
        return Q_total, grads_total, P
