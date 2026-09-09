"""numpy_pwa backend — CPU reference for the shared-ck projection-sum PWA."""
import numpy as np

from .core import ComputeBackend, per_event_dnorm, register_backend
from ampfit.numpy_pwa import NumpyPWA


@register_backend("numpy_pwa")
class NumpyPWABackend(ComputeBackend):
    """CPU backend wrapping the ``NumpyPWA`` kernel (params: ck/m0/g0)."""

    def __init__(self, kernel_config, batch_size=20000):
        self.kernel = NumpyPWA(kernel_config)

    def load_data(self, data_np):
        self._data_np = data_np
        return self.kernel.load_data(data_np)

    def compute(self, params, data_handle, norm=None, return_p=True):
        Q, grads, P = self.kernel.compute(params, data_handle, norm=norm,
                                          return_p=True)
        if norm is not None:
            d = data_handle["data"] if isinstance(data_handle, dict) \
                else self._data_np
            grads["norm"] = per_event_dnorm(
                norm, P, d["weight"], d.get("bkg"))
        return Q, grads, (P if return_p else None)

    def free(self):
        pass

    def __del__(self):
        self.free()
