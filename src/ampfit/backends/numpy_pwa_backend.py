"""numpy_pwa backend — CPU reference for the shared-ck projection-sum PWA."""
import numpy as np

from .core import ComputeBackend, register_backend
from ampfit.numpy_pwa import NumpyPWA


@register_backend("numpy_pwa")
class NumpyPWABackend(ComputeBackend):
    """CPU backend wrapping the ``NumpyPWA`` kernel (params: ck/m0/g0)."""

    def __init__(self, kernel_config, batch_size=20000):
        self.kernel = NumpyPWA(kernel_config)

    def load_data(self, data_np):
        return self.kernel.load_data(data_np)

    def compute(self, params, data_handle, norm=None, return_p=True):
        return self.kernel.compute(params, data_handle, norm=norm,
                                   return_p=return_p)

    def free(self):
        pass

    def __del__(self):
        self.free()
