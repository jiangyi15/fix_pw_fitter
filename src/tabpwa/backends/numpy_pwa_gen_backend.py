"""numpy_pwa_gen backend — NumPy reference for the ``pwa_gen`` basis model."""
from .core import ComputeBackend, register_backend
from tabpwa.numpy_pwa_gen import NumpyPWAGen


@register_backend("numpy_pwa_gen", model="pwa_gen")
class NumpyPWAGenBackend(ComputeBackend):
    """CPU backend wrapping ``NumpyPWAGen`` (params: ck / coeff)."""

    def __init__(self, kernel_config, batch_size=20000):
        self.kernel = NumpyPWAGen(kernel_config)

    def load_data(self, data_np):
        return self.kernel.load_data(data_np)

    def compute(self, params, data_handle, norm=None, return_p=True):
        Q, grads, P = self.kernel.compute(params, data_handle, norm=norm,
                                          return_p=True)
        if norm is not None:
            # native dnorm from the kernel (no host-array slot)
            grads["norm"] = float(self.kernel._last_dnorm)
        return Q, grads, (P if return_p else None)

    def free(self):
        pass

    def __del__(self):
        self.free()
