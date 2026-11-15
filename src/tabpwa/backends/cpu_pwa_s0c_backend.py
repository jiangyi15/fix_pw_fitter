"""cpu_pwa_s0c backend — CPU (C + OpenMP) pure-PWA with s0c pole params."""
from .core import ComputeBackend, register_backend


@register_backend("cpu_pwa_s0c", model="pwa")
class CPUPWAS0CBackend(ComputeBackend):
    """CPU PWA s0c backend (C + OpenMP, float64).

    Same model / parameterisation as ``cuda_v4_pwa_s0c``: direct pole
    parameters ``s0 = m0²`` / ``mult = m0·g0`` inside the kernel, host-side
    chain back to the fitted ``m0``/``g0``.  Parallelised over events.
    """

    def __init__(self, kernel_config, batch_size=50000):
        self.kernel = None   # free/__del__ must be safe if the build fails
        from tabpwa.cpu._pwa_s0c import CPUKernelPWAS0C as K
        self.kernel = K(kernel_config, batch_size=batch_size)

    def load_data(self, data_np):
        return self.kernel.load_data(data_np)

    def compute(self, params, data_handle, norm=None, return_p=True):
        Q, grads, P = self.kernel.compute(params, data_handle, norm=norm,
                                          return_p=return_p)
        if norm is not None:
            # native dnorm from the kernel (no host-array slot)
            grads["norm"] = float(self.kernel._last_dnorm)
        return Q, grads, (P if return_p else None)

    def free(self):
        if self.kernel is not None:
            self.kernel.free()
            self.kernel = None

    def __del__(self):
        self.free()
