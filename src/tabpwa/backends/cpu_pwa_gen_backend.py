"""cpu_pwa_gen backend — CPU (C + OpenMP) general pure-PWA (any n_res).

CPU counterpart of ``cuda_v4_pwa``: the projection-sum PWA with any number
of Breit–Wigner factors per chain (``n_res >= 1``).  Same model / parameter
layout as ``cpu_pwa_s0c`` (direct pole parameters ``s0 = m0²`` /
``mult = m0·g0`` inside the kernel, host-side chain back to the fitted
``m0``/``g0``), but every (wave, factor) carries its own BW expansion, so
multi-resonance chains are built exactly like the CUDA sibling.

Thin alias: it constructs the ``cpu_v5_pwa`` kernel at
``resolution_size=1``, whose branch for that size IS the v4 per-event NLL
bit for bit (one C kernel / one shared library behind both names).
"""
from .core import ComputeBackend, register_backend


@register_backend("cpu_pwa_gen", model="pwa")
@register_backend("cpu_v4_pwa", model="pwa")
class CPUPWAGenBackend(ComputeBackend):
    """CPU PWA general-n_res backend (C + OpenMP, float64, per-event NLL).

    At ``n_res == 1`` it is bit-identical to ``cpu_pwa_s0c``; for
    ``n_res > 1`` it is the CPU sibling of ``cuda_v4_pwa``.
    """

    def __init__(self, kernel_config, batch_size=50000):
        self.kernel = None   # free/__del__ must be safe if the build fails
        from tabpwa.cpu._pwa_v5 import CPUKernelPWAV5 as K
        self.kernel = K(kernel_config, batch_size=batch_size,
                        resolution_size=1)

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
