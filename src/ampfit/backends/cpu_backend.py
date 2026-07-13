"""CPU backend — C + OpenMP + AVX2 accelerated (float64)."""
import numpy as np
from .core import ComputeBackend, register_backend


@register_backend("cpu64_v3")
@register_backend("cpu_v3")
class CPUBackendV3(ComputeBackend):
    """CPU v3 backend (C + OpenMP + AVX2, Catmull-Rom interpolation).

    Uses the same sparse scatter/gather for matrix_gamma as cuda_v3_sparse.
    Parallelised via OpenMP over events (all CPU cores).

    Register as ``"cpu_v3"`` or ``"cpu64_v3"``.
    """
    def __init__(self, kernel_config, batch_size=50000):
        from ampfit.cpu._v3 import CPUKernelV3 as K
        self.kernel = K(kernel_config, batch_size=batch_size)

    def load_data(self, data_np):
        return self.kernel.load_data(data_np)

    def compute(self, params, data_handle, norm=None, return_p=True):
        return self.kernel.compute(params, data_handle, norm=norm,
                                   return_p=return_p)

    def free(self):
        self.kernel.free()
