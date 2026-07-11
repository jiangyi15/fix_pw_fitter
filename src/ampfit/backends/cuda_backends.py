"""CUDA backends — GPU-accelerated computation (f64 and f32)."""
import numpy as np
from .core import ComputeBackend, register_backend


class _CUDABackend(ComputeBackend):
    """Common base for all CUDA backends — delegates to a kernel."""

    def __init__(self, kernel_config, batch_size=50000):
        self.kernel = self._make_kernel(kernel_config, batch_size)

    def _make_kernel(self, kernel_config, batch_size):
        raise NotImplementedError

    def load_data(self, data_np):
        return self.kernel.load_data(data_np)

    def compute(self, params, data_handle, norm=None, return_p=True):
        return self.kernel.compute(params, data_handle, norm=norm)

    def free(self):
        self.kernel.free()


@register_backend("cuda_v2")
@register_backend("cuda64_v2")
class CUDABackendV2(_CUDABackend):
    def _make_kernel(self, kc, bs):
        from ampfit._cuda_v2 import CUDAKernelV2 as K
        return K(kc, batch_size=bs)


@register_backend("cuda32_v2")
class CUDABackendV2F32(_CUDABackend):
    dtype = np.float32
    def _make_kernel(self, kc, bs):
        from ampfit._cuda_v2_f32 import CUDAKernelV2F32 as K
        return K(kc, batch_size=bs)


@register_backend("cuda")
@register_backend("cuda64")
@register_backend("cuda_v3")
@register_backend("cuda64_v3")
class CUDABackendV3(_CUDABackend):
    def _make_kernel(self, kc, bs):
        from ampfit._cuda_v3 import CUDAKernelV3 as K
        return K(kc, batch_size=bs)


@register_backend("cuda32_v3")
class CUDABackendV3F32(_CUDABackend):
    dtype = np.float32
    def _make_kernel(self, kc, bs):
        from ampfit._cuda_v3_f32 import CUDAKernelV3F32 as K
        return K(kc, batch_size=bs)


@register_backend("cuda_mixed_v3")
class CUDABackendV3Mixed(_CUDABackend):
    def _make_kernel(self, kc, bs):
        from ampfit._cuda_v3_mixed import CUDAKernelV3Mixed as K
        return K(kc, batch_size=bs)


@register_backend("cuda_v3_cache")
class CUDABackendV3Cache(_CUDABackend):
    """CUDA v3 cache — lazy amplitude caching for fast data NLL.

    Suitable as the base backend for IntegratedBackend.  On first
    compute() call, caches per-wave complex amplitudes.  Subsequent
    calls skip BW/angular/FF evaluation.
    """
    def _make_kernel(self, kc, bs):
        from ampfit._cuda_v3_cache import CUDAKernelV3Cache as K
        return K(kc, batch_size=bs)


@register_backend("cuda_v3_split")
class CUDABackendV3Split(_CUDABackend):
    """CUDA v3 split — split-kernel variant for lower register pressure.

    Same Catmull-Rom physics as v3, but forward (compute_main) is split
    into compute_bw_amp + amp_reduce_time, and backward (gradient) is
    split into grad_ck + grad_bw_dom + grad_g0 + grad_scalar kernels.
    Reduces register pressure from 56→48 (forward) and 80→48/40 (backward).
    """
    def _make_kernel(self, kc, bs):
        from ampfit._cuda_v3_split import CUDAKernelV3Split as K
        return K(kc, batch_size=bs)
