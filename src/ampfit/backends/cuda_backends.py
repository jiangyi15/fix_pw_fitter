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
        from ampfit.cuda._v2 import CUDAKernelV2 as K
        return K(kc, batch_size=bs)


@register_backend("cuda32_v2")
class CUDABackendV2F32(_CUDABackend):
    dtype = np.float32
    def _make_kernel(self, kc, bs):
        from ampfit.cuda._v2_f32 import CUDAKernelV2F32 as K
        return K(kc, batch_size=bs)


@register_backend("cuda")
@register_backend("cuda64")
@register_backend("cuda_v3")
@register_backend("cuda64_v3")
class CUDABackendV3(_CUDABackend):
    def _make_kernel(self, kc, bs):
        from ampfit.cuda._v3 import CUDAKernelV3 as K
        return K(kc, batch_size=bs)


@register_backend("cuda32_v3")
class CUDABackendV3F32(_CUDABackend):
    dtype = np.float32
    def _make_kernel(self, kc, bs):
        from ampfit.cuda._v3_f32 import CUDAKernelV3F32 as K
        return K(kc, batch_size=bs)


@register_backend("cuda_mixed_v3")
class CUDABackendV3Mixed(_CUDABackend):
    def _make_kernel(self, kc, bs):
        from ampfit.cuda._v3_mixed import CUDAKernelV3Mixed as K
        return K(kc, batch_size=bs)


@register_backend("cuda_v3_cache")
class CUDABackendV3Cache(_CUDABackend):
    """CUDA v3 cache — lazy amplitude caching for fast data NLL.

    Suitable as the base backend for IntegratedBackend.  On first
    compute() call, caches per-wave complex amplitudes.  Subsequent
    calls skip BW/angular/FF evaluation.
    """
    def _make_kernel(self, kc, bs):
        from ampfit.cuda._v3_cache import CUDAKernelV3Cache as K
        return K(kc, batch_size=bs)


@register_backend("cuda_v3_sparse")
class CUDABackendV3Sparse(_CUDABackend):
    """CUDA v3 sparse — sparse scatter/gather for matrix_gamma (99.5% sparse).

    Same split-kernel + FP32 FA as v3_split, but the 288×216 matrix_gamma
    matmul in g_bw and grad_g0 is replaced by scatter/gather via column-index
    array — 0.5% of the original arithmetic.
    """
    def _make_kernel(self, kc, bs):
        from ampfit.cuda._v3_sparse import CUDAKernelV3Sparse as K
        return K(kc, batch_size=bs)


@register_backend("cuda_v3_ampcache")
class CUDABackendV3AmpCache(_CUDABackend):
    """CUDA v3 ampcache — v3 sparse + cached minimal-set angular amplitudes.

    The per-wave amplitude is ``a_w = ck_w·Amp_w(q,angles)/bw_p_w(m)``; the
    angular part (272 unique values/event) is pure kinematics and is cached
    per handle at load_data (fp64 fill, float2 storage), while the BW
    propagator is recomputed each iteration — so m0/g0 remain fitted
    (unlike cuda_v3_cache, which requires them fixed).  Cache is constant
    over the fit, so fp32 storage never moves the minimum.
    """
    def _make_kernel(self, kc, bs):
        from ampfit.cuda._v3_ampcache import CUDAKernelV3AmpCache as K
        return K(kc, batch_size=bs)
