"""cuda_flat_cache_pwa_gen backend — flat ``pwa_gen`` with a cached ``F``.

Thin registration shim; the kernel and its backend-internal dedup live in
:mod:`tabpwa.cuda._flat_cache_pwa_gen`.
"""
from .core import register_backend
from tabpwa.cuda import _flat_cache_pwa_gen as _k


@register_backend("cuda_flat_cache_pwa_gen", model="pwa_gen")
class CUDAFlatCachePWAGenBackend(_k.CUDAFlatCachePWAGenBackend):
    """Registered alias of the wrapper in ``tabpwa.cuda._flat_cache_pwa_gen``."""
