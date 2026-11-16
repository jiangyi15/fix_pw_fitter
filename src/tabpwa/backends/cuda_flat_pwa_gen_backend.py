"""cuda_flat_pwa_gen backend — the flat ``pwa_gen`` config on the GPU.

Thin wrapper: registers the backend class and forwards to the kernel in
:mod:`tabpwa.cuda._flat_pwa_gen`.  The kernel is a standalone case-1 copy
(own ``.cu``/``.so``/wrapper/symbols) and reads the full, non-deduped config.
"""
from .core import register_backend
from tabpwa.cuda import _flat_pwa_gen as _k


@register_backend("cuda_flat_pwa_gen", model="pwa_gen")
class CUDAFlatPWAGenBackend(_k.CUDAFlatPWAGenBackend):
    """Registered alias of the wrapper in ``tabpwa.cuda._flat_pwa_gen``."""
