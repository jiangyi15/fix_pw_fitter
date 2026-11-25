"""cuda_flat_pwa_gen2 backend — the flat ``pwa_gen`` tile with four reductions.

Thin registration shim; the kernel lives in
:mod:`tabpwa.cuda._flat_pwa_gen2`.
"""
from .core import register_backend
from tabpwa.cuda import _flat_pwa_gen2 as _k


@register_backend("cuda_flat_pwa_gen2", model="pwa_gen", tags=("fp32_cache",))
class CUDAFlatPwaGen2Backend(_k.CUDAFlatPwaGen2Backend):
    """Registered alias of the wrapper in ``tabpwa.cuda._flat_pwa_gen2``."""
