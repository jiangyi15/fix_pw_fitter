"""cuda_flat_pwa_gen_s0e5 backend — the s0e fused tile on the flat ``pwa_gen`` config.

Case-1 copy of ``cuda_flat_pwa_gen_s0e2``; the kernel lives in
:mod:`tabpwa.cuda._flat_pwa_gen_s0e5`.  A multi-BW chain (``n_res != 1``) is
rejected at construction with a ``ValueError`` naming ``cuda_flat_pwa_gen2``.
"""
from .core import register_backend
from tabpwa.cuda import _flat_pwa_gen_s0e5 as _k


@register_backend("cuda_flat_pwa_gen_s0e5", model="pwa_gen", tags=('n_res1', "fp32_cache"))
class CUDAFlatPwaGenS0E5Backend(_k.CUDAFlatPwaGenS0E5Backend):
    """Registered alias of the wrapper in ``tabpwa.cuda._flat_pwa_gen_s0e5``."""
