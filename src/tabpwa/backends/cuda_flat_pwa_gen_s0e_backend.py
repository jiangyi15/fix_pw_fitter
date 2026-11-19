"""cuda_flat_pwa_gen_s0e backend — the s0e fused tile on the flat ``pwa_gen`` config.

Thin registration shim; the kernel lives in
:mod:`tabpwa.cuda._flat_pwa_gen_s0e`.  A multi-BW chain (``n_res != 1``) is
rejected at construction with a ``ValueError`` naming ``cuda_flat_pwa_gen2``.
"""
from .core import register_backend
from tabpwa.cuda import _flat_pwa_gen_s0e as _k


@register_backend("cuda_flat_pwa_gen_s0e", model="pwa_gen", tags=('n_res1',))
class CUDAFlatPwaGenS0EBackend(_k.CUDAFlatPwaGenS0EBackend):
    """Registered alias of the wrapper in ``tabpwa.cuda._flat_pwa_gen_s0e``."""
