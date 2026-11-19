"""cuda_flat_pwa_gen_s0c — s0c's compute over the flat ``pwa_gen`` config.

Thin registration shim; the kernel lives in
:mod:`tabpwa.cuda._flat_pwa_gen_s0c`.  A multi-BW chain (``n_res != 1``) is
rejected at construction with a ``ValueError`` naming ``cuda_flat_pwa_gen2``.
"""
from .core import register_backend
from tabpwa.cuda import _flat_pwa_gen_s0c as _k


@register_backend("cuda_flat_pwa_gen_s0c", model="pwa_gen", tags=('n_res1',))
class CUDAFlatPwaGenS0CBackend(_k.CUDAFlatPwaGenS0CBackend):
    """Registered alias of the wrapper in ``tabpwa.cuda._flat_pwa_gen_s0c``."""
