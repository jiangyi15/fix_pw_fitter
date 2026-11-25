"""cuda_flat_pwa_gen_s0e4 backend — the s0e fused tile on the flat
``pwa_gen`` config with the coefficient pre-gathered in CSR order and the
heavy ``gamma`` rows split across two threads.

Thin registration shim; the kernel lives in
:mod:`tabpwa.cuda._flat_pwa_gen_s0e4`.  A multi-BW chain (``n_res != 1``) is
rejected at construction with a ``ValueError`` naming ``cuda_flat_pwa_gen2``.
"""
from .core import register_backend
from tabpwa.cuda import _flat_pwa_gen_s0e4 as _k


@register_backend("cuda_flat_pwa_gen_s0e4", model="pwa_gen", tags=('n_res1', "fp32_cache"))
class CUDAFlatPwaGenS0E4Backend(_k.CUDAFlatPwaGenS0E4Backend):
    """Registered alias of the wrapper in ``tabpwa.cuda._flat_pwa_gen_s0e4``."""
