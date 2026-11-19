"""cuda_flat_pwa_gen_s0e2 backend — the s0e fused tile on the flat ``pwa_gen``
config, with the packed-CSR step and the two-chain sparse dot.

Thin registration shim; the kernel lives in
:mod:`tabpwa.cuda._flat_pwa_gen_s0e2`.  A multi-BW chain (``n_res != 1``) is
rejected at construction with a ``ValueError`` naming ``cuda_flat_pwa_gen2``.
"""
from .core import register_backend
from tabpwa.cuda import _flat_pwa_gen_s0e2 as _k


@register_backend("cuda_flat_pwa_gen_s0e2", model="pwa_gen")
class CUDAFlatPwaGenS0E2Backend(_k.CUDAFlatPwaGenS0E2Backend):
    """Registered alias of the wrapper in ``tabpwa.cuda._flat_pwa_gen_s0e2``."""
