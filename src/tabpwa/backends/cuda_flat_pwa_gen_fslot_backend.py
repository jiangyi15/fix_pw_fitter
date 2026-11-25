"""cuda_flat_pwa_gen_fslot backend — factor-level dedup of the gradient sweep.

Thin registration shim; the kernel lives in
:mod:`tabpwa.cuda._flat_pwa_gen_fslot`.
"""
from .core import register_backend
from tabpwa.cuda import _flat_pwa_gen_fslot as _k


@register_backend("cuda_flat_pwa_gen_fslot", model="pwa_gen", tags=("fp32_cache",))
class CUDAFlatPwaGenFSlotBackend(_k.FlatPWAGenFSlotBackend):
    """Registered alias of the wrapper in ``tabpwa.cuda._flat_pwa_gen_fslot``."""