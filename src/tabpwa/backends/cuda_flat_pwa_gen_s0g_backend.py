"""cuda_flat_pwa_gen_s0g backend — the flat ``pwa_gen`` tile with the fixed
cache's block and the free block in ONE un-staged pass.

Case-1 copy of ``cuda_flat_pwa_gen_s0f``; the kernel lives in
:mod:`tabpwa.cuda._flat_pwa_gen_s0g`.  A wave whose every ``coeff`` column is
fixed has an iteration-invariant denominator, so its whole contribution is
cached per handle; unlike s0f there is no F/B pass, no ``A1``/``dA`` buffer and
no scratch — a block carries hundreds of events, and a ``resolution_size > 1``
group closes inside it.

Pass ``fixed_cols`` + ``fixed_values`` (the ONE declaration; names go through
``fixed_coeff_spec.resolve_fixed_cols``); with nothing declared the kernel is
the parent's path.  A multi-BW chain (``n_res != 1``) is rejected at construction
naming ``cuda_flat_pwa_gen2``; a log-sum group too large for one block is
rejected naming ``cuda_flat_pwa_gen_s0f``.  With a mask on,
``grads["coeff"][m] == 0.0`` for every column of a fixed wave.
"""
from .core import register_backend
from tabpwa.cuda import _flat_pwa_gen_s0g as _k


@register_backend("cuda_flat_pwa_gen_s0g", model="pwa_gen",
                  tags=("n_res1", "fixed_coeff"))
class CUDAFlatPwaGenS0GBackend(_k.CUDAFlatPwaGenS0GBackend):
    """Registered alias of the wrapper in ``tabpwa.cuda._flat_pwa_gen_s0g``."""
