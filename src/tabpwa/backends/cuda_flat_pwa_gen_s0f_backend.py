"""cuda_flat_pwa_gen_s0f backend — the s0e fused tile on the flat ``pwa_gen``
config with a FIXED-WAVE amplitude cache.

Case-1 copy of ``cuda_flat_pwa_gen_s0e2``; the kernel lives in
:mod:`tabpwa.cuda._flat_pwa_gen_s0f`.  A wave whose every ``coeff`` column is
fixed has an iteration-invariant denominator, so its whole contribution is
cached per handle.  Pass ``fixed_cols`` + ``fixed_values`` TOGETHER — the ONE
declaration (either alone raises); the name helper is
``fixed_coeff_spec.resolve_fixed_cols``.  With nothing declared the kernel runs
the parent's code path (mask-off is not bit-identical: the summation order
differs by ~2e-15).

A multi-BW chain (``n_res != 1``) is rejected at construction with a
``ValueError`` naming ``cuda_flat_pwa_gen2``.  With a mask on,
``grads["coeff"][m] == 0.0`` for every column of a fixed wave — a fixed
parameter has no ``x`` direction.
"""
from .core import register_backend
from tabpwa.cuda import _flat_pwa_gen_s0f as _k


@register_backend("cuda_flat_pwa_gen_s0f", model="pwa_gen",
                  tags=("n_res1", "fixed_coeff", "fp32_cache"))
class CUDAFlatPwaGenS0FBackend(_k.CUDAFlatPwaGenS0FBackend):
    """Registered alias of the wrapper in ``tabpwa.cuda._flat_pwa_gen_s0f``."""
