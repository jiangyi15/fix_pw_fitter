"""cuda_flat_pwa_gen_s0c2 — s0c's compute over the flat ``pwa_gen`` config,
with the three per-chunk feature reductions fused into ONE launch.

Case-1 copy of :mod:`tabpwa.backends.cuda_flat_pwa_gen_s0c`; the kernel lives
in :mod:`tabpwa.cuda._flat_pwa_gen_s0c2` and is bit-identical to s0c (same tree
body, same atomics, only the launch structure changes).  A multi-BW chain
(``n_res != 1``) is rejected at construction with a ``ValueError`` naming
``cuda_flat_pwa_gen2``.
"""
from .core import register_backend
from tabpwa.cuda import _flat_pwa_gen_s0c2 as _k


@register_backend("cuda_flat_pwa_gen_s0c2", model="pwa_gen")
class CUDAFlatPwaGenS0C2Backend(_k.CUDAFlatPwaGenS0C2Backend):
    """Registered alias of the wrapper in ``tabpwa.cuda._flat_pwa_gen_s0c2``."""
