"""cuda_flat_pwa_gen_ampcache backend — the flat ``pwa_gen`` config with every
coefficient pinned, so the whole per-wave amplitude is a per-handle cache.

Case-1 copy of ``cuda_flat_pwa_gen_s0f`` specialised to "every coefficient is
pinned"; the kernel lives in :mod:`tabpwa.cuda._flat_pwa_gen_ampcache`.  The
flat family's analogue of ``cuda32_v4_pwa_cache``: one cache, one ck
contraction per iteration, and ``grads["coeff"] == 0`` for every column by
contract.

ONE lifecycle (``cuda_v4_pwa_cache``'s): ``load_data`` never fills, and each
handle's cache is filled on that handle's FIRST ``compute`` from
``params["coeff"]`` — with no ``coeff=`` constructor argument, so the state
cannot be reached two ways.  From the handle's second call on the snapshot is
strict and a ``coeff`` that moved raises ``ValueError`` (use
``cuda_flat_pwa_gen_s0e2`` when the weights float).  A multi-BW chain
(``n_res > 1``) is supported: the denominator is the product over the factor
axis, exactly as the reference computes it.

This module OWNS the ``coeff_pinned`` capability tag (s0f/s0g's split): the
kernel module registers the same name untagged, and a name registered twice
keeps only the winning registration's tags — so declaring it in one place is
what keeps "which suites sweep this backend" from depending on import order.
"""
from .core import register_backend
from tabpwa.cuda import _flat_pwa_gen_ampcache as _k


@register_backend("cuda_flat_pwa_gen_ampcache", model="pwa_gen",
                  tags=("coeff_pinned", "fp32_cache"))
class CUDAFlatPwaGenAmpCacheRegistered(_k.CUDAFlatPwaGenAmpCacheBackend):
    """Registered alias of the wrapper in
    ``tabpwa.cuda._flat_pwa_gen_ampcache``.  The ONLY place the
    ``coeff_pinned`` tag is declared."""
