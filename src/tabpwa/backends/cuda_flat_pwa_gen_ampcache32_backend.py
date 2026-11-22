"""cuda_flat_pwa_gen_ampcache32 backend — ``cuda_flat_pwa_gen_ampcache`` with
the per-iteration contraction in fp32 arithmetic.

Case-1 copy of ``cuda_flat_pwa_gen_ampcache`` (which is itself a case-1 copy of
``cuda_flat_pwa_gen_s0f`` specialised to "every coefficient is pinned"); the
kernel lives in :mod:`tabpwa.cuda._flat_pwa_gen_ampcache32`.  The FILL and the
GRAM stay fp64 — the delicate interpolated denominator ``D`` is evaluated once
per handle there — and only the ``A``/``dck`` contraction runs in fp32, with
every objective-feeding reduction in fp64.  ``grads["coeff"] == 0`` for every
column, by contract.

ONE lifecycle (``cuda_v4_pwa_cache``'s): ``load_data`` never fills, and each
handle's cache is filled on that handle's FIRST ``compute`` from
``params["coeff"]`` — with no ``coeff=`` constructor argument, so the state
cannot be reached two ways.  From the handle's second call on the snapshot is
strict and a ``coeff`` that moved raises ``ValueError`` (use
``cuda_flat_pwa_gen_s0e2`` when the weights float).  A multi-BW chain
(``n_res > 1``) is supported: the denominator is the product over the factor
axis, exactly as the reference computes it.

This module OWNS this backend's ``coeff_pinned`` capability tag (s0f/s0g's
split): the kernel module registers the same name untagged, and a name
registered twice keeps only the winning registration's tags — so declaring it
in one place is what keeps "which suites sweep this backend" from depending on
import order.
"""
from .core import register_backend
from tabpwa.cuda import _flat_pwa_gen_ampcache32 as _k


@register_backend("cuda_flat_pwa_gen_ampcache32", model="pwa_gen",
                  tags=("coeff_pinned",))
class CUDAFlatPwaGenAmpCache32Registered(_k.CUDAFlatPwaGenAmpCache32Backend):
    """Registered alias of the wrapper in
    ``tabpwa.cuda._flat_pwa_gen_ampcache32``.  The ONLY place this backend's
    ``coeff_pinned`` tag is declared."""
