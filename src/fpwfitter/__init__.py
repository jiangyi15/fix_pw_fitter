"""
fpwfitter  –  Fixed Partial Waves Fitter  (GPU-accelerated)

All data (F_data, w_data, B_data) is uploaded to GPU at creation and
kept there permanently.  Evaluation runs entirely on GPU.

Quick start::

    from fpwfitter import FpwFitter

    # One-shot from MC data (M pre-computed via NumPy, data uploaded to GPU)
    fitter = FpwFitter.from_mc(
        F_data, F_mc, w_data, w_mc, B_data, B_mc, purity=0.8
    )

    # Instant from pre-computed M (no MC data needed)
    fitter = FpwFitter.from_M(F_data, w_data, B_data, M, N_b, purity=0.8)

    # Evaluate  (all on GPU, only NLL + gradient copied back)
    nll, grad = fitter.evaluate(c)
"""

from .core import FpwFitter
from .compute_m import compute_M, compute_M_mmap

__all__ = ["FpwFitter", "compute_M", "compute_M_mmap"]
__version__ = "0.1.0"
