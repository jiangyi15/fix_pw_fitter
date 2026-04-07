"""
fpwfitter  –  Fixed Partial Waves Fitter  (GPU + CPU backends)

Import CUDA version (default, requires GPU)::

    from fpwfitter import FpwFitter

Import mixed-precision version (FP32 compute, ~50× faster FP throughput)::

    from fpwfitter import FpwFitterMP

Import pure-NumPy version (no GPU required)::

    from fpwfitter import NumpyFitter

Both have identical API::

    fitter = FpwFitter.from_mc(F_data, F_mc, w_data, w_mc, B_data, B_mc)
    nll, grad = fitter.evaluate(c)
"""

from .core import FpwFitter
from .mp import FpwFitterMP
from .ref_numpy import NumpyFitter
from .compute_m import compute_M, compute_M_mmap


__all__ = ["FpwFitter", "FpwFitterMP", "NumpyFitter",
           "compute_M", "compute_M_mmap"]
__version__ = "0.1.0"
