"""
ampfit — Amplitude analysis fitting framework.

Provides a complete pipeline for partial wave amplitude analysis:
  - Config loading (particle models, decay chains)
  - NumPy and CUDA compute kernels with exact gradients
  - Parameter constraint management (fixed, same, scale)
  - Global Fitter class for optimizer-ready NLL computation
"""

from .config_loader import Config
from .numpy_kernel import NumpyKernel
from .param_constraint import (
    ParameterConstraint, BoundTransform,
    VariableRegistry, ConstraintManager,
)

try:
    from ._cuda import CUDAKernel
except Exception:
    CUDAKernel = None

from .fitter import Fitter
from . import backends
from .bw_form_factor import form_factor as bw_form_factor

__all__ = [
    "Config",
    "NumpyKernel",
    "CUDAKernel",
    "ParameterConstraint",
    "BoundTransform",
    "VariableRegistry",
    "ConstraintManager",
    "Fitter",
    "backends",
    "bw_form_factor",
]
