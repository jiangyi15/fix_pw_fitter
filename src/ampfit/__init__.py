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
    CKProduct, BoundTransform,
    VariableRegistry, ConstraintManager,
    Transform, BWParamsTransform,
    transform_from_dict,
    Prior, GaussianPrior,
    prior_from_dict,
)

try:                       # public alias for the CUDA v3 kernel
    from .cuda._v3 import CUDAKernelV3 as CUDAKernel
except Exception:
    CUDAKernel = None

from .fitter import Fitter
from . import backends
from . import constrain_plugins  # noqa: F401 — register constraint handlers
from .bw_form_factor import (
    BarrierFactor,
    barrier_names,
    build_barrier,
    form_factor as bw_form_factor,
    register_barrier,
)

__all__ = [
    "Config",
    "NumpyKernel",
    "CUDAKernel",
    "CKProduct",
    "BoundTransform",
    "VariableRegistry",
    "ConstraintManager",
    "Transform",
    "BWParamsTransform",
    "transform_from_dict",
    "Prior",
    "GaussianPrior",
    "prior_from_dict",
    "Fitter",
    "backends",
    "bw_form_factor",
    "BarrierFactor",
    "register_barrier",
    "build_barrier",
    "barrier_names",
]
