"""
Compute backends for ampfit Fitter.

Each backend wraps a kernel implementation with a uniform interface.
Backends are registered with ``@register_backend(name)`` and
instantiated with ``create_backend(name, kernel_config, **kwargs)``.
"""
from .core import ALL_BACKENDS, register_backend, create_backend, \
    DataHandle, ComputeBackend
from . import numpy_backend   # noqa: F401 — register NumpyBackend
from . import cuda_backends   # noqa: F401 — register CUDA backends
from . import onnx_backend    # noqa: F401 — register ONNXBackend

__all__ = [
    "ALL_BACKENDS", "register_backend", "create_backend",
    "DataHandle", "ComputeBackend",
]
