"""
Compute backends for ampfit Fitter.

Each backend wraps a kernel implementation with a uniform interface.
Backends are registered with ``@register_backend(name)`` and
instantiated via :func:`create_backend`.

:func:`create_backend` accepts a plain name or a dict::

    create_backend("cuda_v3", kc)
    create_backend({"name": "integrated", "base": "cuda_v3"}, kc)

Kwargs whose values are strings or dicts are recursively resolved
as backend specs — so ``"base"`` in the example above is itself
evaluated as a backend before being passed to ``IntegratedBackend``.
"""
from .core import ALL_BACKENDS, register_backend, create_backend, \
    DataHandle, ComputeBackend

from . import numpy_backend        # noqa: F401 — register NumpyBackend
from . import cuda_backends        # noqa: F401 — register CUDA backends
from . import onnx_backend         # noqa: F401 — register ONNXBackend
from . import integrated_backend   # noqa: F401 — register IntegratedBackend
from . import cpu_backend          # noqa: F401 — register CPU backends
from . import shard_backend        # noqa: F401 — register ShardBackend


__all__ = [
    "ALL_BACKENDS", "register_backend", "create_backend",
    "DataHandle", "ComputeBackend",
]
