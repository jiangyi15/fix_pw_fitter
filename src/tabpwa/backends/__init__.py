"""
Compute backends for tabpwa Fitter.

Each backend wraps a kernel implementation with a uniform interface.
Backends are registered with ``@register_backend(name, model=...)`` (a
single name per decorator; stack it for aliases) and instantiated via
:func:`create_backend`.  ``model`` is the amplitude-model *name* (a string)
or ``None`` for a model-independent backend; each model derives its valid
backend set from :func:`backends_for_model`.

:func:`create_backend` accepts a plain name or a dict::

    create_backend("cuda_v3", kc)
    create_backend({"name": "integrated", "base": "cuda_v3"}, kc)

Kwargs are passed through to the constructor unchanged; composers such as
``integrated`` / ``shard`` receive their nested ``base`` / ``backends``
specs unchanged and resolve them with :func:`create_backend` themselves.
"""
from .core import ALL_BACKENDS, UNIVERSAL_BACKENDS, MODEL_BACKENDS, \
    register_backend, backends_for_model, backend_class, \
    resolve_backend_spec, create_backend, DataHandle, ComputeBackend

from . import numpy_backend        # noqa: F401 — register NumpyBackend
from . import cuda_backends        # noqa: F401 — register CUDA backends
from . import onnx_backend         # noqa: F401 — register ONNXBackend
from . import integrated_backend   # noqa: F401 — register IntegratedBackend
from . import cpu_backend          # noqa: F401 — register CPU backends
from . import shard_backend        # noqa: F401 — register ShardBackend
from . import numpy_pwa_backend    # noqa: F401 — register NumpyPWABackend
from . import integrated_pwa_backend  # noqa: F401 — register IntegratedPWABackend

__all__ = [
    "ALL_BACKENDS", "UNIVERSAL_BACKENDS", "MODEL_BACKENDS",
    "register_backend", "backends_for_model", "backend_class",
    "resolve_backend_spec", "create_backend", "DataHandle", "ComputeBackend",
]
