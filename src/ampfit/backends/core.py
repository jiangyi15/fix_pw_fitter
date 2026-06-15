"""Core types and registration machinery for compute backends."""
import numpy as np

ALL_BACKENDS = {}


def register_backend(name):
    """Decorator: register a backend class under *name*."""
    def _f(cls):
        ALL_BACKENDS[name] = cls
        return cls
    return _f


def create_backend(name, kernel_config, **kwargs):
    """Factory: instantiate a backend by name."""
    if name not in ALL_BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. "
                         f"Available: {list(ALL_BACKENDS.keys())}")
    return ALL_BACKENDS[name](kernel_config, **kwargs)


class DataHandle:
    """Opaque handle for data loaded on a backend."""
    pass


class ComputeBackend:
    """Abstract compute backend.

    Subclasses must implement:
      load_data(self, data_np) -> DataHandle
      compute(self, params, data_handle, norm) -> (Q, grads_dict, P)
      free(self)
    """
    dtype = np.float64

    def load_data(self, data_np):
        raise NotImplementedError

    def compute(self, params, data_handle, norm=None):
        raise NotImplementedError

    def free(self):
        pass
