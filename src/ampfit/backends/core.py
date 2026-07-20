"""Core types and registration machinery for compute backends."""
import numpy as np

ALL_BACKENDS = {}


def register_backend(name):
    """Decorator: register a backend class under *name*."""
    def _f(cls):
        ALL_BACKENDS[name] = cls
        return cls
    return _f


def eval_backend_spec(spec, kernel_config):
    """Recursively resolve a backend spec to an instance.

    Supported forms:

    * ``"numpy"`` — simple name.
    * ``{"name": "cuda_v3", "batch_size": 50000}`` — name + kwargs.
    * ``{"name": "integrated", "base": "cuda_v3"}`` — nested spec;
      *base* is itself a backend spec, resolved recursively.

    Any kwarg whose value is a ``str`` or ``dict`` is treated as a
    nested backend spec and resolved before being passed to the
    parent backend's constructor.

    Returns:
        A :class:`ComputeBackend` instance.
    """
    if isinstance(spec, str):
        if spec not in ALL_BACKENDS:
            raise ValueError(f"Unknown backend '{spec}'. "
                             f"Available: {list(ALL_BACKENDS.keys())}")
        return ALL_BACKENDS[spec](kernel_config)

    if not isinstance(spec, dict):
        raise TypeError(f"Expected str or dict, got {type(spec).__name__}")

    spec = dict(spec)
    name = spec.pop("name", None)
    if name is None:
        raise ValueError("Dict spec must have a 'name' key; "
                         f"got keys: {list(spec.keys())}")
    if name not in ALL_BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. "
                         f"Available: {list(ALL_BACKENDS.keys())}")

    cls = ALL_BACKENDS[name]

    # Pass all kwargs through directly — backend constructors that
    # need nested backends (e.g. ``base``) call create_backend themselves.
    return cls(kernel_config, **spec)


def create_backend(spec, kernel_config, **kwargs):
    """Factory: instantiate a backend.

    Args:
        spec: string name, or dict with ``"name"`` + kwargs.
              Kwarg values that are strings or dicts are recursively
              resolved as backend specs.
        kernel_config: config dict from ``Config.build_all_index()``.
        **kwargs: extra arguments (convenience, merged into dict spec).

    Returns:
        A :class:`ComputeBackend` instance.
    """
    import gc
    gc.collect()
    if isinstance(spec, dict) and kwargs:
        spec = {**spec, **kwargs}
    return eval_backend_spec(spec, kernel_config)


class DataHandle:
    """Opaque handle for data loaded on a backend."""
    def free(self):
        pass
    def __del__(self):
        self.free()


class ComputeBackend:
    """Abstract compute backend.

    Subclasses must implement:
      load_data(self, data_np) -> DataHandle
      compute(self, params, data_handle, norm, return_p) -> (Q, grads_dict, P)
      free(self)
    """
    dtype = np.float64

    def load_data(self, data_np):
        raise NotImplementedError

    def compute(self, params, data_handle, norm=None, return_p=True):
        """Compute forward + backward pass.

        Args:
            params: dict with 'ck', 'm0', 'g0', 'scalar'.
            data_handle: DataHandle from load_data().
            norm: optional float normalization factor.
            return_p: if True, return per-event P; otherwise None.

        Returns:
            (Q, grads_dict, P_or_None).
        """
        raise NotImplementedError

    def free(self):
        pass

    def __del__(self):
        self.free()
