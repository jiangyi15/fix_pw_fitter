"""Core types and registration machinery for compute backends."""
import numpy as np

ALL_BACKENDS = {}
# name -> cls for names that are unambiguous across models (no model needed)
UNIVERSAL_BACKENDS = {}
# name -> cls for model-independent backends (e.g. the shard wrapper)
MODEL_BACKENDS = {}
# model_name -> {name: cls}; a model NAME (string) scopes its own names,
# so the same name (e.g. "default") can map to different classes per model.
_AMBIGUOUS = set()


def per_event_dnorm(norm, P, weight, bkg=None):
    """d(NLL)/d(norm) for the per-event NLL  NLL = -Σ w·log(P/norm + bkg).

    ``dNLL/dnorm = Σ w·P / (norm·(P + bkg·norm))``.

    This is the *analytical reference* a per-event backend reports as its
    native ``_last_dnorm``.  Backends whose objective is NOT the per-event
    log (e.g. cuda_v5_pwa group-log) must supply the value from their own
    kernel instead — never from this helper.
    """
    w = np.asarray(weight, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    if bkg is None or np.isscalar(bkg):
        b = np.full_like(w, 1.0 if bkg is None else float(bkg))
    else:
        b = np.asarray(bkg, dtype=np.float64)
    if b.shape != w.shape:
        b = np.full_like(w, float(np.ravel(b)[0]))
    return float(np.sum(w * P / (norm * (P + b * norm))))


def register_backend(name, model=None):
    """Decorator: register a backend class under a single *name*.

    *model* is the amplitude-model **name** (a plain string — never a
    class), or ``None`` for a model-independent backend.  Stack the
    decorator to add aliases and/or other models::

        @register_backend("cuda64",  model="flavour_tag_mix")
        @register_backend("cuda",    model="flavour_tag_mix")
        @register_backend("default", model="flavour_tag_mix")
        class CUDABackendV3(...): ...

    The same class may be registered as many names and for many models.
    """
    def _f(cls):
        if model is None:
            UNIVERSAL_BACKENDS[name] = cls
        else:
            MODEL_BACKENDS.setdefault(model, {})[name] = cls
        # Flat convenience index: drop a name as soon as it is ambiguous.
        if name in ALL_BACKENDS and ALL_BACKENDS[name] is not cls:
            _AMBIGUOUS.add(name)
            ALL_BACKENDS.pop(name, None)
        elif name not in _AMBIGUOUS:
            ALL_BACKENDS[name] = cls
        return cls
    return _f


def backends_for_model(model_name):
    """Names registered for *model_name* plus the model-independent names.

    *model_name* is used verbatim as the registry key (callers pass the
    canonical ``model.name``); the registry imports nothing from ampfit.
    """
    names = set(UNIVERSAL_BACKENDS)
    names.update(MODEL_BACKENDS.get(model_name, {}))
    return frozenset(names)


def backend_class(name, model=None):
    """Class registered as *name*, scoped to *model* when given."""
    if model is not None:
        cls = MODEL_BACKENDS.get(model, {}).get(name)
        if cls is not None:
            return cls
        cls = UNIVERSAL_BACKENDS.get(name)
        if cls is not None:
            return cls
        raise ValueError(
            f"unknown backend {name!r} for amplitude model {model!r}; "
            f"available: {sorted(backends_for_model(model))}")
    if name in _AMBIGUOUS:
        raise ValueError(
            f"backend {name!r} is registered for several amplitude models; "
            f"pass a model")
    cls = ALL_BACKENDS.get(name)
    if cls is None:
        raise ValueError(
            f"unknown backend {name!r}; available: {sorted(ALL_BACKENDS)}")
    return cls


def _spec_name(spec):
    """Backend name from a spec (``str`` or ``{"name": ...}``), else None."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return spec.get("name")
    return None


def resolve_backend_spec(spec=None, *, config_spec=None, allowed=None):
    """Normalise and validate a backend spec — without constructing it.

    Precedence: explicit *spec* > *config_spec* > ``"default"`` (a backend
    registered under that name for the model).  When *allowed* is given
    (typically ``model.backends``), a named backend outside it raises; a
    backend instance passes through unchanged.

    Pure (no side effects), so callers can validate before building the
    kernel config.
    """
    chosen = spec if spec is not None else config_spec
    if chosen is None:
        chosen = "default"
    name = _spec_name(chosen)
    if name is not None and allowed is not None and name not in allowed:
        raise ValueError(
            f"backend {name!r} is not registered for this amplitude model; "
            f"allowed: {sorted(allowed)}")
    return chosen


def eval_backend_spec(spec, kernel_config, model=None):
    """Recursively resolve a backend spec to an instance.

    *model* is the amplitude-model name used to look the backend up in its
    model-scoped namespace (falls back to the model-independent names).

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
        return backend_class(spec, model)(kernel_config)

    if not isinstance(spec, dict):
        raise TypeError(f"Expected str or dict, got {type(spec).__name__}")

    spec = dict(spec)
    name = spec.pop("name", None)
    if name is None:
        raise ValueError("Dict spec must have a 'name' key; "
                         f"got keys: {list(spec.keys())}")
    cls = backend_class(name, model)

    # Pass all kwargs through directly — backend constructors that
    # need nested backends (e.g. ``base``) call create_backend themselves.
    return cls(kernel_config, **spec)


def create_backend(spec, kernel_config, *, model=None, **kwargs):
    """Factory: instantiate a backend.

    Args:
        spec: string name, or dict with ``"name"`` + kwargs.
              Kwarg values that are strings or dicts are recursively
              resolved as backend specs.
        kernel_config: config dict from ``Config.build_all_index()``.
        **kwargs: extra arguments (convenience, merged into dict spec).

    Returns:
        A :class:`ComputeBackend` instance (an instance *spec* is returned
        unchanged, so callers need no isinstance branch).
    """
    if isinstance(spec, ComputeBackend):
        return spec
    import gc
    gc.collect()
    if kwargs:
        if isinstance(spec, str):
            spec = {"name": spec, **kwargs}
        elif isinstance(spec, dict):
            spec = {**spec, **kwargs}
    return eval_backend_spec(spec, kernel_config, model)


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

    dNLL/dnorm contract: whenever ``compute(..., norm is not None)`` runs,
    the returned ``grads_dict`` MUST contain ``grads["norm"]`` — the native
    d(NLL)/d(norm) of the backend's OWN objective (per-event NLL or the
    v5 group-log).  The fitter reads it from the returned gradient dict; it
    is never derived from P.  Per-event backends may use
    :func:`per_event_dnorm`; group-log/other objectives must provide the
    value from their kernel.
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
