"""Core types and registration machinery for compute backends."""
import numpy as np

ALL_BACKENDS = {}
# Backend name -> frozenset of amplitude-model names it serves
# (None = universal, e.g. the shard wrapper).
BACKEND_MODELS = {}


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


def register_backend(*names, amp_model=None):
    """Decorator: register a backend class under one or more *names*.

    Args:
        *names: backend name(s), e.g. ``@register_backend("cuda", "cuda64")``.
        amp_model: amplitude-model name (or iterable of names) this backend
            serves.  ``None`` (default) means the backend is universal
            (usable by any model, e.g. the ``shard`` wrapper).  Models read
            this back to expose their valid backend set, so a backend only
            declares its model once, next to its implementation.
    """
    models = None if amp_model is None else frozenset(
        [amp_model] if isinstance(amp_model, str) else amp_model)

    def _f(cls):
        for name in names:
            ALL_BACKENDS[name] = cls
            BACKEND_MODELS[name] = models
        return cls
    return _f


def validate_backend_spec(spec, allowed=None):
    """Validate a (possibly nested) backend spec.

    Recursion is driven by each backend class's ``nested_specs`` — the
    backend, not the caller/model, knows which of its kwargs hold nested
    specs (e.g. ``base`` for hyper backends, ``backends`` for shard).

    Args:
        spec: ``str`` name or ``dict`` spec (may nest).
        allowed: optional iterable of permitted backend names; a named
            backend outside it raises ``ValueError``.

    Returns *spec* for call chaining.
    """
    if isinstance(spec, str):
        name, d = spec, None
    elif isinstance(spec, dict):
        name, d = spec.get("name"), spec
    else:
        return spec
    if name is not None and allowed is not None and name not in allowed:
        raise ValueError(
            f"backend {name!r} is not registered for this amplitude model; "
            f"allowed: {sorted(allowed)}")
    if d is not None:
        cls = ALL_BACKENDS.get(name)
        for key in getattr(cls, "nested_specs", ()) if cls else ():
            child = d.get(key)
            if child is None:
                continue
            for c in (child if isinstance(child, (list, tuple)) else [child]):
                validate_backend_spec(c, allowed)
    return spec


def backends_for_model(model_name):
    """Names of the backends registered for *model_name* (incl. universal).

    Accepts an amplitude-model alias (e.g. ``p4_directly``) and normalises
    it to the canonical name registered in ``AMPLITUDE_MODELS``.
    """
    try:
        from ampfit.amp_model import AMPLITUDE_MODELS
        cls = AMPLITUDE_MODELS.get(model_name)
        if cls is not None:
            model_name = cls.name
    except Exception:      # pragma: no cover — amp_model always importable
        pass
    return frozenset(n for n, m in BACKEND_MODELS.items()
                     if m is None or model_name in m)


def resolve_backend_spec(config, spec=None, default=None):
    """Backend spec precedence: explicit *spec* > ``config.backend_spec``
    (the config's ``config: {backend: ...}``) > *default*."""
    if spec is not None:
        return spec
    return getattr(config, "backend_spec", None) or default


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

    dNLL/dnorm contract: whenever ``compute(..., norm is not None)`` runs,
    the returned ``grads_dict`` MUST contain ``grads["norm"]`` — the native
    d(NLL)/d(norm) of the backend's OWN objective (per-event NLL or the
    v5 group-log).  The fitter reads it from the returned gradient dict; it
    is never derived from P.  Per-event backends may use
    :func:`per_event_dnorm`; group-log/other objectives must provide the
    value from their kernel.
    """
    dtype = np.float64
    # Kwargs that hold nested backend specs (a str, dict, or list of them).
    # Declared by backends that compose others, e.g. ("base",) or
    # ("backends",); used by :func:`validate_backend_spec`.
    nested_specs = ()

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
