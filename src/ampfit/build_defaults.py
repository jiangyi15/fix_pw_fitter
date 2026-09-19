"""Build defaults: temporary ``with`` overrides of the built-in defaults.

There is exactly one mechanism: ``with scope(...)``.  Inside the
scope the given values apply; on exit everything is back to the global
defaults::

    import ampfit

    with ampfit.build_defaults.scope(n_interp=4000, d=1.5):
        model = build_amplitude_model(cfg)
        kc = model.build_kernel_config()
    # -> n_interp / d are back to the global defaults

Build code reads :func:`get` (active override, else the global default), so
config files stay purely physics.  Built-in defaults:

* ``n_interp`` — sampling points of the ``fl``/``gamma`` interpolation tables.
* ``d`` — Blatt-Weisskopf radius used when a particle/decay omits it.
* ``barrier`` — default barrier type used when a decay omits it.
* ``complex_tail`` — ``(magnitude_suffix, phase_suffix)`` naming the two
  real slots of a complex ck parameter (default ``("r", "i")``).

``contextvars``-backed, so scopes are isolated per thread/task.  Leaf module.
"""
import contextlib
import contextvars

__all__ = ["DEFAULTS", "get", "current", "scope"]

DEFAULTS = {
    "n_interp": 2000,
    "d": 3.0,
    "barrier": "bw",
    "complex_tail": ("r", "i"),
}

_state: "contextvars.ContextVar[dict | None]" = contextvars.ContextVar(
    "ampfit_build_defaults", default=None)


def _overrides():
    return _state.get() or {}


def get(name, default=None):
    """Active value: the in-scope override, else the global default."""
    overrides = _overrides()
    if name in overrides:
        return overrides[name]
    return DEFAULTS.get(name, default)


def current():
    """Copy of the active settings (global defaults + active overrides)."""
    return {**DEFAULTS, **_overrides()}


@contextlib.contextmanager
def scope(**kwargs):
    """Temporary override; restores the global values on exit."""
    token = _state.set({**_overrides(), **kwargs})
    try:
        yield
    finally:
        _state.reset(token)
