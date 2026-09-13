"""Amplitude-model objects.

An :class:`AmplitudeModel` is constructed with the full :class:`Config` and
owns the model-specific behaviour:

* ``build_kernel_config()``  — produce the kernel-config dict for this
  model (shape/annotate the base index config);
* ``build_params_transform()`` — return the model's parameter transform
  (a :class:`~ampfit.kernel_params.BuildKernelParams` subclass) that maps
  resolved parameters ↔ kernel arrays.  Each model has its OWN transform,
  so there is no runtime "does it have scalars?" branching.

Model selection (highest priority first):

1. top-level ``amp_model: <name>`` in the config, or
2. ``data.amp_model: <name>`` (legacy location; plain string or one-key
   dict), or
3. the default ``"pwa"``.

Registered models:

* ``pwa`` — pure projection-sum PWA: ck/m0/g0 only, no time/mixing/scalars.
* ``flavour_tag_mix`` (aliases ``flour_tag_mix``, ``p4_directly``) — legacy
  time-dependent flavour-tagged mixing: adds the six scalar parameters.
"""

from ampfit.kernel_params import (
    BuildKernelParams, PWAKernelParams, FlavourTagMixKernelParams)

# Legacy time/mixing scalar parameters (D0-D0bar flavour-tagged mixing).
LEGACY_SCALAR_NAMES = ["gamma", "delta_gamma", "delta_m", "A_prod",
                       "poqr", "poqi"]
LEGACY_SCALAR_DEFAULTS = {"gamma": 0.0, "delta_gamma": 0.0, "delta_m": 0.506,
                          "A_prod": 0.0, "poqr": 1.0, "poqi": 0.0}

AMPLITUDE_MODELS = {}


def register_amplitude_model(*names):
    """Register an :class:`AmplitudeModel` subclass under one or more names."""
    def _f(cls):
        for n in names:
            AMPLITUDE_MODELS[n] = cls
        return cls
    return _f


def _model_name_value(v):
    """Normalise one ``amp_model`` value to a name (or None if empty).

    Accepts a plain string, a one-key dict (legacy form), or a list/tuple
    whose first entry is the name.  Empty containers / blank strings yield
    None so the caller can fall back to the next location.
    """
    if isinstance(v, dict):
        v = next(iter(v), None)
    elif isinstance(v, (list, tuple)):
        v = v[0] if v else None
    if isinstance(v, str):
        v = v.strip() or None
    return v if isinstance(v, str) else None


def _amp_model_name(dic):
    """Explicit model name from ``amp_model`` / ``data.amp_model`` (or None).

    An empty top-level ``amp_model`` (``{}`` / ``[]`` / ``""``) counts as
    absent, so ``data.amp_model`` still applies.
    """
    name = _model_name_value(dic.get("amp_model"))
    if name is None:
        name = _model_name_value((dic.get("data") or {}).get("amp_model"))
    return name


def build_amplitude_model(config):
    """Instantiate the config's amplitude model (default ``pwa``)."""
    name = _amp_model_name(config.dic)
    cls = AMPLITUDE_MODELS["pwa"] if name is None else AMPLITUDE_MODELS.get(name)
    if cls is None:
        raise ValueError(
            f"unknown amp_model {name!r}; available: "
            f"{sorted(AMPLITUDE_MODELS)}")
    return cls(config)


class AmplitudeModel:
    """Base class — full access to the :class:`Config` object."""

    name = "pwa"
    default_scalar_names = ()
    default_scalar_defaults = None
    params_transform_cls = PWAKernelParams

    def __init__(self, config):
        self.config = config

    # -- model views (used by Config / Fitter / reporting) --------------
    @property
    def scalar_names(self):
        explicit = self.config.dic.get("scalar_names")
        if explicit is not None:
            return list(explicit)
        return list(self.default_scalar_names)

    @property
    def scalar_defaults(self):
        """Model defaults merged with (and overridden by) the config's
        explicit ``scalar_defaults`` — a partial override keeps the legacy
        per-name fallbacks (e.g. delta_m, poqr)."""
        merged = dict(self.default_scalar_defaults or {})
        explicit = self.config.dic.get("scalar_defaults")
        if explicit:
            merged.update(explicit)
        return merged

    @property
    def n_proj(self):
        """Projection count: explicit ``n_proj`` else external spin states."""
        explicit = self.config.dic.get("n_proj")
        if explicit is not None:
            return int(explicit)
        n = self._spin_state_count(self.config.top)
        for f in self.config.finals:
            n *= self._spin_state_count(f)
        return max(1, n)

    @property
    def angle_formula(self):
        mode = self.config.dic.get("angle_formula", "helicity")
        if mode not in ("helicity", "cache"):
            raise ValueError(
                f"angle_formula must be 'helicity' or 'cache', got {mode!r}")
        return mode

    def _spin_state_count(self, name):
        d = self.config.dic.get("particle", {}).get(name)
        if not isinstance(d, dict):
            return 1
        spins = d.get("spins")
        if spins is not None:
            return max(1, len(list(spins)))
        return int(2 * d.get("J", 0)) + 1

    # -- what the model produces ----------------------------------------
    def build_kernel_config(self):
        """Kernel config for this model (base index config, model-shaped)."""
        return self.config._build_base_kernel_config()

    def build_params_transform(self) -> BuildKernelParams:
        """Resolved ↔ kernel parameter transform for this model.

        The model owns the transform's constructor input, so callers only
        see the returned object.
        """
        return self.params_transform_cls(self)


@register_amplitude_model("pwa")
class PWA(AmplitudeModel):
    """Scalar-free projection-sum PWA (default)."""
    name = "pwa"
    params_transform_cls = PWAKernelParams


@register_amplitude_model("flavour_tag_mix", "flour_tag_mix", "p4_directly")
class FlavourTagMix(AmplitudeModel):
    """Legacy time-dependent flavour-tagged mixing model."""
    name = "flavour_tag_mix"
    default_scalar_names = tuple(LEGACY_SCALAR_NAMES)
    default_scalar_defaults = dict(LEGACY_SCALAR_DEFAULTS)
    params_transform_cls = FlavourTagMixKernelParams
