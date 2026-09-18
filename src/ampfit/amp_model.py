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

import numpy as np

from ampfit.base_model import BaseModel
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
    """Instantiate the amplitude model for *config* (default ``pwa``).

    *config* may be a :class:`~ampfit.config_loader.Config` instance, a config
    file path, or an already-parsed config dict.
    """
    from ampfit.config_loader import Config
    if not hasattr(config, "dic"):
        config = Config(config)              # file path (str) or raw dict
    name = _amp_model_name(config.dic)
    cls = AMPLITUDE_MODELS["pwa"] if name is None else AMPLITUDE_MODELS.get(name)
    if cls is None:
        raise ValueError(
            f"unknown amp_model {name!r}; available: "
            f"{sorted(AMPLITUDE_MODELS)}")
    return cls(config)


class AmplitudeModel(BaseModel):
    """An interpreted physical model plus amplitude-model policy.

    Inherits the decay tree, index/tables and the predefined base kernel
    config from :class:`~ampfit.base_model.BaseModel`; adds only the fitting
    policy (``name``, scalar policy, ``params_transform_cls``) and the
    ``build_kernel_config`` override seam.
    """

    name = "pwa"
    default_scalar_names = ()
    default_scalar_defaults = None
    params_transform_cls = PWAKernelParams

    def __init__(self, config):
        # ``config`` (a Config/BaseModel) has already interpreted the
        # declarations; adopt that same state (shared object references) so
        # there is ONE interpretation and the lazily-filled index data
        # (``unique_*``, ``m0_phys_name`` ...) stays consistent no matter
        # which object the caller reads.
        base = getattr(config, "base_model", None)   # Config facade
        if base is None:
            base = config                             # a BaseModel directly, or a dict
        if isinstance(base, BaseModel):
            self.__dict__.update(base.__dict__)
        else:
            dic = config if isinstance(config, dict) else getattr(config, "dic", config)
            super().__init__(dic, getattr(config, "_config_path", ""))

    # -- model views (used by Fitter / reporting) ------------------------
    @property
    def scalar_names(self):
        explicit = self.dic.get("scalar_names")
        if explicit is not None:
            return list(explicit)
        return list(self.default_scalar_names)

    @property
    def scalar_defaults(self):
        """Model defaults merged with (and overridden by) the config's
        explicit ``scalar_defaults`` — a partial override keeps the legacy
        per-name fallbacks (e.g. delta_m, poqr)."""
        merged = dict(self.default_scalar_defaults or {})
        explicit = self.dic.get("scalar_defaults")
        if explicit:
            merged.update(explicit)
        return merged

    # -- what the model produces ----------------------------------------
    def build_kernel_config(self):
        """Kernel config for this model (base index config, model-shaped)."""
        return self.build_base_kernel_config()

    def build_event_data(self, momenta, kernel_config, *, weight=None,
                         bkg=None):
        """Build kernel-ready event arrays for this model.

        Default (PWA-style): generic tree fill with the identical-particle /
        CP block expansion.  Models whose kernel consumes time/mixing
        override this (see :class:`FlavourTagMix`).
        """
        from ampfit.pwa_build import build_tree_event_data

        chains_by_topo = {}
        for _, chain in self.full_decay.get_partial_waves():
            chains_by_topo[self.topo_index[chain.topo_id()]] = chain
        spinful = [nm for nm in self.finals
                   if float(self.dic["particle"][nm].get("J", 0)) != 0]
        out = build_tree_event_data(self, kernel_config, chains_by_topo,
                                    momenta, spinful_names=spinful)
        n = momenta.shape[0]
        out["weight"] = (np.ones(n) if weight is None
                         else np.asarray(weight, dtype=float))
        if bkg is not None:
            out["bkg"] = np.asarray(bkg, dtype=np.float64).ravel()
        return out

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

    def build_event_data(self, momenta, kernel_config, *, weight=None,
                         bkg=None):
        """Legacy 24-row layout with the ``frac`` / ``time`` mixing inputs."""
        from ampfit.momenta_to_data import momenta_to_data

        n = momenta.shape[0]
        w = np.ones(n) if weight is None else np.asarray(weight, dtype=float)
        d = momenta_to_data(momenta, weight=w)
        return {
            "mass": d["mass"].reshape(n, -1),
            "q": d["q"].reshape(n, -1),
            "angle": d["angles"],
            "frac": d["frac"],
            "time": d["time"],
            "bkg": (d["bkg_raw"] if bkg is None
                    else np.asarray(bkg, dtype=np.float64).ravel()),
            "weight": np.asarray(d["weight"], dtype=float),
        }
