"""Amplitude-model objects.

An :class:`AmplitudeModel` is constructed with the full :class:`Config` and
owns the model-specific behaviour:

* ``build_kernel_config()``  — produce the kernel-config dict for this
  model (shape/annotate the base index config);
* ``build_params_transform()`` — return the model's parameter transform
  (a :class:`~tabpwa.kernel_params.BuildKernelParams` subclass) that maps
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

from tabpwa.base_model import BaseModel
from tabpwa.kernel_params import (
    BuildKernelParams, PWAKernelParams, FlavourTagMixKernelParams,
    PolarisedKernelParams)

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

    *config* may be a :class:`~tabpwa.config_loader.Config` instance, a config
    file path, or an already-parsed config dict.
    """
    from tabpwa.config_loader import RawConfig
    if not hasattr(config, "dic"):
        config = RawConfig(config)           # file path (str) or raw dict
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
    config from :class:`~tabpwa.base_model.BaseModel`; adds only the fitting
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
        # config: a BaseModel (adopt its state), or a raw Config / dict to
        # interpret here.
        if isinstance(config, BaseModel):
            self.__dict__.update(config.__dict__)
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
        from tabpwa.pwa_build import build_tree_event_data

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

    def build_kernel_config(self):
        """Base index config plus the pure-PWA meta keys.

        The kernels/backends read only the base arrays; these meta keys
        describe the PWA basis and are used by the reference engine /
        introspection: external helicity states (``top_states``), wave and
        BW-gamma parameter names, and the identical/CP counts.
        """
        kc = super().build_kernel_config()
        kc["n_identical"] = self.decay_tree.n_perm
        kc["n_cp"] = self.decay_tree.n_cp
        kc["top_states"] = [int(x) for x in self._helicity_top_states()] or None
        kc["wave_names"] = [str(ls) for ls, _ in self.full_decay.get_partial_waves()]
        m0, g0 = [], []
        for _, ch in self.full_decay.get_partial_waves():
            for idx, d in enumerate(ch.decays):
                if idx != 0:
                    nm = d.core.name + "_mass"
                    if nm not in m0:
                        m0.append(nm)
                    for gn in d.core._model.get_gamma_name():
                        if gn not in g0:
                            g0.append(gn)
        kc["m0_names"] = m0
        kc["g0_names"] = g0
        return kc



@register_amplitude_model("polarised_pwa")
class PolarisedPWA(PWA):
    """Projection-sum PWA with a fitted spin-density matrix.

    The intensity couples the top's helicity projections through a spin
    density matrix ``rho = I + sum_l p_l * G_l`` (the default basis is the
    trace-P Hermitian generators, auto-built from the top's helicity
    states; zero parameters reproduces ``pwa`` exactly).  The polarisation
    parameters are declared as model scalars — nothing to configure beyond
    ``amp_model: polarised_pwa`` (fixing/freeing them is the constraint
    system's job).  Currently the top must carry >= 2 helicity states.
    """
    name = "polarised_pwa"
    params_transform_cls = PolarisedKernelParams

    def _n_j(self):
        """Projection count: ``len(spins)`` of the top particle."""
        spins = self.dic["particle"].get(self.top, {}).get("spins") or []
        if spins:
            return len(spins)
        states = self._helicity_top_states()
        if len(states) < 2:
            raise ValueError(
                "polarised_pwa: the top has < 2 helicity states — nothing "
                "to polarise; use amp_model: pwa")
        return len(states)

    @property
    def scalar_names(self):
        explicit = self.dic.get("scalar_names")
        if explicit is not None:
            return list(explicit)
        from tabpwa.spin_density import polarisation_names
        return polarisation_names(self._n_j(), prefix=self.top)

    @property
    def scalar_defaults(self):
        return {n: 0.0 for n in self.scalar_names}

    def build_kernel_config(self):
        """PWA kernel config plus the spin-density meta for the backend."""
        kc = super().build_kernel_config()
        from tabpwa.spin_density import polarisation_names
        P = self._n_j()
        kc["spin_density"] = {"n_j": P,
                              "names": polarisation_names(P, prefix=self.top)}
        return kc


@register_amplitude_model("flavour_tag_mix", "flour_tag_mix", "p4_directly")
class FlavourTagMix(AmplitudeModel):
    """Legacy time-dependent flavour-tagged mixing model."""
    name = "flavour_tag_mix"
    default_scalar_names = tuple(LEGACY_SCALAR_NAMES)
    default_scalar_defaults = dict(LEGACY_SCALAR_DEFAULTS)
    params_transform_cls = FlavourTagMixKernelParams

    def build_event_data(self, momenta, kernel_config, *, weight=None,
                         bkg=None, frac=None, time=None, chunk=None):
        """TD event-data fill, tree-general.

        Two paths:

        * **legacy 4π structure** (four pion finals in two identical
          pairs — the structure :func:`momenta_to_data` implements):
          the production 24-row fill, unchanged.
        * **any other declared structure** (e.g. K*K̄*: no identical
          particles, CP blocks only): the generic tree walk — the same
          per-chain fill the PWA path uses, with the block/topology
          enumeration taken from the decay tree, so the fill and the
          kernel config share one convention by construction.  Rows =
          ``n_blocks × n_topos`` from the declared tree.

        ``frac``/``time`` are per-event mixing inputs; ``None`` fills the
        legacy defaults (0.5 / 0.0).  ``chunk`` bounds the tree-fill
        memory (rows per pass; ``None`` = library default).
        """
        from tabpwa.pwa_build import build_tree_event_data

        n = momenta.shape[0]
        if self._legacy_td_structure():
            from tabpwa.momenta_to_data import momenta_to_data

            w = np.ones(n) if weight is None else np.asarray(weight,
                                                            dtype=float)
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

        # tree-general branch: same walk the PWA path uses
        chains_by_topo = {}
        for _, chain in self.full_decay.get_partial_waves():
            chains_by_topo[self.topo_index[chain.topo_id()]] = chain
        spinful = [nm for nm in self.finals
                   if float(self.dic["particle"][nm].get("J", 0)) != 0]
        out = build_tree_event_data(self, kernel_config, chains_by_topo,
                                    momenta, spinful_names=spinful,
                                    chunk=chunk)
        out["weight"] = (np.ones(n) if weight is None
                         else np.asarray(weight, dtype=float))
        if bkg is not None:
            out["bkg"] = np.asarray(bkg, dtype=np.float64).ravel()
        out["frac"] = (np.full(n, 0.5) if frac is None
                       else np.asarray(frac, dtype=float).ravel())
        out["time"] = (np.zeros(n) if time is None
                       else np.asarray(time, dtype=float).ravel())
        return out

    def _legacy_td_structure(self):
        """True for the 4π structure momenta_to_data implements: four
        pion finals in two identical-particle pairs."""
        tree = self.decay_tree
        if len(self.finals) != 4 or len(tree.identical_groups) != 2:
            return False
        if any(len(g) != 2 for g in tree.identical_groups):
            return False
        from tabpwa.phasespace_b4pi import M_PION
        return all(
            abs(float(self.dic["particle"][nm]["mass"]) - M_PION) <= 1e-3
            for nm in self.finals)
