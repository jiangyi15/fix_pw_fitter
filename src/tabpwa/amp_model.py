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

import itertools

import numpy as np

from tabpwa.base_model import BaseModel, row_block_factors
from tabpwa.pwa_gen_flat_params import FlatKernelParams
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
        see the returned object.  A transform that also needs the built
        kernel config (the flat ``pwa_gen`` one) sets
        ``params_transform_wants_config = True``.
        """
        if getattr(self, "params_transform_wants_config", False):
            return self.params_transform_cls(self, self.build_kernel_config())
        return self.params_transform_cls(self)

    def particle_model(self, name):
        """The particle-model instance called *name* (a BW factor of a chain).

        Owned here because this class owns the decay tree; the ``Fitter`` is a
        composition root and must not walk chains itself.
        """
        for chain in self.full_decay.chains:
            for decay in chain.decays[1:]:
                if decay.core.name == name:
                    return decay.core._model
        raise ValueError(f"Particle '{name}' not found in config")

    def particle_defaults(self):
        """``{param: default}`` from every particle model, deduplicated.

        Lives here, not in the ``Fitter``: the fitter must not walk the decay
        tree or know that a particle model has ``get_defaults()``.
        """
        out, seen = {}, set()
        for chain in self.full_decay.chains:
            for decay in chain.decays[1:]:
                mdl = decay.core._model
                if id(mdl) in seen:
                    continue
                seen.add(id(mdl))
                for k, v in mdl.get_defaults().items():
                    out[k] = float(v)
        return out

    def _particle_mw_transforms(self):
        """Each particle model's own mass/width transform — the tree walk.

        The FITTER must not walk the decay tree or know a particle model's
        interface, so the amplitude model owns that knowledge and there is
        exactly ONE copy of this walk (a second one drifts).

        The two particle interfaces are SEPARATE: only a ``BaseModel`` has a
        mass/width stage, and ``BasisModel`` deliberately does not declare one
        (a basis-only model such as ``kmatrix2`` has no concept of it).  So
        the walk — which crosses both kinds — asks which interface the model
        implements and skips the basis-only ones.  Deduplicated by instance:
        one particle can appear in several chains.
        """
        from tabpwa.particle_model.base import BaseModel as ParticleBaseModel
        out, seen = [], set()
        for chain in self.full_decay.chains:
            for decay in chain.decays[1:]:
                mdl = decay.core._model
                if id(mdl) in seen:
                    continue
                seen.add(id(mdl))
                if not isinstance(mdl, ParticleBaseModel):
                    continue
                tr = mdl.make_mass_width_transform()
                if tr is not None:
                    out.append(tr)
        return out

    def particle_transforms(self):
        """The particle models' own transforms — the ``pwa`` family's list.

        The FITTER must not walk the decay tree or know a particle model's
        interface, so the amplitude model owns that knowledge and this is the
        ONE copy of the walk.  The ``pwa`` kernel reads the couplings these
        transforms produce (its ``g0`` block IS those outputs), so they are the
        whole list.

        A model that needs a different list writes its own: `PWAGen` overrides
        this method outright rather than asking the base for a switch (see
        "Class policy" in AGENTS.md).
        """
        return self._particle_mw_transforms()

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



@register_amplitude_model("pwa_gen", "pwa_basis")
class PWAGen(PWA):
    """Projection-sum PWA on the particle ``BasisModel`` interface.

    For every decay chain whose resonance carries a :class:`BasisModel`
    basis the amplitude is

    .. math::

        A_w(e) = ck_w \\cdot
                 \\frac{fa(e)\\,fl(e)\\,N_{i(w)}(m_e)}
                      {\\sum_j g_j\\,G_j(m_e)}

    expressed in the existing kernel vocabulary:

    * **denominator** — the opaque basis sum ``BW = Σ_j coeff_j·F_j``.  The
      built-in rows stay ``[1, −m², −i·γ_1, …]`` (``m0``/``g0`` coefficients,
      bit-identical to ``pwa``); extra basis rows of the model
      (``m⁴``, ``−i·ρ``, ``+i·s·ρ`` …) are tabulated in ``den_table`` and
      turned on with an explicit ``coeff`` vector
      (:class:`~tabpwa.kernel_params.PWAGenKernelParams`).
    * **numerator** — the model's free numerator rows become extra waves
      sharing the chain's denominator, angular column and form factor, with
      ``ck_w = g_ls[chain] · n_i`` (the ``ck`` outer product).  The
      per-event factor is the tabulated ``N_i(m_e)`` (``n_table`` /
      ``n_index``); row 0 is the constant ``1``.

    Nothing is emitted when a model has no basis extensions, so a
    ``pwa_gen`` config whose resonances use the *default* ``BasisModel``
    basis produces exactly the ``pwa`` kernel config and numbers.
    """

    name = "pwa_gen"
    # ``FlatKernelParams`` needs the built config (it reads basis_table /
    # den_names to shape the flat coeff vector)
    params_transform_cls = FlatKernelParams
    params_transform_wants_config = True

    def __init__(self, config):
        super().__init__(config)
        self._pg_kc = None            # cached kernel config
        self._pg_combs = None         # expanded ck combinations (pre-block)
        self._pg_den_names = []       # per base BW: denominator weight names
        self._pg_bws = []             # per base BW: {name, model, transform, …}

    # ── kernel config ────────────────────────────────────────────────
    def build_kernel_config(self):
        """The flat ``pwa_gen`` kernel config.

        Built from the model's own description (:meth:`_pg_host_records`) plus
        :func:`tabpwa.pwa_gen_config.build_from_groups` — the ``pwa`` base
        config is **not** consulted: its arrays belong to the old kernel
        vocabulary (``bw_order``/``fl_order``/``den_row``/``matrix_angle`` …),
        none of which a flat-config backend reads.

        Only the host-side metadata the fitter needs is carried over: the ck
        map, the block counts and the parameter-name lists.
        """
        if self._pg_kc is not None:
            return self._pg_kc
        with self._default_context():
            # `build_single_index` is the PLAIN-PWA index builder: it emits
            # `g0_index` / `g0_mass_index` / `gamma_table` for a kernel whose
            # denominator is `s0 - m^2 - i*mult*gamma(m)`.  This path has no
            # gamma table -- it reads each resonance's BASIS rows -- so it does
            # not call it.  `_pg_host_records` derives its BW list straight
            # from the decay tree.
            exp, mult, C, P, n_base = self._pg_host_records(
                n_blocks=self.decay_tree.n_perm * self.decay_tree.n_cp)
            from tabpwa.pwa_gen_config import build_from_groups
            kc = build_from_groups(self)
        # ── host metadata (names / counts, no kernel arrays) ────────
        kc["n_blocks"] = int(C)
        kc["n_perm"] = self.decay_tree.n_perm
        kc["n_cp"] = self.decay_tree.n_cp
        kc["n_identical"] = self.decay_tree.n_perm
        kc["n_proj"] = int(P)
        kc["n_proj_base"] = int(P)
        kc["top_states"] = [int(x) for x in self._helicity_top_states()] or None
        # ``ck_map`` is deliberately NOT emitted: it is derived from the same
        # expansion and ``get_ck_map()`` computes it on demand.  Including it
        # here would recurse (``get_ck_map`` -> ``build_kernel_config``).
        # No ``m0_names`` / ``g0_names`` here: they are the PLAIN-PWA width
        # index, read by the pwa kernel path and its tests.  A basis path has
        # no gamma table, and a basis-only model (``kmatrix2``) has no running
        # width at all — asking it for ``get_gamma_name()`` was a category
        # error (and nothing on this path reads the key).  ``kernel_params``
        # takes the m0/g0 names from ``model.g0_phys_name``, which is empty
        # for a pure basis config.
        try:
            from tabpwa.helicity_angle import (decay_chain_to_tree,
                                               tree_vertices, canonical_variables,
                                               to_spin)
            first = self.full_decay.get_partial_waves()[0][1]
            nv = len(tree_vertices(decay_chain_to_tree(first)))
            topj0 = to_spin(first.decays[0].core.J) == 0
            kc["variables"] = canonical_variables(nv, top_j0=topj0)
            kc["top_j0"] = bool(topj0)
        except Exception:
            pass
        self._pg_kc = kc
        return kc

    def particle_transforms(self):
        """pwa_gen's OWN list: basis transforms and constants — no particle ones.

        This model fits basis weights, not ``m0``/``g0``.  Each resonance's
        basis transform is SELF-CONTAINED (it holds the model stage that
        derives its couplings), so nothing here consumes the output of
        ``make_mass_width_transform()``.  Registering one anyway would run it
        on every ``resolve``/``chain_gradient`` for values no kernel receives,
        and would strike the names it produces off the fitted surface
        (``OneModel``'s pinned mass/width is the measured case).

        Written out rather than inherited behind a switch: the two families
        answer a different question, and a flag would hide that in a branch.
        """
        out = list(self.extra_transforms())
        fixed = self.constant_weight_values()
        if fixed:
            from tabpwa.param_constraint import FixedValueTransform
            out.append(FixedValueTransform(fixed))
        return out

    def extra_transforms(self):
        """Each resonance's OWN basis transform — no combined wrapper.

        The constraint manager already unions a list of transforms: it appends
        every ``input_names`` to the parameter surface, removes every
        ``output_names`` (produced, therefore not fitted), chains them in
        registration order and de-duplicates identical ones.  A combined
        wrapper would have to re-derive all of that by hand — and did: it
        walked the tree for the numerator models and re-listed
        ``num_slot_names()``, both of which the per-model transform already
        declares.

        A model with NO transform is legitimate and contributes nothing: its
        basis weights ARE its parameters (``BasisModel.make_basis_transform``
        returns ``None`` for exactly that reason), and the constraint layer
        keeps them because no transform claims to produce them.
        """
        self.build_kernel_config()
        out = []
        for rec in self._pg_bws:
            tr = rec["transform"]
            # A transform that declares neither inputs nor outputs cannot
            # change anything (``OneModel``: a single constant basis row, so
            # the default transform has no weights to produce).  Registering
            # it would only add a no-op to every resolve/backward pass.
            if tr is not None and (tr.output_names or tr.input_names):
                out.append(tr)
        return out


    def constant_weight_values(self):
        """``"one"`` — the m² coefficient, +1 because the row is ``-m²``.

        A shared constant: every resonance's m² term carries exactly 1, so
        there is one name for the whole fit, not one per model.  Emitted only
        when the basis actually uses it.
        """
        den_names = (self._pg_kc or {}).get("den_names") or []
        return {"one": 1.0} if "one" in den_names else {}

    # ── ck map (chain → LS waves → numerator index) ──────────────────
    def get_ck_map(self):
        """Block-expanded ck combinations, numerator heads included.

        Each base chain contributes one ck slot per numerator row:
        ``ck_w = g_ls[chain] · n_i`` (the plain PWA tuple for a row whose
        weight is the constant ``1``, plus the numerator head names).
        """
        self.build_kernel_config()
        combs = self._pg_combs
        n_perm, n_cp, _C = row_block_factors(self.dic)
        ret = []
        for cp in range(n_cp):
            for _ in range(n_perm):
                for j in combs:
                    if cp == 0:
                        ret.append(j)
                    else:
                        ret.append((j[0], j[1].replace("g_ls", "g_lsbar"),
                                    *j[2:]))
        return ret

    def get_partial_waves_params(self):
        """Base (pre-block) ck combinations, numerator heads included."""
        self.build_kernel_config()
        return list(self._pg_combs)

    # ── layout construction ──────────────────────────────────────────
    def _pg_particle_models(self):
        """``{particle name: particle-model instance}`` from the tree."""
        out = {}
        for chain in self.full_decay.chains:
            for decay in chain.decays:
                out.setdefault(decay.core.name, decay.core._model)
        return out

    def _pg_grid(self):
        """Mass grid shared with ``build_gamma_table`` (for ``den_table``)."""
        return self.gamma_grid()

    def _pg_chain_factors(self, chain):
        """Numerator factors of a chain, **one entry per BW factor**.

        Returns ``[{slot, heads, rows, mass_col}]`` in ``bw_order`` order
        (``slot`` = the chain's factor position ``r``): a chain
        ``B -> R1 x, R1 -> R2 y, R2 -> z w`` carries a numerator on both
        ``R1`` and ``R2``, each evaluated at **its own** mass, so the factors
        cannot be collapsed into one row with one mass column.

        ``heads``/``rows`` are the free basis weights and tabulated rows of
        that factor; a factor without a free numerator basis contributes
        ``heads=()`` and a single constant row.
        """
        from tabpwa.particle_model.base import BasisModel

        # factor order along the chain == bw_order order (decays[1:])
        out = []
        for slot, decay in enumerate(chain.decays[1:]):
            mdl = decay.core._model
            heads, rows = [], [None]
            if isinstance(mdl, BasisModel):
                heads = list(mdl.get_num_name())
                if heads:
                    if mdl.make_basis_transform() is None:
                        raise ValueError(
                            f"pwa_gen: particle model '{mdl.name}' declares "
                            f"numerator weights {heads} but no basis "
                            f"transform — the weights would never be resolved")
                    n_rows = len(mdl.num_basis(np.zeros(1)))
                    if n_rows != len(heads):
                        raise ValueError(
                            f"pwa_gen: '{mdl.name}' has {len(heads)} numerator "
                            f"weight names but {n_rows} num_basis rows")
                    rows = [lambda m, mdl=mdl, i=i: np.asarray(
                        mdl.num_basis(m)[i], dtype=complex)
                        for i in range(len(heads))]
            out.append({"slot": slot, "name": decay.core.name,
                        "heads": heads, "rows": rows})
        return out

    def _pg_chain_rows(self, chain):
        """Numerator-row combinations of a chain (count = its wave expansion).

        Compatibility view over :meth:`_pg_chain_factors`: the kernel lays the
        numerator out per (wave, factor) like ``bw_order``, but a chain still
        expands into the product of its factors' free rows, and that product
        is what this returns (one entry per combination).
        """
        import itertools

        facs = [f for f in self._pg_chain_factors(chain)
                if f["heads"] and f["rows"]]
        if not facs:
            return []
        out = []
        for combo in itertools.product(*[[rf for rf in f["rows"]] for f in facs]):
            heads = tuple(h for f, rf in zip(facs, combo)
                          for h in [f["heads"][f["rows"].index(rf)]])
            out.append((heads, combo))
        return out

    def _pg_factor_mass_column(self, chain, slot):
        """Mass column of one BW factor of *chain* (its own resonance).

        ``slot`` indexes the factor in ``bw_order`` order (``decays[1:]``).
        Unlike a chain-wide column, this lets two numerator-bearing
        resonances on one chain read two different masses.
        """
        decay = chain.decays[1 + slot]
        name = decay.core.name
        m_name = f"{name}_mass"
        for b, key in enumerate(self.unique_bw):
            if key[0] == m_name:
                return int(self._pg_base_mass_index[b])
        # no BW slot for this factor (a constant/contact term): its own mass
        # column is still the right argument, taken from the base table
        raise ValueError(
            f"pwa_gen: numerator factor '{name}' has no Breit-Wigner slot")

    def _pg_den_basis(self, name, model, grid):
        """``(weight names, finished rows G_j)`` of a resonance denominator.

        The ROWS come from the model's own ``den_basis`` and the NAMES from
        its own ``get_den_name()`` — one rule, independent of whether a
        transform exists:

        * a transform declares the names as OUTPUTS, so the constraint layer
          resolves them from the physical parameters instead of fitting them;
        * with no transform (``make_basis_transform() -> None``) the names ARE
          the parameters: nothing claims to produce them, so the constraint
          layer keeps them.  That is a legitimate model, not a gap — which is
          why nothing here asks the model for a transform, and why a
          basis-only model need not implement ``get_gamma_name()``.
        """
        # EVERY model is a basis model (``BaseModel`` owns
        # ``num_basis``/``den_basis``), so there is exactly one place that
        # decides the rows — the model itself.  Duplicating the default here
        # is what previously let the builder's ``-m²`` drift out of sync with
        # ``den_weights``' ``-1`` and silently flip every m² term.
        rows = [np.asarray(r, dtype=complex) for r in model.den_basis(grid)]
        names = list(model.get_den_name())
        if len(names) != len(rows):
            raise ValueError(
                f"pwa_gen: '{name}' declares {len(names)} denominator weight "
                f"names but {len(rows)} basis rows")
        # Row 0 is the constant 1 and row 1 is ``-m²``.  The SIGN is part of
        # the row, so the shared ``"one"`` weight is exactly +1 — this check is
        # what keeps a model from flipping the convention silently.
        if not np.allclose(rows[0], 1.0, rtol=0.0, atol=1e-14):
            raise ValueError(
                f"pwa_gen: '{name}' denominator basis row 0 must be 1")
        # Row 1's SIGN is NOT checked: it is whatever the model's
        # ``den_basis`` returns, and the weight that multiplies it comes from
        # the same model, so the pair is self-consistent by construction.
        # (Pinning it here would break a model that legitimately emits ``+m²``
        # with a negated weight — `kmatrix2` does exactly that.)
        return names, rows

    def _pg_host_records(self, n_blocks=1):
        """Resolve the model's own description — **no kernel arrays**.

        A ``pwa_gen`` model needs, independently of which arrays a kernel
        reads:

        * ``_pg_den_names`` / ``_pg_bws`` — the per-resonance denominator
          basis: its weight names and the rows/transform they come from.
          Each resonance's OWN basis transform publishes its weight names as
          outputs (``extra_transforms`` hands exactly those to the constraint
          manager), so the pipeline resolves them without a combined wrapper.
        * ``_pg_combs`` — the expanded ck combinations (``ck_w = g_ls·n_i``),
          which ``get_ck_map`` turns into the parameter slots.

        This is the whole of the kernel-config-independent state; the arrays
        a kernel consumes are built by
        :func:`tabpwa.pwa_gen_config.build_from_groups` from these records.
        """
        from tabpwa.particle_model.base import BasisModel

        models = self._pg_particle_models()
        # the resonance of every BW factor, chain order, first occurrence —
        # derived from the TREE, not from `unique_bw`
        bw_names = []
        for ch in self.full_decay.chains:
            for dec in ch.decays[1:]:
                if dec.core.name not in bw_names:
                    bw_names.append(dec.core.name)
        waves = list(self.full_decay.get_partial_waves())
        n_base = len(waves)
        C = int(n_blocks or 1)
        P = len(self._helicity_external_states())
        grid, _n_min, _n_delta = self._pg_grid()

        # ── denominator basis per (base) unique BW ───────────────────
        #
        # The list is the TREE's BW factors (``decays[1:]``), in chain order —
        # NOT ``unique_bw``.  ``unique_bw`` is a plain-pwa index keyed on
        # ``(m_name, mass_column)``: it exists so the pwa kernel can scatter
        # ``m0``/``g0`` onto its BW slots, and it carries a mass column this
        # path discards.  Walking the tree directly is the same list (verified
        # identical in order and content on the real config) without borrowing
        # an index structure that a basis-only model cannot even populate —
        # that is why ``build_single_index`` had to tolerate an empty gamma
        # list just for this traversal.
        den_names, bws = [], []
        for name in bw_names:
            mdl = models[name]
            # ``make_basis_transform`` is declared ON THE INTERFACE (default
            # ``None`` = "the weights are the parameters"), so every model can
            # answer it — no probe, no "does it have one" branch.
            tr = mdl.make_basis_transform()
            names, rows = self._pg_den_basis(name, mdl, grid)
            den_names.append(names)
            bws.append({"name": name, "model": mdl, "transform": tr,
                        "den_names": names, "rows": rows})
        self._pg_den_names = den_names
        self._pg_bws = bws

        # ── expanded ck combinations ────────────────────────────────
        # One ck slot per (chain, numerator-row combination); the heads of a
        # chain's numerator rows append to its plain ``g_ls`` tuple.
        chain_fac = {}
        per_wave = []
        for ls, chain in waves:
            cid = id(chain)
            if cid not in chain_fac:
                chain_fac[cid] = self._pg_chain_factors(chain)
            per_wave.append(chain_fac[cid])
        base_combs = list(self.full_decay.get_partial_waves_params())

        mult = np.ones(n_base, dtype=np.int32)
        exp = []                       # (base wave, combo|None)
        comb_exp = []                  # the ck tuple of each expanded wave
        for k, facs in enumerate(per_wave):
            free = [f for f in facs if f["heads"] and f["rows"]]
            if not free:
                mult[k] = 1
                exp.append((k, None))
                comb_exp.append(tuple(base_combs[k]))
                continue
            combos = list(itertools.product(*[
                [(f, rf) for rf in f["rows"]] for f in free]))
            mult[k] = len(combos)
            for combo in combos:
                heads = tuple(h for (f, rf) in combo
                              for h in [f["heads"][f["rows"].index(rf)]])
                exp.append((k, combo))
                comb_exp.append(tuple(base_combs[k]) + heads)
        self._pg_combs = comb_exp
        self._pg_mult = mult
        return exp, mult, C, P, n_base

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
