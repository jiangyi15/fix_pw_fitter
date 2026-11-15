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

from tabpwa.base_model import BaseModel, row_block_factors
from tabpwa.kernel_params import (
    BuildKernelParams, PWAKernelParams, FlavourTagMixKernelParams,
    PolarisedKernelParams, PWAGenKernelParams)

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

    def extra_transforms(self):
        """Model-level transforms for the constraint pipeline.

        Registered by :meth:`tabpwa.fitter.Fitter.setup_mass_width_transforms`
        right after the particle models' mass/width transforms (same
        contract: physical parameters → derived slots).  The default is
        empty; ``pwa_gen`` adds its physical → basis-weight transform.
        """
        return []

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
    params_transform_cls = PWAGenKernelParams

    def __init__(self, config):
        super().__init__(config)
        self._pg_kc = None            # cached kernel config
        self._pg_combs = None         # expanded ck combinations (pre-block)
        self._pg_generic = False      # explicit coefficient basis in use
        self._pg_den_names = []       # per base BW: denominator weight names
        self._pg_bws = []             # per base BW: {name, model, transform, …}
        self._pg_coef_transform = None   # combined physical → weights transform

    # ── kernel config ────────────────────────────────────────────────
    def build_kernel_config(self):
        """Base PWA config plus the (optional) basis extensions."""
        if self._pg_kc is not None:
            return self._pg_kc
        with self._default_context():
            kc = super().build_kernel_config()
            self._pg_attach(kc)
        self._pg_kc = kc
        return kc

    def extra_transforms(self):
        """The physical → basis-weight transform (empty for the plain path).

        Registered into the constraint pipeline by
        :meth:`tabpwa.fitter.Fitter.setup_mass_width_transforms`, so the
        basis weights are resolved from the physical parameters (and their
        gradients chained back) exactly like a mass/width transform.
        """
        self.build_kernel_config()
        if not self._pg_generic:
            return []
        return [self._coef_transform()]

    def _coef_transform(self):
        """(cached) combined particle-model → basis-weight transform."""
        if self._pg_coef_transform is None:
            from tabpwa.pwa_gen_transform import PWAgenWeightsTransform
            self._pg_coef_transform = PWAgenWeightsTransform(self)
        return self._pg_coef_transform

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

    def _pg_den_basis(self, name, model, grid, tr):
        """``(weight names, finished rows G_j)`` of a resonance denominator.

        ``weight names`` are the model's own ``get_den_name()`` entries when
        it carries a basis transform (its ``output_names``, reused as the
        coefficient names); otherwise they are internal names — the plain
        preset's ``[s0, poly, m0·g0]`` semantics read back positionally, and
        ``{name}_width`` would collide with the physical width input.
        """
        from tabpwa.particle_model.base import BasisModel

        if isinstance(model, BasisModel):
            rows = [np.asarray(r, dtype=complex) for r in model.den_basis(grid)]
        else:
            gamma_names = list(model.get_gamma_name())
            g = model.gamma(grid)
            rows = ([np.ones_like(grid, dtype=complex),
                     np.asarray(grid, dtype=float) ** 2 + 0j]
                    + [-1j * np.asarray(x, dtype=complex) for x in g])
        if tr is not None:
            names = list(model.get_den_name())
        else:
            n_plain = 2 + len(model.get_gamma_name())
            if len(rows) != n_plain:
                raise ValueError(
                    f"pwa_gen: '{name}' has {len(rows)} denominator basis "
                    f"rows but no basis transform (the plain preset has "
                    f"{n_plain}) — an extended denominator basis needs "
                    f"make_basis_transform()")
            names = [f"pwa_gen::{name}::den{j}" for j in range(len(rows))]
        if len(names) != len(rows):
            raise ValueError(
                f"pwa_gen: '{name}' declares {len(names)} denominator weight "
                f"names but {len(rows)} basis rows")
        # the kernel's two built-in rows are 1 and m² (see the den_row encoding)
        if not np.allclose(rows[0], 1.0, rtol=0.0, atol=1e-14):
            raise ValueError(
                f"pwa_gen: '{name}' denominator basis row 0 must be 1")
        if not np.allclose(rows[1], np.asarray(grid, float) ** 2,
                           rtol=1e-12, atol=1e-14):
            raise ValueError(
                f"pwa_gen: '{name}' denominator basis row 1 must be m²")
        return names, rows

    def _pg_attach(self, kc):
        """Expand the wave axis and attach ``n_*`` / ``den_*`` tables."""
        from tabpwa.particle_model.base import BasisModel

        models = self._pg_particle_models()
        waves = list(self.full_decay.get_partial_waves())
        n_base = len(waves)
        C = int(kc.get("n_blocks", 1) or 1)
        P = int(kc.get("n_proj", 1) or 1)
        n_wave = kc["matrix_angle"].shape[1]
        if n_wave != C * P * n_base:
            raise ValueError(
                f"pwa_gen: n_wave={n_wave} != n_blocks·n_proj·n_wave_base = "
                f"{C}·{P}·{n_base}")
        n_res = int(kc["bw_order"].size // n_wave)
        n_decay = int(kc["fl_order"].size // n_wave)
        base_combs = list(self.full_decay.get_partial_waves_params())
        self._pg_base_mass_index = [int(x) for x in kc["mass_index"]]

        grid, n_min, n_delta = self._pg_grid()

        # ── denominator basis per (base) unique BW ───────────────────
        den_names = []
        bws = []
        extra_rows = []           # tabulated finished basis functions
        extra_pos = {}
        any_transform = False
        for b, key in enumerate(self.unique_bw):
            name = key[0][:-len("_mass")]
            mdl = models[name]
            mk = getattr(mdl, "make_basis_transform", None)
            tr = mk() if mk is not None else None
            if tr is not None:
                any_transform = True
            names, rows = self._pg_den_basis(name, mdl, grid, tr)
            den_names.append(names)
            bws.append({"name": name, "model": mdl, "transform": tr,
                        "den_names": names, "rows": rows})

        # Every row past the two built-ins (1, m²) is tabulated as a
        # finished basis function with an explicit weight.  This must run
        # for ALL Breit–Wigner slots — including plain models — whenever
        # the generic coefficient basis is needed: a config that mixes a
        # BasisModel with an ordinary resonance then has a ``den_row`` for
        # every slot instead of a KeyError.  Plain weights are
        # ``[m0², −1, m0·g_l]`` (see PWAgenWeightsTransform.forward).
        if any_transform:
            for rec in bws:
                name = rec["name"]
                for j in range(2, len(rec["rows"])):
                    pos = extra_pos.get((name, j))
                    if pos is None:
                        pos = len(extra_rows)
                        extra_pos[(name, j)] = pos
                        extra_rows.append(rec["rows"][j])

        n_bw = len(self.unique_bw)
        self._pg_den_names = den_names
        self._pg_bws = bws
        self._pg_generic = bool(any_transform)

        # ── numerator rows per base wave + expanded wave list ────────
        # ── numerator rows per (base wave, FACTOR) ──────────────────
        # A chain may carry a numerator on more than one BW factor
        # (``B -> R1 x, R1 -> R2 y``), each at its own mass, so the layout
        # mirrors ``bw_order``: one entry per (wave, factor), not per wave.
        chain_fac = {}
        per_wave = []
        for ls, chain in waves:
            cid = id(chain)
            if cid not in chain_fac:
                chain_fac[cid] = self._pg_chain_factors(chain)
            per_wave.append(chain_fac[cid])

        n_table = [np.ones_like(grid, dtype=complex)]
        n_row_cache = {}

        def _row_of(fac, rowfun):
            """Table row index of one factor's basis row (0 = constant)."""
            key = (id(fac), id(rowfun))
            if key in n_row_cache:
                return n_row_cache[key]
            g = None if rowfun is None else rowfun(grid)
            if g is None or np.all(g == 1):
                n_row_cache[key] = 0
                return 0
            n_row_cache[key] = len(n_table)
            n_table.append(np.asarray(g, dtype=complex))
            return n_row_cache[key]

        # the expansion of a wave is the product of its factors' free rows
        mult = np.ones(n_base, dtype=np.int32)
        exp = []                  # (base wave, [rowfun per factor])
        comb_exp = []
        for k, (ls, chain) in enumerate(waves):
            facs = per_wave[k]
            free = [(f, rf) for f in facs if f["heads"] for rf in f["rows"]]
            if not free:
                mult[k] = 1
                exp.append((k, None))
                comb_exp.append(tuple(base_combs[k]))
                continue
            import itertools
            combos = list(itertools.product(*[
                [(f, rf) for rf in f["rows"]]
                for f in facs if f["heads"] and f["rows"]]))
            mult[k] = len(combos)
            for combo in combos:
                heads = tuple(h for (f, _rf) in combo
                              for h in
                              [f["heads"][f["rows"].index(_rf)]])
                exp.append((k, combo))
                comb_exp.append(tuple(base_combs[k]) + heads)

        n_exp = len(exp)
        has_num = any(c is not None for _k, c in exp)
        if not has_num and not self._pg_generic:
            # plain path: nothing to add — the config is exactly pwa's
            self._pg_combs = [tuple(c) for c in base_combs]
            return

        # ── expand the per-wave arrays (p-major, block-major) ────────
        strip_mass = self.n_topo * self.n_res
        exp_k = np.array([k for k, _c in exp], dtype=np.int64)
        old_idx = np.concatenate([
            b * P * n_base + p * n_base + exp_k
            for b in range(C) for p in range(P)])
        kc["matrix_angle"] = kc["matrix_angle"][:, old_idx]
        bw_rows = np.asarray(kc["bw_order"]).reshape(-1, n_res)
        kc["bw_order"] = bw_rows[old_idx].reshape(-1)
        fl_rows = np.asarray(kc["fl_order"]).reshape(-1, n_decay)
        kc["fl_order"] = fl_rows[old_idx].reshape(-1)
        if "wave_names" in kc:
            names = []
            for k, _ls in enumerate(waves):
                for _ in range(int(mult[k])):
                    names.append(str(kc["wave_names"][k]))
            kc["wave_names"] = names

        # ── numerator table / per-factor index ──────────────────────
        # ``n_index[f]`` with ``f = w·n_res + r``, exactly like ``bw_order``:
        # factor ``r`` reads the mass column of its OWN BW, so
        # ``N_R1(m_R1) · N_R2(m_R2)`` is expressible.  ``n_index`` only: the
        # mass column of a factor is already ``mass_index[bw_order[f]]`` —
        # the kernel derives it exactly as it does for the denominator, so no
        # second column array is needed (and none can fall out of sync).
        n_wave_e = n_exp
        n_idx = np.zeros((n_wave_e, n_res), dtype=np.int32)
        for i, (k, combo) in enumerate(exp):
            for r, fac in enumerate(per_wave[k]):
                if not fac["heads"]:
                    continue
                chosen = None
                if combo is not None:
                    for (f, rf) in combo:
                        if f is fac:
                            chosen = rf
                            break
                n_idx[i, r] = _row_of(fac, chosen)

        kc["n_table"] = np.stack(n_table)
        kc["n_min"] = n_min
        kc["n_delta"] = n_delta
        # ``n_idx`` is already in ``exp`` order, and ``matrix_angle``'s
        # columns are exactly ``old_idx`` — which selects ``exp`` indices
        # ``p·n_base + k``, i.e. it walks ``exp`` in order once per
        # projection.  So the numerator block is the ``exp``-order rows
        # tiled over (block, projection) — the same shape as the wave axis,
        # factor-minor like ``bw_order``.
        kc["n_index"] = np.ascontiguousarray(
            np.tile(n_idx.reshape(-1), C * P), dtype=np.int32)

        # ── generic coefficient basis ───────────────────────────────
        if self._pg_generic:
            n_slots = max((len(nm) for nm in den_names), default=2)
            base_den = np.full((n_bw, n_slots), -1, dtype=np.int32)
            for b, names in enumerate(den_names):
                name = self.unique_bw[b][0][:-len("_mass")]
                for j in range(len(names)):
                    if j == 0:
                        base_den[b, j] = 0
                    elif j == 1:
                        base_den[b, j] = 1
                    else:
                        pos = extra_pos.get((name, j))
                        if pos is None:            # tail of a mixed config
                            raise ValueError(
                                f"pwa_gen: no tabulated denominator basis "
                                f"row for '{name}' slot {j} — the resonance "
                                f"cannot join a generic coefficient config")
                        base_den[b, j] = 2 + pos
            kc["den_row"] = np.tile(base_den, (C, 1))
            kc["den_table"] = (np.stack(extra_rows) if extra_rows
                               else np.zeros((0, grid.size), dtype=complex))
            kc["den_min"] = n_min
            kc["den_delta"] = n_delta

        self._pg_combs = comb_exp
        kc["ck_map"] = list(comb_exp)


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
