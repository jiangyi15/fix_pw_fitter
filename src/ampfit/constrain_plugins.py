"""
Constraint plugin system for Fitter.

Reads the ``constrains`` section from the YAML config and dispatches
each entry to a registered handler.  Handlers have full access to the
Fitter (including ``fitter.config``) and can inspect the particle model
structure to automatically build constraints.

Usage::

    @register_constrain("my_constraint")
    def handle_my(fitter, spec):
        \"\"\"spec is the parsed YAML value under ``my_constraint:``.\"\"\"
        ...

Then in ``config.yml``::

    constrains:
        my_constraint:
            param1: value1
"""

# ═══════════════════════════════════════════════════════════════════
# Registry + dispatch
# ═══════════════════════════════════════════════════════════════════

_constrain_handlers = {}  # name → (fn, order)


def register_constrain(name, order=100):
    """Decorator: register a handler for a YAML ``constrains`` section key.

    Args:
        name: key in the YAML ``constrains`` section.
        order: execution priority (lower = earlier).  Default 100.
    """
    def _f(fn):
        _constrain_handlers[name] = (fn, order)
        return fn
    return _f


def apply_constrains(fitter, constrains=None):
    """Apply constraint handlers for each key in *constrains*.

    Handlers are sorted by ``order`` (lower first) before execution.
    """
    if constrains is None:
        constrains = fitter.config.dic.get("constrains", {})
    # Sort by order, then execute
    items = [(k, v) for k, v in constrains.items() if k in _constrain_handlers]
    items.sort(key=lambda kv: _constrain_handlers[kv[0]][1])
    for key, spec in items:
        _constrain_handlers[key][0](fitter, spec)


# ═══════════════════════════════════════════════════════════════════
# Structural constraint handlers
# ═══════════════════════════════════════════════════════════════════

@register_constrain("ck_redundancy", order=0)
def _handle_ck_redundancy(fitter, spec):
    """Fix CK scale redundancies by prefix context.

    For each CK term ``[t0, t1, t2, ...]``, process positions in order.
    At each position p > 0, if the prefix ``(t0, ..., t_{p-1})`` has been
    seen before at this position, the term at ``p`` is a ratio parameter
    (free).  Otherwise, fix the term as the reference for this context.

    Concept: for CK products like ``(a+b)(c+d) -> [a,c], [a,d], [b,c]``,
    the first occurrence at each prefix context is fixed as reference;
    subsequent terms at the same position with the same prefix are ratios.

    Also fixes the first ``_total_0`` as overall amplitude reference.
    """
    seen_prefix = set()  # {(pos, prefix_tuple), ...}
    fixed = {}
    for comb in fitter.all_comb:
        for pos, term in enumerate(comb):
            if not isinstance(term, str) or pos == 0:
                continue
            prefix = tuple(comb[:pos])
            key = (pos, prefix)
            if key in seen_prefix:
                continue  # ratio term — leave free
            seen_prefix.add(key)
            r = term + "r"
            if r not in fixed:
                fixed[r] = 1.0
                fixed[term + "i"] = 0.0

    # Also fix the first _total_0 as overall reference
    for comb in fitter.all_comb:
        for term in comb:
            if isinstance(term, str) and "_total_0" in term:
                fixed.setdefault(term + "r", 1.0)
                fixed.setdefault(term + "i", 0.0)
                break
        if any("_total_0" in t for t in comb if isinstance(t, str)):
            break

    if fixed:
        fitter.set_fixed(fixed, reset=False)


@register_constrain("cp_symmetry", order=1)
def _handle_cp_symmetry(fitter, spec):
    """Constrain CP-conjugate pairs using chain structure.

    Collects decays by first intermediate resonance name, then for each
    CP conjugate pair (p/m suffix):
    - Same-groups ``_total_0`` within all R+ and within all R-
    - Same-groups ``_g_ls`` between R+ and R- counterparts
    - Scales by -1 if the decay's outgoing particle has odd J
    - Stores free list for the ``free_var`` handler
    """
    chains = fitter.config.full_decay.chains
    if not chains:
        return

    particle_decays = {}
    particle_chain = {}
    for chain in chains:
        name = chain.decays[1].core.name
        particle_decays.setdefault(name, []).append(chain.decays[1])
        particle_chain.setdefault(name, []).append(chain)

    same_list = []
    free_list = []
    scale_list = {}
    for name, v in particle_chain.items():
        if name.endswith(("p", "m")):
            totals = [str(i).replace("+", ".") + "_total_0" for i in v]
            same_list.append([i + "r" for i in totals])
            same_list.append([i + "i" for i in totals])

    for pname, v1 in particle_decays.items():
        if not pname.endswith("p"):
            continue
        mname = pname[:-1] + "m"
        if mname not in particle_decays:
            continue
        v2 = particle_decays[mname]
        assert len(v1) == len(v2), f"mismatch decay count for {pname}/{mname}"
        # Sort so rhoA comes first (matching old build_constraints order)
        def _sort_key(d):
            name = d.outs[0].name
            return (0, name) if name == "rhoA" else (1, name)
        v1_sorted = sorted(v1, key=_sort_key)
        v2_sorted = sorted(v2, key=_sort_key)
        fix_ref = False
        for va, vb in zip(v1_sorted, v2_sorted):
            for ga, gb in zip(va.get_ls_names(), vb.get_ls_names()):
                if fix_ref:
                    free_list.append(ga + "r")
                    free_list.append(ga + "i")
                    free_list.append(gb + "r")
                    free_list.append(gb + "i")
                fix_ref = True
                same_list.append([ga + "r", gb + "r"])
                same_list.append([ga + "i", gb + "i"])
                if va.outs[0]._model and int(va.outs[0]._model.kwargs.get("J", 0)) % 2 == 1:
                    scale_list[gb + "r"] = -1

    # Apply: free → same → scale
    for rn in free_list:
        if rn in fitter.cm.fixed_slots:
            fitter.set_free(rn)
    for group in same_list:
        if len(set(group)) > 1:
            fitter.set_same([group], reset=False)
    if scale_list:
        fitter.set_scale(scale_list, reset=False)


@register_constrain("cp_symmetry_mass", order=2)
def _handle_cp_symmetry_mass(fitter, spec):
    """Same-group mass and width of CP-conjugate particle pairs.

    For each first-intermediate resonance that has both p and m
    variants (e.g. a1(1260)p/a1(1260)m), same-groups:
      ``{name}p_mass = {name}m_mass``
      ``{name}p_width = {name}m_width``
    """
    names = set()
    for chain in fitter.config.full_decay.chains:
        name = chain.decays[1].core.name
        if name.endswith("p"):
            base = name[:-1]
            if base + "m" in {c.decays[1].core.name
                              for c in fitter.config.full_decay.chains}:
                names.add(base)

    for base in sorted(names):
        for attr in ("_mass", "_width"):
            pn = base + "p" + attr
            mn = base + "m" + attr
            if pn in fitter.cm._all_names and mn in fitter.cm._all_names:
                if fitter.cm.name_res.map.get(pn, pn) != mn:
                    fitter.set_same([[pn, mn]], reset=False)


@register_constrain("prefix_same", order=3)
def _handle_prefix_same(fitter, spec):
    """Same-group params where one prefix is replaced by another.

    YAML::

        constrains:
          prefix_symmetry:
            - [KMA, KMB]              # KMA_* = KMB_*
            - [KMA, KMB, KMC, KM2]    # all four equal

    For each param starting with the *first* prefix in a group, finds
    corresponding names for the other prefixes and same-groups them.
    """
    if not isinstance(spec, (list, tuple)):
        return
    all_params = set()
    for comb in fitter.all_comb:
        for p in comb:
            if isinstance(p, str):
                all_params.add(p)
    for group in spec:
        if not isinstance(group, (list, tuple)) or len(group) < 2:
            continue
        pa = group[0]
        for name in all_params:
            if name.startswith(pa):
                rest = name[len(pa):]
                others = [pb + rest for pb in group[1:]
                          if pb + rest in all_params]
                if others:
                    canon = fitter.cm.name_res.map.get(name, name)
                    for other in others:
                        if other != canon:
                            fitter.set_same([[name, other]], reset=False)


# ═══════════════════════════════════════════════════════════════════
# Transform handlers — run before fix/same/scale to create params
# ═══════════════════════════════════════════════════════════════════

@register_constrain("mass_width_bw", order=10)
def _handle_mass_width_bw(fitter, spec):
    """Auto-add BW constraints for all particle models.

    For each unique particle model in the config, adds a
    :class:`~ampfit.param_constraint.BWParamsTransform` and
    Gaussian priors on ``mass_bw`` and ``width_bw``.

    YAML::

        constrains:
            mass_width_bw:
                sigma_mass: 0.010    # width for mass Gaussian (default)
                sigma_width: 0.005   # width for width Gaussian (default)
    """
    from ampfit.param_constraint import BWParamsTransform, GaussianPrior

    sigma_mass = float(spec.get("sigma_mass", 0.010))
    sigma_width = float(spec.get("sigma_width", 0.005))

    seen = set()
    for chain in fitter.config.full_decay.chains:
        for decay in chain.decays[1:]:
            model = decay.core._model
            mid = id(model)
            if mid in seen:
                continue
            seen.add(mid)

            fitter.cm.add_transform(BWParamsTransform(model))

            mu_mass = float(model.kwargs.get("mass", 0.775))
            mu_width = float(model.kwargs.get("width", 0.1))
            fitter.add_prior(GaussianPrior(
                f"{model.name}_mass_bw", mu=mu_mass, sigma=sigma_mass))
            fitter.add_prior(GaussianPrior(
                f"{model.name}_width_bw", mu=mu_width, sigma=sigma_width))


@register_constrain("custom_transforms", order=11)
def _handle_transforms(fitter, spec):
    """``custom_transforms: [{type: ..., ...}, ...]``.

    Handlers receive ``resolve=fitter.get_particle_model`` for model
    lookups.
    """
    from ampfit.param_constraint import transform_from_dict
    for item in spec:
        tr = transform_from_dict(item, resolve=fitter.get_particle_model)
        fitter.cm.add_transform(tr)


@register_constrain("fix_mass_width_default", order=14)
def _handle_fix_mass_width_default(fitter, spec):
    """Fix all mass and width params to config defaults.

    After this handler runs, ``particle_float`` (order=15) can
    selectively free individual params via the particle config's
    ``float:`` field, and ``fix_var`` (order=20) can override::

        constrains:
            fix_mass_width_default: {}       # fix all mass/width
            # particle_float runs next, frees only those with float:

        particle:
            a1(1260)p:
                float: [mass, width]   # kept free
                mass: 1.2422
                width: 0.466
            a2(1320)p:                 # no float: kept fixed
                mass: 1.323
                width: 0.120
    """
    scalars = {'gamma', 'delta_gamma', 'delta_m', 'A_prod', 'poqr', 'poqi'}
    fixed = {}
    for name, val in fitter.defaults.items():
        if 'g_ls' in name or 'total' in name:
            continue
        if name in scalars:
            continue
        if name.endswith('_mass') or name.endswith('_width'):
            fixed[name] = float(val)
    if fixed:
        fitter.set_fixed(fixed, reset=False)


@register_constrain("particle_float", order=15)
def _handle_particle_float(fitter, spec):
    """Free params listed in particle config's ``float:`` field.

    For each particle in the config with a ``float:`` list, unfixes the
    corresponding ``{particle_name}_{item}`` param::

        particle:
            a1(1260)p:
                float: [mass, width]   # frees a1(1260)p_mass + a1(1260)p_width
                mass: 1.2422
                width: 0.466

    Runs before ``fix_var:`` (order=20) so config can re-fix if needed.
    """
    particles = fitter.config.dic.get("particle", {})
    for pname, pcfg in particles.items():
        if not isinstance(pcfg, dict):
            continue
        float_list = pcfg.get("float")
        if not isinstance(float_list, (list, tuple)):
            continue
        for attr in float_list:
            param = f"{pname}_{attr}"
            if param in fitter.cm._all_names and param in fitter.cm.fixed_slots:
                fitter.set_free(param)


# ═══════════════════════════════════════════════════════════════════
# Config constraint handlers
# ═══════════════════════════════════════════════════════════════════

@register_constrain("fix_var", order=20)
def _handle_fix(fitter, spec):
    """``fix_var: {name: value, ...}`` -> :meth:`Fitter.set_fixed`."""
    fitter.set_fixed(spec, reset=False)


@register_constrain("var_equal", order=21)
def _handle_same(fitter, spec):
    """``var_equal: [[a, b, ...], ...]`` -> :meth:`Fitter.set_same`."""
    fitter.set_same(spec, reset=False)


@register_constrain("scale_var", order=22)
def _handle_scale(fitter, spec):
    """``scale_var: {name: factor, ...}`` -> :meth:`Fitter.set_scale`."""
    fitter.set_scale(spec, reset=False)


@register_constrain("free_var", order=30)
def _handle_free_var(fitter, spec):
    """``free_var: [name, ...]`` — unfix params from config."""
    if isinstance(spec, (list, tuple)):
        names = spec
    elif isinstance(spec, dict):
        names = list(spec.keys())
    else:
        return
    for name in names:
        if name in fitter.cm.fixed_slots:
            fitter.set_free(name)


@register_constrain("mass_width_bounds", order=35)
def _handle_mass_width_bounds(fitter, spec):
    """Auto-set bounds for mass and width params from config defaults.

    Mass params: ``[val - 2, val + 2]``
    Other params: ``[max(0.01, val - 20), min(5.0, val + 4)]``

    Skips ``g_ls``, ``total``, scalar params, and already-fixed params.
    """
    flat = set(fitter.var_registry.flat_names)
    scalars = {'gamma', 'delta_gamma', 'delta_m', 'A_prod', 'poqr', 'poqi'}
    for name, val in fitter.defaults.items():
        if 'g_ls' in name or 'total' in name:
            continue
        if name in scalars:
            continue
        if name not in flat:
            continue
        if name.endswith('_mass'):
            lo, hi = float(val) - 2.0, float(val) + 2.0
        else:
            lo = max(0.01, float(val) - 20.0)
            hi = min(5.0, float(val) + 4.0)
        if name not in fitter.cm.fixed_slots:
            fitter.set_range(name, lo, hi)

    # Manual overrides
    overrides = spec.get("overrides", {}) if isinstance(spec, dict) else {}
    for name, (lo, hi) in overrides.items():
        if name not in fitter.cm.fixed_slots:
            fitter.set_range(name, lo, hi)


@register_constrain("var_range", order=36)
def _handle_var_range(fitter, spec):
    """``var_range: {name: [lo, hi], ...}`` — set parameter bounds."""
    for name, (lo, hi) in spec.items():
        fitter.set_range(name, float(lo), float(hi))


@register_constrain("bounds", order=40)
def _handle_bounds(fitter, spec):
    """``bounds: {name: {low: ..., high: ...}, ...}``."""
    for name, sb in spec.items():
        fitter.set_range(name, sb["low"], sb["height"])


@register_constrain("priors", order=50)
def _handle_priors(fitter, spec):
    """``priors: [{type: ..., ...}, ...]``."""
    from ampfit.param_constraint import prior_from_dict
    for item in spec:
        fitter.add_prior(prior_from_dict(item))
