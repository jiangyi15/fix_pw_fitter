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
# Built-in handlers -- map directly to Fitter methods
# ═══════════════════════════════════════════════════════════════════

@register_constrain("fix_var")
def _handle_fix(fitter, spec):
    """``fix_var: {name: value, ...}`` -> :meth:`Fitter.set_fixed`."""
    fitter.set_fixed(spec, reset=False)


@register_constrain("var_equal")
def _handle_same(fitter, spec):
    """``var_equal: [[a, b, ...], ...]`` -> :meth:`Fitter.set_same`."""
    fitter.set_same(spec, reset=False)


@register_constrain("scale_var")
def _handle_scale(fitter, spec):
    """``scale_var: {name: factor, ...}`` -> :meth:`Fitter.set_scale`."""
    fitter.set_scale(spec, reset=False)


@register_constrain("bounds")
def _handle_bounds(fitter, spec):
    """``bounds: {name: {low: ..., high: ...}, ...}``."""
    for name, sb in spec.items():
        fitter.set_range(name, sb["low"], sb["high"])


# ═══════════════════════════════════════════════════════════════════
# Plugin-style handlers
# ═══════════════════════════════════════════════════════════════════

@register_constrain("priors")
def _handle_priors(fitter, spec):
    """``priors: [{type: ..., ...}, ...]``."""
    from ampfit.param_constraint import prior_from_dict
    for item in spec:
        fitter.add_prior(prior_from_dict(item))


@register_constrain("custom_transforms")
def _handle_transforms(fitter, spec):
    """``custom_transforms: [{type: ..., ...}, ...]``.

    Handlers receive ``resolve=fitter.get_particle_model`` for model
    lookups.
    """
    from ampfit.param_constraint import transform_from_dict
    for item in spec:
        tr = transform_from_dict(item, resolve=fitter.get_particle_model)
        fitter.cm.add_transform(tr)


@register_constrain("mass_width_bw")
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


# ═══════════════════════════════════════════════════════════════════
# Structural constraint handlers (used via constrains section)
# ═══════════════════════════════════════════════════════════════════

@register_constrain("cp_symmetry", order=1)
def _handle_cp_symmetry(fitter, spec):
    """Constrain CP-conjugate pairs using chain structure.

    Collects decays by first intermediate resonance name, then for each
    CP conjugate pair (p/m suffix):
    - Same-groups ``_total_0`` within all R+ and within all R-
    - Same-groups ``_g_ls`` between R+ and R- counterparts
    - Scales by -1 if the decay's outgoing particle has odd J
    - Unfixes all g_ls except one reference
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

    # Apply collected constraints
    for rn in free_list:
        if rn in fitter.cm.fixed_slots:
            fitter.set_free(rn)
    for group in same_list:
        if len(set(group)) > 1:
            fitter.set_same([group], reset=False)
    if scale_list:
        fitter.set_scale(scale_list, reset=False)


@register_constrain("ck_redundancy", order=0)
def _handle_ck_redundancy(fitter, spec):
    """Fix CK scale redundancies by prefix context.

    For each CK term ``[t0, t1, t2, ...]``, process positions in order.
    At each position p > 0, if the prefix ``(t0, ..., t_{p-1})`` has been
    seen before at this position, the term at ``p`` is a ratio parameter
    (free).  Otherwise, fix the term as the reference for this context.

    Concept: for CK products like ``(a+b)(c+d) → [a,b,c,d]``, fix the
    first occurrence ``[a,c]`` (positions 1,2), then ``[a,d]`` and
    ``[b,c]`` are ratios — only the differing position varies freely.

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
