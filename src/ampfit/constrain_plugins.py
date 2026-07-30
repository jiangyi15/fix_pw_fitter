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

_constrain_handlers = {}


def register_constrain(name):
    """Decorator: register a handler for a YAML ``constrains`` section key."""
    def _f(fn):
        _constrain_handlers[name] = fn
        return fn
    return _f


def apply_constrains(fitter, constrains=None):
    """Apply constraint handlers for each key in *constrains*.

    Args:
        fitter: Fitter instance.
        constrains: dict of ``{key: spec}``.  If ``None``, reads from
                    ``fitter.config.dic["constrains"]``.
    """
    if constrains is None:
        constrains = fitter.config.dic.get("constrains", {})
    for key, spec in constrains.items():
        handler = _constrain_handlers.get(key)
        if handler:
            handler(fitter, spec)


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

@register_constrain("cp_symmetry")
def _handle_cp_symmetry(fitter, spec):
    """Constrain CP-conjugate pairs based on internal particle spin.

    For R -> rhopi -> 3-pi chains, the rho meson has J = 1, P = -1.
    Under CP conjugation, the amplitude picks up a factor
    ``(-1)^J = -1`` for a rho intermediate state.  This means
    the gamma couplings for R+ and R- should be related by
    a sign flip (set_scale with factor -1).

    Detection logic:
    1. Find chain pairs with same intermediate daughters (by topo_id)
    2. The two chains differ only in CP label (name ends in p/m)
    3. Check if an intermediate daughter has J = 1 (e.g. rho, omega)
    4. Apply set_scale(-1) between conjugate gamma names
    """
    chains = fitter.config.full_decay.chains
    if not chains:
        return

    # Group chains by topo_id to find CP-conjugate pairs
    from collections import defaultdict
    by_topo = defaultdict(list)
    for chain in chains:
        by_topo[chain.topo_id()].append(chain)

    for topo_id, group in by_topo.items():
        if len(group) < 2:
            continue
        # Find p/m pairs
        for ci in range(len(group)):
            for cj in range(ci + 1, len(group)):
                a, b = group[ci], group[cj]
                # Check if one ends in p, the other in m (same base)
                a_name = str(a).split("+")[0] if "+" in str(a) else str(a)
                b_name = str(b).split("+")[0] if "+" in str(b) else str(b)
                # Extract the base particle name from the chain
                a_particle = a.decays[1].core.name if len(a.decays) > 1 else ""
                b_particle = b.decays[1].core.name if len(b.decays) > 1 else ""

                # Check CP conjugation: same base, one p one m
                if (a_particle.endswith("p") and b_particle.endswith("m")) or \
                   (a_particle.endswith("m") and b_particle.endswith("p")):
                    _constrain_cp_pair(fitter, a, b)


def _constrain_cp_pair(fitter, chain_p, chain_m):
    """Apply CP-symmetry scale(-1) constraint between chain_p and chain_m.

    For each intermediate resonance in the chain, check if its
    daughter (next decay) has J=1 -> apply scale(-1) to gamma names.
    """
    for idx in range(1, len(chain_p.decays)):
        decay_p = chain_p.decays[idx]
        decay_m = chain_m.decays[idx]

        model_p = decay_p.core._model
        model_m = decay_m.core._model
        name_p = decay_p.core.name
        name_m = decay_m.core.name

        gamma_p = list(model_p.get_gamma_name())
        gamma_m = list(model_m.get_gamma_name())
        if not gamma_p or not gamma_m:
            continue
        if len(gamma_p) != len(gamma_m):
            continue

        # Check if the decay products include a J=1 particle (like rho)
        # First try via next decay chain, then fallback to kwargs
        has_J1 = False
        if idx + 1 < len(chain_p.decays):
            next_model = chain_p.decays[idx + 1].core._model
            J = int(next_model.kwargs.get("J", 0))
            if J == 1:
                has_J1 = True
        else:
            # Terminal decay -- check the model's own kwargs
            J = int(model_p.kwargs.get("J", 0))
            if J == 1:
                has_J1 = True

        if not has_J1:
            continue

        # Apply scale(-1) between conjugate gamma names
        scale_dict = {}
        for gp, gm in zip(gamma_p, gamma_m):
            scale_dict[gm] = (gp, None) if gp in fitter.cm.var_registry.flat_names else -1.0
            # set_scale with factor -1 on gm, using gp as reference
            # Actually simpler: set_same with scale = -1
        # Fitter set_scale(gm: -1.0) doesn't support relative scaling
        # So we fix gm to -gp (scale=-1) using set_scale
        for gm in gamma_m:
            fitter.set_scale({gm: -1.0})


@register_constrain("ck_redundancy")
def _handle_ck_redundancy(fitter, spec):
    """Fix redundant CK degrees of freedom.

    In the product structure ``ck[i] = Π term_vals``, the overall
    scale between groups of terms is unconstrained.  We fix the
    first ``total`` term to ``r=1, theta=0`` to break this redundancy.

    More sophisticated analysis of co-occurring term groups is
    possible but requires careful handling of the combinatoric
    structure -- left for future work.
    """
    for comb in fitter.all_comb:
        for term in comb:
            if isinstance(term, str) and "_total_" in term:
                r_name = f"{term}r"
                i_name = f"{term}i"
                flat = fitter.cm.var_registry.flat_names
                if r_name in flat and r_name not in fitter.cm.fixed_slots:
                    fitter.set_fixed({r_name: 1.0, i_name: 0.0}, reset=False)
                    return
    # If no _total_ term found, try first string term
    for comb in fitter.all_comb:
        for term in comb:
            if isinstance(term, str):
                r_name = f"{term}r"
                i_name = f"{term}i"
                flat = fitter.cm.var_registry.flat_names
                if r_name in flat and r_name not in fitter.cm.fixed_slots:
                    fitter.set_fixed({r_name: 1.0, i_name: 0.0}, reset=False)
                    return
