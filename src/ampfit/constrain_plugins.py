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


def apply_constrains(fitter):
    """Process the ``constrains`` section of ``fitter.config``.

    Each top-level key in the ``constrains`` dict is dispatched to the
    registered handler, passing the parsed YAML value as the ``spec``
    argument.

    If no handler is registered for a key, it is silently skipped.
    """
    section = fitter.config.dic.get("constrains", None)
    if not section:
        return
    for key, spec in section.items():
        handler = _constrain_handlers.get(key)
        if handler:
            handler(fitter, spec)


# ═══════════════════════════════════════════════════════════════════
# Built-in handlers — map directly to Fitter methods
# ═══════════════════════════════════════════════════════════════════

@register_constrain("fix_var")
def _handle_fix(fitter, spec):
    """``fix_var: {name: value, ...}`` → :meth:`Fitter.set_fixed`."""
    fitter.set_fixed(spec, reset=False)


@register_constrain("var_equal")
def _handle_same(fitter, spec):
    """``var_equal: [[a, b, ...], ...]`` → :meth:`Fitter.set_same`."""
    fitter.set_same(spec, reset=False)


@register_constrain("scale_var")
def _handle_scale(fitter, spec):
    """``scale_var: {name: factor, ...}`` → :meth:`Fitter.set_scale`."""
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
