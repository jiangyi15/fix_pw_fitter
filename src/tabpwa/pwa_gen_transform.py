"""Physical parameters → basis weights for the ``pwa_gen`` model.

The kernel consumes two derived quantities per event:

* the **denominator coefficients** ``coeff_wj`` — the real weights of the
  resonance's denominator basis rows (the ``BasisModel`` weights,
  ``[s0, poly, m0·g0_l]`` for the plain preset), and
* the **numerator weights** ``n_i`` — complex, stored as the two real
  ``(magnitude, phase)`` slots of the ``ck`` convention
  (:func:`tabpwa.param_constraint.complex_tail`), because they enter the
  partial-wave coupling as ``ck_w = g_ls·n_i``.

:class:`PWAgenWeightsTransform` publishes exactly those slots: its
``output_names`` are the denominator weight names (a ``BasisModel``'s own
``get_den_name()`` / its basis transform's outputs when it has one) followed
by the numerator ``(magnitude, phase)`` slot names.  Its ``input_names`` are
the particle models' physical parameters, so the constraint manager can
register it as a mass/width-style transform: the forward resolve produces
the weights from the physical parameters and the backward chain (the
base-class finite-difference VJP — the same contract
:meth:`BasisModel.make_basis_transform` declares) maps the kernel gradients
back onto them.
"""

import numpy as np

from tabpwa.param_constraint import Transform, complex_tail
from tabpwa.particle_model.base import BasisModel


class PWAgenWeightsTransform(Transform):
    """Combined particle-model → basis-weight transform of one model."""

    def __init__(self, model):
        self.model = model
        bws = list(model._pg_bws)

        # ── inputs: physical parameters of every denominator resonance ──
        inputs = []

        def _add(names):
            for n in names:
                if n not in inputs:
                    inputs.append(n)

        for rec in bws:
            tr = rec["transform"]
            if tr is not None:
                _add(tr.input_names)
            else:
                _add([f"{rec['name']}_mass"])
                _add(rec["model"].get_gamma_name())

        # ── numerator models (free numerator weights) ────────────────
        num_models = []
        for chain in model.full_decay.chains:
            for decay in chain.decays[1:]:
                mdl = decay.core._model
                if isinstance(mdl, BasisModel) and mdl.get_num_name():
                    if all(mdl is not x for x in num_models):
                        num_models.append(mdl)
        for mdl in num_models:
            tr = mdl.make_basis_transform()
            if tr is None:
                raise ValueError(
                    f"pwa_gen: '{mdl.name}' has numerator weights but no "
                    f"basis transform")
            _add(tr.input_names)

        # ── outputs: denominator weights + numerator slot pairs ──────
        outs = []
        for rec in bws:
            for n in rec["den_names"]:
                if n not in outs:
                    outs.append(n)
        for mdl in num_models:
            for n in mdl.num_slot_names():
                if n not in outs:
                    outs.append(n)

        # A name may legitimately appear on BOTH sides: the default basis
        # maps ``{name}_width`` (physical) onto the SAME name as its weight,
        # because for a plain BW the m0·g0 coefficient IS the fitted width.
        # That is a pass-through, not a shadow — the transform's forward
        # decides the value and the constraint layer replaces the input.
        # Only a clash where the transform does NOT produce the name is a
        # genuine ambiguity (two different physical inputs colliding).
        produced = set(outs)
        clash = sorted(n for n in (set(outs) & set(inputs))
                       if n not in produced)

        self.input_names = inputs
        self.output_names = outs
        self._bws = bws
        self._num_models = num_models
        self._num_transforms = [mdl.make_basis_transform()
                                for mdl in num_models]

    # -- forward --------------------------------------------------------
    def forward(self, d):
        out = {}
        for rec in self._bws:
            mdl = rec["model"]
            names = rec["den_names"]
            tr = rec["transform"]
            if tr is not None:
                res = tr.apply_forward(dict(d))
                res.setdefault("one", 1.0)   # a CONSTANT, not an output
                vals = [float(np.real(res[n])) for n in names]
            elif isinstance(mdl, BasisModel):
                vals = [float(np.real(v)) for v in mdl._den_weights(d)]
            else:
                m0 = float(d[f"{rec['name']}_mass"])
                # row 1 is ``-m²``, so its weight is the shared literal
                # ``"one"`` = +1 (the sign lives in the row).  The gamma rows
                # are ``-i·γ_l(m)`` and their weight is ``m0 · g0_l`` — the
                # ``m0`` factor is part of the weight, not of the name.
                vals = ([m0 * m0, 1.0]
                        + [m0 * float(d[g]) for g in mdl.get_gamma_name()])
            for n, v in zip(names, vals):
                out[n] = v

        mag, phase = complex_tail()
        for mdl, tr in zip(self._num_models, self._num_transforms):
            if tr is not None:
                res = tr.apply_forward(dict(d))
                for n in mdl.num_slot_names():
                    out[n] = float(np.real(res[n]))
            else:                                   # pragma: no cover
                for head, w in zip(mdl.get_num_name(), mdl._num_weights(d)):
                    out[f"{head}{mag}"] = float(abs(w))
                    out[f"{head}{phase}"] = float(np.angle(w))
        return out


class DefaultBasisWeightsTransform(Transform):
    """Physical parameters → the basis weights of ANY resonance model.

    Every model is a ratio of two fixed bases (``BaseModel.num_basis`` /
    ``den_basis``), so its weights are a function of the physical parameters
    and can be published as transform OUTPUTS.  That is what makes
    ``pwa_gen`` fit *physics* instead of one free weight per basis row.

    ``input_names`` are the physical ``{name}_mass`` plus the gamma couplings;
    ``output_names`` are exactly :meth:`get_den_name` (the kernel ``coeff``
    names), so the names always match the rows the builder emitted.

    The default basis is ``[1, m², -i·γ_j(m)]`` with weights
    ``[m0², -1, m0·g_j]`` — the plain BW denominator.  A model that overrides
    ``den_basis`` **must** override this too (``kmatrix2`` and
    ``RhoOmegaGS`` do); the base class cannot derive an arbitrary basis'
    weights from ``m0``/``g``.
    """

    _has_inverse = False

    def __init__(self, model):
        self.model = model
        name = model.name
        self.gamma_names = list(model.get_gamma_name())
        self.input_names = [f"{name}_mass"] + list(self.gamma_names)
        # ``"one"`` is a CONSTANT (the m² coefficient, always +1) — it is not a
        # function of the physical parameters, so it is not a transform output.
        # The constraint layer pins it (`set_fixed`), the same mechanism that
        # pins any other fixed parameter.
        self.output_names = [n for n in model.get_den_name() if n != "one"]

    def forward(self, d):
        """``m0²``, ``-1``, ``m0·g_j`` — the uniform basis weights."""
        d = d or {}
        m0 = float(d.get(f"{self.model.name}_mass",
                         self.model.kwargs.get("mass", 0.775)))
        # ``"one"`` is pinned by the constraint layer, not produced here.
        vals = [m0 * m0]
        # Each gamma row's weight is ``m0·g`` and is named ``{g}_t`` — the
        # kernel-facing slot, DISTINCT from the physical fit parameter ``{g}``.
        # Emitting the fit name here would make the transform overwrite its
        # own input (`apply_forward` replaces `output_names` in the resolved
        # dict), losing the physical coupling.
        vals += [m0 * float(d.get(g, 0.0)) for g in self.gamma_names]
        return dict(zip(self.output_names, vals))
