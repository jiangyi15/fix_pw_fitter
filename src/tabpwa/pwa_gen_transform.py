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
            tr = rec["transform"]
            if tr is None:
                # ``None`` is the INTERFACE default (``BasisModel``): the basis
                # weights ARE the raw parameters, so there is nothing to
                # derive.  That is only valid when every weight is a constant
                # the constraint layer pins (``"one"``); a real free weight
                # needs a transform, and THAT is an error.
                free = [n for n in rec["den_names"] if n != "one"]
                if free:
                    raise ValueError(
                        f"pwa_gen: '{rec['name']}' declares free denominator "
                        f"weights {free} but returns no basis transform — a "
                        f"model whose weights are parameters must provide one "
                        f"(make_basis_transform)")
                res = {}
            else:
                res = tr.apply_forward(dict(d))
            # ``"one"`` is the constant m² coefficient (the row is ``-m²``),
            # so THIS transform supplies it directly rather than expecting the
            # per-resonance transform to produce a parameter-independent value
            # — the combined dict is what the kernel consumes, so the constant
            # belongs here, not in a patch on the sub-transform's output.
            for n in rec["den_names"]:
                out[n] = 1.0 if n == "one" else float(np.real(res[n]))

        mag, phase = complex_tail()
        for mdl, tr in zip(self._num_models, self._num_transforms):
            if tr is not None:
                res = tr.apply_forward(dict(d))
                for n in mdl.num_slot_names():
                    out[n] = float(np.real(res[n]))
            else:                                   # pragma: no cover
                for head, w in zip(mdl.get_num_name(), mdl.num_weights(d)):
                    out[f"{head}{mag}"] = float(abs(w))
                    out[f"{head}{phase}"] = float(np.angle(w))
        return out


