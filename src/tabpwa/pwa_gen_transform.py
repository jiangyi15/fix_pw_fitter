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

        clash = sorted(set(outs) & set(inputs))
        if clash:
            raise ValueError(
                f"pwa_gen: basis weight name(s) {clash} shadow a physical "
                f"parameter — rename them in the particle model")

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
                vals = [float(np.real(res[n])) for n in names]
            elif isinstance(mdl, BasisModel):
                vals = [float(np.real(v)) for v in mdl._den_weights(d)]
            else:
                m0 = float(d[f"{rec['name']}_mass"])
                vals = ([m0 * m0, -1.0]
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
