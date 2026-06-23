"""
Amplitude fractions: compute |A|² sums for subsets of partial waves and their ratios.

A subset is defined by a mask on the coupling vector *ck* (set unused entries
to ``0j``).  The fraction of a subset relative to the denominator is::

    R_i = Σ w·|A_{mask_i}|² / Σ w·|A_denom|²

where *w* are the weights from the data handle.

Usage::

    from ampfit.amp_frac import AmplitudeFractions

    af = AmplitudeFractions(fitter, fit_result)

    # Fractions for two subsets in a single sweep
    vals, errs = af.fractions([[0,1,2], [3,4,5]])
    # vals = [0.12, 0.08], errs = [0.01, 0.005]
"""

import numpy as np


class AmplitudeFractions:
    """Amplitude fractions for a fitted amplitude model.

    Args:
        fitter: :class:`~ampfit.fitter.Fitter` instance (with data loaded).
        fit_result: ``OptimizeResult`` from ``fitter.fit()``.
        data: optional data handle for computing |A|² sums.
              If ``None`` (default), uses `fitter._phsp_holder`.
    """

    def __init__(self, fitter, fit_result, data=None):
        self.fitter = fitter
        self.fit_result = fit_result
        self._data = fitter._phsp_holder if data is None else data
        self._params, self._resolved, _, _ = fitter._build_params(fit_result.x)
        self._n_ck = len(self._params["ck"])

    # ── internal helpers ──────────────────────────────────────────

    def _ck_masked(self, mask):
        """Return a copy of *ck* with entries outside *mask* zeroed."""
        ck = self._params["ck"].copy()
        if mask is not None:
            for i in range(self._n_ck):
                if i not in mask:
                    ck[i] = 0.0j
        return ck

    def _compute_total_and_grad(self, p, mask):
        """Run kernel with a given ck mask and return (total, grad_dict).

        ``total = Σ weight·P``  (the ``norm=None`` forward sum).
        ``grad_dict`` maps physical parameter name → d(total)/d(param).
        Includes CK, m0, and g0 gradients.
        """
        p = dict(p)
        ck_masked = self._ck_masked(mask)
        p["ck"] = ck_masked
        Q, grads, _ = self.fitter.backend.compute(p, self._data, norm=None)
        total = float(Q)

        cfg = self.fitter.config
        grad = {}
        # CK gradients: zero out entries disconnected by mask
        grad_ck = grads["ck"].copy()
        if mask is not None:
            for i in range(self._n_ck):
                if i not in mask:
                    grad_ck[i] = 0.0j
        slot_dict = {n: float(v) for n, v in self._resolved.items()}
        ck_grads = self.fitter.cm.pc.backprop_grad(slot_dict, grad_ck)
        grad.update(ck_grads)
        # m0 gradients
        for i, name in enumerate(cfg.m0_phys_name):
            grad[name] = float(grads["m0"][i])
        # g0 gradients
        for i, name in enumerate(cfg.g0_phys_name):
            grad[name] = float(grads["g0"][i])
        return total, grad

    def _phys_to_params(self, p, phys_dict):
        """Apply physical-parameter shifts into a params dict copy."""
        p = dict(p)
        cfg = self.fitter.config
        for name, val in phys_dict.items():
            if name in cfg.m0_phys_name:
                p["m0"][cfg.m0_phys_name.index(name)] = val
            elif name in cfg.g0_phys_name:
                p["g0"][cfg.g0_phys_name.index(name)] = val
        return p

    def _default_param_names(self):
        """Return all varying physical parameter names in the resolved dict.

        Includes CK internal names (from VariableRegistry), m0, and g0.
        """
        resolved = self._resolved
        cfg = self.fitter.config
        names = []
        # CK internal variable names (real/imag parts)
        for n in self.fitter._var_registry._entries:
            names.append(n)
        for n in cfg.m0_phys_name:
            if n in resolved:
                names.append(n)
        for n in cfg.g0_phys_name:
            if n in resolved:
                names.append(n)
        return names

    # ── public API ────────────────────────────────────────────────

    def fractions(self, masks, denominator=None, param_names=None, jac=True):
        """Compute mean fractions and propagated uncertainties.

        ``R_i = Σ w·|A_mask_i|² / Σ w·|A_denom|²``

        Uses a single 3‑point sweep (``jac=False``) or analytical
        gradients (``jac=True``) via ``cal_uncertainties_multi``.

        Args:
            masks: list of ck-index iterables, e.g. ``[[0,1,2], [3,4,5]]``.
            denominator: ck indices for the denominator.  ``None`` = all ck.
            param_names: physical params to vary (default: all m0/g0
                         present in the resolved dict).
            jac: if True, use analytical gradient from the kernel's
                 backward pass instead of finite differences.

        Returns:
            ``(values, errors)`` — two 1-D arrays of length ``len(masks)``.
        """
        if param_names is None:
            param_names = self._default_param_names()

        fitter, R = self.fitter, self

        def fun_3pt(phys):
            """Forward‑only: compute fractions from weighted sums."""
            p = R._phys_to_params(R._params, phys)
            total_den, _ = R._compute_total_and_grad(p, denominator)
            vals = []
            for m in masks:
                total_num, _ = R._compute_total_and_grad(p, m)
                vals.append(total_num / total_den)
            return np.array(vals)

        def fun_jac(phys):
            """Forward + backward: fractions + analytical gradient dicts."""
            p = R._phys_to_params(R._params, phys)

            # Denominator (computed once, shared across all masks)
            total_den, grad_den = R._compute_total_and_grad(p, denominator)

            values = []
            grad_dicts = []
            for m in masks:
                total_num, grad_num = R._compute_total_and_grad(p, m)
                R_i = total_num / total_den
                values.append(R_i)

                d = {}
                all_keys = set(grad_num) | set(grad_den)
                for pname in all_keys:
                    if pname in grad_num and pname in grad_den:
                        d[pname] = (grad_num[pname] / total_den
                                    - total_num / total_den**2 * grad_den[pname])
                    elif pname in grad_num:
                        d[pname] = grad_num[pname] / total_den
                    elif pname in grad_den:
                        d[pname] = -total_num / total_den**2 * grad_den[pname]
                    else:
                        d[pname] = 0.0
                grad_dicts.append(d)

            return values, grad_dicts

        fun = fun_jac if jac else fun_3pt

        vals, cov, corr = fitter.cal_uncertainties_multi(
            fun, param_names, self.fit_result, jac=jac)
        errs = np.sqrt(np.diag(cov))
        return vals, errs
