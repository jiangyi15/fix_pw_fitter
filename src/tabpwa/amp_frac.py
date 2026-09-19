"""
Amplitude fractions: compute |A|² sums for subsets of partial waves and their ratios.

A subset is defined by a mask on the coupling vector *ck* (set unused entries
to ``0j``).  The fraction of a subset relative to the denominator is::

    R_i = Σ w·|A_{mask_i}|² / Σ w·|A_denom|²

where *w* are the weights from the data handle.

Usage::

    from tabpwa.amp_frac import AmplitudeFractions

    af = AmplitudeFractions(fitter, fit_result)

    # Fractions for two subsets in a single sweep
    vals, errs = af.fractions([[0,1,2], [3,4,5]])
    # vals = [0.12, 0.08], errs = [0.01, 0.005]
"""

import numpy as np


class AmplitudeFractions:
    """Amplitude fractions for a fitted amplitude model.

    Args:
        fitter: :class:`~tabpwa.fitter.Fitter` instance (with data loaded).
        fit_result: ``OptimizeResult`` from ``fitter.fit()``.
        data: optional data handle for computing |A|² sums.
              If ``None`` (default), uses `fitter._phsp_holder`.
    """

    def __init__(self, fitter, fit_result, data=None):
        self.fitter = fitter
        self.fit_result = fit_result
        self._data = fitter._phsp_holder if data is None else data
        self._params, self._resolved = fitter.build_params(fit_result.x)
        self._n_ck = len(self._params["ck"])
        self._cache = {}  # mask_tuple -> (total, grad_dict)

    # ── internal helpers ──────────────────────────────────────────

    def _ck_masked(self, mask):
        """Return a copy of *ck* with entries outside *mask* zeroed."""
        ck = self._params["ck"].copy()
        if mask is not None:
            for i in range(self._n_ck):
                if i not in mask:
                    ck[i] = 0.0j
        return ck

    def _mask_key(self, mask):
        """Hashable key for *mask* (``None`` = all ck)."""
        return tuple(sorted(mask)) if mask is not None else None

    def _compute_total_and_grad(self, p, mask):
        """Run kernel with a given ck mask and return (total, grad_dict).

        Results are cached by mask so repeated calls with the same mask
        (e.g. denominator shared across sub-channels) skip the backend.
        """
        key = self._mask_key(mask)
        if key is not None and key in self._cache:
            return self._cache[key]

        p = dict(p)
        p["ck"] = self._ck_masked(mask)
        Q, grads, _ = self.fitter.backend.compute(p, self._data, norm=None)
        total = float(Q)

        # Mask the ck gradient exactly like the couplings; every other
        # parameter group is handed to the transform untouched, so this code
        # never needs to know which groups exist (ck/m0/g0[/scalar]).
        grads = dict(grads)
        if mask is not None:
            ck_grad = np.asarray(grads["ck"]).copy()
            for i in range(self._n_ck):
                if i not in mask:
                    ck_grad[i] = 0.0j
            grads["ck"] = ck_grad
        grad = self.fitter._kernel_builder.backward(grads, self._resolved)

        if key is not None:
            self._cache[key] = (total, grad)
        return total, grad

    def _phys_to_params(self, phys_dict):
        """Kernel params at the best fit with *phys_dict* overrides applied.

        Delegates the name → (array, slot) mapping to the model's parameter
        transform, so every parameter group (ck/m0/g0[/scalar]) is handled
        uniformly.
        """
        kb = self.fitter._kernel_builder
        return kb.forward({**self._resolved, **phys_dict})

    def _default_param_names(self):
        """Free (optimized) parameter names, from the variable registry.

        These are the x-space names the fit covariance is defined over; the
        transform owns the kernel-side grouping, so there is no need to list
        m0/g0 here (``cal_uncertainties_multi`` already restricts to names
        present in the resolved dict).
        """
        return list(self.fitter.cm.var_registry.flat_names)

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
            p = R._phys_to_params(phys)
            total_den, _ = R._compute_total_and_grad(p, denominator)
            vals = []
            for m in masks:
                total_num, _ = R._compute_total_and_grad(p, m)
                vals.append(total_num / total_den)
            return np.array(vals)

        def fun_jac(phys):
            """Forward + backward: fractions + analytical gradient dicts."""
            p = R._phys_to_params(phys)

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

        vals, cov = fitter.cal_uncertainties_multi(
            fun, param_names, self.fit_result, jac=jac)
        errs = np.sqrt(np.diag(cov))
        return vals, errs
