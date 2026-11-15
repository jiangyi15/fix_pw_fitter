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
        data: the sample to integrate over, as a **numpy dict**
              (``mass``/``q``/``angle``/``weight``/``bkg``).  ``None``
              (default) = the fitter's phase-space arrays.  A backend handle
              is not accepted: the sample is loaded here, lazily, and is what
              a batched run slices.
    """

    def __init__(self, fitter, fit_result, data=None):
        if data is not None and not (isinstance(data, dict) and "mass" in data):
            raise TypeError(
                "AmplitudeFractions: data must be a numpy dict of events "
                f"(mass/q/angle/weight/bkg), not {type(data).__name__}")
        self.fitter = fitter
        self.fit_result = fit_result
        self._given = data is not None
        self._data = data            # None = whatever the fitter holds later
        self._handle = None            # resolved lazily, see _get_handle()
        self._own = False              # True when we loaded it ourselves
        self._cache = {}  # mask_tuple -> (total, grad_dict)
        # the one point (fit_result.x); resolved on first use, see _point()
        self._params = self._resolved = None
        self._point()
        self._n_ck = len(self._params["ck"])

    def _point(self):
        """``(x, params, resolved)`` of the fitted point — the instance's only
        point.

        A fraction, a yield or an uncertainty here is a statement about *this
        fit result*, not about an arbitrary parameter vector, so there is
        exactly one point and it is taken from ``fit_result.x``.  Every
        quantity the instance hands out (the integrand, each ``Fwd``, the
        ratio, the covariance) is evaluated there, so they combine
        meaningfully.

        ``Fitter.build_params`` is cached on ``(x, cm.revision)``, so this is
        cheap per call and a constraint change is picked up automatically (the
        revision bump invalidates that cache).
        """
        x = np.asarray(self.fit_result.x, dtype=float).ravel()
        params, resolved = self.fitter.build_params(x)
        if getattr(self, "_params", None) is not params:
            self._params, self._resolved = params, resolved
            self._cache.clear()          # mask memo is per parameter point
        return x, params, resolved

    # ── sample ────────────────────────────────────────────────────

    @property
    def _np(self):
        """The arrays of the sample — read live, not snapshotted.

        ``set_phsp`` after construction must be visible to a batched run;
        caching this at ``__init__`` made ``batch=None`` follow the new sample
        while ``batch=N`` sliced the old one.
        """
        if self._given:
            return self._data
        return getattr(self.fitter, "_phsp_np", None)

    def _get_handle(self):
        """The backend handle of the sample.

        Resolution order: the arrays given to the constructor (loaded here,
        and ours to free); else the fitter's ``_phsp_holder`` from
        ``set_phsp`` (used as-is); else a load of the fitter's
        ``_phsp_np`` arrays.

        The fitter's holder is re-checked on every call: ``set_phsp`` frees and
        replaces it, so a cached one would integrate the *old* sample (and the
        freed one would raise).  Only a handle we loaded ourselves is cached.
        """
        f = self.fitter
        if not self._given:
            holder = getattr(f, "_phsp_holder", None)
            if holder is not None:
                if self._own:                 # the sample changed underneath us
                    self.close()
                self._handle, self._own = holder, False
                return self._handle
        if self._handle is None:
            if self._given and self._np is not None:
                self._handle = f.backend.load_data(self._np)
                self._own = True
            elif self._np is not None:
                self._handle = f.backend.load_data(self._np)
                self._own = True
            else:
                raise RuntimeError(
                    "amplitude fractions: no sample — call set_phsp() on the "
                    "fitter or pass data=arrays")
        return self._handle

    def close(self):
        """Free the handle we loaded (idempotent; the fitter's is untouched)."""
        h, self._handle = self._handle, None
        own, self._own = self._own, False
        if own and h is not None and hasattr(h, "free"):
            h.free()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ── internal helpers ──────────────────────────────────────────

    def _mask_key(self, mask):
        """Hashable key for *mask* (``None`` = all ck)."""
        return tuple(sorted(mask)) if mask is not None else None

    def _compute_total_and_grad(self, p, mask, handle=None, resolved=None):
        """Run kernel with a given ck mask and return (total, grad_dict).

        Results are cached by mask so repeated calls with the same mask
        (e.g. denominator shared across sub-channels) skip the backend.
        *handle* overrides the sample (a chunk); the cache is then bypassed,
        since a mask name no longer identifies a value.
        """
        key = self._mask_key(mask) if handle is None else None
        if key is not None and key in self._cache:
            return self._cache[key]

        p = dict(p)
        # mask the *given* ck, not the best-fit one: the finite-difference
        # path perturbs it through ``_phys_to_params``
        p["ck"] = self._ck_masked(mask, p["ck"])
        Q, grads, _ = self.fitter.backend.compute(
            p, self._get_handle() if handle is None else handle, norm=None)
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
        grad = self.fitter._kernel_builder.backward(
            grads, self._resolved if resolved is None else resolved)

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

    # ── public contributions (what the fitter integrates) ─────────

    def integrand(self, mask):
        """Per-event integrand of one ck mask at the fitted point.

        Returns ``{"params": kernel params, "resolved": physical dict,
        "mask": mask}`` — the per-sample integrand that
        :meth:`~tabpwa.fitter.Fitter.sum_stream` accumulates.  The masked
        ``ck`` is already in ``params``, so the caller of the integration only
        has to pick the measure (phsp, data, a chunk of either) and the
        normalisation.

        Always the model frame at ``fit_result.x``; there is no override —
        a value taken at another point does not belong to this instance's
        gradient (see :meth:`fwd_integrate`).
        """
        params = dict(self._params)
        params["ck"] = self._ck_masked(mask, params["ck"])
        return {"params": params, "resolved": dict(self._resolved),
                "mask": mask}

    def _ck_masked(self, mask, ck=None):
        """Copy of *ck* with entries outside *mask* zeroed (boolean mask)."""
        ck = np.array(self._params["ck"] if ck is None else ck, copy=True)
        if mask is not None:
            keep = np.zeros(ck.shape[0], dtype=bool)
            keep[list(mask)] = True
            ck[~keep] = 0.0j
        return ck

    def fwd_integrate(self, masks, handle):
        """Per-**handle** integrand: one ``Fwd`` per mask for that sample.

        This is the unit :meth:`tabpwa.fitter.Fitter.sum_stream` accumulates;
        the streaming itself (chunking, loading, freeing, summing) is the
        fitter's job, not this class's.  Give it a backend handle and it
        returns ``sum w.|A_mask|^2`` with its gradient for *that* handle::

            out = f.sum_stream(lambda h: af.fwd_integrate(masks, h),
                               data=arrays, batch=50_000)
            R   = out[0] / out[2]                  # value + exact gradient
            band = R.sigma(fit_result.hess_inv)    # no cal_uncertainties call

        The gradient spans the free parameters in ``x`` order, because the
        chaining is delegated to :meth:`tabpwa.fitter.Fitter.fwd_of` (which
        runs ``cm.full_gradient``) — the resolved-name gradient is never
        re-indexed by literal name, so aliases and bounds Jacobians survive.

        Args:
            masks: list of ck-index iterables (``[[1], [2], [1, 2]]``).
            handle: a backend handle (from ``backend.load_data``).
        The point is the fit result's — this class is an *interpretation of
        one fit*, not a function of any parameter vector.  There is therefore
        no ``x=`` and no ``phys=`` override: for another point, build another
        ``AmplitudeFractions`` (or a new fit result).  Both used to exist and
        both were silently wrong — a value evaluated at one point with a
        gradient chained at another (measured: gradient off by 0.22 from the
        FD of the same overridden observable).

        Returns:
            ``[Fwd, ...]`` — one per mask, the ck gradient masked exactly
            like the couplings and carried to the free parameters.  A mask
            over a *fixed* coupling legitimately has a zero gradient.
        """
        f = self.fitter
        # ONE point for the whole call: the kernel params, the backward and the
        # chain all come from the fit result's x, so every returned dual's
        # value and gradient describe the same point — which is what makes a
        # ratio of them meaningful.
        x, params, resolved = self._point()

        out = []
        for mask in masks:
            spec = self.integrand(mask)
            Q, kg, _ = f.backend.compute(spec["params"], handle, norm=None)
            if mask is not None:                     # mask the ck gradient
                ck_g = np.array(kg["ck"], copy=True)
                keep = np.zeros(ck_g.shape[0], dtype=bool)
                keep[list(mask)] = True
                ck_g[~keep] = 0.0j
                kg = dict(kg, ck=ck_g)
            out.append(f.fwd_of(Q, f.fwd_backward(kg, resolved), x))
        return out

    def fractions(self, masks, denominator=None, param_names=None,
                  jac=True, batch=None):
        """Compute mean fractions and propagated uncertainties.

        ``R_i = Σ w·|A_mask_i|² / Σ w·|A_denom|²``

        Values are always the same three weighted sums; only the uncertainty
        route differs: ``jac=True`` (default) uses the kernel's analytic
        gradient, ``jac=False`` a central-difference sweep.  ``batch=N``
        streams either route one chunk at a time (``2·len(x)`` model
        evaluations per sweep for the FD one, so it is proportionally slower).
        ``param_names`` names the parameters handed to the analytic callback;
        an empty or unknown set yields zero errors on every route.

        Args:
            masks: list of ck-index iterables, e.g. ``[[0,1,2], [3,4,5]]``.
            denominator: ck indices for the denominator.  ``None`` = all ck.
            param_names: physical params to vary (default: all m0/g0
                         present in the resolved dict).
            jac: if True, use analytical gradient from the kernel's
                 backward pass instead of finite differences.
            batch: events per chunk.  ``None`` (default) integrates the whole
                 phase space in one pass; ``N`` streams it through
                 :meth:`~tabpwa.fitter.Fitter.sum_stream`, so peak memory is
                 one chunk — same values, more loads.
        Returns:
            ``(values, errors)`` — two 1-D arrays of length ``len(masks)``.
        """
        fitter, R = self.fitter, self

        # Re-derive the parameter point: a constraint change after
        # construction (``set_blind``, ``set_scale``, …) moves the model frame,
        # and the three routes must not disagree because one of them read a
        # stale snapshot.  Cheap: one forward resolve.
        self._point()

        if param_names is None:
            param_names = self._default_param_names()

        # Both analytic routes evaluate the model at the SAME point: the
        # MODEL frame (``self._params``/``self._resolved``, built by
        # ``build_params``), which is what the kernel computes at and what
        # ``cal_uncertainties_multi`` now hands *fun*.  The blinded *report*
        # frame only removes the offset for display; evaluating there would
        # describe a parameter point the fit never fitted.

        def streamed():
            """(denominator Fwd, [mask Fwds]) via the chunked sum."""
            order = [denominator] + list(masks)
            if R._np is None:
                raise ValueError(
                    "fractions: no sample to stream — call set_phsp() on the "
                    "fitter or pass data=arrays")
            out = fitter.sum_stream(
                lambda h: R.fwd_integrate(order, h),
                data=R._np, batch=batch)
            return out[0], out[1:]

        def fun_jac(phys):
            """Forward + backward: fractions + analytical gradient dicts."""
            R._cache.clear()          # the memo is per evaluation, not per mask
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

        hess_inv = getattr(self.fit_result, "hess_inv", None)

        def err(row):
            """``sqrt(row C row^T)``; no covariance -> 0, as always."""
            if hess_inv is None:
                return 0.0
            return float(np.sqrt(max(float(row @ hess_inv @ row), 0.0)))

        if not jac:
            # Values only: no backward pass is needed for a finite-difference
            # sweep, so this skips ``_compute_total_and_grad``'s chaining.
            # The sums are returned with the denominator LAST and divided
            # once: a ratio cannot be summed over chunks (Σ num/den per chunk
            # is not Σnum/Σden).
            def sums_on(handle, phys):
                p = R._phys_to_params(phys)
                return np.array([float(fitter.backend.compute(
                    dict(p, ck=R._ck_masked(m, p["ck"])), handle,
                    norm=None)[0]) for m in list(masks) + [denominator]])

            def obs(phys):
                R._cache.clear()
                if batch is None:
                    s = sums_on(R._get_handle(), phys)
                else:
                    # honour ``batch=``: one chunk resident at a time, same
                    # guarantee as the analytic route (at 2·n sweeps, so the
                    # loads multiply — that is the price of finite differences)
                    s = fitter.sum_stream(
                        lambda h: sums_on(h, phys), data=R._np, batch=batch)
                return s[:-1] / s[-1]

            # x basis: a fraction depends on every parameter, so stepping the
            # flat slots is both complete and the cheaper sweep.  (The named
            # basis of ``fd_jacobian`` is for observables with a small,
            # genuinely complete name set — see ``get_bw_params``.)
            vals, J = fitter.fd_jacobian(obs, fit_result=self.fit_result)
            return list(vals), np.array([err(row) for row in J])

        if batch is None:
            # Whole sample: the established covariance pipeline — analytic
            # grad dicts by resolved name, reduced to ``x`` by
            # ``cm.full_gradient`` inside ``cal_uncertainties_multi``.
            vals, cov = fitter.cal_uncertainties_multi(
                fun_jac, param_names, self.fit_result, jac=True)
            return vals, np.sqrt(np.diag(cov))

        # Streamed: ``fwd_of`` already chained each chunk's gradient to ``x``,
        # so the dual ratio is the observable and the uncertainty is its
        # quadratic form — the same diagonal the unbatched route computes,
        # without a second chaining.
        den, nums = streamed()
        ratios = [n / den for n in nums]

        # ``param_names`` does not restrict the covariance span on either
        # route: it names the parameters handed to the (analytic) callback,
        # and that callback's gradient dict already covers everything the
        # kernel produced.  The streamed route therefore takes the dual's full
        # x gradient, exactly like the unbatched one — masking it to
        # ``param_names`` would make the two disagree, because those names and
        # the kernel's own gradient keys are different vocabularies (12 free
        # slot names vs 22 resolved names, only 2 in common, on
        # ``config_pwa_gen_mb``).
        # An empty/unknown ``param_names`` means "no parameter to vary" on the
        # unbatched route (it returns a zero covariance), so the streamed one
        # must say zero too rather than the full error.
        no_names = not any(n in self._resolved for n in param_names)

        def sigma(r):
            if hess_inv is None or no_names:
                return 0.0
            g = r.grad
            return float(np.sqrt(max(float(g @ hess_inv @ g), 0.0)))

        return [r.value for r in ratios], np.array([sigma(r) for r in ratios])
