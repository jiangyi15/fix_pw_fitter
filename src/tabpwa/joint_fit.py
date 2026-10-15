"""JointFitter — simultaneous fit over several configs sharing ONE ``cm``.

The point of a simultaneous fit is a *single* parameter space: two configs
that both speak about ``rho770_mass`` must optimize one variable, not two.
Instead of merging constraint managers after the fact, the shared
:class:`~tabpwa.param_constraint.ConstraintManager` is defined here, once,
and injected into every sub-:class:`~tabpwa.fitter.Fitter`; each model
registers its names into it, so a name present in two configs lands in the
same slot of the flat ``x`` by construction.  No slicing, no aliasing
layer, no per-model remapping:

* ``get_nll(x)``   = Σ_i ``Fitter_i.get_nll(x)`` (each returns a flat
  gradient in the *shared* registry; names another model does not own are
  simply zero there), so the sum is index-aligned.
* ``fit()``        = the same BFGS loop as ``Fitter.fit`` on that sum.
* ``values_from_dict`` / ``load_fixed_from_dict`` work on the union.

Auto-deal rules for names claimed by more than one config (deterministic,
no prompts — the *latest* registration wins where a choice is needed):

========================  ============================================
fixed values              merged; a later value replaces (WARN)
fixed vs free             union: fixed by ANY config stays fixed
                          (unfreeze explicitly with ``free_var``)
bounds                    intersected across configs (WARN if changed)
scale transforms          per-name replace, latest wins (cm-level)
mass/width transforms     per-name replace, latest wins (cm-level, WARN)
``var_equal`` / same      union-merged alias groups
defaults                  merged, latest wins
blind                     ONE frame — call :meth:`set_blind` here only
========================  ============================================

Usage::

    from tabpwa.joint_fit import JointFitter

    joint = JointFitter(["channelA.yml", "channelB.yml"],
                        backend="cuda_v5_pwa_s0c")
    x0 = joint.initial_values(seed=1)
    print(joint.get_nll(x0)[0])          # Σ over both datasets
    result = joint.fit(x0, maxiter=2000)

    # per-config backends / lazy building also work:
    joint = JointFitter([("channelA.yml", "cuda_v5_pwa_s0c"),
                         {"config": "channelB.yml", "backend": "cpu_v3"}])
    # or: joint = JointFitter(); joint.add_config("channelA.yml", backend=…)

Same model, several datasets — expand the config variants *outside*
(`tabpwa.config_broadcast.ConfigBroadcast`, one diff per variant) and add
each one; they share the same model/cm and therefore one x-space::

    from tabpwa.config_broadcast import ConfigBroadcast

    joint = JointFitter()
    for label, cfg in ConfigBroadcast.items("multi.yml"):
        joint.add_config(cfg, backend="cuda_v5_pwa_s0c", label=label)

where ``multi.yml`` carries

.. code-block:: yaml

    data:
      dat_order: [pip, pim, eta]
      phsp_arr: shared_phsp.npz
    broadcast:
      - {data: {data_arr: run1.npz}}
      - {data: {data_arr: run2.npz}, purity: 0.93, label: run2}
"""
import os
import time
import warnings

import numpy as np


class JointFitter:
    """Simultaneous fit over N configs sharing one constraint manager."""

    def __init__(self, configs=None, backend=None, load_data=True, cm=None):
        """Build the joint fit.

        Args:
            configs: a config path / config dict, or a list of them.  Each
                entry may also be ``(config, backend)`` for a per-config
                backend, or a spec dict
                ``{"config": …, "backend": …, "load_data": …}``.  ``None``
                starts empty (use :meth:`add_config` / :meth:`add_fitter`).
                Config *variants* are expanded outside (see
                :mod:`tabpwa.config_broadcast`).
            backend: backend spec applied to every config that does not
                carry its own (see :class:`~tabpwa.fitter.Fitter`).
            load_data: call each fitter's ``load_all_data()`` (the
                config's ``data:`` section) while adding it.
            cm: optional pre-made shared
                :class:`~tabpwa.param_constraint.ConstraintManager`.
        """
        from tabpwa.param_constraint import ConstraintManager
        self.cm = cm if cm is not None else ConstraintManager([])
        self.fitters = []
        self._last_xk = None
        self.log = []
        # names whose mass/width transforms were replaced while adding
        # configs (SameTopoJointFitter turns this into a strict error)
        self._replacements = []

        if configs is None:
            items = []
        elif isinstance(configs, (str, dict)):
            items = [configs]
        else:
            items = list(configs)

        for item in items:
            be, ld = backend, load_data
            spec = item
            if (isinstance(item, dict) and "config" in item
                    and "decay" not in item):
                spec = item["config"]
                be = item.get("backend", backend)
                ld = item.get("load_data", load_data)
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                spec, be = item
            self.add_config(spec, backend=be, load_data=ld)

    # ── build ─────────────────────────────────────────────────────

    def _warn(self, msg):
        line = f"joint: {msg}"
        self.log.append(line)
        warnings.warn(line, RuntimeWarning, stacklevel=3)

    def add_config(self, config_file, backend=None, load_data=True,
                   label=None):
        """Build one sub-Fitter against the shared cm and add it.

        ``config_file`` is a single config (path or dict) — config
        *variants* stay outside the JointFitter: expand them at the call
        site and add each result here, e.g.::

            from tabpwa.config_broadcast import ConfigBroadcast
            for lbl, cfg in ConfigBroadcast.items("multi.yml"):
                joint.add_config(cfg, backend="cuda_v5_pwa_s0c", label=lbl)

        ``label`` is an optional name for logs (``f._joint_label``); by
        default it is ``"<i>:<basename>"``.  ``load_data=True`` calls the
        fitter's own ``load_all_data()`` (the config's ``data:`` section).
        Returns the new sub-Fitter.
        """
        return self._add_single(config_file, backend=backend,
                                load_data=load_data, label=label)

    def _add_single(self, config_file, backend=None, load_data=True,
                    label=None):
        from tabpwa.fitter import Fitter

        prev_fixed = dict(self.cm.fixed_slots)
        prev_bounds = self.cm.bounds.to_dict()

        f = Fitter(config_file, backend=backend, cm=self.cm)
        f.apply_constrains()
        if load_data:
            f.load_all_data()
        self._replacements.extend(f.setup_mass_width_transforms() or [])

        # ── auto-deal reports ─────────────────────────────────────
        for name, val in self.cm.fixed_slots.items():
            if name in prev_fixed and prev_fixed[name] != val:
                self._warn(
                    f"shared fixed value of {name!r} replaced: "
                    f"{prev_fixed[name]:.6g} -> {val:.6g} (latest wins)")
        for name in prev_fixed:
            if name not in self.cm.fixed_slots:
                self._warn(
                    f"shared fixed {name!r} was RELEASED by this config "
                    f"(no longer fixed in the joint cm)")

        now_bounds = self.cm.bounds.to_dict()
        for name, spec in prev_bounds.items():
            if name not in now_bounds:
                continue
            lo1 = float(now_bounds[name]["low"])
            hi1 = float(now_bounds[name]["high"])
            lo = max(float(spec["low"]), lo1)
            hi = min(float(spec["high"]), hi1)
            if (lo, hi) == (lo1, hi1):
                continue
            if lo >= hi:
                self._warn(
                    f"bounds of shared {name!r} are disjoint "
                    f"([{spec['low']:.4g}, {spec['high']:.4g}] vs "
                    f"[{lo1:.4g}, {hi1:.4g}]); keeping the latest window")
            else:
                self.cm.set_range(name, lo, hi)
                self._warn(
                    f"bounds of shared {name!r} intersected to "
                    f"[{lo:.4g}, {hi:.4g}]")

        f._joint_label = (label if label is not None
                          else self._label_for(config_file, len(self.fitters)))
        self.fitters.append(f)
        return f

    @staticmethod
    def _label_for(config_file, i):
        if isinstance(config_file, dict):
            base = "cfg"
        else:
            base = os.path.basename(str(config_file)) or "cfg"
        return f"{i}:{base}"

    def add_fitter(self, fitter):
        """Add an already-built Fitter that was constructed with ``cm=self.cm``."""
        if fitter.cm is not self.cm:
            raise ValueError(
                "JointFitter.add_fitter needs a Fitter built with "
                "cm=joint.cm (use joint.add_config(...) to build one)")
        self.fitters.append(fitter)
        return fitter

    # ── objective ─────────────────────────────────────────────────

    @property
    def n_models(self):
        return len(self.fitters)

    @property
    def param_names(self):
        """Slot-level names of the shared free-variable vector."""
        return self.cm.var_registry.flat_names

    def free_param_names(self):
        return self.cm.free_param_names()

    def get_nll(self, x):
        """Σ_i NLL_i at the shared *x*; gradient is the index-aligned sum.

        Every sub-Fitter receives the same shared vector: it resolves the
        names it owns and its ``_kernel_builder`` picks them out of the
        resolved dict, so parameters belonging to another model are simply
        not consumed by it.  Returns ``(nll_total, grad_total)`` in the
        shared optimizer space.
        """
        x = np.asarray(x, dtype=float).ravel()
        self._last_xk = x.copy()
        nll = 0.0
        grad = None
        for f in self.fitters:
            q, g = f.get_nll(x)
            nll += q
            grad = np.asarray(g, dtype=np.float64) if grad is None \
                else grad + np.asarray(g, dtype=np.float64)
        return nll, grad

    def get_nll_by_model(self, x):
        """Per-model NLL values at shared *x* (diagnostics)."""
        return [f.get_nll(x)[0] for f in self.fitters]

    # ── start values / (de)serialisation ──────────────────────────

    def initial_values(self, seed=None, use_default=True):
        """Starting vector over the shared free variables."""
        return self.cm.initial_values(seed=seed, use_default=use_default)

    reinitial = initial_values

    def values_dict(self, x):
        """Resolved values in the blinded report frame (name-keyed)."""
        return self.cm.blind_result(
            self.cm.flat_resolve(np.asarray(x, dtype=float).ravel()))

    def values_from_dict(self, data):
        """Flat x from a saved name→value dict (union over all configs).

        Same convention as :meth:`Fitter.values_from_dict`: the dict is in
        the blinded report frame, unblinded here, then pushed through the
        inverse constraint pipeline.  Names not present fall back to the
        deterministic reinit.
        """
        names = self.cm.var_registry.flat_names
        values = data.get("value", data) if isinstance(data, dict) else data
        x = self.reinitial()
        phys = {name: float(values[name]) for name in names if name in values}
        if phys:
            raw = self.cm.inverse(self.cm.unblind_result(phys))
            for i, name in enumerate(names):
                if name in raw:
                    x[i] = float(raw[name])
        return x

    def load_fixed_from_dict(self, data):
        """Set shared fixed values from a saved dict (see Fitter equivalent)."""
        values = data.get("value", data) if isinstance(data, dict) else data
        if not values:
            return
        before_fixed = self.cm.inverse(
            self.cm.unblind_result(values), stop_before='fixed')
        changed = False
        for name in list(self.cm.fixed_slots):
            if name in before_fixed:
                self.cm.fixed_tr.values[name] = float(before_fixed[name])
                changed = True
        if changed:
            self.cm._rebuild()

    def set_blind(self, names, seed, scale=1.0):
        """Blind shared parameters — ONE frame for the whole joint fit.

        Unlike the per-config path this replaces any previous blind state,
        so configs cannot end up in different report frames.
        """
        self.cm.set_blind(names, seed, scale=scale, reset=True)

    # ── optimisation ──────────────────────────────────────────────

    def fit(self, x0=None, maxiter=1000, ftol=1e-5, gtol=1e-3, callback=None,
            method='BFGS', disp=True, options=None, **kwargs):
        """Minimize the summed NLL with the same BFGS loop as ``Fitter.fit``.

        Returns a :class:`~tabpwa.fit_result.FitResult` (BFGS →
        ``hess_inv`` over the shared vector, uncertainties =
        ``sqrt(diag(hess_inv))``).
        """
        from scipy.optimize import minimize
        from .fit_result import FitResult

        if not self.fitters:
            raise RuntimeError("JointFitter has no models (add_config first)")
        if x0 is None:
            x0 = self.initial_values()

        _last = {'nll': None, 'grad': None}

        def nll_and_grad(x):
            nll, grad = self.get_nll(x)
            _last['nll'] = nll
            _last['grad'] = grad
            return nll, grad

        class IterTracker:
            def __init__(self):
                self.n = 0
                self.t_start = time.time()
                self.t_last = time.time()

            def __call__(self, xk):
                self.n += 1
                t_now = time.time()
                dt = t_now - self.t_last
                self.t_last = t_now
                gn = np.linalg.norm(_last['grad'])
                print(f"  iter {self.n:4d}: NLL = {_last['nll']:.11f}, "
                      f"|grad| = {gn:.4e}, +{dt:.2f}s "
                      f"[{t_now - self.t_start:.1f}s]")
                return False

        tracker = IterTracker()
        combined_cb = tracker if callback is None else (
            lambda xk: (tracker(xk), callback(xk)))

        opts = {'maxiter': maxiter, 'gtol': gtol, 'disp': disp}
        opts = {k: v for k, v in opts.items() if v is not None}
        if method in ('L-BFGS-B',):
            opts['ftol'] = ftol
        if options:
            opts.update(options)
        result = minimize(nll_and_grad, np.asarray(x0, dtype=float), jac=True,
                          method=method, options=opts, callback=combined_cb,
                          **kwargs)
        return FitResult.from_result(result)

    # ── lifecycle ─────────────────────────────────────────────────

    def free(self):
        """Free every sub-Fitter's backend/data (shared cm is untouched)."""
        for f in self.fitters:
            try:
                f.backend.free()
            except Exception:
                pass
        self.fitters = []

    def __repr__(self):
        labels = ", ".join(getattr(f, "_joint_label", "?") for f in self.fitters)
        return (f"JointFitter(n_models={self.n_models}, "
                f"n_free={len(self.free_param_names())}, [{labels}])")


class SameTopoJointFitter(JointFitter):
    """Joint fit over one template whose variants share the decay topology.

    The template's ``broadcast:`` list defines the variants; **only the
    ``decay`` section** must be identical across them (the particle level
    is free — different resonance content, ``dat_order``, constraints and
    backends are all allowed).  The shared decay guarantees identical
    observables, which is what makes stacked/grid plots and per-dataset
    reports well-defined; parameters bind wherever their names coincide
    (from one template that is everything).

    Usage::

        joint = SameTopoJointFitter("template.yml",
                                    backend="cuda_v5_pwa_s0c")
        res = joint.fit()
        print(joint.nll_report(res.x))
        joint.stack_plotter(res, output="plots/").plot_common(stacked=True)
    """

    def __init__(self, template, backend=None, load_data=True, cm=None,
                 strict=True, forbidden=("decay",)):
        """Args:
            template: config path or dict carrying ``broadcast:`` (or not:
                then this is a one-config joint fit).
            backend, load_data, cm: as in :class:`JointFitter`.
            strict: verify the decay equality across variants (raise on
                mismatch with a hint to use the plain ``JointFitter``).
            forbidden: top-level keys the template's ``broadcast:`` diffs
                must not touch (passed to
                :class:`~tabpwa.config_broadcast.ConfigBroadcast`).
        """
        from tabpwa.config_broadcast import ConfigBroadcast

        super().__init__(cm=cm)
        self.template = template
        self.forbidden = tuple(forbidden or ())
        cb = ConfigBroadcast(template, forbidden=self.forbidden)
        if strict:
            self._check_same_decay(cb)
        for label, cfg in ConfigBroadcast.items(cb):
            self.add_config(cfg, backend=backend, load_data=load_data,
                            label=label or None)
        if self._replacements:
            names = sorted(set(self._replacements))
            msg = (f"SameTopoJointFitter: variants replace mass/width "
                   f"transforms for shared parameter(s) {names} — the "
                   f"variants differ in particle model, so the same "
                   f"parameter would follow different physics per dataset. "
                   f"Align the configs, or pass strict=False to accept the "
                   f"latest-wins merge.")
            if strict:
                raise ValueError(msg)
            self._warn(msg)

    @staticmethod
    def _check_same_decay(cb):
        """Safety net: variant ``decay`` sections must be deep-equal.

        Normally guaranteed by :class:`ConfigBroadcast`, which rejects
        broadcast diffs that touch ``decay`` — this re-check exists so a
        future code path cannot silently bypass the same-topo guarantee.
        """
        variants = cb.expand()
        if len(variants) < 2:
            return
        ref = variants[0].get("decay")
        for i, variant in enumerate(variants[1:], start=1):
            if variant.get("decay") != ref:
                raise ValueError(
                    f"SameTopoJointFitter: variant {i} ({cb.labels()[i]}) "
                    f"has a different 'decay' section than variant 0 — "
                    f"the variants must share the decay topology.  Use the "
                    f"plain JointFitter to combine different topologies.")

    # ── same-topo extras ──────────────────────────────────────────

    def dataset_labels(self):
        """Per-sub-fitter labels (order matches ``self.fitters``)."""
        return [getattr(f, "_joint_label", f"cfg{i}")
                for i, f in enumerate(self.fitters)]

    def nll_report(self, x=None):
        """Per-dataset NLL breakdown ``{label: value}`` at *x*.

        ``x=None`` uses the last evaluated vector (e.g. the fit result);
        raises when neither is available.
        """
        if x is None:
            x = self._last_xk
        if x is None:
            raise ValueError("nll_report needs x (or a prior evaluation)")
        return {lbl: float(v) for lbl, v in
                zip(self.dataset_labels(), self.get_nll_by_model(x))}

    def wave_report(self):
        """Per-variant wave counts (particle content may differ)."""
        return {lbl: len(f.all_comb)
                for lbl, f in zip(self.dataset_labels(), self.fitters)}

    def stack_plotter(self, result=None, groups=None, output="plots/"):
        """A :class:`~tabpwa.plot_pw_stack.PWStackPlotter` over the variants.

        ``result`` supplies the shared ``x`` (a ``FitResult`` or a flat
        vector); ``None`` uses the last evaluated one.  ``groups`` is a
        single dict for all variants or one per variant (``None`` → no
        group curves, totals only).
        """
        from tabpwa.plot_pw_stack import PWStackPlotter

        x = result.x if hasattr(result, "x") else result
        if x is None:
            x = self._last_xk
        if x is None:
            raise ValueError("stack_plotter needs a result/x (or a prior "
                             "evaluation)")
        if isinstance(groups, list):
            per = groups
        else:
            per = [groups if groups is not None else {} for _ in self.fitters]
        entries = [(f, x, g) for f, g in zip(self.fitters, per)]
        return PWStackPlotter(entries, labels=self.dataset_labels(),
                              output=output)
