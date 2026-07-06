"""
Independent pipeline stages for parameter constraints.

Each stage is a small object with ``forward(input_dict)`` (transform)
and ``backward(grad_dict)`` (gradient backpropagation):

    raw = registry.to_dict(x)
    d   = name_res.forward(raw)       # alias → canonical (same)
    d   = fixed_tr.forward(d)         # inject constant values
    for scale in scale_transforms:    # multiply by scale factors
        d = scale.forward(d)
    ck  = pc.build_ck(d)              # partial-wave amplitudes
    ...
    grad = pc.backprop_grad(d, grad_ck)
    for scale in reversed(scale_transforms):
        grad = scale.backward(grad)
    grad = fixed_tr.backward(grad, d)
    grad = name_res.backward(grad, raw)
    flat = registry.flat_gradient(x, grad)
"""

import numpy as np


class ParameterConstraint:
    """Pure combinatorics: named values → partial-wave amplitudes (ck).

    ::

        ck[i] = ∏_{term ∈ comb[i]} term_value
    """

    def __init__(self, all_comb):
        self.all_comb = list(all_comb)
        self.n_wave = len(all_comb)

    def _get_val(self, slot_dict, name):
        """Get the complex value for a term from slot-level dict.
        
        If the dict has ``name + 'r'`` / ``name + 'i'`` slots, treat
        as a complex ck parameter: ``r·exp(j·θ)``.
        Otherwise use ``name`` directly (real/m0/g0/scalar).
        
        NOTE: coupling parameters not in slot_dict default to 0+0j.
        """
        if name + 'r' in slot_dict:
            r = slot_dict[name + 'r']
            theta = slot_dict.get(name + 'i', 0.0)
            return r * np.exp(1j * theta)
        return slot_dict.get(name, 0.0 + 0.0j)

    def build_ck(self, slot_dict):
        """Compute ck from slot-level real dict.

        Slot dict has ``{name_r: mag, name_i: phase}`` for complex
        ck parameters and ``{name: value}`` for real parameters.
        """
        ck = np.empty(self.n_wave, dtype=complex)
        for i, comb in enumerate(self.all_comb):
            prod = 1.0 + 0.0j
            for term in comb:
                if isinstance(term, str):
                    prod *= self._get_val(slot_dict, term)
                else:
                    prod *= term
            ck[i] = prod
        return ck

    def free_param_names(self):
        """All canonical parameter names from the comb structure (compat)."""
        names = set()
        for comb in self.all_comb:
            for p in comb:
                if isinstance(p, str):
                    names.add(p)
        return sorted(names)

    def backprop_grad(self, slot_dict, grad_ck):
        """Gradient w.r.t. each slot from kernel's ``grad_ck``.

        Returns ``{name_r: dQ/dr, name_i: dQ/dθ}`` for complex ck
        params, ``{name: dQ/dval}`` for real params.

        Uses Wirtinger calculus: for a real-valued Q, the chain rule
        through ``ck = r·exp(j·θ)`` gives::

            dQ/dr = 2 · Re( Σ grad_ck · order · ck / r )
            dQ/dθ = 2 · Re( Σ grad_ck · j · order · ck )
        """
        ck = self.build_ck(slot_dict)
        grads = {}
        for i, comb in enumerate(self.all_comb):
            counts = {}
            for term in comb:
                if isinstance(term, str):
                    counts[term] = counts.get(term, 0) + 1
            for name, order in counts.items():
                dck = order * ck[i]
                if name + 'r' in slot_dict:
                    r = slot_dict[name + 'r']
                    grads[name + 'r'] = grads.get(name + 'r', 0.0) + 2.0 * np.real(grad_ck[i] * dck / r)
                    grads[name + 'i'] = grads.get(name + 'i', 0.0) + 2.0 * np.real(grad_ck[i] * dck * 1j)
                else:
                    v = slot_dict.get(name, 1.0)
                    grads[name] = grads.get(name, 0.0) + 2.0 * np.real(grad_ck[i] * dck / v)
        return grads


# ================================================================
# VariableRegistry — flat vector ↔ named dict
# ================================================================

class VariableRegistry:
    """Maps named variables to the flat real vector ``x``.

    Every entry represents one slot (no complex/real distinction).
    Complex coupling parameters occupy two slots (``name_r``, ``name_i``)
    registered as separate entries.
    """

    def __init__(self):
        self._entries = []
        self._name_to_entry = {}

    def add(self, name):
        self._entries.append(name)
        self._name_to_entry[name] = name

    def add_real(self, name):
        """Convenience: real-valued parameter (single slot)."""
        self.add(name)

    def add_complex(self, base):
        """Convenience: complex parameter → two slots ``base + 'r'``, ``base + 'i'``."""
        self.add(base + 'r')
        self.add(base + 'i')

    @property
    def names(self):
        return list(self._name_to_entry.keys())

    @property
    def flat_names(self):
        return list(self._entries)

    @property
    def n_flat(self):
        return len(self._entries)

    def flat_index(self, name):
        for i, n in enumerate(self._entries):
            if n == name:
                return (i, i + 1)
        raise KeyError(f"Unknown variable: {name}")

    def build_initial(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        x = np.empty(self.n_flat)
        for i in range(self.n_flat):
            x[i] = np.random.uniform(-0.5, 0.5)
        return x

    def build_initial_deterministic(self):
        return np.zeros(self.n_flat)

    # ── forward: flat x → {slot_name: real_value} ──────────────

    def to_dict(self, x):
        return dict(zip(self._entries, x))

    # ── backward: {slot_name: grad} → flat gradient ────────────

    def flat_gradient(self, x, grad_dict):
        flat = np.zeros(self.n_flat)
        for i, n in enumerate(self._entries):
            flat[i] = grad_dict.get(n, 0.0)
        return flat

    # ── compat shims ────────────────────────────────────────────

    def extract_real_dict(self, x):
        return self.to_dict(x)

    def extract_by_target(self, x, target_type):
        return np.array([])

    def backprop_grad(self, x, grad_dict):
        return self.flat_gradient(x, grad_dict)


# ================================================================
# NameResolution — alias→canonical (same-parameter groups)
# ================================================================

class NameResolution:
    """Resolves alias names to canonical names.

    Forward: ``d[canon] = raw[alias]`` (last alias wins for each canon).
    Backward: ``grad[alias] = grad_out[canon]``.

    Keys are exact slot names — no suffix manipulation.
    """

    def __init__(self):
        self.map = {}          # {alias: canonical}

    def set_same(self, groups):
        self.map = {}
        for group in groups:
            if group:
                canon = group[0]
                for alias in group[1:]:
                    self.map[alias] = canon

    def _resolve_slot(self, key):
        return self.map.get(key, key)

    def apply(self, d):
        result = {}
        for k, v in d.items():
            resolved = self._resolve_slot(k)
            result[resolved] = v
        # Inject alias slot names too
        for alias, canon in self.map.items():
            if canon in result:
                result[alias] = result[canon]
        return result

    def inverse(self, d):
        """Reverse of :meth:`apply`: no-op (all keys already present)."""
        return dict(d)

    def chain_grad(self, grad_out, d_in):
        """Reverse of :meth:`apply` — maps resolved grads back to raw keys."""
        result = {}
        for key in d_in:
            resolved = self._resolve_slot(key)
            g = grad_out.get(resolved, 0.0)
            # Also accumulate contributions from aliases that map here
            for alias, canon in self.map.items():
                if canon == resolved and alias in grad_out:
                    g += grad_out[alias]
            result[key] = g
        return result


# ================================================================
# Transform — general parameter transform base class
# ================================================================

class Transform:
    """General base class for parameter-space transforms.

    A ``Transform`` maps from a set of input parameters to a set of
    output parameters, with differentiable forward/backward passes.

    ── ``forward(input_dict) → output_dict``
    ── ``backward(grad_output, d_input=None) → grad_input``
    ── ``inverse(output_dict) → input_dict``  (optional — set
        ``has_inverse = False`` on subclasses that don't support it)

    Parameters
    ----------
    input_names : list of str, optional
        Names of input parameters (used for registry introspection).
    output_names : list of str, optional
        Names of output parameters.
    """

    _has_inverse = True

    def __init__(self, input_names=None, output_names=None):
        self._input_names = list(input_names) if input_names is not None else None
        self._output_names = list(output_names) if output_names is not None else None

    @property
    def input_names(self):
        return list(self._input_names) if self._input_names is not None else []

    @property
    def output_names(self):
        return list(self._output_names) if self._output_names is not None else []

    @property
    def has_inverse(self):
        """Whether this transform supports ``inverse()``."""
        return self._has_inverse

    def forward(self, d):
        """Transform input dict → output dict.

        Should only modify entries in ``output_names``.  Other entries
        are passed through unchanged.
        """
        raise NotImplementedError

    def apply_forward(self, d):
        """Apply :meth:`forward` and return the updated dict.

        Only entries in ``output_names`` may change; values for other keys
        are preserved from the input *d*.  Only ``input_names`` entries are
        passed to :meth:`forward` — transforms should not read values they
        didn't declare as inputs.
        """
        saved = {k: v for k, v in d.items() if k not in self.output_names}
        inputs = {k: v for k, v in d.items() if k in self.input_names}
        result = self.forward(inputs)
        for k, v in saved.items():
            result[k] = v
        return result

    def backward(self, grad_out, d_in=None):
        """Backpropagate gradient from output space to input space.

        Subclasses implement this.  Returns a dict mapping **only**
        ``input_names`` to their gradients.

        Args
        ----
        grad_out : dict
            Gradients w.r.t. output parameters ``{name: value}``.
        d_in : dict or None
            Input dict passed to forward (may be needed for chain rule).

        Returns
        -------
        dict
            Gradients w.r.t. input parameters (only ``input_names`` keys).
        """
        raise NotImplementedError

    def apply_backward(self, grad_out, d_in=None):
        """Apply backward and merge into *grad_out*.

        Rules
        -----
        * Input **and** output: **replace** — the transform's backward
          gives the complete derivative for parameters it both reads
          and writes (e.g. pass-through like ``re_00``).
        * Input **only**: **accumulate** — adds the gradient through the
          transform's gamma path to any existing kernel gradient.
        * Output **only**: **remove** — derived quantities that should
          not be in the optimizer space.
        """
        back = self.backward(grad_out, d_in=d_in)
        for name in self.input_names:
            if name in back:
                if name in self.output_names:
                    grad_out[name] = back[name]          # replace
                else:
                    grad_out[name] = grad_out.get(name, 0.0) + back[name]  # accumulate
        for name in self.output_names:
            if name not in self.input_names:
                grad_out.pop(name, None)
        return grad_out

    def inverse(self, d):
        """Reverse of :meth:`forward`: output dict → input dict.

        Should only modify entries in ``input_names``.
        Raise :class:`NotImplementedError` if unsupported.
        """
        raise NotImplementedError

    def apply_inverse(self, d):
        """Apply :meth:`inverse` and return the updated dict.

        Only entries in ``input_names`` may change; values for other keys
        are preserved from the input *d*.
        """
        saved = {k: v for k, v in d.items() if k not in self.input_names}
        try:
            result = self.inverse(d)
        except NotImplementedError:
            return d
        for k, v in saved.items():
            result[k] = v
        return result


# ================================================================
# LinearTransform — scale + bias a single parameter
# ================================================================

class LinearTransform(Transform):
    """Apply a linear transform to a single named parameter.

    ``forward``:  ``d[name] = factor * d[name] + bias``
    ``backward``: ``grad[name] *= factor``  (bias doesn't affect gradient)
    ``inverse``:  ``d[name] = (d[name] - bias) / factor``

    Input and output share the same parameter name; only the value
    is transformed.  Multiple independent ``LinearTransform`` instances
    are composed as a list in :class:`ConstraintManager`.
    """

    def __init__(self, name, factor, bias=0.0):
        super().__init__(input_names=[name], output_names=[name])
        self.name = name
        self.factor = float(factor)
        self.bias = float(bias)

    def forward(self, d):
        d = dict(d)
        if self.name in d:
            d[self.name] = self.factor * d[self.name] + self.bias
        return d

    def backward(self, grad_out, d_in=None):
        grad = dict(grad_out)
        if self.name in grad:
            grad[self.name] = grad[self.name] * self.factor
        return grad

    def inverse(self, d):
        d = dict(d)
        if self.name in d and self.factor != 0:
            d[self.name] = (d[self.name] - self.bias) / self.factor
        return d


# Backward-compatible alias
ScaleTransform = LinearTransform


# ================================================================
# FixedOverride — inject constant parameter values
# ================================================================

class FixedOverride:
    """Injects fixed (constant) values into the slot-level param dict.

    Works with slot-level names (``name_r``, ``name_i`` for complex ck
    params, ``name`` for real params).  Only overrides keys that exist
    in the dict — fully-fixed ck params are already excluded from the
    registry, so their slot names won't appear.
    """

    def __init__(self):
        self.values = {}       # {slot_name: value}

    def set_fixed(self, fixed):
        self.values = dict(fixed)

    def apply(self, d):
        d = dict(d)
        for name, val in self.values.items():
            d[name] = val  # inject fixed value even if not in raw dict
        return d

    def inverse(self, d):
        """Reverse of :meth:`apply`: no-op (forward re-injects fixed values)."""
        return dict(d)

    def chain_grad(self, grad_out, d_in):
        grad = dict(grad_out)
        for name in self.values:
            grad.pop(name, None)
        return grad


# ================================================================
# Bound constraint helper (re-export)
# ================================================================
from ampfit.boundary import BoundTransform  # noqa: F401


# ================================================================
# ConstraintManager — coordinates the pipeline stages
# ================================================================

# Scalar parameter names (time evolution, production — not per-particle).
# Defined at module level so external code (Fitter, backends) can
# reference them without coupling to ConstraintManager internals.
SCALAR_NAMES = ["gamma", "delta_gamma", "delta_m", "A_prod", "poqr", "poqi"]


class ConstraintManager:
    """Owns :class:`ParameterConstraint`, :class:`VariableRegistry`,
    and the constraint pipeline stages (:class:`NameResolution`,
    :class:`ScaleTransform`, :class:`FixedOverride`).

    Every ``set_*`` method updates the relevant stage and rebuilds
    the registry — trivial for ~200 parameters.

    Parameter names are passed as a flat ``all_names`` list — no
    type distinction (ck / m0 / g0 / scalar separation only exists
    in :meth:`Fitter._build_params`).
    """

    def __init__(self, all_comb, all_names):
        self.all_comb = list(all_comb)
        self._all_names = list(all_names)

        # Pipeline stages (independent objects)
        self.pc = ParameterConstraint(all_comb)
        self.name_res = NameResolution()
        self.scale_transforms = []    # list of ScaleTransform (applied in order)
        self.mass_width_transforms = []  # list of Transform from particle models
        self.fixed_tr = FixedOverride()

        # Bound transforms (flat-index → BoundTransform)
        self.bounds = {}

        # Variable registry (rebuilt on constraint change)
        self._var_registry = None

        self._rebuild()

    # ── public read access ──────────────────────────────────────

    @property
    def var_registry(self):
        return self._var_registry

    @property
    def alias_to_canon(self):
        return self.name_res.map

    @property
    def bound_transforms(self):
        return self.bounds

    # Backward-compat property aliases
    @property
    def fixed_slots(self):
        return self.fixed_tr.values

    @property
    def same_params(self):
        return self.name_res.map

    @property
    def scale_params(self):
        return {tr.name: tr.factor for tr in self.scale_transforms}

    def free_param_names(self):
        return [n for n in self.var_registry.flat_names
                if n not in self.fixed_tr.values]

    def initial_values(self, seed=None):
        return self.var_registry.build_initial(seed=seed)

    # ── modifiers ──────────────────────────────────────────────

    def set_fixed(self, fixed_slots, reset=False):
        if reset:
            self.fixed_tr.values = {}
        self.fixed_tr.values.update({k: float(v) for k, v in fixed_slots.items()})
        self._rebuild()

    def set_same(self, same_params, reset=False):
        if reset:
            self.name_res.map = {}
        for group in same_params:
            if group:
                canon = group[0]
                for alias in group[1:]:
                    self.name_res.map[alias] = canon
        self._rebuild()

    def set_scale(self, scale_params, reset=False):
        if reset:
            self.scale_transforms.clear()
        for name, val in scale_params.items():
            if isinstance(val, (list, tuple)):
                factor, bias = val[0], val[1] if len(val) > 1 else 0.0
            else:
                factor, bias = val, 0.0
            self.scale_transforms.append(LinearTransform(name, factor, bias))
        self._rebuild()

    def set_mass_width_transforms(self, transforms, reset=True):
        if reset:
            self.mass_width_transforms.clear()
            # Don't clear _all_names — it also holds m0/g0/scalar names.
            # Only remove names that were added by a previous call.
            self._all_names = [n for n in self._all_names
                               if not any(n in tr.input_names
                                          for tr in self.mass_width_transforms)]
        for tr in transforms:
            if tr is not None:
                self.mass_width_transforms.append(tr)
                input_set = set(tr.input_names)
                for name in tr.input_names:
                    if name not in self._all_names:
                        self._all_names.append(name)
                for name in tr.output_names:
                    if name not in input_set and name in self._all_names:
                        self._all_names.remove(name)
        self._rebuild()

    def set_free(self, name):
        self.fixed_tr.values.pop(name, None)
        # Remove from same groups
        self.name_res.map = {k: v for k, v in self.name_res.map.items()
                             if k != name and v != name}
        self.scale_transforms = [tr for tr in self.scale_transforms if tr.name != name]
        self._rebuild()

    def set_range(self, name, lo, hi):
        from ampfit.boundary import BoundTransform as _BT
        bt = _BT(lo, hi)
        # Resolve alias to canon (same-constraint)
        name = self.name_res.map.get(name, name)
        try:
            si, ei = self.var_registry.flat_index(name)
            for idx in range(si, ei):
                self.bounds[idx] = bt
        except KeyError:
            for i, n in enumerate(self.var_registry.flat_names):
                if n == name:
                    self.bounds[i] = bt
                    return
            raise ValueError(
                f"Unknown parameter '{name}'. "
                f"Available: {self.var_registry.flat_names[:6]}...")

    def unset_range(self, name):
        name = self.name_res.map.get(name, name)
        try:
            si, ei = self.var_registry.flat_index(name)
            for idx in range(si, ei):
                self.bounds.pop(idx, None)
        except KeyError:
            for i, n in enumerate(self.var_registry.flat_names):
                if n == name:
                    self.bounds.pop(i, None)

    # ── forward pipeline ───────────────────────────────────────

    def resolve(self, raw_dict):
        """Run the full constraint pipeline: same → fixed → scale → mass/width.

        Fixed values are the physical param values; scale is a model factor
        that multiplies ALL params (even fixed ones), matching TFPWA convention
        where scale is applied at the amplitude product level.

        Mass/width transforms (from particle models with running widths) are
        applied last to build derived mass/width values from physical params.
        """
        d = self.name_res.apply(raw_dict)
        d = self.fixed_tr.apply(d)
        for tr in self.scale_transforms:
            d = tr.apply_forward(d)
        for tr in self.mass_width_transforms:
            d = tr.apply_forward(d)
        return d

    def inverse(self, resolved):
        """Reverse of :meth:`resolve`: scale⁻¹ → fixed⁻¹ → same⁻¹.

        Converts physical (post-constraint) values back to raw
        (pre-constraint) values.  Used by ``values_from_dict``.
        """
        d = resolved
        for tr in reversed(self.mass_width_transforms):
            d = tr.apply_inverse(d)
        for tr in reversed(self.scale_transforms):
            d = tr.apply_inverse(d)
        d = self.fixed_tr.inverse(d)
        d = self.name_res.inverse(d)
        return d

    # ── backward pipeline ──────────────────────────────────────

    def chain_gradient(self, grad_resolved, resolved, raw):
        """Reverse of :meth:`resolve` (mass/width → scale → fixed → same).

        Uses :meth:`Transform.apply_backward` on each stage — only
        ``input_names`` are updated; output-only and unrelated gradients
        pass through unchanged.
        """
        grad = grad_resolved
        for tr in reversed(self.mass_width_transforms):
            grad = tr.apply_backward(grad, d_in=resolved)
        for tr in reversed(self.scale_transforms):
            grad = tr.apply_backward(grad)
        grad = self.fixed_tr.chain_grad(grad, resolved)
        grad = self.name_res.chain_grad(grad, raw)
        return grad

    # ── rebuild ─────────────────────────────────────────────────

    def _rebuild(self):
        self._var_registry = VariableRegistry()
        added = set()

        def _add(name):
            """Add parameter to registry, deduplicating through same-constraint."""
            canon = self.name_res.map.get(name, name)
            if canon in added or canon in self.fixed_tr.values:
                return
            added.add(canon)
            self._var_registry.add(canon)

        # All names — no type distinction (ck slots, m0, g0, scalar, extra).
        for name in self._all_names:
            _add(name)


# ================================================================
# ParameterizedObjective wrapper
# ================================================================

class ParameterizedObjective:
    """Wraps a kernel compute function with parameter constraints."""

    def __init__(self, compute_fn, param_constraint, data_holder,
                 m0=None, g0=None, scalar=None, norm=None):
        self.pc = param_constraint
        self.compute_fn = compute_fn
        self.data_holder = data_holder
        self.m0 = m0
        self.g0 = g0
        self.scalar = scalar
        self.norm = norm

    def build_params(self, x):
        ck = self.pc.build_ck(x)
        params = {"ck": ck}
        if self.m0 is not None:
            params["m0"] = self.m0
        if self.g0 is not None:
            params["g0"] = self.g0
        if self.scalar is not None:
            params["scalar"] = self.scalar
        return params

    def __call__(self, x):
        params = self.build_params(x)
        Q, grads, P = self.compute_fn(params, self.data_holder, norm=self.norm)
        grad_ck = grads["ck"]
        grad_x = self.pc.backprop_grad(x, grad_ck)
        return Q, grad_x


# ================================================================
# Self-test
# ================================================================
if __name__ == "__main__":
    from ampfit.config_loader import Config
    config = Config("config_angle.yml")
    all_comb = config.get_ck_map()
    ck_names = sorted({p for comb in all_comb for p in comb if isinstance(p, str)})

    pc = ParameterConstraint(all_comb)
    name_res = NameResolution()
    fixed_tr = FixedOverride()

    # Build a slot-level dict (as produced by VariableRegistry.to_dict)
    # ck params: {name_r: mag, name_i: phase}
    np.random.seed(42)
    raw = {}
    for n in ck_names:
        raw[n + 'r'] = np.random.uniform(0.5, 2.0)
        raw[n + 'i'] = np.random.uniform(-np.pi, np.pi)

    # Test resolve pipeline on slot-level dict
    d = name_res.apply(raw)
    d = fixed_tr.apply(d)
    ck = pc.build_ck(d)
    print(f"ck shape: {ck.shape}, ck[:3]: {ck[:3]}")

    # Gradient test: verify backprop_grad against numerical diff
    test_vec = np.random.randn(pc.n_wave) + 1j * np.random.randn(pc.n_wave)

    def Q(slot_dict):
        return np.real(np.sum(pc.build_ck(slot_dict) * np.conj(test_vec)))

    grad_ck_exact = 0.5 * np.conj(test_vec)

    # Analytical gradient w.r.t. the resolved slot dict
    grad_resolved = pc.backprop_grad(d, grad_ck_exact)
    # Chain back through fixed → same → raw
    grad = fixed_tr.chain_grad(grad_resolved, d)
    grad = name_res.chain_grad(grad, raw)

    eps = 1e-6
    errs = []
    for key in list(raw.keys())[:10]:   # test first 10 slots
        sd = raw.copy()
        sd[key] += eps
        Qp = Q(fixed_tr.apply(name_res.apply(sd)))
        sd[key] -= 2 * eps
        Qm = Q(fixed_tr.apply(name_res.apply(sd)))
        num = (Qp - Qm) / (2 * eps)
        ana = grad.get(key, 0.0)
        err = abs(ana - num)
        errs.append(err)
        print(f"  {key:35s} ana={ana:+.6e} num={num:+.6e} err={err:.2e}")
    print(f"Max error (first 10): {max(errs):.2e}")
    assert max(errs) < 1e-5
    print("✓ Full pipeline gradient verified")
