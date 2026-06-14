"""
Independent pipeline stages for parameter constraints.

Each stage is a small object with ``apply(dict)`` (forward) and
``chain_grad(grad_dict, original_dict)`` (backward):

    raw = registry.to_dict(x)
    d   = name_res.apply(raw)       # alias → canonical (same)
    d   = scale_tr.apply(d)         # multiply by scale factors
    d   = fixed_tr.apply(d)         # inject constant values
    ck  = pc.build_ck(d)            # partial-wave amplitudes
    ...
    grad = pc.backprop_grad(d, grad_ck)
    grad = fixed_tr.chain_grad(grad, d)
    grad = scale_tr.chain_grad(grad, d)
    grad = name_res.chain_grad(grad, raw)
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

    Complex parameters → 2 slots ``[r, θ]``.
    Real parameters     → 1 slot.
    """

    def __init__(self):
        self._entries = []
        self._name_to_entry = {}

    def add_complex(self, name, target=None):
        self._entries.append({'name': name, 'kind': 'complex', 'target': target})
        self._name_to_entry[name] = self._entries[-1]

    def add_real(self, name, target=None):
        self._entries.append({'name': name, 'kind': 'real', 'target': target})
        self._name_to_entry[name] = self._entries[-1]

    @property
    def names(self):
        return [e['name'] for e in self._entries]

    @property
    def flat_names(self):
        result = []
        for e in self._entries:
            if e['kind'] == 'complex':
                result.append(e['name'] + 'r')
                result.append(e['name'] + 'i')
            else:
                result.append(e['name'])
        return result

    @property
    def n_flat(self):
        return sum(2 if e['kind'] == 'complex' else 1 for e in self._entries)

    def flat_index(self, name):
        idx = 0
        for e in self._entries:
            if e['name'] == name:
                return (idx, idx + (2 if e['kind'] == 'complex' else 1))
            idx += 2 if e['kind'] == 'complex' else 1
        raise KeyError(f"Unknown variable: {name}")

    def build_initial(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        x = np.empty(self.n_flat)
        idx = 0
        for e in self._entries:
            if e['kind'] == 'complex':
                x[idx] = np.random.uniform(0.5, 2.0)
                x[idx + 1] = np.random.uniform(-np.pi, np.pi)
                idx += 2
            else:
                x[idx] = np.random.uniform(-0.5, 0.5)
                idx += 1
        return x

    # ── forward: flat x → {slot_name: real_value} ──────────────

    def to_dict(self, x):
        """Flat vector → slot-level real dict.

        Complex ck params: ``{name_r: mag, name_i: phase}``.
        Real params: ``{name: value}``.
        """
        result = {}
        idx = 0
        for e in self._entries:
            if e['kind'] == 'complex':
                result[e['name'] + 'r'] = x[idx]
                result[e['name'] + 'i'] = x[idx + 1]
                idx += 2
            else:
                result[e['name']] = x[idx]
                idx += 1
        return result

    # ── backward: {slot_name: grad} → flat gradient ────────────

    def flat_gradient(self, x, grad_dict):
        """Copies slot-level gradients into the flat vector.

        Complex params: ``flat[i] = grad[name_r]``,
        ``flat[i+1] = grad[name_i]``.
        Real params: ``flat[i] = grad[name]``.
        """
        flat = np.zeros(self.n_flat)
        idx = 0
        for e in self._entries:
            if e['kind'] == 'complex':
                flat[idx] = grad_dict.get(e['name'] + 'r', 0.0)
                flat[idx + 1] = grad_dict.get(e['name'] + 'i', 0.0)
                idx += 2
            else:
                flat[idx] = grad_dict.get(e['name'], 0.0)
                idx += 1
        return flat

    # ── compat shims ────────────────────────────────────────────

    def extract_real_dict(self, x):
        return self.to_dict(x)

    def extract_by_target(self, x, target_type):
        result = []
        idx = 0
        for e in self._entries:
            take = (e['target'] is not None and e['target'][0] == target_type)
            if take:
                if e['kind'] == 'complex':
                    result.extend([x[idx], x[idx + 1]])
                    idx += 2
                else:
                    result.append(x[idx])
                    idx += 1
            else:
                idx += 2 if e['kind'] == 'complex' else 1
        return np.array(result)

    def backprop_grad(self, x, grad_dict):
        return self.flat_gradient(x, grad_dict)


# ================================================================
# NameResolution — alias→canonical (same-parameter groups)
# ================================================================

class NameResolution:
    """Resolves alias names to canonical names.

    Forward: ``d[canon] = raw[alias]`` (last alias wins for each canon).
    Backward: ``grad[alias] = grad_out[canon]``.
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
        """Resolve a slot-level key through the alias map.
        
        ``'nr'`` → ``(canon + 'r')`` if *name* is an alias.
        ``'ni'`` → ``(canon + 'i')`` if *name* is an alias.
        Otherwise returns the key unchanged.
        """
        if key.endswith('r') and key[:-1] in self.map:
            return self.map[key[:-1]] + 'r'
        if key.endswith('i') and key[:-1] in self.map:
            return self.map[key[:-1]] + 'i'
        if key in self.map:
            return self.map[key]
        return key

    def apply(self, d):
        result = {}
        for k, v in d.items():
            result[self._resolve_slot(k)] = v
        # Inject alias slot names too — all_comb may reference them
        for alias, canon in self.map.items():
            for suffix in ('r', 'i'):
                if canon + suffix in result:
                    result[alias + suffix] = result[canon + suffix]
            if canon in result:
                result[alias] = result[canon]
        return result

    def chain_grad(self, grad_out, d_in):
        """Reverse of :meth:`apply` — maps resolved grads back to raw keys."""
        result = {}
        for key in d_in:
            resolved = self._resolve_slot(key)
            g = grad_out.get(resolved, 0.0)
            # Also accumulate contributions from aliases that map here
            for alias, canon in self.map.items():
                for suffix in ('r', 'i'):
                    if canon + suffix == resolved and alias + suffix in grad_out:
                        g += grad_out[alias + suffix]
                if canon == resolved and alias in grad_out:
                    g += grad_out[alias]
            result[key] = g
        return result


# ================================================================
# ScaleTransform — multiply values by scale factors
# ================================================================

class ScaleTransform:
    """Applies scale factors (on *original* name before alias resolution).

    Forward: ``d[name] = d[name] * scale[name]`` (if scale[name] exists).
    Backward: ``grad[name] = grad_out[name] * scale[name]``.
    """

    def __init__(self):
        self.factors = {}      # {original_name: float}

    def set_scale(self, scale):
        self.factors = dict(scale)

    def apply(self, d):
        d = dict(d)
        for name, sf in self.factors.items():
            # Scale only applies to magnitude (name_r), not phase (name_i)
            if name + 'r' in d:
                d[name + 'r'] = d[name + 'r'] * sf
            elif name in d:
                d[name] = d[name] * sf
        return d

    def chain_grad(self, grad_out, d_in):
        grad = dict(grad_out)
        for name, sf in self.factors.items():
            if name + 'r' in grad:
                grad[name + 'r'] = grad[name + 'r'] * sf
            elif name in grad:
                grad[name] = grad[name] * sf
        return grad


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

class ConstraintManager:
    """Owns :class:`ParameterConstraint`, :class:`VariableRegistry`,
    and the constraint pipeline stages (:class:`NameResolution`,
    :class:`ScaleTransform`, :class:`FixedOverride`).

    Every ``set_*`` method updates the relevant stage and rebuilds
    the registry — trivial for ~200 parameters.
    """

    SCALAR_NAMES = ["gamma", "delta_gamma", "delta_m", "A_prod", "poqr", "poqi"]

    def __init__(self, all_comb, m0_names, g0_names):
        self.all_comb = list(all_comb)
        self.m0_names = list(m0_names)
        self.g0_names = list(g0_names)

        # Pipeline stages (independent objects)
        self.pc = ParameterConstraint(all_comb)
        self.name_res = NameResolution()
        self.scale_tr = ScaleTransform()
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
        return self.scale_tr.factors

    def free_param_names(self):
        return self.var_registry.flat_names

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
            self.scale_tr.factors = {}
        self.scale_tr.factors.update(dict(scale_params))
        self._rebuild()

    def set_free(self, name):
        name_r = name + 'r'
        name_i = name + 'i'
        for key in (name, name_r, name_i):
            self.fixed_tr.values.pop(key, None)
        # Remove from same groups
        self.name_res.map = {k: v for k, v in self.name_res.map.items()
                             if k != name and v != name}
        self.scale_tr.factors.pop(name, None)
        self._rebuild()

    def set_range(self, name, lo, hi):
        from ampfit.boundary import BoundTransform as _BT
        bt = _BT(lo, hi)
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
        """Run the full constraint pipeline: same → scale → fixed."""
        d = self.name_res.apply(raw_dict)
        d = self.scale_tr.apply(d)
        d = self.fixed_tr.apply(d)
        return d

    # ── backward pipeline ──────────────────────────────────────

    def chain_gradient(self, grad_resolved, resolved, raw):
        """Reverse of :meth:`resolve`."""
        grad = self.fixed_tr.chain_grad(grad_resolved, resolved)
        grad = self.scale_tr.chain_grad(grad, resolved)
        grad = self.name_res.chain_grad(grad, raw)
        return grad

    # ── rebuild ─────────────────────────────────────────────────

    def _rebuild(self):
        # 1. Collect all ck param names from comb structure
        all_ck_names = set()
        for comb in self.all_comb:
            for p in comb:
                if isinstance(p, str):
                    all_ck_names.add(p)

        # 2. Build VariableRegistry
        self._var_registry = VariableRegistry()

        def _slot_fixed(name, suffix=''):
            if (name + suffix) in self.fixed_tr.values:
                return True
            for a in self.name_res.map.get(name, []):
                if (a + suffix) in self.fixed_tr.values:
                    return True
            return False

        def _name_fixed(name):
            if name in self.fixed_tr.values:
                return True
            for a in self.name_res.map.get(name, []):
                if a in self.fixed_tr.values:
                    return True
            return False

        for name in sorted(all_ck_names):
            canon = self.name_res.map.get(name, name)
            r_fixed = _slot_fixed(canon, 'r')
            i_fixed = _slot_fixed(canon, 'i')
            if not (r_fixed and i_fixed):
                self._var_registry.add_complex(canon, ('ck', canon))

        for name in self.m0_names:
            if not _name_fixed(name):
                self._var_registry.add_real(name, ('m0', name))

        for name in self.g0_names:
            if not _name_fixed(name):
                self._var_registry.add_real(name, ('g0', name))

        for name in self.SCALAR_NAMES:
            if not _name_fixed(name):
                self._var_registry.add_real(name, ('scalar', name))


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
    scale_tr = ScaleTransform()
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
    d = scale_tr.apply(d)
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
    # Chain back through fixed → scale → same → raw
    grad = fixed_tr.chain_grad(grad_resolved, d)
    grad = scale_tr.chain_grad(grad, d)
    grad = name_res.chain_grad(grad, raw)

    eps = 1e-6
    errs = []
    for key in list(raw.keys())[:10]:   # test first 10 slots
        sd = raw.copy()
        sd[key] += eps
        Qp = Q(fixed_tr.apply(scale_tr.apply(name_res.apply(sd))))
        sd[key] -= 2 * eps
        Qm = Q(fixed_tr.apply(scale_tr.apply(name_res.apply(sd))))
        num = (Qp - Qm) / (2 * eps)
        ana = grad.get(key, 0.0)
        err = abs(ana - num)
        errs.append(err)
        print(f"  {key:35s} ana={ana:+.6e} num={num:+.6e} err={err:.2e}")
    print(f"Max error (first 10): {max(errs):.2e}")
    assert max(errs) < 1e-5
    print("✓ Full pipeline gradient verified")
