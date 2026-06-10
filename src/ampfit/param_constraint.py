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

    def build_ck(self, param_dict):
        ck = np.empty(self.n_wave, dtype=complex)
        for i, comb in enumerate(self.all_comb):
            prod = 1.0 + 0.0j
            for term in comb:
                if isinstance(term, str):
                    prod *= param_dict.get(term, 1.0 + 0.0j)
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

    def backprop_grad(self, param_dict, grad_ck):
        """Wirtinger gradient ``dQ/d(name)`` for each name in ``all_comb``."""
        ck = self.build_ck(param_dict)
        grads = {}
        for i, comb in enumerate(self.all_comb):
            counts = {}
            for term in comb:
                if isinstance(term, str):
                    counts[term] = counts.get(term, 0) + 1
            for name, order in counts.items():
                dck = order * ck[i] / param_dict[name]
                grads[name] = grads.get(name, 0j) + grad_ck[i] * dck
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

    # ── forward ─────────────────────────────────────────────────

    def to_dict(self, x):
        """Flat vector ``→ {name: complex_or_real}``."""
        result = {}
        idx = 0
        for e in self._entries:
            if e['kind'] == 'complex':
                r, theta = x[idx], x[idx + 1]
                result[e['name']] = r * np.exp(1j * theta)
                idx += 2
            else:
                result[e['name']] = x[idx]
                idx += 1
        return result

    # ── backward ────────────────────────────────────────────────

    def flat_gradient(self, x, grad_dict):
        """``{name: Wirtinger_grad} → flat real gradient``.

        For complex vars (Wirtinger)::

            dQ/dr = 2·Re(grad·exp(j·θ))
            dQ/dθ = 2·Re(grad·j·r·exp(j·θ))
        """
        flat = np.zeros(self.n_flat)
        idx = 0
        for e in self._entries:
            if e['kind'] == 'complex':
                r, theta = x[idx], x[idx + 1]
                gc = grad_dict.get(e['name'], 0j)
                ejt = np.exp(1j * theta)
                flat[idx] = 2.0 * np.real(gc * ejt)
                flat[idx + 1] = 2.0 * np.real(gc * 1j * r * ejt)
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

    def apply(self, d):
        # Canonical names first
        result = {self.map.get(k, k): v for k, v in d.items()}
        # Also inject alias names with the same value — all_comb may
        # reference either the alias or the canonical name.
        for alias, canon in self.map.items():
            if canon in result:
                result[alias] = result[canon]
        return result

    def chain_grad(self, grad_out, d_in):
        """grad_out was w.r.t. resolved dict; d_in is the input dict."""
        result = {}
        for name in d_in:
            canon = self.map.get(name, name)
            if canon in grad_out:
                result[name] = grad_out[canon]
            # Also add alias contributions: grad[alias] = grad[canon]
            for alias, cn in self.map.items():
                if cn == canon and alias in grad_out:
                    result[name] = result.get(name, 0.0) + grad_out[alias]
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
            if name in d:
                d[name] = d[name] * sf
        return d

    def chain_grad(self, grad_out, d_in):
        grad = dict(grad_out)
        for name, sf in self.factors.items():
            if name in grad:
                grad[name] = grad[name] * sf
        return grad


# ================================================================
# FixedOverride — inject constant parameter values
# ================================================================

class FixedOverride:
    """Injects fixed (constant) values into the param dict.

    Handles both canonical names (``'gamma': 0.0``) and slot-level
    r/i pairs (``{'nr': 1.0, 'ni': 0.0}`` → ``'n': 1+0j``).
    """

    def __init__(self):
        self.values = {}       # {slot_name: value}

    def set_fixed(self, fixed):
        self.values = dict(fixed)

    def _fixed_complex(self):
        """Build ``{canon: complex}`` for fully-fixed r/i slot pairs."""
        result = {}
        for name in list(self.values.keys()):
            if name.endswith('r') and name[:-1] + 'i' in self.values:
                canon = name[:-1]
                result[canon] = self.values[canon + 'r'] + 1j * self.values[canon + 'i']
        return result

    def apply(self, d):
        d = dict(d)
        for name, val in self.values.items():
            if name in d:
                d[name] = val
        d.update(self._fixed_complex())
        return d

    def chain_grad(self, grad_out, d_in):
        grad = dict(grad_out)
        for name in self.values:
            grad.pop(name, None)          # remove slot-level names
        for canon in self._fixed_complex():
            grad.pop(canon, None)         # remove composite canonical names
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

    # Independent pipeline stages
    pc = ParameterConstraint(all_comb)
    name_res = NameResolution()
    scale_tr = ScaleTransform()
    fixed_tr = FixedOverride()

    # Configure them independently
    fixed_tr.set_fixed({n: 1.0 + 0.0j for n in ck_names if n.endswith("g_ls_0")})
    # (no same, no scale for this test)

    np.random.seed(42)
    raw = {n: np.random.randn() + 1j * np.random.randn() for n in ck_names}
    resolved = fixed_tr.apply(raw)    # same → scale → fixed
    ck = pc.build_ck(resolved)
    print(f"ck shape: {ck.shape}, ck[:3]: {ck[:3]}")

    # Gradient test
    test_vec = np.random.randn(pc.n_wave) + 1j * np.random.randn(pc.n_wave)
    Q_fn = lambda pd: np.real(np.sum(pc.build_ck(pd) * np.conj(test_vec)))
    grad_ck_exact = 0.5 * np.conj(test_vec)

    grad_pc = pc.backprop_grad(resolved, grad_ck_exact)
    grad = fixed_tr.chain_grad(grad_pc, resolved)
    # grad now in raw space

    eps = 1e-6
    errs = []
    for name in ck_names[:5]:
        pd = raw.copy()
        pd[name] += eps
        Qp = Q_fn(fixed_tr.apply(pd))
        pd[name] -= 2 * eps
        Qm = Q_fn(fixed_tr.apply(pd))
        num = (Qp - Qm) / (2 * eps)
        ana = grad.get(name, 0j)
        err = abs(2.0 * ana.real - num)
        errs.append(err)
        print(f"  {name[:35]:35s} 2·Re(ana)={2*ana.real:+.6e} num={num:+.6e} err={err:.2e}")
    print(f"Max error: {max(errs):.2e}")
    assert max(errs) < 1e-5
    print("✓ Pipeline stages verified")
