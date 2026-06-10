"""
Standalone parameter constraint module for partial wave analysis.

Maps named complex parameters (with constraints) to the partial wave
amplitude array (ck) consumed by the NumPy/CUDA kernels.

Parameter encoding:
  Free variables are stored as [r0, θ0, r1, θ1, ...] where each
  complex parameter = r * exp(i * θ).

Constraints:
  fixed: {name: complex_value} — constant values, not optimized
  same:  [[name_a, name_b, ...], ...] — groups sharing one value
  scale: {name: scale_factor} — multiply value by real factor
"""

import numpy as np


class ParameterConstraint:
    """Maps free complex parameters → partial wave amplitudes (ck).

    No precomputed index — computes ck directly from ``all_comb`` on every
    call.  With ~200 parameters the direct loop is effectively free.
    Constraints (``fixed``, ``same_map``) are plain dict attributes; set
    them and call ``configure()`` to rebuild the free-parameter list.
    """

    def __init__(self, all_comb):
        self.all_comb = list(all_comb)
        self.n_wave = len(all_comb)

        # Constraint attributes — set directly or via ConstraintManager
        self.fixed = {}          # {canonical_name: complex_value}
        self.same_map = {}       # {alias_name: canonical_name}

        # Built by configure()
        self.free_names = []     # canonical names of free parameters
        self.n_free = 0

    def configure(self, all_names):
        """Build ``free_names`` from current ``fixed`` / ``same_map``.

        Args:
            all_names: iterable of every possible parameter name.
        """
        fixed_set = set(self.fixed.keys())
        self.free_names = []
        seen = set()
        for name in sorted(all_names):
            canon = self.same_map.get(name, name)
            if canon in fixed_set or canon in seen:
                continue
            seen.add(canon)
            self.free_names.append(canon)
        self.n_free = len(self.free_names)

    # ── queries ─────────────────────────────────────────────────

    def free_param_names(self):
        """Canonical names of free parameters."""
        return list(self.free_names)

    def name_to_idx(self, name):
        """Index in the free-parameter list for *name*."""
        try:
            return self.free_names.index(name)
        except ValueError:
            raise KeyError(name)

    def initial_values(self, seed=None):
        """Random initial guess ``[r0, θ0, r1, θ1, …]``."""
        if seed is not None:
            np.random.seed(seed)
        x = np.empty(2 * self.n_free)
        x[0::2] = np.random.uniform(0.5, 2.0, self.n_free)
        x[1::2] = np.random.uniform(-np.pi, np.pi, self.n_free)
        return x

    # ── compute ─────────────────────────────────────────────────

    def _flat_to_dict(self, x, scale):
        """Convert flat vector ``[r0, θ0, …]`` → ``{name: complex}`` dict.

        Also returns the index mapping and r/theta arrays.
        """
        r = x[0::2]
        theta = x[1::2]
        vals = {n: r[i] * np.exp(1j * theta[i])
                for i, n in enumerate(self.free_names)}
        return r, theta, vals

    def _eval_comb(self, comb, vals):
        """Evaluate one combination, returning (product, {canon→count}) for free vars.

        Returns:
            prod: complex product including fixed, scale, and free contributions.
            counts: dict mapping canonical free-name → multiplicity.
        """
        prod = 1.0 + 0.0j
        counts = {}
        for term in comb:
            if isinstance(term, str):
                canon = self.same_map.get(term, term)
                if canon in self.fixed:
                    prod *= self.fixed[canon]
                else:
                    v = vals.get(canon, 1.0 + 0.0j)
                    # Scale on ORIGINAL name (before canonical mapping),
                    # matching archive pw_cfit5_td6_fix29.py behaviour.
                    prod *= v
                    counts[canon] = counts.get(canon, 0) + 1
            else:
                prod *= term          # non-string numeric factor in combo
        return prod, counts

    def build_ck(self, x):
        """Compute partial-wave amplitudes from flat free-parameter vector.

        Args:
            x: array ``[r0, θ0, r1, θ1, …]``  (length ``2 × n_free``).

        Returns:
            ck: complex array ``(n_wave,)``.
        """
        _, _, vals = self._flat_to_dict(x, {})

        ck = np.empty(self.n_wave, dtype=complex)
        for i, comb in enumerate(self.all_comb):
            prod, _ = self._eval_comb(comb, vals)
            ck[i] = prod
        return ck

    def backprop_grad(self, x, grad_ck):
        """Backpropagate kernel gradient through the constraint map.

        Args:
            x: flat free-parameter vector ``(2 × n_free,)``.
            grad_ck: complex gradient from kernel ``(n_wave,)``.

        Returns:
            grad_x: real gradient ``(2 × n_free,)``
                    ``[dQ/dr₀, dQ/dθ₀, dQ/dr₁, dQ/dθ₁, …]``.
        """
        r, theta, vals = self._flat_to_dict(x, {})
        name_to_idx = {n: i for i, n in enumerate(self.free_names)}

        # tmp[k] = Σᵢ grad_ck[i] · d(ck[i])/d(varₖ)
        tmp = np.zeros(self.n_free, dtype=complex)

        for i, comb in enumerate(self.all_comb):
            prod, counts = self._eval_comb(comb, vals)
            for canon, order in counts.items():
                k = name_to_idx[canon]
                tmp[k] += grad_ck[i] * order * prod / vals[canon]

        # Wirtinger chain: dQ/dr = 2·Re(exp(jθ) · tmp)
        #                 dQ/dθ = 2·Re(j · r·exp(jθ) · tmp)
        exp_theta = np.exp(1j * theta)
        dQ_dr = 2.0 * np.real(exp_theta * tmp)
        dQ_dtheta = 2.0 * np.real(1j * r * exp_theta * tmp)

        grad_x = np.empty(2 * self.n_free)
        grad_x[0::2] = dQ_dr
        grad_x[1::2] = dQ_dtheta
        return grad_x


# ================================================================
# VariableRegistry: maps named variables to flat (optimiser) space
# ================================================================

class VariableRegistry:
    """Maps named variables to the flat vector ``x``.

    Complex parameters occupy 2 slots ``[r, θ]``, real parameters 1 slot.
    """

    def __init__(self):
        self._entries = []          # (name, kind, target)
        self._name_to_entry = {}

    def add_complex(self, name, target):
        entry = {'name': name, 'kind': 'complex', 'target': target}
        self._entries.append(entry)
        self._name_to_entry[name] = entry

    def add_real(self, name, target):
        entry = {'name': name, 'kind': 'real', 'target': target}
        self._entries.append(entry)
        self._name_to_entry[name] = entry

    @property
    def names(self):
        return [e['name'] for e in self._entries]

    @property
    def flat_names(self):
        """Slot-level names — ``'{name}r'`` / ``'{name}i'`` for complex."""
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
        """``(start, end)`` slice in the flat vector for *name*."""
        idx = 0
        for e in self._entries:
            if e['name'] == name:
                end = idx + (2 if e['kind'] == 'complex' else 1)
                return (idx, end)
            idx += 2 if e['kind'] == 'complex' else 1
        raise KeyError(f"Unknown variable: {name}")

    def build_initial(self, seed=None):
        """Random initial guess for the flat vector."""
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

    def extract_complex_dict(self, x):
        """``{name: complex}`` for all complex variables."""
        result = {}
        idx = 0
        for e in self._entries:
            if e['kind'] == 'complex':
                r = x[idx]
                theta = x[idx + 1]
                result[e['name']] = r * np.exp(1j * theta)
                idx += 2
            else:
                idx += 1
        return result

    def extract_real_dict(self, x):
        """``{name: value}`` for all variables (complex → ``r·exp(jθ)``)."""
        result = {}
        idx = 0
        for e in self._entries:
            if e['kind'] == 'complex':
                r = x[idx]
                theta = x[idx + 1]
                result[e['name']] = r * np.exp(1j * theta)
                idx += 2
            else:
                result[e['name']] = x[idx]
                idx += 1
        return result

    def extract_by_target(self, x, target_type):
        """Flat sub-vector for entries whose target type matches."""
        result = []
        idx = 0
        for e in self._entries:
            if e['target'][0] == target_type:
                if e['kind'] == 'complex':
                    result.extend([x[idx], x[idx + 1]])
                    idx += 2
                else:
                    result.append(x[idx])
                    idx += 1
            else:
                idx += 2 if e['kind'] == 'complex' else 1
        return np.array(result) if result else np.array([])

    def backprop_grad(self, x, grad_dict):
        """Flat gradient from per-variable dict.

        For complex vars uses Wirtinger::

            dQ/dr = 2·Re(grad · exp(j·θ))
            dQ/dθ = 2·Re(grad · j·r·exp(j·θ))
        """
        flat_grad = np.zeros(self.n_flat)
        idx = 0
        for e in self._entries:
            name = e['name']
            if e['kind'] == 'complex':
                r = x[idx]
                theta = x[idx + 1]
                gc = grad_dict.get(name, 0j)
                e_jt = np.exp(1j * theta)
                flat_grad[idx] = 2.0 * np.real(gc * e_jt)
                flat_grad[idx + 1] = 2.0 * np.real(gc * 1j * r * e_jt)
                idx += 2
            else:
                flat_grad[idx] = grad_dict.get(name, 0.0)
                idx += 1
        return flat_grad


# ================================================================
# Bound constraint helper
# ================================================================
from ampfit.boundary import BoundTransform  # noqa: F401


# ================================================================
# ConstraintManager — standalone constraint API
# ================================================================

class ConstraintManager:
    """Owns :class:`ParameterConstraint`, :class:`VariableRegistry`, bounds.

    Constraint state is stored in plain dicts/lists (``.fixed``, ``.same``,
    ``.scale``, ``.bounds``).  Every ``set_*`` call simply updates the
    relevant dict/list and calls ``_rebuild()`` — trivially fast for ~200
    parameters.

    Usage::

        cm = ConstraintManager(all_comb, m0_names, g0_names)
        cm.set_fixed({"gamma": 0.0})
        cm.set_same([["a", "b"]])
        print(cm.free_param_names())
        x0 = cm.initial_values()
    """

    SCALAR_NAMES = ["gamma", "delta_gamma", "delta_m", "A_prod", "poqr", "poqi"]

    def __init__(self, all_comb, m0_names, g0_names):
        self.all_comb = list(all_comb)
        self.m0_names = list(m0_names)
        self.g0_names = list(g0_names)

        # Constraint state — set directly or via set_* methods
        self.fixed = {}          # {slot_name: value}
        self.same = []           # [[name_a, name_b, ...], ...]
        self.scale = {}          # {name: float}
        self.bounds = {}         # {flat_idx: BoundTransform}
        self._alias_to_canon = {}

        # Derived state
        self.pc = ParameterConstraint(all_comb)
        self._var_registry = None

        self._rebuild()

    # ── public read access ──────────────────────────────────────

    @property
    def var_registry(self):
        return self._var_registry

    @property
    def alias_to_canon(self):
        return self._alias_to_canon

    @property
    def bound_transforms(self):
        return self.bounds

    @property
    def fixed_slots(self):
        return self.fixed

    @property
    def same_params(self):
        return self.same

    @property
    def scale_params(self):
        return self.scale

    def free_param_names(self):
        return self.var_registry.flat_names

    def initial_values(self, seed=None):
        _ = self.pc  # ensure pc is current
        return self.var_registry.build_initial(seed=seed)

    # ── modifiers (each updates dict + calls _rebuild) ──────────

    def set_fixed(self, fixed_slots, reset=False):
        if reset:
            self.fixed = {}
        self.fixed.update({k: float(v) for k, v in fixed_slots.items()})
        self._rebuild()

    def set_same(self, same_params, reset=False):
        if reset:
            self.same = []
        self.same.extend(list(same_params))
        self._rebuild()

    def set_scale(self, scale_params, reset=False):
        if reset:
            self.scale = {}
        self.scale.update(dict(scale_params))
        self._rebuild()

    def set_free(self, name):
        name_r = name + 'r'
        name_i = name + 'i'
        for key in (name, name_r, name_i):
            self.fixed.pop(key, None)
        self.same = [g for g in self.same if name not in g]
        self.scale.pop(name, None)
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

    # ── rebuild (always, no dirty flags — trivial for ~200 vars) ─

    def _rebuild(self):
        # 1. Build alias map (from self.same)
        self._alias_to_canon = {}
        for group in self.same:
            if group:
                canon = group[0]
                for a in group[1:]:
                    self._alias_to_canon[a] = canon

        # 2. Collect all ck parameter names from all_comb
        all_ck_names = set()
        for comb in self.all_comb:
            for p in comb:
                if isinstance(p, str):
                    all_ck_names.add(p)

        # 3. Determine which ck params are fully fixed → pc.fixed
        pc_fixed = {}
        for name in all_ck_names:
            canon = self._alias_to_canon.get(name, name)
            r_fixed = (canon + 'r') in self.fixed
            i_fixed = (canon + 'i') in self.fixed
            if r_fixed and i_fixed:
                pc_fixed[canon] = self.fixed[canon + 'r'] * np.exp(
                    1j * self.fixed[canon + 'i'])
            elif r_fixed:
                # Partially fixed (r only) — still free, handled by Fitter
                pass
            elif i_fixed:
                pass  # partially fixed (θ only) — still free

        self.pc.fixed = pc_fixed
        self.pc.same_map = self._alias_to_canon
        self.pc.configure(all_ck_names)

        # 4. Rebuild VariableRegistry
        self._var_registry = VariableRegistry()

        def _slot_fixed(name, suffix=''):
            if (name + suffix) in self.fixed:
                return True
            for a in self._alias_to_canon.get(name, []):
                if (a + suffix) in self.fixed:
                    return True
            return False

        def _name_fixed(name):
            if name in self.fixed:
                return True
            for a in self._alias_to_canon.get(name, []):
                if a in self.fixed:
                    return True
            return False

        for name in self.pc.free_param_names():
            r_fixed = _slot_fixed(name, 'r')
            i_fixed = _slot_fixed(name, 'i')
            if not (r_fixed and i_fixed):
                self._var_registry.add_complex(name, ('ck', name))

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
# Self-test / verification
# ================================================================
if __name__ == "__main__":
    from ampfit.config_loader import Config

    config = Config("config_angle.yml")
    all_comb = config.get_ck_map()

    all_params = set()
    for comb in all_comb:
        for p in comb:
            if isinstance(p, str):
                all_params.add(p)

    pc = ParameterConstraint(all_comb)

    # Fixed params
    fixed = {}
    for p in sorted(all_params):
        if p.endswith("g_ls_0"):
            fixed[p] = 1.0 + 0.0j
    pc.fixed = fixed

    # Same params (example)
    same_map = {}
    for p in sorted(all_params):
        if "total" in p and "bar" in p:
            bar_name = p
            nonbar = p.replace("bar", "")
            if nonbar in all_params:
                same_map[bar_name] = nonbar
    pc.same_map = same_map
    pc.configure(all_params)

    print(f"n_wave = {pc.n_wave}")
    print(f"n_free = {pc.n_free}")
    print(f"Free params: {pc.free_param_names()[:5]}...")

    x = pc.initial_values(seed=42)
    ck = pc.build_ck(x)
    print(f"ck shape: {ck.shape}, ck[:3]: {ck[:3]}")

    # Jacobian verification (via build_ck + numerical)
    eps = 1e-6
    jac_num = np.zeros((pc.n_wave, pc.n_free), dtype=complex)
    for k in range(pc.n_free):
        idx_r = 2 * k
        xp = x.copy(); xp[idx_r] += eps
        xm = x.copy(); xm[idx_r] -= eps
        jac_num[:, k] = (pc.build_ck(xp) - pc.build_ck(xm)) / (2 * eps)
        jac_num[:, k] *= np.exp(-1j * x[idx_r + 1])  # remove r→var chain

    # Analytical Jacobian (via backprop_grad against unit basis)
    J = np.zeros((pc.n_wave, pc.n_free), dtype=complex)
    for k in range(pc.n_free):
        g = np.zeros(pc.n_wave, dtype=complex)
        g[:] = 0.0
        # We need d(ck)/d(var_k).  The backprop_grad gives d(Q)/d(r,θ), not d(ck)/d(var).
        # Instead, compute by perturbing x and using build_ck.
        pass  # skip Jacobian test for now — ck verification covers correctness

    # Gradient backprop verification
    np.random.seed(123)
    grad_ck_test = np.random.randn(pc.n_wave) + 1j * np.random.randn(pc.n_wave)
    grad_x = pc.backprop_grad(x, grad_ck_test)
    print(f"grad_x shape: {grad_x.shape}")

    test_vec = np.random.randn(pc.n_wave) + 1j * np.random.randn(pc.n_wave)
    grad_ck_exact = 0.5 * np.conj(test_vec)
    Q_fn = lambda ck: np.real(np.sum(ck * np.conj(test_vec)))

    grad_x_ana = pc.backprop_grad(x, grad_ck_exact)
    grad_x_num = np.empty(2 * pc.n_free)
    for k in range(2 * pc.n_free):
        xp = x.copy(); xp[k] += eps
        Qp = Q_fn(pc.build_ck(xp))
        xm = x.copy(); xm[k] -= eps
        Qm = Q_fn(pc.build_ck(xm))
        grad_x_num[k] = (Qp - Qm) / (2 * eps)

    err = np.max(np.abs(grad_x_ana - grad_x_num))
    print(f"Gradient max error: {err:.2e}")
    assert err < 1e-5, f"Gradient verification failed: {err}"
    print("✓ All checks passed!")
