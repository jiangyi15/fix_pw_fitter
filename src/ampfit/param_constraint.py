"""
Standalone parameter constraint module for partial wave analysis.

Maps named complex parameters (with constraints) to the partial wave
amplitude array (ck) consumed by the NumPy/CUDA kernels.

Parameter encoding:
  Free variables are stored as [r0, θ0, r1, θ1, ...] where each
  complex parameter = r * exp(i * θ).

Constraints:
  fixed_params: {name: complex_value} — constant values, not optimized
  same_params:  [[name_a, name_b, ...], ...] — groups sharing one value
  scale_params: {name: scale_factor} — multiply value by real factor

Usage:
    pc = ParameterConstraint(all_comb,
                             fixed_params=fixed_params,
                             same_params=same_params,
                             scale_params=scale_params)
    
    x0 = pc.initial_values()         # random initial guess
    ck = pc.build_ck(x)              # complex (n_wave,) for kernel
    grad_x = pc.backprop_grad(x, grad_ck)  # chain kernel gradient back
"""

import numpy as np
from collections import defaultdict, Counter


class ParameterConstraint:
    """Maps constrained named parameters to partial wave amplitudes (ck)."""

    def __init__(self, all_comb, fixed_params=None,
                 same_params=None, scale_params=None):
        """
        Args:
            all_comb: list of tuples of parameter names. Each tuple defines
                      the product that gives one partial wave amplitude.
                      Shape: (n_wave,). Same as config.get_ck_map().
            fixed_params: dict of {name: complex_value} for fixed parameters.
            same_params: list of lists of names that share one value.
            scale_params: dict of {name: scale_factor} for scaled params.
        """
        self.all_comb = list(all_comb)
        self.n_wave = len(all_comb)

        # Collect all unique parameter names
        self._all_params = set()
        for comb in all_comb:
            for p in comb:
                if isinstance(p, str):
                    self._all_params.add(p)

        # Fixed parameters
        self.fixed_params = dict(fixed_params or {})

        # Same-parameter groups
        self.same_params = list(same_params or [])

        # Build the canonical name mapping
        self._canonical_map = {}  # aliased_name -> canonical_name
        for group in self.same_params:
            if group:
                canon = group[0]
                for name in group[1:]:
                    self._canonical_map[name] = canon

        # Scale factors
        self.scale_params = dict(scale_params or {})

        # Build free parameter list (unique variable parameters)
        self._free_params = []
        for p in sorted(self._all_params):
            # Resolve canonical name
            canon = self._canonical_map.get(p, p)
            # Skip fixed params
            if canon in self.fixed_params:
                continue
            # Skip duplicates (same canonical name appears multiple times)
            if canon not in self._free_params:
                self._free_params.append(canon)

        self.n_free_vars = len(self._free_params)

        # Precompute multiplicity and indexing for fast Jacobian
        self._build_index()

    def _build_index(self):
        """Precompute lookup tables for fast build_ck and Jacobian."""
        # For each free variable, which combinations does it appear in,
        # and with what multiplicity?
        # var_idx_map: canon_name -> var_index
        self._var_index = {name: i for i, name in enumerate(self._free_params)}

        # For each combination, store (var_idx, multiplicity) pairs
        self._comb_vars = []  # list of list of (var_idx, order)
        self._comb_fixed_scale = []  # list of complex scale per combo

        for comb in self.all_comb:
            counts = Counter()
            fixed_scale = 1.0 + 0.0j
            for p in comb:
                if isinstance(p, str):
                    canon = self._canonical_map.get(p, p)
                    # Scale check on the ORIGINAL name (before alias→canonical
                    # mapping), matching archive pw_cfit5_td6_fix29.py behavior:
                    #   if j in scale_params: tmp.append(scale_params[j])
                    scale_val = self.scale_params.get(p)
                    if scale_val is not None:
                        fixed_scale *= scale_val
                    # Fixed check on both original and canonical
                    fixed_val = self.fixed_params.get(p) or self.fixed_params.get(canon)
                    if fixed_val is not None:
                        fixed_scale *= fixed_val
                    else:
                        counts[canon] += 1
                else:
                    # Non-string (numeric scale factor directly in combo)
                    fixed_scale *= p

            # Convert to list of (var_idx, order)
            comb_vars = []
            for canon, order in counts.items():
                if canon in self._var_index:
                    comb_vars.append((self._var_index[canon], order))
            self._comb_vars.append(comb_vars)
            self._comb_fixed_scale.append(fixed_scale)

    def free_param_names(self):
        """Return list of free parameter names (canonical)."""
        return list(self._free_params)

    def initial_values(self, seed=None):
        """Random initial values for the free parameter vector.
        
        Returns:
            array of shape (2 * n_free_vars,) — [r0, θ0, r1, θ1, ...]
        """
        if seed is not None:
            np.random.seed(seed)
        n = self.n_free_vars
        x = np.empty(2 * n)
        x[0::2] = np.random.uniform(0.5, 2.0, n)   # magnitude r
        x[1::2] = np.random.uniform(-np.pi, np.pi, n)  # phase θ
        return x

    def build_ck(self, x):
        """Build complex partial wave amplitudes from real variable vector.
        
        Args:
            x: array of shape (2 * n_free_vars,), [r0, θ0, r1, θ1, ...]
        
        Returns:
            ck: complex array of shape (n_wave,)
        """
        # Convert to complex free parameters
        r = x[0::2]
        theta = x[1::2]
        free_vals = r * np.exp(1j * theta)  # (n_free_vars,) complex

        # Build ck for each combination
        ck = np.empty(self.n_wave, dtype=np.complex128)
        for i, (comb_vars, fixed_scale) in enumerate(
                zip(self._comb_vars, self._comb_fixed_scale)):
            val = fixed_scale
            for var_idx, order in comb_vars:
                val *= free_vals[var_idx] ** order
            ck[i] = val

        return ck

    def build_jac(self, x):
        """Compute the complex Jacobian d(ck)/d(free_var).
        
        jac[i, k] = d(ck[i]) / d(var[k])  (complex derivative)
        
        Args:
            x: array of shape (2 * n_free_vars,)
        
        Returns:
            jac: complex array of shape (n_wave, n_free_vars)
        """
        r = x[0::2]
        theta = x[1::2]
        free_vals = r * np.exp(1j * theta)

        jac = np.zeros((self.n_wave, self.n_free_vars), dtype=np.complex128)

        for i, (comb_vars, fixed_scale) in enumerate(
                zip(self._comb_vars, self._comb_fixed_scale)):
            # Fully compute ck[i] once
            val = fixed_scale
            for var_idx, order in comb_vars:
                val *= free_vals[var_idx] ** order

            # For each var in this combo: d(ck[i])/d(var[k])
            for var_idx, order in comb_vars:
                if order == 1:
                    jac[i, var_idx] = val / free_vals[var_idx]
                else:
                    jac[i, var_idx] = order * val / free_vals[var_idx]

        return jac

    def backprop_grad(self, x, grad_ck, return_real_grad=True):
        """Backpropagate kernel gradient through parameter constraints.
        
        Given dQ/d(ck) from the kernel backward pass, compute
        dQ/d(x) for the optimizer.
        
        Args:
            x: real variable vector (2 * n_free_vars,), [r0, θ0, ...]
            grad_ck: complex gradient from kernel, (n_wave,)
                     Each element = ∂Q/∂(Re(ck_i)) + j * ∂Q/∂(Im(ck_i))
            return_real_grad: if True, return gradient w.r.t. [r, θ] format
                              if False, return complex gradient w.r.t. free_var
        
        Returns:
            If return_real_grad: array (2 * n_free_vars,)
              [dQ/dr_0, dQ/dθ_0, dQ/dr_1, dQ/dθ_1, ...]
            If not return_real_grad: complex array (n_free_vars,)
              Wirtinger dQ/d(free_var)
        """
        r = x[0::2]
        theta = x[1::2]
        free_vals = r * np.exp(1j * theta)

        # Complex Jacobian: jac[i,k] = d(ck[i]) / d(var[k])
        jac = self.build_jac(x)  # (n_wave, n_free_vars) complex

        # The kernel's grad_ck[i] = dQ/d(ck_i)  (standard Wirtinger derivative)
        #    = 1/2 * (∂Q/∂Re(ck_i) - j * ∂Q/∂Im(ck_i))
        # For real-valued functions, the chain rule is:
        #   dQ/d(p) = 2 * Re( sum_i dQ/d(ck_i) * d(ck_i)/d(p) )
        #
        # Our variables: var_k = r_k * exp(j*θ_k)
        #   d(ck[i])/d(r_k)   = jac[i,k] * exp(j*θ_k)
        #   d(ck[i])/d(θ_k)   = jac[i,k] * j * r_k * exp(j*θ_k)
        #                      = jac[i,k] * j * free_vals[k]

        tmp = np.sum(grad_ck[:, None] * jac, axis=0)

        if not return_real_grad:
            # Complex gradient w.r.t. free_var_k
            grad_dr = 2.0 * np.real(np.exp(1j * theta) * tmp)
            grad_dtheta = 2.0 * np.real(1j * free_vals * tmp)
            dQ_dvar_real = (grad_dr * np.cos(theta)
                            - grad_dtheta * np.sin(theta) / r)
            dQ_dvar_imag = (grad_dr * np.sin(theta)
                            + grad_dtheta * np.cos(theta) / r)
            return dQ_dvar_real + 1j * dQ_dvar_imag

        # Real gradients dQ/d(r_k), dQ/d(θ_k)
        # dQ/d(r_k) = 2 * Re( exp(j*θ_k) * tmp[k] )
        # dQ/d(θ_k) = 2 * Re( j * free_vals[k] * tmp[k] )
        dQ_dr = 2.0 * np.real(np.exp(1j * theta) * tmp)
        dQ_dtheta = 2.0 * np.real(1j * free_vals * tmp)

        # Interleave: [dQ/dr_0, dQ/dθ_0, dQ/dr_1, dQ/dθ_1, ...]
        grad_x = np.empty(2 * self.n_free_vars)
        grad_x[0::2] = dQ_dr
        grad_x[1::2] = dQ_dtheta
        return grad_x


# ================================================================
# VariableRegistry: maps named variables to flat vector indices
# ================================================================

class VariableRegistry:
    """Maps named variables to flat vector (optimizer-space) indices.

    Complex parameters occupy 2 slots [r, θ], real parameters occupy 1 slot.
    The registry works with a flat ``x`` vector of real values.
    """

    def __init__(self):
        self._entries = []          # (name, kind, target)
        self._name_to_entry = {}

    def add_complex(self, name, target):
        """Add a complex variable (2 slots: ``name_r``, ``name_i``)."""
        entry = {'name': name, 'kind': 'complex', 'target': target}
        self._entries.append(entry)
        self._name_to_entry[name] = entry

    def add_real(self, name, target):
        """Add a real variable (1 slot: ``name``)."""
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
        """Return ``(start, end)`` slice in the flat vector for *name*."""
        idx = 0
        for e in self._entries:
            if e['name'] == name:
                end = idx + (2 if e['kind'] == 'complex' else 1)
                return (idx, end)
            idx += 2 if e['kind'] == 'complex' else 1
        raise KeyError(f"Unknown variable: {name}")

    def build_initial(self, seed=None):
        """Random initial guess for all variables in the flat vector."""
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
        """Return ``{name: complex}`` for all complex variables."""
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
        """Return ``{name: value_or_complex}`` for all variables."""
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
        """Flat sub-vector for entries whose target matches *target_type*."""
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
        """Build flat gradient from per-variable dict.

        For complex vars uses Wirtinger::

            dQ/dr      = 2·Re(grad · exp(j·θ))
            dQ/dθ      = 2·Re(grad · j·r·exp(j·θ))
        """
        flat_grad = np.zeros(self.n_flat)
        idx = 0
        for e in self._entries:
            name = e['name']
            if e['kind'] == 'complex':
                r = x[idx]
                theta = x[idx + 1]
                grad_complex = grad_dict.get(name, 0j)
                exp_theta = np.exp(1j * theta)
                flat_grad[idx] = 2.0 * np.real(grad_complex * exp_theta)
                flat_grad[idx + 1] = 2.0 * np.real(grad_complex * 1j * r * exp_theta)
                idx += 2
            else:
                flat_grad[idx] = grad_dict.get(name, 0.0)
                idx += 1
        return flat_grad


# ================================================================
# Bound constraint helper: maps unbounded -> bounded via arctan
# ================================================================
from ampfit.boundary import BoundTransform  # noqa: F401


# ================================================================
# ConstraintManager — standalone constraint API
# ================================================================

class ConstraintManager:
    """Standalone parameter constraint manager.

    Owns the :class:`ParameterConstraint` (partial-wave amplitude coefficients),
    the :class:`VariableRegistry` (flat variable space), bound transforms,
    fixed slots, same-parameter groups, scale factors, and alias maps.

    Each ``set_*`` method only marks the relevant subsystem dirty
    (either *pc* or the variable registry), avoiding unnecessary rebuilds.

    Usage::

        cm = ConstraintManager(all_comb, config.m0_phys_name, config.g0_phys_name)
        cm.set_fixed({"gamma": 0.0})
        cm.set_same([["a", "b"]])
        cm.set_scale({"x": -1})
        cm.set_range("delta_m", 0.3, 0.8)
        print(cm.free_param_names())
        x0 = cm.initial_values()
    """

    SCALAR_NAMES = ["gamma", "delta_gamma", "delta_m", "A_prod", "poqr", "poqi"]

    def __init__(self, all_comb, m0_names, g0_names):
        self.all_comb = list(all_comb)
        self.m0_names = list(m0_names)
        self.g0_names = list(g0_names)

        # Constraint state
        self._fixed_slots = {}
        self._same_params = []
        self._scale_params = {}
        self._bound_transforms = {}          # {flat_idx: BoundTransform}
        self._alias_to_canon = {}

        # Lazy-built with per-subsystem dirty flags
        self._pc = None
        self._var_registry = None
        self._pc_dirty = True
        self._vr_dirty = True

    # ── Public read access ──────────────────────────────────────

    @property
    def pc(self):
        """Lazy :class:`ParameterConstraint` (rebuilt when same/scale change)."""
        if self._pc_dirty:
            self._rebuild_pc()
        return self._pc

    @property
    def var_registry(self):
        """Lazy :class:`VariableRegistry` (rebuilt when free-variable space changes)."""
        if self._vr_dirty:
            self._rebuild_var_registry()
        return self._var_registry

    @property
    def fixed_slots(self):
        return self._fixed_slots

    @property
    def same_params(self):
        return self._same_params

    @property
    def scale_params(self):
        return self._scale_params

    @property
    def bound_transforms(self):
        return self._bound_transforms

    @property
    def alias_to_canon(self):
        return self._alias_to_canon

    # ── Constraint modifiers (fine-grained dirty flags) ─────────

    def set_fixed(self, fixed_slots, reset=False):
        """Fix parameter slot(s).  *Additive* — call multiple times.

        Only marks the variable registry dirty (``ParameterConstraint``
        always uses ``fixed_params={}`` at this level).
        """
        if reset:
            self._fixed_slots = {}
        self._fixed_slots.update({k: float(v) for k, v in fixed_slots.items()})
        self._vr_dirty = True

    def set_same(self, same_params, reset=False):
        """Share a value across parameter names. *Additive*."""
        if reset:
            self._same_params = []
        self._same_params.extend(list(same_params))
        self._pc_dirty = True       # canonical mapping changes
        self._vr_dirty = True       # alias map changes

    def set_scale(self, scale_params, reset=False):
        """Multiply a parameter by a real scale factor. *Additive*.

        Only marks ``ParameterConstraint`` dirty (scale changes
        ``_comb_fixed_scale``); the variable registry is unaffected.
        """
        if reset:
            self._scale_params = {}
        self._scale_params.update(dict(scale_params))
        self._pc_dirty = True

    def set_free(self, name):
        """Unfix a parameter — removes from fixed/same/scale."""
        name_r = name + 'r'
        name_i = name + 'i'
        for key in (name, name_r, name_i):
            self._fixed_slots.pop(key, None)

        was_same = any(name in g for g in self._same_params)
        self._same_params = [g for g in self._same_params if name not in g]

        was_scaled = name in self._scale_params
        self._scale_params.pop(name, None)

        self._vr_dirty = True
        if was_same or was_scaled:
            self._pc_dirty = True

    def set_range(self, name, lo, hi):
        """Bound a parameter via bijective arctan transform.  *Additive*.

        Does **not** dirty either subsystem — only updates
        ``_bound_transforms``.
        """
        from ampfit.boundary import BoundTransform as _BT
        bt = _BT(lo, hi)
        try:
            si, ei = self.var_registry.flat_index(name)
            for idx in range(si, ei):
                self._bound_transforms[idx] = bt
        except KeyError:
            for i, n in enumerate(self.var_registry.flat_names):
                if n == name:
                    self._bound_transforms[i] = bt
                    return
            raise ValueError(
                f"Unknown parameter '{name}'. "
                f"Available: {self.var_registry.flat_names[:6]}...")

    def unset_range(self, name):
        """Remove the arctan bound on *name*."""
        try:
            si, ei = self.var_registry.flat_index(name)
            for idx in range(si, ei):
                self._bound_transforms.pop(idx, None)
        except KeyError:
            for i, n in enumerate(self.var_registry.flat_names):
                if n == name:
                    self._bound_transforms.pop(i, None)

    # ── Queries ─────────────────────────────────────────────────

    def free_param_names(self):
        """Slot-level names of all free variables."""
        return self.var_registry.flat_names

    def initial_values(self, seed=None):
        """Random initial guess in the flat variable space."""
        _ = self.pc          # ensure PC is built
        return self.var_registry.build_initial(seed=seed)

    # ── Internal rebuilds ───────────────────────────────────────

    def _build_alias_map(self):
        alias_to_canon = {}
        for group in self._same_params:
            if group:
                canon = group[0]
                for a in group[1:]:
                    alias_to_canon[a] = canon
        self._alias_to_canon = alias_to_canon

    def _rebuild_pc(self):
        self._build_alias_map()
        self._pc = ParameterConstraint(
            self.all_comb,
            fixed_params={},          # complex-level fixes handled by _fixed_slots
            same_params=self._same_params,
            scale_params=self._scale_params,
        )
        self._pc_dirty = False

    def _rebuild_var_registry(self):
        # Ensure pc and alias map are current
        if self._pc_dirty:
            self._rebuild_pc()
        elif not self._alias_to_canon:
            self._build_alias_map()

        def _slot_fixed(name, suffix=''):
            if (name + suffix) in self._fixed_slots:
                return True
            for a in self._alias_to_canon.get(name, []):
                if (a + suffix) in self._fixed_slots:
                    return True
            return False

        def _name_fixed(name):
            if name in self._fixed_slots:
                return True
            for a in self._alias_to_canon.get(name, []):
                if a in self._fixed_slots:
                    return True
            return False

        self._var_registry = VariableRegistry()

        # Ck parameters (complex) — skip if both r and i fully fixed
        for name in self._pc.free_param_names():
            r_fixed = _slot_fixed(name, 'r')
            i_fixed = _slot_fixed(name, 'i')
            if not (r_fixed and i_fixed):
                self._var_registry.add_complex(name, ('ck', name))

        # M0 parameters (real)
        for name in self.m0_names:
            if not _name_fixed(name):
                self._var_registry.add_real(name, ('m0', name))

        # G0 parameters (real)
        for name in self.g0_names:
            if not _name_fixed(name):
                self._var_registry.add_real(name, ('g0', name))

        # Scalar/time parameters (real)
        for name in self.SCALAR_NAMES:
            if not _name_fixed(name):
                self._var_registry.add_real(name, ('scalar', name))

        self._vr_dirty = False


# ================================================================
# Complete optimizer-ready objective wrapper
# ================================================================
class ParameterizedObjective:
    """Wraps a kernel compute function with parameter constraints.
    
    Usage:
        kernel = CUDAKernel(config)
        data = kernel.load_data(data_dict)
        
        pc = ParameterConstraint(all_comb, fixed_params=fixed_params, ...)
        
        obj = ParameterizedObjective(kernel.compute, pc, data)
        
        # Optimizer calls:
        Q, grads = obj(x)  # returns negative log-likelihood + gradient
        
        # Or access kernel params directly:
        params = obj.build_params(x, m0=m0, g0=g0, scalar=scalar)
    """

    def __init__(self, compute_fn, param_constraint, data_holder,
                 m0=None, g0=None, scalar=None, norm=None):
        """
        Args:
            compute_fn: callable(params, data_holder) -> (Q, grads, P)
            param_constraint: ParameterConstraint instance
            data_holder: GPUDataHolder (or numpy data dict)
            m0: array of m0 values (or None for defaults)
            g0: array of g0 values (or None for defaults)
            scalar: list of scalar params (or None for defaults)
            norm: optional normalization factor
        """
        self.pc = param_constraint
        self.compute_fn = compute_fn
        self.data_holder = data_holder
        self.m0 = m0
        self.g0 = g0
        self.scalar = scalar
        self.norm = norm

        self._last_params = None

    def build_params(self, x):
        """Build full params dict from free variable vector x."""
        ck = self.pc.build_ck(x)
        params = {
            "ck": ck,
            "m0": self.m0,
            "g0": self.g0,
            "scalar": self.scalar,
        }
        # Filter None values
        params = {k: v for k, v in params.items() if v is not None}
        return params

    def __call__(self, x):
        """Compute negative log-likelihood and its gradient w.r.t. x.
        
        Args:
            x: real variable vector (2 * n_free_vars,)
        
        Returns:
            Q: scalar (negative log-likelihood)
            grad_x: gradient array same shape as x
        """
        params = self.build_params(x)
        Q, grads, P = self.compute_fn(params, self.data_holder, norm=self.norm)

        # Backpropagate gradient through parameter constraint
        grad_ck = grads["ck"]
        grad_x = self.pc.backprop_grad(x, grad_ck, return_real_grad=True)

        return Q, grad_x


# ================================================================
# Self-test / verification
# ================================================================
if __name__ == "__main__":
    from ampfit.config_loader import Config

    config = Config("config_angle.yml")
    all_comb = config.get_ck_map()

    # Build constraint system similar to pw_cfit5_td6_fix29.py
    all_params = set()
    for comb in all_comb:
        for p in comb:
            if isinstance(p, str):
                all_params.add(p)

    # Fixed params (example: some g_ls_0)
    fixed_params = {}
    for p in sorted(all_params):
        if p.endswith("g_ls_0"):
            fixed_params[p] = 1.0 + 0.0j

    # Same params (example: g_ls_1 == g_lsbar_1)
    same_params = []
    total_params_all = [p for p in sorted(all_params) if "total_0" in p]
    # ...
    # For testing, use a simple system

    pc = ParameterConstraint(all_comb, fixed_params=fixed_params)

    print(f"n_wave = {pc.n_wave}")
    print(f"n_free_vars = {pc.n_free_vars}")
    print(f"Free params: {pc.free_param_names()[:10]}...")

    # Test build_ck
    x = pc.initial_values(seed=42)
    ck = pc.build_ck(x)
    print(f"\nck shape: {ck.shape}")
    print(f"ck[:3]: {ck[:3]}")

    # Test Jacobian via numerical differentiation
    eps = 1e-6
    jac_num = np.zeros((pc.n_wave, pc.n_free_vars), dtype=np.complex128)
    x_test = x.copy()
    for k in range(pc.n_free_vars):
        # Perturb var k's magnitude
        idx_r = 2 * k
        x_test[idx_r] += eps
        ck_plus = pc.build_ck(x_test)
        x_test[idx_r] -= 2 * eps
        ck_minus = pc.build_ck(x_test)
        x_test[idx_r] += eps
        d_r = (ck_plus - ck_minus) / (2 * eps)
        # Need to get d/d(r_k) of complex: using chain rule
        # d(ck)/d(r_k) = d(ck)/d(var_k) * d(var_k)/d(r_k) = jac[i,k] * exp(j*θ_k)
        # So jac[i,k] = d(ck)/d(r_k) * exp(-j*θ_k)
        theta = x_test[idx_r + 1]
        jac_num[:, k] = d_r * np.exp(-1j * theta)

    jac_ana = pc.build_jac(x)
    err = np.max(np.abs(jac_ana - jac_num))
    print(f"\nJacobian max error: {err:.2e}")
    assert err < 1e-5, f"Jacobian verification failed: {err}"
    print("✓ Jacobian verified!")

    # Test gradient backprop
    np.random.seed(123)
    grad_ck_test = np.random.randn(pc.n_wave) + 1j * np.random.randn(pc.n_wave)
    grad_x = pc.backprop_grad(x, grad_ck_test)
    print(f"\nBackprop gradient shape: {grad_x.shape}")

    # Verify gradient numerically
    # Define a test function Q = Re(sum(ck * conj(test_vec)))
    # ∂Q/∂Re(ck_i) = Re(test_vec[i]), ∂Q/∂Im(ck_i) = Im(test_vec[i])
    # Standard Wirtinger: dQ/d(ck_i) = 1/2 * (∂Q/∂Re - j*∂Q/∂Im) = 1/2 * conj(test_vec)
    test_vec = np.random.randn(pc.n_wave) + 1j * np.random.randn(pc.n_wave)
    grad_ck_exact = 0.5 * np.conj(test_vec)  # Standard Wirtinger derivative
    Q_fn = lambda ck: np.real(np.sum(ck * np.conj(test_vec)))

    grad_x_ana = pc.backprop_grad(x, grad_ck_exact)

    grad_x_num = np.empty(2 * pc.n_free_vars)
    for k in range(2 * pc.n_free_vars):
        xp = x.copy()
        xp[k] += eps
        Qp = Q_fn(pc.build_ck(xp))
        xp[k] -= 2 * eps
        Qm = Q_fn(pc.build_ck(xp))
        grad_x_num[k] = (Qp - Qm) / (2 * eps)

    err_grad = np.max(np.abs(grad_x_ana - grad_x_num))
    print(f"Gradient backprop max error: {err_grad:.2e}")
    assert err_grad < 1e-5, f"Gradient backprop failed: {err_grad}"
    print("✓ Gradient backprop verified!")

    print("\nAll tests passed!")
