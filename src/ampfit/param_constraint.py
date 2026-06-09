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

        # Build per-variable scale: check canonical AND alias names
        self._var_scale = {}
        for name in self._free_params:
            s = self.scale_params.get(name)
            if s is None:
                # Check aliases of this canonical name
                for alias, canon in self._canonical_map.items():
                    if canon == name and alias in self.scale_params:
                        s = self.scale_params[alias]
                        break
            if s is not None:
                self._var_scale[name] = s

        # For each combination, store (var_idx, multiplicity) pairs
        self._comb_vars = []  # list of list of (var_idx, order)
        self._comb_fixed_scale = []  # list of complex scale per combo

        for comb in self.all_comb:
            counts = Counter()
            fixed_scale = 1.0 + 0.0j
            for p in comb:
                if isinstance(p, str):
                    canon = self._canonical_map.get(p, p)
                    fixed_val = self.fixed_params.get(canon) or self.fixed_params.get(p)
                    if fixed_val is not None:
                        fixed_scale *= fixed_val
                    elif canon in self._var_scale:
                        counts[canon] += 1
                        fixed_scale *= self._var_scale[canon]
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
# Bound constraint helper: maps unbounded -> bounded via sin transform
# ================================================================
# Re-export BoundTransform from boundary module (complete implementation)
from ampfit.boundary import BoundTransform  # noqa: F401


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
