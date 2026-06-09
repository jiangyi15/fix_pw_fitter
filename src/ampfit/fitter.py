"""
Global Fitter class: single entry point for amplitude analysis.

Usage:
    fitter = Fitter("config_angle.yml")
    
    # Set parameter constraints (optional)
    fitter.set_fixed(fixed_params)
    fitter.set_same(same_params)  
    fitter.set_scale(scale_params)
    
    # Set datasets
    fitter.set_data(data_dict)
    fitter.set_phsp(phsp_dict)
    fitter.set_default_params(m0=m0_arr, g0=g0_arr, scalar=scalar_list)
    
    # Compute NLL (for optimizer)
    x = fitter.initial_values()          # initial guess in free variable space
    nll, grad_x = fitter.get_nll(x)      # returns (scalar, 2*n_vars array)
    
    # Or compute NLL from raw params (no constraint transformation)
    nll, grads = fitter.get_nll_raw(params_dict)
"""

import numpy as np


class VariableRegistry:
    """Maps named variables to flat vector indices and kernel slots."""
    
    def __init__(self):
        self._entries = []  # list of (name, kind, target)
        self._name_to_entry = {}  # name -> entry
    
    def add_complex(self, name, target):
        """Add a complex variable (2 flat slots: name_r, name_i)."""
        entry = {'name': name, 'kind': 'complex', 'target': target}
        self._entries.append(entry)
        self._name_to_entry[name] = entry
    
    def add_real(self, name, target):
        """Add a real variable (1 flat slot: name)."""
        entry = {'name': name, 'kind': 'real', 'target': target}
        self._entries.append(entry)
        self._name_to_entry[name] = entry
    
    @property
    def names(self):
        return [e['name'] for e in self._entries]
    
    @property
    def flat_names(self):
        """Slot-level names: '{name}r', '{name}i' for complex, '{name}' for real.
        
        Length matches n_flat (and the flat vector x).
        Example: ['B->...total_0r', 'B->...total_0i', 'gamma', ...]
        """
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
        """Total number of real values in the flat vector."""
        return sum(2 if e['kind'] == 'complex' else 1 for e in self._entries)
    
    def flat_index(self, name):
        """Return (start, end) indices in the flat vector for a named variable.
        
        For real: returns (i, i+1)
        For complex: returns (i, i+2) where i = r, i+1 = imag
        """
        idx = 0
        for e in self._entries:
            if e['name'] == name:
                end = idx + (2 if e['kind'] == 'complex' else 1)
                return (idx, end)
            idx += 2 if e['kind'] == 'complex' else 1
        raise KeyError(f"Unknown variable: {name}")
    
    def build_initial(self, seed=None):
        """Build initial flat vector with random values."""
        import numpy as np
        if seed is not None:
            np.random.seed(seed)
        x = np.empty(self.n_flat)
        idx = 0
        for e in self._entries:
            if e['kind'] == 'complex':
                r = np.random.uniform(0.5, 2.0)
                theta = np.random.uniform(-np.pi, np.pi)
                x[idx] = r
                x[idx + 1] = theta
                idx += 2
            else:
                x[idx] = np.random.uniform(-0.5, 0.5)
                idx += 1
        return x
    
    def extract_complex_dict(self, x):
        """Extract {name: complex} for all complex variables from flat x."""
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
        """Extract {name: value} for all real variables from flat x."""
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
        """Extract flat sub-vector for all entries with matching target type.
        
        Args:
            x: flat vector with all variables
            target_type: string like 'ck', 'scalar', 'm0', 'g0'
        
        Returns:
            flat sub-vector with only matching entries (in order they appear)
        """
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
        """Build flat gradient from dict of {name: complex_grad} or {name: real_grad}.
        
        For complex vars: grad_dict[name] = dQ/d(var) (complex Wirtinger derivative)
        The real gradient w.r.t. r, theta is:
          dQ/dr = 2 * Re(dQ/d(var) * exp(j*theta))
          dQ/dtheta = 2 * Re(dQ/d(var) * j * r * exp(j*theta))
        """
        import numpy as np
        flat_grad = np.zeros(self.n_flat)
        idx = 0
        for e in self._entries:
            name = e['name']
            if e['kind'] == 'complex':
                r = x[idx]
                theta = x[idx + 1]
                grad_complex = grad_dict.get(name, 0j)
                # Wirtinger: dQ/dr = 2*Re(grad * exp(j*θ)), dQ/dθ = 2*Re(grad * j * r * exp(j*θ))
                exp_theta = np.exp(1j * theta)
                flat_grad[idx] = 2.0 * np.real(grad_complex * exp_theta)
                flat_grad[idx + 1] = 2.0 * np.real(grad_complex * 1j * r * exp_theta)
                idx += 2
            else:
                flat_grad[idx] = grad_dict.get(name, 0.0)
                idx += 1
        return flat_grad


class Fitter:
    """Global fitter: config → objects → compute with norm constraint."""

    def __init__(self, config_file="config_angle.yml"):
        """Load config, create kernel and parameter constraint."""
        # Lazy imports so this module can be imported without CUDA etc.
        from ampfit.config_loader import Config
        from ampfit._cuda import CUDAKernel
        from ampfit.param_constraint import ParameterConstraint

        self.config = Config(config_file)
        self.kernel_config = self.config.build_all_index()
        self.kernel = CUDAKernel(self.kernel_config)

        # ck_map (list of param name tuples, one per partial wave)
        self.all_comb = self.config.get_ck_map()
        self.n_wave = len(self.all_comb)

        # Physical parameter dimensions
        self.n_m0 = len(self.config.m0_phys_name)
        self.n_g0 = len(self.config.g0_phys_name)

        # Constraint storage (built lazily)
        self._fixed_params = {}
        self._same_params = []
        self._scale_params = {}
        self._pc = None  # ParameterConstraint (built when needed)

        # Data holders (created by set_data / set_phsp)
        self.data_holder = None
        self.phsp_holder = None

        # Raw numpy data (needed for norm gradient computation)
        self._data_np = None
        self._phsp_np = None

        # Default physical params (used when not passed to get_nll)
        self.default_m0 = None
        self.default_g0 = None
        self.default_scalar = None

        # Bound transforms and fixed scalar values
        self._bound_transforms = {}        # {flat_idx: BoundTransform}
        self._fixed_scalars = {}           # {name: value} for fixed scalar params

    # ------------------------------------------------------------------
    # Constraint setup
    # ------------------------------------------------------------------
    def set_fixed(self, fixed_params):
        """Set fixed (constant) parameters: {name: value}.
        
        Works for all parameter types: ck (complex), m0/g0 (real), scalar (real).
        Fixed params are excluded from the VariableRegistry and flat vector x.
        """
        ck_fixed = {}
        scalar_fixed = {}
        scalar_names = {"gamma", "delta_gamma", "delta_m", "A_prod", "poqr", "poqi"}
        m0_names = set(self.config.m0_phys_name)
        g0_names = set(self.config.g0_phys_name)
        for name, val in fixed_params.items():
            if name in scalar_names or name in m0_names or name in g0_names:
                scalar_fixed[name] = float(val)
            else:
                ck_fixed[name] = val
        self._fixed_params = ck_fixed
        self._fixed_scalars = scalar_fixed
        self._rebuild_pc()

    def set_same(self, same_params):
        """Set same-parameter groups: [[name_a, name_b, ...], ...]."""
        self._same_params = list(same_params)
        self._rebuild_pc()

    def set_scale(self, scale_params):
        """Set scale factors: {name: scale_factor}."""
        self._scale_params = dict(scale_params)
        self._rebuild_pc()

    def _rebuild_pc(self):
        """Build or rebuild the ParameterConstraint."""
        from ampfit.param_constraint import ParameterConstraint
        self._pc = ParameterConstraint(
            self.all_comb,
            fixed_params=self._fixed_params,
            same_params=self._same_params,
            scale_params=self._scale_params,
        )
        self._rebuild_var_registry()

    @property
    def pc(self):
        """Lazily built ParameterConstraint (empty if not explicitly configured)."""
        if self._pc is None:
            self._rebuild_pc()
        return self._pc

    # ------------------------------------------------------------------
    # Data setup
    # ------------------------------------------------------------------
    def set_data(self, data):
        """Set data (real events) for negative log-likelihood.
        
        Args:
            data: dict with keys 'mass', 'q', 'angle', 'frac', 'time',
                  'weight', 'bkg' (optional).
        """
        self._data_np = data
        self.data_holder = self.kernel.load_data(data)

    def set_phsp(self, phsp):
        """Set phase-space data for normalization integral.
        
        Args:
            phsp: dict with same structure as data.
        """
        self._phsp_np = phsp
        self.phsp_holder = self.kernel.load_data(phsp)

    # ------------------------------------------------------------------
    # Bound constraints on parameters
    # ------------------------------------------------------------------
    def set_range(self, name, lo, hi):
        """Set a bound constraint on a parameter via sin-transform.
        
        Args:
            name: parameter name (base name or slot name with r/i suffix).
                  Examples: 'gamma', 'B->...total_0', 'B->...total_0r'
            lo: lower bound.
            hi: upper bound.
        """
        from ampfit.boundary import BoundTransform
        bt = BoundTransform(lo, hi)

        # Try base name first (e.g. 'gamma' → one slot, 'B->...total_0' → two slots)
        try:
            si, ei = self._var_registry.flat_index(name)
            for idx in range(si, ei):
                self._bound_transforms[idx] = bt
            return
        except KeyError:
            pass

        # Try flat slot name (e.g. 'B->...total_0r')
        flat_n = self._var_registry.flat_names
        for i, n in enumerate(flat_n):
            if n == name:
                self._bound_transforms[i] = bt
                return

        raise ValueError(
            f"Unknown parameter '{name}'. "
            f"Available: {self._var_registry.flat_names[:6]}... "
        )

    def _rebuild_var_registry(self):
        """Build or rebuild the VariableRegistry after constraint changes.
        
        All params (ck, mass, width, scalar) are added as variables by default.
        Fixed params are excluded from the registry.
        """
        self._var_registry = VariableRegistry()
        # Add free ck parameters (complex, from pc)
        for name in self.pc.free_param_names():
            self._var_registry.add_complex(name, ('ck', name))
        # Add m0 parameters (real), excluding fixed ones
        for name in self.config.m0_phys_name:
            if name not in self._fixed_scalars:
                self._var_registry.add_real(name, ('m0', name))
        # Add g0 parameters (real), excluding fixed ones
        for name in self.config.g0_phys_name:
            if name not in self._fixed_scalars:
                self._var_registry.add_real(name, ('g0', name))
        # Add scalar/time parameters (real), excluding fixed ones
        for name in ["gamma", "delta_gamma", "delta_m", "A_prod", "poqr", "poqi"]:
            if name not in self._fixed_scalars:
                self._var_registry.add_real(name, ('scalar', name))

    def _n_flat_vars(self):
        """Total number of flat variables: ck vars + free time params."""
        return self._var_registry.n_flat

    def initial_values(self, seed=None):
        """Random initial guess for all free variables (ck + scalar).
        
        Returns:
            array of shape (n_flat,) matching free_param_names() length.
        """
        return self._var_registry.build_initial(seed=seed)

    def free_param_names(self):
        """Slot-level names of all free variables. Length matches x0.
        
        Complex vars: '{name}r', '{name}i'
        Real vars:    '{name}'
        
        Example: ['B->...total_0r', 'B->...total_0i', 'gamma', ...]
        """
        return self._var_registry.flat_names

    def set_default_params(self, m0=None, g0=None, scalar=None):
        """Set default physical parameters (used when not passed to get_nll)."""
        if m0 is not None:
            self.default_m0 = np.array(m0, dtype=np.float64)
        if g0 is not None:
            self.default_g0 = np.array(g0, dtype=np.float64)
        if scalar is not None:
            self.default_scalar = np.array(scalar, dtype=np.float64)

    def _check_data_loaded(self):
        """Raise if data or phsp not set."""
        if self.data_holder is None:
            raise RuntimeError("Data not set. Call set_data() first.")
        if self.phsp_holder is None:
            raise RuntimeError("Phase space not set. Call set_phsp() first.")

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------
    def _build_base_params(self, ck, m0, g0, scalar):
        """Build the params dict from components, using defaults for None."""
        if m0 is None:
            m0 = self.default_m0
        if g0 is None:
            g0 = self.default_g0
        if scalar is None:
            scalar = self.default_scalar
        # Auto-generate default arrays if still None
        if m0 is None:
            m0 = np.ones(self.n_m0, dtype=np.float64) * 0.8
        if g0 is None:
            g0 = np.ones(self.n_g0, dtype=np.float64) * 0.1
        if scalar is None:
            scalar = [0.6, 0.01, 0.506, 0.01, 0.9, 0.2]
        return {"ck": ck, "m0": m0, "g0": g0, "scalar": list(scalar)}

    def _compute_norm_derivative(self, norm, P, data):
        """Compute d(NLL)/d(norm) from kernel forward outputs.
        
        NLL = -sum(weight * log(P/norm + bkg))
        
        d(NLL)/d(norm) = sum(weight * P / (norm * (P + bkg * norm)))
        """
        weight = data["weight"]
        bkg = data.get("bkg", 0.0)
        if np.isscalar(bkg):
            bkg = np.full_like(weight, bkg)
        denom = norm * (P + bkg * norm)
        return np.sum(weight * P / denom)

    def get_nll_raw(self, params):
        """Compute NLL from full params dict (no constraint transformation).
        
        Args:
            params: dict with keys 'ck', 'm0', 'g0', 'scalar'.
        
        Returns:
            (nll, grads) where grads is a dict with the same keys containing
            the total gradient including the norm contribution.
        """
        self._check_data_loaded()

        # 1. Norm from phase space (computed without norm factor)
        norm, norm_grads, _ = self.kernel.compute(
            params, self.phsp_holder, norm=None
        )
        norm = float(norm)  # ensure Python float for CFFI

        # 2. NLL from data (with norm)
        nll, grads, P = self.kernel.compute(
            params, self.data_holder, norm=norm
        )

        # 3. dNLL/dnorm
        dNLL_dnorm = self._compute_norm_derivative(norm, P, self._data_np)

        # 4. Combine gradients: total = direct + norm_chain
        total_grads = {}
        for key in grads:
            if key == "ck":
                # ck gradient: direct + norm_chain
                total_grads[key] = grads[key] + dNLL_dnorm * norm_grads[key]
            elif key == "scalar":
                total_grads[key] = grads[key] + dNLL_dnorm * norm_grads[key]
            elif key in ("m0", "g0"):
                total_grads[key] = grads[key] + dNLL_dnorm * norm_grads[key]
            else:
                total_grads[key] = grads[key]

        return nll, total_grads

    def get_nll(self, x, m0=None, g0=None):
        """Compute NLL and gradient w.r.t. the flat variable vector.
        
        The flat vector x contains all free variables:
          [ck0_r, ck0_θ, ..., m0_0, m0_1, ..., g0_0, g0_1, ..., scalar_0, ...]
        
        Args:
            x: flat variable vector (length = free_param_names()).
        
        Returns:
            (nll, grad_x) where grad_x has the same shape as x.
        """
        from ampfit.boundary import apply_bounds, apply_bound_grads

        # 1. Apply bound transforms
        x_mapped = apply_bounds(x, self._bound_transforms)

        # 2. Extract ck vars for ParameterConstraint.build_ck
        x_ck = self._var_registry.extract_by_target(x_mapped, 'ck')

        # 3. Build m0, g0, scalar arrays from x + fixed values
        scalar_names = ["gamma", "delta_gamma", "delta_m", "A_prod", "poqr", "poqi"]

        # Start from defaults or fallback values
        if self.default_m0 is not None:
            m0_arr = self.default_m0.copy()
        else:
            m0_arr = np.ones(self.n_m0, dtype=np.float64) * 0.8
        if self.default_g0 is not None:
            g0_arr = self.default_g0.copy()
        else:
            g0_arr = np.ones(self.n_g0, dtype=np.float64) * 0.1
        if self.default_scalar is not None:
            scalar_arr = list(self.default_scalar)
        else:
            scalar_arr = [0.6, 0.01, 0.506, 0.01, 0.9, 0.2]

        # Override with fixed values
        for name, val in self._fixed_scalars.items():
            if name in self.config.m0_phys_name:
                m0_arr[self.config.m0_phys_name.index(name)] = val
            elif name in self.config.g0_phys_name:
                g0_arr[self.config.g0_phys_name.index(name)] = val
            elif name in scalar_names:
                scalar_arr[scalar_names.index(name)] = val

        # Override with free values from x (those in the registry)
        vals = self._var_registry.extract_real_dict(x_mapped)
        for name, val in vals.items():
            if name in self.config.m0_phys_name:
                m0_arr[self.config.m0_phys_name.index(name)] = val
            elif name in self.config.g0_phys_name:
                g0_arr[self.config.g0_phys_name.index(name)] = val
            elif name in scalar_names:
                scalar_arr[scalar_names.index(name)] = val

        # 4. Build ck + params
        ck = self.pc.build_ck(x_ck)
        params = {"ck": ck, "m0": m0_arr, "g0": g0_arr, "scalar": scalar_arr}

        # 5. Compute NLL with norm
        nll, total_grads = self.get_nll_raw(params)

        # 6. Build flat gradient: ck, m0, g0, scalar parts
        # ck: pc.backprop_grad maps 448 partial waves → ck vars in r/θ format
        grad_ck = self.pc.backprop_grad(x_ck, total_grads["ck"])

        # m0, g0, scalar: pick gradients for free params by name
        grad_extra = []
        for target, names_list in [('m0', self.config.m0_phys_name),
                                    ('g0', self.config.g0_phys_name),
                                    ('scalar', scalar_names)]:
            arr = np.asarray(total_grads[target])
            for name in names_list:
                if name in self._var_registry._name_to_entry:
                    idx = list(names_list).index(name)
                    grad_extra.append(arr[idx])

        if len(grad_extra):
            grad_flat = np.concatenate([grad_ck] + [np.atleast_1d(g) for g in grad_extra])
        else:
            grad_flat = grad_ck.copy()

        # 7. Apply bound gradient correction
        grad_flat = apply_bound_grads(grad_flat, x, self._bound_transforms)

        return nll, grad_flat

    # ------------------------------------------------------------------
    # Convenience / utility
    # ------------------------------------------------------------------
    def free(self):
        """Free all GPU memory."""
        if self.data_holder is not None:
            self.data_holder.free()
        if self.phsp_holder is not None:
            self.phsp_holder.free()
        self.kernel.free()


# ====================================================================
# Example usage
# ====================================================================
if __name__ == "__main__":
    import numpy as np

    print("=" * 70)
    print("Fitter example")
    print("=" * 70)

    # Create fitter from config
    fitter = Fitter("config_angle.yml")
    print(f"n_wave = {fitter.n_wave}")

    # ---- Set constraints (like pw_cfit5_td6_fix29.py) ----
    all_params = set()
    for comb in fitter.all_comb:
        for p in comb:
            if isinstance(p, str):
                all_params.add(p)

    fixed_params = {}
    for p in sorted(all_params):
        if p.endswith("g_ls_0"):
            fixed_params[p] = 1.0 + 0.0j
    fitter.set_fixed(fixed_params)

    print(f"Free params: {fitter.pc.n_free_vars}")
    print(f"Free param names: {fitter.free_param_names()[:3]}...")

    # ---- Generate random test data ----
    n_events_data = 200
    n_events_phsp = 500

    data = {
        "mass": np.random.random((n_events_data, 48)),
        "q": np.random.random((n_events_data, 72)),
        "angle": np.random.random((n_events_data, 24, 3)),
        "frac": np.random.random((n_events_data,)),
        "time": np.random.random((n_events_data,)),
        "bkg": np.random.random((n_events_data,)) * 0.01,
        "weight": np.ones((n_events_data,)),
    }
    phsp = {
        "mass": np.random.random((n_events_phsp, 48)),
        "q": np.random.random((n_events_phsp, 72)),
        "angle": np.random.random((n_events_phsp, 24, 3)),
        "frac": np.random.random((n_events_phsp,)),
        "time": np.random.random((n_events_phsp,)),
        "bkg": np.zeros((n_events_phsp,)),
        "weight": np.ones((n_events_phsp,)),
    }

    fitter.set_data(data)
    fitter.set_phsp(phsp)
    fitter.set_default_params(
        m0=np.random.random(fitter.n_m0) + 2,
        g0=np.random.random(fitter.n_g0) + 0.1,
        scalar=[0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    )

    # ---- Test get_nll with constraint variables ----
    x0 = fitter.initial_values(seed=42)
    print(f"\nx0 shape: {x0.shape}")
    import time
    t0 = time.time()
    nll, grad_x = fitter.get_nll(x0)
    t = (time.time() - t0) * 1000
    print(f"NLL = {nll:.6f}, time = {t:.1f} ms")
    print(f"grad_x range: [{grad_x.min():.4f}, {grad_x.max():.4f}]")

    # ---- Verify gradient numerically ----
    eps = 1e-5
    errs = []
    for k in range(min(5, len(x0))):
        xp = x0.copy()
        xp[k] += eps
        nll_p, _ = fitter.get_nll(xp)
        xm = x0.copy()
        xm[k] -= eps
        nll_m, _ = fitter.get_nll(xm)
        num = (nll_p - nll_m) / (2 * eps)
        err = abs(grad_x[k] - num) / (max(abs(num), 1e-10) + 1e-10)
        errs.append(err)
        print(f"  x[{k:2d}]: ana={grad_x[k]:+.6e} num={num:+.6e} rel_err={err:.2e}")

    print(f"\nMax relative error (first 5): {max(errs):.2e}")
    print("✓ Fitter works!" if max(errs) < 0.05 else "⚠ Check gradient accuracy")

    fitter.free()
