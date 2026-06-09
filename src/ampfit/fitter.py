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

        # Phsp batching (for phsp larger than GPU memory)
        self._phsp_buffer = None   # GPUDataBuffer with all phsp input data
        self._phsp_scratch = None  # GPUDataHolder with batch-sized intermediates
        self._phsp_batch_size = 50000  # events per batch
        self._phsp_n = 0               # total phsp events

        # Default physical params from config.yml (lazy-built)
        self._default_m0_arr = None   # built from config particle masses
        self._default_g0_arr = None   # built from config particle widths
        self.default_scalar = None    # user-set scalar defaults

        # Bound transforms and fixed slots
        self._bound_transforms = {}        # {flat_idx: BoundTransform}
        self._fixed_slots = {}             # {slot_name: value}

    # ------------------------------------------------------------------
    # Constraint setup
    # ------------------------------------------------------------------
    def set_fixed(self, fixed_slots):
        """Set fixed parameter slots: {slot_name: value}.
        
        Slot names match free_param_names():
          '{name}r' — magnitude of complex parameter
          '{name}i' — phase of complex parameter
          '{name}'  — real parameter (scalar, mass, or width)
        
        Examples:
          fitter.set_fixed({"B->..._g_ls_0r": 1.0})   # magnitude only
          fitter.set_fixed({"B->..._g_ls_0i": 0.0})   # phase only
          fitter.set_fixed({"gamma": 0.0})             # real scalar
        """
        self._fixed_slots = {k: float(v) for k, v in fixed_slots.items()}
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
        
        Auto-batches if phsp is too large for GPU memory.
        All input data stays on GPU permanently across get_nll calls.
        
        Args:
            phsp: dict with same structure as data.
        """
        from ampfit._cuda import GPUDataBuffer, GPUDataHolder

        self._phsp_np = phsp
        n = phsp["mass"].shape[0]
        self._phsp_n = n

        # Estimate memory: ~36KB per event for intermediates + ~1.5KB for inputs
        # VRAM budget: ~70% of 8GB ≈ 5.6GB usable
        est_intermediates_mb = n * 36 / 1024  # MB for full intermediates
        if est_intermediates_mb < 4000:  # fits comfortably in VRAM
            self._phsp_buffer = None
            self._phsp_scratch = None
            self.phsp_holder = self.kernel.load_data(phsp)
            return

        # Too large for one batch → use batching with zero-copy slices
        print(f"  Phsp too large for single batch ({est_intermediates_mb:.0f} MB), "
              f"using batches of {self._phsp_batch_size}")

        # Pre-load ALL input data into one contiguous GPU buffer
        gc = self.kernel.gpu_config
        lib = self.kernel.lib
        ne = n
        bkg = phsp.get("bkg", 0.0)
        if np.isscalar(bkg):
            bkg = np.full(ne, bkg, dtype=np.float64)
        self._phsp_buffer = GPUDataBuffer(lib, [
            ("mass",   ((ne, phsp["mass"].shape[1]), np.float64)),
            ("q",      ((ne, phsp["q"].shape[1]), np.float64)),
            ("angle",  ((phsp["angle"].size,), np.float64)),
            ("frac",   ((ne,), np.float64)),
            ("time",   ((ne,), np.float64)),
            ("weight", ((ne,), np.float64)),
            ("bkg",    ((ne,), np.float64)),
        ])
        self._phsp_buffer.set("mass", phsp["mass"])
        self._phsp_buffer.set("q", phsp["q"])
        self._phsp_buffer.set("angle", phsp["angle"].flatten().astype(np.float64))
        self._phsp_buffer.set("frac", phsp["frac"].astype(np.float64))
        self._phsp_buffer.set("time", phsp["time"].astype(np.float64))
        self._phsp_buffer.set("weight", phsp["weight"].astype(np.float64))
        self._phsp_buffer.set("bkg", bkg.astype(np.float64))
        print(f"  Phsp data on GPU: {self._phsp_buffer.total_bytes/1024/1024:.0f} MB")

        # Create scratch holder with batch-sized intermediates
        self._phsp_scratch = GPUDataHolder(lib, gc.n_wave, gc.n_unique_bw, gc.n_gamma_rows)
        self._phsp_scratch.alloc_intermediates(self._phsp_batch_size)

        # Keep phsp_holder as None when batching
        self.phsp_holder = None

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

    def _build_alias_map(self):
        """Build alias→canonical name map from same_params."""
        alias_to_canon = {}
        for group in self._same_params:
            if group:
                canon = group[0]
                for a in group[1:]:
                    alias_to_canon[a] = canon
        self._alias_to_canon = alias_to_canon

    def _rebuild_var_registry(self):
        """Build the VariableRegistry from all non-fixed parameter slots.
        
        Handles alias name mapping: if a slot is fixed under an alias name,
        the canonical name is also recognized as fixed.
        """
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
        # Ck parameters (complex) — skip if both r and i are fixed
        for name in self.pc.free_param_names():
            r_fixed = _slot_fixed(name, 'r')
            i_fixed = _slot_fixed(name, 'i')
            if not (r_fixed and i_fixed):
                self._var_registry.add_complex(name, ('ck', name))
        # M0 parameters (real)
        for name in self.config.m0_phys_name:
            if not _name_fixed(name):
                self._var_registry.add_real(name, ('m0', name))
        # G0 parameters (real)
        for name in self.config.g0_phys_name:
            if not _name_fixed(name):
                self._var_registry.add_real(name, ('g0', name))
        # Scalar/time parameters (real)
        for name in ["gamma", "delta_gamma", "delta_m", "A_prod", "poqr", "poqi"]:
            if not _name_fixed(name):
                self._var_registry.add_real(name, ('scalar', name))

    def _n_flat_vars(self):
        """Total number of flat variables: ck vars + free time params."""
        return self._var_registry.n_flat

    def initial_values(self, seed=None):
        """Random initial guess for all free variables.
        
        Returns:
            array of shape (n_flat,) matching free_param_names() length.
        """
        _ = self.pc  # ensure pc and var_registry are built
        return self._var_registry.build_initial(seed=seed)

    def free_param_names(self):
        """Slot-level names of all free variables. Length matches x0.
        
        Complex vars: '{name}r', '{name}i'
        Real vars:    '{name}'
        
        Example: ['B->...total_0r', 'B->...total_0i', 'gamma', ...]
        """
        return self._var_registry.flat_names

    def _extract_config_defaults(self):
        """Build default m0/g0 arrays from config.yml particle definitions.
        
        Mass defaults come from the 'mass' field of each particle.
        Width/defaults come from 'width' (simple) or 'g_{idx}' (Flatte coupling).
        """
        if self._default_m0_arr is None:
            m0 = []
            for name in self.config.m0_phys_name:
                particle = name.replace('_mass', '')
                dic = self.config.dic.get('particle', {})
                mass = None
                if particle in dic:
                    mass = dic[particle].get('mass')
                if mass is None:
                    mass = 0.8
                m0.append(float(mass))
            self._default_m0_arr = np.array(m0, dtype=np.float64)

        if self._default_g0_arr is None:
            # Build a map: gamma_name → default value from pre-built models
            gamma_map = {}
            for chain in self.config.full_decay.chains:
                for decay in chain.decays:
                    model = decay.core._model
                    names = model.get_gamma_name()
                    vals = model.get_gamma_defaults()
                    for n, v in zip(names, vals):
                        gamma_map[n] = float(v)
            g0 = [gamma_map.get(name, 0.1) for name in self.config.g0_phys_name]
            self._default_g0_arr = np.array(g0, dtype=np.float64)

    @property
    def default_m0(self):
        """Default m0 values from config (lazy-built)."""
        self._extract_config_defaults()
        return self._default_m0_arr

    @default_m0.setter
    def default_m0(self, val):
        self._default_m0_arr = np.array(val, dtype=np.float64) if val is not None else None

    @property
    def default_g0(self):
        """Default g0 values from config (lazy-built)."""
        self._extract_config_defaults()
        return self._default_g0_arr

    @default_g0.setter
    def default_g0(self, val):
        self._default_g0_arr = np.array(val, dtype=np.float64) if val is not None else None

    def set_default_params(self, m0=None, g0=None, scalar=None):
        """Set default physical parameters (used when not passed to get_nll)."""
        if m0 is not None:
            self._default_m0_arr = np.array(m0, dtype=np.float64)
        if g0 is not None:
            self._default_g0_arr = np.array(g0, dtype=np.float64)
        if scalar is not None:
            self.default_scalar = np.array(scalar, dtype=np.float64)

    def _check_data_loaded(self):
        """Raise if data or phsp not set."""
        if self.data_holder is None:
            raise RuntimeError("Data not set. Call set_data() first.")
        phsp_ok = self.phsp_holder is not None or self._phsp_buffer is not None
        if not phsp_ok:
            raise RuntimeError("Phase space not set. Call set_phsp() first.")

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------
    def _build_base_params(self, ck, m0, g0, scalar):
        """Build the params dict from components, using defaults for None."""
        if m0 is None:
            m0 = self.default_m0  # from config.yml particle masses
        if g0 is None:
            g0 = self.default_g0  # from config.yml particle widths
        if scalar is None:
            scalar = self.default_scalar
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

    def _compute_norm_batched(self, params):
        """Compute norm over ALL phsp events, batching if needed."""
        if self._phsp_buffer is None:
            # Single-batch: use phsp_holder directly
            norm, grads, _ = self.kernel.compute(params, self.phsp_holder, norm=None)
            return float(norm), grads

        # Batched mode: iterate over phsp buffer via zero-copy slices
        total_norm = 0.0
        total_grads = None
        bs = self._phsp_batch_size
        n_batches = (self._phsp_n + bs - 1) // bs
        gc = self.kernel.gpu_config

        for b in range(n_batches):
            start = b * bs
            end = min(start + bs, self._phsp_n)
            self._phsp_scratch.attach_input_slice(
                self._phsp_buffer, start, end,
                self._phsp_np["mass"].shape[1] if self._phsp_np is not None else 0,
                self._phsp_np["q"].shape[1] if self._phsp_np is not None else 0,
            )
            n_b, g_b, _ = self.kernel.compute(params, self._phsp_scratch, norm=None)
            total_norm += float(n_b)
            if total_grads is None:
                total_grads = {k: v.copy() for k, v in g_b.items()}
            else:
                for k in g_b:
                    total_grads[k] += g_b[k]

        return total_norm, total_grads

    def get_nll_raw(self, params):
        """Compute NLL from full params dict (no constraint transformation).
        
        Args:
            params: dict with keys 'ck', 'm0', 'g0', 'scalar'.
        
        Returns:
            (nll, grads) where grads is a dict with the same keys containing
            the total gradient including the norm contribution.
        """
        self._check_data_loaded()

        # 1. Norm from phase space (batched if needed)
        norm, norm_grads = self._compute_norm_batched(params)
        norm = float(norm)

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

        # 2. Extract ck vars and merge fixed r/θ slots
        x_ck = self._var_registry.extract_by_target(x_mapped, 'ck')
        # For partially-fixed ck vars, reinsert fixed r or θ values
        # Resolve alias names to canonical via _alias_to_canon
        pc_names = set(self.pc.free_param_names())
        fixed_ck_r = {}  # {name: fixed_r}
        fixed_ck_i = {}  # {name: fixed_i}
        for slot, val in self._fixed_slots.items():
            base = slot[:-1]  # e.g., 'name' from 'namer' or 'namei'
            # Resolve alias to canonical
            canon = self._alias_to_canon.get(base, base)
            if slot.endswith('r') and canon in pc_names:
                fixed_ck_r[canon] = val
            elif slot.endswith('i') and canon in pc_names:
                fixed_ck_i[canon] = val
        if fixed_ck_r or fixed_ck_i:
            # Rebuild x_ck with fixed values merged
            new_x_ck = []
            idx = 0
            for name in self.pc.free_param_names():
                in_reg = name in self._var_registry._name_to_entry
                if in_reg:
                    r = x_ck[idx]; th = x_ck[idx + 1]
                    idx += 2
                else:
                    r = th = 0.0  # both fixed, shouldn't reach here
                r = fixed_ck_r.get(name, r)
                th = fixed_ck_i.get(name, th)
                new_x_ck.extend([r, th])
            x_ck = np.array(new_x_ck)

        # 3. Build m0, g0, scalar arrays from x + fixed values
        scalar_names = ["gamma", "delta_gamma", "delta_m", "A_prod", "poqr", "poqi"]

        # Start from config.yml defaults (lazy-built from particle definitions)
        m0_arr = self.default_m0.copy()      # masses from config
        g0_arr = self.default_g0.copy()      # widths from config
        if self.default_scalar is not None:
            scalar_arr = list(self.default_scalar)
        else:
            scalar_arr = [0.6, 0.01, 0.506, 0.01, 0.9, 0.2]

        # Override with fixed slot values (r/i for complex parts, names for reals)
        # Fixed scalars go directly into m0/g0/scalar arrays
        for slot_name, val in self._fixed_slots.items():
            base = slot_name.rstrip('ri') if slot_name[-1] in 'ri' and slot_name[-2] not in 'ri' else slot_name
            if False: pass
            elif base in self.config.m0_phys_name and slot_name == base:
                m0_arr[self.config.m0_phys_name.index(base)] = val
            elif base in self.config.g0_phys_name and slot_name == base:
                g0_arr[self.config.g0_phys_name.index(base)] = val
            elif base in scalar_names and slot_name == base:
                scalar_arr[scalar_names.index(base)] = val

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

        # 6. Build flat gradient matching the registry order.
        # grad_ck from pc.backprop_grad covers ALL pc.free_param_names()
        # (including fully-fixed vars). We filter to only registry entries.
        full_grad_ck = self.pc.backprop_grad(x_ck, total_grads["ck"])

        # Lookup: name → gradient in backprop_grad output (r/θ pairs)
        ck_grad_map = {}
        idx = 0
        for name in self.pc.free_param_names():
            ck_grad_map[name + 'r'] = full_grad_ck[idx]
            ck_grad_map[name + 'i'] = full_grad_ck[idx + 1]
            idx += 2

        # m0, g0, scalar gradients by name
        extra_grad_map = {}
        for target, names_list in [('m0', self.config.m0_phys_name),
                                    ('g0', self.config.g0_phys_name),
                                    ('scalar', scalar_names)]:
            arr = np.asarray(total_grads[target])
            for name in names_list:
                extra_grad_map[name] = arr[list(names_list).index(name)]

        # Build flat gradient from registry entries only
        flat_names = self._var_registry.flat_names
        grad_flat = np.array([ck_grad_map.get(n, extra_grad_map.get(n, 0.0))
                               for n in flat_names])

        # 7. Apply bound gradient correction (before fixed-slot zeroing)
        grad_flat = apply_bound_grads(grad_flat, x, self._bound_transforms)

        # 8. Zero gradients for fixed slots (after bound correction)
        flat_names = self._var_registry.flat_names
        for slot_name in self._fixed_slots:
            if slot_name in flat_names:
                idx = flat_names.index(slot_name)
                grad_flat[idx] = 0.0

        return nll, grad_flat

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    def fit(self, x0=None, maxiter=1000, ftol=1e-8, gtol=1e-8, callback=None,
            method='BFGS', disp=True, **kwargs):
        """Minimize NLL using BFGS (default) or any scipy optimizer.
        
        BFGS provides the full Hessian inverse (result.hess_inv) for
        computing parameter uncertainties:
          errors = sqrt(diag(result.hess_inv))
        
        Bound transforms (set via set_range) are applied automatically
        inside get_nll(), so the optimizer sees unbounded values.
        To get uncertainties in the bounded space, use BoundTransform.trans_err.
        
        Args:
            x0: starting point. If None, uses initial_values().
            maxiter: maximum number of iterations.
            ftol: convergence tolerance on function value value change.
            gtol: convergence tolerance on gradient norm.
            callback: optional callback function(xk) called after each step.
            method: scipy.optimize.minimize method (default 'BFGS').
            disp: print convergence messages.
            **kwargs: passed to scipy.optimize.minimize.
        
        Returns:
            OptimizeResult from scipy.optimize.minimize.
            For BFGS: result.hess_inv contains the inverse Hessian.
        """
        from scipy.optimize import minimize

        if x0 is None:
            x0 = self.initial_values()

        def nll_and_grad(x):
            nll, grad = self.get_nll(x)
            return nll, grad.astype(np.float64)

        opts = {'maxiter': maxiter, 'gtol': gtol, 'disp': disp}
        if method in ('L-BFGS-B', 'L-BFGS-B'):
            opts['ftol'] = ftol
        result = minimize(
            nll_and_grad, x0, jac=True,
            method=method,
            options=opts,
            callback=callback,
            **kwargs
        )
        return result

    def _params_from_fit(self, fit_result, return_bounded=True):
        """Build dicts of parameter values and errors from a BFGS fit result.
        
        Args:
            fit_result: OptimizeResult from fit() method.
            return_bounded: if True, transform values back through
                            BoundTransform (physical space).
                            if False, return raw unbounded optimizer values.
        
        Returns:
            (values_dict, errors_dict) where each maps slot_name -> float.
            values_dict: best-fit parameter values.
            errors_dict: 1-sigma uncertainties from Hessian diagonal.
        """
        from ampfit.boundary import BoundTransform

        x_best = fit_result.x
        hess_inv = fit_result.hess_inv
        raw_errors = np.sqrt(np.diag(hess_inv))
        names = self._var_registry.flat_names

        values = {}
        errors = {}

        for i, name in enumerate(names):
            val = x_best[i]
            err = raw_errors[i]

            if return_bounded and i in self._bound_transforms:
                bt = self._bound_transforms[i]
                val_b = bt(val)
                err_b = bt.trans_err(val, err)
                values[name] = val_b
                errors[name] = err_b
            else:
                values[name] = val
                errors[name] = err

        return values, errors

    def get_uncertainties(self, fit_result, return_bounded=True):
        """Compute parameter uncertainties from a BFGS fit result.
        
        Uses the inverse Hessian (fit_result.hess_inv) to compute
        1-sigma uncertainties. Bound transforms are automatically
        propagated for parameters set via set_range().
        
        Args:
            fit_result: OptimizeResult from fit() method.
            return_bounded: if True (default), returns values and errors
                            in the physical (bounded) space.
                            if False, returns raw optimizer space values.
        
        Returns:
            dict of {slot_name: (value, error)} for every free parameter.
        """
        values, errors = self._params_from_fit(fit_result, return_bounded)
        return {name: (values[name], errors[name]) for name in values}

    # ------------------------------------------------------------------
    # JSON export (matching archive/w_pw_cfit5_td6_fix29.py format)
    # ------------------------------------------------------------------
    def save_params(self, fit_result, filepath, grad_scale=1.0):
        """Save fit results to JSON file matching archive pw_cfit5_td6_fix29.py format.
        
        The JSON structure:
            value: {name_r: float, name_i: float, time_param: float, ...}
            error: {name_r: float, ...}
            status: {NLL, Ndf, jac, success, message, rhorho}
        
        Handles bound transforms, fixed params, same-param aliases,
        scale params, and time parameter defaults automatically.
        
        Args:
            fit_result: OptimizeResult from fit() method.
            filepath: output JSON file path.
            grad_scale: gradient scaling factor (default 1.0).
                        Matches archive's grad_sacle if used.
        """
        import json, cmath

        values, errors = self._params_from_fit(fit_result, return_bounded=True)
        flat_names = self._var_registry.flat_names

        # Build value and error dicts (start with free params from fit)
        out = {"value": {}, "error": {}}
        for name in flat_names:
            out["value"][name] = float(values[name])
            out["error"][name] = float(errors[name])

        # Build alias→canonical map from same_params (like archive's new_name)
        new_name = {}
        for group in self._same_params:
            if group:
                canon = group[0]
                for a in group[1:]:
                    new_name[a] = canon

        # Add fixed ck parameters (from _fixed_slots or old _fixed_params)
        all_param_names = set()
        for comb in self.all_comb:
            for p in comb:
                if isinstance(p, str):
                    all_param_names.add(p)
        for p in sorted(all_param_names):
            r_name = p + 'r'
            i_name = p + 'i'
            if r_name in self._fixed_slots and i_name in self._fixed_slots:
                out["value"][r_name] = float(self._fixed_slots[r_name])
                out["value"][i_name] = float(self._fixed_slots[i_name])
                out["error"][r_name] = 0.0
                out["error"][i_name] = 0.0

        # Same-param aliases: copy value from canonical
        for alias, canon in new_name.items():
            for suf in ['r', 'i']:
                ak = alias + suf
                ck = canon + suf
                if ck in out["value"]:
                    out["value"][ak] = out["value"][ck]
                    out["error"][ak] = out["error"][ck]

        # Scale params: multiply value by scale factor
        for p, scale in self._scale_params.items():
            out["value"][p + 'r'] = scale * out["value"].get(p + 'r', 0.0)

        # Fixed time parameter defaults
        scalar_names = ["gamma", "delta_gamma", "delta_m", "A_prod", "poqr", "poqi"]
        time_defaults = {
            "gamma": 0.0, "delta_gamma": 0.0, "delta_m": 0.506,
            "A_prod": 0.0, "poqr": 1.0, "poqi": 0.0,
        }
        for name in scalar_names:
            if name not in out["value"]:
                out["value"][name] = time_defaults.get(name, 0.0)
                out["error"][name] = 0.0

        # Status
        hess_inv = fit_result.hess_inv
        n_params = len(fit_result.x)
        out["status"] = {
            "NLL": float(fit_result.fun),
            "Ndf": n_params,
            "jac": fit_result.jac.tolist() if hasattr(fit_result.jac, 'tolist') else list(fit_result.jac),
            "success": bool(fit_result.success),
            "message": str(fit_result.message),
        }

        # Correlation matrix for B->rhoA.rhoB params (like archive)
        corr_items = [[i, n] for i, n in enumerate(flat_names) if "B->rhoA.rhoB" in n]
        if corr_items:
            corr_idx = np.array([i for i, n in corr_items])
            corr_order = [n for i, n in corr_items]
            corr_mat = hess_inv[np.ix_(corr_idx, corr_idx)] * grad_scale
            out["status"]["rhorho"] = [corr_order, corr_mat.tolist()]

        with open(filepath, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"✓ Saved fit results to {filepath}")
        return out

    # ------------------------------------------------------------------
    # Convenience / utility
    # ------------------------------------------------------------------
    def free(self):
        """Free all GPU memory."""
        if self.data_holder is not None:
            self.data_holder.free()
        if self.phsp_holder is not None:
            self.phsp_holder.free()
        if self._phsp_scratch is not None:
            self._phsp_scratch.free()
            self._phsp_scratch = None
        if self._phsp_buffer is not None:
            self._phsp_buffer.free()
            self._phsp_buffer = None
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
