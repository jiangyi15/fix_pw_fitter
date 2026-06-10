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

import time
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

        # Constraint storage (built lazily, additive by default)
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

        # Background / purity parameters (computed during set_phsp/set_data)
        self._purity = None           # purity fraction from config
        self._N_b = None              # sum(phsp_weight * bkg) / sum(phsp_weight)
        self._bkg_scale = None        # (1-purity)/purity / N_b for bkg scaling
        self._log_purity_const = 0.0  # -log(purity) * sum(data_weight)

        # Bound transforms and fixed slots
        self._bound_transforms = {}        # {flat_idx: BoundTransform}
        self._fixed_slots = {}             # {slot_name: value}

    # ------------------------------------------------------------------
    # Constraint setup — all additive/interactive by default
    # ------------------------------------------------------------------
    def set_fixed(self, fixed_slots, reset=False):
        """Fix parameter slot(s) to a constant value. *Additive* — call multiple times.

        Slot names match free_param_names():
          '{name}r' — magnitude of complex parameter
          '{name}i' — phase of complex parameter
          '{name}'  — real parameter (scalar, mass, or width)

        Pass ``reset=True`` to clear all previously fixed slots first.

        Examples::

            fitter.set_fixed({"B->..._g_ls_0r": 1.0})     # magnitude fixed
            fitter.set_fixed({"B->..._g_ls_0i": 0.0})     # phase fixed separately
            fitter.set_fixed({"gamma": 0.0})               # real scalar fixed
            fitter.set_fixed({"delta_m": 0.506, "A_prod": 0.0})  # multiple at once
        """
        if reset:
            self._fixed_slots = {}
        self._fixed_slots.update({k: float(v) for k, v in fixed_slots.items()})
        self._rebuild_pc()

    def set_same(self, same_params, reset=False):
        """Share a value across parameter names. *Additive* — call multiple times.

        Each group is a list of parameter names that share a single optimised value
        (the first name is the canonical representative).

        Pass ``reset=True`` to clear all previously registered groups first.

        Example::

            fitter.set_same([["B->a1p->rhoA_g_ls_1", "B->a1m->rhoA_g_ls_1"]])
            fitter.set_same([["B->a1p->f0_total_0", "B->a1m->f0_total_0"]])
        """
        if reset:
            self._same_params = []
        self._same_params.extend(list(same_params))
        self._rebuild_pc()

    def set_scale(self, scale_params, reset=False):
        """Multiply a parameter by a real scale factor. *Additive* — call multiple times.

        The scale is applied to the *original* name (before alias→canonical mapping),
        matching the archive ``pw_cfit5_td6_fix29.py`` convention.

        Pass ``reset=True`` to clear all previously registered scale factors first.

        Example::

            fitter.set_scale({"B->a1m->rhoA_g_ls_0": -1})   # flip sign
        """
        if reset:
            self._scale_params = {}
        self._scale_params.update(dict(scale_params))
        self._rebuild_pc()

    def set_free(self, name):
        """Unfix a previously fixed parameter so it becomes free again.

        Removes *name* (and the corresponding ``'r'`` / ``'i'`` slots for complex
        parameters) from the fixed set.  Also removes *name* from any
        same‑parameter group or scale mapping.

        Parameters not previously set via any ``set_*`` method are free by
        default, so this is only needed to reverse an earlier ``set_fixed``,
        ``set_same``, or ``set_scale`` call.

        Example::

            fitter.set_fixed({"gamma": 0.0})
            fitter.set_free("gamma")           # now free again
        """
        # Remove from fixed slots
        name_r = name + 'r'
        name_i = name + 'i'
        for key in (name, name_r, name_i):
            self._fixed_slots.pop(key, None)

        # Remove from same-parameter groups
        self._same_params = [
            g for g in self._same_params if name not in g
        ]

        # Remove from scale factors
        self._scale_params.pop(name, None)

        self._rebuild_pc()

    def unset_range(self, name):
        """Remove the bound (arctan) constraint on *name*, making it unbounded.

        Example::

            fitter.set_range("gamma", -0.3, 0.3)
            fitter.unset_range("gamma")      # back to unbounded
        """
        try:
            si, ei = self.var_registry.flat_index(name)
            for idx in range(si, ei):
                self._bound_transforms.pop(idx, None)
            return
        except KeyError:
            pass

        flat_n = self.var_registry.flat_names
        for i, n in enumerate(flat_n):
            if n == name:
                self._bound_transforms.pop(i, None)
                return

        # Not an error — silently ignore unknown names (they weren't bound)

    def _rebuild_pc(self):
        """Build or rebuild the ParameterConstraint."""
        from ampfit.param_constraint import ParameterConstraint
        self._pc = ParameterConstraint(
            self.all_comb,
            fixed_params={},          # complex-level fixes handled by _fixed_slots in get_nll()
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
        
        Reads purity from config and scales bkg:
          bkg_scaled = (1-purity)/purity * bkg_raw / N_b
        where N_b is the average phsp background (computed in set_phsp).
        
        Also stores -log(purity)*sum(weights) as a constant NLL offset.
        
        Args:
            data: dict with keys 'mass', 'q', 'angle', 'frac', 'time',
                  'weight', 'bkg' (optional).
        """
        # Read purity from config
        purity = self.config.dic.get('purity')
        if purity is None:
            purity = self.config.dic.get('data', {}).get('purity',
                     self.config.dic.get('data', {}).get('bg_frac', None))
        self._purity = float(purity) if purity is not None else None

        # Copy data and scale bkg if we have purity + phsp background
        data = dict(data)
        if self._purity is not None and self._N_b is not None and self._N_b > 0:
            bkg_raw = data.get("bkg", 0.0)
            if np.isscalar(bkg_raw):
                bkg_raw = np.full(data["mass"].shape[0], bkg_raw, dtype=np.float64)
            p = self._purity
            self._bkg_scale = (1.0 - p) / p / self._N_b
            data["bkg"] = bkg_raw * self._bkg_scale
            self._log_purity_const = -np.log(p) * np.sum(data.get("weight", np.ones(data["mass"].shape[0])))
        else:
            self._bkg_scale = None
            self._log_purity_const = 0.0

        self._data_np = data
        self.data_holder = self.kernel.load_data(data)

    def set_phsp(self, phsp):
        """Set phase-space data for normalization integral.
        
        Normalizes phsp weights to sum to 1 and computes N_b = mean bkg
        (weighted average of phsp bkg). These are used by set_data for
        the purity-based background scaling.
        
        Auto-batches if phsp is too large for GPU memory.
        All input data stays on GPU permanently across get_nll calls.
        
        Args:
            phsp: dict with same structure as data.
        """
        from ampfit._cuda import GPUDataBuffer, GPUDataHolder

        # Normalize phsp weights to sum to 1
        phsp = dict(phsp)
        w = phsp.get("weight", np.ones(phsp["mass"].shape[0]))
        w_sum = np.sum(w)
        if w_sum > 0:
            phsp["weight"] = w / w_sum

        # Compute N_b = weighted average of bkg over phsp
        b = phsp.get("bkg", np.zeros(phsp["mass"].shape[0]))
        if np.isscalar(b):
            b = np.full(phsp["mass"].shape[0], b, dtype=np.float64)
        self._N_b = float(np.sum(phsp["weight"] * b)) if w_sum > 0 else 0.0

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
            si, ei = self.var_registry.flat_index(name)
            for idx in range(si, ei):
                self._bound_transforms[idx] = bt
            return
        except KeyError:
            pass

        # Try flat slot name (e.g. 'B->...total_0r')
        flat_n = self.var_registry.flat_names
        for i, n in enumerate(flat_n):
            if n == name:
                self._bound_transforms[i] = bt
                return

        raise ValueError(
            f"Unknown parameter '{name}'. "
            f"Available: {self.var_registry.flat_names[:6]}... "
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

    def values_from_dict(self, data):
        """Build the flat x vector from a save_params JSON dict.
        
        Reads physical (bounded) values from the 'value' section,
        reverses archive-style scale factors, and inverts bound
        transforms to get optimizer-space x.
        
        Args:
            data: dict from save_params() JSON (keys 'value', 'error', ...).
        
        Returns:
            flat vector x suitable for get_nll() or fit().
        """
        _ = self.pc
        names = self._var_registry.flat_names
        values = data.get("value", data) if isinstance(data, dict) else data
        x = np.empty(len(names))

        for i, name in enumerate(names):
            if name in values:
                val = float(values[name])
                # Reverse archive-style scale on r-slots
                for p, s in self._scale_params.items():
                    if name == p + 'r' and s != 0:
                        val /= s
                if i in self._bound_transforms:
                    bt = self._bound_transforms[i]
                    val = bt.inverse(val)
                x[i] = val
            else:
                x[i] = self._var_registry.build_initial()[i]

        return x

    @property
    def var_registry(self):
        """Lazily-built VariableRegistry (built by _rebuild_pc)."""
        _ = self.pc  # trigger lazy build
        return self._var_registry

    def free_param_names(self):
        """Slot-level names of all free variables. Length matches x0.
        
        Complex vars: '{name}r', '{name}i'
        Real vars:    '{name}'
        
        Example: ['B->...total_0r', 'B->...total_0i', 'gamma', ...]
        """
        return self.var_registry.flat_names

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

        # 4. Add purity constant: -log(purity) * sum(weight)
        # Kernel computes -w*log(P/norm + bkg_scaled).
        # With bkg_scaled = (1-purity)/purity * bkg_raw/N_b, we get:
        #   Q_kernel = -w*log(P/norm + (1-p)/p * b/N_b)
        #            = -w*log(p*P/norm + (1-p)*b/N_b) + w*log(p)
        # So Q_true = Q_kernel - w*log(purity)
        # NLL_true = NLL_kernel - log(purity)*sum(w)
        nll = nll + self._log_purity_const

        # 5. Combine gradients: total = direct + norm_chain
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

        # Cache the last evaluation — the callback reads from here
        # instead of re-running the expensive GPU forward+backward pass.
        _last = {'nll': None, 'grad': None}

        def nll_and_grad(x):
            nll, grad = self.get_nll(x)
            _last['nll'] = nll
            _last['grad'] = grad
            return nll, grad.astype(np.float64)

        # Default callback: print NLL + timing at each iteration
        class IterTracker:
            def __init__(self):
                self.n = 0
                self.t_start = time.time()
                self.t_last = time.time()
            def __call__(self, xk):
                self.n += 1
                t_now = time.time()
                dt = t_now - self.t_last
                t_elapsed = t_now - self.t_start
                self.t_last = t_now
                gn = np.linalg.norm(_last['grad'])
                print(f"  iter {self.n:4d}: NLL = {_last['nll']:.11f}, |grad| = {gn:.4e}, +{dt:.2f}s [{t_elapsed:.1f}s]")
                return False

        tracker = IterTracker()
        user_cb = callback
        if user_cb is None:
            combined_cb = tracker
        else:
            def combined_cb(xk):
                tracker(xk)
                user_cb(xk)

        opts = {'maxiter': maxiter, 'gtol': gtol, 'disp': disp}
        if method in ('L-BFGS-B', 'L-BFGS-B'):
            opts['ftol'] = ftol
        result = minimize(
            nll_and_grad, x0, jac=True,
            method=method,
            options=opts,
            callback=combined_cb,
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
    # Plotting
    # ------------------------------------------------------------------
    def plot(self, result=None, x=None, params=None, prefix="plots/",
             n_bins=50, cols=4, figsize=(15, 10), show=False):
        """Plot data vs phsp distributions, saving figures to a directory.
        
        For each variable, shows two histograms:
          - Data (weighted by data weight)
          - Phsp weighted by P × phsp_weight (the model prediction)
        
        Figures are saved as:
          {prefix}mass.png      — all mass columns (48 subplots)
          {prefix}angles.png    — all angle positions × components (72 subplots)
          {prefix}time.png      — time distribution
        
        Args:
            result: OptimizeResult from Fitter.fit(), or a flat array x.
            x: flat variable vector (alternative to result).
            params: full params dict (instead of result/x).
            prefix: directory path for output figures (default "plots/").
                    Created automatically if it doesn't exist.
            n_bins: number of histogram bins.
            cols: number of columns in the subplot grid.
            figsize: figure width; height is auto-scaled per figure.
            show: if True, call plt.show() in addition to saving.
        """
        import matplotlib.pyplot as plt
        import os
        from ampfit.boundary import apply_bounds

        # Resolve the flat x vector
        if params is not None:
            pass  # use params directly
        elif result is not None:
            if isinstance(result, np.ndarray):
                x = result
            elif hasattr(result, 'x'):
                x = result.x
            else:
                x = result
        if x is None and params is None:
            raise ValueError("Provide result, x, or params.")
        if x is not None:
            x_mapped = apply_bounds(x, self._bound_transforms)
            raw_ck = self._var_registry.extract_by_target(x_mapped, 'ck')
            pc_names = set(self.pc.free_param_names())
            f_ck_r = {}
            f_ck_i = {}
            for slot, val in self._fixed_slots.items():
                base = slot[:-1]
                canon = self._alias_to_canon.get(base, base)
                if slot.endswith('r') and canon in pc_names:
                    f_ck_r[canon] = val
                elif slot.endswith('i') and canon in pc_names:
                    f_ck_i[canon] = val
            if f_ck_r or f_ck_i:
                new_x = []; idx = 0
                for name in self.pc.free_param_names():
                    in_reg = name in self._var_registry._name_to_entry
                    r = raw_ck[idx] if in_reg else 0.0
                    th = raw_ck[idx + 1] if in_reg else 0.0
                    if in_reg: idx += 2
                    new_x.extend([f_ck_r.get(name, r), f_ck_i.get(name, th)])
                raw_ck = np.array(new_x)
            ck = self.pc.build_ck(raw_ck)
            params = self._build_base_params(ck, None, None, None)

        # Compute norm and probabilities (handles batched phsp)
        norm, _ = self._compute_norm_batched(params)
        norm = float(norm)
        _, _, P_data = self.kernel.compute(params, self.data_holder, norm=norm)

        # Compute P_phsp (handle batched mode)
        if self._phsp_buffer is not None:
            # Batched mode: compute P per batch, concatenate
            P_phsp_list = []
            bs = self._phsp_batch_size
            n_batches = (self._phsp_n + bs - 1) // bs
            for b in range(n_batches):
                start = b * bs
                end = min(start + bs, self._phsp_n)
                self._phsp_scratch.attach_input_slice(
                    self._phsp_buffer, start, end,
                    self._phsp_np["mass"].shape[1], self._phsp_np["q"].shape[1])
                _, _, P_b = self.kernel.compute(params, self._phsp_scratch, norm=None)
                P_phsp_list.append(P_b[:self._phsp_scratch.n_events])
            P_phsp = np.concatenate(P_phsp_list)
        else:
            _, _, P_phsp = self.kernel.compute(params, self.phsp_holder, norm=None)

        data_np = self._data_np
        phsp_np = self._phsp_np
        dw = data_np["weight"]
        pw = phsp_np["weight"] * P_phsp
        ne_d, ne_p = len(P_data), len(P_phsp)

        # Ensure output directory
        os.makedirs(prefix, exist_ok=True)

        def _make_hist(ax, label, d, p):
            lo = min(d.min(), p.min())
            hi = max(d.max(), p.max())
            if hi - lo < 1e-12:
                hi = lo + 1.0
            bins = np.linspace(lo, hi, n_bins + 1)
            bin_w = bins[1] - bins[0]
            bin_c = (bins[:-1] + bins[1:]) / 2

            # Data: weighted counts with sqrt(sum(w²)) error bars
            data_y, _ = np.histogram(d, bins=bins, weights=dw)
            data_w2, _ = np.histogram(d, bins=bins, weights=dw ** 2)
            data_err = np.sqrt(data_w2)

            ax.errorbar(bin_c, data_y, yerr=data_err, fmt='o',
                        color='C0', label='data', markersize=3, capsize=2)

            # Phsp: histogram weighted by pw, scaled to match data integral
            phsp_y, _ = np.histogram(p, bins=bins, weights=pw)
            data_total = data_y.sum()
            phsp_total = phsp_y.sum()
            scale = data_total / phsp_total if phsp_total > 0 else 1.0
            ax.bar(bin_c, phsp_y * scale, width=bin_w * 0.9,
                   alpha=0.4, color='C1', label='phsp×P', align='center')

            ax.set_xlabel(label, fontsize=7)
            ax.tick_params(labelsize=6)

        def _save_figure(fig, name):
            path = os.path.join(prefix, name)
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  saved {path}")

        # ---- Mass ----
        n_mass = data_np["mass"].shape[1]
        n_rows = (n_mass + cols - 1) // cols
        fig, axes = plt.subplots(n_rows, cols,
            figsize=(figsize[0], 2.5 * n_rows), squeeze=False)
        for i in range(n_mass):
            _make_hist(axes.flatten()[i], f"mass[{i}]",
                       data_np["mass"][:, i], phsp_np["mass"][:, i])
        for i in range(n_mass, len(axes.flatten())):
            axes.flatten()[i].set_visible(False)
        plt.tight_layout()
        _save_figure(fig, "mass.png")

        # ---- Angles ----
        n_pos = data_np["angle"].shape[1]  # 24
        n_comp = 3
        angle_vars = []
        for pos in range(n_pos):
            for comp in range(n_comp):
                angle_vars.append((
                    f"angle[{pos},{comp}]",
                    data_np["angle"].reshape(ne_d, -1, 3)[:, pos, comp],
                    phsp_np["angle"].reshape(ne_p, -1, 3)[:, pos, comp],
                ))
        n_ang = len(angle_vars)  # 72
        n_rows = (n_ang + cols - 1) // cols
        fig, axes = plt.subplots(n_rows, cols,
            figsize=(figsize[0], 2.5 * n_rows), squeeze=False)
        for i, (label, d, p) in enumerate(angle_vars):
            _make_hist(axes.flatten()[i], label, d, p)
        for i in range(n_ang, len(axes.flatten())):
            axes.flatten()[i].set_visible(False)
        plt.tight_layout()
        _save_figure(fig, "angles.png")

        # ---- Time ----
        fig, ax = plt.subplots(1, 1, figsize=(figsize[0], 3))
        _make_hist(ax, "time", data_np["time"], phsp_np["time"])
        ax.legend(fontsize=8)
        plt.tight_layout()
        _save_figure(fig, "time.png")

        if show:
            plt.show()

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
