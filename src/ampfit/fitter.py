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
from ampfit.param_constraint import SCALAR_NAMES


class Fitter:
    """Global fitter: config → objects → compute with norm constraint."""

    def __init__(self, config_file="config_angle.yml", backend=None):
        """Load config, create kernel and parameter constraint.

        Args:
            config_file: path to YAML config.
            backend: a :class:`ComputeBackend` instance, a ``str``, or
                     a ``dict``.  Strings are resolved via
                     :func:`create_backend`.  Dicts use::

                         {"name": "integrated", "base": "cuda_v3"}

                     where ``"base"`` is itself a recursive backend spec.
        """
        from ampfit.config_loader import Config
        from ampfit.param_constraint import ConstraintManager
        from ampfit.backends import create_backend

        self.config = Config(config_file)
        self.kernel_config = self.config.build_all_index()

        # Resolve backend
        if backend is None or backend == "cuda":
            backend = create_backend("cuda64", self.kernel_config)
        elif isinstance(backend, (str, dict)):
            backend = create_backend(backend, self.kernel_config)
        # 'backend' is now a ComputeBackend instance
        self.backend = backend

        # ck_map (list of param name tuples, one per partial wave)
        self.all_comb = self.config.get_ck_map()
        self.n_wave = len(self.all_comb)

        # Physical parameter dimensions
        self.n_m0 = len(self.config.m0_phys_name)
        self.n_g0 = len(self.config.g0_phys_name)

        # Standalone constraint manager — flat name list, no type distinction.
        all_ck_bases = {p for comb in self.all_comb for p in comb if isinstance(p, str)}
        all_names = (sorted([n + 'r' for n in all_ck_bases] +
                            [n + 'i' for n in all_ck_bases]) +
                     list(self.config.m0_phys_name) +
                     list(self.config.g0_phys_name) +
                     list(SCALAR_NAMES))
        self.cm = ConstraintManager(self.all_comb, all_names)
        # Auto-register mass/width transforms from particle models
        self.setup_mass_width_transforms()

        # Data holders (created by set_data / set_phsp)
        self._data_holder = None
        self._phsp_holder = None

        # Raw numpy data (needed for norm gradient computation)
        self._data_np = None
        self._phsp_np = None

        # Phsp batching — always used (backends split into GPU-sized batches)
        self._phsp_scratch = None  # GPUDataHolder with batch-sized intermediates
        self._phsp_batch_size = 50000  # events per batch
        self._phsp_n = 0               # total phsp events

        # Default physical params (lazy-built unified dict)
        self._defaults = None         # {name: physical_default, ...}

        # Background / purity parameters (computed during set_phsp/set_data)
        self._purity = None           # purity fraction from config
        self._N_b = None              # sum(phsp_weight * bkg) / sum(phsp_weight)
        self._bkg_scale = None        # (1-purity)/purity / N_b for bkg scaling
        self._log_purity_const = 0.0  # -log(purity) * sum(data_weight)

        # Last evaluated x vector (stored by get_nll for interrupt checkpoint)
        self._last_xk = None

    # ------------------------------------------------------------------
    # Constraint setup — all delegated to ConstraintManager
    # ------------------------------------------------------------------
    def set_fixed(self, fixed_slots, reset=False):
        """Fix parameter slot(s) to a constant value.  *Additive* — call multiple times.

        Delegates to :attr:`cm` (marks only variable-registry dirty).
        """
        self.cm.set_fixed(fixed_slots, reset=reset)

    def set_same(self, same_params, reset=False):
        """Share a value across parameter names.  *Additive*.

        Delegates to :attr:`cm` (marks PC + variable-registry dirty).
        """
        self.cm.set_same(same_params, reset=reset)

    def set_scale(self, scale_params, reset=False):
        """Multiply a parameter by a real scale factor.  *Additive*.

        Delegates to :attr:`cm` (marks only PC dirty).
        """
        self.cm.set_scale(scale_params, reset=reset)

    def setup_mass_width_transforms(self):
        """Collect mass/width transforms from all particle models and register them.

        Iterates over all decay chains, calls ``make_mass_width_transform()``
        on each model, and adds non-``None`` results to the constraint pipeline.
        Duplicate model instances (same particle appearing in multiple chains)
        are registered only once.
        """
        transforms = []
        seen = set()
        for chain in self.config.full_decay.chains:
            for decay in chain.decays[1:]:
                model = decay.core._model
                mid = id(model)
                if mid in seen:
                    continue
                seen.add(mid)
                tfm = model.make_mass_width_transform()
                if tfm is not None:
                    transforms.append(tfm)
        self.cm.set_mass_width_transforms(transforms)

    def set_free(self, name):
        """Unfix a previously fixed parameter so it becomes free again.

        Delegates to :attr:`cm`.
        """
        self.cm.set_free(name)

    def set_range(self, name, lo, hi):
        """Bound a parameter via bijective arctan transform.  *Additive*.

        Delegates to :attr:`cm` (no dirty flags touched).
        """
        self.cm.set_range(name, lo, hi)

    def unset_range(self, name):
        """Remove the arctan bound on *name*.

        Delegates to :attr:`cm`.
        """
        self.cm.unset_range(name)

    # ------------------------------------------------------------------
    # Backward-compat property aliases → cm
    # ------------------------------------------------------------------
    @property
    def pc(self):
        """Lazily built :class:`ParameterConstraint` (via :attr:`cm`)."""
        return self.cm.pc

    @property
    def _var_registry(self):
        return self.cm.var_registry

    @property
    def _fixed_slots(self):
        return self.cm.fixed_slots

    @property
    def _same_params(self):
        return self.cm.same_params

    @property
    def _scale_params(self):
        return self.cm.scale_params

    @property
    def _bound_transforms(self):
        return self.cm.bound_transforms

    @property
    def _alias_to_canon(self):
        return self.cm.alias_to_canon

    @property
    def var_registry(self):
        return self.cm.var_registry

    # ------------------------------------------------------------------
    # Data loading and setup
    # ------------------------------------------------------------------

    @staticmethod
    def load_npz(npz_path, max_events=None):
        """Load ``.npz`` data and format for the kernel.

        Args:
            npz_path: path to ``.npz`` file with ``mass``, ``q``,
                      ``angle``/``angles``, ``time``, ``frac``,
                      ``bkg_raw``, ``weight`` arrays.
            max_events: if given, subsample to this many events
                        (deterministic seed 0).

        Returns:
            ``(data_dict, n_events)``.
        """
        import numpy as np
        data = np.load(npz_path)
        if "angles" in data and "angle" not in data:
            data = dict(data)
            data["angle"] = data.pop("angles")

        n_events = data["mass"].shape[0]
        if max_events is not None and max_events < n_events:
            n_events = max_events
            idx = np.random.RandomState(0).choice(
                data["mass"].shape[0], n_events, replace=False)
        else:
            idx = slice(None)

        out = {
            "mass": data["mass"][idx].reshape(n_events, -1),
            "q": data["q"][idx].reshape(n_events, -1),
            "angle": data["angle"][idx].reshape(n_events, -1, 3),
            "time": data["time"][idx].astype(np.float64),
            "frac": data["frac"][idx].astype(np.float64),
            "bkg": data["bkg_raw"][idx].astype(np.float64),
            "weight": data["weight"][idx].astype(np.float64),
        }
        assert not np.any(np.isnan(out["mass"])), "NaN in mass"
        return out, n_events

    def set_data(self, data):
        """Set data (real events) for negative log-likelihood.

        Reads purity from config and scales bkg:
          bkg_scaled = (1-purity)/purity * bkg_raw / N_b
        where N_b is the average phsp background (computed in set_phsp).

        Also stores -log(purity)*sum(weights) as a constant NLL offset.

        IMPORTANT: Call set_phsp() BEFORE set_data() for correct purity
        correction. If set_phsp() is not called first, purity correction
        is skipped (treated as purity=1.0).

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
            if self._purity is not None and self._N_b is None:
                import warnings
                warnings.warn(
                    "set_data() called before set_phsp(): purity correction skipped. "
                    "Call set_phsp() first for correct background scaling."
                )

        self._data_np = data
        self._data_holder = self.backend.load_data(data)

    def set_phsp(self, phsp):
        """Set phase-space data for normalization integral.

        Normalizes phsp weights to sum to 1 and computes N_b = mean bkg
        (weighted average of phsp bkg). These are used by set_data for
        the purity-based background scaling.

        Auto-batches if phsp is too large for GPU memory (CUDA backend).
        All input data stays on GPU permanently across get_nll calls.

        Args:
            phsp: dict with same structure as data.
        """
        # Normalize phsp weights to sum to 1.0 (matches reference TFPWA convention)
        phsp = dict(phsp)
        n = phsp["mass"].shape[0]
        w = phsp.get("weight", np.ones(n))
        w_sum = np.sum(w)
        if w_sum > 0:
            phsp["weight"] = w / w_sum

        # Compute N_b = weighted average of bkg over phsp
        # (weights sum to 1, so sum(weight*bkg) is the weighted mean)
        b = phsp.get("bkg", np.zeros(phsp["mass"].shape[0]))
        if np.isscalar(b):
            b = np.full(phsp["mass"].shape[0], b, dtype=np.float64)
        self._N_b = float(np.sum(phsp["weight"] * b)) if w_sum > 0 else 0.0

        self._phsp_np = phsp
        n = phsp["mass"].shape[0]
        self._phsp_n = n

        # Always use batched mode (split into GPU-sized chunks)
        if hasattr(self.backend, 'prepare_phsp_batched'):
            self.backend.prepare_phsp_batched(phsp, n)
            self._phsp_scratch = getattr(self.backend, '_phsp_scratch', None)
            self._phsp_holder = self._phsp_scratch
        else:
            # Backends without batched support fallback to manual batching
            self._phsp_scratch = self.backend.load_data(phsp)
            self._phsp_holder = self._phsp_scratch
            # phsp_buffer is None → _compute_norm_batched uses fallback loop

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

    def reinitial(self, seed=None):
        """Deterministic flat vector using stored physical defaults.

        Parameters with a config default (mass, width, scalar) use that
        value exactly.  Parameters without defaults (ck coupling slots)
        get a small random offset to break symmetry.
        """
        _ = self.pc
        rng = np.random.RandomState(seed)
        names = self._var_registry.flat_names
        defaults = self.defaults
        x = np.empty(len(names))
        for i, name in enumerate(names):
            if name in defaults:
                val = float(defaults[name])
                if i in self._bound_transforms:
                    bt = self._bound_transforms[i]
                    val = bt.inverse(val)
                x[i] = val
            else:
                x[i] = rng.uniform(-0.01, 0.01)
        return x

    def values_from_dict(self, data):
        """Build the flat x vector from a save_params JSON dict.
        
        Reads physical (bounded) values from the 'value' section,
        reverses constraints via ``cm.inverse()``, and inverts bound
        transforms to get optimizer-space x.
        
        Args:
            data: dict from save_params() JSON (keys 'value', 'error', ...).
        
        Returns:
            flat vector x suitable for get_nll() or fit().
        """
        _ = self.pc
        names = self._var_registry.flat_names
        values = data.get("value", data) if isinstance(data, dict) else data
        # Start from deterministic defaults, then override with JSON
        x = self.reinitial()

        # Build physical dict from JSON and invert constraints
        phys = {name: float(values[name]) for name in names if name in values}
        if phys:
            raw = self.cm.inverse(phys)
            for i, name in enumerate(names):
                if name in raw:
                    val = float(raw[name])
                    if i in self._bound_transforms:
                        bt = self._bound_transforms[i]
                        val = bt.inverse(val)
                    x[i] = val

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
        return self.cm.free_param_names()

    @property
    def defaults(self):
        """Unified physical default values for all params (lazy-built)."""
        if self._defaults is None:
            d = {}
            # Defaults from particle models (mass + gamma/width)
            # Skip the top decay (B) — it has no fit parameters
            seen = set()
            for chain in self.config.full_decay.chains:
                for decay in chain.decays[1:]:
                    model = decay.core._model
                    mid = id(model)
                    if mid in seen:
                        continue
                    seen.add(mid)
                    for k, v in model.get_defaults().items():
                        d[k] = float(v)
            # Scalar defaults
            scalar_base = {"gamma": 0.0, "delta_gamma": 0.0, "delta_m": 0.506,
                           "A_prod": 0.0, "poqr": 1.0, "poqi": 0.0}
            for name in SCALAR_NAMES:
                d.setdefault(name, scalar_base.get(name, 0.0))
            self._defaults = d
        return self._defaults

    @defaults.setter
    def defaults(self, val):
        self._defaults = dict(val) if val else {}

    def set_default_params(self, m0=None, g0=None, scalar=None):
        """Set default physical parameters (overrides config defaults)."""
        if self._defaults is None:
            _ = self.defaults  # trigger lazy build
        if m0 is not None:
            for name, val in zip(self.config.m0_phys_name, m0):
                self._defaults[name] = float(val)
        if g0 is not None:
            for name, val in zip(self.config.g0_phys_name, g0):
                self._defaults[name] = float(val)
        if scalar is not None:
            for name, val in zip(SCALAR_NAMES, scalar):
                self._defaults[name] = float(val)

    def _check_data_loaded(self):
        """Raise if data or phsp not set."""
        if self._data_holder is None:
            raise RuntimeError("Data not set. Call set_data() first.")
        phsp_ok = self._phsp_n > 0
        if not phsp_ok:
            raise RuntimeError("Phase space not set. Call set_phsp() first.")

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------
    def _build_base_params(self, ck, m0, g0, scalar):
        """Build the params dict from components, using defaults for None."""
        defaults = self.defaults
        if m0 is None:
            m0 = np.array([defaults.get(n, 0.0) for n in self.config.m0_phys_name])
        if g0 is None:
            g0 = np.array([defaults.get(n, 0.0) for n in self.config.g0_phys_name])
        if scalar is None:
            scalar = [defaults.get(n, 0.0) for n in SCALAR_NAMES]
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
        # Use float64 to avoid overflow with f32 backends
        P = np.asarray(P, dtype=np.float64)
        weight = np.asarray(weight, dtype=np.float64)
        bkg = np.asarray(bkg, dtype=np.float64)
        denom = norm * (P + bkg * norm)
        return np.sum(weight * P / denom)

    def _compute_norm_batched(self, params):
        """Compute norm over ALL phsp events."""
        norm, grads, _ = self.backend.compute(params, self._phsp_scratch,
                                              norm=None, return_p=False)
        return float(norm), grads

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
        # Weights sum to 1 from set_phsp(), so kernel Q = sum(P*w) = mean(P*w) = norm.
        # Kernel's norm_grad is already d(norm)/dparam — no scaling needed.
        norm = float(norm)
        for key in norm_grads:
            if norm_grads[key] is not None:
                norm_grads[key] = np.asarray(norm_grads[key])

        # 2. NLL from data (with norm)
        nll, grads, P = self.backend.compute(
            params, self._data_holder, norm=norm
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
            if key in ("ck", "scalar", "m0", "g0"):
                g = np.asarray(grads[key])
                ng = np.asarray(norm_grads[key])
                total_grads[key] = g + dNLL_dnorm * ng
            else:
                total_grads[key] = grads[key]

        return nll, total_grads

    # ── unified pipeline (forward + backward) ─────────────────────

    def _build_params(self, x):
        """Full forward pipeline: flat x → kernel params dict.

        Returns ``(params, resolved, raw, x_mapped)``.
        """
        from ampfit.boundary import apply_bounds

        x_mapped = apply_bounds(x, self._bound_transforms)
        raw = self._var_registry.to_dict(x_mapped)
        resolved = self.cm.resolve(raw)
        ck = self.cm.pc.build_ck(resolved)

        # Build m0, g0, scalar from defaults + resolved values
        defaults = self.defaults
        m0_arr = np.array([defaults.get(n, 0.0) for n in self.config.m0_phys_name])
        g0_arr = np.array([defaults.get(n, 0.0) for n in self.config.g0_phys_name])
        scalar_names = SCALAR_NAMES
        scalar_arr = [defaults.get(n, 0.0) for n in scalar_names]

        for name, val in resolved.items():
            if name in self.config.m0_phys_name:
                m0_arr[self.config.m0_phys_name.index(name)] = val
            elif name in self.config.g0_phys_name:
                g0_arr[self.config.g0_phys_name.index(name)] = val
            elif name in scalar_names:
                scalar_arr[scalar_names.index(name)] = val

        params = {"ck": ck, "m0": m0_arr, "g0": g0_arr, "scalar": scalar_arr}
        return params, resolved, raw, x_mapped

    def _flat_gradient(self, total_grads, resolved, raw, x_mapped, x):
        """Full backward pipeline: kernel grads → flat gradient."""
        from ampfit.boundary import apply_bound_grads

        scalar_names = SCALAR_NAMES

        # Per-name grads from ck combinatorics
        grad_dict = self.cm.pc.backprop_grad(resolved, total_grads["ck"])

        # Merge m0, g0, scalar gradients
        # (extra names like standalone width params beyond the backend's
        # array size are handled via the transform's backward gradient)
        for target, names_list in [('m0', self.config.m0_phys_name),
                                    ('g0', self.config.g0_phys_name),
                                    ('scalar', scalar_names)]:
            arr = np.asarray(total_grads[target])
            for i, name in enumerate(names_list):
                if i < len(arr):
                    grad_dict[name] = grad_dict.get(name, 0.0) + arr[i]

        # Chain back through constraints
        grad_raw = self.cm.chain_gradient(grad_dict, resolved, raw)
        grad_flat = self._var_registry.flat_gradient(x_mapped, grad_raw)
        grad_flat = apply_bound_grads(grad_flat, x, self._bound_transforms)

        # Zero fixed slots
        for slot_name in self._fixed_slots:
            if slot_name in self._var_registry.flat_names:
                idx = self._var_registry.flat_names.index(slot_name)
                grad_flat[idx] = 0.0
        return grad_flat

    def get_nll(self, x, m0=None, g0=None):
        """Compute NLL and gradient w.r.t. the flat variable vector.

        Pipeline::

            x → _build_params → kernel → _flat_gradient → (nll, grad)

        Args:
            x: flat variable vector (length = ``free_param_names()``).

        Returns:
            ``(nll, grad_x)`` where ``grad_x`` has the same shape as ``x``.
        """
        self._last_xk = x.copy()
        params, resolved, raw, x_mapped = self._build_params(x)
        nll, total_grads = self.get_nll_raw(params)
        grad_flat = self._flat_gradient(total_grads, resolved, raw, x_mapped, x)
        return nll, grad_flat

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    def fit(self, x0=None, maxiter=1000, ftol=1e-5, gtol=1e-3, callback=None,
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

    def _params_from_fit(self, fit_result, return_bounded=True, hess_inv=None):
        """Build dicts of parameter values and errors from a BFGS fit result.
        
        Args:
            fit_result: OptimizeResult from fit() method.
            return_bounded: if True, transform values back through
                            BoundTransform (physical space).
                            if False, return raw unbounded optimizer values.
            hess_inv: optional inverse Hessian. If None, uses
                      ``fit_result.hess_inv``.
        
        Returns:
            (values_dict, errors_dict) where each maps slot_name -> float.
            values_dict: best-fit parameter values.
            errors_dict: 1-sigma uncertainties from Hessian diagonal.
        """
        from ampfit.boundary import BoundTransform

        x_best = fit_result.x
        if hess_inv is None:
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

    def get_uncertainties(self, fit_result, return_bounded=True, use_cached=True):
        """Compute parameter uncertainties from a fit result.

        Uses the inverse Hessian from the fit, or computes a numerical
        Hessian via finite-difference gradients when *use_cached* is
        ``False``.

        Bound transforms are automatically propagated for parameters
        set via ``set_range()``.

        Args:
            fit_result: OptimizeResult from ``fit()`` method.
            return_bounded: if True (default), returns values and errors
                            in the physical (bounded) space.
                            if False, returns raw optimizer space values.
            use_cached: if True (default), use ``fit_result.hess_inv``.
                        if False, compute numerical Hessian from
                        ``fit_result.x`` via :meth:`compute_numerical_hessian`.

        Returns:
            dict of {slot_name: (value, error)} for every free parameter.
        """
        if not use_cached:
            import numpy as np
            H = self.compute_numerical_hessian(fit_result.x)
            hess_inv = np.linalg.inv(H)
            values, errors = self._params_from_fit(fit_result, return_bounded,
                                                    hess_inv=hess_inv)
        else:
            values, errors = self._params_from_fit(fit_result, return_bounded)
        return {name: (values[name], errors[name]) for name in values}

    def compute_numerical_hessian(self, x, eps=1e-5):
        """Numerical Hessian via 2-point gradient difference.

        For each parameter *j*, perturbs ``x[j] ± eps``, evaluates the
        gradient, and approximates::

            H[i, j] = (grad_i(x+eps·eⱼ) - grad_i(x-eps·eⱼ)) / (2·eps)

        This requires ``2·N`` gradient evaluations (N = free params).

        The result can replace ``fit_result.hess_inv``::

            import numpy as np
            H = fitter.compute_numerical_hessian(result.x)
            result.hess_inv = np.linalg.inv(H)

        Args:
            x: flat parameter vector at the minimum (``result.x``).
            eps: finite-difference step (default 1e-5).

        Returns:
            ``(N, N)`` ndarray — the symmetric Hessian matrix d²(NLL)/dx².
        """
        import numpy as np
        n = len(x)
        H = np.zeros((n, n))

        for j in range(n):
            xp = x.copy(); xp[j] += eps
            _, gp = self.get_nll(xp)
            xm = x.copy(); xm[j] -= eps
            _, gm = self.get_nll(xm)
            H[:, j] = (gp - gm) / (2 * eps)

        # Symmetrise
        return (H + H.T) / 2

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot(self, result=None, x=None, params=None, prefix="plots/",
             n_bins=50, cols=4, figsize=(15, 10), show=False):
        """Plot data vs phsp distributions, saving figures to a directory.
        
        For each variable, shows two histograms:
          - Data (weighted by data weight)
          - Phsp weighted by P × phsp_weight (the model prediction)
        
        Only 1/8 of total variables are plotted (first 6 mass + 9 angle).
        Angle[..., 0] mapped to [-π, π]; angle[..., 1,2] use cos transform.
        
        Figures are saved as:
          {prefix}mass.png      — first 6 mass columns
          {prefix}angles.png    — first 9 angle components (3 pos × 3 comp)
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
            params, _, _, _ = self._build_params(x)

        # Compute norm and probabilities (handles batched phsp)
        norm, _ = self._compute_norm_batched(params)
        norm = float(norm)
        _, _, P_data = self.backend.compute(params, self._data_holder, norm=norm)

        # Compute P_phsp (handle batched mode)
        if hasattr(self.backend, '_phsp_buffer') and self.backend._phsp_buffer is not None:
            # Original CUDA batched mode: GPUDataBuffer with slicing
            P_phsp_list = []
            bs = self._phsp_batch_size
            n_batches = (self._phsp_n + bs - 1) // bs
            for b in range(n_batches):
                start = b * bs
                end = min(start + bs, self._phsp_n)
                self._phsp_scratch.attach_input_slice(
                    self.backend._phsp_buffer, start, end,
                    self._phsp_np["mass"].shape[1], self._phsp_np["q"].shape[1])
                _, _, P_b = self.backend.compute(params, self._phsp_scratch, norm=None)
                P_phsp_list.append(P_b[:self._phsp_scratch.n_events])
            P_phsp = np.concatenate(P_phsp_list)
        else:
            phsp_h = self._phsp_holder
            if phsp_h is None:
                raise RuntimeError("No phsp data loaded; call set_phsp() first")
            _, _, P_phsp = self.backend.compute(params, phsp_h, norm=None)

        data_np = self._data_np
        phsp_np = self._phsp_np
        dw = data_np["weight"]
        ne_d, ne_p = len(P_data), len(P_phsp)

        # Signal and background weights for phsp model histograms
        purity = self._purity if self._purity is not None else 1.0
        data_total = float(np.sum(dw))

        pw_sig = phsp_np["weight"] * P_phsp                  # unnormalized signal
        pw_bkg = phsp_np["weight"] * phsp_np.get("bkg", np.zeros(ne_p))

        sig_norm = float(np.sum(pw_sig))                     # = ∫P·w  (norm)
        bkg_norm = float(np.sum(pw_bkg))                     # = ∫bkg·w (N_b)

        sig_scale = data_total * purity / sig_norm if sig_norm > 0 else 0.0
        bkg_scale = data_total * (1.0 - purity) / bkg_norm if bkg_norm > 0 else 0.0

        # Ensure output directory
        os.makedirs(prefix, exist_ok=True)

        def _make_hist(ax, label, d, p, bins_range=None, n_bins_override=None):
            if bins_range is not None:
                lo, hi = bins_range
            else:
                lo = min(d.min(), p.min())
                hi = max(d.max(), p.max())
            if hi - lo < 1e-12:
                hi = lo + 1.0
            nb = n_bins_override if n_bins_override is not None else n_bins
            bins = np.linspace(lo, hi, nb + 1)
            bin_w = bins[1] - bins[0]
            bin_c = (bins[:-1] + bins[1:]) / 2

            # Data: weighted counts with sqrt(sum(w²)) error bars
            data_y, _ = np.histogram(d, bins=bins, weights=dw)
            data_w2, _ = np.histogram(d, bins=bins, weights=dw ** 2)
            data_err = np.sqrt(data_w2)

            # Model components (histogram over phsp)
            sig_y, _ = np.histogram(p, bins=bins, weights=pw_sig * sig_scale)
            bkg_y, _ = np.histogram(p, bins=bins, weights=pw_bkg * bkg_scale)

            # Stacked histogram: background at bottom, signal on top
            ax.bar(bin_c, bkg_y, width=bin_w, alpha=0.5,
                   color='C3', label='bkg', align='center')
            ax.bar(bin_c, sig_y, width=bin_w, alpha=0.5,
                   color='C1', label='signal', align='center', bottom=bkg_y)

            # Data points on top
            ax.errorbar(bin_c, data_y, yerr=data_err, fmt='o',
                        color='C0', label='data', markersize=3, capsize=2)

            ax.set_xlabel(label, fontsize=7)
            ax.tick_params(labelsize=6)

        def _save_figure(fig, name):
            path = os.path.join(prefix, name)
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  saved {path}")

        # ---- Mass (first 6 columns = 48/8, 3×2 grid, range (0.2,5.2), 100 bins) ----
        n_mass_total = data_np["mass"].shape[1]  # 48
        n_mass_plot = n_mass_total // 8            # 6
        fig, axes = plt.subplots(3, 2,
            figsize=(figsize[0] * 0.5, 6.5), squeeze=False)
        for i in range(n_mass_plot):
            _make_hist(axes.flatten()[i], f"mass[{i}]",
                       data_np["mass"][:, i], phsp_np["mass"][:, i],
                       bins_range=(0.2, 5.2), n_bins_override=100)
        plt.tight_layout()
        _save_figure(fig, "mass.png")

        # ---- Angles (first 9 = 72/8, 3 positions × 3 components, 3×3 grid) ----
        d_angle = data_np["angle"].reshape(ne_d, -1, 3)
        p_angle = phsp_np["angle"].reshape(ne_p, -1, 3)
        n_pos_plot = d_angle.shape[1] // 8  # 3 positions

        fig, axes = plt.subplots(3, 3,
            figsize=(figsize[0] * 0.75, 7), squeeze=False)
        for pos in range(n_pos_plot):
            d0 = (d_angle[:, pos, 0] + np.pi) % (2 * np.pi) - np.pi
            d1 = np.cos(d_angle[:, pos, 1])
            d2 = np.cos(d_angle[:, pos, 2])
            p0 = (p_angle[:, pos, 0] + np.pi) % (2 * np.pi) - np.pi
            p1 = np.cos(p_angle[:, pos, 1])
            p2 = np.cos(p_angle[:, pos, 2])
            for cmp, (d, p, lbl) in enumerate([
                    (d0, p0, f"angle_phi [{pos},0]"),
                    (d1, p1, f"cos_theta1[{pos},1]"),
                    (d2, p2, f"cos_theta2[{pos},2]"),
            ]):
                ax = axes[pos, cmp]
                _make_hist(ax, lbl, d, p)
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

        *fit_result* can be:
          - an ``OptimizeResult`` from :meth:`fit()` — saves values + errors + status
          - a flat numpy array ``x`` — saves values only (no errors/status)

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
        import json, cmath, os

        # Accept flat x vector directly (for checkpoint saves)
        if isinstance(fit_result, np.ndarray):
            x = fit_result
            hess_inv = None
            fun = float('nan')
            jac = None
            success = False
            message = "No fit result"
        else:
            x = fit_result.x
            hess_inv = getattr(fit_result, 'hess_inv', None)
            fun = float(fit_result.fun)
            jac = fit_result.jac
            success = bool(fit_result.success)
            message = str(fit_result.message)

        flat_names = self._var_registry.flat_names
        # Resolved physical values (post-constraints)
        _, resolved, _, _ = self._build_params(x)

        # Errors from Hessian (transformed to physical space)
        if hess_inv is not None:
            _, errors = self._params_from_fit(fit_result, return_bounded=True)
        else:
            errors = {}

        out = {"value": {}, "error": {}}
        # Store physical values for all resolved keys (canon + alias)
        for name, val in resolved.items():
            out["value"][name] = float(val)
        # Errors only for free params (matches reference JSON convention)
        free_names = set(self.free_param_names())
        for name in free_names:
            if name in errors:
                out["error"][name] = float(errors[name])

        # Add all defaults to value (including fixed params not in flat_names)
        for name, val in self.defaults.items():
            if name not in out["value"]:
                out["value"][name] = float(val)

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
                if r_name not in out["value"]:
                    out["value"][r_name] = float(self._fixed_slots[r_name])
                if i_name not in out["value"]:
                    out["value"][i_name] = float(self._fixed_slots[i_name])

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
        n_params = len(x)
        out["status"] = {
            "NLL": fun,
            "Ndf": n_params,
            "jac": jac.tolist() if hasattr(jac, 'tolist') else (list(jac) if jac is not None else []),
            "success": success,
            "message": message,
        }

        # Correlation matrix for B->rhoA.rhoB params (like archive)
        if hess_inv is not None:
            corr_items = [[i, n] for i, n in enumerate(flat_names) if "B->rhoA.rhoB" in n]
            if corr_items:
                corr_idx = np.array([i for i, n in corr_items])
                corr_order = [n for i, n in corr_items]
                corr_mat = hess_inv[np.ix_(corr_idx, corr_idx)] * grad_scale
                out["status"]["rhorho"] = [corr_order, corr_mat.tolist()]

        with open(filepath, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"✓ Saved fit results to {filepath}")

        # Save error matrix (inverse Hessian) alongside the JSON
        if hasattr(fit_result, 'hess_inv') and fit_result.hess_inv is not None:
            err_path = os.path.splitext(filepath)[0] + "_error_matrix.npy"
            np.save(err_path, fit_result.hess_inv)
            print(f"✓ Saved error matrix to {err_path}")

        return out

    # ------------------------------------------------------------------
    # Convenience / utility
    def save_hessian(self, fit_result, filepath):
        """Save the inverse Hessian to a ``.npy`` file.

        Args:
            fit_result: OptimizeResult from ``fit()`` (must have
                        ``.hess_inv``).
            filepath: output path (convention: ``*.npy``).
        """
        import numpy as np
        hess_inv = getattr(fit_result, 'hess_inv', None)
        if hess_inv is None:
            raise ValueError("fit_result has no hess_inv")
        np.save(filepath, hess_inv)

    def load_results(self, json_path, hessian_path=None):
        """Load fit results from a JSON file and optionally the error matrix.

        The JSON file should have been saved by :meth:`save_params`.
        ``x`` is reconstructed from the ``"value"`` dict via
        :meth:`values_from_dict`.

        If *hessian_path* is not given, the method looks for
        ``{json_stem}_error_matrix.npy`` alongside the JSON and loads
        it automatically if present.

        Args:
            json_path: path to the JSON file from :meth:`save_params`.
            hessian_path: optional path to a ``.npy`` file.  If ``None``
                          (default), auto-detects ``_error_matrix.npy``.

        Returns:
            SimpleNamespace with ``.x`` (and ``.hess_inv`` if the error
            matrix was found).
        """
        import json, numpy as np, os
        from types import SimpleNamespace
        with open(json_path) as f:
            data = json.load(f)
        res = SimpleNamespace()
        res.x = self.values_from_dict(data.get("value", data))
        if hessian_path is None:
            auto_path = os.path.splitext(json_path)[0] + "_error_matrix.npy"
            if os.path.exists(auto_path):
                hessian_path = auto_path
        if hessian_path:
            res.hess_inv = np.load(hessian_path)
        return res

    def save_constraints(self, filepath):
        """Save all constraints (fixed, same, scale, bounds) to JSON.

        Args:
            filepath: output JSON path.
        """
        import json
        from collections import defaultdict
        flat_names = self._var_registry.flat_names

        # Fixed slots
        fixed = dict(self._fixed_slots)

        # Same-param groups: reconstruct from alias→canon dict
        alias_map = self._same_params  # {alias: canon}
        canon_groups = defaultdict(list)
        for alias, canon in alias_map.items():
            canon_groups[canon].append(alias)
        same = [[canon] + sorted(aliases) for canon, aliases in canon_groups.items()]

        # Scale params
        scale = dict(self._scale_params)

        # Bounds: name → {"low": a, "high": b}  (de-duplicated by value)
        bt = self._bound_transforms
        bounds = {}
        seen = {}
        for i, name in enumerate(flat_names):
            if i in bt:
                t = bt[i]
                key = (t.a, t.b)
                if key not in seen:
                    seen[key] = name
                    bounds[name] = {"low": t.a, "high": t.b}

        out = {"fixed": fixed, "same": same, "scale": scale, "bounds": bounds}
        with open(filepath, "w") as f:
            json.dump(out, f, indent=2)
        print(f"✓ Saved constraints to {filepath}")

    def load_constraints(self, filepath):
        """Load constraints from a JSON file saved by :meth:`save_constraints`.

        Resets and re-applies all fixed/same/scale/bound settings.
        Also normalises old-format bare keys (without ``r``/``i`` suffix)
        to full slot names for backward compatibility.
        """
        import json
        with open(filepath) as f:
            data = json.load(f)

        self.set_fixed(data.get("fixed", {}), reset=True)
        self.set_same(data.get("same", []), reset=True)

        # Backward compat: normalise old bare scale keys (without ``r`` suffix)
        # before calling set_scale (scale_params is now a snapshot dict)
        scale_data = dict(data.get("scale", {}))
        names = set(self.var_registry.flat_names)
        for key in list(scale_data):
            if key not in names and key + 'r' in names:
                scale_data[key + 'r'] = scale_data.pop(key)
        self.set_scale(scale_data, reset=True)

        # Re-apply bounds
        for name, spec in data.get("bounds", {}).items():
            self.set_range(name, spec["low"], spec["high"])

        print(f"✓ Loaded constraints from {filepath}")

    # ------------------------------------------------------------------
    def _grad_flat(self, fun, param_names, resolved, raw, x_mapped, x0, jac=False):
        """Compute optimizer-space gradient of *fun* w.r.t. *param_names*.

        When *jac* is ``True``, ``fun(phys_dict)`` must return
        ``(value, {name: gradient, ...})``.
        Otherwise a 3-point finite difference is computed.
        """
        import numpy as np
        from ampfit.boundary import apply_bound_grads

        names = [n for n in param_names if n in resolved]
        if not names:
            return None

        if jac:
            _, grad_dict = fun({n: float(resolved[n]) for n in names})
        else:
            grad_dict = {}
            for name in names:
                base = float(resolved[name])
                eps = 1e-5 * max(1.0, abs(base))
                rp = dict(resolved); rp[name] = base + eps
                vp = fun({n: float(rp[n]) for n in names})
                rm = dict(resolved); rm[name] = base - eps
                vm = fun({n: float(rm[n]) for n in names})
                grad_dict[name] = (vp - vm) / (2 * eps)

        if not grad_dict:
            return None
        grad_raw = self.cm.chain_gradient(grad_dict, resolved, raw)
        grad_flat = self._var_registry.flat_gradient(x_mapped, grad_raw)
        return apply_bound_grads(grad_flat, x0, self._bound_transforms)

    def cal_uncertainties(self, fun, param_names, fit_result, jac=False):
        """Propagate fit uncertainties to a function of physical parameters.

        When ``jac=False`` (default), a 3‑point finite difference is
        used for the gradient.  When ``jac=True``, ``fun`` must return
        ``(value, {name: gradient, ...})`` — an analytical gradient
        dict mapping each parameter name to its partial derivative.

        Args:
            fun: callable ``fun(phys_dict) → float``, where *phys_dict*
                 maps parameter names to their best-fit values.  With
                 ``jac=True`` returns ``(float, dict)`` instead.
            param_names: list of physical parameter names that *fun*
                         depends on (e.g. ``["a1(1260)p_mass"]``).
            fit_result: OptimizeResult from ``fit()`` (has ``.x`` and
                        ``.hess_inv``).
            jac: if ``True``, *fun* returns an analytical gradient dict.

        Returns:
            (value, uncertainty) where *value* is ``fun(best_fit_phys)``
            and *uncertainty* is the 1σ error propagated from the fit.
        """
        x0 = fit_result.x
        hess_inv = getattr(fit_result, 'hess_inv', None)
        _, resolved, raw, x_mapped = self._build_params(x0)

        names = [n for n in param_names if n in resolved]
        if not names:
            raise ValueError(f"None of {param_names} found in resolved parameters")

        fit_dict = {n: float(resolved[n]) for n in names}
        if jac:
            value, _ = fun(fit_dict)
        else:
            value = fun(fit_dict)

        if hess_inv is None:
            return value, 0.0

        gf = self._grad_flat(fun, param_names, resolved, raw, x_mapped, x0, jac=jac)
        if gf is None:
            return value, 0.0
        std = float(np.sqrt(max(gf @ hess_inv @ gf, 0.0)))
        return value, std

    def cal_uncertainties_multi(self, fun, param_names, fit_result, jac=False):
        """Covariance and correlation between multiple observables.

        When ``jac=False`` (default), a 3‑point finite difference is
        used.  When ``jac=True``, ``fun`` must return
        ``(list[values], list[dict])`` — a list of observable values
        and a list of gradient dicts (one per observable).

        Args:
            fun: callable ``fun(phys_dict) → list[float]`` (or
                 ``(list[float], list[dict])`` when ``jac=True``).
            param_names: list of physical parameter names.
            fit_result: OptimizeResult from ``fit()``.
            jac: if ``True``, *fun* returns analytical gradients.

        Returns:
            (values, cov, corr) where *values* is the list of function
            values at best fit, *cov* is the covariance matrix
            ``(n_obs × n_obs)``, and *corr* is the correlation matrix.
        """
        import numpy as np
        from ampfit.boundary import apply_bound_grads

        x0 = fit_result.x
        hess_inv = getattr(fit_result, 'hess_inv', None)
        _, resolved, raw, x_mapped = self._build_params(x0)
        names = [n for n in param_names if n in resolved]

        fit_dict = {n: float(resolved[n]) for n in names}

        if jac:
            values, grad_dicts = fun(fit_dict)
            values = list(values)
        else:
            values = list(fun(fit_dict))
            grad_dicts = [{} for _ in range(len(values))]
            for name in names:
                base = float(resolved[name])
                eps = 1e-5 * max(1.0, abs(base))
                rp = dict(resolved); rp[name] = base + eps
                vp = fun({n: float(rp[n]) for n in names})
                rm = dict(resolved); rm[name] = base - eps
                vm = fun({n: float(rm[n]) for n in names})
                for i in range(len(values)):
                    grad = (vp[i] - vm[i]) / (2 * eps)
                    if grad != 0.0:
                        grad_dicts[i][name] = grad

        n_obs = len(values)
        if hess_inv is None or not names:
            return values, np.zeros((n_obs, n_obs)), np.eye(n_obs)

        # Chain each observable's gradient to optimizer space
        G = np.zeros((n_obs, len(x0)))
        for i in range(n_obs):
            if not grad_dicts[i]:
                continue
            grad_raw = self.cm.chain_gradient(grad_dicts[i], resolved, raw)
            gf = self._var_registry.flat_gradient(x_mapped, grad_raw)
            G[i] = apply_bound_grads(gf, x0, self._bound_transforms)

        cov = G @ hess_inv @ G.T
        d = np.sqrt(np.maximum(np.diag(cov), 0.0))
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = cov / np.outer(d, d)
            corr[np.isnan(corr)] = 0.0
            corr = np.clip(corr, -1, 1)
        return values, cov, corr

    def get_bw_params(self, particle_name, fit_result):
        """BW peak mass and width for a single resonance from fit result.

        Calls ``model.get_bw_params()`` with the best-fit mass and width
        values for the given particle.  Computes 1σ uncertainties via
        :meth:`cal_uncertainties`.

        Args:
            particle_name: resonance name from config, e.g. ``"a1(1260)p"``.
            fit_result: OptimizeResult from ``fit()`` method (has ``.x``
                        and ``.hess_inv``).

        Returns:
            dict with keys ``"mass_bw"``, ``"width_bw"``,
            ``"mass_bw_err"``, ``"width_bw_err"``.
        """
        import numpy as np

        # --- 1. Find the particle model ---
        model = None
        for chain in self.config.full_decay.chains:
            for decay in chain.decays[1:]:
                if decay.core.name == particle_name:
                    model = decay.core._model
                    break
            if model is not None:
                break
        if model is None:
            raise ValueError(f"Particle '{particle_name}' not found in config")

        # --- 2. Collect parameter names and build fit dict ---
        _, resolved, _, _ = self._build_params(fit_result.x)
        param_names = []
        mass_name = f"{particle_name}_mass"
        if mass_name in resolved:
            param_names.append(mass_name)
        for gname in model.get_gamma_name():
            if gname in resolved:
                param_names.append(gname)
        # Also collect standalone width param (ck_matrix_v2 — not in gamma names)
        width_name = f"{particle_name}_width"
        if width_name in resolved and width_name not in param_names:
            param_names.append(width_name)
        if not param_names:
            raise ValueError(f"No fitted parameters found for '{particle_name}'")

        # --- 3. BW params and covariance via cal_uncertainties_multi ---
        def bw_fun(d):
            bw = model.get_bw_params(d)
            return [bw["mass_bw"], bw["width_bw"]]
        vals, cov, _ = self.cal_uncertainties_multi(bw_fun, param_names, fit_result)

        return {"mass_bw": vals[0], "width_bw": vals[1],
                "mass_bw_err": float(np.sqrt(max(cov[0, 0], 0.0))),
                "width_bw_err": float(np.sqrt(max(cov[1, 1], 0.0))),
                "mass_width_cov": float(cov[0, 1])}

    def free(self):
        """Free all memory held by the backend."""
        if self._data_holder is not None and hasattr(self._data_holder, 'free'):
            self._data_holder.free()
        if self._phsp_holder is not None and hasattr(self._phsp_holder, 'free'):
            self._phsp_holder.free()
        self.backend.free()
        self._data_holder = None
        self._phsp_holder = None


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
    print(f"Free param names ({len(fitter.free_param_names())}):")
    for n in fitter.free_param_names():
        print(f"  {n}")

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
