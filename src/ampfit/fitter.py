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


class Fitter:
    """Global fitter: config → objects → compute with norm constraint."""

    def __init__(self, config_file="config_angle.yml", backend=None):
        """Load config, create kernel and parameter constraint.

        Args:
            config_file: path to YAML config.
            backend: a :class:`ComputeBackend` instance, or a string shortcut.
                     Strings: ``"cuda"`` (default f64), ``"cuda32"`` (f32),
                     ``"cuda64"``, ``"numpy"``, ``"onnx"``/``"onnx_cpu"``,
                     ``"onnx_cuda"``.
        """
        from ampfit.config_loader import Config
        from ampfit.param_constraint import ConstraintManager
        from ampfit.backends import CUDABackend, NumpyBackend, ONNXBackend

        self.config = Config(config_file)
        self.kernel_config = self.config.build_all_index()

        # Resolve backend
        if backend is None or backend == "cuda":
            backend = CUDABackend(self.kernel_config, dtype="float64")
        elif isinstance(backend, str):
            if backend == "cuda64":
                backend = CUDABackend(self.kernel_config, dtype="float64")
            elif backend == "cuda32":
                backend = CUDABackend(self.kernel_config, dtype="float32")
            elif backend == "numpy":
                backend = NumpyBackend(self.kernel_config)
            elif backend in ("onnx", "onnx_cpu"):
                backend = ONNXBackend(
                    kernel_config=self.kernel_config,
                    providers=["CPUExecutionProvider"],
                )
            elif backend == "onnx_cuda":
                backend = ONNXBackend(
                    kernel_config=self.kernel_config,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
            else:
                raise ValueError(f"Unknown backend: {backend}")
        # 'backend' is now a ComputeBackend instance
        self.backend = backend

        # ck_map (list of param name tuples, one per partial wave)
        self.all_comb = self.config.get_ck_map()
        self.n_wave = len(self.all_comb)

        # Physical parameter dimensions
        self.n_m0 = len(self.config.m0_phys_name)
        self.n_g0 = len(self.config.g0_phys_name)

        # Standalone constraint manager (no rebuild on PC — fine-grained dirty flags)
        self.cm = ConstraintManager(
            self.all_comb,
            self.config.m0_phys_name,
            self.config.g0_phys_name,
        )

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
        self.data_holder = self.backend.load_data(data)

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

        # Check if backend supports batched phsp (CUDA with large datasets)
        if hasattr(self.backend, 'prepare_phsp_batched'):
            est_mb = n * 36 / 1024
            if est_mb >= 4000:
                self.backend.prepare_phsp_batched(phsp, n)
                self.phsp_holder = None
                self._phsp_buffer = True  # flag for _check_data_loaded
                return

        # Default: load directly
        self.phsp_holder = self.backend.load_data(phsp)

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
            norm, grads, _ = self.backend.compute(params, self.phsp_holder, norm=None)
            return float(norm), grads

        # Batched mode: delegate to backend
        if hasattr(self.backend, 'compute_norm_batched'):
            return self.backend.compute_norm_batched(params)

        # Fallback: iterate manually
        total_norm = 0.0
        total_grads = None
        bs = self._phsp_batch_size
        n_batches = (self._phsp_n + bs - 1) // bs
        gc = self.backend.kernel.gpu_config

        for b in range(n_batches):
            start = b * bs
            end = min(start + bs, self._phsp_n)
            self._phsp_scratch.attach_input_slice(
                self._phsp_buffer, start, end,
                self._phsp_np["mass"].shape[1] if self._phsp_np is not None else 0,
                self._phsp_np["q"].shape[1] if self._phsp_np is not None else 0,
            )
            n_b, g_b, _ = self.backend.compute(params, self._phsp_scratch, norm=None)
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
        nll, grads, P = self.backend.compute(
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
        m0_arr = self.default_m0.copy()
        g0_arr = self.default_g0.copy()
        scalar_names = self.cm.SCALAR_NAMES
        if self.default_scalar is not None:
            scalar_arr = list(self.default_scalar)
        else:
            scalar_arr = [0.6, 0.01, 0.506, 0.01, 0.9, 0.2]

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

        scalar_names = self.cm.SCALAR_NAMES

        # Per-name grads from ck combinatorics
        grad_dict = self.cm.pc.backprop_grad(resolved, total_grads["ck"])

        # Merge m0, g0, scalar gradients
        for target, names_list in [('m0', self.config.m0_phys_name),
                                    ('g0', self.config.g0_phys_name),
                                    ('scalar', scalar_names)]:
            arr = np.asarray(total_grads[target])
            for i, name in enumerate(names_list):
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
            params, _, _, _ = self._build_params(x)

        # Compute norm and probabilities (handles batched phsp)
        norm, _ = self._compute_norm_batched(params)
        norm = float(norm)
        _, _, P_data = self.backend.compute(params, self.data_holder, norm=norm)

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
                _, _, P_b = self.backend.compute(params, self._phsp_scratch, norm=None)
                P_phsp_list.append(P_b[:self._phsp_scratch.n_events])
            P_phsp = np.concatenate(P_phsp_list)
        else:
            _, _, P_phsp = self.backend.compute(params, self.phsp_holder, norm=None)

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
        """Free all memory held by the backend."""
        if self.data_holder is not None and hasattr(self.data_holder, 'free'):
            self.data_holder.free()
        if self.phsp_holder is not None and hasattr(self.phsp_holder, 'free'):
            self.phsp_holder.free()
        self.backend.free()
        self.data_holder = None
        self.phsp_holder = None


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
