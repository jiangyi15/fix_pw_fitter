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


class Fitter:
    """Global fitter: config → objects → compute with norm constraint."""

    def __init__(self, config_file="config_angle.yml"):
        """Load config, create kernel and parameter constraint."""
        # Lazy imports so this module can be imported without CUDA etc.
        from config_loader import Config
        from cuda_kernel_cffi import CUDAKernel
        from param_constraint import ParameterConstraint

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

    # ------------------------------------------------------------------
    # Constraint setup
    # ------------------------------------------------------------------
    def set_fixed(self, fixed_params):
        """Set fixed (constant) parameters: {name: complex_value}."""
        self._fixed_params = dict(fixed_params)
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
        from param_constraint import ParameterConstraint
        self._pc = ParameterConstraint(
            self.all_comb,
            fixed_params=self._fixed_params,
            same_params=self._same_params,
            scale_params=self._scale_params,
        )

    @property
    def pc(self):
        """Lazily built ParameterConstraint (empty if not explicitly configured)."""
        if self._pc is None:
            self._rebuild_pc()
        return self._pc

    def initial_values(self, seed=None):
        """Random initial guess for the free variable vector x.
        
        Returns:
            array of shape (2 * n_free_vars,): [r0, θ0, r1, θ1, ...]
        """
        return self.pc.initial_values(seed=seed)

    def free_param_names(self):
        """Names of free parameters (after constraint reduction)."""
        return self.pc.free_param_names()

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

    def get_nll(self, x, m0=None, g0=None, scalar=None):
        """Compute NLL and its gradient w.r.t. constrained variables x.
        
        Args:
            x: real variable vector (2 * n_free_vars,), 
               [r0, θ0, r1, θ1, ...].
            m0, g0, scalar: override default physical params (optional).
        
        Returns:
            (nll, grad_x) where grad_x has the same shape as x.
        """
        # Build ck from constraint variables
        ck = self.pc.build_ck(x)

        # Build full params dict
        params = self._build_base_params(ck, m0, g0, scalar)

        # Compute NLL with norm (handles gradient combination internally)
        nll, total_grads = self.get_nll_raw(params)

        # Backpropagate ck gradient through parameter constraints
        grad_x = self.pc.backprop_grad(x, total_grads["ck"])

        return nll, grad_x

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
