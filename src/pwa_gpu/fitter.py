"""
Unified PWA fitter: wraps config loading, kernel setup, and parameter management.

Internally uses ConstraintMapper (which wraps ParamMapper) for parameter transforms.
Users can set constraints directly on the fitter.

Usage:
    from pwa_gpu.fitter import PWAFitter
    
    fitter = PWAFitter("config.yml").auto_tables()
    fitter.load_params("a.json")
    fitter.set_fixed('delta_m', 0.0)
    fitter.set_equal('m0/1', 'm0/0')
    
    data = fitter.load_data(mass, q, angles, time, frac, weights, bkg)
    q_val, grads = fitter.compute(data, N=norm)
"""

import os
import numpy as np
from pwa_gpu.parse_config import parse_config
from pwa_gpu.build_tables import build_tables as _build_tables, load_tables as _load_tables, save_tables as _save_tables
from pwa_gpu.param_mapper import ParamMapper
from pwa_gpu.constraint_mapper import ConstraintMapper
from pwa_gpu import PWAGPU, PWAData


class PWAFitter:
    """
    Full PWA pipeline using ConstraintMapper for parameter transforms.
    
    Physical params are loaded via load_params().
    Constraints (fix, equal, linear) are applied on top via ConstraintMapper.
    compute() goes through: model → constraints → physical → kernel → GPU → back.
    """

    def __init__(self, config_path):
        self.config_path = config_path
        self._cfg = None
        self._pw_list = None
        self._kw_list = None
        self._mapper = None
        self._cst = None  # ConstraintMapper
        self._fitter = None
        self._params = None  # last loaded physical params
        self.parse()

    def _ensure_cst(self):
        """Create ConstraintMapper if needed."""
        if self._cst is None:
            self._cst = ConstraintMapper(self.mapper)
        return self._cst

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def parse(self):
        self._cfg, self._pw_list, self._kw_list = parse_config(self.config_path)
        return self

    def build_tables(self, save_path=None):
        if save_path:
            self.save_tables(save_path)
        self._cfg = _build_tables(self._cfg, self._pw_list, self._kw_list,
                                   self.config_path, save_path)
        self._mapper = ParamMapper(self._cfg)
        self._cst = None  # reset
        return self

    def save_tables(self, path):
        if self._cfg is not None and 'gamma_table' in self._cfg:
            _save_tables(self._cfg, path)
        return self

    def load_tables(self, path):
        _load_tables(self._cfg, path)
        self._mapper = ParamMapper(self._cfg)
        self._cst = None
        return self

    def auto_tables(self, npz_path=None):
        if npz_path is None:
            npz_path = self.config_path + '.tables.npz'
        if os.path.exists(npz_path):
            self.load_tables(npz_path)
        else:
            self.build_tables(npz_path)
        return self

    @property
    def cfg(self):
        return self._cfg

    @property
    def mapper(self):
        if self._mapper is None:
            self._mapper = ParamMapper(self._cfg)
        return self._mapper

    # ------------------------------------------------------------------
    # Constraints (forwarded to ConstraintMapper)
    # ------------------------------------------------------------------

    def set_fixed(self, key, value=None):
        """Fix a parameter. Removed from fit variables."""
        self._ensure_cst().set_fixed(key, value)
        return self

    def set_equal(self, target, source):
        """Make target equal to source (one fit variable)."""
        self._ensure_cst().set_equal(target, source)
        return self

    def set_linear(self, target, sources):
        """Linear combination: target = Σ coeff_i * source_i."""
        self._ensure_cst().set_linear(target, sources)
        return self

    def clear_constraints(self):
        """Remove all constraints."""
        if self._cst:
            self._cst.clear()
        return self

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def load_params(self, json_path):
        """Load fitted parameters from a.json into the mapper."""
        self._params = self.mapper.load_params(json_path, self.config_path)
        return self._params

    @property
    def params(self):
        """Last loaded physical params dict."""
        if self._params is None:
            mapper = self.mapper
            total = np.random.randn(len(mapper.totals)) + 1j * np.random.randn(len(mapper.totals))
            gls = np.random.randn(len(mapper.gls)) + 1j * np.random.randn(len(mapper.gls))
            glsbar = np.random.randn(len(mapper.glsbar)) + 1j * np.random.randn(len(mapper.glsbar))
            m0 = np.random.rand(mapper.n_m0_phys) + 1.5
            g0 = np.random.rand(mapper.n_g0_phys) * 0.1 + 0.05
            self._params = {
                'total': total, 'g_ls': gls, 'g_lsbar': glsbar,
                'm0': m0, 'g0': g0,
                'delta_m': 0.5, 'delta_g': 0.02, 'g': 0.65,
                'ap': 0.01, 'lam': 0.7, 'phi': 0.1,
            }
        return self._params

    # ------------------------------------------------------------------
    # GPU setup
    # ------------------------------------------------------------------

    def _ensure_fitter(self):
        if self._fitter is None:
            if self._cfg is None:
                self.parse()
            if 'gamma_table' not in self._cfg:
                self.build_tables()
            self._fitter = PWAGPU(self._cfg)
        return self._fitter

    def load_data(self, mass, q, angles, time_arr, frac, weights, bkg):
        """Upload signal data to GPU. Returns PWAData object."""
        fitter = self._ensure_fitter()
        return PWAData(fitter, mass, q, angles, time_arr, frac, weights, bkg)

    def load_phsp(self, mass, q, angles, time_arr, frac, phsp_weights=None):
        """Upload phase-space data to GPU. Returns PWAData object (unit weights)."""
        fitter = self._ensure_fitter()
        n = len(mass)
        if phsp_weights is not None:
            w = phsp_weights
        else:
            w = np.ones(n)
        t = time_arr if time_arr is not None else np.zeros(n)
        f = frac if frac is not None else np.zeros(n)
        return PWAData(fitter, mass, q, angles, t, f, w, np.zeros(n))

    # ------------------------------------------------------------------
    # Full fit (optimization) via scipy
    # ------------------------------------------------------------------

    def fit(self, data, phsp, method='L-BFGS-B', options=None):
        """
        Run full fit: minimize negative log-likelihood with scipy.
        
        Uses phase-space for normalization, ConstraintMapper for parameters.
        
        Args:
            data: PWAData for signal events
            phsp: PWAData for phase-space events
            method: scipy optimization method (default: 'L-BFGS-B')
            options: dict passed to scipy.optimize.minimize
        
        Returns:
            scipy.optimize.OptimizeResult with updated parameters
        """
        import scipy.optimize
        f_fit = self.make_fit_func(data, phsp)
        x0 = self.get_free_values()
        if options is None:
            options = {'disp': True, 'maxiter': 200}
        result = scipy.optimize.minimize(f_fit, x0, jac=True,
                                          method=method, options=options)
        # Update stored params with fitted values
        model_opt = self.unpack(result.x)
        # Convert model → physical params dict
        cst = self._ensure_cst()
        phys = cst.get_physical(model_opt)
        self._params = phys
        result.phys_params = phys
        return result

    # ------------------------------------------------------------------
    # Parameter transforms (delegated to ConstraintMapper)
    # ------------------------------------------------------------------

    @property
    def cst(self):
        return self._ensure_cst()

    def get_free_keys(self):
        return self.cst.get_free_keys()

    def pack(self, values_dict, keys=None):
        return self.cst.pack(values_dict, keys)

    def unpack(self, x, keys=None):
        return self.cst.unpack(x, keys)

    def get_free_values(self, phys_params=None):
        if phys_params is None:
            phys_params = self.params
        return self.cst.free_flat_from_phys(phys_params)

    def make_fit_func(self, data, phsp, N_phsp=None):
        """
        Create a callable for scipy.optimize.minimize with jac=True.
        
        Returns: fun(x) → (neg_log_likelihood, gradient_array)
        """
        cst = self.cst
        gpu = self._ensure_fitter()
        keys = cst.get_free_keys()

        def func(x):
            model_dict = cst.unpack(x, keys)
            ck, mk, gk, sc = cst.to_kernel(model_dict)

            if N_phsp is not None:
                norm = N_phsp
                q_data, grads_k = gpu.compute((ck, mk, gk, *sc), data, norm)
            else:
                # Step 1: compute norm + gradients from phsp
                q_phsp, grads_p = gpu.compute((ck, mk, gk, *sc), phsp, None)
                n_phsp = phsp.n_events
                norm = q_phsp / n_phsp if n_phsp > 0 else 1.0

                # Step 2: compute NLL on data with normalization
                q_data, grads_k = gpu.compute((ck, mk, gk, *sc), data, norm)

                # Step 3: combine gradients: ∂J/∂θ += ∂J/∂N * ∂N/∂θ
                # ∂N/∂θ = (∂q_phsp/∂θ) / N_phsp  (norm = q_phsp / N_phsp)
                # grad_N = ∂J/∂N from kernel (grad_scalar[6])
                grad_N = grads_k.get('N', 0)
                if abs(grad_N) > 0 and n_phsp > 0:
                    scale = grad_N / n_phsp
                    grads_k['ck']   += scale * grads_p['ck']
                    grads_k['m0']   += scale * grads_p['m0']
                    grads_k['g0']   += scale * grads_p['g0']
                    for sk in ['delta_m','delta_g','g','ap','lam','phi']:
                        if sk in grads_k and sk in grads_p:
                            grads_k[sk] = grads_k.get(sk, 0) + scale * grads_p.get(sk, 0)

            nll = -q_data
            # Negate gradients: nll = -q_data, so ∇nll = -∇q_data
            grad_sc = np.array([-(grads_k[k] if k in grads_k and grads_k[k] is not None else 0)
                                for k in ['delta_m','delta_g','g','ap','lam','phi','N']])
            model_grads = cst.from_kernel(-grads_k['ck'], -grads_k['m0'], -grads_k['g0'], grad_sc)
            grad_flat = cst.pack(model_grads, keys)
            return nll, grad_flat

        return func

    def compute(self, data, params=None, N=None):
        """
        Full forward+backward compute via ConstraintMapper.
        
        Builds model params from physical params, applies constraints,
        computes on GPU, and backpropagates gradients through constraints.
        
        Args:
            data: PWAData object
            params: physical params dict (optional, uses self.params if None)
            N: normalization (None for chi-square)
        
        Returns:
            q_val, grads_dict (model gradients, with constraints applied)
        """
        if params is None:
            params = self.params
        model_dict = self._ensure_cst().build_model_dict(params)
        cst = self._ensure_cst()
        fitter = self._ensure_fitter()
        return cst.compute(fitter, data, model_dict, N)

    def to_kernel(self, params=None):
        """Convert physical params to kernel arrays (via ConstraintMapper)."""
        if params is None:
            params = self.params
        model_dict = self._ensure_cst().build_model_dict(params)
        cst = self._ensure_cst()
        return cst.to_kernel(model_dict)

    def from_kernel(self, grad_ck, grad_m0_k, grad_g0_k, grad_scalar):
        """Convert kernel gradients to physical then model gradients."""
        cst = self._ensure_cst()
        return cst.from_kernel(grad_ck, grad_m0_k, grad_g0_k, grad_scalar)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def summary(self):
        """Print configuration summary."""
        if self._cfg is None:
            print("Config not loaded")
            return
        cfg = self._cfg
        print(f"Waves:      {cfg['n_waves']}")
        print(f"n_m0:       {cfg['n_m0']} ({cfg['n_m0_phys']} phys)")
        print(f"n_g0:       {cfg['n_g0']} ({cfg['n_g0_phys']} phys)")
        print(f"n_bf_types: {cfg['n_bf_types']}")
        print(f"n_basis:    {cfg['n_basis']}")
        print(f"Perm:       {cfg.get('n_perm', 1)}")
        if self._mapper:
            m = self._mapper
            print(f"totals:     {len(m.totals)}")
            print(f"g_ls:       {len(m.gls)}")
            print(f"g_lsbar:    {len(m.glsbar)}")
        if self._cst:
            print(f"Constraints: {len(self._cst._constraints)}")
