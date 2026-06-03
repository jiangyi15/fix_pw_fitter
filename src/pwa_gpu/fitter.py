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

    def _build_model_dict(self, phys_params=None):
        """Build model params dict from physical params (one-to-one by default)."""
        if phys_params is None:
            phys_params = self.params
        m = self.mapper
        d = {}
        for k in m.totals: d[f'total/{k}'] = phys_params['total'][m.totals[k]]
        for (dn, ls), i in m.gls.items(): d[f'g_ls/{dn}/{ls}'] = phys_params['g_ls'][i]
        for (dn, ls), i in m.glsbar.items(): d[f'g_lsbar/{dn}/{ls}'] = phys_params['g_lsbar'][i]
        for i in range(m.n_m0_phys): d[f'm0/{i}'] = phys_params['m0'][i]
        for i in range(m.n_g0_phys): d[f'g0/{i}'] = phys_params['g0'][i]
        for k in ['delta_m','delta_g','g','ap','lam','phi']: d[k] = phys_params[k]
        return d

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
    # Full likelihood fit via ConstraintMapper
    # ------------------------------------------------------------------

    def fit(self, data, phsp, params=None, N_phsp=None):
        """
        Compute full likelihood with normalization from phase space.
        
        Uses ConstraintMapper for all parameter transforms (constraints applied).
        
        Args:
            data: PWAData for signal events
            phsp: PWAData for phase-space events (normalization integral)
            params: physical params dict (optional, uses self.params if None)
            N_phsp: norm override. If None, computed as sum(|A|²)/N_phsp_events
        
        Returns:
            (neg_log_likelihood, model_grads, norm_value)
            where model_grads has constraints applied (fixed→0, equal→accumulated)
        """
        if params is None:
            params = self.params
        model_dict = self._build_model_dict(params)
        cst = self._ensure_cst()
        fitter = self._ensure_fitter()

        # Step 1: compute normalization from phase space
        ck, mk, gk, sc = cst.to_kernel(model_dict)
        
        if N_phsp is not None:
            norm = N_phsp
        else:
            # Compute norm = Σ |A|² / N_phsp (chi-square mode on phsp)
            q_phsp, _ = fitter.compute((ck, mk, gk, *sc), phsp, None)
            n_phsp = phsp.n_events
            norm = q_phsp / n_phsp if n_phsp > 0 else 1.0

        # Step 2: compute NLL on data with normalization
        # kernel does: q = -Σ w * log(p/N + bkg) with likelihood mode
        q_data, grads_k = fitter.compute((ck, mk, gk, *sc), data, norm)

        # Step 3: propagate gradients through constraints
        grad_scalar = np.array([grads_k[k] for k in ['delta_m','delta_g','g','ap','lam','phi','N']])
        nll = -q_data  # negative log-likelihood
        model_grads = cst.from_kernel(grads_k['ck'], grads_k['m0'], grads_k['g0'], grad_scalar)

        return nll, model_grads, norm

    # ------------------------------------------------------------------
    # Optimization interface (scipy.optimize.minimize compatible)
    # ------------------------------------------------------------------
    # Free parameters: all keys that are NOT targets of any constraint.
    # The fit function takes a flat array x of free param values,
    # returns (neg_log_likelihood, gradient_array).

    def get_free_keys(self):
        """Return list of free parameter keys (not constrained)."""
        mapper = self.mapper
        cst = self._ensure_cst()
        constrained = set(cst._constraints.keys())
        free = []
        # All physical param keys
        for k in mapper.totals: free.append(f'total/{k}')
        for (dn, ls) in mapper.gls: free.append(f'g_ls/{dn}/{ls}')
        for (dn, ls) in mapper.glsbar: free.append(f'g_lsbar/{dn}/{ls}')
        for i in range(mapper.n_m0_phys): free.append(f'm0/{i}')
        for i in range(mapper.n_g0_phys): free.append(f'g0/{i}')
        for k in ['delta_m','delta_g','g','ap','lam','phi']: free.append(k)
        # Remove constrained keys
        free = [k for k in free if k not in constrained]
        return free

    def _get_from_grads(self, grads_dict, key):
        """Extract a value from grads_dict (which has array keys like 'total')."""
        mapper = self.mapper
        if key.startswith('total/'):
            name = key[6:]
            return grads_dict['total'][mapper.totals.get(name, 0)]
        elif key.startswith('g_ls/'):
            parts = key[5:].split('/')
            if len(parts) >= 2:
                return grads_dict['g_ls'][mapper.gls.get((parts[0], int(parts[1])), 0)]
        elif key.startswith('g_lsbar/'):
            parts = key[8:].split('/')
            if len(parts) >= 2:
                return grads_dict['g_lsbar'][mapper.glsbar.get((parts[0], int(parts[1])), 0)]
        elif key.startswith('m0/'):
            idx = int(key[3:])
            return grads_dict['m0'][idx] if idx < len(grads_dict['m0']) else 0.0
        elif key.startswith('g0/'):
            idx = int(key[3:])
            return grads_dict['g0'][idx] if idx < len(grads_dict['g0']) else 0.0
        elif key in grads_dict:
            return grads_dict[key]
        return 0.0

    def pack(self, params_dict, keys=None):
        """Pack a subset of params into a flat numpy array.
        Args:
            params_dict: dict with keys like 'total', 'g_ls', etc. (arrays)
                        OR flat dict with keys like 'total/B->rhoA.rhoB'
            keys: list of keys to pack (default: free keys)
        Returns: flat float64 array (complex→[re, im] pairs)
        """
        if keys is None:
            keys = self.get_free_keys()
        # Detect format: array-based or key-based
        has_arrays = any(k in ('total','g_ls','g_lsbar','m0','g0') for k in params_dict)
        vals = []
        for k in keys:
            if has_arrays:
                v = self._get_from_grads(params_dict, k)
            else:
                v = params_dict.get(k, 0)
            if isinstance(v, (complex, np.complexfloating)):
                vals.extend([v.real, v.imag])
            else:
                vals.append(float(v))
        return np.array(vals, dtype=np.float64)

    def unpack(self, x, keys=None):
        """Unpack flat array into model params dict (individual keys).
        Args:
            x: flat numpy array
            keys: list of keys (default: free keys)
        Returns: dict of {key: value} for constraint_mapper.to_kernel()
        """
        if keys is None:
            keys = self.get_free_keys()
        d = {}
        i = 0
        for k in keys:
            orig = self._get_original_type(k)
            if orig == 'complex':
                d[k] = x[i] + 1j * x[i+1]
                i += 2
            else:
                d[k] = x[i]
                i += 1
        return d

    def _get_original_type(self, key):
        """Check if a parameter is complex."""
        mapper = self.mapper
        if key.startswith('total/') or key.startswith('g_ls/') or key.startswith('g_lsbar/'):
            return 'complex'
        return 'real'

    def get_free_values(self, params_dict=None):
        """Get flat initial values for free parameters.
        Args:
            params_dict: source params (default: self.params)
        Returns: flat float64 array
        """
        if params_dict is None:
            params_dict = self.params
        model_dict = self._build_model_dict(params_dict)
        return self.pack(model_dict)

    def make_fit_func(self, data, phsp, N_phsp=None):
        """
        Create a callable for scipy.optimize.minimize.
        
        Returns: fun(x) → (neg_log_likelihood, gradient_array)
        """
        fitter = self
        mapper = self.mapper
        cst = self._ensure_cst()
        keys = self.get_free_keys()

        def func(x):
            # Unpack x → model dict
            model_dict = fitter.unpack(x, keys)
            # Forward: model → kernel
            ck, mk, gk, sc = cst.to_kernel(model_dict)
            gpu = fitter._ensure_fitter()

            # Norm from phsp
            if N_phsp is not None:
                norm = N_phsp
            else:
                q_phsp, _ = gpu.compute((ck, mk, gk, *sc), phsp, None)
                n_phsp = phsp.n_events
                norm = q_phsp / n_phsp if n_phsp > 0 else 1.0

            # NLL on data
            q_data, grads_k = gpu.compute((ck, mk, gk, *sc), data, norm)
            nll = -q_data

            # Gradients through constraints
            grad_scalar = np.array([grads_k[k] for k in ['delta_m','delta_g','g','ap','lam','phi','N']])
            model_grads = cst.from_kernel(
                grads_k['ck'], grads_k['m0'], grads_k['g0'], grad_scalar)

            # Pack gradients into flat array (same order as x)
            grad_flat = fitter.pack(model_grads, keys)

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
        model_dict = self._build_model_dict(params)
        cst = self._ensure_cst()
        fitter = self._ensure_fitter()
        return cst.compute(fitter, data, model_dict, N)

    def to_kernel(self, params=None):
        """Convert physical params to kernel arrays (via ConstraintMapper)."""
        if params is None:
            params = self.params
        model_dict = self._build_model_dict(params)
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
