"""
Unified PWA fitter: wraps config loading, kernel setup, and parameter management.

Usage:
    from pwa_gpu.fitter import PWAFitter
    
    fitter = PWAFitter("config.yml")
    fitter.build_tables()
    fitter.load_params("a.json")
    
    data = fitter.load_data(mass, q, angles, time, frac, weights, bkg)
    q_val, grads = fitter.compute(data, N=norm)
"""

import os
import numpy as np
from pwa_gpu.parse_config import parse_config
from pwa_gpu.build_tables import build_tables as _build_tables
from pwa_gpu.param_mapper import ParamMapper, load_params
from pwa_gpu import PWAGPU, PWAData


class PWAFitter:
    """
    Single class wrapping the full PWA pipeline:
      parse_config → build_tables → ParamMapper → PWAGPU → compute
    
    All methods are accessible directly on this object.
    """

    def __init__(self, config_path):
        self.config_path = config_path
        self._cfg = None
        self._pw_list = None
        self._kw_list = None
        self._mapper = None
        self._fitter = None
        self._params = None  # last loaded params

        # Step 1: parse config
        self.parse()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def parse(self):
        """Parse the YAML config (called automatically on init)."""
        self._cfg, self._pw_list, self._kw_list = parse_config(self.config_path)
        return self

    def build_tables(self):
        """Build gamma/bf interpolation tables."""
        self._cfg = _build_tables(self._cfg, self._pw_list, self._kw_list, self.config_path)
        self._mapper = ParamMapper(self._cfg)
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
    # Parameters
    # ------------------------------------------------------------------

    def load_params(self, json_path):
        """Load fitted parameters from a.json into the mapper."""
        self._params = load_params(json_path, self.mapper, self.config_path)
        return self._params

    @property
    def params(self):
        """Last loaded params dict, or random defaults."""
        if self._params is None:
            # Generate random defaults
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

    def params_from_kernel(self, ck, m0_k, g0_k):
        """
        Convert kernel arrays back to physical params.
        Inverse of mapper.to_kernel (up to g_ls/g_lsbar ambiguity).
        Only useful for m0/g0 which are just index maps.
        """
        mapper = self.mapper
        # m0/g0: sum by phys index
        m0_phys = np.zeros(mapper.n_m0_phys)
        g0_phys = np.zeros(mapper.n_g0_phys)
        for ki, pi in enumerate(mapper.m0_phys_index):
            m0_phys[pi] = m0_k[ki]
        for ki, pi in enumerate(mapper.g0_phys_index):
            g0_phys[pi] = g0_k[ki]
        return {'m0': m0_phys, 'g0': g0_phys,
                'ck': ck, 'm0_k': m0_k, 'g0_k': g0_k}

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
        """Upload data to GPU."""
        fitter = self._ensure_fitter()
        return PWAData(fitter, mass, q, angles, time_arr, frac, weights, bkg)

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def to_kernel(self, params=None):
        """Convert physical params to kernel arrays."""
        mapper = self.mapper
        if params is None:
            params = self.params
        scalars = (params['delta_m'], params['delta_g'], params['g'],
                   params['ap'], params['lam'], params['phi'])
        return mapper.to_kernel(
            params['total'], params['g_ls'], params['g_lsbar'],
            params['m0'], params['g0'], scalars)

    def from_kernel(self, grad_ck, grad_m0_k, grad_g0_k, grad_scalar):
        """Convert kernel gradients to physical gradients."""
        return self.mapper.from_kernel_cached(
            grad_ck, grad_m0_k, grad_g0_k, grad_scalar)

    def compute(self, data, params=None, N=None):
        """
        Full forward+backward compute.
        
        Args:
            data: PWAData object
            params: dict with 'total', 'g_ls', 'g_lsbar', 'm0', 'g0',
                    'delta_m', 'delta_g', 'g', 'ap', 'lam', 'phi'
            N: normalization (None for chi-square)
        
        Returns:
            q_val, grads_dict
        """
        if params is None:
            params = self.params
        fitter = self._ensure_fitter()
        return self.mapper.compute(fitter, data, params, N)

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
