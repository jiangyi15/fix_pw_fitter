"""
Parameter mapper: physical parameters ↔ kernel (ck, m0, g0) arrays.

The kernel works with expanded arrays (448 waves, 80 m0, 92 g0).
Physical parameters are much fewer (42 total, ~90 g_ls, 12 m0_phys, 12 g0_phys).

Forward:  phys_params → ck, m0_kernel, g0_kernel
Backward: grad_ck, grad_m0, grad_g0 → grad_phys_params
"""

import numpy as np
from collections import OrderedDict


class ParamMapper:
    """
    Maps between physical fit parameters and kernel parameter arrays.
    
    Usage:
        mapper = ParamMapper(cfg)
        ck, m0_k, g0_k = mapper.to_kernel(total, gls, glsbar, m0_phys, g0_phys, scalars)
        grads = mapper.from_kernel(grad_ck, grad_m0_k, grad_g0_k, grad_scalar)
    """

    def __init__(self, cfg):
        wave_info = cfg.get('wave_info', [])
        n_perm = cfg.get('n_perm', 1)

        # --- Collect unique physical parameter names ---
        self.totals = OrderedDict()    # name → index
        self.gls = OrderedDict()       # (name, ls) → index
        self.glsbar = OrderedDict()    # (name, ls) → index
        self.ck_formulas = []          # per wave: list of (type, idx)
        self.ck_pw_id = []             # per wave: pw_id for debugging

        for w in wave_info:
            formula = []
            for typ, name, ls_idx in w['ck_formula']:
                if typ == 'total':
                    if name not in self.totals:
                        self.totals[name] = len(self.totals)
                    formula.append(('total', self.totals[name]))
                elif typ == 'g_ls':
                    key = (name, ls_idx)
                    if key not in self.gls:
                        self.gls[key] = len(self.gls)
                    formula.append(('g_ls', self.gls[key]))
                elif typ == 'g_lsbar':
                    key = (name, ls_idx)
                    if key not in self.glsbar:
                        self.glsbar[key] = len(self.glsbar)
                    formula.append(('g_lsbar', self.glsbar[key]))
            self.ck_formulas.append(formula)
            self.ck_pw_id.append(w.get('pw_id', -1))

        # --- m0/g0 phys index maps ---
        self.m0_phys_index = cfg['m0_phys_index']  # (n_m0,) maps kernel → phys
        self.g0_phys_index = cfg['g0_phys_index']  # (n_g0,) maps kernel → phys
        self.n_m0_phys = cfg['n_m0_phys']
        self.n_g0_phys = cfg['n_g0_phys']

        self.n_waves = len(self.ck_formulas)
        self.n_m0 = len(self.m0_phys_index)
        self.n_g0 = len(self.g0_phys_index)

        print(f"  ParamMapper: {len(self.totals)} totals, {len(self.gls)} g_ls, "
              f"{len(self.glsbar)} g_lsbar, {self.n_m0_phys} m0_phys, {self.n_g0_phys} g0_phys")

    # ------------------------------------------------------------------
    # Forward: physical → kernel
    # ------------------------------------------------------------------

    def to_kernel(self, total, gls, glsbar, m0_phys, g0_phys, scalars):
        """
        Convert physical params to kernel arrays.
        
        Args:
            total: complex array (n_totals,) — production couplings
            gls:   complex array (n_gls,) — B decay helicity couplings
            glsbar: complex array (n_glsbar,) — Bbar decay helicity couplings
            m0_phys: float array (n_m0_phys,) — physical masses
            g0_phys: float array (n_g0_phys,) — physical widths
            scalars: tuple of (delta_m, delta_g, g, ap, lam, phi)
        """
        # ck = product of factors per wave (cache for backward)
        ck = np.zeros(self.n_waves, dtype=np.complex128)
        self._cache_ck_vals = []
        self._cache_ck_vals_full = []
        for i, formula in enumerate(self.ck_formulas):
            vals = []
            for typ, idx in formula:
                if typ == 'total':
                    vals.append(total[idx])
                elif typ == 'g_ls':
                    vals.append(gls[idx])
                elif typ == 'g_lsbar':
                    vals.append(glsbar[idx])
            prod = np.prod(vals)
            ck[i] = prod
            self._cache_ck_vals.append(vals)
            self._cache_ck_vals_full.append(prod)

        # m0/g0: index into phys arrays
        m0_k = m0_phys[self.m0_phys_index]
        g0_k = g0_phys[self.g0_phys_index]

        return ck, m0_k, g0_k, tuple(scalars)

    # ------------------------------------------------------------------
    # Backward: kernel gradients → physical gradients
    # ------------------------------------------------------------------

    def from_kernel(self, grad_ck, grad_m0_k, grad_g0_k, grad_scalar):
        """Backward using cached values from last to_kernel call."""
        if not hasattr(self, '_cache_ck_vals'):
            raise RuntimeError("Call to_kernel() before from_kernel()")

        grad_total = np.zeros(len(self.totals), dtype=np.complex128)
        grad_gls = np.zeros(len(self.gls), dtype=np.complex128)
        grad_glsbar = np.zeros(len(self.glsbar), dtype=np.complex128)

        for i, formula in enumerate(self.ck_formulas):
            gck = grad_ck[i]
            prod_full = self._cache_ck_vals_full[i]
            vals = self._cache_ck_vals[i]
            for j, (typ, phys_idx) in enumerate(formula):
                f_j = vals[j]
                if abs(f_j) > 1e-15:
                    contrib = gck * prod_full / f_j
                else:
                    contrib = gck
                    for k, (t2, p2) in enumerate(formula):
                        if k != j:
                            contrib *= vals[k]
                if typ == 'total':
                    grad_total[phys_idx] += contrib
                elif typ == 'g_ls':
                    grad_gls[phys_idx] += contrib
                elif typ == 'g_lsbar':
                    grad_glsbar[phys_idx] += contrib

        # m0/g0: gradient accumulation by phys index
        grad_m0_phys = np.zeros(self.n_m0_phys)
        for ki, pi in enumerate(self.m0_phys_index):
            grad_m0_phys[pi] += grad_m0_k[ki]
        grad_g0_phys = np.zeros(self.n_g0_phys)
        for ki, pi in enumerate(self.g0_phys_index):
            grad_g0_phys[pi] += grad_g0_k[ki]

        return {
            'total': grad_total, 'g_ls': grad_gls, 'g_lsbar': grad_glsbar,
            'm0': grad_m0_phys, 'g0': grad_g0_phys,
            'delta_m': grad_scalar[0], 'delta_g': grad_scalar[1],
            'g': grad_scalar[2], 'ap': grad_scalar[3],
            'lam': grad_scalar[4], 'phi': grad_scalar[5],
            'N': grad_scalar[6],
        }

    def from_kernel_cached(self, grad_ck, grad_m0_k, grad_g0_k, grad_scalar):
        """Backward using cached forward values from last to_kernel call."""
        # Accumulate per physical parameter
        grad_total = np.zeros(len(self.totals), dtype=np.complex128)
        grad_gls = np.zeros(len(self.gls), dtype=np.complex128)
        grad_glsbar = np.zeros(len(self.glsbar), dtype=np.complex128)

        for i, formula in enumerate(self.ck_formulas):
            if not hasattr(self, '_cache_ck_vals'):
                break
            gck = grad_ck[i]
            # For each factor in this wave's ck formula:
            for j, (typ, phys_idx) in enumerate(formula):
                # dJ/df_j = dJ/d(ck) * ∏_{k≠j} f_k = grad_ck * ck / f_j
                # If f_j ≈ 0, this blows up — use full product / f_j from cache
                f_j = self._cache_ck_vals[i][j]
                if abs(f_j) > 1e-15:
                    contrib = gck * self._cache_ck_vals_full[i] / f_j
                else:
                    # f_j = 0, compute product excluding f_j
                    contrib = gck
                    for k, (t2, p2) in enumerate(formula):
                        if k != j:
                            contrib *= self._cache_ck_vals[i][k]
                if typ == 'total':
                    grad_total[phys_idx] += contrib
                elif typ == 'g_ls':
                    grad_gls[phys_idx] += contrib
                elif typ == 'g_lsbar':
                    grad_glsbar[phys_idx] += contrib

        # m0/g0: sum kernel gradients by phys index
        grad_m0_phys = np.zeros(self.n_m0_phys)
        for ki, pi in enumerate(self.m0_phys_index):
            grad_m0_phys[pi] += grad_m0_k[ki]

        grad_g0_phys = np.zeros(self.n_g0_phys)
        for ki, pi in enumerate(self.g0_phys_index):
            grad_g0_phys[pi] += grad_g0_k[ki]

        return {
            'total': grad_total, 'g_ls': grad_gls, 'g_lsbar': grad_glsbar,
            'm0': grad_m0_phys, 'g0': grad_g0_phys,
            'delta_m': grad_scalar[0], 'delta_g': grad_scalar[1],
            'g': grad_scalar[2], 'ap': grad_scalar[3],
            'lam': grad_scalar[4], 'phi': grad_scalar[5],
            'N': grad_scalar[6],
        }

    # ------------------------------------------------------------------
    # Combined forward + backward with caching
    # ------------------------------------------------------------------

    def compute(self, fitter, data, params_dict, N=None):
        """
        Full forward+backward: phys_params → kernel → compute → grads_phys.
        
        Args:
            fitter: PWAGPU instance
            data: PWAData instance
            params_dict: dict with keys 'total', 'g_ls', 'g_lsbar', 
                        'm0', 'g0', and scalars
            N: normalization (None for chi-square)
        
        Returns:
            q_val, phys_grads_dict
        """
        total = params_dict['total']
        gls = params_dict['g_ls']
        glsbar = params_dict['g_lsbar']
        m0_phys = params_dict['m0']
        g0_phys = params_dict['g0']
        scalars = (params_dict.get(k, 0) for k in 
                   ['delta_m', 'delta_g', 'g', 'ap', 'lam', 'phi'])

        # Forward
        ck, m0_k, g0_k, scalars_t = self.to_kernel(total, gls, glsbar, m0_phys, g0_phys, scalars)
        params = (ck, m0_k, g0_k, *scalars_t)

        # Kernel compute
        q_val, grads_k = fitter.compute(params, data, N)

        # Backward
        phys_grads = self.from_kernel_cached(
            grads_k['ck'], grads_k['m0'], grads_k['g0'],
            np.array([grads_k[k] for k in ['delta_m','delta_g','g','ap','lam','phi','N']])
        )

        return q_val, phys_grads


# Shorthand
def create_mapper(cfg):
    return ParamMapper(cfg)
