"""
ConstraintMapper: model parameters → physical parameters → kernel arrays.

Uses ParamMapper internally for the physical ↔ kernel transform.
Adds a constraint layer: fix, equal, linear on physical parameters.

Architecture:
  Model params → [Constraints] → Physical params → [ParamMapper] → Kernel arrays
  Model grads ← [Constraints⁻¹] ← Physical grads ← [ParamMapper⁻¹] ← Kernel grads
"""

import numpy as np


class ConstraintMapper:
    """
    Adds constraints (fix, equal, linear) on top of ParamMapper.

    Physical parameter naming (same as ParamMapper):
        total/{name}            — complex production coupling
        g_ls/{dname}/{ls}       — complex B helicity coupling
        g_lsbar/{dname}/{ls}    — complex Bbar helicity coupling
        m0/{idx}                — real mass
        g0/{idx}                — real width/scale
        delta_m, delta_g, g, ap, lam, phi — scalars

    Usage:
        cst = ConstraintMapper(mapper)
        
        cst.set_fixed('delta_m', 0.0)
        cst.set_equal('m0/0', 'm0/1')
        cst.set_linear('m0/5', [('m0/0', 0.5), ('m0/1', 0.5)])
        
        # Forward: model params → kernel
        ck, mk, gk, sc = cst.to_kernel(model_params)
        
        # Backward: kernel grads → model grads
        model_grads = cst.from_kernel(grad_ck, grad_m0, grad_g0, grad_scalar)
    """

    def __init__(self, mapper):
        self.mapper = mapper
        # Constraints: {key: {'fixed': val|True} | {'equal_to': key} | {'linear': [(k,c),...]}}
        self._constraints = {}

    # ------------------------------------------------------------------
    # Setting constraints
    # ------------------------------------------------------------------

    def set_fixed(self, key, value=None):
        """Fix a parameter. Removed from fit variables."""
        self._constraints[key] = {'fixed': value if value is not None else True}
        return self

    def set_equal(self, target, source):
        """target = source (share one fit variable)."""
        self._constraints[target] = {'equal_to': source}
        return self

    def set_linear(self, target, sources):
        """target = Σ coeff_i * source_i. sources: list of (key, coeff)."""
        self._constraints[target] = {'linear': sources}
        return self

    def clear(self):
        """Remove all constraints."""
        self._constraints.clear()
        return self

    # ------------------------------------------------------------------
    # Forward: model params → physical params → kernel
    # ------------------------------------------------------------------

    def _resolve(self, key, model_params):
        """Get value of a parameter, following constraints."""
        c = self._constraints.get(key, {})
        if 'fixed' in c:
            return c['fixed'] if not isinstance(c['fixed'], bool) else model_params.get(key, 0)
        if 'equal_to' in c:
            return self._resolve(c['equal_to'], model_params)
        if 'linear' in c:
            return sum(sc * self._resolve(sk, model_params) for sk, sc in c['linear'])
        return model_params.get(key, 0)

    def get_physical(self, model_params):
        """Convert model params dict to physical params dict."""
        m = self.mapper

        def phys_arr(ptype, size, key_fn):
            arr = np.zeros(size, dtype=np.complex128 if ptype in ('total', 'g_ls', 'g_lsbar') else np.float64)
            for i in range(size):
                kidx = key_fn(i)
                if kidx:
                    arr[i] = self._resolve(kidx, model_params)
            return arr

        total = phys_arr('total', len(m.totals), lambda i: f'total/{list(m.totals.keys())[i]}')
        gls = phys_arr('g_ls', len(m.gls), lambda i: f'g_ls/{list(m.gls.keys())[i][0]}/{list(m.gls.keys())[i][1]}')
        glsbar = phys_arr('g_lsbar', len(m.glsbar), lambda i: f'g_lsbar/{list(m.glsbar.keys())[i][0]}/{list(m.glsbar.keys())[i][1]}')
        m0 = phys_arr('m0', m.n_m0_phys, lambda i: f'm0/{i}')
        g0 = phys_arr('g0', m.n_g0_phys, lambda i: f'g0/{i}')

        return {
            'total': total, 'g_ls': gls, 'g_lsbar': glsbar,
            'm0': m0, 'g0': g0,
            'delta_m': self._resolve('delta_m', model_params),
            'delta_g': self._resolve('delta_g', model_params),
            'g': self._resolve('g', model_params),
            'ap': self._resolve('ap', model_params),
            'lam': self._resolve('lam', model_params),
            'phi': self._resolve('phi', model_params),
        }

    def to_kernel(self, model_params):
        """Model params → physical params → kernel arrays."""
        phys = self.get_physical(model_params)
        return self.mapper.to_kernel(
            phys['total'], phys['g_ls'], phys['g_lsbar'],
            phys['m0'], phys['g0'],
            (phys['delta_m'], phys['delta_g'], phys['g'],
             phys['ap'], phys['lam'], phys['phi']))

    # ------------------------------------------------------------------
    # Backward: kernel grads → physical grads → model grads
    # ------------------------------------------------------------------

    def from_kernel(self, grad_ck, grad_m0, grad_g0, grad_scalar):
        """Kernel grads → physical grads → model grads (with constraints)."""
        # Step 1: kernel → physical (via ParamMapper)
        phys_grads = self.mapper.from_kernel_cached(grad_ck, grad_m0, grad_g0, grad_scalar)
        # Step 2: physical → model (apply constraint backprop)
        return self._physical_to_model_grads(phys_grads)

    def _physical_to_model_grads(self, phys_grads):
        """Backpropagate physical gradients through constraints to get model gradients."""
        m = self.mapper
        model_grads = {}

        for ptype in ['total', 'g_ls', 'g_lsbar', 'm0', 'g0']:
            size = {'total': len(m.totals), 'g_ls': len(m.gls),
                    'g_lsbar': len(m.glsbar), 'm0': m.n_m0_phys, 'g0': m.n_g0_phys}[ptype]
            arr = np.zeros_like(phys_grads[ptype])

            for i in range(size):
                # Build the constraint key
                if ptype == 'total': key = f'total/{list(m.totals.keys())[i]}'
                elif ptype == 'g_ls': key = f'g_ls/{list(m.gls.keys())[i][0]}/{list(m.gls.keys())[i][1]}'
                elif ptype == 'g_lsbar': key = f'g_lsbar/{list(m.glsbar.keys())[i][0]}/{list(m.glsbar.keys())[i][1]}'
                elif ptype in ('m0', 'g0'): key = f'{ptype}/{i}'
                else: key = ''

                c = self._constraints.get(key, {})
                if 'fixed' in c:
                    arr[i] = 0  # no gradient for fixed params
                elif 'equal_to' in c:
                    # Gradient accumulates at the source
                    target_key = c['equal_to']
                    # Find target index and add
                    self._add_to(target_key, phys_grads[ptype][i], arr, ptype, m)
                elif 'linear' in c:
                    # Distribute gradient: ∂/∂source = ∂/∂target * coeff
                    for sk, coeff in c['linear']:
                        self._add_to(sk, phys_grads[ptype][i] * coeff, arr, ptype, m)
                else:
                    arr[i] = phys_grads[ptype][i]

            model_grads[ptype] = arr

        # Scalars pass through (unless fixed)
        for sname in ['delta_m', 'delta_g', 'g', 'ap', 'lam', 'phi']:
            c = self._constraints.get(sname, {})
            model_grads[sname] = 0 if 'fixed' in c else phys_grads.get(sname, 0)

        model_grads['N'] = phys_grads.get('N', 0)

        return model_grads

    def _add_to(self, target_key, grad_val, arr, ptype, m):
        """Add a gradient contribution to the appropriate array slot."""
        parts = target_key.split('/')
        if ptype == 'total':
            if parts[0] == 'total' and parts[1] in m.totals:
                arr[m.totals[parts[1]]] += grad_val
        elif ptype == 'g_ls':
            if len(parts) >= 3:
                dname, ls = parts[1], int(parts[2])
                if (dname, ls) in m.gls:
                    arr[m.gls[(dname, ls)]] += grad_val
        elif ptype == 'g_lsbar':
            if len(parts) >= 3:
                dname, ls = parts[1], int(parts[2])
                if (dname, ls) in m.glsbar:
                    arr[m.glsbar[(dname, ls)]] += grad_val
        elif ptype in ('m0', 'g0'):
            if len(parts) >= 2:
                idx = int(parts[1])
                if idx < len(arr):
                    arr[idx] += grad_val

    # ------------------------------------------------------------------
    # Parameter management (flat array interface for optimization)
    # ------------------------------------------------------------------

    def get_free_keys(self):
        """Return list of free parameter keys (not constrained / not derived)."""
        constrained = set(self._constraints.keys())
        m = self.mapper
        free = []
        for k in m.totals: free.append(f'total/{k}')
        for (dn, ls) in m.gls: free.append(f'g_ls/{dn}/{ls}')
        for (dn, ls) in m.glsbar: free.append(f'g_lsbar/{dn}/{ls}')
        for i in range(m.n_m0_phys): free.append(f'm0/{i}')
        for i in range(m.n_g0_phys): free.append(f'g0/{i}')
        for k in ['delta_m','delta_g','g','ap','lam','phi']: free.append(k)
        return [k for k in free if k not in constrained]

    def _is_complex_key(self, key):
        """Check if a parameter key is complex-valued."""
        return key.startswith('total/') or key.startswith('g_ls/') or key.startswith('g_lsbar/')

    def pack(self, values_dict, keys=None):
        """Pack selected parameter values into a flat float64 array.
        
        Complex → [re, im] pairs. Real → single value.
        
        Args:
            values_dict: dict with keys like 'total', 'g_ls', etc. (arrays),
                         or flat dict with keys like 'total/B->rhoA.rhoB'
            keys: list of keys to pack (default: free keys)
        Returns: flat float64 array
        """
        if keys is None:
            keys = self.get_free_keys()
        has_arrays = any(k in ('total','g_ls','g_lsbar','m0','g0') for k in values_dict)
        vals = []
        for k in keys:
            if has_arrays:
                v = self._get_from_array(values_dict, k)
            else:
                v = values_dict.get(k, 0)
            if self._is_complex_key(k):
                vals.extend([v.real, v.imag])
            else:
                vals.append(float(v))
        return np.array(vals, dtype=np.float64)

    def unpack(self, x, keys=None):
        """Unpack flat array into model params dict.
        
        Args:
            x: flat float64 array
            keys: list of keys (default: free keys)
        Returns: dict of {key: scalar_or_complex} for to_kernel()
        """
        if keys is None:
            keys = self.get_free_keys()
        d = {}
        i = 0
        for k in keys:
            if self._is_complex_key(k):
                d[k] = x[i] + 1j * x[i + 1]
                i += 2
            else:
                d[k] = x[i]
                i += 1
        return d

    def _get_from_array(self, arr_dict, key):
        """Extract individual value from array-based grads_dict."""
        m = self.mapper
        if key.startswith('total/'):
            return arr_dict['total'][m.totals.get(key[6:], 0)]
        if key.startswith('g_ls/'):
            p = key[5:].split('/')
            if len(p) >= 2:
                return arr_dict['g_ls'][m.gls.get((p[0], int(p[1])), 0)]
        if key.startswith('g_lsbar/'):
            p = key[8:].split('/')
            if len(p) >= 2:
                return arr_dict['g_lsbar'][m.glsbar.get((p[0], int(p[1])), 0)]
        if key.startswith('m0/'):
            idx = int(key[3:])
            return arr_dict['m0'][idx] if idx < len(arr_dict['m0']) else 0.0
        if key.startswith('g0/'):
            idx = int(key[3:])
            return arr_dict['g0'][idx] if idx < len(arr_dict['g0']) else 0.0
        if key in arr_dict:
            return arr_dict[key]
        return 0.0

    def get_free_values(self, model_dict=None):
        """Get flat array of free parameter values from a model dict."""
        if model_dict is None:
            model_dict = {}
        return self.pack(model_dict)

    # ------------------------------------------------------------------
    # Full compute pipeline
    # ------------------------------------------------------------------

    def compute(self, fitter, data, model_params, N=None):
        """Model params → kernel → compute → model grads."""
        ck, mk, gk, sc = self.to_kernel(model_params)
        params = (ck, mk, gk, *sc)
        q_val, grads_k = fitter.compute(params, data, N)
        grad_scalar = np.array([grads_k[k] for k in ['delta_m','delta_g','g','ap','lam','phi','N']])
        model_grads = self.from_kernel(grads_k['ck'], grads_k['m0'], grads_k['g0'], grad_scalar)
        return q_val, model_grads
