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
        # Complex format: 'rect' (re, im) or 'polar' (mag, phase)
        self._complex_format = 'polar'

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

    def set_equal_group(self, keys):
        """All keys share the same fit variable (first key is canonical).
        
        Matches same_params pattern from reference fit.
        Usage: cst.set_equal_group(['m0/0', 'm0/1', 'm0/2'])
        """
        if not keys:
            return self
        target = keys[0]
        for k in keys[1:]:
            self._constraints[k] = {'equal_to': target}
        return self

    def set_scale(self, target, source, factor):
        """target = factor * source. Uses set_linear internally."""
        self._constraints[target] = {'linear': [(source, factor)]}
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

        total = phys_arr('total', len(m.totals), lambda i: self._total_key(list(m.totals.keys())[i]))
        gls = phys_arr('g_ls', len(m.gls), lambda i: self._gls_key(*list(m.gls.keys())[i]))
        glsbar = phys_arr('g_lsbar', len(m.glsbar), lambda i: self._glsbar_key(*list(m.glsbar.keys())[i]))
        m0_key_to_idx = {}
        for rname, pidx in m._res_m0_map.items():
            key = self._m0_key(rname)
            if key not in m0_key_to_idx: m0_key_to_idx[key] = pidx
        g0_key_to_idx = {}
        for rname, plist in m._res_g0_map.items():
            for gi, pidx in enumerate(plist):
                key = self._g0_key(rname, plist, gi)
                if key not in g0_key_to_idx: g0_key_to_idx[key] = pidx
        m0_keys_list = list(m0_key_to_idx.keys())
        g0_keys_list = list(g0_key_to_idx.keys())
        m0 = phys_arr('m0', m.n_m0_phys, lambda i: m0_keys_list[i] if i < len(m0_keys_list) else f'm0_{i}')
        g0 = phys_arr('g0', m.n_g0_phys, lambda i: g0_keys_list[i] if i < len(g0_keys_list) else f'g0_{i}')

        # a.json scalar key pattern: {top}_{scalar}  (top='B' by convention)
        _scalar_map = [('B_delta_m', 'delta_m'), ('B_delta_gamma', 'delta_g'),
                       ('B_gamma', 'g'), ('B_A_prod', 'ap'),
                       ('B_poqr', 'lam'), ('B_poqi', 'phi')]
        s = {}
        for ajkey, skey in _scalar_map:
            s[skey] = self._resolve(ajkey, model_params)
        return {
            'total': total, 'g_ls': gls, 'g_lsbar': glsbar,
            'm0': m0, 'g0': g0, **s,
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
                if ptype == 'total': key = self._total_key(list(m.totals.keys())[i])
                elif ptype == 'g_ls': key = self._gls_key(*list(m.gls.keys())[i])
                elif ptype == 'g_lsbar': key = self._glsbar_key(*list(m.glsbar.keys())[i])
                elif ptype == 'm0':
                    rname = next((rn for rn, p in m._res_m0_map.items() if p == i), None)
                    key = self._m0_key(rname) if rname else f'm0_{i}'
                elif ptype == 'g0':
                    rname = next((rn for rn, pl in m._res_g0_map.items() if i in pl), None)
                    if rname:
                        plist = m._res_g0_map.get(rname, [])
                        gi = plist.index(i) if i in plist else 0
                        key = self._g0_key(rname, plist, gi)
                    else:
                        key = f'g0_{i}'
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

        # Scalars: a.json keys {B_delta_m, ...} → model grads
        _skeys = ['B_delta_m', 'B_delta_gamma', 'B_gamma', 'B_A_prod', 'B_poqr', 'B_poqi']
        _mnames = ['delta_m', 'delta_g', 'g', 'ap', 'lam', 'phi']
        for ajkey, mname in zip(_skeys, _mnames):
            c = self._constraints.get(ajkey, {})
            model_grads[mname] = 0 if 'fixed' in c else phys_grads.get(mname, 0)

        model_grads['N'] = phys_grads.get('N', 0)

        return model_grads

    def _add_to(self, target_key, grad_val, arr, ptype, m):
        """Add a gradient contribution to the appropriate array slot."""
        if ptype == 'total':
            # key is {name}_total_0
            for k, v in m.totals.items():
                if target_key.startswith(f'{k}_total_'):
                    arr[v] += grad_val; break
        elif ptype == 'g_ls':
            for (dn, ls), v in m.gls.items():
                if target_key == f'{dn}_g_ls_{ls}':
                    arr[v] += grad_val; break
        elif ptype == 'g_lsbar':
            for (dn, ls), v in m.glsbar.items():
                if target_key == f'{dn}_g_lsbar_{ls}':
                    arr[v] += grad_val; break
        elif ptype == 'm0':
            for rn, pidx in m._res_m0_map.items():
                if target_key == f'{rn}_mass':
                    if pidx < len(arr): arr[pidx] += grad_val; break
        elif ptype == 'g0':
            for rn, plist in m._res_g0_map.items():
                for gi, pidx in enumerate(plist):
                    expected = f'{rn}_width' if len(plist) == 1 else f'{rn}_g_{gi}'
                    if target_key == expected:
                        if pidx < len(arr): arr[pidx] += grad_val; break

    # ------------------------------------------------------------------
    # Parameter management (flat array interface for optimization)
    # ------------------------------------------------------------------

    def build_model_dict(self, phys_params):
        """Build model params dict (individual keys) from physical params dict (arrays).
        
        Inverse of get_physical(): phys_params → {key: value} for to_kernel().
        """
        m = self.mapper
        d = {}
        for k in m.totals: d[self._total_key(k)] = phys_params['total'][m.totals[k]]
        for (dn, ls), i in m.gls.items(): d[self._gls_key(dn, ls)] = phys_params['g_ls'][i]
        for (dn, ls), i in m.glsbar.items(): d[self._glsbar_key(dn, ls)] = phys_params['g_lsbar'][i]
        for rname, pidx in m._res_m0_map.items():
            key = self._m0_key(rname)
            if key not in d and pidx < len(phys_params['m0']):
                d[key] = phys_params['m0'][pidx]
        for rname, plist in m._res_g0_map.items():
            for gi, pidx in enumerate(plist):
                key = self._g0_key(rname, plist, gi)
                if key not in d and pidx < len(phys_params['g0']):
                    d[key] = phys_params['g0'][pidx]
        for k, sk in [('B_delta_m','delta_m'),('B_delta_gamma','delta_g'),('B_gamma','g'),
                       ('B_A_prod','ap'),('B_poqr','lam'),('B_poqi','phi')]:
            d[k] = phys_params[sk]
        return d

    # ------------------------------------------------------------------
    # Key naming (a.json convention from save_params)
    # ------------------------------------------------------------------

    def _total_key(self, name):
        """a.json key for total coupling: {wave_name}_total_0"""
        return f'{name}_total_0'

    def _gls_key(self, dname, ls):
        """a.json key for g_ls: {decay_name}_g_ls_{ls}"""
        return f'{dname}_g_ls_{ls}'

    def _glsbar_key(self, dname, ls):
        """a.json key for g_lsbar: {decay_name}_g_lsbar_{ls}"""
        return f'{dname}_g_lsbar_{ls}'

    def _m0_key(self, res_name):
        """a.json key for mass: {res_name}_mass"""
        return f'{res_name}_mass'

    def _g0_key(self, res_name, plist, gi=0):
        """a.json key for width: {res_name}_width or {res_name}_g_{gi} (Flatte)"""
        return f'{res_name}_width' if len(plist) == 1 else f'{res_name}_g_{gi}'

    def set_complex_format(self, fmt):
        """Set complex parameter encoding: 'rect' (re, im) or 'polar' (mag, phase).
        
        'rect':  flat array stores [re, im, re, im, ...]
        'polar': flat array stores [mag, phase, mag, phase, ...]
        """
        if fmt not in ('rect', 'polar'):
            raise ValueError(f"Unknown format: {fmt}")
        self._complex_format = fmt
        return self

    def _to_complex(self, a, b):
        """Convert two floats to complex based on format."""
        if self._complex_format == 'polar':
            return a * np.exp(1j * b)
        return a + 1j * b

    def _from_complex(self, z):
        """Convert complex to two floats based on format."""
        if self._complex_format == 'polar':
            return abs(z), np.angle(z)
        return z.real, z.imag

    def get_free_keys(self):
        """Return list of free parameter keys matching a.json naming.
        
        Keys match the save_params output format:
            {wave_name}_total_0         — production coupling
            {decay_name}_g_ls_{ls}       — B helicity coupling
            {decay_name}_g_lsbar_{ls}    — Bbar helicity coupling
            {res_name}_mass              — resonance mass
            {res_name}_width             — resonance width
            {res_name}_g_{i}             — FlatteC coupling
            B_delta_m, B_delta_gamma, ... — scalars
        """
        constrained = set(self._constraints.keys())
        m = self.mapper
        free = []
        for k in m.totals: free.append(self._total_key(k))
        for (dn, ls) in m.gls: free.append(self._gls_key(dn, ls))
        for (dn, ls) in m.glsbar: free.append(self._glsbar_key(dn, ls))
        m0_keys = {}
        for rname, pidx in m._res_m0_map.items():
            if pidx not in m0_keys:
                m0_keys[pidx] = self._m0_key(rname)
        for pidx in sorted(m0_keys):
            free.append(m0_keys[pidx])
        g0_keys = {}
        for rname, plist in m._res_g0_map.items():
            for gi, pidx in enumerate(plist):
                key = self._g0_key(rname, plist, gi)
                if pidx not in g0_keys:
                    g0_keys[pidx] = key
        for pidx in sorted(g0_keys):
            free.append(g0_keys[pidx])
        for k in ['B_delta_m', 'B_delta_gamma', 'B_gamma', 'B_A_prod', 'B_poqr', 'B_poqi']:
            free.append(k)
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
                a, b = self._from_complex(v)
                vals.extend([a, b])
            else:
                vals.append(float(v))
        return np.array(vals, dtype=np.float64)

    def unpack(self, x, keys=None):
        """Unpack flat array into model params dict.
        
        Supports both 'rect' (re, im) and 'polar' (mag, phase) formats.
        
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
                d[k] = self._to_complex(x[i], x[i + 1])
                i += 2
            else:
                d[k] = x[i]
                i += 1
        return d

    def _get_from_array(self, arr_dict, key):
        """Extract individual value from array-based grads_dict.
        
        Keys follow a.json convention:
            {name}_total_0, {dname}_g_ls_{ls}, {rname}_mass, ...
        """
        m = self.mapper
        for k, v in m.totals.items():
            if key == f'{k}_total_0':
                return arr_dict['total'][v]
        for (dn, ls), v in m.gls.items():
            if key == f'{dn}_g_ls_{ls}':
                return arr_dict['g_ls'][v]
        for (dn, ls), v in m.glsbar.items():
            if key == f'{dn}_g_lsbar_{ls}':
                return arr_dict['g_lsbar'][v]
        for rn, pidx in m._res_m0_map.items():
            if key == f'{rn}_mass':
                return arr_dict['m0'][pidx] if pidx < len(arr_dict['m0']) else 0.0
        for rn, plist in m._res_g0_map.items():
            for gi, pidx in enumerate(plist):
                expected = f'{rn}_width' if len(plist) == 1 else f'{rn}_g_{gi}'
                if key == expected:
                    return arr_dict['g0'][pidx] if pidx < len(arr_dict['g0']) else 0.0
        # Scalars: map B_delta_m → delta_m, etc.
        _scalar_map = {'B_delta_m': 'delta_m', 'B_delta_gamma': 'delta_g',
                       'B_gamma': 'g', 'B_A_prod': 'ap',
                       'B_poqr': 'lam', 'B_poqi': 'phi'}
        if key in _scalar_map and _scalar_map[key] in arr_dict:
            return arr_dict[_scalar_map[key]]
        if key in arr_dict:
            return arr_dict[key]
        return 0.0

    def get_free_values(self, model_dict=None):
        """Get flat array of free parameter values from a model dict."""
        if model_dict is None:
            model_dict = {}
        return self.pack(model_dict)

    def free_flat_from_phys(self, phys_params):
        """Convenience: physical params dict → flat free-param array."""
        return self.pack(self.build_model_dict(phys_params))

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
