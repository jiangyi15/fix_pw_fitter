"""
Parameter manager: handles constraints, fixed params, and linear transforms
between model parameters and physical parameters.

Physical params = mapper expects: total (42c), g_ls (31c), g_lsbar (3c), 
                                   m0 (12r), g0 (15r), + 6 scalars
Model params = fit variables: fewer after applying constraints.

Usage:
    mgr = ParamManager(mapper)
    mgr.fix('m0', 0, 0.769)           # fix rho mass
    mgr.equal(['g_ls', ('B',0), ('B',1)])  # share g_ls[0] and g_ls[1]
    
    phys = mgr.to_physical()           # → params dict for mapper.compute()
    model_grad = mgr.to_model_grad(grads)  # → gradients for fitter
"""

import numpy as np
from collections import OrderedDict


class ParamManager:
    """
    Manages parameter constraints (fix, equal, linear) for the fit.
    
    Internal naming:
        total/{name}          — complex production coupling
        g_ls/{dname}/{ls}     — complex B helicity coupling
        g_lsbar/{dname}/{ls}  — complex Bbar helicity coupling
        m0/{idx}              — real mass
        g0/{idx}              — real width/scale
        {scalar_name}         — real scalar
    
    Usage:
        mgr = ParamManager(mapper)
        mgr.load(params)
        mgr.fix('rhoA_mass', 0.769)         # fix by alias
        mgr.set_alias('rhoA_mass', 'm0/2')  # add custom alias
        mgr.equal(['rhoA_mass', 'rhoB_mass'])
    """

    def __init__(self, mapper):
        self.mapper = mapper
        self._alias = {}  # external_name → internal_name
        self._params = OrderedDict()
        self._model_params = OrderedDict()
        self._constraints = []

        # Register all physical parameters (internal names only)
        for name, idx in mapper.totals.items():
            self._params[f'total/{name}'] = {'type': 'total', 'idx': idx, 'is_real': False, 'value': 0j}
        for (dname, ls), idx in mapper.gls.items():
            self._params[f'g_ls/{dname}/{ls}'] = {'type': 'g_ls', 'idx': idx, 'is_real': False, 'value': 0j}
        for (dname, ls), idx in mapper.glsbar.items():
            self._params[f'g_lsbar/{dname}/{ls}'] = {'type': 'g_lsbar', 'idx': idx, 'is_real': False, 'value': 0j}
        for idx in range(mapper.n_m0_phys):
            self._params[f'm0/{idx}'] = {'type': 'm0', 'idx': idx, 'is_real': True, 'value': 0.0}
        for idx in range(mapper.n_g0_phys):
            self._params[f'g0/{idx}'] = {'type': 'g0', 'idx': idx, 'is_real': True, 'value': 0.0}
        for name in ['delta_m', 'delta_g', 'g', 'ap', 'lam', 'phi']:
            self._params[name] = {'type': name, 'idx': 0, 'is_real': True, 'value': 0.0}

        # Default: identity mapping
        self._reset()

    def _reset(self):
        """Reset to identity mapping (no constraints)."""
        self._model_params = OrderedDict()
        self._constraints = []
        for key in self._params:
            self._model_params[key] = key  # identity

    # ------------------------------------------------------------------
    # Aliases (external name → internal name mapping)
    # ------------------------------------------------------------------

    def set_alias(self, external, internal):
        """Add an alias mapping external name → internal parameter name.
        
        Example: mgr.set_alias('rhoA_mass', 'm0/2')
                 mgr.set_alias('B_delta_m', 'delta_m')
        """
        if internal not in self._params:
            raise KeyError(f"Unknown internal parameter: {internal}")
        self._alias[external] = internal
        return self

    def add_aliases_from_json(self, json_path):
        """Read a.json parameter names and create aliases for them.
        
        Uses the same matching logic as load_params to connect a.json names
        to internal parameter names.
        """
        import json
        with open(json_path) as f:
            data = json.load(f)
        mapper = self.mapper

        # totals: match by first resonance prefix
        for tname in mapper.totals:
            for jkey in data:
                if jkey.startswith(tname) and '_total_0' in jkey:
                    base = jkey[:-1]  # strip r/i
                    self._alias[base] = f'total/{tname}'
                    break

        # g_ls: match by decay parent before '_g_ls_'
        for (dname, ls) in mapper.gls:
            pat = f'_g_ls_{ls}'
            for jkey in data:
                if pat in jkey and '_g_lsbar_' not in jkey:
                    prefix = jkey.split(pat)[0]
                    parent = prefix.split('->')[0] if '->' in prefix else prefix
                    if parent == dname:
                        self._alias[f'{prefix}{pat}'] = f'g_ls/{dname}/{ls}'
                        break

        # g_lsbar
        for (dname, ls) in mapper.glsbar:
            pat = f'_g_lsbar_{ls}'
            for jkey in data:
                if pat in jkey:
                    prefix = jkey.split(pat)[0]
                    parent = prefix.split('->')[0] if '->' in prefix else prefix
                    if parent == dname:
                        self._alias[f'{prefix}{pat}'] = f'g_lsbar/{dname}/{ls}'
                        break

        # m0, g0: match by resonance name + suffix
        for jkey in data:
            if jkey.endswith('_mass'):
                rname = jkey[:-5]  # remove '_mass'
                # Find which m0/{idx} has this resonance name
                # We need particle config to map rname → phys index
                # For now, skip — user adds these manually via set_alias
                pass
            if jkey.endswith('_width'):
                rname = jkey[:-6]
                pass

        # Scalars
        scalar_map = {'B_delta_m': 'delta_m', 'B_delta_gamma': 'delta_g',
                      'B_gamma': 'g', 'B_poqr': 'lam', 'B_poqi': 'phi',
                      'B_A_prod': 'ap'}
        for jname, internal in scalar_map.items():
            if jname in data:
                self._alias[jname] = internal

        return self

    def _resolve(self, key):
        """Resolve external name → internal name."""
        if key in self._params:
            return key
        if key in self._alias:
            return self._alias[key]
        raise KeyError(f"Unknown parameter: {key}")

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def fix(self, *args):
        """Fix a parameter to a value.
        
        Args:
            mgr.fix('rhoA_mass')           # fix to current value
            mgr.fix('rhoA_mass', 0.769)    # fix to 0.769
            mgr.fix('B_delta_m', 0.0)      # fix delta_m to 0
        """
        key = self._resolve(args[0])
        if len(args) > 1:
            val = args[1]
            self._params[key]['value'] = val
            self._params[key]['fixed'] = val
        else:
            self._params[key]['fixed'] = True
        if key in self._model_params:
            del self._model_params[key]
        return self

    def unfix(self, key):
        """Unfix a parameter (make it a fit variable again)."""
        key = self._resolve(key)
        self._params[key].pop('fixed', None)
        if key not in self._model_params:
            self._model_params[key] = key
        return self

    def equal(self, keys):
        """
        Make multiple parameters share one fit variable.
        
        Usage: mgr.equal(['rhoA_mass', 'rhoB_mass'])
        """
        resolved = [self._resolve(k) for k in keys]
        target = resolved[0]
        for key in resolved:
            if key in self._params:
                self._model_params[key] = target
        return self

    def linear(self, target_key, sources):
        """
        Linear combination: target = sum(coeff_i * source_i).
        
        Usage: mgr.linear('some_param', [('rhoA_mass', 0.5), ('rhoB_mass', 0.5)])
        """
        target_key = self._resolve(target_key)
        sources = [(self._resolve(k), c) for k, c in sources]
        if target_key not in self._params:
            raise KeyError(f"Unknown parameter: {target_key}")
        self._constraints.append(('linear', target_key, sources))
        if target_key in self._model_params:
            del self._model_params[target_key]
        return self

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def load(self, params_dict):
        """Load values from a params dict (as returned by load_params)."""
        for key, info in self._params.items():
            if info['type'] in ('total', 'g_ls', 'g_lsbar'):
                arr = params_dict[info['type']]
                if info['idx'] < len(arr):
                    info['value'] = arr[info['idx']]
            elif info['type'] in ('m0', 'g0'):
                arr = params_dict[info['type']]
                if info['idx'] < len(arr):
                    info['value'] = arr[info['idx']]
            elif info['type'] in params_dict:
                info['value'] = params_dict[info['type']]
        return self

    def get_params_dict(self):
        """Build physical params dict for mapper.compute()."""
        return {
            'total': self._build_array('total'),
            'g_ls': self._build_array('g_ls'),
            'g_lsbar': self._build_array('g_lsbar'),
            'm0': self._build_array('m0'),
            'g0': self._build_array('g0'),
            'delta_m': self._get_scalar('delta_m'),
            'delta_g': self._get_scalar('delta_g'),
            'g': self._get_scalar('g'),
            'ap': self._get_scalar('ap'),
            'lam': self._get_scalar('lam'),
            'phi': self._get_scalar('phi'),
        }

    def _build_array(self, ptype):
        """Build a physical parameter array of the given type."""
        mapper = self.mapper
        if ptype == 'total':    size = len(mapper.totals)
        elif ptype == 'g_ls':   size = len(mapper.gls)
        elif ptype == 'g_lsbar': size = len(mapper.glsbar)
        elif ptype == 'm0':     size = mapper.n_m0_phys
        elif ptype == 'g0':     size = mapper.n_g0_phys
        else: return np.array([])

        dtype = np.complex128 if ptype in ('total', 'g_ls', 'g_lsbar') else np.float64
        arr = np.zeros(size, dtype=dtype)
        for key, info in self._params.items():
            if info['type'] == ptype:
                # Apply constraints: fixed values or model param mapping
                val = self._get_value(key)
                if info['is_real']:
                    arr[info['idx']] = val.real if isinstance(val, complex) else val
                else:
                    arr[info['idx']] = val
        return arr

    def _get_value(self, key):
        """Get the value of a parameter, applying constraints."""
        info = self._params[key]
        # Check if fixed
        if 'fixed' in info:
            if isinstance(info['fixed'], bool):
                return info['value']  # use current value
            return info['fixed']  # use specified value

        # Check if mapped to another variable (equal constraint)
        mapped_to = self._model_params.get(key, key)
        if mapped_to != key:
            return self._get_value(mapped_to)

        # Check linear constraints
        for ctype, tkey, sources in self._constraints:
            if tkey == key:
                val = 0.0 + 0j if not info['is_real'] else 0.0
                for skey, coeff in sources:
                    val += coeff * self._get_value(skey)
                return val

        return info['value']

    def _get_scalar(self, name):
        try:
            return self._params[name]['value']
        except KeyError:
            return 0.0

    def set(self, key, value):
        """Set a parameter value directly."""
        if key in self._params:
            self._params[key]['value'] = value
        return self

    # ------------------------------------------------------------------
    # Gradient backpropagation
    # ------------------------------------------------------------------

    def to_model_grad(self, phys_grads):
        """
        Convert physical gradients to model gradients.
        
        phys_grads: dict from mapper.from_kernel_cached() with keys
                    'total', 'g_ls', 'g_lsbar', 'm0', 'g0', scalars
        
        Returns: dict with same keys, containing only model-free gradients
                 (fixed params have zero gradient)
        """
        # Start with physical gradients
        model_grads = {}
        for ptype in ['total', 'g_ls', 'g_lsbar', 'm0', 'g0']:
            arr = np.zeros_like(phys_grads[ptype])
            for key, info in self._params.items():
                if info['type'] != ptype:
                    continue
                if 'fixed' in info:
                    continue  # gradient = 0
                mapped_to = self._model_params.get(key, key)
                if mapped_to != key:
                    # Equal constraint: accumulate gradient at target
                    target_info = self._params[mapped_to]
                    arr[target_info['idx']] += phys_grads[ptype][info['idx']]
                else:
                    arr[info['idx']] = phys_grads[ptype][info['idx']]
            model_grads[ptype] = arr

        # Scalars
        for name in ['delta_m', 'delta_g', 'g', 'ap', 'lam', 'phi', 'N']:
            model_grads[name] = phys_grads.get(name, 0)

        return model_grads

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def summary(self):
        """Print all parameters and their constraints."""
        print(f"Total parameters: {len(self._params)}")
        print(f"Model (fit) variables: {len(self._model_params)}")
        print(f"Fixed: {sum(1 for v in self._params.values() if 'fixed' in v)}")
        n_constrained = len(self._params) - len(self._model_params)
        print(f"Constrained (linear eq): {n_constrained}")
        print()
        for key, info in self._params.items():
            val = info['value']
            is_fixed = 'fixed' in info
            mapped = self._model_params.get(key, key)
            status = ' [FIXED]' if is_fixed else (' [→' + str(mapped) + ']' if mapped != key else '')
            if info['is_real']:
                print(f"  {key:40s} = {val:12.6f}{status}")
            else:
                print(f"  {key:40s} = {val.real:10.6f} + {val.imag:10.6f}j{status}")


# Shorthand
def create_manager(mapper):
    return ParamManager(mapper)
