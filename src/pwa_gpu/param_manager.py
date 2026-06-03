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
    Manages the mapping between physical parameters and model (fit) parameters.
    
    Parameter naming:
        total/{name}       — complex production coupling
        g_ls/{name}/{idx}  — complex B helicity coupling
        g_lsbar/{name}/{idx} — complex Bbar helicity coupling
        m0/{idx}           — real mass
        g0/{idx}           — real width/scale
        {scalar_name}      — real scalar
    
    Example:
        mgr = ParamManager(mapper)
        mgr.fix('m0/0')                  # fix first mass
        mgr.fix('delta_m', 0.5)          # fix delta_m to 0.5
        mgr.equal(['m0/0', 'm0/1'])      # share two masses
    """

    def __init__(self, mapper):
        self.mapper = mapper
        # Build parameter registry
        self._params = OrderedDict()  # name → {type, idx, value, is_real}
        self._model_params = OrderedDict()  # flat fit variables
        self._constraints = []  # list of constraint functions

        # Register all physical parameters
        for name, idx in mapper.totals.items():
            key = f'total/{name}'
            self._params[key] = {'type': 'total', 'idx': idx, 'is_real': False, 'value': 0j}

        for (dname, ls), idx in mapper.gls.items():
            key = f'g_ls/{dname}/{ls}'
            self._params[key] = {'type': 'g_ls', 'idx': idx, 'is_real': False, 'value': 0j}

        for (dname, ls), idx in mapper.glsbar.items():
            key = f'g_lsbar/{dname}/{ls}'
            self._params[key] = {'type': 'g_lsbar', 'idx': idx, 'is_real': False, 'value': 0j}

        for idx in range(mapper.n_m0_phys):
            self._params[f'm0/{idx}'] = {'type': 'm0', 'idx': idx, 'is_real': True, 'value': 0.0}

        for idx in range(mapper.n_g0_phys):
            self._params[f'g0/{idx}'] = {'type': 'g0', 'idx': idx, 'is_real': True, 'value': 0.0}

        for name in ['delta_m', 'delta_g', 'g', 'ap', 'lam', 'phi']:
            self._params[name] = {'type': name, 'idx': 0, 'is_real': True, 'value': 0.0}

        # By default: each physical param is its own model param
        self._reset()

    def _reset(self):
        """Reset to identity mapping (no constraints)."""
        self._model_params = OrderedDict()
        self._constraints = []
        for key in self._params:
            self._model_params[key] = key  # identity
        self._build()

    def _build(self):
        """Build the forward/backward mapping functions."""
        pass  # done on-the-fly in to_physical/to_model_grad

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def fix(self, *args):
        """Fix a parameter to a value. Usage:
            mgr.fix('m0/0')           # fix to current value
            mgr.fix('m0/0', 0.769)    # fix to 0.769
            mgr.fix('delta_m', 0.5)   # fix delta_m to 0.5
        """
        key = args[0]
        if key not in self._params:
            raise KeyError(f"Unknown parameter: {key}")
        if len(args) > 1:
            val = args[1]
            self._params[key]['value'] = val
            self._params[key]['fixed'] = val
        else:
            self._params[key]['fixed'] = True
        # Remove from model params (it's not a fit variable)
        if key in self._model_params:
            del self._model_params[key]
        return self

    def unfix(self, key):
        """Unfix a parameter (make it a fit variable again)."""
        self._params[key].pop('fixed', None)
        if key not in self._model_params:
            self._model_params[key] = key
        return self

    def equal(self, keys):
        """
        Make multiple parameters equal (share one fit variable).
        Usage: mgr.equal(['m0/0', 'm0/1'])
        """
        # Map all to the first key
        target = keys[0]
        for key in keys:
            if key in self._params:
                self._model_params[key] = target
        return self

    def linear(self, target_key, sources):
        """
        Linear combination: target = sum(coeff_i * source_i)
        
        Usage: mgr.linear('m0/5', [('m0/0', 0.5), ('m0/1', 0.5)])
        """
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
