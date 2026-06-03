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
        grads = mapper.from_kernel_cached(grad_ck, grad_m0_k, grad_g0_k, grad_scalar)
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
        self._res_m0_map = cfg.get('res_m0_map', {})
        self._res_g0_map = cfg.get('res_g0_map', {})

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
            gck = grad_ck[i]  # ∂J/∂ck_w*  (Wirtinger derivative from kernel)
            prod_full = self._cache_ck_vals_full[i]  # ck_w
            vals = self._cache_ck_vals[i]  # [total, g_ls_0, ...]
            for j, (typ, phys_idx) in enumerate(formula):
                f_j = vals[j]
                # ∂J/∂f_j* = ∂J/∂ck* * ∂ck*/∂f_j* = grad_ck * conj(∂ck/∂f_j)
                # ck = ∏_k f_k,  ∂ck/∂f_j = ∏_{k≠j} f_k = ck / f_j
                # ∂ck*/∂f_j* = conj(ck / f_j)
                # So: ∂J/∂f_j* = grad_ck * conj(ck / f_j)
                if abs(f_j) > 1e-15:
                    contrib = gck * np.conj(prod_full / f_j)
                else:
                    # f_j ≈ 0: compute product excluding f_j directly
                    contrib = gck * np.conj(np.prod([vals[k] for k in range(len(formula)) if k != j]))
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
        grad_total = np.zeros(len(self.totals), dtype=np.complex128)
        grad_gls = np.zeros(len(self.gls), dtype=np.complex128)
        grad_glsbar = np.zeros(len(self.glsbar), dtype=np.complex128)

        if not hasattr(self, '_cache_ck_vals'):
            return {...}  # will fill below

        for i, formula in enumerate(self.ck_formulas):
            gck = grad_ck[i]
            prod_full = self._cache_ck_vals_full[i]
            vals = self._cache_ck_vals[i]
            for j, (typ, phys_idx) in enumerate(formula):
                f_j = vals[j]
                # ∂J/∂f_j* = ∂J/∂ck* * conj(∂ck/∂f_j) = grad_ck * conj(ck / f_j)
                if abs(f_j) > 1e-15:
                    contrib = gck * np.conj(prod_full / f_j)
                else:
                    contrib = gck * np.conj(np.prod([vals[k] for k in range(len(formula)) if k != j]))
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

    # ------------------------------------------------------------------
    # Load parameters from a.json
    # ------------------------------------------------------------------

    def load_params(self, json_path, config_path=None):
        """Load fitted values from a.json into a params dict.

        Args:
            json_path: path to a.json
            config_path: path to config.yml (for m0/g0 resonance matching)

        Returns: params_dict for mapper.to_kernel() or ConstraintMapper
        """
        return _load_params_impl(json_path, self, config_path)

    def save_params(self, json_path, params_dict):
        """Save params dict to a.json format (mag*exp(i*phase) encoding).
        
        Uses resonance name → phys index mappings stored from build_kernel_config.
        
        Args:
            json_path: output path
            params_dict: physical params dict (total, g_ls, g_lsbar, m0, g0, scalars)
        """
        import json
        out = {}

        m = self

        # totals
        for tname, tidx in m.totals.items():
            z = params_dict['total'][tidx]
            out[f'{tname}_total_0r'] = abs(z)
            out[f'{tname}_total_0i'] = np.angle(z)

        # g_ls
        for (dname, ls), gidx in m.gls.items():
            z = params_dict['g_ls'][gidx]
            out[f'{dname}_g_ls_{ls}r'] = abs(z)
            out[f'{dname}_g_ls_{ls}i'] = np.angle(z)

        # g_lsbar
        for (dname, ls), gidx in m.glsbar.items():
            z = params_dict['g_lsbar'][gidx]
            out[f'{dname}_g_lsbar_{ls}r'] = abs(z)
            out[f'{dname}_g_lsbar_{ls}i'] = np.angle(z)

        # Scalars
        scalar_map = {'B_delta_m': 'delta_m', 'B_delta_gamma': 'delta_g',
                      'B_gamma': 'g', 'B_A_prod': 'ap',
                      'B_poqr': 'lam', 'B_poqi': 'phi'}
        for jname, sname in scalar_map.items():
            out[jname] = float(params_dict.get(sname, 0))

        # m0: build (mass_val, model) → phys index from params_dict order
        # The phys index is assigned by sorted (mass_val, model), matching load_params
        m0_keys = sorted({(v, k) for k, v in m._res_m0_map.items()},  # Wrong!
                         key=lambda x: x[0])  # Actually need grouping by (val, model)
        # Rebuild: for each resonance, get its (mass_val, model) from config
        # Since we don't have config here, use the stored mapping directly
        seen_m0 = set()
        for res_name, pidx in sorted(m._res_m0_map.items(), key=lambda x: x[1]):
            # pidx is the phys index assigned by parser
            if pidx < len(params_dict['m0']):
                out[f'{res_name}_mass'] = float(params_dict['m0'][pidx])

        # g0: use stored mapping, check if FlatteC (multiple g entries)
        for res_name, pidx_list in m._res_g0_map.items():
            is_flatte = len(pidx_list) > 1
            for i, pidx in enumerate(pidx_list):
                if pidx < len(params_dict['g0']):
                    val = float(params_dict['g0'][pidx])
                    if is_flatte:
                        out[f'{res_name}_g_{i}'] = val
                    elif i == 0:
                        out[f'{res_name}_width'] = val

        with open(json_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"  Saved {len(out)} params to {json_path}")
        return out


# Shorthand
def create_mapper(cfg):
    return ParamMapper(cfg)


# ====================================================================
# Load parameters from a.json (fitted values)
# ====================================================================

def load_params(json_path, mapper, config_path=None):
    """Standalone: load fitted values from a.json (calls mapper.load_params)."""
    return mapper.load_params(json_path, config_path)


def _load_params_impl(json_path, mapper, config_path=None):
    """
    Load fitted parameter values from a tf_pwa a.json into the params_dict
    format expected by ParamMapper.compute().

    a.json uses naming: {full_chain}_{param_type}_{idx}{r|i}
    with complex stored as mag * exp(i * phase) where r=mag, i=phase.

    Args:
        json_path: path to a.json
        mapper: ParamMapper instance
        config_path: path to config.yml (for m0/g0 resonance matching)

    Returns: dict with keys 'total', 'g_ls', 'g_lsbar', 'm0', 'g0',
             and scalars 'delta_m', 'delta_g', 'g', 'ap', 'lam', 'phi'
    """
    import json
    import yaml
    with open(json_path) as f:
        data = json.load(f)

    # Helper: get complex value from r/i pair
    # JSON stores complex as: mag * exp(i * phase), where r=mag, i=phase
    def get_complex(key):
        mag = data.get(key + 'r', 0)
        phase = data.get(key + 'i', 0)
        return mag * np.exp(1j * phase)

    # --- total couplings ---
    # JSON: {B_decay_chain}_total_0{r|i} where B_decay_chain starts with
    # 'B->{res1}.{B_level_daughter}' where res1 is the cascade resonance.
    # The mapper uses 'B->{res1}.{res2}' with the sub-resonance.
    # We match by the first resonance (res1) which is always the cascade
    # resonance in the JSON key.
    total_arr = np.zeros(len(mapper.totals), dtype=np.complex128)
    for tname, tidx in mapper.totals.items():
        # Get first resonance name (before first '.')
        dot_pos = tname.find('.')
        prefix = tname if dot_pos < 0 else tname[:dot_pos]
        for jkey in data:
            if jkey.startswith(prefix) and '_total_0' in jkey:
                base_key = jkey[:-1]  # strip r or i
                total_arr[tidx] = get_complex(base_key)
                break

    # --- g_ls ---
    # JSON patterns:
    #   B-level:   'B->{d1}.{d2}_g_ls_{idx}{r|i}'  → dname='B'
    #   Sub-decay: '{parent}->{d1}.{d2}_g_ls_{idx}{r|i}' → dname='{parent}'
    # We match by checking the decay parent name (before '->').
    gls_arr = np.zeros(len(mapper.gls), dtype=np.complex128)
    for (dname, ls_idx), gidx in mapper.gls.items():
        pat = f'_g_ls_{ls_idx}'
        for jkey in data:
            if '_g_lsbar_' in jkey:
                continue
            if pat not in jkey:
                continue
            # Extract the decay parent from JSON key
            prefix = jkey.split(pat)[0]
            # prefix is like 'B->a2(1320)p.pim2' or 'a2(1320)p->rhoA.pip2'
            # The decay parent is the part before '->' (or the whole string if no '->')
            parent = prefix.split('->')[0] if '->' in prefix else prefix
            if parent == dname:
                gls_arr[gidx] = get_complex(f'{prefix}_g_ls_{ls_idx}')
                break

    # --- g_lsbar ---
    glsbar_arr = np.zeros(len(mapper.glsbar), dtype=np.complex128)
    for (dname, ls_idx), gidx in mapper.glsbar.items():
        pat = f'_g_lsbar_{ls_idx}'
        for jkey in data:
            if pat not in jkey:
                continue
            prefix = jkey.split(pat)[0]
            parent = prefix.split('->')[0] if '->' in prefix else prefix
            if parent == dname:
                glsbar_arr[gidx] = get_complex(f'{prefix}_g_lsbar_{ls_idx}')
                break

    # --- m0/g0 from config YAML particle section ---
    m0_arr = np.zeros(mapper.n_m0_phys, dtype=np.float64)
    g0_arr = np.zeros(mapper.n_g0_phys, dtype=np.float64)
    if config_path:
        with open(config_path) as f:
            ycfg = yaml.safe_load(f)
        particle = ycfg.get('particle', {})
        decay = ycfg.get('decay', {})

        def get_resonance_names(name, visited=None):
            """Yield actual resonance names from an intermediate name.
            Follows both particle multiplets and decay definitions."""
            if visited is None: visited = set()
            if name in visited: return
            visited.add(name)
            # Check particle section
            props = particle.get(name)
            if props is None:
                return
            if isinstance(props, list):
                # Multiplet: each entry is a resonance name
                for p in props:
                    if isinstance(p, str):
                        sub = particle.get(p, {})
                        if isinstance(sub, dict) and 'J' in sub:
                            yield p
                        else:
                            yield from get_resonance_names(p, visited)
            elif isinstance(props, dict) and 'J' in props:
                yield name
                return
            # Also follow decay definitions (intermediates like pipid)
            decay_daughters = decay.get(name, [])
            for item in decay_daughters:
                if isinstance(item, str) and item not in finals:
                    yield from get_resonance_names(item, visited)

        used_resonances = set()
        finals = set(ycfg.get('particle', {}).get('$finals', []))
        b_decay_lines = decay.get(ycfg.get('particle', {}).get('$top', 'B'), [])
        for decay_line in b_decay_lines:
            if isinstance(decay_line, list):
                for item in decay_line:
                    if isinstance(item, str) and item not in finals:
                        for rname in get_resonance_names(item):
                            if rname not in finals:
                                used_resonances.add(rname)

        # --- m0 ---
        m0_keys = []
        for rname in sorted(used_resonances):
            if rname in finals:
                continue
            props = particle.get(rname, {})
            if isinstance(props, dict) and 'J' in props:
                m0_keys.append((props.get('mass', 0), props.get('model', 'BW'), rname))

        seen = set()
        m0_sorted = []
        for m_val, model, rname in sorted(m0_keys, key=lambda x: (x[0], x[1])):
            key = (m_val, model)
            if key not in seen:
                seen.add(key)
                m0_sorted.append((m_val, model, rname))

        for i, (m_val, model, rname) in enumerate(m0_sorted):
            if i >= mapper.n_m0_phys:
                break
            jkey = f'{rname}_mass'
            m0_arr[i] = data.get(jkey, m_val)

        # --- g0 ---
        width_keys = []
        flatte_items = []
        for rname in sorted(used_resonances):
            if rname in finals:
                continue
            props = particle.get(rname, {})
            if isinstance(props, dict) and 'J' in props:
                model = props.get('model', 'BW')
                if model == 'FlatteC':
                    for k, v in props.items():
                        if k.startswith('g_') and v:
                            flatte_items.append((k, v, rname))
                elif props.get('width', 0) > 0:
                    width_keys.append((props['width'], model, rname))

        seen = set()
        w_sorted = []
        for w_val, model, rname in sorted(width_keys, key=lambda x: (x[0], x[1])):
            key = (w_val, model)
            if key not in seen:
                seen.add(key)
                w_sorted.append((w_val, model, rname))

        for i, (w_val, model, rname) in enumerate(w_sorted):
            if i >= mapper.n_g0_phys:
                break
            jkey = f'{rname}_width'
            # For non-Flatte: gamma_table already contains full physical width
            # g0 = 1.0 (no additional scaling). FlatteC handles separately below.
            g0_arr[i] = 1.0

        for i, (k, v, rname) in enumerate(sorted(flatte_items)):
            idx = len(w_sorted) + i
            if idx < mapper.n_g0_phys:
                jkey = f'{rname}_{k}'
                # FlatteC: gamma_table stores i*q/m (kinematic only), g0 = coupling
                g0_arr[idx] = data.get(jkey, v)
    else:
        for i, val in enumerate(sorted(set(v for k, v in data.items() if k.endswith('_mass')))):
            if i < mapper.n_m0_phys:
                m0_arr[i] = val
        for i, val in enumerate(sorted(set(v for k, v in data.items() if k.endswith('_width')))):
            if i < mapper.n_g0_phys:
                g0_arr[i] = val

    # --- scalars ---
    # JSON: B_delta_m → delta_m, B_delta_gamma → delta_g, B_gamma → g
    # B_poqr → lam, B_poqi → phi (imaginary), B_A_prod → ap
    scalar_map = {
        'B_delta_m': 'delta_m', 'B_delta_gamma': 'delta_g',
        'B_gamma': 'g', 'B_poqr': 'lam', 'B_poqi': 'phi',
        'B_A_prod': 'ap',
    }
    scalars = {}
    for jkey, sname in scalar_map.items():
        scalars[sname] = data.get(jkey, 0)

    params = {
        'total': total_arr, 'g_ls': gls_arr, 'g_lsbar': glsbar_arr,
        'm0': m0_arr, 'g0': g0_arr,
        **scalars,
    }

    # Print loaded stats
    n_total = sum(1 for v in total_arr if abs(v) > 0)
    n_gls = sum(1 for v in gls_arr if abs(v) > 0)
    n_glsbar = sum(1 for v in glsbar_arr if abs(v) > 0)
    n_m0 = sum(1 for v in m0_arr if v > 0)
    n_g0 = sum(1 for v in g0_arr if v > 0)
    print(f"  Loaded params: {n_total}/{len(total_arr)} total, "
          f"{n_gls}/{len(gls_arr)} g_ls, {n_glsbar}/{len(glsbar_arr)} g_lsbar, "
          f"{n_m0}/{mapper.n_m0_phys} m0, {n_g0}/{mapper.n_g0_phys} g0")

    return params


def load_params_file(json_path, cfg_or_mapper):
    """Load params from a.json, creating a mapper if needed."""
    if isinstance(cfg_or_mapper, ParamMapper):
        mapper = cfg_or_mapper
    else:
        mapper = ParamMapper(cfg_or_mapper)
    return load_params(json_path, mapper)
