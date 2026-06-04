#!/usr/bin/env python3
"""
Full PWA fit: load converted data, build tables, fit with BFGS, save results.

Handles parameter constraints, boundary transforms for time parameters,
and multiple random starts (following pw_cfit5_td6_fix29.py conventions).

Usage:
    python run_fit.py config_angle.yml --out fit_results
"""

import os, sys, json, time, argparse, cmath
import numpy as np


class Trans:
    """Boundary transform: y = k*sin(x/k) + bias, maps ℝ → [a,b]."""
    def __init__(self, a, b):
        self.a = min(a, b)
        self.b = max(a, b)
        self.k = (self.b - self.a) / 2
        self.bias = (self.b + self.a) / 2
    def __call__(self, x):
        return self.k * np.sin(x / max(self.k, 1e-15)) + self.bias
    def grad(self, x):
        return np.cos(x / max(self.k, 1e-15))
    def inv(self, y):
        y = (y - self.a) % (self.b - self.a) + self.a
        return np.arcsin(((y - self.bias) / max(self.k, 1e-15) + 1) % 2 - 1) * self.k
    def trans_err(self, x, e):
        return np.abs(self.grad(x)) * e


def run_fit(config_path, out_dir='fit_results', method='BFGS', maxiter=200,
            fix_2pi=True, gamma_table_path=None, n_start=5):
    """Run full PWA fit and save results."""
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pwa_gpu.fitter import PWAFitter
    
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. Initialize fitter ----
    print("Initializing fitter...", flush=True)
    t0 = time.time()
    fitter = PWAFitter(config_path)
    
    # ---- 2. Build or load tables ----
    if gamma_table_path and os.path.exists(gamma_table_path):
        print(f"Loading tables from {gamma_table_path}...", flush=True)
        fitter.load_tables(gamma_table_path)
    else:
        print("Building tables...", flush=True)
        fitter.build_tables(gamma_table_path)
    print(f"  Tables ready ({time.time()-t0:.1f}s)", flush=True)

    # ---- 3. Load parameters ----
    config_dir = os.path.dirname(os.path.abspath(config_path))
    a_json = os.path.join(config_dir, 'a.json')
    if os.path.exists(a_json):
        print(f"Loading params from {a_json}...", flush=True)
        fitter.load_params(a_json)
    else:
        print("No a.json found, using random params", flush=True)
    fitter.summary()

    # ---- 4. Set constraints (following pw_cfit5_td6_fix29.py conventions) ----
    if fix_2pi:
        print("Fixing 2π resonances...", flush=True)
        fitter.fix_2pi_resonances()

    # ---- Reference constraints (pw_cfit5_td6_fix29.py) ----
    mapper = fitter.mapper
    cst = fitter.cst

    # Snapshot all parameter keys (including those about to be fixed)
    all_known_keys = set(cst.get_free_keys())

    # Step 1: Fix g_ls_0 that the reference keeps fixed.
    # Ref initially fixes ALL g_ls_0 (line 40), then cascade loop (lines 63-90)
    # selectively unfixes some. Here we fix only the ones that STAY fixed.
    all_keys = all_known_keys.copy()

    # 1a: B-level g_ls_0 — ALL fixed
    for k in all_keys:
        if k.startswith('B->') and k.endswith('_g_ls_0') and '_g_lsbar_' not in k:
            cst.set_fixed(k, 1.0 + 0.0j)

    # 1b: VV sub-decay g_ls_0 (rhoA, rhoB, f0(500), f0(980), NR0 → finals)
    for k in all_keys:
        vv_daughters = ['rhoA->', 'rhoB->', 'f0(500)->', 'f0(980)->', 'NR0->',
                        'f2(1270)->']  # f2 is cascade sub-sub-decay
        if any(k.startswith(d) for d in vv_daughters):
            if k.endswith('_g_ls_0'):
                cst.set_fixed(k, 1.0 + 0.0j)

    # 1c: First cascade coupling g_ls per resonance (rhoA LS=0, m+p versions)
    cascade_resonances = [
        "a1(1260)", "a1(1640)", "a2(1320)", "pi1300", "pi1600",
        "pi2(1670)", "pi1(1600)"]
    for r1 in cascade_resonances:
        for key in [f"{r1}m->rhoA.pim2_g_ls_0", f"{r1}p->rhoA.pip2_g_ls_0"]:
            if key in all_keys:
                cst.set_fixed(key, 1.0 + 0.0j)

    # 1d: Pole.0 and prod.0 (ref line 42-43)
    for k in all_keys:
        if k.endswith('_pole_0') or k.endswith('_prod_0'):
            cst.set_fixed(k, 1.0 + 0.0j)

    # Step 2: Fix VV rhoA.rhoB total (ref line 51-53)
    fix_total = "B->rhoA.rhoBrhoA->pip1.pim1rhoB->pip2.pim2_total_0"
    if fix_total in cst.get_free_keys():
        cst.set_fixed(fix_total, 1.0 + 0.0j)
        print(f"  Fixed {fix_total} = 1+0j", flush=True)

    # Step 3: Cascade constraints (ref lines 63-90)
    # For each resonance, first g_ls stays fixed, rest get same_params mirror.
    # rhoA modes get scale_factor -1 (CP sign) → set_linear(p, [(m, -1.0)])
    # Non-rhoA modes get p = m → set_equal(p, m)
    sub_modes = ["rhoA", "f0(500)", "f0(980)", "f2(1270)"]
    # Swap resonance charge AND daughter final state (pip2↔pim2 only;
    # pip1/pim1 are always the identical π⁺/π⁻ and should NOT be swapped)
    swp = lambda k, src, dst: (
        k.replace(src, dst)
         .replace('pip2', 'P2').replace('pim2', 'pip2').replace('P2', 'pim2'))

    for r1 in cascade_resonances:
        fixed_first = True  # ref line 66
        p_totals, m_totals = [], []
        for r2 in sub_modes:
            # Total names
            name_p = f"B->{r1}p.pim2{r1}p->{r2}.pip2{r2}->pip1.pim1_total_0"
            name_m = swp(name_p, f'{r1}p', f'{r1}m')
            for k in [name_p, name_m]:
                if k in cst.get_free_keys():
                    (p_totals if 'p.pim2' in k else m_totals).append(k)
            # Sub-decay g_ls: ref checks key_m existence in all_params, not free-ness
            for idx in range(3):
                key_m = f"{r1}m->{r2}.pim2_g_ls_{idx}"
                key_p = f"{r1}p->{r2}.pip2_g_ls_{idx}"
                if key_m not in all_known_keys and key_p not in all_known_keys:
                    continue  # not a parameter at all
                if fixed_first:
                    fixed_first = False  # first g_ls per resonance stays fixed
                else:
                    if key_m in cst.get_free_keys() and key_p in cst.get_free_keys():
                        if r2 == "rhoA":
                            # CP-odd: p = -1 × m (scale_params[m] = -1 in ref)
                            cst.set_linear(key_p, [(key_m, -1.0)])
                        else:
                            cst.set_equal(key_p, key_m)
        # Equate totals across sub-decay modes (ref lines 89-90):
        # only rhoA version stays free for each charge
        for totals in [p_totals, m_totals]:
            if len(totals) > 1:
                for t in totals[1:]:
                    cst.set_equal(t, totals[0])

    # Step 4: KMA = KMB (ref lines 101-104)
    for kk in list(cst.get_free_keys()):
        for j in ['pole', 'prod']:
            for i in range(3):
                suf = f"{j}_{i}"
                if f'KMA_{suf}' in kk:
                    kmb = kk.replace(f'KMA_{suf}', f'KMB_{suf}')
                    if kmb in cst.get_free_keys():
                        cst.set_equal(kmb, kk)

    # Step 5: Fix K-matrix high indices (ref lines 105-110)
    for fix_key in list(cst.get_free_keys()):
        for j in ['pole', 'prod']:
            for i in range(3, 5):
                for prefix in ['KMA_', 'KMB_', 'KMC_', 'KM2_']:
                    if f"{prefix}{j}_{i}" in fix_key:
                        cst.set_fixed(fix_key, 0.0 + 0.0j)
                        break

    # Step 6: Fix scalar parameters the reference keeps fixed.
    # The reference (pw_cfit5_td6_fix29.py lines 19-28) has:
    #   fix_time_params = ["A_prod", "delta_gamma", "delta_m", "poqr", "poqi"]
    #   free_time_params = ["gamma"]
    # Fix to their LOADED values by resolving from the current model dict.
    # set_fixed(k) without value stores True → unpack returns 0 — wrong!
    model_all = cst.build_model_dict(fitter.params)
    for k in list(cst.get_free_keys()):
        if not cst._is_complex_key(k) and k != 'B_gamma':
            val = cst._resolve(k, model_all)
            cst.set_fixed(k, val)

    # Print free parameters for user reference (a.json format)
    cst = fitter.cst
    free = cst.get_free_keys()
    print(f"  Free parameters: {len(free)}", flush=True)

    def _categorize(key):
        if key.endswith('_total_0'):
            return 'total'
        if '_g_lsbar_' in key:
            return 'g_lsbar'
        if '_g_ls_' in key:
            return 'g_ls'
        if key.endswith('_mass'):
            return 'm0'
        if key.endswith('_width') or '_g_' in key:
            return 'g0'
        return 'scalar'

    for cat in ['total', 'g_ls', 'g_lsbar', 'm0', 'g0']:
        cat_keys = [k for k in free if _categorize(k) == cat]
        if cat_keys:
            print(f"    {cat} ({len(cat_keys)}):")
            for k in cat_keys[:8]:
                print(f"      {k}")
            if len(cat_keys) > 8:
                print(f"      ... ({len(cat_keys)-8} more)")
    scalar_keys = [k for k in free if _categorize(k) == 'scalar']
    if scalar_keys:
        print(f"    scalars: {scalar_keys}", flush=True)

    # ---- 5. Load data ----
    data_dir = os.path.dirname(os.path.abspath(config_path))
    # Try to find converted data
    for dpath in [os.path.join(config_dir, '..', 'converted_data'),
                  os.path.join(os.path.dirname(config_dir), 'converted_data'),
                  os.path.join(os.path.dirname(os.path.dirname(config_dir)), 'converted_data')]:
        data_npz = os.path.join(dpath, 'data_arrays.npz')
        phsp_npz = os.path.join(dpath, 'phsp_arrays.npz')
        if os.path.exists(data_npz) and os.path.exists(phsp_npz):
            break
    else:
        data_npz = 'converted_data/data_arrays.npz'
        phsp_npz = 'converted_data/phsp_arrays.npz'
    
    print(f"Loading data from {data_npz}...", flush=True)
    d = np.load(data_npz)
    p = np.load(phsp_npz)
    
    data = fitter.load_data(
        mass=d['mass'], q=d['q'], angles=d['angles'],
        time_arr=d['time'], frac=d['frac'],
        weights=np.ones(len(d['mass'])), bkg=d['bkg'])
    phsp = fitter.load_phsp(
        mass=p['mass'], q=p['q'], angles=p['angles'],
        time_arr=p['time'], frac=p['frac'],
        phsp_weights=p.get('weight', np.ones(len(p['mass']))))
    
    print(f"  Signal: {d['mass'].shape[0]} events")
    print(f"  Phsp:   {p['mass'].shape[0]} events")

    # ---- 6. Boundary transforms ----
    # Load config for nominal mass/width values
    import yaml
    with open(config_path) as f:
        ycfg = yaml.safe_load(f)
    particle_sec = ycfg.get('particle', {})

    # Bounds for scalars (following pw_cfit5_td6_fix29.py)
    phys_bounds = {
        'B_delta_m': [0.3, 0.8],
        'B_delta_gamma': [-0.3, 0.3],
        'B_gamma': [-0.3, 0.3],
        'B_poqi': [-np.pi, np.pi],
        'B_A_prod': [-0.5, 0.5],
        'B_poqr': [0.0, 2.0],
    }
    # Mass bounds: ±50% around nominal value;  width bounds: +200%/-90% (positive, not zero)
    for rname, props in particle_sec.items():
        if isinstance(props, dict) and 'J' in props:
            nom_mass = props.get('mass', 0)
            if nom_mass > 0:
                phys_bounds[f'{rname}_mass'] = [nom_mass * 0.5, nom_mass * 1.5]
            nom_width = props.get('width', 0)
            if nom_width > 0:
                phys_bounds[f'{rname}_width'] = [max(nom_width * 0.1, 0.001), nom_width * 3.0]
            # FlatteC couplings
            for k, v in props.items():
                if k.startswith('g_') and isinstance(v, (int, float)) and v > 0:
                    phys_bounds[f'{rname}_{k}'] = [v * 0.1, v * 3.0]

    free_keys = fitter.get_free_keys()
    bound_trans = {}
    for ki, key in enumerate(free_keys):
        if key in phys_bounds:
            a, b = phys_bounds[key]
            bound_trans[ki] = Trans(a, b)

    # ---- 7. Build fit function with boundary transforms ----
    f_fit = fitter.make_fit_func(data, phsp)
    x0 = fitter.get_free_values()
    n_free = len(x0)

    # Print initial parameter values
    print(f"\nInitial parameters ({n_free} components across {len(free_keys)} keys):", flush=True)
    idx = 0
    for ki, key in enumerate(free_keys):
        if cst._is_complex_key(key):
            mag = x0[idx]
            phase = x0[idx + 1]
            print(f"  [{idx:3d}] {key:55s}  mag={mag:.6f}", flush=True)
            print(f"  [{idx+1:3d}] {'':55s}  phase={phase:.6f}", flush=True)
            idx += 2
        else:
            print(f"  [{idx:3d}] {key:55s}  = {x0[idx]:.6f}", flush=True)
            idx += 1

    def nll_wrapped(x):
        """Apply boundary transforms before calling the GPU fit function."""
        new_x = x.copy()
        for ki, tr in bound_trans.items():
            new_x[ki] = tr(x[ki])
        nll, g = f_fit(new_x)
        new_g = g.copy()
        for ki, tr in bound_trans.items():
            new_g[ki] = tr.grad(x[ki]) * g[ki]
        return nll, new_g

    # ---- 8. Multiple random starts ----
    import scipy.optimize
    best_result = None
    best_nll = 1e30
    fit_time = 0.0

    for start_idx in range(n_start):
        # Random starting point: phases uniform in [0,2π), magnitudes random
        if start_idx == 0 and os.path.exists(a_json):
            x_start = x0.copy()  # start from loaded params
        else:
            x_start = np.random.uniform(-2, 2, n_free)
            # Set bounded params to reasonable initial values
            for ki, tr in bound_trans.items():
                x_start[ki] = tr.inv(np.random.uniform(tr.a + 0.1*(tr.b-tr.a),
                                                        tr.b - 0.1*(tr.b-tr.a)))

        t0 = time.time()
        try:
            result = scipy.optimize.minimize(
                nll_wrapped, x_start, jac=True, method=method,
                options={'disp': False, 'maxiter': maxiter})
        except Exception as e:
            print(f"  Start {start_idx} failed: {e}", flush=True)
            continue

        fit_time = time.time() - t0
        nll_val = result.fun

        print(f"  Start {start_idx}: NLL={nll_val:.4f}, "
              f"iter={result.nit}, time={fit_time:.1f}s, "
              f"success={result.success}", flush=True)

        if nll_val < best_nll:
            best_nll = nll_val
            best_result = result

    if best_result is None:
        print("All fits failed!", flush=True)
        return

    result = best_result
    # fit_time was set in the loop for the best result

    # ---- 9. Build reference-format JSON output ----
    # Reference (pw_cfit5_td6_fix29.py lines 809-850):
    #   {"value": {param_name: float, ...},
    #    "error": {param_name: float, ...},
    #    "status": {NLL, Ndf, jac, success, message, rhorho}}
    # Complex params stored as {name}r = magnitude, {name}i = phase (POLAR).
    # Real scalars stored as plain name.
    # Fixed, aliased (new_name), and scaled params all written to value dict.
    x_phys = result.x.copy()
    for ki, tr in bound_trans.items():
        x_phys[ki] = tr(result.x[ki])

    # ── Build model dict from fitted result ──
    model_opt = cst.unpack(x_phys, free_keys)
    phys = cst.get_physical(model_opt)
    fitter._params = phys
    # Full model dict including fixed/aliased/scaled params
    model_all = cst.build_model_dict(phys)

    # ── Collect constraint metadata (mirrors ref's fixed_params/new_name/scale_params) ──
    # We need: which params are fixed, which are aliased, which have linear scaling.
    # These are stored in cst._constraints.
    ref_fixed = {}        # {name: complex_value}  — fixed params
    ref_new_name = {}     # {alias_name: source_name} — aliased params
    ref_scale_params = {} # {name: scale_factor} — linear scale on r-component

    for key, cinfo in cst._constraints.items():
        if 'fixed' in cinfo:
            val = cinfo['fixed']
            if isinstance(val, bool):
                # True = fixed to current model value
                ref_fixed[key] = model_all.get(key, 0)
            elif isinstance(val, (int, float)):
                # Real scalar fixed to a value
                ref_fixed[key] = val
            else:
                ref_fixed[key] = val
        elif 'equal_to' in cinfo:
            ref_new_name[key] = cinfo['equal_to']
        elif 'linear' in cinfo:
            src, coeff = cinfo['linear'][0]
            ref_scale_params[key] = coeff
            ref_new_name[key] = src

    # ── Build params_order: flat index → param name (mirroring ref) ──
    # Free complex params first (2 indices each: {name}r, {name}i),
    # then free real scalars.
    free_keys = cst.get_free_keys()
    free_complex = [k for k in free_keys if cst._is_complex_key(k)]
    free_real    = [k for k in free_keys if not cst._is_complex_key(k)]
    params_order = {}
    for i, k in enumerate(free_complex):
        params_order[2*i]     = f"{k}r"
        params_order[2*i+1]   = f"{k}i"
    offset = len(free_complex) * 2
    for i, k in enumerate(free_real):
        params_order[offset + i] = k

    # ── Build value dict ──
    value = {}
    # 1. Free complex: mag/phase from x_phys (in polar)
    #    Reference: x is [mag1, phase1, mag2, phase2, ...] in polar
    #    Our x_phys is also [mag1, phase1, ...] from free_flat_from_phys → unpack
    #    Actually our unpack returns model dict with complex values.
    #    We need to map free_keys → polar pairs.
    for i, k in enumerate(free_complex):
        val = model_all.get(k, 0+0j)
        value[f"{k}r"] = abs(val)
        value[f"{k}i"] = cmath.phase(val)

    # 2. Free real:
    for k in free_real:
        value[k] = float(model_all.get(k, 0))

    # 3. Bound-transformed values (reference applies forward transform to value)
    for ki, tr in bound_trans.items():
        name = params_order[ki]
        # Only apply if name ends with r,i (complex) — transform raw value
        if name in value:
            value[name] = tr(x_phys[ki] if 'i' not in name else 0)
        else:
            # Real scalar with bound
            if name in params_order.values():
                idx_in_flat = [ki2 for ki2, n2 in params_order.items() if n2 == name]
                if idx_in_flat:
                    raw = result.x[idx_in_flat[0]]
                    value[name] = tr(raw)

    # 4. Fixed params: write as mag/phase
    for j, jv in ref_fixed.items():
        value[f"{j}r"] = abs(complex(jv))
        value[f"{j}i"] = cmath.phase(complex(jv))

    # 5. Aliased params: copy from source (reference new_name)
    for j, jv in ref_new_name.items():
        value[f"{j}r"] = value.get(f"{jv}r", 0)
        value[f"{j}i"] = value.get(f"{jv}i", 0)

    # 6. Scale params: multiply r-component by scale factor
    for j, jv in ref_scale_params.items():
        value[f"{j}r"] = jv * value.get(f"{j}r", 0)

    # 7. Time params (reference fix_time_params + free gamma)
    #    Reference names: gamma, A_prod, delta_gamma, delta_m, poqr, poqi
    #    Our names: B_gamma, B_A_prod, B_delta_gamma, B_delta_m, B_poqr, B_poqi
    time_name_map = {'B_gamma': 'gamma', 'B_A_prod': 'A_prod',
                     'B_delta_gamma': 'delta_gamma', 'B_delta_m': 'delta_m',
                     'B_poqr': 'poqr', 'B_poqi': 'poqi'}
    for our_key, ref_key in time_name_map.items():
        val = model_all.get(our_key, 0)
        value[ref_key] = float(val)

    # ── Build error dict from Hessian ──
    errors = {}
    if hasattr(result, 'hess_inv'):
        try:
            h = result.hess_inv
            if hasattr(h, 'todense'):
                h = h.todense()
            h = np.asarray(h)
            # Propagate boundary transforms
            g_vec = np.ones(n_free)
            for ki, tr in bound_trans.items():
                g_vec[ki] = tr.grad(result.x[ki])
            hess_phys = g_vec[:, None] * h * g_vec[None, :]
            for i, k in enumerate(free_complex):
                errors[f"{k}r"] = float(np.sqrt(max(hess_phys[2*i, 2*i], 0)))
                errors[f"{k}i"] = float(np.sqrt(max(hess_phys[2*i+1, 2*i+1], 0)))
            for i, k in enumerate(free_real):
                idx = offset + i
                err = float(np.sqrt(max(hess_phys[idx, idx], 0)))
                # Propagate through bound transform if applicable
                for ki, tr in bound_trans.items():
                    if params_order.get(ki) == k:
                        x_val = value.get(k, 0)
                        err = tr.trans_err(tr.inv(x_val), err)
                errors[k] = err
        except Exception as e:
            print(f"  Hessian error: {e}", flush=True)

    # ── Build status dict ──
    corr_name = [[i, n] for i, n in params_order.items()
                 if 'B->rhoA.rhoB' in n]
    corr_idx = np.array([i[0] for i in corr_name], dtype=int)
    corr_order = [i[1] for i in corr_name]
    if hasattr(result, 'hess_inv'):
        try:
            h = result.hess_inv
            if hasattr(h, 'todense'):
                h = h.todense()
            h = np.asarray(h)
            corr_mat = h[corr_idx][:, corr_idx]
        except Exception:
            corr_mat = None
    else:
        corr_mat = None

    final_params = {
        'value': value,
        'error': errors,
        'status': {
            'NLL': float(result.fun),
            'Ndf': len(result.x),
            'jac': list(result.jac) if result.jac is not None else [],
            'success': bool(result.success),
            'message': str(result.message),
            'rhorho': [corr_order, corr_mat.tolist() if corr_mat is not None else []],
        }
    }

    # ── Save reference-format JSON ──
    params_path = os.path.join(out_dir, 'final_params.json')
    with open(params_path, 'w') as f:
        json.dump(final_params, f, indent=2)
    print(f"  Reference-format JSON: {params_path}", flush=True)

    # Also save .npz for convenience
    result_npz = os.path.join(out_dir, 'fit_results.npz')
    phys_arrs = {}
    for k, v in phys.items():
        if isinstance(v, np.ndarray):
            phys_arrs[k] = v
        elif isinstance(v, complex):
            phys_arrs[k + '_re'] = np.array([v.real])
            phys_arrs[k + '_im'] = np.array([v.imag])
    np.savez(result_npz, **phys_arrs)
    print(f"  Results NPZ: {result_npz}", flush=True)

    print(f"\n{'='*50}", flush=True)
    print(f"Fit {'SUCCEEDED' if result.success else 'FAILED'}", flush=True)
    print(f"  NLL:       {result.fun:.4f}")
    print(f"  Iterations: {result.nit}")
    print(f"  Time:       {fit_time:.1f}s")
    print(f"  Status:     {result.status}")
    print(f"{'='*50}", flush=True)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PWA fit')
    parser.add_argument('config', help='config_angle.yml path')
    parser.add_argument('--out', default='fit_results', help='output directory')
    parser.add_argument('--method', default='BFGS')
    parser.add_argument('--maxiter', type=int, default=200)
    parser.add_argument('--no-fix-2pi', action='store_true')
    parser.add_argument('--tables', default=None)
    parser.add_argument('--n-start', type=int, default=5,
                        help='number of random starts')

    args = parser.parse_args()
    run_fit(config_path=args.config, out_dir=args.out, method=args.method,
            maxiter=args.maxiter, fix_2pi=not args.no_fix_2pi,
            gamma_table_path=args.tables, n_start=args.n_start)
