#!/usr/bin/env python3
"""
Full PWA fit: load converted data, build tables, fit with BFGS, save results.

Handles parameter constraints, boundary transforms for time parameters,
and multiple random starts (following pw_cfit5_td6_fix29.py conventions).

Usage:
    python run_fit.py config_angle.yml --out fit_results
"""

import os, sys, json, time, argparse
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
    keys = cst.get_free_keys()

    # Fix reference total and first LS to 1+0j
    for pref, klist in [('total', [k for k in keys if k.endswith('_total_0')]),
                        ('g_ls', [k for k in keys if k.endswith('_g_ls_0') and '_g_lsbar_' not in k])]:
        if klist:
            fitter.set_fixed(klist[0], 1.0 + 0.0j)
            print(f"  Fixed {klist[0]} = 1+0j", flush=True)

    # Equalize mirror pairs: aX(m) ↔ aX(p) g_ls (CP symmetry)
    mirror_gls_pairs = [
        ('a1(1260)m', 'a1(1260)p'), ('a1(1640)m', 'a1(1640)p'),
        ('a2(1320)m', 'a2(1320)p'), ('pi2(1670)m', 'pi2(1670)p'),
        ('pi1300m', 'pi1300p'), ('pi1600m', 'pi1600p'),
        ('pi1(1600)m', 'pi1(1600)p')]
    for rm, rp in mirror_gls_pairs:
        for kp in keys:
            if kp.startswith(f'{rp}->') and '_g_ls_' in kp and '_g_lsbar_' not in kp:
                km = kp.replace(rp, rm)
                if km in keys and kp in keys:
                    # km = -1 * kp (CP sign convention from reference)
                    cst.set_linear(km, [(kp, -1.0)])
                    print(f"  Set {kp} = {km} with scale=-1", flush=True)

    # Equalize KMA↔KMB pole/prod parameters
    for kk in list(keys):
        if 'KMA_' in kk:
            kmb = kk.replace('KMA_', 'KMB_')
            if kmb in keys:
                fitter.set_equal(kmb, kk)
                print(f"  Set {kmb} = {kk}", flush=True)

    # Fix K-matrix pole/prod indices 3-5 to 0
    for fix_key in list(keys):
        for suffix in ['_pole_3', '_pole_4', '_pole_5', '_prod_3', '_prod_4', '_prod_5']:
            if suffix in fix_key and ('KMA_' in fix_key or 'KMB_' in fix_key or 'KMC_' in fix_key or 'KM2_' in fix_key):
                fitter.set_fixed(fix_key, 0.0 + 0.0j)
                print(f"  Fixed {fix_key} = 0+0j", flush=True)

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
    print(f"\nInitial parameters ({n_free} free params, {len(free_keys)} keys):", flush=True)
    idx = 0
    for ki, key in enumerate(free_keys[:20]):  # first 20 keys
        if cst._is_complex_key(key):
            z = cst._to_complex(x0[idx], x0[idx+1])
            print(f"  {key:55s} = {abs(z):.4f} * exp({np.angle(z):.4f}i)",
                  flush=True)
            idx += 2
        else:
            print(f"  {key:55s} = {x0[idx]:.6f}", flush=True)
            idx += 1
    if len(free_keys) > 20:
        print(f"  ... ({len(free_keys)-20} more)", flush=True)

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

    # ---- 9. Extract results with transforms ----
    # Physical parameter values (applying reverse transform)
    x_phys = result.x.copy()
    for ki, tr in bound_trans.items():
        x_phys[ki] = tr(result.x[ki])

    model_opt = fitter.unpack(x_phys)
    cst = fitter.cst
    phys = cst.get_physical(model_opt)
    fitter._params = phys

    # Save params to a.json format
    params_path = os.path.join(out_dir, 'final_params.json')
    fitter.mapper.save_params(params_path, phys)
    print(f"  Params saved: {params_path}", flush=True)

    # Build result dict
    result_dict = {
        'success': result.success,
        'status': result.status,
        'fun': float(result.fun),
        'nfev': result.nfev,
        'nit': result.nit,
        'n_free_params': len(result.x),
        'x_opt': x_phys.tolist(),
        'x_raw': result.x.tolist(),
        'free_keys': free_keys,
        'message': str(result.message),
    }

    # Hessian inverse with boundary propagation
    if hasattr(result, 'hess_inv'):
        try:
            h = result.hess_inv
            if hasattr(h, 'todense'):
                h = h.todense()
            h = np.asarray(h)
            # Propagate through boundary transforms: H_ij → g_i * H_ij * g_j
            g_vec = np.ones(n_free)
            for ki, tr in bound_trans.items():
                g_vec[ki] = tr.grad(result.x[ki])
            hess_phys = g_vec[:, None] * h * g_vec[None, :]
            result_dict['hess_inv'] = hess_phys.tolist()
            result_dict['errors'] = {free_keys[i]: float(np.sqrt(max(hess_phys[i,i], 0)))
                                      for i in range(n_free)}
        except Exception as e:
            print(f"  Could not save hess_inv: {e}", flush=True)
            result_dict['hess_inv'] = None

    # Save phys params
    phys_arrs = {}
    for k, v in phys.items():
        if isinstance(v, np.ndarray):
            phys_arrs[k] = v
        elif isinstance(v, complex):
            phys_arrs[k + '_re'] = np.array([v.real])
            phys_arrs[k + '_im'] = np.array([v.imag])

    result_npz = os.path.join(out_dir, 'fit_results.npz')
    np.savez(result_npz, **result_dict, **phys_arrs)
    print(f"  Results saved: {result_npz}", flush=True)

    # Summary JSON
    meta_out = os.path.join(out_dir, 'fit_meta.json')
    with open(meta_out, 'w') as f:
        json.dump({'success': result.success, 'status': result.status,
                   'fun': float(result.fun), 'nfev': result.nfev,
                   'nit': result.nit, 'n_free': n_free,
                   'message': str(result.message),
                   'errors': result_dict.get('errors')},
                  f, indent=2)

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
