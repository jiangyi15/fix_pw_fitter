#!/usr/bin/env python3
"""
Full PWA fit: load converted data, build tables, fit with BFGS, save results.

Usage:
    python run_fit.py config_angle.yml --data converted_data/data_arrays.npz \\
        --phsp converted_data/phsp_arrays.npz --out fit_results.npz
"""

import os, sys, json, time, argparse
import numpy as np


def run_fit(config_path, data_npz, phsp_npz, out_path='fit_results.npz',
            method='BFGS', maxiter=200, fix_2pi=True, gamma_table_path=None):
    """Run full PWA fit and save results."""
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pwa_gpu.fitter import PWAFitter

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
    # Look for a.json alongside config
    config_dir = os.path.dirname(os.path.abspath(config_path))
    a_json = os.path.join(config_dir, 'a.json')
    if os.path.exists(a_json):
        print(f"Loading params from {a_json}...", flush=True)
        fitter.load_params(a_json)
    else:
        print("No a.json found, using random params", flush=True)
    fitter.summary()

    # ---- 4. Set constraints ----
    if fix_2pi:
        print("Fixing 2π resonances...", flush=True)
        t1 = time.time()
        fitter.fix_2pi_resonances()
        print(f"  {time.time()-t1:.1f}s", flush=True)

    # ---- 5. Load data ----
    print("Loading data...", flush=True)
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
    print(f"Data mass stride: {d['mass'].shape[1]*d['mass'].shape[2]} "
          f"(expect {fitter.cfg.get('n_mass_columns', '?')})")

    # ---- 6. Run fit ----
    options = {'disp': True, 'maxiter': maxiter}
    print(f"\nRunning BFGS fit ({method}, maxiter={maxiter})...", flush=True)
    t0 = time.time()

    import scipy.optimize
    f_fit = fitter.make_fit_func(data, phsp)
    x0 = fitter.get_free_values()

    result = scipy.optimize.minimize(
        f_fit, x0, jac=True, method=method,
        options=options,
    )
    fit_time = time.time() - t0
    print(f"  Fit done ({fit_time:.1f}s)", flush=True)

    # ---- 7. Extract results ----
    model_opt = fitter.unpack(result.x)
    cst = fitter.cst
    phys = cst.get_physical(model_opt)
    fitter._params = phys

    # Save params to a.json format
    params_path = out_path.replace('.npz', '_params.json')
    fitter.mapper.save_params(params_path, phys)
    print(f"  Params saved: {params_path}", flush=True)

    # Build result dict
    result_dict = {
        'success': result.success,
        'status': result.status,
        'fun': float(result.fun),
        'nfev': result.nfev,
        'nit': result.nit,
        'fit_time': fit_time,
        'n_free_params': len(result.x),
        'x': result.x.tolist(),
        'free_keys': fitter.get_free_keys(),
        'message': str(result.message),
    }

    # Hessian inverse
    if hasattr(result, 'hess_inv'):
        try:
            h = result.hess_inv
            if hasattr(h, 'todense'):
                h = h.todense()
            result_dict['hess_inv'] = np.asarray(h).tolist()
        except Exception as e:
            print(f"  Could not save hess_inv: {e}", flush=True)
            result_dict['hess_inv'] = None

    # Save phys params in the npz
    phys_arrs = {}
    for k, v in phys.items():
        if isinstance(v, np.ndarray):
            phys_arrs[k] = v
        elif isinstance(v, complex):
            phys_arrs[k + '_re'] = np.array([v.real])
            phys_arrs[k + '_im'] = np.array([v.imag])

    np.savez(out_path, **result_dict, **phys_arrs)
    print(f"  Results saved: {out_path}", flush=True)

    # Also save as JSON (without large arrays)
    meta_out = out_path.replace('.npz', '_meta.json')
    with open(meta_out, 'w') as f:
        json.dump({k: result_dict[k] for k in ['success','status','fun',
                  'nfev','nit','fit_time','n_free_params','message']},
                  f, indent=2)
    print(f"  Meta saved: {meta_out}", flush=True)

    # Print summary
    print(f"\n{'='*50}", flush=True)
    print(f"Fit {'SUCCEEDED' if result.success else 'FAILED'}", flush=True)
    print(f"  NLL:       {result.fun:.4f}")
    print(f"  Iterations: {result.nit}")
    print(f"  Time:       {fit_time:.1f}s")
    print(f"  Status:     {result.status}")
    print(f"  Message:    {result.message}")
    print(f"{'='*50}", flush=True)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PWA fit')
    parser.add_argument('config', help='config_angle.yml path')
    parser.add_argument('--data', default='converted_data/data_arrays.npz',
                        help='signal data .npz')
    parser.add_argument('--phsp', default='converted_data/phsp_arrays.npz',
                        help='phsp data .npz')
    parser.add_argument('--out', default='fit_results.npz',
                        help='output .npz path')
    parser.add_argument('--method', default='BFGS')
    parser.add_argument('--maxiter', type=int, default=200)
    parser.add_argument('--no-fix-2pi', action='store_true',
                        help='Do not fix 2π resonance params')
    parser.add_argument('--tables', default=None,
                        help='Pre-computed gamma tables .npz')

    args = parser.parse_args()
    run_fit(config_path=args.config, data_npz=args.data, phsp_npz=args.phsp,
            out_path=args.out, method=args.method, maxiter=args.maxiter,
            fix_2pi=not args.no_fix_2pi, gamma_table_path=args.tables)
