#!/usr/bin/env python3
"""
Real NLL computation using ampfit.

Usage:
    python run_fit.py                      # Full fit with all data
    python run_fit.py --debug              # Quick test with 1K events
"""
import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ampfit import Fitter


def main():
    parser = argparse.ArgumentParser(description="NLL computation with ampfit")
    parser.add_argument("--debug", action="store_true", help="Use 1K data / 10K phsp")
    parser.add_argument("--backend", default="cuda_v3",
                        help='Compute backend: a name like "cuda_v3", '
                             'or a YAML dict like '
                             '\'{name: integrated, base: cuda_v3}\' '
                             "(note: spaces required after colons)")
    parser.add_argument("--config", default="config_angle.yml")
    parser.add_argument("--data", default=None,
                        help="data .npz (default: use the config's "
                             "data_arr / prefix files via load_all_data)")
    parser.add_argument("--phsp", default=None,
                        help="phsp .npz (default: config-driven)")
    parser.add_argument("--check-grad", action="store_true", help="Verify gradient")
    parser.add_argument("--fit", action="store_true", help="Run BFGS minimization")
    parser.add_argument("--maxiter", type=int, default=200, help="Max fit iterations")
    parser.add_argument("--save", type=str, default=None, help="Save fit results to JSON")
    parser.add_argument("--plot", type=str, nargs='?', const='plots/',
                        default=None, help="Plot distributions (optional: output dir)")
    parser.add_argument("--init", type=str, default=None,
                        help="Initial parameters JSON file (from save_params output)")
    args = parser.parse_args()

    # ==================================================================
    # 1. Create fitter with constraints
    # ==================================================================
    print("=" * 70)
    print("SETUP")
    print("=" * 70)

    # Parse backend: YAML dict or plain name
    backend_spec = args.backend
    if isinstance(backend_spec, str) and backend_spec.strip().startswith("{"):
        import yaml
        backend_spec = yaml.safe_load(backend_spec)

    fitter = Fitter(args.config, backend=backend_spec)

    # All structural + config constraints via plugin system
    # (mass/width fixing/floating is controlled by the config:
    #  fix_mass_width_default: true + particle float: fields)
    fitter.apply_constrains()

    # KMA/KMB/KMC/KM2 for i>=3: fixed to 0
    for prefix in ["KMA", "KMB", "KMC", "KM2"]:
        for j in ["pole", "prod"]:
            for i in range(3, 5):
                key = f"{prefix}_{j}.{i}"
                if key in fitter.cm._all_names:
                    fitter.set_fixed({key + "r": 0.0, key + "i": 0.0})

    print(f"Fixed: {len(fitter.cm.fixed_slots)} slots, "
          f"Same: {len(fitter.cm.same_params)} aliases, "
          f"Scale: {len(fitter.cm.scale_params)} transforms")

    # ==================================================================
    # 2. Load data
    # ==================================================================
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    t0 = time.time()
    have_both = bool(args.data) and bool(args.phsp)
    if not have_both and (bool(args.data) or bool(args.phsp)):
        import warnings
        warnings.warn("Provide BOTH --data and --phsp, or neither; "
                      "falling back to the config datasets.")
    if have_both:
        # explicit npz for both datasets
        n_comp = int(fitter.kernel_config["angle_k"].shape[1])
        max_data = 1000 if args.debug else None
        max_phsp = 10000 if args.debug else None
        data_np, n_data = Fitter.load_npz(args.data, max_events=max_data,
                                         n_angle_comp=n_comp)
        phsp_np, n_phsp = Fitter.load_npz(args.phsp, max_events=max_phsp,
                                         n_angle_comp=n_comp)
        fitter.set_phsp(phsp_np)
        fitter.set_data(data_np)
    else:
        # both datasets come from the config (load_all_data installs them)
        data_np, phsp_np = fitter.load_all_data()
        n_data = data_np["mass"].shape[0]
        n_phsp = phsp_np["mass"].shape[0]
        print(f"  [config] loaded data + phsp via load_all_data()")
    load_time = time.time() - t0
    print(f"  Loaded {n_data:,} data + {n_phsp:,} phsp events in {load_time:.2f}s")

    n_free = len(fitter.free_param_names())
    print(f"  Free params: {n_free}")
    # 3. Compute NLL
    # ==================================================================
    print("\n" + "=" * 70)
    print("COMPUTING NLL")
    print("=" * 70)

    if args.init:
        import json
        with open(args.init) as f:
            init_data = json.load(f)
        x0 = fitter.values_from_dict(init_data)
        fitter.load_fixed_from_dict(init_data)
        print(f"Initialized from {args.init}")
    else:
        x0 = fitter.initial_values(seed=None)
    print(f"x0 shape: {x0.shape}")

    # Print initial parameters (physical values after transforms)
    names = fitter.free_param_names()
    phy_val = fitter.cm.flat_resolve(x0, stop_after="bounds")
    print(f"\n  Initial parameters:")
    for name, val in zip(names, x0):
        print(f"    {name:50s} = {phy_val.get(name):+.6f}  (x={val:+.6f})")
    print()

    t0 = time.time()
    nll, grad_x = fitter.get_nll(x0)
    elapsed = time.time() - t0
    print(f"NLL = {nll:.6f}, time = {elapsed:.2f}s")
    print(f"Grad range: [{grad_x.min():.4f}, {grad_x.max():.4f}]")

    # ==================================================================
    # 4. Verify gradient (optional)
    # ==================================================================
    if args.check_grad:
        print("\n" + "=" * 70)
        print("GRADIENT VERIFICATION")
        print("=" * 70)
        eps = 1e-5
        for k in range(min(3, len(x0))):
            xp = x0.copy(); xp[k] += eps
            nll_p, _ = fitter.get_nll(xp)
            xm = x0.copy(); xm[k] -= eps
            nll_m, _ = fitter.get_nll(xm)
            num = (nll_p - nll_m) / (2 * eps)
            err = abs(grad_x[k] - num) / (max(abs(num), 1e-10) + 1e-10)
            status = "✓" if err < 0.01 else "✗"
            print(f"  x[{k:2d}]: ana={grad_x[k]:+.4e} num={num:+.4e} rel_err={err:.2e} {status}")

    # ==================================================================
    # 5. Fit (optional)
    # ==================================================================
    result = None
    fit_time = 0.0
    if args.fit:
        print("\n" + "=" * 70)
        print("FITTING")
        print("=" * 70)

        t0 = time.time()
        try:
            result = fitter.fit(x0, maxiter=args.maxiter, disp=True)
        except KeyboardInterrupt:
            pass
        fit_time = time.time() - t0

        if result is None or not hasattr(result, 'fun'):
            if fitter._last_xk is not None:
                ckpt_path = args.save if args.save else f"{os.path.splitext(os.path.basename(args.config))[0]}_fit_results.json"
                ckpt_dir = os.path.dirname(ckpt_path) if os.path.dirname(ckpt_path) else '.'
                save_path = os.path.join(ckpt_dir, "checkpoint.json")
                fitter.save_params(fitter._last_xk, save_path)
                print(f"\n  Fit interrupted after {fit_time:.1f}s")
                print(f"  Checkpoint saved to {save_path}")
                print(f"  Resume with: --init {save_path}")
            else:
                print(f"\n  Fit interrupted after {fit_time:.1f}s - no iterations completed")
            sys.exit(1)

        print(f"\n  Fit time: {fit_time:.2f}s")
        print(f"  Final NLL: {result.fun:.6f}")
        print(f"  nfev: {result.nfev}, nit: {result.nit}")
        print(f"  success: {result.success}")

        # Print uncertainties for all free parameters
        uncert = fitter.get_uncertainties(result)
        names = list(uncert.keys())
        vals_err = [(n, uncert[n][0], uncert[n][1]) for n in names]
        print(f"\n  Fitted parameters:")
        for name, val, err in vals_err:
            print(f"    {name:50s} = {val:+.6f} +/- {err:.6f}")

        # Save results
        save_path = args.save
        if save_path is None:
            prefix = os.path.splitext(os.path.basename(args.config))[0]
            save_path = f"{prefix}_fit_results.json"
        fitter.save_params(result, save_path)
        print(f"  Results saved to {save_path}")

        # Save constraints alongside results
        constraints_path = os.path.splitext(save_path)[0] + "_constraints.json"
        fitter.save_constraints(constraints_path)

        # Plot post-fit distributions
        if args.plot:
            fitter.plot(result, prefix=args.plot)
            print(f"  Plots saved to {args.plot}")

    # ==================================================================
    # 6. Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("-" * 70)
    print(f"    Data       {n_data:>10,}")
    print(f"    Phsp       {n_phsp:>10,}")
    print(f"    Free vars  {n_free:>10,}")
    print(f"    Load time  {load_time:>10.2f}s")
    if args.fit and result is not None and hasattr(result, 'fun'):
        print(f"    Fit time   {fit_time:>10.2f}s")
        print(f"    Fit NLL    {result.fun:>10.6f}")
        print(f"    nfev/nit   {result.nfev:>4d} / {result.nit}")
    elif args.fit:
        print(f"    NLL        {nll:>10.6f}")
        print(f"    Compute    {elapsed:>10.2f}s")
    print("=" * 70)

    fitter.free()


if __name__ == "__main__":
    main()
