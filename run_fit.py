#!/usr/bin/env python3
"""
Real NLL computation using ampfit with constraints from pw_cfit5_td6_fix29.py.

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


def build_constraints(all_comb):
    """Build fixed/same/scale constraints matching pw_cfit5_td6_fix29.py."""
    all_params = set()
    for comb in all_comb:
        for p in comb:
            if isinstance(p, str):
                all_params.add(p)

    fixed_slots = {}
    same_params = []
    scale_params = {}

    # --- Fixed slots: '{name}r' / '{name}i' for complex, '{name}' for real ---
    for p in sorted(all_params):
        if p.endswith("g_ls_0"):
            fixed_slots[p + 'r'] = 1.0
            fixed_slots[p + 'i'] = 0.0
        elif p.endswith("pole.0"):
            fixed_slots[p + 'r'] = 1.0
            fixed_slots[p + 'i'] = 0.0
        elif p.endswith("point_5"):
            fixed_slots[p + 'r'] = 1.0
            fixed_slots[p + 'i'] = 0.0
        elif p.endswith("fix1"):
            fixed_slots[p + 'r'] = 1.0
            fixed_slots[p + 'i'] = 0.0

    # Fix B->rhoA.rhoB total (matching reference TFPWA)
    fix_total = "B->rhoA.rhoBrhoA->pip1.pim1rhoB->pip2.pim2_total_0"
    if fix_total in all_params:
        fixed_slots[fix_total + 'r'] = 1.0
        fixed_slots[fix_total + 'i'] = 0.0

    # --- Same params and scales ---
    for r1 in ["a1(1260)", "a1(1640)", "a2(1320)", "pi1300",
               "pi1600", "a2(1700)", "pi2(1670)", "pi1(1600)"]:
        name_ps, name_ms = [], []
        fixed = True
        for r2 in ["rhoA", "f0(500)", "f0(980)", "f2(1270)"]:
            name_p = f"B->{r1}p.pim2{r1}p->{r2}.pip2{r2}->pip1.pim1_total_0"
            name_m = f"B->{r1}m.pip2{r1}m->{r2}.pim2{r2}->pip1.pim1_total_0"
            if name_p in all_params:
                name_ps.append(name_p)
                name_ms.append(name_m)
            for idx in range(3):
                key_m = f"{r1}m->{r2}.pim2_g_ls_{idx}"
                if key_m in all_params:
                    if fixed:
                        fixed = False
                    else:
                        key_p = f"{r1}p->{r2}.pip2_g_ls_{idx}"
                        if key_m + 'r' in fixed_slots:
                            del fixed_slots[key_m + 'r']
                            del fixed_slots[key_m + 'i']
                        if key_p + 'r' in fixed_slots:
                            del fixed_slots[key_p + 'r']
                            del fixed_slots[key_p + 'i']
                        same_params.append([key_m, key_p])
                    if r2 == "rhoA":
                        scale_params[key_m] = -1
        same_params.append(name_ps)
        same_params.append(name_ms)

    # KMA/KMB symmetry
    for i in range(3):
        for j in ["pole", "prod"]:
            ka = f"KMA_{j}.{i}"
            kb = f"KMB_{j}.{i}"
            if ka in all_params:
                same_params.append([ka, kb])
    for i in range(3, 5):
        for j in ["pole", "prod"]:
            for prefix in ["KMA", "KMB", "KMC", "KM2"]:
                key = f"{prefix}_{j}.{i}"
                if key in all_params:
                    fixed_slots[key + 'r'] = 0.0
                    fixed_slots[key + 'i'] = 0.0

    return fixed_slots, same_params, scale_params


def load_npz(npz_path, max_events=None):
    """Load .npz data and format for the kernel."""
    data = np.load(npz_path)
    if "angles" in data and "angle" not in data:
        data = dict(data)
        data["angle"] = data.pop("angles")

    n_events = data["mass"].shape[0]
    if max_events is not None and max_events < n_events:
        n_events = max_events
        idx = np.random.RandomState(0).choice(data["mass"].shape[0], n_events, replace=False)
    else:
        idx = slice(None)

    out = {
        "mass": data["mass"][idx].reshape(n_events, -1),
        "q": data["q"][idx].reshape(n_events, -1),
        "angle": data["angle"][idx].reshape(n_events, -1, 3),
        "time": data["time"][idx].astype(np.float64),
        "frac": data["frac"][idx].astype(np.float64),
        "bkg": data["bkg_raw"][idx].astype(np.float64),
        "weight": data["weight"][idx].astype(np.float64),
    }
    assert not np.any(np.isnan(out["mass"])), "NaN in mass"
    return out, n_events


def main():
    parser = argparse.ArgumentParser(description="NLL computation with ampfit")
    parser.add_argument("--debug", action="store_true", help="Use 1K data / 10K phsp")
    parser.add_argument("--backend", default="cuda",
                        choices=["cuda", "cuda32", "cuda64", 
                                 "cuda_v2", "cuda64_v2", "cuda32_v2",
                                 "cuda_v3", "cuda64_v3", "cuda32_v3",
                                 "numpy", "onnx", "onnx_cpu", "onnx_cuda"],
                        help="Compute backend")
    parser.add_argument("--config", default="config_angle.yml")
    parser.add_argument("--data", default="data/data_arrays.npz")
    parser.add_argument("--phsp", default="data/phsp_arrays.npz")
    parser.add_argument("--check-grad", action="store_true", help="Verify gradient")
    parser.add_argument("--fit", action="store_true", help="Run BFGS minimization")
    parser.add_argument("--maxiter", type=int, default=200, help="Max fit iterations")
    parser.add_argument("--save", type=str, default=None, help="Save fit results to JSON")
    parser.add_argument("--plot", type=str, nargs='?', const='plots/',
                        default=None, help="Plot distributions (optional: output dir)")
    parser.add_argument("--fix-mass-width", action="store_true",
                        help="Fix all mass and width parameters to config defaults")
    parser.add_argument("--init", type=str, default=None,
                        help="Initial parameters JSON file (from save_params output)")
    args = parser.parse_args()

    # ==================================================================
    # 1. Create fitter with constraints
    # ==================================================================
    print("=" * 70)
    print("SETUP")
    print("=" * 70)

    fitter = Fitter(args.config, backend=args.backend)
    fixed_slots, same_params, scale_params = build_constraints(fitter.all_comb)
    # Optionally add mass/width fixes to the fixed slots
    if args.fix_mass_width:
        for name in fitter.config.m0_phys_name:
            val = float(fitter.defaults[name])
            fixed_slots[name] = val
        for name in fitter.config.g0_phys_name:
            val = float(fitter.defaults[name])
            fixed_slots[name] = val

    # gamma is free (fitted time parameter) — do NOT fix to 0
    fixed_slots["delta_gamma"] = 0.0
    fixed_slots["delta_m"]= 0.506
    fixed_slots["A_prod"] = 0.0
    fixed_slots["poqr"] = 1.0
    fixed_slots["poqi"] = 0.0

    for name in []: # "a1(1260)", "a1(1640)", "a2(1320)"]:
        if f"{name}p_mass" in fixed_slots:
            del fixed_slots[f"{name}p_mass"]
        if f"{name}m_mass" in fixed_slots:
            del fixed_slots[f"{name}m_mass"]
        if f"{name}p_width" in fixed_slots:
            del fixed_slots[f"{name}p_width"]
        if f"{name}m_width" in fixed_slots:
            del fixed_slots[f"{name}m_width"]
        same_params.append([f"{name}p_mass",f"{name}m_mass"])
        same_params.append([f"{name}p_width",f"{name}m_width"])

    fitter.set_fixed(fixed_slots)
    fitter.set_same(same_params)
    fitter.set_scale(scale_params)
    print(f"Fixed: {len(fixed_slots)} slots, Same: {len(same_params)} groups, Scale: {len(scale_params)}")

    # Set boundary ranges for masses and widths
    for name in fitter.config.m0_phys_name:
        val = float(fitter.defaults[name])
        if name not in fitter._fixed_slots:
            fitter.set_range(name, val - 0.1, val + 0.1)
    for name in fitter.config.g0_phys_name:
        if name not in fitter._fixed_slots:
            val = float(fitter.defaults[name])
            lo = max(0.0, val - 0.1)
            hi = min(2.0, val + 0.1)
            fitter.set_range(name, lo, hi)

    # ==================================================================
    # 2. Load data
    # ==================================================================
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    max_data = 1000 if args.debug else None
    max_phsp = 10000 if args.debug else None

    t0 = time.time()
    data_np, n_data = load_npz(args.data, max_events=max_data)
    phsp_np, n_phsp = load_npz(args.phsp, max_events=max_phsp)
    load_time = time.time() - t0
    print(f"  Loaded {n_data:,} data + {n_phsp:,} phsp events in {load_time:.2f}s")


    t0 = time.time()
    fitter.set_phsp(phsp_np)
    phsp_gpu_time = time.time() - t0
    print(f"  Phsp → GPU: {phsp_gpu_time:.2f}s")

    t0 = time.time()
    fitter.set_data(data_np)
    data_gpu_time = time.time() - t0
    print(f"  Data → GPU: {data_gpu_time:.2f}s")

    # m0 and g0 default values come from config.yml particle definitions
    # (lazy-loaded by Fitter.defaults)
    n_free = len(fitter.free_param_names())
    print(f"  Free params: {n_free}")

    # ==================================================================
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
        print(f"Initialized from {args.init}")
    else:
        x0 = fitter.initial_values(seed=42)
    print(f"x0 shape: {x0.shape}")

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
        result = fitter.fit(x0, maxiter=args.maxiter, disp=True)
        fit_time = time.time() - t0
        print(f"\n  Fit time: {fit_time:.2f}s")
        print(f"  Final NLL: {result.fun:.6f}")
        print(f"  nfev: {result.nfev}, nit: {result.nit}")
        print(f"  success: {result.success}")

        # Print uncertainties for top parameters
        uncert = fitter.get_uncertainties(result)
        names = list(uncert.keys())

        # Show largest-magnitude free parameters
        vals_err = [(n, uncert[n][0], uncert[n][1]) for n in names]
        vals_err.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  Top free parameters:")
        for name, val, err in vals_err[:5]:
            print(f"    {name:50s} = {val:+.6f} ± {err:.6f}")

        # Save results if requested
        save_path = args.save
        if save_path is None:
            prefix = os.path.splitext(os.path.basename(args.config))[0]
            save_path = f"{prefix}_fit_results.json"
        fitter.save_params(result, save_path)
        print(f"  Results saved to {save_path}")

        # Plot post-fit distributions
        if args.plot:
            fitter.plot(result, prefix=args.plot)
            print(f"  Plots saved to {args.plot}")

    # ==================================================================
    # 6. Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("─" * 70)
    print(f"    Data       {n_data:>10,}")
    print(f"    Phsp       {n_phsp:>10,}")
    print(f"    Free vars  {n_free:>10,}")
    print(f"    Load time  {load_time:>10.2f}s")
    if args.fit:
        print(f"    Fit time   {fit_time:>10.2f}s")
        print(f"    Fit NLL    {result.fun:>10.6f}")
        print(f"    nfev/nit   {result.nfev:>4d} / {result.nit}")
    else:
        print(f"    NLL        {nll:>10.6f}")
        print(f"    Compute    {elapsed:>10.2f}s")
    print("=" * 70)

    fitter.free()


if __name__ == "__main__":
    main()
