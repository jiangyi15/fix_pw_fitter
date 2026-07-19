#!/usr/bin/env python3
"""Phase 2: GP active learning from conditional scan results.

Loads the conditional scan points saved by profile_likelihood_2d.py
(or any points in the _points/ directory), runs GP active learning
with profiled fits at contour-focused locations.

Usage:
    # After profile_likelihood_2d.py --output profile:
    python scripts/profile_gp_2d.py results.json config.yml \\
        --param1 mass --param2 width --input profile_points/ --output profile

    # Resume from partial results:
    python scripts/profile_gp_2d.py results.json config.yml \\
        --param1 mass --param2 width --input profile_points/ --output profile
"""

import sys, os, json, argparse
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ampfit import Fitter
from ampfit.gp import GP


# ── Acquisition ────────────────────────────────────────────────────

def acquire_contour(mu, std, level=2.30):
    """Pick points inside the contour (ΔNLL < level), score by uncertainty."""
    below = mu < level
    if np.any(below):
        return np.where(below, std, 0.0)
    return np.zeros_like(mu)


def quadratic_from_hessian(cov22, v0_1, v0_2):
    H22 = np.linalg.inv(cov22)
    def q(v1, v2):
        dv = np.array([v1 - v0_1, v2 - v0_2])
        return 0.5 * dv @ H22 @ dv
    return q


# ── Helpers ────────────────────────────────────────────────────────

def load_fit_result(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("value", data), data.get("error", {}), data.get("status", {}).get("NLL")


def run_profiled_fit(fitter, param1, param2, v1, v2,
                     x_best_all, free_names_all, maxiter=200, hess_inv_full=None):
    fitter.set_fixed({param1: v1, param2: v2})
    free_now = fitter.free_param_names()
    x0 = np.array([x_best_all[free_names_all.index(n)] for n in free_now])
    opts = {}
    if hess_inv_full is not None:
        idx = [free_names_all.index(n) for n in free_now]
        H_full = np.linalg.inv(np.asarray(hess_inv_full))
        H_sub = H_full[np.ix_(idx, idx)]
        H_sub = (H_sub + H_sub.T) / 2
        # SVD with singular value clipping for robust positive definite inversion
        U, s, Vt = np.linalg.svd(H_sub)
        s = np.maximum(s, 1e-3 * s.max())
        hess_inv0 = (U / s) @ Vt
        hess_inv0 = (hess_inv0 + hess_inv0.T) / 2  # scipy checks exact symmetry
        opts['hess_inv0'] = hess_inv0
    try:
        nll_at_x0, _grad = fitter.get_nll(x0)
        print(f"    NLL at x0 = {nll_at_x0:.4f} (fix {param1}={v1:.4f}, {param2}={v2:.4f})", flush=True)
        result = fitter.fit(x0, maxiter=maxiter, disp=True,
                            options=opts if opts else None)
        return result.fun, result
    except Exception as e:
        print(f"    Fit error: {e}")
        return np.nan, None


# ── Main ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="GP active learning (Phase 2)")
    ap.add_argument("fit_json", help="Fit result JSON")
    ap.add_argument("config", help="Fitter config YAML")
    ap.add_argument("--param1", required=True)
    ap.add_argument("--param2", required=True)
    ap.add_argument("--input", default="profile_points",
                    help="Directory with conditional scan results (default: profile_points/)")
    ap.add_argument("--output", default="profile")
    ap.add_argument("--grid", type=int, default=200)
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--mle", action="store_true", help="Optimize GP hyperparameters via MLE")
    ap.add_argument("--backend", default="cuda_v3_sparse")
    ap.add_argument("--fit-maxiter", type=int, default=200)
    ap.add_argument("--sigma-thresh", type=float, default=0.05)
    args = ap.parse_args()

    base = os.path.splitext(args.output)[0]
    pts_dir = args.input
    os.makedirs(pts_dir, exist_ok=True)

    # ── Load fitter ───────────────────────────────────────────────
    print(f"Loading: {args.fit_json}")
    fitter = Fitter(args.config, backend=args.backend)
    fitter.set_default_params()
    cp = os.path.splitext(args.fit_json)[0] + "_constraints.json"
    if os.path.exists(cp):
        fitter.load_constraints(cp)
    val_dict, err_dict, nll0_saved = load_fit_result(args.fit_json)
    res = fitter.load_results(args.fit_json)
    x_best = res.x
    hess_inv_full = getattr(res, 'hess_inv', None)
    try:
        fitter.load_all_data()
    except Exception as e:
        print(f"WARNING: no data ({e})")

    free_names = fitter.free_param_names()
    idx1, idx2 = free_names.index(args.param1), free_names.index(args.param2)
    v0_1 = float(val_dict.get(args.param1, 0.0))
    v0_2 = float(val_dict.get(args.param2, 0.0))
    nll0 = nll0_saved or 0.0

    # Round-trip check: does x_best reproduce nll0?
    nll_check, _ = fitter.get_nll(x_best)
    print(f"  nll0 (from status)  = {nll0:.4f}")
    print(f"  nll at x_best       = {nll_check:.4f}")
    print(f"  round-trip match: {abs(nll_check - nll0) < 1.0}")

    # ── Grid range (from scan data) + quadratic prior (from Hessian) ──
    # The grid covers the scan data extent with 20% padding.
    # No pre-defined "range" — the GP's acquisition tells us where to sample.
    # ── Hessian → prior + grid range ──────────────────────────────
    # Quadratic prior from conditional Hessian (valid near minimum).
    if hess_inv_full is not None:
        H_full = np.linalg.inv(np.asarray(hess_inv_full))
        H_pp = H_full[np.ix_([idx1, idx2], [idx1, idx2])]
        H_pp = (H_pp + H_pp.T) / 2
        cond_cov = np.linalg.inv(H_pp)
        J1 = fitter.cm.bounds[args.param1].grad(0) if args.param1 in fitter.cm.bounds else 1.0
        J2 = fitter.cm.bounds[args.param2].grad(0) if args.param2 in fitter.cm.bounds else 1.0
        cond_cov_phys = cond_cov.copy()
        cond_cov_phys[0] *= J1; cond_cov_phys[:, 0] *= J1
        cond_cov_phys[1] *= J2; cond_cov_phys[:, 1] *= J2
        cov22 = cond_cov_phys
    else:
        err1 = float(err_dict.get(args.param1, abs(v0_1)*0.05))
        err2 = float(err_dict.get(args.param2, abs(v0_2)*0.05))
        cov22 = np.array([[err1**2, 0], [0, err2**2]])
    quad_mean = quadratic_from_hessian(cov22, v0_1, v0_2)
    H22 = np.linalg.inv(cov22)
    ls = 1.0 / np.sqrt(np.maximum(np.linalg.eigvalsh(H22), 1e-10))

    # ── Load existing points ──────────────────────────────────────
    X_train = [[v0_1, v0_2]]
    y_train = [0.0]
    ftol = 1e-5  # profiled fit noise (very small)
    sigma_n_vec = [ftol]  # centre point: ΔNLL=0 exactly, tiny noise
    is_cond = [True]  # centre is from conditional scan
    lo1, hi1 = v0_1, v0_1
    lo2, hi2 = v0_2, v0_2
    if os.path.isdir(pts_dir):
        for fn in sorted(os.listdir(pts_dir)):
            if not fn.endswith('.json'):
                continue
            with open(os.path.join(pts_dir, fn)) as f:
                dat = json.load(f)
            if dat.get("failed"):
                continue
            v1 = float(dat.get("value", {}).get(args.param1, float('nan')))
            v2 = float(dat.get("value", {}).get(args.param2, float('nan')))
            dn = dat.get("delta_nll", float('nan'))
            nll = dat.get("status", {}).get("NLL", float('nan'))
            if np.isfinite(nll) and np.isfinite(v1) and np.isfinite(v2):
                pt = [v1, v2]
                if pt not in X_train:
                    X_train.append(pt)
                    delta = nll - nll0  # ΔNLL from status.NLL (ignore delta_nll field)
                    y_train.append(delta)
                    cond = not fn.startswith('gp_')
                    is_cond.append(cond)
                    # Conditional points: σ = |ΔNLL| (large noise far from min)
                    # Profiled points:    σ = ftol (tiny, trusted)
                    noise = ftol if not cond else abs(delta)
                    sigma_n_vec.append(noise)
                    lo1, hi1 = min(lo1, v1), max(hi1, v1)
                    lo2, hi2 = min(lo2, v2), max(hi2, v2)

    n_init = len(X_train)
    X_arr = np.array(X_train)
    y_arr = np.array(y_train)
    sn_arr = np.array(sigma_n_vec)
    is_cond_arr = np.array(is_cond)
    print(f"Loaded {n_init} points ({n_init-1} from scan)")

    # ── Quadratic prior from conditional scan data ───────────────
    # Fit y ≈ α·dm² + β·dw² + γ·dm·dw to conditional points NEAR minimum
    # (exclude large ΔNLL — non-quadratic far from minimum)
    fit_ok = False
    cond_idx = np.where(is_cond_arr)[0]
    # Only use points with y_train < 20 (≈ 2× the 3σ contour level)
    near_min = np.abs(y_arr[cond_idx]) < 20.0
    fit_idx = cond_idx[near_min]
    if len(fit_idx) >= 4:
        dm = X_arr[fit_idx, 0] - v0_1
        dw = X_arr[fit_idx, 1] - v0_2
        A = np.column_stack([dm**2, dm*dw, dw**2])
        y_fit = y_arr[fit_idx]
        coeffs, res, rank, sv = np.linalg.lstsq(A, y_fit, rcond=None)
        if rank == 3 and np.all(np.isfinite(coeffs)):
            alpha, gamma, beta = coeffs
            # Hessian-like matrix for ΔNLL_cond = 2*y:
            # H = [[4α, 2γ], [2γ, 4β]]
            H_fit = np.array([[4*alpha, 2*gamma], [2*gamma, 4*beta]])
            H_fit = (H_fit + H_fit.T) / 2
            if np.linalg.cond(H_fit) < 1e12:
                cond_cov_fit = np.linalg.inv(H_fit)
                cov22 = cond_cov_fit
                fit_ok = True
                def quad_mean(v1, v2):
                    d1, d2 = v1 - v0_1, v2 - v0_2
                    return alpha*d1**2 + gamma*d1*d2 + beta*d2**2
                print(f"  Fit conditional: α={alpha:.4f}, γ={gamma:.4f}, β={beta:.4f}")

    if not fit_ok:
        # Fallback: conditional Hessian from saved fit
        if hess_inv_full is not None:
            H_full = np.linalg.inv(np.asarray(hess_inv_full))
            H_pp = H_full[np.ix_([idx1, idx2], [idx1, idx2])]
            H_pp = (H_pp + H_pp.T) / 2
            cond_cov = np.linalg.inv(H_pp)
            J1 = fitter.cm.bounds[args.param1].grad(0) if args.param1 in fitter.cm.bounds else 1.0
            J2 = fitter.cm.bounds[args.param2].grad(0) if args.param2 in fitter.cm.bounds else 1.0
            cond_cov_phys = cond_cov.copy()
            cond_cov_phys[0] *= J1; cond_cov_phys[:, 0] *= J1
            cond_cov_phys[1] *= J2; cond_cov_phys[:, 1] *= J2
            cov22 = cond_cov_phys
        else:
            err1 = float(err_dict.get(args.param1, abs(v0_1)*0.05))
            err2 = float(err_dict.get(args.param2, abs(v0_2)*0.05))
            cov22 = np.array([[err1**2, 0], [0, err2**2]])
        quad_mean = quadratic_from_hessian(cov22, v0_1, v0_2)

    H22 = np.linalg.inv(cov22)
    ls = 1.0 / np.sqrt(np.maximum(np.linalg.eigvalsh(H22), 1e-10))
    print(f"  Length scales: mass={ls[0]:.6f} width={ls[1]:.6f}")

    # Grid from conditional fit (5σ range centered on best-fit)
    err1 = np.sqrt(cov22[0, 0])
    err2 = np.sqrt(cov22[1, 1])
    n_sigma = 5.0
    lo1, hi1 = v0_1 - n_sigma * err1, v0_1 + n_sigma * err1
    lo2, hi2 = v0_2 - n_sigma * err2, v0_2 + n_sigma * err2
    print(f"  Grid: {args.param1}∈[{lo1:.4f},{hi1:.4f}], {args.param2}∈[{lo2:.4f},{hi2:.4f}]")

    # ── GP grid ───────────────────────────────────────────────────
    g1 = np.linspace(lo1, hi1, args.grid)
    g2 = np.linspace(lo2, hi2, args.grid)
    G1, G2 = np.meshgrid(g1, g2)
    grid_pts = np.column_stack([G1.ravel(), G2.ravel()])

    # ── GP fit ────────────────────────────────────────────────────
    mask_near = np.abs(y_arr) < 20
    if np.any(mask_near):
        sigma_f = max(1.0, y_arr[mask_near].std())
    else:
        sigma_f = max(1.0, y_arr.std())
    gp = GP(length_scale=ls, sigma_f=sigma_f, sigma_n=0.01)
    y_resid = y_arr - np.array([quad_mean(x[0], x[1]) for x in X_arr])
    if args.mle:
        gp.optimize(X_arr, y_resid, sigma_n_vec=sn_arr)
        print(f"  MLE: ls={gp.length_scale[0]:.6f},{gp.length_scale[1]:.6f} sigma_f={gp.sigma_f:.4f}")
    else:
        gp.fit(X_arr, y_resid, sigma_n_vec=sn_arr)

    # ── Active learning ───────────────────────────────────────────
    # Start from max existing gp_* number to resume incremental profiling
    start_it = 0
    if os.path.isdir(pts_dir):
        for fn in os.listdir(pts_dir):
            if fn.startswith('gp_') and fn.endswith('.json'):
                try:
                    n = int(fn[3:6])
                    start_it = max(start_it, n + 1)
                except ValueError:
                    pass
    print(f"GP active learning (max {args.max_iter} iters, starting from {start_it})...")
    penalized = set()
    for it in range(start_it, start_it + args.max_iter):
        mu_r, std = gp.predict(grid_pts)
        mu = mu_r + np.array([quad_mean(p[0], p[1]) for p in grid_pts])
        std_g = std.reshape(args.grid, args.grid)
        acq = acquire_contour(mu, std)

        # Pick best, skipping penalized and existing points
        for _ in range(len(grid_pts)):
            best = int(np.argmax(acq))
            v1b, v2b = grid_pts[best]
            key = (round(v1b, 6), round(v2b, 6))
            if key in penalized:
                acq[best] = -1e10
                continue
            close = any(np.all(np.abs(X_arr - [v1b, v2b]) <
                               [0.01*(hi1-lo1), 0.01*(hi2-lo2)], axis=1))
            if not close:
                break
            acq[best] = -1e10
        else:
            print("  No valid points left"); break

        # Predicted ΔNLL and uncertainty from GP (for the selected point)
        pred_idx = np.argmin(np.abs(grid_pts[:, 0] - v1b) + np.abs(grid_pts[:, 1] - v2b))
        pred_dn = mu[pred_idx]
        pred_std = std[pred_idx]
        print(f"  Selected: ({v1b:.6f}, {v2b:.6f}) ΔNLL≈{pred_dn:.2f} ± {pred_std:.2f}", flush=True)
        nll, result = run_profiled_fit(fitter, args.param1, args.param2,
                                        v1b, v2b, x_best, free_names,
                                        maxiter=args.fit_maxiter,
                                        hess_inv_full=hess_inv_full)
        if not np.isfinite(nll):
            print(f"  {it+1}: ({v1b:.4f}, {v2b:.4f}) FAILED", flush=True)
            penalized.add(key)
            continue

        dn = nll - nll0
        print(f"  {it+1}: fitted ΔNLL={dn:.2f}  (GP predicted ΔNLL={pred_dn:.2f} ± {pred_std:.2f})", flush=True)
        fn = os.path.join(pts_dir, f"gp_{it:03d}.json")
        fitter.save_params(result, fn)

        X_arr = np.vstack([X_arr, [v1b, v2b]])
        y_arr = np.hstack([y_arr, dn])
        sn_arr = np.hstack([sn_arr, ftol])
        y_resid = y_arr - np.array([quad_mean(x[0], x[1]) for x in X_arr])
        if args.mle:
            gp.optimize(X_arr, y_resid, sigma_n_vec=sn_arr)
        else:
            gp.fit(X_arr, y_resid, sigma_n_vec=sn_arr)

        max_s = std_g.max()
        print(f"  {it+1}: ({v1b:.4f}, {v2b:.4f}) ΔNLL={dn:.2f}  max_std={max_s:.4f}")
        if max_s < args.sigma_thresh and it > 5:
            print("  Converged"); break

    # ── Save ──────────────────────────────────────────────────────
    mu_r, std_f = gp.predict(grid_pts)
    mu_f = mu_r + np.array([quad_mean(p[0], p[1]) for p in grid_pts])
    gp_path = base + "_gp.json"
    with open(gp_path, 'w') as f:
        json.dump({
            "param1": args.param1, "param2": args.param2,
            "grid": {"g1": g1.tolist(), "g2": g2.tolist(),
                     "mu": mu_f.reshape(args.grid, args.grid).tolist(),
                     "std": std_f.reshape(args.grid, args.grid).tolist()},
            "n_evaluations": len(X_arr),
        }, f, indent=2)
    print(f"\nGP surface: {gp_path}")


if __name__ == "__main__":
    main()
