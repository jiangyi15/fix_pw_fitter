#!/usr/bin/env python3
"""Plot 2-parameter likelihood profile using GP trained from point files.

Usage:
    python scripts/plot_profile_2d.py profile_rhoA_points/ \\
        --param1 rhoA_mass --param2 rhoA_width \\
        --fit-result results.json --output profile.pdf
"""

import sys, os, json, argparse, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ampfit.gp import GP


def main():
    ap = argparse.ArgumentParser(description="Plot 2D likelihood profile")
    ap.add_argument("points_dir", help="Directory with profiling result JSONs")
    ap.add_argument("--param1", required=True)
    ap.add_argument("--param2", required=True)
    ap.add_argument("--fit-result", required=True,
                    help="Fit result JSON (for best-fit values)")
    ap.add_argument("--output", "-o", default=None, help="Output file")
    ap.add_argument("--grid", type=int, default=200, help="GP evaluation grid")
    ap.add_argument("--gp-only", action="store_true",
                    help="Use only gp_* (profiled) points, ignore cond_* scan data")
    ap.add_argument("--mle", action="store_true",
                    help="Optimize GP hyperparameters via MLE")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    # ── Load best-fit values ──────────────────────────────────────
    with open(args.fit_result) as f:
        fit_data = json.load(f)
    v0_1 = float(fit_data.get("value", {}).get(args.param1, 0))
    v0_2 = float(fit_data.get("value", {}).get(args.param2, 0))
    nll0 = fit_data.get("status", {}).get("NLL", float('nan'))

    # ── Load profile points ───────────────────────────────────────
    pts_files = sorted(glob.glob(os.path.join(args.points_dir, "*.json")))
    if not pts_files:
        print(f"No JSON files found in {args.points_dir}")
        sys.exit(1)

    v1_list, v2_list, dn_list = [], [], []
    pt_type = []  # 'c' for cond (scan), 'p' for gp (profiled)
    for fp in pts_files:
        with open(fp) as f:
            dat = json.load(f)
        v1 = dat.get("value", {}).get(args.param1)
        v2 = dat.get("value", {}).get(args.param2)
        nll = dat.get("status", {}).get("NLL")
        if v1 is None or v2 is None or not np.isfinite(nll):
            continue
        v1_list.append(float(v1))
        v2_list.append(float(v2))
        dn_list.append(float(nll) - nll0)
        fn_base = os.path.basename(fp)
        pt_type.append('c' if fn_base.startswith('cond') else 'p')

    dn_arr = np.array(dn_list)
    ev = np.column_stack([v1_list, v2_list, dn_arr])
    pt_type_arr = np.array(pt_type)
    valid = np.isfinite(dn_arr)
    min_dn = np.nanmin(dn_arr[valid]) if np.any(valid) else 0
    print(f"  {len(ev)} points loaded")

    # ── Quadratic prior from conditional scan data ───────────────
    # (always uses cond_* data regardless of --gp-only)
    fit_ok = False
    is_cond = pt_type_arr == 'c'
    cond_idx = np.where(is_cond & valid)[0]
    near_min = np.abs(dn_arr[cond_idx]) < 20.0
    fit_idx = cond_idx[near_min]
    if len(fit_idx) >= 4:
            dm = ev[fit_idx, 0] - v0_1
            dw = ev[fit_idx, 1] - v0_2
            A = np.column_stack([dm**2, dm*dw, dw**2])
            y_fit = dn_arr[fit_idx]
            coeffs, res, rank, sv = np.linalg.lstsq(A, y_fit, rcond=None)
            if rank == 3 and np.all(np.isfinite(coeffs)):
                alpha, gamma, beta = coeffs
                H_fit = np.array([[4*alpha, 2*gamma], [2*gamma, 4*beta]])
                H_fit = (H_fit + H_fit.T) / 2
                if np.linalg.cond(H_fit) < 1e12:
                    cov22 = np.linalg.inv(H_fit)
                    quad_mean = lambda v1, v2, a=alpha, g=gamma, b=beta: a*(v1-v0_1)**2 + g*(v1-v0_1)*(v2-v0_2) + b*(v2-v0_2)**2
                    fit_ok = True
                    print(f"  Fit conditional: α={alpha:.4f}, γ={gamma:.4f}, β={beta:.4f}")

    if not fit_ok:
        # Fallback: profiled errors from fit result
        err1 = float(fit_data.get("error", {}).get(args.param1, abs(v0_1)*0.05))
        err2 = float(fit_data.get("error", {}).get(args.param2, abs(v0_2)*0.05))
        cov22 = np.array([[err1**2, 0], [0, err2**2]])
        quad_mean = lambda v1, v2: 0.5 * (v1-v0_1)**2 / cov22[0,0] + 0.5 * (v2-v0_2)**2 / cov22[1,1]

    H22 = np.linalg.inv(cov22)
    ls = 1.0 / np.sqrt(np.maximum(np.linalg.eigvalsh(H22), 1e-10))
    print(f"  Length scales: {args.param1}={ls[0]:.6f} {args.param2}={ls[1]:.6f}")

    # ── GP training ──────────────────────────────────────────────
    X_train = np.vstack([[v0_1, v0_2], ev[valid, :2]])
    y_train = np.concatenate([[0.0], dn_arr[valid]])
    ftol = 1e-5
    noises = [ftol]  # centre point: ΔNLL=0 exactly
    for v in np.where(valid)[0]:
        noises.append(ftol if pt_type_arr[v] == 'p' else abs(dn_arr[v]))
    sigma_n_vec = np.array(noises)

    # gp-only: exclude cond_* from GP training but keep for plot
    if args.gp_only:
        keep = np.concatenate([[True], pt_type_arr[valid] == 'p'])
        X_train = X_train[keep]
        y_train = y_train[keep]
        sigma_n_vec = sigma_n_vec[keep]
    # sigma_f from points near minimum (avoid outliers inflating it)
    mask_near = np.abs(y_train) < 20
    if np.any(mask_near):
        sigma_f = max(1.0, y_train[mask_near].std())
    else:
        sigma_f = max(1.0, y_train.std())
    gp = GP(length_scale=ls, sigma_f=sigma_f, sigma_n=0.01)
    y_resid = y_train - np.array([quad_mean(x[0], x[1]) for x in X_train])
    if args.mle:
        gp.optimize(X_train, y_resid, sigma_n_vec=sigma_n_vec)
        print(f"  MLE: ls={gp.length_scale[0]:.6f},{gp.length_scale[1]:.6f} sigma_f={gp.sigma_f:.4f}")
    else:
        gp.fit(X_train, y_resid, sigma_n_vec=sigma_n_vec)

    # ── GP prediction grid from conditional errors ──────────────
    err1 = np.sqrt(cov22[0, 0])
    err2 = np.sqrt(cov22[1, 1])
    n_sigma = 5.0
    lo1, hi1 = v0_1 - n_sigma * err1, v0_1 + n_sigma * err1
    lo2, hi2 = v0_2 - n_sigma * err2, v0_2 + n_sigma * err2
    g1 = np.linspace(lo1, hi1, args.grid)
    g2 = np.linspace(lo2, hi2, args.grid)
    G1, G2 = np.meshgrid(g1, g2)
    grid_pts = np.column_stack([G1.ravel(), G2.ravel()])
    mu_r, std = gp.predict(grid_pts)
    mu = mu_r + np.array([quad_mean(p[0], p[1]) for p in grid_pts])
    mu2d = mu.reshape(args.grid, args.grid)
    std2d = std.reshape(args.grid, args.grid)

    # ── Profiled 1σ uncertainties ─────────────────────────────────
    # mu2d axes: axis 0 = param2, axis 1 = param1 (from meshgrid)
    prof_mass = np.min(mu2d, axis=0)
    prof_width = np.min(mu2d, axis=1)
    m1 = g1[prof_mass < 0.5]
    m2 = g2[prof_width < 0.5]
    if len(m1) >= 2:
        lo, hi = m1[0], m1[-1]
        step = (hi1 - lo1) / (args.grid - 1)
        lo_err = v0_1 - lo + step
        hi_err = hi - v0_1 + step
        print(f"  Profiled 1σ: {args.param1} = {v0_1:.6f} + {hi_err:.6f} - {lo_err:.6f}")
    if len(m2) >= 2:
        lo, hi = m2[0], m2[-1]
        step = (hi2 - lo2) / (args.grid - 1)
        lo_err = v0_2 - lo + step
        hi_err = hi - v0_2 + step
        print(f"  Profiled 1σ: {args.param2} = {v0_2:.6f} + {hi_err:.6f} - {lo_err:.6f}")

    # ── Plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: GP mean (plot sqrt(2·ΔNLL) for even spacing, colorbar shows ΔNLL)
    ax = axes[0]
    vmax = max(12, np.nanmax(mu))  # keep vmax for scatter colorbar
    z2d = np.sqrt(np.maximum(2 * mu2d, 0))  # sqrt(2·ΔNLL)
    vmax_z = np.sqrt(2 * vmax)
    ks = list(range(1, 7))  # 1σ through 6σ
    z_levels = list(range(1, int(vmax_z) + 1))
    cf = ax.contourf(g1, g2, z2d, levels=[0] + z_levels, cmap='viridis', alpha=0.7)
    cb = plt.colorbar(cf, ax=ax, label='ΔNLL')
    # Colorbar ticks: show ΔNLL = k²/2 at position k
    cb.set_ticks(list(range(0, int(vmax_z) + 1)))
    cb.set_ticklabels(['0'] + [f'{k}²/2' for k in range(1, int(vmax_z) + 1)])
    # Contour lines at integer σ levels
    cs = ax.contour(g1, g2, z2d, levels=z_levels[:5],
                    colors='k', linewidths=1.5, linestyles='--')
    ax.clabel(cs, fmt={k: f'{k}σ' for k in z_levels[:5]}, fontsize=9)

    # Scatter with same color scale as contourf
    if not args.gp_only:
        mask_c = (pt_type_arr == 'c') & valid
        if np.any(mask_c):
            ax.scatter(ev[mask_c, 0], ev[mask_c, 1], c=ev[mask_c, 2],
                       s=40, cmap='viridis', vmin=0, vmax=vmax,
                       edgecolors='k', linewidths=0.5, label='Conditional', zorder=5)
    mask_p = (pt_type_arr == 'p') & valid
    if np.any(mask_p):
        ax.scatter(ev[mask_p, 0], ev[mask_p, 1], c=ev[mask_p, 2],
                   s=10, cmap='viridis', vmin=0, vmax=vmax,
                   edgecolors='none', label='Profiled', zorder=5)
    ax.scatter(v0_1, v0_2, c='red', s=120, marker='*', zorder=6,
               edgecolors='k', linewidths=1, label='Best fit')
    ax.set_xlabel(args.param1)
    ax.set_ylabel(args.param2)
    ax.set_title('GP ΔNLL')
    ax.set_xlim(lo1, hi1)
    ax.set_ylim(lo2, hi2)
    ax.legend()

    # Panel 2: GP uncertainty
    ax = axes[1]
    im = ax.imshow(std2d, origin='lower',
                   extent=[g1[0], g1[-1], g2[0], g2[-1]],
                   aspect='auto', cmap='Reds', alpha=0.8)
    plt.colorbar(im, ax=ax, label='GP std')
    for typ, s in ([('c', 20), ('p', 5)] if not args.gp_only else [('p', 5)]):
        mask = (pt_type_arr == typ) & valid
        if np.any(mask):
            ax.scatter(ev[mask, 0], ev[mask, 1], c=ev[mask, 2],
                       s=s, cmap='viridis', vmin=0, vmax=vmax, alpha=0.5)
    ax.scatter(v0_1, v0_2, c='red', s=80, marker='*', zorder=6,
               edgecolors='k', linewidths=1)
    ax.set_xlabel(args.param1)
    ax.set_ylabel(args.param2)
    ax.set_title('GP uncertainty')
    ax.set_xlim(lo1, hi1)
    ax.set_ylim(lo2, hi2)

    plt.tight_layout()
    out = args.output or f"{os.path.basename(args.points_dir.rstrip('/'))}_profile.pdf"
    plt.savefig(out, dpi=args.dpi)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
