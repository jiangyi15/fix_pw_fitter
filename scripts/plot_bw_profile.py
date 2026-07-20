#!/usr/bin/env python3
"""Plot ΔNLL contour map of BW mass/width from profiling results.

Uses GP trained on cond_*/gp_* points instead of Hessian covariance.

Usage:
    python scripts/plot_bw_profile.py profile_rhoA_points/ rhoA \\
        --fit-result results.json -o bw_profile.pdf
    python scripts/plot_bw_profile.py profile_rhoA_points/ rhoA \\
        --fit-result results.json --ref pdg.csv --gp-only
"""

import sys, os, json, argparse, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Ellipse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ampfit import Fitter
from ampfit.gp import GP


def main():
    ap = argparse.ArgumentParser(description="2D BW contour from profiling results")
    ap.add_argument("points_dir", help="Directory with profiling JSONs (cond_*, gp_*)")
    ap.add_argument("particle", help="Particle name, e.g. rhoA")
    ap.add_argument("--fit-result", required=True,
                    help="Fit result JSON (for best-fit values)")
    ap.add_argument("--config", default="config_amp.yml",
                    help="Fitter config YAML")
    ap.add_argument("--backend", default="cuda_v3_sparse")
    ap.add_argument("-o", "--output", default=None, help="Output file")
    ap.add_argument("--grid", type=int, default=500, help="GP evaluation grid")
    ap.add_argument("--mle", action="store_true",
                    help="Optimize GP hyperparameters via MLE")
    ap.add_argument("--gp-only", action="store_true",
                    help="Use only gp_* points for GP training")
    ap.add_argument("--ref", default=None,
                    help="CSV with reference values (MeV): "
                         "name,mass,mass_err,width,width_err")
    args = ap.parse_args()

    # ── Load fitter and get BW params ─────────────────────────────
    fitter = Fitter(args.config, backend=args.backend)
    cp = os.path.splitext(args.fit_result)[0] + "_constraints.json"
    if os.path.exists(cp):
        fitter.load_constraints(cp)
    r = fitter.load_results(args.fit_result)
    with open(args.fit_result) as f:
        fit_data = json.load(f)
    nll0 = fit_data.get("status", {}).get("NLL", float('nan'))

    # Best-fit BW values from resolved params (no Hessian needed)
    _, resolved_best = fitter._build_params(r.x)
    model = None
    for chain in fitter.config.full_decay.chains:
        for decay in chain.decays[1:]:
            if decay.core.name == args.particle:
                model = decay.core._model
                break
        if model is not None:
            break
    if model is None:
        raise ValueError(f"Particle '{args.particle}' not found in config")
    bw = model.get_bw_params(resolved_best)
    m0 = bw["mass_bw"]
    w0 = bw["width_bw"]
    print(f"  {args.particle}: m={m0*1000:.2f} MeV, Γ={w0*1000:.2f} MeV")

    # Profiling param names (raw parameters used in scan)
    # Get them from the particle model's mass/width names
    free_names = fitter.free_param_names()
    mass_name = f"{args.particle}_mass"
    width_name = f"{args.particle}_width"
    # Find indices in free_names
    try:
        idx1 = free_names.index(mass_name)
        idx2 = free_names.index(width_name)
    except ValueError:
        # Fallback: try generic names
        mass_name = next((n for n in free_names if n.endswith('_mass')), None)
        width_name = next((n for n in free_names if n.endswith('_width')), None)
        idx1 = free_names.index(mass_name) if mass_name else None
        idx2 = free_names.index(width_name) if width_name else None
    if idx1 is None or idx2 is None:
        raise ValueError(f"Could not find profiling params for '{args.particle}'")

    # Full physical dict at best-fit (from fit result JSON)
    full_phys = fit_data.get("value", {}).copy()

    # Convert to MeV for plotting
    scale = 1000.0
    m0_mev = m0 * scale
    w0_mev = w0 * scale

    # ── Load profile points and convert to BW space ───────────────
    pts_files = sorted(glob.glob(os.path.join(args.points_dir, "*.json")))
    if not pts_files:
        print(f"No JSON files found in {args.points_dir}")
        sys.exit(1)

    bw_list = []  # (mass_bw, width_bw) for each point
    dn_list = []
    pt_type = []
    for fp in pts_files:
        with open(fp) as f:
            dat = json.load(f)
        v1 = dat.get("value", {}).get(mass_name)
        v2 = dat.get("value", {}).get(width_name)
        nll = dat.get("status", {}).get("NLL")
        if v1 is None or v2 is None or not np.isfinite(nll):
            continue
        # Build full physical dict at this point
        phys = full_phys.copy()
        phys[mass_name] = float(v1)
        phys[width_name] = float(v2)
        # Convert to opt space and get resolved params
        x_pt = fitter.values_from_dict({"value": phys})
        _, resolved = fitter._build_params(x_pt)
        # Get BW params at this point
        bw_pt = model.get_bw_params(resolved)
        bw_list.append([bw_pt["mass_bw"], bw_pt["width_bw"]])
        dn_list.append(float(nll) - nll0)
        fn_base = os.path.basename(fp)
        pt_type.append('c' if fn_base.startswith('cond') else 'p')

    if not bw_list:
        print("No valid points loaded")
        sys.exit(1)

    bw_arr = np.array(bw_list)  # (N, 2) with (mass_bw, width_bw) in GeV
    dn_arr = np.array(dn_list)
    pt_type_arr = np.array(pt_type)
    valid = np.isfinite(dn_arr)
    print(f"  Loaded {np.sum(valid)} points in BW space")

    # ── Quadratic prior from conditional scan data ───────────────
    fit_ok = False
    is_cond = pt_type_arr == 'c'
    cond_idx = np.where(is_cond & valid)[0]
    near_min = np.abs(dn_arr[cond_idx]) < 20.0
    fit_idx = cond_idx[near_min]
    if len(fit_idx) >= 4:
        dm = bw_arr[fit_idx, 0] - m0
        dw = bw_arr[fit_idx, 1] - w0
        A = np.column_stack([dm**2, dm*dw, dw**2])
        y_fit = dn_arr[fit_idx]
        coeffs, res, rank, sv = np.linalg.lstsq(A, y_fit, rcond=None)
        if rank == 3 and np.all(np.isfinite(coeffs)):
            alpha, gamma, beta = coeffs
            H_fit = np.array([[4*alpha, 2*gamma], [2*gamma, 4*beta]])
            H_fit = (H_fit + H_fit.T) / 2
            if np.linalg.cond(H_fit) < 1e12:
                cov22 = np.linalg.inv(H_fit)
                quad_mean = lambda v1, v2, a=alpha, g=gamma, b=beta: a*(v1-m0)**2 + g*(v1-m0)*(v2-w0) + b*(v2-w0)**2
                fit_ok = True
                print(f"  Fit conditional: α={alpha:.4f}, γ={gamma:.4f}, β={beta:.4f}")

    if not fit_ok:
        err1 = float(fit_data.get("error", {}).get(mass_name, abs(m0)*0.05))
        err2 = float(fit_data.get("error", {}).get(width_name, abs(w0)*0.05))
        cov22 = np.array([[err1**2, 0], [0, err2**2]])
        quad_mean = lambda v1, v2: 0.5 * (v1-m0)**2 / cov22[0,0] + 0.5 * (v2-w0)**2 / cov22[1,1]

    H22 = np.linalg.inv(cov22)
    ls = 1.0 / np.sqrt(np.maximum(np.linalg.eigvalsh(H22), 1e-10))

    # ── GP training ──────────────────────────────────────────────
    X_train = np.vstack([[m0, w0], bw_arr[valid, :2]])
    y_train = np.concatenate([[0.0], dn_arr[valid]])
    ftol = 1e-5
    noises = [ftol]
    for v in np.where(valid)[0]:
        noises.append(ftol if pt_type_arr[v] == 'p' else abs(dn_arr[v]))
    sigma_n_vec = np.array(noises)

    if args.gp_only:
        keep = np.concatenate([[True], pt_type_arr[valid] == 'p'])
        X_train = X_train[keep]
        y_train = y_train[keep]
        sigma_n_vec = sigma_n_vec[keep]

    mask_near = np.abs(y_train) < 20
    sigma_f = max(1.0, y_train[mask_near].std()) if np.any(mask_near) else max(1.0, y_train.std())

    gp = GP(length_scale=ls, sigma_f=sigma_f, sigma_n=0.01)
    y_resid = y_train - np.array([quad_mean(x[0], x[1]) for x in X_train])
    if args.mle:
        gp.optimize(X_train, y_resid, sigma_n_vec=sigma_n_vec)
        print(f"  MLE: ls={gp.length_scale[0]:.6f},{gp.length_scale[1]:.6f} sigma_f={gp.sigma_f:.4f}")
    else:
        gp.fit(X_train, y_resid, sigma_n_vec=sigma_n_vec)

    # ── Load reference points for grid expansion ────────────────
    ref_m_vals, ref_w_vals = [], []
    if args.ref is not None and os.path.exists(args.ref):
        import csv
        with open(args.ref) as _rf:
            for row in csv.DictReader(_rf):
                if row.get("mass", "").strip():
                    ref_m_vals.append(float(row["mass"]))
                if row.get("width", "").strip():
                    ref_w_vals.append(float(row["width"]))

    # ── GP prediction grid (in MeV) ──────────────────────────────
    err1 = np.sqrt(cov22[0, 0]) * scale
    err2 = np.sqrt(cov22[1, 1]) * scale
    n_sigma = 5.0
    m_lo, m_hi = m0_mev - n_sigma * err1, m0_mev + n_sigma * err1
    w_lo, w_hi = w0_mev - n_sigma * err2, w0_mev + n_sigma * err2
    # Expand grid to cover reference points
    if ref_m_vals:
        m_lo = min(m_lo, min(ref_m_vals) - 5)
        m_hi = max(m_hi, max(ref_m_vals) + 5)
    if ref_w_vals:
        w_lo = min(w_lo, min(ref_w_vals) - 5)
        w_hi = max(w_hi, max(ref_w_vals) + 5)
    ng = args.grid
    m_grid = np.linspace(m_lo, m_hi, ng)
    w_grid = np.linspace(w_lo, w_hi, ng)
    MM, WW = np.meshgrid(m_grid, w_grid)
    # Predict on grid (convert MeV → original units for GP)
    g1 = m_grid / scale
    g2 = w_grid / scale
    G1, G2 = np.meshgrid(g1, g2)
    grid_pts = np.column_stack([G1.ravel(), G2.ravel()])
    mu_r, std = gp.predict(grid_pts)
    mu = mu_r + np.array([quad_mean(p[0], p[1]) for p in grid_pts])
    dNLL = mu.reshape(ng, ng)

    # ── Profiled 1σ uncertainties ───────────────────────────────
    # dNLL axes: axis 0 = width, axis 1 = mass (from meshgrid)
    prof_mass = np.min(dNLL, axis=0)   # min over width for each mass
    prof_width = np.min(dNLL, axis=1)  # min over mass for each width
    m1 = m_grid[prof_mass < 0.5]
    w1 = w_grid[prof_width < 0.5]
    sig_m_mev = 0.0
    sig_w_mev = 0.0
    m_err_lo = m_err_hi = 0.0
    w_err_lo = w_err_hi = 0.0
    if len(m1) >= 2:
        lo, hi = m1[0], m1[-1]
        step_m = (m_hi - m_lo) / (ng - 1)
        m_err_lo = m0_mev - lo + step_m
        m_err_hi = hi - m0_mev + step_m
        sig_m_mev = (m_err_lo + m_err_hi) / 2
        print(f"  Profiled 1σ: mass = {m0_mev:.3f} + {m_err_hi:.3f} - {m_err_lo:.3f} MeV  (step={step_m:.3f})")
    if len(w1) >= 2:
        lo, hi = w1[0], w1[-1]
        step_w = (w_hi - w_lo) / (ng - 1)
        w_err_lo = w0_mev - lo + step_w
        w_err_hi = hi - w0_mev + step_w
        sig_w_mev = (w_err_lo + w_err_hi) / 2
        print(f"  Profiled 1σ: width = {w0_mev:.3f} + {w_err_hi:.3f} - {w_err_lo:.3f} MeV  (step={step_w:.3f})")

    # ── Plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))

    sigmas = [1, 2, 3, 4, 5]
    C_lev = [s**2 / 2 for s in sigmas]

    # 1σ error bars at center (MeV)
    if sig_m_mev > 0 and sig_w_mev > 0:
        label = f'$m={m0_mev:.2f}^{{+{m_err_hi:.2f}}}_{{-{m_err_lo:.2f}}}$ MeV  $\\Gamma={w0_mev:.2f}^{{+{w_err_hi:.2f}}}_{{-{w_err_lo:.2f}}}$ MeV'
        ax.errorbar(m0_mev, w0_mev,
                    xerr=np.array([[m_err_lo], [m_err_hi]]),
                    yerr=np.array([[w_err_lo], [w_err_hi]]),
                    fmt='k+', ms=8, capsize=3, lw=1.5, label=label)

    # Axis limits (MeV)
    all_x = [m0_mev]; all_y = [w0_mev]
    all_x_w = [max(m_err_lo, m_err_hi)] if m_err_hi > 0 else [sig_m_mev]
    all_y_w = [max(w_err_lo, w_err_hi)] if w_err_hi > 0 else [sig_w_mev]

    # Reference values from CSV (handles symmetric and asymmetric errors)
    if args.ref is not None and os.path.exists(args.ref):
        import csv
        _ref_colors = ['r', 'b', 'g', 'orange', 'purple', 'c', 'm', 'y']
        _ref_markers = ['D', 's', '^', 'v', 'o', 'p', 'h', '*']
        with open(args.ref) as _rf:
            for i, row in enumerate(csv.DictReader(_rf)):
                r_name = row.get("name", "").strip()
                r_m = float(row["mass"]) if row.get("mass", "").strip() else None
                r_w = float(row["width"]) if row.get("width", "").strip() else None
                if r_m is None and r_w is None:
                    continue
                # Asymmetric or symmetric mass errors
                melo = row.get("mass_err_lo", "").strip()
                mehi = row.get("mass_err_hi", "").strip()
                if melo and mehi:
                    r_me = (float(melo), float(mehi))
                else:
                    r_me = float(row.get("mass_err", 0)) if row.get("mass_err", "").strip() else 0
                # Asymmetric or symmetric width errors
                welo = row.get("width_err_lo", "").strip()
                wehi = row.get("width_err_hi", "").strip()
                if welo and wehi:
                    r_we = (float(welo), float(wehi))
                else:
                    r_we = float(row.get("width_err", 0)) if row.get("width_err", "").strip() else 0
                # Max error for axis limits
                r_me_max = max(r_me) if isinstance(r_me, tuple) else float(r_me) if r_me else 0
                r_we_max = max(r_we) if isinstance(r_we, tuple) else float(r_we) if r_we else 0
                if r_m is not None:
                    all_x.append(r_m); all_x_w.append(r_me_max)
                if r_w is not None:
                    all_y.append(r_w); all_y_w.append(r_we_max)
                c = _ref_colors[i % len(_ref_colors)]
                m = _ref_markers[i % len(_ref_markers)]
                # Format errors for matplotlib (handle asymmetric (lo,hi) tuples)
                def _fmt_err(e):
                    if e is None:
                        return None
                    if isinstance(e, tuple):
                        return np.array(e).reshape(2, 1)
                    return e
                ax.errorbar(r_m if r_m else 0, r_w if r_w else 0,
                            xerr=_fmt_err(r_me) if r_m else None,
                            yerr=_fmt_err(r_we) if r_w else None,
                            fmt='none', color=c, marker=m, ms=5,
                            capsize=3, lw=1.2, label=r_name,
                            markeredgecolor=c, markerfacecolor=c)

    # Axis limits match the grid range
    m_lo, m_hi = m_grid[0], m_grid[-1]
    w_lo, w_hi = w_grid[0], w_grid[-1]

    # GP ΔNLL contour
    if np.any(np.isfinite(dNLL)):
        vmax = np.nanmax(dNLL)
        lev = [l for l in C_lev if l <= vmax * 1.05]
        if lev:
            cf = ax.contourf(MM, WW, dNLL, levels=[0] + lev, cmap='viridis', alpha=0.7)
            cs = ax.contour(MM, WW, dNLL, levels=lev, colors='k', linewidths=1.0)
            fmt = {l: rf'${s}\sigma$' for l, s in zip(lev, sigmas[:len(lev)])}
            ax.clabel(cs, fmt=fmt, fontsize=8)
            cbar = fig.colorbar(cf, ax=ax, label=r'$\Delta$NLL')
            cbar.set_ticks([0] + lev)
            cbar.set_ticklabels(['0'] + [rf'$({s})^2/2$' for s in sigmas[:len(lev)]])

    ax.set_xlim(m_lo, m_hi)
    ax.set_ylim(w_lo, w_hi)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.set_xlabel(f"$m_{{\\mathrm{{BW}}}}$ (MeV)")
    ax.set_ylabel(r"$\Gamma_{\mathrm{BW}}$ (MeV)")
    ax.set_title("BW profile — GP surface")
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = args.output or f"{os.path.basename(args.points_dir.rstrip('/'))}_bw.pdf"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
