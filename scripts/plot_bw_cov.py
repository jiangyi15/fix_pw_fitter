#!/usr/bin/env python3
"""Plot ΔNLL contour map of BW mass/width with chi²/2 confidence levels.

Usage:
    python scripts/plot_bw_cov.py fit_output8/results.json a1(1260)p
"""
import sys, os, json, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Ellipse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description="2D contour plot for BW mass/width")
    ap.add_argument("results_json")
    ap.add_argument("particle", help="Particle name, e.g. a1(1260)p")
    ap.add_argument("--config", default="config_amp.yml")
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--scan", type=int, default=100,
                    help="Grid resolution for contour")
    ap.add_argument("-o", "--output", default=None, help="Save to file")
    ap.add_argument("--ref", default=None,
                    help="CSV with reference values (MeV). Columns: "
                         "name,mass,mass_err,width,width_err (symmetric) or "
                         "name,mass,mass_err_lo,mass_err_hi,width,width_err_lo,width_err_hi (asymmetric)")
    ap.add_argument("--format", default="png",
                    help="Output format: png, pdf, svg (default: png)")
    args = ap.parse_args()

    from ampfit import Fitter
    from ampfit.utils import fmt_meas, fmt_particle
    f = Fitter(args.config, backend=args.backend)
    cp = os.path.splitext(args.results_json)[0] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)
    r = f.load_results(args.results_json)

    # Load NLL from JSON status
    try:
        with open(args.results_json) as _f:
            _rj = json.load(_f)
        _nll = _rj.get('status', {}).get('NLL', None)
    except Exception:
        _nll = None

    bw = f.get_bw_params(args.particle, r)
    m0, w0 = bw["mass_bw"] * 1000, bw["width_bw"] * 1000
    cov = np.array([[bw["mass_bw_err"]**2, bw["mass_width_cov"]],
                    [bw["mass_width_cov"], bw["width_bw_err"]**2]]) * 1e6
    sig_m = bw["mass_bw_err"] * 1000
    sig_w = bw["width_bw_err"] * 1000
    rho = bw["mass_width_cov"] / (bw["mass_bw_err"] * bw["width_bw_err"]) if sig_m > 0 and sig_w > 0 else 0.0

    m_str = fmt_meas(m0, sig_m)
    w_str = fmt_meas(w0, sig_w)
    pname = fmt_particle(args.particle, full=False)
    print(f"  {pname}: m={m_str.strip('$')} MeV,  Γ={w_str.strip('$')} MeV,  ρ={rho:.4f}")

    # ── Gaussian ΔNLL from covariance matrix ─────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    sigmas = [1, 2, 3, 4, 5]
    C_lev = [s**2 / 2 for s in sigmas]  # ΔNLL = a²/2

    # ── 1σ error bars at center ──────────────────────────────────
    if sig_m > 0 and sig_w > 0:
        m_str = fmt_meas(m0, sig_m)
        w_str = fmt_meas(w0, sig_w)
        label = f'$m={m_str.strip("$")}$ MeV  $\\Gamma={w_str.strip("$")}$ MeV'
        if _nll is not None:
            label += f'\nNLL={_nll:.2f}'
        ax.errorbar(m0, w0, xerr=sig_m, yerr=sig_w,
                    fmt='k+', ms=8, capsize=3, lw=1.5, label=label)

    # ── Collect all points for axis limits ────────────────────────
    all_x = [m0]
    all_y = [w0]
    all_x_w = [sig_m]
    all_y_w = [sig_w]

    # ── Reference values from CSV (MeV) ──
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
                # Asymmetric or symmetric errors
                def _read_err(col, col_lo, col_hi):
                    lo = row.get(col_lo, None)
                    hi = row.get(col_hi, None)
                    if lo is not None and hi is not None:
                        return (float(lo), float(hi))
                    v = row.get(col, "").strip()
                    return float(v) if v else 0
                r_me = _read_err("mass_err", "mass_err_lo", "mass_err_hi")
                r_we = _read_err("width_err", "width_err_lo", "width_err_hi")
                # Use max error for axis limit calculation
                r_me_max = max(r_me) if isinstance(r_me, tuple) else r_me
                r_we_max = max(r_we) if isinstance(r_we, tuple) else r_we
                if r_m is not None:
                    all_x.append(r_m); all_x_w.append(r_me_max)
                if r_w is not None:
                    all_y.append(r_w); all_y_w.append(r_we_max)
                c = _ref_colors[i % len(_ref_colors)]
                m = _ref_markers[i % len(_ref_markers)]
                # Use separate color/marker args to support named colors
                ax.errorbar(r_m if r_m else 0, r_w if r_w else 0,
                            xerr=r_me if r_m else None,
                            yerr=r_we if r_w else None,
                            fmt='none', color=c, marker=m, ms=5,
                            capsize=3, lw=1.2, label=r_name,
                            markeredgecolor=c, markerfacecolor=c)

    # Set axis limits including all points
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_x_w = np.array(all_x_w)
    all_y_w = np.array(all_y_w)
    x_lo = (all_x - all_x_w * 1.5).min()
    x_hi = (all_x + all_x_w * 1.5).max()
    y_lo = (all_y - all_y_w * 1.5).min()
    y_hi = (all_y + all_y_w * 1.5).max()
    # Ensure at least ±5.5σ for the contour
    x_lo = min(x_lo, m0 - 5.5 * sig_m)
    x_hi = max(x_hi, m0 + 5.5 * sig_m)
    y_lo = min(y_lo, w0 - 5.5 * sig_w)
    y_hi = max(y_hi, w0 + 5.5 * sig_w)
    m_lo, m_hi = x_lo, x_hi
    w_lo, w_hi = y_lo, y_hi
    ng = max(args.scan, 2)
    m_grid = np.linspace(m_lo, m_hi, ng)
    w_grid = np.linspace(w_lo, w_hi, ng)

    # ── Gaussian ΔNLL contour ────────────────────────────────────
    MM, WW = np.meshgrid(m_grid, w_grid)
    dm = MM - m0; dw = WW - w0
    if sig_m > 0 and sig_w > 0:
        det = cov[0, 0] * cov[1, 1] - cov[0, 1]**2
        inv = np.array([[cov[1, 1], -cov[0, 1]], [-cov[0, 1], cov[0, 0]]]) / det
        dNLL = 0.5 * (inv[0, 0] * dm**2 + 2 * inv[0, 1] * dm * dw + inv[1, 1] * dw**2)
    else:
        dNLL = np.full_like(MM, np.nan)

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
    ax.set_xlabel(rf"$m_{{\mathrm{{BW}}}}$ [${pname}$] (MeV)")
    ax.set_ylabel(rf"$\Gamma_{{\mathrm{{BW}}}}$ [${pname}$] (MeV)")
    ax.set_title(f"${pname}$ — BW covariance ($\\rho$={rho:.4f})")
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = args.output or f"bw_cov_{args.particle}.{args.format}"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
