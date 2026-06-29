#!/usr/bin/env python3
"""Plot 2D Gaussian contours of BW mass/width covariance.

Usage:
    python scripts/plot_bw_cov.py fit_output8/results.json a1(1260)p
    python scripts/plot_bw_cov.py fit_output8/results.json a1(1260)p --scan 20
"""

import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description="2D covariance plot for BW mass/width")
    ap.add_argument("results_json")
    ap.add_argument("particle", help="Particle name, e.g. a1(1260)p")
    ap.add_argument("--config", default="config_amp.yml")
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--scan", type=int, default=0,
                    help="Grid size for NLL scan (0 = Gaussian only)")
    ap.add_argument("--sigma", type=float, nargs=3, default=[1.0, 2.0, 3.0],
                    help="Sigma levels to plot")
    ap.add_argument("-o", "--output", default=None, help="Save to file")
    args = ap.parse_args()

    from ampfit import Fitter
    f = Fitter(args.config, backend=args.backend)
    cp = os.path.splitext(args.results_json)[0] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)
    r = f.load_results(args.results_json)

    bw = f.get_bw_params(args.particle, r)
    m0, w0 = bw["mass_bw"], bw["width_bw"]
    cov = np.array([[bw["mass_bw_err"]**2, bw["mass_width_cov"]],
                    [bw["mass_width_cov"], bw["width_bw_err"]**2]])
    sig_m = bw["mass_bw_err"]
    sig_w = bw["width_bw_err"]
    rho = bw["mass_width_cov"] / (sig_m * sig_w) if sig_m > 0 and sig_w > 0 else 0.0

    print(f"  {args.particle}: mass={m0:.6f}±{sig_m:.6f}, width={w0:.6f}±{sig_w:.6f}, ρ={rho:.4f}")

    # ── Gaussian contours ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    # 2D Gaussian ellipse: angle from covariance
    if sig_m > 0 and sig_w > 0:
        ells = []
        for n_sig in args.sigma:
            # Ellipse axes = n_sig * sqrt(eigenvalues)
            vals, vecs = np.linalg.eigh(cov)
            theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            for n in [n_sig]:
                width = 2 * n * np.sqrt(vals[1])
                height = 2 * n * np.sqrt(vals[0])
                ell = Ellipse(xy=(m0, w0), width=width, height=height,
                              angle=theta, edgecolor=f'C0', fc='None',
                              lw=1.5, ls='-',
                              label=f'{n:.0f}σ Gaussian')
                ax.add_patch(ell)
    else:
        ax.plot(m0, w0, 'kx', ms=8, label='best fit')

    # ── Optional NLL scan grid ────────────────────────────────────
    if args.scan > 0:
        # Build param dict and evaluate NLL on a grid
        def nll_at(m_val, w_val):
            # Build a resolved dict from scratch
            _, resolved, _, _ = f._build_params(r.x)
            resolved[f"{args.particle}_mass"] = float(m_val)
            resolved[f"{args.particle}_width"] = float(w_val)
            # Rebuild params from resolved (approximate: use raw x and override)
            # Direct approach: modify x and re-run get_nll
            # This requires inverse mapping — compute numerically
            x_try = r.x.copy()
            names = f._var_registry.flat_names
            mass_key = f"{args.particle}_mass"
            width_key = f"{args.particle}_width"
            for i, n in enumerate(names):
                if n == mass_key:
                    x_try[i] = m_val
                elif n == width_key:
                    x_try[i] = w_val
            nll, _ = f.get_nll(x_try)
            return nll

        m_grid = np.linspace(m0 - 3.5 * sig_m, m0 + 3.5 * sig_m, args.scan)
        w_grid = np.linspace(w0 - 3.5 * sig_w, w0 + 3.5 * sig_w, args.scan)
        MM, WW = np.meshgrid(m_grid, w_grid)
        NLL = np.full_like(MM, np.nan)
        nll0 = None
        for i in range(args.scan):
            for j in range(args.scan):
                try:
                    n = nll_at(MM[i, j], WW[i, j])
                    NLL[i, j] = n
                    if nll0 is None or n < nll0:
                        nll0 = n
                except Exception:
                    pass
        if nll0 is not None and np.any(np.isfinite(NLL)):
            dNLL = NLL - nll0
            levels = [0.5 * s**2 for s in args.sigma]  # ΔNLL = ½σ²
            # Only plot levels within range
            vmax = np.nanmax(dNLL)
            levels_ok = [l for l in levels if l <= vmax * 1.1]
            if levels_ok:
                cs = ax.contour(MM, WW, dNLL, levels=levels_ok,
                                colors='C1', linestyles='--', linewidths=1.5)
                # Label with ΔNLL value
                fmt = {l: f'{l:.1f}' for l in levels_ok}
                ax.clabel(cs, fmt=fmt, fontsize=8)
                # Add legend entry
                ax.plot([], [], '--', color='C1', lw=1.5, label='NLL scan')

    ax.set_xlabel(rf"$m_{{\rm BW}}$ [{args.particle}] (GeV)")
    ax.set_ylabel(rf"$\Gamma_{{\rm BW}}$ [{args.particle}] (GeV)")
    ax.set_title(f"{args.particle} — BW covariance (ρ={rho:.4f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = args.output or f"bw_cov_{args.particle}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
