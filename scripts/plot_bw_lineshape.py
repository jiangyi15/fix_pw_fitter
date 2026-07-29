#!/usr/bin/env python3
"""
Plot Breit-Wigner lineshapes Im(D)/|D|² where D = m₀² − m² − im₀Γ(m).

Γ(m) = Σ gᵢ · γᵢ(m) uses the actual gamma parameter values from the fit
(g0_phys entries), not just a simple width normalization.

Each resonance gets its own subplot in a grid.

Usage:
    python scripts/plot_bw_lineshape.py config_amp.yml -o bw_plot.pdf
    python scripts/plot_bw_lineshape.py config_amp.yml results.json -o bw_plot.pdf
    python scripts/plot_bw_lineshape.py config_amp.yml --resonances rhoA f0(500)
"""

import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(
        description="Plot BW lineshapes Im(D)/|D|²")
    ap.add_argument("config")
    ap.add_argument("results_json", nargs="?", default=None)
    ap.add_argument("--mass-lo", type=float, default=None,
                    help="Lower mass limit (default: 2*m_pi ≈ 0.28)")
    ap.add_argument("--mass-hi", type=float, default=None,
                    help="Upper mass limit (default: m_B - m_pi ≈ 5.14)")
    ap.add_argument("--n-points", type=int, default=2000)
    ap.add_argument("-o", "--output", default="bw_lineshape.pdf")
    ap.add_argument("--format", default=None,
                    help="Image format (png, pdf, svg, …). Inferred from --output extension if omitted.")
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--resonances", nargs="*", default=None)
    args = ap.parse_args()

    from ampfit import Fitter
    import matplotlib.pyplot as plt

    f = Fitter(args.config, backend=args.backend)

    if args.results_json:
        cp = os.path.splitext(args.results_json)[0] + "_constraints.json"
        if os.path.exists(cp):
            f.load_constraints(cp)
        r = f.load_results(args.results_json)
        x = r.x
    else:
        x = f.initial_values(seed=42)
    _, resolved = f.build_params(x)

    m_pi = 0.13957
    m_B = 5.279
    mass_lo = args.mass_lo if args.mass_lo is not None else 2 * m_pi
    mass_hi = args.mass_hi if args.mass_hi is not None else m_B - m_pi
    m_grid = np.linspace(mass_lo, mass_hi, args.n_points)

    seen = set()
    resonances = []
    for chain in f.config.full_decay.chains:
        for decay in chain.decays[1:]:
            model = decay.core._model
            mid = id(model)
            if mid in seen:
                continue
            seen.add(mid)
            name = decay.core.name
            disp = decay.core.display

            # Full amplitude + gamma components
            invD = model.amplitude_raw(m_grid, resolved)
            gamma_vals = np.asarray(model.gamma(m_grid), dtype=complex)
            gamma_names = list(model.get_gamma_name())
            g0_vals = [float(resolved[gn]) for gn in gamma_names]

            resonances.append({
                "name": name, "disp": disp, 
                "invD": invD,
                "gamma_fn": model.gamma,
                "gamma_names": gamma_names,
                "gamma_vals": gamma_vals,
                "g0_vals": g0_vals,
                "m0": float(resolved[f"{name}_mass"]),
            })

    if args.resonances:
        resonances = [r for r in resonances if r["name"] in args.resonances]

    if not resonances:
        print("No resonances found")
        sys.exit(1)

    n = len(resonances)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.5 * n), squeeze=False)
    fig.subplots_adjust(hspace=0.35)

    for idx, res in enumerate(resonances):
        ax_ls = axes[idx, 0]
        ax_ar = axes[idx, 1]

        m0 = res["m0"]
        invD = res["invD"]
        invD_re, invD_im = invD.real, invD.imag
        lineshape = invD_im  # Im(1/D) = BW absorptive part
        gamma_names = res["gamma_names"]
        gamma_vals = res["gamma_vals"]
        g0_vals = res["g0_vals"]

        # ── Lineshape ─────────────────────────────────────────
        ax_ls.plot(m_grid, lineshape, "b-", linewidth=1.5, label="total")
        # Individual Re(Γᵢ) contributions (skip if only one component)
        D = 1.0 / invD  # for per-component scale
        if len(g0_vals) > 1:
            scale = m0 / np.abs(D)**2
            # Group CK-matrix components: diag re_aa individually, sum off-diag re_ab and im_ab
            diag = {}
            off_re = None
            off_im = None
            others = {}
            for i in range(min(len(g0_vals), gamma_vals.shape[0])):
                raw = g0_vals[i] * gamma_vals[i].real * scale
                gn = res["gamma_names"][i]
                if "_re_" in gn:
                    idx = gn.split("_re_")[1]    # e.g. "0_0", "0_1"
                    parts = idx.split("_")
                    if len(parts) == 2 and parts[0] == parts[1]:
                        diag[f"re_{parts[0]}"] = raw
                    else:
                        off_re = raw if off_re is None else off_re + raw
                elif "_im_" in gn:
                    off_im = raw if off_im is None else off_im + raw
                else:
                    label = gn.split("_", 1)[1] if "_" in gn else gn
                    others[label] = raw
            labels = []
            for label, curve in {**diag, **others}.items():
                ax_ls.plot(m_grid, curve, "--", linewidth=0.8, alpha=0.7, label=label)
                labels.append(label)
            if off_re is not None:
                ax_ls.plot(m_grid, off_re, "--", linewidth=0.8, alpha=0.7, label="re_ab")
                labels.append("re_ab")
            if off_im is not None:
                ax_ls.plot(m_grid, off_im, "--", linewidth=0.8, alpha=0.7, label="im_ab")
                labels.append("im_ab")
            total_chars = sum(len(l) for l in labels)
            ax_ls.legend(fontsize=7, ncol=max(1, total_chars // 50 + 1))
        ax_ls.axvline(m0, color="grey", linestyle=":", linewidth=0.8)
        ax_ls.set_title(f"{res['disp']}  (m₀={m0:.3f})", fontsize=9)
        ax_ls.set_xlim(mass_lo, mass_hi)
        ax_ls.set_ylabel(r"$m_0\,\mathrm{Re}\Gamma\,/\,|D|^2$")
        ax_ls.set_xlabel(r"m (GeV)")
        ax_ls.grid(True, alpha=0.3)

        # ── Argand diagram (Im(1/D) vs Re(1/D)) ──────────────
        # Colour-code by mass
        colours = plt.cm.viridis((m_grid - m_grid[0]) / (m_grid[-1] - m_grid[0]))
        ax_ar.scatter(invD_re, invD_im, c=colours, s=3, alpha=0.8)
        ax_ar.plot(invD_re, invD_im, "b-", linewidth=0.5, alpha=0.5)
        # Marker at m₀
        idx_m0 = np.argmin(np.abs(m_grid - m0))
        ax_ar.plot(invD_re[idx_m0], invD_im[idx_m0], "ko", markersize=4)
        ax_ar.axhline(0, color="grey", linewidth=0.5)
        ax_ar.axvline(0, color="grey", linewidth=0.5)
        ax_ar.set_title("Argand  (1/D)", fontsize=9)
        ax_ar.set_xlabel(r"Re(1/D)")
        ax_ar.set_ylabel(r"Im(1/D)")
        ax_ar.grid(True, alpha=0.3)
        # Align y-axis range between lineshape and Argand
        y_ls = ax_ls.get_ylim()
        y_min = min(y_ls[0], np.min(invD_im))
        y_max = max(y_ls[1], np.max(invD_im))
        pad = (y_max - y_min) * 0.1
        ax_ls.set_ylim(y_min - pad, y_max + pad)
        ax_ar.set_ylim(y_min - pad, y_max + pad)
        x_pad = max(np.abs(invD_re)) * 0.1
        ax_ar.set_xlim(np.min(invD_re) - x_pad, np.max(invD_re) + x_pad)

    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {args.output}")


if __name__ == "__main__":
    main()
