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
    ap.add_argument("--mass-lo", type=float, default=0.2)
    ap.add_argument("--mass-hi", type=float, default=5.2)
    ap.add_argument("--n-points", type=int, default=2000)
    ap.add_argument("-o", "--output", default="bw_lineshape.pdf")
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--resonances", nargs="*", default=None)
    args = ap.parse_args()

    from ampfit import Fitter
    import matplotlib.pyplot as plt

    f = Fitter(args.config, backend=args.backend)

    resolved = None
    if args.results_json:
        cp = os.path.splitext(args.results_json)[0] + "_constraints.json"
        if os.path.exists(cp):
            f.load_constraints(cp)
        r = f.load_results(args.results_json)
        if r.x is not None and len(r.x) > 0:
            _, resolved, _, _ = f._build_params(r.x)

    m_grid = np.linspace(args.mass_lo, args.mass_hi, args.n_points)

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
            gamma_fn = getattr(model, "gamma", None)
            if gamma_fn is None:
                continue
            kw = getattr(model, "kwargs", {})

            # Mass
            mass_key = f"{name}_mass"
            m0 = float(resolved[mass_key]) if (resolved and mass_key in resolved) else float(kw.get("mass", 0.775))

            # Gamma parameter names and their values
            gamma_names = list(model.get_gamma_name()) if hasattr(model, "get_gamma_name") else []
            gamma_defaults = list(model.get_gamma_defaults()) if hasattr(model, "get_gamma_defaults") else []
            g0_vals = []
            for i, gn in enumerate(gamma_names):
                if resolved and gn in resolved:
                    g0_vals.append(float(resolved[gn]))
                elif i < len(gamma_defaults):
                    g0_vals.append(float(gamma_defaults[i]))
                else:
                    g0_vals.append(1.0)

            # Width from fit (used only for display, not for normalization)
            width_key = f"{name}_width"
            w0 = float(resolved[width_key]) if (resolved and width_key in resolved) else float(kw.get("width", 0.1))

            resonances.append({
                "name": name, "m0": m0, "width": w0,
                "gamma_fn": gamma_fn,
                "g0_vals": g0_vals,
            })

    if args.resonances:
        resonances = [r for r in resonances if r["name"] in args.resonances]

    if not resonances:
        print("No resonances found")
        sys.exit(1)

    n = len(resonances)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for idx, res in enumerate(resonances):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        m0, w0 = res["m0"], res["width"]
        g0_vals = res["g0_vals"]
        gamma_vals = np.asarray(res["gamma_fn"](m_grid), dtype=complex)

        # Running width Γ(m) = Σ gᵢ · γᵢ(m)
        if gamma_vals.ndim > 1:
            Gamma_m = np.zeros_like(m_grid, dtype=float)
            for i in range(min(len(g0_vals), gamma_vals.shape[0])):
                Gamma_m += g0_vals[i] * np.real(gamma_vals[i])
        else:
            Gamma_m = np.real(gamma_vals) * (g0_vals[0] if g0_vals else 1.0)

        Gamma_m = np.maximum(Gamma_m, 0)

        # Im(D)/|D|² = m₀·Γ(m) / ((m₀²−m²)² + (m₀·Γ(m))²)
        ReD = m0**2 - m_grid**2
        ImD = m0 * Gamma_m
        denom = ReD**2 + ImD**2
        lineshape = np.divide(ImD, denom, where=denom > 1e-30, out=np.zeros_like(denom))

        ax.plot(m_grid, lineshape, "b-", linewidth=1.5)
        ax.axvline(m0, color="grey", linestyle=":", linewidth=0.8)
        ax.set_title(f"{res['name']}  (m₀={m0:.3f}, Γ={w0:.3f})", fontsize=9)
        ax.set_xlim(args.mass_lo, args.mass_hi)
        ax.set_ylabel(r"Im(D)/|D|²")
        ax.set_xlabel("m (GeV)")
        ax.grid(True, alpha=0.3)

    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {args.output}")


if __name__ == "__main__":
    main()
