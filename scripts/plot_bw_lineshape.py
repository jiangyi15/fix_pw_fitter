#!/usr/bin/env python3
"""
Plot Breit-Wigner lineshapes Im(D)/|D|² where D = m₀² − m² − im₀Γ(m).

Γ(m) is the running width from the particle model's gamma() method.
Uses fitted mass/width and gamma parameters from a fit results file.

Usage:
    python scripts/plot_bw_lineshape.py config_amp.yml -o bw_plot.pdf
    python scripts/plot_bw_lineshape.py config_amp.yml results.json -o bw_plot.pdf
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

    # Load fit results if provided
    resolved = None
    if args.results_json:
        cp = os.path.splitext(args.results_json)[0] + "_constraints.json"
        if os.path.exists(cp):
            f.load_constraints(cp)
        r = f.load_results(args.results_json)
        if r.x is not None and len(r.x) > 0:
            _, resolved, _, _ = f._build_params(r.x)

    m_grid = np.linspace(args.mass_lo, args.mass_hi, args.n_points)

    # Collect unique particle models
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

            # Mass and width from resolved (fit result) or config defaults
            mass_key = f"{name}_mass"
            width_key = f"{name}_width"
            if resolved and mass_key in resolved:
                m0 = float(resolved[mass_key])
            else:
                m0 = float(kw.get("mass", 0.775))
            if resolved and width_key in resolved:
                w0 = float(resolved[width_key])
            else:
                w0 = float(kw.get("width", 0.1))

            resonances.append({
                "name": name, "m0": m0, "width": w0,
                "gamma_fn": gamma_fn, "model": model,
            })

    if args.resonances:
        resonances = [r for r in resonances if r["name"] in args.resonances]

    if not resonances:
        print("No resonances found")
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for res in resonances:
        m0, w0 = res["m0"], res["width"]
        gamma_vals = np.asarray(res["gamma_fn"](m_grid), dtype=complex)
        if gamma_vals.ndim > 1:
            Gamma_m = np.sum(np.real(gamma_vals), axis=0)
        else:
            Gamma_m = np.real(gamma_vals)

        # Normalize so that Γ(m₀) = w0
        idx_m0 = np.argmin(np.abs(m_grid - m0))
        Gamma_m0 = Gamma_m[idx_m0]
        if Gamma_m0 > 0:
            Gamma_m *= w0 / Gamma_m0

        # Im(D)/|D|² = m₀·Γ(m) / ((m₀²−m²)² + (m₀·Γ(m))²)
        ReD = m0**2 - m_grid**2
        ImD = m0 * np.maximum(Gamma_m, 0)
        denom = ReD**2 + ImD**2
        lineshape = np.divide(ImD, denom, where=denom > 1e-30, out=np.zeros_like(denom))

        integral = np.trapezoid(lineshape, m_grid)
        normed = lineshape / integral if integral > 0 else lineshape

        axes[0].plot(m_grid, normed, label=res["name"], linewidth=1.5)
        axes[1].plot(m_grid, lineshape, label=res["name"], linewidth=1.5)

    for ax in axes:
        ax.set_xlabel("m (GeV)")
        ax.legend(fontsize=6, ncol=2)
        ax.set_xlim(args.mass_lo, args.mass_hi)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"Im(D)/|D|² (normalized)")
    axes[1].set_ylabel(r"Im(D)/|D|²")

    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {args.output}")


if __name__ == "__main__":
    main()
