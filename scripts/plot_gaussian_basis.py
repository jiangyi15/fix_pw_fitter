#!/usr/bin/env python3
"""Plot GaussianBasis amplitude contributions from a fit.

Shows each Gaussian basis particle's CK-weighted amplitude and their
total combined lineshape.

Usage::

    # All particles
    python scripts/plot_gaussian_basis.py config.yml results.json -o plot.pdf

    # Filter by particle name pattern
    python scripts/plot_gaussian_basis.py config.yml results.json --pattern MI00
    python scripts/plot_gaussian_basis.py config.yml results.json --pattern "MI0[12]"
    python scripts/plot_gaussian_basis.py config.yml results.json --pattern "p$"
"""

import sys, os, re, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description="Plot GaussianBasis lineshape")
    ap.add_argument("config")
    ap.add_argument("results_json")
    ap.add_argument("--mass-lo", type=float, default=0.28)
    ap.add_argument("--mass-hi", type=float, default=5.14)
    ap.add_argument("--n-points", type=int, default=500)
    ap.add_argument("-o", "--output", default="gaussian_basis.pdf")
    ap.add_argument("--pattern", default=None,
                    help="Regex filter on particle name (e.g. 'MI00', 'p$', 'MI0[12]')")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ampfit import Fitter

    # ── Load ────────────────────────────────────────────────────
    os.chdir(os.path.dirname(os.path.abspath(args.config)))
    f = Fitter(args.config, backend="numpy")
    cp = os.path.splitext(args.results_json)[0] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)
    r = f.load_results(args.results_json)
    _, resolved = f.build_params(r.x)

    m_grid = np.linspace(args.mass_lo, args.mass_hi, args.n_points)

    # ── Build CK and collect per-particle amplitudes ────────────
    seen_models = set()
    particle_map = {}  # name → (model, mu, sigma)
    for chain in f.config.full_decay.chains:
        for decay in chain.decays[1:]:
            model = decay.core._model
            if type(model).__name__ != "GaussianBasisModel":
                continue
            mid = id(model)
            if mid in seen_models:
                continue
            seen_models.add(mid)
            name = decay.core.name
            mu = float(model.kwargs.get("mu", 0.775))
            sigma = float(model.kwargs.get("sigma", 0.1))
            particle_map[name] = (model, mu, sigma)

    if not particle_map:
        print("No GaussianBasis particles found")
        sys.exit(1)

    # Filter by pattern if given
    if args.pattern:
        filtered = {n: v for n, v in particle_map.items()
                    if re.search(args.pattern, n)}
        if not filtered:
            print(f"Pattern '{args.pattern}' matched no particles. Available: {list(particle_map)}")
            sys.exit(1)
        print(f"Filtered {len(particle_map)} → {len(filtered)} particles by '{args.pattern}'")
        particle_map = filtered

    print(f"Found {len(particle_map)} GaussianBasis particles")

    # Pre-compute amplitude_raw for each particle at all m_grid points
    amp_raw = {}
    for name, (model, mu, sigma) in particle_map.items():
        amp_raw[name] = model.amplitude_raw(m_grid, resolved)

    # Accumulate CK-weighted amplitude per particle
    ck_arr = f.pc.build_ck(resolved)  # complex[448]
    per_particle = {name: np.zeros(len(m_grid), dtype=complex)
                    for name in particle_map}

    for i, comb in enumerate(f.all_comb):
        for term in comb:
            if isinstance(term, str):
                for pname in particle_map:
                    if term.startswith(pname):
                        per_particle[pname] += amp_raw[pname] * ck_arr[i]
                        break

    # Group by CP variant (strip trailing 'p'/'m')
    from collections import defaultdict
    grouped = defaultdict(list)
    for name in sorted(per_particle):
        variant = name[-1]  # 'p' or 'm'
        grouped[variant].append(name)

    # ── Plot ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.08, wspace=0.08)
    ax_revm = fig.add_subplot(gs[0, 0])                # x=Re, y=m
    ax_sq = fig.add_subplot(gs[0, 1])                   # x=m, y=|A|²
    ax_ar = fig.add_subplot(gs[1, 0], sharex=ax_revm)   # Argand: share Re
    ax_im = fig.add_subplot(gs[1, 1], sharex=ax_sq, sharey=ax_ar)  # share m + Im

    # Plot each variant's total only (no individual particle lines)
    for variant, names in grouped.items():
        total_var = np.zeros(len(m_grid), dtype=complex)
        for name in names:
            A = per_particle[name]
            total_var += A

        ls = '-' if variant == 'p' else '--'
        ax_revm.plot(total_var.real, m_grid, 'k' + ls, linewidth=2, label=f'Total ({variant})')
        ax_sq.plot(m_grid, np.abs(total_var)**2, 'k' + ls, linewidth=2, label=f'Total ({variant})')
        ax_im.plot(m_grid, total_var.imag, 'k' + ls, linewidth=2, label=f'Total ({variant})')
        ax_ar.plot(total_var.real, total_var.imag, 'k' + ls, linewidth=1.5,
                   label=f'Total ({variant})')

    for ax in [ax_revm, ax_im, ax_sq]:
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.legend(fontsize=5, ncol=2)
        ax.grid(True, alpha=0.3)

    # Hide redundant tick labels on shared axes
    ax_revm.tick_params(labelbottom=False)   # x shared with Argand
    ax_sq.tick_params(labelbottom=False)     # x shared with bottom-right
    ax_im.tick_params(labelleft=False)       # y shared with Argand

    ax_revm.set_xlabel("Re(amplitude)")
    ax_revm.set_ylabel("m (GeV)")
    ax_sq.set_xlabel("m (GeV)")
    ax_sq.set_ylabel("|amplitude|²")
    ax_sq.yaxis.set_label_position("right")
    ax_sq.tick_params(labelright=True, labelleft=False)
    ax_im.set_xlabel("m (GeV)")
    ax_im.set_ylabel("Im(amplitude)")

    # Argand
    ax_ar.axhline(0, color="grey", linewidth=0.5)
    ax_ar.axvline(0, color="grey", linewidth=0.5)
    ax_ar.legend(fontsize=7)
    ax_ar.grid(True, alpha=0.3)
    ax_ar.set_xlabel("Re(amplitude)")
    ax_ar.set_ylabel("Im(amplitude)")

    # Symmetric range for Argand; sharex/sharey align the rest
    r = max(np.abs(ax_ar.get_xlim()).max(), np.abs(ax_ar.get_ylim()).max()) * 1.1
    ax_ar.set_xlim(-r, r)
    ax_ar.set_ylim(-r, r)

    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {args.output}")


if __name__ == "__main__":
    main()
