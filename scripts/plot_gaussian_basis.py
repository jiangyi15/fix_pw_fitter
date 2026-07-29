#!/usr/bin/env python3
"""Plot GaussianBasis amplitude contributions from a fit.

Shows each Gaussian basis function (scaled by its CK coefficient) and
their total as a combined lineshape.

Usage::

    python scripts/plot_gaussian_basis.py /path/to/config_amp.yml /path/to/results.json -o gauss_basis.pdf
"""

import sys, os, argparse
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
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ampfit import Fitter

    # ── Load ────────────────────────────────────────────────────
    # Change to config directory so relative NPY paths resolve
    os.chdir(os.path.dirname(os.path.abspath(args.config)))
    f = Fitter(args.config, backend="numpy")
    cp = os.path.splitext(args.results_json)[0] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)
    r = f.load_results(args.results_json)
    _, resolved = f.build_params(r.x)

    m_grid = np.linspace(args.mass_lo, args.mass_hi, args.n_points)

    # ── Collect GaussianBasis particles ─────────────────────────
    seen = set()
    basis_list = []
    for chain in f.config.full_decay.chains:
        for decay in chain.decays[1:]:
            model = decay.core._model
            if type(model).__name__ != "GaussianBasisModel":
                continue
            mid = id(model)
            if mid in seen:
                continue
            seen.add(mid)

            name = decay.core.name
            # Amplitude = 1/D → Gaussian shape (no CK scaling)
            A = model.amplitude_raw(m_grid, resolved)

            mu = float(model.kwargs.get("mu", 0.775))
            sigma = float(model.kwargs.get("sigma", 0.1))
            disp = decay.core.display or name

            basis_list.append({
                "name": name, "disp": disp,
                "mu": mu, "sigma": sigma,
                "A": A,
            })
            print(f"  {name}: μ={mu:.3f} σ={sigma:.3f}  |A(μ)|={abs(A[np.argmin(np.abs(m_grid-mu))]):.4f}")

    if not basis_list:
        print("No GaussianBasis particles found in config")
        sys.exit(1)

    # ── Plot basis functions ────────────────────────────────────
    # Sort by mu for clean legend
    basis_list.sort(key=lambda b: b["mu"])
    colours = plt.cm.viridis(np.linspace(0, 1, len(basis_list)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for b, c in zip(basis_list, colours):
        ax1.plot(m_grid, b["A"].real, color=c, linewidth=0.8,
                 label=f"μ={b['mu']:.2f}")
        ax2.plot(m_grid, b["A"].imag, color=c, linewidth=0.8,
                 label=f"μ={b['mu']:.2f}")

    # Sum (unweighted — each CK=1)
    total_re = sum(b["A"].real for b in basis_list)
    total_im = sum(b["A"].imag for b in basis_list)
    ax1.plot(m_grid, total_re, "k-", linewidth=2, label="Sum")
    ax2.plot(m_grid, total_im, "k-", linewidth=2, label="Sum")

    for ax in [ax1, ax2]:
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.legend(fontsize=6, ncol=3)
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel("Re(amplitude)")
    ax1.set_title("Gaussian basis functions (CK = 1 each)")
    ax2.set_ylabel("Im(amplitude)")
    ax2.set_xlabel("m (GeV)")

    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {args.output}")


if __name__ == "__main__":
    main()
