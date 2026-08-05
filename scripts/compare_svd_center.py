#!/usr/bin/env python3
"""
Compare mean-centered vs raw SVD reduction for ExpSplineSVD.

The mean-centered variant spends one of its ``n_reduce`` rows on the
exact k-mean baseline (weight 1), so at ``n_reduce = n + 1`` it holds
``n`` SVD components *plus* the correct baseline term.  This plot
compares, per number of SVD components *n*:

* non-centered:  ``n_reduce = n``        (all rows are SVD components)
* centered:      ``n_reduce = n + 1``    (n components + exact baseline)

i.e. it answers: is the extra baseline term worth the extra row?

Usage::

    python scripts/compare_svd_center.py [-o out.png]
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ampfit.particle_model import build_particle

# ── Physics configuration (well-conditioned regime) ────────────────
M0, G0 = 0.5, 1.0
K_RANGE = [0.1, 2.0]
M_RANGE = [0.28, 2.0]
N_INTERP = 200          # reference spline basis (free at runtime)
N_MASS_PTS = 10000
GAMMA_CLIP = 1e6
M_TEST = np.linspace(0.30, 1.95, 800)
K_TEST = np.linspace(0.2, 1.9, 8)


def amp_err(n_reduce, mean_center):
    """Worst |A|^2 error / peak for the given reduction configuration."""
    model = build_particle("sigma", mass=M0, width=G0, model="ExpSplineSVD",
                           k=1.0, k_range=K_RANGE, n_interp=N_INTERP,
                           n_reduce=n_reduce, n_mass_pts=N_MASS_PTS,
                           m_range=M_RANGE, gamma_clip=GAMMA_CLIP,
                           mean_center=mean_center)
    tfm = model.make_mass_width_transform()
    g_rows = np.array(model.gamma(M_TEST))  # (n_reduce, n_mass)
    worst = 0.0
    for k in K_TEST:
        out = tfm.forward({"sigma_k": k})
        w = np.array([out[n] for n in model.get_gamma_name()])
        g = w @ g_rows
        A = 1.0 / (M0 ** 2 - M_TEST ** 2 - 1j * M0 * g)
        p_true = np.exp(-k * (M_TEST ** 2 - M0 ** 2)) ** 2
        worst = max(worst, float(np.max(np.abs(np.abs(A) ** 2 - p_true))
                                 / np.max(p_true)))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default="compare_svd_center.png")
    args = ap.parse_args()

    n_comp = list(range(1, 17))          # number of SVD components
    err_none = []                        # n rows, all components
    err_center = []                      # n+1 rows, n components + baseline
    for n in n_comp:
        e_none = amp_err(n, mean_center=False)
        e_center = amp_err(n + 1, mean_center=True)
        err_none.append(e_none)
        err_center.append(e_center)
        print(f"n_comp={n:2d}: no-mean(n rows)={e_none:.3e}   "
              f"centered(n+1 rows)={e_center:.3e}")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(n_comp, err_none, "o-", color="tab:red", lw=2, ms=6,
            label="no mean-centering  ($n_{\\mathrm{reduce}}=n$)")
    ax.plot(n_comp, err_center, "s-", color="tab:blue", lw=2, ms=6,
            label="mean-centered  ($n_{\\mathrm{reduce}}=n+1$)")

    # Mark where the centered version wins
    better = [n for n, (a, b) in enumerate(zip(err_none, err_center), start=1)
              if b < a]
    if better:
        ax.annotate(
            "centered wins: the extra\nbaseline term is exact\n"
            f"(first at n={better[0]})",
            xy=(better[0], err_center[better[0] - 1]),
            xytext=(better[0] + 3, max(err_center) * 0.3),
            fontsize=9, color="tab:blue",
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.2))

    ax.set_yscale("log")
    ax.set_xlabel("number of SVD components  $n$")
    ax.set_ylabel(r"max $|\,|A|^2 - |A_{\exp}|^2\,|$ / peak $|A_{\exp}|^2$")
    ax.set_title(
        rf"Mean-centered (n+1 rows) vs raw (n rows)   "
        rf"($n_{{\mathrm{{interp}}}}={N_INTERP}$, $k\in{K_RANGE}$, "
        rf"$m\leq{M_RANGE[1]}$ GeV)")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {args.output}")


if __name__ == "__main__":
    main()
