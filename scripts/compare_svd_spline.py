#!/usr/bin/env python3
"""
Compare ExpSplineSVD (SVD-reduced spline-k) vs normal ExpSpline.

For a fixed physics configuration, plots the worst-case |A|^2 error
(relative to the peak of the true exponential amplitude) as a function
of ``n_reduce``, one curve per ``n_interp``, together with the
ExpSpline baseline (its own between-grid spline error, dashed).

Usage::

    python scripts/compare_svd_spline.py [-o out.png]
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ampfit.particle_model import build_particle
from ampfit.particle_model.spline_k_model import (
    KToSplineWeightsTransform, ExpSplineModel,
)

# ── Physics configuration (well-conditioned regime) ────────────────
M0, G0 = 0.5, 1.0
K_RANGE = [0.1, 2.0]
M_RANGE = [0.28, 2.0]
N_MASS_PTS = 10000   # model default — SVD-grid resolution
GAMMA_CLIP = 1e6
N_INTERP_LIST = [10, 20, 40, 60]
N_REDUCE_MAX = max(N_INTERP_LIST)
# Test masses (dense) and k values between grid points (worst case)
M_TEST = np.linspace(0.30, 1.95, 800)
K_TEST = np.linspace(0.2, 1.9, 8)


def true_amp(m, k):
    return np.exp(-k * (m ** 2 - M0 ** 2))


def amp_from_gamma(gamma_m, m):
    """Amplitude from blended gamma(m)."""
    return 1.0 / (M0 ** 2 - m ** 2 - 1j * M0 * gamma_m)


def amp_err(gamma_of_k, k_vals):
    """Worst |A|^2 error / peak, over test k and mass."""
    worst = 0.0
    for k in k_vals:
        g = gamma_of_k(k)
        A = amp_from_gamma(g, M_TEST)
        p_true = np.abs(true_amp(M_TEST, k)) ** 2
        worst = max(worst, float(np.max(np.abs(np.abs(A) ** 2 - p_true))
                                 / np.max(p_true)))
    return worst


def exp_spline_err(n_interp):
    """ExpSpline baseline: spline-blended gamma vs true exp amplitude."""
    model = build_particle("sigma", mass=M0, width=G0, model="ExpSpline",
                           k=1.0, k_range=K_RANGE, n_interp=n_interp)
    tfm = KToSplineWeightsTransform(
        "sigma_k", "sigma_mass", model.get_gamma_name(),
        mass_fixed=M0, k_min=K_RANGE[0], k_max=K_RANGE[1])
    g_rows = np.array([model.gamma_k(M_TEST, ki) for ki in
                       np.linspace(K_RANGE[0], K_RANGE[1], n_interp)])

    def gamma_of_k(k):
        out = tfm.forward({"sigma_k": k})
        w = np.array([out[n] for n in model.get_gamma_name()])
        return w @ g_rows

    return amp_err(gamma_of_k, K_TEST)


def svd_spline_err(n_interp, n_reduce):
    """ExpSplineSVD: reduced gamma vs true exp amplitude."""
    model = build_particle("sigma", mass=M0, width=G0, model="ExpSplineSVD",
                           k=1.0, k_range=K_RANGE, n_interp=n_interp,
                           n_reduce=n_reduce, n_mass_pts=N_MASS_PTS,
                           m_range=M_RANGE, gamma_clip=GAMMA_CLIP)
    tfm = model.make_mass_width_transform()
    g_rows = np.array(model.gamma(M_TEST))  # (n_reduce, n_mass)

    def gamma_of_k(k):
        out = tfm.forward({"sigma_k": k})
        w = np.array([out[n] for n in model.get_gamma_name()])
        return w @ g_rows

    return amp_err(gamma_of_k, K_TEST)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default="compare_svd_spline.png",
                    help="figure 1: error vs n_reduce per n_interp")
    ap.add_argument("--win-output", default="compare_svd_spline_win.png",
                    help="figure 2: cost (kernel rows) vs accuracy")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(N_INTERP_LIST)))
    colors = dict(zip(N_INTERP_LIST, cmap))

    baselines = {}
    print(f"ExpSpline baselines: k={K_RANGE}, m<={M_RANGE[1]}")
    for n_interp in N_INTERP_LIST:
        e = exp_spline_err(n_interp)
        baselines[n_interp] = e
        print(f"  n_interp={n_interp:3d}: |A|^2 err/peak = {e:.3e}")

    n_reduce_vals = list(range(2, N_REDUCE_MAX + 1))
    for n_interp in N_INTERP_LIST:
        errs = []
        for r in n_reduce_vals:
            if r > n_interp:
                break
            e = svd_spline_err(n_interp, r)
            errs.append(e)
            print(f"  SVD n_interp={n_interp:3d} n_reduce={r:2d}: {e:.3e}")
        xs = n_reduce_vals[:len(errs)]
        ax.plot(xs, errs, "o-", color=colors[n_interp], ms=4, lw=1.5,
                label=f"SVD n$_k$={n_interp}")
        # ExpSpline baseline (dashed, same color)
        ax.axhline(baselines[n_interp], color=colors[n_interp], ls="--",
                   lw=1.2)
        ax.annotate(f"ExpSpline\nn$_k$={n_interp}\n{baselines[n_interp]:.1e}",
                    xy=(N_REDUCE_MAX - 1, baselines[n_interp]),
                    xytext=(N_REDUCE_MAX - 1.5, baselines[n_interp] * 3),
                    fontsize=8, color=colors[n_interp], va="center",
                    arrowprops=dict(arrowstyle="-", color=colors[n_interp],
                                    lw=0.8))

    ax.set_yscale("log")
    ax.set_xlabel(r"$n_{\mathrm{reduce}}$  (SVD components kept)")
    ax.set_ylabel(r"max $|\,|A|^2 - |A_{\exp}|^2\,|$ / peak $|A_{\exp}|^2$")
    ax.set_title(
        rf"ExpSplineSVD reduction vs ExpSpline   "
        rf"($k\in{K_RANGE}$, $m\in[{M_RANGE[0]},{M_RANGE[1]}]$ GeV, "
        rf"$\gamma_{{clip}}={GAMMA_CLIP:g}$)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {args.output}")

    _make_win_plot(args.win_output)


def _make_win_plot(out_path):
    """Cost–accuracy trade-off: kernel rows vs error for both models.

    ExpSpline pays n_interp rows for its accuracy; ExpSplineSVD pays
    only n_reduce rows (the SVD basis), independently of the n_interp
    reference resolution — the "win".
    """
    N_REF = 200          # SVD reference spline basis (free at runtime)
    exp_rows = [10, 20, 40, 60, 100, 200]
    svd_red = [2, 3, 4, 5, 6, 8, 10, 12, 16]

    exp_errs = [exp_spline_err(n) for n in exp_rows]
    svd_errs = [svd_spline_err(N_REF, r) for r in svd_red]

    for n, e in zip(exp_rows, exp_errs):
        print(f"  [win] ExpSpline rows={n:4d}: {e:.3e}")
    for r, e in zip(svd_red, svd_errs):
        print(f"  [win] SVD n_interp={N_REF} rows={r:3d}: {e:.3e}")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(exp_rows, exp_errs, "o-", color="tab:red", lw=2, ms=7,
            label="ExpSpline  (rows $= n_{\\mathrm{interp}}$)")
    ax.plot(svd_red, svd_errs, "s-", color="tab:blue", lw=2, ms=7,
            label=f"ExpSplineSVD  ($n_{{\\mathrm{{interp}}}}={N_REF}$, "
                  "rows $= n_{\\mathrm{reduce}}$)")

    # Annotate the two "win" statements
    i10 = svd_red.index(10)
    ax.annotate("same 10 rows:\nExp 5.9e-3 → SVD 5.7e-8\n(~5 orders)",
                xy=(10, svd_errs[i10]), xytext=(16, 5e-5),
                fontsize=9, color="tab:blue",
                arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.2))

    e60 = exp_errs[exp_rows.index(60)]
    ax.annotate("same ~5e-8 accuracy:\n60 rows (Exp) vs 10 rows (SVD)\n6× fewer",
                xy=(60, e60), xytext=(70, 1e-5),
                fontsize=9, color="tab:red",
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.2))

    ax.set_yscale("log")
    ax.set_xlabel("kernel gamma rows per event  (fit cost)")
    ax.set_ylabel(r"max $|\,|A|^2 - |A_{\exp}|^2\,|$ / peak $|A_{\exp}|^2$")
    ax.set_title(
        rf"Cost vs accuracy: ExpSplineSVD decouples basis size from cost   "
        rf"($k\in{K_RANGE}$, $m\leq{M_RANGE[1]}$ GeV)")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
