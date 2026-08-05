#!/usr/bin/env python3
"""
Compare the 2-parameter SVD model (Exp2DSplineSVD) with the 1-parameter
model (ExpSplineSVD).

Both models reproduce an exponential amplitude

    1D: A = exp(-k(m² - m0²))          (b = 0 limit)
    2D: A = exp(-(a + b·i)(m² - m0²))  (decay + oscillation)

The plot shows the worst complex-amplitude error vs ``n_reduce``
(total kernel rows), evaluated **off the parameter grid** through the
real model transforms.  Each model is built once at full rank and the
first ``r`` basis rows are used per point (same result as building per
``r``, but the one-time 50×50 SVD build is not repeated).

Curves:

* 2D model, 12×12 grid
* 2D model, 50×50 grid  (finer → much lower accuracy floor)
* 1D model, n_interp = 200 reference

Usage::

    python scripts/compare_svd_2d.py [-o out.png]
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ampfit.particle_model import build_particle
from ampfit.particle_model.spline_k_model import spline_weights

M0, G0 = 0.5, 1.0
A_RANGE = [0.1, 2.0]
B_RANGE = [0.1, 2.0]
M_RANGE = [0.28, 2.0]
N_MASS_PTS = 6000
GAMMA_CLIP = 1e6
M_TEST = np.linspace(0.30, 1.95, 600)

# test points off the parameter grid (worst case for the spline)
AB_TEST = [(0.35, 0.28), (0.8, 0.7), (1.25, 1.1), (1.7, 1.6), (1.95, 1.9)]
K_TEST = [0.35, 0.8, 1.25, 1.7, 1.95]


def build_2d(n_a, n_b):
    """2D model at full rank (all rows kept), for slicing by r."""
    return build_particle(
        "sigma", mass=M0, width=G0, model="Exp2DSplineSVD",
        a=1.0, b=0.5, a_range=A_RANGE, b_range=B_RANGE,
        n_a=n_a, n_b=n_b, n_reduce=n_a * n_b + 1,
        n_mass_pts=N_MASS_PTS, m_range=M_RANGE, gamma_clip=GAMMA_CLIP)


def build_1d(n_interp):
    """1D model at full rank, for slicing by r."""
    return build_particle(
        "sigma", mass=M0, width=G0, model="ExpSplineSVD",
        k=1.0, k_range=A_RANGE, n_interp=n_interp,
        n_reduce=n_interp + 1,
        n_mass_pts=N_MASS_PTS, m_range=M_RANGE, gamma_clip=GAMMA_CLIP)


def _amp_from_gamma(g):
    return 1.0 / (M0 ** 2 - M_TEST ** 2 - 1j * M0 * g)


def err_2d_sliced(model, r):
    """Worst off-grid complex-|A| error with the first r basis rows."""
    tfm = model.make_mass_width_transform()
    g_rows = np.array(model.gamma(M_TEST))[:r]
    worst = 0.0
    for a, b in AB_TEST:
        wa, _ = spline_weights(a, tfm._h_a, tfm._a_grid)
        wb, _ = spline_weights(b, tfm._h_b, tfm._b_grid)
        w = np.outer(wa, wb).ravel()
        w_r = model._projection[:r] @ w
        g = w_r @ g_rows
        A = _amp_from_gamma(g)
        A_true = np.exp(-(a + 1j * b) * (M_TEST ** 2 - M0 ** 2))
        worst = max(worst, float(np.max(np.abs(A - A_true))
                                 / np.max(np.abs(A_true))))
    return worst


def err_1d_sliced(model, r):
    """Worst off-grid complex-|A| error with the first r basis rows."""
    tfm = model.make_mass_width_transform()
    g_rows = np.array(model.gamma(M_TEST))[:r]
    worst = 0.0
    for k in K_TEST:
        w, _ = spline_weights(k, tfm._h_matrix, tfm._k_grid)
        w_r = model._projection[:r] @ w
        g = w_r @ g_rows
        A = _amp_from_gamma(g)
        A_true = np.exp(-k * (M_TEST ** 2 - M0 ** 2))
        worst = max(worst, float(np.max(np.abs(A - A_true))
                                 / np.max(np.abs(A_true))))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default="compare_svd_2d.png")
    args = ap.parse_args()

    print("building models (50×50 SVD takes ~15 s)...")
    m12 = build_2d(12, 12)
    m50 = build_2d(50, 50)
    m1d = build_1d(200)
    print("  done")

    r_vals = list(range(3, 28))
    e12 = [err_2d_sliced(m12, r) for r in r_vals]
    e50 = [err_2d_sliced(m50, r) for r in r_vals]
    e1 = [err_1d_sliced(m1d, r) for r in r_vals]

    print("   r |    2D 12x12    2D 50x50    1D nk=200")
    for r, a, b, c in zip(r_vals, e12, e50, e1):
        print(f"{r:3d} | {a:12.3e} {b:12.3e} {c:12.3e}")

    for tgt in [1e-2, 1e-4, 1e-6]:
        r12 = next((r for r, e in zip(r_vals, e12) if e < tgt), None)
        r50 = next((r for r, e in zip(r_vals, e50) if e < tgt), None)
        r1 = next((r for r, e in zip(r_vals, e1) if e < tgt), None)
        print(f"  r for |A| err < {tgt:.0e}:  2D(12x12)={r12}  "
              f"2D(50x50)={r50}  1D={r1}")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(r_vals, e12, "o-", color="tab:blue", lw=2, ms=5,
            label=r"2D $e^{-(a+bi)(m^2-m_0^2)}$, 12$\times$12 grid")
    ax.plot(r_vals, e50, "s-", color="tab:purple", lw=2, ms=5,
            label=r"2D $e^{-(a+bi)(m^2-m_0^2)}$, 50$\times$50 grid")
    ax.plot(r_vals, e1, "^-", color="tab:red", lw=2, ms=5,
            label=r"1D $e^{-k(m^2-m_0^2)}$  ($n_{\mathrm{interp}}=200$)")
    ax.set_yscale("log")
    ax.set_xlabel("$n_{\\mathrm{reduce}}$  (kernel gamma rows)")
    ax.set_ylabel(r"max $|A - A_{\mathrm{true}}| / \max|A_{\mathrm{true}}|$")
    ax.set_title(
        rf"2D vs 1D SVD reduction, off-grid test points   "
        rf"($a,b,k\in[0.1,2]$, $m\leq{M_RANGE[1]}$ GeV)")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {args.output}")


if __name__ == "__main__":
    main()
