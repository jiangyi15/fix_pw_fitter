#!/usr/bin/env python3
"""Plot 2D adaptive-bin pulls for mass column pairs.

Shows ``mass[:,0] vs mass[:,1]``, ``mass[:,2] vs mass[:,3]``,
``mass[:,4] vs mass[:,5]`` in a grid.  Each panel uses the adaptive
equal-quantile pull plot (data scatter + total-fit pull rectangles +
colorbar) via :meth:`PWGroupPlotter.plot_2d`.

Usage::

    python scripts/plot_2d_mass.py fit_results.json -o mass_2d.png
    python scripts/plot_2d_mass.py fit_results.json --config config_amp.yml
"""

import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ampfit import Fitter
from ampfit.plot_pw_groups import PWGroupPlotter


def main():
    ap = argparse.ArgumentParser(description="2D mass pull plots")
    ap.add_argument("fit_json", help="JSON from Fitter.save_params()")
    ap.add_argument("--config", default="config_amp.yml")
    ap.add_argument("--data", default="data/data_arrays.npz")
    ap.add_argument("--phsp", default="data/phsp_arrays.npz")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--backend", default="cuda_v3")
    ap.add_argument("-o", "--output", default="mass_2d.png")
    ap.add_argument("--binning", default=None,
                    help="Adaptive binning spec, e.g. [[2,2]]*3. "
                         "Default: n = max(log4(N_data/50), 2) levels of [[2,2]]")
    ap.add_argument("--format", default="png",
                    help="Image format (default: png)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Parse binning spec like "[[2,2]]*3", or derive from data size:
    # n = max(int(log4(N_data / 50)), 2) levels of [[2,2]]
    if args.binning is None:
        data_np0, _nd = Fitter.load_npz(args.data, max_events=args.max_events)
        n = max(int(np.log(data_np0["mass"].shape[0] / 50.0) / np.log(4)), 2)
        binning = [[2, 2]] * n
        print(f"  auto binning: [[2,2]]*{n}  (N_data={data_np0['mass'].shape[0]:,})")
    else:
        binning = eval(args.binning, {"__builtins__": {}}, {})
        if not isinstance(binning, list):
            raise SystemExit(f"--binning must be a list, got {args.binning!r}")

    # ── Load (same pattern as scripts/plot_pw_groups.py) ────────
    f = Fitter(args.config, backend=args.backend)
    cp = os.path.splitext(args.fit_json)[0] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)
    else:
        f.apply_constrains()
        import json as _json
        with open(args.fit_json) as _fh:
            f.load_fixed_from_dict(_json.load(_fh))

    data_np, nd = Fitter.load_npz(args.data, max_events=args.max_events)
    phsp_np, np_ = Fitter.load_npz(args.phsp, max_events=args.max_events)
    print(f"  Loaded {nd:,} data + {np_:,} phsp events")
    f.set_phsp(phsp_np)
    f.set_data(data_np)

    r = f.load_results(args.fit_json)
    if r.x is None or len(r.x) == 0:
        sys.exit(1)

    # Total-only model weight: empty groups → compute() does a single
    # pass for P_total (no per-group decomposition needed for 2D pulls).
    plotter = PWGroupPlotter(f, r, {}).compute()
    print(f"  scale={plotter._scale:.4f}")

    # ── Pairs of mass columns ───────────────────────────────────
    # 4 permutations × 2 CP × 3 types × 2 comps = 48 columns.
    # Each perm (not incl. CP) occupies 6 columns:
    #   [pipi1, pipi2, pipipip, pipi1, pipipim, pipi1]
    # perm p block (CP=0): cols 6p .. 6p+5
    perm_cols = {  # panel key → [(x_col, y_col) per permutation, ...]
        (0, 1): [(6 * p + 0, 6 * p + 1) for p in range(4)],   # pipi1 vs pipi2
        (2, 3): [(6 * p + 2, 6 * p + 3) for p in range(4)],   # pipipip vs pipi1
        (4, 5): [(6 * p + 4, 6 * p + 5) for p in range(4)],   # pipipim vs pipi1
    }
    pairs = [(0, 1), (2, 3), (4, 5)]
    pipi1 = r"$m(\pi^+\pi^-)_1$"
    pipi2 = r"$m(\pi^+\pi^-)_2$"
    pipipip = r"$m(\pi^+\pi^+\pi^-)$"
    pipipim = r"$m(\pi^+\pi^-\pi^-)$"
    pair_labels = {
        (0, 1): (pipi1, pipi2),
        (2, 3): (pipipip, pipi1),
        (4, 5): (pipipim, pipi1),
    }

    # Physical ranges: m(pi pi) in [2 m_pi, m_B - 2 m_pi],
    # m(pi pi pi) in [3 m_pi, m_B - m_pi]
    m_pi = 0.13957
    m_B = 5.279
    r_pipi = (2 * m_pi, m_B - 2 * m_pi)
    r_3pi = (3 * m_pi, m_B - m_pi)
    pair_ranges = {
        (0, 1): (r_pipi, r_pipi),
        (2, 3): (r_3pi, r_pipi),
        (4, 5): (r_3pi, r_pipi),
    }

    n_cols = min(3, len(pairs))
    n_rows = (len(pairs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 5.5 * n_rows), squeeze=False)

    for ax_i, (i, j) in enumerate(pairs):
        ax = axes.flat[ax_i]
        x_range, y_range = pair_ranges[(i, j)]
        plotter.plot_2d(
            lambda x, a=i, b=j: [x["mass"][:, a], x["mass"][:, b]],
            list(pair_labels[(i, j)]),
            f"mass_{i}_{j}", binning=binning, output=None, ax=ax,
            x_range=x_range, y_range=y_range, scatter_step=10)

    for ax_i in range(len(pairs), n_rows * n_cols):
        axes.flat[ax_i].set_visible(False)

    fig.suptitle(f"2D adaptive pulls  ({args.fit_json})", fontsize=11)
    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight",
                format=args.format)
    plt.close(fig)
    print(f"  saved {args.output}")

    # ── Helper: combine the 4 permutations for a panel ──────────
    def combined_scatter(pair_key):
        xs, ys = [], []
        for xc, yc in perm_cols[pair_key]:
            xs.append(d[:, xc])
            ys.append(d[:, yc])
        return np.concatenate(xs), np.concatenate(ys)

    # ── Separate figure: full data scatter (4 perms combined) ──
    d = f._data_np["mass"]
    out_base = os.path.splitext(args.output)[0]
    fig2, axes2 = plt.subplots(n_rows, n_cols,
                               figsize=(6 * n_cols, 5.5 * n_rows),
                               squeeze=False)
    for ax_i, (i, j) in enumerate(pairs):
        ax = axes2.flat[ax_i]
        x_range, y_range = pair_ranges[(i, j)]
        sx, sy = combined_scatter((i, j))
        ax.scatter(sx, sy, s=1, c="black")
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_xlabel(pair_labels[(i, j)][0])
        ax.set_ylabel(pair_labels[(i, j)][1])
        ax.grid(True, alpha=0.3)
    for ax_i in range(len(pairs), n_rows * n_cols):
        axes2.flat[ax_i].set_visible(False)
    fig2.suptitle(f"data scatter (4 perms)  ({args.fit_json})", fontsize=11)
    plt.tight_layout()
    out2 = f"{out_base}_scatter.png"
    fig2.savefig(out2, dpi=300, bbox_inches="tight", format=args.format)
    plt.close(fig2)
    print(f"  saved {out2}")

    # ── Zoomed scatter: all panels focused on mass < 1.8 ────────
    x_hi = 1.8
    zr_pipi = (2 * m_pi, x_hi)    # pi+pi- zoom range
    zr_3pi = (3 * m_pi, x_hi)     # 3pi zoom range
    zoom_ranges = {
        (0, 1): (zr_pipi, zr_pipi),
        (2, 3): (zr_3pi, zr_pipi),
        (4, 5): (zr_3pi, zr_pipi),
    }
    fig3, axes3 = plt.subplots(n_rows, n_cols,
                               figsize=(6 * n_cols, 5.5 * n_rows),
                               squeeze=False)
    for ax_i, (i, j) in enumerate(pairs):
        ax = axes3.flat[ax_i]
        x_range, y_range = zoom_ranges[(i, j)]
        sx, sy = combined_scatter((i, j))
        ax.scatter(sx, sy, s=1, c="black")
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_xlabel(pair_labels[(i, j)][0])
        ax.set_ylabel(pair_labels[(i, j)][1])
        ax.grid(True, alpha=0.3)
    for ax_i in range(len(pairs), n_rows * n_cols):
        axes3.flat[ax_i].set_visible(False)
    fig3.suptitle(f"zoom scatter (4 perms), mass < {x_hi}  ({args.fit_json})",
                  fontsize=11)
    plt.tight_layout()
    out3 = f"{out_base}_scatter_zoom.png"
    fig3.savefig(out3, dpi=300, bbox_inches="tight", format=args.format)
    plt.close(fig3)
    print(f"  saved {out3}")


if __name__ == "__main__":
    main()
