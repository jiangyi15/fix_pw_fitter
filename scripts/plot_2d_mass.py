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
from ampfit.plot_pw_groups import PWGroupPlotter, discover_groups


def main():
    ap = argparse.ArgumentParser(description="2D mass pull plots")
    ap.add_argument("fit_json", help="JSON from Fitter.save_params()")
    ap.add_argument("--config", default="config_amp.yml")
    ap.add_argument("--data", default="data/data_arrays.npz")
    ap.add_argument("--phsp", default="data/phsp_arrays.npz")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--backend", default="cuda_v3")
    ap.add_argument("-o", "--output", default="mass_2d.png")
    ap.add_argument("--binning", default="[[2,2]]*3",
                    help="Adaptive binning spec (default [[2,2]]*3)")
    ap.add_argument("--format", default="png",
                    help="Image format (default: png)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Parse binning spec like "[[2,2]]*3"
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

    data_np, nd = Fitter.load_npz(args.data, max_events=args.max_events)
    phsp_np, np_ = Fitter.load_npz(args.phsp, max_events=args.max_events)
    print(f"  Loaded {nd:,} data + {np_:,} phsp events")
    f.set_phsp(phsp_np)
    f.set_data(data_np)

    r = f.load_results(args.fit_json)
    if r.x is None or len(r.x) == 0:
        sys.exit(1)

    groups = discover_groups(f.config)
    plotter = PWGroupPlotter(f, r, groups).compute()
    print(f"  {len(plotter.labels)} groups, scale={plotter._scale:.4f}")

    # ── Pairs of mass columns ───────────────────────────────────
    # Column order: [pipi1, pipi2], [pipipip, pipi1], [pipipim, pipi1]
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

    n_cols = min(3, len(pairs))
    n_rows = (len(pairs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 5.5 * n_rows), squeeze=False)

    for ax_i, (i, j) in enumerate(pairs):
        ax = axes.flat[ax_i]
        plotter.plot_2d(
            lambda x, a=i, b=j: [x["mass"][:, a], x["mass"][:, b]],
            list(pair_labels[(i, j)]),
            f"mass_{i}_{j}", binning=binning, output=None, ax=ax)

    for ax_i in range(len(pairs), n_rows * n_cols):
        axes.flat[ax_i].set_visible(False)

    fig.suptitle(f"2D adaptive pulls  ({args.fit_json})", fontsize=11)
    plt.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight",
                format=args.format)
    plt.close(fig)
    print(f"  saved {args.output}")


if __name__ == "__main__":
    main()
