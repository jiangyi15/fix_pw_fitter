#!/usr/bin/env python3
"""
Plot partial-wave group contributions from a fit result.

Usage:
    python scripts/plot_pw_groups.py fit_results.json -o plots/
"""

import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ampfit import Fitter
from ampfit.plot_pw_groups import PWGroupPlotter, discover_groups
from run_fit import build_constraints


def main():
    ap = argparse.ArgumentParser(
        description="Plot partial-wave group distributions")
    ap.add_argument("fit_json")
    ap.add_argument("--config", default="config_amp.yml")
    ap.add_argument("--data", default="data/data_arrays.npz")
    ap.add_argument("--phsp", default="data/phsp_arrays.npz")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--backend", default="cuda_v3")
    ap.add_argument("-o", "--output", default="plots/")
    args = ap.parse_args()

    f = Fitter(args.config, backend=args.backend)
    cp = os.path.splitext(args.fit_json)[0] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)
    else:
        fs, sp, sc = build_constraints(f.all_comb)
        f.set_fixed(fs); f.set_same(sp); f.set_scale(sc)

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

    os.makedirs(args.output, exist_ok=True)

    # Mass
    nm = f._data_np["mass"].shape[1] // 8
    plotter.plot_var(
        lambda x: [x["mass"][:, i] for i in range(nm)],
        [f"mass[{i}]" for i in range(nm)],
        0.2, 5.2, 100, "mass", legend=True, output=args.output)

    # Angles
    def angle_var(x):
        a = x["angle"].reshape(x["angle"].shape[0], -1, 3)
        out = []
        for pos in range(3):
            out.append((a[:, pos, 0] + np.pi) % (2 * np.pi) - np.pi)
            out.append(np.cos(a[:, pos, 1]))
            out.append(np.cos(a[:, pos, 2]))
        return out
    ar = [(-np.pi, np.pi), (-1, 1), (-1, 1)] * 3
    al = [f"angle[{p},{c}]" for p in range(3) for c in range(3)]
    plotter.plot_var(angle_var, al, 0, 1, 50, "angles",
                     ranges=ar, output=args.output)

    # Time
    plotter.plot_var(lambda x: [x["time"]], ["time"], 0, 10, 50, "time",
                     output=args.output)


if __name__ == "__main__":
    main()
