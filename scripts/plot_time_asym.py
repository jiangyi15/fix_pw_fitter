#!/usr/bin/env python3
"""
Plot time-dependent CP asymmetry from a fit result.

Uses the theoretical formula with integrals I, Ī, J computed from
the kernel, and data asymmetries binned in time.

Usage:
    python scripts/plot_time_asym.py fit_results.json -o plots/
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ampfit import Fitter
from ampfit.plot_pw_groups import PWGroupPlotter, discover_groups


def main():
    ap = argparse.ArgumentParser(
        description="Plot time-dependent CP asymmetry")
    ap.add_argument("fit_json")
    ap.add_argument("--config", default="config_amp.yml")
    ap.add_argument("--data", default="data/data_arrays.npz")
    ap.add_argument("--phsp", default="data/phsp_arrays_frac.npz",
                    help="Phsp file with proper frac values")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--backend", default="cuda_v3")
    ap.add_argument("-o", "--output", default="plots/")
    ap.add_argument("--format", default="png",
                    help="Image format (default: png)")
    ap.add_argument("--n-bins", type=int, default=20,
                    help="Number of time bins")
    args = ap.parse_args()

    f = Fitter(args.config, backend=args.backend)
    cp = os.path.splitext(args.fit_json)[0] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)
    else:
        f.apply_constrains()
        fs2 = {}
        for name in f.config.m0_phys_name:
            fs2[name] = float(f.cm.defaults[name])
        for name in f.config.g0_phys_name:
            fs2[name] = float(f.cm.defaults[name])
        if fs2:
            f.set_fixed(fs2, reset=False)

    pn, _ = Fitter.load_npz(args.phsp, max_events=args.max_events)
    dn, _ = Fitter.load_npz(args.data, max_events=args.max_events)
    print(f"  Loaded {len(dn['weight']):,} data + {len(pn['weight']):,} phsp events")
    f.set_phsp(pn)
    f.set_data(dn)

    r = f.load_results(args.fit_json)
    if r.x is None or len(r.x) == 0:
        print("ERROR: could not reconstruct x from", args.fit_json)
        sys.exit(1)

    groups = discover_groups(f.config)
    plotter = PWGroupPlotter(f, r, groups).compute()

    os.makedirs(args.output, exist_ok=True)
    plotter.plot_time_asymmetry(t_min=0, t_max=10, n_bins=args.n_bins,
                                output=args.output, prefix="time_asym",
                                fmt=args.format)


if __name__ == "__main__":
    main()
