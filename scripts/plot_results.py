#!/usr/bin/env python3
"""
Plot data vs model distributions from a fit result.

Usage:
    python scripts/plot_results.py fit_results.json -o plots/
    python scripts/plot_results.py fit_results.json -o plots/ --show
"""

import sys, os, argparse
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ampfit import Fitter


def main():
    ap = argparse.ArgumentParser(description="Plot distributions from fit result")
    ap.add_argument("fit_json", help="JSON from Fitter.save_params()")
    ap.add_argument("--config", default="config_amp.yml", help="Config YAML")
    ap.add_argument("--data", default="data/data_arrays.npz", help="Data NPZ")
    ap.add_argument("--phsp", default="data/phsp_arrays.npz", help="Phase-space NPZ")
    ap.add_argument("--max-events", type=int, default=None,
                    help="Limit events (faster testing)")
    ap.add_argument("--backend", default="cuda_v3", help="Compute backend")
    ap.add_argument("-o", "--output", default="plots/",
                    help="Output directory for plots (default: plots/)")
    ap.add_argument("--show", action="store_true", help="Call plt.show()")
    ap.add_argument("--n-bins", type=int, default=50, help="Number of bins")
    args = ap.parse_args()

    # ── Setup fitter ──────────────────────────────────────────────
    f = Fitter(args.config, backend=args.backend)

    # Auto-detect and load constraints
    constraints_path = os.path.splitext(args.fit_json)[0] + "_constraints.json"
    if os.path.exists(constraints_path):
        f.load_constraints(constraints_path)
    else:
        print("WARNING: no constraints file found, using defaults")
        from run_fit import build_constraints
        fs, sp, sc = build_constraints(f.all_comb)
        f.set_fixed(fs)
        f.set_same(sp)
        f.set_scale(sc)

    # ── Load data ─────────────────────────────────────────────────
    data_np, n_data = Fitter.load_npz(args.data, max_events=args.max_events)
    phsp_np, n_phsp = Fitter.load_npz(args.phsp, max_events=args.max_events)
    print(f"  Loaded {n_data:,} data + {n_phsp:,} phsp events")

    f.set_phsp(phsp_np)
    f.set_data(data_np)

    # ── Load fit result ───────────────────────────────────────────
    fit_result = f.load_results(args.fit_json)
    if fit_result.x is None or len(fit_result.x) == 0:
        print("ERROR: could not reconstruct x from", args.fit_json)
        sys.exit(1)

    x0 = fit_result.x
    nll, _ = f.get_nll(x0)
    print(f"  NLL at fit result: {nll:.4f}")

    # ── Plot ──────────────────────────────────────────────────────
    f.plot(result=x0, prefix=args.output, n_bins=args.n_bins, show=args.show)
    print(f"  Plots saved to {args.output}/")


if __name__ == "__main__":
    main()
