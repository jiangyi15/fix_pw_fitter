#!/usr/bin/env python3
"""
Plot partial-wave group contributions from a fit result.

Usage:
    python scripts/plot_pw_groups.py fit_results.json -o plots/
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ampfit.plot_pw_groups import discover_groups
from ampfit.plot_pw_common import run


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
    ap.add_argument("--format", default="png",
                    help="Image format (default: png)")
    ap.add_argument("--merge-mi", action="store_true",
                    help="Merge MI0{i} basis particles into a single MI0 group")
    args = ap.parse_args()

    merge = [("^MI0\\d", "MI0")] if args.merge_mi else None
    if merge:
        print("  merged MI0{i} -> MI0")

    def groups(config):
        return discover_groups(config, merge=merge)

    run(args.fit_json, args.config, args.data, args.phsp,
        args.max_events, args.backend, args.output, args.format,
        groups, "groups")


if __name__ == "__main__":
    main()
