#!/usr/bin/env python3
"""
Partial-wave group plot for pure-PWA mode (P = Σ_p |A_p|², shared ck).

Groups (per chain / per resonance / per LS wave) are drawn as smoothed
intensity curves overlaid on the data, following the exact convention of
the legacy ``plot_pw_groups`` (reduced ck model per group, global scale,
bkg from ``bkg`` weights/purity).  Panels are the *generic* event
variables: every mass column and every canonical φ-first angle component.

Usage:
    python scripts/plot_pwa_groups.py results.json \\
        --config config_pwa.yml \\
        --data data_arrays.npz --phsp phsp_arrays.npz \\
        --by resonance -o plots_pwa/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import argparse

import numpy as np

from ampfit import Fitter
from ampfit.plot_pw_groups import PWGroupPlotter
from ampfit.plot_pwa_groups import (
    discover_pwa_groups, pwa_mass_varfun, pwa_angle_varfun,
    angle_variable_labels, var_ranges)


def main():
    ap = argparse.ArgumentParser(
        description="Plot partial-wave group distributions (pure-PWA mode)")
    ap.add_argument("fit_json")
    ap.add_argument("--config", default="config_pwa.yml")
    ap.add_argument("--data",
                    help="data .npz (kernel arrays); omit with --phsp to "
                         "use the config's data/phsp section")
    ap.add_argument("--phsp",
                    help="phsp .npz (kernel arrays); omit with --data to "
                         "use the config's data/phsp section")
    ap.add_argument("--backend", default="numpy_pwa",
                    help="backend computing the pwa model (default numpy_pwa)")
    ap.add_argument("--by", default="resonance",
                    choices=["chain", "resonance", "ls"],
                    help="group policy (default: resonance)")
    ap.add_argument("--merge", action="append", default=None,
                    metavar="REGEX=LABEL",
                    help="merge matching group keys into LABEL "
                         "(repeatable, e.g. '^MI0\\\\d=MI0')")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--mass-bin", type=float, default=0.05)
    ap.add_argument("--angle-bin", type=float, default=0.10)
    ap.add_argument("--smooth", type=float, default=1.0,
                    help="Gaussian smoothing sigma in units of bin width")
    ap.add_argument("--no-pull", action="store_true",
                    help="do not draw the (data-model)/sigma pull row "
                         "below each mass panel")
    ap.add_argument("-o", "--output", default="plots_pwa/")
    ap.add_argument("--format", default="png")
    args = ap.parse_args()

    merge = None
    if args.merge:
        merge = []
        for item in args.merge:
            pat, label = item.split("=", 1)
            merge.append((pat, label))

    f = Fitter(args.config, backend=args.backend)
    kc = f.config.build_all_index()
    n_comp = (int(len(kc["variables"])) if "variables" in kc else 3)
    cp = os.path.splitext(args.fit_json)[0] + "_constraints.json"
    if os.path.exists(cp):
        f.load_constraints(cp)

    # ── data / phsp sources ────────────────────────────────────────
    # Provide --data/--phsp npz files explicitly, or omit both and the
    # config's ``data`` / ``phsp`` section (npz arrays or 4-momentum
    # prefix + _weight[/_bg_value] files) is used via load_all_data().
    if bool(args.data) != bool(args.phsp):
        sys.exit("give both --data and --phsp, or neither (falls back to "
                 "the config data section)")
    if args.data:
        data_np, nd = Fitter.load_npz(args.data,
                                      max_events=args.max_events,
                                      n_angle_comp=n_comp)
        phsp_np, np_ = Fitter.load_npz(args.phsp,
                                       max_events=args.max_events,
                                       n_angle_comp=n_comp)
        print(f"  Loaded {nd:,} data + {np_:,} phsp events")
        f.set_phsp(phsp_np)
        f.set_data(data_np)
    else:
        data_np, phsp_np = f.load_all_data()
        print(f"  Loaded {len(data_np['weight']):,} data + "
              f"{len(phsp_np['weight']):,} phsp events from config")

    r = f.load_results(args.fit_json)
    if r.x is None or len(r.x) == 0:
        sys.exit("no parameter vector in the fit result")
    if not hasattr(f.config, "full_decay"):
        sys.exit("plot_pwa_groups needs a config with full_decay "
                 "(pure-PWA mode)")

    groups = discover_pwa_groups(f.config, by=args.by, merge=merge)
    plotter = PWGroupPlotter(f, r, groups).compute()
    print(f"  {len(plotter.labels)} groups ({args.by}): {plotter.labels}")

    os.makedirs(args.output, exist_ok=True)

    # ── mass panels (all mass columns) ──────────────────────────────
    n_mass = data_np["mass"].shape[1]
    if n_mass:
        plotter.plot_var(
            pwa_mass_varfun,
            [f"mass[{i}]" for i in range(n_mass)],
            0.2, 5.2, args.mass_bin, "mass", output=args.output,
            fmt=args.format, unit="GeV",
            smooth_sigma=args.smooth, legend=True,
            ranges=var_ranges(data_np, phsp_np, pwa_mass_varfun),
            show_pull=not args.no_pull)

    # ── angle panels (canonical φ-first components) ─────────────────
    n_var = len(pwa_angle_varfun(data_np))
    if n_var:
        plotter.plot_var(
            pwa_angle_varfun,
            angle_variable_labels(data_np),
            -np.pi, np.pi, args.angle_bin, "angles", output=args.output,
            fmt=args.format, unit="", legend=True,
            smooth_sigma=args.smooth, show_pull=False,
            ranges=var_ranges(data_np, phsp_np, pwa_angle_varfun))

    print(f"  saved to {args.output}")


if __name__ == "__main__":
    main()
