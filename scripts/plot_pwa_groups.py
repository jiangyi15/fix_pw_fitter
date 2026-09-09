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
from ampfit.pwa_build import pwa_event_data_tree
from ampfit.plot_pwa_groups import (
    config_plot_items, discover_pwa_groups, pwa_mass_varfun,
    pwa_angle_varfun, angle_variable_labels, var_ranges)


def _load_event_arrays(cfg, kc, path, n_comp):
    """Kernel event arrays from an .npz (kernel arrays) or a 4-momentum .npy."""
    if str(path).endswith(".npz"):
        ev, _ = Fitter.load_npz(path, n_angle_comp=n_comp)
    else:
        pws = list(cfg.full_decay.get_partial_waves())
        byt = {cfg.topo_index[ch.topo_id()]: ch for _, ch in pws}
        mom = np.load(path)
        ev = pwa_event_data_tree(cfg, kc, byt, mom)
    return ev


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
    ap.add_argument("--mc-uncert", action="store_true",
                    help="include the MC (signal/background) statistical "
                         "uncertainty per bin in the pull/chi2 errors "
                         "(sigma^2 = sum_data w^2 + sum_sig w^2 + sum_bkg w^2)")
    ap.add_argument("--no-pull", action="store_true",
                    help="do not draw the (data-model)/sigma pull row "
                         "below every variable panel")
    ap.add_argument("-o", "--output", default=None,
                    help="output dir (default plots_pwa/, or plots_pwa_rec/ "
                         "when rec sources are used)")
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
    # Rec mode (no explicit files + config declares data_rec): the loader
    # data comes from ``data_rec`` (original rows, *_weight/*_bg sidecars
    # kept) while the phsp stays the config ``phsp`` — the WEIGHT source.
    # The ``phsp_rec`` rows (when declared) are only handed to plot_var as
    # the VARIABLE source through ``PWGroupPlotter.use_rec``.
    if bool(args.data) != bool(args.phsp):
        sys.exit("give both --data and --phsp, or neither (falls back to "
                 "the config data section)")
    rec_used = False
    dc = f.config.dic["data"]
    if not args.data and dc.get("data_rec"):
        rec_used = True
        dc["data"] = dc["data_rec"]     # keep *_weight / *_bg_value sidecars
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
              f"{len(phsp_np['weight']):,} phsp events from config "
              f"{'(rec data rows)' if rec_used else ''}")

    r = f.load_results(args.fit_json)
    if r.x is None or len(r.x) == 0:
        sys.exit("no parameter vector in the fit result")
    if not hasattr(f.config, "full_decay"):
        sys.exit("plot_pwa_groups needs a config with full_decay "
                 "(pure-PWA mode)")

    groups = discover_pwa_groups(f.config, by=args.by, merge=merge)
    plotter = PWGroupPlotter(f, r, groups)
    if rec_used:
        # weights stay on the config phsp (f._phsp_np); the ORIGINAL rows
        # (phsp_rec, or phsp itself when absent) are only the VARIABLE
        # source read by plot_var via use_rec.
        pv = f._phsp_np
        pr = dc.get("phsp_rec")
        if pr:
            pv = _load_event_arrays(f.config, kc, pr, n_comp)
            if pv["weight"].shape[0] != f._phsp_np["weight"].shape[0]:
                sys.exit("phsp_rec rows must equal the weight phsp rows "
                         "(phsp may carry no resolution size)")
        plotter.use_rec(data_rec_np=f._data_np, phsp_rec_np=pv)
    plotter.compute()
    print(f"  {len(plotter.labels)} groups ({args.by}): {plotter.labels}")

    args.output = args.output or ("plots_pwa_rec/" if rec_used
                                  else "plots_pwa/")
    os.makedirs(args.output, exist_ok=True)

    # ── one figure per config-plot variable (ReadVar items) ─────────
    items = config_plot_items(f.config, data_np, phsp_np)
    if items:
        print("  variables from config plot section (one figure each): " +
              ", ".join(it["key"] for it in items))
    for it in items:
        lo, hi = it["range"]
        plotter.plot_var(
            it["varfun"], [it["label"]], lo, hi, it["width"], it["stem"],
            output=args.output, fmt=args.format, unit=it["unit"],
            smooth_sigma=args.smooth, legend=it["legend"],
            ranges=[it["range"]], show_pull=not args.no_pull,
            mc_uncert=args.mc_uncert)
    if not items:
        # fall back to every kernel column of the loaded arrays
        n_mass = data_np["mass"].shape[1]
        if n_mass:
            plotter.plot_var(
                pwa_mass_varfun,
                [f"mass[{i}]" for i in range(n_mass)],
                0.2, 5.2, args.mass_bin, "mass", output=args.output,
                fmt=args.format, unit="GeV",
                smooth_sigma=args.smooth, legend=True,
                ranges=var_ranges(data_np, phsp_np, pwa_mass_varfun),
                show_pull=not args.no_pull, mc_uncert=args.mc_uncert)
        n_var = len(pwa_angle_varfun(data_np))
        if n_var:
            plotter.plot_var(
                pwa_angle_varfun,
                angle_variable_labels(data_np),
                -np.pi, np.pi, args.angle_bin, "angles",
                output=args.output, fmt=args.format, unit="", legend=True,
                smooth_sigma=args.smooth, show_pull=False,
                mc_uncert=args.mc_uncert,
                ranges=var_ranges(data_np, phsp_np, pwa_angle_varfun))

    print(f"  saved to {args.output}")


if __name__ == "__main__":
    main()
