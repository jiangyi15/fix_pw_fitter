#!/usr/bin/env python3
"""Generate pure-PWA toy data for the generic run_fit driver.

Writes kernel-format ``.npz`` files (data + phsp) plus an ``init.json``
start point (constraint-driven: '_total_0' -> 1, couplings (r=1, θ=0)),
so run_fit needs no --toy flag:

    python gen_toy_pwa.py --config config_pwa.yml
    python run_fit.py --config config_pwa.yml --backend numpy_pwa \\
        --data data_pwa.npz --phsp phsp_pwa.npz --init init_pwa.json --fit

Flat phase space is generated with the config masses (two-body product +
inverse boost chain); toy events are drawn without replacement from an
independent proposal sample weighted by the model density at a seed ck.
"""
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from ampfit.config_loader import Config
from ampfit.pwa_build import pwa_event_data, generate_pwa_phsp
from ampfit.numpy_pwa import NumpyPWA


def _npz(data):
    """Kernel-format npz dict — only the keys the PWA path actually needs
    (mass/q/angle/weight).  time/frac/bkg_raw are legacy-mixing inputs and
    are omitted to save disk + load memory; the loader fills defaults."""
    return {
        "mass": data["mass"],
        "q": data["q"],
        "angle": data["angle"],
        "weight": data["weight"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_pwa.yml")
    ap.add_argument("--nph", type=int, default=8000)
    ap.add_argument("--nprop", type=int, default=40000)
    ap.add_argument("--ndata", type=int, default=4000)
    ap.add_argument("--out-data", default="data_pwa.npz")
    ap.add_argument("--out-phsp", default="phsp_pwa.npz")
    ap.add_argument("--out-init", default="init_pwa.json")
    args = ap.parse_args()

    cfg = Config(args.config)
    kc = cfg.build_all_index()
    chain = cfg.full_decay.get_partial_waves()[0][1]

    # ── independent flat samples ───────────────────────────────────────────
    phsp = pwa_event_data(cfg, kc,
                          generate_pwa_phsp(cfg, chain, args.nph, seed=11))
    prop = pwa_event_data(cfg, kc,
                          generate_pwa_phsp(cfg, chain, args.nprop, seed=22))

    rng = np.random.RandomState(42)
    n_ck = kc["matrix_angle"].shape[1] // kc["n_proj"]
    ck0 = rng.normal(size=n_ck) + 1j * rng.normal(size=n_ck)
    m0 = np.full(int(np.max(kc["m0_index"])) + 1, 0.769)
    g0 = np.full(int(np.max(kc["g0_index"])) + 1, 0.10)

    npw = NumpyPWA(kc)
    _, _, P0 = npw.compute({"ck": ck0, "m0": m0, "g0": g0},
                           npw.load_data(prop))
    if P0 is None:
        raise SystemExit("model compute returned no P")
    prob = np.clip(np.asarray(P0, dtype=float), 0, None)
    prob /= prob.sum()
    idx = rng.choice(args.nprop, size=args.ndata, replace=False, p=prob)
    data = {kk: vv[idx] for kk, vv in prop.items()}

    np.savez(args.out_data, **_npz(data))
    np.savez(args.out_phsp, **_npz(phsp))
    print(f"wrote {args.out_data} ({args.ndata} events), "
          f"{args.out_phsp} ({args.nph} events)")

    # ── constraint-driven start point for run_fit --init ──────────────────
    init = {}
    for comb in cfg.full_decay.get_partial_waves_params():
        for p in comb:
            if not isinstance(p, str):
                continue
            if "_total_0" in p:
                init[p] = 1.0
            else:
                init.setdefault(p + "r", 1.0)
                init.setdefault(p + "i", 0.0)
    with open(args.out_init, "w") as f:
        json.dump(init, f, indent=1)
    print(f"wrote {args.out_init} ({len(init)} entries)")


if __name__ == "__main__":
    main()
