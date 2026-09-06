#!/usr/bin/env python3
"""End-to-end pure-PWA fit through the standard Fitter pipeline.

Same YAML config + backend family as any legacy model.  All gradients are
ANALYTIC — kernel backward (numpy_pwa / cuda_v4_pwa) → norm chain
(integrated_pwa Gram) → CKProduct (r, θ) → ConstraintManager transform.
No numerical Jacobian anywhere.

Data: flat phase space generated on the fly with the config masses
(generate_pwa_phsp); toy events drawn without replacement from an
independent proposal sample weighted by the model density at a seed ck.

Usage:
    python fit_pwa.py [--cuda] [--nph 8000] [--ndata 8000]
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from ampfit.config_loader import Config
from ampfit.pwa_build import pwa_event_data, generate_pwa_phsp
from ampfit.numpy_pwa import NumpyPWA
from ampfit import Fitter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_pwa.yml")
    ap.add_argument("--cuda", action="store_true",
                    help="use cuda_v4_pwa (default numpy_pwa)")
    ap.add_argument("--nph", type=int, default=8000)
    ap.add_argument("--nprop", type=int, default=40000)
    ap.add_argument("--ndata", type=int, default=8000)
    ap.add_argument("--maxiter", type=int, default=60)
    args = ap.parse_args()

    cfg = Config(args.config)
    kc = cfg.build_all_index()
    N = kc["matrix_angle"].shape[1] // kc["n_proj"]
    print(f"model {cfg.top}  n_wave {kc['matrix_angle'].shape[1]}  "
          f"n_proj {kc['n_proj']}  N(ck) {N}")

    # ── independent flat samples: norm phsp + toy proposal ────────────────
    chain = cfg.full_decay.get_partial_waves()[0][1]
    phsp = pwa_event_data(cfg, kc,
                          generate_pwa_phsp(cfg, chain, args.nph, seed=11))
    prop = pwa_event_data(cfg, kc,
                          generate_pwa_phsp(cfg, chain, args.nprop, seed=22))

    rng = np.random.RandomState(42)
    n_m0 = int(np.max(kc["m0_index"])) + 1
    n_g0 = int(np.max(kc["g0_index"])) + 1
    ck0 = rng.normal(size=N) + 1j * rng.normal(size=N)
    m0 = np.full(n_m0, 0.769)
    g0 = np.full(n_g0, 0.10)

    npw = NumpyPWA(kc)
    _, _, P0 = npw.compute({"ck": ck0, "m0": m0, "g0": g0},
                           npw.load_data(prop))
    if P0 is None:
        raise SystemExit("model compute returned no P")
    prob = np.clip(np.asarray(P0, dtype=float), 0, None)
    if prob.sum() <= 0:
        raise SystemExit("zero model probability on the proposal sample")
    prob /= prob.sum()
    idx = rng.choice(args.nprop, size=args.ndata, replace=False, p=prob)
    data = {kk: vv[idx] for kk, vv in prop.items()}

    # ── standard Fitter: analytic gradients end to end ────────────────────
    backend = "cuda_v4_pwa" if args.cuda else "numpy_pwa"
    fitter = Fitter(args.config, backend=backend)
    fitter.set_phsp(phsp)
    fitter.set_data(data)
    fitter.apply_constrains()

    # normalization terms fixed to 1; couplings start at (r=1, θ=0)
    start = {}
    for comb in fitter.all_comb:
        for p in comb:
            if not isinstance(p, str):
                continue
            if "_total_0" in p:
                start[p] = 1.0
            else:
                start.setdefault(p + "r", 1.0)
                start.setdefault(p + "i", 0.0)
    x0 = fitter.values_from_dict(start)

    nll0, _ = fitter.get_nll(x0)
    print(f"start NLL {nll0:.3f}")
    res = fitter.fit(x0, maxiter=args.maxiter)
    nll1 = res.fun if hasattr(res, "fun") and res.fun is not None else \
        fitter.get_nll(res.x)[0]
    print(f"fit  NLL {nll1:.3f}  success={res.success}  nit={res.nit}")
    print(f"analytic-gradient fit via Fitter (kernel → norm → CKProduct → "
          f"constraints); no numerical Jacobian used.")


if __name__ == "__main__":
    main()
