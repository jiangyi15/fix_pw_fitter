#!/usr/bin/env python3
"""End-to-end pure-PWA fit: same YAML config + backend family.

Demonstrates the full pipeline for a single-block projection-sum PWA
(e.g. config_pwa.yml, J/ψ → π⁺π⁻η):

    Config -> build_all_index()           (generic, any J/proj/angles)
    momenta -> pwa_event_data()           (canonical kernel buffers)
    norm      -> integrated_pwa backend   (Gram matrix)
    data NLL  -> base backend             (cuda_v4_pwa or numpy_pwa)
    BFGS      -> recovers the ck used to generate the toy data

Usage:
    python fit_pwa.py [--config config_pwa.yml] [--cuda]
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from ampfit.config_loader import Config
from ampfit.pwa_build import pwa_event_data
from ampfit.numpy_pwa import NumpyPWA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_pwa.yml")
    ap.add_argument("--cuda", action="store_true",
                    help="use cuda_v4_pwa as the data base backend")
    ap.add_argument("--nph", type=int, default=4000)
    ap.add_argument("--nprop", type=int, default=10000)
    ap.add_argument("--ndata", type=int, default=300)
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--method", default="L-BFGS-B")
    args = ap.parse_args()

    cfg = Config(args.config)
    kc = cfg.build_all_index()
    N = kc["matrix_angle"].shape[1] // kc["n_proj"]   # shared ck length
    print(f"model {cfg.top} n_wave {kc['matrix_angle'].shape[1]} "
          f"n_proj {kc['n_proj']}  N {N}")

    # ── phsp / data buffers (canonical layout) ────────────────────────────
    mom = np.load("data/phsp.npy")
    phsp_mom = mom[:args.nph]
    phsp = pwa_event_data(cfg, kc, phsp_mom)

    # toy data: events importance-sampled from the model at a random ck0
    rng = np.random.RandomState(42)
    n_m0 = int(np.max(kc["m0_index"])) + 1
    n_g0 = int(np.max(kc["g0_index"])) + 1
    ck0 = rng.normal(size=N) + 1j * rng.normal(size=N)
    m0 = np.full(n_m0, 0.769)
    g0 = np.full(n_g0, 0.10)
    npw = NumpyPWA(kc)
    _, _, P0 = npw.compute({"ck": ck0, "m0": m0, "g0": g0},
                           npw.load_data(phsp))
    prob = np.clip(P0, 0, None)
    if prob.sum() <= 0:
        raise SystemExit("model probability zero on phsp sample")
    prob /= prob.sum()
    idx = rng.choice(args.nph, size=args.ndata, p=prob)
    data = {kk: vv[idx] for kk, vv in phsp.items()}

    # ── backend: integrated_pwa (Gram norm) + base ────────────────────────
    from ampfit.backends import create_backend
    base = "cuda_v4_pwa" if args.cuda else "numpy_pwa"
    be = create_backend({"name": "integrated_pwa", "base": base}, kc)
    ph = be.load_data(phsp)
    dh = be.load_data(data)
    phsp_wsum = float(phsp["weight"].sum())
    phsp_n = dict(phsp)
    phsp_n["weight"] = phsp["weight"] / phsp_wsum   # weights sum to 1
    del ph
    ph = be.load_data(phsp_n)

    # ── loss over the ck RATIOS (NLL invariant under a global complex ck
    # scale, so ck[0] is anchored to 1: 4 free real params for N=3) ─
    M = N - 1

    def params_from(x):
        ck = np.concatenate([[1 + 0j], x[:M] + 1j * x[M:]])
        return {"ck": ck, "m0": m0, "g0": g0}

    def loss(x):
        params = params_from(x)
        norm, _, _ = be.compute(params, ph, norm=None, return_p=False)
        Q, _, P = be.compute(params, dh, norm=norm)
        return float(Q)

    from scipy.optimize import minimize
    rng0 = np.random.RandomState(0)
    x0 = rng0.normal(scale=0.3, size=2 * M)

    f0 = loss(x0)
    print(f"start NLL {f0:.3f}   (true ck used for toy data)")
    kw = {} if args.method == "Powell" else {"jac": "2-point"}
    res = minimize(loss, x0, method=args.method, options={"maxiter": args.maxiter}, **kw)
    ck_fit = params_from(res.x)["ck"]
    print(f"fit  NLL {res.fun:.3f}  success={res.success}  "
          f"nit={res.nit}  msg={res.message}")
    z_true = ck0 / ck0[0]
    print("ck_true (rel to ck0[0])", np.round(z_true, 3))
    print("ck_fit  (ck[0]=1)      ", np.round(ck_fit, 3))
    print("ck corr", float(np.abs(np.vdot(z_true, ck_fit))
                           / (np.linalg.norm(z_true) * np.linalg.norm(ck_fit))))


if __name__ == "__main__":
    main()
