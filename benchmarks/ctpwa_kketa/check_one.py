#!/usr/bin/env python3
"""One backend, one evaluation: dump Q and P so two backends can be compared.

    python check_one.py --backend cuda_v4_pwa_s0e --out /tmp/x.npz

Each backend runs in its own process (never two CUDA contexts in one process).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from tabpwa import Fitter            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yml"))
    ap.add_argument("--backend", required=True)
    ap.add_argument("--max-events", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    f = Fitter(a.config, backend=a.backend)
    phsp = f.load_dataset("phsp", max_events=a.max_events)
    data = f.load_dataset("data", max_events=a.max_events)
    f.set_phsp(phsp)
    f.set_data(data)
    f.apply_constrains()
    mw = {n: float(f.defaults[n])
          for n in dict.fromkeys(f.var_registry.flat_names)
          if n.endswith(("_mass", "_width")) and n in f.defaults}
    f.set_fixed(mw)

    x0 = f.initial_values(seed=1)
    params, _ = f.build_params(x0)
    h = f.backend.load_data(data)
    Q, _g, P = f.backend.compute(params, h, norm=1.0)
    nll, _ = f.get_nll(x0)
    np.savez(a.out, Q=np.asarray(Q, dtype=float), P=np.asarray(P, dtype=float),
             nll=np.asarray(nll, dtype=float), x=x0, ck=np.asarray(params["ck"]))
    print(f"{a.backend:22s} x={len(x0):3d}  Q={float(np.asarray(Q).sum()):+.10e}  "
          f"nll={float(nll):+.10e}  P[0:3]={np.asarray(P)[:3]}", flush=True)
    f.backend.free()


if __name__ == "__main__":
    main()
