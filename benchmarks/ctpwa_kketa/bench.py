#!/usr/bin/env python3
"""Time one `Fitter.get_nll` (norm + data NLL + gradients) at the CTPWA size.

    python bench.py --backend cuda_flat_pwa_gen_s0e2 --rsize 17 --repeat 5

One backend per process, on the samples written by `make_data.py`.  Reports
ms/iteration, Mev/s over data + phase space, and the NLL, so a backend cannot
look fast by computing less.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from tabpwa import Fitter            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yml"))
    ap.add_argument("--backend", default="cuda_flat_pwa_gen_s0e2")
    ap.add_argument("--rsize", type=int, default=0,
                    help="0 = do not pass resolution_size (pwa family)")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--base", default=None,
                    help="composer base, e.g. cuda_v4_pwa_cache for integrated_pwa")
    ap.add_argument("--max-events", type=int, default=0)
    args = ap.parse_args()

    t0 = time.perf_counter()
    spec = {"name": args.backend}
    if args.base:
        spec["base"] = args.base
    if args.rsize:
        spec["resolution_size"] = args.rsize
    f = Fitter(args.config, backend=spec)
    kw = {"max_events": args.max_events} if args.max_events else {}
    phsp = f.load_dataset("phsp", **kw)
    data = f.load_dataset("data", **kw)
    f.set_phsp(phsp)
    f.set_data(data)
    ne, nb = len(data["weight"]), len(phsp["weight"])
    f.apply_constrains()
    x0 = f.initial_values(seed=1)
    t_load = time.perf_counter() - t0

    f.get_nll(x0)                                   # warm-up
    ts = []
    nll = float("nan")
    for _ in range(args.repeat):
        a = time.perf_counter()
        nll, _g = f.get_nll(x0)
        ts.append(time.perf_counter() - a)
    t = min(ts)
    print(f"{args.backend:28s} ne={ne:>9,} nphsp={nb:>9,}  "
          f"{t*1e3:8.1f} ms/iter  {(ne+nb)/t/1e6:6.2f} Mev/s  NLL={nll:+.6f}  "
          f"(load {t_load:.0f} s)", flush=True)
    f.backend.free()


if __name__ == "__main__":
    main()
