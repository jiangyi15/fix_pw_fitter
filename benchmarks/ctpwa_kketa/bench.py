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
    ap.add_argument("--fix-mw", action="store_true",
                    help="fix every mass/width at its config value (the paper's "
                         "first column; BWR masses/widths are FREE by default)")
    ap.add_argument("--free-mw", default="",
                    help="comma-separated mass/width names to leave FREE "
                         "(implies --fix-mw for the rest)")
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
    if args.fix_mw or args.free_mw:
        keep = {s.strip() for s in args.free_mw.split(",") if s.strip()}
        mw = {n: float(f.defaults[n])
              for n in dict.fromkeys(f.var_registry.flat_names)
              if n.endswith(("_mass", "_width")) and n not in keep
              and n in f.defaults}
        f.set_fixed(mw)
    x0 = f.initial_values(seed=1)
    t_load = time.perf_counter() - t0

    n_free = len(list(f.pc.free_param_names()))
    n_x = len(x0)
    f.get_nll(x0)                                   # warm-up
    ts = []
    nll = float("nan")
    for _ in range(args.repeat):
        a = time.perf_counter()
        nll, _g = f.get_nll(x0)
        ts.append(time.perf_counter() - a)
    t = min(ts)
    print(f"{args.backend:28s} x={n_x:3d} ne={ne:>9,} nphsp={nb:>9,}  "
          f"{t*1e3:8.1f} ms/iter  {(ne+nb)/t/1e6:6.2f} Mev/s  NLL={nll:+.6f}  "
          f"(load {t_load:.0f} s)", flush=True)
    f.backend.free()


if __name__ == "__main__":
    main()
