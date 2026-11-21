#!/usr/bin/env python3
"""Time one `Fitter.get_nll` (norm + data NLL + gradients) at the CTPWA size.

    python bench.py --backend cuda_flat_pwa_gen_s0e2 \
        --config config_gen.yml --rsize 17 --repeat 5

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
    ap.add_argument("--backend", default="cuda_v4_pwa_s0e",
                    help="a backend registered for --config's model; the flat "
                         "family needs --config config_gen.yml")
    ap.add_argument("--rsize", type=int, default=0,
                    help="0 = do not pass resolution_size (pwa family)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--base", default=None,
                    help="composer base, e.g. cuda_v4_pwa_cache for integrated_pwa")
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--max-data", type=int, default=0,
                    help="cap the data sample only (default: --max-events)")
    ap.add_argument("--max-phsp", type=int, default=0,
                    help="cap the phase-space sample only (default: --max-events)")
    ap.add_argument("--shard", default="",
                    help="comma-separated worker backends for the `shard` "
                         "delegate, e.g. cuda_flat_pwa_gen_s0e2,cpu_flat_pwa_gen_cached")
    ap.add_argument("--weights", default="",
                    help="comma-separated per-worker split weights, e.g. 57,43")
    ap.add_argument("--fix-mw", action="store_true",
                    help="fix every mass/width at its config value (the paper's "
                         "first column; BWR masses/widths are FREE by default)")
    ap.add_argument("--vary", action="store_true",
                    help="perturb x on every iteration, so couplings and any free "
                         "mass/width change as they do in a real fit (default: the "
                         "same x, which can hide a value-keyed cache)")
    ap.add_argument("--vary-eps", type=float, default=1e-3,
                    help="relative step of --vary (per slot)")
    ap.add_argument("--free-mw", default="",
                    help="comma-separated mass/width names to leave FREE "
                         "(implies --fix-mw for the rest)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    spec = {"name": args.backend}
    if args.shard:
        spec = {"name": "shard", "start_method": "spawn",
                "backends": [b.strip() for b in args.shard.split(",") if b.strip()]}
        if args.weights:
            spec["weights"] = [float(w) for w in args.weights.split(",")]
        label = f"shard[{args.shard}]{args.weights and ' ' + args.weights or ''}"
    else:
        label = args.backend
    if args.weights and not args.shard:
        ap.error("--weights only applies to --shard")
    if args.base:
        spec["base"] = args.base
    if args.rsize:
        if args.shard:
            ap.error("--rsize must go to the workers: --shard takes bare backend "
                     "names (a shard-level resolution_size would leave the workers "
                     "at 1 while the parent advertises N)")
        spec["resolution_size"] = args.rsize
    f = Fitter(args.config, backend=spec)
    t_init = time.perf_counter()
    nd = args.max_data or args.max_events
    np_ = args.max_phsp or args.max_events
    phsp = f.load_dataset("phsp", **({"max_events": np_} if np_ else {}))
    data = f.load_dataset("data", **({"max_events": nd} if nd else {}))
    t_io = time.perf_counter()
    f.set_phsp(phsp)
    f.set_data(data)
    t_upload = time.perf_counter()
    ne, nb = len(data["weight"]), len(phsp["weight"])
    f.apply_constrains()
    if args.fix_mw or args.free_mw:
        keep = {s.strip() for s in args.free_mw.split(",") if s.strip()}
        known = set(dict.fromkeys(f.var_registry.flat_names))
        unknown = sorted(keep - known)
        if unknown:
            avail = sorted(n for n in known if n.endswith(("_mass", "_width")))
            raise SystemExit(
                f"bench.py: unknown --free-mw name(s): {unknown}\n"
                f"  the fitted mass/width names are (note the charge suffix, e.g. "
                f"K3st1780p_mass, not K3st1780_mass):\n  {avail}")
        mw = {n: float(f.defaults[n])
              for n in dict.fromkeys(f.var_registry.flat_names)
              if n.endswith(("_mass", "_width")) and n not in keep
              and n in f.defaults}
        f.set_fixed(mw)
    x0 = f.initial_values(seed=1)
    t_prep = time.perf_counter()
    t_load = t_prep - t0

    n_x = len(x0)
    names = [spec.get("name", "")] + list(spec.get("backends", []))
    if any("integrated" in n for n in names):
        free_mw = [n for n in dict.fromkeys(f.var_registry.flat_names)
                   if n.endswith(("_mass", "_width"))]
        if free_mw:
            raise SystemExit(
                f"bench.py: {spec} is an integrated (Gram) backend, which zeroes "
                f"the m0/g0 gradient blocks, but {len(free_mw)} mass/width "
                f"parameters are free: the printed ms would look fine and the "
                f"gradient would be wrong in {len(free_mw)} directions.  Pass "
                f"--fix-mw, or use a per-event kernel.")
    f.get_nll(x0)                                   # warm-up (lazy precompute)
    t_warm = time.perf_counter()
    ts = []
    nll = float("nan")
    rng = np.random.default_rng(12345)
    x_cur = x0.copy()
    for _ in range(args.repeat):
        if args.vary:
            # a real fit moves every free slot; scale the step by the slot itself
            x_cur = x0 + args.vary_eps * (1.0 + np.abs(x0)) * rng.standard_normal(x0.size)
        a = time.perf_counter()
        nll, _g = f.get_nll(x_cur)
        ts.append(time.perf_counter() - a)
    t = min(ts)
    print(f"{label:28s} x={n_x:3d} ne={ne:>9,} nphsp={nb:>9,}  "
          f"{t*1e3:8.1f} ms/iter  {(ne+nb)/t/1e6:6.2f} Mev/s  NLL={nll:+.6f}"
          + ("  [vary]" if args.vary else ""),
          flush=True)
    print(f"{'':28s} prep: init {t_init - t0:5.1f} | file i/o {t_io - t_init:5.1f} | "
          f"upload+caches {t_upload - t_io:5.1f} | constrain {t_prep - t_upload:5.1f} | "
          f"first NLL {t_warm - t_prep:6.1f} | total {t_warm - t0:5.1f} s",
          flush=True)
    f.backend.free()


if __name__ == "__main__":
    main()
