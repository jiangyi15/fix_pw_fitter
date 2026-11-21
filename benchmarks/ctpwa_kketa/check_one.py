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
    ap.add_argument("--max-events", type=int, default=2000,
                    help="0 = the full sample; default cap for both samples")
    ap.add_argument("--max-data", type=int, default=0,
                    help="cap the data sample only (else --max-events)")
    ap.add_argument("--max-phsp", type=int, default=0,
                    help="cap the phase-space sample only (else --max-events)")
    ap.add_argument("--fix-mw", action="store_true")
    ap.add_argument("--free-mw", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    f = Fitter(a.config, backend=a.backend)
    nd = a.max_data or a.max_events
    np_ = a.max_phsp or a.max_events
    phsp = f.load_dataset("phsp", **({"max_events": np_} if np_ else {}))
    data = f.load_dataset("data", **({"max_events": nd} if nd else {}))
    f.set_phsp(phsp)
    f.set_data(data)
    f.apply_constrains()
    keep = {s.strip() for s in a.free_mw.split(",") if s.strip()}
    known = set(dict.fromkeys(f.var_registry.flat_names))
    unknown = sorted(keep - known)
    if unknown:
        avail = sorted(n for n in known if n.endswith(("_mass", "_width")))
        raise SystemExit(f"check_one.py: unknown --free-mw name(s): {unknown}\n"
                         f"  available: {avail}")
    mw = {n: float(f.defaults[n])
          for n in dict.fromkeys(f.var_registry.flat_names)
          if n.endswith(("_mass", "_width")) and n not in keep
          and n in f.defaults}
    if a.fix_mw or a.free_mw:
        f.set_fixed(mw)

    x0 = f.initial_values(seed=1)
    params, _ = f.build_params(x0)
    # reuse the handle set_data() already uploaded: loading the same data again
    # allocates a SECOND full set of per-handle caches (s0e: ~2 kB/event), which
    # is enough to exhaust a small device
    h = getattr(f, "_data_holder", None)
    if h is None:
        h = f.backend.load_data(data)
    Q, _g, P = f.backend.compute(params, h, norm=1.0)
    nll, grad = f.get_nll(x0)
    np.savez(a.out, Q=np.asarray(Q, dtype=float), P=np.asarray(P, dtype=float),
             nll=np.asarray(nll, dtype=float), grad=np.asarray(grad, dtype=float),
             x=x0, ck=np.asarray(params["ck"]))
    print(f"{a.backend:22s} x={len(x0):3d}  Q={float(np.asarray(Q).sum()):+.10e}  "
          f"nll={float(nll):+.10e}  P[0:3]={np.asarray(P)[:3]}", flush=True)
    f.backend.free()


if __name__ == "__main__":
    main()
