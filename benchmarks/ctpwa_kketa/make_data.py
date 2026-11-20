#!/usr/bin/env python3
"""Generate the CTPWA benchmark samples with the framework's OWN generator

NOTE: currently BROKEN for 3-body chains — `tabpwa.pwa_build.generate_pwa_phsp`
returns four-vectors with negative energies that sum to zero (Sigma E = 0,
m^2 = -9.59) on `jpsi -> R + c`, including on `tests/config_pwa.yml`.  Use
`make_flat_data.py` for a TIMING sample until that is fixed.

`toy_pwa_generate` builds phase space with the framework's `generate_pwa_phsp`
(the exact inverse of `decay_angles_from_momenta`) and accepts events by
rejection against the model's |A|², so the sample is a model toy rather than
hand-made kinematics.

    python make_data.py --n-data 1000000 --n-phsp 5000000

Both outputs are kernel-format `.npz` (`.data_arr` / `.phsp_arr`), which the
config loads with `Fitter.load_dataset`.
"""
import argparse
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yml"))
    ap.add_argument("--backend", default="cuda_v4_pwa_cache")
    ap.add_argument("--n-data", type=int, default=1_000_000)
    ap.add_argument("--n-phsp", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--batch", type=int, default=50000)
    args = ap.parse_args()

    from tabpwa import Fitter
    from tabpwa.toy_pwa import toy_pwa_generate

    os.makedirs(args.out, exist_ok=True)
    f = Fitter(args.config, backend=args.backend)
    f.apply_constrains()

    t0 = time.perf_counter()
    data, phsp, x0 = toy_pwa_generate(f, args.n_data, seed=args.seed,
                                      n_phsp=args.n_phsp,
                                      batch_size=args.batch, verbose=True)
    dt = time.perf_counter() - t0
    np.savez(os.path.join(args.out, "data.npz"), **{f"{k}": v for k, v in data.items()})
    np.savez(os.path.join(args.out, "phsp.npz"), **{f"{k}": v for k, v in phsp.items()})
    print(f"data {len(data['weight']):,} events | phsp {len(phsp['weight']):,} "
          f"events | {dt:.0f} s", flush=True)
    f.backend.free()


if __name__ == "__main__":
    main()
