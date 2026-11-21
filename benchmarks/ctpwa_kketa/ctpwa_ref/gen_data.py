#!/usr/bin/env python3
"""Generate CTPWA's own flat phase space for the J/psi -> K+ K- eta benchmark.

    python gen_data.py --config config.yml --n-data 1000000 --n-phsp 5000000

Uses `ctpwa.PhaseSpaceGenerator` (their GENBOD), so the sample is produced by
their code in their own format (`.dat`: one line per particle per event,
"E px py pz", final states in `Data.order`).  Flat phase space, i.e. the same
construction as this repository's `make_data.py`.
"""
import argparse
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yml"))
    ap.add_argument("--n-data", type=int, default=1_000_000)
    ap.add_argument("--n-phsp", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--batch", type=int, default=1_000_000)
    a = ap.parse_args()

    import ctpwa

    print("backend:", ctpwa.backend())
    gen = ctpwa.PhaseSpaceGenerator(a.config, "Jpsi")
    t0 = time.perf_counter()
    for name, n, seed in (("data.dat", a.n_data, a.seed),
                          ("phsp.dat", a.n_phsp, a.seed + 7919)):
        path = os.path.join(a.out, name)
        written = 0
        with open(path, "w") as fh:
            for k in range(0, n, a.batch):
                m = min(a.batch, n - k)
                ev = gen.generate(m, seed=seed + k)
                np.savetxt(fh, np.asarray(ev, dtype=np.float64).reshape(-1, 4),
                           fmt="%.8f %.8f %.8f %.8f")
                written += m
                print(f"  {name}: {written:,}/{n:,}", flush=True)
        size = os.path.getsize(path) / 1e9
        print(f"{name}: {written:,} events, {size:.2f} GB", flush=True)
    print(f"total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
