#!/usr/bin/env python3
"""Kernel-format arrays at the CTPWA benchmark size (TIMING sample).

The cost of one iteration depends on the array shapes and sizes, not on the
values, so this writes the `pwa_gen`/`pwa` kernel inputs directly in the
config's own valid ranges instead of generating four-momenta and converting
them:

    mass   uniform in the kernel's [m_min, m_max]
    q      uniform in the kernel's [q_min, q_max]
    angle  uniform in [-pi, pi)
    weight 1, bkg 0

Use `make_data.py` (toy generation) when a *physics* sample is needed; this is
for the performance comparison only, and it is labelled as such.

    python make_flat_data.py --n-data 1000000 --n-phsp 5000000
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yml"))
    ap.add_argument("--n-data", type=int, default=1_000_000)
    ap.add_argument("--n-phsp", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    a = ap.parse_args()

    from tabpwa import Fitter
    f = Fitter(a.config, backend="numpy_pwa")
    kc = f.kernel_config
    rng = np.random.default_rng(a.seed)

    def arrays(n):
        mi = np.asarray(kc["mass_index"])
        n_mass = int(mi.max()) + 1
        n_q = int(np.asarray(kc["fl_q_index"]).max()) + 1
        ai = np.asarray(kc["angle_index"])
        n_at = int(ai.max()) + 1
        nac = int(np.asarray(kc["angle_k"]).shape[1])
        m_lo = float(kc.get("m_min", 1.0))
        n_bins = max(1, int(np.asarray(kc["gamma_table"]).shape[1] - 1))
        m_hi = m_lo + float(kc.get("m_delta", 1.0)) * n_bins
        q_lo = float(kc.get("q_min", 0.0))
        q_hi = q_lo + float(kc.get("q_delta", 1.0)) * n_bins
        return {
            "mass": rng.uniform(m_lo, m_hi, (n, n_mass)),
            "q": rng.uniform(max(q_lo, 1e-3), q_hi, (n, n_q)),
            "angle": rng.uniform(-np.pi, np.pi, (n, n_at, nac)),
            "weight": np.ones(n),
            "bkg": np.zeros(n),
        }

    os.makedirs(a.out, exist_ok=True)
    for name, n in (("data.npz", a.n_data), ("phsp.npz", a.n_phsp)):
        d = arrays(n)
        np.savez(os.path.join(a.out, name), **d)
        print(f"{name}: {n:,} events, mass {d['mass'].shape} q {d['q'].shape} "
              f"angle {d['angle'].shape}", flush=True)


if __name__ == "__main__":
    main()
