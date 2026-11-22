#!/usr/bin/env python3
"""A/B the flat `pwa_gen` GPU backends on the same shape, interleaved.

The GPU on this machine throttles hard (observed 180 MHz .. 2730 MHz inside
one session), so a sequential comparison is worthless.  This harness keeps
all the kernels alive at once, alternates them every round and reports the
min per variant, so each sees the same clock distribution.

  cuda_flat_pwa_gen        full config, F recomputed every call
  cuda_flat_cache_pwa_gen  F cached at load_data (the only per-iteration
                           difference now that the builder folds the basis)
  numpy_flat               CPU reference (for scale only)

    python tests/benchmark_flat_pwa_gen.py
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from tabpwa import Fitter
from tabpwa.backends import create_backend

CFGS = ("tests/config_pwa_gen_km.yml", "tests/config_pwa_gen_mb.yml",
        "tests/config_pwa_gen_same.yml")
GPU = ("cuda_flat_pwa_gen", "cuda_flat_cache_pwa_gen")


def _in_grid(kc, ne, seed=0):
    """Events strictly inside both interpolation grids (clamping diverges).

    DELIBERATELY not ``_flat_grid._in_grid``: a benchmark wants a flat weight
    (``weight=1``) and a fixed background (``bkg=0.01``) so the per-iteration
    cost is the kernel's, not a random draw's — the sampled mass/q/angle recipe
    is the same and the test-side copy is what the gates use.
    """
    rng = np.random.RandomState(seed)
    n_basis = int(np.asarray(kc["basis_table"]).shape[1])
    n_fl = int(np.asarray(kc["fl_table"]).shape[-1])
    m_min, m_del = float(kc["m_min"]), float(kc["m_delta"])
    q_min, q_del = float(kc["q_min"]), float(kc["q_delta"])
    n_mass = int(np.max(kc["num_m_index"])) + 1
    n_q = int(np.max(kc["fl_q_index"])) + 1
    n_at = int(np.max(kc["angle_index"])) + 1
    nac = int(np.asarray(kc["angle_k"]).shape[1])
    return {
        "mass": m_min + 0.05 * m_del * n_basis
                + rng.rand(ne, n_mass) * (0.9 * m_del * n_basis),
        "q": q_min + 0.05 * q_del * n_fl
             + rng.rand(ne, n_q) * (0.9 * q_del * n_fl),
        "angle": rng.uniform(-np.pi, np.pi, (ne, n_at, nac)),
        "weight": np.ones(ne),
        "bkg": np.full(ne, 0.01),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="20000,65536")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--norm", type=float, default=1.3)
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",") if x]

    for cfg in CFGS:
        f = Fitter(cfg)
        f.apply_constrains()
        kc = f.kernel_config
        x = f.initial_values(seed=1)
        params, _ = f.build_params(x)
        n_basis = int(np.asarray(kc["basis_table"]).shape[0])
        print("=" * 76)
        print(f"{cfg}   n_wave={np.asarray(kc['fl_table']).shape[0]} "
              f"n_basis={n_basis} n_res={np.asarray(kc['num_table']).shape[1]}")
        print("=" * 76)
        for ne in sizes:
            data = _in_grid(kc, ne)
            insts = {}
            for name in GPU:
                try:
                    k = create_backend(name, kc, model="pwa_gen")
                    insts[name] = (k, k.load_data(data))
                except Exception as exc:                     # noqa: BLE001
                    print(f"  {name:26s} SKIP {type(exc).__name__}: "
                          f"{str(exc)[:60]}")
            if not insts:
                continue
            times = {n: [] for n in insts}
            for _ in range(args.rounds):
                for name, (k, h) in insts.items():
                    k.compute(params, h, norm=args.norm)      # warm
                    t0 = time.perf_counter()
                    k.compute(params, h, norm=args.norm)
                    times[name].append((time.perf_counter() - t0) * 1e3)
            base = None
            for name, (k, h) in insts.items():
                t = min(times[name])
                if base is None:
                    base = t
                extra = ""
                if name == "cuda_flat_cache_pwa_gen":
                    extra = f"  ({base / t:.2f}x vs flat)"
                print(f"  ne={ne:6d}  {name:26s} {t:8.3f} ms  "
                      f"{ne / t / 1e3:6.2f} M/s{extra}")
                k.free()
        print()


if __name__ == "__main__":
    main()
