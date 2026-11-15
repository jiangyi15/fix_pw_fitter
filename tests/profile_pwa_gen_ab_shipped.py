#!/usr/bin/env python3
"""A/B the SHIPPED pwa_gen GPU kernels, interleaved.

This is the number that goes in a commit message: both sides are the
repository's own libraries loaded through their own wrappers, so it measures
what a fit actually gets.  Interleaved round-robin because this laptop GPU
throttles from 2730 to 180 MHz inside a single run (a sequential comparison
is worthless: the same kernel measured 5.6 ms and 12.8 ms minutes apart).

    python tests/profile_pwa_gen_ab_shipped.py
    python tests/profile_pwa_gen_ab_shipped.py --n 262144 --rounds 5

Columns:
  ``ms``      best of N rounds (least-throttled)
  ``med``     median across rounds (what a long fit sees)
  ``ratio``   bwt / bw  (< 1 means the tuned kernel is faster)
  ``check``   Q / P / ck / coeff relative differences on the same call
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from benchmark_pwa_gen_bw import _mb_case
from benchmark_pwa_gen_s0e import _km_case
from tabpwa.cuda._v5_pwa_gen_bw import CUDAKernelV5PWAGenBW
from tabpwa.cuda._v5_pwa_gen_bwt import CUDAKernelV5PWAGenBWT

KERNELS = {"bw (shipped)": CUDAKernelV5PWAGenBW,
           "bwt (tuned)": CUDAKernelV5PWAGenBWT}


def _rel(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="km", choices=["km", "mb"])
    ap.add_argument("--n", type=int, default=65536)
    ap.add_argument("--rs", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=9)
    args = ap.parse_args()

    kc, data, params, norm = (_km_case(n_events=args.n) if args.shape == "km"
                              else _mb_case(n_events=args.n))
    norm = float(norm)
    n_wave = int(kc["matrix_angle"].shape[1])
    print(f"=== {args.shape}: n={args.n}, n_wave={n_wave}, n_res="
          f"{np.asarray(kc['bw_order']).size // n_wave}, "
          f"n_proj={kc.get('n_proj', 1)}, rs={args.rs}, "
          f"{args.rounds} interleaved rounds ===")

    insts = {}
    for name, cls in KERNELS.items():
        k = cls(kc, resolution_size=args.rs)
        insts[name] = (k, k.load_data(data))
    try:
        times = {name: [] for name in insts}
        first = {}
        for i in range(args.rounds):
            for name, (k, h) in insts.items():
                k.compute(params, h, norm=norm)             # warm
                t0 = time.perf_counter()
                out = k.compute(params, h, norm=norm)
                times[name].append((time.perf_counter() - t0) * 1e3)
                if i == 0:
                    first[name] = out

        base = min(times["bw (shipped)"])
        base_med = np.median(times["bw (shipped)"])
        print(f"{'kernel':<16} {'ms':>8} {'med':>8} {'ratio':>7} {'medratio':>9}")
        for name in insts:
            print(f"{name:<16} {min(times[name]):>8.3f} "
                  f"{np.median(times[name]):>8.3f} "
                  f"{min(times[name]) / base:>7.3f} "
                  f"{np.median(times[name]) / base_med:>9.3f}")

        Qa, ga, Pa = first["bw (shipped)"]
        Qb, gb, Pb = first["bwt (tuned)"]
        print(f"  check: dQ={_rel(Qa, Qb):.2e}  dP={_rel(Pa, Pb):.2e}  "
              f"dck={_rel(ga['ck'], gb['ck']):.2e}  "
              f"dcoeff={_rel(ga['coeff'], gb['coeff']):.2e}")
    finally:
        for k, h in insts.values():
            h.free()
            k.free()


if __name__ == "__main__":
    main()
