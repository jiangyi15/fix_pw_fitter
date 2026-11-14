#!/usr/bin/env python3
"""Benchmark ``cuda_v5_pwa_gen_s0e`` against ``cuda_v5_pwa_s0e``.

Two shapes, both at ``resolution_size ∈ {1, 8}`` and 65k / 262k events:

1. **198-wave N ≡ 1** — the synthetic template case of
   ``benchmark_pwa_kernels`` (``tests/config_pwa.yml`` tiled to N = 198).
   The two kernels must be ~equal: the gen copy differs only by the
   (absent) numerator and an identical basis table.
2. ``kmatrix2`` (``tests/config_pwa_gen_km.yml``) tiled to 198 waves.  Only
   ``cuda_v5_pwa_gen_s0e`` evaluates the real ``pwa_gen`` model; the plain
   ``cuda_v5_pwa_s0e`` is timed on the SAME kernel config (it ignores
   ``den_row`` / ``n_index`` and uses the s0c-preset basis) as the cost
   reference for the extra basis rows (``n_basis`` 5 vs 3).

The kernels are instantiated DIRECTLY (not through ``create_backend``): the
scope guard legitimately refuses a ``pwa_gen`` config to ``cuda_v5_pwa_s0e``,
and the benchmark needs that exact A/B pair.

Both per-call times are ``compute(norm=<scalar>, return_p=True)`` — the full
data NLL + gradients (the per-iteration hot path).  The numerator lives in
the load-time amp cache, so it costs nothing here; its one-time fill cost is
reported separately (``load_data``).

Usage (repo root)::

    python tests/benchmark_pwa_gen_s0e.py
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from tabpwa.amp_model import build_amplitude_model
from tabpwa.cuda._v5_pwa_gen_s0e import CUDAKernelV5PwaGenS0E
from tabpwa.cuda._v5_pwa_s0e import CUDAKernelV5PWAS0E
from benchmark_pwa_kernels import _rate, _time, synthetic_case

SIZES = [65536, 262144]
RSIZES = [1, 8]
N_WAVE = 198
CFG_PWA = "tests/config_pwa.yml"
CFG_KM = "tests/config_pwa_gen_km.yml"

_WARMUP, _TRIALS = 2, 5


def _bench(fn):
    for _ in range(_WARMUP):
        fn()
    t0 = time.perf_counter()
    for _ in range(_TRIALS):
        fn()
    return (time.perf_counter() - t0) / _TRIALS


def _km_case(n_wave_target=N_WAVE, n_events=256):
    """kmatrix2 kernel config tiled to ~``n_wave_target`` waves + random data."""
    from benchmark_pwa_kernels import tile_case

    kc0 = build_amplitude_model(CFG_KM).build_kernel_config()
    P0 = int(kc0.get("n_proj", 1) or 1)
    nw0 = int(kc0["matrix_angle"].shape[1])
    N0 = nw0 // P0
    k = max(1, int(round(n_wave_target / nw0)))
    kc = tile_case(kc0, P0, k)          # n_wave = nw0 * k (projections kept)
    # the pwa_gen per-wave tables follow the per-projection tiling
    for key in ("n_index", "n_mass_index"):
        a = np.asarray(kc0[key])
        kc[key] = np.concatenate([np.tile(a[p * N0:(p + 1) * N0], k)
                                  for p in range(P0)])
    n_wave = int(kc["matrix_angle"].shape[1])

    rng = np.random.default_rng(7)
    gmin = float(kc["gamma_min"])
    gspan = float(kc["gamma_delta"]) * np.asarray(kc["gamma_table"]).shape[-1]
    n_mass = int(np.max(kc["mass_index"])) + 1
    n_mom = int(np.max(kc["fl_q_index"])) + 1
    n_ang = int(np.max(kc["angle_index"])) + 1
    nac = int(kc["angle_k"].shape[-1])
    data = {
        "mass": rng.uniform(gmin + 0.05 * gspan, gmin + 0.95 * gspan,
                            (n_events, n_mass)),
        "q": rng.uniform(0.0, 1.0, (n_events, n_mom)),
        "angle": rng.uniform(-np.pi, np.pi, (n_events, n_ang, nac)),
        "weight": np.ones(n_events),
        "bkg": np.full(n_events, 0.01),
    }
    n_basis = int(np.asarray(kc["den_row"]).shape[1])
    params = {
        "ck": rng.normal(size=n_wave // P0) + 1j * rng.normal(size=n_wave // P0),
        "m0": rng.uniform(2.0, 3.0, int(np.max(kc["m0_index"])) + 1),
        "g0": rng.uniform(0.05, 0.2, int(np.max(kc["g0_index"])) + 1),
        "coeff": rng.normal(size=(n_wave, n_basis)) * 0.3,
    }
    norm = float(n_events) * 0.5
    return kc, data, params, norm


def _row(label, kc, data_full, params, norm, sizes, factories):
    print(f"{label}   (n_wave={int(kc['matrix_angle'].shape[1])})")
    header = (f"{'n_events':>9}"
              + "".join(f" | {name:>22}" for name, _ in factories)
              + f" | {'load_data':>12}")
    print(header)
    print("-" * len(header))
    for n in sizes:
        data = {k: v[:n] for k, v in data_full.items()}
        cells = []
        t_load = float("nan")
        for name, make in factories:
            try:
                k = make()
                t_load = _bench(lambda: k.load_data(data))
                h = k.load_data(data)
                t = _bench(lambda: k.compute(params, h, norm=norm,
                                             return_p=True))
                cells.append(f"{_rate(n, t)}  {_time(t)}")
                k.free()
            except Exception as exc:                  # noqa: BLE001
                cells.append(f"{'SKIP':>22}")
                print(f"  [{name}] {type(exc).__name__}: {exc}")
        print(f"{n:>9}" + "".join(f" | {c:>22}" for c in cells)
              + f" | {_time(t_load):>12}")
    print()


def main():
    global _WARMUP, _TRIALS
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--rsizes", default=",".join(str(r) for r in RSIZES))
    args = ap.parse_args()
    _WARMUP, _TRIALS = args.warmup, args.trials
    sizes = [int(x) for x in args.sizes.split(",") if x]
    rsizes = [int(x) for x in args.rsizes.split(",") if x]

    n_max = max(sizes)
    print(f"building the {N_WAVE}-wave synthetic case (n_events {n_max}) ...")
    kc, data_max, _phsp, params, norm, info = synthetic_case(
        CFG_PWA, N_WAVE, 1, n_max)
    print(f"template N0={info['N0']} x{info['k']} -> N={info['N']}, "
          f"n_wave={info['n_wave']}\n")
    kc_km, data_km, params_km, norm_km = _km_case(N_WAVE, n_max)
    print(f"kmatrix2: n_wave={int(kc_km['matrix_angle'].shape[1])}, "
          f"n_proj={kc_km.get('n_proj')}, n_index rows="
          f"{sorted(set(np.asarray(kc_km['n_index']).tolist()))}\n")

    for rs in rsizes:
        print(f"=== resolution_size = {rs} ===")
        _row("A) N ≡ 1, 198 waves", kc, data_max, params, norm, sizes,
             [("s0e", lambda rs=rs, kc=kc: CUDAKernelV5PWAS0E(
                 kc, resolution_size=rs)),
              ("gen_s0e", lambda rs=rs, kc=kc: CUDAKernelV5PwaGenS0E(
                  kc, resolution_size=rs))])
        _row("B) kmatrix2 basis, 198 waves", kc_km, data_km, params_km,
             norm_km, sizes,
             [("s0e (s0 preset)", lambda rs=rs, kc=kc_km: CUDAKernelV5PWAS0E(
                 kc, resolution_size=rs)),
              ("gen_s0e (pwa_gen)", lambda rs=rs, kc=kc_km:
                  CUDAKernelV5PwaGenS0E(kc, resolution_size=rs))])
        print()


if __name__ == "__main__":
    main()
