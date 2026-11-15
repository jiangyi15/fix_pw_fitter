#!/usr/bin/env python3
"""Benchmark ``cuda_v5_pwa_gen_bw`` against ``cuda_v5_pwa_gen_flat``.

Both kernels are the same fused tile / grouped log-sum ``pwa_gen`` kernel;
they differ ONLY in how the denominator coefficient block is laid out:

* ``flat`` — one slot range per **(wave, factor)**: ``coeff[n_wave·n_res·n_basis]``
  (the weights of a shared BW are repeated once per referring wave);
* ``bw``   — one slot range per **unique base BW**: ``coeff[n_bw·n_basis]``,
  reached by every referring wave through ``bw_order``.

The observable consequences, in the order they show up in a call:

1. the per-call **H2D upload** of ``coeff`` and the **D2H download** of
   ``dcoeff`` shrink from ``n_flat`` to ``n_bw·n_basis`` doubles.  At these
   block sizes this is ~8 kB per iteration — negligible against the kernel;
2. the **basis matvec** (phase A) is evaluated once per **(event, BW)**
   instead of once per **(event, wave, factor)**, because every referring
   wave reads the same basis sum.  This is the real difference: it divides
   the u-cache reads of phase A by ``n_fact`` (594 -> 1 on the kmatrix2
   shape below);
3. the **coefficient gradient** (phase D) sums all copies of a BW in
   registers (warp reduction over ``factors_of_bw``) and then reads the
   basis values once per **(event, BW, slot)** instead of once per
   (event, copy, slot), with ONE write per slot per block.

So this is an A/B of the block layout on the SAME config, data and params at
the SAME sizes.  Two shapes:

* ``tests/config_pwa_gen_km.yml`` tiled to ~198 waves (the realistic
  ``pwa_gen`` shape: 198·5 = 990 flat slots vs 5 per-BW slots);
* ``tests/config_pwa_gen_mb.yml`` (``n_res == 2`` — every wave's denominator
  is a product of two shared BWs) tiled to 96 waves (192·2·5 = 1920 flat
  slots vs 8).  The tiling is capped at 96 waves because the flat sibling's
  shared tile (which must carry the per-(wave, factor) BW factor cache)
  exceeds the device's 99 KB opt-in limit beyond that; both variants are
  measured on the same shape.

For each variant we report the context ``create`` time, ``load_data``
(one-time amp/u-cache fill, includes the numerator), and ``compute``
(``norm=<scalar>, return_p=True`` — the per-iteration hot path), best of N
with the trial spread, plus the coefficient-block sizes.

Usage (repo root)::

    python tests/benchmark_pwa_gen_bw.py
    python tests/benchmark_pwa_gen_bw.py --sizes 20000,65536,262144 --trials 7
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from tabpwa.amp_model import build_amplitude_model
from tabpwa.cuda._v5_pwa_gen_bw import CUDAKernelV5PWAGenBW
from tabpwa.cuda._v5_pwa_gen_flat import CUDAKernelV5PWAGenFlat
from benchmark_pwa_kernels import _rate, _time, tile_case
from benchmark_pwa_gen_s0e import _km_case

SIZES = [20000, 65536, 262144]
RSIZES = [1, 8]
N_WAVE = 198
N_WAVE_MB = 96          # flat sibling's shared tile limit (see the docstring)
CFG_MB = "tests/config_pwa_gen_mb.yml"

VARIANTS = (("gen_flat", CUDAKernelV5PWAGenFlat),
            ("gen_bw", CUDAKernelV5PWAGenBW))

_WARMUP, _TRIALS, _RS = 2, 5, 1


def _trials(fn):
    """(best, spread) of ``_TRIALS`` timed calls, after ``_WARMUP`` warm-ups."""
    for _ in range(_WARMUP):
        fn()
    ts = []
    for _ in range(_TRIALS):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    best = min(ts)
    return best, (max(ts) - best) / best if best > 0 else 0.0


def _mb_case(n_wave_target=N_WAVE_MB, n_events=256):
    """The ``n_res == 2`` kernel config tiled to ~``n_wave_target`` waves."""
    kc0 = build_amplitude_model(CFG_MB).build_kernel_config()
    P0 = int(kc0.get("n_proj", 1) or 1)
    nw0 = int(kc0["matrix_angle"].shape[1])
    N0 = nw0 // P0
    k = max(1, int(round(n_wave_target / nw0)))
    kc = tile_case(kc0, P0, k)          # n_wave = nw0 * k (projections kept)
    # the per-wave tables follow the per-projection tiling
    for key in ("n_index",):
        a = np.asarray(kc0[key])
        kc[key] = np.concatenate([np.tile(a[p * N0:(p + 1) * N0], k)
                                  for p in range(P0)])
    n_wave = int(kc["matrix_angle"].shape[1])

    rng = np.random.default_rng(11)
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
        "ck": rng.normal(size=n_wave // P0)
        + 1j * rng.normal(size=n_wave // P0),
        "coeff": rng.normal(size=(int(np.asarray(kc["bw_order"]).size),
                                  n_basis)),
    }
    return kc, data, params, float(n_events) * 0.5


def _row(label, kc, data_full, params, norm, sizes):
    n_wave = int(kc["matrix_angle"].shape[1])
    n_res = int(np.asarray(kc["bw_order"]).size) // n_wave
    print(f"{label}   (n_wave={n_wave}, n_proj={kc.get('n_proj')}, "
          f"n_res={n_res}, "
          f"n_bw={len(np.unique(np.asarray(kc['bw_order'])))})")
    header = (f"{'n_events':>9}" + "".join(f" | {name:>25}" for name, _ in VARIANTS)
              + f" | {'blk flat/bw':>12} | {'bw/flat':>8}")
    print(header)
    print("-" * len(header))
    ratios = []
    for n in sizes:
        data = {k: v[:n] for k, v in data_full.items()}
        cells, bests, blocks, traffic = [], {}, [], []
        for name, cls in VARIANTS:
            t0 = time.perf_counter()
            k = cls(kc, resolution_size=_RS)
            t_create = time.perf_counter() - t0
            t_load, _sp_l = _trials(lambda: k.load_data(data))
            h = k.load_data(data)
            t_cmp, sp_c = _trials(lambda: k.compute(params, h, norm=norm,
                                                    return_p=True))
            bests[name] = t_cmp
            nb = int(k.n_basis_tot)
            blocks.append(nb)
            traffic.append((2 * k.n_wave_base + nb) * 8)
            cells.append(f"{_rate(n, t_cmp)} {_time(t_cmp)}"
                         f" ±{100 * sp_c:.1f}%")
            print(f"  [{name:>8}] create {_time(t_create)}"
                  f"  load_data {_time(t_load)}"
                  f"  coeff block {nb} doubles"
                  f"  H2D {(2 * k.n_wave_base + nb) * 8 / 1e3:.1f} kB"
                  f"  D2H {(2 + 2 * k.n_wave_base + nb) * 8 / 1e3:.1f} kB")
            h.free()
            k.free()
        r = bests["gen_bw"] / bests["gen_flat"]
        ratios.append(r)
        print(f"{n:>9}" + "".join(f" | {c:>25}" for c in cells)
              + f" | {blocks[0]:>5}/{blocks[1]:<5} | {r:>7.3f}x")
    print()
    return ratios


def main():
    global _WARMUP, _TRIALS, _RS
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

    print(f"building the kmatrix2 / two-BW cases (n_events {n_max}) ...")
    kc_km, data_km, params_km, norm_km = _km_case(N_WAVE, n_max)
    kc_mb, data_mb, params_mb, norm_mb = _mb_case(N_WAVE_MB, n_max)

    all_ratios = {}
    for rs in rsizes:
        _RS = rs
        print(f"\n=== resolution_size = {rs} ===")
        all_ratios[("km", rs)] = _row("A) kmatrix2 basis, ~198 waves",
                                      kc_km, data_km, params_km, norm_km,
                                      sizes)
        all_ratios[("mb", rs)] = _row("B) two shared BWs (n_res = 2), 96 waves",
                                      kc_mb, data_mb, params_mb, norm_mb,
                                      sizes)

    print("=" * 78)
    print(f"  compute-time ratio gen_bw / gen_flat (< 1 = shared BW faster), "
          f"best of {_TRIALS}:")
    print("=" * 78)
    for (case, rs), rs_list in all_ratios.items():
        print(f"  {case:>3} rs={rs}: "
              + ", ".join(f"{s}: {r:.3f}x" for s, r in zip(sizes, rs_list)))
    print("\nThe block-size win (n_flat -> n_bw·n_basis) is only ~8 kB of "
          "traffic per call\nand does not move the needle; the speedup comes "
          "from evaluating the basis\nmatvec once per (event, BW) instead of "
          "once per (event, wave, factor), and\nfrom reading the basis values "
          "once per (event, BW, slot) in the gradient\nphase instead of once "
          "per (event, copy, slot).")


if __name__ == "__main__":
    main()
