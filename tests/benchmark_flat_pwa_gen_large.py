#!/usr/bin/env python3
"""Benchmark the flat `pwa_gen` GPU backends on the `benchmark_pwa_kernels`
shape.

Uses the SAME shape convention as ``tests/benchmark_pwa_kernels.py`` so the
numbers are comparable with the pure-PWA family it benchmarks:

* ``--n-comp`` is ``N`` — the shared-ck length **per projection**
  (``n_wave = n_proj · N`` total);
* ``--n-proj`` sets the projection count (``2`` matches the reference);
* the wave axis is tiled from the template so the actual ``N`` is
  ``N0 · round(n_comp / N0)``.

The kernel-config vocabulary differs (the flat config has no ``matrix_angle``
/ ``bw_order`` / ``fl_order``), so the tiling is expressed in the flat terms:
``fl_table`` / ``num_table`` / ``gamma_matrix`` / ``angle_matrix`` repeat
along their wave axis, and — because the builder has already folded the basis
— the repeated waves SHARE the same basis rows, so ``gamma_matrix`` grows
along its wave axis only.

Measured interleaved (all kernels live, alternating, min per variant): this
GPU throttles 180..2730 MHz inside a session, so a sequential comparison is
worthless.

    python tests/benchmark_flat_pwa_gen_large.py
    python tests/benchmark_flat_pwa_gen_large.py --n-comp 100 --n-proj 2 \
        --preset large --trials 5
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

TEMPLATE = "tests/config_pwa_gen_km.yml"
PRESETS = {"small": [1024, 4096, 10000],
           "large": [65536, 262144, 1048576]}
GPU = ("cuda_flat_pwa_gen", "cuda_flat_cache_pwa_gen")


def _fit_proj(a, axis, n_proj):
    """Repeat/truncate the projection axis to exactly *n_proj*."""
    P0 = a.shape[axis]
    if n_proj == P0:
        return a
    reps = int(np.ceil(n_proj / P0))
    sl = tuple(slice(None) if i != axis else slice(0, n_proj)
               for i in range(a.ndim))
    return np.repeat(a, reps, axis=axis)[sl]


def tile_flat(kc, n_comp, n_proj):
    """Tile the flat config to ``n_wave = n_proj · N``.

    ``N = N0·k`` is the shared-ck length; the projection axis is fitted to
    ``n_proj`` first, then the wave axis is tiled ``k`` times.  The basis is
    already folded by the builder, so the repeated waves select the SAME
    rows: ``gamma_matrix`` grows along its wave axis and its column count is
    unchanged.
    """
    out = dict(kc)
    N0 = int(np.asarray(kc["fl_table"]).shape[0])
    k = max(1, int(round(n_comp / N0)))
    P = int(n_proj)

    fl_tab = np.asarray(kc["fl_table"])          # (N0, n_decay, bins)
    num_tab = np.asarray(kc["num_table"])        # (N0, n_res, bins)
    am = np.asarray(kc["angle_matrix"])          # (N0, P0, n_ab)
    fl_q = np.asarray(kc["fl_q_index"])          # (N0, n_decay)
    num_m = np.asarray(kc["num_m_index"])        # (N0, n_res)
    gam = np.asarray(kc["gamma_matrix"])         # (N0, n_res, nb)

    P0 = am.shape[1]
    out["fl_table"] = np.tile(fl_tab, (k, 1, 1))
    out["num_table"] = np.tile(num_tab, (k, 1, 1))
    out["fl_q_index"] = np.tile(fl_q, (k, 1))
    out["num_m_index"] = np.tile(num_m, (k, 1))
    out["angle_matrix"] = np.tile(_fit_proj(am, 1, P), (k, 1, 1))
    out["gamma_matrix"] = np.tile(gam, (k, 1, 1))
    return out


def _events(kc, ne, seed=123):
    """Events inside the config's own mass/momentum grids."""
    rng = np.random.default_rng(seed)
    n_basis = int(np.asarray(kc["basis_table"]).shape[1])
    n_fl = int(np.asarray(kc["fl_table"]).shape[-1])
    m_min, m_del = float(kc["m_min"]), float(kc["m_delta"])
    q_min, q_del = float(kc["q_min"]), float(kc["q_delta"])
    n_mass = int(np.max(kc["num_m_index"])) + 1
    n_q = int(np.max(kc["fl_q_index"])) + 1
    n_at = int(np.max(kc["angle_index"])) + 1
    nac = int(np.asarray(kc["angle_k"]).shape[1])
    return {
        "mass": m_min + rng.uniform(0.05, 0.95, (ne, n_mass)) * m_del * n_basis,
        "q": q_min + rng.uniform(0.05, 0.95, (ne, n_q)) * q_del * n_fl,
        "angle": rng.uniform(-np.pi, np.pi, (ne, n_at, nac)),
        "weight": np.ones(ne),
        "bkg": np.full(ne, 0.01),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-comp", type=int, default=100,
                    help="N, the shared-ck length PER PROJECTION")
    ap.add_argument("--n-proj", type=int, default=2)
    ap.add_argument("--preset", default="large", choices=sorted(PRESETS))
    ap.add_argument("--compute-sizes", default=None,
                    help="comma separated event counts (overrides the preset)")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--no-numerator", action="store_true",
                    help="force every numerator row to the constant 1")
    args = ap.parse_args()

    f = Fitter(TEMPLATE)
    f.apply_constrains()
    kc = tile_flat(f.kernel_config, args.n_comp, args.n_proj)
    if args.no_numerator:
        kc = dict(kc)
        kc["num_table"] = np.ones_like(np.asarray(kc["num_table"]))

    n_wave = int(np.asarray(kc["fl_table"]).shape[0])
    n_proj = int(np.asarray(kc["angle_matrix"]).shape[1])
    n_res = int(np.asarray(kc["num_table"]).shape[1])
    n_basis = int(np.asarray(kc["basis_table"]).shape[0])
    gam = np.asarray(kc["gamma_matrix"])
    print("=" * 78)
    print(f"flat pwa_gen: N={n_wave} n_proj={n_proj} n_res={n_res} "
          f"n_basis={n_basis}  ({TEMPLATE})"
          + ("   [numerator = 1]" if args.no_numerator else ""))
    print(f"  gamma_matrix {gam.shape} nnz={int((gam != 0).sum())} "
          f"density={ (gam != 0).sum() / gam.size:.3f}")
    print("=" * 78)

    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    rng = np.random.default_rng(7)
    ck0 = np.asarray(params["ck"])
    params = dict(params)
    params["ck"] = np.tile(ck0, (n_wave + ck0.size - 1) // ck0.size)[:n_wave]
    params["coeff"] = rng.uniform(0.3, 1.2, n_basis)
    norm = None

    sizes = ([int(s) for s in args.compute_sizes.split(",") if s]
             if args.compute_sizes else PRESETS[args.preset])
    header = (f"{'n_events':>10}" + "".join(f" | {n:>24}" for n in GPU))
    print(header)
    print("-" * len(header))
    for ne in sizes:
        data = _events(kc, ne)
        norm = float(ne) * 0.5
        insts = {}
        for name in GPU:
            try:
                k = create_backend(name, kc, model="pwa_gen")
                insts[name] = (k, k.load_data(data))
            except Exception as exc:                       # noqa: BLE001
                print(f"  {name} SKIP: {type(exc).__name__}: {str(exc)[:60]}")
        if not insts:
            continue
        times = {n: [] for n in insts}
        for _ in range(args.warmup):
            for k, h in insts.values():
                k.compute(params, h, norm=norm)
        for _ in range(args.trials):
            for name, (k, h) in insts.items():
                t0 = time.perf_counter()
                k.compute(params, h, norm=norm)
                times[name].append((time.perf_counter() - t0) * 1e3)
        cells = []
        for name in GPU:
            if name not in insts:
                cells.append(f"{'SKIP':>24}")
                continue
            t = min(times[name])
            cells.append(f"{ne / t / 1e3:8.2f} M/s {t:8.2f} ms")
        print(f"{ne:>10}" + "".join(f" | {c:>24}" for c in cells))
        for k, h in insts.values():
            k.free()

    if len(times) == 2:
        base = min(times[GPU[0]])
        got = min(times[GPU[1]])
        print(f"\n  cache / flat = {base / got:.2f}x  "
              f"(cached F + basis + fa vs none)")


if __name__ == "__main__":
    main()
