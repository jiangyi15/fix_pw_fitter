#!/usr/bin/env python3
"""A/B ``cuda_flat_pwa_gen_s0e`` vs ``cuda_flat_pwa_gen_s0e2`` — the ILP change.

s0e2 is a case-1 copy of s0e whose tile kernel differs in exactly two
instruction-level ways:

1. the CSR step is ONE packed 64-bit load (``m | (u << 32)``) instead of two
   4-byte loads from ``nz_m`` / ``nz_u``;
2. the sparse dot runs TWO independent fp64 accumulation chains, so a wave
   with ``nnz(k) > 1`` does not serialize on a single FMA dependency chain.

This script measures what that buys: ``us/iter`` and ``ns/event`` on the
``pwa_gen`` fixture, at ``ne`` 4000 / 50000, ``rsize`` 1 / 17 (per-event and
grouped objective), ``norm`` None / 1.3.

The GPU on this machine throttles hard inside a session (observed 180 MHz ..
2730 MHz), so both kernels are kept alive and **interleaved** every round and
the MIN per variant is reported — that is the same discipline
``benchmark_flat_pwa_gen.py`` uses.  Each round is a steady-state loop of 20
computes after one warm-up; ``--rounds`` sets how many rounds are interleaved.

NO timing assertions: the machine may be busy.  The script only runs and
prints.

    python tests/benchmark_s0e_ilp.py
    python tests/benchmark_s0e_ilp.py --ne 4000 --rsize 17 --rounds 5

Profiling the two tile kernels (they live in different ``.so`` files and have
different symbol names, so a ``regex:`` filter picks the right one):

    ncu --target-processes all --kernel-name regex:gps0_tile_kernel \
        --set full -o /tmp/opencode/s0e_tile python tests/benchmark_s0e_ilp.py
    ncu --target-processes all --kernel-name regex:gps02_tile_kernel \
        --set full -o /tmp/opencode/s0e2_tile python tests/benchmark_s0e_ilp.py
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

CFG = "tests/config_pwa_gen_km.yml"
VARIANTS = ("cuda_flat_pwa_gen_s0e", "cuda_flat_pwa_gen_s0e2")
#: the tile kernel symbol of each variant, for the printed `ncu` commands
KERNELS = {"cuda_flat_pwa_gen_s0e": "gps0_tile_kernel",
           "cuda_flat_pwa_gen_s0e2": "gps02_tile_kernel"}


def _events(kc, ne, seed=123):
    """Events strictly inside both interpolation grids (clamping diverges)."""
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


def _time_us(fn, iters):
    """Wall time of ``iters`` calls, in microseconds per call."""
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=CFG)
    ap.add_argument("--ne", default="4000,50000")
    ap.add_argument("--rsize", default="1,17")
    ap.add_argument("--norm", default="none,1.3")
    ap.add_argument("--iters", type=int, default=20,
                    help="computes per steady-state round (after 1 warm-up)")
    ap.add_argument("--rounds", type=int, default=5,
                    help="interleaved rounds; the MIN per variant is reported "
                         "and the order is reversed every round so a clock "
                         "ramp cannot favour one variant")
    ap.add_argument("--batch-size", type=int, default=50000)
    args = ap.parse_args()

    nes = [int(x) for x in args.ne.split(",") if x]
    rsizes = [int(x) for x in args.rsize.split(",") if x]
    norms = [None if x.strip().lower() in ("none", "null", "")
             else float(x) for x in args.norm.split(",") if x.strip()]

    f = Fitter(args.cfg)
    f.apply_constrains()
    kc = f.kernel_config
    params, _resolved = f.build_params(f.initial_values(seed=1))

    print(f"# {args.cfg}: n_wave {int(np.asarray(kc['fl_table']).shape[0])}, "
          f"n_basis {int(np.asarray(kc['basis_table']).shape[0])}; "
          f"{args.iters} computes/round, {args.rounds} interleaved rounds, "
          f"min reported")
    print(f"# variants: {', '.join(VARIANTS)}")
    hdr = (f"{'ne':>7} {'rsize':>5} {'norm':>5} "
           + " ".join(f"{v.replace('cuda_flat_pwa_gen_', ''):>21}" for v in VARIANTS)
           + "   ratio(s0e/s0e2)")
    print(hdr)
    print("-" * len(hdr))

    for ne in nes:
        data = _events(kc, ne)
        for rs in rsizes:
            bes = {}
            hs = {}
            for v in VARIANTS:
                be = create_backend(v, kc, model="pwa_gen",
                                    resolution_size=rs,
                                    batch_size=args.batch_size)
                bes[v] = be
                hs[v] = be.load_data(data)
            try:
                for norm in norms:
                    cells = {}
                    for v in VARIANTS:
                        # one warm-up, then interleaved steady-state rounds
                        bes[v].compute(params, hs[v], norm=norm)
                    for _ in range(args.rounds):
                        # reverse the order every round: a clock ramp then
                        # cannot favour one variant
                        for v in (VARIANTS if _ % 2 == 0
                                  else VARIANTS[::-1]):
                            us = _time_us(
                                lambda v=v, norm=norm: bes[v].compute(
                                    params, hs[v], norm=norm),
                                args.iters)
                            cells[v] = min(cells.get(v, 1e30), us)
                    a, b = (cells[VARIANTS[0]], cells[VARIANTS[1]])
                    nm = "None" if norm is None else f"{norm:g}"
                    print(f"{ne:>7} {rs:>5} {nm:>5} "
                          + " ".join(f"{cells[v]:>9.1f} us "
                                     f"{cells[v] * 1e3 / ne:>7.1f} ns/e"
                                     for v in VARIANTS)
                          + f"   {a / b:>6.3f}")
            finally:
                for v in VARIANTS:
                    hs[v].free()
                    bes[v].free()

    print()
    print("# profile the two tile kernels (different .so, different symbols):")
    for v in VARIANTS:
        print(f"  ncu --target-processes all "
              f"--kernel-name regex:{KERNELS[v]} --set full "
              f"-o /tmp/opencode/{v.replace('cuda_flat_pwa_gen_', '')}_tile "
              f"python tests/benchmark_s0e_ilp.py")


if __name__ == "__main__":
    main()
