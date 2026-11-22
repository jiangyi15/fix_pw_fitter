#!/usr/bin/env python3
"""fp32 vs fp64 ``cuda_flat_pwa_gen_ampcache`` on the CTPWA kketa shape.

The fp64 parent is MEASURED fp64-pipe bound (`ncu`: FP64 pipe 85 % of peak,
DRAM 11 %, L2 13 %, 40 % of warp cycles in CTA-barrier stalls).  Its
per-iteration work is a bilinear contraction over the pinned cache, so the
fp32 variant (`cuda_flat_pwa_gen_ampcache32`) halves the staged bytes and moves
the MACs to the fp32 pipe while leaving the FILL, the GRAM and every reduction
in fp64.  This script measures what that is worth, interleaved (the box
throttles, so sequential numbers are meaningless):

  * ``ampcache``    — the fp64 parent (a `float2` cache, fp64 arithmetic);
  * ``ampcache32``  — this variant (a `float2` cache, fp32 arithmetic, fp64
                      reductions).

Same sample, same config, same `Fitter.get_nll` (norm + data + gradients), min
of ``--repeat`` interleaved per sweep and min over ``--sweeps``.

``load`` never fills: the ONE lifecycle fills a handle on its FIRST compute,
so ``first`` carries the ``Afix`` fill and ``steady`` carries neither.

    python tests/benchmark_ampcache32.py                      # 0.3M + 0.3M
    python tests/benchmark_ampcache32.py --n-data 1000000 --n-phsp 5000000

The kernel-resource section prints ``kernel_stats()`` for both: the fp32
variant should need the same or fewer registers, and a non-zero ``local_bytes``
(spills) would invalidate the comparison — the script says so.

MEASURED (this box, min-of-5 interleaved over 2 sweeps, CTPWA kketa 34 waves x
2 projections, 544 B/event cache, `rsize` 1):

    sample                fp64 steady     fp32 steady     ratio
    0.3 M + 0.3 M          4.2 ms          2.3 ms         1.79x
       ns/event            6.94            3.87
       cache stream       117.6 GB/s      210.9 GB/s
    1 M + 5 M             33.8 ms         14.3 ms         2.36x
       ns/event            5.64            2.39
       cache stream       112.6 GB/s      265.7 GB/s

and `ncu` on ONE 600 k-event chunk (326 MB of cache, past L2): 1.47 ms vs
5.85 ms (3.98x), DRAM 34 % vs 8.6 %, fp64 pipe 56 % vs 87 %, long-scoreboard
stall 40 % vs 3.9 %, barrier stall 42 % vs 41 %.  So it did NOT become
memory-bound: the fp32 contraction is gone from the fp64 pipe, but the fp64
SCALAR edge the contract keeps (`dqp = weight/norm·scale` per element, and the
fp64 `P_e` accumulation) is now the largest single pipe consumer, and the tile
still pays ~40 % of warp cycles on barriers.  Turning the fp64 pipe near idle
would need those moved too, which is exactly what "reductions stay fp64"
forbids.
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
from benchmark_s0f import _moving_columns, _pin_other_mw


def _load_sample(fitter, cfg, n_data, n_phsp, seed):
    """The config's own sample when it exists, else synthetic in-grid events."""
    data_path = fitter.config.dic.get("data", {}).get("data_arr")
    phsp_path = fitter.config.dic.get("data", {}).get("phsp_arr")
    base = os.path.dirname(os.path.abspath(cfg))
    if data_path:
        data_path = os.path.join(base, data_path)
    if phsp_path:
        phsp_path = os.path.join(base, phsp_path)
    if data_path and os.path.exists(data_path):
        data = fitter.load_dataset(
            os.path.basename(data_path).split("_arr")[0], path=data_path,
            max_events=n_data or None)
        phsp = fitter.load_dataset(
            os.path.basename(phsp_path).split("_arr")[0], path=phsp_path,
            max_events=n_phsp or None)
        return data, phsp
    from benchmark_flat_vs_pwa import _events
    kc = fitter.kernel_config
    return (_events(kc, n_data or 4000, seed),
            _events(kc, n_phsp or 4000, seed + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default="benchmarks/ctpwa_kketa/config_gen.yml")
    ap.add_argument("--rsize", type=int, default=1)
    ap.add_argument("--n-data", type=int, default=300000)
    ap.add_argument("--n-phsp", type=int, default=300000)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=50000)
    ap.add_argument("--keep-free-mw", default="")
    args = ap.parse_args()

    f = Fitter(args.config)
    f.apply_constrains()
    free_mw = tuple(s.strip() for s in args.keep_free_mw.split(",") if s.strip())
    _pin_other_mw(f, free_mw)
    kc = f.kernel_config
    x = f.initial_values(seed=1)
    n_basis = int(np.asarray(kc["basis_table"]).shape[0])

    moving = _moving_columns(f, x)
    if moving.size:
        den = [str(v) for v in np.asarray(kc["den_names"]).ravel()]
        names = [den[int(i)] for i in moving[:6]]
        raise SystemExit(
            "benchmark_ampcache32.py: the coefficients are not pinned — "
            f"{moving.size}/{n_basis} columns still move ({names}…); pin the "
            "parameters that move them or this is not an all-pinned compare")

    data, phsp = _load_sample(f, args.config, args.n_data, args.n_phsp, seed=7)

    specs = [("ampcache", "cuda_flat_pwa_gen_ampcache"),
             ("ampcache32", "cuda_flat_pwa_gen_ampcache32")]

    def measure(name):
        be = create_backend(name, kc, model="pwa_gen",
                           resolution_size=args.rsize,
                           batch_size=args.batch_size)
        try:
            f.backend = be
            if f._data_holder is not None:
                f._data_holder.free()
                f._data_holder = None
            t0 = time.perf_counter()
            f.set_phsp(phsp)
            f.set_data(data)
            load = time.perf_counter() - t0
            t1 = time.perf_counter()
            f.get_nll(x)
            first = time.perf_counter() - t1
            f.get_nll(x)                      # warm-up, not timed
            ts = []
            for _ in range(args.repeat):
                t = time.perf_counter()
                f.get_nll(x)
                ts.append(time.perf_counter() - t)
            return load, first, ts, be.kernel.kernel_stats()
        finally:
            if f._data_holder is not None:
                f._data_holder.free()
                f._data_holder = None
            if f._phsp_holder is not None:
                f._phsp_holder.free()
                f._phsp_holder = None
            f._phsp_scratch = None
            be.free()

    ne_d = len(np.asarray(data["weight"]))
    ne_p = len(np.asarray(phsp["weight"]))
    n_ev = ne_d + ne_p
    n_wave = int(np.asarray(kc["fl_table"]).shape[0])
    n_proj = max(1, int(np.asarray(kc["matrix_angle"]).shape[1]) // n_wave)
    #: cache bytes MOVED per get_nll: the data handle is walked twice (forward
    #: A, backward dck) and the phase-space handle once (the norm's forward)
    cache_per_iter = 8 * n_proj * n_wave * (2 * ne_d + ne_p)

    print(f"config      {args.config}")
    print(f"shape       n_wave {n_wave}  n_proj {n_proj}  n_basis {n_basis}  "
          f"rsize {args.rsize}")
    print(f"events      data {ne_d}  phsp {ne_p}  total {n_ev}")
    print(f"pinned      every coefficient (moving columns 0/{n_basis}); "
          f"x has {len(x)} free slot(s)")

    steady = {label: float("inf") for label, _n in specs}
    load_t, first, stats = {}, {}, {}
    for _sweep in range(args.sweeps):
        for label, name in specs:
            ld, fi, ts, st = measure(name)
            load_t.setdefault(label, ld)
            first.setdefault(label, fi)
            stats[label] = st
            steady[label] = min(steady[label], min(ts))

    print()
    best = min(steady.values())
    for label, _n in specs:
        sms = steady[label] * 1e3
        print(f"{label:12s} load {load_t[label] * 1e3:8.1f} ms   "
              f"first {first[label] * 1e3:7.1f} ms   steady {sms:8.1f} ms   "
              f"{n_ev / steady[label] / 1e6:7.2f} Mev/s   "
              f"{sms * 1e6 / n_ev:6.2f} ns/event   "
              f"{cache_per_iter / steady[label] / 1e9:6.1f} GB/s   "
              f"{sms / (best * 1e3):5.2f}x")
    print("  `load` never fills: the ONE lifecycle fills a handle on its FIRST "
          "compute, so `first` carries the Afix fill and `steady` does not.")
    print(f"  GB/s is the CACHE stream: 8·n_proj·n_wave·(2·ne_data + ne_phsp) "
          f"= {cache_per_iter / 1e9:.1f} GB moved per iteration.")

    ratio = steady["ampcache"] / steady["ampcache32"]
    verdict = ("faster" if ratio > 1.02
               else "SLOWER" if ratio < 0.98 else "equal (within 2 %)")
    print()
    print(f"ampcache32 vs ampcache (fp64): {ratio:.2f}x -> {verdict} at the "
          f"SAME {8 * n_proj * n_wave} B/event cache")

    print()
    for label, _n in specs:
        st = stats[label]
        spill = " SPILLS" if st["local_bytes"] else ""
        print(f"{label:12s} tile regs {st['regs']:3d}  local {st['local_bytes']}"
              f" B  blocks/SM {st['blocks_per_sm']:2d}  "
              f"warps/SM {st['warps_per_sm']:3d}  "
              f"max_threads {st['max_threads']:4d}{spill}")
    if stats["ampcache32"]["local_bytes"] or stats["ampcache"][
            "local_bytes"]:
        print("  -> a SPILLING kernel makes this comparison meaningless: the "
              "register budget is a build choice, not the arithmetic.")


if __name__ == "__main__":
    main()
