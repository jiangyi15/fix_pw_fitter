"""How much does a FACTOR-level (bw) dedup buy on a multi-BW config?

`cuda_flat_pwa_gen2` collapses whole WAVE blocks whose entire gamma block is
identical (`slot_map`, key = the block bytes), so it pays only when two waves are
copies of each other.  It does NOT collapse the individual FACTOR rows across
waves: two topologies that share one resonance (and differ in the other) still
compute that shared factor twice.

`tests/config_pwa_gen_mb2.yml` is the fixture for that case (three topologies
sharing `pipi`).  This harness reports, on the host, how many denominator dots
per event each scheme needs, and times one backend so a copy has a baseline.
Nothing here is a test: the counts are the CONFIG's structure, the timings are
the tile's, and the tile has been measured fp64-issue/barrier bound (s0e5's
130 -> 74 nnz wave dedup was neutral), so expect the forward dots to move
little and the backward/atomics to be the win.

The factor-level copy is `cuda_flat_pwa_gen_fslot` (backward sweep over distinct
factor rows, gated to `n_res > 1` AND `n_fslot < n_slot·n_res` — a real saving
against the parent's own dot count, so `_mb` stays on the parent path); its
numbers are gated in `tests/test_flat_dedup.py`.

    python tests/benchmark_mb_factor.py                     # gen2, both fixtures
    python tests/benchmark_mb_factor.py --n 20000            # sample size
    python tests/benchmark_mb_factor.py --backend cuda_flat_pwa_gen_fslot
"""
import argparse
import copy
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tabpwa import Fitter                                    # noqa: E402
from tabpwa.amp_model import build_amplitude_model            # noqa: E402

MB1 = os.path.join(os.path.dirname(__file__), "config_pwa_gen_mb.yml")
MB2 = os.path.join(os.path.dirname(__file__), "config_pwa_gen_mb2.yml")


def factor_counts(kc):
    """(wave blocks, factor rows) — the two dedup levels, by their own keys."""
    g = np.asarray(kc["gamma_matrix"], float)
    nw, nr, nb = g.shape
    # both keys are the gamma ROW (what `slot_map` hashes); the basis mass
    # column is a property of the BASIS row, not of (wave, factor)
    wave_blocks = len({tuple(r.tobytes() for r in g[k]) for k in range(nw)})
    factor_rows = len({g[k, r].tobytes()
                       for k in range(nw) for r in range(nr)})
    n_nz = int(np.count_nonzero(g))
    return dict(n_wave=nw, n_res=nr, n_basis=nb, rows=nw * nr,
                wave_blocks=wave_blocks, factor_rows=factor_rows,
                dots_wave=wave_blocks * nr, dots_factor=factor_rows,
                n_nz=n_nz)


def report(path):
    with open(path) as fh:
        import yaml
        dic = yaml.safe_load(fh)
    kc = build_amplitude_model(dic).build_kernel_config()
    c = factor_counts(kc)
    save = 100.0 * (1.0 - c["dots_factor"] / c["dots_wave"])
    print(f"\n{os.path.basename(path)}")
    print(f"  n_wave {c['n_wave']}  n_res {c['n_res']}  n_basis {c['n_basis']}  "
          f"n_nz {c['n_nz']}   (wave,factor) rows {c['rows']}")
    print(f"  distinct WAVE blocks {c['wave_blocks']}   distinct FACTOR rows "
          f"{c['factor_rows']}")
    print(f"  denominator dots/event: wave-level {c['dots_wave']}  "
          f"factor-level {c['dots_factor']}  ({save:+.0f} %)")
    return c


def time_backend(path, n, backend="cuda_flat_pwa_gen2", reps=3):
    """min-of-`reps` `get_nll` (data + phase space) — the fit's own cost."""
    f = Fitter(path, backend=backend)
    f.apply_constrains()
    x0 = f.initial_values(seed=1)
    f.set_phsp(f.generate_phsp(max(2000, n // 4), seed=1))
    f.set_data(f.generate_toy(x0, n, seed=2))
    f.get_nll(x0)                                    # warm-up (kernel + load)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f.get_nll(x0)
        ts.append(time.perf_counter() - t)
    t = min(ts)
    print(f"  {backend}: get_nll {1e3 * t:.2f} ms  "
          f"({n / t / 1e3:.2f} kev/s on a {n}-event sample), min of {reps}")
    f.free()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="data events")
    ap.add_argument("--backend", default="cuda_flat_pwa_gen2")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-time", action="store_true")
    args = ap.parse_args()
    for path in (MB1, MB2):
        report(path)
        if not args.no_time:
            try:
                time_backend(path, args.n, args.backend, args.reps)
            except Exception as exc:              # a box without CUDA
                print(f"  {args.backend}: skipped ({type(exc).__name__}: "
                      f"{str(exc)[:70]})")


if __name__ == "__main__":
    main()