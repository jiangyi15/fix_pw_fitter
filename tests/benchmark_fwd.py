#!/usr/bin/env python3
"""Cost of :mod:`tabpwa.fwd` forward-mode duals.

Reports per-operation wall time for the two regimes a PWA fit sees —
`n_var ≈ 50` (mass/width-only) and `n_var ≈ 1000` (`ck` free) — and the
overhead of a ratio built from duals versus plain floats, so the numbers can
be compared with a kernel ``compute`` call (~0.5–30 ms).

Usage::

    python tests/benchmark_fwd.py
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tabpwa.fwd import Fwd, Layout  # noqa: E402


def _bench(fn, repeat=20000):
    for _ in range(200):                      # warmup
        fn()
    best = np.inf
    trials = 5
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(repeat):
            fn()
        best = min(best, time.perf_counter() - t0)
    return best / repeat * 1e6                # µs


def main():
    rows = []
    for n in (50, 1000):
        lay = Layout([f"p{i}" for i in range(n)])
        a = Fwd.of(2.0, {f"p{i}": 1.0 for i in range(n)}, lay)
        b = Fwd.of(3.0, {f"p{i}": -1.0 for i in range(n)}, lay)
        rows.append((
            n,
            _bench(lambda: a + b),
            _bench(lambda: a * b),
            _bench(lambda: a / b),
            _bench(lambda: a.sqrt()),
            _bench(lambda: Fwd.of(1.0, {"p0": 1.0}, lay)),
        ))

    print(f"{'n_var':>6} {'a+b':>8} {'a*b':>8} {'a/b':>8} {'sqrt':>8} {'create':>8}"
          "   (µs/op)")
    for n, add, mul, div, sq, new in rows:
        print(f"{n:6d} {add:8.3f} {mul:8.3f} {div:8.3f} {sq:8.3f} {new:8.3f}")

    for n in (50, 1000):
        lay = Layout([f"p{i}" for i in range(n)])
        num = Fwd.of(3.0, {f"p{i}": 0.5 for i in range(n)}, lay)
        den = Fwd.of(12.0, {f"p{i}": -0.25 for i in range(n)}, lay)

        def ratio():
            for _ in range(100):
                num / den

        def plain():
            for _ in range(100):
                3.0 / 12.0

        print(f"n_var={n:5d}: 100 ratios {_bench(ratio, 2000):7.1f} µs   "
              f"plain floats {_bench(plain, 2000):7.1f} µs")

    print("\nreference: one kernel compute() over 200k events ≈ 0.5–30 ms")


if __name__ == "__main__":
    sys.exit(main())
