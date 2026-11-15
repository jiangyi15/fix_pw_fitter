#!/usr/bin/env python3
"""Cost of :mod:`tabpwa.fwd` forward-mode duals.

Reports per-operation wall time for the two regimes a PWA fit sees —
`n_var ≈ 50` (mass/width-only) and `n_var ≈ 1000` (`ck` free) — and the
overhead of a ratio built from duals versus plain floats, so the numbers can
be compared with a kernel ``compute`` call (~0.5–30 ms).

There is no ``Layout``: a dual's gradient is a dense vector, so the parameter
space is simply its length.  ``Fwd(value, grad)`` is the constructor;
``Fitter.fwd_of`` / ``Fitter.fwd_parameter`` build them from names.

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

from tabpwa.fwd import Fwd  # noqa: E402


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
        a = Fwd(2.0, np.ones(n))
        b = Fwd(3.0, -np.ones(n))
        seed = np.zeros(n)
        seed[0] = 1.0
        rows.append((
            n,
            _bench(lambda: a + b),
            _bench(lambda: a * b),
            _bench(lambda: a / b),
            _bench(lambda: a.sqrt()),
            _bench(lambda: Fwd(1.0, seed)),
        ))

    print(f"{'n_var':>6} {'a+b':>8} {'a*b':>8} {'a/b':>8} {'sqrt':>8} {'create':>8}"
          "   (µs/op)")
    for n, add, mul, div, sq, new in rows:
        print(f"{n:6d} {add:8.3f} {mul:8.3f} {div:8.3f} {sq:8.3f} {new:8.3f}")

    for n in (50, 1000):
        num = Fwd(3.0, np.full(n, 0.5))
        den = Fwd(12.0, np.full(n, -0.25))

        def ratio():
            for _ in range(100):
                num / den

        def plain():
            for _ in range(100):
                3.0 / 12.0

        print(f"n_var={n:5d}: 100 ratios {_bench(ratio, 2000):7.1f} µs   "
              f"plain floats {_bench(plain, 2000):7.1f} µs")

    # sigma(cov) is one quadratic form: n^2 work, the only O(n^2) operation
    for n in (50, 1000):
        d = Fwd(1.0, np.full(n, 1.0 / n))
        cov = np.eye(n)
        print(f"n_var={n:5d}: sigma(eye) {_bench(lambda: d.sigma(cov), 200):7.1f} µs")

    print("\nreference: one kernel compute() over 200k events ≈ 0.5–30 ms")


if __name__ == "__main__":
    sys.exit(main())
