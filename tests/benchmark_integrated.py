#!/usr/bin/env python3
"""Benchmark Fitter.get_nll() — Integrated vs CUDAv3 with data:phsp=1:10.

Uses the Fitter API directly, exercising the full pipeline:
  x → _build_params → resolve → kernel → _flat_gradient → (nll, grad)
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ampfit import Fitter

CONFIG_FILE = "config_amp.yml"
WARMUP = 5
TRIALS = 20
BATCH_SIZES = [64, 256, 1024, 10000]


def make_data(n):
    rng = np.random.default_rng(42)
    return {
        'mass': rng.uniform(2, 3, (n, 48)),
        'q': rng.uniform(0, 1, (n, 72)),
        'angle': rng.uniform(-np.pi, np.pi, (n, 24, 3)),
        'frac': rng.random(n), 'time': rng.random(n),
        'bkg': rng.random(n) * 0.01, 'weight': np.ones(n),
    }


print("=" * 70)
print("  Fitter.get_nll() — full pipeline benchmark")
print("  data : phsp = 1 : 10")
print("=" * 70)

for label, backend in [("cuda_v3",  "cuda_v3"),
                       ("integrated", {"name": "integrated", "base": "cuda_v3"})]:
    print(f"\n── {label} ──")
    print(f"{'n_data':>8} | {'n_phsp':>8} | {'get_nll':>15}")
    print("-" * 35)

    for n in BATCH_SIZES:
        n_data = n
        n_phsp = n * 10

        fitter = Fitter(CONFIG_FILE, backend=backend)
        fitter.set_data(make_data(n_data))
        fitter.set_phsp(make_data(n_phsp))
        x0 = fitter.initial_values()

        for _ in range(WARMUP):
            fitter.get_nll(x0)
        t0 = time.perf_counter()
        for _ in range(TRIALS):
            fitter.get_nll(x0)
        t = (time.perf_counter() - t0) / TRIALS
        eps = n_data / t

        print(f"  {n_data:>5}  | {n_phsp:>5}  | {eps:>13.0f}/s")

print("\n" + "=" * 70)
print("  get_nll includes: constraint resolve, norm(phsp), data NLL,")
print("  gradient backprop, chain gradient, bound gradient")
print("=" * 70)
