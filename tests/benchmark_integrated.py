#!/usr/bin/env python3
"""Benchmark IntegratedBackend: Gram matrix norm + base backend for data NLL.

The Integrated backend pre-computes Gram matrices from phsp data for
fast O(ng²) norm evaluation, while the base backend (cuda_v3) handles
per-event data NLL.  During BFGS, norm is evaluated at every iteration
while data NLL is only needed once per iteration.

This benchmark tests the full ``compute(params, data, norm=norm)`` path
with a realistic data : phsp ratio of 1 : 10.
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ampfit.config_loader import Config
from ampfit.backends import create_backend

CONFIG_FILE = "config_angle.yml"
WARMUP = 5
TRIALS = 20
BATCH_SIZES = [64, 256, 1024, 10000]

config = Config(CONFIG_FILE)
kc = config.build_all_index()
ck_map = config.get_ck_map()

rng = np.random.default_rng(42)
params = {
    'ck': rng.normal(size=len(ck_map)) + 1j * rng.normal(size=len(ck_map)),
    'm0': rng.random(int(np.max(kc['m0_index'])) + 1) + 2,
    'g0': rng.random(int(np.max(kc['g0_index'])) + 1) + 0.1,
    'scalar': [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
}

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
print("  IntegratedBackend Benchmarks (cuda_v3 base)")
print("  data:phsp = 1:10")
print("=" * 70)

for label, bname in [("CUDAv3 (ref)", "cuda_v3"),
                     ("Integrated",  "integrated")]:
    print(f"\n── {label} ──")
    print(f"{'n_data':>8} | {'n_phsp':>8} | {'Full NLL (norm+data)':>20}")
    print("-" * 45)

    for n in BATCH_SIZES:
        n_data = n
        n_phsp = n * 10

        # Create data and phsp
        data = make_data(n_data)
        phsp = make_data(n_phsp)
        phsp['bkg'] = np.zeros(n_phsp)

        be = create_backend(bname, kc)
        dh_data = be.load_data(data)
        dh_phsp = be.load_data(phsp)  # also triggers Gram pre-computation for Integrated

        # Full NLL: compute norm from phsp, then NLL from data
        norm, _, _ = be.compute(params, dh_phsp, norm=None, return_p=False)
        norm = float(norm)
        for _ in range(WARMUP):
            be.compute(params, dh_data, norm=norm)
        t0 = time.perf_counter()
        for _ in range(TRIALS):
            be.compute(params, dh_data, norm=norm)
        t_full = (time.perf_counter() - t0) / TRIALS

        eps_full = n_data / t_full

        print(f"  {n_data:>5}  | {n_phsp:>5}  | {eps_full:>15.0f}/s")

        del dh_data, dh_phsp, be

print("\n" + "=" * 70)
print("  Full NLL = norm(phsp, return_p=False) + data_NLL(data)")
print("  data : phsp = 1 : 10")
print("=" * 70)
