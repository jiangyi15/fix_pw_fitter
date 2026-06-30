#!/usr/bin/env python3
"""Benchmark all backends at various batch sizes."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ampfit.config_loader import Config
from ampfit.backends import create_backend

CONFIG_FILE = "config_angle.yml"
WARMUP = 5
TRIALS = 20
BATCH_SIZES = [64, 256, 1024]

def make_data(n):
    rng = np.random.default_rng(42)
    return {
        'mass': rng.uniform(2, 3, (n, 48)),
        'q': rng.uniform(0, 1, (n, 72)),
        'angle': rng.uniform(-np.pi, np.pi, (n, 24, 3)),
        'frac': rng.random(n),
        'time': rng.random(n),
        'bkg': rng.random(n) * 0.01,
        'weight': np.ones(n),
    }

config = Config(CONFIG_FILE)
kc = config.build_all_index()
ck_map = config.get_ck_map()

rng = np.random.default_rng(42)
params = {
    'ck': rng.normal(size=len(ck_map)) + 1j * rng.normal(size=len(ck_map)),
    'm0': rng.random(len(kc["m0_index"])) + 2,
    'g0': rng.random(len(kc["g0_index"])) + 0.1,
    'scalar': [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
}

backends = [
    ("NumPy",   "numpy"),
    ("CUDAv3",  "cuda_v3"),
    ("CUDA32v3","cuda32_v3"),
    ("CUDAv2",  "cuda_v2"),
    ("CUDA32v2","cuda32_v2"),
]

print(f"{'n_events':>8}", end="")
for label, _ in backends:
    print(f" | {label:>10}", end="")
print()
print("-" * (8 + 14 * len(backends)))

for n in BATCH_SIZES:
    sys.stdout.write(f"  n={n:>5}...")
    sys.stdout.flush()
    data = make_data(n)
    rates = []
    for label, bname in backends:
        try:
            be = create_backend(bname, kc)
            dh = be.load_data(data)
            for _ in range(WARMUP):
                be.compute(params, dh)
            t0 = time.perf_counter()
            for _ in range(TRIALS):
                be.compute(params, dh)
            t = (time.perf_counter() - t0) / TRIALS
            del dh, be
            eps = n / t
            rates.append(f"{eps:>10.0f}/s")
        except Exception as e:
            rates.append(f"{'SKIP':>10}")
    print(f"\r  {n:>5}    | {' | '.join(rates)}")

print("-" * (8 + 14 * len(backends)))
