#!/usr/bin/env python3
"""Benchmark CUDA v2 (linear interp) vs v3 (Catmull-Rom) at various sizes."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ampfit.config_loader import Config
from ampfit.backends import create_backend

CONFIG_FILE = "config_angle.yml"
WARMUP = 5
TRIALS = 20
BATCH_SIZES = [64, 256, 1024, 4096]

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

def make_data(n):
    rng = np.random.default_rng(42)
    return {
        'mass': rng.uniform(2, 3, (n, 48)),
        'q': rng.uniform(0, 1, (n, 72)),
        'angle': rng.uniform(-np.pi, np.pi, (n, 24, 3)),
        'frac': rng.random(n), 'time': rng.random(n),
        'bkg': rng.random(n) * 0.01, 'weight': np.ones(n),
    }

print("=" * 65)
print("  CUDA v2 (linear) vs v3 (Catmull-Rom) — RTX 3070 Ti")
print("=" * 65)
print(f"{'n_events':>8} | {'v2 f64':>10} | {'v2 f32':>10} | {'v3 f64':>10} | {'v3 f32':>10}")
print("-" * 55)

for n in BATCH_SIZES:
    data = make_data(n)
    rates = []
    for bname in ["cuda_v2", "cuda32_v2", "cuda_v3", "cuda32_v3"]:
        try:
            be = create_backend(bname, kc)
            dh = be.load_data(data)
            for _ in range(WARMUP): be.compute(params, dh)
            t0 = time.perf_counter()
            for _ in range(TRIALS): be.compute(params, dh)
            t = (time.perf_counter() - t0) / TRIALS
            del dh, be
            eps = n / t
            rates.append(f"{eps:>10.0f}/s")
        except Exception as e:
            rates.append(f"{'SKIP':>10}")
    print(f"  {n:>5}    | {' | '.join(rates)}")

print("-" * 55)
print("  v3 uses Catmull-Rom (matches NumPy exactly)")
print("  v2 uses linear interpolation")
