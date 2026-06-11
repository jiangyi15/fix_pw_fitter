#!/usr/bin/env python3
"""Benchmark original CUDA vs merged-index CUDA kernel."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ampfit.config_loader import Config
from ampfit._cuda import CUDAKernel
from ampfit._cuda_merged import CUDAMergedKernel

CONFIG_FILE = "config_angle.yml"
WARMUP = 5
TRIALS = 20
BATCH_SIZES = [64, 256, 1024, 4096, 8192]

np.random.seed(42)
config = Config(CONFIG_FILE)
kernel_config = config.build_all_index()
ck_map = config.get_ck_map()

base_params = {
    'ck': np.random.randn(len(ck_map)) + 1j*np.random.randn(len(ck_map)),
    'm0': np.random.rand(len(config.m0_phys_name)) + 2,
    'g0': np.random.rand(len(config.g0_phys_name)) + 0.1,
    'scalar': [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
}

def make_data(n):
    np.random.seed(42)
    return {
        'mass': np.random.random((n, 48)),
        'q': np.random.random((n, 72)),
        'angle': np.random.random((n, 24, 3)),
        'frac': np.random.random((n,)),
        'time': np.random.random((n,)),
        'bkg': np.random.random((n,)) * 0.01,
        'weight': np.ones((n,)),
    }

print("=" * 65)
print("  CUDA Merged-Index Kernel Benchmark (RTX 3070 Ti)")
print("  Trade-off: 896 BW evals vs 216 (4.1×) but no scatter step")
print("=" * 65)
print(f"{'n_events':>8} | {'Original':>10} | {'Merged':>10} | {'Ratio':>7} | {'Winner':>8}")
print("-" * 50)

for n in BATCH_SIZES:
    sys.stdout.write(f"  n={n:>5}..."); sys.stdout.flush()
    data = make_data(n)

    # Original CUDA
    orig = CUDAKernel(kernel_config)
    dh = orig.load_data(data)
    for _ in range(WARMUP):
        orig.compute(base_params, dh)
    t0 = time.perf_counter()
    for _ in range(TRIALS):
        orig.compute(base_params, dh)
    t_orig = (time.perf_counter() - t0) / TRIALS * 1000
    orig.free()

    # Merged-index CUDA
    test = CUDAMergedKernel(kernel_config)
    dh2 = test.load_data(data)
    for _ in range(WARMUP):
        test.compute(base_params, dh2)
    t0 = time.perf_counter()
    for _ in range(TRIALS):
        test.compute(base_params, dh2)
    t_merged = (time.perf_counter() - t0) / TRIALS * 1000
    test.free()

    ratio = t_merged / t_orig
    winner = "Original" if ratio > 1.05 else "Merged" if ratio < 0.95 else "~Same"
    print(f"\r  n={n:>5} | {t_orig:>8.3f}ms | {t_merged:>8.3f}ms | {ratio:>6.3f} | {winner:>8}")

print("-" * 50)
print("  Ratio > 1 = Original faster, < 1 = Merged-index faster")
print("=" * 65)
