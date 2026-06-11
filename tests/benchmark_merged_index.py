#!/usr/bin/env python3
"""Benchmark: original numpy kernel vs merged-index kernel."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ampfit.config_loader import Config
from ampfit.numpy_kernel import NumpyKernelCorrect
from ampfit.merged_index_kernel import MergedIndexKernel

CONFIG_FILE = "config_angle.yml"
WARMUP = 3
TRIALS = 10
BATCH_SIZES = [64, 256, 1024, 4096]

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

ref_kernel = NumpyKernelCorrect(kernel_config)
test_kernel = MergedIndexKernel(kernel_config)

print("=" * 65)
print("  Merged-Index Kernel Benchmark (NumPy)")
print("  Trade-off: 896 BW evals vs 216 (4.1×) but no scatter step")
print("=" * 65)
print(f"\n{'n_events':>8} | {'Original':>10} | {'Merged':>10} | {'Ratio':>7} | {'Winner':>8}")
print("-" * 50)

for n in BATCH_SIZES:
    sys.stdout.write(f"  n={n:>5}...")
    sys.stdout.flush()
    
    np.random.seed(42)
    data = {
        'mass': np.random.random((n, 48)),
        'q': np.random.random((n, 72)),
        'angle': np.random.random((n, 24, 3)),
        'frac': np.random.random((n,)),
        'time': np.random.random((n,)),
        'bkg': np.random.random((n,)) * 0.01,
        'weight': np.ones((n,)),
    }

    for _ in range(WARMUP):
        ref_kernel._compute(base_params, data)
        test_kernel._compute(base_params, data)

    t0 = time.perf_counter()
    for _ in range(TRIALS):
        ref_kernel._compute(base_params, data)
    t_ref = (time.perf_counter() - t0) / TRIALS * 1000

    t0 = time.perf_counter()
    for _ in range(TRIALS):
        test_kernel._compute(base_params, data)
    t_test = (time.perf_counter() - t0) / TRIALS * 1000

    ratio = t_test / t_ref
    winner = "Original" if ratio > 1.05 else "Merged" if ratio < 0.95 else "~Same"
    print(f"\r  n={n:>5} | {t_ref:>8.2f}ms | {t_test:>8.2f}ms | {ratio:>6.3f} | {winner:>8}")

print("-" * 50)
print("  Ratio > 1 = Original faster, < 1 = Merged-index faster")
print("=" * 65)
