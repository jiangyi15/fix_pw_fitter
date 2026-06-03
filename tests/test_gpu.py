#!/usr/bin/env python
"""
Test script to compare GPU and numpy implementations of PWA computation.

Usage: python -m tests.test_gpu  [--waves N] [--events N]
"""

import numpy as np
import time
import sys
import argparse

parser = argparse.ArgumentParser(description='Test GPU PWA implementation')
parser.add_argument('--waves', type=int, default=8, help='Number of waves (even number)')
parser.add_argument('--events', type=int, default=10000, help='Number of events')
args = parser.parse_args()

# Import both implementations
try:
    from tests.ref_numpy import PWAFitter
    numpy_available = True
except ImportError:
    try:
        from ref_numpy import PWAFitter
        numpy_available = True
    except ImportError as e:
        print(f"Warning: Could not import PWAFitter from ref_numpy: {e}")
        numpy_available = False

try:
    from pwa_gpu import PWAGPU
    gpu_available = True
except ImportError as e:
    print(f"Warning: Could not import PWAGPU from pwa_gpu: {e}")
    gpu_available = False
except Exception as e:
    print(f"Warning: GPU initialization failed: {e}")
    gpu_available = False


np.random.seed(42)

# ================================================================
# Generate test configuration
# ================================================================

# Model dimensions
n_events = args.events
n_waves = args.waves
n_m0 = n_waves // 2 + 1
n_g0 = n_waves // 2
n_res_per_wave = 2
n_decays_per_wave = 2
n_bf_types = 3
n_basis = n_waves * 2
n_ang_per_basis = 3

n_mass_cols = 6
n_q_cols = 6
n_ang_cols = 8

N_norm = 50000.0

print("=" * 70)
print("GPU vs Numpy PWA Implementation Comparison")
print("=" * 70)
print()

print("[1] Generating test configuration and data...")
print(f"    Events: {n_events}")
print(f"    Waves: {n_waves}")
print(f"    Normalization: {N_norm}")

config = {
    'bw_index': np.random.randint(0, n_mass_cols, n_m0).astype(np.int32),
    'gamma_index': np.random.randint(0, n_mass_cols, n_g0).astype(np.int32),
    'bw_order': np.random.randint(0, n_m0, n_waves * n_res_per_wave).astype(np.int32),
    'bf_index': np.random.randint(0, n_q_cols, n_bf_types).astype(np.int32),
    'bf_order': np.random.randint(0, n_bf_types, n_waves * n_decays_per_wave).astype(np.int32),
    'ang_index': np.random.randint(0, n_ang_cols, (n_basis, n_ang_per_basis)).astype(np.int32),
    'ang_k': np.random.randn(n_basis, n_ang_per_basis),
    'ang_b': np.random.randn(n_basis, n_ang_per_basis),
    'matrix_gamma': np.random.randn(n_m0, n_g0),
    'matrix_ang': (np.random.randn(n_waves, n_basis) +
                   1j * np.random.randn(n_waves, n_basis)).astype(np.complex128),
    'gamma_table': (np.random.randn(n_g0, 100) +
                    1j * np.random.randn(n_g0, 100)).astype(np.complex128),
    'bf_table': np.exp(np.linspace(0, 1, 100).reshape(1, -1)).astype(np.float64),
    'g_min': 0.0,
    'g_delta': 0.01,
    'q_min': 0.0,
    'q_delta': 0.01,
}

# Generate data
# mass: (events, topo=3, res=2) -> mass_flat: (events, 6)
mass = np.random.rand(n_events, 3, 2) * 0.5 + 0.5
# q: (events, topo=2, decays=3) -> q_flat: (events, 6)
q = np.random.rand(n_events, 2, 3) * 0.5 + 0.25
# angles: (events, topo=2, ang=4) -> angles_flat: (events, 8)
angles = np.random.rand(n_events, 2, 4) * np.pi
time_arr = np.random.rand(n_events) * 10
frac_arr = np.random.rand(n_events) * 0.4 - 0.2
weights = np.ones(n_events)
bkg = np.random.rand(n_events) * 0.1

data_tuple = (mass, q, angles, time_arr, frac_arr, weights, bkg)

# Generate parameters
ck = np.random.randn(n_waves) + 1j * np.random.randn(n_waves)
m0 = np.random.rand(n_m0) + 1.5
g0 = np.random.rand(n_g0) * 0.1 + 0.05

params = (ck, m0, g0, 0.5, 0.02, 0.65, 0.01, 0.7, 0.1)

# ================================================================
# Initialize fitters
# ================================================================
print()
print("[2] Initializing fitters...")

if numpy_available:
    print("    Creating PWAFitter (numpy)...")
    fitter_np = PWAFitter(config)

if gpu_available:
    print("    Creating PWAGPU...")
    fitter_gpu = PWAGPU(config)

# ================================================================
# Data loading
# ================================================================
if gpu_available:
    print()
    print("[3] Loading data...")
    from pwa_gpu import PWAData
    data_gpu = PWAData(fitter_gpu, *data_tuple)
    print(f"Loaded {n_events} events to GPU")

# ================================================================
# Warmup
# ================================================================
if gpu_available and numpy_available:
    print()
    print("[4] Warmup run...")
    q_gpu, _ = fitter_gpu.compute(params, data_gpu, N_norm)
    q_np, _ = fitter_np.compute(params, data_tuple, N_norm)
    print("    Warmup completed successfully")

# ================================================================
# Benchmark
# ================================================================
if gpu_available and numpy_available:
    print()
    print("[5] Performance comparison (averaged over 5 runs)...")

    n_runs = 5

    # GPU timing
    gpu_times = []
    for _ in range(n_runs):
        t0 = time.time()
        q_gpu, _ = fitter_gpu.compute(params, data_gpu, N_norm)
        gpu_times.append((time.time() - t0) * 1000)

    # Numpy timing
    np_times = []
    for _ in range(n_runs):
        t0 = time.time()
        q_np, _ = fitter_np.compute(params, data_tuple, N_norm)
        np_times.append((time.time() - t0) * 1000)

    gpu_avg = np.mean(gpu_times)
    np_avg = np.mean(np_times)
    speedup = np_avg / gpu_avg if gpu_avg > 0 else float('inf')

    q_diff = np.max(np.abs(q_gpu - q_np))
    q_diff_avg = np.mean(np.abs(q_gpu - q_np))

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print("Timing:")
    print(f"  GPU time (avg):      {gpu_avg:.2f} ms")
    print(f"  Numpy time (avg):     {np_avg:.2f} ms")
    print(f"  Speedup:              {speedup:.1f}x")
    print()
    print("Accuracy (q_val):")
    print(f"  Max difference:   {q_diff:.6e}")
    print(f"  Avg difference:   {q_diff_avg:.6e}")

# ================================================================
# Detailed comparison
# ================================================================
if gpu_available and numpy_available:
    print()
    print("[6] Detailed comparison of p values...")
    print("    Note: Direct p-value comparison requires extracting intermediate")
    print("          results from numpy implementation. Comparing q_val instead.")
    print(f"    q_gpu:   {q_gpu:.10f}")
    print(f"    q_numpy: {q_np:.10f}")
    print(f"    diff:    {q_gpu - q_np:.6e}")
    rel_diff = abs(q_gpu - q_np) / (abs(q_np) + 1e-10)
    print(f"    relative: {rel_diff:.6e}")

# ================================================================
# Gradient comparison (N mode)
# ================================================================
if gpu_available and numpy_available:
    print()
    print("[7] Gradient comparison...")

    # Recompute to get grads
    q_gpu, grads_gpu = fitter_gpu.compute(params, data_gpu, N_norm)
    q_np, grads_np = fitter_np.compute(params, data_tuple, N_norm)

    param_names = ['ck', 'm0', 'g0', 'N', 'delta_m', 'delta_g', 'g', 'ap', 'lam', 'phi']

    print()
    print("  Parameter gradients (GPU vs Numpy):")
    for name in param_names:
        ga = grads_np[name]
        gg = grads_gpu[name]
        if ga is None or gg is None:
            print(f"  {name:12s}: skipped (None)")
            continue
        diff = np.abs(np.atleast_1d(gg).flatten() - np.atleast_1d(ga).flatten())
        max_abs = np.max(diff)
        max_rel = np.max(diff / (np.abs(np.atleast_1d(ga).flatten()) + 1e-10))
        print(f"  {name:12s}: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}")

# ================================================================
# Gradient comparison (N=None / chi-square mode)
# ================================================================
if gpu_available and numpy_available:
    print()
    print("[8] Gradient comparison (N=None / chi-square mode)...")

    q_gpu_c, grads_gpu_c = fitter_gpu.compute(params, data_gpu, None)
    q_np_c, grads_np_c = fitter_np.compute(params, data_tuple, None)

    print(f"  q_val GPU: {q_gpu_c:.10f}, q_val NP: {q_np_c:.10f}")
    for name in param_names:
        ga = grads_np_c[name]
        gg = grads_gpu_c[name]
        if ga is None or gg is None:
            print(f"  {name:12s}: skipped (None)")
            continue
        diff = np.abs(np.atleast_1d(gg).flatten() - np.atleast_1d(ga).flatten())
        max_abs = np.max(diff)
        max_rel = np.max(diff / (np.abs(np.atleast_1d(ga).flatten()) + 1e-10))
        print(f"  {name:12s}: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}")

# ================================================================
# Summary
# ================================================================
if gpu_available and numpy_available:
    print()
    print("=" * 70)
    max_rel_all = 0
    for name in param_names:
        ga = grads_np[name]
        gg = grads_gpu[name]
        if ga is not None and gg is not None:
            diff = np.abs(np.atleast_1d(gg).flatten() - np.atleast_1d(ga).flatten())
            rel = np.max(diff / (np.abs(np.atleast_1d(ga).flatten()) + 1e-10))
            max_rel_all = max(max_rel_all, rel)
    if max_rel_all < 1e-2:
        verdict = "EXCELLENT - Results match within numerical precision"
    elif max_rel_all < 0.1:
        verdict = "GOOD - Small numerical differences"
    else:
        verdict = "WARNING - Significant numerical differences detected"
    print(f"VERDICT: {verdict}")
    print("=" * 70)

if not gpu_available:
    print("\nGPU test skipped (not available)")
if not numpy_available:
    print("\nNumpy test skipped (not available)")
