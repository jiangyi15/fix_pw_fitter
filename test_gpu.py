#!/usr/bin/env python
"""
Test script to compare GPU and numpy implementations of PWA computation.

Usage: python test_gpu.py
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


def generate_test_config():
    """Generate a realistic configuration for testing"""
    np.random.seed(42)
    
    # Realistic dimensions
    n_events = args.events
    n_waves = args.waves  # Must be even
    n_m0 = 5     # Number of resonance masses
    n_g0 = 3     # Number of gamma parameters
    n_res_per_wave = 2      # Resonances per wave
    n_decays_per_wave = 3   # Decay channels per wave
    n_bf_types = 6          # Barrier factor types
    n_basis = 10            # Angular basis functions
    n_ang_per_basis = 3     # Angles per basis
    
    # Data dimensions
    n_mass_cols = 4 * 3   # mass_flat columns
    n_q_cols = 4 * 4      # q_flat columns
    n_ang_cols = 3 * 5    # angles_flat columns
    
    # Interpolation table sizes
    n_gamma_points = 200
    n_bf_points = 150
    
    config = {
        # Index arrays
        'bw_index': np.random.randint(0, n_mass_cols, n_m0),
        'gamma_index': np.random.randint(0, n_mass_cols, n_g0),
        'bw_order': np.random.randint(0, n_m0, n_waves * n_res_per_wave),
        'bf_index': np.random.randint(0, n_q_cols, n_bf_types),
        'bf_order': np.random.randint(0, n_bf_types, n_waves * n_decays_per_wave),
        'ang_index': np.random.randint(0, n_ang_cols, (n_basis, n_ang_per_basis)),
        
        # Angular coefficients
        'ang_k': np.random.randn(n_basis, n_ang_per_basis).astype(np.float64),
        'ang_b': np.random.randn(n_basis, n_ang_per_basis).astype(np.float64),
        
        # Coupling matrices
        'matrix_gamma': np.random.randn(n_m0, n_g0).astype(np.float64),
        'matrix_ang': (np.random.randn(n_waves, n_basis) + 
                       1j * np.random.randn(n_waves, n_basis)).astype(np.complex128),
        
        # Interpolation tables (REAL - gamma is a real width parameter)
        'gamma_table': np.random.randn(n_g0, n_gamma_points).astype(np.float64),
        'bf_table': np.exp(np.linspace(0, 1, n_bf_points).reshape(1, -1)).astype(np.float64),
        
        # Interpolation parameters
        'g_min': 0.0,
        'g_delta': 0.005,
        'q_min': 0.0,
        'q_delta': 0.005,
    }
    
    return config, n_events, n_waves


def generate_test_data(n_events):
    """Generate test data similar to fitter.py format"""
    np.random.seed(123)
    
    # Mass: (events, topology, resonance) 
    mass = np.random.rand(n_events, 4, 3) * 0.4 + 0.6  # range [0.6, 1.0] GeV
    
    # Momentum transfer q: (events, topology, decays)
    q = np.random.rand(n_events, 4, 4) * 0.3 + 0.1    # range [0.1, 0.4] GeV
    
    # Angles: (events, topology, angles)
    angles = np.random.rand(n_events, 3, 5) * np.pi   # range [0, pi]
    
    # Time: decay time
    time_arr = np.random.exponential(1.5, n_events)   # decay time distribution
    
    # Mixing fraction
    frac = np.random.uniform(-0.2, 0.2, n_events)
    
    # Weights (usually all 1.0)
    weights = np.ones(n_events, dtype=np.float64)
    
    # Background fraction
    bkg = np.random.uniform(0.01, 0.05, n_events)
    
    return (mass, q, angles, time_arr, frac, weights, bkg)


def generate_test_params(config, n_waves):
    """Generate test parameters"""
    np.random.seed(456)
    
    n_waves = n_waves  # Use provided value
    n_m0 = len(config['bw_index'])
    n_g0 = len(config['gamma_index'])
    
    # Complex coupling coefficients
    ck = (np.random.randn(n_waves) + 1j * np.random.randn(n_waves)).astype(np.complex128)
    
    # Resonance masses (GeV)
    m0 = (np.random.rand(n_m0) * 0.3 + 1.2).astype(np.float64)  # range [1.2, 1.5] GeV
    
    # Width parameters
    g0 = (np.random.rand(n_g0) * 0.1 + 0.05).astype(np.float64)  # range [0.05, 0.15] GeV
    
    # Time-dependent mixing parameters
    delta_m = 0.5065   # Mass difference (ps^-1)
    delta_g = 0.001    # Width difference (ps^-1)
    g = 0.657          # Average width (ps^-1)
    
    # CP violation parameters
    ap = 0.01          # |A_f/A_fbar| - 1
    lam = 0.75         # |lambda|
    phi = 0.15         # CPV phase (rad)
    
    return (ck, m0, g0, delta_m, delta_g, g, ap, lam, phi)


def main():
    print("=" * 70)
    print("GPU vs Numpy PWA Implementation Comparison")
    print("=" * 70)
    
    # Check availability
    if not numpy_available:
        print("ERROR: Numpy implementation not available")
        sys.exit(1)
    
    if not gpu_available:
        print("ERROR: GPU implementation not available")
        sys.exit(1)
    
    # Generate configuration and data
    print("\n[1] Generating test configuration and data...")
    config, n_events, n_waves = generate_test_config()
    data = generate_test_data(n_events)
    params = generate_test_params(config, n_waves)
    N = 50000.0  # Normalization factor
    
    print(f"    Events: {n_events}")
    print(f"    Waves: {len(params[0])}")
    print(f"    Normalization: {N}")
    
    # Create fitters
    print("\n[2] Initializing fitters...")
    
    # Numpy fitter
    print("    Creating PWAFitter (numpy)...")
    fitter_numpy = PWAFitter(config)
    
    # GPU fitter
    print("    Creating PWAGPU...")
    try:
        fitter_gpu = PWAGPU(config)
    except Exception as e:
        print(f"    ERROR: Failed to create GPU fitter: {e}")
        sys.exit(1)
    
    # Load data to GPU
    print("\n[3] Loading data...")
    try:
        data_gpu = fitter_gpu.load_data(data)
        print(f"Loaded {data_gpu.n_events} events to GPU")
    except Exception as e:
        print(f"    ERROR: Failed to load data to GPU: {e}")
        sys.exit(1)
    
    # Warmup run
    print("\n[4] Warmup run...")
    try:
        q_gpu, _ = fitter_gpu.compute(params, data_gpu, N)
        q_numpy, grad_numpy = fitter_numpy.compute(params, data, N)
        print("    Warmup completed successfully")
    except Exception as e:
        print(f"    ERROR during warmup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Timed runs
    print("\n[5] Performance comparison (averaged over 5 runs)...")
    
    n_runs = 5
    gpu_times = []
    numpy_times = []
    p_diffs = []
    q_diffs = []
    
    for i in range(n_runs):
        # GPU computation (forward + gradient combined)
        t0 = time.perf_counter()
        q_gpu, _ = fitter_gpu.compute(params, data_gpu, N)
        t_gpu = time.perf_counter() - t0
        gpu_times.append(t_gpu)
        
        # Numpy computation (forward + gradient)
        t0 = time.perf_counter()
        q_numpy, grad_numpy = fitter_numpy.compute(params, data, N)
        t_numpy = time.perf_counter() - t0
        numpy_times.append(t_numpy)
        
        # Compare results
        # Need to get p from numpy computation
        # Looking at ref_numpy, it computes p internally but doesn't return it directly
        # We'll need to modify or extract it
        
        # For now, compare q_val
        q_diffs.append(abs(q_gpu - q_numpy))
    
    # Compute statistics
    avg_gpu_time = np.mean(gpu_times)
    avg_numpy_time = np.mean(numpy_times)
    speedup = avg_numpy_time / avg_gpu_time if avg_gpu_time > 0 else 0
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\nTiming:")
    print(f"  GPU time (avg):   {avg_gpu_time*1000:8.2f} ms")
    print(f"  Numpy time (avg): {avg_numpy_time*1000:8.2f} ms")
    print(f"  Speedup:          {speedup:8.1f}x")
    
    print(f"\nAccuracy (q_val):")
    print(f"  Max difference:   {max(q_diffs):.6e}")
    print(f"  Avg difference:   {np.mean(q_diffs):.6e}")
    
    # Try to compare p values by re-running and capturing
    print("\n[6] Detailed comparison of p values...")
    try:
        q_gpu, _ = fitter_gpu.compute(params, data_gpu, N)
        
        # For numpy, we can access last_p after compute
        q_numpy, _ = fitter_numpy.compute(params, data, N)
        
        # For now, let's just report that comparison requires modification
        print("    Note: Direct p-value comparison requires extracting intermediate")
        print("          results from numpy implementation. Comparing q_val instead.")
        print(f"    q_gpu:   {q_gpu:.10f}")
        print(f"    q_numpy: {q_numpy:.10f}")
        print(f"    diff:    {abs(q_gpu - q_numpy):.6e}")
        
        # Relative difference
        rel_diff = abs(q_gpu - q_numpy) / (abs(q_numpy) + 1e-10)
        print(f"    relative: {rel_diff:.6e}")
        
    except Exception as e:
        print(f"    ERROR during detailed comparison: {e}")
    
    # Gradient comparison
    print("\n[7] Gradient comparison...")
    param_names = ['ck', 'm0', 'g0', 'N', 'delta_m', 'delta_g', 'g', 'ap', 'lam', 'phi']
    try:
        q_gpu, grad_gpu = fitter_gpu.compute(params, data_gpu, N)
        
        print(f"\n  Parameter gradients (GPU vs Numpy):")
        
        for i, name in enumerate(param_names):
            gg = grad_gpu[name]
            gn = grad_numpy[name]
            if gg is None or gn is None:
                print(f"  {name:12s}: skipped")
                continue
            
            gg_flat = np.atleast_1d(gg).flatten()
            gn_flat = np.atleast_1d(gn).flatten()
            
            abs_diff = np.max(np.abs(gg_flat - gn_flat))
            rel_diff = abs_diff / (np.max(np.abs(gn_flat)) + 1e-10)
            
            print(f"  {name:12s}: max_abs={abs_diff:.6e}, max_rel={rel_diff:.6e}")
            
    except Exception as e:
        print(f"    ERROR during gradient comparison: {e}")
        import traceback
        traceback.print_exc()
    
    # Gradient comparison with N=None (chi-square mode)
    print("\n[8] Gradient comparison (N=None / chi-square mode)...")
    try:
        q_gpu0, grad_gpu0 = fitter_gpu.compute(params, data_gpu, None)
        q_np0, grad_np0 = fitter_numpy.compute(params, data, None)
        
        print(f"  q_val GPU: {q_gpu0:.10f}, q_val NP: {q_np0:.10f}")
        for i, name in enumerate(param_names):
            gg = grad_gpu0[name]
            gn = grad_np0[name]
            if gg is None and gn is None:
                print(f"  {name:12s}: skipped (None)")
                continue
            gg_flat = np.atleast_1d(gg).flatten()
            gn_flat = np.atleast_1d(gn).flatten()
            abs_diff = np.max(np.abs(gg_flat - gn_flat))
            rel_diff = abs_diff / (np.max(np.abs(gn_flat)) + 1e-10)
            print(f"  {name:12s}: max_abs={abs_diff:.6e}, max_rel={rel_diff:.6e}")
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Final verdict
    print(f"\n{'='*70}")
    avg_q_diff = np.mean(q_diffs)
    if avg_q_diff < 1e-6:
        print("VERDICT: EXCELLENT - Results match within numerical precision")
    elif avg_q_diff < 1e-4:
        print("VERDICT: GOOD - Results match within acceptable tolerance")
    elif avg_q_diff < 1e-2:
        print("VERDICT: ACCEPTABLE - Small differences detected")
    else:
        print("VERDICT: POOR - Significant differences detected")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
