"""Test merged gradient calculations"""
import numpy as np
import time
from config_loader import Config
from numpy_kernel import NumpyKernel
from numpy_kernel_merged import NumpyKernelMergedGradients


def test_correctness():
    """Verify merged gradients produce correct results"""
    print("="*60)
    print("Testing merged gradient kernel...")
    print("="*60)
    
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    
    kernel_orig = NumpyKernel(kernel_config)
    kernel_merged = NumpyKernelMergedGradients(kernel_config)
    
    ck_map = config.get_ck_map()
    n_events = 100
    np.random.seed(42)
    
    data = {
        "mass": np.random.random((n_events, 2*3*8)),
        "q": np.random.random((n_events, 3*3*8)),
        "angle": np.random.random((n_events, 3*8, 3)),
        "frac": np.random.random((n_events,)),
        "time": np.random.random((n_events,)),
        "bkg": np.random.random((n_events,)) * 0.01,
        "weight": np.ones((n_events,)),
    }
    
    params = {
        "ck": np.random.random(len(ck_map)) + 1j*np.random.random(len(ck_map)),
        "m0": np.random.random(len(config.m0_phys_name)) + 2,
        "g0": np.random.random(len(config.g0_phys_name)) + 0.1,
        "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }
    
    norm = 1.0
    
    # Test with norm
    print("\n--- Testing with norm ---")
    Q_orig, grads_orig, P_orig = kernel_orig._compute(params, data, norm=norm)
    Q_merged, grads_merged, P_merged = kernel_merged._compute(params, data, norm=norm)
    
    print(f"Loss (original): {Q_orig:.10f}")
    print(f"Loss (merged):   {Q_merged:.10f}")
    print(f"Loss difference: {abs(Q_orig - Q_merged):.2e}")
    
    atol = 1e-10
    print(f"\nGradient checks:")
    
    ck_diff = np.max(np.abs(grads_orig['ck'] - grads_merged['ck']))
    print(f"  ck: {ck_diff:.2e} - {'✓' if ck_diff < atol else '✗'}")
    
    m0_diff = np.max(np.abs(grads_orig['m0'] - grads_merged['m0']))
    print(f"  m0: {m0_diff:.2e} - {'✓' if m0_diff < atol else '✗'}")
    
    g0_diff = np.max(np.abs(grads_orig['g0'] - grads_merged['g0']))
    print(f"  g0: {g0_diff:.2e} - {'✓' if g0_diff < atol else '✗'}")
    
    print("\n  Time gradient checks:")
    for i, name in enumerate(['Gamma', 'Delta_Gamma', 'Delta_m', 'A_p', 'poq_rho', 'pop_phi']):
        orig_val = grads_orig['scalar'][i]
        merged_val = grads_merged['scalar'][i]
        
        # Handle complex numbers
        if np.iscomplexobj(orig_val):
            diff = abs(orig_val - merged_val)
        else:
            diff = abs(orig_val - merged_val)
        
        print(f"    {name:15s}: {diff:.2e} - {'✓' if diff < atol else '✗'}")
    
    success = (abs(Q_orig - Q_merged) < atol and ck_diff < atol and m0_diff < atol)
    print("\n" + "="*60)
    print(f"{'✓ PASSED' if success else '✗ FAILED'}")
    print("="*60)
    
    return success


def benchmark():
    """Benchmark merged gradient implementation"""
    print("\n" + "="*60)
    print("Benchmarking merged gradient kernel...")
    print("="*60)
    
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    
    kernel_orig = NumpyKernel(kernel_config)
    kernel_merged = NumpyKernelMergedGradients(kernel_config)
    
    ck_map = config.get_ck_map()
    
    event_sizes = [100, 500, 1000]
    n_iterations = 10
    
    results = []
    
    for n_events in event_sizes:
        print(f"\n--- {n_events} events, {n_iterations} iterations ---")
        
        np.random.seed(42)
        data = {
            "mass": np.random.random((n_events, 2*3*8)),
            "q": np.random.random((n_events, 3*3*8)),
            "angle": np.random.random((n_events, 3*8, 3)),
            "frac": np.random.random((n_events,)),
            "time": np.random.random((n_events,)),
            "bkg": np.random.random((n_events,)) * 0.01,
            "weight": np.ones((n_events,)),
        }
        
        params = {
            "ck": np.random.random(len(ck_map)) + 1j*np.random.random(len(ck_map)),
            "m0": np.random.random(len(config.m0_phys_name)) + 2,
            "g0": np.random.random(len(config.g0_phys_name)) + 0.1,
            "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
        }
        
        norm = 1.0
        
        # Original
        start = time.time()
        for _ in range(n_iterations):
            kernel_orig._compute(params, data, norm=norm)
        time_orig = time.time() - start
        
        # Merged gradients
        start = time.time()
        for _ in range(n_iterations):
            kernel_merged._compute(params, data, norm=norm)
        time_merged = time.time() - start
        
        print(f"Original:  {time_orig*1000:7.1f} ms")
        print(f"Merged:    {time_merged*1000:7.1f} ms  ({time_merged/time_orig:.2f}x)")
        
        results.append({
            'events': n_events,
            'orig': time_orig,
            'merged': time_merged,
            'speedup': time_orig / time_merged,
        })
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Events':<10} {'Original':<12} {'Merged':<12} {'Speedup':<10}")
    print("-"*60)
    for r in results:
        print(f"{r['events']:<10} {r['orig']*1000:>10.1f}ms {r['merged']*1000:>10.1f}ms {r['speedup']:>8.2f}x")
    print("="*60)
    
    avg_speedup = np.mean([r['speedup'] for r in results])
    print(f"\nAverage speedup: {avg_speedup:.2f}x")


if __name__ == "__main__":
    correct = test_correctness()
    if correct:
        benchmark()
