"""Test merged index operations"""
import numpy as np
import time
from config_loader import Config
from numpy_kernel import NumpyKernel
from numpy_kernel_fully_merged import NumpyKernelFullyMerged
from numpy_kernel_merged_indices import NumpyKernelMergedIndices


def test_correctness():
    """Verify merged indices kernel produces correct results"""
    print("="*60)
    print("Testing merged indices kernel...")
    print("="*60)
    
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    
    kernel_orig = NumpyKernel(kernel_config)
    kernel_merged_idx = NumpyKernelMergedIndices(kernel_config)
    
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
    
    print("\n--- Testing with norm ---")
    Q_orig, grads_orig, P_orig = kernel_orig._compute(params, data, norm=norm)
    Q_merged, grads_merged, P_merged = kernel_merged_idx._compute(params, data, norm=norm)
    
    print(f"Loss (original):       {Q_orig:.10f}")
    print(f"Loss (merged indices): {Q_merged:.10f}")
    print(f"Loss difference:       {abs(Q_orig - Q_merged):.2e}")
    
    atol = 1e-8  # Slightly relaxed due to different operation order
    print(f"\nGradient checks:")
    
    ck_diff = np.max(np.abs(grads_orig['ck'] - grads_merged['ck']))
    print(f"  ck: {ck_diff:.2e} - {'✓' if ck_diff < atol else '✗'}")
    
    m0_diff = np.max(np.abs(grads_orig['m0'] - grads_merged['m0']))
    print(f"  m0: {m0_diff:.2e} - {'✓' if m0_diff < atol else '✗'}")
    
    g0_diff = np.max(np.abs(grads_orig['g0'] - grads_merged['g0']))
    print(f"  g0: {g0_diff:.2e} - {'✓' if g0_diff < atol else '✗'}")
    
    for i, name in enumerate(['Gamma', 'Delta_Gamma', 'Delta_m', 'A_p', 'poq_rho', 'pop_phi']):
        orig_val = grads_orig['scalar'][i]
        merged_val = grads_merged['scalar'][i]
        diff = abs(orig_val - merged_val)
        print(f"  {name:15s}: {diff:.2e} - {'✓' if diff < atol else '✗'}")
    
    success = (abs(Q_orig - Q_merged) < atol)
    print("\n" + "="*60)
    print(f"{'✓ PASSED' if success else '✗ FAILED'}")
    print("="*60)
    
    return success


def benchmark():
    """Benchmark merged indices implementation"""
    print("\n" + "="*60)
    print("Benchmarking merged indices kernel...")
    print("="*60)
    
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    
    kernel_orig = NumpyKernel(kernel_config)
    kernel_fully = NumpyKernelFullyMerged(kernel_config)
    kernel_merged_idx = NumpyKernelMergedIndices(kernel_config)
    
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
        
        # Fully merged (previous best)
        start = time.time()
        for _ in range(n_iterations):
            kernel_fully._compute(params, data, norm=norm)
        time_fully = time.time() - start
        
        # Merged indices (NEW)
        start = time.time()
        for _ in range(n_iterations):
            kernel_merged_idx._compute(params, data, norm=norm)
        time_merged_idx = time.time() - start
        
        print(f"Original:         {time_orig*1000:7.1f} ms")
        print(f"Fully merged:     {time_fully*1000:7.1f} ms  ({time_fully/time_orig:.2f}x)")
        print(f"Merged indices:   {time_merged_idx*1000:7.1f} ms  ({time_merged_idx/time_orig:.2f}x)")
        
        if time_merged_idx < time_fully:
            improvement = (time_fully - time_merged_idx) / time_fully * 100
            print(f"  → Merged indices FASTER by {improvement:.1f}% ✓")
        else:
            slowdown = (time_merged_idx - time_fully) / time_fully * 100
            print(f"  → Merged indices SLOWER by {slowdown:.1f}% ✗")
        
        results.append({
            'events': n_events,
            'orig': time_orig,
            'fully': time_fully,
            'merged_idx': time_merged_idx,
        })
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"\n{'Events':<10} {'Original':<12} {'Fully Merged':<15} {'Merged Indices':<17} {'Winner':<10}")
    print("-"*70)
    for r in results:
        winner = 'Merged Idx' if r['merged_idx'] < r['fully'] else 'Fully'
        print(f"{r['events']:<10} {r['orig']*1000:>10.1f}ms {r['fully']*1000:>13.1f}ms "
              f"{r['merged_idx']*1000:>15.1f}ms   {winner}")
    print("="*60)
    
    avg_speedup_fully = np.mean([r['orig']/r['fully'] for r in results])
    avg_speedup_merged = np.mean([r['orig']/r['merged_idx'] for r in results])
    
    print(f"\nAverage speedup:")
    print(f"  Fully merged:    {avg_speedup_fully:.2f}x")
    print(f"  Merged indices:  {avg_speedup_merged:.2f}x")
    print(f"  Winner:          {'Merged Indices' if avg_speedup_merged > avg_speedup_fully else 'Fully Merged'}")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    print("Optimization applied:")
    print("  Before: m0[m0_index] → compute → [bw_order]")
    print("  After:  m0[m0_index[bw_order]] directly")
    print("\nExpected benefits:")
    print("  - 1 fewer np.take operation per parameter")
    print("  - Reduced intermediate array allocation")
    print("  - Better cache locality")
    print("\nActual impact:")
    if avg_speedup_merged > avg_speedup_fully:
        print(f"  ✓ Positive - {((avg_speedup_merged/avg_speedup_fully - 1) * 100):.1f}% improvement")
    else:
        print(f"  ✗ Negative - {((1 - avg_speedup_merged/avg_speedup_fully) * 100):.1f}% slowdown")
        print("  Possible reasons:")
        print("    - np.take already highly optimized")
        print("    - Memory-bound, operation reduction doesn't help")
        print("    - Additional complexity offsets gains")


if __name__ == "__main__":
    correct = test_correctness()
    if correct:
        benchmark()
