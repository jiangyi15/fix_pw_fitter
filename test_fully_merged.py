"""Test fully merged amplitude calculations"""
import numpy as np
import time
from config_loader import Config
from numpy_kernel import NumpyKernel
from numpy_kernel_merged import NumpyKernelMergedGradients
from numpy_kernel_fully_merged import NumpyKernelFullyMerged


def test_correctness():
    """Verify fully merged kernel produces correct results"""
    print("="*60)
    print("Testing fully merged kernel...")
    print("="*60)
    
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    
    kernel_orig = NumpyKernel(kernel_config)
    kernel_fully_merged = NumpyKernelFullyMerged(kernel_config)
    
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
    Q_fully, grads_fully, P_fully = kernel_fully_merged._compute(params, data, norm=norm)
    
    print(f"Loss (original):      {Q_orig:.10f}")
    print(f"Loss (fully merged): {Q_fully:.10f}")
    print(f"Loss difference:      {abs(Q_orig - Q_fully):.2e}")
    
    atol = 1e-10
    print(f"\nGradient checks:")
    
    ck_diff = np.max(np.abs(grads_orig['ck'] - grads_fully['ck']))
    print(f"  ck: {ck_diff:.2e} - {'✓' if ck_diff < atol else '✗'}")
    
    m0_diff = np.max(np.abs(grads_orig['m0'] - grads_fully['m0']))
    print(f"  m0: {m0_diff:.2e} - {'✓' if m0_diff < atol else '✗'}")
    
    g0_diff = np.max(np.abs(grads_orig['g0'] - grads_fully['g0']))
    print(f"  g0: {g0_diff:.2e} - {'✓' if g0_diff < atol else '✗'}")
    
    for i, name in enumerate(['Gamma', 'Delta_Gamma', 'Delta_m', 'A_p', 'poq_rho', 'pop_phi']):
        orig_val = grads_orig['scalar'][i]
        fully_val = grads_fully['scalar'][i]
        diff = abs(orig_val - fully_val)
        print(f"  {name:15s}: {diff:.2e} - {'✓' if diff < atol else '✗'}")
    
    success = (abs(Q_orig - Q_fully) < atol and ck_diff < atol and m0_diff < atol)
    print("\n" + "="*60)
    print(f"{'✓ PASSED' if success else '✗ FAILED'}")
    print("="*60)
    
    return success


def benchmark():
    """Benchmark fully merged implementation"""
    print("\n" + "="*60)
    print("Benchmarking fully merged kernel...")
    print("="*60)
    
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    
    kernel_orig = NumpyKernel(kernel_config)
    kernel_merged = NumpyKernelMergedGradients(kernel_config)
    kernel_fully = NumpyKernelFullyMerged(kernel_config)
    
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
        
        # Fully merged
        start = time.time()
        for _ in range(n_iterations):
            kernel_fully._compute(params, data, norm=norm)
        time_fully = time.time() - start
        
        print(f"Original:       {time_orig*1000:7.1f} ms")
        print(f"Merged grads:   {time_merged*1000:7.1f} ms  ({time_merged/time_orig:.2f}x)")
        print(f"Fully merged:   {time_fully*1000:7.1f} ms  ({time_fully/time_orig:.2f}x)")
        
        results.append({
            'events': n_events,
            'orig': time_orig,
            'merged': time_merged,
            'fully': time_fully,
        })
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"\n{'Events':<10} {'Original':<12} {'Merged':<12} {'Fully Merged':<15} {'Best Speedup':<12}")
    print("-"*60)
    for r in results:
        best_speedup = r['orig'] / min(r['merged'], r['fully'])
        winner = 'Merged' if r['merged'] < r['fully'] else 'Fully'
        print(f"{r['events']:<10} {r['orig']*1000:>10.1f}ms {r['merged']*1000:>10.1f}ms "
              f"{r['fully']*1000:>13.1f}ms {best_speedup:>10.2f}x ({winner})")
    print("="*60)
    
    avg_speedup_merged = np.mean([r['orig']/r['merged'] for r in results])
    avg_speedup_fully = np.mean([r['orig']/r['fully'] for r in results])
    
    print(f"\nAverage speedup:")
    print(f"  Merged gradients: {avg_speedup_merged:.2f}x")
    print(f"  Fully merged:     {avg_speedup_fully:.2f}x")
    print(f"  Winner:           {'Fully merged' if avg_speedup_fully > avg_speedup_merged else 'Merged gradients'}")


if __name__ == "__main__":
    correct = test_correctness()
    if correct:
        benchmark()
