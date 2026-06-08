"""Test batched kernel for correctness and performance"""
import numpy as np
import time
from config_loader import Config
from numpy_kernel import NumpyKernel
from numpy_kernel_selective_cache import NumpyKernelSelectiveCache
from numpy_kernel_batched import NumpyKernelBatched


def test_correctness():
    """Verify batched kernel produces correct results"""
    print("="*70)
    print("Testing batched kernel correctness...")
    print("="*70)
    
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    
    kernel_orig = NumpyKernel(kernel_config)
    kernel_batched = NumpyKernelBatched(kernel_config, optimal_batch_size=100)
    
    ck_map = config.get_ck_map()
    
    # Test with different data sizes
    for n_events in [100, 500, 1000]:
        print(f"\n--- Testing {n_events} events ---")
        
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
        
        # Compare results
        Q_orig, grads_orig, P_orig = kernel_orig._compute(params, data, norm=norm)
        Q_batch, grads_batch, P_batch = kernel_batched._compute(params, data, norm=norm)
        
        # Check loss
        Q_diff = abs(Q_orig - Q_batch)
        print(f"  Loss diff:      {Q_diff:.2e} {'✓' if Q_diff < 1e-10 else '✗'}")
        
        # Check gradients
        atol = 1e-10
        ck_diff = np.max(np.abs(grads_orig['ck'] - grads_batch['ck']))
        m0_diff = np.max(np.abs(grads_orig['m0'] - grads_batch['m0']))
        g0_diff = np.max(np.abs(grads_orig['g0'] - grads_batch['g0']))
        
        print(f"  Gradient ck:    {ck_diff:.2e} {'✓' if ck_diff < atol else '✗'}")
        print(f"  Gradient m0:    {m0_diff:.2e} {'✓' if m0_diff < atol else '✗'}")
        print(f"  Gradient g0:    {g0_diff:.2e} {'✓' if g0_diff < atol else '✗'}")
        
        # Check P
        P_diff = np.max(np.abs(P_orig - P_batch))
        print(f"  Probabilities:  {P_diff:.2e} {'✓' if P_diff < atol else '✗'}")


def benchmark():
    """Benchmark batched kernel"""
    print("\n" + "="*70)
    print("Benchmarking batched kernel...")
    print("="*70)
    
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    
    kernel_orig = NumpyKernel(kernel_config)
    kernel_opt = NumpyKernelSelectiveCache(kernel_config)
    kernel_batched = NumpyKernelBatched(kernel_config, optimal_batch_size=100)
    
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
        
        # Selective cache
        start = time.time()
        for _ in range(n_iterations):
            kernel_opt._compute(params, data, norm=norm)
        time_opt = time.time() - start
        
        # Batched
        start = time.time()
        for _ in range(n_iterations):
            kernel_batched._compute(params, data, norm=norm)
        time_batched = time.time() - start
        
        print(f"Original:        {time_orig*1000:7.1f} ms  (1.00x)")
        print(f"Selective Cache: {time_opt*1000:7.1f} ms  ({time_orig/time_opt:.2f}x)")
        print(f"Batched:         {time_batched*1000:7.1f} ms  ({time_orig/time_batched:.2f}x)")
        
        results.append({
            'events': n_events,
            'orig': time_orig,
            'opt': time_opt,
            'batched': time_batched,
        })
    
    # Summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(f"\n{'Events':<10} {'Original':<15} {'Selective':<15} {'Batched':<15} {'Best':<15}")
    print("-"*70)
    
    for r in results:
        speedup_opt = r['orig'] / r['opt']
        speedup_batched = r['orig'] / r['batched']
        
        if speedup_batched > speedup_opt:
            best = f"Batched ({speedup_batched:.2f}x)"
        else:
            best = f"Selective ({speedup_opt:.2f}x)"
        
        print(f"{r['events']:<10} {r['orig']*1000:>13.1f}ms {r['opt']*1000:>13.1f}ms "
              f"{r['batched']*1000:>13.1f}ms {best}")
    
    print("="*70)
    
    # Analysis
    print("\nKEY FINDINGS:")
    print("="*70)
    
    # Calculate improvement
    for r in results:
        if r['events'] > 100:  # Only meaningful for larger datasets
            speedup_opt = r['orig'] / r['opt']
            speedup_batched = r['orig'] / r['batched']
            improvement = (speedup_batched - speedup_opt) / speedup_opt * 100
            print(f"{r['events']} events: Batched is {improvement:+.1f}% {'better' if improvement > 0 else 'worse'}")


if __name__ == "__main__":
    test_correctness()
    benchmark()
