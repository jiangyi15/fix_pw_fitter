"""Test truly optimized kernel"""
import numpy as np
import time
from config_loader import Config
from numpy_kernel import NumpyKernel
from numpy_kernel_truly_optimized import NumpyKernelSelectiveCache


def test_correctness():
    """Verify truly optimized kernel produces correct results"""
    print("="*60)
    print("Testing correctness of truly optimized kernel...")
    print("="*60)
    
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    
    kernel_orig = NumpyKernel(kernel_config)
    kernel_opt = NumpyKernelSelectiveCache(kernel_config)
    
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
    Q_opt, grads_opt, P_opt = kernel_opt._compute(params, data, norm=norm)
    
    print(f"Loss (original): {Q_orig:.10f}")
    print(f"Loss (optimized): {Q_opt:.10f}")
    print(f"Loss difference: {abs(Q_orig - Q_opt):.2e}")
    
    atol = 1e-10
    print(f"\nGradient checks:")
    
    ck_diff = np.max(np.abs(grads_orig['ck'] - grads_opt['ck']))
    print(f"  ck: {ck_diff:.2e} - {'✓' if ck_diff < atol else '✗'}")
    
    m0_diff = np.max(np.abs(grads_orig['m0'] - grads_opt['m0']))
    print(f"  m0: {m0_diff:.2e} - {'✓' if m0_diff < atol else '✗'}")
    
    g0_diff = np.max(np.abs(grads_orig['g0'] - grads_opt['g0']))
    print(f"  g0: {g0_diff:.2e} - {'✓' if g0_diff < atol else '✗'}")
    
    for i, name in enumerate(['Gamma', 'Delta_Gamma', 'Delta_m', 'A_p', 'poq_rho', 'pop_phi']):
        diff = abs(grads_orig['scalar'][i] - grads_opt['scalar'][i])
        print(f"  {name}: {diff:.2e} - {'✓' if diff < atol else '✗'}")
    
    success = (abs(Q_orig - Q_opt) < atol and ck_diff < atol and m0_diff < atol)
    print("\n" + "="*60)
    print(f"{'✓ PASSED' if success else '✗ FAILED'}")
    print("="*60)
    
    return success


def benchmark():
    """Benchmark all three implementations"""
    print("\n" + "="*60)
    print("Benchmarking all implementations...")
    print("="*60)
    
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    
    kernel_orig = NumpyKernel(kernel_config)
    kernel_bad = __import__('numpy_kernel_optimized').NumpyKernelOptimized(kernel_config)
    kernel_good = NumpyKernelSelectiveCache(kernel_config)
    
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
        
        # Bad cache (over-caching)
        start = time.time()
        for _ in range(n_iterations):
            kernel_bad._compute(params, data, norm=norm)
        time_bad = time.time() - start
        
        # Good cache (selective)
        start = time.time()
        for _ in range(n_iterations):
            kernel_good._compute(params, data, norm=norm)
        time_good = time.time() - start
        
        print(f"Original:      {time_orig*1000:7.1f} ms")
        print(f"Bad cache:     {time_bad*1000:7.1f} ms  ({time_bad/time_orig:.2f}x)")
        print(f"Good cache:    {time_good*1000:7.1f} ms  ({time_good/time_orig:.2f}x)")
        
        results.append({
            'events': n_events,
            'orig': time_orig,
            'bad': time_bad,
            'good': time_good,
        })
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Events':<10} {'Original':<12} {'Bad Cache':<12} {'Good Cache':<12}")
    print("-"*60)
    for r in results:
        print(f"{r['events']:<10} {r['orig']*1000:>10.1f}ms {r['bad']*1000:>10.1f}ms {r['good']*1000:>10.1f}ms")
    print("="*60)


if __name__ == "__main__":
    correct = test_correctness()
    if correct:
        benchmark()
