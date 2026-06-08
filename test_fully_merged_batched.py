"""Test if fully merged + batching performs better"""
import numpy as np
import time
from config_loader import Config
from numpy_kernel import NumpyKernel
from numpy_kernel_selective_cache import NumpyKernelSelectiveCache
from numpy_kernel_batched import NumpyKernelBatched
from numpy_kernel_fully_merged import NumpyKernelFullyMerged
from numpy_kernel_fully_merged_batched import NumpyKernelFullyMergedBatched


def benchmark():
    """Compare all combinations"""
    print("="*70)
    print("FULLY MERGED + BATCHING PERFORMANCE TEST")
    print("="*70)
    
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    
    kernels = {
        "Original": NumpyKernel(kernel_config),
        "Selective Cache": NumpyKernelSelectiveCache(kernel_config),
        "Selective+Batch": NumpyKernelBatched(kernel_config, optimal_batch_size=100),
        "Fully Merged": NumpyKernelFullyMerged(kernel_config),
        "Fully+Batch": NumpyKernelFullyMergedBatched(kernel_config, optimal_batch_size=100),
    }
    
    ck_map = config.get_ck_map()
    
    event_sizes = [100, 500, 1000]
    n_iterations = 10
    
    for n_events in event_sizes:
        print(f"\n{'='*70}")
        print(f"{n_events} events, {n_iterations} iterations")
        print(f"{'='*70}")
        
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
        
        # Test correctness first
        Q_orig, _, _ = kernels["Original"]._compute(params, data, norm=None)
        
        results = {}
        for name, kernel in kernels.items():
            # Verify correctness
            Q_test, grads_test, _ = kernel._compute(params, data, norm=None)
            diff = abs(Q_orig - Q_test)
            
            # Benchmark
            start = time.time()
            for _ in range(n_iterations):
                kernel._compute(params, data, norm=None)
            elapsed = time.time() - start
            
            results[name] = elapsed
            
            correct = "✓" if diff < 1e-10 else "✗"
            print(f"{name:<20} {elapsed*1000:>7.1f}ms  diff={diff:.2e} {correct}")
        
        # Show speedups
        baseline = results["Original"]
        print(f"\nSpeedups:")
        for name in ["Selective Cache", "Selective+Batch", "Fully Merged", "Fully+Batch"]:
            speedup = baseline / results[name]
            print(f"  {name:<20} {speedup:.2f}x")
        
        # Compare batched versions
        selective_batch = baseline / results["Selective+Batch"]
        fully_batch = baseline / results["Fully+Batch"]
        
        if fully_batch > selective_batch:
            print(f"\n  → Fully+Batch WINS by {(fully_batch/selective_batch-1)*100:+.1f}%")
        else:
            print(f"\n  → Selective+Batch WINS by {(selective_batch/fully_batch-1)*100:+.1f}%")


if __name__ == "__main__":
    benchmark()
