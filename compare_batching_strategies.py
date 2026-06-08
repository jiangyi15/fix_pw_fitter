"""Clear comparison of batching strategies"""
import numpy as np
import time
from config_loader import Config
from numpy_kernel import NumpyKernel
from numpy_kernel_selective_cache import NumpyKernelSelectiveCache
from numpy_kernel_batched import NumpyKernelBatched
from numpy_kernel_fully_merged import NumpyKernelFullyMerged
from numpy_kernel_fully_merged_batched import NumpyKernelFullyMergedBatched


def compare():
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    
    kernels = {
        "Original": NumpyKernel(kernel_config),
        "Selective (no batch)": NumpyKernelSelectiveCache(kernel_config),
        "Selective + Batch": NumpyKernelBatched(kernel_config, optimal_batch_size=100),
        "Fully Merged (no batch)": NumpyKernelFullyMerged(kernel_config),
        "Fully Merged + Batch": NumpyKernelFullyMergedBatched(kernel_config, optimal_batch_size=100),
    }
    
    ck_map = config.get_ck_map()
    
    print("="*80)
    print("BATCHING STRATEGY COMPARISON")
    print("="*80)
    print("\nQuestion: Does Fully Merged + Batch outperform Selective + Batch?")
    print("="*80)
    
    for n_events in [100, 500, 1000]:
        print(f"\n{n_events} Events:")
        print("-"*80)
        
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
        
        results = {}
        for name, kernel in kernels.items():
            # Warm up
            kernel._compute(params, data, norm=None)
            
            # Benchmark
            start = time.time()
            for _ in range(10):
                kernel._compute(params, data, norm=None)
            elapsed = time.time() - start
            results[name] = elapsed
        
        baseline = results["Original"]
        
        # Print results
        print(f"{'Strategy':<30} {'Time (ms)':<12} {'Speedup':<10} {'vs Selective+Batch':<20}")
        print("-"*80)
        
        selective_batch_time = results["Selective + Batch"]
        
        for name, elapsed in sorted(results.items(), key=lambda x: x[1]):
            speedup = baseline / elapsed
            vs_selective = ""
            
            if "Batch" in name and name != "Selective + Batch":
                diff = (selective_batch_time - elapsed) / selective_batch_time * 100
                vs_selective = f"{'✓' if diff > 0 else '✗'} {diff:+.1f}%"
            
            print(f"{name:<30} {elapsed*1000:>10.1f}  {speedup:>8.2f}x  {vs_selective}")
        
        # Determine winner
        best_batch = min(
            (name for name in results if "Batch" in name),
            key=lambda x: results[x]
        )
        print(f"\n  Best batched strategy: {best_batch}")


if __name__ == "__main__":
    compare()
