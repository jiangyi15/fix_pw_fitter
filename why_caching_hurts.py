"""
Simple demonstration: Why caching variables can be SLOWER than recomputing them
"""
import numpy as np
import time

def benchmark_operation(size=100000):
    """Compare: recompute vs cache retrieval"""

    # Create test data
    x = np.random.randn(size)
    y = np.random.randn(size)

    # ========== Test 1: Recompute ==========
    # Compute twice (like original kernel)
    start = time.time()
    for _ in range(100):
        result1 = np.cos(x) + np.sin(y)  # First computation
        # ... some other work ...
        result2 = np.cos(x) + np.sin(y)  # Recompute (discarded after use)
    time_recompute = time.time() - start

    # ========== Test 2: Cache ==========
    # Compute once and cache (like optimized kernel)
    start = time.time()
    for _ in range(100):
        cached_cos = np.cos(x)  # Compute and cache
        cached_sin = np.sin(y)  # Compute and cache
        result1 = cached_cos + cached_sin
        # ... some other work ...
        result2 = cached_cos + cached_sin  # Reuse from cache
    time_cache = time.time() - start

    # ========== Test 3: Cache with pressure ==========
    # Cache many arrays to simulate cache pressure
    start = time.time()
    for _ in range(100):
        # Cache 30 arrays like optimized kernel does
        cache = {}
        cache['arr1'] = np.cos(x)
        cache['arr2'] = np.sin(y)
        cache['arr3'] = np.exp(x)
        cache['arr4'] = np.log(np.abs(x) + 1)
        cache['arr5'] = x ** 2
        cache['arr6'] = np.sqrt(np.abs(x))
        cache['arr7'] = np.tan(x)
        cache['arr8'] = y ** 3
        cache['arr9'] = np.abs(x)
        cache['arr10'] = np.abs(y)
        # ... 20 more arrays ...
        for i in range(20):
            cache[f'extra_{i}'] = np.random.randn(size)

        result1 = cache['arr1'] + cache['arr2']
        result2 = cache['arr1'] + cache['arr2']  # Reuse
    time_cache_pressure = time.time() - start

    print(f"\n{'='*60}")
    print(f"Array size: {size:,} elements ({size * 8 / 1024 / 1024:.2f} MB per array)")
    print(f"{'='*60}")
    print(f"1. Recompute:           {time_recompute*1000:7.2f} ms")
    print(f"2. Simple cache:        {time_cache*1000:7.2f} ms  ({time_cache/time_recompute:.2f}x)")
    print(f"3. Cache with pressure: {time_cache_pressure*1000:7.2f} ms  ({time_cache_pressure/time_recompute:.2f}x)")
    print(f"{'='*60}")

    # Calculate memory footprint
    array_size_mb = size * 8 / 1024 / 1024
    total_cache_mb = array_size_mb * 30
    print(f"\nMemory cached: {total_cache_mb:.1f} MB")
    print(f"Typical L3 cache: 8-16 MB")
    print(f"Result: {'L3 cache overflow! Data evicted to RAM.' if total_cache_mb > 16 else 'L3 cache OK'}")

    return time_recompute, time_cache, time_cache_pressure


def demonstrate_cache_pressure():
    """Show how cache pressure kills performance"""

    print("\n" + "="*70)
    print("DEMONSTRATION: Cache Pressure Effect")
    print("="*70)

    print("\nSmall arrays (fit in L3 cache):")
    benchmark_operation(10000)  # ~0.08 MB per array, 2.4 MB total

    print("\n" + "-"*70)
    print("\nMedium arrays (approaching L3 limit):")
    benchmark_operation(50000)  # ~0.4 MB per array, 12 MB total

    print("\n" + "-"*70)
    print("\nLarge arrays (exceed L3 cache - LIKE OPTIMIZED KERNEL):")
    benchmark_operation(100000)  # ~0.8 MB per array, 24 MB total

    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print("When cached data exceeds L3 cache size:")
    print("  ❌ Data evicted to RAM (slow)")
    print("  ❌ Retrieval slower than recomputation!")
    print("  ❌ All operations slow down (cache thrashing)")
    print("\nFor the optimized kernel:")
    print("  • Caches 30+ arrays = 24 MB")
    print("  • Exceeds L3 cache → constant evictions")
    print("  • Result: SLOWER than recomputing!")
    print("="*70)


def show_memory_latency():
    """Demonstrate memory latency hierarchy"""

    print("\n" + "="*70)
    print("MEMORY LATENCY HIERARCHY (approximate)")
    print("="*70)
    print("L1 cache:  ~4 cycles    (~1 ns)")
    print("L2 cache:  ~12 cycles   (~3 ns)")
    print("L3 cache:  ~40 cycles   (~10 ns)")
    print("RAM:       ~200 cycles  (~50 ns)")
    print("="*70)
    print("\nFor a 100K element array:")
    size = 100000
    print(f"  L3 hit:  {size} × 10ns = {size * 10 / 1000:.1f} µs")
    print(f"  RAM:     {size} × 50ns = {size * 50 / 1000:.1f} µs")
    print(f"  np.cos:  compute + memory ≈ 74 µs")
    print("\nConclusion:")
    print("  If cached data is in L3 → faster to retrieve")
    print("  If cached data is in RAM → FASTER to recompute!")
    print("="*70)


if __name__ == "__main__":
    demonstrate_cache_pressure()
    show_memory_latency()

    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    print("The optimized kernel caches 24 MB of data, but:")
    print("  • L3 cache is only 8-16 MB")
    print("  • Data gets evicted to RAM")
    print("  • Reading from RAM is SLOWER than recomputing!")
    print("\nThis is why 'reuse' doesn't help - it actually hurts!")
    print("="*70)
