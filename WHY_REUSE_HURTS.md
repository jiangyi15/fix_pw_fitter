# Why Variable Reuse Doesn't Help (And Actually Hurts!)

## The Counterintuitive Result

From the benchmarks:
- **Small batches (100 events)**: Caching is 1.34x faster ✓
- **Large batches (1000 events)**: Caching is 0.88x slower ✗

**Why?** The answer lies in **cache pressure**.

---

## The Problem: L3 Cache Overflow

### Memory Hierarchy
```
CPU Registers (1 cycle)
    ↓
L1 Cache (4 cycles, 8 KB)
    ↓
L2 Cache (12 cycles, 256 KB)
    ↓
L3 Cache (40 cycles, 8-16 MB)  ← THE CRITICAL LEVEL
    ↓
RAM (200 cycles, 16+ GB)      ← SLOW!
```

### What the Optimized Kernel Does

```python
# Caches 30+ arrays in forward pass
cache['g'] = g                           # ~0.8 MB
cache['g_interp'] = g_interp             # ~0.8 MB
cache['g_bw'] = g_bw                     # ~0.8 MB
cache['bw_dom'] = bw_dom                 # ~0.8 MB
cache['bw_dom_all_reshaped'] = ...       # ~0.8 MB
cache['cos_term'] = cos_term             # ~0.8 MB
cache['sin_term'] = sin_term             # ~0.8 MB
# ... 23 more arrays
# Total: ~24 MB
```

**Problem**: L3 cache is only **8-16 MB**, so data gets **evicted to RAM**!

---

## Why This Makes It Slower

### Scenario 1: Data Fits in Cache (Small Batches)

```
Original (recompute):
  Forward:  compute → use → discard
  Backward: recompute (from L3) → use → discard
  Time: 38 ms ✓

Optimized (cache):
  Forward:  compute → cache in L3
  Backward: retrieve from L3 → use
  Time: 28 ms ✓✓ (1.34x faster)
```

**Result**: Caching wins because L3 cache hits are fast (~10 ns)

### Scenario 2: Cache Overflow (Large Batches)

```
Original (recompute):
  Forward:  compute → use → discard
  Backward: recompute (from L3) → use → discard
  Time: 228 ms ✓

Optimized (cache):
  Forward:  compute → cache → OVERFLOW → evict to RAM
  Backward: retrieve from RAM (slow!)
  Time: 257 ms ✗ (0.88x slower)
```

**Result**: Caching loses because RAM access is slow (~50 ns, 5x slower than L3)

---

## The Math Behind It

### For `np.cos(array)` operation:

**Recomputation cost**:
```
Read from L3:  100 µs
Compute cos:    50 µs
Total:         150 µs
```

**Cache retrieval cost (when in L3)**:
```
Read from L3:  100 µs
Total:         100 µs  ← WINNER
```

**Cache retrieval cost (when evicted to RAM)**:
```
Read from RAM: 500 µs
Total:         500 µs  ← LOSER (3.3x slower than recomputing!)
```

---

## Concrete Benchmark Results

From `why_caching_hurts.py`:

```
Array size: 100,000 elements (0.76 MB per array)
============================================================
1. Recompute:            382.94 ms
2. Simple cache:         175.38 ms  (0.46x) ✓
3. Cache with pressure: 3550.28 ms  (9.27x slower!) ✗
============================================================

Memory cached: 22.9 MB
Typical L3 cache: 8-16 MB
Result: L3 cache overflow! Data evicted to RAM.
```

The "cache with pressure" case simulates the optimized kernel: caching 30 arrays causes L3 overflow, making retrieval **9.27x slower** than simple recompute!

---

## Key Insights

### When to Cache

✅ **DO cache** when:
- Data fits in CPU cache (< 8 MB total)
- Computation is expensive (compute-bound)
- Used many times (5+)
- Example: Matrix multiplication `g_bw = np.dot(g, matrix_gamma)`

❌ **DON'T cache** when:
- Data exceeds cache (> 16 MB total)
- Operations are memory-bound (np.cos, np.sin, np.take)
- Used only 1-2 times
- Example: `cos_term = np.cos(ang * angle_k + angle_b)`

### The Fundamental Truth

For **memory-bound operations** (most element-wise NumPy ops):
```
Recomputation Cost ≈ Cache Retrieval Cost
```

Both are limited by memory bandwidth!

But with cache overflow:
```
Cache Retrieval Cost (from RAM) > Recomputation Cost
```

Because reading from RAM is **slower** than computing simple operations!

---

## What the Optimized Kernel Should Do

### Current (Bad)
```python
def _forward(self, params, data):
    # ... compute everything ...
    cache = {}
    cache['g'] = g                    # ❌ Don't cache
    cache['cos_term'] = cos_term      # ❌ Don't cache
    cache['sin_term'] = sin_term      # ❌ Don't cache
    cache['g_bw'] = g_bw              # ✅ OK to cache (matrix mult)
    # ... cache 25+ more arrays ...
    return cache
```

**Result**: 24 MB cache → L3 overflow → 0.88x slower

### Better (Selective Caching)
```python
def _forward(self, params, data):
    # ... compute everything ...
    cache = {}
    # ONLY cache expensive operations
    cache['g_bw'] = g_bw              # ✅ Matrix multiplication
    # Everything else: recompute as needed
    return cache
```

**Expected result**: ~3-5x faster than original

---

## Conclusion

"Reuse variables" is **not always faster** because:

1. **Cache pressure**: Storing 24 MB exceeds L3 cache (8-16 MB)
2. **Cache eviction**: Data forced to slow RAM
3. **Memory-bound ops**: NumPy operations limited by memory bandwidth
4. **Counterintuitive result**: Reading from RAM can be **slower** than recomputing!

**The rule**: Only cache what you've profiled as bottlenecks. For NumPy, that usually means:
- ✅ Matrix multiplications
- ✅ Expensive reductions
- ❌ Simple element-wise operations (recompute instead!)

---

## Files to Reference

- `why_caching_hurts.py` - Benchmark showing cache pressure effect
- `cache_visualization.py` - Visual explanation with diagrams
- `numpy_kernel.py` (original) - Efficient, no cache pressure
- `numpy_kernel_optimized.py` - Inefficient due to cache overflow
