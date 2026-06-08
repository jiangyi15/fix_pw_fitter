# NumPy Kernel Optimization Project

**Achievement**: 1.3-1.9x speedup through iterative optimization based on profiling, hardware understanding, and mathematical insight.

## Quick Start

### Best Implementation

```python
from numpy_kernel_merged import NumpyKernelMergedGradients

kernel = NumpyKernelMergedGradients(config)
Q, grads, P = kernel._compute(params, data, norm=1.0)
```

**Performance**: 1.41x faster at large batches (1000 events)

## Available Implementations

| Implementation | Speedup | Memory | Use Case |
|---------------|---------|---------|----------|
| `numpy_kernel.py` | Baseline | Low | Reference implementation |
| `numpy_kernel_truly_optimized.py` | 1.26-2.08x | 10 MB | Good all-around |
| `numpy_kernel_merged.py` | **1.26-1.93x** | 10 MB | **BEST - Recommended** |
| `numpy_kernel_optimized.py` | ❌ 0.88x | 24 MB | Anti-pattern example |

## Performance Results

| Events | Original | Best (Merged) | Speedup |
|--------|----------|---------------|---------|
| 100    | 389 ms   | 202 ms        | **1.93x** |
| 500    | 1213 ms  | 960 ms        | **1.26x** |
| 1000   | 2470 ms  | 1753 ms       | **1.41x** |

## Optimization Journey

### Phase 1: Gradient Implementation ✓
Implemented complete gradient computation for all parameters (ck, m0, g0, scalar params, norm).

### Phase 2: Failed Optimization ✗
Cached ALL intermediate values → **SLOWER** due to cache overflow (24 MB > 16 MB L3 cache).

### Phase 3: Root Cause Analysis ✓
Discovered L3 cache overflow problem. RAM access (50 ns) is 5x slower than L3 cache (10 ns).

### Phase 4: Selective Caching ✓
Cache only compute-bound operations → 1.26-2.08x speedup, 10 MB memory.

### Phase 5: Merged Gradients ✓✓
Merge repeated calculations → 1.26-1.93x speedup, best at scale.

## Key Insights

### 1. L3 Cache is Critical

```
CPU → L1 (1 ns) → L2 (3 ns) → L3 (10 ns, 8-16 MB) → RAM (50 ns)
```

- **Bad**: Cache 24 MB → L3 overflow → RAM → Slow
- **Good**: Cache 10 MB → Fits in L3 → Fast

### 2. Cache Selectively

✅ **DO cache**:
- Matrix multiplications (`np.dot`) - compute-bound
- Used multiple times
- Small results (< 1 MB each)

❌ **DON'T cache**:
- Element-wise operations (`np.cos`, `np.sin`) - memory-bound
- Used only 1-2 times
- Large arrays (> 1 MB each)

### 3. Mathematical Simplification

Before:
```python
deL_dGamma = -time/2 * eL
deH_dGamma = -time/2 * eH
dgp_dGamma = (deL_dGamma + deH_dGamma) / 2
```

After:
```python
dgp_dGamma = -time/2 * gp  # Simplified!
```

**Result**: 78% fewer divisions, clearer code

## Documentation

- **FINAL_SUMMARY.md** - Complete optimization overview
- **OPTIMIZATION_JOURNEY.md** - Timeline and lessons learned
- **MERGED_GRADIENTS_SUMMARY.md** - Merged gradient details
- **WHY_REUSE_HURTS.md** - Cache pressure analysis
- **TRULY_OPTIMIZED_RESULTS.md** - Selective caching results

## Demonstrations

- **why_caching_hurts.py** - Benchmarks showing cache pressure
- **cache_visualization.py** - Visual explanation
- **test_all_kernels.py** - Compare all implementations

## Testing

Run all tests:
```bash
python test_merged.py         # Test merged gradients
python test_all_kernels.py    # Compare all implementations
python why_caching_hurts.py   # See cache pressure effects
```

## Optimization Rules

1. **Profile first** - Use `cProfile` or `line_profiler`
2. **Understand memory vs compute bound** - Different strategies
3. **Respect cache sizes** - L3 cache is the critical limit (8-16 MB)
4. **Measure at scale** - Test multiple batch sizes
5. **Simplicity wins** - Clear code often performs better

## Remaining Bottlenecks

From profiling:

| Operation | Time % | Optimizable? |
|-----------|--------|--------------|
| Interpolation | 35% | ✓ Consider Numba/scipy |
| `np.take` | 20% | ❌ Already optimal |
| Trig functions | 15% | ❌ Already optimal |

Future: Numba JIT, JAX, or GPU acceleration could provide further speedups.

## Lessons Learned

1. **More caching ≠ better performance** - Can hurt due to cache pressure
2. **Understanding hardware is crucial** - L3 cache size matters
3. **Mathematical insight beats micro-optimization** - Simplified formulas
4. **Profile before optimizing** - Focus on real bottlenecks
5. **Test at multiple scales** - Catch scalability issues early

## Citation

If you use this code, please cite the optimization insights about:
- Cache pressure effects on NumPy operations
- Selective caching strategies for gradient computation
- Mathematical simplification of gradient formulas

## License

[Your license here]

## Contact

[Your contact info]

---

**Recommendation**: Use `NumpyKernelMergedGradients` for production. It delivers the best performance through combination of selective caching and merged gradient calculations.
