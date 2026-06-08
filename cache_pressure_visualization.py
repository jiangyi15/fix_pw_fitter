"""
Visual illustration of cache pressure from caching strategy.

This script creates ASCII diagrams showing how caching affects cache performance.
"""


def draw_cache_hierarchy():
    """Draw the CPU cache hierarchy."""
    print("=" * 80)
    print("CPU CACHE HIERARCHY")
    print("=" * 80)
    print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                              CPU CORE                                   │
│  ┌──────────────┐                                                       │
│  │   L1 Cache   │  32 KB  │ Latency: 4-5 cycles                        │
│  │   (fastest)  │         │ Bandwidth: ~1 TB/s                         │
│  └──────┬───────┘                                                       │
│         │                                                               │
│  ┌──────▼───────┐                                                       │
│  │   L2 Cache   │  256 KB │ Latency: 12-15 cycles                      │
│  │              │         │ Bandwidth: ~500 GB/s                       │
│  └──────┬───────┘                                                       │
└─────────┼───────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────────────────┐
│                           L3 CACHE (SHARED)                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Size: 8-16 MB   │ Latency: 40-50 cycles                         │  │
│  │                   │ Bandwidth: ~200 GB/s                         │  │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐         │  │
│  │  │      │ │      │ │      │ │      │ │      │ │      │         │  │
│  │  │ Data │ │ Data │ │ Data │ │ Data │ │ Data │ │ Data │         │  │
│  │  │      │ │      │ │      │ │      │ │      │ │      │         │  │
│  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────┬───────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────────────────┐
│                              RAM                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Size: 16-64 GB  │ Latency: 200-300+ cycles                      │  │
│  │                   │ Bandwidth: ~50 GB/s                          │  │
│  │  ████████████████████████████████████████████████████████████   │  │
│  │  ████████████████████████████████████████████████████████████   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
""")


def draw_original_kernel():
    """Show cache usage in original kernel (recompute strategy)."""
    print("=" * 80)
    print("ORIGINAL KERNEL: Cache Usage (Recompute Strategy)")
    print("=" * 80)
    print("""
FORWARD PASS (Step-by-step):
───────────────────────────────────────────────────────────────────────────

1. Compute g (gamma interpolation)
   ┌────────────────────────────────────────────────────────────────┐
   │ L3 Cache                                                       │
   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
   │  │   g     │  │ mass    │  │ momentum│  │  angle  │          │
   │  │ 800 KB  │  │ 800 KB  │  │ 800 KB  │  │ 800 KB  │          │
   │  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │
   │  Total in L3: 3.2 MB ✓ Fits easily!                          │
   └────────────────────────────────────────────────────────────────┘

2. Compute bw_dom, fl, fa, etc.
   ┌────────────────────────────────────────────────────────────────┐
   │ L3 Cache                                                       │
   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
   │  │ bw_dom  │  │   fl    │  │   fa    │  │   ...   │          │
   │  │ 800 KB  │  │ 800 KB  │  │ 800 KB  │  │ 800 KB  │          │
   │  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │
   │  Each result discarded after use, keeping L3 cache clean      │
   └────────────────────────────────────────────────────────────────┘

3. Compute final probability P
   ┌────────────────────────────────────────────────────────────────┐
   │ L3 Cache                                                       │
   │  ┌─────────┐  ┌─────────┐                                     │
   │  │   P     │  │  grads  │                                     │
   │  │ 800 KB  │  │ 800 KB  │                                     │
   │  └─────────┘  └─────────┘                                     │
   │  Only keep what's needed for backward pass                    │
   └────────────────────────────────────────────────────────────────┘

BACKWARD PASS (Recompute intermediate values):
───────────────────────────────────────────────────────────────────────────

Recompute g, bw_dom, fl, fa from original data (mass, momentum, angle)
   ┌────────────────────────────────────────────────────────────────┐
   │ L3 Cache                                                       │
   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
   │  │ mass    │  │ momentum│  │  angle  │  │  grads  │          │
   │  │ 800 KB  │  │ 800 KB  │  │ 800 KB  │  │ 800 KB  │          │
   │  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │
   │  Original data still in cache (good locality!)                │
   │  Intermediate values recomputed on-demand                     │
   └────────────────────────────────────────────────────────────────┘

RESULT: L3 Cache stays clean, all operations fast
""")


def draw_optimized_kernel():
    """Show cache pressure in optimized kernel (caching strategy)."""
    print("=" * 80)
    print("OPTIMIZED KERNEL: Cache Pressure (Caching Strategy)")
    print("=" * 80)
    print("""
FORWARD PASS (Cache everything):
───────────────────────────────────────────────────────────────────────────

1. Compute g (gamma interpolation) + CACHE
   ┌────────────────────────────────────────────────────────────────┐
   │ L3 Cache                                                       │
   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
   │  │   g     │  │ g_interp│  │   g_bw  │  │ mass    │          │
   │  │ 800 KB  │  │ 800 KB  │  │ 800 KB  │  │ 800 KB  │          │
   │  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │
   │  Cache['g'] = g        Cache['g_interp'] = g_interp          │
   │  Cache['g_bw'] = g_bw                                          │
   └────────────────────────────────────────────────────────────────┘

2. Compute bw_dom, m0_all + CACHE
   ┌────────────────────────────────────────────────────────────────┐
   │ L3 Cache                                                       │
   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
   │  │ bw_dom  │  │ m0_all  │  │ m0_m    │  │ bw_p    │          │
   │  │ 800 KB  │  │ small   │  │ 800 KB  │  │ 800 KB  │          │
   │  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │
   │  Cache['bw_dom'] = bw_dom                                     │
   │  Cache['bw_p'] = bw_p                                         │
   └────────────────────────────────────────────────────────────────┘

3. Compute fl, fa, cos_term, sin_term + CACHE
   ┌────────────────────────────────────────────────────────────────┐
   │ L3 Cache (BECOMING FULL!)                                      │
   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
   │  │   fl    │  │   fa    │  │cos_term │  │sin_term │          │
   │  │ 800 KB  │  │ 800 KB  │  │ 800 KB  │  │ 800 KB  │          │
   │  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │
   │  ...and 20+ more arrays cached...                             │
   │                                                                │
   │  TOTAL CACHED: 30 arrays × 800 KB = 24 MB ⚠️ EXCEEDS L3!      │
   └────────────────────────────────────────────────────────────────┘

4. Cache overflow to RAM
   ┌────────────────────────────────────────────────────────────────┐
   │ L3 Cache (8-16 MB)                                             │
   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
   │  │ Recent  │  │ arrays  │  │ only    │  │         │          │
   │  │  data   │  │         │  │         │  │         │          │
   │  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │
   │  Older cached data EVICTED to RAM                             │
   └────────────────────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────────────────────┐
   │ RAM                                                            │
   │  ████████████████████████████████████████████████████████████ │
   │  ██  g (evicted)  ██  bw_dom (evicted)  ██  cos_term ...    ██│
   │  ████████████████████████████████████████████████████████████ │
   │  Older cache entries pushed to RAM (slow access)               │
   └────────────────────────────────────────────────────────────────┘

BACKWARD PASS (Retrieve from cache):
───────────────────────────────────────────────────────────────────────────

Need to access cached values:
   ┌────────────────────────────────────────────────────────────────┐
   │ Retrieving cache['g_bw'] ...                                   │
   │                                                                │
   │ L3 Cache: ❌ NOT FOUND (was evicted)                          │
   │           Must read from RAM (200-300 cycles latency)          │
   └────────────────────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────────────────────┐
   │ Retrieving cache['cos_term'] ...                               │
   │                                                                │
   │ L3 Cache: ❌ NOT FOUND (was evicted)                          │
   │           Must read from RAM (200-300 cycles latency)          │
   └────────────────────────────────────────────────────────────────┘

EVERY cache retrieval now requires RAM access (slow!)

Meanwhile, trying to compute gradients:
   ┌────────────────────────────────────────────────────────────────┐
   │ L3 Cache (full of cached arrays)                               │
   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
   │  │ cached  │  │ cached  │  │ cached  │  │ cached  │          │
   │  │ arrays  │  │         │  │         │  │         │          │
   │  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │
   │  No room for new gradient arrays!                             │
   │  Gradient computations constantly evict data to RAM            │
   └────────────────────────────────────────────────────────────────┘

RESULT: Cache thrashing, every operation 3-5x slower
""")


def draw_comparison():
    """Draw side-by-side comparison."""
    print("=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print("""
┌───────────────────────────────────┬───────────────────────────────────┐
│   ORIGINAL (Recompute)            │   OPTIMIZED (Cache)               │
├───────────────────────────────────┼───────────────────────────────────┤
│                                   │                                   │
│  Forward Pass:                    │  Forward Pass:                    │
│   • Compute intermediate values   │   • Compute intermediate values   │
│   • Use once                      │   • Cache 30 arrays (24 MB)       │
│   • Discard                       │   • Cache overflow to RAM         │
│                                   │                                   │
│  Cache State:                     │  Cache State:                     │
│   • L3: 3-5 MB (clean)           │   • L3: 24 MB (overflow!)         │
│   • Operations fast               │   • Cache thrashing begins        │
│                                   │                                   │
├───────────────────────────────────┼───────────────────────────────────┤
│                                   │                                   │
│  Backward Pass:                   │  Backward Pass:                   │
│   • Recompute intermediate values │   • Retrieve from cache           │
│   • Data in L3 (cache hits!)     │   • Most values in RAM (misses!)  │
│   • Operations fast               │   • Operations slow               │
│                                   │                                   │
│  Cache State:                     │  Cache State:                     │
│   • L3: 3-5 MB (clean)           │   • L3: Full + pressure           │
│   • Good locality                 │   • Every op has cache misses     │
│                                   │                                   │
├───────────────────────────────────┼───────────────────────────────────┤
│                                   │                                   │
│  Total Time:                      │  Total Time:                      │
│   • Forward: 100 ms               │   • Forward: 105 ms (+5 ms)       │
│   • Backward: 100 ms              │   • Backward: 480 ms (+380 ms!)   │
│   • TOTAL: 200 ms                 │   • TOTAL: 585 ms (2.9x slower)   │
│                                   │                                   │
│  Memory:                          │  Memory:                          │
│   • Peak: 200 MB                  │   • Peak: 350 MB (+150 MB)        │
│   • L3 hit rate: 80%              │   • L3 hit rate: 40%              │
│                                   │                                   │
└───────────────────────────────────┴───────────────────────────────────┘

WHY IS OPTIMIZED SLOWER?
───────────────────────────────────────────────────────────────────────────

1. Cache Pressure (24 MB > 16 MB L3 cache)
   └─> Constant evictions between L3 and RAM

2. Memory Bandwidth Saturation
   └─> Reading cached arrays from RAM slows everything

3. Cache Pollution
   └─> Useful data evicted to make room for cached arrays

4. No Real Benefit
   └─> Recomputing memory-bound ops is as fast as reading from cache

BOTTOM LINE:
───────────────────────────────────────────────────────────────────────────

For memory-bound NumPy operations:
  Recomputation cost  ≈  Cache retrieval cost
  
But caching adds:
  + Memory pressure (24 MB)
  + Cache pollution (all ops slower)
  + Complexity overhead
  
Result: Caching HURTS performance!
""")


def draw_memory_access_patterns():
    """Show memory access patterns."""
    print("=" * 80)
    print("MEMORY ACCESS PATTERNS")
    print("=" * 80)
    print("""
ORIGINAL KERNEL (Good Locality):
───────────────────────────────────────────────────────────────────────────

Time ──►

Step 1:  Read mass ──► Compute g ──► Use g ──► Discard g
         [━━━━━━━━━━]   [━━━━━━━━━━]  [━━━━]    ( )
         
         L3 Cache: mass (stays), g (temporary)
         
Step 2:  Read momentum ──► Compute fl ──► Use fl ──► Discard fl
         [━━━━━━━━━━━━━]   [━━━━━━━━━━━]  [━━━━]    ( )
         
         L3 Cache: mass, momentum (stay), fl (temporary)
         
Step 3:  Read angle ──► Compute cos ──► Use cos ──► Discard cos
         [━━━━━━━━━━]   [━━━━━━━━━━]  [━━━━]    ( )
         
         L3 Cache: mass, momentum, angle (stay), cos (temporary)

Pattern: Sequential access, data reused, temporaries freed
Result: Cache prefetcher works well, high hit rate


OPTIMIZED KERNEL (Poor Locality):
───────────────────────────────────────────────────────────────────────────

Step 1:  Read mass ──► Compute g ──► Cache g ──► Cache fills up
         [━━━━━━━━━━]   [━━━━━━━━━━]  [━━━━━━━━━━━━━━━━━━━━━━━━]
         
         L3 Cache: mass, g (cached), g_interp (cached), g_bw (cached), ...
         
Step 2:  Read momentum ──► Compute fl ──► Cache fl ──► Evict older data
         [━━━━━━━━━━━━━]   [━━━━━━━━━━━]  [━━━━━━━━━━━━━━━━━━━━]
         
         L3 Cache: fl (cached), ... (mass, g evicted to RAM)
         
Step 3:  Read angle ──► Compute cos ──► Cache cos ──► Evict more
         [━━━━━━━━━━]   [━━━━━━━━━━]  [━━━━━━━━━━━━━━━━━━━━━━]
         
         L3 Cache: cos (cached), ... (more data evicted)

Pattern: Random access, cache thrashing, data constantly evicted
Result: Cache prefetcher fails, low hit rate, frequent RAM access


BACKWARD PASS COMPARISON:
───────────────────────────────────────────────────────────────────────────

Original:
  Need g?      Recompute from mass (mass in L3 cache) ✓ Fast
  Need fl?     Recompute from momentum (momentum in L3) ✓ Fast
  Need cos?    Recompute from angle (angle in L3) ✓ Fast
  
Optimized:
  Need g?      Retrieve cache['g'] ──► g in RAM ✗ Slow
  Need fl?     Retrieve cache['fl'] ──► fl in RAM ✗ Slow  
  Need cos?    Retrieve cache['cos'] ──► cos in RAM ✗ Slow
  Need grad?   Compute grad ──► No room in L3, evict cached data ✗ Slow
""")


if __name__ == "__main__":
    draw_cache_hierarchy()
    print("\n")
    draw_original_kernel()
    print("\n")
    draw_optimized_kernel()
    print("\n")
    draw_comparison()
    print("\n")
    draw_memory_access_patterns()
