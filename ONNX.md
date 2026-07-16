# ONNX Model — Cross-Backend Comparison

This ONNX model implements the optimized algorithm from [DERIVATION.tex](DERIVATION.tex)
Section 7 (page 8).  Benchmark results on **RTX 3070 Ti Laptop GPU** unless
otherwise noted.

## Model Structure

| Property | Value |
|----------|-------|
| Opset version | 11 |
| Total nodes | 75 |
| MatMul ops | 12 |
| Inputs | 10 (F_real, F_imag, w, B, M_real, M_imag, N_b, purity, c_real, c_imag) |
| Outputs | 3 (nll, grad_real, grad_imag) |
| Precision | float32 |
| N dimension | dynamic (default) or fixed via `n_events=` |

### Dynamic vs fixed batch

The model accepts an optional `n_events` parameter.  When omitted (default), the
event dimension N is **dynamic** — one model works for any number of events.
When set to an integer, N is **fixed** and all shapes are fully concrete,
allowing ONNX Runtime to apply more aggressive constant-folding and
memory-planning.

```python
# dynamic N — any number of events
model = build_fpwfitter_graph(n_comp=100, n_proj=2)

# fixed N = 50000
model = build_fpwfitter_graph(n_comp=100, n_proj=2, n_events=50000)
```

Fixed N also makes the internal Reshape shapes explicit (e.g. `[100000, 100]`
instead of `[-1, 100]`), which removes shape-inference overhead during graph
execution.

### Key geometric optimisation

Instead of unrolling per-projection j (which would make the graph size scale
linearly with j), the ONNX model reshapes F from `(N, j, k)` → `(N*j, k)` and
does a single complex MatMul for all projections simultaneously.  This keeps
the graph at 75 nodes regardless of `n_proj` — no unrolling.

## Accuracy Cross-Check (seed=1001, N=1,000,000, k=100, j=2)

| Metric | NumPy FP64 | ONNX CUDA FP32 | Difference |
|--------|------------|----------------|-----------|
| NLL | -1949837.49 | -1949838.00 | 2.6×10⁻⁷ rel |
| grad L2 | — | — | **9.0×10⁻⁷ rel** |

ONNX float32 accuracy is well within the 5×10⁻⁶ typical requirement.

## Latency Comparison at N=1,000,000, k=100, j=2

| Backend | Time | Events/s | vs ONNX CPU |
|---------|------|----------|-------------|
| NumPy FP64 | 4512 ms | 0.22 M/s | 0.06× |
| **ONNX CPU** (baseline) | **287 ms** | **3.5 M/s** | **1×** |
| ONNX CUDA (full eval, CPU→GPU) | 158.8 ms | 6.3 M/s | 1.8× |
| Custom CUDA FP64 (README) | 23.5 ms | 42.6 M/s | 12× |
| **ONNX CUDA + io_binding** | **20.1 ms** | **49.8 M/s** | **14×** |
| **Custom CUDA FP32** (README) | **11.4 ms** | **87.7 M/s** | **25×** |

## Full Result Matrix

### Custom CUDA (from README.md)

| n_data | n_comp | CPU | GPU FP64 | GPU FP32 | FP64 vs CPU | FP32 vs CPU |
|--------|--------|-----|----------|----------|-------------|-------------|
| 10,000 | 20 | 2.7 ms | 0.2 ms | 0.1 ms | 18× | 24× |
| 50,000 | 50 | 45 ms | 0.8 ms | 0.3 ms | 60× | 150× |
| 100,000 | 50 | 92 ms | 1.4 ms | 0.6 ms | 66× | 153× |
| 100,000 | 100 | 154 ms | 2.5 ms | 1.2 ms | 61× | 128× |
| **1,000,000** | **100** | **1.37 s** | **23.5 ms** | **11.4 ms** | **58×** | **120×** |

### ONNX (this benchmark)

| n_data | n_comp | ONNX CPU | ONNX CUDA (full) | ONNX CUDA (io_binding) |
|--------|--------|----------|------------------|------------------------|
| 100,000 | 100 | — | — | 3.2 ms |
| 1,000,000 | 100 | **287 ms** | **158.8 ms** | **20.1 ms** |

### NumPy FP64 reference (this benchmark)

| n_data | n_comp | Time | Events/s |
|--------|--------|------|----------|
| 1,000,000 | 100 | **4.51 s** | 0.22 M/s |

## Throughput Summary (N=1,000,000, k=100)

| Backend | Events/s | GFLOP/s (effective) | Bottleneck |
|---------|----------|-------------------|------------|
| NumPy FP64 | 0.22 M/s | 0.8 | CPU (single-thread BLAS) |
| **ONNX CPU** | **3.5 M/s** | **12** | CPU (multi-thread BLAS) |
| Custom CUDA FP64 | 42.6 M/s | 145 | GPU memory bandwidth |
| **ONNX CUDA + io_binding** | **49.8 M/s** | **170** | GPU kernel launch overhead |
| **Custom CUDA FP32** | **87.7 M/s** | **299** | GPU memory bandwidth |
| ONNX CUDA (CPU→GPU) | 6.3 M/s | 21.5 | **PCIe** (10.1 GB/s) |

## Transfer Bottleneck

The ONNX model accepts data arrays (F, w, B) as *inputs*, not embedded
initializers.  For `N=1M` the per-call transfer is:

| Tensor | Size |
|--------|------|
| F_real | 800 MB |
| F_imag | 800 MB |
| w | 4 MB |
| B | 4 MB |
| M, c, etc. | < 1 MB |
| **Total** | **~1.6 GB** |

At ~10 GB/s effective PCIe 3.0 x16 bandwidth this takes ~**125 ms** —
79 % of the 158.8 ms full-eval time.

## Impact for a Real Fit Loop

A typical L-BFGS fit runs 10–50 iterations, updating `c` each time.
The fitter must keep data GPU-resident (via io_binding) to avoid per-iteration PCIe cost.

| Backend | Per iteration | 50 iterations | Events/s |
|---------|--------------|---------------|----------|
| NumPy FP64 | 4.51 s | 226 s | 0.22 M/s |
| **ONNX CPU** | 287 ms | 14.4 s | 3.5 M/s |
| Custom CUDA FP64 | 23.5 ms | 1.18 s | 42.6 M/s |
| ONNX CUDA + io_binding | 20.1 ms | **1.01 s** | 49.8 M/s |
| Custom CUDA FP32 | 11.4 ms | **0.57 s** | 87.7 M/s |
| ONNX CUDA (CPU feeds, no io_binding) | 158.8 ms | **7.94 s** | 6.3 M/s |

**Key**: for a real fit loop you *must* use io_binding so data stays on GPU.
Otherwise PCIe transfer dominates.

## How to Reproduce

```python
import onnxruntime as ort
import numpy as np
import onnx
from create_onnx_model import build_fpwfitter_graph

# Build fixed-N model
model = build_fpwfitter_graph(n_comp=100, n_proj=2, n_events=1_000_000)
onnx.save(model, "fpwfitter_1m.onnx")

# ONNX Runtime CUDA
sess = ort.InferenceSession(model.SerializeToString(),
                            providers=['CUDAExecutionProvider'])

# Use io_binding for GPU-resident data
io = sess.io_binding()
for name, arr in data.items():
    io.bind_cpu_input(name, arr)
for o in sess.get_outputs():
    io.bind_output(o.name)

# Fit loop — only c changes
for iteration in range(50):
    c_new = compute_new_c(...)
    io.bind_cpu_input("c_real", np.ascontiguousarray(c_new.real))
    io.bind_cpu_input("c_imag", np.ascontiguousarray(c_new.imag))
    sess.run_with_iobinding(io)
    io.synchronize_outputs()
    nll, grad = ...  # read outputs
```

## Why ONNX is slower

| Factor | Custom CUDA | ONNX Runtime |
|--------|-------------|--------------|
| Kernel launches | 2 (1 fused + 1 ZGEMV) | 75 (one per op) |
| Forward fusion | A→S→P→NLL→G→S_corr in one kernel | No fusion |
| Reductions | Warp shuffle (`__shfl_down`) | Separate kernel |
| Complex MatVec | cuBLAS ZGEMV (native) | 4 real MatMul + Add/Sub |
| Memory layout | Coalesced `(KC, N, JP)` | Row-major |
| Data residence | GPU at creation | CPU, must transfer |

## When to use ONNX

- Portable inference without compiling CUDA
- Integrating into an ONNX Runtime pipeline
- Moderate perf requirements (< 50 iterations, ~1 fit/s)

Use `FpwFitter` / `FpwFitterMP` for peak GPU performance.
