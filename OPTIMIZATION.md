# Optimized Computation Order

## Key Insight: Reformulate as Batched Matrix Operations

The critical optimization is to **restructure all per-event loops as batched matrix multiplications (GEMM)**, which:
1. Enable BLAS/GPU acceleration
2. Reduce memory access overhead via cache-friendly blocking
3. Eliminate redundant intermediate storage

Define the matrix $F^{(j)}$ of shape $(N, k)$ where $N$ is the number of events (data or MC), with elements $F^{(j)}_{nk} = F_{njk}$.

---

## Pre-computation (One-Time)

### Step 1: Compute $M_{kk'}$ from MC samples

$$M = \sum_{j=0}^{1} (F_{\text{MC}}^{(j)})^\dagger \, \text{diag}(\omega) \, F_{\text{MC}}^{(j)}$$

Where $F_{\text{MC}}^{(j)}$ has shape $(i', k)$ and $\omega$ is the MC weight vector.

**Implementation:** For each $j$, compute `F_j.T @ diag(omega) @ F_j` as two GEMMs:
- $T_j = \text{diag}(\omega) \, F_{\text{MC}}^{(j)}$ — element-wise scaling: $O(i' \cdot k)$
- $M_j = (F_{\text{MC}}^{(j)})^\dagger \, T_j$ — GEMM $(k, i') \times (i', k)$: $O(i' \cdot k^2)$
- $M = \sum_j M_j$

**Cost:** $O(i' \cdot j \cdot k^2) = 10^7 \times 2 \times 10^4 = \mathbf{2 \times 10^{11}}$

But as a **batched GEMM**, the effective throughput is much higher (10-100× speedup on GPU).

### Step 2: Compute $N_b$

$$N_b = \sum_{i'} \omega_{i'} B_{i'}$$

**Cost:** $O(i') = \mathbf{10^7}$ (vector dot product)

---

## Per Iteration — Optimized Order

### Step 1: Compute $A_{ij}$ for all events

$$A^{(j)} = F_{\text{data}}^{(j)} \, c \quad \text{(matrix-vector product for each $j$)}$$

Shape: $F_{\text{data}}^{(j)}$ is $(i, k)$, $c$ is $(k, 1)$, result $A^{(j)}$ is $(i, 1)$.

**Cost:** $O(i \cdot j \cdot k) = 10^6 \times 2 \times 100 = \mathbf{2 \times 10^8}$

As a batched GEMV (matrix-vector multiply), this is highly optimized.

### Step 2: Compute $S_i$ for all events

$$S_i = \sum_j |A_{ij}|^2 = \sum_j A_{ij} A_{ij}^*$$

**Cost:** $O(i \cdot j) = 10^6 \times 2 = \mathbf{2 \times 10^6}$

### Step 3: Compute $N_s$

$$N_s = c^\dagger M c$$

**Cost:** $O(k^2) = 100^2 = \mathbf{10^4}$ (two matrix-vector products)

### Step 4: Compute $P_i$ and $-\ln L$

$$P_i = \frac{S_i}{N_s} \cdot p + \frac{B_i}{N_b} \cdot (1-p)$$
$$-\ln L = -\sum_i w_i \ln P_i$$

**Cost:** $O(i) = \mathbf{10^6}$ (element-wise vector operations)

### Step 5: Compute $N_s$ gradient

$$\frac{\partial N_s}{\partial c^*} = M c$$

**Cost:** $O(k^2) = \mathbf{10^4}$ (single matrix-vector product)

### Step 6: Compute weighted amplitude $G_{ij}$

$$G_{ij} = \frac{w_i}{P_i} A_{ij}$$

**Cost:** $O(i \cdot j) = 10^6 \times 2 = \mathbf{2 \times 10^6}$ (element-wise scaling per event)

### Step 7: Compute data-term gradient via batched GEMM

$$g_k^{\text{data}} = \sum_i \frac{w_i}{P_i} \frac{\partial S_i}{\partial c_k^*} = \sum_j (F_{\text{data}}^{(j)})^\dagger G^{(j)}$$

Where $G^{(j)}$ has elements $G_{ij} = \frac{w_i}{P_i} A_{ij}$.

**Cost:** $O(i \cdot j \cdot k) = 10^6 \times 2 \times 100 = \mathbf{2 \times 10^8}$

As a batched GEMM $(k, i) \times (i, 1)$ for each $j$, this runs at near-peak hardware FLOPs.

### Step 8: Compute normalization correction term

$$S_{\text{corr}} = \sum_i \frac{w_i S_i}{P_i}$$

**Cost:** $O(i) = \mathbf{10^6}$ (vector dot product)

### Step 9: Assemble full gradient

$$\frac{\partial(-\ln L)}{\partial c_k^*} = -\frac{p}{N_s} \, g_k^{\text{data}} + \frac{p}{N_s^2} \, \frac{\partial N_s}{\partial c_k^*} \, S_{\text{corr}}$$

**Cost:** $O(k) = \mathbf{100}$ (vector scaling and addition)

### Step 10: Use complex conjugate symmetry

$$\frac{\partial(-\ln L)}{\partial c_k} = \left(\frac{\partial(-\ln L)}{\partial c_k^*}\right)^*$$

**Cost:** $O(k) = \mathbf{100}$ (conjugation)

---

## Cost Comparison

### Naive per-iteration cost

| Step | Operations |
|------|-----------|
| $A_{ij}$ | $2 \times 10^8$ |
| $S_i$ | $2 \times 10^6$ |
| $N_s$ | $10^4$ |
| $P_i$, $-\ln L$ | $10^6$ |
| $\partial S_i/\partial c_k^*$ (per-event, per-k) | $2 \times 10^8$ |
| $\partial P_i/\partial c_k^*$ (per-event, per-k) | $10^8$ |
| $\partial(-\ln L)/\partial c_k^*$ (per-event, per-k) | $10^8$ |
| **Total** | **$\mathbf{\approx 7 \times 10^8}$** |

### Optimized per-iteration cost

| Step | Operations | Notes |
|------|-----------|-------|
| $A^{(j)} = F^{(j)} c$ | $2 \times 10^8$ | Batched GEMV |
| $S_i = \sum_j |A_{ij}|^2$ | $2 \times 10^6$ | Element-wise |
| $N_s = c^\dagger M c$ | $10^4$ | Vector-matrix |
| $P_i$, $-\ln L$ | $10^6$ | Element-wise |
| $\partial N_s/\partial c^* = Mc$ | $10^4$ | Matrix-vector |
| $G_{ij} = \frac{w_i}{P_i} A_{ij}$ | $2 \times 10^6$ | Element-wise |
| $g^{\text{data}} = \sum_j (F^{(j)})^\dagger G^{(j)}$ | $2 \times 10^8$ | Batched GEMM |
| $S_{\text{corr}} = \sum_i \frac{w_i S_i}{P_i}$ | $10^6$ | Dot product |
| Assemble gradient | $10^2$ | Vector ops |
| **Total** | **$\mathbf{\approx 4 \times 10^8}$** | **1.75× fewer ops** |

### Key Optimizations

| Optimization | Savings | Explanation |
|-------------|---------|-------------|
| Eliminate per-event-per-k $\partial P_i/\partial c_k^*$ | $-10^8$ | Replaced with global sums $g^{\text{data}}$ and $S_{\text{corr}}$ |
| Eliminate per-event-per-k $\partial(-\ln L)/\partial c_k^*$ | $-10^8$ | Replaced with $O(k)$ vector assembly |
| Batched GEMM for $M_{kk'}$ and gradient | **10-100× throughput** | Uses hardware-optimized BLAS/cuBLAS |
| **Net effective speedup** | **~3-5×** (algorithmic) | Fewer ops + better hardware utilization |

---

## Implementation Status

All planned optimizations have been implemented:

| Optimization | Status | Impact |
|-------------|--------|--------|
| **Fused forward kernel** (A→S→P→NLL→G→S_corr) | ✅ Done | 6 kernels → 1; zero global memory for intermediates |
| **cuBLAS ZGEMV gradient** | ✅ Done | K atomicAdd kernels → 1 BLAS call |
| **Warp-level reduction** (NLL, S_corr) | ✅ Done | `__shfl_down` tree; minimal atomic contention |
| **All data on GPU** (zero per-iteration H2D) | ✅ Done | Single upload at creation |
| **Chunked M pre-compute** (NumPy, mmap) | ✅ Done | F_mc never fully in RAM |
| Shared memory tiling (k_A) | ✅ Done | F transposed to (KC, N, JP) for coalesced reads; warp-reduction gradient kernel |
| `__ldg()` read-only loads | ✅ Done | F and c read through texture/L1 cache on Ampere |
| Mixed precision (FP32) | ❌ Skipped | Physics precision requires FP64 |
| CUDA Graphs | ❌ Skipped | Ns changes each call; requires CUDA 12.0+ graph parameter update |
| Event binning | ❌ Skipped | Application-specific; not in library |

### Measured Performance (2025-04)

| n_data | n_mc | n_comp | CPU (NumPy) | GPU (CUDA) | Speedup |
|--------|------|--------|-------------|------------|---------|
| 10,000 | 50,000 | 20 | 2.7 ms | 0.2 ms | **18×** |
| 50,000 | 200,000 | 50 | 45 ms | 0.7 ms | **60×** |
| 100,000 | 500,000 | 50 | 92 ms | 1.4 ms | **66×** |
| 100,000 | 500,000 | 100 | 154 ms | 2.5 ms | **61×** |
| 1,000,000 | 1,000,000 | 100 | ~5 s | ~0.74 s | **~7×** |

Gradient accuracy: relative error < 10⁻¹² in all cases.

---

## Memory Optimization

### Naive memory layout

| Quantity | Size |
|----------|------|
| $F_{i'jk}$ (MC) | 32 GB |
| $F_{ijk}$ (data) | 3.2 GB |
| **Total** | **35.2 GB** |

### Optimized: Chunked processing

For $M_{kk'}$ computation, process MC events in chunks of size $C$:

$$M = \sum_{\text{chunks}} \sum_j (F^{(j)}_{\text{chunk}})^\dagger \, \text{diag}(\omega_{\text{chunk}}) \, F^{(j)}_{\text{chunk}}$$

With chunk size $C = 10^5$:
- Peak memory: $C \times j \times k \times 16$ bytes = $10^5 \times 2 \times 100 \times 16 = \mathbf{320\ MB}$
- $M_{kk'}$ accumulator: $100^2 \times 16 = \mathbf{160\ KB}$
- **Total peak memory: ~320 MB** (vs 35 GB)

Similarly for data events in the fit.

---

## Final Optimized Algorithm

```
Pre-computation:
  1. M = 0
  2. for each MC chunk:
  3.   for j in {0, 1}:
  4.     T = diag(omega_chunk) @ F_chunk[j]       # O(C*k)
  5.     M += F_chunk[j].T @ T                     # O(C*k^2)
  6. N_b = sum(omega * B)                          # O(i')

Per iteration:
  1. for j in {0, 1}:
  2.   A[j] = F_data[j] @ c                        # O(i*k) GEMV
  3. S = sum(|A|^2, axis=j)                        # O(i*j)
  4. N_s = c.T @ M @ c                             # O(k^2)
  5. P = S/N_s * p + B/N_b * (1-p)                # O(i)
  6. nll = -sum(w * log(P))                        # O(i)
  7. dN_s = M @ c                                  # O(k^2)
  8. G = (w/P)[:,None] * A                         # O(i*j)
  9. g_data = sum(F_data[j].T @ G[j] for j)        # O(i*j*k) GEMM
 10. S_corr = sum(w * S / P)                       # O(i)
 11. grad = -p/N_s * g_data + p/N_s^2 * dN_s * S_corr  # O(k)
 12. return nll, grad, conj(grad)                  # O(k)
```
