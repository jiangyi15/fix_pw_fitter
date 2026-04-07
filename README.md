# Fixed Partial Waves Fitter

This package is used to fit partial waves analysis with fixed shape partial waves.

For the fixed partial waves, the amplitude is only related to the coupling constant. It greatly simplifies the problem of partial wave analysis.

## Formula

For the fixed partial waves, all dynamic amplitudes are cached as $F_{ijk}$, where $i$ is for the events, $j$ for the projections, $k$ for the partial waves or components. The coupling parameters are $c_k$, which are complex numbers.

The total amplitude is $A_{ij} = \sum_{k} c_{k} F_{ijk}$, and the probability density is $S_{i} = \sum_{j} | A_{ij} |^2$.

The total probability density is the combination of signal and background. The background is also a fixed shape as $B_{i}$. The ratio is another fixed constant $purity$. The total probability density is $P_i = S_{i}/ N_s * purity + B_{i}/ N_b * (1-purity)$.
$N_s$ and $N_b$ are the sum of MC samples $N_{s} = \sum_{i'} \omega_{i'} S_{i'}$, $N_{b} = \sum_{i'} \omega_{i'} B_{i'}$, where $i'$ is the index of events of MC, and $\omega_{i'}$ is the weight of MC.

To reduce the calculation, the MC value is reordered as
$$
N_{s} = \sum_{i'} \sum_{j} \omega_{i'} \sum_{k}\sum_{k'} c_{k} F_{i'jk} c_{k'}^{\*} F_{i'jk'}^{\*} =
\sum_{k}\sum_{k'} c_{k} c_{k'}^{\*} M_{kk'}
$$, where $M_{kk'}= \sum_{i'} \sum_{j}  \omega_{i'} F_{i'jk} F_{i'jk'}^{*}$.

In the fit, we minimize $-\ln L = - \sum w_i \ln P_{i}$. $w_i$ is the weight of data.

## Gradients

To reduce the calculation of gradients, we use complex number gradients.

$\partial |\sum_{k} c_k F_{ijk}|^2/\partial c_k = \sum_{k'} F_{ijk} F_{ijk}^{\*} c_{k'}^{\*} =  F_{ijk} A_{ij}^{\*}$. Using the complex conjugate relation we can directly get $\partial (-\ln L)/\partial c_k^{\*} = (\partial (-\ln L)/\partial c_k)^{\*}$. This reduces many computations. The other parts are similar and can be evaluated using chain rules. Using common sub-expressions, we can reduce much more computations.

## Performance

The implementation uses a **single fused CUDA kernel** for the forward pass (A → S → P → NLL → G → S_corr) and cuBLAS ZGEMV for the gradient. All intermediates stay in registers — no global memory traffic between stages.

### GPU Speedup vs NumPy CPU (single thread)

| n_data | n_mc | n_comp | CPU | GPU | Speedup |
|--------|------|--------|-----|-----|---------|
| 10,000 | 50,000 | 20 | 4.9 ms | 0.2 ms | **31×** |
| 50,000 | 200,000 | 50 | 47 ms | 1.8 ms | **26×** |
| 100,000 | 500,000 | 50 | 89 ms | 2.7 ms | **33×** |
| 100,000 | 500,000 | 100 | 156 ms | 4.1 ms | **38×** |

For n_data = 10⁶, n_comp = 100, a single evaluate takes **~0.74 s** on GPU vs **~5 s** on CPU (~7×, still CPU-bound at large sizes).

### M Pre-computation

The overlap matrix $M_{kk'}$ is computed in Python via NumPy with chunked processing (supports mmap for large F_mc). Typical: **0.8 s** for n_mc = 500,000, n_comp = 50.

### Key Optimizations

1. **Fused forward kernel** — 6 kernel launches replaced by 1; all intermediates in registers
2. **cuBLAS ZGEMV gradient** — replaces K separate atomicAdd kernels with a single BLAS call
3. **Warp-level reduction** — NLL and S_corr use `__shfl_down` instead of `atomicAdd`
4. **All data on GPU** — uploaded once at creation, zero H2D transfers during evaluate
5. **Chunked M pre-compute** — F_mc never fully in RAM (supports mmap for n_mc > 10⁷)

