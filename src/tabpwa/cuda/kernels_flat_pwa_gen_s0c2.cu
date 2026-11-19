/**
 * cuda_flat_pwa_gen_s0c2 — `cuda_flat_pwa_gen_s0c` with the three per-chunk
 * feature reductions fused into ONE launch.  Everything else is s0c's file,
 * symbol for symbol (`gsc_` -> `gsc2_`, `GSC_` -> `GSC2_`, `..._s0c` ->
 * `..._s0c2`).
 *
 * ── the ONE change, and why it is worth a copy ──────────────────────────
 *
 * Measured on `tests/config_pwa_gen_km.yml`, ne = 4000, per compute:
 *
 *     gsc_amp_kernel                        123.0 us   78 % SM  (real work)
 *     gsc_pwa_reduce_kernel                  19.4 us
 *     gsc_grad_ck_kernel                     18.7 us
 *     gsc_grad_coeff_kernel                  14.5 us
 *     gsc_reduce_sum_kernel                   9.9 us
 *     gsc_reduce_sum_features_tree_kernel     16.25 us  x 3 launches = 48.7 us
 *
 * The three tree reductions move a handful of doubles (grid 4 at ne 4000,
 * 128 threads, one atomicAdd per feature for ~26/64 values): 48.7 us is
 * ~21 % of the ~234 us iteration and is pure launch + atomic overhead.  One
 * launch with `grid = dim3(n_chunks, 3)` — buffer selected by `blockIdx.y` —
 * replaces them.
 *
 * MEASURED end to end (`tests/benchmark_s0c_variants.py`, idle card, min of 5
 * interleaved rounds, km fixture, one chunk): ne 4000 s0c 129.5 us -> s0c2
 * 121.1 us = **8.4 us / 6.9 %**, so two launches cost ~4.2 us each on the wall
 * clock (ncu's per-launch 16.25 us was profiler-inflated).  At ne 20000 and
 * 50000 the same fixed cost is 1.6 % and 0.1-1.4 % (the amp kernel dominates);
 * the fusion is per CHUNK, so at rsize 17 / ne 50000, where the chunk rule
 * gives 2 chunks, s0c2 still measures 1.4-1.6 % ahead.
 *
 * On the PRODUCTION shape (`config_kn4_plot.yml`: n_basis 64 > 32) the acc
 * fallback applies, so the gain is 0.0 % by design — that path is a
 * different kernel (one block per feature) and fusing it needs a
 * `dim3(max_nf, 3)` grid, i.e. a second fused kernel, not this one.
 *
 * ── bit-identical by construction ──────────────────────────────────────
 *
 * Each `blockIdx.y` slice runs the UNCHANGED tree body: same block shape
 * (128), same chunk rule (1024), same shared-memory layout (`f*blockDim.x +
 * tid`), same tree fold order, same `atomicAdd` per feature — and the three
 * slices write to three DISJOINT output ranges, so the atomics of one slice
 * cannot reorder the other's.  The fusion is therefore a pure launch-structure
 * change: `np.array_equal` against s0c on Q/P/dck/dcoeff/dnorm at rsize 1 and
 * 17, norm None and 1.3 (`tests/test_cuda_flat_pwa_gen_s0c2.py`) AND a
 * C-level `memcmp` of the fused launch against the three separate ones.
 *
 * Scope of the word "bit-identical", measured: s0c's OWN cross-block atomics
 * (the Q reduce, `grad_ck`'s per-block accumulation of `gck_*_p`) already make
 * it non-reproducible on a large sample — s0c vs ITSELF differs at ne 4000 on
 * the km fixture (dck/Q), and on the production shape too.  So the gate sits
 * at the reproducible shapes (one chunk, one gradient block: 64 events), where
 * s0c vs s0c is exact and s0c2 equals it bit for bit.  The fusion adds no
 * nondeterminism of its own: within a slice the blocks, chunking, fold order
 * and per-feature atomic are s0c's, and the three slices never share a
 * destination address.
 *
 * Deliberately NOT fused: `gsc2_launch_reduce_sum_stream` (Q).  Its reduction
 * order (whole-sample grid-stride, one block per 256 events) differs from the
 * tree kernel's chunked order, so folding it into the same launch would
 * change Q's last bits.  It is 9.9 us and the bitwise gate is worth more.
 * Also unchanged: the `n_features > 32` acc-kernel path — if ANY of the three
 * buffers needs it, the fused launcher falls back to the three separate
 * chunked launches, exactly as s0c runs them.
 *
 * Case-1 copy (kernel policy §1): own `.cu`, own `.so`, own symbols, own
 * wrapper.  `kernels_flat_pwa_gen_s0c.cu` / `_flat_pwa_gen_s0c.py` and every
 * `s0*` / `s0e*` file stay byte-identical.
 *
 * ── why s0c's compute and not a new one ─────────────────────────────────
 *
 * `s0c` was designed against a *uniform coefficient basis*: its denominator
 * is already
 *
 *     BW_w = Σ_i M_wi·F_i,        dQ/dM_wi = 2·Re(dQ/dBW_w·F_i)
 *
 * which IS the flat config's general basis sum
 *
 *     D_e,w = Σ_m gamma[w,m]·coeff[m]·basis_m(m_e) .
 *
 * `s0c` is not specialised to `[1, -m², -iG]`; that is only what its own
 * wrapper happens to fill.  So this file is s0c's kernel with the FILL
 * replaced by the flat config's vocabulary — no new algebra, no new tile.
 *
 * What that buys, measured on the real `jpsi_eta_pi_pi` config:
 *
 *     per event        s0c (plain pwa)     flat pwa_gen
 *     interps              44                 43        <- a wash
 *     matvec terms        520               92 sparse    <- 5.7x cheaper
 *                                          1794 dense     <- 3.5x dearer
 *
 * so the CSR of TRUE non-zeros is mandatory, not an optimisation.
 *
 * ── the memory design this file preserves ──────────────────────────────
 *
 * `s0c` stores NO per-event interpolation cache: the handle holds `mass`,
 * `weight`, `bkg`, the deduplicated numerator cache, and nothing else; the
 * basis interpolation is recomputed per event per iteration into SHARED
 * memory (`gsc2_amp_kernel`).  That is why s0c is 2.87 GB resident where the
 * `s0e` tile is 5.91 GB, and it is the property this backend must inherit.
 *
 * ── the basis rotation (the one non-obvious piece) ─────────────────────
 *
 * s0e's/s0c's device formula is `bw_r += M*Z_i ; bw_i -= M*Z_r`, i.e. it
 * consumes `F = -i·Z`.  Rather than special-case any `basis_kind`, the
 * shared slot table stores
 *
 *     Z_m(e) = i · basis_table[m](m_{e, basis_m_index[m]})
 *
 * so `-i·Z` recovers the basis row for EVERY row uniformly
 * (`-i·(i·basis) == basis`, verified `max|recovered - basis| == 0` over all
 * 69 rows of the real config, and in numpy before the kernel was trusted).
 * `basis_kind` does not appear in this backend's vocabulary at all.
 *
 * ── the two structural dedups (byte-exact keys, never sampled values) ──
 *
 * 1. **uslot** — `(basis row bytes, mass column)`: 69 -> 43 on the real
 *    config.  NOT `basis_kind` (that is a factor-local slot index, so a
 *    kind-keyed fold merges every resonance's running width into one slot:
 *    16 "slots" where the truth is 43).
 * 2. **amp** — `Amp = fa·fl·num` keyed on every `matrix_angle` column of the
 *    wave (p-major, `p·n_wave + k`), the `fl_table` row + `fl_q_index`, and
 *    the `num_table` row + `num_m_index`: 26 -> 13 waves on the real config.
 *    A slot is a WAVE, so the cache carries the projection axis:
 *    `[ne, n_uniq, n_proj]`.
 *
 * Both folds live in the wrapper (`uslot_map`/`amp_slot_map`, imported from
 * `_flat_pwa_gen_s0e` — one implementation) and are proved in numpy.
 *
 * ── sparse gamma rows ──────────────────────────────────────────────────
 *
 * A `(wave, factor)` gamma row selects a handful of `basis_table` rows out
 * of `n_basis`.  The host emits the row's TRUE non-zeros as a CSR
 * (`gs_lo[k]`/`gs_hi[k]` index `nz_m`/`nz_u`), so phase 1 costs `nnz(k)` and
 * phase 5 walks the same entries.  A row does NOT have to be contiguous: a
 * hole is simply absent, so there is no `g == 0` test to get wrong.  The real
 * config has holes in 23 of its 26 waves — 92 entries vs 1794 columns.
 * `create` refuses an all-zero row (D would be 0) and refuses a gamma value
 * that is not exactly 0 or 1 (the m-major phase 5 below factors
 * `Σ_{k∋m} dQ/dD_k` out of the sum, which is exact for a 0/1 selector and
 * only for that).
 *
 * ── the algebra, unchanged from s0c/s0e ────────────────────────────────
 *
 *   D_{e,k}        = Σ_{m∈nz(k)} coeff[m]·basis_m(m_e)      (shared by p)
 *   common_{e,p,k} = Amp_{e,slot(k),p} · conj(D)/|D|²
 *   A_{e,p}        = Σ_k ck_k·common_{e,p,k}
 *   P_e            = Σ_p |A_{e,p}|²
 *   u_{e,k}        = Σ_p dQ/dA_{e,p}·common_{e,p,k}          (= dQ/dck_k)
 *   dQ/dD_{e,k}    = -ck_k·u_{e,k}/D_{e,k}
 *   dQ/dcoeff_m    = 2·Re Σ_{k∋m} dQ/dD_{e,k}·F_m(m_e)
 *
 * `dQ/dck` and `dQ/dD` come from the SAME complex number `u` (the u-merge),
 * and the phase-5 reduction is m-major: because `Z` depends on the basis row
 * only, `Σ_{k∋m} dQ/dD_k` can be summed FIRST and multiplied by `Z_m` once,
 * which keeps the per-event scratch at `n_basis` (never `n_nz`) — a dense
 * 100-wave/69-row config would otherwise need `n_wave·n_basis` per event.
 *
 * The data NLL is the cuda_v5 grouped log-sum, closed on device:
 *
 *   q_e = w_e·(P_e/norm + bkg_e),  S_g = Σ_{e∈g} q_e,  W_g = Σ_{e∈g} w_e
 *   Q   = -Σ_g W_g·log(S_g/W_g),   dQ/dP_e = -(W_g/S_g)·(w_e/norm)
 *   dnorm = Σ_g W_g·Pb_g/(S_g·norm²),  Pb_g = Σ_{e∈g} w_e·P_e
 *
 * with `grp_sum` -> `grp_nll` -> `grp_scale` exactly as s0c runs them
 * (`resolution_size == 1` reproduces the per-event NLL).
 *
 * ── chunking: bounded by batch_size, never by the sample ───────────────
 *
 * Every scratch array is the CONTEXT's, allocated once at `create` from a
 * chunk length `cap` and reused by every chunk of every call — `ne` never
 * enters an allocation.  `compute` walks the sample in chunks passing
 * OFFSET POINTERS plus the chunk length as the kernel's `ne`, so no kernel
 * body knows about batching.  The chunk rule has ONE definition (`gsc2_chunk`)
 * and is ASKED for (`cuda_cap_*`), never mirrored in Python:
 *
 *     cap = batch_size > 0 ? batch_size : 50000
 *     cap = min(cap, 1<<20)
 *     if (rsize > 1) { a = (cap/rsize)*rsize ; cap = a > 0 ? a : rsize }
 *
 * `cap` being a multiple of `rsize` is what stops a log-sum group straddling
 * two chunks (that was a 49 % error once in gen2).
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/complex.h>
#include <cstdio>
#include <cstring>
#include <cmath>

using complex = thrust::complex<double>;

#define GSC2_BLOCK        256       /* threads per block                     */
#define GSC2_DEF_BATCH    50000     /* default events per GPU chunk          */
#define GSC2_MAX_CAP      (1 << 20) /* ceiling on the chunk length           */
#define GSC2_MAX_THREADS  1024      /* cap for the 1:1 entry kernels         */

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = call;                                             \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                               \
        }                                                                    \
    } while (0)

/* ── Catmull-Rom helpers — the repo-wide idiom, verbatim from s0c ─────── */
__device__ double gsc2_cr1d(double pm1, double p0, double p1, double p2,
                           double t) {
    return p0 + 0.5 * t * (
        -pm1 + p1 + t * (2.0*pm1 - 5.0*p0 + 4.0*p1 - p2
                         + t * (-pm1 + 3.0*p0 - 3.0*p1 + p2)));
}

__device__ complex gsc2_interp_c(
    const double* __restrict__ table_real,
    const double* __restrict__ table_imag,
    int type_idx, double x,
    double xmin, double inv_delta, int n_bins
) {
    double diff = (x - xmin) * inv_delta;
    int xbin = max(0, min((int)floor(diff), n_bins - 2));
    double t = max(0.0, min(diff - xbin, 1.0));
    int base = type_idx * n_bins + xbin;
    int end_ = (type_idx + 1) * n_bins - 1;
    int im1 = base > type_idx * n_bins ? base - 1 : base;
    int i2  = base + 2 <= end_ ? base + 2 : base + 1;
    double real_val = gsc2_cr1d(
        table_real[im1], table_real[base], table_real[base + 1], table_real[i2], t);
    double imag_val = gsc2_cr1d(
        table_imag[im1], table_imag[base], table_imag[base + 1], table_imag[i2], t);
    return complex(real_val, imag_val);
}

__device__ double gsc2_interp_r(
    const double* __restrict__ table,
    int type_idx, double x,
    double xmin, double xdelta, int n_bins
) {
    double diff = (x - xmin) / xdelta;
    int xbin = max(0, min((int)floor(diff), n_bins - 2));
    double t = max(0.0, min(diff - xbin, 1.0));
    int base = type_idx * n_bins + xbin;
    int end_ = (type_idx + 1) * n_bins - 1;
    int im1 = base > type_idx * n_bins ? base - 1 : base;
    int i2  = base + 2 <= end_ ? base + 2 : base + 1;
    return gsc2_cr1d(table[im1], table[base], table[base + 1], table[i2], t);
}

/*=============================================================================
 * KERNEL 1 (load-time): the numerator cache, Amp = fa·fl·num.
 *
 * One block per event.  One thread per DEDUPLICATED slot (a slot is a wave,
 * see `amp_slot_map`) emits all `n_proj` of its projections, so the angular
 * basis `ka_a = Π_j cos(angle·angle_k + angle_b)` is evaluated once per
 * (event, basis row) in shared memory and reused by every projection — the
 * gen2/1bw fills recompute it per (wave, projection).
 *
 * Nothing here depends on a fitted parameter (`ck` multiplies the amplitude,
 * `coeff` the denominator), so this runs once per `load_data` and is stored
 * as float2 — the same fp32 floor the sibling flat kernels are gated at.
 *===========================================================================*/
__global__ void gsc2_fill_amp_kernel(
    const double* __restrict__ angle,
    const int* __restrict__ angle_index,
    const double* __restrict__ angle_k,
    const double* __restrict__ angle_b,
    const double* __restrict__ matrix_angle_real,
    const double* __restrict__ matrix_angle_imag,
    const double* __restrict__ momentum,
    const int* __restrict__ fl_q_index,
    const double* __restrict__ fl_table,
    double fl_min, double fl_delta, int fl_bins,
    const double* __restrict__ mass,
    const int* __restrict__ num_m_index,
    const double* __restrict__ num_real,
    const double* __restrict__ num_imag,
    double m_min, double m_inv_delta, int m_bins,
    const int* __restrict__ rep_of_slot,
    int n_wave, int n_proj, int n_decay, int n_res,
    int n_angle_k, int n_angle_total, int n_angle_comp,
    int n_momentum, int n_mass, int n_uniq,
    float2* __restrict__ amp_cache,
    int n_events
) {
    int event_idx = blockIdx.x;
    if (event_idx >= n_events) return;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;

    extern __shared__ double s_dyn_fill[];
    double* s_ka = s_dyn_fill;                       // [n_angle_k]

    for (int a = tid; a < n_angle_k; a += block_sz) {
        int pos = angle_index[a];
        double ka = 1.0;
        for (int comp = 0; comp < n_angle_comp; comp++) {
            int idx = ((size_t)event_idx * n_angle_total + pos) * n_angle_comp
                      + comp;
            ka *= cos(angle[idx] * angle_k[a * n_angle_comp + comp]
                      + angle_b[a * n_angle_comp + comp]);
        }
        s_ka[a] = ka;
    }
    __syncthreads();

    const int PW = n_proj * n_wave;
    for (int s = tid; s < n_uniq; s += block_sz) {
        const int k = rep_of_slot[s];
        // numerator: Π_r num_table[k,r]( m_{e, num_m_index[k,r]} )
        double nr = 1.0, ni = 0.0;
        for (int r = 0; r < n_res; r++) {
            const double mv = mass[(size_t)event_idx * n_mass
                                   + num_m_index[k * n_res + r]];
            complex z = gsc2_interp_c(num_real, num_imag, k * n_res + r, mv,
                                     m_min, m_inv_delta, m_bins);
            const double tr = nr * z.real() - ni * z.imag();
            ni = nr * z.imag() + ni * z.real();
            nr = tr;
        }
        // barrier: Π_d fl_table[k,d]( q_{e, fl_q_index[k,d]} )
        double fl = 1.0;
        for (int d = 0; d < n_decay; d++) {
            const double q = momentum[(size_t)event_idx * n_momentum
                                      + fl_q_index[k * n_decay + d]];
            fl *= gsc2_interp_r(fl_table, k * n_decay + d, q,
                               fl_min, fl_delta, fl_bins);
        }
        const double Fr = fl * nr, Fi = fl * ni;
        for (int p = 0; p < n_proj; p++) {
            double far = 0.0, fai = 0.0;
            for (int a = 0; a < n_angle_k; a++) {
                const double ka = s_ka[a];
                // matrix_angle is p-major: column p·n_wave + k, row a
                const size_t o = (size_t)a * PW + (size_t)(p * n_wave + k);
                far += ka * matrix_angle_real[o];
                fai += ka * matrix_angle_imag[o];
            }
            amp_cache[((size_t)event_idx * n_uniq + s) * n_proj + p] =
                make_float2((float)(Fr * far - Fi * fai),
                            (float)(Fr * fai + Fi * far));
        }
    }
}

/*=============================================================================
 * KERNEL 2 (per iteration): the denominator and the cached amplitude.
 *
 * One block per event (s0c's shape).  Phase 1 interpolates each UNIQUE
 * (basis-row, mass-column) pair ONCE into shared scratch — the slot table is
 * massively redundant (waves share rows; the "1" row is constant): 69 basis
 * rows -> 43 slots on the real config.  Phase 2 is the sparse matvec over
 * each wave's TRUE non-zeros plus the `Amp·conj(D)/|D|²` it feeds.
 *
 * The interpolated values are saved per basis row (`g_interp`) because the
 * gradient kernel runs multi-event scheduled in a separate launch and cannot
 * reach this block's shared memory.  Two waves that select the same basis
 * row write the SAME bits (the value is a function of `(event, slot)` only),
 * so the duplicate write is benign.
 *===========================================================================*/
__global__ void gsc2_amp_kernel(
    const double* __restrict__ mass, int n_mass,
    const double* __restrict__ coeff,          // [n_basis]
    const int* __restrict__ uslot_row,         // [n_uslot] basis-table row
    const int* __restrict__ uslot_col,         // [n_uslot] mass column
    int n_uslot,
    const double* __restrict__ bas_r, const double* __restrict__ bas_i,
    double m_min, double m_inv_delta, int m_bins,
    const int* __restrict__ nz_m,              // [n_nz] basis column
    const int* __restrict__ nz_u,              // [n_nz] -> uslot
    const int* __restrict__ gs_lo, const int* __restrict__ gs_hi,
    const float2* __restrict__ amp_cache,      // [ne, n_uniq, n_proj]
    const int* __restrict__ slot_of_wave,      // [n_wave]
    int n_wave, int n_proj, int n_uniq, int n_basis,
    double* __restrict__ D_real, double* __restrict__ D_imag,
    double* __restrict__ g_interp_real, double* __restrict__ g_interp_imag,
    double* __restrict__ common_real, double* __restrict__ common_imag,
    int n_events
) {
    int event_idx = blockIdx.x;
    if (event_idx >= n_events) return;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;

    extern __shared__ double s_dyn_bw[];
    double* s_mass = s_dyn_bw;
    double* s_us_r = s_dyn_bw + n_mass;             // [n_uslot]
    double* s_us_i = s_dyn_bw + n_mass + n_uslot;

    for (int i = tid; i < n_mass; i += block_sz)
        s_mass[i] = mass[(size_t)event_idx * n_mass + i];
    // s_mass must be COMPLETE before the slot interp reads another thread's
    // column (uslot_col < n_mass): the missing barrier left
    // scheduling-dependent stale reads in the s0b/s0c lineage.
    __syncthreads();

    for (int u = tid; u < n_uslot; u += block_sz) {
        complex z = gsc2_interp_c(bas_r, bas_i, uslot_row[u],
                                 s_mass[uslot_col[u]],
                                 m_min, m_inv_delta, m_bins);
        // Z = i·basis = (-basis_i, basis_r);  the kernel's `-i·Z` is `basis`
        s_us_r[u] = -z.imag();
        s_us_i[u] =  z.real();
    }
    __syncthreads();

    for (int k = tid; k < n_wave; k += block_sz) {
        double dr = 0.0, di = 0.0;
        const int lo = gs_lo[k], hi = gs_hi[k];
        for (int i = lo; i < hi; i++) {
            const double M = coeff[nz_m[i]];
            const int u = nz_u[i];
            dr += M * s_us_i[u];          // -i·Z: Re = +Z_i
            di -= M * s_us_r[u];          //        Im = -Z_r
        }
        const size_t bo = (size_t)event_idx * n_wave + k;
        D_real[bo] = dr;
        D_imag[bo] = di;

        const double nrm = dr * dr + di * di;
        const size_t ao =
            ((size_t)event_idx * n_uniq + slot_of_wave[k]) * n_proj;
        for (int p = 0; p < n_proj; p++) {
            const float2 A = amp_cache[ao + p];
            const size_t o = ((size_t)event_idx * n_proj + p) * n_wave + k;
            common_real[o] = ((double)A.x * dr + (double)A.y * di) / nrm;
            common_imag[o] = ((double)A.y * dr - (double)A.x * di) / nrm;
        }
        for (int i = lo; i < hi; i++) {
            const size_t o = (size_t)event_idx * n_basis + nz_m[i];
            const int u = nz_u[i];
            g_interp_real[o] = s_us_r[u];
            g_interp_imag[o] = s_us_i[u];
        }
    }
}

/*=============================================================================
 * KERNEL 3: PWA forward — A_p, P = Σ_p|A_p|², Q and dQ/dA_p.
 * Verbatim from cuda_v5_pwa_s0c (multi-event scheduling: thread-per-event
 * for small n_base, warp-per-event above).
 *===========================================================================*/
__global__ void gsc2_pwa_reduce_kernel(
    const double* __restrict__ common_r,
    const double* __restrict__ common_i,
    const double* __restrict__ ck_real,
    const double* __restrict__ ck_imag,
    const double* __restrict__ weight,
    const double* __restrict__ bkg,
    int n_wave,               // P·N — entries per event
    int n_proj,               // P
    int n_events,
    int use_norm, double norm,
    double* __restrict__ Q_out,
    double* __restrict__ P_out,
    double* __restrict__ dQ_dA_r, double* __restrict__ dQ_dA_i
) {
    int n_base = n_wave / n_proj;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gstride = gridDim.x * blockDim.x;

    if (n_base <= 32) {
        for (int event_idx = gtid; event_idx < n_events; event_idx += gstride) {
            const double* cr = common_r + (size_t)event_idx * n_wave;
            const double* ci = common_i + (size_t)event_idx * n_wave;
            double P = 0.0;
            for (int p = 0; p < n_proj; p++) {
                double ar = 0.0, ai = 0.0;
                for (int k = 0; k < n_base; k++) {
                    int w = p * n_base + k;
                    double ckr = ck_real[k], cki = ck_imag[k];
                    double cmr = cr[w], cmi = ci[w];
                    ar += ckr * cmr - cki * cmi;
                    ai += ckr * cmi + cki * cmr;
                }
                P += ar * ar + ai * ai;
            }
            P_out[event_idx] = P;
            double wval = weight[event_idx];
            double dQ_dP;
            if (use_norm == 0) {
                Q_out[event_idx] = wval * P;
                dQ_dP = wval;
            } else {
                // cuda_v5: store the raw conj(A_p); the per-group sums
                // Pb/Bb are reduced by gsc2_grp_sum, the per-group
                // NLL/dnorm accumulate on device (gsc2_grp_nll), and
                // gsc2_grp_scale multiplies each event by w_e·c_g.
                dQ_dP = 1.0;
            }
            size_t ab = (size_t)event_idx * n_proj;
            for (int p = 0; p < n_proj; p++) {
                double ar = 0.0, ai = 0.0;
                for (int k = 0; k < n_base; k++) {
                    int w = p * n_base + k;
                    double ckr = ck_real[k], cki = ck_imag[k];
                    double cmr = cr[w], cmi = ci[w];
                    ar += ckr * cmr - cki * cmi;
                    ai += ckr * cmi + cki * cmr;
                }
                dQ_dA_r[ab + p] = dQ_dP * ar;
                dQ_dA_i[ab + p] = -dQ_dP * ai;
            }
        }
        return;
    }

    // ── large n_base: one warp per event, shuffle reduction ─────────
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int e0 = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const int estride = gridDim.x * warps_per_block;
    for (int event_idx = e0; event_idx < n_events; event_idx += estride) {
        const double* cr = common_r + (size_t)event_idx * n_wave;
        const double* ci = common_i + (size_t)event_idx * n_wave;
        double P = 0.0;
        for (int p = 0; p < n_proj; p++) {
            double ar = 0.0, ai = 0.0;
            for (int k = lane; k < n_base; k += 32) {
                int w = p * n_base + k;
                double ckr = ck_real[k], cki = ck_imag[k];
                double cmr = cr[w], cmi = ci[w];
                ar += ckr * cmr - cki * cmi;
                ai += ckr * cmi + cki * cmr;
            }
            for (int off = 16; off > 0; off >>= 1) {
                ar += __shfl_down_sync(0xffffffffu, ar, off);
                ai += __shfl_down_sync(0xffffffffu, ai, off);
            }
            if (lane == 0) P += ar * ar + ai * ai;
        }
        double dQ_dP = 0.0;
        if (lane == 0) {
            P_out[event_idx] = P;
            double wval = weight[event_idx];
            if (use_norm == 0) {
                Q_out[event_idx] = wval * P;
                dQ_dP = wval;
            } else {
                dQ_dP = 1.0;
            }
        }
        dQ_dP = __shfl_sync(0xffffffffu, dQ_dP, 0);
        size_t ab = (size_t)event_idx * n_proj;
        for (int p = 0; p < n_proj; p++) {
            double ar = 0.0, ai = 0.0;
            for (int k = lane; k < n_base; k += 32) {
                int w = p * n_base + k;
                double ckr = ck_real[k], cki = ck_imag[k];
                double cmr = cr[w], cmi = ci[w];
                ar += ckr * cmr - cki * cmi;
                ai += ckr * cmi + cki * cmr;
            }
            for (int off = 16; off > 0; off >>= 1) {
                ar += __shfl_down_sync(0xffffffffu, ar, off);
                ai += __shfl_down_sync(0xffffffffu, ai, off);
            }
            if (lane == 0) {
                dQ_dA_r[ab + p] = dQ_dP * ar;
                dQ_dA_i[ab + p] = -dQ_dP * ai;
            }
        }
    }
}

/*=============================================================================
 * KERNEL 3b/3c/3d: the grouped log-sum — cuda_v5_pwa_s0c's structure with
 * cuda_v5_pwa_s0cw's GROUP-WEIGHTED log, which is the flat config's own
 * objective (`numpy_flat`):
 *
 *     Q_g     = -W_g · log( (Pb_g/norm + Bb_g) / W_g )
 *     dnorm_g =  W_g · Pb_g / (norm · (Pb_g + norm·Bb_g))
 *     c_g     = -W_g / (Pb_g + norm·Bb_g)      (per-event grads: w_e·c_g)
 *
 * with W_g = Σ_{e∈g} w_e.  s0c's plain `-log(Pb/norm + Bb)` is the case
 * W_g == 1, i.e. group-NORMALISED weights; the flat config's `numpy_flat`
 * reference uses the total weight, so this backend must too (measured: the
 * s0c form is off by up to 1.8e+01 on the normed anchor).  A fully
 * zero-weight group contributes nothing.
 *===========================================================================*/
__global__ void gsc2_grp_sum_kernel(
    const double* __restrict__ P_out,
    const double* __restrict__ weight,
    const double* __restrict__ bkg,
    double* __restrict__ gP, double* __restrict__ gB,
    double* __restrict__ gW,
    int n_events, int rsize
) {
    int g = blockIdx.x;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    int e0 = g * rsize;
    int n = (e0 + rsize < n_events) ? rsize : (n_events - e0);
    if (n <= 0) return;

    extern __shared__ double sdyn[];
    double* sP = sdyn;
    double* sB = sdyn + bs;
    double* sW = sdyn + 2 * bs;
    double sp = 0.0, sb = 0.0, sw = 0.0;
    for (int i = tid; i < n; i += bs) {
        int e = e0 + i;
        sp += weight[e] * P_out[e];
        sb += weight[e] * bkg[e];
        sw += weight[e];
    }
    sP[tid] = sp; sB[tid] = sb; sW[tid] = sw;
    __syncthreads();
    for (int st = bs / 2; st > 0; st >>= 1) {
        if (tid < st) {
            sP[tid] += sP[tid + st];
            sB[tid] += sB[tid + st];
            sW[tid] += sW[tid + st];
        }
        __syncthreads();
    }
    if (tid == 0) { gP[g] = sP[0]; gB[g] = sB[0]; gW[g] = sW[0]; }
}

__global__ void gsc2_grp_nll_kernel(
    const double* __restrict__ grpP, const double* __restrict__ grpB,
    const double* __restrict__ grpW,
    int n_groups, double norm,
    double* __restrict__ oQ, double* __restrict__ oDn
) {
    extern __shared__ double sd[];
    double* sq = sd;
    double* sn = sd + blockDim.x;
    int tid = threadIdx.x;
    double q = 0.0, dn = 0.0;
    for (int g = tid; g < n_groups; g += blockDim.x) {
        double Pb = grpP[g], Bb = grpB[g];
        double w = grpW[g];
        if (w > 0.0) {              /* a zero-weight group is inert */
            q += -w * log((Pb / norm + Bb) / w);
            dn += w * Pb / (norm * (Pb + Bb * norm));
        }
    }
    sq[tid] = q; sn[tid] = dn;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) { sq[tid] += sq[tid + st]; sn[tid] += sn[tid + st]; }
        __syncthreads();
    }
    if (tid == 0) { atomicAdd(oQ, sq[0]); atomicAdd(oDn, sn[0]); }
}

__global__ void gsc2_grp_scale_kernel(
    double* __restrict__ dar, double* __restrict__ dai,
    const double* __restrict__ weight,
    const double* __restrict__ gP, const double* __restrict__ gB,
    const double* __restrict__ gW,
    double norm, int rsize, int n_proj, int n_events
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_events * n_proj;
    if (idx >= total) return;
    int e = idx / n_proj;
    int g = e / rsize;
    double Pb = gP[g];
    double c = -gW[g] / (Pb + norm * gB[g]);
    double w = weight[e] * c;
    dar[idx] *= w;
    dai[idx] *= w;
}

/*=============================================================================
 * KERNEL 4: the u-merge — dQ/dck_k, and dQ/dD_{e,k} into `pv`.
 *
 *   u_{e,k}      = Σ_p dQ/dA_{e,p}·common_{e,p,k}     (shared by a wave)
 *   dQ/dck_k     = Σ_e u_{e,k}
 *   pv_{e,k}     = dQ/dD_{e,k} = -ck_k·u_{e,k}/D_{e,k}
 *
 * s0c's kernel computed only the first; the second was recomputed in its
 * gradient kernel because that kernel was per-(wave, slot).  This one's phase
 * 5 is m-major (a basis row may be selected by several waves), so the chain
 * is evaluated ONCE per (e,k) here — the same number of complex divisions
 * s0c performs, just moved to the pass that already holds `u` in registers.
 *===========================================================================*/
__global__ void gsc2_grad_ck_kernel(
    const double* __restrict__ common_r,
    const double* __restrict__ common_i,
    const double* __restrict__ dQ_dA_r,
    const double* __restrict__ dQ_dA_i,
    const double* __restrict__ D_r, const double* __restrict__ D_i,
    const double* __restrict__ ck_real, const double* __restrict__ ck_imag,
    double* __restrict__ grad_ck_real_partial,
    double* __restrict__ grad_ck_imag_partial,
    double* __restrict__ pv_r, double* __restrict__ pv_i,
    int n_wave, int n_proj,
    int n_events
) {
    int n_base = n_wave / n_proj;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gstride = gridDim.x * blockDim.x;

    if (n_base <= 8) {
        for (int event_idx = gtid; event_idx < n_events; event_idx += gstride) {
            const double* cr = common_r + (size_t)event_idx * n_wave;
            const double* ci = common_i + (size_t)event_idx * n_wave;
            const double* dar = dQ_dA_r + (size_t)event_idx * n_proj;
            const double* dai = dQ_dA_i + (size_t)event_idx * n_proj;
            const double* Dr = D_r + (size_t)event_idx * n_base;
            const double* Di = D_i + (size_t)event_idx * n_base;
            size_t base = (size_t)event_idx * n_base;
            for (int k = 0; k < n_base; k++) {
                double sr = 0.0, si = 0.0;
                for (int p = 0; p < n_proj; p++) {
                    int w = p * n_base + k;
                    double a_r = dar[p], a_i = dai[p];
                    double cmr = cr[w], cmi = ci[w];
                    sr += a_r * cmr - a_i * cmi;
                    si += a_r * cmi + a_i * cmr;
                }
                grad_ck_real_partial[base + k] = sr;
                grad_ck_imag_partial[base + k] = si;
                // dQ/dD = -ck·u/D  (1/D = conj(D)/|D|²)
                const double bpn = Dr[k] * Dr[k] + Di[k] * Di[k];
                const double obw_r = Dr[k] / bpn;
                const double obw_i = -Di[k] / bpn;
                const double qr = ck_real[k] * sr - ck_imag[k] * si;
                const double qi = ck_real[k] * si + ck_imag[k] * sr;
                pv_r[base + k] = -(qr * obw_r - qi * obw_i);
                pv_i[base + k] = -(qr * obw_i + qi * obw_r);
            }
        }
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int e0 = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const int estride = gridDim.x * warps_per_block;
    for (int event_idx = e0; event_idx < n_events; event_idx += estride) {
        const double* cr = common_r + (size_t)event_idx * n_wave;
        const double* ci = common_i + (size_t)event_idx * n_wave;
        const double* dar = dQ_dA_r + (size_t)event_idx * n_proj;
        const double* dai = dQ_dA_i + (size_t)event_idx * n_proj;
        const double* Dr = D_r + (size_t)event_idx * n_base;
        const double* Di = D_i + (size_t)event_idx * n_base;
        size_t base = (size_t)event_idx * n_base;
        for (int k = lane; k < n_base; k += 32) {
            double sr = 0.0, si = 0.0;
            for (int p = 0; p < n_proj; p++) {
                int w = p * n_base + k;
                double a_r = dar[p], a_i = dai[p];
                double cmr = cr[w], cmi = ci[w];
                sr += a_r * cmr - a_i * cmi;
                si += a_r * cmi + a_i * cmr;
            }
            grad_ck_real_partial[base + k] = sr;
            grad_ck_imag_partial[base + k] = si;
            const double bpn = Dr[k] * Dr[k] + Di[k] * Di[k];
            const double obw_r = Dr[k] / bpn;
            const double obw_i = -Di[k] / bpn;
            const double qr = ck_real[k] * sr - ck_imag[k] * si;
            const double qi = ck_real[k] * si + ck_imag[k] * sr;
            pv_r[base + k] = -(qr * obw_r - qi * obw_i);
            pv_i[base + k] = -(qr * obw_i + qi * obw_r);
        }
    }
}

/*=============================================================================
 * KERNEL 5: dQ/dcoeff, m-major over the basis rows.
 *
 *   dQ/dcoeff_m = 2·Re Σ_{k∋m} dQ/dD_{e,k}·F_m(m_e),  F_m = -i·Z_m
 *
 * `Z_m` depends only on the basis row, so the wave sum can be taken first
 * and multiplied by `Z_m` once — the reduction axis is `n_basis`, never
 * `n_nz`, which is what keeps a dense 100-wave/69-row config from needing
 * `n_wave·n_basis` per event.  `m_ptr`/`m_waves` is the host-built CSR of
 * "which waves select this basis row".
 *
 * Exactness needs the selector to be 0/1 (a gamma weight `g` would have to
 * multiply each term BEFORE the sum): `create` refuses anything else.
 *===========================================================================*/
__global__ void gsc2_grad_coeff_kernel(
    const double* __restrict__ pv_r, const double* __restrict__ pv_i,
    const double* __restrict__ g_interp_r,
    const double* __restrict__ g_interp_i,
    const int* __restrict__ m_ptr, const int* __restrict__ m_waves,
    int n_wave, int n_basis,
    double* __restrict__ grad_coeff_partial,
    int n_events
) {
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gstride = gridDim.x * blockDim.x;

    if (n_basis <= 32) {
        for (int event_idx = gtid; event_idx < n_events; event_idx += gstride) {
            const double* pvr = pv_r + (size_t)event_idx * n_wave;
            const double* pvi = pv_i + (size_t)event_idx * n_wave;
            const size_t go = (size_t)event_idx * n_basis;
            for (int m = 0; m < n_basis; m++) {
                double sr = 0.0, si = 0.0;
                const int lo = m_ptr[m], hi = m_ptr[m + 1];
                // a row no wave selects is identically zero; its interp was
                // never written, so it must not be read (0 * NaN would poison
                // an unused coefficient)
                if (lo == hi) { grad_coeff_partial[go + m] = 0.0; continue; }
                for (int j = lo; j < hi; j++) {
                    const int k = m_waves[j];
                    sr += pvr[k];
                    si += pvi[k];
                }
                const double zr = g_interp_r[go + m];
                const double zi = g_interp_i[go + m];
                grad_coeff_partial[go + m] = 2.0 * (sr * zi + si * zr);
            }
        }
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int e0 = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const int estride = gridDim.x * warps_per_block;
    for (int event_idx = e0; event_idx < n_events; event_idx += estride) {
        const double* pvr = pv_r + (size_t)event_idx * n_wave;
        const double* pvi = pv_i + (size_t)event_idx * n_wave;
        const size_t go = (size_t)event_idx * n_basis;
        for (int m = lane; m < n_basis; m += 32) {
            double sr = 0.0, si = 0.0;
            const int lo = m_ptr[m], hi = m_ptr[m + 1];
            if (lo == hi) { grad_coeff_partial[go + m] = 0.0; continue; }
            for (int j = lo; j < hi; j++) {
                const int k = m_waves[j];
                sr += pvr[k];
                si += pvi[k];
            }
            const double zr = g_interp_r[go + m];
            const double zi = g_interp_i[go + m];
            grad_coeff_partial[go + m] = 2.0 * (sr * zi + si * zr);
        }
    }
}

/*=============================================================================
 * Reduction kernels (unchanged from the shared implementation)
 *===========================================================================*/
__global__ void gsc2_reduce_sum_kernel(const double* input, double* output,
                                      int n) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < n) ? input[idx] : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, sdata[0]);
}

__global__ void gsc2_reduce_sum_features_acc_kernel(
    const double* input, double* output,
    int n_events, int n_features
) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int feat = blockIdx.x;
    if (feat >= n_features) return;
    double sum = 0.0;
    for (int i = tid; i < n_events; i += blockDim.x) {
        sum += input[feat + (size_t)i * n_features];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) sdata[tid] += sdata[tid + st];
        __syncthreads();
    }
    if (tid == 0) output[feat] += sdata[0];
}

// Coalesced chunked features-reduce (s0b's): one block per event chunk, one
// private shared slot per (feature, thread), one global atomicAdd per feature.
__global__ void gsc2_reduce_sum_features_tree_kernel(
    const double* __restrict__ input,
    double* __restrict__ output,
    int n_events, int n_features, int chunk
) {
    extern __shared__ double sacc[];   // [f·blockDim.x + tid]
    const int bd = blockDim.x;
    const int tid = threadIdx.x;
    const int c0 = blockIdx.x * chunk;
    int c1 = c0 + chunk;
    if (c1 > n_events) c1 = n_events;
    for (int f = 0; f < n_features; f++) sacc[f * bd + tid] = 0.0;

    const long long base = (long long)c0 * n_features;
    const int total = (c1 - c0) * n_features;
    for (int idx = tid; idx < total; idx += bd) {
        int f = (int)(idx % n_features);
        sacc[f * bd + tid] += input[base + idx];
    }
    __syncthreads();
    for (int st = bd / 2; st > 0; st >>= 1) {
        if (tid < st) {
            for (int f = 0; f < n_features; f++)
                sacc[f * bd + tid] += sacc[f * bd + tid + st];
        }
        __syncthreads();
    }
    if (tid < n_features) atomicAdd(&output[tid], sacc[tid * bd]);
}

/* s0c2's ONE change: the three per-chunk feature reductions, ONE launch.
 *
 * The body is `gsc2_reduce_sum_features_tree_kernel`'s, unchanged — same block
 * shape (`blockDim.x`), same chunk rule, same `f*blockDim.x + tid` shared
 * layout, same tree fold order, one `atomicAdd` per feature.  The only addition
 * is that `blockIdx.y` picks the (input, output, n_features) triple, so a
 * `dim3(n_chunks, 3)` grid runs all three buffers side by side.  The three
 * slices write DISJOINT output ranges, so their atomics cannot reorder each
 * other's: every y-slice produces bit-for-bit what the separate launch it
 * replaces produced.
 *
 * Shared memory must be sized for the LARGEST n_features of the three
 * (`max(nf0,nf1,nf2)*128*8`); a slice with fewer features simply uses a prefix.
 */
__global__ void gsc2_reduce_features3_tree_kernel(
    const double* __restrict__ in0, double* __restrict__ out0, int nf0,
    const double* __restrict__ in1, double* __restrict__ out1, int nf1,
    const double* __restrict__ in2, double* __restrict__ out2, int nf2,
    int n_events, int chunk
) {
    const double* __restrict__ input;
    double* __restrict__ output;
    int n_features;
    switch (blockIdx.y) {
        case 1:  input = in1; output = out1; n_features = nf1; break;
        case 2:  input = in2; output = out2; n_features = nf2; break;
        default: input = in0; output = out0; n_features = nf0; break;
    }

    extern __shared__ double sacc[];   // [f·blockDim.x + tid]
    const int bd = blockDim.x;
    const int tid = threadIdx.x;
    const int c0 = blockIdx.x * chunk;
    int c1 = c0 + chunk;
    if (c1 > n_events) c1 = n_events;
    for (int f = 0; f < n_features; f++) sacc[f * bd + tid] = 0.0;

    const long long base = (long long)c0 * n_features;
    const int total = (c1 - c0) * n_features;
    for (int idx = tid; idx < total; idx += bd) {
        int f = (int)(idx % n_features);
        sacc[f * bd + tid] += input[base + idx];
    }
    __syncthreads();
    for (int st = bd / 2; st > 0; st >>= 1) {
        if (tid < st) {
            for (int f = 0; f < n_features; f++)
                sacc[f * bd + tid] += sacc[f * bd + tid + st];
        }
        __syncthreads();
    }
    if (tid < n_features) atomicAdd(&output[tid], sacc[tid * bd]);
}

/*: how many feature-reduction launches this process has issued (fixture only:
 *: it lets `test_fused_reduction_matches_the_three_separate_ones` assert the
 *: LAUNCH COUNT, which is the whole point of the fusion).  The exported
 *: accessors live inside the `extern "C"` block below, with the other reports. */
static int g_gsc2_feature_launches = 0;

void gsc2_launch_reduce_sum_stream(const double* input, double* output, int n,
                                  cudaStream_t stream) {
    int grid_size = (n + GSC2_BLOCK - 1) / GSC2_BLOCK;
    if (grid_size < 1) grid_size = 1;
    gsc2_reduce_sum_kernel<<<grid_size, GSC2_BLOCK,
                            GSC2_BLOCK * sizeof(double), stream>>>(input, output, n);
    CUDA_CHECK(cudaGetLastError());
}

void gsc2_launch_reduce_features_chunked(const double* input, double* output,
                                        int n_events, int n_features,
                                        cudaStream_t stream) {
    if (n_events <= 0) return;
    if (n_features <= 32) {
        int chunk = 1024;
        int n_chunks = (n_events + chunk - 1) / chunk;
        if (n_chunks < 1) n_chunks = 1;
        gsc2_reduce_sum_features_tree_kernel<<<n_chunks, 128,
            n_features * 128 * sizeof(double), stream>>>(
            input, output, n_events, n_features, chunk);
    } else {
        gsc2_reduce_sum_features_acc_kernel<<<n_features, GSC2_BLOCK,
            GSC2_BLOCK * sizeof(double), stream>>>(
            input, output, n_events, n_features);
    }
    CUDA_CHECK(cudaGetLastError());
}

/* The three buffers in ONE tree launch (`grid = dim3(n_chunks, 3)`).
 *
 * The `n_features > 32` case keeps s0c's acc kernel and therefore s0c's THREE
 * launches: its per-feature block grid (`n_features` blocks) has nothing in
 * common with the chunked tree grid, so there is no shared launch to make.
 * The fallback is the identical condition s0c applies, buffer by buffer —
 * `gsc2_launch_reduce_features_chunked` already picks the per-buffer path, so
 * the fallback reproduces s0c exactly when ANY of the three is wide.
 */
void gsc2_launch_reduce_features3(const double* in0, double* out0, int nf0,
                                  const double* in1, double* out1, int nf1,
                                  const double* in2, double* out2, int nf2,
                                  int n_events, cudaStream_t stream) {
    if (n_events <= 0) return;
    if (nf0 > 32 || nf1 > 32 || nf2 > 32) {
        gsc2_launch_reduce_features_chunked(in0, out0, n_events, nf0, stream);
        gsc2_launch_reduce_features_chunked(in1, out1, n_events, nf1, stream);
        gsc2_launch_reduce_features_chunked(in2, out2, n_events, nf2, stream);
        g_gsc2_feature_launches += 3;
        return;
    }
    const int chunk = 1024;
    int n_chunks = (n_events + chunk - 1) / chunk;
    if (n_chunks < 1) n_chunks = 1;
    int nmax = nf0;
    if (nf1 > nmax) nmax = nf1;
    if (nf2 > nmax) nmax = nf2;
    dim3 grid(n_chunks, 3);
    gsc2_reduce_features3_tree_kernel<<<grid, 128,
        (size_t)nmax * 128 * sizeof(double), stream>>>(
        in0, out0, nf0, in1, out1, nf1, in2, out2, nf2, n_events, chunk);
    g_gsc2_feature_launches += 1;
    CUDA_CHECK(cudaGetLastError());
}

/* ── Upload helpers (plain C, before extern "C") ─────────────────────── */
static int* _gsc2_up_i(const int* src, int n) {
    if (n <= 0 || src == NULL) return NULL;
    int* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(int),
                          cudaMemcpyHostToDevice));
    return d;
}

static double* _gsc2_up_d(const double* src, int n) {
    if (n <= 0 || src == NULL) return NULL;
    double* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(double),
                          cudaMemcpyHostToDevice));
    return d;
}

/* The chunk rule.  ONE definition; the wrapper ASKS (`cuda_cap_*`). */
static int gsc2_chunk(int batch_size, int rsize) {
    int cap = (batch_size > 0) ? batch_size : GSC2_DEF_BATCH;
    if (cap > GSC2_MAX_CAP) cap = GSC2_MAX_CAP;
    if (rsize > 1) {
        const int a = (cap / rsize) * rsize;
        cap = (a > 0) ? a : rsize;   /* never 0, never unaligned to a group */
    }
    return cap;
}

extern "C" {

typedef struct {
    /* tables + indices (device) */
    double *fl_table; int *fl_q_index;
    double *num_r, *num_i; int *num_m_index;
    double *bas_r, *bas_i;
    int *uslot_row, *uslot_col; int n_uslot;
    int *nz_m, *nz_u, *gs_lo, *gs_hi; int n_nz;
    int *m_ptr, *m_waves;
    int *angle_index; double *angle_k, *angle_b;
    double *am_r, *am_i;
    int *slot_of_wave, *rep_of_slot; int n_uniq;
    double m_min, m_delta, m_inv_delta, q_min, q_delta;
    int m_bins, q_bins;
    /* dims */
    int n_wave, n_proj, n_res, n_decay, n_basis, n_ab, n_comp;
    int n_mass, n_momentum, n_angle_total;
    int batch_size, rsize, cap, s_ngroups;
    int smem_bytes;
    /* per-chunk scratch (the CONTEXT's: sized by `cap`, never by `ne`) */
    double *D_r, *D_i;              /* cap · n_wave                */
    double *gint_r, *gint_i;        /* cap · n_basis               */
    double *common_r, *common_i;    /* cap · n_proj · n_wave       */
    double *pv_r, *pv_i;            /* cap · n_wave                */
    double *dQdA_r, *dQdA_i;        /* cap · n_proj                */
    double *Q_out, *P_out;          /* cap                         */
    double *grpP, *grpB, *grpW;     /* cap                         */
    double *gck_r_p, *gck_i_p;      /* cap · n_wave                */
    double *gco_p;                  /* cap · n_basis               */
    /* persistent accumulators + parameter staging */
    double *acc, *h_acc;            /* [Q, dnorm, gck_r(N), gck_i(N), gco(NB)] */
    double *p_ck_r, *p_ck_i, *p_coeff;
    double *h_ck_r, *h_ck_i, *h_coeff;
    cudaStream_t stream;
} GSC2Context;

/* ONE SAMPLE: per-EVENT data only.  NO interpolation cache — the basis
   interpolation is recomputed per iteration into shared memory, which is the
   memory property this backend inherits from s0c. */
typedef struct {
    double *mass, *weight, *bkg;
    float2 *amp;                    /* [ne, n_uniq, n_proj] */
    double *h_P;                    /* pinned per-event P staging */
    int ne, n_mass;
} GSC2Handle;

void cuda_free_context_flat_pwa_gen_s0c2(void* vctx);   /* fwd decl */

void* cuda_create_context_flat_pwa_gen_s0c2(
    const double* fl_table, int n_fl,
    const int* fl_q_index, int n_fq,
    const double* num_r, const double* num_i, int n_num,
    const int* num_m_index, int n_nm,
    const double* bas_r, const double* bas_i, int n_bas,
    const int* uslot_row, int n_ur, const int* uslot_col, int n_uc,
    const int* slot_of_basis, int n_sb,
    const int* nz_k, int n_nk, const int* nz_m, int n_nzm,
    const int* nz_u, int n_nzu,
    const int* gs_lo, int n_lo, const int* gs_hi, int n_hi,
    const int* m_ptr, int n_mp, const int* m_waves, int n_mw,
    const int* slot_of_wave, int n_sw, const int* rep_of_slot, int n_rp,
    int n_uniq,
    const int* angle_index, int n_ai,
    const double* angle_k, int n_ak, const double* angle_b, int n_ab_,
    const double* am_r, const double* am_i, int n_am,
    double m_min, double m_delta, int m_bins,
    double q_min, double q_delta, int q_bins,
    int n_wave, int n_proj, int n_res, int n_decay, int n_basis,
    int n_ab, int n_comp, int n_mass, int n_momentum, int n_angle_total,
    int batch_size, int rsize
) {
    if (n_res != 1) {
        /* the wrapper rejects this first (naming the general backend); the
           kernel refuses too rather than treating a product of factors as one
           basis sum */
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0c2: n_res must be 1, got %d — use "
                "cuda_flat_pwa_gen2 for multi-BW chains\n", n_res);
        return NULL;
    }
    if (n_sw != n_wave || n_rp != n_uniq) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0c2: slot_of_wave/rep_of_slot length "
                "mismatch (%d vs n_wave %d, %d vs n_uniq %d)\n",
                n_sw, n_wave, n_rp, n_uniq);
        return NULL;
    }
    if (n_nk != n_nzm || n_nk != n_nzu || n_lo != n_wave || n_hi != n_wave) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0c2: nz_*/gs_* length mismatch "
                "(nz %d/%d/%d, gs %d/%d, n_wave %d)\n",
                n_nk, n_nzm, n_nzu, n_lo, n_hi, n_wave);
        return NULL;
    }
    if (n_mp != n_basis + 1 || n_sb != n_basis) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0c2: m_ptr/slot_of_basis length mismatch "
                "(m_ptr %d vs n_basis+1 %d, slot_of_basis %d vs %d)\n",
                n_mp, n_basis + 1, n_sb, n_basis);
        return NULL;
    }
    /* Both table families are (rows, bins) with `bins` = the shared grid size:
       the numerator and the basis sample the SAME mass grid (`m_min`/
       `m_delta`), and the barrier table its own momentum grid.  Deriving
       either bin count from a truncated size would read a neighbouring row,
       so both are checked to be a whole number of rows. */
    {
        const int n_fl_rows = n_wave * n_decay;
        const int n_num_rows = n_wave * n_res;
        if (n_fl_rows <= 0 || n_fq != n_fl_rows || n_fl % n_fl_rows != 0
                || n_fl / n_fl_rows < 2) {
            fprintf(stderr,
                    "cuda_flat_pwa_gen_s0c2: fl_table is %d entries for %d "
                    "rows (fl_q_index %d) — expected (n_wave·n_decay, bins)\n",
                    n_fl, n_fl_rows, n_fq);
            return NULL;
        }
        if (n_nm != n_num_rows || n_num % n_num_rows != 0
                || n_num / n_num_rows < 2) {
            fprintf(stderr,
                    "cuda_flat_pwa_gen_s0c2: num_table is %d entries for %d "
                    "rows (num_m_index %d)\n", n_num, n_num_rows, n_nm);
            return NULL;
        }
        if (n_bas != n_basis * (n_num / n_num_rows)) {
            fprintf(stderr,
                    "cuda_flat_pwa_gen_s0c2: basis_table is %d entries for "
                    "n_basis %d — expected the numerator's bin count %d\n",
                    n_bas, n_basis, n_num / n_num_rows);
            return NULL;
        }
    }
    if (n_uniq <= 0 || n_ur <= 0 || n_wave <= 0 || n_basis <= 0
            || n_nk <= 0) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0c2: empty shape (n_uniq %d, n_uslot %d, "
                "n_wave %d, n_basis %d, n_nz %d)\n",
                n_uniq, n_ur, n_wave, n_basis, n_nk);
        return NULL;
    }

    GSC2Context* c = (GSC2Context*)calloc(1, sizeof(GSC2Context));
    if (!c) return NULL;
    c->n_wave = n_wave;
    c->n_proj = (n_proj > 0) ? n_proj : 1;
    c->n_res = n_res;
    c->n_decay = n_decay;
    c->n_basis = n_basis;
    c->n_ab = n_ab;
    c->n_comp = n_comp;
    c->n_mass = n_mass;
    c->n_momentum = n_momentum;
    c->n_angle_total = n_angle_total;
    c->n_uniq = n_uniq;
    c->n_uslot = n_ur;
    c->n_nz = n_nk;
    c->rsize = (rsize > 0) ? rsize : 1;
    c->batch_size = batch_size;
    c->cap = gsc2_chunk(batch_size, c->rsize);
    c->s_ngroups = (c->cap + c->rsize - 1) / c->rsize;

    c->fl_table = _gsc2_up_d(fl_table, n_fl);
    c->fl_q_index = _gsc2_up_i(fl_q_index, n_fq);
    c->num_r = _gsc2_up_d(num_r, n_num);
    c->num_i = _gsc2_up_d(num_i, n_num);
    c->num_m_index = _gsc2_up_i(num_m_index, n_nm);
    c->bas_r = _gsc2_up_d(bas_r, n_bas);
    c->bas_i = _gsc2_up_d(bas_i, n_bas);
    c->uslot_row = _gsc2_up_i(uslot_row, n_ur);
    c->uslot_col = _gsc2_up_i(uslot_col, n_uc);
    c->nz_m = _gsc2_up_i(nz_m, n_nzm);
    c->nz_u = _gsc2_up_i(nz_u, n_nzu);
    c->gs_lo = _gsc2_up_i(gs_lo, n_lo);
    c->gs_hi = _gsc2_up_i(gs_hi, n_hi);
    c->m_ptr = _gsc2_up_i(m_ptr, n_mp);
    c->m_waves = _gsc2_up_i(m_waves, n_mw);
    c->slot_of_wave = _gsc2_up_i(slot_of_wave, n_sw);
    c->rep_of_slot = _gsc2_up_i(rep_of_slot, n_rp);
    c->angle_index = _gsc2_up_i(angle_index, n_ai);
    c->angle_k = _gsc2_up_d(angle_k, n_ak);
    c->angle_b = _gsc2_up_d(angle_b, n_ab_);
    c->am_r = _gsc2_up_d(am_r, n_am);
    c->am_i = _gsc2_up_d(am_i, n_am);
    c->m_min = m_min; c->m_delta = m_delta; c->m_bins = m_bins;
    c->m_inv_delta = 1.0 / m_delta;
    c->q_min = q_min; c->q_delta = q_delta; c->q_bins = q_bins;
    (void)slot_of_basis;   /* folded into uslot_col/nz_u host-side */

    /* the per-event shared tile of `gsc2_amp_kernel` */
    {
        const size_t sh = (size_t)(c->n_mass + 2 * c->n_uslot) * sizeof(double);
        int optin = 0;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               0);
        if ((int)sh > optin) {
            fprintf(stderr,
                    "cuda_flat_pwa_gen_s0c2: shared tile %zu B exceeds the "
                    "opt-in limit %d B (n_mass %d, n_uslot %d)\n",
                    sh, optin, c->n_mass, c->n_uslot);
            cuda_free_context_flat_pwa_gen_s0c2(c);
            return NULL;
        }
        c->smem_bytes = (int)sh;
        /* Above the 48 KiB DEFAULT the launch needs the opt-in on the
           FUNCTION; the attribute is per FUNCTION and raise-only monotonic so
           two live contexts cannot fight over it. */
        static size_t opt_in_max = 0;
        if (sh > 48 * 1024 && sh > opt_in_max) {
            cudaError_t e = cudaFuncSetAttribute(
                (const void*)gsc2_amp_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, sh);
            if (e != cudaSuccess) {
                fprintf(stderr,
                        "cuda_flat_pwa_gen_s0c2: cannot opt in to %zu B shared "
                        "memory (%s)\n", sh, cudaGetErrorString(e));
                cuda_free_context_flat_pwa_gen_s0c2(c);
                return NULL;
            }
            opt_in_max = sh;
        }
    }

    /* ── per-chunk scratch: sized from `cap`, never from `ne` ─────────── */
    {
        const size_t bs = (size_t)c->cap;
        const size_t nw = (size_t)c->n_wave, P = (size_t)c->n_proj;
        const size_t NB = (size_t)c->n_basis;
        int ok = 1;
        #define A(p, n) ok = ok && (cudaMalloc(&c->p, (n) * sizeof(double)) \
                                    == cudaSuccess)
        A(D_r, bs * nw); A(D_i, bs * nw);
        A(gint_r, bs * NB); A(gint_i, bs * NB);
        A(common_r, bs * P * nw); A(common_i, bs * P * nw);
        A(pv_r, bs * nw); A(pv_i, bs * nw);
        A(dQdA_r, bs * P); A(dQdA_i, bs * P);
        A(Q_out, bs); A(P_out, bs);
        A(grpP, bs); A(grpB, bs); A(grpW, bs);
        A(gck_r_p, bs * nw); A(gck_i_p, bs * nw);
        A(gco_p, bs * NB);
        #undef A
        if (!ok) {
            fprintf(stderr,
                    "cuda_flat_pwa_gen_s0c2: scratch allocation failed "
                    "(cap %d, n_wave %d, n_basis %d, n_proj %d)\n",
                    c->cap, c->n_wave, c->n_basis, c->n_proj);
            cuda_free_context_flat_pwa_gen_s0c2(c);
            return NULL;
        }
    }

    /* ── persistent accumulators + parameter staging ──────────────────── */
    {
        const size_t N = (size_t)c->n_wave, NB = (size_t)c->n_basis;
        const size_t acc_total = 2 + 2 * N + NB;
        CUDA_CHECK(cudaMalloc(&c->acc, acc_total * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&c->h_acc, acc_total * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c->p_ck_r, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c->p_ck_i, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c->p_coeff, NB * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&c->h_ck_r, N * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&c->h_ck_i, N * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&c->h_coeff, NB * sizeof(double)));
        CUDA_CHECK(cudaStreamCreate(&c->stream));
    }
    return c;
}

void cuda_free_context_flat_pwa_gen_s0c2(void* vctx) {
    GSC2Context* c = (GSC2Context*)vctx;
    if (!c) return;
    #define F(p) if (c->p) cudaFree((void*)c->p)
    #define FH(p) if (c->p) cudaFreeHost(c->p)
    F(fl_table); F(fl_q_index); F(num_r); F(num_i); F(num_m_index);
    F(bas_r); F(bas_i);
    F(uslot_row); F(uslot_col);
    F(nz_m); F(nz_u); F(gs_lo); F(gs_hi);
    F(m_ptr); F(m_waves);
    F(angle_index); F(angle_k); F(angle_b); F(am_r); F(am_i);
    F(slot_of_wave); F(rep_of_slot);
    F(D_r); F(D_i); F(gint_r); F(gint_i);
    F(common_r); F(common_i); F(pv_r); F(pv_i);
    F(dQdA_r); F(dQdA_i); F(Q_out); F(P_out); F(grpP); F(grpB); F(grpW);
    F(gck_r_p); F(gck_i_p); F(gco_p);
    F(acc); F(p_ck_r); F(p_ck_i); F(p_coeff);
    FH(h_acc); FH(h_ck_r); FH(h_ck_i); FH(h_coeff);
    #undef F
    #undef FH
    if (c->stream) cudaStreamDestroy(c->stream);
    free(c);
}

/* Fill the numerator cache for one chunk.  `mom`/`ang`/`mass` are all passed
   CHUNK-OFFSET by the caller: the fill's `event_idx` is chunk-local, exactly
   as in `compute`, so a chunk after the first must not read the sample's
   first events. */
static void gsc2_fill_chunk(GSC2Context* c, GSC2Handle* h,
                           const double* mom, const double* ang,
                           int nb, int base_event) {
    int fill_b = c->n_uniq;
    if (fill_b < GSC2_BLOCK) fill_b = GSC2_BLOCK;
    if (fill_b > GSC2_MAX_THREADS) fill_b = GSC2_MAX_THREADS;
    const size_t sh = (size_t)c->n_ab * sizeof(double);
    gsc2_fill_amp_kernel<<<nb, fill_b, sh, c->stream>>>(
        ang, c->angle_index, c->angle_k, c->angle_b,
        c->am_r, c->am_i,
        mom, c->fl_q_index, c->fl_table, c->q_min, c->q_delta, c->q_bins,
        h->mass + (size_t)base_event * c->n_mass,
        c->num_m_index, c->num_r, c->num_i,
        c->m_min, c->m_inv_delta, c->m_bins,
        c->rep_of_slot,
        c->n_wave, c->n_proj, c->n_decay, c->n_res,
        c->n_ab, c->n_angle_total, c->n_comp,
        c->n_momentum, c->n_mass, c->n_uniq,
        h->amp + (size_t)base_event * c->n_uniq * c->n_proj, nb);
    CUDA_CHECK(cudaGetLastError());
}

void* cuda_load_data_flat_pwa_gen_s0c2(
    void* vctx, const double* mass, const double* mom, const double* ang,
    const double* weight, const double* bkg, int ne
) {
    GSC2Context* c = (GSC2Context*)vctx;
    if (!c) return NULL;
    GSC2Handle* h = (GSC2Handle*)calloc(1, sizeof(GSC2Handle));
    if (!h) return NULL;
    h->ne = ne;
    h->n_mass = c->n_mass;
    /* an EMPTY sample is a legal load: every buffer is absent and `compute`
       returns the identity rather than failing (a zero-byte cudaMalloc is an
       error, so the whole body is guarded). */
    if (ne <= 0) return h;

    h->mass = _gsc2_up_d(mass, (int)((size_t)ne * c->n_mass));
    h->weight = _gsc2_up_d(weight, ne);
    h->bkg = _gsc2_up_d(bkg, ne);
    if (cudaMalloc(&h->amp, (size_t)ne * c->n_uniq * c->n_proj
                            * sizeof(float2)) != cudaSuccess) goto fail;
    CUDA_CHECK(cudaMallocHost(&h->h_P, (size_t)ne * sizeof(double)));

    /* the numerator cache: chunked, so the transient device copies of
       momentum/angle are bounded by `cap`, never by the sample.
       They are fp64 — every flat sibling (gen2/1bw/s0e) uploads the
       kinematics at their native precision for the fill; s0c's own wrapper
       narrows them to float, which costs ~5e-5 on Q against the fp64
       reference (measured: -624783.204285 vs the flat family's
       -624783.229197).  There is no bound to that here: the fill is
       load-time work, and the fp32 floor of this backend is the `amp` cache
       (which the siblings share). */
    {
        int fb = c->cap;
        if (fb > ne) fb = ne;
        if (!h->mass || !h->weight || !h->bkg) goto fail;
        double* gpu_mom = NULL;
        double* gpu_ang = NULL;
        if (cudaMalloc(&gpu_mom,
                       (size_t)fb * c->n_momentum * sizeof(double))
                != cudaSuccess) goto fail;
        if (cudaMalloc(&gpu_ang, (size_t)fb * c->n_angle_total * c->n_comp
                                 * sizeof(double)) != cudaSuccess) {
            cudaFree(gpu_mom);
            goto fail;
        }
        int fail_fill = 0;
        for (int base = 0; base < ne && !fail_fill; base += fb) {
            int nb = (ne - base > fb) ? fb : (ne - base);
            cudaMemcpyAsync(gpu_mom,
                            mom + (size_t)base * c->n_momentum,
                            (size_t)nb * c->n_momentum * sizeof(double),
                            cudaMemcpyHostToDevice, c->stream);
            cudaMemcpyAsync(gpu_ang,
                            ang + (size_t)base * c->n_angle_total * c->n_comp,
                            (size_t)nb * c->n_angle_total * c->n_comp
                            * sizeof(double),
                            cudaMemcpyHostToDevice, c->stream);
            gsc2_fill_chunk(c, h, gpu_mom, gpu_ang, nb, base);
            fail_fill = (cudaGetLastError() != cudaSuccess);
        }
        cudaFree(gpu_mom); cudaFree(gpu_ang);
        if (fail_fill) goto fail;
    }
    return h;

fail:
    if (h->mass) cudaFree(h->mass);
    if (h->weight) cudaFree(h->weight);
    if (h->bkg) cudaFree(h->bkg);
    if (h->amp) cudaFree(h->amp);
    if (h->h_P) cudaFreeHost(h->h_P);
    free(h);
    return NULL;
}

void cuda_free_data_flat_pwa_gen_s0c2(void* vh) {
    GSC2Handle* h = (GSC2Handle*)vh;
    if (!h) return;
    if (h->mass) { cudaFree(h->mass); h->mass = NULL; }
    if (h->weight) { cudaFree(h->weight); h->weight = NULL; }
    if (h->bkg) { cudaFree(h->bkg); h->bkg = NULL; }
    if (h->amp) { cudaFree(h->amp); h->amp = NULL; }
    if (h->h_P) { cudaFreeHost(h->h_P); h->h_P = NULL; }
    free(h);
}

/* One launch per chunk.  Batching needs no kernel logic: the chunk is
   expressed as POINTER OFFSETS into the per-event data plus the chunk length
   as the kernel's `ne`, so `event_idx` is chunk-local by construction. */
/* The amp kernel's block: one thread per (wave or unique slot), rounded to a
 * warp.  ONE definition, used by the launch AND by `kernel_stats` (the
 * occupancy it reports is the one the fit gets). */
static int gsc2_amp_threads(const GSC2Context* c) {
    int amp_t = (c->n_wave > c->n_uslot) ? c->n_wave : c->n_uslot;
    amp_t = ((amp_t + 31) / 32) * 32;
    if (amp_t < 32) amp_t = 32;
    if (amp_t > GSC2_MAX_THREADS) amp_t = GSC2_MAX_THREADS;
    return amp_t;
}

static void gsc2_launch_chunk(GSC2Context* c, GSC2Handle* h, int st, int nb,
                             double norm, int use_norm) {
    const int N = c->n_wave, P = c->n_proj, NB = c->n_basis;
    const int W = P * N;
    int amp_t = gsc2_amp_threads(c);

    /* K2: denominator + common (the basis interp is recomputed here) */
    gsc2_amp_kernel<<<nb, amp_t, (size_t)c->smem_bytes, c->stream>>>(
        h->mass + (size_t)st * c->n_mass, c->n_mass,
        c->p_coeff,
        c->uslot_row, c->uslot_col, c->n_uslot,
        c->bas_r, c->bas_i, c->m_min, c->m_inv_delta, c->m_bins,
        c->nz_m, c->nz_u, c->gs_lo, c->gs_hi,
        h->amp + (size_t)st * c->n_uniq * P, c->slot_of_wave,
        N, P, c->n_uniq, NB,
        c->D_r, c->D_i, c->gint_r, c->gint_i, c->common_r, c->common_i, nb);
    CUDA_CHECK(cudaGetLastError());

    /* K3: PWA forward */
    {
        int grid = (nb + GSC2_BLOCK - 1) / GSC2_BLOCK;
        if (grid > 4096) grid = 4096;
        if (grid < 1) grid = 1;
        gsc2_pwa_reduce_kernel<<<grid, GSC2_BLOCK, 0, c->stream>>>(
            c->common_r, c->common_i,
            c->p_ck_r, c->p_ck_i,
            h->weight + st, h->bkg + st,
            W, P, nb, use_norm, norm,
            c->Q_out, c->P_out, c->dQdA_r, c->dQdA_i);
        CUDA_CHECK(cudaGetLastError());
    }

    /* K3b/3c/3d: the grouped log-sum (data NLL only) */
    if (use_norm == 1) {
        const int rsize = c->rsize;
        const int ngrp = (nb + rsize - 1) / rsize;
        gsc2_grp_sum_kernel<<<ngrp, GSC2_BLOCK, 3 * GSC2_BLOCK * sizeof(double),
                             c->stream>>>(
            c->P_out, h->weight + st, h->bkg + st,
            c->grpP, c->grpB, c->grpW, nb, rsize);
        CUDA_CHECK(cudaGetLastError());
        gsc2_grp_nll_kernel<<<1, GSC2_BLOCK, 2 * GSC2_BLOCK * sizeof(double),
                             c->stream>>>(
            c->grpP, c->grpB, c->grpW, ngrp, norm, c->acc, c->acc + 1);
        CUDA_CHECK(cudaGetLastError());
        gsc2_grp_scale_kernel<<<(nb * P + GSC2_BLOCK - 1) / GSC2_BLOCK,
                               GSC2_BLOCK, 0, c->stream>>>(
            c->dQdA_r, c->dQdA_i, h->weight + st, c->grpP, c->grpB, c->grpW,
            norm, rsize, P, nb);
        CUDA_CHECK(cudaGetLastError());
    }

    /* K4: the u-merge — dQ/dck partials and pv = dQ/dD */
    {
        int grid = (nb + GSC2_BLOCK - 1) / GSC2_BLOCK;
        if (grid > 4096) grid = 4096;
        if (grid < 1) grid = 1;
        gsc2_grad_ck_kernel<<<grid, GSC2_BLOCK, 0, c->stream>>>(
            c->common_r, c->common_i, c->dQdA_r, c->dQdA_i,
            c->D_r, c->D_i, c->p_ck_r, c->p_ck_i,
            c->gck_r_p, c->gck_i_p, c->pv_r, c->pv_i, W, P, nb);
        CUDA_CHECK(cudaGetLastError());
    }

    /* K5: dQ/dcoeff, m-major over the basis rows */
    {
        int grid = (nb + GSC2_BLOCK - 1) / GSC2_BLOCK;
        if (grid > 4096) grid = 4096;
        if (grid < 1) grid = 1;
        gsc2_grad_coeff_kernel<<<grid, GSC2_BLOCK, 0, c->stream>>>(
            c->pv_r, c->pv_i, c->gint_r, c->gint_i,
            c->m_ptr, c->m_waves, N, NB, c->gco_p, nb);
        CUDA_CHECK(cudaGetLastError());
    }
    (void)NB;
}

void cuda_compute_flat_pwa_gen_s0c2(
    void* vctx, void* vh, const double* ck_r, const double* ck_i,
    const double* coeff, double norm, int use_norm,
    double* oQ, double* oDn, double* oP,
    double* og_ck_r, double* og_ck_i, double* og_coeff)
{
    GSC2Context* c = (GSC2Context*)vctx;
    GSC2Handle* h = (GSC2Handle*)vh;
    if (!c || !h) return;
    const int N = c->n_wave, NB = c->n_basis;
    const int off_gck_r = 2, off_gck_i = 2 + N, off_gco = 2 + 2 * N;
    const size_t acc_total = (size_t)2 + 2 * N + NB;

    if (h->ne <= 0) {          /* empty sample: the identity */
        *oQ = 0.0;
        if (oDn) *oDn = 0.0;
        for (int i = 0; i < N; i++) { og_ck_r[i] = 0.0; og_ck_i[i] = 0.0; }
        for (int i = 0; i < NB; i++) og_coeff[i] = 0.0;
        return;
    }

    memcpy(c->h_ck_r, ck_r, (size_t)N * sizeof(double));
    memcpy(c->h_ck_i, ck_i, (size_t)N * sizeof(double));
    memcpy(c->h_coeff, coeff, (size_t)NB * sizeof(double));
    cudaMemcpyAsync(c->p_ck_r, c->h_ck_r, (size_t)N * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    cudaMemcpyAsync(c->p_ck_i, c->h_ck_i, (size_t)N * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    cudaMemcpyAsync(c->p_coeff, c->h_coeff, (size_t)NB * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    /* the accumulators span the WHOLE sample: zeroed once, read once */
    cudaMemsetAsync(c->acc, 0, acc_total * sizeof(double), c->stream);

    const int bs = (c->cap > 0) ? c->cap : 1;
    for (int st = 0; st < h->ne; st += bs) {
        const int nb = (h->ne - st > bs) ? bs : (h->ne - st);
        gsc2_launch_chunk(c, h, st, nb, norm, use_norm);
        CUDA_CHECK(cudaGetLastError());

        if (oP != NULL) {
            cudaMemcpyAsync(h->h_P + st, c->P_out, (size_t)nb * sizeof(double),
                            cudaMemcpyDeviceToHost, c->stream);
        }
        if (use_norm == 0) {
            /* linear phsp path: per-event Q (dnorm is 0 by definition) */
            /* NOT folded into the fused launch below: this reduce's order
             * (whole-sample grid stride) differs from the tree kernel's
             * chunked order, so folding it would change Q's last bits. */
            gsc2_launch_reduce_sum_stream(c->Q_out, c->acc, nb, c->stream);
        }
        /* s0c2: the three feature reductions are ONE launch (s0c: three) */
        gsc2_launch_reduce_features3(c->gck_r_p, c->acc + off_gck_r, N,
                                     c->gck_i_p, c->acc + off_gck_i, N,
                                     c->gco_p, c->acc + off_gco, NB,
                                     nb, c->stream);
    }

    cudaMemcpyAsync(c->h_acc, c->acc, acc_total * sizeof(double),
                    cudaMemcpyDeviceToHost, c->stream);
    CUDA_CHECK(cudaStreamSynchronize(c->stream));
    *oQ = c->h_acc[0];
    if (oDn) *oDn = c->h_acc[1];
    for (int j = 0; j < N; j++) og_ck_r[j] = c->h_acc[off_gck_r + j];
    for (int j = 0; j < N; j++) og_ck_i[j] = c->h_acc[off_gck_i + j];
    for (int j = 0; j < NB; j++) og_coeff[j] = c->h_acc[off_gco + j];
    if (oP != NULL) memcpy(oP, h->h_P, (size_t)h->ne * sizeof(double));
}

/* ── reports: each has exactly ONE definition, in C ─────────────────── */

int cuda_cap_flat_pwa_gen_s0c2(void* vctx) {
    GSC2Context* c = (GSC2Context*)vctx;
    return c ? c->cap : 0;
}

int cuda_ngroups_flat_pwa_gen_s0c2(void* vctx) {
    GSC2Context* c = (GSC2Context*)vctx;
    return c ? c->s_ngroups : 0;
}

int cuda_smem_bytes_flat_pwa_gen_s0c2(void* vctx) {
    GSC2Context* c = (GSC2Context*)vctx;
    return c ? c->smem_bytes : 0;
}

int cuda_get_device_count_flat_pwa_gen_s0c2(void) {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
}

void cuda_get_device_name_flat_pwa_gen_s0c2(char* buf, int len) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    snprintf(buf, len, "%s", p.name);
}

/* ── s0c2 diagnostics: the fusion's own two questions ─────────────────── */

/* How many feature-reduction launches this process has issued.  s0c issues 3
 * per chunk, s0c2 issues 1 (3 only on the `n_features > 32` fallback). */
int cuda_feature_launch_count_flat_pwa_gen_s0c2(void) {
    return g_gsc2_feature_launches;
}

void cuda_reset_feature_launch_count_flat_pwa_gen_s0c2(void) {
    g_gsc2_feature_launches = 0;
}

/* The amp kernel's block for this context (the occupancy report's block). */
int cuda_amp_threads_flat_pwa_gen_s0c2(void* vctx) {
    GSC2Context* c = (GSC2Context*)vctx;
    return c ? gsc2_amp_threads(c) : 0;
}

/* Compiled resource usage of `gsc2_amp_kernel` for THIS shape's dynamic shared
 * memory: ``out = [regs, local_bytes, blocks_per_sm, warps_per_sm,
 * max_threads, static_smem]``.  ``local_bytes > 0`` means it SPILLS. */
int cuda_stats_flat_pwa_gen_s0c2(void* vctx, int* out) {
    GSC2Context* c = (GSC2Context*)vctx;
    if (!c || !out) return 1;
    cudaFuncAttributes a;
    cudaFuncGetAttributes(&a, (const void*)gsc2_amp_kernel);
    const int block = gsc2_amp_threads(c);
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, (const void*)gsc2_amp_kernel, block, (size_t)c->smem_bytes);
    out[0] = a.numRegs;
    out[1] = (int)a.localSizeBytes;
    out[2] = blocks;
    out[3] = blocks * (block / 32);
    out[4] = a.maxThreadsPerBlock;
    out[5] = (int)a.sharedSizeBytes;
    return 0;
}

/* The fixture's direct A/B: run the FUSED launch and the THREE separate tree
 * launches over the same host data and report whether every output is
 * bit-identical.  Returns 1 when equal (and writes the compared count to
 * *out_count), 0 otherwise; a negative value is an error.
 *
 * The separate path is the tree kernel launched per buffer — the exact code
 * s0c runs — so this checks the fusion itself rather than a rewrite of it.
 */
int cuda_check_fused_reduce_flat_pwa_gen_s0c2(
    const double* in0, int nf0, const double* in1, int nf1,
    const double* in2, int nf2, int n_events, int* out_count)
{
    const int nft[3] = {nf0, nf1, nf2};
    const double* hin[3] = {in0, in1, in2};
    double *din[3] = {NULL, NULL, NULL}, *dfu[3] = {NULL, NULL, NULL};
    double *hsep[3] = {NULL, NULL, NULL}, *hfu[3] = {NULL, NULL, NULL};
    int rc = -1;
    int any = 0;
    for (int i = 0; i < 3; i++) {
        if (nft[i] <= 0) continue;
        any = 1;
        if (cudaMalloc(&din[i], (size_t)n_events * nft[i] * sizeof(double))
                != cudaSuccess) goto done;
        if (cudaMemcpy(din[i], hin[i],
                       (size_t)n_events * nft[i] * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) goto done;
        if (cudaMalloc(&dfu[i], (size_t)nft[i] * sizeof(double)) != cudaSuccess)
            goto done;
        if (cudaMemset(dfu[i], 0, (size_t)nft[i] * sizeof(double))
                != cudaSuccess) goto done;
        if (cudaMallocHost(&hsep[i], (size_t)nft[i] * sizeof(double))
                != cudaSuccess) goto done;
        if (cudaMallocHost(&hfu[i], (size_t)nft[i] * sizeof(double))
                != cudaSuccess) goto done;
    }
    if (!any) { rc = 1; goto done; }
    {
        double *dsep[3] = {NULL, NULL, NULL};
        /* three separate tree launches = what s0c does */
        for (int i = 0; i < 3; i++) {
            if (nft[i] <= 0) continue;
            if (cudaMalloc(&dsep[i], (size_t)nft[i] * sizeof(double))
                    != cudaSuccess) {
                for (int j = 0; j < 3; j++) cudaFree(dsep[j]);
                goto done;
            }
            if (cudaMemset(dsep[i], 0, (size_t)nft[i] * sizeof(double))
                    != cudaSuccess) {
                for (int j = 0; j < 3; j++) cudaFree(dsep[j]);
                goto done;
            }
            gsc2_launch_reduce_features_chunked(din[i], dsep[i], n_events,
                                                nft[i], 0);
        }
        /* the fused one */
        gsc2_launch_reduce_features3(din[0], dfu[0], nf0,
                                     din[1], dfu[1], nf1,
                                     din[2], dfu[2], nf2, n_events, 0);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            for (int j = 0; j < 3; j++) cudaFree(dsep[j]);
            goto done;
        }
        int equal = 1, count = 0;
        for (int i = 0; i < 3; i++) {
            if (nft[i] <= 0) continue;
            if (cudaMemcpy(hsep[i], dsep[i], (size_t)nft[i] * sizeof(double),
                           cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(hfu[i], dfu[i], (size_t)nft[i] * sizeof(double),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                for (int j = 0; j < 3; j++) cudaFree(dsep[j]);
                goto done;
            }
            if (memcmp(hsep[i], hfu[i], (size_t)nft[i] * sizeof(double)) != 0)
                equal = 0;
            count += nft[i];
        }
        for (int j = 0; j < 3; j++) cudaFree(dsep[j]);
        if (out_count) *out_count = count;
        rc = equal;
    }
done:
    for (int i = 0; i < 3; i++) {
        if (din[i]) cudaFree(din[i]);
        if (dfu[i]) cudaFree(dfu[i]);
        if (hsep[i]) cudaFreeHost(hsep[i]);
        if (hfu[i]) cudaFreeHost(hfu[i]);
    }
    return rc;
}

} /* extern "C" */
