/**
 * OPTIMIZED CUDA kernels v3 ampcache - Catmull-Rom, sparse matrix_gamma,
 * with a cached minimal-set ANGULAR amplitude.
 *
 * The per-wave amplitude reads a per-event cached angular factor:
 * common_amp_w = Amp_slot[w] / bw_p_w
 *   - Amp_slot (272 unique/event) = fa·fl, pure kinematics, filled once at
 *     load_data (amp_cache_fill_kernel, fp64 math, float2 storage — constant
 *     over the fit so fp32 storage is a fixed per-event factor that does not
 *     move the minimum).  momentum/angle are uploaded ONLY transiently for
 *     this fill and are NOT kept on the device.
 *   - 1/bw_p recomputed every iteration (m0/g0 float) by
 *     amp_cache_amp_kernel (no fallback FA/FL path exists).
 *   - compute_gram builds the Gram matrices from the same cache
 *     (amp_cache_amp_kernel + common_amp_to_groups_kernel), so the kernel
 *     never needs angle/q after the fill.
 * Everything downstream (reduction, time evolution, gradients) is the
 * unchanged sparse code; m0/g0 gradients still flow.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/complex.h>
#include <cstdio>
#include <cmath>

using complex = thrust::complex<double>;

// ── Named constants ──
#define BLOCK_SIZE      256     // threads per block (RTX 3070 Ti optimal)
#define N_SCALAR        6       // scalar gradient count
#define DEFAULT_BATCH_SIZE 50000 // default events per GPU batch

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
        } \
    } while(0)

// ── Catmull-Rom interpolation helpers ──────────────────────────────────
// Uniform Catmull-Rom: p(t) = 0.5 * (
//     (2*p1) + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t² + (-p0+3*p1-3*p2+p3)*t³ )
// Uses 4 control points: p[-1]=p0, p[0]=p1, p[1]=p2, p[2]=p3
// Clamps edge bins to maintain C¹ at boundaries.

__device__ double catmull_rom_1d(
    double pm1, double p0, double p1, double p2, double t
) {
    return p0 + 0.5 * t * (
        -pm1 + p1 + t * (2.0*pm1 - 5.0*p0 + 4.0*p1 - p2 + t * (-pm1 + 3.0*p0 - 3.0*p1 + p2))
    );
}

// Complex Catmull-Rom for gamma table (uses inv_delta)
__device__ complex interp_complex_device(
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
    // Need 4 control points: pm1 at base-1, p0 at base, p1 at base+1, p2 at base+2
    // Clamp to [type_start, end_], duplicate edge values
    int im1 = base > type_idx * n_bins ? base - 1 : base;
    int i2  = base + 2 <= end_ ? base + 2 : base + 1;
    double real_val = catmull_rom_1d(
        table_real[im1], table_real[base], table_real[base + 1], table_real[i2], t);
    double imag_val = catmull_rom_1d(
        table_imag[im1], table_imag[base], table_imag[base + 1], table_imag[i2], t);
    return complex(real_val, imag_val);
}

// Real Catmull-Rom for FL factor
__device__ double interp_real_device(
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
    return catmull_rom_1d(
        table[im1], table[base], table[base + 1], table[i2], t);
}

// FP32 real Catmull-Rom for FL factor (uses inv_delta = 1/delta)
__device__ float interp_real_device_f32(
    const float* __restrict__ table,
    int type_idx, float x,
    float xmin, float inv_delta, int n_bins
) {
    float diff = (x - xmin) * inv_delta;
    int xbin = max(0, min((int)floor(diff), n_bins - 2));
    float t = max(0.0f, min(diff - (float)xbin, 1.0f));
    int base = type_idx * n_bins + xbin;
    int end_ = (type_idx + 1) * n_bins - 1;
    int im1 = base > type_idx * n_bins ? base - 1 : base;
    int i2  = base + 2 <= end_ ? base + 2 : base + 1;
    return catmull_rom_1d(
        table[im1], table[base], table[base + 1], table[i2], (double)t);
}

//=============================================================================
// KERNEL 1: g_bw computation (parallelized within each event)
//=============================================================================
// Each block handles one event.
// Phase 1: Threads cooperatively compute g[i] for all gamma_rows
// Phase 2: Threads cooperatively compute g_bw[j] = sum_i g[i] * matrix_gamma[i,j]
//          using coalesced global reads of matrix_gamma
//=============================================================================
__global__ void compute_g_bw_kernel(
    const double* __restrict__ mass,
    const double* __restrict__ g0,
    const int* __restrict__ g0_index,
    const int* __restrict__ g0_mass_index,
    const int* __restrict__ gamma_col_idx,
    const double* __restrict__ gamma_table_real,
    const double* __restrict__ gamma_table_imag,
    double gamma_min, double gamma_inv_delta,
    int n_gamma_rows, int n_unique_bw, int n_mass, int gamma_table_bins,
    double* __restrict__ g_interp_real,
    double* __restrict__ g_interp_imag,
    double* __restrict__ g_bw_real,
    double* __restrict__ g_bw_imag,
    int n_events
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;

    // Shared memory: [s_g_real (ng), s_g_imag (ng), s_gbw_r (nu), s_gbw_i (nu)]
    extern __shared__ double s_dyn_gbw[];
    double* s_g_real = s_dyn_gbw;
    double* s_g_imag = s_dyn_gbw + n_gamma_rows;
    double* s_gbw_r  = s_dyn_gbw + 2 * n_gamma_rows;
    double* s_gbw_i  = s_dyn_gbw + 2 * n_gamma_rows + n_unique_bw;

    // Zero shared g_bw scratch
    for (int i = tid; i < n_unique_bw; i += block_sz) {
        s_gbw_r[i] = 0.0;
        s_gbw_i[i] = 0.0;
    }
    __syncthreads();

    // Phase 1: CR + g0 multiply (loop over gamma rows with stride)
    for (int gamma_idx = tid; gamma_idx < n_gamma_rows; gamma_idx += block_sz) {
        int g0_idx = g0_index[gamma_idx];
        double g0_val = g0[g0_idx];
        double mass_val = mass[event_idx * n_mass + g0_mass_index[gamma_idx]];

        complex g_interp = interp_complex_device(
            gamma_table_real, gamma_table_imag,
            g0_idx, mass_val,
            gamma_min, gamma_inv_delta, gamma_table_bins
        );

        g_interp_real[event_idx * n_gamma_rows + gamma_idx] = g_interp.real();
        g_interp_imag[event_idx * n_gamma_rows + gamma_idx] = g_interp.imag();

        complex g_val = g0_val * g_interp;
        s_g_real[gamma_idx] = g_val.real();
        s_g_imag[gamma_idx] = g_val.imag();
    }
    __syncthreads();

    // Phase 2: Sparse scatter (each gamma row to its column)
    for (int gamma_idx = tid; gamma_idx < n_gamma_rows; gamma_idx += block_sz) {
        int col = gamma_col_idx[gamma_idx];
        atomicAdd(&s_gbw_r[col], s_g_real[gamma_idx]);
        atomicAdd(&s_gbw_i[col], s_g_imag[gamma_idx]);
    }
    __syncthreads();

    // Write accumulated g_bw to global
    for (int c = tid; c < n_unique_bw; c += block_sz) {
        g_bw_real[event_idx * n_unique_bw + c] = s_gbw_r[c];
        g_bw_imag[event_idx * n_unique_bw + c] = s_gbw_i[c];
    }
}

//=============================================================================
// KERNEL 2b: Gram‑matrix reduction — batch of A0/A1 → Mpp/Mmm/Mpm
//=============================================================================
// Grid of (ng, ng) blocks × 1 thread each.
//=============================================================================
__global__ void gram_reduce_kernel_v3(
    const double* __restrict__ A0_real, const double* __restrict__ A0_imag,
    const double* __restrict__ A1_real, const double* __restrict__ A1_imag,
    const double* __restrict__ weight,
    int n_events, int ng,
    double* __restrict__ Mpp_r, double* __restrict__ Mpp_i,
    double* __restrict__ Mmm_r, double* __restrict__ Mmm_i,
    double* __restrict__ Mpm_r, double* __restrict__ Mpm_i
) {
    int gi = blockIdx.x;
    int gj = blockIdx.y;
    if (gi >= ng || gj >= ng) return;

    double sum_pp_r = 0.0, sum_pp_i = 0.0;
    double sum_mm_r = 0.0, sum_mm_i = 0.0;
    double sum_pm_r = 0.0, sum_pm_i = 0.0;

    for (int e = 0; e < n_events; e++) {
        double w = weight[e];
        int base = e * ng;
        double a0ri = A0_real[base + gi], a0ii = A0_imag[base + gi];
        double a0rj = A0_real[base + gj], a0ij = A0_imag[base + gj];
        sum_pp_r += w * (a0ri * a0rj + a0ii * a0ij);
        sum_pp_i += w * (a0ri * a0ij - a0ii * a0rj);

        double a1ri = A1_real[base + gi], a1ii = A1_imag[base + gi];
        double a1rj = A1_real[base + gj], a1ij = A1_imag[base + gj];
        sum_mm_r += w * (a1ri * a1rj + a1ii * a1ij);
        sum_mm_i += w * (a1ri * a1ij - a1ii * a1rj);

        sum_pm_r += w * (a0ri * a1rj + a0ii * a1ij);
        sum_pm_i += w * (a0ri * a1ij - a0ii * a1rj);
    }

    Mpp_r[gi * ng + gj] = sum_pp_r;
    Mpp_i[gi * ng + gj] = sum_pp_i;
    Mmm_r[gi * ng + gj] = sum_mm_r;
    Mmm_i[gi * ng + gj] = sum_mm_i;
    Mpm_r[gi * ng + gj] = sum_pm_r;
    Mpm_i[gi * ng + gj] = sum_pm_i;
}

//=============================================================================
// KERNEL 2a0: common_amp → Gram A0/A1 groups (cached angular amplitudes)
//=============================================================================
// Each block handles one event; thread t reduces the 4 permutation blocks:
//   A0[e,g] = Σ_p common_amp[e, p·ng + g]          (B0,  waves [0, n))
//   A1[e,g] = Σ_p common_amp[e, n + p·ng + g]      (B0bar, waves [n, 2n))
// where n = n_wave/2 = 4·ng (identical-particle permutations are separate
// blocks of ng).  Same grouping as the original gram_common_kernel_v3.
//=============================================================================
__global__ void common_amp_to_groups_kernel(
    const double* __restrict__ common_r,
    const double* __restrict__ common_i,
    double* __restrict__ A0_r, double* __restrict__ A0_i,
    double* __restrict__ A1_r, double* __restrict__ A1_i,
    int n_wave, int n_events
) {
    int e = blockIdx.x;
    int t = threadIdx.x;
    int ng = n_wave / 8;
    int n = n_wave / 2;                 // 4·ng
    if (t >= ng) return;
    double r0 = 0.0, i0 = 0.0, r1 = 0.0, i1 = 0.0;
    const double* cr = common_r + (size_t)e * n_wave;
    const double* ci = common_i + (size_t)e * n_wave;
    #pragma unroll
    for (int p = 0; p < 4; p++) {
        int w0 = p * ng + t;                       // B0 perm block p
        r0 += cr[w0]; i0 += ci[w0];
        int w1 = n + p * ng + t;                   // B0bar perm block p
        r1 += cr[w1]; i1 += ci[w1];
    }
    size_t ab = (size_t)e * ng + t;
    A0_r[ab] = r0; A0_i[ab] = i0;
    A1_r[ab] = r1; A1_i[ab] = i1;
}

//=============================================================================
// KERNEL 2a-cache: amp_cache_fill — one-time per-event angular-amplitude cache
//=============================================================================
// Each block handles one event.  Caches the minimal set of angular
// amplitudes Amp_s = fa·fl per cache slot s (272 for the current configs),
// where fa = Σ_k ka_k·matrix_angle[k, rep_of_slot[s]] and
// fl = Π_d form_factor(q).  Pure per-event kinematics → filled once at
// load_data; the fit then recomputes only the BW propagator per iteration.
// fp64 throughout (fill is a one-time cost).
//=============================================================================
__global__ void amp_cache_fill_kernel(
    const float* __restrict__ angle,
    const int* __restrict__ angle_index,
    const double* __restrict__ angle_k,
    const double* __restrict__ angle_b,
    const double* __restrict__ matrix_angle_real,
    const double* __restrict__ matrix_angle_imag,
    const float* __restrict__ momentum,
    const int* __restrict__ fl_type,
    const int* __restrict__ fl_q_index,
    const int* __restrict__ fl_order,
    const double* __restrict__ fl_table,
    double fl_min, double fl_delta,
    const int* __restrict__ rep_of_slot,
    int n_wave, int n_angle_k, int n_angle_total, int n_angle_comp,
    int n_decay, int n_momentum, int fl_table_bins,
    int n_uniq,
    float2* __restrict__ amp_cache,
    int n_events
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;

    extern __shared__ double s_dyn_amp[];
    double* s_ka = s_dyn_amp;                 // [n_angle_k]

    // Phase 1: angular basis products ka_k (fp64, as in gram_common_kernel_v3)
    for (int k_idx = tid; k_idx < n_angle_k; k_idx += block_sz) {
        int angle_pos = angle_index[k_idx];
        double ka = 1.0;
        for (int comp = 0; comp < n_angle_comp; comp++) {
            int idx = event_idx * n_angle_total * n_angle_comp
                      + angle_pos * n_angle_comp + comp;
            ka *= cos((double)angle[idx] * angle_k[k_idx * n_angle_comp + comp]
                      + angle_b[k_idx * n_angle_comp + comp]);
        }
        s_ka[k_idx] = ka;
    }
    __syncthreads();

    // Phase 2: per cache slot, evaluate Amp at the representative wave
    for (int s = tid; s < n_uniq; s += block_sz) {
        int w = rep_of_slot[s];
        // fa = Σ_k s_ka[k] · matrix_angle[k, w]
        double fr = 0.0, fi = 0.0;
        for (int k = 0; k < n_angle_k; k++) {
            int idx = k * n_wave + w;
            fr += s_ka[k] * matrix_angle_real[idx];
            fi += s_ka[k] * matrix_angle_imag[idx];
        }
        // fl = Π_d form_factor(q at fl_q_index[fl_order[w,d]])
        double fl = 1.0;
        for (int d = 0; d < n_decay; d++) {
            int fl_idx = fl_order[w * n_decay + d];
            double q = (double)momentum[event_idx * n_momentum + fl_q_index[fl_idx]];
            fl *= interp_real_device(fl_table, fl_type[fl_idx],
                                     q, fl_min, fl_delta, fl_table_bins);
        }
        // Stored as float2: the value is constant over the whole fit, so the
        // ~1e-7 rounding is a fixed per-event factor and does not move the
        // minimum (halves the cache memory vs fp64).
        amp_cache[event_idx * n_uniq + s] = make_float2((float)(fr * fl),
                                                        (float)(fi * fl));
    }
}

//=============================================================================
// KERNEL 2a-cache: cached forward — BW propagator × cached angular amp
//=============================================================================
// Each block handles one event, 1:1 thread→wave.  Same as
// compute_bw_amp_kernel but the FL/FA part is replaced by a single read of
// the cached angular amplitude (float2, promoted on read) via slot_of_wave[wave]
// — so per fit
// iteration only the BW denominator product is recomputed (m0/g0 float).
//=============================================================================
__global__ void amp_cache_amp_kernel(
    const double* __restrict__ mass,
    const int* __restrict__ m0_index,
    const int* __restrict__ mass_index,
    const int* __restrict__ bw_order,
    const double* __restrict__ g_bw_real,
    const double* __restrict__ g_bw_imag,
    int n_wave, int n_res, int n_unique_bw,
    int n_mass,
    const double* __restrict__ m0,
    const float2* __restrict__ amp_cache,
    const int* __restrict__ slot_of_wave,
    int n_uniq,
    double* __restrict__ bw_p_real, double* __restrict__ bw_p_imag,
    double* __restrict__ common_amp_factor_real,
    double* __restrict__ common_amp_factor_imag,
    double* __restrict__ bw_dom_real, double* __restrict__ bw_dom_imag,
    int n_events
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int wave_idx = tid;

    extern __shared__ double s_dyn_bw2[];
    double* s_mass     = s_dyn_bw2;
    double* s_gbw_real = s_dyn_bw2 + n_mass;
    double* s_gbw_imag = s_dyn_bw2 + n_mass + n_unique_bw;

    for (int i = tid; i < n_mass; i += blockDim.x)
        s_mass[i] = mass[event_idx * n_mass + i];
    for (int bw_idx = tid; bw_idx < n_unique_bw; bw_idx += blockDim.x) {
        int base = event_idx * n_unique_bw + bw_idx;
        s_gbw_real[bw_idx] = g_bw_real[base];
        s_gbw_imag[bw_idx] = g_bw_imag[base];
    }
    __syncthreads();

    if (wave_idx >= n_wave) return;

    // BW product (identical to compute_bw_amp_kernel)
    double bw_p_r = 1.0, bw_p_i = 0.0;
    #pragma unroll
    for (int res_idx = 0; res_idx < n_res; res_idx++) {
        int bw_idx = bw_order[wave_idx * n_res + res_idx];
        double m0v = m0[m0_index[bw_idx]];
        double mv = s_mass[mass_index[bw_idx]];
        double m0s = m0v * m0v, ms = mv * mv;
        double dr = m0s - ms + m0v * s_gbw_imag[bw_idx];
        double di = -m0v * s_gbw_real[bw_idx];
        double nr = bw_p_r * dr - bw_p_i * di;
        double ni = bw_p_r * di + bw_p_i * dr;
        bw_p_r = nr; bw_p_i = ni;
        bw_dom_real[event_idx * n_unique_bw + bw_idx] = dr;
        bw_dom_imag[event_idx * n_unique_bw + bw_idx] = di;
    }
    bw_p_real[event_idx * n_wave + wave_idx] = bw_p_r;
    bw_p_imag[event_idx * n_wave + wave_idx] = bw_p_i;

    // common_amp = cached_Amp[slot] / bw_p  (fp32 cache promoted to fp64)
    float2 A = amp_cache[event_idx * n_uniq + slot_of_wave[wave_idx]];
    double nrm = bw_p_r * bw_p_r + bw_p_i * bw_p_i;
    common_amp_factor_real[event_idx * n_wave + wave_idx] =
        ((double)A.x * bw_p_r + (double)A.y * bw_p_i) / nrm;
    common_amp_factor_imag[event_idx * n_wave + wave_idx] =
        ((double)A.y * bw_p_r - (double)A.x * bw_p_i) / nrm;
}

//=============================================================================
// KERNEL 2b: Amplitude reduction + time evolution + NLL
//=============================================================================
// Each block handles one event.
// Reads common_amp (from global) + ck → tree-reduces ap/am → time evolution → Q/P
// No ka_prod or per-wave BW/FL computation — focused on reduction + scalar math.
//=============================================================================
__global__ void amp_reduce_time_kernel(
    const double* __restrict__ common_amp_factor_real,
    const double* __restrict__ common_amp_factor_imag,
    const double* __restrict__ ck_real,
    const double* __restrict__ ck_imag,
    const double* __restrict__ frac,
    const double* __restrict__ time,
    const double* __restrict__ weight,
    const double* __restrict__ bkg,
    double Gamma, double Delta_Gamma, double Delta_m,
    double A_p, double poq_rho, double pop_phi,
    double* __restrict__ Q_out,
    double* __restrict__ P_out,
    double* __restrict__ pap_real, double* __restrict__ pap_imag,
    double* __restrict__ pam_real, double* __restrict__ pam_imag,
    double* __restrict__ gp_real, double* __restrict__ gp_imag,
    double* __restrict__ gm_real, double* __restrict__ gm_imag,
    double* __restrict__ poq_real, double* __restrict__ poq_imag,
    double* __restrict__ ap_real, double* __restrict__ ap_imag,
    double* __restrict__ am_real, double* __restrict__ am_imag,
    double* __restrict__ dQ_dP,
    int n_wave, int n_events, int use_norm, double norm
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;
    int n_wave_half = n_wave / 2;

    double ap_sum_r = 0.0, ap_sum_i = 0.0;
    double am_sum_r = 0.0, am_sum_i = 0.0;

    for (int i = tid; i < n_wave_half; i += block_sz) {
        complex ck_i(ck_real[i], ck_imag[i]);
        complex common_i(
            common_amp_factor_real[event_idx * n_wave + i],
            common_amp_factor_imag[event_idx * n_wave + i]);
        complex prod_i = ck_i * common_i;
        ap_sum_r += prod_i.real();
        ap_sum_i += prod_i.imag();

        complex ck_j(ck_real[n_wave_half + i], ck_imag[n_wave_half + i]);
        complex common_j(
            common_amp_factor_real[event_idx * n_wave + n_wave_half + i],
            common_amp_factor_imag[event_idx * n_wave + n_wave_half + i]);
        complex prod_j = ck_j * common_j;
        am_sum_r += prod_j.real();
        am_sum_i += prod_j.imag();
    }

    // Tree reduction
    __shared__ double s_ap_r[256];
    __shared__ double s_ap_i[256];
    __shared__ double s_am_r[256];
    __shared__ double s_am_i[256];

    s_ap_r[tid] = ap_sum_r;
    s_ap_i[tid] = ap_sum_i;
    s_am_r[tid] = am_sum_r;
    s_am_i[tid] = am_sum_i;
    __syncthreads();

    for (int s = block_sz / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_ap_r[tid] += s_ap_r[tid + s];
            s_ap_i[tid] += s_ap_i[tid + s];
            s_am_r[tid] += s_am_r[tid + s];
            s_am_i[tid] += s_am_i[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        complex ap(s_ap_r[0], s_ap_i[0]);
        complex am(s_am_r[0], s_am_i[0]);
        ap_real[event_idx] = ap.real();
        ap_imag[event_idx] = ap.imag();
        am_real[event_idx] = am.real();
        am_imag[event_idx] = am.imag();

        double t = time[event_idx];
        complex i_const(0.0, 1.0);
        complex eL = exp(-i_const * t * complex(-Delta_m / 2, -(Gamma + Delta_Gamma / 2) / 2));
        complex eH = exp(-i_const * t * complex(Delta_m / 2, -(Gamma - Delta_Gamma / 2) / 2));
        complex gp = (eL + eH) / 2.0;
        complex gm = (eL - eH) / 2.0;

        gp_real[event_idx] = gp.real();
        gp_imag[event_idx] = gp.imag();
        gm_real[event_idx] = gm.real();
        gm_imag[event_idx] = gm.imag();

        complex poq = poq_rho * exp(i_const * pop_phi);
        poq_real[event_idx] = poq.real();
        poq_imag[event_idx] = poq.imag();

        complex pap = gp * ap + gm * poq * am;
        complex pam_val = (gm / poq) * ap + gp * am;

        pap_real[event_idx] = pap.real();
        pap_imag[event_idx] = pap.imag();
        pam_real[event_idx] = pam_val.real();
        pam_imag[event_idx] = pam_val.imag();

        double pb = thrust::norm(pap);
        double pbbar = thrust::norm(pam_val);
        double frac_val = frac[event_idx];
        double P = frac_val * pb * (1.0 - A_p) + (1.0 - frac_val) * pbbar * (1.0 + A_p);
        P_out[event_idx] = P;

        double weight_val = weight[event_idx];
        double bkg_val = bkg[event_idx];

        if (use_norm == 0) {
            Q_out[event_idx] = weight_val * P;
            dQ_dP[event_idx] = weight_val;
        } else {
            Q_out[event_idx] = -weight_val * log(P / norm + bkg_val);
            dQ_dP[event_idx] = -weight_val / (P + bkg_val * norm);
        }
    }
}

//=============================================================================
// KERNEL 3a: ck gradient kernel — Wirtinger dQ/dck[i] = dQ_da * common_amp[i]
//=============================================================================
// Each block handles one event.  Event-constant dQ_dap/dQ_dam computed once
// in shared memory (instead of every thread recomputing the same values).
//=============================================================================
__global__ void grad_ck_kernel(
    const double* __restrict__ common_amp_factor_real,
    const double* __restrict__ common_amp_factor_imag,
    const double* __restrict__ pap_real, const double* __restrict__ pap_imag,
    const double* __restrict__ pam_real, const double* __restrict__ pam_imag,
    const double* __restrict__ gp_real, const double* __restrict__ gp_imag,
    const double* __restrict__ gm_real, const double* __restrict__ gm_imag,
    const double* __restrict__ poq_real, const double* __restrict__ poq_imag,
    const double* __restrict__ dQ_dP,
    const double* __restrict__ frac,
    double A_p,
    int n_wave,
    double* __restrict__ grad_ck_real_partial,
    double* __restrict__ grad_ck_imag_partial,
    int n_events
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;

    // Thread 0 computes event-constants, broadcasts via shared
    __shared__ double s_dQ_dap_r, s_dQ_dap_i, s_dQ_dam_r, s_dQ_dam_i;
    if (tid == 0) {
        complex pap(pap_real[event_idx], pap_imag[event_idx]);
        complex pam(pam_real[event_idx], pam_imag[event_idx]);
        complex gp(gp_real[event_idx], gp_imag[event_idx]);
        complex gm(gm_real[event_idx], gm_imag[event_idx]);
        complex poq(poq_real[event_idx], poq_imag[event_idx]);
        double dQ_dP_val = dQ_dP[event_idx];
        double frac_val = frac[event_idx];

        double dQ_dpb = dQ_dP_val * frac_val * (1.0 - A_p);
        double dQ_dpbbar = dQ_dP_val * (1.0 - frac_val) * (1.0 + A_p);

        complex dQ_dap = dQ_dpb * conj(pap) * gp
                       + dQ_dpbbar * conj(pam) * (gm / poq);
        complex dQ_dam = dQ_dpb * conj(pap) * gm * poq
                       + dQ_dpbbar * conj(pam) * gp;

        s_dQ_dap_r = dQ_dap.real();
        s_dQ_dap_i = dQ_dap.imag();
        s_dQ_dam_r = dQ_dam.real();
        s_dQ_dam_i = dQ_dam.imag();
    }
    __syncthreads();

    double dQ_dap_r = s_dQ_dap_r, dQ_dap_i = s_dQ_dap_i;
    double dQ_dam_r = s_dQ_dam_r, dQ_dam_i = s_dQ_dam_i;

    int n_wave_half = n_wave / 2;
    for (int i = tid; i < n_wave_half; i += block_sz) {
        double cr = common_amp_factor_real[event_idx * n_wave + i];
        double ci = common_amp_factor_imag[event_idx * n_wave + i];
        // dQ_dap * common_amp (complex multiply)
        double gr = dQ_dap_r * cr - dQ_dap_i * ci;
        double gi = dQ_dap_r * ci + dQ_dap_i * cr;
        grad_ck_real_partial[event_idx * n_wave + i] = gr;
        grad_ck_imag_partial[event_idx * n_wave + i] = gi;

        double c2r = common_amp_factor_real[event_idx * n_wave + n_wave_half + i];
        double c2i = common_amp_factor_imag[event_idx * n_wave + n_wave_half + i];
        double g2r = dQ_dam_r * c2r - dQ_dam_i * c2i;
        double g2i = dQ_dam_r * c2i + dQ_dam_i * c2r;
        grad_ck_real_partial[event_idx * n_wave + n_wave_half + i] = g2r;
        grad_ck_imag_partial[event_idx * n_wave + n_wave_half + i] = g2i;
    }
}

//=============================================================================
// KERNEL 3b: bw_dom + m0 gradient kernel
//=============================================================================
// For each wave×res pair, computes dQ_dbw_dom contribution and m0 gradient.
// Writes dQ_dbw_dom to global memory for the next kernel (grad_g0_kernel).
//=============================================================================
__global__ void grad_bw_dom_kernel(
    const double* __restrict__ bw_p_real, const double* __restrict__ bw_p_imag,
    const double* __restrict__ common_amp_factor_real,
    const double* __restrict__ common_amp_factor_imag,
    const double* __restrict__ pap_real, const double* __restrict__ pap_imag,
    const double* __restrict__ pam_real, const double* __restrict__ pam_imag,
    const double* __restrict__ gp_real, const double* __restrict__ gp_imag,
    const double* __restrict__ gm_real, const double* __restrict__ gm_imag,
    const double* __restrict__ poq_real, const double* __restrict__ poq_imag,
    const double* __restrict__ dQ_dP,
    const double* __restrict__ bw_dom_real, const double* __restrict__ bw_dom_imag,
    const double* __restrict__ g_bw_real, const double* __restrict__ g_bw_imag,
    const double* __restrict__ frac, const int* __restrict__ m0_index,
    const int* __restrict__ bw_order, const double* __restrict__ m0,
    const double* __restrict__ ck_real, const double* __restrict__ ck_imag,
    double A_p,
    int n_wave, int n_res, int n_unique_bw,
    double* __restrict__ grad_m0_partial,
    double* __restrict__ dQ_dbw_dom_real,
    double* __restrict__ dQ_dbw_dom_imag,
    int n_events
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;

    // Event-constants via thread 0 → shared
    __shared__ double s_dQ_ap_r, s_dQ_ap_i, s_dQ_am_r, s_dQ_am_i;
    if (tid == 0) {
        double pr = pap_real[event_idx], pi = pap_imag[event_idx];
        double amr = pam_real[event_idx], ami = pam_imag[event_idx];
        double gpr = gp_real[event_idx], gpi = gp_imag[event_idx];
        double gmr = gm_real[event_idx], gmi = gm_imag[event_idx];
        double pqr = poq_real[event_idx], pqi = poq_imag[event_idx];
        double dP = dQ_dP[event_idx], fv = frac[event_idx];

        double dQ_dpb = dP * fv * (1.0 - A_p);
        double dQ_dpbbar = dP * (1.0 - fv) * (1.0 + A_p);

        // d_pb_dap = conj(pap) * gp
        // d_pbbar_dap = conj(pam) * (gm / poq)
        // dQ_ap = dQ_dpb * conj(pap) * gp + dQ_dpbbar * conj(pam) * (gm / poq)

        // First compute 1/poq
        double poq_norm = pqr * pqr + pqi * pqi;
        double poq_inv_r = pqr / poq_norm;
        double poq_inv_i = -pqi / poq_norm;

        // conj(pap) * gp = (pr - i*pi) * (gpr + i*gpi) = (pr*gpr + pi*gpi) + i*(pr*gpi - pi*gpr)
        double c1_r = pr * gpr + pi * gpi;
        double c1_i = pr * gpi - pi * gpr;

        // conj(pam) * (gm / poq) = conj(pam) * gm * inv_poq
        double gm_poq_r = gmr * poq_inv_r - gmi * poq_inv_i;
        double gm_poq_i = gmr * poq_inv_i + gmi * poq_inv_r;
        double c2_r = amr * gm_poq_r + ami * gm_poq_i;
        double c2_i = amr * gm_poq_i - ami * gm_poq_r;

        s_dQ_ap_r = dQ_dpb * c1_r + dQ_dpbbar * c2_r;
        s_dQ_ap_i = dQ_dpb * c1_i + dQ_dpbbar * c2_i;

        // d_pb_dam = conj(pap) * gm * poq
        double gm_poq_fwd_r = gmr * pqr - gmi * pqi;
        double gm_poq_fwd_i = gmr * pqi + gmi * pqr;
        double c3_r = pr * gm_poq_fwd_r + pi * gm_poq_fwd_i;
        double c3_i = pr * gm_poq_fwd_i - pi * gm_poq_fwd_r;

        // d_pbbar_dam = conj(pam) * gp
        double c4_r = amr * gpr + ami * gpi;
        double c4_i = amr * gpi - ami * gpr;

        s_dQ_am_r = dQ_dpb * c3_r + dQ_dpbbar * c4_r;
        s_dQ_am_i = dQ_dpb * c3_i + dQ_dpbbar * c4_i;
    }
    __syncthreads();

    double dQ_ap_r = s_dQ_ap_r, dQ_ap_i = s_dQ_ap_i;
    double dQ_am_r = s_dQ_am_r, dQ_am_i = s_dQ_am_i;

    // Zero dQ_dbw_dom and m0 outputs
    for (int bw_idx = tid; bw_idx < n_unique_bw; bw_idx += block_sz) {
        grad_m0_partial[event_idx * n_unique_bw + bw_idx] = 0.0;
        dQ_dbw_dom_real[event_idx * n_unique_bw + bw_idx] = 0.0;
        dQ_dbw_dom_imag[event_idx * n_unique_bw + bw_idx] = 0.0;
    }
    __syncthreads();

    int n_wave_half = n_wave / 2;
    // 1:1 thread→wave (wave_idx = tid, guard for idle threads)
    if (tid < n_wave) {
        int wave_idx = tid;
        double bpr = bw_p_real[event_idx * n_wave + wave_idx];
        double bpi = bw_p_imag[event_idx * n_wave + wave_idx];
        double car = common_amp_factor_real[event_idx * n_wave + wave_idx];
        double cai = common_amp_factor_imag[event_idx * n_wave + wave_idx];

        // dQ_da = (wave_idx < n_wave_half) ? dQ_ap : dQ_am
        double dqa_r = (wave_idx < n_wave_half) ? dQ_ap_r : dQ_am_r;
        double dqa_i = (wave_idx < n_wave_half) ? dQ_ap_i : dQ_am_i;

        double ckr = ck_real[wave_idx], cki = ck_imag[wave_idx];

        // one_over_bw = 1/bw_p
        double bpn = bpr * bpr + bpi * bpi;
        double obw_r = bpr / bpn;
        double obw_i = -bpi / bpn;

        // ck * one_over_bw (complex multiply)
        double ck_obw_r = ckr * obw_r - cki * obw_i;
        double ck_obw_i = ckr * obw_i + cki * obw_r;

        // -ck_obw * common_amp = -(ck_obw * common_amp)
        double neg_mul_r = -(ck_obw_r * car - ck_obw_i * cai);
        double neg_mul_i = -(ck_obw_r * cai + ck_obw_i * car);

        // dQ_dbw_p = dQ_da * neg_mul
        double ddbr = dqa_r * neg_mul_r - dqa_i * neg_mul_i;
        double ddbi = dqa_r * neg_mul_i + dqa_i * neg_mul_r;

        #pragma unroll
        for (int res_idx = 0; res_idx < n_res; res_idx++) {
            int bw_idx = bw_order[wave_idx * n_res + res_idx];
            double bdr = bw_dom_real[event_idx * n_unique_bw + bw_idx];
            double bdi = bw_dom_imag[event_idx * n_unique_bw + bw_idx];

            // contrib = dQ_dbw_p * (bw_p / bw_dom) = dQ_dbw_p * bw_p * (1/bw_dom)
            double bdn = bdr * bdr + bdi * bdi;
            double inv_bd_r = bdr / bdn;
            double inv_bd_i = -bdi / bdn;
            double bwp_div_bd_r = bpr * inv_bd_r - bpi * inv_bd_i;
            double bwp_div_bd_i = bpr * inv_bd_i + bpi * inv_bd_r;

            double cr = ddbr * bwp_div_bd_r - ddbi * bwp_div_bd_i;
            double ci = ddbr * bwp_div_bd_i + ddbi * bwp_div_bd_r;

            atomicAdd(&dQ_dbw_dom_real[event_idx * n_unique_bw + bw_idx], cr);
            atomicAdd(&dQ_dbw_dom_imag[event_idx * n_unique_bw + bw_idx], ci);

            double m0_val = m0[m0_index[bw_idx]];
            double gbr = g_bw_real[event_idx * n_unique_bw + bw_idx];
            double gbi = g_bw_imag[event_idx * n_unique_bw + bw_idx];
            // dbw_dom_dm0 = 2*m0 - i*g_bw  = 2*m0 + gbi - i*gbr
            double dm0_re = 2.0 * m0_val + gbi;
            double dm0_im = -gbr;
            // Wirtinger: ∂Q/∂m0 = 2·Re(contrib * dbw_dom_dm0)
            double w = 2.0 * (cr * dm0_re - ci * dm0_im);
            atomicAdd(&grad_m0_partial[event_idx * n_unique_bw + bw_idx], w);
        }
    }
}

//=============================================================================
// KERNEL 3c: g0 gradient kernel — FP32 inner loop for matrix multiply
//=============================================================================
// Reads dQ_dbw_dom from global (written by grad_bw_dom_kernel),
// converts to dQ_dg_bw = dQ_dbw_dom * (-1j * m0), then
// multiplies by matrix_gamma to get g0 gradient.
//
// Inner dot product uses FP32 (float matrix_gamma, float accumulate)
// to achieve 64× throughput vs FP64.  Final Wirtinger step in FP64.
//=============================================================================
__global__ void grad_g0_kernel(
    double* __restrict__ dQ_dbw_dom_real,
    double* __restrict__ dQ_dbw_dom_imag,
    const double* __restrict__ g_interp_real,
    const double* __restrict__ g_interp_imag,
    const double* __restrict__ m0,
    const int* __restrict__ m0_index,
    const int* __restrict__ gamma_col_idx,
    int n_unique_bw, int n_gamma_rows,
    double* __restrict__ grad_g0_partial,
    int n_events
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;

    // Convert dQ_dbw_dom → dQ_dg_bw in-place
    for (int bw_idx = tid; bw_idx < n_unique_bw; bw_idx += block_sz) {
        double m0_val = m0[m0_index[bw_idx]];
        double dr = dQ_dbw_dom_real[event_idx * n_unique_bw + bw_idx];
        double di = dQ_dbw_dom_imag[event_idx * n_unique_bw + bw_idx];
        dQ_dbw_dom_real[event_idx * n_unique_bw + bw_idx] = m0_val * di;
        dQ_dbw_dom_imag[event_idx * n_unique_bw + bw_idx] = -m0_val * dr;
    }
    __syncthreads();

    // Sparse gather using gamma_col_idx (each row has exactly one 1.0 entry)
    for (int gamma_idx = tid; gamma_idx < n_gamma_rows; gamma_idx += block_sz) {
        int col = gamma_col_idx[gamma_idx];
        double sum_r = dQ_dbw_dom_real[event_idx * n_unique_bw + col];
        double sum_i = dQ_dbw_dom_imag[event_idx * n_unique_bw + col];
        double gr = g_interp_real[event_idx * n_gamma_rows + gamma_idx];
        double gi = g_interp_imag[event_idx * n_gamma_rows + gamma_idx];
        grad_g0_partial[event_idx * n_gamma_rows + gamma_idx] =
            2.0 * (sum_r * gr - sum_i * gi);
    }
}

//=============================================================================
// KERNEL 3d: Scalar (time-evolution) gradient kernel
//=============================================================================
// Thread 0 only per event — computes Γ, ΔΓ, Δm, A_p, poq_rho, pop_phi grads.
//=============================================================================
__global__ void grad_scalar_kernel(
    const double* __restrict__ P,
    const double* __restrict__ pap_real, const double* __restrict__ pap_imag,
    const double* __restrict__ pam_real, const double* __restrict__ pam_imag,
    const double* __restrict__ gp_real, const double* __restrict__ gp_imag,
    const double* __restrict__ gm_real, const double* __restrict__ gm_imag,
    const double* __restrict__ poq_real, const double* __restrict__ poq_imag,
    const double* __restrict__ ap_real, const double* __restrict__ ap_imag,
    const double* __restrict__ am_real, const double* __restrict__ am_imag,
    const double* __restrict__ dQ_dP,
    const double* __restrict__ frac, const double* __restrict__ time,
    double Gamma, double Delta_Gamma, double Delta_m,
    double A_p, double poq_rho, double pop_phi,
    double* __restrict__ grad_Gamma_partial,
    double* __restrict__ grad_DeltaGamma_partial,
    double* __restrict__ grad_DeltaM_partial,
    double* __restrict__ grad_Ap_partial,
    double* __restrict__ grad_poq_rho_partial,
    double* __restrict__ grad_pop_phi_partial,
    int n_events
) {
    int event_idx = blockIdx.x;
    if (threadIdx.x != 0) return;

    complex pap(pap_real[event_idx], pap_imag[event_idx]);
    complex pam(pam_real[event_idx], pam_imag[event_idx]);
    complex gp(gp_real[event_idx], gp_imag[event_idx]);
    complex gm(gm_real[event_idx], gm_imag[event_idx]);
    complex poq(poq_real[event_idx], poq_imag[event_idx]);
    complex ap(ap_real[event_idx], ap_imag[event_idx]);
    complex am(am_real[event_idx], am_imag[event_idx]);
    double dQ_dP_val = dQ_dP[event_idx];
    double frac_val = frac[event_idx];
    double t = time[event_idx];

    double pb = thrust::norm(pap);
    double pbbar = thrust::norm(pam);
    double dP_dAp = -frac_val * pb + (1.0 - frac_val) * pbbar;
    grad_Ap_partial[event_idx] = dQ_dP_val * dP_dAp;
    grad_Gamma_partial[event_idx] = dQ_dP_val * (-t) * P[event_idx];

    double dQ_dpb = dQ_dP_val * frac_val * (1.0 - A_p);
    double dQ_dpbbar = dQ_dP_val * (1.0 - frac_val) * (1.0 + A_p);

    complex d_pb_dgp = conj(pap) * ap;
    complex d_pb_dgm = conj(pap) * poq * am;
    complex d_pbbar_dgp = conj(pam) * am;
    complex d_pbbar_dgm = conj(pam) * ap / poq;

    complex dQ_dgp = dQ_dpb * d_pb_dgp + dQ_dpbbar * d_pbbar_dgp;
    complex dQ_dgm = dQ_dpb * d_pb_dgm + dQ_dpbbar * d_pbbar_dgm;

    complex dgp_dDeltaGamma = (-t / 4.0) * gm;
    complex dgm_dDeltaGamma = (-t / 4.0) * gp;
    complex dgp_dDeltaM = complex(0.0, t / 2.0) * gm;
    complex dgm_dDeltaM = complex(0.0, t / 2.0) * gp;

    grad_DeltaGamma_partial[event_idx] = 2.0 * (dQ_dgp * dgp_dDeltaGamma + dQ_dgm * dgm_dDeltaGamma).real();
    grad_DeltaM_partial[event_idx] = 2.0 * (dQ_dgp * dgp_dDeltaM + dQ_dgm * dgm_dDeltaM).real();

    complex d_pb_dpoq = conj(pap) * gm * am;
    complex d_pbbar_dpoq = conj(pam) * (-gm / (poq * poq)) * ap;
    complex dQ_dpoq = dQ_dpb * d_pb_dpoq + dQ_dpbbar * d_pbbar_dpoq;

    complex exp_phi = exp(complex(0.0, 1.0) * pop_phi);
    grad_poq_rho_partial[event_idx] = 2.0 * (dQ_dpoq * exp_phi).real();
    grad_pop_phi_partial[event_idx] = 2.0 * (dQ_dpoq * poq_rho * complex(0.0, 1.0) * exp_phi).real();
}

//=============================================================================
// Reduction kernels (kept from original implementation)
//=============================================================================
__global__ void reduce_sum_kernel(const double* input, double* output, int n) {
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

__global__ void reduce_sum_complex_kernel(
    const double* real_in, const double* imag_in,
    double* real_out, double* imag_out, int n
) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double* sreal = sdata;
    double* simag = sdata + blockDim.x;
    sreal[tid] = (idx < n) ? real_in[idx] : 0.0;
    simag[tid] = (idx < n) ? imag_in[idx] : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { sreal[tid] += sreal[tid + s]; simag[tid] += simag[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { atomicAdd(real_out, sreal[0]); atomicAdd(imag_out, simag[0]); }
}

__global__ void reduce_sum_features_kernel(
    const double* input, double* output,
    int n_events, int n_features
) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int feat = blockIdx.x;
    if (feat >= n_features) return;
    double sum = 0.0;
    for (int i = tid; i < n_events; i += blockDim.x) {
        sum += input[feat + i * n_features];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[feat] = sdata[0];
}

__global__ void reduce_sum_complex_features_kernel(
    const double* real_in, const double* imag_in,
    double* real_out, double* imag_out,
    int n_events, int n_features
) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int feat = blockIdx.x;
    if (feat >= n_features) return;
    double real_sum = 0.0, imag_sum = 0.0;
    for (int i = tid; i < n_events; i += blockDim.x) {
        real_sum += real_in[feat + i * n_features];
        imag_sum += imag_in[feat + i * n_features];
    }
    double* sreal = sdata;
    double* simag = sdata + blockDim.x;
    sreal[tid] = real_sum;
    simag[tid] = imag_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { sreal[tid] += sreal[tid + s]; simag[tid] += simag[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { real_out[feat] = sreal[0]; imag_out[feat] = simag[0]; }
}

// ── Upload helpers (plain C, before extern "C") ──
void* _up_int(const int* src, int n) {
    int* d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, src, n * sizeof(int), cudaMemcpyHostToDevice)); return d;
}
void* _up_dbl(const double* src, int n) {
    double* d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d, src, n * sizeof(double), cudaMemcpyHostToDevice)); return d;
}
// Upload double[] as float[] on GPU
float* _up_f32(const double* src, int n) {
    float* d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    float* buf = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) buf[i] = (float)src[i];
    CUDA_CHECK(cudaMemcpy(d, buf, n * sizeof(float), cudaMemcpyHostToDevice));
    free(buf); return d;
}

//=============================================================================
// Host-callable launch functions
//=============================================================================
extern "C" {

// Structs for clean unified API: (Context*, Data*, Params*, norm, use_norm)
typedef struct {
    // Event data (GPU): mass only — momentum/angle are consumed by the
    // one-time cache fill at load_data and are not kept on the device.
    const double* mass;
    const double* frac; const double* time; const double* weight; const double* bkg;
    // Scratch buffers (GPU)
    double* g_interp_real; double* g_interp_imag;
    double* g_bw_real; double* g_bw_imag;
    double* Q_out; double* P_out;
    double* pap_real; double* pap_imag; double* pam_real; double* pam_imag;
    double* gp_real; double* gp_imag; double* gm_real; double* gm_imag;
    double* poq_real; double* poq_imag;
    double* bw_p_real; double* bw_p_imag;
    double* common_amp_factor_real; double* common_amp_factor_imag;
    double* ap_real; double* ap_imag; double* am_real; double* am_imag;
    double* dQ_dP;
    double* bw_dom_real; double* bw_dom_imag;
    double* grad_ck_real_partial; double* grad_ck_imag_partial;
    double* grad_m0_partial; double* grad_g0_partial;
    double* grad_Gamma_partial; double* grad_DeltaGamma_partial;
    double* grad_DeltaM_partial; double* grad_Ap_partial;
    double* grad_poq_rho_partial; double* grad_pop_phi_partial;
    double* dQ_dbw_dom_real;     // [n_events * n_unique_bw] — partial grad intermediate
    double* dQ_dbw_dom_imag;     // [n_events * n_unique_bw]
    // Angular-amplitude cache (cuda_v3_ampcache): per-event float2[n_uniq].
    // non-NULL after load_data (cache built there); NULL if that failed.
    float2* amp_cache;           // rebased per batch: base + st * n_uniq
    int n_events;
} ComputeData;

typedef struct {
    // Index arrays (GPU)
    const int* m0_index; const int* g0_index;
    const int* g0_mass_index; const int* mass_index;
    const int* fl_type; const int* fl_q_index;
    const int* bw_order; const int* fl_order; const int* angle_index;
    // Constant arrays (GPU)
    const double* angle_k; const double* angle_b;
    const double* matrix_angle_real; const double* matrix_angle_imag;
    const double* gamma_table_real; const double* gamma_table_imag;
    double gamma_min; double gamma_delta; double gamma_inv_delta; int gamma_table_bins;
    const double* matrix_gamma;
    const int* gamma_col_idx;  // [n_gamma_rows] — maps gamma row → unique_bw col (sparse matrix_gamma)
    const double* fl_table; double fl_min; double fl_delta; double fl_inv_delta; int fl_table_bins;
    // Dimensions
    int n_wave; int n_res; int n_decay; int n_unique_bw;
    int n_gamma_rows; int n_mass; int n_momentum;
    int n_angle_k; int n_angle_total; int n_angle_comp;
    int batch_size;
    int n_m0_params;   // actual m0/g0 array sizes (from max(index)+1)
    int n_g0_params;
    // cuda_v3_ampcache: minimal angular-cache layout
    const int* slot_of_wave;   // [n_wave]  wave → cache slot
    const int* rep_of_slot;    // [n_uniq]  slot → representative wave
    int n_uniq;                // number of unique angular amplitudes
    ComputeData* scratch;
    double* Q_red_gpu;
    // Profiling events
    int n_profile;
    int pt_enabled;      // non-zero → record + accumulate timings
    cudaEvent_t pe[10];  // up to 10 timing points
    double pt[10];       // elapsed ms per segment
} ComputeContext;

typedef struct {
    const double* ck_real; const double* ck_imag;
    const double* m0; const double* g0;
    double Gamma; double Delta_Gamma; double Delta_m;
    double A_prod; double poq_rho; double pop_phi;
} ComputeParams;

cudaError_t cuda_alloc(void** ptr, size_t size) { return cudaMalloc(ptr, size); }
cudaError_t cuda_free(void* ptr) { return cudaFree(ptr); }
cudaError_t cuda_memcpy_to_device(void* dst, const void* src, size_t size) {
    return cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}
cudaError_t cuda_memcpy_to_host(void* dst, const void* src, size_t size) {
    return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}
cudaError_t cuda_memset(void* ptr, int value, size_t size) { return cudaMemset(ptr, value, size); }
int cuda_get_device_count() { int count; cudaGetDeviceCount(&count); return count; }
cudaError_t cuda_get_device_name(char* name, int len) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err == cudaSuccess) strncpy(name, prop.name, len);
    return err;
}

//── g_bw launch wrapper ────────────────────────────────────────────
void launch_compute_g_bw(
    const double* mass, const double* g0,
    const int* g0_index, const int* g0_mass_index,
    const int* gamma_col_idx,
    const double* gamma_table_real, const double* gamma_table_imag,
    double gamma_min, double gamma_delta,
    int n_gamma_rows, int n_unique_bw, int n_mass, int gamma_table_bins,
    double* g_interp_real, double* g_interp_imag,
    double* g_bw_real, double* g_bw_imag,
    int n_events) {

    size_t shmem = (2 * n_gamma_rows + 2 * n_unique_bw) * sizeof(double);
    int gt = (n_gamma_rows < 256) ? 256 : (n_gamma_rows > 1024 ? 1024 : n_gamma_rows);
    double gamma_inv_delta = 1.0 / gamma_delta;
    compute_g_bw_kernel<<<n_events, gt, shmem>>>(
        mass, g0, g0_index, g0_mass_index, gamma_col_idx,
        gamma_table_real, gamma_table_imag,
        gamma_min, gamma_inv_delta,
        n_gamma_rows, n_unique_bw, n_mass, gamma_table_bins,
        g_interp_real, g_interp_imag,
        g_bw_real, g_bw_imag, n_events);
}

// Reduction launch wrappers (same as original)
void launch_reduce_sum(const double* input, double* output, int n) {
    
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_sum_kernel<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(input, output, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_reduce_sum_complex(const double* real_in, const double* imag_in,
    double* real_out, double* imag_out, int n) {
    
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_sum_complex_kernel<<<grid_size, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(double)>>>(
        real_in, imag_in, real_out, imag_out, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_reduce_sum_features(const double* input, double* output,
    int n_events, int n_features) {
    
    reduce_sum_features_kernel<<<n_features, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(
        input, output, n_events, n_features);
}

void launch_reduce_sum_complex_features(const double* real_in, const double* imag_in,
    double* real_out, double* imag_out,
    int n_events, int n_features) {
    
    reduce_sum_complex_features_kernel<<<n_features, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(double)>>>(
        real_in, imag_in, real_out, imag_out, n_events, n_features);
    CUDA_CHECK(cudaGetLastError());
}

// ── Gram matrix kernel launches ────────────────────────────────
void launch_common_amp_to_groups(
    const double* common_r, const double* common_i,
    double* A0_real, double* A0_imag,
    double* A1_real, double* A1_imag,
    int n_wave, int n_events) {

    int ng = n_wave / 8;
    int blk = ng < 32 ? 32 : ng;
    if (blk > 256) blk = 256;
    common_amp_to_groups_kernel<<<n_events, blk>>>(
        common_r, common_i, A0_real, A0_imag, A1_real, A1_imag,
        n_wave, n_events);
}

void launch_gram_reduce_v3(
    const double* A0_real, const double* A0_imag,
    const double* A1_real, const double* A1_imag,
    const double* weight,
    int n_events, int ng,
    double* Mpp_r, double* Mpp_i,
    double* Mmm_r, double* Mmm_i,
    double* Mpm_r, double* Mpm_i) {

    dim3 grid(ng, ng);
    gram_reduce_kernel_v3<<<grid, 1>>>(
        A0_real, A0_imag, A1_real, A1_imag, weight,
        n_events, ng,
        Mpp_r, Mpp_i, Mmm_r, Mmm_i, Mpm_r, Mpm_i);
}

// Unified launch: (Context*, Data*, Params*, norm, use_norm)
void launch_compute_all(
    const ComputeContext* ctx, ComputeData* data,
    const ComputeParams* params, double norm, int use_norm
) {
    int nw = ctx->n_wave, nu = ctx->n_unique_bw, ng = ctx->n_gamma_rows;
    int ne = data->n_events;
    int ip = 0;
    ComputeContext* cc = (ComputeContext*)ctx;

    cudaEventRecord(cc->pe[ip++], 0);

    //── K1: g_bw ──────────────────────────────────────────────────────
    launch_compute_g_bw(
        data->mass, params->g0, ctx->g0_index, ctx->g0_mass_index,
        ctx->gamma_col_idx,
        ctx->gamma_table_real, ctx->gamma_table_imag,
        ctx->gamma_min, ctx->gamma_delta,
        ctx->n_gamma_rows, ctx->n_unique_bw, ctx->n_mass, ctx->gamma_table_bins,
        data->g_interp_real, data->g_interp_imag,
        data->g_bw_real, data->g_bw_imag, ne);
    cudaEventRecord(cc->pe[ip++], 0);

    //── K2: cached forward — BW denominator recomputed per iteration, angular
    //     amplitude read from the per-handle cache (no FA/FL/fallback path).
    {
        size_t shmem = (ctx->n_mass + 2 * nu) * sizeof(double);
        amp_cache_amp_kernel<<<ne, nw, shmem>>>(
            data->mass,
            ctx->m0_index, ctx->mass_index, ctx->bw_order,
            data->g_bw_real, data->g_bw_imag,
            nw, ctx->n_res, nu, ctx->n_mass, params->m0,
            data->amp_cache, ctx->slot_of_wave, ctx->n_uniq,
            data->bw_p_real, data->bw_p_imag,
            data->common_amp_factor_real, data->common_amp_factor_imag,
            data->bw_dom_real, data->bw_dom_imag, ne);
    }
    cudaEventRecord(cc->pe[ip++], 0);

    //── K3: amp_reduce_time ───────────────────────────────────────────
    amp_reduce_time_kernel<<<ne, BLOCK_SIZE>>>(
        data->common_amp_factor_real, data->common_amp_factor_imag,
        params->ck_real, params->ck_imag,
        data->frac, data->time, data->weight, data->bkg,
        params->Gamma, params->Delta_Gamma, params->Delta_m,
        params->A_prod, params->poq_rho, params->pop_phi,
        data->Q_out, data->P_out,
        data->pap_real, data->pap_imag, data->pam_real, data->pam_imag,
        data->gp_real, data->gp_imag, data->gm_real, data->gm_imag,
        data->poq_real, data->poq_imag,
        data->ap_real, data->ap_imag, data->am_real, data->am_imag, data->dQ_dP,
        nw, ne, use_norm, norm);
    cudaEventRecord(cc->pe[ip++], 0);

    //── K4: grad_ck ───────────────────────────────────────────────────
    grad_ck_kernel<<<ne, BLOCK_SIZE>>>(
        data->common_amp_factor_real, data->common_amp_factor_imag,
        data->pap_real, data->pap_imag, data->pam_real, data->pam_imag,
        data->gp_real, data->gp_imag, data->gm_real, data->gm_imag,
        data->poq_real, data->poq_imag,
        data->dQ_dP, data->frac, params->A_prod, nw,
        data->grad_ck_real_partial, data->grad_ck_imag_partial, ne);
    cudaEventRecord(cc->pe[ip++], 0);

    //── K5: grad_bw_dom (1:1 thread→wave) ─────────────────────────
    grad_bw_dom_kernel<<<ne, nw>>>(
        data->bw_p_real, data->bw_p_imag,
        data->common_amp_factor_real, data->common_amp_factor_imag,
        data->pap_real, data->pap_imag, data->pam_real, data->pam_imag,
        data->gp_real, data->gp_imag, data->gm_real, data->gm_imag,
        data->poq_real, data->poq_imag, data->dQ_dP,
        data->bw_dom_real, data->bw_dom_imag,
        data->g_bw_real, data->g_bw_imag,
        data->frac, ctx->m0_index, ctx->bw_order, params->m0,
        params->ck_real, params->ck_imag, params->A_prod,
        nw, ctx->n_res, nu,
        data->grad_m0_partial,
        data->dQ_dbw_dom_real, data->dQ_dbw_dom_imag, ne);
    cudaEventRecord(cc->pe[ip++], 0);

    //── K6: grad_g0 (sparse gather) ────────────────────────────────
    grad_g0_kernel<<<ne, BLOCK_SIZE>>>(
        data->dQ_dbw_dom_real, data->dQ_dbw_dom_imag,
        data->g_interp_real, data->g_interp_imag,
        params->m0, ctx->m0_index, ctx->gamma_col_idx,
        nu, ng, data->grad_g0_partial, ne);
    cudaEventRecord(cc->pe[ip++], 0);

    //── K7: grad_scalar ──────────────────────────────────────────────
    grad_scalar_kernel<<<ne, 1>>>(
        data->P_out, data->pap_real, data->pap_imag,
        data->pam_real, data->pam_imag,
        data->gp_real, data->gp_imag, data->gm_real, data->gm_imag,
        data->poq_real, data->poq_imag,
        data->ap_real, data->ap_imag, data->am_real, data->am_imag,
        data->dQ_dP, data->frac, data->time,
        params->Gamma, params->Delta_Gamma, params->Delta_m,
        params->A_prod, params->poq_rho, params->pop_phi,
        data->grad_Gamma_partial, data->grad_DeltaGamma_partial,
        data->grad_DeltaM_partial, data->grad_Ap_partial,
        data->grad_poq_rho_partial, data->grad_pop_phi_partial, ne);
    cudaEventRecord(cc->pe[ip++], 0);

    cc->n_profile = ip;
    cudaGetLastError();
}

typedef struct {
    // Event data kept on device: mass (recomputed every fit iteration) and
    // the scalar arrays.  momentum/angle are NOT kept — they are uploaded
    // transiently only for the one-time cache fill in cuda_load_data_v3.
    const double* m;
    const double* f; const double* t; const double* w;
    const double* b;
    int ne; int nm; int nmom; int nat; int nac;
    // cuda_v3_ampcache: per-handle angular-amplitude cache (fp32 — constant
    // over the fit, so the rounding is a fixed factor and never moves the
    // minimum; halves the memory vs fp64)
    float2* amp_cache;    // [ne * n_uniq], built at load_data
    int n_uniq;           // cache slots per event (0 = no cache)
} DataHandle2;

void* cuda_create_context_v3(
    const int* m0_i,int n1, const int* g0_i,int n2,
    const int* g0_m,int n3, const int* mass_i,int n4,
    const int* fl_t,int n5, const int* fl_q,int n6,
    const int* bw_o,int n7, const int* fl_o,int n8,
    const int* ang_i,int n9,
    const double* ak,int n10, const double* ab,int n11,
    const double* mar,int n12, const double* mai,int n13,
    const double* gtr,int n14, const double* gti,int n15,
    double gmin,double gdel,int gbins,
    const double* mg,int n16,
    const int* gci,int ngci,
    const double* ft,int n17, double flmin,double fldel,int fbins,
    int nw,int nr,int nd,int nub,int ngr,
    int nm,int nmom,int nak_,int nat,int nac,
    int n_m0p, int n_g0p,
    int batch_size,
    const int* slot_of_wave,int n_slot,
    const int* rep_of_slot,int n_rep,
    int n_uniq
) {
    ComputeContext* c = (ComputeContext*)calloc(1, sizeof(ComputeContext));
    c->m0_index = (int*)_up_int(m0_i, n1); c->g0_index = (int*)_up_int(g0_i, n2);
    c->g0_mass_index = (int*)_up_int(g0_m, n3); c->mass_index = (int*)_up_int(mass_i, n4);
    c->fl_type = (int*)_up_int(fl_t, n5); c->fl_q_index = (int*)_up_int(fl_q, n6);
    c->bw_order = (int*)_up_int(bw_o, n7); c->fl_order = (int*)_up_int(fl_o, n8);
    c->angle_index = (int*)_up_int(ang_i, n9);
    c->angle_k = (double*)_up_dbl(ak, n10); c->angle_b = (double*)_up_dbl(ab, n11);
    c->matrix_angle_real = (double*)_up_dbl(mar, n12); c->matrix_angle_imag = (double*)_up_dbl(mai, n13);
    c->gamma_table_real = (double*)_up_dbl(gtr, n14); c->gamma_table_imag = (double*)_up_dbl(gti, n15);
    c->gamma_min = gmin; c->gamma_delta = gdel; c->gamma_table_bins = gbins;
    c->matrix_gamma = (double*)_up_dbl(mg, n16);
    c->gamma_col_idx = (int*)_up_int(gci, ngci);
    c->fl_table = (double*)_up_dbl(ft, n17); c->fl_min = flmin; c->fl_delta = fldel; c->fl_table_bins = fbins;
    c->n_wave = nw; c->n_res = nr; c->n_decay = nd;
    c->n_unique_bw = nub; c->n_gamma_rows = ngr;
    c->n_mass = nm; c->n_momentum = nmom; c->n_angle_k = nak_; c->n_angle_total = nat; c->n_angle_comp = nac;
    c->n_m0_params = n_m0p; c->n_g0_params = n_g0p;
    c->batch_size = batch_size > 0 ? batch_size : DEFAULT_BATCH_SIZE;
    // cuda_v3_ampcache: minimal angular-cache layout (wave → slot, slot → rep)
    c->slot_of_wave = (n_slot > 0) ? (int*)_up_int(slot_of_wave, n_slot) : NULL;
    c->rep_of_slot = (n_rep > 0) ? (int*)_up_int(rep_of_slot, n_rep) : NULL;
    c->n_uniq = (n_uniq > 0) ? n_uniq : 0;

    // Pre-allocate scratch buffers when batch_size is known
    if (c->batch_size > 0) {
        int bs = c->batch_size;
        c->scratch = (ComputeData*)calloc(1, sizeof(ComputeData));
        #define S(f) CUDA_CHECK(cudaMalloc(&c->scratch->f, bs * sizeof(double)))
        #define S2(f,n) CUDA_CHECK(cudaMalloc(&c->scratch->f, bs * (n) * sizeof(double)))
        S2(g_interp_real, ngr); S2(g_interp_imag, ngr);
        S2(g_bw_real, nub); S2(g_bw_imag, nub);
        S(Q_out); S(P_out); S(pap_real); S(pap_imag); S(pam_real); S(pam_imag);
        S(gp_real); S(gp_imag); S(gm_real); S(gm_imag); S(poq_real); S(poq_imag);
        S2(bw_p_real, nw); S2(bw_p_imag, nw);
        S2(common_amp_factor_real, nw); S2(common_amp_factor_imag, nw);
        S(ap_real); S(ap_imag); S(am_real); S(am_imag); S(dQ_dP);
        S2(bw_dom_real, nub); S2(bw_dom_imag, nub);
        S2(grad_ck_real_partial, nw); S2(grad_ck_imag_partial, nw);
        S2(grad_m0_partial, nub); S2(grad_g0_partial, ngr);
        S(grad_Gamma_partial); S(grad_DeltaGamma_partial);
        S(grad_DeltaM_partial); S(grad_Ap_partial);
        S(grad_poq_rho_partial); S(grad_pop_phi_partial);
        S2(dQ_dbw_dom_real, nub); S2(dQ_dbw_dom_imag, nub);
        #undef S
        #undef S2
        CUDA_CHECK(cudaMalloc(&c->Q_red_gpu, 8));
        // Create profile events
        for (int i = 0; i < 10; i++) cudaEventCreate(&c->pe[i]);
        c->n_profile = 0;
        c->pt_enabled = 0;
        for (int i = 0; i < 10; i++) c->pt[i] = 0.0;
    } else {
        c->scratch = NULL;
        c->Q_red_gpu = NULL;
    }
    return c;
}
void cuda_free_context_v3(void* vctx) {
    ComputeContext* c = (ComputeContext*)vctx;
    #define F(p) cudaFree((void*)c->p)
    F(m0_index); F(g0_index); F(g0_mass_index); F(mass_index);
    F(fl_type); F(fl_q_index); F(bw_order); F(fl_order); F(angle_index);
    F(angle_k); F(angle_b); F(matrix_angle_real); F(matrix_angle_imag);
    F(gamma_table_real); F(gamma_table_imag); F(matrix_gamma); F(gamma_col_idx); F(fl_table);
    if (c->slot_of_wave) cudaFree((void*)c->slot_of_wave);
    if (c->rep_of_slot) cudaFree((void*)c->rep_of_slot);
    #undef F
    // Free pre-allocated scratch
    if (c->scratch) {
        #define SF(f) cudaFree(c->scratch->f)
        SF(g_interp_real); SF(g_interp_imag); SF(g_bw_real); SF(g_bw_imag);
        SF(Q_out); SF(P_out); SF(pap_real); SF(pap_imag); SF(pam_real); SF(pam_imag);
        SF(gp_real); SF(gp_imag); SF(gm_real); SF(gm_imag); SF(poq_real); SF(poq_imag);
        SF(bw_p_real); SF(bw_p_imag); SF(common_amp_factor_real); SF(common_amp_factor_imag);
        SF(ap_real); SF(ap_imag); SF(am_real); SF(am_imag); SF(dQ_dP);
        SF(bw_dom_real); SF(bw_dom_imag);
        SF(grad_ck_real_partial); SF(grad_ck_imag_partial);
        SF(grad_m0_partial); SF(grad_g0_partial);
        SF(grad_Gamma_partial); SF(grad_DeltaGamma_partial);
        SF(grad_DeltaM_partial); SF(grad_Ap_partial);
        SF(grad_poq_rho_partial); SF(grad_pop_phi_partial);
        SF(dQ_dbw_dom_real); SF(dQ_dbw_dom_imag);
        #undef SF
        free(c->scratch);
    }
    if (c->Q_red_gpu) cudaFree(c->Q_red_gpu);
    for (int i = 0; i < 10; i++) cudaEventDestroy(c->pe[i]);
    free(c);
}

void cuda_get_profile_v3(void* vctx, double* out_times, int* out_n) {
    ComputeContext* c = (ComputeContext*)vctx;
    for (int i = 0; i < c->n_profile - 1 && i < 9; i++)
        out_times[i] = c->pt[i];
    *out_n = c->n_profile - 1;
    // Reset for next call
    for (int i = 0; i < 10; i++) c->pt[i] = 0.0;
    c->n_profile = 0;
}

void cuda_enable_profile_v3(void* vctx, int enable) {
    ComputeContext* c = (ComputeContext*)vctx;
    c->pt_enabled = enable;
}

// Forward decls: the angular cache is allocated once per handle and filled
// in event-chunks from TRANSIENT momentum/angle device buffers (uploaded per
// chunk only for the one-time fill, then freed — never kept on the device).
static int alloc_handle_amp_cache(ComputeContext* c, DataHandle2* h);
static int fill_amp_cache_chunk(ComputeContext* c, DataHandle2* h,
                                const float* mom, const float* ang,
                                int n_events, int base_event);

void* cuda_load_data_v3(void* vctx,
    const double* mass,int nmass, const double* mom,int nmom,
    const double* ang,int nang, const double* frac,const double* time,
    const double* wgt,const double* bkg,int ne
) {
    DataHandle2* h = (DataHandle2*)calloc(1, sizeof(DataHandle2));
    ComputeContext* c_ctx = (ComputeContext*)vctx;
    int nac = c_ctx ? c_ctx->n_angle_comp : 3;
    h->m = (const double*)_up_dbl(mass, ne * nmass);
    h->f = (const double*)_up_dbl(frac, ne);
    h->t = (const double*)_up_dbl(time, ne);
    h->w = (const double*)_up_dbl(wgt, ne);
    h->b = (const double*)_up_dbl(bkg, ne);
    h->ne = ne; h->nm = nmass; h->nmom = nmom; h->nat = nang; h->nac = nac;

    if (c_ctx) {
        if (!alloc_handle_amp_cache(c_ctx, h)) goto fail;
        // momentum/angle are uploaded per chunk (bounded by batch_size),
        // used for the fill, then freed.  The device chunk buffers and the
        // host f32 conversion temps are allocated once and reused.
        int fb = c_ctx->batch_size > 0 ? c_ctx->batch_size : ne;
        if (fb > ne) fb = ne;
        float* gpu_mom = NULL; float* gpu_ang = NULL;
        float* tmp_mom = NULL; float* tmp_ang = NULL;
        if (cudaMalloc(&gpu_mom, (size_t)fb * nmom * sizeof(float))
                != cudaSuccess) goto fail;
        if (cudaMalloc(&gpu_ang, (size_t)fb * nang * nac * sizeof(float))
                != cudaSuccess) { cudaFree(gpu_mom); goto fail; }
        tmp_mom = (float*)malloc((size_t)fb * nmom * sizeof(float));
        tmp_ang = (float*)malloc((size_t)fb * nang * nac * sizeof(float));
        int failed = 0;
        for (int base = 0; base < ne && !failed; base += fb) {
            int nb = (ne - base > fb) ? fb : (ne - base);
            for (int i = 0; i < nb * nmom; i++)
                tmp_mom[i] = (float)mom[(size_t)base * nmom + i];
            for (int i = 0; i < nb * nang * nac; i++)
                tmp_ang[i] = (float)ang[(size_t)base * nang * nac + i];
            cudaMemcpy(gpu_mom, tmp_mom, (size_t)nb * nmom * sizeof(float),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_ang, tmp_ang, (size_t)nb * nang * nac * sizeof(float),
                       cudaMemcpyHostToDevice);
            failed = !fill_amp_cache_chunk(c_ctx, h, gpu_mom, gpu_ang, nb, base);
        }
        free(tmp_mom); free(tmp_ang);
        cudaFree(gpu_mom); cudaFree(gpu_ang);
        if (failed) goto fail;
    }
    return h;
fail:
    cudaFree((void*)h->m); cudaFree((void*)h->f); cudaFree((void*)h->t);
    cudaFree((void*)h->w); cudaFree((void*)h->b);
    if (h->amp_cache) cudaFree(h->amp_cache);
    free(h);
    return NULL;
}
void cuda_free_data_v3(void* vh) {
    if (!vh) return;
    DataHandle2* h = (DataHandle2*)vh;
    cudaFree((void*)h->m); cudaFree((void*)h->f);
    cudaFree((void*)h->t); cudaFree((void*)h->w); cudaFree((void*)h->b);
    if (h->amp_cache) cudaFree(h->amp_cache);
    free(h);
}

void cuda_gram_matrix_v3(void* vctx, void* vdh,
    const double* m0, const double* g0,
    double* oMpp_r, double* oMpp_i,
    double* oMmm_r, double* oMmm_i,
    double* oMpm_r, double* oMpm_i) {

    ComputeContext* c = (ComputeContext*)vctx;
    DataHandle2* h = (DataHandle2*)vdh;
    int ne = h->ne, bs = c->batch_size;
    if (ne < bs) bs = ne;
    int nbat = (ne + bs - 1) / bs;
    int nw = c->n_wave, nu = c->n_unique_bw, ng = c->n_gamma_rows;
    int ng2 = nw / 8;

    const double* gpu_m0 = (const double*)_up_dbl(m0, c->n_m0_params);
    const double* gpu_g0 = (const double*)_up_dbl(g0, c->n_g0_params);

    ComputeData s;
    if (c->scratch) {
        s = *c->scratch;
    } else {
        memset(&s, 0, sizeof(ComputeData));
        #define S(f) CUDA_CHECK(cudaMalloc(&s.f, bs * sizeof(double)))
        #define S2(f,n) CUDA_CHECK(cudaMalloc(&s.f, bs * (n) * sizeof(double)))
        S2(g_interp_real,ng); S2(g_interp_imag,ng);
        S2(g_bw_real,nu); S2(g_bw_imag,nu);
        S2(common_amp_factor_real,nw); S2(common_amp_factor_imag,nw);
        S2(bw_p_real,nw); S2(bw_p_imag,nw);
        S2(bw_dom_real,nu); S2(bw_dom_imag,nu);
        #undef S
        #undef S2
    }

    double *A0r, *A0i, *A1r, *A1i;
    size_t a_sz = (size_t)bs * ng2 * sizeof(double);
    CUDA_CHECK(cudaMalloc(&A0r, a_sz)); CUDA_CHECK(cudaMalloc(&A0i, a_sz));
    CUDA_CHECK(cudaMalloc(&A1r, a_sz)); CUDA_CHECK(cudaMalloc(&A1i, a_sz));

    size_t g_sz = (size_t)ng2 * ng2 * sizeof(double);
    double *Mpp_r, *Mpp_i, *Mmm_r, *Mmm_i, *Mpm_r, *Mpm_i;
    CUDA_CHECK(cudaMalloc(&Mpp_r, g_sz)); CUDA_CHECK(cudaMalloc(&Mpp_i, g_sz));
    CUDA_CHECK(cudaMalloc(&Mmm_r, g_sz)); CUDA_CHECK(cudaMalloc(&Mmm_i, g_sz));
    CUDA_CHECK(cudaMalloc(&Mpm_r, g_sz)); CUDA_CHECK(cudaMalloc(&Mpm_i, g_sz));

    memset(oMpp_r, 0, g_sz); memset(oMpp_i, 0, g_sz);
    memset(oMmm_r, 0, g_sz); memset(oMmm_i, 0, g_sz);
    memset(oMpm_r, 0, g_sz); memset(oMpm_i, 0, g_sz);

    double* hbuf = (double*)malloc(g_sz);

    for (int b = 0; b < nbat; b++) {
        int st = b * bs;
        int nb = (ne - st > bs) ? bs : (ne - st);

        ComputeData d = s;
        d.mass = h->m + st * h->nm;
        d.weight = h->w + st;
        d.n_events = nb;
        d.amp_cache = (h->amp_cache != NULL)
            ? (h->amp_cache + (size_t)st * c->n_uniq) : NULL;

        cudaMemset(d.g_interp_real, 0, bs * ng * 8);
        cudaMemset(d.g_interp_imag, 0, bs * ng * 8);
        cudaMemset(d.g_bw_real, 0, bs * nu * 8);
        cudaMemset(d.g_bw_imag, 0, bs * nu * 8);

        launch_compute_g_bw(
            d.mass, gpu_g0, c->g0_index, c->g0_mass_index,
            c->gamma_col_idx,
            c->gamma_table_real, c->gamma_table_imag,
            c->gamma_min, c->gamma_delta,
            c->n_gamma_rows, c->n_unique_bw, c->n_mass, c->gamma_table_bins,
            d.g_interp_real, d.g_interp_imag,
            d.g_bw_real, d.g_bw_imag, nb);
        CUDA_CHECK(cudaGetLastError());

        // cached forward: common_amp = cached_Amp[slot]/bw_p per wave (fp64)
        {
            size_t shmem = (c->n_mass + 2 * nu) * sizeof(double);
            amp_cache_amp_kernel<<<nb, nw, shmem>>>(
                d.mass,
                c->m0_index, c->mass_index, c->bw_order,
                d.g_bw_real, d.g_bw_imag,
                nw, c->n_res, nu, c->n_mass, gpu_m0,
                d.amp_cache, c->slot_of_wave, c->n_uniq,
                d.bw_p_real, d.bw_p_imag,
                d.common_amp_factor_real, d.common_amp_factor_imag,
                d.bw_dom_real, d.bw_dom_imag, nb);
        }
        CUDA_CHECK(cudaGetLastError());

        launch_common_amp_to_groups(
            d.common_amp_factor_real, d.common_amp_factor_imag,
            A0r, A0i, A1r, A1i, nw, nb);
        CUDA_CHECK(cudaGetLastError());

        launch_gram_reduce_v3(
            A0r, A0i, A1r, A1i, d.weight, nb, ng2,
            Mpp_r, Mpp_i, Mmm_r, Mmm_i, Mpm_r, Mpm_i);
        CUDA_CHECK(cudaGetLastError());

        cudaMemcpy(hbuf, Mpp_r, g_sz, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < (size_t)ng2 * ng2; i++) oMpp_r[i] += hbuf[i];
        cudaMemcpy(hbuf, Mpp_i, g_sz, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < (size_t)ng2 * ng2; i++) oMpp_i[i] += hbuf[i];
        cudaMemcpy(hbuf, Mmm_r, g_sz, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < (size_t)ng2 * ng2; i++) oMmm_r[i] += hbuf[i];
        cudaMemcpy(hbuf, Mmm_i, g_sz, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < (size_t)ng2 * ng2; i++) oMmm_i[i] += hbuf[i];
        cudaMemcpy(hbuf, Mpm_r, g_sz, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < (size_t)ng2 * ng2; i++) oMpm_r[i] += hbuf[i];
        cudaMemcpy(hbuf, Mpm_i, g_sz, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < (size_t)ng2 * ng2; i++) oMpm_i[i] += hbuf[i];
    }

    free(hbuf);
    cudaFree((void*)gpu_m0); cudaFree((void*)gpu_g0);
    cudaFree(A0r); cudaFree(A0i); cudaFree(A1r); cudaFree(A1i);
    cudaFree(Mpp_r); cudaFree(Mpp_i); cudaFree(Mmm_r); cudaFree(Mmm_i);
    cudaFree(Mpm_r); cudaFree(Mpm_i);
    if (!c->scratch) {
        #define F(p) do { if(s.p) cudaFree(s.p); } while(0)
        F(g_interp_real); F(g_interp_imag); F(g_bw_real); F(g_bw_imag);
        F(common_amp_factor_real); F(common_amp_factor_imag);
        F(bw_p_real); F(bw_p_imag); F(bw_dom_real); F(bw_dom_imag);
        #undef F
    }
}

// ── cuda_v3_ampcache: per-handle angular-amplitude cache ──
// Allocates ne·n_uniq float2 (fp64-computed fa·fl stored as float2 — constant
// over the fit, so the fp32 rounding is a fixed per-event factor that does
// not move the minimum).  Filled at load_data in event chunks of
// batch_size; momentum/angle device copies are transient per chunk.
static int alloc_handle_amp_cache(ComputeContext* c, DataHandle2* h) {
    if (h->amp_cache != NULL || h->n_uniq > 0) return 1;
    if (c->n_uniq <= 0 || !c->rep_of_slot || !c->matrix_angle_real) return 0;
    size_t need = (size_t)h->ne * c->n_uniq * sizeof(float2);
    float2* amp_cache = NULL;
    if (cudaMalloc(&amp_cache, need) != cudaSuccess) return 0;
    h->amp_cache = amp_cache;
    h->n_uniq = c->n_uniq;
    return 1;
}

static int fill_amp_cache_chunk(ComputeContext* c, DataHandle2* h,
                                const float* mom, const float* ang,
                                int n_events, int base_event) {
    size_t shmem = (size_t)c->n_angle_k * sizeof(double);
    int fill_b = c->n_uniq;
    if (fill_b < 256) fill_b = 256;
    if (fill_b > 1024) fill_b = 1024;
    amp_cache_fill_kernel<<<n_events, fill_b, shmem>>>(
        ang, c->angle_index,
        c->angle_k, c->angle_b,
        c->matrix_angle_real, c->matrix_angle_imag,
        mom, c->fl_type, c->fl_q_index, c->fl_order,
        c->fl_table, c->fl_min, c->fl_delta,
        c->rep_of_slot,
        c->n_wave, c->n_angle_k, c->n_angle_total, c->n_angle_comp,
        c->n_decay, c->n_momentum, c->fl_table_bins, c->n_uniq,
        h->amp_cache + (size_t)base_event * c->n_uniq, n_events);
    return cudaGetLastError() == cudaSuccess;
}

void cuda_compute_v3(void* vctx, void* vdh,
    const double* ck_r,const double* ck_i,
    const double* m0,const double* g0,
    double G,double DG,double DM,double Ap,double pr,double pp,
    double nv,int use_norm,
    double* oQ,double* oP,
    double* ogck_r,double* ogck_i,
    double* ogm0,double* ogg0,
    double* ogsc
) {
    ComputeContext* c = (ComputeContext*)vctx;
    DataHandle2* h = (DataHandle2*)vdh;
    // The angular cache was built at load_data; no build check needed here.
    int ne = h->ne, bs = c->batch_size;
    if (ne < bs) bs = ne;
    int nbat = (ne + bs - 1) / bs;
    int nw = c->n_wave, nu = c->n_unique_bw, ng = c->n_gamma_rows;

    // Upload per-call params via _up_dbl (always fresh)
    ComputeParams p;
    p.ck_real = (double*)_up_dbl(ck_r, nw);
    p.ck_imag = (double*)_up_dbl(ck_i, nw);
    p.m0 = (double*)_up_dbl(m0, c->n_m0_params);
    p.g0 = (double*)_up_dbl(g0, c->n_g0_params);
    p.Gamma = G; p.Delta_Gamma = DG; p.Delta_m = DM;
    p.A_prod = Ap; p.poq_rho = pr; p.pop_phi = pp;

    // Use context-allocated scratch (avoids per-call cudaMalloc/free)
    ComputeData s;
    if (c->scratch) {
        s = *c->scratch;
    } else {
        memset(&s, 0, sizeof(ComputeData));
        #define S(f) CUDA_CHECK(cudaMalloc(&s.f, bs * sizeof(double)))
        #define S2(f,n) CUDA_CHECK(cudaMalloc(&s.f, bs * (n) * sizeof(double)))
        S2(g_interp_real,ng); S2(g_interp_imag,ng);
        S2(g_bw_real,nu); S2(g_bw_imag,nu);
        S(Q_out); S(P_out); S(pap_real); S(pap_imag); S(pam_real); S(pam_imag);
        S(gp_real); S(gp_imag); S(gm_real); S(gm_imag); S(poq_real); S(poq_imag);
        S2(bw_p_real,nw); S2(bw_p_imag,nw);
        S2(common_amp_factor_real,nw); S2(common_amp_factor_imag,nw);
        S(ap_real); S(ap_imag); S(am_real); S(am_imag); S(dQ_dP);
        S2(bw_dom_real,nu); S2(bw_dom_imag,nu);
        S2(grad_ck_real_partial,nw); S2(grad_ck_imag_partial,nw);
        S2(grad_m0_partial,nu); S2(grad_g0_partial,ng);
        S(grad_Gamma_partial); S(grad_DeltaGamma_partial);
        S(grad_DeltaM_partial); S(grad_Ap_partial);
        S(grad_poq_rho_partial); S(grad_pop_phi_partial);
        S2(dQ_dbw_dom_real, nu); S2(dQ_dbw_dom_imag, nu);
        #undef S
        #undef S2
    }

    *oQ = 0; memset(oP, 0, ne * 8);
    memset(ogck_r, 0, nw * 8); memset(ogck_i, 0, nw * 8);
    memset(ogm0, 0, nu * 8); memset(ogg0, 0, ng * 8);
    memset(ogsc, 0, N_SCALAR * sizeof(double));

    // Zero reduction output buffers (stale from previous call)
    cudaMemset(s.g_bw_real, 0, bs * nu * 8);
    cudaMemset(s.g_bw_imag, 0, bs * nu * 8);
    cudaMemset(s.g_interp_real, 0, bs * ng * 8);
    cudaMemset(s.g_interp_imag, 0, bs * ng * 8);

    double* Ph = (double*)malloc(bs * 8);
    double* gck_buf = (double*)malloc(nw * 8);
    double* gm0_buf = (double*)malloc(nu * 8);
    double* gg0_buf = (double*)malloc(ng * 8);

    for (int b = 0; b < nbat; b++) {
        int st = b * bs;
        int nb = (ne - st > bs) ? bs : (ne - st);

        ComputeData d = s;
        d.mass = h->m + st * h->nm;
        d.frac = h->f + st; d.time = h->t + st;
        d.weight = h->w + st; d.bkg = h->b + st;
        d.n_events = nb;
        // cuda_v3_ampcache: rebase the per-handle cache by absolute event
        // offset st (never a per-batch offset)
        d.amp_cache = (h->amp_cache != NULL) ? (h->amp_cache + (size_t)st * c->n_uniq) : NULL;

        launch_compute_all(c, &d, &p, nv, use_norm);

        // Accumulate profile timings (disabled by default, enabled via cuda_enable_profile_v3)
        if (c->n_profile > 1 && c->pt_enabled) {
            cudaEventSynchronize(c->pe[c->n_profile - 1]);
            float ms;
            for (int pi = 0; pi < c->n_profile - 1; pi++) {
                cudaEventElapsedTime(&ms, c->pe[pi], c->pe[pi+1]);
                c->pt[pi] += ms;
            }
        }

        // Clear any pending errors from launch_compute_all
        cudaGetLastError();

        // CPU sum for Q (reliable, no stale-buffer edge case)
        cudaMemcpy(Ph, d.Q_out, nb * 8, cudaMemcpyDeviceToHost);
        for (int i = 0; i < nb; i++) *oQ += Ph[i];

        cudaMemcpy(Ph, d.P_out, nb * 8, cudaMemcpyDeviceToHost);
        memcpy(oP + st, Ph, nb * 8);

        // GPU reductions: sum per-event gradients across events
        launch_reduce_sum_features(d.grad_ck_real_partial, s.g_bw_real, nb, nw);
        cudaMemcpy(gck_buf, s.g_bw_real, nw * 8, cudaMemcpyDeviceToHost);
        for (int j = 0; j < nw; j++) ogck_r[j] += gck_buf[j];

        launch_reduce_sum_features(d.grad_ck_imag_partial, s.g_bw_imag, nb, nw);
        cudaMemcpy(gck_buf, s.g_bw_imag, nw * 8, cudaMemcpyDeviceToHost);
        for (int j = 0; j < nw; j++) ogck_i[j] += gck_buf[j];

        launch_reduce_sum_features(d.grad_m0_partial, s.g_interp_real, nb, nu);
        cudaMemcpy(gm0_buf, s.g_interp_real, nu * 8, cudaMemcpyDeviceToHost);
        for (int j = 0; j < nu; j++) ogm0[j] += gm0_buf[j];

        launch_reduce_sum_features(d.grad_g0_partial, s.g_interp_imag, nb, ng);
        cudaMemcpy(gg0_buf, s.g_interp_imag, ng * 8, cudaMemcpyDeviceToHost);
        for (int j = 0; j < ng; j++) ogg0[j] += gg0_buf[j];

        // Scalar gradients: download per-event and sum on CPU
        #define SA(f, idx) do { \
            double* bf = (double*)malloc(nb * 8); \
            cudaMemcpy(bf, d.f, nb * 8, cudaMemcpyDeviceToHost); \
            for (int i = 0; i < nb; i++) ogsc[idx] += bf[i]; \
            free(bf); \
        } while(0)
        SA(grad_Gamma_partial,0); SA(grad_DeltaGamma_partial,1);
        SA(grad_DeltaM_partial,2); SA(grad_Ap_partial,3);
        SA(grad_poq_rho_partial,4); SA(grad_pop_phi_partial,5);
        #undef SA
    }

    // Free scratch (only if allocated per-call, not from context)
    if (!c->scratch) {
        #define F(p) do { if(s.p) cudaFree(s.p); } while(0)
        F(g_interp_real); F(g_interp_imag); F(g_bw_real); F(g_bw_imag);
        F(Q_out); F(P_out); F(pap_real); F(pap_imag); F(pam_real); F(pam_imag);
        F(gp_real); F(gp_imag); F(gm_real); F(gm_imag); F(poq_real); F(poq_imag);
        F(bw_p_real); F(bw_p_imag); F(common_amp_factor_real); F(common_amp_factor_imag);
        F(ap_real); F(ap_imag); F(am_real); F(am_imag); F(dQ_dP);
        F(bw_dom_real); F(bw_dom_imag);
        F(grad_ck_real_partial); F(grad_ck_imag_partial);
        F(grad_m0_partial); F(grad_g0_partial);
        F(grad_Gamma_partial); F(grad_DeltaGamma_partial); F(grad_DeltaM_partial);
        F(grad_Ap_partial);         F(grad_poq_rho_partial); F(grad_pop_phi_partial);
        F(dQ_dbw_dom_real); F(dQ_dbw_dom_imag);
        #undef F
    }
    // Q_red_gpu is allocated in context, freed in cuda_free_context_v3
    free(Ph); free(gck_buf); free(gm0_buf); free(gg0_buf);

    cudaFree((void*)p.ck_real); cudaFree((void*)p.ck_imag);
    cudaFree((void*)p.m0); cudaFree((void*)p.g0);
}

} // extern "C"
