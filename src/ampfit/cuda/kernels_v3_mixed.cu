/**
 * OPTIMIZED CUDA kernels v3 — mixed‑precision (f32 bulk compute, f64 time+scalars)
 *
 * Based on kernels_v3_f32.cu.
 *
 * Strategy:
 *   - Bulk compute (g_bw, bw_p, fa, fl, common_amp, angle factors, tree reduction)
 *     stays in float (f32) for maximum kernel throughput.
 *   - Time evolution (exponential of complex matrix with Gamma/Delta_m) uses
 *     double (f64) for precision.
 *   - Scalar gradients (Gamma, Delta_Gamma, Delta_m, A_p, poq_rho, pop_phi)
 *     are computed in double for precision.
 *   - gp/gm/poq and dQ_dP are stored as double* (intermediate buffers between
 *     forward and gradient kernels).
 *   - All other intermediate buffers (pap/pam, bw_p, common_amp, ap/am,
 *     bw_dom, g_interp, g_bw) remain float*.
 *   - ck/m0/g0 parameters remain float* (uploaded as float).
 *   - ck/m0/g0 gradients remain float*.
 *
 * Key optimizations inherited from f32 version:
 *   1. g_bw computation: parallelized across gamma_rows using shared memory
 *   2. Main compute: ka_prod in shared memory, wave-level parallelism
 *   3. g0 gradient: pre-compute dQ_dbw_dom, then matrix-vector multiply
 *   4. Eliminated heap allocation (new double[])
 *   5. __restrict__ pointers for better compiler optimization
 *   6. Block size optimized for RTX 3070 Ti (compute 8.6)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/complex.h>
#include <cstdio>
#include <cmath>

using complex = thrust::complex<float>;
using complex_d = thrust::complex<double>;

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

// ── Catmull-Rom interpolation helpers (float) ──────────────────────────────
__device__ float catmull_rom_1d(
    float pm1, float p0, float p1, float p2, float t
) {
    float t2 = t * t;
    float t3 = t2 * t;
    return 0.5f * (
        (2.0f * p0)
        + (-pm1 + p1) * t
        + (2.0f * pm1 - 5.0f * p0 + 4.0f * p1 - p2) * t2
        + (-pm1 + 3.0f * p0 - 3.0f * p1 + p2) * t3
    );
}

// Complex Catmull-Rom for gamma table
__device__ complex interp_complex_device(
    const float* __restrict__ table_real,
    const float* __restrict__ table_imag,
    int type_idx, float x,
    float xmin, float xdelta, int n_bins
) {
    float diff = (x - xmin) / xdelta;
    int xbin = max(0, min((int)floorf(diff), n_bins - 2));
    float t = max(0.0f, min(diff - xbin, 1.0f));
    int base = type_idx * n_bins + xbin;
    int type_end = (type_idx + 1) * n_bins - 1;
    int type_start = type_idx * n_bins;
    int im1 = max(type_start, base - 1);
    int i0  = base;
    int i1  = min(base + 1, type_end);
    int i2  = min(base + 2, type_end);
    int jm1 = (xbin == 0) ? i0 : im1;
    int j2  = (xbin == n_bins - 2) ? i1 : i2;
    float real_val = catmull_rom_1d(
        table_real[jm1], table_real[i0], table_real[i1], table_real[j2], t);
    float imag_val = catmull_rom_1d(
        table_imag[jm1], table_imag[i0], table_imag[i1], table_imag[j2], t);
    return complex(real_val, imag_val);
}

// Real Catmull-Rom for FL factor
__device__ float interp_real_device(
    const float* __restrict__ table,
    int type_idx, float x,
    float xmin, float xdelta, int n_bins
) {
    float diff = (x - xmin) / xdelta;
    int xbin = max(0, min((int)floorf(diff), n_bins - 2));
    float t = max(0.0f, min(diff - xbin, 1.0f));
    int base = type_idx * n_bins + xbin;
    int type_end = (type_idx + 1) * n_bins - 1;
    int type_start = type_idx * n_bins;
    int im1 = max(type_start, base - 1);
    int i0  = base;
    int i1  = min(base + 1, type_end);
    int i2  = min(base + 2, type_end);
    int jm1 = (xbin == 0) ? i0 : im1;
    int j2  = (xbin == n_bins - 2) ? i1 : i2;
    return catmull_rom_1d(
        table[jm1], table[i0], table[i1], table[j2], t);
}

//=============================================================================
// KERNEL 1: g_bw computation (parallelized within each event) — f32
//=============================================================================
__global__ void compute_g_bw_kernel(
    const float* __restrict__ mass,
    const float* __restrict__ g0,
    const int* __restrict__ g0_index,
    const int* __restrict__ g0_mass_index,
    const float* __restrict__ matrix_gamma,
    const float* __restrict__ gamma_table_real,
    const float* __restrict__ gamma_table_imag,
    float gamma_min, float gamma_delta,
    int n_gamma_rows, int n_unique_bw, int n_mass, int gamma_table_bins,
    float* __restrict__ g_interp_real,
    float* __restrict__ g_interp_imag,
    float* __restrict__ g_bw_real,
    float* __restrict__ g_bw_imag,
    int n_events
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;

    extern __shared__ float s_g_dyn[];
    float* s_g_real = s_g_dyn;
    float* s_g_imag = s_g_dyn + n_gamma_rows;

    for (int gamma_idx = tid; gamma_idx < n_gamma_rows; gamma_idx += block_sz) {
        int g0_idx = g0_index[gamma_idx];
        float g0_val = g0[g0_idx];
        float mass_val = mass[event_idx * n_mass + g0_mass_index[gamma_idx]];

        complex g_interp = interp_complex_device(
            gamma_table_real, gamma_table_imag,
            g0_idx, mass_val,
            gamma_min, gamma_delta, gamma_table_bins
        );

        g_interp_real[event_idx * n_gamma_rows + gamma_idx] = g_interp.real();
        g_interp_imag[event_idx * n_gamma_rows + gamma_idx] = g_interp.imag();

        complex g_val = g0_val * g_interp;
        s_g_real[gamma_idx] = g_val.real();
        s_g_imag[gamma_idx] = g_val.imag();
    }
    __syncthreads();

    int cols_per_thread = (n_unique_bw + block_sz - 1) / block_sz;
    int col_start = tid * cols_per_thread;
    int col_end = min(col_start + cols_per_thread, n_unique_bw);

    for (int c = col_start; c < col_end; c++) {
        float sum_r = 0.0f, sum_i = 0.0f;
        for (int i = 0; i < n_gamma_rows; i++) {
            float mg = matrix_gamma[i * n_unique_bw + c];
            sum_r += s_g_real[i] * mg;
            sum_i += s_g_imag[i] * mg;
        }
        g_bw_real[event_idx * n_unique_bw + c] = sum_r;
        g_bw_imag[event_idx * n_unique_bw + c] = sum_i;
    }
}

//=============================================================================
// KERNEL 2a: Gram‑matrix common factor — f32 (kept same as f32 version)
//=============================================================================
__global__ void gram_common_kernel_v3_mixed(
    const float* __restrict__ mass,
    const float* __restrict__ momentum,
    const float* __restrict__ angle,
    const float* __restrict__ weight,
    const int* __restrict__ m0_index,
    const int* __restrict__ fl_type,
    const int* __restrict__ mass_index,
    const int* __restrict__ fl_q_index,
    const int* __restrict__ bw_order,
    const int* __restrict__ fl_order,
    const int* __restrict__ angle_index,
    const float* __restrict__ angle_k,
    const float* __restrict__ angle_b,
    const float* __restrict__ matrix_angle_real,
    const float* __restrict__ matrix_angle_imag,
    const float* __restrict__ g_bw_real,
    const float* __restrict__ g_bw_imag,
    const float* __restrict__ fl_table,
    float fl_min, float fl_delta,
    int n_wave, int n_res, int n_decay, int n_unique_bw,
    int n_mass, int n_momentum, int n_angle_k, int n_angle_total, int n_angle_comp,
    int fl_table_bins,
    const float* __restrict__ m0,
    float* __restrict__ A0_real, float* __restrict__ A0_imag,
    float* __restrict__ A1_real, float* __restrict__ A1_imag,
    int n_events
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;
    int n = n_wave / 2;
    int ng = n_wave / 8;

    extern __shared__ float s_sh[];
    float* s_ka = s_sh;
    float* s_grp = s_sh + n_angle_k;

    for (int k_idx = tid; k_idx < n_angle_k; k_idx += block_sz) {
        int angle_pos = angle_index[k_idx];
        float ka = 1.0f;
        for (int comp = 0; comp < n_angle_comp; comp++) {
            int idx = event_idx * n_angle_total * n_angle_comp
                      + angle_pos * n_angle_comp + comp;
            ka *= cosf(angle[idx] * angle_k[k_idx * n_angle_comp + comp]
                       + angle_b[k_idx * n_angle_comp + comp]);
        }
        s_ka[k_idx] = ka;
    }
    __syncthreads();

    if (tid < 4 * ng) s_grp[tid] = 0.0f;
    __syncthreads();

    int waves_per_thread = (n_wave + block_sz - 1) / block_sz;
    int wave_start = tid * waves_per_thread;
    int wave_end = min(wave_start + waves_per_thread, n_wave);

    for (int w = wave_start; w < wave_end; w++) {
        complex bw_p(1.0f, 0.0f);
        for (int r = 0; r < n_res; r++) {
            int bw_idx = bw_order[w * n_res + r];
            float m0v = m0[m0_index[bw_idx]];
            float mv = mass[event_idx * n_mass + mass_index[bw_idx]];
            complex gbw(g_bw_real[event_idx * n_unique_bw + bw_idx],
                        g_bw_imag[event_idx * n_unique_bw + bw_idx]);
            complex dom(m0v * m0v - mv * mv + m0v * gbw.imag(),
                        -m0v * gbw.real());
            bw_p *= dom;
        }

        complex fa_val(0.0f, 0.0f);
        for (int k = 0; k < n_angle_k; k++) {
            int idx = k * n_wave + w;
            fa_val += s_ka[k] * complex(matrix_angle_real[idx], matrix_angle_imag[idx]);
        }
        complex common_amp = complex(1.0f, 0.0f) / bw_p * fa_val;

        float fl = 1.0f;
        for (int d = 0; d < n_decay; d++) {
            int fl_idx = fl_order[w * n_decay + d];
            float q = momentum[event_idx * n_momentum + fl_q_index[fl_idx]];
            fl *= interp_real_device(fl_table, fl_type[fl_idx],
                                     q, fl_min, fl_delta, fl_table_bins);
        }
        common_amp *= fl;

        bool is_B0 = w < n;
        int g = is_B0 ? w % ng : (w - n) % ng;
        int slab_offset = is_B0 ? 0 : (2 * ng);
        atomicAdd(&s_grp[slab_offset + g], common_amp.real());
        atomicAdd(&s_grp[slab_offset + ng + g], common_amp.imag());
    }
    __syncthreads();

    if (tid < ng) {
        A0_real[event_idx * ng + tid] = s_grp[0 * ng + tid];
        A0_imag[event_idx * ng + tid] = s_grp[1 * ng + tid];
        A1_real[event_idx * ng + tid] = s_grp[2 * ng + tid];
        A1_imag[event_idx * ng + tid] = s_grp[3 * ng + tid];
    }
}

//=============================================================================
// KERNEL 2b: Gram‑matrix reduction — f32 (kept same as f32 version)
//=============================================================================
__global__ void gram_reduce_kernel_v3_mixed(
    const float* __restrict__ A0_real, const float* __restrict__ A0_imag,
    const float* __restrict__ A1_real, const float* __restrict__ A1_imag,
    const float* __restrict__ weight,
    int n_events, int ng,
    float* __restrict__ Mpp_r, float* __restrict__ Mpp_i,
    float* __restrict__ Mmm_r, float* __restrict__ Mmm_i,
    float* __restrict__ Mpm_r, float* __restrict__ Mpm_i
) {
    int gi = blockIdx.x;
    int gj = blockIdx.y;
    if (gi >= ng || gj >= ng) return;

    float sum_pp_r = 0.0f, sum_pp_i = 0.0f;
    float sum_mm_r = 0.0f, sum_mm_i = 0.0f;
    float sum_pm_r = 0.0f, sum_pm_i = 0.0f;

    for (int e = 0; e < n_events; e++) {
        float w = weight[e];
        int base = e * ng;
        float a0ri = A0_real[base + gi], a0ii = A0_imag[base + gi];
        float a0rj = A0_real[base + gj], a0ij = A0_imag[base + gj];
        sum_pp_r += w * (a0ri * a0rj + a0ii * a0ij);
        sum_pp_i += w * (a0ri * a0ij - a0ii * a0rj);

        float a1ri = A1_real[base + gi], a1ii = A1_imag[base + gi];
        float a1rj = A1_real[base + gj], a1ij = A1_imag[base + gj];
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
// KERNEL 2c: Main forward computation
//   - Phases 1-3: f32 (ka_prod, bw_p, fa, fl, common_amp, ap/am tree reduce)
//   - Phase 4:    f64 (time evolution gp/gm/poq, probability, Q/dQ_dP)
//=============================================================================
__global__ void compute_main_kernel(
    const float* __restrict__ mass,
    const float* __restrict__ momentum,
    const float* __restrict__ angle,
    const float* __restrict__ frac,
    const float* __restrict__ time,
    const float* __restrict__ weight,
    const float* __restrict__ bkg,
    const int* __restrict__ m0_index,
    const int* __restrict__ fl_type,
    const int* __restrict__ mass_index,
    const int* __restrict__ fl_q_index,
    const int* __restrict__ bw_order,
    const int* __restrict__ fl_order,
    const int* __restrict__ angle_index,
    const float* __restrict__ angle_k,
    const float* __restrict__ angle_b,
    const float* __restrict__ matrix_angle_real,
    const float* __restrict__ matrix_angle_imag,
    const float* __restrict__ g_bw_real,
    const float* __restrict__ g_bw_imag,
    const float* __restrict__ fl_table,
    float fl_min, float fl_delta,
    int n_wave, int n_res, int n_decay, int n_unique_bw,
    int n_mass, int n_momentum, int n_angle_k, int n_angle_total, int n_angle_comp,
    int fl_table_bins,
    const float* __restrict__ ck_real,
    const float* __restrict__ ck_imag,
    const float* __restrict__ m0,
    // f64 scalar params
    double Gamma, double Delta_Gamma, double Delta_m,
    double A_p, double poq_rho, double pop_phi,
    // Output buffers
    double* __restrict__ Q_out,
    double* __restrict__ P_out,
    float* __restrict__ pap_real, float* __restrict__ pap_imag,
    float* __restrict__ pam_real, float* __restrict__ pam_imag,
    // f64 buffers for gp/gm/poq/dQ_dP
    double* __restrict__ gp_real, double* __restrict__ gp_imag,
    double* __restrict__ gm_real, double* __restrict__ gm_imag,
    double* __restrict__ poq_real, double* __restrict__ poq_imag,
    // f32 buffers for intermediate values
    float* __restrict__ bw_p_real, float* __restrict__ bw_p_imag,
    float* __restrict__ common_amp_factor_real,
    float* __restrict__ common_amp_factor_imag,
    float* __restrict__ ap_real, float* __restrict__ ap_imag,
    float* __restrict__ am_real, float* __restrict__ am_imag,
    double* __restrict__ dQ_dP,
    float* __restrict__ bw_dom_real, float* __restrict__ bw_dom_imag,
    int n_events, int use_norm, double norm
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;

    //=========================================================================
    // Phase 1: Compute ka_prod (f32)
    //=========================================================================
    extern __shared__ float s_ka_prod[];

    for (int k_idx = tid; k_idx < n_angle_k; k_idx += block_sz) {
        int angle_pos = angle_index[k_idx];
        float ka_prod = 1.0f;
        for (int comp = 0; comp < n_angle_comp; comp++) {
            int angle_idx = event_idx * n_angle_total * n_angle_comp + angle_pos * n_angle_comp + comp;
            float k_val = angle_k[k_idx * n_angle_comp + comp];
            float b_val = angle_b[k_idx * n_angle_comp + comp];
            ka_prod *= cosf(angle[angle_idx] * k_val + b_val);
        }
        s_ka_prod[k_idx] = ka_prod;
    }
    __syncthreads();

    //=========================================================================
    // Phase 2: Compute bw_p, fa, fl_factor, common_amp (f32)
    //=========================================================================
    int waves_per_thread = (n_wave + block_sz - 1) / block_sz;
    int wave_start = tid * waves_per_thread;
    int wave_end = min(wave_start + waves_per_thread, n_wave);

    for (int wave_idx = wave_start; wave_idx < wave_end; wave_idx++) {
        // --- bw_p = product of bw_dom over resonances (f32) ---
        complex bw_p_val(1.0f, 0.0f);
        for (int res_idx = 0; res_idx < n_res; res_idx++) {
            int bw_idx = bw_order[wave_idx * n_res + res_idx];
            float m0_val = m0[m0_index[bw_idx]];
            float mass_val = mass[event_idx * n_mass + mass_index[bw_idx]];
            float m0_sq = m0_val * m0_val;
            float mass_sq = mass_val * mass_val;

            complex g_bw_val(
                g_bw_real[event_idx * n_unique_bw + bw_idx],
                g_bw_imag[event_idx * n_unique_bw + bw_idx]
            );

            // bw_dom = m0² - m² + m0*g_bw.imag  -  1j*m0*g_bw.real
            complex bw_dom(m0_sq - mass_sq + m0_val * g_bw_val.imag(),
                           -m0_val * g_bw_val.real());

            bw_p_val *= bw_dom;

            bw_dom_real[event_idx * n_unique_bw + bw_idx] = bw_dom.real();
            bw_dom_imag[event_idx * n_unique_bw + bw_idx] = bw_dom.imag();
        }
        bw_p_real[event_idx * n_wave + wave_idx] = bw_p_val.real();
        bw_p_imag[event_idx * n_wave + wave_idx] = bw_p_val.imag();

        // --- fa = dot(ka_prod, matrix_angle_row) (f32) ---
        complex fa(0.0f, 0.0f);
        for (int k_idx = 0; k_idx < n_angle_k; k_idx++) {
            int idx = k_idx * n_wave + wave_idx;
            fa += s_ka_prod[k_idx] * complex(
                matrix_angle_real[idx], matrix_angle_imag[idx]);
        }

        // --- FL factor (real interpolation, f32) ---
        float fl_p = 1.0f;
        for (int decay_idx = 0; decay_idx < n_decay; decay_idx++) {
            int fl_idx = fl_order[wave_idx * n_decay + decay_idx];
            float fl_q_val = momentum[event_idx * n_momentum + fl_q_index[fl_idx]];
            fl_p *= interp_real_device(fl_table, fl_type[fl_idx],
                                       fl_q_val, fl_min, fl_delta, fl_table_bins);
        }

        // --- common_amp = (1/bw_p) * fa * fl_p (f32) ---
        complex one_over_bw = complex(1.0f, 0.0f) / bw_p_val;
        complex common_amp = one_over_bw * fa * fl_p;
        common_amp_factor_real[event_idx * n_wave + wave_idx] = common_amp.real();
        common_amp_factor_imag[event_idx * n_wave + wave_idx] = common_amp.imag();
    }
    __syncthreads();

    //=========================================================================
    // Phase 3: Compute amplitudes ap, am (f32 tree reduction)
    //=========================================================================
    int n_wave_half = n_wave / 2;
    float ap_sum_r = 0.0f, ap_sum_i = 0.0f;
    float am_sum_r = 0.0f, am_sum_i = 0.0f;

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

    // Tree reduction (f32)
    __shared__ float s_ap_r[256];
    __shared__ float s_ap_i[256];
    __shared__ float s_am_r[256];
    __shared__ float s_am_i[256];

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
        complex ap_f(s_ap_r[0], s_ap_i[0]);
        complex am_f(s_am_r[0], s_am_i[0]);
        ap_real[event_idx] = ap_f.real();
        ap_imag[event_idx] = ap_f.imag();
        am_real[event_idx] = am_f.real();
        am_imag[event_idx] = am_f.imag();

        //=========================================================================
        // Phase 4: Time evolution and probability (f64, thread 0 only)
        //=========================================================================
        double t = (double)time[event_idx];
        complex_d i_const(0.0, 1.0);
        complex_d eL = exp(-i_const * t * complex_d(-Delta_m / 2.0, -(Gamma + Delta_Gamma / 2.0) / 2.0));
        complex_d eH = exp(-i_const * t * complex_d(Delta_m / 2.0, -(Gamma - Delta_Gamma / 2.0) / 2.0));
        complex_d gp = (eL + eH) / 2.0;
        complex_d gm = (eL - eH) / 2.0;

        gp_real[event_idx] = gp.real();
        gp_imag[event_idx] = gp.imag();
        gm_real[event_idx] = gm.real();
        gm_imag[event_idx] = gm.imag();

        complex_d poq = poq_rho * exp(i_const * pop_phi);
        poq_real[event_idx] = poq.real();
        poq_imag[event_idx] = poq.imag();

        // Compute pap/pam in double, then store as float
        complex_d ap_d((double)ap_f.real(), (double)ap_f.imag());
        complex_d am_d((double)am_f.real(), (double)am_f.imag());
        complex_d pap_d = ap_d * gp + am_d * gm * poq;
        complex_d pam_d = am_d * gp + ap_d * gm / poq;

        pap_real[event_idx] = (float)pap_d.real();
        pap_imag[event_idx] = (float)pap_d.imag();
        pam_real[event_idx] = (float)pam_d.real();
        pam_imag[event_idx] = (float)pam_d.imag();

        double pb = thrust::norm(pap_d);
        double pbbar = thrust::norm(pam_d);
        double frac_val = (double)frac[event_idx];
        double P = frac_val * pb * (1.0 - A_p) + (1.0 - frac_val) * pbbar * (1.0 + A_p);
        P_out[event_idx] = P;

        double weight_val = (double)weight[event_idx];
        double bkg_val = (double)bkg[event_idx];

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
// KERNEL 3: Optimized gradient computation
//   - Phases 1-3: f32 (ck/m0/g0 gradients)
//   - Phase 4:    f64 (scalar gradients)
//=============================================================================
__global__ void gradient_kernel(
    const double* __restrict__ P,
    const float* __restrict__ pap_real, const float* __restrict__ pap_imag,
    const float* __restrict__ pam_real, const float* __restrict__ pam_imag,
    // gp/gm/poq loaded as double* (computed in f64 by forward kernel)
    const double* __restrict__ gp_real, const double* __restrict__ gp_imag,
    const double* __restrict__ gm_real, const double* __restrict__ gm_imag,
    const double* __restrict__ poq_real, const double* __restrict__ poq_imag,
    // f32 intermediate buffers from forward pass
    const float* __restrict__ bw_p_real, const float* __restrict__ bw_p_imag,
    const float* __restrict__ common_amp_factor_real,
    const float* __restrict__ common_amp_factor_imag,
    const float* __restrict__ ap_real, const float* __restrict__ ap_imag,
    const float* __restrict__ am_real, const float* __restrict__ am_imag,
    // dQ_dP is double* (computed in f64 by forward kernel)
    const double* __restrict__ dQ_dP,
    const float* __restrict__ bw_dom_real, const float* __restrict__ bw_dom_imag,
    const float* __restrict__ g_interp_real, const float* __restrict__ g_interp_imag,
    const float* __restrict__ g_bw_real, const float* __restrict__ g_bw_imag,
    const float* __restrict__ frac, const float* __restrict__ time,
    const int* __restrict__ m0_index, const int* __restrict__ g0_index,
    const int* __restrict__ bw_order,
    const float* __restrict__ matrix_gamma,
    // f32 params
    const float* __restrict__ m0, const float* __restrict__ g0,
    const float* __restrict__ ck_real, const float* __restrict__ ck_imag,
    // f64 scalar params
    double Gamma, double Delta_Gamma, double Delta_m,
    double A_p, double poq_rho, double pop_phi,
    int n_wave, int n_res, int n_unique_bw, int n_gamma_rows, int n_mass,
    // Output: ck/m0/g0 gradients in f32
    float* __restrict__ grad_ck_real_partial,
    float* __restrict__ grad_ck_imag_partial,
    float* __restrict__ grad_m0_partial,
    float* __restrict__ grad_g0_partial,
    // Output: scalar gradients in f64
    double* __restrict__ grad_Gamma_partial,
    double* __restrict__ grad_DeltaGamma_partial,
    double* __restrict__ grad_DeltaM_partial,
    double* __restrict__ grad_Ap_partial,
    double* __restrict__ grad_poq_rho_partial,
    double* __restrict__ grad_pop_phi_partial,
    int n_events
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;

    // Load forward outputs (f32) — pap/pam/ap/am are float*
    complex pap(pap_real[event_idx], pap_imag[event_idx]);
    complex pam(pam_real[event_idx], pam_imag[event_idx]);
    complex ap(ap_real[event_idx], ap_imag[event_idx]);
    complex am(am_real[event_idx], am_imag[event_idx]);

    // Load gp/gm/poq as double (from f64 forward output)
    complex_d gp_d(gp_real[event_idx], gp_imag[event_idx]);
    complex_d gm_d(gm_real[event_idx], gm_imag[event_idx]);
    complex_d poq_d(poq_real[event_idx], poq_imag[event_idx]);

    // Cast down to f32 for phases 1-3
    complex gp_f = complex((float)gp_d.real(), (float)gp_d.imag());
    complex gm_f = complex((float)gm_d.real(), (float)gm_d.imag());
    complex poq_f = complex((float)poq_d.real(), (float)poq_d.imag());

    double dQ_dP_val = dQ_dP[event_idx];   // f64
    float dQ_dP_f = (float)dQ_dP_val;      // f32 for float phases

    float frac_val = frac[event_idx];

    //=========================================================================
    // Common intermediate values (needed in both f32 and f64 paths)
    //=========================================================================
    // f32 versions for phases 1-3
    float pb_f = thrust::norm(pap);
    float pbbar_f = thrust::norm(pam);
    float dP_dpb_f = frac_val * (1.0f - (float)A_p);
    float dP_dpbbar_f = (1.0f - frac_val) * (1.0f + (float)A_p);
    float dP_dAp_f = -frac_val * pb_f + (1.0f - frac_val) * pbbar_f;
    grad_Ap_partial[event_idx] = (double)dQ_dP_f * (double)dP_dAp_f;

    float dQ_dpb_f = dQ_dP_f * dP_dpb_f;
    float dQ_dpbbar_f = dQ_dP_f * dP_dpbbar_f;

    complex d_pb_dap_f = conj(pap) * gp_f;
    complex d_pb_dam_f = conj(pap) * gm_f * poq_f;
    complex d_pbbar_dap_f = conj(pam) * (gm_f / poq_f);
    complex d_pbbar_dam_f = conj(pam) * gp_f;

    complex dQ_dap_val_f = dQ_dpb_f * d_pb_dap_f + dQ_dpbbar_f * d_pbbar_dap_f;
    complex dQ_dam_val_f = dQ_dpb_f * d_pb_dam_f + dQ_dpbbar_f * d_pbbar_dam_f;

    //=========================================================================
    // Phase 1: ck gradients (f32)
    //=========================================================================
    int n_wave_half = n_wave / 2;

    for (int i = tid; i < n_wave_half; i += block_sz) {
        complex common_i(
            common_amp_factor_real[event_idx * n_wave + i],
            common_amp_factor_imag[event_idx * n_wave + i]);
        complex grad_ck = dQ_dap_val_f * common_i;
        grad_ck_real_partial[event_idx * n_wave + i] = grad_ck.real();
        grad_ck_imag_partial[event_idx * n_wave + i] = grad_ck.imag();

        complex common_j(
            common_amp_factor_real[event_idx * n_wave + n_wave_half + i],
            common_amp_factor_imag[event_idx * n_wave + n_wave_half + i]);
        complex grad_ck2 = dQ_dam_val_f * common_j;
        grad_ck_real_partial[event_idx * n_wave + n_wave_half + i] = grad_ck2.real();
        grad_ck_imag_partial[event_idx * n_wave + n_wave_half + i] = grad_ck2.imag();
    }

    //=========================================================================
    // Phase 2: Pre-compute dQ_dbw_dom for all unique_bw (f32)
    //=========================================================================
    extern __shared__ float s_grad_dyn[];
    float* s_dQ_dbw_dom_real = s_grad_dyn;
    float* s_dQ_dbw_dom_imag = s_grad_dyn + n_unique_bw;
    float* s_dQ_dg_bw_real    = s_grad_dyn + 2 * n_unique_bw;
    float* s_dQ_dg_bw_imag    = s_grad_dyn + 3 * n_unique_bw;

    for (int bw_idx = tid; bw_idx < n_unique_bw; bw_idx += block_sz) {
        grad_m0_partial[event_idx * n_unique_bw + bw_idx] = 0.0f;
        s_dQ_dbw_dom_real[bw_idx] = 0.0f;
        s_dQ_dbw_dom_imag[bw_idx] = 0.0f;
    }
    __syncthreads();

    for (int wave_idx = tid; wave_idx < n_wave; wave_idx += block_sz) {
        complex bw_p_val(bw_p_real[event_idx * n_wave + wave_idx],
                         bw_p_imag[event_idx * n_wave + wave_idx]);
        complex one_over_bw = complex(1.0f, 0.0f) / bw_p_val;
        complex common_amp(
            common_amp_factor_real[event_idx * n_wave + wave_idx],
            common_amp_factor_imag[event_idx * n_wave + wave_idx]);
        complex dQ_da = (wave_idx < n_wave_half) ? dQ_dap_val_f : dQ_dam_val_f;
        complex ck(ck_real[wave_idx], ck_imag[wave_idx]);
        complex dQ_dbw_p = dQ_da * (-ck * one_over_bw * common_amp);

        for (int res_idx = 0; res_idx < n_res; res_idx++) {
            int bw_idx = bw_order[wave_idx * n_res + res_idx];

            complex bw_dom_val(
                bw_dom_real[event_idx * n_unique_bw + bw_idx],
                bw_dom_imag[event_idx * n_unique_bw + bw_idx]);
            complex dQ_dbw_dom_contrib = dQ_dbw_p * (bw_p_val / bw_dom_val);

            atomicAdd(&s_dQ_dbw_dom_real[bw_idx], dQ_dbw_dom_contrib.real());
            atomicAdd(&s_dQ_dbw_dom_imag[bw_idx], dQ_dbw_dom_contrib.imag());

            float m0_val = m0[m0_index[bw_idx]];
            complex g_bw_val(g_bw_real[event_idx * n_unique_bw + bw_idx],
                             g_bw_imag[event_idx * n_unique_bw + bw_idx]);
            complex dbw_dom_dm0 = complex(2.0f * m0_val, 0.0f) - complex(0.0f, 1.0f) * g_bw_val;
            atomicAdd(&grad_m0_partial[event_idx * n_unique_bw + bw_idx],
                2.0f * (dQ_dbw_dom_contrib * dbw_dom_dm0).real());
        }
    }
    __syncthreads();

    // Convert dQ_dbw_dom to dQ_dg_bw = dQ_dbw_dom * (-1j * m0)
    if (tid < n_unique_bw) {
        float m0_val = m0[m0_index[tid]];
        complex dm0(s_dQ_dbw_dom_real[tid], s_dQ_dbw_dom_imag[tid]);
        complex dg_bw = dm0 * complex(0.0f, -m0_val);
        s_dQ_dg_bw_real[tid] = dg_bw.real();
        s_dQ_dg_bw_imag[tid] = dg_bw.imag();
    }
    __syncthreads();

    //=========================================================================
    // Phase 3: g0 gradient via matrix-vector multiply (f32)
    //=========================================================================
    for (int gamma_idx = tid; gamma_idx < n_gamma_rows; gamma_idx += block_sz) {
        float sum_r = 0.0f, sum_i = 0.0f;
        for (int bw_idx = 0; bw_idx < n_unique_bw; bw_idx++) {
            float mg = matrix_gamma[gamma_idx * n_unique_bw + bw_idx];
            sum_r += s_dQ_dg_bw_real[bw_idx] * mg;
            sum_i += s_dQ_dg_bw_imag[bw_idx] * mg;
        }

        complex dQ_dg_val(sum_r, sum_i);
        complex g_interp_val(
            g_interp_real[event_idx * n_gamma_rows + gamma_idx],
            g_interp_imag[event_idx * n_gamma_rows + gamma_idx]);

        grad_g0_partial[event_idx * n_gamma_rows + gamma_idx] =
            2.0f * (dQ_dg_val * g_interp_val).real();
    }

    //=========================================================================
    // Phase 4: Time evolution gradients (f64, thread 0 only)
    //=========================================================================
    if (tid == 0) {
        // Cast pap/pam/ap/am to double for f64 computation
        complex_d pap_d((double)pap.real(), (double)pap.imag());
        complex_d pam_d((double)pam.real(), (double)pam.imag());
        complex_d ap_d((double)ap.real(), (double)ap.imag());
        complex_d am_d((double)am.real(), (double)am.imag());

        double t_d = (double)time[event_idx];
        double frac_d = (double)frac_val;

        double pb_d = thrust::norm(pap_d);
        double pbbar_d = thrust::norm(pam_d);
        double dP_dpb_d = frac_d * (1.0 - A_p);
        double dP_dpbbar_d = (1.0 - frac_d) * (1.0 + A_p);
        // recompute dQ_dpb/dQ_dpbbar in double for scalar grads
        double dQ_dpb_d = dQ_dP_val * dP_dpb_d;
        double dQ_dpbbar_d = dQ_dP_val * dP_dpbbar_d;

        // d_pb/dgp etc in double
        complex_d d_pb_dgp = conj(pap_d) * ap_d;
        complex_d d_pb_dgm = conj(pap_d) * poq_d * am_d;
        complex_d d_pbbar_dgp = conj(pam_d) * am_d;
        complex_d d_pbbar_dgm = conj(pam_d) * ap_d / poq_d;

        complex_d dQ_dgp = dQ_dpb_d * d_pb_dgp + dQ_dpbbar_d * d_pbbar_dgp;
        complex_d dQ_dgm = dQ_dpb_d * d_pb_dgm + dQ_dpbbar_d * d_pbbar_dgm;

        complex_d dgp_dGamma = (-t_d / 2.0) * gp_d;
        complex_d dgm_dGamma = (-t_d / 2.0) * gm_d;
        complex_d dgp_dDeltaGamma = (-t_d / 4.0) * gm_d;
        complex_d dgm_dDeltaGamma = (-t_d / 4.0) * gp_d;
        complex_d dgp_dDeltaM = complex_d(0.0, t_d / 2.0) * gm_d;
        complex_d dgm_dDeltaM = complex_d(0.0, t_d / 2.0) * gp_d;

        grad_Gamma_partial[event_idx] = 2.0 * (dQ_dgp * dgp_dGamma + dQ_dgm * dgm_dGamma).real();
        grad_DeltaGamma_partial[event_idx] = 2.0 * (dQ_dgp * dgp_dDeltaGamma + dQ_dgm * dgm_dDeltaGamma).real();
        grad_DeltaM_partial[event_idx] = 2.0 * (dQ_dgp * dgp_dDeltaM + dQ_dgm * dgm_dDeltaM).real();

        complex_d d_pb_dpoq = conj(pap_d) * gm_d * am_d;
        complex_d d_pbbar_dpoq = conj(pam_d) * (-gm_d / (poq_d * poq_d)) * ap_d;
        complex_d dQ_dpoq = dQ_dpb_d * d_pb_dpoq + dQ_dpbbar_d * d_pbbar_dpoq;

        complex_d exp_phi = exp(complex_d(0.0, 1.0) * pop_phi);
        grad_poq_rho_partial[event_idx] = 2.0 * (dQ_dpoq * exp_phi).real();
        grad_pop_phi_partial[event_idx] = 2.0 * (dQ_dpoq * poq_rho * complex_d(0.0, 1.0) * exp_phi).real();
    }
}

//=============================================================================
// Reduction kernels (f32, for ck/m0/g0 gradients)
//=============================================================================
__global__ void reduce_sum_kernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, sdata[0]);
}

__global__ void reduce_sum_complex_kernel(
    const float* real_in, const float* imag_in,
    float* real_out, float* imag_out, int n
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float* sreal = sdata;
    float* simag = sdata + blockDim.x;
    sreal[tid] = (idx < n) ? real_in[idx] : 0.0f;
    simag[tid] = (idx < n) ? imag_in[idx] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { sreal[tid] += sreal[tid + s]; simag[tid] += simag[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { atomicAdd(real_out, sreal[0]); atomicAdd(imag_out, simag[0]); }
}

__global__ void reduce_sum_features_kernel(
    const float* input, float* output,
    int n_events, int n_features
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int feat = blockIdx.x;
    if (feat >= n_features) return;
    float sum = 0.0f;
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
    const float* real_in, const float* imag_in,
    float* real_out, float* imag_out,
    int n_events, int n_features
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int feat = blockIdx.x;
    if (feat >= n_features) return;
    float real_sum = 0.0f, imag_sum = 0.0f;
    for (int i = tid; i < n_events; i += blockDim.x) {
        real_sum += real_in[feat + i * n_features];
        imag_sum += imag_in[feat + i * n_features];
    }
    float* sreal = sdata;
    float* simag = sdata + blockDim.x;
    sreal[tid] = real_sum;
    simag[tid] = imag_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { sreal[tid] += sreal[tid + s]; simag[tid] += simag[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { real_out[feat] = sreal[0]; imag_out[feat] = simag[0]; }
}

//=============================================================================
// Host-callable launch functions
//=============================================================================
extern "C" {

// ── Structs for clean unified API ──
// Bulk intermediates: float*.  Time-evolution / scalar: double*.

typedef struct {
    // Event data (GPU) — all float
    const float* mass; const float* momentum; const float* angle;
    const float* frac; const float* time; const float* weight; const float* bkg;
    // Scratch buffers (GPU) — float (bulk compute)
    float* g_interp_real; float* g_interp_imag;
    float* g_bw_real; float* g_bw_imag;
    double* Q_out; double* P_out;
    float* pap_real; float* pap_imag; float* pam_real; float* pam_imag;
    // Time-evolution buffers — double (f64 precision)
    double* gp_real; double* gp_imag; double* gm_real; double* gm_imag;
    double* poq_real; double* poq_imag;
    // Float intermediate buffers
    float* bw_p_real; float* bw_p_imag;
    float* common_amp_factor_real; float* common_amp_factor_imag;
    float* ap_real; float* ap_imag; float* am_real; float* am_imag;
    // dQ_dP is double (f64 from forward)
    double* dQ_dP;
    float* bw_dom_real; float* bw_dom_imag;
    // Gradient buffers: ck/m0/g0 in float, scalars in double
    float* grad_ck_real_partial; float* grad_ck_imag_partial;
    float* grad_m0_partial; float* grad_g0_partial;
    double* grad_Gamma_partial; double* grad_DeltaGamma_partial;
    double* grad_DeltaM_partial; double* grad_Ap_partial;
    double* grad_poq_rho_partial; double* grad_pop_phi_partial;
    int n_events;
} ComputeData;

typedef struct {
    // Index arrays (GPU)
    const int* m0_index; const int* g0_index;
    const int* g0_mass_index; const int* mass_index;
    const int* fl_type; const int* fl_q_index;
    const int* bw_order; const int* fl_order; const int* angle_index;
    // Constant arrays (GPU) — float
    const float* angle_k; const float* angle_b;
    const float* matrix_angle_real; const float* matrix_angle_imag;
    const float* gamma_table_real; const float* gamma_table_imag;
    float gamma_min; float gamma_delta; int gamma_table_bins;
    const float* matrix_gamma;
    const float* fl_table; float fl_min; float fl_delta; int fl_table_bins;
    // Dimensions
    int n_wave; int n_res; int n_decay; int n_unique_bw;
    int n_gamma_rows; int n_mass; int n_momentum;
    int n_angle_k; int n_angle_total; int n_angle_comp;
    int batch_size;
    int n_m0_params;
    int n_g0_params;
    ComputeData* scratch;
    float* Q_red_gpu;
} ComputeContext;

typedef struct {
    const float* ck_real; const float* ck_imag;
    const float* m0; const float* g0;
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

// ── Launch wrappers ──

void launch_compute_g_bw(
    const float* mass, const float* g0,
    const int* g0_index, const int* g0_mass_index,
    const float* matrix_gamma,
    const float* gamma_table_real, const float* gamma_table_imag,
    float gamma_min, float gamma_delta,
    int n_gamma_rows, int n_unique_bw, int n_mass, int gamma_table_bins,
    float* g_interp_real, float* g_interp_imag,
    float* g_bw_real, float* g_bw_imag,
    int n_events) {

    size_t shmem = 2 * n_gamma_rows * sizeof(float);
    compute_g_bw_kernel<<<n_events, BLOCK_SIZE, shmem>>>(
        mass, g0, g0_index, g0_mass_index, matrix_gamma,
        gamma_table_real, gamma_table_imag,
        gamma_min, gamma_delta,
        n_gamma_rows, n_unique_bw, n_mass, gamma_table_bins,
        g_interp_real, g_interp_imag,
        g_bw_real, g_bw_imag, n_events);
}

void launch_compute_main(
    const float* mass, const float* momentum, const float* angle,
    const float* frac, const float* time, const float* weight, const float* bkg,
    const int* m0_index, const int* fl_type,
    const int* mass_index, const int* fl_q_index,
    const int* bw_order, const int* fl_order, const int* angle_index,
    const float* angle_k, const float* angle_b,
    const float* matrix_angle_real, const float* matrix_angle_imag,
    const float* g_bw_real, const float* g_bw_imag,
    const float* fl_table,
    float fl_min, float fl_delta,
    int n_wave, int n_res, int n_decay, int n_unique_bw,
    int n_mass, int n_momentum, int n_angle_k, int n_angle_total, int n_angle_comp,
    int fl_table_bins,
    const float* ck_real, const float* ck_imag, const float* m0,
    // f64 scalars
    double Gamma, double Delta_Gamma, double Delta_m,
    double A_p, double poq_rho, double pop_phi,
    // Output buffers
    double* Q_out, double* P_out,
    float* pap_real, float* pap_imag, float* pam_real, float* pam_imag,
    double* gp_real, double* gp_imag, double* gm_real, double* gm_imag,
    double* poq_real, double* poq_imag,
    float* bw_p_real, float* bw_p_imag,
    float* common_amp_factor_real, float* common_amp_factor_imag,
    float* ap_real, float* ap_imag, float* am_real, float* am_imag,
    double* dQ_dP,
    float* bw_dom_real, float* bw_dom_imag,
    int n_events, int use_norm, double norm) {

    compute_main_kernel<<<n_events, BLOCK_SIZE, n_angle_k * sizeof(float)>>>(
        mass, momentum, angle, frac, time, weight, bkg,
        m0_index, fl_type, mass_index, fl_q_index,
        bw_order, fl_order, angle_index,
        angle_k, angle_b, matrix_angle_real, matrix_angle_imag,
        g_bw_real, g_bw_imag, fl_table,
        fl_min, fl_delta,
        n_wave, n_res, n_decay, n_unique_bw,
        n_mass, n_momentum, n_angle_k, n_angle_total, n_angle_comp,
        fl_table_bins,
        ck_real, ck_imag, m0,
        Gamma, Delta_Gamma, Delta_m, A_p, poq_rho, pop_phi,
        Q_out, P_out,
        pap_real, pap_imag, pam_real, pam_imag,
        gp_real, gp_imag, gm_real, gm_imag,
        poq_real, poq_imag,
        bw_p_real, bw_p_imag,
        common_amp_factor_real, common_amp_factor_imag,
        ap_real, ap_imag, am_real, am_imag, dQ_dP,
        bw_dom_real, bw_dom_imag,
        n_events, use_norm, norm);
}

void launch_gradient(
    const double* P,
    const float* pap_real, const float* pap_imag,
    const float* pam_real, const float* pam_imag,
    const double* gp_real, const double* gp_imag,
    const double* gm_real, const double* gm_imag,
    const double* poq_real, const double* poq_imag,
    const float* bw_p_real, const float* bw_p_imag,
    const float* common_amp_factor_real, const float* common_amp_factor_imag,
    const float* ap_real, const float* ap_imag,
    const float* am_real, const float* am_imag,
    const double* dQ_dP,
    const float* bw_dom_real, const float* bw_dom_imag,
    const float* g_interp_real, const float* g_interp_imag,
    const float* g_bw_real, const float* g_bw_imag,
    const float* frac, const float* time,
    const int* m0_index, const int* g0_index, const int* bw_order,
    const float* matrix_gamma,
    const float* m0, const float* g0, const float* ck_real, const float* ck_imag,
    // f64 scalars
    double Gamma, double Delta_Gamma, double Delta_m,
    double A_p, double poq_rho, double pop_phi,
    int n_wave, int n_res, int n_unique_bw, int n_gamma_rows, int n_mass,
    // Output: f32 grads
    float* grad_ck_real_partial, float* grad_ck_imag_partial,
    float* grad_m0_partial, float* grad_g0_partial,
    // Output: f64 scalar grads
    double* grad_Gamma_partial, double* grad_DeltaGamma_partial,
    double* grad_DeltaM_partial, double* grad_Ap_partial,
    double* grad_poq_rho_partial, double* grad_pop_phi_partial,
    int n_events) {

    size_t shmem_grad = 4 * n_unique_bw * sizeof(float);
    gradient_kernel<<<n_events, BLOCK_SIZE, shmem_grad>>>(
        P, pap_real, pap_imag, pam_real, pam_imag,
        gp_real, gp_imag, gm_real, gm_imag, poq_real, poq_imag,
        bw_p_real, bw_p_imag, common_amp_factor_real, common_amp_factor_imag,
        ap_real, ap_imag, am_real, am_imag, dQ_dP,
        bw_dom_real, bw_dom_imag, g_interp_real, g_interp_imag,
        g_bw_real, g_bw_imag,
        frac, time,
        m0_index, g0_index, bw_order, matrix_gamma,
        m0, g0, ck_real, ck_imag,
        Gamma, Delta_Gamma, Delta_m, A_p, poq_rho, pop_phi,
        n_wave, n_res, n_unique_bw, n_gamma_rows, n_mass,
        grad_ck_real_partial, grad_ck_imag_partial,
        grad_m0_partial, grad_g0_partial,
        grad_Gamma_partial, grad_DeltaGamma_partial,
        grad_DeltaM_partial, grad_Ap_partial,
        grad_poq_rho_partial, grad_pop_phi_partial,
        n_events);
}

// ── Reduction launch wrappers (f32) ──

void launch_reduce_sum(const float* input, float* output, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_sum_kernel<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(input, output, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_reduce_sum_complex(const float* real_in, const float* imag_in,
    float* real_out, float* imag_out, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_sum_complex_kernel<<<grid_size, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float)>>>(
        real_in, imag_in, real_out, imag_out, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_reduce_sum_features(const float* input, float* output,
    int n_events, int n_features) {
    reduce_sum_features_kernel<<<n_features, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        input, output, n_events, n_features);
}

void launch_reduce_sum_complex_features(const float* real_in, const float* imag_in,
    float* real_out, float* imag_out,
    int n_events, int n_features) {
    reduce_sum_complex_features_kernel<<<n_features, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float)>>>(
        real_in, imag_in, real_out, imag_out, n_events, n_features);
    CUDA_CHECK(cudaGetLastError());
}

// ── Gram matrix kernel launches (f32, same as f32 version) ──

void launch_gram_common_v3_mixed(
    const ComputeContext* ctx, ComputeData* data,
    const float* m0,
    float* A0_real, float* A0_imag,
    float* A1_real, float* A1_imag) {

    int ng = ctx->n_wave / 8;
    int n_events = data->n_events;
    int shmem = (ctx->n_angle_k + 4 * ng) * sizeof(float);
    gram_common_kernel_v3_mixed<<<n_events, BLOCK_SIZE, shmem>>>(
        data->mass, data->momentum, data->angle, data->weight,
        ctx->m0_index, ctx->fl_type,
        ctx->mass_index, ctx->fl_q_index,
        ctx->bw_order, ctx->fl_order, ctx->angle_index,
        ctx->angle_k, ctx->angle_b,
        ctx->matrix_angle_real, ctx->matrix_angle_imag,
        data->g_bw_real, data->g_bw_imag,
        ctx->fl_table, ctx->fl_min, ctx->fl_delta,
        ctx->n_wave, ctx->n_res, ctx->n_decay, ctx->n_unique_bw,
        ctx->n_mass, ctx->n_momentum, ctx->n_angle_k, ctx->n_angle_total, ctx->n_angle_comp,
        ctx->fl_table_bins, m0,
        A0_real, A0_imag, A1_real, A1_imag, n_events);
}

void launch_gram_reduce_v3_mixed(
    const float* A0_real, const float* A0_imag,
    const float* A1_real, const float* A1_imag,
    const float* weight,
    int n_events, int ng,
    float* Mpp_r, float* Mpp_i,
    float* Mmm_r, float* Mmm_i,
    float* Mpm_r, float* Mpm_i) {

    dim3 grid(ng, ng);
    gram_reduce_kernel_v3_mixed<<<grid, 1>>>(
        A0_real, A0_imag, A1_real, A1_imag, weight,
        n_events, ng,
        Mpp_r, Mpp_i, Mmm_r, Mmm_i, Mpm_r, Mpm_i);
}

// ── Unified launch: (Context*, Data*, Params*, norm, use_norm) ──

void launch_compute_all(
    const ComputeContext* ctx, ComputeData* data,
    const ComputeParams* params, double norm, int use_norm
) {
    launch_compute_g_bw(
        data->mass, params->g0, ctx->g0_index, ctx->g0_mass_index,
        ctx->matrix_gamma,
        ctx->gamma_table_real, ctx->gamma_table_imag,
        ctx->gamma_min, ctx->gamma_delta,
        ctx->n_gamma_rows, ctx->n_unique_bw, ctx->n_mass, ctx->gamma_table_bins,
        data->g_interp_real, data->g_interp_imag,
        data->g_bw_real, data->g_bw_imag,
        data->n_events);

    launch_compute_main(
        data->mass, data->momentum, data->angle,
        data->frac, data->time, data->weight, data->bkg,
        ctx->m0_index, ctx->fl_type,
        ctx->mass_index, ctx->fl_q_index,
        ctx->bw_order, ctx->fl_order, ctx->angle_index,
        ctx->angle_k, ctx->angle_b,
        ctx->matrix_angle_real, ctx->matrix_angle_imag,
        data->g_bw_real, data->g_bw_imag,
        ctx->fl_table, ctx->fl_min, ctx->fl_delta,
        ctx->n_wave, ctx->n_res, ctx->n_decay, ctx->n_unique_bw,
        ctx->n_mass, ctx->n_momentum, ctx->n_angle_k, ctx->n_angle_total, ctx->n_angle_comp,
        ctx->fl_table_bins,
        params->ck_real, params->ck_imag, params->m0,
        params->Gamma, params->Delta_Gamma, params->Delta_m,
        params->A_prod, params->poq_rho, params->pop_phi,
        data->Q_out, data->P_out,
        data->pap_real, data->pap_imag, data->pam_real, data->pam_imag,
        data->gp_real, data->gp_imag, data->gm_real, data->gm_imag,
        data->poq_real, data->poq_imag,
        data->bw_p_real, data->bw_p_imag,
        data->common_amp_factor_real, data->common_amp_factor_imag,
        data->ap_real, data->ap_imag, data->am_real, data->am_imag, data->dQ_dP,
        data->bw_dom_real, data->bw_dom_imag,
        data->n_events, use_norm, norm);

    launch_gradient(
        data->P_out,
        data->pap_real, data->pap_imag, data->pam_real, data->pam_imag,
        data->gp_real, data->gp_imag, data->gm_real, data->gm_imag,
        data->poq_real, data->poq_imag,
        data->bw_p_real, data->bw_p_imag,
        data->common_amp_factor_real, data->common_amp_factor_imag,
        data->ap_real, data->ap_imag, data->am_real, data->am_imag, data->dQ_dP,
        data->bw_dom_real, data->bw_dom_imag,
        data->g_interp_real, data->g_interp_imag,
        data->g_bw_real, data->g_bw_imag,
        data->frac, data->time,
        ctx->m0_index, ctx->g0_index, ctx->bw_order, ctx->matrix_gamma,
        params->m0, params->g0, params->ck_real, params->ck_imag,
        params->Gamma, params->Delta_Gamma, params->Delta_m,
        params->A_prod, params->poq_rho, params->pop_phi,
        ctx->n_wave, ctx->n_res, ctx->n_unique_bw, ctx->n_gamma_rows, ctx->n_mass,
        data->grad_ck_real_partial, data->grad_ck_imag_partial,
        data->grad_m0_partial, data->grad_g0_partial,
        data->grad_Gamma_partial, data->grad_DeltaGamma_partial,
        data->grad_DeltaM_partial, data->grad_Ap_partial,
        data->grad_poq_rho_partial, data->grad_pop_phi_partial,
        data->n_events);
    // Flush gradient kernel errors
    cudaGetLastError();
}

// ── Upload helpers (plain C) ──
static void* _up_int(const int* src, int n) {
    int* d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, src, n * sizeof(int), cudaMemcpyHostToDevice)); return d;
}
static void* _up_flt(const float* src, int n) {
    float* d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, src, n * sizeof(float), cudaMemcpyHostToDevice)); return d;
}

// ════════════════════════════════════════════════════════════════════════
//  v3_mixed API — mixed-precision (f32 bulk, f64 time+scalars)
// ════════════════════════════════════════════════════════════════════════

typedef struct { const float* m; const float* mo; const float* a;
    const float* f; const float* t; const float* w; const float* b;
    int ne; int nm; int nmom; int nat; int nac;
} DataHandle2;

void* cuda_create_context_v3_mixed(
    const int* m0_i,int n1, const int* g0_i,int n2,
    const int* g0_m,int n3, const int* mass_i,int n4,
    const int* fl_t,int n5, const int* fl_q,int n6,
    const int* bw_o,int n7, const int* fl_o,int n8,
    const int* ang_i,int n9,
    const float* ak,int n10, const float* ab,int n11,
    const float* mar,int n12, const float* mai,int n13,
    const float* gtr,int n14, const float* gti,int n15,
    float gmin,float gdel,int gbins,
    const float* mg,int n16,
    const float* ft,int n17, float flmin,float fldel,int fbins,
    int nw,int nr,int nd,int nub,int ngr,
    int nm,int nmom,int nak_,int nat,int nac,
    int n_m0p, int n_g0p,
    int batch_size
) {
    ComputeContext* c = (ComputeContext*)calloc(1, sizeof(ComputeContext));
    c->m0_index = (int*)_up_int(m0_i, n1); c->g0_index = (int*)_up_int(g0_i, n2);
    c->g0_mass_index = (int*)_up_int(g0_m, n3); c->mass_index = (int*)_up_int(mass_i, n4);
    c->fl_type = (int*)_up_int(fl_t, n5); c->fl_q_index = (int*)_up_int(fl_q, n6);
    c->bw_order = (int*)_up_int(bw_o, n7); c->fl_order = (int*)_up_int(fl_o, n8);
    c->angle_index = (int*)_up_int(ang_i, n9);
    c->angle_k = (float*)_up_flt(ak, n10); c->angle_b = (float*)_up_flt(ab, n11);
    c->matrix_angle_real = (float*)_up_flt(mar, n12); c->matrix_angle_imag = (float*)_up_flt(mai, n13);
    c->gamma_table_real = (float*)_up_flt(gtr, n14); c->gamma_table_imag = (float*)_up_flt(gti, n15);
    c->gamma_min = gmin; c->gamma_delta = gdel; c->gamma_table_bins = gbins;
    c->matrix_gamma = (float*)_up_flt(mg, n16);
    c->fl_table = (float*)_up_flt(ft, n17); c->fl_min = flmin; c->fl_delta = fldel; c->fl_table_bins = fbins;
    c->n_wave = nw; c->n_res = nr; c->n_decay = nd;
    c->n_unique_bw = nub; c->n_gamma_rows = ngr;
    c->n_mass = nm; c->n_momentum = nmom; c->n_angle_k = nak_; c->n_angle_total = nat; c->n_angle_comp = nac;
    c->n_m0_params = n_m0p; c->n_g0_params = n_g0p;
    c->batch_size = batch_size > 0 ? batch_size : DEFAULT_BATCH_SIZE;

    // Pre-allocate scratch buffers when batch_size is known
    if (c->batch_size > 0) {
        int bs = c->batch_size;
        c->scratch = (ComputeData*)calloc(1, sizeof(ComputeData));
        // Float buffers (bulk compute)
        CUDA_CHECK(cudaMalloc(&c->scratch->g_interp_real, bs * ngr * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c->scratch->g_interp_imag, bs * ngr * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c->scratch->g_bw_real, bs * nub * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c->scratch->g_bw_imag, bs * nub * sizeof(float)));
        // Double output buffers
        CUDA_CHECK(cudaMalloc(&c->scratch->Q_out, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c->scratch->P_out, bs * sizeof(double)));
        // Float amplitude buffers
        CUDA_CHECK(cudaMalloc(&c->scratch->pap_real, bs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c->scratch->pap_imag, bs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c->scratch->pam_real, bs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c->scratch->pam_imag, bs * sizeof(float)));
        // Double time-evolution buffers
        CUDA_CHECK(cudaMalloc(&c->scratch->gp_real, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c->scratch->gp_imag, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c->scratch->gm_real, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c->scratch->gm_imag, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c->scratch->poq_real, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c->scratch->poq_imag, bs * sizeof(double)));
        // Float intermediate buffers
        CUDA_CHECK(cudaMalloc(&c->scratch->bw_p_real, bs * nw * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c->scratch->bw_p_imag, bs * nw * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c->scratch->common_amp_factor_real, bs * nw * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c->scratch->common_amp_factor_imag, bs * nw * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c->scratch->ap_real, bs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c->scratch->ap_imag, bs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c->scratch->am_real, bs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c->scratch->am_imag, bs * sizeof(float)));
        // Double dQ_dP
        CUDA_CHECK(cudaMalloc(&c->scratch->dQ_dP, bs * sizeof(double)));
        // Float bw_dom
        CUDA_CHECK(cudaMalloc(&c->scratch->bw_dom_real, bs * nub * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c->scratch->bw_dom_imag, bs * nub * sizeof(float)));
        // Float ck/m0/g0 gradient buffers
        CUDA_CHECK(cudaMalloc(&c->scratch->grad_ck_real_partial, bs * nw * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c->scratch->grad_ck_imag_partial, bs * nw * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c->scratch->grad_m0_partial, bs * nub * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c->scratch->grad_g0_partial, bs * ngr * sizeof(float)));
        // Double scalar gradient buffers
        CUDA_CHECK(cudaMalloc(&c->scratch->grad_Gamma_partial, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c->scratch->grad_DeltaGamma_partial, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c->scratch->grad_DeltaM_partial, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c->scratch->grad_Ap_partial, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c->scratch->grad_poq_rho_partial, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c->scratch->grad_pop_phi_partial, bs * sizeof(double)));

        CUDA_CHECK(cudaMalloc(&c->Q_red_gpu, 4));
    } else {
        c->scratch = NULL;
        c->Q_red_gpu = NULL;
    }
    return c;
}

void cuda_free_context_v3_mixed(void* vctx) {
    ComputeContext* c = (ComputeContext*)vctx;
    #define F(p) cudaFree((void*)c->p)
    F(m0_index); F(g0_index); F(g0_mass_index); F(mass_index);
    F(fl_type); F(fl_q_index); F(bw_order); F(fl_order); F(angle_index);
    F(angle_k); F(angle_b); F(matrix_angle_real); F(matrix_angle_imag);
    F(gamma_table_real); F(gamma_table_imag); F(matrix_gamma); F(fl_table);
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
        #undef SF
        free(c->scratch);
    }
    if (c->Q_red_gpu) cudaFree(c->Q_red_gpu);
    free(c);
}

void* cuda_load_data_v3_mixed(void* vctx,
    const float* mass,int nmass, const float* mom,int nmom,
    const float* ang,int nang, const float* frac,const float* time,
    const float* wgt,const float* bkg,int ne
) {
    DataHandle2* h = (DataHandle2*)calloc(1, sizeof(DataHandle2));
    ComputeContext* c_ctx = (ComputeContext*)vctx;
    int nac = c_ctx ? c_ctx->n_angle_comp : 3;
    h->m = (const float*)_up_flt(mass, ne * nmass);
    h->mo = (const float*)_up_flt(mom, ne * nmom);
    h->a = (const float*)_up_flt(ang, ne * nang * nac);
    h->f = (const float*)_up_flt(frac, ne);
    h->t = (const float*)_up_flt(time, ne);
    h->w = (const float*)_up_flt(wgt, ne);
    h->b = (const float*)_up_flt(bkg, ne);
    h->ne = ne; h->nm = nmass; h->nmom = nmom; h->nat = nang; h->nac = nac;
    return h;
}

void cuda_free_data_v3_mixed(void* vh) {
    DataHandle2* h = (DataHandle2*)vh;
    cudaFree((void*)h->m); cudaFree((void*)h->mo); cudaFree((void*)h->a);
    cudaFree((void*)h->f); cudaFree((void*)h->t); cudaFree((void*)h->w); cudaFree((void*)h->b);
    free(h);
}

void cuda_gram_matrix_v3_mixed(void* vctx, void* vdh,
    const float* m0, const float* g0,
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

    const float* gpu_m0 = (const float*)_up_flt(m0, c->n_m0_params);
    const float* gpu_g0 = (const float*)_up_flt(g0, c->n_g0_params);

    ComputeData s;
    if (c->scratch) {
        s = *c->scratch;
    } else {
        memset(&s, 0, sizeof(ComputeData));
        CUDA_CHECK(cudaMalloc(&s.g_interp_real, bs * ng * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.g_interp_imag, bs * ng * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.g_bw_real, bs * nu * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.g_bw_imag, bs * nu * sizeof(float)));
    }

    float *A0r, *A0i, *A1r, *A1i;
    size_t a_sz = (size_t)bs * ng2 * sizeof(float);
    CUDA_CHECK(cudaMalloc(&A0r, a_sz)); CUDA_CHECK(cudaMalloc(&A0i, a_sz));
    CUDA_CHECK(cudaMalloc(&A1r, a_sz)); CUDA_CHECK(cudaMalloc(&A1i, a_sz));

    size_t g_sz = (size_t)ng2 * ng2 * sizeof(float);
    float *Mpp_r, *Mpp_i, *Mmm_r, *Mmm_i, *Mpm_r, *Mpm_i;
    CUDA_CHECK(cudaMalloc(&Mpp_r, g_sz)); CUDA_CHECK(cudaMalloc(&Mpp_i, g_sz));
    CUDA_CHECK(cudaMalloc(&Mmm_r, g_sz)); CUDA_CHECK(cudaMalloc(&Mmm_i, g_sz));
    CUDA_CHECK(cudaMalloc(&Mpm_r, g_sz)); CUDA_CHECK(cudaMalloc(&Mpm_i, g_sz));

    memset(oMpp_r, 0, g_sz * 2); memset(oMpp_i, 0, g_sz * 2);
    memset(oMmm_r, 0, g_sz * 2); memset(oMmm_i, 0, g_sz * 2);
    memset(oMpm_r, 0, g_sz * 2); memset(oMpm_i, 0, g_sz * 2);

    float* hbuf = (float*)malloc(g_sz);

    for (int b = 0; b < nbat; b++) {
        int st = b * bs;
        int nb = (ne - st > bs) ? bs : (ne - st);

        ComputeData d = s;
        d.mass = h->m + st * h->nm;
        d.momentum = h->mo + st * h->nmom;
        d.angle = h->a + st * h->nat * h->nac;
        d.weight = h->w + st;
        d.n_events = nb;

        cudaMemset(d.g_interp_real, 0, bs * ng * sizeof(float));
        cudaMemset(d.g_interp_imag, 0, bs * ng * sizeof(float));
        cudaMemset(d.g_bw_real, 0, bs * nu * sizeof(float));
        cudaMemset(d.g_bw_imag, 0, bs * nu * sizeof(float));

        launch_compute_g_bw(
            d.mass, gpu_g0, c->g0_index, c->g0_mass_index,
            c->matrix_gamma,
            c->gamma_table_real, c->gamma_table_imag,
            c->gamma_min, c->gamma_delta,
            c->n_gamma_rows, c->n_unique_bw, c->n_mass, c->gamma_table_bins,
            d.g_interp_real, d.g_interp_imag,
            d.g_bw_real, d.g_bw_imag, nb);
        CUDA_CHECK(cudaGetLastError());

        launch_gram_common_v3_mixed(c, &d, gpu_m0, A0r, A0i, A1r, A1i);
        CUDA_CHECK(cudaGetLastError());

        launch_gram_reduce_v3_mixed(
            A0r, A0i, A1r, A1i, d.weight, nb, ng2,
            Mpp_r, Mpp_i, Mmm_r, Mmm_i, Mpm_r, Mpm_i);
        CUDA_CHECK(cudaGetLastError());

        cudaMemcpy(hbuf, Mpp_r, g_sz, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < (size_t)ng2 * ng2; i++) oMpp_r[i] += (double)hbuf[i];
        cudaMemcpy(hbuf, Mpp_i, g_sz, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < (size_t)ng2 * ng2; i++) oMpp_i[i] += (double)hbuf[i];
        cudaMemcpy(hbuf, Mmm_r, g_sz, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < (size_t)ng2 * ng2; i++) oMmm_r[i] += (double)hbuf[i];
        cudaMemcpy(hbuf, Mmm_i, g_sz, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < (size_t)ng2 * ng2; i++) oMmm_i[i] += (double)hbuf[i];
        cudaMemcpy(hbuf, Mpm_r, g_sz, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < (size_t)ng2 * ng2; i++) oMpm_r[i] += (double)hbuf[i];
        cudaMemcpy(hbuf, Mpm_i, g_sz, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < (size_t)ng2 * ng2; i++) oMpm_i[i] += (double)hbuf[i];
    }

    free(hbuf);
    cudaFree((void*)gpu_m0); cudaFree((void*)gpu_g0);
    cudaFree(A0r); cudaFree(A0i); cudaFree(A1r); cudaFree(A1i);
    cudaFree(Mpp_r); cudaFree(Mpp_i); cudaFree(Mmm_r); cudaFree(Mmm_i);
    cudaFree(Mpm_r); cudaFree(Mpm_i);
    if (!c->scratch) {
        cudaFree(s.g_interp_real); cudaFree(s.g_interp_imag);
        cudaFree(s.g_bw_real); cudaFree(s.g_bw_imag);
    }
}

void cuda_compute_v3_mixed(void* vctx, void* vdh,
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
    int ne = h->ne, bs = c->batch_size;
    if (ne < bs) bs = ne;
    int nbat = (ne + bs - 1) / bs;
    int nw = c->n_wave, nu = c->n_unique_bw, ng = c->n_gamma_rows;

    // Convert f64 params → f32 for GPU upload
    ComputeParams p;
    {
        float* ckr_f = (float*)malloc(nw * sizeof(float));
        float* cki_f = (float*)malloc(nw * sizeof(float));
        float* m0_f = (float*)malloc(c->n_m0_params * sizeof(float));
        float* g0_f = (float*)malloc(c->n_g0_params * sizeof(float));
        for (int i = 0; i < nw; i++) { ckr_f[i] = (float)ck_r[i]; cki_f[i] = (float)ck_i[i]; }
        for (int i = 0; i < c->n_m0_params; i++) m0_f[i] = (float)m0[i];
        for (int i = 0; i < c->n_g0_params; i++) g0_f[i] = (float)g0[i];
        p.ck_real = (float*)_up_flt(ckr_f, nw);
        p.ck_imag = (float*)_up_flt(cki_f, nw);
        p.m0 = (float*)_up_flt(m0_f, c->n_m0_params);
        p.g0 = (float*)_up_flt(g0_f, c->n_g0_params);
        free(ckr_f); free(cki_f); free(m0_f); free(g0_f);
    }
    p.Gamma = G; p.Delta_Gamma = DG; p.Delta_m = DM;
    p.A_prod = Ap; p.poq_rho = pr; p.pop_phi = pp;

    // Use context-allocated scratch (avoids per-call cudaMalloc/free)
    ComputeData s;
    if (c->scratch) {
        s = *c->scratch;
    } else {
        memset(&s, 0, sizeof(ComputeData));
        // Float buffers (bulk compute)
        CUDA_CHECK(cudaMalloc(&s.g_interp_real, bs * ng * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.g_interp_imag, bs * ng * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.g_bw_real, bs * nu * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.g_bw_imag, bs * nu * sizeof(float)));
        // Double output buffers
        CUDA_CHECK(cudaMalloc(&s.Q_out, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.P_out, bs * sizeof(double)));
        // Float amplitude buffers
        CUDA_CHECK(cudaMalloc(&s.pap_real, bs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.pap_imag, bs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.pam_real, bs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.pam_imag, bs * sizeof(float)));
        // Double time-evolution buffers
        CUDA_CHECK(cudaMalloc(&s.gp_real, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.gp_imag, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.gm_real, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.gm_imag, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.poq_real, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.poq_imag, bs * sizeof(double)));
        // Float intermediate buffers
        CUDA_CHECK(cudaMalloc(&s.bw_p_real, bs * nw * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.bw_p_imag, bs * nw * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.common_amp_factor_real, bs * nw * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.common_amp_factor_imag, bs * nw * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.ap_real, bs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.ap_imag, bs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.am_real, bs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.am_imag, bs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.dQ_dP, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.bw_dom_real, bs * nu * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.bw_dom_imag, bs * nu * sizeof(float)));
        // Float gradient buffers
        CUDA_CHECK(cudaMalloc(&s.grad_ck_real_partial, bs * nw * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.grad_ck_imag_partial, bs * nw * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.grad_m0_partial, bs * nu * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.grad_g0_partial, bs * ng * sizeof(float)));
        // Double scalar gradient buffers
        CUDA_CHECK(cudaMalloc(&s.grad_Gamma_partial, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.grad_DeltaGamma_partial, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.grad_DeltaM_partial, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.grad_Ap_partial, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.grad_poq_rho_partial, bs * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.grad_pop_phi_partial, bs * sizeof(double)));
    }

    *oQ = 0.0; memset(oP, 0, ne * sizeof(double));
    memset(ogck_r, 0, nw * sizeof(double)); memset(ogck_i, 0, nw * sizeof(double));
    memset(ogm0, 0, nu * sizeof(double)); memset(ogg0, 0, ng * sizeof(double));
    memset(ogsc, 0, N_SCALAR * sizeof(double));

    // Zero reduction output buffers (stale from previous call)
    cudaMemset(s.g_bw_real, 0, bs * nu * sizeof(float));
    cudaMemset(s.g_bw_imag, 0, bs * nu * sizeof(float));
    cudaMemset(s.g_interp_real, 0, bs * ng * sizeof(float));
    cudaMemset(s.g_interp_imag, 0, bs * ng * sizeof(float));

    double* Ph = (double*)malloc(bs * sizeof(double));

    for (int b = 0; b < nbat; b++) {
        int st = b * bs;
        int nb = (ne - st > bs) ? bs : (ne - st);

        ComputeData d = s;
        d.mass = h->m + st * h->nm;
        d.momentum = h->mo + st * h->nmom;
        d.angle = h->a + st * h->nat * h->nac;
        d.frac = h->f + st; d.time = h->t + st;
        d.weight = h->w + st; d.bkg = h->b + st;
        d.n_events = nb;

        launch_compute_all(c, &d, &p, nv, use_norm);

        // Clear any pending errors from launch_compute_all
        cudaGetLastError();

        // CPU sum for Q (reliable, no stale-buffer edge case)
        cudaMemcpy(Ph, d.Q_out, nb * sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < nb; i++) *oQ += Ph[i];

        cudaMemcpy(Ph, d.P_out, nb * sizeof(double), cudaMemcpyDeviceToHost);
        memcpy(oP + st, Ph, nb * sizeof(double));

        // GPU reductions: sum per-event gradients across events (f32 for ck/m0/g0)
        launch_reduce_sum_features(d.grad_ck_real_partial, s.g_bw_real, nb, nw);
        {
            float* buf = (float*)malloc(nw * sizeof(float));
            cudaMemcpy(buf, s.g_bw_real, nw * sizeof(float), cudaMemcpyDeviceToHost);
            for (int j = 0; j < nw; j++) ogck_r[j] += (double)buf[j];
            free(buf);
        }

        launch_reduce_sum_features(d.grad_ck_imag_partial, s.g_bw_imag, nb, nw);
        {
            float* buf = (float*)malloc(nw * sizeof(float));
            cudaMemcpy(buf, s.g_bw_imag, nw * sizeof(float), cudaMemcpyDeviceToHost);
            for (int j = 0; j < nw; j++) ogck_i[j] += (double)buf[j];
            free(buf);
        }

        launch_reduce_sum_features(d.grad_m0_partial, s.g_interp_real, nb, nu);
        {
            float* buf = (float*)malloc(nu * sizeof(float));
            cudaMemcpy(buf, s.g_interp_real, nu * sizeof(float), cudaMemcpyDeviceToHost);
            for (int j = 0; j < nu; j++) ogm0[j] += (double)buf[j];
            free(buf);
        }

        launch_reduce_sum_features(d.grad_g0_partial, s.g_interp_imag, nb, ng);
        {
            float* buf = (float*)malloc(ng * sizeof(float));
            cudaMemcpy(buf, s.g_interp_imag, ng * sizeof(float), cudaMemcpyDeviceToHost);
            for (int j = 0; j < ng; j++) ogg0[j] += (double)buf[j];
            free(buf);
        }

        // Scalar gradients: download per-event and sum on CPU (f64)
        #define SA(f, idx) do { \
            double* bf = (double*)malloc(nb * sizeof(double)); \
            cudaMemcpy(bf, d.f, nb * sizeof(double), cudaMemcpyDeviceToHost); \
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
        F(grad_Ap_partial); F(grad_poq_rho_partial); F(grad_pop_phi_partial);
        #undef F
    }
    free(Ph);

    cudaFree((void*)p.ck_real); cudaFree((void*)p.ck_imag);
    cudaFree((void*)p.m0); cudaFree((void*)p.g0);
}

} // extern "C"
