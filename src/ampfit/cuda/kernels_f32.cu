/**
 * OPTIMIZED CUDA kernels - parallelized inner loops for maximum GPU utilization
 *
 * Key optimizations:
 * 1. g_bw computation: parallelized across gamma_rows using shared memory
 * 2. Main compute: ka_prod in shared memory, wave-level parallelism
 * 3. g0 gradient: pre-compute dQ_dbw_dom, then matrix-vector multiply
 * 4. Eliminated heap allocation (new float[])
 * 5. __restrict__ pointers for better compiler optimization
 * 6. Block size optimized for RTX 3070 Ti (compute 8.6)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/complex.h>
#include <cstdio>
#include <cmath>

using complex = thrust::complex<float>;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
        } \
    } while(0)

// Complex interpolation for gamma table
__device__ complex interp_complex_device(
    const float* __restrict__ table_real,
    const float* __restrict__ table_imag,
    int type_idx, float x,
    float xmin, float xdelta, int n_bins
) {
    float diff = (x - xmin) / xdelta;
    int xbin = max(0, min((int)floor(diff), n_bins - 2));
    float delta = diff - xbin;
    int left_idx = type_idx * n_bins + xbin;
    int right_idx = left_idx + 1;
    float left_real = table_real[left_idx];
    float right_real = table_real[right_idx];
    float left_imag = table_imag[left_idx];
    float right_imag = table_imag[right_idx];
    float real_val = (right_real - left_real) * delta + left_real;
    float imag_val = (right_imag - left_imag) * delta + left_imag;
    return complex(real_val, imag_val);
}

// Real interpolation for FL factor
__device__ float interp_real_device(
    const float* __restrict__ table,
    int type_idx, float x,
    float xmin, float xdelta, int n_bins
) {
    float diff = (x - xmin) / xdelta;
    int xbin = max(0, min((int)floor(diff), n_bins - 2));
    float delta = diff - xbin;
    int left_idx = type_idx * n_bins + xbin;
    return (table[left_idx + 1] - table[left_idx]) * delta + table[left_idx];
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

    // Shared memory for g values (all gamma rows)
    __shared__ float s_g_real[288];  // n_gamma_rows = 288
    __shared__ float s_g_imag[288];

    // Phase 1: Each thread computes g for its assigned gamma rows
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

    // Phase 2: Matrix-vector multiply: g_bw[j] = sum_i g[i] * matrix_gamma[i, j]
    // Each thread handles a contiguous block of unique_bw columns
    // Load matrix_gamma with coalesced access (consecutive threads read consecutive columns)
    int cols_per_thread = (n_unique_bw + block_sz - 1) / block_sz;
    int col_start = tid * cols_per_thread;
    int col_end = min(col_start + cols_per_thread, n_unique_bw);

    for (int c = col_start; c < col_end; c++) {
        float sum_r = 0.0, sum_i = 0.0;
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
// KERNEL 2: Main forward computation (bw_p, angular factors, amplitudes, prob)
//=============================================================================
// Each block handles one event.
// Shared memory: ka_prod for all angle_k values (336 floats)
// Each thread handles multiple waves for bw_p, fa, common_amp computations
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
    int n_mass, int n_momentum, int n_angle_k, int n_angle_total,
    int fl_table_bins,
    const float* __restrict__ ck_real,
    const float* __restrict__ ck_imag,
    const float* __restrict__ m0,
    float Gamma, float Delta_Gamma, float Delta_m,
    float A_p, float poq_rho, float pop_phi,
    float* __restrict__ Q_out,
    float* __restrict__ P_out,
    float* __restrict__ pap_real, float* __restrict__ pap_imag,
    float* __restrict__ pam_real, float* __restrict__ pam_imag,
    float* __restrict__ gp_real, float* __restrict__ gp_imag,
    float* __restrict__ gm_real, float* __restrict__ gm_imag,
    float* __restrict__ poq_real, float* __restrict__ poq_imag,
    float* __restrict__ bw_p_real, float* __restrict__ bw_p_imag,
    float* __restrict__ common_amp_factor_real,
    float* __restrict__ common_amp_factor_imag,
    float* __restrict__ ap_real, float* __restrict__ ap_imag,
    float* __restrict__ am_real, float* __restrict__ am_imag,
    float* __restrict__ dQ_dP,
    float* __restrict__ bw_dom_real, float* __restrict__ bw_dom_imag,
    int n_events, int use_norm, float norm
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;

    //=========================================================================
    // Phase 1: Compute ka_prod for all angle_k values (parallel across threads)
    //=========================================================================
    __shared__ float s_ka_prod[336];  // n_angle_k = 336

    for (int k_idx = tid; k_idx < n_angle_k; k_idx += block_sz) {
        int angle_pos = angle_index[k_idx];
        float ka_prod = 1.0;
        for (int comp = 0; comp < 3; comp++) {
            int angle_idx = event_idx * n_angle_total * 3 + angle_pos * 3 + comp;
            float k_val = angle_k[k_idx * 3 + comp];
            float b_val = angle_b[k_idx * 3 + comp];
            ka_prod *= cos(angle[angle_idx] * k_val + b_val);
        }
        s_ka_prod[k_idx] = ka_prod;
    }
    __syncthreads();

    //=========================================================================
    // Phase 2: Compute bw_p, fa, fl_factor, common_amp for each wave
    // Each thread handles its assigned subset of waves
    //=========================================================================
    int waves_per_thread = (n_wave + block_sz - 1) / block_sz;
    int wave_start = tid * waves_per_thread;
    int wave_end = min(wave_start + waves_per_thread, n_wave);

    for (int wave_idx = wave_start; wave_idx < wave_end; wave_idx++) {
        // --- bw_p = product of bw_dom over resonances ---
        complex bw_p_val(1.0, 0.0);
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

        // --- fa = dot(ka_prod, matrix_angle_row) ---
        complex fa(0.0, 0.0);
        for (int k_idx = 0; k_idx < n_angle_k; k_idx++) {
            int idx = k_idx * n_wave + wave_idx;
            fa += s_ka_prod[k_idx] * complex(
                matrix_angle_real[idx], matrix_angle_imag[idx]);
        }

        // --- FL factor (real interpolation) ---
        float fl_p = 1.0;
        for (int decay_idx = 0; decay_idx < n_decay; decay_idx++) {
            int fl_idx = fl_order[wave_idx * n_decay + decay_idx];
            float fl_q_val = momentum[event_idx * n_momentum + fl_q_index[fl_idx]];
            fl_p *= interp_real_device(fl_table, fl_type[fl_idx],
                                       fl_q_val, fl_min, fl_delta, fl_table_bins);
        }

        // --- common_amp = (1/bw_p) * fa * fl_p ---
        complex one_over_bw = complex(1.0, 0.0) / bw_p_val;
        complex common_amp = one_over_bw * fa * fl_p;
        common_amp_factor_real[event_idx * n_wave + wave_idx] = common_amp.real();
        common_amp_factor_imag[event_idx * n_wave + wave_idx] = common_amp.imag();
    }
    __syncthreads();

    //=========================================================================
    // Phase 3: Compute amplitudes ap, am (parallelized across threads)
    //=========================================================================
    int n_wave_half = n_wave / 2;
    float ap_sum_r = 0.0, ap_sum_i = 0.0;
    float am_sum_r = 0.0, am_sum_i = 0.0;

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

    // Reduce across threads within block
    __shared__ float s_ap_r[256]; // max block size
    __shared__ float s_ap_i[256];
    __shared__ float s_am_r[256];
    __shared__ float s_am_i[256];

    s_ap_r[tid] = ap_sum_r;
    s_ap_i[tid] = ap_sum_i;
    s_am_r[tid] = am_sum_r;
    s_am_i[tid] = am_sum_i;
    __syncthreads();

    // Tree reduction
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

        //=========================================================================
        // Phase 4: Time evolution and probability (scalar per event, thread 0 only)
        //=========================================================================
        float t = time[event_idx];
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

        float pb = thrust::norm(pap);
        float pbbar = thrust::norm(pam_val);
        float frac_val = frac[event_idx];
        float P = frac_val * pb * (1.0 - A_p) + (1.0 - frac_val) * pbbar * (1.0 + A_p);
        P_out[event_idx] = P;

        float weight_val = weight[event_idx];
        float bkg_val = bkg[event_idx];

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
//=============================================================================
// Each block handles one event.
// Key optimization for g0 gradient:
//   Instead of triple-nested loop (288 × 448 × 3 = 387K iterations),
//   we:
//   1. Pre-compute dQ_dbw_dom for all bw_idx (shared memory)
//   2. Convert to dQ_dg_bw[bw_idx] = dQ_dbw_dom * (-1j * m0)
//   3. Compute g0 gradient as dot product with matrix_gamma
//      (same shared-memory pattern as forward g_bw kernel)
//=============================================================================
__global__ void gradient_kernel(
    const float* __restrict__ P,
    const float* __restrict__ pap_real, const float* __restrict__ pap_imag,
    const float* __restrict__ pam_real, const float* __restrict__ pam_imag,
    const float* __restrict__ gp_real, const float* __restrict__ gp_imag,
    const float* __restrict__ gm_real, const float* __restrict__ gm_imag,
    const float* __restrict__ poq_real, const float* __restrict__ poq_imag,
    const float* __restrict__ bw_p_real, const float* __restrict__ bw_p_imag,
    const float* __restrict__ common_amp_factor_real,
    const float* __restrict__ common_amp_factor_imag,
    const float* __restrict__ ap_real, const float* __restrict__ ap_imag,
    const float* __restrict__ am_real, const float* __restrict__ am_imag,
    const float* __restrict__ dQ_dP,
    const float* __restrict__ bw_dom_real, const float* __restrict__ bw_dom_imag,
    const float* __restrict__ g_interp_real, const float* __restrict__ g_interp_imag,
    const float* __restrict__ g_bw_real, const float* __restrict__ g_bw_imag,
    const float* __restrict__ frac, const float* __restrict__ time,
    const int* __restrict__ m0_index, const int* __restrict__ g0_index,
    const int* __restrict__ bw_order,
    const float* __restrict__ matrix_gamma,
    const float* __restrict__ m0, const float* __restrict__ g0,
    const float* __restrict__ ck_real, const float* __restrict__ ck_imag,
    float Gamma, float Delta_Gamma, float Delta_m,
    float A_p, float poq_rho, float pop_phi,
    int n_wave, int n_res, int n_unique_bw, int n_gamma_rows, int n_mass,
    float* __restrict__ grad_ck_real_partial,
    float* __restrict__ grad_ck_imag_partial,
    float* __restrict__ grad_m0_partial,
    float* __restrict__ grad_g0_partial,
    float* __restrict__ grad_Gamma_partial,
    float* __restrict__ grad_DeltaGamma_partial,
    float* __restrict__ grad_DeltaM_partial,
    float* __restrict__ grad_Ap_partial,
    float* __restrict__ grad_poq_rho_partial,
    float* __restrict__ grad_pop_phi_partial,
    int n_events
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;

    // Load forward outputs
    complex pap(pap_real[event_idx], pap_imag[event_idx]);
    complex pam(pam_real[event_idx], pam_imag[event_idx]);
    complex gp(gp_real[event_idx], gp_imag[event_idx]);
    complex gm(gm_real[event_idx], gm_imag[event_idx]);
    complex poq(poq_real[event_idx], poq_imag[event_idx]);
    complex ap(ap_real[event_idx], ap_imag[event_idx]);
    complex am(am_real[event_idx], am_imag[event_idx]);

    float dQ_dP_val = dQ_dP[event_idx];
    float frac_val = frac[event_idx];
    float t = time[event_idx];

    float pb = thrust::norm(pap);
    float pbbar = thrust::norm(pam);
    float dP_dpb = frac_val * (1.0 - A_p);
    float dP_dpbbar = (1.0 - frac_val) * (1.0 + A_p);
    float dP_dAp = -frac_val * pb + (1.0 - frac_val) * pbbar;

    grad_Ap_partial[event_idx] = dQ_dP_val * dP_dAp;

    float dQ_dpb = dQ_dP_val * dP_dpb;
    float dQ_dpbbar = dQ_dP_val * dP_dpbbar;

    complex d_pb_dap = conj(pap) * gp;
    complex d_pb_dam = conj(pap) * gm * poq;
    complex d_pbbar_dap = conj(pam) * (gm / poq);
    complex d_pbbar_dam = conj(pam) * gp;

    complex dQ_dap_val = dQ_dpb * d_pb_dap + dQ_dpbbar * d_pbbar_dap;
    complex dQ_dam_val = dQ_dpb * d_pb_dam + dQ_dpbbar * d_pbbar_dam;

    //=========================================================================
    // Phase 1: ck gradients (parallel across waves)
    //=========================================================================
    int n_wave_half = n_wave / 2;

    for (int i = tid; i < n_wave_half; i += block_sz) {
        complex common_i(
            common_amp_factor_real[event_idx * n_wave + i],
            common_amp_factor_imag[event_idx * n_wave + i]);
        complex grad_ck = dQ_dap_val * common_i;
        grad_ck_real_partial[event_idx * n_wave + i] = grad_ck.real();
        grad_ck_imag_partial[event_idx * n_wave + i] = grad_ck.imag();

        complex common_j(
            common_amp_factor_real[event_idx * n_wave + n_wave_half + i],
            common_amp_factor_imag[event_idx * n_wave + n_wave_half + i]);
        complex grad_ck2 = dQ_dam_val * common_j;
        grad_ck_real_partial[event_idx * n_wave + n_wave_half + i] = grad_ck2.real();
        grad_ck_imag_partial[event_idx * n_wave + n_wave_half + i] = grad_ck2.imag();
    }

    //=========================================================================
    // Phase 2: Pre-compute dQ_dbw_dom for all unique_bw
    // and accumulate m0 gradients simultaneously
    //=========================================================================
    __shared__ float s_dQ_dbw_dom_real[216];  // n_unique_bw = 216
    __shared__ float s_dQ_dbw_dom_imag[216];
    __shared__ float s_dQ_dg_bw_real[216];
    __shared__ float s_dQ_dg_bw_imag[216];

    // Initialize m0 partials to zero
    for (int bw_idx = tid; bw_idx < n_unique_bw; bw_idx += block_sz) {
        grad_m0_partial[event_idx * n_unique_bw + bw_idx] = 0.0;
        s_dQ_dbw_dom_real[bw_idx] = 0.0;
        s_dQ_dbw_dom_imag[bw_idx] = 0.0;
    }
    __syncthreads();

    // Accumulate dQ_dbw_dom and m0 gradient contributions
    // This replaces the per-gamma_idx recomputation with a single pass
    for (int wave_idx = tid; wave_idx < n_wave; wave_idx += block_sz) {
        complex bw_p_val(bw_p_real[event_idx * n_wave + wave_idx],
                         bw_p_imag[event_idx * n_wave + wave_idx]);
        complex one_over_bw = complex(1.0, 0.0) / bw_p_val;
        complex common_amp(
            common_amp_factor_real[event_idx * n_wave + wave_idx],
            common_amp_factor_imag[event_idx * n_wave + wave_idx]);
        complex dQ_da = (wave_idx < n_wave_half) ? dQ_dap_val : dQ_dam_val;
        complex ck(ck_real[wave_idx], ck_imag[wave_idx]);
        complex dQ_dbw_p = dQ_da * (-ck * one_over_bw * common_amp);

        for (int res_idx = 0; res_idx < n_res; res_idx++) {
            int bw_idx = bw_order[wave_idx * n_res + res_idx];

            complex bw_dom_val(
                bw_dom_real[event_idx * n_unique_bw + bw_idx],
                bw_dom_imag[event_idx * n_unique_bw + bw_idx]);
            complex dQ_dbw_dom_contrib = dQ_dbw_p * (bw_p_val / bw_dom_val);

            // Accumulate shared memory (avoiding atomics - each thread handles different wave groups)
            // Threads may conflict for same bw_idx across waves - use atomicAdd
            atomicAdd(&s_dQ_dbw_dom_real[bw_idx], dQ_dbw_dom_contrib.real());
            atomicAdd(&s_dQ_dbw_dom_imag[bw_idx], dQ_dbw_dom_contrib.imag());

            // m0 gradient (atomic: multiple wave/res pairs can map to same bw_idx)
            float m0_val = m0[m0_index[bw_idx]];
            complex g_bw_val(g_bw_real[event_idx * n_unique_bw + bw_idx],
                             g_bw_imag[event_idx * n_unique_bw + bw_idx]);
            complex dbw_dom_dm0 = complex(2.0 * m0_val, 0.0) - complex(0.0, 1.0) * g_bw_val;
            atomicAdd(&grad_m0_partial[event_idx * n_unique_bw + bw_idx],
                2.0 * (dQ_dbw_dom_contrib * dbw_dom_dm0).real());
        }
    }
    __syncthreads();

    // Convert dQ_dbw_dom to dQ_dg_bw = dQ_dbw_dom * (-1j * m0)
    if (tid < n_unique_bw) {
        float m0_val = m0[m0_index[tid]];
        // dQ_dg_bw = dQ_dbw_dom * complex(0, -m0_val)
        complex dm0(s_dQ_dbw_dom_real[tid], s_dQ_dbw_dom_imag[tid]);
        complex dg_bw = dm0 * complex(0.0, -m0_val);
        s_dQ_dg_bw_real[tid] = dg_bw.real();
        s_dQ_dg_bw_imag[tid] = dg_bw.imag();
    }
    __syncthreads();

    //=========================================================================
    // Phase 3: g0 gradient via matrix-vector multiply with shared memory
    // g_bw[gamma_idx] = sum_{bw_idx} dQ_dg_bw[bw_idx] * matrix_gamma[gamma_idx, bw_idx]
    //
    // Same pattern as forward g_bw kernel but transposed:
    // Each thread handles one gamma_idx, iterates over all bw_idx
    //=========================================================================
    for (int gamma_idx = tid; gamma_idx < n_gamma_rows; gamma_idx += block_sz) {
        float sum_r = 0.0, sum_i = 0.0;
        for (int bw_idx = 0; bw_idx < n_unique_bw; bw_idx++) {
            float mg = matrix_gamma[gamma_idx * n_unique_bw + bw_idx];
            sum_r += s_dQ_dg_bw_real[bw_idx] * mg;
            sum_i += s_dQ_dg_bw_imag[bw_idx] * mg;
        }

        complex dQ_dg_val(sum_r, sum_i);
        complex g_interp_val(
            g_interp_real[event_idx * n_gamma_rows + gamma_idx],
            g_interp_imag[event_idx * n_gamma_rows + gamma_idx]);

        // Wirtinger gradient for real parameter: ∂Q/∂g0 = 2*Re(dQ_dg * g_interp)
        grad_g0_partial[event_idx * n_gamma_rows + gamma_idx] =
            2.0 * (dQ_dg_val * g_interp_val).real();
    }

    //=========================================================================
    // Phase 4: Time evolution gradients (scalar, thread 0 only)
    //=========================================================================
    if (tid == 0) {
        complex d_pb_dgp = conj(pap) * ap;
        complex d_pb_dgm = conj(pap) * poq * am;
        complex d_pbbar_dgp = conj(pam) * am;
        complex d_pbbar_dgm = conj(pam) * ap / poq;

        complex dQ_dgp = dQ_dpb * d_pb_dgp + dQ_dpbbar * d_pbbar_dgp;
        complex dQ_dgm = dQ_dpb * d_pb_dgm + dQ_dpbbar * d_pbbar_dgm;

        complex dgp_dGamma = (-t / 2.0) * gp;
        complex dgm_dGamma = (-t / 2.0) * gm;
        complex dgp_dDeltaGamma = (-t / 4.0) * gm;
        complex dgm_dDeltaGamma = (-t / 4.0) * gp;
        complex dgp_dDeltaM = complex(0.0, t / 2.0) * gm;
        complex dgm_dDeltaM = complex(0.0, t / 2.0) * gp;

        grad_Gamma_partial[event_idx] = 2.0 * (dQ_dgp * dgp_dGamma + dQ_dgm * dgm_dGamma).real();
        grad_DeltaGamma_partial[event_idx] = 2.0 * (dQ_dgp * dgp_dDeltaGamma + dQ_dgm * dgm_dDeltaGamma).real();
        grad_DeltaM_partial[event_idx] = 2.0 * (dQ_dgp * dgp_dDeltaM + dQ_dgm * dgm_dDeltaM).real();

        complex d_pb_dpoq = conj(pap) * gm * am;
        complex d_pbbar_dpoq = conj(pam) * (-gm / (poq * poq)) * ap;
        complex dQ_dpoq = dQ_dpb * d_pb_dpoq + dQ_dpbbar * d_pbbar_dpoq;

        complex exp_phi = exp(complex(0.0, 1.0) * pop_phi);
        grad_poq_rho_partial[event_idx] = 2.0 * (dQ_dpoq * exp_phi).real();
        grad_pop_phi_partial[event_idx] = 2.0 * (dQ_dpoq * poq_rho * complex(0.0, 1.0) * exp_phi).real();
    }
}

//=============================================================================
// Reduction kernels (kept from original implementation)
//=============================================================================
__global__ void reduce_sum_kernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
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
    const float* real_in, const float* imag_in,
    float* real_out, float* imag_out, int n
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float* sreal = sdata;
    float* simag = sdata + blockDim.x;
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
    const float* input, float* output,
    int n_events, int n_features
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int feat = blockIdx.x;
    if (feat >= n_features) return;
    float sum = 0.0;
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
    float real_sum = 0.0, imag_sum = 0.0;
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

// Structs for unified launch API (float32 variant)
typedef struct {
    const float* mass; const float* momentum; const float* angle;
    const float* frac; const float* time; const float* weight; const float* bkg;
    const float* ck_real; const float* ck_imag; const float* m0; const float* g0;
    float Gamma; float Delta_Gamma; float Delta_m;
    float A_prod; float poq_rho; float pop_phi;
    float norm_val; int use_norm;
} ComputeData;

typedef struct {
    const int* m0_index; const int* g0_index;
    const int* g0_mass_index; const int* mass_index;
    const int* fl_type; const int* fl_q_index;
    const int* bw_order; const int* fl_order; const int* angle_index;
} ComputeIndices;

typedef struct {
    const float* angle_k; const float* angle_b;
    const float* matrix_angle_real; const float* matrix_angle_imag;
    const float* gamma_table_real; const float* gamma_table_imag;
    float gamma_min; float gamma_delta; int gamma_table_bins;
    const float* matrix_gamma;
    const float* fl_table; float fl_min; float fl_delta; int fl_table_bins;
} ComputeConstants;

typedef struct {
    int n_events; int n_wave; int n_res; int n_decay;
    int n_unique_bw; int n_gamma_rows;
    int n_mass; int n_momentum; int n_angle_k; int n_angle_total;
} ComputeDims;

typedef struct {
    float* g_interp_real; float* g_interp_imag;
    float* g_bw_real; float* g_bw_imag;
    float* Q_out; float* P_out;
    float* pap_real; float* pap_imag;
    float* pam_real; float* pam_imag;
    float* gp_real; float* gp_imag;
    float* gm_real; float* gm_imag;
    float* poq_real; float* poq_imag;
    float* bw_p_real; float* bw_p_imag;
    float* common_amp_factor_real; float* common_amp_factor_imag;
    float* ap_real; float* ap_imag;
    float* am_real; float* am_imag;
    float* dQ_dP;
    float* bw_dom_real; float* bw_dom_imag;
    float* grad_ck_real_partial; float* grad_ck_imag_partial;
    float* grad_m0_partial; float* grad_g0_partial;
    float* grad_Gamma_partial; float* grad_DeltaGamma_partial;
    float* grad_DeltaM_partial; float* grad_Ap_partial;
    float* grad_poq_rho_partial; float* grad_pop_phi_partial;
} ComputeScratch;

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

// Optimized forward: two-kernel approach
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

    int block_size = 256;
    compute_g_bw_kernel<<<n_events, block_size>>>(
        mass, g0, g0_index, g0_mass_index, matrix_gamma,
        gamma_table_real, gamma_table_imag,
        gamma_min, gamma_delta,
        n_gamma_rows, n_unique_bw, n_mass, gamma_table_bins,
        g_interp_real, g_interp_imag,
        g_bw_real, g_bw_imag, n_events);
    CUDA_CHECK(cudaGetLastError());
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
    int n_mass, int n_momentum, int n_angle_k, int n_angle_total,
    int fl_table_bins,
    const float* ck_real, const float* ck_imag, const float* m0,
    float Gamma, float Delta_Gamma, float Delta_m,
    float A_p, float poq_rho, float pop_phi,
    float* Q_out, float* P_out,
    float* pap_real, float* pap_imag, float* pam_real, float* pam_imag,
    float* gp_real, float* gp_imag, float* gm_real, float* gm_imag,
    float* poq_real, float* poq_imag,
    float* bw_p_real, float* bw_p_imag,
    float* common_amp_factor_real, float* common_amp_factor_imag,
    float* ap_real, float* ap_imag, float* am_real, float* am_imag, float* dQ_dP,
    float* bw_dom_real, float* bw_dom_imag,
    int n_events, int use_norm, float norm) {

    int block_size = 256;
    compute_main_kernel<<<n_events, block_size>>>(
        mass, momentum, angle, frac, time, weight, bkg,
        m0_index, fl_type, mass_index, fl_q_index,
        bw_order, fl_order, angle_index,
        angle_k, angle_b, matrix_angle_real, matrix_angle_imag,
        g_bw_real, g_bw_imag, fl_table,
        fl_min, fl_delta,
        n_wave, n_res, n_decay, n_unique_bw,
        n_mass, n_momentum, n_angle_k, n_angle_total,
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
    CUDA_CHECK(cudaGetLastError());
}

// Optimized backward: single kernel with improved g0 gradient
void launch_gradient(
    const float* P, const float* pap_real, const float* pap_imag,
    const float* pam_real, const float* pam_imag,
    const float* gp_real, const float* gp_imag,
    const float* gm_real, const float* gm_imag,
    const float* poq_real, const float* poq_imag,
    const float* bw_p_real, const float* bw_p_imag,
    const float* common_amp_factor_real, const float* common_amp_factor_imag,
    const float* ap_real, const float* ap_imag,
    const float* am_real, const float* am_imag,
    const float* dQ_dP,
    const float* bw_dom_real, const float* bw_dom_imag,
    const float* g_interp_real, const float* g_interp_imag,
    const float* g_bw_real, const float* g_bw_imag,
    const float* frac, const float* time,
    const int* m0_index, const int* g0_index, const int* bw_order,
    const float* matrix_gamma,
    const float* m0, const float* g0, const float* ck_real, const float* ck_imag,
    float Gamma, float Delta_Gamma, float Delta_m,
    float A_p, float poq_rho, float pop_phi,
    int n_wave, int n_res, int n_unique_bw, int n_gamma_rows, int n_mass,
    float* grad_ck_real_partial, float* grad_ck_imag_partial,
    float* grad_m0_partial, float* grad_g0_partial,
    float* grad_Gamma_partial, float* grad_DeltaGamma_partial,
    float* grad_DeltaM_partial, float* grad_Ap_partial,
    float* grad_poq_rho_partial, float* grad_pop_phi_partial,
    int n_events) {

    int block_size = 256;
    gradient_kernel<<<n_events, block_size>>>(
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
    CUDA_CHECK(cudaGetLastError());
}

// Reduction launch wrappers (same as original)
void launch_reduce_sum(const float* input, float* output, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    reduce_sum_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(input, output, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_reduce_sum_complex(const float* real_in, const float* imag_in,
    float* real_out, float* imag_out, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    reduce_sum_complex_kernel<<<grid_size, block_size, 2 * block_size * sizeof(float)>>>(
        real_in, imag_in, real_out, imag_out, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_reduce_sum_features(const float* input, float* output,
    int n_events, int n_features) {
    int block_size = 256;
    reduce_sum_features_kernel<<<n_features, block_size, block_size * sizeof(float)>>>(
        input, output, n_events, n_features);
    CUDA_CHECK(cudaGetLastError());
}

void launch_reduce_sum_complex_features(const float* real_in, const float* imag_in,
    float* real_out, float* imag_out,
    int n_events, int n_features) {
    int block_size = 256;
    reduce_sum_complex_features_kernel<<<n_features, block_size, 2 * block_size * sizeof(float)>>>(
        real_in, imag_in, real_out, imag_out, n_events, n_features);
    CUDA_CHECK(cudaGetLastError());
}

// Unified launch — calls g_bw + main forward + gradient in sequence
void launch_compute_all(
    const ComputeData* d, const ComputeIndices* idx,
    const ComputeConstants* c, const ComputeDims* dim,
    ComputeScratch* s
) {
    launch_compute_g_bw(
        d->mass, d->g0, idx->g0_index, idx->g0_mass_index,
        c->matrix_gamma,
        c->gamma_table_real, c->gamma_table_imag,
        c->gamma_min, c->gamma_delta,
        dim->n_gamma_rows, dim->n_unique_bw, dim->n_mass, c->gamma_table_bins,
        s->g_interp_real, s->g_interp_imag,
        s->g_bw_real, s->g_bw_imag,
        dim->n_events);

    launch_compute_main(
        d->mass, d->momentum, d->angle,
        d->frac, d->time, d->weight, d->bkg,
        idx->m0_index, idx->fl_type,
        idx->mass_index, idx->fl_q_index,
        idx->bw_order, idx->fl_order, idx->angle_index,
        c->angle_k, c->angle_b,
        c->matrix_angle_real, c->matrix_angle_imag,
        s->g_bw_real, s->g_bw_imag,
        c->fl_table, c->fl_min, c->fl_delta,
        dim->n_wave, dim->n_res, dim->n_decay, dim->n_unique_bw,
        dim->n_mass, dim->n_momentum, dim->n_angle_k, dim->n_angle_total,
        c->fl_table_bins,
        d->ck_real, d->ck_imag, d->m0,
        d->Gamma, d->Delta_Gamma, d->Delta_m,
        d->A_prod, d->poq_rho, d->pop_phi,
        s->Q_out, s->P_out,
        s->pap_real, s->pap_imag, s->pam_real, s->pam_imag,
        s->gp_real, s->gp_imag, s->gm_real, s->gm_imag,
        s->poq_real, s->poq_imag,
        s->bw_p_real, s->bw_p_imag,
        s->common_amp_factor_real, s->common_amp_factor_imag,
        s->ap_real, s->ap_imag, s->am_real, s->am_imag, s->dQ_dP,
        s->bw_dom_real, s->bw_dom_imag,
        dim->n_events, d->use_norm, d->norm_val);

    launch_gradient(
        s->P_out,
        s->pap_real, s->pap_imag, s->pam_real, s->pam_imag,
        s->gp_real, s->gp_imag, s->gm_real, s->gm_imag,
        s->poq_real, s->poq_imag,
        s->bw_p_real, s->bw_p_imag,
        s->common_amp_factor_real, s->common_amp_factor_imag,
        s->ap_real, s->ap_imag, s->am_real, s->am_imag, s->dQ_dP,
        s->bw_dom_real, s->bw_dom_imag,
        s->g_interp_real, s->g_interp_imag,
        s->g_bw_real, s->g_bw_imag,
        d->frac, d->time,
        idx->m0_index, idx->g0_index, idx->bw_order, c->matrix_gamma,
        d->m0, d->g0, d->ck_real, d->ck_imag,
        d->Gamma, d->Delta_Gamma, d->Delta_m,
        d->A_prod, d->poq_rho, d->pop_phi,
        dim->n_wave, dim->n_res, dim->n_unique_bw, dim->n_gamma_rows, dim->n_mass,
        s->grad_ck_real_partial, s->grad_ck_imag_partial,
        s->grad_m0_partial, s->grad_g0_partial,
        s->grad_Gamma_partial, s->grad_DeltaGamma_partial,
        s->grad_DeltaM_partial, s->grad_Ap_partial,
        s->grad_poq_rho_partial, s->grad_pop_phi_partial,
        dim->n_events);
}

// ── Upload helpers ──
static void* _up_int(const int* src, int n) {
    int* d; cudaMalloc(&d, n * sizeof(int));
    cudaMemcpy(d, src, n * sizeof(int), cudaMemcpyHostToDevice); return d;
}
static void* _up_flt(const float* src, int n) {
    float* d; cudaMalloc(&d, n * sizeof(float));
    cudaMemcpy(d, src, n * sizeof(float), cudaMemcpyHostToDevice); return d;
}

// Internal context: holds persistent GPU pointers
typedef struct {
    ComputeIndices idx; ComputeConstants cst; ComputeDims dim;
} _F32Ctx;

typedef struct { ComputeData* d; ComputeScratch* s; } _F32DH;

void* cuda_create_context(
    const int* m0_i, int n1, const int* g0_i, int n2,
    const int* g0_m, int n3, const int* mass_i, int n4,
    const int* fl_t, int n5, const int* fl_q, int n6,
    const int* bw_o, int n7, const int* fl_o, int n8,
    const int* ang_i, int n9,
    const float* ak, int n10, const float* ab, int n11,
    const float* mar, int n12, const float* mai, int n13,
    const float* gtr, int n14, const float* gti, int n15,
    float gmin, float gdel, int gbins,
    const float* mg, int n16,
    const float* ft, int n17, float flmin, float fldel, int fbins,
    int nw, int nr, int nd, int nub, int ngr,
    int nm, int nmom, int nak_, int nat
) {
    _F32Ctx* c = (_F32Ctx*)malloc(sizeof(_F32Ctx));
    c->idx.m0_index = (int*)_up_int(m0_i, n1); c->idx.g0_index = (int*)_up_int(g0_i, n2);
    c->idx.g0_mass_index = (int*)_up_int(g0_m, n3); c->idx.mass_index = (int*)_up_int(mass_i, n4);
    c->idx.fl_type = (int*)_up_int(fl_t, n5); c->idx.fl_q_index = (int*)_up_int(fl_q, n6);
    c->idx.bw_order = (int*)_up_int(bw_o, n7); c->idx.fl_order = (int*)_up_int(fl_o, n8);
    c->idx.angle_index = (int*)_up_int(ang_i, n9);
    c->cst.angle_k = (float*)_up_flt(ak, n10); c->cst.angle_b = (float*)_up_flt(ab, n11);
    c->cst.matrix_angle_real = (float*)_up_flt(mar, n12); c->cst.matrix_angle_imag = (float*)_up_flt(mai, n13);
    c->cst.gamma_table_real = (float*)_up_flt(gtr, n14); c->cst.gamma_table_imag = (float*)_up_flt(gti, n15);
    c->cst.gamma_min = gmin; c->cst.gamma_delta = gdel; c->cst.gamma_table_bins = gbins;
    c->cst.matrix_gamma = (float*)_up_flt(mg, n16);
    c->cst.fl_table = (float*)_up_flt(ft, n17); c->cst.fl_min = flmin; c->cst.fl_delta = fldel; c->cst.fl_table_bins = fbins;
    c->dim.n_wave = nw; c->dim.n_res = nr; c->dim.n_decay = nd;
    c->dim.n_unique_bw = nub; c->dim.n_gamma_rows = ngr;
    c->dim.n_mass = nm; c->dim.n_momentum = nmom; c->dim.n_angle_k = nak_; c->dim.n_angle_total = nat;
    return c;
}

void* cuda_load_data(void* vctx, const float* mass, const float* mom,
    const float* ang, const float* frac, const float* time,
    const float* wgt, const float* bkg, int ne
) {
    _F32Ctx* c = (_F32Ctx*)vctx;
    ComputeData* d = (ComputeData*)malloc(sizeof(ComputeData));
    d->mass = (float*)_up_flt(mass, ne * c->dim.n_mass);
    d->momentum = (float*)_up_flt(mom, ne * c->dim.n_momentum);
    d->angle = (float*)_up_flt(ang, ne * c->dim.n_angle_total * 3);
    d->frac = (float*)_up_flt(frac, ne);
    d->time = (float*)_up_flt(time, ne);
    d->weight = (float*)_up_flt(wgt, ne);
    d->bkg = (float*)_up_flt(bkg, ne);
    c->dim.n_events = ne;
    int nw = c->dim.n_wave, nu = c->dim.n_unique_bw, ng = c->dim.n_gamma_rows;
    ComputeScratch* s = (ComputeScratch*)malloc(sizeof(ComputeScratch));
#define S(f) cudaMalloc(&s->f, ne * sizeof(float))
#define S2(f,n) cudaMalloc(&s->f, ne * (n) * sizeof(float))
    S2(g_interp_real, ng); S2(g_interp_imag, ng);
    S2(g_bw_real, nu); S2(g_bw_imag, nu);
    S(Q_out); S(P_out); S(pap_real); S(pap_imag); S(pam_real); S(pam_imag);
    S(gp_real); S(gp_imag); S(gm_real); S(gm_imag); S(poq_real); S(poq_imag);
    S2(bw_p_real, nw); S2(bw_p_imag, nw);
    S2(common_amp_factor_real, nw); S2(common_amp_factor_imag, nw);
    S(ap_real); S(ap_imag); S(am_real); S(am_imag); S(dQ_dP);
    S2(bw_dom_real, nu); S2(bw_dom_imag, nu);
    S2(grad_ck_real_partial, nw); S2(grad_ck_imag_partial, nw);
    S2(grad_m0_partial, nu); S2(grad_g0_partial, ng);
    S(grad_Gamma_partial); S(grad_DeltaGamma_partial); S(grad_DeltaM_partial);
    S(grad_Ap_partial); S(grad_poq_rho_partial); S(grad_pop_phi_partial);
#undef S
    _F32DH* fdh = (_F32DH*)malloc(sizeof(_F32DH));
    fdh->d = d; fdh->s = s;
    return fdh;
}

void cuda_compute(void* vctx, void* vdh,
    const float* ck_r, const float* ck_i,
    const float* m0, const float* g0,
    float Gamma, float DG, float DM,
    float Ap, float pr, float pp,
    float norm_val, int use_norm,
    float* Q_out, float* P_out,
    float* gck_r, float* gck_i,
    float* gm0_out, float* gg0_out,
    float* gsc_out,
    int n_wave, int n_unique_bw, int n_gamma_rows
) {
    _F32Ctx* c = (_F32Ctx*)vctx;
    _F32DH* fdh = (_F32DH*)vdh;
    ComputeData* d = fdh->d;
    ComputeScratch* s = fdh->s;
    int ne = c->dim.n_events;

    d->ck_real = (float*)_up_flt(ck_r, n_wave);
    d->ck_imag = (float*)_up_flt(ck_i, n_wave);
    d->m0 = (float*)_up_flt(m0, n_unique_bw);
    d->g0 = (float*)_up_flt(g0, n_gamma_rows);
    d->Gamma = Gamma; d->Delta_Gamma = DG; d->Delta_m = DM;
    d->A_prod = Ap; d->poq_rho = pr; d->pop_phi = pp;
    d->norm_val = norm_val; d->use_norm = use_norm;

    launch_compute_all(d, &c->idx, &c->cst, &c->dim, s);

    cudaMemcpy(Q_out, s->Q_out, ne * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(P_out, s->P_out, ne * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(gck_r, s->grad_ck_real_partial, ne * n_wave * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(gck_i, s->grad_ck_imag_partial, ne * n_wave * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(gm0_out, s->grad_m0_partial, ne * n_unique_bw * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(gg0_out, s->grad_g0_partial, ne * n_gamma_rows * 4, cudaMemcpyDeviceToHost);

    float buf[6];
    cudaMemcpy(buf, s->grad_Gamma_partial, ne * 4, cudaMemcpyDeviceToHost);
    gsc_out[0]=0; for(int i=0;i<ne;i++) gsc_out[0]+=buf[i];
    cudaMemcpy(buf, s->grad_DeltaGamma_partial, ne * 4, cudaMemcpyDeviceToHost);
    gsc_out[1]=0; for(int i=0;i<ne;i++) gsc_out[1]+=buf[i];
    cudaMemcpy(buf, s->grad_DeltaM_partial, ne * 4, cudaMemcpyDeviceToHost);
    gsc_out[2]=0; for(int i=0;i<ne;i++) gsc_out[2]+=buf[i];
    cudaMemcpy(buf, s->grad_Ap_partial, ne * 4, cudaMemcpyDeviceToHost);
    gsc_out[3]=0; for(int i=0;i<ne;i++) gsc_out[3]+=buf[i];
    cudaMemcpy(buf, s->grad_poq_rho_partial, ne * 4, cudaMemcpyDeviceToHost);
    gsc_out[4]=0; for(int i=0;i<ne;i++) gsc_out[4]+=buf[i];
    cudaMemcpy(buf, s->grad_pop_phi_partial, ne * 4, cudaMemcpyDeviceToHost);
    gsc_out[5]=0; for(int i=0;i<ne;i++) gsc_out[5]+=buf[i];

    cudaFree((void*)d->ck_real); cudaFree((void*)d->ck_imag);
    cudaFree((void*)d->m0); cudaFree((void*)d->g0);
}

void cuda_free_context(void* vctx) {
    _F32Ctx* c = (_F32Ctx*)vctx;
    #define F(p) cudaFree((void*)c->idx.p)
    F(m0_index); F(g0_index); F(g0_mass_index); F(mass_index);
    F(fl_type); F(fl_q_index); F(bw_order); F(fl_order); F(angle_index);
    #undef F
    #define F(p) cudaFree((void*)c->cst.p)
    F(angle_k); F(angle_b); F(matrix_angle_real); F(matrix_angle_imag);
    F(gamma_table_real); F(gamma_table_imag); F(matrix_gamma); F(fl_table);
    #undef F
    free(c);
}

void cuda_free_data(void* vdh) {
    _F32DH* fdh = (_F32DH*)vdh;
    ComputeData* d = fdh->d; ComputeScratch* s = fdh->s;
    cudaFree((void*)d->mass); cudaFree((void*)d->momentum);
    cudaFree((void*)d->angle); cudaFree((void*)d->frac);
    cudaFree((void*)d->time); cudaFree((void*)d->weight); cudaFree((void*)d->bkg);
    free(d);
    #define F(p) cudaFree(s->p)
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
    free(s); free(fdh);
}

} // extern "C"
