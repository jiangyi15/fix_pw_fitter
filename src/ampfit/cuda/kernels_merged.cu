/**
 * OPTIMIZED CUDA kernels - MERGED-INDEX BW VARIANT
 *
 * This file is derived from kernels.cu. It adds two new kernels that
 * use pre-merged BW index arrays (bw_m0_index, bw_mass_index) so that
 * each of the 896 wave×res positions directly indexes m0 and mass
 * arrays without a bw_order indirection step.
 *
 * All original kernels remain unchanged and are included verbatim.
 *
 * Key differences from kernels.cu:
 * 1. compute_main_merged_kernel - replaces bw_order+m0_index+mass_index
 *    with bw_m0_index+bw_mass_index (size 896); g_bw is pre-scattered
 *    to (N, 896); bw_dom stored per-position (896).
 * 2. gradient_merged_kernel - bw_dom is (N, 896) per-position; still
 *    uses bw_order to map back to unique BW indices for m0/g0 grads.
 * 3. Corresponding launch wrappers in extern "C".
 *
 * Key optimizations (inherited from kernels.cu):
 * 1. g_bw computation: parallelized across gamma_rows using shared memory
 * 2. Main compute: ka_prod in shared memory, wave-level parallelism
 * 3. g0 gradient: pre-compute dQ_dbw_dom, then matrix-vector multiply
 * 4. Eliminated heap allocation (new double[])
 * 5. __restrict__ pointers for better compiler optimization
 * 6. Block size optimized for RTX 3070 Ti (compute 8.6)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/complex.h>
#include <cstdio>
#include <cmath>

using complex = thrust::complex<double>;

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
    const double* __restrict__ table_real,
    const double* __restrict__ table_imag,
    int type_idx, double x,
    double xmin, double xdelta, int n_bins
) {
    double diff = (x - xmin) / xdelta;
    int xbin = max(0, min((int)floor(diff), n_bins - 2));
    double delta = diff - xbin;
    int left_idx = type_idx * n_bins + xbin;
    int right_idx = left_idx + 1;
    double left_real = table_real[left_idx];
    double right_real = table_real[right_idx];
    double left_imag = table_imag[left_idx];
    double right_imag = table_imag[right_idx];
    double real_val = (right_real - left_real) * delta + left_real;
    double imag_val = (right_imag - left_imag) * delta + left_imag;
    return complex(real_val, imag_val);
}

// Real interpolation for FL factor
__device__ double interp_real_device(
    const double* __restrict__ table,
    int type_idx, double x,
    double xmin, double xdelta, int n_bins
) {
    double diff = (x - xmin) / xdelta;
    int xbin = max(0, min((int)floor(diff), n_bins - 2));
    double delta = diff - xbin;
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
    const double* __restrict__ mass,
    const double* __restrict__ g0,
    const int* __restrict__ g0_index,
    const int* __restrict__ g0_mass_index,
    const double* __restrict__ matrix_gamma,
    const double* __restrict__ gamma_table_real,
    const double* __restrict__ gamma_table_imag,
    double gamma_min, double gamma_delta,
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

    // Shared memory for g values (all gamma rows)
    __shared__ double s_g_real[288];  // n_gamma_rows = 288
    __shared__ double s_g_imag[288];

    // Phase 1: Each thread computes g for its assigned gamma rows
    for (int gamma_idx = tid; gamma_idx < n_gamma_rows; gamma_idx += block_sz) {
        int g0_idx = g0_index[gamma_idx];
        double g0_val = g0[g0_idx];
        double mass_val = mass[event_idx * n_mass + g0_mass_index[gamma_idx]];

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
        double sum_r = 0.0, sum_i = 0.0;
        for (int i = 0; i < n_gamma_rows; i++) {
            double mg = matrix_gamma[i * n_unique_bw + c];
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
// Shared memory: ka_prod for all angle_k values (336 doubles)
// Each thread handles multiple waves for bw_p, fa, common_amp computations
//=============================================================================
__global__ void compute_main_kernel(
    const double* __restrict__ mass,
    const double* __restrict__ momentum,
    const double* __restrict__ angle,
    const double* __restrict__ frac,
    const double* __restrict__ time,
    const double* __restrict__ weight,
    const double* __restrict__ bkg,
    const int* __restrict__ m0_index,
    const int* __restrict__ fl_type,
    const int* __restrict__ mass_index,
    const int* __restrict__ fl_q_index,
    const int* __restrict__ bw_order,
    const int* __restrict__ fl_order,
    const int* __restrict__ angle_index,
    const double* __restrict__ angle_k,
    const double* __restrict__ angle_b,
    const double* __restrict__ matrix_angle_real,
    const double* __restrict__ matrix_angle_imag,
    const double* __restrict__ g_bw_real,
    const double* __restrict__ g_bw_imag,
    const double* __restrict__ fl_table,
    double fl_min, double fl_delta,
    int n_wave, int n_res, int n_decay, int n_unique_bw,
    int n_mass, int n_momentum, int n_angle_k, int n_angle_total,
    int fl_table_bins,
    const double* __restrict__ ck_real,
    const double* __restrict__ ck_imag,
    const double* __restrict__ m0,
    double Gamma, double Delta_Gamma, double Delta_m,
    double A_p, double poq_rho, double pop_phi,
    double* __restrict__ Q_out,
    double* __restrict__ P_out,
    double* __restrict__ pap_real, double* __restrict__ pap_imag,
    double* __restrict__ pam_real, double* __restrict__ pam_imag,
    double* __restrict__ gp_real, double* __restrict__ gp_imag,
    double* __restrict__ gm_real, double* __restrict__ gm_imag,
    double* __restrict__ poq_real, double* __restrict__ poq_imag,
    double* __restrict__ bw_p_real, double* __restrict__ bw_p_imag,
    double* __restrict__ common_amp_factor_real,
    double* __restrict__ common_amp_factor_imag,
    double* __restrict__ ap_real, double* __restrict__ ap_imag,
    double* __restrict__ am_real, double* __restrict__ am_imag,
    double* __restrict__ dQ_dP,
    double* __restrict__ bw_dom_real, double* __restrict__ bw_dom_imag,
    int n_events, int use_norm, double norm
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;

    //=========================================================================
    // Phase 1: Compute ka_prod for all angle_k values (parallel across threads)
    //=========================================================================
    __shared__ double s_ka_prod[336];  // n_angle_k = 336

    for (int k_idx = tid; k_idx < n_angle_k; k_idx += block_sz) {
        int angle_pos = angle_index[k_idx];
        double ka_prod = 1.0;
        for (int comp = 0; comp < 3; comp++) {
            int angle_idx = event_idx * n_angle_total * 3 + angle_pos * 3 + comp;
            double k_val = angle_k[k_idx * 3 + comp];
            double b_val = angle_b[k_idx * 3 + comp];
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
            double m0_val = m0[m0_index[bw_idx]];
            double mass_val = mass[event_idx * n_mass + mass_index[bw_idx]];
            double m0_sq = m0_val * m0_val;
            double mass_sq = mass_val * mass_val;

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
        double fl_p = 1.0;
        for (int decay_idx = 0; decay_idx < n_decay; decay_idx++) {
            int fl_idx = fl_order[wave_idx * n_decay + decay_idx];
            double fl_q_val = momentum[event_idx * n_momentum + fl_q_index[fl_idx]];
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

    // Reduce across threads within block
    __shared__ double s_ap_r[256]; // max block size
    __shared__ double s_ap_i[256];
    __shared__ double s_am_r[256];
    __shared__ double s_am_i[256];

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
// KERNEL 2b: Merged-index main forward computation
//=============================================================================
// Uses pre-merged index arrays instead of bw_order + m0_index indirection.
// Merged arrays are uploaded separately: bw_m0_index (896,), bw_mass_index (896,).
// g_bw is already scattered to 896 positions: g_bw_real/bw_imag are (N, 896).
// bw_dom is stored per-position (896) instead of per-unique-BW (216).
//
// Same Phase 1, 3, 4 as compute_main_kernel; Phase 2 differs.
//=============================================================================
__global__ void compute_main_merged_kernel(
    const double* __restrict__ mass,
    const double* __restrict__ momentum,
    const double* __restrict__ angle,
    const double* __restrict__ frac,
    const double* __restrict__ time,
    const double* __restrict__ weight,
    const double* __restrict__ bkg,
    const int* __restrict__ bw_m0_index,      // replaces m0_index + bw_order; (n_total_positions,)
    const int* __restrict__ fl_type,
    const int* __restrict__ bw_mass_index,    // replaces mass_index + bw_order; (n_total_positions,)
    const int* __restrict__ fl_q_index,
    const int* __restrict__ fl_order,
    const int* __restrict__ angle_index,
    const double* __restrict__ angle_k,
    const double* __restrict__ angle_b,
    const double* __restrict__ matrix_angle_real,
    const double* __restrict__ matrix_angle_imag,
    const double* __restrict__ g_bw_real,     // pre-scattered (N, n_total_positions)
    const double* __restrict__ g_bw_imag,
    const double* __restrict__ fl_table,
    double fl_min, double fl_delta,
    int n_wave, int n_res, int n_decay, int n_total_positions,
    int n_mass, int n_momentum, int n_angle_k, int n_angle_total,
    int fl_table_bins,
    const double* __restrict__ ck_real,
    const double* __restrict__ ck_imag,
    const double* __restrict__ m0,
    double Gamma, double Delta_Gamma, double Delta_m,
    double A_p, double poq_rho, double pop_phi,
    double* __restrict__ Q_out,
    double* __restrict__ P_out,
    double* __restrict__ pap_real, double* __restrict__ pap_imag,
    double* __restrict__ pam_real, double* __restrict__ pam_imag,
    double* __restrict__ gp_real, double* __restrict__ gp_imag,
    double* __restrict__ gm_real, double* __restrict__ gm_imag,
    double* __restrict__ poq_real, double* __restrict__ poq_imag,
    double* __restrict__ bw_p_real, double* __restrict__ bw_p_imag,
    double* __restrict__ common_amp_factor_real,
    double* __restrict__ common_amp_factor_imag,
    double* __restrict__ ap_real, double* __restrict__ ap_imag,
    double* __restrict__ am_real, double* __restrict__ am_imag,
    double* __restrict__ dQ_dP,
    double* __restrict__ bw_dom_real, double* __restrict__ bw_dom_imag,  // (N, n_total_positions)
    int n_events, int use_norm, double norm
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;

    //=========================================================================
    // Phase 1: Compute ka_prod for all angle_k values (parallel across threads)
    //=========================================================================
    __shared__ double s_ka_prod[336];  // n_angle_k = 336

    for (int k_idx = tid; k_idx < n_angle_k; k_idx += block_sz) {
        int angle_pos = angle_index[k_idx];
        double ka_prod = 1.0;
        for (int comp = 0; comp < 3; comp++) {
            int angle_idx = event_idx * n_angle_total * 3 + angle_pos * 3 + comp;
            double k_val = angle_k[k_idx * 3 + comp];
            double b_val = angle_b[k_idx * 3 + comp];
            ka_prod *= cos(angle[angle_idx] * k_val + b_val);
        }
        s_ka_prod[k_idx] = ka_prod;
    }
    __syncthreads();

    //=========================================================================
    // Phase 2: Compute bw_p, fa, fl_factor, common_amp for each wave
    // Uses pre-merged BW indices — no bw_order indirection needed.
    // g_bw is pre-scattered to per-position layout.
    // bw_dom stored at per-position offset (n_total_positions).
    //=========================================================================
    int waves_per_thread = (n_wave + block_sz - 1) / block_sz;
    int wave_start = tid * waves_per_thread;
    int wave_end = min(wave_start + waves_per_thread, n_wave);

    for (int wave_idx = wave_start; wave_idx < wave_end; wave_idx++) {
        // --- bw_p = product of bw_dom over resonances ---
        complex bw_p_val(1.0, 0.0);
        for (int res_idx = 0; res_idx < n_res; res_idx++) {
            int pos = wave_idx * n_res + res_idx;
            double m0_val = m0[bw_m0_index[pos]];              // direct — no bw_order lookup
            double mass_val = mass[event_idx * n_mass + bw_mass_index[pos]];  // direct
            double m0_sq = m0_val * m0_val;
            double mass_sq = mass_val * mass_val;

            complex g_bw_val(
                g_bw_real[event_idx * n_total_positions + pos],  // pre-scattered
                g_bw_imag[event_idx * n_total_positions + pos]
            );

            // bw_dom = m0² - m² + m0*g_bw.imag  -  1j*m0*g_bw.real
            complex bw_dom(m0_sq - mass_sq + m0_val * g_bw_val.imag(),
                          -m0_val * g_bw_val.real());
            bw_p_val *= bw_dom;

            // Store per-position bw_dom (NOT at unique BW index)
            bw_dom_real[event_idx * n_total_positions + pos] = bw_dom.real();
            bw_dom_imag[event_idx * n_total_positions + pos] = bw_dom.imag();
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
        double fl_p = 1.0;
        for (int decay_idx = 0; decay_idx < n_decay; decay_idx++) {
            int fl_idx = fl_order[wave_idx * n_decay + decay_idx];
            double fl_q_val = momentum[event_idx * n_momentum + fl_q_index[fl_idx]];
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

    // Reduce across threads within block
    __shared__ double s_ap_r[256]; // max block size
    __shared__ double s_ap_i[256];
    __shared__ double s_am_r[256];
    __shared__ double s_am_i[256];

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
    const double* __restrict__ P,
    const double* __restrict__ pap_real, const double* __restrict__ pap_imag,
    const double* __restrict__ pam_real, const double* __restrict__ pam_imag,
    const double* __restrict__ gp_real, const double* __restrict__ gp_imag,
    const double* __restrict__ gm_real, const double* __restrict__ gm_imag,
    const double* __restrict__ poq_real, const double* __restrict__ poq_imag,
    const double* __restrict__ bw_p_real, const double* __restrict__ bw_p_imag,
    const double* __restrict__ common_amp_factor_real,
    const double* __restrict__ common_amp_factor_imag,
    const double* __restrict__ ap_real, const double* __restrict__ ap_imag,
    const double* __restrict__ am_real, const double* __restrict__ am_imag,
    const double* __restrict__ dQ_dP,
    const double* __restrict__ bw_dom_real, const double* __restrict__ bw_dom_imag,
    const double* __restrict__ g_interp_real, const double* __restrict__ g_interp_imag,
    const double* __restrict__ g_bw_real, const double* __restrict__ g_bw_imag,
    const double* __restrict__ frac, const double* __restrict__ time,
    const int* __restrict__ m0_index, const int* __restrict__ g0_index,
    const int* __restrict__ bw_order,
    const double* __restrict__ matrix_gamma,
    const double* __restrict__ m0, const double* __restrict__ g0,
    const double* __restrict__ ck_real, const double* __restrict__ ck_imag,
    double Gamma, double Delta_Gamma, double Delta_m,
    double A_p, double poq_rho, double pop_phi,
    int n_wave, int n_res, int n_unique_bw, int n_gamma_rows, int n_mass,
    double* __restrict__ grad_ck_real_partial,
    double* __restrict__ grad_ck_imag_partial,
    double* __restrict__ grad_m0_partial,
    double* __restrict__ grad_g0_partial,
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

    // Load forward outputs
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
    double dP_dpb = frac_val * (1.0 - A_p);
    double dP_dpbbar = (1.0 - frac_val) * (1.0 + A_p);
    double dP_dAp = -frac_val * pb + (1.0 - frac_val) * pbbar;

    grad_Ap_partial[event_idx] = dQ_dP_val * dP_dAp;

    double dQ_dpb = dQ_dP_val * dP_dpb;
    double dQ_dpbbar = dQ_dP_val * dP_dpbbar;

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
    __shared__ double s_dQ_dbw_dom_real[216];  // n_unique_bw = 216
    __shared__ double s_dQ_dbw_dom_imag[216];
    __shared__ double s_dQ_dg_bw_real[216];
    __shared__ double s_dQ_dg_bw_imag[216];

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
            double m0_val = m0[m0_index[bw_idx]];
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
        double m0_val = m0[m0_index[tid]];
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
        double sum_r = 0.0, sum_i = 0.0;
        for (int bw_idx = 0; bw_idx < n_unique_bw; bw_idx++) {
            double mg = matrix_gamma[gamma_idx * n_unique_bw + bw_idx];
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
// KERNEL 3b: Merged-index gradient computation
//=============================================================================
// Same as gradient_kernel but bw_dom is stored per-position (n_total_positions)
// instead of per-unique-BW (n_unique_bw). Still uses bw_order to map back from
// per-position to unique BW indices for m0/g0 gradient scatter.
// g_bw is still read from the original (N, n_unique_bw) storage.
//=============================================================================
__global__ void gradient_merged_kernel(
    const double* __restrict__ P,
    const double* __restrict__ pap_real, const double* __restrict__ pap_imag,
    const double* __restrict__ pam_real, const double* __restrict__ pam_imag,
    const double* __restrict__ gp_real, const double* __restrict__ gp_imag,
    const double* __restrict__ gm_real, const double* __restrict__ gm_imag,
    const double* __restrict__ poq_real, const double* __restrict__ poq_imag,
    const double* __restrict__ bw_p_real, const double* __restrict__ bw_p_imag,
    const double* __restrict__ common_amp_factor_real,
    const double* __restrict__ common_amp_factor_imag,
    const double* __restrict__ ap_real, const double* __restrict__ ap_imag,
    const double* __restrict__ am_real, const double* __restrict__ am_imag,
    const double* __restrict__ dQ_dP,
    const double* __restrict__ bw_dom_real, const double* __restrict__ bw_dom_imag,
    const double* __restrict__ g_interp_real, const double* __restrict__ g_interp_imag,
    const double* __restrict__ g_bw_real, const double* __restrict__ g_bw_imag,
    const double* __restrict__ frac, const double* __restrict__ time,
    const int* __restrict__ m0_index, const int* __restrict__ g0_index,
    const int* __restrict__ bw_order,
    const double* __restrict__ matrix_gamma,
    const double* __restrict__ m0, const double* __restrict__ g0,
    const double* __restrict__ ck_real, const double* __restrict__ ck_imag,
    double Gamma, double Delta_Gamma, double Delta_m,
    double A_p, double poq_rho, double pop_phi,
    int n_wave, int n_res, int n_unique_bw, int n_total_positions,
    int n_gamma_rows, int n_mass,
    double* __restrict__ grad_ck_real_partial,
    double* __restrict__ grad_ck_imag_partial,
    double* __restrict__ grad_m0_partial,
    double* __restrict__ grad_g0_partial,
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

    // Load forward outputs
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
    double dP_dpb = frac_val * (1.0 - A_p);
    double dP_dpbbar = (1.0 - frac_val) * (1.0 + A_p);
    double dP_dAp = -frac_val * pb + (1.0 - frac_val) * pbbar;

    grad_Ap_partial[event_idx] = dQ_dP_val * dP_dAp;

    double dQ_dpb = dQ_dP_val * dP_dpb;
    double dQ_dpbbar = dQ_dP_val * dP_dpbbar;

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
    //
    // bw_dom is stored per-position (n_total_positions offset).
    // bw_order maps back to unique BW indices for scatter/gradient.
    //=========================================================================
    __shared__ double s_dQ_dbw_dom_real[216];  // n_unique_bw = 216
    __shared__ double s_dQ_dbw_dom_imag[216];
    __shared__ double s_dQ_dg_bw_real[216];
    __shared__ double s_dQ_dg_bw_imag[216];

    // Initialize m0 partials to zero
    for (int bw_idx = tid; bw_idx < n_unique_bw; bw_idx += block_sz) {
        grad_m0_partial[event_idx * n_unique_bw + bw_idx] = 0.0;
        s_dQ_dbw_dom_real[bw_idx] = 0.0;
        s_dQ_dbw_dom_imag[bw_idx] = 0.0;
    }
    __syncthreads();

    // Accumulate dQ_dbw_dom and m0 gradient contributions
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
            int pos = wave_idx * n_res + res_idx;
            int bw_idx = bw_order[pos];  // map to unique BW index for scatter

            // Read bw_dom from per-position storage
            complex bw_dom_val(
                bw_dom_real[event_idx * n_total_positions + pos],
                bw_dom_imag[event_idx * n_total_positions + pos]);
            complex dQ_dbw_dom_contrib = dQ_dbw_p * (bw_p_val / bw_dom_val);

            // Accumulate shared memory at unique BW index
            atomicAdd(&s_dQ_dbw_dom_real[bw_idx], dQ_dbw_dom_contrib.real());
            atomicAdd(&s_dQ_dbw_dom_imag[bw_idx], dQ_dbw_dom_contrib.imag());

            // m0 gradient (atomic: multiple wave/res pairs can map to same bw_idx)
            double m0_val = m0[m0_index[bw_idx]];
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
        double m0_val = m0[m0_index[tid]];
        complex dm0(s_dQ_dbw_dom_real[tid], s_dQ_dbw_dom_imag[tid]);
        complex dg_bw = dm0 * complex(0.0, -m0_val);
        s_dQ_dg_bw_real[tid] = dg_bw.real();
        s_dQ_dg_bw_imag[tid] = dg_bw.imag();
    }
    __syncthreads();

    //=========================================================================
    // Phase 3: g0 gradient via matrix-vector multiply with shared memory
    //=========================================================================
    for (int gamma_idx = tid; gamma_idx < n_gamma_rows; gamma_idx += block_sz) {
        double sum_r = 0.0, sum_i = 0.0;
        for (int bw_idx = 0; bw_idx < n_unique_bw; bw_idx++) {
            double mg = matrix_gamma[gamma_idx * n_unique_bw + bw_idx];
            sum_r += s_dQ_dg_bw_real[bw_idx] * mg;
            sum_i += s_dQ_dg_bw_imag[bw_idx] * mg;
        }

        complex dQ_dg_val(sum_r, sum_i);
        complex g_interp_val(
            g_interp_real[event_idx * n_gamma_rows + gamma_idx],
            g_interp_imag[event_idx * n_gamma_rows + gamma_idx]);

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

//=============================================================================
// Host-callable launch functions
//=============================================================================
extern "C" {

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
    const double* mass, const double* g0,
    const int* g0_index, const int* g0_mass_index,
    const double* matrix_gamma,
    const double* gamma_table_real, const double* gamma_table_imag,
    double gamma_min, double gamma_delta,
    int n_gamma_rows, int n_unique_bw, int n_mass, int gamma_table_bins,
    double* g_interp_real, double* g_interp_imag,
    double* g_bw_real, double* g_bw_imag,
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
    const double* mass, const double* momentum, const double* angle,
    const double* frac, const double* time, const double* weight, const double* bkg,
    const int* m0_index, const int* fl_type,
    const int* mass_index, const int* fl_q_index,
    const int* bw_order, const int* fl_order, const int* angle_index,
    const double* angle_k, const double* angle_b,
    const double* matrix_angle_real, const double* matrix_angle_imag,
    const double* g_bw_real, const double* g_bw_imag,
    const double* fl_table,
    double fl_min, double fl_delta,
    int n_wave, int n_res, int n_decay, int n_unique_bw,
    int n_mass, int n_momentum, int n_angle_k, int n_angle_total,
    int fl_table_bins,
    const double* ck_real, const double* ck_imag, const double* m0,
    double Gamma, double Delta_Gamma, double Delta_m,
    double A_p, double poq_rho, double pop_phi,
    double* Q_out, double* P_out,
    double* pap_real, double* pap_imag, double* pam_real, double* pam_imag,
    double* gp_real, double* gp_imag, double* gm_real, double* gm_imag,
    double* poq_real, double* poq_imag,
    double* bw_p_real, double* bw_p_imag,
    double* common_amp_factor_real, double* common_amp_factor_imag,
    double* ap_real, double* ap_imag, double* am_real, double* am_imag, double* dQ_dP,
    double* bw_dom_real, double* bw_dom_imag,
    int n_events, int use_norm, double norm) {

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

// Merged-index forward launch
// Uses pre-merged bw_m0_index / bw_mass_index (n_total_positions elements)
// and pre-scattered g_bw_real / g_bw_imag (N, n_total_positions).
void launch_compute_main_merged(
    const double* mass, const double* momentum, const double* angle,
    const double* frac, const double* time, const double* weight, const double* bkg,
    const int* bw_m0_index, const int* fl_type,
    const int* bw_mass_index, const int* fl_q_index,
    const int* fl_order, const int* angle_index,
    const double* angle_k, const double* angle_b,
    const double* matrix_angle_real, const double* matrix_angle_imag,
    const double* g_bw_real, const double* g_bw_imag,
    const double* fl_table,
    double fl_min, double fl_delta,
    int n_wave, int n_res, int n_decay, int n_total_positions,
    int n_mass, int n_momentum, int n_angle_k, int n_angle_total,
    int fl_table_bins,
    const double* ck_real, const double* ck_imag, const double* m0,
    double Gamma, double Delta_Gamma, double Delta_m,
    double A_p, double poq_rho, double pop_phi,
    double* Q_out, double* P_out,
    double* pap_real, double* pap_imag, double* pam_real, double* pam_imag,
    double* gp_real, double* gp_imag, double* gm_real, double* gm_imag,
    double* poq_real, double* poq_imag,
    double* bw_p_real, double* bw_p_imag,
    double* common_amp_factor_real, double* common_amp_factor_imag,
    double* ap_real, double* ap_imag, double* am_real, double* am_imag, double* dQ_dP,
    double* bw_dom_real, double* bw_dom_imag,
    int n_events, int use_norm, double norm) {

    int block_size = 256;
    compute_main_merged_kernel<<<n_events, block_size>>>(
        mass, momentum, angle, frac, time, weight, bkg,
        bw_m0_index, fl_type, bw_mass_index, fl_q_index,
        fl_order, angle_index,
        angle_k, angle_b, matrix_angle_real, matrix_angle_imag,
        g_bw_real, g_bw_imag, fl_table,
        fl_min, fl_delta,
        n_wave, n_res, n_decay, n_total_positions,
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
    const double* P, const double* pap_real, const double* pap_imag,
    const double* pam_real, const double* pam_imag,
    const double* gp_real, const double* gp_imag,
    const double* gm_real, const double* gm_imag,
    const double* poq_real, const double* poq_imag,
    const double* bw_p_real, const double* bw_p_imag,
    const double* common_amp_factor_real, const double* common_amp_factor_imag,
    const double* ap_real, const double* ap_imag,
    const double* am_real, const double* am_imag,
    const double* dQ_dP,
    const double* bw_dom_real, const double* bw_dom_imag,
    const double* g_interp_real, const double* g_interp_imag,
    const double* g_bw_real, const double* g_bw_imag,
    const double* frac, const double* time,
    const int* m0_index, const int* g0_index, const int* bw_order,
    const double* matrix_gamma,
    const double* m0, const double* g0, const double* ck_real, const double* ck_imag,
    double Gamma, double Delta_Gamma, double Delta_m,
    double A_p, double poq_rho, double pop_phi,
    int n_wave, int n_res, int n_unique_bw, int n_gamma_rows, int n_mass,
    double* grad_ck_real_partial, double* grad_ck_imag_partial,
    double* grad_m0_partial, double* grad_g0_partial,
    double* grad_Gamma_partial, double* grad_DeltaGamma_partial,
    double* grad_DeltaM_partial, double* grad_Ap_partial,
    double* grad_poq_rho_partial, double* grad_pop_phi_partial,
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

// Merged-index gradient launch
// bw_dom_real/imag are (N, n_total_positions) — per-position storage.
// bw_order is still needed to map back to unique BW indices for gradient scatter.
// g_bw_real/imag remain (N, n_unique_bw) — original non-scattered storage.
void launch_gradient_merged(
    const double* P, const double* pap_real, const double* pap_imag,
    const double* pam_real, const double* pam_imag,
    const double* gp_real, const double* gp_imag,
    const double* gm_real, const double* gm_imag,
    const double* poq_real, const double* poq_imag,
    const double* bw_p_real, const double* bw_p_imag,
    const double* common_amp_factor_real, const double* common_amp_factor_imag,
    const double* ap_real, const double* ap_imag,
    const double* am_real, const double* am_imag,
    const double* dQ_dP,
    const double* bw_dom_real, const double* bw_dom_imag,
    const double* g_interp_real, const double* g_interp_imag,
    const double* g_bw_real, const double* g_bw_imag,
    const double* frac, const double* time,
    const int* m0_index, const int* g0_index, const int* bw_order,
    const double* matrix_gamma,
    const double* m0, const double* g0, const double* ck_real, const double* ck_imag,
    double Gamma, double Delta_Gamma, double Delta_m,
    double A_p, double poq_rho, double pop_phi,
    int n_wave, int n_res, int n_unique_bw, int n_total_positions,
    int n_gamma_rows, int n_mass,
    double* grad_ck_real_partial, double* grad_ck_imag_partial,
    double* grad_m0_partial, double* grad_g0_partial,
    double* grad_Gamma_partial, double* grad_DeltaGamma_partial,
    double* grad_DeltaM_partial, double* grad_Ap_partial,
    double* grad_poq_rho_partial, double* grad_pop_phi_partial,
    int n_events) {

    int block_size = 256;
    gradient_merged_kernel<<<n_events, block_size>>>(
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
        n_wave, n_res, n_unique_bw, n_total_positions, n_gamma_rows, n_mass,
        grad_ck_real_partial, grad_ck_imag_partial,
        grad_m0_partial, grad_g0_partial,
        grad_Gamma_partial, grad_DeltaGamma_partial,
        grad_DeltaM_partial, grad_Ap_partial,
        grad_poq_rho_partial, grad_pop_phi_partial,
        n_events);
    CUDA_CHECK(cudaGetLastError());
}

// Reduction launch wrappers (same as original)
void launch_reduce_sum(const double* input, double* output, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    reduce_sum_kernel<<<grid_size, block_size, block_size * sizeof(double)>>>(input, output, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_reduce_sum_complex(const double* real_in, const double* imag_in,
    double* real_out, double* imag_out, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    reduce_sum_complex_kernel<<<grid_size, block_size, 2 * block_size * sizeof(double)>>>(
        real_in, imag_in, real_out, imag_out, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_reduce_sum_features(const double* input, double* output,
    int n_events, int n_features) {
    int block_size = 256;
    reduce_sum_features_kernel<<<n_features, block_size, block_size * sizeof(double)>>>(
        input, output, n_events, n_features);
    CUDA_CHECK(cudaGetLastError());
}

void launch_reduce_sum_complex_features(const double* real_in, const double* imag_in,
    double* real_out, double* imag_out,
    int n_events, int n_features) {
    int block_size = 256;
    reduce_sum_complex_features_kernel<<<n_features, block_size, 2 * block_size * sizeof(double)>>>(
        real_in, imag_in, real_out, imag_out, n_events, n_features);
    CUDA_CHECK(cudaGetLastError());
}

// Scatter g_bw from (N, n_unique_bw) to (N, n_total_positions) via bw_order
__global__ void scatter_g_bw_kernel(
    const double* __restrict__ src_real, const double* __restrict__ src_imag,
    const int* __restrict__ bw_order,
    int n_events, int n_unique_bw, int n_total_positions,
    double* __restrict__ dst_real, double* __restrict__ dst_imag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_events * n_total_positions;
    if (idx >= total) return;
    int event_idx = idx / n_total_positions;
    int pos = idx % n_total_positions;
    int bw_idx = bw_order[pos];
    dst_real[idx] = src_real[event_idx * n_unique_bw + bw_idx];
    dst_imag[idx] = src_imag[event_idx * n_unique_bw + bw_idx];
}

void launch_scatter_g_bw(
    const double* src_real, const double* src_imag,
    const int* bw_order,
    int n_events, int n_unique_bw, int n_total_positions,
    double* dst_real, double* dst_imag) {
    int total = n_events * n_total_positions;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    scatter_g_bw_kernel<<<grid_size, block_size>>>(
        src_real, src_imag, bw_order, n_events, n_unique_bw, n_total_positions,
        dst_real, dst_imag);
    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"
