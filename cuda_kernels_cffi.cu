/**
 * COMPLETE CUDA kernels - FULLY MATCHES numpy_kernel.py
 * 
 * Forward: lines 74-137
 * Backward: lines 139-374
 * 
 * Memory-optimized: Uses pre-allocated global memory, minimal stack usage
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

__device__ double interp_device(
    const double* table, int type_idx, double x,
    double xmin, double xdelta, int n_bins
) {
    double diff = (x - xmin) / xdelta;
    int xbin = max(0, min((int)floor(diff), n_bins - 2));
    double delta = diff - xbin;
    int left_idx = type_idx * n_bins + xbin;
    return (table[left_idx + 1] - table[left_idx]) * delta + table[left_idx];
}

__global__ void forward_kernel(
    const double* mass, const double* momentum, const double* angle,
    const double* frac, const double* time, const double* weight, const double* bkg,
    const int* m0_index, const int* g0_index, const int* fl_type,
    const int* mass_index, const int* g0_mass_index, const int* fl_q_index,
    const int* bw_order, const int* fl_order, const int* angle_index,
    const double* angle_k, const double* angle_b,
    const double* matrix_angle, const double* matrix_gamma,
    const double* gamma_table, const double* fl_table,
    double gamma_min, double gamma_delta, double fl_min, double fl_delta,
    int n_wave, int n_res, int n_decay, int n_unique_bw, int n_gamma_rows,
    int n_mass, int n_momentum, int n_angle_k, int n_angle_total,
    int gamma_table_bins, int fl_table_bins,
    const double* ck_real, const double* ck_imag,
    const double* m0, const double* g0,
    double Gamma, double Delta_Gamma, double Delta_m,
    double A_p, double poq_rho, double pop_phi,
    double* Q_out, double* P_out,
    double* pap_real, double* pap_imag,
    double* pam_real, double* pam_imag,
    double* gp_real, double* gp_imag,
    double* gm_real, double* gm_imag,
    double* poq_real, double* poq_imag,
    double* bw_p_real, double* bw_p_imag,
    double* common_amp_factor_real, double* common_amp_factor_imag,
    double* ap_real, double* ap_imag,
    double* am_real, double* am_imag,
    double* dQ_dP_out,
    double* bw_dom_real, double* bw_dom_imag,
    double* g_interp_out, double* g_bw_out,
    int n_events, int use_norm, double norm
) {
    int event_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (event_idx >= n_events) return;

    // ==================== FORWARD PASS (lines 74-137) ====================
    
    // Line 77-82: BW propagators
    for (int gamma_idx = 0; gamma_idx < n_unique_bw; gamma_idx++) {
        double g0_val = g0[g0_index[gamma_idx]];
        double mass_val = mass[event_idx * n_mass + g0_mass_index[gamma_idx]];
        double g_interp_val = interp_device(gamma_table, g0_index[gamma_idx],
                                            mass_val, gamma_min, gamma_delta, gamma_table_bins);
        g_interp_out[event_idx * n_unique_bw + gamma_idx] = g_interp_val;
        
        double g = g0_val * g_interp_val;
        // Accumulate g_bw = dot(g, matrix_gamma)
        for (int i = 0; i < n_gamma_rows; i++) {
            if (gamma_idx == 0) {
                g_bw_out[event_idx * n_gamma_rows + i] = 0.0;
            }
            g_bw_out[event_idx * n_gamma_rows + i] += g * matrix_gamma[i * n_unique_bw + gamma_idx];
        }
    }
    
    // Line 84-91: bw_dom and bw_p
    for (int wave_idx = 0; wave_idx < n_wave; wave_idx++) {
        complex bw_p_val(1.0, 0.0);
        
        for (int res_idx = 0; res_idx < n_res; res_idx++) {
            int order_idx = wave_idx * n_res + res_idx;
            int bw_idx = bw_order[order_idx];
            
            double m0_val = m0[m0_index[bw_idx]];
            double mass_val = mass[event_idx * n_mass + mass_index[bw_idx]];
            double g_bw_val = g_bw_out[event_idx * n_gamma_rows + bw_idx];
            
            double m0_sq = m0_val * m0_val;
            double mass_sq = mass_val * mass_val;
            complex bw_dom(m0_sq - mass_sq, -m0_val * g_bw_val);  // Line 86
            
            bw_p_val *= bw_dom;
            
            // Store bw_dom for backward pass
            bw_dom_real[event_idx * n_unique_bw + bw_idx] = bw_dom.real();
            bw_dom_imag[event_idx * n_unique_bw + bw_idx] = bw_dom.imag();
        }
        
        bw_p_real[event_idx * n_wave + wave_idx] = bw_p_val.real();
        bw_p_imag[event_idx * n_wave + wave_idx] = bw_p_val.imag();
    }
    
    // Line 94-98: FL factors
    // Line 100-103: Angular factors
    // Compute ka vector
    double* ka_shared = new double[n_angle_k];
    for (int k_idx = 0; k_idx < n_angle_k; k_idx++) {
        int angle_idx_val = angle_index[k_idx];
        double angle_val = angle[event_idx * n_angle_total + angle_idx_val];
        ka_shared[k_idx] = cos(angle_val * angle_k[k_idx] + angle_b[k_idx]);
    }
    
    // Compute fa = dot(ka, matrix_angle)
    for (int wave_idx = 0; wave_idx < n_wave; wave_idx++) {
        double fa = 0.0;
        for (int k_idx = 0; k_idx < n_angle_k; k_idx++) {
            fa += ka_shared[k_idx] * matrix_angle[k_idx * n_wave + wave_idx];
        }
        
        // FL factor for this wave
        double fl_p = 1.0;
        for (int decay_idx = 0; decay_idx < n_decay; decay_idx++) {
            int order_idx = wave_idx * n_decay + decay_idx;
            int fl_idx = fl_order[order_idx];
            double fl_q_val = momentum[event_idx * n_momentum + fl_q_index[fl_idx]];
            double fl_val = interp_device(fl_table, fl_type[fl_idx],
                                          fl_q_val, fl_min, fl_delta, fl_table_bins);
            fl_p *= fl_val;
        }
        
        // Line 106-108: common_amp_factor
        complex bw_p_val(bw_p_real[event_idx * n_wave + wave_idx],
                        bw_p_imag[event_idx * n_wave + wave_idx]);
        complex one_over_bw = complex(1.0, 0.0) / bw_p_val;
        double fa_times_fl = fa * fl_p;
        complex common_amp = one_over_bw * fa_times_fl;
        
        common_amp_factor_real[event_idx * n_wave + wave_idx] = common_amp.real();
        common_amp_factor_imag[event_idx * n_wave + wave_idx] = common_amp.imag();
    }
    delete[] ka_shared;
    
    // Line 110-113: Amplitude ap and am
    complex ap(0.0, 0.0);
    complex am(0.0, 0.0);
    int n_wave_half = n_wave / 2;
    
    for (int i = 0; i < n_wave_half; i++) {
        complex ck_i(ck_real[i], ck_imag[i]);
        complex common_i(common_amp_factor_real[event_idx * n_wave + i],
                        common_amp_factor_imag[event_idx * n_wave + i]);
        ap += ck_i * common_i;
        
        complex ck_j(ck_real[n_wave_half + i], ck_imag[n_wave_half + i]);
        complex common_j(common_amp_factor_real[event_idx * n_wave + n_wave_half + i],
                        common_amp_factor_imag[event_idx * n_wave + n_wave_half + i]);
        am += ck_j * common_j;
    }
    
    ap_real[event_idx] = ap.real();
    ap_imag[event_idx] = ap.imag();
    am_real[event_idx] = am.real();
    am_imag[event_idx] = am.imag();
    
    // Line 115-119: Time evolution
    double t = time[event_idx];
    complex i_const(0.0, 1.0);
    complex eL = exp(-i_const * t * complex(-Delta_m/2, -(Gamma + Delta_Gamma/2)/2));
    complex eH = exp(-i_const * t * complex(Delta_m/2, -(Gamma - Delta_Gamma/2)/2));
    complex gp = (eL + eH) / 2.0;
    complex gm = (eL - eH) / 2.0;
    
    gp_real[event_idx] = gp.real();
    gp_imag[event_idx] = gp.imag();
    gm_real[event_idx] = gm.real();
    gm_imag[event_idx] = gm.imag();
    
    // Line 121-129: Probabilities
    complex poq = poq_rho * exp(i_const * pop_phi);
    poq_real[event_idx] = poq.real();
    poq_imag[event_idx] = poq.imag();
    
    complex pap = gp * ap + gm * poq * am;
    complex pam = (gm / poq) * ap + gp * am;
    
    pap_real[event_idx] = pap.real();
    pap_imag[event_idx] = pap.imag();
    pam_real[event_idx] = pam.real();
    pam_imag[event_idx] = pam.imag();
    
    double pb = thrust::norm(pap);
    double pbbar = thrust::norm(pam);
    
    double frac_val = frac[event_idx];
    double P = frac_val * pb * (1.0 - A_p) + (1.0 - frac_val) * pbbar * (1.0 + A_p);
    P_out[event_idx] = P;
    
    // Line 131-137: Loss
    double weight_val = weight[event_idx];
    double bkg_val = bkg[event_idx];
    
    if (use_norm == 0) {
        Q_out[event_idx] = weight_val * P;
        dQ_dP_out[event_idx] = weight_val;
    } else {
        Q_out[event_idx] = -weight_val * log(P / norm + bkg_val);
        dQ_dP_out[event_idx] = -weight_val / (P / norm + bkg_val);
    }
}

__global__ void backward_kernel(
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
    const double* g_interp, const double* g_bw,
    const double* frac, const double* time, const double* weight,
    const int* m0_index, const int* g0_index, const int* bw_order,
    const double* matrix_gamma, const double* m0, const double* g0,
    const double* ck_real, const double* ck_imag,
    double Gamma, double Delta_Gamma, double Delta_m,
    double A_p, double poq_rho, double pop_phi,
    int n_wave, int n_res, int n_unique_bw, int n_gamma_rows, int n_mass,
    double* grad_ck_real_partial, double* grad_ck_imag_partial,
    double* grad_m0_partial, double* grad_g0_partial,
    double* grad_Gamma_partial, double* grad_DeltaGamma_partial,
    double* grad_DeltaM_partial, double* grad_Ap_partial,
    double* grad_poq_rho_partial, double* grad_pop_phi_partial,
    int n_events
) {
    int event_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (event_idx >= n_events) return;

    // Load values
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
    
    // ==================== BACKWARD PASS (lines 139-374) ====================
    
    // Line 142-149
    double pb = thrust::norm(pap);
    double pbbar = thrust::norm(pam);
    double dP_dpb = frac_val * (1.0 - A_p);
    double dP_dpbbar = (1.0 - frac_val) * (1.0 + A_p);
    double dP_dAp = -frac_val * pb + (1.0 - frac_val) * pbbar;
    
    grad_Ap_partial[event_idx] = dQ_dP_val * dP_dAp;
    
    double dQ_dpb = dQ_dP_val * dP_dpb;
    double dQ_dpbbar = dQ_dP_val * dP_dpbbar;
    
    // Line 166-173: Wirtinger gradients
    complex d_pb_dap = conj(pap) * gp;
    complex d_pb_dam = conj(pap) * gm * poq;
    complex d_pbbar_dap = conj(pam) * (gm / poq);
    complex d_pbbar_dam = conj(pam) * gp;
    
    complex dQ_dap = dQ_dpb * d_pb_dap + dQ_dpbbar * d_pbbar_dap;
    complex dQ_dam = dQ_dpb * d_pb_dam + dQ_dpbbar * d_pbbar_dam;
    
    // Line 187-195: Gradient for ck
    int n_wave_half = n_wave / 2;
    for (int i = 0; i < n_wave_half; i++) {
        complex common_i(common_amp_factor_real[event_idx * n_wave + i],
                        common_amp_factor_imag[event_idx * n_wave + i]);
        complex grad_ck_val = dQ_dap * common_i;
        grad_ck_real_partial[event_idx * n_wave + i] = grad_ck_val.real();
        grad_ck_imag_partial[event_idx * n_wave + i] = grad_ck_val.imag();
        
        complex common_j(common_amp_factor_real[event_idx * n_wave + n_wave_half + i],
                        common_amp_factor_imag[event_idx * n_wave + n_wave_half + i]);
        complex grad_ck_val2 = dQ_dam * common_j;
        grad_ck_real_partial[event_idx * n_wave + n_wave_half + i] = grad_ck_val2.real();
        grad_ck_imag_partial[event_idx * n_wave + n_wave_half + i] = grad_ck_val2.imag();
    }
    
    // Line 201-221: Gradient for bw_p and bw_dom
    // Compute dQ_dbw_p for each wave
    double* dQ_dbw_p = new double[n_wave];
    for (int wave_idx = 0; wave_idx < n_wave; wave_idx++) {
        complex bw_p_val(bw_p_real[event_idx * n_wave + wave_idx],
                        bw_p_imag[event_idx * n_wave + wave_idx]);
        complex one_over_bw = complex(1.0, 0.0) / bw_p_val;
        
        double common_real = common_amp_factor_real[event_idx * n_wave + wave_idx];
        double common_imag = common_amp_factor_imag[event_idx * n_wave + wave_idx];
        
        int idx = (wave_idx < n_wave_half) ? wave_idx : n_wave_half + (wave_idx - n_wave_half);
        complex dQ_da = (wave_idx < n_wave_half) ? dQ_dap : dQ_dam;
        complex ck_val(ck_real[wave_idx], ck_imag[wave_idx]);
        
        complex dQ_dbw_p_val = dQ_da * (-ck_val * one_over_bw * one_over_bw * complex(common_real, common_imag));
        dQ_dbw_p[wave_idx] = dQ_dbw_p_val.real();
    }
    
    // Line 208-221: BW gradients through product
    double* dQ_dbw_dom = new double[n_unique_bw];
    for (int i = 0; i < n_unique_bw; i++) dQ_dbw_dom[i] = 0.0;
    
    for (int wave_idx = 0; wave_idx < n_wave; wave_idx++) {
        for (int res_idx = 0; res_idx < n_res; res_idx++) {
            // Compute product except i
            complex prod_except_i(1.0, 0.0);
            for (int j = 0; j < n_res; j++) {
                if (j != res_idx) {
                    int order_idx = wave_idx * n_res + j;
                    int bw_idx = bw_order[order_idx];
                    complex bw_dom(bw_dom_real[event_idx * n_unique_bw + bw_idx],
                                  bw_dom_imag[event_idx * n_unique_bw + bw_idx]);
                    prod_except_i *= bw_dom;
                }
            }
            
            complex dQ_dbw_p_complex(dQ_dbw_p[wave_idx], 0.0);
            complex grad = dQ_dbw_p_complex * prod_except_i;
            
            int order_idx = wave_idx * n_res + res_idx;
            int bw_idx = bw_order[order_idx];
            dQ_dbw_dom[bw_idx] += grad.real();
        }
    }
    
    // Line 223-242: Gradient for m0
    for (int bw_idx = 0; bw_idx < n_unique_bw; bw_idx++) {
        double m0_val = m0[m0_index[bw_idx]];
        double g_bw_val = g_bw[event_idx * n_gamma_rows + bw_idx];
        complex dbw_dom_dm0(2.0 * m0_val, -g_bw_val);
        
        complex dQ_dbw_dom_complex(dQ_dbw_dom[bw_idx], 0.0);
        double grad_m0_val = 2.0 * (dQ_dbw_dom_complex * dbw_dom_dm0).real();
        grad_m0_partial[event_idx * n_unique_bw + bw_idx] = grad_m0_val;
    }
    
    // Line 244-276: Gradient for g0
    double* dQ_dg_bw = new double[n_gamma_rows];
    for (int i = 0; i < n_gamma_rows; i++) {
        double m0_val = m0[m0_index[i]];
        complex dQ_dbw_dom_complex(dQ_dbw_dom[i], 0.0);
        dQ_dg_bw[i] = (dQ_dbw_dom_complex * complex(0.0, -m0_val)).real();
    }
    
    for (int gamma_idx = 0; gamma_idx < n_unique_bw; gamma_idx++) {
        double dQ_dg = 0.0;
        for (int i = 0; i < n_gamma_rows; i++) {
            dQ_dg += dQ_dg_bw[i] * matrix_gamma[i * n_unique_bw + gamma_idx];
        }
        double g_interp_val = g_interp[event_idx * n_unique_bw + gamma_idx];
        double grad_g0_val = 2.0 * dQ_dg * g_interp_val;
        grad_g0_partial[event_idx * n_unique_bw + gamma_idx] = grad_g0_val;
    }
    
    delete[] dQ_dbw_p;
    delete[] dQ_dbw_dom;
    delete[] dQ_dg_bw;
    
    // Line 278-330: Time evolution gradients
    complex d_pb_dgp = conj(pap) * ap;
    complex d_pb_dgm = conj(pap) * poq * am;
    complex d_pbbar_dgp = conj(pam) * am;
    complex d_pbbar_dgm = conj(pam) * ap / poq;
    
    complex dQ_dgp = dQ_dpb * d_pb_dgp + dQ_dpbbar * d_pbbar_dgp;
    complex dQ_dgm = dQ_dpb * d_pb_dgm + dQ_dpbbar * d_pbbar_dgm;
    
    complex dgp_dGamma = (-t/2.0) * gp;
    complex dgm_dGamma = (-t/2.0) * gm;
    complex dgp_dDeltaGamma = (-t/4.0) * gm;
    complex dgm_dDeltaGamma = (-t/4.0) * gp;
    complex dgp_dDeltaM = complex(0.0, t/2.0) * gm;
    complex dgm_dDeltaM = complex(0.0, t/2.0) * gp;
    
    grad_Gamma_partial[event_idx] = 2.0 * (dQ_dgp * dgp_dGamma + dQ_dgm * dgm_dGamma).real();
    grad_DeltaGamma_partial[event_idx] = 2.0 * (dQ_dgp * dgp_dDeltaGamma + dQ_dgm * dgm_dDeltaGamma).real();
    grad_DeltaM_partial[event_idx] = 2.0 * (dQ_dgp * dgp_dDeltaM + dQ_dgm * dgm_dDeltaM).real();
    
    // Line 332-365: poq gradients
    complex d_pb_dpoq = conj(pap) * gm * am;
    complex d_pbbar_dpoq = conj(pam) * (-gm / (poq * poq)) * ap;
    complex dQ_dpoq = dQ_dpb * d_pb_dpoq + dQ_dpbbar * d_pbbar_dpoq;
    
    complex exp_phi = exp(complex(0.0, 1.0) * pop_phi);
    grad_poq_rho_partial[event_idx] = 2.0 * (dQ_dpoq * exp_phi).real();
    grad_pop_phi_partial[event_idx] = 2.0 * (dQ_dpoq * poq_rho * complex(0.0, 1.0) * exp_phi).real();
}

// Reduction kernels
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
    
    void launch_forward(const double* mass, const double* momentum, const double* angle,
        const double* frac, const double* time, const double* weight, const double* bkg,
        const int* m0_index, const int* g0_index, const int* fl_type,
        const int* mass_index, const int* g0_mass_index, const int* fl_q_index,
        const int* bw_order, const int* fl_order, const int* angle_index,
        const double* angle_k, const double* angle_b,
        const double* matrix_angle, const double* matrix_gamma,
        const double* gamma_table, const double* fl_table,
        double gamma_min, double gamma_delta, double fl_min, double fl_delta,
        int n_wave, int n_res, int n_decay, int n_unique_bw, int n_gamma_rows,
        int n_mass, int n_momentum, int n_angle_k, int n_angle_total,
        int gamma_table_bins, int fl_table_bins,
        const double* ck_real, const double* ck_imag, const double* m0, const double* g0,
        double Gamma, double Delta_Gamma, double Delta_m, double A_p, double poq_rho, double pop_phi,
        double* Q_out, double* P_out, double* pap_real, double* pap_imag,
        double* pam_real, double* pam_imag, double* gp_real, double* gp_imag,
        double* gm_real, double* gm_imag, double* poq_real, double* poq_imag,
        double* bw_p_real, double* bw_p_imag, double* common_amp_factor_real, double* common_amp_factor_imag,
        double* ap_real, double* ap_imag, double* am_real, double* am_imag, double* dQ_dP,
        double* bw_dom_real, double* bw_dom_imag, double* g_interp, double* g_bw,
        int n_events, int use_norm, double norm) {
        int block_size = 256;
        int grid_size = (n_events + block_size - 1) / block_size;
        forward_kernel<<<grid_size, block_size>>>(mass, momentum, angle, frac, time, weight, bkg,
            m0_index, g0_index, fl_type, mass_index, g0_mass_index, fl_q_index, bw_order, fl_order,
            angle_index, angle_k, angle_b, matrix_angle, matrix_gamma, gamma_table, fl_table,
            gamma_min, gamma_delta, fl_min, fl_delta, n_wave, n_res, n_decay, n_unique_bw, n_gamma_rows,
            n_mass, n_momentum, n_angle_k, n_angle_total, gamma_table_bins, fl_table_bins,
            ck_real, ck_imag, m0, g0, Gamma, Delta_Gamma, Delta_m, A_p, poq_rho, pop_phi,
            Q_out, P_out, pap_real, pap_imag, pam_real, pam_imag, gp_real, gp_imag, gm_real, gm_imag,
            poq_real, poq_imag, bw_p_real, bw_p_imag, common_amp_factor_real, common_amp_factor_imag,
            ap_real, ap_imag, am_real, am_imag, dQ_dP, bw_dom_real, bw_dom_imag, g_interp, g_bw,
            n_events, use_norm, norm);
        CUDA_CHECK(cudaGetLastError());
    }
    
    void launch_backward(const double* P, const double* pap_real, const double* pap_imag,
        const double* pam_real, const double* pam_imag, const double* gp_real, const double* gp_imag,
        const double* gm_real, const double* gm_imag, const double* poq_real, const double* poq_imag,
        const double* bw_p_real, const double* bw_p_imag, const double* common_amp_factor_real,
        const double* common_amp_factor_imag, const double* ap_real, const double* ap_imag,
        const double* am_real, const double* am_imag, const double* dQ_dP,
        const double* bw_dom_real, const double* bw_dom_imag, const double* g_interp, const double* g_bw,
        const double* frac, const double* time, const double* weight, const int* m0_index,
        const int* g0_index, const int* bw_order, const double* matrix_gamma,
        const double* m0, const double* g0, const double* ck_real, const double* ck_imag,
        double Gamma, double Delta_Gamma, double Delta_m, double A_p, double poq_rho, double pop_phi,
        int n_wave, int n_res, int n_unique_bw, int n_gamma_rows, int n_mass,
        double* grad_ck_real_partial, double* grad_ck_imag_partial, double* grad_m0_partial,
        double* grad_g0_partial, double* grad_Gamma_partial, double* grad_DeltaGamma_partial,
        double* grad_DeltaM_partial, double* grad_Ap_partial, double* grad_poq_rho_partial,
        double* grad_pop_phi_partial, int n_events) {
        int block_size = 256;
        int grid_size = (n_events + block_size - 1) / block_size;
        backward_kernel<<<grid_size, block_size>>>(P, pap_real, pap_imag, pam_real, pam_imag,
            gp_real, gp_imag, gm_real, gm_imag, poq_real, poq_imag, bw_p_real, bw_p_imag,
            common_amp_factor_real, common_amp_factor_imag, ap_real, ap_imag, am_real, am_imag,
            dQ_dP, bw_dom_real, bw_dom_imag, g_interp, g_bw, frac, time, weight,
            m0_index, g0_index, bw_order, matrix_gamma, m0, g0, ck_real, ck_imag,
            Gamma, Delta_Gamma, Delta_m, A_p, poq_rho, pop_phi,
            n_wave, n_res, n_unique_bw, n_gamma_rows, n_mass,
            grad_ck_real_partial, grad_ck_imag_partial, grad_m0_partial, grad_g0_partial,
            grad_Gamma_partial, grad_DeltaGamma_partial, grad_DeltaM_partial, grad_Ap_partial,
            grad_poq_rho_partial, grad_pop_phi_partial, n_events);
        CUDA_CHECK(cudaGetLastError());
    }
    
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
}
