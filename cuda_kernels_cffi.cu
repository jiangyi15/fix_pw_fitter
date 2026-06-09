/**
 * COMPLETE CUDA kernels - FIXED for complex gamma_table
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

// Interpolation for complex table
__device__ complex interp_complex_device(
    const double* table_real,
    const double* table_imag,
    int type_idx,
    double x,
    double xmin,
    double xdelta,
    int n_bins
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

// Interpolation for real table
__device__ double interp_real_device(
    const double* table,
    int type_idx,
    double x,
    double xmin,
    double xdelta,
    int n_bins
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
    const double* matrix_angle_real, const double* matrix_angle_imag,  // Complex matrix
    const double* matrix_gamma,
    const double* gamma_table_real, const double* gamma_table_imag,  // Complex table
    const double* fl_table,  // Real table
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
    double* g_interp_real_out, double* g_interp_imag_out,  // Complex output
    double* g_bw_real_out, double* g_bw_imag_out,  // Complex output
    int n_events, int use_norm, double norm
) {
    int event_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (event_idx >= n_events) return;

    // Line 77-82: BW propagators
    // g0_index has length n_gamma_rows (288), so we must iterate over all of them
    // This gives us g_interp and g with 288 elements each
    // Then g_bw = g @ matrix_gamma where matrix_gamma is (288, 216)
    for (int gamma_idx = 0; gamma_idx < n_gamma_rows; gamma_idx++) {
        double g0_val = g0[g0_index[gamma_idx]];
        double mass_val = mass[event_idx * n_mass + g0_mass_index[gamma_idx]];
        
        // Complex interpolation for gamma_table
        complex g_interp_val = interp_complex_device(
            gamma_table_real, gamma_table_imag,
            g0_index[gamma_idx], mass_val,
            gamma_min, gamma_delta, gamma_table_bins
        );
        
        g_interp_real_out[event_idx * n_gamma_rows + gamma_idx] = g_interp_val.real();
        g_interp_imag_out[event_idx * n_gamma_rows + gamma_idx] = g_interp_val.imag();
        
        complex g = g0_val * g_interp_val;
        
        // Accumulate g_bw = dot(g, matrix_gamma)
        // g has shape (n_events, n_gamma_rows=288)
        // matrix_gamma has shape (n_gamma_rows=288, n_unique_bw=216)
        // g_bw should have shape (n_events, n_unique_bw=216)
        // g_bw[j] = sum over gamma_idx of (g[gamma_idx] * matrix_gamma[gamma_idx, j])
        for (int j = 0; j < n_unique_bw; j++) {
            if (gamma_idx == 0) {
                g_bw_real_out[event_idx * n_unique_bw + j] = 0.0;
                g_bw_imag_out[event_idx * n_unique_bw + j] = 0.0;
            }
            double mg = matrix_gamma[gamma_idx * n_unique_bw + j];
            g_bw_real_out[event_idx * n_unique_bw + j] += g.real() * mg;
            g_bw_imag_out[event_idx * n_unique_bw + j] += g.imag() * mg;
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
            
            complex g_bw_val(
                g_bw_real_out[event_idx * n_unique_bw + bw_idx],
                g_bw_imag_out[event_idx * n_unique_bw + bw_idx]
            );
            
            double m0_sq = m0_val * m0_val;
            double mass_sq = mass_val * mass_val;
            // bw_dom = m0² - m² - 1j*m0*g_bw
            // If g_bw = a + 1j*b, then:
            // bw_dom.real = m0² - m² + m0*b (where b = g_bw.imag)
            // bw_dom.imag = -m0*a (where a = g_bw.real)
            complex bw_dom(m0_sq - mass_sq + m0_val * g_bw_val.imag(), 
                          -m0_val * g_bw_val.real());
            
            bw_p_val *= bw_dom;
            
            bw_dom_real[event_idx * n_unique_bw + bw_idx] = bw_dom.real();
            bw_dom_imag[event_idx * n_unique_bw + bw_idx] = bw_dom.imag();
        }
        
        bw_p_real[event_idx * n_wave + wave_idx] = bw_p_val.real();
        bw_p_imag[event_idx * n_wave + wave_idx] = bw_p_val.imag();
    }
    
    // Line 100-103: Angular factors
    // angle_k and angle_b have shape (n_angle_k, 3)
    // angle has shape (n_events, n_angle_total, 3)
    double* ka_shared = new double[n_angle_k];
    for (int k_idx = 0; k_idx < n_angle_k; k_idx++) {
        int angle_pos = angle_index[k_idx];  // which angle position (0..n_angle_total-1)
        
        // Compute product of cosines over 3 angle components
        double ka_prod = 1.0;
        for (int comp = 0; comp < 3; comp++) {
            // angle is stored as (n_events, n_angle_total, 3)
            int angle_idx = event_idx * n_angle_total * 3 + angle_pos * 3 + comp;
            double angle_val = angle[angle_idx];
            
            // angle_k and angle_b are stored as (n_angle_k, 3)
            double k_val = angle_k[k_idx * 3 + comp];
            double b_val = angle_b[k_idx * 3 + comp];
            
            ka_prod *= cos(angle_val * k_val + b_val);
        }
        ka_shared[k_idx] = ka_prod;
    }
    
    // Compute fa = dot(ka, matrix_angle) - complex result
    for (int wave_idx = 0; wave_idx < n_wave; wave_idx++) {
        complex fa(0.0, 0.0);
        for (int k_idx = 0; k_idx < n_angle_k; k_idx++) {
            int idx = k_idx * n_wave + wave_idx;
            double ma_real = matrix_angle_real[idx];
            double ma_imag = matrix_angle_imag[idx];
            fa += ka_shared[k_idx] * complex(ma_real, ma_imag);
        }
        
        // Line 94-98: FL factor for this wave (real interpolation)
        double fl_p = 1.0;
        for (int decay_idx = 0; decay_idx < n_decay; decay_idx++) {
            int order_idx = wave_idx * n_decay + decay_idx;
            int fl_idx = fl_order[order_idx];
            double fl_q_val = momentum[event_idx * n_momentum + fl_q_index[fl_idx]];
            double fl_val = interp_real_device(fl_table, fl_type[fl_idx],
                                               fl_q_val, fl_min, fl_delta, fl_table_bins);
            fl_p *= fl_val;
        }
        
        // Line 106-108: common_amp_factor
        complex bw_p_val(bw_p_real[event_idx * n_wave + wave_idx],
                        bw_p_imag[event_idx * n_wave + wave_idx]);
        complex one_over_bw = complex(1.0, 0.0) / bw_p_val;
        complex fa_times_fl = fa * fl_p;
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

// Backward kernel - similar fixes
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
    const double* g_interp_real, const double* g_interp_imag,
    const double* g_bw_real, const double* g_bw_imag,
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
    
    complex dQ_dap = dQ_dpb * d_pb_dap + dQ_dpbbar * d_pbbar_dap;
    complex dQ_dam = dQ_dpb * d_pb_dam + dQ_dpbbar * d_pbbar_dam;
    
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
    
    // ==================== GRADIENT FOR m0 (REAL PARAMETER) ====================
    // bw_dom = m0² - m² - 1j*m0*g_bw
    // dbw_dom/dm0 = 2*m0 - 1j*g_bw (complex derivative)
    // For real m0 using Wirtinger: ∂Q/∂m0 = 2*Re(dQ_dbw_dom * (2*m0 - 1j*g_bw))
    // We accumulate over all wave/res pairs that map to each bw_idx
    
    // Initialize m0 partials to zero
    for (int bw_idx = 0; bw_idx < n_unique_bw; bw_idx++) {
        grad_m0_partial[event_idx * n_unique_bw + bw_idx] = 0.0;
    }
    
    // Iterate over wave/res pairs to accumulate m0 gradient contributions
    for (int wave_idx = 0; wave_idx < n_wave; wave_idx++) {
        // Load bw_p for this wave
        complex bw_p_val(bw_p_real[event_idx * n_wave + wave_idx],
                         bw_p_imag[event_idx * n_wave + wave_idx]);
        complex one_over_bw = complex(1.0, 0.0) / bw_p_val;
        
        // Load common_amp_factor for this wave
        complex common_amp(common_amp_factor_real[event_idx * n_wave + wave_idx],
                          common_amp_factor_imag[event_idx * n_wave + wave_idx]);
        
        // Determine dQ_da for this wave (Wirtinger derivative)
        complex dQ_da = (wave_idx < n_wave_half) ? dQ_dap : dQ_dam;
        
        // Load ck for this wave
        complex ck(ck_real[wave_idx], ck_imag[wave_idx]);
        
        // dQ_dbw_p = dQ_da * (-ck * one_over_bw * common_amp)
        complex dQ_dbw_p = dQ_da * (-ck * one_over_bw * common_amp);
        
        // Backprop through product: bw_p = prod(bw_dom_i)
        for (int res_idx = 0; res_idx < n_res; res_idx++) {
            int order_idx = wave_idx * n_res + res_idx;
            int bw_idx = bw_order[order_idx];
            
            // Load bw_dom for this bw_idx
            complex bw_dom_val(bw_dom_real[event_idx * n_unique_bw + bw_idx],
                              bw_dom_imag[event_idx * n_unique_bw + bw_idx]);
            
            // prod_except_i = bw_p / bw_dom_val (complex division)
            complex dQ_dbw_dom_contrib = dQ_dbw_p * (bw_p_val / bw_dom_val);
            
            // m0 gradient contribution for this bw_idx
            double m0_val = m0[m0_index[bw_idx]];
            complex g_bw_val(g_bw_real[event_idx * n_unique_bw + bw_idx],
                            g_bw_imag[event_idx * n_unique_bw + bw_idx]);
            
            // dbw_dom_dm0 = 2*m0 - 1j*g_bw
            complex dbw_dom_dm0 = complex(2.0 * m0_val, 0.0) - complex(0.0, 1.0) * g_bw_val;
            
            // Wirtinger gradient for real parameter: ∂Q/∂m0 = 2*Re(∂Q/∂bw_dom * ∂bw_dom/∂m0)
            grad_m0_partial[event_idx * n_unique_bw + bw_idx] += 2.0 * (dQ_dbw_dom_contrib * dbw_dom_dm0).real();
        }
    }
    
    // ==================== GRADIENT FOR g0 (REAL PARAMETER) ====================
    // Chain: g0 (real) -> g (complex) -> g_bw (complex) -> bw_dom (complex) -> Q (real)
    //
    // dQ_dg_bw[bw_idx] = dQ_dbw_dom[bw_idx] * (-1j * m0_all[bw_idx])
    // dQ_dg[gamma_idx] = sum_{bw_idx} dQ_dg_bw[bw_idx] * matrix_gamma[gamma_idx, bw_idx]
    // grad_g0[gamma_idx] = 2 * Re(dQ_dg[gamma_idx] * g_interp[gamma_idx])
    //
    // To avoid storing dQ_dbw_dom for all bw_idx, we recompute contributions
    // by iterating over wave/res pairs for each gamma_idx (outer loop).
    // This gives O(n_gamma_rows * n_wave * n_res) complexity per event.
    
    // Initialize g0 partials to zero
    for (int gamma_idx = 0; gamma_idx < n_gamma_rows; gamma_idx++) {
        grad_g0_partial[event_idx * n_gamma_rows + gamma_idx] = 0.0;
    }
    
    for (int gamma_idx = 0; gamma_idx < n_gamma_rows; gamma_idx++) {
        complex dQ_dg_val(0.0, 0.0);
        
        for (int wave_idx = 0; wave_idx < n_wave; wave_idx++) {
            // Reload bw_p, common_amp, and ck for this wave
            complex bw_p_val(bw_p_real[event_idx * n_wave + wave_idx],
                             bw_p_imag[event_idx * n_wave + wave_idx]);
            complex one_over_bw = complex(1.0, 0.0) / bw_p_val;
            complex common_amp(common_amp_factor_real[event_idx * n_wave + wave_idx],
                              common_amp_factor_imag[event_idx * n_wave + wave_idx]);
            complex dQ_da = (wave_idx < n_wave_half) ? dQ_dap : dQ_dam;
            complex ck(ck_real[wave_idx], ck_imag[wave_idx]);
            complex dQ_dbw_p = dQ_da * (-ck * one_over_bw * common_amp);
            
            for (int res_idx = 0; res_idx < n_res; res_idx++) {
                int order_idx = wave_idx * n_res + res_idx;
                int bw_idx = bw_order[order_idx];
                
                complex bw_dom_val(bw_dom_real[event_idx * n_unique_bw + bw_idx],
                                  bw_dom_imag[event_idx * n_unique_bw + bw_idx]);
                complex dQ_dbw_dom_contrib = dQ_dbw_p * (bw_p_val / bw_dom_val);
                
                double m0_val = m0[m0_index[bw_idx]];
                double mg = matrix_gamma[gamma_idx * n_unique_bw + bw_idx];
                
                // dQ_dg_bw = dQ_dbw_dom * (-1j * m0_all) = dQ_dbw_dom * complex(0, -m0)
                dQ_dg_val += dQ_dbw_dom_contrib * complex(0.0, -m0_val) * mg;
            }
        }
        
        // g_interp for this gamma_idx
        complex g_interp_val(g_interp_real[event_idx * n_gamma_rows + gamma_idx],
                            g_interp_imag[event_idx * n_gamma_rows + gamma_idx]);
        
        // Wirtinger gradient for real parameter: ∂Q/∂g0 = 2*Re(dQ_dg * g_interp)
        grad_g0_partial[event_idx * n_gamma_rows + gamma_idx] = 2.0 * (dQ_dg_val * g_interp_val).real();
    }
    
    // Time evolution gradients
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

// Column-wise reduction: sum over events (first dimension) for each feature
// Input: input[feat + event_idx * n_features] for event_idx in [0, n_events)
// Output: output[feat] for feat in [0, n_features)
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

// Column-wise reduction for complex values
__global__ void reduce_sum_complex_features_kernel(
    const double* real_in, const double* imag_in,
    double* real_out, double* imag_out,
    int n_events, int n_features
) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int feat = blockIdx.x;
    
    if (feat >= n_features) return;
    
    double real_sum = 0.0;
    double imag_sum = 0.0;
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
        const double* matrix_angle_real, const double* matrix_angle_imag,
        const double* matrix_gamma,
        const double* gamma_table_real, const double* gamma_table_imag,
        const double* fl_table,
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
        double* bw_dom_real, double* bw_dom_imag,
        double* g_interp_real, double* g_interp_imag,
        double* g_bw_real, double* g_bw_imag,
        int n_events, int use_norm, double norm) {
        int block_size = 256;
        int grid_size = (n_events + block_size - 1) / block_size;
        forward_kernel<<<grid_size, block_size>>>(mass, momentum, angle, frac, time, weight, bkg,
            m0_index, g0_index, fl_type, mass_index, g0_mass_index, fl_q_index, bw_order, fl_order,
            angle_index, angle_k, angle_b, matrix_angle_real, matrix_angle_imag, matrix_gamma,
            gamma_table_real, gamma_table_imag, fl_table,
            gamma_min, gamma_delta, fl_min, fl_delta, n_wave, n_res, n_decay, n_unique_bw, n_gamma_rows,
            n_mass, n_momentum, n_angle_k, n_angle_total, gamma_table_bins, fl_table_bins,
            ck_real, ck_imag, m0, g0, Gamma, Delta_Gamma, Delta_m, A_p, poq_rho, pop_phi,
            Q_out, P_out, pap_real, pap_imag, pam_real, pam_imag, gp_real, gp_imag, gm_real, gm_imag,
            poq_real, poq_imag, bw_p_real, bw_p_imag, common_amp_factor_real, common_amp_factor_imag,
            ap_real, ap_imag, am_real, am_imag, dQ_dP, bw_dom_real, bw_dom_imag,
            g_interp_real, g_interp_imag, g_bw_real, g_bw_imag,
            n_events, use_norm, norm);
        CUDA_CHECK(cudaGetLastError());
    }
    
    void launch_backward(const double* P, const double* pap_real, const double* pap_imag,
        const double* pam_real, const double* pam_imag, const double* gp_real, const double* gp_imag,
        const double* gm_real, const double* gm_imag, const double* poq_real, const double* poq_imag,
        const double* bw_p_real, const double* bw_p_imag, const double* common_amp_factor_real,
        const double* common_amp_factor_imag, const double* ap_real, const double* ap_imag,
        const double* am_real, const double* am_imag, const double* dQ_dP,
        const double* bw_dom_real, const double* bw_dom_imag,
        const double* g_interp_real, const double* g_interp_imag,
        const double* g_bw_real, const double* g_bw_imag,
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
            dQ_dP, bw_dom_real, bw_dom_imag, g_interp_real, g_interp_imag, g_bw_real, g_bw_imag,
            frac, time, weight, m0_index, g0_index, bw_order, matrix_gamma, m0, g0, ck_real, ck_imag,
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
}
