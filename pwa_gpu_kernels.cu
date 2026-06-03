// PWA GPU CUDA kernels
// Single file with all computation for B->phi phi partial wave analysis

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>

typedef cuDoubleComplex cdouble;

// Complex helpers
__device__ cdouble cmul(cdouble a, cdouble b) { return cuCmul(a, b); }
__device__ cdouble cdiv(cdouble a, cdouble b) { return cuCdiv(a, b); }
__device__ cdouble cadd(cdouble a, cdouble b) { return cuCadd(a, b); }
__device__ cdouble csub(cdouble a, cdouble b) { return cuCsub(a, b); }
__device__ cdouble cscale(cdouble a, double s) { return make_cuDoubleComplex(cuCreal(a) * s, cuCimag(a) * s); }
__device__ cdouble cconj(cdouble a) { return cuConj(a); }
__device__ double cabs2(cdouble a) { return cuCreal(a) * cuCreal(a) + cuCimag(a) * cuCimag(a); }
__device__ cdouble make_c(double re, double im) { return make_cuDoubleComplex(re, im); }
__device__ cdouble cexp_c(cdouble z) {
    double exp_x = exp(cuCreal(z));
    double y = cuCimag(z);
    return make_cuDoubleComplex(exp_x * cos(y), exp_x * sin(y));
}

// ============================================================
// Compute kernel: one thread per event
// ============================================================
extern "C"
__global__ void pwa_compute_kernel(
    // Output
    double* __restrict__ p_out,           // (n_events,)
    cdouble* __restrict__ amp_p_out,      // (n_events,)
    cdouble* __restrict__ amp_m_out,      // (n_events,)
    // Gradient inputs (NULL in forward-only mode)
    const double* __restrict__ grad_p,    // (n_events,) = w/(N*q) or NULL
    // Gradient outputs (only written when grad_p != NULL)
    double* __restrict__ grad_ck_re,      // (n_waves,)
    double* __restrict__ grad_ck_im,      // (n_waves,)
    double* __restrict__ grad_m0_out,     // (n_m0,)
    double* __restrict__ grad_g0_out,     // (n_g0,)
    double* __restrict__ grad_delta_m_out,// (1,)
    double* __restrict__ grad_delta_g_out,// (1,)
    double* __restrict__ grad_g_out,      // (1,)
    double* __restrict__ grad_ap_out,     // (1,)
    double* __restrict__ grad_lam_out,    // (1,)
    double* __restrict__ grad_phi_out,    // (1,)
    double* __restrict__ grad_N_out,      // (1,)
    // Parameters
    const cdouble* __restrict__ ck,       // (n_waves,)
    const double* __restrict__ m0,        // (n_m0,)
    const double* __restrict__ g0,        // (n_g0,)
    double delta_m, double delta_g, double g_val,
    double ap, double lam, double phi,
    double N_val,
        // Data
    const double* __restrict__ mass_flat,
    const double* __restrict__ q_flat,
    const double* __restrict__ angles_flat,
    const double* __restrict__ time_arr,
    const double* __restrict__ frac_arr,
    // Config
    const int* __restrict__ bw_index,
    const int* __restrict__ gamma_index,
    const int* __restrict__ bw_order,
    const int* __restrict__ bf_index,
    const int* __restrict__ bf_order,
    const int* __restrict__ ang_index,
    const double* __restrict__ ang_k,
    const double* __restrict__ ang_b,
    const double* __restrict__ matrix_gamma,
    const cdouble* __restrict__ matrix_ang,
    const cdouble* __restrict__ gamma_table,
    const double* __restrict__ bf_table,
    // Dimensions
    int n_events, int n_waves, int n_m0, int n_g0,
    int n_res_per_wave, int n_decays_per_wave,
    int n_bf_types, int n_basis, int n_ang_per_basis,
    int n_gamma_points, int n_bf_points,
    int mass_stride, int q_stride, int ang_stride,
    double g_min, double g_delta, double q_min, double q_delta
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_events) return;

    // Recompute forward pass (needed for gradients)
    // [Same as compute kernel up to amp_p, amp_m]

    double m_bw[16];
    for (int i = 0; i < n_m0 && i < 16; i++) {
        m_bw[i] = mass_flat[e * mass_stride + bw_index[i]];
    }

    double m_gamma_vals[16];
    for (int i = 0; i < n_g0 && i < 16; i++) {
        m_gamma_vals[i] = mass_flat[e * mass_stride + gamma_index[i]];
    }

    cdouble gi[16];
    for (int i = 0; i < n_g0 && i < 16; i++) {
        double diff = (m_gamma_vals[i] - g_min) / g_delta;
        int idx = (int)diff;
        if (idx < 0) idx = 0;
        if (idx >= n_gamma_points - 1) idx = n_gamma_points - 2;
        double delta = diff - idx;
        cdouble fl = gamma_table[i * n_gamma_points + idx];
        cdouble fr = gamma_table[i * n_gamma_points + idx + 1];
        gi[i] = cadd(fl, cscale(csub(fr, fl), delta));
    }

    cdouble g_vals[16];
    for (int i = 0; i < n_g0 && i < 16; i++) g_vals[i] = cscale(gi[i], g0[i]);

    cdouble gamma[16];
    for (int i = 0; i < n_m0 && i < 16; i++) {
        cdouble sum = make_c(0.0, 0.0);
        for (int j = 0; j < n_g0 && j < 16; j++) {
            sum = cadd(sum, cscale(g_vals[j], matrix_gamma[i * n_g0 + j]));
        }
        gamma[i] = sum;
    }

    cdouble bwall[16];
    for (int i = 0; i < n_m0 && i < 16; i++) {
        bwall[i] = make_c(m0[i]*m0[i] - m_bw[i]*m_bw[i] + m0[i]*cuCimag(gamma[i]), -m0[i]*cuCreal(gamma[i]));
    }

    cdouble bw[200];
    int total_bw = n_waves * n_res_per_wave;
    for (int i = 0; i < total_bw && i < 200; i++) bw[i] = bwall[bw_order[i]];

    double q_bf[16];
    for (int i = 0; i < n_bf_types && i < 16; i++) {
        q_bf[i] = q_flat[e * q_stride + bf_index[i]];
    }

    double bfall[16];
    for (int i = 0; i < n_bf_types && i < 16; i++) {
        double diff = (q_bf[i] - q_min) / q_delta;
        int idx = (int)diff;
        if (idx < 0) idx = 0;
        if (idx >= n_bf_points - 1) idx = n_bf_points - 2;
        double delta = diff - idx;
        double fl = bf_table[i * n_bf_points + idx];
        double fr = bf_table[i * n_bf_points + idx + 1];
        bfall[i] = fl + (fr - fl) * delta;
    }

    double bf[300];
    int total_bf = n_waves * n_decays_per_wave;
    for (int i = 0; i < total_bf && i < 300; i++) bf[i] = bfall[bf_order[i]];

    double ang_f[16];
    for (int b = 0; b < n_basis && b < 16; b++) {
        double prod = 1.0;
        for (int j = 0; j < n_ang_per_basis; j++) {
            int idx = ang_index[b * n_ang_per_basis + j];
            double theta = angles_flat[e * ang_stride + idx];
            double val = theta * ang_k[b * n_ang_per_basis + j] + ang_b[b * n_ang_per_basis + j];
            prod *= cos(val);
        }
        ang_f[b] = prod;
    }

    cdouble ag[100];
    for (int w = 0; w < n_waves && w < 100; w++) {
        cdouble sum = make_c(0.0, 0.0);
        for (int b = 0; b < n_basis && b < 16; b++) {
            sum = cadd(sum, cmul(matrix_ang[w * n_basis + b], make_c(ang_f[b], 0.0)));
        }
        ag[w] = sum;
    }

    cdouble bwprod[100];
    for (int w = 0; w < n_waves && w < 100; w++) {
        cdouble prod = make_c(1.0, 0.0);
        for (int r = 0; r < n_res_per_wave; r++) prod = cmul(prod, bw[w * n_res_per_wave + r]);
        bwprod[w] = prod;
    }

    double bfprod[100];
    for (int w = 0; w < n_waves && w < 100; w++) {
        double prod = 1.0;
        for (int d = 0; d < n_decays_per_wave; d++) prod *= bf[w * n_decays_per_wave + d];
        bfprod[w] = prod;
    }

    cdouble inv_bwprod[100];
    for (int w = 0; w < n_waves && w < 100; w++) inv_bwprod[w] = cdiv(make_c(1.0, 0.0), bwprod[w]);

    cdouble a_full[100];
    for (int w = 0; w < n_waves && w < 100; w++) {
        cdouble tmp = cmul(ck[w], inv_bwprod[w]);
        tmp = cscale(tmp, bfprod[w]);
        a_full[w] = cmul(tmp, ag[w]);
    }

    cdouble amp0 = make_c(0.0, 0.0);
    cdouble amp1 = make_c(0.0, 0.0);
    int half_waves = n_waves / 2;
    for (int w = 0; w < half_waves; w++) amp0 = cadd(amp0, a_full[w]);
    for (int w = half_waves; w < n_waves; w++) amp1 = cadd(amp1, a_full[w]);

    double t = time_arr[e];
    double f = frac_arr[e];

    cdouble gL = cexp_c(make_c(-(g_val + delta_g) * t / 2.0, t * delta_m / 2.0));
    cdouble gH = cexp_c(make_c(-(g_val - delta_g) * t / 2.0, -t * delta_m / 2.0));

    cdouble gp = cscale(cadd(gL, gH), 0.5);
    cdouble gm = cscale(csub(gL, gH), 0.5);

    cdouble pq = cscale(cexp_c(make_c(0.0, phi)), lam);

    cdouble exp_miphii = cexp_c(make_c(0.0, -phi));

    // ---- Compute amp_p, amp_m, and p (always, for both forward and gradient) ----
    // amp_p = gp * amp0 + gm * pq * amp1
    cdouble amp_p = cadd(cmul(amp0, gp), cmul(cmul(gm, pq), amp1));
    // amp_m = gm/pq * amp0 + gp * amp1
    cdouble amp_m = cadd(cmul(cdiv(gm, pq), amp0), cmul(amp1, gp));
    double pt_p = cabs2(amp_p);
    double pt_m = cabs2(amp_m);
    // p = (1-f)*pt_p*(1-ap) + f*pt_m*(1+ap)
    double p_val = (1.0 - f) * pt_p * (1.0 - ap) + f * pt_m * (1.0 + ap);

    // Store forward pass results (only if output pointers provided)
    if (p_out != NULL) {
        p_out[e] = p_val;
        amp_p_out[e] = amp_p;
        amp_m_out[e] = amp_m;
    }

    // ---- Gradient computation (only when grad_p is provided) ----
    if (grad_p != NULL) {
    // Read stored forward pass results (may differ from computed in gradient mode if using saved values)
    // In practice, for gradient mode p_out/amp_p_out/amp_m_out are the same memory as p_arr/etc
    // Grad of p w.r.t. pt_p, pt_m
    double grad_pt_p = grad_p[e] * (1.0 - f) * (1.0 - ap);
    double grad_pt_m = grad_p[e] * f * (1.0 + ap);

    // dpt_p/dpq* = amp_p * conj(gm * amp1)
    cdouble dpt_p_dpq_star = cmul(amp_p, cconj(cmul(gm, amp1)));
    // dpt_m/dpq* = -amp_m * conj(gm * amp0) / conj(pq)^2
    cdouble tmp1 = cconj(cmul(gm, amp0));
    cdouble tmp2 = cconj(cmul(pq, pq));
    cdouble dpt_m_dpq_star = cscale(cdiv(cmul(amp_m, tmp1), tmp2), -1.0);
    // grad_pq = 2 * (grad_pt_p * dpt_p_dpq* + grad_pt_m * dpt_m_dpq*)
    cdouble grad_pq = cscale(cadd(cscale(dpt_p_dpq_star, grad_pt_p),
                                  cscale(dpt_m_dpq_star, grad_pt_m)), 2.0);

    // ---- Gradient w.r.t. ap ----
    // dp/dap = -(1-f)*pt_p + f*pt_m
    double dp_dap = -(1.0 - f) * pt_p + f * pt_m;
    atomicAdd(grad_ap_out, grad_p[e] * dp_dap);

    // --- Gradient w.r.t. N (normalization) ---
    // grad_N = -grad_p[e] * p_val / N
    // q = p/N + bkg, grad_p = w/(N*q)
    // dq_val/dN = -w * p / (N^2 * q) = -grad_p * p / N
    atomicAdd(grad_N_out, -grad_p[e] * p_val / N_val);

    // ---- Gradient w.r.t. lam ----
    // dJ/d(lam) = Re(grad_pq * exp(-i*phi))
    atomicAdd(grad_lam_out, cuCreal(cmul(grad_pq, exp_miphii)));

    // ---- Gradient w.r.t. phi ----
    cdouble tmp_phi = cmul(make_c(0.0, -1.0), cmul(grad_pq, cconj(pq)));
    atomicAdd(grad_phi_out, cuCreal(tmp_phi));

    // ---- Gradient w.r.t. amp0, amp1 ----
    cdouble dpt_p_damp0 = cmul(amp_p, cconj(gp));
    cdouble dpt_p_damp1 = cmul(amp_p, cconj(cmul(gm, pq)));
    cdouble dpt_m_damp0 = cmul(amp_m, cconj(cdiv(gm, pq)));
    cdouble dpt_m_damp1 = cmul(amp_m, cconj(gp));

    cdouble grad_amp0 = cadd(
        cscale(dpt_p_damp0, 2.0 * grad_pt_p),
        cscale(dpt_m_damp0, 2.0 * grad_pt_m)
    );
    cdouble grad_amp1 = cadd(
        cscale(dpt_p_damp1, 2.0 * grad_pt_p),
        cscale(dpt_m_damp1, 2.0 * grad_pt_m)
    );

    // ---- Gradient w.r.t. ck ----
    cdouble prefactor[16];
    for (int w = 0; w < n_waves && w < 16; w++) {
        prefactor[w] = cmul(cmul(inv_bwprod[w], make_c(bfprod[w], 0.0)), ag[w]);
    }

    for (int w = 0; w < n_waves && w < 16; w++) {
        cdouble grad_a = (w < half_waves) ? grad_amp0 : grad_amp1;
        cdouble grad_ck_val = cmul(grad_a, cconj(prefactor[w]));
        atomicAdd(&(grad_ck_re[w]), cuCreal(grad_ck_val));
        atomicAdd(&(grad_ck_im[w]), cuCimag(grad_ck_val));
    }

    // ---- Gradient w.r.t. m0 ----
    // Through BW: d(D_BW)/dm0 = 2*m0 - i*gamma
    // Through inv_bwprod: d(inv_bwprod)/d(bw_i) = -inv_bwprod / bw_i
    // Practical gradient: ∇_{bw_i} J = ∇_{inv_bwprod} J * conj(-inv_bwprod / bw_i)
    for (int w = 0; w < n_waves && w < 16; w++) {
        cdouble grad_a = (w < half_waves) ? grad_amp0 : grad_amp1;
        cdouble grad_inv = cmul(grad_a, cconj(cmul(ck[w], cmul(make_c(bfprod[w], 0.0), ag[w]))));

        for (int r = 0; r < n_res_per_wave; r++) {
            int bw_idx = bw_order[w * n_res_per_wave + r];
            cdouble bw_i = bw[w * n_res_per_wave + r];
            cdouble grad_bw_i = cmul(cscale(grad_inv, -1.0), cconj(cdiv(inv_bwprod[w], bw_i)));

            // d(bwall)/dm0 = 2*m0 + Im(gamma) - i*Re(gamma)
            cdouble dbwall_dm0 = make_c(2.0 * m0[bw_idx] + cuCimag(gamma[bw_idx]), -cuCreal(gamma[bw_idx]));
            cdouble grad_m0_val = cmul(grad_bw_i, cconj(dbwall_dm0));
            atomicAdd(&(grad_m0_out[bw_idx]), cuCreal(grad_m0_val));
        }
    }

    // ---- Gradient w.r.t. g0 ----
    // Through BW: d(bwall)/d(gamma) when gamma is complex
    for (int w = 0; w < n_waves && w < 16; w++) {
        cdouble grad_a = (w < half_waves) ? grad_amp0 : grad_amp1;
        cdouble grad_inv = cmul(grad_a, cconj(cmul(ck[w], cmul(make_c(bfprod[w], 0.0), ag[w]))));

        for (int r = 0; r < n_res_per_wave; r++) {
            int bw_idx = bw_order[w * n_res_per_wave + r];
            cdouble bw_i = bw[w * n_res_per_wave + r];
            cdouble grad_bw_i = cmul(cscale(grad_inv, -1.0), cconj(cdiv(inv_bwprod[w], bw_i)));

            // bwall = (m0² - m² + m0*Im(gamma)) + i*(-m0*Re(gamma))
            // d(bwall)/d(Re(gamma)) = -i*m0
            // d(bwall)/d(Im(gamma)) = m0
            cdouble dbwall_dRe_gamma = make_c(0.0, -m0[bw_idx]);
            cdouble dbwall_dIm_gamma = make_c(m0[bw_idx], 0.0);

            // grad_gamma = grad_bw_i * conj(d(bwall)/d(gamma))
            cdouble grad_gamma_val = cmul(grad_bw_i, cconj(dbwall_dRe_gamma));

            // gamma = einsum("mg,eg->em", matrix_gamma, g_vals)
            // gamma is complex: gamma = A + i*B where A = Re(sum) and B = Im(sum)
            // dJ/d(g_vals) = dJ/d(Re(gamma)) * matrix_gamma + dJ/d(Im(gamma)) * i*matrix_gamma
            // = matrix_gamma * (dJ/d(Re(gamma)) + i*dJ/d(Im(gamma)))
            // The practical gradient ∇_{g_vals} J = matrix_gamma * conj(∇_{gamma} J)
            // Where ∇_{gamma} J = dbwall_dRe_gamma_part + i*dbwall_dIm_gamma_part

            cdouble grad_gamma_complex = make_c(cuCreal(grad_gamma_val), cuCreal(cmul(grad_bw_i, cconj(dbwall_dIm_gamma))));

            for (int j = 0; j < n_g0 && j < 16; j++) {
                // grad_g_vals[j] += matrix_gamma[bw_idx][j] * grad_gamma_complex
                cdouble grad_g_vals_j = cscale(grad_gamma_complex, matrix_gamma[bw_idx * n_g0 + j]);
                // g_vals = g0 * gi
                // grad_g0[j] += Re(grad_g_vals_j * conj(gi[j]))
                double grad_g0_val = cuCreal(cmul(grad_g_vals_j, cconj(gi[j])));
                atomicAdd(&(grad_g0_out[j]), grad_g0_val);
            }
        }
    }

    // ---- Gradient w.r.t. delta_m, delta_g, g ----
    // d(gL)/d(delta_m) = i*t/2 * gL
    cdouble dgL_ddm = cscale(gL, 0.5 * t);
    dgL_ddm = cmul(make_c(0.0, 1.0), dgL_ddm);
    cdouble dgH_ddm = cscale(gH, -0.5 * t);
    dgH_ddm = cmul(make_c(0.0, 1.0), dgH_ddm);

    cdouble dgp_ddm = cscale(cadd(dgL_ddm, dgH_ddm), 0.5);
    cdouble dgm_ddm = cscale(csub(dgL_ddm, dgH_ddm), 0.5);

    // d(gL)/d(delta_g) = -t/2 * gL
    cdouble dgL_ddg = cscale(gL, -0.5 * t);
    cdouble dgH_ddg = cscale(gH, 0.5 * t);

    cdouble dgp_ddg = cscale(cadd(dgL_ddg, dgH_ddg), 0.5);
    cdouble dgm_ddg = cscale(csub(dgL_ddg, dgH_ddg), 0.5);

    // d(gL)/d(g) = -t/2 * gL
    cdouble dgL_dgg = cscale(gL, -0.5 * t);
    cdouble dgH_dgg = cscale(gH, -0.5 * t);

    cdouble dgp_dgg = cscale(cadd(dgL_dgg, dgH_dgg), 0.5);
    cdouble dgm_dgg = cscale(csub(dgL_dgg, dgH_dgg), 0.5);

    // d(pt_p)/d(gp*) = amp_p * conj(amp0)
    cdouble dpt_p_dgp_star = cmul(amp_p, cconj(amp0));
    cdouble dpt_p_dgm_star = cmul(amp_p, cconj(cmul(pq, amp1)));
    cdouble dpt_m_dgp_star = cmul(amp_m, cconj(amp1));
    cdouble dpt_m_dgm_star = cmul(amp_m, cconj(cdiv(amp0, pq)));

    // For delta_m
    cdouble A = cadd(cmul(dpt_p_dgp_star, cconj(dgp_ddm)), cmul(dpt_p_dgm_star, cconj(dgm_ddm)));
    cdouble B = cadd(cmul(dpt_m_dgp_star, cconj(dgp_ddm)), cmul(dpt_m_dgm_star, cconj(dgm_ddm)));
    double grad_dm = 2.0 * (grad_pt_p * cuCreal(A) + grad_pt_m * cuCreal(B));
    atomicAdd(grad_delta_m_out, grad_dm);

    // For delta_g
    A = cadd(cmul(dpt_p_dgp_star, cconj(dgp_ddg)), cmul(dpt_p_dgm_star, cconj(dgm_ddg)));
    B = cadd(cmul(dpt_m_dgp_star, cconj(dgp_ddg)), cmul(dpt_m_dgm_star, cconj(dgm_ddg)));
    double grad_dg = 2.0 * (grad_pt_p * cuCreal(A) + grad_pt_m * cuCreal(B));
    atomicAdd(grad_delta_g_out, grad_dg);

    // For g
    A = cadd(cmul(dpt_p_dgp_star, cconj(dgp_dgg)), cmul(dpt_p_dgm_star, cconj(dgm_dgg)));
    B = cadd(cmul(dpt_m_dgp_star, cconj(dgp_dgg)), cmul(dpt_m_dgm_star, cconj(dgm_dgg)));
    double grad_gg = 2.0 * (grad_pt_p * cuCreal(A) + grad_pt_m * cuCreal(B));
    atomicAdd(grad_g_out, grad_gg);
    }  // end if (grad_p != NULL)
}
// ============================================================
// Context struct for persistent GPU memory
// ============================================================
typedef struct {
    // Device data pointers
    double *d_mass_flat, *d_q_flat, *d_angles_flat;
    double *d_time_arr, *d_frac_arr;
    // Device config pointers
    int *d_bw_index, *d_gamma_index, *d_bw_order;
    int *d_bf_index, *d_bf_order, *d_ang_index;
    double *d_ang_k, *d_ang_b;
    double *d_matrix_gamma;
    cdouble *d_matrix_ang;
    cdouble *d_gamma_table;
    double *d_bf_table;
    // Sizes
    int n_events;
    int mass_stride, q_stride, ang_stride;
} PWAContext;
// ============================================================
// Host wrapper functions (called from Python via cffi)
// ============================================================
extern "C" {
void* pwa_create_context(
    // Data (host pointers)
    const double* mass_flat,
    const double* q_flat,
    const double* angles_flat,
    const double* time_arr,
    const double* frac_arr,
    // Config (host pointers)
    const int* bw_index,
    const int* gamma_index,
    const int* bw_order,
    const int* bf_index,
    const int* bf_order,
    const int* ang_index,
    const double* ang_k,
    const double* ang_b,
    const double* matrix_gamma,
    const cdouble* matrix_ang,
    const cdouble* gamma_table,
    const double* bf_table,
    // Dimensions
    int n_events, int n_waves, int n_m0, int n_g0,
    int n_res_per_wave, int n_decays_per_wave,
    int n_bf_types, int n_basis, int n_ang_per_basis,
    int n_gamma_points, int n_bf_points,
    int mass_stride, int q_stride, int ang_stride
) {
    PWAContext* ctx = (PWAContext*)malloc(sizeof(PWAContext));
    if (!ctx) return NULL;

    ctx->n_events = n_events;
    ctx->mass_stride = mass_stride;
    ctx->q_stride = q_stride;
    ctx->ang_stride = ang_stride;

    // Allocate device memory for data
    cudaMalloc(&ctx->d_mass_flat, n_events * mass_stride * sizeof(double));
    cudaMalloc(&ctx->d_q_flat, n_events * q_stride * sizeof(double));
    cudaMalloc(&ctx->d_angles_flat, n_events * ang_stride * sizeof(double));
    cudaMalloc(&ctx->d_time_arr, n_events * sizeof(double));
    cudaMalloc(&ctx->d_frac_arr, n_events * sizeof(double));

    // Copy data to device
    cudaMemcpy(ctx->d_mass_flat, mass_flat, n_events * mass_stride * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_q_flat, q_flat, n_events * q_stride * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_angles_flat, angles_flat, n_events * ang_stride * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_time_arr, time_arr, n_events * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_frac_arr, frac_arr, n_events * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate and copy config arrays
    int n_waves_times_res = n_waves * n_res_per_wave;
    int n_waves_times_dec = n_waves * n_decays_per_wave;
    int n_basis_times_ang = n_basis * n_ang_per_basis;

    cudaMalloc(&ctx->d_bw_index, n_m0 * sizeof(int));
    cudaMalloc(&ctx->d_gamma_index, n_g0 * sizeof(int));
    cudaMalloc(&ctx->d_bw_order, n_waves_times_res * sizeof(int));
    cudaMalloc(&ctx->d_bf_index, n_bf_types * sizeof(int));
    cudaMalloc(&ctx->d_bf_order, n_waves_times_dec * sizeof(int));
    cudaMalloc(&ctx->d_ang_index, n_basis_times_ang * sizeof(int));
    cudaMalloc(&ctx->d_ang_k, n_basis_times_ang * sizeof(double));
    cudaMalloc(&ctx->d_ang_b, n_basis_times_ang * sizeof(double));
    cudaMalloc(&ctx->d_matrix_gamma, n_m0 * n_g0 * sizeof(double));
    cudaMalloc(&ctx->d_matrix_ang, n_waves * n_basis * sizeof(cdouble));
    cudaMalloc(&ctx->d_gamma_table, n_g0 * n_gamma_points * sizeof(cdouble));
    cudaMalloc(&ctx->d_bf_table, n_bf_types * n_bf_points * sizeof(double));

    cudaMemcpy(ctx->d_bw_index, bw_index, n_m0 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_gamma_index, gamma_index, n_g0 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_bw_order, bw_order, n_waves_times_res * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_bf_index, bf_index, n_bf_types * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_bf_order, bf_order, n_waves_times_dec * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_ang_index, ang_index, n_basis_times_ang * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_ang_k, ang_k, n_basis_times_ang * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_ang_b, ang_b, n_basis_times_ang * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_matrix_gamma, matrix_gamma, n_m0 * n_g0 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_matrix_ang, matrix_ang, n_waves * n_basis * sizeof(cdouble), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_gamma_table, gamma_table, n_g0 * n_gamma_points * sizeof(cdouble), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_bf_table, bf_table, n_bf_types * n_bf_points * sizeof(double), cudaMemcpyHostToDevice);

    return (void*)ctx;
}
void pwa_destroy_context(void* ctx_ptr) {
    if (!ctx_ptr) return;
    PWAContext* ctx = (PWAContext*)ctx_ptr;
    cudaFree(ctx->d_mass_flat);
    cudaFree(ctx->d_q_flat);
    cudaFree(ctx->d_angles_flat);
    cudaFree(ctx->d_time_arr);
    cudaFree(ctx->d_frac_arr);
    cudaFree(ctx->d_bw_index);
    cudaFree(ctx->d_gamma_index);
    cudaFree(ctx->d_bw_order);
    cudaFree(ctx->d_bf_index);
    cudaFree(ctx->d_bf_order);
    cudaFree(ctx->d_ang_index);
    cudaFree(ctx->d_ang_k);
    cudaFree(ctx->d_ang_b);
    cudaFree(ctx->d_matrix_gamma);
    cudaFree(ctx->d_matrix_ang);
    cudaFree(ctx->d_gamma_table);
    cudaFree(ctx->d_bf_table);
    free(ctx);
}
void pwa_compute_wrapper(
    void* ctx_ptr,
    double* p_out,
    cdouble* amp_p_out,
    cdouble* amp_m_out,
    const cdouble* ck,
    const double* m0,
    const double* g0,
    double delta_m, double delta_g, double g_val,
    double ap, double lam, double phi,
    int n_waves, int n_m0, int n_g0,
    int n_res_per_wave, int n_decays_per_wave,
    int n_bf_types, int n_basis, int n_ang_per_basis,
    int n_gamma_points, int n_bf_points,
    double g_min, double g_delta, double q_min, double q_delta
) {
    PWAContext* ctx = (PWAContext*)ctx_ptr;
    int n_events = ctx->n_events;
    int threads = 256;
    int blocks = (n_events + threads - 1) / threads;
    
    // Allocate device output arrays
    double *d_p_out, *d_m0, *d_g0;
    cdouble *d_amp_p_out, *d_amp_m_out, *d_ck;
    
    cudaMalloc(&d_p_out, n_events * sizeof(double));
    cudaMalloc(&d_amp_p_out, n_events * sizeof(cdouble));
    cudaMalloc(&d_amp_m_out, n_events * sizeof(cdouble));
    cudaMalloc(&d_ck, n_waves * sizeof(cdouble));
    cudaMalloc(&d_m0, n_m0 * sizeof(double));
    cudaMalloc(&d_g0, n_g0 * sizeof(double));
    
    // Copy parameters to device
    cudaMemcpy(d_ck, ck, n_waves * sizeof(cdouble), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m0, m0, n_m0 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g0, g0, n_g0 * sizeof(double), cudaMemcpyHostToDevice);
    
    // Launch kernel (forward-only: NULL gradient inputs/outputs)
    pwa_compute_kernel<<<blocks, threads>>>(
        d_p_out, d_amp_p_out, d_amp_m_out,
        NULL,  // grad_p = NULL (forward-only)
        NULL, NULL, NULL, NULL,  // grad_ck_re, im, m0, g0
        NULL, NULL, NULL,  // grad_delta_m, delta_g, g
        NULL, NULL, NULL, NULL,  // grad_ap, lam, phi, N
        d_ck, d_m0, d_g0,
        delta_m, delta_g, g_val, ap, lam, phi, 0.0,  // N_val=0 (unused in forward)
        ctx->d_mass_flat, ctx->d_q_flat, ctx->d_angles_flat,
        ctx->d_time_arr, ctx->d_frac_arr,
        ctx->d_bw_index, ctx->d_gamma_index, ctx->d_bw_order,
        ctx->d_bf_index, ctx->d_bf_order,
        ctx->d_ang_index, ctx->d_ang_k, ctx->d_ang_b,
        ctx->d_matrix_gamma, ctx->d_matrix_ang,
        ctx->d_gamma_table, ctx->d_bf_table,
        n_events, n_waves, n_m0, n_g0,
        n_res_per_wave, n_decays_per_wave,
        n_bf_types, n_basis, n_ang_per_basis,
        n_gamma_points, n_bf_points,
        ctx->mass_stride, ctx->q_stride, ctx->ang_stride,
        g_min, g_delta, q_min, q_delta
    );
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(p_out, d_p_out, n_events * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(amp_p_out, d_amp_p_out, n_events * sizeof(cdouble), cudaMemcpyDeviceToHost);
    cudaMemcpy(amp_m_out, d_amp_m_out, n_events * sizeof(cdouble), cudaMemcpyDeviceToHost);
    
    // Free device output arrays
    cudaFree(d_p_out);
    cudaFree(d_amp_p_out);
    cudaFree(d_amp_m_out);
    cudaFree(d_ck);
    cudaFree(d_m0);
    cudaFree(d_g0);
}

void pwa_grad_wrapper(
    void* ctx_ptr,
    double* grad_p,
    double* grad_ck_re,
    double* grad_ck_im,
    double* grad_m0_out,
    double* grad_g0_out,
    double* grad_delta_m_out,
    double* grad_delta_g_out,
    double* grad_g_out,
    double* grad_ap_out,
    double* grad_lam_out,
    double* grad_phi_out,
    double* grad_N_out,
    const cdouble* ck,
    const double* m0,
    const double* g0,
    double delta_m, double delta_g, double g_val,
    double ap, double lam, double phi,
    double N_val,
    int n_waves, int n_m0, int n_g0,
    int n_res_per_wave, int n_decays_per_wave,
    int n_bf_types, int n_basis, int n_ang_per_basis,
    int n_gamma_points, int n_bf_points,
    double g_min, double g_delta, double q_min, double q_delta
) {
    PWAContext* ctx = (PWAContext*)ctx_ptr;
    int n_events = ctx->n_events;
    int threads = 256;
    int blocks = (n_events + threads - 1) / threads;
    
    // Allocate device memory (only gradient inputs/outputs, not forward outputs)
    cdouble *d_ck;
    double *d_m0, *d_g0;
    double *d_grad_p;
    double *d_grad_ck_re, *d_grad_ck_im;
    double *d_grad_m0_out, *d_grad_g0_out;
    double *d_grad_delta_m_out, *d_grad_delta_g_out, *d_grad_g_out;
    double *d_grad_ap_out, *d_grad_lam_out, *d_grad_phi_out;
    double *d_grad_N_out;
    
    cudaMalloc(&d_grad_p, n_events * sizeof(double));
    
    cudaMalloc(&d_grad_ck_re, n_waves * sizeof(double));
    cudaMalloc(&d_grad_ck_im, n_waves * sizeof(double));
    cudaMalloc(&d_grad_m0_out, n_m0 * sizeof(double));
    cudaMalloc(&d_grad_g0_out, n_g0 * sizeof(double));
    cudaMalloc(&d_grad_delta_m_out, sizeof(double));
    cudaMalloc(&d_grad_delta_g_out, sizeof(double));
    cudaMalloc(&d_grad_g_out, sizeof(double));
    cudaMalloc(&d_grad_ap_out, sizeof(double));
    cudaMalloc(&d_grad_lam_out, sizeof(double));
    cudaMalloc(&d_grad_phi_out, sizeof(double));
    cudaMalloc(&d_grad_N_out, sizeof(double));
    
    cudaMalloc(&d_ck, n_waves * sizeof(cdouble));
    cudaMalloc(&d_m0, n_m0 * sizeof(double));
    cudaMalloc(&d_g0, n_g0 * sizeof(double));
    
    // Initialize gradients to zero
    cudaMemset(d_grad_ck_re, 0, n_waves * sizeof(double));
    cudaMemset(d_grad_ck_im, 0, n_waves * sizeof(double));
    cudaMemset(d_grad_m0_out, 0, n_m0 * sizeof(double));
    cudaMemset(d_grad_g0_out, 0, n_g0 * sizeof(double));
    cudaMemset(d_grad_delta_m_out, 0, sizeof(double));
    cudaMemset(d_grad_delta_g_out, 0, sizeof(double));
    cudaMemset(d_grad_g_out, 0, sizeof(double));
    cudaMemset(d_grad_ap_out, 0, sizeof(double));
    cudaMemset(d_grad_lam_out, 0, sizeof(double));
    cudaMemset(d_grad_phi_out, 0, sizeof(double));
    cudaMemset(d_grad_N_out, 0, sizeof(double));
    
    // Copy parameters to device (no forward results needed - recomputed in kernel)
    cudaMemcpy(d_grad_p, grad_p, n_events * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ck, ck, n_waves * sizeof(cdouble), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m0, m0, n_m0 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g0, g0, n_g0 * sizeof(double), cudaMemcpyHostToDevice);
    
    // Launch kernel (gradient mode: forward outputs NULL, gradient outputs provided)
    pwa_compute_kernel<<<blocks, threads>>>(
        NULL, NULL, NULL,  // p_out, amp_p_out, amp_m_out (forward outputs not needed)
        d_grad_p,
        d_grad_ck_re, d_grad_ck_im,
        d_grad_m0_out, d_grad_g0_out,
        d_grad_delta_m_out, d_grad_delta_g_out, d_grad_g_out,
        d_grad_ap_out, d_grad_lam_out, d_grad_phi_out,
        d_grad_N_out,
        d_ck, d_m0, d_g0,
        delta_m, delta_g, g_val, ap, lam, phi,
        N_val,
        ctx->d_mass_flat, ctx->d_q_flat, ctx->d_angles_flat,
        ctx->d_time_arr, ctx->d_frac_arr,
        ctx->d_bw_index, ctx->d_gamma_index, ctx->d_bw_order,
        ctx->d_bf_index, ctx->d_bf_order,
        ctx->d_ang_index, ctx->d_ang_k, ctx->d_ang_b,
        ctx->d_matrix_gamma, ctx->d_matrix_ang,
        ctx->d_gamma_table, ctx->d_bf_table,
        n_events, n_waves, n_m0, n_g0,
        n_res_per_wave, n_decays_per_wave,
        n_bf_types, n_basis, n_ang_per_basis,
        n_gamma_points, n_bf_points,
        ctx->mass_stride, ctx->q_stride, ctx->ang_stride,
        g_min, g_delta, q_min, q_delta
    );
    cudaDeviceSynchronize();
    
    // Copy results back
    cudaMemcpy(grad_ck_re, d_grad_ck_re, n_waves * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_ck_im, d_grad_ck_im, n_waves * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_m0_out, d_grad_m0_out, n_m0 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_g0_out, d_grad_g0_out, n_g0 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_delta_m_out, d_grad_delta_m_out, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_delta_g_out, d_grad_delta_g_out, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_g_out, d_grad_g_out, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_ap_out, d_grad_ap_out, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_lam_out, d_grad_lam_out, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_phi_out, d_grad_phi_out, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_N_out, d_grad_N_out, sizeof(double), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_grad_p);
    cudaFree(d_grad_ck_re);
    cudaFree(d_grad_ck_im);
    cudaFree(d_grad_m0_out);
    cudaFree(d_grad_g0_out);
    cudaFree(d_grad_delta_m_out);
    cudaFree(d_grad_delta_g_out);
    cudaFree(d_grad_g_out);
    cudaFree(d_grad_ap_out);
    cudaFree(d_grad_lam_out);
    cudaFree(d_grad_phi_out);
    cudaFree(d_grad_N_out);
    cudaFree(d_ck);
    cudaFree(d_m0);
    cudaFree(d_g0);
}

}  // extern "C"
