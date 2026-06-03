// PWA GPU CUDA kernels
// Single file with all computation for B->phi phi partial wave analysis

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>

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
    double* __restrict__ p_out,           // (n_events,) or NULL
    cdouble* __restrict__ amp_p_out,      // (n_events,) or NULL
    cdouble* __restrict__ amp_m_out,      // (n_events,) or NULL
    double* __restrict__ q_val_out,       // scalar, sum(w * log(q)) or NULL
    // Gradient outputs (only computed when weights != NULL)
    double* __restrict__ grad_ck_re,      // (n_waves,)
    double* __restrict__ grad_ck_im,      // (n_waves,)
    double* __restrict__ grad_m0_out,     // (n_m0,)
    double* __restrict__ grad_g0_out,     // (n_g0,)
    double* __restrict__ grad_scalar_out, // [7] = {delta_m, delta_g, g, ap, lam, phi, N}
    // Parameter arrays (uploaded per compute call)
    const cdouble* __restrict__ ck,
    const double* __restrict__ m0,
    const double* __restrict__ g0,
    double delta_m, double delta_g, double g_val,
    double ap, double lam, double phi,
    double N_val,
    int do_likelihood,
    // Weights + background (non-NULL = compute gradients)
    const double* __restrict__ weights,
    const double* __restrict__ bkg_arr,
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
    double g_min, double g_delta, double q_min, double q_delta,
    // Scratch buffer for per-thread temporary arrays
    char* __restrict__ scratch_buf,
    size_t per_thread_size
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_events) return;

    // Setup per-thread scratch pointers
    // Align spos to 16 bytes before each cdouble cast to avoid misaligned access
    char* s = scratch_buf + e * per_thread_size;
    size_t spos = 0;
    int total_bw = n_waves * n_res_per_wave;
    int total_bf = n_waves * n_decays_per_wave;
    cdouble* bw = (cdouble*)(s + spos); spos += total_bw * sizeof(cdouble);
    double* bf = (double*)(s + spos); spos += total_bf * sizeof(double);
    spos = (spos + 15) & ~15;
    cdouble* ag = (cdouble*)(s + spos); spos += n_waves * sizeof(cdouble);
    cdouble* bwprod = (cdouble*)(s + spos); spos += n_waves * sizeof(cdouble);
    cdouble* a_full = (cdouble*)(s + spos); spos += n_waves * sizeof(cdouble);
    double* m_bw = (double*)(s + spos); spos += n_m0 * sizeof(double);
    double* m_gamma_vals = (double*)(s + spos); spos += n_g0 * sizeof(double);
    spos = (spos + 15) & ~15;
    cdouble* gi = (cdouble*)(s + spos); spos += n_g0 * sizeof(cdouble);
    cdouble* g_vals = (cdouble*)(s + spos); spos += n_g0 * sizeof(cdouble);
    cdouble* gamma = (cdouble*)(s + spos); spos += n_m0 * sizeof(cdouble);
    cdouble* bwall = (cdouble*)(s + spos); spos += n_m0 * sizeof(cdouble);
    double* q_bf = (double*)(s + spos); spos += n_bf_types * sizeof(double);
    double* bfall = (double*)(s + spos); spos += n_bf_types * sizeof(double);
    double* ang_f = (double*)(s + spos);

    // ===== Forward pass =====
    for (int i = 0; i < n_m0; i++) {
        m_bw[i] = mass_flat[e * mass_stride + bw_index[i]];
    }

    for (int i = 0; i < n_g0; i++) {
        m_gamma_vals[i] = mass_flat[e * mass_stride + gamma_index[i]];
    }

    for (int i = 0; i < n_g0; i++) {
        double diff = (m_gamma_vals[i] - g_min) / g_delta;
        int idx = (int)diff;
        if (idx < 0) idx = 0;
        if (idx >= n_gamma_points - 1) idx = n_gamma_points - 2;
        double delta = diff - idx;
        cdouble fl = gamma_table[i * n_gamma_points + idx];
        cdouble fr = gamma_table[i * n_gamma_points + idx + 1];
        gi[i] = cadd(fl, cscale(csub(fr, fl), delta));
        g_vals[i] = cscale(gi[i], g0[i]);
    }

    for (int i = 0; i < n_m0; i++) {
        cdouble sum = make_c(0.0, 0.0);
        for (int j = 0; j < n_g0; j++) {
            sum = cadd(sum, cscale(g_vals[j], matrix_gamma[i * n_g0 + j]));
        }
        gamma[i] = sum;
    }

    for (int i = 0; i < n_m0; i++) {
        bwall[i] = make_c(m0[i]*m0[i] - m_bw[i]*m_bw[i] + m0[i]*cuCimag(gamma[i]), -m0[i]*cuCreal(gamma[i]));
    }

    for (int i = 0; i < total_bw; i++) bw[i] = bwall[bw_order[i]];

    for (int i = 0; i < n_bf_types; i++) {
        q_bf[i] = q_flat[e * q_stride + bf_index[i]];
    }

    for (int i = 0; i < n_bf_types; i++) {
        double diff = (q_bf[i] - q_min) / q_delta;
        int idx = (int)diff;
        if (idx < 0) idx = 0;
        if (idx >= n_bf_points - 1) idx = n_bf_points - 2;
        double delta = diff - idx;
        double fl = bf_table[i * n_bf_points + idx];
        double fr = bf_table[i * n_bf_points + idx + 1];
        bfall[i] = fl + (fr - fl) * delta;
    }

    for (int i = 0; i < total_bf; i++) bf[i] = bfall[bf_order[i]];

    for (int b = 0; b < n_basis; b++) {
        double prod = 1.0;
        for (int j = 0; j < n_ang_per_basis; j++) {
            int idx = ang_index[b * n_ang_per_basis + j];
            double theta = angles_flat[e * ang_stride + idx];
            double val = theta * ang_k[b * n_ang_per_basis + j] + ang_b[b * n_ang_per_basis + j];
            prod *= cos(val);
        }
        ang_f[b] = prod;
    }

    for (int w = 0; w < n_waves; w++) {
        cdouble sum = make_c(0.0, 0.0);
        for (int b = 0; b < n_basis; b++) {
            sum = cadd(sum, cmul(matrix_ang[w * n_basis + b], make_c(ang_f[b], 0.0)));
        }
        ag[w] = sum;
    }

    for (int w = 0; w < n_waves; w++) {
        cdouble prod = make_c(1.0, 0.0);
        for (int r = 0; r < n_res_per_wave; r++) prod = cmul(prod, bw[w * n_res_per_wave + r]);
        bwprod[w] = prod;
    }

    for (int w = 0; w < n_waves; w++) {
        double bfprod_w = 1.0;
        for (int d = 0; d < n_decays_per_wave; d++) bfprod_w *= bf[w * n_decays_per_wave + d];
        cdouble inv_bwprod_w = cdiv(make_c(1.0, 0.0), bwprod[w]);
        a_full[w] = cmul(cscale(cmul(ck[w], inv_bwprod_w), bfprod_w), ag[w]);
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

    // Accumulate q_val on GPU (avoids CPU reduction, critical for large datasets)
    // q_val = sum_e w[e] * (do_likelihood ? log(p/N + bkg) : p)
    bool do_grad = (weights != NULL);
    if (q_val_out != NULL && do_grad) {
        double contrib;
        if (do_likelihood) {
            double bkg_val = bkg_arr ? bkg_arr[e] : 0.0;
            double q_val = p_val / N_val + bkg_val;
            contrib = weights[e] * log(q_val);
        } else {
            contrib = weights[e] * p_val;
        }
        atomicAdd(q_val_out, contrib);
    }

    // Compute grad_p on the fly from weights + p_val (avoids CPU round-trip)
    double grad_p_val;
    if (do_grad) {
        if (do_likelihood) {
            // Likelihood mode: q = p/N + bkg, grad_p = w/(N*q)
            double bkg_val = bkg_arr ? bkg_arr[e] : 0.0;
            double q_val = p_val / N_val + bkg_val;
            grad_p_val = weights[e] / (N_val * q_val);
        } else {
            // Chi-square mode: q = p, grad_p = w
            grad_p_val = weights[e];
        }
    }

    // ---- Gradient computation (only when weights are provided) ----
    if (do_grad) {
    // Grad of p w.r.t. pt_p, pt_m
    double grad_pt_p = grad_p_val * (1.0 - f) * (1.0 - ap);
    double grad_pt_m = grad_p_val * f * (1.0 + ap);

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
    atomicAdd(&grad_scalar_out[3], grad_p_val * dp_dap);

    // --- Gradient w.r.t. N (normalization, only in likelihood mode) ---
    if (do_likelihood) {
        // grad_N = -grad_p_val * p_val / N
        // q = p/N + bkg, grad_p = w/(N*q)
        // dq_val/dN = -w * p / (N^2 * q) = -grad_p * p / N
        atomicAdd(&grad_scalar_out[6], -grad_p_val * p_val / N_val);
    }

    // ---- Gradient w.r.t. lam ----
    // dJ/d(lam) = Re(grad_pq * exp(-i*phi))
    atomicAdd(&grad_scalar_out[4], cuCreal(cmul(grad_pq, exp_miphii)));

    // ---- Gradient w.r.t. phi ----
    cdouble tmp_phi = cmul(make_c(0.0, -1.0), cmul(grad_pq, cconj(pq)));
    atomicAdd(&grad_scalar_out[5], cuCreal(tmp_phi));

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
    for (int w = 0; w < n_waves; w++) {
        double bfprod_w = 1.0;
        for (int d = 0; d < n_decays_per_wave; d++) bfprod_w *= bf[w * n_decays_per_wave + d];
        cdouble pref = cmul(cmul(cdiv(make_c(1.0, 0.0), bwprod[w]), make_c(bfprod_w, 0.0)), ag[w]);
        cdouble grad_a = (w < half_waves) ? grad_amp0 : grad_amp1;
        cdouble grad_ck_val = cmul(grad_a, cconj(pref));
        double r = cuCreal(grad_ck_val);
        double i = cuCimag(grad_ck_val);
        for (int off = 16; off > 0; off >>= 1) {
            r += __shfl_down_sync(0xffffffff, r, off);
            i += __shfl_down_sync(0xffffffff, i, off);
        }
        if ((threadIdx.x & 31) == 0) {
            atomicAdd(&(grad_ck_re[w]), r);
            atomicAdd(&(grad_ck_im[w]), i);
        }
    }

    // ---- Gradient w.r.t. m0 ----
    // Through BW: d(D_BW)/dm0 = 2*m0 - i*gamma
    // Through inv_bwprod: d(inv_bwprod)/d(bw_i) = -inv_bwprod / bw_i
    // Practical gradient: ∇_{bw_i} J = ∇_{inv_bwprod} J * conj(-inv_bwprod / bw_i)
    for (int w = 0; w < n_waves; w++) {
        cdouble grad_a = (w < half_waves) ? grad_amp0 : grad_amp1;
        double bfprod_w = 1.0;
        for (int d = 0; d < n_decays_per_wave; d++) bfprod_w *= bf[w * n_decays_per_wave + d];
        cdouble grad_inv = cmul(grad_a, cconj(cmul(ck[w], cmul(make_c(bfprod_w, 0.0), ag[w]))));
        cdouble inv_bwprod_w = cdiv(make_c(1.0, 0.0), bwprod[w]);

        for (int r = 0; r < n_res_per_wave; r++) {
            int bw_idx = bw_order[w * n_res_per_wave + r];
            cdouble bw_i = bw[w * n_res_per_wave + r];
            cdouble grad_bw_i = cmul(cscale(grad_inv, -1.0), cconj(cdiv(inv_bwprod_w, bw_i)));

            // d(bwall)/dm0 = 2*m0 + Im(gamma) - i*Re(gamma)
            cdouble dbwall_dm0 = make_c(2.0 * m0[bw_idx] + cuCimag(gamma[bw_idx]), -cuCreal(gamma[bw_idx]));
            cdouble grad_m0_val = cmul(grad_bw_i, cconj(dbwall_dm0));
            atomicAdd(&(grad_m0_out[bw_idx]), cuCreal(grad_m0_val));
        }
    }

    // ---- Gradient w.r.t. g0 ----
    // Through BW: d(bwall)/d(gamma) when gamma is complex
    for (int w = 0; w < n_waves; w++) {
        cdouble grad_a = (w < half_waves) ? grad_amp0 : grad_amp1;
        double bfprod_w = 1.0;
        for (int d = 0; d < n_decays_per_wave; d++) bfprod_w *= bf[w * n_decays_per_wave + d];
        cdouble grad_inv = cmul(grad_a, cconj(cmul(ck[w], cmul(make_c(bfprod_w, 0.0), ag[w]))));
        cdouble inv_bwprod_w = cdiv(make_c(1.0, 0.0), bwprod[w]);

        for (int r = 0; r < n_res_per_wave; r++) {
            int bw_idx = bw_order[w * n_res_per_wave + r];
            cdouble bw_i = bw[w * n_res_per_wave + r];
            cdouble grad_bw_i = cmul(cscale(grad_inv, -1.0), cconj(cdiv(inv_bwprod_w, bw_i)));

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

            for (int j = 0; j < n_g0; j++) {
                // grad_g_vals[j] += matrix_gamma[bw_idx][j] * grad_gamma_complex
                cdouble grad_g_vals_j = cscale(grad_gamma_complex, matrix_gamma[bw_idx * n_g0 + j]);
                // g_vals = g0 * gi
                // grad_g0[j] += Re(grad_g_vals_j * conj(gi[j]))
                double grad_g0_val = cuCreal(cmul(grad_g_vals_j, cconj(gi[j])));
                for (int off = 16; off > 0; off >>= 1)
                    grad_g0_val += __shfl_down_sync(0xffffffff, grad_g0_val, off);
                if ((threadIdx.x & 31) == 0)
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
    atomicAdd(&grad_scalar_out[0], grad_dm);

    // For delta_g
    A = cadd(cmul(dpt_p_dgp_star, cconj(dgp_ddg)), cmul(dpt_p_dgm_star, cconj(dgm_ddg)));
    B = cadd(cmul(dpt_m_dgp_star, cconj(dgp_ddg)), cmul(dpt_m_dgm_star, cconj(dgm_ddg)));
    double grad_dg = 2.0 * (grad_pt_p * cuCreal(A) + grad_pt_m * cuCreal(B));
    atomicAdd(&grad_scalar_out[1], grad_dg);

    // For g
    A = cadd(cmul(dpt_p_dgp_star, cconj(dgp_dgg)), cmul(dpt_p_dgm_star, cconj(dgm_dgg)));
    B = cadd(cmul(dpt_m_dgp_star, cconj(dgp_dgg)), cmul(dpt_m_dgm_star, cconj(dgm_dgg)));
    double grad_gg = 2.0 * (grad_pt_p * cuCreal(A) + grad_pt_m * cuCreal(B));
    atomicAdd(&grad_scalar_out[2], grad_gg);
    }  // end if (grad_p != NULL)
}
// ============================================================
// Context struct for persistent GPU memory
// ============================================================
// ============================================================
// Data and Config structs (split so config is reused across datasets)
// ============================================================
typedef struct {
    // Device data pointers (per-event)
    double *d_mass_flat, *d_q_flat, *d_angles_flat;
    double *d_time_arr, *d_frac_arr;
    // Sizes
    int n_events;
    int mass_stride, q_stride, ang_stride;
} PWAData;

typedef struct {
    // Device config pointers (shared across datasets)
    int *d_bw_index, *d_gamma_index, *d_bw_order;
    int *d_bf_index, *d_bf_order, *d_ang_index;
    double *d_ang_k, *d_ang_b;
    double *d_matrix_gamma;
    cdouble *d_matrix_ang;
    cdouble *d_gamma_table;
    double *d_bf_table;
    // Pre-allocated scratch buffer for per-thread temp arrays
    char* d_scratch;
    size_t per_thread_size;
    int max_batch_size;
} PWAConfig;
// ============================================================
// Host wrapper functions (called from Python via cffi)
// ============================================================
extern "C" {

void* pwa_create_config(
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
    int n_waves, int n_m0, int n_g0,
    int n_res_per_wave, int n_decays_per_wave,
    int n_bf_types, int n_basis, int n_ang_per_basis,
    int n_gamma_points, int n_bf_points
) {
    PWAConfig* cfg = (PWAConfig*)calloc(1, sizeof(PWAConfig));
    if (!cfg) return NULL;

    // Allocate and copy config arrays
    int n_waves_times_res = n_waves * n_res_per_wave;
    int n_waves_times_dec = n_waves * n_decays_per_wave;
    int n_basis_times_ang = n_basis * n_ang_per_basis;

    cudaMalloc(&cfg->d_bw_index, n_m0 * sizeof(int));
    cudaMalloc(&cfg->d_gamma_index, n_g0 * sizeof(int));
    cudaMalloc(&cfg->d_bw_order, n_waves_times_res * sizeof(int));
    cudaMalloc(&cfg->d_bf_index, n_bf_types * sizeof(int));
    cudaMalloc(&cfg->d_bf_order, n_waves_times_dec * sizeof(int));
    cudaMalloc(&cfg->d_ang_index, n_basis_times_ang * sizeof(int));
    cudaMalloc(&cfg->d_ang_k, n_basis_times_ang * sizeof(double));
    cudaMalloc(&cfg->d_ang_b, n_basis_times_ang * sizeof(double));
    cudaMalloc(&cfg->d_matrix_gamma, n_m0 * n_g0 * sizeof(double));
    cudaMalloc(&cfg->d_matrix_ang, n_waves * n_basis * sizeof(cdouble));
    cudaMalloc(&cfg->d_gamma_table, n_g0 * n_gamma_points * sizeof(cdouble));
    cudaMalloc(&cfg->d_bf_table, n_bf_types * n_bf_points * sizeof(double));

    cudaMemcpy(cfg->d_bw_index, bw_index, n_m0 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cfg->d_gamma_index, gamma_index, n_g0 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cfg->d_bw_order, bw_order, n_waves_times_res * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cfg->d_bf_index, bf_index, n_bf_types * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cfg->d_bf_order, bf_order, n_waves_times_dec * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cfg->d_ang_index, ang_index, n_basis_times_ang * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cfg->d_ang_k, ang_k, n_basis_times_ang * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cfg->d_ang_b, ang_b, n_basis_times_ang * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cfg->d_matrix_gamma, matrix_gamma, n_m0 * n_g0 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cfg->d_matrix_ang, matrix_ang, n_waves * n_basis * sizeof(cdouble), cudaMemcpyHostToDevice);
    cudaMemcpy(cfg->d_gamma_table, gamma_table, n_g0 * n_gamma_points * sizeof(cdouble), cudaMemcpyHostToDevice);
    cudaMemcpy(cfg->d_bf_table, bf_table, n_bf_types * n_bf_points * sizeof(double), cudaMemcpyHostToDevice);

    // Pre-allocate scratch buffer for per-thread temp arrays.
    // Must match the kernel's layout including 16-byte alignment pads.
    int total_bw = n_waves * n_res_per_wave;
    int total_bf = n_waves * n_decays_per_wave;
    // Starting estimate (non-interleaved, no padding)
    // Removed bfprod (n_waves*double) and inv_bwprod (n_waves*cdouble)
    // Current layout: bw(cd), bf(d), pad, ag(cd), bwprod(cd), a_full(cd), m_bw(d),
    //   m_gamma_vals(d), pad, gi(cd), g_vals(cd), gamma(cd), bwall(cd), q_bf(d), bfall(d), ang_f(d)
    cfg->per_thread_size = total_bw * (size_t)sizeof(cdouble) + total_bf * (size_t)sizeof(double)
         + n_waves * 3 * (size_t)sizeof(cdouble)
         + n_m0 * 2 * (size_t)sizeof(cdouble) + n_m0 * (size_t)sizeof(double)
         + n_g0 * 2 * (size_t)sizeof(cdouble) + n_g0 * (size_t)sizeof(double)
         + n_bf_types * 2 * (size_t)sizeof(double) + n_basis * (size_t)sizeof(double);
    // Add alignment overhead: two 16-byte alignment pads before cdouble sections
    // (max 15 bytes each) + final round to 16.
    cfg->per_thread_size += 48;
    cfg->per_thread_size = (cfg->per_thread_size + 15) & ~15ULL;
    cfg->max_batch_size = 8192;
    cudaMalloc(&cfg->d_scratch, cfg->max_batch_size * cfg->per_thread_size);

    return (void*)cfg;
}

void* pwa_create_data(
    // Data (host pointers)
    const double* mass_flat,
    const double* q_flat,
    const double* angles_flat,
    const double* time_arr,
    const double* frac_arr,
    // Dimensions
    int n_events, int mass_stride, int q_stride, int ang_stride
) {
    PWAData* data = (PWAData*)calloc(1, sizeof(PWAData));
    if (!data) return NULL;

    data->n_events = n_events;
    data->mass_stride = mass_stride;
    data->q_stride = q_stride;
    data->ang_stride = ang_stride;

    // Allocate device memory for data
    cudaMalloc(&data->d_mass_flat, n_events * mass_stride * sizeof(double));
    cudaMalloc(&data->d_q_flat, n_events * q_stride * sizeof(double));
    cudaMalloc(&data->d_angles_flat, n_events * ang_stride * sizeof(double));
    cudaMalloc(&data->d_time_arr, n_events * sizeof(double));
    cudaMalloc(&data->d_frac_arr, n_events * sizeof(double));

    // Copy data to device
    cudaMemcpy(data->d_mass_flat, mass_flat, n_events * mass_stride * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(data->d_q_flat, q_flat, n_events * q_stride * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(data->d_angles_flat, angles_flat, n_events * ang_stride * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(data->d_time_arr, time_arr, n_events * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(data->d_frac_arr, frac_arr, n_events * sizeof(double), cudaMemcpyHostToDevice);

    return (void*)data;
}
void pwa_destroy_config(void* cfg_ptr) {
    if (!cfg_ptr) return;
    PWAConfig* cfg = (PWAConfig*)cfg_ptr;
    cudaFree(cfg->d_bw_index);
    cudaFree(cfg->d_gamma_index);
    cudaFree(cfg->d_bw_order);
    cudaFree(cfg->d_bf_index);
    cudaFree(cfg->d_bf_order);
    cudaFree(cfg->d_ang_index);
    cudaFree(cfg->d_ang_k);
    cudaFree(cfg->d_ang_b);
    cudaFree(cfg->d_matrix_gamma);
    cudaFree(cfg->d_matrix_ang);
    cudaFree(cfg->d_gamma_table);
    cudaFree(cfg->d_bf_table);
    cudaFree(cfg->d_scratch);
    free(cfg);
}
void pwa_destroy_data(void* data_ptr) {
    if (!data_ptr) return;
    PWAData* data = (PWAData*)data_ptr;
    cudaFree(data->d_mass_flat);
    cudaFree(data->d_q_flat);
    cudaFree(data->d_angles_flat);
    cudaFree(data->d_time_arr);
    cudaFree(data->d_frac_arr);
    free(data);
}
// Combined forward + gradient: one kernel launch, one set of transfers
void pwa_compute(
    void* cfg_ptr,
    void* data_ptr,
    double* p_out, cdouble* amp_p_out, cdouble* amp_m_out,
    double* q_val_out,  // scalar: sum(w*log(q)) — NULL to skip
    double* grad_ck_re, double* grad_ck_im,
    double* grad_m0_out, double* grad_g0_out,
    double* grad_scalar_out,  // [7] = {delta_m, delta_g, g, ap, lam, phi, N}
    const cdouble* ck, const double* m0, const double* g0,
    double delta_m, double delta_g, double g_val,
    double ap, double lam, double phi,
    double N_val, int do_likelihood,
    const double* weights, const double* bkg_arr,
    int n_waves, int n_m0, int n_g0,
    int n_res_per_wave, int n_decays_per_wave,
    int n_bf_types, int n_basis, int n_ang_per_basis,
    int n_gamma_points, int n_bf_points,
    double g_min, double g_delta, double q_min, double q_delta
) {
    PWAConfig* cfg = (PWAConfig*)cfg_ptr;
    PWAData* data = (PWAData*)data_ptr;
    int n_events = data->n_events;
    int threads = 256;
    
    // Allocate all device buffers (init NULL so cleanup labels are safe)
    double *d_p_out = NULL, *d_m0 = NULL, *d_g0 = NULL, *d_weights = NULL, *d_bkg_arr = NULL;
    cdouble *d_amp_p_out = NULL, *d_amp_m_out = NULL, *d_ck = NULL;
    double *d_q_val_out = NULL;
    double *d_grad_ck_re = NULL, *d_grad_ck_im = NULL;
    double *d_grad_m0_out = NULL, *d_grad_g0_out = NULL;
    double *d_grad_scalar_out = NULL;
    cudaError_t cuerr;
    
    // Allocate all temporary buffers; goto cleanup on any failure
    if (cudaMalloc(&d_p_out, n_events * sizeof(double)) != cudaSuccess) goto cleanup_ck;
    if (cudaMalloc(&d_amp_p_out, n_events * sizeof(cdouble)) != cudaSuccess) goto cleanup_p;
    if (cudaMalloc(&d_amp_m_out, n_events * sizeof(cdouble)) != cudaSuccess) goto cleanup_amp_p;
    if (cudaMalloc(&d_q_val_out, sizeof(double)) != cudaSuccess) goto cleanup_amp_m;
    if (cudaMalloc(&d_ck, n_waves * sizeof(cdouble)) != cudaSuccess) goto cleanup_q_val;
    if (cudaMalloc(&d_m0, n_m0 * sizeof(double)) != cudaSuccess) goto cleanup_ck_buf;
    if (cudaMalloc(&d_g0, n_g0 * sizeof(double)) != cudaSuccess) goto cleanup_m0;
    if (cudaMalloc(&d_weights, n_events * sizeof(double)) != cudaSuccess) goto cleanup_g0;
    if (cudaMalloc(&d_bkg_arr, n_events * sizeof(double)) != cudaSuccess) goto cleanup_weights;
    if (cudaMalloc(&d_grad_ck_re, n_waves * sizeof(double)) != cudaSuccess) goto cleanup_bkg;
    if (cudaMalloc(&d_grad_ck_im, n_waves * sizeof(double)) != cudaSuccess) goto cleanup_grad_re;
    if (cudaMalloc(&d_grad_m0_out, n_m0 * sizeof(double)) != cudaSuccess) goto cleanup_grad_im;
    if (cudaMalloc(&d_grad_g0_out, n_g0 * sizeof(double)) != cudaSuccess) goto cleanup_grad_m0;
    if (cudaMalloc(&d_grad_scalar_out, 7 * sizeof(double)) != cudaSuccess) goto cleanup_grad_g0;
    
    // Upload params + weights once
    cudaMemcpy(d_ck, ck, n_waves * sizeof(cdouble), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m0, m0, n_m0 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g0, g0, n_g0 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, n_events * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bkg_arr, bkg_arr, n_events * sizeof(double), cudaMemcpyHostToDevice);
    
    // Zero accumulation buffers
    cudaMemset(d_q_val_out, 0, sizeof(double));
    cudaMemset(d_grad_ck_re, 0, n_waves * sizeof(double));
    cudaMemset(d_grad_ck_im, 0, n_waves * sizeof(double));
    cudaMemset(d_grad_m0_out, 0, n_m0 * sizeof(double));
    cudaMemset(d_grad_g0_out, 0, n_g0 * sizeof(double));
    cudaMemset(d_grad_scalar_out, 0, 7 * sizeof(double));
    
    // Single pass: batches with forward + gradient in one kernel launch
    for (int start = 0; start < n_events; start += cfg->max_batch_size) {
        int batch_size = cfg->max_batch_size;
        if (start + batch_size > n_events) batch_size = n_events - start;
        int blocks = (batch_size + threads - 1) / threads;
        
        pwa_compute_kernel<<<blocks, threads>>>(
            d_p_out + start, d_amp_p_out + start, d_amp_m_out + start,
            d_q_val_out,
            d_grad_ck_re, d_grad_ck_im,
            d_grad_m0_out, d_grad_g0_out,
            d_grad_scalar_out,
            d_ck, d_m0, d_g0,
            delta_m, delta_g, g_val, ap, lam, phi, N_val, do_likelihood,
            d_weights + start, d_bkg_arr + start,
            data->d_mass_flat + start * (size_t)data->mass_stride,
            data->d_q_flat + start * (size_t)data->q_stride,
            data->d_angles_flat + start * (size_t)data->ang_stride,
            data->d_time_arr + start,
            data->d_frac_arr + start,
            cfg->d_bw_index, cfg->d_gamma_index, cfg->d_bw_order,
            cfg->d_bf_index, cfg->d_bf_order,
            cfg->d_ang_index, cfg->d_ang_k, cfg->d_ang_b,
            cfg->d_matrix_gamma, cfg->d_matrix_ang,
            cfg->d_gamma_table, cfg->d_bf_table,
            batch_size, n_waves, n_m0, n_g0,
            n_res_per_wave, n_decays_per_wave,
            n_bf_types, n_basis, n_ang_per_basis,
            n_gamma_points, n_bf_points,
            data->mass_stride, data->q_stride, data->ang_stride,
            g_min, g_delta, q_min, q_delta,
            cfg->d_scratch, cfg->per_thread_size
        );
    }
    cudaDeviceSynchronize();
    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "CUDA error after pwa_compute_kernel: %s\n", cudaGetErrorString(cuerr));
        goto cleanup_grad_g0;
    }
    
    // Copy all results back (single sync point)
    if (p_out) cudaMemcpy(p_out, d_p_out, n_events * sizeof(double), cudaMemcpyDeviceToHost);
    if (amp_p_out) cudaMemcpy(amp_p_out, d_amp_p_out, n_events * sizeof(cdouble), cudaMemcpyDeviceToHost);
    if (amp_m_out) cudaMemcpy(amp_m_out, d_amp_m_out, n_events * sizeof(cdouble), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_ck_re, d_grad_ck_re, n_waves * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_ck_im, d_grad_ck_im, n_waves * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_m0_out, d_grad_m0_out, n_m0 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_g0_out, d_grad_g0_out, n_g0 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_scalar_out, d_grad_scalar_out, 7 * sizeof(double), cudaMemcpyDeviceToHost);
    if (q_val_out) cudaMemcpy(q_val_out, d_q_val_out, sizeof(double), cudaMemcpyDeviceToHost);
    
    // Free all device memory (normal path)
    cudaFree(d_ck); cudaFree(d_m0); cudaFree(d_g0);
    cudaFree(d_weights); cudaFree(d_bkg_arr);
    cudaFree(d_grad_ck_re); cudaFree(d_grad_ck_im);
    cudaFree(d_grad_m0_out); cudaFree(d_grad_g0_out);
    cudaFree(d_grad_scalar_out);
    cudaFree(d_q_val_out);
    cudaFree(d_p_out); cudaFree(d_amp_p_out); cudaFree(d_amp_m_out);
    return;
    
    // Error cleanup: free what was allocated, unwind in reverse
cleanup_grad_g0:  cudaFree(d_grad_g0_out);
cleanup_grad_m0:  cudaFree(d_grad_m0_out);
cleanup_grad_im:  cudaFree(d_grad_ck_im);
cleanup_grad_re:  cudaFree(d_grad_ck_re);
cleanup_bkg:      cudaFree(d_bkg_arr);
cleanup_weights:  cudaFree(d_weights);
cleanup_g0:       cudaFree(d_g0);
cleanup_m0:       cudaFree(d_m0);
cleanup_ck_buf:   cudaFree(d_ck);
cleanup_q_val:    cudaFree(d_q_val_out);
cleanup_amp_m:    cudaFree(d_amp_m_out);
cleanup_amp_p:    cudaFree(d_amp_p_out);
cleanup_p:        cudaFree(d_p_out);
cleanup_ck:       return;
}

}  // extern "C"

// ================================================================
// PROFILING ENTRY POINT (compiled with: nvcc -DPROFILE_MAIN ...)
// ================================================================
#ifdef PROFILE_MAIN

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    return (double)ms;
}

static double rand_uniform(double lo, double hi) {
    return lo + (hi - lo) * ((double)rand() / (double)RAND_MAX);
}

static void print_sec(const char* title) {
    printf("\n%s\n", title);
    printf("------------------------------------------------------------\n");
}

typedef struct {
    int n_events;
    int n_waves;
    int n_m0;
    int n_g0;
    int n_res_per_wave;
    int n_decays_per_wave;
    int n_bf_types;
    int n_basis;
    int n_ang_per_basis;
    int n_gamma_points;
    int n_bf_points;
    int mass_stride, q_stride, ang_stride;
    double g_min, g_delta, q_min, q_delta;
} ProfCfg;

static void gen_data(ProfCfg* cfg,
    double** mass_flat, double** q_flat, double** angles_flat,
    double** time_arr, double** frac_arr,
    int** bw_index, int** gamma_index, int** bw_order,
    int** bf_index, int** bf_order, int** ang_index,
    double** ang_k, double** ang_b,
    double** matrix_gamma, cdouble** matrix_ang,
    cdouble** gamma_table, double** bf_table,
    cdouble** ck, double** m0_ptr, double** g0_ptr)
{
    int ev = cfg->n_events, w = cfg->n_waves, m0n = cfg->n_m0, g0n = cfg->n_g0;
    int nr = cfg->n_res_per_wave, nd = cfg->n_decays_per_wave;
    int nbf = cfg->n_bf_types, nb = cfg->n_basis, nap = cfg->n_ang_per_basis;
    int ngp = cfg->n_gamma_points, nbp = cfg->n_bf_points;
    int ms = cfg->mass_stride, qs = cfg->q_stride, as = cfg->ang_stride;
    size_t sz;
#define ALLOC(ptr, cnt, typ) do { sz = (cnt) * sizeof(typ); (ptr) = (typ*)malloc(sz); } while(0)
    ALLOC(*mass_flat,    ev*ms, double);
    ALLOC(*q_flat,       ev*qs, double);
    ALLOC(*angles_flat,  ev*as, double);
    ALLOC(*time_arr,     ev, double);
    ALLOC(*frac_arr,     ev, double);
    ALLOC(*bw_index,     m0n, int);
    ALLOC(*gamma_index,  g0n, int);
    ALLOC(*bw_order,     w*nr, int);
    ALLOC(*bf_index,     nbf, int);
    ALLOC(*bf_order,     w*nd, int);
    ALLOC(*ang_index,    nb*nap, int);
    ALLOC(*ang_k,        nb*nap, double);
    ALLOC(*ang_b,        nb*nap, double);
    ALLOC(*matrix_gamma, m0n*g0n, double);
    ALLOC(*matrix_ang,   w*nb, cdouble);
    ALLOC(*gamma_table,  g0n*ngp, cdouble);
    ALLOC(*bf_table,     nbf*nbp, double);
    ALLOC(*ck,           w, cdouble);
    ALLOC(*m0_ptr,       m0n, double);
    ALLOC(*g0_ptr,       g0n, double);
#undef ALLOC
    srand(42);
    for (int i = 0; i < ev*ms; i++) (*mass_flat)[i] = rand_uniform(0.6,1.0);
    for (int i = 0; i < ev*qs; i++) (*q_flat)[i] = rand_uniform(0.1,0.4);
    for (int i = 0; i < ev*as; i++) (*angles_flat)[i] = rand_uniform(0.0,3.14159);
    for (int i = 0; i < ev; i++)    (*time_arr)[i] = rand_uniform(0.0,5.0);
    for (int i = 0; i < ev; i++)    (*frac_arr)[i] = rand_uniform(-0.2,0.2);
    for (int i = 0; i < m0n; i++)   (*bw_index)[i] = rand() % ms;
    for (int i = 0; i < g0n; i++)   (*gamma_index)[i] = rand() % ms;
    for (int i = 0; i < w*nr; i++)  (*bw_order)[i] = rand() % m0n;
    for (int i = 0; i < nbf; i++)   (*bf_index)[i] = rand() % qs;
    for (int i = 0; i < w*nd; i++)  (*bf_order)[i] = rand() % nbf;
    for (int i = 0; i < nb*nap; i++) {
        (*ang_index)[i] = rand() % as;
        (*ang_k)[i] = rand_uniform(-2,2);
        (*ang_b)[i] = rand_uniform(-2,2);
    }
    for (int i = 0; i < m0n*g0n; i++) (*matrix_gamma)[i] = rand_uniform(-1,1);
    for (int i = 0; i < w*nb; i++) (*matrix_ang)[i] = make_cuDoubleComplex(rand_uniform(-1,1),rand_uniform(-1,1));
    for (int i = 0; i < g0n*ngp; i++) (*gamma_table)[i] = make_cuDoubleComplex(rand_uniform(0,0.5),0);
    for (int i = 0; i < nbf*nbp; i++) (*bf_table)[i] = rand_uniform(0,1);
    for (int i = 0; i < w; i++) (*ck)[i] = make_cuDoubleComplex(rand_uniform(-1,1),rand_uniform(-1,1));
    for (int i = 0; i < m0n; i++) (*m0_ptr)[i] = rand_uniform(1.2,1.5);
    for (int i = 0; i < g0n; i++) (*g0_ptr)[i] = rand_uniform(0.05,0.15);
}

static void free_data(void** arrs, int n) {
    for (int i = 0; i < n; i++) if (arrs[i]) free(arrs[i]);
}

int main() {
    printf("============================================================\n");
    printf("  PWA GPU Kernel Profiler\n");
    printf("============================================================\n");

    ProfCfg cfg;
    cfg.n_events          = 5000;
    cfg.n_waves           = 64;
    cfg.n_m0              = 256;
    cfg.n_g0              = 256;
    cfg.n_res_per_wave    = 2;
    cfg.n_decays_per_wave = 3;
    cfg.n_bf_types        = 6;
    cfg.n_basis           = 10;
    cfg.n_ang_per_basis   = 3;
    cfg.n_gamma_points    = 200;
    cfg.n_bf_points       = 150;
    cfg.mass_stride       = 12;
    cfg.q_stride          = 16;
    cfg.ang_stride        = 15;
    cfg.g_min = 0; cfg.g_delta = 0.005;
    cfg.q_min = 0; cfg.q_delta = 0.005;

    int n_runs = 10, warmup = 3;
    printf("\nConfig: events=%d waves=%d m0=%d g0=%d\n",
           cfg.n_events, cfg.n_waves, cfg.n_m0, cfg.n_g0);

    double *mf, *qf, *af, *tarr, *farr;
    int *bwi, *gi, *bwo, *bfi, *bfo, *ai;
    double *ak, *ab, *mg;
    cdouble *ma, *gt;
    double *bft;
    cdouble *ck;
    double *m0h, *g0h;
    gen_data(&cfg, &mf, &qf, &af, &tarr, &farr,
             &bwi, &gi, &bwo, &bfi, &bfo, &ai,
             &ak, &ab, &mg, &ma, &gt, &bft, &ck, &m0h, &g0h);

    double delta_m=0.5065, delta_g=0.001, g_val=0.657;
    double ap=0.01, lam=0.75, phi=0.15, N_val=50000;

    void* cfg_p = pwa_create_config(
        bwi, gi, bwo, bfi, bfo, ai,
        ak, ab, mg, ma, gt, bft,
        cfg.n_waves, cfg.n_m0, cfg.n_g0,
        cfg.n_res_per_wave, cfg.n_decays_per_wave,
        cfg.n_bf_types, cfg.n_basis, cfg.n_ang_per_basis,
        cfg.n_gamma_points, cfg.n_bf_points);
    if (!cfg_p) { fprintf(stderr,"Config failed\n"); return 1; }
    void* data_p = pwa_create_data(
        mf, qf, af, tarr, farr,
        cfg.n_events, cfg.mass_stride, cfg.q_stride, cfg.ang_stride);
    if (!data_p) { pwa_destroy_config(cfg_p); return 1; }
    PWAConfig* pcfg = (PWAConfig*)cfg_p;
    PWAData* pdata = (PWAData*)data_p;
    printf("Config: batch=%d, scratch=%zu bytes\n",
           pcfg->max_batch_size, pcfg->per_thread_size);

    double* p_out    = (double*)malloc(cfg.n_events*sizeof(double));
    cdouble* ap_out  = (cdouble*)malloc(cfg.n_events*sizeof(cdouble));
    cdouble* am_out  = (cdouble*)malloc(cfg.n_events*sizeof(cdouble));
    double* gpr      = (double*)malloc((size_t)cfg.n_events*sizeof(double));
    for (int i = 0; i < cfg.n_events; i++) gpr[i] = 1.0/cfg.n_events;

    double *gcr = (double*)calloc(cfg.n_waves,sizeof(double));
    double *gci = (double*)calloc(cfg.n_waves,sizeof(double));
    double *gm0 = (double*)calloc(cfg.n_m0,sizeof(double));
    double *gg0 = (double*)calloc(cfg.n_g0,sizeof(double));
    double *gsc = (double*)calloc(7,sizeof(double));  // [delta_m, delta_g, g, ap, lam, phi, N]
    double *bkg_arr_h = (double*)calloc(cfg.n_events, sizeof(double));  // zero background

    cudaEvent_t es, ee;
    cudaEventCreate(&es); cudaEventCreate(&ee);

    /* ===== [1] Warmup ===== */
    print_sec("[1] Warmup");
    for (int i = 0; i < warmup; i++) {
        pwa_compute(cfg_p, data_p, p_out, ap_out, am_out, NULL,
            gcr, gci, gm0, gg0, gsc,
            ck, m0h, g0h, delta_m, delta_g, g_val, ap, lam, phi, N_val, 1,
            gpr, bkg_arr_h,
            cfg.n_waves, cfg.n_m0, cfg.n_g0,
            cfg.n_res_per_wave, cfg.n_decays_per_wave,
            cfg.n_bf_types, cfg.n_basis, cfg.n_ang_per_basis,
            cfg.n_gamma_points, cfg.n_bf_points,
            cfg.g_min, cfg.g_delta, cfg.q_min, cfg.q_delta);
    }
    printf("Warmup done\n");

    /* ===== [2] Combined e2e (forward+gradient) ===== */
    print_sec("[2] Combined End-to-End (pwa_compute)");
    double fwd_e2e = 0;
    for (int r = 0; r < n_runs; r++) {
        cudaEventRecord(es);
        pwa_compute(cfg_p, data_p, p_out, ap_out, am_out, NULL,
            gcr, gci, gm0, gg0, gsc,
            ck, m0h, g0h, delta_m, delta_g, g_val, ap, lam, phi, N_val, 1,
            gpr, bkg_arr_h,
            cfg.n_waves, cfg.n_m0, cfg.n_g0,
            cfg.n_res_per_wave, cfg.n_decays_per_wave,
            cfg.n_bf_types, cfg.n_basis, cfg.n_ang_per_basis,
            cfg.n_gamma_points, cfg.n_bf_points,
            cfg.g_min, cfg.g_delta, cfg.q_min, cfg.q_delta);
        cudaEventRecord(ee);
        double t = elapsed_ms(es, ee);
        fwd_e2e += t;
        printf("  Run %2d: %8.3f ms\n", r+1, t);
    }
    fwd_e2e /= n_runs;
    printf("  Avg: %8.3f ms\n", fwd_e2e);

    /* ===== [3] Combined e2e (identical to [2]) ===== */
    double grad_e2e = fwd_e2e;

    /* ===== [4] Kernel-only timing ===== */
    print_sec("[4] Kernel-Only Timing");

    int ev = cfg.n_events, w = cfg.n_waves, maxb = pcfg->max_batch_size;
    cdouble *d_ck;   cudaMalloc(&d_ck, w*sizeof(cdouble));
    double *d_m0;    cudaMalloc(&d_m0, cfg.n_m0*sizeof(double));
    double *d_g0;    cudaMalloc(&d_g0, cfg.n_g0*sizeof(double));
    double *d_po;    cudaMalloc(&d_po, ev*sizeof(double));
    cdouble *d_apo;  cudaMalloc(&d_apo, ev*sizeof(cdouble));
    cdouble *d_amo;  cudaMalloc(&d_amo, ev*sizeof(cdouble));
    cudaMemcpy(d_ck, ck, w*sizeof(cdouble), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m0, m0h, cfg.n_m0*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g0, g0h, cfg.n_g0*sizeof(double), cudaMemcpyHostToDevice);

    /* Forward kernel only */
    double fwd_ker = 0;
    for (int r = 0; r < n_runs; r++) {
        cudaEventRecord(es);
        for (int s = 0; s < ev; s += maxb) {
            int bs = (s+maxb > ev) ? ev-s : maxb;
            int bl = (bs+255)/256;
            pwa_compute_kernel<<<bl,256>>>(
                d_po+s, d_apo+s, d_amo+s, NULL,
                NULL, NULL, NULL, NULL, NULL,
                d_ck, d_m0, d_g0,
                delta_m, delta_g, g_val, ap, lam, phi, 0.0, 0,
                NULL, NULL,  // no weights, no bkg (forward-only)
                pdata->d_mass_flat + s*(size_t)pdata->mass_stride,
                pdata->d_q_flat + s*(size_t)pdata->q_stride,
                pdata->d_angles_flat + s*(size_t)pdata->ang_stride,
                pdata->d_time_arr + s, pdata->d_frac_arr + s,
                pcfg->d_bw_index, pcfg->d_gamma_index, pcfg->d_bw_order,
                pcfg->d_bf_index, pcfg->d_bf_order,
                pcfg->d_ang_index, pcfg->d_ang_k, pcfg->d_ang_b,
                pcfg->d_matrix_gamma, pcfg->d_matrix_ang,
                pcfg->d_gamma_table, pcfg->d_bf_table,
                bs, w, cfg.n_m0, cfg.n_g0,
                cfg.n_res_per_wave, cfg.n_decays_per_wave,
                cfg.n_bf_types, cfg.n_basis, cfg.n_ang_per_basis,
                cfg.n_gamma_points, cfg.n_bf_points,
                pdata->mass_stride, pdata->q_stride, pdata->ang_stride,
                cfg.g_min, cfg.g_delta, cfg.q_min, cfg.q_delta,
                pcfg->d_scratch, pcfg->per_thread_size);
        }
        cudaDeviceSynchronize();
        cudaEventRecord(ee);
        double t = elapsed_ms(es, ee);
        fwd_ker += t;
        printf("  Fwd kernel %2d: %8.3f ms\n", r+1, t);
    }
    fwd_ker /= n_runs;
    printf("  Fwd kernel avg: %8.3f ms\n", fwd_ker);

    /* Gradient kernel only */
    double *d_weights; cudaMalloc(&d_weights, ev*sizeof(double));
    cudaMemcpy(d_weights, gpr, ev*sizeof(double), cudaMemcpyHostToDevice);
    double *d_gcr, *d_gci, *d_gm0, *d_gg0, *d_grad_scalar;
    cudaMalloc(&d_gcr, w*sizeof(double)); cudaMalloc(&d_gci, w*sizeof(double));
    cudaMalloc(&d_gm0, cfg.n_m0*sizeof(double)); cudaMalloc(&d_gg0, cfg.n_g0*sizeof(double));
    cudaMalloc(&d_grad_scalar, 7*sizeof(double));

    double grad_ker = 0;
    for (int r = 0; r < n_runs; r++) {
        cudaMemset(d_gcr, 0, w*sizeof(double));
        cudaMemset(d_gci, 0, w*sizeof(double));
        cudaMemset(d_gm0, 0, cfg.n_m0*sizeof(double));
        cudaMemset(d_gg0, 0, cfg.n_g0*sizeof(double));
        cudaMemset(d_grad_scalar, 0, 7*sizeof(double));

        cudaEventRecord(es);
        for (int s = 0; s < ev; s += maxb) {
            int bs = (s+maxb > ev) ? ev-s : maxb;
            int bl = (bs+255)/256;
            pwa_compute_kernel<<<bl,256>>>(
                NULL, NULL, NULL, NULL,
                d_gcr, d_gci, d_gm0, d_gg0,
                d_grad_scalar,
                d_ck, d_m0, d_g0,
                delta_m, delta_g, g_val, ap, lam, phi, N_val, 1,
                d_weights, NULL,
                pdata->d_mass_flat + s*(size_t)pdata->mass_stride,
                pdata->d_q_flat + s*(size_t)pdata->q_stride,
                pdata->d_angles_flat + s*(size_t)pdata->ang_stride,
                pdata->d_time_arr + s, pdata->d_frac_arr + s,
                pcfg->d_bw_index, pcfg->d_gamma_index, pcfg->d_bw_order,
                pcfg->d_bf_index, pcfg->d_bf_order,
                pcfg->d_ang_index, pcfg->d_ang_k, pcfg->d_ang_b,
                pcfg->d_matrix_gamma, pcfg->d_matrix_ang,
                pcfg->d_gamma_table, pcfg->d_bf_table,
                bs, w, cfg.n_m0, cfg.n_g0,
                cfg.n_res_per_wave, cfg.n_decays_per_wave,
                cfg.n_bf_types, cfg.n_basis, cfg.n_ang_per_basis,
                cfg.n_gamma_points, cfg.n_bf_points,
                pdata->mass_stride, pdata->q_stride, pdata->ang_stride,
                cfg.g_min, cfg.g_delta, cfg.q_min, cfg.q_delta,
                pcfg->d_scratch, pcfg->per_thread_size);
        }
        cudaDeviceSynchronize();
        cudaEventRecord(ee);
        double t = elapsed_ms(es, ee);
        grad_ker += t;
        printf("  Grad kernel %2d: %8.3f ms\n", r+1, t);
    }
    grad_ker /= n_runs;
    printf("  Grad kernel avg: %8.3f ms\n", grad_ker);

    cudaFree(d_weights); cudaFree(d_gcr); cudaFree(d_gci);
    cudaFree(d_gm0); cudaFree(d_gg0);
    cudaFree(d_grad_scalar);

    /* ===== [5] Memory overhead ===== */
    print_sec("[5] Memory Management Overhead");
    double malloc_t=0, free_t=0;
    for (int r = 0; r < n_runs; r++) {
        double *a; cdouble *b, *c;
        cudaEventRecord(es);
        cudaMalloc(&a, ev*sizeof(double));
        cudaMalloc(&b, ev*sizeof(cdouble));
        cudaMalloc(&c, w*sizeof(cdouble));
        cudaEventRecord(ee);
        malloc_t += elapsed_ms(es,ee);
        cudaEventRecord(es);
        cudaFree(a); cudaFree(b); cudaFree(c);
        cudaEventRecord(ee);
        free_t += elapsed_ms(es,ee);
    }
    double grad_mt=0, grad_mst=0, grad_ft=0;
    for (int r = 0; r < n_runs; r++) {
        double *d1,*d2,*d3,*d4,*d5,*d6,*d7,*d8,*d9,*d10,*d11,*d12,*d13;
        cudaEventRecord(es);
        cudaMalloc(&d1,ev*sizeof(double)); cudaMalloc(&d2,w*sizeof(double));
        cudaMalloc(&d3,w*sizeof(double)); cudaMalloc(&d4,cfg.n_m0*sizeof(double));
        cudaMalloc(&d5,cfg.n_g0*sizeof(double)); cudaMalloc(&d6,sizeof(double));
        cudaMalloc(&d7,sizeof(double)); cudaMalloc(&d8,sizeof(double));
        cudaMalloc(&d9,sizeof(double)); cudaMalloc(&d10,sizeof(double));
        cudaMalloc(&d11,sizeof(double)); cudaMalloc(&d12,sizeof(double));
        cudaMalloc(&d13,sizeof(double));
        cudaEventRecord(ee);
        grad_mt += elapsed_ms(es,ee);

        cudaEventRecord(es);
        cudaMemset(d2,0,w*sizeof(double)); cudaMemset(d3,0,w*sizeof(double));
        cudaMemset(d4,0,cfg.n_m0*sizeof(double)); cudaMemset(d5,0,cfg.n_g0*sizeof(double));
        cudaMemset(d6,0,sizeof(double)); cudaMemset(d7,0,sizeof(double));
        cudaMemset(d8,0,sizeof(double)); cudaMemset(d9,0,sizeof(double));
        cudaMemset(d10,0,sizeof(double)); cudaMemset(d11,0,sizeof(double));
        cudaMemset(d12,0,sizeof(double));
        cudaEventRecord(ee);
        grad_mst += elapsed_ms(es,ee);

        cudaEventRecord(es);
        cudaFree(d1); cudaFree(d2); cudaFree(d3); cudaFree(d4); cudaFree(d5);
        cudaFree(d6); cudaFree(d7); cudaFree(d8); cudaFree(d9); cudaFree(d10);
        cudaFree(d11); cudaFree(d12); cudaFree(d13);
        cudaEventRecord(ee);
        grad_ft += elapsed_ms(es,ee);
    }
    printf("  Forward overhead (3 allocs+3 frees): %.3f ms (%.1f%% of e2e)\n",
           (malloc_t+free_t)/n_runs, 100*(malloc_t+free_t)/n_runs/fwd_e2e);
    printf("  Gradient overhead (13+11+13):        %.3f ms (%.1f%% of e2e)\n",
           (grad_mt+grad_mst+grad_ft)/n_runs, 100*(grad_mt+grad_mst+grad_ft)/n_runs/grad_e2e);

    /* ===== [6] Memory transfer ===== */
    print_sec("[6] Memory Transfer Bandwidth");
    size_t pbytes = (size_t)w*sizeof(cdouble)+(size_t)cfg.n_m0*sizeof(double)+(size_t)cfg.n_g0*sizeof(double);
    size_t dbytes = (size_t)ev*(size_t)(pdata->mass_stride+pdata->q_stride+pdata->ang_stride+2)*sizeof(double);
    size_t rbytes = (size_t)ev*(sizeof(double)+2*sizeof(cdouble));
    double h2dp=0, h2dd=0, d2hr=0;
    for (int r = 0; r < 10; r++) {
        cudaEventRecord(es);
        cudaMemcpy(d_ck, ck, w*sizeof(cdouble), cudaMemcpyHostToDevice);
        cudaMemcpy(d_m0, m0h, cfg.n_m0*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_g0, g0h, cfg.n_g0*sizeof(double), cudaMemcpyHostToDevice);
        cudaEventRecord(ee);
        h2dp += elapsed_ms(es,ee);
    }
    double *dh = (double*)malloc(dbytes);
    double *dd; cudaMalloc(&dd, dbytes);
    for (int r = 0; r < 10; r++) {
        cudaEventRecord(es);
        cudaMemcpy(dd, dh, dbytes, cudaMemcpyHostToDevice);
        cudaEventRecord(ee);
        h2dd += elapsed_ms(es,ee);
    }
    double *dr = (double*)malloc(rbytes);
    for (int r = 0; r < 10; r++) {
        cudaEventRecord(es);
        cudaMemcpy(dr, dd, rbytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(ee);
        d2hr += elapsed_ms(es,ee);
    }
    cudaFree(dd); free(dh); free(dr);

    printf("  H2D params (%7zu B):  %.3f ms  %5.1f GB/s\n", pbytes, h2dp/10,
           (pbytes/1e9)/(h2dp/10/1000));
    printf("  H2D data   (%7zu B):  %.3f ms  %5.1f GB/s\n", dbytes, h2dd/10,
           (dbytes/1e9)/(h2dd/10/1000));
    printf("  D2H results(%7zu B):  %.3f ms  %5.1f GB/s\n", rbytes, d2hr/10,
           (rbytes/1e9)/(d2hr/10/1000));

    /* ===== [7] Summary ===== */
    print_sec("[7] Bottleneck Summary");
    printf("  E2E Forward:  %.3f ms\n", fwd_e2e);
    printf("  E2E Gradient: %.3f ms  (%.2fx fwd)\n", grad_e2e, grad_e2e/fwd_e2e);
    printf("  Fwd kernel:   %.3f ms  (%.1f%% of e2e)\n", fwd_ker, 100*fwd_ker/fwd_e2e);
    printf("  Grad kernel:  %.3f ms  (%.1f%% of e2e)\n", grad_ker, 100*grad_ker/grad_e2e);
    printf("  Overhead:     %.3f ms  (%.1f%% of fwd e2e)\n",
           fwd_e2e-fwd_ker, 100*(fwd_e2e-fwd_ker)/fwd_e2e);
    double alloc_free = (malloc_t+free_t)/n_runs;
    double h2d_d2h = h2dp/10 + d2hr/10;
    printf("  Alloc+Free:   %.3f ms  (%.1f%% of fwd e2e)\n", alloc_free, 100*alloc_free/fwd_e2e);
    printf("  H2D+D2H:      %.3f ms  (%.1f%% of fwd e2e)\n", h2d_d2h, 100*h2d_d2h/fwd_e2e);
    printf("\n  Bottleneck: ");
    if (fwd_ker > 0.5*fwd_e2e) printf("KERNEL COMPUTE DOMINANT (%.0f%%)\n", 100*fwd_ker/fwd_e2e);
    else if (alloc_free > 0.2*fwd_e2e) printf("MEMORY ALLOC/FREE SIGNIFICANT (%.0f%%)\n", 100*alloc_free/fwd_e2e);
    else if (h2d_d2h > 0.2*fwd_e2e) printf("TRANSFER SIGNIFICANT (%.0f%%)\n", 100*h2d_d2h/fwd_e2e);
    else printf("Balanced (kernel=%.0f%%, alloc=%.0f%%, transfer=%.0f%%)\n",
           100*fwd_ker/fwd_e2e, 100*alloc_free/fwd_e2e, 100*h2d_d2h/fwd_e2e);
    printf("  Scratch buf:  %.1f MB\n", (double)(pcfg->max_batch_size*pcfg->per_thread_size)/1e6);

    /* Cleanup */
    cudaFree(d_po); cudaFree(d_apo); cudaFree(d_amo);
    cudaFree(d_ck); cudaFree(d_m0); cudaFree(d_g0);
    cudaEventDestroy(es); cudaEventDestroy(ee);
    pwa_destroy_data(data_p);
    pwa_destroy_config(cfg_p);
    free(p_out); free(ap_out); free(am_out); free(gpr);
    free(gcr); free(gci); free(gm0); free(gg0); free(gsc);
    void* all[] = {mf, qf, af, tarr, farr, bwi, gi, bwo, bfi, bfo, ai,
                   ak, ab, mg, ma, gt, bft, ck, m0h, g0h};
    free_data(all, 19);
    printf("\nDone.\n");
    return 0;
}
#endif  // PROFILE_MAIN
