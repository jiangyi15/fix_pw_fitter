/**
 * CPU kernel v3 — Catmull-Rom interpolation backend in pure C + OpenMP.
 *
 * API mirrors the CUDA v3_kernels.cu design:
 *   cpu_create_context_v3() — store config
 *   cpu_load_data_v3()      — wrap data arrays
 *   cpu_compute_v3()        — forward + backward pass (OpenMP parallel)
 *   cpu_free_context_v3()
 *
 * Parallelisation: OpenMP over events (no intra-event CUDA-style tiling).
 *
 * Sparse scatter/gather for matrix_gamma (same as v3_sparse CUDA).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <complex.h>

/* ── Named constants ─────────────────────────────────────────── */
#define N_SCALAR  6

/* ── Catmull-Rom interpolation ───────────────────────────────── */
static inline double catmull_rom_1d(
    double pm1, double p0, double p1, double p2, double t)
{
    return p0 + 0.5 * t * (
        -pm1 + p1
        + t * (2.0*pm1 - 5.0*p0 + 4.0*p1 - p2
               + t * (-pm1 + 3.0*p0 - 3.0*p1 + p2))
    );
}

static inline double complex interp_complex(
    const double* table_real, const double* table_imag,
    int type_idx, double x,
    double xmin, double inv_delta, int n_bins)
{
    double diff = (x - xmin) * inv_delta;
    int xbin = (int)floor(diff);
    if (xbin < 0) xbin = 0;
    if (xbin > n_bins - 2) xbin = n_bins - 2;
    double t = diff - xbin;
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;

    int base = type_idx * n_bins + xbin;
    int end_ = (type_idx + 1) * n_bins - 1;
    int im1 = (base > type_idx * n_bins) ? base - 1 : base;
    int i2  = (base + 2 <= end_) ? base + 2 : base + 1;

    double real_val = catmull_rom_1d(
        table_real[im1], table_real[base],
        table_real[base + 1], table_real[i2], t);
    double imag_val = catmull_rom_1d(
        table_imag[im1], table_imag[base],
        table_imag[base + 1], table_imag[i2], t);
    return real_val + I * imag_val;
}

static inline double interp_real(
    const double* table, int type_idx, double x,
    double xmin, double inv_delta, int n_bins)
{
    double diff = (x - xmin) * inv_delta;
    int xbin = (int)floor(diff);
    if (xbin < 0) xbin = 0;
    if (xbin > n_bins - 2) xbin = n_bins - 2;
    double t = diff - xbin;
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;

    int base = type_idx * n_bins + xbin;
    int end_ = (type_idx + 1) * n_bins - 1;
    int im1 = (base > type_idx * n_bins) ? base - 1 : base;
    int i2  = (base + 2 <= end_) ? base + 2 : base + 1;

    return catmull_rom_1d(
        table[im1], table[base], table[base + 1], table[i2], t);
}

/* ── Structures ──────────────────────────────────────────────── */

typedef struct {
    /* Index arrays (pointers to Python CFFI buffers) */
    const int* m0_index;
    const int* g0_index;
    const int* g0_mass_index;
    const int* mass_index;
    const int* fl_type;
    const int* fl_q_index;
    const int* bw_order;
    const int* fl_order;
    const int* angle_index;
    /* Constant arrays */
    const double* angle_k;
    const double* angle_b;
    const double* matrix_angle_real;
    const double* matrix_angle_imag;
    const double* gamma_table_real;
    const double* gamma_table_imag;
    double gamma_min;
    double gamma_inv_delta;
    int gamma_table_bins;
    const double* matrix_gamma;
    const double* fl_table;
    double fl_min;
    double fl_inv_delta;
    int fl_table_bins;
    /* Sparse matrix_gamma metadata (gamma_col_idx[gamma_row] = unique_bw_col) */
    const int* gamma_col_idx;
    int n_gamma_rows_sparse;
    /* Dimensions */
    int n_wave;
    int n_res;
    int n_decay;
    int n_unique_bw;
    int n_gamma_rows;
    int n_mass;
    int n_momentum;
    int n_angle_k;
    int n_angle_total;
    int n_angle_comp;
    int n_m0_params;
    int n_g0_params;
} CPUContext;

typedef struct {
    const double* mass;
    const double* momentum;
    const double* angle;
    const double* frac;
    const double* time;
    const double* weight;
    const double* bkg;
    int n_events;
    int n_mass;
    int n_momentum;
    int n_angle_total;
    int n_angle_comp;
} CPUData;

/* ── Per-event computation: forward + backward ──────────────── */

static void compute_event(
    const CPUContext* ctx,
    const double* mass,     /* flat: [n_mass] */
    const double* momentum, /* flat: [n_momentum] */
    const double* angle,    /* flat: [n_angle_total * n_angle_comp] */
    double frac_val,
    double time_val,
    double weight_val,
    double bkg_val,
    const double* ck_real, const double* ck_imag,
    const double* m0, const double* g0,
    double Gamma, double Delta_Gamma, double Delta_m,
    double A_p, double poq_rho, double pop_phi,
    int use_norm, double norm,
    double* out_Q,
    double* out_P,
    double* out_gck_r,  /* [n_wave] */
    double* out_gck_i,
    double* out_gm0,    /* [n_unique_bw] */
    double* out_gg0,    /* [n_gamma_rows] */
    double* out_gsc     /* [N_SCALAR] */
) {
    int n_wave = ctx->n_wave;
    int n_res = ctx->n_res;
    int n_decay = ctx->n_decay;
    int n_unique_bw = ctx->n_unique_bw;
    int n_gamma_rows = ctx->n_gamma_rows;
    int n_wave_half = n_wave / 2;

    /* VLA arrays for dynamic sizes */
    double g_interp_real[n_gamma_rows];
    double g_interp_imag[n_gamma_rows];
    double g_bw_real[n_unique_bw];
    double g_bw_imag[n_unique_bw];

    /* Zero g_bw accumulators */
    for (int j = 0; j < n_unique_bw; j++) {
        g_bw_real[j] = 0.0;
        g_bw_imag[j] = 0.0;
    }

    for (int gi = 0; gi < n_gamma_rows; gi++) {
        int g0_idx = ctx->g0_index[gi];
        double g0_val = g0[g0_idx];
        double mass_val = mass[ctx->g0_mass_index[gi]];

        double complex g_interp = interp_complex(
            ctx->gamma_table_real, ctx->gamma_table_imag,
            g0_idx, mass_val,
            ctx->gamma_min, ctx->gamma_inv_delta, ctx->gamma_table_bins);

        g_interp_real[gi] = creal(g_interp);
        g_interp_imag[gi] = cimag(g_interp);

        double complex g_val = g0_val * g_interp;
        /* Sparse scatter: matrix_gamma has one non-zero per row */
        int col = ctx->gamma_col_idx[gi];
        g_bw_real[col] += creal(g_val);
        g_bw_imag[col] += cimag(g_val);
    }

    /* ── Phase 2: ka_prod, bw_p, fa, FL, common_amp for each wave ── */
    double ka_prod[ctx->n_angle_k];

    for (int k = 0; k < ctx->n_angle_k; k++) {
        int angle_pos = ctx->angle_index[k];
        double ka = 1.0;
        for (int c = 0; c < ctx->n_angle_comp; c++) {
            int idx = angle_pos * ctx->n_angle_comp + c;
            double ak = ctx->angle_k[k * ctx->n_angle_comp + c];
            double ab = ctx->angle_b[k * ctx->n_angle_comp + c];
            ka *= cos(angle[idx] * ak + ab);
        }
        ka_prod[k] = ka;
    }

    double common_amp_real[n_wave];
    double common_amp_imag[n_wave];
    double bw_p_real[n_wave];
    double bw_p_imag[n_wave];
    double bw_dom_real[n_unique_bw];
    double bw_dom_imag[n_unique_bw];

    for (int w = 0; w < n_wave; w++) {
        /* bw_p = product of bw_dom over resonances */
        double complex bw_p = 1.0 + 0.0 * I;
        for (int r = 0; r < n_res; r++) {
            int bw_idx = ctx->bw_order[w * n_res + r];
            double m0_val = m0[ctx->m0_index[bw_idx]];
            double mass_val = mass[ctx->mass_index[bw_idx]];
            double m0_sq = m0_val * m0_val;
            double mass_sq = mass_val * mass_val;

            double complex g_bw_val = g_bw_real[bw_idx] + I * g_bw_imag[bw_idx];
            /* bw_dom = m0² - m² + m0*Im(g_bw)  -  i*m0*Re(g_bw)
               = m0² - m² - i*m0*(Re(g_bw) + i*Im(g_bw))
               = m0² - m² - i*m0*g_bw  ✓ */
            double complex bw_dom = (m0_sq - mass_sq + m0_val * g_bw_imag[bw_idx])
                                    + I * (-m0_val * g_bw_real[bw_idx]);
            bw_p *= bw_dom;

            bw_dom_real[bw_idx] = creal(bw_dom);
            bw_dom_imag[bw_idx] = cimag(bw_dom);
        }
        bw_p_real[w] = creal(bw_p);
        bw_p_imag[w] = cimag(bw_p);

        /* fa = dot(ka_prod, matrix_angle_row) */
        double complex fa = 0.0 + 0.0 * I;
        for (int k = 0; k < ctx->n_angle_k; k++) {
            int idx = k * n_wave + w;
            fa += ka_prod[k] * (ctx->matrix_angle_real[idx] + I * ctx->matrix_angle_imag[idx]);
        }

        /* FL factor (real interpolation) */
        double fl_p = 1.0;
        for (int d = 0; d < n_decay; d++) {
            int fl_idx = ctx->fl_order[w * n_decay + d];
            double fl_q = momentum[ctx->fl_q_index[fl_idx]];
            fl_p *= interp_real(ctx->fl_table, ctx->fl_type[fl_idx],
                                fl_q, ctx->fl_min, ctx->fl_inv_delta, ctx->fl_table_bins);
        }

        /* common_amp = (1/bw_p) * fa * fl_p */
        double complex common_amp = fa * fl_p / bw_p;
        common_amp_real[w] = creal(common_amp);
        common_amp_imag[w] = cimag(common_amp);
    }

    /* ── Phase 3: ap, am (sum of ck * common_amp over waves) ── */
    double complex ap = 0.0 + 0.0 * I;
    double complex am = 0.0 + 0.0 * I;
    for (int i = 0; i < n_wave_half; i++) {
        double complex ck_i = ck_real[i] + I * ck_imag[i];
        double complex common_i = common_amp_real[i] + I * common_amp_imag[i];
        ap += ck_i * common_i;

        double complex ck_j = ck_real[n_wave_half + i] + I * ck_imag[n_wave_half + i];
        double complex common_j = common_amp_real[n_wave_half + i]
                                  + I * common_amp_imag[n_wave_half + i];
        am += ck_j * common_j;
    }

    /* ── Phase 4: Time evolution and probability ── */
    double complex i_const = I;
    double complex eL = cexp(-i_const * time_val *
        (-Delta_m / 2.0 + I * (-(Gamma + Delta_Gamma / 2.0) / 2.0)));
    double complex eH = cexp(-i_const * time_val *
        (Delta_m / 2.0  + I * (-(Gamma - Delta_Gamma / 2.0) / 2.0)));
    double complex gp = (eL + eH) / 2.0;
    double complex gm = (eL - eH) / 2.0;

    double complex poq = poq_rho * cexp(I * pop_phi);

    double complex pap = gp * ap + gm * poq * am;
    double complex pam_val = (gm / poq) * ap + gp * am;

    double pb = creal(pap * conj(pap));   /* |pap|² */
    double pbbar = creal(pam_val * conj(pam_val)); /* |pam|² */
    double P = frac_val * pb * (1.0 - A_p)
               + (1.0 - frac_val) * pbbar * (1.0 + A_p);
    *out_P = P;

    double Q;
    double dQ_dP_val;
    if (use_norm == 0) {
        Q = weight_val * P;
        dQ_dP_val = weight_val;
    } else {
        Q = -weight_val * log(P / norm + bkg_val);
        dQ_dP_val = -weight_val / (P + bkg_val * norm);
    }
    *out_Q = Q;

    /* ==================== BACKWARD PASS ==================== */

    /* Probability gradients */
    double dP_dpb = frac_val * (1.0 - A_p);
    double dP_dpbbar = (1.0 - frac_val) * (1.0 + A_p);
    double dP_dAp = -frac_val * pb + (1.0 - frac_val) * pbbar;

    double dQ_dpb = dQ_dP_val * dP_dpb;
    double dQ_dpbbar = dQ_dP_val * dP_dpbbar;

    /* Wirtinger gradients */
    double complex d_pb_dap = conj(pap) * gp;
    double complex d_pb_dam = conj(pap) * gm * poq;
    double complex d_pbbar_dap = conj(pam_val) * (gm / poq);
    double complex d_pbbar_dam = conj(pam_val) * gp;

    double complex dQ_dap_val = dQ_dpb * d_pb_dap + dQ_dpbbar * d_pbbar_dap;
    double complex dQ_dam_val = dQ_dpb * d_pb_dam + dQ_dpbbar * d_pbbar_dam;

    /* ── ck gradient ── */
    for (int i = 0; i < n_wave_half; i++) {
        double complex common_i = common_amp_real[i] + I * common_amp_imag[i];
        double complex gck = dQ_dap_val * common_i;
        out_gck_r[i] = creal(gck);
        out_gck_i[i] = cimag(gck);

        double complex common_j = common_amp_real[n_wave_half + i]
                                  + I * common_amp_imag[n_wave_half + i];
        double complex gck2 = dQ_dam_val * common_j;
        out_gck_r[n_wave_half + i] = creal(gck2);
        out_gck_i[n_wave_half + i] = cimag(gck2);
    }

    /* ── Backprop through bw_p (and hence m0, g0) ── */
    /* dQ/dbw_p = dQ_da * (-ck / bw_p² * fa * fl)
                = dQ_da * (-ck * one_over_bw * common_amp / bw_p)  (no, simpler:)
       a = ck * (1/bw_p) * fa * fl = ck * common_amp
       Wait: common_amp = (1/bw_p) * fa * fl
       a = ck * common_amp
       ∂a/∂(1/bw_p) = ck * fa * fl
       But we need ∂Q/∂bw_p not ∂Q/∂(1/bw_p).
       
       a = ck * fa * fl / bw_p
       ∂a/∂bw_p = -ck * fa * fl / bw_p² = -ck * common_amp / bw_p = -ck * (1/bw_p) * common_amp
       
       Actually: common_amp = fa * fl / bw_p
       Let's define one_over_bw = 1/bw_p
       Then a = ck * one_over_bw * fa * fl
             = ck * common_amp
       ∂a/∂bw_p = ck * fa * fl * (-1/bw_p²) = -ck * common_amp / bw_p
       
       And dQ_dbw_p = dQ_da * (-ck * common_amp / bw_p) 
    */

    /* Pre-compute dQ_dbw_dom for each unique_bw */
    double dQ_dbw_dom_real[n_unique_bw];
    double dQ_dbw_dom_imag[n_unique_bw];
    for (int j = 0; j < n_unique_bw; j++) {
        dQ_dbw_dom_real[j] = 0.0;
        dQ_dbw_dom_imag[j] = 0.0;
    }

    for (int w = 0; w < n_wave; w++) {
        double complex bw_p = bw_p_real[w] + I * bw_p_imag[w];
        double complex one_over_bw = 1.0 / bw_p;

        /* dQ_da for this wave */
        double complex dQ_da;
        if (w < n_wave_half) {
            dQ_da = dQ_dap_val;
        } else {
            dQ_da = dQ_dam_val;
        }

        /* a_wave = ck * one_over_bw * fa * fl */
        /* But we already have common_amp... */
        double complex ck_w = (w < n_wave_half)
            ? (ck_real[w] + I * ck_imag[w])
            : (ck_real[w] + I * ck_imag[w]);

        /* dQ_dbw_p = dQ_da * (-ck * common_amp / bw_p) 
           = -dQ_da * ck * one_over_bw * common_amp / bw_p? No...
           
           common_amp = one_over_bw * fa * fl
           a_wave = ck * common_amp = ck * one_over_bw * fa * fl
           ∂a/∂bw_p = -ck * one_over_bw² * fa * fl = -ck * one_over_bw * common_amp / bw_p?
           
           Actually:
           Let f = fa * fl (doesn't depend on bw_p)
           common_amp = f / bw_p
           a = ck * f / bw_p
           ∂a/∂bw_p = -ck * f / bw_p² = -ck * common_amp / bw_p
           
           And dQ_dbw_p = dQ_da * ∂a/∂bw_p = dQ_da * (-ck * common_amp / bw_p) */
        
        double complex common_amp = common_amp_real[w] + I * common_amp_imag[w];
        double complex dQ_dbw_p = -dQ_da * ck_w * common_amp / bw_p; /* complex scalar */

        /* Product backprop: bw_p = bw_dom[bw_0] * bw_dom[bw_1] * ... * bw_dom[bw_{n_res-1}]
           ∂Q/∂bw_dom[bw_idx] = dQ_dbw_p * ∏_{r≠idx} bw_dom[bw_order[w*n_res+r]]
           
           Let's compute it:
           bw_p = prod_r bw_dom[bw_order[w*n_res+r]]
           ∂Q/∂bw_dom[k] = dQ_dbw_p * ∏_{r: bw_order[w*n_res+r] ≠ k} bw_dom[bw_order[w*n_res+r]] 
                          = dQ_dbw_p * bw_p / bw_dom[k]
        */
        for (int r = 0; r < n_res; r++) {
            int bw_idx = ctx->bw_order[w * n_res + r];
            double complex bw_dom = bw_dom_real[bw_idx] + I * bw_dom_imag[bw_idx];
            /* ∂bw_p/∂bw_dom[bw_idx] = bw_p / bw_dom[bw_idx] */
            double complex dbw_p_dbw_dom = bw_p / bw_dom;
            double complex dQ_dbw_dom_val = dQ_dbw_p * dbw_p_dbw_dom;
            dQ_dbw_dom_real[bw_idx] += creal(dQ_dbw_dom_val);
            dQ_dbw_dom_imag[bw_idx] += cimag(dQ_dbw_dom_val);
        }
    }

    /* ── m0 gradient (real parameter) ── */
    /* bw_dom = m0² - m² - i*m0*g_bw
       ∂bw_dom/∂m0 = 2*m0 - i*g_bw (complex derivative, holomorphic)
       ∂Q/∂m0 = 2*Re(∂Q/∂bw_dom * ∂bw_dom/∂m0)
              = 2*Re(dQ_dbw_dom * (2*m0 - I*g_bw)) */
    for (int bw_idx = 0; bw_idx < n_unique_bw; bw_idx++) {
        double complex dQ_dbw_dom_val = dQ_dbw_dom_real[bw_idx] + I * dQ_dbw_dom_imag[bw_idx];
        double m0_val = m0[ctx->m0_index[bw_idx]];
        double complex g_bw_val = g_bw_real[bw_idx] + I * g_bw_imag[bw_idx];
        double complex dbw_dom_dm0 = 2.0 * m0_val - I * g_bw_val;
        double dm0 = 2.0 * creal(dQ_dbw_dom_val * dbw_dom_dm0);
        /* Scatter by m0_index */
        out_gm0[bw_idx] = dm0;
    }

    /* ── g0 gradient (real parameter) ── */
    /* First compute dQ_dg_bw = dQ_dbw_dom * ∂bw_dom/∂g_bw
       ∂bw_dom/∂g_bw = -i*m0 (derivative w.r.t. complex scalar g_bw) */
    for (int bw_idx = 0; bw_idx < n_unique_bw; bw_idx++) {
        double complex dQ_dbw_dom_val = dQ_dbw_dom_real[bw_idx] + I * dQ_dbw_dom_imag[bw_idx];
        double m0_val = m0[ctx->m0_index[bw_idx]];
        double complex dQ_dg_bw = dQ_dbw_dom_val * (-I * m0_val);

        /* dQ/dg[i] = sum over unique_bw columns dQ_dg_bw * matrix_gamma[i, bw_idx] */
        /* But matrix_gamma is sparse: only one non-zero per row gamma_row -> gamma_col_idx[gi] */
        /* Reset output g0 */
        out_gg0[bw_idx] = 0.0; /* Actually we need per-gamma-row, not per-bw */
    }

    /* Transpose scatter: for each gamma row, add contribution from its unique_bw column */
    /* dQ_dg = dQ_dg_bw_sparse: g[i] contributes only to g_bw[gamma_col_idx[i]]
       ∂g_bw[c]/∂g[i] = 1.0 if c == gamma_col_idx[i] else 0.0
       So dQ/dg[i] = dQ_dg_bw[gamma_col_idx[i]] 
       
       But we also need the g_interp factor: g[i] = g0[g0_idx] * g_interp[i]
       ∂Q/∂g0[g0_idx] = Σ_i (g0_idx == g0_index[i]) * dQ/dg[i] * g_interp[i] */
    for (int gi = 0; gi < n_gamma_rows; gi++) {
        int col = ctx->gamma_col_idx[gi];
        double complex dQ_dg_bw_val;
        /* dQ_dg_bw is per-column, but we need to select the right column */
        /* We'll just compute it again directly */
        double m0_val = m0[ctx->m0_index[col]];
        double complex dQ_dbw_dom_val = dQ_dbw_dom_real[col] + I * dQ_dbw_dom_imag[col];
        double complex dQ_dg_bw = dQ_dbw_dom_val * (-I * m0_val);

        double complex g_interp = g_interp_real[gi] + I * g_interp_imag[gi];
        double dg0 = 2.0 * creal(dQ_dg_bw * g_interp);
        out_gg0[gi] = dg0;
    }

    /* ── Scalar gradients ── */
    out_gsc[0] = 0.0; out_gsc[1] = 0.0; out_gsc[2] = 0.0;
    out_gsc[3] = 0.0; out_gsc[4] = 0.0; out_gsc[5] = 0.0;

    out_gsc[3] = dQ_dP_val * dP_dAp;  /* A_p gradient */

    /* Time-evolution parameter gradients */
    double complex d_pb_dgp = conj(pap) * ap;
    double complex d_pb_dgm = conj(pap) * poq * am;
    double complex d_pbbar_dgp = conj(pam_val) * am;
    double complex d_pbbar_dgm = conj(pam_val) * ap / poq;

    double complex dQ_dgp = dQ_dpb * d_pb_dgp + dQ_dpbbar * d_pbbar_dgp;
    double complex dQ_dgm = dQ_dpb * d_pb_dgm + dQ_dpbbar * d_pbbar_dgm;

    /* dgp/dGamma = -t/2 * gp,  dgm/dGamma = -t/2 * gm */
    double complex dgp_dG = -time_val / 2.0 * gp;
    double complex dgm_dG = -time_val / 2.0 * gm;

    /* dgp/dΔΓ = -t/4 * gm,  dgm/dΔΓ = -t/4 * gp */
    double complex dgp_dDG = -time_val / 4.0 * gm;
    double complex dgm_dDG = -time_val / 4.0 * gp;

    /* dgp/dΔm = i*t/2 * gm,  dgm/dΔm = i*t/2 * gp */
    double complex dgp_dDM = I * time_val / 2.0 * gm;
    double complex dgm_dDM = I * time_val / 2.0 * gp;

    /* ∂Q/∂x = 2*Re(dQ/dgp * dgp/dx + dQ/dgm * dgm/dx) for real x */
    out_gsc[0] = 2.0 * creal(dQ_dgp * dgp_dG + dQ_dgm * dgm_dG);       /* Gamma */
    out_gsc[1] = 2.0 * creal(dQ_dgp * dgp_dDG + dQ_dgm * dgm_dDG);     /* Delta_Gamma */
    out_gsc[2] = 2.0 * creal(dQ_dgp * dgp_dDM + dQ_dgm * dgm_dDM);     /* Delta_m */

    /* poq gradients */
    double complex d_pb_dpoq = conj(pap) * gm * am;
    double complex d_pbbar_dpoq = conj(pam_val) * (-gm / (poq * poq)) * ap;
    double complex dQ_dpoq = dQ_dpb * d_pb_dpoq + dQ_dpbbar * d_pbbar_dpoq;

    double complex exp_phi = cexp(I * pop_phi);
    out_gsc[4] = 2.0 * creal(dQ_dpoq * exp_phi);                     /* poq_rho */
    out_gsc[5] = 2.0 * creal(dQ_dpoq * poq_rho * I * exp_phi);       /* pop_phi */
}

/* ── Public CFFI-exported API ───────────────────────────────── */

CPUContext* cpu_create_context_v3(
    const int* m0_i, int n1, const int* g0_i, int n2,
    const int* g0_m, int n3, const int* mass_i, int n4,
    const int* fl_t, int n5, const int* fl_q, int n6,
    const int* bw_o, int n7, const int* fl_o, int n8,
    const int* ang_i, int n9,
    const double* ak, int n10, const double* ab, int n11,
    const double* mar, int n12, const double* mai, int n13,
    const double* gtr, int n14, const double* gti, int n15,
    double gmin, double gdel, int gbins,
    const double* mg, int n16,
    const double* ft, int n17, double flmin, double fldel, int fbins,
    int nw, int nr, int nd, int nub, int ngr,
    int nm, int nmom, int nak_, int nat, int nac,
    int n_m0p, int n_g0p,
    const int* gamma_col_idx, int n_gamma_cols
)
{
    CPUContext* c = (CPUContext*)calloc(1, sizeof(CPUContext));
    if (!c) return NULL;

    /* Store pointers (Python CFFI keeps buffers alive) */
    c->m0_index = m0_i;
    c->g0_index = g0_i;
    c->g0_mass_index = g0_m;
    c->mass_index = mass_i;
    c->fl_type = fl_t;
    c->fl_q_index = fl_q;
    c->bw_order = bw_o;
    c->fl_order = fl_o;
    c->angle_index = ang_i;
    c->angle_k = ak;
    c->angle_b = ab;
    c->matrix_angle_real = mar;
    c->matrix_angle_imag = mai;
    c->gamma_table_real = gtr;
    c->gamma_table_imag = gti;
    c->gamma_min = gmin;
    c->gamma_inv_delta = 1.0 / gdel;
    c->gamma_table_bins = gbins;
    c->matrix_gamma = mg;
    c->fl_table = ft;
    c->fl_min = flmin;
    c->fl_inv_delta = 1.0 / fldel;
    c->fl_table_bins = fbins;
    c->gamma_col_idx = gamma_col_idx;
    c->n_gamma_rows_sparse = n_gamma_cols;

    c->n_wave = nw;
    c->n_res = nr;
    c->n_decay = nd;
    c->n_unique_bw = nub;
    c->n_gamma_rows = ngr;
    c->n_mass = nm;
    c->n_momentum = nmom;
    c->n_angle_k = nak_;
    c->n_angle_total = nat;
    c->n_angle_comp = nac;
    c->n_m0_params = n_m0p;
    c->n_g0_params = n_g0p;

    return c;
}

void cpu_free_context_v3(CPUContext* ctx)
{
    if (ctx) free(ctx);
}

CPUData* cpu_load_data_v3(
    const double* mass, int nmass,
    const double* mom, int nmom,
    const double* ang, int nang,
    const double* frac, const double* time,
    const double* weight, const double* bkg,
    int ne
) {
    CPUData* d = (CPUData*)calloc(1, sizeof(CPUData));
    if (!d) return NULL;
    d->mass = mass;
    d->momentum = mom;
    d->angle = ang;
    d->frac = frac;
    d->time = time;
    d->weight = weight;
    d->bkg = bkg;
    d->n_events = ne;
    d->n_mass = nmass;
    d->n_momentum = nmom;
    d->n_angle_total = nang;
    d->n_angle_comp = 3;  /* hard-coded, same as CUDA */
    return d;
}

void cpu_free_data_v3(CPUData* d)
{
    if (d) {
        /* Don't free the data arrays — Python owns them */
        free(d);
    }
}

void cpu_compute_v3(
    CPUContext* ctx, CPUData* dh,
    const double* ck_r, const double* ck_i,
    const double* m0, const double* g0,
    double Gamma, double DG, double DM,
    double Ap, double pr, double pp,
    double norm_val, int use_norm,
    double* oQ, double* oP,
    double* ogck_r, double* ogck_i,
    double* ogm0, double* ogg0,
    double* ogsc
) {
    int ne = dh->n_events;
    int nw = ctx->n_wave;
    int nu = ctx->n_unique_bw;
    int ng = ctx->n_gamma_rows;

    /* Allocate thread-local gradient accumulators */
    int n_threads = omp_get_max_threads();
    double *tls_gck_r = (double*)calloc(n_threads * nw, sizeof(double));
    double *tls_gck_i = (double*)calloc(n_threads * nw, sizeof(double));
    double *tls_gm0   = (double*)calloc(n_threads * nu, sizeof(double));
    double *tls_gg0   = (double*)calloc(n_threads * ng, sizeof(double));
    double *tls_gsc   = (double*)calloc(n_threads * N_SCALAR, sizeof(double));

    /* Per-event output buffers (stack-allocated inside the loop) */
    double Q_total = 0.0;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double* my_gck_r = tls_gck_r + tid * nw;
        double* my_gck_i = tls_gck_i + tid * nw;
        double* my_gm0   = tls_gm0 + tid * nu;
        double* my_gg0   = tls_gg0 + tid * ng;
        double* my_gsc   = tls_gsc + tid * N_SCALAR;

        #pragma omp for reduction(+:Q_total)
        for (int e = 0; e < ne; e++) {
            double event_Q = 0.0;
            double event_P = 0.0;
            double event_gck_r[nw];
            double event_gck_i[nw];
            double event_gm0[nu];
            double event_gg0[ng];
            double event_gsc[N_SCALAR];

            /* Per-event data slice */
            const double* mass_slice = dh->mass + e * dh->n_mass;
            const double* mom_slice  = dh->momentum + e * dh->n_momentum;
            const double* ang_slice  = dh->angle + e * dh->n_angle_total * dh->n_angle_comp;

            compute_event(
                ctx,
                mass_slice, mom_slice, ang_slice,
                dh->frac[e], dh->time[e], dh->weight[e], dh->bkg[e],
                ck_r, ck_i,
                m0, g0,
                Gamma, DG, DM, Ap, pr, pp,
                use_norm, norm_val,
                &event_Q, &event_P,
                event_gck_r, event_gck_i,
                event_gm0, event_gg0, event_gsc
            );

            Q_total += event_Q;
            oP[e] = event_P;

            /* Accumulate to thread-local storage */
            for (int i = 0; i < nw; i++) {
                my_gck_r[i] += event_gck_r[i];
                my_gck_i[i] += event_gck_i[i];
            }
            for (int i = 0; i < nu; i++) {
                my_gm0[i] += event_gm0[i];
            }
            for (int i = 0; i < ng; i++) {
                my_gg0[i] += event_gg0[i];
            }
            for (int i = 0; i < N_SCALAR; i++) {
                my_gsc[i] += event_gsc[i];
            }
        }
    }

    *oQ = Q_total;

    /* Reduce thread-local accumulators to output arrays */
    for (int t = 0; t < n_threads; t++) {
        double* src_gck_r = tls_gck_r + t * nw;
        double* src_gck_i = tls_gck_i + t * nw;
        double* src_gm0   = tls_gm0 + t * nu;
        double* src_gg0   = tls_gg0 + t * ng;
        double* src_gsc   = tls_gsc + t * N_SCALAR;

        if (t == 0) {
            memcpy(ogck_r, src_gck_r, nw * sizeof(double));
            memcpy(ogck_i, src_gck_i, nw * sizeof(double));
            memcpy(ogm0,   src_gm0,   nu * sizeof(double));
            memcpy(ogg0,   src_gg0,   ng * sizeof(double));
            memcpy(ogsc,   src_gsc,   N_SCALAR * sizeof(double));
        } else {
            for (int i = 0; i < nw; i++) {
                ogck_r[i] += src_gck_r[i];
                ogck_i[i] += src_gck_i[i];
            }
            for (int i = 0; i < nu; i++) ogm0[i] += src_gm0[i];
            for (int i = 0; i < ng; i++) ogg0[i] += src_gg0[i];
            for (int i = 0; i < N_SCALAR; i++) ogsc[i] += src_gsc[i];
        }
    }

    /* NOTE: ogm0 is [n_unique_bw] and ogg0 is [n_gamma_rows].
       The Python wrapper scatters these by m0_index / g0_index
       to produce the final per-parameter gradients. */

    free(tls_gck_r);
    free(tls_gck_i);
    free(tls_gm0);
    free(tls_gg0);
    free(tls_gsc);
}
