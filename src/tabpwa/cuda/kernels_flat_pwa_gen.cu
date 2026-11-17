/**
 * cuda_flat_pwa_gen — the flat `pwa_gen` configuration on the GPU.
 *
 * Case-1 copy (kernel policy §2.1): new file, own symbols, nothing shared
 * with another kernel.  It consumes the arrays emitted by
 * `tabpwa/pwa_gen_config.build_from_groups` **as emitted** — the full,
 * non-deduped form — reading each table at the index the config names.
 *
 *     ka_{e,a}  = Π_j cos( angle_{e, angle_index[a], j}·angle_k[a,j] + angle_b[a,j] )
 *     fa_{e,k,p} = Σ_a ka_{e,a} · am[k,p,a]   (am = matrix_angle
 *                     re-ordered from its p-major (a, p*n_wave+k) form
 *                     by the wrapper, so this loop is unchanged)
 *     fl_{e,k}  = Π_d fl_table[k,d]( q_{e, fl_q_index[k,d]} )
 *     nu_{e,k}  = Π_r num_table[k,r]( m_{e, num_m_index[k,r]} )        (complex)
 *     s_{e,k,l} = Σ_m gamma[k,l,m]·coeff[m]·basis_m(m_e)               (complex)
 *     D_{e,k}   = Π_l s_{e,k,l}
 *     common_{e,k} = fl·nu / D                     (NO angular factor)
 *     A_{e,p}   = Σ_k ck_k · common_{e,k} · fa_{e,k,p}
 *     P_e       = Σ_p |A_{e,p}|²
 *
 * `coeff` is flat and `n_basis` long: the builder appends one entry per
 * (factor, basis row) and never shares, so each basis column belongs to
 * exactly one (wave, factor).  A caller that ran the host-side `dedup` pass
 * must NOT feed the folded tables here — this kernel is the full-form
 * consumer.
 *
 * Backward, matching `numpy_flat` exactly (which applies the same algebra
 * `numpy_pwa` does — `grad_ck` is the raw Wirtinger half; the 2·Re real
 * convention is applied downstream by `CKProduct.backprop_grad`):
 *
 *     dQ/dA_p        = dQ/dP_e · conj(A_p)              (normed: -w/(P+b·norm))
 *     dQ/dck_k       = Σ_p dQ/dA_p · common_k · fa_{k,p}
 *     dQ/dcommon_k   = Σ_p dQ/dA_p · ck_k · fa_{k,p}
 *     dQ/dD_k        = -dQ/dcommon_k · common_k / D_k
 *     dQ/ds_{k,l}    = dQ/dD_k · Π_{l'≠l} s_{k,l'}       (prefix·suffix, no divide)
 *     dQ/dcoeff_m    = Σ_{k,l} 2·Re Σ_e dQ/ds_{k,l}·gamma[k,l,m]·basis_m(m_e)
 *
 * One fused kernel over a tile of `FLAT_TE` events; `ka` is evaluated once
 * per (event, angle basis) and reused by every (wave, projection).
 *
 * ## Grouped log-sum objective (`resolution_size` = rs > 1, normed only)
 *
 * The per-event NLL is replaced by one log per group of `rs` contiguous
 * events (group g covers global events [g·rs, min((g+1)·rs, ne))):
 *
 *     W_g  = Σ_{e∈g} w_e
 *     S_g  = Σ_{e∈g} w_e·(P_e/norm + bkg_e)          Pb_g = Σ_{e∈g} w_e·P_e
 *     Q    = -Σ_g W_g·log(S_g/W_g)                   (skipped when W_g <= 0)
 *     dQ/dP_e = -(W_g/S_g)·(w_e/norm)                (scale_g = -W_g/S_g)
 *     dnorm   = Σ_g W_g·Pb_g/(S_g·norm²)
 *
 * A group of 100 events spans several tiles, so the tile kernel cannot hold
 * a group's partials in shared memory.  The grouped path is a multi-pass
 * shape over the SAME tile kernel with an `int mode` switch:
 *
 *   mode 1 (P)    : forward, write P_out[ge], atomicAdd the group partials
 *                   into the global gS/gPb/gW arrays (no gradient phases).
 *   `grp_nll`     : one thread per group folds them into Q/dnorm (into `acc`)
 *                   and writes gscale[g] = -W_g/S_g.
 *   `scale`       : one thread per event, dqp[ge] = gscale[ge/rs]·w_e/norm.
 *   mode 2 (grad) : forward again, build dA = dqp[ge]·conj(A), gradient phases.
 *
 * `rs == 1` and `use_norm == 0` keep the original single-pass shape (mode 0)
 * untouched, so `rs == 1` follows exactly the per-event arithmetic (the Q /
 * dnorm sums are still `atomicAdd`-order dependent, as they always were).
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>

#define FLAT_BLOCK 256
#define FLAT_TE    4          /* events per tile (shared-memory bound) */
#define FLAT_BATCH 50000

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                \
        }                                                                    \
    } while (0)

/* ── Catmull-Rom interpolation (the repo-wide idiom) ──────────────────── */
__device__ double flat_cr1d(double pm1, double p0, double p1, double p2,
                            double t) {
    return p0 + 0.5 * t * (-pm1 + p1 + t * (2.0 * pm1 - 5.0 * p0 + 4.0 * p1
                                            - p2 + t * (-pm1 + 3.0 * p0
                                                        - 3.0 * p1 + p2)));
}

__device__ double flat_interp_real(const double* __restrict__ t, int row,
                                   double x, double xmin, double xdelta,
                                   int bins) {
    double diff = (x - xmin) / xdelta;
    int b = max(0, min((int)floor(diff), bins - 2));
    double s = max(0.0, min(diff - b, 1.0));
    int base = row * bins + b;
    int lo = row * bins, hi = (row + 1) * bins - 1;
    int im1 = base > lo ? base - 1 : base;
    int i2 = base + 2 <= hi ? base + 2 : base + 1;
    return flat_cr1d(t[im1], t[base], t[base + 1], t[i2], s);
}

__device__ void flat_interp_c(const double* __restrict__ tr,
                              const double* __restrict__ ti, int row,
                              double x, double xmin, double xdelta, int bins,
                              double* orr, double* oii) {
    double diff = (x - xmin) / xdelta;
    int b = max(0, min((int)floor(diff), bins - 2));
    double s = max(0.0, min(diff - b, 1.0));
    int base = row * bins + b;
    int lo = row * bins, hi = (row + 1) * bins - 1;
    int im1 = base > lo ? base - 1 : base;
    int i2 = base + 2 <= hi ? base + 2 : base + 1;
    *orr = flat_cr1d(tr[im1], tr[base], tr[base + 1], tr[i2], s);
    *oii = flat_cr1d(ti[im1], ti[base], ti[base + 1], ti[i2], s);
}

/* One denominator factor s_{e,k,l} = Σ_m gamma·coeff·basis_m(m_e). */
__device__ void flat_den_factor(
    const double* __restrict__ bas_r, const double* __restrict__ bas_i,
    const int* __restrict__ bas_m, const double* __restrict__ gamma,
    const double* __restrict__ coeff,
    const double* __restrict__ mass, int ge, int n_mass,
    double m_min, double m_delta, int m_bins,
    int n_basis, int k, int l, int n_res, double* out_r, double* out_i)
{
    double sr = 0.0, si = 0.0;
    const int gbase = (k * n_res + l) * n_basis;
    for (int m = 0; m < n_basis; m++) {
        const double g = gamma[gbase + m];
        if (g == 0.0) continue;
        const double mv = mass[(size_t)ge * n_mass + bas_m[m]];
        double vr, vi;
        flat_interp_c(bas_r, bas_i, m, mv, m_min, m_delta, m_bins, &vr, &vi);
        const double w = g * coeff[m];
        sr += w * vr;
        si += w * vi;
    }
    *out_r = sr; *out_i = si;
}

/* fa_{e,k,p} = Σ_a ka_{e,a}·am[k,p,a] (complex); am is the
 * wrapper's (n_wave, n_proj, n_ab) view of the p-major matrix_angle. */
__device__ void flat_fa(const double* __restrict__ s_ka,
                        const double* __restrict__ am_r,
                        const double* __restrict__ am_i,
                        int e, int k, int p, int n_ab, int n_proj,
                        double* out_r, double* out_i)
{
    double far = 0.0, fai = 0.0;
    for (int a = 0; a < n_ab; a++) {
        const double ka = s_ka[e * n_ab + a];
        if (ka == 0.0) continue;
        const int o = (k * n_proj + p) * n_ab + a;
        far += ka * am_r[o];
        fai += ka * am_i[o];
    }
    *out_r = far; *out_i = fai;
}

__global__ void flat_tile_kernel(
    /* config (fixed for the fit) */
    const double* __restrict__ fl_table,   /* [n_wave*n_decay, q_bins] */
    const int* __restrict__ fl_q_index,    /* [n_wave, n_decay] */
    const double* __restrict__ num_r,      /* [n_wave*n_res, m_bins] */
    const double* __restrict__ num_i,
    const int* __restrict__ num_m_index,   /* [n_wave, n_res] */
    const double* __restrict__ bas_r,      /* [n_basis, m_bins] */
    const double* __restrict__ bas_i,
    const int* __restrict__ bas_m_index,   /* [n_basis] */
    const double* __restrict__ gamma,      /* [n_wave*n_res*n_basis] */
    const int* __restrict__ angle_index,   /* [n_ab] */
    const double* __restrict__ angle_k,    /* [n_ab, n_comp] */
    const double* __restrict__ angle_b,
    const double* __restrict__ am_r,       /* [n_wave, n_proj, n_ab] */
    const double* __restrict__ am_i,
    double m_min, double m_delta, int m_bins,
    double q_min, double q_delta, int q_bins,
    /* per-call */
    const double* __restrict__ ck_r, const double* __restrict__ ck_i,
    const double* __restrict__ coeff,
    double norm, int use_norm,
    /* grouped log-sum (mode 1/2 only; NULL in the single-pass path) */
    int mode, int rsize,
    double* __restrict__ gS, double* __restrict__ gPb,
    double* __restrict__ gW, const double* __restrict__ dqp,
    /* data */
    const double* __restrict__ mass, const double* __restrict__ mom,
    const double* __restrict__ ang,
    const double* __restrict__ weight, const double* __restrict__ bkg,
    /* dims */
    int ne, int n_wave, int n_proj, int n_res, int n_decay, int n_basis,
    int n_ab, int n_comp, int n_mass, int n_momentum, int n_angle_total,
    /* outputs */
    double* __restrict__ P_out,
    double* __restrict__ g_ck_r, double* __restrict__ g_ck_i,
    double* __restrict__ g_coeff,
    double* __restrict__ acc)              /* [2]: Q, dnorm */
{
    extern __shared__ double sh[];
    /* s_ka   [TE·n_ab]                    real
       s_fl   [TE·n_wave]                  real
       s_nu   [TE·n_wave·2]                complex
       s_s    [TE·n_wave·n_res·2]          complex   per-factor denominators
       s_D    [TE·n_wave·2]                complex   their product
       s_A    [TE·n_proj·2]                complex
       s_dA   [TE·n_proj·2]                complex
       s_pe   [TE]  s_q [TE]                                        */
    double* s_ka = sh;
    double* s_fl = s_ka + (size_t)FLAT_TE * n_ab;
    double* s_nu = s_fl + (size_t)FLAT_TE * n_wave;
    double* s_s  = s_nu + (size_t)FLAT_TE * n_wave * 2;
    double* s_D  = s_s  + (size_t)FLAT_TE * n_wave * n_res * 2;
    double* s_A  = s_D  + (size_t)FLAT_TE * n_wave * 2;
    double* s_dA = s_A  + (size_t)FLAT_TE * n_proj * 2;
    double* s_pe = s_dA + (size_t)FLAT_TE * n_proj * 2;
    double* s_q  = s_pe + FLAT_TE;

    const int tid = threadIdx.x, nt = blockDim.x;
    const int e0 = blockIdx.x * FLAT_TE;
    const int evn = min(FLAT_TE, ne - e0);
    if (evn <= 0) return;

    /* ── phase 0: ka — once per (event, angle basis) ──────────────────── */
    for (int idx = tid; idx < evn * n_ab; idx += nt) {
        const int e = idx / n_ab, a = idx - e * n_ab;
        const int ge = e0 + e;
        const int col = angle_index[a];
        double prod = 1.0;
        for (int j = 0; j < n_comp; j++) {
            const double th =
                ang[((size_t)ge * n_angle_total + col) * n_comp + j];
            prod *= cos(th * angle_k[a * n_comp + j]
                        + angle_b[a * n_comp + j]);
        }
        s_ka[idx] = prod;
    }
    __syncthreads();

    /* ── phase 1: fl, nu, the per-factor s_{k,l} and their product D ──── */
    for (int idx = tid; idx < evn * n_wave; idx += nt) {
        const int e = idx / n_wave, k = idx - e * n_wave;
        const int ge = e0 + e;

        double fl = 1.0;
        for (int d = 0; d < n_decay; d++) {
            const int qc = fl_q_index[k * n_decay + d];
            const double qv = mom[(size_t)ge * n_momentum + qc];
            fl *= flat_interp_real(fl_table, k * n_decay + d, qv,
                                   q_min, q_delta, q_bins);
        }
        double nr = 1.0, ni = 0.0;
        for (int r = 0; r < n_res; r++) {
            const int mc = num_m_index[k * n_res + r];
            const double mv = mass[(size_t)ge * n_mass + mc];
            double vr, vi;
            flat_interp_c(num_r, num_i, k * n_res + r, mv,
                          m_min, m_delta, m_bins, &vr, &vi);
            const double t_r = nr * vr - ni * vi;
            ni = nr * vi + ni * vr;
            nr = t_r;
        }
        s_fl[idx] = fl;
        s_nu[idx * 2 + 0] = nr;
        s_nu[idx * 2 + 1] = ni;

        double dr = 1.0, di = 0.0;
        for (int l = 0; l < n_res; l++) {
            double sr, si;
            flat_den_factor(bas_r, bas_i, bas_m_index, gamma, coeff,
                            mass, ge, n_mass, m_min, m_delta, m_bins,
                            n_basis, k, l, n_res, &sr, &si);
            s_s[(idx * n_res + l) * 2 + 0] = sr;
            s_s[(idx * n_res + l) * 2 + 1] = si;
            const double t_r = dr * sr - di * si;
            di = dr * si + di * sr;
            dr = t_r;
        }
        s_D[idx * 2 + 0] = dr;
        s_D[idx * 2 + 1] = di;
    }
    __syncthreads();

    /* ── phase 2: A_{e,p} = Σ_k ck_k · (fl·nu/D) · fa_{k,p} ───────────── */
    for (int idx = tid; idx < evn * n_proj; idx += nt) {
        const int e = idx / n_proj, p = idx - e * n_proj;
        double ar = 0.0, ai = 0.0;
        for (int k = 0; k < n_wave; k++) {
            double far, fai;
            flat_fa(s_ka, am_r, am_i, e, k, p, n_ab, n_proj, &far, &fai);
            const double flr = s_fl[e * n_wave + k];
            const double nur = s_nu[(e * n_wave + k) * 2 + 0];
            const double nui = s_nu[(e * n_wave + k) * 2 + 1];
            const double fr = flr * nur, fi = flr * nui;   /* F = fl·nu */
            const double dr = s_D[(e * n_wave + k) * 2 + 0];
            const double di = s_D[(e * n_wave + k) * 2 + 1];
            const double den = dr * dr + di * di;
            /* common = F/D */
            const double crn = (fr * dr + fi * di) / den;
            const double cin = (fi * dr - fr * di) / den;
            /* common · fa */
            const double tr = crn * far - cin * fai;
            const double ti = crn * fai + cin * far;
            ar += ck_r[k] * tr - ck_i[k] * ti;
            ai += ck_r[k] * ti + ck_i[k] * tr;
        }
        s_A[idx * 2 + 0] = ar;
        s_A[idx * 2 + 1] = ai;
    }
    __syncthreads();

    /* ── phase 3: P_e, the objective, and dQ/dP ───────────────────────── */
    for (int e = tid; e < evn; e += nt) {
        double pe = 0.0;
        for (int p = 0; p < n_proj; p++) {
            const double ar = s_A[(e * n_proj + p) * 2 + 0];
            const double ai = s_A[(e * n_proj + p) * 2 + 1];
            pe += ar * ar + ai * ai;
        }
        const int ge = e0 + e;
        const double wv = weight[ge];
        const double bv = bkg[ge];
        s_pe[e] = pe;
        s_q[e] = use_norm ? wv * (pe / norm + bv) : wv * pe;
        if (P_out) P_out[ge] = pe;
        if (mode == 1) {
            /* grouped log-sum, pass A: per-group partials (a group spans
               tiles, so these live in global memory) */
            const int g = ge / rsize;
            atomicAdd(gS + g, wv * (pe / norm + bv));
            atomicAdd(gPb + g, wv * pe);
            atomicAdd(gW + g, wv);
        } else if (mode == 2) {
            /* grouped log-sum, pass B: the objective is already folded */
        } else if (use_norm) {
            atomicAdd(acc + 0, -wv * log(pe / norm + bv));
            atomicAdd(acc + 1, wv * pe / (norm * (pe + bv * norm)));
        } else {
            atomicAdd(acc + 0, wv * pe);
        }
    }
    __syncthreads();

    if (mode == 1) return;      /* pass A: no gradient phases */

    for (int idx = tid; idx < evn * n_proj; idx += nt) {
        const int e = idx / n_proj;
        const int ge = e0 + e;
        const double wv = weight[ge], bv = bkg[ge];
        const double pe = s_pe[e];
        /* mode 2 carries the group scale from the global dqp array */
        const double dqpe = (mode == 2) ? dqp[ge]
                         : (use_norm ? -wv / (pe + bv * norm) : wv);
        const double ar = s_A[idx * 2 + 0], ai = s_A[idx * 2 + 1];
        s_dA[idx * 2 + 0] = dqpe * ar;      /* dQ/dA = dQ/dP · conj(A) */
        s_dA[idx * 2 + 1] = -dqpe * ai;
    }
    __syncthreads();

    /* ── phase 4: grad ck — thread per k, serial over the tile ────────── */
    for (int k = tid; k < n_wave; k += nt) {
        double sr = 0.0, si = 0.0;
        for (int e = 0; e < evn; e++) {
            const double flr = s_fl[e * n_wave + k];
            const double nur = s_nu[(e * n_wave + k) * 2 + 0];
            const double nui = s_nu[(e * n_wave + k) * 2 + 1];
            const double fr = flr * nur, fi = flr * nui;
            const double dr = s_D[(e * n_wave + k) * 2 + 0];
            const double di = s_D[(e * n_wave + k) * 2 + 1];
            const double den = dr * dr + di * di;
            const double crn = (fr * dr + fi * di) / den;
            const double cin = (fi * dr - fr * di) / den;
            for (int p = 0; p < n_proj; p++) {
                double far, fai;
                flat_fa(s_ka, am_r, am_i, e, k, p, n_ab, n_proj, &far, &fai);
                const double gar = s_dA[(e * n_proj + p) * 2 + 0];
                const double gai = s_dA[(e * n_proj + p) * 2 + 1];
                /* dQ/dck_k += dQ/dA_p · common_k · fa_{k,p} */
                const double tr = crn * far - cin * fai;
                const double ti = crn * fai + cin * far;
                sr += gar * tr - gai * ti;
                si += gar * ti + gai * tr;
            }
        }
        atomicAdd(g_ck_r + k, sr);
        atomicAdd(g_ck_i + k, si);
    }
    __syncthreads();

    /* ── phase 5: grad coeff ───────────────────────────────────────────
       dQ/dD_k = -dQ/dcommon_k · common_k / D_k,
       dQ/dcommon_k = Σ_p dQ/dA_p · ck_k · fa_{k,p},
       dQ/ds_{k,l} = dQ/dD_k · Π_{l'≠l} s_{k,l'}   (prefix·suffix),
       dQ/dcoeff_m += 2Re Σ_e dQ/ds_{k,l}·gamma[k,l,m]·basis_m(m_e).      */
    for (int k = tid; k < n_wave; k += nt) {
        for (int l = 0; l < n_res; l++) {
            for (int m = 0; m < n_basis; m++) {
                const double g = gamma[(k * n_res + l) * n_basis + m];
                if (g == 0.0) continue;
                double out = 0.0;
                for (int e = 0; e < evn; e++) {
                    const int ge = e0 + e;
                    /* dQ/dcommon_k */
                    double dcr = 0.0, dci = 0.0;
                    for (int p = 0; p < n_proj; p++) {
                        double far, fai;
                        flat_fa(s_ka, am_r, am_i, e, k, p, n_ab, n_proj,
                                &far, &fai);
                        const double gar = s_dA[(e * n_proj + p) * 2 + 0];
                        const double gai = s_dA[(e * n_proj + p) * 2 + 1];
                        /* dQ/dA·fa·ck */
                        const double tr = gar * far - gai * fai;
                        const double ti = gar * fai + gai * far;
                        dcr += ck_r[k] * tr - ck_i[k] * ti;
                        dci += ck_r[k] * ti + ck_i[k] * tr;
                    }
                    const double flr = s_fl[e * n_wave + k];
                    const double nur = s_nu[(e * n_wave + k) * 2 + 0];
                    const double nui = s_nu[(e * n_wave + k) * 2 + 1];
                    const double fr = flr * nur, fi = flr * nui;
                    const double dr = s_D[(e * n_wave + k) * 2 + 0];
                    const double di = s_D[(e * n_wave + k) * 2 + 1];
                    const double den = dr * dr + di * di;
                    const double crn = (fr * dr + fi * di) / den;
                    const double cin = (fi * dr - fr * di) / den;
                    /* dQ/dD = -dQ/dcommon · common / D */
                    const double qr = -(dcr * crn - dci * cin);
                    const double qi = -(dcr * cin + dci * crn);
                    const double ddr = (qr * dr + qi * di) / den;
                    const double ddi = (qi * dr - qr * di) / den;
                    /* Π_{l'≠l} s_{k,l'} */
                    double pr = 1.0, pi = 0.0;
                    for (int l2 = 0; l2 < n_res; l2++) {
                        if (l2 == l) continue;
                        const double ur =
                            s_s[((e * n_wave + k) * n_res + l2) * 2 + 0];
                        const double ui =
                            s_s[((e * n_wave + k) * n_res + l2) * 2 + 1];
                        const double t_r = pr * ur - pi * ui;
                        pi = pr * ui + pi * ur;
                        pr = t_r;
                    }
                    const double gsr = ddr * pr - ddi * pi;
                    const double gsi = ddi * pr + ddr * pi;
                    /* · gamma · basis_m(m_e) */
                    const double mv = mass[(size_t)ge * n_mass + bas_m_index[m]];
                    double br, bi;
                    flat_interp_c(bas_r, bas_i, m, mv, m_min, m_delta,
                                  m_bins, &br, &bi);
                    /* reference: 2·Re(gs · g · b) — a COMPLEX PRODUCT, so
                       the cross terms subtract (Re(gs·b) = gsr·br − gsi·bi),
                       NOT the dot product (which would be Re(gs·conj b)). */
                    out += 2.0 * (gsr * br - gsi * bi) * g;
                }
                atomicAdd(g_coeff + m, out);
            }
        }
    }
}

/*=============================================================================
 * Grouped log-sum helpers (resolution_size > 1, normed calls)
 *===========================================================================*/

/* Fold the per-group partials into Q/dnorm and publish the group scale.
   One block over the groups, tree-reduced into `acc` (Q, dnorm). */
__global__ void flat_grp_nll_kernel(
    const double* __restrict__ gS, const double* __restrict__ gPb,
    const double* __restrict__ gW,
    int n_groups, double norm,
    double* __restrict__ gscale, double* __restrict__ acc)
{
    extern __shared__ double sh[];
    const int tid = threadIdx.x, bs = blockDim.x;
    double* sq = sh;
    double* sn = sh + bs;
    double q = 0.0, dn = 0.0;
    for (int g = tid; g < n_groups; g += bs) {
        const double W = gW[g];
        if (W <= 0.0) { gscale[g] = 0.0; continue; }
        const double S = gS[g];
        q += -W * log(S / W);
        dn += W * gPb[g] / (S * norm * norm);
        gscale[g] = -W / S;
    }
    sq[tid] = q; sn[tid] = dn;
    __syncthreads();
    for (int st = bs / 2; st > 0; st >>= 1) {
        if (tid < st) { sq[tid] += sq[tid + st]; sn[tid] += sn[tid + st]; }
        __syncthreads();
    }
    if (tid == 0) { atomicAdd(acc + 0, sq[0]); atomicAdd(acc + 1, sn[0]); }
}

/* dqp[ge] = gscale[ge/rsize] · w_e/norm — one thread per event. */
__global__ void flat_scale_kernel(
    const double* __restrict__ gscale, const double* __restrict__ weight,
    double norm, int rsize, int ne, double* __restrict__ dqp)
{
    const int ge = blockIdx.x * blockDim.x + threadIdx.x;
    if (ge >= ne) return;
    dqp[ge] = gscale[ge / rsize] * weight[ge] / norm;
}

/*=============================================================================
 * Host side
 *===========================================================================*/
/* Per-LOAD state.  Every `load_data` call allocates its own handle, so the
   phase-space sample and the data sample coexist: the ``Fitter`` keeps both
   alive for the whole fit and integrates the norm over the phase-space one
   while evaluating the likelihood on the data one.  Nothing here may live on
   ``FlatCtx`` — the context is shared by all handles of one kernel instance. */
typedef struct {
    /* loaded data (device) */
    double *mass, *mom, *ang, *weight, *bkg;
    double *P_dev;
    double *h_P;                /* pinned per-event P staging [ne] */
    int ne;
    /* grouped log-sum scratch (rsize > 1 only) */
    int rsize;                  /* group size captured at load time */
    int n_groups;
    double *gS, *gPb, *gW, *gscale, *dqp;
} FlatDataHandle;

typedef struct {
    /* config arrays (device) */
    int *fl_q_index, *num_m_index, *bas_m_index, *angle_index;
    double *fl_table, *num_r, *num_i, *bas_r, *bas_i, *gamma;
    double *angle_k, *angle_b, *am_r, *am_i;
    double m_min, m_delta, q_min, q_delta;
    int m_bins, q_bins;
    int n_wave, n_proj, n_res, n_decay, n_basis, n_ab, n_comp;
    int n_mass, n_momentum, n_angle_total;
    int batch_size;
    int rsize;                  /* grouped log-sum group size (>= 1) */
    /* per-call param staging (pinned) */
    double *h_ck_r, *h_ck_i, *h_coeff;
    double *p_ck_r, *p_ck_i, *p_coeff;
    double *acc, *h_acc;
    double *g_ck_r, *g_ck_i, *g_coeff;
    cudaStream_t stream;
    int smem_bytes;
} FlatCtx;

static int* _up_int(const int* src, int n) {
    if (n <= 0) return NULL;
    int* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(int),
                          cudaMemcpyHostToDevice));
    return d;
}

static double* _up_dbl(const double* src, int n) {
    if (n <= 0) return NULL;
    double* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(double),
                          cudaMemcpyHostToDevice));
    return d;
}

static size_t _flat_shm_bytes(int n_ab, int n_wave, int n_proj, int n_res) {
    double d = (double)FLAT_TE * (n_ab + n_wave + 2.0 * n_wave
                                  + 2.0 * n_wave * n_res + 2.0 * n_wave
                                  + 2.0 * n_proj + 2.0 * n_proj + 2.0);
    return (size_t)(d * sizeof(double));
}

extern "C" {

void cuda_free_data_flat_pwa_gen(void* vh);   /* fwd: used by load_data fail */

void* cuda_create_context_flat_pwa_gen(
    const double* fl_table, int n_fl_table, const int* fl_q_index,
    int n_fl_q, const double* num_r, const double* num_i, int n_num,
    const int* num_m_index, int n_num_m, const double* bas_r,
    const double* bas_i, int n_bas, const int* bas_m_index, int n_bas_m,
    const double* gamma, int n_gamma, const int* angle_index, int n_ai,
    const double* angle_k, int n_ak, const double* angle_b, int n_ab_,
    const double* am_r, const double* am_i, int n_am,
    double m_min, double m_delta, int m_bins,
    double q_min, double q_delta, int q_bins,
    int n_wave, int n_proj, int n_res, int n_decay, int n_basis,
    int n_ab, int n_comp, int n_mass, int n_momentum, int n_angle_total,
    int batch_size, int rsize)
{
    FlatCtx* c = (FlatCtx*)calloc(1, sizeof(FlatCtx));
    if (!c) return NULL;
    c->fl_table = _up_dbl(fl_table, n_fl_table);
    c->fl_q_index = _up_int(fl_q_index, n_fl_q);
    c->num_r = _up_dbl(num_r, n_num);
    c->num_i = _up_dbl(num_i, n_num);
    c->num_m_index = _up_int(num_m_index, n_num_m);
    c->bas_r = _up_dbl(bas_r, n_bas);
    c->bas_i = _up_dbl(bas_i, n_bas);
    c->bas_m_index = _up_int(bas_m_index, n_bas_m);
    c->gamma = _up_dbl(gamma, n_gamma);
    c->angle_index = _up_int(angle_index, n_ai);
    c->angle_k = _up_dbl(angle_k, n_ak);
    c->angle_b = _up_dbl(angle_b, n_ab_);
    c->am_r = _up_dbl(am_r, n_am);
    c->am_i = _up_dbl(am_i, n_am);
    c->m_min = m_min; c->m_delta = m_delta; c->m_bins = m_bins;
    c->q_min = q_min; c->q_delta = q_delta; c->q_bins = q_bins;
    c->n_wave = n_wave; c->n_proj = n_proj; c->n_res = n_res;
    c->n_decay = n_decay; c->n_basis = n_basis; c->n_ab = n_ab;
    c->n_comp = n_comp; c->n_mass = n_mass; c->n_momentum = n_momentum;
    c->n_angle_total = n_angle_total;
    c->batch_size = batch_size > 0 ? batch_size : FLAT_BATCH;
    c->rsize = rsize > 0 ? rsize : 1;

    c->smem_bytes = (int)_flat_shm_bytes(n_ab, n_wave, n_proj, n_res);
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (c->smem_bytes > optin) {
        fprintf(stderr,
                "cuda_flat_pwa_gen: shared tile %d B exceeds the device "
                "opt-in limit %d B (n_wave %d, n_res %d, n_basis %d)\n",
                c->smem_bytes, optin, n_wave, n_res, n_basis);
        free(c);
        return NULL;
    }

    CUDA_CHECK(cudaStreamCreate(&c->stream));
    CUDA_CHECK(cudaMalloc(&c->p_ck_r, (size_t)n_wave * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->p_ck_i, (size_t)n_wave * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->p_coeff, (size_t)n_basis * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->g_ck_r, (size_t)n_wave * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->g_ck_i, (size_t)n_wave * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->g_coeff, (size_t)n_basis * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&c->h_ck_r, (size_t)n_wave * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&c->h_ck_i, (size_t)n_wave * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&c->h_coeff, (size_t)n_basis * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->acc, 2 * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&c->h_acc, 2 * sizeof(double)));
    return c;
}

void* cuda_load_data_flat_pwa_gen(void* vctx, const double* mass,
                                  const int* n_mass_, const double* mom,
                                  const int* n_mom_, const double* ang,
                                  const int* n_ang_, const double* weight,
                                  const double* bkg, int ne)
{
    const FlatCtx* c = (const FlatCtx*)vctx;
    (void)n_mass_; (void)n_mom_; (void)n_ang_;
    if (!c || ne <= 0) return NULL;
    FlatDataHandle* h = (FlatDataHandle*)calloc(1, sizeof(FlatDataHandle));
    if (!h) return NULL;
    h->ne = ne;
    h->mass = _up_dbl(mass, ne * c->n_mass);
    h->mom = _up_dbl(mom, ne * c->n_momentum);
    h->ang = _up_dbl(ang, ne * c->n_angle_total * c->n_comp);
    h->weight = _up_dbl(weight, ne);
    h->bkg = _up_dbl(bkg, ne);
    if (cudaMalloc(&h->P_dev, (size_t)ne * sizeof(double)) != cudaSuccess)
        goto fail;
    if (cudaMallocHost(&h->h_P, (size_t)ne * sizeof(double)) != cudaSuccess) {
        h->h_P = NULL;
        goto fail;
    }
    /* grouped log-sum scratch: n_groups arrays + one dqp slot per event */
    h->rsize = c->rsize > 0 ? c->rsize : 1;
    if (h->rsize > 1) {
        h->n_groups = (ne + h->rsize - 1) / h->rsize;
        if (h->n_groups > 0) {
            if (cudaMalloc(&h->gS, (size_t)h->n_groups * sizeof(double))
                    != cudaSuccess) goto fail;
            if (cudaMalloc(&h->gPb, (size_t)h->n_groups * sizeof(double))
                    != cudaSuccess) goto fail;
            if (cudaMalloc(&h->gW, (size_t)h->n_groups * sizeof(double))
                    != cudaSuccess) goto fail;
            if (cudaMalloc(&h->gscale, (size_t)h->n_groups * sizeof(double))
                    != cudaSuccess) goto fail;
            if (cudaMalloc(&h->dqp, (size_t)ne * sizeof(double))
                    != cudaSuccess) goto fail;
        }
    }
    return h;
fail:
    cudaGetLastError();          /* clear the sticky error for the caller */
    cuda_free_data_flat_pwa_gen(h);
    return NULL;
}

void cuda_free_data_flat_pwa_gen(void* vh) {
    if (!vh) return;
    FlatDataHandle* h = (FlatDataHandle*)vh;
    cudaFree(h->mass); cudaFree(h->mom); cudaFree(h->ang);
    cudaFree(h->weight); cudaFree(h->bkg); cudaFree(h->P_dev);
    cudaFree(h->gS); cudaFree(h->gPb); cudaFree(h->gW);
    cudaFree(h->gscale); cudaFree(h->dqp);
    if (h->h_P) cudaFreeHost(h->h_P);
    free(h);
}

void cuda_free_context_flat_pwa_gen(void* vctx) {
    FlatCtx* c = (FlatCtx*)vctx;
    if (!c) return;
    #define F(p) if (c->p) cudaFree(c->p)
    #define FH(p) if (c->p) cudaFreeHost(c->p)
    F(fl_table); F(fl_q_index); F(num_r); F(num_i); F(num_m_index);
    F(bas_r); F(bas_i); F(bas_m_index); F(gamma); F(angle_index);
    F(angle_k); F(angle_b); F(am_r); F(am_i);
    F(p_ck_r); F(p_ck_i); F(p_coeff); F(acc);
    F(g_ck_r); F(g_ck_i); F(g_coeff);
    FH(h_ck_r); FH(h_ck_i); FH(h_coeff); FH(h_acc);
    #undef F
    #undef FH
    if (c->stream) cudaStreamDestroy(c->stream);
    free(c);
}

void cuda_compute_flat_pwa_gen(
    void* vctx, void* vdh, const double* ck_r, const double* ck_i,
    const double* coeff, double norm, int use_norm,
    double* oQ, double* oDn, double* oP,
    double* og_ck_r, double* og_ck_i, double* og_coeff)
{
    FlatCtx* c = (FlatCtx*)vctx;
    FlatDataHandle* h = (FlatDataHandle*)vdh;
    const int N = c->n_wave, NB = c->n_basis;
    const int ne = h ? h->ne : 0;

    memcpy(c->h_ck_r, ck_r, (size_t)N * sizeof(double));
    memcpy(c->h_ck_i, ck_i, (size_t)N * sizeof(double));
    memcpy(c->h_coeff, coeff, (size_t)NB * sizeof(double));
    cudaMemcpyAsync(c->p_ck_r, c->h_ck_r, (size_t)N * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    cudaMemcpyAsync(c->p_ck_i, c->h_ck_i, (size_t)N * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    cudaMemcpyAsync(c->p_coeff, c->h_coeff, (size_t)NB * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);

    cudaMemsetAsync(c->acc, 0, 2 * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_ck_r, 0, (size_t)N * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_ck_i, 0, (size_t)N * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_coeff, 0, (size_t)NB * sizeof(double), c->stream);
    memset(og_ck_r, 0, (size_t)N * sizeof(double));
    memset(og_ck_i, 0, (size_t)N * sizeof(double));
    memset(og_coeff, 0, (size_t)NB * sizeof(double));

    /* grouped log-sum objective: only for a normed call with rsize > 1 and
       the scratch present (i.e. it was allocated for this data load) */
    const int rs = (h && h->rsize > 0) ? h->rsize : 1;
    const int grouped = (rs > 1 && use_norm && ne > 0 &&
                         h->gS && h->dqp) ? 1 : 0;
    const int n_group = grouped ? h->n_groups : 0;

    if (ne > 0) {
        const int tiles = (ne + FLAT_TE - 1) / FLAT_TE;
        size_t shm = (size_t)c->smem_bytes;
        if (grouped) {
            cudaMemsetAsync(h->gS, 0, (size_t)n_group * sizeof(double),
                            c->stream);
            cudaMemsetAsync(h->gPb, 0, (size_t)n_group * sizeof(double),
                            c->stream);
            cudaMemsetAsync(h->gW, 0, (size_t)n_group * sizeof(double),
                            c->stream);
        }
        /* pass A (grouped) or the single fused pass (rs == 1 / no norm) */
        const int mode_a = grouped ? 1 : 0;
        flat_tile_kernel<<<tiles, FLAT_BLOCK, shm, c->stream>>>(
            c->fl_table, c->fl_q_index, c->num_r, c->num_i, c->num_m_index,
            c->bas_r, c->bas_i, c->bas_m_index, c->gamma,
            c->angle_index, c->angle_k, c->angle_b, c->am_r, c->am_i,
            c->m_min, c->m_delta, c->m_bins, c->q_min, c->q_delta, c->q_bins,
            c->p_ck_r, c->p_ck_i, c->p_coeff, norm, use_norm,
            mode_a, rs, h->gS, h->gPb, h->gW, NULL,
            h->mass, h->mom, h->ang, h->weight, h->bkg,
            ne, c->n_wave, c->n_proj, c->n_res, c->n_decay, c->n_basis,
            c->n_ab, c->n_comp, c->n_mass, c->n_momentum, c->n_angle_total,
            oP ? h->P_dev : NULL,
            c->g_ck_r, c->g_ck_i, c->g_coeff, c->acc);
        CUDA_CHECK(cudaGetLastError());

        if (grouped) {
            /* fold the group partials, then the per-event dQ/dP */
            flat_grp_nll_kernel<<<1, FLAT_BLOCK, 2 * FLAT_BLOCK * sizeof(double),
                                  c->stream>>>(
                h->gS, h->gPb, h->gW, n_group, norm, h->gscale, c->acc);
            CUDA_CHECK(cudaGetLastError());
            const int gblk = (ne + FLAT_BLOCK - 1) / FLAT_BLOCK;
            flat_scale_kernel<<<gblk, FLAT_BLOCK, 0, c->stream>>>(
                h->gscale, h->weight, norm, rs, ne, h->dqp);
            CUDA_CHECK(cudaGetLastError());
            /* pass B: recompute the forward quantities, now with the
               finished per-event dQ/dP, and run the gradient phases */
            flat_tile_kernel<<<tiles, FLAT_BLOCK, shm, c->stream>>>(
                c->fl_table, c->fl_q_index, c->num_r, c->num_i,
                c->num_m_index, c->bas_r, c->bas_i, c->bas_m_index, c->gamma,
                c->angle_index, c->angle_k, c->angle_b, c->am_r, c->am_i,
                c->m_min, c->m_delta, c->m_bins, c->q_min, c->q_delta,
                c->q_bins,
                c->p_ck_r, c->p_ck_i, c->p_coeff, norm, use_norm,
                2, rs, h->gS, h->gPb, h->gW, h->dqp,
                h->mass, h->mom, h->ang, h->weight, h->bkg,
                ne, c->n_wave, c->n_proj, c->n_res, c->n_decay, c->n_basis,
                c->n_ab, c->n_comp, c->n_mass, c->n_momentum,
                c->n_angle_total,
                NULL, c->g_ck_r, c->g_ck_i, c->g_coeff, c->acc);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    if (oP && h)
        cudaMemcpyAsync(h->h_P, h->P_dev, (size_t)ne * sizeof(double),
                        cudaMemcpyDeviceToHost, c->stream);
    cudaMemcpyAsync(c->h_acc, c->acc, 2 * sizeof(double),
                    cudaMemcpyDeviceToHost, c->stream);
    cudaMemcpyAsync(og_ck_r, c->g_ck_r, (size_t)N * sizeof(double),
                    cudaMemcpyDeviceToHost, c->stream);
    cudaMemcpyAsync(og_ck_i, c->g_ck_i, (size_t)N * sizeof(double),
                    cudaMemcpyDeviceToHost, c->stream);
    cudaMemcpyAsync(og_coeff, c->g_coeff, (size_t)NB * sizeof(double),
                    cudaMemcpyDeviceToHost, c->stream);
    CUDA_CHECK(cudaStreamSynchronize(c->stream));

    *oQ = c->h_acc[0];
    *oDn = c->h_acc[1];
    if (oP) memcpy(oP, h->h_P, (size_t)ne * sizeof(double));
}

int cuda_get_device_count_flat_pwa_gen(void) {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
}

void cuda_get_device_name_flat_pwa_gen(char* buf, int len) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    snprintf(buf, len, "%s", p.name);
}

}  /* extern "C" */
