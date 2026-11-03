/* ══════════════════════════════════════════════════════════════════════
 * kernels_cpu_pwa_s0c.c — pure-PWA (projection-sum, shared ck) CPU kernel
 * with the "s0c" parameterisation (direct pole parameters).
 *
 * Model:  P(e) = Σ_p |A_p(e)|² ,  A_p(e) = Σ_k ck_k·common[e, p·N+k]
 *
 *   common[e,w] = amp[e,w] / BW_w(e)
 *   BW_w(e)     = s0_w − m_w(e)² − i·Σ_{l∈rows(w)} mult_l·Z[e,l]
 *
 * with the s0c outside transform
 *
 *   s0_w  = m0_w² ,     mult_l = m0_{w(l)}·g0_l
 *
 * (`m0`, `g0` live on the host; the C code works purely in s0/mult and
 * returns dQ/ds0 and dQ/dmult — the wrapper chains to m0/g0, exactly like
 * `cuda_v4_pwa_s0c`).
 *
 * Iteration-invariant inputs (the τ-u-cache analogue) are precomputed once
 * per data load in Python and stored on the handle:
 *
 *   amp[e,w]  = fa(e)·fl_p(e)          (angular basis · form factors)
 *   Z[e,l]    = Catmull-Rom γ-table interp at mass[e, g0_mass_index_w[l]]
 *   mass, weight, bkg, and the derived index arrays
 *
 * API (cffi, see cpu/_pwa_s0c.py):
 *   cpu_pwa_s0c_load()     — build a data handle (borrows the arrays)
 *   cpu_pwa_s0c_compute()  — forward NLL + gradients (OpenMP over events)
 *   cpu_pwa_s0c_free()     — release the handle
 *
 * Parallelisation: OpenMP over events with per-thread accumulators,
 * reduced at the end (same pattern as kernels_cpu_v3.c).
 * ══════════════════════════════════════════════════════════════════════ */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <complex.h>

/* ── data handle ─────────────────────────────────────────────────── */

typedef struct {
    int ne;             /* n events */
    int n_wave;         /* P·N entries per event */
    int N;              /* shared-ck length (n_wave / n_proj) */
    int n_proj;         /* projection count P */
    int n_gamma_rows;   /* merged (wave, γ-row) entries */
    int n_mass;         /* mass columns */

    const double* mass;   /* (ne, n_mass)               */
    const double* amp_r;  /* (ne, n_wave) iteration-invariant amplitude */
    const double* amp_i;
    const double* Zr;     /* (ne, n_gamma_rows) γ-interp, real */
    const double* Zi;     /*                     imag */
    const double* weight; /* (ne) */
    const double* bkg;    /* (ne) */

    /* p-major dedup expansion: one BW slot per wave, γ rows grouped
       wave-major (row_start gives each wave's half-open row range). */
    const int* m0_index_w;    /* (n_wave) wave -> m0 param slot (host only) */
    const int* mass_index_w;  /* (n_wave) wave -> mass column */
    const int* row_start;     /* (n_wave) first γ row of each wave */
} PWAS0CData;

void* cpu_pwa_s0c_load(
    const double* mass, int n_mass,
    const double* amp_r, const double* amp_i,
    const double* Z_r, const double* Z_i,
    const double* w, const double* bkg, int ne,
    const int* m0_index_w,
    const int* mass_index_w,
    const int* row_start,
    int n_wave, int N, int n_proj, int n_gamma_rows)
{
    PWAS0CData* d = (PWAS0CData*)calloc(1, sizeof(PWAS0CData));
    if (!d) return NULL;
    d->ne = ne;
    d->n_wave = n_wave;
    d->N = N;
    d->n_proj = n_proj;
    d->n_gamma_rows = n_gamma_rows;
    d->n_mass = n_mass;

    d->mass = mass;
    d->amp_r = amp_r;
    d->amp_i = amp_i;
    d->Zr = Z_r;
    d->Zi = Z_i;
    d->weight = w;
    d->bkg = bkg;

    d->m0_index_w = m0_index_w;
    d->mass_index_w = mass_index_w;
    d->row_start = row_start;
    return (void*)d;
}

void cpu_pwa_s0c_free(void* h)
{
    if (h) free(h);
}


/* ── forward + backward (manual real/imag; no __divdc3) ─────────── */

double cpu_pwa_s0c_compute(
    void* h,
    const double* s0, const double* mult,
    const double* ck_r, const double* ck_i,
    double norm, int use_norm,
    double* ogck_r, double* ogck_i,
    double* ogs0, double* ogmult,
    double* P_out, double* dnorm_out)
{
    PWAS0CData* d = (PWAS0CData*)h;
    const int ne = d->ne;
    const int nw = d->n_wave;
    const int N  = d->N;
    const int P  = d->n_proj;
    const int ng = d->n_gamma_rows;

    const int nt = omp_get_max_threads();
    /* padded per-thread gradient slices (64-byte multiples) */
    const int padN = ((N  + 7) / 8) * 8;
    const int padW = ((nw + 7) / 8) * 8;
    const int padG = ((ng + 7) / 8) * 8;

    double* tg_r  = (double*)calloc((size_t)nt * padN, sizeof(double));
    double* tg_i  = (double*)calloc((size_t)nt * padN, sizeof(double));
    double* tgs0  = (double*)calloc((size_t)nt * padW, sizeof(double));
    double* tgm   = (double*)calloc((size_t)nt * padG, sizeof(double));
    double* tQ    = (double*)calloc((size_t)nt, sizeof(double));
    double* tDn   = (double*)calloc((size_t)nt, sizeof(double));
    if (!tg_r || !tg_i || !tgs0 || !tgm || !tQ || !tDn) {
        free(tg_r); free(tg_i); free(tgs0); free(tgm); free(tQ); free(tDn);
        return NAN;
    }

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double* gck_r = tg_r + (size_t)tid * padN;
        double* gck_i = tg_i + (size_t)tid * padN;
        double* gs0   = tgs0 + (size_t)tid * padW;
        double* gm    = tgm  + (size_t)tid * padG;

        double* ir = (double*)malloc((size_t)nw * sizeof(double));
        double* ii = (double*)malloc((size_t)nw * sizeof(double));
        double* cr = (double*)malloc((size_t)nw * sizeof(double));
        double* ci = (double*)malloc((size_t)nw * sizeof(double));
        double* Ar = (double*)malloc((size_t)P  * sizeof(double));
        double* Ai = (double*)malloc((size_t)P  * sizeof(double));
        double* dQr = (double*)malloc((size_t)P * sizeof(double));
        double* dQi = (double*)malloc((size_t)P * sizeof(double));
        double Ql = 0.0, dnl = 0.0;

        #pragma omp for schedule(static)
        for (int e = 0; e < ne; e++) {
            const double* ms = d->mass + (size_t)e * d->n_mass;
            const double* zr = d->Zr + (size_t)e * ng;
            const double* zi = d->Zi + (size_t)e * ng;
            const double* ar = d->amp_r + (size_t)e * nw;
            const double* ai = d->amp_i + (size_t)e * nw;
            const double we = d->weight[e];
            const double be = d->bkg[e];

            for (int w = 0; w < nw; w++) {
                double m = ms[d->mass_index_w[w]];
                int r0 = d->row_start[w];
                int r1 = (w + 1 < nw) ? d->row_start[w + 1] : ng;
                double gr = 0.0, gi = 0.0;
                #pragma omp simd reduction(+:gr,gi)
                for (int l = r0; l < r1; l++) {
                    gr += mult[l] * zr[l];
                    gi += mult[l] * zi[l];
                }
                /* BW = s0 - m^2 - i*g,  g = gr + i*gi
                   -> real = s0 - m^2 + gi,  imag = -gr */
                double a = s0[w] - m * m + gi;
                double b = -gr;
                double den = a * a + b * b;
                double irw =  a / den;                 /* 1/BW real */
                double iiw = -b / den;                 /* 1/BW imag */
                ir[w] = irw; ii[w] = iiw;
                double amr = ar[w], ami = ai[w];
                cr[w] = amr * irw - ami * iiw;
                ci[w] = amr * iiw + ami * irw;
            }

            for (int p = 0; p < P; p++) {
                double sr = 0.0, si = 0.0;
                const int base = p * N;
                #pragma omp simd reduction(+:sr,si)
                for (int k = 0; k < N; k++) {
                    double ckr = ck_r[k], cki = ck_i[k];
                    double cmr = cr[base + k], cmi = ci[base + k];
                    sr += ckr * cmr - cki * cmi;
                    si += ckr * cmi + cki * cmr;
                }
                Ar[p] = sr; Ai[p] = si;
            }

            double Pe = 0.0;
            for (int p = 0; p < P; p++)
                Pe += Ar[p] * Ar[p] + Ai[p] * Ai[p];

            double dQ_dP;
            if (use_norm) {
                double ratio = Pe / norm;
                Ql -= we * log(ratio + be);
                dQ_dP = -we / (norm * (ratio + be));
                dnl += we * (Pe / (norm * norm)) / (ratio + be);
            } else {
                Ql += we * Pe;
                dQ_dP = we;
            }

            for (int p = 0; p < P; p++) {
                dQr[p] =  dQ_dP * Ar[p];
                dQi[p] = -dQ_dP * Ai[p];
            }

            for (int p = 0; p < P; p++) {
                const int base = p * N;
                const double dr = dQr[p], di = dQi[p];
                #pragma omp simd
                for (int k = 0; k < N; k++) {
                    double cmr = cr[base + k], cmi = ci[base + k];
                    gck_r[k] += dr * cmr - di * cmi;
                    gck_i[k] += dr * cmi + di * cmr;
                }
            }

            for (int w = 0; w < nw; w++) {
                const int p = w / N;
                const int k = w - p * N;
                /* dQ/dc = dQdA[p]·ck[k] */
                double dr = dQr[p] * ck_r[k] - dQi[p] * ck_i[k];
                double di = dQr[p] * ck_i[k] + dQi[p] * ck_r[k];
                /* common/BW = common·(1/BW) */
                double qr = cr[w] * ir[w] - ci[w] * ii[w];
                double qi = cr[w] * ii[w] + ci[w] * ir[w];
                /* dBW = -(dQdc·(common/BW)) */
                double bwr = -(dr * qr - di * qi);
                double bwi = -(dr * qi + di * qr);
                gs0[w] += 2.0 * bwr;
                int r0 = d->row_start[w];
                int r1 = (w + 1 < nw) ? d->row_start[w + 1] : ng;
                #pragma omp simd
                for (int l = r0; l < r1; l++) {
                    /* 2·Re(dBW·(−i·Z)) = 2·(bwr·Zi + bwi·Zr) */
                    gm[l] += 2.0 * (bwr * zi[l] + bwi * zr[l]);
                }
            }

            if (P_out) P_out[e] = Pe;
        }

        tQ[tid] += Ql;
        tDn[tid] += dnl;
        free(ir); free(ii); free(cr); free(ci);
        free(Ar); free(Ai); free(dQr); free(dQi);
    }

    memset(ogck_r, 0, (size_t)N * sizeof(double));
    memset(ogck_i, 0, (size_t)N * sizeof(double));
    memset(ogs0, 0, (size_t)nw * sizeof(double));
    memset(ogmult, 0, (size_t)ng * sizeof(double));
    double Q_total = 0.0, dn_total = 0.0;
    for (int t = 0; t < nt; t++) {
        const double* g1 = tg_r + (size_t)t * padN;
        const double* g2 = tg_i + (size_t)t * padN;
        for (int k = 0; k < N; k++) { ogck_r[k] += g1[k]; ogck_i[k] += g2[k]; }
        const double* s1 = tgs0 + (size_t)t * padW;
        for (int i = 0; i < nw; i++) ogs0[i] += s1[i];
        const double* m1 = tgm + (size_t)t * padG;
        for (int i = 0; i < ng; i++) ogmult[i] += m1[i];
        Q_total += tQ[t];
        dn_total += tDn[t];
    }
    free(tg_r); free(tg_i); free(tgs0); free(tgm); free(tQ); free(tDn);
    if (dnorm_out) *dnorm_out = dn_total;
    return Q_total;
}
