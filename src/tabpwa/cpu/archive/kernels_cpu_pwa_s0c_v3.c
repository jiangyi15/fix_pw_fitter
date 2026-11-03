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
    int n_basis;        /* uniform coefficient slots per wave */
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
    int n_wave, int N, int n_proj, int n_gamma_rows, int n_basis)
{
    PWAS0CData* d = (PWAS0CData*)calloc(1, sizeof(PWAS0CData));
    if (!d) return NULL;
    d->ne = ne;
    d->n_wave = n_wave;
    d->N = N;
    d->n_proj = n_proj;
    d->n_gamma_rows = n_gamma_rows;
    d->n_basis = n_basis;
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



/* ── forward + backward (uniform coeff; slot 0 = s0) ─────────────── */

double cpu_pwa_s0c_compute(
    void* h,
    const double* coeff,
    const double* ck_r, const double* ck_i,
    double norm, int use_norm,
    double* ogcoeff,
    double* ogck_r, double* ogck_i,
    double* P_out, double* dnorm_out)
{
    PWAS0CData* d = (PWAS0CData*)h;
    const int ne = d->ne;
    const int nw = d->n_wave;
    const int N  = d->N;
    const int P  = d->n_proj;
    const int ng = d->n_gamma_rows;
    const int nf = d->n_basis;
    const int nc = nw * nf;

    const int nt = omp_get_max_threads();
    const int padC = ((nc + 7) / 8) * 8;
    const int padN = ((N  + 7) / 8) * 8;
    double* tgc = (double*)calloc((size_t)nt * padC, sizeof(double));
    double* tg_r = (double*)calloc((size_t)nt * padN, sizeof(double));
    double* tg_i = (double*)calloc((size_t)nt * padN, sizeof(double));
    double* tQ  = (double*)calloc((size_t)nt, sizeof(double));
    double* tDn = (double*)calloc((size_t)nt, sizeof(double));
    if (!tgc || !tg_r || !tg_i || !tQ || !tDn) {
        free(tgc); free(tg_r); free(tg_i); free(tQ); free(tDn);
        return NAN;
    }

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double* gco = tgc + (size_t)tid * padC;
        double* gck_r = tg_r + (size_t)tid * padN;
        double* gck_i = tg_i + (size_t)tid * padN;

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
                const double* cw = coeff + (size_t)w * nf;
                double m2 = ms[d->mass_index_w[w]];
                m2 *= m2;
                int r0 = d->row_start[w];
                int r1 = (w + 1 < nw) ? d->row_start[w + 1] : ng;
                double gr = 0.0, gi = 0.0;
                #pragma omp simd reduction(+:gr,gi)
                for (int l = r0; l < r1; l++) {
                    double mm = cw[2 + (l - r0)];
                    gr += mm * zr[l];
                    gi += mm * zi[l];
                }
                /* BW = c0·1 + c1·m² + Σ_l c_{2+l}·(-i·Z_l) */
                double bwr = cw[0] + cw[1] * m2 + gi;
                double bwi = -gr;
                double den = bwr * bwr + bwi * bwi;
                double irw =  bwr / den;
                double iiw = -bwi / den;
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
                double dr = dQr[p] * ck_r[k] - dQi[p] * ck_i[k];
                double di = dQr[p] * ck_i[k] + dQi[p] * ck_r[k];
                double qr = cr[w] * ir[w] - ci[w] * ii[w];
                double qi = cr[w] * ii[w] + ci[w] * ir[w];
                double dBW_r = -(dr * qr - di * qi);
                double dBW_i = -(dr * qi + di * qr);
                double* gc = gco + (size_t)w * nf;
                double m2 = ms[d->mass_index_w[w]]; m2 *= m2;
                gc[0] += 2.0 * dBW_r;
                gc[1] += 2.0 * dBW_r * m2;
                int r0 = d->row_start[w];
                int r1 = (w + 1 < nw) ? d->row_start[w + 1] : ng;
                #pragma omp simd
                for (int l = r0; l < r1; l++) {
                    gc[2 + (l - r0)] += 2.0 * (dBW_r * zi[l] + dBW_i * zr[l]);
                }
            }

            if (P_out) P_out[e] = Pe;
        }

        tQ[tid] += Ql;
        tDn[tid] += dnl;
        free(ir); free(ii); free(cr); free(ci);
        free(Ar); free(Ai); free(dQr); free(dQi);
    }

    memset(ogcoeff, 0, (size_t)nc * sizeof(double));
    memset(ogck_r, 0, (size_t)N * sizeof(double));
    memset(ogck_i, 0, (size_t)N * sizeof(double));
    double Q_total = 0.0, dn_total = 0.0;
    for (int t = 0; t < nt; t++) {
        const double* g = tgc + (size_t)t * padC;
        for (int i = 0; i < nc; i++) ogcoeff[i] += g[i];
        const double* gr = tg_r + (size_t)t * padN;
        const double* gi = tg_i + (size_t)t * padN;
        for (int k = 0; k < N; k++) {
            ogck_r[k] += gr[k];
            ogck_i[k] += gi[k];
        }
        Q_total += tQ[t];
        dn_total += tDn[t];
    }
    free(tgc); free(tg_r); free(tg_i); free(tQ); free(tDn);
    if (dnorm_out) *dnorm_out = dn_total;
    return Q_total;
}
