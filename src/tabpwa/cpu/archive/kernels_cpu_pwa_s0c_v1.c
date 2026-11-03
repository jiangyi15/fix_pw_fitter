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

/* ── forward + backward ──────────────────────────────────────────── */

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

    const int n_threads = omp_get_max_threads();

    /* per-thread accumulators */
    double complex* t_gck  = (double complex*)calloc((size_t)n_threads * N,
                                                     sizeof(double complex));
    double* t_gs0   = (double*)calloc((size_t)n_threads * nw, sizeof(double));
    double* t_gmult = (double*)calloc((size_t)n_threads * ng, sizeof(double));
    double* t_Q     = (double*)calloc((size_t)n_threads, sizeof(double));
    double* t_dn    = (double*)calloc((size_t)n_threads, sizeof(double));
    if (!t_gck || !t_gs0 || !t_gmult || !t_Q || !t_dn) {
        free(t_gck); free(t_gs0); free(t_gmult); free(t_Q); free(t_dn);
        return NAN;
    }

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double complex* gck = t_gck + (size_t)tid * N;
        double* gs0 = t_gs0 + (size_t)tid * nw;
        double* gmult = t_gmult + (size_t)tid * ng;
        double Ql = 0.0;
        double dnl = 0.0;

        /* per-thread scratch (heap, one allocation per call) */
        double complex* invbw = (double complex*)malloc((size_t)nw *
                                                        sizeof(double complex));
        double complex* common = (double complex*)malloc((size_t)nw *
                                                         sizeof(double complex));
        double complex* A = (double complex*)malloc((size_t)P *
                                                    sizeof(double complex));
        double* dQdA_r = (double*)malloc((size_t)P * sizeof(double));
        double* dQdA_i = (double*)malloc((size_t)P * sizeof(double));

        #pragma omp for schedule(static)
        for (int e = 0; e < ne; e++) {
            const double* ms = d->mass + (size_t)e * d->n_mass;
            const double* zr = d->Zr + (size_t)e * ng;
            const double* zi = d->Zi + (size_t)e * ng;
            const double* ar = d->amp_r + (size_t)e * nw;
            const double* ai = d->amp_i + (size_t)e * nw;
            const double we = d->weight[e];
            const double be = d->bkg[e];

            /* ── BW denominators + common amplitudes ───────────────── */
            for (int w = 0; w < nw; w++) {
                double m = ms[d->mass_index_w[w]];
                int r0 = d->row_start[w];
                int r1 = (w + 1 < nw) ? d->row_start[w + 1] : ng;
                double complex g = 0.0;
                for (int l = r0; l < r1; l++)
                    g += mult[l] * (zr[l] + I * zi[l]);
                double complex inv = 1.0 / (s0[w] - m * m - I * g);
                invbw[w] = inv;
                /* same op order as numpy_pwa: (1/BW)·amp */
                common[w] = inv * (ar[w] + I * ai[w]);
            }

            /* ── projection sums A_p = Σ_k ck_k·common[p·N+k] ──────── */
            for (int p = 0; p < P; p++) {
                double complex acc = 0.0;
                const double complex* cp = common + (size_t)p * N;
                for (int k = 0; k < N; k++)
                    acc += (ck_r[k] + I * ck_i[k]) * cp[k];
                A[p] = acc;
            }

            double Pe = 0.0;
            for (int p = 0; p < P; p++)
                Pe += creal(A[p]) * creal(A[p]) + cimag(A[p]) * cimag(A[p]);

            /* ── objective ─────────────────────────────────────────── */
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

            /* ── back-prop through the projection sum ──────────────── */
            for (int p = 0; p < P; p++) {
                dQdA_r[p] = dQ_dP * creal(A[p]);
                dQdA_i[p] = -dQ_dP * cimag(A[p]);
            }

            /* grad_ck[k] += Σ_p dQ_dA[p]·common[p·N+k] */
            for (int p = 0; p < P; p++) {
                const double complex* cp = common + (size_t)p * N;
                const double dr = dQdA_r[p];
                const double di = dQdA_i[p];
                for (int k = 0; k < N; k++) {
                    double cr = creal(cp[k]);
                    double ci = cimag(cp[k]);
                    gck[k] += (dr * cr - di * ci) + I * (dr * ci + di * cr);
                }
            }

            /* dQ/ds0 and dQ/dmult through common = amp/BW:
               dQ/dBW = dQ/dc·(−common/BW),  dQ/dc = dQ_dA[p]·ck_k
               ∂BW/∂s0 = 1,  ∂BW/∂mult_l = −i·Z_l                    */
            for (int w = 0; w < nw; w++) {
                const int p = w / N;
                const int k = w - p * N;
                double complex dQdc =
                    (dQdA_r[p] + I * dQdA_i[p]) * (ck_r[k] + I * ck_i[k]);
                double complex dBW = dQdc * (-common[w] * invbw[w]);
                gs0[w] += 2.0 * creal(dBW);
                int r0 = d->row_start[w];
                int r1 = (w + 1 < nw) ? d->row_start[w + 1] : ng;
                for (int l = r0; l < r1; l++) {
                    /* dBW·(−i·Z_l) */
                    double complex t = dBW * (-I * (zr[l] + I * zi[l]));
                    gmult[l] += 2.0 * creal(t);
                }
            }

            if (P_out) P_out[e] = Pe;
        }

        /* thread-local slots: disjoint per tid, no synchronisation needed */
        t_Q[tid] += Ql;
        t_dn[tid] += dnl;

        free(invbw);
        free(common);
        free(A);
        free(dQdA_r);
        free(dQdA_i);
    }

    /* Reduce thread-local gradient arrays (each thread wrote its own
       slice, so the reduction is a plain host-side sum). */
    memset(ogck_r, 0, (size_t)N * sizeof(double));
    memset(ogck_i, 0, (size_t)N * sizeof(double));
    memset(ogs0, 0, (size_t)nw * sizeof(double));
    memset(ogmult, 0, (size_t)ng * sizeof(double));
    double Q_total = 0.0, dn_total = 0.0;
    for (int t = 0; t < n_threads; t++) {
        double complex* gck = t_gck + (size_t)t * N;
        for (int k = 0; k < N; k++) {
            ogck_r[k] += creal(gck[k]);
            ogck_i[k] += cimag(gck[k]);
        }
        double* gs0 = t_gs0 + (size_t)t * nw;
        for (int i = 0; i < nw; i++) ogs0[i] += gs0[i];
        double* gmult = t_gmult + (size_t)t * ng;
        for (int i = 0; i < ng; i++) ogmult[i] += gmult[i];
        Q_total += t_Q[t];
        dn_total += t_dn[t];
    }

    free(t_gck); free(t_gs0); free(t_gmult); free(t_Q); free(t_dn);
    if (dnorm_out) *dnorm_out = dn_total;
    return Q_total;
}
