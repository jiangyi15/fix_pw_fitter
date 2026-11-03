/* ══════════════════════════════════════════════════════════════════════
 * kernels_cpu_pwa_s0c.c — pure-PWA (projection-sum, shared ck) CPU kernel
 * on an OPAQUE basis.
 *
 * Model:  P(e) = Σ_p |A_p(e)|² ,  A_p(e) = Σ_k ck_k·common[e, p·N+k]
 *
 *   common[e,w] = amp[e,w] / BW_w(e)
 *   BW_w(e)     = Σ_{i<n_basis} coeff[w·n_basis + i] · F[w,i](e)
 *
 * The kernel knows NOTHING about what the coefficients mean.  Every slot
 * of a wave is one basis function, supplied as a per-event table value
 * Z (real/imag) through the basis-selection matrix M — which is 0/1 (one
 * basis function per slot, sparse one-hot) and is stored here as column
 * indices rather than a dense matrix:
 *
 *   M[w·n_basis + i] = u  ==  slot_map[w·n_basis + i]   (0/1 selection)
 *   F_r/F_i[e, u]                       (the basis FUNCTION at event e)
 *   coeff[w·n_basis + i]                (the caller's weight for it)
 *
 * The table holds the finished basis functions, so the kernel never
 * rotates anything (−i·γ, −m², 1 are produced once at load):
 *
 *   ∂Q/∂coeff[w,i] = 2·Re( ∂Q/∂BW_w · F[w,i] )
 *                  = 2·(Re(∂Q/∂BW_w)·F_r − Im(∂Q/∂BW_w)·F_i)
 *
 * so a slot is a slot: no slot index, no basis kind, no sign and no
 * physical parameter identity is special-cased anywhere in this file.
 * Note that a 0/1 selection says only WHICH basis function a slot uses —
 * it says nothing about the coefficient, which is an arbitrary number the
 * caller supplies (and which is fitted, not assumed).  Choosing the basis
 * and the coefficient meanings (s0c, for instance: F = 1, −m², −i·γ over
 * coefficients m0², 1, m0·g0) is entirely the caller's job — see
 * ``cpu/_pwa_s0c.py``.
 *
 * Iteration-invariant per-event inputs are precomputed once per data load
 * in Python and borrowed by the handle:
 *
 *   amp[e,w]  = fa(e)·fl_p(e)          (angular basis · form factors)
 *   F[e,u]    = basis function at the unique slot u
 *   weight, bkg, slot_map
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

/* ── data handle ─────────────────────────────────────────────────── */

typedef struct {
    int ne;             /* n events */
    int n_wave;         /* P·N entries per event */
    int N;              /* shared-ck length (n_wave / n_proj) */
    int n_proj;         /* projection count P */
    int n_basis;        /* basis slots per wave */
    int n_uslot;        /* unique basis slots per event */

    const double* amp_r;  /* (ne, n_wave) iteration-invariant amplitude */
    const double* amp_i;
    const double* Fr;     /* (ne, n_uslot) basis functions */
    const double* Fi;
    const int* slot_map;  /* (n_wave · n_basis) -> unique slot */
    const double* weight; /* (ne) */
    const double* bkg;    /* (ne) */
} PWAS0CData;

void* cpu_pwa_s0c_load(
    const double* amp_r, const double* amp_i,
    const double* F_r, const double* F_i, const int* slot_map,
    int n_uslot,
    const double* w, const double* bkg, int ne,
    int n_wave, int N, int n_proj, int n_basis)
{
    PWAS0CData* d = (PWAS0CData*)calloc(1, sizeof(PWAS0CData));
    if (!d) return NULL;
    d->ne = ne;
    d->n_wave = n_wave;
    d->N = N;
    d->n_proj = n_proj;
    d->n_basis = n_basis;
    d->n_uslot = n_uslot;

    d->amp_r = amp_r;
    d->amp_i = amp_i;
    d->Fr = F_r;
    d->Fi = F_i;
    d->slot_map = slot_map;
    d->weight = w;
    d->bkg = bkg;
    return (void*)d;
}

void cpu_pwa_s0c_free(void* h)
{
    if (h) free(h);
}



/* ── forward + backward (opaque basis) ───────────────────────────── */

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
    const int nu = d->n_uslot;
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
            const double* fr = d->Fr + (size_t)e * nu;
            const double* fi = d->Fi + (size_t)e * nu;
            const double* ar = d->amp_r + (size_t)e * nw;
            const double* ai = d->amp_i + (size_t)e * nw;
            const double we = d->weight[e];
            const double be = d->bkg[e];

            for (int w = 0; w < nw; w++) {
                const double* cw = coeff + (size_t)w * nf;
                const int* sm = d->slot_map + (size_t)w * nf;
                /* BW = Σ_i coeff_i · F_i  (F is stored ready-made) */
                double bwr = 0.0, bwi = 0.0;
                #pragma omp simd reduction(+:bwr,bwi)
                for (int i = 0; i < nf; i++) {
                    const double M = cw[i];
                    const int u = sm[i];
                    bwr += M * fr[u];
                    bwi += M * fi[u];
                }
                const double den = bwr * bwr + bwi * bwi;
                const double irw =  bwr / den;    /* 1/BW real */
                const double iiw = -bwi / den;    /* 1/BW imag */
                ir[w] = irw; ii[w] = iiw;
                const double amr = ar[w], ami = ai[w];
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
                const double dr = dQr[p] * ck_r[k] - dQi[p] * ck_i[k];
                const double di = dQr[p] * ck_i[k] + dQi[p] * ck_r[k];
                /* common/BW = common·(1/BW); d(common)/dBW = −common/BW */
                const double qr = cr[w] * ir[w] - ci[w] * ii[w];
                const double qi = cr[w] * ii[w] + ci[w] * ir[w];
                const double dBW_r = -(dr * qr - di * qi);
                const double dBW_i = -(dr * qi + di * qr);
                /* ∂Q/∂coeff_i = 2·Re(dBW·F_i) */
                const int* sm2 = d->slot_map + (size_t)w * nf;
                double* gc = gco + (size_t)w * nf;
                #pragma omp simd
                for (int i = 0; i < nf; i++) {
                    const int u = sm2[i];
                    gc[i] += 2.0 * (dBW_r * fr[u] - dBW_i * fi[u]);
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
