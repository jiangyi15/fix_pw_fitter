/* ══════════════════════════════════════════════════════════════════════
 * kernels_cpu_pwa_s0c.c — pure-PWA (projection-sum, shared ck) CPU kernel
 * on an OPAQUE basis, evaluated EVENT-BLOCK / SOA.
 *
 * Model:  P(e) = Σ_p |A_p(e)|² ,  A_p(e) = Σ_k ck_k·common[e, p·N+k]
 *
 *   common[e,w] = amp[e,w] / BW_w(e)
 *   BW_w(e)     = Σ_{i<n_basis} coeff[w·n_basis + i] · F[w,i](e)
 *
 * The kernel knows NOTHING about what the coefficients mean.  Every slot
 * of a wave is one basis function, supplied as a per-event table through
 * the basis-selection matrix M — which is 0/1 (one basis function per
 * slot, sparse one-hot) and is stored here as column indices:
 *
 *   M[w·n_basis + i] = u  ==  slot_map[w·n_basis + i]
 *   F_r/F_i[u·ne + e]                   (basis function, EVENTS CONTIGUOUS)
 *   coeff[w·n_basis + i]                (the caller's weight for it)
 *
 * The parameters enter ONLY as ``coeff`` and leave ONLY as ``grad_coeff``
 * with the single uniform formula
 *
 *   ∂Q/∂coeff[w,i] = 2·Re( ∂Q/∂BW_w · F[w,i] )
 *                  = 2·(Re(∂Q/∂BW_w)·F_r − Im(∂Q/∂BW_w)·F_i)
 *
 * so a slot is a slot: no slot index, no basis kind, no sign and no
 * physical parameter identity is special-cased anywhere in this file.
 * A 0/1 selection says only WHICH basis function a slot uses — nothing
 * about the coefficient, which is an arbitrary (fitted) number supplied
 * by the caller.  Choosing the basis and the coefficient meanings (s0c,
 * for instance: F = 1, −m², −i·γ over coefficients m0², 1, m0·g0) is the
 * caller's job — see ``cpu/_pwa_s0c.py``.
 *
 * ── Event-block (SoA) evaluation ───────────────────────────────────
 * Every pass is vector over a tile of SOA_TILE consecutive events, so
 *
 *   * the 0/1 slot indirection is resolved ONCE per wave (not per event),
 *   * F[u][e0..e0+TE] and amp[w][e0..e0+TE] are contiguous loads,
 *   * the 1/BW division vectorizes (no serial latency chain),
 *   * the projection sum accumulates vectors: A_p[e] += ck_k·common[w][e].
 *
 * Tables are therefore stored EVENT-CONTIGUOUS:
 *   amp_r/amp_i  [n_wave · ne]     amp_r[w·ne + e]
 *   F_r/F_i      [n_uslot · ne]    F_r[u·ne + e]
 * Per-thread tile buffers: common, common/BW, A_p, dQ/dA, BW scratch.
 *
 * API (cffi, see cpu/_pwa_s0c.py):
 *   cpu_pwa_s0c_load()     — build a data handle (borrows the arrays)
 *   cpu_pwa_s0c_compute()  — forward NLL + gradients (OpenMP over tiles)
 *   cpu_pwa_s0c_free()     — release the handle
 * ══════════════════════════════════════════════════════════════════════ */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#ifndef SOA_TILE
#define SOA_TILE 64        /* events per tile */
#endif

/* ── data handle ─────────────────────────────────────────────────── */

typedef struct {
    int ne;             /* n events */
    int n_wave;         /* P·N entries per event */
    int N;              /* shared-ck length (n_wave / n_proj) */
    int n_proj;         /* projection count P */
    int n_basis;        /* basis slots per wave */
    int n_uslot;        /* unique basis slots per event */

    const double* amp_r;  /* [n_wave · ne]  event-contiguous */
    const double* amp_i;
    const double* Fr;     /* [n_uslot · ne] event-contiguous */
    const double* Fi;
    const int* slot_map;  /* [n_wave · n_basis] -> unique slot */
    const double* weight; /* [ne] */
    const double* bkg;    /* [ne] */
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



/* ── forward + backward, event-block SoA ─────────────────────────── */

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
    const int TE = SOA_TILE;
    const int ntiles = (ne + TE - 1) / TE;

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
    (void)nu;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double* gco  = tgc  + (size_t)tid * padC;
        double* gckr = tg_r + (size_t)tid * padN;
        double* gcki = tg_i + (size_t)tid * padN;
        double Ql = 0.0, dnl = 0.0;

        /* per-thread tile buffers (allocated once, reused by every tile) */
        const size_t nwT = (size_t)nw * TE;
        const size_t pT  = (size_t)P * TE;
        double* cr   = (double*)malloc(nwT * sizeof(double)); /* common   */
        double* ci   = (double*)malloc(nwT * sizeof(double));
        double* qr   = (double*)malloc(nwT * sizeof(double)); /* common/BW*/
        double* qi   = (double*)malloc(nwT * sizeof(double));
        double* bwr  = (double*)malloc((size_t)TE * sizeof(double));
        double* bwi  = (double*)malloc((size_t)TE * sizeof(double));
        double* Ar   = (double*)malloc(pT * sizeof(double));
        double* Ai   = (double*)malloc(pT * sizeof(double));
        double* dQr  = (double*)malloc(pT * sizeof(double));
        double* dQi  = (double*)malloc(pT * sizeof(double));
        double* Pe_b = (double*)malloc((size_t)TE * sizeof(double));

        #pragma omp for schedule(static)
        for (int tile = 0; tile < ntiles; tile++) {
            const int e0 = tile * TE;
            const int nen = (ne - e0 > TE) ? TE : (ne - e0);

            /* ── BW, 1/BW, common = amp/BW, common/BW ──────────────── */
            for (int w = 0; w < nw; w++) {
                const double* cwp = coeff + (size_t)w * nf;
                const int* sm = d->slot_map + (size_t)w * nf;
                #pragma omp simd
                for (int b = 0; b < nen; b++) { bwr[b] = 0.0; bwi[b] = 0.0; }
                for (int i = 0; i < nf; i++) {
                    const double M = cwp[i];
                    const int u = sm[i];
                    const double* fr = d->Fr + (size_t)u * ne + e0;
                    const double* fi = d->Fi + (size_t)u * ne + e0;
                    #pragma omp simd
                    for (int b = 0; b < nen; b++) {
                        bwr[b] += M * fr[b];
                        bwi[b] += M * fi[b];
                    }
                }
                const double* amr = d->amp_r + (size_t)w * ne + e0;
                const double* ami = d->amp_i + (size_t)w * ne + e0;
                double* pr = cr + (size_t)w * TE;
                double* pi = ci + (size_t)w * TE;
                double* xr = qr + (size_t)w * TE;
                double* xi = qi + (size_t)w * TE;
                #pragma omp simd
                for (int b = 0; b < nen; b++) {
                    const double br = bwr[b], bi = bwi[b];
                    const double den = br * br + bi * bi;
                    const double ir =  br / den;          /* 1/BW real */
                    const double ii = -bi / den;          /* 1/BW imag */
                    const double ar = amr[b], ai = ami[b];
                    const double m_r = ar * ir - ai * ii; /* common    */
                    const double m_i = ar * ii + ai * ir;
                    pr[b] = m_r; pi[b] = m_i;
                    xr[b] = m_r * ir - m_i * ii;          /* common/BW */
                    xi[b] = m_r * ii + m_i * ir;
                }
            }

            /* ── projection sums: A_p[e] += ck_k·common[p·N+k][e] ──── */
            for (int p = 0; p < P; p++) {
                #pragma omp simd
                for (int b = 0; b < nen; b++) {
                    Ar[p * TE + b] = 0.0;
                    Ai[p * TE + b] = 0.0;
                }
            }
            for (int k = 0; k < N; k++) {
                const double ckr = ck_r[k], cki = ck_i[k];
                for (int p = 0; p < P; p++) {
                    const int w = p * N + k;
                    const double* pr = cr + (size_t)w * TE;
                    const double* pi = ci + (size_t)w * TE;
                    double* ar = Ar + p * TE;
                    double* ai = Ai + p * TE;
                    #pragma omp simd
                    for (int b = 0; b < nen; b++) {
                        ar[b] += ckr * pr[b] - cki * pi[b];
                        ai[b] += ckr * pi[b] + cki * pr[b];
                    }
                }
            }

            /* ── P, Q, dQ/dP, dQ/dA ────────────────────────────────── */
            const double* wgt = d->weight + e0;
            const double* bkg = d->bkg + e0;
            if (use_norm) {
                #pragma omp simd reduction(+:Ql,dnl)
                for (int b = 0; b < nen; b++) {
                    double Pe = 0.0;
                    for (int p = 0; p < P; p++)
                        Pe += Ar[p * TE + b] * Ar[p * TE + b]
                            + Ai[p * TE + b] * Ai[p * TE + b];
                    const double ratio = Pe / norm;
                    const double denom = ratio + bkg[b];
                    Ql -= wgt[b] * log(denom);
                    const double dQ_dP = -wgt[b] / (norm * denom);
                    dnl += wgt[b] * (Pe / (norm * norm)) / denom;
                    Pe_b[b] = Pe;
                    for (int p = 0; p < P; p++) {
                        dQr[p * TE + b] =  dQ_dP * Ar[p * TE + b];
                        dQi[p * TE + b] = -dQ_dP * Ai[p * TE + b];
                    }
                }
            } else {
                #pragma omp simd reduction(+:Ql)
                for (int b = 0; b < nen; b++) {
                    double Pe = 0.0;
                    for (int p = 0; p < P; p++)
                        Pe += Ar[p * TE + b] * Ar[p * TE + b]
                            + Ai[p * TE + b] * Ai[p * TE + b];
                    const double dQ_dP = wgt[b];
                    Ql += wgt[b] * Pe;
                    Pe_b[b] = Pe;
                    for (int p = 0; p < P; p++) {
                        dQr[p * TE + b] =  dQ_dP * Ar[p * TE + b];
                        dQi[p * TE + b] = -dQ_dP * Ai[p * TE + b];
                    }
                }
            }
            if (P_out) {
                #pragma omp simd
                for (int b = 0; b < nen; b++) P_out[e0 + b] = Pe_b[b];
            }

            /* ── ck gradient: Σ_p Σ_e dQ/dA_p·common[p·N+k] ────────── */
            for (int k = 0; k < N; k++) {
                double sr = 0.0, si = 0.0;
                for (int p = 0; p < P; p++) {
                    const int w = p * N + k;
                    const double* dr = dQr + p * TE;
                    const double* di = dQi + p * TE;
                    const double* pr = cr + (size_t)w * TE;
                    const double* pi = ci + (size_t)w * TE;
                    #pragma omp simd reduction(+:sr,si)
                    for (int b = 0; b < nen; b++) {
                        sr += dr[b] * pr[b] - di[b] * pi[b];
                        si += dr[b] * pi[b] + di[b] * pr[b];
                    }
                }
                gckr[k] += sr;
                gcki[k] += si;
            }

            /* ── coefficient gradient: 2·Re(dBW·F_i) ───────────────── */
            for (int w = 0; w < nw; w++) {
                const int p = w / N;
                const int k = w - p * N;
                const double ckr = ck_r[k], cki = ck_i[k];
                const double* dr = dQr + p * TE;
                const double* di = dQi + p * TE;
                const double* xr = qr + (size_t)w * TE;
                const double* xi = qi + (size_t)w * TE;
                #pragma omp simd
                for (int b = 0; b < nen; b++) {
                    /* dQ/dcommon = dQ/dA_p·ck_k */
                    const double ur = dr[b] * ckr - di[b] * cki;
                    const double ui = dr[b] * cki + di[b] * ckr;
                    /* dBW = −(dQ/dcommon)·(common/BW) */
                    bwr[b] = -(ur * xr[b] - ui * xi[b]);
                    bwi[b] = -(ur * xi[b] + ui * xr[b]);
                }
                const int* sm = d->slot_map + (size_t)w * nf;
                double* gc = gco + (size_t)w * nf;
                for (int i = 0; i < nf; i++) {
                    const int u = sm[i];
                    const double* fr = d->Fr + (size_t)u * ne + e0;
                    const double* fi = d->Fi + (size_t)u * ne + e0;
                    double acc = 0.0;
                    #pragma omp simd reduction(+:acc)
                    for (int b = 0; b < nen; b++)
                        acc += 2.0 * (bwr[b] * fr[b] - bwi[b] * fi[b]);
                    gc[i] += acc;
                }
            }
        }

        tQ[tid] += Ql;
        tDn[tid] += dnl;
        free(cr); free(ci); free(qr); free(qi);
        free(bwr); free(bwi); free(Ar); free(Ai); free(dQr); free(dQi);
        free(Pe_b);
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
