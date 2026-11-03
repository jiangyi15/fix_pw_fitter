/* ══════════════════════════════════════════════════════════════════════
 * kernels_cpu_pwa_gen.c — pure-PWA (projection-sum, shared ck) CPU kernel
 * on an OPAQUE basis with ANY number of Breit–Wigner factors per wave.
 *
 * CPU counterpart of ``cuda_v4_pwa`` (the general n_res model).  Case-1
 * copy of ``kernels_cpu_pwa_s0c.c``: same event-tile SoA structure, same
 * opaque-basis design, same padded per-thread gradient slices; only the
 * per-wave BW factor dimension is added.
 *
 * Model:  P(e) = Σ_p |A_p(e)|² ,  A_p(e) = Σ_k ck_k·common[e, p·N+k]
 *
 *   bw_{w,r}(e) = Σ_i coeff[(w·n_res + r)·n_basis + i]·F[slot_map[… i]](e)
 *   BW_w(e)     = Π_r bw_{w,r}(e)
 *   invBW_w     = Π_r (1/bw_{w,r})          (only per-factor divisions)
 *   common_w    = amp_w · invBW_w
 *
 * The kernel knows NOTHING about what the coefficients mean.  Every
 * (wave, factor) carries ``n_basis`` slots, each one basis function,
 * supplied as a per-event table through the one-hot selection matrix M
 * (0/1, one basis function per slot) stored here as column indices:
 *
 *   M[(w·n_res+r)·n_basis + i] = u  ==  slot_map[(w·n_res+r)·n_basis + i]
 *   F_r/F_i[u·ne + e]                   (basis function, EVENTS CONTIGUOUS)
 *   coeff[(w·n_res+r)·n_basis + i]      (the caller's weight for it)
 *
 * The parameters enter ONLY as ``coeff`` and leave ONLY as ``grad_coeff``
 * with the single uniform formula
 *
 *   ∂Q/∂coeff[(w,r,i)] = 2·Re( dBW_w · (1/bw_{w,r}) · F_i(e) )
 *   dBW_w              = -(dQ/dcommon_w)·common_w
 *
 * so a slot is a slot: no slot index, no basis kind, no sign and no
 * physical parameter identity is special-cased anywhere in this file.
 * At ``n_res == 1`` the file reproduces ``kernels_cpu_pwa_s0c.c`` bit for
 * bit (1·x − 0·y == x, and the single-factor product folds exactly).
 *
 * ── Event-block (SoA) evaluation ───────────────────────────────────
 * Every pass is vector over a tile of TE consecutive events, so
 *
 *   * the 0/1 slot indirection is resolved ONCE per (wave, factor),
 *   * F[u][e0..e0+TE] and amp[w][e0..e0+TE] are contiguous loads,
 *   * the 1/bw divisions vectorize (no serial latency chain),
 *   * the projection sum accumulates vectors: A_p[e] += ck_k·common[w][e].
 *
 * Tables are therefore stored EVENT-CONTIGUOUS:
 *   amp_r/amp_i  [n_wave · ne]        amp_r[w·ne + e]
 *   F_r/F_i      [n_uslot · ne]       F_r[u·ne + e]
 * Per-thread tile buffers (heap, allocated once per thread): common,
 * 1/bw per (wave, factor, event) — layout [w][r][e], event-contiguous so
 * the vector loops stay contiguous — A_p, dQ/dA, BW scratch.  At n_res==1
 * the tile holds common·(1/BW) instead (the gradient phase then needs no
 * second multiply; bit-identical to the s0c kernel, same loop shape).
 *
 * TE is derived from the model (one knob each, both default 64):
 *   n_res == 1  -> SOA_TILE       (the s0c tile — bit-identical tiling)
 *   n_res >= 2  -> SOA_TILE_MULTI (measured: 64 beats 32 by ~1.2x on a
 *                200-wave model and ~1.3x on a 1000-wave one; 128 is
 *                worse again — the tile is streamed, so the per-thread
 *                footprint n_wave·n_res·TE·16 B (405 KB at nw=198,
 *                n_res=2, TE=64) only costs when it leaves L2/LLC, and a
 *                bigger tile amortizes the per-tile gathers).
 *
 * Compile with -DPWA_GEN_RECOMPUTE_INV to benchmark the alternative
 * gradient phase (recompute bw_{w,r} instead of storing 1/bw): measured
 * 1.3-1.5x SLOWER on the wave-dense multi-BW model, so the stored tile
 * is the default.
 *
 * API (cffi, see cpu/_pwa_gen.py):
 *   cpu_pwa_gen_load()     — build a data handle (borrows the arrays)
 *   cpu_pwa_gen_compute()  — forward NLL + gradients (OpenMP over tiles)
 *   cpu_pwa_gen_free()     — release the handle
 * ══════════════════════════════════════════════════════════════════════ */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#ifndef SOA_TILE
#define SOA_TILE 64        /* events per tile */
#endif
#ifndef SOA_TILE_MULTI
#define SOA_TILE_MULTI 64  /* events per tile, n_res >= 2 (own knob) */
#endif

/* ── data handle ─────────────────────────────────────────────────── */

typedef struct {
    int ne;             /* n events */
    int n_wave;         /* P·N entries per event */
    int N;              /* shared-ck length (n_wave / n_proj) */
    int n_proj;         /* projection count P */
    int n_res;          /* BW factors per wave */
    int n_basis;        /* basis slots per (wave, factor) */
    int n_uslot;        /* unique basis slots per event */
    int n_fact;         /* n_wave · n_res — (wave, factor) entries */

    const double* amp_r;  /* [n_wave · ne]  event-contiguous */
    const double* amp_i;
    const double* Fr;     /* [n_uslot · ne] event-contiguous */
    const double* Fi;
    const int* slot_map;  /* [n_fact · n_basis] -> unique slot */
    const double* weight; /* [ne] */
    const double* bkg;    /* [ne] */
} PWAGenData;

void* cpu_pwa_gen_load(
    const double* amp_r, const double* amp_i,
    const double* F_r, const double* F_i, const int* slot_map,
    int n_uslot,
    const double* w, const double* bkg, int ne,
    int n_wave, int N, int n_proj, int n_res, int n_basis)
{
    PWAGenData* d = (PWAGenData*)calloc(1, sizeof(PWAGenData));
    if (!d) return NULL;
    d->ne = ne;
    d->n_wave = n_wave;
    d->N = N;
    d->n_proj = n_proj;
    d->n_res = n_res;
    d->n_basis = n_basis;
    d->n_uslot = n_uslot;
    d->n_fact = n_wave * n_res;

    d->amp_r = amp_r;
    d->amp_i = amp_i;
    d->Fr = F_r;
    d->Fi = F_i;
    d->slot_map = slot_map;
    d->weight = w;
    d->bkg = bkg;
    return (void*)d;
}

void cpu_pwa_gen_free(void* h)
{
    if (h) free(h);
}



/* ── one (wave, factor) proportional denominator over a tile ─────── */
/* bw_{w,r}(e) = Σ_i coeff[f·nf + i]·F[slot_map[f·nf + i]](e), f = w·n_res+r.
 * One basis pass; the shared-cache tile is left untouched. */
static inline void pwa_gen_factor(const double* coeff, const int* slot_map,
                                  int nf, const double* Fr, const double* Fi,
                                  int ne, int e0, int nen, int f,
                                  double* bwr, double* bwi)
{
    const double* cwp = coeff + (size_t)f * nf;
    const int* sm = slot_map + (size_t)f * nf;
    #pragma omp simd
    for (int b = 0; b < nen; b++) { bwr[b] = 0.0; bwi[b] = 0.0; }
    for (int i = 0; i < nf; i++) {
        const double M = cwp[i];
        const int u = sm[i];
        const double* fr = Fr + (size_t)u * ne + e0;
        const double* fi = Fi + (size_t)u * ne + e0;
        #pragma omp simd
        for (int b = 0; b < nen; b++) {
            bwr[b] += M * fr[b];
            bwi[b] += M * fi[b];
        }
    }
}

/* ── forward + backward, event-block SoA ─────────────────────────── */

double cpu_pwa_gen_compute(
    void* h,
    const double* coeff,
    const double* ck_r, const double* ck_i,
    double norm, int use_norm,
    double* ogcoeff,
    double* ogck_r, double* ogck_i,
    double* P_out, double* dnorm_out)
{
    PWAGenData* d = (PWAGenData*)h;
    const int ne = d->ne;
    const int nw = d->n_wave;
    const int N  = d->N;
    const int P  = d->n_proj;
    const int nr = d->n_res;
    const int nu = d->n_uslot;
    const int nf = d->n_basis;
    const int nfact = d->n_fact;
    const int nc = nfact * nf;
    const int TE = (nr <= 1) ? SOA_TILE : SOA_TILE_MULTI;
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
        const size_t nfT = (size_t)nfact * TE;
        const size_t pT  = (size_t)P * TE;
        double* cr   = (double*)malloc(nwT * sizeof(double)); /* common     */
        double* ci   = (double*)malloc(nwT * sizeof(double));
        double* ir   = (double*)malloc(nfT * sizeof(double)); /* per-factor
                                        tile: 1/bw_{w,r}   (n_res >= 2)
                                        or common/BW_w    (n_res == 1) */
        double* ii   = (double*)malloc(nfT * sizeof(double));
        double* bwr  = (double*)malloc((size_t)TE * sizeof(double));
        double* bwi  = (double*)malloc((size_t)TE * sizeof(double));
        double* ur   = (double*)malloc((size_t)TE * sizeof(double));
        double* ui   = (double*)malloc((size_t)TE * sizeof(double));
        double* dr   = (double*)malloc((size_t)TE * sizeof(double));
        double* di   = (double*)malloc((size_t)TE * sizeof(double));
        double* Ar   = (double*)malloc(pT * sizeof(double));
        double* Ai   = (double*)malloc(pT * sizeof(double));
        double* dQr  = (double*)malloc(pT * sizeof(double));
        double* dQi  = (double*)malloc(pT * sizeof(double));
        double* Pe_b = (double*)malloc((size_t)TE * sizeof(double));

        #pragma omp for schedule(static)
        for (int tile = 0; tile < ntiles; tile++) {
            const int e0 = tile * TE;
            const int nen = (ne - e0 > TE) ? TE : (ne - e0);

            /* ── per-factor 1/bw, then BW product, common = amp/BW ─── */
            for (int w = 0; w < nw; w++) {
                const double* amr = d->amp_r + (size_t)w * ne + e0;
                const double* ami = d->amp_i + (size_t)w * ne + e0;
                double* pr = cr + (size_t)w * TE;
                double* pi = ci + (size_t)w * TE;
#ifdef PWA_GEN_RECOMPUTE_INV
                /* EXPERIMENT (benchmark only, default OFF): no per-factor
                 * 1/bw tile — the product folds as the factors are built
                 * and the gradient phase recomputes bw_{w,r} from the
                 * basis (one extra basis pass + 2 divisions per factor).
                 * Compile with -DPWA_GEN_RECOMPUTE_INV to compare; the
                 * default stores the tile (measured faster). */
                #pragma omp simd
                for (int b = 0; b < nen; b++) { dr[b] = 1.0; di[b] = 0.0; }
                for (int r = 0; r < nr; r++) {
                    const int f = w * nr + r;
                    pwa_gen_factor(coeff, d->slot_map, nf, d->Fr, d->Fi,
                                   ne, e0, nen, f, bwr, bwi);
                    #pragma omp simd
                    for (int b = 0; b < nen; b++) {
                        const double den = bwr[b] * bwr[b] + bwi[b] * bwi[b];
                        const double irr =  bwr[b] / den;
                        const double iii = -bwi[b] / den;
                        const double ar2 = dr[b], ai2 = di[b];
                        dr[b] = ar2 * irr - ai2 * iii;
                        di[b] = ar2 * iii + ai2 * irr;
                    }
                }
                #pragma omp simd
                for (int b = 0; b < nen; b++) {
                    const double xrr = dr[b], xii = di[b];   /* invBW */
                    const double ar = amr[b], ai = ami[b];
                    pr[b] = ar * xrr - ai * xii;
                    pi[b] = ar * xii + ai * xrr;
                }
#else
                /* the per-factor 1/bw tile is filled once and reused by the
                 * gradient phase.  At nr == 1 the single factor IS the
                 * propagator: the fold and the common product ride along
                 * in the same pass (no ring of multiply-by-1, exactly the
                 * s0c kernel's loop shape → bit-identical). */
                for (int r = 0; r < nr; r++) {
                    const int f = w * nr + r;
                    pwa_gen_factor(coeff, d->slot_map, nf, d->Fr, d->Fi,
                                   ne, e0, nen, f, bwr, bwi);
                    double* xr = ir + (size_t)f * TE;
                    double* xi = ii + (size_t)f * TE;
                    if (nr == 1) {
                        /* the single factor IS the propagator: the tile
                         * keeps common·(1/BW) (the gradient phase needs no
                         * second multiply), and the common product rides
                         * along in the same pass — exactly the s0c loop
                         * shape → bit-identical. */
                        #pragma omp simd
                        for (int b = 0; b < nen; b++) {
                            const double br = bwr[b], bi = bwi[b];
                            const double den = br * br + bi * bi;
                            const double irr =  br / den;   /* 1/BW real */
                            const double iii = -bi / den;   /* 1/BW imag */
                            const double ar = amr[b], ai = ami[b];
                            const double m_r = ar * irr - ai * iii;
                            const double m_i = ar * iii + ai * irr;
                            pr[b] = m_r; pi[b] = m_i;       /* common */
                            xr[b] = m_r * irr - m_i * iii;  /* common/BW */
                            xi[b] = m_r * iii + m_i * irr;
                        }
                    } else {
                        #pragma omp simd
                        for (int b = 0; b < nen; b++) {
                            const double br = bwr[b], bi = bwi[b];
                            const double den = br * br + bi * bi;
                            xr[b] =  br / den;              /* 1/bw real */
                            xi[b] = -bi / den;              /* 1/bw imag */
                        }
                    }
                }
                if (nr == 1) continue;
                /* invBW = Π_r 1/bw_r */
                #pragma omp simd
                for (int b = 0; b < nen; b++) { bwr[b] = 1.0; bwi[b] = 0.0; }
                for (int r = 0; r < nr; r++) {
                    const int f = w * nr + r;
                    const double* xr = ir + (size_t)f * TE;
                    const double* xi = ii + (size_t)f * TE;
                    #pragma omp simd
                    for (int b = 0; b < nen; b++) {
                        const double ar2 = bwr[b], ai2 = bwi[b];
                        bwr[b] = ar2 * xr[b] - ai2 * xi[b];
                        bwi[b] = ar2 * xi[b] + ai2 * xr[b];
                    }
                }
                #pragma omp simd
                for (int b = 0; b < nen; b++) {
                    const double xrr = bwr[b], xii = bwi[b];
                    const double ar = amr[b], ai = ami[b];
                    pr[b] = ar * xrr - ai * xii;      /* common = amp/BW */
                    pi[b] = ar * xii + ai * xrr;
                }
#endif
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
                    const double* xr = dQr + p * TE;
                    const double* xi = dQi + p * TE;
                    const double* pr = cr + (size_t)w * TE;
                    const double* pi = ci + (size_t)w * TE;
                    #pragma omp simd reduction(+:sr,si)
                    for (int b = 0; b < nen; b++) {
                        sr += xr[b] * pr[b] - xi[b] * pi[b];
                        si += xr[b] * pi[b] + xi[b] * pr[b];
                    }
                }
                gckr[k] += sr;
                gcki[k] += si;
            }

            /* ── coefficient gradient: 2·Re(dBW·(1/bw_r)·F_i) ─────── */
            for (int w = 0; w < nw; w++) {
                const int p = w / N;
                const int k = w - p * N;
                const double ckr = ck_r[k], cki = ck_i[k];
                const double* qr_ = dQr + p * TE;
                const double* qi_ = dQi + p * TE;
                const double* pr = cr + (size_t)w * TE;
                const double* pi = ci + (size_t)w * TE;
                #pragma omp simd
                for (int b = 0; b < nen; b++) {
                    /* dQ/dcommon = dQ/dA_p·ck_k */
                    ur[b] = qr_[b] * ckr - qi_[b] * cki;
                    ui[b] = qr_[b] * cki + qi_[b] * ckr;
                }
                for (int r = 0; r < nr; r++) {
                    const int f = w * nr + r;
#ifdef PWA_GEN_RECOMPUTE_INV
                    /* EXPERIMENT (benchmark only, default OFF): recompute
                     * bw_{w,r} in the gradient phase instead of reading the
                     * stored 1/bw tile — one extra basis pass + 2 divisions
                     * per (wave, factor, event).  Compile with
                     * -DPWA_GEN_RECOMPUTE_INV to compare; the default
                     * keeps the stored tile (measured faster). */
                    pwa_gen_factor(coeff, d->slot_map, nf, d->Fr, d->Fi,
                                   ne, e0, nen, f, bwr, bwi);
                    #pragma omp simd
                    for (int b = 0; b < nen; b++) {
                        const double den = bwr[b] * bwr[b]
                                         + bwi[b] * bwi[b];
                        const double irr =  bwr[b] / den;
                        const double iii = -bwi[b] / den;
                        const double m_r = pr[b] * irr - pi[b] * iii;
                        const double m_i = pr[b] * iii + pi[b] * irr;
                        dr[b] = -(ur[b] * m_r - ui[b] * m_i);
                        di[b] = -(ur[b] * m_i + ui[b] * m_r);
                    }
#else
                    const double* xr = ir + (size_t)f * TE;
                    const double* xi = ii + (size_t)f * TE;
                    if (nr == 1) {
                        /* tile holds common·(1/BW): dBW = −(dQ/dcommon)·tile */
                        #pragma omp simd
                        for (int b = 0; b < nen; b++) {
                            dr[b] = -(ur[b] * xr[b] - ui[b] * xi[b]);
                            di[b] = -(ur[b] * xi[b] + ui[b] * xr[b]);
                        }
                    } else {
                        #pragma omp simd
                        for (int b = 0; b < nen; b++) {
                            /* dBW = −(dQ/dcommon)·common·(1/bw_r) */
                            const double m_r = pr[b] * xr[b] - pi[b] * xi[b];
                            const double m_i = pr[b] * xi[b] + pi[b] * xr[b];
                            dr[b] = -(ur[b] * m_r - ui[b] * m_i);
                            di[b] = -(ur[b] * m_i + ui[b] * m_r);
                        }
                    }
#endif
                    const int* sm = d->slot_map + (size_t)f * nf;
                    double* gc = gco + (size_t)f * nf;
                    for (int i = 0; i < nf; i++) {
                        const int u = sm[i];
                        const double* fr = d->Fr + (size_t)u * ne + e0;
                        const double* fi = d->Fi + (size_t)u * ne + e0;
                        double acc = 0.0;
                        #pragma omp simd reduction(+:acc)
                        for (int b = 0; b < nen; b++)
                            acc += 2.0 * (dr[b] * fr[b] - di[b] * fi[b]);
                        gc[i] += acc;
                    }
                }
            }
        }

        tQ[tid] += Ql;
        tDn[tid] += dnl;
        free(cr); free(ci); free(ir); free(ii);
        free(bwr); free(bwi); free(ur); free(ui); free(dr); free(di);
        free(Ar); free(Ai); free(dQr); free(dQi);
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
