/* ══════════════════════════════════════════════════════════════════════
 * kernels_cpu_pwa_v5.c — pure-PWA (projection-sum, shared ck) CPU kernel
 * on an OPAQUE basis with ANY number of Breit–Wigner factors per wave,
 * with the v5 **grouped log-sum** data NLL.
 *
 * CPU counterpart of ``cuda_v5_pwa`` — the general (any n_res)
 * projection-sum PWA of the v4 CPU kernel (``kernels_cpu_pwa_gen.c``,
 * now superseded + archived; itself the counterpart of ``cuda_v4_pwa``)
 * with ONE difference: in the data-NLL branch (``use_norm``) the per-event
 * log is replaced by ONE log per ``resolution_size``-sized group of
 * contiguous events,
 *
 *   q_e   = w_e·(P_e/norm + bkg_e)
 *   S_g   = Σ_{e∈g} q_e
 *   Q     = −Σ_g log(S_g)
 *   dQ/dP_e   = −(w_e/norm)/S_g
 *   dQ/dnorm  = Σ_g (1/S_g)·(Σ_{e∈g} w_e·P_e)/norm²
 *
 * for ``resolution_size > 1``.  ``resolution_size <= 1`` keeps the v4
 * per-event form (see "Bit-identity anchors") — the two live side by
 * side in the same kernel.
 *
 * The phase-space path (``use_norm == 0``) is unchanged (linear Σ w·P)
 * and the per-event returned P stays ``P_sig = Σ_p |A_p|²``.
 *
 * ── How the tile/common pass is reused (no second BW pass) ─────────
 * The gen kernel computes one per-event tile of ``common = amp/BW`` and
 * feeds BOTH gradient phases from it.  v5 keeps exactly that, and only
 * makes sure the group structure never forces a re-evaluation:
 *
 *   rs <= TE   chunk = rs·(TE // rs)  events: every chunk starts on a
 *              group boundary and holds WHOLE groups, so the group sum
 *              S_g is available inside the chunk (only the very last
 *              group may be partial when ne is not a multiple of rs).
 *              Within the chunk: common → A_p → P_e/q_e → S_g, Q, dnorm
 *              → dQ/dP_e → the two gradient phases on the same tile.
 *              The vector width stays the chunk (64 lanes for
 *              rs = 1, 2, 4, 8, …).
 *
 *   rs > TE    a group spans chunks, so the WORK UNIT becomes the GROUP:
 *              each thread walks a whole group, tile by tile (same
 *              per-chunk ``common`` tile, still computed once and reused
 *              by both gradient phases).  Linearity of the whole backward
 *              chain in dQ/dP_e is used: each event's gradient is
 *              accumulated with the per-event factor f_e = w_e/norm into
 *              per-thread per-GROUP temporaries (grad_ck[N],
 *              grad_coeff[n_wave·n_res·n_basis], ~6 KB); when the group
 *              ends, const_g = −1/S_g is applied ONCE and the temporaries
 *              are added into the per-thread totals (and zeroed).  Q and
 *              dnorm ride along as running group sums — no second BW pass
 *              anywhere.
 *
 * (These are the two `resolution_size` regimes; the group formula is only
 * entered for rs > 1, see below.)
 *
 * ── Bit-identity anchors ───────────────────────────────────────────
 *   * ``use_norm == 0`` : bit-identical to ``kernels_cpu_pwa_gen.c``
 *     (same code, same tile order).
 *   * ``resolution_size <= 1`` : the exact v4 per-event path
 *     (``−Σ w·log(P/norm+b)``), so it is bit-identical to the v4 kernel
 *     for ANY weights — this is what lets ``cpu_pwa_gen`` be a thin alias
 *     of this file at rsize = 1.  The one-event GROUP formula is
 *     deliberately NOT used there: it would give −log(w·(P/norm+b)) =
 *     v4 − log(w), i.e. a different objective (and dQ/dP) by the
 *     constant −Σ log(w) for non-unit weights.
 *   * ``resolution_size > 1`` : the grouped log-sum above (no v4
 *     analogue exists, so nothing to be identical to).
 *
 * Everything else (opaque basis, s0c preset of the coefficients, host
 * chain dQ/ds0, dQ/dmult → dQ/dm0, dQ/dg0) is the v4 kernel's; the
 * kernel still knows nothing about what a coefficient means.  This file
 * SUPERSEDES ``kernels_cpu_pwa_gen.c`` (kernel policy case 3: a green
 * case-1 copy takes over; the old source is archived under
 * ``cpu/archive/kernels_cpu_pwa_gen_v1.c`` for rollback).
 *
 * The gen file's compile-time experiment macro (``-DPWA_GEN_RECOMPUTE_INV``)
 * is NOT carried over — only the measured-default path is copied.
 *
 * API (cffi, see cpu/_pwa_v5.py):
 *   cpu_pwa_v5_load()     — build a data handle (borrows the arrays)
 *   cpu_pwa_v5_compute()  — forward NLL + gradients (OpenMP over tiles)
 *   cpu_pwa_v5_free()     — release the handle
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
} PWAV5Data;

void* cpu_pwa_v5_load(
    const double* amp_r, const double* amp_i,
    const double* F_r, const double* F_i, const int* slot_map,
    int n_uslot,
    const double* w, const double* bkg, int ne,
    int n_wave, int N, int n_proj, int n_res, int n_basis)
{
    PWAV5Data* d = (PWAV5Data*)calloc(1, sizeof(PWAV5Data));
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

void cpu_pwa_v5_free(void* h)
{
    if (h) free(h);
}



/* ── one (wave, factor) proportional denominator over a tile ─────── */
/* bw_{w,r}(e) = Σ_i coeff[f·nf + i]·F[slot_map[f·nf + i]](e), f = w·n_res+r.
 * One basis pass; the shared-cache tile is left untouched. */
__attribute__((always_inline)) static inline void pwa_v5_factor(const double* coeff, const int* slot_map,
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

/* ── ck gradient over a tile segment [b0, b1) ───────────────────── */
/* Σ_p Σ_{e} dQ/dA_p·common[p·N+k]  (the gen kernel's loop, restricted
 * to a segment so a group boundary can split a chunk). */
__attribute__((always_inline)) static inline void pwa_v5_grad_ck(
    const double* dQr, const double* dQi,
    const double* cr, const double* ci,
    int TE, int P, int N, int b0, int b1,
    double* gckr, double* gcki)
{
    for (int k = 0; k < N; k++) {
        double sr = 0.0, si = 0.0;
        for (int p = 0; p < P; p++) {
            const int w = p * N + k;
            const double* xr = dQr + p * TE;
            const double* xi = dQi + p * TE;
            const double* pr = cr + (size_t)w * TE;
            const double* pi = ci + (size_t)w * TE;
            #pragma omp simd reduction(+:sr,si)
            for (int b = b0; b < b1; b++) {
                sr += xr[b] * pr[b] - xi[b] * pi[b];
                si += xr[b] * pi[b] + xi[b] * pr[b];
            }
        }
        gckr[k] += sr;
        gcki[k] += si;
    }
}

/* ── coefficient gradient over a tile segment [b0, b1) ───────────── */
/* ∂Q/∂coeff[(w,r,i)] = 2·Re(dBW_w·(1/bw_{w,r})·F_i), dBW = −(dQ/dcommon)·common */
__attribute__((always_inline)) static inline void pwa_v5_grad_coeff(
    const PWAV5Data* d,
    const double* ck_r, const double* ck_i,
    const double* dQr, const double* dQi,
    const double* cr, const double* ci,
    const double* ir, const double* ii,
    int e0, int TE, int b0, int b1,
    double* ur, double* ui, double* dr, double* di,
    double* gco)
{
    const int nw = d->n_wave, N = d->N, nr = d->n_res, nf = d->n_basis;
    const int ne = d->ne;
    for (int w = 0; w < nw; w++) {
        const int p = w / N;
        const int k = w - p * N;
        const double ckr = ck_r[k], cki = ck_i[k];
        const double* qr_ = dQr + p * TE;
        const double* qi_ = dQi + p * TE;
        const double* pr = cr + (size_t)w * TE;
        const double* pi = ci + (size_t)w * TE;
        /* dQ/dcommon = dQ/dA_p·ck_k */
        #pragma omp simd
        for (int b = b0; b < b1; b++) {
            ur[b] = qr_[b] * ckr - qi_[b] * cki;
            ui[b] = qr_[b] * cki + qi_[b] * ckr;
        }
        for (int r = 0; r < nr; r++) {
            const int f = w * nr + r;
            const double* xr = ir + (size_t)f * TE;
            const double* xi = ii + (size_t)f * TE;
            if (nr == 1) {
                /* tile holds common·(1/BW): dBW = −(dQ/dcommon)·tile */
                #pragma omp simd
                for (int b = b0; b < b1; b++) {
                    dr[b] = -(ur[b] * xr[b] - ui[b] * xi[b]);
                    di[b] = -(ur[b] * xi[b] + ui[b] * xr[b]);
                }
            } else {
                /* dBW = −(dQ/dcommon)·common·(1/bw_r) */
                #pragma omp simd
                for (int b = b0; b < b1; b++) {
                    const double m_r = pr[b] * xr[b] - pi[b] * xi[b];
                    const double m_i = pr[b] * xi[b] + pi[b] * xr[b];
                    dr[b] = -(ur[b] * m_r - ui[b] * m_i);
                    di[b] = -(ur[b] * m_i + ui[b] * m_r);
                }
            }
            const int* sm = d->slot_map + (size_t)f * nf;
            double* gc = gco + (size_t)f * nf;
            for (int i = 0; i < nf; i++) {
                const int u = sm[i];
                const double* fr = d->Fr + (size_t)u * ne + e0;
                const double* fi = d->Fi + (size_t)u * ne + e0;
                double acc = 0.0;
                #pragma omp simd reduction(+:acc)
                for (int b = b0; b < b1; b++)
                    acc += 2.0 * (dr[b] * fr[b] - di[b] * fi[b]);
                gc[i] += acc;
            }
        }
    }
}

/* ── per-tile ``common`` + projection sums ──────────────────────── */
/* Fills the tile buffers for events [e0, e0+nen):
 *   ir/ii[(w,r)] = 1/bw_{w,r}  (n_res >= 2)  or  common_w·(1/BW_w) (n_res == 1)
 *   cr/ci[w]     = common_w    = amp_w / BW_w
 *   Ar/Ai[p]     = Σ_k ck_k·common_{p·N+k}
 * THIS is the only place the BW denominators are evaluated — both
 * gradient phases run off these buffers.  Every statement is per event
 * (no reduction order), so the values are independent of the caller's
 * vectorisation/tile split. */
__attribute__((always_inline)) static inline void pwa_v5_common_tile(
    const PWAV5Data* d, const double* coeff,
    const double* ck_r, const double* ck_i,
    int e0, int nen, int TE,
    double* cr, double* ci, double* ir, double* ii,
    double* Ar, double* Ai, double* bwr, double* bwi)
{
    const int ne = d->ne, nw = d->n_wave, N = d->N, P = d->n_proj;
    const int nr = d->n_res, nf = d->n_basis;

    /* ── per-factor 1/bw, then BW product, common = amp/BW ─────────── */
    for (int w = 0; w < nw; w++) {
        const double* amr = d->amp_r + (size_t)w * ne + e0;
        const double* ami = d->amp_i + (size_t)w * ne + e0;
        double* pr = cr + (size_t)w * TE;
        double* pi = ci + (size_t)w * TE;
        /* the per-factor 1/bw tile is filled once and reused by the
         * gradient phase.  At nr == 1 the single factor IS the
         * propagator: the fold and the common product ride along in the
         * same pass (no ring of multiply-by-1, exactly the gen kernel's
         * loop shape → bit-identical). */
        for (int r = 0; r < nr; r++) {
            const int f = w * nr + r;
            pwa_v5_factor(coeff, d->slot_map, nf, d->Fr, d->Fi,
                          ne, e0, nen, f, bwr, bwi);
            double* xr = ir + (size_t)f * TE;
            double* xi = ii + (size_t)f * TE;
            if (nr == 1) {
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
    }

    /* ── projection sums: A_p[e] += ck_k·common[p·N+k][e] ──────────── */
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
}

/* Per-event P_e = Σ_p |A_p|² and dQ/dA_p = dQ/dP·(A_p, −A_p) for one
 * event b of the tile.  (dQ/dP is the caller's scalar for the event.) */
#define PWA_V5_EVENT_PQ(b)                                                   \
    do {                                                                     \
        double Pe_ = 0.0;                                                    \
        for (int p = 0; p < P; p++)                                          \
            Pe_ += Ar[p * TE + (b)] * Ar[p * TE + (b)]                       \
                 + Ai[p * TE + (b)] * Ai[p * TE + (b)];                      \
        Pe_b[(b)] = Pe_;                                                     \
    } while (0)

/* ── forward + backward, event-block SoA, grouped log-sum NLL ────── */

double cpu_pwa_v5_compute(
    void* h,
    const double* coeff,
    const double* ck_r, const double* ck_i,
    double norm, int use_norm, int resolution_size,
    double* ogcoeff,
    double* ogck_r, double* ogck_i,
    double* P_out, double* dnorm_out)
{
    PWAV5Data* d = (PWAV5Data*)h;
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
    /* group width (0/1 = one event per group).  Groups exist only in the
     * DATA-NLL branch: ``norm == None`` is the phase-space path, whose
     * per-event linear Σ w·P does not know about groups at all — so it
     * always runs the plain TE tile (rs is irrelevant there). */
    const int rs = (resolution_size > 1) ? resolution_size : 1;
    const int grouped = (use_norm != 0);
    /* grouped && rs > TE: a group spans chunks → the WORK UNIT is the group */
    const int spans = grouped && (rs > TE);
    /* grouped && rs <= TE: chunk = rs·(TE//rs) — whole groups, same width */
    const int chunk = (grouped && !spans) ? (rs * (TE / rs)) : TE;
    const int ntiles = (ne + chunk - 1) / chunk;
    const int ngroups = (ne + rs - 1) / rs;

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
    int alloc_fail = 0;

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
        double* dqP  = (double*)malloc((size_t)TE * sizeof(double)); /* dQ/dP_e */
        /* per-GROUP gradient temporaries (rs > TE branch only, whole group
         * owned by this thread) */
        double* gpr  = NULL;   /* grad_ck real */
        double* gpi  = NULL;   /* grad_ck imag */
        double* gpc  = NULL;   /* grad_coeff   */
        if (spans) {
            gpr = (double*)calloc((size_t)padN, sizeof(double));
            gpi = (double*)calloc((size_t)padN, sizeof(double));
            gpc = (double*)calloc((size_t)padC, sizeof(double));
        }
        const int common_bufs_ok = (cr && ci && ir && ii && bwr && bwi);
        const int grad_bufs_ok = (ur && ui && dr && di && Ar && Ai
                                  && dQr && dQi && Pe_b && dqP);
        const int bufs_ok = common_bufs_ok && grad_bufs_ok
                            && (!spans || (gpr && gpi && gpc));
        if (!bufs_ok) alloc_fail = 1;

        if (spans) {
            /* ── rs > TE: one WORK UNIT = one whole group ──────────────
             * The group's tiles are walked by the SAME thread, so the
             * running group sums and the per-group gradient temporaries
             * are thread-local by construction.  Every tile still
             * evaluates ``common`` once (pwa_v5_common_tile) and feeds
             * both gradient phases from that tile. */
            #pragma omp for schedule(static)
            for (int g = 0; g < ngroups; g++) {
                const int g0 = g * rs;
                const int g1 = (ne - g0 > rs) ? (g0 + rs) : ne;
                double S = 0.0, swp = 0.0;
                for (int e0 = g0; e0 < g1; e0 += TE) {
                    const int nen = (g1 - e0 > TE) ? TE : (g1 - e0);
                    const double* wgt = d->weight + e0;
                    const double* bkg = d->bkg + e0;
                    if (!bufs_ok) continue;
                    pwa_v5_common_tile(d, coeff, ck_r, ck_i, e0, nen, TE,
                                       cr, ci, ir, ii, Ar, Ai, bwr, bwi);
                    /* P_e, q_e (group sum) and the per-event factor
                     * f_e = w_e/norm (the group's const_g is applied at
                     * the end, see below) */
                    #pragma omp simd reduction(+:S,swp)
                    for (int b = 0; b < nen; b++) {
                        PWA_V5_EVENT_PQ(b);
                        S += wgt[b] * (Pe_b[b] / norm + bkg[b]);
                        swp += wgt[b] * Pe_b[b];
                    }
                    for (int b = 0; b < nen; b++) {
                        const double dQ_dP = wgt[b] / norm;   /* f_e */
                        for (int p = 0; p < P; p++) {
                            dQr[p * TE + b] =  dQ_dP * Ar[p * TE + b];
                            dQi[p * TE + b] = -dQ_dP * Ai[p * TE + b];
                        }
                    }
                    if (P_out) {
                        #pragma omp simd
                        for (int b = 0; b < nen; b++) P_out[e0 + b] = Pe_b[b];
                    }
                    pwa_v5_grad_ck(dQr, dQi, cr, ci, TE, P, N, 0, nen,
                                   gpr, gpi);
                    pwa_v5_grad_coeff(d, ck_r, ck_i, dQr, dQi, cr, ci, ir, ii,
                                      e0, TE, 0, nen, ur, ui, dr, di, gpc);
                }
                if (!bufs_ok) continue;
                /* group complete: Q, dnorm and const_g = −1/S_g once */
                Ql -= log(S);
                dnl += (swp / (norm * norm)) / S;
                const double cg = -1.0 / S;
                for (int i = 0; i < nc; i++) {
                    gco[i] += cg * gpc[i];
                    gpc[i] = 0.0;
                }
                for (int k = 0; k < N; k++) {
                    gckr[k] += cg * gpr[k];  gpr[k] = 0.0;
                    gcki[k] += cg * gpi[k];  gpi[k] = 0.0;
                }
            }
        } else {
            #pragma omp for schedule(static)
            for (int tile = 0; tile < ntiles; tile++) {
                const int e0 = tile * chunk;
                const int nen = (ne - e0 > chunk) ? chunk : (ne - e0);
                const double* wgt = d->weight + e0;
                const double* bkg = d->bkg + e0;

                if (!bufs_ok) continue;
                pwa_v5_common_tile(d, coeff, ck_r, ck_i, e0, nen, TE,
                                   cr, ci, ir, ii, Ar, Ai, bwr, bwi);

                if (!use_norm) {
                    /* ── phase space: linear Σ w·P, per event (gen) ──── */
                    #pragma omp simd reduction(+:Ql)
                    for (int b = 0; b < nen; b++) {
                        PWA_V5_EVENT_PQ(b);
                        const double dQ_dP = wgt[b];
                        Ql += wgt[b] * Pe_b[b];
                        for (int p = 0; p < P; p++) {
                            dQr[p * TE + b] =  dQ_dP * Ar[p * TE + b];
                            dQi[p * TE + b] = -dQ_dP * Ai[p * TE + b];
                        }
                    }
                    if (P_out) {
                        #pragma omp simd
                        for (int b = 0; b < nen; b++) P_out[e0 + b] = Pe_b[b];
                    }
                    pwa_v5_grad_ck(dQr, dQi, cr, ci, TE, P, N, 0, nen,
                                   gckr, gcki);
                    pwa_v5_grad_coeff(d, ck_r, ck_i, dQr, dQi, cr, ci, ir, ii,
                                      e0, TE, 0, nen, ur, ui, dr, di, gco);
                } else if (rs == 1) {
                    /* ── resolution_size <= 1: the EXACT v4 per-event path
                     *  Q     = −Σ_e w_e·log(P_e/norm + b_e)
                     *  dnorm = Σ_e w_e·(P_e/norm²)/(P_e/norm + b_e)
                     *  dQ/dP_e = −w_e/(norm·(P_e/norm + b_e))
                     * This is what ``cpu_pwa_gen``/``cuda_v4_pwa`` compute,
                     * so rsize <= 1 is BIT-IDENTICAL to cpu_pwa_gen for
                     * ANY weights (the alias contract rests on this).
                     *
                     * Why not route rsize == 1 through the group formula
                     * below?  A one-event group would give
                     * −log(w_e·(P_e/norm + b_e)) = v4 − log(w_e), i.e. the
                     * objective would differ from v4 by the constant
                     * −Σ_e log(w_e) whenever the weights are not 1 — a
                     * different likelihood (and a different dQ/dP).  The
                     * per-event branch keeps v4 semantics exactly; the
                     * group formula is only used for rsize > 1, where no
                     * v4 analogue exists.  Branched per chunk, so the
                     * rsize == 1 case costs nothing. ─────────────────── */
                    #pragma omp simd reduction(+:Ql,dnl)
                    for (int b = 0; b < nen; b++) {
                        PWA_V5_EVENT_PQ(b);
                        const double ratio = Pe_b[b] / norm;
                        const double denom = ratio + bkg[b];
                        Ql -= wgt[b] * log(denom);
                        const double dQ_dP = -wgt[b] / (norm * denom);
                        dnl += wgt[b] * (Pe_b[b] / (norm * norm)) / denom;
                        for (int p = 0; p < P; p++) {
                            dQr[p * TE + b] =  dQ_dP * Ar[p * TE + b];
                            dQi[p * TE + b] = -dQ_dP * Ai[p * TE + b];
                        }
                    }
                    if (P_out) {
                        #pragma omp simd
                        for (int b = 0; b < nen; b++) P_out[e0 + b] = Pe_b[b];
                    }
                    pwa_v5_grad_ck(dQr, dQi, cr, ci, TE, P, N, 0, nen,
                                   gckr, gcki);
                    pwa_v5_grad_coeff(d, ck_r, ck_i, dQr, dQi, cr, ci, ir, ii,
                                      e0, TE, 0, nen, ur, ui, dr, di, gco);
                } else {
                    /* ── grouped log-sum, whole groups inside the chunk ──
                     *  Phase 1 walks the groups (P_e, S_g, Q, dnorm) and
                     *  stores the finished per-event dQ/dP_e; phase 2 then
                     *  runs the dQ/dA and the two gradient phases ONCE at
                     *  the full chunk width, so rs = 2/4/8/… keeps the
                     *  64-lane vectorisation (the common tile is reused
                     *  exactly as in the v4 path). */
                    int b = 0;
                    while (b < nen) {
                        int b1 = b + (rs - ((e0 + b) % rs));
                        if (b1 > nen) b1 = nen;
                        double S = 0.0, swp = 0.0;
                        #pragma omp simd reduction(+:S,swp)
                        for (int i = b; i < b1; i++) {
                            PWA_V5_EVENT_PQ(i);
                            S += wgt[i] * (Pe_b[i] / norm + bkg[i]);
                            swp += wgt[i] * Pe_b[i];   /* Σ w·P for dnorm */
                        }
                        Ql -= log(S);
                        dnl += (swp / (norm * norm)) / S;
                        #pragma omp simd
                        for (int i = b; i < b1; i++)
                            dqP[i] = -(wgt[i] / norm) / S;
                        b = b1;
                    }
                    for (int p = 0; p < P; p++) {
                        const double* ar = Ar + p * TE;
                        const double* ai = Ai + p * TE;
                        double* gr = dQr + p * TE;
                        double* gi = dQi + p * TE;
                        #pragma omp simd
                        for (int i = 0; i < nen; i++) {
                            gr[i] =  dqP[i] * ar[i];
                            gi[i] = -dqP[i] * ai[i];
                        }
                    }
                    if (P_out) {
                        #pragma omp simd
                        for (int i = 0; i < nen; i++) P_out[e0 + i] = Pe_b[i];
                    }
                    pwa_v5_grad_ck(dQr, dQi, cr, ci, TE, P, N, 0, nen,
                                   gckr, gcki);
                    pwa_v5_grad_coeff(d, ck_r, ck_i, dQr, dQi, cr, ci, ir, ii,
                                      e0, TE, 0, nen, ur, ui, dr, di, gco);
                }
            }
        }

        tQ[tid] += Ql;
        tDn[tid] += dnl;
        free(cr); free(ci); free(ir); free(ii);
        free(bwr); free(bwi); free(ur); free(ui); free(dr); free(di);
        free(Ar); free(Ai); free(dQr); free(dQi);
        free(Pe_b); free(dqP);
        free(gpr); free(gpi); free(gpc);
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
    if (alloc_fail) return NAN;      /* a per-thread tile buffer failed */
    if (dnorm_out) *dnorm_out = dn_total;
    return Q_total;
}
