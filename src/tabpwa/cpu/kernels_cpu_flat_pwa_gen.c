/* ══════════════════════════════════════════════════════════════════════
 * kernels_cpu_flat_pwa_gen.c — the flat `pwa_gen` config on the CPU
 * (C + OpenMP + AVX-512 fp64), with the SEMANTICS of `numpy_flat`.
 *
 *   P(e) = Σ_p |A_p(e)|²
 *   A_p(e) = Σ_k ck_k·fa[e,k,p]·fl[e,k]·num[e,k] / D[e,k]
 *   D[e,k] = Π_r  Σ_m w[k,r,m]·basis_m(m_e) ,  w[k,r,m] = gamma[k,r,m]·coeff[m]
 *
 * measured against `src/tabpwa/backends/numpy_flat_backend.py` (the fp64
 * oracle): Q, P, dck, dcoeff and dnorm at the fp64 round-off floor, because
 * NOTHING here is fp32 and nothing is carried over from a previous call.
 *
 * ── why a new kernel (kernel policy §1) ────────────────────────────────
 *
 * No working kernel is touched: this is a standalone `.c` with its own
 * symbols (`cpu_flat_pwa_gen_*`), its own `.so` (built by
 * `tabpwa.cpu.build`: gcc -O3 -march=native -fopenmp) and its own wrapper
 * module.  The flat config had no CPU implementation at all (`numpy_flat`
 * is the plain-loops reference) and the GPU siblings are CUDA.  The GPU
 * tile spends 5.9 GB of device cache on the parameter-independent factors
 * (`us`, `amp`); this kernel recomputes them every call, which is what
 * makes it a fallback at large `ne` on a box where fp64 is 1/64 of the
 * fp32 rate and the 16 cores are otherwise idle.
 *
 * ── the vectorisation: EVENTS are the lanes ────────────────────────────
 *
 * One AVX-512 register holds 8 events (`__m512d`), so a "tile" is 8 events
 * and every per-event quantity is a vector:
 *
 *   ka[a]   = Π_j cos(angle[e, col_a, j]·k_a[j] + b_a[j])          (n_ab)
 *   fa[c]   = Σ_a ka[a]·matrix_angle[a,c]           (CSR over the nz)
 *   fl[k]   = Π_d interp(fl_table[row], q[e, qcol])       (per deduped use)
 *   num[k]  = Π_r interp(num_table[row], m[e, mcol])               (complex)
 *   F[u]    = interp(basis_table[row], m[e, mcol])                 (uslot)
 *   D[k]    = Π_r Σ_m w[k,r,m]·F[slot(m)]
 *   A_p     = Σ_k ck_k·common[k]·fa[c(p,k)]
 *   P       = Σ_p |A_p|²
 *
 * i.e. exactly the GPU tile's arithmetic with an event per lane instead of
 * a thread per (event, wave).  Tables and accumulators are fp64 end to end.
 *
 * Five folds, all EXACT and all measured on the production shape:
 *
 *   fl uses      (row bytes, q column)   52 -> 26   (host, byte-exact key)
 *   num uses     (row bytes, m column)   26 ->  9   (host, byte-exact key)
 *   basis uses   (row bytes, m column)   64 -> 48   (host, byte-exact key)
 *   fa columns   (matrix_angle col bytes)52 -> 26   (host, byte-exact key)
 *   angle basis  cos(0·angle + b) is the constant cos(b) (folded),
 *                cos(0·angle + 0) = 1 (dropped)          (in `create`)
 *
 * so a tile performs 140 Catmull-Rom evaluations per event, not 200, walks
 * `matrix_angle`'s TRUE non-zeros (304 of 2704 entries), and the earlier
 * per-(wave,factor) numerator loop (26 complex interpolations for 9 distinct
 * rows) is gone — the `num` stage went from 13 % to 4 % of the data pass.
 *
 * ── the Catmull-Rom weights are per (x column), not per row ────────────
 *
 * `numpy_flat` interpolates LINEARLY; this kernel uses the CUDA tile's
 * Catmull-Rom `gps0_cr1d` (the difference is 1.4e-3/n_interp² ~ 3e-7, hence
 * the measured floor).  The bin and the four CR weights depend only on the
 * event's position inside its bin, i.e. on the x COLUMN, so they are
 * computed once per column per tile (`cfpg_bin`) and reused by every table
 * row sampled there: 9 columns -> 140 rows.
 *
 * ── the one fast path that is not a plain `cos` ────────────────────────
 *
 * `angle_k` is a difference of spin projections, i.e. an INTEGER in every
 * `pwa_gen` config (int32 here, |k| ≤ 5 on the production shape), and a
 * component's argument is `k·angle + b`.  For integral `k` the basis needs
 * only `cos(angle)` and `sin(angle)` per (event, angle column, component) —
 * 2 libmvec calls per pair per tile instead of one `cos` per component —
 * and each component becomes a Chebyshev pair,
 *
 *     cos(k·x + b) = T_|k|(c)·cos b − sgn(k)·U_|k|−1(c)·s·sin b ,
 *     c = cos x, s = sin x,
 *
 * at ~3 FMA for |k| = 1 and ~19 for |k| = 5.  It is NOT assumed: `create`
 * enables it only when every `k` is integral and |k| ≤ CFPG_MAX_K, and a
 * context built with `flags = CFPG_FLAG_GENERAL_COS` forces the general
 * path — the fixture compares the two on a real sample.
 *
 * ── gradients: the same raw Wirtinger half as the GPU siblings ─────────
 *
 * With u_k = Σ_p conj(A_p)·fa (the "u-merge": `common` is shared by the
 * projections of a wave, so dQ/dck and dQ/dD are one complex number):
 *
 *   v_k          = u_k·common_k
 *   pv_k         = -v_k/D_k
 *   grad_ck_k    = Σ_e dQ/dP_e·v_k                     (NO factor 2)
 *   dQ/ds_kr     = dQ/dP_e·ck_k·pv_k·Π_{l≠r} s_l
 *   grad_coeff_m = Σ_e dQ/dP_e·Σ_{kr} 2Re(ck·pv·others·gamma·F_m)
 *   dnorm        = per-event: Σ_e w·P/(norm·(P + b·norm))
 *                  grouped  : Σ_g W_g·Pb_g/(S_g·norm²)
 *
 * `grad_ck` is the raw Wirtinger half — `CKProduct.backprop_grad` applies
 * the 2·Re real-parameter convention downstream, so a factor 2 here would
 * double-count (identical to `numpy_flat` and `cuda_flat_pwa_gen_s0e`).
 *
 * The gradient is LINEAR in dQ/dP (v, pv and g need only the forward
 * numbers), which is why the grouped objective is ONE pass: a work unit
 * walks its events once, keeps the per-event `v` and `g`
 * (rsize × (2N + NB) doubles) and closes the group once its log-sum is
 * known.  `resolution_size == 1`, and `norm == None` in every mode, close
 * per event and need no per-group storage at all.
 *
 * ── batching ───────────────────────────────────────────────────────────
 *
 * `batch_size` changes NOTHING: the sample is walked in event tiles of 8
 * and the parallel axis is the tile (or the group), not a chunk.  There is
 * no device buffer to bound, the handle owns the whole sample, and the
 * numbers are therefore bit-identical for every `batch_size` (the
 * reference `numpy_flat` ignores it as well).
 *
 * ── the timing hook ────────────────────────────────────────────────────
 *
 * `cpu_flat_pwa_gen_compute(..., long long* stage)` accumulates `rdtsc`
 * cycle counts on thread 0 per stage (0 = w = gamma·coeff, 1 = the shared
 * grid/bin work, 2 = ka, 3 = fa, 4 = fl, 5 = num, 6 = den, 7 = fwd+bwd,
 * 8 = accumulate) when `stage != NULL`; the benchmark harness scales them
 * by the measured wall time.  In a fit the pointer is NULL and the whole
 * instrumentation is a predictable-not-taken branch.
 * ══════════════════════════════════════════════════════════════════════ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>
#include <dlfcn.h>

#if !defined(__AVX512F__)
#error "cpu_flat_pwa_gen is the AVX-512 fp64 flat-config kernel: build it with -march=native (or -mavx512f -mavx512vl -mavx512dq -mfma) on an AVX-512 host"
#endif

#define CFPG_LANES   8      /* AVX-512 fp64 lanes = events per tile        */
#define CFPG_MAXCAP  1024   /* per-thread group buffer (events); rsize cap */
#define CFPG_NSTAGE  9
#define CFPG_MAX_K   8      /* |k| ceiling of the Chebyshev fast path      */
/*: force the general (libmvec `cos`) angle basis instead of the integral-k
 *  Chebyshev path — a test/benchmark hook, never used by the wrapper */
#define CFPG_FLAG_GENERAL_COS 1

/* ── libmvec (glibc's vectorised cos/log), resolved with dlsym so that a
 *    build without `-lmvec` still loads and falls back to scalar libm ── */
typedef __m512d (*cfpg_vfn)(__m512d);
static cfpg_vfn cfpg_vcos = NULL;
static cfpg_vfn cfpg_vlog = NULL;
static int cfpg_vmath_ok = 0;

static void cfpg_vmath_init(void)
{
    void *h = dlopen("libmvec.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!h) h = dlopen("libmvec.so", RTLD_LAZY | RTLD_GLOBAL);
    if (h) {
        cfpg_vcos = (cfpg_vfn)(void (*)(void))dlsym(h, "_ZGVeN8v_cos");
        cfpg_vlog = (cfpg_vfn)(void (*)(void))dlsym(h, "_ZGVeN8v_log");
    }
    cfpg_vmath_ok = (cfpg_vcos != NULL && cfpg_vlog != NULL) ? 1 : 0;
}

static inline __m512d cfpg_cos8(__m512d v)
{
    if (__builtin_expect(cfpg_vmath_ok == 1, 1)) return cfpg_vcos(v);
    double b[8];
    _mm512_storeu_pd(b, v);
    for (int i = 0; i < 8; i++) b[i] = cos(b[i]);
    return _mm512_loadu_pd(b);
}

static inline __m512d cfpg_sin8(__m512d v)
{
    if (__builtin_expect(cfpg_vmath_ok == 1, 1))
        return cfpg_vcos(_mm512_sub_pd(v,
                                       _mm512_set1_pd(1.5707963267948966)));
    double b[8];
    _mm512_storeu_pd(b, v);
    for (int i = 0; i < 8; i++) b[i] = sin(b[i]);
    return _mm512_loadu_pd(b);
}

static inline __m512d cfpg_log8(__m512d v)
{
    if (__builtin_expect(cfpg_vmath_ok == 1, 1)) return cfpg_vlog(v);
    double b[8];
    _mm512_storeu_pd(b, v);
    for (int i = 0; i < 8; i++) b[i] = log(b[i]);
    return _mm512_loadu_pd(b);
}

/* ── the Catmull-Rom weights of (pm1, p0, p1, p2) ───────────────────────
 *
 *     p0 + 0.5t(-pm1 + p1 + t(2pm1 - 5p0 + 4p1 - p2
 *                             + t(-pm1 + 3p0 - 3p1 + p2)))
 *   = w0·pm1 + w1·p0 + w2·p1 + w3·p2
 */
static inline void cfpg_cr_weights(__m512d t, double *w)
{
    const __m512d half = _mm512_set1_pd(0.5);
    const __m512d t2 = _mm512_mul_pd(t, t);
    const __m512d h = _mm512_mul_pd(half, t);
    const __m512d w0 = _mm512_mul_pd(h, _mm512_fmadd_pd(
                           t, _mm512_sub_pd(_mm512_set1_pd(2.0), t),
                           _mm512_set1_pd(-1.0)));
    const __m512d w1 = _mm512_fmadd_pd(_mm512_mul_pd(half, t2),
                         _mm512_fmsub_pd(_mm512_set1_pd(3.0), t,
                                         _mm512_set1_pd(5.0)),
                         _mm512_set1_pd(1.0));
    const __m512d w2 = _mm512_mul_pd(h, _mm512_fmadd_pd(
                           t, _mm512_fnmadd_pd(_mm512_set1_pd(3.0), t,
                                               _mm512_set1_pd(4.0)),
                           _mm512_set1_pd(1.0)));
    const __m512d w3 = _mm512_mul_pd(_mm512_mul_pd(half, t2),
                         _mm512_sub_pd(t, _mm512_set1_pd(1.0)));
    _mm512_storeu_pd(w + 0, w0);
    _mm512_storeu_pd(w + 8, w1);
    _mm512_storeu_pd(w + 16, w2);
    _mm512_storeu_pd(w + 24, w3);
}

/*: one x column's grid position for the tile — the four neighbouring
 *  RELATIVE bin indices (clamped at the row edges exactly like the CUDA
 *  tile's `gps0_interp_*`) and the four Catmull-Rom weights.  Shared by
 *  every table row sampled at that column. */
static inline void cfpg_bin(int *idx, double *w, __m512d x, double xmin,
                            double xdelta, int bins)
{
    const __m512d diff = _mm512_div_pd(_mm512_sub_pd(x, _mm512_set1_pd(xmin)),
                                       _mm512_set1_pd(xdelta));
    __m256i b = _mm512_cvtpd_epi32(_mm512_floor_pd(diff));
    b = _mm256_max_epi32(b, _mm256_setzero_si256());
    b = _mm256_min_epi32(b, _mm256_set1_epi32(bins - 2));
    __m512d s = _mm512_sub_pd(diff, _mm512_cvtepi32_pd(b));
    s = _mm512_min_pd(_mm512_max_pd(s, _mm512_setzero_pd()),
                      _mm512_set1_pd(1.0));
    _mm256_storeu_si256((__m256i*)(idx + 8), b);
    _mm256_storeu_si256((__m256i*)(idx + 16),
                        _mm256_add_epi32(b, _mm256_set1_epi32(1)));
    _mm256_storeu_si256((__m256i*)(idx + 0),
                        _mm256_max_epi32(_mm256_sub_epi32(
                                             b, _mm256_set1_epi32(1)),
                                         _mm256_setzero_si256()));
    _mm256_storeu_si256((__m256i*)(idx + 24),
                        _mm256_min_epi32(_mm256_add_epi32(
                                             b, _mm256_set1_epi32(2)),
                                         _mm256_set1_epi32(bins - 1)));
    cfpg_cr_weights(s, w);
}

/*: Catmull-Rom of one table row from a column's precomputed bin/weights.
 *  `row_base` is `table + row*bins`, the indices are already row-relative. */
static inline __m512d cfpg_interp(const double *row_base, const int *idx,
                                  const double *w, __mmask8 m)
{
    const __m512d z = _mm512_setzero_pd();
    const __m512d pm1 = _mm512_mask_i32gather_pd(
        z, m, _mm256_loadu_si256((const __m256i*)(idx + 0)), row_base, 8);
    const __m512d p0 = _mm512_mask_i32gather_pd(
        z, m, _mm256_loadu_si256((const __m256i*)(idx + 8)), row_base, 8);
    const __m512d p1 = _mm512_mask_i32gather_pd(
        z, m, _mm256_loadu_si256((const __m256i*)(idx + 16)), row_base, 8);
    const __m512d p2 = _mm512_mask_i32gather_pd(
        z, m, _mm256_loadu_si256((const __m256i*)(idx + 24)), row_base, 8);
    __m512d v = _mm512_fmadd_pd(_mm512_loadu_pd(w + 8), p0,
                                _mm512_mul_pd(_mm512_loadu_pd(w + 0), pm1));
    v = _mm512_fmadd_pd(_mm512_loadu_pd(w + 16), p1, v);
    return _mm512_fmadd_pd(_mm512_loadu_pd(w + 24), p2, v);
}

/* ══════════════════════════════════════════════════════════════════════
 * layouts — ONE definition each, used by `create` (allocation) and by the
 * kernel (pointer offsets), so the two cannot disagree.
 * ══════════════════════════════════════════════════════════════════════ */
typedef struct {
    int mw, qw;                    /* [n_mass·32], [n_mom·32]  CR weights  */
    int cs, sn;                    /* [n_aj·8] cos/sin of an angle pair    */
    int ka;                        /* [n_ab·8]                             */
    int fa_r, fa_i;                /* [n_fa·8]                             */
    int flv;                       /* [n_fl_u·8]                           */
    int flw;                       /* [n_wave·8]                           */
    int nuv_r, nuv_i;              /* [n_num_u·8]                          */
    int Fu_r, Fu_i;                /* [n_uslot·8]                          */
    int s_r, s_i;                  /* [n_wave·n_res·8]                     */
    int oth_r, oth_i;              /* [n_wave·n_res·8]                     */
    int den_r, den_i, rden;        /* [n_wave·8]                           */
    int cm_r, cm_i;                /* [n_wave·8]                           */
    int Ar, Ai;                    /* [n_proj·8]                           */
    int a2r, a2i;                  /* [n_wave·n_res·8]                     */
    int tile_total;
    int seg_vr, seg_vi, seg_g;     /* [n_wave·cap], [n_basis·cap]          */
    int acc_ckr, acc_cki, acc_g;   /* [n_wave·8], [n_basis·8]              */
    int dqp;                       /* [cap]                                */
    int total;
} CFPGLayout;

/*: int32 scratch — the bin index vectors of the mass / momentum columns */
typedef struct {
    int midx;                      /* [n_mass·32] */
    int qidx;                      /* [n_mom·32]  */
    int total;
} CFPGILayout;

static CFPGLayout cfpg_layout(int A, int n_fa, int N, int P, int R, int U,
                              int NB, int n_mass, int n_mom, int n_fl_u,
                              int n_num_u, int n_aj, int cap)
{
    CFPGLayout L;
    int o = 0;
    L.mw    = o; o += n_mass * 32;
    L.qw    = o; o += n_mom * 32;
    L.cs    = o; o += n_aj * CFPG_LANES;
    L.sn    = o; o += n_aj * CFPG_LANES;
    L.ka    = o; o += A * CFPG_LANES;
    L.fa_r  = o; o += n_fa * CFPG_LANES;
    L.fa_i  = o; o += n_fa * CFPG_LANES;
    L.flv   = o; o += n_fl_u * CFPG_LANES;
    L.flw   = o; o += N * CFPG_LANES;
    L.nuv_r = o; o += n_num_u * CFPG_LANES;
    L.nuv_i = o; o += n_num_u * CFPG_LANES;
    L.Fu_r  = o; o += U * CFPG_LANES;
    L.Fu_i  = o; o += U * CFPG_LANES;
    L.s_r   = o; o += N * R * CFPG_LANES;
    L.s_i   = o; o += N * R * CFPG_LANES;
    L.oth_r = o; o += N * R * CFPG_LANES;
    L.oth_i = o; o += N * R * CFPG_LANES;
    L.den_r = o; o += N * CFPG_LANES;
    L.den_i = o; o += N * CFPG_LANES;
    L.rden  = o; o += N * CFPG_LANES;
    L.cm_r  = o; o += N * CFPG_LANES;
    L.cm_i  = o; o += N * CFPG_LANES;
    L.Ar    = o; o += P * CFPG_LANES;
    L.Ai    = o; o += P * CFPG_LANES;
    L.a2r   = o; o += N * R * CFPG_LANES;
    L.a2i   = o; o += N * R * CFPG_LANES;
    L.tile_total = o;
    L.seg_vr = o; o += N * cap;
    L.seg_vi = o; o += N * cap;
    L.seg_g  = o; o += NB * cap;
    L.acc_ckr = o; o += N * CFPG_LANES;
    L.acc_cki = o; o += N * CFPG_LANES;
    L.acc_g   = o; o += NB * CFPG_LANES;
    L.dqp     = o; o += cap;
    L.total   = o;
    return L;
}

/* ══════════════════════════════════════════════════════════════════════
 * context and one loaded sample
 * ══════════════════════════════════════════════════════════════════════ */
typedef struct {
    /* interpolation descriptors, all host-deduped (byte-exact keys) */
    const double *fl_tab;              int fl_bins;
    const int *fl_u_row, *fl_u_col;    int n_fl_u;
    const int *fl_use;
    const double *num_r, *num_i;       int m_bins;
    const int *num_u_row, *num_u_col;  int n_num_u;
    const int *num_use;
    const double *bas_r, *bas_i;
    const int *us_row, *us_col;        int n_uslot;
    unsigned char *us_const;           /* 1 = the row is a constant */
    double *us_cr, *us_ci;             /* its value, when us_const */
    /* denominator CSR: per (wave, factor) the TRUE non-zeros */
    const int *g_off, *g_m, *g_u;
    const double *g_w;                 int n_nz;
    /* angular basis, compacted (cos(0·angle + b) folded into a constant) */
    int n_ab, n_comp, n_aj, n_col;
    int *ach_off, n_ach;               /* per row: [off[a], off[a+1]) */
    int *ach_col, *ach_j, *ach_pair;   /* angle column, component, pair */
    double *ach_k;                     /* the component's k (general path) */
    int *ach_kabs, *ach_sgn;           /* |k|, sign(k) (integral path) */
    double *ach_b, *ach_cb, *ach_sb;   /* b, cos b, sin b */
    double *ach_aconst;                /* Π cos(b) of the folded k == 0 */
    int int_k;                         /* 1 = use the Chebyshev path */
    const int *angle_index;
    const double *angle_k, *angle_b;
    const int *fa_ptr, *fa_row;
    const double *fa_r, *fa_i;         int n_fa;
    const int *fa_slot;
    double m_min, m_delta, q_min, q_delta;
    int N, P, R, D, NB;
    int n_mass, n_momentum;
    int rsize, cap, n_threads;
    CFPGLayout L;
    CFPGILayout I;
    double *scratch;                   /* n_threads · L.total  doubles */
    int *iscratch;                     /* n_threads · I.total  int32   */
    double *w_nz;                      /* [n_nz]  gamma·coeff, per call */
} CFPGCtx;

typedef struct {
    const double *mass_s;   /* [n_mass·ne]      mass_s[col·ne + e]  */
    const double *q_s;      /* [n_momentum·ne]                      */
    const double *ang_s;    /* [n_angle_flat·ne]                    */
    const double *weight;   /* [ne] */
    const double *bkg;      /* [ne] */
    double *P;              /* [ne]  per-event P (output of the call) */
    int ne;
} CFPGData;

static inline __mmask8 cfpg_mask(int n)
{
    return (n >= CFPG_LANES) ? (__mmask8)0xFF : (__mmask8)((1u << n) - 1);
}

/* ── forward + per-event gradient basis for a tile of n ≤ 8 events ─────
 *
 * Writes P into `h->P[e0 .. e0+n)` and, when `want_vg`, the segment
 * quantities at event offset `so` of the thread's seg buffers:
 *   seg_vr/vi[k] = v_k = u_k·common_k
 *   seg_g[m]     = Σ_{kr} 2Re(ck·pv·others·gamma·F_m)   (unscaled dQ/dcoeff)
 * so a caller that later learns dQ/dP_e accumulates the gradient with one
 * dot product per (k, m) — that is what makes the grouped objective a
 * single pass over the data.
 */
static void cfpg_tile(CFPGCtx *c, CFPGData *h, double *t, int *it, int e0,
                      int n, const double *ck_r, const double *ck_i,
                      int so, int want_vg, long long *st)
{
    const int N = c->N, P = c->P, R = c->R, D = c->D, NB = c->NB;
    const int A = c->n_ab, U = c->n_uslot, nfa = c->n_fa;
    const int ne = h->ne;
    const __mmask8 m = cfpg_mask(n);
    const CFPGLayout L = c->L;
    const CFPGILayout I = c->I;
    int *midx = it + I.midx, *qidx = it + I.qidx;
    double *mw = t + L.mw, *qw = t + L.qw;
    double *ka = t + L.ka;
    double *fa_r = t + L.fa_r, *fa_i = t + L.fa_i;
    double *flv = t + L.flv, *flw = t + L.flw;
    double *nuv_r = t + L.nuv_r, *nuv_i = t + L.nuv_i;
    double *Fu_r = t + L.Fu_r, *Fu_i = t + L.Fu_i;
    double *s_r = t + L.s_r, *s_i = t + L.s_i;
    double *oth_r = t + L.oth_r, *oth_i = t + L.oth_i;
    double *den_r = t + L.den_r, *den_i = t + L.den_i, *rden = t + L.rden;
    double *cm_r = t + L.cm_r, *cm_i = t + L.cm_i;
    double *Ar = t + L.Ar, *Ai = t + L.Ai;
    double *a2r = t + L.a2r, *a2i = t + L.a2i;
    const int cap = c->cap;
    double *seg_vr = t + L.seg_vr, *seg_vi = t + L.seg_vi, *seg_g = t + L.seg_g;
    const long long t0 = st ? __rdtsc() : 0;

    /* ── 1. the shared grid positions: ONE divide per x column, and the
     *      bin indices + CR weights every row sampled there reuses ────── */
    for (int col = 0; col < c->n_mass; col++)
        cfpg_bin(midx + col * 32, mw + col * 32,
                 _mm512_maskz_loadu_pd(m, h->mass_s + (size_t)col * ne + e0),
                 c->m_min, c->m_delta, c->m_bins);
    for (int col = 0; col < c->n_momentum; col++)
        cfpg_bin(qidx + col * 32, qw + col * 32,
                 _mm512_maskz_loadu_pd(m, h->q_s + (size_t)col * ne + e0),
                 c->q_min, c->q_delta, c->fl_bins);
    const long long t1 = st ? __rdtsc() : 0;

    /* ── 2. the angular basis ─────────────────────────────────────────
     *
     * general path: one libmvec `cos` per component.
     * integral-k  : cos/sin of each (angle column, component) pair once,
     *               then a Chebyshev pair per component. */
    if (!c->int_k) {
        for (int a = 0; a < A; a++) {
            __m512d prod = _mm512_set1_pd(c->ach_aconst[a]);
            for (int i = c->ach_off[a]; i < c->ach_off[a + 1]; i++) {
                const double *base =
                    h->ang_s + (size_t)(c->ach_col[i] * c->n_comp) * ne + e0;
                const __m512d x =
                    _mm512_maskz_loadu_pd(m, base + (size_t)c->ach_j[i] * ne);
                prod = _mm512_mul_pd(prod, cfpg_cos8(_mm512_fmadd_pd(
                    x, _mm512_set1_pd(c->ach_k[i]),
                    _mm512_set1_pd(c->ach_b[i]))));
            }
            _mm512_storeu_pd(ka + a * 8, prod);
        }
    } else {
        double *cs = t + L.cs, *sn = t + L.sn;
        for (int i = 0; i < c->n_aj; i++) {
            const __m512d x = _mm512_maskz_loadu_pd(
                m, h->ang_s + (size_t)i * ne + e0);
            _mm512_storeu_pd(cs + i * 8, cfpg_cos8(x));
            _mm512_storeu_pd(sn + i * 8, cfpg_sin8(x));
        }
        for (int a = 0; a < A; a++) {
            __m512d prod = _mm512_set1_pd(c->ach_aconst[a]);
            for (int i = c->ach_off[a]; i < c->ach_off[a + 1]; i++) {
                const int pair = c->ach_pair[i];
                const __m512d cv = _mm512_loadu_pd(cs + pair * 8);
                const __m512d sv = _mm512_loadu_pd(sn + pair * 8);
                const int kk = c->ach_kabs[i];
                if (__builtin_expect(kk == 1, 1)) {
                    /* cos(x + b) = cos x·cos b − sin x·sin b  (T_1 = c, U_0 = 1) */
                    prod = _mm512_mul_pd(prod, _mm512_fmsub_pd(
                        cv, _mm512_set1_pd(c->ach_cb[i]),
                        _mm512_mul_pd(sv, _mm512_set1_pd(
                            (double)c->ach_sgn[i] * c->ach_sb[i]))));
                    continue;
                }
                const __m512d two_c = _mm512_add_pd(cv, cv);
                /* T_|k|(c) */
                __m512d tm1 = _mm512_set1_pd(1.0), tc = cv;
                for (int q = 2; q <= kk; q++) {
                    const __m512d tn = _mm512_fmsub_pd(two_c, tc, tm1);
                    tm1 = tc;
                    tc = tn;
                }
                /* U_{|k|-1}(c) */
                __m512d um1 = _mm512_set1_pd(1.0);
                __m512d uc = (kk >= 2) ? two_c : _mm512_set1_pd(1.0);
                for (int q = 2; q < kk; q++) {
                    const __m512d un = _mm512_fmsub_pd(two_c, uc, um1);
                    um1 = uc;
                    uc = un;
                }
                const __m512d cterm = _mm512_mul_pd(
                    tc, _mm512_set1_pd(c->ach_cb[i]));
                const __m512d sterm = _mm512_mul_pd(
                    _mm512_mul_pd(uc, sv),
                    _mm512_set1_pd((double)c->ach_sgn[i] * c->ach_sb[i]));
                prod = _mm512_mul_pd(prod, _mm512_sub_pd(cterm, sterm));
            }
            _mm512_storeu_pd(ka + a * 8, prod);
        }
    }
    const long long t2 = st ? __rdtsc() : 0;

    /* ── 3. fa = ka · matrix_angle, over the matrix's TRUE non-zeros ─── */
    for (int col = 0; col < nfa; col++) {
        __m512d sr = _mm512_setzero_pd(), si = _mm512_setzero_pd();
        for (int i = c->fa_ptr[col]; i < c->fa_ptr[col + 1]; i++) {
            const __m512d kv = _mm512_loadu_pd(ka + c->fa_row[i] * 8);
            sr = _mm512_fmadd_pd(kv, _mm512_set1_pd(c->fa_r[i]), sr);
            si = _mm512_fmadd_pd(kv, _mm512_set1_pd(c->fa_i[i]), si);
        }
        _mm512_storeu_pd(fa_r + col * 8, sr);
        _mm512_storeu_pd(fa_i + col * 8, si);
    }
    const long long t3 = st ? __rdtsc() : 0;

    /* ── 4. barriers: fl[k] = Π_d interp(fl_table, q) ────────────────── */
    for (int u = 0; u < c->n_fl_u; u++)
        _mm512_storeu_pd(flv + u * 8,
                         cfpg_interp(c->fl_tab
                                     + (size_t)c->fl_u_row[u] * c->fl_bins,
                                     qidx + c->fl_u_col[u] * 32,
                                     qw + c->fl_u_col[u] * 32, m));
    for (int k = 0; k < N; k++) {
        __m512d p = _mm512_loadu_pd(flv + c->fl_use[k * D] * 8);
        for (int d = 1; d < D; d++)
            p = _mm512_mul_pd(p,
                              _mm512_loadu_pd(flv + c->fl_use[k * D + d] * 8));
        _mm512_storeu_pd(flw + k * 8, p);
    }
    const long long t4 = st ? __rdtsc() : 0;

    /* ── 5. numerator: ONE complex interpolation per deduped use ─────── */
    for (int u = 0; u < c->n_num_u; u++) {
        const int row = c->num_u_row[u];
        const int *idx = midx + c->num_u_col[u] * 32;
        const double *w = mw + c->num_u_col[u] * 32;
        const size_t off = (size_t)row * (size_t)c->m_bins;
        _mm512_storeu_pd(nuv_r + u * 8,
                         cfpg_interp(c->num_r + off, idx, w, m));
        _mm512_storeu_pd(nuv_i + u * 8,
                         cfpg_interp(c->num_i + off, idx, w, m));
    }
    const long long t5 = st ? __rdtsc() : 0;

    /* ── 6. denominators: F at the unique slots, then D = Π_r Σ w·F ──── */
    for (int u = 0; u < U; u++) {
        if (c->us_const[u]) {
            _mm512_storeu_pd(Fu_r + u * 8, _mm512_set1_pd(c->us_cr[u]));
            _mm512_storeu_pd(Fu_i + u * 8, _mm512_set1_pd(c->us_ci[u]));
            continue;
        }
        const int row = c->us_row[u];
        const int *idx = midx + c->us_col[u] * 32;
        const double *w = mw + c->us_col[u] * 32;
        const size_t off = (size_t)row * (size_t)c->m_bins;
        _mm512_storeu_pd(Fu_r + u * 8,
                         cfpg_interp(c->bas_r + off, idx, w, m));
        _mm512_storeu_pd(Fu_i + u * 8,
                         cfpg_interp(c->bas_i + off, idx, w, m));
    }
    const double *w = c->w_nz;
    for (int kr = 0; kr < N * R; kr++) {
        __m512d sr = _mm512_setzero_pd(), si = _mm512_setzero_pd();
        for (int i = c->g_off[kr]; i < c->g_off[kr + 1]; i++) {
            const __m512d wv = _mm512_set1_pd(w[i]);
            sr = _mm512_fmadd_pd(
                wv, _mm512_loadu_pd(Fu_r + c->g_u[i] * 8), sr);
            si = _mm512_fmadd_pd(
                wv, _mm512_loadu_pd(Fu_i + c->g_u[i] * 8), si);
        }
        _mm512_storeu_pd(s_r + kr * 8, sr);
        _mm512_storeu_pd(s_i + kr * 8, si);
    }
    for (int k = 0; k < N; k++) {
        __m512d dr = _mm512_loadu_pd(s_r + (k * R) * 8);
        __m512d di = _mm512_loadu_pd(s_i + (k * R) * 8);
        for (int r = 1; r < R; r++) {
            const __m512d er = _mm512_loadu_pd(s_r + (k * R + r) * 8);
            const __m512d ei = _mm512_loadu_pd(s_i + (k * R + r) * 8);
            const __m512d tr = _mm512_sub_pd(_mm512_mul_pd(dr, er),
                                             _mm512_mul_pd(di, ei));
            di = _mm512_fmadd_pd(dr, ei, _mm512_mul_pd(di, er));
            dr = tr;
        }
        _mm512_storeu_pd(den_r + k * 8, dr);
        _mm512_storeu_pd(den_i + k * 8, di);
        _mm512_storeu_pd(rden + k * 8,
                         _mm512_maskz_mov_pd(
                             m, _mm512_div_pd(
                                    _mm512_set1_pd(1.0),
                                    _mm512_fmadd_pd(dr, dr,
                                                    _mm512_mul_pd(di, di)))));
        /* others_{k,r} = Π_{l≠r} s_l  (1 at n_res == 1) */
        for (int r = 0; r < R; r++) {
            __m512d pr = _mm512_set1_pd(1.0), pi = _mm512_setzero_pd();
            for (int l = 0; l < R; l++) {
                if (l == r) continue;
                const __m512d er = _mm512_loadu_pd(s_r + (k * R + l) * 8);
                const __m512d ei = _mm512_loadu_pd(s_i + (k * R + l) * 8);
                const __m512d tr = _mm512_sub_pd(_mm512_mul_pd(pr, er),
                                                 _mm512_mul_pd(pi, ei));
                pi = _mm512_fmadd_pd(pr, ei, _mm512_mul_pd(pi, er));
                pr = tr;
            }
            _mm512_storeu_pd(oth_r + (k * R + r) * 8, pr);
            _mm512_storeu_pd(oth_i + (k * R + r) * 8, pi);
        }
    }
    const long long t6 = st ? __rdtsc() : 0;

    /* ── 7. common, A_p, P_e ─────────────────────────────────────────── */
    for (int k = 0; k < N; k++) {
        __m512d nr = _mm512_loadu_pd(nuv_r + c->num_use[k * R] * 8);
        __m512d ni = _mm512_loadu_pd(nuv_i + c->num_use[k * R] * 8);
        for (int r = 1; r < R; r++) {
            const __m512d er = _mm512_loadu_pd(
                nuv_r + c->num_use[k * R + r] * 8);
            const __m512d ei = _mm512_loadu_pd(
                nuv_i + c->num_use[k * R + r] * 8);
            const __m512d tr = _mm512_sub_pd(_mm512_mul_pd(nr, er),
                                             _mm512_mul_pd(ni, ei));
            ni = _mm512_fmadd_pd(nr, ei, _mm512_mul_pd(ni, er));
            nr = tr;
        }
        const __m512d fl = _mm512_loadu_pd(flw + k * 8);
        nr = _mm512_mul_pd(nr, fl);
        ni = _mm512_mul_pd(ni, fl);
        /* common = (num·fl)·conj(D)/|D|² */
        const __m512d dr = _mm512_loadu_pd(den_r + k * 8);
        const __m512d di = _mm512_loadu_pd(den_i + k * 8);
        const __m512d rd = _mm512_loadu_pd(rden + k * 8);
        _mm512_storeu_pd(cm_r + k * 8,
                         _mm512_mul_pd(_mm512_fmadd_pd(
                                           ni, di, _mm512_mul_pd(nr, dr)),
                                       rd));
        _mm512_storeu_pd(cm_i + k * 8,
                         _mm512_mul_pd(_mm512_fmsub_pd(
                                           ni, dr, _mm512_mul_pd(nr, di)),
                                       rd));
    }
    for (int p = 0; p < P; p++) {
        __m512d ar = _mm512_setzero_pd(), ai = _mm512_setzero_pd();
        for (int k = 0; k < N; k++) {
            const int col = c->fa_slot[p * N + k];
            const __m512d cr = _mm512_loadu_pd(cm_r + k * 8);
            const __m512d ci = _mm512_loadu_pd(cm_i + k * 8);
            const __m512d fr = _mm512_loadu_pd(fa_r + col * 8);
            const __m512d fi = _mm512_loadu_pd(fa_i + col * 8);
            const __m512d xr = _mm512_fmsub_pd(_mm512_set1_pd(ck_r[k]), cr,
                                               _mm512_mul_pd(
                                                   _mm512_set1_pd(ck_i[k]),
                                                   ci));
            const __m512d xi = _mm512_fmadd_pd(_mm512_set1_pd(ck_r[k]), ci,
                                               _mm512_mul_pd(
                                                   _mm512_set1_pd(ck_i[k]),
                                                   cr));
            ar = _mm512_fmadd_pd(xr, fr, ar);
            ar = _mm512_fnmadd_pd(xi, fi, ar);
            ai = _mm512_fmadd_pd(xr, fi, ai);
            ai = _mm512_fmadd_pd(xi, fr, ai);
        }
        _mm512_storeu_pd(Ar + p * 8, ar);
        _mm512_storeu_pd(Ai + p * 8, ai);
    }
    {
        __m512d pe = _mm512_setzero_pd();
        for (int p = 0; p < P; p++) {
            const __m512d ar = _mm512_loadu_pd(Ar + p * 8);
            const __m512d ai = _mm512_loadu_pd(Ai + p * 8);
            pe = _mm512_fmadd_pd(ar, ar, pe);
            pe = _mm512_fmadd_pd(ai, ai, pe);
        }
        _mm512_mask_storeu_pd(h->P + e0, m, pe);
    }

    /* ── 8. v_k, pv_k, a2 and the unscaled dQ/dcoeff of the tile ─────── */
    for (int k = 0; k < N; k++) {
        __m512d ur = _mm512_setzero_pd(), ui = _mm512_setzero_pd();
        for (int p = 0; p < P; p++) {
            const int col = c->fa_slot[p * N + k];
            const __m512d ar = _mm512_loadu_pd(Ar + p * 8);
            const __m512d ai = _mm512_loadu_pd(Ai + p * 8);
            const __m512d fr = _mm512_loadu_pd(fa_r + col * 8);
            const __m512d fi = _mm512_loadu_pd(fa_i + col * 8);
            /* conj(A_p)·fa */
            ur = _mm512_fmadd_pd(ar, fr, ur);
            ur = _mm512_fmadd_pd(ai, fi, ur);
            ui = _mm512_fmadd_pd(ar, fi, ui);
            ui = _mm512_fnmadd_pd(ai, fr, ui);
        }
        const __m512d cr = _mm512_loadu_pd(cm_r + k * 8);
        const __m512d ci = _mm512_loadu_pd(cm_i + k * 8);
        /* v = u·common, deliberately WITHOUT ck: grad_ck must not carry it */
        const __m512d vr = _mm512_sub_pd(_mm512_mul_pd(ur, cr),
                                         _mm512_mul_pd(ui, ci));
        const __m512d vi = _mm512_fmadd_pd(ur, ci, _mm512_mul_pd(ui, cr));
        /* pv = -v/D = -v·conj(D)/|D|² */
        const __m512d dr = _mm512_loadu_pd(den_r + k * 8);
        const __m512d di = _mm512_loadu_pd(den_i + k * 8);
        const __m512d rd = _mm512_loadu_pd(rden + k * 8);
        const __m512d pvr = _mm512_mul_pd(
            _mm512_fnmadd_pd(_mm512_set1_pd(1.0),
                             _mm512_fmadd_pd(vi, di, _mm512_mul_pd(vr, dr)),
                             _mm512_setzero_pd()), rd);
        const __m512d pvi = _mm512_mul_pd(
            _mm512_fmsub_pd(vr, di, _mm512_mul_pd(vi, dr)), rd);
        for (int r = 0; r < R; r++) {
            const __m512d orr = _mm512_loadu_pd(oth_r + (k * R + r) * 8);
            const __m512d oii = _mm512_loadu_pd(oth_i + (k * R + r) * 8);
            /* ck·pv, then × others, then × 2 (the 2Re of the chain) */
            const __m512d xr = _mm512_fmsub_pd(_mm512_set1_pd(ck_r[k]), pvr,
                                               _mm512_mul_pd(
                                                   _mm512_set1_pd(ck_i[k]),
                                                   pvi));
            const __m512d xi = _mm512_fmadd_pd(_mm512_set1_pd(ck_r[k]), pvi,
                                               _mm512_mul_pd(
                                                   _mm512_set1_pd(ck_i[k]),
                                                   pvr));
            const __m512d yr = _mm512_sub_pd(_mm512_mul_pd(xr, orr),
                                             _mm512_mul_pd(xi, oii));
            const __m512d yi = _mm512_fmadd_pd(xr, oii,
                                               _mm512_mul_pd(xi, orr));
            _mm512_storeu_pd(a2r + (k * R + r) * 8, _mm512_add_pd(yr, yr));
            _mm512_storeu_pd(a2i + (k * R + r) * 8, _mm512_add_pd(yi, yi));
        }
        if (want_vg) {
            _mm512_mask_storeu_pd(seg_vr + (size_t)k * cap + so, m, vr);
            _mm512_mask_storeu_pd(seg_vi + (size_t)k * cap + so, m, vi);
        }
    }
    if (want_vg) {
        /* g_m = Σ_{kr} 2Re(ck·pv·others·gamma·F_m): one FMA pair per nz.
           It accumulates over (k, r), so the lanes of this tile are cleared
           first — a tile offset reused by a later tile must not inherit the
           previous tile's g. */
        for (int mm = 0; mm < NB; mm++)
            _mm512_mask_storeu_pd(seg_g + (size_t)mm * cap + so, m,
                                  _mm512_setzero_pd());
        for (int kr = 0; kr < N * R; kr++) {
            const __m512d xr = _mm512_loadu_pd(a2r + kr * 8);
            const __m512d xi = _mm512_loadu_pd(a2i + kr * 8);
            for (int i = c->g_off[kr]; i < c->g_off[kr + 1]; i++) {
                const int mu = c->g_u[i], mm = c->g_m[i];
                const __m512d wv = _mm512_set1_pd(c->g_w[i]);
                double *gp = seg_g + (size_t)mm * cap + so;
                __m512d acc = _mm512_loadu_pd(gp);
                acc = _mm512_fmadd_pd(
                    xr, _mm512_mul_pd(wv, _mm512_loadu_pd(Fu_r + mu * 8)),
                    acc);
                acc = _mm512_fnmadd_pd(
                    xi, _mm512_mul_pd(wv, _mm512_loadu_pd(Fu_i + mu * 8)),
                    acc);
                _mm512_storeu_pd(gp, acc);
            }
        }
    }
    if (st) {
        const long long t7 = __rdtsc();
        st[1] += t1 - t0;      /* grid + weights */
        st[2] += t2 - t1;      /* ka             */
        st[3] += t3 - t2;      /* fa             */
        st[4] += t4 - t3;      /* fl             */
        st[5] += t5 - t4;      /* num            */
        st[6] += t6 - t5;      /* den            */
        st[7] += t7 - t6;      /* fwd + bwd      */
    }
}

/* ── accumulate a segment's gradient with the now-known dQ/dP_e ──────── */
static void cfpg_seg_acc(CFPGCtx *c, double *t, __m512d dqp, int so, int n,
                         long long *st)
{
    const long long t0 = st ? __rdtsc() : 0;
    const int N = c->N, NB = c->NB, cap = c->cap;
    const CFPGLayout L = c->L;
    const double *seg_vr = t + L.seg_vr, *seg_vi = t + L.seg_vi;
    const double *seg_g = t + L.seg_g;
    double *acr = t + L.acc_ckr, *aci = t + L.acc_cki, *acg = t + L.acc_g;
    dqp = _mm512_maskz_mov_pd(cfpg_mask(n), dqp);
    for (int k = 0; k < N; k++) {
        double *pr = acr + k * 8, *pi = aci + k * 8;
        _mm512_storeu_pd(pr, _mm512_fmadd_pd(
            dqp, _mm512_loadu_pd(seg_vr + (size_t)k * cap + so),
            _mm512_loadu_pd(pr)));
        _mm512_storeu_pd(pi, _mm512_fmadd_pd(
            dqp, _mm512_loadu_pd(seg_vi + (size_t)k * cap + so),
            _mm512_loadu_pd(pi)));
    }
    for (int mm = 0; mm < NB; mm++) {
        double *pg = acg + mm * 8;
        _mm512_storeu_pd(pg, _mm512_fmadd_pd(
            dqp, _mm512_loadu_pd(seg_g + (size_t)mm * cap + so),
            _mm512_loadu_pd(pg)));
    }
    if (st) st[8] += __rdtsc() - t0;
}

void cpu_flat_pwa_gen_free_ctx(void *v);   /* used by create's failure paths */

/* ══════════════════════════════════════════════════════════════════════
 * create / free / load
 * ══════════════════════════════════════════════════════════════════════ */
static int cfpg_is_const_row(const double *a, const double *b, int bins)
{
    for (int i = 1; i < bins; i++)
        if (a[i] != a[0] || b[i] != b[0]) return 0;
    return 1;
}

void* cpu_flat_pwa_gen_create(
    const double *fl_tab, int n_fl_rows, int n_fl_bins,
    const int *fl_u_row, const int *fl_u_col, int n_fl_u,
    const int *fl_use,
    const double *num_r, const double *num_i, int n_num_rows,
    const int *num_u_row, const int *num_u_col, int n_num_u,
    const int *num_use,
    const double *bas_r, const double *bas_i, int n_bas_rows,
    const int *us_row, const int *us_col, int n_uslot,
    const int *g_off, const int *g_m, const int *g_u, const double *g_w,
    int n_nz,
    const int *angle_index, const double *angle_k, const double *angle_b,
    int n_ab, int n_comp,
    const int *fa_ptr, const int *fa_row, const double *fa_r,
    const double *fa_i, const int *fa_slot, int n_fa,
    double m_min, double m_delta, int m_bins,
    double q_min, double q_delta, int q_bins,
    int n_wave, int n_proj, int n_res, int n_decay, int n_basis,
    int n_mass, int n_momentum, int n_ang_flat,
    int rsize, int batch_size, int flags)
{
    (void)n_num_rows; (void)n_bas_rows; (void)batch_size;
    if (!fl_tab || !num_r || !num_i || !bas_r || !bas_i || !fa_r || !fa_i ||
        !angle_index || !angle_k || !angle_b)
        return NULL;
    if (n_wave <= 0 || n_proj <= 0 || n_res <= 0 || n_decay <= 0 ||
        n_basis <= 0 || n_fl_u <= 0 || n_uslot <= 0 || n_ab <= 0 ||
        n_fa <= 0 || n_fl_rows <= 0 || n_num_u <= 0 || n_ang_flat <= 0 ||
        m_bins < 2 || q_bins < 2 || n_comp <= 0 || n_mass <= 0 ||
        n_momentum <= 0 || n_fl_bins < 2)
        return NULL;
    cfpg_vmath_init();
    CFPGCtx *c = (CFPGCtx*)calloc(1, sizeof(CFPGCtx));
    if (!c) return NULL;

    /* ── the compacted angular basis ────────────────────────────────────
     * cos(k·angle + b): k == 0 components never need an angle (they are the
     * constant cos(b), folded into `ach_aconst`), everything else keeps its
     * (column, component, k, cos b, sin b). */
    int n_col = 0;
    for (int a = 0; a < n_ab; a++)
        if (angle_index[a] + 1 > n_col) n_col = angle_index[a] + 1;
    c->n_ach = n_ab * n_comp;
    c->ach_off = (int*)calloc((size_t)n_ab + 1, sizeof(int));
    c->ach_col = (int*)calloc((size_t)c->n_ach, sizeof(int));
    c->ach_j = (int*)calloc((size_t)c->n_ach, sizeof(int));
    c->ach_pair = (int*)calloc((size_t)c->n_ach, sizeof(int));
    c->ach_k = (double*)calloc((size_t)c->n_ach, sizeof(double));
    c->ach_kabs = (int*)calloc((size_t)c->n_ach, sizeof(int));
    c->ach_sgn = (int*)calloc((size_t)c->n_ach, sizeof(int));
    c->ach_b = (double*)calloc((size_t)c->n_ach, sizeof(double));
    c->ach_cb = (double*)calloc((size_t)c->n_ach, sizeof(double));
    c->ach_sb = (double*)calloc((size_t)c->n_ach, sizeof(double));
    c->ach_aconst = (double*)calloc((size_t)n_ab, sizeof(double));
    if (!c->ach_off || !c->ach_col || !c->ach_j || !c->ach_pair ||
        !c->ach_k ||
        !c->ach_kabs || !c->ach_sgn || !c->ach_b || !c->ach_cb ||
        !c->ach_sb || !c->ach_aconst) {
        free(c);
        return NULL;
    }
    int int_k = 1, n_ach = 0;
    for (int a = 0; a < n_ab; a++) {
        double ac = 1.0;
        c->ach_off[a] = n_ach;
        for (int j = 0; j < n_comp; j++) {
            const double k = angle_k[a * n_comp + j];
            const double b = angle_b[a * n_comp + j];
            const double rk = (double)(long long)(k < 0 ? -floor(-k + 0.5)
                                                       : floor(k + 0.5));
            if (k != rk || fabs(k) > CFPG_MAX_K) int_k = 0;
            if (k == 0.0) {          /* the constant cos(b), exactly */
                ac *= cos(b);        /* cos(0·angle + b) = cos(b) */
                continue;
            }
            c->ach_col[n_ach] = angle_index[a];
            c->ach_pair[n_ach] = angle_index[a] * n_comp + j;
            c->ach_j[n_ach] = j;
            c->ach_k[n_ach] = k;
            c->ach_kabs[n_ach] = (int)fabs(k);
            c->ach_sgn[n_ach] = (k > 0.0) ? 1 : -1;
            c->ach_b[n_ach] = b;
            c->ach_cb[n_ach] = cos(b);
            c->ach_sb[n_ach] = sin(b);
            n_ach++;
        }
        c->ach_aconst[a] = ac;
    }
    c->ach_off[n_ab] = n_ach;
    c->n_ach = n_ach;
    c->int_k = (int_k && !(flags & CFPG_FLAG_GENERAL_COS)) ? 1 : 0;

    c->fl_tab = fl_tab;  c->fl_bins = n_fl_bins;
    c->fl_u_row = fl_u_row; c->fl_u_col = fl_u_col; c->n_fl_u = n_fl_u;
    c->fl_use = fl_use;
    c->num_r = num_r; c->num_i = num_i; c->m_bins = m_bins;
    c->num_u_row = num_u_row; c->num_u_col = num_u_col; c->n_num_u = n_num_u;
    c->num_use = num_use;
    c->bas_r = bas_r; c->bas_i = bas_i;
    c->us_row = us_row; c->us_col = us_col; c->n_uslot = n_uslot;
    c->us_const = (unsigned char*)calloc((size_t)n_uslot, 1);
    c->us_cr = (double*)calloc((size_t)n_uslot, sizeof(double));
    c->us_ci = (double*)calloc((size_t)n_uslot, sizeof(double));
    if (!c->us_const || !c->us_cr || !c->us_ci) {
        cpu_flat_pwa_gen_free_ctx(c);
        return NULL;
    }
    for (int u = 0; u < n_uslot; u++)
        if (cfpg_is_const_row(bas_r + (size_t)us_row[u] * m_bins,
                              bas_i + (size_t)us_row[u] * m_bins, m_bins)) {
            c->us_const[u] = 1;
            c->us_cr[u] = bas_r[(size_t)us_row[u] * m_bins];
            c->us_ci[u] = bas_i[(size_t)us_row[u] * m_bins];
        }
    c->g_off = g_off; c->g_m = g_m; c->g_u = g_u; c->g_w = g_w;
    c->n_nz = n_nz;
    c->angle_index = angle_index;
    c->angle_k = angle_k; c->angle_b = angle_b;
    c->n_ab = n_ab; c->n_comp = n_comp; c->n_col = n_col;
    c->n_aj = n_col * n_comp;
    c->fa_ptr = fa_ptr; c->fa_row = fa_row;
    c->fa_r = fa_r; c->fa_i = fa_i;
    c->n_fa = n_fa; c->fa_slot = fa_slot;
    c->m_min = m_min; c->m_delta = m_delta;
    c->q_min = q_min; c->q_delta = q_delta;
    c->N = n_wave; c->P = n_proj; c->R = n_res; c->D = n_decay;
    c->NB = n_basis;
    c->n_mass = n_mass; c->n_momentum = n_momentum;
    c->rsize = (rsize > 0) ? rsize : 1;
    /* the group buffer holds rsize events; beyond the cap a group is
       closed in a two-pass fallback (same numbers, 2x the work) */
    c->cap = (c->rsize < CFPG_LANES) ? CFPG_LANES : c->rsize;
    if (c->cap > CFPG_MAXCAP) c->cap = CFPG_MAXCAP;
    c->n_threads = omp_get_max_threads();
    if (c->n_threads < 1) c->n_threads = 1;
    c->L = cfpg_layout(n_ab, n_fa, n_wave, n_proj, n_res, n_uslot, n_basis,
                       n_mass, n_momentum, n_fl_u, n_num_u, c->n_aj, c->cap);
    c->I.midx = 0;
    c->I.qidx = n_mass * 32;
    c->I.total = (n_mass + n_momentum) * 32;
    c->scratch = (double*)calloc((size_t)c->n_threads * c->L.total,
                                 sizeof(double));
    c->iscratch = (int*)calloc((size_t)c->n_threads * c->I.total,
                               sizeof(int));
    c->w_nz = (double*)calloc((size_t)(n_nz > 0 ? n_nz : 1), sizeof(double));
    if (!c->scratch || !c->iscratch || !c->w_nz) {
        cpu_flat_pwa_gen_free_ctx(c);
        return NULL;
    }
    return c;
}

void cpu_flat_pwa_gen_free_ctx(void *v)
{
    CFPGCtx *c = (CFPGCtx*)v;
    if (!c) return;
    free(c->ach_off); free(c->ach_col); free(c->ach_j);
    free(c->ach_pair); free(c->ach_k);
    free(c->ach_kabs); free(c->ach_sgn); free(c->ach_b);
    free(c->ach_cb); free(c->ach_sb);
    free(c->ach_aconst);
    free(c->us_const); free(c->us_cr); free(c->us_ci);
    free(c->scratch);
    free(c->iscratch);
    free(c->w_nz);
    free(c);
}

void* cpu_flat_pwa_gen_load(void *v, const double *mass_s, const double *q_s,
                            const double *ang_s, const double *weight,
                            const double *bkg, int ne)
{
    CFPGCtx *c = (CFPGCtx*)v;
    if (!c) return NULL;
    CFPGData *h = (CFPGData*)calloc(1, sizeof(CFPGData));
    if (!h) return NULL;
    h->ne = ne;
    if (ne <= 0) return h;
    h->mass_s = mass_s; h->q_s = q_s; h->ang_s = ang_s;
    h->weight = weight; h->bkg = bkg;
    h->P = (double*)calloc((size_t)ne, sizeof(double));
    if (!h->P) { free(h); return NULL; }
    return h;
}

void cpu_flat_pwa_gen_free(void *v)
{
    CFPGData *h = (CFPGData*)v;
    if (!h) return;
    free(h->P);
    free(h);
}

/* ══════════════════════════════════════════════════════════════════════
 * compute — the whole objective and its gradients, one pass
 * ══════════════════════════════════════════════════════════════════════ */
double cpu_flat_pwa_gen_compute(
    void *vc, void *vh,
    const double *coeff, const double *ck_r, const double *ck_i,
    double norm, int use_norm,
    double *g_ck_r, double *g_ck_i, double *g_coeff,
    double *P_out, int return_p, double *dnorm_out,
    long long *stage)
{
    CFPGCtx *c = (CFPGCtx*)vc;
    CFPGData *h = (CFPGData*)vh;
    const int N = c->N, NB = c->NB, cap = c->cap;
    if (stage) for (int s = 0; s < CFPG_NSTAGE; s++) stage[s] = 0;
    memset(g_ck_r, 0, sizeof(double) * (size_t)N);
    memset(g_ck_i, 0, sizeof(double) * (size_t)N);
    memset(g_coeff, 0, sizeof(double) * (size_t)NB);
    *dnorm_out = 0.0;
    if (!h || h->ne <= 0) return 0.0;
    const int ne = h->ne;
    const long long tW = stage ? __rdtsc() : 0;

    /* w = gamma·coeff: the only parameter-dependent table, once per call */
    for (int i = 0; i < c->n_nz; i++)
        c->w_nz[i] = c->g_w[i] * coeff[c->g_m[i]];
    if (stage) stage[0] += __rdtsc() - tW;

    const int pathA = (!use_norm) || (c->rsize == 1);
    const int ngroups = (ne + c->rsize - 1) / c->rsize;
    const int ntiles = (ne + CFPG_LANES - 1) / CFPG_LANES;
    const int nth = omp_get_max_threads() < c->n_threads
                    ? omp_get_max_threads() : c->n_threads;
    double *tQ = (double*)calloc((size_t)nth, sizeof(double));
    double *tD = (double*)calloc((size_t)nth, sizeof(double));
    if (!tQ || !tD) { free(tQ); free(tD); return 0.0; }

    #pragma omp parallel num_threads(nth)
    {
        const int tid = omp_get_thread_num();
        double *t = c->scratch + (size_t)tid * c->L.total;
        int *it = c->iscratch + (size_t)tid * c->I.total;
        const CFPGLayout L = c->L;
        double *dqp = t + L.dqp;
        long long *st = (stage && tid == 0) ? stage : NULL;
        double Q = 0.0, Dn = 0.0;
        memset(t + L.acc_ckr, 0, sizeof(double) * (size_t)(N * 8));
        memset(t + L.acc_cki, 0, sizeof(double) * (size_t)(N * 8));
        memset(t + L.acc_g, 0, sizeof(double) * (size_t)(NB * 8));

        if (pathA) {
            /* one event at a time: dQ/dP closes inside the tile */
            #pragma omp for schedule(static)
            for (int tile = 0; tile < ntiles; tile++) {
                const int e0 = tile * CFPG_LANES;
                const int n = (ne - e0 < CFPG_LANES) ? ne - e0 : CFPG_LANES;
                const __mmask8 m = cfpg_mask(n);
                cfpg_tile(c, h, t, it, e0, n, ck_r, ck_i, 0, 1, st);
                const __m512d wv = _mm512_maskz_loadu_pd(m, h->weight + e0);
                const __m512d bv = _mm512_maskz_loadu_pd(m, h->bkg + e0);
                const __m512d P = _mm512_maskz_loadu_pd(m, h->P + e0);
                __m512d dq;
                if (!use_norm) {
                    Q += _mm512_mask_reduce_add_pd(m, _mm512_mul_pd(wv, P));
                    dq = wv;
                } else {
                    const __m512d nv = _mm512_set1_pd(norm);
                    /* pdf = P/norm + b = (P + b·norm)/norm */
                    const __m512d den = _mm512_fmadd_pd(bv, nv, P);
                    Q -= _mm512_mask_reduce_add_pd(
                        m, _mm512_mul_pd(wv, cfpg_log8(
                               _mm512_div_pd(den, nv))));
                    dq = _mm512_maskz_mov_pd(
                        m, _mm512_div_pd(_mm512_set1_pd(-1.0), den));
                    dq = _mm512_mul_pd(dq, wv);
                    Dn += _mm512_mask_reduce_add_pd(
                        m, _mm512_div_pd(_mm512_mul_pd(wv, P),
                                         _mm512_mul_pd(nv, den)));
                }
                cfpg_seg_acc(c, t, dq, 0, n, st);
            }
        } else {
            /* grouped log-sum: one work unit = one group.  The per-event
               (v, g) live in the segment buffer, so the group closes with
               no second walk over the data. */
            #pragma omp for schedule(static)
            for (int g = 0; g < ngroups; g++) {
                const int e0 = g * c->rsize;
                const int neg = (ne - e0 < c->rsize) ? ne - e0 : c->rsize;
                const int one_pass = (neg <= cap);
                if (one_pass) {
                    for (int so = 0; so < neg; so += CFPG_LANES) {
                        const int n = (neg - so < CFPG_LANES)
                                      ? neg - so : CFPG_LANES;
                        cfpg_tile(c, h, t, it, e0 + so, n, ck_r, ck_i, so, 1,
                                  st);
                    }
                } else {
                    /* rsize beyond the group buffer: forward, close, then
                       recompute the forward to collect v/g (same numbers) */
                    for (int so = 0; so < neg; so += CFPG_LANES) {
                        const int n = (neg - so < CFPG_LANES)
                                      ? neg - so : CFPG_LANES;
                        cfpg_tile(c, h, t, it, e0 + so, n, ck_r, ck_i, 0, 0,
                                  st);
                    }
                }
                double S = 0.0, Pb = 0.0, W = 0.0;
                for (int e = 0; e < neg; e++) {
                    const double we = h->weight[e0 + e];
                    const double pe = h->P[e0 + e];
                    S += we * (pe / norm + h->bkg[e0 + e]);
                    Pb += we * pe;
                    W += we;
                }
                const int safe = (W > 0.0);
                if (safe) {
                    Q -= W * log(S / W);
                    Dn += W * Pb / (S * norm * norm);
                }
                const double scale = safe ? -W / S : 0.0;
                const int step = one_pass ? CFPG_LANES : cap;
                for (int so = 0; so < neg; so += step) {
                    const int nsl = (neg - so < step) ? neg - so : step;
                    if (!one_pass) {
                        for (int sl = 0; sl < nsl; sl += CFPG_LANES) {
                            const int nn = (nsl - sl < CFPG_LANES)
                                           ? nsl - sl : CFPG_LANES;
                            cfpg_tile(c, h, t, it, e0 + so + sl, nn, ck_r,
                                      ck_i, sl, 1, st);
                        }
                    }
                    for (int sl = 0; sl < nsl; sl += CFPG_LANES) {
                        const int nn = (nsl - sl < CFPG_LANES)
                                       ? nsl - sl : CFPG_LANES;
                        const __mmask8 m = cfpg_mask(nn);
                        for (int e = 0; e < nn; e++)
                            dqp[e] = scale * h->weight[e0 + so + sl + e] / norm;
                        cfpg_seg_acc(c, t, _mm512_maskz_loadu_pd(m, dqp),
                                     so + sl, nn, st);
                    }
                }
            }
        }
        tQ[tid] = Q;
        tD[tid] = Dn;
    }

    double Q = 0.0, Dn = 0.0;
    for (int i = 0; i < nth; i++) { Q += tQ[i]; Dn += tD[i]; }
    free(tQ);
    free(tD);
    /* per-thread vector accumulators -> the caller's flat arrays (fixed
       order: thread index, then wave / basis row) */
    for (int tid = 0; tid < nth; tid++) {
        const double *t = c->scratch + (size_t)tid * c->L.total;
        for (int k = 0; k < N; k++) {
            g_ck_r[k] += _mm512_reduce_add_pd(
                             _mm512_loadu_pd(t + c->L.acc_ckr + k * 8));
            g_ck_i[k] += _mm512_reduce_add_pd(
                             _mm512_loadu_pd(t + c->L.acc_cki + k * 8));
        }
        for (int mm = 0; mm < NB; mm++)
            g_coeff[mm] += _mm512_reduce_add_pd(
                               _mm512_loadu_pd(t + c->L.acc_g + mm * 8));
    }
    *dnorm_out = Dn;
    if (return_p && P_out) memcpy(P_out, h->P, sizeof(double) * (size_t)ne);
    return Q;
}

/* ══════════════════════════════════════════════════════════════════════
 * debug dump — the tile intermediates of the first 8 events (fixture only)
 * ══════════════════════════════════════════════════════════════════════ */
int cpu_flat_pwa_gen_debug(
    void *vc, void *vh, const double *coeff, const double *ck_r,
    const double *ck_i, double *out)
{
    CFPGCtx *c = (CFPGCtx*)vc;
    CFPGData *h = (CFPGData*)vh;
    const CFPGLayout L = c->L;
    double *t = c->scratch;
    int *it = c->iscratch;
    for (int i = 0; i < c->n_nz; i++)
        c->w_nz[i] = c->g_w[i] * coeff[c->g_m[i]];
    const int n = (h->ne < CFPG_LANES) ? h->ne : CFPG_LANES;
    cfpg_tile(c, h, t, it, 0, n, ck_r, ck_i, 0, 1, NULL);
    int o = 0;
    const int cnt[15] = {
        c->n_ab * 8, c->n_fa * 8, c->n_fa * 8, c->n_fl_u * 8, c->N * 8,
        c->n_num_u * 8, c->n_num_u * 8, c->n_uslot * 8, c->n_uslot * 8,
        c->N * 8, c->N * 8, c->N * 8, c->N * 8, c->P * 8, c->P * 8};
    const double *src[15] = {
        t + L.ka, t + L.fa_r, t + L.fa_i, t + L.flv, t + L.flw,
        t + L.nuv_r, t + L.nuv_i, t + L.Fu_r, t + L.Fu_i,
        t + L.den_r, t + L.den_i, t + L.cm_r, t + L.cm_i, t + L.Ar, t + L.Ai};
    for (int k = 0; k < 15; k++) {
        memcpy(out + o, src[k], sizeof(double) * (size_t)cnt[k]);
        o += cnt[k];
    }
    memcpy(out + o, h->P, sizeof(double) * (size_t)n);
    o += n;
    return o;
}
