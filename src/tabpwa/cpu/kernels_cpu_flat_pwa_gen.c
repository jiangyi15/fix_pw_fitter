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
 * A tile is 8 events and every per-event quantity is a vector of them.  The
 * WIDTH is a compile-time abstraction (`vpd` / `CFPG_LANES`, see the block
 * after the includes): one zmm on AVX-512, two ymm halves on AVX2+FMA.  The
 * CFPG_LANES LOGICAL lanes are the same in both builds, so the tile boundaries, the
 * parallel partition, the lane -> event mapping and every reduction's
 * grouping are identical and the two builds are bit-identical (gated in
 * `tests/test_cpu_flat_pwa_gen.py`).  Concretely:
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
 * Two builds, ONE algorithm: AVX-512 (one zmm) and AVX2+FMA (two ymm halves),
 * bit-identical by construction — see THE WIDTH below.  MEASURED on the
 * production shape (8.7 M events, rsize 17, `Fitter.get_nll`, 32 threads):
 * AVX-512 411 ms / 21.3 Mev/s (data pass) and 466 ms / 22.0 Mev/s (all
 * events); AVX2+FMA 455 ms / 19.2 and 520 ms / 19.7 — i.e. only ~10 % apart,
 * because the kernel is gather-bound, not FLOP-bound.
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
 * CFPG_LANES = accumulate) when `stage != NULL`; the benchmark harness scales them
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

/*: the per-group SCALAR arithmetic (S/Pb/W, `scale`, `dqp`) must round the
 *  same way in both builds.  GCC contracts `a*b + c` into an FMA under
 *  `-ffp-contract=fast` (the default), but WHICH scalar loops it contracts
 *  differs between the AVX-512 and the AVX2 cost model — measured: the two
 *  builds then disagree on Q by 1 ulp at ne = 4 and on dck/dcoeff by 1 ulp
 *  at ne = 5.  Turning contraction off for this file costs nothing (the
 *  vector path uses explicit fma intrinsics, unaffected by the flag) and
 *  makes the two builds bit-identical. */
#pragma GCC optimize ("fp-contract=off")

#define CFPG_MAXCAP  1024   /* per-thread group buffer (events); rsize cap */
#define CFPG_NSTAGE  9
#define CFPG_MAX_K   8      /* |k| ceiling of the Chebyshev fast path      */
/*: force the general (libmvec `cos`) angle basis instead of the integral-k
 *  Chebyshev path — a test/benchmark hook, never used by the wrapper */
#define CFPG_FLAG_GENERAL_COS 1

/* ══════════════════════════════════════════════════════════════════════
 * THE WIDTH — ONE algorithm, TWO builds.
 *
 * `vpd` is a vector of `CFPG_LANES` (= 8) events.  AVX-512 puts it in one
 * zmm; AVX2+FMA puts it in two ymm halves.  Everything below is written
 * once, against `vpd`, so there is no second implementation to keep in
 * sync — and because the number of LOGICAL lanes is 8 in BOTH builds, the
 * lane -> event mapping, the tile boundaries and every reduction's
 * grouping are identical too: the two builds are bit-identical, not merely
 * close (a 4-lane TILE would regroup Q and the gradient accumulators by
 * event%4 instead of event%8 and differ at ~1e-16; the per-lane arithmetic
 * is unaffected either way).
 *
 * Three things are deliberately NOT the hardware width:
 *   * `vgather` zeroes the lanes past `nl` in BOTH builds (the AVX-512
 *     masked gather does it; AVX2 applies the same mask afterwards), so an
 *     inactive lane carries 0 through the whole chain instead of whatever
 *     the clamped index happened to hit;
 *   * `vhsum`/`vhsum_n` reduce with ONE explicit tree, written out, in
 *     both builds (not `_mm512_reduce_add_pd`'s macro, whose association is
 *     a header detail);
 *   * `cfpg_log8` calls the 4-WIDE libmvec entry in both builds: glibc's
 *     8-wide `log` is not bit-equal to its 4-wide one (measured 5040 of
 *     1.6 M samples differ), while its 8-wide `cos` is (0 of 1.6 M), so
 *     `cos`/`sin` may use each build's own width and `log` may not.
 * ══════════════════════════════════════════════════════════════════════ */
#if defined(__AVX512F__) && defined(__AVX512VL__)
#define CFPG_ISA      1                    /* 1 = one zmm per vpd        */
#define CFPG_ISA_NAME "avx512f"
#define CFPG_LANES    8                    /* logical lanes = events/tile */
typedef __m512d vpd;
typedef __m256i vi32;

static inline __mmask8 cfpg_mask(int nl) { return (__mmask8)((1u << nl) - 1u); }
static inline vpd vzero(void) { return _mm512_setzero_pd(); }
static inline vpd vset1(double x) { return _mm512_set1_pd(x); }
static inline vpd vload(const double *p) { return _mm512_loadu_pd(p); }
static inline void vstore(double *p, vpd v) { _mm512_storeu_pd(p, v); }
static inline vpd vloadn(int nl, const double *p)
{ return _mm512_maskz_loadu_pd(cfpg_mask(nl), p); }
static inline void vstoren(double *p, int nl, vpd v)
{ _mm512_mask_storeu_pd(p, cfpg_mask(nl), v); }
static inline vpd vmaskz(int nl, vpd v)
{ return _mm512_maskz_mov_pd(cfpg_mask(nl), v); }
static inline vpd vadd(vpd a, vpd b) { return _mm512_add_pd(a, b); }
static inline vpd vsub(vpd a, vpd b) { return _mm512_sub_pd(a, b); }
static inline vpd vmul(vpd a, vpd b) { return _mm512_mul_pd(a, b); }
static inline vpd vdiv(vpd a, vpd b) { return _mm512_div_pd(a, b); }
static inline vpd vmin(vpd a, vpd b) { return _mm512_min_pd(a, b); }
static inline vpd vmax(vpd a, vpd b) { return _mm512_max_pd(a, b); }
static inline vpd vfma(vpd a, vpd b, vpd c) { return _mm512_fmadd_pd(a, b, c); }
static inline vpd vfmsub(vpd a, vpd b, vpd c)
{ return _mm512_fmsub_pd(a, b, c); }
static inline vpd vfnmadd(vpd a, vpd b, vpd c)
{ return _mm512_fnmadd_pd(a, b, c); }
static inline vpd vgather(const int *idx, const double *base, int nl)
{ return _mm512_mask_i32gather_pd(_mm512_setzero_pd(), cfpg_mask(nl),
                                  _mm256_loadu_si256((const __m256i*)idx),
                                  base, 8); }
static inline vi32 vfloor_i(vpd v)
{ return _mm512_cvtpd_epi32(_mm512_floor_pd(v)); }
static inline vpd vi2pd(vi32 v) { return _mm512_cvtepi32_pd(v); }
static inline vi32 vizero(void) { return _mm256_setzero_si256(); }
static inline vi32 vi_max(vi32 a, int k)
{ return _mm256_max_epi32(a, _mm256_set1_epi32(k)); }
static inline vi32 vi_min(vi32 a, int k)
{ return _mm256_min_epi32(a, _mm256_set1_epi32(k)); }
static inline vi32 vi_add(vi32 a, int k)
{ return _mm256_add_epi32(a, _mm256_set1_epi32(k)); }
static inline vi32 vi_sub(vi32 a, int k)
{ return _mm256_sub_epi32(a, _mm256_set1_epi32(k)); }
static inline void vstorei(int *p, vi32 v) { _mm256_storeu_si256((__m256i*)p, v); }
static inline void vsplit(vpd v, __m256d *lo, __m256d *hi)
{ *lo = _mm512_castpd512_pd256(v); *hi = _mm512_extractf64x4_pd(v, 1); }
static inline vpd vjoin(__m256d lo, __m256d hi)
{ return _mm512_insertf64x4(_mm512_castpd256_pd512(lo), hi, 1); }

#elif defined(__AVX2__) && defined(__FMA__)
#define CFPG_ISA      0                    /* 0 = two ymm halves per vpd */
#define CFPG_ISA_NAME "avx2"
#define CFPG_LANES    8                    /* logical lanes = events/tile */
typedef struct { __m256d a, b; } vpd;
typedef struct { __m128i a, b; } vi32;

/*: lane mask for `k` live lanes of a 4-lane half (0..4) — the AVX2
 *  masked load/store/AND masks, so a partial tile never reads past `ne`
 *  and never keeps a value in a lane the caller did not ask for. */
static inline __m256i cfpg_m4(int k)
{
    static const long long bits[5][4] = {
        {0, 0, 0, 0}, {-1, 0, 0, 0}, {-1, -1, 0, 0},
        {-1, -1, -1, 0}, {-1, -1, -1, -1}};
    if (k < 0) k = 0;
    if (k > 4) k = 4;
    return _mm256_loadu_si256((const __m256i*)bits[k]);
}
static inline vpd vzero(void)
{ vpd r; r.a = r.b = _mm256_setzero_pd(); return r; }
static inline vpd vset1(double x)
{ vpd r; r.a = r.b = _mm256_set1_pd(x); return r; }
static inline vpd vload(const double *p)
{ vpd r; r.a = _mm256_loadu_pd(p); r.b = _mm256_loadu_pd(p + 4); return r; }
static inline void vstore(double *p, vpd v)
{ _mm256_storeu_pd(p, v.a); _mm256_storeu_pd(p + 4, v.b); }
static inline vpd vloadn(int nl, const double *p)
{
    vpd r;
    r.a = _mm256_maskload_pd(p, cfpg_m4(nl));
    r.b = _mm256_maskload_pd(p + 4, cfpg_m4(nl - 4));
    return r;
}
static inline void vstoren(double *p, int nl, vpd v)
{
    _mm256_maskstore_pd(p, cfpg_m4(nl), v.a);
    _mm256_maskstore_pd(p + 4, cfpg_m4(nl - 4), v.b);
}
static inline vpd vmaskz(int nl, vpd v)
{
    vpd r;
    r.a = _mm256_and_pd(v.a, _mm256_castsi256_pd(cfpg_m4(nl)));
    r.b = _mm256_and_pd(v.b, _mm256_castsi256_pd(cfpg_m4(nl - 4)));
    return r;
}
static inline vpd vadd(vpd x, vpd y)
{ vpd r; r.a = _mm256_add_pd(x.a, y.a); r.b = _mm256_add_pd(x.b, y.b); return r; }
static inline vpd vsub(vpd x, vpd y)
{ vpd r; r.a = _mm256_sub_pd(x.a, y.a); r.b = _mm256_sub_pd(x.b, y.b); return r; }
static inline vpd vmul(vpd x, vpd y)
{ vpd r; r.a = _mm256_mul_pd(x.a, y.a); r.b = _mm256_mul_pd(x.b, y.b); return r; }
static inline vpd vdiv(vpd x, vpd y)
{ vpd r; r.a = _mm256_div_pd(x.a, y.a); r.b = _mm256_div_pd(x.b, y.b); return r; }
static inline vpd vmin(vpd x, vpd y)
{ vpd r; r.a = _mm256_min_pd(x.a, y.a); r.b = _mm256_min_pd(x.b, y.b); return r; }
static inline vpd vmax(vpd x, vpd y)
{ vpd r; r.a = _mm256_max_pd(x.a, y.a); r.b = _mm256_max_pd(x.b, y.b); return r; }
static inline vpd vfma(vpd x, vpd y, vpd z)
{ vpd r; r.a = _mm256_fmadd_pd(x.a, y.a, z.a);
  r.b = _mm256_fmadd_pd(x.b, y.b, z.b); return r; }
static inline vpd vfmsub(vpd x, vpd y, vpd z)
{ vpd r; r.a = _mm256_fmsub_pd(x.a, y.a, z.a);
  r.b = _mm256_fmsub_pd(x.b, y.b, z.b); return r; }
static inline vpd vfnmadd(vpd x, vpd y, vpd z)
{ vpd r; r.a = _mm256_fnmadd_pd(x.a, y.a, z.a);
  r.b = _mm256_fnmadd_pd(x.b, y.b, z.b); return r; }
static inline vpd vgather(const int *idx, const double *base, int nl)
{
    vpd r;
    r.a = _mm256_i32gather_pd(base, _mm_loadu_si128((const __m128i*)idx), 8);
    r.b = _mm256_i32gather_pd(base, _mm_loadu_si128((const __m128i*)(idx + 4)),
                              8);
    return vmaskz(nl, r);            /* 0 past `nl`, exactly as AVX-512 */
}
static inline vi32 vfloor_i(vpd v)
{
    vi32 r;
    r.a = _mm256_cvtpd_epi32(_mm256_floor_pd(v.a));
    r.b = _mm256_cvtpd_epi32(_mm256_floor_pd(v.b));
    return r;
}
static inline vpd vi2pd(vi32 v)
{ vpd r; r.a = _mm256_cvtepi32_pd(v.a); r.b = _mm256_cvtepi32_pd(v.b); return r; }
static inline vi32 vizero(void)
{ vi32 r; r.a = r.b = _mm_setzero_si128(); return r; }
static inline vi32 vi_max(vi32 v, int k)
{
    vi32 r; const __m128i kk = _mm_set1_epi32(k);
    r.a = _mm_max_epi32(v.a, kk); r.b = _mm_max_epi32(v.b, kk); return r;
}
static inline vi32 vi_min(vi32 v, int k)
{
    vi32 r; const __m128i kk = _mm_set1_epi32(k);
    r.a = _mm_min_epi32(v.a, kk); r.b = _mm_min_epi32(v.b, kk); return r;
}
static inline vi32 vi_add(vi32 v, int k)
{
    vi32 r; const __m128i kk = _mm_set1_epi32(k);
    r.a = _mm_add_epi32(v.a, kk); r.b = _mm_add_epi32(v.b, kk); return r;
}
static inline vi32 vi_sub(vi32 v, int k)
{
    vi32 r; const __m128i kk = _mm_set1_epi32(k);
    r.a = _mm_sub_epi32(v.a, kk); r.b = _mm_sub_epi32(v.b, kk); return r;
}
static inline void vstorei(int *p, vi32 v)
{ _mm_storeu_si128((__m128i*)p, v.a); _mm_storeu_si128((__m128i*)(p + 4), v.b); }
static inline void vsplit(vpd v, __m256d *lo, __m256d *hi)
{ *lo = v.a; *hi = v.b; }
static inline vpd vjoin(__m256d lo, __m256d hi)
{ vpd r; r.a = lo; r.b = hi; return r; }

#else
#error "cpu_flat_pwa_gen needs AVX2+FMA (AVX-512 is used when the compiler enables it, e.g. -march=native on a Zen4/modern-Intel host)"
#endif

/*: the canonical horizontal reduction — ONE tree, written out, used by both
 *  builds (and by `vhsum_n` after zeroing the lanes past `nl`, which is what
 *  the AVX-512 masked reduce does internally).  Order matters: it is part of
 *  the bit-identity contract, so it may not be an intrinsic whose tree is a
 *  header detail. */
static inline double vhsum(vpd v)
{
    double t[CFPG_LANES];
    vstore(t, v);
    return ((t[0] + t[4]) + (t[2] + t[6])) + ((t[1] + t[5]) + (t[3] + t[7]));
}

static inline double vhsum_n(int nl, vpd v) { return vhsum(vmaskz(nl, v)); }

/* ── libmvec (glibc's vectorised cos/log), resolved with dlsym so that a
 *    build without `-lmvec` still loads and falls back to scalar libm ── */
typedef vpd (*cfpg_vfn8)(vpd);
typedef __m256d (*cfpg_vfn4)(__m256d);
static cfpg_vfn8 cfpg_vcos8 = NULL;
static cfpg_vfn4 cfpg_vcos4 = NULL;
static cfpg_vfn4 cfpg_vlog4 = NULL;
static int cfpg_vmath_ok = 0;

static void cfpg_vmath_init(void)
{
    void *h = dlopen("libmvec.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!h) h = dlopen("libmvec.so", RTLD_LAZY | RTLD_GLOBAL);
    if (h) {
        cfpg_vcos4 = (cfpg_vfn4)(void (*)(void))dlsym(h, "_ZGVdN4v_cos");
        cfpg_vlog4 = (cfpg_vfn4)(void (*)(void))dlsym(h, "_ZGVdN4v_log");
#if CFPG_ISA
        cfpg_vcos8 = (cfpg_vfn8)(void (*)(void))dlsym(h, "_ZGVeN8v_cos");
#endif
    }
    cfpg_vmath_ok = (cfpg_vcos4 != NULL && cfpg_vlog4 != NULL
#if CFPG_ISA
                     && cfpg_vcos8 != NULL
#endif
                     ) ? 1 : 0;
}

/*: cos of 8 events.  Each build uses its own width: the 8-wide and 4-wide
 *  libmvec `cos` are bit-equal (measured over the production argument range,
 *  0 of 1.6 M samples differ), so this is the same number either way. */
static inline vpd cfpg_cos8(vpd v)
{
    if (__builtin_expect(cfpg_vmath_ok == 1, 1)) {
#if CFPG_ISA
        return cfpg_vcos8(v);
#else
        __m256d lo, hi;
        vsplit(v, &lo, &hi);
        return vjoin(cfpg_vcos4(lo), cfpg_vcos4(hi));
#endif
    }
    double b[CFPG_LANES];
    vstore(b, v);
    for (int i = 0; i < CFPG_LANES; i++) b[i] = cos(b[i]);
    return vload(b);
}

static inline vpd cfpg_sin8(vpd v)
{
    return cfpg_cos8(vsub(v, vset1(1.5707963267948966)));
}

/*: log of 8 events — the 4-WIDE entry in BOTH builds, because glibc's
 *  8-wide `log` is not bit-equal to its 4-wide one (5040 of 1.6 M samples
 *  differ) and Q must not depend on which one the build could reach. */
static inline vpd cfpg_log8(vpd v)
{
    if (__builtin_expect(cfpg_vmath_ok == 1, 1)) {
        __m256d lo, hi;
        vsplit(v, &lo, &hi);
        return vjoin(cfpg_vlog4(lo), cfpg_vlog4(hi));
    }
    double b[CFPG_LANES];
    vstore(b, v);
    for (int i = 0; i < CFPG_LANES; i++) b[i] = log(b[i]);
    return vload(b);
}

/* ── the Catmull-Rom weights of (pm1, p0, p1, p2) ───────────────────────
 *
 *     p0 + 0.5t(-pm1 + p1 + t(2pm1 - 5p0 + 4p1 - p2
 *                             + t(-pm1 + 3p0 - 3p1 + p2)))
 *   = w0·pm1 + w1·p0 + w2·p1 + w3·p2
 */
static inline void cfpg_cr_weights(vpd t, double *w)
{
    const vpd half = vset1(0.5);
    const vpd t2 = vmul(t, t);
    const vpd h = vmul(half, t);
    const vpd w0 = vmul(h, vfma(t, vsub(vset1(2.0), t), vset1(-1.0)));
    const vpd w1 = vfma(vmul(half, t2), vfmsub(vset1(3.0), t, vset1(5.0)),
                        vset1(1.0));
    const vpd w2 = vmul(h, vfma(t, vfnmadd(vset1(3.0), t, vset1(4.0)),
                                vset1(1.0)));
    const vpd w3 = vmul(vmul(half, t2), vsub(t, vset1(1.0)));
    vstore(w + 0 * CFPG_LANES, w0);
    vstore(w + 1 * CFPG_LANES, w1);
    vstore(w + 2 * CFPG_LANES, w2);
    vstore(w + 3 * CFPG_LANES, w3);
}

/*: one x column's grid position for the tile — the four neighbouring
 *  RELATIVE bin indices (clamped at the row edges exactly like the CUDA
 *  tile's `gps0_interp_*`) and the four Catmull-Rom weights.  Shared by
 *  every table row sampled at that column. */
static inline void cfpg_bin(int *idx, double *w, vpd x, double xmin,
                            double xdelta, int bins)
{
    const vpd diff = vdiv(vsub(x, vset1(xmin)), vset1(xdelta));
    vi32 b = vfloor_i(diff);
    b = vi_max(b, 0);
    b = vi_min(b, bins - 2);
    vpd s = vsub(diff, vi2pd(b));
    s = vmin(vmax(s, vzero()), vset1(1.0));
    vstorei(idx + 0 * CFPG_LANES, vi_max(vi_sub(b, 1), 0));
    vstorei(idx + 1 * CFPG_LANES, b);
    vstorei(idx + 2 * CFPG_LANES, vi_add(b, 1));
    vstorei(idx + 3 * CFPG_LANES, vi_min(vi_add(b, 2), bins - 1));
    cfpg_cr_weights(s, w);
}

/*: Catmull-Rom of one table row from a column's precomputed bin/weights.
 *  `row_base` is `table + row*bins`, the indices are already row-relative. */
static inline vpd cfpg_interp(const double *row_base, const int *idx,
                              const double *w, int nl)
{
    const vpd pm1 = vgather(idx + 0 * CFPG_LANES, row_base, nl);
    const vpd p0 = vgather(idx + 1 * CFPG_LANES, row_base, nl);
    const vpd p1 = vgather(idx + 2 * CFPG_LANES, row_base, nl);
    const vpd p2 = vgather(idx + 3 * CFPG_LANES, row_base, nl);
    vpd v = vfma(vload(w + 1 * CFPG_LANES), p0,
                 vmul(vload(w + 0 * CFPG_LANES), pm1));
    v = vfma(vload(w + 2 * CFPG_LANES), p1, v);
    return vfma(vload(w + 3 * CFPG_LANES), p2, v);
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
    L.mw    = o; o += n_mass * (4 * CFPG_LANES);
    L.qw    = o; o += n_mom * (4 * CFPG_LANES);
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

/*: the ISA this `.so` was built for ("avx512f" = one zmm per vpd,
 *  "avx2" = two ymm halves) — reported, never assumed by the caller. */
const char *cpu_flat_pwa_gen_isa(void) { return CFPG_ISA_NAME; }

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
    const int m = n;   /* live lanes of this tile */
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
        cfpg_bin(midx + col * (4 * CFPG_LANES), mw + col * (4 * CFPG_LANES),
                 vloadn(m, h->mass_s + (size_t)col * ne + e0),
                 c->m_min, c->m_delta, c->m_bins);
    for (int col = 0; col < c->n_momentum; col++)
        cfpg_bin(qidx + col * (4 * CFPG_LANES), qw + col * (4 * CFPG_LANES),
                 vloadn(m, h->q_s + (size_t)col * ne + e0),
                 c->q_min, c->q_delta, c->fl_bins);
    const long long t1 = st ? __rdtsc() : 0;

    /* ── 2. the angular basis ─────────────────────────────────────────
     *
     * general path: one libmvec `cos` per component.
     * integral-k  : cos/sin of each (angle column, component) pair once,
     *               then a Chebyshev pair per component. */
    if (!c->int_k) {
        for (int a = 0; a < A; a++) {
            vpd prod = vset1(c->ach_aconst[a]);
            for (int i = c->ach_off[a]; i < c->ach_off[a + 1]; i++) {
                const double *base =
                    h->ang_s + (size_t)(c->ach_col[i] * c->n_comp) * ne + e0;
                const vpd x =
                    vloadn(m, base + (size_t)c->ach_j[i] * ne);
                prod = vmul(prod, cfpg_cos8(vfma(
                    x, vset1(c->ach_k[i]),
                    vset1(c->ach_b[i]))));
            }
            vstore(ka + a * CFPG_LANES, prod);
        }
    } else {
        double *cs = t + L.cs, *sn = t + L.sn;
        for (int i = 0; i < c->n_aj; i++) {
            const vpd x = vloadn(
                m, h->ang_s + (size_t)i * ne + e0);
            vstore(cs + i * CFPG_LANES, cfpg_cos8(x));
            vstore(sn + i * CFPG_LANES, cfpg_sin8(x));
        }
        for (int a = 0; a < A; a++) {
            vpd prod = vset1(c->ach_aconst[a]);
            for (int i = c->ach_off[a]; i < c->ach_off[a + 1]; i++) {
                const int pair = c->ach_pair[i];
                const vpd cv = vload(cs + pair * CFPG_LANES);
                const vpd sv = vload(sn + pair * CFPG_LANES);
                const int kk = c->ach_kabs[i];
                if (__builtin_expect(kk == 1, 1)) {
                    /* cos(x + b) = cos x·cos b − sin x·sin b  (T_1 = c, U_0 = 1) */
                    prod = vmul(prod, vfmsub(
                        cv, vset1(c->ach_cb[i]),
                        vmul(sv, vset1(
                            (double)c->ach_sgn[i] * c->ach_sb[i]))));
                    continue;
                }
                const vpd two_c = vadd(cv, cv);
                /* T_|k|(c) */
                vpd tm1 = vset1(1.0), tc = cv;
                for (int q = 2; q <= kk; q++) {
                    const vpd tn = vfmsub(two_c, tc, tm1);
                    tm1 = tc;
                    tc = tn;
                }
                /* U_{|k|-1}(c) */
                vpd um1 = vset1(1.0);
                vpd uc = (kk >= 2) ? two_c : vset1(1.0);
                for (int q = 2; q < kk; q++) {
                    const vpd un = vfmsub(two_c, uc, um1);
                    um1 = uc;
                    uc = un;
                }
                const vpd cterm = vmul(
                    tc, vset1(c->ach_cb[i]));
                const vpd sterm = vmul(
                    vmul(uc, sv),
                    vset1((double)c->ach_sgn[i] * c->ach_sb[i]));
                prod = vmul(prod, vsub(cterm, sterm));
            }
            vstore(ka + a * CFPG_LANES, prod);
        }
    }
    const long long t2 = st ? __rdtsc() : 0;

    /* ── 3. fa = ka · matrix_angle, over the matrix's TRUE non-zeros ─── */
    for (int col = 0; col < nfa; col++) {
        vpd sr = vzero(), si = vzero();
        for (int i = c->fa_ptr[col]; i < c->fa_ptr[col + 1]; i++) {
            const vpd kv = vload(ka + c->fa_row[i] * CFPG_LANES);
            sr = vfma(kv, vset1(c->fa_r[i]), sr);
            si = vfma(kv, vset1(c->fa_i[i]), si);
        }
        vstore(fa_r + col * CFPG_LANES, sr);
        vstore(fa_i + col * CFPG_LANES, si);
    }
    const long long t3 = st ? __rdtsc() : 0;

    /* ── 4. barriers: fl[k] = Π_d interp(fl_table, q) ────────────────── */
    for (int u = 0; u < c->n_fl_u; u++)
        vstore(flv + u * CFPG_LANES,
                         cfpg_interp(c->fl_tab
                                     + (size_t)c->fl_u_row[u] * c->fl_bins,
                                     qidx + c->fl_u_col[u] * (4 * CFPG_LANES),
                                     qw + c->fl_u_col[u] * (4 * CFPG_LANES), m));
    for (int k = 0; k < N; k++) {
        vpd p = vload(flv + c->fl_use[k * D] * CFPG_LANES);
        for (int d = 1; d < D; d++)
            p = vmul(p,
                              vload(flv + c->fl_use[k * D + d] * CFPG_LANES));
        vstore(flw + k * CFPG_LANES, p);
    }
    const long long t4 = st ? __rdtsc() : 0;

    /* ── 5. numerator: ONE complex interpolation per deduped use ─────── */
    for (int u = 0; u < c->n_num_u; u++) {
        const int row = c->num_u_row[u];
        const int *idx = midx + c->num_u_col[u] * (4 * CFPG_LANES);
        const double *w = mw + c->num_u_col[u] * (4 * CFPG_LANES);
        const size_t off = (size_t)row * (size_t)c->m_bins;
        vstore(nuv_r + u * CFPG_LANES,
                         cfpg_interp(c->num_r + off, idx, w, m));
        vstore(nuv_i + u * CFPG_LANES,
                         cfpg_interp(c->num_i + off, idx, w, m));
    }
    const long long t5 = st ? __rdtsc() : 0;

    /* ── 6. denominators: F at the unique slots, then D = Π_r Σ w·F ──── */
    for (int u = 0; u < U; u++) {
        if (c->us_const[u]) {
            vstore(Fu_r + u * CFPG_LANES, vset1(c->us_cr[u]));
            vstore(Fu_i + u * CFPG_LANES, vset1(c->us_ci[u]));
            continue;
        }
        const int row = c->us_row[u];
        const int *idx = midx + c->us_col[u] * (4 * CFPG_LANES);
        const double *w = mw + c->us_col[u] * (4 * CFPG_LANES);
        const size_t off = (size_t)row * (size_t)c->m_bins;
        vstore(Fu_r + u * CFPG_LANES,
                         cfpg_interp(c->bas_r + off, idx, w, m));
        vstore(Fu_i + u * CFPG_LANES,
                         cfpg_interp(c->bas_i + off, idx, w, m));
    }
    const double *w = c->w_nz;
    for (int kr = 0; kr < N * R; kr++) {
        vpd sr = vzero(), si = vzero();
        for (int i = c->g_off[kr]; i < c->g_off[kr + 1]; i++) {
            const vpd wv = vset1(w[i]);
            sr = vfma(
                wv, vload(Fu_r + c->g_u[i] * CFPG_LANES), sr);
            si = vfma(
                wv, vload(Fu_i + c->g_u[i] * CFPG_LANES), si);
        }
        vstore(s_r + kr * CFPG_LANES, sr);
        vstore(s_i + kr * CFPG_LANES, si);
    }
    for (int k = 0; k < N; k++) {
        vpd dr = vload(s_r + (k * R) * CFPG_LANES);
        vpd di = vload(s_i + (k * R) * CFPG_LANES);
        for (int r = 1; r < R; r++) {
            const vpd er = vload(s_r + (k * R + r) * CFPG_LANES);
            const vpd ei = vload(s_i + (k * R + r) * CFPG_LANES);
            const vpd tr = vsub(vmul(dr, er),
                                             vmul(di, ei));
            di = vfma(dr, ei, vmul(di, er));
            dr = tr;
        }
        vstore(den_r + k * CFPG_LANES, dr);
        vstore(den_i + k * CFPG_LANES, di);
        vstore(rden + k * CFPG_LANES,
                         vmaskz(
                             m, vdiv(
                                    vset1(1.0),
                                    vfma(dr, dr,
                                                    vmul(di, di)))));
        /* others_{k,r} = Π_{l≠r} s_l  (1 at n_res == 1) */
        for (int r = 0; r < R; r++) {
            vpd pr = vset1(1.0), pi = vzero();
            for (int l = 0; l < R; l++) {
                if (l == r) continue;
                const vpd er = vload(s_r + (k * R + l) * CFPG_LANES);
                const vpd ei = vload(s_i + (k * R + l) * CFPG_LANES);
                const vpd tr = vsub(vmul(pr, er),
                                                 vmul(pi, ei));
                pi = vfma(pr, ei, vmul(pi, er));
                pr = tr;
            }
            vstore(oth_r + (k * R + r) * CFPG_LANES, pr);
            vstore(oth_i + (k * R + r) * CFPG_LANES, pi);
        }
    }
    const long long t6 = st ? __rdtsc() : 0;

    /* ── 7. common, A_p, P_e ─────────────────────────────────────────── */
    for (int k = 0; k < N; k++) {
        vpd nr = vload(nuv_r + c->num_use[k * R] * CFPG_LANES);
        vpd ni = vload(nuv_i + c->num_use[k * R] * CFPG_LANES);
        for (int r = 1; r < R; r++) {
            const vpd er = vload(
                nuv_r + c->num_use[k * R + r] * CFPG_LANES);
            const vpd ei = vload(
                nuv_i + c->num_use[k * R + r] * CFPG_LANES);
            const vpd tr = vsub(vmul(nr, er),
                                             vmul(ni, ei));
            ni = vfma(nr, ei, vmul(ni, er));
            nr = tr;
        }
        const vpd fl = vload(flw + k * CFPG_LANES);
        nr = vmul(nr, fl);
        ni = vmul(ni, fl);
        /* common = (num·fl)·conj(D)/|D|² */
        const vpd dr = vload(den_r + k * CFPG_LANES);
        const vpd di = vload(den_i + k * CFPG_LANES);
        const vpd rd = vload(rden + k * CFPG_LANES);
        vstore(cm_r + k * CFPG_LANES,
                         vmul(vfma(
                                           ni, di, vmul(nr, dr)),
                                       rd));
        vstore(cm_i + k * CFPG_LANES,
                         vmul(vfmsub(
                                           ni, dr, vmul(nr, di)),
                                       rd));
    }
    for (int p = 0; p < P; p++) {
        vpd ar = vzero(), ai = vzero();
        for (int k = 0; k < N; k++) {
            const int col = c->fa_slot[p * N + k];
            const vpd cr = vload(cm_r + k * CFPG_LANES);
            const vpd ci = vload(cm_i + k * CFPG_LANES);
            const vpd fr = vload(fa_r + col * CFPG_LANES);
            const vpd fi = vload(fa_i + col * CFPG_LANES);
            const vpd xr = vfmsub(vset1(ck_r[k]), cr,
                                               vmul(
                                                   vset1(ck_i[k]),
                                                   ci));
            const vpd xi = vfma(vset1(ck_r[k]), ci,
                                               vmul(
                                                   vset1(ck_i[k]),
                                                   cr));
            ar = vfma(xr, fr, ar);
            ar = vfnmadd(xi, fi, ar);
            ai = vfma(xr, fi, ai);
            ai = vfma(xi, fr, ai);
        }
        vstore(Ar + p * CFPG_LANES, ar);
        vstore(Ai + p * CFPG_LANES, ai);
    }
    {
        vpd pe = vzero();
        for (int p = 0; p < P; p++) {
            const vpd ar = vload(Ar + p * CFPG_LANES);
            const vpd ai = vload(Ai + p * CFPG_LANES);
            pe = vfma(ar, ar, pe);
            pe = vfma(ai, ai, pe);
        }
        vstoren(h->P + e0, m, pe);
    }

    /* ── 8. v_k, pv_k, a2 and the unscaled dQ/dcoeff of the tile ─────── */
    for (int k = 0; k < N; k++) {
        vpd ur = vzero(), ui = vzero();
        for (int p = 0; p < P; p++) {
            const int col = c->fa_slot[p * N + k];
            const vpd ar = vload(Ar + p * CFPG_LANES);
            const vpd ai = vload(Ai + p * CFPG_LANES);
            const vpd fr = vload(fa_r + col * CFPG_LANES);
            const vpd fi = vload(fa_i + col * CFPG_LANES);
            /* conj(A_p)·fa */
            ur = vfma(ar, fr, ur);
            ur = vfma(ai, fi, ur);
            ui = vfma(ar, fi, ui);
            ui = vfnmadd(ai, fr, ui);
        }
        const vpd cr = vload(cm_r + k * CFPG_LANES);
        const vpd ci = vload(cm_i + k * CFPG_LANES);
        /* v = u·common, deliberately WITHOUT ck: grad_ck must not carry it */
        const vpd vr = vsub(vmul(ur, cr),
                                         vmul(ui, ci));
        const vpd vi = vfma(ur, ci, vmul(ui, cr));
        /* pv = -v/D = -v·conj(D)/|D|² */
        const vpd dr = vload(den_r + k * CFPG_LANES);
        const vpd di = vload(den_i + k * CFPG_LANES);
        const vpd rd = vload(rden + k * CFPG_LANES);
        const vpd pvr = vmul(
            vfnmadd(vset1(1.0),
                             vfma(vi, di, vmul(vr, dr)),
                             vzero()), rd);
        const vpd pvi = vmul(
            vfmsub(vr, di, vmul(vi, dr)), rd);
        for (int r = 0; r < R; r++) {
            const vpd orr = vload(oth_r + (k * R + r) * CFPG_LANES);
            const vpd oii = vload(oth_i + (k * R + r) * CFPG_LANES);
            /* ck·pv, then × others, then × 2 (the 2Re of the chain) */
            const vpd xr = vfmsub(vset1(ck_r[k]), pvr,
                                               vmul(
                                                   vset1(ck_i[k]),
                                                   pvi));
            const vpd xi = vfma(vset1(ck_r[k]), pvi,
                                               vmul(
                                                   vset1(ck_i[k]),
                                                   pvr));
            const vpd yr = vsub(vmul(xr, orr),
                                             vmul(xi, oii));
            const vpd yi = vfma(xr, oii,
                                               vmul(xi, orr));
            vstore(a2r + (k * R + r) * CFPG_LANES, vadd(yr, yr));
            vstore(a2i + (k * R + r) * CFPG_LANES, vadd(yi, yi));
        }
        if (want_vg) {
            vstoren(seg_vr + (size_t)k * cap + so, m, vr);
            vstoren(seg_vi + (size_t)k * cap + so, m, vi);
        }
    }
    if (want_vg) {
        /* g_m = Σ_{kr} 2Re(ck·pv·others·gamma·F_m): one FMA pair per nz.
           It accumulates over (k, r), so the lanes of this tile are cleared
           first — a tile offset reused by a later tile must not inherit the
           previous tile's g. */
        for (int mm = 0; mm < NB; mm++)
            vstoren(seg_g + (size_t)mm * cap + so, m,
                                  vzero());
        for (int kr = 0; kr < N * R; kr++) {
            const vpd xr = vload(a2r + kr * CFPG_LANES);
            const vpd xi = vload(a2i + kr * CFPG_LANES);
            for (int i = c->g_off[kr]; i < c->g_off[kr + 1]; i++) {
                const int mu = c->g_u[i], mm = c->g_m[i];
                const vpd wv = vset1(c->g_w[i]);
                double *gp = seg_g + (size_t)mm * cap + so;
                vpd acc = vload(gp);
                acc = vfma(
                    xr, vmul(wv, vload(Fu_r + mu * CFPG_LANES)),
                    acc);
                acc = vfnmadd(
                    xi, vmul(wv, vload(Fu_i + mu * CFPG_LANES)),
                    acc);
                vstore(gp, acc);
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
static void cfpg_seg_acc(CFPGCtx *c, double *t, vpd dqp, int so, int n,
                         long long *st)
{
    const long long t0 = st ? __rdtsc() : 0;
    const int N = c->N, NB = c->NB, cap = c->cap;
    const CFPGLayout L = c->L;
    const double *seg_vr = t + L.seg_vr, *seg_vi = t + L.seg_vi;
    const double *seg_g = t + L.seg_g;
    double *acr = t + L.acc_ckr, *aci = t + L.acc_cki, *acg = t + L.acc_g;
    dqp = vmaskz(n, dqp);
    for (int k = 0; k < N; k++) {
        double *pr = acr + k * CFPG_LANES, *pi = aci + k * CFPG_LANES;
        vstore(pr, vfma(
            dqp, vload(seg_vr + (size_t)k * cap + so),
            vload(pr)));
        vstore(pi, vfma(
            dqp, vload(seg_vi + (size_t)k * cap + so),
            vload(pi)));
    }
    for (int mm = 0; mm < NB; mm++) {
        double *pg = acg + mm * CFPG_LANES;
        vstore(pg, vfma(
            dqp, vload(seg_g + (size_t)mm * cap + so),
            vload(pg)));
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
    c->I.qidx = n_mass * (4 * CFPG_LANES);
    c->I.total = (n_mass + n_momentum) * (4 * CFPG_LANES);
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
        memset(t + L.acc_ckr, 0, sizeof(double) * (size_t)(N * CFPG_LANES));
        memset(t + L.acc_cki, 0, sizeof(double) * (size_t)(N * CFPG_LANES));
        memset(t + L.acc_g, 0, sizeof(double) * (size_t)(NB * CFPG_LANES));

        if (pathA) {
            /* one event at a time: dQ/dP closes inside the tile */
            #pragma omp for schedule(static)
            for (int tile = 0; tile < ntiles; tile++) {
                const int e0 = tile * CFPG_LANES;
                const int n = (ne - e0 < CFPG_LANES) ? ne - e0 : CFPG_LANES;
                const int m = n;   /* live lanes of this tile */
                cfpg_tile(c, h, t, it, e0, n, ck_r, ck_i, 0, 1, st);
                const vpd wv = vloadn(m, h->weight + e0);
                const vpd bv = vloadn(m, h->bkg + e0);
                const vpd P = vloadn(m, h->P + e0);
                vpd dq;
                if (!use_norm) {
                    Q += vhsum_n(m, vmul(wv, P));
                    dq = wv;
                } else {
                    const vpd nv = vset1(norm);
                    /* pdf = P/norm + b = (P + b·norm)/norm */
                    const vpd den = vfma(bv, nv, P);
                    Q -= vhsum_n(
                        m, vmul(wv, cfpg_log8(
                               vdiv(den, nv))));
                    dq = vmaskz(
                        m, vdiv(vset1(-1.0), den));
                    dq = vmul(dq, wv);
                    Dn += vhsum_n(
                        m, vdiv(vmul(wv, P),
                                         vmul(nv, den)));
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
                        const int m = nn;
                        for (int e = 0; e < nn; e++)
                            dqp[e] = scale * h->weight[e0 + so + sl + e] / norm;
                        cfpg_seg_acc(c, t, vloadn(m, dqp),
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
            g_ck_r[k] += vhsum(
                             vload(t + c->L.acc_ckr + k * CFPG_LANES));
            g_ck_i[k] += vhsum(
                             vload(t + c->L.acc_cki + k * CFPG_LANES));
        }
        for (int mm = 0; mm < NB; mm++)
            g_coeff[mm] += vhsum(
                               vload(t + c->L.acc_g + mm * CFPG_LANES));
    }
    *dnorm_out = Dn;
    if (return_p && P_out) memcpy(P_out, h->P, sizeof(double) * (size_t)ne);
    return Q;
}

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
    const int cnt[17] = {
        c->n_ab * CFPG_LANES, c->n_fa * CFPG_LANES, c->n_fa * CFPG_LANES,
        c->n_fl_u * CFPG_LANES, c->N * CFPG_LANES,
        c->n_num_u * CFPG_LANES, c->n_num_u * CFPG_LANES,
        c->n_uslot * CFPG_LANES, c->n_uslot * CFPG_LANES,
        c->N * CFPG_LANES, c->N * CFPG_LANES, c->N * CFPG_LANES,
        c->N * CFPG_LANES, c->P * CFPG_LANES, c->P * CFPG_LANES,
        c->N * c->cap, c->N * c->cap};
    const double *src[17] = {
        t + L.ka, t + L.fa_r, t + L.fa_i, t + L.flv, t + L.flw,
        t + L.nuv_r, t + L.nuv_i, t + L.Fu_r, t + L.Fu_i,
        t + L.den_r, t + L.den_i, t + L.cm_r, t + L.cm_i, t + L.Ar, t + L.Ai,
        t + L.seg_vr, t + L.seg_vi};
    for (int k = 0; k < 17; k++) {
        memcpy(out + o, src[k], sizeof(double) * (size_t)cnt[k]);
        o += cnt[k];
    }
    memcpy(out + o, t + L.seg_g, sizeof(double) * (size_t)(c->NB * c->cap));
    o += c->NB * c->cap;
    memcpy(out + o, h->P, sizeof(double) * (size_t)n);
    o += n;
    return o;
}

