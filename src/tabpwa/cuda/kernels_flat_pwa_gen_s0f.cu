/**
 * cuda_flat_pwa_gen_s0f — the s0e2 fused tile with a FIXED-WAVE amplitude
 * cache, on the flat `pwa_gen` config.
 *
 * Case-1 copy (kernel policy §1): own `.cu`, own `.so`, own wrapper, own
 * symbols.  `kernels_flat_pwa_gen_s0e2.cu` and every other kernel/wrapper in
 * the tree stay byte-identical.
 *
 * ── what s0f adds: waves whose denominator cannot move ───────────────────
 *
 * A wave's denominator is
 *
 *     D_{e,k} = Σ_{m ∈ nz(k)} coeff[m] · Z_m(e)          (Z = the `us` cache)
 *
 * If every column `m` in `nz(k)` has a FIXED `coeff` value, `D` is
 * iteration-invariant, and so is the whole amplitude contribution
 *
 *     Afix[e,k,p] = Amp_{e,slot(k),p} · conj(D)/|D|²
 *
 * which is exactly phase A's `common` for that wave.  The HOST declares the
 * fixed COLUMNS and derives, at build time (see the wrapper):
 *
 *   * `n_fix` and a PERMUTATION of the wave axis that puts the fixed waves
 *     first (each group in its original order; identity when nothing is
 *     fixed);
 *   * the CSR / `gamma` of the FREE waves only, renumbered `0 .. n_free-1`;
 *   * the fixed waves' own (small) CSR, kept only so `gps0f_fill_afix` can
 *     evaluate their `D` once.
 *
 * Per loaded handle the first `compute` fills
 *
 *     Afix[e, k, p]        (float2, index ((e·n_fix) + k)·n_proj + p)
 *
 * with `gps0f_fill_afix`: phase A's sparse dot verbatim (same packed 64-bit
 * `nz_pair` step, same two fp64 accumulation chains, same `1/|D|²`), then the
 * same `(amp.x·D_r + amp.y·D_i)·rden` / `(amp.y·D_r − amp.x·D_i)·rden`
 * expressions phase A uses for the free waves.  The cache therefore stores
 * `common`, not `1/D`: a fixed wave needs no `pv` at all.
 *
 * Per iteration the fixed block LIVES OUTSIDE THE TILE, because it is linear
 * in `ck` and keeping it inside made every 8-event chunk pay the tile's six
 * barriers:
 *
 *   F:  A1[e,p] = Σ_{k<n_fix} ck_k·Afix[e,k,p]           (no smem, no barriers)
 *   T:  phase A runs over the FREE waves only (`N = n_free`); phase B starts
 *       its warp reduction from `A1` (read once, on lane 0) and adds the free
 *       sum; phases G/B2 are the parent's, and B2 also writes the finished
 *       `dA` to `dA_g`; phase C is the parent's free pass.  No `pv` is
 *       written for a fixed wave;
 *   B:  dck_k = Σ_{e,p} dA[e,p]·Afix[e,k,p]               (a GEMV into `g_ck`)
 *   * phase D walks the FREE CSR only.  Consequence: `dQ/dcoeff[m]` is
 *     EXACTLY 0 for every column of a fixed wave (a fixed column has no `x`
 *     direction; the constraint layer drops those entries).
 *
 * `n_fix == 0` launches neither F nor B and allocates neither buffer, so the
 * mask-off path is the parent's.  `split_fixed=0` keeps the old in-tile fixed
 * loops (a benchmark lever, not the production path).
 *
 * `ck` arrives in PERMUTED wave order and `dck` is returned in that order;
 * the wrapper gathers `ck[perm]` on the way in and un-permutes on the way out,
 * so the caller always sees model wave order.
 *
 * `n_fix == 0` is the identity permutation and no cache at all, so the kernel
 * executes the parent's code path unchanged — pinned by `np.array_equal`
 * against `cuda_flat_pwa_gen_s0e2` in `tests/test_cuda_flat_pwa_gen_s0f.py`
 * (Q, P, dck, dcoeff, dnorm).
 *
 * ── memory ──────────────────────────────────────────────────────────────
 *
 * Per handle: `8 · n_proj · n_fix · ne` bytes.  It is a real per-sample cost,
 * and the handle that carries it is the DATA one; the phase-space handle
 * (`norm` pass) fills its own copy, because the same `compute` runs on it.
 * No allocation ever scales with `ne` for the working scratch (the tile is
 * shared memory only); `batch_size` bounds the chunk, not this cache.
 *
 * ── everything else is the parent (s0e2) ────────────────────────────────
 *
 * The basis rotation (`Z = i·basis_table`, the kernel's `-i·Z` recovers the
 * row uniformly, no `basis_kind` anywhere), the two byte-exact dedups (`us` on
 * `(row bytes, mass column)`, `amp` on the whole numerator of every
 * projection), the one-pass algebra (u-merge: `dQ/dck` and `dQ/dD` come from
 * the same `Pk`), the grouped log-sum closed inside the block (path A for
 * `rsize <= te`, path B for `rsize > te`), the batch_size chunk rule (`cap`,
 * published through `cuda_cap_*`; `ne` never enters an allocation) and the
 * packed-CSR/two-chain sparse dot are the parent's, unchanged.
 *
 * Precision policy is unchanged: the stored caches (`us`, `amp`, `Afix`) are
 * float2, every real arithmetic step stays fp64 (gradients included).
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstring>
#include <cmath>

#define GPS0F_BLOCK       256      /* threads per block (multiple of 32) */
/* Launch bounds: the SECOND parameter is a REGISTER BUDGET choice, not a
 * correctness one.  Measured on sm_120: 5 blocks/SM => <=51 registers, which
 * is what the 2-chain dot needs (without it the compiler takes 56 and loses
 * 17 occupancy points).  The right budget is per ARCH, so it is a macro:
 *
 *   per build   TABPWA_NVCC_FLAGS_SM70="-DGPS0F_MIN_BLOCKS=6"
 *               (the arch-keyed extra-flags hook; -D applies to every pass of
 *               that one nvcc invocation)
 *   per pass    an `#if __CUDA_ARCH__` block here, if ONE fat binary must
 *               differ between its cubins (__CUDA_ARCH__ is set per device
 *               pass; the default below leaves the bound arch-uniform)
 *
 * Volta has 64 warps/SM (8 x 256-thread blocks) but only 64K registers, so 6
 * blocks there wants <=42 registers.
 *
 * MEASURED on sm_120 (0 spills at every N): N=4 -> 64 registers -> 32 warps;
 * N=5 -> 48 -> 40 warps (the default, matching s0e's occupancy); N=6 -> 40 ->
 * 48 warps (100 %).  MEASURED on sm_70: N=4 is clean, N=5 spills 56 B/192 B,
 * N=6 spills 132 B/444 B — hence the arch-aware default above and the
 * per-arch override, e.g. `TABPWA_NVCC_FLAGS_SM70=-DGPS0F_MIN_BLOCKS=3`.
 * Pick from a measurement, never from taste. */
#ifndef GPS0F_MIN_BLOCKS
#  if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
/* MEASURED with -Xptxas -v, tile kernel, N = min blocks per SM:
 *   arch   N=4           N=5
 *   70     clean         56 B stores / 192 B loads
 *   75     clean         clean
 *   80     clean         28 / 36
 *   86     clean         28 / 36
 *   89     clean         28 / 36
 *   90     clean         8 / 8
 *   120    clean         clean
 * N=4 is clean EVERYWHERE, so it is the default; N=5 (48 registers, 40 warps
 * instead of 64 registers, 32 warps) is the better point where it does not
 * spill, and 6 (40 registers, 48 warps = 100 %) is the A/B candidate here.
 * __CUDA_ARCH__ is defined per DEVICE pass, which is the granularity a fat
 * binary needs; the env hook overrides the default per build. */
#    define GPS0F_MIN_BLOCKS 5      /* Blackwell consumer: clean, matches s0e */
#  else
#    define GPS0F_MIN_BLOCKS 4      /* every other arch measured: 5 spills */
#  endif
#endif
#define GPS0F_TILE_BOUNDS __launch_bounds__(GPS0F_BLOCK, GPS0F_MIN_BLOCKS)
#define GPS0F_DEF_TE      8        /* default events per tile */
#define GPS0F_MAX_CAP     (1 << 20)  /* ceiling on the chunk length  */
/*: soft shared-memory budget for choosing `te`.  Below this the launch needs
 *  no opt-in (so no `cudaFuncSetAttribute` and the best occupancy); above it
 *  `create` shrinks `te` first and only then crosses the opt-in. */
#define GPS0F_SMEM_BUDGET (48 * 1024)

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = call;                                             \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                               \
        }                                                                    \
    } while (0)

/* ── Catmull-Rom (the repo-wide idiom, identical to gen2/1bw) ─────────── */
__device__ double gps0f_cr1d(double pm1, double p0, double p1, double p2,
                            double t) {
    return p0 + 0.5 * t * (-pm1 + p1 + t * (2.0 * pm1 - 5.0 * p0 + 4.0 * p1
                                            - p2 + t * (-pm1 + 3.0 * p0
                                                        - 3.0 * p1 + p2)));
}

__device__ double gps0f_interp_real(const double* __restrict__ t, int row,
                                   double x, double xmin, double xdelta,
                                   int bins) {
    double diff = (x - xmin) / xdelta;
    int b = max(0, min((int)floor(diff), bins - 2));
    double s = max(0.0, min(diff - b, 1.0));
    int base = row * bins + b;
    int lo = row * bins, hi = (row + 1) * bins - 1;
    int im1 = base > lo ? base - 1 : base;
    int i2 = base + 2 <= hi ? base + 2 : base + 1;
    return gps0f_cr1d(t[im1], t[base], t[base + 1], t[i2], s);
}

__device__ void gps0f_interp_c(const double* __restrict__ tr,
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
    *orr = gps0f_cr1d(tr[im1], tr[base], tr[base + 1], tr[i2], s);
    *oii = gps0f_cr1d(ti[im1], ti[base], ti[base + 1], ti[i2], s);
}

/*=============================================================================
 * LOAD-TIME FILL 1 — the us-cache: Z = i·basis_table[m] at the event mass.
 *
 * One thread per (event, uslot).  `uslot_row[u]` / `uslot_col[u]` are the
 * deduplicated `(table row, mass column)` pair; nothing here depends on a
 * fitted parameter, so this is computed ONCE per `load_data` and reused by
 * every iteration.  Stored as float2 (`Z` is complex) — the same fp32 cache
 * floor the sibling flat kernels are gated at.
 *===========================================================================*/
__global__ void gps0f_fill_us(
    const double* __restrict__ bas_r, const double* __restrict__ bas_i,
    const int* __restrict__ uslot_row, const int* __restrict__ uslot_col,
    double m_min, double m_delta, int m_bins,
    const double* __restrict__ mass, int n_mass,
    int ne, int n_uslot, float2* __restrict__ us)
{
    const long long total = (long long)ne * (long long)n_uslot;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         g < total; g += stride) {
        const int e = (int)(g / n_uslot);
        const int u = (int)(g - (long long)e * n_uslot);
        const double mv = mass[(size_t)e * n_mass + uslot_col[u]];
        double vr, vi;
        gps0f_interp_c(bas_r, bas_i, uslot_row[u], mv,
                      m_min, m_delta, m_bins, &vr, &vi);
        /* Z = i·basis = (-basis_i, basis_r);  the kernel's `-i·Z` is `basis`. */
        us[g] = make_float2((float)(-vi), (float)vr);
    }
}

/*=============================================================================
 * LOAD-TIME FILL 2 — the amp-cache: Amp = fa·fl·num per (event, wave, p).
 *
 * The angular basis `ka_a = Π_j cos(angle·angle_k + angle_b)` is evaluated
 * once per (event, basis row) in shared memory and reused by every
 * projection and every deduplicated wave slot — the flat family's fill
 * kernels recompute it per (wave, projection), which is 26·2 = 52x the work
 * for the same numbers.
 *
 * `n_num == 0` (a config with no numerator rows) is not a special case here:
 * `num_table` always has at least the constant row, so `n_res >= 1`.
 *===========================================================================*/
__global__ void gps0f_fill_amp(
    const double* __restrict__ fl_table, const int* __restrict__ fl_q_index,
    const double* __restrict__ num_r, const double* __restrict__ num_i,
    const int* __restrict__ num_m_index,
    double m_min, double m_delta, int m_bins,
    double q_min, double q_delta, int q_bins,
    const int* __restrict__ angle_index, const double* __restrict__ angle_k,
    const double* __restrict__ angle_b,
    const double* __restrict__ am_r, const double* __restrict__ am_i,
    const double* __restrict__ mass, const double* __restrict__ mom,
    const double* __restrict__ ang,
    const int* __restrict__ rep_of_slot,
    int ne, int n_uniq, int n_wave, int n_proj, int n_res, int n_decay,
    int n_ab, int n_comp, int n_angle_total, int n_mass, int n_momentum,
    float2* __restrict__ amp)
{
    const int e = blockIdx.x;
    if (e >= ne) return;
    const int tid = threadIdx.x, nt = blockDim.x;

    extern __shared__ double ska[];
    for (int a = tid; a < n_ab; a += nt) {
        const int col = angle_index[a];
        double prod = 1.0;
        for (int j = 0; j < n_comp; j++) {
            prod *= cos(ang[((size_t)e * n_angle_total + col) * n_comp + j]
                        * angle_k[a * n_comp + j]
                        + angle_b[a * n_comp + j]);
        }
        ska[a] = prod;
    }
    __syncthreads();

    const int PW = n_proj * n_wave;
    for (int s = tid; s < n_uniq; s += nt) {
        const int k = rep_of_slot[s];
        /* numerator: Π_r num_table[k, r](m_{e, num_m_index[k, r]}) */
        double nr = 1.0, ni = 0.0;
        for (int r = 0; r < n_res; r++) {
            const double mv =
                mass[(size_t)e * n_mass + num_m_index[k * n_res + r]];
            double vr, vi;
            gps0f_interp_c(num_r, num_i, k * n_res + r, mv,
                          m_min, m_delta, m_bins, &vr, &vi);
            const double tr = nr * vr - ni * vi;
            ni = nr * vi + ni * vr;
            nr = tr;
        }
        /* barrier: Π_d fl_table[k, d](q_{e, fl_q_index[k, d]}) */
        double fl = 1.0;
        for (int d = 0; d < n_decay; d++) {
            const double qv =
                mom[(size_t)e * n_momentum + fl_q_index[k * n_decay + d]];
            fl *= gps0f_interp_real(fl_table, k * n_decay + d, qv,
                                   q_min, q_delta, q_bins);
        }
        const double Fr = fl * nr, Fi = fl * ni;
        for (int p = 0; p < n_proj; p++) {
            double far = 0.0, fai = 0.0;
            for (int a = 0; a < n_ab; a++) {
                const double ka = ska[a];
                /* matrix_angle is p-major: column p·n_wave + k, row a */
                const size_t o = (size_t)a * PW + (size_t)(p * n_wave + k);
                far += ka * am_r[o];
                fai += ka * am_i[o];
            }
            amp[((size_t)e * n_uniq + s) * n_proj + p] =
                make_float2((float)(Fr * far - Fi * fai),
                            (float)(Fr * fai + Fi * far));
        }
    }
}

/*=============================================================================
 * LOAD-TIME FILL 3 — the fixed-wave amplitude cache.
 *
 * A fixed wave's denominator cannot move, so its whole amplitude contribution
 * `Afix[e,k,p] = Amp·conj(D)/|D|²` is iteration-invariant: it is EXACTLY
 * phase A's `common` for that wave, evaluated once per loaded handle.
 *
 * The sparse dot below is phase A's body verbatim — the same packed 64-bit
 * `nz_pair` step, the same two independent fp64 accumulation chains, the same
 * `1/|D|²` — so the cached value is the value the free path would produce for
 * the same `D` (it is cast to float2 only on the way out, like every other
 * cache here).  `fix_slot_of_wave[k]` is the ORIGINAL wave's amplitude slot
 * (the wave axis is permuted for the tile, but the cache is indexed by the
 * fixed block's own `k`).
 *
 * One thread per `(event, fixed wave)`; chunking is the caller's (the fill
 * runs ONCE for the whole handle, so a `ne` here is the sample length).
 *===========================================================================*/
__global__ void gps0f_fill_afix(
    const float2* __restrict__ us,             /* [ne, n_uslot] Z = i·basis */
    const float2* __restrict__ amp,            /* [ne, n_uniq, P]          */
    const int* __restrict__ fix_slot_of_wave,  /* [n_fix] -> amp slot      */
    const long long* __restrict__ fix_pair,    /* [nnz_fix] m | (u << 32)  */
    const int* __restrict__ fix_lo,            /* [n_fix] CSR start        */
    const int* __restrict__ fix_hi,            /* [n_fix] CSR end          */
    const double* __restrict__ coeff,          /* [n_basis]                */
    int n_fix, int ne, int n_uslot, int n_uniq, int n_proj,
    float2* __restrict__ afix)                 /* [ne, n_fix, P]           */
{
    const long long total = (long long)ne * (long long)n_fix;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         g < total; g += stride) {
        const int e = (int)(g / n_fix);
        const int k = (int)(g - (long long)e * n_fix);
        const float2* uu = us + (size_t)e * n_uslot;
        const int lo = fix_lo[k], hi = fix_hi[k];
        double sr0 = 0.0, si0 = 0.0, sr1 = 0.0, si1 = 0.0;
        int i = lo;
        const int two = lo + ((hi - lo) & ~1);
        for (; i < two; i += 2) {
            const long long p0 = fix_pair[i], p1 = fix_pair[i + 1];
            const double c0 = coeff[(int)(p0 & 0xFFFFFFFFLL)];
            const double c1 = coeff[(int)(p1 & 0xFFFFFFFFLL)];
            const float2 z0 = uu[(int)(p0 >> 32)];
            const float2 z1 = uu[(int)(p1 >> 32)];
            sr0 += c0 * (double)z0.y;
            si0 -= c0 * (double)z0.x;
            sr1 += c1 * (double)z1.y;
            si1 -= c1 * (double)z1.x;
        }
        if (i < hi) {
            const long long p = fix_pair[i];
            const double c = coeff[(int)(p & 0xFFFFFFFFLL)];
            const float2 z = uu[(int)(p >> 32)];
            sr0 += c * (double)z.y;
            si0 -= c * (double)z.x;
        }
        const double sr = sr0 + sr1, si = si0 + si1;
        const double rden = 1.0 / (sr * sr + si * si);
        const size_t abase = (size_t)e * n_uniq * n_proj
                             + (size_t)fix_slot_of_wave[k] * n_proj;
        for (int p = 0; p < n_proj; p++) {
            const float2 a = amp[abase + p];
            afix[((size_t)e * n_fix + k) * n_proj + p] =
                make_float2(
                    (float)(((double)a.x * sr + (double)a.y * si) * rden),
                    (float)(((double)a.y * sr - (double)a.x * si) * rden));
        }
    }
}

/*=============================================================================
 * THE FIXED BLOCK, OUTSIDE THE TILE — kernels F and B.
 *
 * Keeping the fixed block inside the tile means every 8-event chunk pays the
 * tile's six barriers for it.  The cache makes the block LINEAR in `ck`, so
 * its whole per-iteration contribution is one gather-and-sum:
 *
 *   F:  A1[e,p] = Σ_{k<n_fix} ck_k·Afix[e,k,p]
 *   B:  dck_k   = Σ_{e,p} dA[e,p]·Afix[e,k,p]        (k < n_fix)
 *
 * Both are barrier-free global passes over buffers that already exist.  The
 * tile then starts phase B from `A1` (one float2 read per (e,p), added once,
 * on lane 0) and writes its finished `dA` to a global buffer, so B is an
 * ordinary GEMV.  With `n_fix == 0` neither kernel is launched and neither
 * buffer exists: the mask-off path is the parent's, unchanged.
 *===========================================================================*/

/* F: one thread per (e,p).  For a fixed `e` the `(k,p)` pairs are contiguous
 * (index `((e·n_fix)+k)·P + p`), so a warp covering consecutive `p` of one
 * event reads a contiguous run and then strides by `P` per k. */
__global__ void gps0f_fwd_fixed(
    const float2* __restrict__ afix,           /* [ne, n_fix, P]          */
    const double* __restrict__ ck_r, const double* __restrict__ ck_i,
    int n_fix, int ne, int P,
    double2* __restrict__ a1)                  /* [ne, P]  fp64 on purpose */
{
    const long long total = (long long)ne * (long long)P;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         g < total; g += stride) {
        const int e = (int)(g / P);
        const int p = (int)(g - (long long)e * P);
        const float2* base = afix + ((size_t)e * n_fix) * P + p;
        double ar = 0.0, ai = 0.0;
        for (int k = 0; k < n_fix; k++) {
            const float2 a = base[(size_t)k * P];
            const double ckr = ck_r[k], cki = ck_i[k];
            ar += ckr * (double)a.x - cki * (double)a.y;
            ai += ckr * (double)a.y + cki * (double)a.x;
        }
        /* fp64: A1 is multiplied by `ck`, so a float2 store would make Q a
           step function of `ck` and break the finite-difference gate at
           h = 1e-6 (measured: a 1.4 % FD error on the fixed `dck`). */
        a1[g] = make_double2(ar, ai);
    }
}

/* B: one warp per fixed wave, a chunk of `(e,p)` elements staged in shared.
 *
 * The earlier shape (one warp per (e,p), lane = wave) needed a shared
 * `atomicAdd` per (element, wave); measured 8.4 ns/event, about a third of
 * the whole masked call.  Here the `dA` chunk is staged once (with the path-B
 * group scale already folded in) together with each element's `Afix` offset,
 * every warp owns a wave, the lane partials reduce by warp shuffle, and ONE
 * global `atomicAdd` per (block, wave) lands in the parent's `g_ck`.  No
 * shared atomics at all, and `dA`/`dscale` are read once per element. */
__global__ void gps0f_bwd_fixed_wave(
    const float2* __restrict__ afix,           /* [ne, n_fix, P]          */
    const float2* __restrict__ da,             /* [ne, P]  the tile's dA  */
    const double* __restrict__ dscale,         /* [ne] path B's group cs  */
    int n_fix, int ne, int P,
    double* __restrict__ g_ck_r, double* __restrict__ g_ck_i)
{
    extern __shared__ float2 sda[];            /* [blockDim]              */
    long long* soff = (long long*)(sda + blockDim.x);  /* [blockDim]       */
    const int tid = threadIdx.x, nt = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, nwarps = nt >> 5;
    const long long total = (long long)ne * (long long)P;
    const long long stride = (long long)gridDim.x * nt;
    for (long long base = (long long)blockIdx.x * nt; base < total;
         base += stride) {
        const long long g = base + tid;
        const int nv = (int)((total - base < nt) ? (total - base) : nt);
        if (tid < nv) {
            const long long gg = base + tid;
            const int e = (int)(gg / P);
            const int p = (int)(gg - (long long)e * P);
            const float2 d = da[gg];
            /* path B's dA_g is UNSCALED (its group constant closes only at
               the end of the tile block), so fold the per-event factor here;
               path A already baked it in and passes dscale == NULL. */
            const double sc = (dscale != NULL) ? dscale[e] : 1.0;
            sda[tid] = make_float2((float)((double)d.x * sc),
                                   (float)((double)d.y * sc));
            soff[tid] = (long long)e * n_fix * P + p;
        }
        __syncthreads();
        for (int k = warp; k < n_fix; k += nwarps) {
            double ur = 0.0, ui = 0.0;
            for (int i = lane; i < nv; i += 32) {
                const float2 d = sda[i];
                const float2 a = afix[soff[i] + (long long)k * P];
                ur += (double)d.x * (double)a.x - (double)d.y * (double)a.y;
                ui += (double)d.x * (double)a.y + (double)d.y * (double)a.x;
            }
            for (int off = 16; off > 0; off >>= 1) {
                ur += __shfl_down_sync(0xffffffffu, ur, off);
                ui += __shfl_down_sync(0xffffffffu, ui, off);
            }
            if (lane == 0) {
                if (ur != 0.0) atomicAdd(g_ck_r + k, ur);
                if (ui != 0.0) atomicAdd(g_ck_i + k, ui);
            }
        }
        __syncthreads();   /* the shared chunk is reused */
    }
}

/* B (EVENT-MAJOR): one thread per `(e,p)`, the k loop contiguous in
 * `Afix[e][k][p]` — the same cache lines the forward pass just read — and the
 * per-wave partials in shared, so the only global atomics are ONE per
 * (block, wave).  This replaces the wave-major mapping (kept as
 * `gps0f_bwd_fixed_wave` for the A/B gate) which strided
 * `8·P·n_fix` bytes per step and reached only ~87 GB/s. */
__global__ void gps0f_bwd_fixed(
    const float2* __restrict__ afix,           /* [ne, n_fix, P]          */
    const float2* __restrict__ da,             /* [ne, P]  the tile's dA  */
    const double* __restrict__ dscale,         /* [ne] path B's group cs  */
    int n_fix, int ne, int P,
    double* __restrict__ g_ck_r, double* __restrict__ g_ck_i)
{
    extern __shared__ float2 sda[];            /* [blockDim]              */
    long long* soff = (long long*)(sda + blockDim.x);  /* [blockDim]       */
    double* sfus = (double*)(soff + blockDim.x);       /* [2·8·n_fix]      */
    const int tid = threadIdx.x, nt = blockDim.x;
    const int nsl0 = (n_fix > 0) ? (nt / n_fix) : 1;
    const int nsl = (nsl0 > 8) ? 8 : (nsl0 < 1 ? 1 : nsl0);
    const int nact = n_fix * nsl;
    const long long total = (long long)ne * (long long)P;
    const long long stride = (long long)gridDim.x * nt;
    for (long long base = (long long)blockIdx.x * nt; base < total;
         base += stride) {
        const int nv = (int)((total - base < nt) ? (total - base) : nt);
        if (tid < nv) {
            const long long gg = base + tid;
            const int e = (int)(gg / P);
            const int p = (int)(gg - (long long)e * P);
            const float2 d = da[gg];
            const double sc = (dscale != NULL) ? dscale[e] : 1.0;
            sda[tid] = make_float2((float)((double)d.x * sc),
                                   (float)((double)d.y * sc));
            soff[tid] = (long long)e * n_fix * P + p;
        }
        for (int i = tid; i < 2 * nact; i += nt) sfus[i] = 0.0;
        __syncthreads();
        for (int t = tid; t < nact; t += nt) {
            const int k = t % n_fix;
            const int sl = t / n_fix;
            double ur = 0.0, ui = 0.0;
            for (int i = sl; i < nv; i += nsl) {
                const float2 d = sda[i];
                const float2 a = afix[soff[i] + (long long)k * P];
                ur += (double)d.x * (double)a.x - (double)d.y * (double)a.y;
                ui += (double)d.x * (double)a.y + (double)d.y * (double)a.x;
            }
            if (ur != 0.0) atomicAdd(&sfus[(k * nsl + sl) * 2 + 0], ur);
            if (ui != 0.0) atomicAdd(&sfus[(k * nsl + sl) * 2 + 1], ui);
        }
        __syncthreads();
        for (int k = tid; k < n_fix; k += nt) {
            double ur = 0.0, ui = 0.0;
            for (int sl = 0; sl < nsl; sl++) {
                ur += sfus[(k * nsl + sl) * 2 + 0];
                ui += sfus[(k * nsl + sl) * 2 + 1];
            }
            if (ur != 0.0) atomicAdd(g_ck_r + k, ur);
            if (ui != 0.0) atomicAdd(g_ck_i + k, ui);
        }
        __syncthreads();   /* the shared chunk is reused */
    }
}

/*=============================================================================
 * The shared tile — ONE formula, used by the host (`create`) and the kernel
 * (the pointer offsets), so the launch size and the kernel's view cannot
 * disagree.  All offsets are in DOUBLES.
 *
 *   cr, ci  [te·W]     common of the FREE waves (W = P·Nfree, p-major)
 *   D       [2·te·Nfree] the finished denominator of a free wave
 *   rd      [te·Nfree]  1/|D|²  (one real reciprocal for every division)
 *   pv      [2·te·Nfree] dQ/dD_{e,k} (FREE waves only — a fixed wave has none)
 *   A, dA   [2·te·P]   A_{e,p} and dQ/dA_{e,p}
 *   pe, q, sc [te]     P_e, the group summand q_e, the group scale
 *   gck     [2·Nall]   per-wave shared accumulator for dQ/dck (fixed + free)
 *   gc      [NB]       per-coefficient shared accumulator (GLOBAL column axis)
 *   tot     [3]        path B: S_g, Pb_g, W_g totals
 *   gr, gi  [Nall]     path B: per-group dQ/dck accumulators
 *   gcb     [NB]       path B: per-group dQ/dcoeff accumulator
 *===========================================================================*/
typedef struct {
    int cr, ci, D, rd, pv, A, dA, pe, q, sc, gck, gc, tot, fus,
        gr, gi, gcb, total;
} GPS0FLayout;

__host__ __device__ __forceinline__ GPS0FLayout gps0f_layout(
    int te, int P, int Nfree, int Nall, int NB)
{
    GPS0FLayout L;
    const int W = P * Nfree;
    int o = 0;
    L.cr  = o; o += te * W;
    L.ci  = o; o += te * W;
    L.D   = o; o += 2 * te * Nfree;
    L.rd  = o; o += te * Nfree;
    L.pv  = o; o += 2 * te * Nfree;
    L.A   = o; o += 2 * te * P;
    L.dA  = o; o += 2 * te * P;
    L.pe  = o; o += te;
    L.q   = o; o += te;
    L.sc  = o; o += te;
    L.gck = o; o += 2 * Nall;
    L.gc  = o; o += NB;
    L.tot = o; o += 3;
    /* the all-fixed fused branch's per-(wave, slice) partials: 2·8·Nall.
       Zero bytes when free waves remain (the branch cannot run). */
    L.fus = o; o += (Nfree == 0) ? 16 * Nall : 0;
    L.gr  = o; o += Nall;
    L.gi  = o; o += Nall;
    L.gcb = o; o += NB;
    L.total = o;
    return L;
}

/*=============================================================================
 * The per-iteration tile: phases A -> B -> G -> B2 -> C -> D, one pass.
 *===========================================================================*/
__global__ void GPS0F_TILE_BOUNDS gps0f_tile_kernel(
    const float2* __restrict__ us,         /* [ne, n_uslot]   Z = i·basis   */
    const float2* __restrict__ amp,        /* [ne, n_uniq, P] fa·fl·num     */
    const int* __restrict__ slot_of_wave,  /* [Nall] -> amp slot (PERMUTED) */
    const float2* __restrict__ afix,       /* [ne, n_fix, P]  fixed common  */
    int n_fix,                             /* waves [0, n_fix) are fixed    */
    const double2* __restrict__ a1,        /* [ne, P] F's fixed sum, or NULL */
    float2* __restrict__ da_g,             /* [ne, P] the dA B reads, or NULL */
    double* __restrict__ dscale,           /* [ne] path B group scale, or NULL */
    const int* __restrict__ nz_m,          /* [nnz_free] basis column       */
    const int* __restrict__ nz_u,          /* [nnz_free] -> uslot           */
    const long long* __restrict__ nz_pair, /* [nnz_free] m | (u << 32)      */
    const int* __restrict__ nz_k,          /* [nnz_free] free wave, 0..N-1  */
    const int* __restrict__ gs_lo,         /* [N] free CSR start into nz_*  */
    const int* __restrict__ gs_hi,         /* [N] free CSR end              */
    const double* __restrict__ gamma,      /* [N·n_basis] free rows         */
    int n_nz,                              /* FREE non-zeros                */
    const double* __restrict__ coeff,      /* [n_basis]                     */
    const double* __restrict__ ck_r, const double* __restrict__ ck_i,
    const double* __restrict__ weight, const double* __restrict__ bkg,
    double norm, int use_norm,
    int rsize, int per_blk, int te,
    int N, int P, int NB, int n_uslot, int n_uniq, int Nall,
    int ne, int fused,
    double* __restrict__ P_dev,            /* [ne]  per-event P (output) */
    double* __restrict__ g_ck_r, double* __restrict__ g_ck_i,
    double* __restrict__ g_coeff,
    double* __restrict__ Q_acc, double* __restrict__ dnorm_acc)
{
    const int tid = threadIdx.x, nt = blockDim.x;
    const GPS0FLayout L = gps0f_layout(te, P, N, Nall, NB);

    extern __shared__ double sh[];
    double* s_cr  = sh + L.cr;
    double* s_ci  = sh + L.ci;
    double* s_D   = sh + L.D;
    double* s_rd  = sh + L.rd;
    double* s_pv  = sh + L.pv;
    double* s_A   = sh + L.A;
    double* s_dA  = sh + L.dA;
    double* s_pe  = sh + L.pe;
    double* s_q   = sh + L.q;
    double* s_sc  = sh + L.sc;
    double* s_gck = sh + L.gck;
    double* s_gc  = sh + L.gc;
    double* s_tot = sh + L.tot;
    double* s_fus = sh + L.fus;
    double* s_gr  = sh + L.gr;
    double* s_gi  = sh + L.gi;
    double* s_gcb = sh + L.gcb;

    const int e0 = blockIdx.x * per_blk;
    const int evn = min(per_blk, ne - e0);
    if (evn <= 0) return;
    /* path B: one block owns ONE whole group and walks it internally */
    const int pathB = (per_blk > te);
    const int rs = (rsize < 1) ? 1 : rsize;
    const int nchunk = pathB ? ((evn + te - 1) / te) : 1;

    /* gradient targets: the global accumulators (path A) or the block's
       per-group accumulators (path B, scaled once at group close) */
    double* d_ck_r = g_ck_r;
    double* d_ck_i = g_ck_i;
    double* d_co = g_coeff;
    if (pathB) {
        d_ck_r = s_gr; d_ck_i = s_gi; d_co = s_gcb;
        for (int k = tid; k < Nall; k += nt) { s_gr[k] = 0.0; s_gi[k] = 0.0; }
        for (int m = tid; m < NB; m += nt) s_gcb[m] = 0.0;
        if (tid == 0) { s_tot[0] = 0.0; s_tot[1] = 0.0; s_tot[2] = 0.0; }
    }
    for (int k = tid; k < 2 * Nall; k += nt) s_gck[k] = 0.0;
    for (int m = tid; m < NB; m += nt) s_gc[m] = 0.0;
    __syncthreads();

    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int nwarps = nt >> 5;

    /* ── ALL-FIXED FUSED BRANCH (`fused`, uniform per launch) ──────────
     *
     * When every wave is fixed there is no free CSR and no `common`: A is
     * built straight from `Afix`, the objective is closed in this block and
     * the fixed block's dck is back-propagated here.  F, the tile's phases
     * and B are not launched, and A1/dA do not exist for this handle.  The
     * branch is UNIFORM (the host sets `fused` from the context), so every
     * `__syncthreads()` below is reached by all threads of the block; the
     * mixed path after it keeps its barrier sequence byte-for-byte.
     *
     * Shared: s_A, s_dA (2·te·P each, already in the layout), s_pe/s_q/s_sc
     * (te), s_gck (2·Nall) and s_tot[0] (the one group's scale).
     * The backward pass is the second k-sweep: same `Afix` lines, per-wave
     * partials in shared, one global atomic per (block, wave). */
    if (fused) {
        const int ev = evn;
        for (int idx = tid; idx < ev * P; idx += nt) {
            const int e = idx / P, p = idx - e * P;
            const float2* base = afix + ((size_t)(e0 + e) * n_fix) * P + p;
            double ar = 0.0, ai = 0.0;
            for (int k = 0; k < n_fix; k++) {
                const float2 a = base[(size_t)k * P];
                const double cr = ck_r[k], ci = ck_i[k];
                ar += cr * (double)a.x - ci * (double)a.y;
                ai += cr * (double)a.y + ci * (double)a.x;
            }
            s_A[idx * 2 + 0] = ar;
            s_A[idx * 2 + 1] = ai;
        }
        __syncthreads();
        for (int e = tid; e < ev; e += nt) {
            double Pe = 0.0;
            for (int p = 0; p < P; p++) {
                const double ar = s_A[(e * P + p) * 2 + 0];
                const double ai = s_A[(e * P + p) * 2 + 1];
                Pe += ar * ar + ai * ai;
            }
            const int ge = e0 + e;
            const double wv = weight[ge], bv = bkg[ge];
            s_pe[e] = Pe;
            s_q[e] = use_norm ? wv * (Pe / norm + bv) : wv * Pe;
            P_dev[ge] = Pe;
        }
        __syncthreads();
        if (rs == 1) {
            for (int e = tid; e < ev; e += nt) {
                const double wv = weight[e0 + e];
                if (use_norm) {
                    if (wv > 0.0) {
                        s_sc[e] = -wv / s_q[e];
                        atomicAdd(Q_acc, -wv * log(s_q[e] / wv));
                        atomicAdd(dnorm_acc, wv * (wv * s_pe[e])
                                  / (s_q[e] * norm * norm));
                    } else {
                        s_sc[e] = 0.0;
                    }
                } else {
                    s_sc[e] = 1.0;
                    atomicAdd(Q_acc, s_q[e]);
                }
            }
            __syncthreads();
            for (int idx = tid; idx < ev * P; idx += nt) {
                const int e = idx / P;
                double dqp = use_norm ? (weight[e0 + e] / norm)
                                      : weight[e0 + e];
                dqp *= s_sc[e];
                const double ar = s_A[idx * 2 + 0];
                const double ai = s_A[idx * 2 + 1];
                s_dA[idx * 2 + 0] = dqp * ar;
                s_dA[idx * 2 + 1] = -dqp * ai;
            }
        } else {
            /* whole groups per block (per_blk is a multiple of rsize): the
               group sums are parallel over the groups, one barrier before the
               per-event dA scale (s_sc[g] = -W_g/S_g) */
            const int ng = (ev + rs - 1) / rs;
            for (int g = tid; g < ng; g += nt) {
                const int a = g * rs;
                const int b = (a + rs < ev) ? (a + rs) : ev;
                double S = 0.0, WP = 0.0, W = 0.0;
                for (int e = a; e < b; e++) {
                    S += s_q[e];
                    WP += weight[e0 + e] * s_pe[e];
                    W += weight[e0 + e];
                }
                if (use_norm) {
                    if (W > 0.0) {
                        s_sc[g] = -W / S;
                        atomicAdd(Q_acc, -W * log(S / W));
                        atomicAdd(dnorm_acc, W * WP / (S * norm * norm));
                    } else {
                        s_sc[g] = 0.0;
                    }
                } else {
                    s_sc[g] = 1.0;
                    atomicAdd(Q_acc, S);
                }
            }
            __syncthreads();
            for (int idx = tid; idx < ev * P; idx += nt) {
                const int e = idx / P;
                double dqp = use_norm ? (weight[e0 + e] / norm)
                                      : weight[e0 + e];
                dqp *= s_sc[e / rs];
                const double ar = s_A[idx * 2 + 0];
                const double ai = s_A[idx * 2 + 1];
                s_dA[idx * 2 + 0] = dqp * ar;
                s_dA[idx * 2 + 1] = -dqp * ai;
            }
        }
        __syncthreads();
        /* BACKWARD: the second k-sweep, but with (wave, slice) partials.
         *
         * The naive shape (thread = (e,p), one shared atomic per (e,p,k))
         * measured 35-51 ns/event — the shared atomic on 2·n_fix addresses
         * from every thread dominated.  Here a thread owns (k, slice) and
         * walks the block's elements at a stride of `nsl`, so the shared
         * atomic count drops from 2·ev·P·n_fix to 2·n_fix·nsl, and the block
         * still does exactly ONE global atomic per wave.  The Afix reads keep
         * the contiguous k run (the same lines phase 1 read). */
        const int nsl0 = (n_fix > 0) ? (nt / n_fix) : 1;
        const int nsl = (nsl0 > 8) ? 8 : (nsl0 < 1 ? 1 : nsl0);
        const int nact = n_fix * nsl;
        const int elems = ev * P;
        for (int i = tid; i < 2 * nact; i += nt) s_fus[i] = 0.0;
        __syncthreads();
        for (int t = tid; t < nact; t += nt) {
            const int k = t % n_fix;
            const int sl = t / n_fix;
            double ur = 0.0, ui = 0.0;
            for (int i = sl; i < elems; i += nsl) {
                const int e = i / P, p = i - e * P;
                const float2 a =
                    afix[((size_t)(e0 + e) * n_fix + k) * P + p];
                const double dr = s_dA[i * 2 + 0], di = s_dA[i * 2 + 1];
                ur += dr * (double)a.x - di * (double)a.y;
                ui += dr * (double)a.y + di * (double)a.x;
            }
            if (ur != 0.0) atomicAdd(&s_fus[(k * nsl + sl) * 2 + 0], ur);
            if (ui != 0.0) atomicAdd(&s_fus[(k * nsl + sl) * 2 + 1], ui);
        }
        __syncthreads();
        for (int k = tid; k < n_fix; k += nt) {
            double ur = 0.0, ui = 0.0;
            for (int sl = 0; sl < nsl; sl++) {
                ur += s_fus[(k * nsl + sl) * 2 + 0];
                ui += s_fus[(k * nsl + sl) * 2 + 1];
            }
            if (ur != 0.0) atomicAdd(g_ck_r + k, ur);
            if (ui != 0.0) atomicAdd(g_ck_i + k, ui);
        }
        return;
    }

    for (int c = 0; c < nchunk; c++) {
        const int ce0 = e0 + c * te;
        int nen = pathB ? (evn - c * te) : evn;
        if (nen > te) nen = te;

        /* ── phase A: D_{e,k}, 1/|D|², and common_{e,p,k} ─────────────────
         *
         * The denominator is shared by the projections (it belongs to the
         * CHAIN, not to the entry), so it is computed once per (e,k) and the
         * thread then emits all P common values of its (e,k) slice — that is
         * also why no barrier is needed between the two.
         *
         * `1/|D|²` is a REAL reciprocal: D is complex, so `1/D` is
         * `conj(D)/|D|²` and one fp64 divide serves every consumer
         * (fp64 divide is ~1/64 rate and unpipelined on a consumer GPU).
         *
         * s0f: `N` is the FREE wave count; a fixed wave's `common` is already
         * in `Afix` and this loop never visits it. */
        for (int idx = tid; idx < nen * N; idx += nt) {
            const int e = idx / N, k = idx - e * N;
            const int kw = n_fix + k;        /* permuted: free block follows */
            const float2* uu = us + (size_t)(ce0 + e) * n_uslot;
            const int lo = gs_lo[k], hi = gs_hi[k];
            /* s0e2's ONE packed 64-bit load per CSR step —
               (coeff index) | (uslot << 32) — and TWO independent fp64
               accumulation chains, so a k that walks nnz > 1 does not
               serialize on a single dependency.  Same fp64 ops, same
               arithmetic; only the summation order differs (~1e-16). */
            double sr0 = 0.0, si0 = 0.0, sr1 = 0.0, si1 = 0.0;
            int i = lo;
            const int two = lo + ((hi - lo) & ~1);
            for (; i < two; i += 2) {
                const long long p0 = nz_pair[i], p1 = nz_pair[i + 1];
                const double c0 = coeff[(int)(p0 & 0xFFFFFFFFLL)];
                const double c1 = coeff[(int)(p1 & 0xFFFFFFFFLL)];
                const float2 z0 = uu[(int)(p0 >> 32)];
                const float2 z1 = uu[(int)(p1 >> 32)];
                sr0 += c0 * (double)z0.y;
                si0 -= c0 * (double)z0.x;
                sr1 += c1 * (double)z1.y;
                si1 -= c1 * (double)z1.x;
            }
            if (i < hi) {
                const long long p = nz_pair[i];
                const double c = coeff[(int)(p & 0xFFFFFFFFLL)];
                const float2 z = uu[(int)(p >> 32)];
                sr0 += c * (double)z.y;
                si0 -= c * (double)z.x;
            }
            const double sr = sr0 + sr1, si = si0 + si1;
            const double rden = 1.0 / (sr * sr + si * si);
            s_D[idx * 2 + 0] = sr;
            s_D[idx * 2 + 1] = si;
            s_rd[idx] = rden;
            const size_t abase =
                (size_t)(ce0 + e) * n_uniq * P + (size_t)slot_of_wave[kw] * P;
            for (int p = 0; p < P; p++) {
                const float2 a = amp[abase + p];
                const int o = (e * P + p) * N + k;
                s_cr[o] = ((double)a.x * sr + (double)a.y * si) * rden;
                s_ci[o] = ((double)a.y * sr - (double)a.x * si) * rden;
            }
        }
        __syncthreads();

        /* ── phase B: A_{e,p}, P_e, the group summand q_e ─────────────────
         * One warp per event; the k reduction is a warp shuffle (no shared
         * atomics, unlike the gen2/1bw shape).
         *
         * s0f: the free block is the parent's loop (`cr[k]`, `ck_r[n_fix+k]`);
         * the fixed block adds its lane's share of `ck_k·Afix` before the SAME
         * shuffle, so the reduction is one and the arithmetic is unchanged
         * when `n_fix == 0` (adding exactly 0.0). */
        for (int e = wid; e < nen; e += nwarps) {
            for (int p = 0; p < P; p++) {
                const double* cr = s_cr + (e * P + p) * N;
                const double* ci = s_ci + (e * P + p) * N;
                /* the split's fixed sum starts the reduction ONCE (lane 0 —
                   every lane starting from it would multiply it by 32); with
                   no split this is the parent's zero start. */
                double ar = 0.0, ai = 0.0;
                if (a1 != NULL && lane == 0) {
                    const double2 v = a1[(size_t)(ce0 + e) * P + p];
                    ar = v.x;
                    ai = v.y;
                }
                for (int k = lane; k < N; k += 32) {
                    const int kw = n_fix + k;
                    const double ckr = ck_r[kw], cki = ck_i[kw];
                    ar += ckr * cr[k] - cki * ci[k];
                    ai += ckr * ci[k] + cki * cr[k];
                }
                /* no split: the fixed block lives here, as before */
                if (a1 == NULL) {
                    for (int k = lane; k < n_fix; k += 32) {
                        const float2 a =
                            afix[((size_t)(ce0 + e) * n_fix + k) * P + p];
                        const double ckr = ck_r[k], cki = ck_i[k];
                        ar += ckr * (double)a.x - cki * (double)a.y;
                        ai += ckr * (double)a.y + cki * (double)a.x;
                    }
                }
                for (int off = 16; off > 0; off >>= 1) {
                    ar += __shfl_down_sync(0xffffffffu, ar, off);
                    ai += __shfl_down_sync(0xffffffffu, ai, off);
                }
                if (lane == 0) {
                    s_A[(e * P + p) * 2 + 0] = ar;
                    s_A[(e * P + p) * 2 + 1] = ai;
                }
            }
            __syncwarp();
            if (lane == 0) {
                double Pe = 0.0;
                for (int p = 0; p < P; p++) {
                    const double ar = s_A[(e * P + p) * 2 + 0];
                    const double ai = s_A[(e * P + p) * 2 + 1];
                    Pe += ar * ar + ai * ai;
                }
                const int ge = ce0 + e;
                const double wv = weight[ge], bv = bkg[ge];
                s_pe[e] = Pe;
                s_q[e] = use_norm ? wv * (Pe / norm + bv) : wv * Pe;
                P_dev[ge] = Pe;
            }
        }
        __syncthreads();

        /* ── phase G: the group summaries ──────────────────────────────────
         * path A: whole groups live in this tile -> s_sc[g] holds -W_g/S_g.
         * path B: thread 0 folds this inner chunk into the block's totals;
         * the scale exists only at group close. */
        if (!pathB) {
            const int ng = (nen + rs - 1) / rs;
            for (int g = tid; g < ng; g += nt) {
                const int a = g * rs;
                const int b = (a + rs < nen) ? (a + rs) : nen;
                double S = 0.0, WP = 0.0, W = 0.0;
                for (int e = a; e < b; e++) {
                    S += s_q[e];
                    WP += weight[ce0 + e] * s_pe[e];
                    W += weight[ce0 + e];
                }
                if (use_norm) {
                    if (W > 0.0) {              /* a zero-weight group is inert */
                        s_sc[g] = -W / S;
                        atomicAdd(Q_acc, -W * log(S / W));
                        atomicAdd(dnorm_acc, W * WP / (S * norm * norm));
                    } else {
                        s_sc[g] = 0.0;
                    }
                } else {
                    s_sc[g] = 1.0;              /* linear objective */
                    atomicAdd(Q_acc, S);
                }
            }
        } else if (tid == 0) {
            double S = 0.0, WP = 0.0, W = 0.0;
            for (int e = 0; e < nen; e++) {
                S += s_q[e];
                WP += weight[ce0 + e] * s_pe[e];
                W += weight[ce0 + e];
            }
            s_tot[0] += S; s_tot[1] += WP; s_tot[2] += W;
        }
        __syncthreads();

        /* ── phase B2: dQ/dP_e -> dQ/dA_p ─────────────────────────────────
         * path B keeps the UNSCALED per-event factor: the group constant is
         * applied to the finished accumulators at close. */
        for (int idx = tid; idx < nen * P; idx += nt) {
            const int e = idx / P, p = idx - e * P;
            const double wv = weight[ce0 + e];
            double dqp = use_norm ? (wv / norm) : wv;
            if (!pathB) dqp *= s_sc[e / rs];
            const double ar = s_A[(e * P + p) * 2 + 0];
            const double ai = s_A[(e * P + p) * 2 + 1];
            s_dA[idx * 2 + 0] = dqp * ar;
            s_dA[idx * 2 + 1] = -dqp * ai;
            /* kernel B is the only reader, so the write exists only when the
               split is on (dA_g == NULL is the parent's path) */
            if (da_g != NULL) {
                da_g[(size_t)(ce0 + e) * P + p] =
                    make_float2((float)(dqp * ar), (float)(-dqp * ai));
            }
        }
        __syncthreads();

        /* ── phase C: u_{e,k} = Σ_p dQ/dA_{e,p}·common_{e,p,k}  (= dQ/dck_k)
         *            dQ/dD_{e,k} = -ck_k·u_{e,k}/D_{e,k}
         *
         * The two reductions the split kernels do in separate passes are the
         * same complex number here, because `common` is shared by the `p` of
         * a wave (the u-merge, at one BW per chain).  `dQ/dD` carries the
         * `ck` factor — `common` is linear in the amplitude and `D` is not.
         *
         * s0f: the free block is the parent's loop.  The fixed block has no
         * `D` (hence no `pv`): its `common` IS the cache.  With the split on,
         * its `dQ/dck_k` is kernel B's GEMV over the finished `dA`; with the
         * split off it is the in-tile loop below.  Either way `dQ/dcoeff[m]`
         * for a fixed column is exactly 0 — the accepted contract (a fixed
         * column has no `x`). */
        for (int idx = tid; idx < nen * N; idx += nt) {
            const int e = idx / N, k = idx - e * N;
            const int kw = n_fix + k;
            const double ckr = ck_r[kw], cki = ck_i[kw];
            double ur = 0.0, ui = 0.0;
            for (int p = 0; p < P; p++) {
                const int o = (e * P + p) * N + k;
                const double dr = s_dA[(e * P + p) * 2 + 0];
                const double di = s_dA[(e * P + p) * 2 + 1];
                const double cr = s_cr[o], ci = s_ci[o];
                ur += dr * cr - di * ci;
                ui += dr * ci + di * cr;
            }
            atomicAdd(&s_gck[2 * kw + 0], ur);
            atomicAdd(&s_gck[2 * kw + 1], ui);
            const double dr = s_D[idx * 2 + 0], di = s_D[idx * 2 + 1];
            const double rden = s_rd[idx];
            const double qr = ckr * ur - cki * ui;   /* ck·u */
            const double qi = ckr * ui + cki * ur;
            s_pv[idx * 2 + 0] = -(qr * dr + qi * di) * rden;  /* -ck·u/D */
            s_pv[idx * 2 + 1] = -(qi * dr - qr * di) * rden;
        }
        if (n_fix > 0 && a1 == NULL) {
            for (int idx = tid; idx < nen * n_fix; idx += nt) {
                const int e = idx / n_fix, k = idx - e * n_fix;
                const double ckr = ck_r[k], cki = ck_i[k];
                double ur = 0.0, ui = 0.0;
                for (int p = 0; p < P; p++) {
                    const float2 a =
                        afix[((size_t)(ce0 + e) * n_fix + k) * P + p];
                    const double dr = s_dA[(e * P + p) * 2 + 0];
                    const double di = s_dA[(e * P + p) * 2 + 1];
                    ur += dr * (double)a.x - di * (double)a.y;
                    ui += dr * (double)a.y + di * (double)a.x;
                }
                atomicAdd(&s_gck[2 * k + 0], ur);
                atomicAdd(&s_gck[2 * k + 1], ui);
            }
        }
        __syncthreads();

        /* ── phase D: dQ/dcoeff_m = 2Re Σ dQ/dD_{e,k}·gamma[k,m]·basis_m(m_e)
         *
         * Walked over the host-built CSR of TRUE non-zeros — one thread per
         * entry, serial over the tile — so a folded or holed row costs its
         * non-zeros, not its span (92 entries over 26 waves on the real
         * config, with interior holes in 23 of them).
         *
         * `Z` is stored rotated (`Z = i·basis`), so `F_m = -i·Z_m` and
         * `2Re(dQ/dD·F_m) = 2·(pv_r·Z_i + pv_i·Z_r)`.
         *
         * s0f: only the FREE CSR exists here (`n_nz` entries, `k` renumbered
         * 0..N-1) — a fixed wave has no `pv`, so its columns get exactly
         * 0.0, never a stale `s_gc` reading. */
        for (int idx = tid; idx < n_nz; idx += nt) {
            const int k = nz_k[idx], m = nz_m[idx], u = nz_u[idx];
            const double g = gamma[k * NB + m];
            double out = 0.0;
            for (int e = 0; e < nen; e++) {
                const float2 z = us[(size_t)(ce0 + e) * n_uslot + u];
                const size_t o = (size_t)(e * N + k) * 2;
                out += 2.0 * (s_pv[o + 0] * (double)z.y
                              + s_pv[o + 1] * (double)z.x);
            }
            if (out != 0.0) atomicAdd(&s_gc[m], out * g);
        }
        __syncthreads();

        if (pathB) {
            /* fold this inner chunk into the block's group accumulators
               (plain adds: every index has one owner after the barrier) */
            for (int k = tid; k < Nall; k += nt) {
                s_gr[k] += s_gck[2 * k + 0];
                s_gi[k] += s_gck[2 * k + 1];
                s_gck[2 * k + 0] = 0.0;
                s_gck[2 * k + 1] = 0.0;
            }
            for (int m = tid; m < NB; m += nt) {
                s_gcb[m] += s_gc[m];
                s_gc[m] = 0.0;
            }
            __syncthreads();   /* the tile is reused by the next inner chunk */
        }
    }

    /* ── flush ────────────────────────────────────────────────────────────
     * path A: the accumulators are complete (one tile = whole groups).
     * path B: close the group, then rescale the finished accumulators by
     *         const_g = -W_g/S_g (0 for a zero-weight group). */
    if (pathB) {
        if (tid == 0) {
            const double S = s_tot[0], WP = s_tot[1], W = s_tot[2];
            if (use_norm) {
                if (W > 0.0) {
                    atomicAdd(Q_acc, -W * log(S / W));
                    atomicAdd(dnorm_acc, W * WP / (S * norm * norm));
                    s_sc[0] = -W / S;
                } else {
                    s_sc[0] = 0.0;
                }
            } else {
                atomicAdd(Q_acc, S);
                s_sc[0] = 1.0;
            }
        }
        __syncthreads();
        const double cs = s_sc[0];
        /* path B's dA_g was written with the UNSCALED per-event factor (its
           group constant exists only now), so publish the scale for B */
        if (dscale != NULL) {
            for (int e = 0; e < evn; e++) dscale[e0 + e] = cs;
        }
        for (int k = tid; k < Nall; k += nt) {
            atomicAdd(g_ck_r + k, cs * s_gr[k]);
            atomicAdd(g_ck_i + k, cs * s_gi[k]);
        }
        for (int m = tid; m < NB; m += nt)
            if (s_gcb[m] != 0.0) atomicAdd(g_coeff + m, cs * s_gcb[m]);
    } else {
        for (int k = tid; k < Nall; k += nt) {
            atomicAdd(g_ck_r + k, s_gck[2 * k + 0]);
            atomicAdd(g_ck_i + k, s_gck[2 * k + 1]);
        }
        for (int m = tid; m < NB; m += nt)
            if (s_gc[m] != 0.0) atomicAdd(g_coeff + m, s_gc[m]);
    }
}

/*=============================================================================
 * Host side
 *===========================================================================*/
typedef struct {
    int *fl_q_index, *num_m_index, *bas_m_index, *angle_index;
    int *uslot_row, *uslot_col, *slot_of_wave, *rep_of_slot;
    int *nz_k, *nz_m, *nz_u, *gs_lo, *gs_hi;
    long long *nz_pair;               /* [n_nz_free] packed FREE CSR        */
    /* the fixed block's own CSR (fill only — the tile never walks it) */
    int *fix_lo, *fix_hi, *fix_slot_of_wave;
    long long *fix_pair;              /* [n_nz_fix] packed                   */
    double *fl_table, *num_r, *num_i, *bas_r, *bas_i, *gamma;
    double *angle_k, *angle_b, *am_r, *am_i;
    double m_min, m_delta, q_min, q_delta;
    int m_bins, q_bins;
    int n_wave, n_proj, n_res, n_decay, n_basis;
    int n_ab, n_comp, n_mass, n_momentum, n_angle_total;
    int n_uniq, n_uslot;
    int n_nz;                         /* FREE non-zeros (phase D)            */
    int n_fix, n_free;                /* permuted wave-axis split            */
    int n_nz_fix;                     /* fixed-block non-zeros               */
    int split_fixed;                  /* 1 = F/T/B, 0 = fixed block in T     */
    int bwd_mode;                     /* 0 = event-major B, 1 = wave-major   */
    int fused_enable;                 /* 1 = allow the all-fixed fused tile  */
    int launches[3];                  /* [F, B, fused] counts (the gates)    */
    int batch_size, rsize;
    /* chunk rule (one definition, in `create`) */
    int cap, s_ngroups;
    /* tile size chosen at create (depends on rsize + the shape) */
    int te;
    int smem_bytes;
    /* the split's scratch: ONE per context, `cap`-sized (it depends on
       batch_size / n_proj / the mask, never on the sample).  A mask change
       recreates the context, so it is reallocated with it — correct, since
       it is mask-gated.  NULL on the fused and mask-less paths. */
    double2 *a1;                      /* [cap, n_proj]  F's A1           */
    float2 *da;                       /* [cap, n_proj]  the tile's dA     */
    double *dscale;                   /* [cap]          path B's scale    */
    /* parameter staging (refreshed per call, never malloc'd per call) */
    double *p_ck_r, *p_ck_i, *p_coeff;
    /* the mask's DECLARED VALUES, as a FULL `n_basis` vector so the fill
       kernel reads it exactly like `coeff` (no index remap, no forked
       kernel).  Context-owned, hence per mask, and NEVER NULL when
       `n_fix > 0` — `create` refuses that combination, so there is no lazy
       path. */
    double *fixed_vals;
    double *h_ck_r, *h_ck_i, *h_coeff;
    double *acc, *h_acc;              /* [Q, dnorm] */
    double *g_ck_r, *g_ck_i, *g_coeff;
    cudaStream_t stream;
} GPS0FCtx;

/* ONE SAMPLE.  Per-EVENT data only: the two caches, weight/bkg, and the
   caller-visible per-event P OUTPUT (not scratch — see the file header). */
typedef struct {
    double *weight, *bkg;
    float2 *us;           /* [ne, n_uslot]        Z = i·basis        */
    float2 *amp;          /* [ne, n_uniq, n_proj] fa·fl·num          */
    float2 *afix;         /* [ne, n_fix, n_proj]  fixed common        */
    double *P_dev;        /* [ne] per-event P (output of the call)   */
    double *h_P;          /* pinned staging for the one D2H          */
    int ne;
    int afix_ready;       /* the per-handle Afix fill has run        */
} GPS0FHandle;

static int* _gps0f_up_i(const int* src, int n) {
    if (n <= 0 || src == NULL) return NULL;
    int* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(int),
                          cudaMemcpyHostToDevice));
    return d;
}

static long long* _gps0f_up_l(const long long* src, int n) {
    if (n <= 0 || src == NULL) return NULL;
    long long* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(long long)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(long long),
                          cudaMemcpyHostToDevice));
    return d;
}

static double* _gps0f_up_d(const double* src, int n) {
    if (n <= 0 || src == NULL) return NULL;
    double* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(double),
                          cudaMemcpyHostToDevice));
    return d;
}

/* The chunk rule.  ONE definition, asked for through `cuda_cap_*`. */
static int gps0f_chunk(int batch_size, int rsize)
{
    int cap = (batch_size > 0) ? batch_size : 50000;
    if (cap > GPS0F_MAX_CAP) cap = GPS0F_MAX_CAP;
    if (rsize > 1) {
        const int a = (cap / rsize) * rsize;
        cap = (a > 0) ? a : rsize;   /* never 0, never unaligned to a group */
    }
    return cap;
}

/* The tile size.  Goals, in order: (1) whole groups per tile when `rsize` is
 * small enough (path A — the group sums then live in the tile); (2) a tile of
 * at least ~GPS0F_DEF_TE events so a block has real work; (3) the shared tile
 * inside GPS0F_SMEM_BUDGET, which is what actually binds.
 *
 * MEASURED on the real 8.75 M-event `jpsi_eta_pi_pi` sample (n_wave 26,
 * n_proj 2, n_basis 69, rsize 17, `Fitter.get_nll`, min of 4):
 *
 *     te   1     2     4     8     17
 *     ms   476   385   344   325   333
 *
 * i.e. a tile of GPS0F_DEF_TE events with path B (one block per group, walked
 * internally in 8-event chunks) beats a 17-event tile that holds whole groups
 * — smaller shared memory means better occupancy, and that outweighs the
 * group-walk bookkeeping.  So `te` is kept near GPS0F_DEF_TE and only rounded
 * UP to a whole number of groups when `rsize <= GPS0F_DEF_TE` (where path A is
 * free).  On the 100-wave benchmark shape the same rule shrinks te to 4 by the
 * shared-memory budget, which is that shape's measured optimum too (6.04 vs
 * 6.08 ms at te=8, 5.66 ms with rsize 17). */
static int gps0f_pick_te(int rsize, int P, int Nfree, int Nall, int NB)
{
    int te;
    if (rsize > 1 && rsize <= GPS0F_DEF_TE) {
        const int k = (GPS0F_DEF_TE + rsize - 1) / rsize;   /* ceil */
        te = rsize * (k < 1 ? 1 : k);                     /* whole groups */
    } else {
        te = GPS0F_DEF_TE;         /* rsize == 1, or path B (rsize > DEF_TE) */
    }
    /* `gps0f_launch` floors a tile to whole groups on its own, so shrinking is
     * always valid: below `rsize` the block simply owns one group (path B). */
    while (te > 1) {
        const size_t sm =
            (size_t)gps0f_layout(te, P, Nfree, Nall, NB).total
            * sizeof(double);
        if (sm <= (size_t)GPS0F_SMEM_BUDGET) break;
        te -= 1;
    }
    if (te < 1) te = 1;
    /* The budget shrink above steps by ONE event, which can land on a tile
     * `create` rejects (`te >= rsize` must be a whole number of groups) — a
     * wide shape at rsize 4 shrank 8 -> 7 and failed to construct.  Floor to
     * a whole group; the smaller tile is by definition still inside the
     * budget, and `te < rsize` (one group walked internally) stays legal. */
    if (rsize > 1 && te >= rsize) te = (te / rsize) * rsize;
    return te < 1 ? 1 : te;
}

/* The tile size when the split is ACTIVE (`n_fix > 0`).
 *
 * The parent's rule is calibrated for the full-wave tile; here the shared
 * tile holds only the free block, so its 8-event default is too small.
 * MEASURED on the CTPWA 6 M-event sample, rsize 17, split on, interleaved
 * min-of-3, `get_nll`:
 *
 *     te            8      17      34      68     136
 *     mixed 27/7  96.2   91.8    92.3    93.8   108.5   ms
 *     all-fixed   71.7   68.8    65.9    64.9    64.5   ms
 *
 * i.e. ONE whole group (`te == rsize`) is the mixed optimum, while the
 * all-fixed tile is barrier-only and grows until the budget binds.  So:
 * whole groups when free waves remain, and the largest whole-group tile up
 * to `8·rsize` when `n_free == 0` (both still inside the smem budget).
 * Mask-off never reaches here: `n_fix == 0` keeps the parent's rule, hence
 * its path. */
static int gps0f_pick_te_split(int rsize, int P, int Nfree, int Nall, int NB,
                               int base_te)
{
    const int unit = (rsize > 1) ? rsize : 8;
    const auto fits = [&](int te) {
        return (size_t)gps0f_layout(te, P, Nfree, Nall, NB).total
               * sizeof(double) <= (size_t)GPS0F_SMEM_BUDGET;
    };
    int te = base_te;
    if (Nfree > 0) {
        /* one whole group (path A), if the smaller tile holds it */
        if (rsize > te && fits(rsize)) te = rsize;
        return te;
    }
    /* no free waves: walk the whole-group ladder up to 8 groups */
    for (int k = 1; k <= 8; k *= 2) {
        const int cand = unit * k;
        if (!fits(cand)) break;
        te = cand;
    }
    return te;
}

extern "C" {

void cuda_free_context_flat_pwa_gen_s0f(void* vctx);   /* fwd decl */
static void gps0f_fill_afix_launch(GPS0FCtx*, GPS0FHandle*);

void* cuda_create_context_flat_pwa_gen_s0f(
    const double* fl_table, int n_fl,
    const int* fl_q_index, int n_fq,
    const double* num_r, const double* num_i, int n_num,
    const int* num_m_index, int n_nm,
    const double* bas_r, const double* bas_i, int n_bas,
    const int* bas_m_index, int n_bm,
    const int* uslot_row, int n_ur, const int* uslot_col, int n_uc,
    const int* slot_of_wave, int n_sw, const int* rep_of_slot, int n_rp,
    int n_uniq, int n_uslot,
    const int* nz_k, int n_nk, const int* nz_m, int n_nzm,
    const int* nz_u, int n_nzu,
    const long long* nz_pair, int n_np,
    const int* gs_lo, int n_lo, const int* gs_hi, int n_hi,
    const int* fix_lo, int n_flo, const int* fix_hi, int n_fhi,
    const int* fix_slot_of_wave, int n_fsw,
    const long long* fix_pair, int n_nfp,
    int n_fix, int n_free,
    const double* gamma, int n_g,
    const int* angle_index, int n_ai,
    const double* angle_k, int n_ak, const double* angle_b, int n_ab_,
    const double* am_r, const double* am_i, int n_am,
    double m_min, double m_delta, int m_bins,
    double q_min, double q_delta, int q_bins,
    int n_wave, int n_proj, int n_res, int n_decay, int n_basis,
    int n_ab, int n_comp, int n_mass, int n_momentum, int n_angle_total,
    int batch_size, int rsize, int te_hint, int split_fixed,
    int bwd_mode, int fused_enable,
    const double* fixed_vals, int n_fv)
{
    if (n_res != 1) {
        /* the wrapper rejects this first (with the general backend named);
           the kernel refuses too rather than reading a 3-D gamma as 1 factor */
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0f: n_res must be 1, got %d — use "
                "cuda_flat_pwa_gen2 for multi-BW chains\n", n_res);
        return NULL;
    }
    if (n_sw != n_wave || n_rp != n_uniq) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0f: slot_of_wave/rep_of_slot length "
                "mismatch (%d vs n_wave %d, %d vs n_uniq %d)\n",
                n_sw, n_wave, n_rp, n_uniq);
        return NULL;
    }
    if (n_fix < 0 || n_free < 0 || n_fix + n_free != n_wave) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0f: the permuted wave split must cover the "
                "axis exactly (n_fix %d + n_free %d != n_wave %d)\n",
                n_fix, n_free, n_wave);
        return NULL;
    }
    if (n_nk != n_nzm || n_nk != n_nzu || n_np != n_nzm
            || n_lo != n_free || n_hi != n_free) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0f: FREE CSR length mismatch "
                "(nz %d/%d/%d/%d, gs %d/%d, n_free %d)\n",
                n_nk, n_nzm, n_nzu, n_np, n_lo, n_hi, n_free);
        return NULL;
    }
    if (n_flo != n_fix || n_fhi != n_fix || n_fsw != n_fix) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0f: fixed-block length mismatch "
                "(lo %d, hi %d, slot %d, n_fix %d)\n",
                n_flo, n_fhi, n_fsw, n_fix);
        return NULL;
    }
    if (fixed_vals == NULL && n_fix > 0) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0f: a coefficient mask is declared WITH its values — "
                "this context has n_fix %d and no fixed_vals (the cache could "
                "not be filled at load)\n", n_fix);
        return NULL;
    }
    /* one entry per basis column: the fill reads the vector exactly like
       `coeff`.  `n_fix == 0` with values is legal and inert — the mask may
       declare columns that no wave reads entirely (it then caches nothing,
       which the wrapper warns about); the reverse (a fixed wave with no
       values) is refused below, because the fill would have nothing to read. */
    if (fixed_vals != NULL && n_fv != n_basis) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0f: declared values need one entry per "
                "basis column (%d, got %d)\n", n_basis, n_fv);
        return NULL;
    }
    if (n_fix == 0 && n_nfp != 0) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0f: no fixed waves but %d fixed CSR "
                "entries\n", n_nfp);
        return NULL;
    }
    if (n_uniq <= 0 || n_uslot <= 0 || n_wave <= 0 || n_basis <= 0) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0f: empty shape (n_uniq %d, n_uslot %d, "
                "n_wave %d, n_basis %d)\n", n_uniq, n_uslot, n_wave, n_basis);
        return NULL;
    }
    GPS0FCtx* c = (GPS0FCtx*)calloc(1, sizeof(GPS0FCtx));
    if (!c) return NULL;

    c->n_wave = n_wave;
    c->n_proj = (n_proj > 0) ? n_proj : 1;
    c->n_res = n_res;
    c->n_decay = n_decay;
    c->n_basis = n_basis;
    c->n_ab = n_ab;
    c->n_comp = n_comp;
    c->n_mass = n_mass;
    c->n_momentum = n_momentum;
    c->n_angle_total = n_angle_total;
    c->n_uniq = n_uniq;
    c->n_uslot = n_uslot;
    c->n_nz = n_nk;
    c->n_fix = n_fix;
    c->n_free = n_free;
    c->n_nz_fix = n_nfp;
    c->split_fixed = (split_fixed != 0) ? 1 : 0;
    c->launches[0] = 0;
    c->launches[1] = 0;
    c->launches[2] = 0;
    c->bwd_mode = (bwd_mode != 0) ? 1 : 0;
    c->fused_enable = (fused_enable != 0) ? 1 : 0;
    c->rsize = (rsize > 0) ? rsize : 1;
    c->batch_size = batch_size;

    /* ── the chunk rule (see `gps0f_chunk`) ─────────────────────────────── */
    c->cap = gps0f_chunk(batch_size, c->rsize);
    c->s_ngroups = (c->cap + c->rsize - 1) / c->rsize;

    /* ── the tile size ────────────────────────────────────────────────── */
    const int P = c->n_proj;
    c->te = (te_hint > 0) ? te_hint
                          : gps0f_pick_te(c->rsize, P, n_free, n_wave,
                                          n_basis);
    if (te_hint <= 0 && n_fix > 0) {
        c->te = gps0f_pick_te_split(c->rsize, P, n_free, n_wave, n_basis,
                                    c->te);
    }
    /* a hint must still hold whole groups (never straddle a group boundary),
       and it must fit the device: the hint is a test/benchmark hook, and a
       silently wrong tile is exactly the failure this guards against */
    if (c->rsize > 1 && c->te >= c->rsize && (c->te % c->rsize) != 0) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0f: te %d is not a multiple of rsize %d — "
                "a tile would split a log-sum group\n", c->te, c->rsize);
        free(c);
        return NULL;
    }
    if (c->te < 1) c->te = 1;
    {
        GPS0FLayout lay = gps0f_layout(c->te, P, n_free, n_wave, n_basis);
        c->smem_bytes = (int)(lay.total * (long long)sizeof(double));
        int optin = 0;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               0);
        if (c->smem_bytes > optin) {
            fprintf(stderr,
                    "cuda_flat_pwa_gen_s0f: shared tile %d B exceeds the "
                    "opt-in limit %d B (n_wave %d, n_proj %d, n_basis %d, "
                    "te %d)\n",
                    c->smem_bytes, optin, n_wave, c->n_proj, n_basis, c->te);
            free(c);
            return NULL;
        }
        /* Past the 48 KiB DEFAULT the launch needs the opt-in size on the
           FUNCTION, not merely a size check against the device: without it
           the launch fails with "invalid argument" and — because CUDA_CHECK
           only PRINTS — compute() would silently return Q = 0.

           The attribute is per FUNCTION, so two live contexts of different
           tile sizes would fight; a smaller one created second would LOWER it
           and break the larger one's launch.  Raise-only monotonic makes it
           the process-wide maximum, which every context is <= by
           construction, so any create order works. */
        static size_t opt_in_max = 0;
        if ((size_t)c->smem_bytes > opt_in_max) {
            cudaError_t e = cudaFuncSetAttribute(
                (const void*)gps0f_tile_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, c->smem_bytes);
            if (e != cudaSuccess) {
                fprintf(stderr,
                        "cuda_flat_pwa_gen_s0f: cannot opt in to %d B shared "
                        "memory (%s); n_wave %d, te %d\n",
                        c->smem_bytes, cudaGetErrorString(e), n_wave, c->te);
                free(c);
                return NULL;
            }
            opt_in_max = (size_t)c->smem_bytes;
        }
    }

    c->fl_table = _gps0f_up_d(fl_table, n_fl);
    c->fl_q_index = _gps0f_up_i(fl_q_index, n_fq);
    c->num_r = _gps0f_up_d(num_r, n_num);
    c->num_i = _gps0f_up_d(num_i, n_num);
    c->num_m_index = _gps0f_up_i(num_m_index, n_nm);
    c->bas_r = _gps0f_up_d(bas_r, n_bas);
    c->bas_i = _gps0f_up_d(bas_i, n_bas);
    c->bas_m_index = _gps0f_up_i(bas_m_index, n_bm);
    c->uslot_row = _gps0f_up_i(uslot_row, n_ur);
    c->uslot_col = _gps0f_up_i(uslot_col, n_uc);
    c->slot_of_wave = _gps0f_up_i(slot_of_wave, n_sw);
    c->rep_of_slot = _gps0f_up_i(rep_of_slot, n_rp);
    c->nz_k = _gps0f_up_i(nz_k, n_nk);
    c->nz_m = _gps0f_up_i(nz_m, n_nzm);
    c->nz_u = _gps0f_up_i(nz_u, n_nzu);
    c->nz_pair = _gps0f_up_l(nz_pair, n_np);
    c->gs_lo = _gps0f_up_i(gs_lo, n_lo);
    c->gs_hi = _gps0f_up_i(gs_hi, n_hi);
    c->fix_lo = _gps0f_up_i(fix_lo, n_flo);
    c->fix_hi = _gps0f_up_i(fix_hi, n_fhi);
    c->fix_slot_of_wave = _gps0f_up_i(fix_slot_of_wave, n_fsw);
    c->fix_pair = _gps0f_up_l(fix_pair, n_nfp);
    c->gamma = _gps0f_up_d(gamma, n_g);
    c->angle_index = _gps0f_up_i(angle_index, n_ai);
    c->angle_k = _gps0f_up_d(angle_k, n_ak);
    c->angle_b = _gps0f_up_d(angle_b, n_ab_);
    c->am_r = _gps0f_up_d(am_r, n_am);
    c->am_i = _gps0f_up_d(am_i, n_am);
    /* variant B: the declared VALUES live in the CONTEXT — it is
       per mask, so they can never leak into another mask's cache */
    c->fixed_vals = _gps0f_up_d(fixed_vals, n_fv);
    c->m_min = m_min; c->m_delta = m_delta; c->m_bins = m_bins;
    c->q_min = q_min; c->q_delta = q_delta; c->q_bins = q_bins;

    /* the split's scratch: ONE allocation per CONTEXT, `cap`-sized.  The
       handle duplicates nothing (before this, two handles each carried a
       `ne`-sized copy).  Mask-gated: none for the all-fixed fused path, none
       when nothing is fixed or the split is off. */
    if (c->split_fixed && c->n_fix > 0
            && !(c->fused_enable && c->n_free == 0 && c->rsize <= c->te)) {
        const size_t ncap = (size_t)((c->cap > 0) ? c->cap : 1);
        if (cudaMalloc(&c->a1, ncap * (size_t)c->n_proj * sizeof(double2))
                != cudaSuccess
                || cudaMalloc(&c->da,
                              ncap * (size_t)c->n_proj * sizeof(float2))
                != cudaSuccess
                || cudaMalloc(&c->dscale, ncap * sizeof(double))
                != cudaSuccess) {
            fprintf(stderr,
                    "cuda_flat_pwa_gen_s0f: split scratch allocation failed "
                    "(cap %d, n_proj %d)\n", c->cap, c->n_proj);
            cuda_free_context_flat_pwa_gen_s0f(c);
            return NULL;
        }
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

void cuda_free_context_flat_pwa_gen_s0f(void* vctx) {
    GPS0FCtx* c = (GPS0FCtx*)vctx;
    if (!c) return;
    #define F(p) if (c->p) cudaFree((void*)c->p)
    #define FH(p) if (c->p) cudaFreeHost(c->p)
    F(fl_table); F(fl_q_index); F(num_r); F(num_i); F(num_m_index);
    F(bas_r); F(bas_i); F(bas_m_index);
    F(uslot_row); F(uslot_col); F(slot_of_wave); F(rep_of_slot);
    F(nz_k); F(nz_m); F(nz_u); F(nz_pair); F(gs_lo); F(gs_hi); F(gamma);
    F(fix_lo); F(fix_hi); F(fix_slot_of_wave); F(fix_pair);
    F(angle_index); F(angle_k); F(angle_b); F(am_r); F(am_i);
    F(a1); F(da); F(dscale);
    F(fixed_vals);
    F(p_ck_r); F(p_ck_i); F(p_coeff);
    F(g_ck_r); F(g_ck_i); F(g_coeff);
    F(acc);
    FH(h_ck_r); FH(h_ck_i); FH(h_coeff); FH(h_acc);
    #undef F
    #undef FH
    if (c->stream) cudaStreamDestroy(c->stream);
    free(c);
}

void* cuda_load_data_flat_pwa_gen_s0f(
    void* vctx, const double* mass, const double* mom, const double* ang,
    const double* weight, const double* bkg, int ne)
{
    GPS0FCtx* c = (GPS0FCtx*)vctx;
    if (!c) return NULL;
    GPS0FHandle* h = (GPS0FHandle*)calloc(1, sizeof(GPS0FHandle));
    if (!h) return NULL;
    h->ne = ne;
    /* An EMPTY sample is a legal load (a fitter that has not been given data
       yet, or a zero-row selection): every buffer is simply absent, and
       `compute` must then return the identity (Q = 0, zero gradients) rather
       than failing.  A zero-byte cudaMalloc is an error, so the whole body is
       guarded rather than leaving a NULL-pointer trap for `compute`.

       `afix_ready = 1` on the way out: there is nothing to fill, and
       `compute` refuses an unfilled masked handle — an empty sample must take
       the NORMAL identity path, not that stop. */
    if (ne <= 0) { h->afix_ready = 1; return h; }
    h->weight = _gps0f_up_d(weight, ne);
    h->bkg = _gps0f_up_d(bkg, ne);
    if (cudaMalloc(&h->us, (size_t)ne * c->n_uslot * sizeof(float2))
            != cudaSuccess) goto fail;
    if (cudaMalloc(&h->amp, (size_t)ne * c->n_uniq * c->n_proj
                           * sizeof(float2)) != cudaSuccess) goto fail;
    /* the fixed-wave cache: 8·P·n_fix bytes per event, this handle's own */
    if (c->n_fix > 0 && cudaMalloc(&h->afix,
            (size_t)ne * c->n_fix * c->n_proj * sizeof(float2))
            != cudaSuccess) goto fail;
    /* the split's scratch: 2 buffers of float2[ne, P] + a per-event scale.
       With the split off (or nothing fixed) none of them exists at all. */
    /* The split's buffers are needed for the mixed path AND for the all-fixed
       FALLBACK (a group that cannot fit one block: rsize > te).  Only the
       fused-eligible all-fixed case allocates none. */
    if (cudaMalloc(&h->P_dev, (size_t)ne * sizeof(double)) != cudaSuccess)
        goto fail;
    CUDA_CHECK(cudaMallocHost(&h->h_P, (size_t)ne * sizeof(double)));

    /* ── the two load-time caches ─────────────────────────────────────────
     *
     * The kinematics are consumed ONLY here, so they are uploaded as transient
     * device copies and released immediately afterwards — exactly what
     * gen2/1bw do.  They are NOT read through the host pointers: a kernel
     * dereferencing a pageable host pointer happens to work on a device with
     * host page-table access (verified on this Blackwell part) but is not
     * portable, and on a device without it the fill would silently read
     * garbage.  Every other flat kernel copies first; so does this one. */
    {
        double* t_mass = _gps0f_up_d(mass, ne * c->n_mass);
        double* t_mom = _gps0f_up_d(mom, ne * c->n_momentum);
        double* t_ang = _gps0f_up_d(ang, ne * c->n_angle_total * c->n_comp);
        if (!t_mass || !t_mom || !t_ang) {
            if (t_mass) cudaFree(t_mass);
            if (t_mom) cudaFree(t_mom);
            if (t_ang) cudaFree(t_ang);
            goto fail;
        }
        {
            const long long total = (long long)ne * c->n_uslot;
            const int threads = GPS0F_BLOCK;
            long long blocks = (total + threads - 1) / threads;
            if (blocks > 65535) blocks = 65535;
            if (blocks < 1) blocks = 1;
            gps0f_fill_us<<<(unsigned)blocks, threads, 0, c->stream>>>(
                c->bas_r, c->bas_i, c->uslot_row, c->uslot_col,
                c->m_min, c->m_delta, c->m_bins,
                t_mass, c->n_mass, ne, c->n_uslot, h->us);
            CUDA_CHECK(cudaGetLastError());
        }
        {
            int fb = c->n_uniq < 32 ? 32 : (c->n_uniq < GPS0F_BLOCK
                                            ? GPS0F_BLOCK : c->n_uniq);
            if (fb > 1024) fb = 1024;
            const size_t shm = (size_t)c->n_ab * sizeof(double);
            gps0f_fill_amp<<<ne, fb, shm, c->stream>>>(
                c->fl_table, c->fl_q_index, c->num_r, c->num_i,
                c->num_m_index,
                c->m_min, c->m_delta, c->m_bins,
                c->q_min, c->q_delta, c->q_bins,
                c->angle_index, c->angle_k, c->angle_b, c->am_r, c->am_i,
                t_mass, t_mom, t_ang, c->rep_of_slot,
                ne, c->n_uniq, c->n_wave, c->n_proj, c->n_res, c->n_decay,
                c->n_ab, c->n_comp, c->n_angle_total, c->n_mass,
                c->n_momentum, h->amp);
            CUDA_CHECK(cudaGetLastError());
        }
        /* THE fill: every masked context carries its values, so the cache is
         * filled HERE — a handle is fully prepared by `load_data`
         * (`afix_ready` set), the first compute is guarded from the start and
         * no compute ever fills.  The launch is enqueued after `us`/`amp` on
         * the same stream, so it reads them complete. */
        if (c->n_fix > 0 && c->fixed_vals != NULL && h->afix != NULL) {
            gps0f_fill_afix_launch(c, h);
            h->afix_ready = 1;
        }
        CUDA_CHECK(cudaStreamSynchronize(c->stream));
        cudaFree(t_mass); cudaFree(t_mom); cudaFree(t_ang);
    }
    return h;

fail:
    if (h->weight) cudaFree(h->weight);
    if (h->bkg) cudaFree(h->bkg);
    if (h->us) cudaFree(h->us);
    if (h->amp) cudaFree(h->amp);
    if (h->afix) cudaFree(h->afix);
    if (h->P_dev) cudaFree(h->P_dev);
    if (h->h_P) cudaFreeHost(h->h_P);
    free(h);
    return NULL;
}

void cuda_free_data_flat_pwa_gen_s0f(void* vh) {
    GPS0FHandle* h = (GPS0FHandle*)vh;
    if (!h) return;
    if (h->weight) { cudaFree(h->weight); h->weight = NULL; }
    if (h->bkg) { cudaFree(h->bkg); h->bkg = NULL; }
    if (h->us) { cudaFree(h->us); h->us = NULL; }
    if (h->amp) { cudaFree(h->amp); h->amp = NULL; }
    if (h->afix) { cudaFree(h->afix); h->afix = NULL; }
    if (h->P_dev) { cudaFree(h->P_dev); h->P_dev = NULL; }
    if (h->h_P) { cudaFreeHost(h->h_P); h->h_P = NULL; }
    free(h);
}

/* One launch per chunk.  Batching needs no kernel logic: the chunk is
   expressed as POINTER OFFSETS into the per-event data plus the chunk length
   as the kernel's `ne`, so `ce0`/`ge` are chunk-local by construction. */
static void gps0f_launch(GPS0FCtx* c, GPS0FHandle* h, int st, int nb,
                        double norm, int use_norm)
{
    const int grouped = (use_norm && c->rsize > 1);
    /* ALL-FIXED FUSED: every wave fixed, the split on, and a group fits one
     * block's shared A/dA (`rsize <= te`; for rsize == 1 any te works).  The
     * block then owns whole groups (`per_blk == rsize`), so a group can never
     * straddle two blocks; `rsize > te` falls back to the general F/T/B path
     * for this call (a group would not fit one block's tile). */
    const int fused = (c->fused_enable && c->split_fixed && c->n_fix > 0
                       && c->n_free == 0 && nb > 0 && c->rsize <= c->te);
    int per_blk;
    if (fused) {
        /* rsize > 1: as many WHOLE groups as fit the tile (rsize <= te is
         * already required), so a block is not stuck at rsize events */
        per_blk = (c->rsize > 1)
                  ? c->rsize * (c->te / c->rsize) : c->te;
    } else if (!grouped) {
        per_blk = c->te;
    } else if (c->rsize <= c->te) {
        per_blk = c->rsize * (c->te / c->rsize);
    } else {
        per_blk = c->rsize;
    }
    if (per_blk < 1) per_blk = 1;
    if (fused) c->launches[2]++;
    const int tiles = (nb + per_blk - 1) / per_blk;
    const int pathB = grouped && (per_blk > c->te);
    const int split = (c->split_fixed && c->n_fix > 0 && c->a1 != NULL);
    gps0f_tile_kernel<<<tiles, GPS0F_BLOCK, (size_t)c->smem_bytes, c->stream>>>(
        h->us + (size_t)st * c->n_uslot,
        h->amp + (size_t)st * c->n_uniq * c->n_proj,
        c->slot_of_wave,
        (c->n_fix > 0
             ? h->afix + (size_t)st * c->n_fix * c->n_proj : NULL),
        c->n_fix,
        /* the split's intermediates are PER-CHUNK buffers (cap-sized), and
           the tile indexes them chunk-locally — no `st` offset */
        split ? c->a1 : NULL,
        split ? c->da : NULL,
        (split && pathB) ? c->dscale : NULL,
        c->nz_m, c->nz_u, c->nz_pair, c->nz_k,
        c->gs_lo, c->gs_hi,
        c->gamma, c->n_nz, c->p_coeff, c->p_ck_r, c->p_ck_i,
        h->weight + st, h->bkg + st,
        norm, use_norm, c->rsize, per_blk, c->te,
        c->n_free, c->n_proj, c->n_basis, c->n_uslot, c->n_uniq, c->n_wave,
        nb, fused,
        h->P_dev + st,
        c->g_ck_r, c->g_ck_i, c->g_coeff, c->acc, c->acc + 1);
}

/* Kernel F for ONE chunk `[st, st+nb)`: the pointers are offset and the
 * kernel sees the chunk length as its `ne`, exactly like the tile.  This is
 * what keeps `Afix[c]` hot in L2 between F, the tile and B (one chunk of
 * `Afix` at 50 000 events is 25.9 MiB against a 36 MiB L2; the old
 * once-per-sample F/B streamed the whole 6 M-event cache twice). */
static void gps0f_fwd_fixed_launch(GPS0FCtx* c, GPS0FHandle* h, int st,
                                   int nb)
{
    const long long total = (long long)nb * (long long)c->n_proj;
    if (total <= 0) return;
    long long blocks = (total + GPS0F_BLOCK - 1) / GPS0F_BLOCK;
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;
    gps0f_fwd_fixed<<<(unsigned)blocks, GPS0F_BLOCK, 0, c->stream>>>(
        h->afix + (size_t)st * c->n_fix * c->n_proj,
        c->p_ck_r, c->p_ck_i, c->n_fix, nb, c->n_proj,
        c->a1);
    CUDA_CHECK(cudaGetLastError());
    c->launches[0]++;
}

/* Kernel B for ONE chunk (after that chunk's tile, same `Afix[c]` lines). */
static void gps0f_bwd_fixed_launch(GPS0FCtx* c, GPS0FHandle* h, int st,
                                   int nb, int use_scale)
{
    const long long total = (long long)nb * (long long)c->n_proj;
    if (total <= 0) return;
    long long blocks;
    size_t shm;
    if (c->bwd_mode == 1) {
        /* the wave-major mapping, kept for the A/B gate */
        const int nwarps = GPS0F_BLOCK / 32;
        blocks = (total + nwarps - 1) / nwarps;
        shm = GPS0F_BLOCK * (sizeof(float2) + sizeof(long long));
    } else {
        blocks = (total + GPS0F_BLOCK - 1) / GPS0F_BLOCK;
        shm = GPS0F_BLOCK * (sizeof(float2) + sizeof(long long))
              + (size_t)(16 * c->n_fix) * sizeof(double);
    }
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;
    if (c->bwd_mode == 1) {
        gps0f_bwd_fixed_wave<<<(unsigned)blocks, GPS0F_BLOCK, shm, c->stream>>>(
            h->afix + (size_t)st * c->n_fix * c->n_proj,
            c->da, use_scale ? c->dscale : NULL,
            c->n_fix, nb, c->n_proj, c->g_ck_r, c->g_ck_i);
    } else {
        gps0f_bwd_fixed<<<(unsigned)blocks, GPS0F_BLOCK, shm, c->stream>>>(
            h->afix + (size_t)st * c->n_fix * c->n_proj,
            c->da, use_scale ? c->dscale : NULL,
            c->n_fix, nb, c->n_proj, c->g_ck_r, c->g_ck_i);
    }
    CUDA_CHECK(cudaGetLastError());
    c->launches[1]++;
}

/* The fixed-wave cache, filled ONCE per handle — at `load_data`, from the
 * VALUES the mask was declared with (`c->fixed_vals`, per mask, in the
 * context).  It walks the ORIGINAL fixed-block CSR and stores phase A's
 * `common`, so the tile can later read it instead of recomputing `D`.  The
 * kernel is not forked: the caller passes the pointer. */
static void gps0f_fill_afix_launch(GPS0FCtx* c, GPS0FHandle* h)
{
    if (c->n_fix <= 0 || !h->afix || h->ne <= 0) return;
    if (c->fixed_vals == NULL) {
        /* create's invariant (n_fix > 0 => values) — the two guards agree */
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0f: no declared values (n_fix %d) — "
                "the cache cannot be filled\n", c->n_fix);
        return;
    }
    const long long total = (long long)h->ne * (long long)c->n_fix;
    long long blocks = (total + GPS0F_BLOCK - 1) / GPS0F_BLOCK;
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;
    gps0f_fill_afix<<<(unsigned)blocks, GPS0F_BLOCK, 0, c->stream>>>(
        h->us, h->amp, c->fix_slot_of_wave, c->fix_pair,
        c->fix_lo, c->fix_hi,
        /* create REFUSES a masked context without values, so this is the
           only source: a NULL here would read never-uploaded p_coeff */
        c->fixed_vals,
        c->n_fix, h->ne, c->n_uslot, c->n_uniq, c->n_proj, h->afix);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_compute_flat_pwa_gen_s0f(
    void* vctx, void* vh, const double* ck_r, const double* ck_i,
    const double* coeff, double norm, int use_norm,
    double* oQ, double* oDn, double* oP,
    double* og_ck_r, double* og_ck_i, double* og_coeff)
{
    GPS0FCtx* c = (GPS0FCtx*)vctx;
    GPS0FHandle* h = (GPS0FHandle*)vh;
    if (!c || !h) return;
    const int N = c->n_wave, NB = c->n_basis;

    memcpy(c->h_ck_r, ck_r, (size_t)N * sizeof(double));
    memcpy(c->h_ck_i, ck_i, (size_t)N * sizeof(double));
    memcpy(c->h_coeff, coeff, (size_t)NB * sizeof(double));
    cudaMemcpyAsync(c->p_ck_r, c->h_ck_r, (size_t)N * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    cudaMemcpyAsync(c->p_ck_i, c->h_ck_i, (size_t)N * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    cudaMemcpyAsync(c->p_coeff, c->h_coeff, (size_t)NB * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    /* the fixed-wave cache is filled by `load_data` (the mask is declared
       WITH its values, so the coefficients exist there).  `compute` NEVER
       fills: an unfilled masked handle is unreachable by construction, and
       silently computing from an uninitialised cache would be worse than
       stopping, so it is a hard (printed) failure. */
    if (c->n_fix > 0 && !h->afix_ready) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0f: handle has no Afix cache (n_fix %d) — load_data must "
                "fill it; refusing to compute\n", c->n_fix);
        return;
    }
    /* the accumulators span the WHOLE sample: zero once, read once */
    cudaMemsetAsync(c->acc, 0, 2 * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_ck_r, 0, (size_t)N * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_ck_i, 0, (size_t)N * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_coeff, 0, (size_t)NB * sizeof(double), c->stream);

    const int ne_total = h->ne;
    /* The split is INTERLEAVED per chunk: F(c) -> tile(c) -> B(c), so the
     * chunk's `Afix` slice is read by F, the tile and B while it is still in
     * L2 (one `cap` chunk of Afix fits; the old once-per-sample F/B streamed
     * the whole cache twice).  Nothing is launched when the split is off or
     * nothing is fixed, and the all-fixed fused branch never reaches here.
     * `cap` is a multiple of `rsize`, so a chunk never splits a group. */
    const int grouped = (use_norm && c->rsize > 1);
    const int pathB = grouped && (c->rsize > c->te);
    const int split = (c->split_fixed && c->n_fix > 0
                       && c->a1 != NULL && ne_total > 0);

    const int bs = (c->cap > 0) ? c->cap : 1;
    for (int st = 0; st < ne_total; st += bs) {
        const int nb = (ne_total - st > bs) ? bs : (ne_total - st);
        if (split) gps0f_fwd_fixed_launch(c, h, st, nb);
        gps0f_launch(c, h, st, nb, norm, use_norm);
        CUDA_CHECK(cudaGetLastError());
        if (split) gps0f_bwd_fixed_launch(c, h, st, nb, pathB);
    }
    /* the caller-visible P: ONE D2H for the whole sample (the buffer is the
       handle's per-event OUTPUT, allocated once per load, never per call) */
    if (oP != NULL && ne_total > 0) {
        cudaMemcpyAsync(h->h_P, h->P_dev, (size_t)ne_total * sizeof(double),
                        cudaMemcpyDeviceToHost, c->stream);
    }
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
    if (oP != NULL && ne_total > 0)
        memcpy(oP, h->h_P, (size_t)ne_total * sizeof(double));
}

/* ── reports: every one of these has exactly ONE definition, in C ─────── */

/* The tile size `gps0f_layout` gives for an arbitrary shape — the same
 * function the kernel and `create` use, so a test can assert the buffer
 * arithmetic without a second, drifting copy in Python. */
long long cuda_tile_bytes_flat_pwa_gen_s0f(int te, int n_wave, int n_proj,
                                           int n_basis, int n_free)
{
    GPS0FLayout L = gps0f_layout(te, n_proj, n_free, n_wave, n_basis);
    return (long long)L.total * (long long)sizeof(double);
}

/* The fixed-wave cache's size for a sample of `ne` events, in B. */
long long cuda_fix_bytes_flat_pwa_gen_s0f(void* vctx, int ne) {
    GPS0FCtx* c = (GPS0FCtx*)vctx;
    if (!c || ne <= 0) return 0;
    return (long long)ne * c->n_fix * c->n_proj
           * (long long)sizeof(float2);
}

/* The split's scratch size for THIS CONTEXT, in B — ONE allocation, shared
 * by every handle (it is `cap`-sized, so it does NOT depend on `ne`, hence
 * the argument is ignored).  Zero on the fused and mask-less paths. */
long long cuda_aux_bytes_flat_pwa_gen_s0f(void* vctx, int ne) {
    (void)ne;
    GPS0FCtx* c = (GPS0FCtx*)vctx;
    if (!c || !c->split_fixed || c->n_fix <= 0) return 0;
    if (c->n_free == 0 && c->fused_enable
            && c->rsize <= c->te) return 0;
    const long long ncap = (c->cap > 0) ? c->cap : 1;
    return ncap * c->n_proj * (long long)sizeof(double2)
           + ncap * c->n_proj * (long long)sizeof(float2)
           + ncap * (long long)sizeof(double);
}

/* The chunk plan for a sample of `ne` events: out[0] = cap, out[1] = the
 * number of chunks, out[2] = 1 when every chunk start and every non-final
 * chunk length is a multiple of `rsize` (a group never straddles a chunk). */
void cuda_chunk_plan_flat_pwa_gen_s0f(void* vctx, int ne, int* out) {
    GPS0FCtx* c = (GPS0FCtx*)vctx;
    if (!out) return;
    out[0] = c ? c->cap : 0;
    out[1] = 0;
    out[2] = 1;                 /* no chunks: vacuously aligned */
    if (!c || ne <= 0) return;
    const int bs = (c->cap > 0) ? c->cap : 1;
    int n = 0, aligned = 1;
    for (int st = 0; st < ne; st += bs) {
        const int nb = (ne - st > bs) ? bs : (ne - st);
        const int last = (st + nb == ne);
        if (c->rsize > 1) {
            if ((st % c->rsize) != 0 || (!last && (nb % c->rsize) != 0))
                aligned = 0;
        }
        n++;
    }
    out[1] = n;
    out[2] = aligned;
}

/* Kernel F / B launch counts since `create` — the mask-off gate asserts both
 * are 0 (they are not launched, and no buffer exists). */
int cuda_fused_count_flat_pwa_gen_s0f(void* vctx) {
    GPS0FCtx* c = (GPS0FCtx*)vctx;
    return c ? c->launches[2] : 0;
}

void cuda_split_counts_flat_pwa_gen_s0f(void* vctx, int* out) {
    GPS0FCtx* c = (GPS0FCtx*)vctx;
    if (!out) return;
    out[0] = c ? c->launches[0] : 0;
    out[1] = c ? c->launches[1] : 0;
}

/* 1 when this handle's `Afix` cache has been filled (with declared values
 * that is already true right after `load_data`). */
int cuda_handle_afix_ready_flat_pwa_gen_s0f(void* vh) {
    GPS0FHandle* h = (GPS0FHandle*)vh;
    return (h != NULL && h->afix_ready) ? 1 : 0;
}

/* 1 when this context carries DECLARED VALUES (the eager-fill path). */
int cuda_has_fixed_values_flat_pwa_gen_s0f(void* vctx) {
    GPS0FCtx* c = (GPS0FCtx*)vctx;
    return (c != NULL && c->fixed_vals != NULL) ? 1 : 0;
}

int cuda_n_fix_flat_pwa_gen_s0f(void* vctx) {
    GPS0FCtx* c = (GPS0FCtx*)vctx;
    return c ? c->n_fix : 0;
}

int cuda_n_free_flat_pwa_gen_s0f(void* vctx) {
    GPS0FCtx* c = (GPS0FCtx*)vctx;
    return c ? c->n_free : 0;
}

int cuda_n_nz_free_flat_pwa_gen_s0f(void* vctx) {
    GPS0FCtx* c = (GPS0FCtx*)vctx;
    return c ? c->n_nz : 0;
}

int cuda_n_nz_fix_flat_pwa_gen_s0f(void* vctx) {
    GPS0FCtx* c = (GPS0FCtx*)vctx;
    return c ? c->n_nz_fix : 0;
}

int cuda_smem_bytes_flat_pwa_gen_s0f(void* vctx) {
    GPS0FCtx* c = (GPS0FCtx*)vctx;
    return c ? c->smem_bytes : 0;
}

/* Runtime resource report (host side, no kernel launch).
 *
 * CUDA exposes the compiled resource usage per FUNCTION, which is what the
 * `__launch_bounds__` / `-D..._MIN_BLOCKS` choice is trading against:
 *   out[0] registers/thread          (cudaFuncGetAttributes.numRegs)
 *   out[1] local bytes > 0  => the kernel SPILLS
 *   out[2] blocks/SM at the REAL block size and dynamic smem
 *   out[3] warps/SM  (= blocks x GPS0F_BLOCK/32)
 *   out[4] maxThreadsPerBlock
 *   out[5] static shared bytes
 * Everything here is a property of the compiled cubin; `smem` is the tile's
 * dynamic shared memory for the CURRENT shape (c->smem_bytes), so the
 * occupancy figure is the one the fit actually gets.
 */
extern "C" void cuda_stats_flat_pwa_gen_s0f(int smem, int* out)
{
    if (!out) return;
    cudaFuncAttributes a;
    cudaFuncGetAttributes(&a, (const void*)gps0f_tile_kernel);
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, (const void*)gps0f_tile_kernel, GPS0F_BLOCK, (size_t)(smem > 0 ? smem : 0));
    out[0] = a.numRegs;
    out[1] = (int)a.localSizeBytes;
    out[2] = blocks;
    out[3] = blocks * (GPS0F_BLOCK / 32);
    out[4] = a.maxThreadsPerBlock;
    out[5] = (int)a.sharedSizeBytes;
}

int cuda_cap_flat_pwa_gen_s0f(void* vctx) {
    GPS0FCtx* c = (GPS0FCtx*)vctx;
    return c ? c->cap : 0;
}

int cuda_ngroups_flat_pwa_gen_s0f(void* vctx) {
    GPS0FCtx* c = (GPS0FCtx*)vctx;
    return c ? c->s_ngroups : 0;
}

int cuda_te_flat_pwa_gen_s0f(void* vctx) {
    GPS0FCtx* c = (GPS0FCtx*)vctx;
    return c ? c->te : 0;
}

int cuda_get_device_count_flat_pwa_gen_s0f(void) {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
}

void cuda_get_device_name_flat_pwa_gen_s0f(char* buf, int len) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    snprintf(buf, len, "%s", p.name);
}

}  /* extern "C" */
