/**
 * cuda_flat_pwa_gen_s0g — the s0f fixed-wave cache with the FREE block
 * UN-STAGED, all in one kernel pass.
 *
 * Case-1 copy (kernel policy §1): own `.cu`, own `.so`, own wrapper, own
 * symbols.  `kernels_flat_pwa_gen_s0f.cu` and every other kernel/wrapper in
 * the tree stay byte-identical.
 *
 * ── why ─────────────────────────────────────────────────────────────────
 *
 * s0f's mixed path splits the wave axis: the fixed block lives outside the
 * tile (F: `A1 = Σ ck·Afix`, B: `dck = Σ dA·Afix`) and the tile stages the
 * FREE block's `common = Amp·conj(D)/|D|²` in shared, which pins its tile to
 * one whole group (`te == rsize`) and costs SIX barriers per chunk.  Profiled
 * on the CTPWA sample (27 fixed / 7 free, rsize 17, 6 M events): the tile was
 * 67 % of the call, F 14 %, B 19 %, and the fixed-wave ladder was FLAT
 * (33/32/27 fixed → 90.5/89.8/93.1 ms) — i.e. the mixed time was the tile's
 * pipeline, not the hidden block's arithmetic.
 *
 * s0g removes the staging.  A thread owns one element `(e, p)` and walks the
 * free CSR itself:
 *
 *     D_k   = Σ_{m ∈ nz(k)} coeff_m · Z_m(e)          (CSR walk, fp64)
 *     A_{e,p} = Σ_{k<nfix} ck_k·Afix[e,k,p]           (contiguous k, fp64)
 *             + Σ_{k<N}   ck_k·Amp[e,slot(k),p]·conj(D_k)/|D_k|²
 *     P_e   = Σ_p |A_{e,p}|²  ;  the objective's per-event terms
 *     dA_{e,p} = dQ/dP_e·conj(A_{e,p})
 *     dck_k += Σ_{e,p} dA·conj(Afix[e,k,p])           k < nfix
 *     u_k   = Σ_p dA_{e,p}·common_{k,p} ;  pv_k = -ck_k·u_k/D_k
 *     dck_k += u_k  ;  dcoeff_m += Σ_e 2·Re(pv_k·conj(Z_m))·gamma[k,m]
 *
 * There is no F, no B, no `A1`, no `dA` buffer and no `dscale`: the fixed
 * block is summed in the same per-element loop, so its `Afix` reads are the
 * very lines phase 1 already streamed, and its `dck` is a (wave, slice)
 * reduction over the shared `dA`.
 *
 * Accepted cost (measured, not hidden): `D_k` is recomputed — once per
 * projection in the forward (`P` CSR passes) and once per `(e, k)` in the
 * backward, instead of one staged copy.  With 7 free waves / 15 non-zeros the
 * `us` working set of a block is ~36 KB, i.e. L1-resident, so the second walk
 * is not an HBM re-stream.
 *
 * ── barriers ────────────────────────────────────────────────────────────
 *
 * `rsize == 1`: 3 per BLOCK (after A/|A|², after dA, after the backward —
 * the last one publishes the shared `dck`/`pv` partials) — the tile's group
 * close is per event and every thread derives its own scale from the shared
 * `|A_{e,p}|²`, so q and dA need no barrier between them.
 * `rsize > 1`: 5 (the same three plus the group close: q -> sums -> scale).  A block owns
 * WHOLE groups (`te` is a multiple of `rsize`), so a group never straddles a
 * block; `rsize > 1` with a group that does not fit one block is refused at
 * construction and names `cuda_flat_pwa_gen_s0f`.
 *
 * ── everything else is s0f ──────────────────────────────────────────────
 *
 * The fixed-wave cache (`Afix`, filled once per handle on the first compute),
 * the two byte-exact dedups (`us` on `(row bytes, mass column)`, `amp` on the
 * whole numerator of every projection), the basis rotation, the grouped
 * log-sum objective, the batch_size chunk rule and the fp64 accumulation
 * policy are the parent's, unchanged.
 *
 * `grads["coeff"] == 0` at a declared-fixed column stays the contract (a
 * fixed wave has no `D`, hence no `pv`).
 *
 * Precision: `A` is accumulated in fp64 (fp32 fails the h=1e-6 `ck` FD);
 * `Afix`/`Amp` stay float2; `dcoeff` accumulates in fp64, one global atomic
 * per CSR entry after a register reduction.
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstring>
#include <cmath>

/* A group is AVAILABLE when its total weight is non-zero — judged against
 * an ABSOLUTE epsilon, because a group whose weights cancel can sum to a
 * few 1e-17 that is not exactly 0, and `W != 0` would then take a log of a
 * meaningless ratio.  The SIGN of W_g is part of the formula
 * (`-W_g·log(S_g/W_g)`), so a negative total is computed, not dropped, and
 * the weights of a group need NOT share a sign (a signed sWeight bin is a
 * legitimate group: its mean pdf must then come out positive). */
/* The W each guard tests is the GROUP TOTAL (a double), not a tile's
 * `int W = n_proj * n_wave` that shares the scope in the tile kernels. */
#define GP_GROUP_EPS 1e-12

#define GPS0G_BLOCK       256      /* threads per block (multiple of 32) */
/* Launch bounds: the SECOND parameter is a REGISTER BUDGET choice, not a
 * correctness one.  Measured on sm_120: 5 blocks/SM => <=51 registers, which
 * is what the 2-chain dot needs (without it the compiler takes 56 and loses
 * 17 occupancy points).  The right budget is per ARCH, so it is a macro:
 *
 *   per build   TABPWA_NVCC_FLAGS_SM70="-DGPS0G_MIN_BLOCKS=6"
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
 * per-arch override, e.g. `TABPWA_NVCC_FLAGS_SM70=-DGPS0G_MIN_BLOCKS=3`.
 * Pick from a measurement, never from taste. */
#ifndef GPS0G_MIN_BLOCKS
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
#    define GPS0G_MIN_BLOCKS 5      /* Blackwell consumer: clean, matches s0e */
#  else
#    define GPS0G_MIN_BLOCKS 4      /* every other arch measured: 5 spills */
#  endif
#endif
#define GPS0G_TILE_BOUNDS __launch_bounds__(GPS0G_BLOCK, GPS0G_MIN_BLOCKS)
#define GPS0G_MAX_TE      256      /* events per block ceiling     */
#define GPS0G_MAX_CAP     (1 << 20)  /* ceiling on the chunk length  */
/*: soft shared-memory budget for choosing `te`.  Below this the launch needs
 *  no opt-in (so no `cudaFuncSetAttribute` and the best occupancy); above it
 *  `create` shrinks `te` first and only then crosses the opt-in. */
#define GPS0G_SMEM_BUDGET (48 * 1024)

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = call;                                             \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                               \
        }                                                                    \
    } while (0)

/* ── Catmull-Rom (the repo-wide idiom, identical to gen2/1bw) ─────────── */
__device__ double gps0g_cr1d(double pm1, double p0, double p1, double p2,
                            double t) {
    return p0 + 0.5 * t * (-pm1 + p1 + t * (2.0 * pm1 - 5.0 * p0 + 4.0 * p1
                                            - p2 + t * (-pm1 + 3.0 * p0
                                                        - 3.0 * p1 + p2)));
}

__device__ double gps0g_interp_real(const double* __restrict__ t, int row,
                                   double x, double xmin, double xdelta,
                                   int bins) {
    double diff = (x - xmin) / xdelta;
    int b = max(0, min((int)floor(diff), bins - 2));
    double s = max(0.0, min(diff - b, 1.0));
    int base = row * bins + b;
    int lo = row * bins, hi = (row + 1) * bins - 1;
    int im1 = base > lo ? base - 1 : base;
    int i2 = base + 2 <= hi ? base + 2 : base + 1;
    return gps0g_cr1d(t[im1], t[base], t[base + 1], t[i2], s);
}

__device__ void gps0g_interp_c(const double* __restrict__ tr,
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
    *orr = gps0g_cr1d(tr[im1], tr[base], tr[base + 1], tr[i2], s);
    *oii = gps0g_cr1d(ti[im1], ti[base], ti[base + 1], ti[i2], s);
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
__global__ void gps0g_fill_us(
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
        gps0g_interp_c(bas_r, bas_i, uslot_row[u], mv,
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
__global__ void gps0g_fill_amp(
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
            gps0g_interp_c(num_r, num_i, k * n_res + r, mv,
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
            fl *= gps0g_interp_real(fl_table, k * n_decay + d, qv,
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
__global__ void gps0g_fill_afix(
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
 * The shared tile — ONE formula, used by the host (`create`) and the kernel
 * (the pointer offsets), so the launch size and the kernel's view cannot
 * disagree.  All offsets are in DOUBLES.
 *
 *   A     [2·te·P]   A_{e,p} (fp64), REUSED IN PLACE as dQ/dA_{e,p}
 *   ape   [te·P]     |A_{e,p}|² — the cross-p sum that forms P_e
 *   pe, q, sc [te]   P_e, the group summand q_e, the group scale
 *   pv    [2·te·N]   dQ/dD_{e,k} (free waves only — a fixed wave has none)
 *   gck   [2·Nall]   per-wave shared accumulator for dQ/dck (fixed + free)
 *===========================================================================*/
typedef struct {
    int A, ape, pe, q, sc, pv, gck, total;
} GPS0GLayout;

__host__ __device__ __forceinline__ GPS0GLayout gps0g_layout(
    int te, int P, int Nfree, int Nall, int NB)
{
    (void)NB;
    GPS0GLayout L;
    int o = 0;
    L.A   = o; o += 2 * te * P;
    L.ape = o; o += te * P;
    L.pe  = o; o += te;
    L.q   = o; o += te;
    L.sc  = o; o += te;
    L.pv  = o; o += 2 * te * Nfree;
    L.gck = o; o += 2 * Nall;
    L.total = o;
    return L;
}

/*=============================================================================
 * The per-iteration tile: ONE pass, no staged `common`.
 *
 * Element (e, p) = one thread; the free CSR is walked per projection in the
 * forward and per (e,k) in the backward (the accepted cost in the header).
 * `A_{e,p}` is shared because `P_e` needs every projection and the backward
 * is wave-major; it is overwritten by `dA` in place.
 *===========================================================================*/
__global__ void GPS0G_TILE_BOUNDS gps0g_tile_kernel(
    const float2* __restrict__ us,         /* [ne, n_uslot]   Z = i·basis   */
    const float2* __restrict__ amp,        /* [ne, n_uniq, P] fa·fl·num     */
    const int* __restrict__ slot_of_wave,  /* [n_wave] -> amp slot (PERMUTED) */
    const float2* __restrict__ afix,       /* [ne, n_fix, P]  fixed common  */
    int n_fix,                             /* waves [0, n_fix) are fixed    */
    const int* __restrict__ nz_m,          /* [n_nz] basis column           */
    const int* __restrict__ nz_u,          /* [n_nz] -> uslot               */
    const long long* __restrict__ nz_pair, /* [n_nz] m | (u << 32)          */
    const int* __restrict__ nz_k,          /* [n_nz] free wave, 0..N-1      */
    const int* __restrict__ gs_lo,         /* [N] free CSR start            */
    const int* __restrict__ gs_hi,         /* [N] free CSR end              */
    const double* __restrict__ gamma,      /* [N·n_basis] free rows         */
    int n_nz,                              /* FREE non-zeros                */
    const double* __restrict__ coeff,      /* [n_basis]                     */
    const double* __restrict__ ck_r, const double* __restrict__ ck_i,
    const double* __restrict__ weight, const double* __restrict__ bkg,
    double norm, int use_norm,
    int rsize, int te,
    int N, int P, int NB, int n_uslot, int n_uniq, int Nall,
    int ne,
    double* __restrict__ P_dev,            /* [ne]  per-event P (output)    */
    double* __restrict__ g_ck_r, double* __restrict__ g_ck_i,
    double* __restrict__ g_coeff,
    double* __restrict__ Q_acc, double* __restrict__ dnorm_acc)
{
    const int tid = threadIdx.x, nt = blockDim.x;
    const GPS0GLayout L = gps0g_layout(te, P, N, Nall, NB);

    extern __shared__ double sh[];
    double* s_A   = sh + L.A;
    double* s_ape = sh + L.ape;
    double* s_pe  = sh + L.pe;
    double* s_q   = sh + L.q;
    double* s_sc  = sh + L.sc;
    double* s_pv  = sh + L.pv;
    double* s_gck = sh + L.gck;

    const int e0 = (int)blockIdx.x * te;
    const int evn = min(te, ne - e0);
    if (evn <= 0) return;
    const int rs = (rsize < 1) ? 1 : rsize;
    const int nper = evn * P;

    for (int k = tid; k < 2 * Nall; k += nt) s_gck[k] = 0.0;

    /* ── phase 1: A_{e,p} = fixed block + free block, both fp64 ─────────
     * The fixed block is a contiguous k-run over `Afix` (the same lines the
     * backward re-reads); the free block walks the free CSR with phase A's
     * packed 64-bit step and two fp64 chains, and forms `common` for THIS
     * projection.  Nothing is staged: `common` is rebuilt where it is needed
     * again. */
    for (int idx = tid; idx < nper; idx += nt) {
        const int e = idx / P, p = idx - e * P;
        const int ge = e0 + e;
        double ar = 0.0, ai = 0.0;
        if (n_fix > 0) {
            const float2* base = afix + ((size_t)ge * n_fix) * P + p;
            for (int k = 0; k < n_fix; k++) {
                const float2 a = base[(size_t)k * P];
                const double cr = ck_r[k], ci = ck_i[k];
                ar += cr * (double)a.x - ci * (double)a.y;
                ai += cr * (double)a.y + ci * (double)a.x;
            }
        }
        if (N > 0) {
            const float2* uu = us + (size_t)ge * n_uslot;
            const size_t abase = (size_t)ge * n_uniq * P;
            for (int k = 0; k < N; k++) {
                const int lo = gs_lo[k], hi = gs_hi[k];
                double sr0 = 0.0, si0 = 0.0, sr1 = 0.0, si1 = 0.0;
                int i = lo;
                const int two = lo + ((hi - lo) & ~1);
                for (; i < two; i += 2) {
                    const long long q0 = nz_pair[i], q1 = nz_pair[i + 1];
                    const double c0 = coeff[(int)(q0 & 0xFFFFFFFFLL)];
                    const double c1 = coeff[(int)(q1 & 0xFFFFFFFFLL)];
                    const float2 z0 = uu[(int)(q0 >> 32)];
                    const float2 z1 = uu[(int)(q1 >> 32)];
                    sr0 += c0 * (double)z0.y;
                    si0 -= c0 * (double)z0.x;
                    sr1 += c1 * (double)z1.y;
                    si1 -= c1 * (double)z1.x;
                }
                if (i < hi) {
                    const long long q0 = nz_pair[i];
                    const double c0 = coeff[(int)(q0 & 0xFFFFFFFFLL)];
                    const float2 z0 = uu[(int)(q0 >> 32)];
                    sr0 += c0 * (double)z0.y;
                    si0 -= c0 * (double)z0.x;
                }
                const double sr = sr0 + sr1, si = si0 + si1;
                const double rden = 1.0 / (sr * sr + si * si);
                const int kw = n_fix + k;
                const float2 a = amp[abase + (size_t)slot_of_wave[kw] * P + p];
                const double cr = ((double)a.x * sr + (double)a.y * si) * rden;
                const double ci = ((double)a.y * sr - (double)a.x * si) * rden;
                const double ckr = ck_r[kw], cki = ck_i[kw];
                ar += ckr * cr - cki * ci;
                ai += ckr * ci + cki * cr;
            }
        }
        s_A[idx * 2 + 0] = ar;
        s_A[idx * 2 + 1] = ai;
        s_ape[idx] = ar * ar + ai * ai;
    }
    __syncthreads();

    /* ── phase 2: the objective, then dQ/dA in place ───────────────────── */
    if (rs == 1) {
        /* Per-event group: EVERY thread derives its own scale from the shared
         * `|A_{e,p}|²` (P redundant reads), so q and dA need no barrier
         * between them — the host-level Q/dnorm terms are written once, on
         * p == 0. */
        for (int idx = tid; idx < nper; idx += nt) {
            const int e = idx / P, p = idx - e * P;
            const int ge = e0 + e;
            double Pe = 0.0;
            for (int j = 0; j < P; j++) Pe += s_ape[e * P + j];
            const double wv = weight[ge], bv = bkg[ge];
            const double q = use_norm ? wv * (Pe / norm + bv) : wv * Pe;
            double scale;
            if (use_norm) {
                if (fabs(wv) > GP_GROUP_EPS) {
                    scale = -wv / q;
                    if (p == 0) {
                        atomicAdd(Q_acc, -wv * log(q / wv));
                        atomicAdd(dnorm_acc, wv * (wv * Pe)
                                  / (q * norm * norm));
                    }
                } else {
                    scale = 0.0;
                }
            } else {
                scale = 1.0;
                if (p == 0) atomicAdd(Q_acc, q);
            }
            if (p == 0) {
                s_pe[e] = Pe;
                s_q[e] = q;
                P_dev[ge] = Pe;
            }
            const double dqp = (use_norm ? wv / norm : wv) * scale;
            const double ar = s_A[idx * 2 + 0], ai = s_A[idx * 2 + 1];
            s_A[idx * 2 + 0] = dqp * ar;
            s_A[idx * 2 + 1] = -dqp * ai;
        }
    } else {
        for (int idx = tid; idx < nper; idx += nt) {
            const int e = idx / P, p = idx - e * P;
            if (p != 0) continue;
            const int ge = e0 + e;
            double Pe = 0.0;
            for (int j = 0; j < P; j++) Pe += s_ape[e * P + j];
            const double wv = weight[ge], bv = bkg[ge];
            s_pe[e] = Pe;
            s_q[e] = use_norm ? wv * (Pe / norm + bv) : wv * Pe;
            P_dev[ge] = Pe;
        }
        __syncthreads();
        /* group close: a block owns whole groups, so the sums are complete */
        const int ng = (evn + rs - 1) / rs;
        for (int g = tid; g < ng; g += nt) {
            const int a = g * rs;
            const int b = (a + rs < evn) ? (a + rs) : evn;
            double S = 0.0, WP = 0.0, W = 0.0;
            for (int e = a; e < b; e++) {
                S += s_q[e];
                WP += weight[e0 + e] * s_pe[e];
                W += weight[e0 + e];
            }
            if (use_norm) {
                if (fabs(W) > GP_GROUP_EPS) {
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
        for (int idx = tid; idx < nper; idx += nt) {
            const int e = idx / P;
            const double wv = weight[e0 + e];
            const double dqp = (use_norm ? wv / norm : wv) * s_sc[e / rs];
            const double ar = s_A[idx * 2 + 0], ai = s_A[idx * 2 + 1];
            s_A[idx * 2 + 0] = dqp * ar;
            s_A[idx * 2 + 1] = -dqp * ai;
        }
    }
    __syncthreads();

    /* ── phase 3: the backward, both blocks ────────────────────────────── */
    if (n_fix > 0) {
        /* fixed block: thread owns (k, slice) and walks the block's elements,
         * so the shared atomics are on a (wave, slice) grid rather than one
         * per element.  `Afix` reads are the contiguous k-run phase 1 used. */
        const int nsl0 = (n_fix > 0) ? (nt / n_fix) : 1;
        const int nsl = (nsl0 > 8) ? 8 : (nsl0 < 1 ? 1 : nsl0);
        const int nact = n_fix * nsl;
        for (int t = tid; t < nact; t += nt) {
            const int k = t % n_fix;
            const int sl = t / n_fix;
            double ur = 0.0, ui = 0.0;
            for (int i = sl; i < nper; i += nsl) {
                const int e = i / P, p = i - e * P;
                const float2 a = afix[((size_t)(e0 + e) * n_fix + k) * P + p];
                const double dr = s_A[i * 2 + 0], di = s_A[i * 2 + 1];
                ur += dr * (double)a.x - di * (double)a.y;
                ui += dr * (double)a.y + di * (double)a.x;
            }
            if (ur != 0.0) atomicAdd(&s_gck[2 * k + 0], ur);
            if (ui != 0.0) atomicAdd(&s_gck[2 * k + 1], ui);
        }
    }
    if (N > 0) {
        /* free block: thread per (e,k).  `D_k` is rebuilt here (the accepted
         * second walk), the projections are folded into `u`, `dck` gets `u`
         * and `pv` is staged for the coefficient pass. */
        for (int idx = tid; idx < evn * N; idx += nt) {
            const int e = idx / N, k = idx - e * N;
            const int ge = e0 + e;
            const float2* uu = us + (size_t)ge * n_uslot;
            const int lo = gs_lo[k], hi = gs_hi[k];
            double sr0 = 0.0, si0 = 0.0, sr1 = 0.0, si1 = 0.0;
            int i = lo;
            const int two = lo + ((hi - lo) & ~1);
            for (; i < two; i += 2) {
                const long long q0 = nz_pair[i], q1 = nz_pair[i + 1];
                const double c0 = coeff[(int)(q0 & 0xFFFFFFFFLL)];
                const double c1 = coeff[(int)(q1 & 0xFFFFFFFFLL)];
                const float2 z0 = uu[(int)(q0 >> 32)];
                const float2 z1 = uu[(int)(q1 >> 32)];
                sr0 += c0 * (double)z0.y;
                si0 -= c0 * (double)z0.x;
                sr1 += c1 * (double)z1.y;
                si1 -= c1 * (double)z1.x;
            }
            if (i < hi) {
                const long long q0 = nz_pair[i];
                const double c0 = coeff[(int)(q0 & 0xFFFFFFFFLL)];
                const float2 z0 = uu[(int)(q0 >> 32)];
                sr0 += c0 * (double)z0.y;
                si0 -= c0 * (double)z0.x;
            }
            const double sr = sr0 + sr1, si = si0 + si1;
            const double rden = 1.0 / (sr * sr + si * si);
            const int kw = n_fix + k;
            const size_t abase = (size_t)ge * n_uniq * P
                                 + (size_t)slot_of_wave[kw] * P;
            double ur = 0.0, ui = 0.0;
            for (int p = 0; p < P; p++) {
                const float2 a = amp[abase + p];
                const double cr = ((double)a.x * sr + (double)a.y * si) * rden;
                const double ci = ((double)a.y * sr - (double)a.x * si) * rden;
                const double dr = s_A[(e * P + p) * 2 + 0];
                const double di = s_A[(e * P + p) * 2 + 1];
                ur += dr * cr - di * ci;
                ui += dr * ci + di * cr;
            }
            if (ur != 0.0) atomicAdd(&s_gck[2 * kw + 0], ur);
            if (ui != 0.0) atomicAdd(&s_gck[2 * kw + 1], ui);
            const double ckr = ck_r[kw], cki = ck_i[kw];
            const double qr = ckr * ur - cki * ui;   /* ck·u */
            const double qi = ckr * ui + cki * ur;
            s_pv[idx * 2 + 0] = -(qr * sr + qi * si) * rden;   /* -ck·u/D */
            s_pv[idx * 2 + 1] = -(qi * sr - qr * si) * rden;
        }
    }
    __syncthreads();

    /* ── phase 4: dQ/dcoeff (one thread per free CSR entry, serial over the
     * block) and the dck flush.  The coefficient column is accumulated in a
     * register and landed with ONE global atomic, so the pass needs no
     * barrier after it — the `s_pv` reads are covered by the barrier above. */
    for (int idx = tid; idx < n_nz; idx += nt) {
        const int k = nz_k[idx], m = nz_m[idx], u = nz_u[idx];
        const double g = gamma[k * NB + m];
        double out = 0.0;
        for (int e = 0; e < evn; e++) {
            const float2 z = us[(size_t)(e0 + e) * n_uslot + u];
            const size_t o = (size_t)(e * N + k) * 2;
            out += 2.0 * (s_pv[o + 0] * (double)z.y
                          + s_pv[o + 1] * (double)z.x);
        }
        if (out != 0.0) atomicAdd(g_coeff + m, out * g);
    }
    for (int k = tid; k < Nall; k += nt) {
        if (s_gck[2 * k + 0] != 0.0) atomicAdd(g_ck_r + k, s_gck[2 * k + 0]);
        if (s_gck[2 * k + 1] != 0.0) atomicAdd(g_ck_i + k, s_gck[2 * k + 1]);
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
    int tiles_run;                    /* tile launches (ONE per chunk)       */
    int batch_size, rsize;
    /* chunk rule (one definition, in `create`) */
    int cap, s_ngroups;
    /* events per block chosen at create (a whole number of groups when
       rsize > 1; depends on rsize, n_proj and the free-wave count) */
    int te;
    int smem_bytes;
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
} GPS0GCtx;

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
} GPS0GHandle;

static int* _gps0g_up_i(const int* src, int n) {
    if (n <= 0 || src == NULL) return NULL;
    int* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(int),
                          cudaMemcpyHostToDevice));
    return d;
}

static long long* _gps0g_up_l(const long long* src, int n) {
    if (n <= 0 || src == NULL) return NULL;
    long long* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(long long)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(long long),
                          cudaMemcpyHostToDevice));
    return d;
}

static double* _gps0g_up_d(const double* src, int n) {
    if (n <= 0 || src == NULL) return NULL;
    double* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(double),
                          cudaMemcpyHostToDevice));
    return d;
}

/* The chunk rule.  ONE definition, asked for through `cuda_cap_*`. */
static int gps0g_chunk(int batch_size, int rsize)
{
    int cap = (batch_size > 0) ? batch_size : 50000;
    if (cap > GPS0G_MAX_CAP) cap = GPS0G_MAX_CAP;
    if (rsize > 1) {
        const int a = (cap / rsize) * rsize;
        cap = (a > 0) ? a : rsize;   /* never 0, never unaligned to a group */
    }
    return cap;
}

/* Events per block.  ONE rule: the largest `te` whose shared tile fits
 * GPS0G_SMEM_BUDGET, capped at GPS0G_MAX_TE, floored to a WHOLE number of
 * log-sum groups when `rsize > 1`.
 *
 * Unlike s0f this is not a tuning knob: the free block's `common` is not
 * staged, so the tile holds only `A_{e,p}` (2·P), `|A|²` (P), the three
 * per-event scalars and `pv` (2·N) — no `common`, no `D`.  That is what lets
 * a block carry hundreds of events (s0f's mixed tile was pinned to one
 * group, `rsize` events, by the staged `common`).
 *
 * `rsize > 1` REQUIRES `te >= rsize` (a group closes inside ONE block); when
 * even one group does not fit, this returns 0 and `create` refuses, naming
 * `cuda_flat_pwa_gen_s0f` — the documented fallback. */
static int gps0g_pick_te(int rsize, int P, int Nfree, int Nall, int NB)
{
    int te = GPS0G_MAX_TE;
    while (te > 1
           && (size_t)gps0g_layout(te, P, Nfree, Nall, NB).total
                  * sizeof(double) > (size_t)GPS0G_SMEM_BUDGET) {
        te = (te > 16) ? te - 8 : te - 1;
    }
    if (te < 1) te = 1;
    if (rsize > 1) te = (te / rsize) * rsize;   /* whole groups, or 0 */
    return (te < 1) ? 0 : te;
}

extern "C" {

void cuda_free_context_flat_pwa_gen_s0g(void* vctx);   /* fwd decl */
static void gps0g_fill_afix_launch(GPS0GCtx*, GPS0GHandle*);

void* cuda_create_context_flat_pwa_gen_s0g(
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
    int batch_size, int rsize, int te_hint,
    const double* fixed_vals, int n_fv)
{
    if (n_res != 1) {
        /* the wrapper rejects this first (with the general backend named);
           the kernel refuses too rather than reading a 3-D gamma as 1 factor */
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0g: n_res must be 1, got %d — use "
                "cuda_flat_pwa_gen2 for multi-BW chains\n", n_res);
        return NULL;
    }
    if (n_sw != n_wave || n_rp != n_uniq) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0g: slot_of_wave/rep_of_slot length "
                "mismatch (%d vs n_wave %d, %d vs n_uniq %d)\n",
                n_sw, n_wave, n_rp, n_uniq);
        return NULL;
    }
    if (n_fix < 0 || n_free < 0 || n_fix + n_free != n_wave) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0g: the permuted wave split must cover the "
                "axis exactly (n_fix %d + n_free %d != n_wave %d)\n",
                n_fix, n_free, n_wave);
        return NULL;
    }
    if (n_nk != n_nzm || n_nk != n_nzu || n_np != n_nzm
            || n_lo != n_free || n_hi != n_free) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0g: FREE CSR length mismatch "
                "(nz %d/%d/%d/%d, gs %d/%d, n_free %d)\n",
                n_nk, n_nzm, n_nzu, n_np, n_lo, n_hi, n_free);
        return NULL;
    }
    if (n_flo != n_fix || n_fhi != n_fix || n_fsw != n_fix) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0g: fixed-block length mismatch "
                "(lo %d, hi %d, slot %d, n_fix %d)\n",
                n_flo, n_fhi, n_fsw, n_fix);
        return NULL;
    }
    if (fixed_vals == NULL && n_fix > 0) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0g: a coefficient mask is declared WITH its values — "
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
                "cuda_flat_pwa_gen_s0g: declared values need one entry per "
                "basis column (%d, got %d)\n", n_basis, n_fv);
        return NULL;
    }
    if (n_fix == 0 && n_nfp != 0) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0g: no fixed waves but %d fixed CSR "
                "entries\n", n_nfp);
        return NULL;
    }
    if (n_uniq <= 0 || n_uslot <= 0 || n_wave <= 0 || n_basis <= 0) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0g: empty shape (n_uniq %d, n_uslot %d, "
                "n_wave %d, n_basis %d)\n", n_uniq, n_uslot, n_wave, n_basis);
        return NULL;
    }
    GPS0GCtx* c = (GPS0GCtx*)calloc(1, sizeof(GPS0GCtx));
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
    c->tiles_run = 0;
    c->rsize = (rsize > 0) ? rsize : 1;
    c->batch_size = batch_size;

    /* ── the chunk rule (see `gps0g_chunk`) ─────────────────────────────── */
    c->cap = gps0g_chunk(batch_size, c->rsize);
    c->s_ngroups = (c->cap + c->rsize - 1) / c->rsize;

    /* ── the tile size: ONE rule (see `gps0g_pick_te`) ────────────────── */
    const int P = c->n_proj;
    const int te_max = gps0g_pick_te(c->rsize, P, n_free, n_wave, n_basis);
    c->te = (te_hint > 0) ? te_hint : te_max;
    if (c->rsize > 1 && c->te < c->rsize) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0g: a log-sum group (%d events) does not "
                "fit one block (max %d events with n_wave %d, n_proj %d, "
                "n_basis %d) — use cuda_flat_pwa_gen_s0f, which walks a group "
                "inside one block\n",
                c->rsize, te_max, n_wave, c->n_proj, n_basis);
        free(c);
        return NULL;
    }
    /* a hint must still hold whole groups (never straddle a group boundary),
       and it must fit the device: the hint is a test/benchmark hook, and a
       silently wrong tile is exactly the failure this guards against */
    if (c->rsize > 1 && c->te >= c->rsize && (c->te % c->rsize) != 0) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0g: te %d is not a multiple of rsize %d — "
                "a tile would split a log-sum group\n", c->te, c->rsize);
        free(c);
        return NULL;
    }
    if (c->te < 1) c->te = 1;
    {
        GPS0GLayout lay = gps0g_layout(c->te, P, n_free, n_wave, n_basis);
        c->smem_bytes = (int)(lay.total * (long long)sizeof(double));
        int optin = 0;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               0);
        if (c->smem_bytes > optin) {
            fprintf(stderr,
                    "cuda_flat_pwa_gen_s0g: shared tile %d B exceeds the "
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
                (const void*)gps0g_tile_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, c->smem_bytes);
            if (e != cudaSuccess) {
                fprintf(stderr,
                        "cuda_flat_pwa_gen_s0g: cannot opt in to %d B shared "
                        "memory (%s); n_wave %d, te %d\n",
                        c->smem_bytes, cudaGetErrorString(e), n_wave, c->te);
                free(c);
                return NULL;
            }
            opt_in_max = (size_t)c->smem_bytes;
        }
    }

    c->fl_table = _gps0g_up_d(fl_table, n_fl);
    c->fl_q_index = _gps0g_up_i(fl_q_index, n_fq);
    c->num_r = _gps0g_up_d(num_r, n_num);
    c->num_i = _gps0g_up_d(num_i, n_num);
    c->num_m_index = _gps0g_up_i(num_m_index, n_nm);
    c->bas_r = _gps0g_up_d(bas_r, n_bas);
    c->bas_i = _gps0g_up_d(bas_i, n_bas);
    c->bas_m_index = _gps0g_up_i(bas_m_index, n_bm);
    c->uslot_row = _gps0g_up_i(uslot_row, n_ur);
    c->uslot_col = _gps0g_up_i(uslot_col, n_uc);
    c->slot_of_wave = _gps0g_up_i(slot_of_wave, n_sw);
    c->rep_of_slot = _gps0g_up_i(rep_of_slot, n_rp);
    c->nz_k = _gps0g_up_i(nz_k, n_nk);
    c->nz_m = _gps0g_up_i(nz_m, n_nzm);
    c->nz_u = _gps0g_up_i(nz_u, n_nzu);
    c->nz_pair = _gps0g_up_l(nz_pair, n_np);
    c->gs_lo = _gps0g_up_i(gs_lo, n_lo);
    c->gs_hi = _gps0g_up_i(gs_hi, n_hi);
    c->fix_lo = _gps0g_up_i(fix_lo, n_flo);
    c->fix_hi = _gps0g_up_i(fix_hi, n_fhi);
    c->fix_slot_of_wave = _gps0g_up_i(fix_slot_of_wave, n_fsw);
    c->fix_pair = _gps0g_up_l(fix_pair, n_nfp);
    c->gamma = _gps0g_up_d(gamma, n_g);
    c->angle_index = _gps0g_up_i(angle_index, n_ai);
    c->angle_k = _gps0g_up_d(angle_k, n_ak);
    c->angle_b = _gps0g_up_d(angle_b, n_ab_);
    c->am_r = _gps0g_up_d(am_r, n_am);
    c->am_i = _gps0g_up_d(am_i, n_am);
    /* variant B: the declared VALUES live in the CONTEXT — it is
       per mask, so they can never leak into another mask's cache */
    c->fixed_vals = _gps0g_up_d(fixed_vals, n_fv);
    c->m_min = m_min; c->m_delta = m_delta; c->m_bins = m_bins;
    c->q_min = q_min; c->q_delta = q_delta; c->q_bins = q_bins;

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

void cuda_free_context_flat_pwa_gen_s0g(void* vctx) {
    GPS0GCtx* c = (GPS0GCtx*)vctx;
    if (!c) return;
    #define F(p) if (c->p) cudaFree((void*)c->p)
    #define FH(p) if (c->p) cudaFreeHost(c->p)
    F(fl_table); F(fl_q_index); F(num_r); F(num_i); F(num_m_index);
    F(bas_r); F(bas_i); F(bas_m_index);
    F(uslot_row); F(uslot_col); F(slot_of_wave); F(rep_of_slot);
    F(nz_k); F(nz_m); F(nz_u); F(nz_pair); F(gs_lo); F(gs_hi); F(gamma);
    F(fix_lo); F(fix_hi); F(fix_slot_of_wave); F(fix_pair);
    F(angle_index); F(angle_k); F(angle_b); F(am_r); F(am_i);
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

void* cuda_load_data_flat_pwa_gen_s0g(
    void* vctx, const double* mass, const double* mom, const double* ang,
    const double* weight, const double* bkg, int ne)
{
    GPS0GCtx* c = (GPS0GCtx*)vctx;
    if (!c) return NULL;
    GPS0GHandle* h = (GPS0GHandle*)calloc(1, sizeof(GPS0GHandle));
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
    h->weight = _gps0g_up_d(weight, ne);
    h->bkg = _gps0g_up_d(bkg, ne);
    if (cudaMalloc(&h->us, (size_t)ne * c->n_uslot * sizeof(float2))
            != cudaSuccess) goto fail;
    if (cudaMalloc(&h->amp, (size_t)ne * c->n_uniq * c->n_proj
                           * sizeof(float2)) != cudaSuccess) goto fail;
    /* the fixed-wave cache: 8·P·n_fix bytes per event, this handle's own */
    if (c->n_fix > 0 && cudaMalloc(&h->afix,
            (size_t)ne * c->n_fix * c->n_proj * sizeof(float2))
            != cudaSuccess) goto fail;
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
        double* t_mass = _gps0g_up_d(mass, ne * c->n_mass);
        double* t_mom = _gps0g_up_d(mom, ne * c->n_momentum);
        double* t_ang = _gps0g_up_d(ang, ne * c->n_angle_total * c->n_comp);
        if (!t_mass || !t_mom || !t_ang) {
            if (t_mass) cudaFree(t_mass);
            if (t_mom) cudaFree(t_mom);
            if (t_ang) cudaFree(t_ang);
            goto fail;
        }
        {
            const long long total = (long long)ne * c->n_uslot;
            const int threads = GPS0G_BLOCK;
            long long blocks = (total + threads - 1) / threads;
            if (blocks > 65535) blocks = 65535;
            if (blocks < 1) blocks = 1;
            gps0g_fill_us<<<(unsigned)blocks, threads, 0, c->stream>>>(
                c->bas_r, c->bas_i, c->uslot_row, c->uslot_col,
                c->m_min, c->m_delta, c->m_bins,
                t_mass, c->n_mass, ne, c->n_uslot, h->us);
            CUDA_CHECK(cudaGetLastError());
        }
        {
            int fb = c->n_uniq < 32 ? 32 : (c->n_uniq < GPS0G_BLOCK
                                            ? GPS0G_BLOCK : c->n_uniq);
            if (fb > 1024) fb = 1024;
            const size_t shm = (size_t)c->n_ab * sizeof(double);
            gps0g_fill_amp<<<ne, fb, shm, c->stream>>>(
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
            gps0g_fill_afix_launch(c, h);
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

void cuda_free_data_flat_pwa_gen_s0g(void* vh) {
    GPS0GHandle* h = (GPS0GHandle*)vh;
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

/* ONE launch per chunk.  Batching needs no kernel logic: the chunk is
 * expressed as POINTER OFFSETS into the per-event data plus the chunk length
 * as the kernel's `ne`, so `e0`/`ge` are chunk-local by construction.  A block
 * carries `te` events (whole groups when `rsize > 1`), and the chunk is a
 * multiple of `rsize`, so a group never straddles a block or a chunk. */
static void gps0g_launch(GPS0GCtx* c, GPS0GHandle* h, int st, int nb,
                         double norm, int use_norm)
{
    if (nb <= 0) return;
    const int tiles = (nb + c->te - 1) / c->te;
    gps0g_tile_kernel<<<tiles, GPS0G_BLOCK, (size_t)c->smem_bytes, c->stream>>>(
        h->us + (size_t)st * c->n_uslot,
        h->amp + (size_t)st * c->n_uniq * c->n_proj,
        c->slot_of_wave,
        (c->n_fix > 0
             ? h->afix + (size_t)st * c->n_fix * c->n_proj : NULL),
        c->n_fix,
        c->nz_m, c->nz_u, c->nz_pair, c->nz_k,
        c->gs_lo, c->gs_hi,
        c->gamma, c->n_nz, c->p_coeff, c->p_ck_r, c->p_ck_i,
        h->weight + st, h->bkg + st,
        norm, use_norm, c->rsize, c->te,
        c->n_free, c->n_proj, c->n_basis, c->n_uslot, c->n_uniq, c->n_wave,
        nb, h->P_dev + st,
        c->g_ck_r, c->g_ck_i, c->g_coeff, c->acc, c->acc + 1);
    c->tiles_run++;
    CUDA_CHECK(cudaGetLastError());
}

/* The fixed-wave cache, filled ONCE per handle — at `load_data`, from the
 * VALUES the mask was declared with (`c->fixed_vals`, per mask, in the
 * context).  It walks the ORIGINAL fixed-block CSR and stores phase A's
 * `common`, so the tile can later read it instead of recomputing `D`.  The
 * kernel is not forked: the caller passes the pointer. */
static void gps0g_fill_afix_launch(GPS0GCtx* c, GPS0GHandle* h)
{
    if (c->n_fix <= 0 || !h->afix || h->ne <= 0) return;
    if (c->fixed_vals == NULL) {
        /* create's invariant (n_fix > 0 => values) — the two guards agree */
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0g: no declared values (n_fix %d) — "
                "the cache cannot be filled\n", c->n_fix);
        return;
    }
    const long long total = (long long)h->ne * (long long)c->n_fix;
    long long blocks = (total + GPS0G_BLOCK - 1) / GPS0G_BLOCK;
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;
    gps0g_fill_afix<<<(unsigned)blocks, GPS0G_BLOCK, 0, c->stream>>>(
        h->us, h->amp, c->fix_slot_of_wave, c->fix_pair,
        c->fix_lo, c->fix_hi,
        /* create REFUSES a masked context without values, so this is the
           only source: a NULL here would read never-uploaded p_coeff */
        c->fixed_vals,
        c->n_fix, h->ne, c->n_uslot, c->n_uniq, c->n_proj, h->afix);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_compute_flat_pwa_gen_s0g(
    void* vctx, void* vh, const double* ck_r, const double* ck_i,
    const double* coeff, double norm, int use_norm,
    double* oQ, double* oDn, double* oP,
    double* og_ck_r, double* og_ck_i, double* og_coeff)
{
    GPS0GCtx* c = (GPS0GCtx*)vctx;
    GPS0GHandle* h = (GPS0GHandle*)vh;
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
                "cuda_flat_pwa_gen_s0g: handle has no Afix cache (n_fix %d) — load_data must "
                "fill it; refusing to compute\n", c->n_fix);
        return;
    }
    /* the accumulators span the WHOLE sample: zero once, read once */
    cudaMemsetAsync(c->acc, 0, 2 * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_ck_r, 0, (size_t)N * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_ck_i, 0, (size_t)N * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_coeff, 0, (size_t)NB * sizeof(double), c->stream);

    const int ne_total = h->ne;
    /* ONE kernel per chunk: fixed and free blocks in the same pass, so there
     * is no F/B to interleave and nothing to keep in L2 between them. */
    const int bs = (c->cap > 0) ? c->cap : 1;
    for (int st = 0; st < ne_total; st += bs) {
        const int nb = (ne_total - st > bs) ? bs : (ne_total - st);
        gps0g_launch(c, h, st, nb, norm, use_norm);
        CUDA_CHECK(cudaGetLastError());
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

/* The tile size `gps0g_layout` gives for an arbitrary shape — the same
 * function the kernel and `create` use, so a test can assert the buffer
 * arithmetic without a second, drifting copy in Python. */
long long cuda_tile_bytes_flat_pwa_gen_s0g(int te, int n_wave, int n_proj,
                                           int n_basis, int n_free)
{
    GPS0GLayout L = gps0g_layout(te, n_proj, n_free, n_wave, n_basis);
    return (long long)L.total * (long long)sizeof(double);
}

/* The fixed-wave cache's size for a sample of `ne` events, in B. */
long long cuda_fix_bytes_flat_pwa_gen_s0g(void* vctx, int ne) {
    GPS0GCtx* c = (GPS0GCtx*)vctx;
    if (!c || ne <= 0) return 0;
    return (long long)ne * c->n_fix * c->n_proj
           * (long long)sizeof(float2);
}

/* The chunk plan for a sample of `ne` events: out[0] = cap, out[1] = the
 * number of chunks, out[2] = 1 when every chunk start and every non-final
 * chunk length is a multiple of `rsize` (a group never straddles a chunk). */
void cuda_chunk_plan_flat_pwa_gen_s0g(void* vctx, int ne, int* out) {
    GPS0GCtx* c = (GPS0GCtx*)vctx;
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

/* Tile launches since `create` — ONE per chunk, which is the whole point:
 * there is no F and no B to count. */
int cuda_tile_count_flat_pwa_gen_s0g(void* vctx) {
    GPS0GCtx* c = (GPS0GCtx*)vctx;
    return c ? c->tiles_run : 0;
}

/* 1 when this handle's `Afix` cache has been filled (with declared values
 * that is already true right after `load_data`). */
int cuda_handle_afix_ready_flat_pwa_gen_s0g(void* vh) {
    GPS0GHandle* h = (GPS0GHandle*)vh;
    return (h != NULL && h->afix_ready) ? 1 : 0;
}

/* 1 when this context carries DECLARED VALUES (the eager-fill path). */
int cuda_has_fixed_values_flat_pwa_gen_s0g(void* vctx) {
    GPS0GCtx* c = (GPS0GCtx*)vctx;
    return (c != NULL && c->fixed_vals != NULL) ? 1 : 0;
}

int cuda_n_fix_flat_pwa_gen_s0g(void* vctx) {
    GPS0GCtx* c = (GPS0GCtx*)vctx;
    return c ? c->n_fix : 0;
}

int cuda_n_free_flat_pwa_gen_s0g(void* vctx) {
    GPS0GCtx* c = (GPS0GCtx*)vctx;
    return c ? c->n_free : 0;
}

int cuda_n_nz_free_flat_pwa_gen_s0g(void* vctx) {
    GPS0GCtx* c = (GPS0GCtx*)vctx;
    return c ? c->n_nz : 0;
}

int cuda_n_nz_fix_flat_pwa_gen_s0g(void* vctx) {
    GPS0GCtx* c = (GPS0GCtx*)vctx;
    return c ? c->n_nz_fix : 0;
}

int cuda_smem_bytes_flat_pwa_gen_s0g(void* vctx) {
    GPS0GCtx* c = (GPS0GCtx*)vctx;
    return c ? c->smem_bytes : 0;
}

/* Runtime resource report (host side, no kernel launch).
 *
 * CUDA exposes the compiled resource usage per FUNCTION, which is what the
 * `__launch_bounds__` / `-D..._MIN_BLOCKS` choice is trading against:
 *   out[0] registers/thread          (cudaFuncGetAttributes.numRegs)
 *   out[1] local bytes > 0  => the kernel SPILLS
 *   out[2] blocks/SM at the REAL block size and dynamic smem
 *   out[3] warps/SM  (= blocks x GPS0G_BLOCK/32)
 *   out[4] maxThreadsPerBlock
 *   out[5] static shared bytes
 * Everything here is a property of the compiled cubin; `smem` is the tile's
 * dynamic shared memory for the CURRENT shape (c->smem_bytes), so the
 * occupancy figure is the one the fit actually gets.
 */
extern "C" void cuda_stats_flat_pwa_gen_s0g(int smem, int* out)
{
    if (!out) return;
    cudaFuncAttributes a;
    cudaFuncGetAttributes(&a, (const void*)gps0g_tile_kernel);
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, (const void*)gps0g_tile_kernel, GPS0G_BLOCK, (size_t)(smem > 0 ? smem : 0));
    out[0] = a.numRegs;
    out[1] = (int)a.localSizeBytes;
    out[2] = blocks;
    out[3] = blocks * (GPS0G_BLOCK / 32);
    out[4] = a.maxThreadsPerBlock;
    out[5] = (int)a.sharedSizeBytes;
}

int cuda_cap_flat_pwa_gen_s0g(void* vctx) {
    GPS0GCtx* c = (GPS0GCtx*)vctx;
    return c ? c->cap : 0;
}

int cuda_ngroups_flat_pwa_gen_s0g(void* vctx) {
    GPS0GCtx* c = (GPS0GCtx*)vctx;
    return c ? c->s_ngroups : 0;
}

/* The events per block this shape allows (0 = an `rsize` group cannot fit
 * one block, i.e. the caller must use `cuda_flat_pwa_gen_s0f`). */
int cuda_te_for_flat_pwa_gen_s0g(int rsize, int n_proj, int n_free,
                                 int n_wave, int n_basis)
{
    return gps0g_pick_te((rsize > 0) ? rsize : 1,
                         (n_proj > 0) ? n_proj : 1,
                         (n_free > 0) ? n_free : 0,
                         (n_wave > 0) ? n_wave : 1,
                         (n_basis > 0) ? n_basis : 1);
}

int cuda_te_flat_pwa_gen_s0g(void* vctx) {
    GPS0GCtx* c = (GPS0GCtx*)vctx;
    return c ? c->te : 0;
}

int cuda_get_device_count_flat_pwa_gen_s0g(void) {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
}

void cuda_get_device_name_flat_pwa_gen_s0g(char* buf, int len) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    snprintf(buf, len, "%s", p.name);
}

}  /* extern "C" */
