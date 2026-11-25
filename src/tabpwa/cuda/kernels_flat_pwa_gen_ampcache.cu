/**
 * cuda_flat_pwa_gen_ampcache — the flat `pwa_gen` config with EVERY
 * coefficient pinned, so the whole per-wave amplitude is a per-handle cache.
 *
 * Case-1 copy (kernel policy §1): own `.cu`, own `.so`, own wrapper, own
 * symbols.  `kernels_flat_pwa_gen_s0f.cu` / `…s0g.cu` and every other
 * kernel/wrapper in the tree stay byte-identical.
 *
 * ── what is cached, and why ──────────────────────────────────────────────
 *
 * A wave's contribution is
 *
 *     A_{e,k,p} = ck_k · Amp_{e,slot(k),p} · conj(D_{e,k}) / |D_{e,k}|²
 *     D_{e,k}   = Π_{l<n_res} Σ_{m ∈ nz(k,l)} gamma[k,l,m]·coeff[m]·Z_m(e)
 *
 * (Z = the `us` cache).  The denominator is the reference's own FORM — a
 * product over the factor axis, each factor one sparse dot over the basis rows
 * that factor selects (CSR row `k·n_res + l`).  `n_res == 1` is the single-row
 * case and takes one dot and one `1/|D|²`, i.e. the arithmetic this kernel had
 * before the axis was general — bit for bit.
 *
 * `s0f` caches that for the waves whose `coeff` columns are all fixed.  Here
 * EVERY coefficient is pinned (the flat family's analogue of
 * `cuda32_v4_pwa_cache`), so every wave qualifies and the cache holds the
 * WHOLE amplitude contribution of every wave:
 *
 *     Afix[e, k, p]        (float2, index ((e·n_wave) + k)·n_proj + p)
 *
 * filled ONCE per loaded handle by `gps0a_fill_afix`, whose per-factor sparse
 * dot is `s0f`'s fixed-wave fill verbatim (same packed 64-bit `nz_pair` step,
 * same two fp64 accumulation chains, same `1/|D|²`, same
 * `(a.x·sr + a.y·si)·rden` / `(a.y·sr − a.x·si)·rden`), so the cached number
 * is exactly the one the free path would compute for the same `D`.
 *
 * What remains per iteration is the ck contraction — the whole per-iteration
 * work:
 *
 *     A[e,p] = Σ_k ck_k·Afix[e,k,p]
 *     P_e    = Σ_p |A[e,p]|²
 *     dck_k  = Σ_{e,p} dA[e,p]·Afix[e,k,p]
 *     dcoeff = 0                     for EVERY column (there is no `D` here)
 *
 * NO conjugation on `Afix` in the backward: `dA = dQ/dP·conj(A)` ALREADY
 * carries it (the reference's `grad_ck = Σ dQ_dA·common·fa` with
 * `dQ_dA = dQ_dP·conj(A)`), so conjugating again would be `|A|²` and the
 * fit would descend in the wrong direction — the one mistake a copy of this
 * kernel can make silently.
 *
 * so the per-ITERATION work has no `us` walk, no `gamma`, no coefficient CSR,
 * no `pv`, no coefficient-gradient pass and no F/T/B split: ONE tile kernel per
 * chunk, with two sweeps over the cache (forward for A, backward for dck) and
 * FOUR barriers per block at `rsize == 1` (FIVE at `rsize > 1`, which adds the
 * group close).  `dQ/dcoeff` being exactly 0 is the CONTRACT, not a
 * shortcut — a pinned coefficient has no direction in `x`, and the constraint
 * layer drops those entries.
 *
 * ── the grouped log-sum (`resolution_size`) ──────────────────────────────
 *
 * `s0f`'s fused branch, kept as it is: a block owns WHOLE groups (`per_blk`
 * a multiple of `rsize`), the group sums are closed in the block, and
 * `rsize <= te` is the rule that makes that possible (`rsize == 1` is the
 * per-event objective and needs no group at all).  A group that cannot fit
 * one block's tile (`rsize > te`) is walked by ONE block in `te`-sized inner
 * chunks — `s0f`'s path-B semantics, which is where its F/T/B fallback ends
 * up: the group totals accumulate in `s_tot`, the scale closes once, and the
 * backward re-reads the group's `Afix` slice with the finished constant.  The
 * cache is iteration-invariant, so the second read costs a gather, not a
 * recompute of `D`.
 *
 * ── memory ──────────────────────────────────────────────────────────────
 *
 * Per handle: `8 · n_proj · n_wave · ne` bytes — the whole cache, and nothing
 * else scales with `ne` (the tile is shared memory, `batch_size` bounds the
 * chunk).  The kernel keeps no scratch of its own.  A handle is filled ONCE,
 * by its FIRST `compute` (the wrapper's `_fill_once`, `cuda_v4_pwa_cache`'s
 * lifecycle: per-handle `_filled` / `_coeff`, the first call is the fill and
 * every later one is guarded against the vector it was filled from).  There is
 * no eager path at `load_data` and no `coeff=` constructor form — one
 * lifecycle, so the state cannot be reached twice, differently.  The Gram
 * (`cuda_gram_…`, below) goes through the same `_fill_once`.
 *
 * Precision policy is the flat family's: the stored caches (`us`, `amp`,
 * `Afix`) are float2, every real arithmetic step stays fp64 (gradients
 * included).
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

#define GPS0A_BLOCK       256      /* threads per block (multiple of 32) */
/* Launch bounds: the SECOND parameter is a REGISTER BUDGET choice, not a
 * correctness one.  The same per-arch reasoning as s0f/s0g applies (a fat
 * binary needs `__CUDA_ARCH__`; a per-build override is
 * `TABPWA_NVCC_FLAGS_SM70="-DGPS0A_MIN_BLOCKS=6"`).  Measured: N=4 is clean
 * on every arch, N=5 is clean on 75/120 and spills elsewhere. */
#ifndef GPS0A_MIN_BLOCKS
#  if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#    define GPS0A_MIN_BLOCKS 5
#  else
#    define GPS0A_MIN_BLOCKS 4
#  endif
#endif
#define GPS0A_TILE_BOUNDS __launch_bounds__(GPS0A_BLOCK, GPS0A_MIN_BLOCKS)
#define GPS0A_MAX_TE      256      /* events per block ceiling     */
#define GPS0A_MAX_CAP     (1 << 20)  /* ceiling on the chunk length  */
/*: soft shared-memory budget for choosing `te`.  Below this the launch needs
 *  no opt-in (so no `cudaFuncSetAttribute` and the best occupancy); above it
 *  `create` shrinks `te` first and only then crosses the opt-in. */
#define GPS0A_SMEM_BUDGET (48 * 1024)

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = call;                                             \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                               \
        }                                                                    \
    } while (0)

/* ── Catmull-Rom (the repo-wide idiom, identical to gen2/1bw/s0e/s0f) ─── */
__device__ double gps0a_cr1d(double pm1, double p0, double p1, double p2,
                            double t) {
    return p0 + 0.5 * t * (-pm1 + p1 + t * (2.0 * pm1 - 5.0 * p0 + 4.0 * p1
                                            - p2 + t * (-pm1 + 3.0 * p0
                                                        - 3.0 * p1 + p2)));
}

__device__ double gps0a_interp_real(const double* __restrict__ t, int row,
                                   double x, double xmin, double xdelta,
                                   int bins) {
    double diff = (x - xmin) / xdelta;
    int b = max(0, min((int)floor(diff), bins - 2));
    double s = max(0.0, min(diff - b, 1.0));
    int base = row * bins + b;
    int lo = row * bins, hi = (row + 1) * bins - 1;
    int im1 = base > lo ? base - 1 : base;
    int i2 = base + 2 <= hi ? base + 2 : base + 1;
    return gps0a_cr1d(t[im1], t[base], t[base + 1], t[i2], s);
}

__device__ void gps0a_interp_c(const double* __restrict__ tr,
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
    *orr = gps0a_cr1d(tr[im1], tr[base], tr[base + 1], tr[i2], s);
    *oii = gps0a_cr1d(ti[im1], ti[base], ti[base + 1], ti[i2], s);
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
__global__ void gps0a_fill_us(
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
        gps0a_interp_c(bas_r, bas_i, uslot_row[u], mv,
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
 * projection and every deduplicated wave slot.
 *===========================================================================*/
__global__ void gps0a_fill_amp(
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
            gps0a_interp_c(num_r, num_i, k * n_res + r, mv,
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
            fl *= gps0a_interp_real(fl_table, k * n_decay + d, qv,
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
 * LOAD-TIME FILL 3 — the amplitude cache (the ONLY per-iteration cache).
 *
 * Every coefficient is pinned, so every wave's denominator is
 * iteration-invariant and its whole contribution
 * `Afix[e,k,p] = Amp·conj(D)/|D|²` is phase A's `common`, evaluated once per
 * loaded handle.
 *
 * The denominator is the REFERENCE's own form — a product over the factor axis
 *
 *     D_k = Π_{l<n_res}  s_{k,l}
 *     s_{k,l} = Σ_m gamma[k,l,m]·coeff[m]·Z_m(e)
 *
 * so the CSR is indexed by the flat `(wave, factor)` row `k·n_res + l`
 * (`row_start`), and each factor's sparse dot is phase A's body verbatim — the
 * same packed 64-bit `nz_pair` step, the same two independent fp64
 * accumulation chains — with the row's gamma weight folded into the
 * coefficient.  `n_res == 1` is the ONE-row case: a single dot and a single
 * `1/|D|²`, i.e. the arithmetic this kernel had before the factor axis
 * existed (bit for bit).  For `n_res > 1` the row dots are multiplied as
 * complex numbers and the reciprocal is taken once, on the product — exactly
 * what the reference divides by.
 *
 * One thread per `(event, wave)`; chunking is the caller's (the fill runs
 * ONCE for the whole handle, so `ne` here is the sample length).
 *===========================================================================*/
/* One factor's dot: s = Σ_i nz_val[i]·coeff[col_i]·Z_{uslot_i}(e). */
__device__ __forceinline__ void gps0a_row_dot(
    const float2* __restrict__ uu,
    const long long* __restrict__ nz_pair,
    const double* __restrict__ nz_val,
    const double* __restrict__ coeff,
    int lo, int hi, double* __restrict__ osr, double* __restrict__ osi)
{
    double sr0 = 0.0, si0 = 0.0, sr1 = 0.0, si1 = 0.0;
    int i = lo;
    const int two = lo + ((hi - lo) & ~1);
    for (; i < two; i += 2) {
        const long long p0 = nz_pair[i], p1 = nz_pair[i + 1];
        const double c0 = nz_val[i] * coeff[(int)(p0 & 0xFFFFFFFFLL)];
        const double c1 = nz_val[i + 1] * coeff[(int)(p1 & 0xFFFFFFFFLL)];
        const float2 z0 = uu[(int)(p0 >> 32)];
        const float2 z1 = uu[(int)(p1 >> 32)];
        sr0 += c0 * (double)z0.y;
        si0 -= c0 * (double)z0.x;
        sr1 += c1 * (double)z1.y;
        si1 -= c1 * (double)z1.x;
    }
    if (i < hi) {
        const long long p = nz_pair[i];
        const double c = nz_val[i] * coeff[(int)(p & 0xFFFFFFFFLL)];
        const float2 z = uu[(int)(p >> 32)];
        sr0 += c * (double)z.y;
        si0 -= c * (double)z.x;
    }
    *osr = sr0 + sr1;
    *osi = si0 + si1;
}

__global__ void gps0a_fill_afix(
    const float2* __restrict__ us,             /* [ne, n_uslot] Z = i·basis */
    const float2* __restrict__ amp,            /* [ne, n_uniq, P]          */
    const int* __restrict__ slot_of_wave,      /* [n_wave] -> amp slot     */
    const long long* __restrict__ nz_pair,     /* [nnz] m | (u << 32)      */
    const double* __restrict__ nz_val,         /* [nnz] the gamma weight   */
    const int* __restrict__ row_start,         /* [n_wave·n_res + 1]       */
    const double* __restrict__ coeff,          /* [n_basis]                */
    int n_wave, int n_res, int ne, int n_uslot, int n_uniq, int n_proj,
    float2* __restrict__ afix)                 /* [ne, n_wave, P]          */
{
    const long long total = (long long)ne * (long long)n_wave;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         g < total; g += stride) {
        const int e = (int)(g / n_wave);
        const int k = (int)(g - (long long)e * n_wave);
        const float2* uu = us + (size_t)e * n_uslot;
        double sr, si;
        if (n_res == 1) {
            /* ONE factor: one dot, one reciprocal — unchanged arithmetic */
            const int row = k;
            gps0a_row_dot(uu, nz_pair, nz_val, coeff,
                          row_start[row], row_start[row + 1], &sr, &si);
        } else {
            /* D = Π_l s_l over the factor axis, then ONE reciprocal */
            double dr = 1.0, di = 0.0;
            for (int l = 0; l < n_res; l++) {
                const int row = k * n_res + l;
                double ar, ai;
                gps0a_row_dot(uu, nz_pair, nz_val, coeff,
                              row_start[row], row_start[row + 1], &ar, &ai);
                const double tr = dr * ar - di * ai;
                di = dr * ai + di * ar;
                dr = tr;
            }
            sr = dr;
            si = di;
        }
        const double rden = 1.0 / (sr * sr + si * si);
        const size_t abase = (size_t)e * n_uniq * n_proj
                             + (size_t)slot_of_wave[k] * n_proj;
        for (int p = 0; p < n_proj; p++) {
            const float2 a = amp[abase + p];
            afix[((size_t)e * n_wave + k) * n_proj + p] =
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
 *   pe, q, sc [te]   P_e, the group summand q_e, the group scale
 *   fus   [16·N]     per-(wave, slice) partials for dQ/dck (2·8·N)
 *   tot   [3]        path B: S_g, Pb_g, W_g
 *===========================================================================*/
typedef struct {
    int A, pe, q, sc, fus, tot, total;
} GPS0ALayout;

__host__ __device__ __forceinline__ GPS0ALayout gps0a_layout(
    int te, int P, int Nall)
{
    GPS0ALayout L;
    int o = 0;
    L.A   = o; o += 2 * te * P;
    L.pe  = o; o += te;
    L.q   = o; o += te;
    L.sc  = o; o += te;
    L.fus = o; o += 16 * Nall;
    L.tot = o; o += 3;
    L.total = o;
    return L;
}

/*=============================================================================
 * ONE pass over (`e0`, `e0+ev): A_{e,p} = Σ_k ck_k·Afix[e,k,p].
 *
 * A thread owns one (e,p) element and walks its `Afix` k-run contiguously —
 * `((e·N)+k)·P+p` strides by `P` floats, i.e. one cache line per one or two
 * k — which is the same line pattern the backward re-reads.
 *
 * MEASURED (and why the layout stays as it is): the k-innermost alternative
 * `Afix[e,p·N+k]` — a pure permutation inside each event's block, so the same
 * numbers in the same kernel — was A/B'd at 0.999x (0.6 M) and 0.998x (6 M),
 * i.e. nothing.  `ncu` says why: the tile is FP64-THROUGHPUT bound (FP64 pipe
 * 85 %, DRAM 11 %, L2 13 % of peak), so the cache's byte pattern is not what
 * limits it — the contraction's fp64 instruction count is.  The addressing is
 * therefore not a lever here, and a second index map is dead weight.
 *===========================================================================*/
__device__ void gps0a_build_A(
    const float2* __restrict__ afix,
    const double* __restrict__ ck_r, const double* __restrict__ ck_i,
    int N, int P, int e0, int ev, double* __restrict__ s_A)
{
    const int tid = threadIdx.x, nt = blockDim.x;
    for (int idx = tid; idx < ev * P; idx += nt) {
        const int e = idx / P, p = idx - e * P;
        const float2* base = afix + ((size_t)(e0 + e) * N) * P + p;
        double ar = 0.0, ai = 0.0;
        for (int k = 0; k < N; k++) {
            const float2 a = base[(size_t)k * P];
            const double cr = ck_r[k], ci = ck_i[k];
            ar += cr * (double)a.x - ci * (double)a.y;
            ai += cr * (double)a.y + ci * (double)a.x;
        }
        s_A[idx * 2 + 0] = ar;
        s_A[idx * 2 + 1] = ai;
    }
}

/*=============================================================================
 * dQ/dck = Σ_{e,p} dQ/dA_{e,p}·Afix[e,k,p], with (wave, slice) shared
 * partials.  NO conjugation on `Afix` here: `s_dA` is already
 * `dQ/dP·conj(A)` (the caller folds the conjugate in, exactly as the
 * reference's `dQ_dA = dQ_dP·conj(A)` does), so `dA·Afix` is
 * `dQ/dP·|A|²·common·fa` — the correct Wirtinger half.
 *
 * The naive shape (thread = (e,p), one shared atomic per (e,p,k)) measured
 * 35-51 ns/event on s0f: the shared atomic on 2·N addresses from every
 * thread dominated.  Here a thread owns (k, slice) and walks the block's
 * elements at a stride of `nsl`, so the shared atomics drop from
 * 2·ev·P·N to 2·N·nsl and the block does ONE global atomic per wave at the
 * flush.  The `Afix` reads keep the contiguous k run.
 *
 * Partials ACCUMULATE into `s_fus` (path B calls this once per inner chunk),
 * so the caller zeroes it once and flushes once.
 *===========================================================================*/
__device__ void gps0a_bwd(
    const float2* __restrict__ afix,
    const double* __restrict__ s_dA,
    int N, int P, int e0, int ev, double* __restrict__ s_fus)
{
    const int tid = threadIdx.x, nt = blockDim.x;
    const int nsl0 = (N > 0) ? (nt / N) : 1;
    const int nsl = (nsl0 > 8) ? 8 : (nsl0 < 1 ? 1 : nsl0);
    const int nact = N * nsl;
    const int elems = ev * P;
    for (int t = tid; t < nact; t += nt) {
        const int k = t % N;
        const int sl = t / N;
        double ur = 0.0, ui = 0.0;
        for (int i = sl; i < elems; i += nsl) {
            const int e = i / P, p = i - e * P;
            const float2 a = afix[((size_t)(e0 + e) * N + k) * P + p];
            const double dr = s_dA[i * 2 + 0], di = s_dA[i * 2 + 1];
            ur += dr * (double)a.x - di * (double)a.y;
            ui += dr * (double)a.y + di * (double)a.x;
        }
        if (ur != 0.0) atomicAdd(&s_fus[(k * nsl + sl) * 2 + 0], ur);
        if (ui != 0.0) atomicAdd(&s_fus[(k * nsl + sl) * 2 + 1], ui);
    }
}

/* The (wave, slice) grid the partials live on — ONE definition, used by the
 * zeroing loop, the backward and the flush. */
__device__ __forceinline__ int gps0a_nsl(int N, int nt)
{
    const int nsl0 = (N > 0) ? (nt / N) : 1;
    return (nsl0 > 8) ? 8 : (nsl0 < 1 ? 1 : nsl0);
}

/*=============================================================================
 * THE GRAM — the phase-space pre-integration the pinned cache makes exact
 *
 *     D[k,j] = Σ_e w_e Σ_p conj(Afix[e,k,p])·Afix[e,j,p]     (N×N, N = n_wave)
 *
 * Conjugation is on the FIRST index, which is `integrated_pwa_gen`'s
 * `einsum("e,ekp,ejp->kj", w, c.conj(), c)` and `cuda_gram_from_cache_v4`'s
 * `conj(c_k)·c_j` — the SAME convention, so
 *
 *     Σ_e w_e P_e = Re(ck† · D · ck)     and     ∂/∂ck = conj(D · ck)
 *
 * with no transpose and no extra sign.  The cache is the WHOLE input: every
 * coefficient is pinned, so `Afix` is iteration-invariant and the Gram is a
 * pure reduction — no interpolation, no basis row, no parameter, and the
 * `coeff` it is valid at is the one the cache was filled from (the wrapper's
 * guard, not this kernel).
 *
 * ── the mapping ──────────────────────────────────────────────────────────
 *
 * A block owns a contiguous run of `ev` events.  It stages `GRAM_TE` events at
 * a time into SHARED as `float2 sA[te][N][P]` (a straight contiguous copy of
 * that slice of the cache) and each thread owns a FIXED partition of the UPPER
 * TRIANGLE `k <= j` — `ntri = N(N+1)/2` pairs, the row-major index
 * `idx = k(2N−k+1)/2 + (j−k)` inverted in closed form.  A thread accumulates
 * its elements in registers over the whole tile and does ONE shared
 * read-modify-write per element per tile, so:
 *
 *   * the inner accumulation needs no atomics at all — every element has
 *     exactly one owner thread in the block (that is why `sD` can be a plain
 *     `+=` and not a shared `atomicAdd`);
 *   * the shared traffic is `2P` float2 reads per element per event, instead
 *     of one global read per element per event.
 *
 * The accumulator is the WHOLE Gram in shared (`2·N²` doubles — 18.5 KB at
 * the CTPWA N = 34), so the block does ONE global `atomicAdd` per element at
 * the very end and nothing per event.
 *
 * `D[j,k] = conj(D[k,j])` is written from the SAME product, so the conjugate
 * symmetry is not a rounding accident (only the cross-block atomic order can
 * separate the two triangles, MEASURED 1.9e-16 relative) and the arithmetic is
 * halved.
 *
 * ── the re-association (accepted, and why) ───────────────────────────────
 *
 * Each block sums its own events in a fixed order and `atomicAdd`s its partial
 * into the global Gram, so the total is the sum of the per-block partials in
 * whatever order the blocks retire.  The result is therefore ORDER-DEPENDENT
 * at ~1e-15 relative: fp64 partials of fp32 cache entries, and the numpy
 * oracle sums event chunks in order, so the two differ by exactly that much
 * and no more.  That is the flat family's accepted floor (`s0g`'s fixed-wave
 * `dck` differs from `s0f`'s at 1.8e-8 for a milder re-association, and the
 * kernels agree with each other to 1e-16).  The Gram is built ONCE per pinned
 * `coeff`, so nothing per-iteration rides on it.
 *
 * `weight` is the per-event phase-space weight and may be NULL (1.0
 * everywhere): the wrapper uploads it per call and the handle's own copy is
 * NOT consulted.
 *===========================================================================*/
#define GPS0A_GRAM_TE    8        /* events staged in shared at a time */

/* The kernel's shared need, in DOUBLES — ONE definition, used by the launch
 * and by `cuda_gram_bytes_*` (a second copy could drift from the launch). */
__host__ __device__ __forceinline__ int gps0a_gram_doubles(int N, int P)
{
    return 2 * N * N + GPS0A_GRAM_TE * N * P;
}

/* ntri = N(N+1)/2 and the row-major upper-triangle index inverted in closed
 * form (row k starts at k(2N−k+1)/2); the two correction steps absorb the
 * sqrt's rounding and are each taken at most once in practice. */
__device__ __forceinline__ void gps0a_tri(int idx, int N, int* ok, int* oj)
{
    const double d = sqrt((double)((2 * N + 1) * (2 * N + 1) - 8 * idx));
    int k = (int)((2.0 * N + 1.0 - d) * 0.5);
    if (k < 0) k = 0;
    if (k > N - 1) k = N - 1;
    while (k > 0 && (k * (2 * N - k + 1)) / 2 > idx) k--;
    while (k < N - 1 && ((k + 1) * (2 * N - (k + 1) + 1)) / 2 <= idx) k++;
    *ok = k;
    *oj = k + (idx - (k * (2 * N - k + 1)) / 2);
}

__global__ void __launch_bounds__(GPS0A_BLOCK) gps0a_gram_kernel(
    const float2* __restrict__ afix,       /* [ne, N, P]  the whole cache     */
    const double* __restrict__ weight,     /* [ne] or NULL                    */
    int N, int P, int ne, int ev, int ntri,
    double* __restrict__ D)                /* [2·N·N] interleaved (re, im)    */
{
    const int tid = threadIdx.x, nt = blockDim.x;
    extern __shared__ double sh[];
    double* sD = sh;                                  /* 2·N·N            */
    float2* sA = (float2*)(sh + 2 * (size_t)N * (size_t)N);  /* TE·N·P      */

    const int e0 = (int)blockIdx.x * ev;
    if (e0 >= ne) return;
    const int evn = min(ev, ne - e0);

    for (int i = tid; i < 2 * N * N; i += nt) sD[i] = 0.0;
    __syncthreads();

    for (int t0 = 0; t0 < evn; t0 += GPS0A_GRAM_TE) {
        const int te = min(GPS0A_GRAM_TE, evn - t0);
        const float2* src = afix + (size_t)(e0 + t0) * (size_t)N * (size_t)P;
        const int stage = te * N * P;
        for (int i = tid; i < stage; i += nt) sA[i] = src[i];
        __syncthreads();
        /* element OUTER, event INNER: the accumulator is a register, so the
         * shared Gram is touched once per element per TILE */
        for (int idx = tid; idx < ntri; idx += nt) {
            int k, j;
            gps0a_tri(idx, N, &k, &j);
            double sr = 0.0, si = 0.0;
            for (int e = 0; e < te; e++) {
                const double w = (weight != NULL) ? weight[e0 + t0 + e] : 1.0;
                const float2* sa = sA + (size_t)e * (size_t)N * (size_t)P;
                double er = 0.0, ei = 0.0;
                for (int p = 0; p < P; p++) {
                    /* conj(a_k)·a_j — the conjugation is on the FIRST index */
                    const float2 ak = sa[k * P + p], aj = sa[j * P + p];
                    er += (double)ak.x * (double)aj.x
                          + (double)ak.y * (double)aj.y;
                    ei += (double)ak.x * (double)aj.y
                          - (double)ak.y * (double)aj.x;
                }
                sr += w * er;
                si += w * ei;
            }
            sD[2 * (k * N + j)] += sr;      /* ONE owner thread: no atomic */
            sD[2 * (k * N + j) + 1] += si;
        }
        __syncthreads();          /* the tile is re-staged by the next pass */
    }

    /* ── flush: ONE global atomic per element, and the mirror from the SAME
     *    product, so the conjugate symmetry is structural rather than a
     *    rounding accident ──────────────────────────────────────────────── */
    for (int idx = tid; idx < ntri; idx += nt) {
        int k, j;
        gps0a_tri(idx, N, &k, &j);
        const double sr = sD[2 * (k * N + j)];
        const double si = sD[2 * (k * N + j) + 1];
        if (sr != 0.0) atomicAdd(&D[2 * (k * N + j)], sr);
        if (si != 0.0) atomicAdd(&D[2 * (k * N + j) + 1], si);
        if (j != k) {
            if (sr != 0.0) atomicAdd(&D[2 * (j * N + k)], sr);
            if (si != 0.0) atomicAdd(&D[2 * (j * N + k) + 1], -si);
        }
    }
}

/* Events per block: enough blocks to fill the device (8 per SM) and a whole
 * number of staging tiles, so a block stages `ev`/GRAM_TE full tiles.  Small
 * samples fall back to one tile per block (correct, just fewer blocks). */
static int gps0a_gram_ev(int ne, int n_sm)
{
    const int nb = 8 * ((n_sm > 0) ? n_sm : 1);
    int ev = (ne + nb - 1) / nb;
    if (ev < GPS0A_GRAM_TE) ev = GPS0A_GRAM_TE;
    ev = ((ev + GPS0A_GRAM_TE - 1) / GPS0A_GRAM_TE) * GPS0A_GRAM_TE;
    return ev;
}

/* Opt the Gram kernel into its shared tile.  Past the 48 KiB default a launch
 * fails with "invalid argument" unless the FUNCTION carries the attribute, and
 * the attribute is per (function, DEVICE): one module can be launched from
 * several devices in one process, so the bookkeeping is a small per-device
 * table rather than one process-wide `static`.  A device that has not been
 * opted in for this size is opted in again (a single `static` would make the
 * SECOND device skip it and fail every Gram launch there), and the remembered
 * size is the largest opted in on that device, so a re-visit never LOWERS a
 * live context's limit.
 *
 * A shape that does not fit only disables the Gram (`gram_ok = 0`): the
 * pinned-cache compute path is never refused. */
#define GPS0A_MAX_DEV  16
static int gps0a_prep_dev[GPS0A_MAX_DEV] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
static int gps0a_prep_smem[GPS0A_MAX_DEV];

static int gps0a_gram_prepare(int smem)
{
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0) dev = 0;
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           dev);
    if (smem > optin) return 0;
    int slot = -1, spare = -1;
    for (int i = 0; i < GPS0A_MAX_DEV; i++) {
        if (gps0a_prep_dev[i] == dev) { slot = i; break; }
        if (spare < 0 && gps0a_prep_dev[i] < 0) spare = i;
    }
    if (slot < 0) slot = (spare >= 0) ? spare : 0;   /* >16 devices: reuse 0 */
    if (gps0a_prep_dev[slot] == dev && gps0a_prep_smem[slot] >= smem) return 1;
    cudaError_t e = cudaFuncSetAttribute(
        (const void*)gps0a_gram_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_ampcache: the Gram's %d B shared tile "
                "cannot be opted in on device %d (%s)\n",
                smem, dev, cudaGetErrorString(e));
        return 0;
    }
    gps0a_prep_dev[slot] = dev;
    gps0a_prep_smem[slot] = smem;
    return 1;
}

/*=============================================================================
 * The per-iteration tile.
 *
 * Path A (`per_blk <= te`): a block owns WHOLE groups and closes them in the
 * block — s0f's fused branch.  `rsize == 1` derives each event's scale
 * directly (S = q_e, W = w_e), which is the same closed form as the group
 * loop at rs = 1.
 *
 * Path B (`per_blk == rsize > te`): a block owns ONE group and walks it in
 * `te`-sized chunks — pass 1 accumulates the group totals, the close happens
 * once, pass 2 re-reads the cache with the finished constant and
 * back-propagates.  (This is where s0f's F/T/B fallback ends up; the cache
 * makes the second read a gather, not a `D` recompute.)
 *===========================================================================*/
__global__ void GPS0A_TILE_BOUNDS gps0a_tile_kernel(
    const float2* __restrict__ afix,       /* [ne, N, P]  the whole cache    */
    const double* __restrict__ ck_r, const double* __restrict__ ck_i,
    const double* __restrict__ weight, const double* __restrict__ bkg,
    double norm, int use_norm,
    int rsize, int per_blk, int te,
    int N, int P, int ne,
    double* __restrict__ P_dev,            /* [ne]  per-event P (output)    */
    double* __restrict__ g_ck_r, double* __restrict__ g_ck_i,
    double* __restrict__ Q_acc, double* __restrict__ dnorm_acc)
{
    const int tid = threadIdx.x, nt = blockDim.x;
    const GPS0ALayout L = gps0a_layout(te, P, N);

    extern __shared__ double sh[];
    double* s_A   = sh + L.A;
    double* s_pe  = sh + L.pe;
    double* s_q   = sh + L.q;
    double* s_sc  = sh + L.sc;
    double* s_fus = sh + L.fus;
    double* s_tot = sh + L.tot;

    const int e0 = (int)blockIdx.x * per_blk;
    const int evn = min(per_blk, ne - e0);
    if (evn <= 0) return;
    const int rs = (rsize < 1) ? 1 : rsize;
    const int pathB = (per_blk > te);
    const int nchunk = pathB ? ((evn + te - 1) / te) : 1;

    const int nsl = gps0a_nsl(N, nt);
    for (int i = tid; i < 2 * (N * nsl); i += nt) s_fus[i] = 0.0;

    if (pathB) {
        /* ── pass 1: A, P_e, and the group's totals, over the inner chunks ─ */
        if (tid == 0) { s_tot[0] = 0.0; s_tot[1] = 0.0; s_tot[2] = 0.0; }
        __syncthreads();
        for (int c = 0; c < nchunk; c++) {
            const int ce0 = e0 + c * te;
            const int ev = (evn - c * te > te) ? te : (evn - c * te);
            gps0a_build_A(afix, ck_r, ck_i, N, P, ce0, ev, s_A);
            __syncthreads();
            for (int e = tid; e < ev; e += nt) {
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
            __syncthreads();
            if (tid == 0) {
                /* one group per block: the totals are the group's */
                for (int e = 0; e < ev; e++) {
                    s_tot[0] += s_q[e];
                    s_tot[1] += weight[ce0 + e] * s_pe[e];
                    s_tot[2] += weight[ce0 + e];
                }
            }
            __syncthreads();          /* the tile is reused by the next chunk */
        }
        /* ── the close: the group's constant exists only now ─────────────── */
        if (tid == 0) {
            const double S = s_tot[0], WP = s_tot[1], W = s_tot[2];
            if (use_norm) {
                if (fabs(W) > GP_GROUP_EPS) {
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
        /* ── pass 2: dA with the finished constant, then dck ─────────────── */
        for (int c = 0; c < nchunk; c++) {
            const int ce0 = e0 + c * te;
            const int ev = (evn - c * te > te) ? te : (evn - c * te);
            gps0a_build_A(afix, ck_r, ck_i, N, P, ce0, ev, s_A);
            __syncthreads();
            for (int idx = tid; idx < ev * P; idx += nt) {
                const int e = idx / P;
                const double dqp = (use_norm ? weight[ce0 + e] / norm
                                             : weight[ce0 + e]) * cs;
                const double ar = s_A[idx * 2 + 0];
                const double ai = s_A[idx * 2 + 1];
                s_A[idx * 2 + 0] = dqp * ar;
                s_A[idx * 2 + 1] = -dqp * ai;
            }
            __syncthreads();
            gps0a_bwd(afix, s_A, N, P, ce0, ev, s_fus);
            __syncthreads();
        }
    } else {
        /* ── path A: whole groups, closed inside this block ──────────────── */
        gps0a_build_A(afix, ck_r, ck_i, N, P, e0, evn, s_A);
        __syncthreads();
        if (rs == 1) {
            /* per-event group: S = q_e, W = w_e — every thread derives its
               own scale from this thread's element, so no extra barrier */
            for (int e = tid; e < evn; e += nt) {
                double Pe = 0.0;
                for (int p = 0; p < P; p++) {
                    const double ar = s_A[(e * P + p) * 2 + 0];
                    const double ai = s_A[(e * P + p) * 2 + 1];
                    Pe += ar * ar + ai * ai;
                }
                const int ge = e0 + e;
                const double wv = weight[ge], bv = bkg[ge];
                const double q = use_norm ? wv * (Pe / norm + bv) : wv * Pe;
                s_pe[e] = Pe;
                s_q[e] = q;
                P_dev[ge] = Pe;
                if (use_norm) {
                    if (fabs(wv) > GP_GROUP_EPS) {
                        s_sc[e] = -wv / q;
                        atomicAdd(Q_acc, -wv * log(q / wv));
                        atomicAdd(dnorm_acc, wv * (wv * Pe)
                                  / (q * norm * norm));
                    } else {
                        s_sc[e] = 0.0;
                    }
                } else {
                    s_sc[e] = 1.0;
                    atomicAdd(Q_acc, q);
                }
            }
            __syncthreads();
            for (int idx = tid; idx < evn * P; idx += nt) {
                const int e = idx / P;
                const double dqp = (use_norm ? weight[e0 + e] / norm
                                             : weight[e0 + e]) * s_sc[e];
                const double ar = s_A[idx * 2 + 0];
                const double ai = s_A[idx * 2 + 1];
                s_A[idx * 2 + 0] = dqp * ar;
                s_A[idx * 2 + 1] = -dqp * ai;
            }
        } else {
            for (int e = tid; e < evn; e += nt) {
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
            /* group close: the block owns whole groups, so the sums are
               complete here */
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
                    if (fabs(W) > GP_GROUP_EPS) {              /* a zero-weight group is inert */
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
            __syncthreads();
            for (int idx = tid; idx < evn * P; idx += nt) {
                const int e = idx / P;
                const double dqp = (use_norm ? weight[e0 + e] / norm
                                             : weight[e0 + e]) * s_sc[e / rs];
                const double ar = s_A[idx * 2 + 0];
                const double ai = s_A[idx * 2 + 1];
                s_A[idx * 2 + 0] = dqp * ar;
                s_A[idx * 2 + 1] = -dqp * ai;
            }
        }
        __syncthreads();
        gps0a_bwd(afix, s_A, N, P, e0, evn, s_fus);
        __syncthreads();
    }

    /* ── flush: ONE global atomic per wave ──────────────────────────────── */
    for (int k = tid; k < N; k += nt) {
        double ur = 0.0, ui = 0.0;
        for (int sl = 0; sl < nsl; sl++) {
            ur += s_fus[(k * nsl + sl) * 2 + 0];
            ui += s_fus[(k * nsl + sl) * 2 + 1];
        }
        if (ur != 0.0) atomicAdd(g_ck_r + k, ur);
        if (ui != 0.0) atomicAdd(g_ck_i + k, ui);
    }
}

/*=============================================================================
 * Host side
 *===========================================================================*/
typedef struct {
    int *fl_q_index, *num_m_index, *bas_m_index, *angle_index;
    int *uslot_row, *uslot_col, *slot_of_wave, *rep_of_slot;
    int *row_start;                   /* [n_wave·n_res + 1] CSR rows      */
    long long *nz_pair;               /* [n_nz] packed (uslot << 32 | col) */
    double *nz_val;                   /* [n_nz] the gamma weight           */
    double *fl_table, *num_r, *num_i, *bas_r, *bas_i;
    double *angle_k, *angle_b, *am_r, *am_i;
    double m_min, m_delta, q_min, q_delta;
    int m_bins, q_bins;
    int n_wave, n_proj, n_res, n_decay, n_basis;
    int n_ab, n_comp, n_mass, n_momentum, n_angle_total;
    int n_uniq, n_uslot;
    int n_nz;                         /* CSR entries (every wave×factor)  */
    int n_row;                        /* CSR rows = n_wave·n_res          */
    int tiles_run;                    /* tile launches (ONE per chunk)       */
    int batch_size, rsize;
    /* chunk rule (one definition, in `create`) */
    int cap, s_ngroups;
    /* events per block chosen at create (a whole number of groups when
       rsize > 1; depends on rsize, n_proj and the wave count) */
    int te;
    int smem_bytes;
    /* parameter staging (refreshed per call, never malloc'd per call) */
    double *p_ck_r, *p_ck_i, *p_coeff;
    double *h_ck_r, *h_ck_i, *h_coeff;
    double *acc, *h_acc;              /* [Q, dnorm] */
    double *g_ck_r, *g_ck_i;
    /* the Gram reduction (see `gps0a_gram_kernel`): its scratch lives HERE so
       a build allocates nothing per call, and its weight buffer grows to the
       largest batch ever asked for (a streamed build asks for one batch) */
    int gram_ok;                      /* the shared tile fits this device  */
    int gram_smem;                    /* its size, bytes                   */
    int gram_ntri;                    /* N(N+1)/2 upper-triangle pairs     */
    int n_sm;                         /* SMs, for the events-per-block rule */
    double *d_gram;                   /* [2·N·N] device accumulator        */
    double *d_gram_w;                 /* [gram_w_cap] the uploaded weight  */
    int gram_w_cap;
    cudaStream_t stream;
} GPS0ACtx;

/* ONE SAMPLE.  Per-EVENT data only: the three caches, weight/bkg, and the
   caller-visible per-event P OUTPUT (not scratch — see the file header). */
typedef struct {
    double *weight, *bkg;
    float2 *us;           /* [ne, n_uslot]         Z = i·basis       */
    float2 *amp;          /* [ne, n_uniq, n_proj]  fa·fl·num          */
    float2 *afix;         /* [ne, n_wave, n_proj]  the whole cache    */
    double *P_dev;        /* [ne] per-event P (output of the call)    */
    double *h_P;          /* pinned staging for the one D2H           */
    int ne;
    int afix_ready;       /* the per-handle Afix fill has run         */
} GPS0AHandle;

static int* _gps0a_up_i(const int* src, int n) {
    if (n <= 0 || src == NULL) return NULL;
    int* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(int),
                          cudaMemcpyHostToDevice));
    return d;
}

static long long* _gps0a_up_l(const long long* src, int n) {
    if (n <= 0 || src == NULL) return NULL;
    long long* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(long long)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(long long),
                          cudaMemcpyHostToDevice));
    return d;
}

static double* _gps0a_up_d(const double* src, int n) {
    if (n <= 0 || src == NULL) return NULL;
    double* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(double),
                          cudaMemcpyHostToDevice));
    return d;
}

/* The chunk rule.  ONE definition, asked for through `cuda_cap_*`. */
static int gps0a_chunk(int batch_size, int rsize)
{
    int cap = (batch_size > 0) ? batch_size : 50000;
    if (cap > GPS0A_MAX_CAP) cap = GPS0A_MAX_CAP;
    if (rsize > 1) {
        const int a = (cap / rsize) * rsize;
        cap = (a > 0) ? a : rsize;   /* never 0, never unaligned to a group */
    }
    return cap;
}

/* The events-per-block rule.
 *
 * The tile holds A/dA (2·te·P) and the tiny group/partial arrays only — the
 * cache is read from global memory — so `te` never binds the shared budget at
 * any real shape.  MEASURED anyway (CTPWA 34-wave shape, 0.6 M events, one
 * `compute`, interleaved min-of-5, ns/event — two whole runs in opposite
 * orders agreeing to 0.3 %):
 *
 *     rsize 1,  te     32    48    64    96   128   256
 *               ns   7.05  7.22  6.97  7.09  7.19  8.49
 *
 *     rsize 17, te     68    85   102   136   170   (whole groups)
 *               ns   7.06  6.97  6.90  6.55  6.98
 *
 * So the optimum is ~64 events per block, and ~8 GROUPS when a group is
 * larger than that (a grouped tile pays one extra barrier per tile, which
 * ~128–136 events amortise).  A BIGGER tile is measurably WORSE (−22 % at
 * 256, rsize 1), which is the opposite of the no-cache s0g sibling: there the
 * tile recomputes the CSR and wants the block; here it streams a cache.
 * These are also the tiles s0f's own all-fixed ladder lands on (64 at
 * rsize 1, 136 at 17), so the two cache kernels are compared at equal work.
 *
 * The rule: `te = clamp(8·rsize, 64, GPS0A_MAX_TE)`, floored to a WHOLE
 * number of groups (a log-sum group never straddles two blocks) and shrunk
 * until the layout fits the 48 KiB budget.  `rsize > te` after that stays
 * legal: the block owns one group and walks it in `te`-sized inner chunks
 * (path B).
 */
static int gps0a_pick_te(int rsize, int P, int Nall)
{
    int te = 64;
    if (rsize > 1 && 8 * rsize > te) te = 8 * rsize;
    if (te > GPS0A_MAX_TE) te = GPS0A_MAX_TE;
    while (te > 1
           && (size_t)gps0a_layout(te, P, Nall).total * sizeof(double)
                  > (size_t)GPS0A_SMEM_BUDGET) {
        te -= 1;
    }
    if (te < 1) te = 1;
    /* the shrink steps by ONE event, which can land off a group boundary —
     * floor to a whole group (a smaller tile is by definition still inside
     * the budget, and `te < rsize` stays legal: path B) */
    if (rsize > 1 && te >= rsize) te = (te / rsize) * rsize;
    return te < 1 ? 1 : te;
}

extern "C" {

static void gps0a_fill_afix_launch(GPS0ACtx*, GPS0AHandle*);

void* cuda_create_context_flat_pwa_gen_ampcache(
    const double* fl_table, int n_fl,
    const int* fl_q_index, int n_fq,
    const double* num_r, const double* num_i, int n_num,
    const int* num_m_index, int n_nm,
    const double* bas_r, const double* bas_i, int n_bas,
    const int* bas_m_index, int n_bm,
    const int* uslot_row, int n_ur, const int* uslot_col, int n_uc,
    const int* slot_of_wave, int n_sw, const int* rep_of_slot, int n_rp,
    int n_uniq, int n_uslot,
    const long long* nz_pair, int n_np,
    const double* nz_val, int n_nv,
    const int* row_start, int n_rs,
    const int* angle_index, int n_ai,
    const double* angle_k, int n_ak, const double* angle_b, int n_ab_,
    const double* am_r, const double* am_i, int n_am,
    double m_min, double m_delta, int m_bins,
    double q_min, double q_delta, int q_bins,
    int n_wave, int n_proj, int n_res, int n_decay, int n_basis,
    int n_ab, int n_comp, int n_mass, int n_momentum, int n_angle_total,
    int batch_size, int rsize, int te_hint)
{
    /* A multi-BW chain (`n_res > 1`) is supported: the CSR is one row per
       `(wave, factor)` and the fill multiplies the row dots.  What is left to
       refuse is a CSR that does not describe that product — a row count that
       is not `n_wave·n_res + 1`, a weight array that is not one per non-zero,
       or a factor count below 1. */
    if (n_res < 1) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_ampcache: n_res must be >= 1, got %d\n",
                n_res);
        return NULL;
    }
    if (n_sw != n_wave || n_rp != n_uniq) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_ampcache: slot_of_wave/rep_of_slot length "
                "mismatch (%d vs n_wave %d, %d vs n_uniq %d)\n",
                n_sw, n_wave, n_rp, n_uniq);
        return NULL;
    }
    if (n_rs != n_wave * n_res + 1) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_ampcache: the CSR needs one row per "
                "(wave, factor) plus the terminator (%d, expected "
                "n_wave·n_res + 1 = %d)\n", n_rs, n_wave * n_res + 1);
        return NULL;
    }
    if (n_nv != n_np) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_ampcache: nz_val has %d entries, expected "
                "one per CSR non-zero (%d)\n", n_nv, n_np);
        return NULL;
    }
    if (n_rs > 0 && n_np != row_start[n_rs - 1]) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_ampcache: the CSR terminator is %d but "
                "there are %d non-zeros\n", row_start[n_rs - 1], n_np);
        return NULL;
    }
    if (n_uniq <= 0 || n_uslot <= 0 || n_wave <= 0 || n_basis <= 0) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_ampcache: empty shape (n_uniq %d, "
                "n_uslot %d, n_wave %d, n_basis %d)\n",
                n_uniq, n_uslot, n_wave, n_basis);
        return NULL;
    }
    GPS0ACtx* c = (GPS0ACtx*)calloc(1, sizeof(GPS0ACtx));
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
    c->n_nz = n_np;
    c->n_row = (n_rs > 0) ? (n_rs - 1) : 0;
    c->tiles_run = 0;
    c->rsize = (rsize > 0) ? rsize : 1;
    c->batch_size = batch_size;

    /* ── the chunk rule (see `gps0a_chunk`) ─────────────────────────────── */
    c->cap = gps0a_chunk(batch_size, c->rsize);
    c->s_ngroups = (c->cap + c->rsize - 1) / c->rsize;

    /* ── the tile size (see `gps0a_pick_te`) ──────────────────────────── */
    const int P = c->n_proj;
    c->te = (te_hint > 0) ? te_hint : gps0a_pick_te(c->rsize, P, n_wave);
    /* a hint must still hold whole groups (never straddle a group boundary),
       and it must fit the device: the hint is a test/benchmark hook, and a
       silently wrong tile is exactly the failure this guards against */
    if (c->rsize > 1 && c->te >= c->rsize && (c->te % c->rsize) != 0) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_ampcache: te %d is not a multiple of rsize "
                "%d — a tile would split a log-sum group\n",
                c->te, c->rsize);
        free(c);
        return NULL;
    }
    if (c->te < 1) c->te = 1;
    {
        GPS0ALayout lay = gps0a_layout(c->te, P, n_wave);
        c->smem_bytes = (int)(lay.total * (long long)sizeof(double));
        int optin = 0;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               0);
        if (c->smem_bytes > optin) {
            fprintf(stderr,
                    "cuda_flat_pwa_gen_ampcache: shared tile %d B exceeds the "
                    "opt-in limit %d B (n_wave %d, n_proj %d, te %d)\n",
                    c->smem_bytes, optin, n_wave, c->n_proj, c->te);
            free(c);
            return NULL;
        }
        /* Past the 48 KiB DEFAULT the launch needs the opt-in size on the
           FUNCTION, not merely a size check against the device: without it
           the launch fails with "invalid argument" and — because CUDA_CHECK
           only PRINTS — compute() would silently return Q = 0.

           The attribute is per FUNCTION, so two live contexts of different
           tile sizes would fight; raise-only monotonic makes it the
           process-wide maximum, which every context is <= by construction. */
        static size_t opt_in_max = 0;
        if ((size_t)c->smem_bytes > opt_in_max) {
            cudaError_t e = cudaFuncSetAttribute(
                (const void*)gps0a_tile_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, c->smem_bytes);
            if (e != cudaSuccess) {
                fprintf(stderr,
                        "cuda_flat_pwa_gen_ampcache: cannot opt in to %d B "
                        "shared memory (%s); n_wave %d, te %d\n",
                        c->smem_bytes, cudaGetErrorString(e), n_wave, c->te);
                free(c);
                return NULL;
            }
            opt_in_max = (size_t)c->smem_bytes;
        }
    }

    c->fl_table = _gps0a_up_d(fl_table, n_fl);
    c->fl_q_index = _gps0a_up_i(fl_q_index, n_fq);
    c->num_r = _gps0a_up_d(num_r, n_num);
    c->num_i = _gps0a_up_d(num_i, n_num);
    c->num_m_index = _gps0a_up_i(num_m_index, n_nm);
    c->bas_r = _gps0a_up_d(bas_r, n_bas);
    c->bas_i = _gps0a_up_d(bas_i, n_bas);
    c->bas_m_index = _gps0a_up_i(bas_m_index, n_bm);
    c->uslot_row = _gps0a_up_i(uslot_row, n_ur);
    c->uslot_col = _gps0a_up_i(uslot_col, n_uc);
    c->slot_of_wave = _gps0a_up_i(slot_of_wave, n_sw);
    c->rep_of_slot = _gps0a_up_i(rep_of_slot, n_rp);
    c->nz_pair = _gps0a_up_l(nz_pair, n_np);
    c->nz_val = _gps0a_up_d(nz_val, n_nv);
    c->row_start = _gps0a_up_i(row_start, n_rs);
    c->angle_index = _gps0a_up_i(angle_index, n_ai);
    c->angle_k = _gps0a_up_d(angle_k, n_ak);
    c->angle_b = _gps0a_up_d(angle_b, n_ab_);
    c->am_r = _gps0a_up_d(am_r, n_am);
    c->am_i = _gps0a_up_d(am_i, n_am);
    c->m_min = m_min; c->m_delta = m_delta; c->m_bins = m_bins;
    c->q_min = q_min; c->q_delta = q_delta; c->q_bins = q_bins;

    CUDA_CHECK(cudaStreamCreate(&c->stream));
    CUDA_CHECK(cudaMalloc(&c->p_ck_r, (size_t)n_wave * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->p_ck_i, (size_t)n_wave * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->p_coeff, (size_t)n_basis * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->g_ck_r, (size_t)n_wave * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->g_ck_i, (size_t)n_wave * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&c->h_ck_r, (size_t)n_wave * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&c->h_ck_i, (size_t)n_wave * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&c->h_coeff, (size_t)n_basis * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->acc, 2 * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&c->h_acc, 2 * sizeof(double)));

    /* ── the Gram (see `gps0a_gram_kernel`) ──────────────────────────────
     * Its shared tile and its N×N device accumulator are sized HERE, once per
     * context, so a build allocates nothing.  A shape whose tile does not fit
     * the device only disables the Gram (`gram_ok = 0`): the pinned-cache
     * compute path has no dependency on it. */
    c->gram_ntri = n_wave * (n_wave + 1) / 2;
    c->gram_smem = (int)((long long)gps0a_gram_doubles(n_wave, c->n_proj)
                         * (long long)sizeof(double));
    cudaDeviceGetAttribute(&c->n_sm, cudaDevAttrMultiProcessorCount, 0);
    c->gram_ok = gps0a_gram_prepare(c->gram_smem);
    if (c->gram_ok) {
        if (cudaMalloc(&c->d_gram, (size_t)(2 * n_wave * n_wave)
                                  * sizeof(double)) != cudaSuccess) {
            c->d_gram = NULL;
            c->gram_ok = 0;
        }
    }
    return c;
}

void cuda_free_context_flat_pwa_gen_ampcache(void* vctx) {
    GPS0ACtx* c = (GPS0ACtx*)vctx;
    if (!c) return;
    #define F(p) if (c->p) cudaFree((void*)c->p)
    #define FH(p) if (c->p) cudaFreeHost(c->p)
    F(fl_table); F(fl_q_index); F(num_r); F(num_i); F(num_m_index);
    F(bas_r); F(bas_i); F(bas_m_index);
    F(uslot_row); F(uslot_col); F(slot_of_wave); F(rep_of_slot);
    F(nz_pair); F(nz_val); F(row_start);
    F(angle_index); F(angle_k); F(angle_b); F(am_r); F(am_i);
    F(p_ck_r); F(p_ck_i); F(p_coeff);
    F(g_ck_r); F(g_ck_i);
    F(acc);
    F(d_gram); F(d_gram_w);
    FH(h_ck_r); FH(h_ck_i); FH(h_coeff); FH(h_acc);
    #undef F
    #undef FH
    if (c->stream) cudaStreamDestroy(c->stream);
    free(c);
}

void* cuda_load_data_flat_pwa_gen_ampcache(
    void* vctx, const double* mass, const double* mom, const double* ang,
    const double* weight, const double* bkg, int ne)
{
    GPS0ACtx* c = (GPS0ACtx*)vctx;
    if (!c) return NULL;
    GPS0AHandle* h = (GPS0AHandle*)calloc(1, sizeof(GPS0AHandle));
    if (!h) return NULL;
    h->ne = ne;
    /* An EMPTY sample is a legal load (a fitter that has not been given data
       yet, or a zero-row selection): every buffer is simply absent, and
       `compute` must then return the identity (Q = 0, zero gradients) rather
       than failing.  A zero-byte cudaMalloc is an error, so the whole body is
       guarded rather than leaving a NULL-pointer trap for `compute`.

       `afix_ready = 1` on the way out: there is nothing to fill, and
       `compute` refuses an unfilled handle — an empty sample must take the
       NORMAL identity path, not that stop. */
    if (ne <= 0) { h->afix_ready = 1; return h; }
    h->weight = _gps0a_up_d(weight, ne);
    h->bkg = _gps0a_up_d(bkg, ne);
    if (cudaMalloc(&h->us, (size_t)ne * c->n_uslot * sizeof(float2))
            != cudaSuccess) goto fail;
    if (cudaMalloc(&h->amp, (size_t)ne * c->n_uniq * c->n_proj
                           * sizeof(float2)) != cudaSuccess) goto fail;
    /* THE cache: 8·P·n_wave bytes per event, this handle's own */
    if (cudaMalloc(&h->afix, (size_t)ne * c->n_wave * c->n_proj
                             * sizeof(float2)) != cudaSuccess) goto fail;
    if (cudaMalloc(&h->P_dev, (size_t)ne * sizeof(double)) != cudaSuccess)
        goto fail;
    CUDA_CHECK(cudaMallocHost(&h->h_P, (size_t)ne * sizeof(double)));

    /* ── the two load-time caches ─────────────────────────────────────────
     *
     * The kinematics are consumed ONLY here, so they are uploaded as transient
     * device copies and released immediately afterwards — exactly what
     * gen2/1bw/s0f do.  They are NOT read through the host pointers: a kernel
     * dereferencing a pageable host pointer happens to work on a device with
     * host page-table access but is not portable, and on a device without it
     * the fill would silently read garbage. */
    {
        double* t_mass = _gps0a_up_d(mass, ne * c->n_mass);
        double* t_mom = _gps0a_up_d(mom, ne * c->n_momentum);
        double* t_ang = _gps0a_up_d(ang, ne * c->n_angle_total * c->n_comp);
        if (!t_mass || !t_mom || !t_ang) {
            if (t_mass) cudaFree(t_mass);
            if (t_mom) cudaFree(t_mom);
            if (t_ang) cudaFree(t_ang);
            goto fail;
        }
        {
            const long long total = (long long)ne * c->n_uslot;
            const int threads = GPS0A_BLOCK;
            long long blocks = (total + threads - 1) / threads;
            if (blocks > 65535) blocks = 65535;
            if (blocks < 1) blocks = 1;
            gps0a_fill_us<<<(unsigned)blocks, threads, 0, c->stream>>>(
                c->bas_r, c->bas_i, c->uslot_row, c->uslot_col,
                c->m_min, c->m_delta, c->m_bins,
                t_mass, c->n_mass, ne, c->n_uslot, h->us);
            CUDA_CHECK(cudaGetLastError());
        }
        {
            int fb = c->n_uniq < 32 ? 32 : (c->n_uniq < GPS0A_BLOCK
                                            ? GPS0A_BLOCK : c->n_uniq);
            if (fb > 1024) fb = 1024;
            const size_t shm = (size_t)c->n_ab * sizeof(double);
            gps0a_fill_amp<<<ne, fb, shm, c->stream>>>(
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
        CUDA_CHECK(cudaStreamSynchronize(c->stream));
        cudaFree(t_mass); cudaFree(t_mom); cudaFree(t_ang);
    }
    /* NOTE: `afix` is NOT filled here even though the kinematics are at hand.
       The fill needs the COEFFICIENTS, which this entry point does not take,
       and the lifecycle is ONE: the wrapper's `_fill_once` calls
       `cuda_fill_afix_…` on the handle's FIRST `compute` (or Gram) and guards
       every later call.  A load never fills. */
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

/* THE cache fill: once per handle, from a `coeff` vector.  `compute` never
 * reaches this — the wrapper is the only caller, and it raises rather than
 * re-filling.  Returns 1 on success. */
int cuda_fill_afix_flat_pwa_gen_ampcache(void* vctx, void* vh,
                                         const double* coeff)
{
    GPS0ACtx* c = (GPS0ACtx*)vctx;
    GPS0AHandle* h = (GPS0AHandle*)vh;
    if (!c || !h || !coeff) return 0;
    if (h->ne <= 0) { h->afix_ready = 1; return 1; }
    if (!h->afix) return 0;
    memcpy(c->h_coeff, coeff, (size_t)c->n_basis * sizeof(double));
    cudaMemcpyAsync(c->p_coeff, c->h_coeff, (size_t)c->n_basis * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    gps0a_fill_afix_launch(c, h);
    cudaError_t e = cudaStreamSynchronize(c->stream);
    if (e != cudaSuccess) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_ampcache: the Afix fill failed (%s)\n",
                cudaGetErrorString(e));
        return 0;
    }
    h->afix_ready = 1;
    return 1;
}

void cuda_free_data_flat_pwa_gen_ampcache(void* vh) {
    GPS0AHandle* h = (GPS0AHandle*)vh;
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

/* The fill launch: one thread per (event, wave) over the WHOLE handle. */
static void gps0a_fill_afix_launch(GPS0ACtx* c, GPS0AHandle* h)
{
    if (!c || !h || !h->afix || h->ne <= 0) return;
    const long long total = (long long)h->ne * (long long)c->n_wave;
    long long blocks = (total + GPS0A_BLOCK - 1) / GPS0A_BLOCK;
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;
    gps0a_fill_afix<<<(unsigned)blocks, GPS0A_BLOCK, 0, c->stream>>>(
        h->us, h->amp, c->slot_of_wave, c->nz_pair, c->nz_val, c->row_start,
        c->p_coeff, c->n_wave, c->n_res, h->ne, c->n_uslot, c->n_uniq,
        c->n_proj, h->afix);
    CUDA_CHECK(cudaGetLastError());
}

/* One launch per chunk.  Batching needs no kernel logic: the chunk is
   expressed as POINTER OFFSETS into the per-event cache plus the chunk length
   as the kernel's `ne`, so `e0` is chunk-local by construction. */
void cuda_compute_flat_pwa_gen_ampcache(
    void* vctx, void* vh, const double* ck_r, const double* ck_i,
    double norm, int use_norm,
    double* oQ, double* oDn, double* oP,
    double* og_ck_r, double* og_ck_i)
{
    GPS0ACtx* c = (GPS0ACtx*)vctx;
    GPS0AHandle* h = (GPS0AHandle*)vh;
    if (!c || !h) return;
    const int N = c->n_wave;

    /* the cache is filled by the wrapper's `_fill_once` on this handle's
       FIRST compute (or Gram), never by a load.  `cuda_compute_…` itself
       never fills: silently computing from an uninitialised cache would be
       worse than stopping, so an unfilled handle is a hard (printed)
       failure. */
    if (!h->afix_ready) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_ampcache: handle has no Afix cache (ne %d)"
                " — the coefficients must be pinned before it is computed; "
                "refusing to compute\n", h->ne);
        return;
    }

    memcpy(c->h_ck_r, ck_r, (size_t)N * sizeof(double));
    memcpy(c->h_ck_i, ck_i, (size_t)N * sizeof(double));
    cudaMemcpyAsync(c->p_ck_r, c->h_ck_r, (size_t)N * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    cudaMemcpyAsync(c->p_ck_i, c->h_ck_i, (size_t)N * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    /* the accumulators span the WHOLE sample: zero once, read once */
    cudaMemsetAsync(c->acc, 0, 2 * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_ck_r, 0, (size_t)N * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_ck_i, 0, (size_t)N * sizeof(double), c->stream);

    const int ne_total = h->ne;
    const int grouped = (use_norm && c->rsize > 1);
    const int bs = (c->cap > 0) ? c->cap : 1;
    for (int st = 0; st < ne_total; st += bs) {
        const int nb = (ne_total - st > bs) ? bs : (ne_total - st);
        /* the tile's own per-block split: whole groups when they fit the
           tile, one group walked internally when they do not */
        int per_blk;
        if (!grouped) {
            per_blk = c->te;
        } else if (c->rsize <= c->te) {
            per_blk = c->rsize * (c->te / c->rsize);
        } else {
            per_blk = c->rsize;
        }
        if (per_blk < 1) per_blk = 1;
        const int tiles = (nb + per_blk - 1) / per_blk;
        gps0a_tile_kernel<<<tiles, GPS0A_BLOCK, (size_t)c->smem_bytes,
                            c->stream>>>(
            h->afix + (size_t)st * c->n_wave * c->n_proj,
            c->p_ck_r, c->p_ck_i,
            h->weight + st, h->bkg + st,
            norm, use_norm, c->rsize, per_blk, c->te,
            c->n_wave, c->n_proj, nb,
            h->P_dev + st,
            c->g_ck_r, c->g_ck_i, c->acc, c->acc + 1);
        CUDA_CHECK(cudaGetLastError());
        c->tiles_run++;
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
    CUDA_CHECK(cudaStreamSynchronize(c->stream));
    *oQ = c->h_acc[0];
    *oDn = c->h_acc[1];
    if (oP != NULL && ne_total > 0)
        memcpy(oP, h->h_P, (size_t)ne_total * sizeof(double));
}

/* ── reports: every one of these has exactly ONE definition, in C ─────── */

/* THE GRAM — see the `gps0a_gram_kernel` header for the mapping.  `weight`
 * is a HOST pointer to `ne` per-event weights (NULL = 1.0 everywhere); it is
 * uploaded per call into the context's own buffer and the handle's copy is
 * never consulted, so a caller-supplied vector really is the one used.
 *
 * `out` is `2·N·N` doubles, row-major `(k·N + j)·2 + {0 = re, 1 = im}` — i.e.
 * exactly a C-contiguous `complex128 (N, N)` array, which is why the wrapper
 * hands its output buffer straight through.
 *
 * Returns 1 on success, 0 when the Gram is unavailable for this shape or the
 * launch failed (the wrapper raises; zeroing silently would be worse).  Ask
 * `cuda_gram_ok_flat_pwa_gen_ampcache` FIRST when a fallback exists — that is
 * the shape check, so a caller does not pay a cache fill to learn it. */
int cuda_gram_flat_pwa_gen_ampcache(void* vctx, void* vh,
                                    const double* weight, double* out)
{
    GPS0ACtx* c = (GPS0ACtx*)vctx;
    GPS0AHandle* h = (GPS0AHandle*)vh;
    if (!c || !h || !out) return 0;
    const int N = c->n_wave;
    const size_t gsz = (size_t)(2 * N * N) * sizeof(double);
    if (!c->gram_ok || !c->d_gram) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_ampcache: no GPU Gram for this shape "
                "(n_wave %d, n_proj %d needs %d B of shared memory)\n",
                N, c->n_proj, c->gram_smem);
        return 0;
    }
    if (h->ne <= 0) {                    /* an empty sample: the zero Gram */
        memset(out, 0, gsz);
        return 1;
    }
    if (!h->afix_ready) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_ampcache: handle has no Afix cache (ne %d)"
                " — the Gram needs the pinned coefficients, i.e. a filled "
                "handle\n", h->ne);
        return 0;
    }
    if (weight != NULL) {
        if (h->ne > c->gram_w_cap) {
            if (c->d_gram_w) { cudaFree(c->d_gram_w); c->d_gram_w = NULL; }
            if (cudaMalloc(&c->d_gram_w, (size_t)h->ne * sizeof(double))
                    != cudaSuccess) {
                c->gram_w_cap = 0;
                return 0;
            }
            c->gram_w_cap = h->ne;
        }
        cudaMemcpyAsync(c->d_gram_w, weight, (size_t)h->ne * sizeof(double),
                        cudaMemcpyHostToDevice, c->stream);
    }
    cudaMemsetAsync(c->d_gram, 0, gsz, c->stream);
    const int ev = gps0a_gram_ev(h->ne, c->n_sm);
    const int blocks = (h->ne + ev - 1) / ev;
    gps0a_gram_kernel<<<blocks, GPS0A_BLOCK, (size_t)c->gram_smem, c->stream>>>(
        h->afix, (weight != NULL) ? c->d_gram_w : NULL,
        N, c->n_proj, h->ne, ev, c->gram_ntri, c->d_gram);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "cuda_flat_pwa_gen_ampcache: the Gram launch failed "
                "(%s)\n", cudaGetErrorString(e));
        return 0;
    }
    cudaMemcpyAsync(out, c->d_gram, gsz, cudaMemcpyDeviceToHost, c->stream);
    e = cudaStreamSynchronize(c->stream);
    if (e != cudaSuccess) {
        fprintf(stderr, "cuda_flat_pwa_gen_ampcache: the Gram copy failed "
                "(%s)\n", cudaGetErrorString(e));
        return 0;
    }
    return 1;
}

/* The Gram kernel's shared tile for an arbitrary shape (the same
 * `gps0a_gram_doubles` the launch uses), and the events-per-block rule — so a
 * test can assert both without a second, drifting copy in Python. */
long long cuda_gram_bytes_flat_pwa_gen_ampcache(int n_wave, int n_proj)
{
    return (long long)gps0a_gram_doubles(n_wave, n_proj > 0 ? n_proj : 1)
           * (long long)sizeof(double);
}

/* 1 when THIS context's shape can build a Gram at all — its shared tile fits
 * the device's opt-in limit and the N×N accumulator was allocated.  A caller
 * that HAS a fallback (`integrated_pwa_gen`'s numpy builder) asks this BEFORE
 * paying a cache fill, instead of discovering the refusal from a failed call:
 * `cuda_gram_flat_pwa_gen_ampcache` returns 0 for every shape above the
 * ceiling (2·N² + 8·N·P doubles past `cudaDevAttrMaxSharedMemoryPerBlockOptin`
 * — N=34/2 is 22.8 KB and fits a 101 KB box; N=76 already does not). */
int cuda_gram_ok_flat_pwa_gen_ampcache(void* vctx)
{
    GPS0ACtx* c = (GPS0ACtx*)vctx;
    return (c != NULL && c->gram_ok && c->d_gram != NULL) ? 1 : 0;
}

void cuda_gram_plan_flat_pwa_gen_ampcache(int ne, int* out)
{
    if (!out) return;
    int n_sm = 0;
    cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, 0);
    const int ev = gps0a_gram_ev(ne, n_sm);
    out[0] = ev;
    out[1] = (ne > 0) ? (ne + ev - 1) / ev : 0;
    out[2] = n_sm;
}

/* The tile size `gps0a_layout` gives for an arbitrary shape — the same
 * function the kernel and `create` use, so a test can assert the buffer
 * arithmetic without a second, drifting copy in Python. */
long long cuda_tile_bytes_flat_pwa_gen_ampcache(int te, int n_wave, int n_proj)
{
    GPS0ALayout L = gps0a_layout(te, n_proj, n_wave);
    return (long long)L.total * (long long)sizeof(double);
}

/* THE cache's size for a sample of `ne` events, in B: 8·P·n_wave·ne. */
long long cuda_cache_bytes_flat_pwa_gen_ampcache(void* vctx, int ne) {
    GPS0ACtx* c = (GPS0ACtx*)vctx;
    if (!c || ne <= 0) return 0;
    return (long long)ne * c->n_wave * c->n_proj * (long long)sizeof(float2);
}

/* 1 when this handle's `Afix` cache has been filled. */
int cuda_handle_afix_ready_flat_pwa_gen_ampcache(void* vh) {
    GPS0AHandle* h = (GPS0AHandle*)vh;
    return (h != NULL && h->afix_ready) ? 1 : 0;
}

int cuda_tile_count_flat_pwa_gen_ampcache(void* vctx) {
    GPS0ACtx* c = (GPS0ACtx*)vctx;
    return c ? c->tiles_run : 0;
}

int cuda_smem_bytes_flat_pwa_gen_ampcache(void* vctx) {
    GPS0ACtx* c = (GPS0ACtx*)vctx;
    return c ? c->smem_bytes : 0;
}

int cuda_cap_flat_pwa_gen_ampcache(void* vctx) {
    GPS0ACtx* c = (GPS0ACtx*)vctx;
    return c ? c->cap : 0;
}

int cuda_te_flat_pwa_gen_ampcache(void* vctx) {
    GPS0ACtx* c = (GPS0ACtx*)vctx;
    return c ? c->te : 0;
}

int cuda_ngroups_flat_pwa_gen_ampcache(void* vctx) {
    GPS0ACtx* c = (GPS0ACtx*)vctx;
    return c ? c->s_ngroups : 0;
}

/* The chunk plan for a sample of `ne` events: out[0] = cap, out[1] = the
 * number of chunks, out[2] = 1 when every chunk start and every non-final
 * chunk length is a multiple of `rsize` (a group never straddles a chunk). */
void cuda_chunk_plan_flat_pwa_gen_ampcache(void* vctx, int ne, int* out) {
    GPS0ACtx* c = (GPS0ACtx*)vctx;
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

/* The events-per-block plan for a sample of `ne`: out[0] = te, out[1] = the
 * per-block event count at this chunk's head, out[2] = 1 when a block walks
 * its group internally (path B: `rsize > te`). */
void cuda_block_plan_flat_pwa_gen_ampcache(void* vctx, int* out) {
    GPS0ACtx* c = (GPS0ACtx*)vctx;
    if (!out) return;
    if (!c) { out[0] = 0; out[1] = 0; out[2] = 0; return; }
    int per_blk;
    if (!(c->rsize > 1)) per_blk = c->te;
    else if (c->rsize <= c->te) per_blk = c->rsize * (c->te / c->rsize);
    else per_blk = c->rsize;
    out[0] = c->te;
    out[1] = per_blk;
    out[2] = (per_blk > c->te) ? 1 : 0;
}

/* Runtime resource report (host side, no kernel launch).
 *   out[0] registers/thread          (cudaFuncGetAttributes.numRegs)
 *   out[1] local bytes > 0  => the kernel SPILLS
 *   out[2] blocks/SM at the REAL block size and dynamic smem
 *   out[3] warps/SM  (= blocks x GPS0A_BLOCK/32)
 *   out[4] maxThreadsPerBlock
 *   out[5] static shared bytes
 */
extern "C" void cuda_stats_flat_pwa_gen_ampcache(int smem, int* out)
{
    if (!out) return;
    cudaFuncAttributes a;
    cudaFuncGetAttributes(&a, (const void*)gps0a_tile_kernel);
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, (const void*)gps0a_tile_kernel, GPS0A_BLOCK,
        (size_t)(smem > 0 ? smem : 0));
    out[0] = a.numRegs;
    out[1] = (int)a.localSizeBytes;
    out[2] = blocks;
    out[3] = blocks * (GPS0A_BLOCK / 32);
    out[4] = a.maxThreadsPerBlock;
    out[5] = (int)a.sharedSizeBytes;
}

/* The same report for the LOAD-TIME `Afix` fill (it carries the factor-axis
 * row loop, so its register pressure is its own). */
extern "C" void cuda_fill_stats_flat_pwa_gen_ampcache(int* out)
{
    if (!out) return;
    cudaFuncAttributes a;
    cudaFuncGetAttributes(&a, (const void*)gps0a_fill_afix);
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, (const void*)gps0a_fill_afix, GPS0A_BLOCK, 0);
    out[0] = a.numRegs;
    out[1] = (int)a.localSizeBytes;
    out[2] = blocks;
    out[3] = blocks * (GPS0A_BLOCK / 32);
    out[4] = a.maxThreadsPerBlock;
    out[5] = (int)a.sharedSizeBytes;
}

/* ... and for the GRAM reduction.  It runs at `GPS0A_BLOCK` threads with the
 * shape's own shared tile, so the occupancy report is the real one — a
 * non-zero `local_bytes` here means the accumulator spilled, which is the one
 * way this kernel can quietly lose its whole point. */
extern "C" void cuda_gram_stats_flat_pwa_gen_ampcache(int smem, int* out)
{
    if (!out) return;
    cudaFuncAttributes a;
    cudaFuncGetAttributes(&a, (const void*)gps0a_gram_kernel);
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, (const void*)gps0a_gram_kernel, GPS0A_BLOCK,
        (size_t)(smem > 0 ? smem : 0));
    out[0] = a.numRegs;
    out[1] = (int)a.localSizeBytes;
    out[2] = blocks;
    out[3] = blocks * (GPS0A_BLOCK / 32);
    out[4] = a.maxThreadsPerBlock;
    out[5] = (int)a.sharedSizeBytes;
}

int cuda_get_device_count_flat_pwa_gen_ampcache(void) {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
}

void cuda_get_device_name_flat_pwa_gen_ampcache(char* buf, int len) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    snprintf(buf, len, "%s", p.name);
}

}  /* extern "C" */
