/**
 * cuda_flat_pwa_gen_1bw — the flat `pwa_gen` tile with ONE BW PER CHAIN.
 *
 * Case-1 copy (kernel policy §1): own `.cu`, own `.so`, own wrapper, own
 * symbols.  It is a copy of `kernels_flat_pwa_gen2.cu` — that file and its
 * wrapper stay byte-identical — specialised to `n_res == 1` and extended with
 * the one genuinely new optimisation the specialisation buys, **contiguous
 * gamma runs**.  The wrapper enforces `n_res == 1` (a multi-BW chain raises
 * with the general backend named), exactly as `cuda_v4_pwa_1bw` does.
 *
 * What the specialisation removes, relative to `cuda_flat_pwa_gen2`:
 *
 * 1. **The `template<int NR>` fork.**  `NR == 1` was the template's *special*
 *    instantiation; here it is the only code path.  `D = Π_l s_l` has one
 *    factor, so `D` IS that factor: the `s_s` buffer never exists, the `l`
 *    loop disappears, `dQ/ds_{e,k,0}` IS `dQ/dD_{e,k}`, and the per-factor
 *    product bookkeeping is the identity.
 *
 * 2. **The `Π_{l'≠l} s` hoist inside the slot collapse (`s_w`) is dropped —
 *    that part IS dead at one factor.**  It is the empty product, identically
 *    1, so `s_w` has no content and no per-`(e,slot,l)` loop has anything to
 *    compute.  gen2 already allocated none of it at its `NR == 1`
 *    instantiation.
 *
 *    **The collapse itself is KEPT, because it is not dead.**  The other half
 *    of it — `dsum_{e,slot} = Σ_{k∈slot} dQ/dD_{e,k}` — is what reduces the
 *    ``dQ/dcoeff`` dot products from ``n_wave·run`` to ``n_slot·run``, and
 *    that reduction is real at one factor: measured on the benchmark shape
 *    (n_wave 100, one slot) the collapse is worth **1.16×** (5.36 -> 4.64 ms),
 *    i.e. dropping it costs more than the whole `s_s` saving bought.  So the
 *    gate, the CSR wave list and the `s_dd` buffer stay; ``s_w`` /
 *    ``slot_rep[slot]``-as-a-product-source / `gamma_slot` do not — the slot's
 *    gamma row is read through its representative wave (`slot_rep[sl]`), which
 *    is bit-identical to it by construction (that equality is *why* the slot
 *    exists), so a separate uploaded copy would only be a second source of the
 *    same bits.  The deep-factor (`n_res > MAXRES`) fallback is gone too: one
 *    factor can never reach it.
 *
 * 3. **Sparse gamma rows — the new win of this kernel.**  `gamma[k, 0, :]` is
 *    not a dense selector: a factor selects a handful of `basis_table` rows
 *    out of `n_basis`.  The host emits the row's **true non-zeros** as a CSR
 *    (`gs_lo[k]`/`gs_hi[k]` index the flat `nz_wave_m` list) at CREATE time,
 *    so phase 1 iterates exactly `nnz(k)` columns instead of `n_basis`, and
 *    phase 5 walks a host-built `(row, m)` list of the same entries.
 *
 *    The run does NOT have to be contiguous, and the kernel has no `g == 0`
 *    test in phase 1 for that reason: the list contains no zeros.  (An earlier
 *    form required contiguity and refused a row with holes, claiming a hole
 *    "would make it use basis columns the factor does not select" — false,
 *    since `0 * value == 0`; it only forfeited the optimisation on any config
 *    whose basis is folded: 290 vs 92 iterations on the real one.)
 *
 *    Measured per config (basis rows iterated per event, `ref` = full-`m`
 *    loop, `run` = this kernel — never a loss, `run <= n_basis` always):
 *
 *        config        NB  n_res   ref  run  ratio
 *        pwa_gen.yml    3    1       3    3   1.00   (dense)
 *        pwa_gen_km.yml 5    1      10   10   1.00   (dense)
 *        _ident.yml    10    1      40   20   2.00
 *        _same.yml     10    1      40   20   2.00
 *        _cp.yml       20    1     160   40   4.00
 *        _mb.yml        8    2      64   32   2.00   (rejected here: n_res=2)
 *
 *    The gain is exactly where the matrix is sparse, which is why `_cp` (20
 *    basis rows over 4 waves of 5) is the one that shows it.  `num_table`
 *    needs no such treatment: its per-use loop is over `r` in `[0, n_res)`,
 *    i.e. ONE iteration here, and it runs once per `load_data`, not per
 *    iteration.
 *
 * Kept from gen2 (unchanged in substance): the `u`-merge of `dQ/dck` and
 * `dQ/dcommon`, the `1/|D|²` real-reciprocal cache, the grouped
 * `resolution_size` path (mode 1 forward / mode 2 backward with the saved
 * `A`), the raise-only-monotonic `cudaFuncSetAttribute` opt-in, and the
 * context-reported tile size (the host must never mirror the formula).
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
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

#define GP1B_BLOCK 256
#define GP1B_TE    8            /* events per tile */
/*: deepest factor list the fully parallel coeff-gradient pass handles; past
 *  this the kernel falls back to the per-wave shape (still correct, just not
 *  spread over the whole block), and the slot collapse is forced off (the
 *  fallback has no `dsum` buffer).  At n_res == 1 only one factor exists, so
 *  the fallback is unreachable from the wrapper — it is kept because the pass
 *  is a straight copy of gen2's and removing the guard would make the code
 *  silently wrong if the factor count were ever widened again. */
#define GP1B_MAXRES 16
/*: projections handled per thread in phase 2; past this the kernel keeps the
 *  one-thread-per-(e,p) shape (a larger per-thread accumulator array would
 *  spill to local memory). */
#define GP1B_MAXPROJ 4

/*: The largest chunk `create` will reserve scratch for.  `create` does not
 *  know the sample length (`ne` arrives with `load_data`), so the scratch is
 *  sized from `batch_size` — and an unbounded `batch_size` (a caller passing
 *  10**9 to mean "one chunk", which is a legitimate and tested setting) would
 *  then reserve tens of GB for a sample far smaller than that.  One million
 *  events is ~58 MB of scratch at `n_proj = 2` and larger than any sample this
 *  kernel is used on, so the ceiling costs nothing and makes `batch_size` a
 *  *hint* above it rather than a memory demand. */
#define GP1B_MAX_CAP (1 << 20)

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                \
        }                                                                    \
    } while (0)

/* ── Catmull-Rom (the repo-wide idiom) ────────────────────────────────── */
__device__ double gp1b_cr1d(double pm1, double p0, double p1, double p2,
                            double t) {
    return p0 + 0.5 * t * (-pm1 + p1 + t * (2.0 * pm1 - 5.0 * p0 + 4.0 * p1
                                            - p2 + t * (-pm1 + 3.0 * p0
                                                        - 3.0 * p1 + p2)));
}

__device__ double gp1b_interp_real(const double* __restrict__ t, int row,
                                   double x, double xmin, double xdelta,
                                   int bins) {
    double diff = (x - xmin) / xdelta;
    int b = max(0, min((int)floor(diff), bins - 2));
    double s = max(0.0, min(diff - b, 1.0));
    int base = row * bins + b;
    int lo = row * bins, hi = (row + 1) * bins - 1;
    int im1 = base > lo ? base - 1 : base;
    int i2 = base + 2 <= hi ? base + 2 : base + 1;
    return gp1b_cr1d(t[im1], t[base], t[base + 1], t[i2], s);
}

__device__ void gp1b_interp_c(const double* __restrict__ tr,
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
    *orr = gp1b_cr1d(tr[im1], tr[base], tr[base + 1], tr[i2], s);
    *oii = gp1b_cr1d(ti[im1], ti[base], ti[base + 1], ti[i2], s);
}

/* ── load-time cache fill: cache_{e,k,p} = F_{e,k} · fa_{e,k,p} ────────
 *
 * `F = fl · Π num` carries no ck and no coeff, and `fa` is the angle
 * projection sum — neither depends on a fitted quantity, so both are fixed
 * for the whole fit.  They are stored FUSED, per `(e, k, p)`:
 *
 *     cache[(e·W + k)·P + p] = F_{e,k} · fa_{e,k,p}       (float2)
 *
 * The hot path needs `(F/D)·fa`, and since `D` is COMPLEX
 * `1/D = conj(D)/|D|²`, so
 *
 *     (F/D)·fa = (F·fa)·conj(D)·rden            rden = 1/|D|²
 *
 * i.e. 4 multiplies per `(e,k)` at ANY `n_proj` instead of rebuilding `F/D`
 * and then multiplying by `fa`.  It also retires the whole `fa_cache`:
 *
 *     before   F 1.82 + fa 3.64 + basis 4.83 = 10.29 GB   (per handle, 8.7M ev)
 *     after    F·fa 3.64        + basis 4.83 =  8.47 GB   → −1.82 GB (18%)
 *
 * `dQ/dD` then needs TWO reciprocals (`dcr` carries `F`, so its chain to
 * `dQ/dD` is `F/D²`), while `dQ/dck` is just `u'·conj(D)·rden`.  Both forms
 * were proved equal to the split ones in numpy for random complex `F` AND for
 * `F == 1` (where the fused kernel must reduce exactly to the split one)
 * before this edit.
 *
 * `n_res == 1` by construction, but the loop over `r` is kept as the config's
 * own `num_table` shape (one iteration) rather than pinned to a constant, so
 * the array is read where the builder emitted it.
 */
__global__ void gp1b_fill_kernel(
    const double* __restrict__ fl_table, const int* __restrict__ fl_q_index,
    const double* __restrict__ num_r, const double* __restrict__ num_i,
    const int* __restrict__ num_m_index,
    double m_min, double m_delta, int m_bins,
    double q_min, double q_delta, int q_bins,
    const double* __restrict__ mass, const double* __restrict__ mom,
    const int* __restrict__ angle_index, const double* __restrict__ angle_k,
    const double* __restrict__ angle_b,
    const double* __restrict__ am_r, const double* __restrict__ am_i,
    const double* __restrict__ ang, int n_ab, int n_comp, int n_angle_total,
    int ne, int n_wave, int n_proj, int n_res, int n_decay,
    int n_mass, int n_momentum,
    float2* __restrict__ cache)              /* [ne, n_wave, n_proj] F·fa */
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = ne * n_wave;
    if (gid >= total) return;
    const int e = gid / n_wave, k = gid - e * n_wave;

    double fl = 1.0;
    for (int d = 0; d < n_decay; d++) {
        const int qc = fl_q_index[k * n_decay + d];
        const double qv = mom[(size_t)e * n_momentum + qc];
        fl *= gp1b_interp_real(fl_table, k * n_decay + d, qv,
                               q_min, q_delta, q_bins);
    }
    double nr = 1.0, ni = 0.0;
    for (int r = 0; r < n_res; r++) {
        const int mc = num_m_index[k * n_res + r];
        const double mv = mass[(size_t)e * n_mass + mc];
        double vr, vi;
        gp1b_interp_c(num_r, num_i, k * n_res + r, mv,
                      m_min, m_delta, m_bins, &vr, &vi);
        const double t_r = nr * vr - ni * vi;
        ni = nr * vi + ni * vr;
        nr = t_r;
    }
    const double Fr = fl * nr, Fi = fl * ni;
    for (int p = 0; p < n_proj; p++) {
        double far = 0.0, fai = 0.0;
        for (int a = 0; a < n_ab; a++) {
            const int col = angle_index[a];
            double prod = 1.0;
            for (int j = 0; j < n_comp; j++) {
                const double th =
                    ang[((size_t)e * n_angle_total + col) * n_comp + j];
                prod *= cos(th * angle_k[a * n_comp + j]
                            + angle_b[a * n_comp + j]);
            }
            const int o = (k * n_proj + p) * n_ab + a;
            far += prod * am_r[o];
            fai += prod * am_i[o];
        }
        cache[(size_t)gid * n_proj + p] =
            make_float2((float)(Fr * far - Fi * fai),
                        (float)(Fr * fai + Fi * far));
    }
}

/* ── load-time fill of the one remaining coeff/ck-independent cache ────
 *
 * ``basis_m(m_e)`` depends on (m, e) only, so it is fixed for the whole fit.
 * `fa` used to get a cache of its own here; it is now folded into the `F`
 * fill above (that is where the 1.82 GB went).
 */
__global__ void gp1b_fill_static_kernel(
    const double* __restrict__ bas_r, const double* __restrict__ bas_i,
    const int* __restrict__ bas_m_index,
    double m_min, double m_delta, int m_bins,
    const double* __restrict__ mass,
    int ne, int n_basis, int n_mass,
    float2* __restrict__ bas_cache)
{
    /* basis: one thread per (event, basis row) */
    const int nb_total = ne * n_basis;
    for (int g = blockIdx.x * blockDim.x + threadIdx.x; g < nb_total;
         g += gridDim.x * blockDim.x) {
        const int e = g / n_basis, m = g - e * n_basis;
        const double mv = mass[(size_t)e * n_mass + bas_m_index[m]];
        double vr, vi;
        gp1b_interp_c(bas_r, bas_i, m, mv, m_min, m_delta, m_bins, &vr, &vi);
        bas_cache[g] = make_float2((float)vr, (float)vi);
    }
}

/* ── the tile's shared-memory layout, in doubles ───────────────────────
 *
 * ONE formula, used by the host (`smem_bytes`) and by the kernel (the pointer
 * offsets), so the launch size and the kernel's view can never disagree.
 *
 *   D      [TE·W·2]       the finished denominator D_{e,k} (= the ONE factor)
 *   rd     [TE·W]         1/|D|²   (one real reciprocal, all divisions become
 *                         multiplies — the biggest single win in the base)
 *   A      [TE·P·2]       A_{e,p}
 *   dA     [TE·P·2]       dQ/dA_{e,p}
 *   pe     [TE]           P_e
 *   gc     [4·NB]         per-coefficient shared accumulator
 *   gck    [2·W]          per-wave shared accumulator for dQ/dck
 *   pv     [TE·W·2]       dQ/dD_{e,k}      (per-wave path, or the slot sum's
 *                         input when collapsed)
 *   dd     [TE·NS·2]      Σ_{k∈slot} dQ/dD_{e,k}   (only when collapsed)
 *
 * With ``NS == 1`` and the benchmark shape (TE=8, W=100, P=2, NB=5) that is
 * 34 464 B — exactly gen2's `NR == 1` tile, because the only buffer gen2
 * carried there beyond this set was `s_dd`, which is still needed.
 */
typedef struct {
    int D, rd, A, dA, pe, gc, gck, pv, dd, total;
} GP1BLayout;

__host__ __device__ __forceinline__ GP1BLayout gp1b_layout(
    int collapsed, int W, int P, int NB, int NS)
{
    GP1BLayout L;
    int o = 0;
    L.D   = o; o += 2 * GP1B_TE * W;
    L.rd  = o; o += GP1B_TE * W;
    L.A   = o; o += 2 * GP1B_TE * P;
    L.dA  = o; o += 2 * GP1B_TE * P;
    L.pe  = o; o += GP1B_TE;
    L.gc  = o; o += 4 * NB;
    L.gck = o; o += 2 * W;
    L.pv  = o; o += 2 * GP1B_TE * W;
    L.dd  = o; o += collapsed ? 2 * GP1B_TE * (NS > 0 ? NS : 1) : 0;
    L.total = o;
    return L;
}

/* ── the per-iteration tile ─────────────────────────────────────────────
 *
 * ``mode`` — the grouped log-sum (``resolution_size``) plumbing:
 *
 * * ``0`` — the whole per-event objective in one launch (``rsize == 1`` or
 *   ``norm == None``).
 * * ``1`` — the forward half: write ``P_e``, save ``A`` to ``dA_buf``, and
 *   accumulate the three per-group sums (``S_g``, ``Pb_g``, ``W_g``).
 * * ``2`` — the backward half: ``A`` is read back from ``dA_buf`` and
 *   ``dQ/dP_e`` from ``dqp`` (already scaled by the group's ``-W_g/S_g``), so
 *   the projection sum does not run at all.
 *
 * The group partition stays ``g = ge / rsize`` over the GLOBAL event index: a
 * group of 10…100 events straddles several tiles, and the scale factor exists
 * only after every tile has run, which is what ``gp1b_grp_nll_kernel`` +
 * ``gp1b_scale_kernel`` provide.
 *
 * ``gamma`` is flat ``[n_wave·n_basis]`` (one factor in the middle axis).
 * Each row's true non-zeros are a CSR: ``[gs_lo[k], gs_hi[k])`` indexes the
 * flat ``nz_wave_m`` list (phase 1).  ``nz_row``/``nz_m`` are the phase-5
 * GROUP axis — the wave when the per-wave path runs, the SLOT when the
 * collapse does (the wrapper builds whichever the gate selected, deduplicated
 * per group).  A row may have interior holes; both lists hold only
 * non-zeros, so neither loop needs a ``g == 0`` test to be exact.
 */
__global__ void gp1b_tile_kernel(
    const float2* __restrict__ cache,      /* [ne, n_wave, n_proj]  F·fa  */
    const float2* __restrict__ bas_cache,  /* [ne, n_basis]    basis_m(m) */
    const double* __restrict__ gamma,      /* [n_wave, n_basis]  (n_res==1) */
    const int* __restrict__ nz_wave_m,     /* [nnz] per-wave nonzero cols */
    const int* __restrict__ gs_lo,         /* [n_wave]  CSR start into it  */
    const int* __restrict__ gs_hi,         /* [n_wave]  CSR end            */
    const int* __restrict__ slot_rep,      /* [n_slot]  a wave of the slot */
    const int* __restrict__ slot_ptr,      /* [n_slot+1]  CSR              */
    const int* __restrict__ slot_waves,    /* [n_wave]    CSR              */
    const int* __restrict__ nz_row,        /* [n_nz]    wave or slot       */
    const int* __restrict__ nz_m,          /* [n_nz]    basis column       */
    int use_slots, int n_slot,
    const double* __restrict__ ck_r, const double* __restrict__ ck_i,
    const double* __restrict__ coeff,
    double norm, int use_norm,
    const double* __restrict__ weight, const double* __restrict__ bkg,
    const double* __restrict__ dqp,        /* [ne]  mode 2 only           */
    const double* __restrict__ dA_in,      /* [2·ne·n_proj]  mode 2 only  */
    int rsize, int mode, int n_nz,
    int ne, int n_wave, int n_proj, int n_basis,
    double* __restrict__ P_out,
    double* __restrict__ g_ck_r, double* __restrict__ g_ck_i,
    double* __restrict__ g_coeff,
    double* __restrict__ acc,
    double* __restrict__ gS, double* __restrict__ gPb,
    double* __restrict__ gW,
    double* __restrict__ dA_out)           /* [2·ne·n_proj]  mode 1 only  */
{
    const int W = n_wave, P = n_proj, NB = n_basis, NS = n_slot;
    const GP1BLayout L = gp1b_layout(use_slots, W, P, NB, NS);

    extern __shared__ double sh[];
    double* s_D  = sh + L.D;
    double* s_rd = sh + L.rd;
    double* s_A  = sh + L.A;
    double* s_dA = sh + L.dA;
    double* s_pe = sh + L.pe;
    double* s_gc = sh + L.gc;
    double* s_gck = sh + L.gck;
    double* s_pv = sh + L.pv;
    double* s_dd = sh + L.dd;

    const int tid = threadIdx.x, nt = blockDim.x;
    for (int i = tid; i < 4 * NB; i += nt) s_gc[i] = 0.0;
    for (int i = tid; i < 2 * W; i += nt) s_gck[i] = 0.0;
    if (use_slots) {
        const int nd = GP1B_TE * (NS > 0 ? NS : 1) * 2;
        for (int i = tid; i < nd; i += nt) s_dd[i] = 0.0;
    }
    __syncthreads();
    const int e0 = blockIdx.x * GP1B_TE;
    const int evn = min(GP1B_TE, ne - e0);
    if (evn <= 0) return;

    /* ── phase 1: the denominator — a dot product over the CACHED basis ──
     *
     * Exactly one BW per chain, so `D` IS that factor, with no product
     * bookkeeping and no `s_s` buffer.  The inner loop walks the wave's true
     * non-zeros through the CSR (`nz_wave_m`), so a folded row costs its nnz
     * and a row with holes costs its non-zeros, not its span.  No `g == 0`
     * test: the list contains no zeros by construction.
     */
    for (int idx = tid; idx < evn * W; idx += nt) {
        const int e = idx / W, k = idx - e * W;
        const float2* bv = bas_cache + (size_t)(e0 + e) * NB;
        const int gb = k * NB;
        const int lo = gs_lo[k], hi = gs_hi[k];
        double sr = 0.0, si = 0.0;
        for (int i = lo; i < hi; i++) {
            const int m = nz_wave_m[i];
            const double w = gamma[gb + m] * coeff[m];
            sr += w * (double)bv[m].x;
            si += w * (double)bv[m].y;
        }
        s_D[idx * 2 + 0] = sr;
        s_D[idx * 2 + 1] = si;
        /* |D|² is a REAL scalar even though the basis (and D) is complex, so
           the shared part of `F/D` and of `dQ/dD` is one real reciprocal.
           fp64 divide is ~1/64 rate and unpipelined on a consumer GPU. */
        s_rd[idx] = 1.0 / (sr * sr + si * si);
    }
    __syncthreads();

    /* ── phase 2: A_{e,p} = Σ_k ck_k · F/D · fa   (cache = F·fa) ──────────
     *
     * Skipped in mode 2: the grouped backward launch reads the `A` that mode 1
     * saved instead of re-running this reduction over `n_wave` (which is the
     * entire reason the grouped path used to cost a second forward pass).
     *
     * The reduction over `k` is spread across the block by `(e, lane)` — NOT
     * `(e, p, lane)` — so one thread owns every projection of its `k` slice.
     *
     * `cache` holds `F·fa`, so `(F/D)·fa` is `cache/D` — one real reciprocal
     * (`s_rd`) and two multiplies per `(e,k,p)`, instead of rebuilding `F/D`
     * and then multiplying by a separate `fa`. */
    if (mode != 2) {
        if (P <= GP1B_MAXPROJ) {
            const int n_out = evn;
            for (int i = tid; i < n_out * P; i += nt) {
                s_A[i * 2 + 0] = 0.0;
                s_A[i * 2 + 1] = 0.0;
            }
            __syncthreads();
            const int nlane = (nt + n_out - 1) / n_out;
            const int lane = tid / n_out;
            const int e = tid - lane * n_out;
            if (e < n_out) {
                double ar[GP1B_MAXPROJ], ai[GP1B_MAXPROJ];
                for (int p = 0; p < P; p++) { ar[p] = 0.0; ai[p] = 0.0; }
                for (int k = lane; k < W; k += nlane) {
                    const double dr = s_D[(e * W + k) * 2 + 0];
                    const double di = s_D[(e * W + k) * 2 + 1];
                    const double rden = s_rd[e * W + k];
                    const double ckr = ck_r[k], cki = ck_i[k];
                    const float2* Ff =
                        cache + ((size_t)(e0 + e) * W + k) * P;
                    for (int p = 0; p < P; p++) {
                        const double tr = ((double)Ff[p].x * dr
                                          + (double)Ff[p].y * di) * rden;
                        const double ti = ((double)Ff[p].y * dr
                                          - (double)Ff[p].x * di) * rden;
                        ar[p] += ckr * tr - cki * ti;
                        ai[p] += ckr * ti + cki * tr;
                    }
                }
                for (int p = 0; p < P; p++) {
                    atomicAdd(&s_A[(e * P + p) * 2 + 0], ar[p]);
                    atomicAdd(&s_A[(e * P + p) * 2 + 1], ai[p]);
                }
            }
        } else {
            /* very many projections: one thread per (e,p), as before */
            const int n_out = evn * P;
            for (int i = tid; i < n_out; i += nt) {
                s_A[i * 2 + 0] = 0.0;
                s_A[i * 2 + 1] = 0.0;
            }
            __syncthreads();
            const int nlane = (nt + n_out - 1) / n_out;
            const int lane = tid / n_out;
            const int idx = tid - lane * n_out;
            if (idx < n_out) {
                const int e = idx / P, p = idx - e * P;
                double ar = 0.0, ai = 0.0;
                for (int k = lane; k < W; k += nlane) {
                    const float2 Ff =
                        cache[((size_t)(e0 + e) * W + k) * P + p];
                    const double dr = s_D[(e * W + k) * 2 + 0];
                    const double di = s_D[(e * W + k) * 2 + 1];
                    const double rden = s_rd[e * W + k];
                    const double tr = ((double)Ff.x * dr
                                      + (double)Ff.y * di) * rden;
                    const double ti = ((double)Ff.y * dr
                                      - (double)Ff.x * di) * rden;
                    ar += ck_r[k] * tr - ck_i[k] * ti;
                    ai += ck_r[k] * ti + ck_i[k] * tr;
                }
                atomicAdd(&s_A[idx * 2 + 0], ar);
                atomicAdd(&s_A[idx * 2 + 1], ai);
            }
        }
    }
    __syncthreads();

    /* ── phase 3: P_e, the objective, and (mode 1) the saved A ─────────── */
    if (mode != 2) {
        for (int e = tid; e < evn; e += nt) {
            double pe = 0.0;
            for (int p = 0; p < P; p++) {
                const double ar = s_A[(e * P + p) * 2 + 0];
                const double ai = s_A[(e * P + p) * 2 + 1];
                pe += ar * ar + ai * ai;
            }
            const int ge = e0 + e;
            const double wv = weight[ge], bv = bkg[ge];
            s_pe[e] = pe;
            if (P_out) P_out[ge] = pe;
            /* mode 1 keeps A: it is the ONLY forward output the backward
               launch consumes, and re-deriving it is a whole extra pass. */
            if (dA_out) {
                for (int p = 0; p < P; p++) {
                    const size_t o = (size_t)ge * P + p;
                    dA_out[o * 2 + 0] = s_A[(e * P + p) * 2 + 0];
                    dA_out[o * 2 + 1] = s_A[(e * P + p) * 2 + 1];
                }
            }
            if (mode == 1) {
                /* forward half of the grouped log-sum: per-group PARTIAL
                   sums, keyed on the GLOBAL group index. */
                const int g = ge / rsize;
                atomicAdd(gS + g, wv * (pe / norm + bv));
                atomicAdd(gPb + g, wv * pe);
                atomicAdd(gW + g, wv);
            } else {
                if (use_norm) {
                    atomicAdd(acc + 0, -wv * log(pe / norm + bv));
                    atomicAdd(acc + 1, wv * pe / (norm * (pe + bv * norm)));
                } else {
                    atomicAdd(acc + 0, wv * pe);
                }
            }
        }
        __syncthreads();
        if (mode == 1) return;          /* forward only: no gradient work */
    }

    /* ── dQ/dA_{e,p} ────────────────────────────────────────────────────── */
    for (int idx = tid; idx < evn * P; idx += nt) {
        const int ge = e0 + idx / P;
        double dqp_v;
        if (mode == 2) {
            /* already the finished -(W_g/S_g)·(w_e/norm) */
            dqp_v = dqp[ge];
            const size_t o = (size_t)ge * P + (idx % P);
            s_dA[idx * 2 + 0] = dqp_v * dA_in[o * 2 + 0];
            s_dA[idx * 2 + 1] = -dqp_v * dA_in[o * 2 + 1];
        } else {
            const double wv = weight[ge], bv = bkg[ge];
            const double pe = s_pe[idx / P];
            dqp_v = use_norm ? -wv / (pe + bv * norm) : wv;
            s_dA[idx * 2 + 0] = dqp_v * s_A[idx * 2 + 0];
            s_dA[idx * 2 + 1] = -dqp_v * s_A[idx * 2 + 1];
        }
    }
    __syncthreads();

    /* ── phase 4: u', dQ/dck and dQ/dD (the u-merge, kept from gen2) ───────
     *
     * The cache holds `F·fa`, so the projection reduction carries `F` too:
     *
     *     u'_{e,k}      = Σ_p dQ/dA_{e,p}·(F_{e,k}·fa_{e,p,k})   (WITH F)
     *     dQ/dck_k     += u'_{e,k}/D_{e,k}
     *     dQ/dD_{e,k}   = -ck_k·u'_{e,k}/D_{e,k}²
     *
     * The split spelling was `u = Σ_p dQ/dA·fa` (no F), `dQ/dck = Σ common·u`
     * and `dQ/dD = -ck·u·F/D²`; both were proved equal in numpy for random
     * complex `F` and for `F == 1`.  One pass still produces both, so `common`
     * is rebuilt once per `(e,k)` instead of twice, and with one factor
     * `dQ/ds_{e,k,0}` IS `dQ/dD_{e,k}` (the `Π_{l'≠l} s` product is empty),
     * so this is also phase 5's input.
     *
     * `dQ/dD` needs TWO reciprocals because `u'` already carries `F`: its
     * chain to `dQ/dD` is `F/D²`, not `1/D²`. */
    for (int idx = tid; idx < evn * W; idx += nt) {
        const int e = idx / W, k = idx - e * W;
        const int ge = e0 + e;
        double ur = 0.0, ui = 0.0;
        {
            const float2* Ff = cache + (size_t)ge * W * P + (size_t)k * P;
            for (int p = 0; p < P; p++) {
                const double gar = s_dA[(e * P + p) * 2 + 0];
                const double gai = s_dA[(e * P + p) * 2 + 1];
                ur += gar * (double)Ff[p].x - gai * (double)Ff[p].y;
                ui += gar * (double)Ff[p].y + gai * (double)Ff[p].x;
            }
        }
        const double dr = s_D[idx * 2 + 0], di = s_D[idx * 2 + 1];
        const double rden = s_rd[idx];
        /* `ur,ui` already carry F, so this is w = u'/D = (F/D)·u — the SAME
           quantity `common·u` the unfused kernel built.  dck is done. */
        const double wr = (ur * dr + ui * di) * rden;
        const double wi = (ui * dr - ur * di) * rden;
        atomicAdd(&s_gck[2 * k + 0], wr);
        atomicAdd(&s_gck[2 * k + 1], wi);
        /* dQ/dD = -(ck·u')/D² — two reciprocals, since u' carries F and the
           chain to dQ/dD is F/D².  Verified against numpy for random F AND
           for F == 1 (where it must reduce to the unfused expression). */
        const double qr0 = ck_r[k] * ur - ck_i[k] * ui;
        const double qi0 = ck_r[k] * ui + ck_i[k] * ur;
        const double qr = -(qr0 * dr + qi0 * di) * rden;
        const double qi = -(qi0 * dr - qr0 * di) * rden;
        s_pv[(size_t)idx * 2 + 0] = (qr * dr + qi * di) * rden;
        s_pv[(size_t)idx * 2 + 1] = (qi * dr - qr * di) * rden;
    }
    __syncthreads();

    /* ── dsum_{e,slot} = Σ_{k∈slot} dQ/dD_{e,k} ───────────────────────────
     * One thread per (e,slot) reads its slot's waves from the CSR list — no
     * shared atomics, so no contention when a slot holds most of the waves
     * (the n_slot == 1 shape, where a contended atomicAdd measured 2.25 ms of
     * 6.3 in gen2).  This is the half of the collapse that has CONTENT at one
     * factor: it turns `n_wave·run` dot products into `n_slot·run`. */
    if (use_slots) {
        for (int i = tid; i < evn * NS; i += nt) {
            const int sl = i % NS, e = i / NS;
            double sr = 0.0, si = 0.0;
            for (int j = slot_ptr[sl]; j < slot_ptr[sl + 1]; j++) {
                const size_t o = (size_t)(e * W + slot_waves[j]) * 2;
                sr += s_pv[o + 0];
                si += s_pv[o + 1];
            }
            s_dd[(e * NS + sl) * 2 + 0] = sr;
            s_dd[(e * NS + sl) * 2 + 1] = si;
        }
        __syncthreads();
    }

    /* one global atomic per (wave, block) for dQ/dck, as before */
    for (int k = tid; k < W; k += nt) {
        atomicAdd(g_ck_r + k, s_gck[2 * k + 0]);
        atomicAdd(g_ck_i + k, s_gck[2 * k + 1]);
    }

    /* ── phase 5: the coefficient gradient ────────────────────────────────
       dQ/dcoeff_m += 2Re Σ_e dQ/ds_{e,k,0}·gamma[k,m]·basis_m(m_e)

       Walked over the host-built `nz_row`/`nz_m` list of run entries, not over
       `n_wave·n_basis`: that list is exactly the contiguous run, so a dense row
       costs what it did and a sparse one costs its run.  `nz_row` is the
       group axis of the selected path — the wave (per-wave) or the slot
       (collapsed, whose gamma row is read through `slot_rep[sl]`, bit-equal to
       it by construction). */
    if (n_wave > 0) {
        if (use_slots) {
            for (int idx = tid; idx < n_nz; idx += nt) {
                const int sl = nz_row[idx], m = nz_m[idx];
                const double g = gamma[slot_rep[sl] * NB + m];
                if (g == 0.0) continue;
                double out = 0.0;
                for (int e = 0; e < evn; e++) {
                    const float2 bv = bas_cache[(size_t)(e0 + e) * NB + m];
                    const double dr = s_dd[(e * NS + sl) * 2 + 0];
                    const double di = s_dd[(e * NS + sl) * 2 + 1];
                    out += 2.0 * (dr * (double)bv.x - di * (double)bv.y);
                }
                atomicAdd(&s_gc[m], out * g);
            }
        } else {
            for (int idx = tid; idx < n_nz; idx += nt) {
                const int k = nz_row[idx], m = nz_m[idx];
                const double g = gamma[k * NB + m];
                if (g == 0.0) continue;
                double out = 0.0;
                for (int e = 0; e < evn; e++) {
                    const float2 bv = bas_cache[(size_t)(e0 + e) * NB + m];
                    const size_t o = (size_t)(e * W + k) * 2;
                    out += 2.0 * (s_pv[o + 0] * (double)bv.x
                                  - s_pv[o + 1] * (double)bv.y);
                }
                atomicAdd(&s_gc[m], out * g);
            }
        }
        __syncthreads();
        for (int m = tid; m < NB; m += nt)
            if (s_gc[m] != 0.0) atomicAdd(g_coeff + m, s_gc[m]);
    }
}

/* ── grouped log-sum: the per-group sums -> Q, dnorm and the scale ──────
 *
 *     W_g = Σ w,  S_g = Σ w·(P/norm + bkg),  Pb_g = Σ w·P
 *     Q      += -W_g·log(S_g/W_g)            (only when |W_g| > GP_GROUP_EPS)
 *     dnorm  +=  W_g·Pb_g / (S_g·norm²)
 *     scale_g = -W_g/S_g                     (consumed by gp1b_scale_kernel)
 */
__global__ void gp1b_grp_nll_kernel(
    const double* __restrict__ gS, const double* __restrict__ gPb,
    const double* __restrict__ gW, int n_groups, double norm,
    double* __restrict__ gscale, double* __restrict__ acc)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    const double W = gW[g], S = gS[g];
    if (!(fabs(W) > GP_GROUP_EPS)) {
        gscale[g] = 0.0;
        return;
    }
    gscale[g] = -W / S;
    atomicAdd(acc + 0, -W * log(S / W));
    atomicAdd(acc + 1, W * gPb[g] / (S * norm * norm));
}

/* ── dQ/dP_e = scale_{g(e)} · w_e/norm, one thread per event ──────────── */
__global__ void gp1b_scale_kernel(
    const double* __restrict__ gscale,
    const double* __restrict__ weight, int ne, int rsize, double norm,
    double* __restrict__ dqp)
{
    const int ge = blockIdx.x * blockDim.x + threadIdx.x;
    if (ge >= ne) return;
    dqp[ge] = gscale[ge / rsize] * (weight[ge] / norm);
}

/*=============================================================================
 * Host side
 *===========================================================================*/
typedef struct {
    int *fl_q_index, *num_m_index, *bas_m_index, *angle_index;
    int *gs_lo, *gs_hi, *slot_rep, *slot_ptr, *slot_waves;
    int *nz_wave_k, *nz_wave_m;     /* per-wave run entries            */
    int *nz_slot_k, *nz_slot_m;     /* per-slot run entries (collapsed) */
    double *fl_table, *num_r, *num_i, *bas_r, *bas_i, *gamma;
    double *angle_k, *angle_b, *am_r, *am_i;
    double m_min, m_delta, q_min, q_delta;
    int m_bins, q_bins;
    int n_wave, n_proj, n_res, n_decay, n_basis, n_ab, n_comp;
    int n_mass, n_momentum, n_angle_total;
    int n_nz_wave, n_nz_slot;
    int n_slot, use_slots;
    int batch_size;
    int rsize;            /* events per grouped-log-sum group (>= 1) */
    double *h_ck_r, *h_ck_i, *h_coeff;
    double *p_ck_r, *p_ck_i, *p_coeff;
    double *acc, *h_acc;
    double *g_ck_r, *g_ck_i, *g_coeff;
    cudaStream_t stream;
    int smem_bytes;
    /* BATCH-SIZED WORKING SCRATCH.  Every internal array that used to be
       `ne`-sized lives here instead, sized by `batch_size`, allocated ONCE
       at create and reused for every chunk of every call — no per-call
       malloc/free, and the fitter's `ne` (which it does not know at create)
       no longer matters.  Only the per-EVENT DATA (cache, basis, weight,
       bkg) stays in the handle. */
    int cap;              /* events per chunk, aligned to rsize; never 0 */
    double *s_P, *h_sP;   /* per-chunk P (device + pinned host) */
    double *s_dA;         /* per-chunk saved A   (rsize > 1) */
    double *s_dqp;        /* per-chunk dQ/dP scale */
    double *s_gS, *s_gPb, *s_gW, *s_gscale;   /* per-chunk group arrays */
    int s_ngroups;        /* groups a chunk can hold */
} GP1BCtx;

/* ONE SAMPLE.  Everything sized by ``ne`` lives here, so two handles (the
   fitter's phase space and its data) never alias.  Only the per-EVENT data is
   kept: `cache` (F·fa), `bas_cache`, `weight`, `bkg`.  Everything that is
   scratch is the CONTEXT's batch-sized working set (see `GP1BCtx`). */
typedef struct {
    double *weight, *bkg;
    float2 *cache;        /* F·fa                 [ne, n_wave, n_proj]  */
    float2 *bas_cache;    /* basis_m(m_e)         [ne, n_basis]         */
    int ne;
    int rsize;
} GP1BDataHandle;

static int* _gp1b_up_i(const int* src, int n) {
    if (n <= 0) return NULL;
    int* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(int),
                          cudaMemcpyHostToDevice));
    return d;
}

static double* _gp1b_up_d(const double* src, int n) {
    if (n <= 0) return NULL;
    double* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(double),
                          cudaMemcpyHostToDevice));
    return d;
}

/* The one launch.
 *
 * BATCHING NEEDS NO KERNEL CHANGE.  The chunk is expressed purely as POINTER
 * OFFSETS into the per-event data plus a chunk length passed as the kernel's
 * `ne`, so the kernel's own `e0 + e` / `ge` are chunk-local by construction:
 *
 *     cache     -> h->cache     + st·n_wave·n_proj
 *     bas_cache -> h->bas_cache + st·n_basis
 *     weight    -> h->weight + st        bkg -> h->bkg + st
 *
 * That is why `gp1b_tile_kernel`, `gp1b_grp_nll_kernel` and `gp1b_scale_kernel`
 * carry no batching logic.  `st` is always a multiple of `rsize` (see the
 * chunk rule in `create`), so the chunk-local group index `ge / rsize` is
 * consistent within a chunk and a group never straddles one.
 */
static void gp1b_launch(GP1BCtx* c, GP1BDataHandle* h, int st, int nb,
                        double norm, int use_norm, int mode, double* P_out,
                        double* dA_out, double* gS, double* gPb, double* gW)
{
    const int tiles = (nb + GP1B_TE - 1) / GP1B_TE;
    const size_t sm = (size_t)c->smem_bytes;
    const double* dA_in = (mode == 2) ? c->s_dA : NULL;
    const float2* cache = h->cache + (size_t)st * c->n_wave * c->n_proj;
    const float2* bas_cache = h->bas_cache + (size_t)st * c->n_basis;
    const double* weight = h->weight + st;
    const double* bkg = h->bkg + st;
    /* the phase-5 group axis follows the path the gate chose */
    const int* nz_row = c->use_slots ? c->nz_slot_k : c->nz_wave_k;
    const int* nz_m = c->use_slots ? c->nz_slot_m : c->nz_wave_m;
    const int n_nz = c->use_slots ? c->n_nz_slot : c->n_nz_wave;
    gp1b_tile_kernel<<<tiles, GP1B_BLOCK, sm, c->stream>>>(
        cache, bas_cache, c->gamma,
        c->nz_wave_m, c->gs_lo, c->gs_hi,
        c->slot_rep, c->slot_ptr, c->slot_waves,
        nz_row, nz_m, c->use_slots, c->n_slot,
        c->p_ck_r, c->p_ck_i, c->p_coeff, norm, use_norm,
        weight, bkg, c->s_dqp, dA_in, h->rsize, mode, n_nz,
        nb, c->n_wave, c->n_proj, c->n_basis,
        P_out, c->g_ck_r, c->g_ck_i, c->g_coeff, c->acc,
        gS, gPb, gW, dA_out);
}

extern "C" {

void* cuda_create_context_flat_pwa_gen_1bw(
    const double* fl_table, int n_fl, const int* fl_q_index, int n_fq,
    const double* num_r, const double* num_i, int n_num,
    const int* num_m_index, int n_nm, const double* bas_r,
    const double* bas_i, int n_bas, const int* bas_m_index, int n_bm,
    const double* gamma, int n_g,
    const int* gs_lo, int n_lo, const int* gs_hi, int n_hi,
    const int* slot_rep, int n_sr, const int* slot_ptr, int n_sp,
    const int* slot_waves, int n_swv,
    int n_slot, int use_slots,
    const int* nz_wave_k, int n_nzk, const int* nz_wave_m, int n_nzm,
    const int* nz_slot_k, int n_nzsk, const int* nz_slot_m, int n_nzsm,
    const int* angle_index, int n_ai,
    const double* angle_k, int n_ak, const double* angle_b, int n_ab_,
    const double* am_r, const double* am_i, int n_am,
    double m_min, double m_delta, int m_bins,
    double q_min, double q_delta, int q_bins,
    int n_wave, int n_proj, int n_res, int n_decay, int n_basis,
    int n_ab, int n_comp, int n_mass, int n_momentum, int n_angle_total,
    int batch_size, int rsize)
{
    if (n_res != 1) {
        /* the wrapper rejects this first; the kernel refuses too rather than
           reading a 3-D gamma as if it had one factor. */
        fprintf(stderr,
                "cuda_flat_pwa_gen_1bw: n_res must be 1, got %d — use "
                "cuda_flat_pwa_gen2 for multi-BW chains\n", n_res);
        return NULL;
    }
    if (n_lo != n_wave || n_hi != n_wave) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_1bw: gs_lo/gs_hi must have n_wave = %d "
                "entries, got %d/%d\n", n_wave, n_lo, n_hi);
        return NULL;
    }
    if (n_nzk != n_nzm || n_nzsk != n_nzsm) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_1bw: nz_k/nz_m length mismatch "
                "(%d/%d, %d/%d)\n", n_nzk, n_nzm, n_nzsk, n_nzsm);
        return NULL;
    }
    GP1BCtx* c = (GP1BCtx*)calloc(1, sizeof(GP1BCtx));
    if (!c) return NULL;
    c->rsize = (rsize > 0) ? rsize : 1;
    /* ── THE CHUNK RULE (verbatim from gen2 / `cuda_v5_pwa_s0c`) ──────────
     *
     * A chunk must hold WHOLE log-sum groups, so the per-chunk stride is the
     * batch rounded DOWN to a multiple of `rsize`: group `g` then lives
     * entirely inside one chunk (`g = e // rsize` with `e` chunk-local), and
     * the whole grouped pipeline (mode 1 -> grp_nll -> scale -> mode 2) runs
     * inside the chunk — nothing per-group survives a chunk cut and the
     * partition does not depend on `batch_size`.  `cap` is what the scratch
     * below is sized by; `ne` (unknown at create) never enters it.
     *
     * Both inputs are guarded.  A non-positive `batch_size` must not produce a
     * zero-length scratch (a zero-event chunk would be an infinite loop in
     * `compute`), and neither must the alignment: `a == 0` means
     * `batch_size < rsize`, and an unaligned cap there would let a group
     * straddle two chunks — measured wrong (Q off by 49 %) before the guard.
     * Raising the cap to one whole group is the smallest legal chunk, not a
     * fallback. */
    c->cap = (batch_size > 0) ? batch_size : 50000;
    if (c->cap > GP1B_MAX_CAP) c->cap = GP1B_MAX_CAP;
    if (c->rsize > 1) {
        const int a = (c->cap / c->rsize) * c->rsize;
        c->cap = (a > 0) ? a : c->rsize;
    }
    c->s_ngroups = (c->cap + c->rsize - 1) / c->rsize;
    c->fl_table = _gp1b_up_d(fl_table, n_fl);
    c->fl_q_index = _gp1b_up_i(fl_q_index, n_fq);
    c->num_r = _gp1b_up_d(num_r, n_num);
    c->num_i = _gp1b_up_d(num_i, n_num);
    c->num_m_index = _gp1b_up_i(num_m_index, n_nm);
    c->bas_r = _gp1b_up_d(bas_r, n_bas);
    c->bas_i = _gp1b_up_d(bas_i, n_bas);
    c->bas_m_index = _gp1b_up_i(bas_m_index, n_bm);
    c->gamma = _gp1b_up_d(gamma, n_g);
    c->gs_lo = _gp1b_up_i(gs_lo, n_lo);
    c->gs_hi = _gp1b_up_i(gs_hi, n_hi);
    c->slot_rep = _gp1b_up_i(slot_rep, n_sr);
    c->slot_ptr = _gp1b_up_i(slot_ptr, n_sp);
    c->slot_waves = _gp1b_up_i(slot_waves, n_swv);
    c->nz_wave_k = _gp1b_up_i(nz_wave_k, n_nzk);
    c->nz_wave_m = _gp1b_up_i(nz_wave_m, n_nzm);
    c->nz_slot_k = _gp1b_up_i(nz_slot_k, n_nzsk);
    c->nz_slot_m = _gp1b_up_i(nz_slot_m, n_nzsm);
    c->n_nz_wave = n_nzk;
    c->n_nz_slot = n_nzsk;
    c->n_slot = (n_slot > 0) ? n_slot : 1;
    c->angle_index = _gp1b_up_i(angle_index, n_ai);
    c->angle_k = _gp1b_up_d(angle_k, n_ak);
    c->angle_b = _gp1b_up_d(angle_b, n_ab_);
    c->am_r = _gp1b_up_d(am_r, n_am);
    c->am_i = _gp1b_up_d(am_i, n_am);
    c->m_min = m_min; c->m_delta = m_delta; c->m_bins = m_bins;
    c->q_min = q_min; c->q_delta = q_delta; c->q_bins = q_bins;
    c->n_wave = n_wave; c->n_proj = n_proj; c->n_res = n_res;
    c->n_decay = n_decay; c->n_basis = n_basis; c->n_ab = n_ab;
    c->n_comp = n_comp; c->n_mass = n_mass; c->n_momentum = n_momentum;
    c->n_angle_total = n_angle_total;
    c->batch_size = batch_size;
    /* the collapse is a per-(k,m) -> per-(slot,m) reassociation.  The wrapper
       decides (its `SLOT_GATE_RATIO`); `n_res == 1` cannot reach the
       deep-factor fallback that lacks a dsum buffer, so nothing narrows it
       further here. */
    c->use_slots = (use_slots && c->n_slot > 0 && n_res <= GP1B_MAXRES) ? 1 : 0;

    GP1BLayout lay = gp1b_layout(c->use_slots, n_wave, n_proj, n_basis,
                                 c->n_slot);
    c->smem_bytes = (int)(lay.total * sizeof(double));
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (c->smem_bytes > optin) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_1bw: shared tile %d B exceeds the "
                "opt-in limit %d B (n_wave %d, n_proj %d, n_basis %d, "
                "n_slot %d)\n",
                c->smem_bytes, optin, n_wave, n_proj, n_basis, c->n_slot);
        free(c);
        return NULL;
    }
    /* Past the 48 KiB default the launch needs the opt-in size on the
       FUNCTION, not merely a ``smem_bytes <= optin`` check against the
       device: without it the launch fails with "invalid argument" and the
       post-launch CUDA_CHECK is never reached, so compute() would silently
       return Q = 0.

       The attribute is per FUNCTION, not per context, so two live contexts of
       different tile sizes would fight: a smaller context created second would
       LOWER the opt-in and break the larger one's launch (measured: the second
       create made every subsequent launch fail with "invalid argument" and
       return Q = 0).  Raising only — never lowering — makes the attribute the
       process-wide maximum, which every context's own size is <= by
       construction, so any create order works. */
    {
        static size_t opt_in_max = 0;
        const void* fn = (const void*)gp1b_tile_kernel;
        if ((size_t)c->smem_bytes > opt_in_max) {
            cudaError_t e = cudaFuncSetAttribute(
                fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                c->smem_bytes);
            if (e != cudaSuccess) {
                fprintf(stderr,
                        "cuda_flat_pwa_gen_1bw: cannot opt in to %d B shared "
                        "memory (%s); n_wave %d\n",
                        c->smem_bytes, cudaGetErrorString(e), n_wave);
                free(c);
                return NULL;
            }
            opt_in_max = (size_t)c->smem_bytes;
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
    /* ── the batch-sized working scratch, allocated ONCE, at the END ───────
     * Sized by `cap` (events per chunk), never by the sample's `ne` — which is
     * unknown here and is the whole point of the change.  Deliberately the
     * last thing `create` does: every failure path above returns after
     * `free(c)` without a partial allocation to unwind.  Pinned host staging
     * for `P` (it is copied out per chunk), plain device memory for the rest.
     */
    CUDA_CHECK(cudaMalloc(&c->s_P, (size_t)c->cap * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&c->h_sP, (size_t)c->cap * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->s_dqp, (size_t)c->cap * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->s_gS, (size_t)c->s_ngroups * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->s_gPb, (size_t)c->s_ngroups * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->s_gW, (size_t)c->s_ngroups * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->s_gscale,
                          (size_t)c->s_ngroups * sizeof(double)));
    /* the saved `A`: only the grouped objective consumes it (mode 2), so a
       per-event fit (rsize == 1) never runs that launch and allocates nothing
       here. */
    c->s_dA = NULL;
    if (c->rsize > 1)
        CUDA_CHECK(cudaMalloc(&c->s_dA,
                              (size_t)c->cap * n_proj * 2 * sizeof(double)));
    return c;
}

void* cuda_load_data_flat_pwa_gen_1bw(void* vctx, const double* mass,
                                      const int* nm, const double* mom,
                                      const int* nmom, const double* ang,
                                      const int* nang, const double* weight,
                                      const double* bkg, int ne)
{
    GP1BCtx* c = (GP1BCtx*)vctx;
    (void)nm; (void)nmom; (void)nang;
    /* A FRESH handle per load: the fitter holds the phase-space and the data
       handles at the same time, so nothing here may live on the context. */
    GP1BDataHandle* h = (GP1BDataHandle*)calloc(1, sizeof(GP1BDataHandle));
    if (!h) return NULL;
    h->ne = ne;
    h->rsize = (c->rsize > 0) ? c->rsize : 1;
    /* mass / momentum / angles are consumed ONLY by the fill kernels: after
       them the hot path reads F, basis and fa from their caches. */
    double* t_mass = _gp1b_up_d(mass, ne * c->n_mass);
    double* t_mom = _gp1b_up_d(mom, ne * c->n_momentum);
    double* t_ang = _gp1b_up_d(ang, ne * c->n_angle_total * c->n_comp);
    h->weight = _gp1b_up_d(weight, ne);
    h->bkg = _gp1b_up_d(bkg, ne);
    CUDA_CHECK(cudaMalloc(&h->cache,
                          (size_t)ne * c->n_wave * c->n_proj
                          * sizeof(float2)));   /* F·fa, per (e,k,p) */
    CUDA_CHECK(cudaMalloc(&h->bas_cache,
                          (size_t)ne * c->n_basis * sizeof(float2)));
    /* NOTHING else is per-sample: P, dA, dqp and the group arrays are the
       CONTEXT's batch-sized scratch (see `create`), so this handle holds only
       the per-EVENT data and its footprint is the whole point of the change. */

    /* F·fa is independent of ck/coeff: fill it ONCE, here */
    const int total = ne * c->n_wave;
    const int threads = GP1B_BLOCK;
    const int blocks = (total + threads - 1) / threads;
    gp1b_fill_kernel<<<blocks, threads, 0, c->stream>>>(
        c->fl_table, c->fl_q_index, c->num_r, c->num_i, c->num_m_index,
        c->m_min, c->m_delta, c->m_bins, c->q_min, c->q_delta, c->q_bins,
        t_mass, t_mom,
        c->angle_index, c->angle_k, c->angle_b, c->am_r, c->am_i,
        t_ang, c->n_ab, c->n_comp, c->n_angle_total,
        ne, c->n_wave, c->n_proj, c->n_res, c->n_decay,
        c->n_mass, c->n_momentum, h->cache);

    /* the one remaining coeff/ck-independent cache: basis_m(m_e) */
    const int st_total = ne * c->n_wave * c->n_proj;
    const int st_blocks = (st_total + threads - 1) / threads;
    gp1b_fill_static_kernel<<<st_blocks, threads, 0, c->stream>>>(
        c->bas_r, c->bas_i, c->bas_m_index, c->m_min, c->m_delta, c->m_bins,
        t_mass, ne, c->n_basis, c->n_mass, h->bas_cache);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(c->stream));

    /* the kinematics are no longer needed: every consumer reads a cache */
    cudaFree(t_mass); cudaFree(t_mom); cudaFree(t_ang);
    return h;
}

/* Takes the HANDLE (never the context), is safe on NULL and safe twice: every
   pointer is NULLed after it is released.  Only per-EVENT data lives here —
   the batch-sized scratch is the CONTEXT's, freed by the function below. */
void cuda_free_data_flat_pwa_gen_1bw(void* vh) {
    GP1BDataHandle* h = (GP1BDataHandle*)vh;
    if (!h) return;
    if (h->weight) { cudaFree(h->weight); h->weight = NULL; }
    if (h->bkg) { cudaFree(h->bkg); h->bkg = NULL; }
    if (h->cache) { cudaFree(h->cache); h->cache = NULL; }
    if (h->bas_cache) { cudaFree(h->bas_cache); h->bas_cache = NULL; }
    free(h);
}

void cuda_free_context_flat_pwa_gen_1bw(void* vctx) {
    GP1BCtx* c = (GP1BCtx*)vctx;
    if (!c) return;
    #define F(p) if (c->p) cudaFree(c->p)
    #define FH(p) if (c->p) cudaFreeHost(c->p)
    F(fl_table); F(fl_q_index); F(num_r); F(num_i); F(num_m_index);
    F(bas_r); F(bas_i); F(bas_m_index); F(gamma);
    F(gs_lo); F(gs_hi); F(slot_rep); F(slot_ptr); F(slot_waves);
    F(nz_wave_k); F(nz_wave_m); F(nz_slot_k); F(nz_slot_m);
    F(angle_index);
    F(angle_k); F(angle_b); F(am_r); F(am_i);
    F(p_ck_r); F(p_ck_i); F(p_coeff); F(acc);
    F(g_ck_r); F(g_ck_i); F(g_coeff);
    FH(h_ck_r); FH(h_ck_i); FH(h_coeff); FH(h_acc);
    /* the batch-sized working scratch */
    F(s_P); F(s_dA); F(s_dqp); F(s_gS); F(s_gPb); F(s_gW); F(s_gscale);
    FH(h_sP);
    #undef F
    #undef FH
    if (c->stream) cudaStreamDestroy(c->stream);
    free(c);
}

void cuda_compute_flat_pwa_gen_1bw(
    void* vctx, void* vh, const double* ck_r, const double* ck_i,
    const double* coeff, double norm, int use_norm,
    double* oQ, double* oDn, double* oP,
    double* og_ck_r, double* og_ck_i, double* og_coeff)
{
    GP1BCtx* c = (GP1BCtx*)vctx;
    GP1BDataHandle* h = (GP1BDataHandle*)vh;
    const int N = c->n_wave, NB = c->n_basis;
    if (!h) return;
    memcpy(c->h_ck_r, ck_r, (size_t)N * sizeof(double));
    memcpy(c->h_ck_i, ck_i, (size_t)N * sizeof(double));
    memcpy(c->h_coeff, coeff, (size_t)NB * sizeof(double));
    cudaMemcpyAsync(c->p_ck_r, c->h_ck_r, (size_t)N * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    cudaMemcpyAsync(c->p_ck_i, c->h_ck_i, (size_t)N * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    cudaMemcpyAsync(c->p_coeff, c->h_coeff, (size_t)NB * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    /* The accumulators span the WHOLE sample and live on the context, so they
       are zeroed ONCE here and read back ONCE at the end — independent of how
       many chunks the sample is split into. */
    cudaMemsetAsync(c->acc, 0, 2 * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_ck_r, 0, (size_t)N * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_ck_i, 0, (size_t)N * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_coeff, 0, (size_t)NB * sizeof(double), c->stream);

    /* ── the chunk loop ────────────────────────────────────────────────────
     *
     * `c->cap` events at a time (a whole number of log-sum groups, see
     * `create`), with the per-event data addressed through OFFSET pointers so
     * the kernels see a chunk-local event index.  Everything the pipeline
     * needs per chunk — P, saved A, dqp, the group sums — is context scratch
     * sized by `cap`; only `cache`/`bas_cache`/`weight`/`bkg` scale with `ne`,
     * and they stay fully resident in the handle.
     *
     * `oP` is the CALLER's buffer (`h->ne` long) and is written at `oP + st`;
     * nothing about P is chunk-sized on the caller's side.
     *
     * The grouped log-sum needs the WHOLE group before it can scale dQ/dP.
     * The chunk holds whole groups, so the entire sequence
     * (mode 1 -> grp_nll -> scale -> mode 2) runs inside the chunk and the
     * group indices the kernels use are chunk-local by construction.  rsize
     * == 1 (and the phsp path) stay the single launch they were, per chunk. */
    const int ne_total = h->ne;
    int bs = c->cap;
    if (bs <= 0) bs = 1;
    if (ne_total < bs) bs = ne_total;
    const int nbat = (ne_total > 0) ? (ne_total + bs - 1) / bs : 0;
    for (int b = 0; b < nbat; b++) {
        const int st = b * bs;
        const int nb = (ne_total - st > bs) ? bs : (ne_total - st);
        const int grouped = (use_norm && h->rsize > 1 && c->s_ngroups > 0
                             && c->s_dA != NULL);
        if (grouped) {
            cudaMemsetAsync(c->s_gS, 0,
                            (size_t)c->s_ngroups * sizeof(double), c->stream);
            cudaMemsetAsync(c->s_gPb, 0,
                            (size_t)c->s_ngroups * sizeof(double), c->stream);
            cudaMemsetAsync(c->s_gW, 0,
                            (size_t)c->s_ngroups * sizeof(double), c->stream);
            gp1b_launch(c, h, st, nb, norm, use_norm, 1,
                        oP ? c->s_P : NULL,
                        c->s_dA, c->s_gS, c->s_gPb, c->s_gW);
            CUDA_CHECK(cudaGetLastError());
            const int ngroups = (nb + h->rsize - 1) / h->rsize;
            const int gblk = 256;
            gp1b_grp_nll_kernel<<<(ngroups + gblk - 1) / gblk, gblk, 0,
                                  c->stream>>>(
                c->s_gS, c->s_gPb, c->s_gW, ngroups, norm, c->s_gscale,
                c->acc);
            gp1b_scale_kernel<<<(nb + gblk - 1) / gblk, gblk, 0,
                                c->stream>>>(
                c->s_gscale, h->weight + st, nb, h->rsize, norm, c->s_dqp);
            CUDA_CHECK(cudaGetLastError());
            gp1b_launch(c, h, st, nb, norm, use_norm, 2, NULL, NULL,
                        NULL, NULL, NULL);
            CUDA_CHECK(cudaGetLastError());
        } else {
            gp1b_launch(c, h, st, nb, norm, use_norm, 0,
                        oP ? c->s_P : NULL, NULL, NULL, NULL, NULL);
            CUDA_CHECK(cudaGetLastError());
        }
        if (oP) {
            cudaMemcpyAsync(c->h_sP, c->s_P, (size_t)nb * sizeof(double),
                            cudaMemcpyDeviceToHost, c->stream);
            CUDA_CHECK(cudaStreamSynchronize(c->stream));
            memcpy(oP + st, c->h_sP, (size_t)nb * sizeof(double));
        }
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
}

/* The saved-`A` footprint of the context's working scratch, in BYTES PER
 * CHUNK (0 when `rsize == 1`, i.e. when the grouped path — the only consumer —
 * is not in play).
 *
 * CHOSEN: the context, not the handle.  The `A` buffer is no longer
 * handle-side (it is batch-sized scratch on the context, allocated once and
 * `ne`-independent), so a handle has no `A` allocation to report.  The symbol
 * name is kept — callers only ever used it to assert the allocation policy —
 * and the wrapper turns this per-chunk figure into a per-sample one
 * (`min(cap, ne)`), which is the sample's own share of the shared scratch. */
long long cuda_handle_dA_bytes_flat_pwa_gen_1bw(void* vctx) {
    GP1BCtx* c = (GP1BCtx*)vctx;
    if (!c || c->s_dA == NULL) return 0;
    return (long long)c->cap * c->n_proj * 2 * (long long)sizeof(double);
}

/* The tile size `gp1b_layout` gives for an arbitrary shape — the SAME function
 * the kernel and `create` use, exposed so a test can assert the buffer
 * arithmetic without a second, drifting copy in Python. */
long long cuda_tile_bytes_flat_pwa_gen_1bw(int collapsed, int n_slot,
                                            int n_wave, int n_proj,
                                            int n_basis) {
    GP1BLayout L = gp1b_layout(collapsed, n_wave, n_proj, n_basis, n_slot);
    return (long long)L.total * (long long)sizeof(double);
}

int cuda_get_device_count_flat_pwa_gen_1bw(void) {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
}

/* The dynamic shared-memory size the context was created with.  The host must
 * NOT recompute it: `gp1b_layout` is the single definition, and a second copy
 * in the wrapper is exactly the drift that makes a launch silently overrun its
 * tile (measured: a 256 B under-count gave "invalid argument" and Q = 0). */
int cuda_smem_bytes_flat_pwa_gen_1bw(void* vctx) {
    GP1BCtx* c = (GP1BCtx*)vctx;
    return c ? c->smem_bytes : 0;
}

/* Events per chunk.  Published for the same reason as `smem_bytes`: the chunk
 * rule (batch rounded down to a whole number of `rsize` groups, never zero)
 * must have ONE definition, and it lives in `create`.  The wrapper uses this to
 * report a sample's share of the shared scratch, never to re-derive the rule. */
int cuda_cap_flat_pwa_gen_1bw(void* vctx) {
    GP1BCtx* c = (GP1BCtx*)vctx;
    return c ? c->cap : 0;
}

void cuda_get_device_name_flat_pwa_gen_1bw(char* buf, int len) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    snprintf(buf, len, "%s", p.name);
}

}  /* extern "C" */
