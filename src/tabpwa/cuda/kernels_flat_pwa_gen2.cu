/**
 * cuda_flat_pwa_gen2 — the flat `pwa_gen` tile with four verified reductions.
 *
 * Case-1 copy (kernel policy §1): own `.cu`, own `.so`, own wrapper, own
 * symbols.  It is a copy of `kernels_flat_cache_pwa_gen.cu` — that file and
 * its wrapper stay byte-identical — with the reductions below.  Unlike the
 * `_cache` variant this one does NOT touch the config or the basis layout: it
 * is the cached-`F`/`fa`/basis machinery plus a different backward pass, so
 * `n_basis` is the config's own row count on both sides.
 *
 * The base kernel computes the same per-iteration quantities as
 * `cuda_flat_cache_pwa_gen` (the cached `F`, `basis_m(m_e)` and `fa`
 * tables), so the two agree on `Q`/`P`/gradients up to summation order.  The
 * four changes are:
 *
 * 1. **`u`-merge.**  The old kernel reduced over the projections `p` twice —
 *    once for `dQ/dck` (its phase 4) and once for `dQ/dcommon` (phase 5's
 *    pass A) — and each time rebuilt `common = F/D`.  Both are the same
 *    reduction:
 *
 *        u_{e,k}       = Σ_p dQ/dA_{e,p} · fa_{e,p,k}     (NO `ck` factor)
 *        dQ/dcommon_k  = ck_k · u_{e,k}
 *        dQ/dck_k      = Σ_e common_{e,k} · u_{e,k}
 *
 *    So phase 4 is gone: `u` is formed once, in one `(e,k)` pass that also
 *    produces `dQ/dD`.  Both of its consumers are in that same iteration, so
 *    `u` never has to be staged — it stays in registers (the shared `s_u`
 *    buffer is therefore not allocated at all).
 *
 * 2. **`n_res == 1` specialisation (template `NR`).**  With one factor
 *    `D = Π_l s_l = s_0` exactly, so `s_s[...,0]` is bit-identical to `s_D`
 *    and `s_s` is only ever *written* (the prefix loop has no iterations).
 *    At `NR == 1` the `s_s` buffer is dropped entirely and the single factor
 *    lives in registers; the compiler also folds the whole `l` loop.  That is
 *    12.8 KiB less shared memory per block at `n_wave = 100`.
 *
 * 3. **`n_slot` collapse of `dQ/dcoeff` (GATED).**  `gamma[k,l,m]` is a 0/1
 *    selector, so waves with the same `gamma[k]` row have bit-identical `s`
 *    and `D`.  Grouping the waves into `n_slot` distinct rows lets the
 *    backward pass
 *
 *        w_{e,slot,l}  = Π_{l'≠l} s_{e,slot,l'}          (once per (e,slot,l))
 *        dsum_{e,slot} = Σ_{k∈slot} dQ/dD_{e,k}
 *        dQ/dcoeff_m  += 2Re Σ_{(slot,l,m): γ≠0} w_{e,slot,l}·dsum_{e,slot}·basis_m
 *
 *    i.e. the per-`(k,l,m)` dot product becomes per-`(slot,l,m)`.  It is
 *    exact (`w` is shared by construction).  Its gain shrinks with the slot
 *    count, and at `n_slot == n_wave` it is exactly break-even (the per-`(e,k)`
 *    accumulation is paid and buys nothing), so the host enables it only when
 *    `2·n_slot <= n_wave`; otherwise the original per-`(k,l,m)` loop runs
 *    unchanged.  Both paths compute the same number (measured ON-vs-OFF:
 *    1.17× at n_slot 1, 1.34× at 25, 1.00× at 100).
 *
 * 4. **`A_{e,p}` cache for the grouped backward launch.**  `resolution_size`
 *    needs two launches (mode 1 = forward, group sums; mode 2 = backward with
 *    the finished `dQ/dP`).  Mode 2 used to re-run the whole forward just to
 *    rebuild `A`, which is the only forward output it consumes.  Mode 1 now
 *    stores `A` (`cap·n_proj` complex, per CHUNK) and mode 2 reads it back,
 *    skipping phase 2 (the projection sum) and phase 3 entirely.  `common =
 *    F/D` would be `ne·n_wave` complex (100 MiB at the benchmark shape) and is
 *    deliberately NOT cached — mode 2 still recomputes `D` in phase 1, which it
 *    needs for `common` and `dQ/dD` anyway.
 *
 * WORKING MEMORY IS BOUNDED BY `batch_size`, NOT BY THE SAMPLE.  The per-EVENT
 * data (`cache` = F·fa, `bas_cache`, `weight`, `bkg`) stays fully resident on
 * the handle — that is the whole sample and it is irreducible.  Everything
 * that is scratch (`P`, saved `A`, `dqp`, the per-group sums) is allocated ONCE
 * on the CONTEXT, sized by `cap` = the chunk length, and reused for every chunk
 * of every call.  `compute` then walks the sample in chunks and passes the
 * per-event pointers OFFSET by the chunk start with the chunk length as the
 * kernel's `ne`, so the kernels' own `e0 + e` / `ge` are already chunk-local
 * and **no `__global__` body changed**.  A chunk always holds WHOLE log-sum
 * groups (`cap` is `batch_size` rounded down to a multiple of `rsize`), so the
 * entire grouped pipeline runs inside one chunk.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>

#define GP_BLOCK 256
#define GP_TE    8            /* events per tile */
/*: deepest factor list the fully parallel coeff-gradient pass handles; past
 *  this the kernel falls back to the per-wave shape (still correct, just not
 *  spread over the whole block), and the slot collapse is forced off (the
 *  fallback has no `dsum` buffer). 16 covers every real config by a wide
 *  margin (the fixtures use 1-2). */
#define GP_MAXRES 16
/*: projections handled per thread in phase 2; past this the kernel keeps the
 *  one-thread-per-(e,p) shape (a larger per-thread accumulator array would
 *  spill to local memory). */
#define GP_MAXPROJ 4

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                \
        }                                                                    \
    } while (0)

/* ── Catmull-Rom (the repo-wide idiom) ────────────────────────────────── */
__device__ double gp2_cr1d(double pm1, double p0, double p1, double p2,
                           double t) {
    return p0 + 0.5 * t * (-pm1 + p1 + t * (2.0 * pm1 - 5.0 * p0 + 4.0 * p1
                                            - p2 + t * (-pm1 + 3.0 * p0
                                                        - 3.0 * p1 + p2)));
}

__device__ double gp2_interp_real(const double* __restrict__ t, int row,
                                  double x, double xmin, double xdelta,
                                  int bins) {
    double diff = (x - xmin) / xdelta;
    int b = max(0, min((int)floor(diff), bins - 2));
    double s = max(0.0, min(diff - b, 1.0));
    int base = row * bins + b;
    int lo = row * bins, hi = (row + 1) * bins - 1;
    int im1 = base > lo ? base - 1 : base;
    int i2 = base + 2 <= hi ? base + 2 : base + 1;
    return gp2_cr1d(t[im1], t[base], t[base + 1], t[i2], s);
}

__device__ void gp2_interp_c(const double* __restrict__ tr,
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
    *orr = gp2_cr1d(tr[im1], tr[base], tr[base + 1], tr[i2], s);
    *oii = gp2_cr1d(ti[im1], ti[base], ti[base + 1], ti[i2], s);
}

/* ── load-time cache fill: F_{e,k} = fl · Π num (no ck, no coeff) ────── */
__global__ void gp2_fill_kernel(
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
        fl *= gp2_interp_real(fl_table, k * n_decay + d, qv,
                              q_min, q_delta, q_bins);
    }
    double nr = 1.0, ni = 0.0;
    for (int r = 0; r < n_res; r++) {
        const int mc = num_m_index[k * n_res + r];
        const double mv = mass[(size_t)e * n_mass + mc];
        double vr, vi;
        gp2_interp_c(num_r, num_i, k * n_res + r, mv,
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

/* ── load-time fill of the two coeff/ck-independent caches ─────────────
 *
 * ``basis_m(m_e)`` depends on (m, e) only and ``fa_{e,k,p}`` on (e, k, p)
 * only, so both are fixed for the whole fit.
 */
__global__ void gp2_fill_static_kernel(
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
        gp2_interp_c(bas_r, bas_i, m, mv, m_min, m_delta, m_bins, &vr, &vi);
        bas_cache[g] = make_float2((float)vr, (float)vi);
    }
}

/*: The largest chunk `create` will reserve scratch for.  `create` does not
 *  know the sample length (`ne` arrives with `load_data`), so the scratch is
 *  sized from `batch_size` — and an unbounded `batch_size` (a caller passing
 *  10**9 to mean "one chunk", which is a legitimate and tested setting) would
 *  then reserve tens of GB for a sample far smaller than that.  One million
 *  events is ~58 MB of scratch at `n_proj = 2` and larger than any sample this
 *  kernel is used on, so the ceiling costs nothing and makes `batch_size` a
 *  *hint* above it rather than a memory demand. */
#define GP_MAX_CAP (1 << 20)

/* ── the tile's shared-memory layout, in doubles ───────────────────────
 *
 * ONE formula, used by the host (`smem_bytes`) and by the kernel (the pointer
 * prefixes), so the launch size and the kernel's view can never disagree.
 *
 * `nres_t` is the compile-time factor count of the instantiation (1 for the
 * `NR == 1` specialisation, `n_res` otherwise), `collapsed` the slot-collapse
 * decision and `NS` the slot count.  Buffers:
 *
 *   s      [TE·W·NRES·2]  per-factor denominator products (SKIPPED at nres_t=1)
 *   D      [TE·W·2]       the finished denominator D_{e,k}
 *   rd     [TE·W]         1/|D|²   (one real reciprocal, all divisions become
 *                         multiplies — the biggest single win in the base)
 *   A      [TE·P·2]       A_{e,p}
 *   dA     [TE·P·2]       dQ/dA_{e,p}
 *   pe     [TE]           P_e
 *   gc     [4·NB]         per-coefficient shared accumulator
 *   gck    [2·W]          per-wave shared accumulator for dQ/dck
 *   pv     dQ/ds_{e,k,l}  [TE·W·NRES·2]   (NOT collapsed)
 *          or dQ/dD_{e,k} [TE·W·2]        (collapsed — the slot sum's input)
 *   dd     [TE·NS·NRES·2] Σ_{k∈slot} dQ/dD_{e,k}   (only when collapsed)
 *   w      [TE·NS·NRES·2] Π_{l'≠l} s_{e,slot,l'}    (collapsed and NRES > 1)
 */
typedef struct {
    int s, D, rd, A, dA, pe, gc, gck, pv, dd, w, total;
} GPLayout;

__host__ __device__ __forceinline__ GPLayout gp2_layout(
    int nres_t, int collapsed, int W, int P, int NB, int NS)
{
    GPLayout L;
    int o = 0;
    L.s   = o; o += (nres_t == 1) ? 0 : 2 * GP_TE * W * nres_t;
    L.D   = o; o += 2 * GP_TE * W;
    L.rd  = o; o += GP_TE * W;
    L.A   = o; o += 2 * GP_TE * P;
    L.dA  = o; o += 2 * GP_TE * P;
    L.pe  = o; o += GP_TE;
    L.gc  = o; o += 4 * NB;
    L.gck = o; o += 2 * W;
    /* dQ/ds is per (e,k,l); the collapsed form only needs dQ/dD per (e,k) */
    L.pv  = o; o += collapsed ? 2 * GP_TE * W : 2 * GP_TE * W * nres_t;
    L.dd  = o; o += collapsed ? 2 * GP_TE * (NS > 0 ? NS : 1) : 0;
    L.w   = o; o += (collapsed && nres_t > 1)
                   ? 2 * GP_TE * (NS > 0 ? NS : 1) * nres_t : 0;
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
 * only after every tile has run, which is what ``gp2_grp_nll_kernel`` +
 * ``gp2_scale_kernel`` provide.
 */
template <int NR>                    /* NR = n_res, or 0 for the dynamic path */
__global__ void gp2_tile_kernel(
    const float2* __restrict__ cache,   /* [ne, n_wave, n_proj]  F·fa  */
    const float2* __restrict__ bas_cache,  /* [ne, n_basis]    basis_m(m) */
    const double* __restrict__ gamma,      /* [n_wave, n_res, n_basis]    */
    const double* __restrict__ gamma_slot, /* [n_slot, n_res, n_basis]    */
    const int* __restrict__ slot_of_wave,  /* [n_wave]                    */
    const int* __restrict__ slot_rep,      /* [n_slot]  first wave of slot */
    const int* __restrict__ slot_ptr,      /* [n_slot+1]  CSR              */
    const int* __restrict__ slot_waves,    /* [n_wave]    CSR              */
    int use_slots, int n_slot,
    const double* __restrict__ ck_r, const double* __restrict__ ck_i,
    const double* __restrict__ coeff,
    double norm, int use_norm,
    const double* __restrict__ weight, const double* __restrict__ bkg,
    const double* __restrict__ dqp,        /* [ne]  mode 2 only           */
    const double* __restrict__ dA_in,      /* [2·ne·n_proj]  mode 2 only  */
    int rsize, int mode,
    int ne, int n_wave, int n_proj, int n_res, int n_basis,
    double* __restrict__ P_out,
    double* __restrict__ g_ck_r, double* __restrict__ g_ck_i,
    double* __restrict__ g_coeff,
    double* __restrict__ acc,
    double* __restrict__ gS, double* __restrict__ gPb,
    double* __restrict__ gW,
    double* __restrict__ dA_out)           /* [2·ne·n_proj]  mode 1 only  */
{
    const int W = n_wave, P = n_proj, NB = n_basis, NS = n_slot;
    const int NRES = (NR > 0) ? NR : n_res;      /* compile-time at NR == 1 */
    const GPLayout L = gp2_layout(NRES, use_slots, W, P, NB, NS);

    extern __shared__ double sh[];
    double* s_s  = sh + L.s;
    double* s_D  = sh + L.D;
    double* s_rd = sh + L.rd;
    double* s_A  = sh + L.A;
    double* s_dA = sh + L.dA;
    double* s_pe = sh + L.pe;
    double* s_gc = sh + L.gc;
    double* s_gck = sh + L.gck;
    double* s_pv = sh + L.pv;
    double* s_dd = sh + L.dd;
    double* s_w  = sh + L.w;

    const int tid = threadIdx.x, nt = blockDim.x;
    for (int i = tid; i < 4 * NB; i += nt) s_gc[i] = 0.0;
    for (int i = tid; i < 2 * W; i += nt) s_gck[i] = 0.0;
    if (use_slots) {
        const int nd = GP_TE * (NS > 0 ? NS : 1) * 2;
        for (int i = tid; i < nd; i += nt) s_dd[i] = 0.0;
    }
    __syncthreads();
    const int e0 = blockIdx.x * GP_TE;
    const int evn = min(GP_TE, ne - e0);
    if (evn <= 0) return;

    /* ── phase 1: the denominator — a dot product over the CACHED basis ──
     *
     * At NR == 1 there is exactly one factor, so `D` IS that factor: the
     * product bookkeeping is the identity and `s_s` would only ever be
     * written, never read — which is why it is not allocated there. */
    for (int idx = tid; idx < evn * W; idx += nt) {
        const int e = idx / W, k = idx - e * W;
        const float2* bv = bas_cache + (size_t)(e0 + e) * NB;
        double dr = 1.0, di = 0.0;
        for (int l = 0; l < NRES; l++) {
            double sr = 0.0, si = 0.0;
            const int gb = (k * NRES + l) * NB;
            for (int m = 0; m < NB; m++) {
                const double g = gamma[gb + m];
                if (g == 0.0) continue;
                const double w = g * coeff[m];
                sr += w * (double)bv[m].x;
                si += w * (double)bv[m].y;
            }
            if (NR != 1) {                       /* dead at NR == 1 */
                const size_t o = ((size_t)idx * NRES + l) * 2;
                s_s[o + 0] = sr;
                s_s[o + 1] = si;
            }
            const double t_r = dr * sr - di * si;
            di = dr * si + di * sr;
            dr = t_r;
        }
        s_D[idx * 2 + 0] = dr;
        s_D[idx * 2 + 1] = di;
        /* |D|² is a REAL scalar even though the basis (and D) is complex, so
           the shared part of `F/D` and of `dQ/dD` is one real reciprocal.
           fp64 divide is ~1/64 rate and unpipelined on a consumer GPU. */
        s_rd[idx] = 1.0 / (dr * dr + di * di);
    }
    __syncthreads();

    /* ── phase 2: A_{e,p} = Σ_k ck_k · F/D · fa   (fa cached) ────────────
     *
     * Skipped in mode 2: the grouped backward launch reads the `A` that mode 1
     * saved instead of re-running this reduction over `n_wave` (which is the
     * entire reason the grouped path used to cost a second forward pass).
     *
     * The reduction over `k` is spread across the block by `(e, lane)` — NOT
     * `(e, p, lane)` — so one thread owns every projection of its `k` slice.
     * That is what makes the reassociation pay: `D` depends on the EVENT, so
     * `w_{e,k} = ck_k·(F_{e,k}/D_{e,k})` is shared by all projections. */
    if (mode != 2) {
        if (P <= GP_MAXPROJ) {
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
                double ar[GP_MAXPROJ], ai[GP_MAXPROJ];
                for (int p = 0; p < P; p++) { ar[p] = 0.0; ai[p] = 0.0; }
                for (int k = lane; k < W; k += nlane) {
                    /* cache = F·fa; (F/D)·fa = cache/D  (one reciprocal) */
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

    /* ── phase 4 (merged into change 1): u, dQ/dck and dQ/dD ─────────────
     *
     * u_{e,k} = Σ_p dQ/dA_{e,p}·fa_{e,p,k}   (no `ck`: shared by both)
     * dQ/dck_k      += common_{e,k}·u_{e,k}
     * dQ/dcommon_k   = ck_k·u_{e,k}
     * dQ/dD_{e,k}    = -dQ/dcommon_k·common_{e,k}/D_{e,k}
     *
     * `common` is rebuilt here (4 + 2 multiplies from the caches) — the old
     * kernel rebuilt it in BOTH reducing passes; there is only one now.
     * `u` feeds the two lines above inside this iteration, so it stays in
     * registers. */
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
        /* `ur,ui` above already carry F (the cache is F·fa), so this is
           w = u'/D = (F/D)·u — the SAME quantity `common·u` the unfused
           kernel built from F and 1/D separately.  dck is done. */
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
        const double ddr = (qr * dr + qi * di) * rden;
        const double ddi = (qi * dr - qr * di) * rden;
        if (use_slots) {
            /* one dQ/dD per (e,k) — summed over the slot's waves in a
               separate pass below, NOT with a shared atomic (at n_slot == 1
               every thread in the tile hammers the same 8 shared slots, which
               measured 2.25 ms of contention on the benchmark shape). */
            s_pv[(size_t)idx * 2 + 0] = ddr;
            s_pv[(size_t)idx * 2 + 1] = ddi;
        } else {
            /* dQ/ds_{e,k,l} = dQ/dD · Π_{l'≠l} s_{e,k,l'}.  Every value is
               read before anything is written; the prefix product is formed
               in registers. */
            for (int l = 0; l < NRES; l++) {
                double pr = 1.0, pi = 0.0;
                if (NR != 1) {                   /* dead at NR == 1 */
                    for (int l2 = 0; l2 < NRES; l2++) {
                        if (l2 == l) continue;
                        const double ur2 = s_s[(size_t)idx * NRES * 2
                                               + l2 * 2 + 0];
                        const double ui2 = s_s[(size_t)idx * NRES * 2
                                               + l2 * 2 + 1];
                        const double t_r = pr * ur2 - pi * ui2;
                        pi = pr * ui2 + pi * ur2;
                        pr = t_r;
                    }
                }
                s_pv[(size_t)idx * NRES * 2 + l * 2 + 0] = ddr * pr - ddi * pi;
                s_pv[(size_t)idx * NRES * 2 + l * 2 + 1] = ddi * pr + ddr * pi;
            }
        }
    }
    __syncthreads();
    /* ── w_{e,slot,l} = Π_{l'≠l} s_{e,slot,l'}, once per (e,slot,l) ───────
     * Not per wave: that IS the saving of the collapse.  At NRES == 1 the
     * empty product is 1 and nothing is stored.  `slot_rep` names a wave of
     * the slot, whose `s` is the slot's `s` by construction. */
    if (use_slots) {
        /* dsum_{e,slot} = Σ_{k∈slot} dQ/dD_{e,k}.  One thread per (e,slot)
           reads its slot's waves from the CSR list — no shared atomics, so no
           contention when a slot holds most of the waves (the n_slot == 1
           shape, where a contended atomicAdd measured 2.25 ms of 6.3). */
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
    }
    if (use_slots && NRES > 1) {
        for (int i = tid; i < evn * NS * NRES; i += nt) {
            const int l = i % NRES;
            const int t = i / NRES;
            const int sl = t % NS, e = t / NS;
            const int rep = slot_rep[sl];
            const size_t sb = (size_t)(e * W + rep) * NRES * 2;
            double pr = 1.0, pi = 0.0;
            for (int l2 = 0; l2 < NRES; l2++) {
                if (l2 == l) continue;
                const double ur2 = s_s[sb + l2 * 2 + 0];
                const double ui2 = s_s[sb + l2 * 2 + 1];
                const double t_r = pr * ur2 - pi * ui2;
                pi = pr * ui2 + pi * ur2;
                pr = t_r;
            }
            const size_t o = ((size_t)(e * NS + sl) * NRES + l) * 2;
            s_w[o + 0] = pr;
            s_w[o + 1] = pi;
        }
    }
    __syncthreads();

    /* one global atomic per (wave, block) for dQ/dck, as before */
    for (int k = tid; k < W; k += nt) {
        atomicAdd(g_ck_r + k, s_gck[2 * k + 0]);
        atomicAdd(g_ck_i + k, s_gck[2 * k + 1]);
    }

    /* ── phase 5: the coefficient gradient ────────────────────────────────
       dQ/dcoeff_m += 2Re Σ_e Σ_l dQ/ds_{e,k,l}·gamma[k,l,m]·basis_m(m_e)

       Two shapes, same number: per-(k,l,m) (the base kernel's) or, when the
       2·n_slot <= n_wave gate holds, per-(slot,l,m).  Both reduce the tile in
       shared `s_gc` so a block contributes one atomic per coefficient. */
    if (n_res <= GP_MAXRES) {
        if (use_slots) {
            for (int idx = tid; idx < NS * NRES * NB; idx += nt) {
                const int m = idx % NB;
                const int t = idx / NB;
                const int l = t % NRES;
                const int sl = t / NRES;
                const double g = gamma_slot[(sl * NRES + l) * NB + m];
                if (g == 0.0) continue;
                double out = 0.0;
                for (int e = 0; e < evn; e++) {
                    const float2 bv = bas_cache[(size_t)(e0 + e) * NB + m];
                    const double dr = s_dd[(e * NS + sl) * 2 + 0];
                    const double di = s_dd[(e * NS + sl) * 2 + 1];
                    double ur = dr, ui = di;
                    if (NRES > 1) {
                        const size_t wo = ((size_t)(e * NS + sl) * NRES + l) * 2;
                        const double wr = s_w[wo + 0], wi = s_w[wo + 1];
                        ur = wr * dr - wi * di;
                        ui = wr * di + wi * dr;
                    }
                    out += 2.0 * (ur * (double)bv.x - ui * (double)bv.y);
                }
                atomicAdd(&s_gc[m], out * g);
            }
        } else {
            for (int idx = tid; idx < W * NRES * NB; idx += nt) {
                const int m = idx % NB;
                const int t = idx / NB;
                const int l = t % NRES;
                const int k = t / NRES;
                const double g = gamma[(k * NRES + l) * NB + m];
                if (g == 0.0) continue;
                double out = 0.0;
                for (int e = 0; e < evn; e++) {
                    const float2 bv = bas_cache[(size_t)(e0 + e) * NB + m];
                    const size_t o = ((size_t)(e * W + k) * NRES + l) * 2;
                    out += 2.0 * (s_pv[o + 0] * (double)bv.x
                                  - s_pv[o + 1] * (double)bv.y);
                }
                atomicAdd(&s_gc[m], out * g);
            }
        }
        __syncthreads();
        for (int m = tid; m < NB; m += nt)
            if (s_gc[m] != 0.0) atomicAdd(g_coeff + m, s_gc[m]);
    } else {
        /* Fallback for an unusually deep factor list (n_res > GP_MAXRES): the
           original per-wave shape, kept so a large config stays correct rather
           than silently truncated.  The host forces the slot collapse off
           there, so this only ever sees the per-(k,l,m) form. */
        for (int k = tid; k < W; k += nt) {
            for (int l = 0; l < NRES; l++) {
                double gsr[GP_TE], gsi[GP_TE];
                for (int e = 0; e < evn; e++) {
                    const int ge = e0 + e;
                    double dcr = 0.0, dci = 0.0;
                    for (int p = 0; p < P; p++) {
                        const float2 Ff =
                            cache[(size_t)ge * W * P + (size_t)k * P + p];
                        const double gar = s_dA[(e * P + p) * 2 + 0];
                        const double gai = s_dA[(e * P + p) * 2 + 1];
                        const double tr = gar * (double)Ff.x
                                          - gai * (double)Ff.y;
                        const double ti = gar * (double)Ff.y
                                          + gai * (double)Ff.x;
                        dcr += ck_r[k] * tr - ck_i[k] * ti;
                        dci += ck_r[k] * ti + ck_i[k] * tr;
                    }
                    const double dr = s_D[(e * W + k) * 2 + 0];
                    const double di = s_D[(e * W + k) * 2 + 1];
                    const double rden = s_rd[e * W + k];
                    /* dcr already carries F, so this chain needs F/D²:
                       two reciprocals, exactly as in site B. */
                    const double qr0 = -(dcr * dr + dci * di) * rden;
                    const double qi0 = -(dci * dr - dcr * di) * rden;
                    const double qr = (qr0 * dr + qi0 * di) * rden;
                    const double qi = (qi0 * dr - qr0 * di) * rden;
                    const double ddr = (qr * dr + qi * di) * rden;
                    const double ddi = (qi * dr - qr * di) * rden;
                    double pr = 1.0, pi = 0.0;
                    for (int l2 = 0; l2 < NRES; l2++) {
                        if (l2 == l) continue;
                        const double ur =
                            s_s[((size_t)(e * W + k) * NRES + l2) * 2 + 0];
                        const double ui =
                            s_s[((size_t)(e * W + k) * NRES + l2) * 2 + 1];
                        const double t_r = pr * ur - pi * ui;
                        pi = pr * ui + pi * ur;
                        pr = t_r;
                    }
                    gsr[e] = ddr * pr - ddi * pi;
                    gsi[e] = ddi * pr + ddr * pi;
                }
                for (int m = 0; m < NB; m++) {
                    const double g = gamma[(k * NRES + l) * NB + m];
                    if (g == 0.0) continue;
                    double out = 0.0;
                    for (int e = 0; e < evn; e++) {
                        const float2 bv =
                            bas_cache[(size_t)(e0 + e) * NB + m];
                        out += 2.0 * (gsr[e] * (double)bv.x
                                      - gsi[e] * (double)bv.y);
                    }
                    atomicAdd(g_coeff + m, out * g);
                }
            }
        }
    }
}

/* ── grouped log-sum: the per-group sums -> Q, dnorm and the scale ──────
 *
 *     W_g = Σ w,  S_g = Σ w·(P/norm + bkg),  Pb_g = Σ w·P
 *     Q      += -W_g·log(S_g/W_g)            (only when W_g > 0)
 *     dnorm  +=  W_g·Pb_g / (S_g·norm²)
 *     scale_g = -W_g/S_g                     (consumed by gp2_scale_kernel)
 */
__global__ void gp2_grp_nll_kernel(
    const double* __restrict__ gS, const double* __restrict__ gPb,
    const double* __restrict__ gW, int n_groups, double norm,
    double* __restrict__ gscale, double* __restrict__ acc)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    const double W = gW[g], S = gS[g];
    if (W <= 0.0) {
        gscale[g] = 0.0;
        return;
    }
    gscale[g] = -W / S;
    atomicAdd(acc + 0, -W * log(S / W));
    atomicAdd(acc + 1, W * gPb[g] / (S * norm * norm));
}

/* ── dQ/dP_e = scale_{g(e)} · w_e/norm, one thread per event ──────────── */
__global__ void gp2_scale_kernel(
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
    int *slot_of_wave, *slot_rep, *slot_ptr, *slot_waves;
    double *fl_table, *num_r, *num_i, *bas_r, *bas_i, *gamma, *gamma_slot;
    double *angle_k, *angle_b, *am_r, *am_i;
    double m_min, m_delta, q_min, q_delta;
    int m_bins, q_bins;
    int n_wave, n_proj, n_res, n_decay, n_basis, n_ab, n_comp;
    int n_mass, n_momentum, n_angle_total;
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
    int cap;              /* events per chunk, aligned to rsize; 0 = no cap */
    double *s_P, *h_sP;   /* per-chunk P (device + pinned host) */
    double *s_dA;         /* per-chunk saved A   (rsize > 1) */
    double *s_dqp;        /* per-chunk dQ/dP scale */
    double *s_gS, *s_gPb, *s_gW, *s_gscale;   /* per-chunk group arrays */
    int s_ngroups;        /* groups a chunk can hold */
} GPCtx;

/* ONE SAMPLE.  Everything sized by ``ne`` lives here, so two handles (the
   fitter's phase space and its data) never alias. */
typedef struct {
    double *weight, *bkg;   /* per-EVENT data, full length ``ne`` */
    float2 *cache;        /* F·fa                [ne, n_wave, n_proj]    */
    float2 *bas_cache;    /* basis_m(m_e)        [ne, n_basis]           */
    int ne;               /* events in THIS sample */
    int rsize;
    /* NOTHING else is ne-sized any more: P, dA, dqp and the group arrays are
       batch-sized scratch on the CONTEXT (see GPCtx).  A chunk is a whole
       number of log-sum groups, so the pipeline runs per chunk and only the
       tiny global accumulators (acc / g_ck / g_coeff) span the whole sample. */
} GPDataHandle;

static int* _gp_up_i(const int* src, int n) {
    if (n <= 0) return NULL;
    int* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(int),
                          cudaMemcpyHostToDevice));
    return d;
}

static double* _gp_up_d(const double* src, int n) {
    if (n <= 0) return NULL;
    double* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(double),
                          cudaMemcpyHostToDevice));
    return d;
}

/* The one launch, dispatched on the `NR` instantiation.
 *
 * BATCHING NEEDS NO KERNEL CHANGE.  The chunk is expressed purely as POINTER
 * OFFSETS into the per-event data plus a chunk length passed as the kernel's
 * `ne`, so the kernel's own `e0 + e` / `ge` are chunk-local by construction:
 *
 *     cache     -> h->cache     + st·n_wave·n_proj
 *     bas_cache -> h->bas_cache + st·n_basis
 *     weight    -> h->weight + st        bkg -> h->bkg + st
 *
 * That is why `gp2_tile_kernel`, `gp2_grp_nll_kernel` and `gp2_scale_kernel`
 * are byte-identical to the pre-batching kernel.  `st` is always a multiple of
 * `rsize` (see the chunk rule in `create`), so the chunk-local group index
 * `ge / rsize` is consistent within a chunk and a group never straddles one.
 */
static void gp2_launch(GPCtx* c, GPDataHandle* h, int st, int nb,
                       double norm, int use_norm,
                       int mode, double* P_out, double* dA_out,
                       double* gS, double* gPb, double* gW)
{
    const int tiles = (nb + GP_TE - 1) / GP_TE;
    const size_t sm = (size_t)c->smem_bytes;
    const double* dA_in = (mode == 2) ? c->s_dA : NULL;
    const float2* cache = h->cache + (size_t)st * c->n_wave * c->n_proj;
    const float2* bas_cache = h->bas_cache + (size_t)st * c->n_basis;
    const double* weight = h->weight + st;
    const double* bkg = h->bkg + st;
    if (c->n_res == 1) {
        gp2_tile_kernel<1><<<tiles, GP_BLOCK, sm, c->stream>>>(
            cache, bas_cache, c->gamma, c->gamma_slot,
            c->slot_of_wave, c->slot_rep, c->slot_ptr, c->slot_waves,
            c->use_slots, c->n_slot,
            c->p_ck_r, c->p_ck_i, c->p_coeff, norm, use_norm,
            weight, bkg, c->s_dqp, dA_in, h->rsize, mode,
            nb, c->n_wave, c->n_proj, c->n_res, c->n_basis,
            P_out, c->g_ck_r, c->g_ck_i, c->g_coeff, c->acc,
            gS, gPb, gW, dA_out);
    } else {
        gp2_tile_kernel<0><<<tiles, GP_BLOCK, sm, c->stream>>>(
            cache, bas_cache, c->gamma, c->gamma_slot,
            c->slot_of_wave, c->slot_rep, c->slot_ptr, c->slot_waves,
            c->use_slots, c->n_slot,
            c->p_ck_r, c->p_ck_i, c->p_coeff, norm, use_norm,
            weight, bkg, c->s_dqp, dA_in, h->rsize, mode,
            nb, c->n_wave, c->n_proj, c->n_res, c->n_basis,
            P_out, c->g_ck_r, c->g_ck_i, c->g_coeff, c->acc,
            gS, gPb, gW, dA_out);
    }
}

extern "C" {

void* cuda_create_context_flat_pwa_gen2(
    const double* fl_table, int n_fl, const int* fl_q_index, int n_fq,
    const double* num_r, const double* num_i, int n_num,
    const int* num_m_index, int n_nm, const double* bas_r,
    const double* bas_i, int n_bas, const int* bas_m_index, int n_bm,
    const double* gamma, int n_g,
    const double* gamma_slot, int n_gs,
    const int* slot_of_wave, int n_sw,
    const int* slot_rep, int n_sr,
    const int* slot_ptr, int n_sp, const int* slot_waves, int n_swv,
    int n_slot, int use_slots,
    const int* angle_index, int n_ai,
    const double* angle_k, int n_ak, const double* angle_b, int n_ab_,
    const double* am_r, const double* am_i, int n_am,
    double m_min, double m_delta, int m_bins,
    double q_min, double q_delta, int q_bins,
    int n_wave, int n_proj, int n_res, int n_decay, int n_basis,
    int n_ab, int n_comp, int n_mass, int n_momentum, int n_angle_total,
    int batch_size, int rsize)
{
    GPCtx* c = (GPCtx*)calloc(1, sizeof(GPCtx));
    if (!c) return NULL;
    c->rsize = (rsize > 0) ? rsize : 1;
    c->fl_table = _gp_up_d(fl_table, n_fl);
    c->fl_q_index = _gp_up_i(fl_q_index, n_fq);
    c->num_r = _gp_up_d(num_r, n_num);
    c->num_i = _gp_up_d(num_i, n_num);
    c->num_m_index = _gp_up_i(num_m_index, n_nm);
    c->bas_r = _gp_up_d(bas_r, n_bas);
    c->bas_i = _gp_up_d(bas_i, n_bas);
    c->bas_m_index = _gp_up_i(bas_m_index, n_bm);
    c->gamma = _gp_up_d(gamma, n_g);
    c->gamma_slot = _gp_up_d(gamma_slot, n_gs);
    c->slot_of_wave = _gp_up_i(slot_of_wave, n_sw);
    c->slot_rep = _gp_up_i(slot_rep, n_sr);
    c->slot_ptr = _gp_up_i(slot_ptr, n_sp);
    c->slot_waves = _gp_up_i(slot_waves, n_swv);
    c->n_slot = (n_slot > 0) ? n_slot : 1;
    c->angle_index = _gp_up_i(angle_index, n_ai);
    c->angle_k = _gp_up_d(angle_k, n_ak);
    c->angle_b = _gp_up_d(angle_b, n_ab_);
    c->am_r = _gp_up_d(am_r, n_am);
    c->am_i = _gp_up_d(am_i, n_am);
    c->m_min = m_min; c->m_delta = m_delta; c->m_bins = m_bins;
    c->q_min = q_min; c->q_delta = q_delta; c->q_bins = q_bins;
    c->n_wave = n_wave; c->n_proj = n_proj; c->n_res = n_res;
    c->n_decay = n_decay; c->n_basis = n_basis; c->n_ab = n_ab;
    c->n_comp = n_comp; c->n_mass = n_mass; c->n_momentum = n_momentum;
    c->n_angle_total = n_angle_total;
    c->batch_size = batch_size;
    /* ── THE CHUNK RULE (copied from `cuda_v5_pwa_s0c`) ────────────────────
     *
     * A chunk must hold WHOLE log-sum groups, so the per-chunk stride is the
     * batch rounded DOWN to a multiple of `rsize`: group `g` then lives
     * entirely inside one chunk (`g = e // rsize` with `e` chunk-local), and
     * the whole grouped pipeline (mode 1 -> grp_nll -> scale -> mode 2) runs
     * inside the chunk — nothing per-group survives a chunk cut and the
     * partition does not depend on `batch_size`.  `cap` is what the scratch
     * below is sized by; `ne` (unknown at create) never enters it.
     *
     * Both inputs are guarded: a non-positive `batch_size` or `rsize` must not
     * produce a zero-length scratch (a zero-event chunk would be an infinite
     * loop in `compute`). */
    c->cap = (c->batch_size > 0) ? c->batch_size : 50000;
    if (c->cap > GP_MAX_CAP) c->cap = GP_MAX_CAP;
    if (c->rsize > 1) {
        const int a = (c->cap / c->rsize) * c->rsize;
        /* `a == 0` means `batch_size < rsize`: a chunk could not hold even ONE
         * whole group, so the alignment would silently leave the cap unaligned
         * and a group would straddle two chunks — measured wrong (`rel` 0.49 vs
         * numpy_flat at rsize 17, batch_size 5/10).  A chunk must hold whole
         * groups, so the cap is raised to one group instead; that is the
         * smallest legal chunk, not a fallback. */
        c->cap = (a > 0) ? a : c->rsize;
    }
    c->s_ngroups = (c->cap + c->rsize - 1) / c->rsize;
    /* the collapse is a per-(k,l,m) -> per-(slot,l,m) reassociation; the
       fallback branch (n_res > GP_MAXRES) has no dsum buffer, so it is off
       there.  The host decides (see the wrapper's `gp2_use_slots`). */
    c->use_slots = (use_slots && n_res <= GP_MAXRES && c->n_slot > 0) ? 1 : 0;

    const int nres_t = (n_res == 1) ? 1 : n_res;
    GPLayout lay = gp2_layout(nres_t, c->use_slots, n_wave, n_proj, n_basis,
                              c->n_slot);
    c->smem_bytes = (int)(lay.total * sizeof(double));
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (c->smem_bytes > optin) {
        fprintf(stderr,
                "cuda_flat_pwa_gen2: shared tile %d B exceeds the "
                "opt-in limit %d B (n_wave %d, n_res %d, n_slot %d)\n",
                c->smem_bytes, optin, n_wave, n_res, c->n_slot);
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
        static size_t opt_in_max[2] = {0, 0};
        const void* fns[2] = {(const void*)gp2_tile_kernel<1>,
                              (const void*)gp2_tile_kernel<0>};
        for (int i = 0; i < 2; i++) {
            if ((size_t)c->smem_bytes <= opt_in_max[i]) continue;
            cudaError_t e = cudaFuncSetAttribute(
                fns[i], cudaFuncAttributeMaxDynamicSharedMemorySize,
                c->smem_bytes);
            if (e != cudaSuccess) {
                fprintf(stderr,
                        "cuda_flat_pwa_gen2: cannot opt in to %d B shared "
                        "memory (%s); n_wave %d, n_res %d\n",
                        c->smem_bytes, cudaGetErrorString(e), n_wave, n_res);
                free(c);
                return NULL;
            }
            opt_in_max[i] = (size_t)c->smem_bytes;
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
    /* ── the batch-sized working scratch, allocated ONCE ───────────────────
     * Sized by `cap` (events per chunk), never by the sample's `ne` — which is
     * unknown here and is the whole point of the change.  Pinned host staging
     * for `P` (it is copied out per chunk), plain device memory for the rest. */
    CUDA_CHECK(cudaMalloc(&c->s_P, (size_t)c->cap * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&c->h_sP, (size_t)c->cap * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->s_dqp, (size_t)c->cap * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->s_gS, (size_t)c->s_ngroups * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->s_gPb, (size_t)c->s_ngroups * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->s_gW, (size_t)c->s_ngroups * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->s_gscale,
                          (size_t)c->s_ngroups * sizeof(double)));
    /* change 4's saved `A`: only the grouped objective consumes it (mode 2),
       so a per-event fit (rsize == 1) never runs that launch and allocates
       nothing here. */
    c->s_dA = NULL;
    if (c->rsize > 1)
        CUDA_CHECK(cudaMalloc(&c->s_dA,
                              (size_t)c->cap * n_proj * 2 * sizeof(double)));
    return c;
}

void* cuda_load_data_flat_pwa_gen2(void* vctx, const double* mass,
                                   const int* nm, const double* mom,
                                   const int* nmom, const double* ang,
                                   const int* nang, const double* weight,
                                   const double* bkg, int ne)
{
    GPCtx* c = (GPCtx*)vctx;
    (void)nm; (void)nmom; (void)nang;
    /* A FRESH handle per load: the fitter holds the phase-space and the data
       handles at the same time, so nothing here may live on the context. */
    GPDataHandle* h = (GPDataHandle*)calloc(1, sizeof(GPDataHandle));
    if (!h) return NULL;
    h->ne = ne;
    h->rsize = (c->rsize > 0) ? c->rsize : 1;
    /* mass / momentum / angles are consumed ONLY by the fill kernels: after
       them the hot path reads F, basis and fa from their caches. */
    double* t_mass = _gp_up_d(mass, ne * c->n_mass);
    double* t_mom = _gp_up_d(mom, ne * c->n_momentum);
    double* t_ang = _gp_up_d(ang, ne * c->n_angle_total * c->n_comp);
    h->weight = _gp_up_d(weight, ne);
    h->bkg = _gp_up_d(bkg, ne);
    CUDA_CHECK(cudaMalloc(&h->cache,
                          (size_t)ne * c->n_wave * c->n_proj
                          * sizeof(float2)));   /* F·fa, per (e,k,p) */
    CUDA_CHECK(cudaMalloc(&h->bas_cache,
                          (size_t)ne * c->n_basis * sizeof(float2)));
    /* NOTHING else is per-sample: P, dA, dqp and the group arrays are the
       CONTEXT's batch-sized scratch (see `create`), so this handle holds only
       the per-EVENT data and its footprint is the whole point of the change. */

    /* F is independent of ck/coeff: fill it ONCE, here */
    const int total = ne * c->n_wave;
    const int threads = GP_BLOCK;
    const int blocks = (total + threads - 1) / threads;
    gp2_fill_kernel<<<blocks, threads, 0, c->stream>>>(
        c->fl_table, c->fl_q_index, c->num_r, c->num_i, c->num_m_index,
        c->m_min, c->m_delta, c->m_bins, c->q_min, c->q_delta, c->q_bins,
        t_mass, t_mom,
        c->angle_index, c->angle_k, c->angle_b, c->am_r, c->am_i,
        t_ang, c->n_ab, c->n_comp, c->n_angle_total,
        ne, c->n_wave, c->n_proj, c->n_res, c->n_decay,
        c->n_mass, c->n_momentum, h->cache);

    /* the two coeff/ck-independent caches: basis_m(m_e) and fa_{e,k,p} */
    const int st_total = ne * c->n_wave * c->n_proj;
    const int st_blocks = (st_total + threads - 1) / threads;
    gp2_fill_static_kernel<<<st_blocks, threads, 0, c->stream>>>(
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
void cuda_free_data_flat_pwa_gen2(void* vh) {
    GPDataHandle* h = (GPDataHandle*)vh;
    if (!h) return;
    if (h->weight) { cudaFree(h->weight); h->weight = NULL; }
    if (h->bkg) { cudaFree(h->bkg); h->bkg = NULL; }
    if (h->cache) { cudaFree(h->cache); h->cache = NULL; }
    if (h->bas_cache) { cudaFree(h->bas_cache); h->bas_cache = NULL; }
    free(h);
}

void cuda_free_context_flat_pwa_gen2(void* vctx) {
    GPCtx* c = (GPCtx*)vctx;
    if (!c) return;
    #define F(p) if (c->p) cudaFree(c->p)
    #define FH(p) if (c->p) cudaFreeHost(c->p)
    F(fl_table); F(fl_q_index); F(num_r); F(num_i); F(num_m_index);
    F(bas_r); F(bas_i); F(bas_m_index); F(gamma); F(gamma_slot);
    F(slot_of_wave); F(slot_rep); F(slot_ptr); F(slot_waves);
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

void cuda_compute_flat_pwa_gen2(
    void* vctx, void* vh, const double* ck_r, const double* ck_i,
    const double* coeff, double norm, int use_norm,
    double* oQ, double* oDn, double* oP,
    double* og_ck_r, double* og_ck_i, double* og_coeff)
{
    GPCtx* c = (GPCtx*)vctx;
    GPDataHandle* h = (GPDataHandle*)vh;
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
     * nothing about P is chunk-sized on the caller's side. */
    const int ne_total = h->ne;
    int bs = c->cap;
    if (bs <= 0) bs = 1;
    if (ne_total < bs) bs = ne_total;
    const int nbat = (ne_total > 0) ? (ne_total + bs - 1) / bs : 0;
    for (int b = 0; b < nbat; b++) {
        const int st = b * bs;
        const int nb = (ne_total - st > bs) ? bs : (ne_total - st);
        /* The grouped log-sum needs the WHOLE group before it can scale
           dQ/dP.  The chunk holds whole groups, so the entire sequence
           (mode 1 -> grp_nll -> scale -> mode 2) runs inside the chunk. */
        const int grouped = (use_norm && h->rsize > 1 && c->s_ngroups > 0
                             && c->s_dA != NULL);
        if (grouped) {
            cudaMemsetAsync(c->s_gS, 0,
                            (size_t)c->s_ngroups * sizeof(double), c->stream);
            cudaMemsetAsync(c->s_gPb, 0,
                            (size_t)c->s_ngroups * sizeof(double), c->stream);
            cudaMemsetAsync(c->s_gW, 0,
                            (size_t)c->s_ngroups * sizeof(double), c->stream);
            gp2_launch(c, h, st, nb, norm, use_norm, 1,
                       oP ? c->s_P : NULL,
                       c->s_dA, c->s_gS, c->s_gPb, c->s_gW);
            CUDA_CHECK(cudaGetLastError());
            const int ngroups = (nb + h->rsize - 1) / h->rsize;
            const int gblk = 256;
            gp2_grp_nll_kernel<<<(ngroups + gblk - 1) / gblk, gblk, 0,
                                 c->stream>>>(
                c->s_gS, c->s_gPb, c->s_gW, ngroups, norm, c->s_gscale,
                c->acc);
            gp2_scale_kernel<<<(nb + gblk - 1) / gblk, gblk, 0, c->stream>>>(
                c->s_gscale, h->weight + st, nb, h->rsize, norm, c->s_dqp);
            CUDA_CHECK(cudaGetLastError());
            gp2_launch(c, h, st, nb, norm, use_norm, 2, NULL, NULL,
                       NULL, NULL, NULL);
            CUDA_CHECK(cudaGetLastError());
        } else {
            gp2_launch(c, h, st, nb, norm, use_norm, 0,
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
 * CHOSEN: the context, not the handle.  The `A` buffer is no longer handle-side
 * (it is batch-sized scratch on the context, allocated once and `ne`-
 * independent), so a handle has no `A` allocation to report.  The symbol name
 * is kept — callers only ever used it to assert the allocation policy — and the
 * wrapper turns this per-chunk figure into a per-sample one (`min(cap, ne)`),
 * which is the sample's own share of the shared scratch. */
long long cuda_handle_dA_bytes_flat_pwa_gen2(void* vctx) {
    GPCtx* c = (GPCtx*)vctx;
    if (!c || c->s_dA == NULL) return 0;
    return (long long)c->cap * c->n_proj * 2 * (long long)sizeof(double);
}

/* The tile size `gp2_layout` gives for an arbitrary shape — the SAME function
 * the kernel and `create` use, exposed so a test can assert the buffer
 * arithmetic without a second, drifting copy in Python. */
long long cuda_tile_bytes_flat_pwa_gen2(int nres_t, int collapsed, int n_wave,
                                        int n_proj, int n_basis, int n_slot) {
    GPLayout L = gp2_layout(nres_t, collapsed, n_wave, n_proj, n_basis,
                            n_slot);
    return (long long)L.total * (long long)sizeof(double);
}

int cuda_get_device_count_flat_pwa_gen2(void) {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
}

/* The dynamic shared-memory size the context was created with.  The host must
 * NOT recompute it: `gp2_layout` is the single definition, and a second copy in
 * the wrapper is exactly the drift that makes a launch silently overrun its
 * tile (measured: a 256 B under-count gave "invalid argument" and Q = 0). */
int cuda_smem_bytes_flat_pwa_gen2(void* vctx) {
    GPCtx* c = (GPCtx*)vctx;
    return c ? c->smem_bytes : 0;
}

/* Events per chunk.  Published for the same reason as `smem_bytes`: the chunk
 * rule (batch rounded down to a whole number of `rsize` groups) must have ONE
 * definition, and it lives in `create`.  The wrapper uses this to report a
 * sample's share of the shared scratch, never to re-derive the rule. */
int cuda_cap_flat_pwa_gen2(void* vctx) {
    GPCtx* c = (GPCtx*)vctx;
    return c ? c->cap : 0;
}

void cuda_get_device_name_flat_pwa_gen2(char* buf, int len) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    snprintf(buf, len, "%s", p.name);
}

}  /* extern "C" */
