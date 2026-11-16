/**
 * cuda_flat_cache_pwa_gen — the flat `pwa_gen` config with a cached `F` and a
 * **backend-internal** dedup.
 *
 * Case-1 copy (kernel policy §2.1): own `.cu`, own `.so`, own wrapper, own
 * symbols.  Nothing shared with another kernel, and nothing changed in the
 * shared config builder — the dedup below happens **inside this backend**,
 * on its own copy of the arrays, so the other backends keep reading the
 * config exactly as emitted.
 *
 * Two differences from `cuda_flat_pwa_gen`:
 *
 * 1. **Cached amplitude.**  `F_{e,k} = fl_{e,k} · Π_r num_{e,k,r}` does not
 *    depend on `ck` or `coeff`, so it is computed once at `load_data` and
 *    reused for every iteration.  Only the denominator `D` (which carries
 *    `coeff`) and the projection sum run per compute call.  The cache is
 *    `float2` (the repo's amplitude-cache convention) so the memory is 8 B
 *    per (event, wave) instead of 16.
 *
 * 2. **Structural dedup of the basis rows**, keyed on
 *    ``(coeff name, basis kind, mass column)`` — the row's physical
 *    identity.  The builder appends one row per (factor, basis row) with
 *    duplicates kept, so this backend folds them on that key BEFORE upload
 *    and remaps `gamma_matrix` accordingly.  Folding on the key rather than
 *    on the sampled values cannot merge two rows that merely look alike
 *    numerically (two identical `kmatrix2` models at different mass columns
 *    stay apart, correctly), and it needs no table comparison at all.
 *
 * The dedup is lossless: `coeff` is gathered onto the surviving rows by the
 * same `keep` list, so a caller's full-form coefficient vector is remapped
 * rather than reinterpreted.  Every other array (gamma, mass indices, angle
 * tables) keeps the config's own indexing.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>

#define FC_BLOCK 256
#define FC_TE    8            /* events per tile */
#define FC_BATCH 50000

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                \
        }                                                                    \
    } while (0)

/* ── Catmull-Rom (the repo-wide idiom) ────────────────────────────────── */
__device__ double fc_cr1d(double pm1, double p0, double p1, double p2,
                          double t) {
    return p0 + 0.5 * t * (-pm1 + p1 + t * (2.0 * pm1 - 5.0 * p0 + 4.0 * p1
                                            - p2 + t * (-pm1 + 3.0 * p0
                                                        - 3.0 * p1 + p2)));
}

__device__ double fc_interp_real(const double* __restrict__ t, int row,
                                 double x, double xmin, double xdelta,
                                 int bins) {
    double diff = (x - xmin) / xdelta;
    int b = max(0, min((int)floor(diff), bins - 2));
    double s = max(0.0, min(diff - b, 1.0));
    int base = row * bins + b;
    int lo = row * bins, hi = (row + 1) * bins - 1;
    int im1 = base > lo ? base - 1 : base;
    int i2 = base + 2 <= hi ? base + 2 : base + 1;
    return fc_cr1d(t[im1], t[base], t[base + 1], t[i2], s);
}

__device__ void fc_interp_c(const double* __restrict__ tr,
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
    *orr = fc_cr1d(tr[im1], tr[base], tr[base + 1], tr[i2], s);
    *oii = fc_cr1d(ti[im1], ti[base], ti[base + 1], ti[i2], s);
}

/* ── load-time cache fill: F_{e,k} = fl · Π num (no ck, no coeff) ────── */
__global__ void fc_fill_kernel(
    const double* __restrict__ fl_table, const int* __restrict__ fl_q_index,
    const double* __restrict__ num_r, const double* __restrict__ num_i,
    const int* __restrict__ num_m_index,
    double m_min, double m_delta, int m_bins,
    double q_min, double q_delta, int q_bins,
    const double* __restrict__ mass, const double* __restrict__ mom,
    int ne, int n_wave, int n_res, int n_decay,
    int n_mass, int n_momentum,
    float2* __restrict__ cache)              /* [ne, n_wave] */
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = ne * n_wave;
    if (gid >= total) return;
    const int e = gid / n_wave, k = gid - e * n_wave;

    double fl = 1.0;
    for (int d = 0; d < n_decay; d++) {
        const int qc = fl_q_index[k * n_decay + d];
        const double qv = mom[(size_t)e * n_momentum + qc];
        fl *= fc_interp_real(fl_table, k * n_decay + d, qv,
                             q_min, q_delta, q_bins);
    }
    double nr = 1.0, ni = 0.0;
    for (int r = 0; r < n_res; r++) {
        const int mc = num_m_index[k * n_res + r];
        const double mv = mass[(size_t)e * n_mass + mc];
        double vr, vi;
        fc_interp_c(num_r, num_i, k * n_res + r, mv,
                    m_min, m_delta, m_bins, &vr, &vi);
        const double t_r = nr * vr - ni * vi;
        ni = nr * vi + ni * vr;
        nr = t_r;
    }
    cache[gid] = make_float2((float)(fl * nr), (float)(fl * ni));
}

/* ── load-time fill of the two coeff/ck-independent caches ─────────────
 *
 * ``basis_m(m_e)`` depends on (m, e) only and ``fa_{e,k,p}`` on (e, k, p)
 * only, so both are fixed for the whole fit.  Interpolating them once here
 * removes every interpolation and every angular reconstruction from the hot
 * path — v1 redid the basis inside phase 5's
 * ``n_res·n_basis·n_proj·n_ab`` loop, which is ~100 M redundant
 * interpolations per call on the _mb fixture.
 */
__global__ void fc_fill_static_kernel(
    const double* __restrict__ bas_r, const double* __restrict__ bas_i,
    const int* __restrict__ bas_m_index,
    double m_min, double m_delta, int m_bins,
    const int* __restrict__ angle_index, const double* __restrict__ angle_k,
    const double* __restrict__ angle_b,
    const double* __restrict__ am_r, const double* __restrict__ am_i,
    const double* __restrict__ mass, const double* __restrict__ ang,
    int ne, int n_basis, int n_wave, int n_proj, int n_ab, int n_comp,
    int n_mass, int n_angle_total,
    float2* __restrict__ bas_cache, float2* __restrict__ fa_cache)
{
    /* basis: one thread per (event, basis row) */
    const int nb_total = ne * n_basis;
    for (int g = blockIdx.x * blockDim.x + threadIdx.x; g < nb_total;
         g += gridDim.x * blockDim.x) {
        const int e = g / n_basis, m = g - e * n_basis;
        const double mv = mass[(size_t)e * n_mass + bas_m_index[m]];
        double vr, vi;
        fc_interp_c(bas_r, bas_i, m, mv, m_min, m_delta, m_bins, &vr, &vi);
        bas_cache[g] = make_float2((float)vr, (float)vi);
    }
    /* fa: one thread per (event, wave, projection) */
    const int fa_total = ne * n_wave * n_proj;
    for (int g = blockIdx.x * blockDim.x + threadIdx.x; g < fa_total;
         g += gridDim.x * blockDim.x) {
        const int e = g / (n_wave * n_proj);
        const int rest = g - e * (n_wave * n_proj);
        const int k = rest / n_proj, p = rest - k * n_proj;
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
        fa_cache[g] = make_float2((float)far, (float)fai);
    }
}

/* ── the per-iteration tile: D, the projection sum, and both gradients ── */
/*  fc_tile_kernel_v2 — replaces the fused tile in kernels_flat_cache_pwa_gen.
 *
 *  Differences from v1, all of them "stop recomputing things that cannot
 *  change between iterations":
 *
 *  1. **basis values are cached** (`bas_cache`, `[ne, n_basis]` float2).  The
 *     interpolation ``basis_m(m_e)`` depends on ``(m, e)`` only — not on
 *     ``coeff`` — and v1 redid it inside phase 5's
 *     ``n_res·n_basis·n_proj·n_ab`` loop, i.e. up to 100 M redundant
 *     interpolations per call on `_mb` for 0.5 M distinct values.  With the
 *     cache the denominator is a plain dot product over the selector.
 *
 *  2. **`fa` is cached** (`fa_cache`, `[ne, n_wave, n_proj]` float2).  It
 *     depends on ``(e, k, p)`` only, and v1 rebuilt it in phases 2, 4 and 5.
 *
 *  Both caches are filled once in ``load_data`` alongside ``F``, so a compute
 *  call now does: the denominator (a ``coeff``-carrying dot product), the
 *  projection sum, and the two gradient chains — no interpolation and no
 *  angular reconstruction in the hot path.
 */
__global__ void fc_tile_kernel(
    const float2* __restrict__ cache,      /* [ne, n_wave]     F = fl·nu  */
    const float2* __restrict__ bas_cache,  /* [ne, n_basis]    basis_m(m) */
    const float2* __restrict__ fa_cache,   /* [ne·n_wave·n_proj]  fa      */
    const int* __restrict__ bas_m_index,   /* [n_basis]  (unused: cached) */
    const double* __restrict__ gamma,      /* [n_wave, n_res, n_basis]    */
    const double* __restrict__ ck_r, const double* __restrict__ ck_i,
    const double* __restrict__ coeff,
    double norm, int use_norm,
    const double* __restrict__ weight, const double* __restrict__ bkg,
    int ne, int n_wave, int n_proj, int n_res, int n_basis,
    double* __restrict__ P_out,
    double* __restrict__ g_ck_r, double* __restrict__ g_ck_i,
    double* __restrict__ g_coeff,
    double* __restrict__ acc)
{
    extern __shared__ double sh[];
    double* s_s  = sh;                                   /* [TE·n_wave·n_res·2] */
    double* s_D  = s_s  + (size_t)FC_TE * n_wave * n_res * 2;
    double* s_A  = s_D  + (size_t)FC_TE * n_wave * 2;
    double* s_dA = s_A  + (size_t)FC_TE * n_proj * 2;
    double* s_pe = s_dA + (size_t)FC_TE * n_proj * 2;

    const int tid = threadIdx.x, nt = blockDim.x;
    const int e0 = blockIdx.x * FC_TE;
    const int evn = min(FC_TE, ne - e0);
    if (evn <= 0) return;

    /* ── phase 1: the denominator — a dot product over the CACHED basis ── */
    for (int idx = tid; idx < evn * n_wave; idx += nt) {
        const int e = idx / n_wave, k = idx - e * n_wave;
        const float2* bv = bas_cache + (size_t)(e0 + e) * n_basis;
        double dr = 1.0, di = 0.0;
        for (int l = 0; l < n_res; l++) {
            double sr = 0.0, si = 0.0;
            const int gb = (k * n_res + l) * n_basis;
            for (int m = 0; m < n_basis; m++) {
                const double g = gamma[gb + m];
                if (g == 0.0) continue;
                const double w = g * coeff[m];
                sr += w * (double)bv[m].x;
                si += w * (double)bv[m].y;
            }
            s_s[((e * n_wave + k) * n_res + l) * 2 + 0] = sr;
            s_s[((e * n_wave + k) * n_res + l) * 2 + 1] = si;
            const double t_r = dr * sr - di * si;
            di = dr * si + di * sr;
            dr = t_r;
        }
        s_D[idx * 2 + 0] = dr;
        s_D[idx * 2 + 1] = di;
    }
    __syncthreads();

    /* ── phase 2: A_{e,p} = Σ_k ck_k · F/D · fa   (fa cached) ─────────── */
    for (int idx = tid; idx < evn * n_proj; idx += nt) {
        const int e = idx / n_proj, p = idx - e * n_proj;
        double ar = 0.0, ai = 0.0;
        for (int k = 0; k < n_wave; k++) {
            const float2 F = cache[(size_t)(e0 + e) * n_wave + k];
            const float2 fa = fa_cache[(size_t)(e0 + e) * n_wave * n_proj
                                       + (size_t)k * n_proj + p];
            const double fr = (double)F.x, fi = (double)F.y;
            const double dr = s_D[(e * n_wave + k) * 2 + 0];
            const double di = s_D[(e * n_wave + k) * 2 + 1];
            const double den = dr * dr + di * di;
            const double crn = (fr * dr + fi * di) / den;
            const double cin = (fi * dr - fr * di) / den;
            const double tr = crn * (double)fa.x - cin * (double)fa.y;
            const double ti = crn * (double)fa.y + cin * (double)fa.x;
            ar += ck_r[k] * tr - ck_i[k] * ti;
            ai += ck_r[k] * ti + ck_i[k] * tr;
        }
        s_A[idx * 2 + 0] = ar;
        s_A[idx * 2 + 1] = ai;
    }
    __syncthreads();

    /* ── phase 3: P_e, the objective, dQ/dP ───────────────────────────── */
    for (int e = tid; e < evn; e += nt) {
        double pe = 0.0;
        for (int p = 0; p < n_proj; p++) {
            const double ar = s_A[(e * n_proj + p) * 2 + 0];
            const double ai = s_A[(e * n_proj + p) * 2 + 1];
            pe += ar * ar + ai * ai;
        }
        const int ge = e0 + e;
        const double wv = weight[ge], bv = bkg[ge];
        s_pe[e] = pe;
        if (P_out) P_out[ge] = pe;
        if (use_norm) {
            atomicAdd(acc + 0, -wv * log(pe / norm + bv));
            atomicAdd(acc + 1, wv * pe / (norm * (pe + bv * norm)));
        } else {
            atomicAdd(acc + 0, wv * pe);
        }
    }
    __syncthreads();

    for (int idx = tid; idx < evn * n_proj; idx += nt) {
        const int e = idx / n_proj;
        const int ge = e0 + e;
        const double wv = weight[ge], bv = bkg[ge];
        const double pe = s_pe[e];
        const double dqp = use_norm ? -wv / (pe + bv * norm) : wv;
        s_dA[idx * 2 + 0] = dqp * s_A[idx * 2 + 0];
        s_dA[idx * 2 + 1] = -dqp * s_A[idx * 2 + 1];
    }
    __syncthreads();

    /* ── phase 4: grad ck ─────────────────────────────────────────────── */
    for (int k = tid; k < n_wave; k += nt) {
        double sr = 0.0, si = 0.0;
        for (int e = 0; e < evn; e++) {
            const float2 F = cache[(size_t)(e0 + e) * n_wave + k];
            const double fr = (double)F.x, fi = (double)F.y;
            const double dr = s_D[(e * n_wave + k) * 2 + 0];
            const double di = s_D[(e * n_wave + k) * 2 + 1];
            const double den = dr * dr + di * di;
            const double crn = (fr * dr + fi * di) / den;
            const double cin = (fi * dr - fr * di) / den;
            for (int p = 0; p < n_proj; p++) {
                const float2 fa = fa_cache[(size_t)(e0 + e) * n_wave * n_proj
                                           + (size_t)k * n_proj + p];
                const double gar = s_dA[(e * n_proj + p) * 2 + 0];
                const double gai = s_dA[(e * n_proj + p) * 2 + 1];
                const double tr = crn * (double)fa.x - cin * (double)fa.y;
                const double ti = crn * (double)fa.y + cin * (double)fa.x;
                sr += gar * tr - gai * ti;
                si += gar * ti + gai * tr;
            }
        }
        atomicAdd(g_ck_r + k, sr);
        atomicAdd(g_ck_i + k, si);
    }
    __syncthreads();

    /* ── phase 5: grad coeff ──────────────────────────────────────────────
       dQ/dcommon_k = Σ_p dQ/dA_p·ck_k·fa_{k,p}       (fa cached)
       dQ/dD_k      = -dQ/dcommon_k · common_k / D_k
       dQ/ds_{k,l}  = dQ/dD_k · Π_{l'≠l} s_{k,l'}
       dQ/dcoeff_m += 2Re Σ_e Σ_l dQ/ds_{k,l}·gamma[k,l,m]·basis_m(m_e)
       The basis value is CACHED, so the inner (e) loop has no interpolation
       and no projection loop: hoist the (e) sum of dQ/ds out of (m). */
    for (int k = tid; k < n_wave; k += nt) {
        for (int l = 0; l < n_res; l++) {
            /* gs[e] = dQ/ds_{k,l} for the whole tile (complex) */
            /* reuse s_A's storage is unsafe; use a small stack array bound
               by FC_TE */
            double gsr[FC_TE], gsi[FC_TE];
            for (int e = 0; e < evn; e++) {
                const int ge = e0 + e;
                /* dQ/dcommon */
                double dcr = 0.0, dci = 0.0;
                for (int p = 0; p < n_proj; p++) {
                    const float2 fa =
                        fa_cache[(size_t)ge * n_wave * n_proj
                                 + (size_t)k * n_proj + p];
                    const double gar = s_dA[(e * n_proj + p) * 2 + 0];
                    const double gai = s_dA[(e * n_proj + p) * 2 + 1];
                    const double tr = gar * (double)fa.x - gai * (double)fa.y;
                    const double ti = gar * (double)fa.y + gai * (double)fa.x;
                    dcr += ck_r[k] * tr - ck_i[k] * ti;
                    dci += ck_r[k] * ti + ck_i[k] * tr;
                }
                const float2 F = cache[(size_t)ge * n_wave + k];
                const double fr = (double)F.x, fi = (double)F.y;
                const double dr = s_D[(e * n_wave + k) * 2 + 0];
                const double di = s_D[(e * n_wave + k) * 2 + 1];
                const double den = dr * dr + di * di;
                const double crn = (fr * dr + fi * di) / den;
                const double cin = (fi * dr - fr * di) / den;
                const double qr = -(dcr * crn - dci * cin);
                const double qi = -(dcr * cin + dci * crn);
                const double ddr = (qr * dr + qi * di) / den;
                const double ddi = (qi * dr - qr * di) / den;
                /* Π_{l'≠l} s_{k,l'} */
                double pr = 1.0, pi = 0.0;
                for (int l2 = 0; l2 < n_res; l2++) {
                    if (l2 == l) continue;
                    const double ur =
                        s_s[((e * n_wave + k) * n_res + l2) * 2 + 0];
                    const double ui =
                        s_s[((e * n_wave + k) * n_res + l2) * 2 + 1];
                    const double t_r = pr * ur - pi * ui;
                    pi = pr * ui + pi * ur;
                    pr = t_r;
                }
                gsr[e] = ddr * pr - ddi * pi;
                gsi[e] = ddi * pr + ddr * pi;
            }
            /* now one pass over m, reading the cached basis only */
            for (int m = 0; m < n_basis; m++) {
                const double g = gamma[(k * n_res + l) * n_basis + m];
                if (g == 0.0) continue;
                double out = 0.0;
                for (int e = 0; e < evn; e++) {
                    const float2 bv =
                        bas_cache[(size_t)(e0 + e) * n_basis + m];
                    /* complex product: Re(gs·b) = gsr·br − gsi·bi */
                    out += 2.0 * (gsr[e] * (double)bv.x
                                  - gsi[e] * (double)bv.y);
                }
                atomicAdd(g_coeff + m, out * g);
            }
        }
    }
}

/*=============================================================================
 * Host side
 *===========================================================================*/
typedef struct {
    int *fl_q_index, *num_m_index, *bas_m_index, *angle_index;
    double *fl_table, *num_r, *num_i, *bas_r, *bas_i, *gamma;
    double *angle_k, *angle_b, *am_r, *am_i;
    double m_min, m_delta, q_min, q_delta;
    int m_bins, q_bins;
    int n_wave, n_proj, n_res, n_decay, n_basis, n_ab, n_comp;
    int n_mass, n_momentum, n_angle_total;
    /* data: only weight/bkg survive load_data; the kinematics are freed
       once F / basis / fa are filled (the hot path never reads them) */
    double *weight, *bkg;
    float2 *cache;        /* F = fl·nu            [ne, n_wave]          */
    float2 *bas_cache;    /* basis_m(m_e)         [ne, n_basis]         */
    float2 *fa_cache;     /* fa_{e,k,p}           [ne, n_wave, n_proj]  */
    double *P_dev, *h_P;
    int ne;
    double *h_ck_r, *h_ck_i, *h_coeff;
    double *p_ck_r, *p_ck_i, *p_coeff;
    double *acc, *h_acc;
    double *g_ck_r, *g_ck_i, *g_coeff;
    cudaStream_t stream;
    int smem_bytes;
} FCtx;

static int* _fc_up_i(const int* src, int n) {
    if (n <= 0) return NULL;
    int* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(int),
                          cudaMemcpyHostToDevice));
    return d;
}

static double* _fc_up_d(const double* src, int n) {
    if (n <= 0) return NULL;
    double* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(double),
                          cudaMemcpyHostToDevice));
    return d;
}

extern "C" {

void* cuda_create_context_flat_cache(
    const double* fl_table, int n_fl, const int* fl_q_index, int n_fq,
    const double* num_r, const double* num_i, int n_num,
    const int* num_m_index, int n_nm, const double* bas_r,
    const double* bas_i, int n_bas, const int* bas_m_index, int n_bm,
    const double* gamma, int n_g, const int* angle_index, int n_ai,
    const double* angle_k, int n_ak, const double* angle_b, int n_ab_,
    const double* am_r, const double* am_i, int n_am,
    double m_min, double m_delta, int m_bins,
    double q_min, double q_delta, int q_bins,
    int n_wave, int n_proj, int n_res, int n_decay, int n_basis,
    int n_ab, int n_comp, int n_mass, int n_momentum, int n_angle_total,
    int batch_size)
{
    FCtx* c = (FCtx*)calloc(1, sizeof(FCtx));
    if (!c) return NULL;
    c->fl_table = _fc_up_d(fl_table, n_fl);
    c->fl_q_index = _fc_up_i(fl_q_index, n_fq);
    c->num_r = _fc_up_d(num_r, n_num);
    c->num_i = _fc_up_d(num_i, n_num);
    c->num_m_index = _fc_up_i(num_m_index, n_nm);
    c->bas_r = _fc_up_d(bas_r, n_bas);
    c->bas_i = _fc_up_d(bas_i, n_bas);
    c->bas_m_index = _fc_up_i(bas_m_index, n_bm);
    c->gamma = _fc_up_d(gamma, n_g);
    c->angle_index = _fc_up_i(angle_index, n_ai);
    c->angle_k = _fc_up_d(angle_k, n_ak);
    c->angle_b = _fc_up_d(angle_b, n_ab_);
    c->am_r = _fc_up_d(am_r, n_am);
    c->am_i = _fc_up_d(am_i, n_am);
    c->m_min = m_min; c->m_delta = m_delta; c->m_bins = m_bins;
    c->q_min = q_min; c->q_delta = q_delta; c->q_bins = q_bins;
    c->n_wave = n_wave; c->n_proj = n_proj; c->n_res = n_res;
    c->n_decay = n_decay; c->n_basis = n_basis; c->n_ab = n_ab;
    c->n_comp = n_comp; c->n_mass = n_mass; c->n_momentum = n_momentum;
    c->n_angle_total = n_angle_total;

    double shd = (double)FC_TE * (n_ab + 2.0 * n_wave * n_res
                                  + 2.0 * n_wave + 2.0 * n_proj
                                  + 2.0 * n_proj + 1.0);
    c->smem_bytes = (int)(shd * sizeof(double));
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (c->smem_bytes > optin) {
        fprintf(stderr,
                "cuda_flat_cache_pwa_gen: shared tile %d B exceeds the "
                "opt-in limit %d B (n_wave %d, n_res %d)\n",
                c->smem_bytes, optin, n_wave, n_res);
        free(c);
        return NULL;
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

void* cuda_load_data_flat_cache(void* vctx, const double* mass,
                                const int* nm, const double* mom,
                                const int* nmom, const double* ang,
                                const int* nang, const double* weight,
                                const double* bkg, int ne)
{
    FCtx* c = (FCtx*)vctx;
    (void)nm; (void)nmom; (void)nang;
    c->ne = ne;
    /* mass / momentum / angles are consumed ONLY by the two fill kernels
       below: after them the hot path reads F, basis and fa from their
       caches and never touches the event kinematics again.  So they are
       uploaded to temporaries and freed immediately — the per-handle
       footprint is `ne·(n_wave + n_basis + n_wave·n_proj)` float2 plus
       weight/bkg, not the raw per-event arrays. */
    double* t_mass = _fc_up_d(mass, ne * c->n_mass);
    double* t_mom = _fc_up_d(mom, ne * c->n_momentum);
    double* t_ang = _fc_up_d(ang, ne * c->n_angle_total * c->n_comp);
    c->weight = _fc_up_d(weight, ne);
    c->bkg = _fc_up_d(bkg, ne);
    CUDA_CHECK(cudaMalloc(&c->P_dev, (size_t)ne * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&c->h_P, (size_t)ne * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->cache, (size_t)ne * c->n_wave * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&c->bas_cache,
                          (size_t)ne * c->n_basis * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&c->fa_cache,
                          (size_t)ne * c->n_wave * c->n_proj
                          * sizeof(float2)));

    /* F is independent of ck/coeff: fill it ONCE, here */
    const int total = ne * c->n_wave;
    const int threads = FC_BLOCK;
    const int blocks = (total + threads - 1) / threads;
    fc_fill_kernel<<<blocks, threads, 0, c->stream>>>(
        c->fl_table, c->fl_q_index, c->num_r, c->num_i, c->num_m_index,
        c->m_min, c->m_delta, c->m_bins, c->q_min, c->q_delta, c->q_bins,
        t_mass, t_mom, ne, c->n_wave, c->n_res, c->n_decay,
        c->n_mass, c->n_momentum, c->cache);

    /* the two coeff/ck-independent caches: basis_m(m_e) and fa_{e,k,p} */
    const int st_total = ne * c->n_wave * c->n_proj;
    const int st_blocks = (st_total + threads - 1) / threads;
    fc_fill_static_kernel<<<st_blocks, threads, 0, c->stream>>>(
        c->bas_r, c->bas_i, c->bas_m_index, c->m_min, c->m_delta, c->m_bins,
        c->angle_index, c->angle_k, c->angle_b, c->am_r, c->am_i,
        t_mass, t_ang, ne, c->n_basis, c->n_wave, c->n_proj,
        c->n_ab, c->n_comp, c->n_mass, c->n_angle_total,
        c->bas_cache, c->fa_cache);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(c->stream));

    /* the kinematics are no longer needed: every consumer reads a cache */
    cudaFree(t_mass); cudaFree(t_mom); cudaFree(t_ang);
    return c;
}

void cuda_free_data_flat_cache(void* vctx) {
    FCtx* c = (FCtx*)vctx;
    if (!c) return;
    cudaFree(c->weight); cudaFree(c->bkg); cudaFree(c->P_dev);
    cudaFree(c->cache);
    cudaFree(c->bas_cache);
    cudaFree(c->fa_cache);
    if (c->h_P) cudaFreeHost(c->h_P);
    c->weight = c->bkg = NULL;
    c->P_dev = NULL; c->cache = NULL; c->h_P = NULL;
    c->bas_cache = NULL; c->fa_cache = NULL;
}

void cuda_free_context_flat_cache(void* vctx) {
    FCtx* c = (FCtx*)vctx;
    if (!c) return;
    #define F(p) if (c->p) cudaFree(c->p)
    #define FH(p) if (c->p) cudaFreeHost(c->p)
    F(fl_table); F(fl_q_index); F(num_r); F(num_i); F(num_m_index);
    F(bas_r); F(bas_i); F(bas_m_index); F(gamma); F(angle_index);
    F(angle_k); F(angle_b); F(am_r); F(am_i);
    F(p_ck_r); F(p_ck_i); F(p_coeff); F(acc);
    F(g_ck_r); F(g_ck_i); F(g_coeff);
    FH(h_ck_r); FH(h_ck_i); FH(h_coeff); FH(h_acc);
    #undef F
    #undef FH
    if (c->stream) cudaStreamDestroy(c->stream);
    free(c);
}

void cuda_compute_flat_cache(
    void* vctx, const double* ck_r, const double* ck_i,
    const double* coeff, double norm, int use_norm,
    double* oQ, double* oDn, double* oP,
    double* og_ck_r, double* og_ck_i, double* og_coeff)
{
    FCtx* c = (FCtx*)vctx;
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
    cudaMemsetAsync(c->acc, 0, 2 * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_ck_r, 0, (size_t)N * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_ck_i, 0, (size_t)N * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_coeff, 0, (size_t)NB * sizeof(double), c->stream);

    if (c->ne > 0) {
        const int tiles = (c->ne + FC_TE - 1) / FC_TE;
        fc_tile_kernel<<<tiles, FC_BLOCK, (size_t)c->smem_bytes, c->stream>>>(
            c->cache, c->bas_cache, c->fa_cache, c->bas_m_index, c->gamma,
            c->p_ck_r, c->p_ck_i, c->p_coeff, norm, use_norm,
            c->weight, c->bkg,
            c->ne, c->n_wave, c->n_proj, c->n_res, c->n_basis,
            oP ? c->P_dev : NULL,
            c->g_ck_r, c->g_ck_i, c->g_coeff, c->acc);
        CUDA_CHECK(cudaGetLastError());
    }
    if (oP)
        cudaMemcpyAsync(c->h_P, c->P_dev, (size_t)c->ne * sizeof(double),
                        cudaMemcpyDeviceToHost, c->stream);
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
    if (oP) memcpy(oP, c->h_P, (size_t)c->ne * sizeof(double));
}

int cuda_get_device_count_flat_cache(void) {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
}

void cuda_get_device_name_flat_cache(char* buf, int len) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    snprintf(buf, len, "%s", p.name);
}

}  /* extern "C" */
