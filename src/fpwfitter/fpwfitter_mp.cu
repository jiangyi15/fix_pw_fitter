/* ================================================================
   fpwfitter_mp.cu  –  Mixed-Precision Fitter (FP32 compute, FP64 I/O)
   ================================================================
   F stored as (KC, N, JP) for coalesced reads.
   All internal arithmetic in FP32 (float/float2).
   Interface accepts/returns FP64 — converted on entry/exit.
   Uses __ldg() for read-only loads on Ampere.
   ================================================================ */

#include "fpwfitter_mp.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ==================== complex helpers (FP32) ==================== */
static __host__ __device__ __forceinline__ float2 cset(float r, float i) {
    return make_float2(r, i);
}
static __host__ __device__ __forceinline__ float cabssq(float2 a) {
    return a.x * a.x + a.y * a.y;
}

/* ==================== launch config ============================= */
static void lcfg(int64_t n, int *nb, int *nt) {
    *nt = 256;
    *nb = (int)((n + *nt - 1) / *nt);
    if (*nb > 65535) *nb = 65535;
}

/* ==================== fused forward kernel ====================== */

static __device__ __forceinline__ float2 ldg_c(const float2 *p) {
    return __ldg(p);
}

__global__ void k_fused_mp(
    const float2 *__restrict__ F,
    const float2 *__restrict__ c,
    const float  *__restrict__ B,
    const float  *__restrict__ w,
    float2       *__restrict__ G,
    float        *__restrict__ P_out,
    float        *__restrict__ nll_out,
    float        *__restrict__ scorr_out,
    int64_t N, int JP, int KC,
    float N_s, float N_b, float pur)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= N) return;

    /* ---- A[i,j] = sum_k F[k,i,j] * c[k]  (FP32) ---- */
    float A_re[4] = {0}, A_im[4] = {0};
    for (int j = 0; j < JP; j++) {
        float re = 0.0f, im = 0.0f;
        for (int k = 0; k < KC; k++) {
            int64_t fidx = k * N * JP + i * JP + j;
            float2 f  = ldg_c(&F[fidx]);
            float2 ck = ldg_c(&c[k]);
            re += f.x * ck.x - f.y * ck.y;
            im += f.x * ck.y + f.y * ck.x;
        }
        A_re[j] = re;
        A_im[j] = im;
    }

    /* ---- S[i] ---- */
    float S = 0.0f;
    for (int j = 0; j < JP; j++)
        S += A_re[j] * A_re[j] + A_im[j] * A_im[j];

    /* ---- P[i] ---- */
    float p_val = S / N_s * pur + B[i] / N_b * (1.0f - pur);
    if (p_val < 1e-30f) p_val = 1e-30f;

    /* ---- G[i,j] ---- */
    float ratio = w[i] / p_val;
    for (int j = 0; j < JP; j++)
        G[i * JP + j] = cset(A_re[j] * ratio, A_im[j] * ratio);

    /* ---- NLL ---- */
    float nll_i = -w[i] * logf(p_val);

    /* ---- S_corr ---- */
    float scorr_i = w[i] * S / p_val;

    /* ---- Optional P output ---- */
    if (P_out) P_out[i] = p_val;

    /* ---- Warp reduction ---- */
    for (int offset = 16; offset > 0; offset >>= 1)
        nll_i += __shfl_down_sync(0xffffffff, nll_i, offset);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(nll_out, nll_i);

    for (int offset = 16; offset > 0; offset >>= 1)
        scorr_i += __shfl_down_sync(0xffffffff, scorr_i, offset);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(scorr_out, scorr_i);
}

/* ==================== gradient helper =========================== */

__global__ void k_conjvec_mp(const float2 *__restrict__ in, float2 *__restrict__ out, int64_t N) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= N) return;
    float2 v = ldg_c(&in[i]);
    out[i] = cset(v.x, -v.y);
}

/* ==================== Host helpers (FP32) ======================= */

static float compute_Ns_host_mp(const float2 *M, const float2 *c, int KC) {
    float re = 0.0f;
    for (int k1 = 0; k1 < KC; k1++) {
        for (int k2 = 0; k2 < KC; k2++) {
            float2 m  = M[k1 * KC + k2];
            float2 c1 = c[k1];
            float2 c2 = c[k2];
            float t_re = m.x * c2.x - m.y * c2.y;
            float t_im = m.x * c2.y + m.y * c2.x;
            re += c1.x * t_re + c1.y * t_im;
        }
    }
    return re;
}

static void compute_dNs_host_mp(const float2 *M, const float2 *c,
                                 float2 *dNs, int KC) {
    for (int k1 = 0; k1 < KC; k1++) {
        float re = 0.0f, im = 0.0f;
        for (int k2 = 0; k2 < KC; k2++) {
            float2 m  = M[k1 * KC + k2];
            float2 ck = c[k2];
            re += m.x * ck.x - m.y * ck.y;
            im += m.x * ck.y + m.y * ck.x;
        }
        dNs[k1] = cset(re, im);
    }
}

/* ==================== Fitter state ============================== */

struct FpwFitterMP {
    int64_t nd;
    int     jp, kc;
    float   pur, Nb, Ns;

    /* GPU-resident data (FP32) — F stored as (KC, N, JP) */
    float2 *dF;       /* (kc, nd, jp)  */
    float  *dw;       /* (nd,)          */
    float  *dB;       /* (nd,)          */
    float2 *hM;       /* (kc, kc) host  */

    /* GPU workspace (FP32) */
    float2 *dG;       /* (nd, jp)    */
    float  *dP;       /* (nd,)      */
    float  *dnll;     /* (1,)       */
    float  *dscorr;   /* (1,)       */
    float2 *dg;       /* (kc,)      */
    float2 *dc;       /* (kc,)      */
    float2 *dGconj;   /* (nd*jp,)   */
    float2 *dz;       /* (kc,)      */

    cublasHandle_t hdl;
    cudaStream_t   strm;
};

/* ==================== public API ================================== */

const char *fpw_strerror(int e) {
    switch (e) {
    case FPW_SUCCESS:    return "Success";
    case FPW_ERR_ALLOC:  return "Memory allocation failed";
    case FPW_ERR_CUDA:   return "CUDA runtime error";
    case FPW_ERR_CUBLAS: return "cuBLAS error";
    default:             return "Unknown error";
    }
}
int    fpw_mp_get_n_comp(const FpwFitterMP *f) { return f->kc; }
float  fpw_mp_get_N_s    (const FpwFitterMP *f) { return f->Ns; }
float  fpw_mp_get_N_b    (const FpwFitterMP *f) { return f->Nb; }

/* ---------------------------------------------------------------- */

int fpw_mp_create(int64_t nd, int jp, int kc,
                  const float *Fd, const float *wd, const float *Bd,
                  const float *M,
                  float Nb, float pur, FpwFitterMP **out)
{
    if (nd <= 0 || jp <= 0 || kc <= 0) return FPW_ERR_ALLOC;

    FpwFitterMP *f = (FpwFitterMP *)calloc(1, sizeof(*f));
    if (!f) return FPW_ERR_ALLOC;

    f->nd = nd; f->jp = jp; f->kc = kc;
    f->pur = pur; f->Nb = Nb; f->Ns = 0.0f;

    f->hM = (float2 *)malloc((int64_t)kc * kc * sizeof(float2));
    if (!f->hM) { free(f); return FPW_ERR_ALLOC; }
    memcpy(f->hM, M, (int64_t)kc * kc * sizeof(float2));

    cudaError_t    ce;
    cublasStatus_t cb;
    int64_t        szF, sz1, szG, szg;

    ce = cudaStreamCreate(&f->strm);
    if (ce != cudaSuccess) goto fail;
    cb = cublasCreate(&f->hdl);
    if (cb != CUBLAS_STATUS_SUCCESS) goto fail;
    cublasSetStream(f->hdl, f->strm);

    szF  = (int64_t)kc * nd * jp * sizeof(float2);
    sz1  = (int64_t)nd * sizeof(float);
    szG  = (int64_t)nd * jp * sizeof(float2);
    szg  = (int64_t)kc  * sizeof(float2);

    #define DA(p, sz) do { ce = cudaMalloc((void **)&(f->p), sz); \
                           if (ce != cudaSuccess) goto fail; } while (0)

    DA(dF,     szF);
    DA(dw,     sz1);
    DA(dB,     sz1);
    DA(dG,     szG);
    DA(dP,     sz1);
    DA(dnll,   sizeof(float));
    DA(dscorr, sizeof(float));
    DA(dg,     szg);
    DA(dc,     szg);
    DA(dGconj, szG);
    DA(dz,     szg);
    #undef DA

    /* ---- Transpose and upload F: (N, JP, KC) → (KC, N, JP) ---- */
    {
        float2 *h_F_trans = (float2 *)malloc((int64_t)kc * nd * jp * sizeof(float2));
        if (!h_F_trans) { ce = cudaErrorMemoryAllocation; goto fail; }
        const float2 *F_in = (const float2 *)Fd;
        for (int64_t k = 0; k < kc; k++) {
            for (int64_t i = 0; i < nd; i++) {
                for (int j = 0; j < jp; j++) {
                    h_F_trans[k * nd * jp + i * jp + j] =
                        F_in[i * jp * kc + j * kc + k];
                }
            }
        }
        ce = cudaMemcpyAsync(f->dF, h_F_trans,
                             (int64_t)kc * nd * jp * sizeof(float2),
                             cudaMemcpyHostToDevice, f->strm);
        free(h_F_trans);
        if (ce != cudaSuccess) goto fail;
    }

    ce = cudaMemcpyAsync(f->dw, wd, sz1, cudaMemcpyHostToDevice, f->strm);
    if (ce != cudaSuccess) goto fail;
    ce = cudaMemcpyAsync(f->dB, Bd, sz1, cudaMemcpyHostToDevice, f->strm);
    if (ce != cudaSuccess) goto fail;

    cudaStreamSynchronize(f->strm);
    *out = f;
    return FPW_SUCCESS;

fail:
    fpw_mp_destroy(f);
    return FPW_ERR_ALLOC;
}

/* ---------------------------------------------------------------- */

void fpw_mp_destroy(FpwFitterMP *f) {
    if (!f) return;
    cudaFree(f->dF);
    cudaFree(f->dw);
    cudaFree(f->dB);
    cudaFree(f->dG);
    cudaFree(f->dP);
    cudaFree(f->dnll);
    cudaFree(f->dscorr);
    cudaFree(f->dg);
    cudaFree(f->dc);
    cudaFree(f->dGconj);
    cudaFree(f->dz);
    free(f->hM);
    if (f->hdl)  cublasDestroy(f->hdl);
    if (f->strm) cudaStreamDestroy(f->strm);
    free(f);
}

/* ---------------------------------------------------------------- */

int fpw_mp_evaluate(FpwFitterMP *f,
                    const double *cr, const double *ci,
                    double *nll,
                    double *gr, double *gi,
                    float *P_out)
{
    cudaError_t    ce;
    int    kc = f->kc, jp = f->jp;
    int64_t nd = f->nd;

    /* ---- Convert c from FP64 → FP32, copy to GPU ---- */
    float2 *h_c = (float2 *)malloc(kc * sizeof(float2));
    if (!h_c) return FPW_ERR_ALLOC;
    for (int k = 0; k < kc; k++)
        h_c[k] = cset((float)cr[k], (float)ci[k]);
    cudaMemcpy(f->dc, h_c, kc * sizeof(float2), cudaMemcpyHostToDevice);

    /* ---- N_s, dN_s (host, FP32) ---- */
    float2 *c_f32 = (float2 *)malloc(kc * sizeof(float2));
    for (int k = 0; k < kc; k++)
        c_f32[k] = cset((float)cr[k], (float)ci[k]);
    f->Ns = compute_Ns_host_mp(f->hM, c_f32, kc);
    if (f->Ns < 1e-30f) f->Ns = 1e-30f;
    if (f->Nb < 1e-30f) f->Nb = 1e-30f;

    float2 *dNs_h = (float2 *)malloc(kc * sizeof(float2));
    compute_dNs_host_mp(f->hM, c_f32, dNs_h, kc);

    /* ---- Zero GPU accumulators ---- */
    cudaMemsetAsync(f->dnll,   0, sizeof(float), f->strm);
    cudaMemsetAsync(f->dscorr, 0, sizeof(float), f->strm);
    cudaMemsetAsync(f->dg,     0, kc * sizeof(float2), f->strm);

    /* ---- FUSED kernel (FP32) ---- */
    { int nb, nt; lcfg(nd, &nb, &nt);
      k_fused_mp<<<nb, nt, 0, f->strm>>>(
          f->dF, f->dc, f->dB, f->dw,
          f->dG, f->dP, f->dnll, f->dscorr,
          nd, jp, kc, f->Ns, f->Nb, f->pur); }

    /* ---- Gradient via cuBLAS SGEMV (FP32) ---- */
    {
        int64_t nj = nd * jp;
        { int nb, nt; lcfg(nj, &nb, &nt);
          k_conjvec_mp<<<nb, nt, 0, f->strm>>>(f->dG, f->dGconj, nj); }

        cuComplex alpha = make_cuComplex(1.0f, 0.0f);
        cuComplex beta  = make_cuComplex(0.0f, 0.0f);
        cublasCgemv(f->hdl, CUBLAS_OP_T,
                    (int)nj, kc,
                    &alpha,
                    (const cuComplex *)f->dF, (int)nj,
                    (const cuComplex *)f->dGconj, 1,
                    &beta,
                    (cuComplex *)f->dz, 1);

        { int nb, nt; lcfg(kc, &nb, &nt);
          k_conjvec_mp<<<nb, nt, 0, f->strm>>>(f->dz, f->dg, kc); }
    }

    /* ---- Copy results back ---- */
    float h_nll = 0.0f, h_scorr = 0.0f;
    float2 *g_h = (float2 *)malloc(kc * sizeof(float2));
    float p, Ns, Ns2, f1, f2;

    ce = cudaMemcpyAsync(&h_nll,   f->dnll,   sizeof(float),
                         cudaMemcpyDeviceToHost, f->strm);
    if (ce != cudaSuccess) goto copy_fail;
    ce = cudaMemcpyAsync(&h_scorr, f->dscorr, sizeof(float),
                         cudaMemcpyDeviceToHost, f->strm);
    if (ce != cudaSuccess) goto copy_fail;
    ce = cudaMemcpyAsync(g_h, f->dg, kc * sizeof(float2),
                         cudaMemcpyDeviceToHost, f->strm);
    if (ce != cudaSuccess) goto copy_fail;
    if (P_out) {
        ce = cudaMemcpyAsync(P_out, f->dP, nd * sizeof(float),
                             cudaMemcpyDeviceToHost, f->strm);
        if (ce != cudaSuccess) goto copy_fail;
    }

    cudaStreamSynchronize(f->strm);

    p   = f->pur;
    Ns  = f->Ns;
    Ns2 = Ns * Ns;
    f1  = -p / Ns;
    f2  = p / Ns2 * h_scorr;

    /* Convert FP32 → FP64 for output */
    for (int k = 0; k < kc; k++) {
        gr[k] = (double)f1 * (double)g_h[k].x + (double)f2 * (double)dNs_h[k].x;
        gi[k] = (double)f1 * (double)g_h[k].y + (double)f2 * (double)dNs_h[k].y;
    }
    *nll = (double)h_nll;

    free(h_c); free(c_f32); free(dNs_h); free(g_h);
    return FPW_SUCCESS;

copy_fail:
    cudaStreamSynchronize(f->strm);
    free(h_c); free(c_f32); free(dNs_h); free(g_h);
    return FPW_ERR_CUDA;
}
