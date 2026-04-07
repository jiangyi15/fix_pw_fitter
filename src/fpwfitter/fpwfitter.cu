/* ================================================================
   fpwfitter.cu  –  Fixed Partial Waves Fitter  (CUDA, all-GPU data)
   ================================================================
   F LAYOUT: (KC, N, JP) for coalesced memory access.
     F[k, i, j] at offset k*N*JP + i*JP + j
   Forward: single fused kernel
   Gradient: cuBLAS ZGEMV
   ================================================================ */

#include "fpwfitter.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ==================== complex helpers =========================== */
static __host__ __device__ __forceinline__ double2 cset(double r, double i) {
    return make_double2(r, i);
}
static __host__ __device__ __forceinline__ double cabssq(double2 a) {
    return a.x * a.x + a.y * a.y;
}

/* ==================== launch config ============================= */
static void lcfg(int64_t n, int *nb, int *nt) {
    *nt = 256;
    *nb = (int)((n + *nt - 1) / *nt);
    if (*nb > 65535) *nb = 65535;
}

/* ==================== fused forward kernel ====================== */

/* __ldg() reads through texture/L1 cache — better for read-only data on Ampere */
static __device__ __forceinline__ double2 ldg_c(const double2 *p) {
    return __ldg(p);
}

__global__ void k_fused(
    const double2 *__restrict__ F,
    const double2 *__restrict__ c,
    const double  *__restrict__ B,
    const double  *__restrict__ w,
    double2       *__restrict__ G,
    double        *__restrict__ P_out,
    double        *__restrict__ nll_out,
    double        *__restrict__ scorr_out,
    int64_t N, int JP, int KC,
    double N_s, double N_b, double pur,
    int64_t F_total_elements)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= N) return;

    /* ---- A[i,j] = sum_k F[k,i,j] * c[k]  ---- */
    double A_re[4] = {0}, A_im[4] = {0};
    for (int j = 0; j < JP; j++) {
        double re = 0.0, im = 0.0;
        for (int k = 0; k < KC; k++) {
            int64_t fidx = k * N * JP + i * JP + j;
            double2 f  = ldg_c(&F[fidx]);
            double2 ck = ldg_c(&c[k]);
            re += f.x * ck.x - f.y * ck.y;
            im += f.x * ck.y + f.y * ck.x;
        }
        A_re[j] = re;
        A_im[j] = im;
    }

    /* ---- S[i] ---- */
    double S = 0.0;
    for (int j = 0; j < JP; j++)
        S += A_re[j] * A_re[j] + A_im[j] * A_im[j];

    /* ---- P[i] ---- */
    double p_val = S / N_s * pur + B[i] / N_b * (1.0 - pur);
    if (p_val < 1e-300) p_val = 1e-300;

    /* ---- G[i,j] ---- */
    double ratio = w[i] / p_val;
    for (int j = 0; j < JP; j++)
        G[i * JP + j] = cset(A_re[j] * ratio, A_im[j] * ratio);

    /* ---- NLL ---- */
    double nll_i = -w[i] * log(p_val);

    /* ---- S_corr ---- */
    double scorr_i = w[i] * S / p_val;

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

/* Conjugate complex vector — use __ldg for read-only input */
__global__ void k_conjvec(const double2 *__restrict__ in, double2 *__restrict__ out, int64_t N) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= N) return;
    double2 v = ldg_c(&in[i]);
    out[i] = cset(v.x, -v.y);
}

/* ==================== Host helpers ============================== */

static double compute_Ns_host(const double2 *M, const double2 *c, int KC) {
    double re = 0.0;
    for (int k1 = 0; k1 < KC; k1++) {
        for (int k2 = 0; k2 < KC; k2++) {
            double2 m  = M[k1 * KC + k2];
            double2 c1 = c[k1];
            double2 c2 = c[k2];
            double t_re = m.x * c2.x - m.y * c2.y;
            double t_im = m.x * c2.y + m.y * c2.x;
            re += c1.x * t_re + c1.y * t_im;
        }
    }
    return re;
}

static void compute_dNs_host(const double2 *M, const double2 *c,
                              double2 *dNs, int KC) {
    for (int k1 = 0; k1 < KC; k1++) {
        double re = 0.0, im = 0.0;
        for (int k2 = 0; k2 < KC; k2++) {
            double2 m  = M[k1 * KC + k2];
            double2 ck = c[k2];
            re += m.x * ck.x - m.y * ck.y;
            im += m.x * ck.y + m.y * ck.x;
        }
        dNs[k1] = cset(re, im);
    }
}

/* ==================== Fitter state ============================== */

struct FpwFitter {
    int64_t nd;
    int     jp, kc;
    double  pur, Nb, Ns;

    /* GPU-resident data — F stored as (KC, N, JP) */
    double2 *dF;       /* (kc, nd, jp)  complex */
    double  *dw;       /* (nd,)                  */
    double  *dB;       /* (nd,)                  */
    double2 *hM;       /* (kc, kc) host copy     */

    /* GPU workspace */
    double2 *dG;       /* (nd, jp)    complex    */
    double  *dP;       /* (nd,)                  */
    double  *dnll;     /* (1,)                   */
    double  *dscorr;   /* (1,)                   */
    double2 *dg;       /* (kc,)     gradient     */
    double2 *dc;       /* (kc,)     coupling     */
    double2 *dGconj;   /* (nd*jp,)  conj(G)      */
    double2 *dz;       /* (kc,)     ZGEMV out    */

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
int    fpw_get_n_comp(const FpwFitter *f) { return f->kc; }
double fpw_get_N_s    (const FpwFitter *f) { return f->Ns; }
double fpw_get_N_b    (const FpwFitter *f) { return f->Nb; }

/* ---------------------------------------------------------------- */

int fpw_create(int64_t nd, int jp, int kc,
               const double *Fd, const double *wd, const double *Bd,
               const double *M,
               double Nb, double pur, FpwFitter **out)
{
    if (nd <= 0 || jp <= 0 || kc <= 0) return FPW_ERR_ALLOC;

    FpwFitter *f = (FpwFitter *)calloc(1, sizeof(*f));
    if (!f) return FPW_ERR_ALLOC;

    f->nd = nd; f->jp = jp; f->kc = kc;
    f->pur = pur; f->Nb = Nb; f->Ns = 0.0;

    f->hM = (double2 *)malloc((int64_t)kc * kc * sizeof(double2));
    if (!f->hM) { free(f); return FPW_ERR_ALLOC; }
    memcpy(f->hM, M, (int64_t)kc * kc * sizeof(double2));

    cudaError_t    ce;
    cublasStatus_t cb;
    int64_t        szF, sz1, szG, szg;

    ce = cudaStreamCreate(&f->strm);
    if (ce != cudaSuccess) goto fail;
    cb = cublasCreate(&f->hdl);
    if (cb != CUBLAS_STATUS_SUCCESS) goto fail;
    cublasSetStream(f->hdl, f->strm);

    szF  = (int64_t)kc * nd * jp * sizeof(double2);  /* (KC, N, JP) */
    sz1  = (int64_t)nd * sizeof(double);
    szG  = (int64_t)nd * jp * sizeof(double2);
    szg  = (int64_t)kc  * sizeof(double2);

    #define DA(p, sz) do { ce = cudaMalloc((void **)&(f->p), sz); \
                           if (ce != cudaSuccess) goto fail; } while (0)

    DA(dF,     szF);
    DA(dw,     sz1);
    DA(dB,     sz1);
    DA(dG,     szG);
    DA(dP,     sz1);
    DA(dnll,   sizeof(double));
    DA(dscorr, sizeof(double));
    DA(dg,     szg);
    DA(dc,     szg);
    DA(dGconj, szG);
    DA(dz,     szg);
    #undef DA

    /* ---- Transpose and upload F: (N, JP, KC) → (KC, N, JP) ---- */
    {
        double2 *h_F_trans = (double2 *)malloc((int64_t)kc * nd * jp * sizeof(double2));
        if (!h_F_trans) { ce = cudaErrorMemoryAllocation; goto fail; }
        const double2 *F_in = (const double2 *)Fd;
        for (int64_t k = 0; k < kc; k++) {
            for (int64_t i = 0; i < nd; i++) {
                for (int j = 0; j < jp; j++) {
                    h_F_trans[k * nd * jp + i * jp + j] =
                        F_in[i * jp * kc + j * kc + k];
                }
            }
        }
        ce = cudaMemcpyAsync(f->dF, h_F_trans,
                             (int64_t)kc * nd * jp * sizeof(double2),
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
    fpw_destroy(f);
    return FPW_ERR_ALLOC;
}

/* ---------------------------------------------------------------- */

void fpw_destroy(FpwFitter *f) {
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

/* ---- Helper: launch all GPU kernels ---- */
static void launch_all_kernels(FpwFitter *f, int64_t nd, int jp, int kc) {
    int64_t nj = nd * jp;

    /* ---- Zero GPU accumulators ---- */
    cudaMemsetAsync(f->dnll,   0, sizeof(double), f->strm);
    cudaMemsetAsync(f->dscorr, 0, sizeof(double), f->strm);
    cudaMemsetAsync(f->dg,     0, kc * sizeof(double2), f->strm);

    /* ---- FUSED kernel ---- */
    { int nb, nt; lcfg(nd, &nb, &nt);
      k_fused<<<nb, nt, 0, f->strm>>>(
          f->dF, f->dc, f->dB, f->dw,
          f->dG, f->dP, f->dnll, f->dscorr,
          nd, jp, kc, f->Ns, f->Nb, f->pur,
          (int64_t)kc * nd * jp); }

    /* ---- Gradient via cuBLAS ZGEMV ---- */
    {
        { int nb, nt; lcfg(nj, &nb, &nt);
          k_conjvec<<<nb, nt, 0, f->strm>>>(f->dG, f->dGconj, nj); }

        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
        cublasZgemv(f->hdl, CUBLAS_OP_T,
                    (int)nj, kc,
                    &alpha,
                    (const cuDoubleComplex *)f->dF, (int)nj,
                    (const cuDoubleComplex *)f->dGconj, 1,
                    &beta,
                    (cuDoubleComplex *)f->dz, 1);

        { int nb, nt; lcfg(kc, &nb, &nt);
          k_conjvec<<<nb, nt, 0, f->strm>>>(f->dz, f->dg, kc); }
    }
}

/* ---------------------------------------------------------------- */

int fpw_evaluate(FpwFitter *f,
                 const double *cr, const double *ci,
                 double *nll,
                 double *gr, double *gi,
                 double *P_out)
{
    cudaError_t    ce;
    int    kc = f->kc, jp = f->jp;
    int64_t nd = f->nd;

    /* ---- Copy c to GPU ---- */
    double2 *h_c = (double2 *)malloc(kc * sizeof(double2));
    if (!h_c) return FPW_ERR_ALLOC;
    for (int k = 0; k < kc; k++)
        h_c[k] = cset(cr[k], ci[k]);
    cudaMemcpy(f->dc, h_c, kc * sizeof(double2), cudaMemcpyHostToDevice);

    /* ---- N_s, dN_s (host) ---- */
    f->Ns = compute_Ns_host(f->hM, h_c, kc);
    if (f->Ns < 1e-300) f->Ns = 1e-300;
    if (f->Nb < 1e-300) f->Nb = 1e-300;

    double2 *dNs_h = (double2 *)malloc(kc * sizeof(double2));
    compute_dNs_host(f->hM, h_c, dNs_h, kc);

    /* ---- Launch all kernels ---- */
    launch_all_kernels(f, nd, jp, kc);

    /* ---- Copy results back ---- */
    double h_nll = 0.0, h_scorr = 0.0;
    double2 *g_h = (double2 *)malloc(kc * sizeof(double2));
    double p, Ns, Ns2, f1, f2;

    ce = cudaMemcpyAsync(&h_nll,   f->dnll,   sizeof(double),
                         cudaMemcpyDeviceToHost, f->strm);
    if (ce != cudaSuccess) goto copy_fail;
    ce = cudaMemcpyAsync(&h_scorr, f->dscorr, sizeof(double),
                         cudaMemcpyDeviceToHost, f->strm);
    if (ce != cudaSuccess) goto copy_fail;
    ce = cudaMemcpyAsync(g_h, f->dg, kc * sizeof(double2),
                         cudaMemcpyDeviceToHost, f->strm);
    if (ce != cudaSuccess) goto copy_fail;
    if (P_out) {
        ce = cudaMemcpyAsync(P_out, f->dP, nd * sizeof(double),
                             cudaMemcpyDeviceToHost, f->strm);
        if (ce != cudaSuccess) goto copy_fail;
    }

    cudaStreamSynchronize(f->strm);

    p   = f->pur;
    Ns  = f->Ns;
    Ns2 = Ns * Ns;
    f1  = -p / Ns;
    f2  = p / Ns2 * h_scorr;

    for (int k = 0; k < kc; k++) {
        gr[k] = f1 * g_h[k].x + f2 * dNs_h[k].x;
        gi[k] = f1 * g_h[k].y + f2 * dNs_h[k].y;
    }
    *nll = h_nll;

    free(h_c); free(dNs_h); free(g_h);
    return FPW_SUCCESS;

copy_fail:
    cudaStreamSynchronize(f->strm);
    free(h_c); free(dNs_h); free(g_h);
    return FPW_ERR_CUDA;
}
