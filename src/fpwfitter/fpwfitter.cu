/* ================================================================
   fpwfitter.cu  –  Fixed Partial Waves Fitter  (CUDA, all-GPU data)
   ================================================================
   ALL data uploaded to GPU at create time.  Evaluate runs entirely
   on GPU — only NLL + gradient copied back.

   Optimised:
     * Single fused kernel for A→S→P→NLL→G→S_corr
     * Warp-level reduction for NLL/S_corr (no atomicAdd)
     * cuBLAS ZGEMV for gradient
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

/* ==================== fused forward kernel ======================
 *
 * Per-thread computation for one event:
 *   A[i,:] → S[i] → P[i] → NLL[i] → G[i,:] → S_corr[i]
 *
 * Only G (needed for gradient) and optional P are written to
 * global memory.  All intermediates stay in registers.
 * ================================================================ */

__global__ void k_fused(
    const double2 *F,       /* (N, JP, KC)  complex */
    const double2 *c,       /* (KC,)        complex */
    const double  *B,       /* (N,)         */
    const double  *w,       /* (N,)         */
    double2       *G,       /* (N, JP)      complex — output */
    double        *P_out,   /* (N,)         optional — output (NULL ok) */
    double        *nll_out, /* (1,)         output */
    double        *scorr_out,/* (1,)        output */
    int64_t N, int JP, int KC,
    double N_s, double N_b, double pur)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= N) return;

    /* ---- A[i,j] = sum_k F[i,j,k] * c[k]  (in registers) ---- */
    double A_re[4] = {0}, A_im[4] = {0};  /* JP ≤ 4 typical */
    for (int j = 0; j < JP; j++) {
        const double2 *Fbase = &F[i * (int64_t)JP * KC + j * KC];
        double re = 0.0, im = 0.0;
        for (int k = 0; k < KC; k++) {
            double2 f  = Fbase[k];
            double2 ck = c[k];
            re += f.x * ck.x - f.y * ck.y;
            im += f.x * ck.y + f.y * ck.x;
        }
        A_re[j] = re;
        A_im[j] = im;
    }

    /* ---- S[i] = sum_j |A[i,j]|^2 ---- */
    double S = 0.0;
    for (int j = 0; j < JP; j++)
        S += A_re[j] * A_re[j] + A_im[j] * A_im[j];

    /* ---- P[i] ---- */
    double p_val = S / N_s * pur + B[i] / N_b * (1.0 - pur);
    if (p_val < 1e-300) p_val = 1e-300;

    /* ---- NLL contribution ---- */
    double nll_i = -w[i] * log(p_val);

    /* ---- G[i,j] = (w[i]/P[i]) * A[i,j]  (write to global) ---- */
    double ratio = w[i] / p_val;
    for (int j = 0; j < JP; j++)
        G[i * JP + j] = cset(A_re[j] * ratio, A_im[j] * ratio);

    /* ---- S_corr contribution ---- */
    double scorr_i = w[i] * S / p_val;

    /* ---- Optional P output ---- */
    if (P_out) P_out[i] = p_val;

    /* ---- Warp-level reduction for NLL ---- */
    /* Sum nll_i within warp, then first thread does atomicAdd */
    for (int offset = 16; offset > 0; offset >>= 1)
        nll_i += __shfl_down_sync(0xffffffff, nll_i, offset);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(nll_out, nll_i);

    /* ---- Warp-level reduction for S_corr ---- */
    for (int offset = 16; offset > 0; offset >>= 1)
        scorr_i += __shfl_down_sync(0xffffffff, scorr_i, offset);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(scorr_out, scorr_i);
}

/* ==================== gradient kernel =========================== */

/* Conjugate complex vector (for ZGEMV trick) */
__global__ void k_conjvec(const double2 *in, double2 *out, int64_t N) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = cset(in[i].x, -in[i].y);
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

    /* GPU-resident data */
    double2 *dF;       /* (nd, jp, kc)  complex */
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
    double2 *dGconj;   /* (nd*jp,)  conj(G) for ZGEMV */
    double2 *dz;       /* (kc,)     ZGEMV output  */

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

    szF  = (int64_t)nd * jp * kc * sizeof(double2);
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
    DA(dGconj, szG);   /* (nd*jp,) */
    DA(dz,     szg);   /* (kc,)    */
    #undef DA

    ce = cudaMemcpyAsync(f->dF, Fd, szF, cudaMemcpyHostToDevice, f->strm);
    if (ce != cudaSuccess) goto fail;
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

/* ---------------------------------------------------------------- */

int fpw_evaluate(FpwFitter *f,
                 const double *cr, const double *ci,
                 double *nll,
                 double *gr, double *gi,
                 double *P_out)
{
    cublasStatus_t cb;
    cudaError_t    ce;
    int    kc = f->kc, jp = f->jp;
    int64_t nd = f->nd;
    int64_t nj = nd * jp;

    /* ---- Copy c to GPU ---- */
    double2 *h_c = (double2 *)malloc(kc * sizeof(double2));
    if (!h_c) return FPW_ERR_ALLOC;
    for (int k = 0; k < kc; k++)
        h_c[k] = cset(cr[k], ci[k]);
    cudaMemcpyAsync(f->dc, h_c, kc * sizeof(double2),
                    cudaMemcpyHostToDevice, f->strm);

    /* ---- N_s, dN_s (host) ---- */
    f->Ns = compute_Ns_host(f->hM, h_c, kc);
    if (f->Ns < 1e-300) f->Ns = 1e-300;
    if (f->Nb < 1e-300) f->Nb = 1e-300;

    double2 *dNs_h = (double2 *)malloc(kc * sizeof(double2));
    compute_dNs_host(f->hM, h_c, dNs_h, kc);

    /* ---- Zero GPU accumulators ---- */
    cudaMemsetAsync(f->dnll,   0, sizeof(double), f->strm);
    cudaMemsetAsync(f->dscorr, 0, sizeof(double), f->strm);

    /* ---- FUSED kernel: A → S → P → NLL → G → S_corr ---- */
    /* Note: always write P to device buffer dP; copy to host P_out later */
    { int nb, nt; lcfg(nd, &nb, &nt);
      k_fused<<<nb, nt, 0, f->strm>>>(
          f->dF, f->dc,
          f->dB, f->dw,
          f->dG, f->dP,
          f->dnll, f->dscorr,
          nd, jp, kc,
          f->Ns, f->Nb, f->pur);
    }

    /* ---- Gradient via cuBLAS ZGEMV ----
     * g[k] = sum_{i,j} conj(F[i,j,k]) * G[i,j]
     *
     * F row-maj (nd, jp, kc) → cuBLAS sees col-maj (kc, nj).
     * OP_N: z(kc) = A(kc,nj) @ conj(G)(nj)
     *   z[k] = sum_idx F[idx,k] * conj(G[idx])
     *   conj(z[k]) = sum_idx conj(F[idx,k]) * G[idx]  ✓
     */
    {
        /* conj(G) */
        { int nb, nt; lcfg(nj, &nb, &nt);
          k_conjvec<<<nb, nt, 0, f->strm>>>(f->dG, f->dGconj, nj); }

        /* ZGEMV: z = F @ conj(G) */
        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
        cb = cublasZgemv(f->hdl, CUBLAS_OP_N,
                         kc, (int)nj,
                         &alpha,
                         (const cuDoubleComplex *)f->dF, kc,
                         (const cuDoubleComplex *)f->dGconj, 1,
                         &beta,
                         (cuDoubleComplex *)f->dz, 1);
        if (cb != CUBLAS_STATUS_SUCCESS) {
            free(h_c); free(dNs_h);
            return FPW_ERR_CUBLAS;
        }

        /* g = conj(z) */
        { int nb, nt; lcfg(kc, &nb, &nt);
          k_conjvec<<<nb, nt, 0, f->strm>>>(f->dz, f->dg, kc); }
    }

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

    /* ---- Assemble gradient ---- */
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
