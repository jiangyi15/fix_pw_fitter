/* ================================================================
   fpwfitter.cu  –  Fixed Partial Waves Fitter  (CUDA, all-GPU data)
   ================================================================

   ALL data (F_data, w_data, B_data, M) is uploaded to GPU at
   fpw_create() and kept there permanently.  fpw_evaluate() runs
   entirely on GPU — no H2D/D2H transfers except the final results.

   DATA LAYOUT (C-order):
     F_data : (n_data, n_proj, n_comp)   complex128
     M      : (n_comp, n_comp)            complex128

   ALGORITHM (optimised order):

     A[i,j]   = sum_k F[i,j,k] c[k]
     S[i]     = sum_j |A[i,j]|^2
     N_s      = c^H M c                              host (k is small)
     P[i]     = S[i]/N_s*p + B[i]/N_b*(1-p)
     NLL      = -sum_i w[i] log(P[i])
     G[i,j]   = (w[i]/P[i]) * A[i,j]
     g[k]    += sum_{i,j} conj(F[i,j,k]) * G[i,j]
     dN_s[k]  = (M @ c)[k]                          host
     S_corr   = sum_i w[i]*S[i]/P[i]
     grad[k]  = -p/N_s*g[k] + p/N_s^2*dN_s[k]*S_corr
   ================================================================ */

#include "fpwfitter.h"
#include <cuda_runtime.h>
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

/* ==================== CUDA kernels ============================== */

/*
 * A[i,j] = sum_k F[i,j,k] * c[k]
 * F:  (N, JP, KC)  complex  — resident on GPU
 * c:  (KC,)        complex
 * A:  (N, JP)      complex  — output
 */
__global__ void k_A(const double2 *F, const double2 *c,
                    double2 *A, int64_t N, int JP, int KC) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= N * JP) return;
    int64_t i  = idx / JP;
    int     j  = idx % JP;
    double re = 0.0, im = 0.0;
    for (int k = 0; k < KC; k++) {
        double2 f = F[i * (int64_t)JP * KC + j * KC + k];
        double2 ck = c[k];
        re += f.x * ck.x - f.y * ck.y;
        im += f.x * ck.y + f.y * ck.x;
    }
    A[idx] = cset(re, im);
}

/*
 * S[i] = sum_j |A[i,j]|^2
 */
__global__ void k_S(const double2 *A, double *S, int64_t N, int JP) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= N) return;
    double s = 0.0;
    for (int j = 0; j < JP; j++)
        s += cabssq(A[i * JP + j]);
    S[i] = s;
}

/*
 * P[i] and NLL  (single kernel, fused)
 */
__global__ void k_PNLL(const double *S, const double *B, const double *w,
                        double *P, double *nll_out,
                        int64_t N, double N_s, double N_b, double pur) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= N) return;
    double p = S[i] / N_s * pur + B[i] / N_b * (1.0 - pur);
    if (p < 1e-300) p = 1e-300;
    P[i] = p;
    atomicAdd(nll_out, -w[i] * log(p));
}

/*
 * G[i,j] = (w[i]/P[i]) * A[i,j]
 */
__global__ void k_G(const double *w, const double *P, const double2 *A,
                    double2 *G, int64_t N, int JP) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= N) return;
    double r = w[i] / P[i];
    for (int j = 0; j < JP; j++)
        G[i * JP + j] = cset(A[i * JP + j].x * r, A[i * JP + j].y * r);
}

/*
 * S_corr = sum_i w[i]*S[i]/P[i]
 */
__global__ void k_Scorr(const double *w, const double *S, const double *P,
                        double *sc, int64_t N) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= N) return;
    atomicAdd(sc, w[i] * S[i] / P[i]);
}

/*
 * Gradient accumulation:
 *   g[k] += sum_{i,j} conj(F[i,j,k]) * G[i,j]
 * Each thread handles a subset of (i,j) for a given k.
 */
__global__ void k_grad_accum(const double2 *F, const double2 *G,
                              double2 *g,
                              int64_t N, int JP, int KC, int k_idx) {
    int64_t total = N * JP;
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    double re = 0.0, im = 0.0;
    for (int64_t idx = tid; idx < total; idx += stride) {
        int64_t i = idx / JP;
        int     j = idx % JP;
        double2 f = F[i * (int64_t)JP * KC + j * KC + k_idx];
        double2 gv = G[idx];
        /* conj(F) * G */
        re += f.x * gv.x + f.y * gv.y;
        im += f.x * gv.y - f.y * gv.x;
    }
    atomicAdd(&g[k_idx].x, re);
    atomicAdd(&g[k_idx].y, im);
}

/*
 * Compute N_s = c^H M c  (host, since KC is small)
 */
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

/*
 * Compute dN_s = M @ c  (host)
 */
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
    int64_t nd;        /* n_data    */
    int     jp, kc;    /* n_proj, n_comp */
    double  pur, Nb;
    double  Ns;

    /* ---- All data resident on GPU ---- */
    double2 *dF;       /* (nd, jp, kc)  complex  */
    double  *dw;       /* (nd,)                  */
    double  *dB;       /* (nd,)                  */
    double2 *hM;       /* (kc, kc) host copy     */

    /* ---- Workspace on GPU ---- */
    double2 *dA;       /* (nd, jp)    complex    */
    double  *dS;       /* (nd,)                  */
    double  *dP;       /* (nd,)                  */
    double2 *dG;       /* (nd, jp)    complex    */
    double  *dnll;     /* (1,)                   */
    double  *dscorr;   /* (1,)                   */
    double2 *dg;       /* (kc,)                  */
    double2 *dc;       /* (kc,)     coupling     */

    cudaStream_t strm;
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

/* ----------------------------------------------------------------
   fpw_create — upload ALL data to GPU
   ---------------------------------------------------------------- */

int fpw_create(int64_t nd, int jp, int kc,
               const double *Fd, const double *wd, const double *Bd,
               const double *M,
               double Nb, double pur, FpwFitter **out)
{
    if (nd <= 0 || jp <= 0 || kc <= 0)
        return FPW_ERR_ALLOC;

    FpwFitter *f = (FpwFitter *)calloc(1, sizeof(*f));
    if (!f) return FPW_ERR_ALLOC;

    f->nd = nd; f->jp = jp; f->kc = kc;
    f->pur = pur; f->Nb = Nb;
    f->Ns = 0.0;

    /* Host copy of M */
    f->hM = (double2 *)malloc((int64_t)kc * kc * sizeof(double2));
    if (!f->hM) { free(f); return FPW_ERR_ALLOC; }
    memcpy(f->hM, M, (int64_t)kc * kc * sizeof(double2));

    cudaError_t ce;
    int64_t szF  = (int64_t)nd * jp * kc * sizeof(double2);
    int64_t sz1  = (int64_t)nd * sizeof(double);
    int64_t szA  = (int64_t)nd * jp * sizeof(double2);
    int64_t szG  = (int64_t)nd * jp * sizeof(double2);
    int64_t szg  = (int64_t)kc  * sizeof(double2);

    ce = cudaStreamCreate(&f->strm);
    if (ce != cudaSuccess) goto fail;

    #define DA(p, sz) do { ce = cudaMalloc((void **)&(p), sz); \
                           if (ce != cudaSuccess) goto fail; } while (0)

    DA(f->dF, szF);
    DA(f->dw, sz1);
    DA(f->dB, sz1);
    DA(f->dA, szA);
    DA(f->dS, sz1);
    DA(f->dP, sz1);
    DA(f->dG, szG);
    DA(f->dnll, sizeof(double));
    DA(f->dscorr, sizeof(double));
    DA(f->dg, szg);
    DA(f->dc, (int64_t)kc * sizeof(double2));
    #undef DA

    /* ---- Upload all data to GPU ---- */
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

/* ----------------------------------------------------------------
   fpw_destroy
   ---------------------------------------------------------------- */

void fpw_destroy(FpwFitter *f) {
    if (!f) return;
    cudaFree(f->dF);
    cudaFree(f->dw);
    cudaFree(f->dB);
    cudaFree(f->dA);
    cudaFree(f->dS);
    cudaFree(f->dP);
    cudaFree(f->dG);
    cudaFree(f->dnll);
    cudaFree(f->dscorr);
    cudaFree(f->dg);
    cudaFree(f->dc);
    free(f->hM);
    if (f->strm) cudaStreamDestroy(f->strm);
    free(f);
}

/* ----------------------------------------------------------------
   fpw_evaluate — entirely on GPU, copy back only NLL + gradient
   ---------------------------------------------------------------- */

int fpw_evaluate(FpwFitter *f,
                 const double *cr, const double *ci,
                 double *nll,
                 double *gr, double *gi,
                 double *P_out)
{
    cudaError_t ce;
    int  kc = f->kc, jp = f->jp;
    int64_t nd = f->nd;

    /* ---- Copy c to GPU ---- */
    double2 *h_c = (double2 *)malloc(kc * sizeof(double2));
    if (!h_c) return FPW_ERR_ALLOC;
    for (int k = 0; k < kc; k++)
        h_c[k] = cset(cr[k], ci[k]);
    cudaMemcpyAsync(f->dc, h_c, kc * sizeof(double2),
                    cudaMemcpyHostToDevice, f->strm);

    /* ---- N_s = c^H M c  (host, kc is small) ---- */
    f->Ns = compute_Ns_host(f->hM, h_c, kc);
    if (f->Ns < 1e-300) f->Ns = 1e-300;
    if (f->Nb < 1e-300) f->Nb = 1e-300;

    /* ---- dN_s = M @ c  (host) ---- */
    double2 *dNs_h = (double2 *)malloc(kc * sizeof(double2));
    compute_dNs_host(f->hM, h_c, dNs_h, kc);

    /* ---- Zero GPU accumulators ---- */
    cudaMemsetAsync(f->dnll,   0, sizeof(double), f->strm);
    cudaMemsetAsync(f->dscorr, 0, sizeof(double), f->strm);
    cudaMemsetAsync(f->dg,     0, kc * sizeof(double2), f->strm);

    /* ---- Compute A[i,j] = sum_k F[i,j,k] * c[k] ---- */
    {
        int nb, nt;
        lcfg(nd * jp, &nb, &nt);
        k_A<<<nb, nt, 0, f->strm>>>(f->dF, f->dc, f->dA, nd, jp, kc);
    }

    /* ---- S[i] = sum_j |A[i,j]|^2 ---- */
    {
        int nb, nt;
        lcfg(nd, &nb, &nt);
        k_S<<<nb, nt, 0, f->strm>>>(f->dA, f->dS, nd, jp);
    }

    /* ---- P[i] and NLL ---- */
    {
        int nb, nt;
        lcfg(nd, &nb, &nt);
        k_PNLL<<<nb, nt, 0, f->strm>>>(f->dS, f->dB, f->dw, f->dP,
                                        f->dnll, nd, f->Ns, f->Nb, f->pur);
    }

    /* ---- G[i,j] = (w[i]/P[i]) * A[i,j] ---- */
    {
        int nb, nt;
        lcfg(nd, &nb, &nt);
        k_G<<<nb, nt, 0, f->strm>>>(f->dw, f->dP, f->dA, f->dG, nd, jp);
    }

    /* ---- Gradient: g[k] += sum_{i,j} conj(F[i,j,k]) * G[i,j] ---- */
    {
        int max_threads = 256;
        int64_t total = nd * jp;
        int blocks = (int)((total + max_threads - 1) / max_threads);
        if (blocks > 65535) blocks = 65535;
        int threads = (blocks < 65535) ? max_threads :
                      (int)((total + 65535 - 1) / 65535);
        if (threads < 1) threads = 1;

        for (int k = 0; k < kc; k++) {
            k_grad_accum<<<blocks, threads, 0, f->strm>>>(
                f->dF, f->dG, f->dg, nd, jp, kc, k);
        }
    }

    /* ---- S_corr = sum_i w[i]*S[i]/P[i] ---- */
    {
        int nb, nt;
        lcfg(nd, &nb, &nt);
        k_Scorr<<<nb, nt, 0, f->strm>>>(f->dw, f->dS, f->dP, f->dscorr, nd);
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

    /* ---- Assemble gradient (host) ---- */
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

    free(h_c);
    free(dNs_h);
    free(g_h);
    return FPW_SUCCESS;

copy_fail:
    cudaStreamSynchronize(f->strm);
    free(h_c); free(dNs_h); free(g_h);
    return FPW_ERR_CUDA;
}
