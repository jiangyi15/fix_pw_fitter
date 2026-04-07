/* ================================================================
   fpwfitter.cu  –  Fixed Partial Waves Fitter  (CUDA)
   ================================================================

   DATA LAYOUT (matches NumPy complex128, C-order):
     F_data : (n_data, n_proj, n_comp)   complex128
     F_mc   : (n_mc,   n_proj, n_comp)   complex128
     All 1-D arrays: (N,)  float64

   ALGORITHM (optimised order):

   Pre-compute (fpw_create):
     M[k1,k2] = sum_{i',j} w_mc[i'] F_mc[i',j,k1] conj(F_mc[i',j,k2])
     N_b      = sum_{i'} w_mc[i'] B_mc[i']

   Per evaluate:
     A[i,j]   = sum_k F_data[i,j,k] c[k]
     S[i]     = sum_j |A[i,j]|^2
     N_s      = sum_{k1,k2} conj(c[k1]) M[k1,k2] c[k2]
     P[i]     = S[i]/N_s*p + B[i]/N_b*(1-p)
     NLL      = -sum_i w[i] log(P[i])
     G[i,j]   = (w[i]/P[i]) * A[i,j]
     g[k]    += sum_{i,j} conj(F_data[i,j,k]) * G[i,j]
     dN_s[k]  = sum_{k2} M[k,k2] c[k2]
     S_corr   = sum_i w[i]*S[i]/P[i]
     grad[k]  = -p/N_s*g[k] + p/N_s^2*dN_s[k]*S_corr
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
static __host__ __device__ __forceinline__ double2 cmul(double2 a, double2 b) {
    return make_double2(a.x * b.x - a.y * b.y,
                       a.x * b.y + a.y * b.x);
}
static __host__ __device__ __forceinline__ double2 cmul_conjL(double2 a, double2 b) {
    /* conj(a) * b */
    return make_double2(a.x * b.x + a.y * b.y,
                       a.x * b.y - a.y * b.x);
}
static __host__ __device__ __forceinline__ double2 cscl(double2 a, double s) {
    return make_double2(a.x * s, a.y * s);
}
static __host__ __device__ __forceinline__ double2 cadd(double2 a, double2 b) {
    return make_double2(a.x + b.x, a.y + b.y);
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
 * F:  (N, JP, KC)  complex
 * c:  (KC,)        complex
 * A:  (N, JP)      complex   — output
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
 * P[i] and NLL
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
        G[i * JP + j] = cscl(A[i * JP + j], r);
}

/*
 * S_corr accumulator
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
 * Each block handles one k value, loops over all (i,j).
 */
__global__ void k_grad_accum(const double2 *F, const double2 *G,
                              double2 *g,
                              int64_t N, int JP, int KC, int k_idx) {
    /* Each thread handles a subset of (i,j) pairs */
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
 * Launch one kernel per k to accumulate gradient.
 * For small KC (≤ 200), launching KC kernels is fine.
 * For large KC, we'd use a single kernel with grid-stride over (i,j,k).
 */
static void launch_grad_kernel(const double2 *F, const double2 *G,
                                double2 *g, int64_t N, int JP, int KC,
                                cudaStream_t strm) {
    /* For kc ≤ 200, launch one block per k */
    int max_threads = 256;
    int64_t total = N * JP;
    int blocks = (int)((total + max_threads - 1) / max_threads);
    if (blocks > 65535) blocks = 65535;
    int threads = (blocks < 65535) ? max_threads :
                  (int)((total + 65535 - 1) / 65535);
    if (threads < 1) threads = 1;

    for (int k = 0; k < KC; k++) {
        k_grad_accum<<<blocks, threads, 0, strm>>>(F, G, g, N, JP, KC, k);
    }
}

/*
 * Compute N_s = c^H M c  (host, since KC is small)
 */
static double compute_Ns_host(const double2 *M, const double2 *c, int KC) {
    double re = 0.0;
    for (int k1 = 0; k1 < KC; k1++) {
        for (int k2 = 0; k2 < KC; k2++) {
            double2 m = M[k1 * KC + k2];
            double2 c1 = c[k1];
            double2 c2 = c[k2];
            /* conj(c1) * m * c2 */
            /* First: m * c2 */
            double t_re = m.x * c2.x - m.y * c2.y;
            double t_im = m.x * c2.y + m.y * c2.x;
            /* Then: conj(c1) * (m*c2) */
            re += c1.x * t_re + c1.y * t_im;
        }
    }
    return re;
}

/*
 * Compute dN_s = M @ c
 */
static void compute_dNs_host(const double2 *M, const double2 *c,
                              double2 *dNs, int KC) {
    for (int k1 = 0; k1 < KC; k1++) {
        double re = 0.0, im = 0.0;
        for (int k2 = 0; k2 < KC; k2++) {
            double2 m = M[k1 * KC + k2];
            double2 ck = c[k2];
            re += m.x * ck.x - m.y * ck.y;
            im += m.x * ck.y + m.y * ck.x;
        }
        dNs[k1] = cset(re, im);
    }
}

/*
 * Compute M from a chunk of MC data.
 * F_mc_chunk: (cur, JP, KC) complex
 * w_mc_chunk: (cur,) real
 * M: (KC, KC) complex  — accumulated (read-modify-write on host)
 */
static void compute_M_chunk_host(const double *F_mc, const double *w_mc,
                                  double2 *M,
                                  int64_t cur, int JP, int KC) {
    const double2 *F = (const double2 *)F_mc;
    for (int64_t ev = 0; ev < cur; ev++) {
        double w = w_mc[ev];
        for (int j = 0; j < JP; j++) {
            for (int k1 = 0; k1 < KC; k1++) {
                double2 f1 = F[ev * (int64_t)JP * KC + j * KC + k1];
                for (int k2 = 0; k2 < KC; k2++) {
                    double2 f2 = F[ev * (int64_t)JP * KC + j * KC + k2];
                    /* M[k1,k2] += w * conj(f1) * f2 */
                    double re = w * (f1.x * f2.x + f1.y * f2.y);
                    double im = w * (f1.x * f2.y - f1.y * f2.x);
                    M[k1 * KC + k2].x += re;
                    M[k1 * KC + k2].y += im;
                }
            }
        }
    }
}

/* ==================== Fitter state ============================== */

struct FpwFitter {
    int64_t nd, nm;    /* n_data, n_mc */
    int     jp, kc;    /* n_proj, n_comp */
    double  pur;
    int64_t cs;        /* chunk size */

    /* Host data — C-order (n_events, n_proj, n_comp) complex */
    const double *hFd;
    const double *hwd;
    const double *hBd;
    const double *hFm;
    const double *hwm;
    const double *hBm;

    /* Pre-computed overlap matrix (host) */
    double2 *hM;
    double   Nb;
    double   Ns;

    /* Device workspace — sized for one chunk */
    double2 *dF;       /* (cs, jp, kc) complex — full chunk */
    double  *dw;       /* (cs,) */
    double  *dB;       /* (cs,) */
    double2 *dA;       /* (cs, jp)    complex */
    double  *dS;       /* (cs,) */
    double  *dP;       /* (cs,) */
    double2 *dG;       /* (cs, jp)    complex */
    double  *dnll;     /* (1,) */
    double  *dscorr;   /* (1,) */
    double2 *dg;       /* (kc,)     — gradient accumulator */

    cublasHandle_t hdl;   /* kept for DDOT */
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

/* ----------------------------------------------------------------
   fpw_create
   ---------------------------------------------------------------- */

int fpw_create(int64_t nd, int64_t nm, int jp, int kc,
               const double *Fd, const double *Fm,
               const double *wd, const double *wm,
               const double *Bd, const double *Bm,
               double pur, int64_t csz, FpwFitter **out)
{
    if (nd <= 0 || nm <= 0 || jp <= 0 || kc <= 0)
        return FPW_ERR_ALLOC;
    if (csz <= 0) csz = 100000;
    if (csz > nm) csz = nm;

    FpwFitter *f = (FpwFitter *)calloc(1, sizeof(*f));
    if (!f) return FPW_ERR_ALLOC;

    f->nd = nd; f->nm = nm; f->jp = jp; f->kc = kc;
    f->pur = pur; f->cs = csz;
    f->hFd = Fd; f->hwd = wd; f->hBd = Bd;
    f->hFm = Fm; f->hwm = wm; f->hBm = Bm;
    f->Nb = 0.0; f->Ns = 0.0;

    cudaError_t    ce;
    int64_t        szF, sz1, szA, szG, szg, nch, stride;
    cublasStatus_t cb;

    ce = cudaStreamCreate(&f->strm);
    if (ce != cudaSuccess) goto fail;

    cb = cublasCreate(&f->hdl);
    if (cb != CUBLAS_STATUS_SUCCESS) goto fail;
    cublasSetStream(f->hdl, f->strm);

    /* ---- device allocations ---- */
    szF  = (int64_t)csz * jp * kc * sizeof(double2);
    sz1  = (int64_t)csz * sizeof(double);
    szA  = (int64_t)csz * jp * sizeof(double2);
    szG  = (int64_t)csz * jp * sizeof(double2);
    szg  = (int64_t)kc  * sizeof(double2);

    #define DA(p, sz) do { ce = cudaMalloc((void **)&(f->p), sz); \
                           if (ce != cudaSuccess) goto fail; } while (0)

    DA(dF,   szF);
    DA(dw,   sz1);
    DA(dB,   sz1);
    DA(dA,   szA);
    DA(dS,   sz1);
    DA(dP,   sz1);
    DA(dG,   szG);
    DA(dnll, sizeof(double));
    DA(dscorr, sizeof(double));
    DA(dg,   szg);
    #undef DA

    /* Host M (pinned) */
    ce = cudaMallocHost((void **)&f->hM, (int64_t)kc * kc * sizeof(double2));
    if (ce != cudaSuccess) goto fail;
    memset(f->hM, 0, (int64_t)kc * kc * sizeof(double2));

    /* ========= Pre-compute M and N_b (chunked, host-side) ======== */

    nch    = (nm + f->cs - 1) / f->cs;
    stride = (int64_t)jp * kc;

    for (int64_t ic = 0; ic < nch; ic++) {
        int64_t off = ic * f->cs;
        int64_t cur = (off + f->cs > nm) ? (nm - off) : f->cs;

        const double *Fchunk = Fm + off * stride * 2; /* *2 for complex doubles */
        const double *wchunk = wm + off;

        /* Compute M contribution on host (simple, correct) */
        compute_M_chunk_host(Fchunk, wchunk, f->hM, cur, jp, kc);
    }

    /* N_b = sum w_mc * B_mc */
    {
        for (int64_t ic = 0; ic < nch; ic++) {
            int64_t off = ic * f->cs;
            int64_t cur = (off + f->cs > nm) ? (nm - off) : f->cs;

            cudaMemcpyAsync(f->dw, wm + off,
                cur * sizeof(double), cudaMemcpyHostToDevice, f->strm);
            cudaMemcpyAsync(f->dB, Bm + off,
                cur * sizeof(double), cudaMemcpyHostToDevice, f->strm);

            double dot;
            cb = cublasDdot(f->hdl, (int)cur, f->dw, 1, f->dB, 1, &dot);
            if (cb != CUBLAS_STATUS_SUCCESS) { fpw_destroy(f); return FPW_ERR_CUBLAS; }
            f->Nb += dot;
        }
    }

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
    cudaFreeHost(f->hM);
    if (f->hdl)  cublasDestroy(f->hdl);
    if (f->strm) cudaStreamDestroy(f->strm);
    free(f);
}

/* ----------------------------------------------------------------
   fpw_evaluate
   ---------------------------------------------------------------- */

int fpw_evaluate(FpwFitter *f,
                 const double *cr, const double *ci,
                 double *nll,
                 double *gr, double *gi,
                 double *P_out)
{
    cudaError_t ce;
    int kc = f->kc, jp = f->jp;
    int64_t nd = f->nd, cs = f->cs;

    /* ---- Build coupling vector on device ---- */
    double2 *h_c = (double2 *)malloc(kc * sizeof(double2));
    if (!h_c) return FPW_ERR_ALLOC;
    for (int k = 0; k < kc; k++)
        h_c[k] = cset(cr[k], ci[k]);

    double2 *d_c;
    ce = cudaMalloc((void **)&d_c, kc * sizeof(double2));
    if (ce != cudaSuccess) { free(h_c); return FPW_ERR_ALLOC; }
    cudaMemcpyAsync(d_c, h_c, kc * sizeof(double2),
                    cudaMemcpyHostToDevice, f->strm);

    /* ---- N_s = c^H M c  (host) ---- */
    f->Ns = compute_Ns_host(f->hM, h_c, kc);
    if (f->Ns < 1e-300) f->Ns = 1e-300;
    if (f->Nb < 1e-300) f->Nb = 1e-300;

    /* ---- dN_s = M @ c  (host, since kc is small) ---- */
    double2 *dNs_h = (double2 *)malloc(kc * sizeof(double2));
    compute_dNs_host(f->hM, h_c, dNs_h, kc);

    /* ---- Zero accumulators ---- */
    double h_nll = 0.0, h_scorr = 0.0;

    /* ---- Gradient accumulator (host) ---- */
    double2 *g_h = (double2 *)calloc(kc, sizeof(double2));
    if (!g_h) { free(h_c); free(dNs_h); cudaFree(d_c); return FPW_ERR_ALLOC; }

    /* ---- Process data in chunks ---- */
    int64_t nch   = (nd + cs - 1) / cs;
    int64_t stride = (int64_t)jp * kc;

    for (int64_t ic = 0; ic < nch; ic++) {
        int64_t off = ic * cs;
        int64_t cur = (off + cs > nd) ? (nd - off) : cs;

        /* Copy chunk to GPU */
        cudaMemcpyAsync(f->dF,
            f->hFd + off * stride * 2,
            cur * jp * kc * sizeof(double2),
            cudaMemcpyHostToDevice, f->strm);
        cudaMemcpyAsync(f->dw,
            f->hwd + off,
            cur * sizeof(double),
            cudaMemcpyHostToDevice, f->strm);
        cudaMemcpyAsync(f->dB,
            f->hBd + off,
            cur * sizeof(double),
            cudaMemcpyHostToDevice, f->strm);

        /* A[i,j] = sum_k F[i,j,k] * c[k] */
        {
            int nb, nt;
            lcfg(cur * jp, &nb, &nt);
            k_A<<<nb, nt, 0, f->strm>>>((const double2 *)f->dF, d_c,
                                         f->dA, cur, jp, kc);
        }

        /* S[i] = sum_j |A[i,j]|^2 */
        {
            int nb, nt;
            lcfg(cur, &nb, &nt);
            k_S<<<nb, nt, 0, f->strm>>>(f->dA, f->dS, cur, jp);
        }

        /* P[i] and NLL */
        {
            double chunk_nll = 0.0;
            cudaMemcpyAsync(f->dnll, &chunk_nll, sizeof(double),
                            cudaMemcpyHostToDevice, f->strm);

            int nb, nt;
            lcfg(cur, &nb, &nt);
            k_PNLL<<<nb, nt, 0, f->strm>>>(f->dS, f->dB, f->dw, f->dP,
                                            f->dnll, cur,
                                            f->Ns, f->Nb, f->pur);

            cudaMemcpyAsync(&chunk_nll, f->dnll, sizeof(double),
                            cudaMemcpyDeviceToHost, f->strm);
            cudaStreamSynchronize(f->strm);
            h_nll += chunk_nll;
        }

        /* G[i,j] = (w[i]/P[i]) * A[i,j] */
        {
            int nb, nt;
            lcfg(cur, &nb, &nt);
            k_G<<<nb, nt, 0, f->strm>>>(f->dw, f->dP, f->dA, f->dG, cur, jp);
        }

        /* Gradient: g[k] += sum_{i,j} conj(F[i,j,k]) * G[i,j] */
        /* Zero device gradient accumulator */
        cudaMemsetAsync(f->dg, 0, kc * sizeof(double2), f->strm);

        launch_grad_kernel((const double2 *)f->dF, f->dG, f->dg,
                           cur, jp, kc, f->strm);

        /* Copy gradient contribution to host and accumulate */
        double2 *g_chunk = (double2 *)malloc(kc * sizeof(double2));
        cudaMemcpyAsync(g_chunk, f->dg, kc * sizeof(double2),
                        cudaMemcpyDeviceToHost, f->strm);
        cudaStreamSynchronize(f->strm);
        for (int k = 0; k < kc; k++) {
            g_h[k].x += g_chunk[k].x;
            g_h[k].y += g_chunk[k].y;
        }
        free(g_chunk);

        /* S_corr */
        {
            double chunk_sc = 0.0;
            cudaMemcpyAsync(f->dscorr, &chunk_sc, sizeof(double),
                            cudaMemcpyHostToDevice, f->strm);

            int nb, nt;
            lcfg(cur, &nb, &nt);
            k_Scorr<<<nb, nt, 0, f->strm>>>(f->dw, f->dS, f->dP,
                                             f->dscorr, cur);

            cudaMemcpyAsync(&chunk_sc, f->dscorr, sizeof(double),
                            cudaMemcpyDeviceToHost, f->strm);
            cudaStreamSynchronize(f->strm);
            h_scorr += chunk_sc;
        }

        /* Optional: copy P_i to output */
        if (P_out) {
            cudaMemcpyAsync(P_out + off, f->dP, cur * sizeof(double),
                            cudaMemcpyDeviceToHost, f->strm);
        }
    }

    cudaStreamSynchronize(f->strm);

    /* ---- Assemble gradient ---- */
    double p = f->pur;
    double Ns = f->Ns;
    double Ns2 = Ns * Ns;
    double factor1 = -p / Ns;
    double factor2 = p / Ns2 * h_scorr;

    for (int k = 0; k < kc; k++) {
        /* grad[k] = factor1 * g_h[k] + factor2 * dNs_h[k] */
        gr[k] = factor1 * g_h[k].x + factor2 * dNs_h[k].x;
        gi[k] = factor1 * g_h[k].y + factor2 * dNs_h[k].y;
    }

    *nll = h_nll;

    free(h_c);
    free(dNs_h);
    free(g_h);
    cudaFree(d_c);

    return FPW_SUCCESS;
}
