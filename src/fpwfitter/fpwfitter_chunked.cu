/* ================================================================
   fpwfitter_chunked.cu  –  Chunked FP64 Fitter (standalone)
   ================================================================
   F_data stays on HOST (complex128).  GPU workspace fits in VRAM.
   Data is uploaded chunk-by-chunk, results accumulated on host.

   Host F layout: (N, JP, KC) complex128 — row-major
   GPU F layout:  (KC, chunk, JP) complex128 — coalesced reads
   ================================================================ */

#include "fpwfitter_chunked.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ==================== complex helpers =========================== */
static __host__ __device__ __forceinline__ double2 cset(double r, double i) {
    return make_double2(r, i);
}

/* ==================== launch config ============================= */
static void lcfg(int64_t n, int *nb, int *nt) {
    *nt = 256;
    *nb = (int)((n + *nt - 1) / *nt);
    if (*nb > 65535) *nb = 65535;
}

/* ==================== chunked forward + gradient kernel ==========
 * F on GPU: (KC, chunk, JP) — coalesced reads (double2 = complex128).
 * Outputs: G (chunk, JP), scalars for NLL/S_corr, gradient accumulation.
 * ================================================================ */

static __device__ __forceinline__ double2 ldg_c(const double2 *p) {
    return __ldg(p);
}

__global__ void k_chunk_fp64(
    const double2 *__restrict__ F,       /* (KC, chunk, JP) */
    const double2 *__restrict__ c,       /* (KC,) */
    const double  *__restrict__ w,       /* (chunk,) */
    const double  *__restrict__ B,       /* (chunk,) */
    double2       *__restrict__ G,       /* (chunk, JP) */
    double        *__restrict__ P_out,   /* (chunk,) */
    double        *__restrict__ nll_out, /* (1,) */
    double        *__restrict__ scorr_out,/* (1,) */
    double2       *__restrict__ g,       /* (KC,) — atomic accumulation */
    int64_t chunk, int JP, int KC,
    double N_s, double N_b, double pur)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= chunk) return;

    /* ---- A[i,j] = sum_k F[k,i,j] * c[k] ---- */
    double A_re[4] = {0}, A_im[4] = {0};
    for (int j = 0; j < JP; j++) {
        double re = 0, im = 0;
        for (int k = 0; k < KC; k++) {
            double2 f  = ldg_c(&F[k * chunk * JP + i * JP + j]);
            double2 ck = ldg_c(&c[k]);
            re += f.x * ck.x - f.y * ck.y;
            im += f.x * ck.y + f.y * ck.x;
        }
        A_re[j] = re;
        A_im[j] = im;
    }

    /* ---- S, P, G ---- */
    double S = 0;
    for (int j = 0; j < JP; j++)
        S += A_re[j] * A_re[j] + A_im[j] * A_im[j];

    double p_val = S / N_s * pur + B[i] / N_b * (1.0 - pur);
    if (p_val < 1e-300) p_val = 1e-300;

    double ratio = w[i] / p_val;
    for (int j = 0; j < JP; j++)
        G[i * JP + j] = cset(A_re[j] * ratio, A_im[j] * ratio);

    /* ---- Gradient: g[k] += conj(F[k,i,j]) * G[i,j] ---- */
    for (int k = 0; k < KC; k++) {
        double gre = 0, gim = 0;
        int64_t fidx = k * chunk * JP + i * JP;
        for (int j = 0; j < JP; j++) {
            double2 f  = ldg_c(&F[fidx + j]);
            double2 gv = G[i * JP + j];
            gre += f.x * gv.x + f.y * gv.y;
            gim += f.x * gv.y - f.y * gv.x;
        }
        atomicAdd(&g[k].x, gre);
        atomicAdd(&g[k].y, gim);
    }

    /* ---- NLL, S_corr ---- */
    double nll_i   = -w[i] * log(p_val);
    double scorr_i =  w[i] * S / p_val;
    if (P_out) P_out[i] = p_val;

    for (int offset = 16; offset > 0; offset >>= 1)
        nll_i += __shfl_down_sync(0xffffffff, nll_i, offset);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(nll_out, nll_i);

    for (int offset = 16; offset > 0; offset >>= 1)
        scorr_i += __shfl_down_sync(0xffffffff, scorr_i, offset);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(scorr_out, scorr_i);
}

/* ==================== Host helpers (FP64) ======================= */

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

struct FpwFitterChunked {
    int64_t nd;
    int     jp, kc;
    double  pur, Nb, Ns;

    /* Host data (kept, uploaded chunk-by-chunk) */
    const double *hF;     /* (N, JP, KC) complex128, host */
    const double *hw;     /* (N,) float64 */
    const double *hB;     /* (N,) float64 */
    double2      *hM;     /* (KC, KC) host */
    double2      *h_g;    /* (KC,) gradient accumulator */

    /* GPU workspace — sized for ONE chunk */
    int64_t chunk;       /* events per chunk */
    int64_t n_chunks;

    double2 *dF;         /* (KC, chunk, JP) — coalesced on GPU */
    double  *dw;         /* (chunk,) */
    double  *dB;         /* (chunk,) */
    double2 *dG;         /* (chunk, JP) */
    double  *dP;         /* (chunk,) */
    double  *dnll;       /* (1,) */
    double  *dscorr;     /* (1,) */
    double2 *dg;         /* (KC,) — per-chunk gradient */
    double2 *dc;         /* (KC,) — coupling vector on GPU */

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
int    fpw_chunked_get_n_comp(const FpwFitterChunked *f) { return f->kc; }
double fpw_chunked_get_N_s   (const FpwFitterChunked *f) { return f->Ns; }
double fpw_chunked_get_N_b   (const FpwFitterChunked *f) { return f->Nb; }

/* ---------------------------------------------------------------- */

int fpw_chunked_create(int64_t nd, int jp, int kc,
                       const double *Fd, const double *wd, const double *Bd,
                       const double *M,
                       double Nb, double pur,
                       int64_t max_vram_mb,
                       FpwFitterChunked **out)
{
    if (nd <= 0 || jp <= 0 || kc <= 0) return FPW_ERR_ALLOC;

    FpwFitterChunked *f = (FpwFitterChunked *)calloc(1, sizeof(*f));
    if (!f) return FPW_ERR_ALLOC;

    f->nd = nd; f->jp = jp; f->kc = kc;
    f->pur = pur; f->Nb = Nb; f->Ns = 0.0;
    f->hF = Fd; f->hw = wd; f->hB = Bd;

    f->hM = (double2 *)malloc((int64_t)kc * kc * sizeof(double2));
    f->h_g = (double2 *)malloc((int64_t)kc * sizeof(double2));
    if (!f->hM || !f->h_g) { free(f); return FPW_ERR_ALLOC; }
    memcpy(f->hM, M, (int64_t)kc * kc * sizeof(double2));

    /* Determine chunk size: fit within max_vram_mb */
    int64_t max_vram_bytes = max_vram_mb > 0 ? max_vram_mb * 1024 * 1024 : 4LL * 1024 * 1024 * 1024;
    /* GPU memory per event (FP64):
     *   dF: KC*JP*16, dG: JP*16, dP:8, dw:8, dB:8 ≈ (KC*JP*2 + JP*2 + 24) bytes/event */
    int64_t bytes_per_event = kc * jp * 16 + jp * 16 + 24;
    int64_t chunk = max_vram_bytes / bytes_per_event;
    if (chunk > nd) chunk = nd;
    if (chunk < 1024) chunk = 1024;  /* minimum */
    f->chunk = chunk;
    f->n_chunks = (nd + chunk - 1) / chunk;

    cudaError_t    ce;
    cublasStatus_t cb;
    int64_t szF, sz1, szG;

    ce = cudaStreamCreate(&f->strm);
    if (ce != cudaSuccess) goto fail;
    cb = cublasCreate(&f->hdl);
    if (cb != CUBLAS_STATUS_SUCCESS) goto fail;
    cublasSetStream(f->hdl, f->strm);

    /* Workspace sized for ONE chunk */
    szF  = (int64_t)kc * chunk * jp * sizeof(double2);  /* (KC, chunk, JP) */
    sz1  = (int64_t)chunk * sizeof(double);
    szG  = (int64_t)chunk * jp * sizeof(double2);

    #define DA(p, sz) do { ce = cudaMalloc((void **)&(f->p), sz); \
                           if (ce != cudaSuccess) goto fail; } while (0)

    DA(dF,     szF);
    DA(dw,     sz1);
    DA(dB,     sz1);
    DA(dG,     szG);
    DA(dP,     sz1);
    DA(dnll,   sizeof(double));
    DA(dscorr, sizeof(double));
    DA(dg,     (int64_t)kc * sizeof(double2));
    DA(dc,     (int64_t)kc * sizeof(double2));
    #undef DA

    cudaStreamSynchronize(f->strm);
    *out = f;
    return FPW_SUCCESS;

fail:
    fpw_chunked_destroy(f);
    return FPW_ERR_ALLOC;
}

/* ---------------------------------------------------------------- */

void fpw_chunked_destroy(FpwFitterChunked *f) {
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
    free(f->hM);
    free(f->h_g);
    if (f->hdl)  cublasDestroy(f->hdl);
    if (f->strm) cudaStreamDestroy(f->strm);
    free(f);
}

/* ---------------------------------------------------------------- */

int fpw_chunked_evaluate(FpwFitterChunked *f,
                         const double *cr, const double *ci,
                         double *nll,
                         double *gr, double *gi,
                         double *P_out)
{
    cudaError_t    ce;
    int    kc = f->kc, jp = f->jp;
    int64_t nd = f->nd, chunk = f->chunk;

    /* ---- Convert c to GPU (FP64) ---- */
    double2 *h_c = (double2 *)malloc(kc * sizeof(double2));
    if (!h_c) return FPW_ERR_ALLOC;
    for (int k = 0; k < kc; k++)
        h_c[k] = cset(cr[k], ci[k]);
    cudaMemcpy(f->dc, h_c, kc * sizeof(double2), cudaMemcpyHostToDevice);

    /* ---- N_s (host, FP64) ---- */
    f->Ns = compute_Ns_host(f->hM, h_c, kc);
    if (f->Ns < 1e-300) f->Ns = 1e-300;
    if (f->Nb < 1e-300) f->Nb = 1e-300;

    double2 *dNs_h = (double2 *)malloc(kc * sizeof(double2));
    compute_dNs_host(f->hM, h_c, dNs_h, kc);

    /* ---- Zero gradient accumulator ---- */
    memset(f->h_g, 0, kc * sizeof(double2));

    double h_nll_total = 0.0;
    double h_scorr_total = 0.0;

    /* ---- Process chunks ---- */
    for (int64_t ic = 0; ic < f->n_chunks; ic++) {
        int64_t off = ic * chunk;
        int64_t cur = (off + chunk > nd) ? (nd - off) : chunk;

        /* ---- Upload chunk to GPU: host (N, JP, KC) → GPU (KC, chunk, JP) ---- */
        /* Transpose during upload using vectorized copy. */
        {
            const double2 *F_in = (const double2 *)(f->hF + off * jp * kc * 2);
            double2 *h_trans = (double2 *)malloc((int64_t)kc * cur * jp * sizeof(double2));
            if (!h_trans) { free(h_c); free(dNs_h); return FPW_ERR_ALLOC; }

            /* Vectorized transpose: read (cur, JP, KC), write (KC, cur, JP) */
            for (int64_t i = 0; i < cur; i++) {
                for (int j = 0; j < jp; j++) {
                    const double2 *src = &F_in[i * jp * kc + j * kc];
                    for (int k = 0; k < kc; k++) {
                        h_trans[k * cur * jp + i * jp + j] = src[k];
                    }
                }
            }
            ce = cudaMemcpy(f->dF, h_trans,
                            (int64_t)kc * cur * jp * sizeof(double2),
                            cudaMemcpyHostToDevice);
            free(h_trans);
            if (ce != cudaSuccess) { free(h_c); free(dNs_h); return FPW_ERR_CUDA; }
        }

        /* Upload w and B for this chunk */
        ce = cudaMemcpy(f->dw, f->hw + off, cur * sizeof(double), cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) { free(h_c); free(dNs_h); return FPW_ERR_CUDA; }
        ce = cudaMemcpy(f->dB, f->hB + off, cur * sizeof(double), cudaMemcpyHostToDevice);
        if (ce != cudaSuccess) { free(h_c); free(dNs_h); return FPW_ERR_CUDA; }

        /* ---- Zero per-chunk accumulators ---- */
        cudaMemset(f->dnll,   0, sizeof(double));
        cudaMemset(f->dscorr, 0, sizeof(double));
        cudaMemset(f->dg,     0, kc * sizeof(double2));

        /* ---- Chunk kernel ---- */
        { int nb, nt; lcfg(cur, &nb, &nt);
          k_chunk_fp64<<<nb, nt, 0, f->strm>>>(
              f->dF, f->dc, f->dw, f->dB,
              f->dG, P_out ? f->dP : NULL,
              f->dnll, f->dscorr, f->dg,
              cur, jp, kc, f->Ns, f->Nb, f->pur); }

        /* ---- Copy chunk results to host ---- */
        double h_nll = 0, h_sc = 0;
        double2 *g_chunk = (double2 *)malloc(kc * sizeof(double2));

        ce = cudaMemcpyAsync(&h_nll,   f->dnll,   sizeof(double),
                             cudaMemcpyDeviceToHost, f->strm);
        if (ce != cudaSuccess) { free(h_c); free(dNs_h); free(g_chunk); return FPW_ERR_CUDA; }
        ce = cudaMemcpyAsync(&h_sc,    f->dscorr, sizeof(double),
                             cudaMemcpyDeviceToHost, f->strm);
        if (ce != cudaSuccess) { free(h_c); free(dNs_h); free(g_chunk); return FPW_ERR_CUDA; }
        ce = cudaMemcpyAsync(g_chunk,  f->dg,     kc * sizeof(double2),
                             cudaMemcpyDeviceToHost, f->strm);
        if (ce != cudaSuccess) { free(h_c); free(dNs_h); free(g_chunk); return FPW_ERR_CUDA; }

        /* Copy P values if requested */
        if (P_out) {
            ce = cudaMemcpyAsync(P_out + off, f->dP, cur * sizeof(double),
                                 cudaMemcpyDeviceToHost, f->strm);
            if (ce != cudaSuccess) { free(h_c); free(dNs_h); free(g_chunk); return FPW_ERR_CUDA; }
        }

        cudaStreamSynchronize(f->strm);

        /* Accumulate */
        h_nll_total   += (double)h_nll;
        h_scorr_total += (double)h_sc;
        for (int k = 0; k < kc; k++) {
            f->h_g[k].x += g_chunk[k].x;
            f->h_g[k].y += g_chunk[k].y;
        }
        free(g_chunk);
    }

    /* ---- Assemble gradient ---- */
    double p = f->pur, Ns = f->Ns, Ns2 = Ns * Ns;
    double f1 = -p / Ns, f2 = p / Ns2 * h_scorr_total;

    for (int k = 0; k < kc; k++) {
        gr[k] = f1 * f->h_g[k].x + f2 * dNs_h[k].x;
        gi[k] = f1 * f->h_g[k].y + f2 * dNs_h[k].y;
    }
    *nll = h_nll_total;

    free(h_c); free(dNs_h);
    return FPW_SUCCESS;
}
