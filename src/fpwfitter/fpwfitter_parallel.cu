/* ================================================================
   fpwfitter_parallel.cu  –  Multi-Stream Parallel Fitter (C level)
   ================================================================
   Creates N independent sub-fitters, each with its own F data.
   Runs all in parallel CUDA streams, sums results on host.
   No complex offset calculations — each fitter owns its data.
   ================================================================ */

#include "fpwfitter_parallel.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        return FPW_ERR_CUDA; \
    } } while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t st = call; \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error: %d\n", st); \
        return FPW_ERR_CUBLAS; \
    } } while(0)

static __host__ __device__ __forceinline__ double2 cset(double r, double i) {
    return make_double2(r, i);
}

static void lcfg(int64_t n, int *nb, int *nt) {
    *nt = 256;
    *nb = (int)((n + *nt - 1) / *nt);
    if (*nb > 65535) *nb = 65535;
}

__global__ void k_fused_par(
    const double2 *__restrict__ F,
    const double2 *__restrict__ c,
    const double  *__restrict__ B,
    const double  *__restrict__ w,
    double2       *__restrict__ G,
    double        *__restrict__ nll_out,
    double        *__restrict__ scorr_out,
    int64_t N, int JP, int KC,
    double N_s, double N_b, double pur)
{
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= N) return;

    double A_re[4] = {0}, A_im[4] = {0};
    for (int j = 0; j < JP; j++) {
        double re = 0.0, im = 0.0;
        for (int k = 0; k < KC; k++) {
            int64_t fidx = k * N * JP + i * JP + j;
            double2 f  = __ldg(&F[fidx]);
            double2 ck = __ldg(&c[k]);
            re += f.x * ck.x - f.y * ck.y;
            im += f.x * ck.y + f.y * ck.x;
        }
        A_re[j] = re;
        A_im[j] = im;
    }

    double S = 0.0;
    for (int j = 0; j < JP; j++)
        S += A_re[j] * A_re[j] + A_im[j] * A_im[j];

    double p_val = S / N_s * pur + B[i] / N_b * (1.0 - pur);
    if (p_val < 1e-300) p_val = 1e-300;

    double ratio = w[i] / p_val;
    for (int j = 0; j < JP; j++)
        G[i * JP + j] = cset(A_re[j] * ratio, A_im[j] * ratio);

    double nll_i = -w[i] * log(p_val);
    double scorr_i = w[i] * S / p_val;

    for (int offset = 16; offset > 0; offset >>= 1)
        nll_i += __shfl_down_sync(0xffffffff, nll_i, offset);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(nll_out, nll_i);

    for (int offset = 16; offset > 0; offset >>= 1)
        scorr_i += __shfl_down_sync(0xffffffff, scorr_i, offset);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(scorr_out, scorr_i);
}

__global__ void k_conjvec_par(const double2 *__restrict__ in, double2 *__restrict__ out, int64_t N) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= N) return;
    double2 v = __ldg(&in[i]);
    out[i] = cset(v.x, -v.y);
}

static double compute_Ns_host(const double2 *M, const double2 *c, int KC) {
    double re = 0.0;
    for (int k1 = 0; k1 < KC; k1++)
        for (int k2 = 0; k2 < KC; k2++) {
            double2 m = M[k1 * KC + k2], c1 = c[k1], c2 = c[k2];
            re += c1.x * (m.x * c2.x - m.y * c2.y) + c1.y * (m.x * c2.y + m.y * c2.x);
        }
    return re;
}

static void compute_dNs_host(const double2 *M, const double2 *c, double2 *dNs, int KC) {
    for (int k1 = 0; k1 < KC; k1++) {
        double re = 0.0, im = 0.0;
        for (int k2 = 0; k2 < KC; k2++) {
            double2 m = M[k1 * KC + k2], ck = c[k2];
            re += m.x * ck.x - m.y * ck.y;
            im += m.x * ck.y + m.y * ck.x;
        }
        dNs[k1] = cset(re, im);
    }
}

/* One sub-fitter per stream */
typedef struct {
    cudaStream_t   strm;
    cublasHandle_t hdl;
    /* Device memory */
    double2 *dF, *dG, *dc, *dg, *dGconj;
    double  *dw, *dB, *dnll, *dscorr;
    /* Host staging */
    double2 *h_g;
    double   h_nll, h_scorr;
    /* Chunk info */
    int64_t nd_chunk;
} SubFitter;

struct FpwFitterParallel {
    int     jp, kc, n_streams;
    double  pur, Nb, Ns;
    double2 *hM;
    double2 *h_g_sum;
    double2 *h_c;
    SubFitter *subs;
};

const char *fpw_par_strerror(int e) {
    switch (e) {
    case FPW_SUCCESS:    return "Success";
    case FPW_ERR_ALLOC:  return "Memory allocation failed";
    case FPW_ERR_CUDA:   return "CUDA runtime error";
    case FPW_ERR_CUBLAS: return "cuBLAS error";
    default:             return "Unknown error";
    }
}
int    fpw_par_get_n_comp(const FpwFitterParallel *f) { return f->kc; }
double fpw_par_get_N_s   (const FpwFitterParallel *f) { return f->Ns; }
double fpw_par_get_N_b   (const FpwFitterParallel *f) { return f->Nb; }

/* ---------------------------------------------------------------- */

/* Helper: transpose and upload one chunk */
static int upload_chunk(SubFitter *sf, const double *Fd_host, int64_t nd, int jp, int kc) {
    sf->nd_chunk = nd;
    int64_t nj = nd * jp;
    int64_t szF  = (int64_t)kc * nd * jp * sizeof(double2);
    int64_t sz1  = nd * sizeof(double);
    int64_t szG  = nj * sizeof(double2);
    int64_t szg  = kc * sizeof(double2);

    CUDA_CHECK(cudaMalloc((void **)&sf->dF,     szF));
    CUDA_CHECK(cudaMalloc((void **)&sf->dw,     sz1));
    CUDA_CHECK(cudaMalloc((void **)&sf->dB,     sz1));
    CUDA_CHECK(cudaMalloc((void **)&sf->dG,     szG));
    CUDA_CHECK(cudaMalloc((void **)&sf->dnll,   sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&sf->dscorr, sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&sf->dg,     szg));
    CUDA_CHECK(cudaMalloc((void **)&sf->dc,     szg));
    CUDA_CHECK(cudaMalloc((void **)&sf->dGconj, szG));
    sf->h_g = (double2 *)malloc(szg);

    /* Transpose F: (N, JP, KC) → (KC, N, JP) */
    double2 *h_trans = (double2 *)malloc(szF);
    if (!h_trans) return FPW_ERR_ALLOC;
    const double2 *F_in = (const double2 *)Fd_host;
    for (int64_t k = 0; k < kc; k++)
        for (int64_t i = 0; i < nd; i++)
            for (int j = 0; j < jp; j++)
                h_trans[k * nd * jp + i * jp + j] = F_in[i * jp * kc + j * kc + k];

    CUDA_CHECK(cudaMemcpyAsync(sf->dF, h_trans, szF, cudaMemcpyHostToDevice, sf->strm));
    free(h_trans);
    return FPW_SUCCESS;
}

int fpw_par_create(int64_t nd, int jp, int kc,
                   const double *Fd, const double *wd, const double *Bd,
                   const double *M,
                   double Nb, double pur,
                   int n_streams,
                   FpwFitterParallel **out)
{
    if (nd <= 0 || jp <= 0 || kc <= 0 || n_streams <= 0) return FPW_ERR_ALLOC;
    if (n_streams > 8) n_streams = 8;

    FpwFitterParallel *f = (FpwFitterParallel *)calloc(1, sizeof(*f));
    if (!f) return FPW_ERR_ALLOC;

    f->jp = jp; f->kc = kc;
    f->pur = pur; f->Nb = Nb; f->Ns = 0.0;
    f->n_streams = n_streams;

    f->hM = (double2 *)malloc((int64_t)kc * kc * sizeof(double2));
    f->h_g_sum = (double2 *)calloc(kc, sizeof(double2));
    f->h_c = (double2 *)malloc(kc * sizeof(double2));
    if (!f->hM || !f->h_g_sum || !f->h_c) {
        free(f->hM); free(f->h_g_sum); free(f->h_c); free(f);
        return FPW_ERR_ALLOC;
    }
    memcpy(f->hM, M, (int64_t)kc * kc * sizeof(double2));

    /* Split data into chunks */
    int64_t base = nd / n_streams;
    int64_t rem  = nd % n_streams;
    int64_t offset = 0;

    f->subs = (SubFitter *)calloc(n_streams, sizeof(SubFitter));
    if (!f->subs) { fpw_par_destroy(f); return FPW_ERR_ALLOC; }

    for (int s = 0; s < n_streams; s++) {
        SubFitter *sf = &f->subs[s];
        int64_t chunk_nd = base + (s < rem ? 1 : 0);

        CUDA_CHECK(cudaStreamCreate(&sf->strm));
        CUBLAS_CHECK(cublasCreate(&sf->hdl));
        CUBLAS_CHECK(cublasSetStream(sf->hdl, sf->strm));

        int rc = upload_chunk(sf, Fd + offset * jp * kc * 2, chunk_nd, jp, kc);
        if (rc != FPW_SUCCESS) { fpw_par_destroy(f); return rc; }
        CUDA_CHECK(cudaMemcpyAsync(sf->dw, wd + offset, chunk_nd * sizeof(double),
                                   cudaMemcpyHostToDevice, sf->strm));
        CUDA_CHECK(cudaMemcpyAsync(sf->dB, Bd + offset, chunk_nd * sizeof(double),
                                   cudaMemcpyHostToDevice, sf->strm));
        offset += chunk_nd;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    *out = f;
    return FPW_SUCCESS;
}

/* ---------------------------------------------------------------- */

void fpw_par_destroy(FpwFitterParallel *f) {
    if (!f) return;
    if (f->subs) {
        for (int s = 0; s < f->n_streams; s++) {
            SubFitter *sf = &f->subs[s];
            cudaFree(sf->dF); cudaFree(sf->dw); cudaFree(sf->dB);
            cudaFree(sf->dG); cudaFree(sf->dnll); cudaFree(sf->dscorr);
            cudaFree(sf->dg); cudaFree(sf->dc); cudaFree(sf->dGconj);
            free(sf->h_g);
            if (sf->hdl)  cublasDestroy(sf->hdl);
            if (sf->strm) cudaStreamDestroy(sf->strm);
        }
        free(f->subs);
    }
    free(f->hM); free(f->h_g_sum); free(f->h_c);
    free(f);
}

/* ---------------------------------------------------------------- */

int fpw_par_evaluate(FpwFitterParallel *f,
                     const double *cr, const double *ci,
                     double *nll, double *gr, double *gi,
                     double *P_data)
{
    int kc = f->kc, jp = f->jp;

    /* Prepare c (same for all sub-fitters) */
    for (int k = 0; k < kc; k++) f->h_c[k] = make_double2(cr[k], ci[k]);
    f->Ns = compute_Ns_host(f->hM, f->h_c, kc);
    if (f->Ns < 1e-300) f->Ns = 1e-300;
    if (f->Nb < 1e-300) f->Nb = 1e-300;

    double2 *dNs_h = (double2 *)malloc(kc * sizeof(double2));
    if (!dNs_h) return FPW_ERR_ALLOC;
    compute_dNs_host(f->hM, f->h_c, dNs_h, kc);

    /* Launch all sub-fitters in parallel */
    for (int s = 0; s < f->n_streams; s++) {
        SubFitter *sf = &f->subs[s];
        int64_t N = sf->nd_chunk;
        int64_t nj = N * jp;

        /* Copy c to device */
        CUDA_CHECK(cudaMemcpyAsync(sf->dc, f->h_c, kc * sizeof(double2),
                                   cudaMemcpyHostToDevice, sf->strm));

        /* Zero accumulators */
        CUDA_CHECK(cudaMemsetAsync(sf->dnll,   0, sizeof(double), sf->strm));
        CUDA_CHECK(cudaMemsetAsync(sf->dscorr, 0, sizeof(double), sf->strm));
        CUDA_CHECK(cudaMemsetAsync(sf->dg,     0, kc * sizeof(double2), sf->strm));

        /* Forward fused kernel */
        { int nb, nt; lcfg(N, &nb, &nt);
          k_fused_par<<<nb, nt, 0, sf->strm>>>(
              sf->dF, sf->dc, sf->dB, sf->dw,
              sf->dG, sf->dnll, sf->dscorr,
              N, jp, kc, f->Ns, f->Nb, f->pur); }

        /* Gradient: conjugate G, then ZGEMV */
        {
            int nb, nt; lcfg(nj, &nb, &nt);
            k_conjvec_par<<<nb, nt, 0, sf->strm>>>(sf->dG, sf->dGconj, nj);

            cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
            cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
            CUBLAS_CHECK(cublasZgemv(sf->hdl, CUBLAS_OP_T,
                        (int)nj, kc, &alpha,
                        (const cuDoubleComplex *)sf->dF, (int)nj,
                        (const cuDoubleComplex *)sf->dGconj, 1,
                        &beta,
                        (cuDoubleComplex *)sf->dg, 1));
        }
    }

    /* Serialize results on host */
    double h_nll_total = 0.0, h_scorr_total = 0.0;
    memset(f->h_g_sum, 0, kc * sizeof(double2));

    for (int s = 0; s < f->n_streams; s++) {
        SubFitter *sf = &f->subs[s];
        CUDA_CHECK(cudaStreamSynchronize(sf->strm));

        CUDA_CHECK(cudaMemcpy(&sf->h_nll, sf->dnll, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&sf->h_scorr, sf->dscorr, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sf->h_g, sf->dg, kc * sizeof(double2), cudaMemcpyDeviceToHost));

        h_nll_total   += sf->h_nll;
        h_scorr_total += sf->h_scorr;
        for (int k = 0; k < kc; k++) {
            /* ZGEMV computes y[k] = sum F[k,i,jp] * conj(G[i,jp])
             * But we need g_data[k] = sum conj(F[k,i,jp]) * G[i,jp] = conj(y[k])
             * conj(y).real = y.real, conj(y).imag = -y.imag */
            f->h_g_sum[k].x += sf->h_g[k].x;
            f->h_g_sum[k].y -= sf->h_g[k].y;
        }
    }

    /* Assemble gradient */
    double p = f->pur, Ns = f->Ns, Ns2 = Ns * Ns;
    double f1 = -p / Ns, f2 = p / Ns2 * h_scorr_total;

    for (int k = 0; k < kc; k++) {
        gr[k] = f1 * f->h_g_sum[k].x + f2 * dNs_h[k].x;
        gi[k] = f1 * f->h_g_sum[k].y + f2 * dNs_h[k].y;
    }
    *nll = h_nll_total;

    free(dNs_h);
    return FPW_SUCCESS;
}
