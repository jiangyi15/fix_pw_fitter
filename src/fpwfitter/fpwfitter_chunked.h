#ifndef FPWFITTER_CHUNKED_H
#define FPWFITTER_CHUNKED_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FpwFitterChunked FpwFitterChunked;

#define FPW_SUCCESS    0
#define FPW_ERR_ALLOC -1
#define FPW_ERR_CUDA  -2
#define FPW_ERR_CUBLAS -3

/**
 * Chunked FP64 fitter — processes data in chunks that fit in VRAM.
 * F_data is kept on host (FP64) and uploaded chunk-by-chunk.
 * Only GPU workspace needs to fit in VRAM.
 *
 * @param F_data_host  host pointer: (N, JP, KC) complex128
 * @param max_vram_mb  maximum VRAM to use for F_data (0 = auto ~4 GB)
 */
int fpw_chunked_create(
    int64_t      n_data,
    int          n_proj,
    int          n_comp,
    const double *F_data_host,  /* (N, JP, KC) complex128, host */
    const double *w_data,       /* (N,)     float64 */
    const double *B_data,       /* (N,)     float64 */
    const double *M,            /* (KC, KC) complex128 */
    double       N_b,
    double       purity,
    int64_t      max_vram_mb,   /* max VRAM for F_data, 0=auto */
    FpwFitterChunked **out);

void fpw_chunked_destroy(FpwFitterChunked *f);

int fpw_chunked_evaluate(
    FpwFitterChunked *f,
    const double *c_real,  const double *c_imag,
    double *nll,
    double *grad_real,     double *grad_imag,
    double *P_data          /* optional, (N,) float64 */);

int    fpw_chunked_get_n_comp(const FpwFitterChunked *f);
double fpw_chunked_get_N_s   (const FpwFitterChunked *f);
double fpw_chunked_get_N_b   (const FpwFitterChunked *f);
const char *fpw_strerror(int err);

#ifdef __cplusplus
}
#endif
#endif
