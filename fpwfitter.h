#ifndef FPWFITTER_H
#define FPWFITTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FpwFitter FpwFitter;

#define FPW_SUCCESS          0
#define FPW_ERR_ALLOC       -1
#define FPW_ERR_INVALID     -2
#define FPW_ERR_CUBLAS      -3
#define FPW_ERR_CUDA        -4

/**
 * Create a fitter.  All array pointers are host (CPU) pointers.
 * The data is kept on the host and transferred in chunks to the GPU.
 *
 * Memory layout (C-order / row-major, matching numpy default):
 *   F_data : (n_data, n_proj, n_comp)  complex128  —  interleaved (re, im)
 *   F_mc   : (n_mc,   n_proj, n_comp)  complex128
 *   w_data : (n_data,)                   float64
 *   w_mc   : (n_mc,)                     float64
 *   B_data : (n_data,)                   float64  —  background at *data* events
 *   B_mc   : (n_mc,)                     float64  —  background at MC events
 *
 * chunk_size: how many events to transfer to GPU in one batch.
 *   0 means auto (100 000).  A 100 k-event chunk needs ~160 MB for F.
 */
int fpw_create(
    int64_t  n_data,
    int64_t  n_mc,
    int      n_proj,
    int      n_comp,
    const double *F_data,
    const double *F_mc,
    const double *w_data,
    const double *w_mc,
    const double *B_data,
    const double *B_mc,
    double   purity,
    int64_t  chunk_size,
    FpwFitter **out
);

void fpw_destroy(FpwFitter *f);

/**
 * Pre-compute M_{kk'} from MC and N_b.  Must be called once before
 * fpw_evaluate().  Processes MC in chunks to keep GPU memory low.
 */
int fpw_precompute(FpwFitter *f);

/**
 * Evaluate -log L and its gradient at the given coupling vector c.
 *
 * c_real / c_imag : shape (n_comp,)
 * nll             : output scalar
 * grad_real/imag  : shape (n_comp,), d(-ln L)/d(c_k^*)
 * P_data          : optional, shape (n_data,), receive P_i values (NULL to skip)
 */
int fpw_evaluate(
    FpwFitter *f,
    const double *c_real,
    const double *c_imag,
    double *nll,
    double *grad_real,
    double *grad_imag,
    double *P_data
);

int     fpw_get_n_comp(const FpwFitter *f);
double  fpw_get_N_s   (const FpwFitter *f);
double  fpw_get_N_b   (const FpwFitter *f);
const char *fpw_strerror(int err);

#ifdef __cplusplus
}
#endif
#endif
