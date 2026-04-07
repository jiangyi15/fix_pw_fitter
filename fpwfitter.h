#ifndef FPWFITTER_H
#define FPWFITTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FpwFitter FpwFitter;

#define FPW_SUCCESS    0
#define FPW_ERR_ALLOC -1
#define FPW_ERR_CUDA  -2
#define FPW_ERR_CUBLAS -3

/**
 * Create a fitter.  All data is uploaded to GPU once at creation.
 * No further CPU↔GPU transfers occur during evaluation.
 *
 * Data layout (C-order / row-major):
 *   F_data : (n_data, n_proj, n_comp)   complex128
 *   w_data : (n_data,)                   float64
 *   B_data : (n_data,)                   float64
 *   M      : (n_comp, n_comp)            complex128  — pre-computed overlap matrix
 */
int fpw_create(
    int64_t      n_data,
    int          n_proj,
    int          n_comp,
    const double *F_data,
    const double *w_data,
    const double *B_data,
    const double *M,
    double       N_b,
    double       purity,
    FpwFitter  **out);

void fpw_destroy(FpwFitter *f);

/** Evaluate -log L and gradient d/d(c*) at coupling vector c. */
int fpw_evaluate(
    FpwFitter *f,
    const double *c_real,  const double *c_imag,
    double *nll,
    double *grad_real,     double *grad_imag,
    double *P_data          /* optional, shape (n_data,), NULL to skip */);

int    fpw_get_n_comp(const FpwFitter *f);
double fpw_get_N_s    (const FpwFitter *f);
double fpw_get_N_b    (const FpwFitter *f);
const char *fpw_strerror(int err);

#ifdef __cplusplus
}
#endif
#endif
