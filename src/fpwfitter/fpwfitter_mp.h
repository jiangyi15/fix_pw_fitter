#ifndef FPWFITTER_MP_H
#define FPWFITTER_MP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FpwFitterMP FpwFitterMP;

#define FPW_SUCCESS    0
#define FPW_ERR_ALLOC -1
#define FPW_ERR_CUDA  -2
#define FPW_ERR_CUBLAS -3

/**
 * Mixed-precision fitter.
 *
 * All internal computation in FP32 (float/float2).
 * F_data, M, w_data, B_data uploaded as FP32.
 * c, nll, gradient returned as FP64 (converted from FP32).
 *
 * This gives ~50× more FP32 throughput on consumer GPUs vs FP64.
 */
int fpw_mp_create(
    int64_t      n_data,
    int          n_proj,
    int          n_comp,
    const float  *F_data,     /* FP32, (n_data, n_proj, n_comp) */
    const float  *w_data,     /* FP32, (n_data,) */
    const float  *B_data,     /* FP32, (n_data,) */
    const float  *M,          /* FP32, (n_comp, n_comp) */
    float        N_b,
    float        purity,
    FpwFitterMP **out);

void fpw_mp_destroy(FpwFitterMP *f);

/** Evaluate -log L and gradient d/d(c*) at coupling vector c (FP64 I/O). */
int fpw_mp_evaluate(
    FpwFitterMP *f,
    const double *c_real,  const double *c_imag,
    double *nll,
    double *grad_real,     double *grad_imag,
    float  *P_data          /* optional, shape (n_data,), FP32 output */);

int    fpw_mp_get_n_comp(const FpwFitterMP *f);
float  fpw_mp_get_N_s    (const FpwFitterMP *f);
float  fpw_mp_get_N_b    (const FpwFitterMP *f);
const char *fpw_strerror(int err);

#ifdef __cplusplus
}
#endif
#endif
