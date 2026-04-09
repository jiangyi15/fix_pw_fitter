#ifndef FPWFITTER_PARALLEL_H
#define FPWFITTER_PARALLEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FpwFitterParallel FpwFitterParallel;

#define FPW_SUCCESS    0
#define FPW_ERR_ALLOC -1
#define FPW_ERR_CUDA  -2
#define FPW_ERR_CUBLAS -3

int fpw_par_create(int64_t n_data,
                   int     n_proj,     int     n_comp,
                   const double *F_data,
                   const double *w_data,
                   const double *B_data,
                   const double *M,
                   double   N_b,
                   double   purity,
                   int      n_streams,   /* 1-8 */
                   FpwFitterParallel **out);

void fpw_par_destroy(FpwFitterParallel *f);

int fpw_par_evaluate(FpwFitterParallel *f,
                     const double *c_real,  const double *c_imag,
                     double *nll,
                     double *grad_real,     double *grad_imag,
                     double *P_data  /* optional, (N,) */);

int    fpw_par_get_n_comp(const FpwFitterParallel *f);
double fpw_par_get_N_s   (const FpwFitterParallel *f);
double fpw_par_get_N_b   (const FpwFitterParallel *f);
const char *fpw_par_strerror(int err);

#ifdef __cplusplus
}
#endif
#endif
