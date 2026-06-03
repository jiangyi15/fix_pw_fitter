"""
CFFI interface definitions matching the CUDA kernel API.
"""

from cffi import FFI

ffi = FFI()

ffi.cdef("""
    typedef struct { double x; double y; } cuDoubleComplex;

    void* pwa_create_config(
        const int* bw_index, const int* gamma_index, const int* bw_order,
        const int* bf_index, const int* bf_order, const int* ang_index,
        const double* ang_k, const double* ang_b,
        const double* matrix_gamma, const cuDoubleComplex* matrix_ang,
        const cuDoubleComplex* gamma_table, const double* bf_table,
        int n_waves, int n_m0, int n_g0,
        int n_res_per_wave, int n_decays_per_wave,
        int n_bf_types, int n_basis, int n_ang_per_basis,
        int n_gamma_points, int n_bf_points
    );
    void pwa_destroy_config(void* cfg);

    void* pwa_create_data(
        const double* mass_flat, const double* q_flat, const double* angles_flat,
        const double* time_arr, const double* frac_arr,
        int n_events, int mass_stride, int q_stride, int ang_stride
    );
    void pwa_destroy_data(void* data);

    void pwa_compute(
        void* cfg, void* data,
        double* p_out, cuDoubleComplex* amp_p_out, cuDoubleComplex* amp_m_out,
        double* q_val_out,
        double* grad_ck_re, double* grad_ck_im,
        double* grad_m0_out, double* grad_g0_out,
        double* grad_scalar_out,
        const cuDoubleComplex* ck, const double* m0, const double* g0,
        double delta_m, double delta_g, double g_val,
        double ap, double lam, double phi,
        double N_val, int do_likelihood,
        const double* weights, const double* bkg_arr,
        int n_waves, int n_m0, int n_g0,
        int n_res_per_wave, int n_decays_per_wave,
        int n_bf_types, int n_basis, int n_ang_per_basis,
        int n_gamma_points, int n_bf_points,
        double g_min, double g_delta, double q_min, double q_delta
    );
""")
