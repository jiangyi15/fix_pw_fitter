/**
 * cuda_v4_pwa — projection-sum PWA kernel (derived from cuda_v3_ampcache).
 *
 * Physics model (no time evolution / no mixing / no scalar params):
 *
 *     P(e) = Σ_p  | A_p(e) |² ,   A_p(e) = Σ_k  ck_k · a_{p,k}(e)
 *
 * where p = 0..P-1 runs over the incoherent *projections* (helicity / spin
 * projections etc.) and k runs over the partial waves.  Every projection
 * shares the SAME wave coefficient ck_k — the projection only changes the
 * *angular part* of the per-wave spatial amplitude a_{p,k}:
 *
 *     a_{p,k}(e) = Amp_{p,k}(e) / bw_k(e)
 *       Amp_{p,k} = fa_{p,k} · fl_k        (angular × form-factor)
 *       bw_k      = ∏_r Breit–Wigner denominator (m0/g0, same for all p)
 *
 * Wave entries are stored p-major: entry index w = p·N + k,  N = n_wave/P.
 * All per-event/per-wave arrays (common_amp, bw_p, …) therefore have
 * n_wave = P·N entries per event, while the coupling array ck has length N
 * (ck[k] is shared by the P entries of base wave k).  matrix_angle / fl_order
 * / bw_order / mass_index / … in the kernel config must be duplicated the
 * same p-major way (angular-only difference per projection).
 *
 * The angular amplitude cache (amp_cache_fill_kernel, fp64 → float2 store,
 * constant over the fit) and the cached forward (amp_cache_amp_kernel) are
 * inherited unchanged from cuda_v3_ampcache — only the BW propagator is
 * recomputed per iteration, so m0/g0 keep flowing.
 *
 * Everything downstream is a projection-generic rewrite of the sparse v3
 * gradients:
 *   - dQ/dA_p        = dQ/dP · conj(A_p)              (per event, per proj)
 *   - dQ/dck_k       = Σ_p dQ/dA_p · common_{p,k}      (sum over projections)
 *   - per-entry BW/m0/g0 chain uses dQ_da := dQ/dA_{p(w)} for entry w.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/complex.h>
#include <cstdio>
#include <cmath>

using complex = thrust::complex<double>;

// ── Named constants ──
#define BLOCK_SIZE      256     // threads per block
#define DEFAULT_BATCH_SIZE 50000 // default events per GPU batch
#define MAX_THREADS     1024    // cap for the 1:1 entry kernels

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
        } \
    } while(0)

// ── Catmull-Rom interpolation helpers ──────────────────────────────────
__device__ double catmull_rom_1d(
    double pm1, double p0, double p1, double p2, double t
) {
    return p0 + 0.5 * t * (
        -pm1 + p1 + t * (2.0*pm1 - 5.0*p0 + 4.0*p1 - p2 + t * (-pm1 + 3.0*p0 - 3.0*p1 + p2))
    );
}

// Complex Catmull-Rom for gamma table (uses inv_delta)
__device__ complex interp_complex_device(
    const double* __restrict__ table_real,
    const double* __restrict__ table_imag,
    int type_idx, double x,
    double xmin, double inv_delta, int n_bins
) {
    double diff = (x - xmin) * inv_delta;
    int xbin = max(0, min((int)floor(diff), n_bins - 2));
    double t = max(0.0, min(diff - xbin, 1.0));
    int base = type_idx * n_bins + xbin;
    int end_ = (type_idx + 1) * n_bins - 1;
    int im1 = base > type_idx * n_bins ? base - 1 : base;
    int i2  = base + 2 <= end_ ? base + 2 : base + 1;
    double real_val = catmull_rom_1d(
        table_real[im1], table_real[base], table_real[base + 1], table_real[i2], t);
    double imag_val = catmull_rom_1d(
        table_imag[im1], table_imag[base], table_imag[base + 1], table_imag[i2], t);
    return complex(real_val, imag_val);
}

// Real Catmull-Rom for FL factor
__device__ double interp_real_device(
    const double* __restrict__ table,
    int type_idx, double x,
    double xmin, double xdelta, int n_bins
) {
    double diff = (x - xmin) / xdelta;
    int xbin = max(0, min((int)floor(diff), n_bins - 2));
    double t = max(0.0, min(diff - xbin, 1.0));
    int base = type_idx * n_bins + xbin;
    int end_ = (type_idx + 1) * n_bins - 1;
    int im1 = base > type_idx * n_bins ? base - 1 : base;
    int i2  = base + 2 <= end_ ? base + 2 : base + 1;
    return catmull_rom_1d(
        table[im1], table[base], table[base + 1], table[i2], t);
}

//=============================================================================
// KERNEL 1: g_bw computation (parallelized within each event)
// (verbatim from cuda_v3_ampcache — pure BW/gamma propagator, projection-free)
//=============================================================================
__global__ void compute_g_bw_kernel(
    const double* __restrict__ mass,
    const double* __restrict__ g0,
    const int* __restrict__ g0_index,
    const int* __restrict__ g0_mass_index,
    const int* __restrict__ gamma_col_idx,
    const double* __restrict__ gamma_table_real,
    const double* __restrict__ gamma_table_imag,
    double gamma_min, double gamma_inv_delta,
    int n_gamma_rows, int n_unique_bw, int n_mass, int gamma_table_bins,
    double* __restrict__ g_interp_real,
    double* __restrict__ g_interp_imag,
    double* __restrict__ g_bw_real,
    double* __restrict__ g_bw_imag,
    int n_events
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;

    extern __shared__ double s_dyn_gbw[];
    double* s_g_real = s_dyn_gbw;
    double* s_g_imag = s_dyn_gbw + n_gamma_rows;
    double* s_gbw_r  = s_dyn_gbw + 2 * n_gamma_rows;
    double* s_gbw_i  = s_dyn_gbw + 2 * n_gamma_rows + n_unique_bw;

    for (int i = tid; i < n_unique_bw; i += block_sz) {
        s_gbw_r[i] = 0.0;
        s_gbw_i[i] = 0.0;
    }
    __syncthreads();

    for (int gamma_idx = tid; gamma_idx < n_gamma_rows; gamma_idx += block_sz) {
        int g0_idx = g0_index[gamma_idx];
        double g0_val = g0[g0_idx];
        double mass_val = mass[event_idx * n_mass + g0_mass_index[gamma_idx]];

        complex g_interp = interp_complex_device(
            gamma_table_real, gamma_table_imag,
            g0_idx, mass_val,
            gamma_min, gamma_inv_delta, gamma_table_bins
        );

        g_interp_real[event_idx * n_gamma_rows + gamma_idx] = g_interp.real();
        g_interp_imag[event_idx * n_gamma_rows + gamma_idx] = g_interp.imag();

        complex g_val = g0_val * g_interp;
        s_g_real[gamma_idx] = g_val.real();
        s_g_imag[gamma_idx] = g_val.imag();
    }
    __syncthreads();

    for (int gamma_idx = tid; gamma_idx < n_gamma_rows; gamma_idx += block_sz) {
        int col = gamma_col_idx[gamma_idx];
        atomicAdd(&s_gbw_r[col], s_g_real[gamma_idx]);
        atomicAdd(&s_gbw_i[col], s_g_imag[gamma_idx]);
    }
    __syncthreads();

    for (int c = tid; c < n_unique_bw; c += block_sz) {
        g_bw_real[event_idx * n_unique_bw + c] = s_gbw_r[c];
        g_bw_imag[event_idx * n_unique_bw + c] = s_gbw_i[c];
    }
}

//=============================================================================
// KERNEL 2a: amp_cache_fill — one-time per-event angular-amplitude cache.
// Verbatim from cuda_v3_ampcache.  n_wave == P·N here (the entry count); the
// rep_of_slot representative entries pick the right matrix_angle column /
// fl_order row, so per-(projection, wave) angular factors are cached exactly.
//=============================================================================
__global__ void amp_cache_fill_kernel(
    const float* __restrict__ angle,
    const int* __restrict__ angle_index,
    const double* __restrict__ angle_k,
    const double* __restrict__ angle_b,
    const double* __restrict__ matrix_angle_real,
    const double* __restrict__ matrix_angle_imag,
    const float* __restrict__ momentum,
    const int* __restrict__ fl_type,
    const int* __restrict__ fl_q_index,
    const int* __restrict__ fl_order,
    const double* __restrict__ fl_table,
    double fl_min, double fl_delta,
    const int* __restrict__ rep_of_slot,
    int n_wave, int n_angle_k, int n_angle_total, int n_angle_comp,
    int n_decay, int n_momentum, int fl_table_bins,
    int n_uniq,
    float2* __restrict__ amp_cache,
    int n_events
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;

    extern __shared__ double s_dyn_amp[];
    double* s_ka = s_dyn_amp;                 // [n_angle_k]

    // Phase 1: angular basis products ka_k (fp64)
    for (int k_idx = tid; k_idx < n_angle_k; k_idx += block_sz) {
        int angle_pos = angle_index[k_idx];
        double ka = 1.0;
        for (int comp = 0; comp < n_angle_comp; comp++) {
            int idx = event_idx * n_angle_total * n_angle_comp
                      + angle_pos * n_angle_comp + comp;
            ka *= cos((double)angle[idx] * angle_k[k_idx * n_angle_comp + comp]
                      + angle_b[k_idx * n_angle_comp + comp]);
        }
        s_ka[k_idx] = ka;
    }
    __syncthreads();

    // Phase 2: per cache slot, evaluate Amp at the representative entry
    for (int s = tid; s < n_uniq; s += block_sz) {
        int w = rep_of_slot[s];
        double fr = 0.0, fi = 0.0;
        for (int k = 0; k < n_angle_k; k++) {
            int idx = k * n_wave + w;
            fr += s_ka[k] * matrix_angle_real[idx];
            fi += s_ka[k] * matrix_angle_imag[idx];
        }
        double fl = 1.0;
        for (int d = 0; d < n_decay; d++) {
            int fl_idx = fl_order[w * n_decay + d];
            double q = (double)momentum[event_idx * n_momentum + fl_q_index[fl_idx]];
            fl *= interp_real_device(fl_table, fl_type[fl_idx],
                                     q, fl_min, fl_delta, fl_table_bins);
        }
        amp_cache[event_idx * n_uniq + s] = make_float2((float)(fr * fl),
                                                        (float)(fi * fl));
    }
}

//=============================================================================
// KERNEL 2b: cached forward — BW propagator × cached angular amp.
// Per event; threads stride over the P·N wave entries.  Same as
// cuda_v3_ampcache's amp_cache_amp_kernel but entry-count generic.
//=============================================================================
__global__ void amp_cache_amp_kernel(
    const double* __restrict__ mass,
    const int* __restrict__ m0_index,
    const int* __restrict__ mass_index,
    const int* __restrict__ bw_order,
    const double* __restrict__ g_bw_real,
    const double* __restrict__ g_bw_imag,
    int n_wave, int n_res, int n_unique_bw,
    int n_mass,
    const double* __restrict__ m0,
    const float2* __restrict__ amp_cache,
    const int* __restrict__ slot_of_wave,
    int n_uniq,
    double* __restrict__ bw_p_real, double* __restrict__ bw_p_imag,
    double* __restrict__ common_amp_factor_real,
    double* __restrict__ common_amp_factor_imag,
    double* __restrict__ bw_dom_real, double* __restrict__ bw_dom_imag,
    int n_events
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;

    extern __shared__ double s_dyn_bw2[];
    double* s_mass     = s_dyn_bw2;
    double* s_gbw_real = s_dyn_bw2 + n_mass;
    double* s_gbw_imag = s_dyn_bw2 + n_mass + n_unique_bw;

    for (int i = tid; i < n_mass; i += blockDim.x)
        s_mass[i] = mass[event_idx * n_mass + i];
    for (int bw_idx = tid; bw_idx < n_unique_bw; bw_idx += blockDim.x) {
        int base = event_idx * n_unique_bw + bw_idx;
        s_gbw_real[bw_idx] = g_bw_real[base];
        s_gbw_imag[bw_idx] = g_bw_imag[base];
    }
    __syncthreads();

    for (int wave_idx = tid; wave_idx < n_wave; wave_idx += block_sz) {
        double bw_p_r = 1.0, bw_p_i = 0.0;
        for (int res_idx = 0; res_idx < n_res; res_idx++) {
            int bw_idx = bw_order[wave_idx * n_res + res_idx];
            double m0v = m0[m0_index[bw_idx]];
            double mv = s_mass[mass_index[bw_idx]];
            double m0s = m0v * m0v, ms = mv * mv;
            double dr = m0s - ms + m0v * s_gbw_imag[bw_idx];
            double di = -m0v * s_gbw_real[bw_idx];
            double nr = bw_p_r * dr - bw_p_i * di;
            double ni = bw_p_r * di + bw_p_i * dr;
            bw_p_r = nr; bw_p_i = ni;
            bw_dom_real[event_idx * n_unique_bw + bw_idx] = dr;
            bw_dom_imag[event_idx * n_unique_bw + bw_idx] = di;
        }
        bw_p_real[event_idx * n_wave + wave_idx] = bw_p_r;
        bw_p_imag[event_idx * n_wave + wave_idx] = bw_p_i;

        float2 A = amp_cache[event_idx * n_uniq + slot_of_wave[wave_idx]];
        double nrm = bw_p_r * bw_p_r + bw_p_i * bw_p_i;
        common_amp_factor_real[event_idx * n_wave + wave_idx] =
            ((double)A.x * bw_p_r + (double)A.y * bw_p_i) / nrm;
        common_amp_factor_imag[event_idx * n_wave + wave_idx] =
            ((double)A.y * bw_p_r - (double)A.x * bw_p_i) / nrm;
    }
}

//=============================================================================
// KERNEL 3: PWA forward — A_p, P = Σ_p|A_p|², Q and dQ/dA_p.
// One block per event.  A_p(e) = Σ_k ck_k·common_{p·N+k}; the shared ck[k]
// (length N) is read for all P projections (angular-only difference lives in
// the common_amp entries, which are stored p-major).
// Outputs (per event):
//   Q_out[e], P_out[e]
//   dQ_dA_r/i[e·P + p] = dQ/dP · conj(A_p)     (Wirtinger ∂Q/∂A_p)
//=============================================================================
__global__ void pwa_amp_reduce_kernel(
    const double* __restrict__ common_r,
    const double* __restrict__ common_i,
    const double* __restrict__ ck_real,
    const double* __restrict__ ck_imag,
    const double* __restrict__ weight,
    const double* __restrict__ bkg,
    int n_wave,               // P·N — entries per event
    int n_proj,               // P
    int n_events,
    int use_norm, double norm,
    double* __restrict__ Q_out,
    double* __restrict__ P_out,
    double* __restrict__ dQ_dA_r, double* __restrict__ dQ_dA_i
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;
    int n_base = n_wave / n_proj;      // N

    extern __shared__ double s_dyn[];
    double* s_red_r = s_dyn;
    double* s_red_i = s_dyn + block_sz;
    double* sA_r = s_dyn + 2 * block_sz;
    double* sA_i = s_dyn + 2 * block_sz + n_proj;

    const double* cr = common_r + (size_t)event_idx * n_wave;
    const double* ci = common_i + (size_t)event_idx * n_wave;

    for (int p = 0; p < n_proj; p++) {
        double ar = 0.0, ai = 0.0;
        // A_p = Σ_k ck[k] · common[p·N + k]
        for (int k = tid; k < n_base; k += block_sz) {
            int w = p * n_base + k;
            double ckr = ck_real[k], cki = ck_imag[k];
            double cmr = cr[w], cmi = ci[w];
            ar += ckr * cmr - cki * cmi;
            ai += ckr * cmi + cki * cmr;
        }
        s_red_r[tid] = ar;
        s_red_i[tid] = ai;
        __syncthreads();
        for (int s = block_sz / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_red_r[tid] += s_red_r[tid + s];
                s_red_i[tid] += s_red_i[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            sA_r[p] = s_red_r[0];
            sA_i[p] = s_red_i[0];
        }
        __syncthreads();    // protect s_red reuse + sA read after the loop
    }

    if (tid == 0) {
        double P = 0.0;
        for (int p = 0; p < n_proj; p++) {
            double ar = sA_r[p], ai = sA_i[p];
            P += ar * ar + ai * ai;
        }
        P_out[event_idx] = P;
        double wval = weight[event_idx];
        double bkg_val = bkg[event_idx];
        double dQ_dP;
        if (use_norm == 0) {
            Q_out[event_idx] = wval * P;
            dQ_dP = wval;
        } else {
            Q_out[event_idx] = -wval * log(P / norm + bkg_val);
            dQ_dP = -wval / (P + bkg_val * norm);
        }
        // dQ/dA_p = dQ/dP · conj(A_p)
        size_t ab = (size_t)event_idx * n_proj;
        for (int p = 0; p < n_proj; p++) {
            dQ_dA_r[ab + p] = dQ_dP * sA_r[p];
            dQ_dA_i[ab + p] = -dQ_dP * sA_i[p];
        }
    }
}

//=============================================================================
// KERNEL 4: ck gradient + per-entry ∂Q/∂common (for the BW chain).
// One block per event.  ck_k is shared by all P projections:
//   grad_ck_partial[e·N + k] = Σ_p  dQ_dA[e,p] · common[e, p·N + k]
//   (the sum over p is accumulated in shared memory inside the block, so the
//    partial written here is already projection-reduced → length N).
//=============================================================================
__global__ void grad_ck_kernel_v4(
    const double* __restrict__ common_r,
    const double* __restrict__ common_i,
    const double* __restrict__ dQ_dA_r,
    const double* __restrict__ dQ_dA_i,
    double* __restrict__ grad_ck_real_partial,
    double* __restrict__ grad_ck_imag_partial,
    int n_wave, int n_proj, int n_events
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;
    int n_base = n_wave / n_proj;      // N

    extern __shared__ double s_dyn[];
    double* s_gk_r = s_dyn;
    double* s_gk_i = s_dyn + n_base;

    for (int k = tid; k < n_base; k += block_sz) {
        s_gk_r[k] = 0.0;
        s_gk_i[k] = 0.0;
    }
    __syncthreads();

    const double* cr = common_r + (size_t)event_idx * n_wave;
    const double* ci = common_i + (size_t)event_idx * n_wave;
    const double* dar = dQ_dA_r + (size_t)event_idx * n_proj;
    const double* dai = dQ_dA_i + (size_t)event_idx * n_proj;

    for (int w = tid; w < n_wave; w += block_sz) {
        int p = w / n_base;
        int k = w - p * n_base;
        double a_r = dar[p], a_i = dai[p];
        double cmr = cr[w], cmi = ci[w];
        // dQ/dck_k += dQ_dA_p · common_w   (complex multiply)
        double gr = a_r * cmr - a_i * cmi;
        double gi = a_r * cmi + a_i * cmr;
        atomicAdd(&s_gk_r[k], gr);
        atomicAdd(&s_gk_i[k], gi);
    }
    __syncthreads();

    for (int k = tid; k < n_base; k += block_sz) {
        grad_ck_real_partial[(size_t)event_idx * n_base + k] = s_gk_r[k];
        grad_ck_imag_partial[(size_t)event_idx * n_base + k] = s_gk_i[k];
    }
}

//=============================================================================
// KERNEL 5: bw_dom + m0 gradient (per wave entry).
// For entry w (projection p = w/N) the effective ∂Q/∂a_w is dQ/dA_p — read
// per event from the dQ_dA buffer.  Then the unchanged sparse chain applies:
//   dQ/dbw_p = dQ_da · (-ck_k · common_w / bw_p_w² …), dQ_da = dQ/dA_p.
// Writes dQ_dbw_dom (for grad_g0) and grad_m0 partials.
//=============================================================================
__global__ void grad_bw_dom_kernel_v4(
    const double* __restrict__ bw_p_real, const double* __restrict__ bw_p_imag,
    const double* __restrict__ common_amp_factor_real,
    const double* __restrict__ common_amp_factor_imag,
    const double* __restrict__ dQ_dA_r, const double* __restrict__ dQ_dA_i,
    const double* __restrict__ bw_dom_real, const double* __restrict__ bw_dom_imag,
    const double* __restrict__ g_bw_real, const double* __restrict__ g_bw_imag,
    const int* __restrict__ m0_index,
    const int* __restrict__ bw_order, const double* __restrict__ m0,
    const double* __restrict__ ck_real, const double* __restrict__ ck_imag,
    int n_wave, int n_res, int n_unique_bw, int n_proj,
    double* __restrict__ grad_m0_partial,
    double* __restrict__ dQ_dbw_dom_real,
    double* __restrict__ dQ_dbw_dom_imag,
    int n_events
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;
    int n_base = n_wave / n_proj;

    for (int bw_idx = tid; bw_idx < n_unique_bw; bw_idx += block_sz) {
        grad_m0_partial[event_idx * n_unique_bw + bw_idx] = 0.0;
        dQ_dbw_dom_real[event_idx * n_unique_bw + bw_idx] = 0.0;
        dQ_dbw_dom_imag[event_idx * n_unique_bw + bw_idx] = 0.0;
    }
    __syncthreads();

    const double* dar = dQ_dA_r + (size_t)event_idx * n_proj;
    const double* dai = dQ_dA_i + (size_t)event_idx * n_proj;

    for (int wave_idx = tid; wave_idx < n_wave; wave_idx += block_sz) {
        int p = wave_idx / n_base;
        int k = wave_idx - p * n_base;
        double bpr = bw_p_real[event_idx * n_wave + wave_idx];
        double bpi = bw_p_imag[event_idx * n_wave + wave_idx];
        double car = common_amp_factor_real[event_idx * n_wave + wave_idx];
        double cai = common_amp_factor_imag[event_idx * n_wave + wave_idx];

        double dqa_r = dar[p];
        double dqa_i = dai[p];

        double ckr = ck_real[k], cki = ck_imag[k];

        double bpn = bpr * bpr + bpi * bpi;
        double obw_r = bpr / bpn;
        double obw_i = -bpi / bpn;

        double ck_obw_r = ckr * obw_r - cki * obw_i;
        double ck_obw_i = ckr * obw_i + cki * obw_r;

        double neg_mul_r = -(ck_obw_r * car - ck_obw_i * cai);
        double neg_mul_i = -(ck_obw_r * cai + ck_obw_i * car);

        double ddbr = dqa_r * neg_mul_r - dqa_i * neg_mul_i;
        double ddbi = dqa_r * neg_mul_i + dqa_i * neg_mul_r;

        for (int res_idx = 0; res_idx < n_res; res_idx++) {
            int bw_idx = bw_order[wave_idx * n_res + res_idx];
            double bdr = bw_dom_real[event_idx * n_unique_bw + bw_idx];
            double bdi = bw_dom_imag[event_idx * n_unique_bw + bw_idx];

            double bdn = bdr * bdr + bdi * bdi;
            double inv_bd_r = bdr / bdn;
            double inv_bd_i = -bdi / bdn;
            double bwp_div_bd_r = bpr * inv_bd_r - bpi * inv_bd_i;
            double bwp_div_bd_i = bpr * inv_bd_i + bpi * inv_bd_r;

            double cr = ddbr * bwp_div_bd_r - ddbi * bwp_div_bd_i;
            double ci = ddbr * bwp_div_bd_i + ddbi * bwp_div_bd_r;

            atomicAdd(&dQ_dbw_dom_real[event_idx * n_unique_bw + bw_idx], cr);
            atomicAdd(&dQ_dbw_dom_imag[event_idx * n_unique_bw + bw_idx], ci);

            double m0_val = m0[m0_index[bw_idx]];
            double gbr = g_bw_real[event_idx * n_unique_bw + bw_idx];
            double gbi = g_bw_imag[event_idx * n_unique_bw + bw_idx];
            double dm0_re = 2.0 * m0_val + gbi;
            double dm0_im = -gbr;
            double wg = 2.0 * (cr * dm0_re - ci * dm0_im);
            atomicAdd(&grad_m0_partial[event_idx * n_unique_bw + bw_idx], wg);
        }
    }
}

//=============================================================================
// KERNEL 6: g0 gradient — FP32 sparse gather (verbatim from v3 sparse chain).
//=============================================================================
__global__ void grad_g0_kernel(
    double* __restrict__ dQ_dbw_dom_real,
    double* __restrict__ dQ_dbw_dom_imag,
    const double* __restrict__ g_interp_real,
    const double* __restrict__ g_interp_imag,
    const double* __restrict__ m0,
    const int* __restrict__ m0_index,
    const int* __restrict__ gamma_col_idx,
    int n_unique_bw, int n_gamma_rows,
    double* __restrict__ grad_g0_partial,
    int n_events
) {
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;

    for (int bw_idx = tid; bw_idx < n_unique_bw; bw_idx += block_sz) {
        double m0_val = m0[m0_index[bw_idx]];
        double dr = dQ_dbw_dom_real[event_idx * n_unique_bw + bw_idx];
        double di = dQ_dbw_dom_imag[event_idx * n_unique_bw + bw_idx];
        dQ_dbw_dom_real[event_idx * n_unique_bw + bw_idx] = m0_val * di;
        dQ_dbw_dom_imag[event_idx * n_unique_bw + bw_idx] = -m0_val * dr;
    }
    __syncthreads();

    for (int gamma_idx = tid; gamma_idx < n_gamma_rows; gamma_idx += block_sz) {
        int col = gamma_col_idx[gamma_idx];
        double sum_r = dQ_dbw_dom_real[event_idx * n_unique_bw + col];
        double sum_i = dQ_dbw_dom_imag[event_idx * n_unique_bw + col];
        double gr = g_interp_real[event_idx * n_gamma_rows + gamma_idx];
        double gi = g_interp_imag[event_idx * n_gamma_rows + gamma_idx];
        grad_g0_partial[event_idx * n_gamma_rows + gamma_idx] =
            2.0 * (sum_r * gr - sum_i * gi);
    }
}

//=============================================================================
// Reduction kernels (unchanged from the shared implementation)
//=============================================================================
__global__ void reduce_sum_kernel(const double* input, double* output, int n) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < n) ? input[idx] : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, sdata[0]);
}

__global__ void reduce_sum_complex_kernel(
    const double* real_in, const double* imag_in,
    double* real_out, double* imag_out, int n
) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double* sreal = sdata;
    double* simag = sdata + blockDim.x;
    sreal[tid] = (idx < n) ? real_in[idx] : 0.0;
    simag[tid] = (idx < n) ? imag_in[idx] : 0.0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { sreal[tid] += sreal[tid + s]; simag[tid] += simag[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { atomicAdd(real_out, sreal[0]); atomicAdd(imag_out, simag[0]); }
}

__global__ void reduce_sum_features_kernel(
    const double* input, double* output,
    int n_events, int n_features
) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int feat = blockIdx.x;
    if (feat >= n_features) return;
    double sum = 0.0;
    for (int i = tid; i < n_events; i += blockDim.x) {
        sum += input[feat + i * n_features];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[feat] = sdata[0];
}

__global__ void reduce_sum_complex_features_kernel(
    const double* real_in, const double* imag_in,
    double* real_out, double* imag_out,
    int n_events, int n_features
) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int feat = blockIdx.x;
    if (feat >= n_features) return;
    double real_sum = 0.0, imag_sum = 0.0;
    for (int i = tid; i < n_events; i += blockDim.x) {
        real_sum += real_in[feat + i * n_features];
        imag_sum += imag_in[feat + i * n_features];
    }
    double* sreal = sdata;
    double* simag = sdata + blockDim.x;
    sreal[tid] = real_sum;
    simag[tid] = imag_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { sreal[tid] += sreal[tid + s]; simag[tid] += simag[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { real_out[feat] = sreal[0]; imag_out[feat] = simag[0]; }
}

// ── Upload helpers (plain C, before extern "C") ──
void* _up_int(const int* src, int n) {
    int* d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, src, n * sizeof(int), cudaMemcpyHostToDevice)); return d;
}
void* _up_dbl(const double* src, int n) {
    double* d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d, src, n * sizeof(double), cudaMemcpyHostToDevice)); return d;
}
float* _up_f32(const double* src, int n) {
    float* d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    float* buf = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) buf[i] = (float)src[i];
    CUDA_CHECK(cudaMemcpy(d, buf, n * sizeof(float), cudaMemcpyHostToDevice));
    free(buf); return d;
}

//=============================================================================
// Host-callable launch functions
//=============================================================================
extern "C" {

typedef struct {
    // Event data (GPU): mass only — momentum/angle consumed by the one-time
    // cache fill at load_data, not kept on the device.
    const double* mass;
    const double* weight; const double* bkg;
    // Scratch buffers (GPU)
    double* g_interp_real; double* g_interp_imag;
    double* g_bw_real; double* g_bw_imag;
    double* Q_out; double* P_out;
    double* bw_p_real; double* bw_p_imag;
    double* common_amp_factor_real; double* common_amp_factor_imag;
    double* dQ_dA_real; double* dQ_dA_imag;        // [bs · n_proj]
    double* grad_ck_real_partial; double* grad_ck_imag_partial;  // [bs · N]
    double* grad_m0_partial; double* grad_g0_partial;
    double* dQ_dbw_dom_real; double* dQ_dbw_dom_imag;  // [bs · n_unique_bw]
    double* bw_dom_real; double* bw_dom_imag;
    // Angular-amplitude cache (per handle): per-event float2[n_uniq].
    float2* amp_cache;
    int n_events;
} ComputeData;

typedef struct {
    // Index arrays (GPU)
    const int* m0_index; const int* g0_index;
    const int* g0_mass_index; const int* mass_index;
    const int* fl_type; const int* fl_q_index;
    const int* bw_order; const int* fl_order; const int* angle_index;
    // Constant arrays (GPU)
    const double* angle_k; const double* angle_b;
    const double* matrix_angle_real; const double* matrix_angle_imag;
    const double* gamma_table_real; const double* gamma_table_imag;
    double gamma_min; double gamma_delta; double gamma_inv_delta; int gamma_table_bins;
    const double* matrix_gamma;
    const int* gamma_col_idx;
    const double* fl_table; double fl_min; double fl_delta; double fl_inv_delta; int fl_table_bins;
    // Dimensions
    int n_wave; int n_res; int n_decay; int n_unique_bw;
    int n_gamma_rows; int n_mass; int n_momentum;
    int n_angle_k; int n_angle_total; int n_angle_comp;
    int batch_size;
    int n_m0_params; int n_g0_params;
    // cuda_v4_pwa: angular-cache layout + projection count
    const int* slot_of_wave;   // [n_wave]  entry → cache slot
    const int* rep_of_slot;    // [n_uniq]  slot → representative entry
    int n_uniq;
    int n_proj;                // P — number of incoherent projections
    ComputeData* scratch;
    double* Q_red_gpu;
} ComputeContext;

typedef struct {
    const double* ck_real; const double* ck_imag;
    const double* m0; const double* g0;
} ComputeParams;

cudaError_t cuda_alloc(void** ptr, size_t size) { return cudaMalloc(ptr, size); }
cudaError_t cuda_free(void* ptr) { return cudaFree(ptr); }
cudaError_t cuda_memcpy_to_device(void* dst, const void* src, size_t size) {
    return cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}
cudaError_t cuda_memcpy_to_host(void* dst, const void* src, size_t size) {
    return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}
cudaError_t cuda_memset(void* ptr, int value, size_t size) { return cudaMemset(ptr, value, size); }
int cuda_get_device_count() { int count; cudaGetDeviceCount(&count); return count; }
cudaError_t cuda_get_device_name(char* name, int len) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err == cudaSuccess) strncpy(name, prop.name, len);
    return err;
}

//── g_bw launch wrapper ────────────────────────────────────────────
void launch_compute_g_bw(
    const double* mass, const double* g0,
    const int* g0_index, const int* g0_mass_index,
    const int* gamma_col_idx,
    const double* gamma_table_real, const double* gamma_table_imag,
    double gamma_min, double gamma_delta,
    int n_gamma_rows, int n_unique_bw, int n_mass, int gamma_table_bins,
    double* g_interp_real, double* g_interp_imag,
    double* g_bw_real, double* g_bw_imag,
    int n_events) {

    size_t shmem = (2 * n_gamma_rows + 2 * n_unique_bw) * sizeof(double);
    int gt = (n_gamma_rows < 256) ? 256 : (n_gamma_rows > 1024 ? 1024 : n_gamma_rows);
    double gamma_inv_delta = 1.0 / gamma_delta;
    compute_g_bw_kernel<<<n_events, gt, shmem>>>(
        mass, g0, g0_index, g0_mass_index, gamma_col_idx,
        gamma_table_real, gamma_table_imag,
        gamma_min, gamma_inv_delta,
        n_gamma_rows, n_unique_bw, n_mass, gamma_table_bins,
        g_interp_real, g_interp_imag,
        g_bw_real, g_bw_imag, n_events);
}

// Reduction launch wrappers
void launch_reduce_sum(const double* input, double* output, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_sum_kernel<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(input, output, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_reduce_sum_complex(const double* real_in, const double* imag_in,
    double* real_out, double* imag_out, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_sum_complex_kernel<<<grid_size, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(double)>>>(
        real_in, imag_in, real_out, imag_out, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_reduce_sum_features(const double* input, double* output,
    int n_events, int n_features) {
    reduce_sum_features_kernel<<<n_features, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(
        input, output, n_events, n_features);
}

void launch_reduce_sum_complex_features(const double* real_in, const double* imag_in,
    double* real_out, double* imag_out,
    int n_events, int n_features) {
    reduce_sum_complex_features_kernel<<<n_features, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(double)>>>(
        real_in, imag_in, real_out, imag_out, n_events, n_features);
    CUDA_CHECK(cudaGetLastError());
}

//=============================================================================
// Unified per-batch launch
//=============================================================================
void launch_compute_all(
    const ComputeContext* ctx, ComputeData* data,
    const ComputeParams* params, double norm, int use_norm
) {
    int nw = ctx->n_wave, nu = ctx->n_unique_bw, ng = ctx->n_gamma_rows;
    int P = ctx->n_proj;
    int N = nw / P;
    int ne = data->n_events;
    int amp_t = nw < MAX_THREADS ? nw : MAX_THREADS;
    if (amp_t < 32) amp_t = 32;

    //── K1: g_bw ──────────────────────────────────────────────────────
    launch_compute_g_bw(
        data->mass, params->g0, ctx->g0_index, ctx->g0_mass_index,
        ctx->gamma_col_idx,
        ctx->gamma_table_real, ctx->gamma_table_imag,
        ctx->gamma_min, ctx->gamma_delta,
        ctx->n_gamma_rows, ctx->n_unique_bw, ctx->n_mass, ctx->gamma_table_bins,
        data->g_interp_real, data->g_interp_imag,
        data->g_bw_real, data->g_bw_imag, ne);
    CUDA_CHECK(cudaGetLastError());

    //── K2: cached forward — BW recomputed per iteration, angular from cache
    {
        size_t shmem = (ctx->n_mass + 2 * nu) * sizeof(double);
        amp_cache_amp_kernel<<<ne, amp_t, shmem>>>(
            data->mass,
            ctx->m0_index, ctx->mass_index, ctx->bw_order,
            data->g_bw_real, data->g_bw_imag,
            nw, ctx->n_res, nu, ctx->n_mass, params->m0,
            data->amp_cache, ctx->slot_of_wave, ctx->n_uniq,
            data->bw_p_real, data->bw_p_imag,
            data->common_amp_factor_real, data->common_amp_factor_imag,
            data->bw_dom_real, data->bw_dom_imag, ne);
    }
    CUDA_CHECK(cudaGetLastError());

    //── K3: PWA forward (P = Σ_p |A_p|², dQ/dA_p) ────────────────────
    {
        size_t shmem = (2 * BLOCK_SIZE + 2 * P) * sizeof(double);
        pwa_amp_reduce_kernel<<<ne, BLOCK_SIZE, shmem>>>(
            data->common_amp_factor_real, data->common_amp_factor_imag,
            params->ck_real, params->ck_imag,
            data->weight, data->bkg,
            nw, P, ne, use_norm, norm,
            data->Q_out, data->P_out,
            data->dQ_dA_real, data->dQ_dA_imag);
    }
    CUDA_CHECK(cudaGetLastError());

    //── K4: ck gradient (projection-reduced partials, length N) ───────
    {
        size_t shmem = 2 * N * sizeof(double);
        grad_ck_kernel_v4<<<ne, BLOCK_SIZE, shmem>>>(
            data->common_amp_factor_real, data->common_amp_factor_imag,
            data->dQ_dA_real, data->dQ_dA_imag,
            data->grad_ck_real_partial, data->grad_ck_imag_partial,
            nw, P, ne);
    }
    CUDA_CHECK(cudaGetLastError());

    //── K5: bw_dom / m0 gradients ────────────────────────────────────
    grad_bw_dom_kernel_v4<<<ne, amp_t>>>(
        data->bw_p_real, data->bw_p_imag,
        data->common_amp_factor_real, data->common_amp_factor_imag,
        data->dQ_dA_real, data->dQ_dA_imag,
        data->bw_dom_real, data->bw_dom_imag,
        data->g_bw_real, data->g_bw_imag,
        ctx->m0_index, ctx->bw_order, params->m0,
        params->ck_real, params->ck_imag,
        nw, ctx->n_res, nu, P,
        data->grad_m0_partial,
        data->dQ_dbw_dom_real, data->dQ_dbw_dom_imag, ne);
    CUDA_CHECK(cudaGetLastError());

    //── K6: g0 gradient (sparse gather) ───────────────────────────────
    grad_g0_kernel<<<ne, BLOCK_SIZE>>>(
        data->dQ_dbw_dom_real, data->dQ_dbw_dom_imag,
        data->g_interp_real, data->g_interp_imag,
        params->m0, ctx->m0_index, ctx->gamma_col_idx,
        nu, ng, data->grad_g0_partial, ne);
    CUDA_CHECK(cudaGetLastError());
}

typedef struct {
    // Event data kept on device: mass (recomputed each iteration) + the
    // scalar weight/bkg arrays.  momentum/angle are NOT kept — transient.
    const double* m;
    const double* w; const double* b;
    int ne; int nm; int nmom; int nat; int nac;
    float2* amp_cache;    // [ne · n_uniq], built at load_data
    int n_uniq;
} DataHandle2;

void cuda_free_context_v4(void* vctx);  // forward decl (used by create on error)

void* cuda_create_context_v4(
    const int* m0_i,int n1, const int* g0_i,int n2,
    const int* g0_m,int n3, const int* mass_i,int n4,
    const int* fl_t,int n5, const int* fl_q,int n6,
    const int* bw_o,int n7, const int* fl_o,int n8,
    const int* ang_i,int n9,
    const double* ak,int n10, const double* ab,int n11,
    const double* mar,int n12, const double* mai,int n13,
    const double* gtr,int n14, const double* gti,int n15,
    double gmin,double gdel,int gbins,
    const double* mg,int n16,
    const int* gci,int ngci,
    const double* ft,int n17, double flmin,double fldel,int fbins,
    int nw,int nr,int nd,int nub,int ngr,
    int nm,int nmom,int nak_,int nat,int nac,
    int n_m0p, int n_g0p,
    int batch_size,
    const int* slot_of_wave,int n_slot,
    const int* rep_of_slot,int n_rep,
    int n_uniq, int n_proj
) {
    ComputeContext* c = (ComputeContext*)calloc(1, sizeof(ComputeContext));
    c->m0_index = (int*)_up_int(m0_i, n1); c->g0_index = (int*)_up_int(g0_i, n2);
    c->g0_mass_index = (int*)_up_int(g0_m, n3); c->mass_index = (int*)_up_int(mass_i, n4);
    c->fl_type = (int*)_up_int(fl_t, n5); c->fl_q_index = (int*)_up_int(fl_q, n6);
    c->bw_order = (int*)_up_int(bw_o, n7); c->fl_order = (int*)_up_int(fl_o, n8);
    c->angle_index = (int*)_up_int(ang_i, n9);
    c->angle_k = (double*)_up_dbl(ak, n10); c->angle_b = (double*)_up_dbl(ab, n11);
    c->matrix_angle_real = (double*)_up_dbl(mar, n12); c->matrix_angle_imag = (double*)_up_dbl(mai, n13);
    c->gamma_table_real = (double*)_up_dbl(gtr, n14); c->gamma_table_imag = (double*)_up_dbl(gti, n15);
    c->gamma_min = gmin; c->gamma_delta = gdel; c->gamma_table_bins = gbins;
    c->matrix_gamma = (double*)_up_dbl(mg, n16);
    c->gamma_col_idx = (int*)_up_int(gci, ngci);
    c->fl_table = (double*)_up_dbl(ft, n17); c->fl_min = flmin; c->fl_delta = fldel; c->fl_table_bins = fbins;
    c->n_wave = nw; c->n_res = nr; c->n_decay = nd;
    c->n_unique_bw = nub; c->n_gamma_rows = ngr;
    c->n_mass = nm; c->n_momentum = nmom; c->n_angle_k = nak_; c->n_angle_total = nat; c->n_angle_comp = nac;
    c->n_m0_params = n_m0p; c->n_g0_params = n_g0p;
    c->batch_size = batch_size > 0 ? batch_size : DEFAULT_BATCH_SIZE;
    c->slot_of_wave = (n_slot > 0) ? (int*)_up_int(slot_of_wave, n_slot) : NULL;
    c->rep_of_slot = (n_rep > 0) ? (int*)_up_int(rep_of_slot, n_rep) : NULL;
    c->n_uniq = (n_uniq > 0) ? n_uniq : 0;
    c->n_proj = n_proj > 0 ? n_proj : 1;
    if (c->n_wave % c->n_proj != 0) {
        fprintf(stderr, "cuda_v4_pwa: n_wave %d must be divisible by n_proj %d\n",
                c->n_wave, c->n_proj);
        cuda_free_context_v4(c);
        return NULL;
    }

    if (c->batch_size > 0) {
        int bs = c->batch_size;
        int N = c->n_wave / c->n_proj;
        c->scratch = (ComputeData*)calloc(1, sizeof(ComputeData));
        #define S(f) CUDA_CHECK(cudaMalloc(&c->scratch->f, bs * sizeof(double)))
        #define S2(f,n) CUDA_CHECK(cudaMalloc(&c->scratch->f, bs * (n) * sizeof(double)))
        S2(g_interp_real, ngr); S2(g_interp_imag, ngr);
        S2(g_bw_real, nub); S2(g_bw_imag, nub);
        S(Q_out); S(P_out);
        S2(bw_p_real, nw); S2(bw_p_imag, nw);
        S2(common_amp_factor_real, nw); S2(common_amp_factor_imag, nw);
        S2(bw_dom_real, nub); S2(bw_dom_imag, nub);
        S2(dQ_dA_real, c->n_proj); S2(dQ_dA_imag, c->n_proj);
        S2(grad_ck_real_partial, N); S2(grad_ck_imag_partial, N);
        S2(grad_m0_partial, nub); S2(grad_g0_partial, ngr);
        S2(dQ_dbw_dom_real, nub); S2(dQ_dbw_dom_imag, nub);
        #undef S
        #undef S2
        CUDA_CHECK(cudaMalloc(&c->Q_red_gpu, 8));
    } else {
        c->scratch = NULL;
        c->Q_red_gpu = NULL;
    }
    return c;
}

void cuda_free_context_v4(void* vctx) {
    ComputeContext* c = (ComputeContext*)vctx;
    if (!c) return;
    #define F(p) cudaFree((void*)c->p)
    F(m0_index); F(g0_index); F(g0_mass_index); F(mass_index);
    F(fl_type); F(fl_q_index); F(bw_order); F(fl_order); F(angle_index);
    F(angle_k); F(angle_b); F(matrix_angle_real); F(matrix_angle_imag);
    F(gamma_table_real); F(gamma_table_imag); F(matrix_gamma); F(gamma_col_idx); F(fl_table);
    if (c->slot_of_wave) cudaFree((void*)c->slot_of_wave);
    if (c->rep_of_slot) cudaFree((void*)c->rep_of_slot);
    #undef F
    if (c->scratch) {
        #define SF(f) cudaFree(c->scratch->f)
        SF(g_interp_real); SF(g_interp_imag); SF(g_bw_real); SF(g_bw_imag);
        SF(Q_out); SF(P_out);
        SF(bw_p_real); SF(bw_p_imag);
        SF(common_amp_factor_real); SF(common_amp_factor_imag);
        SF(bw_dom_real); SF(bw_dom_imag);
        SF(dQ_dA_real); SF(dQ_dA_imag);
        SF(grad_ck_real_partial); SF(grad_ck_imag_partial);
        SF(grad_m0_partial); SF(grad_g0_partial);
        SF(dQ_dbw_dom_real); SF(dQ_dbw_dom_imag);
        #undef SF
        free(c->scratch);
    }
    if (c->Q_red_gpu) cudaFree(c->Q_red_gpu);
    free(c);
}

// Forward decls: the angular cache is allocated once per handle and filled
// in event-chunks from TRANSIENT momentum/angle device buffers.
static int alloc_handle_amp_cache(ComputeContext* c, DataHandle2* h);
static int fill_amp_cache_chunk(ComputeContext* c, DataHandle2* h,
                                const float* mom, const float* ang,
                                int n_events, int base_event);

void* cuda_load_data_v4(void* vctx,
    const double* mass,int nmass, const double* mom,int nmom,
    const double* ang,int nang,
    const double* wgt,const double* bkg,int ne
) {
    DataHandle2* h = (DataHandle2*)calloc(1, sizeof(DataHandle2));
    ComputeContext* c_ctx = (ComputeContext*)vctx;
    int nac = c_ctx ? c_ctx->n_angle_comp : 3;
    h->m = (const double*)_up_dbl(mass, ne * nmass);
    h->w = (const double*)_up_dbl(wgt, ne);
    h->b = (const double*)_up_dbl(bkg, ne);
    h->ne = ne; h->nm = nmass; h->nmom = nmom; h->nat = nang; h->nac = nac;

    if (c_ctx) {
        if (!alloc_handle_amp_cache(c_ctx, h)) goto fail;
        int fb = c_ctx->batch_size > 0 ? c_ctx->batch_size : ne;
        if (fb > ne) fb = ne;
        float* gpu_mom = NULL; float* gpu_ang = NULL;
        float* tmp_mom = NULL; float* tmp_ang = NULL;
        if (cudaMalloc(&gpu_mom, (size_t)fb * nmom * sizeof(float))
                != cudaSuccess) goto fail;
        if (cudaMalloc(&gpu_ang, (size_t)fb * nang * nac * sizeof(float))
                != cudaSuccess) { cudaFree(gpu_mom); goto fail; }
        tmp_mom = (float*)malloc((size_t)fb * nmom * sizeof(float));
        tmp_ang = (float*)malloc((size_t)fb * nang * nac * sizeof(float));
        int failed = 0;
        for (int base = 0; base < ne && !failed; base += fb) {
            int nb = (ne - base > fb) ? fb : (ne - base);
            for (int i = 0; i < nb * nmom; i++)
                tmp_mom[i] = (float)mom[(size_t)base * nmom + i];
            for (int i = 0; i < nb * nang * nac; i++)
                tmp_ang[i] = (float)ang[(size_t)base * nang * nac + i];
            cudaMemcpy(gpu_mom, tmp_mom, (size_t)nb * nmom * sizeof(float),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_ang, tmp_ang, (size_t)nb * nang * nac * sizeof(float),
                       cudaMemcpyHostToDevice);
            failed = !fill_amp_cache_chunk(c_ctx, h, gpu_mom, gpu_ang, nb, base);
        }
        free(tmp_mom); free(tmp_ang);
        cudaFree(gpu_mom); cudaFree(gpu_ang);
        if (failed) goto fail;
    }
    return h;
fail:
    cudaFree((void*)h->m); cudaFree((void*)h->w); cudaFree((void*)h->b);
    if (h->amp_cache) cudaFree(h->amp_cache);
    free(h);
    return NULL;
}

void cuda_free_data_v4(void* vh) {
    if (!vh) return;
    DataHandle2* h = (DataHandle2*)vh;
    cudaFree((void*)h->m);
    cudaFree((void*)h->w); cudaFree((void*)h->b);
    if (h->amp_cache) cudaFree(h->amp_cache);
    free(h);
}

static int alloc_handle_amp_cache(ComputeContext* c, DataHandle2* h) {
    if (h->amp_cache != NULL || h->n_uniq > 0) return 1;
    if (c->n_uniq <= 0 || !c->rep_of_slot || !c->matrix_angle_real) return 0;
    size_t need = (size_t)h->ne * c->n_uniq * sizeof(float2);
    float2* amp_cache = NULL;
    if (cudaMalloc(&amp_cache, need) != cudaSuccess) return 0;
    h->amp_cache = amp_cache;
    h->n_uniq = c->n_uniq;
    return 1;
}

static int fill_amp_cache_chunk(ComputeContext* c, DataHandle2* h,
                                const float* mom, const float* ang,
                                int n_events, int base_event) {
    size_t shmem = (size_t)c->n_angle_k * sizeof(double);
    int fill_b = c->n_uniq;
    if (fill_b < 256) fill_b = 256;
    if (fill_b > 1024) fill_b = 1024;
    amp_cache_fill_kernel<<<n_events, fill_b, shmem>>>(
        ang, c->angle_index,
        c->angle_k, c->angle_b,
        c->matrix_angle_real, c->matrix_angle_imag,
        mom, c->fl_type, c->fl_q_index, c->fl_order,
        c->fl_table, c->fl_min, c->fl_delta,
        c->rep_of_slot,
        c->n_wave, c->n_angle_k, c->n_angle_total, c->n_angle_comp,
        c->n_decay, c->n_momentum, c->fl_table_bins, c->n_uniq,
        h->amp_cache + (size_t)base_event * c->n_uniq, n_events);
    return cudaGetLastError() == cudaSuccess;
}

void cuda_compute_v4(void* vctx, void* vdh,
    const double* ck_r,const double* ck_i,
    const double* m0,const double* g0,
    double nv,int use_norm,
    double* oQ,double* oP,
    double* ogck_r,double* ogck_i,
    double* ogm0,double* ogg0
) {
    ComputeContext* c = (ComputeContext*)vctx;
    DataHandle2* h = (DataHandle2*)vdh;
    int ne = h->ne, bs = c->batch_size;
    if (ne < bs) bs = ne;
    int nbat = (ne + bs - 1) / bs;
    int nw = c->n_wave, nu = c->n_unique_bw, ng = c->n_gamma_rows;
    int N = nw / c->n_proj;

    // Upload per-call params via _up_dbl (always fresh)
    ComputeParams p;
    p.ck_real = (double*)_up_dbl(ck_r, N);
    p.ck_imag = (double*)_up_dbl(ck_i, N);
    p.m0 = (double*)_up_dbl(m0, c->n_m0_params);
    p.g0 = (double*)_up_dbl(g0, c->n_g0_params);

    ComputeData s;
    if (c->scratch) {
        s = *c->scratch;
    } else {
        memset(&s, 0, sizeof(ComputeData));
        #define S(f) CUDA_CHECK(cudaMalloc(&s.f, bs * sizeof(double)))
        #define S2(f,n) CUDA_CHECK(cudaMalloc(&s.f, bs * (n) * sizeof(double)))
        S2(g_interp_real,ng); S2(g_interp_imag,ng);
        S2(g_bw_real,nu); S2(g_bw_imag,nu);
        S(Q_out); S(P_out);
        S2(bw_p_real,nw); S2(bw_p_imag,nw);
        S2(common_amp_factor_real,nw); S2(common_amp_factor_imag,nw);
        S2(bw_dom_real,nu); S2(bw_dom_imag,nu);
        S2(dQ_dA_real, c->n_proj); S2(dQ_dA_imag, c->n_proj);
        S2(grad_ck_real_partial,N); S2(grad_ck_imag_partial,N);
        S2(grad_m0_partial,nu); S2(grad_g0_partial,ng);
        S2(dQ_dbw_dom_real, nu); S2(dQ_dbw_dom_imag, nu);
        #undef S
        #undef S2
    }

    *oQ = 0; memset(oP, 0, ne * 8);
    memset(ogck_r, 0, N * 8); memset(ogck_i, 0, N * 8);
    memset(ogm0, 0, nu * 8); memset(ogg0, 0, ng * 8);

    cudaMemset(s.g_bw_real, 0, bs * nu * 8);
    cudaMemset(s.g_bw_imag, 0, bs * nu * 8);
    cudaMemset(s.g_interp_real, 0, bs * ng * 8);
    cudaMemset(s.g_interp_imag, 0, bs * ng * 8);

    double* Ph = (double*)malloc(bs * 8);
    double* gck_buf = (double*)malloc(N * 8);
    double* gm0_buf = (double*)malloc(nu * 8);
    double* gg0_buf = (double*)malloc(ng * 8);

    for (int b = 0; b < nbat; b++) {
        int st = b * bs;
        int nb = (ne - st > bs) ? bs : (ne - st);

        ComputeData d = s;
        d.mass = h->m + st * h->nm;
        d.weight = h->w + st;
        d.bkg = h->b + st;
        d.n_events = nb;
        d.amp_cache = (h->amp_cache != NULL)
            ? (h->amp_cache + (size_t)st * c->n_uniq) : NULL;

        launch_compute_all(c, &d, &p, nv, use_norm);
        cudaGetLastError();

        cudaMemcpy(Ph, d.Q_out, nb * 8, cudaMemcpyDeviceToHost);
        for (int i = 0; i < nb; i++) *oQ += Ph[i];

        cudaMemcpy(Ph, d.P_out, nb * 8, cudaMemcpyDeviceToHost);
        memcpy(oP + st, Ph, nb * 8);

        launch_reduce_sum_features(d.grad_ck_real_partial, s.g_bw_real, nb, N);
        cudaMemcpy(gck_buf, s.g_bw_real, N * 8, cudaMemcpyDeviceToHost);
        for (int j = 0; j < N; j++) ogck_r[j] += gck_buf[j];

        launch_reduce_sum_features(d.grad_ck_imag_partial, s.g_bw_imag, nb, N);
        cudaMemcpy(gck_buf, s.g_bw_imag, N * 8, cudaMemcpyDeviceToHost);
        for (int j = 0; j < N; j++) ogck_i[j] += gck_buf[j];

        launch_reduce_sum_features(d.grad_m0_partial, s.g_interp_real, nb, nu);
        cudaMemcpy(gm0_buf, s.g_interp_real, nu * 8, cudaMemcpyDeviceToHost);
        for (int j = 0; j < nu; j++) ogm0[j] += gm0_buf[j];

        launch_reduce_sum_features(d.grad_g0_partial, s.g_interp_imag, nb, ng);
        cudaMemcpy(gg0_buf, s.g_interp_imag, ng * 8, cudaMemcpyDeviceToHost);
        for (int j = 0; j < ng; j++) ogg0[j] += gg0_buf[j];
    }

    if (!c->scratch) {
        #define F(p) do { if(s.p) cudaFree(s.p); } while(0)
        F(g_interp_real); F(g_interp_imag); F(g_bw_real); F(g_bw_imag);
        F(Q_out); F(P_out);
        F(bw_p_real); F(bw_p_imag);
        F(common_amp_factor_real); F(common_amp_factor_imag);
        F(bw_dom_real); F(bw_dom_imag);
        F(dQ_dA_real); F(dQ_dA_imag);
        F(grad_ck_real_partial); F(grad_ck_imag_partial);
        F(grad_m0_partial); F(grad_g0_partial);
        F(dQ_dbw_dom_real); F(dQ_dbw_dom_imag);
        #undef F
    }
    free(Ph); free(gck_buf); free(gm0_buf); free(gg0_buf);

    cudaFree((void*)p.ck_real); cudaFree((void*)p.ck_imag);
    cudaFree((void*)p.m0); cudaFree((void*)p.g0);
}

} // extern "C"
