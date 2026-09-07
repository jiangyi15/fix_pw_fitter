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
#include <time.h>
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

// ── fixed-m0/g0 common-amplitude cache conversion kernels ─────────────
__global__ void store_common_d2(const double* __restrict__ re,
                                const double* __restrict__ im,
                                double2* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = make_double2(re[idx], im[idx]);
}
__global__ void load_common_d2(const double2* __restrict__ in,
                               double* __restrict__ re,
                               double* __restrict__ im, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { double2 v = in[idx]; re[idx] = v.x; im[idx] = v.y; }
}


// Fused fixed-cache forward + ck gradient (fpwfitter-style): one block per
// event reads the cached double2 amplitude directly (no double conversion),
// computes A_p, P/Q/dQdP and the projection-reduced ck-gradient partials.
__global__ void pwa_fused_cache_kernel(
    const double2* __restrict__ common,
    const double* __restrict__ ck_real, const double* __restrict__ ck_imag,
    const double* __restrict__ weight, const double* __restrict__ bkg,
    int n_wave, int n_proj, int n_events, int use_norm, double norm,
    double* __restrict__ Q_out, double* __restrict__ P_out,
    double* __restrict__ gk_r, double* __restrict__ gk_i
) {
    int e = blockIdx.x;
    int tid = threadIdx.x;
    int bd = blockDim.x;
    int P = n_proj;
    int N = n_wave / P;
    const double2* cm = common + (size_t)e * n_wave;

    extern __shared__ double s_dyn[];
    double* sR   = s_dyn;
    double* sI   = s_dyn + bd;
    double* sAr  = s_dyn + 2 * bd;
    double* sAi  = s_dyn + 2 * bd + P;
    double* sq   = s_dyn + 2 * bd + 2 * P;

    for (int p = 0; p < P; p++) {
        double ar = 0.0, ai = 0.0;
        for (int k = tid; k < N; k += bd) {
            double2 cv = cm[p * N + k];
            double ckr = ck_real[k], cki = ck_imag[k];
            ar += ckr * cv.x - cki * cv.y;
            ai += ckr * cv.y + cki * cv.x;
        }
        sR[tid] = ar; sI[tid] = ai;
        __syncthreads();
        for (int s2 = bd / 2; s2 > 0; s2 >>= 1) {
            if (tid < s2) { sR[tid] += sR[tid + s2]; sI[tid] += sI[tid + s2]; }
            __syncthreads();
        }
        if (tid == 0) { sAr[p] = sR[0]; sAi[p] = sI[0]; }
        __syncthreads();
    }

    if (tid == 0) {
        double Ps = 0.0;
        for (int p = 0; p < P; p++)
            Ps += sAr[p] * sAr[p] + sAi[p] * sAi[p];
        P_out[e] = Ps;
        double wv = weight[e];
        double bv = bkg[e];
        double q;
        if (use_norm == 0) {
            q = wv * Ps;
            sq[0] = wv;
        } else {
            q = -wv * log(Ps / norm + bv);
            sq[0] = -wv / (Ps + bv * norm);
        }
        Q_out[e] = q;
    }
    __syncthreads();

    // projection-reduced ck gradient partial per (e, k):
    //   dQ/dck_k = Σ_p dQ_dA_p · common[e,p·N+k],  dQ_dA_p = dQdP·conj(A_p)
    double dqp = sq[0];
    for (int k = tid; k < N; k += bd) {
        double gr = 0.0, gi = 0.0;
        for (int p = 0; p < P; p++) {
            double2 cv = cm[p * N + k];
            double dqr = dqp * sAr[p];
            double dqi = -dqp * sAi[p];
            gr += dqr * cv.x - dqi * cv.y;
            gi += dqr * cv.y + dqi * cv.x;
        }
        gk_r[(size_t)e * N + k] = gr;
        gk_i[(size_t)e * N + k] = gi;
    }
}

// fpwfitter-style: ONE THREAD PER EVENT reading the coalesced [k][p][e]
// cache layout (no shared memory / block sync).  Grid = ceil(ne/256).
#define PWA_MAX_P 128
__global__ void pwa_fused_thread_kernel(
    const double2* __restrict__ commonT,
    const double* __restrict__ ck_real, const double* __restrict__ ck_imag,
    const double* __restrict__ weight, const double* __restrict__ bkg,
    int n_wave, int n_proj, int base_event, int nb, int ne_total,
    int use_norm, double norm,
    double* __restrict__ Q_out, double* __restrict__ P_out,
    double* __restrict__ gk_r, double* __restrict__ gk_i
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb) return;
    int eG = base_event + i;
    const int P = n_proj;
    const int N = n_wave / P;

    double aR[PWA_MAX_P], aI[PWA_MAX_P];
    if (P > PWA_MAX_P) return;
    double Ps = 0.0;
    for (int p = 0; p < P; p++) {
        double ar = 0.0, ai = 0.0;
        for (int k = 0; k < N; k++) {
            double2 cv = commonT[(size_t)k * ((size_t)ne_total * P) + (size_t)eG * P + p];
            ar += ck_real[k] * cv.x - ck_imag[k] * cv.y;
            ai += ck_real[k] * cv.y + ck_imag[k] * cv.x;
        }
        aR[p] = ar; aI[p] = ai;
        Ps += ar * ar + ai * ai;
    }
    P_out[i] = Ps;
    double wv = weight[i];
    double bv = bkg[i];
    double dqp;
    if (use_norm == 0) {
        Q_out[i] = wv * Ps;
        dqp = wv;
    } else {
        double den = Ps + bv * norm;
        Q_out[i] = -wv * log(Ps / norm + bv);
        dqp = -wv / den;
    }
    for (int k = 0; k < N; k++) {
        double gr = 0.0, gi = 0.0;
        for (int p = 0; p < P; p++) {
            double2 cv = commonT[(size_t)k * ((size_t)ne_total * P) + (size_t)eG * P + p];
            double dqr = dqp * aR[p];
            double dqi = -dqp * aI[p];
            gr += dqr * cv.x - dqi * cv.y;
            gi += dqr * cv.y + dqi * cv.x;
        }
        gk_r[(size_t)i * N + k] = gr;
        gk_i[(size_t)i * N + k] = gi;
    }
}
#undef PWA_MAX_P

// fpwfitter-style SINGLE forward kernel over the fpwf layout.  One thread per
// event, two passes (no per-thread P-arrays -> no local memory): pass 1 sums
// |A_p|^2 and writes Q/P, pass 2 recomputes A_p and writes the complex
// gradient vector G[e*P+p] = dQdP * conj(A_p).  The ck gradient is then one
// cuBLAS ZGEMV over F (see cuda_compute_v4_cache).
__global__ void pwa_fpwf_forward_kernel(
    const double2* __restrict__ F,
    const double* __restrict__ ck_real, const double* __restrict__ ck_imag,
    const double* __restrict__ weight, const double* __restrict__ bkg,
    int n_wave, int n_proj, int ne,
    int use_norm, const double* __restrict__ norm_d,
    double* __restrict__ Q_out, double* __restrict__ P_out,
    double2* __restrict__ G,
    double* __restrict__ dnsum, double* __restrict__ qsum
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= ne) return;
    const double norm = norm_d[0];
    const int P = n_proj;
    const int N = n_wave / P;
    const size_t neP = (size_t)ne * P;
    double aR[8], aI[8];
    if (P > 8) return;   // register single-pass version; P<=8 covers all models
    double Ps = 0.0;
    for (int p = 0; p < P; p++) {
        const double2* Fp = F + (size_t)p * ne + e;   // coalesced across e
        double ar = 0.0, ai = 0.0;
        for (int k = 0; k < N; k++) {
            double2 cv = Fp[(size_t)k * neP];
            ar += ck_real[k] * cv.x - ck_imag[k] * cv.y;
            ai += ck_real[k] * cv.y + ck_imag[k] * cv.x;
        }
        aR[p] = ar; aI[p] = ai;
        Ps += ar * ar + ai * ai;
    }
    P_out[e] = Ps;
    double wv = weight[e];
    double bv = bkg[e];
    double dqp;
    if (use_norm == 0) {
        double q = wv * Ps;
        Q_out[e] = q;
        if (qsum) atomicAdd(qsum, q);
        dqp = wv;
    } else {
        double den = Ps + bv * norm;
        double q = -wv * log(Ps / norm + bv);
        Q_out[e] = q;
        if (qsum) atomicAdd(qsum, q);
        dqp = -wv / den;
        // d(NLL)/d(norm) partial = -P*dqp/norm ; accumulated on device so the
        // host never needs the whole per-event P for the gradient chain.
        if (dnsum) atomicAdd(dnsum, -(Ps * dqp) / norm);
    }
    // G row = p*ne + e, written from registers (no second F pass)
    for (int p = 0; p < P; p++) {
        double2* Gp = G + (size_t)p * ne + e;
        Gp[0] = make_double2(dqp * aR[p], -dqp * aI[p]);
    }
}

// ck-gradient reduction replacing the cuBLAS ZGEMV.  F is column-per-k
// (column k at F + (long)k*total, contiguous).  Grid is (row segments) x
// (columns handled per pass); a block reduces its row segment of a column
// and atomicAdds into out[k].  Columns are strided over gridDim.y so ANY
// N is supported (gridDim.y stays small); each column stream is read
// coalesced.
__global__ void fpwf_gradreduce_kernel(
    const double2* __restrict__ F,
    const double2* __restrict__ G,
    double2* __restrict__ out,
    int N, int total, int seg_len
) {
    int tid = threadIdx.x;
    int bd = blockDim.x;
    long start = (long)blockIdx.x * seg_len;
    long end = start + seg_len;
    if (end > total) end = total;
    __shared__ double2 sh[256];
    for (int k = blockIdx.y; k < N; k += gridDim.y) {
        const double2* Fc = F + (long)k * total;
        double ar = 0.0, ai = 0.0;
        for (long r = start + tid; r < end; r += bd) {
            double2 f = Fc[r];
            double2 g = G[r];
            ar += f.x * g.x - f.y * g.y;
            ai += f.x * g.y + f.y * g.x;
        }
        sh[tid] = make_double2(ar, ai);
        __syncthreads();
        for (int st = bd / 2; st > 0; st >>= 1) {
            if (tid < st) {
                sh[tid].x += sh[tid + st].x;
                sh[tid].y += sh[tid + st].y;
            }
            __syncthreads();
        }
        if (tid == 0) {
            atomicAdd(&out[k].x, sh[0].x);
            atomicAdd(&out[k].y, sh[0].y);
        }
        __syncthreads();  // protect sh reuse across k iterations
    }
}



// event-major common_cache -> coalesced [k][p][e] layout:
//   dst[k * neP + p * ne + e] = src[e*n_wave + p*N + k] (fpwf col-per-k)
//   row = p*ne + e -> consecutive events at consecutive addresses: fully
//   coalesced warp reads for every (k,p) stream; reduce uses same row order.
__global__ void transpose_common_kernel(
    const double2* __restrict__ src, double2* __restrict__ dst,
    int ne, int P, int N, int n_wave, int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int e = idx / n_wave;
    int w = idx - e * n_wave;
    int p = w / N;
    int k = w - p * N;
    dst[(size_t)k * ((size_t)ne * P) + (size_t)p * ne + e] = src[idx];
}

// D[k,j] = Σ_e w_e Σ_p conj(a_{p,k})·a_{p,j} straight from the cached
// full-amplitude double2 buffer (fixed m0/g0; no BW recompute).
__global__ void gram_cache_event_kernel(
    const double2* __restrict__ common,
    const double* __restrict__ weight,
    int nb, int P, int N,
    double* __restrict__ Dr, double* __restrict__ Di
) {
    int e = blockIdx.x;
    int tid = threadIdx.x;
    int bd = blockDim.x;
    const double2* ce = common + (size_t)e * (size_t)P * N;
    double we = (weight != NULL) ? weight[e] : 1.0;
    int total = N * N;
    for (int t = tid; t < total; t += bd) {
        int k = t / N, j = t % N;
        double sr = 0.0, si = 0.0;
        for (int p = 0; p < P; p++) {
            double2 ck = ce[p * N + k];
            double2 cj = ce[p * N + j];
            // conj(ck) * cj
            sr += ck.x * cj.x + ck.y * cj.y;
            si += ck.x * cj.y - ck.y * cj.x;
        }
        atomicAdd(&Dr[t], we * sr);
        atomicAdd(&Di[t], we * si);
    }
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
    // persistent fpwf evaluate scratch (allocated once, reused every call)
    double2* dgck;   // [N]  gradient-reduction output
    double*  qacc;   // [1]  total-Q device accumulator
    double*  dacc;   // [1]  d(NLL)/d(norm) device accumulator
    // per-call graph inputs (contents updated by memcpy; pointers stable)
    double*  d_norm;  // [1]  norm value
    double*  dck_r;   // [N]  ck real
    double*  dck_i;   // [N]  ck imag
    // CUDA graph experiment (whole fixed-cache evaluation, replayed per call)
    cudaStream_t gstream;
    cudaGraph_t graph;
    cudaGraphExec_t gexec;
    void* graph_handle;    // DataHandle the graph was captured for
    int graph_mode;        // use_norm value the graph was captured with
    int graph_ne;          // ne the graph was captured with (addr-reuse guard)
    const double2* graph_ct;  // common_T the graph was captured with
    int graph_valid;
    int graph_attempted;
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
    // Event data kept on device: ONLY the scalar weight/bkg arrays plus the
    // persistent fixed-amplitude cache.  mass/momentum/angle are transient
    // per-batch inputs to the one-time cache fill (cuda_fill_common_v4_cache),
    // uploaded from host arrays exactly like the angular fill batches —
    // nothing is stored "to be freed later".
    const double* w; const double* b;
    int ne;
    double2* common_cache;   // [ne · n_wave]  per-entry a_{p,k} at fixed m0/g0
    double2* common_T;       // fpwf layout [k][e*P+p] (cuBLAS column per k)
    double2* dG;             // [ne*P] gradient vector G (device)
    double* dQ;              // [ne] per-event Q (device)
    double* dP;              // [ne] per-event P (device)
    int fpwf_ok;             // device bufs allocated
    int cache_valid;
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
    c->dgck = NULL; c->qacc = NULL; c->dacc = NULL;
    c->d_norm = NULL; c->dck_r = NULL; c->dck_i = NULL;
    c->gstream = NULL; c->graph = NULL; c->gexec = NULL;
    c->graph_handle = NULL; c->graph_mode = -1; c->graph_ne = 0;
    c->graph_ct = NULL;
    c->graph_valid = 0; c->graph_attempted = 0;
    {
        int N = c->n_wave / c->n_proj;
        if (cudaMalloc(&c->dgck, (size_t)N * sizeof(double2)) != cudaSuccess ||
            cudaMalloc(&c->qacc, sizeof(double)) != cudaSuccess ||
            cudaMalloc(&c->dacc, sizeof(double)) != cudaSuccess ||
            cudaMalloc(&c->d_norm, sizeof(double)) != cudaSuccess ||
            cudaMalloc(&c->dck_r, (size_t)N * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&c->dck_i, (size_t)N * sizeof(double)) != cudaSuccess) {
            cuda_free_context_v4(c);
            return NULL;
        }
        if (cudaStreamCreateWithFlags(&c->gstream, cudaStreamNonBlocking)
                != cudaSuccess) c->gstream = NULL;
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
    if (c->gexec) { cudaGraphExecDestroy(c->gexec); c->gexec = NULL; }
    if (c->graph) { cudaGraphDestroy(c->graph); c->graph = NULL; }
    if (c->gstream) { cudaStreamDestroy(c->gstream); c->gstream = NULL; }
    if (c->dgck) cudaFree(c->dgck);
    if (c->qacc) cudaFree(c->qacc);
    if (c->dacc) cudaFree(c->dacc);
    if (c->d_norm) cudaFree(c->d_norm);
    if (c->dck_r) cudaFree(c->dck_r);
    if (c->dck_i) cudaFree(c->dck_i);
    if (c->Q_red_gpu) cudaFree(c->Q_red_gpu);
    free(c);
}

// Forward decls (legacy angular-cache path not used by this fixed kernel).

// Standard interface: load_data only keeps weight/bkg on the device.  The
// fixed-amplitude cache is built LAZILY at the first compute() from the
// host arrays retained on the CPU side (cuda_fill_common_v4_cache), so no
// extra arguments break the kernel/backend interface.
void* cuda_load_data_v4(void* vctx,
    const double* mass,int nmass, const double* mom,int nmom,
    const double* ang,int nang,
    const double* wgt,const double* bkg,int ne
) {
    (void)vctx; (void)mass; (void)nmass; (void)mom; (void)nmom;
    (void)ang; (void)nang;
    DataHandle2* h = (DataHandle2*)calloc(1, sizeof(DataHandle2));
    if (!h) return NULL;
    h->ne = ne;
    h->common_cache = NULL;
    h->common_T = NULL;
    h->dG = NULL; h->dQ = NULL; h->dP = NULL;
    h->fpwf_ok = 0;
    h->cache_valid = 0;
    h->w = (const double*)_up_dbl(wgt, ne);
    h->b = (const double*)_up_dbl(bkg, ne);
    if (!h->w || !h->b) goto fail;
    return h;
fail:
    cudaFree((void*)h->w); cudaFree((void*)h->b);
    free(h);
    return NULL;
}

void cuda_free_data_v4(void* vh) {
    if (!vh) return;
    DataHandle2* h = (DataHandle2*)vh;
    cudaFree((void*)h->w); cudaFree((void*)h->b);
    if (h->common_cache) cudaFree(h->common_cache);
    if (h->common_T) cudaFree(h->common_T);
    if (h->dG) cudaFree(h->dG);
    if (h->dQ) cudaFree(h->dQ);
    if (h->dP) cudaFree(h->dP);
    free(h);
}



//=============================================================================
// Lazy one-time cache fill (called from the Python kernel on the first
// compute): build the fixed-m0/g0 full amplitude common[e, p·N+k] directly
// from HOST arrays.  mass / momentum / angle are uploaded per batch exactly
// like the angular-cache fill (nothing is stored on the device handle) and
// discarded after the batch; only common_cache + weight/bkg persist.
// Returns 1 on success.
//=============================================================================
static void launch_common_fill(ComputeContext* c, ComputeData* d,
                             const ComputeParams* p,
                             double2* dst_cache, int cache_base);

int cuda_fill_common_v4_cache(ComputeContext* c, DataHandle2* h,
    const double* m0,int nm0, const double* g0,int ng0,
    const double* mass,int nmass,
    const double* mom,int nmom,
    const double* ang,int nang
) {
    (void)nang;
    if (!c || !h || !c->scratch) return 0;
    int ne = h->ne, nw = c->n_wave, nu = c->n_unique_bw, ng = c->n_gamma_rows;
    int nac = c->n_angle_comp, nat = c->n_angle_total;
    int n_uniq = c->n_uniq;
    if (nm0 != c->n_m0_params || ng0 != c->n_g0_params) return 0;
    size_t nrow_ang = (size_t)nat * nac;

    if (h->common_cache == NULL) {
        if (cudaMalloc(&h->common_cache,
                       (size_t)ne * nw * sizeof(double2)) != cudaSuccess)
            return 0;
    }
    if (h->common_T == NULL) {
        if (cudaMalloc(&h->common_T,
                       (size_t)ne * nw * sizeof(double2)) != cudaSuccess)
            return 0;
    }

    int bs = c->batch_size;
    if (ne < bs) bs = ne;

    ComputeParams p;
    p.ck_real = NULL; p.ck_imag = NULL;
    p.m0 = (double*)_up_dbl(m0, c->n_m0_params);
    p.g0 = (double*)_up_dbl(g0, c->n_g0_params);

    // transient per-batch device buffers (same pattern as the angular fill)
    double* gpu_mass = NULL; float* gpu_mom = NULL;
    float* gpu_ang = NULL;   float2* gpu_amp = NULL;
    if (cudaMalloc(&gpu_mass, (size_t)bs * nmass * sizeof(double))
            != cudaSuccess) goto fail;
    if (cudaMalloc(&gpu_mom, (size_t)bs * nmom * sizeof(float))
            != cudaSuccess) goto fail;
    if (cudaMalloc(&gpu_ang, (size_t)bs * nat * nac * sizeof(float))
            != cudaSuccess) goto fail;
    if (cudaMalloc(&gpu_amp, (size_t)bs * n_uniq * sizeof(float2))
            != cudaSuccess) goto fail;

    for (int base = 0; base < ne; base += bs) {
        int nb = (ne - base > bs) ? bs : (ne - base);
        cudaMemcpy(gpu_mass, mass + (size_t)base * nmass,
                   (size_t)nb * nmass * sizeof(double),
                   cudaMemcpyHostToDevice);
        float* tmp_mom = (float*)malloc((size_t)nb * nmom * sizeof(float));
        float* tmp_ang = (float*)malloc((size_t)nb * nrow_ang * sizeof(float));
        if (!tmp_mom || !tmp_ang) { free(tmp_mom); free(tmp_ang); goto fail; }
        for (int i = 0; i < nb * nmom; i++)
            tmp_mom[i] = (float)mom[(size_t)base * nmom + i];
        for (int i = 0; i < nb * nrow_ang; i++)
            tmp_ang[i] = (float)ang[(size_t)base * nrow_ang + i];
        cudaMemcpy(gpu_mom, tmp_mom, (size_t)nb * nmom * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_ang, tmp_ang, (size_t)nb * nrow_ang * sizeof(float),
                   cudaMemcpyHostToDevice);
        free(tmp_mom); free(tmp_ang);

        ComputeData d = *c->scratch;
        d.mass = gpu_mass;
        d.weight = h->w + base;      // unused by the fill but kept valid
        d.bkg = h->b + base;
        d.n_events = nb;
        d.amp_cache = gpu_amp;

        // angular amp for this batch (minimal-slot factor Amp = fa·fl)
        size_t shmem = (size_t)c->n_angle_k * sizeof(double);
        int fill_b = n_uniq;
        if (fill_b < 256) fill_b = 256;
        if (fill_b > 1024) fill_b = 1024;
        amp_cache_fill_kernel<<<nb, fill_b, shmem>>>(
            gpu_ang, c->angle_index,
            c->angle_k, c->angle_b,
            c->matrix_angle_real, c->matrix_angle_imag,
            gpu_mom, c->fl_type, c->fl_q_index, c->fl_order,
            c->fl_table, c->fl_min, c->fl_delta,
            c->rep_of_slot,
            nw, c->n_angle_k, nat, nac,
            c->n_decay, c->n_momentum, c->fl_table_bins, n_uniq,
            gpu_amp, nb);
        if (cudaGetLastError() != cudaSuccess) goto fail;

        launch_common_fill(c, &d, &p, h->common_cache, base);
    }

    // coalesced [k][p][e] copy for the fused forward kernel
    {
        int total = ne * nw;
        int gb = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        transpose_common_kernel<<<gb, BLOCK_SIZE>>>(
            h->common_cache, h->common_T, ne, c->n_proj,
            nw / c->n_proj, nw, total);
        CUDA_CHECK(cudaGetLastError());
    }

    cudaFree(gpu_mass); cudaFree(gpu_mom);
    cudaFree(gpu_ang); cudaFree(gpu_amp);
    cudaFree((void*)p.m0); cudaFree((void*)p.g0);
    h->cache_valid = 1;
    return 1;
fail:
    if (gpu_mass) cudaFree(gpu_mass);
    if (gpu_mom) cudaFree(gpu_mom);
    if (gpu_ang) cudaFree(gpu_ang);
    if (gpu_amp) cudaFree(gpu_amp);
    if (p.m0) cudaFree((void*)p.m0);
    if (p.g0) cudaFree((void*)p.g0);
    return 0;
}

// K1 (g_bw) + K2 (BW x angular amp) for one batch, stored to the cache.
static void launch_common_fill(ComputeContext* c, ComputeData* d,
                             const ComputeParams* p,
                             double2* dst_cache, int cache_base) {
    int nw = c->n_wave, nu = c->n_unique_bw, ng = c->n_gamma_rows;
    (void)ng;
    int ne = d->n_events;
    int amp_t = nw < MAX_THREADS ? nw : MAX_THREADS;
    if (amp_t < 32) amp_t = 32;

    launch_compute_g_bw(
        d->mass, p->g0, c->g0_index, c->g0_mass_index, c->gamma_col_idx,
        c->gamma_table_real, c->gamma_table_imag,
        c->gamma_min, c->gamma_delta,
        c->n_gamma_rows, c->n_unique_bw, c->n_mass, c->gamma_table_bins,
        d->g_interp_real, d->g_interp_imag,
        d->g_bw_real, d->g_bw_imag, ne);
    CUDA_CHECK(cudaGetLastError());

    size_t shmem = (size_t)(c->n_mass + 2 * nu) * sizeof(double);
    amp_cache_amp_kernel<<<ne, amp_t, shmem>>>(
        d->mass, c->m0_index, c->mass_index, c->bw_order,
        d->g_bw_real, d->g_bw_imag,
        nw, c->n_res, nu, c->n_mass, p->m0,
        d->amp_cache, c->slot_of_wave, c->n_uniq,
        d->bw_p_real, d->bw_p_imag,
        d->common_amp_factor_real, d->common_amp_factor_imag,
        d->bw_dom_real, d->bw_dom_imag, ne);
    CUDA_CHECK(cudaGetLastError());

    int ntot = ne * nw;
    int gb = (ntot + BLOCK_SIZE - 1) / BLOCK_SIZE;
    store_common_d2<<<gb, BLOCK_SIZE>>>(
        d->common_amp_factor_real, d->common_amp_factor_imag,
        dst_cache + (size_t)cache_base * nw, ntot);
    CUDA_CHECK(cudaGetLastError());
}

void cuda_compute_v4_cache(void* vctx, void* vdh,
    const double* ck_r,const double* ck_i,
    const double* m0,const double* g0,
    double nv,int use_norm,
    double* oQ,double* oP,
    double* ogck_r,double* ogck_i,
    double* ogm0,double* ogg0,
    double* odn
) {
    // fpwfitter-style evaluation over the cached full amplitude:
    //   one fused forward launch (whole dataset) writing Q/P and the
    //   gradient vector G[e*P+p] = dQdP*conj(A_p), then a SINGLE cuBLAS
    //   ZGEMV (OP_T over the (ne*P) x N column-major F) gives the whole
    //   ck gradient.  No batching, no per-event gk partials, no
    //   feature-reduces.  m0/g0 are fixed by the cache -> zero grads.
    ComputeContext* c = (ComputeContext*)vctx;
    DataHandle2* h = (DataHandle2*)vdh;
    int ne = h->ne;
    int nw = c->n_wave, nu = c->n_unique_bw, ng = c->n_gamma_rows;
    int P = c->n_proj, N = nw / P;
    size_t np_rows = (size_t)ne * P;

    int want_dn = (odn != NULL && use_norm != 0);
    if (odn) *odn = 0.0;

    // graph/launch shared declarations (kept above the goto guards)
    struct timespec _ts;
    double _hs, _he;
    int _graph_hit;
    int thr = (int)(((size_t)ne + 255) / 256);
    int segrows = 2048;
    int nseg = (int)((np_rows + segrows - 1) / segrows);
    dim3 grid((unsigned)nseg, (unsigned)N);

    ComputeParams p;
    p.ck_real = (double*)_up_dbl(ck_r, N);
    p.ck_imag = (double*)_up_dbl(ck_i, N);
    p.m0 = (double*)_up_dbl(m0, c->n_m0_params);
    p.g0 = (double*)_up_dbl(g0, c->n_g0_params);

    if (!h->cache_valid || !h->common_cache || !h->common_T) {
        *oQ = 0.0;
        memset(oP, 0, (size_t)ne * sizeof(double));
        memset(ogck_r, 0, (size_t)N * sizeof(double));
        memset(ogck_i, 0, (size_t)N * sizeof(double));
        memset(ogm0, 0, (size_t)nu * sizeof(double));
        memset(ogg0, 0, (size_t)ng * sizeof(double));
        goto out;
    }
    if (!h->fpwf_ok) {
        if (cudaMalloc(&h->dG, np_rows * sizeof(double2)) != cudaSuccess ||
            cudaMalloc(&h->dQ, (size_t)ne * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&h->dP, (size_t)ne * sizeof(double)) != cudaSuccess) {
            *oQ = 0.0;
            memset(oP, 0, (size_t)ne * sizeof(double));
            memset(ogck_r, 0, (size_t)N * sizeof(double));
            memset(ogck_i, 0, (size_t)N * sizeof(double));
            memset(ogm0, 0, (size_t)nu * sizeof(double));
            memset(ogg0, 0, (size_t)ng * sizeof(double));
            goto out;
        }
        h->fpwf_ok = 1;
    }

    *oQ = 0;
    memset(oP, 0, (size_t)ne * sizeof(double));
    memset(ogck_r, 0, (size_t)N * sizeof(double));
    memset(ogck_i, 0, (size_t)N * sizeof(double));
    memset(ogm0, 0, (size_t)nu * sizeof(double));
    memset(ogg0, 0, (size_t)ng * sizeof(double));

    // ── fixed-cache evaluation: CUDA-graph replay when available, else
    // plain per-call launches.  norm + ck live in stable device buffers so
    // the captured kernels only ever see fixed pointers.
    clock_gettime(CLOCK_MONOTONIC, &_ts);
    _hs = 1e3 * _ts.tv_sec + _ts.tv_nsec / 1e6;
    _graph_hit = 0;

    #define _UPLOAD_INPUTS(STREAM)                                          \
        do {                                                               \
            cudaMemcpyAsync(c->d_norm, &nv, sizeof(double),                 \
                            cudaMemcpyHostToDevice, STREAM);               \
            cudaMemcpyAsync(c->dck_r, p.ck_real, (size_t)N*sizeof(double), \
                            cudaMemcpyDeviceToDevice, STREAM);             \
            cudaMemcpyAsync(c->dck_i, p.ck_imag, (size_t)N*sizeof(double), \
                            cudaMemcpyDeviceToDevice, STREAM);             \
        } while (0)

    if (c->gstream && c->gexec && c->graph_handle == (void*)h &&
        c->graph_mode == use_norm && c->graph_ne == ne &&
        c->graph_ct == h->common_T) {
        // replay the captured memset+forward+memset+gradreduce graph
        _UPLOAD_INPUTS(c->gstream);
        cudaGraphLaunch(c->gexec, c->gstream);
        cudaStreamSynchronize(c->gstream);
        _graph_hit = 1;
    } else {
        // (re)capture for this (handle, use_norm) pair
        if (c->gexec) { cudaGraphExecDestroy(c->gexec); c->gexec = NULL; }
        if (c->graph) { cudaGraphDestroy(c->graph); c->graph = NULL; }
        cudaStreamBeginCapture(c->gstream, cudaStreamCaptureModeThreadLocal);
        cudaMemsetAsync(c->qacc, 0, sizeof(double), c->gstream);
        cudaMemsetAsync(c->dgck, 0, (size_t)N * sizeof(double2), c->gstream);
        if (use_norm) cudaMemsetAsync(c->dacc, 0, sizeof(double), c->gstream);
        pwa_fpwf_forward_kernel<<<thr, 256, 0, c->gstream>>>(
            h->common_T, c->dck_r, c->dck_i, h->w, h->b,
            nw, P, ne, use_norm, c->d_norm,
            h->dQ, h->dP, h->dG,
            use_norm ? c->dacc : NULL, c->qacc);
        fpwf_gradreduce_kernel<<<grid, 256, 0, c->gstream>>>(
            h->common_T, h->dG, c->dgck, N, (int)np_rows, segrows);
        if (cudaStreamEndCapture(c->gstream, &c->graph) == cudaSuccess &&
            cudaGraphInstantiate(&c->gexec, c->graph, 0) == cudaSuccess) {
            c->graph_handle = h;
            c->graph_mode = use_norm;
            c->graph_ne = ne;
            c->graph_ct = h->common_T;
            _UPLOAD_INPUTS(c->gstream);
            cudaGraphLaunch(c->gexec, c->gstream);
            cudaStreamSynchronize(c->gstream);
            _graph_hit = 1;
        } else {
            // capture unsupported -> plain launches (default stream)
            if (c->graph) { cudaGraphDestroy(c->graph); c->graph = NULL; }
            _UPLOAD_INPUTS(0);   // stream 0 = legacy default stream
            cudaMemset(c->qacc, 0, sizeof(double));
            cudaMemset(c->dgck, 0, (size_t)N * sizeof(double2));
            if (use_norm) cudaMemset(c->dacc, 0, sizeof(double));
            pwa_fpwf_forward_kernel<<<thr, 256>>>(
                h->common_T, c->dck_r, c->dck_i, h->w, h->b,
                nw, P, ne, use_norm, c->d_norm,
                h->dQ, h->dP, h->dG,
                use_norm ? c->dacc : NULL, c->qacc);
            fpwf_gradreduce_kernel<<<grid, 256>>>(
                h->common_T, h->dG, c->dgck, N, (int)np_rows, segrows);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    #undef _UPLOAD_INPUTS

    // ---- host-side scalar/array copies, all AFTER the GPU kernels ----
    cudaMemcpy(oQ, c->qacc, sizeof(double), cudaMemcpyDeviceToHost);
    if (want_dn)
        cudaMemcpy(odn, c->dacc, sizeof(double), cudaMemcpyDeviceToHost);
    if (oP) cudaMemcpy(oP, h->dP, (size_t)ne * sizeof(double),
                       cudaMemcpyDeviceToHost);
    {
        double2* zh = (double2*)malloc((size_t)N * sizeof(double2));
        if (zh) {
            cudaMemcpy(zh, c->dgck, (size_t)N * sizeof(double2),
                       cudaMemcpyDeviceToHost);
            for (int j = 0; j < N; j++) {
                ogck_r[j] = zh[j].x;
                ogck_i[j] = zh[j].y;
            }
            free(zh);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &_ts);
    _he = 1e3 * _ts.tv_sec + _ts.tv_nsec / 1e6;
    fprintf(stderr, "[ampfit timing] %s host-core %.3f ms\n",
            _graph_hit ? "graph" : "launch", _he - _hs);
out:
    cudaFree((void*)p.ck_real); cudaFree((void*)p.ck_imag);
    cudaFree((void*)p.m0); cudaFree((void*)p.g0);
}


void cuda_gram_from_cache_v4(void* vctx, void* vdh,
                             double* oDr, double* oDi) {
    ComputeContext* c = (ComputeContext*)vctx;
    DataHandle2* h = (DataHandle2*)vdh;
    if (!c || !h || !h->cache_valid || !h->common_cache) return;
    int ne = h->ne, nw = c->n_wave, P = c->n_proj, N = nw / P;
    size_t gsz = (size_t)N * N * sizeof(double);
    double* Dr = NULL; double* Di = NULL;
    CUDA_CHECK(cudaMalloc(&Dr, gsz));
    CUDA_CHECK(cudaMalloc(&Di, gsz));
    CUDA_CHECK(cudaMemset(Dr, 0, gsz));
    CUDA_CHECK(cudaMemset(Di, 0, gsz));
    gram_cache_event_kernel<<<ne, 128>>>(
        h->common_cache, h->w, ne, P, N, Dr, Di);
    CUDA_CHECK(cudaGetLastError());
    cudaMemcpy(oDr, Dr, gsz, cudaMemcpyDeviceToHost);
    cudaMemcpy(oDi, Di, gsz, cudaMemcpyDeviceToHost);
    cudaFree(Dr); cudaFree(Di);
}

} // extern "C"
