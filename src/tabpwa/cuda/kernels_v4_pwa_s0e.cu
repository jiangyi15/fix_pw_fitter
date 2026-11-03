/**
 * cuda_v4_pwa_s0e — SINGLE-KERNEL TILE variant of s0d (case-1 copy).
 *
 * s0d already caches the gamma interp at load (us_cache_fill_kernel ->
 * us_real/us_imag + g_interp), so its per-iteration cost is dominated by
 * GLOBAL traffic: common_amp_factor[e][n_wave], bw_dom[e][n_wave],
 * g_interp[e][n_gamma_rows] and the per-event gradient partials are
 * written to global memory and re-read by four separate kernels (K2..K5/K6).
 *
 * s0e fuses the WHOLE per-iteration path into ONE kernel (s0e_tile_kernel)
 * that keeps every intermediate in shared memory, exactly like the CPU
 * sibling (`cpu_pwa_s0c`):
 *
 *   phase A  BW_w, 1/BW_w, common = Amp/BW, common/BW      (per (event, wave))
 *   phase B  A_p = Σ_k ck_k·common, P = Σ_p|A_p|², Q, dQ/dA_p   (warp / event)
 *   phase C  dQ/dck_k = Σ_p dQ/dA_p·common                  (atomics)
 *   phase D  dQ/dcoeff[w,i] = 2·Re(dQ/dBW_w·F_i(e))         (atomics)
 *
 * No batching: ONE launch over the whole event range, grid =
 * ceil(ne / S0E_TE), S0E_TE events per block sharing a [S0E_TE·n_wave]
 * shared tile.  The persistent device accumulation (acc_buf) and the
 * pinned parameter staging of s0b/s0c are kept, so there is still ONE
 * blocking D2H per call.  The handle also carries a per-event P buffer,
 * copied once into the handle's pinned staging when the caller asks for P.
 *
 * The uniform-coefficient basis is unchanged from s0c/s0d:
 *
 *   BW_w = Σ_i M_wi·F_i(e),   F_0 = 1,  F_1 = m²,  F_l = -i·G_l(m)
 *   dQ/dM_wi = 2·Re(dQ/dBW_w · F_i(e))
 *
 * so the device outputs dQ/dcoeff (per (wave, basis slot)) and the WRAPPER
 * chains slot 0 -> dQ/ds0 and the gamma slots -> dQ/dmult onto the fitted
 * m0/g0 (exactly like cpu_pwa_s0c / s0c).
 *
 * Physics model (no time evolution / no mixing / no scalar params):
 *
 *     P(e) = Σ_p  | A_p(e) |² ,   A_p(e) = Σ_k  ck_k · a_{p,k}(e)
 *
 * p = 0..P-1 runs over the incoherent projections, k over the partial
 * waves; all projections share the SAME ck_k (the projection only changes
 * the angular part, which lives in the one-time amp_cache).
 *
 * Derived from cuda_v4_pwa_s0d (itself s0c + the load-time u-cache).  The
 * existing kernels are untouched (kernel policy case 1: standalone copy).
 */

#include <cstring>
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
    const double* __restrict__ us_r,          // s0d: u-cache [nb·n_uslot]
    const double* __restrict__ us_i,
    const double* __restrict__ coeff,         // s0c: [n_wave · n_basis]
    const int* __restrict__ slot_map,         // s0c: [n_wave · n_basis] -> uslot
    int n_wave, int n_basis, int n_uslot,
    const float2* __restrict__ amp_cache,
    const int* __restrict__ slot_of_wave,
    int n_uniq,
    double* __restrict__ common_amp_factor_real,
    double* __restrict__ common_amp_factor_imag,
    double* __restrict__ bw_dom_real, double* __restrict__ bw_dom_imag,
    int n_events
) {
    // s0d u-cache: the gamma interp (row, mass-col) values were filled
    // ONCE per data load by us_cache_fill_kernel (they depend only on the
    // fixed input mass columns + the fixed interp tables — nothing per
    // iteration).  This kernel is therefore a PURE matvec: one block per
    // event, no shared memory, no staging, no __syncthreads() at all.
    const int event_idx = blockIdx.x;
    if (event_idx >= n_events) return;
    const int tid = threadIdx.x;
    const int block_sz = blockDim.x;
    const double* ur = us_r + (size_t)event_idx * n_uslot;
    const double* ui = us_i + (size_t)event_idx * n_uslot;

    for (int wave_idx = tid; wave_idx < n_wave; wave_idx += block_sz) {
        int bw_idx = wave_idx;
        double bw_r = 0.0, bw_i = 0.0;
        int bo = wave_idx * n_basis;
        if (n_uslot > 0) {
            for (int i = 0; i < n_basis; i++) {
                int u = slot_map[bo + i];
                double M = coeff[bo + i];
                bw_r += M * ui[u];               // -i·Z: Re = +Z_i
                bw_i -= M * ur[u];               //       Im = -Z_r
            }
        }

        float2 A = amp_cache[event_idx * n_uniq + slot_of_wave[wave_idx]];
        double nrm = bw_r * bw_r + bw_i * bw_i;
        common_amp_factor_real[event_idx * n_wave + wave_idx] =
            ((double)A.x * bw_r + (double)A.y * bw_i) / nrm;
        common_amp_factor_imag[event_idx * n_wave + wave_idx] =
            ((double)A.y * bw_r - (double)A.x * bw_i) / nrm;
        bw_dom_real[(size_t)event_idx * n_uniq + bw_idx] = bw_r;
        bw_dom_imag[(size_t)event_idx * n_uniq + bw_idx] = bw_i;
    }
}

// ── s0d u-cache fill: the gamma interp for all (event, uslot) pairs,
//    computed ONCE per data load into handle-scoped arrays; also
//    scatters the per-ROW values (g_interp) the grad kernel consumes.
//    One block per event; the ONLY kernel in the s0d pipeline that
//    still contains the staging + barrier pattern (at load time, off
//    the fitting path).
__global__ void us_cache_fill_kernel(
    const double* __restrict__ mass, int n_mass,
    const int* __restrict__ uslot_row,        // [n_uslot] ext-table row
    const int* __restrict__ uslot_col,        // [n_uslot] mass column
    const int* __restrict__ slot_map,         // [n_wave · n_basis] -> uslot
    const int* __restrict__ slot_l,           // [n_wave · n_basis]
                                              //   merged row (-1: non-gamma)
    const double* __restrict__ gamma_table_real,
    const double* __restrict__ gamma_table_imag,
    double gamma_min, double gamma_inv_delta, int gamma_table_bins,
    int n_gamma_rows, int n_uslot, int n_wave, int n_basis,
    double* __restrict__ us_real, double* __restrict__ us_imag,
    double* __restrict__ gi_real, double* __restrict__ gi_imag,
    int n_events
) {
    const int event_idx = blockIdx.x;
    if (event_idx >= n_events) return;
    const int tid = threadIdx.x;
    const int block_sz = blockDim.x;

    extern __shared__ double s_dyn_us[];
    double* s_us_r = s_dyn_us;                // [n_uslot]
    double* s_us_i = s_dyn_us + n_uslot;
    for (int u = tid; u < n_uslot; u += block_sz) {
        complex z = interp_complex_device(
            gamma_table_real, gamma_table_imag, uslot_row[u],
            mass[(size_t)event_idx * n_mass + uslot_col[u]],
            gamma_min, gamma_inv_delta, gamma_table_bins);
        s_us_r[u] = z.real();
        s_us_i[u] = z.imag();
        us_real[(size_t)event_idx * n_uslot + u] = z.real();
        us_imag[(size_t)event_idx * n_uslot + u] = z.imag();
    }
    // s_us must be COMPLETE before the row scatter reads other threads'
    // (row, col) slots — the load-time analogue of the shipped s0c barrier.
    __syncthreads();
    for (int wave_idx = tid; wave_idx < n_wave; wave_idx += block_sz) {
        int bo = wave_idx * n_basis;
        for (int i = 2; i < n_basis; i++) {
            int l = slot_l[bo + i];
            if (l >= 0) {
                int u = slot_map[bo + i];
                gi_real[(size_t)event_idx * n_gamma_rows + l] = s_us_r[u];
                gi_imag[(size_t)event_idx * n_gamma_rows + l] = s_us_i[u];
            }
        }
    }
}

//=============================================================================
// s0e: the fused per-iteration tile kernel.
// One block owns S0E_TE consecutive events; every intermediate (BW, 1/BW,
// common, common/BW, A_p, dQ/dA_p) lives in shared memory, so the forward +
// all gradients complete with NO global round-trip and NO batching.
//
// shared: 4 arrays of [S0E_TE*n_wave] + 4 of [S0E_TE*n_proj] doubles
//=============================================================================
#define S0E_TE    8        /* events per block */
#define S0E_BLOCK 256      /* threads */

/* shared: 4 arrays of [S0E_TE*n_wave] + 4 of [S0E_TE*n_proj] doubles */
#define S0E_SHMEM(nw, P) \
    ((size_t)(4 * S0E_TE * (nw) + 4 * S0E_TE * (P)) * sizeof(double))

__global__ void s0e_tile_kernel(
    const double* __restrict__ coeff,        /* [n_wave*n_basis] */
    const int* __restrict__ slot_map,        /* [n_wave*n_basis] -> uslot */
    const int* __restrict__ slot_of_wave,    /* [n_wave] -> amp cache slot */
    const double* __restrict__ us_r,         /* [ne*n_uslot] */
    const double* __restrict__ us_i,
    const float2* __restrict__ amp_cache,    /* [ne*n_uniq] */
    const double* __restrict__ ck_r, const double* __restrict__ ck_i,
    const double* __restrict__ weight, const double* __restrict__ bkg,
    int n_wave, int N, int n_proj, int n_basis, int n_uslot, int n_uniq,
    int ne, int use_norm, double norm,
    double* __restrict__ Q_acc, double* __restrict__ dnorm_acc,
    double* __restrict__ g_ck_r, double* __restrict__ g_ck_i,
    double* __restrict__ g_coeff,
    double* __restrict__ P_out)              /* may be NULL, [ne] */
{
    const int e0 = blockIdx.x * S0E_TE;
    const int nen = (ne - e0 > S0E_TE) ? S0E_TE : (ne - e0);
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    const int P   = n_proj;
    const int nf  = n_basis;
    const int nc  = n_wave * nf;

    extern __shared__ double sh[];
    double* s_cr = sh;                                   /* common    */
    double* s_ci = s_cr + (size_t)S0E_TE * n_wave;
    double* s_xr = s_ci + (size_t)S0E_TE * n_wave;       /* common/BW */
    double* s_xi = s_xr + (size_t)S0E_TE * n_wave;
    double* s_ar = s_xi + (size_t)S0E_TE * n_wave;       /* A_p       */
    double* s_ai = s_ar + (size_t)S0E_TE * P;
    double* s_dr = s_ai + (size_t)S0E_TE * P;            /* dQ/dA_p   */
    double* s_di = s_dr + (size_t)S0E_TE * P;

    /* phase A: BW, 1/BW, common, common/BW (one thread per (event, wave)) */
    for (int idx = tid; idx < nen * n_wave; idx += nt) {
        const int e = idx / n_wave;
        const int w = idx - e * n_wave;
        const int bo = w * nf;
        const double* ur = us_r + (size_t)(e0 + e) * n_uslot;
        const double* ui = us_i + (size_t)(e0 + e) * n_uslot;
        double bw_r = 0.0, bw_i = 0.0;
        for (int i = 0; i < nf; i++) {
            const double M = coeff[bo + i];
            const int u = slot_map[bo + i];
            bw_r += M * ui[u];       /* -i*Z : Re = +Z_i */
            bw_i -= M * ur[u];       /*        Im = -Z_r */
        }
        const double den = bw_r * bw_r + bw_i * bw_i;
        const double ir =  bw_r / den;
        const double ii = -bw_i / den;
        const float2 A = amp_cache[(size_t)(e0 + e) * n_uniq + slot_of_wave[w]];
        const double ar = (double)A.x, ai = (double)A.y;
        const double m_r = ar * ir - ai * ii;
        const double m_i = ar * ii + ai * ir;
        const int o = e * n_wave + w;
        s_cr[o] = m_r; s_ci[o] = m_i;
        s_xr[o] = m_r * ir - m_i * ii;
        s_xi[o] = m_r * ii + m_i * ir;
    }
    __syncthreads();

    /* phase B: A_p, P, Q, dQ/dA (one warp per event) */
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int nwarps = nt >> 5;
    for (int e = wid; e < nen; e += nwarps) {
        const double* cr = s_cr + (size_t)e * n_wave;
        const double* ci = s_ci + (size_t)e * n_wave;
        for (int p = 0; p < P; p++) {
            double ar = 0.0, ai = 0.0;
            for (int k = lane; k < N; k += 32) {
                const int w = p * N + k;
                const double ckr = ck_r[k], cki = ck_i[k];
                const double mr = cr[w], mi = ci[w];
                ar += ckr * mr - cki * mi;
                ai += ckr * mi + cki * mr;
            }
            for (int off = 16; off > 0; off >>= 1) {
                ar += __shfl_down_sync(0xffffffffu, ar, off);
                ai += __shfl_down_sync(0xffffffffu, ai, off);
            }
            if (lane == 0) { s_ar[e * P + p] = ar; s_ai[e * P + p] = ai; }
        }
        __syncwarp();
        if (lane == 0) {
            double Pe = 0.0;
            for (int p = 0; p < P; p++)
                Pe += s_ar[e * P + p] * s_ar[e * P + p]
                    + s_ai[e * P + p] * s_ai[e * P + p];
            const double wv = weight[e0 + e];
            const double bv = bkg[e0 + e];
            double dQ_dP;
            if (use_norm) {
                const double dnm = Pe / norm + bv;
                atomicAdd(Q_acc, -wv * log(dnm));
                if (dnorm_acc)
                    atomicAdd(dnorm_acc, wv * (Pe / (norm * norm)) / dnm);
                dQ_dP = -wv / (norm * dnm);
            } else {
                atomicAdd(Q_acc, wv * Pe);
                dQ_dP = wv;
            }
            for (int p = 0; p < P; p++) {
                s_dr[e * P + p] =  dQ_dP * s_ar[e * P + p];
                s_di[e * P + p] = -dQ_dP * s_ai[e * P + p];
            }
            if (P_out) P_out[e0 + e] = Pe;
        }
    }
    __syncthreads();

    /* phase C: ck gradient */
    for (int k = tid; k < N; k += nt) {
        double sr = 0.0, si = 0.0;
        for (int p = 0; p < P; p++) {
            const int w = p * N + k;
            for (int e = 0; e < nen; e++) {
                const double dr = s_dr[e * P + p], di = s_di[e * P + p];
                const int o = e * n_wave + w;
                sr += dr * s_cr[o] - di * s_ci[o];
                si += dr * s_ci[o] + di * s_cr[o];
            }
        }
        atomicAdd(g_ck_r + k, sr);
        atomicAdd(g_ck_i + k, si);
    }

    /* phase D: coefficient gradient */
    for (int idx = tid; idx < nc; idx += nt) {
        const int w = idx / nf;
        const int i = idx - w * nf;
        const int p = w / N;
        const int k = w - p * N;
        const int u = slot_map[idx];
        const double ckr = ck_r[k], cki = ck_i[k];
        double acc = 0.0;
        for (int e = 0; e < nen; e++) {
            const double dr = s_dr[e * P + p], di = s_di[e * P + p];
            const int o = e * n_wave + w;
            const double ur = dr * ckr - di * cki;
            const double ui = dr * cki + di * ckr;
            const double dbr = -(ur * s_xr[o] - ui * s_xi[o]);
            const double dbi = -(ur * s_xi[o] + ui * s_xr[o]);
            const size_t uo = (size_t)(e0 + e) * n_uslot + u;
            acc += 2.0 * (dbr * us_i[uo] + dbi * us_r[uo]);
        }
        atomicAdd(g_coeff + idx, acc);
    }
}
//=============================================================================
// PWA wave-Gram reduction: D[k,j] = Σ_e w_e Σ_p conj(a_{p,k})·a_{p,j}
// One block per event; each thread owns one (k,j) cell of N×N.
//=============================================================================
__global__ void gram_v4_event_kernel(
    const double* __restrict__ common_r,
    const double* __restrict__ common_i,
    const double* __restrict__ weight,
    int nb, int P, int N,
    double* __restrict__ Dr, double* __restrict__ Di
) {
    int e = blockIdx.x;
    int tid = threadIdx.x;
    int bd = blockDim.x;
    size_t off = (size_t)e * (size_t)P * N;
    const double* cr = common_r + off;
    const double* ci = common_i + off;
    double we = (weight != NULL) ? weight[e] : 1.0;
    int total = N * N;
    for (int t = tid; t < total; t += bd) {
        int k = t / N, j = t % N;
        double sr = 0.0, si = 0.0;
        for (int p = 0; p < P; p++) {
            double kr = cr[p * N + k], ki = ci[p * N + k];
            double jr = cr[p * N + j], ji = ci[p * N + j];
            // conj(kr+iki) * (jr+iji)
            sr += kr * jr + ki * ji;
            si += kr * ji - ki * jr;
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
// Host-callable entry points
//=============================================================================
extern "C" {

typedef struct {
    // Index arrays (GPU) — the s0c/s0d create signature is kept verbatim.
    const int* m0_index; const int* g0_index;
    const int* g0_mass_index; const int* mass_index;
    const int* fl_type; const int* fl_q_index;
    const int* bw_order; const int* fl_order; const int* angle_index;
    const int* row_start;
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
    int n_m0_params; int n_g0_params;
    // angular-cache layout + projection count
    const int* slot_of_wave;   // [n_wave]  entry -> cache slot
    const int* rep_of_slot;    // [n_uniq]  slot -> representative entry
    int n_uniq;
    int n_proj;                // P — number of incoherent projections
    // s0b: persistent output accumulators
    // [Q, dnorm, gck_r(N), gck_i(N), gcoeff(n_wave*n_basis)]
    double* acc_buf;
    // s0b: persistent PARAMETER buffers, refreshed per call with a plain
    // memcpy (no cudaMalloc/cudaFree per iteration)
    double* p_ck_r; double* p_ck_i;     // [N]
    double* p_coeff;                    // [n_wave · n_basis]
    // s0b: PINNED host staging for the param upload
    double* h_ck_r; double* h_ck_i;
    double* h_coeff;                    // [n_wave · n_basis]
    // s0b: dedicated NON-LEGACY stream
    cudaStream_t stream;
    double* acc_host;                   // pinned [acc_total] output staging
    // uniform basis layout — n_basis slots per wave
    int n_basis;
    // deduped interp tables — one unique (table row, mass column) per entry
    int n_uslot;
    const int* uslot_row;               // [n_uslot]
    const int* uslot_col;               // [n_uslot]
    const int* slot_map;                // [n_wave · n_basis] -> uslot
    const int* slot_l;                  // [n_wave · n_basis] (-1: non-gamma)
    int n_gamma_rows_ext;               // n_gamma_rows + 2 (uploaded table)
    int gamma_table_bins_ext;
} ComputeContext;

typedef struct {
    // Event data kept on device: mass (u-cache source) + weight/bkg.
    const double* m;
    const double* w; const double* b;
    int ne; int nm; int nmom; int nat; int nac;
    float2* amp_cache;    // [ne · n_uniq], built at load_data
    int n_uniq;
    double* us_real;      // s0e u-cache [ne · n_uslot] gamma interp, once
    double* us_imag;
    double* gi_real;      // [ne · n_gamma_rows] row values (u-cache fill)
    double* gi_imag;
    double* h_P;          // pinned per-event P staging [ne]
    double* P_dev;        // s0e: per-event P (device, written by the kernel)
} DataHandle2;

cudaError_t cuda_alloc_s0e(void** ptr, size_t size) { return cudaMalloc(ptr, size); }
cudaError_t cuda_free_s0e(void* ptr) { return cudaFree(ptr); }
cudaError_t cuda_memcpy_to_device_s0e(void* dst, const void* src, size_t size) {
    return cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}
cudaError_t cuda_memcpy_to_host_s0e(void* dst, const void* src, size_t size) {
    return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}
cudaError_t cuda_memset_s0e(void* ptr, int value, size_t size) { return cudaMemset(ptr, value, size); }
int cuda_get_device_count_s0e() { int count; cudaGetDeviceCount(&count); return count; }
void cuda_s0e_set_flags() {
    // busy-wait synchronization — the default (interrupt/blocking) sync
    // costs ~1-2 ms per cudaStreamSynchronize on some drivers.  Must be
    // called BEFORE the primary context is created.
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);
}

cudaError_t cuda_get_device_name_s0e(char* name, int len) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err == cudaSuccess) strncpy(name, prop.name, len);
    return err;
}

// The shared tile of S0E_TE events exceeds the 48 KB static limit, so the
// kernel needs the opt-in dynamic-shared-memory limit (set once per size).
static void s0e_set_shmem(size_t shm) {
    static size_t s_set = 0;
    if (shm > s_set) {
        cudaFuncSetAttribute(s0e_tile_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)shm);
        s_set = shm;
    }
}

void cuda_free_context_s0e(void* vctx);  // forward decl (used by create on error)
void* cuda_create_context_s0e(
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
    int n_uniq, int n_proj,
    const int* row_start,int n_rs,
    int n_basis,
    const int* uslot_row,int n_ur,
    const int* uslot_col,int n_uc,
    const int* slot_map,int n_sm2,
    const int* slot_l,int n_sl,
    int n_gamma_rows_ext,int gamma_table_bins_ext
) {
    (void)batch_size;   // s0e does not batch
    ComputeContext* c = (ComputeContext*)calloc(1, sizeof(ComputeContext));
    c->m0_index = (int*)_up_int(m0_i, n1); c->g0_index = (int*)_up_int(g0_i, n2);
    c->g0_mass_index = (int*)_up_int(g0_m, n3); c->mass_index = (int*)_up_int(mass_i, n4);
    c->fl_type = (int*)_up_int(fl_t, n5); c->fl_q_index = (int*)_up_int(fl_q, n6);
    c->bw_order = (n7 > 0) ? (int*)_up_int(bw_o, n7) : NULL;  // unused (per-wave slots)
    c->fl_order = (int*)_up_int(fl_o, n8);
    c->angle_index = (int*)_up_int(ang_i, n9);
    c->angle_k = (double*)_up_dbl(ak, n10); c->angle_b = (double*)_up_dbl(ab, n11);
    c->matrix_angle_real = (double*)_up_dbl(mar, n12); c->matrix_angle_imag = (double*)_up_dbl(mai, n13);
    c->gamma_table_real = (double*)_up_dbl(gtr, n14); c->gamma_table_imag = (double*)_up_dbl(gti, n15);
    c->gamma_min = gmin; c->gamma_delta = gdel; c->gamma_table_bins = gbins;
    c->n_gamma_rows_ext = n_gamma_rows_ext;
    c->gamma_table_bins_ext = gamma_table_bins_ext;

    c->matrix_gamma = (n16 > 0) ? (double*)_up_dbl(mg, n16) : NULL;  // unused (per-wave rows)
    c->gamma_col_idx = (int*)_up_int(gci, ngci);
    c->fl_table = (double*)_up_dbl(ft, n17); c->fl_min = flmin; c->fl_delta = fldel; c->fl_table_bins = fbins;
    c->n_wave = nw; c->n_res = nr; c->n_decay = nd;
    c->n_unique_bw = nub; c->n_gamma_rows = ngr;
    c->n_mass = nm; c->n_momentum = nmom; c->n_angle_k = nak_; c->n_angle_total = nat; c->n_angle_comp = nac;
    c->n_m0_params = n_m0p; c->n_g0_params = n_g0p;
    c->slot_of_wave = (n_slot > 0) ? (int*)_up_int(slot_of_wave, n_slot) : NULL;
    c->rep_of_slot = (n_rep > 0) ? (int*)_up_int(rep_of_slot, n_rep) : NULL;
    c->row_start = (n_rs > 0) ? (int*)_up_int(row_start, n_rs) : NULL;
    c->n_basis = n_basis > 0 ? n_basis : 1;
    c->n_uslot = n_ur;
    c->uslot_row = (int*)_up_int(uslot_row, n_ur);
    c->uslot_col = (int*)_up_int(uslot_col, n_uc);
    c->slot_map = (int*)_up_int(slot_map, n_sm2);
    c->slot_l = (int*)_up_int(slot_l, n_sl);
    c->n_uniq = (n_uniq > 0) ? n_uniq : 0;
    c->n_proj = n_proj > 0 ? n_proj : 1;
    if (c->n_wave % c->n_proj != 0) {
        fprintf(stderr, "cuda_v4_pwa_s0e: n_wave %d must be divisible by n_proj %d\n",
                c->n_wave, c->n_proj);
        cuda_free_context_s0e(c);
        return NULL;
    }
    // the fused tile lives in shared memory: fail fast if it cannot fit
    {
        size_t shm = S0E_SHMEM(c->n_wave, c->n_proj);
        int shm_opt = 0;
        cudaDeviceGetAttribute(&shm_opt,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        if ((int)shm > shm_opt) {
            fprintf(stderr,
                    "cuda_v4_pwa_s0e: shared tile %zu B exceeds the device "
                    "opt-in limit %d B (n_wave %d, S0E_TE %d)\n",
                    shm, shm_opt, c->n_wave, S0E_TE);
            cuda_free_context_s0e(c);
            return NULL;
        }
    }
    // s0b: persistent accumulators (small, independent of the event count)
    {
        int N = c->n_wave / c->n_proj;
        int acc_total = 2 + 2 * N + c->n_wave * c->n_basis;
        CUDA_CHECK(cudaMalloc(&c->acc_buf, acc_total * sizeof(double)));
    }
    // s0b: persistent parameter buffers + pinned staging + dedicated stream
    {
        int N = c->n_wave / c->n_proj;
        int acc_total = 2 + 2 * N + c->n_wave * c->n_basis;
        CUDA_CHECK(cudaMalloc(&c->p_ck_r, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c->p_ck_i, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c->p_coeff,
                              c->n_wave * c->n_basis * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&c->h_ck_r, N * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&c->h_ck_i, N * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&c->h_coeff,
                                  c->n_wave * c->n_basis * sizeof(double)));
        CUDA_CHECK(cudaStreamCreate(&c->stream));
        CUDA_CHECK(cudaMallocHost(&c->acc_host, acc_total * sizeof(double)));
    }
    return c;
}

void cuda_free_context_s0e(void* vctx) {
    ComputeContext* c = (ComputeContext*)vctx;
    if (!c) return;
    #define F(p) cudaFree((void*)c->p)
    F(m0_index); F(g0_index); F(g0_mass_index); F(mass_index);
    F(fl_type); F(fl_q_index); F(bw_order); F(fl_order); F(angle_index);
    F(angle_k); F(angle_b); F(matrix_angle_real); F(matrix_angle_imag);
    F(gamma_table_real); F(gamma_table_imag); F(matrix_gamma); F(gamma_col_idx); F(fl_table); F(row_start);
    if (c->slot_of_wave) cudaFree((void*)c->slot_of_wave);
    if (c->rep_of_slot) cudaFree((void*)c->rep_of_slot);
    #undef F
    if (c->acc_buf) cudaFree(c->acc_buf);
    if (c->p_ck_r) cudaFree(c->p_ck_r);
    if (c->p_ck_i) cudaFree(c->p_ck_i);
    if (c->p_coeff) cudaFree(c->p_coeff);
    if (c->uslot_row) cudaFree((void*)c->uslot_row);
    if (c->uslot_col) cudaFree((void*)c->uslot_col);
    if (c->slot_map) cudaFree((void*)c->slot_map);
    if (c->slot_l) cudaFree((void*)c->slot_l);
    if (c->h_ck_r) cudaFreeHost(c->h_ck_r);
    if (c->h_ck_i) cudaFreeHost(c->h_ck_i);
    if (c->h_coeff) cudaFreeHost(c->h_coeff);
    if (c->acc_host) cudaFreeHost(c->acc_host);
    if (c->stream) cudaStreamDestroy(c->stream);
    free(c);
}

// Forward decls: the angular cache / u-cache are allocated once per handle.
static int alloc_handle_amp_cache(ComputeContext* c, DataHandle2* h);
static int alloc_handle_us_cache(ComputeContext* c, DataHandle2* h);
static int fill_amp_cache_chunk(ComputeContext* c, DataHandle2* h,
                                const float* mom, const float* ang,
                                int n_events, int base_event);

void* cuda_load_data_s0e(void* vctx,
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
    CUDA_CHECK(cudaMallocHost(&h->h_P, (size_t)ne * sizeof(double)));
    // s0e: per-event P (device) — written by the fused kernel, copied out
    // once when the caller asks for P
    if (cudaMalloc(&h->P_dev, (size_t)ne * sizeof(double)) != cudaSuccess)
        goto fail;

    if (c_ctx) {
        if (!alloc_handle_amp_cache(c_ctx, h)) goto fail;
        int fb = c_ctx->n_uniq * 8;          // transient event chunk
        if (fb < 1) fb = 1;
        if (fb > 16384) fb = 16384;
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
        // u-cache: gamma-interp values depend only on the fixed mass input
        // + the fixed interp tables — fill ONCE per handle.
        if (!alloc_handle_us_cache(c_ctx, h)) goto fail;
        if (c_ctx->n_uslot > 0 || c_ctx->n_gamma_rows > 0) {
            int fb2 = c_ctx->n_wave < 1024 ? c_ctx->n_wave : 1024;
            if (fb2 < 64) fb2 = 64;
            us_cache_fill_kernel<<<ne, fb2,
                (size_t)(2 * c_ctx->n_uslot) * sizeof(double),
                c_ctx->stream>>>(
                h->m, h->nm,
                c_ctx->uslot_row, c_ctx->uslot_col,
                c_ctx->slot_map, c_ctx->slot_l,
                c_ctx->gamma_table_real, c_ctx->gamma_table_imag,
                c_ctx->gamma_min, 1.0 / c_ctx->gamma_delta,
                c_ctx->gamma_table_bins_ext,
                c_ctx->n_gamma_rows, c_ctx->n_uslot,
                c_ctx->n_wave, c_ctx->n_basis,
                h->us_real, h->us_imag, h->gi_real, h->gi_imag, ne);
            if (cudaGetLastError() != cudaSuccess) goto fail;
            cudaStreamSynchronize(c_ctx->stream);
        }
    }
    return h;
fail:
    cudaFree((void*)h->m); cudaFree((void*)h->w); cudaFree((void*)h->b);
    if (h->amp_cache) cudaFree(h->amp_cache);
    if (h->us_real) cudaFree(h->us_real);
    if (h->us_imag) cudaFree(h->us_imag);
    if (h->gi_real) cudaFree(h->gi_real);
    if (h->gi_imag) cudaFree(h->gi_imag);
    if (h->P_dev) cudaFree(h->P_dev);
    if (h->h_P) cudaFreeHost(h->h_P);
    free(h);
    return NULL;
}

void cuda_free_data_s0e(void* vh) {
    if (!vh) return;
    DataHandle2* h = (DataHandle2*)vh;
    cudaFree((void*)h->m);
    cudaFree((void*)h->w); cudaFree((void*)h->b);
    if (h->amp_cache) cudaFree(h->amp_cache);
    if (h->us_real) cudaFree(h->us_real);
    if (h->us_imag) cudaFree(h->us_imag);
    if (h->gi_real) cudaFree(h->gi_real);
    if (h->gi_imag) cudaFree(h->gi_imag);
    if (h->P_dev) cudaFree(h->P_dev);
    if (h->h_P) cudaFreeHost(h->h_P);
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

static int alloc_handle_us_cache(ComputeContext* c, DataHandle2* h) {
    if (h->us_real != NULL || h->gi_real != NULL) return 1;
    size_t nus = (size_t)h->ne * (size_t)c->n_uslot;
    size_t ngi = (size_t)h->ne * (size_t)c->n_gamma_rows;
    if (nus) {
        if (cudaMalloc(&h->us_real, nus * sizeof(double)) != cudaSuccess)
            return 0;
        if (cudaMalloc(&h->us_imag, nus * sizeof(double)) != cudaSuccess)
            return 0;
    }
    if (ngi) {
        if (cudaMalloc(&h->gi_real, ngi * sizeof(double)) != cudaSuccess)
            return 0;
        if (cudaMalloc(&h->gi_imag, ngi * sizeof(double)) != cudaSuccess)
            return 0;
    }
    return 1;
}

static int fill_amp_cache_chunk(ComputeContext* c, DataHandle2* h,
                                const float* mom, const float* ang,
                                int n_events, int base_event) {
    size_t shmem = (size_t)c->n_angle_k * sizeof(double);
    int fill_b = c->n_uniq;
    if (fill_b < 256) fill_b = 256;
    if (fill_b > 1024) fill_b = 1024;
    amp_cache_fill_kernel<<<n_events, fill_b, shmem, c->stream>>>(
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

//=============================================================================
// s0e compute: ONE fused launch over the whole event range (no batching).
//=============================================================================
void cuda_compute_s0e(void* vctx, void* vdh,
    const double* ck_r, const double* ck_i, const double* coeff,
    double norm, int use_norm,
    double* oQ, double* oDn, double* oP,
    double* ogck_r, double* ogck_i, double* ogcoeff
) {
    ComputeContext* c = (ComputeContext*)vctx;
    DataHandle2* h = (DataHandle2*)vdh;
    const int ne = h->ne;
    const int nw = c->n_wave;
    const int P = c->n_proj;
    const int N = nw / P;
    const int nf = c->n_basis;
    const int ncb = nw * nf;
    const int acc_total = 2 + 2 * N + ncb;
    const int off_gck_r = 2, off_gck_i = 2 + N, off_gcoeff = 2 + 2 * N;

    // s0b: refresh the persistent parameter buffers (plain memcpy into the
    // pinned staging, then async H2D — no per-call malloc/free)
    memcpy(c->h_ck_r, ck_r, (size_t)N * sizeof(double));
    memcpy(c->h_ck_i, ck_i, (size_t)N * sizeof(double));
    memcpy(c->h_coeff, coeff, (size_t)ncb * sizeof(double));
    cudaMemcpyAsync(c->p_ck_r, c->h_ck_r, (size_t)N * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    cudaMemcpyAsync(c->p_ck_i, c->h_ck_i, (size_t)N * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    cudaMemcpyAsync(c->p_coeff, c->h_coeff, (size_t)ncb * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);

    *oQ = 0.0; *oDn = 0.0;
    cudaMemsetAsync(c->acc_buf, 0, (size_t)acc_total * sizeof(double),
                    c->stream);
    memset(ogck_r, 0, (size_t)N * sizeof(double));
    memset(ogck_i, 0, (size_t)N * sizeof(double));
    memset(ogcoeff, 0, (size_t)ncb * sizeof(double));

    if (ne > 0) {
        const int tiles = (ne + S0E_TE - 1) / S0E_TE;
        const size_t shm = S0E_SHMEM(nw, P);
        s0e_set_shmem(shm);
        s0e_tile_kernel<<<tiles, S0E_BLOCK, shm, c->stream>>>(
            c->p_coeff, c->slot_map, c->slot_of_wave,
            h->us_real, h->us_imag, h->amp_cache,
            c->p_ck_r, c->p_ck_i,
            h->w, h->b,
            nw, N, P, nf, c->n_uslot, c->n_uniq,
            ne, use_norm, norm,
            c->acc_buf + 0, c->acc_buf + 1,
            c->acc_buf + off_gck_r, c->acc_buf + off_gck_i,
            c->acc_buf + off_gcoeff,
            h->P_dev);
        CUDA_CHECK(cudaGetLastError());
    }

    if (oP != NULL)
        cudaMemcpyAsync(h->h_P, h->P_dev, (size_t)ne * sizeof(double),
                        cudaMemcpyDeviceToHost, c->stream);

    // ONE stream-scoped sync per call; the outputs were collected
    // asynchronously into pinned staging
    cudaMemcpyAsync(c->acc_host, c->acc_buf,
                    (size_t)acc_total * sizeof(double),
                    cudaMemcpyDeviceToHost, c->stream);
    cudaStreamSynchronize(c->stream);
    *oQ += c->acc_host[0];
    *oDn += c->acc_host[1];
    for (int j = 0; j < N; j++) ogck_r[j] += c->acc_host[off_gck_r + j];
    for (int j = 0; j < N; j++) ogck_i[j] += c->acc_host[off_gck_i + j];
    for (int j = 0; j < ncb; j++) ogcoeff[j] += c->acc_host[off_gcoeff + j];
    if (oP != NULL) memcpy(oP, h->h_P, (size_t)ne * sizeof(double));
}

//=============================================================================
// Wave Gram matrix D (N,N) for a loaded phsp handle, chunked over events.
// Still uses the cached forward (common = Amp/BW) + the N×N event reduction
// — the gram path is NOT part of the fused per-iteration kernel.
//   D[k,j] = Σ_e w_e Σ_p conj(common[e,pN+k])·common[e,pN+j]
//=============================================================================
void cuda_gram_matrix_s0e(void* vctx, void* vdh,
    const double* coeff,
    double* oDr, double* oDi
) {
    ComputeContext* c = (ComputeContext*)vctx;
    DataHandle2* h = (DataHandle2*)vdh;
    const int ne = h->ne;
    const int nw = c->n_wave, P = c->n_proj, N = nw / P;
    const int ncb = nw * c->n_basis;

    memcpy(c->h_coeff, coeff, (size_t)ncb * sizeof(double));
    cudaMemcpyAsync(c->p_coeff, c->h_coeff, (size_t)ncb * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);

    int chunk = 4096;
    if (ne < chunk) chunk = ne;
    if (chunk < 1) chunk = 1;
    double *cr = NULL, *ci = NULL, *br = NULL, *bi = NULL;
    CUDA_CHECK(cudaMalloc(&cr, (size_t)chunk * nw * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ci, (size_t)chunk * nw * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&br, (size_t)chunk * nw * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&bi, (size_t)chunk * nw * sizeof(double)));
    double* Dr = NULL; double* Di = NULL;
    size_t gsz = (size_t)N * N * sizeof(double);
    CUDA_CHECK(cudaMalloc(&Dr, gsz));
    CUDA_CHECK(cudaMalloc(&Di, gsz));
    CUDA_CHECK(cudaMemset(Dr, 0, gsz));
    CUDA_CHECK(cudaMemset(Di, 0, gsz));

    int amp_t = nw < MAX_THREADS ? nw : MAX_THREADS;
    if (amp_t < 32) amp_t = 32;
    for (int st = 0; st < ne; st += chunk) {
        int nb = (ne - st > chunk) ? chunk : (ne - st);
        amp_cache_amp_kernel<<<nb, amp_t, 0, c->stream>>>(
            h->us_real + (size_t)st * c->n_uslot,
            h->us_imag + (size_t)st * c->n_uslot,
            c->p_coeff, c->slot_map,
            nw, c->n_basis, c->n_uslot,
            h->amp_cache + (size_t)st * c->n_uniq,
            c->slot_of_wave, c->n_uniq,
            cr, ci, br, bi, nb);
        CUDA_CHECK(cudaGetLastError());
        gram_v4_event_kernel<<<nb, 128, 0, c->stream>>>(
            cr, ci, h->w + st, nb, P, N, Dr, Di);
        CUDA_CHECK(cudaGetLastError());
    }

    cudaStreamSynchronize(c->stream);
    cudaMemcpy(oDr, Dr, gsz, cudaMemcpyDeviceToHost);
    cudaMemcpy(oDi, Di, gsz, cudaMemcpyDeviceToHost);

    cudaFree(cr); cudaFree(ci); cudaFree(br); cudaFree(bi);
    cudaFree(Dr); cudaFree(Di);
}

} // extern "C"
