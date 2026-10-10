/**
 * cuda_v4_pwa_s0c — cuda_v4_pwa_s0b with the UNIFORM COEFFICIENT format
 * and multi-event scheduling across the whole gradient family:
 *
 *   BW_w = Σ_i M_wi·F_i(e),  F_0 = 1, F_1 = m², F_l = -i·G_l(m)
 *
 * every denominator term is coefficient × complex basis (the real terms
 * keep their real fast-path), so every fitted coefficient shares ONE
 * gradient formula  dQ/dM_wi = 2·Re(dQ/dBW_w · F_i(e)):
 * - K5+K6 fuse into ONE grad_coeff kernel: the per-(e,w) chain dQ/dBW
 *   is computed once in registers (the former dQ_dbw_dom global
 *   round-trip and its buffers are gone) and reduced against every
 *   basis function in the same pass;
 * - K4 (ck gradient) and grad_coeff get the K3 multi-event scheduling
 *   (thread-per-event for small n, warp-per-event otherwise — no
 *   shared memory, no atomics, no block-per-event barriers).
 * Same math as s0b; summation orders differ at ~1e-16.
 *
 * cuda_v4_pwa_s0b — cuda_v4_pwa_s0 with DEVICE-SIDE OUTPUT ACCUMULATION:
 * Q, dNLL/dnorm and the four gradient partials are tree-reduced into
 * persistent context accumulators (reduce_sum's atomicAdd + an
 * accumulating features-reduce), so the per-batch blocking D2H copies
 * and host-side cross-batch sums disappear — ONE small blocking copy
 * per call.  Same math; summation order differs from s0 at ~1e-16
 * (tree+atomicAdd vs host-sequential).
 *
 * cuda_v4_pwa_s0 — projection-sum PWA kernel, ONE BW PER CHAIN with
 * DIRECT pole parameters s0 = m0^2 and row multiplier mult = m0*g0
 * (derived from cuda_v4_pwa_1bw; wrapper computes them from the fitted
 * m0/g0 per iteration — pure outside transform).
 *
 *       bw_w(e) = m_w(e)^2 - s0_w - i * mult_l * G_l(m_w(e))
 *
 * (sign convention: the overall minus vs the classical m0^2-m^2+i*m0*g
 * form is absorbed per wave by ck -> -ck; P and all gradients are
 * invariant).  The device outputs dQ/ds0 (per wave) and dQ/dmult (per
 * gamma row, from the stored g_interp); the HOST chains them to the
 * fitted m0/g0 (dQ/dm0 = 2*m0*dQ/ds0 + sum_l g0_l*dQ/dmult_l,
 * dQ/dg0_l = m0_w*dQ/dmult_l) — no m0/g0 gradient algebra on device.
 *
 * SPECIALISATION: every wave carries exactly ONE Breit–Wigner factor
 * (n_res == 1: top -> R + X, R -> ab) and every wave is its OWN BW slot —
 * NO bw_order indirection.  The wrapper expands the per-resonance arrays
 * (m0_index / mass_index / gamma rows) to per-wave, so all BW arrays are
 * indexed by the wave index directly: contiguous reads/writes, no product
 * loop, no atomics (each wave owns its slot), no cross-wave gather.
 * Values equal cuda_v4_pwa at n_res == 1; the m0/g0 gradients differ from
 * the shared-slot layout only in host-side summation order (~1e-16).
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
 * NOTE on g0: the fitted `g0` parameter is the width at the reference
 * defined by the resonance mass FIXED in the config particle table (which
 * builds the running-width gamma_table/fl_table).  It is not the nominal
 * width at the `m0` fit parameter — when the fitted mass differs from that
 * fixed reference mass the physical width runs via g0·γ(m).
 *
 * Wave entries are stored p-major: entry index w = p·N + k,  N = n_wave/P.
 * All per-event/per-wave arrays (common_amp, bw_dom, …) therefore have
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
    const double* __restrict__ coeff,         // s0c: [n_wave · n_basis]
    const int* __restrict__ uslot_row,        // s0c: [n_uslot] ext-table row
    const int* __restrict__ uslot_col,        // s0c: [n_uslot] mass column
    const int* __restrict__ slot_map,         // s0c: [n_wave · n_basis] -> uslot
    const int* __restrict__ slot_l,           // s0c: [n_wave · n_basis]
                                              //   merged row (-1: non-gamma)
    const double* __restrict__ gamma_table_real,
    const double* __restrict__ gamma_table_imag,
    double gamma_min, double gamma_inv_delta, int gamma_table_bins,
    int n_gamma_rows, int n_basis, int n_uslot,
    double* __restrict__ g_interp_real, double* __restrict__ g_interp_imag,
    int n_wave, int n_unique_bw,
    int n_mass,
    const float2* __restrict__ amp_cache,
    const int* __restrict__ slot_of_wave,
    int n_uniq,
    double* __restrict__ common_amp_factor_real,
    double* __restrict__ common_amp_factor_imag,
    double* __restrict__ bw_dom_real, double* __restrict__ bw_dom_imag,
    int n_events
) {
    // s0c: TWO-PHASE table-driven matvec.
    // Phase 1 (per event, cooperative): interpolate each UNIQUE
    // (gamma-table row, mass column) pair ONCE into shared scratch —
    // the slot table is massively redundant (waves share gamma rows;
    // the "1" row is constant): fixture 18 interps/event -> 5,
    // production ~1344 -> ~29.
    // Phase 2: the uniform per-wave matvec over the cached basis:
    //   BW_w = Σ_i M_wi · (-i·Z_{slot_map[w,i]})
    //   bw_r += M·Z_i,  bw_i -= M·Z_r      (the -i rotation at read)
    // The "1" row stores (0,1) and "m²" (0, m_j²) (pre-rotated;
    // Catmull-Rom reproduces quadratics exactly).  Zero-padded slots
    // have M = 0.  Same math as s0b; order differs at ~1e-16.
    int event_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_sz = blockDim.x;

    extern __shared__ double s_dyn_bw2[];
    double* s_mass = s_dyn_bw2;
    double* s_us_r = s_dyn_bw2 + n_mass;          // [n_uslot]
    double* s_us_i = s_dyn_bw2 + n_mass + n_uslot;
    for (int i = tid; i < n_mass; i += blockDim.x)
        s_mass[i] = mass[event_idx * n_mass + i];
    for (int u = tid; u < n_uslot; u += blockDim.x) {
        complex z = interp_complex_device(
            gamma_table_real, gamma_table_imag, uslot_row[u],
            s_mass[uslot_col[u]],
            gamma_min, gamma_inv_delta, gamma_table_bins);
        s_us_r[u] = z.real();
        s_us_i[u] = z.imag();
    }
    __syncthreads();

    for (int wave_idx = tid; wave_idx < n_wave; wave_idx += block_sz) {
        int bw_idx = wave_idx;
        double bw_r = 0.0, bw_i = 0.0;
        int bo = wave_idx * n_basis;
        for (int i = 0; i < n_basis; i++) {
            int u = slot_map[bo + i];
            double M = coeff[bo + i];
            bw_r += M * s_us_i[u];               // -i·Z: Re = +Z_i
            bw_i -= M * s_us_r[u];               //       Im = -Z_r
        }
        bw_dom_real[(size_t)event_idx * n_unique_bw + bw_idx] = bw_r;
        bw_dom_imag[(size_t)event_idx * n_unique_bw + bw_idx] = bw_i;

        float2 A = amp_cache[event_idx * n_uniq + slot_of_wave[wave_idx]];
        double nrm = bw_r * bw_r + bw_i * bw_i;
        common_amp_factor_real[event_idx * n_wave + wave_idx] =
            ((double)A.x * bw_r + (double)A.y * bw_i) / nrm;
        common_amp_factor_imag[event_idx * n_wave + wave_idx] =
            ((double)A.y * bw_r - (double)A.x * bw_i) / nrm;

        // g_interp for the gradient kernel (per merged row, from the
        // deduped values)
        for (int i = 2; i < n_basis; i++) {
            int l = slot_l[bo + i];
            if (l >= 0) {
                int u = slot_map[bo + i];
                g_interp_real[event_idx * n_gamma_rows + l] = s_us_r[u];
                g_interp_imag[event_idx * n_gamma_rows + l] = s_us_i[u];
            }
        }
    }
}

//=============================================================================
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
    double* __restrict__ dnorm_out,
    double* __restrict__ dQ_dA_r, double* __restrict__ dQ_dA_i
) {
    // s0b: MULTI-EVENT scheduling.  The per-event work is tiny (N-term
    // dot products), so one 256-thread block per event wastes the GPU
    // on block scheduling and __syncthreads latency (measured: K3 runs
    // ~200x above its traffic bound).  Small n_base: ONE THREAD per
    // event (serial sum, no shared memory, no barriers).  Large n_base:
    // one WARP per event (strided loads + shuffle reduction).  Both
    // grid-stride over events; A_p is recomputed in a second pass for
    // the dQ/dA write (no per-projection storage needed).
    int n_base = n_wave / n_proj;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gstride = gridDim.x * blockDim.x;

    if (n_base <= 32) {
        for (int event_idx = gtid; event_idx < n_events; event_idx += gstride) {
            const double* cr = common_r + (size_t)event_idx * n_wave;
            const double* ci = common_i + (size_t)event_idx * n_wave;
            double P = 0.0;
            for (int p = 0; p < n_proj; p++) {
                double ar = 0.0, ai = 0.0;
                for (int k = 0; k < n_base; k++) {
                    int w = p * n_base + k;
                    double ckr = ck_real[k], cki = ck_imag[k];
                    double cmr = cr[w], cmi = ci[w];
                    ar += ckr * cmr - cki * cmi;
                    ai += ckr * cmi + cki * cmr;
                }
                P += ar * ar + ai * ai;
            }
            P_out[event_idx] = P;
            double wval = weight[event_idx];
            double bkg_val = bkg[event_idx];
            double dQ_dP;
            if (use_norm == 0) {
                Q_out[event_idx] = wval * P;
                dQ_dP = wval;
                dnorm_out[event_idx] = 0.0;
            } else {
                Q_out[event_idx] = -wval * log(P / norm + bkg_val);
                dQ_dP = -wval / (P + bkg_val * norm);
                dnorm_out[event_idx] = wval * P / (norm * (P + bkg_val * norm));
            }
            size_t ab = (size_t)event_idx * n_proj;
            for (int p = 0; p < n_proj; p++) {
                double ar = 0.0, ai = 0.0;
                for (int k = 0; k < n_base; k++) {
                    int w = p * n_base + k;
                    double ckr = ck_real[k], cki = ck_imag[k];
                    double cmr = cr[w], cmi = ci[w];
                    ar += ckr * cmr - cki * cmi;
                    ai += ckr * cmi + cki * cmr;
                }
                dQ_dA_r[ab + p] = dQ_dP * ar;
                dQ_dA_i[ab + p] = -dQ_dP * ai;
            }
        }
        return;
    }

    // ── large n_base: one warp per event, shuffle reduction ─────────
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int e0 = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const int estride = gridDim.x * warps_per_block;
    for (int event_idx = e0; event_idx < n_events; event_idx += estride) {
        const double* cr = common_r + (size_t)event_idx * n_wave;
        const double* ci = common_i + (size_t)event_idx * n_wave;
        double P = 0.0;
        for (int p = 0; p < n_proj; p++) {
            double ar = 0.0, ai = 0.0;
            for (int k = lane; k < n_base; k += 32) {
                int w = p * n_base + k;
                double ckr = ck_real[k], cki = ck_imag[k];
                double cmr = cr[w], cmi = ci[w];
                ar += ckr * cmr - cki * cmi;
                ai += ckr * cmi + cki * cmr;
            }
            for (int off = 16; off > 0; off >>= 1) {
                ar += __shfl_down_sync(0xffffffffu, ar, off);
                ai += __shfl_down_sync(0xffffffffu, ai, off);
            }
            if (lane == 0) P += ar * ar + ai * ai;
        }
        if (lane == 0) {
            P_out[event_idx] = P;
            double wval = weight[event_idx];
            double bkg_val = bkg[event_idx];
            double dQ_dP;
            if (use_norm == 0) {
                Q_out[event_idx] = wval * P;
                dQ_dP = wval;
                dnorm_out[event_idx] = 0.0;
            } else {
                Q_out[event_idx] = -wval * log(P / norm + bkg_val);
                dQ_dP = -wval / (P + bkg_val * norm);
                dnorm_out[event_idx] = wval * P / (norm * (P + bkg_val * norm));
            }
            size_t ab = (size_t)event_idx * n_proj;
            for (int p = 0; p < n_proj; p++) {
                double ar = 0.0, ai = 0.0;
                for (int k = lane; k < n_base; k += 32) {
                    int w = p * n_base + k;
                    double ckr = ck_real[k], cki = ck_imag[k];
                    double cmr = cr[w], cmi = ci[w];
                    ar += ckr * cmr - cki * cmi;
                    ai += ckr * cmi + cki * cmr;
                }
                for (int off = 16; off > 0; off >>= 1) {
                    ar += __shfl_down_sync(0xffffffffu, ar, off);
                    ai += __shfl_down_sync(0xffffffffu, ai, off);
                }
                if (lane == 0) {
                    dQ_dA_r[ab + p] = dQ_dP * ar;
                    dQ_dA_i[ab + p] = -dQ_dP * ai;
                }
            }
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
    int n_wave, int n_proj,
    int n_events
) {
    // s0c: MULTI-EVENT scheduling (as K3).  grad_ck_partial[e,k] =
    // Σ_p dQ_dA[e,p]·common[e,p·N+k].  Small n_base: ONE THREAD per
    // event (serial k-loop, immediate write — no shared memory, no
    // atomics).  Larger n_base: one WARP per event, lane = k.
    int n_base = n_wave / n_proj;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gstride = gridDim.x * blockDim.x;

    if (n_base <= 8) {
        for (int event_idx = gtid; event_idx < n_events; event_idx += gstride) {
            const double* cr = common_r + (size_t)event_idx * n_wave;
            const double* ci = common_i + (size_t)event_idx * n_wave;
            const double* dar = dQ_dA_r + (size_t)event_idx * n_proj;
            const double* dai = dQ_dA_i + (size_t)event_idx * n_proj;
            size_t base = (size_t)event_idx * n_base;
            for (int k = 0; k < n_base; k++) {
                double sr = 0.0, si = 0.0;
                for (int p = 0; p < n_proj; p++) {
                    int w = p * n_base + k;
                    double a_r = dar[p], a_i = dai[p];
                    double cmr = cr[w], cmi = ci[w];
                    sr += a_r * cmr - a_i * cmi;
                    si += a_r * cmi + a_i * cmr;
                }
                grad_ck_real_partial[base + k] = sr;
                grad_ck_imag_partial[base + k] = si;
            }
        }
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int e0 = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const int estride = gridDim.x * warps_per_block;
    for (int event_idx = e0; event_idx < n_events; event_idx += estride) {
        const double* cr = common_r + (size_t)event_idx * n_wave;
        const double* ci = common_i + (size_t)event_idx * n_wave;
        const double* dar = dQ_dA_r + (size_t)event_idx * n_proj;
        const double* dai = dQ_dA_i + (size_t)event_idx * n_proj;
        size_t base = (size_t)event_idx * n_base;
        for (int k = lane; k < n_base; k += 32) {
            double sr = 0.0, si = 0.0;
            for (int p = 0; p < n_proj; p++) {
                int w = p * n_base + k;
                double a_r = dar[p], a_i = dai[p];
                double cmr = cr[w], cmi = ci[w];
                sr += a_r * cmr - a_i * cmi;
                si += a_r * cmi + a_i * cmr;
            }
            grad_ck_real_partial[base + k] = sr;
            grad_ck_imag_partial[base + k] = si;
        }
    }
}

//=============================================================================
// KERNEL 5: s0 gradient (per wave) + dQ_dbw_dom (for grad_mult).
// ONE BW per wave: dQ/dbw_dom = dQ_da · (-ck_k · common_w / bw_w).
// dQ/ds0_w = -2·dQ_dbw_dom_re   (d(bw)/d(s0) = -1).
//=============================================================================
// s0c: FUSED unified-coefficient gradient (former K5 + K6).
// The denominator is the uniform basis sum
//   BW_w = Σ_i M_wi·F_i,   F_0 = 1,  F_1 = m²,  F_l = -i·G_l
// so every fitted coefficient shares ONE gradient formula
//   dQ/dM_wi = 2·Re(dQ/dBW_w · F_i(e)).
// One kernel computes the per-(e,w) chain dQ/dBW once (in registers —
// the former dQ_dbw_dom global round-trip is gone) and reduces against
// every basis function in the same pass:
//   F_0 = 1        -> grad_s0_partial[e,w]   = 2·ddbr
//   F_1 = m²       -> fixed coefficient, no gradient
//   F_l = -i·G_l   -> grad_mult_partial[e,l] = 2·(ddbr·G_i + dbbi·G_r)
// MULTI-EVENT scheduling (as K3/K4): small n_wave -> one thread per
// event; larger -> one warp per event, lane = wave (each wave's chain
// and row loop is independent — no cross-lane reduction).
__global__ void grad_coeff_kernel(
    const double* __restrict__ common_amp_factor_real,
    const double* __restrict__ common_amp_factor_imag,
    const double* __restrict__ dQ_dA_r, const double* __restrict__ dQ_dA_i,
    const double* __restrict__ bw_dom_real, const double* __restrict__ bw_dom_imag,
    const double* __restrict__ ck_real, const double* __restrict__ ck_imag,
    const double* __restrict__ g_interp_real, const double* __restrict__ g_interp_imag,
    const int* __restrict__ row_start,
    int n_wave, int n_unique_bw, int n_proj, int n_gamma_rows,
    double* __restrict__ grad_s0_partial,
    double* __restrict__ grad_mult_partial,
    int n_events
) {
    int n_base = n_wave / n_proj;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gstride = gridDim.x * blockDim.x;

    if (n_wave <= 32) {
        for (int event_idx = gtid; event_idx < n_events; event_idx += gstride) {
            const double* dar = dQ_dA_r + (size_t)event_idx * n_proj;
            const double* dai = dQ_dA_i + (size_t)event_idx * n_proj;
            const double* gr = g_interp_real + (size_t)event_idx * n_gamma_rows;
            const double* gi = g_interp_imag + (size_t)event_idx * n_gamma_rows;
            size_t bo = (size_t)event_idx * n_unique_bw;
            size_t co = (size_t)event_idx * n_wave;
            size_t mo = (size_t)event_idx * n_gamma_rows;
            for (int wave_idx = 0; wave_idx < n_wave; wave_idx++) {
                int p = wave_idx / n_base;
                int k = wave_idx - p * n_base;
                double bdr = bw_dom_real[bo + wave_idx];
                double bdi = bw_dom_imag[bo + wave_idx];
                double car = common_amp_factor_real[co + wave_idx];
                double cai = common_amp_factor_imag[co + wave_idx];
                double dqa_r = dar[p], dqa_i = dai[p];
                double ckr = ck_real[k], cki = ck_imag[k];
                double bpn = bdr * bdr + bdi * bdi;
                double obw_r = bdr / bpn;
                double obw_i = -bdi / bpn;
                double ck_obw_r = ckr * obw_r - cki * obw_i;
                double ck_obw_i = ckr * obw_i + cki * obw_r;
                double neg_mul_r = -(ck_obw_r * car - ck_obw_i * cai);
                double neg_mul_i = -(ck_obw_r * cai + ck_obw_i * car);
                double ddbr = dqa_r * neg_mul_r - dqa_i * neg_mul_i;
                double dbbi = dqa_r * neg_mul_i + dqa_i * neg_mul_r;
                // F_0 = 1
                grad_s0_partial[bo + wave_idx] = 2.0 * ddbr;
                // F_l = -i·G_l  (this wave's rows)
                int r0 = row_start[wave_idx];
                int r1 = (wave_idx + 1 < n_wave)
                             ? row_start[wave_idx + 1] : n_gamma_rows;
                for (int l = r0; l < r1; l++)
                    grad_mult_partial[mo + l] =
                        2.0 * (ddbr * gi[l] + dbbi * gr[l]);
            }
        }
        return;
    }
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int e0 = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const int estride = gridDim.x * warps_per_block;
    for (int event_idx = e0; event_idx < n_events; event_idx += estride) {
        const double* dar = dQ_dA_r + (size_t)event_idx * n_proj;
        const double* dai = dQ_dA_i + (size_t)event_idx * n_proj;
        const double* gr = g_interp_real + (size_t)event_idx * n_gamma_rows;
        const double* gi = g_interp_imag + (size_t)event_idx * n_gamma_rows;
        size_t bo = (size_t)event_idx * n_unique_bw;
        size_t co = (size_t)event_idx * n_wave;
        size_t mo = (size_t)event_idx * n_gamma_rows;
        for (int wave_idx = lane; wave_idx < n_wave; wave_idx += 32) {
            int p = wave_idx / n_base;
            int k = wave_idx - p * n_base;
            double bdr = bw_dom_real[bo + wave_idx];
            double bdi = bw_dom_imag[bo + wave_idx];
            double car = common_amp_factor_real[co + wave_idx];
            double cai = common_amp_factor_imag[co + wave_idx];
            double dqa_r = dar[p], dqa_i = dai[p];
            double ckr = ck_real[k], cki = ck_imag[k];
            double bpn = bdr * bdr + bdi * bdi;
            double obw_r = bdr / bpn;
            double obw_i = -bdi / bpn;
            double ck_obw_r = ckr * obw_r - cki * obw_i;
            double ck_obw_i = ckr * obw_i + cki * obw_r;
            double neg_mul_r = -(ck_obw_r * car - ck_obw_i * cai);
            double neg_mul_i = -(ck_obw_r * cai + ck_obw_i * car);
            double ddbr = dqa_r * neg_mul_r - dqa_i * neg_mul_i;
            double dbbi = dqa_r * neg_mul_i + dqa_i * neg_mul_r;
            grad_s0_partial[bo + wave_idx] = 2.0 * ddbr;
            int r0 = row_start[wave_idx];
            int r1 = (wave_idx + 1 < n_wave)
                         ? row_start[wave_idx + 1] : n_gamma_rows;
            for (int l = r0; l < r1; l++)
                grad_mult_partial[mo + l] =
                    2.0 * (ddbr * gi[l] + dbbi * gr[l]);
        }
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

// s0b: accumulate variant — adds the per-batch sum into a PERSISTENT
// accumulator (one block per feature -> no cross-block race), so the
// gradient partials never leave the device until the end of the call.
__global__ void reduce_sum_features_acc_kernel(
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
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) sdata[tid] += sdata[tid + st];
        __syncthreads();
    }
    if (tid == 0) output[feat] += sdata[0];
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
    double* Q_out; double* P_out; double* dnorm_out;
    double* common_amp_factor_real; double* common_amp_factor_imag;
    double* dQ_dA_real; double* dQ_dA_imag;        // [bs · n_proj]
    double* grad_ck_real_partial; double* grad_ck_imag_partial;  // [bs · N]
    double* dQ_dbw_dom_real; double* dQ_dbw_dom_imag;  // [bs · n_unique_bw]
    double* grad_s0_partial; double* grad_mult_partial;
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
    const int* row_start;                   // per-wave row range into merged gamma rows
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
    double* acc_buf;   // s0b: persistent output accumulators
                       // [Q, dnorm, gck_r(N), gck_i(N), gs0(nw), gmult(ng)]
    // s0b: persistent PARAMETER buffers — refreshed per call with a
    // plain memcpy (no cudaMalloc/cudaFree per iteration; cudaFree
    // forces a device-wide sync and dominated the fixed call overhead)
    double* p_ck_r; double* p_ck_i;     // [N]
    double* p_coeff;                    // s0c: [n_wave · n_basis]
    // s0b: PINNED host staging for the param upload — pageable sync H2D
    // costs ~0.9 ms/call on this driver; pinned + cudaMemcpyAsync ~10 us
    double* h_ck_r; double* h_ck_i;
    double* h_coeff;                    // s0c: pinned [n_wave · n_basis]
    // s0b: dedicated NON-LEGACY stream — the legacy default stream's
    // blocking copies carry ~1-2 ms of fixed driver latency on this
    // system (measured: sync H2D 0.9 ms, blocking D2H 2.2 ms, both
    // size-independent).  All per-iteration work runs on this stream;
    // the one output sync per call is stream-scoped and cheap.
    cudaStream_t stream;
    double* acc_host;                   // pinned [acc_total] output staging
    // s0c: uniform basis layout — n_basis slots per wave (2 + max rows);
    // every slot (incl. "1" and "m²") is a gamma-table row via
    // slot_row/slot_mass; the table carries 2 extra pre-rotated rows
    int n_basis;
    // s0c: deduped interp tables — one unique (table row, mass column)
    // per entry; slot_map routes each (wave, slot) to its unique interp
    int n_uslot;
    const int* uslot_row;               // [n_uslot]
    const int* uslot_col;               // [n_uslot]
    const int* slot_map;                // [n_wave · n_basis] -> uslot
    const int* slot_l;                  // [n_wave · n_basis] (-1: non-gamma)
    int n_gamma_rows_ext;               // n_gamma_rows + 2 (uploaded table)
    int gamma_table_bins_ext;
} ComputeContext;

typedef struct {
    const double* ck_real; const double* ck_imag;
    const double* coeff;                // s0c: [n_wave · n_basis]
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
void cuda_s0b_set_flags() {
    // s0b: busy-wait synchronization — the default (interrupt/blocking)
    // sync costs ~1-2 ms per cudaStreamSynchronize on this driver
    // (interrupt coalescing), dominating the per-call overhead.  Must be
    // called BEFORE the primary context is created (first CUDA call in
    // the process); silently ignored otherwise.
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);
}

cudaError_t cuda_get_device_name(char* name, int len) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err == cudaSuccess) strncpy(name, prop.name, len);
    return err;
}

//── g_bw launch wrapper ────────────────────────────────────────────

// Reduction launch wrappers
void launch_reduce_sum(const double* input, double* output, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_sum_kernel<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(input, output, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_reduce_sum_stream(const double* input, double* output, int n,
                              cudaStream_t stream) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_sum_kernel<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(double),
                        stream>>>(input, output, n);
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

void launch_reduce_sum_features_acc(const double* input, double* output,
    int n_events, int n_features) {
    reduce_sum_features_acc_kernel<<<n_features, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(
        input, output, n_events, n_features);
    CUDA_CHECK(cudaGetLastError());
}

// s0b: COALESCED chunked features-reduce.  The per-feature layout
// [e·n_features + f] makes a fixed-feature scan stride n_features·8 B
// (every read its own sector) at n_features-block parallelism — measured
// ~10 ms at 300k events.  Instead (n_features <= 32): one block per
// EVENT CHUNK; consecutive threads read consecutive flat elements
// (perfectly coalesced) and accumulate into PRIVATE per-thread shared
// slots [f·blockDim + tid] — no atomics during accumulation — then a
// barrier tree reduces all features simultaneously and one global
// atomicAdd per feature lands in the persistent accumulator.
// n_features > 32 keeps the per-feature-block accumulate kernel (the
// shared tile would exceed the 48 KB limit; large-nf tiling is future
// work).
__global__ void reduce_sum_features_tree_kernel(
    const double* __restrict__ input,
    double* __restrict__ output,
    int n_events, int n_features, int chunk
) {
    extern __shared__ double sacc[];   // [f·blockDim.x + tid]
    const int bd = blockDim.x;
    const int tid = threadIdx.x;
    const int c0 = blockIdx.x * chunk;
    int c1 = c0 + chunk;
    if (c1 > n_events) c1 = n_events;
    for (int f = 0; f < n_features; f++) sacc[f * bd + tid] = 0.0;

    const long long base = (long long)c0 * n_features;
    const int total = (c1 - c0) * n_features;
    for (int idx = tid; idx < total; idx += bd) {
        int f = (int)(idx % n_features);
        sacc[f * bd + tid] += input[base + idx];
    }
    __syncthreads();
    for (int st = bd / 2; st > 0; st >>= 1) {
        if (tid < st) {
            for (int f = 0; f < n_features; f++)
                sacc[f * bd + tid] += sacc[f * bd + tid + st];
        }
        __syncthreads();
    }
    if (tid < n_features) atomicAdd(&output[tid], sacc[tid * bd]);
}

void launch_reduce_sum_features_chunked(const double* input, double* output,
    int n_events, int n_features, cudaStream_t stream) {
    if (n_features <= 32) {
        int chunk = 1024;
        int n_chunks = (n_events + chunk - 1) / chunk;
        if (n_chunks < 1) n_chunks = 1;
        reduce_sum_features_tree_kernel<<<n_chunks, 128,
            n_features * 128 * sizeof(double), stream>>>(
            input, output, n_events, n_features, chunk);
    } else {
        reduce_sum_features_acc_kernel<<<n_features, BLOCK_SIZE,
            BLOCK_SIZE * sizeof(double), stream>>>(
            input, output, n_events, n_features);
    }
    CUDA_CHECK(cudaGetLastError());
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

    //── K2: cached forward — running width + BW fused (per-wave rows);
    //     angular part from the one-time cache
    {
        size_t shmem = (ctx->n_mass + 2 * ctx->n_uslot) * sizeof(double);
        amp_cache_amp_kernel<<<ne, amp_t, shmem, ctx->stream>>>(
            data->mass,
            params->coeff,
            ctx->uslot_row, ctx->uslot_col,
            ctx->slot_map, ctx->slot_l,
            ctx->gamma_table_real, ctx->gamma_table_imag,
            ctx->gamma_min, 1.0 / ctx->gamma_delta, ctx->gamma_table_bins_ext,
            ctx->n_gamma_rows, ctx->n_basis, ctx->n_uslot,
            data->g_interp_real, data->g_interp_imag,
            nw, nu, ctx->n_mass,
            data->amp_cache, ctx->slot_of_wave, ctx->n_uniq,
            data->common_amp_factor_real, data->common_amp_factor_imag,
            data->bw_dom_real, data->bw_dom_imag, ne);
    }
    CUDA_CHECK(cudaGetLastError());

    //── K3: PWA forward (P = Σ_p |A_p|², dQ/dA_p) ────────────────────
    {
        int k3_grid = (ne + BLOCK_SIZE - 1) / BLOCK_SIZE;   // s0b: multi-event
        if (k3_grid > 4096) k3_grid = 4096;
        pwa_amp_reduce_kernel<<<k3_grid, BLOCK_SIZE, 0, ctx->stream>>>(
            data->common_amp_factor_real, data->common_amp_factor_imag,
            params->ck_real, params->ck_imag,
            data->weight, data->bkg,
            nw, P, ne, use_norm, norm,
            data->Q_out, data->P_out, data->dnorm_out,
            data->dQ_dA_real, data->dQ_dA_imag);
    }
    CUDA_CHECK(cudaGetLastError());

    //── K4: ck gradient (projection-reduced partials, length N) ───────
    {
        int k4_grid = (ne + BLOCK_SIZE - 1) / BLOCK_SIZE;  // s0c: multi-event
        if (k4_grid > 4096) k4_grid = 4096;
        grad_ck_kernel_v4<<<k4_grid, BLOCK_SIZE, 0, ctx->stream>>>(
            data->common_amp_factor_real, data->common_amp_factor_imag,
            data->dQ_dA_real, data->dQ_dA_imag,
            data->grad_ck_real_partial, data->grad_ck_imag_partial,
            nw, P, ne);
    }
    CUDA_CHECK(cudaGetLastError());

    //── K5: bw_dom / m0 gradients ────────────────────────────────────
    // s0c: FUSED unified-coefficient gradient (former K5 + K6)
    {
        int k56_grid = (ne + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (k56_grid > 4096) k56_grid = 4096;
        grad_coeff_kernel<<<k56_grid, BLOCK_SIZE, 0, ctx->stream>>>(
            data->common_amp_factor_real, data->common_amp_factor_imag,
            data->dQ_dA_real, data->dQ_dA_imag,
            data->bw_dom_real, data->bw_dom_imag,
            params->ck_real, params->ck_imag,
            data->g_interp_real, data->g_interp_imag,
            ctx->row_start,
            nw, nu, P, ng,
            data->grad_s0_partial, data->grad_mult_partial, ne);
    }
    CUDA_CHECK(cudaGetLastError());

    //── K6: g0 gradient (sparse gather) ───────────────────────────────
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
    double* h_P;          // s0b: pinned per-event P staging [ne]
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
    int n_uniq, int n_proj,
    const int* row_start,int n_rs,
    int n_basis,
    const int* uslot_row,int n_ur,
    const int* uslot_col,int n_uc,
    const int* slot_map,int n_sm2,
    const int* slot_l,int n_sl,
    int n_gamma_rows_ext,int gamma_table_bins_ext
) {
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
    // s0c: the uploaded table is the EXTENDED one (2 extra rows); the
    // kernels index rows [0, n_gamma_rows_ext) — the original row count
    // stays in n_gamma_rows for the gradient buffers
    c->n_gamma_rows_ext = n_gamma_rows_ext;
    c->gamma_table_bins_ext = gamma_table_bins_ext;

    c->matrix_gamma = (n16 > 0) ? (double*)_up_dbl(mg, n16) : NULL;  // unused (per-wave rows)
    c->gamma_col_idx = (int*)_up_int(gci, ngci);
    c->fl_table = (double*)_up_dbl(ft, n17); c->fl_min = flmin; c->fl_delta = fldel; c->fl_table_bins = fbins;
    c->n_wave = nw; c->n_res = nr; c->n_decay = nd;
    c->n_unique_bw = nub; c->n_gamma_rows = ngr;
    c->n_mass = nm; c->n_momentum = nmom; c->n_angle_k = nak_; c->n_angle_total = nat; c->n_angle_comp = nac;
    c->n_m0_params = n_m0p; c->n_g0_params = n_g0p;
    c->batch_size = batch_size > 0 ? batch_size : DEFAULT_BATCH_SIZE;
    c->slot_of_wave = (n_slot > 0) ? (int*)_up_int(slot_of_wave, n_slot) : NULL;
    c->rep_of_slot = (n_rep > 0) ? (int*)_up_int(rep_of_slot, n_rep) : NULL;
    c->row_start = (n_rs > 0) ? (int*)_up_int(row_start, n_rs) : NULL;
    c->n_basis = n_basis > 0 ? n_basis : 1;   // s0c
    c->n_uslot = n_ur;
    c->uslot_row = (int*)_up_int(uslot_row, n_ur);
    c->uslot_col = (int*)_up_int(uslot_col, n_uc);
    c->slot_map = (int*)_up_int(slot_map, n_sm2);
    c->slot_l = (int*)_up_int(slot_l, n_sl);
    c->n_gamma_rows_ext = n_gamma_rows_ext;
    c->gamma_table_bins_ext = gamma_table_bins_ext;
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
        S(Q_out); S(P_out); S(dnorm_out);
        S2(common_amp_factor_real, nw); S2(common_amp_factor_imag, nw);
        S2(bw_dom_real, nub); S2(bw_dom_imag, nub);
        S2(dQ_dA_real, c->n_proj); S2(dQ_dA_imag, c->n_proj);
        S2(grad_ck_real_partial, N); S2(grad_ck_imag_partial, N);
        S2(grad_s0_partial, nw); S2(grad_mult_partial, ngr);
        S2(dQ_dbw_dom_real, nub); S2(dQ_dbw_dom_imag, nub);
        #undef S
        #undef S2
        CUDA_CHECK(cudaMalloc(&c->Q_red_gpu, 8));
    } else {
        c->scratch = NULL;
        c->Q_red_gpu = NULL;
    }
    // s0b: persistent accumulators (small, independent of batch_size)
    {
        int acc_total = 2 + 2 * (c->n_wave / c->n_proj) + c->n_wave
                            + c->n_gamma_rows;
        CUDA_CHECK(cudaMalloc(&c->acc_buf, acc_total * sizeof(double)));
    }
    // s0b: persistent parameter buffers
    {
        int N = c->n_wave / c->n_proj;
        CUDA_CHECK(cudaMalloc(&c->p_ck_r, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c->p_ck_i, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c->p_coeff,
                              c->n_wave * c->n_basis * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&c->h_ck_r, N * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&c->h_ck_i, N * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&c->h_coeff,
                                  c->n_wave * c->n_basis * sizeof(double)));
        CUDA_CHECK(cudaStreamCreate(&c->stream));
        int acc_total = 2 + 2 * (c->n_wave / c->n_proj) + c->n_wave
                            + c->n_gamma_rows;
        CUDA_CHECK(cudaMallocHost(&c->acc_host, acc_total * sizeof(double)));
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
    F(gamma_table_real); F(gamma_table_imag); F(matrix_gamma); F(gamma_col_idx); F(fl_table); F(row_start);
    if (c->slot_of_wave) cudaFree((void*)c->slot_of_wave);
    if (c->rep_of_slot) cudaFree((void*)c->rep_of_slot);
    #undef F
    if (c->scratch) {
        #define SF(f) cudaFree(c->scratch->f)
        SF(g_interp_real); SF(g_interp_imag); SF(dQ_dbw_dom_real); SF(dQ_dbw_dom_imag); SF(g_bw_real); SF(g_bw_imag);
        SF(Q_out); SF(P_out); SF(dnorm_out);
       
        SF(common_amp_factor_real); SF(common_amp_factor_imag);
        SF(bw_dom_real); SF(bw_dom_imag);
        SF(dQ_dA_real); SF(dQ_dA_imag);
        SF(grad_ck_real_partial); SF(grad_ck_imag_partial);
        SF(grad_s0_partial); SF(grad_mult_partial);
       
        #undef SF
        free(c->scratch);
    }
    if (c->Q_red_gpu) cudaFree(c->Q_red_gpu);
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
    CUDA_CHECK(cudaMallocHost(&h->h_P, (size_t)ne * sizeof(double)));   // s0b

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
    if (h->h_P) cudaFreeHost(h->h_P);
    free(h);
    return NULL;
}

void cuda_free_data_v4(void* vh) {
    if (!vh) return;
    DataHandle2* h = (DataHandle2*)vh;
    cudaFree((void*)h->m);
    cudaFree((void*)h->w); cudaFree((void*)h->b);
    if (h->amp_cache) cudaFree(h->amp_cache);
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

void cuda_compute_v4(void* vctx, void* vdh,
    const double* ck_r,const double* ck_i,
    const double* coeff,
    double nv,int use_norm,
    double* oQ,double* oDn,double* oP,
    double* ogck_r,double* ogck_i,
    double* ogs0,double* ogmult
) {
    ComputeContext* c = (ComputeContext*)vctx;
    DataHandle2* h = (DataHandle2*)vdh;
    int ne = h->ne, bs = c->batch_size;
    if (ne < bs) bs = ne;
    int nbat = (ne + bs - 1) / bs;
    int nw = c->n_wave, nu = c->n_unique_bw, ng = c->n_gamma_rows;
    int N = nw / c->n_proj;

    // s0b: refresh the persistent parameter buffers (plain memcpy —
    // no per-call cudaMalloc/cudaFree)
    ComputeParams p;
    p.ck_real = c->p_ck_r; p.ck_imag = c->p_ck_i;
    p.coeff = c->p_coeff;
    memcpy(c->h_ck_r, ck_r, N * sizeof(double));
    memcpy(c->h_ck_i, ck_i, N * sizeof(double));
    memcpy(c->h_coeff, coeff, c->n_wave * c->n_basis * sizeof(double));
    cudaMemcpyAsync(c->p_ck_r, c->h_ck_r, N * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    cudaMemcpyAsync(c->p_ck_i, c->h_ck_i, N * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    cudaMemcpyAsync(c->p_coeff, c->h_coeff,
                    c->n_wave * c->n_basis * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);


    ComputeData s;
    if (c->scratch) {
        s = *c->scratch;
    } else {
        memset(&s, 0, sizeof(ComputeData));
        #define S(f) CUDA_CHECK(cudaMalloc(&s.f, bs * sizeof(double)))
        #define S2(f,n) CUDA_CHECK(cudaMalloc(&s.f, bs * (n) * sizeof(double)))
        S2(g_interp_real,ng); S2(g_interp_imag,ng);
        S2(g_bw_real,nu); S2(g_bw_imag,nu);
        S(Q_out); S(P_out); S(dnorm_out);
        S2(common_amp_factor_real,nw); S2(common_amp_factor_imag,nw);
        S2(bw_dom_real,nu); S2(bw_dom_imag,nu);
        S2(dQ_dA_real, c->n_proj); S2(dQ_dA_imag, c->n_proj);
        S2(grad_ck_real_partial,N); S2(grad_ck_imag_partial,N);
        S2(grad_s0_partial,nw); S2(grad_mult_partial,ng);
        S2(dQ_dbw_dom_real, nu); S2(dQ_dbw_dom_imag, nu);
        #undef S
        #undef S2
    }

    *oQ = 0; *oDn = 0;
    cudaMemsetAsync(c->acc_buf, 0, (2 + 2 * N + nw + ng) * 8, c->stream);
    if (oP != NULL) memset(oP, 0, ne * 8);
    memset(ogck_r, 0, N * 8); memset(ogck_i, 0, N * 8);
    memset(ogs0, 0, nw * 8); memset(ogmult, 0, ng * 8);


    // s0b: P goes through the handle's pinned staging — no host buffer
    // s0b: Q/dnorm/gradients accumulate on device — no per-batch host bufs
    const int acc_total = 2 + 2 * N + nw + ng;
    const int off_gck_r = 2, off_gck_i = 2 + N;
    const int off_gs0 = 2 + 2 * N, off_gmult = 2 + 2 * N + nw;

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

        // s0b: everything accumulates on device (tree reductions,
        // atomicAdd / +=) — no blocking D2H copies inside the batch loop.
        // Per-event P is still copied out when the caller asks for it.
        if (oP != NULL) {   // s0b: async into the handle's pinned staging
            cudaMemcpyAsync(h->h_P + st, d.P_out, nb * 8,
                            cudaMemcpyDeviceToHost, c->stream);
        }

        launch_reduce_sum_stream(d.Q_out, c->acc_buf + 0, nb, c->stream);
        launch_reduce_sum_stream(d.dnorm_out, c->acc_buf + 1, nb, c->stream);
        launch_reduce_sum_features_chunked(d.grad_ck_real_partial,
                                           c->acc_buf + off_gck_r, nb, N,
                                           c->stream);
        launch_reduce_sum_features_chunked(d.grad_ck_imag_partial,
                                           c->acc_buf + off_gck_i, nb, N,
                                           c->stream);
        launch_reduce_sum_features_chunked(d.grad_s0_partial,
                                           c->acc_buf + off_gs0, nb, nw,
                                           c->stream);
        launch_reduce_sum_features_chunked(d.grad_mult_partial,
                                           c->acc_buf + off_gmult, nb, ng,
                                           c->stream);
    }

    // s0b: ONE stream-scoped sync per call; the outputs were collected
    // asynchronously into pinned staging
    {
        cudaMemcpyAsync(c->acc_host, c->acc_buf, acc_total * 8,
                        cudaMemcpyDeviceToHost, c->stream);
        cudaStreamSynchronize(c->stream);
        *oQ += c->acc_host[0];
        *oDn += c->acc_host[1];
        for (int j = 0; j < N; j++) ogck_r[j] += c->acc_host[off_gck_r + j];
        for (int j = 0; j < N; j++) ogck_i[j] += c->acc_host[off_gck_i + j];
        for (int j = 0; j < nw; j++) ogs0[j] += c->acc_host[off_gs0 + j];
        for (int j = 0; j < ng; j++) ogmult[j] += c->acc_host[off_gmult + j];
        if (oP != NULL) memcpy(oP, h->h_P, ne * 8);
    }
    if (!c->scratch) {
        #define F(p) do { if(s.p) cudaFree(s.p); } while(0)
        F(g_interp_real); F(g_interp_imag); F(dQ_dbw_dom_real); F(dQ_dbw_dom_imag); F(g_bw_real); F(g_bw_imag);
        F(Q_out); F(P_out); F(dnorm_out);
       
        F(common_amp_factor_real); F(common_amp_factor_imag);
        F(bw_dom_real); F(bw_dom_imag);
        F(dQ_dA_real); F(dQ_dA_imag);
        F(grad_ck_real_partial); F(grad_ck_imag_partial);
        F(grad_s0_partial); F(grad_mult_partial);
       
        #undef F
    }


}


//=============================================================================
// Wave Gram matrix D (N,N) for a loaded phsp handle, batched over events
// (batch_size slices; per-batch scratch, device N×N accumulator).
//   D[k,j] = Σ_e w_e Σ_p conj(common[e,pN+k])·common[e,pN+j]
//=============================================================================
void cuda_gram_matrix_v4(void* vctx, void* vdh,
    const double* coeff,
    double* oDr, double* oDi
) {
    ComputeContext* c = (ComputeContext*)vctx;
    DataHandle2* h = (DataHandle2*)vdh;
    int ne = h->ne, bs = c->batch_size;
    if (ne < bs) bs = ne;
    int nbat = (ne + bs - 1) / bs;
    int nw = c->n_wave, nu = c->n_unique_bw, ng = c->n_gamma_rows;
    int P = c->n_proj, N = nw / P;

    ComputeParams p;
    p.ck_real = NULL; p.ck_imag = NULL;
    p.coeff = c->p_coeff;   // s0b: persistent buffers
    memcpy(c->h_coeff, coeff, c->n_wave * c->n_basis * sizeof(double));
    cudaMemcpyAsync(c->p_coeff, c->h_coeff,
                    c->n_wave * c->n_basis * sizeof(double),
                    cudaMemcpyHostToDevice);


    ComputeData s;
    if (c->scratch) {
        s = *c->scratch;
    } else {
        memset(&s, 0, sizeof(ComputeData));
        #define S(f) CUDA_CHECK(cudaMalloc(&s.f, bs * sizeof(double)))
        #define S2(f,n) CUDA_CHECK(cudaMalloc(&s.f, bs * (n) * sizeof(double)))
        S2(g_interp_real,ng); S2(g_interp_imag,ng);
        S2(g_bw_real,nu); S2(g_bw_imag,nu);
        S(Q_out); S(P_out); S(dnorm_out);
        S2(common_amp_factor_real,nw); S2(common_amp_factor_imag,nw);
        S2(bw_dom_real,nu); S2(bw_dom_imag,nu);
        S2(dQ_dA_real, P); S2(dQ_dA_imag, P);
        S2(grad_ck_real_partial,N); S2(grad_ck_imag_partial,N);
        S2(grad_s0_partial,nw); S2(grad_mult_partial,ng);
        S2(dQ_dbw_dom_real, nu); S2(dQ_dbw_dom_imag, nu);
        #undef S
        #undef S2
    }

    double* Dr = NULL; double* Di = NULL;
    size_t gsz = (size_t)N * N * sizeof(double);
    CUDA_CHECK(cudaMalloc(&Dr, gsz));
    CUDA_CHECK(cudaMalloc(&Di, gsz));
    CUDA_CHECK(cudaMemset(Dr, 0, gsz));
    CUDA_CHECK(cudaMemset(Di, 0, gsz));

    for (int b = 0; b < nbat; b++) {
        int st = b * bs;
        int nb = (ne - st > bs) ? bs : (ne - st);
        ComputeData d = s;
        d.mass = h->m + (size_t)st * h->nm;
        d.weight = h->w + st;
        d.bkg = h->b + st;
        d.n_events = nb;
        d.amp_cache = (h->amp_cache != NULL)
            ? (h->amp_cache + (size_t)st * c->n_uniq) : NULL;

        cudaMemset(d.g_bw_real, 0, bs * nu * 8);
        int amp_t = nw < MAX_THREADS ? nw : MAX_THREADS;
        if (amp_t < 32) amp_t = 32;
        size_t shmem = (size_t)(c->n_mass + 2 * c->n_uslot) * sizeof(double);
        amp_cache_amp_kernel<<<nb, amp_t, shmem>>>(
            d.mass,
            p.coeff,
            c->uslot_row, c->uslot_col,
            c->slot_map, c->slot_l,
            c->gamma_table_real, c->gamma_table_imag,
            c->gamma_min, 1.0 / c->gamma_delta, c->gamma_table_bins_ext,
            c->n_gamma_rows, c->n_basis, c->n_uslot,
            d.g_interp_real, d.g_interp_imag,
            nw, nu, c->n_mass,
            d.amp_cache, c->slot_of_wave, c->n_uniq,
            d.common_amp_factor_real, d.common_amp_factor_imag,
            d.bw_dom_real, d.bw_dom_imag, nb);
        CUDA_CHECK(cudaGetLastError());

        gram_v4_event_kernel<<<nb, 128>>>(
            d.common_amp_factor_real, d.common_amp_factor_imag,
            d.weight, nb, P, N, Dr, Di);
        CUDA_CHECK(cudaGetLastError());
    }

    cudaMemcpy(oDr, Dr, gsz, cudaMemcpyDeviceToHost);
    cudaMemcpy(oDi, Di, gsz, cudaMemcpyDeviceToHost);

    cudaFree(Dr); cudaFree(Di);
    if (!c->scratch) {
        #define F(p) do { if(s.p) cudaFree(s.p); } while(0)
        F(g_interp_real); F(g_interp_imag); F(dQ_dbw_dom_real); F(dQ_dbw_dom_imag); F(g_bw_real); F(g_bw_imag);
        F(Q_out); F(P_out); F(dnorm_out);
       
        F(common_amp_factor_real); F(common_amp_factor_imag);
        F(bw_dom_real); F(bw_dom_imag);
        F(dQ_dA_real); F(dQ_dA_imag);
        F(grad_ck_real_partial); F(grad_ck_imag_partial);
        F(grad_s0_partial); F(grad_mult_partial);
       
        #undef F
    }
}

} // extern "C"
