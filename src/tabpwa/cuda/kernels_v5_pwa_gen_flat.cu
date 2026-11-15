/**
 * cuda_v5_pwa_gen_flat — case-1 copy of cuda_v5_pwa_gen_s0e whose
 * denominator basis is a **flat, unpadded, per-(wave, factor) contiguous
 * range** (CSR-style) instead of the uniform ``n_basis`` slots per wave,
 * for **any number of Breit–Wigner factors per wave** (``n_res >= 1``):
 *
 *     bw_{w,r}(m) = Σ_{idx = off[f]}^{off[f+1]-1} coeff[idx] · T[g_idx[idx]](m)
 *     BW_w(m)     = Π_r bw_{w,r}(m),                f = w·n_res + r
 *     common_w    = amp_w / BW_w
 *     dQ/dcoeff[idx] = Σ_e (dQ/dBW_w) · Π_{r'≠r} bw_{w,r'} · T[g_idx[idx]](m_e)
 *
 * ``off`` is int32 ``[n_fact + 1]`` with ``n_fact = n_wave · n_res``,
 * ``g_idx`` int32 ``[n_basis_tot]`` and ``coeff``/``dcoeff`` are float64
 * ``[n_basis_tot]`` with ``n_basis_tot = off[n_fact] = Σ_f n_basis_f`` — no
 * padding, so a mixed decay (a 3-row resonance next to a 5-row one) carries
 * ``n_basis_tot < n_fact · n_basis_max``.  ``T`` is the combined basis table
 * this context owns (raw gamma rows + the two pre-rotated rows "1" / "m²" +
 * the ``den_table`` rows, exactly as in cuda_v5_pwa_gen_s0e); ``g_idx[idx]``
 * is the deduped ``(table row, mass column)`` interp slot (``uslot``), the
 * flat analogue of the padded ``slot_map``.
 *
 * The multi-factor gradient is the REFERENCE implementation
 * (``tabpwa/cpu/_pwa_gen.py``) verbatim: with ``d = dQ/dcommon_w`` and
 * ``dBW = −d·common_w``,
 *
 *     dQ/dcoeff[(w,r,i)] = 2·Re( dBW·(1/bw_{w,r})·F_i )
 *
 * and the ``1/bw_{w,r}`` is written as ``other_{w,r} / BW_w`` with
 * ``other_{w,r} = Π_{r'≠r} bw_{w,r'}`` — a prefix·suffix product over the
 * (1–3) factors, never a division by a factor that can vanish.  At
 * ``n_res == 1`` the empty product is exactly ``1`` and the code reduces to
 * the single-factor form (bit-identical: the factor value is the BW, the
 * product is a multiply by 1+0i, and the coefficient gradient is the same
 * negated expression in the same accumulation order).
 *
 * Everything else is inherited verbatim from cuda_v5_pwa_gen_s0e:
 * cuda_v5_pwa_s0e (fused single-kernel tile + the cuda_v5 GROUPED LOG-SUM
 * data NLL) extended to the ``pwa_gen`` basis, with the GROUP-TOTAL-WEIGHT
 * objective of cuda_v5_pwa_s0cw:
 *
 *     q_e      = w_e·(P_e/norm + bkg_e),     S_g = Σ_{e∈g} q_e
 *     W_g      = Σ_{e∈g} w_e
 *     Q        = -Σ_g W_g·log(S_g/W_g)
 *     dQ/dP_e  = -(W_g/S_g)·(w_e/norm)
 *     dQ/dnorm = Σ_g W_g·Pb_g/(S_g·norm²),   Pb_g = Σ_{e∈g} w_e·P_e
 *
 * A fully zero-weight group (W_g == 0) contributes nothing: no log, group
 * scale 0.  At resolution_size == 1 this is ``-w_e·log(w_e·pdf_e/w_e) =
 * -w_e·log(pdf_e)``, i.e. the per-event NLL for ANY weights.
 *
 * The s0e tile kernel is already BASIS-AGNOSTIC: it consumes an explicit
 * ``coeff`` block and returns ``g_coeff``, so the ``pwa_gen`` generic
 * denominator needs NO new compute algebra.  Only the addressing is new:
 *
 *   1. **flat CSR basis addressing** — the padded
 *      ``coeff[f·n_basis + i]`` / ``slot_map[f·n_basis + i]`` indexing is
 *      replaced by ``coeff[off[f] + i]`` / ``g_idx[off[f] + i]`` and the
 *      per-factor loops run over ``off[f] … off[f+1]-1``.  ``wave_of_slot``
 *      (``[n_basis_tot]``) lets the coefficient-gradient phase recover the
 *      wave (hence the projection ``p`` and the ``ck`` column ``k``) from a
 *      flat slot index.  No pinned/padded entry exists anywhere.
 *      ``off`` is indexed by the **factor** ``f = w·n_res + r``, so a wave
 *      with several BW factors owns one contiguous range per factor and
 *      the factors are multiplied (cuda_v4_pwa_s0e / the CPU
 *      ``cpu_pwa_gen`` model).  The gradient phase groups its work per
 *      factor, so the other-factor product is formed once per event
 *      instead of once per slot.
 *   2. **combined denominator table, built HERE** (as in the s0e gen
 *      sibling) — this context receives the raw ``gamma_table`` rows and
 *      the ``pwa_gen`` ``den_table`` rows as data and concatenates them
 *      (inside its own create, into a buffer it owns and uploads once) into
 *      the basis table this kernel indexes:
 *      ``[gamma rows] + ["1"] + ["m²"] + [den rows]``, the finished basis
 *      rows rotated into the kernel's ``F = -i·Z`` convention.  The
 *      existing ``uslot_row`` / ``us_cache_fill_gen_flat_kernel`` machinery
 *      then addresses that table, so this kernel sees no difference between
 *      a gamma row and a tabulated basis row.  Nothing is shared with, or
 *      appended to, any other kernel's table.
 *   3. **numerator factor** ``N_{n_index[w]}(m_e)`` — pure fixed data, so
 *      it is applied ONCE in the amp-cache FILL (load time), never in the
 *      per-iteration loop: each cached ``Amp = fa·fl`` is multiplied by
 *      the interpolated numerator row.  ``n_index[w] == 0`` is the
 *      constant row and is short-circuited, and a config without
 *      ``n_table`` passes NULL — both make an ``N ≡ 1`` config
 *      bit-identical to ``cuda_v5_pwa_s0e``.
 *
 * cuda_v4_pwa_s0e keeps EVERY per-iteration intermediate (BW, 1/BW, common,
 * common/BW, A_p, dQ/dA_p) in shared memory for a tile of S0E_TE events and
 * computes the forward, the ck gradient and the coefficient gradient in ONE
 * launch (per-block accumulation, no batching).  This kernel additionally
 * replaces the per-event NLL with the cuda_v5 grouped log-sum, WITHOUT a
 * second BW/common pass — grouping only changes the per-event scalar
 * dQ/dP_e that feeds the already-fused gradient phases (formulas above).
 *
 *   * rs <= S0E_TE: a block covers rs·(S0E_TE/rs) events, i.e. WHOLE
 *     groups (only the last block's last group can be partial).  The tile
 *     is computed once, the group sums live in shared memory and phase B2
 *     folds them into dQ/dP_e before the gradient phases.
 *   * rs > S0E_TE: one block = one GROUP, walked internally in S0E_TE-event
 *     chunks.  Because W_g/S_g is only known when the group closes, the
 *     gradient phases accumulate with the per-event factor f_e = w_e/norm
 *     into SHARED per-group accumulators (grad_ck[2N] + grad_coeff
 *     [n_basis_tot], ~8 KB at n_wave=198); at group close the block
 *     multiplies them by const_g = -W_g/S_g (0 for W_g == 0) and issues ONE
 *     set of global atomics.  Still exactly one BW pass per event.
 *
 * norm=None stays per-event linear in both paths (Q = Σ w_e·P_e,
 * dQ/dP_e = w_e) — identical to cuda_v4_pwa_s0e.  `P_out` is per-event P.
 * Q / dNLL/dnorm accumulate in the persistent context accumulators
 * (deterministic per block, atomics across blocks, as in the v5 family).
 *
 * The rest is inherited verbatim from cuda_v5_pwa_s0e: the two load-time
 * caches (amp_cache_fill_gen_kernel / us_cache_fill_gen_flat_kernel), the
 * explicit coefficient basis, the pinned parameter staging, the persistent
 * accumulator buffer, and the chunked gram path.
 *
 * Derived from cuda_v5_pwa_gen_s0e; no existing kernel is modified.
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
__device__ double catmull_rom_1d_gen(
    double pm1, double p0, double p1, double p2, double t
) {
    return p0 + 0.5 * t * (
        -pm1 + p1 + t * (2.0*pm1 - 5.0*p0 + 4.0*p1 - p2 + t * (-pm1 + 3.0*p0 - 3.0*p1 + p2))
    );
}

// Complex Catmull-Rom for gamma table (uses inv_delta)
__device__ complex interp_complex_gen_device(
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
    double real_val = catmull_rom_1d_gen(
        table_real[im1], table_real[base], table_real[base + 1], table_real[i2], t);
    double imag_val = catmull_rom_1d_gen(
        table_imag[im1], table_imag[base], table_imag[base + 1], table_imag[i2], t);
    return complex(real_val, imag_val);
}

// Real Catmull-Rom for FL factor
__device__ double interp_real_gen_device(
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
    return catmull_rom_1d_gen(
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
//
// pwa_gen numerator: the cached value is ``fa·fl·N_{n_index[w]}(m_e)`` — the
// numerator is FIXED data, so it is folded in here (load time) instead of
// the per-iteration tile loop.  ``n_index[w] == 0`` is the constant row and
// is short-circuited; a NULL ``n_index`` skips the factor entirely (the
// plain-PWA path, bit-identical to cuda_v5_pwa_s0e).
//=============================================================================
__global__ void amp_cache_fill_gen_kernel(
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
    int n_wave, int n_res, int n_angle_k, int n_angle_total, int n_angle_comp,
    int n_decay, int n_momentum, int fl_table_bins,
    int n_uniq,
    // pwa_gen numerator table (NULL -> no factor)
    const double* __restrict__ n_table_r,
    const double* __restrict__ n_table_i,
    int n_rows, int n_bins, double n_min, double inv_n_delta,
    const int* __restrict__ n_index,
    const int* __restrict__ n_mass_index,
    const double* __restrict__ mass, int n_mass,
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
            fl *= interp_real_gen_device(fl_table, fl_type[fl_idx],
                                     q, fl_min, fl_delta, fl_table_bins);
        }
        double ar = fr * fl, ai = fi * fl;
        if (n_index != NULL) {
            // pwa_gen numerator, one factor per BW (like ``bw_order``):
            // ``f = w·n_res + r`` indexes both the table row and THAT
            // factor's mass column, so two numerator-bearing resonances on
            // one chain multiply N_R1(m_R1)·N_R2(m_R2).  Row 0 is the
            // constant 1 (short-circuited), keeping N == 1 bit-identical.
            const int nf0 = w * n_res;
            for (int r = 0; r < n_res; r++) {
                int ni = n_index[nf0 + r];
                if (ni == 0) continue;
                double me = mass[(size_t)event_idx * n_mass
                                 + n_mass_index[nf0 + r]];
                complex N = interp_complex_gen_device(
                    n_table_r, n_table_i, ni, me,
                    n_min, inv_n_delta, n_bins);
                double nr = N.real(), nii = N.imag();
                double tr = ar * nr - ai * nii;
                ai = ai * nr + ar * nii;
                ar = tr;
            }
        }
        amp_cache[event_idx * n_uniq + s] = make_float2((float)ar, (float)ai);
    }
}
//=============================================================================
// KERNEL 2b: cached forward — BW propagator × cached angular amp.
// Per event; threads stride over the P·N wave entries.  Same as
// cuda_v3_ampcache's amp_cache_amp_gen_kernel but entry-count generic, with
// the flat per-(wave, factor) basis and the BW **product** over the factors.
//=============================================================================
__global__ void amp_cache_amp_gen_flat_kernel(
    const double* __restrict__ us_r,          // s0d: u-cache [nb·n_uslot]
    const double* __restrict__ us_i,
    const double* __restrict__ coeff,         // flat: [n_basis_tot]
    const int* __restrict__ off,              // flat: [n_fact + 1]
    const int* __restrict__ g_idx,            // flat: [n_basis_tot] -> uslot
    int n_wave, int n_res, int n_uslot,
    const float2* __restrict__ amp_cache,
    const int* __restrict__ slot_of_wave,
    int n_uniq,
    double* __restrict__ common_amp_factor_real,
    double* __restrict__ common_amp_factor_imag,
    double* __restrict__ bw_dom_real, double* __restrict__ bw_dom_imag,
    int n_events
) {
    // s0d u-cache: the gamma interp (row, mass-col) values were filled
    // ONCE per data load by us_cache_fill_gen_flat_kernel (they depend only on the
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
        const int f0 = wave_idx * n_res;
        // product of the factors; the empty-product start (1, 0) makes the
        // single-factor path exact (1·bw − 0·bw == bw bit for bit).
        double bw_r = 1.0, bw_i = 0.0;
        if (n_uslot > 0) {
            for (int r = 0; r < n_res; r++) {
                const int f = f0 + r;
                double fr = 0.0, fi = 0.0;
                for (int b = off[f]; b < off[f + 1]; b++) {
                    int u = g_idx[b];
                    double M = coeff[b];
                    fr += M * ui[u];             // -i·Z: Re = +Z_i
                    fi -= M * ur[u];             //       Im = -Z_r
                }
                const double t_r = bw_r * fr - bw_i * fi;
                bw_i = bw_r * fi + bw_i * fr;
                bw_r = t_r;
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
__global__ void us_cache_fill_gen_flat_kernel(
    const double* __restrict__ mass, int n_mass,
    const int* __restrict__ uslot_row,        // [n_uslot] ext-table row
    const int* __restrict__ uslot_col,        // [n_uslot] mass column
    const int* __restrict__ off,              // flat: [n_fact + 1]
    const int* __restrict__ g_idx,            // flat: [n_basis_tot] -> uslot
    const int* __restrict__ slot_l,           // flat: [n_basis_tot]
                                              //   merged row (-1: non-gamma)
    const double* __restrict__ gamma_table_real,
    const double* __restrict__ gamma_table_imag,
    double gamma_min, double gamma_inv_delta, int gamma_table_bins,
    int n_gamma_rows, int n_uslot, int n_fact,
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
        complex z = interp_complex_gen_device(
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
    // per (wave, factor) contiguous range (off is factor-indexed)
    for (int f = tid; f < n_fact; f += block_sz) {
        const int b1 = off[f + 1];
        for (int b = off[f]; b < b1; b++) {
            int l = slot_l[b];
            if (l >= 0) {
                int u = g_idx[b];
                gi_real[(size_t)event_idx * n_gamma_rows + l] = s_us_r[u];
                gi_imag[(size_t)event_idx * n_gamma_rows + l] = s_us_i[u];
            }
        }
    }
}

//=============================================================================
// v5 s0e: the fused per-iteration tile kernel + the GROUPED LOG-SUM data NLL.
//
// One block owns whole GROUPS: rs·(S0E_TE/rs) consecutive events when
// rs <= S0E_TE (the last block excepted — its last group may be partial),
// or exactly one group when rs > S0E_TE (walked internally in S0E_TE-event
// chunks).  Every intermediate (BW, 1/BW, common, common/BW, A_p, dQ/dA_p)
// stays in shared memory, so each event gets exactly ONE BW pass even in
// the rs > TE path.
//
// shared: 4 arrays of [S0E_TE*n_wave] + 4 of [S0E_TE*n_proj] doubles
//         + a tail of per-event P/q/scale (+ the n_res > 1 factor cache)
//         (+ the rs>TE group accumulators)
//=============================================================================
#define S0E_TE    8        /* events per tile (chunk in the rs>TE path) */
#define S0E_BLOCK 256      /* threads */
#define S0E_KACC  8        /* coefficient-gradient accumulators per thread */

/* shared tail: 3 arrays of S0E_TE (P_e, q_e, group scale) + the
   per-(event, factor) BW factor cache (nfr = n_fact when n_res > 1, else 0)
   for the whole-group path, or the same plus [3] totals (S, WP, W) + the
   shared per-group gradient accumulators grad_ck[2N] +
   grad_coeff[n_basis_tot] */
#define S0E_TAIL(nw, P, nb, nfr) \
    (3 * S0E_TE + 2 * S0E_TE * (nfr) + 3 + 2 * ((nw) / (P)) + (nb))

#define S0E_SHM_V5(nw, P, nb, nfr) \
    ((size_t)(4 * S0E_TE * (nw) + 4 * S0E_TE * (P) \
              + S0E_TAIL((nw), (P), (nb), (nfr))) * sizeof(double))

__global__ void s0e_tile_gen_flat_kernel_v5(
    const double* __restrict__ coeff,        /* flat: [n_basis_tot] */
    const int* __restrict__ off,             /* flat: [n_fact + 1] */
    const int* __restrict__ g_idx,           /* flat: [n_basis_tot] -> uslot */
    const int* __restrict__ wave_of_slot,    /* flat: [n_basis_tot] -> wave
                                                (kept for the API/validation;
                                                the gradient phase derives the
                                                wave as f / n_res) */
    const int* __restrict__ slot_of_wave,    /* [n_wave] -> amp cache slot */
    const double* __restrict__ us_r,         /* [ne*n_uslot] */
    const double* __restrict__ us_i,
    const float2* __restrict__ amp_cache,    /* [ne*n_uniq] */
    const double* __restrict__ ck_r, const double* __restrict__ ck_i,
    const double* __restrict__ weight, const double* __restrict__ bkg,
    int n_wave, int n_res, int N, int n_proj, int n_bt, int n_uslot, int n_uniq,
    int ne, int use_norm, double norm, int rsize,
    double* __restrict__ Q_acc, double* __restrict__ dnorm_acc,
    double* __restrict__ g_ck_r, double* __restrict__ g_ck_i,
    double* __restrict__ g_coeff,
    double* __restrict__ P_out)              /* may be NULL, [ne] */
{
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    const int P   = n_proj;
    const int nc  = n_bt;              /* flat basis slots (no padding) */
    const int n_fact = n_wave * n_res; /* (wave, factor) entries */
    const int nfr = (n_res > 1) ? n_fact : 0;  /* factor cache (0: absent) */
    const int rs  = (rsize < 1) ? 1 : rsize;
    const int pathB = (rs > S0E_TE);

    const int per_blk = pathB ? rs : rs * (S0E_TE / rs);
    const int e0  = blockIdx.x * per_blk;
    const int evn = (ne - e0 > per_blk) ? per_blk : (ne - e0);
    if (evn <= 0) return;

    extern __shared__ double sh[];
    double* s_cr = sh;                                   /* common      */
    double* s_ci = s_cr + (size_t)S0E_TE * n_wave;
    double* s_xr = s_ci + (size_t)S0E_TE * n_wave;       /* common/BW   */
    double* s_xi = s_xr + (size_t)S0E_TE * n_wave;
    double* s_ar = s_xi + (size_t)S0E_TE * n_wave;       /* A_p         */
    double* s_ai = s_ar + (size_t)S0E_TE * P;
    double* s_dr = s_ai + (size_t)S0E_TE * P;            /* dQ/dA_p     */
    double* s_di = s_dr + (size_t)S0E_TE * P;
    double* s_pe = s_di + (size_t)S0E_TE * P;            /* P_e         */
    double* s_q  = s_pe + S0E_TE;                        /* q_e         */
    double* s_sc = s_q + S0E_TE;                         /* group scale */
    double* s_fr = s_sc + S0E_TE;                        /* bw_{w,r}     */
    double* s_fi = s_fr + (size_t)S0E_TE * nfr;          /*   n_res > 1  */
    double* s_tot = s_fi + (size_t)S0E_TE * nfr;         /* [3]  path B */
    double* s_gr = s_tot + 3;                            /* [N]  path B */
    double* s_gi = s_gr + N;
    double* s_gc = s_gi + N;                             /* [nc] path B */

    /* gradient targets: the global accumulators (whole-group path) or the
       shared per-group accumulators (rs > TE path; one global atomic set
       per group at close, so the group constant -1/S_g can be applied) */
    double* d_ck_r = g_ck_r;
    double* d_ck_i = g_ck_i;
    double* d_coeff = g_coeff;
    if (pathB) {
        d_ck_r = s_gr; d_ck_i = s_gi; d_coeff = s_gc;
        for (int k = tid; k < N; k += nt) { s_gr[k] = 0.0; s_gi[k] = 0.0; }
        for (int i = tid; i < nc; i += nt) s_gc[i] = 0.0;
        if (tid == 0) { s_tot[0] = 0.0; s_tot[1] = 0.0; s_tot[2] = 0.0; }
        __syncthreads();
    }

    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int nwarps = nt >> 5;
    const int nchunk = pathB ? ((evn + S0E_TE - 1) / S0E_TE) : 1;

    for (int c = 0; c < nchunk; c++) {
        const int ce0 = e0 + c * S0E_TE;
        int nen = pathB ? (evn - c * S0E_TE) : evn;
        if (nen > S0E_TE) nen = S0E_TE;

        /* phase A: per-factor bw_{w,r}, the BW product, 1/BW, common,
           common/BW (one thread per (event, wave)) */
        for (int idx = tid; idx < nen * n_wave; idx += nt) {
            const int e = idx / n_wave;
            const int w = idx - e * n_wave;
            const double* ur = us_r + (size_t)(ce0 + e) * n_uslot;
            const double* ui = us_i + (size_t)(ce0 + e) * n_uslot;
            const int f0 = w * n_res;
            /* the factors are MULTIPLIED; the (1, 0) start is exact, so
               n_res == 1 gives back the single factor bit for bit */
            double bw_r = 1.0, bw_i = 0.0;
            for (int r = 0; r < n_res; r++) {
                const int f = f0 + r;
                double fr = 0.0, fi = 0.0;
                for (int b = off[f]; b < off[f + 1]; b++) {
                    const double M = coeff[b];
                    const int u = g_idx[b];
                    fr += M * ui[u];         /* -i*Z : Re = +Z_i */
                    fi -= M * ur[u];         /*        Im = -Z_r */
                }
                if (nfr) {                   /* n_res > 1: keep the factor */
                    s_fr[e * nfr + f] = fr;
                    s_fi[e * nfr + f] = fi;
                }
                const double t_r = bw_r * fr - bw_i * fi;
                bw_i = bw_r * fi + bw_i * fr;
                bw_r = t_r;
            }
            const double den = bw_r * bw_r + bw_i * bw_i;
            const double ir =  bw_r / den;
            const double ii = -bw_i / den;
            const float2 A = amp_cache[(size_t)(ce0 + e) * n_uniq
                                       + slot_of_wave[w]];
            const double ar = (double)A.x, ai = (double)A.y;
            const double m_r = ar * ir - ai * ii;
            const double m_i = ar * ii + ai * ir;
            const int o = e * n_wave + w;
            s_cr[o] = m_r; s_ci[o] = m_i;
            s_xr[o] = m_r * ir - m_i * ii;
            s_xi[o] = m_r * ii + m_i * ir;
        }
        __syncthreads();

        /* phase B: A_p, P_e and the group-summand q_e (one warp per event) */
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
                const double wv = weight[ce0 + e];
                const double bv = bkg[ce0 + e];
                s_pe[e] = Pe;
                s_q[e] = use_norm ? wv * (Pe / norm + bv) : wv * Pe;
                if (P_out) P_out[ce0 + e] = Pe;
            }
        }
        __syncthreads();

        /* phase G: group summaries -> Q, dNLL/dnorm and the dQ/dP scale.
           rs <= TE: whole groups live in this chunk (a partial last group
           is allowed only in the last block); rs > TE: thread 0 folds this
           chunk into the block's group totals (serially, deterministic) */
        if (!pathB) {
            const int ng = (nen + rs - 1) / rs;
            for (int g = tid; g < ng; g += nt) {
                const int a = g * rs;
                const int b = (a + rs < nen) ? (a + rs) : nen;
                double S = 0.0, WP = 0.0, W = 0.0;
                for (int e = a; e < b; e++) {
                    S += s_q[e];
                    WP += weight[ce0 + e] * s_pe[e];
                    W += weight[ce0 + e];
                }
                if (use_norm) {
                    /* group-total-weight objective: -W·log(S/W) */
                    if (W > 0.0) {
                        s_sc[g] = -W / S;
                        atomicAdd(Q_acc, -W * log(S / W));
                        atomicAdd(dnorm_acc, W * WP / (S * norm * norm));
                    } else {
                        s_sc[g] = 0.0;      /* zero-weight group: inert */
                    }
                } else {
                    s_sc[g] = 1.0;
                    atomicAdd(Q_acc, S);
                }
            }
        } else if (tid == 0) {
            double S = 0.0, WP = 0.0, W = 0.0;
            for (int e = 0; e < nen; e++) {
                S += s_q[e];
                WP += weight[ce0 + e] * s_pe[e];
                W += weight[ce0 + e];
            }
            s_tot[0] += S;
            s_tot[1] += WP;
            s_tot[2] += W;
        }
        __syncthreads();

        /* phase B2: dQ/dP_e -> dQ/dA_p (scaled later in the rs > TE path) */
        for (int e = tid; e < nen; e += nt) {
            const double wv = weight[ce0 + e];
            double dqp = use_norm ? (wv / norm) : wv;
            if (!pathB) dqp *= s_sc[e / rs];
            for (int p = 0; p < P; p++) {
                s_dr[e * P + p] =  dqp * s_ar[e * P + p];
                s_di[e * P + p] = -dqp * s_ai[e * P + p];
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
            atomicAdd(d_ck_r + k, sr);
            atomicAdd(d_ck_i + k, si);
        }

        /* phase D: coefficient gradient, grouped per (wave, factor) so the
           product of the OTHER factors (prefix·suffix over r' != r — no
           division by a factor that can vanish) is formed once per EVENT
           instead of once per slot.  Flat slots; the wave — hence p/k —
           comes from the factor index (f / n_res). */
        for (int f = tid; f < n_fact; f += nt) {
            const int w = f / n_res;
            const int r = f - w * n_res;
            const int b0 = off[f], b1 = off[f + 1];
            const int p = w / N;
            const int k = w - p * N;
            const double ckr = ck_r[k], cki = ck_i[k];
            const int nsl = b1 - b0;
            for (int i0 = 0; i0 < nsl; i0 += S0E_KACC) {
                const int ilim = (nsl - i0 < S0E_KACC)
                    ? (nsl - i0) : S0E_KACC;
                double acc[S0E_KACC];
                for (int j = 0; j < S0E_KACC; j++) acc[j] = 0.0;
                for (int e = 0; e < nen; e++) {
                    const double dr = s_dr[e * P + p], di = s_di[e * P + p];
                    const int o = e * n_wave + w;
                    /* d = dQ/dcommon_w — only projection p touches this
                       wave, so p and k are fixed here */
                    const double ur = dr * ckr - di * cki;
                    const double ui = dr * cki + di * ckr;
                    /* dBW·(1/bw_{w,r}) = -(d·common/BW)·Π_{r'≠r} bw_{w,r'} */
                    double dbr = -(ur * s_xr[o] - ui * s_xi[o]);
                    double dbi = -(ur * s_xi[o] + ui * s_xr[o]);
                    if (n_res > 1) {
                        const int ef = e * nfr + w * n_res;
                        for (int r2 = 0; r2 < n_res; r2++) {
                            if (r2 == r) continue;
                            const double ar2 = dbr, ai2 = dbi;
                            const double br2 = s_fr[ef + r2];
                            const double bi2 = s_fi[ef + r2];
                            dbr = ar2 * br2 - ai2 * bi2;
                            dbi = ar2 * bi2 + ai2 * br2;
                        }
                    }
                    const size_t ubase = (size_t)(ce0 + e) * n_uslot;
                    for (int j = 0; j < ilim; j++) {
                        const int u = g_idx[b0 + i0 + j];
                        const size_t uo = ubase + u;
                        acc[j] += 2.0 * (dbr * us_i[uo] + dbi * us_r[uo]);
                    }
                }
                for (int j = 0; j < ilim; j++)
                    atomicAdd(d_coeff + b0 + i0 + j, acc[j]);
            }
        }
        __syncthreads();   /* shared tile / accumulators reused next chunk */
    }

    /* group close (rs > TE): Q / dNLL/dnorm once, then ONE scaled atomic
       set of the group's gradient accumulators */
    if (pathB) {
        if (tid == 0) {
            const double S = s_tot[0], WP = s_tot[1], W = s_tot[2];
            if (use_norm) {
                if (W > 0.0) {
                    atomicAdd(Q_acc, -W * log(S / W));
                    atomicAdd(dnorm_acc, W * WP / (S * norm * norm));
                    s_sc[0] = -W / S;
                } else {
                    s_sc[0] = 0.0;              /* zero-weight group: inert */
                }
            } else {
                atomicAdd(Q_acc, S);
                s_sc[0] = 1.0;
            }
        }
        __syncthreads();
        const double cs = s_sc[0];
        for (int k = tid; k < N; k += nt) {
            atomicAdd(g_ck_r + k, cs * s_gr[k]);
            atomicAdd(g_ck_i + k, cs * s_gi[k]);
        }
        for (int idx = tid; idx < nc; idx += nt)
            atomicAdd(g_coeff + idx, cs * s_gc[idx]);
    }
}
//=============================================================================
// PWA wave-Gram reduction: D[k,j] = Σ_e w_e Σ_p conj(a_{p,k})·a_{p,j}
// One block per event; each thread owns one (k,j) cell of N×N.
//=============================================================================
__global__ void gram_v4_event_gen_flat_kernel(
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
void* _up_int_gen(const int* src, int n) {
    if (n <= 0 || src == NULL) return NULL;   // empty upload = no buffer
    int* d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, src, n * sizeof(int), cudaMemcpyHostToDevice)); return d;
}
void* _up_dbl_gen(const double* src, int n) {
    if (n <= 0 || src == NULL) return NULL;
    double* d; CUDA_CHECK(cudaMalloc(&d, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d, src, n * sizeof(double), cudaMemcpyHostToDevice)); return d;
}
float* _up_f32_gen(const double* src, int n) {
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
    // [Q, dnorm, gck_r(N), gck_i(N), gcoeff(n_basis_tot)]
    double* acc_buf;
    // s0b: persistent PARAMETER buffers, refreshed per call with a plain
    // memcpy (no cudaMalloc/cudaFree per iteration)
    double* p_ck_r; double* p_ck_i;     // [N]
    double* p_coeff;                    // flat [n_basis_tot]
    // s0b: PINNED host staging for the param upload
    double* h_ck_r; double* h_ck_i;
    double* h_coeff;                    // flat [n_basis_tot]
    // s0b: dedicated NON-LEGACY stream
    cudaStream_t stream;
    double* acc_host;                   // pinned [acc_total] output staging
    // flat CSR basis layout — per-wave contiguous slot ranges, no padding
    int n_basis_tot;                    // off[n_wave] = Sum_w n_basis_w
    const int* off;                     // [n_wave + 1]
    const int* g_idx;                   // [n_basis_tot] -> uslot
    const int* wave_of_slot;            // [n_basis_tot] -> wave
    int rsize;                          // v5 log-sum group size (>= 1)
    // deduped interp tables — one unique (table row, mass column) per entry
    int n_uslot;
    const int* uslot_row;               // [n_uslot]
    const int* uslot_col;               // [n_uslot]
    const int* slot_l;                  // [n_basis_tot] (-1: non-gamma)
    int n_gamma_rows_ext;               // uploaded combined basis table rows
    int gamma_table_bins_ext;
    // pwa_gen: combined basis table row count (gamma + "1" + "m²" + den rows)
    int n_basis_rows;
    // pwa_gen numerator: fixed table N_r/N_i [n_num_rows · n_num_bins]
    const double* n_table_real;
    const double* n_table_imag;
    int n_num_rows; int n_num_bins;
    double n_min; double n_inv_delta;
    const int* n_index;                 // [n_wave]
    const int* n_mass_index;            // [n_wave]
} ComputeContextGen;

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
} DataHandleGen;

int cuda_get_device_count() { int count; cudaGetDeviceCount(&count); return count; }
cudaError_t cuda_get_device_name(char* name, int len) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err == cudaSuccess) strncpy(name, prop.name, len);
    return err;
}

// The shared tile of S0E_TE events (plus, in the rs > TE path, the shared
// per-group gradient accumulators) exceeds the 48 KB static limit, so the
// kernel needs the opt-in dynamic-shared-memory limit (set once per size).
static void genflat_set_shmem(size_t shm) {
    static size_t s_set = 0;
    if (shm > s_set) {
        cudaFuncSetAttribute(s0e_tile_gen_flat_kernel_v5,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)shm);
        s_set = shm;
    }
}

void cuda_free_context_v5(void* vctx);  // forward decl (used by create on error)
void* cuda_create_context_v5_gen_flat(
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
    int n_basis_tot,
    int rsize,
    const int* uslot_row,int n_ur,
    const int* uslot_col,int n_uc,
    // ── flat CSR basis addressing (no padding) ──────────────────────
    const int* off,int n_off,
    const int* g_idx,int n_gidx,
    const int* wave_of_slot,int n_wos,
    const int* slot_l,int n_sl,
    int n_gamma_rows_ext,int gamma_table_bins_ext,
    // ── pwa_gen extensions ──────────────────────────────────────────
    // generic denominator basis (0 = plain s0 preset: two pre-rotated
    // rows "1"/+m²; 1 = pwa_gen: "1"/−m² + the den_table rows)
    int generic_den,
    const double* dtr,int n_dtr, const double* dti,int n_dti, int n_den_rows,
    // numerator table N (NULL when the config has none / is all-constant)
    const double* ntr,int n_ntr, const double* nti,int n_nti,
    int n_num_rows,int n_num_bins,double n_min_t,double n_delta_t,
    const int* n_index,int n_ni, const int* n_mass_index,int n_nmi
) {
    (void)batch_size;   // s0e does not batch
    ComputeContextGen* c = (ComputeContextGen*)calloc(1, sizeof(ComputeContextGen));
    c->m0_index = (int*)_up_int_gen(m0_i, n1); c->g0_index = (int*)_up_int_gen(g0_i, n2);
    c->g0_mass_index = (int*)_up_int_gen(g0_m, n3); c->mass_index = (int*)_up_int_gen(mass_i, n4);
    c->fl_type = (int*)_up_int_gen(fl_t, n5); c->fl_q_index = (int*)_up_int_gen(fl_q, n6);
    c->bw_order = (n7 > 0) ? (int*)_up_int_gen(bw_o, n7) : NULL;  // unused (per-wave slots)
    c->fl_order = (int*)_up_int_gen(fl_o, n8);
    c->angle_index = (int*)_up_int_gen(ang_i, n9);
    c->angle_k = (double*)_up_dbl_gen(ak, n10); c->angle_b = (double*)_up_dbl_gen(ab, n11);
    c->matrix_angle_real = (double*)_up_dbl_gen(mar, n12); c->matrix_angle_imag = (double*)_up_dbl_gen(mai, n13);
    // ── combined BASIS table (this context owns it) ─────────────────
    // Rows, in order:  [0, n_raw)                the received gamma_table
    //                  n_raw                     "1"   -> Z = (0, 1)   => F = 1
    //                  n_raw + 1                 "m²"  -> Z = (0, ±m²)  => F = ∓m²
    //                  n_raw + 2 + r             den_table[r] -> Z = i·F
    // The kernel's convention is F = -i·Z, so a finished basis row F is
    // uploaded rotated (Z = i·F) and the gamma rows arrive unrotated.
    // ``generic_den`` only flips the m² row sign (the pwa_gen den_row
    // encoding uses F = −m²; the plain s0 preset uses F = +m² with
    // coefficient −1).
    {
        int bins = gamma_table_bins_ext > 0 ? gamma_table_bins_ext : gbins;
        int n_raw = (bins > 0) ? (n14 / bins) : 0;
        int n_extra = generic_den ? (n_den_rows > 0 ? n_den_rows : 0) : 0;
        int n_rows_tot = n_raw + 2 + n_extra;
        if (bins <= 0 || n_raw * bins != n14 || n15 != n14
                || (n_extra > 0 && (n_dtr != n_extra * bins
                                    || n_dti != n_dtr))) {
            fprintf(stderr,
                    "cuda_v5_pwa_gen_flat: inconsistent basis table sizes "
                    "(bins %d, gamma %d/%d, den %d rows / %d/%d)\n",
                    bins, n14, n15, n_den_rows, n_dtr, n_dti);
            cuda_free_context_v5(c);
            return NULL;
        }
        double* h_r = (double*)malloc((size_t)n_rows_tot * bins * sizeof(double));
        double* h_i = (double*)malloc((size_t)n_rows_tot * bins * sizeof(double));
        if (!h_r || !h_i) {
            free(h_r); free(h_i);
            cuda_free_context_v5(c);
            return NULL;
        }
        if (n14 > 0) {
            memcpy(h_r, gtr, (size_t)n14 * sizeof(double));
            memcpy(h_i, gti, (size_t)n15 * sizeof(double));
        }
        double* r1 = h_r + (size_t)n_raw * bins;
        double* i1 = h_i + (size_t)n_raw * bins;
        for (int j = 0; j < bins; j++) {
            double grid = gmin + gdel * (double)j;
            r1[j] = 0.0; i1[j] = 1.0;                       /* "1"  */
            r1[bins + j] = 0.0;
            i1[bins + j] = generic_den ? -grid * grid : grid * grid;  /* "m²" */
        }
        for (int r = 0; r < n_extra; r++) {
            double* dr = h_r + (size_t)(n_raw + 2 + r) * bins;
            double* di = h_i + (size_t)(n_raw + 2 + r) * bins;
            for (int j = 0; j < bins; j++) {
                double fr = dtr[(size_t)r * bins + j];
                double fi = dti[(size_t)r * bins + j];
                dr[j] = -fi;      /* Z = i·F = (-F_i, F_r) */
                di[j] =  fr;
            }
        }
        c->gamma_table_real = (double*)_up_dbl_gen(h_r, n_rows_tot * bins);
        c->gamma_table_imag = (double*)_up_dbl_gen(h_i, n_rows_tot * bins);
        c->n_basis_rows = n_rows_tot;
        free(h_r); free(h_i);
    }
    c->gamma_min = gmin; c->gamma_delta = gdel; c->gamma_table_bins = gbins;
    c->n_gamma_rows_ext = n_gamma_rows_ext;
    c->gamma_table_bins_ext = gamma_table_bins_ext > 0
        ? gamma_table_bins_ext : gbins;

    // ── pwa_gen numerator table (fixed data; NULL when absent) ──────
    c->n_num_rows = n_num_rows;
    c->n_num_bins = n_num_bins;
    c->n_min = n_min_t;
    c->n_inv_delta = (n_delta_t != 0.0) ? 1.0 / n_delta_t : 0.0;
    if (n_ntr > 0 && n_nti == n_ntr) {
        c->n_table_real = (double*)_up_dbl_gen(ntr, n_ntr);
        c->n_table_imag = (double*)_up_dbl_gen(nti, n_nti);
    }
    if (n_ni > 0) {
        c->n_index = (int*)_up_int_gen(n_index, n_ni);
        c->n_mass_index = (int*)_up_int_gen(n_mass_index, n_nmi);
    }

    c->matrix_gamma = (n16 > 0) ? (double*)_up_dbl_gen(mg, n16) : NULL;  // unused (per-wave rows)
    c->gamma_col_idx = (int*)_up_int_gen(gci, ngci);
    c->fl_table = (double*)_up_dbl_gen(ft, n17); c->fl_min = flmin; c->fl_delta = fldel; c->fl_table_bins = fbins;
    c->n_wave = nw; c->n_res = nr > 0 ? nr : 1; c->n_decay = nd;
    c->n_unique_bw = nub; c->n_gamma_rows = ngr;
    c->n_mass = nm; c->n_momentum = nmom; c->n_angle_k = nak_; c->n_angle_total = nat; c->n_angle_comp = nac;
    c->n_m0_params = n_m0p; c->n_g0_params = n_g0p;
    c->slot_of_wave = (n_slot > 0) ? (int*)_up_int_gen(slot_of_wave, n_slot) : NULL;
    c->rep_of_slot = (n_rep > 0) ? (int*)_up_int_gen(rep_of_slot, n_rep) : NULL;
    c->row_start = (n_rs > 0) ? (int*)_up_int_gen(row_start, n_rs) : NULL;
    c->n_basis_tot = n_basis_tot > 0 ? n_basis_tot : 1;
    c->rsize = rsize > 0 ? rsize : 1;   // v5: group size of the log-sum NLL
    c->n_uslot = n_ur;
    c->uslot_row = (int*)_up_int_gen(uslot_row, n_ur);
    c->uslot_col = (int*)_up_int_gen(uslot_col, n_uc);
    c->off = (int*)_up_int_gen(off, n_off);
    c->g_idx = (int*)_up_int_gen(g_idx, n_gidx);
    c->wave_of_slot = (int*)_up_int_gen(wave_of_slot, n_wos);
    c->slot_l = (int*)_up_int_gen(slot_l, n_sl);
    // fail fast on an inconsistent CSR range table (checked on the HOST
    // copy — the uploaded buffers are device memory).  ``off`` is indexed
    // by the FACTOR f = w·n_res + r, so it has n_wave·n_res + 1 entries.
    const int n_fact = nw * (nr > 0 ? nr : 1);
    if (!off || n_off != n_fact + 1 || !g_idx || !wave_of_slot
            || off[0] != 0 || off[n_fact] != c->n_basis_tot
            || (int)n_gidx != c->n_basis_tot
            || (int)n_wos != c->n_basis_tot) {
        fprintf(stderr,
                "cuda_v5_pwa_gen_flat: inconsistent flat basis table "
                "(off %d entries, n_fact %d (n_wave %d, n_res %d), "
                "n_basis_tot %d, g_idx %d, wave_of_slot %d)\n",
                n_off, n_fact, nw, nr, c->n_basis_tot, (int)n_gidx,
                (int)n_wos);
        cuda_free_context_v5(c);
        return NULL;
    }
    c->n_uniq = (n_uniq > 0) ? n_uniq : 0;
    c->n_proj = n_proj > 0 ? n_proj : 1;
    if (c->n_wave % c->n_proj != 0) {
        fprintf(stderr, "cuda_v5_pwa_gen_flat: n_wave %d must be divisible by n_proj %d\n",
                c->n_wave, c->n_proj);
        cuda_free_context_v5(c);
        return NULL;
    }
    // the fused tile lives in shared memory: fail fast if it cannot fit
    {
        size_t shm = S0E_SHM_V5(c->n_wave, c->n_proj, c->n_basis_tot,
                                c->n_res > 1 ? n_fact : 0);
        int shm_opt = 0;
        cudaDeviceGetAttribute(&shm_opt,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        if ((int)shm > shm_opt) {
            fprintf(stderr,
                    "cuda_v5_pwa_gen_flat: shared tile %zu B exceeds the device "
                    "opt-in limit %d B (n_wave %d, n_res %d, n_basis_tot %d, "
                    "S0E_TE %d)\n",
                    shm, shm_opt, c->n_wave, c->n_res, c->n_basis_tot, S0E_TE);
            cuda_free_context_v5(c);
            return NULL;
        }
    }
    // s0b: persistent accumulators (small, independent of the event count)
    {
        int N = c->n_wave / c->n_proj;
        int acc_total = 2 + 2 * N + c->n_basis_tot;
        CUDA_CHECK(cudaMalloc(&c->acc_buf, acc_total * sizeof(double)));
    }
    // s0b: persistent parameter buffers + pinned staging + dedicated stream
    {
        int N = c->n_wave / c->n_proj;
        int acc_total = 2 + 2 * N + c->n_basis_tot;
        CUDA_CHECK(cudaMalloc(&c->p_ck_r, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c->p_ck_i, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c->p_coeff,
                              c->n_basis_tot * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&c->h_ck_r, N * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&c->h_ck_i, N * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&c->h_coeff,
                                  c->n_basis_tot * sizeof(double)));
        CUDA_CHECK(cudaStreamCreate(&c->stream));
        CUDA_CHECK(cudaMallocHost(&c->acc_host, acc_total * sizeof(double)));
    }
    return c;
}

void cuda_free_context_v5(void* vctx) {
    ComputeContextGen* c = (ComputeContextGen*)vctx;
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
    if (c->off) cudaFree((void*)c->off);
    if (c->g_idx) cudaFree((void*)c->g_idx);
    if (c->wave_of_slot) cudaFree((void*)c->wave_of_slot);
    if (c->slot_l) cudaFree((void*)c->slot_l);
    if (c->n_table_real) cudaFree((void*)c->n_table_real);
    if (c->n_table_imag) cudaFree((void*)c->n_table_imag);
    if (c->n_index) cudaFree((void*)c->n_index);
    if (c->n_mass_index) cudaFree((void*)c->n_mass_index);
    if (c->h_ck_r) cudaFreeHost(c->h_ck_r);
    if (c->h_ck_i) cudaFreeHost(c->h_ck_i);
    if (c->h_coeff) cudaFreeHost(c->h_coeff);
    if (c->acc_host) cudaFreeHost(c->acc_host);
    if (c->stream) cudaStreamDestroy(c->stream);
    free(c);
}

// Forward decls: the angular cache / u-cache are allocated once per handle.
static int alloc_handle_amp_cache_gen(ComputeContextGen* c, DataHandleGen* h);
static int alloc_handle_us_cache_gen(ComputeContextGen* c, DataHandleGen* h);
static int fill_amp_cache_chunk_gen(ComputeContextGen* c, DataHandleGen* h,
                                const float* mom, const float* ang,
                                int n_events, int base_event);

void* cuda_load_data_v5(void* vctx,
    const double* mass,int nmass, const double* mom,int nmom,
    const double* ang,int nang,
    const double* wgt,const double* bkg,int ne
) {
    DataHandleGen* h = (DataHandleGen*)calloc(1, sizeof(DataHandleGen));
    ComputeContextGen* c_ctx = (ComputeContextGen*)vctx;
    int nac = c_ctx ? c_ctx->n_angle_comp : 3;
    h->m = (const double*)_up_dbl_gen(mass, ne * nmass);
    h->w = (const double*)_up_dbl_gen(wgt, ne);
    h->b = (const double*)_up_dbl_gen(bkg, ne);
    h->ne = ne; h->nm = nmass; h->nmom = nmom; h->nat = nang; h->nac = nac;
    CUDA_CHECK(cudaMallocHost(&h->h_P, (size_t)ne * sizeof(double)));
    // s0e: per-event P (device) — written by the fused kernel, copied out
    // once when the caller asks for P
    if (cudaMalloc(&h->P_dev, (size_t)ne * sizeof(double)) != cudaSuccess)
        goto fail;

    if (c_ctx) {
        if (!alloc_handle_amp_cache_gen(c_ctx, h)) goto fail;
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
            failed = !fill_amp_cache_chunk_gen(c_ctx, h, gpu_mom, gpu_ang, nb, base);
        }
        free(tmp_mom); free(tmp_ang);
        cudaFree(gpu_mom); cudaFree(gpu_ang);
        if (failed) goto fail;
        // u-cache: gamma-interp values depend only on the fixed mass input
        // + the fixed interp tables — fill ONCE per handle.
        if (!alloc_handle_us_cache_gen(c_ctx, h)) goto fail;
        if (c_ctx->n_uslot > 0 || c_ctx->n_gamma_rows > 0) {
            int fb2 = c_ctx->n_wave < 1024 ? c_ctx->n_wave : 1024;
            if (fb2 < 64) fb2 = 64;
            us_cache_fill_gen_flat_kernel<<<ne, fb2,
                (size_t)(2 * c_ctx->n_uslot) * sizeof(double),
                c_ctx->stream>>>(
                h->m, h->nm,
                c_ctx->uslot_row, c_ctx->uslot_col,
                c_ctx->off, c_ctx->g_idx, c_ctx->slot_l,
                c_ctx->gamma_table_real, c_ctx->gamma_table_imag,
                c_ctx->gamma_min, 1.0 / c_ctx->gamma_delta,
                c_ctx->gamma_table_bins_ext,
                c_ctx->n_gamma_rows, c_ctx->n_uslot,
                c_ctx->n_wave * c_ctx->n_res,
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

void cuda_free_data_v5(void* vh) {
    if (!vh) return;
    DataHandleGen* h = (DataHandleGen*)vh;
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

static int alloc_handle_amp_cache_gen(ComputeContextGen* c, DataHandleGen* h) {
    if (h->amp_cache != NULL || h->n_uniq > 0) return 1;
    if (c->n_uniq <= 0 || !c->rep_of_slot || !c->matrix_angle_real) return 0;
    size_t need = (size_t)h->ne * c->n_uniq * sizeof(float2);
    float2* amp_cache = NULL;
    if (cudaMalloc(&amp_cache, need) != cudaSuccess) return 0;
    h->amp_cache = amp_cache;
    h->n_uniq = c->n_uniq;
    return 1;
}

static int alloc_handle_us_cache_gen(ComputeContextGen* c, DataHandleGen* h) {
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

static int fill_amp_cache_chunk_gen(ComputeContextGen* c, DataHandleGen* h,
                                const float* mom, const float* ang,
                                int n_events, int base_event) {
    size_t shmem = (size_t)c->n_angle_k * sizeof(double);
    int fill_b = c->n_uniq;
    if (fill_b < 256) fill_b = 256;
    if (fill_b > 1024) fill_b = 1024;
    amp_cache_fill_gen_kernel<<<n_events, fill_b, shmem, c->stream>>>(
        ang, c->angle_index,
        c->angle_k, c->angle_b,
        c->matrix_angle_real, c->matrix_angle_imag,
        mom, c->fl_type, c->fl_q_index, c->fl_order,
        c->fl_table, c->fl_min, c->fl_delta,
        c->rep_of_slot,
        c->n_wave, c->n_res, c->n_angle_k, c->n_angle_total, c->n_angle_comp,
        c->n_decay, c->n_momentum, c->fl_table_bins, c->n_uniq,
        c->n_table_real, c->n_table_imag,
        c->n_num_rows, c->n_num_bins, c->n_min, c->n_inv_delta,
        (c->n_table_real != NULL) ? c->n_index : NULL,
        (c->n_table_real != NULL) ? c->n_mass_index : NULL,
        h->m + (size_t)base_event * h->nm, h->nm,
        h->amp_cache + (size_t)base_event * c->n_uniq, n_events);
    return cudaGetLastError() == cudaSuccess;
}

//=============================================================================
// v5 gen flat compute: ONE fused launch over the whole event range (no
// batching).  Blocks own whole resolution groups (rs <= S0E_TE) or one group
// each (rs > S0E_TE) — see s0e_tile_gen_flat_kernel_v5.
//
// Named cuda_compute_v5_gen_flat: the flat CSR coefficient block
// (n_basis_tot) -> out dQ/dcoeff in the SAME flat layout (ogcoeff).  The
// 13-argument shape is the one of cuda_compute_v5_coeff; only the layout
// (and hence the symbol) differs.
//=============================================================================
// in: coeff (n_basis_tot, flat)  out: dQ/dcoeff (n_basis_tot, flat)
void cuda_compute_v5_gen_flat(void* vctx, void* vdh,
    const double* ck_r, const double* ck_i, const double* coeff,
    double norm, int use_norm,
    double* oQ, double* oDn, double* oP,
    double* ogck_r, double* ogck_i, double* ogcoeff
) {
    ComputeContextGen* c = (ComputeContextGen*)vctx;
    DataHandleGen* h = (DataHandleGen*)vdh;
    const int ne = h->ne;
    const int nw = c->n_wave;
    const int P = c->n_proj;
    const int N = nw / P;
    const int nbt = c->n_basis_tot;
    const int acc_total = 2 + 2 * N + nbt;
    const int off_gck_r = 2, off_gck_i = 2 + N, off_gcoeff = 2 + 2 * N;
    const int rs = c->rsize;   // v5: group size fixed at context creation

    // s0b: refresh the persistent parameter buffers (plain memcpy into the
    // pinned staging, then async H2D — no per-call malloc/free)
    memcpy(c->h_ck_r, ck_r, (size_t)N * sizeof(double));
    memcpy(c->h_ck_i, ck_i, (size_t)N * sizeof(double));
    memcpy(c->h_coeff, coeff, (size_t)nbt * sizeof(double));
    cudaMemcpyAsync(c->p_ck_r, c->h_ck_r, (size_t)N * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    cudaMemcpyAsync(c->p_ck_i, c->h_ck_i, (size_t)N * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    cudaMemcpyAsync(c->p_coeff, c->h_coeff, (size_t)nbt * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);

    *oQ = 0.0; *oDn = 0.0;
    cudaMemsetAsync(c->acc_buf, 0, (size_t)acc_total * sizeof(double),
                    c->stream);
    memset(ogck_r, 0, (size_t)N * sizeof(double));
    memset(ogck_i, 0, (size_t)N * sizeof(double));
    memset(ogcoeff, 0, (size_t)nbt * sizeof(double));

    if (ne > 0) {
        // blocks hold whole groups: rs·(S0E_TE/rs) events, or one group
        const int per_blk = (rs > S0E_TE) ? rs : rs * (S0E_TE / rs);
        const int tiles = (ne + per_blk - 1) / per_blk;
        const size_t shm = S0E_SHM_V5(nw, P, nbt,
                                      c->n_res > 1 ? nw * c->n_res : 0);
        genflat_set_shmem(shm);
        s0e_tile_gen_flat_kernel_v5<<<tiles, S0E_BLOCK, shm, c->stream>>>(
            c->p_coeff, c->off, c->g_idx, c->wave_of_slot, c->slot_of_wave,
            h->us_real, h->us_imag, h->amp_cache,
            c->p_ck_r, c->p_ck_i,
            h->w, h->b,
            nw, c->n_res, N, P, nbt, c->n_uslot, c->n_uniq,
            ne, use_norm, norm, rs,
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
    CUDA_CHECK(cudaStreamSynchronize(c->stream));
    *oQ += c->acc_host[0];
    *oDn += c->acc_host[1];
    for (int j = 0; j < N; j++) ogck_r[j] += c->acc_host[off_gck_r + j];
    for (int j = 0; j < N; j++) ogck_i[j] += c->acc_host[off_gck_i + j];
    for (int j = 0; j < nbt; j++) ogcoeff[j] += c->acc_host[off_gcoeff + j];
    if (oP != NULL) memcpy(oP, h->h_P, (size_t)ne * sizeof(double));
}

//=============================================================================
// Wave Gram matrix D (N,N) for a loaded phsp handle, chunked over events.
// Still uses the cached forward (common = Amp/BW) + the N×N event reduction
// — the gram path is NOT part of the fused per-iteration kernel.
//   D[k,j] = Σ_e w_e Σ_p conj(common[e,pN+k])·common[e,pN+j]
//=============================================================================
void cuda_gram_matrix_v5(void* vctx, void* vdh,
    const double* coeff,
    double* oDr, double* oDi
) {
    ComputeContextGen* c = (ComputeContextGen*)vctx;
    DataHandleGen* h = (DataHandleGen*)vdh;
    const int ne = h->ne;
    const int nw = c->n_wave, P = c->n_proj, N = nw / P;
    const int nbt = c->n_basis_tot;

    memcpy(c->h_coeff, coeff, (size_t)nbt * sizeof(double));
    cudaMemcpyAsync(c->p_coeff, c->h_coeff, (size_t)nbt * sizeof(double),
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
        amp_cache_amp_gen_flat_kernel<<<nb, amp_t, 0, c->stream>>>(
            h->us_real + (size_t)st * c->n_uslot,
            h->us_imag + (size_t)st * c->n_uslot,
            c->p_coeff, c->off, c->g_idx,
            nw, c->n_res, c->n_uslot,
            h->amp_cache + (size_t)st * c->n_uniq,
            c->slot_of_wave, c->n_uniq,
            cr, ci, br, bi, nb);
        CUDA_CHECK(cudaGetLastError());
        gram_v4_event_gen_flat_kernel<<<nb, 128, 0, c->stream>>>(
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
