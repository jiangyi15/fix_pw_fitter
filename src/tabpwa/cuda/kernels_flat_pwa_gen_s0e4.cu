/**
 * cuda_flat_pwa_gen_s0e4 — the s0e FUSED TILE on the CURRENT flat `pwa_gen`
 * config, with the grouped log-sum objective and batch_size chunking.
 *
 * Case-1 copy (kernel policy §1): own `.cu`, own `.so`, own wrapper, own
 * symbols.  `kernels_flat_pwa_gen2.cu`, `kernels_flat_pwa_gen_1bw.cu`,
 * every `kernels_*_s0*.cu` and their wrappers stay byte-identical.
 *
 * ── s0e4 vs s0e: EXACTLY TWO changes ────────────────────────────────────
 *
 * (1) **The fitted coefficients are PRE-GATHERED into CSR order on the
 *     host**, `cw[i] = coeff[nz_m[i]]` (one `cudaMemcpyAsync` of `n_nz`
 *     doubles per call), so the inner loop has no coefficient indirection:
 *     `cw[i]` is read straight against `us[nz_u[i]]`.  Exact — the same
 *     `double` values, just laid out in the order phase A walks them.
 *
 * (2) **A row with `nnz > GPS04_SPLIT_NNZ` is split into TWO contiguous
 *     halves processed by TWO threads**, so the phase barrier waits for
 *     ~nnz/2 instead of nnz.  Phase A therefore becomes "accumulate D only"
 *     (the two halves of a split row fold together with an `atomicAdd` in
 *     shared memory) and the reciprocal + `common` move to a new phase A2
 *     after a barrier, which also isolates the fp64 divide.
 *
 * Why: measured on the PRODUCTION config
 * (`/home/jiangy/github/project94/pwa_workspace/jpsi_eta_pi_pi/04_model_search/
 * run_423_kmatrix_n_20261119_141110/config_kn4_plot.yml`): n_wave 26,
 * n_basis 64, n_proj 2, 130 non-zeros, nnz per wave
 * `{1: 5, 3: 11, 9: 8, 10: 2}` (mean 5.0, max 10), te 8 → 208 (e,k) items on
 * 256 threads, i.e. ONE item per thread.  The phase barrier therefore waits
 * for the 10-nnz wave: max load 10 vs mean 4.1 = a **2.46x tail**.  Sorting
 * the items by nnz does NOT help (measured: identical) — only SPLITTING a
 * heavy row shortens the critical path.
 *
 * `GPS04_SPLIT_NNZ` (default 6) is the COMPILED default threshold and stays a
 * `-D` knob (`-DGPS04_SPLIT_NNZ=4`); 6 splits the 9/10-nnz kmatrix rows and
 * leaves the 1/3-nnz rows alone.  It is NOT the only way to change the
 * threshold: `cuda_create_context_flat_pwa_gen_s0e4` takes `split_nnz` as its
 * last argument (0 = this default), so a caller can sweep it with one binary:
 * the segmentation is built on the host in `create` and the kernel only reads
 * `n_seg`/`seg_*`.
 *
 * ── MEASURED, and the regime where the split pays ───────────────────────
 *
 * Interleaved A/B on the production shape (`benchmark_s0e_variants.py` and a
 * focused te sweep, ne 20 000–50 000, min of 5 rounds, one process, rsize 17
 * and 1):
 *
 *     s0e  te 8 (its own rule)     25.7 … 29.0 ns/event     1.000
 *     s0e4 T6 te 8 (36 segments)   26.9 … 30.7             0.94 … 0.95
 *     s0e4 T6 te 6                 27.6 … 30.5             0.95
 *     s0e4 T6 te 4                 30.1 … 34.0             0.85 … 0.88
 *     s0e4 T9 te 8 (28 segments)   27.5 … 31.2             0.93
 *
 * i.e. on this shape the split does NOT pay at any tile measured — it is 5–15 %
 * SLOWER, and the reason is arithmetic rather than implementation.  The tail
 * this copy targets (max 10 vs mean 4.1 non-zeros at `te = 8`) is only a
 * barrier-side cost when the tile leaves the block with fewer items than
 * threads; a tile of 8 events × 36 segments = 288 items does NOT fit
 * one-per-thread on 256, so 32 threads take a second segment and the critical
 * path stays at <= 10 non-zeros — while every (e, k) now pays one or two
 * shared-memory `atomicAdd`s plus the extra A2 pass.  Even at `te = 4`
 * (144 items, one segment per thread by construction) the atomics and the
 * second phase cost more than the shortened wait, so the win is a HYPOTHESIS
 * this measurement does not confirm: the fix would be to remove the atomics
 * (pair the two halves in phase A2 through a per-segment slot) before
 * re-testing, which is a different variant.
 *
 * `te` stays s0e's rule in this copy on purpose: the variant is about segments,
 * and re-tuning the tile belongs to a separate A/B (s0e's own te sweep already
 * measured te 8 as this shape's optimum).
 *
 * ── what s0e is, and why it is the structural base ──────────────────────
 *
 * `cuda_v4_pwa_s0e` keeps EVERY per-iteration intermediate (BW, 1/BW,
 * common, A_p, dQ/dA_p, dQ/dD) inside ONE kernel in shared memory for a tile
 * of events: no global round-trip, no per-phase kernel, no separate grad
 * launch.  `cuda_v5_pwa_s0e` adds the grouped log-sum objective on top
 * WITHOUT a second forward pass, because the group scale `-W_g/S_g` is a
 * scalar that only rescales the already-fused `dQ/dP_e`.
 *
 * The flat `pwa_gen` family (gen2, 1bw) instead runs the tile in one launch
 * and then, for `resolution_size > 1`, a second backward launch per group
 * (`mode 1 -> grp_nll -> scale -> mode 2`), i.e. the data is walked twice.
 * This kernel brings the s0e structure to the flat config, so the grouped
 * objective is closed inside the block and there is exactly ONE pass.
 *
 * ── the basis rotation (the one non-obvious piece) ──────────────────────
 *
 * s0e's denominator is `BW_w = Σ_i M_wi·F_i` with the device formula
 * `bw_r += M*Z_i; bw_i -= M*Z_r`, i.e. `F_i = -i·Z_i`.  It therefore stores a
 * PRE-ROTATED table.  The flat config instead has an arbitrary
 * `basis_table` of `n_basis` complex rows.  Rather than special-case any
 * `basis_kind`, the us-cache stores
 *
 *     Z_m(e) = i·basis_table[m](m_{e, basis_m_index[m]})
 *
 * and the kernel's own `-i·Z` recovers the basis row exactly:
 * `-i·(i·basis) = basis` (verified over all 69 rows of the real config:
 * `max|recovered - basis| = 0`).  No kind is ever inspected — every row is
 * handled by the same uniform rotation, which is why `basis_kind` does not
 * exist in this kernel's vocabulary at all.
 *
 * ── the two structural dedups (both keys are BYTE-EXACT table data) ─────
 *
 * 1. **us-cache**: the interpolation of basis row `m` at
 *    `m_{e, basis_m_index[m]}` depends only on `(row content, mass column)`.
 *    Both are byte-exact table data, so folding them is exact; sampled values
 *    are never compared.  On the real `jpsi_eta_pi_pi` config that is
 *    **69 -> 43** slots (`(row, col)` pairs: 30 distinct row byte-patterns x
 *    their mass columns).  The key is deliberately NOT `basis_kind`: that is
 *    a per-factor slot index (0 = s0, 1 = "one", 2 = width, …), so keying on
 *    it merges e.g. every resonance's running width into one slot — measured
 *    16 slots, which is WRONG (different rows).  The row's own bytes are its
 *    physical identity; the mass column is part of the key because two rows
 *    with equal content read at different event columns are different
 *    functions of the event.
 *
 * 2. **amp-cache**: `Amp = fa·fl·num` (the whole ck/coeff-independent
 *    numerator of the model) depends only on the wave's table rows:
 *    the `matrix_angle` column of EVERY projection, the `fl_table` row plus
 *    its `fl_q_index`, and the `num_table` row plus its `num_m_index`.
 *    Byte-exact key, folds **26 -> 13** waves on the real config.  Note
 *    `matrix_angle` is p-major (`column p·n_wave + k`), so the key must hold
 *    ALL `p` columns of the wave — a slot is a WAVE, and the cache carries
 *    the projection axis (`[ne, n_uniq, n_proj]`).
 *
 * ── the algebra (one pass, all of it) ───────────────────────────────────
 *
 * With `W = n_proj·n_wave` entries (`entry = p·N + k`, `N = n_wave`) and one
 * BW per chain (`n_res == 1`, enforced):
 *
 *   D_{e,k}      = Σ_{m∈nz(k)} gamma[k,m]·coeff[m]·basis_m(m_e)   (shared by p)
 *   rden         = 1/|D|²                                   (one real divide)
 *   common_{e,p,k} = Amp_{e,slot(k),p} · conj(D)/|D|²
 *   A_{e,p}      = Σ_k ck_k·common_{e,p,k}
 *   P_e          = Σ_p |A_{e,p}|²
 *   Pk_{e,k}     = Σ_p dQ/dA_{e,p}·ck_k·common_{e,p,k}
 *   dQ/dck_k     = Σ_e Pk_{e,k}                     (= Σ_p dQ/dA·common)
 *   dQ/dD_{e,k}  = -Pk_{e,k}/D_{e,k}
 *   dQ/dcoeff_m  = 2Re Σ_{k∋m} dQ/dD_{e,k}·gamma[k,m]·basis_m(m_e)
 *
 * `dQ/dck` and `dQ/dD` come from the SAME complex number `Pk` (the u-merge:
 * `common` is shared by the projections of a wave, so the two reductions are
 * one), which is why phase 4 is a single pass.  `grad_ck` is the raw
 * Wirtinger half — `CKProduct.backprop_grad` applies the 2·Re convention.
 *
 * Sparse rows: a `gamma[k]` row selects a handful of basis columns out of
 * `n_basis` (92 non-zeros over 26 waves on the real config, and 23 of 26 rows
 * have interior HOLES).  The host emits the row's TRUE non-zeros as a CSR
 * (`gs_lo[k]`/`gs_hi[k]` into `nz_m`/`nz_u`/`nz_k`), so phase 1 costs `nnz(k)`
 * and phase 5 walks the same list.  A hole is simply absent, so no `g == 0`
 * test is needed — `0 * value == 0`.
 *
 * ── the grouped log-sum, closed INSIDE the block ────────────────────────
 *
 *     q_e = w_e·(P_e/norm + bkg_e),  S_g = Σ_{e∈g} q_e,  W_g = Σ_{e∈g} w_e
 *     Q    = -Σ_g W_g·log(S_g/W_g),  dQ/dP_e = -(W_g/S_g)·(w_e/norm)
 *     dnorm = Σ_g W_g·Pb_g/(S_g·norm²),  Pb_g = Σ_{e∈g} w_e·P_e
 *
 * (`resolution_size == 1` reduces to `-Σ w_e·log(pdf_e)` exactly, so there is
 * only ONE normed code path.)  Two block shapes:
 *
 *  * **path A** (`rsize <= te`, one tile holds WHOLE groups): per-group sums
 *    live in shared memory and phase B2 folds `-W_g/S_g` into `dQ/dP_e`
 *    before the gradient phases.
 *  * **path B** (`rsize > te`, real use is 10…100): one block = one GROUP,
 *    walked internally in `te`-event chunks.  `-W_g/S_g` is only known when
 *    the group closes, so the gradient phases accumulate with the per-event
 *    factor `w_e/norm` into shared per-group accumulators and the block
 *    multiplies them by the group constant ONCE at close.  Still exactly one
 *    BW pass per event, and the group is closed inside the block — no
 *    `grp_nll`/`scale` kernel and no saved-`A` buffer exist here at all
 *    (that is 1bw/gen2's second pass, which this kernel does not need).
 *
 * ── chunking: bounded by batch_size, never by the sample ────────────────
 *
 * `cuda_v5_pwa_s0e` allocates and launches per `ne` and SEGFAULTS on the real
 * 8.75 M-event dataset.  This kernel copies gen2's/1bw's discipline instead:
 * the working scratch is the CONTEXT's, sized at create from a chunk length
 * `cap`, and `compute` walks the sample in chunks passing offset pointers plus
 * the chunk length as the kernel's `ne` — so `ne` never enters an allocation.
 * In fact NO working scratch is allocated at all: every intermediate is in
 * shared memory, so the only per-sample allocations are the two load-time
 * CACHES (`us`, `amp` — real data, not scratch) and `P_dev`.  That is why
 * this kernel's memory knob is `batch_size` alone and why it has no
 * `grp_nll`/`scale` kernels and no saved-`A` buffer.
 *
 * The chunk rule is verbatim
 *
 *     cap = batch_size > 0 ? batch_size : 50000
 *     cap = min(cap, 1<<20)
 *     if (rsize > 1) { a = (cap / rsize)·rsize ; cap = a > 0 ? a : rsize }
 *     s_ngroups = ceil(cap / rsize)
 *
 * and it is published (`cuda_cap_*`) so the wrapper ASKS instead of
 * re-deriving it.  `cap` being a multiple of `rsize` is what stops a group
 * straddling two chunks (that was a 49 % error once in gen2); inside a chunk
 * path B is what stops it straddling two TILES.
 *
 * The ONE per-event output buffer that stays on the handle is `P_dev`
 * (`ne` doubles = 70 MB at 8.75 M events, ~1 % of the caches): `P` is the
 * caller-visible OUTPUT of the call, not working scratch, and it is allocated
 * ONCE per `load_data` and written by every chunk.  It is copied out with one
 * D2H at the end of `compute` — not per chunk — which is why there is no
 * per-chunk host sync anywhere in the hot loop.
 *
 * The load-time fills upload `mass`/`mom`/`ang` as TRANSIENT device copies and
 * free them immediately, exactly as gen2/1bw do.  (Reading them through the
 * host pointers happens to work on a device with host page-table access —
 * verified on this Blackwell part — but it is not portable and it pins the
 * numpy pages for the load; the numbers were identical either way, so the
 * portable form is used.)
 *
 * ── shared memory is the binding constraint ────────────────────────────
 *
 * The tile keeps
 *
 *   2·te·W (cr,ci) + 2·te·N (D) + te·N (rd) + 2·te·N (pv)
 *   + 4·te·P (A,dA) + 3·te (pe,q,sc) + 2N + NB (gck,gc) + 3 (tot)
 *   + N + N + NB (gr,gi,gcb)                                  doubles
 *
 * (`gps04_layout` is the ONE definition; the host asks the context rather than
 * mirroring the formula).  On the real shape at `te = 8` that is 17 640 B —
 * under the 48 KiB DEFAULT, so the launch needs no opt-in there.  The opt-in
 * is still applied (raise-only monotonic, per FUNCTION not per context, so two
 * live contexts of different tile sizes cannot fight) for shapes that cross
 * it, and a tile that exceeds the device's opt-in limit is REFUSED by `create`
 * rather than launched into a silent `Q = 0`.
 *
 * `te` is chosen in `gps04_pick_te` from the shape, and the rule is measured
 * rather than assumed.  On the real 8.75 M-event `jpsi_eta_pi_pi` sample
 * (rsize 17, `Fitter.get_nll`, min of 4, one process per setting):
 *
 *     te      1      2      4      8     17      (all beyond rsize -> path B)
 *     ms    476    385    344    325    333
 *
 * A small tile with ONE BLOCK PER GROUP beats a 17-event tile that holds
 * whole groups: better occupancy from the smaller shared tile outweighs the
 * group-walk bookkeeping.  So `te` stays near `GPS04_DEF_TE`, rounded UP to a
 * whole number of groups only when `rsize <= GPS04_DEF_TE` (where path A costs
 * nothing).  On the 100-wave benchmark shape the same rule shrinks `te` by the
 * shared-memory budget, which is that shape's measured optimum as well.
 *
 * ── measured ───────────────────────────────────────────────────────────
 *
 * Real dataset (8 746 347 data + 1 488 104 phsp, rsize 17, one Fitter per
 * process), `Fitter.get_nll`:
 *
 *     NLL = -624783.229197 at batch_size 5000 / 50000 / 200000
 *
 * the same value `cuda_flat_pwa_gen_1bw` and `cuda_flat_pwa_gen2` give on the
 * same sample (the record fit value is -624783.2042846916; the difference is
 * the fit's own convergence, not the objective).  Timing: **324 ms** against
 * 1bw's 408 ms and gen2's 1521 ms — 1.26x faster than the previous best on
 * the one shape where it matters — with 6.26 GB free after the data load
 * (1bw: 2.09 GB).
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstring>
#include <cmath>

#define GPS04_BLOCK       256      /* threads per block (multiple of 32) */
#ifndef GPS04_SPLIT_NNZ
#  define GPS04_SPLIT_NNZ 6      /* measured: splits the 9/10-nnz kmatrix rows, leaves 1/3 alone */
#endif
#define GPS04_DEF_TE      8        /* default events per tile */
#define GPS04_MAX_CAP     (1 << 20)  /* ceiling on the chunk length  */
/*: soft shared-memory budget for choosing `te`.  Below this the launch needs
 *  no opt-in (so no `cudaFuncSetAttribute` and the best occupancy); above it
 *  `create` shrinks `te` first and only then crosses the opt-in. */
#define GPS04_SMEM_BUDGET (48 * 1024)

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = call;                                             \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                               \
        }                                                                    \
    } while (0)

/* ── Catmull-Rom (the repo-wide idiom, identical to gen2/1bw) ─────────── */
__device__ double gps04_cr1d(double pm1, double p0, double p1, double p2,
                            double t) {
    return p0 + 0.5 * t * (-pm1 + p1 + t * (2.0 * pm1 - 5.0 * p0 + 4.0 * p1
                                            - p2 + t * (-pm1 + 3.0 * p0
                                                        - 3.0 * p1 + p2)));
}

__device__ double gps04_interp_real(const double* __restrict__ t, int row,
                                   double x, double xmin, double xdelta,
                                   int bins) {
    double diff = (x - xmin) / xdelta;
    int b = max(0, min((int)floor(diff), bins - 2));
    double s = max(0.0, min(diff - b, 1.0));
    int base = row * bins + b;
    int lo = row * bins, hi = (row + 1) * bins - 1;
    int im1 = base > lo ? base - 1 : base;
    int i2 = base + 2 <= hi ? base + 2 : base + 1;
    return gps04_cr1d(t[im1], t[base], t[base + 1], t[i2], s);
}

__device__ void gps04_interp_c(const double* __restrict__ tr,
                              const double* __restrict__ ti, int row,
                              double x, double xmin, double xdelta, int bins,
                              double* orr, double* oii) {
    double diff = (x - xmin) / xdelta;
    int b = max(0, min((int)floor(diff), bins - 2));
    double s = max(0.0, min(diff - b, 1.0));
    int base = row * bins + b;
    int lo = row * bins, hi = (row + 1) * bins - 1;
    int im1 = base > lo ? base - 1 : base;
    int i2 = base + 2 <= hi ? base + 2 : base + 1;
    *orr = gps04_cr1d(tr[im1], tr[base], tr[base + 1], tr[i2], s);
    *oii = gps04_cr1d(ti[im1], ti[base], ti[base + 1], ti[i2], s);
}

/*=============================================================================
 * LOAD-TIME FILL 1 — the us-cache: Z = i·basis_table[m] at the event mass.
 *
 * One thread per (event, uslot).  `uslot_row[u]` / `uslot_col[u]` are the
 * deduplicated `(table row, mass column)` pair; nothing here depends on a
 * fitted parameter, so this is computed ONCE per `load_data` and reused by
 * every iteration.  Stored as float2 (`Z` is complex) — the same fp32 cache
 * floor the sibling flat kernels are gated at.
 *===========================================================================*/
__global__ void gps04_fill_us(
    const double* __restrict__ bas_r, const double* __restrict__ bas_i,
    const int* __restrict__ uslot_row, const int* __restrict__ uslot_col,
    double m_min, double m_delta, int m_bins,
    const double* __restrict__ mass, int n_mass,
    int ne, int n_uslot, float2* __restrict__ us)
{
    const long long total = (long long)ne * (long long)n_uslot;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         g < total; g += stride) {
        const int e = (int)(g / n_uslot);
        const int u = (int)(g - (long long)e * n_uslot);
        const double mv = mass[(size_t)e * n_mass + uslot_col[u]];
        double vr, vi;
        gps04_interp_c(bas_r, bas_i, uslot_row[u], mv,
                      m_min, m_delta, m_bins, &vr, &vi);
        /* Z = i·basis = (-basis_i, basis_r);  the kernel's `-i·Z` is `basis`. */
        us[g] = make_float2((float)(-vi), (float)vr);
    }
}

/*=============================================================================
 * LOAD-TIME FILL 2 — the amp-cache: Amp = fa·fl·num per (event, wave, p).
 *
 * The angular basis `ka_a = Π_j cos(angle·angle_k + angle_b)` is evaluated
 * once per (event, basis row) in shared memory and reused by every
 * projection and every deduplicated wave slot — the flat family's fill
 * kernels recompute it per (wave, projection), which is 26·2 = 52x the work
 * for the same numbers.
 *
 * `n_num == 0` (a config with no numerator rows) is not a special case here:
 * `num_table` always has at least the constant row, so `n_res >= 1`.
 *===========================================================================*/
__global__ void gps04_fill_amp(
    const double* __restrict__ fl_table, const int* __restrict__ fl_q_index,
    const double* __restrict__ num_r, const double* __restrict__ num_i,
    const int* __restrict__ num_m_index,
    double m_min, double m_delta, int m_bins,
    double q_min, double q_delta, int q_bins,
    const int* __restrict__ angle_index, const double* __restrict__ angle_k,
    const double* __restrict__ angle_b,
    const double* __restrict__ am_r, const double* __restrict__ am_i,
    const double* __restrict__ mass, const double* __restrict__ mom,
    const double* __restrict__ ang,
    const int* __restrict__ rep_of_slot,
    int ne, int n_uniq, int n_wave, int n_proj, int n_res, int n_decay,
    int n_ab, int n_comp, int n_angle_total, int n_mass, int n_momentum,
    float2* __restrict__ amp)
{
    const int e = blockIdx.x;
    if (e >= ne) return;
    const int tid = threadIdx.x, nt = blockDim.x;

    extern __shared__ double ska[];
    for (int a = tid; a < n_ab; a += nt) {
        const int col = angle_index[a];
        double prod = 1.0;
        for (int j = 0; j < n_comp; j++) {
            prod *= cos(ang[((size_t)e * n_angle_total + col) * n_comp + j]
                        * angle_k[a * n_comp + j]
                        + angle_b[a * n_comp + j]);
        }
        ska[a] = prod;
    }
    __syncthreads();

    const int PW = n_proj * n_wave;
    for (int s = tid; s < n_uniq; s += nt) {
        const int k = rep_of_slot[s];
        /* numerator: Π_r num_table[k, r](m_{e, num_m_index[k, r]}) */
        double nr = 1.0, ni = 0.0;
        for (int r = 0; r < n_res; r++) {
            const double mv =
                mass[(size_t)e * n_mass + num_m_index[k * n_res + r]];
            double vr, vi;
            gps04_interp_c(num_r, num_i, k * n_res + r, mv,
                          m_min, m_delta, m_bins, &vr, &vi);
            const double tr = nr * vr - ni * vi;
            ni = nr * vi + ni * vr;
            nr = tr;
        }
        /* barrier: Π_d fl_table[k, d](q_{e, fl_q_index[k, d]}) */
        double fl = 1.0;
        for (int d = 0; d < n_decay; d++) {
            const double qv =
                mom[(size_t)e * n_momentum + fl_q_index[k * n_decay + d]];
            fl *= gps04_interp_real(fl_table, k * n_decay + d, qv,
                                   q_min, q_delta, q_bins);
        }
        const double Fr = fl * nr, Fi = fl * ni;
        for (int p = 0; p < n_proj; p++) {
            double far = 0.0, fai = 0.0;
            for (int a = 0; a < n_ab; a++) {
                const double ka = ska[a];
                /* matrix_angle is p-major: column p·n_wave + k, row a */
                const size_t o = (size_t)a * PW + (size_t)(p * n_wave + k);
                far += ka * am_r[o];
                fai += ka * am_i[o];
            }
            amp[((size_t)e * n_uniq + s) * n_proj + p] =
                make_float2((float)(Fr * far - Fi * fai),
                            (float)(Fr * fai + Fi * far));
        }
    }
}

/*=============================================================================
 * The shared tile — ONE formula, used by the host (`create`) and the kernel
 * (the pointer offsets), so the launch size and the kernel's view cannot
 * disagree.  All offsets are in DOUBLES.
 *
 *   cr, ci  [te·W]     common_{e,p,k} = Amp·conj(D)/|D|²   (W = P·N, p-major)
 *   D       [2·te·N]   the finished denominator D_{e,k}
 *   rd      [te·N]     1/|D|²  (one real reciprocal for every division)
 *   pv      [2·te·N]   dQ/dD_{e,k}
 *   A, dA   [2·te·P]   A_{e,p} and dQ/dA_{e,p}
 *   pe, q, sc [te]     P_e, the group summand q_e, the group scale
 *   gck     [2·N]      per-wave shared accumulator for dQ/dck
 *   gc      [NB]       per-coefficient shared accumulator (GLOBAL column axis)
 *   tot     [3]        path B: S_g, Pb_g, W_g totals
 *   gr, gi  [N]        path B: per-group dQ/dck accumulators
 *   gcb     [NB]       path B: per-group dQ/dcoeff accumulator
 *===========================================================================*/
typedef struct {
    int cr, ci, D, rd, pv, A, dA, pe, q, sc, gck, gc, tot, gr, gi, gcb, total;
} GPS04Layout;

__host__ __device__ __forceinline__ GPS04Layout gps04_layout(
    int te, int W, int P, int N, int NB)
{
    GPS04Layout L;
    int o = 0;
    L.cr  = o; o += te * W;
    L.ci  = o; o += te * W;
    L.D   = o; o += 2 * te * N;
    L.rd  = o; o += te * N;
    L.pv  = o; o += 2 * te * N;
    L.A   = o; o += 2 * te * P;
    L.dA  = o; o += 2 * te * P;
    L.pe  = o; o += te;
    L.q   = o; o += te;
    L.sc  = o; o += te;
    L.gck = o; o += 2 * N;
    L.gc  = o; o += NB;
    L.tot = o; o += 3;
    L.gr  = o; o += N;
    L.gi  = o; o += N;
    L.gcb = o; o += NB;
    L.total = o;
    return L;
}

/*=============================================================================
 * The per-iteration tile: phases A -> B -> G -> B2 -> C -> D, one pass.
 *===========================================================================*/
__global__ void gps04_tile_kernel(
    const float2* __restrict__ us,         /* [ne, n_uslot]   Z = i·basis   */
    const float2* __restrict__ amp,        /* [ne, n_uniq, P] fa·fl·num     */
    const int* __restrict__ slot_of_wave,  /* [n_wave] -> amp slot          */
    const int* __restrict__ nz_m,          /* [nnz] basis column            */
    const int* __restrict__ nz_u,          /* [nnz] -> uslot                */
    const int* __restrict__ nz_k,          /* [nnz] wave                    */
    const int* __restrict__ gs_lo,         /* [n_wave] CSR start into nz_*  */
    const int* __restrict__ gs_hi,         /* [n_wave] CSR end              */
    const double* __restrict__ gamma,      /* [n_wave·n_basis] (n_res == 1) */
    int n_nz,
    /* the segments: a row with nnz > GPS04_SPLIT_NNZ is two contiguous
       halves (part 0 and 1), everything else is one whole row (part 0) */
    const int* __restrict__ seg_k, const int* __restrict__ seg_lo,
    const int* __restrict__ seg_hi, const int* __restrict__ seg_part,
    int n_seg,
    const double* __restrict__ cw,         /* [n_nz] coeff[nz_m[i]] (host-gathered) */
    const double* __restrict__ ck_r, const double* __restrict__ ck_i,
    const double* __restrict__ weight, const double* __restrict__ bkg,
    double norm, int use_norm,
    int rsize, int per_blk, int te,
    int N, int P, int NB, int n_uslot, int n_uniq,
    int ne,
    double* __restrict__ P_dev,            /* [ne]  per-event P (output) */
    double* __restrict__ g_ck_r, double* __restrict__ g_ck_i,
    double* __restrict__ g_coeff,
    double* __restrict__ Q_acc, double* __restrict__ dnorm_acc)
{
    const int tid = threadIdx.x, nt = blockDim.x;
    const int W = P * N;
    const GPS04Layout L = gps04_layout(te, W, P, N, NB);

    extern __shared__ double sh[];
    double* s_cr  = sh + L.cr;
    double* s_ci  = sh + L.ci;
    double* s_D   = sh + L.D;
    double* s_rd  = sh + L.rd;
    double* s_pv  = sh + L.pv;
    double* s_A   = sh + L.A;
    double* s_dA  = sh + L.dA;
    double* s_pe  = sh + L.pe;
    double* s_q   = sh + L.q;
    double* s_sc  = sh + L.sc;
    double* s_gck = sh + L.gck;
    double* s_gc  = sh + L.gc;
    double* s_tot = sh + L.tot;
    double* s_gr  = sh + L.gr;
    double* s_gi  = sh + L.gi;
    double* s_gcb = sh + L.gcb;

    const int e0 = blockIdx.x * per_blk;
    const int evn = min(per_blk, ne - e0);
    if (evn <= 0) return;
    /* path B: one block owns ONE whole group and walks it internally */
    const int pathB = (per_blk > te);
    const int rs = (rsize < 1) ? 1 : rsize;
    const int nchunk = pathB ? ((evn + te - 1) / te) : 1;

    /* gradient targets: the global accumulators (path A) or the block's
       per-group accumulators (path B, scaled once at group close) */
    double* d_ck_r = g_ck_r;
    double* d_ck_i = g_ck_i;
    double* d_co = g_coeff;
    if (pathB) {
        d_ck_r = s_gr; d_ck_i = s_gi; d_co = s_gcb;
        for (int k = tid; k < N; k += nt) { s_gr[k] = 0.0; s_gi[k] = 0.0; }
        for (int m = tid; m < NB; m += nt) s_gcb[m] = 0.0;
        if (tid == 0) { s_tot[0] = 0.0; s_tot[1] = 0.0; s_tot[2] = 0.0; }
    }
    for (int k = tid; k < 2 * N; k += nt) s_gck[k] = 0.0;
    for (int m = tid; m < NB; m += nt) s_gc[m] = 0.0;
    __syncthreads();

    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int nwarps = nt >> 5;

    for (int c = 0; c < nchunk; c++) {
        const int ce0 = e0 + c * te;
        int nen = pathB ? (evn - c * te) : evn;
        if (nen > te) nen = te;

        /* ── phase A: D_{e,k}, accumulated per SEGMENT ────────────────
         * A row with nnz > GPS04_SPLIT_NNZ is two segments handled by two
         * threads (part 1 folds in with an atomicAdd), so the barrier
         * waits for ~nnz/2 instead of nnz.  `cw` is the fitted
         * coefficient pre-gathered in CSR order by the host: the inner
         * loop has no coefficient indirection. */
        for (int i = tid; i < nen * N * 2; i += nt) s_D[i] = 0.0;
        __syncthreads();
        for (int idx = tid; idx < nen * n_seg; idx += nt) {
            const int e = idx / n_seg, s = idx - e * n_seg;
            const int k = seg_k[s];
            const float2* uu = us + (size_t)(ce0 + e) * n_uslot;
            double sr = 0.0, si = 0.0;
            for (int i = seg_lo[s]; i < seg_hi[s]; i++) {
                const double cf = cw[i];
                const float2 z = uu[nz_u[i]];
                sr += cf * (double)z.y;
                si -= cf * (double)z.x;
            }
            const int o = 2 * (e * N + k);
            atomicAdd(&s_D[o], sr);
            atomicAdd(&s_D[o + 1], si);
        }
        __syncthreads();
        /* ── phase A2: 1/|D|² and common_{e,p,k} ─────────────────────
         * The divide is now its own pass (no long chain in front of it),
         * and it is still ONE correctly-rounded divide per (e,k). */
        for (int idx = tid; idx < nen * N; idx += nt) {
            const int e = idx / N, k = idx - e * N;
            const double sr = s_D[2 * idx], si = s_D[2 * idx + 1];
            const double rden = 1.0 / (sr * sr + si * si);
            s_rd[idx] = rden;
            const size_t abase =
                (size_t)(ce0 + e) * n_uniq * P + (size_t)slot_of_wave[k] * P;
            for (int p = 0; p < P; p++) {
                const float2 a = amp[abase + p];
                const int o = (e * P + p) * N + k;
                s_cr[o] = ((double)a.x * sr + (double)a.y * si) * rden;
                s_ci[o] = ((double)a.y * sr - (double)a.x * si) * rden;
            }
        }
        __syncthreads();

        /* ── phase B: A_{e,p}, P_e, the group summand q_e ─────────────────
         * One warp per event; the k reduction is a warp shuffle (no shared
         * atomics, unlike the gen2/1bw shape). */
        for (int e = wid; e < nen; e += nwarps) {
            for (int p = 0; p < P; p++) {
                const double* cr = s_cr + (e * P + p) * N;
                const double* ci = s_ci + (e * P + p) * N;
                double ar = 0.0, ai = 0.0;
                for (int k = lane; k < N; k += 32) {
                    const double ckr = ck_r[k], cki = ck_i[k];
                    ar += ckr * cr[k] - cki * ci[k];
                    ai += ckr * ci[k] + cki * cr[k];
                }
                for (int off = 16; off > 0; off >>= 1) {
                    ar += __shfl_down_sync(0xffffffffu, ar, off);
                    ai += __shfl_down_sync(0xffffffffu, ai, off);
                }
                if (lane == 0) {
                    s_A[(e * P + p) * 2 + 0] = ar;
                    s_A[(e * P + p) * 2 + 1] = ai;
                }
            }
            __syncwarp();
            if (lane == 0) {
                double Pe = 0.0;
                for (int p = 0; p < P; p++) {
                    const double ar = s_A[(e * P + p) * 2 + 0];
                    const double ai = s_A[(e * P + p) * 2 + 1];
                    Pe += ar * ar + ai * ai;
                }
                const int ge = ce0 + e;
                const double wv = weight[ge], bv = bkg[ge];
                s_pe[e] = Pe;
                s_q[e] = use_norm ? wv * (Pe / norm + bv) : wv * Pe;
                P_dev[ge] = Pe;
            }
        }
        __syncthreads();

        /* ── phase G: the group summaries ──────────────────────────────────
         * path A: whole groups live in this tile -> s_sc[g] holds -W_g/S_g.
         * path B: thread 0 folds this inner chunk into the block's totals;
         * the scale exists only at group close. */
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
                    if (W > 0.0) {              /* a zero-weight group is inert */
                        s_sc[g] = -W / S;
                        atomicAdd(Q_acc, -W * log(S / W));
                        atomicAdd(dnorm_acc, W * WP / (S * norm * norm));
                    } else {
                        s_sc[g] = 0.0;
                    }
                } else {
                    s_sc[g] = 1.0;              /* linear objective */
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
            s_tot[0] += S; s_tot[1] += WP; s_tot[2] += W;
        }
        __syncthreads();

        /* ── phase B2: dQ/dP_e -> dQ/dA_p ─────────────────────────────────
         * path B keeps the UNSCALED per-event factor: the group constant is
         * applied to the finished accumulators at close. */
        for (int idx = tid; idx < nen * P; idx += nt) {
            const int e = idx / P, p = idx - e * P;
            const double wv = weight[ce0 + e];
            double dqp = use_norm ? (wv / norm) : wv;
            if (!pathB) dqp *= s_sc[e / rs];
            const double ar = s_A[(e * P + p) * 2 + 0];
            const double ai = s_A[(e * P + p) * 2 + 1];
            s_dA[idx * 2 + 0] = dqp * ar;
            s_dA[idx * 2 + 1] = -dqp * ai;
        }
        __syncthreads();

        /* ── phase C: u_{e,k} = Σ_p dQ/dA_{e,p}·common_{e,p,k}  (= dQ/dck_k)
         *            dQ/dD_{e,k} = -ck_k·u_{e,k}/D_{e,k}
         *
         * The two reductions the split kernels do in separate passes are the
         * same complex number here, because `common` is shared by the `p` of
         * a wave (the u-merge, at one BW per chain).  `dQ/dD` carries the
         * `ck` factor — `common` is linear in the amplitude and `D` is not. */
        for (int idx = tid; idx < nen * N; idx += nt) {
            const int e = idx / N, k = idx - e * N;
            const double ckr = ck_r[k], cki = ck_i[k];
            double ur = 0.0, ui = 0.0;
            for (int p = 0; p < P; p++) {
                const int o = (e * P + p) * N + k;
                const double dr = s_dA[(e * P + p) * 2 + 0];
                const double di = s_dA[(e * P + p) * 2 + 1];
                const double cr = s_cr[o], ci = s_ci[o];
                ur += dr * cr - di * ci;
                ui += dr * ci + di * cr;
            }
            atomicAdd(&s_gck[2 * k + 0], ur);
            atomicAdd(&s_gck[2 * k + 1], ui);
            const double dr = s_D[idx * 2 + 0], di = s_D[idx * 2 + 1];
            const double rden = s_rd[idx];
            const double qr = ckr * ur - cki * ui;   /* ck·u */
            const double qi = ckr * ui + cki * ur;
            s_pv[idx * 2 + 0] = -(qr * dr + qi * di) * rden;  /* -ck·u/D */
            s_pv[idx * 2 + 1] = -(qi * dr - qr * di) * rden;
        }
        __syncthreads();

        /* ── phase D: dQ/dcoeff_m = 2Re Σ dQ/dD_{e,k}·gamma[k,m]·basis_m(m_e)
         *
         * Walked over the host-built CSR of TRUE non-zeros — one thread per
         * entry, serial over the tile — so a folded or holed row costs its
         * non-zeros, not its span (92 entries over 26 waves on the real
         * config, with interior holes in 23 of them).
         *
         * `Z` is stored rotated (`Z = i·basis`), so `F_m = -i·Z_m` and
         * `2Re(dQ/dD·F_m) = 2·(pv_r·Z_i + pv_i·Z_r)`. */
        for (int idx = tid; idx < n_nz; idx += nt) {
            const int k = nz_k[idx], m = nz_m[idx], u = nz_u[idx];
            const double g = gamma[k * NB + m];
            double out = 0.0;
            for (int e = 0; e < nen; e++) {
                const float2 z = us[(size_t)(ce0 + e) * n_uslot + u];
                const size_t o = (size_t)(e * N + k) * 2;
                out += 2.0 * (s_pv[o + 0] * (double)z.y
                              + s_pv[o + 1] * (double)z.x);
            }
            if (out != 0.0) atomicAdd(&s_gc[m], out * g);
        }
        __syncthreads();

        if (pathB) {
            /* fold this inner chunk into the block's group accumulators
               (plain adds: every index has one owner after the barrier) */
            for (int k = tid; k < N; k += nt) {
                s_gr[k] += s_gck[2 * k + 0];
                s_gi[k] += s_gck[2 * k + 1];
                s_gck[2 * k + 0] = 0.0;
                s_gck[2 * k + 1] = 0.0;
            }
            for (int m = tid; m < NB; m += nt) {
                s_gcb[m] += s_gc[m];
                s_gc[m] = 0.0;
            }
            __syncthreads();   /* the tile is reused by the next inner chunk */
        }
    }

    /* ── flush ────────────────────────────────────────────────────────────
     * path A: the accumulators are complete (one tile = whole groups).
     * path B: close the group, then rescale the finished accumulators by
     *         const_g = -W_g/S_g (0 for a zero-weight group). */
    if (pathB) {
        if (tid == 0) {
            const double S = s_tot[0], WP = s_tot[1], W = s_tot[2];
            if (use_norm) {
                if (W > 0.0) {
                    atomicAdd(Q_acc, -W * log(S / W));
                    atomicAdd(dnorm_acc, W * WP / (S * norm * norm));
                    s_sc[0] = -W / S;
                } else {
                    s_sc[0] = 0.0;
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
        for (int m = tid; m < NB; m += nt)
            if (s_gcb[m] != 0.0) atomicAdd(g_coeff + m, cs * s_gcb[m]);
    } else {
        for (int k = tid; k < N; k += nt) {
            atomicAdd(g_ck_r + k, s_gck[2 * k + 0]);
            atomicAdd(g_ck_i + k, s_gck[2 * k + 1]);
        }
        for (int m = tid; m < NB; m += nt)
            if (s_gc[m] != 0.0) atomicAdd(g_coeff + m, s_gc[m]);
    }
}

/*=============================================================================
 * Host side
 *===========================================================================*/
typedef struct {
    int *fl_q_index, *num_m_index, *bas_m_index, *angle_index;
    int *uslot_row, *uslot_col, *slot_of_wave, *rep_of_slot;
    int *nz_k, *nz_m, *nz_u, *gs_lo, *gs_hi;
    /* the phase-A segments (built in `create` from gs_lo/gs_hi and the
       create-time `split_nnz` threshold; the kernel only reads them) */
    int *seg_k, *seg_lo, *seg_hi, *seg_part;
    int *h_seg_k, *h_seg_lo, *h_seg_hi, *h_seg_part;   /* host mirrors */
    int n_seg, split_nnz;
    int *h_nz_m;                      /* host copy of nz_m (CSR order)   */
    double *h_cw, *p_cw;              /* [n_nz] pre-gathered coefficient */
    double *fl_table, *num_r, *num_i, *bas_r, *bas_i, *gamma;
    double *angle_k, *angle_b, *am_r, *am_i;
    double m_min, m_delta, q_min, q_delta;
    int m_bins, q_bins;
    int n_wave, n_proj, n_res, n_decay, n_basis;
    int n_ab, n_comp, n_mass, n_momentum, n_angle_total;
    int n_uniq, n_uslot, n_nz;
    int batch_size, rsize;
    /* chunk rule (one definition, in `create`) */
    int cap, s_ngroups;
    /* tile size chosen at create (depends on rsize + the shape) */
    int te;
    int smem_bytes;
    /* parameter staging (refreshed per call, never malloc'd per call) */
    double *p_ck_r, *p_ck_i, *p_coeff;
    double *h_ck_r, *h_ck_i, *h_coeff;
    double *acc, *h_acc;              /* [Q, dnorm] */
    double *g_ck_r, *g_ck_i, *g_coeff;
    cudaStream_t stream;
} GPS04Ctx;

/* ONE SAMPLE.  Per-EVENT data only: the two caches, weight/bkg, and the
   caller-visible per-event P OUTPUT (not scratch — see the file header). */
typedef struct {
    double *weight, *bkg;
    float2 *us;           /* [ne, n_uslot]        Z = i·basis        */
    float2 *amp;          /* [ne, n_uniq, n_proj] fa·fl·num          */
    double *P_dev;        /* [ne] per-event P (output of the call)   */
    double *h_P;          /* pinned staging for the one D2H          */
    int ne;
} GPS04Handle;

static int* _gps04_up_i(const int* src, int n) {
    if (n <= 0 || src == NULL) return NULL;
    int* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(int),
                          cudaMemcpyHostToDevice));
    return d;
}

static double* _gps04_up_d(const double* src, int n) {
    if (n <= 0 || src == NULL) return NULL;
    double* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(double),
                          cudaMemcpyHostToDevice));
    return d;
}

/* The chunk rule.  ONE definition, asked for through `cuda_cap_*`. */
static int gps04_chunk(int batch_size, int rsize)
{
    int cap = (batch_size > 0) ? batch_size : 50000;
    if (cap > GPS04_MAX_CAP) cap = GPS04_MAX_CAP;
    if (rsize > 1) {
        const int a = (cap / rsize) * rsize;
        cap = (a > 0) ? a : rsize;   /* never 0, never unaligned to a group */
    }
    return cap;
}

/* The tile size.  Goals, in order: (1) whole groups per tile when `rsize` is
 * small enough (path A — the group sums then live in the tile); (2) a tile of
 * at least ~GPS04_DEF_TE events so a block has real work; (3) the shared tile
 * inside GPS04_SMEM_BUDGET, which is what actually binds.
 *
 * MEASURED on the real 8.75 M-event `jpsi_eta_pi_pi` sample (n_wave 26,
 * n_proj 2, n_basis 69, rsize 17, `Fitter.get_nll`, min of 4):
 *
 *     te   1     2     4     8     17
 *     ms   476   385   344   325   333
 *
 * i.e. a tile of GPS04_DEF_TE events with path B (one block per group, walked
 * internally in 8-event chunks) beats a 17-event tile that holds whole groups
 * — smaller shared memory means better occupancy, and that outweighs the
 * group-walk bookkeeping.  So `te` is kept near GPS04_DEF_TE and only rounded
 * UP to a whole number of groups when `rsize <= GPS04_DEF_TE` (where path A is
 * free).  On the 100-wave benchmark shape the same rule shrinks te to 4 by the
 * shared-memory budget, which is that shape's measured optimum too (6.04 vs
 * 6.08 ms at te=8, 5.66 ms with rsize 17). */
static int gps04_pick_te(int rsize, int W, int P, int N, int NB)
{
    int te;
    if (rsize > 1 && rsize <= GPS04_DEF_TE) {
        const int k = (GPS04_DEF_TE + rsize - 1) / rsize;   /* ceil */
        te = rsize * (k < 1 ? 1 : k);                     /* whole groups */
    } else {
        te = GPS04_DEF_TE;         /* rsize == 1, or path B (rsize > DEF_TE) */
    }
    /* `gps04_launch` floors a tile to whole groups on its own, so shrinking is
     * always valid: below `rsize` the block simply owns one group (path B). */
    while (te > 1) {
        const size_t sm =
            (size_t)gps04_layout(te, W, P, N, NB).total * sizeof(double);
        if (sm <= (size_t)GPS04_SMEM_BUDGET) break;
        te -= 1;
    }
    /* The budget shrink must not leave a tile that straddles a log-sum group:
       `create` rejects `te >= rsize` with `te % rsize != 0`, and before this
       floor the picker could return exactly that (rsize 2..7 on a shape whose
       8-event tile exceeds the budget), which made a legal `resolution_size`
       unconstructible with a misleading "context creation failed".  Flooring
       only shrinks the tile, so the budget stays satisfied; below `rsize` the
       block owns a single group (path B), which is legal. */
    if (rsize > 1 && te >= rsize) te -= te % rsize;
    return te < 1 ? 1 : te;
}

extern "C" {

void cuda_free_context_flat_pwa_gen_s0e4(void* vctx);   /* fwd decl */

void* cuda_create_context_flat_pwa_gen_s0e4(
    const double* fl_table, int n_fl,
    const int* fl_q_index, int n_fq,
    const double* num_r, const double* num_i, int n_num,
    const int* num_m_index, int n_nm,
    const double* bas_r, const double* bas_i, int n_bas,
    const int* bas_m_index, int n_bm,
    const int* uslot_row, int n_ur, const int* uslot_col, int n_uc,
    const int* slot_of_wave, int n_sw, const int* rep_of_slot, int n_rp,
    int n_uniq, int n_uslot,
    const int* nz_k, int n_nk, const int* nz_m, int n_nzm,
    const int* nz_u, int n_nzu,
    const int* gs_lo, int n_lo, const int* gs_hi, int n_hi,
    const double* gamma, int n_g,
    const int* angle_index, int n_ai,
    const double* angle_k, int n_ak, const double* angle_b, int n_ab_,
    const double* am_r, const double* am_i, int n_am,
    double m_min, double m_delta, int m_bins,
    double q_min, double q_delta, int q_bins,
    int n_wave, int n_proj, int n_res, int n_decay, int n_basis,
    int n_ab, int n_comp, int n_mass, int n_momentum, int n_angle_total,
    int batch_size, int rsize, int te_hint, int split_nnz)
{
    if (n_res != 1) {
        /* the wrapper rejects this first (with the general backend named);
           the kernel refuses too rather than reading a 3-D gamma as 1 factor */
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0e4: n_res must be 1, got %d — use "
                "cuda_flat_pwa_gen2 for multi-BW chains\n", n_res);
        return NULL;
    }
    if (n_sw != n_wave || n_rp != n_uniq) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0e4: slot_of_wave/rep_of_slot length "
                "mismatch (%d vs n_wave %d, %d vs n_uniq %d)\n",
                n_sw, n_wave, n_rp, n_uniq);
        return NULL;
    }
    if (n_nk != n_nzm || n_nk != n_nzu || n_lo != n_wave || n_hi != n_wave) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0e4: nz_*/gs_* length mismatch "
                "(nz %d/%d/%d, gs %d/%d, n_wave %d)\n",
                n_nk, n_nzm, n_nzu, n_lo, n_hi, n_wave);
        return NULL;
    }
    if (n_uniq <= 0 || n_uslot <= 0 || n_wave <= 0 || n_basis <= 0) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0e4: empty shape (n_uniq %d, n_uslot %d, "
                "n_wave %d, n_basis %d)\n", n_uniq, n_uslot, n_wave, n_basis);
        return NULL;
    }
    if (split_nnz < 0) {
        /* a negative threshold has no meaning: every row would split into
           two halves of a negative width */
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0e4: split_nnz must be >= 0 (0 = the "
                "built-in default %d), got %d\n", GPS04_SPLIT_NNZ, split_nnz);
        return NULL;
    }
    GPS04Ctx* c = (GPS04Ctx*)calloc(1, sizeof(GPS04Ctx));
    if (!c) return NULL;

    c->n_wave = n_wave;
    c->n_proj = (n_proj > 0) ? n_proj : 1;
    c->n_res = n_res;
    c->n_decay = n_decay;
    c->n_basis = n_basis;
    c->n_ab = n_ab;
    c->n_comp = n_comp;
    c->n_mass = n_mass;
    c->n_momentum = n_momentum;
    c->n_angle_total = n_angle_total;
    c->n_uniq = n_uniq;
    c->n_uslot = n_uslot;
    c->n_nz = n_nk;
    c->rsize = (rsize > 0) ? rsize : 1;
    c->batch_size = batch_size;

    /* ── the phase-A segments, and the pre-gathered coefficient layout ────
     *
     * The threshold is a CREATE-TIME parameter (`split_nnz`, 0 = the compiled
     * default GPS04_SPLIT_NNZ), because the segmentation is decided here and
     * the kernel only reads `n_seg`/`seg_*` — so changing it costs one create,
     * never a rebuild.  A row with `nnz > T` becomes TWO contiguous halves
     * (part 0 = [lo, mid), part 1 = [mid, hi), `mid = lo + nnz/2`) handled by
     * two threads; every other row is ONE segment (part 0).  The halves tile
     * the row exactly, so `Σ (seg_hi - seg_lo) == n_nz` and phase A sums the
     * same `cw[i]·Z_{nz_u[i]}` products, only in a different order. */
    c->split_nnz = (split_nnz > 0) ? split_nnz : GPS04_SPLIT_NNZ;
    {
        /* at most TWO segments per row, so 2·n_wave is the worst case */
        const size_t cap = (size_t)(2 * n_wave);
        int* h_k = (int*)malloc(cap * sizeof(int));
        int* h_lo = (int*)malloc(cap * sizeof(int));
        int* h_hi = (int*)malloc(cap * sizeof(int));
        int* h_pa = (int*)malloc(cap * sizeof(int));
        if (!h_k || !h_lo || !h_hi || !h_pa) {
            free(h_k); free(h_lo); free(h_hi); free(h_pa);
            cuda_free_context_flat_pwa_gen_s0e4(c);
            return NULL;
        }
        int nseg = 0;
        for (int k = 0; k < n_wave; k++) {
            const int lo = gs_lo[k], hi = gs_hi[k];
            if (hi - lo > c->split_nnz) {
                const int mid = lo + (hi - lo) / 2;
                h_k[nseg] = k; h_lo[nseg] = lo; h_hi[nseg] = mid; h_pa[nseg] = 0;
                nseg++;
                h_k[nseg] = k; h_lo[nseg] = mid; h_hi[nseg] = hi; h_pa[nseg] = 1;
                nseg++;
            } else {
                h_k[nseg] = k; h_lo[nseg] = lo; h_hi[nseg] = hi; h_pa[nseg] = 0;
                nseg++;
            }
        }
        c->n_seg = nseg;
        c->seg_k = _gps04_up_i(h_k, nseg);
        c->seg_lo = _gps04_up_i(h_lo, nseg);
        c->seg_hi = _gps04_up_i(h_hi, nseg);
        c->seg_part = _gps04_up_i(h_pa, nseg);
        /* the host mirrors, so a caller can read the segmentation back
           (`cuda_segments_*`) instead of re-deriving it in Python */
        c->h_seg_k = (int*)malloc((size_t)nseg * sizeof(int));
        c->h_seg_lo = (int*)malloc((size_t)nseg * sizeof(int));
        c->h_seg_hi = (int*)malloc((size_t)nseg * sizeof(int));
        c->h_seg_part = (int*)malloc((size_t)nseg * sizeof(int));
        if (!c->seg_k || !c->seg_lo || !c->seg_hi || !c->seg_part
                || !c->h_seg_k || !c->h_seg_lo || !c->h_seg_hi
                || !c->h_seg_part) {
            free(h_k); free(h_lo); free(h_hi); free(h_pa);
            cuda_free_context_flat_pwa_gen_s0e4(c);
            return NULL;
        }
        memcpy(c->h_seg_k, h_k, (size_t)nseg * sizeof(int));
        memcpy(c->h_seg_lo, h_lo, (size_t)nseg * sizeof(int));
        memcpy(c->h_seg_hi, h_hi, (size_t)nseg * sizeof(int));
        memcpy(c->h_seg_part, h_pa, (size_t)nseg * sizeof(int));
        free(h_k); free(h_lo); free(h_hi); free(h_pa);
    }
    /* the CSR order of the coefficient, kept on the host: `compute` gathers
       `cw[i] = coeff[h_nz_m[i]]` into it and uploads the result once */
    c->h_nz_m = (int*)malloc((size_t)n_nk * sizeof(int));
    c->h_cw = (double*)malloc((size_t)n_nk * sizeof(double));
    if (!c->h_nz_m || !c->h_cw) {
        cuda_free_context_flat_pwa_gen_s0e4(c);
        return NULL;
    }
    memcpy(c->h_nz_m, nz_m, (size_t)n_nk * sizeof(int));

    /* ── the chunk rule (see `gps04_chunk`) ─────────────────────────────── */
    c->cap = gps04_chunk(batch_size, c->rsize);
    c->s_ngroups = (c->cap + c->rsize - 1) / c->rsize;

    /* ── the tile size ────────────────────────────────────────────────── */
    const int W = c->n_proj * n_wave, P = c->n_proj, N = n_wave;
    c->te = (te_hint > 0) ? te_hint
                          : gps04_pick_te(c->rsize, W, P, N, n_basis);
    /* a hint must still hold whole groups (never straddle a group boundary),
       and it must fit the device: the hint is a test/benchmark hook, and a
       silently wrong tile is exactly the failure this guards against */
    if (c->rsize > 1 && c->te >= c->rsize && (c->te % c->rsize) != 0) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0e4: te %d is not a multiple of rsize %d — "
                "a tile would split a log-sum group\n", c->te, c->rsize);
        free(c);
        return NULL;
    }
    if (c->te < 1) c->te = 1;
    {
        GPS04Layout lay = gps04_layout(c->te, W, P, N, n_basis);
        c->smem_bytes = (int)(lay.total * (long long)sizeof(double));
        int optin = 0;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               0);
        if (c->smem_bytes > optin) {
            fprintf(stderr,
                    "cuda_flat_pwa_gen_s0e4: shared tile %d B exceeds the "
                    "opt-in limit %d B (n_wave %d, n_proj %d, n_basis %d, "
                    "te %d)\n",
                    c->smem_bytes, optin, n_wave, c->n_proj, n_basis, c->te);
            free(c);
            return NULL;
        }
        /* Past the 48 KiB DEFAULT the launch needs the opt-in size on the
           FUNCTION, not merely a size check against the device: without it
           the launch fails with "invalid argument" and — because CUDA_CHECK
           only PRINTS — compute() would silently return Q = 0.

           The attribute is per FUNCTION, so two live contexts of different
           tile sizes would fight; a smaller one created second would LOWER it
           and break the larger one's launch.  Raise-only monotonic makes it
           the process-wide maximum, which every context is <= by
           construction, so any create order works. */
        static size_t opt_in_max = 0;
        if ((size_t)c->smem_bytes > opt_in_max) {
            cudaError_t e = cudaFuncSetAttribute(
                (const void*)gps04_tile_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, c->smem_bytes);
            if (e != cudaSuccess) {
                fprintf(stderr,
                        "cuda_flat_pwa_gen_s0e4: cannot opt in to %d B shared "
                        "memory (%s); n_wave %d, te %d\n",
                        c->smem_bytes, cudaGetErrorString(e), n_wave, c->te);
                free(c);
                return NULL;
            }
            opt_in_max = (size_t)c->smem_bytes;
        }
    }

    c->fl_table = _gps04_up_d(fl_table, n_fl);
    c->fl_q_index = _gps04_up_i(fl_q_index, n_fq);
    c->num_r = _gps04_up_d(num_r, n_num);
    c->num_i = _gps04_up_d(num_i, n_num);
    c->num_m_index = _gps04_up_i(num_m_index, n_nm);
    c->bas_r = _gps04_up_d(bas_r, n_bas);
    c->bas_i = _gps04_up_d(bas_i, n_bas);
    c->bas_m_index = _gps04_up_i(bas_m_index, n_bm);
    c->uslot_row = _gps04_up_i(uslot_row, n_ur);
    c->uslot_col = _gps04_up_i(uslot_col, n_uc);
    c->slot_of_wave = _gps04_up_i(slot_of_wave, n_sw);
    c->rep_of_slot = _gps04_up_i(rep_of_slot, n_rp);
    c->nz_k = _gps04_up_i(nz_k, n_nk);
    c->nz_m = _gps04_up_i(nz_m, n_nzm);
    c->nz_u = _gps04_up_i(nz_u, n_nzu);
    c->gs_lo = _gps04_up_i(gs_lo, n_lo);
    c->gs_hi = _gps04_up_i(gs_hi, n_hi);
    c->gamma = _gps04_up_d(gamma, n_g);
    c->angle_index = _gps04_up_i(angle_index, n_ai);
    c->angle_k = _gps04_up_d(angle_k, n_ak);
    c->angle_b = _gps04_up_d(angle_b, n_ab_);
    c->am_r = _gps04_up_d(am_r, n_am);
    c->am_i = _gps04_up_d(am_i, n_am);
    c->m_min = m_min; c->m_delta = m_delta; c->m_bins = m_bins;
    c->q_min = q_min; c->q_delta = q_delta; c->q_bins = q_bins;

    CUDA_CHECK(cudaStreamCreate(&c->stream));
    CUDA_CHECK(cudaMalloc(&c->p_ck_r, (size_t)n_wave * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->p_ck_i, (size_t)n_wave * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->p_coeff, (size_t)n_basis * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->p_cw, (size_t)n_nk * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->g_ck_r, (size_t)n_wave * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->g_ck_i, (size_t)n_wave * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->g_coeff, (size_t)n_basis * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&c->h_ck_r, (size_t)n_wave * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&c->h_ck_i, (size_t)n_wave * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&c->h_coeff, (size_t)n_basis * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->acc, 2 * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&c->h_acc, 2 * sizeof(double)));
    return c;
}

void cuda_free_context_flat_pwa_gen_s0e4(void* vctx) {
    GPS04Ctx* c = (GPS04Ctx*)vctx;
    if (!c) return;
    #define F(p) if (c->p) cudaFree((void*)c->p)
    #define FH(p) if (c->p) cudaFreeHost(c->p)
    F(fl_table); F(fl_q_index); F(num_r); F(num_i); F(num_m_index);
    F(bas_r); F(bas_i); F(bas_m_index);
    F(uslot_row); F(uslot_col); F(slot_of_wave); F(rep_of_slot);
    F(nz_k); F(nz_m); F(nz_u); F(gs_lo); F(gs_hi); F(gamma);
    F(seg_k); F(seg_lo); F(seg_hi); F(seg_part);
    F(angle_index); F(angle_k); F(angle_b); F(am_r); F(am_i);
    F(p_ck_r); F(p_ck_i); F(p_coeff); F(p_cw);
    F(g_ck_r); F(g_ck_i); F(g_coeff);
    F(acc);
    FH(h_ck_r); FH(h_ck_i); FH(h_coeff); FH(h_acc);
    #undef F
    #undef FH
    if (c->h_nz_m) free(c->h_nz_m);
    if (c->h_cw) free(c->h_cw);
    if (c->h_seg_k) free(c->h_seg_k);
    if (c->h_seg_lo) free(c->h_seg_lo);
    if (c->h_seg_hi) free(c->h_seg_hi);
    if (c->h_seg_part) free(c->h_seg_part);
    if (c->stream) cudaStreamDestroy(c->stream);
    free(c);
}

void* cuda_load_data_flat_pwa_gen_s0e4(
    void* vctx, const double* mass, const double* mom, const double* ang,
    const double* weight, const double* bkg, int ne)
{
    GPS04Ctx* c = (GPS04Ctx*)vctx;
    if (!c) return NULL;
    GPS04Handle* h = (GPS04Handle*)calloc(1, sizeof(GPS04Handle));
    if (!h) return NULL;
    h->ne = ne;
    /* An EMPTY sample is a legal load (a fitter that has not been given data
       yet, or a zero-row selection): every buffer is simply absent, and
       `compute` must then return the identity (Q = 0, zero gradients) rather
       than failing.  A zero-byte cudaMalloc is an error, so the whole body is
       guarded rather than leaving a NULL-pointer trap for `compute`. */
    if (ne <= 0) return h;
    h->weight = _gps04_up_d(weight, ne);
    h->bkg = _gps04_up_d(bkg, ne);
    if (cudaMalloc(&h->us, (size_t)ne * c->n_uslot * sizeof(float2))
            != cudaSuccess) goto fail;
    if (cudaMalloc(&h->amp, (size_t)ne * c->n_uniq * c->n_proj
                           * sizeof(float2)) != cudaSuccess) goto fail;
    if (cudaMalloc(&h->P_dev, (size_t)ne * sizeof(double)) != cudaSuccess)
        goto fail;
    CUDA_CHECK(cudaMallocHost(&h->h_P, (size_t)ne * sizeof(double)));

    /* ── the two load-time caches ─────────────────────────────────────────
     *
     * The kinematics are consumed ONLY here, so they are uploaded as transient
     * device copies and released immediately afterwards — exactly what
     * gen2/1bw do.  They are NOT read through the host pointers: a kernel
     * dereferencing a pageable host pointer happens to work on a device with
     * host page-table access (verified on this Blackwell part) but is not
     * portable, and on a device without it the fill would silently read
     * garbage.  Every other flat kernel copies first; so does this one. */
    {
        double* t_mass = _gps04_up_d(mass, ne * c->n_mass);
        double* t_mom = _gps04_up_d(mom, ne * c->n_momentum);
        double* t_ang = _gps04_up_d(ang, ne * c->n_angle_total * c->n_comp);
        if (!t_mass || !t_mom || !t_ang) {
            if (t_mass) cudaFree(t_mass);
            if (t_mom) cudaFree(t_mom);
            if (t_ang) cudaFree(t_ang);
            goto fail;
        }
        {
            const long long total = (long long)ne * c->n_uslot;
            const int threads = GPS04_BLOCK;
            long long blocks = (total + threads - 1) / threads;
            if (blocks > 65535) blocks = 65535;
            if (blocks < 1) blocks = 1;
            gps04_fill_us<<<(unsigned)blocks, threads, 0, c->stream>>>(
                c->bas_r, c->bas_i, c->uslot_row, c->uslot_col,
                c->m_min, c->m_delta, c->m_bins,
                t_mass, c->n_mass, ne, c->n_uslot, h->us);
            CUDA_CHECK(cudaGetLastError());
        }
        {
            int fb = c->n_uniq < 32 ? 32 : (c->n_uniq < GPS04_BLOCK
                                            ? GPS04_BLOCK : c->n_uniq);
            if (fb > 1024) fb = 1024;
            const size_t shm = (size_t)c->n_ab * sizeof(double);
            gps04_fill_amp<<<ne, fb, shm, c->stream>>>(
                c->fl_table, c->fl_q_index, c->num_r, c->num_i,
                c->num_m_index,
                c->m_min, c->m_delta, c->m_bins,
                c->q_min, c->q_delta, c->q_bins,
                c->angle_index, c->angle_k, c->angle_b, c->am_r, c->am_i,
                t_mass, t_mom, t_ang, c->rep_of_slot,
                ne, c->n_uniq, c->n_wave, c->n_proj, c->n_res, c->n_decay,
                c->n_ab, c->n_comp, c->n_angle_total, c->n_mass,
                c->n_momentum, h->amp);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamSynchronize(c->stream));
        cudaFree(t_mass); cudaFree(t_mom); cudaFree(t_ang);
    }
    return h;

fail:
    if (h->weight) cudaFree(h->weight);
    if (h->bkg) cudaFree(h->bkg);
    if (h->us) cudaFree(h->us);
    if (h->amp) cudaFree(h->amp);
    if (h->P_dev) cudaFree(h->P_dev);
    if (h->h_P) cudaFreeHost(h->h_P);
    free(h);
    return NULL;
}

void cuda_free_data_flat_pwa_gen_s0e4(void* vh) {
    GPS04Handle* h = (GPS04Handle*)vh;
    if (!h) return;
    if (h->weight) { cudaFree(h->weight); h->weight = NULL; }
    if (h->bkg) { cudaFree(h->bkg); h->bkg = NULL; }
    if (h->us) { cudaFree(h->us); h->us = NULL; }
    if (h->amp) { cudaFree(h->amp); h->amp = NULL; }
    if (h->P_dev) { cudaFree(h->P_dev); h->P_dev = NULL; }
    if (h->h_P) { cudaFreeHost(h->h_P); h->h_P = NULL; }
    free(h);
}

/* One launch per chunk.  Batching needs no kernel logic: the chunk is
   expressed as POINTER OFFSETS into the per-event data plus the chunk length
   as the kernel's `ne`, so `ce0`/`ge` are chunk-local by construction. */
static void gps04_launch(GPS04Ctx* c, GPS04Handle* h, int st, int nb,
                        double norm, int use_norm)
{
    const int N = c->n_wave;
    const int grouped = (use_norm && c->rsize > 1);
    int per_blk;
    if (!grouped) {
        per_blk = c->te;
    } else if (c->rsize <= c->te) {
        per_blk = c->rsize * (c->te / c->rsize);
    } else {
        per_blk = c->rsize;
    }
    if (per_blk < 1) per_blk = 1;
    const int tiles = (nb + per_blk - 1) / per_blk;
    gps04_tile_kernel<<<tiles, GPS04_BLOCK, (size_t)c->smem_bytes, c->stream>>>(
        h->us + (size_t)st * c->n_uslot,
        h->amp + (size_t)st * c->n_uniq * c->n_proj,
        c->slot_of_wave, c->nz_m, c->nz_u, c->nz_k, c->gs_lo, c->gs_hi,
        c->gamma, c->n_nz,
        c->seg_k, c->seg_lo, c->seg_hi, c->seg_part, c->n_seg,
        c->p_cw, c->p_ck_r, c->p_ck_i,
        h->weight + st, h->bkg + st,
        norm, use_norm, c->rsize, per_blk, c->te,
        N, c->n_proj, c->n_basis, c->n_uslot, c->n_uniq, nb,
        h->P_dev + st,
        c->g_ck_r, c->g_ck_i, c->g_coeff, c->acc, c->acc + 1);
}

void cuda_compute_flat_pwa_gen_s0e4(
    void* vctx, void* vh, const double* ck_r, const double* ck_i,
    const double* coeff, double norm, int use_norm,
    double* oQ, double* oDn, double* oP,
    double* og_ck_r, double* og_ck_i, double* og_coeff)
{
    GPS04Ctx* c = (GPS04Ctx*)vctx;
    GPS04Handle* h = (GPS04Handle*)vh;
    if (!c || !h) return;
    const int N = c->n_wave, NB = c->n_basis;

    memcpy(c->h_ck_r, ck_r, (size_t)N * sizeof(double));
    memcpy(c->h_ck_i, ck_i, (size_t)N * sizeof(double));
    memcpy(c->h_coeff, coeff, (size_t)NB * sizeof(double));
    cudaMemcpyAsync(c->p_ck_r, c->h_ck_r, (size_t)N * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    cudaMemcpyAsync(c->p_ck_i, c->h_ck_i, (size_t)N * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    cudaMemcpyAsync(c->p_coeff, c->h_coeff, (size_t)NB * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    /* the fitted coefficient in CSR order: `cw[i] = coeff[nz_m[i]]`.  Same
       `double` values, laid out the way phase A walks them, so the inner loop
       has no coefficient indirection.  ONE upload of n_nz doubles per call. */
    for (int i = 0; i < c->n_nz; i++) c->h_cw[i] = c->h_coeff[c->h_nz_m[i]];
    cudaMemcpyAsync(c->p_cw, c->h_cw, (size_t)c->n_nz * sizeof(double),
                    cudaMemcpyHostToDevice, c->stream);
    /* the accumulators span the WHOLE sample: zero once, read once */
    cudaMemsetAsync(c->acc, 0, 2 * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_ck_r, 0, (size_t)N * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_ck_i, 0, (size_t)N * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_coeff, 0, (size_t)NB * sizeof(double), c->stream);

    const int ne_total = h->ne;
    const int bs = (c->cap > 0) ? c->cap : 1;
    for (int st = 0; st < ne_total; st += bs) {
        const int nb = (ne_total - st > bs) ? bs : (ne_total - st);
        gps04_launch(c, h, st, nb, norm, use_norm);
        CUDA_CHECK(cudaGetLastError());
    }
    /* the caller-visible P: ONE D2H for the whole sample (the buffer is the
       handle's per-event OUTPUT, allocated once per load, never per call) */
    if (oP != NULL && ne_total > 0) {
        cudaMemcpyAsync(h->h_P, h->P_dev, (size_t)ne_total * sizeof(double),
                        cudaMemcpyDeviceToHost, c->stream);
    }
    cudaMemcpyAsync(c->h_acc, c->acc, 2 * sizeof(double),
                    cudaMemcpyDeviceToHost, c->stream);
    cudaMemcpyAsync(og_ck_r, c->g_ck_r, (size_t)N * sizeof(double),
                    cudaMemcpyDeviceToHost, c->stream);
    cudaMemcpyAsync(og_ck_i, c->g_ck_i, (size_t)N * sizeof(double),
                    cudaMemcpyDeviceToHost, c->stream);
    cudaMemcpyAsync(og_coeff, c->g_coeff, (size_t)NB * sizeof(double),
                    cudaMemcpyDeviceToHost, c->stream);
    CUDA_CHECK(cudaStreamSynchronize(c->stream));
    *oQ = c->h_acc[0];
    *oDn = c->h_acc[1];
    if (oP != NULL && ne_total > 0)
        memcpy(oP, h->h_P, (size_t)ne_total * sizeof(double));
}

/* ── reports: every one of these has exactly ONE definition, in C ─────── */

/* The tile size `gps04_layout` gives for an arbitrary shape — the same
 * function the kernel and `create` use, so a test can assert the buffer
 * arithmetic without a second, drifting copy in Python. */
long long cuda_tile_bytes_flat_pwa_gen_s0e4(int te, int n_wave, int n_proj,
                                           int n_basis)
{
    GPS04Layout L = gps04_layout(te, n_proj * n_wave, n_proj, n_wave, n_basis);
    return (long long)L.total * (long long)sizeof(double);
}

int cuda_smem_bytes_flat_pwa_gen_s0e4(void* vctx) {
    GPS04Ctx* c = (GPS04Ctx*)vctx;
    return c ? c->smem_bytes : 0;
}

int cuda_cap_flat_pwa_gen_s0e4(void* vctx) {
    GPS04Ctx* c = (GPS04Ctx*)vctx;
    return c ? c->cap : 0;
}

int cuda_ngroups_flat_pwa_gen_s0e4(void* vctx) {
    GPS04Ctx* c = (GPS04Ctx*)vctx;
    return c ? c->s_ngroups : 0;
}

int cuda_te_flat_pwa_gen_s0e4(void* vctx) {
    GPS04Ctx* c = (GPS04Ctx*)vctx;
    return c ? c->te : 0;
}

/* The EFFECTIVE segmentation threshold (0 at create means the compiled
 * default, so this is how a caller learns what was actually used). */
int cuda_split_nnz_flat_pwa_gen_s0e4(void* vctx) {
    GPS04Ctx* c = (GPS04Ctx*)vctx;
    return c ? c->split_nnz : 0;
}

/* Number of phase-A segments (`Σ_k (nnz(k) > T ? 2 : 1)`). */
int cuda_nseg_flat_pwa_gen_s0e4(void* vctx) {
    GPS04Ctx* c = (GPS04Ctx*)vctx;
    return c ? c->n_seg : 0;
}

/* Copy the segment list into `out` (4 blocks of n_seg ints: k | lo | hi |
 * part).  The segmentation has ONE definition — in `create` — so a test asks
 * for it rather than re-deriving it in Python. */
void cuda_segments_flat_pwa_gen_s0e4(void* vctx, int* out) {
    GPS04Ctx* c = (GPS04Ctx*)vctx;
    if (!c || !out) return;
    const size_t n = (size_t)c->n_seg;
    for (size_t i = 0; i < n; i++) {
        out[4 * i + 0] = c->h_seg_k[i];
        out[4 * i + 1] = c->h_seg_lo[i];
        out[4 * i + 2] = c->h_seg_hi[i];
        out[4 * i + 3] = c->h_seg_part[i];
    }
}

/* Runtime resource report (host side, no kernel launch).
 *
 * CUDA exposes the compiled resource usage per FUNCTION:
 *   out[0] registers/thread          (cudaFuncGetAttributes.numRegs)
 *   out[1] local bytes > 0  => the kernel SPILLS
 *   out[2] blocks/SM at the REAL block size and dynamic smem
 *   out[3] warps/SM  (= blocks x GPS04_BLOCK/32)
 *   out[4] maxThreadsPerBlock
 *   out[5] static shared bytes
 * `smem` is the tile's dynamic shared memory for the CURRENT shape, so the
 * occupancy figure is the one the fit actually gets.
 */
void cuda_stats_flat_pwa_gen_s0e4(int smem, int* out)
{
    if (!out) return;
    cudaFuncAttributes a;
    cudaFuncGetAttributes(&a, (const void*)gps04_tile_kernel);
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, (const void*)gps04_tile_kernel, GPS04_BLOCK,
        (size_t)(smem > 0 ? smem : 0));
    out[0] = a.numRegs;
    out[1] = (int)a.localSizeBytes;
    out[2] = blocks;
    out[3] = blocks * (GPS04_BLOCK / 32);
    out[4] = a.maxThreadsPerBlock;
    out[5] = (int)a.sharedSizeBytes;
}

int cuda_get_device_count_flat_pwa_gen_s0e4(void) {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
}

void cuda_get_device_name_flat_pwa_gen_s0e4(char* buf, int len) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    snprintf(buf, len, "%s", p.name);
}

}  /* extern "C" */
