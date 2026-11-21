/**
 * cuda_flat_pwa_gen_s0e5 — the s0e FUSED TILE on the CURRENT flat `pwa_gen`
 * config, with the grouped log-sum objective and batch_size chunking.
 *
 * Case-1 copy (kernel policy §1): own `.cu`, own `.so`, own wrapper, own
 * symbols.  `kernels_flat_pwa_gen2.cu`, `kernels_flat_pwa_gen_1bw.cu`,
 * every `kernels_*_s0*.cu` (including `kernels_flat_pwa_gen_s0e.cu`) and
 * their wrappers stay byte-identical.
 *
 * ── what s0e2 changes against s0e (algebra identical) ───────────────────
 *
 * Two instruction-level changes in the tile kernel, nothing else:
 *
 * 1. **the CSR step is ONE 64-bit load.**  The host packs the per-step pair
 *    `(coeff index, uslot)` into a single `long long`
 *    (`m | (u << 32)`), so phase A issues one 8-byte load per non-zero
 *    instead of two 4-byte loads from two arrays.
 * 2. **the sparse dot runs TWO independent fp64 accumulation chains**, so a
 *    wave with `nnz(k) > 1` does not serialize on one dependency chain
 *    (the FMA latency is hidden by a second chain).  The chains are added at
 *    the end — same fp64 operations, only the summation ORDER differs
 *    (~1e-16 relative).
 *
 * 3. **`__launch_bounds__(GPS05_BLOCK, GPS05_MIN_BLOCKS)`** (= 256, 5) on the tile kernel, with the
 *    block count a `-D`/`#if __CUDA_ARCH__` macro (see below) because the
 *    right register budget is per arch.  Without it the two
 *    chains hold more live values: measured 48 -> 56 registers, occupancy
 *    80 -> 63 %, duration +2-4 %.  The bound restores 48 registers and
 *    parity.  (Measured on the km fixture, n_wave = 2; the production shape
 *    is n_wave = 26 — re-run `tests/benchmark_s0e_variants.py` on an IDLE
 *    GPU before reading anything into these numbers.)
 *
 * VERDICT SO FAR: neutral at best on the fixture measured — the tile is
 * fp64-throughput bound at ~80 % occupancy, not CSR-chain bound, so the
 * extra ILP has nothing to hide and only costs registers.  Kept as a case-1
 * alternative because the fixture's n_wave = 2 under-weights exactly the
 * per-(e,k) work that dominates at production n_wave = 26.
 *
 * Precision policy is unchanged: the stored caches (`us`, `amp`) stay
 * `float2`, every real arithmetic step stays fp64 (gradients included).
 * The `nz_m`/`nz_u` arrays stay in the signature because the FILL kernels
 * still read them (`nz_m`/`nz_u` are also used by phase D).
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
 * (`gps05_layout` is the ONE definition; the host asks the context rather than
 * mirroring the formula).  On the real shape at `te = 8` that is 17 640 B —
 * under the 48 KiB DEFAULT, so the launch needs no opt-in there.  The opt-in
 * is still applied (raise-only monotonic, per FUNCTION not per context, so two
 * live contexts of different tile sizes cannot fight) for shapes that cross
 * it, and a tile that exceeds the device's opt-in limit is REFUSED by `create`
 * rather than launched into a silent `Q = 0`.
 *
 * `te` is chosen in `gps05_pick_te` from the shape, and the rule is measured
 * rather than assumed.  On the real 8.75 M-event `jpsi_eta_pi_pi` sample
 * (rsize 17, `Fitter.get_nll`, min of 4, one process per setting):
 *
 *     te      1      2      4      8     17      (all beyond rsize -> path B)
 *     ms    476    385    344    325    333
 *
 * A small tile with ONE BLOCK PER GROUP beats a 17-event tile that holds
 * whole groups: better occupancy from the smaller shared tile outweighs the
 * group-walk bookkeeping.  So `te` stays near `GPS05_DEF_TE`, rounded UP to a
 * whole number of groups only when `rsize <= GPS05_DEF_TE` (where path A costs
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

#define GPS05_BLOCK       256      /* threads per block (multiple of 32) */
/* Launch bounds: the SECOND parameter is a REGISTER BUDGET choice, not a
 * correctness one.  Measured on sm_120: 5 blocks/SM => <=51 registers, which
 * is what the 2-chain dot needs (without it the compiler takes 56 and loses
 * 17 occupancy points).  The right budget is per ARCH, so it is a macro:
 *
 *   per build   TABPWA_NVCC_FLAGS_SM70="-DGPS05_MIN_BLOCKS=6"
 *               (the arch-keyed extra-flags hook; -D applies to every pass of
 *               that one nvcc invocation)
 *   per pass    an `#if __CUDA_ARCH__` block here, if ONE fat binary must
 *               differ between its cubins (__CUDA_ARCH__ is set per device
 *               pass; the default below leaves the bound arch-uniform)
 *
 * Volta has 64 warps/SM (8 x 256-thread blocks) but only 64K registers, so 6
 * blocks there wants <=42 registers.
 *
 * MEASURED on sm_120 (0 spills at every N): N=4 -> 64 registers -> 32 warps;
 * N=5 -> 48 -> 40 warps (the default, matching s0e's occupancy); N=6 -> 40 ->
 * 48 warps (100 %).  MEASURED on sm_70: N=4 is clean, N=5 spills 56 B/192 B,
 * N=6 spills 132 B/444 B — hence the arch-aware default above and the
 * per-arch override, e.g. `TABPWA_NVCC_FLAGS_SM70=-DGPS05_MIN_BLOCKS=3`.
 * Pick from a measurement, never from taste. */
#ifndef GPS05_MIN_BLOCKS
#  if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
/* MEASURED with -Xptxas -v, tile kernel, N = min blocks per SM:
 *   arch   N=4           N=5
 *   70     clean         56 B stores / 192 B loads
 *   75     clean         clean
 *   80     clean         28 / 36
 *   86     clean         28 / 36
 *   89     clean         28 / 36
 *   90     clean         8 / 8
 *   120    clean         clean
 * N=4 is clean EVERYWHERE, so it is the default; N=5 (48 registers, 40 warps
 * instead of 64 registers, 32 warps) is the better point where it does not
 * spill, and 6 (40 registers, 48 warps = 100 %) is the A/B candidate here.
 * __CUDA_ARCH__ is defined per DEVICE pass, which is the granularity a fat
 * binary needs; the env hook overrides the default per build. */
#    define GPS05_MIN_BLOCKS 5      /* Blackwell consumer: clean, matches s0e */
#  else
#    define GPS05_MIN_BLOCKS 4      /* every other arch measured: 5 spills */
#  endif
#endif
#define GPS05_TILE_BOUNDS __launch_bounds__(GPS05_BLOCK, GPS05_MIN_BLOCKS)
#define GPS05_DEF_TE      8        /* default events per tile */
#define GPS05_MAX_CAP     (1 << 20)  /* ceiling on the chunk length  */
/*: soft shared-memory budget for choosing `te`.  Below this the launch needs
 *  no opt-in (so no `cudaFuncSetAttribute` and the best occupancy); above it
 *  `create` shrinks `te` first and only then crosses the opt-in. */
#define GPS05_SMEM_BUDGET (48 * 1024)

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = call;                                             \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                               \
        }                                                                    \
    } while (0)

/* ── Catmull-Rom (the repo-wide idiom, identical to gen2/1bw) ─────────── */
__device__ double gps05_cr1d(double pm1, double p0, double p1, double p2,
                            double t) {
    return p0 + 0.5 * t * (-pm1 + p1 + t * (2.0 * pm1 - 5.0 * p0 + 4.0 * p1
                                            - p2 + t * (-pm1 + 3.0 * p0
                                                        - 3.0 * p1 + p2)));
}

__device__ double gps05_interp_real(const double* __restrict__ t, int row,
                                   double x, double xmin, double xdelta,
                                   int bins) {
    double diff = (x - xmin) / xdelta;
    int b = max(0, min((int)floor(diff), bins - 2));
    double s = max(0.0, min(diff - b, 1.0));
    int base = row * bins + b;
    int lo = row * bins, hi = (row + 1) * bins - 1;
    int im1 = base > lo ? base - 1 : base;
    int i2 = base + 2 <= hi ? base + 2 : base + 1;
    return gps05_cr1d(t[im1], t[base], t[base + 1], t[i2], s);
}

__device__ void gps05_interp_c(const double* __restrict__ tr,
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
    *orr = gps05_cr1d(tr[im1], tr[base], tr[base + 1], tr[i2], s);
    *oii = gps05_cr1d(ti[im1], ti[base], ti[base + 1], ti[i2], s);
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
__global__ void gps05_fill_us(
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
        gps05_interp_c(bas_r, bas_i, uslot_row[u], mv,
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
__global__ void gps05_fill_amp(
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
            gps05_interp_c(num_r, num_i, k * n_res + r, mv,
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
            fl *= gps05_interp_real(fl_table, k * n_decay + d, qv,
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
} GPS05Layout;

__host__ __device__ __forceinline__ GPS05Layout gps05_layout(
    int te, int W, int P, int N, int NB)
{
    GPS05Layout L;
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
__global__ void GPS05_TILE_BOUNDS gps05_tile_kernel(
    const float2* __restrict__ us,         /* [ne, n_uslot]   Z = i·basis   */
    const float2* __restrict__ amp,        /* [ne, n_uniq, P] fa·fl·num     */
    const int* __restrict__ slot_of_wave,  /* [n_wave] -> amp slot          */
    const int* __restrict__ nz_m,          /* [nnz] basis column            */
    const int* __restrict__ nz_u,          /* [nnz] -> uslot                */
    const long long* __restrict__ nz_pair, /* [nnz] m | (u << 32) packed    */
    const int* __restrict__ nz_k,          /* [nnz] wave                    */
    const int* __restrict__ gs_lo,         /* [n_wave] CSR start into nz_*  */
    const int* __restrict__ gs_hi,         /* [n_wave] CSR end              */
    const int* __restrict__ key_lo,        /* [n_key+1] the key CSR         */
    const int* __restrict__ kw_lo,         /* [n_key+1] -> wave_of          */
    const int* __restrict__ wave_of,       /* [n_wave] waves of a key       */
    const long long* __restrict__ nz_key,  /* [n_nz] the key's pairs        */
    int n_key,
    const double* __restrict__ gamma,      /* [n_wave·n_basis] (n_res == 1) */
    int n_nz,
    const double* __restrict__ coeff,      /* [n_basis]                     */
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
    const GPS05Layout L = gps05_layout(te, W, P, N, NB);

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

        /* ── phase A: D over the DEDUPED keys, 1/|D|², and common_{e,p,k} ──
         *
         * Waves whose gamma row selects the same basis columns (and hence the
         * same mass columns) compute the SAME complex dot, so the host keys
         * them once (`key_of_wave`, production 26 -> 18 keys, 130 -> 74
         * non-zeros per event) and this loop walks (event, key).  Every key's
         * non-zero list is the host's, in the same order, so a key's dot is
         * the same sequence of fp64 operations the per-wave version did — the
         * dedup is numerically EXACT, not an approximation.
         *
         * `1/|D|²` is a REAL reciprocal: D is complex, so `1/D` is
         * `conj(D)/|D|²` and one fp64 divide serves every consumer
         * (fp64 divide is ~1/64 rate and unpipelined on a consumer part).
         * It is computed ONCE per (event, key) and reused by every wave of
         * that key, and later by phase C from `s_rd` — no second divide. */
        for (int idx = tid; idx < nen * n_key; idx += nt) {
            const int e = idx / n_key, kk = idx - e * n_key;
            const float2* uu = us + (size_t)(ce0 + e) * n_uslot;
            const int lo = key_lo[kk], hi = key_lo[kk + 1];
            /* ONE packed 64-bit load per CSR step — (coeff index) |
               (uslot << 32) — and TWO independent fp64 accumulation chains,
               so a key that walks nnz > 1 does not serialize on a single
               dependency. */
            double sr0 = 0.0, si0 = 0.0, sr1 = 0.0, si1 = 0.0;
            int i = lo;
            const int two = lo + ((hi - lo) & ~1);
            for (; i < two; i += 2) {
                const long long p0 = nz_key[i], p1 = nz_key[i + 1];
                const double c0 = coeff[(int)(p0 & 0xFFFFFFFFLL)];
                const double c1 = coeff[(int)(p1 & 0xFFFFFFFFLL)];
                const float2 z0 = uu[(int)(p0 >> 32)];
                const float2 z1 = uu[(int)(p1 >> 32)];
                sr0 = fma(c0, (double)z0.y, sr0);
                si0 = fma(-c0, (double)z0.x, si0);
                sr1 = fma(c1, (double)z1.y, sr1);
                si1 = fma(-c1, (double)z1.x, si1);
            }
            if (i < hi) {
                const long long p = nz_key[i];
                const double c = coeff[(int)(p & 0xFFFFFFFFLL)];
                const float2 z = uu[(int)(p >> 32)];
                sr0 = fma(c, (double)z.y, sr0);
                si0 = fma(-c, (double)z.x, si0);
            }
            const double sr = sr0 + sr1, si = si0 + si1;
            const double rden = 1.0 / fma(sr, sr, si * si);
            /* every wave of this key reads the same D and the same 1/|D|² */
            for (int w = kw_lo[kk]; w < kw_lo[kk + 1]; w++) {
                const int k = wave_of[w];
                const int o2 = e * N + k;
                s_D[o2 * 2 + 0] = sr;
                s_D[o2 * 2 + 1] = si;
                s_rd[o2] = rden;
                const size_t abase = (size_t)(ce0 + e) * n_uniq * P
                                     + (size_t)slot_of_wave[k] * P;
                for (int p = 0; p < P; p++) {
                    const float2 a = amp[abase + p];
                    const int o = (e * P + p) * N + k;
                    s_cr[o] = fma((double)a.x, sr, (double)a.y * si) * rden;
                    s_ci[o] = fma((double)a.y, sr, -(double)a.x * si) * rden;
                }
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
                    ar = fma(ckr, cr[k], ar);
                    ar = fma(-cki, ci[k], ar);
                    ai = fma(ckr, ci[k], ai);
                    ai = fma(cki, cr[k], ai);
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
                    Pe = fma(ar, ar, fma(ai, ai, Pe));
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
                    WP = fma(weight[ce0 + e], s_pe[e], WP);
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
                WP = fma(weight[ce0 + e], s_pe[e], WP);
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
                ur = fma(dr, cr, ur);
                ur = fma(-di, ci, ur);
                ui = fma(dr, ci, ui);
                ui = fma(di, cr, ui);
            }
            atomicAdd(&s_gck[2 * k + 0], ur);
            atomicAdd(&s_gck[2 * k + 1], ui);
            const double dr = s_D[idx * 2 + 0], di = s_D[idx * 2 + 1];
            const double rden = s_rd[idx];
            const double qr = fma(ckr, ur, -cki * ui);   /* ck·u */
            const double qi = fma(ckr, ui, cki * ur);
            s_pv[idx * 2 + 0] = -fma(qr, dr, qi * di) * rden;  /* -ck·u/D */
            s_pv[idx * 2 + 1] = -fma(qi, dr, -qr * di) * rden;
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
            const double g2 = 2.0 * gamma[k * NB + m];   /* exact factor 2 */
            double out = 0.0;
            for (int e = 0; e < nen; e++) {
                const float2 z = us[(size_t)(ce0 + e) * n_uslot + u];
                const size_t o = (size_t)(e * N + k) * 2;
                out = fma(s_pv[o + 0], (double)z.y, out);
                out = fma(s_pv[o + 1], (double)z.x, out);
            }
            if (out != 0.0) atomicAdd(&s_gc[m], out * g2);
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
    long long *nz_pair;               /* [n_nz] m | (u << 32) — packed CSR */
    /* the denominator dedup: `key_of_wave` maps a wave to its key (waves with
       the same gamma pattern and the same mass columns share one dot), the
       key CSR is `key_lo[k]..key_lo[k+1]` into `nz_key`, and
       `kw_lo[k]..kw_lo[k+1]` into `wave_of` lists the waves of key k. */
    int *key_of_wave, *key_lo, *kw_lo, *wave_of;
    long long *nz_key;                /* [n_nz] the key's packed pairs */
    int n_key;
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
} GPS05Ctx;

/* ONE SAMPLE.  Per-EVENT data only: the two caches, weight/bkg, and the
   caller-visible per-event P OUTPUT (not scratch — see the file header). */
typedef struct {
    double *weight, *bkg;
    float2 *us;           /* [ne, n_uslot]        Z = i·basis        */
    float2 *amp;          /* [ne, n_uniq, n_proj] fa·fl·num          */
    double *P_dev;        /* [ne] per-event P (output of the call)   */
    double *h_P;          /* pinned staging for the one D2H          */
    int ne;
} GPS05Handle;

static int* _gps05_up_i(const int* src, int n) {
    if (n <= 0 || src == NULL) return NULL;
    int* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(int),
                          cudaMemcpyHostToDevice));
    return d;
}

static long long* _gps05_up_l(const long long* src, int n) {
    if (n <= 0 || src == NULL) return NULL;
    long long* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(long long)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(long long),
                          cudaMemcpyHostToDevice));
    return d;
}

static double* _gps05_up_d(const double* src, int n) {
    if (n <= 0 || src == NULL) return NULL;
    double* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d, src, (size_t)n * sizeof(double),
                          cudaMemcpyHostToDevice));
    return d;
}

/* The chunk rule.  ONE definition, asked for through `cuda_cap_*`. */
static int gps05_chunk(int batch_size, int rsize)
{
    int cap = (batch_size > 0) ? batch_size : 50000;
    if (cap > GPS05_MAX_CAP) cap = GPS05_MAX_CAP;
    if (rsize > 1) {
        const int a = (cap / rsize) * rsize;
        cap = (a > 0) ? a : rsize;   /* never 0, never unaligned to a group */
    }
    return cap;
}

/* The tile size.  Goals, in order: (1) whole groups per tile when `rsize` is
 * small enough (path A — the group sums then live in the tile); (2) a tile of
 * at least ~GPS05_DEF_TE events so a block has real work; (3) the shared tile
 * inside GPS05_SMEM_BUDGET, which is what actually binds.
 *
 * MEASURED on the real 8.75 M-event `jpsi_eta_pi_pi` sample (n_wave 26,
 * n_proj 2, n_basis 69, rsize 17, `Fitter.get_nll`, min of 4):
 *
 *     te   1     2     4     8     17
 *     ms   476   385   344   325   333
 *
 * i.e. a tile of GPS05_DEF_TE events with path B (one block per group, walked
 * internally in 8-event chunks) beats a 17-event tile that holds whole groups
 * — smaller shared memory means better occupancy, and that outweighs the
 * group-walk bookkeeping.  So `te` is kept near GPS05_DEF_TE and only rounded
 * UP to a whole number of groups when `rsize <= GPS05_DEF_TE` (where path A is
 * free).  On the 100-wave benchmark shape the same rule shrinks te to 4 by the
 * shared-memory budget, which is that shape's measured optimum too (6.04 vs
 * 6.08 ms at te=8, 5.66 ms with rsize 17). */
static int gps05_pick_te(int rsize, int W, int P, int N, int NB)
{
    int te;
    if (rsize > 1 && rsize <= GPS05_DEF_TE) {
        const int k = (GPS05_DEF_TE + rsize - 1) / rsize;   /* ceil */
        te = rsize * (k < 1 ? 1 : k);                     /* whole groups */
    } else {
        te = GPS05_DEF_TE;         /* rsize == 1, or path B (rsize > DEF_TE) */
    }
    /* `gps05_launch` floors a tile to whole groups on its own, so shrinking is
     * always valid: below `rsize` the block simply owns one group (path B). */
    while (te > 1) {
        const size_t sm =
            (size_t)gps05_layout(te, W, P, N, NB).total * sizeof(double);
        if (sm <= (size_t)GPS05_SMEM_BUDGET) break;
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

void cuda_free_context_flat_pwa_gen_s0e5(void* vctx);   /* fwd decl */

void* cuda_create_context_flat_pwa_gen_s0e5(
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
    const long long* nz_pair, int n_np,
    const int* key_of_wave, int n_kow,
    const int* key_lo, int n_kl, const int* kw_lo, int n_kwl,
    const int* wave_of, int n_wo,
    const long long* nz_key, int n_nkey, int n_key,
    const int* gs_lo, int n_lo, const int* gs_hi, int n_hi,
    const double* gamma, int n_g,
    const int* angle_index, int n_ai,
    const double* angle_k, int n_ak, const double* angle_b, int n_ab_,
    const double* am_r, const double* am_i, int n_am,
    double m_min, double m_delta, int m_bins,
    double q_min, double q_delta, int q_bins,
    int n_wave, int n_proj, int n_res, int n_decay, int n_basis,
    int n_ab, int n_comp, int n_mass, int n_momentum, int n_angle_total,
    int batch_size, int rsize, int te_hint)
{
    if (n_res != 1) {
        /* the wrapper rejects this first (with the general backend named);
           the kernel refuses too rather than reading a 3-D gamma as 1 factor */
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0e5: n_res must be 1, got %d — use "
                "cuda_flat_pwa_gen2 for multi-BW chains\n", n_res);
        return NULL;
    }
    if (n_sw != n_wave || n_rp != n_uniq) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0e5: slot_of_wave/rep_of_slot length "
                "mismatch (%d vs n_wave %d, %d vs n_uniq %d)\n",
                n_sw, n_wave, n_rp, n_uniq);
        return NULL;
    }
    if (n_nk != n_nzm || n_nk != n_nzu || n_np != n_nzm
            || n_lo != n_wave || n_hi != n_wave) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0e5: nz_*/gs_* length mismatch "
                "(nz %d/%d/%d/%d, gs %d/%d, n_wave %d)\n",
                n_nk, n_nzm, n_nzu, n_np, n_lo, n_hi, n_wave);
        return NULL;
    }
    if (n_key <= 0 || n_kow != n_wave || n_kl != n_key + 1
            || n_kwl != n_key + 1 || n_wo != n_wave
            || n_nkey <= 0 || n_nkey > n_np) {
        fprintf(stderr,
                "%s: key arrays inconsistent (n_key %d, key_of_wave %d, "
                "key_lo %d, kw_lo %d, wave_of %d, nz_key %d vs nz_pair %d)\n",
                "cuda_flat_pwa_gen_s0e5", n_key, n_kow, n_kl, n_kwl, n_wo,
                n_nkey, n_np);
        return NULL;
    }
    if (n_uniq <= 0 || n_uslot <= 0 || n_wave <= 0 || n_basis <= 0) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0e5: empty shape (n_uniq %d, n_uslot %d, "
                "n_wave %d, n_basis %d)\n", n_uniq, n_uslot, n_wave, n_basis);
        return NULL;
    }
    GPS05Ctx* c = (GPS05Ctx*)calloc(1, sizeof(GPS05Ctx));
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

    /* ── the chunk rule (see `gps05_chunk`) ─────────────────────────────── */
    c->cap = gps05_chunk(batch_size, c->rsize);
    c->s_ngroups = (c->cap + c->rsize - 1) / c->rsize;

    /* ── the tile size ────────────────────────────────────────────────── */
    const int W = c->n_proj * n_wave, P = c->n_proj, N = n_wave;
    c->te = (te_hint > 0) ? te_hint
                          : gps05_pick_te(c->rsize, W, P, N, n_basis);
    /* a hint must still hold whole groups (never straddle a group boundary),
       and it must fit the device: the hint is a test/benchmark hook, and a
       silently wrong tile is exactly the failure this guards against */
    if (c->rsize > 1 && c->te >= c->rsize && (c->te % c->rsize) != 0) {
        fprintf(stderr,
                "cuda_flat_pwa_gen_s0e5: te %d is not a multiple of rsize %d — "
                "a tile would split a log-sum group\n", c->te, c->rsize);
        free(c);
        return NULL;
    }
    if (c->te < 1) c->te = 1;
    {
        GPS05Layout lay = gps05_layout(c->te, W, P, N, n_basis);
        c->smem_bytes = (int)(lay.total * (long long)sizeof(double));
        int optin = 0;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               0);
        if (c->smem_bytes > optin) {
            fprintf(stderr,
                    "cuda_flat_pwa_gen_s0e5: shared tile %d B exceeds the "
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
                (const void*)gps05_tile_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, c->smem_bytes);
            if (e != cudaSuccess) {
                fprintf(stderr,
                        "cuda_flat_pwa_gen_s0e5: cannot opt in to %d B shared "
                        "memory (%s); n_wave %d, te %d\n",
                        c->smem_bytes, cudaGetErrorString(e), n_wave, c->te);
                free(c);
                return NULL;
            }
            opt_in_max = (size_t)c->smem_bytes;
        }
    }

    c->fl_table = _gps05_up_d(fl_table, n_fl);
    c->fl_q_index = _gps05_up_i(fl_q_index, n_fq);
    c->num_r = _gps05_up_d(num_r, n_num);
    c->num_i = _gps05_up_d(num_i, n_num);
    c->num_m_index = _gps05_up_i(num_m_index, n_nm);
    c->bas_r = _gps05_up_d(bas_r, n_bas);
    c->bas_i = _gps05_up_d(bas_i, n_bas);
    c->bas_m_index = _gps05_up_i(bas_m_index, n_bm);
    c->uslot_row = _gps05_up_i(uslot_row, n_ur);
    c->uslot_col = _gps05_up_i(uslot_col, n_uc);
    c->slot_of_wave = _gps05_up_i(slot_of_wave, n_sw);
    c->rep_of_slot = _gps05_up_i(rep_of_slot, n_rp);
    c->nz_k = _gps05_up_i(nz_k, n_nk);
    c->nz_m = _gps05_up_i(nz_m, n_nzm);
    c->nz_u = _gps05_up_i(nz_u, n_nzu);
    c->nz_pair = _gps05_up_l(nz_pair, n_np);
    c->key_of_wave = _gps05_up_i(key_of_wave, n_kow);
    c->key_lo = _gps05_up_i(key_lo, n_kl);
    c->kw_lo = _gps05_up_i(kw_lo, n_kwl);
    c->wave_of = _gps05_up_i(wave_of, n_wo);
    c->nz_key = _gps05_up_l(nz_key, n_nkey);
    c->n_key = n_key;
    c->gs_lo = _gps05_up_i(gs_lo, n_lo);
    c->gs_hi = _gps05_up_i(gs_hi, n_hi);
    c->gamma = _gps05_up_d(gamma, n_g);
    c->angle_index = _gps05_up_i(angle_index, n_ai);
    c->angle_k = _gps05_up_d(angle_k, n_ak);
    c->angle_b = _gps05_up_d(angle_b, n_ab_);
    c->am_r = _gps05_up_d(am_r, n_am);
    c->am_i = _gps05_up_d(am_i, n_am);
    c->m_min = m_min; c->m_delta = m_delta; c->m_bins = m_bins;
    c->q_min = q_min; c->q_delta = q_delta; c->q_bins = q_bins;

    CUDA_CHECK(cudaStreamCreate(&c->stream));
    CUDA_CHECK(cudaMalloc(&c->p_ck_r, (size_t)n_wave * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->p_ck_i, (size_t)n_wave * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&c->p_coeff, (size_t)n_basis * sizeof(double)));
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

void cuda_free_context_flat_pwa_gen_s0e5(void* vctx) {
    GPS05Ctx* c = (GPS05Ctx*)vctx;
    if (!c) return;
    #define F(p) if (c->p) cudaFree((void*)c->p)
    #define FH(p) if (c->p) cudaFreeHost(c->p)
    F(fl_table); F(fl_q_index); F(num_r); F(num_i); F(num_m_index);
    F(bas_r); F(bas_i); F(bas_m_index);
    F(uslot_row); F(uslot_col); F(slot_of_wave); F(rep_of_slot);
    F(nz_k); F(nz_m); F(nz_u); F(nz_pair); F(gs_lo); F(gs_hi); F(gamma);
    F(key_of_wave); F(key_lo); F(kw_lo); F(wave_of); F(nz_key);
    F(angle_index); F(angle_k); F(angle_b); F(am_r); F(am_i);
    F(p_ck_r); F(p_ck_i); F(p_coeff);
    F(g_ck_r); F(g_ck_i); F(g_coeff);
    F(acc);
    FH(h_ck_r); FH(h_ck_i); FH(h_coeff); FH(h_acc);
    #undef F
    #undef FH
    if (c->stream) cudaStreamDestroy(c->stream);
    free(c);
}

void* cuda_load_data_flat_pwa_gen_s0e5(
    void* vctx, const double* mass, const double* mom, const double* ang,
    const double* weight, const double* bkg, int ne)
{
    GPS05Ctx* c = (GPS05Ctx*)vctx;
    if (!c) return NULL;
    GPS05Handle* h = (GPS05Handle*)calloc(1, sizeof(GPS05Handle));
    if (!h) return NULL;
    h->ne = ne;
    /* An EMPTY sample is a legal load (a fitter that has not been given data
       yet, or a zero-row selection): every buffer is simply absent, and
       `compute` must then return the identity (Q = 0, zero gradients) rather
       than failing.  A zero-byte cudaMalloc is an error, so the whole body is
       guarded rather than leaving a NULL-pointer trap for `compute`. */
    if (ne <= 0) return h;
    h->weight = _gps05_up_d(weight, ne);
    h->bkg = _gps05_up_d(bkg, ne);
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
        double* t_mass = _gps05_up_d(mass, ne * c->n_mass);
        double* t_mom = _gps05_up_d(mom, ne * c->n_momentum);
        double* t_ang = _gps05_up_d(ang, ne * c->n_angle_total * c->n_comp);
        if (!t_mass || !t_mom || !t_ang) {
            if (t_mass) cudaFree(t_mass);
            if (t_mom) cudaFree(t_mom);
            if (t_ang) cudaFree(t_ang);
            goto fail;
        }
        {
            const long long total = (long long)ne * c->n_uslot;
            const int threads = GPS05_BLOCK;
            long long blocks = (total + threads - 1) / threads;
            if (blocks > 65535) blocks = 65535;
            if (blocks < 1) blocks = 1;
            gps05_fill_us<<<(unsigned)blocks, threads, 0, c->stream>>>(
                c->bas_r, c->bas_i, c->uslot_row, c->uslot_col,
                c->m_min, c->m_delta, c->m_bins,
                t_mass, c->n_mass, ne, c->n_uslot, h->us);
            CUDA_CHECK(cudaGetLastError());
        }
        {
            int fb = c->n_uniq < 32 ? 32 : (c->n_uniq < GPS05_BLOCK
                                            ? GPS05_BLOCK : c->n_uniq);
            if (fb > 1024) fb = 1024;
            const size_t shm = (size_t)c->n_ab * sizeof(double);
            gps05_fill_amp<<<ne, fb, shm, c->stream>>>(
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

void cuda_free_data_flat_pwa_gen_s0e5(void* vh) {
    GPS05Handle* h = (GPS05Handle*)vh;
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
static void gps05_launch(GPS05Ctx* c, GPS05Handle* h, int st, int nb,
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
    gps05_tile_kernel<<<tiles, GPS05_BLOCK, (size_t)c->smem_bytes, c->stream>>>(
        h->us + (size_t)st * c->n_uslot,
        h->amp + (size_t)st * c->n_uniq * c->n_proj,
        c->slot_of_wave, c->nz_m, c->nz_u, c->nz_pair, c->nz_k,
        c->gs_lo, c->gs_hi,
        c->key_lo, c->kw_lo, c->wave_of, c->nz_key, c->n_key,
        c->gamma, c->n_nz, c->p_coeff, c->p_ck_r, c->p_ck_i,
        h->weight + st, h->bkg + st,
        norm, use_norm, c->rsize, per_blk, c->te,
        N, c->n_proj, c->n_basis, c->n_uslot, c->n_uniq, nb,
        h->P_dev + st,
        c->g_ck_r, c->g_ck_i, c->g_coeff, c->acc, c->acc + 1);
}

void cuda_compute_flat_pwa_gen_s0e5(
    void* vctx, void* vh, const double* ck_r, const double* ck_i,
    const double* coeff, double norm, int use_norm,
    double* oQ, double* oDn, double* oP,
    double* og_ck_r, double* og_ck_i, double* og_coeff)
{
    GPS05Ctx* c = (GPS05Ctx*)vctx;
    GPS05Handle* h = (GPS05Handle*)vh;
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
    /* the accumulators span the WHOLE sample: zero once, read once */
    cudaMemsetAsync(c->acc, 0, 2 * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_ck_r, 0, (size_t)N * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_ck_i, 0, (size_t)N * sizeof(double), c->stream);
    cudaMemsetAsync(c->g_coeff, 0, (size_t)NB * sizeof(double), c->stream);

    const int ne_total = h->ne;
    const int bs = (c->cap > 0) ? c->cap : 1;
    for (int st = 0; st < ne_total; st += bs) {
        const int nb = (ne_total - st > bs) ? bs : (ne_total - st);
        gps05_launch(c, h, st, nb, norm, use_norm);
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

/* The tile size `gps05_layout` gives for an arbitrary shape — the same
 * function the kernel and `create` use, so a test can assert the buffer
 * arithmetic without a second, drifting copy in Python. */
long long cuda_tile_bytes_flat_pwa_gen_s0e5(int te, int n_wave, int n_proj,
                                           int n_basis)
{
    GPS05Layout L = gps05_layout(te, n_proj * n_wave, n_proj, n_wave, n_basis);
    return (long long)L.total * (long long)sizeof(double);
}

int cuda_smem_bytes_flat_pwa_gen_s0e5(void* vctx) {
    GPS05Ctx* c = (GPS05Ctx*)vctx;
    return c ? c->smem_bytes : 0;
}

/* Runtime resource report (host side, no kernel launch).
 *
 * CUDA exposes the compiled resource usage per FUNCTION, which is what the
 * `__launch_bounds__` / `-D..._MIN_BLOCKS` choice is trading against:
 *   out[0] registers/thread          (cudaFuncGetAttributes.numRegs)
 *   out[1] local bytes > 0  => the kernel SPILLS
 *   out[2] blocks/SM at the REAL block size and dynamic smem
 *   out[3] warps/SM  (= blocks x GPS05_BLOCK/32)
 *   out[4] maxThreadsPerBlock
 *   out[5] static shared bytes
 * Everything here is a property of the compiled cubin; `smem` is the tile's
 * dynamic shared memory for the CURRENT shape (c->smem_bytes), so the
 * occupancy figure is the one the fit actually gets.
 */
extern "C" void cuda_stats_flat_pwa_gen_s0e5(int smem, int* out)
{
    if (!out) return;
    cudaFuncAttributes a;
    cudaFuncGetAttributes(&a, (const void*)gps05_tile_kernel);
    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, (const void*)gps05_tile_kernel, GPS05_BLOCK, (size_t)(smem > 0 ? smem : 0));
    out[0] = a.numRegs;
    out[1] = (int)a.localSizeBytes;
    out[2] = blocks;
    out[3] = blocks * (GPS05_BLOCK / 32);
    out[4] = a.maxThreadsPerBlock;
    out[5] = (int)a.sharedSizeBytes;
}

int cuda_cap_flat_pwa_gen_s0e5(void* vctx) {
    GPS05Ctx* c = (GPS05Ctx*)vctx;
    return c ? c->cap : 0;
}

int cuda_ngroups_flat_pwa_gen_s0e5(void* vctx) {
    GPS05Ctx* c = (GPS05Ctx*)vctx;
    return c ? c->s_ngroups : 0;
}

int cuda_te_flat_pwa_gen_s0e5(void* vctx) {
    GPS05Ctx* c = (GPS05Ctx*)vctx;
    return c ? c->te : 0;
}

int cuda_get_device_count_flat_pwa_gen_s0e5(void) {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
}

void cuda_get_device_name_flat_pwa_gen_s0e5(char* buf, int len) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    snprintf(buf, len, "%s", p.name);
}

}  /* extern "C" */
