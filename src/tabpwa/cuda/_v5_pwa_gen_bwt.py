"""
CUDA kernel v5 PWA gen bwt — **case-1 copy** of
:mod:`~tabpwa.cuda._v5_pwa_gen_bw`, tuned by the profile below.  Same model,
same host contract, same algebra; only the fused tile changed:

    bw_b(m)   = Σ_{idx = off_bw[b]}^{off_bw[b+1]-1} coeff[idx] · T[g_idx[idx]](m)
    f         = w·n_res + r,      b = bw_of_factor[f]          (== bw_order[f])
    BW_w(m)   = Π_r bw_{b(w,r)}(m)
    common_w  = amp_w / BW_w
    ∂Q/∂coeff[idx] = Σ_{(w,r): b(w,r)=b} Σ_e (∂Q/∂BW_w)
                       · Π_{r'≠r} bw_{b(w,r')}(m_e) · T[g_idx[idx]](m_e)

Profiled changes (see the ``.cu`` header for the full record):

1. ``S0E_TE`` 8 → 4 — the 50 KiB tile at 198 waves allowed only one resident
   block per SM; 26 KiB puts 3 in flight.
2. one fp64 reciprocal + two multiplies instead of two fp64 divides for the
   complex ``1/BW`` (an fp64 divide is ~24× an fp64 FMA here).

Measured with the interleaved A/B harness (``tests/profile_pwa_gen_ab.py``):
**0.74–0.80×** the ``gen_bw`` compute time across shapes, sizes and
``resolution_size``; Q/P/ck/coeff/dnorm agree to ≤1e-15.

The remainder documents the ``gen_bw`` layout, unchanged here.

* ``off_bw`` int32 ``[n_bw + 1]`` — the contiguous slot range of each **base
  BW**, shared by every (wave, factor) that refers to it;
* ``g_idx`` int32 ``[n_basis_tot]`` — the deduped ``(combined-table row,
  mass column)`` interp slot of each per-BW coefficient;
* ``foff`` / ``flist`` int32 — the ``factors_of_bw`` CSR
  (``[n_bw + 1]`` / ``[n_fact]``): which (wave, factor) entries use each BW;
* ``bw_of_factor`` int32 ``[n_fact]`` — ``bw_order`` passed straight through;
  ``bw_of_slot`` int32 ``[n_basis_tot]`` — the reverse map;
* ``off_lo`` / ``off_hi`` int32 ``[n_fact]`` — ``off_bw[b(f)]`` /
  ``off_bw[b(f)+1]``, i.e. the per-(wave, factor) view of the same ranges
  (no dependent runtime indirection in the hot loop);
* ``coeff`` / ``dcoeff`` float64 ``[n_basis_tot]`` with
  ``n_basis_tot = off_bw[n_bw]`` — the weights stored **once per BW**.

The flat sibling (:mod:`~tabpwa.cuda._v5_pwa_gen_flat`) repeats the same
weights once per wave: its block has ``n_wave·n_basis`` entries where
``n_basis`` suffice (the kmatrix2 fixture: 6·5 = 30 vs 5; a 198-wave config:
990 vs 5).  Storing each weight once is what makes the kernel *cheaper*, not
just smaller: the basis matvec is evaluated once per **(event, BW)** and read
by every referring wave (the flat kernel re-evaluates it per wave), and the
coefficient gradient sums all copies of a BW in a warp reduction, then reads
each basis value once per **(event, BW, slot)** — one write per slot instead
of one per copy.  Measured on the 198-wave kmatrix2 shape: 0.76–0.86× the
flat kernel's compute time (see ``tests/benchmark_pwa_gen_bw.py``).

The multi-factor gradient is the ``cpu_pwa_gen`` model (see
``tabpwa/cpu/_pwa_gen.py``): the factors are multiplied, and the
``1/bw_{b(w,r)}`` of the reference formula is written as the prefix·suffix
product of the *other* factors — never a division by a factor that can
vanish.  At ``n_res == 1`` the empty product is exactly 1.

Everything else is the gen-flat / s0e gen machinery verbatim: the fused
per-iteration tile (BW / 1/BW / common / common⁄BW / A_p / dQ⁄dA_p all in
shared memory, one launch, no batching), the grouped log-sum data NLL with
the group **total** weight (``Q = -Σ_g W_g·log(S_g/W_g)``), the combined
basis table the context owns (raw ``gamma_table`` rows + the pre-rotated
"1" / "m²" rows + the ``den_table`` rows) and the ``pwa_gen`` numerator
``N_{n_index[w]}(m_e)`` folded into the amp-cache fill (load time).

The host contract is **unchanged**: ``compute`` takes ``params["coeff"]`` in
the padded ``(n_fact, n_basis)`` form produced by
:class:`~tabpwa.kernel_params.PWAGenKernelParams` — every copy of a BW
carries the same weights, so the per-BW block is built from each BW's
representative copy — and returns ``grads["coeff"]`` in the same padded shape
the other ``pwa_gen`` kernels return.  The per-BW gradient is expanded back
over the BW's factor copies as an **equal share** ``g_b / n_copies``: the
unchanged ``PWAGenKernelParams._facts_of_bw`` then sums the copies and
recovers the total derivative ``∂Q/∂w_b = Σ_copies ∂Q/∂coeff_copy`` (a
literal broadcast would multiply the fitter's gradient by ``n_copies``).  A
default-basis config (no ``den_row``) keeps the ``m0``/``g0`` path of
``cuda_v5_pwa_gen_s0e``.
"""

import warnings

import numpy as np
from cffi import FFI

_ffi = FFI()
_ffi.cdef("""
void* cuda_create_context_v5_gen_bwt(
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
    int n_m0p,int n_g0p,
    int batch_size,
    const int* slot_of_wave,int n_slot,
    const int* rep_of_slot,int n_rep,
    int n_uniq,int n_proj,
    const int* row_start,int n_rs,
    int n_basis_tot,
    int rsize,
    const int* uslot_row,int n_ur,
    const int* uslot_col,int n_uc,
    const int* off_bw,int n_offb,
    const int* off_lo,int n_offlo,
    const int* off_hi,int n_offhi,
    const int* g_idx,int n_gidx,
    const int* foff,int n_foff,
    const int* flist,int n_flist,
    const int* bw_of_factor,int n_bwof,
    const int* bw_of_slot,int n_bwos,
    const int* slot_l,int n_sl,
    int n_gamma_rows_ext,int gamma_table_bins_ext,
    int generic_den,
    const double* dtr,int n_dtr, const double* dti,int n_dti, int n_den_rows,
    const double* ntr,int n_ntr, const double* nti,int n_nti,
    int n_num_rows,int n_num_bins,double n_min,double n_delta,
    const int* n_index,int n_ni, const int* n_mass_index,int n_nmi);
void cuda_free_context_v5(void*);
void* cuda_load_data_v5(void*,const double*,int,const double*,int,
    const double*,int,const double*,const double*,int);
void cuda_free_data_v5(void*);
void cuda_compute_v5_gen_bwt(void*,void*,
    const double*,const double*,const double*,
    double,int,
    double*,double*,double*,double*,double*,double*);
void cuda_gram_matrix_v5(void*,void*,
    const double*,double*,double*);
int cuda_get_device_count();
int cuda_get_device_name(char*,int);
""")


def _load_lib():
    from tabpwa.cuda.build import ensure
    lib_path = ensure("kernels_v5_pwa_gen_bwt.cu",
                      "libcuda_kernels_v5_pwa_gen_bwt.so")
    if not lib_path:
        raise RuntimeError(
            f"v5 PWA gen bw CUDA library not found at {lib_path}.")
    return _ffi.dlopen(lib_path)


def _amp_cache_layout(config, n_on):
    """Wave → minimal angular-cache-slot mapping (numerator aware).

    House-rule case-1 copy of the shared layout: the dedup key stays
    **equal** to :func:`tabpwa.amp_cache.build_amp_cache_layout` —
    matrix_angle column + fl_order row — extended by the per-wave numerator
    row ``n_index`` / mass column when the numerator is actually used (the
    numerator is folded into the cached ``Amp``, so waves sharing an angular
    entry but carrying different numerator rows must NOT share a slot).
    """
    n_wave = config["matrix_angle"].shape[1]
    n_decay = config["fl_order"].size // n_wave
    cols = config["matrix_angle"].T
    fl_rows = np.asarray(config["fl_order"]).reshape(n_wave, n_decay)
    # factors per wave: the numerator index is laid out per (wave, factor)
    n_res = max(1, np.asarray(config["bw_order"]).size // n_wave)
    if n_on:
        n_idx = np.asarray(config["n_index"], dtype=np.int64)
        # per-(wave, factor) numerator rows and the mass column each factor
        # reads (``mass_index[bw_order]``, the denominator's own choice);
        # both go into the cache key, since they change the cached amp.
        n_col = np.asarray(config["mass_index"], dtype=np.int64)[
            np.asarray(config["bw_order"], dtype=np.int64)]
    keys = {}
    slot_of_wave = np.zeros(n_wave, dtype=np.int32)
    rep_of_slot = []
    for w in range(n_wave):
        key = (cols[w].tobytes(), fl_rows[w].tobytes(),
               n_idx[w * n_res:(w + 1) * n_res].tobytes() if n_on else 0,
               n_col[w * n_res:(w + 1) * n_res].tobytes() if n_on else 0)
        s = keys.get(key)
        if s is None:
            s = len(rep_of_slot)
            keys[key] = s
            rep_of_slot.append(w)
        slot_of_wave[w] = s
    return len(rep_of_slot), slot_of_wave, np.array(rep_of_slot, dtype=np.int32)


class DataHandle:
    def __init__(self, ptr, lib, ne):
        self.ptr = ptr
        self._lib = lib
        self.ne = ne
        self._keep = []

    def free(self):
        if self.ptr is not None:
            self._lib.cuda_free_data_v5(self.ptr)
            self.ptr = None
            self._keep.clear()

    def __del__(self):
        self.free()


class CUDAKernelV5PWAGenBWT:
    """Per-base-BW ``pwa_gen`` fused tile + group-total-weight log-sum NLL.

    ``params`` needs ``ck`` (length N = n_wave/n_proj) plus either
    ``m0``/``g0`` (default basis) or ``coeff`` ``(n_fact, n_basis)`` padded
    (explicit ``den_row`` basis).  Gradients mirror the same keys — the
    ``coeff`` gradient is returned **padded**, exactly like
    ``cuda_v5_pwa_gen_flat``.
    """

    def __init__(self, config, batch_size=50000, resolution_size=1,
                 lib_path=None):
        self._ctx = None
        self._lib = None
        self._ka = []
        self._lib = _load_lib()
        n_dev = self._lib.cuda_get_device_count()
        if n_dev > 0:
            name_buf = _ffi.new("char[256]")
            self._lib.cuda_get_device_name(name_buf, 256)
            print(f"v5 PWA gen bwt GPU: {_ffi.string(name_buf).decode()}")
        else:
            raise RuntimeError("No CUDA devices found")

        c = config
        self.config = c
        self.n_wave = c["matrix_angle"].shape[1]     # P·N entries per event
        self.n_proj = int(c.get("n_proj", 1) or 1)
        if self.n_wave % self.n_proj != 0:
            raise ValueError(
                f"v5 PWA gen bw: n_wave={self.n_wave} not divisible by "
                f"n_proj={self.n_proj}")
        self.n_wave_base = self.n_wave // self.n_proj  # N = ck length
        # ── factors per wave: n_res >= 1, any number ────────────────
        self.bw_order = np.ascontiguousarray(
            np.asarray(c["bw_order"]).reshape(-1), np.int64)
        if self.bw_order.size % self.n_wave != 0:
            raise ValueError(
                f"cuda_v5_pwa_gen_bwt: bw_order size {self.bw_order.size} "
                f"is not a multiple of n_wave={self.n_wave}")
        self.n_res = self.bw_order.size // self.n_wave        # BW factors
        self.n_fact = self.n_wave * self.n_res                # (wave, factor)
        # the BW slot of each (wave, factor) entry — bw_order passes through
        if np.any(self.bw_order < 0):
            raise ValueError("cuda_v5_pwa_gen_bwt: bw_order must be >= 0")
        self.n_bw = int(self.bw_order.max()) + 1 if self.bw_order.size else 0
        self.bw_of_factor = np.ascontiguousarray(
            self.bw_order.astype(np.int32))
        # factors_of_bw CSR: which (wave, factor) entries use each BW
        self.foff = np.concatenate(
            [[0], np.cumsum(np.bincount(self.bw_order,
                                        minlength=self.n_bw))]
        ).astype(np.int32) if self.n_bw else np.zeros(1, np.int32)
        self.flist = np.ascontiguousarray(
            np.argsort(self.bw_order, kind="stable").astype(np.int32))
        for b in range(self.n_bw):
            if int(self.foff[b + 1]) == int(self.foff[b]):
                raise ValueError(
                    f"cuda_v5_pwa_gen_bwt: base BW {b} is referenced by no "
                    f"(wave, factor) entry (bw_order={self.bw_order.tolist()})")
        self._bws = np.arange(self.n_bw, dtype=np.int64)
        mass_index_bw = np.asarray(c["mass_index"])[self.bws_idx]  # [n_bw]

        self.n_decay = c["fl_order"].size // self.n_wave
        self.n_mass = int(np.max(c["mass_index"])) + 1
        self.n_momentum = int(np.max(c["fl_q_index"])) + 1
        self.n_angle_k = c["angle_k"].shape[0]
        self.n_angle_total = (int(np.max(c["angle_index"])) + 1
                              if len(c["angle_index"]) > 0 else 0)
        self.n_angle_comp = c["angle_k"].shape[-1]
        self.n_m0_params = int(np.max(c["m0_index"])) + 1
        self.n_g0_params = int(np.max(c["g0_index"])) + 1

        gt = np.asarray(c["gamma_table"])
        # raw gamma-table row count; the kernel appends "1" and "m²" after it
        one_row = int(np.max(c["g0_index"])) + 1
        if gt.shape[0] != one_row:
            raise ValueError(
                f"cuda_v5_pwa_gen_bwt: gamma_table has {gt.shape[0]} rows, "
                f"expected max(g0_index)+1 = {one_row}")

        # ── numerator (fixed data; applied in the amp-cache fill) ───
        from tabpwa.pwa_gen_support import has_numerator
        self._n_on = has_numerator(c)
        if self._n_on:
            self._n_table = np.asarray(c["n_table"], np.complex128)
            self._n_index = np.ascontiguousarray(
                np.asarray(c["n_index"], np.int32))
            # per (wave, factor): the factor's OWN BW mass column, i.e. the
            # the factor's own BW mass column — the same
            # ``mass_index[bw_order[f]]`` the denominator uses
            self._n_mass_index = np.ascontiguousarray(
                np.asarray(c["mass_index"])[self.bw_order], np.int32)
            # per (wave, factor), like ``bw_order``: each factor reads its
            # OWN resonance's mass column
            if self._n_index.size != self.n_fact \
                    or self._n_mass_index.size != self.n_fact:
                raise ValueError(
                    "cuda_v5_pwa_gen_bwt: n_index/n_mass_index must have "
                    f"{self.n_fact} entries (n_wave·n_res)")
            self._n_min = float(c["n_min"])
            self._n_delta = float(c["n_delta"])
            if self._n_delta == 0.0:
                raise ValueError("cuda_v5_pwa_gen_bwt: n_delta must be != 0")
        else:
            self._n_table = None
            self._n_index = self._n_mass_index = None
            self._n_min = 0.0
            self._n_delta = 1.0

        # ── denominator basis: explicit (den_row) or the s0 preset ──
        self._generic = "den_row" in c
        if self._generic:
            self.n_gamma_rows = 0
            self._init_generic(c, mass_index_bw, one_row)
        else:
            self._init_plain(c, mass_index_bw, one_row)

        # ── angular amplitude cache layout ──────────────────────────
        self.n_uniq, self.slot_of_wave, self.rep_of_slot = \
            _amp_cache_layout(c, self._n_on)

        # v5: the group size of the log-sum data NLL
        self.resolution_size = int(resolution_size)
        if self.resolution_size < 1:
            raise ValueError(
                f"cuda_v5_pwa_gen_bwt: resolution_size must be >= 1 "
                f"(got {self.resolution_size})")
        if self.resolution_size > 1 and batch_size < self.resolution_size:
            raise ValueError(
                f"cuda_v5_pwa_gen_bwt: batch_size ({batch_size}) must be >= "
                f"resolution_size ({self.resolution_size}) so the chunk "
                f"stride can stay resolution-aligned")
        self._last_dnorm = None      # d(NLL)/d(norm) from the last norm call
        self.dnorm_on_gpu = True

        self._create_context(c, gt, batch_size)

    @property
    def bws_idx(self):
        """The base-BW indices (``arange(n_bw)``) used to index the model
        tables (``mass_index`` / ``den_row`` / ``matrix_gamma``)."""
        return self._bws

    # ── per-BW CSR layout helpers ───────────────────────────────────
    def _finish_bw(self, widths):
        """``off_bw`` from the per-BW widths; per-BW ↔ padded mapping."""
        widths = np.asarray(widths, np.int64)
        if widths.size != self.n_bw:
            raise ValueError(
                f"cuda_v5_pwa_gen_bwt: {widths.size} BW widths for "
                f"n_bw={self.n_bw}")
        if np.any(widths < 1):
            raise ValueError(
                "cuda_v5_pwa_gen_bwt: every base BW needs at least one "
                f"basis slot (got {widths.tolist()})")
        self.n_basis_w = widths                     # per BW
        self.off_bw = np.concatenate(
            [[0], np.cumsum(widths)]).astype(np.int32)
        self.n_basis_tot = int(self.off_bw[-1])
        # the hot loop of the tile kernel reads the bounds of each
        # (wave, factor) copy directly (no dependent load chain, exactly the
        # flat sibling's addressing) — the ranges are the BASE BW's
        self.off_lo = np.ascontiguousarray(
            self.off_bw[self.bw_of_factor], np.int32)
        self.off_hi = np.ascontiguousarray(
            self.off_bw[self.bw_of_factor + 1], np.int32)
        # per-BW slot -> its base BW (phase D2 of the tile kernel)
        self.bw_of_slot = np.repeat(
            np.arange(self.n_bw, dtype=np.int32), widths)
        # the number of (wave, factor) copies of each BW
        self.n_copies = np.diff(self.foff).astype(np.int64)
        # representative factor copy of each BW (padded row of the block)
        self._rep_fact = np.asarray(
            [int(self.flist[int(self.foff[b])]) for b in range(self.n_bw)],
            np.int64)
        # per-BW slot -> padded (factor, col) flat index (grad re-expansion)
        src = np.empty(self.n_basis_tot, np.int64)
        for b in range(self.n_bw):
            b0 = int(self.off_bw[b])
            wd = int(widths[b])
            src[b0:b0 + wd] = self._rep_fact[b] * self.n_basis + np.arange(wd)
        self._bw_src = src
        # per-BW slot -> every (factor, col) padded index that carries it,
        # plus the 1/n_copies share of the total (vectorised _unpack)
        dst, srcio, share = [], [], []
        for b in range(self.n_bw):
            b0 = int(self.off_bw[b])
            wd = int(widths[b])
            sl = np.arange(b0, b0 + wd)
            sh = 1.0 / float(self.n_copies[b])
            for f in self.flist[int(self.foff[b]):int(self.foff[b + 1])]:
                dst.append(int(f) * self.n_basis + np.arange(wd))
                srcio.append(sl)
                share.append(np.full(wd, sh))
        self._unp_dst = np.concatenate(dst)
        self._unp_src = np.concatenate(srcio)
        self._unp_share = np.concatenate(share)

    def _init_plain(self, c, mass_index_bw, one_row):
        """s0 preset, per base BW: s0, −1, then the BW's γ rows.

        Each base BW owns one preset — ``s0 = m0²``, ``−1`` (the ``−m²``
        row) and one coefficient per γ row of its ``matrix_gamma`` column —
        shared by every (wave, factor) copy, exactly the sum of what the flat
        sibling stores once per copy.
        """
        self.n_basis = 0
        m0_index_bw = np.asarray(c["m0_index"])[self.bws_idx]     # [n_bw]
        mg_bw = np.asarray(c["matrix_gamma"])[:, self.bws_idx]    # (ng, n_bw)
        rows_idx, rows_bw = np.nonzero(mg_bw.T)                   # BW-major
        # merged gamma-row index of each per-BW slot (this kernel reads the
        # u-cache directly, so the device table is unused; kept for the
        # create signature of the siblings)
        self._gamma_col_idx = np.ascontiguousarray(rows_idx.astype(np.int32))
        self._bw_of_row = np.ascontiguousarray(rows_bw.astype(np.int64))
        g0_index_r = np.asarray(c["g0_index"])[self._bw_of_row]
        g0_mass_index_r = np.asarray(c["g0_mass_index"])[self._bw_of_row]
        self._g0_index_r = g0_index_r
        self._g0_index = np.asarray(c["g0_index"])
        self._g0_mass_index = np.asarray(c["g0_mass_index"])
        self._g0_mass_index_r = g0_mass_index_r
        self._m0_index_bw = m0_index_bw
        self._mass_index_bw = mass_index_bw                # [n_bw]
        self.n_gamma_rows = int(rows_bw.size)
        self.n_unique_bw = self.n_bw
        self.row_start = np.searchsorted(
            self._bw_of_row, np.arange(self.n_bw),
            side="left").astype(np.int32)
        rows_per_bw = np.diff(
            np.append(self.row_start, self.n_gamma_rows)).astype(np.int64)
        # padded n_basis of the s0e gen sibling (comparison / anchors only)
        self.n_basis = int(2 + (int(rows_per_bw.max()) if self.n_bw else 0))

        uniq = {}
        uslot_row, uslot_col = [], []

        def _uslot(row, col):
            key = (int(row), int(col))
            if key not in uniq:
                uniq[key] = len(uslot_row)
                uslot_row.append(key[0])
                uslot_col.append(key[1])
            return uniq[key]

        self._finish_bw(2 + rows_per_bw)
        g_idx = np.empty(self.n_basis_tot, np.int32)
        slot_l = np.full(self.n_basis_tot, -1, np.int32)
        for b in range(self.n_bw):
            b0 = int(self.off_bw[b])
            r0 = int(self.row_start[b])
            r1 = (int(self.row_start[b + 1]) if b + 1 < self.n_bw
                  else self.n_gamma_rows)
            g_idx[b0] = _uslot(one_row, 0)                 # "1"  (constant)
            g_idx[b0 + 1] = _uslot(one_row + 1,
                                   int(mass_index_bw[b]))  # "m²"
            for j in range(r0, r1):
                g_idx[b0 + 2 + j - r0] = _uslot(
                    int(g0_index_r[j]), int(g0_mass_index_r[j]))
                slot_l[b0 + 2 + j - r0] = j
        self.n_uslot = len(uslot_row)
        self.uslot_row = np.asarray(uslot_row, np.int32)
        self.uslot_col = np.asarray(uslot_col, np.int32)
        self.g_idx = g_idx
        self.slot_l = slot_l
        self._n_den_rows = 0
        self._den_table = None

    def _init_generic(self, c, mass_index_bw, one_row):
        """Explicit ``den_row`` basis, packed per base BW, no padding.

        ``den_row`` pads a narrower resonance with the ``ZERO_KIND`` (-1)
        sentinel; those columns are trailing and carry coefficient 0, so the
        per-BW layout simply drops them.  Anything but a trailing padding
        block would break the "contiguous range" contract and is rejected.
        """
        enc = np.asarray(c["den_row"])
        if enc.ndim != 2:
            raise ValueError("cuda_v5_pwa_gen_bwt: den_row must be 2-D")
        enc_bw = enc[self.bws_idx]                     # [n_bw, n_basis]
        self.n_basis = int(enc.shape[1])
        widths = np.zeros(self.n_bw, np.int64)
        for b in range(self.n_bw):
            live = np.nonzero(enc_bw[b] >= 0)[0]
            wd = int(live.max()) + 1 if live.size else 0
            if wd and not np.all(enc_bw[b][:wd] >= 0):
                raise ValueError(
                    "cuda_v5_pwa_gen_bwt: den_row padding must be a "
                    f"trailing block (BW {b}: {enc_bw[b].tolist()})")
            widths[b] = wd
        self._mass_index_bw = mass_index_bw
        self.n_unique_bw = self.n_bw
        # the generic path never reads m0/g0/gamma-row indices on the device
        self._gamma_col_idx = np.zeros(0, np.int32)
        self._g0_index_r = np.zeros(0, np.int32)
        self._g0_mass_index_r = np.zeros(0, np.int32)
        self._bw_of_row = np.zeros(0, np.int64)
        self.row_start = np.zeros(0, np.int32)

        uniq = {}
        uslot_row, uslot_col = [], []

        def _uslot(row, col):
            key = (int(row), int(col))
            if key not in uniq:
                uniq[key] = len(uslot_row)
                uslot_row.append(key[0])
                uslot_col.append(key[1])
            return uniq[key]

        self._finish_bw(widths)
        g_idx = np.empty(self.n_basis_tot, np.int32)
        for b in range(self.n_bw):
            b0 = int(self.off_bw[b])
            col = int(mass_index_bw[b])
            for j in range(int(widths[b])):
                e = int(enc_bw[b, j])
                if e == 0:                                  # F = 1
                    g_idx[b0 + j] = _uslot(one_row, 0)
                elif e == 1:                                # F = −m²
                    g_idx[b0 + j] = _uslot(one_row + 1, col)
                else:                                       # F = den_table[r]
                    g_idx[b0 + j] = _uslot(one_row + 2 + e - 2, col)
        self.n_uslot = len(uslot_row)
        self.uslot_row = np.asarray(uslot_row, np.int32)
        self.uslot_col = np.asarray(uslot_col, np.int32)
        self.g_idx = g_idx
        self.slot_l = np.full(self.n_basis_tot, -1, np.int32)
        den_table = np.asarray(c.get("den_table"))
        self._n_den_rows = int(den_table.shape[0]) if den_table.size else 0
        self._den_table = np.asarray(den_table, np.complex128)

    # ── padded ↔ per-BW coefficient mapping ─────────────────────────
    def _pack(self, coeff_padded):
        """Padded ``(n_fact, n_basis)`` → per-BW ``[n_basis_tot]``.

        Every copy of a base BW carries the same weights
        (:meth:`PWAGenKernelParams.forward` broadcasts them), so one
        representative copy per BW is read.
        """
        flat = np.ascontiguousarray(coeff_padded, np.float64).reshape(-1)
        if flat.size != self.n_fact * self.n_basis:
            raise ValueError(
                f"cuda_v5_pwa_gen_bwt: coeff length {flat.size} != "
                f"n_fact·n_basis = {self.n_fact}·{self.n_basis}")
        return np.ascontiguousarray(flat[self._bw_src])

    def _unpack(self, gbw):
        """Per-BW ``[n_basis_tot]`` gradient → padded ``(n_fact, n_basis)``.

        ``g_b`` is the TOTAL derivative of the shared weights (all copies
        summed), so it is spread as an equal share ``g_b / n_copies`` over
        the copies: the unchanged host sum
        (:meth:`PWAGenKernelParams._facts_of_bw`) then recovers ``g_b``.
        """
        gp = np.zeros(self.n_fact * self.n_basis, np.float64)
        gbw = np.asarray(gbw, np.float64).reshape(-1)
        if gbw.size != self.n_basis_tot:
            raise ValueError(
                f"cuda_v5_pwa_gen_bwt: per-BW gradient length {gbw.size} != "
                f"n_basis_tot = {self.n_basis_tot}")
        gp[self._unp_dst] = gbw[self._unp_src] * self._unp_share
        return gp.reshape(self.n_fact, self.n_basis)

    # ── context ─────────────────────────────────────────────────────
    def _create_context(self, c, gt, batch_size):
        self._ka = []

        def _db(a):
            arr = np.ascontiguousarray(a, np.float64)
            buf = _ffi.from_buffer(arr)
            self._ka.append(buf)
            return _ffi.cast("double*", buf)

        def _ib(a):
            arr = np.ascontiguousarray(a, np.int32)
            buf = _ffi.from_buffer(arr)
            self._ka.append(buf)
            return _ffi.cast("int*", buf)

        def _ib_or_null(a):
            return _ib(a) if len(a) else _ffi.NULL

        ma = c["matrix_angle"]
        one_row = int(np.max(c["g0_index"])) + 1
        gt = np.asarray(gt)
        gt_r = np.ascontiguousarray(np.real(gt).flatten(), np.float64)
        gt_i = np.ascontiguousarray(np.imag(gt).flatten(), np.float64)
        if self._generic and self._n_den_rows > 0:
            dt = self._den_table
            if dt.shape[1] != gt.shape[-1]:
                raise ValueError(
                    "cuda_v5_pwa_gen_bwt: den_table and gamma_table grids "
                    f"differ ({dt.shape[1]} vs {gt.shape[-1]} bins)")
            dt_r = np.ascontiguousarray(np.real(dt).flatten(), np.float64)
            dt_i = np.ascontiguousarray(np.imag(dt).flatten(), np.float64)
            n_den_rows = self._n_den_rows
        else:
            dt_r = dt_i = np.zeros(0, np.float64)
            n_den_rows = 0
        if self._n_on:
            nt = self._n_table
            nt_r = np.ascontiguousarray(np.real(nt).flatten(), np.float64)
            nt_i = np.ascontiguousarray(np.imag(nt).flatten(), np.float64)
        else:
            nt_r = nt_i = np.zeros(0, np.float64)

        self._ctx = self._lib.cuda_create_context_v5_gen_bwt(
            _ffi.NULL, 0,                      # m0_index: host-side only
            _ib_or_null(self._g0_index_r), len(self._g0_index_r),
            _ib_or_null(self._g0_mass_index_r), len(self._g0_mass_index_r),
            _ib(self._mass_index_bw), len(self._mass_index_bw),
            _ib(c["fl_type"]), len(c["fl_type"]),
            _ib(c["fl_q_index"]), len(c["fl_q_index"]),
            _ffi.NULL, 0,                                       # no bw_order
            _ib(c["fl_order"]), len(c["fl_order"]),
            _ib(c["angle_index"]), len(c["angle_index"]),
            _db(c["angle_k"].flatten()), c["angle_k"].size,
            _db(c["angle_b"].flatten()), c["angle_b"].size,
            _db(np.real(ma).flatten()), ma.size,
            _db(np.imag(ma).flatten()), ma.size,
            _db(gt_r), gt_r.size,                              # RAW table
            _db(gt_i), gt_i.size,
            float(c["gamma_min"]), float(c["gamma_delta"]),
            gt.shape[-1],
            _ffi.NULL, 0,                    # matrix_gamma: wrapper-side only
            _ib_or_null(self._gamma_col_idx), len(self._gamma_col_idx),
            _db(c["fl_table"].flatten()), c["fl_table"].size,
            float(c["fl_min"]), float(c["fl_delta"]), c["fl_table"].shape[-1],
            self.n_wave, self.n_res, self.n_decay,
            self.n_unique_bw, self.n_gamma_rows,
            self.n_mass, self.n_momentum,
            self.n_angle_k, self.n_angle_total, self.n_angle_comp,
            self.n_m0_params, self.n_g0_params, batch_size,
            _ib(self.slot_of_wave), len(self.slot_of_wave),
            _ib(self.rep_of_slot), len(self.rep_of_slot),
            self.n_uniq, self.n_proj,
            _ib_or_null(self.row_start), len(self.row_start),
            self.n_basis_tot,
            self.resolution_size,
            _ib(self.uslot_row), len(self.uslot_row),
            _ib(self.uslot_col), len(self.uslot_col),
            _ib(self.off_bw), len(self.off_bw),
            _ib(self.off_lo), len(self.off_lo),
            _ib(self.off_hi), len(self.off_hi),
            _ib(self.g_idx), len(self.g_idx),
            _ib(self.foff), len(self.foff),
            _ib(self.flist), len(self.flist),
            _ib(self.bw_of_factor), len(self.bw_of_factor),
            _ib(self.bw_of_slot), len(self.bw_of_slot),
            _ib(self.slot_l), len(self.slot_l),
            one_row + (2 if not self._generic else 2 + n_den_rows),
            gt.shape[-1],
            1 if self._generic else 0,
            (_db(dt_r) if n_den_rows > 0 else _ffi.NULL), dt_r.size,
            (_db(dt_i) if n_den_rows > 0 else _ffi.NULL), dt_i.size,
            n_den_rows,
            (_db(nt_r) if self._n_on else _ffi.NULL), nt_r.size,
            (_db(nt_i) if self._n_on else _ffi.NULL), nt_i.size,
            (self._n_table.shape[0] if self._n_on else 0),
            (self._n_table.shape[1] if self._n_on else 0),
            self._n_min, self._n_delta,
            # per (wave, factor) -- n_fact entries, NOT n_wave: the arrays
            # are indexed ``f = w·n_res + r``, so a short length would upload
            # only the first wave's factors and leave the rest of the device
            # buffer uninitialised (read back as 0)
            (_ib(self._n_index) if self._n_on else _ffi.NULL),
            (self._n_index.size if self._n_on else 0),
            (_ib(self._n_mass_index) if self._n_on else _ffi.NULL),
            (self._n_mass_index.size if self._n_on else 0))
        if self._ctx == _ffi.NULL:
            raise RuntimeError(
                "cuda_v5_pwa_gen_bwt: context creation failed "
                "(see the stderr message from the kernel)")

    # ── data ────────────────────────────────────────────────────────
    def load_data(self, data):
        ka = []

        def _db(a):
            arr = np.ascontiguousarray(a, np.float64)
            buf = _ffi.from_buffer(arr)
            ka.append(buf)
            return _ffi.cast("double*", buf)

        ne = data["mass"].shape[0]
        if ne % self.resolution_size != 0:
            warnings.warn(
                f"cuda_v5_pwa_gen_bwt: {ne} events are not a multiple of "
                f"resolution_size={self.resolution_size}; the final log-sum "
                f"group is partial.  Resolution-cloud data should contain "
                f"n_original x resolution_size rows.", RuntimeWarning,
                stacklevel=2)
        mass = data["mass"].reshape(ne, -1)[:, :self.n_mass]
        mom = data["q"].reshape(ne, -1)[:, :self.n_momentum]
        ang = data["angle"].reshape(ne, -1)[:, :self.n_angle_total
                                            * self.n_angle_comp]
        nang = self.n_angle_total
        bkg_key = "bkg_raw" if "bkg_raw" in data else "bkg"

        dh = DataHandle(self._lib.cuda_load_data_v5(
            self._ctx, _db(mass), mass.shape[1],
            _db(mom), mom.shape[1], _db(ang), nang,
            _db(data["weight"]), _db(data[bkg_key]), ne), self._lib, ne)
        dh._keep = ka
        return dh

    def _coeff_from_m0g0(self, params):
        """s0 preset (per BW): m0², −1, then m0·g0_j for the BW's γ rows."""
        nm_, gg_ = self.n_m0_params, self.n_g0_params
        m0 = np.zeros(nm_, np.float64)
        m0[:len(params["m0"])] = np.asarray(params["m0"])
        g0 = np.zeros(gg_, np.float64)
        g0[:len(params["g0"])] = np.asarray(params["g0"])
        m0_b = m0[self._m0_index_bw]                       # [n_bw]
        s0_b = m0_b * m0_b
        mult_r = m0_b[self._bw_of_row] * g0[self._g0_index_r]
        coeff = np.zeros(self.n_basis_tot, np.float64)
        for b in range(self.n_bw):
            b0 = int(self.off_bw[b])
            r0 = int(self.row_start[b])
            r1 = (int(self.row_start[b + 1]) if b + 1 < self.n_bw
                  else self.n_gamma_rows)
            coeff[b0] = s0_b[b]
            coeff[b0 + 1] = -1.0
            coeff[b0 + 2:b0 + 2 + (r1 - r0)] = mult_r[r0:r1]
        return coeff, m0_b, g0

    def compute(self, params, data_handle, norm=None, return_p=True):
        """Forward + gradients (s0 preset or explicit ``coeff`` basis)."""
        ck = params["ck"]
        nbase = self.n_wave_base           # N — ck / gradient buffer size
        use_norm = 0 if norm is None else 1
        norm_val = norm if norm is not None else 0.0

        if len(ck) != nbase:
            raise ValueError(
                f"v5 PWA gen bw: ck length {len(ck)} != n_wave/n_proj = "
                f"{nbase} ({self.n_wave}/{self.n_proj})")

        ck_r = np.real(ck).astype(np.float64)
        ck_i = np.imag(ck).astype(np.float64)

        if self._generic:
            # padded (n_fact, n_basis) coefficients -> per-BW block
            coeff = self._pack(params["coeff"])
        else:
            coeff, m0_b, g0 = self._coeff_from_m0g0(params)

        oQ = _ffi.new("double*")
        oDn = _ffi.new("double*")
        oP = (np.zeros(data_handle.ne, np.float64)
              if return_p else None)
        ogck_r = np.zeros(nbase, np.float64)
        ogck_i = np.zeros(nbase, np.float64)
        ogcoeff = np.zeros(self.n_basis_tot, np.float64)

        ka = []

        def _db(a):
            arr = np.ascontiguousarray(a, np.float64)
            buf = _ffi.from_buffer(arr)
            ka.append(buf)
            return _ffi.cast("double*", buf)

        self._lib.cuda_compute_v5_gen_bwt(
            self._ctx, data_handle.ptr,
            _db(ck_r), _db(ck_i), _db(coeff),
            norm_val, use_norm,
            oQ, oDn, (_db(oP) if oP is not None else _ffi.NULL),
            _db(ogck_r), _db(ogck_i), _db(ogcoeff))
        # v5 semantics: the device accumulates dNLL/dnorm (0 without a norm)
        self._last_dnorm = float(oDn[0])

        if self._generic:
            # per-BW gradients, re-expanded to the padded (n_fact, n_basis)
            # shape the other pwa_gen kernels return; the parameter chain
            # lives in the model's basis transform.
            grads = {"ck": ogck_r + 1j * ogck_i,
                     "coeff": self._unpack(ogcoeff)}
            return oQ[0], grads, (oP if return_p else None)

        # ── unpack the per-BW coefficient gradient to (s0, mult) ────
        ogs0 = np.zeros(self.n_bw, np.float64)
        ogmult = np.zeros(self.n_gamma_rows, np.float64)
        for b in range(self.n_bw):
            b0 = int(self.off_bw[b])
            r0 = int(self.row_start[b])
            r1 = (int(self.row_start[b + 1]) if b + 1 < self.n_bw
                  else self.n_gamma_rows)
            ogs0[b] = ogcoeff[b0]
            ogmult[r0:r1] = ogcoeff[b0 + 2:b0 + 2 + (r1 - r0)]

        # ── host chain: (s0, mult) -> (m0, g0), per BW ───────────────
        g0_row = g0[self._g0_index_r]
        dm0_b = 2.0 * m0_b * ogs0 + np.bincount(
            self._bw_of_row, weights=g0_row * ogmult,
            minlength=self.n_bw)
        grad_m0 = np.bincount(self._m0_index_bw, weights=dm0_b,
                              minlength=self.n_m0_params)
        grad_g0 = np.bincount(self._g0_index_r,
                              weights=m0_b[self._bw_of_row] * ogmult,
                              minlength=self.n_g0_params)

        grads = {"ck": ogck_r + 1j * ogck_i,
                 "m0": grad_m0, "g0": grad_g0}
        return oQ[0], grads, (oP if return_p else None)

    def compute_gram(self, phsp_handle, m0, g0):
        """Wave Gram matrix D (N, N) of a loaded phsp handle at m0/g0.

        Only the s0 preset can derive the coefficients from ``m0``/``g0``;
        an explicit ``pwa_gen`` basis must pass ``coeff`` through the
        backend, so the Gram path is not offered there (same rule as
        ``cuda_v5_pwa_gen_s0e``).
        """
        if self._generic:
            raise ValueError(
                "cuda_v5_pwa_gen_bwt: the Gram path needs an explicit "
                "coeff basis — use a backend that supports compute_gram "
                "with pwa_gen (or the plain s0e for an N ≡ 1 config)")
        n = self.n_wave_base
        m0_arr = np.zeros(self.n_m0_params, np.float64)
        m0_arr[:len(m0)] = np.asarray(m0)
        g0_arr = np.zeros(self.n_g0_params, np.float64)
        g0_arr[:len(g0)] = np.asarray(g0)
        coeff, _m0_b, _g0 = self._coeff_from_m0g0({"m0": m0_arr,
                                                  "g0": g0_arr})
        oDr = np.zeros(n * n, np.float64)
        oDi = np.zeros(n * n, np.float64)
        ka = []

        def _db(a):
            arr = np.ascontiguousarray(a, np.float64)
            buf = _ffi.from_buffer(arr)
            ka.append(buf)
            return _ffi.cast("double*", buf)

        self._lib.cuda_gram_matrix_v5(
            self._ctx, phsp_handle.ptr,
            _db(coeff), _db(oDr), _db(oDi))
        D = (oDr + 1j * oDi).reshape(n, n)
        return (D + D.conj().T) / 2.0

    def free(self):
        if self._ctx is not None:
            self._lib.cuda_free_context_v5(self._ctx)
            self._ctx = None
        self._ka.clear()

    def __del__(self):
        self.free()
