"""
CUDA kernel v5 PWA gen flat — the ``pwa_gen`` fused tile kernel of
``cuda_v5_pwa_gen_s0e`` with a **flat, unpadded, per-(wave, factor)
contiguous** denominator basis (CSR-style), for **any** ``n_res >= 1``
Breit–Wigner factors per wave:

    bw_{w,r}(m) = Σ_{idx = off[f]}^{off[f+1]-1} coeff[idx] · T[g_idx[idx]](m)
    BW_w(m)     = Π_r bw_{w,r}(m),                       f = w·n_res + r
    common_w    = amp_w / BW_w
    ∂Q/∂coeff[idx] = Σ_e (∂Q/∂BW_w) · Π_{r'≠r} bw_{w,r'} · T[g_idx[idx]](m_e)

* ``off`` int32 ``[n_fact + 1]``, ``n_fact = n_wave · n_res`` — the flat
  slot range of each **(wave, factor)**;
* ``g_idx`` int32 ``[n_basis_tot]`` — the deduped ``(combined-table row,
  mass column)`` interp slot of each flat coefficient (the flat analogue of
  the padded ``slot_map``);
* ``coeff`` / ``dcoeff`` float64 ``[n_basis_tot]`` with
  ``n_basis_tot = off[n_fact] = Σ_f n_basis_f`` — **no padded slots**, so a
  mixed decay (a 3-row resonance next to a 5-row one) carries
  ``n_basis_tot < n_fact · n_basis_max``.

The multi-factor gradient is the ``cpu_pwa_gen`` model (see
``tabpwa/cpu/_pwa_gen.py``): the factors are multiplied, and the
``1/bw_{w,r}`` of the reference formula is written as the prefix·suffix
product of the *other* factors — never a division by a factor that can
vanish.  At ``n_res == 1`` the empty product is exactly 1 and the numbers
collapse to the single-factor ones (bit-identical, see the kernel header).

Everything else is the s0e gen machinery verbatim: the fused per-iteration
tile (BW / 1/BW / common / common⁄BW / A_p / dQ⁄dA_p all in shared memory,
one launch, no batching), the grouped log-sum data NLL with the group
**total** weight (``Q = -Σ_g W_g·log(S_g/W_g)``), the combined basis table
the context owns (raw ``gamma_table`` rows + the pre-rotated "1" / "m²" rows
+ the ``den_table`` rows) and the ``pwa_gen`` numerator
``N_{n_index[w]}(m_e)`` folded into the amp-cache fill (load time).

The host contract is **unchanged**: ``compute`` still takes
``params["coeff"]`` in the padded ``(n_fact, n_basis)`` form produced by
:class:`~tabpwa.kernel_params.PWAGenKernelParams`, compresses it to the flat
block with this wrapper's own ``off`` / ``g_idx``, and returns
``grads["coeff"]`` in the same padded shape the other ``pwa_gen`` kernels
return (padding columns zero).  A default-basis config (no ``den_row``)
keeps the ``m0``/``g0`` path of ``cuda_v5_pwa_gen_s0e``.

``n_index ≡ 0`` (or no ``n_table``) and an equal-width basis make this
backend **bit-identical** to ``cuda_v5_pwa_gen_s0e`` at the same
``resolution_size`` — the flat layout then only drops the zero-padded
slots, whose contribution is exactly ``0.0``.
"""

import warnings

import numpy as np
from cffi import FFI

_ffi = FFI()
_ffi.cdef("""
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
    const int* off,int n_off,
    const int* g_idx,int n_gidx,
    const int* wave_of_slot,int n_wos,
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
void cuda_compute_v5_gen_flat(void*,void*,
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
    lib_path = ensure("kernels_v5_pwa_gen_flat.cu",
                      "libcuda_kernels_v5_pwa_gen_flat.so")
    if not lib_path:
        raise RuntimeError(
            f"v5 PWA gen flat CUDA library not found at {lib_path}.")
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
    if n_on:
        n_idx = np.asarray(config["n_index"], dtype=np.int64)
        n_col = np.asarray(config["n_mass_index"], dtype=np.int64)
    keys = {}
    slot_of_wave = np.zeros(n_wave, dtype=np.int32)
    rep_of_slot = []
    for w in range(n_wave):
        key = (cols[w].tobytes(), fl_rows[w].tobytes(),
               int(n_idx[w]) if n_on else 0,
               int(n_col[w]) if n_on else 0)
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


class CUDAKernelV5PWAGenFlat:
    """Flat-CSR ``pwa_gen`` fused tile + group-total-weight log-sum NLL.

    ``params`` needs ``ck`` (length N = n_wave/n_proj) plus either
    ``m0``/``g0`` (default basis) or ``coeff`` ``(n_fact, n_basis)`` padded
    (explicit ``den_row`` basis).  Gradients mirror the same keys — the
    ``coeff`` gradient is returned **padded**, exactly like
    ``cuda_v5_pwa_gen_s0e``.
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
            print(f"v5 PWA gen flat GPU: {_ffi.string(name_buf).decode()}")
        else:
            raise RuntimeError("No CUDA devices found")

        c = config
        self.config = c
        self.n_wave = c["matrix_angle"].shape[1]     # P·N entries per event
        self.n_proj = int(c.get("n_proj", 1) or 1)
        if self.n_wave % self.n_proj != 0:
            raise ValueError(
                f"v5 PWA gen flat: n_wave={self.n_wave} not divisible by "
                f"n_proj={self.n_proj}")
        self.n_wave_base = self.n_wave // self.n_proj  # N = ck length
        # ── factors per wave: n_res >= 1, any number ────────────────
        self.bw_order = np.ascontiguousarray(
            np.asarray(c["bw_order"]).reshape(-1), np.int64)
        if self.bw_order.size % self.n_wave != 0:
            raise ValueError(
                f"cuda_v5_pwa_gen_flat: bw_order size {self.bw_order.size} "
                f"is not a multiple of n_wave={self.n_wave}")
        self.n_res = self.bw_order.size // self.n_wave        # BW factors
        self.n_fact = self.n_wave * self.n_res                # (wave, factor)
        # the BW slot of each (wave, factor) entry
        bw = self.bw_order
        mass_index_w = np.asarray(c["mass_index"])[bw]        # [n_fact]

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
                f"cuda_v5_pwa_gen_flat: gamma_table has {gt.shape[0]} rows, "
                f"expected max(g0_index)+1 = {one_row}")

        # ── numerator (fixed data; applied in the amp-cache fill) ───
        from tabpwa.pwa_gen_support import has_numerator
        self._n_on = has_numerator(c)
        if self._n_on:
            self._n_table = np.asarray(c["n_table"], np.complex128)
            self._n_index = np.ascontiguousarray(
                np.asarray(c["n_index"], np.int32))
            self._n_mass_index = np.ascontiguousarray(
                np.asarray(c["n_mass_index"], np.int32))
            if self._n_index.size != self.n_wave \
                    or self._n_mass_index.size != self.n_wave:
                raise ValueError(
                    "cuda_v5_pwa_gen_flat: n_index/n_mass_index must have "
                    "n_wave entries")
            self._n_min = float(c["n_min"])
            self._n_delta = float(c["n_delta"])
            if self._n_delta == 0.0:
                raise ValueError("cuda_v5_pwa_gen_flat: n_delta must be != 0")
        else:
            self._n_table = None
            self._n_index = self._n_mass_index = None
            self._n_min = 0.0
            self._n_delta = 1.0

        # ── denominator basis: explicit (den_row) or the s0 preset ──
        self._generic = "den_row" in c
        if self._generic:
            self.n_gamma_rows = 0
            self._init_generic(c, bw, mass_index_w, one_row)
        else:
            self._init_plain(c, bw, mass_index_w, one_row)

        # ── angular amplitude cache layout ──────────────────────────
        self.n_uniq, self.slot_of_wave, self.rep_of_slot = \
            _amp_cache_layout(c, self._n_on)

        # v5: the group size of the log-sum data NLL
        self.resolution_size = int(resolution_size)
        if self.resolution_size < 1:
            raise ValueError(
                f"cuda_v5_pwa_gen_flat: resolution_size must be >= 1 "
                f"(got {self.resolution_size})")
        if self.resolution_size > 1 and batch_size < self.resolution_size:
            raise ValueError(
                f"cuda_v5_pwa_gen_flat: batch_size ({batch_size}) must be >= "
                f"resolution_size ({self.resolution_size}) so the chunk "
                f"stride can stay resolution-aligned")
        self._last_dnorm = None      # d(NLL)/d(norm) from the last norm call
        self.dnorm_on_gpu = True

        self._create_context(c, gt, bw, batch_size)

    # ── flat CSR layout helpers ─────────────────────────────────────
    def _finish_flat(self, widths):
        """``off`` from the per-(wave, factor) widths; flat↔padded mapping."""
        widths = np.asarray(widths, np.int64)
        if widths.size != self.n_fact:
            raise ValueError(
                f"cuda_v5_pwa_gen_flat: {widths.size} (wave, factor) widths "
                f"for n_fact={self.n_fact} (n_wave {self.n_wave}, "
                f"n_res {self.n_res})")
        if np.any(widths < 1):
            raise ValueError(
                "cuda_v5_pwa_gen_flat: every (wave, factor) needs at least "
                f"one basis slot (got {widths.tolist()})")
        self.n_basis_w = widths
        self.off = np.concatenate(
            [[0], np.cumsum(widths)]).astype(np.int32)
        self.n_basis_tot = int(self.off[-1])
        self.wave_of_slot = np.repeat(
            np.arange(self.n_fact, dtype=np.int32) // self.n_res, widths)
        # flat slot -> padded (factor, col) flat index (grad re-expansion)
        cols = np.concatenate([np.arange(int(w)) for w in widths])
        rows = np.concatenate([
            np.full(int(w), i, np.int64) for i, w in enumerate(widths)])
        self._flat_src = (rows * self.n_basis + cols).astype(np.int64)

    def _init_plain(self, c, bw, mass_index_w, one_row):
        """s0 preset, per (wave, factor): s0, −1, then the factor's γ rows.

        ``bw`` is the flat ``[n_fact]`` BW slot of each (wave, factor) entry,
        so the gamma rows are merged per factor exactly as in the CPU
        sibling ``cpu_pwa_gen`` (whose non-generic preset this mirrors).
        """
        self.n_basis = 0
        m0_index_w = np.asarray(c["m0_index"])[bw]           # [n_fact]
        mg_w = np.asarray(c["matrix_gamma"])[:, bw]     # (n_gamma, n_fact)
        rows_idx, rows_w = np.nonzero(mg_w.T)           # factor-major
        self._gamma_col_idx = np.ascontiguousarray(
            rows_idx.astype(np.int32))                  # gamma row of each
        self._fact_of_row = rows_idx.astype(np.int64)
        rows_w = rows_w.astype(np.int64)
        g0_index_w = np.asarray(c["g0_index"])[rows_w]
        g0_mass_index_w = np.asarray(c["g0_mass_index"])[rows_w]
        self._g0_index_w = g0_index_w
        self._g0_mass_index_w = g0_mass_index_w
        self._m0_index_w = m0_index_w
        self._mass_index_w = mass_index_w               # [n_fact]
        self.n_gamma_rows = int(rows_w.size)
        self.n_unique_bw = self.n_fact          # factor index == BW slot
        self.row_start = np.searchsorted(
            self._fact_of_row, np.arange(self.n_fact),
            side="left").astype(np.int32)
        rows_per_fact = np.diff(
            np.append(self.row_start, self.n_gamma_rows)).astype(np.int64)
        # padded n_basis of the s0e gen sibling (comparison / anchors only)
        self.n_basis = int(2 + (int(rows_per_fact.max())
                                if self.n_fact else 0))

        uniq = {}
        uslot_row, uslot_col = [], []

        def _uslot(row, col):
            key = (int(row), int(col))
            if key not in uniq:
                uniq[key] = len(uslot_row)
                uslot_row.append(key[0])
                uslot_col.append(key[1])
            return uniq[key]

        self._finish_flat(2 + rows_per_fact)
        g_idx = np.empty(self.n_basis_tot, np.int32)
        slot_l = np.full(self.n_basis_tot, -1, np.int32)
        for f in range(self.n_fact):
            b0 = int(self.off[f])
            r0 = int(self.row_start[f])
            r1 = (int(self.row_start[f + 1]) if f + 1 < self.n_fact
                  else self.n_gamma_rows)
            g_idx[b0] = _uslot(one_row, 0)                 # "1"  (constant)
            g_idx[b0 + 1] = _uslot(one_row + 1,
                                   int(mass_index_w[f]))    # "m²"
            for j in range(r0, r1):
                g_idx[b0 + 2 + j - r0] = _uslot(
                    int(g0_index_w[j]), int(g0_mass_index_w[j]))
                slot_l[b0 + 2 + j - r0] = j
        self.n_uslot = len(uslot_row)
        self.uslot_row = np.asarray(uslot_row, np.int32)
        self.uslot_col = np.asarray(uslot_col, np.int32)
        self.g_idx = g_idx
        self.slot_l = slot_l
        self._n_den_rows = 0
        self._den_table = None

    def _init_generic(self, c, bw, mass_index_w, one_row):
        """Explicit ``den_row`` basis, packed per (wave, factor), no padding.

        ``den_row`` pads a narrower resonance with the ``ZERO_KIND`` (-1)
        sentinel; those columns are trailing and carry coefficient 0, so the
        flat layout simply drops them.  Anything but a trailing padding
        block would break the "contiguous range" contract and is rejected.
        Each factor of a wave has its own ``den_row`` row (``bw`` maps the
        factor to its BW slot), so the factors of one wave may have
        different widths.
        """
        enc = np.asarray(c["den_row"])
        if enc.ndim != 2:
            raise ValueError("cuda_v5_pwa_gen_flat: den_row must be 2-D")
        enc_w = enc[bw]                          # [n_fact, n_basis]
        self.n_basis = int(enc.shape[1])
        widths = np.zeros(self.n_fact, np.int64)
        for f in range(self.n_fact):
            live = np.nonzero(enc_w[f] >= 0)[0]
            wd = int(live.max()) + 1 if live.size else 0
            if wd and not np.all(enc_w[f][:wd] >= 0):
                raise ValueError(
                    "cuda_v5_pwa_gen_flat: den_row padding must be a "
                    f"trailing block (factor {f}: {enc_w[f].tolist()})")
            widths[f] = wd
        self._mass_index_w = mass_index_w
        self.n_unique_bw = self.n_fact
        # the generic path never reads m0/g0/gamma-row indices on the device
        self._gamma_col_idx = np.zeros(0, np.int32)
        self._g0_index_w = np.zeros(0, np.int32)
        self._g0_mass_index_w = np.zeros(0, np.int32)
        self._fact_of_row = np.zeros(0, np.int64)
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

        self._finish_flat(widths)
        g_idx = np.empty(self.n_basis_tot, np.int32)
        for f in range(self.n_fact):
            b0 = int(self.off[f])
            col = int(mass_index_w[f])
            for j in range(int(widths[f])):
                e = int(enc_w[f, j])
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

    # ── packed ↔ flat coefficient mapping ───────────────────────────
    def _pack(self, coeff_padded):
        """Padded ``(n_fact, n_basis)`` → flat ``[n_basis_tot]``."""
        flat = np.ascontiguousarray(coeff_padded, np.float64).reshape(-1)
        if flat.size != self.n_fact * self.n_basis:
            raise ValueError(
                f"cuda_v5_pwa_gen_flat: coeff length {flat.size} != "
                f"n_fact·n_basis = {self.n_fact}·{self.n_basis}")
        return np.ascontiguousarray(flat[self._flat_src])

    def _unpack(self, gflat):
        """Flat ``[n_basis_tot]`` gradient → padded ``(n_fact, n_basis)``."""
        gp = np.zeros(self.n_fact * self.n_basis, np.float64)
        gp[self._flat_src] = gflat
        return gp.reshape(self.n_fact, self.n_basis)

    # ── context ─────────────────────────────────────────────────────
    def _create_context(self, c, gt, bw, batch_size):
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
                    "cuda_v5_pwa_gen_flat: den_table and gamma_table grids "
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

        self._ctx = self._lib.cuda_create_context_v5_gen_flat(
            _ffi.NULL, 0,                      # m0_index: host-side only
            _ib_or_null(self._g0_index_w), len(self._g0_index_w),
            _ib_or_null(self._g0_mass_index_w), len(self._g0_mass_index_w),
            _ib(self._mass_index_w), len(self._mass_index_w),
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
            _ib(self.off), len(self.off),
            _ib(self.g_idx), len(self.g_idx),
            _ib(self.wave_of_slot), len(self.wave_of_slot),
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
            (_ib(self._n_index) if self._n_on else _ffi.NULL),
            (self.n_wave if self._n_on else 0),
            (_ib(self._n_mass_index) if self._n_on else _ffi.NULL),
            (self.n_wave if self._n_on else 0))
        if self._ctx == _ffi.NULL:
            raise RuntimeError(
                "cuda_v5_pwa_gen_flat: context creation failed "
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
                f"cuda_v5_pwa_gen_flat: {ne} events are not a multiple of "
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
        """s0 preset (flat), per (wave, factor): m0², −1, then m0·g0_j.

        The factors of a wave are the s0c preset of each BW in the product;
        at ``n_res == 1`` the single factor is the wave's BW, so this is the
        previous per-wave assembly unchanged.
        """
        nm_, gg_ = self.n_m0_params, self.n_g0_params
        m0 = np.zeros(nm_, np.float64)
        m0[:len(params["m0"])] = np.asarray(params["m0"])
        g0 = np.zeros(gg_, np.float64)
        g0[:len(params["g0"])] = np.asarray(params["g0"])
        m0_f = m0[self._m0_index_w]                       # [n_fact]
        s0_f = m0_f * m0_f
        mult_f = m0_f[self._fact_of_row] * g0[self._g0_index_w]
        coeff = np.zeros(self.n_basis_tot, np.float64)
        for f in range(self.n_fact):
            b0 = int(self.off[f])
            r0 = int(self.row_start[f])
            r1 = (int(self.row_start[f + 1]) if f + 1 < self.n_fact
                  else self.n_gamma_rows)
            coeff[b0] = s0_f[f]
            coeff[b0 + 1] = -1.0
            coeff[b0 + 2:b0 + 2 + (r1 - r0)] = mult_f[r0:r1]
        return coeff, m0_f, g0

    def compute(self, params, data_handle, norm=None, return_p=True):
        """Forward + gradients (s0 preset or explicit ``coeff`` basis)."""
        ck = params["ck"]
        nbase = self.n_wave_base           # N — ck / gradient buffer size
        use_norm = 0 if norm is None else 1
        norm_val = norm if norm is not None else 0.0

        if len(ck) != nbase:
            raise ValueError(
                f"v5 PWA gen flat: ck length {len(ck)} != n_wave/n_proj = "
                f"{nbase} ({self.n_wave}/{self.n_proj})")

        ck_r = np.real(ck).astype(np.float64)
        ck_i = np.imag(ck).astype(np.float64)

        if self._generic:
            # padded (n_fact, n_basis) coefficients -> flat CSR block
            coeff = self._pack(params["coeff"])
        else:
            coeff, m0_f, g0 = self._coeff_from_m0g0(params)

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

        self._lib.cuda_compute_v5_gen_flat(
            self._ctx, data_handle.ptr,
            _db(ck_r), _db(ck_i), _db(coeff),
            norm_val, use_norm,
            oQ, oDn, (_db(oP) if oP is not None else _ffi.NULL),
            _db(ogck_r), _db(ogck_i), _db(ogcoeff))
        # v5 semantics: the device accumulates dNLL/dnorm (0 without a norm)
        self._last_dnorm = float(oDn[0])

        if self._generic:
            # raw per-slot gradients, re-expanded to the padded (n_fact,
            # n_basis) shape the other pwa_gen kernels return; the parameter
            # chain lives in the model's basis transform.
            grads = {"ck": ogck_r + 1j * ogck_i,
                     "coeff": self._unpack(ogcoeff)}
            return oQ[0], grads, (oP if return_p else None)

        # ── unpack the flat coefficient gradient to (s0, mult) per factor ──
        ogs0 = np.zeros(self.n_fact, np.float64)
        ogmult = np.zeros(self.n_gamma_rows, np.float64)
        for f in range(self.n_fact):
            b0 = int(self.off[f])
            r0 = int(self.row_start[f])
            r1 = (int(self.row_start[f + 1]) if f + 1 < self.n_fact
                  else self.n_gamma_rows)
            ogs0[f] = ogcoeff[b0]
            ogmult[r0:r1] = ogcoeff[b0 + 2:b0 + 2 + (r1 - r0)]

        # ── host chain: (s0, mult) -> (m0, g0), per factor ────────────
        g0_row = g0[self._g0_index_w]
        dm0_f = 2.0 * m0_f * ogs0 + np.bincount(
            self._fact_of_row, weights=g0_row * ogmult,
            minlength=self.n_fact)
        grad_m0 = np.bincount(self._m0_index_w, weights=dm0_f,
                              minlength=self.n_m0_params)
        grad_g0 = np.bincount(self._g0_index_w,
                              weights=m0_f[self._fact_of_row] * ogmult,
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
                "cuda_v5_pwa_gen_flat: the Gram path needs an explicit "
                "coeff basis — use a backend that supports compute_gram "
                "with pwa_gen (or the plain s0e for an N ≡ 1 config)")
        n = self.n_wave_base
        m0_arr = np.zeros(self.n_m0_params, np.float64)
        m0_arr[:len(m0)] = np.asarray(m0)
        g0_arr = np.zeros(self.n_g0_params, np.float64)
        g0_arr[:len(g0)] = np.asarray(g0)
        coeff, _m0_f, _g0 = self._coeff_from_m0g0({"m0": m0_arr,
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
