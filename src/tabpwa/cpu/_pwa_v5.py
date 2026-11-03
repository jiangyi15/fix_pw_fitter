"""
CPU pure-PWA kernel on an opaque basis with a general BW product per wave
and the v5 **grouped log-sum** data NLL (C + OpenMP, cffi).

CPU counterpart of ``cuda_v5_pwa`` — the *general* projection-sum PWA, i.e.
any number of Breit-Wigner factors per chain (``n_res >= 1``):

    P(e) = Σ_p |A_p(e)|² ,   A_p(e) = Σ_k ck_k·common[e, p·N+k]
    common[e,w] = amp[e,w] / BW_w(e),   BW_w(e) = Π_r bw_{w,r}(e)

with ONE difference from :mod:`tabpwa.cpu._pwa_gen` (the v4 sibling): in
the data-NLL branch events are grouped in contiguous blocks of
``resolution_size`` and one log is taken per group,

    q_e  = w_e·(P_e/norm + bkg_e) ,   S_g = Σ_{e∈g} q_e
    Q    = −Σ_g log(S_g)
    dQ/dP_e  = −(w_e/norm)/S_g
    dQ/dnorm = Σ_g (1/S_g)·(Σ_{e∈g} w_e·P_e)/norm²

The phase-space path (``norm=None``) is unchanged (per-event, linear
Σ w·P) and the returned per-event ``P`` stays ``P_sig = Σ_p |A_p|²``.

``resolution_size <= 1`` takes the EXACT v4 per-event path
(``Q = −Σ w·log(P/norm+b)``), so it is bit-identical to
``cpu_pwa_gen`` for ANY weights — @mod:`tabpwa.cpu._pwa_gen` is a thin
alias of this class at ``resolution_size=1``.  A one-event *group* would
instead give ``−log(w·(P/norm+b))`` = v4 − ``log(w)``, i.e. off by the
constant ``−Σ_e log(w_e)`` for non-unit weights, which is why that size
never uses the group formula.  (Consequence: at ``resolution_size == 1``
with non-unit weights this kernel intentionally differs from
``cuda_v5_pwa``, which applies the group formula even for one event; with
unit weights the two coincide.)

The C kernel works on an **opaque basis**: it knows nothing about s0 or
about what a coefficient means.  Each (wave, factor) contributes
``n_basis`` slots

    bw_{w,r}(e) = Σ_i coeff[(w·n_res+r)·n_basis + i]·F[slot_map[… i]](e)
    ∂Q/∂coeff[(w,r,i)] = 2·Re( dBW_w·(1/bw_{w,r})·F_i )
    dBW_w = −(dQ/dcommon_w)·common_w

so a wave can carry one basis function (a plain coefficient) or a whole
denominator expansion — the kernel code is identical.  The table stores
the finished function F (the −i·γ rotation, the −m² sign and the constant
1 are all baked in at load), so the kernel never rotates anything.

This module builds the **s0c preset of every factor** of that basis

    slot 0   : F =  1          coeff = m0_{w,r}²       (the pole term s0)
    slot 1   : F = −m²         coeff = +1              (a pure indicator)
    slot 2+j : F = −i·γ_j(m)   coeff = m0_{w,r}·g0_j   (running-width terms)

and chains dQ/dcoeff back to the fitted m0/g0 (exactly like the CUDA
siblings), so no m0/g0 algebra enters the kernel.  At ``n_res == 1`` this
module is **bit-identical** to :mod:`tabpwa.cpu._pwa_s0c` (the single
factor folds exactly through the product) and, at ``resolution_size == 1``,
it IS :mod:`tabpwa.cpu._pwa_gen` (that module is a thin alias of this
class; one C kernel / one shared library serves both names).

Iteration-invariant per-event inputs are precomputed once at load time in
NumPy (the CPU analogue of the CUDA τ-u-cache):

    amp[e,w] = (ka·matrix_angle)·Π fl        (angular basis × form factors)
    F[u,e]   = basis function at unique slot u (1, −m², −i·γ interp)

Unique slots are deduped across waves/factors/events, so a production
layout keeps only a handful of entries per event instead of
n_res·(2·n_wave + n_gamma_rows).  Both tables are stored
**event-contiguous** (``[wave][event]`` / ``[slot][event]``) for the tile
kernel's SoA evaluation.  ``compute()`` therefore only recomputes the BW
denominators and the sum over projections — the same data layout / math
as ``numpy_pwa`` (verified to FP64 round-off in
``tests/test_cpu_pwa_v5.py``).

Three C calls::

    cpu_pwa_v5_load     — build a data handle (borrows the arrays)
    cpu_pwa_v5_compute  — forward NLL + gradients (OpenMP over events)
    cpu_pwa_v5_free     — release the handle

The chunk strategy of the C kernel (no second BW/``common`` pass):

* ``resolution_size <= SOA_TILE``: chunk = ``rs·(SOA_TILE // rs)`` events
  (whole groups), so the group sum is available inside the chunk and the
  two gradient phases reuse the same ``common`` tile.
* ``resolution_size > SOA_TILE``: a group spans chunks; each event's
  gradient is accumulated with the per-event factor ``w_e/norm`` into
  per-group temporaries and scaled once by ``−1/S_g`` at the group end —
  again on the same per-chunk ``common`` tile.

Performance note: on SMT CPUs set ``OMP_NUM_THREADS`` to the number of
**physical** cores (16 on a 16C/32T Ryzen 9); using all logical threads was
measured slower (memory/FP64 contention).
"""

import numpy as np
from cffi import FFI

_ffi = FFI()
_ffi.cdef("""
void* cpu_pwa_v5_load(
    const double* amp_r,const double* amp_i,
    const double* Z_r,const double* Z_i,const int* slot_map,
    int n_uslot,
    const double* w,const double* bkg,int ne,
    int n_wave,int N,int n_proj,int n_res,int n_basis);
void cpu_pwa_v5_free(void*);
double cpu_pwa_v5_compute(void* h,
    const double* coeff,
    const double* ck_r,const double* ck_i,
    double norm,int use_norm,int resolution_size,
    double* grad_coeff,
    double* grad_ck_r,double* grad_ck_i,
    double* P_out,double* dnorm);
""")


def _load_lib(lib_path=None):
    """Load the CPU PWA gen shared library, auto-building if stale."""
    if lib_path is not None:
        return _ffi.dlopen(lib_path)
    from tabpwa.cpu.build import ensure
    p = ensure("kernels_cpu_pwa_v5.c", "libcpu_kernels_cpu_pwa_v5.so")
    if not p:
        raise RuntimeError(
            "CPU library could not be built (kernels_cpu_pwa_v5.c). "
            "Build with: gcc -shared -fPIC -O3 -march=native -fopenmp -lm "
            "-o <path> kernels_cpu_pwa_v5.c")
    return _ffi.dlopen(p)


# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------

class DataHandle:
    """Python-side wrapper around a CPU PWA v5 data pointer."""

    def __init__(self, ptr, lib, ne):
        self.ptr = ptr
        self._lib = lib
        self.ne = ne
        self._keep = []

    def free(self):
        if self.ptr is not None:
            self._lib.cpu_pwa_v5_free(self.ptr)
            self.ptr = None
            self._keep.clear()

    def __del__(self):
        self.free()


class CPUKernelPWAV5:
    """General (any n_res) projection-sum PWA with the v5 group NLL.

    Usage::

        kernel = CPUKernelPWAV5(kernel_config, resolution_size=8)
        data = kernel.load_data(data_dict)
        Q, grads, P = kernel.compute(params, data, norm=...)

    ``params`` needs ``ck`` (length N = n_wave/n_proj), ``m0``, ``g0``;
    gradients are returned for the same three keys.
    """

    def __init__(self, config, batch_size=50000, resolution_size=1,
                 lib_path=None):
        self._lib = _load_lib(lib_path)
        self.resolution_size = int(resolution_size)
        if self.resolution_size < 1:
            raise ValueError(
                f"cpu_pwa_v5: resolution_size must be >= 1, got "
                f"{self.resolution_size}")
        print(f"✓ CPU PWA v5 kernel (C + OpenMP, float64, "
              f"resolution_size={self.resolution_size})")

        c = config
        self.config = c
        self.n_wave = c["matrix_angle"].shape[1]       # P·N entries / event
        self.n_proj = int(c.get("n_proj", 1) or 1)
        if self.n_wave % self.n_proj != 0:
            raise ValueError(
                f"cpu_pwa_v5: n_wave={self.n_wave} not divisible by "
                f"n_proj={self.n_proj}")
        self.n_wave_base = self.n_wave // self.n_proj  # N — ck length
        bow = np.asarray(c["bw_order"]).reshape(-1)
        if bow.size % self.n_wave != 0:
            raise ValueError(
                f"cpu_pwa_v5: bw_order size {bow.size} is not a multiple "
                f"of n_wave={self.n_wave}")
        self.n_res = bow.size // self.n_wave
        self.n_fact = self.n_wave * self.n_res          # (wave, factor)

        # ── per-(wave, factor) BW expansion (no bw_order indirection) ──
        # entry f = w·n_res + r  ->  the BW slot bw_order[f]
        m0_index_f = np.asarray(c["m0_index"])[bow]
        mass_index_f = np.asarray(c["mass_index"])[bow]
        # merge matrix_gamma (gamma -> bw) with bw_order (bw -> wave/factor):
        # the used gamma rows get duplicated per (wave, factor), f-major.
        mg_f = np.asarray(c["matrix_gamma"])[:, bow]     # (n_gamma, n_fact)
        nz = np.nonzero(mg_f)
        if np.any(mg_f[nz] != 1.0):
            raise ValueError(
                "cpu_pwa_v5 assumes matrix_gamma entries are exactly 1.0 "
                f"(got {np.unique(mg_f[nz])})")
        fact_of_row, rows_f = np.nonzero(mg_f.T)         # factor-major
        fact_of_row = fact_of_row.astype(np.int64)
        rows_f = rows_f.astype(np.int64)
        g0_index_f = np.asarray(c["g0_index"])[rows_f]
        g0_mass_index_f = np.asarray(c["g0_mass_index"])[rows_f]
        self.n_gamma_rows = int(rows_f.size)
        self.row_start = np.searchsorted(
            fact_of_row, np.arange(self.n_fact),
            side="left").astype(np.int32)
        rows_per_fact = np.diff(np.append(self.row_start,
                                          self.n_gamma_rows))
        self.n_basis = int(2 + (int(rows_per_fact.max())
                                if self.n_fact else 0))

        # ── basis layout (a HOST-side convention; the C kernel is
        #    basis-agnostic — it only sees coeff[] and the slot table) ──
        # For the s0c preset each (wave, factor) carries
        #   slot 0    : F = 1,            (coeff = m0²)
        #   slot 1    : F = −m²,          (coeff = +1)
        #   slot 2+j  : F = −i·γ(row, m)  (coeff = m0·g0_j)
        # so that  bw_{w,r} = Σ_i coeff_i·F_i = m0² − m² − i·Σ mult_j γ_j.
        # EVERY slot — including the constant and the m² term — is one row
        # of the (2-row-extended) gamma table interpolated at the event
        # mass, exactly like the CUDA siblings:
        #   "1"  row: Z = i        -> F = −i·Z = 1
        #   "m²" row: Z = −i·m²    -> F = −i·Z = −m²   (m² sampled on the
        #                                               table grid, CR-exact)
        # so both paths evaluate m² through the SAME interpolation (and
        # clamp identically outside the grid), never from a raw mass.
        gt = c["gamma_table"]
        one_row = int(np.max(c["g0_index"])) + 1
        ext = np.zeros((gt.shape[0] + 2, gt.shape[-1]), np.complex128)
        ext[:gt.shape[0]] = gt
        ext[one_row] = 1j                                  # "1"  -> Z = i
        xgrid = (float(c["gamma_min"]) +
                 float(c["gamma_delta"]) * np.arange(gt.shape[-1]))
        ext[one_row + 1] = -1j * xgrid ** 2                # "m²" -> Z = −i·m²
        self._gt_ext = ext

        uniq = {}
        self._uslot_kind = []      # 0 = const, 1 = m², 2 = γ interp
        self._uslot_row = []       # row index into the EXTENDED table
        self._uslot_col = []       # mass column
        slot_map = np.empty((self.n_fact, self.n_basis), np.int32)

        def _uslot(kind, row, col):
            key = (kind, int(row), int(col))
            u = uniq.get(key)
            if u is None:
                u = len(self._uslot_kind)
                uniq[key] = u
                self._uslot_kind.append(kind)
                self._uslot_row.append(int(row))
                self._uslot_col.append(int(col))
            return u

        for f in range(self.n_fact):
            r0 = int(self.row_start[f])
            r1 = (int(self.row_start[f + 1]) if f + 1 < self.n_fact
                  else self.n_gamma_rows)
            col = int(mass_index_f[f])
            slot_map[f, 0] = _uslot(0, one_row, 0)
            slot_map[f, 1] = _uslot(1, one_row + 1, col)
            for j in range(r0, r1):
                slot_map[f, 2 + j - r0] = _uslot(
                    2, g0_index_f[j], g0_mass_index_f[j])
            for j in range(2 + (r1 - r0), self.n_basis):
                slot_map[f, j] = _uslot(0, one_row, 0)   # zero padded
        self.n_uslot = len(self._uslot_kind)
        self.slot_map = slot_map.reshape(-1).astype(np.int32)

        self._m0_index_f = m0_index_f.astype(np.int32)
        self._mass_index_f = mass_index_f.astype(np.int32)
        self._fact_of_row = fact_of_row
        self._g0_index_f = g0_index_f.astype(np.int64)
        self._g0_mass_index_f = g0_mass_index_f.astype(np.int64)
        self._g0_index_f32 = g0_index_f.astype(np.int32)

        # ── dimension bookkeeping (same as cuda_v4_pwa / cpu_pwa_s0c) ──
        self.n_decay = c["fl_order"].size // self.n_wave
        self.n_mass = int(np.max(c["mass_index"])) + 1
        self.n_momentum = int(np.max(c["fl_q_index"])) + 1
        self.n_angle_k = c["angle_k"].shape[0]
        self.n_angle_total = (int(np.max(c["angle_index"])) + 1
                              if len(c["angle_index"]) > 0 else 0)
        self.n_angle_comp = c["angle_k"].shape[-1]
        self.n_m0_params = int(np.max(c["m0_index"])) + 1
        self.n_g0_params = int(np.max(c["g0_index"])) + 1

        # interpolation tables / index tables used by the one-time cache
        from tabpwa.numpy_kernel import NumpyKernel
        self._nk = NumpyKernel(c)

        self._last_dnorm = None
        self.dnorm_on_gpu = True

    # -- data lifecycle ------------------------------------------------

    def load_data(self, data):
        """Precompute the iteration-invariant per-event cache and upload it.

        Args:
            data: dict with keys ``mass``, ``q``, ``angle``, ``weight``,
                  ``bkg`` (or ``bkg_raw``).

        Returns:
            DataHandle (pass to :meth:`compute`).
        """
        k = self._nk
        ne = data["mass"].shape[0]
        if ne % self.resolution_size != 0:
            import warnings
            warnings.warn(
                f"cpu_pwa_v5: {ne} events are not a multiple of "
                f"resolution_size={self.resolution_size}; the final log-sum "
                f"group is partial.  Resolution-cloud data should contain "
                f"n_original x resolution_size rows.", RuntimeWarning,
                stacklevel=2)
        nw = self.n_wave
        n_mass = self.n_mass
        n_mom = self.n_momentum

        mass = np.array(
            np.asarray(data["mass"], np.float64).reshape(ne, -1)[:, :n_mass],
            dtype=np.float64, order="C", copy=True)
        mom = np.ascontiguousarray(
            np.asarray(data["q"], np.float64).reshape(ne, -1)[:, :n_mom])
        # angle is (ne, n_angle_total, n_angle_comp) from build_event_data;
        # a flattened (ne, n_at*n_ac) input is accepted and reshaped.  The
        # reference numpy_pwa takes axis=-2 on the 3-D array, so keeping the
        # angle-position axis is essential.
        ang_raw = np.asarray(data["angle"], np.float64)
        if ang_raw.ndim == 2:
            ang = ang_raw[:, :self.n_angle_total * self.n_angle_comp] \
                .reshape(ne, self.n_angle_total, self.n_angle_comp)
        elif ang_raw.ndim == 3:
            ang = ang_raw[:, :self.n_angle_total, :self.n_angle_comp]
        else:
            raise ValueError(
                f"cpu_pwa_v5: angle must be 2-D or 3-D, got "
                f"{ang_raw.shape}")
        ang = np.ascontiguousarray(ang)

        # ── iteration-invariant amplitude amp = fa·fl_p ──────────────
        fl_q = np.take(mom, k.fl_q_index, axis=-1)
        fl = k.interp_catmull_rom(k.fl_table, k.fl_type, fl_q,
                                  k.fl_min, k.fl_delta)
        fl_all = np.take(fl, k.fl_order, axis=-1)
        fl_p = np.prod(fl_all.reshape(ne, nw, self.n_decay), axis=-1)

        ang_t = np.take(ang, k.angle_index, axis=-2)
        ka = np.prod(np.cos(ang_t * k.angle_k + k.angle_b), axis=-1)
        fa = ka @ k.matrix_angle
        amp = fa * fl_p                                   # (ne, n_wave)

        # ── iteration-invariant basis functions at the unique slots ──
        # (all of them derived from the SAME extended table + interp)
        # (only ne × n_uslot values — the s0c preset collapses
        #  n_res·(2·n_wave + n_gamma_rows) naive entries to a handful).
        # Stored READY-TO-USE: F = 1, −m², −i·γ (the −i rotation is done
        # here once, so the kernel is a plain complex linear combination).
        nu = self.n_uslot
        F_r = np.zeros((ne, nu), np.float64)
        F_i = np.zeros((ne, nu), np.float64)
        for u in range(nu):
            kind = self._uslot_kind[u]
            col = self._uslot_col[u]
            if kind == 0:                       # F = 1
                F_r[:, u] = 1.0
            elif kind == 1:                     # F = −m²
                F_r[:, u] = -mass[:, col] ** 2
            else:                               # F = −i·γ  =  (γ_i, −γ_r)
                gm = np.take(mass, col, axis=-1)
                g = k.interp_catmull_rom(
                    k.gamma_table, self._uslot_row[u], gm,
                    k.gamma_min, k.gamma_delta)
                F_r[:, u] = np.imag(g)
                F_i[:, u] = -np.real(g)

        # event-contiguous (SoA) layout for the tile kernel: [n_wave][ne]
        amp_t = np.ascontiguousarray(amp.T)
        del amp
        amp_r = np.ascontiguousarray(amp_t.real, np.float64)
        amp_i = np.ascontiguousarray(amp_t.imag, np.float64)
        del amp_t
        weight = np.array(np.asarray(data.get("weight", np.ones(ne)),
                                     np.float64), copy=True).reshape(ne)
        bkg_key = "bkg_raw" if "bkg_raw" in data else "bkg"
        bkg_raw = data.get(bkg_key, np.ones(ne))
        if np.isscalar(bkg_raw):
            bkg = np.full(ne, float(bkg_raw), np.float64)
        else:
            bkg = np.array(np.asarray(bkg_raw, np.float64),
                           copy=True).reshape(ne)

        ka_list = []

        def _db(a):
            arr = np.ascontiguousarray(a, np.float64)
            buf = _ffi.from_buffer(arr)
            ka_list.append((arr, buf))
            return _ffi.cast("double*", buf)

        def _ib(a):
            arr = np.ascontiguousarray(a, np.int32)
            buf = _ffi.from_buffer(arr)
            ka_list.append((arr, buf))
            return _ffi.cast("int*", buf)

        dh = DataHandle(
            self._lib.cpu_pwa_v5_load(
                _db(amp_r), _db(amp_i),
                _db(np.ascontiguousarray(F_r.T, np.float64)),
                _db(np.ascontiguousarray(F_i.T, np.float64)),
                _ib(self.slot_map),
                self.n_uslot,
                _db(weight), _db(bkg), ne,
                nw, self.n_wave_base, self.n_proj, self.n_res,
                self.n_basis),
            self._lib, ne)
        dh._keep = ka_list
        dh._weight = weight
        dh._bkg = bkg
        return dh

    # -- compute -------------------------------------------------------

    def compute(self, params, data_handle, norm=None, return_p=True):
        """Forward NLL + gradients for the grouped-log-sum projection PWA.

        Args:
            params: dict with ``ck`` (complex, length N), ``m0``, ``g0``.
            data_handle: DataHandle from :meth:`load_data`.
            norm: optional normalisation factor (None → unnormalised,
                per-event linear Σ w·P exactly as ``cpu_pwa_gen``).
            return_p: if False, P is not computed/returned (None).

        Returns:
            ``(Q, grads, P)`` with ``grads`` keys ``ck``/``m0``/``g0``.
            ``Q = −Σ_g log Σ_{e∈g} w_e·(P_e/norm + bkg_e)`` when normalised.
        """
        ck = np.asarray(params["ck"])
        nbase = self.n_wave_base
        if len(ck) != nbase:
            raise ValueError(
                f"cpu_pwa_v5: ck length {len(ck)} != n_wave/n_proj = "
                f"{nbase} ({self.n_wave}/{self.n_proj})")

        use_norm = 0 if norm is None else 1
        norm_val = float(norm) if norm is not None else 0.0

        ck_r = np.ascontiguousarray(np.real(ck), np.float64)
        ck_i = np.ascontiguousarray(np.imag(ck), np.float64)
        m0 = np.zeros(self.n_m0_params, np.float64)
        m0[:len(params["m0"])] = np.asarray(params["m0"])
        g0 = np.zeros(self.n_g0_params, np.float64)
        g0[:len(params["g0"])] = np.asarray(params["g0"])

        # ── coefficient assembly for the s0c basis, per (wave, factor) ──
        # slot 0 = m0², slot 1 = the constant 1, slot 2+j = m0·g0_j.
        # (The kernel only sees this vector; the meanings live here.)
        nf = self.n_basis
        nfact = self.n_fact
        m0_f = m0[self._m0_index_f]
        coeff = np.zeros((nfact, nf), np.float64)
        coeff[:, 0] = m0_f * m0_f
        coeff[:, 1] = 1.0
        rs = self.row_start
        for f in range(nfact):
            r0 = int(rs[f])
            r1 = (int(rs[f + 1]) if f + 1 < nfact else self.n_gamma_rows)
            coeff[f, 2:2 + (r1 - r0)] = \
                m0_f[self._fact_of_row[r0:r1]] * g0[self._g0_index_f[r0:r1]]
        coeff = coeff.reshape(-1)

        gcoeff = np.zeros(nfact * nf, np.float64)
        ogck_r = np.zeros(nbase, np.float64)
        ogck_i = np.zeros(nbase, np.float64)
        dnorm = _ffi.new("double*")
        P = (np.zeros(data_handle.ne, np.float64) if return_p else None)

        ka = []

        def _db(a):
            arr = np.ascontiguousarray(a, np.float64)
            buf = _ffi.from_buffer(arr)
            ka.append((arr, buf))
            return _ffi.cast("double*", buf)

        Q = self._lib.cpu_pwa_v5_compute(
            data_handle.ptr,
            _db(coeff), _db(ck_r), _db(ck_i),
            norm_val, use_norm, self.resolution_size,
            _db(gcoeff), _db(ogck_r), _db(ogck_i),
            (_db(P) if P is not None else _ffi.NULL),
            dnorm)
        self._last_dnorm = float(dnorm[0]) if use_norm else None

        gc = gcoeff.reshape(nfact, nf)
        ogs0 = np.ascontiguousarray(gc[:, 0])
        ogmult = np.zeros(self.n_gamma_rows, np.float64)
        for f in range(nfact):
            r0 = int(rs[f])
            r1 = (int(rs[f + 1]) if f + 1 < nfact else self.n_gamma_rows)
            ogmult[r0:r1] = gc[f, 2:2 + (r1 - r0)]

        # ── host chain: (s0, mult) -> (m0, g0), per factor ──────────
        # s0_f = m0_f²               -> dQ/dm0_f += 2·m0_f·dQ/ds0_f
        # mult_l = m0_{f(l)}·g0_l    -> dQ/dm0_f += Σ_l g0_l·dQ/dmult_l
        #                              dQ/dg0_l = m0_{f(l)}·dQ/dmult_l
        g0_row = g0[self._g0_index_f]
        dm0_f = 2.0 * m0_f * ogs0 + np.bincount(
            self._fact_of_row, weights=g0_row * ogmult,
            minlength=nfact)
        grad_m0 = np.bincount(self._m0_index_f, weights=dm0_f,
                              minlength=self.n_m0_params)
        grad_g0 = np.bincount(self._g0_index_f,
                              weights=m0_f[self._fact_of_row] * ogmult,
                              minlength=self.n_g0_params)

        grads = {"ck": ogck_r + 1j * ogck_i,
                 "m0": grad_m0, "g0": grad_g0}
        return Q, grads, P

    # -- cleanup -------------------------------------------------------

    def free(self):
        pass

    def __del__(self):
        self.free()
