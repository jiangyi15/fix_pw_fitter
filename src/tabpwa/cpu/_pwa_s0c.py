"""
CPU pure-PWA kernel with the "s0c" parameterisation (C + OpenMP, cffi).

Mirror of ``cuda_v4_pwa_s0c`` on the CPU: projection-sum PWA

    P(e) = Σ_p |A_p(e)|² ,   A_p(e) = Σ_k ck_k·common[e, p·N+k]
    common[e,w] = amp[e,w] / BW_w(e)
    BW_w(e) = s0_w − m_w(e)² − i·Σ_{l∈rows(w)} mult_l·Z[e,l]

with the s0c "direct pole parameters" outside transform

    s0_w  = m0_w² ,   mult_l = m0_{w(l)}·g0_l

The C kernel works in (s0, mult) and returns dQ/ds0 / dQ/dmult; the
wrapper chains them back to the fitted m0/g0 (exactly like the CUDA
sibling), so no m0/g0 algebra runs on the device.

Iteration-invariant per-event inputs are precomputed once at load time
in NumPy (the CPU analogue of the CUDA τ-u-cache):

    amp[e,w] = (ka·matrix_angle)·Π fl        (angular basis × form factors)
    Z[e,l]   = Catmull-Rom γ-table interp at mass[e, g0_mass_index_w[l]]

so ``compute()`` only recomputes the BW denominators and the sum over
projections — the same data layout / math as ``numpy_pwa`` (verified to
FP64 round-off in ``tests/test_cpu_pwa_s0c.py``).

Three C calls::

    cpu_pwa_s0c_load     — build a data handle (borrows the arrays)
    cpu_pwa_s0c_compute  — forward NLL + gradients (OpenMP over events)
    cpu_pwa_s0c_free     — release the handle

Performance note: on SMT CPUs set ``OMP_NUM_THREADS`` to the number of
**physical** cores (16 on a 16C/32T Ryzen 9); using all logical threads was
measured slower (memory/FP64 contention).
"""

import numpy as np
from cffi import FFI

_ffi = FFI()
_ffi.cdef("""
void* cpu_pwa_s0c_load(
    const double* mass,int n_mass,
    const double* amp_r,const double* amp_i,
    const double* Z_r,const double* Z_i,
    const double* w,const double* bkg,int ne,
    const int* m0_index_w,
    const int* mass_index_w,
    const int* row_start,
    int n_wave,int N,int n_proj,int n_gamma_rows,int n_basis);
void cpu_pwa_s0c_free(void*);
double cpu_pwa_s0c_compute(void* h,
    const double* coeff,
    const double* ck_r,const double* ck_i,
    double norm,int use_norm,
    double* grad_coeff,
    double* grad_ck_r,double* grad_ck_i,
    double* P_out,double* dnorm);
""")


def _load_lib():
    """Load the CPU PWA s0c shared library, auto-building if stale."""
    from tabpwa.cpu.build import ensure
    lib_path = ensure("kernels_cpu_pwa_s0c.c", "libcpu_kernels_cpu_pwa_s0c.so")
    if not lib_path:
        raise RuntimeError(
            "CPU library could not be built (kernels_cpu_pwa_s0c.c). "
            "Build with: gcc -shared -fPIC -O3 -march=native -fopenmp -lm "
            "-o <path> kernels_cpu_pwa_s0c.c")
    return _ffi.dlopen(lib_path)


# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------

class DataHandle:
    """Python-side wrapper around a CPU PWA s0c data pointer."""

    def __init__(self, ptr, lib, ne):
        self.ptr = ptr
        self._lib = lib
        self.ne = ne
        self._keep = []

    def free(self):
        if self.ptr is not None:
            self._lib.cpu_pwa_s0c_free(self.ptr)
            self.ptr = None
            self._keep.clear()

    def __del__(self):
        self.free()


class CPUKernelPWAS0C:
    """s0c-parameterised projection-sum PWA, C + OpenMP.

    Usage::

        kernel = CPUKernelPWAS0C(kernel_config)
        data = kernel.load_data(data_dict)
        Q, grads, P = kernel.compute(params, data, norm=...)

    ``params`` needs ``ck`` (length N = n_wave/n_proj), ``m0``, ``g0``;
    gradients are returned for the same three keys.
    """

    def __init__(self, config, batch_size=50000, lib_path=None):
        self._lib = _load_lib()
        print("✓ CPU PWA s0c kernel (C + OpenMP, float64)")

        c = config
        self.config = c
        self.n_wave = c["matrix_angle"].shape[1]       # P·N entries / event
        self.n_proj = int(c.get("n_proj", 1) or 1)
        if self.n_wave % self.n_proj != 0:
            raise ValueError(
                f"cpu_pwa_s0c: n_wave={self.n_wave} not divisible by "
                f"n_proj={self.n_proj}")
        self.n_wave_base = self.n_wave // self.n_proj  # N — ck length
        self.n_res = c["bw_order"].size // self.n_wave
        if self.n_res != 1:
            raise ValueError(
                "cpu_pwa_s0c needs exactly ONE BW per chain "
                f"(top -> R + X), got n_res = {self.n_res}")

        # ── per-wave BW expansion (no bw_order indirection) ──────────
        bw = np.asarray(c["bw_order"]).reshape(-1)
        if bw.size != self.n_wave:
            raise ValueError("bw_order must be flat (n_wave,) at n_res=1")
        m0_index_w = np.asarray(c["m0_index"])[bw]
        mass_index_w = np.asarray(c["mass_index"])[bw]
        # merge matrix_gamma (gamma -> bw) with bw_order (bw -> wave):
        # the used gamma rows get duplicated per wave, wave-major.
        mg_w = np.asarray(c["matrix_gamma"])[:, bw]      # (n_gamma, n_wave)
        nz = np.nonzero(mg_w)
        if np.any(mg_w[nz] != 1.0):
            raise ValueError(
                "cpu_pwa_s0c assumes matrix_gamma entries are exactly 1.0 "
                f"(got {np.unique(mg_w[nz])})")
        wave_of_row, rows_w = np.nonzero(mg_w.T)         # wave-major
        wave_of_row = wave_of_row.astype(np.int64)
        rows_w = rows_w.astype(np.int64)
        g0_index_w = np.asarray(c["g0_index"])[rows_w]
        g0_mass_index_w = np.asarray(c["g0_mass_index"])[rows_w]
        self.n_gamma_rows = int(rows_w.size)
        self.row_start = np.searchsorted(
            wave_of_row, np.arange(self.n_wave), side="left").astype(np.int32)
        rows_per_wave = np.diff(np.append(self.row_start,
                                          self.n_gamma_rows))
        self.n_basis = int(2 + int(rows_per_wave.max()))

        self._m0_index_w = m0_index_w.astype(np.int32)
        self._mass_index_w = mass_index_w.astype(np.int32)
        self._wave_of_row = wave_of_row
        self._g0_index_w = g0_index_w.astype(np.int64)
        self._g0_mass_index_w = g0_mass_index_w.astype(np.int64)
        self._g0_index_w32 = g0_index_w.astype(np.int32)

        # ── dimension bookkeeping (same as cuda_v4_pwa_s0c) ─────────
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
        nw = self.n_wave
        n_mass = self.n_mass
        n_mom = self.n_momentum
        n_gr = self.n_gamma_rows

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
                f"cpu_pwa_s0c: angle must be 2-D or 3-D, got "
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

        # ── iteration-invariant γ-interp (the τ-u-cache analogue) ────
        g0_mass = np.take(mass, self._g0_mass_index_w, axis=-1)
        Z = k.interp_catmull_rom(
            k.gamma_table, self._g0_index_w, g0_mass,
            k.gamma_min, k.gamma_delta)                   # (ne, n_gamma_rows)

        amp_r = np.ascontiguousarray(np.real(amp), np.float64)
        amp_i = np.ascontiguousarray(np.imag(amp), np.float64)
        Z_r = np.ascontiguousarray(np.real(Z), np.float64)
        Z_i = np.ascontiguousarray(np.imag(Z), np.float64)
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
            self._lib.cpu_pwa_s0c_load(
                _db(mass), n_mass,
                _db(amp_r), _db(amp_i),
                _db(Z_r), _db(Z_i),
                _db(weight), _db(bkg), ne,
                _ib(self._m0_index_w),
                _ib(self._mass_index_w),
                _ib(self.row_start),
                nw, self.n_wave_base, self.n_proj, n_gr, self.n_basis),
            self._lib, ne)
        dh._keep = ka_list
        dh._weight = weight
        dh._bkg = bkg
        return dh

    # -- compute -------------------------------------------------------

    def compute(self, params, data_handle, norm=None, return_p=True):
        """Forward NLL + gradients for the s0c projection-sum PWA.

        Args:
            params: dict with ``ck`` (complex, length N), ``m0``, ``g0``.
            data_handle: DataHandle from :meth:`load_data`.
            norm: optional normalisation factor (None → unnormalised).
            return_p: if False, P is not computed/returned (None).

        Returns:
            ``(Q, grads, P)`` with ``grads`` keys ``ck``/``m0``/``g0``.
        """
        ck = np.asarray(params["ck"])
        nbase = self.n_wave_base
        if len(ck) != nbase:
            raise ValueError(
                f"cpu_pwa_s0c: ck length {len(ck)} != n_wave/n_proj = "
                f"{nbase} ({self.n_wave}/{self.n_proj})")

        use_norm = 0 if norm is None else 1
        norm_val = float(norm) if norm is not None else 0.0

        ck_r = np.ascontiguousarray(np.real(ck), np.float64)
        ck_i = np.ascontiguousarray(np.imag(ck), np.float64)
        m0 = np.zeros(self.n_m0_params, np.float64)
        m0[:len(params["m0"])] = np.asarray(params["m0"])
        g0 = np.zeros(self.n_g0_params, np.float64)
        g0[:len(params["g0"])] = np.asarray(params["g0"])

        # ── uniform coefficient assembly (s0 is slot 0, -1 slot 1) ──
        m0_w = m0[self._m0_index_w]
        nf = self.n_basis
        nw = self.n_wave
        coeff = np.zeros(nw * nf, np.float64)
        coeff[0::nf] = m0_w * m0_w
        coeff[1::nf] = -1.0
        rs = self.row_start
        for w in range(nw):
            r0 = int(rs[w])
            r1 = (int(rs[w + 1]) if w + 1 < nw else self.n_gamma_rows)
            coeff[w * nf + 2:w * nf + 2 + (r1 - r0)] = \
                m0_w[self._wave_of_row[r0:r1]] * g0[self._g0_index_w[r0:r1]]

        gcoeff = np.zeros(nw * nf, np.float64)
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

        Q = self._lib.cpu_pwa_s0c_compute(
            data_handle.ptr,
            _db(coeff), _db(ck_r), _db(ck_i),
            norm_val, use_norm,
            _db(gcoeff), _db(ogck_r), _db(ogck_i),
            (_db(P) if P is not None else _ffi.NULL),
            dnorm)
        self._last_dnorm = float(dnorm[0]) if use_norm else None

        gc = gcoeff.reshape(nw, nf)
        ogs0 = np.ascontiguousarray(gc[:, 0])
        ogmult = np.zeros(self.n_gamma_rows, np.float64)
        for w in range(nw):
            r0 = int(rs[w])
            r1 = (int(rs[w + 1]) if w + 1 < nw else self.n_gamma_rows)
            ogmult[r0:r1] = gc[w, 2:2 + (r1 - r0)]

        # ── host chain: (s0, mult) -> (m0, g0) ──────────────────────
        # s0_w = m0_w²              -> dQ/dm0_w += 2·m0_w·dQ/ds0_w
        # mult_l = m0_w(l)·g0_l     -> dQ/dm0_w += Σ_l g0_l·dQ/dmult_l
        #                              dQ/dg0_l = m0_w(l)·dQ/dmult_l
        g0_row = g0[self._g0_index_w]
        dm0_w = 2.0 * m0_w * ogs0 + np.bincount(
            self._wave_of_row, weights=g0_row * ogmult,
            minlength=self.n_wave)
        grad_m0 = np.bincount(self._m0_index_w, weights=dm0_w,
                              minlength=self.n_m0_params)
        grad_g0 = np.bincount(self._g0_index_w,
                              weights=m0_w[self._wave_of_row] * ogmult,
                              minlength=self.n_g0_params)

        grads = {"ck": ogck_r + 1j * ogck_i,
                 "m0": grad_m0, "g0": grad_g0}
        return Q, grads, P

    # -- cleanup -------------------------------------------------------

    def free(self):
        pass

    def __del__(self):
        self.free()
