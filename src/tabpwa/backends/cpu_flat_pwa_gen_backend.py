"""``cpu_flat_pwa_gen`` — the flat ``pwa_gen`` config on the CPU.

C + OpenMP + AVX-512 fp64 (``kernels_cpu_flat_pwa_gen.c``), with the
**semantics of ``numpy_flat``**: the same ``Q``, ``P``, ``dck``, ``dcoeff``
and ``dnorm`` — at the fp64 round-off floor, because nothing is fp32 and
nothing is cached between calls.

What it is
----------

The flat ``pwa_gen`` family had no CPU implementation (``numpy_flat`` is the
plain-loops fp64 reference, ~0.13 Mev/s) while the GPU tile
(``cuda_flat_pwa_gen_s0e``) runs at ~39 Mev/s because fp64 is 1/64 of fp32 on
a consumer part.  This kernel is the missing fallback: AVX-512 fp64, events
as the vector lanes, recomputing the parameter-independent factors (``fa``,
``fl``, ``num``, the basis rows) on every call instead of caching them —
5.9 GB of device cache on the GPU, none here.

It reads the config **exactly as the builder emits it** (the full,
non-deduped form: ``fl_table`` is ``(n_wave, n_decay, n_interp)`` and
``num_table`` is ``(n_wave, n_res, n_interp)``), like the CUDA siblings, and
rejects an already-``dedup``ed config.

The folds (all byte-exact table data — the ``uslot_map`` / ``amp_slot_map``
principle, a sampled value is never compared):

* ``fl`` uses: ``(fl_table row bytes, q column)``      52 -> 26 (production)
* ``num`` uses: ``(num_table row bytes, m column)``    26 ->  9
* basis rows: ``(basis_table row bytes, m column)``    64 -> 48
* ``matrix_angle`` columns (``fa``)                    52 -> 26

plus two in-kernel folds: a constant basis row skips its interpolation, and
an angular component with ``k == 0`` is the constant ``cos(b)`` (never
reaches the ``cos`` loop).  So the per-event interpolation count is 140
Catmull-Rom evaluations, not 200, and the angle-basis matmul walks
``matrix_angle``'s TRUE non-zeros (304 of 2704 entries).

Handle contract (AGENTS gotcha 15)
----------------------------------

``load_data`` builds a **fresh** C handle that owns its own ``P`` buffer and
borrows the caller's (transposed) arrays; nothing is freed or overwritten
there, so the phase-space handle and the data handle stay independent for
the whole fit.

``batch_size`` changes nothing: the sample is walked in event tiles of 8 and
the parallel axis is the tile (or the log-sum group), never a chunk, so
every ``batch_size`` gives bit-identical numbers.
"""
import os

import numpy as np
from cffi import FFI

from .core import ComputeBackend, register_backend

_ffi = None
_lib = None

#: ``rdtsc`` stage counters reported by ``compute(..., stage_ns=...)``
STAGE_NAMES = ("w=gamma*coeff", "grid+bins", "ka", "fa", "fl", "num", "den",
               "fwd+bwd", "accumulate")


def _load_lib():
    """Build/load ``libcpu_kernels_cpu_flat_pwa_gen.so`` (gcc, cached)."""
    global _ffi, _lib
    if _ffi is None:
        ffi = FFI()
        ffi.cdef("""
void* cpu_flat_pwa_gen_create(
    const double*,int,int, const int*,const int*,int, const int*,
    const double*,const double*,int, const int*,const int*,int, const int*,
    const double*,const double*,int, const int*,const int*,int,
    const int*,const int*,const int*,const double*,int,
    const int*,const double*,const double*,int,int,
    const int*,const int*,const double*,const double*,const int*,int,
    double,double,int, double,double,int,
    int,int,int,int,int, int,int,int, int,int,int);
void cpu_flat_pwa_gen_free_ctx(void*);
void* cpu_flat_pwa_gen_load(void*,const double*,const double*,
    const double*,const double*,const double*,int);
void cpu_flat_pwa_gen_free(void*);
double cpu_flat_pwa_gen_compute(void*,void*,
    const double*,const double*,const double*,
    double,int, double*,double*,double*, double*,int,double*, long long*);
int cpu_flat_pwa_gen_debug(void*,void*,const double*,const double*,
    const double*,double*);
""")
        _ffi = ffi
    if _lib is None:
        from tabpwa.cpu.build import ensure
        lib_path = ensure("kernels_cpu_flat_pwa_gen.c",
                          "libcpu_kernels_cpu_flat_pwa_gen.so")
        if not lib_path:
            raise RuntimeError(
                "cpu_flat_pwa_gen: the CPU library could not be built "
                "(kernels_cpu_flat_pwa_gen.c needs AVX-512: gcc -shared "
                "-fPIC -O3 -march=native -fopenmp -lm)")
        _lib = _ffi.dlopen(lib_path)
    return _lib


# ── host-side dedup: the four byte-exact keys ────────────────────────────

def _dedup_uses(rows, cols, keys):
    """Dedup ``(table row, x column)`` uses on a byte-exact row key.

    ``rows`` is the table-row index each use samples, ``cols`` the event
    column it reads, ``keys[i]`` the row's own bytes (its physical identity).
    Returns ``(u_row, u_col, use)``: per surviving slot its row and column,
    and ``use[i] -> slot``.
    """
    seen, u_row, u_col, use = {}, [], [], np.zeros(len(rows), np.int32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        k = (keys[i], int(c))
        u = seen.get(k)
        if u is None:
            u = len(u_row)
            seen[k] = u
            u_row.append(int(r))
            u_col.append(int(c))
        use[i] = u
    return (np.asarray(u_row, np.int64), np.asarray(u_col, np.int64), use)


def _uslot_map(basis_table, basis_m_index):
    """Dedup basis rows on ``(row bytes, mass column)`` -> (row, col, slot)."""
    bas = np.ascontiguousarray(np.asarray(basis_table, np.complex128))
    bmi = np.ascontiguousarray(np.asarray(basis_m_index, np.int64))
    n = int(bas.shape[0])
    if bmi.size != n:
        raise ValueError(
            f"cpu_flat_pwa_gen: basis_m_index has {bmi.size} entries, "
            f"expected n_basis = {n}")
    re, im = np.ascontiguousarray(bas.real), np.ascontiguousarray(bas.imag)
    seen, u_row, u_col, slot = {}, [], [], np.zeros(n, np.int32)
    for m in range(n):
        k = (re[m].tobytes(), im[m].tobytes(), int(bmi[m]))
        u = seen.get(k)
        if u is None:
            u = len(u_row)
            seen[k] = u
            u_row.append(m)
            u_col.append(int(bmi[m]))
        slot[m] = u
    return (np.asarray(u_row, np.int64), np.asarray(u_col, np.int64), slot)


def _matrix_angle_csr(am, n_wave, n_proj):
    """``fa = ka·matrix_angle`` over the matrix's TRUE non-zeros.

    ``matrix_angle`` is ``(n_ab, n_proj·n_wave)`` complex, p-major.  Returns
    the CSR ``(ptr, row, val_r, val_i)`` per output column plus
    ``(fa_slot, n_fa)`` — columns whose entries are byte-identical collapse
    onto one computation (52 -> 26 on the production config).
    """
    am = np.ascontiguousarray(np.asarray(am, np.complex128))
    n_ab, ncol = am.shape
    if ncol != n_proj * n_wave:
        raise ValueError(
            f"cpu_flat_pwa_gen: matrix_angle has {ncol} columns, expected "
            f"n_proj·n_wave = {n_proj}·{n_wave}")
    ar = np.ascontiguousarray(am.real)
    ai = np.ascontiguousarray(am.imag)
    seen, fa_slot, n_fa = {}, np.zeros(ncol, np.int32), 0
    for c in range(ncol):
        k = (ar[:, c].tobytes(), ai[:, c].tobytes())
        u = seen.get(k)
        if u is None:
            u = n_fa
            seen[k] = u
            n_fa += 1
        fa_slot[c] = u
    rep = np.empty(n_fa, np.int64)
    for c, u in enumerate(fa_slot):
        rep[u] = c
    ptr = np.zeros(n_fa + 1, np.int64)
    row, val_r, val_i = [], [], []
    for u in range(n_fa):
        c = int(rep[u])
        # a complex entry is a non-zero if EITHER plane is non-zero
        idx = np.nonzero((ar[:, c] != 0.0) | (ai[:, c] != 0.0))[0]
        ptr[u] = len(row)
        for a in idx:
            row.append(int(a))
            val_r.append(float(ar[a, c]))
            val_i.append(float(ai[a, c]))
    ptr[n_fa] = len(row)
    return (ptr.astype(np.int32), np.asarray(row, np.int32),
            np.asarray(val_r, np.float64), np.asarray(val_i, np.float64),
            fa_slot, n_fa)


def _gamma_csr(gamma_matrix, n_wave, n_res, n_basis, slot_of_basis):
    """Per ``(wave, factor)`` CSR of the gamma's TRUE non-zeros.

    Returns ``(off, m, u, w)``: ``[off[kr], off[kr+1])`` indexes the basis
    column ``m``, its deduped basis slot ``u`` and the gamma value ``w``.
    A factor with an all-zero row raises: its ``D`` would be 0.
    """
    g = np.ascontiguousarray(np.asarray(gamma_matrix, np.float64))
    if g.shape != (n_wave, n_res, n_basis):
        raise ValueError(
            f"cpu_flat_pwa_gen: gamma_matrix has shape {g.shape}, expected "
            f"(n_wave, n_res, n_basis) = ({n_wave}, {n_res}, {n_basis})")
    gf = g.reshape(n_wave * n_res, n_basis)
    off = np.zeros(n_wave * n_res + 1, np.int64)
    m, u, w = [], [], []
    for kr in range(n_wave * n_res):
        idx = np.nonzero(gf[kr] != 0.0)[0]
        if idx.size == 0:
            raise ValueError(
                f"cpu_flat_pwa_gen: wave/factor {kr} selects no basis column "
                f"(gamma row is all zero) — its denominator would be 0")
        off[kr] = len(m)
        for mm in idx:
            m.append(int(mm))
            u.append(int(slot_of_basis[mm]))
            w.append(float(gf[kr, mm]))
    off[-1] = len(m)
    return (off.astype(np.int32), np.asarray(m, np.int32),
            np.asarray(u, np.int32), np.asarray(w, np.float64))


class CPUFlatPwaGenHandle:
    """One loaded sample: a per-load C handle that owns its ``P`` buffer.

    The fitter holds the phase-space handle and the data handle at the same
    time, so nothing here is shared between loads.
    """

    __slots__ = ("ptr", "ne", "_lib", "_keep")

    def __init__(self, ptr, lib, ne):
        self.ptr = ptr
        self._lib = lib
        self.ne = int(ne)
        self._keep = []

    def free(self):
        """Idempotent: the second call (and a NULL handle) is a no-op."""
        if self.ptr is not None:
            self._lib.cpu_flat_pwa_gen_free(self.ptr)
            self.ptr = None
            self._keep.clear()

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class CPUKernelFlatPwaGen:
    """The flat ``pwa_gen`` config, evaluated event-tile-vectorised on the CPU.

    ``params`` is ``{"ck": complex (n_wave,), "coeff": real (n_basis,)}``;
    ``grads`` is ``{"ck", "coeff"}`` plus ``"norm"`` on a normed call.
    ``grad_ck`` is the raw Wirtinger half (``CKProduct.backprop_grad``
    applies the 2·Re convention downstream).
    """

    def __init__(self, config, batch_size=50000, resolution_size=1,
                 general_cos=False):
        global _lib
        if "fl_index" in config or "num_index" in config:
            raise ValueError(
                "cpu_flat_pwa_gen: pass the FULL (non-deduped) config — this "
                "backend reads every row where the builder emits it, and a "
                "config that already carries fl_index/num_index has a "
                "different row layout")
        resolution_size = int(resolution_size)
        if resolution_size < 1:
            raise ValueError(
                "cpu_flat_pwa_gen: resolution_size must be >= 1, got "
                f"{resolution_size}")
        c = config
        self.config = c
        fl = np.asarray(c["fl_table"])
        num = np.asarray(c["num_table"])
        if fl.ndim != 3 or num.ndim != 3:
            raise ValueError(
                "cpu_flat_pwa_gen: the config must carry the FULL tables — "
                f"fl_table (n_wave, n_decay, n_interp) and num_table "
                f"(n_wave, n_res, n_interp); got {fl.shape} / {num.shape}")

        self.n_wave = int(fl.shape[0])
        self.n_decay = int(fl.shape[1])
        self.n_res = int(num.shape[1])
        self.n_proj = max(1, int(np.asarray(c["matrix_angle"]).shape[1])
                          // self.n_wave)
        self.n_basis = int(np.asarray(c["basis_table"]).shape[0])
        self.n_ab = int(np.asarray(c["angle_k"]).shape[0])
        self.n_comp = int(np.asarray(c["angle_k"]).shape[1])
        self.n_mass = int(np.max(c["num_m_index"])) + 1
        self.n_momentum = int(np.max(c["fl_q_index"])) + 1
        self.n_angle_total = int(np.max(c["angle_index"])) + 1
        self.resolution_size = resolution_size
        self.batch_size = int(batch_size)
        self.general_cos = bool(general_cos)
        self._last_dnorm = None

        fl_tab = np.ascontiguousarray(fl.reshape(-1, fl.shape[-1]),
                                      np.float64)
        num_r = np.ascontiguousarray(num.real.reshape(-1, num.shape[-1]))
        num_i = np.ascontiguousarray(num.imag.reshape(-1, num.shape[-1]))
        bas = np.asarray(c["basis_table"], np.complex128)
        bas_r = np.ascontiguousarray(bas.real)
        bas_i = np.ascontiguousarray(bas.imag)
        gam = np.asarray(c["gamma_matrix"], np.float64)

        # ── the four byte-exact dedups ──────────────────────────────────
        fl_q = np.asarray(c["fl_q_index"], np.int64).reshape(self.n_wave
                                                             * self.n_decay)
        fl_keys = [fl_tab[i].tobytes() for i in range(self.n_wave
                                                     * self.n_decay)]
        self._fl_row, self._fl_col, self._fl_use = _dedup_uses(
            range(self.n_wave * self.n_decay), fl_q, fl_keys)
        num_m = np.asarray(c["num_m_index"], np.int64).reshape(
            self.n_wave * self.n_res)
        num_keys = [(num_r[i].tobytes(), num_i[i].tobytes())
                    for i in range(self.n_wave * self.n_res)]
        self._num_row, self._num_col, self._num_use = _dedup_uses(
            range(self.n_wave * self.n_res), num_m, num_keys)
        (self._us_row, self._us_col,
         self._slot_of_basis) = _uslot_map(bas, c["basis_m_index"])
        self.n_uslot = int(self._us_row.size)
        (self._g_off, self._g_m, self._g_u,
         self._g_w) = _gamma_csr(gam, self.n_wave, self.n_res, self.n_basis,
                                 self._slot_of_basis)
        (self._fa_ptr, self._fa_row, self._fa_r, self._fa_i,
         self._fa_slot, self._n_fa) = _matrix_angle_csr(
            c["matrix_angle"], self.n_wave, self.n_proj)

        self._angle_index = np.ascontiguousarray(c["angle_index"], np.int32)
        self._angle_k = np.ascontiguousarray(np.asarray(c["angle_k"],
                                                        np.float64))
        self._angle_b = np.ascontiguousarray(np.asarray(c["angle_b"],
                                                        np.float64))
        self._m_min = float(c["m_min"])
        self._m_delta = float(c["m_delta"])
        self._q_min = float(c["q_min"])
        self._q_delta = float(c["q_delta"])
        self._m_bins = int(bas.shape[1])
        self._q_bins = int(fl_tab.shape[1])

        if _lib is None:
            _lib = _load_lib()
        self._lib = _lib
        self._keep = []

        def _D(a):
            arr = np.ascontiguousarray(a, np.float64)
            self._keep.append(arr)
            return _ffi.cast("double*", _ffi.from_buffer(arr))

        def _I(a):
            arr = np.ascontiguousarray(a, np.int32)
            self._keep.append(arr)
            return _ffi.cast("int*", _ffi.from_buffer(arr))

        flags = 1 if self.general_cos else 0     # CFPG_FLAG_GENERAL_COS
        self._ctx = self._lib.cpu_flat_pwa_gen_create(
            _D(fl_tab), len(fl_tab), self._q_bins,
            _I(self._fl_row), _I(self._fl_col), int(self._fl_row.size),
            _I(self._fl_use),
            _D(num_r), _D(num_i), int(num.shape[0]),
            _I(self._num_row), _I(self._num_col), int(self._num_row.size),
            _I(self._num_use),
            _D(bas_r), _D(bas_i), int(bas.shape[0]),
            _I(self._us_row), _I(self._us_col), self.n_uslot,
            _I(self._g_off), _I(self._g_m), _I(self._g_u), _D(self._g_w),
            int(self._g_w.size),
            _I(self._angle_index), _D(self._angle_k), _D(self._angle_b),
            self.n_ab, self.n_comp,
            _I(self._fa_ptr), _I(self._fa_row), _D(self._fa_r), _D(self._fa_i),
            _I(self._fa_slot), self._n_fa,
            self._m_min, self._m_delta, self._m_bins,
            self._q_min, self._q_delta, self._q_bins,
            self.n_wave, self.n_proj, self.n_res, self.n_decay, self.n_basis,
            self.n_mass, self.n_momentum,
            self.n_angle_total * self.n_comp,
            self.resolution_size, self.batch_size, flags)
        if not self._ctx:
            raise RuntimeError(
                "cpu_flat_pwa_gen: context creation failed")

    # ── reporting, for the fixture and the benchmark ────────────────────
    def kernel_stats(self):
        """Shape / dedup / vectorisation facts (one dict, no guesswork)."""
        return {
            "backend": "cpu_flat_pwa_gen",
            "lanes": 8,
            "isa": "avx512f (fp64)",
            "threads": len(os.sched_getaffinity(0)),
            "n_wave": self.n_wave, "n_proj": self.n_proj,
            "n_res": self.n_res, "n_decay": self.n_decay,
            "n_basis": self.n_basis, "n_ab": self.n_ab,
            "n_uslot": self.n_uslot, "n_fa": self._n_fa,
            "n_fl_u": int(self._fl_row.size),
            "n_num_u": int(self._num_row.size),
            "n_gamma_nz": int(self._g_w.size),
            "n_fa_nz": int(self._fa_r.size),
            "resolution_size": self.resolution_size,
            "general_cos": self.general_cos,
            "cache_bytes": 0,
        }

    # ── data ────────────────────────────────────────────────────────────
    def load_data(self, data):
        """Transpose the sample to per-column (SoA) fp64 arrays + new handle."""
        mass = np.asarray(data["mass"], np.float64)
        mom = np.asarray(data["q"], np.float64)
        ang = np.asarray(data["angle"], np.float64)
        ne = mass.shape[0]
        mass = mass.reshape(ne, -1)[:, :self.n_mass]
        mom = mom.reshape(ne, -1)[:, :self.n_momentum]
        ang = ang.reshape(ne, self.n_angle_total, self.n_comp)[
            :, :self.n_angle_total, :self.n_comp]
        mass_s = np.ascontiguousarray(mass.T, np.float64)
        q_s = np.ascontiguousarray(mom.T, np.float64)
        ang_s = np.ascontiguousarray(
            ang.transpose(1, 2, 0).reshape(-1, ne), np.float64)
        w = np.ascontiguousarray(np.asarray(
            data.get("weight", np.ones(ne)), np.float64).ravel())
        b = data.get("bkg", 1.0)
        b = np.ascontiguousarray(np.asarray(
            np.full(ne, float(b)) if np.isscalar(b) else b,
            np.float64).ravel())
        keep = [mass_s, q_s, ang_s, w, b]
        ptr = self._lib.cpu_flat_pwa_gen_load(
            self._ctx,
            _ffi.cast("double*", _ffi.from_buffer(mass_s)),
            _ffi.cast("double*", _ffi.from_buffer(q_s)),
            _ffi.cast("double*", _ffi.from_buffer(ang_s)),
            _ffi.cast("double*", _ffi.from_buffer(w)),
            _ffi.cast("double*", _ffi.from_buffer(b)), ne)
        if not ptr:
            raise RuntimeError("cpu_flat_pwa_gen: load_data failed")
        h = CPUFlatPwaGenHandle(ptr, self._lib, ne)
        h._keep = keep
        return h

    def free(self):
        """Context only.  Handles are owned by whoever called ``load_data``."""
        if getattr(self, "_ctx", None):
            self._lib.cpu_flat_pwa_gen_free_ctx(self._ctx)
            self._ctx = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    # ── compute ─────────────────────────────────────────────────────────
    def compute(self, params, data_handle, norm=None, return_p=True,
                stage_ns=None):
        """``(Q, grads, P)``; ``stage_ns`` collects the per-stage split."""
        if data_handle.ptr is None:
            raise RuntimeError(
                "cpu_flat_pwa_gen: this data handle was already freed")
        ck = np.asarray(params["ck"], np.complex128).ravel()
        coeff = np.asarray(params["coeff"], np.float64).ravel()
        if ck.size != self.n_wave:
            raise ValueError(
                f"cpu_flat_pwa_gen: ck has {ck.size} entries, expected "
                f"n_wave = {self.n_wave}")
        if coeff.size != self.n_basis:
            raise ValueError(
                f"cpu_flat_pwa_gen: coeff has {coeff.size} entries, expected "
                f"n_basis = {self.n_basis}")
        ck = np.ascontiguousarray(ck)
        ck_r = np.ascontiguousarray(ck.real)
        ck_i = np.ascontiguousarray(ck.imag)
        coeff = np.ascontiguousarray(coeff)
        ne = data_handle.ne
        gck_r = np.zeros(self.n_wave)
        gck_i = np.zeros(self.n_wave)
        gco = np.zeros(self.n_basis)
        P = np.zeros(ne) if return_p else None
        dnorm = _ffi.new("double*")
        if stage_ns is not None:
            stages = _ffi.new("long long[%d]" % len(STAGE_NAMES))
        else:
            stages = _ffi.NULL
        Q = self._lib.cpu_flat_pwa_gen_compute(
            self._ctx, data_handle.ptr,
            _ffi.cast("double*", _ffi.from_buffer(coeff)),
            _ffi.cast("double*", _ffi.from_buffer(ck_r)),
            _ffi.cast("double*", _ffi.from_buffer(ck_i)),
            float(norm) if norm is not None else 0.0,
            1 if norm is not None else 0,
            _ffi.cast("double*", _ffi.from_buffer(gck_r)),
            _ffi.cast("double*", _ffi.from_buffer(gck_i)),
            _ffi.cast("double*", _ffi.from_buffer(gco)),
            (_ffi.cast("double*", _ffi.from_buffer(P))
             if P is not None else _ffi.NULL),
            1 if return_p else 0, dnorm, stages)
        if stage_ns is not None:
            for i, name in enumerate(STAGE_NAMES):
                stage_ns[name] = stage_ns.get(name, 0) + int(stages[i])
        grads = {"ck": gck_r + 1j * gck_i, "coeff": gco}
        if norm is not None:
            self._last_dnorm = float(dnorm[0])
            grads["norm"] = self._last_dnorm
        else:
            self._last_dnorm = None
        return float(Q), grads, P


@register_backend("cpu_flat_pwa_gen", model="pwa_gen")
class CPUFlatPwaGenBackend(ComputeBackend):
    """Backend wrapper for :class:`CPUKernelFlatPwaGen` (untagged leaf)."""

    def __init__(self, kernel_config, batch_size=50000, resolution_size=1,
                 general_cos=False):
        self.kernel = None   # free/__del__ must be safe if the build fails
        self.kernel = CPUKernelFlatPwaGen(
            kernel_config, batch_size=batch_size,
            resolution_size=resolution_size, general_cos=general_cos)
        self.resolution_size = self.kernel.resolution_size

    def load_data(self, data_np):
        return self.kernel.load_data(data_np)

    def compute(self, params, data_handle, norm=None, return_p=True):
        return self.kernel.compute(params, data_handle, norm=norm,
                                   return_p=return_p)

    def free(self):
        if self.kernel is not None:
            self.kernel.free()
            self.kernel = None

    def __del__(self):
        self.free()
