"""``cuda_flat_pwa_gen2`` — the flat ``pwa_gen`` tile with four reductions.

Case-1 copy of :mod:`tabpwa.cuda._flat_cache_pwa_gen` (kernel policy §1): own
``.cu``, own ``.so``, own symbols, and the base kernel/wrapper stay
byte-identical.  The four changes, all of them exact reassociations of the
same sums:

1. **`u`-merge.**  ``dQ/dck`` and ``dQ/dcommon`` were the same reduction over
   the projections done twice (the base kernel's phases 4 and 5):

       u_{e,k}      = Σ_p dQ/dA_{e,p}·fa_{e,p,k}      (no `ck` factor)
       dQ/dcommon_k = ck_k·u_{e,k}
       dQ/dck_k     = Σ_e common_{e,k}·u_{e,k}

   One ``(e,k)`` pass produces both, so phase 4 is gone and ``common = F/D``
   is rebuilt once per ``(e,k)`` instead of twice.

2. **``n_res == 1`` specialisation** (template ``NR``): ``D = s_0`` exactly,
   so the per-factor ``s_s`` buffer is dropped (never read at one factor) and
   the factor lives in registers.  −12.8 KiB of shared memory at
   ``n_wave = 100``, and the compiler folds the whole ``l`` loop.

3. **`n_slot` collapse of ``dQ/dcoeff`` — GATED.**  ``gamma[k]`` is a 0/1
   selector, so waves sharing a row have bit-identical ``s`` and ``D``; the
   per-``(k,l,m)`` dot product can be reassociated to per-``(slot,l,m)`` with
   one ``dsum_{e,slot} = Σ_{k∈slot} dQ/dD_{e,k}``.  It is a **regression**
   when every wave has its own row (the extra per-``(e,k)`` accumulation buys
   nothing), so the host enables it only when ``2·n_slot <= n_wave``; see
   :func:`slot_threshold_says_use`.  ``slot_collapse=None`` (default) means
   auto; ``True``/``False`` force a path (test hook — both must agree).

4. **``A_{e,p}`` cache for the grouped backward launch.**  ``resolution_size``
   runs the tile twice (mode 1 = forward + group sums, mode 2 = backward with
   the finished ``dQ/dP``).  Mode 2 only ever needed ``A`` from the forward
   pass, so mode 1 stores it (``2·ne·n_proj`` doubles = 2 MiB at 65k/2) and
   mode 2 skips the projection sum entirely.  ``common`` would be
   ``ne·n_wave`` complex (100 MiB) and is deliberately not cached.

Host contract, identical to the other flat backends::

    params = {"ck": complex (n_wave,), "coeff": real (n_basis,)}
    grads  = {"ck", "coeff", "norm"}      # norm only on a normed call
"""
import numpy as np

from tabpwa.backends.core import ComputeBackend, register_backend

_ffi = None
_lib = None

#: The slot-collapse gate: enable only when the per-slot work is strictly less
#: than the per-wave work it replaces.
#:
#: The per-``(k,l,m)`` loop costs ``n_wave·n_res·n_basis`` dot products; the
#: collapsed one costs ``n_slot·n_res·n_basis`` plus one ``(e,k)`` accumulation
#: plus the per-slot ``w`` product.  ``2·n_slot <= n_wave`` is the threshold
#: below which the saved dot products clearly dominate.
#:
#: MEASURED (n_wave=100, 65k events, interleaved min-of-5, collapse ON vs OFF):
#: ``n_slot`` 1 → 1.17×, 10 → 1.22×, 25 → 1.34×, 50 → 1.19×, 51 → 1.17×,
#: 64 → 1.11×, 100 → 1.00×.  So the collapse never LOST in measurement; at
#: ``n_slot == n_wave`` it is exactly break-even (the accumulation is paid and
#: buys nothing).  The gate therefore sits at the break-even boundary: it keeps
#: the strict win and declines the region where there is nothing to gain.
SLOT_GATE_RATIO = 2


def slot_threshold_says_use(n_slot, n_wave, ratio=SLOT_GATE_RATIO):
    """The gate, in one place so a test can ask without a kernel."""
    return int(n_slot) * int(ratio) <= int(n_wave)


def slot_map(gamma_matrix):
    """Distinct ``gamma_matrix[k]`` rows → slot maps and a CSR wave list.

    Two waves in one slot have bit-identical ``s_{e,k,l}`` and ``D_{e,k}`` by
    construction (``s`` is a function of the event, the basis and ``coeff``
    only), which is exactly what makes the collapsed ``dQ/dcoeff`` exact.

    Returns ``(slot_of_wave, slot_rep, rows, slot_ptr, slot_waves)``: the CSR
    pair lists each slot's waves so the kernel can sum ``dQ/dD`` over a slot
    with ONE thread per ``(event, slot)``, never a contended shared atomic.
    """
    g = np.ascontiguousarray(np.asarray(gamma_matrix, np.float64))
    n_wave = int(g.shape[0])
    rows = g.reshape(n_wave, -1)
    seen, slot_of_wave, rep = {}, np.zeros(n_wave, np.int32), []
    for k in range(n_wave):
        key = rows[k].tobytes()
        sl = seen.get(key)
        if sl is None:
            sl = len(rep)
            seen[key] = sl
            rep.append(k)
        slot_of_wave[k] = sl
    rep = np.asarray(rep, np.int32)
    n_slot = int(rep.size)
    counts = np.bincount(slot_of_wave, minlength=n_slot)
    slot_ptr = np.zeros(n_slot + 1, np.int32)
    np.cumsum(counts, out=slot_ptr[1:])
    order = np.argsort(slot_of_wave, kind="stable")
    slot_waves = np.ascontiguousarray(order.astype(np.int32))
    return slot_of_wave, rep, rows[rep], slot_ptr, slot_waves


def _load_lib():
    from cffi import FFI
    from tabpwa.cuda.build import ensure
    global _ffi
    if _ffi is None:
        _ffi = FFI()
        _ffi.cdef("""
void* cuda_create_context_flat_pwa_gen2(
    const double*,int, const int*,int, const double*,const double*,int,
    const int*,int, const double*,const double*,int, const int*,int,
    const double*,int, const double*,int, const int*,int, const int*,int,
    const int*,int, const int*,int, int,int,
    const int*,int, const double*,int, const double*,int,
    const double*,const double*,int,
    double,double,int, double,double,int,
    int,int,int,int,int, int,int,int,int,int, int,int);
void* cuda_load_data_flat_pwa_gen2(void*,const double*,const int*,
    const double*,const int*, const double*,const int*,
    const double*,const double*,int);
void cuda_free_data_flat_pwa_gen2(void*);
void cuda_free_context_flat_pwa_gen2(void*);
void cuda_compute_flat_pwa_gen2(void*,void*,const double*,const double*,
    const double*,double,int, double*,double*,double*,double*,double*,double*);
int cuda_get_device_count_flat_pwa_gen2(void);
long long cuda_tile_bytes_flat_pwa_gen2(int,int,int,int,int,int);
long long cuda_handle_dA_bytes_flat_pwa_gen2(void*);
int cuda_smem_bytes_flat_pwa_gen2(void*);
void cuda_get_device_name_flat_pwa_gen2(char*,int);
""")
    lib_path = ensure("kernels_flat_pwa_gen2.cu",
                      "libcuda_kernels_flat_pwa_gen2.so")
    if not lib_path:
        raise RuntimeError("cuda_flat_pwa_gen2: library not found")
    return _ffi.dlopen(lib_path)


class FlatPwaGen2DataHandle:
    """One loaded sample — a per-load C handle, never the context.

    The fitter holds the phase-space handle and the data handle at the same
    time (``_compute_norm_batched`` integrates over one while ``get_nll_raw``
    evaluates the other), so each ``load_data`` owns its arrays.
    """

    __slots__ = ("ptr", "ne", "_lib", "_keep", "dA_bytes", "__weakref__")

    def __init__(self, ptr, lib, ne):
        self.ptr = ptr
        self._lib = lib
        self.ne = int(ne)
        self._keep = []
        #: bytes of the saved-`A` buffer (0 unless the grouped objective is on)
        self.dA_bytes = int(lib.cuda_handle_dA_bytes_flat_pwa_gen2(ptr))

    def free(self):
        """Idempotent: the second call (and a NULL handle) is a no-op."""
        if self.ptr is not None:
            self._lib.cuda_free_data_flat_pwa_gen2(self.ptr)
            self.ptr = None
            self._keep.clear()

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class CUDAKernelFlatPwaGen2:
    """Fused-tile CUDA implementation of the flat ``pwa_gen`` config."""

    def __init__(self, config, batch_size=50000, resolution_size=1,
                 slot_collapse=None):
        global _lib
        if "fl_index" in config or "num_index" in config:
            raise ValueError(
                "cuda_flat_pwa_gen2: pass the FULL (non-deduped) config — "
                "this backend reads every row where the builder emits it, and "
                "a config that already carries fl_index/num_index has a "
                "different row layout")
        resolution_size = int(resolution_size)
        if resolution_size < 1:
            raise ValueError(
                "cuda_flat_pwa_gen2: resolution_size must be >= 1, got "
                f"{resolution_size}")
        if _lib is None:
            _lib = _load_lib()
        self._lib = _lib
        self.config = config
        if _lib.cuda_get_device_count_flat_pwa_gen2() <= 0:
            raise RuntimeError("cuda_flat_pwa_gen2: no CUDA devices")
        buf = _ffi.new("char[256]")
        _lib.cuda_get_device_name_flat_pwa_gen2(buf, 256)
        self.device_name = _ffi.string(buf).decode()

        c = config
        self.n_wave = int(np.asarray(c["fl_table"]).shape[0])
        # n_proj is encoded p-major in matrix_angle's columns
        self.n_proj = max(1, int(np.asarray(c["matrix_angle"]).shape[1])
                          // self.n_wave)
        self.n_res = int(np.asarray(c["num_table"]).shape[1])
        self.n_decay = int(np.asarray(c["fl_q_index"]).shape[1])
        self.n_ab = int(np.asarray(c["angle_k"]).shape[0])
        self.n_comp = int(np.asarray(c["angle_k"]).shape[1])
        self.n_mass = int(np.max(c["num_m_index"])) + 1
        self.n_momentum = int(np.max(c["fl_q_index"])) + 1
        self.n_angle_total = int(np.max(c["angle_index"])) + 1
        self.batch_size = int(batch_size)
        #: events per grouped-log-sum group; 1 = the per-event objective
        self.resolution_size = resolution_size
        self._last_dnorm = None

        # the basis is already folded by `build_from_groups`; nothing to do
        bas = np.asarray(c["basis_table"], np.complex128)
        bas_m = np.asarray(c["basis_m_index"], np.int64)
        self.n_basis = int(bas.shape[0])
        self.basis_keep = np.arange(self.n_basis, dtype=np.int64)
        self.basis_map = self.basis_keep.copy()
        self.n_basis_full = self.n_basis

        gam = np.asarray(c["gamma_matrix"], np.float64)

        # ── the slot collapse and its gate ──────────────────────────
        (self.slot_of_wave, self.slot_rep, gam_slot,
         self.slot_ptr, self.slot_waves) = slot_map(gam)
        self.n_slot = int(self.slot_rep.size)
        auto = slot_threshold_says_use(self.n_slot, self.n_wave)
        if slot_collapse is None:
            use_slots = auto
        else:
            use_slots = bool(slot_collapse)
        # the deep-factor fallback (n_res > GP_MAXRES) has no dsum buffer
        self.use_slots = bool(use_slots) and self.n_res <= 16
        self.slot_collapse_auto = bool(auto)

        keep_alive = []
        fl_tab = np.asarray(c["fl_table"], np.float64).reshape(
            self.n_wave * self.n_decay, -1)
        num = np.asarray(c["num_table"], np.complex128).reshape(
            self.n_wave * self.n_res, -1)
        # matrix_angle is (n_ab, n_proj*n_wave) p-major (column p*n_wave+k);
        # the kernel indexes (k, p, a), so split the columns as
        # (n_proj, n_wave) and move the wave axis first.  Contiguous for
        # upload; the kernel is untouched.
        am = np.ascontiguousarray(
            np.asarray(c["matrix_angle"], np.complex128).T.reshape(
                self.n_proj, self.n_wave, self.n_ab).transpose(1, 0, 2))

        def D(a):
            arr = np.ascontiguousarray(a, np.float64)
            keep_alive.append(arr)
            return _ffi.cast("double*", _ffi.from_buffer(arr))

        def I(a):
            arr = np.ascontiguousarray(a, np.int32)
            keep_alive.append(arr)
            return _ffi.cast("int*", _ffi.from_buffer(arr))

        fl_q = np.asarray(c["fl_q_index"], np.int64)
        num_m = np.asarray(c["num_m_index"], np.int64)
        self._ctx = _lib.cuda_create_context_flat_pwa_gen2(
            D(fl_tab), fl_tab.size, I(fl_q), fl_q.size,
            D(num.real), D(num.imag), num.size, I(num_m), num_m.size,
            D(bas.real), D(bas.imag), bas.size, I(bas_m), self.n_basis,
            D(gam), int(gam.size),
            D(gam_slot), int(gam_slot.size),
            I(self.slot_of_wave), int(self.slot_of_wave.size),
            I(self.slot_rep), int(self.slot_rep.size),
            I(self.slot_ptr), int(self.slot_ptr.size),
            I(self.slot_waves), int(self.slot_waves.size),
            self.n_slot, 1 if self.use_slots else 0,
            I(np.asarray(c["angle_index"])), self.n_ab,
            D(np.asarray(c["angle_k"], np.float64)),
            int(np.asarray(c["angle_k"]).size),
            D(np.asarray(c["angle_b"], np.float64)),
            int(np.asarray(c["angle_b"]).size),
            D(am.real), D(am.imag), am.size,
            float(c["m_min"]), float(c["m_delta"]), int(bas.shape[1]),
            float(c["q_min"]), float(c["q_delta"]), int(fl_tab.shape[1]),
            self.n_wave, self.n_proj, self.n_res, self.n_decay,
            self.n_basis, self.n_ab, self.n_comp,
            self.n_mass, self.n_momentum, self.n_angle_total,
            self.batch_size, self.resolution_size)
        if not self._ctx:
            raise RuntimeError("cuda_flat_pwa_gen2: context creation failed")
        self._keep = keep_alive
        # the ONE definition of the tile layout lives in `gp2_layout` (C);
        # asking the context avoids a second copy drifting against it
        self.smem_bytes = int(
            _lib.cuda_smem_bytes_flat_pwa_gen2(self._ctx))

    # ── data ────────────────────────────────────────────────────────
    def load_data(self, data):
        """Build a FRESH handle; nothing is freed or overwritten here.

        The fitter keeps several samples alive at once (phase space + data),
        so a ``free_data()`` at the top of this method — or a mutable
        ``self._handle`` slot — would make the second load clobber the first.
        """
        mass = np.ascontiguousarray(np.asarray(data["mass"], np.float64))
        mom = np.ascontiguousarray(np.asarray(data["q"], np.float64))
        ang = np.ascontiguousarray(np.asarray(data["angle"], np.float64))
        if ang.ndim == 2:
            ang = ang.reshape(ang.shape[0], self.n_angle_total, self.n_comp)
        ne = mass.shape[0]
        w = np.ascontiguousarray(np.asarray(
            data.get("weight", np.ones(ne)), np.float64).ravel())
        b = data.get("bkg", np.zeros(ne))
        b = np.ascontiguousarray(np.asarray(
            np.full(ne, float(b)) if np.isscalar(b) else b, np.float64).ravel())
        keep = [mass, mom, ang, w, b]
        ptr = self._lib.cuda_load_data_flat_pwa_gen2(
            self._ctx,
            _ffi.cast("double*", _ffi.from_buffer(mass)), _ffi.NULL,
            _ffi.cast("double*", _ffi.from_buffer(mom)), _ffi.NULL,
            _ffi.cast("double*", _ffi.from_buffer(ang)), _ffi.NULL,
            _ffi.cast("double*", _ffi.from_buffer(w)),
            _ffi.cast("double*", _ffi.from_buffer(b)), ne)
        if not ptr:
            raise RuntimeError(
                "cuda_flat_pwa_gen2: load_data failed (device allocation)")
        h = FlatPwaGen2DataHandle(ptr, self._lib, ne)
        h._keep = keep          # keep the numpy buffers alive with the handle
        return h

    def tile_bytes(self, nres_t=None, collapsed=None):
        """The tile size `gp2_layout` gives — the kernel's own arithmetic.

        Exposed so a test can assert the buffer bookkeeping without a second
        copy of the formula in Python (one such copy already under-counted by
        256 B and turned a launch into a silent ``Q = 0``).
        """
        if nres_t is None:
            nres_t = 1 if self.n_res == 1 else self.n_res
        col = self.use_slots if collapsed is None else bool(collapsed)
        return int(self._lib.cuda_tile_bytes_flat_pwa_gen2(
            int(nres_t), 1 if col else 0, self.n_wave, self.n_proj,
            self.n_basis, self.n_slot))

    def free(self):
        """Context only.  Handles are owned by whoever called ``load_data``."""
        if getattr(self, "_ctx", None):
            self._lib.cuda_free_context_flat_pwa_gen2(self._ctx)
            self._ctx = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    # ── compute ─────────────────────────────────────────────────────
    def compute(self, params, data_handle, norm=None, return_p=True):
        ne = data_handle.ne
        hp = data_handle.ptr
        if hp is None:
            raise RuntimeError(
                "cuda_flat_pwa_gen2: this data handle was already freed")
        ck = np.asarray(params["ck"], np.complex128).ravel()
        coeff = np.asarray(params["coeff"], np.float64).ravel()
        if ck.size != self.n_wave:
            raise ValueError(
                f"cuda_flat_pwa_gen2: ck has {ck.size} entries, expected "
                f"n_wave = {self.n_wave}")
        if coeff.size != self.n_basis:
            raise ValueError(
                f"cuda_flat_pwa_gen2: coeff has {coeff.size} entries, "
                f"expected n_basis = {self.n_basis}")
        coeff = np.ascontiguousarray(coeff[self.basis_keep])
        ckr = np.ascontiguousarray(ck.real)
        cki = np.ascontiguousarray(ck.imag)

        P = np.zeros(ne) if return_p else None
        P_c = _ffi.cast("double*", _ffi.from_buffer(P)) if return_p \
            else _ffi.NULL
        gckr = np.zeros(self.n_wave)
        gcki = np.zeros(self.n_wave)
        gco_dd = np.zeros(self.n_basis)
        oQ = _ffi.new("double*")
        oDn = _ffi.new("double*")
        self._lib.cuda_compute_flat_pwa_gen2(
            self._ctx, hp,
            _ffi.cast("double*", _ffi.from_buffer(ckr)),
            _ffi.cast("double*", _ffi.from_buffer(cki)),
            _ffi.cast("double*", _ffi.from_buffer(coeff)),
            float(norm) if norm is not None else 0.0,
            1 if norm is not None else 0,
            oQ, oDn, P_c,
            _ffi.cast("double*", _ffi.from_buffer(gckr)),
            _ffi.cast("double*", _ffi.from_buffer(gcki)),
            _ffi.cast("double*", _ffi.from_buffer(gco_dd)))
        Q = float(oQ[0])

        gco_out = np.zeros(self.n_basis_full)
        gco_out[self.basis_keep] = gco_dd

        grads = {"ck": gckr + 1j * gcki, "coeff": gco_out}
        if norm is not None:
            self._last_dnorm = float(oDn[0])
            grads["norm"] = self._last_dnorm
        else:
            self._last_dnorm = None
        return Q, grads, (P if return_p else None)


@register_backend("cuda_flat_pwa_gen2", model="pwa_gen")
class CUDAFlatPwaGen2Backend(ComputeBackend):
    """Backend wrapper for :class:`CUDAKernelFlatPwaGen2`."""

    def __init__(self, kernel_config, batch_size=50000, resolution_size=1,
                 slot_collapse=None):
        self.kernel = CUDAKernelFlatPwaGen2(
            kernel_config, batch_size=batch_size,
            resolution_size=resolution_size, slot_collapse=slot_collapse)
        self.resolution_size = self.kernel.resolution_size

    def load_data(self, data_np):
        return self.kernel.load_data(data_np)

    def compute(self, params, data_handle, norm=None, return_p=True):
        return self.kernel.compute(params, data_handle, norm=norm,
                                   return_p=return_p)

    def free(self):
        if getattr(self, "kernel", None) is not None:
            self.kernel.free()
            self.kernel = None

    def __del__(self):
        self.free()
