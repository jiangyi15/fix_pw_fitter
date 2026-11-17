"""``cuda_flat_pwa_gen_1bw`` — the flat ``pwa_gen`` tile with ONE BW PER CHAIN.

Case-1 copy of :mod:`tabpwa.cuda._flat_pwa_gen2` (kernel policy §1): own
``.cu``, own ``.so``, own symbols, and the gen2 kernel/wrapper stay
byte-identical.  Following :mod:`tabpwa.cuda._v4_pwa_1bw`, it takes the **same
config input and conversion** as the general kernel and adds one requirement:

    n_res == 1     (top -> R + X: every chain carries exactly one BW factor)

``n_res`` is ``num_table.shape[1]``.  A multi-BW chain raises a ``ValueError``
naming ``cuda_flat_pwa_gen2`` — this kernel's whole point is that the single
factor needs no product bookkeeping, so it has no general path to fall back to.

What the specialisation buys
----------------------------

1. **One factor is the denominator.**  ``D = Π_l s_l = s_0``, so the per-factor
   ``s_s`` buffer, the ``l`` loop and the ``Π_{l'≠l} s`` product all vanish
   (they were already dead at the gen2 template's ``NR == 1`` instantiation).

2. **The ``n_slot`` collapse is KEPT — its ``w`` half is what dies, not the
   collapse.**  ``Π_{l'≠l} s`` is the empty product (identically 1) at one
   factor, so gen2's ``s_w`` buffer, ``slot_rep``-as-a-product-source and the
   per-``(e,slot,l)`` loop have no content — they are gone, and gen2's
   ``NR == 1`` instantiation never allocated ``s_w`` either.  But the OTHER
   half, ``dsum_{e,slot} = Σ_{k∈slot} dQ/dD_{e,k}``, is what reduces the
   ``dQ/dcoeff`` dot products from ``n_wave·run`` to ``n_slot·run``, and that
   has real content at one factor: measured on the benchmark shape (n_wave
   100, one slot) dropping it costs **1.16×** (4.64 → 5.36 ms) — more than the
   whole ``s_s`` saving bought.  So the gate
   (:func:`slot_threshold_says_use`), the ``s_dd`` buffer and the CSR list
   stay.  Only ``gamma_slot`` (a second copy of bits ``slot_rep`` already
   names) is dropped; the deep-factor (``n_res > 16``) fallback is
   unreachable at one factor and is dropped too.  ``slot_collapse`` stays a
   kwarg for the same reason gen2 has it — the test hook that forces either
   path.

3. **Sparse gamma rows — the new win of this kernel.**  A ``(wave, factor)``
   gamma row selects a handful of ``basis_table`` rows out of ``n_basis``.  The
   host emits the row's **true non-zeros** as a CSR (``gs_lo[k]``/``gs_hi[k]``
   index the flat ``nz_wave_m`` list) at CREATE time, so phase 1 iterates
   exactly ``nnz(k)`` columns instead of ``n_basis`` — and phase 5 walks a
   host-built ``(group, m)`` list of the same entries (deduped across a slot's
   waves when the collapse runs, so the two savings multiply).

   The list does **not** require contiguity.  A row may have interior holes;
   they are simply absent from the list, so no ``g == 0`` test is needed and a
   holed row costs its non-zeros rather than its span.  (An earlier form
   demanded contiguity and refused a holed row, claiming the kernel "would use
   basis columns the factor does not select" — false, since ``0 * value == 0``;
   it only forfeited this optimisation on exactly the configs that need it.
   The real ``jpsi_eta_pi_pi`` config has holes in 23 of 26 waves: 92
   iterations through the list, 290 over the span, 1794 over every column.)

   Basis columns iterated per event (``ref`` = full-``m`` loop, ``run`` = this
   kernel), measured on the six fixtures (all dense WITHIN their span, so the
   CSR and the span agree there)::

       config                    n_basis  n_res   ref  run  ratio
       config_pwa_gen.yml            3      1       3    3   1.00  (dense)
       config_pwa_gen_km.yml         5      1      10   10   1.00  (dense)
       config_pwa_gen_ident.yml     10      1      40   20   2.00
       config_pwa_gen_same.yml      10      1      40   20   2.00
       config_pwa_gen_cp.yml        20      1     160   40   4.00
       config_pwa_gen_mb.yml         8      2      64   32   2.00  (REJECTED: n_res=2)

   ``run <= n_basis`` always, so this is never a loss; it is a win exactly
   where the matrix is sparse, which is why ``_cp`` (20 basis rows over 4
   waves of 5) is the shape that shows it and the dense benchmark shape does
   not.  ``num_table`` needs no equivalent: its loop is over ``r`` in
   ``[0, n_res)`` — one iteration here — and it runs once per ``load_data``.

**The failure mode this optimisation invites, and what pins it.**  A list that
dropped a NON-zero column would silently change the denominator — a wrong
answer with no error anywhere.  :func:`index_runs` is the one implementation,
and ``tests/test_cuda_flat_pwa_gen_1bw.py`` pins it three ways: the CSR content
against each ``gamma`` row, a holed row against ``cuda_flat_pwa_gen2``'s
full-column walk on the SAME holed config (and asserts the hole really moved
``Q``, so the comparison is not vacuous), and ``dcoeff`` against central
differences for both phase-5 paths.  An all-zero row still raises: it would
make ``D == 0``.

Host contract, identical to the other flat backends::

    params = {"ck": complex (n_wave,), "coeff": real (n_basis,)}
    grads  = {"ck", "coeff", "norm"}      # norm only on a normed call
"""
import numpy as np

from tabpwa.backends.core import ComputeBackend, register_backend

_ffi = None
_lib = None

#: The slot-collapse gate, unchanged from gen2: enable only when the per-slot
#: work is strictly less than the per-wave work it replaces.
#:
#: MEASURED at `n_res == 1` on the benchmark shape (n_wave=100, 65k events,
#: interleaved min-of-9): collapse ON 4.64 ms vs OFF 5.36 ms = **1.16×**.  The
#: `w` product that the gate also protects in gen2 does not exist here (empty
#: product), so what the gate buys at one factor is purely the
#: `n_wave·run -> n_slot·run` dot-product reduction.  At `n_slot == n_wave`
#: that reduction is break-even (the `dsum` accumulation is paid and folds one
#: wave), so the gate stays at `2·n_slot <= n_wave`.
SLOT_GATE_RATIO = 2


def slot_threshold_says_use(n_slot, n_wave, ratio=SLOT_GATE_RATIO):
    """The gate, in one place so a test can ask without a kernel."""
    return int(n_slot) * int(ratio) <= int(n_wave)


def slot_map(gamma_matrix):
    """Distinct ``gamma_matrix[k]`` rows → slot maps and a CSR wave list.

    Two waves in one slot have bit-identical ``s_{e,k,0}`` and ``D_{e,k}`` by
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



def index_runs(gamma_matrix, n_wave=None, n_basis=None):
    """Per-wave CSR of a ``n_res == 1`` gamma's TRUE non-zeros.

    Returns ``(gs_lo, gs_hi, nz_k, nz_m)``: two ``(n_wave,)`` int32 arrays and
    the flat ``(n_nz,)`` list of ``(wave, column)`` pairs.  ``[gs_lo[k],
    gs_hi[k])`` indexes ``nz_m``; zeros are never included.

    **The run does NOT have to be contiguous.**  It is emitted as a CSR over
    the row's true non-zeros — ``[gs_lo[k], gs_hi[k])`` indexes ``nz_m`` — so
    a row with interior holes costs exactly its non-zeros instead of the span.
    A hole is harmless either way: ``0 * value == 0``, and the kernel drops the
    per-element ``g == 0`` test because the list contains no zeros.

    (The earlier form required contiguity and raised otherwise, claiming a hole
    "would make it use basis columns the factor does not select" — that was
    false: `[lo,hi)+skip`, an explicit index list and the full walk all give the
    identical value.  The restriction only forfeited the optimisation on any
    config whose basis is folded — e.g. 290 vs 92 iterations on the real one.)

    ``n_res > 1`` is rejected here too (the wrapper checks it earlier, this is
    the belt-and-braces form that the kernel's own create also pins).
    """
    g = np.ascontiguousarray(np.asarray(gamma_matrix, np.float64))
    if g.ndim == 3:
        if g.shape[1] != 1:
            raise ValueError(
                "cuda_flat_pwa_gen_1bw needs exactly ONE BW per chain "
                f"(top -> R + X), got n_res = {g.shape[1]} — use "
                "cuda_flat_pwa_gen2 for multi-BW chains")
        g = g[:, 0, :]
    if g.ndim != 2:
        raise ValueError(
            f"cuda_flat_pwa_gen_1bw: gamma_matrix must be (n_wave, n_basis) "
            f"at n_res == 1, got shape {g.shape}")
    nw, nb = int(g.shape[0]), int(g.shape[1])
    if ((n_wave is not None and int(n_wave) != nw)
            or (n_basis is not None and int(n_basis) != nb)):
        raise ValueError(
            f"cuda_flat_pwa_gen_1bw: gamma_matrix has shape {g.shape}, "
            f"expected ({nw if n_wave is None else int(n_wave)}, "
            f"{nb if n_basis is None else int(n_basis)})")

    gs_lo = np.zeros(nw, np.int32)      # CSR into nz_m (true non-zeros)
    gs_hi = np.zeros(nw, np.int32)
    nz_k, nz_m = [], []
    for k in range(nw):
        idx = np.nonzero(g[k])[0]
        if idx.size == 0:
            raise ValueError(
                f"cuda_flat_pwa_gen_1bw: wave {k} selects no basis column "
                f"(gamma row is all zero) — the denominator would be 0")
        gs_lo[k], gs_hi[k] = len(nz_m), len(nz_m) + idx.size
        nz_k.extend([k] * idx.size)
        nz_m.extend(int(m) for m in idx)
    return (gs_lo, gs_hi,
            np.asarray(nz_k, np.int32), np.asarray(nz_m, np.int32))


def run_entries(gs_lo, gs_hi, nz_m, group_of_wave):
    """The flat ``(group, column)`` list of run entries for phase 5.

    ``group_of_wave`` maps a wave onto its phase-5 group axis — the wave
    itself for the per-wave path, its SLOT for the collapsed one.  Entries of
    one group are deduplicated (a slot's waves share their gamma row, so the
    same column would otherwise be walked once per wave it holds).
    """
    gk, gm = [], []
    seen = set()
    for k in range(len(gs_lo)):
        g = int(group_of_wave[k])
        for i in range(int(gs_lo[k]), int(gs_hi[k])):
            m = int(nz_m[i])
            key = (g, m)
            if key in seen:
                continue
            seen.add(key)
            gk.append(g)
            gm.append(m)
    return np.asarray(gk, np.int32), np.asarray(gm, np.int32)


def _load_lib():
    from cffi import FFI
    from tabpwa.cuda.build import ensure
    global _ffi
    if _ffi is None:
        _ffi = FFI()
        _ffi.cdef("""
void* cuda_create_context_flat_pwa_gen_1bw(
    const double*,int, const int*,int,
    const double*,const double*,int, const int*,int,
    const double*,const double*,int, const int*,int,
    const double*,int,
    const int*,int, const int*,int,
    const int*,int, const int*,int, const int*,int,
    int,int,
    const int*,int, const int*,int, const int*,int, const int*,int,
    const int*,int, const double*,int, const double*,int,
    const double*,const double*,int,
    double,double,int, double,double,int,
    int,int,int,int,int, int,int,int,int,int, int,int);
void* cuda_load_data_flat_pwa_gen_1bw(void*,const double*,const int*,
    const double*,const int*, const double*,const int*,
    const double*,const double*,int);
void cuda_free_data_flat_pwa_gen_1bw(void*);
void cuda_free_context_flat_pwa_gen_1bw(void*);
void cuda_compute_flat_pwa_gen_1bw(void*,void*,const double*,const double*,
    const double*,double,int, double*,double*,double*,double*,double*,double*);
int cuda_get_device_count_flat_pwa_gen_1bw(void);
long long cuda_tile_bytes_flat_pwa_gen_1bw(int,int,int,int,int);
long long cuda_handle_dA_bytes_flat_pwa_gen_1bw(void*);
int cuda_smem_bytes_flat_pwa_gen_1bw(void*);
void cuda_get_device_name_flat_pwa_gen_1bw(char*,int);
""")
    lib_path = ensure("kernels_flat_pwa_gen_1bw.cu",
                      "libcuda_kernels_flat_pwa_gen_1bw.so")
    if not lib_path:
        raise RuntimeError("cuda_flat_pwa_gen_1bw: library not found")
    return _ffi.dlopen(lib_path)


class FlatPwaGen1BWDataHandle:
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
        self.dA_bytes = int(lib.cuda_handle_dA_bytes_flat_pwa_gen_1bw(ptr))

    def free(self):
        """Idempotent: the second call (and a NULL handle) is a no-op."""
        if self.ptr is not None:
            self._lib.cuda_free_data_flat_pwa_gen_1bw(self.ptr)
            self.ptr = None
            self._keep.clear()

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class CUDAKernelFlatPwaGen1BW:
    """Fused-tile CUDA implementation of the flat ``pwa_gen`` config, n_res==1."""

    def __init__(self, config, batch_size=50000, resolution_size=1,
                 slot_collapse=None):
        global _lib
        if "fl_index" in config or "num_index" in config:
            raise ValueError(
                "cuda_flat_pwa_gen_1bw: pass the FULL (non-deduped) config — "
                "this backend reads every row where the builder emits it, and "
                "a config that already carries fl_index/num_index has a "
                "different row layout")
        resolution_size = int(resolution_size)
        if resolution_size < 1:
            raise ValueError(
                "cuda_flat_pwa_gen_1bw: resolution_size must be >= 1, got "
                f"{resolution_size}")
        if _lib is None:
            _lib = _load_lib()
        self._lib = _lib
        self.config = config
        if _lib.cuda_get_device_count_flat_pwa_gen_1bw() <= 0:
            raise RuntimeError("cuda_flat_pwa_gen_1bw: no CUDA devices")
        buf = _ffi.new("char[256]")
        _lib.cuda_get_device_name_flat_pwa_gen_1bw(buf, 256)
        self.device_name = _ffi.string(buf).decode()

        c = config
        self.n_wave = int(np.asarray(c["fl_table"]).shape[0])
        # n_proj is encoded p-major in matrix_angle's columns
        self.n_proj = max(1, int(np.asarray(c["matrix_angle"]).shape[1])
                          // self.n_wave)
        self.n_res = int(np.asarray(c["num_table"]).shape[1])
        if self.n_res != 1:
            raise ValueError(
                "cuda_flat_pwa_gen_1bw needs exactly ONE BW per chain "
                f"(top -> R + X), got n_res = {self.n_res} — use "
                "cuda_flat_pwa_gen2 for multi-BW chains")
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

        # ── the per-wave index list, built here (see index_runs) ──────────
        gam_full = np.asarray(c["gamma_matrix"], np.float64)
        self.gs_lo, self.gs_hi, _nk, _nm = index_runs(
            gam_full, self.n_wave, self.n_basis)
        self.nz_m = _nm            # the true non-zeros, CSR-indexed by gs_*
        # the per-wave form is what the kernel uses when the collapse is off
        self.nz_wave_k, self.nz_wave_m = run_entries(
            self.gs_lo, self.gs_hi, self.nz_m, np.arange(self.n_wave))
        self.n_nz_wave = int(self.nz_wave_k.size)

        # ── the slot collapse and its gate (kept: see the module doc) ──
        (self.slot_of_wave, self.slot_rep, _rows,
         self.slot_ptr, self.slot_waves) = slot_map(gam_full)
        self.n_slot = int(self.slot_rep.size)
        auto = slot_threshold_says_use(self.n_slot, self.n_wave)
        self.use_slots = auto if slot_collapse is None else bool(slot_collapse)
        self.slot_collapse_auto = bool(auto)
        self.nz_slot_k, self.nz_slot_m = run_entries(
            self.gs_lo, self.gs_hi, self.nz_m, self.slot_of_wave)
        self.n_nz_slot = int(self.nz_slot_k.size)
        #: `(ref, run)` basis iterations per event, per path — the measured win
        self.n_nz = (self.n_nz_slot if self.use_slots else self.n_nz_wave)

        gam = np.ascontiguousarray(
            gam_full.reshape(self.n_wave * self.n_res, self.n_basis))

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
        self._ctx = _lib.cuda_create_context_flat_pwa_gen_1bw(
            D(fl_tab), fl_tab.size, I(fl_q), fl_q.size,
            D(num.real), D(num.imag), num.size, I(num_m), num_m.size,
            D(bas.real), D(bas.imag), bas.size, I(bas_m), self.n_basis,
            D(gam), int(gam.size),
            I(self.gs_lo), int(self.gs_lo.size),
            I(self.gs_hi), int(self.gs_hi.size),
            I(self.slot_rep), int(self.slot_rep.size),
            I(self.slot_ptr), int(self.slot_ptr.size),
            I(self.slot_waves), int(self.slot_waves.size),
            self.n_slot, 1 if self.use_slots else 0,
            I(self.nz_wave_k), int(self.nz_wave_k.size),
            I(self.nz_wave_m), int(self.nz_wave_m.size),
            I(self.nz_slot_k), int(self.nz_slot_k.size),
            I(self.nz_slot_m), int(self.nz_slot_m.size),
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
            raise RuntimeError(
                "cuda_flat_pwa_gen_1bw: context creation failed")
        self._keep = keep_alive
        # the ONE definition of the tile layout lives in `gp1b_layout` (C);
        # asking the context avoids a second copy drifting against it
        self.smem_bytes = int(
            _lib.cuda_smem_bytes_flat_pwa_gen_1bw(self._ctx))

    # ── the run, for reporting/tests ────────────────────────────────
    @property
    def run_widths(self):
        """Non-zeros selected by each wave (``<= n_basis``)."""
        return (self.gs_hi - self.gs_lo).astype(np.int64)

    def basis_iters(self):
        """``(ref, run)`` basis-row iterations per event — the measured win."""
        return (self.n_wave * self.n_basis, int(self.n_nz))

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
        ptr = self._lib.cuda_load_data_flat_pwa_gen_1bw(
            self._ctx,
            _ffi.cast("double*", _ffi.from_buffer(mass)), _ffi.NULL,
            _ffi.cast("double*", _ffi.from_buffer(mom)), _ffi.NULL,
            _ffi.cast("double*", _ffi.from_buffer(ang)), _ffi.NULL,
            _ffi.cast("double*", _ffi.from_buffer(w)),
            _ffi.cast("double*", _ffi.from_buffer(b)), ne)
        if not ptr:
            raise RuntimeError(
                "cuda_flat_pwa_gen_1bw: load_data failed (device allocation)")
        h = FlatPwaGen1BWDataHandle(ptr, self._lib, ne)
        h._keep = keep          # keep the numpy buffers alive with the handle
        return h

    def tile_bytes(self, collapsed=None, n_wave=None, n_proj=None,
                   n_basis=None, n_slot=None):
        """The tile size `gp1b_layout` gives — the kernel's own arithmetic.

        Exposed so a test can assert the buffer bookkeeping without a second
        copy of the formula in Python (one such copy already under-counted by
        256 B and turned a launch into a silent ``Q = 0``).
        """
        col = self.use_slots if collapsed is None else bool(collapsed)
        return int(self._lib.cuda_tile_bytes_flat_pwa_gen_1bw(
            1 if col else 0,
            self.n_slot if n_slot is None else int(n_slot),
            self.n_wave if n_wave is None else int(n_wave),
            self.n_proj if n_proj is None else int(n_proj),
            self.n_basis if n_basis is None else int(n_basis)))

    def free(self):
        """Context only.  Handles are owned by whoever called ``load_data``."""
        if getattr(self, "_ctx", None):
            self._lib.cuda_free_context_flat_pwa_gen_1bw(self._ctx)
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
                "cuda_flat_pwa_gen_1bw: this data handle was already freed")
        ck = np.asarray(params["ck"], np.complex128).ravel()
        coeff = np.asarray(params["coeff"], np.float64).ravel()
        if ck.size != self.n_wave:
            raise ValueError(
                f"cuda_flat_pwa_gen_1bw: ck has {ck.size} entries, expected "
                f"n_wave = {self.n_wave}")
        if coeff.size != self.n_basis:
            raise ValueError(
                f"cuda_flat_pwa_gen_1bw: coeff has {coeff.size} entries, "
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
        self._lib.cuda_compute_flat_pwa_gen_1bw(
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


@register_backend("cuda_flat_pwa_gen_1bw", model="pwa_gen")
class CUDAFlatPwaGen1BWBackend(ComputeBackend):
    """Backend wrapper for :class:`CUDAKernelFlatPwaGen1BW`."""

    def __init__(self, kernel_config, batch_size=50000, resolution_size=1,
                 slot_collapse=None):
        self.kernel = CUDAKernelFlatPwaGen1BW(
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
