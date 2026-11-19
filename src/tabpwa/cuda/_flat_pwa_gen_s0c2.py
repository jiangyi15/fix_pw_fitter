"""``cuda_flat_pwa_gen_s0c2`` — ``cuda_flat_pwa_gen_s0c`` with the three
per-chunk feature reductions fused into ONE launch.

Case-1 copy (kernel policy §1): own module, own ``.cu``, own ``.so``, own
symbols, and ``kernels_flat_pwa_gen_s0c.cu`` / ``_flat_pwa_gen_s0c.py`` /
every other ``s0*`` module stay byte-identical.

The one change (and the measurement behind it)
----------------------------------------------

On ``tests/config_pwa_gen_km.yml`` at ne = 4000, s0c spends 16.25 us on each of
THREE ``reduce_sum_features_tree_kernel`` launches that move a handful of
doubles (grid 4, 128 threads, one atomicAdd per feature for ~26/64 values) —
48.7 us of a ~234 us iteration, i.e. ~21 % of the profile.  MEASURED end to end
(``tests/benchmark_s0c_variants.py``, idle card, min of 5 interleaved rounds)
the real saving is **8.4 us / 6.9 %** at ne 4000 (s0c 129.5 us -> 121.1 us): two
launches at ~4.2 us each on the wall clock, so ncu's per-launch 16.25 us was
profiler-inflated.  At ne 20000/50000 it is 0.1-1.6 % (the amp kernel
dominates); on the production shape (``n_basis`` 64 > 32) the acc fallback
applies and the gain is 0.0 % by design.

``gsc2_launch_reduce_features3`` replaces the three with ONE launch whose
``blockIdx.y`` selects the buffer: each y-slice runs the UNCHANGED tree body
(same block shape, chunk, shared layout, fold order, per-feature atomic) and
the three slices write disjoint output ranges, so the result is **bit-identical**
(pinned by ``test_bitwise_vs_s0c`` on Q/P/dck/dcoeff/dnorm).  ``Q``'s own
reduce is deliberately NOT folded in: its reduction order differs, so folding
it would move Q's last bits.

The three ``s0c`` docs below (why s0c's compute, the memory design, the index
structures) are inherited unchanged — this module is s0c's, symbol for symbol.

Why s0c's compute rather than a new one
---------------------------------------

``s0c`` was designed against a *uniform coefficient basis*: its denominator is
already

    BW_w = Σ_i M_wi·F_i,      dQ/dM_wi = 2·Re(dQ/dBW_w · F_i)

which **is** the flat config's general basis sum
``D_e,w = Σ_m gamma[w,m]·coeff[m]·basis_m(m_e)``.  ``s0c`` is not specialised
to ``[1, -m², -iG]`` — that is only what *its* wrapper fills.  So this backend
is s0c's kernel with the fill replaced, not a new kernel.

What that does and does not buy (measured on the real ``jpsi_eta_pi_pi``
config)::

    per event        s0c (plain pwa)     flat pwa_gen
    interps              44                 43          <- a wash
    matvec terms        520               92 sparse      <- 5.7x cheaper
                                         1794 dense       <- 3.5x dearer

The interpolation count is a wash and *that* is what dominates s0c's memory and
its per-iteration cost; the matvec is only cheaper with the CSR, so the CSR is
**mandatory** here, not an optimisation.

The memory property inherited from s0c
--------------------------------------

``s0c`` stores **no per-event interpolation cache**: the handle holds ``mass``,
``weight``, ``bkg`` and the deduplicated numerator cache, and the basis
interpolation is recomputed per event per iteration into shared memory.  That
is why s0c is 2.87 GB resident where the ``s0e`` tile is 5.91 GB, and it is
the property this backend keeps — see the module ``__doc__`` of the ``.cu``.

The three index structures, all reused (never re-implemented here)
-----------------------------------------------------------------

* ``uslot_map`` (:mod:`tabpwa.cuda._flat_pwa_gen_s0e`) — the basis
  interpolation dedup on ``(row bytes, mass column)``: **69 -> 43** on the real
  config.  The key is deliberately NOT ``basis_kind`` (a factor-local slot
  index, which would merge every resonance's running width into one slot).
* ``amp_slot_map`` (:mod:`tabpwa.cuda._flat_pwa_gen_s0e`) — the ck/coeff-free
  numerator ``Amp = fa·fl·num`` dedup, keyed on every ``matrix_angle`` column
  of the wave (p-major), the ``fl`` row + column and the ``num`` row + column:
  **26 -> 13** waves.
* ``index_runs`` (:mod:`tabpwa.cuda._flat_pwa_gen_1bw`) — the per-wave CSR of
  the gamma's TRUE non-zeros: **92** entries against 1794 columns on the real
  config, with interior holes in 23 of 26 waves.  A hole is simply absent, so
  there is no ``g == 0`` test in the kernel to get wrong.

On top of those, this module builds the transposed CSR ``(basis row -> waves
that select it)`` which is what makes the coefficient gradient m-major — see
``_basis_csr``.

Contract
--------

``n_res == 1`` only; a multi-BW chain raises a ``ValueError`` naming
``cuda_flat_pwa_gen2`` (the same contract as ``cuda_flat_pwa_gen_1bw`` /
``cuda_flat_pwa_gen_s0e``).  Host side, identical to the other flat backends::

    params = {"ck": complex (n_wave,), "coeff": real (n_basis,)}
    grads  = {"ck", "coeff", "norm"}      # norm only on a normed call

``grad_ck`` is the **raw Wirtinger half** (``CKProduct.backprop_grad`` applies
the 2·Re convention downstream).  Analytic gradients only.
"""
import numpy as np

from tabpwa.backends.core import ComputeBackend, register_backend

#: the two structural dedups and the sparse-row builder have ONE
#: implementation each; importing them here is what keeps this kernel's
#: index layout from drifting against its siblings.
from tabpwa.cuda._flat_pwa_gen_1bw import index_runs
from tabpwa.cuda._flat_pwa_gen_s0e import amp_slot_map, uslot_map

_ffi = None
_lib = None


def _basis_csr(gamma_matrix, gs_lo, gs_hi, nz_k, nz_m, n_wave, n_basis):
    """``(m_ptr, m_waves, m_cols, order)`` — the transposed, row-major CSR.

    The coefficient gradient is m-major: ``Z_m`` depends on the basis row only,
    so ``Σ_{k∋m} dQ/dD_{e,k}`` can be summed FIRST and multiplied by ``Z_m``
    once.  Without it the per-event scratch would be ``n_wave·n_basis``
    (= 1794 doubles at the real config's shape, and 100·69 on the benchmark
    shape) instead of ``n_basis``.

    ``[m_ptr[m], m_ptr[m+1])`` indexes ``m_waves`` (the waves selecting basis
    row ``m``) and, in the same order, ``m_cols`` (which IS the row: every
    entry of one slot is the row).  ``order`` maps each entry back onto its
    index in the wave-major ``nz_m`` list, so a caller can pair it with
    ``nz_u``.

    Exactness requires the selector to be 0/1: with a weight ``g`` the term
    would be ``Σ_k g_k·(…·Z)`` and factoring ``Z`` out would be wrong.  The
    builder emits exactly 0/1 and this raises otherwise rather than silently
    changing the answer.
    """
    g = np.ascontiguousarray(np.asarray(gamma_matrix, np.float64))
    if g.ndim == 3:
        g = g[:, 0, :]
    nk = np.asarray(nz_k, np.int64)
    nm = np.asarray(nz_m, np.int64)
    vals = g[nk, nm]
    bad = (vals != 0.0) & (vals != 1.0)
    if np.any(bad):
        raise ValueError(
            "cuda_flat_pwa_gen_s0c2: gamma_matrix must be a 0/1 selector — "
            f"wave {int(nk[bad][0])} carries the weight "
            f"{float(vals[bad][0])!r}, and the m-major coefficient gradient "
            "factors the basis row out of the wave sum, which is exact only "
            "for 0/1")
    counts = np.bincount(nm, minlength=int(n_basis))
    m_ptr = np.zeros(int(n_basis) + 1, np.int32)
    np.cumsum(counts, out=m_ptr[1:])
    order = np.argsort(nm, kind="stable")
    return (m_ptr,
            np.ascontiguousarray(nk[order].astype(np.int32)),
            np.ascontiguousarray(nm[order].astype(np.int32)),
            order)


def _load_lib():
    from cffi import FFI
    from tabpwa.cuda.build import ensure
    global _ffi
    if _ffi is None:
        _ffi = FFI()
        _ffi.cdef("""
void* cuda_create_context_flat_pwa_gen_s0c2(
    const double*,int, const int*,int,
    const double*,const double*,int, const int*,int,
    const double*,const double*,int,
    const int*,int, const int*,int,
    const int*,int,
    const int*,int, const int*,int, const int*,int,
    const int*,int, const int*,int,
    const int*,int, const int*,int,
    const int*,int, const int*,int,
    int,
    const int*,int,
    const double*,int, const double*,int,
    const double*,const double*,int,
    double,double,int, double,double,int,
    int,int,int,int,int, int,int,int,int,int, int,int);
void* cuda_load_data_flat_pwa_gen_s0c2(void*,const double*,const double*,
    const double*,const double*,const double*,int);
void cuda_free_data_flat_pwa_gen_s0c2(void*);
void cuda_free_context_flat_pwa_gen_s0c2(void*);
void cuda_compute_flat_pwa_gen_s0c2(void*,void*,const double*,const double*,
    const double*,double,int, double*,double*,double*,double*,double*,double*);
int cuda_get_device_count_flat_pwa_gen_s0c2(void);
int cuda_smem_bytes_flat_pwa_gen_s0c2(void*);
int cuda_cap_flat_pwa_gen_s0c2(void*);
int cuda_ngroups_flat_pwa_gen_s0c2(void*);
void cuda_get_device_name_flat_pwa_gen_s0c2(char*,int);
/* the fusion's diagnostics (fixture / benchmark only) */
int cuda_feature_launch_count_flat_pwa_gen_s0c2(void);
void cuda_reset_feature_launch_count_flat_pwa_gen_s0c2(void);
int cuda_amp_threads_flat_pwa_gen_s0c2(void*);
int cuda_stats_flat_pwa_gen_s0c2(void*, int*);
int cuda_check_fused_reduce_flat_pwa_gen_s0c2(const double*,int,
    const double*,int, const double*,int, int, int*);
""")
    lib_path = ensure("kernels_flat_pwa_gen_s0c2.cu",
                      "libcuda_kernels_flat_pwa_gen_s0c2.so")
    if not lib_path:
        raise RuntimeError("cuda_flat_pwa_gen_s0c2: library not found")
    return _ffi.dlopen(lib_path)


class FlatPwaGenS0C2DataHandle:
    """One loaded sample — a per-load C handle, never the context.

    The fitter holds the phase-space handle and the data handle at the same
    time (``_compute_norm_batched`` integrates over one while ``get_nll_raw``
    evaluates the other), so each ``load_data`` owns its arrays.

    The handle holds per-EVENT data only — ``mass``, ``weight``, ``bkg`` and
    the numerator cache ``amp`` — plus the pinned staging for the per-event
    ``P`` output.  There is deliberately NO interpolation cache: that is the
    memory property inherited from ``cuda_v5_pwa_s0c`` (2.87 GB vs the s0e
    tile's 5.91 GB on the real sample).  Every working array is the CONTEXT's,
    sized by its chunk length ``cap``.
    """

    __slots__ = ("ptr", "ne", "_lib", "_keep", "__weakref__")

    def __init__(self, ptr, lib, ne):
        self.ptr = ptr
        self._lib = lib
        self.ne = int(ne)
        self._keep = []

    def free(self):
        """Idempotent: the second call (and a NULL handle) is a no-op."""
        if self.ptr is not None:
            self._lib.cuda_free_data_flat_pwa_gen_s0c2(self.ptr)
            self.ptr = None
            self._keep.clear()

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class CUDAKernelFlatPwaGenS0C2:
    """s0c's compute over the flat ``pwa_gen`` config (``n_res == 1``)."""

    def __init__(self, config, batch_size=50000, resolution_size=1):
        global _lib
        if "fl_index" in config or "num_index" in config:
            raise ValueError(
                "cuda_flat_pwa_gen_s0c2: pass the FULL (non-deduped) config — "
                "this backend reads every row where the builder emits it, and "
                "a config that already carries fl_index/num_index has a "
                "different row layout")
        resolution_size = int(resolution_size)
        if resolution_size < 1:
            raise ValueError(
                "cuda_flat_pwa_gen_s0c2: resolution_size must be >= 1, got "
                f"{resolution_size}")
        if _lib is None:
            _lib = _load_lib()
        self._lib = _lib
        self.config = config
        if _lib.cuda_get_device_count_flat_pwa_gen_s0c2() <= 0:
            raise RuntimeError("cuda_flat_pwa_gen_s0c2: no CUDA devices")
        buf = _ffi.new("char[256]")
        _lib.cuda_get_device_name_flat_pwa_gen_s0c2(buf, 256)
        self.device_name = _ffi.string(buf).decode()

        c = config
        #: every count from a shape — the flat config stores none
        self.n_wave = int(np.asarray(c["fl_table"]).shape[0])
        self.n_proj = max(1, int(np.asarray(c["matrix_angle"]).shape[1])
                          // self.n_wave)
        self.n_res = int(np.asarray(c["num_table"]).shape[1])
        if self.n_res != 1:
            raise ValueError(
                "cuda_flat_pwa_gen_s0c2 needs exactly ONE BW per chain "
                f"(top -> R + X), got n_res = {self.n_res} — use "
                "cuda_flat_pwa_gen2 for multi-BW chains")
        self.n_decay = int(np.asarray(c["fl_q_index"]).shape[1])
        self.n_ab = int(np.asarray(c["angle_k"]).shape[0])
        self.n_comp = int(np.asarray(c["angle_k"]).shape[1])
        self.n_mass = int(np.max(c["num_m_index"])) + 1
        self.n_momentum = int(np.max(c["fl_q_index"])) + 1
        self.n_angle_total = int(np.max(c["angle_index"])) + 1
        bas = np.ascontiguousarray(np.asarray(c["basis_table"], np.complex128))
        self.n_basis = int(bas.shape[0])
        self.batch_size = int(batch_size)
        self.resolution_size = resolution_size
        self._last_dnorm = None

        # ── the basis-interpolation dedup (69 -> 43 on the real config) ───
        (self.uslot_row, self.uslot_col,
         self.slot_of_basis) = uslot_map(bas, c["basis_m_index"])
        self.n_uslot = int(self.uslot_row.size)

        # ── the numerator dedup (26 -> 13 waves on the real config) ───────
        self.slot_of_wave, self.rep_of_slot = amp_slot_map(
            c["matrix_angle"], c["fl_table"], c["fl_q_index"],
            c["num_table"], c["num_m_index"], self.n_wave, self.n_proj)
        self.n_uniq = int(self.rep_of_slot.size)

        # ── the per-wave CSR of the gamma's TRUE non-zeros ────────────────
        self.gs_lo, self.gs_hi, _nz_k, nz_m = index_runs(
            np.asarray(c["gamma_matrix"], np.float64),
            self.n_wave, self.n_basis)
        self.nz_m = nz_m
        self.n_nz = int(nz_m.size)
        #: the uslot of each run entry — a basis row's slot is fixed, so this
        #: is a gather, not a second dedup
        self.nz_u = np.ascontiguousarray(
            self.slot_of_basis[self.nz_m], np.int32)

        # ── the transposed CSR: basis row -> the waves that select it ─────
        self.m_ptr, self.m_waves, self.m_cols, self._m_order = _basis_csr(
            c["gamma_matrix"], self.gs_lo, self.gs_hi, _nz_k, self.nz_m,
            self.n_wave, self.n_basis)
        self.n_m_entry = int(self.m_waves.size)

        keep_alive = []
        fl_tab = np.ascontiguousarray(np.asarray(c["fl_table"], np.float64)
                                      .reshape(self.n_wave * self.n_decay, -1))
        num = np.ascontiguousarray(np.asarray(c["num_table"], np.complex128)
                                   .reshape(self.n_wave * self.n_res, -1))
        fl_q = np.ascontiguousarray(np.asarray(c["fl_q_index"], np.int32))
        num_m = np.ascontiguousarray(np.asarray(c["num_m_index"], np.int32))
        am = np.ascontiguousarray(np.asarray(c["matrix_angle"], np.complex128))
        bas_m = np.ascontiguousarray(np.asarray(c["basis_m_index"], np.int32))

        def D(a):
            arr = np.ascontiguousarray(a, np.float64)
            keep_alive.append(arr)
            return _ffi.cast("double*", _ffi.from_buffer(arr))

        def I(a):
            arr = np.ascontiguousarray(a, np.int32)
            keep_alive.append(arr)
            return _ffi.cast("int*", _ffi.from_buffer(arr))

        self._ctx = _lib.cuda_create_context_flat_pwa_gen_s0c2(
            D(fl_tab), int(fl_tab.size), I(fl_q), int(fl_q.size),
            D(num.real), D(num.imag), int(num.size), I(num_m), int(num_m.size),
            D(bas.real), D(bas.imag), int(bas.size),
            I(self.uslot_row), int(self.uslot_row.size),
            I(self.uslot_col), int(self.uslot_col.size),
            I(self.slot_of_basis), int(self.slot_of_basis.size),
            I(_nz_k), int(_nz_k.size),
            I(self.nz_m), int(self.nz_m.size),
            I(self.nz_u), int(self.nz_u.size),
            I(self.gs_lo), int(self.gs_lo.size),
            I(self.gs_hi), int(self.gs_hi.size),
            I(self.m_ptr), int(self.m_ptr.size),
            I(self.m_waves), int(self.m_waves.size),
            I(self.slot_of_wave), int(self.slot_of_wave.size),
            I(self.rep_of_slot), int(self.rep_of_slot.size),
            self.n_uniq,
            I(np.asarray(c["angle_index"], np.int32)),
            int(np.asarray(c["angle_index"]).size),
            D(np.asarray(c["angle_k"], np.float64)),
            int(np.asarray(c["angle_k"]).size),
            D(np.asarray(c["angle_b"], np.float64)),
            int(np.asarray(c["angle_b"]).size),
            D(am.real), D(am.imag), int(am.size),
            float(c["m_min"]), float(c["m_delta"]), int(bas.shape[1]),
            float(c["q_min"]), float(c["q_delta"]), int(fl_tab.shape[1]),
            self.n_wave, self.n_proj, self.n_res, self.n_decay, self.n_basis,
            self.n_ab, self.n_comp, self.n_mass, self.n_momentum,
            self.n_angle_total, self.batch_size, self.resolution_size,
            )
        if not self._ctx:
            raise RuntimeError(
                "cuda_flat_pwa_gen_s0c2: context creation failed")
        self._keep = keep_alive
        # the ONE definition of each lives in C; asking the context is what
        # stops a Python copy from drifting against it
        self.smem_bytes = int(
            _lib.cuda_smem_bytes_flat_pwa_gen_s0c2(self._ctx))
        self.cap = int(_lib.cuda_cap_flat_pwa_gen_s0c2(self._ctx))
        self.ngroups = int(_lib.cuda_ngroups_flat_pwa_gen_s0c2(self._ctx))

    # ── reporting, for the fixture ──────────────────────────────────────
    @property
    def run_widths(self):
        """Gamma non-zeros selected by each wave (``<= n_basis``)."""
        return (self.gs_hi - self.gs_lo).astype(np.int64)

    def gamma_iters(self):
        """``(ref, run)`` basis-row iterations per event of the matvec."""
        return (self.n_wave * self.n_basis, int(self.n_nz))

    def coeff_iters(self):
        """``(ref, run)`` wave iterations per event of the coefficient pass.

        ``ref`` is the dense ``n_wave·n_basis`` the per-(wave, row) form would
        need; ``run`` is ``n_entry`` — one per (basis row, selecting wave).
        The reduction AXIS is ``n_basis`` either way, which is the point.
        """
        return (self.n_wave * self.n_basis, int(self.n_m_entry))

    # ── s0c2: the fusion's two diagnostics ──────────────────────────────
    def kernel_stats(self):
        """The compiled resource usage of ``gsc2_amp_kernel`` (no launch).

        ``gsc2_amp_kernel`` is the kernel the compute launches for the real
        work, and its block is SHAPE-dependent (``gsc2_amp_threads``: one
        thread per wave or unique slot, rounded to a warp) while its shared
        memory is dynamic (``smem_bytes``).  Both are asked from C, so the
        occupancy reported is the one the fit actually gets.

        Returns ``dict(regs, local_bytes, blocks_per_sm, warps_per_sm,
        max_threads, static_smem, block, smem)``.  ``local_bytes > 0`` means
        the kernel SPILLS.
        """
        import numpy as np
        out = np.zeros(6, np.int32)
        self._lib.cuda_stats_flat_pwa_gen_s0c2(
            self._ctx, _ffi.cast("int*", _ffi.from_buffer(out)))
        block = int(self._lib.cuda_amp_threads_flat_pwa_gen_s0c2(self._ctx))
        return dict(regs=int(out[0]), local_bytes=int(out[1]),
                    blocks_per_sm=int(out[2]), warps_per_sm=int(out[3]),
                    max_threads=int(out[4]), static_smem=int(out[5]),
                    block=block, smem=int(self.smem_bytes))

    def feature_launch_count(self):
        """Feature-reduction launches issued so far (3 per chunk on s0c)."""
        return int(self._lib.cuda_feature_launch_count_flat_pwa_gen_s0c2())

    def reset_feature_launch_count(self):
        self._lib.cuda_reset_feature_launch_count_flat_pwa_gen_s0c2()

    def check_fused_reduce(self, a, b, c):
        """Run the fused launch and the three separate ones; bit-compare.

        ``a``/``b``/``c`` are ``(ne, nf)`` host arrays for the gck_r / gck_i /
        gco buffers.  Returns ``(equal, n_compared)`` — a direct A/B of the
        fusion against the code s0c runs, independent of ``compute``.
        """
        import numpy as np
        arrs = [np.ascontiguousarray(np.asarray(x, np.float64)) for x in (a, b, c)]
        for i in (0, 1, 2):
            if arrs[i].ndim != 2:
                raise ValueError(
                    f"cuda_flat_pwa_gen_s0c2: buffer {i} must be (ne, nf)")
            if arrs[i].shape[0] != arrs[0].shape[0]:
                raise ValueError(
                    "cuda_flat_pwa_gen_s0c2: the three buffers must share ne")
        n = _ffi.new("int*")
        rc = self._lib.cuda_check_fused_reduce_flat_pwa_gen_s0c2(
            _ffi.cast("double*", _ffi.from_buffer(arrs[0])),
            int(arrs[0].shape[1]),
            _ffi.cast("double*", _ffi.from_buffer(arrs[1])),
            int(arrs[1].shape[1]),
            _ffi.cast("double*", _ffi.from_buffer(arrs[2])),
            int(arrs[2].shape[1]),
            int(arrs[0].shape[0]), n)
        if rc < 0:
            raise RuntimeError(
                "cuda_flat_pwa_gen_s0c2: fused-reduce check failed on the GPU")
        return bool(rc), int(n[0])

    # ── data ────────────────────────────────────────────────────────────
    def load_data(self, data):
        """Build a FRESH handle; nothing is freed or overwritten here.

        The fitter keeps several samples alive at once (phase space + data), so
        a ``free_data()`` at the top of this method — or a mutable
        ``self._handle`` slot — would make the second load clobber the first
        and integrate the norm over the data.
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
            np.full(ne, float(b)) if np.isscalar(b) else b,
            np.float64).ravel())
        keep = [mass, mom, ang, w, b]
        ptr = self._lib.cuda_load_data_flat_pwa_gen_s0c2(
            self._ctx,
            _ffi.cast("double*", _ffi.from_buffer(mass)),
            _ffi.cast("double*", _ffi.from_buffer(mom)),
            _ffi.cast("double*", _ffi.from_buffer(ang)),
            _ffi.cast("double*", _ffi.from_buffer(w)),
            _ffi.cast("double*", _ffi.from_buffer(b)), ne)
        if not ptr:
            raise RuntimeError(
                "cuda_flat_pwa_gen_s0c2: load_data failed (device allocation)")
        h = FlatPwaGenS0C2DataHandle(ptr, self._lib, ne)
        h._keep = keep
        return h

    def free(self):
        """Context only.  Handles are owned by whoever called ``load_data``."""
        if getattr(self, "_ctx", None):
            self._lib.cuda_free_context_flat_pwa_gen_s0c2(self._ctx)
            self._ctx = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    # ── compute ─────────────────────────────────────────────────────────
    def compute(self, params, data_handle, norm=None, return_p=True):
        ne = data_handle.ne
        hp = data_handle.ptr
        if hp is None:
            raise RuntimeError(
                "cuda_flat_pwa_gen_s0c2: this data handle was already freed")
        ck = np.asarray(params["ck"], np.complex128).ravel()
        coeff = np.asarray(params["coeff"], np.float64).ravel()
        if ck.size != self.n_wave:
            raise ValueError(
                f"cuda_flat_pwa_gen_s0c2: ck has {ck.size} entries, expected "
                f"n_wave = {self.n_wave}")
        if coeff.size != self.n_basis:
            raise ValueError(
                f"cuda_flat_pwa_gen_s0c2: coeff has {coeff.size} entries, "
                f"expected n_basis = {self.n_basis}")
        coeff = np.ascontiguousarray(coeff)
        ckr = np.ascontiguousarray(ck.real)
        cki = np.ascontiguousarray(ck.imag)

        P = np.zeros(ne) if return_p else None
        P_c = (_ffi.cast("double*", _ffi.from_buffer(P)) if return_p
               else _ffi.NULL)
        gckr = np.zeros(self.n_wave)
        gcki = np.zeros(self.n_wave)
        gco = np.zeros(self.n_basis)
        oQ = _ffi.new("double*")
        oDn = _ffi.new("double*")
        self._lib.cuda_compute_flat_pwa_gen_s0c2(
            self._ctx, hp,
            _ffi.cast("double*", _ffi.from_buffer(ckr)),
            _ffi.cast("double*", _ffi.from_buffer(cki)),
            _ffi.cast("double*", _ffi.from_buffer(coeff)),
            float(norm) if norm is not None else 0.0,
            1 if norm is not None else 0,
            oQ, oDn, P_c,
            _ffi.cast("double*", _ffi.from_buffer(gckr)),
            _ffi.cast("double*", _ffi.from_buffer(gcki)),
            _ffi.cast("double*", _ffi.from_buffer(gco)))
        Q = float(oQ[0])

        grads = {"ck": gckr + 1j * gcki, "coeff": gco}
        if norm is not None:
            self._last_dnorm = float(oDn[0])
            grads["norm"] = self._last_dnorm
        else:
            self._last_dnorm = None
        return Q, grads, (P if return_p else None)


@register_backend("cuda_flat_pwa_gen_s0c2", model="pwa_gen")
class CUDAFlatPwaGenS0C2Backend(ComputeBackend):
    """Backend wrapper for :class:`CUDAKernelFlatPwaGenS0C2`."""

    def __init__(self, kernel_config, batch_size=50000, resolution_size=1):
        self.kernel = CUDAKernelFlatPwaGenS0C2(
            kernel_config, batch_size=batch_size,
            resolution_size=resolution_size)
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
