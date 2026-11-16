"""``cuda_flat_cache_pwa_gen`` — flat ``pwa_gen`` on the GPU, cached ``F`` +
**backend-internal** dedup.

Case-1 copy: own ``.cu``, own ``.so``, own wrapper, own symbols.  Two things
distinguish it from :mod:`tabpwa.cuda._flat_pwa_gen`:

1. **Cached amplitude.**  ``F_{e,k} = fl_{e,k}·Π_r num_{e,k,r}`` does not
   depend on ``ck`` or ``coeff``, so it is filled once in ``load_data`` and
   reused every iteration; only the denominator (which carries ``coeff``)
   and the projection sum run per compute.

2. **Structural dedup, done HERE.**  The shared builder appends one basis row
   per (factor, basis row) with duplicates kept; this wrapper folds them on
   ``(coeff name, basis kind, mass column)`` — the row's physical identity —
   and remaps ``gamma_matrix`` accordingly, before anything is uploaded.
   Nothing in :mod:`tabpwa.pwa_gen_config` is touched, so every other backend
   keeps reading the config exactly as emitted.

The fold is lossless: ``coeff`` is gathered onto the surviving rows with the
same ``keep`` list, so a caller's full-form vector is remapped rather than
reinterpreted.  Folding on the structural key (not on sampled values) cannot
merge two rows that merely look alike numerically — two identical
``kmatrix2`` models at different mass columns stay apart, correctly.

Host contract, identical to the other flat backends::

    params = {"ck": complex (n_wave,), "coeff": real (n_basis,)}
    grads  = {"ck", "coeff", "norm"}      # norm only on a normed call
"""
import numpy as np

from tabpwa.backends.core import ComputeBackend, register_backend

_ffi = None
_lib = None


def _load_lib():
    from cffi import FFI
    from tabpwa.cuda.build import ensure
    global _ffi
    if _ffi is None:
        _ffi = FFI()
        _ffi.cdef("""
void* cuda_create_context_flat_cache(
    const double*,int, const int*,int, const double*,const double*,int,
    const int*,int, const double*,const double*,int, const int*,int,
    const double*,int, const int*,int, const double*,int,
    const double*,int, const double*,const double*,int,
    double,double,int, double,double,int,
    int,int,int,int,int, int,int,int,int,int, int);
void* cuda_load_data_flat_cache(void*,const double*,const int*,
    const double*,const int*, const double*,const int*,
    const double*,const double*,int);
void cuda_free_data_flat_cache(void*);
void cuda_free_context_flat_cache(void*);
void cuda_compute_flat_cache(void*,const double*,const double*,
    const double*,double,int, double*,double*,double*,double*,double*,double*);
int cuda_get_device_count_flat_cache(void);
void cuda_get_device_name_flat_cache(char*,int);
""")
    lib_path = ensure("kernels_flat_cache_pwa_gen.cu",
                      "libcuda_kernels_flat_cache_pwa_gen.so")
    if not lib_path:
        raise RuntimeError("cuda_flat_cache_pwa_gen: library not found")
    return _ffi.dlopen(lib_path)


def dedup_basis(config):
    """Fold the basis rows on ``(coeff name, kind, mass column)``.

    Returns ``(keep, mapping)``: ``keep`` lists the row indices that survive
    (in order), ``mapping`` maps each original row onto its survivor.  This is
    the same fold :func:`tabpwa.pwa_gen_config.dedup` performs, expressed here
    so it happens **inside this backend** and leaves the shared builder — and
    therefore every other backend — untouched.
    """
    names = config["den_names"]
    col = np.asarray(config["basis_m_index"], np.int64)
    bas = np.asarray(config["basis_table"])
    if len(names) != bas.shape[0]:
        raise ValueError(
            f"cuda_flat_cache_pwa_gen: den_names has {len(names)} entries but "
            f"basis_table has {bas.shape[0]} rows")

    # the basis KIND is the row's position within its factor's block, which is
    # what distinguishes "1" / "−m²" / "den_table[r]" structurally.  Recover it
    # from the selector: every factor uses a contiguous run of columns of the
    # same length, so the position inside that run IS the kind.
    gam = np.asarray(config["gamma_matrix"])
    kind = np.zeros(len(names), np.int64)
    for k in range(gam.shape[0]):
        for l in range(gam.shape[1]):
            cols = np.nonzero(gam[k, l])[0]
            for pos, c in enumerate(cols):
                kind[c] = pos

    seen, keep, mapping = {}, [], np.zeros(len(names), np.int64)
    for i in range(len(names)):
        key = (str(names[i]), int(kind[i]), int(col[i]))
        pos = seen.get(key)
        if pos is None:
            pos = len(keep)
            seen[key] = pos
            keep.append(i)
        mapping[i] = pos
    return np.array(keep, np.int64), mapping


class CUDAKernelFlatCachePWAGen:
    """Fused-tile CUDA implementation with a cached amplitude and deduped basis."""

    def __init__(self, config, batch_size=50000):
        global _lib
        if "fl_index" in config or "num_index" in config:
            raise ValueError(
                "cuda_flat_cache_pwa_gen: pass the FULL (non-deduped) config "
                "— this backend does its own dedup, and a config that already "
                "carries fl_index/num_index has a different row layout")
        if _lib is None:
            _lib = _load_lib()
        self._lib = _lib
        self.config = config
        if _lib.cuda_get_device_count_flat_cache() <= 0:
            raise RuntimeError("cuda_flat_cache_pwa_gen: no CUDA devices")
        buf = _ffi.new("char[256]")
        _lib.cuda_get_device_name_flat_cache(buf, 256)
        self.device_name = _ffi.string(buf).decode()

        c = config
        self.n_wave = int(np.asarray(c["fl_table"]).shape[0])
        self.n_proj = int(np.asarray(c["angle_matrix"]).shape[1])
        self.n_res = int(np.asarray(c["num_table"]).shape[1])
        self.n_decay = int(np.asarray(c["fl_q_index"]).shape[1])
        self.n_ab = int(np.asarray(c["angle_k"]).shape[0])
        self.n_comp = int(np.asarray(c["angle_k"]).shape[1])
        self.n_mass = int(np.max(c["num_m_index"])) + 1
        self.n_momentum = int(np.max(c["fl_q_index"])) + 1
        self.n_angle_total = int(np.max(c["angle_index"])) + 1
        self.batch_size = int(batch_size)
        self._last_dnorm = None

        # ── the backend-internal dedup ──────────────────────────────
        keep, mapping = dedup_basis(c)
        self.basis_keep = keep
        self.basis_map = mapping
        bas = np.asarray(c["basis_table"], np.complex128)[keep]
        bas_m = np.asarray(c["basis_m_index"], np.int64)[keep]
        self.n_basis_full = int(np.asarray(c["basis_table"]).shape[0])
        self.n_basis = int(bas.shape[0])

        gam_old = np.asarray(c["gamma_matrix"], np.float64)
        gam = np.zeros((gam_old.shape[0], gam_old.shape[1], self.n_basis))
        for m_old in range(gam_old.shape[2]):
            np.maximum.at(gam, (slice(None), slice(None), mapping[m_old]),
                          gam_old[:, :, m_old])

        keep_alive = []
        fl_tab = np.asarray(c["fl_table"], np.float64).reshape(
            self.n_wave * self.n_decay, -1)
        num = np.asarray(c["num_table"], np.complex128).reshape(
            self.n_wave * self.n_res, -1)
        am = np.asarray(c["angle_matrix"], np.complex128)

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
        self._ctx = _lib.cuda_create_context_flat_cache(
            D(fl_tab), fl_tab.size, I(fl_q), fl_q.size,
            D(num.real), D(num.imag), num.size, I(num_m), num_m.size,
            D(bas.real), D(bas.imag), bas.size, I(bas_m), self.n_basis,
            D(gam), int(gam.size),
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
            self.batch_size)
        if not self._ctx:
            raise RuntimeError(
                "cuda_flat_cache_pwa_gen: context creation failed")
        self._keep = keep_alive
        self._handle = None

    # ── data ────────────────────────────────────────────────────────
    def load_data(self, data):
        self.free_data()
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
        self._keep_data = keep
        self._handle = self._lib.cuda_load_data_flat_cache(
            self._ctx,
            _ffi.cast("double*", _ffi.from_buffer(mass)), _ffi.NULL,
            _ffi.cast("double*", _ffi.from_buffer(mom)), _ffi.NULL,
            _ffi.cast("double*", _ffi.from_buffer(ang)), _ffi.NULL,
            _ffi.cast("double*", _ffi.from_buffer(w)),
            _ffi.cast("double*", _ffi.from_buffer(b)), ne)
        return {"handle": self._handle, "ne": ne, "_keep": keep}

    def free_data(self):
        if getattr(self, "_handle", None) is not None:
            self._lib.cuda_free_data_flat_cache(self._handle)
            self._handle = None

    def free(self):
        self.free_data()
        if getattr(self, "_ctx", None):
            self._lib.cuda_free_context_flat_cache(self._ctx)
            self._ctx = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    # ── compute ─────────────────────────────────────────────────────
    def compute(self, params, data_handle, norm=None, return_p=True):
        h = data_handle["handle"]
        ne = data_handle["ne"]
        ck = np.asarray(params["ck"], np.complex128).ravel()
        coeff_full = np.asarray(params["coeff"], np.float64).ravel()
        if ck.size != self.n_wave:
            raise ValueError(
                f"cuda_flat_cache_pwa_gen: ck has {ck.size} entries, expected "
                f"n_wave = {self.n_wave}")
        if coeff_full.size != self.n_basis_full:
            raise ValueError(
                f"cuda_flat_cache_pwa_gen: coeff has {coeff_full.size} "
                f"entries, expected the FULL n_basis = {self.n_basis_full} "
                f"(this backend dedups internally; the caller keeps the "
                f"full-form vector)")
        # gather the caller's full-form weights onto the surviving rows
        coeff = np.ascontiguousarray(coeff_full[self.basis_keep])
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
        self._lib.cuda_compute_flat_cache(
            self._ctx,
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

        # Scatter the deduped coefficient gradient back to the FULL row space.
        #
        # The kernel accumulated one total per SURVIVING row (it sums every
        # (wave, factor) copy that uses it).  That total belongs to the ONE
        # surviving row — the folded copies carry ZERO, so the vector holds
        # each weight's derivative exactly once and a plain sum over the rows
        # gives the total without over-counting.  Replicating it onto every
        # copy (``gco_dd[basis_map]``) would multiply the weight's gradient by
        # the number of copies.
        gco_out = np.zeros(self.n_basis_full)
        gco_out[self.basis_keep] = gco_dd

        grads = {"ck": gckr + 1j * gcki, "coeff": gco_out}
        if norm is not None:
            self._last_dnorm = float(oDn[0])
            grads["norm"] = self._last_dnorm
        else:
            self._last_dnorm = None
        return Q, grads, (P if return_p else None)


@register_backend("cuda_flat_cache_pwa_gen", model="pwa_gen")
class CUDAFlatCachePWAGenBackend(ComputeBackend):
    """Backend wrapper for :class:`CUDAKernelFlatCachePWAGen`."""

    def __init__(self, kernel_config, batch_size=50000):
        self.kernel = CUDAKernelFlatCachePWAGen(kernel_config,
                                                batch_size=batch_size)

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
