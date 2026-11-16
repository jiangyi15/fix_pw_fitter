"""``cuda_flat_pwa_gen`` — the flat ``pwa_gen`` config on the GPU.

Case-1 copy: own ``.cu``, own ``.so``, own wrapper, own symbols
(``cuda_*_flat_pwa_gen``).  Nothing is shared with another kernel.

It consumes the arrays emitted by
:func:`tabpwa.pwa_gen_config.build_from_groups` **as emitted** — the full,
non-deduped form.  A config that went through the host-side
:func:`tabpwa.pwa_gen_config.dedup` pass is rejected with a clear error: the
folded tables carry ``fl_index``/``num_index``, which this kernel does not
read, so accepting them would silently mis-index every row.

Host contract, identical to ``numpy_flat``::

    params = {"ck": complex (n_wave,), "coeff": real (n_basis,)}
    grads  = {"ck", "coeff", "norm"}          # norm only on a normed call
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
void* cuda_create_context_flat_pwa_gen(
    const double*,int, const int*,int, const double*,const double*,int,
    const int*,int, const double*,const double*,int, const int*,int,
    const double*,int, const int*,int, const double*,int,
    const double*,int, const double*,const double*,int,
    double,double,int, double,double,int,
    int,int,int,int,int, int,int,int,int,int, int);
void* cuda_load_data_flat_pwa_gen(void*,const double*,const int*,
    const double*,const int*, const double*,const int*,
    const double*,const double*,int);
void cuda_free_data_flat_pwa_gen(void*);
void cuda_free_context_flat_pwa_gen(void*);
void cuda_compute_flat_pwa_gen(void*,const double*,const double*,
    const double*,double,int, double*,double*,double*,double*,double*,double*);
int cuda_get_device_count_flat_pwa_gen(void);
void cuda_get_device_name_flat_pwa_gen(char*,int);
""")
    lib_path = ensure("kernels_flat_pwa_gen.cu", "libcuda_kernels_flat_pwa_gen.so")
    if not lib_path:
        raise RuntimeError(
            "cuda_flat_pwa_gen: library not found at "
            f"{lib_path} (no nvcc?)")
    return _ffi.dlopen(lib_path)


def _db(a):
    """numpy float64 array -> cffi double* (must stay alive for the call)."""
    arr = np.ascontiguousarray(a, np.float64)
    return arr, _ffi.from_buffer(arr)


def _ib(a):
    arr = np.ascontiguousarray(a, np.int32)
    return arr, _ffi.from_buffer(arr)


class CUDAKernelFlatPWAGen:
    """Fused-tile CUDA implementation of the flat ``pwa_gen`` config."""

    def __init__(self, config, batch_size=50000):
        global _lib
        if "fl_index" in config or "num_index" in config:
            raise ValueError(
                "cuda_flat_pwa_gen: this kernel reads the FULL (non-deduped) "
                "config; the given one carries fl_index/num_index, i.e. it "
                "was produced by pwa_gen_config.dedup().  Build the config "
                "with build_from_groups() for this backend.")
        if _lib is None:
            _lib = _load_lib()
        self._lib = _lib
        self.config = config
        n = _lib.cuda_get_device_count_flat_pwa_gen()
        if n <= 0:
            raise RuntimeError("cuda_flat_pwa_gen: no CUDA devices")
        buf = _ffi.new("char[256]")
        _lib.cuda_get_device_name_flat_pwa_gen(buf, 256)
        self.device_name = _ffi.string(buf).decode()

        c = config
        self.n_wave = int(np.asarray(c["fl_table"]).shape[0])
        self.n_proj = int(np.asarray(c["angle_matrix"]).shape[1])
        self.n_res = int(np.asarray(c["num_table"]).shape[1])
        self.n_decay = int(np.asarray(c["fl_q_index"]).shape[1])
        self.n_basis = int(np.asarray(c["basis_table"]).shape[0])
        self.n_ab = int(np.asarray(c["angle_k"]).shape[0])
        self.n_comp = int(np.asarray(c["angle_k"]).shape[1])
        self.n_mass = int(np.max(c["num_m_index"])) + 1
        self.n_momentum = int(np.max(c["fl_q_index"])) + 1
        self.n_angle_total = int(np.max(c["angle_index"])) + 1
        self.batch_size = int(batch_size)
        self._last_dnorm = None

        # keep the uploads alive until create returns
        keep = []
        fl_tab = np.asarray(c["fl_table"], np.float64).reshape(
            self.n_wave * self.n_decay, -1)
        num = np.asarray(c["num_table"], np.complex128).reshape(
            self.n_wave * self.n_res, -1)
        bas = np.asarray(c["basis_table"], np.complex128)
        am = np.asarray(c["angle_matrix"], np.complex128)

        def D(a):
            arr, buf = _db(a)
            keep.append(arr)
            return _ffi.cast("double*", buf)

        def I(a):
            arr, buf = _ib(a)
            keep.append(arr)
            return _ffi.cast("int*", buf)

        fl_q = np.asarray(c["fl_q_index"], np.int64)
        num_m = np.asarray(c["num_m_index"], np.int64)
        self._ctx = _lib.cuda_create_context_flat_pwa_gen(
            D(fl_tab), fl_tab.size, I(fl_q), fl_q.size,
            D(num.real), D(num.imag), num.size,
            I(num_m), num_m.size,
            D(bas.real), D(bas.imag), bas.size,
            I(np.asarray(c["basis_m_index"])), self.n_basis,
            D(np.asarray(c["gamma_matrix"], np.float64)),
            int(np.asarray(c["gamma_matrix"]).size),
            I(np.asarray(c["angle_index"])), self.n_ab,
            D(np.asarray(c["angle_k"], np.float64)),
            int(np.asarray(c["angle_k"]).size),
            D(np.asarray(c["angle_b"], np.float64)),
            int(np.asarray(c["angle_b"]).size),
            D(am.real), D(am.imag), am.size,
            float(c["m_min"]), float(c["m_delta"]),
            int(bas.shape[1]),
            float(c["q_min"]), float(c["q_delta"]),
            int(fl_tab.shape[1]),
            self.n_wave, self.n_proj, self.n_res, self.n_decay,
            self.n_basis, self.n_ab, self.n_comp,
            self.n_mass, self.n_momentum, self.n_angle_total,
            self.batch_size)
        if not self._ctx:
            raise RuntimeError(
                "cuda_flat_pwa_gen: context creation failed (see the kernel's "
                "stderr message)")
        self._keep = keep
        self._handle = None

    # ── data ────────────────────────────────────────────────────────
    def load_data(self, data):
        self.free_data()
        mass = np.ascontiguousarray(np.asarray(data["mass"], np.float64))
        mom = np.ascontiguousarray(np.asarray(data["q"], np.float64))
        ang = np.ascontiguousarray(np.asarray(data["angle"], np.float64))
        if ang.ndim == 2:      # flattened (ne, n_at*n_ac) is accepted
            ang = ang.reshape(ang.shape[0], self.n_angle_total, self.n_comp)
        ne = mass.shape[0]
        w = np.ascontiguousarray(np.asarray(
            data.get("weight", np.ones(ne)), np.float64).ravel())
        b = data.get("bkg", np.zeros(ne))
        b = np.ascontiguousarray(np.asarray(
            np.full(ne, float(b)) if np.isscalar(b) else b, np.float64).ravel())
        keep = [mass, mom, ang, w, b]
        self._keep_data = keep
        self._handle = self._lib.cuda_load_data_flat_pwa_gen(
            self._ctx,
            _ffi.cast("double*", _ffi.from_buffer(mass)), _ffi.NULL,
            _ffi.cast("double*", _ffi.from_buffer(mom)), _ffi.NULL,
            _ffi.cast("double*", _ffi.from_buffer(ang)), _ffi.NULL,
            _ffi.cast("double*", _ffi.from_buffer(w)),
            _ffi.cast("double*", _ffi.from_buffer(b)), ne)
        return {"handle": self._handle, "ne": ne, "_keep": keep}

    def free_data(self):
        if self._handle is not None:
            self._lib.cuda_free_data_flat_pwa_gen(self._handle)
            self._handle = None

    def free(self):
        self.free_data()
        if getattr(self, "_ctx", None):
            self._lib.cuda_free_context_flat_pwa_gen(self._ctx)
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
        coeff = np.asarray(params["coeff"], np.float64).ravel()
        if ck.size != self.n_wave:
            raise ValueError(
                f"cuda_flat_pwa_gen: ck has {ck.size} entries, expected "
                f"n_wave = {self.n_wave}")
        if coeff.size != self.n_basis:
            raise ValueError(
                f"cuda_flat_pwa_gen: coeff has {coeff.size} entries, "
                f"expected n_basis = {self.n_basis}")
        ckr = np.ascontiguousarray(ck.real)
        cki = np.ascontiguousarray(ck.imag)
        cf = np.ascontiguousarray(coeff)

        # P must be a real host buffer the kernel writes into (a
        # ``_ffi.new("double*")`` is a pointer-to-pointer, i.e. NULL here)
        P = np.zeros(ne) if return_p else None
        P_c = _ffi.cast("double*", _ffi.from_buffer(P)) if return_p \
            else _ffi.NULL
        gckr = np.zeros(self.n_wave)
        gcki = np.zeros(self.n_wave)
        gco = np.zeros(self.n_basis)
        oQ = _ffi.new("double*")
        oDn = _ffi.new("double*")
        self._lib.cuda_compute_flat_pwa_gen(
            self._ctx,
            _ffi.cast("double*", _ffi.from_buffer(ckr)),
            _ffi.cast("double*", _ffi.from_buffer(cki)),
            _ffi.cast("double*", _ffi.from_buffer(cf)),
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


@register_backend("cuda_flat_pwa_gen", model="pwa_gen")
class CUDAFlatPWAGenBackend(ComputeBackend):
    """Backend wrapper for :class:`CUDAKernelFlatPWAGen`."""

    def __init__(self, kernel_config, batch_size=50000):
        self.kernel = CUDAKernelFlatPWAGen(kernel_config,
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
