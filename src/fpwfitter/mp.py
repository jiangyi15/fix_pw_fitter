"""
mp.py  –  FpwFitterMP class  (Mixed-Precision, FP32 compute / FP64 I/O).
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from cffi import FFI

from ._build import ensure_lib

# ---------------------------------------------------------------------------
# CFFI setup
# ---------------------------------------------------------------------------

_ffi = FFI()

_ffi.cdef("""
    typedef struct FpwFitterMP FpwFitterMP;

    int fpw_mp_create(
        long long n_data,
        int     n_proj,     int     n_comp,
        const float *F_data,
        const float *w_data,
        const float *B_data,
        const float *M,
        float   N_b,
        float   purity,
        FpwFitterMP **out);

    void fpw_mp_destroy(FpwFitterMP *f);

    int fpw_mp_evaluate(
        FpwFitterMP *f,
        const double *c_real,  const double *c_imag,
        double *nll,
        double *grad_real,     double *grad_imag,
        float  *P_data);

    int    fpw_mp_get_n_comp(const FpwFitterMP *f);
    float  fpw_mp_get_N_s    (const FpwFitterMP *f);
    float  fpw_mp_get_N_b    (const FpwFitterMP *f);
    const char *fpw_strerror(int err);
""")

_src_dir = Path(__file__).parent

# ---------------------------------------------------------------------------
# FpwFitterMP  (Mixed Precision)
# ---------------------------------------------------------------------------

_lib_mp_cache = None


def _load_lib_mp():
    global _lib_mp_cache
    if _lib_mp_cache is not None:
        return _lib_mp_cache
    ensure_lib()
    _lib_mp_cache = _ffi.dlopen(str(_src_dir / "libfpwfitter_mp.so"))
    return _lib_mp_cache


class FpwFitterMP:
    """Mixed-Precision Fixed Partial Waves Fitter.

    All internal computation in FP32 (float/float2).
    Data uploaded as FP32, coupling vector and gradient in FP64.

    Uses ~50× more FP32 throughput than FP64 on consumer GPUs.
    """

    def __init__(self, handle, lib, F_data, w_data, B_data,
                 n_comp, M=None, N_b=None):
        self._handle = handle
        self._lib = lib
        self._F_data = F_data
        self._w_data = w_data
        self._B_data = B_data
        self._n_comp = n_comp
        self._M = M
        self._N_b = N_b

    @classmethod
    def from_M(cls, F_data, w_data, B_data, M, N_b, purity=1.0):
        """Create from pre-computed overlap matrix M."""
        lib = _load_lib_mp()
        n_data = int(F_data.shape[0])
        n_proj = int(F_data.shape[1])
        n_comp = int(F_data.shape[2])

        # Convert to FP32
        F32 = np.ascontiguousarray(F_data, dtype=np.complex64)
        w32 = np.ascontiguousarray(w_data, dtype=np.float32)
        B32 = np.ascontiguousarray(B_data, dtype=np.float32)
        M32 = np.ascontiguousarray(M, dtype=np.complex64)

        p_Fd = _ffi.cast("const float *", _ffi.from_buffer(F32))
        p_wd = _ffi.cast("const float *", _ffi.from_buffer(w32))
        p_Bd = _ffi.cast("const float *", _ffi.from_buffer(B32))
        p_M  = _ffi.cast("const float *", _ffi.from_buffer(M32))

        handle = _ffi.new("FpwFitterMP **")
        err = lib.fpw_mp_create(n_data, n_proj, n_comp,
                                p_Fd, p_wd, p_Bd, p_M,
                                np.float32(N_b), np.float32(purity), handle)
        if err != 0:
            msg = _ffi.string(lib.fpw_strerror(err)).decode()
            raise RuntimeError(f"fpw_mp_create failed: {msg}")
        return cls(handle[0], lib, F32, w32, B32, n_comp,
                   M=M.copy(), N_b=N_b)

    @classmethod
    def from_mc(cls, F_data, F_mc, w_data, w_mc, B_data, B_mc,
                purity=1.0, chunk_size=100_000):
        """Create from raw MC data (M pre-computed via NumPy)."""
        import time
        from .compute_m import compute_M
        t0 = time.perf_counter()
        M, N_b = compute_M(F_mc, w_mc, B_mc, chunk_size)
        t1 = time.perf_counter()
        print(f"  M pre-compute: {t1-t0:.3f}s  "
              f"(n_mc={F_mc.shape[0]}, n_comp={F_mc.shape[2]})")
        return cls.from_M(F_data, w_data, B_data, M, N_b, purity)

    @property
    def n_comp(self):
        return self._n_comp

    @property
    def N_s(self):
        return float(self._lib.fpw_mp_get_N_s(self._handle))

    @property
    def N_b(self):
        return float(self._lib.fpw_mp_get_N_b(self._handle))

    def get_M(self):
        if self._M is not None:
            return self._M.copy(), self._N_b
        raise AttributeError("M not available")

    def save_M(self, path):
        M, N_b = self.get_M()
        np.savez(str(path), M=M, N_b=N_b)

    @classmethod
    def load_M(cls, path, F_data, w_data, B_data, purity=1.0):
        data = np.load(str(path))
        return cls.from_M(F_data, w_data, B_data, data['M'],
                          float(data['N_b']), purity)

    def evaluate(self, c, return_P=False):
        """Evaluate -log L and gradient d/d(c*)."""
        c = np.ascontiguousarray(c, dtype=np.complex128)
        assert c.shape == (self._n_comp,)
        nll_out = np.zeros(1, dtype=np.float64)
        grad    = np.zeros(self._n_comp, dtype=np.complex128)

        P_data = _ffi.NULL
        if return_P:
            n_data = self._F_data.shape[0]
            P_data = np.ascontiguousarray(np.zeros(n_data, dtype=np.float32))

        cr = np.ascontiguousarray(c.real)
        ci = np.ascontiguousarray(c.imag)
        gr = np.ascontiguousarray(grad.real)
        gi = np.ascontiguousarray(grad.imag)
        p_cr  = _ffi.cast("const double *", _ffi.from_buffer(cr))
        p_ci  = _ffi.cast("const double *", _ffi.from_buffer(ci))
        p_nll = _ffi.cast("double *", _ffi.from_buffer(nll_out))
        p_gr  = _ffi.cast("double *", _ffi.from_buffer(gr))
        p_gi  = _ffi.cast("double *", _ffi.from_buffer(gi))
        p_P   = _ffi.NULL if P_data is _ffi.NULL else _ffi.cast("float *", _ffi.from_buffer(P_data))

        err = self._lib.fpw_mp_evaluate(self._handle, p_cr, p_ci, p_nll, p_gr, p_gi, p_P)
        if err != 0:
            msg = _ffi.string(self._lib.fpw_strerror(err)).decode()
            raise RuntimeError(f"fpw_mp_evaluate failed: {msg}")

        grad.real[:] = gr
        grad.imag[:] = gi

        if return_P:
            return nll_out[0], grad, P_data.astype(np.float64)
        return nll_out[0], grad

    def __del__(self):
        try:
            if hasattr(self, '_handle') and self._handle is not None and self._handle != _ffi.NULL:
                self._lib.fpw_mp_destroy(self._handle)
                self._handle = None
        except Exception:
            pass
