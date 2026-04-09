"""
parallel.py  –  FpwFitterParallel: Multi-stream CUDA parallel fitter.

Splits data into N chunks processed in parallel CUDA streams at the C level.
No Python GIL overhead — parallelism is entirely in CUDA.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from cffi import FFI

from ._build import ensure_lib

_ffi = FFI()

_ffi.cdef("""
    typedef struct FpwFitterParallel FpwFitterParallel;

    int fpw_par_create(
        long long n_data,
        int     n_proj,     int     n_comp,
        const double *F_data,
        const double *w_data,
        const double *B_data,
        const double *M,
        double   N_b,
        double   purity,
        int      n_streams,
        FpwFitterParallel **out);

    void fpw_par_destroy(FpwFitterParallel *f);

    int fpw_par_evaluate(
        FpwFitterParallel *f,
        const double *c_real,  const double *c_imag,
        double *nll,
        double *grad_real,     double *grad_imag,
        double *P_data);

    int    fpw_par_get_n_comp(const FpwFitterParallel *f);
    double fpw_par_get_N_s    (const FpwFitterParallel *f);
    double fpw_par_get_N_b    (const FpwFitterParallel *f);
    const char *fpw_par_strerror(int err);
""")

_src_dir = Path(__file__).parent

_lib_cache = None

def _load_lib():
    global _lib_cache
    if _lib_cache is not None:
        return _lib_cache
    ensure_lib()
    _lib_cache = _ffi.dlopen(str(_src_dir / "libfpwfitter_parallel.so"))
    return _lib_cache


class FpwFitterParallel:
    """Multi-stream parallel FP64 Fixed Partial Waves Fitter.

    Data is split into N chunks, each processed in a separate CUDA stream
    at the C/CUDA level. Results are summed on host. Avoids Python GIL.

    Parameters
    ----------
    n_streams : int
        Number of CUDA streams (1-8). Default: 2.
    """

    def __init__(self, handle, lib, n_streams):
        self._handle = handle
        self._lib = lib
        self._n_streams = n_streams

    @classmethod
    def from_M(cls, F_data, w_data, B_data, M, N_b, purity=1.0,
               n_streams=2):
        """Create from pre-computed overlap matrix M."""
        lib = _load_lib()
        n_data = int(F_data.shape[0])
        n_proj = int(F_data.shape[1])
        n_comp = int(F_data.shape[2])

        F64 = np.ascontiguousarray(F_data, dtype=np.complex128)
        w64 = np.ascontiguousarray(w_data, dtype=np.float64)
        B64 = np.ascontiguousarray(B_data, dtype=np.float64)
        M64 = np.ascontiguousarray(M, dtype=np.complex128)

        p_Fd = _ffi.cast("const double *", _ffi.from_buffer(F64))
        p_wd = _ffi.cast("const double *", _ffi.from_buffer(w64))
        p_Bd = _ffi.cast("const double *", _ffi.from_buffer(B64))
        p_M  = _ffi.cast("const double *", _ffi.from_buffer(M64))

        handle = _ffi.new("FpwFitterParallel **")
        err = lib.fpw_par_create(
            n_data, n_proj, n_comp,
            p_Fd, p_wd, p_Bd, p_M,
            N_b, purity,
            n_streams, handle,
        )
        if err != 0:
            msg = _ffi.string(lib.fpw_par_strerror(err)).decode()
            raise RuntimeError(f"fpw_par_create failed: {msg}")

        return cls(handle[0], lib, n_streams)

    @property
    def n_comp(self):
        return self._lib.fpw_par_get_n_comp(self._handle)

    @property
    def N_s(self):
        return self._lib.fpw_par_get_N_s(self._handle)

    @property
    def N_b(self):
        return self._lib.fpw_par_get_N_b(self._handle)

    def evaluate(self, c, return_P=False):
        """Evaluate -log L and gradient d/d(c*)."""
        c = np.ascontiguousarray(c, dtype=np.complex128)
        assert c.shape == (self.n_comp,)
        nll_out = np.zeros(1, dtype=np.float64)
        grad    = np.zeros(self.n_comp, dtype=np.complex128)

        P_data = _ffi.NULL
        if return_P:
            n_data = int(self._lib.fpw_par_get_n_comp(self._handle))  # placeholder
            P_data = np.ascontiguousarray(np.zeros(n_data, dtype=np.float64))

        cr = np.ascontiguousarray(c.real)
        ci = np.ascontiguousarray(c.imag)
        gr = np.ascontiguousarray(grad.real)
        gi = np.ascontiguousarray(grad.imag)
        p_cr  = _ffi.cast("const double *", _ffi.from_buffer(cr))
        p_ci  = _ffi.cast("const double *", _ffi.from_buffer(ci))
        p_nll = _ffi.cast("double *", _ffi.from_buffer(nll_out))
        p_gr  = _ffi.cast("double *", _ffi.from_buffer(gr))
        p_gi  = _ffi.cast("double *", _ffi.from_buffer(gi))
        p_P   = _ffi.NULL if P_data is _ffi.NULL else _ffi.cast("double *", _ffi.from_buffer(P_data))

        err = self._lib.fpw_par_evaluate(self._handle, p_cr, p_ci, p_nll, p_gr, p_gi, p_P)
        if err != 0:
            msg = _ffi.string(self._lib.fpw_par_strerror(err)).decode()
            raise RuntimeError(f"fpw_par_evaluate failed: {msg}")

        grad.real[:] = gr
        grad.imag[:] = gi

        if return_P:
            return nll_out[0], grad, P_data
        return nll_out[0], grad

    def __del__(self):
        try:
            if hasattr(self, '_handle') and self._handle is not None and self._handle != _ffi.NULL:
                self._lib.fpw_par_destroy(self._handle)
                self._handle = None
        except Exception:
            pass
