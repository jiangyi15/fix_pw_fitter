"""
chunked.py  –  FpwFitterChunked: FP64 chunked fitter for large datasets.

For datasets too large to fit in GPU memory (e.g. 10M events × 200 components
= 64 GB in FP64), this class uploads data chunk-by-chunk, keeping FP64 precision.
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
    typedef struct FpwFitterChunked FpwFitterChunked;

    int fpw_chunked_create(
        long long n_data,
        int     n_proj,     int     n_comp,
        const double *F_data_host,
        const double *w_data,
        const double *B_data,
        const double *M,
        double   N_b,
        double   purity,
        long long max_vram_mb,
        FpwFitterChunked **out);

    void fpw_chunked_destroy(FpwFitterChunked *f);

    int fpw_chunked_evaluate(
        FpwFitterChunked *f,
        const double *c_real,  const double *c_imag,
        double *nll,
        double *grad_real,     double *grad_imag,
        double *P_data);

    int    fpw_chunked_get_n_comp(const FpwFitterChunked *f);
    double fpw_chunked_get_N_s    (const FpwFitterChunked *f);
    double fpw_chunked_get_N_b    (const FpwFitterChunked *f);
    const char *fpw_strerror(int err);
""")

_src_dir = Path(__file__).parent

# ---------------------------------------------------------------------------
# FpwFitterChunked
# ---------------------------------------------------------------------------

_lib_chunked_cache = None


def _load_lib_chunked():
    global _lib_chunked_cache
    if _lib_chunked_cache is not None:
        return _lib_chunked_cache
    ensure_lib()
    _lib_chunked_cache = _ffi.dlopen(str(_src_dir / "libfpwfitter_chunked.so"))
    return _lib_chunked_cache


class FpwFitterChunked:
    """Chunked FP64 Fixed Partial Waves Fitter.

    F_data stays on host (FP64), uploaded chunk-by-chunk to fit in VRAM.
    All computation in FP64 for maximum precision with large datasets.

    Parameters
    ----------
    max_vram_mb : int
        Maximum VRAM to use for F_data (0 = auto ~4 GB).
        Chunk size = max_vram_mb / (KC * JP * 16).
    """

    def __init__(self, handle, lib, F_data, w_data, B_data,
                 n_comp, M=None, N_b=None, chunk_size=0):
        self._handle = handle
        self._lib = lib
        self._F_data = F_data
        self._w_data = w_data
        self._B_data = B_data
        self._n_comp = n_comp
        self._M = M
        self._N_b = N_b
        self._chunk_size = chunk_size

    @classmethod
    def from_M(cls, F_data, w_data, B_data, M, N_b, purity=1.0,
               max_vram_mb=0):
        """Create from pre-computed overlap matrix M."""
        lib = _load_lib_chunked()
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

        handle = _ffi.new("FpwFitterChunked **")
        err = lib.fpw_chunked_create(
            n_data, n_proj, n_comp,
            p_Fd, p_wd, p_Bd, p_M,
            N_b, purity,
            max_vram_mb, handle,
        )
        if err != 0:
            msg = _ffi.string(lib.fpw_strerror(err)).decode()
            raise RuntimeError(f"fpw_chunked_create failed: {msg}")

        # Compute chunk size for reporting
        bytes_per_event = n_comp * n_proj * 16 + n_proj * 16 + 24
        max_vram_bytes = max_vram_mb * 1024 * 1024 if max_vram_mb > 0 else 4 * 1024**3
        chunk = max_vram_bytes // bytes_per_event
        if chunk > n_data:
            chunk = n_data
        if chunk < 1024:
            chunk = 1024

        return cls(handle[0], lib, F64, w64, B64, n_comp,
                   M=M.copy(), N_b=N_b, chunk_size=int(chunk))

    @classmethod
    def from_mc(cls, F_data, F_mc, w_data, w_mc, B_data, B_mc,
                purity=1.0, max_vram_mb=0, chunk_size=100_000):
        """Create from raw MC data (M pre-computed via NumPy)."""
        import time
        from .compute_m import compute_M
        t0 = time.perf_counter()
        M, N_b = compute_M(F_mc, w_mc, B_mc, chunk_size)
        t1 = time.perf_counter()
        print(f"  M pre-compute: {t1-t0:.3f}s  "
              f"(n_mc={F_mc.shape[0]}, n_comp={F_mc.shape[2]})")
        return cls.from_M(F_data, w_data, B_data, M, N_b, purity, max_vram_mb)

    @property
    def n_comp(self):
        return self._n_comp

    @property
    def N_s(self):
        return self._lib.fpw_chunked_get_N_s(self._handle)

    @property
    def N_b(self):
        return self._lib.fpw_chunked_get_N_b(self._handle)

    def get_M(self):
        if self._M is not None:
            return self._M.copy(), self._N_b
        raise AttributeError("M not available")

    def save_M(self, path):
        M, N_b = self.get_M()
        np.savez(str(path), M=M, N_b=N_b)

    @classmethod
    def load_M(cls, path, F_data, w_data, B_data, purity=1.0,
               max_vram_mb=0):
        """Load M from a .npz file and create a chunked fitter."""
        data = np.load(str(path))
        return cls.from_M(F_data, w_data, B_data, data['M'],
                          float(data['N_b']), purity, max_vram_mb)

    def evaluate(self, c, return_P=False):
        """Evaluate -log L and gradient d/d(c*)."""
        c = np.ascontiguousarray(c, dtype=np.complex128)
        assert c.shape == (self._n_comp,)
        nll_out = np.zeros(1, dtype=np.float64)
        grad    = np.zeros(self._n_comp, dtype=np.complex128)

        P_data = _ffi.NULL
        if return_P:
            n_data = self._F_data.shape[0]
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

        err = self._lib.fpw_chunked_evaluate(self._handle, p_cr, p_ci, p_nll, p_gr, p_gi, p_P)
        if err != 0:
            msg = _ffi.string(self._lib.fpw_strerror(err)).decode()
            raise RuntimeError(f"fpw_chunked_evaluate failed: {msg}")

        grad.real[:] = gr
        grad.imag[:] = gi

        if return_P:
            return nll_out[0], grad, P_data
        return nll_out[0], grad

    def __del__(self):
        try:
            if hasattr(self, '_handle') and self._handle is not None and self._handle != _ffi.NULL:
                self._lib.fpw_chunked_destroy(self._handle)
                self._handle = None
        except Exception:
            pass
