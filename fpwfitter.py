"""
fpwfitter  –  Python interface to the CUDA Fixed Partial Waves Fitter.

Usage::

    from fpwfitter import FpwFitter

    fitter = FpwFitter(
        F_data,    # (n_data, n_proj, n_comp)  complex128
        F_mc,      # (n_mc,   n_proj, n_comp)  complex128
        w_data,    # (n_data,)                  float64
        w_mc,      # (n_mc,)                    float64
        B_data,    # (n_data,)                  float64
        B_mc,      # (n_mc,)                    float64
        purity=0.8,
        chunk_size=100000,
    )

    nll, grad = fitter.evaluate(c)
    # c : complex128 array of shape (n_comp,)
    # nll : float
    # grad : complex128 array of shape (n_comp,)  — d(-ln L)/d(c*)
"""

from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from cffi import FFI

# ---------------------------------------------------------------------------
# CFFI setup – out-of-line API mode
# ---------------------------------------------------------------------------

_src_dir = Path(__file__).parent

_ffi = FFI()

# CFFI type definitions (matching fpwfitter.h)
_ffi.cdef("""
    typedef struct FpwFitter FpwFitter;

    int fpw_create(
        long long n_data,   long long n_mc,
        int     n_proj,     int     n_comp,
        const double *F_data,  const double *F_mc,
        const double *w_data,  const double *w_mc,
        const double *B_data,  const double *B_mc,
        double   purity,   long long chunk_size,
        FpwFitter **out);

    void fpw_destroy(FpwFitter *f);

    int fpw_evaluate(
        FpwFitter *f,
        const double *c_real,  const double *c_imag,
        double *nll,
        double *grad_real,     double *grad_imag,
        double *P_data);

    int    fpw_get_n_comp(const FpwFitter *f);
    double fpw_get_N_s    (const FpwFitter *f);
    double fpw_get_N_b    (const FpwFitter *f);
    const char *fpw_strerror(int err);
""")

# ---------------------------------------------------------------------------
# Shared library loading / compilation
# ---------------------------------------------------------------------------

def _get_lib_path() -> Path:
    """Return the path to the compiled shared library."""
    return _src_dir / "libfpwfitter.so"


def _ensure_compiled():
    """Compile the CUDA code if the shared library doesn't exist."""
    lib = _get_lib_path()
    if lib.exists():
        return
    print("Compiling fpwfitter CUDA library...")
    import subprocess
    nvcc = os.environ.get("NVCC", "nvcc")
    cmd = [
        nvcc,
        "-Xcompiler", "-fPIC",
        "-shared",
        "-O3",
        "-arch=native",
        str(_src_dir / "fpwfitter.cu"),
        "-lcublas",
        "-o", str(lib),
    ]
    subprocess.check_call(cmd)
    print(f"  → {lib}")


# ---------------------------------------------------------------------------
# High-level wrapper class
# ---------------------------------------------------------------------------

class FpwFitter:
    """Fixed Partial Waves Fitter (GPU-accelerated)."""

    def __init__(
        self,
        F_data:   np.ndarray,
        F_mc:     np.ndarray,
        w_data:   np.ndarray,
        w_mc:     np.ndarray,
        B_data:   np.ndarray,
        B_mc:     np.ndarray,
        purity:   float = 1.0,
        chunk_size: int = 0,
    ):
        """
        Parameters
        ----------
        F_data : (n_data, n_proj, n_comp) complex128
        F_mc   : (n_mc,   n_proj, n_comp) complex128
        w_data : (n_data,) float64
        w_mc   : (n_mc,)   float64
        B_data : (n_data,) float64  – background PDF at data events
        B_mc   : (n_mc,)   float64  – background PDF at MC events
        purity : float  – signal fraction
        chunk_size : int  – GPU chunk size (0 = auto ≈ 100k)
        """
        _ensure_compiled()

        self._lib = _ffi.dlopen(str(_get_lib_path()))

        # Validate inputs
        assert F_data.dtype == np.complex128
        assert F_mc.dtype   == np.complex128
        assert w_data.dtype == np.float64
        assert w_mc.dtype   == np.float64
        assert B_data.dtype == np.float64
        assert B_mc.dtype   == np.float64

        n_data = np.int64(F_data.shape[0])
        n_mc   = np.int64(F_mc.shape[0])
        n_proj = int(F_data.shape[1])
        n_comp = int(F_data.shape[2])

        assert F_data.shape == (n_data, n_proj, n_comp)
        assert F_mc.shape   == (n_mc,   n_proj, n_comp)
        assert w_data.shape == (n_data,)
        assert w_mc.shape   == (n_mc,)
        assert B_data.shape == (n_data,)
        assert B_mc.shape   == (n_mc,)

        # Ensure contiguous C-order
        F_data = np.ascontiguousarray(F_data)
        F_mc   = np.ascontiguousarray(F_mc)
        w_data = np.ascontiguousarray(w_data)
        w_mc   = np.ascontiguousarray(w_mc)
        B_data = np.ascontiguousarray(B_data)
        B_mc   = np.ascontiguousarray(B_mc)

        # Keep references so arrays aren't garbage-collected
        self._F_data = F_data
        self._F_mc   = F_mc
        self._w_data = w_data
        self._w_mc   = w_mc
        self._B_data = B_data
        self._B_mc   = B_mc
        self._n_comp = n_comp

        # Get FFI pointers
        p_Fd = _ffi.cast("const double *", _ffi.from_buffer(F_data))
        p_Fm = _ffi.cast("const double *", _ffi.from_buffer(F_mc))
        p_wd = _ffi.cast("const double *", _ffi.from_buffer(w_data))
        p_wm = _ffi.cast("const double *", _ffi.from_buffer(w_mc))
        p_Bd = _ffi.cast("const double *", _ffi.from_buffer(B_data))
        p_Bm = _ffi.cast("const double *", _ffi.from_buffer(B_mc))

        handle = _ffi.new("FpwFitter **")
        err = self._lib.fpw_create(
            n_data, n_mc, n_proj, n_comp,
            p_Fd, p_Fm, p_wd, p_wm, p_Bd, p_Bm,
            purity, chunk_size, handle,
        )
        if err != 0:
            msg = _ffi.string(self._lib.fpw_strerror(err)).decode()
            raise RuntimeError(f"fpw_create failed: {msg}")

        self._handle = handle[0]
        self._n_comp = n_comp

    @property
    def n_comp(self) -> int:
        return self._n_comp

    @property
    def N_s(self) -> float:
        return self._lib.fpw_get_N_s(self._handle)

    @property
    def N_b(self) -> float:
        return self._lib.fpw_get_N_b(self._handle)

    def evaluate(
        self,
        c: np.ndarray,
        return_P: bool = False,
    ) -> tuple[float, np.ndarray] | tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate -log L and its gradient.

        Parameters
        ----------
        c : (n_comp,) complex128  – coupling vector
        return_P : bool  – if True, also return P_i values

        Returns
        -------
        nll : float
        grad : (n_comp,) complex128  – d(-ln L)/d(c*)
        P_data : (n_data,) float64  (only if return_P=True)
        """
        c = np.ascontiguousarray(c, dtype=np.complex128)
        assert c.shape == (self._n_comp,)

        nll_out = np.zeros(1, dtype=np.float64)
        grad    = np.zeros(self._n_comp, dtype=np.complex128)

        # P_data buffer
        P_data = _ffi.NULL
        if return_P:
            n_data = self._F_data.shape[0]
            P_data = np.ascontiguousarray(np.zeros(n_data, dtype=np.float64))

        # Get contiguous real/imag views
        cr = np.ascontiguousarray(c.real)
        ci = np.ascontiguousarray(c.imag)
        gr = np.ascontiguousarray(grad.real)
        gi = np.ascontiguousarray(grad.imag)

        p_cr = _ffi.cast("const double *", _ffi.from_buffer(cr))
        p_ci = _ffi.cast("const double *", _ffi.from_buffer(ci))
        p_nll = _ffi.cast("double *", _ffi.from_buffer(nll_out))
        p_gr  = _ffi.cast("double *", _ffi.from_buffer(gr))
        p_gi  = _ffi.cast("double *", _ffi.from_buffer(gi))
        p_P   = _ffi.NULL if P_data is _ffi.NULL else _ffi.cast("double *", _ffi.from_buffer(P_data))

        err = self._lib.fpw_evaluate(
            self._handle, p_cr, p_ci, p_nll, p_gr, p_gi, p_P,
        )
        if err != 0:
            msg = _ffi.string(self._lib.fpw_strerror(err)).decode()
            raise RuntimeError(f"fpw_evaluate failed: {msg}")

        # Copy results back from contiguous buffers
        grad.real[:] = gr
        grad.imag[:] = gi

        if return_P:
            return nll_out[0], grad, P_data
        return nll_out[0], grad

    def __del__(self):
        try:
            if hasattr(self, '_handle') and self._handle is not None and self._handle != _ffi.NULL:
                self._lib.fpw_destroy(self._handle)
                self._handle = None
        except Exception:
            pass
