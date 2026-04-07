"""
core.py  –  FpwFitter class  (CFFI wrapper).
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
    typedef struct FpwFitter FpwFitter;

    int fpw_create(
        long long n_data,
        int     n_proj,     int     n_comp,
        const double *F_data,
        const double *w_data,
        const double *B_data,
        const double *M,
        double   N_b,
        double   purity,
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

_src_dir = Path(__file__).parent


# ---------------------------------------------------------------------------
# FpwFitter
# ---------------------------------------------------------------------------

class FpwFitter:
    """Fixed Partial Waves Fitter — all data resident on GPU."""

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

    # ---- class-method constructors ---------------------------------------

    @classmethod
    def from_M(
        cls,
        F_data:   np.ndarray,
        w_data:   np.ndarray,
        B_data:   np.ndarray,
        M:        np.ndarray,
        N_b:      float,
        purity:   float = 1.0,
    ) -> "FpwFitter":
        """
        Create a fitter from a pre-computed overlap matrix M.

        All data is uploaded to GPU once and kept there permanently.
        """
        lib = _load_lib()

        n_data = int(F_data.shape[0])
        n_proj = int(F_data.shape[1])
        n_comp = int(F_data.shape[2])

        F_data = np.ascontiguousarray(F_data)
        w_data = np.ascontiguousarray(w_data)
        B_data = np.ascontiguousarray(B_data)
        M      = np.ascontiguousarray(M)

        p_Fd = _ffi.cast("const double *", _ffi.from_buffer(F_data))
        p_wd = _ffi.cast("const double *", _ffi.from_buffer(w_data))
        p_Bd = _ffi.cast("const double *", _ffi.from_buffer(B_data))
        p_M  = _ffi.cast("const double *", _ffi.from_buffer(M))

        handle = _ffi.new("FpwFitter **")
        err = lib.fpw_create(
            n_data, n_proj, n_comp,
            p_Fd, p_wd, p_Bd, p_M,
            N_b, purity, handle,
        )
        if err != 0:
            msg = _ffi.string(lib.fpw_strerror(err)).decode()
            raise RuntimeError(f"fpw_create failed: {msg}")

        return cls(handle[0], lib, F_data, w_data, B_data, n_comp,
                   M=M.copy(), N_b=N_b)

    @classmethod
    def from_mc(
        cls,
        F_data:   np.ndarray,
        F_mc:     np.ndarray,
        w_data:   np.ndarray,
        w_mc:     np.ndarray,
        B_data:   np.ndarray,
        B_mc:     np.ndarray,
        purity:   float = 1.0,
        chunk_size: int = 100_000,
    ) -> "FpwFitter":
        """
        Create a fitter from raw MC data.

        Pre-computes M via NumPy (chunked), then uploads all data to GPU.
        F_mc is NOT kept after creation.
        """
        import time
        from .compute_m import compute_M

        t0 = time.perf_counter()
        M, N_b = compute_M(F_mc, w_mc, B_mc, chunk_size)
        t1 = time.perf_counter()
        print(f"  M pre-compute: {t1-t0:.3f}s  "
              f"(n_mc={F_mc.shape[0]}, n_comp={F_mc.shape[2]})")

        return cls.from_M(F_data, w_data, B_data, M, N_b, purity)

    @classmethod
    def from_mc_file(
        cls,
        F_data:     np.ndarray,
        w_data:     np.ndarray,
        B_data:     np.ndarray,
        F_mc_file:  str | Path,
        w_mc:       np.ndarray,
        B_mc:       np.ndarray,
        F_mc_shape: tuple[int, int, int],
        purity:     float = 1.0,
        chunk_size: int = 100_000,
    ) -> "FpwFitter":
        """Create a fitter from F_mc stored as a raw-binary memmap file."""
        from .compute_m import compute_M_mmap
        M, N_b = compute_M_mmap(F_mc_file, F_mc_shape, w_mc, B_mc, chunk_size)
        return cls.from_M(F_data, w_data, B_data, M, N_b, purity)

    # ---- properties -------------------------------------------------------

    @property
    def n_comp(self) -> int:
        return self._n_comp

    @property
    def N_s(self) -> float:
        return self._lib.fpw_get_N_s(self._handle)

    @property
    def N_b(self) -> float:
        return self._lib.fpw_get_N_b(self._handle)

    # ---- accessors --------------------------------------------------------

    def get_M(self) -> tuple[np.ndarray, float]:
        """Return (M, N_b)."""
        if self._M is not None:
            return self._M.copy(), self._N_b
        raise AttributeError("M not available")

    def save_M(self, path: str | Path):
        """Save M and N_b to a .npz file."""
        M, N_b = self.get_M()
        np.savez(str(path), M=M, N_b=N_b)

    @classmethod
    def load_M(
        cls,
        path: str | Path,
        F_data: np.ndarray,
        w_data: np.ndarray,
        B_data: np.ndarray,
        purity: float = 1.0,
    ) -> "FpwFitter":
        """Load M from a .npz file and create a fitter."""
        data = np.load(str(path))
        M = data['M']
        N_b = float(data['N_b'])
        return cls.from_M(F_data, w_data, B_data, M, N_b, purity)

    # ---- evaluate ---------------------------------------------------------

    def evaluate(
        self,
        c: np.ndarray,
        return_P: bool = False,
    ) -> tuple[float, np.ndarray] | tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate -log L and its gradient.  All computation on GPU.

        Parameters
        ----------
        c : (n_comp,) complex128
        return_P : bool  — also return P_i values

        Returns
        -------
        nll : float
        grad : (n_comp,) complex128  — d(-ln L)/d(c*)
        P_data : (n_data,) float64  (only if return_P=True)
        """
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

        err = self._lib.fpw_evaluate(
            self._handle, p_cr, p_ci, p_nll, p_gr, p_gi, p_P,
        )
        if err != 0:
            msg = _ffi.string(self._lib.fpw_strerror(err)).decode()
            raise RuntimeError(f"fpw_evaluate failed: {msg}")

        grad.real[:] = gr
        grad.imag[:] = gi

        if return_P:
            return nll_out[0], grad, P_data
        return nll_out[0], grad

    # ---- cleanup ----------------------------------------------------------

    def __del__(self):
        try:
            if hasattr(self, '_handle') and self._handle is not None and self._handle != _ffi.NULL:
                self._lib.fpw_destroy(self._handle)
                self._handle = None
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_lib_cache = None


def _load_lib():
    """Load (or auto-build) the shared library, cached on first call."""
    global _lib_cache
    if _lib_cache is not None:
        return _lib_cache
    lib_path = ensure_lib()
    _lib_cache = _ffi.dlopen(str(lib_path))
    return _lib_cache
