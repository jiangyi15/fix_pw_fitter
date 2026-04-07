"""
fpwfitter  –  Fixed Partial Waves Fitter  (GPU-accelerated)

All data (F_data, w_data, B_data) is uploaded to GPU at creation and
kept there permanently.  Evaluation runs entirely on GPU.

Usage::

    # Mode A: from raw MC data (M pre-computed via NumPy)
    fitter = FpwFitter.from_mc(
        F_data, F_mc, w_data, w_mc, B_data, B_mc, purity=0.8
    )

    # Mode B: from pre-computed M (no MC data needed)
    fitter = FpwFitter.from_M(
        F_data, w_data, B_data, M, N_b, purity=0.8
    )

    # Evaluate  (all on GPU, only NLL + gradient copied back)
    nll, grad = fitter.evaluate(c)
"""

from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from cffi import FFI

# ---------------------------------------------------------------------------
# CFFI setup
# ---------------------------------------------------------------------------

_src_dir = Path(__file__).parent
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

# ---------------------------------------------------------------------------
# Shared library
# ---------------------------------------------------------------------------

def _get_lib_path() -> Path:
    return _src_dir / "libfpwfitter.so"


def _ensure_compiled():
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
        "-o", str(lib),
    ]
    subprocess.check_call(cmd)
    print(f"  → {lib}")


# ---------------------------------------------------------------------------
# M pre-computation (NumPy, chunked)
# ---------------------------------------------------------------------------

def compute_M(
    F_mc:       np.ndarray,
    w_mc:       np.ndarray,
    B_mc:       np.ndarray,
    chunk_size: int = 100_000,
) -> tuple[np.ndarray, float]:
    """
    Compute the overlap matrix M and N_b from MC data.

    Parameters
    ----------
    F_mc : (n_mc, n_proj, n_comp) complex128
    w_mc : (n_mc,) float64
    B_mc : (n_mc,) float64
    chunk_size : events per chunk

    Returns
    -------
    M  : (n_comp, n_comp) complex128
    N_b : float
    """
    n_mc, n_proj, n_comp = F_mc.shape
    M = np.zeros((n_comp, n_comp), dtype=np.complex128)
    N_b = 0.0
    n_chunks = (n_mc + chunk_size - 1) // chunk_size

    for ic in range(n_chunks):
        off = ic * chunk_size
        cur = min(chunk_size, n_mc - off)

        F_chunk = F_mc[off:off + cur]
        w_chunk = w_mc[off:off + cur]
        B_chunk = B_mc[off:off + cur]

        for j in range(n_proj):
            F_j = F_chunk[:, j, :]
            M += np.dot(np.conj(F_j).T, w_chunk[:, None] * F_j)

        N_b += np.dot(w_chunk, B_chunk)

    return M, N_b


def compute_M_mmap(
    F_mc_path:    str | Path,
    F_mc_shape:   tuple,
    w_mc:         np.ndarray,
    B_mc:         np.ndarray,
    chunk_size:   int = 100_000,
) -> tuple[np.ndarray, float]:
    """Compute M from an mmapped .npy file on disk."""
    F_mc = np.load(str(F_mc_path), mmap_mode='r')
    return compute_M(F_mc, w_mc, B_mc, chunk_size)


# ---------------------------------------------------------------------------
# FpwFitter class
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

    @classmethod
    def from_M(
        cls,
        F_data:   np.ndarray,
        w_data:   np.ndarray,
        B_data:   np.ndarray,
        M:        np.ndarray,
        N_b:      float,
        purity:   float = 1.0,
    ) -> FpwFitter:
        """
        Create a fitter from a pre-computed overlap matrix M.

        All data is uploaded to GPU once and kept there permanently.
        """
        _ensure_compiled()
        lib = _ffi.dlopen(str(_get_lib_path()))

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
    ) -> FpwFitter:
        """
        Create a fitter from raw MC data.

        Pre-computes M via NumPy (chunked), then uploads all data to GPU.
        F_mc is NOT kept after creation.
        """
        import time
        t0 = time.perf_counter()
        M, N_b = compute_M(F_mc, w_mc, B_mc, chunk_size)
        t1 = time.perf_counter()
        print(f"  M pre-compute: {t1-t0:.3f}s  (n_mc={F_mc.shape[0]}, n_comp={F_mc.shape[2]})")

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
        F_mc_shape: tuple,
        purity:     float = 1.0,
        chunk_size: int = 100_000,
    ) -> FpwFitter:
        """Create a fitter from F_mc stored in a .npy file (mmap'd)."""
        M, N_b = compute_M_mmap(F_mc_file, F_mc_shape, w_mc, B_mc, chunk_size)
        return cls.from_M(F_data, w_data, B_data, M, N_b, purity)

    # ---- Properties ----

    @property
    def n_comp(self) -> int:
        return self._n_comp

    @property
    def N_s(self) -> float:
        return self._lib.fpw_get_N_s(self._handle)

    @property
    def N_b(self) -> float:
        return self._lib.fpw_get_N_b(self._handle)

    # ---- Accessors ----

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
    ) -> FpwFitter:
        """Load M from a .npz file and create a fitter."""
        data = np.load(str(path))
        M = data['M']
        N_b = float(data['N_b'])
        return cls.from_M(F_data, w_data, B_data, M, N_b, purity)

    # ---- Evaluate ----

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

    def __del__(self):
        try:
            if hasattr(self, '_handle') and self._handle is not None and self._handle != _ffi.NULL:
                self._lib.fpw_destroy(self._handle)
                self._handle = None
        except Exception:
            pass
