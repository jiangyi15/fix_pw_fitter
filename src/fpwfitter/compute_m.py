"""
compute_m.py  –  Overlap matrix M and background normalisation N_b.

Uses chunked NumPy computation (supports mmap) so that F_mc
never needs to reside fully in RAM.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Public API
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
    F_mc_shape:   tuple[int, int, int],
    w_mc:         np.ndarray,
    B_mc:         np.ndarray,
    chunk_size:   int = 100_000,
) -> tuple[np.ndarray, float]:
    """
    Compute M from a raw-binary memapped file on disk.

    The file must contain ``complex128`` data in C-order with the given
    shape ``(n_mc, n_proj, n_comp)``.
    """
    F_mc = np.memmap(str(F_mc_path), dtype=np.complex128,
                     mode='r', shape=F_mc_shape)
    return compute_M(F_mc, w_mc, B_mc, chunk_size)
