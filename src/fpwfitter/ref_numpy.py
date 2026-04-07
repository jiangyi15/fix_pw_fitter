"""
ref_numpy.py  –  Pure NumPy reference implementation.

Used as a fallback when CUDA is unavailable, or for validation.
Identical API to the CUDA backend — just slower.
"""

from __future__ import annotations

import numpy as np
from .compute_m import compute_M, compute_M_mmap


class NumpyFitter:
    """Pure NumPy Fixed Partial Waves Fitter (CPU-only)."""

    def __init__(self, F_data, w_data, B_data, M, N_b, purity):
        self._F_data = np.asarray(F_data, dtype=np.complex128)
        self._w_data = np.asarray(w_data, dtype=np.float64)
        self._B_data = np.asarray(B_data, dtype=np.float64)
        self._M      = np.asarray(M, dtype=np.complex128)
        self._N_b    = float(N_b)
        self._purity = float(purity)
        self._n_comp = F_data.shape[2]
        self._Ns     = 0.0

    @classmethod
    def from_M(cls, F_data, w_data, B_data, M, N_b, purity=1.0):
        return cls(F_data, w_data, B_data, M, N_b, purity)

    @classmethod
    def from_mc(cls, F_data, F_mc, w_data, w_mc, B_data, B_mc,
                purity=1.0, chunk_size=100_000):
        import time
        t0 = time.perf_counter()
        M, N_b = compute_M(F_mc, w_mc, B_mc, chunk_size)
        t1 = time.perf_counter()
        print(f"  M pre-compute: {t1-t0:.3f}s  "
              f"(n_mc={F_mc.shape[0]}, n_comp={F_mc.shape[2]})")
        return cls(F_data, w_data, B_data, M, N_b, purity)

    @classmethod
    def from_mc_file(cls, F_data, w_data, B_data,
                     F_mc_file, w_mc, B_mc, F_mc_shape,
                     purity=1.0, chunk_size=100_000):
        M, N_b = compute_M_mmap(F_mc_file, F_mc_shape, w_mc, B_mc, chunk_size)
        return cls(F_data, w_data, B_data, M, N_b, purity)

    @property
    def n_comp(self):
        return self._n_comp

    @property
    def N_s(self):
        return self._Ns

    @property
    def N_b(self):
        return self._N_b

    def get_M(self):
        return self._M.copy(), self._N_b

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
        c = np.asarray(c, dtype=np.complex128).ravel()
        assert c.shape == (self._n_comp,)

        F   = self._F_data
        w   = self._w_data
        B   = self._B_data
        M   = self._M
        Nb  = self._N_b
        pur = self._purity

        # Forward
        A  = np.einsum('ijk,k->ij', F, c)          # (N, JP)
        S  = np.sum(np.abs(A) ** 2, axis=1)        # (N,)
        Ns = np.real(np.vdot(c, M @ c))
        if Ns < 1e-300: Ns = 1e-300
        if Nb < 1e-300: Nb = 1e-300

        P  = S / Ns * pur + B / Nb * (1 - pur)
        P  = np.maximum(P, 1e-300)
        nll = -np.dot(w, np.log(P))

        self._Ns = Ns

        # Gradient
        #   g_data[k] = sum_{i,j} conj(F[i,j,k]) * (w[i]/P[i]) * A[i,j]
        ratio = w / P                                         # (N,)
        G     = A * ratio[:, None]                            # (N, JP)
        g_data = np.einsum('ijk,ij->k', np.conj(F), G)       # (k,)

        #   dN_s/dc* = M @ c
        dNs_dc = M @ c                                        # (k,)

        #   S_corr = sum_i w[i] * S[i] / P[i]
        S_corr = np.sum(w * S / P)

        #   grad = -pur/Ns * g_data + pur/Ns^2 * dNs_dc * S_corr
        grad = -pur / Ns * g_data + pur / Ns**2 * dNs_dc * S_corr

        if return_P:
            return float(nll), grad, P.copy()
        return float(nll), grad
