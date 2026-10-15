"""`polarised` backend — the spin-density (``polarised_pwa``) backend.

Wraps the rho-aware :class:`~tabpwa.numpy_pwa.NumpyPolarisedPWA` kernel:
params ``{ck, m0, g0, scalar}`` with the intensity

    P(e) = sum_{j,j'} rho[j,j'] * A_j(e) * conj(A_j'(e)),
    rho  = I + sum_l p_l * G_l  (trace-P Hermitian generators),

``scalar`` being the fitted polarisation parameters.  All-zero ``scalar``
reproduces the plain projection-sum ``pwa`` intensity exactly.  The
kernel returns the raw ``dQ/drho`` matrix; this wrapper contracts it
with the generators into the per-parameter ``scalar`` gradients.

Alternatively (``base=<spec>``) the rho contraction is realised through a
Cholesky factorisation and a nested per-slot-ck PWA kernel: with
``rho = L·Lᴴ`` the same intensity is obtained from the per-slot couplings
``ck''[m, j, k] = L[j, m]·ck[k]`` fed to the ``numpy_pslot`` base (whose
extra slot axis sums ``L·Lᴴ`` back into ``rho``).  This path is
mathematically identical to the direct one and is the reference for the
CUDA expansion.
"""
import numpy as np

from .core import ComputeBackend, register_backend


def _expand_polarised_config(kc):
    """P-major duplication of the polarised kernel config for the per-slot base.

    The polarised config already owns ``n_proj = P`` projections
    (``n_wave = P·N``).  One extra copy per projection turns the entry
    layout into ``e'' = m·(P·N) + j·N + k`` (``n_wave'' = P²·N``) while
    ``n_proj`` stays ``P`` — exactly the slot layout the per-slot-ck
    ``NumpyPWAPslot`` kernel expects for ``ck''[m, j, k] = L[j, m]·ck[k]``.
    """
    from tabpwa.base_model import _projection_duplicate
    P = int(kc["n_proj"])
    return _projection_duplicate(kc, P)


@register_backend("polarised", model="polarised_pwa")
@register_backend("default", model="polarised_pwa")
class PolarisedBackend(ComputeBackend):
    """numpy_pwa + the spin-density-matrix intensity contraction.

    Parameters
    ----------
    kernel_config : dict
        Config from the ``polarised_pwa`` model (``spin_density`` carries
        ``n_j`` and the parameter names).
    base : str or ComputeBackend, optional
        When given, the rho contraction is realised through the Cholesky
        ``ck''[m, j, k] = L[j, m]·ck[k]`` fed to this nested per-slot PWA
        backend (spec resolved via :func:`create_backend` with
        ``model="pwa"`` on the :func:`_expand_polarised_config`'d config).
        ``None`` (default) uses the direct :class:`NumpyPolarisedPWA`
        path.
    """

    def __init__(self, kernel_config, base=None, **kw):
        from tabpwa.numpy_pwa import NumpyPolarisedPWA
        from tabpwa.spin_density import generators

        sd = kernel_config.get("spin_density") or {}
        self._n_j = int(sd.get("n_j", 0))
        self._gen = generators(self._n_j)

        self.kernel_config = kernel_config
        self._P = int(kernel_config["n_proj"])
        self._N = int(np.asarray(kernel_config["matrix_angle"]).shape[1]) \
            // self._P

        if base is None:
            self.kernel = NumpyPolarisedPWA(kernel_config)
            self.base = None
        else:
            from . import create_backend
            expanded = _expand_polarised_config(kernel_config)
            self.base = create_backend(base, expanded, model="pwa")
            # The direct rho-aware kernel is not used on the pslot path;
            # expose the nested kernel for callers that reach for
            # ``.kernel`` (mirrors the composing-backend convention).
            self.kernel = self.base.kernel

    # -- data -------------------------------------------------------------
    def load_data(self, data_np):
        src = self.base if self.base is not None else self.kernel
        return src.load_data(data_np)

    # -- forward + gradients ----------------------------------------------
    def compute(self, params, data_handle, norm=None, return_p=True):
        """NLL/grads with ``P = sum rho[j,j'] A_j conj(A_j')``.

        Returns grads ``{ck, m0, g0, scalar, norm}`` — ``scalar`` is the
        gradient w.r.t. the polarisation parameters.
        """
        if self.base is None:
            Q, grads, P = self.kernel.compute(params, data_handle, norm=norm,
                                              return_p=True)
            drho = grads.pop("drho")
            grads["scalar"] = np.array([float(np.real(np.sum(g * drho)))
                                        for g in self._gen])
            if norm is not None:
                # native dnorm from the kernel (no host-array slot)
                grads["norm"] = float(self.kernel._last_dnorm)
            return Q, grads, (P if return_p else None)

        return self._compute_pslot(params, data_handle, norm, return_p)

    def _compute_pslot(self, params, data_handle, norm, return_p):
        P = self._P
        N = self._N
        spin = np.asarray(params["scalar"], dtype=float)
        ck = np.asarray(params["ck"], dtype=complex)
        if len(ck) != N:
            raise ValueError(
                f"polarised (numpy_pslot base): ck length {len(ck)} != "
                f"n_wave/n_proj = {N}")

        # rho = I + Σ_l p_l·G_l = L·Lᴴ (Cholesky, L lower triangular).
        rho = np.eye(P, dtype=complex) + np.einsum(
            "l,lij->ij", spin, np.asarray(self._gen))
        L = np.linalg.cholesky(rho)

        # per-slot couplings ck''[m, j, k] = L[j, m]·ck[k]; the base sums
        # over (m, j, k) so that Σ_m |Ã_m|² = Σ_{j,j'} rho[j,j'] A_j conj(A_j').
        ck2 = np.einsum("jm,k->mjk", L, ck).reshape(-1)

        Q, g2, Pe = self.base.compute(
            {"ck": ck2, "m0": params["m0"], "g0": params["g0"]},
            data_handle, norm=norm, return_p=return_p)

        # The base kernel returns holomorphic gradients ∂Q/∂z (its own
        # convention: dQ = 2·Re(∂Q/∂z·dz)).  g[m,j,k] = ∂Q/∂ck''[m,j,k].
        g = np.asarray(g2["ck"]).reshape(P, P, N)
        grads = {"m0": g2["m0"], "g0": g2["g0"]}
        # ∂Q/∂ck[k] = Σ_{m,j} g[m,j,k]·L[j,m]     (ck''[m,j,k] = L[j,m]·ck[k])
        grads["ck"] = np.einsum("mjk,jm->k", g, L)
        # G_L[j,m] = ∂Q/∂L[j,m] = Σ_k g[m,j,k]·ck[k]
        G_L = np.einsum("mjk,k->jm", g, ck)

        # Chain L = cholesky(rho) back to rho.  From drho = dL·Lᴴ + L·dLᴴ
        # (L lower) one gets dL = L·Φ(L⁻¹·drho·L⁻ᴴ) with Φ the lower
        # triangle incl. half the diagonal; its adjoint gives (with
        # B = G_Lᵀ·L, P = Φ(B)):
        #   G_rho = K + Kᴴ,  K = Liᵀ·Φ(B)ᵀ·conj(Li),  Li = L⁻¹
        # so that dQ = Σ_{a,b} G_rho[a,b]·drho[a,b] + c.c. — identical to
        # the direct kernel's raw dQ/drho (Hermitian).
        Li = np.linalg.inv(L)
        B = G_L.T @ L
        Pmat = np.tril(B, -1) + 0.5 * np.diag(np.diag(B))
        K = Li.T @ Pmat.T @ np.conj(Li)
        G_rho = K + np.conj(K.T)
        grads["scalar"] = np.array([float(np.real(np.sum(G_rho * gg)))
                                    for gg in self._gen])

        if norm is not None:
            grads["norm"] = float(g2.get("norm", self.base.kernel._last_dnorm))
        return Q, grads, (Pe if return_p else None)

    def free(self):
        base = getattr(self, "base", None)
        if base is not None:
            base.free()

    def __del__(self):
        self.free()
