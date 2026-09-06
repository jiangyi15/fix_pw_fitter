"""
integrated_pwa — Gram-matrix hyper backend for the projection-sum PWA.

Mirrors the legacy ``integrated`` backend but for the shared-ck pure-PWA
model (no time/mixing/scalars):

    P(e) = Σ_p |A_p(e)|² ,  A_p = Σ_k ck_k · a_{p,k}(e)

* the phase-space norm is pre-integrated into the wave Gram matrix

      D[k,k'] = Σ_e w_e Σ_p conj(a_{p,k}(e)) · a_{p,k'}(e)
      norm    = Σ_{k,k'} ck_k · conj(ck_k') · D[k,k']     (O(N²) / iter)

  built for the current m0/g0 (fp64 numpy reference),
* the data NLL is delegated to a base backend (default ``cuda_v4_pwa``).

Protocol matches ``IntegratedBackend`` (see fitter.get_nll_raw):
``compute(params, phsp_handle, norm=None, return_p=False)`` returns the
norm as Q and norm gradients ``{"ck": D@ck, "m0": 0, "g0": 0}`` (m0/g0 are
frozen at the Gram pre-integration, exactly like the legacy integrated
backend); ``compute(params, data_handle, norm=...)`` delegates to the base.
"""
import numpy as np

from .core import ComputeBackend, register_backend
from . import create_backend
from ampfit.integrated_pwa import IntegratedPWA as _GramPWA


@register_backend("integrated_pwa")
class IntegratedPWABackend(ComputeBackend):
    """Hyper backend: Gram norm (pure PWA) + base backend for data NLL."""

    def __init__(self, kernel_config, base="cuda_v4_pwa"):
        self._kernel_config = kernel_config
        self._gram = _GramPWA(kernel_config)
        if isinstance(base, ComputeBackend):
            self.base = base
        else:
            self.base = create_backend(base, kernel_config)

    # ── data loading ────────────────────────────────────────────────────
    class _Bundle:
        def __init__(self, handle, data_np):
            self.handle = handle
            self.data = dict(data_np)
            self.m0 = self.g0 = None
            self.D = None

        def free(self):
            if hasattr(self.handle, "free"):
                self.handle.free()
            self.D = None

    def load_data(self, data_np):
        h = self.base.load_data(data_np)
        return self._Bundle(h, data_np)

    # ── Gram norm (phsp) ────────────────────────────────────────────────
    def _ensure_gram(self, bundle, m0, g0):
        if bundle.D is not None and np.array_equal(m0, bundle.m0) \
                and np.array_equal(g0, bundle.g0):
            return
        d = dict(bundle.data)
        w = d.get("weight")
        bundle.D = self._gram.gram(d, m0, g0, weight=w)
        bundle.m0 = np.asarray(m0).copy()
        bundle.g0 = np.asarray(g0).copy()

    # ── ComputeBackend interface ─────────────────────────────────────────
    def compute(self, params, data_handle, norm=None, return_p=True):
        if norm is not None or return_p:
            # data NLL (norm given) or per-event P (plot path) → base
            Q, grads, P = self.base.compute(params, data_handle.handle,
                                            norm=norm, return_p=return_p)
            grads["m0"] = np.zeros_like(grads["m0"])
            grads["g0"] = np.zeros_like(grads["g0"])
            return Q, grads, P

        # fast Gram norm over the phsp bundle
        m0 = np.asarray(params["m0"])
        g0 = np.asarray(params["g0"])
        self._ensure_gram(data_handle, m0, g0)
        ck = np.asarray(params["ck"])
        norm_val = self._gram.norm(ck, data_handle.D)
        grads = {
            "ck": data_handle.D @ ck,          # dNorm/dRe(ck)=2Re(D@ck)
            "m0": np.zeros(len(m0)),
            "g0": np.zeros(len(g0)),
        }
        return float(norm_val), grads, None

    def free(self):
        try:
            self.base.free()
        except Exception:
            pass
