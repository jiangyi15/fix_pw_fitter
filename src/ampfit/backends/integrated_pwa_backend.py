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

    def __init__(self, kernel_config, base="cuda_v4_pwa", mc_batch=5000):
        """Hyper backend: Gram norm (pure PWA) + base backend for data NLL.

        Args:
            kernel_config: kernel config dict.
            base: backend spec for the data NLL.
            mc_batch: phase-space (MC) batch size for the streamed Gram
                build — per batch the phsp is loaded to the base kernel,
                Gram-reduced and freed (memory bound by one batch).
                Larger batches (e.g. 20000–50000) build D faster; 5000
                keeps the transient memory minimal.
        """
        self._kernel_config = kernel_config
        self.mc_batch = int(mc_batch)
        self._gram = _GramPWA(kernel_config)
        if isinstance(base, ComputeBackend):
            self.base = base
        else:
            self.base = create_backend(base, kernel_config)

    # ── data loading ────────────────────────────────────────────────────
    # The base (GPU data) handle is created LAZILY only when a data NLL /
    # per-event P is requested; the phsp norm path streams per-batch
    # kernel.load() → compute_gram() → free() so the whole phsp never
    # resides on the device at once.
    class _Bundle:
        def __init__(self, data_np):
            self.handle = None          # base handle, lazily created
            self.data = dict(data_np)
            self.m0 = self.g0 = None
            self.D = None

        def free(self):
            if self.handle is not None and hasattr(self.handle, "free"):
                self.handle.free()
                self.handle = None
            self.D = None

    def load_data(self, data_np):
        return self._Bundle(data_np)

    def _get_data_handle(self, bundle):
        if bundle.handle is None:
            bundle.handle = self.base.load_data(bundle.data)
        return bundle.handle

    # ── Gram norm (phsp): streamed in batches ────────────────────────────
    def _ensure_gram(self, bundle, m0, g0, batch=None):
        if batch is None:
            batch = self.mc_batch
        if bundle.D is not None and np.array_equal(m0, bundle.m0) \
                and np.array_equal(g0, bundle.g0):
            return
        base_kernel = getattr(self.base, "kernel", None)
        gpu_gram = getattr(base_kernel, "compute_gram", None)
        d = bundle.data
        ne = d["mass"].shape[0]
        w = d.get("weight")

        D = None
        for b0 in range(0, ne, batch):
            b1 = min(b0 + batch, ne)
            sub = {k: v[b0:b1] for k, v in d.items()
                   if isinstance(v, np.ndarray)}
            if gpu_gram is not None:
                # streaming: load the batch, gram it, free it
                h = self.base.load_data(sub)
                try:
                    M = gpu_gram(h, m0, g0)
                finally:
                    h.free()
            else:
                M = self._gram.gram(sub, m0, g0,
                                    weight=sub.get("weight"))
            D = M if D is None else D + M

        bundle.D = (D + D.conj().T) / 2.0
        bundle.m0 = np.asarray(m0).copy()
        bundle.g0 = np.asarray(g0).copy()

    # ── ComputeBackend interface ─────────────────────────────────────────
    def compute(self, params, data_handle, norm=None, return_p=True):
        if norm is not None or return_p:
            # data NLL (norm given) or per-event P (plot path) → base
            h = self._get_data_handle(data_handle)
            Q, grads, P = self.base.compute(params, h,
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
