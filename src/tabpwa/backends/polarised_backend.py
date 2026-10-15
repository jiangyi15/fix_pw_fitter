"""`polarised` backend — the spin-density (``polarised_pwa``) backend.

Thin registration wrapper around the rho-aware
:class:`~tabpwa.numpy_pwa.NumpyPolarisedPWA` kernel: params
``{ck, m0, g0, scalar}`` with the intensity

    P(e) = sum_{j,j'} rho[j,j'] * A_j(e) * conj(A_j'(e)),
    rho  = I + sum_l p_l * G_l  (trace-P Hermitian generators),

``scalar`` being the fitted polarisation parameters.  All-zero ``scalar``
reproduces the plain projection-sum ``pwa`` intensity exactly.  The
kernel returns the finished per-parameter gradients
``{ck, m0, g0, scalar}`` — ``scalar`` being dQ/dp_l, already contracted
with the generators.

Note: this contraction cannot be expressed through the *unmodified*
shared-ck projection-sum kernels (v4/numpy_pwa) — the per-projection
mixing weights ``L[j,m]`` of ``rho = L L^H`` differ per projection and
change every iteration, while the shared-ck interface applies ONE ``ck``
vector to every projection; ``L`` can never be baked into the config
arrays (they are fixed at ``load_data``).  A CUDA port therefore requires
a kernel-side change (e.g. the rho contraction in the reduce kernel);
until then this backend is the numpy reference.
Registered as ``"numpy_polarised"`` for the model — ``polarised_pwa``
has NO ``"default"`` backend: constructing a ``Fitter`` without an
explicit ``backend=`` fails, by design (the numpy reference must never
be picked up silently; pass ``backend="numpy_polarised"`` explicitly).
"""
from .core import ComputeBackend, register_backend


@register_backend("numpy_polarised", model="polarised_pwa")
class PolarisedBackend(ComputeBackend):
    """numpy_pwa + the spin-density-matrix intensity contraction."""

    def __init__(self, kernel_config, **kw):
        """Args:
            kernel_config: config from the ``polarised_pwa`` model
                (``spin_density`` carries ``n_j`` and the parameter
                names).
        """
        from tabpwa.numpy_pwa import NumpyPolarisedPWA
        self.kernel = NumpyPolarisedPWA(kernel_config)

    def load_data(self, data_np):
        return self.kernel.load_data(data_np)

    def compute(self, params, data_handle, norm=None, return_p=True):
        """NLL/grads with ``P = sum rho[j,j'] A_j conj(A_j')``.

        Returns grads ``{ck, m0, g0, scalar}`` (+ ``norm`` when the
        gradient w.r.t. the norm is available) — ``scalar`` is the
        gradient w.r.t. the polarisation parameters.
        """
        Q, grads, P = self.kernel.compute(params, data_handle, norm=norm,
                                          return_p=True)
        if norm is not None:
            # native dnorm from the kernel (no host-array slot)
            grads["norm"] = float(self.kernel._last_dnorm)
        return Q, grads, (P if return_p else None)

    def free(self):
        pass

    def __del__(self):
        self.free()
