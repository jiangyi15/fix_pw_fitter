"""`polarised` backend — numpy_pwa + spin-density contraction.

Wraps :class:`~tabpwa.numpy_pwa.NumpyPWABackend` (registered as
``numpy_pwa``): the per-projection amplitudes and the plain kernels are
unchanged; only the intensity contraction gains the fitted spin-density
matrix ``rho = I + sum_l p_l * G_l`` built from the model's ``spin``
parameters (the polarisation vector ``px/py/pz`` for two helicity
states, generic ``p1..p{P*P-1}`` beyond).

The spin parameters are model-declared scalars on ``polarised_pwa``
(registered in ``amp_model.py``) — nothing to configure beyond
``amp_model: polarised_pwa``; fixing/freeing them is the constraint
system's job.

With all spin parameters at zero, ``rho = I`` and the intensity is
exactly the plain ``numpy_pwa`` projection sum.
"""
import numpy as np

from .core import ComputeBackend, register_backend
from .numpy_pwa_backend import NumpyPWABackend


@register_backend("polarised", model="polarised_pwa")
@register_backend("default", model="polarised_pwa")
class PolarisedBackend(ComputeBackend):
    """numpy_pwa + the spin-density-matrix intensity contraction."""

    def __init__(self, kernel_config, base="numpy_pwa", **kw):
        """Args:
            kernel_config: config from the ``polarised_pwa`` model
                (``spin_density`` carries ``n_j`` and the parameter
                names).
            base: base backend spec (default ``numpy_pwa``).
        """
        from tabpwa.backends import create_backend
        from tabpwa.spin_density import generators

        self.base = create_backend(base, kernel_config, **kw)
        assert isinstance(self.base, NumpyPWABackend), (
            "PolarisedBackend currently requires the numpy_pwa base")
        sd = kernel_config.get("spin_density") or {}
        self.spin_names = list(sd.get("names", []))
        self._n_j = int(sd.get("n_j", 0))
        self._gen = generators(self._n_j)

    # -- delegates -------------------------------------------------------

    def load_data(self, data):
        return self.base.load_data(data)

    def free(self):
        self.base.free()

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    # -- compute -----------------------------------------------------------

    def compute(self, params, data_handle, norm=None, return_p=True):
        """NLL/grads with ``P = sum rho[j,j'] A_j conj(A_j')``.

        Returns grads ``{ck, m0, g0, spin}`` — ``spin`` is the gradient
        w.r.t. the polarisation parameters.
        """
        from tabpwa.spin_density import rho_matrix

        spin = np.asarray(params["scalar"], dtype=float)
        # P = the top's spin-state count (== len(spins)); the parameter
        # count is P^2 - 1 (trace-P Hermitian basis)
        rho = rho_matrix(spin, self._n_j)
        Q, grads, P = self.base.compute(params, data_handle, norm=norm,
                                        return_p=return_p, rho=rho)
        drho = grads.pop("drho")
        grads["scalar"] = np.array([float(np.real(np.sum(g * drho)))
                                    for g in self._gen])
        return Q, grads, P
