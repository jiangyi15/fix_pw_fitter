"""
Experimental exponential lineshape with CR k-interpolation.

The BW amplitude is parametrised as::

    A(m) = 1 / (m_0^2 - m^2 - i*m_0*Sum g_a(k)*Gamma_a(m))
         = exp(-k*(m^2 - m_0^2))

    g_a(k_b) = delta_{a,b}   (Kronecker delta at k-grid points)

See :class:`~ampfit.particle_model.interp_k_model.InterpKModel`
for the generic k-interpolation base class.

YAML usage::

    particle:
      sigma:
        mass: 0.5
        model: Exp
        k: 1.0               # initial k
        k_range: [0.1, 5.0]  # k min/max
        n_interp: 50         # CR interpolation points in k
"""

import numpy as np
from .interp_k_model import InterpKModel
from .base import register_model


@register_model("Exp")
class ExpModel(InterpKModel):
    """Exponential lineshape with CR-interpolated k.

    Amplitude: ``A(m) = exp(-k*(m^2 - m_0^2))``.
    The gamma function at each k_i gives exact amplitude at k = k_i,
    and CR weights blend between grid points.

    Parameters (from YAML config):
        mass        -- nominal mass (m_0)
        width       -- reference g_0 (default 1.0)
        k           -- initial/default k value (default 1.0)
        k_range     -- [k_min, k_max] (default [0.1, 5.0])
        n_interp    -- CR interpolation points (default 50)
    """

    def amplitude_k(self, m, k):
        r"""Physics amplitude: ``A(m) = exp(-k*(m^2 - m_0^2))``.

        The base :meth:`InterpKModel.gamma_k` derives the running-width
        gamma rows automatically.
        """
        m0 = float(self.kwargs.get("mass", 0.775))
        return np.exp(-k * (m ** 2 - m0 ** 2))
