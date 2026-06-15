"""Flatté parametrisation for coupled-channel resonances.

Based on the TFPWA reference implementation (``amp/flatte.py``).
The propagator denominator is::

    R(m) = 1 / (m₀² - m² - i·m₀·Σᵢ gᵢ · qᵢ(m) / m)

where qᵢ(m) is the complex breakup momentum for channel i::

    qᵢ =  √((m²-(mᵢ₁+mᵢ₂)²)(m²-(mᵢ₁-mᵢ₂)²)) / (2m)    (above threshold)
    qᵢ =  i·√(|(m²-(mᵢ₁+mᵢ₂)²)(m²-(mᵢ₁-mᵢ₂)²)|) / (2m)  (below threshold)

YAML usage::

    particle:
      f0(980):
        mass: 0.99
        model: FlatteC
        mass_list: [[0.13957, 0.13957], [0.49368, 0.49368]]
        g_0: 0.3
        g_1: 0.6
"""

import numpy as np
from .base import BaseModel, register_model


def _breakup_momentum(m, m1, m2):
    """Complex breakup momentum for a two-body decay at mass *m*.

    Returns real above threshold, purely imaginary below threshold.
    """
    s = m ** 2
    mabp = m1 + m2
    mabm = m1 - m2
    p2 = (s - mabp * mabp) * (s - mabm * mabm) / (4.0 * s)
    return np.emath.sqrt(p2)


@register_model("FlatteC")
class FlatteCModel(BaseModel):
    """Flatté parametrisation with multiple coupled channels.

    The ``gamma(m)`` returns a list of ``qᵢ(m)/m`` (one per channel).
    The framework computes::

        g_bw = Σᵢ g₀ᵢ · gammaᵢ(m)
        bw_dom = m₀² - m² - i·m₀·g_bw

    which matches the Flatté denominator.

    Required YAML arg:
        mass_list: [[m11, m12], [m21, m22], ...]
            List of daughter-mass pairs, one per channel.

    Optional YAML args:
        g_i:  coupling constant for channel i (default 0.1)
    """

    def get_gamma_count(self):
        return len(self.kwargs["mass_list"])

    def get_gamma_name(self):
        return [f"{self.name}_g{i}" for i in range(self.get_gamma_count())]

    def get_gamma_defaults(self):
        return [
            float(self.kwargs.get(f"g_{i}", 0.1))
            for i in range(self.get_gamma_count())
        ]

    def gamma(self, m):
        mass_list = self.kwargs["mass_list"]
        out = []
        for ch in mass_list:
            m1, m2 = ch
            q = _breakup_momentum(m, m1, m2)
            out.append(q / m)
        return out
