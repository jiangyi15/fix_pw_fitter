"""Relativistic Breit-Wigner with running width (BWR).

Based on the TFPWA reference implementation (``breit_wigner.py``).
The propagator denominator is::

    BW(m) = 1 / (m₀² - m² - i·m₀·Γ(m))

    Γ(m) = g₀ · (q/q₀)^{2L+1} · (m₀/m) · B'_L(q,q₀,d)²

where q is the breakup momentum and B'_L is the Blatt-Weisskopf
barrier factor for orbital angular momentum L.

YAML usage::

    particle:
      rho(770):
        mass: 0.775
        width: 0.149
        model: BWR
        L: 1
        daug2Mass: 0.13957039
        daug3Mass: 0.1349768
"""

import numpy as np
from .base import BaseModel, register_model

# ── Helper functions (shared with GS_rho) ──────────────────────


def _two_body_cm_mom(m, m1, m2):
    """Real breakup momentum, 0 below threshold."""
    s = m ** 2
    m12s = (m1 + m2) ** 2
    m12d = (m1 - m2) ** 2
    p2 = (s - m12s) * (s - m12d)
    return np.sqrt(np.where(p2 > 0, p2, 0)) / (2 * m)


def _bprime_poly(L, z):
    """Blatt-Weisskopf barrier polynomial at order *L*."""
    coeff = {
        0: [1.0],
        1: [1.0, 1.0],
        2: [1.0, 3.0, 9.0],
        3: [1.0, 6.0, 45.0, 225.0],
        4: [1.0, 10.0, 135.0, 1575.0, 11025.0],
        5: [1.0, 15.0, 315.0, 6300.0, 99225.0, 893025.0],
    }
    c = coeff.get(L, [1.0])
    val = np.zeros_like(z)
    for ci in c:
        val = val * z + ci
    return val


# ── Model class ─────────────────────────────────────────────────


@register_model("BWR")
class BWRModel(BaseModel):
    """Relativistic Breit-Wigner with running width.

    The ``gamma(m)`` return value encodes the running width such that::

        bw_dom = m₀² - m² - i·m₀·g₀·gamma(m)
               = m₀² - m² - i·m₀·Γ(m)

    which matches the BWR denominator.

    Parameters from YAML config:
        mass          — pole mass m₀
        width         — nominal width g₀
        L             — orbital angular momentum (default 0)
        d             — impact parameter in GeV⁻¹ (default 3.0)
        daug2Mass     — first daughter mass (required if no decay tree)
        daug3Mass     — second daughter mass (required if no decay tree)
    """

    def get_gamma_defaults(self):
        return [float(self.kwargs.get("width", 0.1))]

    def gamma(self, m):
        M  = float(self.kwargs.get("M",  self.kwargs.get("mass",  0.775)))
        L  = int(self.kwargs.get("L", 0))
        d  = float(self.kwargs.get("d", 3.0))

        # Breakup momenta — use decay tree masses (consistent with TFPWA)
        if self._parent and self._parent._decays:
            d1 = float(self._parent._decays[0].outs[0].mass)
            d2 = float(self._parent._decays[0].outs[1].mass)
            q  = _two_body_cm_mom(m, d1, d2)
            q0 = _two_body_cm_mom(M, d1, d2)
        else:
            raise ValueError(f"BWR model '{self.name}': no decay tree available for q0")

        # Running width factor: (q/q₀)^{2L+1} · (m₀/m) · B'_L(q,q₀,d)²
        qq0 = (q / q0) ** (2 * L + 1)
        mm0 = M / m
        bp2 = _bprime_poly(L, (q0 * d) ** 2) / _bprime_poly(L, (q * d) ** 2)

        gamma_m = qq0 * mm0 * bp2
        return [gamma_m]
