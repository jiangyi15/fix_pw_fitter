"""
Rho-Omega interference lineshape: GS_rho × BW_omega.

Fixed shape: ``_fixed_shape(m)`` returns ``GS(m; m_ρ, Γ_ρ) · BW(m; m_ω, Γ_ω)``,
the product of a Gounaris-Sakurai rho(770) propagator and a constant-width
Breit-Wigner omega(782) propagator.

All parameters are fixed at YAML config values via ``_FixMassWidthTransform``
— nothing is fitted.

YAML usage::

    particle:
      rho_omega:
        mass: 0.775
        width: 0.149
        model: RhoOmega
        L: 1
        daug2Mass: 0.13957039
        daug3Mass: 0.1349768
        omega_mass: 0.78266
        omega_width: 0.00868
"""
import numpy as np
from .models_builtin import FixedShapeModel, register_model
from .gs_rho_model import _two_body_cm_mom, _gamma_run, _fs_fun

_DEF_M2 = 0.13957039  # π⁺
_DEF_M3 = 0.1349768   # π⁰


@register_model("RhoOmega")
class RhoOmegaModel(FixedShapeModel):
    """GS ρ(770) × BW ω(782) interference lineshape.

    ``fixed_shape(m)`` returns the product GS(m)·BW(m).
    """

    def fixed_shape(self, m):
        M     = float(self.kwargs.get("M",  self.kwargs.get("mass",  0.775)))
        g0    = float(self.kwargs.get("width", 0.149))
        L     = int(self.kwargs.get("L", 1))
        d     = float(self.kwargs.get("d", 3.0))
        m2    = float(self.kwargs.get("daug2Mass", _DEF_M2))
        m3    = float(self.kwargs.get("daug3Mass", _DEF_M3))
        m_om  = float(self.kwargs.get("omega_mass", 0.78266))
        w_om  = float(self.kwargs.get("omega_width", 0.00868))

        # ── Breakup momenta ─────────────────────────────────────
        if self._parent and self._parent._decays:
            d1 = float(self._parent._decays[0].outs[0].mass)
            d2 = float(self._parent._decays[0].outs[1].mass)
            q  = _two_body_cm_mom(m, d1, d2)
            q0 = _two_body_cm_mom(M, d1, d2)
        else:
            q  = _two_body_cm_mom(m, m2, m3)
            q0 = _two_body_cm_mom(M, m2, m3)

        # ── GS rho denominator ──────────────────────────────────
        gamma_gs = _gamma_run(m, 1.0, q, q0, L, M, d)
        fs_val   = _fs_fun(m ** 2, M ** 2, 1.0, _DEF_M2, _DEF_M3)
        d_gs = (M ** 2 - m ** 2) - 1j * M * g0 * (gamma_gs + 1j * fs_val / M)

        # ── BW omega denominator ─────────────────────────────────
        d_bw = (m_om ** 2 - m ** 2) - 1j * m_om * w_om

        # ── Amplitude = GS · BW ─────────────────────────────────
        return 1.0 / d_gs * 1.0 / d_bw
