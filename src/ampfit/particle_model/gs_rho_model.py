"""Gounaris-Sakurai parametrisation for the rho(770) resonance.

Based on the TFPWA reference implementation (``breit_wigner.py``).
The propagator denominator is::

    GS(m) = 1 / (m0² - m² + fs(m²) - i·m0·Γ(m))

where Γ(m) is the running width and fs(m²) is the GS mass-shift term.
The numerator is a constant (handled by the framework).

YAML usage::

    particle:
      rho(770):
        mass: 0.775
        width: 0.149
        model: GS_rho
        L: 1                       # orbital angular momentum
        daug2Mass: 0.13957039      # π⁺ mass
        daug3Mass: 0.1349768       # π⁰ mass
"""

import numpy as np
from .base import BaseModel, register_model

# ── Helper functions (NumPy translation of TF reference) ────────


def _two_body_cm_mom(m, m1, m2):
    """Relative momentum for a two-body decay at mass *m*.

    ``p = sqrt((m² - (m1+m2)²)(m² - (m1-m2)²)) / (2m)``,
    clamped to 0 below threshold.
    """
    s = m ** 2
    m12s = (m1 + m2) ** 2
    m12d = (m1 - m2) ** 2
    p2 = (s - m12s) * (s - m12d)
    # Clamp to 0 below threshold, divide by (2m)² then sqrt
    p = np.sqrt(np.where(p2 > 0, p2, 0)) / (2 * m)
    return p


def _h_fun(s, m1, m2):
    """GS h(s) function.

    TFPWA reference::

        k = cal_monentum(m, m1, m2)   # standard q_cm (NO *m factor)
        return 2/pi * k/sqrt(s) * log((sqrt(s) + 2*k) / sm)
    """
    pi = np.pi
    sm = m1 + m2
    sqrt_s = np.sqrt(s)
    k = _two_body_cm_mom(sqrt_s, m1, m2)  # standard q_cm, NO *m
    return (2.0 / pi) * (k / sqrt_s) * np.log((sqrt_s + 2.0 * k) / sm)


def _dh_ds_fun(s, m1, m2):
    """Derivative dh/ds at *s*."""
    pi = np.pi
    sqrt_s = np.sqrt(s)
    k = _two_body_cm_mom(sqrt_s, m1, m2)
    ret = _h_fun(s, m1, m2) * (1.0 / (8.0 * k ** 2) - 1.0 / (2.0 * s))
    ret = ret + 1.0 / (2.0 * pi * s)
    return ret


def _d_fun(s, m1, m2):
    """GS D(s) function (used for normalisation).

    TFPWA reference::

        k = cal_monentum(m, m1, m2)   # standard q_cm (NO *m factor)
        sm24 = (sm*sm)/4
        ret = 3/pi * sm24/k² * log((m+2k)/sm) + m/(2*pi*k) - sm24*m/(pi*k³)
    """
    pi = np.pi
    sm = m1 + m2
    m = np.sqrt(s)
    k = _two_body_cm_mom(m, m1, m2)  # standard q_cm, NO *m
    sm24 = (sm * sm) / 4.0
    ret = (3.0 / pi) * sm24 / k ** 2 * np.log((m + 2.0 * k) / sm)
    ret = ret + m / (2.0 * pi * k)
    ret = ret - sm24 * m / (pi * k ** 3)
    return ret


def _fs_fun(s, m2, gam, m1, m2m):
    """GS mass-shift function fs(s).

    TFPWA reference::

        k_s  = cal_monentum(sqrt(s), m1, m2m)    # standard q_cm, NO *m
        k_m2 = cal_monentum(sqrt(m2), m1, m2m)   # standard q_cm, NO *m
        f = gam * m2 / k_m2**3 * [k_s**2*(h(s)-h(m2)) + (m2-s)*k_m2**2*dh_ds(m2)]
    """
    k_s = _two_body_cm_mom(np.sqrt(s), m1, m2m)   # standard q_cm
    k_m2 = _two_body_cm_mom(np.sqrt(m2), m1, m2m)  # standard q_cm
    f = gam * m2 / k_m2 ** 3
    f *= k_s ** 2 * (_h_fun(s, m1, m2m) - _h_fun(m2, m1, m2m)) \
         + (m2 - s) * k_m2 ** 2 * _dh_ds_fun(m2, m1, m2m)
    return f


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
    # Horner's method for polyval
    val = np.zeros_like(z)
    for ci in c:
        val = val * z + ci
    return val


def _bprime(L, q, q0, d):
    """Blatt-Weisskopf barrier factor B'_L(q, q0, d)."""
    z = (q * d) ** 2
    z0 = (q0 * d) ** 2
    num = np.sqrt(_bprime_poly(L, z0))
    den = np.sqrt(_bprime_poly(L, z))
    return num / den


def _gamma_run(m, g0, q, q0, L, m0, d):
    """Running width Γ(m) for the relativistic BW."""
    qq0 = (q / q0) ** (2 * L + 1)
    mm0 = m0 / m
    bp = _bprime(L, q, q0, d) ** 2
    return g0 * qq0 * mm0 * bp


# ── Model class ─────────────────────────────────────────────────


@register_model("GS_rho")
class GSRhoModel(BaseModel):
    """Gounaris-Sakurai lineshape for the rho(770) resonance.

    The ``gamma(m)`` return value encodes both the running width
    and the GS mass-shift term such that::

        bw_dom = m0² - m² - i·m0·g0·gamma(m)
               = m0² - m² + fs(m²) - i·m0·Γ(m)

    which matches the GS denominator.

    Parameters from YAML config:
        mass          — pole mass m0
        width         — nominal width g0
        L             — orbital angular momentum (default 1)
        d             — impact parameter in GeV⁻¹ (default 3.0)
        daug2Mass     — first daughter mass (default π⁺, 0.13957039)
        daug3Mass     — second daughter mass (default π⁰, 0.1349768)
    """

    # Default daughter masses (π⁺, π⁰)
    _def_m2 = 0.13957039
    _def_m3 = 0.1349768

    def get_defaults(self):
        return {f"{self.name}_mass": float(self.kwargs.get("mass", 0.775)),
                f"{self.name}_width": float(self.kwargs.get("width", 0.149))}

    def gamma(self, m):
        M  = float(self.kwargs.get("M",  self.kwargs.get("mass",  0.775)))
        g0 = float(self.kwargs.get("width", 0.149))
        L  = int(self.kwargs.get("L", 1))
        d  = float(self.kwargs.get("d", 3.0))
        m2 = float(self.kwargs.get("daug2Mass", self._def_m2))
        m3 = float(self.kwargs.get("daug3Mass", self._def_m3))

        # Breakup momenta — both q and q0 use decay tree masses (consistent with TFPWA)
        if self._parent and self._parent._decays:
            d1 = float(self._parent._decays[0].outs[0].mass)
            d2 = float(self._parent._decays[0].outs[1].mass)
            q  = _two_body_cm_mom(m, d1, d2)
            q0 = _two_body_cm_mom(M, d1, d2)
        else:
            q  = _two_body_cm_mom(m, m2, m3)
            q0 = _two_body_cm_mom(M, m2, m3)

        # Running width (without the g0 factor)
        gamma_run = _gamma_run(m, 1.0, q, q0, L, M, d)

        # Mass-shift term — uses PDG masses (c_daug2Mass/c_daug3Mass in TFPWA)
        fs_val = _fs_fun(m ** 2, M ** 2, 1.0, self._def_m2, self._def_m3)

        # gamma(m) = Gamma(m)/g0 + i * fs(m²) / m0
        # _fs_fun(gam=1) returns fs_without_g0 coupling.
        # TFPWA's bw_dom = M² - m² + g0*fs - i*M*Γ.
        # The kernel computes: g = g0*gamma, bw_dom = M² - m² - i*M*g.
        # We need: -i*M*g = g0*fs - i*M*Γ → gamma = Γ/g0 + i*fs/M.
        gamma_m = gamma_run + 1j * fs_val / M
        return [gamma_m]
