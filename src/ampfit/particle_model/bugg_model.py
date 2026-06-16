"""Bugg parametrisation for the sigma/f0(500) resonance.

Implements the full energy-dependent width with Adler zero and 2π, 2K,
2η, 4π coupled channels, based on the TFPWA reference implementation.

All Bugg-specific parameters are hardcoded.  The YAML only needs::

    particle:
      f0(500):
        mass: 0.953
        model: Bugg
"""

import numpy as np
from .base import BaseModel, register_model

# ── Physical constants ─────────────────────────────────────────
_mPi   = 0.139570
_mK    = 0.493677
_mEta  = 0.547863

# ── Bugg parametrisation constants (fixed) ─────────────────────
_b1      = 1.302
_b2      = 0.340
_A       = 2.426
_g4pi    = 0.011
_g2K     = 0.6
_g2eta   = 0.2
_alpha   = 1.3
_sA_factor = 0.41          # sA = factor * mPi²
_s0_4pi    = 7.082 / 2.845
_lambda_4pi = 2.845


# ── Helper functions ────────────────────────────────────────────

def _rho(s, s0):
    """Phase-space factor ``max(0, 1 - 4*s0/s)``."""
    return np.maximum(0.0, 1.0 - 4.0 * s0 / s)


def _rho2(s, s0):
    """Complex phase space ``√(1 - 4*s0/s)``."""
    return np.sqrt((1.0 - 4.0 * s0 / s) + 0j)


def _q(s, s0):
    """``|s - 4*s0|``."""
    return np.abs(s - 4.0 * s0)


def _rho_4pi(s, lam, s0):
    """Smooth step for the 4π threshold."""
    return 1.0 / (1.0 + np.exp(lam * (s0 - s)))


def _buggj1(s, m0):
    """Bugg J1 function (dispersive term)."""
    m0_2 = m0 * m0
    rp = np.sqrt(_rho(s, m0_2))
    add = np.where(
        rp > 0.0,
        rp * np.log((1.0 - rp) / (1.0 + rp)),
        np.ones_like(rp),
    )
    return (2.0 + add) / np.pi


def _gamma_4pi(s, M, g4pi, lam, s0):
    """4π width contribution."""
    out = np.where(
        s > 16.0 * _mPi * _mPi,
        g4pi * _rho_4pi(s, lam, s0) / _rho_4pi(M * M, lam, s0),
        0.0,
    )
    return out


# ── Model class ─────────────────────────────────────────────────

@register_model("Bugg")
class BuggModel(BaseModel):
    """Bugg parametrisation for the sigma/f0(500) resonance.

    Full energy-dependent width from the TFPWA reference implementation.
    All Bugg-specific parameters are hardcoded; only the pole mass
    (``mass``) needs to be set in YAML.
    """

    def get_gamma_defaults(self):
        return [float(self.kwargs.get("width", 1.0))]

    def gamma(self, m):
        s = m ** 2
        # BUGG model uses a fixed sigma mass (M=0.953) regardless of config
        self._bugg_M = 0.953
        M = self._bugg_M
        M2 = M * M
        mPi2 = _mPi ** 2
        mK2  = _mK ** 2
        mEta2 = _mEta ** 2
        sA = _sA_factor * mPi2

        # Dispersive term
        z = _buggj1(s, _mPi) - _buggj1(M2, _mPi)

        # g1sg · adlerZero
        g1sg      = M * (_b1 + _b2 * s) * np.exp(-(s - M2) / _A)
        adlerZero = (s - sA) / (M2 - sA)

        # Coupled-channel widths (complex)
        gamma_2pi  = (g1sg * adlerZero + 0j) * _rho2(s, mPi2)
        gamma_2K   = (_g2K * g1sg * s / M2
                      * np.exp(-_alpha * _q(s, mK2))
                      + 0j) * _rho2(s, mK2)
        gamma_2eta = (_g2eta * g1sg * s / M2
                      * np.exp(-_alpha * _q(s, mEta2))
                      + 0j) * _rho2(s, mEta2)
        gamma_4pi  = M * _gamma_4pi(s, M, _g4pi, _lambda_4pi, _s0_4pi)

        Gamma_tot = gamma_2pi + gamma_2K + gamma_2eta + gamma_4pi
        bw_term = g1sg * adlerZero * z + 1j * Gamma_tot

        # Framework: dom = m0² - m² - i·m0·g0·gamma(m)
        # TFPWA:     dom = M² - m² - g1sg·A·z - i·Γ_tot  (with M=0.953)
        # We need gamma such that kernel produces M² - m² - bw_term
        # ⇒  gamma = i·(M² - m0² - bw_term) / (m0·g0)
        g0 = float(self.kwargs.get("width", 1.0))
        m0 = float(self.kwargs.get("mass", M))
        gamma = 1j * (M * M - m0 * m0 - bw_term) / (m0 * g0)
        return [gamma]


# ── Quick self-test ─────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .base import build_particle
    b = build_particle("f0", model="Bugg", mass=0.953, width=1.0)
    m = np.linspace(0.3, 2.0, 1000)
    g = b.gamma(m)[0]
    # Reconstruct the full propagator
    # (same check as the original TF script)
    s = m ** 2
    M = 0.953
    g0 = 1.0
    bw = 1.0 / (M ** 2 - s - 1j * M * g0 * g)
    plt.plot(m, np.real(bw), label="Re")
    plt.plot(m, np.imag(bw), label="Im")
    plt.plot(m, np.abs(bw) ** 2, label="|BW|²")
    plt.xlabel("m (GeV)")
    plt.legend()
    plt.savefig("/tmp/bugg_test.png")
    print("Saved /tmp/bugg_test.png")
