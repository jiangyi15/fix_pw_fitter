"""ππ S-wave K-matrix fixed-shape model (AmpGen kMatrix convention).

Implements the 5-channel Anisovich–Sarantsev K-matrix for the ππ S-wave
(channels: pipi, KK, 4pi, etaeta, etaetaprime) as a :class:`FixedShapeModel`.  K-matrix constants follow the Anisovich–Sarantsev reference.

Amplitude (matching ``AmpGen/src/Lineshapes/kMatrix.cpp``)::

    K(s)     = Σ_poles g_p·g_pᵀ / (s_p − s) + scatter + Adler zero
    F(s)     = (I − i·ρ(s)·K(s))⁻¹
    pole_{a}_{N} = Σ_i F[a,i]·g_N[i] / (m_N² − s)   (pole-N production)
    prod_{a}_{N} = F[a,N]·(1 − s0_prod) / (s − s0_prod)  (production bg)

``term`` selects ONE contribution with an F-row (number 0-4 or channel
name) and pole/channel index *N* (0-based)::

    pole_0 / pole_pipi_0   -> F[pipi,:]·g_pole0  (AmpGen kMatrix:pole.0)
    pole_1_2 / pole_KK_2   -> F[KK,:]·g_pole2    (AmpGen kMatrix:poleKK.2)
    prod_0_3 / prod_pipi_3 -> F[pipi,3] production bg

YAML::

    particle:
      k_res:
        mass: 1.23
        width: 0.4
        model: pipi_swave
        term: pole_0         # F[pipi,:]·g_pole0  (default F-row 0)
        term: pole_KK_2      # F[KK,:]·g_pole2
        term: prod_pipi_3    # F[pipi,3] production bg

    The coupling of each term is fitted through the CK product (g_ls);
    the shape carries no multiplicative constant.
"""

import numpy as np
from .base import register_model
from .models_builtin import FixedShapeModel

# ═══════════════════════════════════════════════════════════════
# Anisovich–Sarantsev K-matrix constants (5 poles × 5 channels)
# Channel order: pipi, KK, 4pi, etaeta, etaetaprime
# ═══════════════════════════════════════════════════════════════

CHANNELS = ["pipi", "KK", "4pi", "etaeta", "etaetaprime"]
CHANNEL_INDEX = {c: i for i, c in enumerate(CHANNELS)}

POLE_MASS = np.array([0.651, 1.2036, 1.55817, 1.21, 1.82206])

POLE_G = np.array([
    [0.22889, -0.55377, 0.0, -0.39899, -0.34639],
    [0.94128, 0.55095, 0.0, 0.39065, 0.31503],
    [0.36856, 0.23888, 0.55639, 0.1834, 0.18681],
    [0.3365, 0.40907, 0.85679, 0.19906, -0.00984],
    [0.18171, -0.17558, -0.79658, -0.00355, 0.22358],
])

F_SCATT = np.array([0.23399, 0.15044, -0.20545, 0.32825, 0.35412])

SA, SA0 = 1.0, -0.15
S0_SCATT, S0_PROD = -3.92637, -0.165753

# Phase-space thresholds 4m² (or mass sums) for each channel
M_PI, M_K, M_ETA, M_ETAP = 0.139570, 0.493677, 0.547862, 0.967780

N_CH = 5


@register_model("pipi_swave")
class PipiSModel(FixedShapeModel):
    """K-matrix ππ S-wave production amplitude as a fixed shape.

    Constants are the standard Anisovich–Sarantsev values (fixed in
    this file).  ``term`` selects a single production contribution
    with explicit F-row *a* and index *N* (0-based)::

        pole_{N} / pole_{a}_{N} / pole_{chan}_{N}
          -> sum_i F[a,i]·g_N[i] / (m_N^2 - s)
        prod_{N} / prod_{a}_{N} / prod_{chan}_{N}
          -> F[a,N]·(1 - s0_prod)/(s - s0_prod)

    *a* is an F-row number or channel name (pipi/KK/4pi/etaeta/
    etaetaprime); default F-row is 0 (pipi).  The overall coupling is
    fitted via the CK product (g_ls); the shape itself carries no
    multiplicative constant.
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.term = str(kwargs.get("term", "pole_0_0"))
        # Validate the term selector
        kind, a, N = self._parse_term(self.term)
        self._term = (kind, a, N)


    @staticmethod
    def _parse_term(term):
        """Parse term -> (kind, row, idx).

        Accepted forms::

            pole_{N}            -> pole production, F-row 0 (pipi), pole N
            pole_{a}_{N}        -> pole production, F-row a, pole N
            pole_{chan}_{N}     -> pole production, F-row chan (name), pole N
            prod_{N} / prod_{a}_{N} / prod_{chan}_{N}  (same for bg)

        *a* is a 0-based F-row number or a channel name
        (pipi/KK/4pi/etaeta/etaetaprime); *N* is the pole/channel index.
        """
        parts = term.split("_")
        kind = parts[0]
        if kind not in ("pole", "prod"):
            raise ValueError(f"pipi_swave: unknown kind in '{term}' "
                             "(use pole_... or prod_...)")

        def _channel_or_int(s):
            if s.isdigit():
                v = int(s)
                if not (0 <= v < N_CH):
                    raise ValueError(f"pipi_swave: F-row {v} out of range "
                                     f"(0..{N_CH - 1})")
                return v
            if s not in CHANNEL_INDEX:
                raise ValueError(f"pipi_swave: unknown channel '{s}' "
                                 f"(use {CHANNELS} or 0..{N_CH - 1})")
            return CHANNEL_INDEX[s]

        if len(parts) == 2:                  # pole_{N}  (default row 0)
            if not parts[1].isdigit():
                raise ValueError(f"pipi_swave: bad term='{term}'")
            return (kind, 0, int(parts[1]))
        if len(parts) == 3:                  # pole_{a}_{N} / pole_{chan}_{N}
            if not parts[2].isdigit():
                raise ValueError(f"pipi_swave: bad term='{term}'")
            a = _channel_or_int(parts[1])
            N = int(parts[2])
            if not (0 <= N < N_CH):
                raise ValueError(f"pipi_swave: index N={N} out of range "
                                 f"(0..{N_CH - 1})")
            return (kind, a, N)
        raise ValueError(f"pipi_swave: bad term='{term}' "
                         "(use pole_{N}, pole_{a}_{N}, or pole_{chan}_{N})")

    # ── K-matrix core ──────────────────────────────────────────

    def _k_matrix(self, s):
        """K(s) as (..., n_ch, n_ch).  s: (...,) array of mass²."""
        s = np.asarray(s, dtype=float)
        # K[q,c] = Σ_p g[p,q]·g[p,c] / (s_p − s),  s_p = m_p² + i·1e-6
        sp = POLE_MASS ** 2 + 1e-6 * 1j
        K = np.einsum(
            'pq,pc,ps->sqc',
            POLE_G, POLE_G,
            1.0 / (sp[:, None] - s[None, :]))
        # Scatter terms: scattPart(i,0) + scattPart(0,i) (AmpGen)
        for i in range(N_CH):
            sc = F_SCATT[i] * (1 - S0_SCATT) / (s - S0_SCATT)
            sc = np.where(np.abs(s - S0_SCATT) > 1e-12, sc, 0)
            K[..., 0, i] += sc
            if i != 0:
                K[..., i, 0] += sc
        # Adler zero
        adler = ((1 - SA0) * (s - SA * M_PI * M_PI / 2) / (s - SA0))
        K = K * adler[..., None, None]
        return K

    def _rho(self, s):
        """Phase space ρ(s): (..., n_ch) — matches AmpGen phsp_*."""
        s = np.asarray(s, dtype=float)
        rho = np.zeros(s.shape + (N_CH,), dtype=complex)
        # pipi(0), KK(1), etaeta(3), etaetaprime(4): phsp_twoBody
        for c, (m0, m1) in [(0, (M_PI, M_PI)), (1, (M_K, M_K)),
                            (3, (M_ETA, M_ETA)), (4, (M_ETA, M_ETAP))]:
            rho[..., c] = np.emath.sqrt(1 - (m0 + m1) ** 2 / s)
        # 4pi channel (index 2): polynomial below s=1, twoBody(2m_pi, 2m_pi) above
        rho[..., 2] = np.emath.sqrt(1 - (2 * M_PI + 2 * M_PI) ** 2 / s)
        mask = s <= 1.0
        if np.any(mask):
            sv = s[mask]
            rho[mask, 2] = (0.00051 - .01933 * sv + .13851 * sv ** 2
                            - .2084 * sv ** 3 - .29744 * sv ** 4
                            + .13655 * sv ** 5 + 1.07885 * sv ** 6)
        return rho

    def _propagator(self, s):
        """F(s) = (I − i·ρ·K)⁻¹: (..., n_ch, n_ch)."""
        K = self._k_matrix(s)
        rho = self._rho(s)
        mat = np.eye(N_CH)[None, :, :] - 1j * K * rho[..., None, :]
        return np.linalg.inv(mat)

    # ── FixedShapeModel interface ───────────────────────────────

    def fixed_shape(self, m):
        """AmpGen-style production amplitude for the selected term."""
        s = np.asarray(m, dtype=float) ** 2
        F = self._propagator(s)                       # (..., n_ch, n_ch)
        kind, a, N = self._term
        eps = 1e-6
        if kind == "pole":       # Σ_i F[a,i]·g_N[i] / (m_N² − s)
            P = np.einsum('...c,c->...', F[..., a, :], POLE_G[N])
            return P / (eps + POLE_MASS[N] ** 2 - s)
        if kind == "prod":       # F[a,N]·(1−s0p)/(s−s0p)
            return F[..., a, N] * (1 - S0_PROD) / (s - S0_PROD)
        raise ValueError(f"pipi_swave: bad term kind '{kind}'")
