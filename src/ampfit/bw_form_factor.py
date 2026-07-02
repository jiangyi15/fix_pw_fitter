"""
Blatt-Weisskopf form factor (barrier factor).

Standalone implementation matching ampfit's ``build_fl_table`` convention:
``F_L(q) = q^L · B'_L(q, q0_ref, d)`` with the TFPWA normalisation
``F_L(q0_ref) = q0_ref^L``.
"""
import numpy as np


# ── Barrier polynomials ───────────────────────────────────────────

def _bprime_poly(L, z):
    """Blatt-Weisskopf barrier polynomial at order *L*.

    ``z = (q·R)²`` where *q* is the breakup momentum and *R* the
    meson radius.
    """
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


def barrier_ratio(L, q, q0, d=3.0):
    r"""Blatt-Weisskopf barrier ratio :math:`B'_L(q, q_0, d)`.

    .. math::

        B'_L(q, q_0, d) = \sqrt{\frac{P_L(z_0)}{P_L(z)}}
        \qquad z = (q d)^2

    where :math:`P_L` is the barrier polynomial.

    Args:
        L: orbital angular momentum (0 … 5).
        q: breakup momentum (GeV), scalar or array.
        q0: reference breakup momentum (GeV).
        d: meson radius (GeV\ :sup:`-1`).  Default 3.0.

    Returns:
        Barrier ratio :math:`B'_L` (dimensionless, = 1 at q = q0).
    """
    z = (q * d) ** 2
    z0 = (q0 * d) ** 2
    return np.sqrt(_bprime_poly(L, z0)) / np.sqrt(_bprime_poly(L, z))


def form_factor(L, q, q0_ref=1.0, d=3.0):
    r"""Blatt-Weisskopf form factor :math:`F_L(q)`.

    .. math::

        F_0(q) &= 1 \\
        F_L(q) &= q^L \; B'_L(q, q_0^{\mathrm{ref}}, d) \quad (L > 0)

    Normalised so that :math:`F_L(q_0^{\mathrm{ref}}) = (q_0^{\mathrm{ref}})^L`,
    matching ampfit's ``build_fl_table`` convention.

    Args:
        L: orbital angular momentum (0 … 5).
        q: breakup momentum (GeV), scalar or ndarray.
        q0_ref: reference momentum (GeV).  Default 1.0.
        d: meson radius (GeV\ :sup:`-1`).  Default 3.0.

    Returns:
        ``F_L(q)`` — scalar if *q* is scalar, else ndarray.
    """
    scalar_in = np.ndim(q) == 0
    qa = np.asarray(q, dtype=np.float64)
    if L == 0:
        out = np.ones_like(qa)
    else:
        out = (qa ** L) * barrier_ratio(L, qa, q0_ref, d)
    return float(out) if scalar_in else out
