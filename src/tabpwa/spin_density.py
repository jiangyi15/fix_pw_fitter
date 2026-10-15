"""Spin-density-matrix basis for the polarised-PWA model.

Default basis (no configuration needed): the trace-P Hermitian basis over
the top's helicity states — ``rho = I + sum_l p_l * G_l`` with ``I`` the
P-dimensional identity (trace = P) and ``G_l`` the P^2-1 traceless
Hermitian generators normalised to ``Tr(G_a G_b) = 2 delta_ab`` (the
generalised Pauli basis).

* P = 2: 3 generators = px, py, pz (the standard polarisation vector,
  ``rho = (I + px sx + py sy + pz sz)`` in Pauli normalisation).
* P = 3: 8 generators (px, py, pz, p4, ..., p8).

Zero parameters => ``rho = I`` => the intensity is exactly the plain PWA
projection sum ``sum_j |A_j|^2``.

Positivity (``rho`` positive semidefinite) is NOT enforced: the maximum
likelihood normally stays in the physical region and the fitted values are
reported as measured (bounds can be added via ``var_range``).
"""
import numpy as np


def polarisation_names(P, prefix=""):
    """Parameter names for *P* helicity states.

    P = 2: the standard polarisation vector ``px, py, pz``; larger P:
    the generic ``p1, p2, ..., p{P*P-1}``.  *prefix* is prepended to each
    name (e.g. the top particle name: ``jpsi_px``, ``jpsi_py``, ...).
    """
    if P < 2:
        raise ValueError(
            f"polarised_pwa needs >= 2 helicity states on the top (got "
            f"P={P}) — use amp_model: pwa for a scalar top")
    stem = f"{prefix}_" if prefix else ""
    if P == 2:
        return [f"{stem}px", f"{stem}py", f"{stem}pz"]
    return [f"{stem}p{i}" for i in range(1, P * P)]


def generators(P):
    """The P^2-1 traceless Hermitian generators (Tr(G_a G_b) = 2 delta_ab).

    Order: sym(0,1), anti(0,1), diag(0), then sym/anti for the remaining
    pairs and the remaining diagonals — so for P = 2 the first three are
    exactly the px, py, pz (Pauli) directions.
    """
    gen = []

    def _sym(a, b):
        m = np.zeros((P, P), complex)
        m[a, b] = m[b, a] = 1.0
        gen.append(m)

    def _anti(a, b):
        m = np.zeros((P, P), complex)
        m[a, b] = 1.0j
        m[b, a] = -1.0j
        gen.append(m)

    def _diag(a):
        m = np.zeros((P, P), complex)
        m[a, a] = 1.0
        m[P - 1, P - 1] = -1.0
        gen.append(m)

    if P >= 2:
        _sym(0, 1)
        _anti(0, 1)
        _diag(0)
    for a, b in ((a, b) for a in range(P) for b in range(a + 1, P)
                 if (a, b) != (0, 1)):
        _sym(a, b)
        _anti(a, b)
    for a in range(1, P - 1):
        _diag(a)
    return gen


def rho_matrix(params, P):
    """``rho = I + sum_l p_l * G_l`` from the real parameter vector."""
    params = np.asarray(params, dtype=float)
    gen = generators(P)
    if len(params) != len(gen):
        raise ValueError(
            f"spin-density parameter count {len(params)} != {len(gen)} "
            f"for P={P} ({len(gen)} traceless Hermitian generators)")
    rho = np.eye(P, dtype=complex)
    for val, g in zip(params, gen):
        rho = rho + val * g
    return rho
