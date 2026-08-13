"""Convert B → 4π 4-momenta to the fitter's data npz arrays.

Input: per-event 4-momenta of the four pions, ordered
``[π⁺₁, π⁻₁, π⁺₂, π⁻₂]`` (shape ``(n, 4, 4)``, each ``(E, px, py, pz)``).
Output: the ``mass`` / ``q`` / ``angles`` arrays consumed by the kernel.

Rows: ``24 = 8 blocks × 3 topologies``, row index ``= block*3 + topo``.
Blocks 0–3 are the four identical-particle permutations of the input
pion order; blocks 4–7 are the same permutations with the CP-conjugate
charge assignment (π⁺ ↔ π⁻; the overall CP sign lives in the amplitude
via the ``g_lsbar`` waves, not in the kinematic arrays).

Per row the kinematics are:

    mass[row, 0], mass[row, 1]  — invariant masses of the two
                                  resonances in the topology
    q[row, 0..2]                — breakup momenta q(B), q(decay₁),
                                  q(decay₂) in the respective rest frames
    angles[row, 0..2]           — [φ, θ₁, θ₂] (azimuth in [-π, π],
                                  polar angles in [0, π])

Topologies
----------
0.  B → ρ₁(π⁺₁π⁻₁) ρ₂(π⁺₂π⁻₂):
      θ₁ : in the ρ₁ rest frame, angle of π⁺₁ w.r.t. the ρ₁ flight
           direction (helicity frame)
      θ₂ : in the ρ₂ rest frame, angle of π⁺₂ w.r.t. the ρ₂ flight
           direction
      φ  : azimuth of π⁺₁ around the ρ₁ axis, referenced from π⁺₂
           (the angle between the two di-pion planes).
1.  B → R₁(R₂(π⁺₁π⁻₁)π⁺₂) π⁻₂  (sequential chain, bachelor π⁻₂):
      in the R₁ rest frame: θ₁ (supplement of the angle between R₂ and
      the bachelor), φ = azimuth of the R₂-decay plane (π⁺₁π⁻₁) around
      the R₂ axis referenced from the bachelor; in the R₂ rest frame
      (sequential R₁→R₂ boost): θ₂ = angle of π⁺₁ w.r.t. the R₂ flight
      direction (helicity axis).
2.  mirror of 1: B → R₁(R₂(π⁺₁π⁻₁)π⁻₂) π⁺₂ — same construction with
    the middle pion and bachelor exchanged.

The CP blocks (4-7) flip the sign of the azimuth φ.

Usage::

    from ampfit.phasespace_b4pi import generate_b4pi
    from ampfit.momenta_to_data import momenta_to_data
    ev = generate_b4pi(10000)
    data = momenta_to_data(ev["momenta"])
"""

import numpy as np

from ampfit.phasespace_b4pi import two_body_momentum


# ═══════════════════════════════════════════════════════════════════
# Lorentz helpers (operate on (..., 4) 4-momenta, velocity (..., 3))
# ═══════════════════════════════════════════════════════════════════

def _boost(p4, v):
    """Boost 4-momenta by velocity *v* (to the rest frame of *v*)."""
    v = np.asarray(v, dtype=float)
    p4 = np.asarray(p4, dtype=float)
    E, p = p4[..., 0], p4[..., 1:]
    v2 = np.sum(v * v, axis=-1)
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - v2, 1e-300))
    vdotp = np.sum(v * p, axis=-1)
    p_par = (gamma - 1.0) * vdotp / np.maximum(v2, 1e-300)
    Eo = gamma * (E - vdotp)
    po = p + p_par[..., None] * v - gamma[..., None] * E[..., None] * v
    return np.concatenate([Eo[..., None], po], axis=-1)


def _inv_mass(p4):
    """Invariant mass of a 4-momentum (..., 4) -> (...,)."""
    return np.sqrt(np.maximum(p4[..., 0] ** 2 - np.sum(p4[..., 1:] ** 2,
                                                       axis=-1), 0.0))


def _unit(v):
    """Unit 3-vectors (..., 3); zero vectors stay zero."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


def _polar_cos(a, b):
    """cos of the angle between 3-vectors a and b (clipped)."""
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    c = np.sum(a * b, axis=-1) / np.maximum(na * nb, 1e-300)
    return np.clip(c, -1.0, 1.0)


def _azimuth(vec, axis, ref):
    """Azimuth of *vec* around *axis*, measured from *ref*, in [-π, π].

    Project vec and ref onto the plane ⊥ axis; take ref's direction as
    azimuth 0 and use arctan2 for the signed angle.
    """
    a = _unit(axis)
    vp = vec - np.sum(vec * a, axis=-1, keepdims=True) * a
    rp = ref - np.sum(ref * a, axis=-1, keepdims=True) * a
    e1 = _unit(rp)
    e2 = np.cross(a, e1)
    return np.arctan2(np.sum(vp * e2, axis=-1), np.sum(vp * e1, axis=-1))


# ═══════════════════════════════════════════════════════════════════
# Topology 0: B → ρ₁(π⁺₁π⁻₁) ρ₂(π⁺₂π⁻₂)  (vectorised over events)
# ═══════════════════════════════════════════════════════════════════

def _topo_rhorho(p, m_B):
    """Vectorised ρρ topology; p (n, 4, 4), order [π⁺₁, π⁻₁, π⁺₂, π⁻₂].

    *p* must already be in the B rest frame; *m_B* is the per-event B
    mass (n,).  Returns ``(m1, m2, q0, q1, q2, phi, theta1, theta2)``.
    """
    P1, P2 = p[:, 0] + p[:, 1], p[:, 2] + p[:, 3]   # ρ₁, ρ₂ in the B frame
    m1, m2 = _inv_mass(P1), _inv_mass(P2)
    n1 = _unit(P1[:, 1:])                           # ρ₁ flight direction

    q0 = two_body_momentum(m_B, m1, m2)             # B breakup momentum
    # pion masses from the data (per event, per pion)
    mp = _inv_mass(p)                               # (n, 4)
    q1 = two_body_momentum(m1, mp[:, 0], mp[:, 1])  # ρ₁ → π⁺₁π⁻₁
    q2 = two_body_momentum(m2, mp[:, 2], mp[:, 3])  # ρ₂ → π⁺₂π⁻₂

    # π⁺₁ in the ρ₁ rest frame: θ₁ vs the ρ₁ flight direction (+n1)
    a1 = _boost(p[:, 0], P1[:, 1:] / P1[:, 0:1])
    cos_t1 = _polar_cos(a1[:, 1:], n1)
    # π⁺₂ in the ρ₂ rest frame: θ₂ vs the ρ₂ flight direction (−n1)
    c1 = _boost(p[:, 2], P2[:, 1:] / P2[:, 0:1])
    cos_t2 = _polar_cos(c1[:, 1:], -n1)

    # azimuth of π⁺₁ around the ρ₁ axis, referenced from π⁺₂ (plane angle)
    phi = _azimuth(p[:, 0][:, 1:], n1, p[:, 2][:, 1:])

    return m1, m2, q0, q1, q2, phi, np.arccos(cos_t1), np.arccos(cos_t2)


# ═══════════════════════════════════════════════════════════════════
# Topology 1: B → R₁(R₂(π⁺₁π⁻₁)π⁺₂) π⁻₂  (sequential chain)
# ═══════════════════════════════════════════════════════════════════

def _topo_chain(p, m_B):
    """Vectorised chain topology; p (n, 4, 4), order [π⁺₁, π⁻₁, π⁺₂, π⁻₂].

    R₂ = (π⁺₁, π⁻₁), R₁ = (π⁺₁, π⁻₁, π⁺₂), bachelor π⁻₂.  *p* must be
    in the B rest frame; *m_B* is the per-event B mass (n,).

    Returns ``(m_R1, m_R2, q0, q1, q2, phi, theta1, theta2)``:
      θ₁, φ in the R₁ rest frame: cos θ₁ = p̂(R₂)·p̂(π⁻₂), φ = azimuth
      of the R₂-decay plane (π⁺₁π⁻₁) around the R₂ axis referenced
      from π⁻₂.  θ₂ in the R₂ rest frame (sequential R₁→R₂ boost):
      cos θ₂ = p̂(π⁺₁)·p̂(π⁺₂ + π⁻₂)  (the direction opposite R₂).
    """
    R2 = p[:, 0] + p[:, 1]
    R1 = R2 + p[:, 2]
    d = p[:, 3]                                     # bachelor π⁻₂
    m_R2, m_R1 = _inv_mass(R2), _inv_mass(R1)
    mp = _inv_mass(p)                               # (n, 4) pion masses

    q0 = two_body_momentum(m_B, m_R1, mp[:, 3])     # B → R₁ π⁻₂
    q1 = two_body_momentum(m_R1, m_R2, mp[:, 2])    # R₁ → R₂ π⁺₂
    q2 = two_body_momentum(m_R2, mp[:, 0], mp[:, 1])  # R₂ → π⁺₁π⁻₁

    # ── R₁ rest frame ─────────────────────────────────────────────
    beta1 = R1[:, 1:] / R1[:, 0:1]
    R2r = _boost(R2, beta1)
    dr = _boost(d, beta1)
    ar = _boost(p[:, 0], beta1)
    br = _boost(p[:, 1], beta1)

    # θ₁: supplement of the angle between R₂ and the bachelor π⁻₂
    cos_t1 = -_polar_cos(R2r[:, 1:], dr[:, 1:])
    # azimuth of the R₂ decay plane (π⁺₁π⁻₁) around the R₂ axis,
    # referenced from the {R₂, π⁻₂} plane
    phi = _azimuth(ar[:, 1:], R2r[:, 1:], dr[:, 1:])

    # ── R₂ rest frame (sequential boost R₁ then R₂) ───────────────
    # θ₂: angle of π⁺₁ w.r.t. the R₂ flight direction (helicity axis,
    # i.e. the R₂ momentum as seen in the R₁ frame)
    beta2 = R2r[:, 1:] / R2r[:, 0:1]
    arr = _boost(ar, beta2)
    cos_t2 = _polar_cos(arr[:, 1:], R2r[:, 1:])

    return m_R1, m_R2, q0, q1, q2, phi, \
        np.arccos(cos_t1), np.arccos(cos_t2)


# Topology 2: mirror of the chain — same R₂ = (π⁺₁, π⁻₁) but the
# middle pion is π⁻₂ and the bachelor π⁺₂:
# B → R₁(R₂(π⁺₁π⁻₁)π⁻₂) π⁺₂.
def _topo_chain_mirror(p, m_B):
    return _topo_chain(p[:, [0, 1, 3, 2]], m_B)


# ═══════════════════════════════════════════════════════════════════
# Pion permutations (identical particles) and CP exchange
# ═══════════════════════════════════════════════════════════════════

# 4 identical-particle permutations of [π⁺₁, π⁻₁, π⁺₂, π⁻₂]
# (order matches the reference data: B blocks swap π⁻ before π⁺)
_IDENTICAL_PERMS = [
    (0, 1, 2, 3),   # identity
    (0, 3, 2, 1),   # swap π⁻₁ ↔ π⁻₂
    (2, 1, 0, 3),   # swap π⁺₁ ↔ π⁺₂
    (2, 3, 0, 1),   # swap both
]

# CP blocks (4-7) use a reversed permutation order: [id, π⁺, π⁻, both]
_CP_PERM_ORDER = [0, 2, 1, 3]

# CP: exchange π⁺ ↔ π⁻  ([π⁺₁,π⁻₁,π⁺₂,π⁻₂] → [π⁻₁,π⁺₁,π⁻₂,π⁺₂])
_CP_INDEX = (1, 0, 3, 2)

# Topology builders (vectorised), indexed by topology number
_TOPOLOGIES = [_topo_rhorho, _topo_chain, _topo_chain_mirror]


def momenta_to_data(momenta, weight=None, frac=None, time=None):
    """Convert B → 4π momenta to the fitter data npz arrays.

    Parameters
    ----------
    momenta : ndarray (n, 4, 4)
        Pion 4-momenta (E, px, py, pz), ordered [π⁺₁, π⁻₁, π⁺₂, π⁻₂]
        (any frame; boosted to the B rest frame internally).  The
        per-event B mass and pion masses are taken from the data.
    weight, frac, time : ndarray (n,), optional
        Per-event weight / tagging fraction / decay time.  Defaults:
        weight = 1, frac = 0.5, time = 0.

    Returns
    -------
    dict with ``mass`` (n, 24, 2), ``q`` (n, 24, 3), ``angles``
    (n, 24, 3), ``frac``, ``time``, ``bkg_raw``, ``weight``.

    NOTE: only topology 0 (ρρ) is implemented so far; the rows of
    topologies 1 and 2 are left at zero.
    """
    momenta = np.asarray(momenta, dtype=float)
    n = len(momenta)
    mass = np.zeros((n, 24, 2))
    q = np.zeros((n, 24, 3))
    angles = np.zeros((n, 24, 3))

    # Boost every event to the B rest frame (the input may be in the
    # lab frame) and use the actual per-event B mass for q(B).
    tot = momenta.sum(1)
    mB_ev = _inv_mass(tot)
    betaB = tot[:, 1:] / tot[:, 0:1]
    pb = _boost(momenta, betaB[:, None, :])         # (n, 4, 4) B rest frame

    for b in range(8):
        perm = _IDENTICAL_PERMS[_CP_PERM_ORDER[b % 4] if b >= 4 else b % 4]
        order = np.array(perm)
        if b >= 4:                              # CP block
            order = order[list(_CP_INDEX)]
        pe = pb[:, order]                       # (n, 4, 4) permuted
        for t, topo in enumerate(_TOPOLOGIES):
            row = b * 3 + t
            m1, m2, q0, q1, q2, phi, th1, th2 = topo(pe, mB_ev)
            if b >= 4:                          # CP: azimuth flips sign
                phi = -phi
            mass[:, row, 0], mass[:, row, 1] = m1, m2
            q[:, row] = np.stack([q0, q1, q2], axis=-1)
            angles[:, row] = np.stack([phi, th1, th2], axis=-1)

    return {
        "mass": mass,
        "q": q,
        "angles": angles,
        "frac": (np.ones(n) * 0.5 if frac is None
                 else np.asarray(frac, dtype=float)),
        "time": (np.zeros(n) if time is None
                 else np.asarray(time, dtype=float)),
        "bkg_raw": np.zeros(n),
        "weight": (np.ones(n) if weight is None
                   else np.asarray(weight, dtype=float)),
    }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-events", type=int, default=100000)
    ap.add_argument("-o", "--output", default="data_arrays.npz")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    from ampfit.phasespace_b4pi import generate_b4pi
    ev = generate_b4pi(args.n_events, seed=args.seed)
    data = momenta_to_data(ev["momenta"])
    np.savez(args.output, mass=data["mass"], q=data["q"],
             angles=data["angles"], frac=data["frac"], time=data["time"],
             bkg_raw=data["bkg_raw"], weight=data["weight"])
    print(f"wrote {args.n_events} events -> {args.output}")
