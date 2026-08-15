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

from ampfit.phasespace_b4pi import (two_body_momentum, build_momenta, M_PION,
                                    M_B_MESON)


# ═══════════════════════════════════════════════════════════════════
# Lorentz helpers (operate on (..., 4) 4-momenta, velocity (..., 3))
# ═══════════════════════════════════════════════════════════════════

def _boost3_op(v):
    """Precompute the velocity-dependent factors of a spatial boost.

    The three boosts inside a chain topology share the same velocity, so
    `v²`, `γ` and `(γ−1)/v²` are computed once here and reused.
    Returns ``(v, gamma, gm1_v2)`` for :func:`_boost` / :func:`_boost3`.
    """
    v = np.asarray(v, dtype=float)
    v2 = np.sum(v * v, axis=-1)
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - v2, 1e-300))
    gm1_v2 = (gamma - 1.0) / np.maximum(v2, 1e-300)
    return v, gamma, gm1_v2


def _boost(p4, vop):
    """Boost 4-momenta by a precomputed velocity operator from
    :func:`_boost3_op`.  Returns full boosted 4-vectors (E, p)."""
    p4 = np.asarray(p4, dtype=float)
    v, gamma, gm1_v2 = vop
    E, p = p4[..., 0], p4[..., 1:]
    vdotp = np.sum(v * p, axis=-1)
    Eo = gamma * (E - vdotp)
    po = p + (gm1_v2 * vdotp - gamma * E)[..., None] * v
    return np.concatenate([Eo[..., None], po], axis=-1)


def _boost3(p4, vop):
    """Boost only the spatial part of a 4-momentum by a precomputed
    velocity operator from :func:`_boost3_op`.

    The azimuth φ needs only the boosted 3-momenta, so the (unused) time
    component and the final concatenation are skipped here.
    """
    p4 = np.asarray(p4, dtype=float)
    v, gamma, gm1_v2 = vop
    E, p = p4[..., 0], p4[..., 1:]
    vdotp = np.sum(v * p, axis=-1)
    return p + (gm1_v2 * vdotp - gamma * E)[..., None] * v


def _inv_mass_sq(p4):
    """Invariant mass-squared of a 4-momentum (..., 4) -> (...,)."""
    return np.maximum(p4[..., 0] ** 2 - np.sum(p4[..., 1:] ** 2, axis=-1),
                      0.0)


def _unit(v):
    """Unit 3-vectors (..., 3); zero vectors stay zero."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


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
# Mass descriptors: every invariant mass is keyed by a plain sorted
# tuple of the pion indices involved — built with ``tuple(sorted(..))``.
# The cache maps these directly to mass values.
# ═══════════════════════════════════════════════════════════════════

_B = (0, 1, 2, 3)     # mass descriptor: the B meson = all four pions


class _LazyDict(dict):
    """Index→value dict that computes a missing entry lazily on first
    access (via :meth:`dict.__missing__`) and then stores it."""

    def __init__(self, compute):
        super().__init__()
        self._compute = compute

    def __missing__(self, d):
        v = self._compute(d)
        self[d] = v
        return v


class _PionCache:
    """Lazy cache of per-event kinematics, keyed by pion-index tuples.

    The 8 blocks × 3 topologies share this one object.  Every invariant
    mass lives in the flat :attr:`mass` dict — one entry per descriptor,
    a sorted tuple of the pion indices involved (built with
    ``tuple(sorted(..))``): ``_B`` (0,1,2,3), a pair ``(i,j)``, a triple
    ``(i,j,k)``, or a single pion ``(i,)`` (and the mass-squares in
    :attr:`sq`) — all computed lazily on first access as the invariant
    mass of the summed pion 4-momenta.  The two generic physics
    functions, the breakup momentum :meth:`q` and the helicity angle
    :meth:`theta`, are lazy index→value dicts memoized by their
    descriptors, so identical physical angles (e.g. block 0 θ₁ ==
    block 3 θ₂) are computed once.
    """

    def __init__(self, pb):
        self._pb = pb
        self.mass = _LazyDict(self._mass_of)
        self.sq = _LazyDict(self._sq_of)
        self._q = _LazyDict(self._q_of)
        self._theta = _LazyDict(self._theta_of)

    def p(self, i):
        """The (n, 4) 4-momentum of pion *i* (view)."""
        return self._pb[:, i]

    # ── invariants from the raw pion momenta ───────────────────────
    def _sq_of(self, d):
        """(n,) invariant mass² of the pions in descriptor *d* — the one
        place the invariant is computed; computed once."""
        return _inv_mass_sq(self._pb[:, d].sum(1))

    def _mass_of(self, d):
        """(n,) invariant mass of the pions in descriptor *d* — the
        cached mass² (single source) sqrt'd."""
        return np.sqrt(self.sq[d])

    def _q_of(self, key):
        """Breakup momentum ``q(X; 1, 2)`` for the descriptor triple."""
        X, d1, d2 = key
        return two_body_momentum(self.mass[X], self.mass[d1], self.mass[d2])

    def _theta_of(self, key):
        """Helicity angle (radians) of ``A -> 1+2`` in ``X -> A+C``."""
        X, A, C, d1, d2 = key
        return np.arccos(_helicity_cos(
            self.q(X, A, C), self.q(A, d1, d2),
            self.mass[X], self.mass[A], self.mass[C],
            self.mass[d1], self.mass[d2],
            self._cross2(d1, C), self._cross2(d2, C)))

    def _cross2(self, d1, C):
        """(n,) ``m²(daughter-1, C)`` — the pair/triple invariant of the
        pions involved, looked up directly by their sorted index tuple."""
        return self.sq[tuple(sorted(d1 + C))]

    def q(self, X, d1, d2):
        """Breakup momentum ``q(X; 1, 2)`` — memoized by the mass
        descriptors ``(X, d1, d2)``."""
        return self._q[(X, d1, d2)]

    def theta(self, X, A, C, d1, d2):
        """Helicity angle (radians) of ``A -> 1+2`` in the chain
        ``X -> A+C`` — memoized by the five mass descriptors, so identical
        angles across the 24 rows are computed once."""
        return self._theta[(X, A, C, d1, d2)]


def _helicity_cos(qX, qA, mX, mA, mC, m1, m2, m1C2, m2C2):
    """cos of the helicity angle of ``A -> 1+2`` in the chain ``X->A+C``.

    Pure invariant-mass formula (triangle law), no boosts needed::

        cosθ = A [m²(1C) − m²(2C) − (m1²−m2²)((X²+A²−C²)/A² − 1)]
               / (4·q(X;A,C)·q(A;1,2)·X)

    *qX*, *qA* are the pre-computed breakup momenta ``q(X;A,C)`` and
    ``q(A;1,2)`` — the caller already has them, so they are passed in
    instead of recomputed.  *m1C2*, *m2C2* are the invariant mass-squares
    of the cross pairs (1,C) and (2,C).
    """
    X, A, C = mX, mA, mC
    num = m1C2 - m2C2 - (m1 ** 2 - m2 ** 2) * \
        ((X ** 2 + A ** 2 - C ** 2) / A ** 2 - 1.0)
    return np.clip(A * num / (4.0 * qX * qA * X), -1.0, 1.0)


# ═══════════════════════════════════════════════════════════════════
# Topology 0: B → ρ₁(π⁺₁π⁻₁) ρ₂(π⁺₂π⁻₂)  (vectorised over events)
# ═══════════════════════════════════════════════════════════════════

def _topo_rhorho(K, o):
    """Vectorised ρρ topology; *o* = (o0, o1, o2, o3) pion indices in the
    block's order (permuted), *K* the shared :class:`_PionCache` (B is
    the implicit parent).  The block's pions are ``K.p(o0..o3)``.

    Returns ``(m1, m2, q0, q1, q2, phi, theta1, theta2)``.
    """
    o0, o1, o2, o3 = o
    P1, P2 = tuple(sorted((o0, o1))), tuple(sorted((o2, o3)))  # ρ₁, ρ₂ pairs
    i0, i1, i2, i3 = (o0,), (o1,), (o2,), (o3,)

    m1 = K.mass[P1]                         # ρ₁ = (π⁺₁, π⁻₁)
    m2 = K.mass[P2]                         # ρ₂ = (π⁺₂, π⁻₂)
    q0 = K.q(_B, P1, P2)                    # B breakup momentum

    q1 = K.q(P1, i0, i1)                    # ρ₁ → π⁺₁π⁻₁
    th1 = K.theta(_B, P1, P2, i0, i1)       # θ₁ (helicity of ρ₁)
    q2 = K.q(P2, i2, i3)                    # ρ₂ → π⁺₂π⁻₂
    th2 = K.theta(_B, P2, P1, i2, i3)       # θ₂ (helicity of ρ₂)

    # azimuth of π⁺₁ around the ρ₁ axis (P1 direction), from π⁺₂
    p0, p1, p2 = K.p(o0), K.p(o1), K.p(o2)
    phi = _azimuth(p0[:, 1:], (p0 + p1)[:, 1:], p2[:, 1:])

    return m1, m2, q0, q1, q2, phi, th1, th2


# ═══════════════════════════════════════════════════════════════════
# Topology 1: B → R₁(R₂(π⁺₁π⁻₁)π⁺₂) π⁻₂  (sequential chain)
# ═══════════════════════════════════════════════════════════════════

def _topo_chain(K, o):
    """Vectorised chain topology; *o* = (o0, o1, o2, o3) pion indices in
    the block's order, *K* the shared :class:`_PionCache`.

    R₂ = (π⁺₁, π⁻₁), R₁ = (π⁺₁, π⁻₁, π⁺₂), bachelor π⁻₂.

    Returns ``(m_R1, m_R2, q0, q1, q2, phi, theta1, theta2)``.
    """
    o0, o1, o2, o3 = o
    P2 = tuple(sorted((o0, o1)))                     # R₂ = (π⁺₁, π⁻₁)
    T1 = tuple(sorted((o0, o1, o2)))                 # R₁ = all but bachelor
    i0, i1, i2, i3 = (o0,), (o1,), (o2,), (o3,)

    m_R2 = K.mass[P2]
    m_R1 = K.mass[T1]

    q0 = K.q(_B, T1, i3)                            # B → R₁ π⁻₂
    th1 = K.theta(_B, T1, i3, P2, i2)               # θ₁ (helicity of R₁)
    q1 = K.q(T1, P2, i2)                            # R₁ → R₂ π⁺₂
    q2 = K.q(P2, i0, i1)                            # R₂ → π⁺₁π⁻₁
    th2 = K.theta(T1, P2, i2, i0, i1)               # θ₂ (helicity of R₂)

    # ── φ: one boost to the R₁ rest frame ─────────────────────────
    # (only here do we need the actual 4-vectors, to build R₁'s axis)
    p0, p1, p2, p3 = K.p(o0), K.p(o1), K.p(o2), K.p(o3)
    R2 = p0 + p1
    R1 = R2 + p2
    beta1 = R1[:, 1:] / R1[:, 0:1]
    vop = _boost3_op(beta1)                 # v², γ, (γ−1)/v² — once for 3 boosts
    phi = _azimuth(_boost3(p0, vop), _boost3(R2, vop), _boost3(p3, vop))

    return m_R1, m_R2, q0, q1, q2, phi, th1, th2


# Topology 2: mirror of the chain — same R₂ = (π⁺₁, π⁻₁) but the
# middle pion is π⁻₂ and the bachelor π⁺₂:
# B → R₁(R₂(π⁺₁π⁻₁)π⁻₂) π⁺₂.
_MIRROR_PERM = [0, 1, 3, 2]


def _topo_chain_mirror(K, o):
    """Mirror of the chain — swap π⁺₂ ↔ π⁻₂ (indices 2, 3); the
    R₂ = (π⁺₁, π⁻₁) pair is unchanged."""
    return _topo_chain(K, (o[0], o[1], o[3], o[2]))


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


def _as_1d(v, n, default):
    """Return a length-*n* float array: fill with *default* if None,
    broadcast scalars, otherwise pass through."""
    if v is None:
        return np.full(n, default)
    v = np.asarray(v, dtype=float)
    return np.full(n, float(v)) if v.ndim == 0 else v


def momenta_to_data_full(momenta, weight=None, frac=None, time=None):
    """Convert B → 4π momenta to the fitter data npz arrays (reference).

    Computes all 24 rows directly from the momenta (8 blocks × 3
    topologies).  This is the exact reference implementation; the
    default :func:`momenta_to_data` produces the same output by
    computing only the 4 B blocks and transforming them to the 4 CP
    blocks.

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

    The 24 rows are all filled: ``8 blocks × 3 topologies``
    (ρρ, chain, chain mirror), see the module docstring for the
    conventions.
    """
    momenta = np.asarray(momenta, dtype=float)
    n = len(momenta)
    mass = np.zeros((n, 24, 2))
    q = np.zeros((n, 24, 3))
    angles = np.zeros((n, 24, 3))

    # Boost every event to the B rest frame (the input may be in the
    # lab frame).
    tot = momenta.sum(1)
    betaB = tot[:, 1:] / tot[:, 0:1]
    pb = _boost(momenta, _boost3_op(betaB[:, None, :]))  # (n, 4, 4) B rest frame

    # Shared kinematic cache — all blocks index it by pion order; no
    # per-block tensor copies, masses computed once.
    K = _PionCache(pb)

    for b in range(8):
        perm = _IDENTICAL_PERMS[_CP_PERM_ORDER[b % 4] if b >= 4 else b % 4]
        order = np.array(perm)
        if b >= 4:                              # CP: exchange π⁺ ↔ π⁻
            order = order[list(_CP_INDEX)]
        o = tuple(int(x) for x in order)
        for t, topo in enumerate(_TOPOLOGIES):
            row = b * 3 + t
            m1, m2v, q0, q1, q2, phi, th1, th2 = topo(K, o)
            if b >= 4:                          # CP: azimuth flips sign
                phi = -phi
            mass[:, row, 0], mass[:, row, 1] = m1, m2v
            q[:, row] = np.stack([q0, q1, q2], axis=-1)
            angles[:, row] = np.stack([phi, th1, th2], axis=-1)

    return {
        "mass": mass,
        "q": q,
        "angles": angles,
        "frac": _as_1d(frac, n, 0.5),
        "time": _as_1d(time, n, 0.0),
        "bkg_raw": np.zeros(n),
        "weight": _as_1d(weight, n, 1.0),
    }


def momenta_to_data(momenta, weight=None, frac=None, time=None):
    """Convert B → 4π momenta to the fitter data npz arrays (default).

    The 4 B blocks (blocks 0–3) are computed directly; the 4 CP blocks
    (4–7) are filled by transforming the B values (CP: π⁺ ↔ π⁻) — no
    CP recomputation.  Output is identical to the reference
    :func:`momenta_to_data_full` (same test tolerances).

    Each CP block *c* (4–7) is the charge conjugate of the B block with
    the same identical-particle permutation shape,
    ``s = _CP_PERM_ORDER.index(c % 4)``.  The exact relations are:

      ρρ row:        mass, q identical;  θ₁, θ₂ → π−θ;    φ → −φ
      chain row:     partner is the source *mirror* row:
                     mass, q, θ₁ identical; θ₂ → π−θ₂;   φ → π−φ
      mirror row:    partner is the source *chain* row:
                     mass, q, θ₁ identical; θ₂ → π−θ₂;   φ → π−φ
    """
    momenta = np.asarray(momenta, dtype=float)
    n = len(momenta)
    mass = np.zeros((n, 24, 2))
    q = np.zeros((n, 24, 3))
    angles = np.zeros((n, 24, 3))

    # Boost every event to the B rest frame.
    tot = momenta.sum(1)
    betaB = tot[:, 1:] / tot[:, 0:1]
    pb = _boost(momenta, _boost3_op(betaB[:, None, :]))  # (n, 4, 4) B rest frame

    # Shared kinematic cache.
    K = _PionCache(pb)

    # ── B blocks (0-3): computed directly ─────────────────────────
    for b in range(4):
        o = tuple(_IDENTICAL_PERMS[b])
        for t, topo in enumerate(_TOPOLOGIES):
            row = b * 3 + t
            m1, m2v, q0, q1, q2, phi, th1, th2 = topo(K, o)
            mass[:, row, 0], mass[:, row, 1] = m1, m2v
            q[:, row] = np.stack([q0, q1, q2], axis=-1)
            angles[:, row] = np.stack([phi, th1, th2], axis=-1)

    # ── CP blocks (4-7): one vectorised gather + transforms ────────
    # Source B row for each of the 12 CP rows (rows 12..23): the ρρ row
    # of the same-shape block, the chain row <- source *mirror* row,
    # the mirror row <- source *chain* row.
    cp_src = np.array([0, 2, 1,  6, 8, 7,  3, 5, 4,  9, 11, 10])
    rho = np.array([1, 0, 0] * 4, dtype=bool)          # ρρ rows of the CP blocks

    mass[:, 12:] = mass[:, cp_src]
    q[:, 12:] = q[:, cp_src]

    phi = angles[:, :, 0]
    src_phi = phi[:, cp_src]
    # ρρ rows: φ → −φ; chain/mirror rows: φ → π−φ (wrapped)
    angles[:, 12:, 0] = np.where(
        rho, -src_phi, (np.pi - src_phi + np.pi) % (2 * np.pi) - np.pi)
    # ρρ rows: θ₁ → π−θ₁; chain/mirror rows: θ₁ unchanged
    angles[:, 12:, 1] = np.where(
        rho, np.pi - angles[:, cp_src, 1], angles[:, cp_src, 1])
    # all rows: θ₂ → π−θ₂
    angles[:, 12:, 2] = np.pi - angles[:, cp_src, 2]

    return {
        "mass": mass,
        "q": q,
        "angles": angles,
        "frac": _as_1d(frac, n, 0.5),
        "time": _as_1d(time, n, 0.0),
        "bkg_raw": np.zeros(n),
        "weight": _as_1d(weight, n, 1.0),
    }


def momenta_to_data_samesign(momenta, m_pi=M_PION):
    """Build the same-charge-pair kinematics of the topology
    B → (π⁺₁π⁺₂)(π⁻₁π⁻₂).

    Returns the five phase-space variables::

        m_pp, m_mm     — m(π⁺₁π⁺₂), m(π⁻₁π⁻₂)
        cos_theta1     — cos of the (π⁺π⁺) helicity angle, restricted to
                         [0, 1]
        cos_theta2     — cos of the (π⁻π⁻) helicity angle, restricted to
                         [0, 1]
        phi            — azimuth of π⁺₁ around the (π⁺π⁺) axis, from π⁻₁

    The helicity cosines are clipped to **[0, 1]** (θ ∈ [0, π/2]): each
    pair holds identical pions, so (θ) and (π − θ) are the same physical
    state — restricting to the upper hemisphere removes the double
    counting, and the result is invariant under swapping the two π⁺ (or
    the two π⁻) of a pair.
    """
    momenta = np.asarray(momenta, dtype=float)
    n = len(momenta)
    # boost to the B rest frame (the kinematics are frame-independent)
    tot = momenta.sum(1)
    betaB = tot[:, 1:] / tot[:, 0:1]
    pb = _boost(momenta, _boost3_op(betaB[:, None, :]))
    K = _PionCache(pb)

    P_pp, P_mm = (0, 2), (1, 3)          # (π⁺₁,π⁺₂) and (π⁻₁,π⁻₂)
    m1, m2 = K.mass[P_pp], K.mass[P_mm]
    mB = K.mass[_B]
    q0 = K.q(_B, P_pp, P_mm)
    q1 = K.q(P_pp, (0,), (2,))
    q2 = K.q(P_mm, (1,), (3,))

    # raw helicity cosines (daughter w.r.t. the pair axis)
    c1 = _helicity_cos(q0, q1, mB, m1, m2, m_pi, m_pi,
                       K.sq[(0, 1, 3)], K.sq[(1, 2, 3)])
    c2 = _helicity_cos(q0, q2, mB, m2, m1, m_pi, m_pi,
                       K.sq[(0, 1, 2)], K.sq[(0, 2, 3)])
    # restrict to cosθ ∈ [0, 1]: reflect the daughter into the upper
    # hemisphere (identical pions make θ and π−θ the same state)
    up1 = c1 >= 0.0
    up2 = c2 >= 0.0
    cos_t1 = np.where(up1, c1, -c1)
    cos_t2 = np.where(up2, c2, -c2)

    # φ must use the *relabelled* daughters (the ones kept by the θ
    # restriction): vec = the upper π⁺, ref = the upper π⁻
    n = len(momenta)
    idx = np.arange(n)
    p_plus = pb[idx, np.where(up1, 0, 2)]        # (n, 4) upper π⁺
    p_minus = pb[idx, np.where(up2, 1, 3)]       # (n, 4) upper π⁻
    axis = (pb[:, 0] + pb[:, 2])[:, 1:]          # (π⁺π⁺) momentum
    phi = _azimuth(p_plus[:, 1:], axis, p_minus[:, 1:])

    return {"m_pp": m1, "m_mm": m2,
            "cos_theta1": cos_t1,
            "cos_theta2": cos_t2,
            "phi": phi}


def data_to_momentum_samesign(m_pp, m_mm, cos_theta1, cos_theta2, phi,
                              m_B=M_B_MESON, m_pi=M_PION):
    """Build B → (π⁺₁π⁺₂)(π⁻₁π⁻₂) momenta from the same-charge-pair
    kinematics of :func:`momenta_to_data_samesign`.

    The geometry is identical to :func:`data_to_momentum`
    (:func:`build_momenta`): pair₁ of mass *m_pp* along +z, pair₂ of
    mass *m_mm* along −z, with the helicity cosines *cos_theta1/2* and
    azimuth *phi*.  Only the *output order* differs — the pairs are
    same-charge, so pair₁ holds the two π⁺ and pair₂ the two π⁻::

        build_momenta → [π⁺₁, π⁺₂, π⁻₁, π⁻₂]   (same-charge pairs)
        standard order → [π⁺₁, π⁻₁, π⁺₂, π⁻₂]   (permutation (0, 2, 1, 3))

    The azimuth sign is flipped (−φ): the samesign φ is measured from
    the π⁻ reference, the generator's from the π⁺ plane.

    Returns
    -------
    momenta : ndarray (n, 4, 4) in the B rest frame, ordered
        [π⁺₁, π⁻₁, π⁺₂, π⁻₂].
    """
    m_pp = np.asarray(m_pp, dtype=float)
    mom = build_momenta(m_pp, m_mm, cos_theta1, cos_theta2, -phi,
                        m_B=m_B, m_pi=m_pi)
    return mom[:, [0, 2, 1, 3]]


def data_to_momentum(data, m_pi=M_PION):
    """Reverse of :func:`momenta_to_data`: reconstruct B → 4π momenta.

    The kinematic arrays encode the event in the same parametrisation
    as the flat phase-space generator :func:`generate_b4pi` — row 0 of
    each block is the ρρ topology with the generator's variables::

        mass[row, 0], mass[row, 1]  =  m₁, m₂        (the two di-pion masses)
        q[row, 0..2]                =  q_B, q₁, q₂   (breakup momenta)
        angles[row, 0..2]           =  φ, θ₁, θ₂     (azimuth, helicities)

    so the momenta are rebuilt with the generator's canonical geometry
    (B at rest, ρ₁ along +z, π⁺₁ decay plane at azimuth 0, ρ₂ plane
    rotated by the azimuth).  Only the *first* row of each block is
    used (all 24 rows describe the same event).

    Parameters
    ----------
    data : dict
        The arrays from :func:`momenta_to_data` (``mass`` (n,24,2),
        ``q`` (n,24,3), ``angles`` (n,24,3), ...).
    m_pi : float
        Pion mass (default :data:`M_PION`).

    Returns
    -------
    momenta : ndarray (n, 4, 4)
        Pion 4-momenta (E, px, py, pz) in the B rest frame, ordered
        [π⁺₁, π⁻₁, π⁺₂, π⁻₂].
    """
    mass = np.asarray(data["mass"])[:, 0]       # (n, 2) = (m₁, m₂)
    q = np.asarray(data["q"])[:, 0]             # (n, 3) = (q_B, q₁, q₂)
    ang = np.asarray(data["angles"])[:, 0]      # (n, 3) = (φ, θ₁, θ₂)
    m1, m2 = mass[:, 0], mass[:, 1]
    qB, q1, q2 = q[:, 0], q[:, 1], q[:, 2]
    phi, th1, th2 = ang[:, 0], ang[:, 1], ang[:, 2]

    # the data azimuth is measured from π⁺₂, the generator's from π⁺₁
    # (the two planes), so the generator angle is −φ
    return build_momenta(m1, m2, np.cos(th1), np.cos(th2), -phi,
                         q_B=qB, m_pi=m_pi)


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
