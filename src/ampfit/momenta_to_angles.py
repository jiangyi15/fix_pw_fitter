"""
momenta_to_angles — per-vertex Euler angles (φ_v, θ_v) from final-state
4-momenta along a two-body decay chain.

Convention (matches the helicity-angle engine and its angular formulas):

* Work in the top (CM) rest frame; the top carries a reference triad
  (x0, y0, z0) — by default the identity axes.
* At a two-body vertex  P → A + B, boost A and B into P's rest frame.
  The two daughter momenta are back-to-back.  Take the *first* daughter's
  momentum direction as the new z axis (z1):
      y1 = normalize(z0 × z1)          (z0 = parent reference z)
      x1 = y1 × z1
  and record the Euler pair
      θ_v = angle(z0, z1)
      φ_v = atan2(z1·y0, z1·x0)        (y0 → y1, z0 → z1 rotation)
  for vertex v.
* Recurse down the decay tree, boosting step by step into every decay rest
  frame.  A produced daughter's own reference triad is built from its actual
  momentum direction — the second (antipodal) daughter therefore naturally
  receives the (φ−π, π−θ) rotation of the parent axis system.
* Reference axes are carried across boosts as 4-vectors (pure-spatial in
  their own rest frame), so transverse components survive the boost exactly.

Output: ``angles[v] = (φ_v, θ_v)`` with v = position in ``chain.decays``
(DFS pre-order), i.e. the same ordering the helicity engine uses.
"""

import math

import numpy as np

# 4-vector layout: (E, px, py, pz)


def _norm(v):
    n = float(np.linalg.norm(v))
    return n


def _unit3(v):
    v = np.asarray(v, dtype=float)
    n = _norm(v)
    if n < 1e-12:
        return None
    return v / n


def _boost_4vector(p4, beta):
    """Lorentz-boost a 4-vector into the frame moving with velocity *beta*
    (standard form: E' = γ(E − β·p),  p' = p + (γ−1)(β·p)/β²·β − γEβ)."""
    p4 = np.asarray(p4, dtype=float)
    beta = np.asarray(beta, dtype=float)
    b2 = float(beta @ beta)
    if b2 < 1e-14:
        return p4
    gamma = 1.0 / np.sqrt(1.0 - b2)
    E, p = p4[0], p4[1:]
    bp = float(beta @ p)
    p_par = (bp / b2) * beta
    E2 = gamma * (E - bp)
    p2 = p + (gamma - 1.0) * p_par - gamma * E * beta
    return np.array([E2, p2[0], p2[1], p2[2]])


def _child_names(chain):
    return [str(o) for o in chain.decays[0].outs]


def build_node_momenta(chain, final_momenta):
    """4-momenta (in CM) of every particle of the chain.

    Args:
        chain: ampfit DecayChain (DFS pre-order decays).
        final_momenta: dict final-particle-name → (E,px,py,pz) in the CM
            (top rest) frame; all final momenta sum to (m_top, 0,0,0).
    Returns:
        dict name → (E,px,py,pz).
    """
    mom = dict(final_momenta)
    leaves = set()
    for d in chain.decays:
        for o in d.outs:
            if o.name not in [dd.core.name for dd in chain.decays]:
                leaves.add(o.name)

    # bottoms-up: inner = sum of its descendant leaves
    decay_cores = {d.core.name: d for d in chain.decays}

    def descend(name, seen):
        if name in mom:
            return mom[name]
        d = decay_cores.get(name)
        if d is None:
            raise KeyError(name)
        total = None
        for o in d.outs:
            m = descend(o.name, seen)
            total = m if total is None else total + m
        mom[name] = total
        return total

    descend(chain.top, set())
    return mom


def decay_angles_from_momenta(chain, final_momenta, top_triad=None):
    """Per-vertex Euler angles (φ_v, θ_v) along the decay chain.

    Args:
        chain: ampfit DecayChain.
        final_momenta: dict name → (E,px,py,pz) of the final particles, in
            the top CM frame (sums to (m_top, 0,0,0)).
        top_triad: optional (x0,y0,z0) orthonormal triad of the top at rest
            (default identity).

    Returns:
        ``[ (φ_v, θ_v), ... ]`` in ``chain.decays`` order.
    """
    mom = build_node_momenta(chain, final_momenta)
    if top_triad is None:
        top_triad = np.eye(3)

    decays = chain.decays
    core_idx = {d.core.name: i for i, d in enumerate(decays)}

    # triad per particle: 3x3 rows [x,y,z], expressed in its own rest frame
    # (stored as 3-vectors; only needed for angular reference).
    triad = {decays[0].core.name: np.asarray(top_triad, dtype=float)}
    angles = {}

    for i, d in enumerate(decays):
        pname = d.core.name
        p4 = mom[pname]
        E = float(p4[0])
        beta_p = p4[1:] / E if E > 0 else np.zeros(3)
        T = triad[pname]
        z0 = T[2]
        x0 = T[0]
        y0 = T[1]

        # boost the two children into pname's rest frame
        c0, c1 = d.outs[0].name, d.outs[1].name
        q0 = _boost_4vector(mom[c0], beta_p)
        q1 = _boost_4vector(mom[c1], beta_p)
        z1 = _unit3(q0[1:])
        if z1 is None:
            z1 = _unit3(q1[1:])
        if z1 is None:
            z1 = np.array([0.0, 0.0, 1.0])

        cz = float(np.clip(z0 @ z1, -1.0, 1.0))
        theta = float(np.arccos(cz))
        # φ from the spherical decomposition of z1 in the parent triad
        xc = float(z1 @ x0)
        yc = float(z1 @ y0)
        phi = float(np.arctan2(yc, xc))
        angles[i] = (phi, theta)

        # build the new triad of the first daughter (z along its momentum)
        y1v = np.cross(z0, z1)
        n_y = _norm(y1v)
        if n_y < 1e-12:                      # z1 ∥ z0 — collinear: share ONE
            # perpendicular reference x for BOTH children (z=±z1), so the
            # antipodal daughter gets the same x and proper (x,−y,−z).
            ref = (np.array([1.0, 0.0, 0.0]) if abs(z0[0]) < 0.9
                   else np.array([0.0, 1.0, 0.0]))
            xref = np.cross(z0, ref)
            nxr = _norm(xref)
            xref = xref / nxr if nxr > 1e-12 else np.array([1.0, 0.0, 0.0])
            z1n = np.asarray(z1, dtype=float) / _norm(z1)
            y1 = np.cross(z1n, xref)
            x1 = np.asarray(xref, dtype=float)
            z1b = -z1n
            y1b = np.cross(z1b, xref)
            x1b = np.asarray(xref, dtype=float)
            t0 = np.stack([x1, y1, z1n])
            t1 = np.stack([x1b, y1b, z1b])
        else:
            y1 = y1v / n_y
            x1 = np.cross(y1, z1)
            t0 = np.stack([x1, y1, z1])
            z1b = _unit3(q1[1:])
            if z1b is None:
                z1b = -z1
            y1b = np.cross(z0, z1b)
            nyb = _norm(y1b)
            if nyb < 1e-12:
                y1b = y1
            else:
                y1b = y1b / nyb
            y1b = np.asarray(y1b, dtype=float)
            x1b = np.cross(y1b, z1b)
            t1 = np.stack([x1b, y1b, z1b])

        if c0 in core_idx and c0 not in triad:
            triad[c0] = t0
        if c1 in core_idx and c1 not in triad:
            triad[c1] = t1

    return [angles[i] for i in range(len(decays))]


def _two_body_momentum(M, m1, m2):
    """|p| of each daughter in a two-body decay M → m1 + m2 (rest)."""
    return float(np.sqrt(max(((M ** 2 - (m1 + m2) ** 2)
                              * (M ** 2 - (m1 - m2) ** 2)), 0.0))) / (2 * M)


def angles_to_momenta(chain, angles, top_triad=None):
    """Inverse of :func:`decay_angles_from_momenta`.

    Given one Euler pair ``(φ_v, θ_v)`` per vertex (in ``chain.decays``
    order) and the particle masses of the chain, build the final-state
    4-momenta in the top (CM) rest frame.

    The procedure mirrors the forward direction exactly:
      * parent triad + (φ, θ) give the first-daughter direction
        u = sinθ(cosφ·x + sinφ·y) + cosθ·z  (two-body |p| from the masses);
      * the daughter triad is built with z along its momentum and
        y = z_parent × z (the second daughter automatically gets the
        opposite-axis rotation);
      * daughter rest-frame momenta are boosted back into the current frame
        with the parent's velocity (rest→CM uses boost(·, −β_parent)).

    Returns a dict final-particle-name → (E, px, py, pz).
    """
    if top_triad is None:
        top_triad = np.eye(3)

    masses = {}
    for d in chain.decays:
        masses[d.core.name] = float(d.core.mass)
        for o in d.outs:
            masses.setdefault(o.name, float(o.mass))
    decays = chain.decays
    core_idx = {d.core.name: i for i, d in enumerate(decays)}

    p4 = {decays[0].core.name: np.array([masses[decays[0].core.name], 0., 0., 0.])}
    triad = {decays[0].core.name: np.asarray(top_triad, dtype=float)}

    def outs_of(name):
        for d in decays:
            if d.core.name == name:
                return [o.name for o in d.outs]
        return []

    def _new_triad(z0, z1):
        """(x1,y1,z1) for the first daughter (z along z1)."""
        y1v = np.cross(z0, z1)
        n = _norm(y1v)
        if n < 1e-12:
            ref = (np.array([1.0, 0.0, 0.0]) if abs(z0[0]) < 0.9
                   else np.array([0.0, 1.0, 0.0]))
            v = np.cross(z0, ref)
            y1 = v / _norm(v) if _norm(v) > 1e-12 else np.array([0.0, 1.0, 0.0])
        else:
            y1 = y1v / n
        x1 = np.cross(y1, z1)
        return np.stack([x1, y1, z1])

    for i, d in enumerate(decays):
        pname = d.core.name
        M = masses[pname]
        phi, theta = angles[i]
        T = triad[pname]
        x0, y0, z0 = T[0], T[1], T[2]
        c0, c1 = outs_of(pname)
        m0, m1 = masses[c0], masses[c1]
        p = _two_body_momentum(M, m0, m1)
        u = (math.sin(theta) * (math.cos(phi) * x0 + math.sin(phi) * y0)
             + math.cos(theta) * z0)
        E0 = math.hypot(p, m0)
        E1 = math.hypot(p, m1)
        rest0 = np.array([E0, *(p * u).tolist()])
        rest1 = np.array([E1, *(-p * u).tolist()])
        # boost into the current frame (parent may be moving)
        beta_p = p4[pname][1:] / p4[pname][0] if p4[pname][0] > 0 else np.zeros(3)
        p4[c0] = _boost_4vector(rest0, -beta_p)
        p4[c1] = _boost_4vector(rest1, -beta_p)
        # reference triads of the daughters (z along their momentum)
        triad[c0] = _new_triad(z0, u)
        triad[c1] = _new_triad(z0, -u)

    return {o.name: p4[o.name] for d in decays for o in d.outs
            if o.name not in core_idx}
