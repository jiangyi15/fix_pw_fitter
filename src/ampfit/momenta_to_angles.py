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
        if n_y < 1e-12:                      # z1 ∥ z0 — pick a reference plane
            ref = (np.array([1.0, 0.0, 0.0]) if abs(z0[0]) < 0.9
                   else np.array([0.0, 1.0, 0.0]))
            y1v = np.cross(z0, ref)
            if _norm(y1v) < 1e-12:
                y1v = np.array([0.0, 1.0, 0.0])
            y1 = y1v / _norm(y1v)
        else:
            y1 = y1v / n_y
        x1 = np.cross(y1, z1)

        # triad for child0 (z along its momentum) — exact
        t0 = np.stack([x1, y1, z1])
        # triad for child1 (antipodal: z along −z1) — built from q1 direction
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
