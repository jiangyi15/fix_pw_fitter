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
    """Per-vertex Euler angles (θ_v, φ_v) via top→leaf successive boosts.

    The event momenta are carried DOWN the decay tree: at every vertex the
    parent is at rest (top in the CM, then each resonance boosted step by
    step along its own velocity in the frame where it was produced), so the
    helicity axes follow the tree and no single-boost/Wigner distortion
    appears.  Frames are stored as (x, z) pairs (y = z×x).
    """
    mom_cm = build_node_momenta(chain, final_momenta)
    if top_triad is None:
        top_triad = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # x, z
    else:
        top_triad = np.asarray(top_triad, dtype=float)
        if top_triad.shape == (3, 3):
            top_triad = top_triad[[0, 2]]

    decays = chain.decays
    vmap = {d.core.name: i for i, d in enumerate(decays)}

    # subtree members per particle (names incl itself)
    subtree = {}

    def collect(name):
        if name in subtree:
            return subtree[name]
        out = {name}
        for d in decays:
            if d.core.name == name:
                for o in d.outs:
                    out |= collect(o.name)
        subtree[name] = out
        return out

    for d in decays:
        collect(d.core.name)

    def child_xz(zc, x0, z0):
        zc = np.asarray(zc, dtype=float)
        xv = zc * float(z0 @ zc) - z0      # R_y(theta)R_z(phi) image of parent x
        n = _norm(xv)
        if n > 1e-9:
            xc = xv / n
        else:
            ref = (np.array([1.0, 0.0, 0.0]) if abs(z0[0]) < 0.9
                   else np.array([0.0, 1.0, 0.0]))
            v = np.cross(z0, ref)
            nv = _norm(v)
            xc = v / nv if nv > 1e-12 else np.array([1.0, 0.0, 0.0])
        return np.stack([xc, zc])

    angles = {}

    def rec(name, p4_in_rest, T):
        """*name* is at rest; p4_in_rest maps its subtree particles to
        4-momenta in this rest frame."""
        i = vmap[name]
        d = decays[i]
        c0, c1 = d.outs[0].name, d.outs[1].name
        q0 = p4_in_rest[c0]
        q1 = p4_in_rest[c1]
        x0, z0 = T[0], T[1]
        y0 = np.cross(z0, x0)
        z1 = _unit3(q0[1:])
        if z1 is None:
            z1 = _unit3(q1[1:])
        if z1 is None:
            z1 = np.array([0.0, 0.0, 1.0])
        theta = float(np.arccos(np.clip(float(z0 @ z1), -1.0, 1.0)))
        phi = float(np.arctan2(float(z1 @ y0), float(z1 @ x0)))
        angles[i] = (phi, theta)

        z1b = _unit3(q1[1:])
        if z1b is None:
            z1b = -z1
        t0 = child_xz(z1, x0, z0)
        t1 = child_xz(z1b, x0, z0)

        for c, tc in ((c0, t0), (c1, t1)):
            if c in vmap:                      # inner: descend (successive boost)
                qc = p4_in_rest[c]
                Ec = float(qc[0])
                beta = qc[1:] / Ec if Ec > 0 else np.zeros(3)
                sub = {}
                for nm in subtree[c]:
                    sub[nm] = _boost_4vector(p4_in_rest[nm], beta)
                rec(c, sub, tc)

    top = decays[0].core.name
    rec(top, mom_cm, np.asarray(top_triad, dtype=float))
    return [angles[i] for i in range(len(decays))]



def _two_body_momentum(M, m1, m2):
    """|p| of each daughter in a two-body decay M → m1 + m2 (rest)."""
    return float(np.sqrt(max(((M ** 2 - (m1 + m2) ** 2)
                              * (M ** 2 - (m1 - m2) ** 2)), 0.0))) / (2 * M)


def angles_to_momenta(chain, angles, top_triad=None):
    """Inverse of :func:`decay_angles_from_momenta`.

    Given one Euler pair ``(φ_v, θ_v)`` per vertex (in ``chain.decays``
    order) and the particle masses of the chain, build the final-state
    4-momenta in the top (CM) rest frame.  Frames are stored as (x, z)
    pairs with y = z×x derived on demand.

    Returns a dict final-particle-name → (E, px, py, pz).
    """
    if top_triad is None:
        top_triad = np.array([[1.0, 0.0, 0.0],      # x
                              [0.0, 0.0, 1.0]])     # z
    else:
        top_triad = np.asarray(top_triad, dtype=float)
        if top_triad.shape == (3, 3):
            top_triad = top_triad[[0, 2]]

    masses = {}
    for d in chain.decays:
        masses[d.core.name] = float(d.core.mass)
        for o in d.outs:
            masses.setdefault(o.name, float(o.mass))
    decays = chain.decays
    core_idx = {d.core.name: i for i, d in enumerate(decays)}

    p4 = {decays[0].core.name:
          np.array([masses[decays[0].core.name], 0., 0., 0.])}
    triad = {decays[0].core.name: np.asarray(top_triad, dtype=float)}

    def outs_of(name):
        for d in decays:
            if d.core.name == name:
                return [o.name for o in d.outs]
        return []

    def child_xz(zc, x0, z0):
        zc = np.asarray(zc, dtype=float)
        xv = zc * float(z0 @ zc) - z0      # R_y(theta)R_z(phi) image of parent x
        n = _norm(xv)
        if n > 1e-9:
            xc = xv / n
        else:
            ref = (np.array([1.0, 0.0, 0.0]) if abs(z0[0]) < 0.9
                   else np.array([0.0, 1.0, 0.0]))
            v = np.cross(z0, ref)
            nv = _norm(v)
            xc = v / nv if nv > 1e-12 else np.array([1.0, 0.0, 0.0])
        return np.stack([xc, zc])

    for i, d in enumerate(decays):
        pname = d.core.name
        M = masses[pname]
        phi, theta = angles[i]
        T = triad[pname]
        x0, z0 = T[0], T[1]
        y0 = np.cross(z0, x0)
        c0, c1 = outs_of(pname)
        m0, m1 = masses[c0], masses[c1]
        p = _two_body_momentum(M, m0, m1)
        u = (math.sin(theta) * (math.cos(phi) * x0 + math.sin(phi) * y0)
             + math.cos(theta) * z0)
        rest0 = np.array([math.hypot(p, m0), *(p * u).tolist()])
        rest1 = np.array([math.hypot(p, m1), *(-p * u).tolist()])
        beta_p = p4[pname][1:] / p4[pname][0] if p4[pname][0] > 0 else np.zeros(3)
        p4[c0] = _boost_4vector(rest0, -beta_p)
        p4[c1] = _boost_4vector(rest1, -beta_p)
        triad[c0] = child_xz(u, x0, z0)
        triad[c1] = child_xz(-u, x0, z0)

    return {o.name: p4[o.name] for d in decays for o in d.outs
            if o.name not in core_idx}



# ---------------------------------------------------------------------------
# B→4π: original-layout angle arrays (all 24 rows)
# ---------------------------------------------------------------------------
# The new per-vertex geometry reproduces the ρρ row exactly, but the
# sequential "chain" / "mirror" rows of momenta_to_data use their own
# topology-specific helicity definitions.  For drop-in compatibility of the
# event files this module provides the original 24-row layout directly
# (delegating to ampfit.momenta_to_data), which matches by construction for
# every topology / identical-particle permutation / CP block.



# ---------------------------------------------------------------------------
# Alignment euler angles for spinful finals (multi-topology processes)
# ---------------------------------------------------------------------------
# The aligned frame of a particle is the recursive helicity frame our
# vertex geometry builds (each vertex maps the parent triad (x0,y0,z0) to the
# child triad (xc,zc) by the rotation between the two right-handed triads).
# Successive two-body boosts are collinear with the local z axis, which
# leaves the transverse axes intact, so the chain total rotation from the top
# frame to a particle is just the product of the per-vertex rotations.
# For a spinful final *f* in chain C relative to a reference chain R the
# alignment rotation is R_align = R_tot(C,f) . R_tot(R,f)^T and its z-y-z
# euler angles are the (alpha, beta, gamma) the angular engine consumes.


def _triad_to_matrix(T):
    """(N,2,3) rows (x,z) -> (N,3,3) columns (x, y=z x x, z)."""
    x = T[:, 0]
    z = T[:, 1]
    y = _cross_vec(z, x)
    return np.stack([x, y, z], axis=-1)


def _chain_total_rotations(chain, mom_name_arrays):
    """Total SO(3) rotation (top frame -> particle frame) per particle.

    Returns ``dict name -> (n_events,3,3)`` for every particle of *chain*
    (leaves and inner resonances).  *mom_name_arrays* maps particle name to
    an ``(n_events,4)`` momentum array in the top rest frame.
    """
    from ampfit.helicity_angle import decay_chain_leaves
    leaves = decay_chain_leaves(chain)
    names = [o.name for o in leaves]
    arrs = [np.asarray(mom_name_arrays[nm], dtype=float) for nm in names]
    N = arrs[0].shape[0]
    mom = dict(zip(names, arrs))
    decays = chain.decays
    vmap = {d.core.name: i for i, d in enumerate(decays)}
    subtree = {}

    def collect(name):
        if name in subtree:
            return subtree[name]
        out = {name}
        for d in decays:
            if d.core.name == name:
                for o in d.outs:
                    out |= collect(o.name)
        subtree[name] = out
        return out

    for d in decays:
        collect(d.core.name)
    core_names = {d.core.name for d in decays}
    leaves_set = {o.name for d in decays for o in d.outs} - core_names
    for name in subtree:
        if name in mom:
            continue
        tot = None
        for nm in subtree[name]:
            if nm == name or nm not in leaves_set:
                continue
            m = mom[nm]
            tot = m if tot is None else tot + m
        if tot is not None:
            mom[name] = tot

    def child_xz(zc, x0, z0):
        dot = np.einsum('ni,ni->n', z0, zc)
        xv = zc * dot[:, None] - z0
        n = np.linalg.norm(xv, axis=-1)
        ref = np.where((np.abs(z0[:, 0]) < 0.9)[:, None],
                       np.tile([1.0, 0.0, 0.0], (N, 1)),
                       np.tile([0.0, 1.0, 0.0], (N, 1)))
        fcross = _cross_vec(z0, ref)
        fn = np.linalg.norm(fcross, axis=-1)
        fallback = np.where((fn > 1e-12)[:, None],
                            fcross / np.maximum(fn, 1e-12)[:, None],
                            np.tile([0.0, 1.0, 0.0], (N, 1)))
        xc = np.where((n > 1e-9)[:, None],
                      xv / np.maximum(n, 1e-12)[:, None], fallback)
        return np.stack([xc, zc], axis=1)          # (N,2,3)

    Rtot = {}
    top = decays[0].core.name
    top_T = np.stack([np.tile([1.0, 0.0, 0.0], (N, 1)),
                      np.tile([0.0, 0.0, 1.0], (N, 1))], axis=1)
    Rtot[top] = np.tile(np.eye(3), (N, 1, 1))

    def rec(name, p4, T, Rparent):
        d = decays[vmap[name]]
        c0, c1 = d.outs[0].name, d.outs[1].name
        q0, q1 = p4[c0], p4[c1]
        x0, z0 = T[:, 0], T[:, 1]
        z1 = _unit3_vec(q0[:, 1:])
        bad = np.linalg.norm(z1, axis=-1) < 1e-12
        z1 = np.where(bad[:, None], _unit3_vec(q1[:, 1:]), z1)
        z1 = np.where((np.linalg.norm(z1, axis=-1) < 1e-12)[:, None],
                      np.tile([0.0, 0.0, 1.0], (N, 1)), z1)
        z1b = _unit3_vec(q1[:, 1:])
        badb = np.linalg.norm(z1b, axis=-1) < 1e-12
        z1b = np.where(badb[:, None], -z1, z1b)
        M0 = _triad_to_matrix(T)
        for c, zc in ((c0, z1), (c1, z1b)):
            tc = child_xz(zc, x0, z0)
            Rv = np.matmul(_triad_to_matrix(tc), np.swapaxes(M0, -1, -2))
            Rtot[c] = np.matmul(Rv, Rparent)
            if c in vmap:
                qc = p4[c]
                Ec = qc[:, 0]
                beta = np.zeros_like(qc[:, 1:])
                ok = np.abs(Ec) > 1e-12
                beta[ok] = qc[ok, 1:] / Ec[ok][:, None]
                sub = {}
                for nm in subtree[c]:
                    sub[nm] = _boost4_vec(p4[nm], beta)
                rec(c, sub, tc, Rtot[c])

    rec(top, mom, top_T, Rtot[top])
    return Rtot


def _rotation_matrix_to_su2(M):
    """(...,3,3) proper rotation -> (...,2,2) SU(2), unit quaternion route."""
    M = np.asarray(M, dtype=float)
    q0 = 0.5 * np.sqrt(np.maximum(1.0 + M[..., 0, 0] + M[..., 1, 1]
                                  + M[..., 2, 2], 0.0))
    q1 = (M[..., 2, 1] - M[..., 1, 2]) / np.maximum(4 * q0, 1e-12)
    q2 = (M[..., 0, 2] - M[..., 2, 0]) / np.maximum(4 * q0, 1e-12)
    q3 = (M[..., 1, 0] - M[..., 0, 1]) / np.maximum(4 * q0, 1e-12)
    neg = q0 < 0
    for q in (q0, q1, q2, q3):
        q = np.where(neg, -q, q)
    sh = q0.shape
    U = np.zeros(sh + (2, 2), dtype=np.complex128)
    U[..., 0, 0] = q0 - 1j * q3
    U[..., 0, 1] = -q2 - 1j * q1
    U[..., 1, 0] = q2 - 1j * q1
    U[..., 1, 1] = q0 + 1j * q3
    return U


def _massless_mask(p4, tol=1e-6):
    """Per-event mask for massless particles: m/E < tol (no rest frame)."""
    p4 = np.asarray(p4, dtype=float)
    E = np.abs(p4[..., 0])
    m2 = np.clip(E * E - np.sum(p4[..., 1:] ** 2, axis=-1), 0.0, None)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (np.sqrt(m2) / np.maximum(E, 1e-300)) < tol


def _chain_su2_frames(chain, mom_name_arrays):
    """tf-pwa-style per-particle SU(2) frame matrices for one chain.

    Returns ``(r_matrix, b_matrix)`` dicts name -> (n_events,2,2) where

      * ``b_matrix[j]`` = Boost_z_from_p(p_rest[j])  (j in its parent rest)
      * ``r_matrix[j]`` = U_vertex . b_matrix[parent] . r_matrix[parent]
                          (U_vertex = SU(2) of the vertex triad rotation),
      reproducing ``cal_helicity_angle`` in tf-pwa/cal_angle.py.
    """
    from ampfit.helicity_angle import decay_chain_leaves
    from ampfit.su2 import (Identity, _mul, Boost_z_from_p, Rotation_y,
                            Rotation_z)
    leaves = decay_chain_leaves(chain)
    names = [o.name for o in leaves]
    arrs = [np.asarray(mom_name_arrays[nm], dtype=float) for nm in names]
    N = arrs[0].shape[0]
    mom = dict(zip(names, arrs))
    decays = chain.decays
    vmap = {d.core.name: i for i, d in enumerate(decays)}
    subtree = {}

    def collect(name):
        if name in subtree:
            return subtree[name]
        out = {name}
        for d in decays:
            if d.core.name == name:
                for o in d.outs:
                    out |= collect(o.name)
        subtree[name] = out
        return out

    for d in decays:
        collect(d.core.name)
    core_names = {d.core.name for d in decays}
    leaves_set = {o.name for d in decays for o in d.outs} - core_names
    for name in subtree:
        if name in mom:
            continue
        tot = None
        for nm in subtree[name]:
            if nm == name or nm not in leaves_set:
                continue
            m = mom[nm]
            tot = m if tot is None else tot + m
        if tot is not None:
            mom[name] = tot

    I = Identity(N)
    r_matrix = {decays[0].core.name: I}
    b_matrix = {decays[0].core.name: I}

    def child_xz(zc, x0, z0):
        dot = np.einsum('ni,ni->n', z0, zc)
        xv = zc * dot[:, None] - z0
        n = np.linalg.norm(xv, axis=-1)
        ref = np.where((np.abs(z0[:, 0]) < 0.9)[:, None],
                       np.tile([1.0, 0.0, 0.0], (N, 1)),
                       np.tile([0.0, 1.0, 0.0], (N, 1)))
        fcross = _cross_vec(z0, ref)
        fn = np.linalg.norm(fcross, axis=-1)
        fallback = np.where((fn > 1e-12)[:, None],
                            fcross / np.maximum(fn, 1e-12)[:, None],
                            np.tile([0.0, 1.0, 0.0], (N, 1)))
        xc = np.where((n > 1e-9)[:, None],
                      xv / np.maximum(n, 1e-12)[:, None], fallback)
        return np.stack([xc, zc], axis=1)          # (N,2,3)

    top_T = np.stack([np.tile([1.0, 0.0, 0.0], (N, 1)),
                      np.tile([0.0, 0.0, 1.0], (N, 1))], axis=1)

    def rec(name, p4, T):
        d = decays[vmap[name]]
        c0, c1 = d.outs[0].name, d.outs[1].name
        q0, q1 = p4[c0], p4[c1]
        x0, z0 = T[:, 0], T[:, 1]
        z1 = _unit3_vec(q0[:, 1:])
        bad = np.linalg.norm(z1, axis=-1) < 1e-12
        z1 = np.where(bad[:, None], _unit3_vec(q1[:, 1:]), z1)
        z1 = np.where((np.linalg.norm(z1, axis=-1) < 1e-12)[:, None],
                      np.tile([0.0, 0.0, 1.0], (N, 1)), z1)
        z1b = _unit3_vec(q1[:, 1:])
        badb = np.linalg.norm(z1b, axis=-1) < 1e-12
        z1b = np.where(badb[:, None], -z1, z1b)
        y0 = _cross_vec(z0, x0)
        # slot-0 daughter angles measured from the parent axes
        cos_t0 = np.clip(np.einsum('ni,ni->n', z0, z1), -1.0, 1.0)
        theta0 = np.arccos(cos_t0)
        phi0 = np.arctan2(np.einsum('ni,ni->n', z1, y0),
                          np.einsum('ni,ni->n', z1, x0))
        for c, qc, slot in ((c0, q0, 0), (c1, q1, 1)):
            tc = child_xz(-z1 if slot else z1, x0, z0)
            # tf-pwa per-vertex SU2 from the daughter's helicity angles
            # (slot 1 uses phi0 - pi UNWRAPPED: wrapping would flip the
            #  SU(2) sign and break the tf-pwa branch)
            theta_c = theta0 if slot == 0 else (np.pi - theta0)
            phi_c = phi0 if slot == 0 else (phi0 - np.pi)
            Uv = _mul(Rotation_y(theta_c), Rotation_z(phi_c))
            rp = r_matrix[name]
            bp = b_matrix[name]
            r_matrix[c] = _mul(_mul(Uv, bp), rp)
            bm = Boost_z_from_p(qc)
            ml = _massless_mask(qc)
            if np.any(ml):
                bm = np.where(ml[..., None, None], Identity(qc.shape[0]), bm)
            b_matrix[c] = bm
            if c in vmap:
                Ec = qc[:, 0]
                beta = np.zeros_like(qc[:, 1:])
                ok = np.abs(Ec) > 1e-12
                beta[ok] = qc[ok, 1:] / Ec[ok][:, None]
                sub = {}
                for nm in subtree[c]:
                    sub[nm] = _boost4_vec(p4[nm], beta)
                rec(c, sub, tc)

    rec(decays[0].core.name, mom, top_T)
    return r_matrix, b_matrix


def _cm_reference_frames(mom_name_arrays, spinful_names):
    """Center-of-mass alignment reference per spinful final (tf-pwa rule2).

    For each final the reference frame is built from the LAB axes and the
    particle's own CM momentum only - completely independent of any decay
    chain.  ``r_ref = inv(r) Bp r`` with r the rotation taking the lab z
    axis onto the momentum direction and Bp the boost along it
    (``aligned_angle_ref_rule2`` in tf-pwa/cal_angle.py); the reference
    boost matrix is the identity.
    """
    from ampfit.su2 import Identity, _mul, inv as su2_inv, Boost_z_from_p
    r_ref = {}
    for nm in spinful_names:
        p = np.asarray(mom_name_arrays[nm], dtype=float)      # (N,4) in CM
        N = p.shape[0]
        p3 = _unit3_vec(p[:, 1:])
        bad = np.linalg.norm(p3, axis=-1) < 1e-12
        p3 = np.where(bad[:, None],
                      np.tile([0.0, 0.0, 1.0], (N, 1)), p3)
        z0 = np.tile([0.0, 0.0, 1.0], (N, 1))
        x0 = np.tile([1.0, 0.0, 0.0], (N, 1))
        y0 = _cross_vec(z0, x0)
        # triad (xc, p3) with the same x convention as the chain geometry
        xc = child_xz_cm(p3, x0, z0)
        yc = _cross_vec(p3, xc)
        Mc = np.stack([xc, yc, p3], axis=-1)
        M0 = np.tile(np.eye(3), (N, 1, 1))
        Rv = np.matmul(Mc, np.swapaxes(M0, -1, -2))
        r = _rotation_matrix_to_su2(Rv)
        Bp = Boost_z_from_p(p)
        rr = _mul(_mul(su2_inv(r), Bp), r)
        ml = _massless_mask(p)
        if np.any(ml):
            rr = np.where(ml[..., None, None], Identity(p.shape[0]), rr)
        r_ref[nm] = rr
    return r_ref


def child_xz_cm(zc, x0, z0):
    """child_xz triad (xc, zc) used by _cm_reference_frames (batch)."""
    N = zc.shape[0]
    dot = np.einsum('ni,ni->n', z0, zc)
    xv = zc * dot[:, None] - z0
    n = np.linalg.norm(xv, axis=-1)
    ref = np.where((np.abs(z0[:, 0]) < 0.9)[:, None],
                   np.tile([1.0, 0.0, 0.0], (N, 1)),
                   np.tile([0.0, 1.0, 0.0], (N, 1)))
    fcross = _cross_vec(z0, ref)
    fn = np.linalg.norm(fcross, axis=-1)
    fallback = np.where((fn > 1e-12)[:, None],
                        fcross / np.maximum(fn, 1e-12)[:, None],
                        np.tile([0.0, 1.0, 0.0], (N, 1)))
    xc = np.where((n > 1e-9)[:, None],
                  xv / np.maximum(n, 1e-12)[:, None], fallback)
    return xc


def aligned_euler_from_momenta(chains, mom_name_arrays, spinful_names,
                               final_rest=False, align_ref="center_mass"):
    """Alignment euler angles for spinful finals shared by >1 topology.

    Mirrors ``cal_angle_from_particle`` in tf-pwa/cal_angle.py with either
    alignment reference:

    * ``align_ref="center_mass"`` (default): each spinful final is aligned to
      the LAB/center-of-mass axes built from its own CM momentum only, so the
      result is INDEPENDENT of the decay chains (rule2).  Every chain gets a
      real alignment.
    * ``align_ref="chain"``: rule1 - the reference is the chain where the
      final is a direct top child (else the first chain) and its slice is
      zero.

    Every chain carries per-particle SU(2) frame matrices (rotation AND
    boost).  For chain *C* the aligned rotation of final *f* is

        R = r_ref[f] . inv(r_chain[f])
        (final_rest)  R = b_ref[f] . R . inv(b_chain[f])

    with ``(alpha, beta, gamma) = euler(R)`` (exact SU(2) sign, full 4pi).

    Args:
        chains: list of ampfit DecayChain (active topologies).
        mom_name_arrays: dict final-name -> (n_events, 4) momentum in CM.
        spinful_names: canonical-order list of spinful final names.
        final_rest: wrap R with the final-rest boosts (default False).
        align_ref: "center_mass" or "chain".

    Returns:
        dict name -> (n_events, n_chains, 3) euler columns ordered like
        *chains*, or None for a single active topology.
    """
    if len(chains) <= 1:
        return None
    from ampfit.su2 import Identity, _mul, inv as su2_inv, get_euler_angle
    frames = [_chain_su2_frames(dc, mom_name_arrays) for dc in chains]
    if align_ref == "center_mass":
        ref_r = _cm_reference_frames(mom_name_arrays, spinful_names)
        ref_b = {nm: Identity(mom_name_arrays[spinful_names[0]].shape[0])
                 for nm in spinful_names}
        ref_index = {nm: -1 for nm in spinful_names}    # no chain slice is zero
    else:
        ref_r = {}
        ref_b = {}
        for nm in spinful_names:
            dc = next((c for c in chains
                       if any(o.name == nm for o in c.decays[0].outs)),
                      None)
            dc = dc if dc is not None else chains[0]
            i = chains.index(dc)
            ref_r[nm] = frames[i][0][nm]
            ref_b[nm] = frames[i][1][nm]
        ref_index = {nm: chains.index(
            next((c for c in chains
                  if any(o.name == nm for o in c.decays[0].outs)), chains[0]))
            for nm in spinful_names}
    out = {}
    n_chains = len(chains)
    for nm in spinful_names:
        N = mom_name_arrays[spinful_names[0]].shape[0]
        cols = []
        rr, br = ref_r[nm], ref_b[nm]
        for i, (rm, bm) in enumerate(frames):
            if align_ref != "center_mass" and i == ref_index[nm]:
                a = np.zeros(N)
                b = np.zeros(N)
                g = np.zeros(N)
            else:
                R = _mul(rr, su2_inv(rm[nm]))
                if final_rest:
                    R = _mul(_mul(br, R), su2_inv(bm[nm]))
                a, b, g = get_euler_angle(R)
            cols.append(np.stack([a, b, g], axis=-1))
        out[nm] = np.stack(cols, axis=1)                # (N, n_chains, 3)
    return out



def momenta_to_data_angles(momenta, weight=None, frac=None, time=None):
    """B→4π data dict identical to ``ampfit.momenta_to_data``.

    Returns the canonical 24-row arrays (mass/q/angles/frac/time/bkg/weight)
    computed by the original geometry — guaranteed equal to the reference for
    all ρρ / chain / chain-mirror rows and all permutation & CP blocks.
    """
    from ampfit.momenta_to_data import momenta_to_data as _orig
    return _orig(momenta, weight=weight, frac=frac, time=time)


# ---------------------------------------------------------------------------
# per-vertex → original row-format triplets (B→4π)
# ---------------------------------------------------------------------------
# Relations derived numerically against ampfit.momenta_to_data (all
# identical-particle permutation blocks, to ~1e-13):
#   ρρ (two resonances each → ππ):   (φ,θ₁,θ₂) = (φ₁+φ₂, θ(R1), θ(R2))
#   chain (B→R1(→R2π⁺)π⁻):          (φ,θ₁,θ₂) = (φ₂+π, θ(R1), θ(R2))
# where *v* is the per-vertex angle list in chain.decays order and R2 is the
# deepest (→ππ) vertex.

def rho_original_triplet(v):
    """per-vertex (ρρ chain: B,R1,R2 → (φ,θ₁,θ₂) in momenta_to_data format."""
    return (wrap_angle(v[1][0] + v[2][0]), v[1][1], v[2][1])


def chain_original_triplet(v):
    """per-vertex (nested chain: B,R1,R2 → (φ,θ₁,θ₂) format."""
    return (wrap_angle(v[2][0]), v[1][1], v[2][1])


def wrap_angle(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Vectorized momenta → per-vertex angles (batch over events)
# ---------------------------------------------------------------------------
def _boost4_vec(p, beta):
    """Vectorized _boost_4vector for p4 shape (N,4), beta (N,3)."""
    p = np.asarray(p, dtype=float)
    beta = np.asarray(beta, dtype=float)
    b2 = np.sum(beta * beta, axis=-1)
    ok = b2 > 1e-14
    out = np.empty_like(p)
    g = np.sqrt(np.maximum(1.0 - b2, 1e-30))
    gamma = np.divide(1.0, g, out=np.ones_like(b2), where=ok)
    bp = np.einsum('ni,ni->n', beta, p[:, 1:])
    E2 = gamma * (p[:, 0] - bp)
    coef = np.where(ok, (gamma - 1.0) / np.where(ok, b2, 1.0), 0.0)
    p_par = beta * coef[:, None] * bp[:, None]
    p2 = p[:, 1:] + p_par - (gamma * p[:, 0])[:, None] * beta
    p2 = np.where(ok[:, None], p2, p[:, 1:])
    E2 = np.where(ok, E2, p[:, 0])
    out[:, 0] = E2
    out[:, 1:] = p2
    return out


def _unit3_vec(v):
    """Unit vectors for (N,3); zero rows stay zero."""
    n = np.linalg.norm(v, axis=-1)
    out = np.zeros_like(v)
    ok = n > 1e-12
    out[ok] = v[ok] / n[ok][:, None]
    return out


def _cross_vec(a, b):
    return np.cross(a, b)


def decay_angles_vectorized(chain, momenta):
    """Vectorized :func:`decay_angles_from_momenta`.

    *momenta*: per-final-particle 4-vectors in the chain's leaf order — either
    an ``(n_events, n_finals, 4)`` array or a list of ``n_finals`` arrays each
    of shape ``(n_events, 4)`` (E,px,py,pz in the top rest frame).  Works for
    any number of final particles.  Returns ``(phi, theta)`` arrays of shape
    ``(n_events, n_vertices)`` in ``chain.decays`` order.
    """
    # leaf names in pre-order
    from ampfit.helicity_angle import decay_chain_leaves
    leaves = decay_chain_leaves(chain)
    names = [o.name for o in leaves]
    if isinstance(momenta, np.ndarray) and momenta.ndim == 3:
        assert momenta.shape[1] == len(names)
        arrs = [np.asarray(momenta[:, j], dtype=float) for j in range(momenta.shape[1])]
    else:
        arrs = [np.asarray(m, dtype=float) for m in momenta]
        assert len(arrs) == len(names)
        N = arrs[0].shape[0]
        for a in arrs:
            assert a.shape == (N, 4)
    mom = {nm: arrs[j] for j, nm in enumerate(names)}
    decays = chain.decays
    vmap = {d.core.name: i for i, d in enumerate(decays)}

    # inner node momenta = sum of descendant leaves (vectorized)
    subtree = {}

    def collect(name):
        if name in subtree:
            return subtree[name]
        out = {name}
        for d in decays:
            if d.core.name == name:
                for o in d.outs:
                    out |= collect(o.name)
        subtree[name] = out
        return out

    for d in decays:
        collect(d.core.name)
    core_names = {d.core.name for d in decays}
    leaves_set = {o.name for d in decays for o in d.outs} - core_names
    for name in subtree:
        if name in mom:
            continue
        tot = None
        for nm in subtree[name]:
            if nm == name or nm not in leaves_set:
                continue
            m = mom[nm]
            tot = m if tot is None else tot + m
        if tot is not None:
            mom[name] = tot
    N = arrs[0].shape[0]

    def child_xz(zc, x0, z0):
        dot = np.einsum('ni,ni->n', z0, zc)
        xv = zc * dot[:, None] - z0      # rotation image of parent x
        n = np.linalg.norm(xv, axis=-1)
        ref = np.where((np.abs(z0[:, 0]) < 0.9)[:, None],
                       np.tile([1.0, 0.0, 0.0], (N, 1)),
                       np.tile([0.0, 1.0, 0.0], (N, 1)))
        fcross = _cross_vec(z0, ref)
        fn = np.linalg.norm(fcross, axis=-1)
        fallback = np.where((fn > 1e-12)[:, None], fcross / np.maximum(fn, 1e-12)[:, None],
                            np.tile([0.0, 1.0, 0.0], (N, 1)))
        xc = np.where((n > 1e-9)[:, None], xv / np.maximum(n, 1e-12)[:, None], fallback)
        return np.stack([xc, zc], axis=1)          # (N,2,3)

    phi = np.zeros((N, len(decays)))
    theta = np.zeros((N, len(decays)))
    top_triad = np.stack([np.tile([1.0, 0.0, 0.0], (N, 1)),
                          np.tile([0.0, 0.0, 1.0], (N, 1))], axis=1)  # (N,2,3)

    def rec(name, p4, T):
        i = vmap[name]
        d = decays[i]
        c0, c1 = d.outs[0].name, d.outs[1].name
        q0 = p4[c0]
        q1 = p4[c1]
        x0 = T[:, 0]
        z0 = T[:, 1]
        y0 = _cross_vec(z0, x0)
        z1 = _unit3_vec(q0[:, 1:])
        bad = np.linalg.norm(z1, axis=-1) < 1e-12
        z1 = np.where(bad[:, None], _unit3_vec(q1[:, 1:]), z1)
        z1 = np.where((np.linalg.norm(z1, axis=-1) < 1e-12)[:, None],
                      np.tile([0.0, 0.0, 1.0], (N, 1)), z1)
        cosv = np.clip(np.einsum('ni,ni->n', z0, z1), -1.0, 1.0)
        theta[:, i] = np.arccos(cosv)
        phi[:, i] = np.arctan2(np.einsum('ni,ni->n', z1, y0),
                               np.einsum('ni,ni->n', z1, x0))
        z1b = _unit3_vec(q1[:, 1:])
        badb = np.linalg.norm(z1b, axis=-1) < 1e-12
        z1b = np.where(badb[:, None], -z1, z1b)
        t0 = child_xz(z1, x0, z0)
        t1 = child_xz(z1b, x0, z0)
        for c, tc in ((c0, t0), (c1, t1)):
            if c in vmap:
                qc = p4[c]
                Ec = qc[:, 0]
                beta = np.zeros_like(qc[:, 1:])
                ok = np.abs(Ec) > 1e-12
                beta[ok] = qc[ok, 1:] / Ec[ok][:, None]
                sub = {}
                for nm in subtree[c]:
                    sub[nm] = _boost4_vec(p4[nm], beta)
                rec(c, sub, tc)

    rec(decays[0].core.name, mom, top_triad)
    return phi, theta


# ---------------------------------------------------------------------------
# Angle/|p|-based aligned frames (no 4-momentum, no duplicate kinematics)
# ---------------------------------------------------------------------------
def _frames_from_angles(chain, ph, th, q, m_node):
    """Per-particle SU(2) frame matrices built ONLY from the already computed
    per-vertex angles and per-decay two-body |p| (+ masses).

    The vertex rotation reproduces tf-pwa exactly (cal_helicity_angle):
    slot 0 ``Ry(theta) Rz(phi)`` and slot 1 (antipodal daughter)
    ``Ry(pi-theta) Rz(phi-pi)`` - confirmed to match tf-pwa's per-vertex SU2
    to machine precision on shared events (no sign canonicalization; the
    slot-1 phi in [-2pi, 0) must keep its sign).  Boost magnitudes come
    from ``q`` (the per-decay |p| already in the event data) with the child's
    own mass (leaf: rest mass; resonance: its event invariant mass), i.e.
    ``omega = asinh(q/m)`` - the same numbers the 4-momentum path derives.

    Returns ``(r_matrix, b_matrix)`` dicts particle-name -> (n_events,2,2)
    mirroring ``_chain_su2_frames``.
    """
    from ampfit.su2 import (Identity, Rotation_z, Rotation_y, Boost_z,
                            _mul)
    n = ph.shape[0]
    decays = chain.decays
    cmap = {d.core.name: d for d in decays}
    idx_of = {d.core.name: i for i, d in enumerate(decays)}
    I = Identity(n)
    r_matrix = {decays[0].core.name: I}
    b_matrix = {decays[0].core.name: I}

    # children mass per event: given arrays (leaf const, resonance event)
    def m_child(name):
        return m_node[name]

    order = [decays[0].core.name]
    # pre-order over decays (decays list is already DFS pre-order)
    for d in decays:
        phi = ph[:, idx_of[d.core.name]]
        theta = th[:, idx_of[d.core.name]]
        b_core = b_matrix[d.core.name]
        r_core = r_matrix[d.core.name]
        for s, c in enumerate(d.outs):
            nm = c.name
            if s == 0:
                # tf-pwa per-vertex order: Rotation_y(theta)*Rotation_z(phi)
                Uv = _mul(Rotation_y(theta), Rotation_z(phi))
            else:
                Uv = _mul(Rotation_y(np.pi - theta),
                          Rotation_z(phi - np.pi))
            # NOTE: no SU(2) sign canonicalization - tf-pwa stores r exactly
            # as Ry(beta)*Rz(alpha) (slot-1 phi in [-2pi,0) must keep its sign)
            r_matrix[nm] = _mul(_mul(Uv, b_core), r_core)
            mc = m_child(nm)
            q_d = q[:, idx_of[d.core.name]]
            with np.errstate(divide='ignore', invalid='ignore'):
                om = np.arcsinh(np.divide(q_d, np.maximum(mc, 1e-300)))
            bad = mc < 1e-6 * np.maximum(np.sqrt(q_d * q_d + mc * mc), 1e-300)
            B = Boost_z(om)
            if np.any(bad):
                B = np.where(bad[..., None, None], I, B)
            b_matrix[nm] = B
    return r_matrix, b_matrix


def aligned_euler_from_chain(chain, ph, th, q, m_node, cm_p4,
                             spinful_names, final_rest=False):
    """Aligned euler (alpha,beta,gamma) of *spinful_names* for ONE chain
    from precomputed per-vertex angles / |p| only (center-of-mass ref).

    ``cm_p4``: dict spinful-name -> (n_events,4) momentum of that final in
    the CM (only its direction/boost for the reference is used).  Equivalent
    to ``aligned_euler_from_momenta([chain], ...)["nm"][:, 0, :]``.
    """
    from ampfit.su2 import _mul, inv as su2_inv, get_euler_angle
    r_matrix, b_matrix = _frames_from_angles(chain, ph, th, q, m_node)
    ref = _cm_reference_frames(cm_p4, spinful_names)
    out = {}
    for nm in spinful_names:
        rr = ref[nm]
        R = _mul(rr, su2_inv(r_matrix[nm]))
        if final_rest:
            R = _mul(R, su2_inv(b_matrix[nm]))      # cm reference b = identity
        a, b, g = get_euler_angle(R)
        out[nm] = np.stack([a, b, g], axis=-1)
    return out
