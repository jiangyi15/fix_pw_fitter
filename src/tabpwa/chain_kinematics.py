"""Chain kinematics: canonical (vertex masses + Euler angles) of one topo
chain, and the vectorised two-body reconstruction back to final momenta.

The canonical variables of a chain are, per decay vertex in ``chain.decays``
DFS pre-order (parent before children):

* ``M[i]``   — the invariant mass of the subtree below vertex ``i``
  (``M[0]`` is the top mass = event sqrt(s); leaves are never vertices),
* ``φ[i]``, ``θ[i]`` — the two Euler angles of the vertex in the canonical
  (successive-boost helicity) frames used by ``decay_angles_from_momenta``.

Forward  : ``canonical_of_chain(cfg, chain, mom_cm) -> M, phi, theta``
           with ``mom_cm`` (n, n_finals, 4) in the top (CM) rest frame and
           leaves in the same order ``decay_chain_leaves(chain)`` returns.
           The result is column-identical to the topology row that
           ``pwa_event_data_tree`` fills for this chain.

Inverse  : ``chain_meta(cfg, chain)`` captures the pure tree structure, and
           ``reconstruct_from_canonical(meta, M, phi, theta) -> mom_cm``
           rebuilds final 4-momenta (n, n_finals, 4, ``meta["finals"]``
           order) from arbitrary (smeared) vertex masses and angles — the
           inverse of the forward map, so a smeared (M, φ, θ) turns back
           into one physical event of exactly those variables.
"""
import json

import numpy as np

from tabpwa.helicity_angle import decay_chain_leaves


# ---------------------------------------------------------------------------
# forward: momenta -> canonical variables of the chosen chain
# ---------------------------------------------------------------------------

def _inv_mass(m4):
    e2 = m4[..., 0] * m4[..., 0]
    p2 = np.sum(m4[..., 1:] * m4[..., 1:], axis=-1)
    return np.sqrt(np.clip(e2 - p2, 0.0, None))


def canonical_of_chain(cfg, chain, mom_cm):
    """Canonical per-vertex masses + Euler angles for *chain*.

    Args:
        cfg: tabpwa Config (particle masses / decay structure).
        chain: partial-wave chain of the chosen topology.
        mom_cm: (n, n_finals, 4) final momenta in ``cfg.finals`` order and
            in the top (CM) rest frame.
    Returns:
        (M, phi, theta): each (n, n_decay).  M[:,0] is the top mass.
    """
    from tabpwa.momenta_to_angles import decay_angles_vectorized

    leaves = decay_chain_leaves(chain)
    names = [o.name for o in leaves]
    order = {nm: j for j, nm in enumerate(cfg.finals)}
    mom = np.stack([mom_cm[:, order[nm]] for nm in names], axis=1)

    ph, th = decay_angles_vectorized(chain, mom)

    cmap = {d.core.name: d for d in chain.decays}
    M = np.empty((mom.shape[0], len(chain.decays)))
    for i, d in enumerate(chain.decays):
        # invariant mass of the descendant leaves of this decay core
        stack, acc = [d.core.name], []
        while stack:
            x = stack.pop()
            dd = cmap.get(x)
            if dd is None:
                acc.append(names.index(x))
            else:
                stack += [o.name for o in dd.outs]
        tot = np.zeros((mom.shape[0], 4))
        for li in acc:
            tot = tot + mom[:, li]
        M[:, i] = _inv_mass(tot)
    return M, ph, th


def chain_meta(cfg, chain):
    """Serialisable tree structure of *chain* (reconstruction recipe).

    ``decays`` is in the same DFS pre-order as ``chain.decays`` and each
    entry is ``[core_name, [out0, out1]]``; ``inner`` is the set of core
    names (variable-mass vertices); ``rest`` maps every particle name to its
    nominal rest mass.  ``finals`` is ``cfg.finals`` order for the output
    momenta.  Leaves that are identical-particle permutations keep their
    original names — the caller must use a chain whose leaf set equals the
    finals.
    """
    decays = [[d.core.name, [str(o) for o in d.outs]] for d in chain.decays]
    inner = {d.core.name for d in chain.decays}
    # rest masses are only needed for FINAL leaves (inner vertices carry their
    # own per-event/smeared invariant mass)
    rest = {nm: float(cfg.dic["particle"][nm]["mass"]) for nm in cfg.finals}
    return {"decays": decays, "inner": sorted(inner), "rest": rest,
            "finals": list(cfg.finals)}


def meta_from_json(text):
    return json.loads(text)


# ---------------------------------------------------------------------------
# helpers (batch over events)
# ---------------------------------------------------------------------------

def _boost_rest_to_lab(p4, beta):
    """Boost a set of 4-vectors by velocity *beta* (rest -> moving frame)."""
    b2 = np.sum(beta * beta, axis=-1)
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-300))
    bp = np.einsum("ni,ni->n", beta, p4[:, 1:])
    coef = np.zeros_like(b2)
    m = b2 > 1e-300
    coef[m] = (gamma[m] - 1.0) / b2[m]
    out = np.empty_like(p4)
    out[:, 0] = gamma * (p4[:, 0] + bp)
    out[:, 1:] = (p4[:, 1:] + coef[:, None] * beta * bp[:, None]
                  + gamma[:, None] * p4[:, 0:1] * beta)
    return out


def _two_body_p(M, m1, m2):
    return np.sqrt(np.clip(
        (M * M - (m1 + m2) ** 2) * (M * M - (m1 - m2) ** 2), 0.0, None)) \
        / (2.0 * M)


def _child_xz(zc, x0, z0):
    """(x, z) triad of a child moving along *zc*, image of parent's Rz(φ) x."""
    xv = zc * np.sum(z0 * zc, axis=-1, keepdims=True) - z0
    n = np.linalg.norm(xv, axis=-1)
    good = n > 1e-9
    ref = np.where(np.abs(z0[:, 0:1]) < 0.9, [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0])
    v = np.cross(z0, ref)
    nv = np.linalg.norm(v, axis=-1)
    fall = v / np.maximum(nv[:, None], 1e-12)
    xc = np.where(good[:, None], xv / np.maximum(n[:, None], 1e-12), fall)
    return np.stack([xc, zc], axis=1)     # (n, 2, 3)


def reconstruct_from_canonical(meta, M, phi, theta):
    """Inverse map: canonical variables -> final momenta in the top rest.

    Boosts are applied STEP BY STEP down/up the decay chain: every vertex
    builds its daughters in ITS OWN rest frame and boosts each inner
    daughter's subtree by that daughter's velocity in this frame (never a
    single direct boost to the CM), so the frames follow the same
    successive-boost convention as the forward map and the round trip is
    exact for chains of any depth.

    Args:
        meta: dict from :func:`chain_meta`.
        M, phi, theta: (n, n_vertices) arrays in ``meta["decays"]`` order.
    Returns:
        (n, n_finals, 4) array in ``meta["finals"]`` order.
    """
    n = M.shape[0]
    decays = meta["decays"]
    inner = set(meta["inner"])
    rest = meta["rest"]
    core_idx = {d[0]: i for i, d in enumerate(decays)}

    def leaf_mass(name):
        return np.full(n, rest[name])

    child_m = {}
    for i, (core, outs) in enumerate(decays):
        cm = []
        for o in outs:
            if o in inner:
                cm.append(M[:, core_idx[o]])
            else:
                cm.append(leaf_mass(o))
        child_m[i] = cm

    top_triad = np.tile([[[1., 0., 0.], [0., 0., 1.]]], (n, 1, 1))

    def rec(core, T):
        """Return {final-name: p4} of the subtree in *core*'s rest frame."""
        if core not in core_idx:                    # final leaf at rest
            p = np.zeros((n, 4))
            p[:, 0] = leaf_mass(core)
            return {core: p}
        i = core_idx[core]
        outs = decays[i][1]
        ph, th = phi[:, i], theta[:, i]
        x0, z0 = T[:, 0], T[:, 1]
        y0 = np.cross(z0, x0)
        m0, m1 = child_m[i]
        p = _two_body_p(M[:, i], m0, m1)
        u = (np.sin(th)[:, None] * (np.cos(ph)[:, None] * x0
                                    + np.sin(ph)[:, None] * y0)
             + np.cos(th)[:, None] * z0)
        E0 = np.sqrt(p * p + m0 * m0)
        E1 = np.sqrt(p * p + m1 * m1)
        rest0 = np.column_stack([E0, p[:, None] * u])     # child0 in this frame
        rest1 = np.column_stack([E1, -p[:, None] * u])
        T0 = _child_xz(u, x0, z0)
        T1 = _child_xz(-u, x0, z0)

        out = {}
        for child, qrest, Tc in ((outs[0], rest0, T0),
                                 (outs[1], rest1, T1)):
            beta = qrest[:, 1:] / np.maximum(qrest[:, 0:1], 1e-12)
            if child in core_idx:
                # boost the child-rest subtree ONE level into this frame
                for nm, p4 in rec(child, Tc).items():
                    out[nm] = _boost_rest_to_lab(p4, beta)
            else:
                out[child] = qrest              # final leaf already here
        return out

    leaves = rec(decays[0][0], top_triad)
    return np.stack([leaves[o] for o in meta["finals"]], axis=1)
