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

from ampfit.helicity_angle import decay_chain_leaves


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
        cfg: ampfit Config (particle masses / decay structure).
        chain: partial-wave chain of the chosen topology.
        mom_cm: (n, n_finals, 4) final momenta in ``cfg.finals`` order and
            in the top (CM) rest frame.
    Returns:
        (M, phi, theta): each (n, n_decay).  M[:,0] is the top mass.
    """
    from ampfit.momenta_to_angles import decay_angles_vectorized

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
    nv = len(decays)
    core_idx = {d[0]: i for i, d in enumerate(decays)}

    def leaf_mass(name):
        m = np.full(n, rest[name])
        return m

    # per-vertex two-body child masses used at each vertex (n arrays)
    child_m = {}
    for i, (core, outs) in enumerate(decays):
        cm = []
        for o in outs:
            if o in inner:
                cm.append(M[:, core_idx[o]])
            else:
                cm.append(leaf_mass(o))
        child_m[i] = cm

    # momenta in the CM/top rest frame
    p4 = {decays[0][0]: np.column_stack([M[:, 0], np.zeros((n, 3))])}
    triad = {decays[0][0]: np.tile([[[1., 0., 0.], [0., 0., 1.]]], (n, 1, 1))}

    for i, (core, outs) in enumerate(decays):
        pname = core
        Mp = M[:, i]
        ph, th = phi[:, i], theta[:, i]
        x0, z0 = triad[pname][:, 0], triad[pname][:, 1]
        y0 = np.cross(z0, x0)
        c0, c1 = outs[0], outs[1]
        m0, m1 = child_m[i]
        p = _two_body_p(Mp, m0, m1)
        u = (np.sin(th)[:, None] * (np.cos(ph)[:, None] * x0
                                    + np.sin(ph)[:, None] * y0)
             + np.cos(th)[:, None] * z0)
        E0 = np.sqrt(p * p + m0 * m0)
        E1 = np.sqrt(p * p + m1 * m1)
        rest0 = np.column_stack([E0, p[:, None] * u])
        rest1 = np.column_stack([E1, -p[:, None] * u])
        # parent momentum in the frame everything is expressed in
        pp = p4[pname]
        beta = pp[:, 1:] / np.maximum(pp[:, 0:1], 1e-12)
        p4[c0] = _boost_rest_to_lab(rest0, beta)
        p4[c1] = _boost_rest_to_lab(rest1, beta)
        triad[c0] = _child_xz(u, x0, z0)
        triad[c1] = _child_xz(-u, x0, z0)

    out = {o: p4[o] for i, (_c, outs) in enumerate(decays) for o in outs
           if o not in inner}
    return np.stack([out[o] for o in meta["finals"]], axis=1)
