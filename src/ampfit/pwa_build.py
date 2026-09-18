"""
pwa_build — build pure-PWA kernel configs (single-flavour, projection-sum).

This builds kernel-config dicts for the ``cuda_v4_pwa`` / ``numpy_pwa``
family directly from a YAML Config, without the legacy 8-block
flavour/CP/identical-particle machinery:

    n_wave = n_proj · N          entries p-major, w = p·N + k
    A_p(e) = Σ_k ck_k · a_{p,k}(e)    one shared ck (length N)
    P(e)   = Σ_p |A_p(e)|²

where the projections p are the *external spin projections* of the top
particle (helicity projections) and the per-projection angular factors
``a_{p,k}`` come from the numeric helicity engine (``amplitude_monomials``)
with that projection's external helicity — so the p-major columns really
differ per projection (no placeholder duplication).

Event-row duplication factors are read from the config ``data`` section:
``identical_particles`` (identical final particles) and ``cp_particles``
(charge-conjugate partner).  Currently only the factor-1 case (no identical,
no CP partner, e.g. ``config_pwa.yml`` J/ψ → π⁺π⁻η) is expanded; the
factors are still reported for future block expansion.

Angle layout is the canonical phi-first full 2·n_vertices columns (top J≠0,
no gauge drop):  [(0,φ),(1,φ),…,(0,θ),(1,θ),…], one event row per event.

The BW / running-width / form-factor tables replicate
``Config.build_single_index`` for a single topology block.
"""

import math

import numpy as np


def pwa_duplication_factors(cfg):
    """Config-driven event-row duplication factors.

    Returns ``(n_id, n_cp, desc)`` where
      * ``n_id``: number of identical-particle permutations =
        ∏_groups len(group)!  (from ``data.identical_particles``);
      * ``n_cp``: 2 when a CP partner is declared (``data.cp_particles``),
        else 1.

    Declarations are lists of groups; each group lists the equivalent final
    particles (identical) or the CP-pair mapping (cp).
    """
    from ampfit.decay_tree import symmetry_factors
    data = (getattr(cfg, "dic", None) or {}).get("data") or {}
    n_id, n_cp, _, id_groups, cp_groups = symmetry_factors(data)
    return n_id, n_cp, (id_groups, cp_groups)


def _two_body_q(M, m1, m2):
    """Two-body breakup momentum sqrt(λ)/2M (positive if kinematically open)."""
    x = (M * M - (m1 + m2) ** 2) * (M * M - (m1 - m2) ** 2)
    return np.sqrt(np.clip(x, 0.0, None)) / (2.0 * M)


def _two_body_p(M, m1, m2):
    """Two-body breakup momentum (scalar/array), 0 below threshold."""
    return np.sqrt(np.clip(
        (M * M - (m1 + m2) ** 2) * (M * M - (m1 - m2) ** 2), 0.0, None)) \
        / (2.0 * M)


def _boost_vec(p, beta):
    """Lorentz boost (batch over rows) by velocity *beta* (rest -> moving)."""
    b2 = np.sum(beta * beta, axis=-1)
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-300))
    bp = np.einsum('ni,ni->n', beta, p[:, 1:])
    coef = np.where(b2 > 1e-300, (gamma - 1.0) / np.where(b2 > 1e-300, b2, 1.0), 0.0)
    out = np.empty_like(p)
    out[:, 0] = gamma * (p[:, 0] + bp)
    out[:, 1:] = p[:, 1:] + coef[:, None] * beta * bp[:, None] \
        + gamma[:, None] * p[:, 0:1] * beta
    return out


def generate_pwa_phsp(cfg, chain, n_events, seed=None, weights_out=False):
    """Flat 3-body phase space of the chain via the product of two-body
    decays + an inverse boost chain (same masses as the config).

    For a decay top(M) -> (sub -> a b) + c the event mass *s* of the sub
    system is sampled with density  p_top(M; s, m_c) · p_sub(s; m_a, m_b)
    (uniform 3-body phase space), every two-body decay is generated
    isotropically in its rest frame, and the daughters are boosted up the
    chain into the top rest frame — the exact inverse of
    ``decay_angles_from_momenta``.

    Returns final 4-momenta (n_events, n_finals, 4) in ``cfg.finals``
    order (optionally (momenta, weights)).
    """
    rng = np.random.RandomState(seed) if seed is not None \
        else np.random.RandomState()
    sub = chain.decays[1]                 # sub -> a + b  (two pions)
    names = [o.name for o in sub.outs]
    tops_out = [o.name for o in chain.decays[0].outs]
    bachelor = [o for o in tops_out if o not in (sub.core.name,)][0]

    M_top = float(cfg.dic["particle"][chain.decays[0].core.name]["mass"])
    m_a = float(cfg.dic["particle"][names[0]]["mass"])
    m_b = float(cfg.dic["particle"][names[1]]["mass"])
    m_c = float(cfg.dic["particle"][bachelor]["mass"])

    s_min = m_a + m_b
    s_max = M_top - m_c
    # ── sample s with weight p_top(s)*p_sub(s) (flat 3-body) by rejection ──
    grid = np.linspace(s_min, s_max, 600)
    wgrid = _two_body_p(np.full_like(grid, M_top), grid, m_c) \
        * _two_body_p(grid, m_a, m_b)
    wmax = float(wgrid.max()) * 1.0001

    s = np.empty(n_events)
    got = 0
    while got < n_events:
        cand = rng.uniform(s_min, s_max, min(8192, n_events - got))
        w = _two_body_p(np.full_like(cand, M_top), cand, np.full_like(cand, m_c)) \
            * _two_body_p(cand, m_a, m_b)
        keep = rng.uniform(0, wmax, len(cand)) < w
        k = int(keep.sum())
        s[got:got + k] = cand[keep]
        got += k

    # ── orientations (isotropic at every vertex) ──────────────────────────
    def _unit_vec(n):
        z = rng.uniform(-1, 1, n)
        phi = rng.uniform(0, 2 * np.pi, n)
        r = np.sqrt(np.maximum(1 - z * z, 0.0))
        return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)

    u_top = _unit_vec(n_events)          # sub direction in top rest
    u_sub = _unit_vec(n_events)          # a direction in sub rest

    # sub vertex in its rest frame
    p_sub = _two_body_p(s, m_a, m_b)
    Ea = np.sqrt(p_sub ** 2 + m_a ** 2)
    Eb = np.sqrt(p_sub ** 2 + m_b ** 2)
    pa3 = p_sub[:, None] * u_sub
    pb3 = -pa3
    pa_rest = np.concatenate([Ea[:, None], pa3], axis=-1)
    pb_rest = np.concatenate([Eb[:, None], pb3], axis=-1)

    # boost daughters from the sub rest frame up to the top rest frame:
    # the sub system moves with velocity beta in the top rest frame
    p_top = _two_body_p(np.full(n_events, M_top), s, np.full(n_events, m_c))
    E_sub = np.sqrt(p_top ** 2 + s ** 2)
    sub4 = np.concatenate([E_sub[:, None], (p_top[:, None] * u_top)],
                         axis=-1)
    beta = sub4[:, 1:] / sub4[:, 0:1]

    pa = _boost_vec(pa_rest, beta)
    pb = _boost_vec(pb_rest, beta)
    top4 = np.zeros(n_events)                     # top rest frame: (M, 0, 0, 0)
    top4[:] = 0.0
    top4 = np.tile([M_top, 0.0, 0.0, 0.0], (n_events, 1))
    pc = top4 - sub4                       # eta = top 4-momentum - sub system

    out = {names[0]: pa, names[1]: pb, bachelor: pc}
    mom = np.stack([out[f] for f in cfg.finals], axis=1)
    if weights_out:
        return mom, np.ones(n_events)
    return mom




def _duck_chain_from_struct(cfg, tid):
    """A pure-geometry chain-like object for a DECLARED topology slot.

    Built from the decay structure path (pairing tree), with no partial
    waves / resonance replacement - used to fill REAL mass/q/angle rows for
    pairings that declare no resonances.  The object only needs
    ``decays[*].core.name`` / ``.outs[*].name`` and a ``.decays`` list for
    ``decay_chain_leaves`` / ``decay_angles_vectorized``.
    """
    from types import SimpleNamespace
    for path in (getattr(cfg, "decay_struct", None) or cfg.struct):
        outs_all = {}
        for p, outs, _kw in path:
            outs_all[p] = list(outs)

        def _leaves(n):
            if n not in outs_all:
                return [n]
            out = []
            for o in outs_all[n]:
                out += _leaves(o)
            return out

        cores = [p for p in outs_all if p != cfg.top]
        key = tuple(sorted(tuple(sorted(_leaves(c))) for c in cores))
        if cfg.topo_index.get(key) != tid:
            continue
        decays = []
        for p, outs, _kw in path:
            core = SimpleNamespace(name=p)
            outs_p = [SimpleNamespace(name=o) for o in outs]
            decays.append(SimpleNamespace(core=core, outs=outs_p))
        return SimpleNamespace(decays=decays, top=cfg.top)
    return None


def _boost_to_cm(momenta):
    """Boost each event to its own centre-of-mass frame (per-particle boost)."""
    tot = momenta.sum(axis=1)
    beta = -(tot[:, 1:] / tot[:, 0:1])
    return np.stack([_boost_vec(momenta[:, j], beta)
                     for j in range(momenta.shape[1])], axis=1)


def pwa_event_data_tree(cfg, kc, chains_by_topo, momenta, spinful_names=(),
                        cm_boost=True):
    """Tree-shape event buffers (mass/q/angle [+alignment]) by FLAT loops.

    Only the angles need a tree walk, and ``decay_angles_vectorized``
    already does it.  Everything else is plain per-decay list arithmetic
    over ``chain.decays``:

    * ``mass`` (n, n_topo*n_res): slot ``n_res*tid + (idx-1)`` <- invariant
      mass of decay ``idx>0`` (sum of its descendant leaves),
    * ``q``    (n, n_topo*n_decay): slot ``n_decay*tid + idx`` <- two-body
      |p| of decay *idx* from its parent/child masses,
    * ``angle``(n, n_topo, n_base [+3*len(spinful_names)]): canonical
      phi-first vertex columns; plus per-row alignment euler columns when
      the spinful finals are shared by >1 active topology.

    Every DECLARED topology row is filled with real values: active rows use
    their partial-wave chain, rows whose pairing declares no resonances are
    filled from the pairing's pure-geometry tree (``_duck_chain_from_struct``,
    no partial waves needed).  *momenta*: (n_events, n_finals, 4) in
    ``cfg.finals`` order.
    """
    from ampfit.helicity_angle import decay_chain_leaves
    from ampfit.momenta_to_angles import decay_angles_vectorized

    # ``cfg`` may be a Config/AmplitudeModel (``full_decay`` alias) or a DecayTree.
    full_decay = getattr(cfg, "full_decay", None) or getattr(cfg, "full")
    if not list(full_decay.get_partial_waves()):
        raise ValueError("no partial waves in config")
    n_decay, n_res, n_topo = cfg.n_decay, cfg.n_res, cfg.n_topo
    finals = list(cfg.finals)
    n = momenta.shape[0]

    mom0 = np.asarray(momenta, dtype=float)
    if cm_boost:
        mom0 = _boost_to_cm(mom0)
    leaf_of_name = {finals[j]: mom0[:, j] for j in range(len(finals))}

    real = [chains_by_topo.get(t) for t in range(n_topo)]
    rows = [ch if ch is not None else _duck_chain_from_struct(cfg, t)
            for t, ch in enumerate(real)]
    active = [ch for ch in real if ch is not None]
    need_align = len(active) > 1 and bool(spinful_names)
    nv = len(rows[0].decays)

    mass = np.zeros((n, n_topo * n_res))
    q = np.zeros((n, n_topo * n_decay))
    if kc is not None and kc.get("variables"):
        vars_ = list(kc["variables"])
    else:
        # derive the canonical angle basis from the tree (kc-free); same
        # source as Config.build_base_kernel_config sets kc["variables"]
        from ampfit.helicity_angle import (decay_chain_to_tree, tree_vertices,
                                           canonical_variables, to_spin)
        ref = active[0] if active else rows[0]
        _nv = len(tree_vertices(decay_chain_to_tree(ref)))
        _topj0 = to_spin(ref.decays[0].core.J) == 0
        vars_ = canonical_variables(_nv, top_j0=_topj0)
    n_base = len(vars_)
    ang = np.zeros((n, n_topo, n_base + (3 * len(spinful_names)
                                         if need_align else 0)))

    cm_p4 = {nm: leaf_of_name[nm] for nm in spinful_names} if need_align \
        else None

    for tid, chain in enumerate(rows):
        if chain is None:
            continue
        # angles already tree-based; this is a flat call per topology row
        leaves = decay_chain_leaves(chain)
        names = [o.name for o in leaves]
        mom = np.stack([leaf_of_name[nm] for nm in names], axis=1)
        ph, th = decay_angles_vectorized(chain, mom)
        if vars_:
            for j, (v, kind) in enumerate(vars_):
                ang[:, tid, j] = ph[:, v] if kind == 'phi' else th[:, v]
        else:
            ang[:, tid, :nv] = ph
            ang[:, tid, nv:2 * nv] = th

        # ── flat fill over the decays list ────────────────────────────────
        # leaf positions of every decay core (single stack pass per chain)
        cmap = {d.core.name: d for d in chain.decays}
        core_leaves = []
        for d in chain.decays:
            stack, seen, acc = [d.core.name], set(), []
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                dd = cmap.get(x)
                if dd is None:
                    acc.append(names.index(x))
                else:
                    stack += [o.name for o in dd.outs]
            core_leaves.append(acc)

        # invariant masses always from the DATA: a core from its descendant
        # leaves, a leaf from its own momentum (E^2 - |p|^2) - no table mass
        def _inv(m4):
            e2 = m4[..., 0] * m4[..., 0]
            p2 = np.sum(m4[..., 1:] * m4[..., 1:], axis=-1)
            return np.sqrt(np.clip(e2 - p2, 0., None))

        inv_m = []
        for acc in core_leaves:
            cm = np.zeros((n, 4))
            for li in acc:
                cm = cm + mom[:, li]
            inv_m.append(_inv(cm))
        idx_of = {d.core.name: i for i, d in enumerate(chain.decays)}
        inv_leaf = {nm: _inv(mom[:, names.index(nm)]) for nm in names}
        for idx, d in enumerate(chain.decays):
            if idx > 0:
                mass[:, n_res * tid + idx - 1] = inv_m[idx]
            mm = [(inv_m[idx_of[o.name]] if o.name in idx_of
                   else inv_leaf[o.name]) for o in d.outs]
            # two-body |p| with a data-only threshold clamp (mirrors the
            # legacy pwa_event_data convention of np.maximum(M, m1+m2))
            Mp = np.maximum(inv_m[idx], mm[0] + mm[1])
            q[:, n_decay * tid + idx] = _two_body_p(Mp, mm[0], mm[1])

        if need_align and chain in active:
            from ampfit.momenta_to_angles import aligned_euler_from_chain
            part = getattr(cfg, "dic", None)
            part = part["particle"] if part else getattr(cfg, "particle_spec", {})
            m_node = {nm: np.full(n, float(part[nm]["mass"]))
                      for nm in names}
            for idx in range(n_decay):
                if idx > 0:
                    m_node[chain.decays[idx].core.name] = inv_m[idx]
            eul = aligned_euler_from_chain(
                chain, ph, th,
                q[:, n_decay * tid:n_decay * (tid + 1)], m_node,
                cm_p4, list(spinful_names))
            for f, nm in enumerate(spinful_names):
                ang[:, tid, n_base + 3 * f:n_base + 3 * f + 3] = eul[nm]
    return {"mass": mass, "q": q, "angle": ang,
            "weight": np.ones(n), "bkg": np.ones(n)}


# ── generic identical-particle / CP block expansion ──────────────────────
def block_orders(finals, data):
    """Block column orders from a config ``data`` section."""
    from ampfit.decay_tree import block_column_orders, symmetry_factors
    _, _, _, id_groups, cp_groups = symmetry_factors(data or {})
    return block_column_orders(finals, id_groups, cp_groups)


def build_tree_event_data(cfg, kc, chains_by_topo, momenta, spinful_names=(),
                          cm_boost=True, blocks=None):
    """Tree event data with identical-particle / CP block expansion.

    Wraps :func:`pwa_event_data_tree` per block: each declared block reorders
    the momentum columns (identical permutations x CP exchange) and is filled
    separately, then the blocks are concatenated along the topology axis,
    matching the kernel's ``n_blocks`` index shifts.  ``n_blocks == 1`` gives
    exactly the single-block result.
    """
    if blocks is None:
        tree = cfg if hasattr(cfg, "block_orders") else getattr(cfg, "decay_tree", None)
        if tree is not None:
            blocks = tree.block_orders()
        else:
            data = (getattr(cfg, "dic", None) or {}).get("data") or {}
            blocks = block_orders(list(getattr(cfg, "finals")), data)
    if len(blocks) == 1:
        return pwa_event_data_tree(cfg, kc, chains_by_topo, momenta,
                                   spinful_names=spinful_names,
                                   cm_boost=cm_boost)
    # Boost to CM ONCE; the CP map (conjugate columns + p -> -p) is applied
    # in that frame, matching the legacy order (rest frame first).
    mom0 = _boost_to_cm(np.asarray(momenta, dtype=float)) if cm_boost \
        else np.asarray(momenta, dtype=float)
    outs = []
    for order, is_cp in blocks:
        mom_b = mom0[:, list(order)]
        if is_cp:
            # CP: conjugate columns AND reverse the 3-momentum (parity).
            mom_b = np.array(mom_b, dtype=float, copy=True)
            mom_b[:, :, 1:] *= -1.0
        outs.append(pwa_event_data_tree(cfg, kc, chains_by_topo, mom_b,
                                        spinful_names=spinful_names,
                                        cm_boost=False))
    return {
        "mass": np.concatenate([d["mass"] for d in outs], axis=1),
        "q": np.concatenate([d["q"] for d in outs], axis=1),
        "angle": np.concatenate([d["angle"] for d in outs], axis=1),
        "weight": outs[0]["weight"],
        "bkg": outs[0]["bkg"],
    }
