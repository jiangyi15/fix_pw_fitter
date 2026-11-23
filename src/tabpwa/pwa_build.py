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

Phase space lives here too: :func:`generate_pwa_phsp` is the flat sampler for a
decay chain, any topology (see its docstring — it dispatches between the
three-body closed form and the general two-body fold).

The BW / running-width / form-factor tables replicate
``Config.build_single_index`` for a single topology block.
"""

import functools

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
    from tabpwa.decay_tree import symmetry_factors
    data = (getattr(cfg, "dic", None) or {}).get("data") or {}
    n_id, n_cp, _, id_groups, cp_groups = symmetry_factors(data)
    return n_id, n_cp, (id_groups, cp_groups)


def _two_body_p(M, m1, m2):
    """Two-body breakup momentum (scalar/array), 0 below threshold.

    A MASSLESS parent has no phase space, and ``0/0`` here would poison every
    weight that touches it (the three-body grid starts at ``s_min``, which is 0
    when both sub-daughters are massless — one ``nan`` there made ``wgrid.max()``
    ``nan`` and the sampler refuse a perfectly physical config).  ``M == 0``
    therefore returns 0 instead of ``nan``; every other caller is unchanged.
    """
    M = np.asarray(M, float)
    num = np.sqrt(np.clip(
        (M * M - (m1 + m2) ** 2) * (M * M - (m1 - m2) ** 2), 0.0, None))
    den = 2.0 * M
    out = np.zeros(np.broadcast(num, den).shape)
    ok = den > 0
    np.divide(num, den, out=out, where=ok)
    return out if out.shape else float(out)


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


def _generate_three_body_phsp(cfg, chain, n_events, seed=None,
                             weights_out=False):
    """The closed-form sampler for ``top -> (sub -> a b) + c``.

    Kept as its own function so the shape every existing fixture uses is
    bit-for-bit what it always was; ``generate_pwa_phsp`` dispatches here for
    that topology and to the general fold otherwise.

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

    M_top = _require_mass(cfg, chain.decays[0].core.name)
    m_a = _require_mass(cfg, names[0])
    m_b = _require_mass(cfg, names[1])
    m_c = _require_mass(cfg, bachelor)

    s_min = m_a + m_b
    s_max = M_top - m_c
    if s_max <= s_min:
        raise ValueError(
            f"phase space: top mass {M_top:.6g} is at or below the sum of the "
            f"final masses {s_min + m_c:.6g}; there is no phase space to "
            f"sample (this generator would loop forever)")
    # ── sample s with weight p_top(s)*p_sub(s) (flat 3-body) by rejection ──
    grid = np.linspace(s_min, s_max, 600)
    wgrid = _two_body_p(np.full_like(grid, M_top), grid, m_c) \
        * _two_body_p(grid, m_a, m_b)
    if not float(wgrid.max()) > 0:
        raise ValueError(
            f"phase space: the three-body weight is zero on the whole mass "
            f"window [{s_min:.6g}, {s_max:.6g}] — the top is at threshold")
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


def _require_mass(model, name):
    """The declared mass of ``name`` — with a message that says what is wrong.

    A final MUST have one (it sets the on-shell condition) and so must the
    TOP (it sets the total energy).  A node between them is never asked for a
    mass here — the chain is flattened to its finals.
    """
    dic = getattr(model, "dic", None) or {}
    entry = (dic.get("particle") or {}).get(name) or {}
    if "mass" not in entry:
        raise KeyError(
            f"phase space: particle {name!r} has no 'mass' in the config's "
            f"particle table (a final needs one for its on-shell condition, "
            f"the top for the total energy)")
    return float(entry["mass"])


def _check_distinct_finals(chain):
    """Refuse a chain whose final names repeat.

    The momenta are returned as ``{name: 4-vector}`` and the event stack is
    built by name, so ``top -> pip + pip`` would keep ONE of the two pions and
    silently produce an event that violates energy-momentum conservation.  The
    framework's own event builder keys on names too, so this is not a
    limitation of this function.
    """
    finals = [o.name for d in chain.decays for o in d.outs
              if o.name not in {x.core.name for x in chain.decays}]
    dup = sorted({n for n in finals if finals.count(n) > 1})
    if dup:
        raise ValueError(
            f"phase space: the chain has repeated final name(s) {dup}; the "
            f"event arrays are keyed by name, so two daughters with one name "
            f"cannot both be stored (declare distinct names or drop the "
            f"identical-particle declaration)")


def _chain_is_three_body(chain):
    """``top -> (sub -> a b) + c`` and nothing deeper — the closed-form shape."""
    if len(chain.decays) != 2:
        return False
    top = chain.decays[0]
    sub = chain.decays[1]
    if len(top.outs) != 2 or len(sub.outs) != 2:
        return False
    # exactly one top daughter is itself the sub-decay
    names = {o.name for o in top.outs}
    return sub.core.name in names


def generate_pwa_phsp(cfg, chain, n_events, seed=None, weights_out=False,
                      batch=200_000, top_p=None):
    """Flat phase space of ``chain`` — any topology.

    Dispatches on the chain's SHAPE:

    * ``top -> (sub -> a b) + c`` takes :func:`_generate_three_body_phsp`, the
      closed-form sampler (the shape every fixture uses);
    * anything else — a ternary/n-ary vertex, a 2+2 cascade, a deeper tree —
      takes the general two-body fold, which is the topic of this module's
      header.

    ``cfg`` is an ``AmplitudeModel``/``Config`` — it must expose ``dic`` (the
    particle table) and ``finals``; a plain dict is NOT accepted.  Returns
    ``(n_events, n_finals, 4)`` 4-momenta ``(E, px, py, pz)`` in ``cfg.finals``
    order, in the top rest frame — or in the frame the top's momentum names
    when ``top_p`` is given (a ``(n, rng) -> (n, 3)`` callable in GeV/c, e.g.
    :func:`tabpwa.phasespace.fix_beam`; optionally ``(momenta, weights)``).

    The particle MODELS are never consulted — only topology and final masses —
    so a node with ``model: skip`` needs no case: its invariant mass is a
    sampled variable like any other internal node's.
    """
    _check_distinct_finals(chain)
    if n_events < 0:
        raise ValueError(f"phase space: n_events must be >= 0, got {n_events}")
    if n_events == 0:
        # both samplers would otherwise mishandle it (the closed form returns
        # an empty stack, the general path raises a bare "need at least one
        # array to concatenate")
        empty = np.empty((0, len(getattr(cfg, "finals", []) or []), 4))
        return (empty, np.ones(0)) if weights_out else empty
    if _chain_is_three_body(chain):
        if top_p is not None:
            # the closed form is a frozen fast path; do not silently ignore a
            # requested momentum — route it through the general sampler instead
            return _generate_general_phsp(cfg, chain, n_events, seed=seed,
                                          batch=batch, weights_out=weights_out,
                                          top_p=top_p)
        return _generate_three_body_phsp(cfg, chain, n_events, seed=seed,
                                         weights_out=weights_out)
    return _generate_general_phsp(cfg, chain, n_events, seed=seed,
                                  batch=batch, weights_out=weights_out,
                                  top_p=top_p)


@functools.lru_cache(maxsize=16)
def _compiled_phsp(m0, masses):
    """The compiled topology for ``(m0, daughter masses)`` — parse + ceilings once.

    ``PhaseSpace`` is immutable after ``__init__`` and ``top_p`` is per call, so
    the key is just the masses; a repeated dataset pays nothing.  MEASURED the
    fixed cost is 0.2 / 5.7 / 9.6 / 13.7 ms for 3 / 5 / 7 / 9 daughters, up to
    ~56 % of a 400-event call at n >= 5.
    """
    from tabpwa.phasespace import PhaseSpace
    return PhaseSpace(m0, masses)


def _chain_leaves(chain):
    """The chain's finals in depth-first order."""
    dmap = {d.core.name: [o.name for o in d.outs] for d in chain.decays}
    leaves = []

    def collect(name):
        if name not in dmap:
            leaves.append(name)
            return
        for o in dmap[name]:
            collect(o)

    collect(chain.decays[0].core.name)
    return leaves


def _generate_general_phsp(model, chain, n_events, seed=None, batch=200_000,
                          weights_out=False, top_p=None):
    """The general path: the chain's finals as a flat mass tree, then sample.

    The sampler lives in :mod:`tabpwa.phasespace` and takes only masses, so a
    ``model: skip`` node needs no case here and the decay grouping is
    irrelevant to the measure — a flat list of the final masses is the whole
    input.  This function only maps the chain onto that list and the result
    back onto ``cfg.finals`` order.
    """
    leaves = _chain_leaves(chain)
    m_top = _require_mass(model, chain.decays[0].core.name)
    masses = [_require_mass(model, name) for name in leaves]
    # ``weights_out=True`` would come back as a TUPLE and the zip below would
    # pair the finals with (momenta, ones) — so never ask the sampler for
    # weights here.  The sample is unweighted; this adapter adds its own.
    mom_list = _compiled_phsp(m_top, tuple(masses)).generate(
        n_events, seed=seed, weights_out=False, batch=batch, top_p=top_p)

    by_name = dict(zip(leaves, mom_list))
    finals = list(getattr(model, "finals", None) or leaves)
    out = np.stack([by_name[f] for f in finals], axis=1)
    if weights_out:
        return out, np.ones(len(out))
    return out


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
    from tabpwa.helicity_angle import decay_chain_leaves
    from tabpwa.momenta_to_angles import decay_angles_vectorized

    # ``cfg`` may be a Config/AmplitudeModel (``full_decay`` alias) or a DecayTree.
    full_decay = getattr(cfg, "full_decay", None) or getattr(cfg, "full")
    if not list(full_decay.get_partial_waves()):
        raise ValueError("no partial waves in config")
    n_decay, n_topo = cfg.n_decay, cfg.n_topo
    n_res = cfg.n_res
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
        from tabpwa.helicity_angle import (decay_chain_to_tree, tree_vertices,
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
                # STRUCTURAL column: a dropped factor's column simply stops
                # being referenced by the config, it is never renumbered
                mass[:, n_res * tid + idx - 1] = inv_m[idx]
            mm = [(inv_m[idx_of[o.name]] if o.name in idx_of
                   else inv_leaf[o.name]) for o in d.outs]
            # two-body |p| with a data-only threshold clamp (mirrors the
            # legacy pwa_event_data convention of np.maximum(M, m1+m2))
            Mp = np.maximum(inv_m[idx], mm[0] + mm[1])
            q[:, n_decay * tid + idx] = _two_body_p(Mp, mm[0], mm[1])

        if need_align and chain in active:
            from tabpwa.momenta_to_angles import aligned_euler_from_chain
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
    from tabpwa.decay_tree import block_column_orders, symmetry_factors
    _, _, _, id_groups, cp_groups = symmetry_factors(data or {})
    return block_column_orders(finals, id_groups, cp_groups)


def build_tree_event_data(cfg, kc, chains_by_topo, momenta, spinful_names=(),
                          cm_boost=True, blocks=None, chunk=1_000_000):
    """Tree event data with identical-particle / CP block expansion.

    Wraps :func:`pwa_event_data_tree` per block: each declared block reorders
    the momentum columns (identical permutations x CP exchange) and is filled
    separately, then the blocks are concatenated along the topology axis,
    matching the kernel's ``n_blocks`` index shifts.  ``n_blocks == 1`` gives
    exactly the single-block result.

    The fill is purely row-wise (per-event boosts/angles), so *momenta* are
    processed in row chunks of ``chunk`` events (default 1e6) to bound peak
    memory on large files; the concatenated result is identical to the
    all-at-once fill.  ``chunk=None`` forces a single pass.

    Args:
        momenta: (n, n_finals, 4) final-state 4-momenta.
        chunk: row-chunk size for the fill (None = all rows at once).
    """
    n = momenta.shape[0]
    if chunk is not None and n > chunk:
        outs_all = []
        for lo in range(0, n, chunk):
            # materialize one chunk at a time (momenta may be a read-only
            # np.load mmap of a large file)
            outs_all.append(build_tree_event_data(
                cfg, kc, chains_by_topo,
                np.asarray(momenta[lo:lo + chunk], dtype=float),
                spinful_names=spinful_names, cm_boost=cm_boost,
                blocks=blocks, chunk=None))
        return {k: np.concatenate([o[k] for o in outs_all], axis=0)
                for k in outs_all[0]}
    momenta = np.asarray(momenta, dtype=float)
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
