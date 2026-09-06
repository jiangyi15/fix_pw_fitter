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
    data_d = cfg.dic.get("data") or {}
    id_groups = data_d.get("identical_particles") or []
    cp_groups = data_d.get("cp_particles") or []
    n_id = 1
    for grp in id_groups:
        n_id *= math.factorial(len(grp))
    n_cp = 2 if cp_groups else 1
    return n_id, n_cp, (id_groups, cp_groups)


def build_pwa_kernel_config(cfg):
    """Kernel-config dict for a pure-PWA Config (single flavour block).

    Returns the same keys as ``Config.build_all_index()`` plus ``n_proj``
    and ``wave_names``.  Wave ordering = ``full_decay.get_partial_waves()``
    (chain-major).  One ck per partial wave (length N), shared across the
    n_proj spin projections.
    """
    n_id, n_cp, (id_groups, cp_groups) = pwa_duplication_factors(cfg)
    if n_id != 1 or n_cp != 1:
        raise NotImplementedError(
            "pwa_build: identical-particle/cp row-block expansion not yet "
            f"implemented (n_id={n_id}, n_cp={n_cp}); use a single-flavour "
            "config without identical/CP partners for now")

    from ampfit.helicity_angle import (
        decay_chain_to_tree, tree_vertices, tree_leaves,
        amplitude_monomials, _reduce_layout, canonical_variables, to_spin)

    waves_iter = cfg.full_decay.get_partial_waves()       # [(ls, chain), …]
    if not waves_iter:
        raise ValueError("no partial waves in config")
    N = len(waves_iter)
    n_decay = len(waves_iter[0][1].decays)
    n_res = n_decay - 1
    # single topology block (assumed identical for all active chains)
    topo_id_map = cfg.topo_index
    topo = 0
    if len(topo_id_map) > 1:
        # all wave-active chains must share one topology for this builder
        topos = {topo_id_map[ch.topo_id()] for _, ch in waves_iter}
        if len(topos) != 1:
            raise NotImplementedError(
                "pwa_build: multiple topologies not yet supported")
        topo = next(iter(topos))

    # external spin projections (helicities) of the top
    top_core = waves_iter[0][1].decays[0].core
    top_spins = getattr(top_core, "spins", None)
    if top_spins is None:
        top_states = [to_spin(m)
                      for m in range(-to_spin(top_core.J), to_spin(top_core.J) + 1)]
    else:
        top_states = [to_spin(m) for m in top_spins]
    P = cfg.n_proj if getattr(cfg, "n_proj", None) else len(top_states)
    if P != len(top_states):
        raise ValueError(
            f"config n_proj={P} != number of top helicity states "
            f"{len(top_states)}")

    # ── resonance / width / form-factor unique tables (single block) ───────
    m0_phys, g0_phys, unique_l = [], [], []
    unique_bw, unique_gamma, unique_fl = [], [], []
    bw_gamma = {}
    for ls, chain in waves_iter:
        for li in ls:
            lv = int(li[0])
            if lv not in unique_l:
                unique_l.append(lv)
        for idx, decay in enumerate(chain.decays):
            if idx != 0:                       # sub-decay resonance
                m_name = decay.core.name + "_mass"
                if m_name not in m0_phys:
                    m0_phys.append(m_name)
                m_idx = n_res * topo + idx - 1
                bw_id = (m_name, m_idx)
                if bw_id not in unique_bw:
                    unique_bw.append(bw_id)
                tmp = []
                for g0 in decay.core._model.get_gamma_name():
                    if g0 not in g0_phys:
                        g0_phys.append(g0)
                    g_id = (g0, m_idx)
                    if g_id not in unique_gamma:
                        unique_gamma.append(g_id)
                    tmp.append(g_id)
                bw_gamma[bw_id] = tmp
            fl_id = (int(ls[idx][0]), n_decay * topo + idx)
            if fl_id not in unique_fl:
                unique_fl.append(fl_id)

    # matrix_gamma (gamma rows → unique bw columns)
    matrix_gamma = np.zeros((len(unique_gamma), len(unique_bw)))
    for bi, k in enumerate(unique_bw):
        for j in bw_gamma[k]:
            matrix_gamma[unique_gamma.index(j), bi] = 1.0

    # ── per-wave angular columns: (proj p, base wave k), entries p-major ──
    # canonical variable columns of each chain (full 2·nv phi-first layout)
    nv_list = []
    for _, chain in waves_iter:
        tree = decay_chain_to_tree(chain)
        nv_list.append(len(tree_vertices(tree)))
    nv = nv_list[0]
    if any(v != nv for v in nv_list):
        raise NotImplementedError(
            "pwa_build: chains with different vertex counts not supported")
    variables = canonical_variables(nv, top_j0=False)     # 2·nv columns

    key_set = {}                       # mono key → matrix row
    bw_order = []
    fl_order = []
    cols = {}                          # (p, k) -> [(row, coeff), …]
    for kk, (ls, chain) in enumerate(waves_iter):
        for idx in range(n_decay):     # bw/fl rows mirror build_single_index
            if idx != 0:
                m_name = chain.decays[idx].core.name + "_mass"
                bw_order.append(unique_bw.index((m_name,
                                                 n_res * topo + idx - 1)))
            fl_id = (int(ls[idx][0]), n_decay * topo + idx)
            fl_order.append(unique_fl.index(fl_id))
        tree = decay_chain_to_tree(chain)
        leaves = tree_leaves(tree)
        n_leaves = len(leaves)
        for p, lam_top in enumerate(top_states):
            _, mono = amplitude_monomials(
                tree, tuple(ls), lam_top, tuple(0 for _ in range(n_leaves)))
            _, mono = _reduce_layout(mono, nv, (), phi_first=True)
            row_list = []
            for key, coef in mono.items():
                if abs(coef) < 1e-12:
                    continue
                if key not in key_set:
                    key_set[key] = len(key_set)
                row_list.append((key_set[key], coef))
            cols[(p, kk)] = row_list

    basis = [None] * len(key_set)
    for key, i in key_set.items():
        basis[i] = key

    nac = len(variables)
    angle_k = np.zeros((len(basis), nac))
    angle_b = np.zeros((len(basis), nac))
    angle_index = np.zeros(len(basis), dtype=np.int32)
    for i, key in enumerate(basis):
        for j, (kind, f) in enumerate(key):
            fr = float(f)
            angle_k[i, j] = (round(fr) if abs(fr - round(fr)) < 1e-9 else fr)
            angle_b[i, j] = 0.0 if kind == 'c' else -math.pi / 2.0

    matrix_angle = np.zeros((len(basis), P * N), dtype=complex)
    for (p, kk), row_list in cols.items():
        col = p * N + kk
        for (r, c) in row_list:
            matrix_angle[r, col] += c

    # per-entry row arrays are stored p-major too: duplicate the base-wave
    # bw_order / fl_order rows P times (entry w = p*N + k owns the rows of
    # base wave k).  Index/unique arrays (m0/mass/g0/fl_q/gamma) stay shared.
    bw_order = np.concatenate([np.array(bw_order, dtype=np.int32)] * P)
    fl_order = np.concatenate([np.array(fl_order, dtype=np.int32)] * P)

    # ── mass / q / gamma / fl index arrays ────────────────────────────────
    m0_index = np.array([m0_phys.index(k[0]) for k in unique_bw], dtype=np.int32)
    mass_index = np.array([k[1] for k in unique_bw], dtype=np.int32)
    g0_index = np.array([g0_phys.index(k[0]) for k in unique_gamma],
                        dtype=np.int32)
    g0_mass_index = np.array([k[1] for k in unique_gamma], dtype=np.int32)
    fl_type = np.array([unique_l.index(k[0]) for k in unique_fl],
                       dtype=np.int32)
    fl_q_index = np.array([k[1] for k in unique_fl], dtype=np.int32)

    gamma_table_d, g_min, g_delta = cfg.build_gamma_table()
    gamma_table = np.stack([gamma_table_d[name] for name in g0_phys], axis=0)
    fl_table, fl_min, fl_delta = cfg.build_fl_table(unique_l)

    n_m0_params = len(m0_phys)
    n_g0_params = len(g0_phys)

    ret = {
        "m0_index": m0_index,
        "g0_index": g0_index,
        "g0_mass_index": g0_mass_index,
        "mass_index": mass_index,
        "fl_type": fl_type,
        "fl_q_index": fl_q_index,
        "bw_order": bw_order,
        "fl_order": fl_order,
        "angle_index": angle_index,
        "angle_k": angle_k,
        "angle_b": angle_b,
        "matrix_angle": matrix_angle,
        "gamma_table": gamma_table,
        "gamma_min": g_min,
        "gamma_delta": g_delta,
        "matrix_gamma": matrix_gamma,
        "fl_table": fl_table,
        "fl_min": fl_min,
        "fl_delta": fl_delta,
        "n_proj": P,
        # metadata for data building / later block expansion
        "n_identical": n_id,
        "n_cp": n_cp,
        "variables": variables,          # [(vertex, 'phi'|'theta'), …]
        "wave_names": [pw[1].__str__() + str(pw[0]) for pw in waves_iter],
        "top_states": top_states,
        # Fitter-facing names (single flavour block, one ck per wave)
        "m0_names": list(m0_phys),
        "g0_names": list(g0_phys),
    }
    # one ck per partial wave, ordered as in full_decay.get_partial_waves
    pw_params = list(cfg.full_decay.get_partial_waves_params())
    if len(pw_params) != N:
        raise ValueError(
            f"pwa_build: partial-wave params ({len(pw_params)}) != waves "
            f"({N})")
    ret["ck_map"] = pw_params
    return ret


def _two_body_q(M, m1, m2):
    """Two-body breakup momentum sqrt(λ)/2M (positive if kinematically open)."""
    x = (M * M - (m1 + m2) ** 2) * (M * M - (m1 - m2) ** 2)
    return np.sqrt(np.clip(x, 0.0, None)) / (2.0 * M)


def pwa_event_data(cfg, kc, momenta):
    """Per-event kernel buffers (mass/q/angle) for a pure-PWA Config.

    *momenta*: (n_events, n_finals, 4) final 4-vectors in ``cfg.finals``
    order (any frame; the geometry is boost-invariant).

    All ACTIVE topologies are filled: for every topology slot referenced by
    the kernel (``angle_index`` rows, ``mass_index``/``fl_q_index``
    columns) a representative partial-wave chain of that topology provides
    the per-vertex canonical angles and the sub-system kinematics.  This
    supports configurations where several pairings have resonances at once
    (each pairing = one topology slot).  Topology slots without any active
    chain are left zero (the kernel never reads them).

    Returns a dict with ``mass`` (n, n_mass_slots), ``q`` (n, n_fl_slots),
    ``angle`` (n, n_angle_total, n_angle_comp), ``weight``, ``bkg`` where
    the column/row layouts match the kernel index arrays.
    """
    from ampfit.helicity_angle import decay_chain_leaves
    from ampfit.momenta_to_angles import decay_angles_vectorized

    waves_iter = cfg.full_decay.get_partial_waves()
    if not waves_iter:
        raise ValueError("no partial waves in config")
    finals = list(cfg.finals)
    n = momenta.shape[0]

    # representative chain per topology slot (first active chain of each topo)
    chain_by_topo = {}
    for ls, chain in waves_iter:
        tid = cfg.topo_index.get(chain.topo_id())
        if tid is not None and tid not in chain_by_topo:
            chain_by_topo[tid] = chain
    n_rows = int(np.max(kc["angle_index"])) + 1     # rows = topology slots
    n_decay = len(waves_iter[0][1].decays)
    n_mass = int(np.max(kc["mass_index"])) + 1
    n_q = int(np.max(kc["fl_q_index"])) + 1

    mass = np.zeros((n, n_mass))
    q = np.zeros((n, n_q))
    ang = None
    vars_ = None

    for tid in range(n_rows):
        chain = chain_by_topo.get(tid)
        if chain is None:
            continue
        names = [o.name for o in decay_chain_leaves(chain)]
        order = [names.index(f) for f in finals]
        mom = np.asarray(momenta[:, order], dtype=float)

        # invariant mass of the sub-system (decays[1] children)
        sub_out = [o.name for o in chain.decays[1].outs]
        m_a = float(cfg.dic["particle"][sub_out[0]]["mass"])
        m_b = float(cfg.dic["particle"][sub_out[1]]["mass"])
        tot = mom.sum(axis=1)
        mtop = np.sqrt(np.clip(tot ** 2 @ np.array([1, -1, -1, -1.]),
                               0.0, None))
        pa = mom[:, finals.index(sub_out[0])]
        pb = mom[:, finals.index(sub_out[1])]
        m_sub = np.sqrt(np.clip(
            (pa + pb) ** 2 @ np.array([1, -1, -1, -1.]), 0.0, None))
        bachelor = [f for f in finals if f not in sub_out][0]
        m_c = float(cfg.dic["particle"][bachelor]["mass"])

        # mass / q slots of this topology (decay idx = 1 for the sub decay)
        m_idx = cfg.n_res * tid + 1 - 1            # n_res = n_decay - 1
        mass[:, m_idx] = m_sub
        q[:, cfg.n_decay * tid + 0] = _two_body_q(
            np.maximum(mtop, m_sub + m_c), m_sub, m_c)
        q[:, cfg.n_decay * tid + 1] = _two_body_q(m_sub, m_a, m_b)

        # canonical phi-first per-vertex angles of this topology's chain
        ph, th = decay_angles_vectorized(chain, mom)
        if vars_ is None:
            vars_ = kc["variables"]         # [(vertex, 'phi'|'theta'), …]
            ang = np.zeros((n, n_rows, len(vars_)))
        for j, (v, kind) in enumerate(vars_):
            ang[:, tid, j] = ph[:, v] if kind == 'phi' else th[:, v]

    if ang is None:
        raise ValueError("no angle rows filled — empty wave set")
    return {
        "mass": mass,
        "q": q,
        "angle": ang,
        "weight": np.ones(n),
        "bkg": np.ones(n),          # default background contribution = 1
    }


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
