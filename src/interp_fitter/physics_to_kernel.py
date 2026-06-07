"""Convert a PhysicsModel into a Kernel config dict.

Each DecayChain is its own topology with independent data blocks for
mass, q, and angle.  Within a topology, all waves share the same set
of angular basis rows (``ang_order``, ``matrix_ang``).  Different
topologies do *not* share basis rows because they reference different
data columns.

Wave ordering: all LS combos of chain 0, then chain 1, …
First ``nwaves // 2`` waves are CP0 (B), the rest CP1 (B̄).
"""

from __future__ import annotations

import numpy as np
from .config_builder import PhysicsModel, DecayChain, Particle
from .angular_formula import compute_amplitude, AmpTerm, Factor
from .models import get_model, phase_space


# ---------------------------------------------------------------------------
#  Blatt-Weisskopf form factors   F_L(q) = q^L / sqrt(B'_L(z))
#  with  z = (q * radius)^2   and  radius = 3 GeV^{-1} by default.
# ---------------------------------------------------------------------------

def _bw_poly(L: int, z):
    """Blatt-Weisskopf polynomial B'_L(z) for orbital angular momentum L."""
    if L == 0:
        return 1.0
    if L == 1:
        return 1.0 + z
    if L == 2:
        return 9.0 + 3.0 * z + z ** 2
    if L == 3:
        return 225.0 + 45.0 * z + 6.0 * z ** 2 + z ** 3
    if L == 4:
        return 11025.0 + 1575.0 * z + 135.0 * z ** 2 + 10.0 * z ** 3 + z ** 4
    raise ValueError(f"Blatt-Weisskopf polynomial not implemented for L={L}")


def bw_form_factor(q, L: int, radius: float = 3.0):
    """F_L(q) = q^L / sqrt(B'_L((q·radius)²))."""
    z = (q * radius) ** 2
    return q ** L / np.sqrt(_bw_poly(L, z))


def build_fl_table(
    L_values: set[int],
    n_int: int = 200,
    q_min: float = 0.0,
    q_max: float = 2.0,
    radius: float = 3.0,
) -> np.ndarray:
    """Build the (n_type, n_int) form-factor interpolation table.

    Each row corresponds to a unique orbital angular momentum *L*.
    """
    table = np.zeros((len(L_values), n_int), dtype=np.float64)
    sorted_L = sorted(L_values)
    for row, L in enumerate(sorted_L):
        q = np.linspace(q_min, q_max, n_int, endpoint=False)
        table[row] = bw_form_factor(q, L, radius)
    return table


def _resonance_key(part: Particle | None) -> tuple:
    if part is None:
        return ()
    p = part.props
    channels = tuple(tuple(sorted(c.items())) for c in p.get("channels", []))
    return (p.get("J"), p.get("P"), p.get("mass"), p.get("width"),
            p.get("model"), channels)


def _particle(physics, name):
    return physics.particles.get(name)


def _amp_terms(chain, ls) -> list[tuple[str, str, AmpTerm]]:
    """Yield all (hk, lsk, term) from compute_amplitude for one LS."""
    amp = compute_amplitude(chain, list(ls))
    for hk, ls_dict in amp.items():
        for lsk, terms in ls_dict.items():
            for term in terms:
                yield (hk, lsk, term)


def physics_model_to_config(
    physics: PhysicsModel,
    n_int_gamma: int = 200,
    n_int_fl: int = 200,
    with_cp: bool = False,
) -> dict:
    """Convert a PhysicsModel to a Kernel config dict.

    Parameters
    ----------
    with_cp : bool
        If True, each chain is duplicated: first copy → CP0, second copy → CP1
        with separate (shifted) data columns. Total chains double.
    """
    # Duplicate chains for CP groups (each gets its own data columns)
    base_chains = physics.decay_chains
    if with_cp:
        chains = list(base_chains) + list(base_chains)
    else:
        chains = list(base_chains)
    n_topo = len(chains)

    # ------------------------------------------------------------------
    # Phase 1 — Per-chain metadata: LS combos, angle formula
    # ------------------------------------------------------------------
    chain_ls: list[list[tuple]] = []
    chain_terms: list[list[tuple]] = []  # list of (hk, term) per chain
    all_hel_keys: set[str] = set()

    for dc in chains:
        lss = dc.ls_combinations()
        chain_ls.append(lss)
        terms = []
        for ls in lss:
            for hk, lsk, term in _amp_terms(dc, list(ls)):
                all_hel_keys.add(hk)
                terms.append((hk, term))
        chain_terms.append(terms)

    hel_keys_sorted = sorted(all_hel_keys)
    nhel = len(hel_keys_sorted)

    n_waves_per_chain = [len(lss) for lss in chain_ls]
    nwaves = sum(n_waves_per_chain)
    wave_offsets = []
    wo = 0
    for nw in n_waves_per_chain:
        wave_offsets.append(wo)
        wo += nw

    # ------------------------------------------------------------------
    # Phase 2 — Per-topology factor registry
    # ------------------------------------------------------------------
    topo_factors: list[list[tuple]] = []     # [(name, func, k), ...]
    topo_factor_idx: list[dict] = []         # {(n,f,k): local_idx}
    topo_n_ang: list[int] = []

    for ci in range(n_topo):
        reg: dict = {}
        flist: list = []
        for hk, term in chain_terms[ci]:
            for f in term.factors:
                key = (f.name, f.func, f.k)
                if key not in reg:
                    reg[key] = len(flist)
                    flist.append(key)
        topo_factor_idx.append(reg)
        topo_factors.append(flist)
        topo_n_ang.append(len(flist))

    # ------------------------------------------------------------------
    # Phase 3 — Global angle arrays
    # ------------------------------------------------------------------
    # angle_index, angle_k, angle_b concatenate all topologies' factors
    topo_angle_offsets = []
    ao = 0
    for ci in range(n_topo):
        topo_angle_offsets.append(ao)
        ao += topo_n_ang[ci]
    ndim_angle = ao

    angle_index_list: list[int] = []
    angle_k_list: list[float] = []
    angle_b_list: list[float] = []

    for ci in range(n_topo):
        offset = topo_angle_offsets[ci]
        for (name, func, k) in topo_factors[ci]:
            angle_index_list.append(offset + topo_factor_idx[ci][(name, func, k)])
            angle_k_list.append(k / 2.0)
            angle_b_list.append(-np.pi / 2 if func == "sin" else 0.0)

    # ------------------------------------------------------------------
    # Phase 4 — Per-topology basis: unique factor_tuples → basis rows
    # ------------------------------------------------------------------
    # Each topology has its own basis rows. Within a topology, identical
    # factor_tuples from different LS/helicities are deduplicated.
    topo_basis_sets: list[dict] = []  # [{factor_tup: local_basis_idx}]
    topo_basis_list: list[list] = []  # [[factor_tup, ...]]
    topo_nbasis: list[int] = []

    for ci in range(n_topo):
        bset: dict = {}
        blist: list = []
        for hk, term in chain_terms[ci]:
            ftup = tuple(sorted((f.name, f.func, f.k) for f in term.factors))
            if ftup not in bset:
                bset[ftup] = len(blist)
                blist.append(ftup)
        topo_basis_sets.append(bset)
        topo_basis_list.append(blist)
        topo_nbasis.append(len(blist))

    nbasis_total = sum(topo_nbasis)

    # ------------------------------------------------------------------
    # Phase 5 — ang_order
    # ------------------------------------------------------------------
    # One row per global basis entry. Each row lists the global angle
    # indices (0..ndim_angle-1) for the factors in that basis term.
    ang_order_list: list[list[int]] = []
    for ci in range(n_topo):
        offset = topo_angle_offsets[ci]
        reg = topo_factor_idx[ci]
        for ftup in topo_basis_list[ci]:
            row = [offset + reg[(n, f, k)] for (n, f, k) in ftup]
            ang_order_list.append(row)

    n_per = max(len(r) for r in ang_order_list) if ang_order_list else 1
    ang_order_arr = np.zeros((nbasis_total, n_per), dtype=np.int32)
    for i, row in enumerate(ang_order_list):
        for j, idx in enumerate(row):
            ang_order_arr[i, j] = idx

    # ------------------------------------------------------------------
    # Phase 6 — matrix_ang
    # ------------------------------------------------------------------
    max_nhel = nhel
    matrix_ang = np.zeros((nbasis_total, nwaves * max_nhel), dtype=complex)

    global_basis_row = 0
    for ci in range(n_topo):
        n_topo_basis = topo_nbasis[ci]
        bset = topo_basis_sets[ci]
        wo = wave_offsets[ci]

        for li, ls in enumerate(chain_ls[ci]):
            for hk, lsk, term in _amp_terms(chains[ci], list(ls)):
                ftup = tuple(sorted((f.name, f.func, f.k) for f in term.factors))
                bi = global_basis_row + bset[ftup]
                hel_idx = hel_keys_sorted.index(hk)
                gc = (wo + li) * max_nhel + hel_idx
                matrix_ang[bi, gc] += term.coeff

        global_basis_row += n_topo_basis

    # ------------------------------------------------------------------
    # Phase 7 — Topology data column offsets (mass, q)
    # ------------------------------------------------------------------
    topo_decays_2b: list[list] = []
    topo_nres: list[int] = []
    topo_ndec: list[int] = []
    topo_mass_offset: list[int] = []
    topo_q_offset: list[int] = []

    mass_offset = 0
    q_offset = 0
    for ci, dc in enumerate(chains):
        d2b = [d for d in dc.decays if len(d.children) == 2]
        N = len(d2b)
        nres = N - 1 if N > 0 else 0
        ndec = N
        topo_decays_2b.append(d2b)
        topo_nres.append(nres)
        topo_ndec.append(ndec)
        topo_mass_offset.append(mass_offset)
        topo_q_offset.append(q_offset)
        mass_offset += nres
        q_offset += ndec

    ndim_mass = mass_offset
    ndim_q = q_offset

    # ------------------------------------------------------------------
    # Phase 8 — BW arrays
    # ------------------------------------------------------------------
    n_bw = ndim_mass
    bw_index = np.zeros(n_bw, dtype=np.int32)
    m0_index = np.zeros(n_bw, dtype=np.int32)

    res_key_to_m0: dict = {}
    bwi = 0
    for ci in range(n_topo):
        mo = topo_mass_offset[ci]
        d2b = topo_decays_2b[ci]
        for pos in range(topo_nres[ci]):
            bw_index[bwi] = mo + pos
            rname = d2b[pos + 1].parent
            part = _particle(physics, rname)
            rkey = _resonance_key(part)
            if rkey not in res_key_to_m0:
                res_key_to_m0[rkey] = len(res_key_to_m0)
            m0_index[bwi] = res_key_to_m0[rkey]
            bwi += 1

    n_m0 = len(res_key_to_m0)
    nres_max = max(topo_nres) if topo_nres else 0
    bw_order = np.zeros(nwaves * nres_max, dtype=np.int64)
    wi = 0
    for ci in range(n_topo):
        nwt = n_waves_per_chain[ci]
        mo = topo_mass_offset[ci]
        nr = topo_nres[ci]
        for _ in range(nwt):
            for pos in range(nr):
                bw_order[wi * nres_max + pos] = mo + pos
            wi += 1

    # ------------------------------------------------------------------
    # Phase 9 — Gamma arrays
    # ------------------------------------------------------------------
    # Collect unique resonance types, their models, and child masses
    rkey_info: list[tuple] = []          # (rkey, part, nchan, child_masses)
    rkey_seen: set = set()
    for ci in range(n_topo):
        d2b = topo_decays_2b[ci]
        for pos in range(topo_nres[ci]):
            rname = d2b[pos + 1].parent
            part = _particle(physics, rname)
            if part is None or _resonance_key(part) in rkey_seen:
                continue
            rkey = _resonance_key(part)
            rkey_seen.add(rkey)
            model_name = part.props.get("model", "BW")
            model_cls = get_model(model_name)
            if model_cls is None:
                raise ValueError(f"Unknown model '{model_name}' for particle '{rname}'")
            nchan = model_cls.n_channels(part)
            child_parts = d2b[pos + 1].child_particles or []
            child_masses = [cp.props.get("mass", 0.0) for cp in child_parts if cp is not None]
            rkey_info.append((rkey, part, nchan, child_masses))

    # Determine mass grid
    all_mass_vals = [
        p.props["mass"] for p in physics.particles.values() if "mass" in p.props
    ]
    if all_mass_vals:
        mass_min = min(all_mass_vals) * 0.5
        mass_max = max(all_mass_vals) * 1.5
    else:
        mass_min, mass_max = 0.0, 2.0
    gamma_min = float(mass_min)
    gamma_delta = float((mass_max - mass_min) / n_int_gamma)

    # Build gamma_table rows and gamma_type mapping
    # gamma_table = (n_gamma_table_rows, n_int) — each row is a phase-space curve
    # For each model, one row per channel
    gamma_type_rows: list[int] = []  # per-channel: global row index in gamma_table
    gamma_table_rows: list[np.ndarray] = []
    rkey_to_kt_row: dict = {}  # (rkey, ch) → global gamma_type index

    for rkey, part, nchan, child_masses in rkey_info:
        model_name = part.props.get("model", "BW")
        model_cls = get_model(model_name)
        if model_cls is None:
            rows = np.ones((nchan, n_int_gamma), dtype=np.complex64)
        else:
            rows = model_cls.gamma_table(part, child_masses,
                                         n_int_gamma, gamma_min, mass_max)
        for ch in range(nchan):
            idx = len(gamma_type_rows)
            gamma_type_rows.append(idx)
            rkey_to_kt_row[(rkey, ch)] = idx
        gamma_table_rows.append(rows)

    gamma_table = np.concatenate(gamma_table_rows, axis=0)
    n_gamma_table_rows = len(gamma_type_rows)

    gamma_type_list: list[int] = []
    gamma_index_list: list[int] = []
    g0_index_list: list[int] = []
    g0_counter = 0
    rkch_to_g0: dict = {}

    for ci in range(n_topo):
        d2b = topo_decays_2b[ci]
        mo = topo_mass_offset[ci]
        for pos in range(topo_nres[ci]):
            rname = d2b[pos + 1].parent
            part = _particle(physics, rname)
            rkey = _resonance_key(part)
            nchan = 0
            for rk, _, nc, _ in rkey_info:
                if rk == rkey:
                    nchan = nc
                    break
            if nchan == 0:
                nchan = 1
            for ch in range(nchan):
                gamma_type_list.append(rkey_to_kt_row.get((rkey, ch), 0))
                gamma_index_list.append(mo + pos)
                chkey = (rkey, ch)
                if chkey not in rkch_to_g0:
                    rkch_to_g0[chkey] = g0_counter
                    g0_counter += 1
                g0_index_list.append(rkch_to_g0[chkey])

    n_g0 = g0_counter
    n_gamma = len(gamma_type_list)

    mat_gamma = np.zeros((n_bw, n_gamma), dtype=float)
    gi = 0
    bwi = 0
    for ci in range(n_topo):
        d2b = topo_decays_2b[ci]
        for pos in range(topo_nres[ci]):
            rname = d2b[pos + 1].parent
            part = _particle(physics, rname)
            rkey = _resonance_key(part)
            nchan = 0
            for rk, _, nc, _ in rkey_info:
                if rk == rkey:
                    nchan = nc
                    break
            if nchan == 0:
                nchan = 1
            br = part.props.get("branching", [1.0] * nchan) if part else [1.0] * nchan
            for ch in range(nchan):
                mat_gamma[bwi, gi + ch] = br[ch] if ch < len(br) else 1.0
            gi += nchan
            bwi += 1

    # ------------------------------------------------------------------
    # Phase 10 — Form factor arrays
    # ------------------------------------------------------------------
    ndec_max = max(topo_ndec) if topo_ndec else 0
    q_index = np.zeros(ndim_q, dtype=np.int32)
    fl_order = np.zeros(nwaves * ndec_max, dtype=np.int32)

    # Determine L for each decay position from first LS combination
    all_L: set[int] = set()
    fl_L = np.zeros(ndim_q, dtype=np.int32)  # L per q_index entry
    for ci in range(n_topo):
        qo = topo_q_offset[ci]
        nd = topo_ndec[ci]
        d2b = topo_decays_2b[ci]
        if chain_ls[ci]:
            first_ls = chain_ls[ci][0]
            for pos in range(nd):
                L = first_ls[pos][0]
                all_L.add(L)
                fl_L[qo + pos] = L

    # Build fl_table from unique L values
    fl_table = build_fl_table(all_L, n_int=n_int_fl, q_min=0.0, q_max=2.0, radius=3.0)
    n_fl_type = len(all_L)
    l_to_type = {L: i for i, L in enumerate(sorted(all_L))}
    fl_type = np.zeros(ndim_q, dtype=np.int32)

    wi = 0
    for ci in range(n_topo):
        nwt = n_waves_per_chain[ci]
        qo = topo_q_offset[ci]
        nd = topo_ndec[ci]
        for pos in range(nd):
            q_index[qo + pos] = qo + pos
            fl_type[qo + pos] = l_to_type[fl_L[qo + pos]]
        for _ in range(nwt):
            for pos in range(nd):
                fl_order[wi * ndec_max + pos] = qo + pos
            wi += 1

    fl_min = 0.0
    fl_delta = float(2.0 / n_int_fl)

    # ------------------------------------------------------------------
    # Phase 11 — Final config
    # ------------------------------------------------------------------
    config = {
        "gamma_table": gamma_table.astype(np.complex64),
        "fl_table": fl_table.astype(np.float32),
        "matrix_gamma": mat_gamma.astype(np.float32),
        "matrix_ang": matrix_ang.astype(np.complex64),

        "gamma_type": np.array(gamma_type_list, dtype=np.int32),
        "gamma_index": np.array(gamma_index_list, dtype=np.int32),
        "gamma_min": gamma_min,
        "gamma_delta": gamma_delta,

        "g0_index": np.array(g0_index_list, dtype=np.int32),

        "m0_index": m0_index.astype(np.int32),
        "bw_index": bw_index.astype(np.int32),
        "bw_order": bw_order.astype(np.int64),

        "q_index": q_index.astype(np.int32),
        "fl_type": fl_type.astype(np.int32),
        "fl_min": fl_min,
        "fl_delta": fl_delta,
        "fl_order": fl_order.astype(np.int32),

        "angle_index": np.array(angle_index_list, dtype=np.int32),
        "angle_k": np.array(angle_k_list, dtype=np.float32),
        "angle_b": np.array(angle_b_list, dtype=np.float32),
        "ang_order": ang_order_arr,
        "nhelicities": max_nhel,
    }

    return config
