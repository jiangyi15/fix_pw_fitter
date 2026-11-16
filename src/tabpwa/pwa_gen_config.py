"""Kernel config for the projection-sum PWA, built from the decay-group loop.

The model, per (expanded wave ``k``, projection ``p``, event ``i``):

    F_{ipk} = fa_{pk} · Π_d fl_table[k, d]( q_{i, fl_q_index[k, d]} )
                     · Π_r num_table[k, r]( m_{i, num_m_index[k, r]} )
    fa_{pk} = Σ_a angle_matrix[k, p, a]
                     · Π_j cos( angle_{i, angle_index[a], j}·angle_k[a, j] + angle_b[a, j] )
    D_{ik}  = Π_l Σ_m gamma_matrix[k, l, m] · basis_table[m]( m_{i, basis_m_index[m]} )

Layout rules, which are the point of this config:

* **``fl`` and ``num`` are fully expanded**: one table row per (wave, decay)
  and per (wave, res).  A wave that repeats another's barrier form gets its
  own copy — no ``fl_type`` indirection, no index to keep in sync.
* **the angular basis is keyed on ``(topo, k, b)``** — the *topology*, not the
  wave, matching ``BaseModel.unique_angle_basis`` — so the waves of one
  topology share basis rows and ``angle_index[a]`` is the topology's
  event-angle column base.
* **the denominator basis is one flat list** (``basis_table``) shared through
  an explicit ``gamma_matrix``.  A row is appended the first time a
  ``(kind, mass column)`` pair is needed; duplicates are appended as separate
  entries rather than repaired, so the matrix stays a clean per-(wave, res)
  block selector.  The matrix carries **structure only** (0/1): the fitted
  weights are ``params["coeff"]`` at run time, and this builder does not
  touch them.
* **no counts are stored.**  ``n_wave`` / ``n_proj`` / ``n_res`` / ``n_decay``
  / ``n_basis`` / ``n_angle_basis`` / ``n_interp`` are all ``shape[...]`` of
  the arrays below, and the kernel asserts the cross-array agreement once at
  create time.

Keys
----
``fl_table``      (n_wave, n_decay, n_interp)  float64  barrier rows
``fl_q_index``    (n_wave, n_decay)            int32    momentum column
``num_table``     (n_wave, n_res, n_interp)    complex   row 0 of ``n_interp`` ≡ 1
``num_m_index``   (n_wave, n_res)              int32    mass column
``basis_table``   (n_basis, n_interp)          complex  finished basis rows
``basis_m_index`` (n_basis,)                   int32    mass column
``gamma_matrix``  (n_wave, n_res, n_basis)     float64  block selector
``angle_k``       (n_angle_basis, n_angle)     int32    cos (0) / sin (1)
``angle_b``       (n_angle_basis, n_angle)     float64  phase offset
``angle_index``   (n_angle_basis,)             int32    event angle column
``angle_matrix``  (n_wave, n_proj, n_angle_basis) complex
``m_min``/``m_delta``                          float    mass grid (num + basis)
``q_min``/``q_delta``                          float    momentum grid (fl)
``wave_names``    list[str]                             host metadata
"""
import itertools

import numpy as np


def build_from_groups(model, n_interp=None):
    """Build the flat kernel config by looping over the decay groups.

    A *base* wave is one partial wave of the decay tree; it **expands** into
    one wave per combination of its free numerator rows, so the arrays are
    indexed by the expanded wave ``k`` (this is why ``fl``/``num`` can be
    fully expanded without ambiguity: each expanded wave has exactly one
    numerator row per factor).
    """
    from tabpwa.build_defaults import get as _bget

    if n_interp is None:
        n_interp = int(_bget("n_interp", 2000))

    waves = list(model.full_decay.get_partial_waves())
    n_base = len(waves)
    topo_id = model.topo_index
    n_decay = int(model.n_decay)
    n_res = int(model.n_res)
    states = model._helicity_external_states()
    P = len(states)

    mass_grid, m_min, m_delta = model.gamma_grid(n_interp)
    q_lo, q_hi = model.get_max_q_range()
    q_grid = np.clip(np.linspace(q_lo, q_hi, n_interp), 0, np.inf)
    q_min, q_delta = float(q_grid[0]), float(q_grid[1] - q_grid[0])

    # ── expand the wave axis: one wave per free-numerator combination ──
    # The enumeration is local by design: it is the *config's* definition of
    # the wave axis.  The old ``_pg_attach`` had its own copy to emit
    # ``bw_order``/``n_index``; that path is being removed, so there is
    # nothing left for this one to drift against.
    per_wave_factors = [model._pg_chain_factors(chain) for _ls, chain in waves]
    expansion = []                      # (base index, {factor slot: rowfun})
    for k, facs in enumerate(per_wave_factors):
        free = [f for f in facs if f["heads"] and f["rows"]]
        if not free:
            expansion.append((k, {}))
            continue
        for combo in itertools.product(*[list(f["rows"]) for f in free]):
            expansion.append((k, {f["slot"]: rf for f, rf in zip(free, combo)}))
    n_wave = len(expansion)

    # ── per-(wave, decay): momentum column and barrier row ────────────
    fl_q_index = np.zeros((n_wave, n_decay), np.int32)
    fl_table = np.zeros((n_wave, n_decay, n_interp), np.float64)
    wave_names = [""] * n_wave
    for k, (base, _combo) in enumerate(expansion):
        ls, chain = waves[base]
        topo = topo_id[chain.topo_id()]
        wave_names[k] = str(chain.topo_id())
        for d, decay in enumerate(chain.decays):
            fl_q_index[k, d] = n_decay * topo + d
            fl_table[k, d] = _barrier_row(model._barrier(decay, ls[d][0]),
                                          q_grid)

    # ── per-(wave, res): numerator row and mass column ────────────────
    num_table = np.ones((n_wave, n_res, n_interp), np.complex128)
    num_m_index = np.zeros((n_wave, n_res), np.int32)
    for k, (base, combo) in enumerate(expansion):
        ls, chain = waves[base]
        for r, fac in enumerate(per_wave_factors[base]):
            num_m_index[k, r] = _mass_column(model, chain, r)
            rowfun = combo.get(fac["slot"])
            if rowfun is not None:
                num_table[k, r] = np.asarray(rowfun(mass_grid), complex)

    # ── angular basis (shared) + the per-(wave, proj) weights ─────────
    # The basis is keyed on ``(topo, k, b)`` — the TOPOLOGY, not the wave —
    # exactly as the model's ``unique_angle_basis`` does.  Two waves of one
    # topology therefore share basis rows, and ``angle_index[b]`` (the
    # topology) is the event-angle column base their vertices start at.
    angle_terms = {}
    wave_topo = []
    for k, (base, _combo) in enumerate(expansion):
        ls, chain = waves[base]
        wave_topo.append(topo_id[chain.topo_id()])
        for p in range(P):
            angle_terms[(k, p)] = model._wave_angle_terms(chain, ls, states[p])

    basis_pos, angle_k, angle_b = {}, [], []
    for k in range(n_wave):
        topo = wave_topo[k]
        for p in range(P):
            for term in angle_terms[(k, p)]:
                key = (topo, tuple(term["k"]), tuple(term["b"]))
                if key not in basis_pos:
                    basis_pos[key] = len(angle_k)
                    angle_k.append(list(term["k"]))
                    angle_b.append([0.0 if x == "cos" else -np.pi / 2
                                    for x in term["b"]])

    angle_index = np.array([key[0] for key in basis_pos], np.int32)
    angle_matrix = np.zeros((n_wave, P, len(basis_pos)), np.complex128)
    for k in range(n_wave):
        topo = wave_topo[k]
        for p in range(P):
            for term in angle_terms[(k, p)]:
                key = (topo, tuple(term["k"]), tuple(term["b"]))
                angle_matrix[k, p, basis_pos[key]] += term["coeffs"]

    # ── denominator: one flat basis list + the per-(wave, res) selector ──
    # ``den_names`` is the flat coefficient-name list: one entry per appended
    # basis slot, in the same order as ``basis_table``.  The model's own
    # ``get_den_name()`` entries are reused verbatim (``k2_s0``, ``k2_poly``,
    # …), so a coefficient is named by the basis row it weights.
    #
    # Every factor APPENDS its rows unconditionally — no sharing is resolved
    # here, so ``n_basis`` is ``Σ_factors width`` and identical entries appear
    # as separate rows.  ``gamma_matrix`` is then a clean 0/1 block selector
    # (one 1 per factor per own row, 0 elsewhere).  Folding the duplicates is
    # :func:`dedup`'s job and only :func:`dedup`'s: keeping the builder purely
    # additive means the "full" form and the "deduped" form differ in exactly
    # one way (table size + index values), never in how a row was chosen.
    models = model._pg_particle_models()
    basis_rows, basis_col, basis_name = [], [], []
    gamma_cols = []                     # per (wave, res): its basis columns
    for k, (base, _combo) in enumerate(expansion):
        ls, chain = waves[base]
        row = []
        for r, _fac in enumerate(per_wave_factors[base]):
            name = chain.decays[r + 1].core.name
            mdl = models[name]
            mk = getattr(mdl, "make_basis_transform", None)
            tr = mk() if mk is not None else None
            names, rows = model._pg_den_basis(name, mdl, mass_grid, tr)
            col = int(num_m_index[k, r])
            slots = []
            for j, g in enumerate(rows):
                # kernel encoding: 0 = F=1, 1 = F=-m^2 (sign flipped), 2+r = row
                if j == 0:
                    kind, val = 0, np.ones_like(mass_grid, complex)
                elif j == 1:
                    kind, val = 1, -(mass_grid.astype(float) ** 2) + 0j
                else:
                    kind, val = 2 + (j - 2), np.asarray(g, complex)
                slots.append(len(basis_rows))
                basis_rows.append(val)
                basis_col.append(col)
                basis_name.append(str(names[j]))
            row.append(slots)
        gamma_cols.append(row)

    gamma_matrix = np.zeros((n_wave, n_res, len(basis_rows)), np.float64)
    for k, row in enumerate(gamma_cols):
        for r, slots in enumerate(row):
            for c in slots:
                gamma_matrix[k, r, c] = 1.0

    return {
        "fl_table": fl_table,
        "fl_q_index": fl_q_index,
        "num_table": num_table,
        "num_m_index": num_m_index,
        "basis_table": np.stack(basis_rows),
        "basis_m_index": np.array(basis_col, np.int32),
        "den_names": basis_name,
        "gamma_matrix": gamma_matrix,
        "angle_k": np.array(angle_k, np.int32),
        "angle_b": np.array(angle_b, np.float64),
        "angle_index": angle_index,
        "angle_matrix": angle_matrix,
        "m_min": m_min,
        "m_delta": m_delta,
        "q_min": q_min,
        "q_delta": q_delta,
        "wave_names": wave_names,
    }


# ── helpers ─────────────────────────────────────────────────────────

def dedup(config):
    """Compress duplicate rows of the flat config.  Returns a new dict.

    The builder emits every row *where it is used* — one ``fl`` row per
    (wave, decay), one ``num`` row per (wave, res), one basis entry per
    (factor, basis slot).  That is the honest, index-free form; it also
    repeats identical rows a lot.  This pass folds them:

    ===========  ==================================  ==================
    table        a duplicate is                       produces
    ===========  ==================================  ==================
    ``fl``       row equal **and** ``q_index`` equal  ``fl_index[k, d]``
    ``num``      row equal **and** ``m_index`` equal  ``num_index[k, r]``
    ``basis``    row equal **and** ``m_index`` equal  ``gamma_matrix`` remap
    ===========  ==================================  ==================

    ``fl_q_index`` / ``num_m_index`` / ``basis_m_index`` stay per-use, so a
    consumer still reads exactly one index per (wave, decay) or
    (wave, res) — the dedup changes only how many *table* rows exist.

    The mass column is part of the key, always: two rows that look equal but
    read different event columns are different functions of the event, so
    they must not be folded.  This applies to a *constant* row too — the
    numerator's ``≡ 1`` at column 0 and at column 1 stay separate entries, so
    every index a consumer reads still names the column its row belongs to.
    """
    out = dict(config)

    # ── fl: (n_wave, n_decay, n_interp) + q_index -> (n_forms, n_interp) ──
    fl = np.asarray(config["fl_table"])
    q = np.asarray(config["fl_q_index"])
    n_wave, n_decay, n_interp = fl.shape
    flat_rows = fl.reshape(n_wave * n_decay, n_interp)
    keep, mapping = _dedup_rows(flat_rows, q.reshape(-1))
    out["fl_table"] = flat_rows[keep]
    out["fl_index"] = mapping.reshape(n_wave, n_decay).astype(np.int32)
    out["fl_q_index"] = q                      # per use, unchanged

    # ── num: (n_wave, n_res, n_interp) + m_index -> (n_rows, n_interp) ──
    num = np.asarray(config["num_table"])
    mi = np.asarray(config["num_m_index"])
    n_res = num.shape[1]
    flat_num = num.reshape(n_wave * n_res, n_interp)
    # strict key: a duplicate is row-equal AND mass-column-equal.  A constant
    # row is NOT special-cased — ``1@col0`` and ``1@col1`` stay separate, so
    # the index a consumer reads is always the column the row belongs to.
    keep_n, mapping_n = _dedup_rows(flat_num, mi.reshape(-1))
    out["num_table"] = flat_num[keep_n]
    out["num_index"] = mapping_n.reshape(n_wave, n_res).astype(np.int32)
    out["num_m_index"] = mi

    # ── basis: rows + m_index + names, with the selector remapped ────
    # The builder appends one entry per (factor, basis row), so duplicates are
    # the norm; this is the ONLY place they are folded.
    bas = np.asarray(config["basis_table"])
    bmi = np.asarray(config["basis_m_index"])
    names = config.get("den_names")
    keep_b, mapping_b = _dedup_rows(bas, bmi, names)
    out["basis_table"] = bas[keep_b]
    out["basis_m_index"] = bmi[keep_b]

    if names is not None:
        out["den_names"] = [names[i] for i in keep_b]

    # the old→new column map, so a caller holding a full-config coefficient
    # vector can gather it onto the surviving rows (``coeff[keep_b]``)
    out["basis_keep"] = keep_b
    out["basis_map"] = mapping_b

    # remap the selector: an old column's flag moves to the column its row
    # survived at.  The selector is 0/1, so a fold ORs the flags — two old
    # columns folding into one new column still mean "this factor uses it".
    gam = np.asarray(config["gamma_matrix"])
    new_gam = np.zeros((n_wave, n_res, len(keep_b)), np.float64)
    for m_old in range(len(bmi)):
        np.maximum.at(new_gam, (slice(None), slice(None), mapping_b[m_old]),
                      gam[:, :, m_old])
    out["gamma_matrix"] = new_gam
    return out


def _dedup_rows(rows, extra_index, names=None):
    """Fold equal rows that also share *extra_index* (and *names*).

    Args:
        rows: ``(N, n_interp)`` table rows.
        extra_index: ``(N,)`` ints; two rows only fold when these agree too
            (the mass / momentum column a row reads).
        names: optional ``(N,)`` labels.  When given, two rows only fold when
            their names agree as well — two basis rows that happen to be
            numerically equal but are *different coefficients* must stay
            apart, or one fitted weight would silently drive both.

    Returns:
        ``(keep, mapping)`` — the indices of the rows that survived, and a
        per-input ``[N]`` mapping into the surviving list.
    """
    seen, keep, mapping = {}, [], np.zeros(len(rows), np.int64)
    for i in range(len(rows)):
        key = (rows[i].tobytes(), int(extra_index[i]),
               None if names is None else names[i])
        pos = seen.get(key)
        if pos is None:
            pos = len(keep)
            seen[key] = pos
            keep.append(i)
        mapping[i] = pos
    return np.array(keep, np.int64), mapping


def _mass_column(model, chain, r):
    """The event mass column that BW ``r`` of ``chain`` reads.

    ``n_res·topo + r`` — the topology block plus the resonance's position in
    the chain, which is the same column the denominator reads (the old
    builder's ``n_res * topo + idx - 1``).  Deriving it here rather than
    storing a second array is what keeps the numerator and the propagator
    from ever disagreeing about which mass they evaluate.
    """
    topo = model.topo_index[chain.topo_id()]
    return int(model.n_res * topo + r)


def _barrier_row(form, q):
    """One barrier form sampled on the momentum grid."""
    from tabpwa.bw_form_factor import BarrierFactor, build_barrier
    if isinstance(form, BarrierFactor):
        return form.factor(q)
    if isinstance(form, (tuple, list)):
        kind, d, L = form
        return build_barrier(kind, L=L, q0_ref=1.0, d=d).factor(q)
    return build_barrier("bw", L=int(form), q0_ref=1.0).factor(q)
