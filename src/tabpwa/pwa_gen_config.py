"""Kernel config for the projection-sum PWA, built from the decay-group loop.

The model, per (expanded wave ``k``, projection ``p``, event ``i``):

    F_{ipk} = fa_{pk} · Π_d fl_table[k, d]( q_{i, fl_q_index[k, d]} )
                     · Π_r num_table[k, r]( m_{i, num_m_index[k, r]} )
    fa_{p·n_wave+k} = Σ_a matrix_angle[a, p·n_wave+k]
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
  an explicit ``gamma_matrix``, and it is **folded here** on
  ``(coeff name, basis kind, mass column)``.  That triple is the row's
  physical identity, so folding it is lossless: every consumer derives
  ``n_basis`` from ``basis_table.shape[0]``, and the parameter surface is
  name-keyed, so a fitter sees the same weights with fewer rows.  The mass
  column is part of the key deliberately — two rows that read different
  event columns are different functions of the event (the same resonance in
  two topologies), and merging them would drive a row sampled at the wrong
  mass.  ``basis_kind`` / ``basis_key`` record the triple.  The matrix
  carries **structure only** (0/1): the fitted weights are
  ``params["coeff"]`` at run time.
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
``matrix_angle``  (n_angle_basis, n_proj·n_wave) complex, p-major
``m_min``/``m_delta``                          float    mass grid (num + basis)
``q_min``/``q_delta``                          float    momentum grid (fl)
``wave_names``    list[str]                             host metadata
"""
import itertools

import numpy as np


def drop_unit_factors(kc):
    """Drop the BW factors that are the IDENTITY in every wave.

    A pure KERNEL-CONFIG optimisation, on the parsed dict only: no particle
    model, no name, no flag is consulted — everything is read from the arrays
    ``build_from_groups`` emitted.

    Factor ``r`` is droppable when, in EVERY wave, its DENOMINATOR selects
    only basis rows named ``"one"`` AND its NUMERATOR is the constant 1 —
    the factor axis is the denominator's (``basis_m_index`` /
    ``gamma_matrix``), but a unit denominator does not by itself make a factor
    trivial: ``spline_c_idx`` states its denominator as ``[1]`` while its
    NUMERATOR carries the whole lineshape, and dropping it would silently
    delete the expansion.  So ``num_table[:, r]`` is checked as well, and
    ``skip`` (numerator 1, denominator 1) stays droppable.  Those ``r`` are
    removed from the factor axis
    ``(wave, r)`` — ``num_table``, ``num_m_index``, ``gamma_matrix`` — and the
    basis rows that only a dropped factor selected are pruned with them (the
    shared ``"one"`` row survives whenever a kept factor uses it).

    The axis is SHARED, so a factor that is the identity in only some waves is
    NOT dropped (a partial drop would silently mis-align the remaining
    factors); ``kc`` itself is returned in that case, and whenever nothing is
    droppable, so existing configs are bit-identical.

    The mass COLUMNS are NOT touched: ``num_m_index`` / ``basis_m_index`` stay
    STRUCTURAL indices into the event ``mass`` array (which still carries a
    column per factor).  Only the factor axis shrinks; the kernel never reads
    the dropped factor's column.  A factor that is 1 in only SOME waves is
    left alone.
    """
    if "gamma_matrix" not in kc or "den_names" not in kc:
        return kc
    gam = np.asarray(kc["gamma_matrix"])
    if gam.ndim != 3:
        return kc
    n_wave, n_res, n_basis = gam.shape
    den = [str(n) for n in kc["den_names"]]
    if len(den) != n_basis:
        return kc
    # a row of the denominator is the constant ONE only when all three say so:
    # the selection (gamma_matrix), the ROW ITSELF (basis_table == 1) and the
    # weight it is multiplied by (den_names -> the fixed "one" coefficient,
    # which carries no parameter and is pinned to 1)
    tab = kc.get("basis_table")
    tab = np.asarray(tab) if tab is not None else None
    is_one = np.zeros(len(den), bool)
    for j, name in enumerate(den):
        if str(name) != "one":
            continue
        if tab is None or np.allclose(np.asarray(tab[j]).ravel(), 1.0,
                                      rtol=0.0, atol=1e-14):
            is_one[j] = True
    # the numerator is part of the factor's identity: a unit denominator is
    # necessary but not sufficient (``spline_c_idx`` expands in the numerator)
    num_tab = kc.get("num_table")
    if num_tab is not None:
        num_tab = np.asarray(num_tab)
        if num_tab.ndim != 3 or num_tab.shape[1] != n_res:
            # FAIL CLOSED.  A present-but-unreadable numerator axis must NOT
            # restore the pre-guard behaviour silently: the factor would be
            # dropped while its numerator survived nowhere.  ``dedup()``
            # emits a 2-D ``num_table`` (+ ``num_index``), so a deduped
            # config cannot answer this question at all.
            raise ValueError(
                f"drop_unit_factors: 'num_table' has shape {num_tab.shape}, "
                f"expected (n_wave, n_res={n_res}, n_interp) — the numerator "
                f"axis must be readable to decide whether a factor is the "
                f"identity.  Pass the FULL config (``build_from_groups`` "
                f"output); a deduped one carries a 2-D num_table")
    drop = []
    for r in range(n_res):
        identity = (num_tab is None
                    or bool(np.allclose(num_tab[:, r], 1.0,
                                        rtol=0.0, atol=1e-14)))
        if not identity:
            continue
        for k in range(n_wave):
            sel = np.nonzero(gam[k, r])[0]
            if sel.size == 0 or not np.all(is_one[sel]):
                identity = False
                break
        if identity:
            drop.append(r)
    if not drop:
        return kc
    keep = np.ones(n_res, bool)
    keep[drop] = False
    if not keep.any():
        # never empty the axis: a config with no factor left is refused by the
        # builder, and dropping everything here would be a silent no-op shape
        return kc

    out = {**kc}
    for key in ("num_table", "num_m_index"):
        if key in out and np.shape(out[key])[1:2] == (n_res,):
            out[key] = np.asarray(out[key])[:, keep]
    gam = gam[:, keep, :]
    used = np.unique(np.nonzero(gam.reshape(-1, n_basis))[1]) \
        if n_wave * int(keep.sum()) else np.zeros(0, int)
    out["gamma_matrix"] = gam[:, :, used]
    # `num_m_index`/`basis_m_index` keep their STRUCTURAL values: they are
    # columns of the event `mass` array, which stays structural (the builder
    # writes every factor).  Only the ROWS of `basis_m_index` are pruned, with
    # the basis rows they label.
    for key in ("basis_table", "basis_m_index", "basis_m_key", "basis_kind"):
        if key in out:
            arr = out[key]
            out[key] = [arr[i] for i in used] if key == "basis_key" \
                else np.asarray(arr)[used]
    out["den_names"] = [den[i] for i in used]
    out["basis_key"] = [tuple(kc["basis_key"][i]) for i in used] \
        if "basis_key" in kc else out.get("basis_key")
    # the factor axis is gone from these two; they are plain-pwa keys and the
    # flat config does not emit them, so nothing to renumber here
    return out


def build_from_groups(model, n_interp=None, drop_unit=True):
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
    n_res = int(model.n_res)                 # structural
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

    # ── block expansion: identical permutations x CP ────────────────────
    # A declared `identical_particles` permutation and a `cp_particles`
    # conjugation each add a BLOCK.  The event data concatenates one column
    # band per block (`build_tree_event_data`: `mass` -> n_blocks·n_topo·n_res,
    # `q` -> n_blocks·n_topo·n_decay, `angle` -> n_blocks·n_topo), so a block
    # is exactly a COLUMN OFFSET: the wave axis grows to n_blocks·n_wave and
    # every per-wave index is rebased by `blk * <per-block stride>`.
    #
    # This is why no backend needs to know about blocks: they read
    # `fl_q_index` / `num_m_index` / `angle_index` as plain column numbers and
    # iterate `k` over `fl_table.shape[0]`.  Emitting the expanded rows makes
    # `n_wave` == `len(ck)` on its own (the parameter side already expands,
    # `get_ck_map`: cp-outer, perm-inner, combo innermost).
    #
    # The ORDER must match `get_ck_map` / `block_orders`: block
    # `b = cp * n_perm + perm`, which is what `DecayTree.block_orders()`
    # yields (all permutations, then the CP copies of them).
    # `DecayTree.block_orders()` is the single source of the block list; its
    # order (all permutations, then their CP copies) matches `get_ck_map`'s
    # (cp-outer, perm-inner), so block `b` is ck block `b`.
    blocks = model.decay_tree.block_orders()
    n_blocks = len(blocks)
    if n_blocks > 1:
        base_expansion = expansion
        expansion = [(blk, base, combo)
                     for blk, (_order, _is_cp) in enumerate(blocks)
                     for (base, combo) in base_expansion]
        n_wave = len(expansion)
    else:
        expansion = [(0, base, combo) for (base, combo) in expansion]
    # a block contributes one topology-sized band to each event buffer
    q_blk_stride = int(model.n_topo) * n_decay      # q  : n_topo*n_decay
    m_blk_stride = int(model.n_topo) * n_res        # mass: n_topo*n_res
    a_blk_stride = int(model.n_topo)                # angle: n_topo

    # ── per-(wave, decay): momentum column and barrier row ────────────
    fl_q_index = np.zeros((n_wave, n_decay), np.int32)
    fl_table = np.zeros((n_wave, n_decay, n_interp), np.float64)
    wave_names = [""] * n_wave
    for k, (blk, base, _combo) in enumerate(expansion):
        ls, chain = waves[base]
        topo = topo_id[chain.topo_id()]
        wave_names[k] = str(chain.topo_id())
        for d, decay in enumerate(chain.decays):
            fl_q_index[k, d] = blk * q_blk_stride + n_decay * topo + d
            fl_table[k, d] = _barrier_row(model._barrier(decay, ls[d][0]),
                                          q_grid)

    # ── per-(wave, res): numerator row and mass column ────────────────
    num_table = np.ones((n_wave, n_res, n_interp), np.complex128)
    num_m_index = np.zeros((n_wave, n_res), np.int32)
    for k, (blk, base, combo) in enumerate(expansion):
        ls, chain = waves[base]
        for r, fac in enumerate(per_wave_factors[base]):
            num_m_index[k, r] = blk * m_blk_stride + _mass_column(model,
                                                                  chain, r)
            rowfun = combo.get(fac["slot"])
            if rowfun is not None:
                num_table[k, r] = np.asarray(rowfun(mass_grid), complex)

    # ── angular basis (shared) + the per-(wave, proj) weights ─────────
    # The basis is keyed on ``(column base, k, b)`` — the topology, not the
    # wave — exactly as the model's ``unique_angle_basis`` does.  Two waves of
    # one topology therefore share basis rows, and ``angle_index[b]`` (the
    # topology) is the event-angle column base their vertices start at.  The
    # key uses the BLOCK-REBASED column, so a CP block gets its own rows: its
    # data columns are the conjugated ones, a different function of the event.
    #
    # The formula itself is block-INVARIANT: `_wave_angle_terms` depends only
    # on (chain, ls, projection), and a CP block is the same chain evaluated
    # on conjugated momenta (the CP transform is applied to the DATA, in
    # `build_tree_event_data`, not to the formula).
    angle_terms = {}
    wave_topo = []
    for k, (blk, base, _combo) in enumerate(expansion):
        ls, chain = waves[base]
        col_base = blk * a_blk_stride + topo_id[chain.topo_id()]
        wave_topo.append(col_base)
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
    # ``matrix_angle`` keeps the historical layout — ``(n_angle_basis,
    # n_proj·n_wave)`` with the projection M AJOR, column ``p·n_wave + k`` — so
    # the long-standing idiom still works verbatim:
    #
    #     ka  = Π_j cos(angle·angle_k + angle_b)        (ne, n_angle_basis)
    #     fa  = ka @ kc["matrix_angle"]                 (ne, n_proj·n_wave)
    #     fa[:, p*n_wave + k]                           projection p, wave k
    #
    # ``n_proj`` is therefore just ``matrix_angle.shape[1] // n_wave`` — it is
    # NOT a separate count and NOT an axis of this array.  ``ck`` stays one
    # slot per BASE wave (``len == n_wave``) and is shared by every
    # projection, which is the flat config's one deliberate change from the
    # pre-flat layout.
    matrix_angle = np.zeros((len(basis_pos), P * n_wave), np.complex128)
    for k in range(n_wave):
        topo = wave_topo[k]
        for p in range(P):
            for term in angle_terms[(k, p)]:
                key = (topo, tuple(term["k"]), tuple(term["b"]))
                matrix_angle[basis_pos[key], p * n_wave + k] += term["coeffs"]

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
    # The basis is folded HERE, on the structural key
    # ``(coeff name, basis kind, mass column)``.  Two rows with the same triple
    # are the same physical quantity, so folding them is lossless: every
    # consumer derives ``n_basis`` from ``basis_table.shape[0]`` and the
    # parameter surface is name-keyed, so a caller sees fewer rows and the
    # same weights.  Rows that merely look numerically alike are NOT folded —
    # two identical resonances at different mass columns stay apart, because
    # the mass column is part of the key (they are different functions of the
    # event, and merging them would drive a row sampled at the wrong mass).
    models = model._pg_particle_models()
    basis_rows, basis_col, basis_name, basis_kind = [], [], [], []
    seen = {}
    gamma_cols = []                     # per (wave, res): its basis columns
    for k, (_blk, base, _combo) in enumerate(expansion):
        ls, chain = waves[base]
        row = []
        for r, fac in enumerate(per_wave_factors[base]):
            # the FACTOR's own name, not `decays[r+1]`: a CONSTANT factor is
            # skipped, so the factor slot and the decay index diverge
            name = fac["name"]
            mdl = models[name]
            # No transform is needed to STATE the basis: the rows come from
            # the model and the names from ``get_den_name()``.  Whether a
            # transform produces those names is a constraint-layer question,
            # answered by `PWAGen.extra_transforms()`.
            names, rows = model._pg_den_basis(name, mdl, mass_grid)
            col = int(num_m_index[k, r])
            slots = []
            for j, g in enumerate(rows):
                # The rows are used VERBATIM as the model returns them.  The
                # sign of the m² row therefore lives in the ROW, not in a
                # consumer-side flip: `den_basis` decides it and the weight
                # for that row is the shared literal ``"one"`` (fixed to 1).
                # Flipping one side without the other silently reverses every
                # m² term.
                kind = j
                val = np.asarray(g, complex)
                key = (str(names[j]), int(kind), col)
                pos = seen.get(key)
                if pos is None:
                    pos = len(basis_rows)
                    seen[key] = pos
                    basis_rows.append(val)
                    basis_col.append(col)
                    basis_name.append(str(names[j]))
                    basis_kind.append(int(kind))
                slots.append(pos)
            row.append(slots)
        gamma_cols.append(row)

    gamma_matrix = np.zeros((n_wave, n_res, len(basis_rows)), np.float64)
    for k, row in enumerate(gamma_cols):
        for r, slots in enumerate(row):
            for c in slots:
                gamma_matrix[k, r, c] = 1.0

    ret = {
        "fl_table": fl_table,
        "fl_q_index": fl_q_index,
        "num_table": num_table,
        "num_m_index": num_m_index,
        "basis_table": np.stack(basis_rows),
        "basis_m_index": np.array(basis_col, np.int32),
        "basis_kind": np.array(basis_kind, np.int32),
        "basis_key": list(seen),
        "den_names": basis_name,
        "gamma_matrix": gamma_matrix,
        "angle_k": np.array(angle_k, np.int32),
        "angle_b": np.array(angle_b, np.float64),
        "angle_index": angle_index,
        "matrix_angle": matrix_angle,
        "m_min": m_min,
        "m_delta": m_delta,
        "q_min": q_min,
        "q_delta": q_delta,
        "wave_names": wave_names,
    }
    # `drop_unit=False` keeps every factor, which is the reference the
    # optimisation must agree with: same amplitude, one factor fewer
    return drop_unit_factors(ret) if drop_unit else ret



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
    ===========  ==================================  ==================

    The **basis is not folded here** — ``build_from_groups`` already did it,
    on ``(coeff name, basis kind, mass column)``.  This pass only folds
    ``fl`` and ``num``, because those are the two tables whose
    *dimensionality* changes (3-D -> 2-D) and which therefore need the index
    arrays.

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

    # ── basis: ALREADY FOLDED BY THE BUILDER ─────────────────────────
    # ``build_from_groups`` folds the basis on ``(coeff name, kind, mass
    # column)``, so nothing is left to compress here.  The selector and the
    # index arrays pass through unchanged; ``dedup`` only folds ``fl`` and
    # ``num`` (which DO change table dimensionality and therefore need the
    # ``fl_index``/``num_index`` arrays).
    out["basis_table"] = np.asarray(config["basis_table"])
    out["basis_m_index"] = np.asarray(config["basis_m_index"])
    if "den_names" in config:
        out["den_names"] = list(config["den_names"])
    if "basis_kind" in config:
        out["basis_kind"] = np.asarray(config["basis_kind"])
    if "basis_key" in config:
        out["basis_key"] = list(config["basis_key"])
    keep_b = np.arange(np.asarray(config["basis_table"]).shape[0], dtype=np.int64)
    mapping_b = keep_b.copy()
    out["basis_keep"] = keep_b
    out["basis_map"] = mapping_b
    gam = np.asarray(config["gamma_matrix"])
    new_gam = gam
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
