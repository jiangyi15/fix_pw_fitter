"""The flat ``pwa_gen`` family: the CONFIG contract, kernels as parameters.

ONE file for what a flat CUDA leaf promises about a CONFIG, beyond the shared
compute suite (:mod:`_flat_compute`): which configs it accepts and refuses, and
that its numbers do not move when a layout optimisation is applied to that
config.

The KERNELS ARE PARAMETERS.  What a leaf must DO is a contract, so it is written
once and parametrised over the registry names; what genuinely differs is a
per-kernel DATA entry in :data:`KERNELS` — the same shape as
:data:`_flat_compute.FLOORS`.  A gate that exists for one kernel is one field of
that entry, never a reason to keep a second file.

Nothing here inspects HOW a kernel is built: no dedup-map internals, no CSR
structure, no launch counters, no symbol lists, no module/`.so` identity.  Those
turn a legitimate re-implementation into a red test and are not part of the
contract.  What IS the contract, and what replaces the old per-kernel index
pins, is OBSERVABLE:

  * the full-vs-``dedup``ed layout pair is numerically IDENTICAL where both are
    accepted (that is the real content of "the dedup must not move a number"),
    and the leaf that reads the full form REFUSES the folded one;
  * the slot collapse — the leaf's own grouping of identical waves — is a
    reassociation, so forcing it ON or OFF must give the same numbers, and both
    must match the fp64 reference;
  * ``s0c2``'s fused reductions are a launch-structure change, so it must agree
    with ``s0c`` bit for bit, on the fixtures AND on a ``n_basis > 32`` config
    (which is where the fallback lives — no counter is read for that);
  * the refusals: a multi-BW chain on an ``n_res1`` leaf (naming the general
    backend), a non-0/1 gamma weight, an empty gamma row, a config past the
    device's shared-memory limit.

Test data stays INSIDE both interpolation grids (``_flat_grid._in_grid``) —
outside, the tables clamp and CPU/GPU diverge.
"""
import os

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import backends_for_model, create_backend
from tabpwa.pwa_gen_config import dedup

from _flat_compute import CONFIG_OWNER, SPECS as LEAVES
from _flat_grid import (CFG_CP, CFG_IDENT, CFG_KM, CFG_MB, CFG_ONE, CFG_SAME,
                        QUALIFYING, _in_grid)

# ── the fixtures ────────────────────────────────────────────────────
ALL_CFGS = [CFG_ONE, CFG_KM, CFG_MB, CFG_IDENT, CFG_SAME, CFG_CP]
CFG_IDS = ["one", "km", "mb", "ident", "same", "cp"]
#: the SECOND multi-BW fixture: three topologies sharing one factor, so the
#: factor-level gate has something to fold that the wave collapse does not
CFG_MB2 = "tests/config_pwa_gen_mb2.yml"
#: the IDENTICAL-PARTICLE multi-BW fixture: `n_blocks > 1` AND `n_res > 1`
#: (32 waves = 8 blocks x 4, n_res 2), the combination `_mb`/`_mb2` do not
#: reach — the gamma row of a factor is shared across CP/permutation blocks too
CFG_4PI = "tests/config_pwa_gen_4pi_mb.yml"
CFG_ID = {c: i for c, i in zip(ALL_CFGS, CFG_IDS)}

# ── the kernels: one entry per registered flat leaf ─────────────────
#: ``floors`` is ``(rel, P, coeff)`` — the SAME per-backend numbers
#: :data:`_flat_compute.FLOORS` records, so a leaf cannot be looser here than in
#: the shared suite.  The flags select which contract tests apply:
#:
#:   multi_bw    — accepts a multi-WB chain (no ``n_res1`` tag); the multi-BW
#:                 anchor and gradient checks run only where it does
#:   slot        — has a forced ``slot_collapse`` (its own wave grouping)
#:   rsize_opts  — the grouped objective's forced-collapse variants to sweep
#:   wide        — carries a shared-memory tile, so the opt-in gates apply
#:   hole        — the sparse/CSR gamma path: a hole must not move a number
#:   gamma01     — the m-major reduction: the gamma weights must be 0/1
#:   zero_coeff  — an all-zero weight vector is the empty model (NaN, no crash)
#:   reused_handle — has a load-time cache, so a reused handle is a contract
KERNELS = {
    "gen": dict(
        name="cuda_flat_pwa_gen",
        # fp64 and uncached: MEASURED 3.9e-15 / 3.4e-15 / 9.0e-15 against the
        # reference (the same record as _flat_compute.FLOORS, which this file
        # may not be looser than)
        floors=(1e-12, 1e-12, 1e-12),
        multi_bw=True,
        zero_coeff=True,
    ),
    "cache": dict(
        name="cuda_flat_cache_pwa_gen",
        floors=(1e-6, 1e-6, 1e-4),
        multi_bw=True,
        zero_coeff=True,
        reused_handle=True,
    ),
    "gen2": dict(
        name="cuda_flat_pwa_gen2",
        floors=(2e-5, 3e-5, 1e-4),
        multi_bw=True,
        slot=True,
        rsize_opts=(None,),
        wide=True,
        zero_coeff=True,
    ),
    "fslot": dict(
        # the FACTOR-LEVEL dedup copy of gen2: same forward pass, same fp64
        # `s_pv`, a backward sweep over distinct (wave, factor) rows instead of
        # wave blocks.  `slot=True` because the parent's `slot_collapse` knob
        # is still there (it is forced off whenever the factor gate engages).
        name="cuda_flat_pwa_gen_fslot",
        floors=(2e-5, 3e-5, 1e-4),
        multi_bw=True,
        slot=True,
        rsize_opts=(None,),
        wide=True,
        zero_coeff=True,
    ),
    "1bw": dict(
        name="cuda_flat_pwa_gen_1bw",
        floors=(2e-5, 3e-5, 1e-4),
        multi_bw=False,
        slot=True,
        rsize_opts=(False, True),
        wide=True,
        hole=True,
        zero_coeff=True,
    ),
    "s0c": dict(
        name="cuda_flat_pwa_gen_s0c",
        floors=(2e-5, 4e-5, 2e-4),
        multi_bw=False,
        hole=True,
        gamma01=True,
        zero_coeff=True,
    ),
    "s0c2": dict(
        name="cuda_flat_pwa_gen_s0c2",
        floors=(2e-5, 4e-5, 2e-4),
        multi_bw=False,
        hole=True,
        gamma01=True,
        zero_coeff=True,
    ),
}

#: the table above is hand-written, so it is checked against the OWNERSHIP rule
#: in ``_flat_compute``: a leaf registered later is in neither this table nor
#: the owner map, so it fails here instead of being silently uncovered by every
#: config-contract file.
_ME = os.path.basename(__file__)
_OWNED = {n for n, f in CONFIG_OWNER.items() if f == _ME}
_TABLE = {k["name"] for k in KERNELS.values()}
assert _TABLE == _OWNED, (
    f"{_ME}'s KERNELS is out of sync with _flat_compute.CONFIG_OWNER: "
    f"missing {sorted(_OWNED - _TABLE)}, stale {sorted(_TABLE - _OWNED)}")
assert set(CONFIG_OWNER) == set(LEAVES) | set(
    backends_for_model("pwa_gen", tag="coeff_pinned"))


#: the fp32 cache floor: the load-time amplitude is float2, so agreement with
#: the fp64 `numpy_flat` is limited by fp32, not by the algebra.  MEASURED over
#: the wide-range fixtures (`_ident`/`_cp` span 7.5e-4 .. 7.2e2, so `_rel` —
#: worst absolute / global max — is set by the near-zero events).
CACHE_FLOOR = 2e-5
CACHE_FLOOR_P = 3e-5
CACHE_FLOOR_COEFF = 1e-4
NH = 1e-6

#: ``rs = 1`` is the shared anchor's sweep; these are the group sizes only the
#: grouped path covers here (including a group larger than the 8-event tile).
RSIZES = (2, 4, 8, 10, 25, 100)

#: a config with more basis rows than the fused feature reduction's register
#: budget, so the ``n_basis > 32`` fallback is reached (no counter is read: the
#: fallback is gated by the parity it must not disturb).
WIDE_BASIS = 40

#: the production config the s0c pair's memory case was measured on.  The
#: fixtures are tiny by design; this is the shape real fits use.
REAL_CFG = ("/home/jiangy/github/project94/pwa_workspace/jpsi_eta_pi_pi/"
            "04_model_search/current/config.yml")
#: MEASURED on `REAL_CFG` (26 waves, 62 basis rows, 48 events, norm 1.3): the
#: worst of the six leaves against `numpy_flat` is Q 1.45e-7, P 1.60e-5,
#: ck 4.60e-7, coeff 1.04e-5 (the fp64 leaf alone: 2.6e-16 / 1.8e-15 / 3.4e-15
#: / 2.5e-15).  The bound tracks P's dynamic range on a production config
#: (wider than any fixture), so it is its own number — never a loosening of the
#: fixtures' floors.  It is deliberately LOOSE (96x-6900x against the numbers
#: above) because this is the wide-range net: it must hold for every leaf on a
#: shape no fixture has, not pin any one of them.  (It read Q 4.7e-6 /
#: P 5.7e-4 / ck 6.1e-6 / coeff 3.4e-4 while the reference interpolated
#: two-point linearly — the chord error was most of it.)
REAL_FLOOR = (1e-3, 3e-3, 1e-3)


def _kernels(*keys):
    return [k for k in KERNELS if k in keys]


MULTI_BW = _kernels("gen", "cache", "gen2", "fslot")
SLOT = _kernels("gen2", "fslot", "1bw")
RSIZE = _kernels("gen2", "fslot", "1bw")
WIDE = _kernels("cache", "gen2", "fslot", "1bw")
HOLE = _kernels("1bw", "s0c", "s0c2")
GAMMA01 = _kernels("s0c", "s0c2")
N_RES1 = _kernels("1bw", "s0c", "s0c2")


# ── helpers ─────────────────────────────────────────────────────────

def _fit(cfg):
    f = Fitter(cfg)
    f.apply_constrains()
    return f


def _setup(cfg, ne=64, seed=0):
    """``(kernel_config, params, data)`` for a fixture."""
    f = _fit(cfg)
    kc = f.kernel_config
    params, _res = f.build_params(f.initial_values(seed=1))
    return kc, params, _in_grid(kc, ne, seed)


def _backend(name, kc, **kw):
    return create_backend(name, kc, model="pwa_gen", **kw)


def _numpy_flat(kc, rs=1):
    return create_backend("numpy_flat", kc, model="pwa_gen",
                          resolution_size=rs)


def _rel(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _replicate(data, rs):
    """A group-replicated sample (weights normalised so Σw is the group count)."""
    out = {k: np.repeat(np.asarray(v), rs, axis=0) for k, v in data.items()}
    out["weight"] = np.full(len(np.asarray(data["weight"])) * rs, 1.0 / rs)
    return out


def _wide_kc(n_wave_target, cfg=CFG_KM):
    """A config tiled to ``n_wave_target`` waves and 2 projections.

    Above ~108 waves the shared tile needs the device's opt-in shared-memory
    limit, so this is the shape the opt-in gates are written on.
    """
    from benchmark_flat_vs_pwa import tile_flat
    f = _fit(cfg)
    kc = dict(tile_flat(f.kernel_config, n_wave_target, 2))
    kc["num_table"] = np.ones_like(np.asarray(kc["num_table"]))
    return f, kc


def _wide_params(f, kc):
    """Parameters sized for a WIDE config (its own n_wave / n_basis)."""
    n_wave = int(np.asarray(kc["fl_table"]).shape[0])
    n_basis = int(np.asarray(kc["basis_table"]).shape[0])
    p0, _ = f.build_params(f.initial_values(seed=1))
    ck0 = np.asarray(p0["ck"])
    return {"ck": np.tile(ck0, (n_wave + ck0.size - 1) // ck0.size)[:n_wave],
            "coeff": np.random.default_rng(7).uniform(0.3, 1.2, n_basis)}


def _pad_basis(kc, n_basis, params):
    """The same model with ``n_basis`` rows: the extra ones select no wave.

    A padded row is a copy of row 0 with a ZERO gamma column, so it contributes
    nothing to ``D`` and its weight is never read — the numbers must be those of
    the unpadded config.  This is how a config wider than 32 basis rows is
    obtained without inventing a model.
    """
    kc = dict(kc)
    bas = np.asarray(kc["basis_table"])
    pad = int(n_basis) - bas.shape[0]
    assert pad > 0
    kc["basis_table"] = np.concatenate([bas, np.repeat(bas[:1], pad, axis=0)],
                                       axis=0)
    kc["basis_m_index"] = np.concatenate(
        [np.asarray(kc["basis_m_index"], np.int64), np.zeros(pad, np.int64)])
    gam = np.asarray(kc["gamma_matrix"])
    kc["gamma_matrix"] = np.concatenate(
        [gam, np.zeros(gam.shape[:2] + (pad,))], axis=2)
    for nm in ("den_names", "basis_kind", "basis_m_key", "basis_key"):
        if nm in kc:
            v = kc[nm]
            kc[nm] = (np.asarray(list(v) + [v[0]] * pad)
                      if isinstance(v, np.ndarray) else list(v) + [v[0]] * pad)
    params = dict(params)
    params["coeff"] = np.concatenate(
        [np.asarray(params["coeff"]), np.zeros(pad)])
    return kc, params


def _real_config():
    """``(kc, params)`` of the production config, built as a ``pwa_gen`` model."""
    if not os.path.exists(REAL_CFG):
        pytest.skip(f"real config not present: {REAL_CFG}")
    d = dict(Fitter(REAL_CFG).config.dic)
    d["amp_model"] = "pwa_gen"
    d["backend"] = {"name": "numpy_flat"}
    f = Fitter(d)
    f.apply_constrains()
    params, _res = f.build_params(f.initial_values(seed=1))
    return f.kernel_config, params


# ── the moved sample helper: the reformat changed no sample ────────────

def _in_grid_head(kc, ne, seed=0):
    """The PRE-MOVE body (``HEAD:tests/test_cuda_flat_cache_pwa_gen._in_grid``).

    Reproduced here only to pin :func:`_flat_grid._in_grid` to it: the move
    reformatted the arithmetic, and "behaviourally identical" is a claim that
    has to be checkable, or a later tidy-up could silently shift every measured
    floor that depends on the sample.
    """
    rng = np.random.RandomState(seed)
    n_basis = int(np.asarray(kc["basis_table"]).shape[1])
    n_fl = int(np.asarray(kc["fl_table"]).shape[-1])
    m_min, m_del = float(kc["m_min"]), float(kc["m_delta"])
    q_min, q_del = float(kc["q_min"]), float(kc["q_delta"])
    n_mass = int(np.max(kc["num_m_index"])) + 1
    n_q = int(np.max(kc["fl_q_index"])) + 1
    n_at = int(np.max(kc["angle_index"])) + 1
    nac = int(np.asarray(kc["angle_k"]).shape[1])
    return {
        "mass": m_min + 0.05 * m_del * n_basis
                + rng.rand(ne, n_mass) * (0.9 * m_del * n_basis),
        "q": q_min + 0.05 * q_del * n_fl
             + rng.rand(ne, n_q) * (0.9 * q_del * n_fl),
        "angle": rng.uniform(-np.pi, np.pi, (ne, n_at, nac)),
        "weight": 0.5 + rng.rand(ne),
        "bkg": 0.05 + 0.1 * rng.rand(ne),
    }


@pytest.mark.parametrize("cfg", [CFG_ONE, CFG_KM, CFG_CP], ids=CFG_ID.get)
@pytest.mark.parametrize("seed", [0, 3])
def test_the_shared_sample_is_bit_for_bit_the_moved_recipe(cfg, seed):
    """``_flat_grid._in_grid`` == the pre-move recipe, array for array."""
    kc, _params, _data = _setup(cfg)
    moved = _in_grid(kc, 24, seed)
    head = _in_grid_head(kc, 24, seed)
    assert set(moved) == set(head)
    for key in head:
        assert np.array_equal(moved[key], head[key]), key


# ════════════════════════════════════════════════════════════════════
# 1. model scope
# ════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("key", list(KERNELS))
def test_registration(key):
    """Every flat leaf is a ``pwa_gen`` backend and not a plain-``pwa`` one."""
    allowed = set(backends_for_model("pwa_gen"))
    assert KERNELS[key]["name"] in allowed
    assert "numpy_flat" in allowed
    assert KERNELS[key]["name"] not in backends_for_model("pwa")


# ════════════════════════════════════════════════════════════════════
# 2. the dedup layout pair: parity where accepted, refusal where not
# ════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("cfg", ALL_CFGS, ids=CFG_ID.get)
def test_the_dedup_layout_does_not_move_a_number(cfg):
    """``dedup`` folds duplicate table rows and changes NO number.

    This is the real content of the old per-kernel index pins: ``dedup``
    rewrites ``fl``/``num`` as ``(row, index)`` pairs, so a fold that merged two
    rows that are NOT the same function of the event — or a mass column that
    drifted out of the key — shows up here as a changed ``Q``/``P``/gradient,
    independently of any kernel.

    ``numpy_flat`` is the one backend that reads BOTH layouts, which is what
    makes the parity observable; every CUDA leaf reads the full form only and
    refuses the folded one (the next test).
    """
    kc, params, data = _setup(cfg)
    full = _numpy_flat(kc)
    folded = _numpy_flat(dedup(kc))
    try:
        Qf, gf, Pf = full.compute(params, full.load_data(data), norm=1.3)
        Qd, gd, Pd = folded.compute(params, folded.load_data(data), norm=1.3)
        assert Qd == Qf, (cfg, Qd, Qf)
        assert np.array_equal(Pd, Pf), cfg
        assert np.array_equal(gd["ck"], gf["ck"]), cfg
        assert np.array_equal(gd["coeff"], gf["coeff"]), cfg
        assert np.array_equal(gd["norm"], gf["norm"]), cfg
    finally:
        full.free()
        folded.free()


@pytest.mark.parametrize("key", list(KERNELS))
def test_a_deduped_config_is_refused(key):
    """The folded layout is rejected loudly, and the error says why.

    A leaf that reads every table row would silently mis-index a folded one
    (``fl_index``/``num_index`` are the per-use indices it does not read), so
    the refusal is the contract — never a best-effort run.
    """
    kc, _params, _data = _setup(CFG_KM)
    folded = dedup(kc)
    assert "fl_index" in folded and "num_index" in folded
    with pytest.raises(ValueError, match="FULL"):
        _backend(KERNELS[key]["name"], folded)


# ════════════════════════════════════════════════════════════════════
# 3. the numeric contract on the configs the leaves were measured with
# ════════════════════════════════════════════════════════════════════

#: The multi-BW shapes, which differ in what the denominator axis multiplies:
#: `_mb` one topology, `_mb2` three sharing a factor, `4pi` identical-particle
#: blocks AND two topologies at once (`n_blocks > 1` x `n_res > 1`).
MULTI_BW_CFGS = [CFG_MB, CFG_MB2, CFG_4PI]
MULTI_BW_IDS = ["mb", "mb2", "4pi"]


@pytest.mark.parametrize("cfg", MULTI_BW_CFGS, ids=MULTI_BW_IDS)
@pytest.mark.parametrize("norm", [None, 1.3])
@pytest.mark.parametrize("key", MULTI_BW)
def test_the_multi_bw_anchor(key, norm, cfg):
    """Every multi-BW shape against ``numpy_flat``, at the leaf's own floor.

    The shared anchor sweeps the ``n_res == 1`` fixtures only, so the multi-BW
    shapes are anchored here — with the SAME per-backend numbers
    :data:`_flat_compute.FLOORS` records, so a leaf cannot be looser here.
    The CONFIG axis is what makes this the contract rather than one shape:
    `_mb` has one topology, `_mb2` three sharing a factor, and `4pi` is the
    only place where identical-particle blocks multiply the wave axis AND a
    two-resonance chain multiplies the factor axis (32 waves = 8 blocks x 4).
    """
    rel, floor_p, floor_c = KERNELS[key]["floors"]
    kc, params, data = _setup(cfg)
    ref = _numpy_flat(kc)
    gpu = _backend(KERNELS[key]["name"], kc)
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=norm)
        Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=norm)
        assert _rel(Pg, Pr) < floor_p, "P"
        assert _rel(Qg, Qr) < rel, "Q"
        assert _rel(gg["ck"], gr["ck"]) < rel, "dck"
        assert _rel(gg["coeff"], gr["coeff"]) < floor_c, "dcoeff"
        if norm is not None:
            assert _rel(gpu.kernel._last_dnorm, ref._last_dnorm) < rel, "dnorm"
    finally:
        ref.free()
        gpu.free()


@pytest.mark.parametrize("cfg", QUALIFYING, ids=CFG_ID.get)
@pytest.mark.parametrize("key", SLOT)
def test_the_slot_collapse_is_numerically_transparent(key, cfg):
    """Grouping identical waves is a reassociation: the numbers may not move.

    The leaf's own "dedup" is the slot collapse — several waves share one gamma
    row, so the denominator of a slot is computed once.  A wrong grouping (a
    wave folded into a slot whose row is not its own) would be repeatable, so
    the check is threefold: ON vs OFF must agree, BOTH must match the fp64
    reference, and each must carry the right gradients.  ``slot_collapse`` is
    the public knob that forces either path.
    """
    kc, params, data = _setup(cfg)
    ref = _numpy_flat(kc)
    on = _backend(KERNELS[key]["name"], kc, slot_collapse=True)
    off = _backend(KERNELS[key]["name"], kc, slot_collapse=False)
    try:
        # THE GATE IS NOT VACUOUS: the knob really selects the two paths, and on
        # `_ident`/`_cp` the collapse really folds waves (n_slot < n_wave).  A
        # backend that ignored `slot_collapse` would compare one path against
        # itself and pass the numeric half below while testing nothing.
        assert on.kernel.use_slots and not off.kernel.use_slots, (
            key, on.kernel.use_slots, off.kernel.use_slots)
        if cfg in (CFG_IDENT, CFG_CP):
            assert on.kernel.n_slot < on.kernel.n_wave, (
                key, cfg, on.kernel.n_slot, on.kernel.n_wave)
        for norm in (None, 1.3):
            Qa, ga, Pa = on.compute(params, on.load_data(data), norm=norm)
            Qb, gb, Pb = off.compute(params, off.load_data(data), norm=norm)
            assert _rel(Pa, Pb) < CACHE_FLOOR_P, "P (ON vs OFF)"
            assert _rel(Qa, Qb) < CACHE_FLOOR, "Q (ON vs OFF)"
            assert _rel(ga["ck"], gb["ck"]) < CACHE_FLOOR, "dck (ON vs OFF)"
            assert _rel(ga["coeff"], gb["coeff"]) < CACHE_FLOOR_COEFF, \
                "dcoeff (ON vs OFF)"
            if norm is not None:
                assert _rel(on.kernel._last_dnorm,
                            off.kernel._last_dnorm) < CACHE_FLOOR
            # ... and both are right, not merely equal
            Qr, gr, _Pr = ref.compute(params, ref.load_data(data), norm=norm)
            assert _rel(Qa, Qr) < CACHE_FLOOR and _rel(Qb, Qr) < CACHE_FLOOR
            assert _rel(ga["coeff"], gr["coeff"]) < CACHE_FLOOR_COEFF
            assert _rel(gb["coeff"], gr["coeff"]) < CACHE_FLOOR_COEFF
    finally:
        on.free()
        off.free()
        ref.free()


@pytest.mark.parametrize("key", HOLE)
def test_a_gamma_hole_does_not_move_a_number(key):
    """An interior zero in a gamma row is a zero WEIGHT, not a wider row.

    Arithmetically the term was always ``0 * value``, so dropping it from the
    sparse list is exact — and the answer on the holed config must still be the
    reference's.  A list that dropped a NON-zero column instead would change the
    denominator, which is why the hole is first shown to be REAL.
    """
    kc, params, data = _setup(CFG_KM)
    kc = dict(kc)
    gam = np.array(kc["gamma_matrix"], np.float64).copy()
    assert gam.shape[1] == 1
    holed = gam.copy()
    col = int(np.nonzero(holed[0, 0])[0][1])     # a NON-zero column of wave 0
    holed[0, 0, col] = 0.0
    kc["gamma_matrix"] = holed

    ref = _numpy_flat(kc)
    gpu = _backend(KERNELS[key]["name"], kc)
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=1.3)
        # the hole is real: it moved Q away from the unholed config
        kc0, _p, _d = _setup(CFG_KM)
        plain = _numpy_flat(kc0)
        try:
            Q0 = plain.compute(params, plain.load_data(data), norm=1.3)[0]
        finally:
            plain.free()
        assert abs(Qr - Q0) > 1e-6 * max(abs(Q0), 1.0)
        assert _rel(Qg, Qr) < CACHE_FLOOR, "Q"
        assert _rel(Pg, Pr) < CACHE_FLOOR_P, "P"
        assert _rel(gg["ck"], gr["ck"]) < CACHE_FLOOR, "dck"
        assert _rel(gg["coeff"], gr["coeff"]) < CACHE_FLOOR_COEFF, "dcoeff"
    finally:
        ref.free()
        gpu.free()


@pytest.mark.parametrize("rs", RSIZES)
@pytest.mark.parametrize("key,collapsed",
                         [(k, c) for k in RSIZE
                          for c in KERNELS[k]["rsize_opts"]],
                         ids=[f"{k}-{('auto' if c is None else ('on' if c else 'off'))}"
                              for k in RSIZE for c in KERNELS[k]["rsize_opts"]])
def test_the_grouped_objective_at_other_group_sizes(key, collapsed, rs):
    """The grouped log-sum, at every group size, against ``numpy_flat``.

    ``rs = 1`` is the shared anchor's; these are the sizes only the grouped path
    covers, including a group (25) larger than the 8-event tile.  The forced
    collapse variants pin both of the leaf's paths at a grouped objective.
    """
    kc, params, base = _setup(CFG_KM, 32, 7)
    data = _replicate(base, rs)
    ref = _numpy_flat(kc, rs)
    kw = {} if collapsed is None else {"slot_collapse": collapsed}
    gpu = _backend(KERNELS[key]["name"], kc, resolution_size=rs, **kw)
    try:
        if collapsed is not None:
            # a forced collapse must be the path the kernel actually took, or
            # the two "variants" are the same run
            assert gpu.kernel.use_slots is bool(collapsed), (
                rs, collapsed, gpu.kernel.use_slots)
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=1.3)
        assert _rel(Qg, Qr) <= 1e-6, f"rs={rs} Q"
        assert _rel(Pg, Pr) <= 1e-6, f"rs={rs} P"
        assert _rel(gg["ck"], gr["ck"]) <= 1e-6, f"rs={rs} dck"
        assert _rel(gg["coeff"], gr["coeff"]) <= 1e-6, f"rs={rs} dcoeff"
        assert gpu.kernel._last_dnorm == pytest.approx(ref._last_dnorm,
                                                       rel=1e-6), "dnorm"
    finally:
        ref.free()
        gpu.free()


@pytest.mark.parametrize("cfg", MULTI_BW_CFGS, ids=MULTI_BW_IDS)
@pytest.mark.parametrize("key", MULTI_BW)
def test_gradients_match_fd_on_the_multi_bw_chain(key, cfg):
    """``dcoeff`` and ``dck`` against central differences of this kernel's ``Q``.

    The shared suite runs the same FD sweeps on the ``n_res == 1`` fixtures; a
    gradient is only as right as the config it was checked on, so the multi-BW
    Product denominator (its own per-factor accumulation) gets the same
    independent check here — on each multi-BW shape, `4pi` included, where the
    blocks and the factor axis multiply each other.
    """
    kc, params, data = _setup(cfg, 48)
    gpu = _backend(KERNELS[key]["name"], kc)
    h = gpu.load_data(data)
    try:
        _Q0, g0, _ = gpu.compute(params, h, norm=None)
        coeff = np.asarray(params["coeff"], np.float64).copy()
        for i in range(coeff.size):
            up, dn = coeff.copy(), coeff.copy()
            up[i] += NH
            dn[i] -= NH
            pu, pn = dict(params), dict(params)
            pu["coeff"], pn["coeff"] = up, dn
            fd = (gpu.compute(pu, h, norm=None)[0]
                  - gpu.compute(pn, h, norm=None)[0]) / (2 * NH)
            assert np.isclose(g0["coeff"][i], fd, rtol=1e-4, atol=1e-8), (
                f"coeff row {i}: kernel {g0['coeff'][i]!r} vs FD {fd!r}")
        ck = np.asarray(params["ck"], np.complex128).copy()
        for i in range(ck.size):
            for part in (0, 1):
                d = np.zeros_like(ck)
                d[i] = NH if part == 0 else 1j * NH
                pu, pn = dict(params), dict(params)
                pu["ck"], pn["ck"] = ck + d, ck - d
                fd = (gpu.compute(pu, h, norm=None)[0]
                      - gpu.compute(pn, h, norm=None)[0]) / (2 * NH)
                # the kernel's dck is the raw Wirtinger half; the real
                # parameter convention downstream is 2·Re / -2·Im
                dres = (2.0 * np.real(g0["ck"][i]) if part == 0
                        else -2.0 * np.imag(g0["ck"][i]))
                assert np.isclose(dres, fd, rtol=1e-4, atol=1e-8), (
                    f"ck[{i}] part {part}: {dres!r} vs {fd!r}")
    finally:
        h.free()
        gpu.free()


@pytest.mark.parametrize("key", list(KERNELS))
def test_zero_coefficients_zero_the_amplitude(key):
    """All-zero basis weights ⇒ D = 0 ⇒ an empty model, reported as NaN.

    A caller that fixes every weight at zero has no model: the kernel must not
    crash, and every ``P`` must be NaN rather than a garbage number (0/0).
    """
    kc, params, data = _setup(CFG_KM, 16)
    gpu = _backend(KERNELS[key]["name"], kc)
    try:
        p = dict(params)
        p["coeff"] = np.zeros_like(np.asarray(params["coeff"]))
        _Q, _g, P = gpu.compute(p, gpu.load_data(data), norm=None)
        assert not np.any(np.isfinite(P))
    finally:
        gpu.free()


@pytest.mark.parametrize("key", [k for k in KERNELS
                                 if KERNELS[k].get("reused_handle")])
def test_a_reused_handle_sees_the_new_parameters(key):
    """The load-time cache is a function of the EVENT, never of the weights.

    Computing twice on one handle with changed parameters must equal a fresh
    handle — a stale cache would show up here, while a cache that wrongly
    captured the weights would fail the very first comparison.
    """
    kc, params, data = _setup(CFG_KM)
    gpu = _backend(KERNELS[key]["name"], kc)
    h = gpu.load_data(data)
    try:
        Q1, _g1, P1 = gpu.compute(params, h, norm=None)
        p2 = dict(params)
        p2["ck"] = np.asarray(params["ck"]) * 1.7
        Q2, g2, P2 = gpu.compute(p2, h, norm=None)
        Q3, g3, P3 = gpu.compute(p2, gpu.load_data(data), norm=None)
        assert abs(Q2 - Q3) <= 1e-6 * abs(Q3), (Q2, Q3)
        assert _rel(P2, P3) < CACHE_FLOOR
        assert _rel(g2["ck"], g3["ck"]) < CACHE_FLOOR
        # the two parameter points really are different
        assert _rel(P1, P2) > 1e-6 and Q1 != Q2
    finally:
        gpu.free()


@pytest.mark.parametrize("key", list(KERNELS))
def test_the_real_config_matches_numpy_flat(key):
    """The production config, through every leaf, against ``numpy_flat``.

    The fixtures are tiny by design; this is the shape a real fit uses (26
    waves, 62 basis rows), so a leaf that only agrees on toy configs is caught
    here.  The bound is its own (:data:`REAL_FLOOR`, measured): a production
    config's ``P`` spans a wider dynamic range than any fixture.
    """
    rel, floor_p, floor_c = REAL_FLOOR
    kc, params = _real_config()
    data = _in_grid(kc, 48, 3)
    ref = _numpy_flat(kc)
    gpu = _backend(KERNELS[key]["name"], kc)
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=1.3)
        assert _rel(Pg, Pr) < floor_p, "P"
        assert _rel(Qg, Qr) < rel, "Q"
        assert _rel(gg["ck"], gr["ck"]) < rel, "dck"
        assert _rel(gg["coeff"], gr["coeff"]) < floor_c, "dcoeff"
    finally:
        ref.free()
        gpu.free()


# ════════════════════════════════════════════════════════════════════
# 4. the refusals, as contract
# ════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("key", list(KERNELS))
def test_multiple_bws_per_chain_is_rejected_or_accepted(key):
    """An ``n_res1`` leaf refuses it and NAMES the general backend.

    ``_flat_compute`` sweeps the accept side through the registry; the message
    is only pinned where the refusal lives, because "use ``gen2``" is the whole
    value of the error.
    """
    kc, _params, _data = _setup(CFG_MB)
    assert int(np.asarray(kc["num_table"]).shape[1]) == 2
    if KERNELS[key]["multi_bw"]:
        _backend(KERNELS[key]["name"], kc).free()
        return
    with pytest.raises(ValueError, match="exactly ONE BW per chain"):
        _backend(KERNELS[key]["name"], kc)
    with pytest.raises(ValueError, match="cuda_flat_pwa_gen2"):
        _backend(KERNELS[key]["name"], kc)


@pytest.mark.parametrize("key", GAMMA01)
def test_a_weighted_gamma_is_refused(key):
    """The m-major reduction factors the basis row out — exact only for 0/1.

    A weight of 0.5 would make the factored sum wrong, so such a config is
    refused rather than silently mis-computed.
    """
    kc, _params, _data = _setup(CFG_KM)
    kc = dict(kc)
    gam = np.array(kc["gamma_matrix"], np.float64).copy()
    gam[0, 0, 1] = 0.5
    kc["gamma_matrix"] = gam
    with pytest.raises(ValueError, match="0/1"):
        _backend(KERNELS[key]["name"], kc)


@pytest.mark.parametrize("key", HOLE)
def test_an_empty_gamma_row_is_refused(key):
    """A wave whose row is all zero has no denominator (``D = 0``)."""
    kc, _params, _data = _setup(CFG_KM)
    kc = dict(kc)
    gam = np.array(kc["gamma_matrix"], np.float64).copy()
    gam[0, 0, :] = 0.0
    kc["gamma_matrix"] = gam
    with pytest.raises(ValueError, match="all zero"):
        _backend(KERNELS[key]["name"], kc)


@pytest.mark.parametrize("key", WIDE)
def test_a_config_past_the_device_limit_is_refused(key):
    """Past the shared-memory ceiling the create fails loudly, not with Q = 0."""
    _f, kc = _wide_kc(470)                     # ~176 KiB/block
    with pytest.raises((RuntimeError, ValueError)):
        _backend(KERNELS[key]["name"], kc)


@pytest.mark.parametrize("n_wave_target", [110, 200])
@pytest.mark.parametrize("key", WIDE)
def test_a_wide_config_still_computes(key, n_wave_target):
    """A config whose tile needs the device's opt-in must compute, not return 0.

    A kernel that never raises the shared-memory attribute passes every small
    fixture and then fails AT LAUNCH for a real wave count, returning Q = 0
    with a printed warning.  A SMALLER live context is created first, so the
    gate also covers the opt-in being lowered by a later create.
    """
    kc, _params, _data = _setup(CFG_KM)
    small = _backend(KERNELS[key]["name"], kc)      # alive across the wide call
    _f, wide_kc = _wide_kc(n_wave_target)
    n_wave = int(np.asarray(wide_kc["fl_table"]).shape[0])
    assert n_wave >= 108, "fixture is not wide enough to need the opt-in"
    from benchmark_flat_vs_pwa import _events
    wide_params = _wide_params(_fit(CFG_KM), wide_kc)
    be = _backend(KERNELS[key]["name"], wide_kc)
    try:
        Q, _g, P = be.compute(wide_params, be.load_data(_events(wide_kc, 2048)),
                              norm=1024.0)
    finally:
        be.free()
        small.free()
    assert np.isfinite(Q) and Q > 0.0, (
        f"n_wave={n_wave}: Q={Q!r} — the launch was rejected (shared-memory "
        f"opt-in missing?) and the error was swallowed")
    assert np.all(np.asarray(P) >= 0.0)


# ════════════════════════════════════════════════════════════════════
# 5. s0c2's fused reductions: the same numbers, from fewer launches
# ════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("cfg", [CFG_ONE, CFG_KM], ids=CFG_ID.get)
@pytest.mark.parametrize("rs", [1, 17])
@pytest.mark.parametrize("norm", [None, 1.3])
def test_the_fused_reductions_agree_with_s0c(cfg, norm, rs):
    """The fusion is a launch-structure change: not one bit may move.

    Both kernels run the same tree body (same block shape, chunk, fold order,
    one ``atomicAdd`` per feature) into disjoint output ranges, so exact
    equality is the only acceptable answer — and it is what "fewer launches,
    same result" means numerically.

    The 64-event shape is the REPRODUCIBLE one (one chunk, one gradient block):
    ``s0c``'s own cross-block atomics make it differ from ITSELF at ne 4000, so
    a larger sample could not be gated bitwise by any kernel.
    """
    kc, params, data = _setup(cfg)      # 64 events, seed 0: the reproducible shape
    ref = _backend("cuda_flat_pwa_gen_s0c", kc, resolution_size=rs)
    new = _backend("cuda_flat_pwa_gen_s0c2", kc, resolution_size=rs)
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=norm)
        Qn, gn, Pn = new.compute(params, new.load_data(data), norm=norm)
        assert Qr == Qn, f"{cfg} rs={rs} norm={norm}: Q {Qr!r} != {Qn!r}"
        assert np.array_equal(Pr, Pn), "P"
        assert np.array_equal(gr["ck"], gn["ck"]), "dck"
        assert np.array_equal(gr["coeff"], gn["coeff"]), "dcoeff"
        if norm is not None:
            assert ref.kernel._last_dnorm == new.kernel._last_dnorm, "dnorm"
        else:
            assert (ref.kernel._last_dnorm is None
                    and new.kernel._last_dnorm is None)
    finally:
        ref.free()
        new.free()


def test_the_fused_reductions_agree_on_a_wide_basis():
    """The same parity on a ``n_basis > 32`` config — the fallback's contract.

    Above 32 basis rows the fused feature reduction cannot keep every output in
    registers, so it falls back to s0c's own three-launch form.  Which path ran
    is not observable (and is deliberately not inspected): the numbers are.
    """
    kc, params, data = _setup(CFG_CP, 48, 0)
    kc, params = _pad_basis(kc, WIDE_BASIS, params)
    assert int(np.asarray(kc["basis_table"]).shape[0]) == WIDE_BASIS > 32
    ref = _backend("cuda_flat_pwa_gen_s0c", kc)
    new = _backend("cuda_flat_pwa_gen_s0c2", kc)
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        Qn, gn, Pn = new.compute(params, new.load_data(data), norm=1.3)
        assert Qr == Qn, (Qr, Qn)
        assert np.array_equal(Pr, Pn), "P"
        assert np.array_equal(gr["ck"], gn["ck"]), "dck"
        assert np.array_equal(gr["coeff"], gn["coeff"]), "dcoeff"
    finally:
        ref.free()
        new.free()


def test_a_shared_factor_is_redundant_beyond_the_wave_collapse():
    """The fixture that makes a FACTOR-level dedup non-vacuous.

    ``slot_map`` collapses whole wave blocks, so it only pays when two waves are
    copies of each other.  ``config_pwa_gen_mb.yml`` is one topology — every wave
    shares both factors, the wave collapse is complete and a factor-level key
    adds nothing (2 factor rows = 2 dots).  Once TWO topologies share a
    resonance (`config_pwa_gen_mb2.yml`: three topologies, all with the same
    `pipi`), the wave collapse leaves 3 blocks x 2 factors = 6 dots where 4
    factor rows suffice — the shared parent is computed once per topology
    instead of once per wave block.  This pins the STRUCTURE that
    ``cuda_flat_pwa_gen_fslot`` exploits; the numbers it must NOT move are
    :func:`test_the_factor_dedup_does_not_move_a_number`'s.
    """
    import numpy as np

    from tabpwa.amp_model import build_amplitude_model

    def counts(path):
        kc = build_amplitude_model(path).build_kernel_config()
        g = np.asarray(kc["gamma_matrix"], float)
        nw, nr, _nb = g.shape
        # the key is the gamma ROW, at both levels — that is what the kernel's
        # `slot_map` hashes (`rows[k].tobytes()`).  The mass column is NOT part
        # of it: it is a property of the BASIS row (`basis_m_index[m]`, one
        # entry per basis row, shared by every wave and factor), so two rows
        # with the same gamma select the same basis rows and the same column.
        blocks = len({tuple(r.tobytes() for r in g[k]) for k in range(nw)})
        rows = len({g[k, r].tobytes()
                    for k in range(nw) for r in range(nr)})
        return nw, nr, blocks, rows

    nw1, nr1, b1, f1 = counts("tests/config_pwa_gen_mb.yml")
    assert (nw1, nr1, b1, f1) == (4, 2, 1, 2)
    assert f1 == b1 * nr1                     # one topology: nothing to gain

    nw2, nr2, b2, f2 = counts("tests/config_pwa_gen_mb2.yml")
    assert (nw2, nr2, b2, f2) == (12, 2, 3, 4)
    assert f2 < b2 * nr2                      # 3 topologies sharing `pipi`
    assert (b2 * nr2) - f2 == 2               # the shared factor, once per extra topo

    # ... and the shape where blocks and the factor axis multiply each other
    nw3, nr3, b3, f3 = counts("tests/config_pwa_gen_4pi_mb.yml")
    assert (nw3, nr3, b3, f3) == (32, 2, 8, 12)
    assert f3 < b3 * nr3                      # 12 factor rows against 8 x 2 dots


#: The factor-path bound: an fp64 ASSOCIATION floor, not a cache floor.
#:
#: ``np.array_equal`` against the parent is NOT available for any second
#: kernel, and that is a property of the parent, not of this copy: phase 2 of
#: the tile splits the wave axis over ``nlane = ceil(256/n_out)`` threads and
#: each lane ``atomicAdd``s its partial into ``s_A``, so ``A`` — hence ``P``,
#: ``Q``, ``dck`` and ``dcoeff`` — depends on the warp retirement order.
#: MEASURED (``config_pwa_gen_mb2.yml``, ne 200, norm 1.3, two instances of the
#: SAME backend): Q 4 u, P 5 u, ck 2 u, coeff 4 u of spread, and on the mb2
#: grouped run at ne 64 the dck spread alone is 376 u.  The gate below is
#: therefore a floor on the ASSOCIATION, ~8 orders tighter than the family's
#: fp32 cache floor (2e-5), and it is measured, not chosen:
FSLOT_FLOOR = 1e-13


def test_the_factor_dedup_does_not_move_a_number():
    """``fslot`` vs ``gen2``: the factor path where it pays, the parent where not.

    The change is backward-only and a pure reassociation — every row of a
    factor carries that factor's own gamma row, so summing ``dQ/ds`` over a
    factor's rows before the dot product is the same number — so
    ``Q``/``P``/``dck``/``dnorm`` must not move and ``dcoeff`` may only
    re-associate.  The gate is :data:`FSLOT_FLOOR` (see the note above for why
    it cannot be ``array_equal``).

    BOTH SIDES OF THE GATE are pinned: ``_mb2`` (4 factor rows against the
    parent's 6 dots) takes the factor path, ``_mb`` (2 against 2) must NOT —
    there the CSR accumulation buys no dot and measured 4 % slower — and its
    numbers are then the parent's own path.  MEASURED (ne 64, seed 0,
    rsize 1, norm 1.3, six paired draws): mb2 Q 8.1e-16, P 2.0e-16, ck 3.9e-16,
    dcoeff 2.9e-16, against a same-kernel baseline of Q 1.0e-15 / ck 3.9e-16 /
    dcoeff 4.3e-16 on that sample; the mb pair is the parent path either way
    and matches to its own spread.
    """
    for cfg, n_fslot, n_rows, gate_on in ((CFG_MB2, 4, 24, True),
                                          (CFG_MB, 2, 8, False)):
        kc, params, data = _setup(cfg)
        new = _backend("cuda_flat_pwa_gen_fslot", kc)
        ref = _backend("cuda_flat_pwa_gen2", kc)
        try:
            assert new.kernel.n_fslot == n_fslot, (cfg, new.kernel.n_fslot)
            assert new.kernel.n_wave * new.kernel.n_res == n_rows, cfg
            assert new.kernel.parent_dots == ref.kernel.n_slot * ref.kernel.n_res
            assert new.kernel.use_fslot is gate_on, (cfg, new.kernel.use_fslot)
            # the two paths are exclusive, and the parent's own is unchanged
            assert new.kernel.use_slots is (not gate_on), cfg
            assert ref.kernel.use_slots, cfg
            for norm in (None, 1.3):
                Qn, gn, Pn = new.compute(params, new.load_data(data), norm=norm)
                Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=norm)
                assert _rel(Qn, Qr) < FSLOT_FLOOR, (cfg, norm, "Q")
                assert _rel(Pn, Pr) < FSLOT_FLOOR, (cfg, norm, "P")
                assert _rel(gn["ck"], gr["ck"]) < FSLOT_FLOOR, (cfg, norm, "dck")
                assert _rel(gn["coeff"], gr["coeff"]) < FSLOT_FLOOR, \
                    (cfg, norm, "dcoeff")
                if norm is not None:
                    assert _rel(new.kernel._last_dnorm,
                                ref.kernel._last_dnorm) < FSLOT_FLOOR, \
                        (cfg, norm, "dnorm")
        finally:
            new.free()
            ref.free()


def test_the_factor_gate_asks_for_a_real_saving():
    """The gate is STRICT: at equality the CSR accumulation buys no dot.

    The factor sweep pays one per-row reduction and saves ``parent_dots -
    n_fslot`` dot products, so ``n_fslot == parent_dots`` is the break-even
    where it can only lose (MEASURED on ``_mb``: 4 % slower than the parent).
    ``_mb`` is exactly that shape — one topology, 2 factor rows against its own
    collapsed 1 x 2 — and it is the case a ``<`` (rather than ``<=``) exists
    for; ``_mb2``/``4pi`` are the ``<`` side.
    """
    from tabpwa.cuda._flat_pwa_gen_fslot import fslot_threshold_says_use

    for n in (1, 2, 3, 4, 6, 12, 16):
        assert fslot_threshold_says_use(n, n) is False, n
        if n > 1:
            assert fslot_threshold_says_use(n - 1, n) is True, n
    assert fslot_threshold_says_use(2, 2) is False       # _mb: 2 vs 1 x 2
    assert fslot_threshold_says_use(4, 6) is True        # _mb2: 4 vs 3 x 2
    assert fslot_threshold_says_use(12, 16) is True      # 4pi: 12 vs 8 x 2


def test_the_factor_gate_is_off_at_one_resonance_per_chain():
    """``n_res == 1``: nothing NEW to group, so the PARENT's path runs verbatim.

    With one factor per chain a row IS a wave, so the factor slots are exactly
    the parent's wave slots (``n_fslot == n_slot <= n_wave``): the gate would
    buy the same dot count as the collapse the parent already does, so it is
    off and the factor buffer is never allocated — the point of the gate,
    since the CSR sweep would otherwise be pure overhead.
    """
    for cfg in (CFG_ONE, CFG_KM, CFG_IDENT):
        kc, params, data = _setup(cfg)
        be = _backend("cuda_flat_pwa_gen_fslot", kc)
        try:
            assert int(np.asarray(kc["num_table"]).shape[1]) == 1, cfg
            assert not be.kernel.fslot_gate and not be.kernel.use_fslot, cfg
            assert be.kernel.n_fslot == be.kernel.n_slot <= be.kernel.n_wave, cfg
            # ... and the numbers are still the parent's own band
            ref = _backend("cuda_flat_pwa_gen2", kc)
            try:
                Qn, gn, Pn = be.compute(params, be.load_data(data), norm=1.3)
                Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
                assert _rel(Qn, Qr) < FSLOT_FLOOR, cfg
                assert _rel(gn["coeff"], gr["coeff"]) < FSLOT_FLOOR, cfg
            finally:
                ref.free()
        finally:
            be.free()


#: On the 4pi fixture the fp32 cache floor is its own number — the family's
#: `2^-24` cache ulp (6e-8) read on a wider P dynamic range than any other
#: fixture, so `_rel` (worst absolute / global max) sits above it.  MEASURED
#: against `numpy_flat` (`_in_grid`, seed 0, ne 64/120/200, norm None/1.3, both
#: parameter points, worst of the matrix): Q 5.7e-8, P 4.4e-7, dck 2.6e-7,
#: dcoeff 2.6e-7.  A sample's numbers, not a constant: another run reads
#: 7.2e-9 / 4.9e-8 / 2e-8 / 1.9e-8 on the same fixtures.  5e-6 keeps an >=11x
#: margin on the worse of the two and is still 4x tighter than the family's
#: measured WIDE floor (2e-5).
FOURPI_REF_FLOOR = 5e-6


def _conditioned_params(kc, seed=7):
    """A second, independently drawn parameter point for the 4pi shape.

    Not a rescue from a broken point — the model-default point passes too
    (that is asserted first) — but the check that the parity does not depend
    on where the fit sits: a random ``ck`` (complex, modulus 0.5-1.5) against
    ``coeff`` 0.3-1.2, which is ``_wide_params``' COEFF half (its ``ck`` is the
    model's own, so only the coefficient half is shared).
    """
    rng = np.random.default_rng(seed)
    n_wave = int(np.asarray(kc["fl_table"]).shape[0])
    n_basis = int(np.asarray(kc["basis_table"]).shape[0])
    return {"ck": rng.uniform(0.5, 1.5, n_wave)
                  * np.exp(1j * rng.uniform(-np.pi, np.pi, n_wave)),
            "coeff": rng.uniform(0.3, 1.2, n_basis)}


def test_the_factor_dedup_on_identical_particle_blocks():
    """``n_blocks > 1`` AND ``n_res > 1``: the 4pi multi-BW shape.

    Every other multi-BW fixture is one block and every block fixture is
    ``n_res == 1``, so this is the only place the two axes multiply: 8
    identical-particle blocks x 4 waves = 32, ``n_res`` 2, and 12 distinct
    factor rows against the parent's 8 blocks x 2 = 16 dots.  A grouping that
    mixed a row's BLOCK with its factor (or sized the factor buffers by the
    block count) shows up here and nowhere else.

    Evaluated at BOTH parameter points — the model default the rest of the
    suite uses and :func:`_conditioned_params`' — and at each one the
    comparison is asserted to be MEANINGFUL (both kernels recomputed on their
    own handle first, and required reproducible to 1e-13, so the rest cannot
    be measuring launch order), then against the parent, then against
    ``numpy_flat``.
    """
    kc, params_default, data = _setup(CFG_4PI, 64)
    be = _backend("cuda_flat_pwa_gen_fslot", kc)
    ref = _backend("cuda_flat_pwa_gen2", kc)
    try:
        # the fixture really is the combination this test exists for
        assert int(np.asarray(kc["num_table"]).shape[1]) == 2
        assert be.kernel.n_wave == 32 and be.kernel.n_res == 2
        assert be.kernel.n_slot == 8, be.kernel.n_slot       # 8 CP/perm blocks
        assert be.kernel.n_fslot == 12, be.kernel.n_fslot
        assert be.kernel.use_fslot and not be.kernel.use_slots
        assert be.kernel.n_fslot < ref.kernel.n_slot * ref.kernel.n_res, (
            "the fixture must make the factor dedup non-vacuous",
            be.kernel.n_fslot, ref.kernel.n_slot * ref.kernel.n_res)
        # ... and the parent's own collapse is on for this shape, so the two
        # paths really are different paths (not a fixture where both are no-ops)
        assert ref.kernel.use_slots, "the parent must take its collapsed path"
        assert 2 * ref.kernel.n_slot <= ref.kernel.n_wave

        def check(ptag, params):
            for ne in (64, 120):
                d = _in_grid(kc, ne, 0)
                for norm in (None, 1.3):
                    # the comparison is only meaningful where the kernels repeat
                    h = be.load_data(d)
                    r0 = be.compute(params, h, norm=norm)
                    r1 = be.compute(params, h, norm=norm)
                    h.free()
                    assert _rel(r0[2], r1[2]) < FSLOT_FLOOR, \
                        (ptag, ne, norm, "fslot repeat")
                    hr = ref.load_data(d)
                    s0 = ref.compute(params, hr, norm=norm)
                    s1 = ref.compute(params, hr, norm=norm)
                    hr.free()
                    assert _rel(s0[2], s1[2]) < FSLOT_FLOOR, \
                        (ptag, ne, norm, "gen2 repeat")

                    Qn, gn, Pn = r1
                    Qr, gr, Pr = s1
                    assert _rel(Qn, Qr) < FSLOT_FLOOR, (ptag, ne, norm, "Q")
                    assert _rel(Pn, Pr) < FSLOT_FLOOR, (ptag, ne, norm, "P")
                    assert _rel(gn["ck"], gr["ck"]) < FSLOT_FLOOR, \
                        (ptag, ne, norm, "dck")
                    assert _rel(gn["coeff"], gr["coeff"]) < FSLOT_FLOOR, \
                        (ptag, ne, norm, "dcoeff")
                    if norm is not None:
                        assert _rel(be.kernel._last_dnorm,
                                    ref.kernel._last_dnorm) < FSLOT_FLOOR, \
                            (ptag, ne, norm, "dnorm")

                    # ... and both are the physics, not merely equal to each other
                    npr = _backend("numpy_flat", kc)
                    try:
                        Qq, gq, Pq = npr.compute(params, npr.load_data(d),
                                                 norm=norm)
                    finally:
                        npr.free()
                    assert _rel(Pn, Pq) < FOURPI_REF_FLOOR, \
                        (ptag, ne, norm, "P vs ref")
                    assert _rel(Qn, Qq) < FOURPI_REF_FLOOR, \
                        (ptag, ne, norm, "Q vs ref")
                    assert _rel(gn["coeff"], gq["coeff"]) < FOURPI_REF_FLOOR, \
                        (ptag, ne, norm, "dcoeff vs ref")

        check("model default", params_default)
        check("conditioned", _conditioned_params(kc))

        # the grouped objective on the same shape, at a group larger than a tile
        rep = {k: np.repeat(np.asarray(v), 10, axis=0) for k, v in data.items()}
        rep["weight"] = np.full(len(np.asarray(data["weight"])) * 10, 1.0 / 10)
        ga = _backend("cuda_flat_pwa_gen_fslot", kc, resolution_size=10)
        gb = _backend("cuda_flat_pwa_gen2", kc, resolution_size=10)
        try:
            Qa, xa, Pa = ga.compute(params_default, ga.load_data(rep), norm=1.3)
            Qb, xb, Pb = gb.compute(params_default, gb.load_data(rep), norm=1.3)
            assert _rel(Qa, Qb) < FSLOT_FLOOR, "Q (rsize 10)"
            assert _rel(Pa, Pb) < FSLOT_FLOOR, "P (rsize 10)"
            assert _rel(xa["coeff"], xb["coeff"]) < FSLOT_FLOOR, "dcoeff (rsize 10)"
            assert _rel(ga.kernel._last_dnorm,
                        gb.kernel._last_dnorm) < FSLOT_FLOOR, "dnorm (rsize 10)"
        finally:
            ga.free()
            gb.free()
    finally:
        be.free()
        ref.free()
