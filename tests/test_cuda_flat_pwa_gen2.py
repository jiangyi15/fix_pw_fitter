"""``cuda_flat_pwa_gen2`` — the four reductions of the flat ``pwa_gen`` tile.

Gates:
1. registration — ``pwa_gen`` scope only;
2. **anchor vs ``numpy_flat``** on every block/structure fixture (``_km``,
   ``_mb`` with ``n_res == 2``, ``_ident`` with 2 slots, ``_cp`` with 4) for
   ``Q``/``P``/``dck``/``dcoeff``/``dnorm`` at the float2 cache floor;
3. **both ``n_slot``-collapse paths agree**: the same config and data forced
   ON and OFF.  The builder never produces ``n_slot == n_wave`` (a chain fixes
   its basis, so its waves share a gamma row), so the "gate would refuse"
   branch is not reachable from a YAML fixture — the force flag is the honest
   way to cover it.  Reported ``n_slot`` per fixture is asserted too, so the
   real-world coverage is pinned rather than implied;
4. **``dcoeff`` against central differences** — the gate that catches a wrong
   collapse (a mis-summed ``dsum`` would still be self-consistent between the
   two paths only if both were wrong the same way; FD is the independent check);
5. ``resolution_size`` (grouped log-sum) against ``numpy_flat`` at rsize
   1..100, including change 4's saved-``A`` backward launch;
6. the ``rsize == 1`` path allocates no ``A`` buffer and the grouped one
   allocates exactly ``2·ne·n_proj`` doubles;
7. the ``n_res == 1`` specialisation really halves the tile, and a wide config
   still crosses the shared-memory opt-in.

Test data stays INSIDE both interpolation grids (outside, the tables clamp and
CPU/GPU diverge — see ``test_cuda_flat_cache_pwa_gen._in_grid``).
"""
import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import backends_for_model, create_backend
from tabpwa.cuda._flat_pwa_gen2 import (
    CUDAKernelFlatPwaGen2, slot_map, slot_threshold_says_use)

CFG_KM = "tests/config_pwa_gen_km.yml"
CFG_MB = "tests/config_pwa_gen_mb.yml"
CFG_IDENT = "tests/config_pwa_gen_ident.yml"
CFG_CP = "tests/config_pwa_gen_cp.yml"
#: the fixtures where the GATE itself decides differently: ``config_pwa_gen``
#: is a single wave, so ``2·n_slot <= n_wave`` is FALSE and the collapse is off
#: on the auto path; ``_same`` and the block fixtures take the collapsed path.
CFG_ONE = "tests/config_pwa_gen.yml"
CFG_SAME = "tests/config_pwa_gen_same.yml"
ALL_CFGS = [CFG_ONE, CFG_KM, CFG_MB, CFG_IDENT, CFG_SAME, CFG_CP]

# The GPU caches F / basis_m / fa as float2, so agreement with `numpy_flat` is
# limited by fp32 and its size tracks P's DYNAMIC RANGE, not n_basis alone.
# MEASURED (gen2 vs numpy_flat, ne=64, two seeds): `_km`/`_same` ~1e-7;
# `_ident`/`_cp` ~6e-6 (Q), ~3e-5 (dck), ~3e-5 (dcoeff) -- `_ident` spans
# 7.5e-4 .. 7.2e2, so `_rel` (worst absolute / global max) is set by the
# near-zero events.  The exactness check is ON-vs-OFF, which holds to 1e-16.
CACHE_FLOOR = 2e-5
CACHE_FLOOR_P = 3e-5
CACHE_FLOOR_COEFF = 1e-4
NH = 1e-6

#: observed on the fixtures — pinned so a change in the builder's gamma folding
#: (which would silently move the gate's real-world coverage) is visible.
EXPECTED_SLOTS = {CFG_ONE: 1, CFG_KM: 1, CFG_MB: 1, CFG_IDENT: 2,
                  CFG_SAME: 2, CFG_CP: 4}
#: what the auto gate picks for each fixture
EXPECTED_GATE = {CFG_ONE: False, CFG_KM: True, CFG_MB: True, CFG_IDENT: True,
                 CFG_SAME: True, CFG_CP: True}


def _in_grid(kc, ne, seed=0):
    """Events strictly inside BOTH interpolation grids (the base-kernel recipe)."""
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


def _replicate(data, rs, normalised=True):
    out = {k: np.repeat(np.asarray(v), rs, axis=0) for k, v in data.items()}
    if normalised:
        out["weight"] = np.full(len(np.asarray(data["weight"])) * rs, 1.0 / rs)
    return out


def _rel(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _pair(cfg, ne=64, seed=0, **kw):
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    data = _in_grid(kc, ne, seed)
    x = f.initial_values(seed=1)
    params, _resolved = f.build_params(x)
    ref = create_backend("numpy_flat", kc, model="pwa_gen",
                         resolution_size=kw.get("resolution_size", 1))
    gpu = create_backend("cuda_flat_pwa_gen2", kc, model="pwa_gen", **kw)
    return f, ref, gpu, params, data, kc


# ── gate 1: registration ─────────────────────────────────────────────

def test_registration():
    assert "cuda_flat_pwa_gen2" in set(backends_for_model("pwa_gen"))
    assert "cuda_flat_pwa_gen2" not in set(backends_for_model("pwa"))


def test_the_base_kernel_and_wrapper_are_not_reused():
    """A case-1 copy: own module, own source, own library, own symbols."""
    import tabpwa.cuda._flat_pwa_gen2 as new
    import tabpwa.cuda._flat_cache_pwa_gen as old
    assert new is not old
    assert new.CUDAKernelFlatPwaGen2 is not old.CUDAKernelFlatCachePWAGen
    f = Fitter(CFG_KM)
    f.apply_constrains()
    a = CUDAKernelFlatPwaGen2(f.kernel_config)
    b = old.CUDAKernelFlatCachePWAGen(f.kernel_config)
    try:
        # two distinct .so files, not one shared library
        assert a._lib is not b._lib
        # and the symbols really are renamed
        assert hasattr(a._lib, "cuda_compute_flat_pwa_gen2")
        assert not hasattr(a._lib, "cuda_compute_flat_cache")
    finally:
        a.free()
        b.free()


# ── gate 2: the anchor, per block/structure fixture ──────────────────

@pytest.mark.parametrize("cfg", ALL_CFGS)
@pytest.mark.parametrize("norm", [None, 1.3])
def test_anchor_matches_numpy_flat(cfg, norm):
    f, ref, gpu, params, data, kc = _pair(cfg)
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=norm)
        Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=norm)
        assert _rel(Pg, Pr) < CACHE_FLOOR_P, "P"
        assert _rel(Qg, Qr) < CACHE_FLOOR, "Q"
        assert _rel(gg["ck"], gr["ck"]) < CACHE_FLOOR, "dck"
        assert _rel(gg["coeff"], gr["coeff"]) < CACHE_FLOOR_COEFF, "dcoeff"
        if norm is not None:
            assert _rel(gpu.kernel._last_dnorm, ref._last_dnorm) < CACHE_FLOOR
    finally:
        ref.free()
        gpu.free()


@pytest.mark.parametrize("cfg", ALL_CFGS)
def test_the_observed_slot_count_is_pinned(cfg):
    """``n_slot`` and the auto gate's decision, per fixture.

    The record of the gate's real coverage: ``config_pwa_gen`` (one wave) is
    the case where the gate genuinely turns the collapse OFF, so the branch is
    reachable from a real config and not merely defensive.
    """
    f = Fitter(cfg)
    f.apply_constrains()
    gpu = CUDAKernelFlatPwaGen2(f.kernel_config)
    try:
        assert gpu.n_slot == EXPECTED_SLOTS[cfg], (
            f"{cfg}: n_slot = {gpu.n_slot}, pinned at "
            f"{EXPECTED_SLOTS[cfg]} — if the builder's gamma folding changed, "
            f"the gate's coverage changed with it")
        assert gpu.use_slots is EXPECTED_GATE[cfg], (
            f"{cfg}: the auto gate chose use_slots={gpu.use_slots}, expected "
            f"{EXPECTED_GATE[cfg]} (n_slot={gpu.n_slot}, n_wave={gpu.n_wave})")
        # the slots are a partition: every wave in exactly one, and the
        # representative is a member of its own slot
        assert len(gpu.slot_of_wave) == gpu.n_wave
        assert gpu.slot_of_wave.max() + 1 == gpu.n_slot
        for sl, rep in enumerate(gpu.slot_rep):
            assert gpu.slot_of_wave[rep] == sl
    finally:
        gpu.free()


def test_the_gate_actually_turns_off_on_a_real_config():
    """``n_slot == n_wave`` is unreachable, but ``2·n_slot > n_wave`` is not."""
    f = Fitter(CFG_ONE)
    f.apply_constrains()
    k = CUDAKernelFlatPwaGen2(f.kernel_config)
    try:
        assert k.n_wave == 1 and k.n_slot == 1
        assert not slot_threshold_says_use(k.n_slot, k.n_wave)
        assert k.use_slots is False
    finally:
        k.free()


# ── gate 3: BOTH slot-collapse paths ────────────────────────────────

@pytest.mark.parametrize("cfg", ALL_CFGS)
@pytest.mark.parametrize("norm", [None, 1.3])
def test_slot_collapse_on_and_off_agree(cfg, norm):
    """The collapse is an exact reassociation; forcing either path must match.

    The builder cannot produce ``n_slot == n_wave`` (a chain fixes its basis,
    so all its waves share a ``gamma`` row), which is why the "gate refuses"
    branch has no YAML fixture.  ``slot_collapse`` is the test hook that forces
    it instead of a contrived config that would test the wrong thing.
    """
    f, ref, _unused, params, data, kc = _pair(cfg)
    on = CUDAKernelFlatPwaGen2(kc, slot_collapse=True)
    off = CUDAKernelFlatPwaGen2(kc, slot_collapse=False)
    try:
        assert on.use_slots and not off.use_slots
        Qa, ga, Pa = on.compute(params, on.load_data(data), norm=norm)
        Qb, gb, Pb = off.compute(params, off.load_data(data), norm=norm)
        assert _rel(Pa, Pb) < CACHE_FLOOR_P, "P"
        assert _rel(Qa, Qb) < CACHE_FLOOR, "Q"
        assert _rel(ga["ck"], gb["ck"]) < CACHE_FLOOR, "dck"
        assert _rel(ga["coeff"], gb["coeff"]) < CACHE_FLOOR_COEFF, "dcoeff"
        if norm is not None:
            assert _rel(on._last_dnorm, off._last_dnorm) < CACHE_FLOOR
        # and both are right, not merely equal.  This compares against
        # `numpy_flat`, so it inherits the float2 CACHE floor -- dcoeff sums a
        # rounding per (e,k,m) term.  ON vs OFF above is the exactness check.
        Qr, gr, _Pr = ref.compute(params, ref.load_data(data), norm=norm)
        assert _rel(Qa, Qr) < CACHE_FLOOR and _rel(Qb, Qr) < CACHE_FLOOR
        assert _rel(ga["coeff"], gr["coeff"]) < CACHE_FLOOR_COEFF
        assert _rel(gb["coeff"], gr["coeff"]) < CACHE_FLOOR_COEFF
    finally:
        on.free()
        off.free()
        ref.free()


@pytest.mark.parametrize("cfg", [CFG_IDENT, CFG_CP])
def test_slot_sharing_is_non_trivial_there(cfg):
    """The fixtures that DO share slots have several waves per slot.

    At ``n_slot == n_wave`` every slot holds one wave and ``w_{e,slot,l}`` is a
    trivial product; ``_ident`` (2 slots / 4 waves) and ``_cp`` (4 / 8) exercise
    a real shared ``w``, which is where a wrong collapse would show.
    """
    f = Fitter(cfg)
    f.apply_constrains()
    gpu = CUDAKernelFlatPwaGen2(f.kernel_config, slot_collapse=True)
    try:
        assert gpu.n_slot < gpu.n_wave
        assert gpu.n_wave // gpu.n_slot >= 2
    finally:
        gpu.free()


def test_the_gate_threshold_is_the_documented_one():
    """One comparison, in one place, with the documented win condition.

    Measured ON-vs-OFF was 1.17×/1.22×/1.34× at n_slot 1/10/25 and exactly
    break-even (1.00×) at ``n_slot == n_wave``; the gate sits at that
    boundary, so it keeps every strict win and declines the no-gain region.
    """
    assert slot_threshold_says_use(1, 100)
    assert slot_threshold_says_use(50, 100)
    assert not slot_threshold_says_use(51, 100)
    assert not slot_threshold_says_use(100, 100)   # the break-even shape
    # a single-slot config is always worth collapsing, whatever n_wave
    assert slot_threshold_says_use(1, 2)
    # ... and a one-wave config never is (the real CFG_ONE case)
    assert not slot_threshold_says_use(1, 1)


def test_slot_map_is_a_partition_of_identical_gamma_rows():
    f = Fitter(CFG_CP)
    f.apply_constrains()
    gam = np.asarray(f.kernel_config["gamma_matrix"])
    sow, rep, rows, ptr, waves = slot_map(gam)
    # the representative's row IS the slot's row
    for sl in range(rep.size):
        assert np.array_equal(gam[rep[sl]].ravel(), rows[sl].ravel())
    # every wave maps to a slot whose row equals its own
    for k in range(gam.shape[0]):
        assert np.array_equal(gam[k].ravel(), rows[sow[k]].ravel())
    # CSR consistency: the list of a slot is exactly its members
    assert ptr.size == rep.size + 1
    for sl in range(rep.size):
        members = sorted(waves[ptr[sl]:ptr[sl + 1]].tolist())
        assert members == [k for k in range(gam.shape[0]) if sow[k] == sl]


# ── gate 4: finite differences (the independent check on dcoeff) ─────

@pytest.mark.parametrize("cfg", [CFG_KM, CFG_MB, CFG_CP])
def test_coeff_gradient_matches_central_differences(cfg):
    """``dcoeff`` must be ∂Q/∂coeff of THIS kernel's Q, per basis row.

    Independent of both paths: a wrong ``dsum`` or ``w`` would be caught here
    even if the ON/OFF comparison were blind to it (both could share the bug
    only by construction, which the FD pass rules out).
    """
    f, _ref, gpu, params, data, kc = _pair(cfg, ne=48)
    h = gpu.load_data(data)
    try:
        _Q0, g0, _ = gpu.compute(params, h, norm=None)
        coeff = np.asarray(params["coeff"], np.float64).copy()
        for i in range(min(coeff.size, 6)):
            up, dn = coeff.copy(), coeff.copy()
            up[i] += NH
            dn[i] -= NH
            pu, pn = dict(params), dict(params)
            pu["coeff"], pn["coeff"] = up, dn
            fd = (gpu.compute(pu, h, norm=None)[0]
                  - gpu.compute(pn, h, norm=None)[0]) / (2 * NH)
            assert np.isclose(g0["coeff"][i], fd, rtol=1e-4, atol=1e-8), (
                f"{cfg} row {i}: kernel {g0['coeff'][i]!r} vs FD {fd!r}")
    finally:
        gpu.free()


@pytest.mark.parametrize("cfg", [CFG_ONE, CFG_MB, CFG_CP])
def test_coeff_gradient_matches_fd_with_the_collapse_off(cfg):
    """Same FD check on the per-(k,l,m) path, so neither branch is untested.

    ``CFG_ONE`` is the fixture whose AUTO gate already lands here (n_wave=1);
    the others force the path to cover ``n_res == 2`` and the slot-sharing case
    on it as well.
    """
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    data = _in_grid(kc, 40)
    gpu = CUDAKernelFlatPwaGen2(kc, slot_collapse=False)
    h = gpu.load_data(data)
    try:
        assert not gpu.use_slots
        _Q0, g0, _ = gpu.compute(params, h, norm=None)
        coeff = np.asarray(params["coeff"], np.float64).copy()
        for i in range(min(coeff.size, 4)):
            up, dn = coeff.copy(), coeff.copy()
            up[i] += NH
            dn[i] -= NH
            pu, pn = dict(params), dict(params)
            pu["coeff"], pn["coeff"] = up, dn
            fd = (gpu.compute(pu, h, norm=None)[0]
                  - gpu.compute(pn, h, norm=None)[0]) / (2 * NH)
            assert np.isclose(g0["coeff"][i], fd, rtol=1e-4, atol=1e-8), (
                f"{cfg} row {i} (collapse OFF): {g0['coeff'][i]!r} vs {fd!r}")
    finally:
        gpu.free()


def test_ck_gradient_matches_fd():
    """``dck`` after the u-merge — the same independent check."""
    f, _ref, gpu, params, data, kc = _pair(CFG_MB, ne=48)
    h = gpu.load_data(data)
    try:
        ck = np.asarray(params["ck"], np.complex128).copy()
        _Q0, g0, _ = gpu.compute(params, h, norm=None)
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
                if part == 0:
                    dres = 2.0 * np.real(g0["ck"][i])
                else:
                    dres = -2.0 * np.imag(g0["ck"][i])
                assert np.isclose(dres, fd, rtol=1e-4, atol=1e-8), (
                    f"ck[{i}] part {part}: {dres!r} vs {fd!r}")
    finally:
        gpu.free()


# ── gate 5: resolution_size (change 4's grouped path) ───────────────

RSIZES = (1, 2, 4, 8, 10, 25, 100)


@pytest.mark.parametrize("rs", RSIZES)
def test_resolution_size_matches_numpy_flat(rs):
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    data = _replicate(_in_grid(kc, 32, seed=7), rs)

    ref = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=rs)
    gpu = create_backend("cuda_flat_pwa_gen2", kc, model="pwa_gen",
                         resolution_size=rs)
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=1.3)
        assert _rel(Qg, Qr) <= 1e-6, f"rs={rs} Q"
        assert _rel(Pg, Pr) <= 1e-6, f"rs={rs} P"
        assert _rel(gg["ck"], gr["ck"]) <= 1e-6, f"rs={rs} dck"
        assert _rel(gg["coeff"], gr["coeff"]) <= 1e-6, f"rs={rs} dcoeff"
        assert gpu.kernel._last_dnorm == pytest.approx(
            ref._last_dnorm, rel=1e-6), f"rs={rs} dnorm"
    finally:
        ref.free()
        gpu.free()


def test_group_larger_than_a_tile_is_one_log():
    """rs 25 > the 8-event tile: the saved-A path must not split the group."""
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    rs = 25
    data = _replicate(_in_grid(kc, 8, seed=9), rs)
    ref = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=rs)
    gpu = create_backend("cuda_flat_pwa_gen2", kc, model="pwa_gen",
                         resolution_size=rs)
    try:
        Qr, _gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        Qg, _gg, _Pg = gpu.compute(params, gpu.load_data(data), norm=1.3)
        # Fused cache (F·fa in ONE float2) rounds once where the split path
        # rounded F and fa separately: the fp32 floor shifts by ~4e-7
        # (measured fused-vs-split directly), so use the shared bound.
        assert _rel(Qg, Qr) <= CACHE_FLOOR
    finally:
        ref.free()
        gpu.free()


def test_grouped_path_skips_the_second_forward_pass():
    """Change 4: mode 2 reads A back, so its gradients match mode 0's.

    ``rsize == 1`` runs one launch; ``rsize > 1`` runs two and the second has
    no projection sum.  The numbers must be identical either way (a different
    rsize is a different objective, so the check is per-objective against the
    reference, plus a rerun stability check on one handle).
    """
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    data = _replicate(_in_grid(kc, 32, seed=7), 4)
    gpu = create_backend("cuda_flat_pwa_gen2", kc, model="pwa_gen",
                         resolution_size=4)
    h = gpu.load_data(data)
    try:
        Q1, g1, P1 = gpu.compute(params, h, norm=1.3)
        Q2, g2, P2 = gpu.compute(params, h, norm=1.3)
        assert _rel(Q1, Q2) <= 1e-12
        assert np.array_equal(P1, P2)
        assert _rel(g1["ck"], g2["ck"]) <= 1e-12
        assert _rel(g1["coeff"], g2["coeff"]) <= 1e-12
    finally:
        gpu.free()


# ── gate 6: the saved-A allocation policy ───────────────────────────

def test_rsize_one_allocates_no_A_buffer():
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    k = CUDAKernelFlatPwaGen2(kc, resolution_size=1)
    try:
        h = k.load_data(_in_grid(kc, 64))
        assert h.dA_bytes == 0
        h.free()
    finally:
        k.free()


def test_the_grouped_path_allocates_exactly_two_ne_nproj_doubles():
    from benchmark_flat_vs_pwa import tile_flat
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = dict(tile_flat(f.kernel_config, 99, 2))
    kc["num_table"] = np.ones_like(np.asarray(kc["num_table"]))
    k = CUDAKernelFlatPwaGen2(kc, resolution_size=10)
    h = None
    try:
        from benchmark_flat_vs_pwa import _events
        h = k.load_data(_events(kc, 4096))
        assert k.n_proj == 2
        assert h.dA_bytes == 4096 * 2 * 2 * 8
        assert h.dA_bytes == 4096 * 2 * 2 * 8 // 1    # 4 KiB per 1k events
    finally:
        if h is not None:
            h.free()
        k.free()


def test_the_A_buffer_is_released_and_nulled():
    """``free_data`` must give the 2 MiB back — a leak would be per handle."""
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = dict(f.kernel_config)
    k = CUDAKernelFlatPwaGen2(kc, resolution_size=10)
    try:
        h = k.load_data(_in_grid(kc, 64))
        assert h.dA_bytes > 0
        h.free()
        assert h.ptr is None
        h.free()                       # idempotent
    finally:
        k.free()


# ── gate 7: shared memory (change 2) and the opt-in ceiling ─────────

@pytest.mark.parametrize("nres", [1, 2])
def test_the_nres1_specialisation_drops_the_factor_buffer(nres):
    """At ``n_res == 1`` there is no per-factor ``s_s``: the tile is smaller."""
    cfgs = {1: CFG_KM, 2: CFG_MB}
    f = Fitter(cfgs[nres])
    f.apply_constrains()
    k = CUDAKernelFlatPwaGen2(f.kernel_config)
    try:
        assert k.n_res == nres
        # the tile the context was created with IS what gp2_layout says
        assert k.smem_bytes == k.tile_bytes()
        if nres == 1:
            # the specialisation must drop the per-factor buffers that the
            # SAME shape would allocate with a 2-factor list: s_s (2·TE·W·2)
            # and the second factor's pv slot (2·TE·W)
            te, w = 8, k.n_wave
            dropped = k.tile_bytes(nres_t=2) - k.tile_bytes(nres_t=1)
            assert dropped == (2 * te * w * 2 + 2 * te * w) * 8, dropped
        else:
            # the general path really does carry both
            assert k.tile_bytes(nres_t=1) < k.tile_bytes(nres_t=2)
    finally:
        k.free()


def test_the_tile_is_smaller_than_the_base_kernel_at_the_same_shape():
    """44.8 KiB -> 33.7 KiB at n_wave = 100 (the base kernel's own formula)."""
    from benchmark_flat_vs_pwa import tile_flat
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = dict(tile_flat(f.kernel_config, 99, 2))
    kc["num_table"] = np.ones_like(np.asarray(kc["num_table"]))
    k = CUDAKernelFlatPwaGen2(kc)
    try:
        w, p = k.n_wave, k.n_proj
        nb, nres = k.n_basis, k.n_res
        old = (8 * (nb + 2.0 * w * nres + 2.0 * w + 2.0 * p
                    + 2.0 * p + 1.0)
               + 4.0 * nb + 1.0 * 8 * w + 2.0 * 8 * w * nres) * 8
        assert k.smem_bytes < old, (k.smem_bytes, old)
        assert 32000 < k.smem_bytes < 36000, (
            f"{k.smem_bytes} B at n_wave={w} — expected ~33.7 KiB")
    finally:
        k.free()


def test_the_smem_report_matches_the_launch():
    """The size the host reports IS the one the context was created with.

    The wrapper must not recompute ``gp2_layout`` — a second copy in Python
    silently under-counted by 256 B once, and the launch then failed with
    "invalid argument" while the error macro only printed, so ``Q`` came back 0.
    """
    f = Fitter(CFG_KM)
    f.apply_constrains()
    for sc in (True, False):
        k = CUDAKernelFlatPwaGen2(f.kernel_config, slot_collapse=sc)
        try:
            assert k.smem_bytes > 0
            assert k.smem_bytes == int(
                k._lib.cuda_smem_bytes_flat_pwa_gen2(k._ctx))
        finally:
            k.free()


def test_two_live_contexts_of_different_tile_sizes_both_launch():
    """The opt-in is per FUNCTION, not per context.

    A second, SMALLER context must not lower the attribute the larger one
    needs — that made every later launch fail with "invalid argument" and return
    Q = 0 (measured before the raise-only fix).
    """
    from benchmark_flat_vs_pwa import tile_flat
    f = Fitter(CFG_KM)
    f.apply_constrains()
    big_kc = dict(tile_flat(f.kernel_config, 99, 2))
    big_kc["num_table"] = np.ones_like(np.asarray(big_kc["num_table"]))
    small_kc = f.kernel_config
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)

    big = CUDAKernelFlatPwaGen2(big_kc)
    small = CUDAKernelFlatPwaGen2(small_kc)
    try:
        assert big.smem_bytes > small.smem_bytes
        # parameters sized for the WIDE config (n_wave = 100)
        x0 = f.initial_values(seed=1)
        p0, _ = f.build_params(x0)
        ck0 = np.asarray(p0["ck"])
        n_w = big.n_wave
        n_b = int(np.asarray(big_kc["basis_table"]).shape[0])
        wide = {"ck": np.tile(ck0, (n_w + ck0.size - 1) // ck0.size)[:n_w],
                "coeff": np.random.default_rng(7).uniform(0.3, 1.2, n_b)}
        Qbig, _g, _P = big.compute(wide, big.load_data(_in_grid(big_kc, 256)),
                                   norm=128.0)
        assert np.isfinite(Qbig) and Qbig > 0.0, (
            "the wide context's launch was rejected after a narrower one was "
            "created — the opt-in was lowered")
    finally:
        big.free()
        small.free()


def test_an_over_large_tile_is_refused_not_silently_empty():
    from benchmark_flat_vs_pwa import tile_flat
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = dict(tile_flat(f.kernel_config, 470, 2))
    kc["num_table"] = np.ones_like(np.asarray(kc["num_table"]))
    with pytest.raises((RuntimeError, ValueError)):
        CUDAKernelFlatPwaGen2(kc)


@pytest.mark.parametrize("n_wave_target", [110, 200])
def test_wide_config_crosses_the_shared_memory_optin(n_wave_target):
    """A tile above 48 KiB must compute, not return 0."""
    from benchmark_flat_vs_pwa import tile_flat, _events
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = dict(tile_flat(f.kernel_config, n_wave_target, 2))
    kc["num_table"] = np.ones_like(np.asarray(kc["num_table"]))
    n_wave = int(np.asarray(kc["fl_table"]).shape[0])
    n_basis = int(np.asarray(kc["basis_table"]).shape[0])
    x = f.initial_values(seed=1)
    p0, _ = f.build_params(x)
    ck0 = np.asarray(p0["ck"])
    params = {"ck": np.tile(ck0, (n_wave + ck0.size - 1) // ck0.size)[:n_wave],
              "coeff": np.random.default_rng(7).uniform(0.3, 1.2, n_basis)}
    be = create_backend("cuda_flat_pwa_gen2", kc, model="pwa_gen")
    try:
        Q, _g, P = be.compute(params, be.load_data(_events(kc, 2048)),
                              norm=1024.0)
    finally:
        be.free()
    assert np.isfinite(Q) and Q > 0.0, (
        f"n_wave={n_wave}: Q={Q!r} — the launch was rejected (shared-memory "
        f"opt-in) and the error was swallowed")
    assert np.all(np.asarray(P) >= 0.0)


# ── contract ────────────────────────────────────────────────────────

def test_rejects_an_already_deduped_config():
    from tabpwa.pwa_gen_config import build_from_groups, dedup
    m = Fitter(CFG_KM).model
    dd = dedup(build_from_groups(m, n_interp=200))
    with pytest.raises(ValueError, match="FULL"):
        CUDAKernelFlatPwaGen2(dd)


def test_resolution_size_below_one_is_rejected():
    f = Fitter(CFG_KM)
    f.apply_constrains()
    with pytest.raises(ValueError, match="resolution_size"):
        CUDAKernelFlatPwaGen2(f.kernel_config, resolution_size=0)


def test_wrong_parameter_lengths_are_rejected():
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    k = CUDAKernelFlatPwaGen2(kc)
    h = k.load_data(_in_grid(kc, 8))
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    try:
        bad = dict(params)
        bad["ck"] = np.zeros(k.n_wave + 1, complex)
        with pytest.raises(ValueError, match="n_wave"):
            k.compute(bad, h, norm=None)
        bad = dict(params)
        bad["coeff"] = np.zeros(k.n_basis + 1)
        with pytest.raises(ValueError, match="n_basis"):
            k.compute(bad, h, norm=None)
    finally:
        k.free()
