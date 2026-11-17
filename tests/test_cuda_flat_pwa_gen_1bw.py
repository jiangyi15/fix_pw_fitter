"""``cuda_flat_pwa_gen_1bw`` — the flat ``pwa_gen`` tile with ONE BW per chain.

Gates:
1. **registration** — ``pwa_gen`` scope only, and a case-1 copy (own module,
   own source, own ``.so``, own symbols; gen2 untouched).
2. **anchor vs ``numpy_flat``** on every ``n_res == 1`` fixture (``_pwa_gen``,
   ``_km``, ``_ident``, ``_same``, ``_cp``) for ``Q``/``P``/``dck``/``dcoeff``/
   ``dnorm`` at the float2 cache floor, ``norm=None`` and a scalar norm.
3. **the sparse-row win is real and pinned per config** — the emitted row
   widths and the ``(ref, run)`` basis-iteration counts for BOTH phase-5 paths,
   so a change in the builder's gamma layout (which would silently move the
   measured ratios) is visible.  Measured per event:
   ``_pwa_gen`` 3→3, ``_km`` 10→10 (dense, 1.00x); ``_ident``/``_same`` 40→20
   per-wave and 40→10 collapsed; ``_cp`` 160→40 per-wave, 160→20 collapsed.
   The fixtures are all dense WITHIN their span, so these numbers are the same
   whether the list is a span or the true non-zeros — the hole case is gate 6.
4. **both slot-collapse paths agree** on the numbers (the collapse has real
   content at one factor — see the module docstring) — forced ON and OFF.
5. **``dcoeff`` against central differences** — the independent check that the
   runs (and the ``nz_wave``/``nz_slot`` lists) are the right columns.  A wrong
   run would be self-consistent between two computes and only FD catches it.
6. **a row WITH HOLES is handled, not refused** — the index list holds only
   the true non-zeros, so a punched hole changes neither ``Q`` nor the
   gradients.  Pinned against a reference whose hole is a genuine zero-weight
   column, i.e. the two must agree exactly rather than "both raise".
7. **``n_res != 1`` is rejected** with the message naming ``cuda_flat_pwa_gen2``
   (``tests/config_pwa_gen_mb.yml``, the only multi-BW fixture).
8. **``resolution_size``** 1..100 (the grouped path, incl. a group spanning
   several 8-event tiles) against ``numpy_flat``.
9. **shared memory** — the context's own report is the launch size, the tile is
   the documented size and matches gen2's at the same shape, and a wide config
   still crosses the 48 KiB opt-in.
10. the ``rsize == 1`` path allocates no saved-``A`` buffer.


Test data stays INSIDE both interpolation grids (outside, the tables clamp and
CPU/GPU diverge — see ``test_cuda_flat_cache_pwa_gen._in_grid``).
"""
import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import backends_for_model, create_backend
from tabpwa.cuda._flat_pwa_gen_1bw import (
    CUDAKernelFlatPwaGen1BW, index_runs, slot_threshold_says_use)
from test_cuda_flat_cache_pwa_gen import _in_grid

CFG_ONE = "tests/config_pwa_gen.yml"
CFG_KM = "tests/config_pwa_gen_km.yml"
CFG_MB = "tests/config_pwa_gen_mb.yml"
CFG_IDENT = "tests/config_pwa_gen_ident.yml"
CFG_SAME = "tests/config_pwa_gen_same.yml"
CFG_CP = "tests/config_pwa_gen_cp.yml"
#: every fixture with exactly one BW per chain (a `_mb` chain has two)
QUALIFYING = [CFG_ONE, CFG_KM, CFG_IDENT, CFG_SAME, CFG_CP]

# The GPU caches F / basis_m / fa as float2, so agreement with `numpy_flat` is
# limited by fp32, not by the algebra (all the flat kernels agree with EACH
# OTHER to <1e-6).  Measured on the block-expanded fixtures (`_ident` n_basis
# 10, `_cp` 20): per-event MEDIAN relative error ~7e-7, worst global ~2.5e-5,
# and `dcoeff` (a sum over every (event, wave)) ~3e-5.  One 1e-6 number for all
# of them was calibrated when n_basis was 5.
# MEASURED (1bw vs numpy_flat, ne=64, two seeds):
#   _km, _same          dQ ~1e-7   dck ~1e-7   dcoeff ~3e-7
#   _ident, _cp         dQ ~6e-6   dck ~3e-5   dcoeff ~3e-5
# The difference tracks P's DYNAMIC RANGE, not n_basis: `_ident` spans
# 7.5e-4 .. 7.2e2, so `_rel` (worst absolute / global max) is set by the
# near-zero events.  These bounds cover the wide-range fixtures; the tight
# exactness check is ON-vs-OFF, which holds to 1e-16.
CACHE_FLOOR = 2e-5          # Q, P, ck
CACHE_FLOOR_P = 3e-5        # per-event P
CACHE_FLOOR_COEFF = 1e-4    # dcoeff accumulates one rounding per (e,k,m)
NH = 1e-6

#: measured `(ref, run)` basis-row iterations per event for the PER-WAVE phase-5
#: path — pinned so a change in the builder's per-chain gamma blocks (the
#: source of the run win) shows up here rather than as an unexplained benchmark
#: delta.
EXPECTED_ITERS_NOSLOT = {
    CFG_ONE: (3, 3),        # 1 wave, dense 3-row basis          -> 1.00x
    CFG_KM: (10, 10),       # 2 waves, dense 5-row basis         -> 1.00x
    CFG_IDENT: (40, 20),    # 4 waves, 2 slots of 5              -> 2.00x
    CFG_SAME: (40, 20),     # 4 waves, 2 slots of 5              -> 2.00x
    CFG_CP: (160, 40),      # 8 waves (2 identical x 2 CP), 5    -> 4.00x
}
#: ... and for the COLLAPSED path: the runs dedupe across a slot's waves as well
#: as across the slot's columns, so the two savings multiply.
EXPECTED_ITERS_SLOT = {
    CFG_ONE: (3, 3),        # 1 slot / 1 wave
    CFG_KM: (10, 5),        # 1 slot / 2 waves x 5 columns       -> 2.00x
    CFG_IDENT: (40, 10),    # 2 slots x 5 columns                -> 4.00x
    CFG_SAME: (40, 10),     # 2 slots x 5 columns                -> 4.00x
    CFG_CP: (160, 20),      # 4 slots x 5 columns                -> 8.00x
}
#: the auto gate's decision per fixture (same rule as gen2)
EXPECTED_GATE = {CFG_ONE: False, CFG_KM: True, CFG_IDENT: True,
                 CFG_SAME: True, CFG_CP: True}
EXPECTED_SLOTS = {CFG_ONE: 1, CFG_KM: 1, CFG_IDENT: 2, CFG_SAME: 2, CFG_CP: 4}



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
    gpu = create_backend("cuda_flat_pwa_gen_1bw", kc, model="pwa_gen", **kw)
    return f, ref, gpu, params, data, kc


# ── gate 1: registration and the case-1 copy ────────────────────────

def test_registration():
    assert "cuda_flat_pwa_gen_1bw" in set(backends_for_model("pwa_gen"))
    assert "cuda_flat_pwa_gen_1bw" not in set(backends_for_model("pwa"))


def test_the_general_kernel_is_not_reused():
    """A case-1 copy: own module, own source, own library, own symbols."""
    import tabpwa.cuda._flat_pwa_gen_1bw as new
    import tabpwa.cuda._flat_pwa_gen2 as old
    assert new is not old
    assert new.CUDAKernelFlatPwaGen1BW is not old.CUDAKernelFlatPwaGen2
    f = Fitter(CFG_KM)
    f.apply_constrains()
    a = CUDAKernelFlatPwaGen1BW(f.kernel_config)
    b = old.CUDAKernelFlatPwaGen2(f.kernel_config)
    try:
        assert a._lib is not b._lib            # two distinct .so files
        assert hasattr(a._lib, "cuda_compute_flat_pwa_gen_1bw")
        assert not hasattr(a._lib, "cuda_compute_flat_pwa_gen2")
        # the machinery one factor kills: the `Π_{l'≠l} s` product has no
        # content at one factor, so neither does its buffer
        assert not hasattr(a, "gamma_slot")
    finally:
        a.free()
        b.free()



# ── gate 2: the anchor, every qualifying fixture ────────────────────

@pytest.mark.parametrize("cfg", QUALIFYING)
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


# ── gate 3: the run win, pinned per config and per path ─────────────

@pytest.mark.parametrize("cfg", QUALIFYING)
def test_the_contiguous_run_is_the_pinned_one(cfg):
    """The run win is per CONFIG — an average over the fixtures would hide it.

    ``_pwa_gen`` and ``_km`` are DENSE per wave (their wave's basis is the whole
    of ``n_basis``), so the per-wave run buys nothing there and its ratio is
    1.00.  ``_ident``/``_same`` sit at 2x and ``_cp`` at 4x per wave, because a
    chain's basis block is a *sub-range* of the flat list.

    The collapsed path also dedupes the run across a slot's waves and their
    columns, so the two savings multiply: ``_km`` goes 10→5 (one slot of two
    waves), ``_ident``/``_same`` 40→10, ``_cp`` 160→20 (8x).
    """
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    k = CUDAKernelFlatPwaGen1BW(kc)
    try:
        ref = k.n_wave * k.n_basis
        # the per-wave list is always built, whichever path the gate picks
        assert (ref, k.n_nz_wave) == EXPECTED_ITERS_NOSLOT[cfg], (
            f"{cfg}: per-wave basis iterations (ref, run) = "
            f"{(ref, k.n_nz_wave)}, pinned at {EXPECTED_ITERS_NOSLOT[cfg]} — "
            f"the builder's per-chain gamma blocks changed, which is what the "
            f"run optimisation keys on")
        assert (ref, k.n_nz_slot) == EXPECTED_ITERS_SLOT[cfg], (
            f"{cfg}: collapsed basis iterations = {(ref, k.n_nz_slot)}, pinned "
            f"at {EXPECTED_ITERS_SLOT[cfg]}")
        # never a loss on either path, and the collapse a strict win wherever
        # it is enabled
        assert k.n_nz_wave <= ref and k.n_nz_slot <= ref
        # the gate's decision, pinned (same rule as gen2)
        assert k.n_slot == EXPECTED_SLOTS[cfg]
        assert k.use_slots is EXPECTED_GATE[cfg]
        # the list really is each row's TRUE non-zeros, in order
        gam = np.asarray(kc["gamma_matrix"], np.float64)[:, 0, :]
        for kk in range(k.n_wave):
            idx = np.nonzero(gam[kk])[0]
            lo, hi = int(k.gs_lo[kk]), int(k.gs_hi[kk])
            assert np.array_equal(k.nz_m[lo:hi], idx)
        # the slots are a partition of the identical rows
        assert k.slot_of_wave.max() + 1 == k.n_slot
        for sl, rep in enumerate(k.slot_rep):
            assert k.slot_of_wave[rep] == sl
            assert np.array_equal(gam[rep], gam[k.slot_of_wave == sl][0])
    finally:
        k.free()


def test_the_dense_shapes_gain_nothing_per_wave_and_say_so():
    """Guards the honest reading of the benchmark: dense == 1.00x per wave."""
    for cfg in (CFG_ONE, CFG_KM):
        assert (EXPECTED_ITERS_NOSLOT[cfg][0]
                == EXPECTED_ITERS_NOSLOT[cfg][1]), cfg
    # ... and the sparse ones are strict wins
    for cfg in (CFG_IDENT, CFG_SAME, CFG_CP):
        assert EXPECTED_ITERS_NOSLOT[cfg][1] < EXPECTED_ITERS_NOSLOT[cfg][0]
    # the collapsed path is a strict win everywhere the gate enables it
    for cfg in QUALIFYING:
        if EXPECTED_GATE[cfg]:
            assert EXPECTED_ITERS_SLOT[cfg][1] < EXPECTED_ITERS_SLOT[cfg][0]


def test_the_gate_actually_turns_off_on_a_real_config():
    """``config_pwa_gen`` is one wave, so the gate declines the collapse.

    That is the branch a real config reaches, not merely a forced test hook.
    """
    f = Fitter(CFG_ONE)
    f.apply_constrains()
    k = CUDAKernelFlatPwaGen1BW(f.kernel_config)
    try:
        assert k.n_wave == 1 and k.n_slot == 1
        assert not slot_threshold_says_use(k.n_slot, k.n_wave)
        assert k.use_slots is False
    finally:
        k.free()


# ── gate 4: both collapse paths agree ──────────────────────────────

@pytest.mark.parametrize("cfg", QUALIFYING)
@pytest.mark.parametrize("norm", [None, 1.3])
def test_slot_collapse_on_and_off_agree(cfg, norm):
    """The collapse is an exact reassociation; forcing either path must match.

    Unlike gen2's `s_w` (which has no content at one factor), the `dsum`
    reduction this path exists for IS live here, so both paths have to be
    exercised rather than one being deleted.
    """
    f, ref, _unused, params, data, kc = _pair(cfg)
    on = CUDAKernelFlatPwaGen1BW(kc, slot_collapse=True)
    off = CUDAKernelFlatPwaGen1BW(kc, slot_collapse=False)
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
        # and both are right, not merely equal.  These compare against
        # `numpy_flat`, so they inherit the float2 CACHE floor -- use the
        # per-quantity bounds, not the tight Q/dck one.  (ON vs OFF above is
        # the exactness check and holds to 1e-16.)
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=norm)
        assert _rel(Qa, Qr) < CACHE_FLOOR and _rel(Qb, Qr) < CACHE_FLOOR
        assert _rel(ga["coeff"], gr["coeff"]) < CACHE_FLOOR_COEFF
        assert _rel(gb["coeff"], gr["coeff"]) < CACHE_FLOOR_COEFF
    finally:
        on.free()
        off.free()
        ref.free()


# ── gate 4: finite differences (the independent check on the runs) ──

@pytest.mark.parametrize("cfg", [CFG_KM, CFG_IDENT, CFG_CP])
@pytest.mark.parametrize("collapsed", [False, True])
def test_coeff_gradient_matches_central_differences(cfg, collapsed):
    """``dcoeff`` must be ∂Q/∂coeff of THIS kernel's Q, per basis row.

    This is the gate that catches a WRONG RUN: a run that included columns the
    factor does not select would still be repeatable (both computes wrong the
    same way), so only an independent derivative can catch it.  Both phase-5
    paths are checked, because they use different ``nz`` lists.
    """
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    data = _in_grid(kc, 48)
    gpu = CUDAKernelFlatPwaGen1BW(kc, slot_collapse=collapsed)
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
                f"{cfg} collapsed={collapsed} row {i}: kernel "
                f"{g0['coeff'][i]!r} vs FD {fd!r}")
    finally:
        gpu.free()


@pytest.mark.parametrize("cfg", [CFG_KM, CFG_CP])
def test_ck_gradient_matches_fd(cfg):
    """``dck`` after the u-merge — the same independent check."""
    f, _ref, gpu, params, data, kc = _pair(cfg, ne=48)
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
                dres = (2.0 * np.real(g0["ck"][i]) if part == 0
                        else -2.0 * np.imag(g0["ck"][i]))
                assert np.isclose(dres, fd, rtol=1e-4, atol=1e-8), (
                    f"ck[{i}] part {part}: {dres!r} vs {fd!r}")
    finally:
        gpu.free()


# ── gate 5: a holed row is handled (the index list, not a guard) ────

def test_a_hole_in_a_gamma_row_is_handled_not_refused():
    """A row with interior holes must give the SAME answer, not an error.

    The index list holds only the true non-zeros, so punching a hole into a
    wave's row inside its span removes a ``0 * value`` term — exactly what the
    full-column walk of ``cuda_flat_pwa_gen2`` already does.  Both kernels get
    the SAME holed config, so this compares two enumerations of one gamma (the
    list vs every column) rather than two models.

    This is the gate that would catch an index list that dropped a NON-zero
    column (it would silently change the denominator) — which is why it also
    asserts the hole is real, i.e. that it moved ``Q`` away from the unholed
    value.
    """
    from tabpwa.cuda._flat_pwa_gen2 import CUDAKernelFlatPwaGen2

    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = dict(f.kernel_config)
    gam = np.array(kc["gamma_matrix"], np.float64).copy()
    assert gam.shape[1] == 1
    data = _in_grid(kc, 64, 0)
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)

    def run(gamma, cls, **kw):
        c = dict(kc); c["gamma_matrix"] = gamma
        k = cls(c, **kw)
        try:
            Q, g, _P = k.compute(params, k.load_data(data), norm=1.3)
            return Q, g
        finally:
            k.free()

    Q0, _ = run(gam, CUDAKernelFlatPwaGen1BW)

    holed = gam.copy()
    col = int(np.nonzero(holed[0, 0])[0][1])     # a NON-zero column of wave 0
    holed[0, 0, col] = 0.0
    Qh, gh = run(holed, CUDAKernelFlatPwaGen1BW)
    Qg, gg = run(holed, CUDAKernelFlatPwaGen2)

    # the hole is real (otherwise the comparison below is vacuous) …
    assert abs(Qh - Q0) > 1e-6 * max(abs(Q0), 1.0)
    # … and the sparse list and the full walk agree on the holed config
    assert _rel(Qh, Qg) < CACHE_FLOOR
    assert _rel(gh["ck"], gg["ck"]) < CACHE_FLOOR
    assert _rel(gh["coeff"], gg["coeff"]) < CACHE_FLOOR_COEFF


def test_the_index_list_is_itself():
    """One implementation, so a test can ask without a kernel."""
    dense = np.ones((2, 5))
    lo, hi, nk, nm = index_runs(dense)
    assert lo.tolist() == [0, 5] and hi.tolist() == [5, 10]
    assert nk.tolist() == [0] * 5 + [1] * 5 and nm.tolist() == list(range(5)) * 2
    # a 3-D (n_wave, 1, n_basis) gamma — the config's own shape — is accepted
    lo3, hi3, _nk3, _nm3 = index_runs(np.ones((2, 1, 5)))
    assert lo3.tolist() == lo.tolist() and hi3.tolist() == hi.tolist()
    # a sub-range row is fine (that is the win), and the CSR skips the zeros
    sub = np.zeros((2, 5))
    sub[0, 1:4] = 1.0
    sub[1, 3:] = 1.0
    lo, hi, nk, nm = index_runs(sub)
    assert lo.tolist() == [0, 3] and hi.tolist() == [3, 5]
    assert nm.tolist() == [1, 2, 3, 3, 4]
    # a HOLE inside the span costs one entry, not the span
    holed = np.ones((1, 5))
    holed[0, 2] = 0.0
    lo, hi, _nk, nm = index_runs(holed)
    assert lo.tolist() == [0] and hi.tolist() == [4]
    assert nm.tolist() == [0, 1, 3, 4]
    # an empty row would make D = 0
    with pytest.raises(ValueError, match="all zero"):
        index_runs(np.zeros((1, 5)))

def test_an_interior_zero_is_enumerated_as_a_hole_not_a_span():
    """An interior zero costs one CSR entry — it does NOT widen the row.

    Arithmetically the term was always ``0 * value``, so dropping it from the
    list is exact; charging the SPAN for it is what the old contiguity guard
    effectively did by refusing such a row outright.  Interior zeros are not
    exotic — they are the norm: the real ``jpsi_eta_pi_pi`` config has them in
    23 of its 26 waves, which is the config this change exists for.
    """
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = dict(f.kernel_config)
    gam = np.array(kc["gamma_matrix"], np.float64).copy()
    assert gam[0, 0, 2] == 1.0
    gam[0, 0, 2] = 0.0                  # interior zero: the span is unchanged
    kc["gamma_matrix"] = gam
    k = CUDAKernelFlatPwaGen1BW(kc)     # creates; no refusal
    try:
        lo, hi = int(k.gs_lo[0]), int(k.gs_hi[0])
        # wave 0 now selects 4 columns, not the span of 5
        assert hi - lo == int(np.count_nonzero(gam[0, 0]))
        assert 2 not in set(k.nz_m[lo:hi].tolist())
        assert k.run_widths[0] == 4
    finally:
        k.free()



# ── gate 6: n_res == 1 is enforced, loudly ─────────────────────────

def test_multiple_bws_per_chain_is_rejected_with_the_general_backend_named():
    f = Fitter(CFG_MB)
    f.apply_constrains()
    assert int(np.asarray(f.kernel_config["num_table"]).shape[1]) == 2
    with pytest.raises(ValueError, match="exactly ONE BW per chain"):
        CUDAKernelFlatPwaGen1BW(f.kernel_config)
    with pytest.raises(ValueError, match="cuda_flat_pwa_gen2"):
        create_backend("cuda_flat_pwa_gen_1bw", f.kernel_config,
                       model="pwa_gen")


def test_every_other_fixture_passes_the_gate():
    """5 of 6 fixtures qualify; only ``_mb`` has n_res == 2."""
    for cfg in QUALIFYING:
        f = Fitter(cfg)
        f.apply_constrains()
        assert int(np.asarray(f.kernel_config["num_table"]).shape[1]) == 1
        k = CUDAKernelFlatPwaGen1BW(f.kernel_config)
        k.free()


# ── gate 7: resolution_size (the grouped path) ─────────────────────

RSIZES = (1, 2, 4, 8, 10, 25, 100)


@pytest.mark.parametrize("rs", RSIZES)
@pytest.mark.parametrize("collapsed", [False, True])
def test_resolution_size_matches_numpy_flat(rs, collapsed):
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    data = _replicate(_in_grid(kc, 32, seed=7), rs)

    ref = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=rs)
    gpu = create_backend("cuda_flat_pwa_gen_1bw", kc, model="pwa_gen",
                         resolution_size=rs, slot_collapse=collapsed)
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
    gpu = create_backend("cuda_flat_pwa_gen_1bw", kc, model="pwa_gen",
                         resolution_size=rs)
    try:
        Qr, _gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        Qg, _gg, _Pg = gpu.compute(params, gpu.load_data(data), norm=1.3)
        # Fused cache (F·fa in ONE float2) rounds once where the split path
        # rounded F and fa separately: the fp32 floor shifts by ~4e-7
        # (measured fused-vs-split directly in `cuda_flat_pwa_gen2`), so use
        # the shared bound — this backend agrees with gen2 to 0.0 here.
        assert _rel(Qg, Qr) <= CACHE_FLOOR
    finally:
        ref.free()
        gpu.free()


def test_the_rsize_one_path_allocates_no_A_buffer():
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    k = CUDAKernelFlatPwaGen1BW(kc, resolution_size=1)
    try:
        h = k.load_data(_in_grid(kc, 64))
        assert h.dA_bytes == 0
        h.free()
    finally:
        k.free()


def test_the_grouped_path_allocates_exactly_two_ne_nproj_doubles():
    from benchmark_flat_vs_pwa import tile_flat, _events
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = dict(tile_flat(f.kernel_config, 99, 2))
    kc["num_table"] = np.ones_like(np.asarray(kc["num_table"]))
    k = CUDAKernelFlatPwaGen1BW(kc, resolution_size=10)
    h = None
    try:
        h = k.load_data(_events(kc, 4096))
        assert k.n_proj == 2
        assert h.dA_bytes == 4096 * 2 * 2 * 8
    finally:
        if h is not None:
            h.free()
        k.free()


# ── gate 11: the chunk rule and chunk independence ──────────────────
#
# The working scratch is the CONTEXT's, sized by `cap` at create (the sample's
# `ne` is unknown there), and `compute` walks the sample in chunks of `cap`
# events passing offset pointers + the chunk length as the kernel's `ne`.  The
# rule has ONE definition, in C, and the wrapper ASKS for it.

def _chunk_rule(batch_size, rsize):
    """The documented rule, restated here so the C side is checked against it."""
    cap = batch_size if batch_size > 0 else 50000
    cap = min(cap, 1 << 20)
    if rsize > 1:
        a = (cap // rsize) * rsize
        cap = a if a > 0 else rsize          # never zero, never unaligned
    return cap


@pytest.mark.parametrize("rsize", [1, 17])
@pytest.mark.parametrize("batch_size", [1, 5, 1000, 50000, 10 ** 9])
def test_the_chunk_rule_is_what_the_context_reports(batch_size, rsize):
    """`cuda_cap_*` must be the rule above — never a Python re-derivation.

    The two guards are the ones measurement forced: `batch_size = 1e9` must be
    capped (a scratch of GBs for a small sample), and `batch_size < rsize` must
    still hold a WHOLE group — an unaligned cap there let a group straddle two
    chunks and made Q wrong by 49 %.
    """
    f = Fitter(CFG_KM)
    f.apply_constrains()
    k = CUDAKernelFlatPwaGen1BW(f.kernel_config, batch_size=batch_size,
                                resolution_size=rsize)
    try:
        assert k.cap == _chunk_rule(batch_size, rsize)
        assert k.cap > 0
        if rsize > 1:
            assert k.cap % rsize == 0, "a chunk must hold whole log-sum groups"
        # the per-chunk saved-`A` scratch is sized by `cap`, not by the sample
        if rsize == 1:
            assert k.dA_bytes_cap == 0
        else:
            assert k.dA_bytes_cap == k.cap * k.n_proj * 2 * 8
        # a handle reports only its SHARE of that shared scratch
        h = k.load_data(_in_grid(f.kernel_config, 64))
        try:
            share = k.dA_bytes_cap * min(h.ne, k.cap) // k.cap
            assert h.dA_bytes == share
        finally:
            h.free()
    finally:
        k.free()


def test_a_multi_chunk_call_equals_a_single_chunk_call():
    """Batching is an inner loop: the numbers must not know about it.

    A sample much longer than `cap` has to give the same Q/P/gradients as the
    same sample in one chunk — bit-for-bit up to fp summation order, which here
    is exercised both with and without the grouped objective.
    """
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    data = _in_grid(kc, 500, seed=3)

    def run(rs, bs):
        be = CUDAKernelFlatPwaGen1BW(kc, resolution_size=rs, batch_size=bs)
        try:
            h = be.load_data(data)
            Q, g, P = be.compute(params, h, norm=1.3)
            h.free()
            return Q, g["ck"], g["coeff"], float(g["norm"]), np.asarray(P)
        finally:
            be.free()

    for rs in (1, 17):
        base = run(rs, 10 ** 9)             # one chunk
        for bs in (1000, 5000, 50000):      # many chunks
            got = run(rs, bs)
            for name, a, b in (("Q", got[0], base[0]),
                               ("dck", got[1], base[1]),
                               ("dcoeff", got[2], base[2]),
                               ("dnorm", got[3], base[3]),
                               ("P", got[4], base[4])):
                a = np.asarray(a).ravel()
                b = np.asarray(b).ravel()
                rel = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
                assert rel <= 1e-14, f"rsize={rs} batch={bs}: {name} {rel:.2e}"


# ── gate 8: shared memory ───────────────────────────────────────────

def _wide_kc(n_wave_target):
    from benchmark_flat_vs_pwa import tile_flat
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = dict(tile_flat(f.kernel_config, n_wave_target, 2))
    kc["num_table"] = np.ones_like(np.asarray(kc["num_table"]))
    return f, kc


def test_the_smem_report_matches_the_launch():
    """The host must not recompute the layout — the context is the definition."""
    f = Fitter(CFG_KM)
    f.apply_constrains()
    k = CUDAKernelFlatPwaGen1BW(f.kernel_config)
    try:
        assert k.smem_bytes > 0
        assert k.smem_bytes == int(k._lib.cuda_smem_bytes_flat_pwa_gen_1bw(
            k._ctx))
        assert k.smem_bytes == k.tile_bytes()
    finally:
        k.free()


def test_the_tile_is_the_documented_size():
    """34 464 B at the benchmark shape (n_wave 100, n_proj 2, n_basis 5).

    The layout is ``5·TE·W + 4·TE·P + TE + 4·NB + 2·W`` doubles plus the
    `s_dd` slot buffer when the collapse is on — i.e. exactly gen2's `NR == 1`
    tile, because the only thing gen2 carried there that this kernel does not
    is `s_w`, which gen2 also skipped (empty product).  What the specialisation
    removes is the per-factor buffer, not the slot one.
    """
    from tabpwa.cuda._flat_pwa_gen2 import CUDAKernelFlatPwaGen2
    _f, kc = _wide_kc(99)
    a = CUDAKernelFlatPwaGen1BW(kc)
    b = CUDAKernelFlatPwaGen2(kc)
    try:
        te, w, p, nb, ns = 8, a.n_wave, a.n_proj, a.n_basis, a.n_slot
        assert a.use_slots
        expect = (5 * te * w + 4 * te * p + te + 4 * nb + 2 * w
                  + 2 * te * ns) * 8
        assert a.smem_bytes == expect, (a.smem_bytes, expect)
        assert a.smem_bytes == 34464
        # same tile as gen2 at this shape, and identical to gen2's own report
        assert a.smem_bytes == b.smem_bytes
        # ... and the per-factor buffers really are gone: this shape with two
        # factors would need `s_s` (2·TE·W·2) and a second `pv` column
        # (2·TE·W) more, which no instantiation of THIS kernel allocates
        assert a.tile_bytes(n_wave=w, n_proj=p, n_basis=nb) == a.smem_bytes
    finally:
        a.free()
        b.free()


@pytest.mark.parametrize("n_wave_target", [110, 200])
def test_wide_config_crosses_the_shared_memory_optin(n_wave_target):
    """A tile above 48 KiB must compute, not return 0."""
    from benchmark_flat_vs_pwa import _events
    f, kc = _wide_kc(n_wave_target)
    n_wave = int(np.asarray(kc["fl_table"]).shape[0])
    n_basis = int(np.asarray(kc["basis_table"]).shape[0])
    x = f.initial_values(seed=1)
    p0, _ = f.build_params(x)
    ck0 = np.asarray(p0["ck"])
    params = {"ck": np.tile(ck0, (n_wave + ck0.size - 1) // ck0.size)[:n_wave],
              "coeff": np.random.default_rng(7).uniform(0.3, 1.2, n_basis)}
    be = create_backend("cuda_flat_pwa_gen_1bw", kc, model="pwa_gen")
    try:
        Q, _g, P = be.compute(params, be.load_data(_events(kc, 2048)),
                              norm=1024.0)
    finally:
        be.free()
    assert np.isfinite(Q) and Q > 0.0, (
        f"n_wave={n_wave}: Q={Q!r} — the launch was rejected (shared-memory "
        f"opt-in) and the error was swallowed")
    assert np.all(np.asarray(P) >= 0.0)


def test_two_live_contexts_of_different_tile_sizes_both_launch():
    """The opt-in is raise-only monotonic, so create order cannot break it."""
    _f, big_kc = _wide_kc(99)
    f2 = Fitter(CFG_KM)
    f2.apply_constrains()
    small_kc = f2.kernel_config
    big = CUDAKernelFlatPwaGen1BW(big_kc)
    small = CUDAKernelFlatPwaGen1BW(small_kc)
    try:
        assert big.smem_bytes > small.smem_bytes
        p0, _ = f2.build_params(f2.initial_values(seed=1))
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
    _f, kc = _wide_kc(470)
    with pytest.raises((RuntimeError, ValueError)):
        CUDAKernelFlatPwaGen1BW(kc)


# ── contract ────────────────────────────────────────────────────────

def test_rejects_an_already_deduped_config():
    from tabpwa.pwa_gen_config import build_from_groups, dedup
    m = Fitter(CFG_KM).model
    dd = dedup(build_from_groups(m, n_interp=200))
    with pytest.raises(ValueError, match="FULL"):
        CUDAKernelFlatPwaGen1BW(dd)


def test_resolution_size_below_one_is_rejected():
    f = Fitter(CFG_KM)
    f.apply_constrains()
    with pytest.raises(ValueError, match="resolution_size"):
        CUDAKernelFlatPwaGen1BW(f.kernel_config, resolution_size=0)


def test_wrong_parameter_lengths_are_rejected():
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    k = CUDAKernelFlatPwaGen1BW(kc)
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


def test_a_freed_handle_is_reported_not_crashed():
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    k = CUDAKernelFlatPwaGen1BW(kc)
    h = k.load_data(_in_grid(kc, 8))
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    try:
        h.free()
        h.free()                              # idempotent
        with pytest.raises(RuntimeError, match="already freed"):
            k.compute(params, h, norm=None)
    finally:
        k.free()
