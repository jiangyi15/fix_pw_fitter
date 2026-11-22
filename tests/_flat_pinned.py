"""The contract suite every PINNED-coefficient flat ``pwa_gen`` kernel shares.

``coeff_pinned`` kernels (``tests/_flat_compute.py`` subtracts the tag) refuse
the free-coefficient contract that suite exists to test: every coefficient is
declared fixed, so the whole per-wave amplitude is a load-time cache,
``grads["coeff"] == 0`` for every column by contract, and a ``coeff`` that
moved after the cache was built RAISES.  That is a CONTRACT of its own, and it
is dtype-independent — the fp64 ``ampcache`` and its fp32 contraction sibling
must satisfy exactly the same gates.  This module writes those gates ONCE.

The backend list is DERIVED from the registry (``backends_for_model("pwa_gen",
tag="coeff_pinned")``) and parametrised over it; what genuinely differs between
the variants lives in ``TABLES``, one entry per backend:

    floors / mb_floors   parity vs ``numpy_flat``, the one-BW set and ``_mb``
                         (``n_res == 2``) SEPARATELY — copied verbatim from each
                         variant's own file, never a family maximum
    fd                   the central-difference step and its tolerance.  Not
                         shared: the fp32 ``Q`` carries a fixed contraction
                         offset that varies with ``ck``, so at the fp64 step of
                         ``1e-6`` the check condemns a gradient that is right to
                         ``1e-7`` — the fp32 file measured the step at ``1e-2``
    stable / fresh       the repeat-call and fresh-handle bounds each file used
    make_kernel          the kernel class, for the handle-lifecycle tests (they
                         drive `load_data`/`free` directly, not through a
                         backend)

The case-1 gate (own module/``.so``/symbols, one library per process, the
parent source byte-identical) is NOT written here: it is the ONE shared table
``tests/_flat_isolation.py``, selected there and collected by
``test_flat_isolation.py``.  This file only asserts that every pinned kernel
HAS a row in it.

Local, because it is dtype- or kernel-specific and not shareable: the build-time
helper copies (``uslot_map`` / ``amp_slot_map`` / ``den_runs`` — the parent's
denominator builder is not the same callable; the ``np.array_equal`` pin against
those originals is ``tests/_flat_isolation.py``'s ``HELPER_COPIES``), the fp32
second-rounding measurement against its fp64 parent, the Gram-vs-parent
reduction, and the s0f all-fixed cross-check and its end-to-end ``Fitter`` runs.

NOT tested here, deliberately — each is HOW a kernel works, not what it returns:
``te``, the block/tiling plan, the CSR's internal layout, the kernel's attribute
surface, and register/spill counts.  A tile size that changed the arithmetic is
caught by the numeric gates at every ``resolution_size``; a re-tiling or a CSR
repack that keeps the numbers is not a defect.  The speed side of those levers
is what ``tests/benchmark_ampcache*.py`` prints.
"""
import importlib
import os

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import backend_tags, backends_for_model, create_backend
from tabpwa.cuda._flat_pwa_gen_ampcache import (
    CUDAKernelFlatPwaGenAmpCache)
from tabpwa.cuda._flat_pwa_gen_ampcache32 import (
    CUDAKernelFlatPwaGenAmpCache32)
from _flat_compute import CONFIG_OWNER, TAG_VOCABULARY, _rel, _replicate
from _flat_grid import CFG_CP, CFG_IDENT, CFG_KM, CFG_MB, CFG_ONE, _in_grid
#: the ONE case-1 table (module/symbols/parent + every source digest).  It
#: covers the pinned kernels, the fixed-coefficient ones and the tile family,
#: so the property is written once — ``test_flat_isolation.py`` runs it.
from _flat_isolation import ISOLATION, module_of

#: the ONE-BW fixtures the pinned kernels' floors were measured on.  NOT
#: ``_flat_grid.QUALIFYING``: that set also carries ``_same``, on which no
#: pinned anchor was measured (a wider set would be a NEW gate, not a move).
ONE_BW = (CFG_ONE, CFG_KM, CFG_IDENT, CFG_CP)
#: the parity matrix: the one-BW fixtures AND the multi-BW one.  The union of
#: the two files' `PARITY` sets (the fp64 file's omitted `_one`; its floors were
#: MEASURED over `ONE_BW`, which includes it, so gating it is not new).
PARITY = ONE_BW + (CFG_MB,)

#: the grouped-objective scenario (34 events, seed 5, rsize 17, un-normalised
#: weights) — MEASURED 1.4e-7 on P and 5.9e-8 on Q; the family's grouped floor
GROUPED = 5e-6
#: the GRAM's floor is the `float2` CACHE's, not the algebra's: MEASURED
#: ``max|ΔD|/max|D|`` 7.2e-8 (km, n_wave 2) and 2.5e-7 (mb, n_wave 4, n_res 2)
#: at 200 events, seed 0; the bound carries a ~4x margin.  Both variants reduce
#: the SAME fp64 products from the same cache, so this is not per backend.
GRAM_FLOOR = 1e-6
#: ``D`` is built from the same product on both triangles; only the cross-block
#: atomic order separates them, MEASURED max|D − D†|/max|D| = 1.9e-16.
GRAM_HERM = 1e-12
#: ``Re(ck† D ck)`` (the quantity a fit consumes) against the oracle and
#: against the base's own ``Σ w P``: MEASURED 1.4e-8 (km, vs numpy) / 1.8e-8
#: (km, vs s0e2) / 2.0e-7 (mb) -> the family's fp32 floor.
GRAM_NORM = 5e-6

#: Per-backend facts.  Every number is COPIED from that variant's own fixture
#: (HEAD), so the consolidation cannot widen a gate; where the two files gated
#: the same quantity at different bounds, each entry keeps its own — the `_mb`
#: floors genuinely differ (the fp32 contraction is the extra rounding there).
#:
#:   floors / mb_floors -> (Q, P, dck, dnorm) relative bounds
TABLES = {
    "cuda_flat_pwa_gen_ampcache": {
        # MEASURED over ONE_BW (64 events, seed 0, rsize 17 replication,
        # norm None and 1.3): Q 9.1e-7, P 3.3e-6, dck 1.3e-6, dnorm 2.9e-7;
        # the bounds carry a ~3x margin.
        "make_kernel": CUDAKernelFlatPwaGenAmpCache,
        "floors": (3e-6, 1e-5, 5e-6, 1e-6),
        # `_mb` (n_res == 2) MEASURED SEPARATELY: Q 6.1e-7, P 1.0e-7,
        # dck 1.2e-7, dnorm 3.1e-9 (bounds ~3-5x).
        "mb_floors": (2e-6, 5e-7, 5e-7, 1e-8),
        # (step, rtol, atol) — the fp64 kernel's own central difference
        "fd": (1e-6, 1e-4, 1e-8),
        "stable": 1e-13,
        "fresh": 1e-14,
    },
    "cuda_flat_pwa_gen_ampcache32": {
        # MEASURED against fp64 `numpy_flat`: Q 8.9e-7, P 3.3e-6, dck 1.3e-6,
        # dnorm 2.9e-7 — the parent's own fp32-CACHE floors to within a few
        # percent (the `float2` cache dominates; the fp32 contraction adds
        # nothing visible here).
        "make_kernel": CUDAKernelFlatPwaGenAmpCache32,
        "floors": (3e-6, 1e-5, 5e-6, 1e-6),
        # `_mb` MEASURED SEPARATELY: Q 1.7e-6, P 5.6e-8, dck 1.5e-7,
        # dnorm 2.3e-9 (bounds ~3-4x).  Its Q floor is the LOOSER of the two
        # files' — the second rounding is real on the factor product, so
        # borrowing the fp64 2e-6 here would be a new, unmet gate.
        "mb_floors": (5e-6, 2e-7, 5e-7, 1e-8),
        # (step, rtol, atol) — NOT the parent's 1e-6: this kernel's `Q` carries
        # a fixed fp32-contraction offset that varies with `ck`, so at 1e-6 the
        # central difference divides that offset by 2e-6 and condemns a
        # gradient right to 1e-7.  MEASURED FD error 1.1e-1 (1e-6), 8.7e-4
        # (1e-4), 1.2e-4 (1e-3), 5.4e-6 (1e-2) -> step 1e-2, bound 1e-3.
        "fd": (1e-2, 1e-3, 1e-4),
        "stable": 1e-14,
        "fresh": 1e-14,
    },
}

#: the DERIVED list — never a name written out here
PINNED = sorted(backends_for_model("pwa_gen", tag="coeff_pinned"))
SPECS = PINNED

assert SPECS, (
    "no `coeff_pinned` backend is registered — the tag is the selector, so an "
    "empty set means this suite silently tests nothing")
assert set(TABLES) == set(SPECS), (
    "every coeff_pinned backend needs its OWN table entry (a register-only "
    "addition must fail here, not inherit another kernel's floors): missing "
    f"{sorted(set(SPECS) - set(TABLES))}, stale "
    f"{sorted(set(TABLES) - set(SPECS))}")
assert set(SPECS) <= set(ISOLATION), (
    "every coeff_pinned backend needs a case-1 row (the shared "
    f"_flat_isolation table): missing {sorted(set(SPECS) - set(ISOLATION))}")

#: the OWNERSHIP rule in ``_flat_compute``: this module owns the config contract
#: of every ``coeff_pinned`` leaf.  A misdirected owner — one of the two moved to
#: another family file — fails here instead of being covered by nothing.
_ME = os.path.basename(__file__)
_OWNED = {n for n, f in CONFIG_OWNER.items() if f == _ME}
assert _OWNED == set(SPECS), (
    f"{_ME}'s pinned backends are out of sync with "
    f"_flat_compute.CONFIG_OWNER: missing {sorted(set(SPECS) - _OWNED)}, "
    f"stale {sorted(_OWNED - set(SPECS))}")


def _params(f):
    return f.build_params(f.initial_values(seed=1))[0]


def _pair(name, cfg, rs=1, **kw):
    """``(fitter, kc, numpy_flat, pinned)`` on one config and one rsize.

    No ``coeff=``: the cache is filled on a handle's FIRST ``compute`` (the one
    lifecycle), so nothing has to be declared up front.
    """
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    ref = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=rs)
    gpu = create_backend(name, kc, model="pwa_gen", resolution_size=rs, **kw)
    return f, kc, ref, gpu


# ── the selector ───────────────────────────────────────────────────────

@pytest.mark.parametrize("name", SPECS)
def test_registration_and_the_pinned_tag(name):
    assert name in set(backends_for_model("pwa_gen"))
    assert name not in set(backends_for_model("pwa"))
    assert backend_tags(name, "pwa_gen") == frozenset({"coeff_pinned"})


def test_the_tag_is_declared_metadata_and_the_set_is_not_empty():
    """``coeff_pinned`` is DECLARED at registration — this suite selects on it."""
    reg = set(backends_for_model("pwa_gen"))
    assert "coeff_pinned" in TAG_VOCABULARY
    assert SPECS and set(SPECS) <= reg
    assert not set(SPECS) & set(backends_for_model("pwa_gen",
                                                   tag="reference"))
    assert not set(SPECS) & set(backends_for_model("pwa_gen", tag="delegate"))
    assert not set(SPECS) & set(backends_for_model("pwa_gen", tag="n_res1"))
    unknown = {t for n in reg for t in backend_tags(n, "pwa_gen")} \
        - TAG_VOCABULARY
    assert not unknown, f"undeclared tags in the registry: {sorted(unknown)}"
    # every entry is a complete table, so a later variant cannot half-register
    for name in SPECS:
        missing = {"make_kernel", "floors", "mb_floors", "fd", "stable",
                   "fresh"} - set(TABLES[name])
        assert not missing, (name, sorted(missing))
        assert "parent" in ISOLATION[name], name


# ── the pinned kernel's own shape (case 1 is `_flat_isolation`) ─────────

@pytest.mark.parametrize("norm", [None, 1.3])
@pytest.mark.parametrize("rs", [1, 17])
@pytest.mark.parametrize("cfg", PARITY)
@pytest.mark.parametrize("name", SPECS)
def test_matches_numpy_flat_at_the_measured_floor(name, cfg, rs, norm):
    """Q / P / dck / dnorm against fp64 ``numpy_flat`` at the MEASURED floor.

    ``_mb`` carries TWO factors per wave, so its cache is filled through the
    factor product; its own floor (measured on that fixture) applies there.
    """
    qf, pf, cf, df = TABLES[name]["mb_floors" if cfg == CFG_MB else "floors"]
    f, kc, ref, gpu = _pair(name, cfg, rs=rs)
    params = _params(f)
    base = _in_grid(kc, 64, seed=0)
    data = _replicate(base, rs) if rs > 1 else base
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=norm)
        Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=norm)
        assert _rel(Pg, Pr) < pf, f"P {_rel(Pg, Pr):.2e}"
        assert _rel(Qg, Qr) < qf, f"Q {_rel(Qg, Qr):.2e}"
        assert _rel(gg["ck"], gr["ck"]) < cf, (
            f"dck {_rel(gg['ck'], gr['ck']):.2e}")
        if norm is not None:
            assert _rel(gpu.kernel._last_dnorm, ref._last_dnorm) < df, (
                f"dnorm {_rel(gpu.kernel._last_dnorm, ref._last_dnorm):.2e}")
    finally:
        ref.free()
        gpu.free()


# ── the pinned contract ────────────────────────────────────────────────

@pytest.mark.parametrize("cfg", [CFG_KM, CFG_MB])
@pytest.mark.parametrize("rs", [1, 17])
@pytest.mark.parametrize("name", SPECS)
def test_the_coefficient_gradient_is_exactly_zero(name, cfg, rs):
    """``grads["coeff"]`` is 0.0 for EVERY column — by contract, not by luck.

    A pinned coefficient has no direction in ``x``: the cache holds no ``D``
    and there is no coefficient pass.  Not "small", not ``-0.0``: exactly 0.
    Gated on the multi-BW fixture too (two CSR rows per wave, one product,
    still no coefficient gradient).
    """
    f, kc, ref, gpu = _pair(name, cfg, rs=rs)
    params = _params(f)
    base = _in_grid(kc, 48, seed=2)
    data = _replicate(base, rs) if rs > 1 else base
    try:
        _Q, g, _P = gpu.compute(params, gpu.load_data(data), norm=1.3)
        c = np.asarray(g["coeff"])
        assert c.shape == (gpu.kernel.n_basis,)
        assert np.array_equal(c, np.zeros(gpu.kernel.n_basis)), (
            "the coefficient gradient must be exactly zero")
        assert not np.signbit(c).any(), "a -0.0 is still not a direction"
        # and the reference really does have a non-zero one, so the zero above
        # is this kernel's contract and not a degenerate fixture
        _Qr, gref, _Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        assert np.abs(gref["coeff"]).max() > 0.0
    finally:
        ref.free()
        gpu.free()


@pytest.mark.parametrize("cfg", [CFG_KM, CFG_MB])
@pytest.mark.parametrize("rs", [1, 17])
@pytest.mark.parametrize("name", SPECS)
def test_ck_gradient_matches_central_differences(name, cfg, rs):
    """``dck`` is the one derivative that exists — it must be right.

    The step is the BACKEND's (see ``TABLES``): a pinned fp32 kernel's ``Q``
    carries a fixed contraction offset that varies with ``ck``.
    """
    nh, rtol, atol = TABLES[name]["fd"]
    f, kc, ref, gpu = _pair(name, cfg, rs=rs)
    params = _params(f)
    base = _in_grid(kc, 48, seed=3)
    data = _replicate(base, rs) if rs > 1 else base
    h = gpu.load_data(data)
    try:
        ck = np.asarray(params["ck"], np.complex128).copy()
        _Q0, g0, _ = gpu.compute(params, h, norm=None)
        for i in range(ck.size):
            for part in (0, 1):
                d = np.zeros_like(ck)
                d[i] = nh if part == 0 else 1j * nh
                pu, pn = dict(params), dict(params)
                pu["ck"], pn["ck"] = ck + d, ck - d
                fd = (gpu.compute(pu, h, norm=None)[0]
                      - gpu.compute(pn, h, norm=None)[0]) / (2 * nh)
                # the kernel's dck is the raw Wirtinger half; the real
                # parameter convention downstream is 2·Re / -2·Im
                dres = (2.0 * np.real(g0["ck"][i]) if part == 0
                        else -2.0 * np.imag(g0["ck"][i]))
                assert np.isclose(dres, fd, rtol=rtol, atol=atol), (
                    f"ck[{i}] part {part}: {dres!r} vs {fd!r}")
    finally:
        h.free()
        gpu.free()


@pytest.mark.parametrize("cfg", [CFG_KM, CFG_MB])
@pytest.mark.parametrize("name", SPECS)
def test_a_moved_coefficient_raises_the_exact_message(name, cfg):
    """Strict from the SECOND call on: no zeroing, no silent staleness."""
    guard = importlib.import_module(module_of(name)).GUARD
    f, kc, ref, gpu = _pair(name, cfg)
    params = _params(f)
    h = gpu.load_data(_in_grid(kc, 16, seed=4))
    try:
        gpu.compute(params, h, norm=None)          # the fill, and the snapshot
        moved = np.array(params["coeff"], np.float64, copy=True)
        moved[0] += 1e-12
        bad = dict(params)
        bad["coeff"] = moved
        with pytest.raises(ValueError) as ei:
            gpu.compute(bad, h, norm=None)
        assert str(ei.value) == guard
        assert str(ei.value).startswith(name + ":")
        assert "coeff moved after the cache was built" in str(ei.value)
        assert "cuda_flat_pwa_gen_s0e2" in str(ei.value)
        assert gpu.kernel.fills == 1, "a refused call must not re-fill"
        # the ORIGINAL coefficients still work (nothing was destroyed)
        gpu.compute(params, h, norm=None)
        # ... and the snapshot is PER HANDLE: a fresh handle may legitimately
        # be filled from a different vector (that is the fitter starting over)
        h2 = gpu.load_data(_in_grid(kc, 16, seed=4))
        gpu.compute(bad, h2, norm=None)
        assert gpu.kernel.fills == 2
        h2.free()
    finally:
        h.free()
        gpu.free()
        ref.free()


# ── the ONE lifecycle (per handle) ─────────────────────────────────────

@pytest.mark.parametrize("name", SPECS)
def test_the_one_lifecycle_fills_each_handle_just_once(name):
    """``load_data`` never fills; a handle's FIRST ``compute`` does, and only it.

    The snapshot is PER HANDLE — two handles are two fills, each guarded its
    own — and nothing about it is global: a handle's first compute gives the
    same numbers however many handles the kernel has seen.  There is no
    ``coeff=`` constructor form to reach the same state another way.
    """
    t = TABLES[name]
    f, kc, ref, gpu = _pair(name, CFG_KM)
    params = _params(f)
    coeff = np.asarray(params["coeff"], np.float64)
    data = _in_grid(kc, 32, seed=9)
    data_b = _in_grid(kc, 32, seed=10)
    try:
        h = gpu.load_data(data)
        h2 = gpu.load_data(data_b)
        assert gpu.kernel.loads == 2 and gpu.kernel.fills == 0
        for hh in (h, h2):
            assert hh._filled is False and hh._coeff is None
            assert not gpu.kernel.afix_ready(hh), "the load filled the cache"
        first = gpu.compute(params, h, norm=1.3)
        assert h._filled is True
        assert np.array_equal(h._coeff, coeff)
        assert gpu.kernel.afix_ready(h), (
            "the device flag and the lifecycle flag are set by the same fill")
        assert not gpu.kernel.afix_ready(h2), (
            "a fill must not leak to another handle")
        assert gpu.kernel.fills == 1, "one handle must cost one fill"
        gpu.compute(params, h2, norm=None)
        assert gpu.kernel.fills == 2, "two handles must cost two fills"
        # a repeat call re-fills nothing and is the same call
        for _ in range(3):
            again = gpu.compute(params, h, norm=1.3)
            assert _rel(again[0], first[0]) <= t["stable"]
            assert _rel(again[1]["ck"], first[1]["ck"]) <= t["stable"]
            assert _rel(again[2], first[2]) <= t["stable"]
            gpu.compute(params, h2, norm=1.3)
        assert gpu.kernel.fills == 2, "a repeat call re-filled a handle"
        assert gpu.kernel.loads == 2
        assert np.array_equal(h._coeff, coeff)
        # a later handle's FIRST compute is still the same computation
        h.free()
        h3 = gpu.load_data(data)
        fresh = gpu.compute(params, h3, norm=1.3)
        assert _rel(fresh[0], first[0]) <= t["fresh"]
        assert _rel(fresh[1]["ck"], first[1]["ck"]) <= t["fresh"]
        assert _rel(fresh[2], first[2]) <= t["fresh"]
        assert gpu.kernel.fills == 3
        h3.free()
        h2.free()
    finally:
        gpu.free()
        ref.free()


@pytest.mark.parametrize("name", SPECS)
def test_an_empty_sample_is_the_identity(name):
    f, kc, ref, gpu = _pair(name, CFG_KM)
    params = _params(f)
    data = _in_grid(kc, 16, seed=11)
    empty = {k: np.asarray(v)[:0] for k, v in data.items()}
    try:
        h = gpu.load_data(empty)
        assert not h._filled, "a load must not fill, not even an empty one"
        Q, g, P = gpu.compute(params, h, norm=1.3)
        assert h._filled is True                   # there is nothing to fill
        assert Q == 0.0
        assert np.abs(g["ck"]).max() == 0.0
        assert np.abs(g["coeff"]).max() == 0.0
        assert g["norm"] == 0.0
        assert P is not None and P.size == 0
        # the Gram of an empty sample is the zero matrix, by the same rule
        D = gpu.kernel.compute_gram(h, None, params["coeff"])
        assert D.shape == (gpu.kernel.n_wave, gpu.kernel.n_wave)
        assert np.abs(D).max() == 0.0
        h.free()
    finally:
        gpu.free()
        ref.free()


# ── the grouped objective ──────────────────────────────────────────────

@pytest.mark.parametrize("cfg", [CFG_KM, CFG_MB])
@pytest.mark.parametrize("rs", [1, 17])
@pytest.mark.parametrize("name", SPECS)
def test_the_grouped_objective_at_rsize_one_and_seventeen(name, cfg, rs):
    """The short final group included (34 events = 2 groups, the second 17)."""
    f, kc, ref, gpu = _pair(name, cfg, rs=rs)
    params = _params(f)
    data = _in_grid(kc, 34, seed=5)
    try:
        Qr, _gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=1.3)
        assert _rel(Pg, Pr) < GROUPED, f"P {_rel(Pg, Pr):.2e}"
        assert _rel(Qg, Qr) < GROUPED, f"Q {_rel(Qg, Qr):.2e}"
        assert np.abs(gg["coeff"]).max() == 0.0
    finally:
        ref.free()
        gpu.free()


@pytest.mark.parametrize("rs", [2, 3, 8, 10, 25, 64, 100, 257])
@pytest.mark.parametrize("name", SPECS)
def test_every_group_size_matches_the_reference(name, rs):
    """The grouped objective across the group sizes real fits use (10..100).

    ``rs`` 25/100 put MANY groups in one block; 257 is past the tile ceiling,
    so it is the path-B walk.
    """
    f, kc, ref, gpu = _pair(name, CFG_KM, rs=rs)
    params = _params(f)
    data = _in_grid(kc, 200, seed=6)
    try:
        Qr, _gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=1.3)
        assert _rel(Pg, Pr) < GROUPED, f"P {_rel(Pg, Pr):.2e}"
        assert _rel(Qg, Qr) < GROUPED, f"Q {_rel(Qg, Qr):.2e}"
        assert np.abs(gg["coeff"]).max() == 0.0
    finally:
        ref.free()
        gpu.free()


@pytest.mark.parametrize("name", SPECS)
def test_a_group_larger_than_the_tile_is_walked_internally(name):
    """``rsize > te`` is path B: ONE block per group, walked in chunks.

    A pinned kernel has no free path at all, so it keeps the group in one block
    and re-reads the cache with the finished constant.  The objective must be
    the same one.
    """
    f, kc, ref, gpu = _pair(name, CFG_KM, rs=10 ** 6)
    params = _params(f)
    data = _in_grid(kc, 34, seed=5)
    try:
        Qr, _gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=1.3)
        assert _rel(Pg, Pr) < GROUPED, f"P {_rel(Pg, Pr):.2e}"
        assert _rel(Qg, Qr) < GROUPED, f"Q {_rel(Qg, Qr):.2e}"
        assert np.abs(gg["coeff"]).max() == 0.0
    finally:
        ref.free()
        gpu.free()


@pytest.mark.parametrize("name", SPECS)
def test_rsize_one_is_the_default_and_the_per_event_objective(name):
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    data = _in_grid(kc, 48, seed=4)
    dflt = create_backend(name, kc, model="pwa_gen")
    one = create_backend(name, kc, model="pwa_gen", resolution_size=1)
    try:
        a = dflt.compute(params, dflt.load_data(data), norm=1.3)
        b = one.compute(params, one.load_data(data), norm=1.3)
        assert _rel(a[0], b[0]) <= 1e-14
        assert _rel(a[2], b[2]) <= 1e-14
    finally:
        dflt.free()
        one.free()


# ── the GRAM (the phase-space pre-integration) ─────────────────────────

def _gram_setup(cfg, ne=200, seed=0):
    """``(fitter, kc, params, data, oracle)`` — no backend created."""
    from tabpwa.integrated_pwa_gen import IntegratedPWAFlat
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    data = _in_grid(kc, ne, seed=seed)
    return f, kc, params, data, IntegratedPWAFlat(kc)


@pytest.mark.parametrize("cfg", [CFG_KM, CFG_MB])
@pytest.mark.parametrize("name", SPECS)
def test_compute_gram_matches_the_numpy_oracle(name, cfg):
    """The device reduction against the fp64 oracle, at the fp32-cache floor.

    Also the one property the mapping guarantees: both triangles come from the
    SAME product, so the asymmetry is only the cross-block atomic order.
    """
    _f, _kc, params, data, gram = _gram_setup(cfg)
    gpu = create_backend(name, _kc, model="pwa_gen")
    h = gpu.load_data(data)
    try:
        Dg = gpu.kernel.compute_gram(h, np.asarray(data["weight"]),
                                     params["coeff"])
        Dn = gram.gram(data, params["coeff"])
        assert Dg.shape == Dn.shape == (gpu.kernel.n_wave, gpu.kernel.n_wave)
        rel = np.abs(Dg - Dn).max() / np.abs(Dn).max()
        assert rel < GRAM_FLOOR, f"D {rel:.2e}"
        asym = np.abs(Dg - Dg.conj().T).max() / np.abs(Dg).max()
        assert asym < GRAM_HERM, f"Hermitian asymmetry {asym:.2e}"
        # the norm is the quantity that matters, and it is dominated by the
        # diagonal — gate it on its own so a small off-diagonal drift cannot
        # hide behind a large max|D|
        ck = np.asarray(params["ck"], np.complex128)
        assert abs(gram.norm(ck, Dg) - gram.norm(ck, Dn)) \
            <= GRAM_NORM * abs(gram.norm(ck, Dn))
    finally:
        h.free()
        gpu.free()


@pytest.mark.parametrize("name", SPECS)
def test_the_gram_uses_the_weight_vector_it_is_given(name):
    """A non-uniform ``weight`` reaches the reduction (a uniform test would
    pass with the weights ignored entirely)."""
    _f, kc, params, data, gram = _gram_setup(CFG_KM)
    gpu = create_backend(name, kc, model="pwa_gen")
    h = gpu.load_data(data)
    try:
        w2 = 1.0 + 0.5 * np.cos(np.arange(len(np.asarray(data["weight"]))))
        D2 = gpu.kernel.compute_gram(h, w2, params["coeff"])
        # the oracle multiplies the data's own weights by `weight=`, so the
        # comparison sample carries ones
        ones = dict(data)
        ones["weight"] = np.ones_like(np.asarray(data["weight"]))
        Dn2 = gram.gram(ones, params["coeff"], weight=w2)
        rel = np.abs(D2 - Dn2).max() / np.abs(Dn2).max()
        assert rel < GRAM_FLOOR, f"weighted D {rel:.2e}"
        # it really changed the answer
        D1 = gpu.kernel.compute_gram(h, None, params["coeff"])
        Dn1 = gram.gram(ones, params["coeff"])
        assert np.abs(D1 - Dn1).max() / np.abs(Dn1).max() < GRAM_FLOOR
        assert np.abs(D2 - D1).max() / np.abs(D1).max() > 1e-2
    finally:
        h.free()
        gpu.free()


@pytest.mark.parametrize("name", SPECS)
def test_the_gram_goes_through_the_one_lifecycle(name):
    """An unfilled handle needs the coeff; a filled one is guarded; freed is
    reported."""
    guard = importlib.import_module(module_of(name)).GUARD
    _f, kc, params, data, _gram = _gram_setup(CFG_KM, ne=32)
    gpu = create_backend(name, kc, model="pwa_gen")
    h = gpu.load_data(data)
    try:
        with pytest.raises(ValueError, match="needs the pinned coeff"):
            gpu.kernel.compute_gram(h, None, None)
        assert not h._filled and gpu.kernel.fills == 0
        gpu.kernel.compute_gram(h, None, params["coeff"])
        assert h._filled and gpu.kernel.fills == 1
        moved = np.array(params["coeff"], np.float64, copy=True)
        moved[-1] += 1e-9
        with pytest.raises(ValueError) as ei:
            gpu.kernel.compute_gram(h, None, moved)
        assert str(ei.value) == guard
        assert gpu.kernel.fills == 1, "a refused Gram must not re-fill"
        # a freed handle is reported, not crashed
        h.free()
        with pytest.raises((RuntimeError, ValueError), match="already freed"):
            gpu.kernel.compute_gram(h, None, params["coeff"])
    finally:
        gpu.free()


# ── guards and edges ───────────────────────────────────────────────────

@pytest.mark.parametrize("name", SPECS)
def test_rejects_an_already_deduped_config(name):
    from tabpwa.pwa_gen_config import build_from_groups, dedup
    m = Fitter(CFG_KM).model
    dd = dedup(build_from_groups(m, n_interp=200))
    with pytest.raises(ValueError, match="FULL"):
        create_backend(name, dd, model="pwa_gen")


@pytest.mark.parametrize("name", SPECS)
def test_resolution_size_below_one_is_rejected(name):
    f = Fitter(CFG_KM)
    f.apply_constrains()
    with pytest.raises(ValueError, match="resolution_size"):
        create_backend(name, f.kernel_config, resolution_size=0,
                       model="pwa_gen")


@pytest.mark.parametrize("name", SPECS)
def test_wrong_parameter_lengths_are_rejected(name):
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    be = create_backend(name, kc, model="pwa_gen")
    h = be.load_data(_in_grid(kc, 8, seed=1))
    try:
        bad = dict(params)
        bad["ck"] = np.zeros(be.kernel.n_wave + 1, complex)
        with pytest.raises(ValueError, match="n_wave"):
            be.compute(bad, h, norm=None)
        bad = dict(params)
        bad["coeff"] = np.zeros(be.kernel.n_basis + 1)
        with pytest.raises(ValueError, match="n_basis"):
            be.compute(bad, h, norm=None)
    finally:
        h.free()
        be.free()


@pytest.mark.parametrize("name", SPECS)
def test_a_freed_handle_is_reported_and_free_is_idempotent(name):
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    k = TABLES[name]["make_kernel"](kc)
    h = k.load_data(_in_grid(kc, 8, seed=1))
    k.compute(params, h, norm=1.3)
    h.free()
    h.free()                     # idempotent handle
    assert h.ptr is None
    with pytest.raises((RuntimeError, ValueError), match="already freed"):
        k.compute(params, h, norm=None)
    k.free()
    k.free()                     # idempotent context
    assert k._ctx is None


@pytest.mark.parametrize("name", SPECS)
def test_two_live_handles_are_independent(name):
    """The fitter holds phsp AND data at once — each load owns its cache."""
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    k = TABLES[name]["make_kernel"](kc)
    try:
        h1 = k.load_data(_in_grid(kc, 16, seed=1))
        h2 = k.load_data(_in_grid(kc, 16, seed=2))
        assert h1.ptr != h2.ptr
        Qa, _g, Pa = k.compute(params, h1, norm=None)
        Qb, _g, Pb = k.compute(params, h2, norm=None)
        Qa2 = k.compute(params, h1, norm=None)[0]
        assert abs(Qa - Qa2) <= 1e-12 * max(abs(Qa), 1.0)
        assert _rel(Pa, Pb) > 1e-6          # ... and they are different data
        h1.free()
        h2.free()
    finally:
        k.free()
