"""``cpu_flat_pwa_gen`` — the CPU (C + OpenMP + AVX-512 fp64) flat leaf.

The SHARED contract (Q/P/dck/dcoeff against ``numpy_flat``, FD gradients,
``resolution_size`` 1 vs 17, chunk invariance, handle independence, the input
errors) lives in ``tests/_flat_compute.py`` and reaches this backend through
the REGISTRY, because it is registered as an untagged ``pwa_gen`` leaf.  This
file keeps only what is specific to this kernel:

* the agreement floor it is gated at (``FLOORS`` in ``_flat_compute``),
* the grouped objective at ``rsize 17`` — the group-TOTAL-weight form
  ``Q = -Σ_g W_g·log(S_g/W_g)``, checked against its own numpy formula from the
  kernel's P, not only against the reference,
* the shape contract derived from the arrays (``n_wave`` / ``n_res`` /
  ``n_proj`` from shapes, never a stored count),
* ``batch_size`` changing NOTHING (bit-identical, not merely close),
* two live handles independent / a freed handle reported,
* the four host-side dedups being byte-exact (they must not move a number),
* the standalone artefact: own module, own ``.c``/``.so``, own symbols, and an
  AVX-512 fp64 build with no fp32 anywhere.
"""
import os

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import backend_tags, backends_for_model, create_backend
from _flat_grid import _in_grid

from _flat_compute import (CFG_KM, CONFIG_OWNER, FLOORS, GROUPED, QUALIFYING,
                           SPECS, _rel, _replicate)

NAME = "cpu_flat_pwa_gen"

#: the OWNERSHIP rule in ``_flat_compute``: this file owns exactly this leaf's
#: config contract.  A misdirected owner (the name moved to another family file)
#: fails here instead of being covered by nothing.
_ME = os.path.basename(__file__)
_OWNED = {n for n, f in CONFIG_OWNER.items() if f == _ME}
assert _OWNED == {NAME}, (
    f"{_ME}'s leaf is out of sync with _flat_compute.CONFIG_OWNER: "
    f"missing {sorted({NAME} - _OWNED)}, stale {sorted(_OWNED - {NAME})}")


def _rel_val(a, b):
    return _rel(a, b)


def _pair(cfg, rs=1, ne=64, seed=0):
    """(fitter, kernel config, params, data) for a fixture."""
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    base = _in_grid(kc, ne, seed)
    data = _replicate(base, rs) if rs > 1 else base
    return f, kc, params, data


# ── 1. the fp64 agreement floor ─────────────────────────────────────────

@pytest.mark.parametrize("cfg", QUALIFYING)
@pytest.mark.parametrize("norm", [None, 1.3])
@pytest.mark.parametrize("rs", [1, 17])
def test_agrees_with_numpy_flat_at_the_floor(cfg, norm, rs):
    """It must reproduce the fp64 oracle at the floor its own file records.

    ``FLOORS[NAME]`` is the number this backend is gated at by the shared
    suite, so the same numbers are asserted here directly (a floor that only
    lives in the shared file could drift away from this file's claim).
    """
    rel, p, coeff, abs_ok, cfgs = FLOORS[NAME]
    assert cfg in cfgs
    _f, kc, params, data = _pair(cfg, rs)
    ref = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=rs)
    cpu = create_backend(NAME, kc, model="pwa_gen", resolution_size=rs)
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=norm)
        Qc, gc, Pc = cpu.compute(params, cpu.load_data(data), norm=norm)
        for label, a, b, bound in (
                ("Q", Qc, Qr, rel), ("P", Pc, Pr, p),
                ("dck", gc["ck"], gr["ck"], rel),
                ("dcoeff", gc["coeff"], gr["coeff"], coeff)):
            worst = float(np.abs(np.asarray(a).ravel()
                                 - np.asarray(b).ravel()).max())
            value = worst / max(np.abs(np.asarray(b)).max(), 1e-300)
            assert value < bound or (abs_ok and worst < 1e-4), (
                f"{cfg} rs={rs} norm={norm}: {label} rel {value:.2e} "
                f"(abs {worst:.2e})")
        if norm is not None:
            assert _rel_val(cpu.kernel._last_dnorm, ref._last_dnorm) < rel
    finally:
        ref.free()
        cpu.free()


#: The worst agreement this leaf reaches against ``numpy_flat``, per quantity,
#: over ``QUALIFYING x rsize {1,17} x norm {None, 1.3}``.  RECORDED here so the
#: floor below is checked against a measurement rather than against nothing:
#: if a floor is ever tightened below these, this test fails.
MEASURED_FP64 = {"Q": 1.22e-15, "P": 3.11e-15, "ck": 1.86e-15,
                 "coeff": 6.09e-15}


def test_the_recorded_floor_is_the_measured_fp64_margin():
    """The floor is 1e-12, and this fp64 kernel clears it by >=160x.

    Plain ``rel`` (``abs_ok`` False) on purpose: with the absolute fallback on,
    the 1e-4 fallback would be what binds and the 1e-12 would never decide.
    Both sides were fp64 before this was 1e-12, so what the leaf measured was
    NOT accumulation: while the reference interpolated the flat tables with
    two-point LINEAR interpolation the CPU kernel's agreement was
    Q 9.3e-7 / P 3.0e-6 / dck 1.4e-6 / dcoeff 5.0e-6 — the chord error of the
    `-m²` row, which is what the old s0c floor (2e-5 / 4e-5 / 2e-4) was sized
    for.  The reference reads Catmull-Rom now, the difference above is gone,
    and the floor follows the measurement down.
    """
    assert FLOORS[NAME] == (1e-12, 1e-12, 1e-12, False, tuple(QUALIFYING))
    assert FLOORS[NAME][0] > 100 * MEASURED_FP64["Q"]
    assert FLOORS[NAME][1] > 100 * MEASURED_FP64["P"]
    assert FLOORS[NAME][0] > 100 * MEASURED_FP64["ck"]
    assert FLOORS[NAME][2] > 100 * MEASURED_FP64["coeff"]


# ── 2. the grouped objective, checked against ITS OWN formula ───────────

@pytest.mark.parametrize("cfg", [CFG_KM, "tests/config_pwa_gen_cp.yml"])
def test_grouped_objective_is_the_group_total_weight_form(cfg):
    """``Q = -Σ_g W_g·log(S_g/W_g)``, recomputed from the kernel's own P.

    The reference comparison alone would pass a kernel that grouped the
    wrong way; this recomputes the objective from the P the kernel returned,
    so the group bookkeeping (total weight ``W_g``, short last group, zero
    weight groups) is gated independently.
    """
    _f, kc, params, data = _pair(cfg, 1, ne=34, seed=5)
    rs = 17
    data = _replicate(data, 1)
    w = np.asarray(data["weight"], float)
    n = (w.size // rs) * rs
    wg = w[:n].reshape(-1, rs)
    data["weight"] = w.copy()
    data["weight"][:n] = (wg / wg.sum(axis=1, keepdims=True)).reshape(-1)
    norm = 1.3
    cpu = create_backend(NAME, kc, model="pwa_gen", resolution_size=rs)
    try:
        h = cpu.load_data(data)
        Q, _g, P = cpu.compute(params, h, norm=norm)
        P = np.asarray(P)
        w = np.asarray(data["weight"], float)      # the group-NORMALISED one
        ng = (P.size + rs - 1) // rs
        pad = ng * rs - P.size
        wg = np.concatenate([w, np.zeros(pad)]).reshape(ng, rs)
        pg = np.concatenate([P, np.zeros(pad)]).reshape(ng, rs)
        bg = np.concatenate([np.asarray(data["bkg"], float),
                             np.zeros(pad)]).reshape(ng, rs)
        W = wg.sum(axis=1)
        S = (wg * (pg / norm + bg)).sum(axis=1)
        safe = W > 0
        expect = -np.sum(W[safe] * np.log(S[safe] / W[safe]))
        assert abs(Q - expect) <= 1e-12 * max(abs(expect), 1.0)
        # the grouped objective is NOT the per-event sum (a Jensen shift):
        # with group-normalised weights Σw differs per group, so the plain
        # per-event log must differ measurably
        per_event = -np.sum(w * np.log(P / norm + np.asarray(
            data["bkg"], float)))
        assert abs(per_event - Q) > 1e-6 * max(abs(Q), 1.0)
        # and against the reference
        ref = create_backend("numpy_flat", kc, model="pwa_gen",
                            resolution_size=rs)
        try:
            Qr = ref.compute(params, ref.load_data(data), norm=norm)[0]
            assert _rel(Q, Qr) < GROUPED
        finally:
            ref.free()
    finally:
        cpu.free()


# ── 3. the shape contract (derived, never stored) ───────────────────────

@pytest.mark.parametrize("cfg", [CFG_KM, "tests/config_pwa_gen_mb.yml"])
def test_shapes_are_derived_from_the_arrays(cfg):
    _f, kc, _params, _data = _pair(cfg)
    be = create_backend(NAME, kc, model="pwa_gen")
    k = be.kernel
    try:
        fl = np.asarray(kc["fl_table"])
        num = np.asarray(kc["num_table"])
        assert k.n_wave == fl.shape[0]
        assert k.n_res == num.shape[1]
        assert k.n_decay == fl.shape[1]
        assert k.n_proj == np.asarray(kc["matrix_angle"]).shape[1] // k.n_wave
        assert k.n_basis == np.asarray(kc["basis_table"]).shape[0]
        st = k.kernel_stats()
        # 8 LOGICAL lanes in both builds; the ISA is reported by the .so
        assert st["lanes"] == 8 and st["isa"] in ("avx512f", "avx2")
        assert st["n_wave"] == k.n_wave and st["n_basis"] == k.n_basis
        # the four dedups never GROW a table, and never change n_basis
        assert st["n_fl_u"] <= k.n_wave * k.n_decay
        assert st["n_num_u"] <= k.n_wave * k.n_res
        assert st["n_uslot"] <= k.n_basis
        assert st["n_fa"] <= k.n_proj * k.n_wave
        assert st["cache_bytes"] == 0        # nothing cached from a call
    finally:
        be.free()


# ── 4. batch_size is an inner loop that does not exist here ─────────────

@pytest.mark.parametrize("rs", [1, 17])
def test_batch_size_changes_nothing(rs):
    """BIT-identical, not merely close: there is no chunk in this kernel.

    The sample is walked in event tiles of 8 and the parallel axis is the tile
    (or the log-sum group), so ``batch_size`` cannot move a number.  The
    shared suite's 1e-14 gate is therefore cleared exactly.
    """
    _f, kc, params, data = _pair(CFG_KM, 1, ne=500, seed=3)
    base = None
    for bs in (1, 5, 1000, 10 ** 9):
        be = create_backend(NAME, kc, model="pwa_gen", resolution_size=rs,
                            batch_size=bs)
        try:
            h = be.load_data(data)
            Q, g, P = be.compute(params, h, norm=1.3)
            got = (Q, np.asarray(g["ck"]).copy(), np.asarray(g["coeff"]).copy(),
                   float(g["norm"]), np.asarray(P).copy())
        finally:
            be.free()
        if base is None:
            base = got
        else:
            assert got[0] == base[0], f"bs={bs}: Q {got[0]!r} != {base[0]!r}"
            for a, b in ((got[1], base[1]), (got[2], base[2]),
                         (got[4], base[4])):
                assert np.array_equal(a, b), f"bs={bs}: not bit-identical"
            assert got[3] == base[3]


# ── 5. handles: two live, one freed ─────────────────────────────────────

def test_two_live_handles_are_independent_and_a_freed_one_is_reported():
    """The fitter holds phsp AND data at once (AGENTS gotcha 15)."""
    _f, kc, params, _data = _pair(CFG_KM, ne=16)
    be = create_backend(NAME, kc, model="pwa_gen")
    k = be.kernel
    try:
        a = _in_grid(kc, 16, 1)
        b = _in_grid(kc, 16, 2)
        h1, h2 = k.load_data(a), k.load_data(b)
        assert h1.ptr != h2.ptr
        Qa1, _g, Pa = k.compute(params, h1, norm=None)
        Qb, _g, Pb = k.compute(params, h2, norm=None)
        Qa2 = k.compute(params, h1, norm=None)[0]
        assert Qa1 == Qa2                       # h2's load did not clobber h1
        assert _rel(Pa, Pb) > 1e-6              # ... and they differ
        assert Qa1 != Qb
        h1.free()
        h1.free()                               # idempotent
        with pytest.raises(RuntimeError, match="already freed"):
            k.compute(params, h1, norm=None)
        h2.free()
    finally:
        be.free()


# ── 6. the dedups are EXACT (byte-exact keys, never a sampled value) ────

def test_the_four_dedups_are_byte_exact():
    """A dedup must be a fold on table bytes, so the rows must still match.

    Reconstructs every use from its deduped slot and demands ``array_equal``
    — a dedup that merged two different rows would still produce repeatable
    numbers, so only this pins it.
    """
    from tabpwa.backends.cpu_flat_pwa_gen_backend import (_dedup_uses,
                                                          _uslot_map)
    f, kc, _params, _data = _pair("tests/config_pwa_gen_km.yml")
    fl = np.asarray(kc["fl_table"]).reshape(-1,
                                           np.asarray(kc["fl_table"]).shape[-1])
    row, col, use = _dedup_uses(
        range(fl.shape[0]), np.asarray(kc["fl_q_index"]).ravel(),
        [fl[i].tobytes() for i in range(fl.shape[0])])
    assert row.size <= fl.shape[0]
    for i in range(fl.shape[0]):
        u = use[i]
        assert np.array_equal(fl[row[u]], fl[i])
        assert col[u] == np.asarray(kc["fl_q_index"]).ravel()[i]
    bas = np.asarray(kc["basis_table"])
    urow, ucol, slot = _uslot_map(bas, kc["basis_m_index"])
    bmi = np.asarray(kc["basis_m_index"])
    for m in range(bas.shape[0]):
        u = slot[m]
        assert np.array_equal(bas[urow[u]], bas[m])
        assert ucol[u] == bmi[m]


# ── 7. registration / artefact ──────────────────────────────────────────

def test_is_an_untagged_pwa_gen_leaf():
    reg = backends_for_model("pwa_gen")
    assert NAME in reg
    assert backend_tags(NAME, "pwa_gen") == frozenset()   # an untagged LEAF
    assert NAME in SPECS                                   # the shared suite
    assert NAME in FLOORS


def test_own_source_library_and_symbols():
    """A NEW kernel: its own ``.c``, its own ``.so``, its own symbols."""
    import tabpwa.cpu as cpu_pkg
    from tabpwa.backends import cpu_flat_pwa_gen_backend as mod
    src = os.path.join(os.path.dirname(os.path.abspath(cpu_pkg.__file__)),
                       "kernels_cpu_flat_pwa_gen.c")
    assert os.path.exists(src)
    text = open(src).read()
    # no fp32 anywhere in the kernel, and the AVX-512 fp64 path is the only one
    assert "float" not in text
    assert "__m512d" in text and "_mm512_mask_i32gather_pd" in text
    assert "cpu_flat_pwa_gen_compute" in text
    assert '#include <immintrin.h>' in text
    # the module's own build product, and the symbols it needs
    be = create_backend(NAME, Fitter(CFG_KM).kernel_config, model="pwa_gen")
    try:
        lib = be.kernel._lib
        for sym in ("cpu_flat_pwa_gen_create", "cpu_flat_pwa_gen_load",
                    "cpu_flat_pwa_gen_compute", "cpu_flat_pwa_gen_free",
                    "cpu_flat_pwa_gen_free_ctx"):
            assert getattr(lib, sym) is not None
        assert mod._lib is lib
    finally:
        be.free()


# ── 8. the integral-k Chebyshev path is exact ───────────────────────────

@pytest.mark.parametrize("cfg", [CFG_KM, "tests/config_pwa_gen_cp.yml"])
def test_the_integral_k_fast_path_equals_the_general_cos_path(cfg):
    """``cos(k·x + b)`` via Chebyshev == via libmvec's ``cos``, to round-off.

    ``create`` enables the fast path only when every ``k`` is integral
    (|k| <= 8) and a context can be built with ``general_cos=True`` to force
    the libmvec path — so this is an A/B of the two on the same sample, not
    an assertion that they are the same code.  Both are fp64; the agreement
    is therefore expected at ~1e-15, far below the 4e-5 floor.
    """
    _f, kc, params, data = _pair(cfg, ne=64, seed=2)
    ref = create_backend("numpy_flat", kc, model="pwa_gen")
    fast = create_backend(NAME, kc, model="pwa_gen")
    gen = create_backend(NAME, kc, model="pwa_gen", general_cos=True)
    try:
        assert fast.kernel.kernel_stats()["general_cos"] is False
        assert gen.kernel.kernel_stats()["general_cos"] is True
        hf = fast.load_data(data)
        hg = gen.load_data(data)
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        Qf, gf, Pf = fast.compute(params, hf, norm=1.3)
        Qg, gg, Pg = gen.compute(params, hg, norm=1.3)
        # the two paths agree far below the floor
        assert _rel(Pf, Pg) < 1e-13, "P"
        assert abs(Qf - Qg) <= 1e-13 * max(abs(Qg), 1.0), "Q"
        assert _rel(gf["ck"], gg["ck"]) < 1e-13, "dck"
        assert _rel(gf["coeff"], gg["coeff"]) < 1e-13, "dcoeff"
        # ... and both are the reference's answer at the floor
        rel, p, coeff, _abs_ok, _cfgs = FLOORS[NAME]
        assert _rel(Pf, Pr) < p and _rel(Pg, Pr) < p
        assert _rel(gf["coeff"], gr["coeff"]) < coeff
    finally:
        ref.free()
        fast.free()
        gen.free()


def test_an_integral_k_config_uses_the_fast_path():
    """Every fixture's ``angle_k`` is integral, so the fast path is on."""
    for cfg in (CFG_KM, "tests/config_pwa_gen_cp.yml", QUALIFYING[0]):
        _f, kc, _params, _data = _pair(cfg, ne=8)
        be = create_backend(NAME, kc, model="pwa_gen")
        try:
            ak = np.asarray(kc["angle_k"])
            assert np.all(ak == np.round(ak))
            assert np.abs(ak).max() <= 8
            assert be.kernel.kernel_stats()["general_cos"] is False
        finally:
            be.free()


# ── 9. the two BUILDS are bit-identical ────────────────────────────────

def _build_avx2(tmp_path):
    """Compile the kernel with explicit AVX2+FMA flags (the tree's own build
    is `-march=native`, i.e. AVX-512 on an AVX-512 host)."""
    import shutil
    import subprocess

    import tabpwa.cpu as cpu_pkg
    gcc = shutil.which("gcc") or shutil.which("cc")
    if gcc is None:
        pytest.skip("no C compiler to build the AVX2 variant")
    src = os.path.join(os.path.dirname(os.path.abspath(cpu_pkg.__file__)),
                       "kernels_cpu_flat_pwa_gen.c")
    out = str(tmp_path / "libcpu_flat_pwa_gen_avx2.so")
    r = subprocess.run([gcc, "-shared", "-fPIC", "-O3", "-mavx2", "-mfma",
                        "-fopenmp", "-o", out, src, "-lm", "-lmvec"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        r = subprocess.run([gcc, "-shared", "-fPIC", "-O3", "-mavx2", "-mfma",
                            "-fopenmp", "-o", out, src, "-lm"],
                           capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"cannot build the AVX2 variant here: {r.stderr[:200]}")
    return out


@pytest.fixture
def two_builds(tmp_path):
    """(native kernel factory, avx2 kernel factory), restoring the module."""
    from tabpwa.backends import cpu_flat_pwa_gen_backend as mod
    mod._load_lib()
    ffi, native = mod._ffi, mod._lib
    avx2 = ffi.dlopen(_build_avx2(tmp_path))

    def make(lib, kernel_config, **kw):
        mod._lib = lib
        try:
            return mod.CPUKernelFlatPwaGen(kernel_config, **kw)
        finally:
            mod._lib = native

    yield (lambda kc, **kw: make(native, kc, **kw),
           lambda kc, **kw: make(avx2, kc, **kw))
    mod._lib = native


@pytest.mark.parametrize("cfg", [QUALIFYING[0], CFG_KM,
                                 "tests/config_pwa_gen_mb.yml"])
@pytest.mark.parametrize("rs", [1, 17])
@pytest.mark.parametrize("norm", [None, 1.3])
def test_the_two_builds_are_bit_identical(two_builds, cfg, rs, norm):
    """AVX-512 and AVX2+FMA must produce the SAME fp64 numbers, bit for bit.

    8 LOGICAL lanes in both builds (one zmm, or two ymm halves) means the
    tile boundaries, the parallel partition, the lane -> event mapping and
    every reduction's grouping are the same, and `#pragma GCC optimize
    ("fp-contract=off")` pins the per-group SCALAR rounding — without it the
    two builds disagree by 1 ulp on Q/dck (measured while porting).  This is
    an `array_equal` gate, not a tolerance: a 1-ulp drift here would mean the
    "one algorithm, two builds" claim is false.
    """
    mk_nat, mk_avx2 = two_builds
    _f, kc, params, data = _pair(cfg, rs)
    kn = mk_nat(kc, resolution_size=rs)
    ka = mk_avx2(kc, resolution_size=rs)
    try:
        hn, ha = kn.load_data(data), ka.load_data(data)
        Qn, gn, Pn = kn.compute(params, hn, norm=norm)
        Qa, ga, Pa = ka.compute(params, ha, norm=norm)
        assert Qn == Qa, f"{cfg} rs={rs} norm={norm}: Q {Qn!r} vs {Qa!r}"
        assert np.array_equal(Pn, Pa), "P"
        assert np.array_equal(gn["ck"], ga["ck"]), "dck"
        assert np.array_equal(gn["coeff"], ga["coeff"]), "dcoeff"
        if norm is not None:
            assert kn._last_dnorm == ka._last_dnorm, "dnorm"
    finally:
        kn.free()
        ka.free()


def test_the_width_is_a_compile_time_abstraction():
    """The body is written ONCE against `vpd`; both widths are declared."""
    import tabpwa.cpu as cpu_pkg
    src = os.path.join(os.path.dirname(os.path.abspath(cpu_pkg.__file__)),
                       "kernels_cpu_flat_pwa_gen.c")
    text = open(src).read()
    assert "typedef __m512d vpd;" in text          # AVX-512 branch
    assert "typedef struct { __m256d a, b; } vpd;" in text   # AVX2 branch
    assert "#elif defined(__AVX2__) && defined(__FMA__)" in text
    # neither width -> a loud build failure, and no fp32 anywhere
    assert '#error "cpu_flat_pwa_gen needs AVX2+FMA' in text
    assert "float" not in text
    # ONE algorithm: no second copy of the tile body behind an #if
    assert text.count("static void cfpg_tile(") == 1
    assert text.count("static inline void cfpg_bin(") == 1
    assert text.count("static inline vpd cfpg_interp(") == 1
