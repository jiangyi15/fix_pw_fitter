"""The compute suite every flat ``pwa_gen`` CUDA backend shares (registry-driven).

These backends implement ONE function — ``load_data(data)`` returning a handle
and ``compute(params, handle, norm=...) -> (Q, grads, P)`` — so the tests for
what that function must DO are written once, parametrised over the REGISTRY:
``backends_to_test()`` derives the list from ``backends_for_model("pwa_gen")``,
so a backend registered later is covered without touching this file.  Nothing
about a variant's internals belongs here.

Shared: Q/P/grads vs the fp64 ``numpy_flat`` reference (both norms, rsize 1 and
17), the grouped objective, ``rsize = 1`` == the per-event NLL, FD gradients,
chunk invariance (multi-chunk == single-chunk, a chunk boundary does not move
the fill), handle independence / freed handle, and the input errors.

The list of backends comes from the REGISTRY and its TAGS (``reference`` /
``delegate`` / ``n_res1``, declared at registration) — not from a name list, so
a backend registered later is covered automatically and an exclusion is a
capability, never a silent skip.

Local (each variant's own file): dedup/index pins, CSR, tile/smem, launch
counts, path A/B, the parent A/B, the measured handle, own-module/`.so`/symbols,
registration, and the sparse-row edge cases.  Tolerances are the ones the
per-variant files used: the fp32 cache floor is ABSOLUTE, so ``_floor_ok``
accepts either the relative or the absolute bound.

``test_flat_compute.py`` imports this module so pytest collects the suite under
``[<backend name>]`` ids.
"""
import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import (backend_tags, backends_for_model,
                             create_backend)
from test_cuda_flat_cache_pwa_gen import _in_grid

CFG_ONE = "tests/config_pwa_gen.yml"
CFG_KM = "tests/config_pwa_gen_km.yml"
CFG_MB = "tests/config_pwa_gen_mb.yml"
CFG_IDENT = "tests/config_pwa_gen_ident.yml"
CFG_SAME = "tests/config_pwa_gen_same.yml"
CFG_CP = "tests/config_pwa_gen_cp.yml"
#: every fixture with exactly one BW per chain (a `_mb` chain has two)
QUALIFYING = [CFG_ONE, CFG_KM, CFG_IDENT, CFG_SAME, CFG_CP]

# ── the fp32 cache floor ────────────────────────────────────────────────
# The one load-time cache is the numerator `amp = fa·fl·num` as float2, so
# agreement with the fp64 `numpy_flat` reference is limited by fp32, not by the
# algebra.  The floor is therefore a MEASURED per-backend number, never a
# family maximum: the dense backends reach 1e-6, the tile family 3e-5 on P.
# For `s0c`/`s0c2` the `_ident`/`_cp` fixtures span the widest P dynamic range,
# so their file also gated the ABSOLUTE error (1e-4) — `abs_ok` records that,
# and `_floor_ok` is used only for those two.
CACHE_FLOOR_ABS = 1e-4
NH = 1e-6

#: Per-backend agreement floors, copied VERBATIM from each variant's own file
#: (HEAD), plus whether that file used `_floor_ok` (rel OR the absolute floor)
#: or a plain `_rel`, and the CONFIG SET its anchor was measured on.  A
#: consolidation may not weaken a gate, and the family genuinely differs:
#:
#:   * `cuda_flat_pwa_gen` / `cuda_flat_cache_pwa_gen` gated at 1e-6 pure
#:     `_rel` on KM+MB only (their files never ran `_ident`/`_cp`).  On the
#:     wider qualifying set these two MEASURE P 2.97e-6 / 3.54e-6 and coeff
#:     4.97e-6 / 4.70e-6 (rsize 1, widest of rs 1/17 x norm None/1.3), i.e. the
#:     1e-6 claim is shape-bound — so the record keeps the shapes it was
#:     measured on instead of inventing a wider claim.
#:   * `cuda_flat_cache_pwa_gen` never compared the raw `coeff` vector in its
#:     anchor; its own coeff gate is PER NAME at 1e-4 (the deduped raw vector
#:     reads 1.8e-6 on KM, which is why 1e-6 would be a NEW, unmet gate).
#:   * the `s0e`/`gen2`/`1bw` family: 2e-5 / P 3e-5 / coeff 1e-4, strict `_rel`.
#:   * `s0c`/`s0c2`: 2e-5 / P 4e-5 / coeff 2e-4, `_floor_ok` (rel OR abs 1e-4).
#:
#:   name -> (rel, p, coeff, abs_ok, cfgs)
FLOORS = {
    "cuda_flat_cache_pwa_gen": (1e-6, 1e-6, 1e-4, False, (CFG_KM, CFG_MB)),
    "cuda_flat_pwa_gen": (1e-6, 1e-6, 1e-6, False, (CFG_KM, CFG_MB)),
    "cuda_flat_pwa_gen2": (2e-5, 3e-5, 1e-4, False, tuple(QUALIFYING)),
    "cuda_flat_pwa_gen_1bw": (2e-5, 3e-5, 1e-4, False, tuple(QUALIFYING)),
    "cuda_flat_pwa_gen_s0c": (2e-5, 4e-5, 2e-4, True, tuple(QUALIFYING)),
    "cuda_flat_pwa_gen_s0c2": (2e-5, 4e-5, 2e-4, True, tuple(QUALIFYING)),
    "cuda_flat_pwa_gen_s0e": (2e-5, 3e-5, 1e-4, False, tuple(QUALIFYING)),
    "cuda_flat_pwa_gen_s0e2": (2e-5, 3e-5, 1e-4, False, tuple(QUALIFYING)),
    "cuda_flat_pwa_gen_s0e3": (2e-5, 3e-5, 1e-4, False, tuple(QUALIFYING)),
    "cuda_flat_pwa_gen_s0e4": (2e-5, 3e-5, 1e-4, False, tuple(QUALIFYING)),
}

#: the tag vocabulary — DECLARED at registration in ``src/tabpwa/backends``,
#: never guessed from a name here.
#:   reference — the NumPy oracle the suite compares against
#:   delegate  — wraps another backend instead of implementing the math
#:   n_res1    — REFUSES a multi-BW chain (``n_res != 1`` raises)
TAG_VOCABULARY = frozenset({"reference", "delegate", "n_res1"})


def backends_to_test():
    """Every registered ``pwa_gen`` backend that IMPLEMENTS the function.

    Derived from the registry and its capability tags: references (the oracle)
    and delegates (a forwarding wrapper — ``auto``/``shard`` — whose own tests
    pin the forwarding) are subtracted, so no name is written out here and a
    backend registered later is covered automatically.
    """
    reg = set(backends_for_model("pwa_gen"))
    ref = set(backends_for_model("pwa_gen", tag="reference"))
    deleg = set(backends_for_model("pwa_gen", tag="delegate"))
    assert ref and deleg, "the tag vocabulary is not declared at registration"
    assert ref <= reg and deleg <= reg
    leaves = reg - ref - deleg
    assert leaves | ref | deleg == reg, (
        "registry names neither tested nor tagged: "
        f"{sorted(reg - leaves - ref - deleg)}")
    assert not ref & deleg
    unknown = {t for n in reg for t in backend_tags(n, "pwa_gen")} - TAG_VOCABULARY
    assert not unknown, f"undeclared tags in the registry: {sorted(unknown)}"
    nres1 = set(backends_for_model("pwa_gen", tag="n_res1"))
    assert nres1 <= leaves, sorted(nres1 - leaves)
    return sorted(leaves)


#: the registry loop — the list is DERIVED, never written out by hand
SPECS = backends_to_test()
#: the multi-BW guard, by tag, so the skip names the capability
N_RES1 = frozenset(backends_for_model("pwa_gen", tag="n_res1"))


def _floor_ok(a, b, rel_bound, abs_bound=CACHE_FLOOR_ABS):
    """The fp32 floor is ABSOLUTE; ``_rel`` divides by the value.

    Accepting either bound keeps the gate meaningful for a near-zero Q without
    loosening it for the ordinary ones (the absolute bound IS the measured
    shared-cache gap).
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    worst = float(np.abs(a - b).max())
    return worst / max(np.abs(b).max(), 1e-300) < rel_bound or worst < abs_bound


def _rel(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _gate(name, a, b, which="rel", cfg=None):
    """The floor for THAT backend on THAT fixture class.

    Inside the configs its own file measured, the backend's own bound applies
    (the dense ones reach 1e-6, the tile family 3e-5 — one family-wide number
    would loosen some of them).  Outside that set, the family's measured
    wide-range floor applies, so the case is still GATED rather than skipped.
    """
    rel, p, coeff, abs_ok, cfgs = FLOORS[name]
    if cfg is not None and cfg not in cfgs:
        return _rel(a, b) < WIDE[which]
    # "ck" shares the generic rel bound; WIDE carries it separately
    bound = {"rel": rel, "p": p, "ck": rel, "coeff": coeff}[which]
    return _floor_ok(a, b, bound) if abs_ok else _rel(a, b) < bound


def _cfgs(name):
    """The config set THIS backend's own anchor was measured on."""
    return FLOORS[name][4]


#: On a fixture OUTSIDE a backend's recorded set the gate is the family's own
#: measured wide-range floor, never a skip: dropping the case would test the
#: backend on nothing at all, and lifting its 1e-6 there would be a gate that
#: cannot pass.  MEASURED over the whole family on ``_ident``/``_cp``,
#: rsize 1/17, worst of norm 1.3 and norm None: Q 9.3e-7, P 3.5e-6,
#: ck 1.4e-6, coeff 5.0e-6 — the 2-5x margins below.  Q and ck are the
#: norm-dependent pair (norm=None is the raw sum, so its gradient is not
#: divided by the normalisation).
WIDE = {"rel": 2e-6, "p": 1e-5, "ck": 5e-6, "coeff": 2e-5}
#: The grouped-objective scenario (34 events, seed 5, rsize 17, un-normalised
#: weights) is its own: MEASURED worst 1.3e-6 across the family -> 4x margin.
GROUPED = 5e-6


assert set(FLOORS) == set(SPECS), (
    "every registered leaf needs its own measured floor - a missing entry "
    f"must not inherit another backend's: {sorted(set(SPECS) - set(FLOORS))}")


def _replicate(data, rs):
    out = {k: np.repeat(np.asarray(v), rs, axis=0) for k, v in data.items()}
    out["weight"] = np.full(len(np.asarray(data["weight"])) * rs, 1.0 / rs)
    return out


def test_backend_tags_are_declared_metadata():
    """Tags are DECLARED at registration — the suite selects on them."""
    assert backend_tags("numpy_flat", "pwa_gen") == frozenset({"reference"})
    assert backend_tags("cuda_flat_pwa_gen_s0c2", "pwa_gen") \
        == frozenset({"n_res1"})
    assert backend_tags("auto", "pwa_gen") == frozenset({"delegate"})
    # an untagged backend returns an empty frozenset, not None
    assert backend_tags("cuda_v4_pwa", "pwa") == frozenset()


@pytest.mark.parametrize("cfg", QUALIFYING)
@pytest.mark.parametrize("norm", [None, 1.3])
@pytest.mark.parametrize("rs", [1, 17])
@pytest.mark.parametrize("name", SPECS)
def test_anchor_matches_numpy_flat(name, cfg, norm, rs):
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    # the sibling fixtures' anchor shape (64 events, seed 0), so the floors
    # below are the same measurements they were calibrated on
    base = _in_grid(kc, 64, 0)
    data = _replicate(base, rs) if rs > 1 else base
    params, _ = f.build_params(f.initial_values(seed=1))
    ref = create_backend("numpy_flat", kc, model="pwa_gen",
                         resolution_size=rs)
    gpu = create_backend(name, kc, model="pwa_gen",
                         resolution_size=rs)
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=norm)
        Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=norm)
        assert _gate(name, Pg, Pr, "p", cfg=cfg), "P"
        assert _gate(name, Qg, Qr, cfg=cfg), "Q"
        assert _gate(name, gg["ck"], gr["ck"], "ck", cfg=cfg), "dck"
        assert _gate(name, gg["coeff"], gr["coeff"], "coeff", cfg=cfg), (
            "dcoeff")
        if norm is not None:
            assert _gate(name, gpu.kernel._last_dnorm, ref._last_dnorm,
                         cfg=cfg), "dnorm"
    finally:
        ref.free()
        gpu.free()


@pytest.mark.parametrize("cfg", QUALIFYING)
@pytest.mark.parametrize("name", SPECS)
def test_one_group_per_event_reduces_to_the_per_event_nll(name, cfg):
    """``rsize = 1`` is the per-event log, so it must match ``rsize = None``.

    Also pins that the GROUPED objective is the flat family's group-TOTAL
    weight form (`-W_g·log(S_g/W_g)`, `numpy_flat`'s), not s0c's plain
    `-log(Pb/norm + Bb)` — the two differ by ``W_g`` whenever a group's weights
    do not sum to 1.  Un-normalised weights make that difference visible.
    """
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    data = _in_grid(kc, 34, 5)              # 34 events: a short final group
    params, _ = f.build_params(f.initial_values(seed=1))
    ref = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=17)
    gpu = create_backend(name, kc, model="pwa_gen",
                         resolution_size=17)
    try:
        Qr, _gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        Qg, _gg, Pg = gpu.compute(params, gpu.load_data(data), norm=1.3)
        assert _rel(Pg, Pr) < GROUPED, "P"
        assert _rel(Qg, Qr) < GROUPED, "Q"
    finally:
        ref.free()
        gpu.free()
    # and with group-NORMALISED weights both forms coincide, which is what
    # makes the per-event value a valid cross-check of the grouped one
    d = _replicate(data, 1)
    w = np.asarray(d["weight"], float)
    n = (w.size // 17) * 17
    wg = w[:n].reshape(-1, 17)
    d["weight"] = w.copy()
    d["weight"][:n] = (wg / wg.sum(axis=1, keepdims=True)).reshape(-1)
    ref = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=17)
    gpu = create_backend(name, kc, model="pwa_gen",
                         resolution_size=17)
    try:
        hr = ref.load_data(d)
        hg = gpu.load_data(d)
        Qr = ref.compute(params, hr, norm=1.3)[0]
        Qg = gpu.compute(params, hg, norm=1.3)[0]
        assert _rel(Qg, Qr) < GROUPED, "Q (normalised weights)"
    finally:
        ref.free()
        gpu.free()


@pytest.mark.parametrize("cfg", [CFG_KM, CFG_IDENT, CFG_CP])
@pytest.mark.parametrize("rs", [1, 17])
@pytest.mark.parametrize("name", SPECS)
def test_coeff_gradient_matches_central_differences(name, cfg, rs):
    """``dcoeff`` must be ∂Q/∂coeff of THIS kernel's Q, per basis row.

    This is the gate that catches a wrong CSR or a wrong m-major
    transposition: a map that included the wrong columns would still be
    repeatable (both computes wrong the same way), so only an independent
    derivative can catch it.
    """
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    base = _in_grid(kc, 48)
    data = _replicate(base, rs) if rs > 1 else base
    gpu = create_backend(name, kc, resolution_size=rs, model="pwa_gen")
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
                f"{cfg} rs={rs} row {i}: kernel {g0['coeff'][i]!r} "
                f"vs FD {fd!r}")
    finally:
        gpu.free()


@pytest.mark.parametrize("cfg", [CFG_KM, CFG_CP])
@pytest.mark.parametrize("name", SPECS)
def test_ck_gradient_matches_fd(name, cfg):
    """``dck`` after the u-merge — the same independent check."""
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    data = _in_grid(kc, 48)
    gpu = create_backend(name, kc, model="pwa_gen")
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


@pytest.mark.parametrize("name", SPECS)
def test_a_multi_chunk_call_equals_a_single_chunk_call(name, ):
    """Batching is an inner loop: the numbers must not know about it.

    Exercised with and without the grouped objective, and at batch sizes that
    straddle both guards (1/5 are below one group at rsize 17; 1e9 is capped).
    """
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    data = _in_grid(kc, 500, 3)

    def run(rs, bs):
        be = create_backend(name, kc, resolution_size=rs, batch_size=bs, model="pwa_gen")
        try:
            h = be.load_data(data)
            Q, g, P = be.compute(params, h, norm=1.3)
            h.free()
            return Q, g["ck"], g["coeff"], float(g["norm"]), np.asarray(P)
        finally:
            be.free()

    for rs in (1, 17):
        base = run(rs, 10 ** 9)             # one chunk
        for bs in (1, 5, 1000, 5000, 50000):
            got = run(rs, bs)
            for label, a, b in (("Q", got[0], base[0]),
                               ("dck", got[1], base[1]),
                               ("dcoeff", got[2], base[2]),
                               ("dnorm", got[3], base[3]),
                               ("P", got[4], base[4])):
                rel = _rel(a, b)
                assert rel <= 1e-14, f"rsize={rs} batch={bs}: {label} {rel:.2e}"


@pytest.mark.parametrize("name", SPECS)
def test_a_chunk_boundary_does_not_move_the_fill(name, ):
    """The load-time fill is chunked too — so a chunk must read ITS events.

    Regression gate: the fill's `event_idx` is chunk-local, so passing the
    sample's first events to every chunk leaves the cache of every chunk after
    the first computed from the wrong masses (measured: Q off by 8.5 on 16
    events at batch_size 1).  Pinned by a sample much longer than the chunk.
    """
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    data = _in_grid(kc, 37, 11)
    k = create_backend(name, kc, batch_size=4, model="pwa_gen")
    h = k.load_data(data)
    k2 = create_backend(name, kc, batch_size=10 ** 6, model="pwa_gen")
    h2 = k2.load_data(data)
    try:
        Qa, _ga, Pa = k.compute(params, h, norm=1.3)
        Qb, _gb, Pb = k2.compute(params, h2, norm=1.3)
        assert _rel(Pa, Pb) <= 1e-14
        assert abs(Qa - Qb) <= 1e-12 * max(abs(Qb), 1.0)
        # a per-event quantity: the LAST events prove the fill ran per chunk
        assert abs(float(Pa[-1]) - float(Pb[-1])) <= 1e-13 * max(
            abs(float(Pb[-1])), 1.0)
    finally:
        k.free()
        k2.free()


@pytest.mark.parametrize("name", SPECS)
def test_two_live_handles_are_independent(name, ):
    """The fitter holds phsp AND data at once — each load owns its arrays."""
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    k = create_backend(name, kc, model="pwa_gen")
    params, _ = f.build_params(f.initial_values(seed=1))
    a = _in_grid(kc, 16, 1)
    b = _in_grid(kc, 16, 2)
    try:
        h1 = k.load_data(a)
        h2 = k.load_data(b)
        assert h1.ptr != h2.ptr
        Qa1, _g, Pa = k.compute(params, h1, norm=None)
        Qb, _g, Pb = k.compute(params, h2, norm=None)
        Qa2 = k.compute(params, h1, norm=None)[0]
        # h2's load did not clobber h1's
        assert abs(Qa1 - Qa2) <= 1e-12 * max(abs(Qa1), 1.0)
        assert _rel(Pa, Pb) > 1e-6            # ... and they are different data
        h1.free()
        h2.free()
    finally:
        k.free()


@pytest.mark.parametrize("name", SPECS)
def test_a_freed_handle_is_reported_not_crashed(name, ):
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    k = create_backend(name, kc, model="pwa_gen")
    h = k.load_data(_in_grid(kc, 8))
    params, _ = f.build_params(f.initial_values(seed=1))
    try:
        h.free()
        h.free()                              # idempotent
        with pytest.raises((RuntimeError, ValueError),
                           match="already freed"):
            k.compute(params, h, norm=None)
    finally:
        k.free()


@pytest.mark.parametrize("name", SPECS)
def test_rejects_an_already_deduped_config(name, ):
    from tabpwa.pwa_gen_config import build_from_groups, dedup
    m = Fitter(CFG_KM).model
    dd = dedup(build_from_groups(m, n_interp=200))
    with pytest.raises(ValueError, match="FULL"):
        create_backend(name, dd, model="pwa_gen")


@pytest.mark.parametrize("name", SPECS)
def test_wrong_parameter_lengths_are_rejected(name, ):
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    k = create_backend(name, kc, model="pwa_gen")
    h = k.load_data(_in_grid(kc, 8))
    params, _ = f.build_params(f.initial_values(seed=1))
    try:
        bad = dict(params)
        bad["ck"] = np.zeros(k.kernel.n_wave + 1, complex)
        with pytest.raises(ValueError, match="n_wave"):
            k.compute(bad, h, norm=None)
        bad = dict(params)
        bad["coeff"] = np.zeros(k.kernel.n_basis + 1)
        with pytest.raises(ValueError, match="n_basis"):
            k.compute(bad, h, norm=None)
    finally:
        k.free()


@pytest.mark.parametrize("name", SPECS)
def test_resolution_size_below_one_is_rejected(name, ):
    f = Fitter(CFG_KM)
    f.apply_constrains()
    with pytest.raises(ValueError, match="resolution_size"):
        create_backend(name, f.kernel_config, resolution_size=0, model="pwa_gen")

@pytest.mark.parametrize("name", SPECS)
def test_resolution_size_one_is_the_per_event_nll(name, ):
    """`rsize = 1` is the default AND the per-event objective."""
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    data = _in_grid(kc, 48, 4)
    params, _ = f.build_params(f.initial_values(seed=1))
    dflt = create_backend(name, kc, model="pwa_gen")
    one = create_backend(name, kc, model="pwa_gen", resolution_size=1)
    ref = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=1)
    try:
        Qa, _ga, Pa = dflt.compute(params, dflt.load_data(data), norm=1.3)
        Qb, _gb, Pb = one.compute(params, one.load_data(data), norm=1.3)
        Qr, _gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        assert _rel(Qa, Qb) <= 1e-14
        assert _rel(Pa, Pb) <= 1e-14
        assert _gate(name, Qb, Qr) and _gate(name, Pb, Pr, "p")
    finally:
        dflt.free()
        one.free()
        ref.free()


@pytest.mark.parametrize("name", SPECS)
def test_norm_none_and_a_scalar_norm_are_different_objectives(name, ):
    """`norm=None` is the raw sum; `norm=v` is the normalised NLL."""
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    data = _in_grid(kc, 48, 6)
    params, _ = f.build_params(f.initial_values(seed=1))
    be = create_backend(name, kc, model="pwa_gen")
    ref = create_backend("numpy_flat", kc, model="pwa_gen")
    try:
        h = be.load_data(data)
        hr = ref.load_data(data)
        Qn = be.compute(params, h, norm=None)[0]
        Qv = be.compute(params, h, norm=1.3)[0]
        Rn = ref.compute(params, hr, norm=None)[0]
        Rv = ref.compute(params, hr, norm=1.3)[0]
        assert _gate(name, Qn, Rn) and _gate(name, Qv, Rv)
        assert abs(Qn - Qv) > 1e-9 * max(abs(Qn), 1.0)
    finally:
        be.free()
        ref.free()


@pytest.mark.parametrize("name", SPECS)
def test_multi_bw_chain_is_rejected_or_accepted(name, ):
    """`n_res1`-tagged backends guard it; every other leaf takes the chain."""
    if name in N_RES1:
        pytest.skip(f"{name}: tagged n_res1 — its wrapper REFUSES a multi-BW "
                    f"chain (n_res != 1 raises), and its own file keeps the "
                    f"'raises, names cuda_flat_pwa_gen2' gate")
    f = Fitter(CFG_MB)
    f.apply_constrains()
    assert int(np.asarray(f.kernel_config["num_table"]).shape[1]) == 2
    create_backend(name, f.kernel_config, model="pwa_gen").free()
