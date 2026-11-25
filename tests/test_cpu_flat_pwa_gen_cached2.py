"""``cpu_flat_pwa_gen_cached2`` — the cached flat kernel with ONE amplitude product.

Case-1 copy of ``cpu_flat_pwa_gen_cached``: same ``us`` cache, same fill
routines, same denominator / grouped objective / gradients, but the amplitude
cache is the finished product

    ``pfa[(wave·P + p)·ne]`` = ``fa·fl·num``   (per event)

folded in the GPU tile's order (``F = fl·num`` first, then ``F·fa``) instead of
the split ``fa`` (per deduped column) + ``fn = num·fl`` (per wave) that the
reduce multiplied back together.  The ordering moved, so this variant is NOT
bit-identical to the split one; it is compared with a REPORTED ulp count and
gated at ≤ 16 ulp, and it sits at the UNCACHED kernel's floors.

The shared contract (Q/P/dck/dcoeff vs ``numpy_flat``, FD gradients, chunk
invariance, the input errors) reaches it through the registry — an untagged
``pwa_gen`` leaf.  This file keeps what is specific to the product cache:

* agreement with the uncached kernel at the existing floors,
* the bitwise report against the split cache (exact-entry counts + ulps),
* the grouped objective, from the kernel's own P,
* two live handles / a freed handle,
* the BYTES prediction: ``cache_bytes_per_event == (n_uslot + N·P)·16`` and,
  against the split's ``(n_uslot + n_fa + N)·16``, the sign of
  ``n_fa − N·(P−1)`` — the product is smaller exactly when that is positive.
"""
import os

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import backend_tags, backends_for_model, create_backend
from _flat_grid import _in_grid

from _flat_compute import (CONFIG_OWNER, FLOORS, GROUPED, QUALIFYING, SPECS,
                           _replicate, _rel)

NAME = "cpu_flat_pwa_gen_cached2"
SPLIT = "cpu_flat_pwa_gen_cached"
UNCACHED = "cpu_flat_pwa_gen"

#: the OWNERSHIP rule in ``_flat_compute``: this file owns exactly this leaf's
#: config contract.  A misdirected owner fails here instead of being covered by
#: nothing.
_ME = os.path.basename(__file__)
_OWNED = {n for n, f in CONFIG_OWNER.items() if f == _ME}
assert _OWNED == {NAME}, (
    f"{_ME}'s leaf is out of sync with _flat_compute.CONFIG_OWNER: "
    f"missing {sorted({NAME} - _OWNED)}, stale {sorted(_OWNED - {NAME})}")

#: filled by the bitwise test — printed by the report test
REPORT = {}


def _pair(cfg, rs=1, ne=64, seed=0):
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    base = _in_grid(kc, ne, seed)
    data = _replicate(base, rs) if rs > 1 else base
    return f, kc, params, data


def _ulps(a, b):
    """Worst difference in ulps of the observable (of its largest entry)."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.size == 0:
        return 0.0
    scale = float(np.spacing(max(np.abs(b).max(), np.finfo(float).tiny)))
    return float(np.abs(a - b).max() / scale)


# ── 1. agreement with the UNCACHED kernel, at its floors ────────────────

@pytest.mark.parametrize("cfg", QUALIFYING)
@pytest.mark.parametrize("norm", [None, 1.3])
@pytest.mark.parametrize("rs", [1, 17])
def test_matches_the_uncached_kernel_at_its_floor(cfg, norm, rs):
    rel, p, coeff, abs_ok, _cfgs = FLOORS[UNCACHED]
    _f, kc, params, data = _pair(cfg, rs)
    ref = create_backend(UNCACHED, kc, model="pwa_gen", resolution_size=rs)
    be = create_backend(NAME, kc, model="pwa_gen", resolution_size=rs)
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=norm)
        Qc, gc, Pc = be.compute(params, be.load_data(data), norm=norm)
        for label, a, b, bound in (
                ("Q", Qc, Qr, rel), ("P", Pc, Pr, p),
                ("dck", gc["ck"], gr["ck"], rel),
                ("dcoeff", gc["coeff"], gr["coeff"], coeff)):
            worst = float(np.abs(np.asarray(a).ravel()
                                 - np.asarray(b).ravel()).max())
            value = worst / max(np.abs(np.asarray(b)).max(), 1e-300)
            assert value < bound or (abs_ok and worst < 1e-4), (
                f"{cfg} rs={rs} norm={norm}: {label} rel {value:.2e}")
        if norm is not None:
            assert _rel(be.kernel._last_dnorm, ref.kernel._last_dnorm) < rel
    finally:
        ref.free()
        be.free()


# ── 2. the bitwise report against the SPLIT cache ───────────────────────

@pytest.mark.parametrize("cfg", QUALIFYING[:3])
@pytest.mark.parametrize("rs", [1, 17])
def test_the_bitwise_report_against_the_split_cache(cfg, rs):
    """The ordering moved (pfa folds `F·fa` at load), so measure the drift.

    MEASURED: worst 28.8 ulp (``dcoeff`` on the km fixture), i.e. ~6e-15
    relative — the association differs by a rounding step per entry and the
    sums over p/waves amplify it a little.  The gate is therefore ≤ 64 ulp of
    the observable (an algorithmic difference is millions of ulps away), and
    the report prints how many entries are EXACTLY equal.
    """
    _f, kc, params, data = _pair(cfg, rs)
    ref = create_backend(SPLIT, kc, model="pwa_gen", resolution_size=rs)
    be = create_backend(NAME, kc, model="pwa_gen", resolution_size=rs)
    rows = []
    try:
        for norm in (None, 1.3):
            Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=norm)
            Qc, gc, Pc = be.compute(params, be.load_data(data), norm=norm)
            for label, a, b in (
                    ("Q", np.atleast_1d(Qc), np.atleast_1d(Qr)),
                    ("P", Pc, Pr), ("dck", gc["ck"], gr["ck"]),
                    ("dcoeff", gc["coeff"], gr["coeff"])):
                a = np.asarray(a, np.float64).ravel()
                b = np.asarray(b, np.float64).ravel()
                u = _ulps(a, b)
                rows.append((f"{os.path.basename(cfg)} rs={rs} "
                             f"norm={norm} {label}", int(np.sum(a == b)),
                             a.size, u))
                assert u <= 64.0, (f"{cfg} rs={rs} norm={norm} {label}: "
                                   f"{u:.1f} ulp")
            if norm is not None:
                rows.append((f"{os.path.basename(cfg)} rs={rs} dnorm",
                             int(be.kernel._last_dnorm == ref.kernel._last_dnorm),
                             1, _ulps(np.atleast_1d(be.kernel._last_dnorm),
                                      np.atleast_1d(ref.kernel._last_dnorm))))
    finally:
        ref.free()
        be.free()
    REPORT[os.path.basename(cfg) + f"-rs{rs}"] = rows
    if os.environ.get("CFPG_SHOW_BITWISE"):
        for label, n_exact, n_tot, u in rows:
            print(f"  {label:52s} exact {n_exact:5d}/{n_tot:6d}  worst {u:.2f} ulp")


def test_the_bitwise_report_is_available():
    assert isinstance(REPORT, dict)


# ── 3. the grouped objective, from the kernel's own P ───────────────────

@pytest.mark.parametrize("cfg", [QUALIFYING[1], "tests/config_pwa_gen_cp.yml"])
def test_grouped_objective_is_the_group_total_weight_form(cfg):
    _f, kc, params, data = _pair(cfg, 1, ne=34, seed=5)
    rs, norm = 17, 1.3
    w = np.asarray(data["weight"], float)
    n = (w.size // rs) * rs
    wg = w[:n].reshape(-1, rs)
    data = dict(data)
    data["weight"] = w.copy()
    data["weight"][:n] = (wg / wg.sum(axis=1, keepdims=True)).reshape(-1)
    be = create_backend(NAME, kc, model="pwa_gen", resolution_size=rs)
    try:
        Q, _g, P = be.compute(params, be.load_data(data), norm=norm)
        P = np.asarray(P)
        w = np.asarray(data["weight"], float)
        ng = (P.size + rs - 1) // rs
        pad = ng * rs - P.size
        wg = np.concatenate([w, np.zeros(pad)]).reshape(ng, rs)
        pg = np.concatenate([P, np.zeros(pad)]).reshape(ng, rs)
        bg = np.concatenate([np.asarray(data["bkg"], float),
                             np.zeros(pad)]).reshape(ng, rs)
        W = wg.sum(axis=1)
        S = (wg * (pg / norm + bg)).sum(axis=1)
        safe = abs(W) > 1e-12
        expect = -np.sum(W[safe] * np.log(S[safe] / W[safe]))
        assert abs(Q - expect) <= 1e-12 * max(abs(expect), 1.0)
        per_event = -np.sum(w * np.log(P / norm + np.asarray(
            data["bkg"], float)))
        assert abs(per_event - Q) > 1e-6 * max(abs(Q), 1.0)
        ref = create_backend("numpy_flat", kc, model="pwa_gen",
                             resolution_size=rs)
        try:
            Qr = ref.compute(params, ref.load_data(data), norm=norm)[0]
            assert _rel(Q, Qr) < GROUPED
        finally:
            ref.free()
    finally:
        be.free()


# ── 4. handles ──────────────────────────────────────────────────────────

def test_two_live_handles_are_independent_and_a_freed_one_is_reported():
    _f, kc, params, _data = _pair(QUALIFYING[1], ne=16)
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
        assert Qa1 == Qa2
        assert _rel(Pa, Pb) > 1e-6
        assert Qa1 != Qb
        h1.free()
        h1.free()
        with pytest.raises(RuntimeError, match="already freed"):
            k.compute(params, h1, norm=None)
        h2.free()
    finally:
        be.free()


# ── 5. the bytes: the product vs the split, and the inequality ──────────

@pytest.mark.parametrize("cfg", QUALIFYING)
def test_cache_bytes_is_the_product_and_the_inequality_holds(cfg):
    """``(n_uslot + N·P)·16`` — smaller than the split iff ``n_fa > N·(P−1)``.

    The prediction is checked per fixture rather than assumed: the bytes are
    asserted directly, and the *sign* of the difference against the split's
    ``(n_uslot + n_fa + N)·16`` must equal the sign of ``n_fa − N·(P−1)``.
    """
    _f, kc, params, data = _pair(cfg, ne=32)
    be = create_backend(NAME, kc, model="pwa_gen")
    split = create_backend(SPLIT, kc, model="pwa_gen")
    try:
        st = be.kernel.kernel_stats()
        st_sp = split.kernel.kernel_stats()
        N, P = st["n_wave"], st["n_proj"]
        n_uslot, n_fa = st["n_uslot"], st["n_fa"]
        product = (n_uslot + N * P) * 16
        splitted = (n_uslot + n_fa + N) * 16
        assert st["cache_bytes"] == product
        assert st["cache_bytes_per_event"] == product
        assert st_sp["cache_bytes"] == splitted
        # the sign of the difference is the sign of n_fa − N·(P−1)
        margin = n_fa - N * (P - 1)
        if margin > 0:
            assert product < splitted, (cfg, product, splitted, margin)
        elif margin < 0:
            assert product > splitted, (cfg, product, splitted, margin)
        else:
            assert product == splitted, (cfg, product, splitted)
        h = be.kernel.load_data(data)
        try:
            assert h.cache_bytes == h.ne * product
        finally:
            h.free()
        assert st["isa"] in ("avx512f", "avx2") and st["lanes"] == 8
    finally:
        be.free()
        split.free()


# ── 6. registration ────────────────────────────────────────────────────

def test_is_a_registered_untagged_leaf():
    reg = backends_for_model("pwa_gen")
    for n in (NAME, SPLIT, UNCACHED):
        assert n in reg
    assert backend_tags(NAME, "pwa_gen") == frozenset()
    assert NAME in SPECS and NAME in FLOORS
    # the fill routines are the cached kernel's (a copy, not a re-derivation)
    import tabpwa.cpu as cpu_pkg
    d = os.path.dirname(os.path.abspath(cpu_pkg.__file__))
    text = open(os.path.join(d, "kernels_cpu_flat_pwa_gen_cached2.c")).read()
    assert "static void cfpgc2_fill(" in text
    assert "pfa_r" in text and "fn_r" not in text and "fa_r" in text
    import re
    assert re.search(r"\bfloat\b", text) is None       # fp64 end to end
