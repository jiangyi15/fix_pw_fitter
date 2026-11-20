"""``cpu_flat_pwa_gen_cached`` — the flat ``pwa_gen`` kernel with LOAD-TIME caches.

Case-1 copy of ``cpu_flat_pwa_gen`` (own ``.c``, own ``.so``, own symbols):
the same algorithm and the same fp64 arithmetic, plus the two caches the GPU
tile keeps in VRAM (``us`` = the basis rows at the deduped slots, ``amp`` =
``fa·fl·num``) built once per ``load_data`` — here as three fp64 arrays
(``us``, ``fa``, ``fn = num·fl``), because caching the combined product would
associate the multiplications differently and break bit-identity.

The SHARED contract (Q/P/dck/dcoeff vs ``numpy_flat``, FD gradients, chunk
invariance, the input errors) reaches this backend through the registry
(``tests/_flat_compute.py``) because it is an untagged ``pwa_gen`` leaf.  This
file keeps what is specific to the caches:

* **bit-identity** against the uncached ``cpu_flat_pwa_gen`` — the caches are
  the same code, so this must be EXACT (``np.array_equal``), 3 fixtures ×
  rsize 1/17 × norm None/1.3;
* the grouped objective, recomputed from the kernel's own P;
* two live handles independent / a freed handle reported;
* ``cache_bytes`` = ``(n_uslot + n_fa + n_wave)·16`` per event (1600 B on the
  production shape — twice the GPU's ``float2`` figure because this is fp64).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tabpwa import Fitter
from tabpwa.backends import backend_tags, backends_for_model, create_backend
from test_cuda_flat_cache_pwa_gen import _in_grid

from _flat_compute import (CFG_KM, FLOORS, GROUPED, QUALIFYING, SPECS, _rel,
                           _replicate)

NAME = "cpu_flat_pwa_gen_cached"
NAME_UNCACHED = "cpu_flat_pwa_gen"
#: the fixtures that qualify (one BW per chain) — the mb fixture is n_res == 2
IDENTITY_CFGS = [QUALIFYING[0], CFG_KM, "tests/config_pwa_gen_mb.yml"]


def _pair(cfg, rs=1, ne=64, seed=0):
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    base = _in_grid(kc, ne, seed)
    data = _replicate(base, rs) if rs > 1 else base
    return f, kc, params, data


# ── 1. bit-identity against the uncached kernel ─────────────────────────

@pytest.mark.parametrize("cfg", IDENTITY_CFGS)
@pytest.mark.parametrize("rs", [1, 17])
@pytest.mark.parametrize("norm", [None, 1.3])
def test_bit_identical_to_the_uncached_kernel(cfg, rs, norm):
    """The caches are the same code, so the numbers must be EXACT.

    Not a tolerance: `np.array_equal` on Q, P, dck, dcoeff and dnorm.  This
    is the gate that pins the cache contents (``fn = num·fl`` rather than the
    combined product) and the fact that the fill's interpolation is the same
    `cfpgc_bin`/`cfpgc_interp` the uncached tile calls.
    """
    _f, kc, params, data = _pair(cfg, rs)
    ref = create_backend(NAME_UNCACHED, kc, model="pwa_gen",
                         resolution_size=rs)
    cut = create_backend(NAME, kc, model="pwa_gen", resolution_size=rs)
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=norm)
        Qc, gc, Pc = cut.compute(params, cut.load_data(data), norm=norm)
        assert Qr == Qc, f"{cfg} rs={rs} norm={norm}: Q {Qr!r} vs {Qc!r}"
        assert np.array_equal(Pr, Pc), "P"
        assert np.array_equal(gr["ck"], gc["ck"]), "dck"
        assert np.array_equal(gr["coeff"], gc["coeff"]), "dcoeff"
        if norm is not None:
            assert ref.kernel._last_dnorm == cut.kernel._last_dnorm, "dnorm"
    finally:
        ref.free()
        cut.free()


@pytest.mark.parametrize("rs", [1, 17])
def test_bit_identical_over_a_longer_sample(rs):
    """A sample longer than one tile, and a short final group (37 events)."""
    _f, kc, params, data = _pair(CFG_KM, rs, ne=37, seed=11)
    ref = create_backend(NAME_UNCACHED, kc, model="pwa_gen", resolution_size=rs)
    cut = create_backend(NAME, kc, model="pwa_gen", resolution_size=rs)
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        Qc, gc, Pc = cut.compute(params, cut.load_data(data), norm=1.3)
        assert Qr == Qc and np.array_equal(Pr, Pc)
        assert np.array_equal(gr["ck"], gc["ck"])
        assert np.array_equal(gr["coeff"], gc["coeff"])
        assert ref.kernel._last_dnorm == cut.kernel._last_dnorm
    finally:
        ref.free()
        cut.free()


# ── 2. the grouped objective, from the kernel's own P ───────────────────

@pytest.mark.parametrize("cfg", [CFG_KM, "tests/config_pwa_gen_cp.yml"])
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
        safe = W > 0
        expect = -np.sum(W[safe] * np.log(S[safe] / W[safe]))
        assert abs(Q - expect) <= 1e-12 * max(abs(expect), 1.0)
        # ... and it is NOT the per-event log (a Jensen shift)
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


# ── 3. handles + the cache accounting ───────────────────────────────────

def test_two_live_handles_are_independent_and_a_freed_one_is_reported():
    """The fitter holds phsp AND data at once — each handle owns its caches."""
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
        assert Qa1 == Qa2                  # h2's load did not clobber h1
        assert _rel(Pa, Pb) > 1e-6         # ... and they are different data
        assert Qa1 != Qb
        h1.free()
        h1.free()                          # idempotent
        with pytest.raises(RuntimeError, match="already freed"):
            k.compute(params, h1, norm=None)
        h2.free()
    finally:
        be.free()


def test_cache_bytes_is_the_three_fp64_arrays():
    """``(n_uslot + n_fa + n_wave)·16`` B/event, and the handle owns it all.

    On the production shape (n_uslot 48, n_fa 26, n_wave 26) that is
    1600 B/event — 2x the GPU's ``float2`` figure (854 B/event measured on
    `s0e`) because every cached value here is fp64; the fp64 storage is what
    makes the bit-identity gate pass.
    """
    _f, kc, params, data = _pair(CFG_KM, ne=32)
    be = create_backend(NAME, kc, model="pwa_gen")
    k = be.kernel
    try:
        st = k.kernel_stats()
        assert st["isa"] in ("avx512f", "avx2")
        assert st["lanes"] == 8
        per_event = (st["n_uslot"] + st["n_fa"] + st["n_wave"]) * 16
        assert st["cache_bytes"] == per_event
        assert per_event > 0                      # the caches exist ...
        h = k.load_data(data)
        try:
            assert h.cache_bytes == h.ne * per_event
            assert h.cache_bytes > 0
        finally:
            h.free()
        # ... whereas the uncached leaf caches nothing at all
        ref = create_backend(NAME_UNCACHED, kc, model="pwa_gen")
        try:
            assert ref.kernel.kernel_stats()["cache_bytes"] == 0
        finally:
            ref.free()
    finally:
        be.free()


def test_is_a_registered_untagged_leaf():
    reg = backends_for_model("pwa_gen")
    assert NAME in reg and NAME_UNCACHED in reg
    assert backend_tags(NAME, "pwa_gen") == frozenset()
    assert NAME in SPECS and NAME in FLOORS


def test_is_a_standalone_case1_copy():
    """Its own `.c`/`.so`/symbols; the uncached kernel is untouched."""
    import tabpwa.cpu as cpu_pkg
    d = os.path.dirname(os.path.abspath(cpu_pkg.__file__))
    new = os.path.join(d, "kernels_cpu_flat_pwa_gen_cached.c")
    old = os.path.join(d, "kernels_cpu_flat_pwa_gen.c")
    assert os.path.exists(new) and os.path.exists(old)
    text = open(new).read()
    # the tile is SPLIT here (fill at load, reduce per iteration) — the only
    # structural change of this case-1 copy
    assert "cfpgc_tile(" not in text
    assert "static void cfpgc_fill(" in text
    assert "static void cfpgc_reduce(" in text
    assert "cpu_flat_pwa_gen_cached_compute" in text
    # no fp32 type anywhere, and no header shared with the parent
    import re
    assert re.search(r"\bfloat\b", text) is None
    assert "kernels_cpu_flat_pwa_gen.h" not in text
    # it does not import the parent's module (a copy is standalone)
    assert "from ._flat_pwa_gen" not in text and "import _flat_pwa_gen" not in text
