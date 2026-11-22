"""The FIXED-COEFFICIENT (mask) contract every flat ``pwa_gen`` backend shares.

A backend that accepts ``fixed_cols`` + ``fixed_values`` promises one thing:
a wave whose every ``coeff`` column is declared fixed is iteration-invariant,
so its whole contribution can be cached per handle — and the numbers must not
move.  The tests for what that promise MEANS are written once here,
parametrised over the REGISTRY: ``SPECS_MASKED`` is
``backends_for_model("pwa_gen", tag="fixed_coeff")``, so a backend registered
later with the tag is covered without touching this file.  No backend name is
written out: the selection is the capability, declared at registration.

Shared (the contract):

1. mask-on vs the fp64 ``numpy_flat`` reference — Q/P/dck over the qualifying
   fixtures x ``rsize {1, 17}`` x ``norm {None, 1.3}``, ``dcoeff`` on the FREE
   columns, at the STRICT ``_rel`` floors of ``_flat_compute.FLOORS`` (an
   absolute escape would accept a mask bug on a near-zero ``P``);
2. ``grads["coeff"] == 0.0`` EXACTLY at every column a CACHED wave reads;
3. a partly fixed wave keeps its ``D``: still differentiated, its wave is not
   cached, and ``n_fix`` does not change;
4. the ``FIXED_CHANGED`` guard — a declared-fixed value that moves raises with
   the kernel's exact message (and launches nothing), a free column does not;
5. the fill runs in ``load_data`` and the guard runs from the FIRST compute
   (there is no snapshot): ``afix_ready`` is true for every handle right after
   the load, and a fresh backend's first compute already rejects a moved value;
6. the declaration is ONE thing — columns + values — and both levels (kernel
   constructor and the ``fixed_coeff`` wrapper) refuse a mask without values,
   values without a mask, an under-covering ``{col: value}`` map and a
   length-``n_basis`` 0/1 vector; ``set_fixed_cols(None)`` clears;
7. ``set_fixed_coeffs(fitter)`` == the explicit ``(cols, values)``;
8. a mask that caches nothing warns ONCE, naming the wave and the columns it
   still reads; a PARTIAL mask is silent (run under ``simplefilter("error")``);
9. ``fix_bytes`` is ``8·n_proj·n_fix·ne`` per handle, and a mask change
   re-loads each retained sample exactly once;
10. masked chunk invariance — multi-chunk == single-chunk, ``P`` EXACT;
11. a short final group on the masked path;
12. ``return_p=False`` does not change the objective;
13. an empty sample is the identity (and its cache is READY);
14. ``get_nll`` end-to-end through the ``fixed_coeff`` wrapper with two live
    handles, and the masked columns contribute EXACTLY nothing to the reduced
    ``x`` gradient;
15. **no mask means the parent** — with nothing declared the numbers must be
    ``cuda_flat_pwa_gen_s0e2``'s, ``P`` EXACTLY (a different oracle from the
    family floor, so it is a real check of the off switch);
16. **a group no block can hold** — a ``resolution_size`` larger than the
    biggest tile must either compute the grouped objective correctly or REFUSE
    at construction naming a usable alternative, never silently mis-close it;
17. the per-PARAMETER alias map (one mass moves BOTH charge sectors' columns)
    and that each caching kernel HAS a case-1 row (the property itself — own
    module/``.so``/symbols, the parent source byte-identical — is the shared
    ``tests/_flat_isolation.py``);
18. the split's scratch belongs to the CONTEXT, not to a handle — a
    ``load_data`` adds no ``ne``-sized buffer (the measured 336-MB-per-handle
    regression).  Measured in exact BYTES with :func:`tabpwa.monitor.free_bytes`
    (the sampler's MiB granularity cannot see it), DIFFERENTIALLY against the
    parent kernel on a large sample: the legitimate ``us``/``amp`` caches cancel
    and the documented ``fix_bytes`` cache is subtracted, so only a per-handle
    buffer is left.  A tiny sample cannot gate this (both loads cost 0 B at 16
    events); a kernel with no scratch has nothing to allocate and the gate says
    so explicitly rather than passing vacuously.
19. **a multi-BW chain is REFUSED** — every kernel here is one BW per chain, so
    a chain whose amplitude is a PRODUCT of two BWs is rejected at construction
    and the message names the documented fallback ``cuda_flat_pwa_gen2`` (a
    bare ``ValueError`` would let the recommendation rot away unnoticed).

A test here is ``(kernel config, params, sample) -> numbers``, plus the
refusals and the guard.  It is NOT an audit of which branch ran: the
``split_counts`` / ``fused_count`` / ``tile_count`` / ``chunk_plan`` / ``te`` /
``smem_bytes`` reports of a particular kernel are implementation detail, and no
gate is written on them.  ``aux_bytes`` is the one exception, and only as the
SIZE the load may not re-allocate (gate 18) — never as a branch report.

``test_flat_masked.py`` imports this module so pytest collects the suite under
``[<backend name>]`` ids.
"""
import functools
import importlib
import json
import math
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

from tabpwa import Fitter, monitor
from tabpwa.backends import (backend_tags, backends_for_model,
                             create_backend)
from tabpwa.fixed_coeff_spec import MaskNotEngaged

from _flat_compute import (CFG_CP, CFG_IDENT, CFG_KM, CFG_MB, CFG_ONE, CFG_SAME,
                           CONFIG_OWNER, FLOORS, QUALIFYING, SPECS as LEAVES,
                           TAG_VOCABULARY, _gate, _rel, _replicate,
                           other_gpu_users)
from _flat_grid import _in_grid
#: the ONE case-1 table; this file only asserts COVERAGE (a ``fixed_coeff``
#: backend without a row fails at import) — the property runs in
#: ``test_flat_isolation.py``.
from _flat_isolation import ISOLATION

WRAP = "fixed_coeff"
REF = "numpy_flat"

#: the mask-OFF oracle: with nothing declared, a masked kernel must be the
#: parent's path, i.e. agree with ``cuda_flat_pwa_gen_s0e2``.  ``P`` has one
#: writer per event, so it is compared EXACTLY; Q/dck/dcoeff/dnorm are
#: ``atomicAdd``-ordered across blocks, so they are gated at 1e-14 — the
#: STRICTER of the two figures the old per-kernel fixtures measured (s0f 1e-14,
#: s0g 1e-13).  Measured on this matrix the worst quantity is 3.3e-15, so the
#: stricter bound keeps a 3x margin and s0g's looser one is not carried over.
PARENT_TOL = 1e-14

#: the masks this suite runs.  Each ``(config, columns)`` pair is a REAL
#: declaration for that fixture: the fixtures' waves each read a contiguous
#: band of basis columns (``gamma_matrix``), and the pair either fixes whole
#: waves (``n_free == 0`` where the fixture has one wave) or leaves a MIXED
#: fixed/free wave axis.  The columns are chosen from the fixture's own rows,
#: never from a name prefix.
MASKED = [
    (CFG_ONE, [0, 1, 2]),                              # 1 wave, all fixed
    (CFG_KM, [0, 1, 2, 3, 4]),                         # 2 waves, all fixed
    (CFG_IDENT, list(range(10))),                      # 4 waves, ALL fixed
    (CFG_IDENT, [5, 6, 7, 8, 9]),                      # waves 2,3 fixed / 0,1 free
    (CFG_SAME, [5, 6, 7, 8, 9]),                       # the same, on `_same`
    (CFG_CP, [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]),     # 4 fixed / 4 free, CP blocks
]
MASK_IDS = [f"{cfg.split('/')[-1]}-{len(cols)}" for cfg, cols in MASKED]

#: every mask runs on a fixture whose floor the backend's own file MEASURED —
#: a mask on an unmeasured shape would be gated at a borrowed number
assert {cfg for cfg, _cols in MASKED} <= set(QUALIFYING)


def backends_to_test():
    """Every registered ``pwa_gen`` backend that honours a coefficient mask.

    Derived from the registry and its DECLARED tags — the same shape as
    ``_flat_compute.backends_to_test``: the set must be non-empty, must be a
    subset of the flat leaves, every selected name needs a measured floor in
    ``_flat_compute.FLOORS`` (so a new backend cannot inherit another's), and
    the tag vocabulary must cover what the registry carries.
    """
    masked = set(backends_for_model("pwa_gen", tag="fixed_coeff"))
    leaves = set(LEAVES)
    assert masked, "no pwa_gen backend declares the `fixed_coeff` tag"
    assert masked <= leaves, (
        f"tagged masked but not a flat leaf: {sorted(masked - leaves)}")
    assert masked <= set(FLOORS), (
        "a masked backend needs its own measured floor in "
        f"_flat_compute.FLOORS: {sorted(masked - set(FLOORS))}")
    unknown = {t for n in backends_for_model("pwa_gen")
               for t in backend_tags(n, "pwa_gen")} - TAG_VOCABULARY
    assert not unknown, f"undeclared tags in the registry: {sorted(unknown)}"
    return sorted(masked)


#: the registry loop — the list is DERIVED, never written out by hand
SPECS_MASKED = backends_to_test()

#: the OWNERSHIP rule in ``_flat_compute``: this module owns the config contract
#: of every ``fixed_coeff`` leaf (it is the one that runs the mask contract over
#: the tag), so the owner map must say exactly that.  A misdirected owner — say
#: ``cuda_flat_pwa_gen_s0g`` reassigned to another family file — makes this fail
#: instead of silently leaving the name covered by nothing.
_ME = os.path.basename(__file__)
_OWNED = {n for n, f in CONFIG_OWNER.items() if f == _ME}
assert _OWNED == set(SPECS_MASKED), (
    f"{_ME}'s backends are out of sync with _flat_compute.CONFIG_OWNER: "
    f"missing {sorted(set(SPECS_MASKED) - _OWNED)}, "
    f"stale {sorted(_OWNED - set(SPECS_MASKED))}")

#: the CTPWA benchmark's ten free mass/width parameters (benchmarks/ctpwa_kketa
#: README §4); every other mass/width is pinned, as ``bench.py --free-mw`` does
CTPWA_CFG = "benchmarks/ctpwa_kketa/config_gen.yml"
CTPWA_FREE_MW = ("phi1680_mass", "phi1680_width", "X1750_mass", "X1750_width",
                 "omega2220_mass", "omega2220_width", "K3st1780p_mass",
                 "K3st1780p_width", "K4st2045p_mass", "K4st2045p_width")


# ── helpers ────────────────────────────────────────────────────────────

def _params(f):
    return f.build_params(f.initial_values(seed=1))[0]


def _fv(kc, params, cols):
    """The full ``coeff`` vector at the declared columns.

    A mask is declared WITH its values (the cache is filled at ``load_data``),
    so every masked backend in this suite is built with these.
    """
    n_basis = int(np.asarray(kc["basis_table"]).shape[0])
    coeff = np.asarray(params["coeff"], np.float64)
    out = np.zeros(n_basis, np.float64)
    c = np.asarray(list(cols), np.int64).ravel()
    out[c] = coeff[c]
    return out


def _masked(name, cfg, cols, rs=1, f=None, **kw):
    """``(fitter, kernel_config, masked backend)`` on one config/mask."""
    if f is None:
        f = Fitter(cfg)
        f.apply_constrains()
    kc = f.kernel_config
    be = create_backend(name, kc, model="pwa_gen", resolution_size=rs,
                        fixed_cols=list(cols),
                        fixed_values=_fv(kc, _params(f), cols), **kw)
    return f, kc, be


def _references(kc, rs=1):
    return create_backend(REF, kc, model="pwa_gen", resolution_size=rs)


def _wrapper(name, kc, **kw):
    return create_backend({"name": WRAP, "base": name}, kc, model="pwa_gen",
                          **kw)


def _gamma_2d(kc, n_wave):
    """``gamma_matrix`` as one row per wave (both masked kernels are n_res1)."""
    g = np.asarray(kc["gamma_matrix"], np.float64)
    g = g.reshape(-1, g.shape[-1])
    assert g.shape[0] == n_wave, (
        f"n_res != 1: {g.shape[0]} rows for {n_wave} waves")
    return g


def _cached_columns(k, gam):
    """Every column a CACHED wave reads — from the kernel's own wave list."""
    cols = set()
    for w in np.asarray(k.fixed_waves, np.int64).ravel():
        cols.update(int(c) for c in np.nonzero(gam[w])[0])
    return np.asarray(sorted(cols), np.int64)


def _fixed_changed(be):
    """The kernel's own ``FIXED_CHANGED`` text (derived, never written out)."""
    mod = importlib.import_module(type(be.kernel).__module__)
    return mod.FIXED_CHANGED


def _moving_by_param(f, x, step=1e-3):
    """``{free parameter: [coeff columns its perturbation moves]}``.

    The ground truth for a mask: it does not assume a name prefix, so an
    ALIASED parameter is reported through the columns it really moves.
    """
    c0 = np.asarray(f.build_params(x)[0]["coeff"], np.float64)
    out = {}
    for i, nm in enumerate(f.fwd_names()):
        h = step * max(abs(float(x[i])), 1.0)
        moved = set()
        for sgn in (1.0, -1.0):
            xp = np.asarray(x, np.float64).copy()
            xp[i] += sgn * h
            c = np.asarray(f.build_params(xp)[0]["coeff"], np.float64)
            moved.update(int(j) for j in np.nonzero(c != c0)[0])
        out[nm] = sorted(moved)
    return out


def _pin_other_mw(f, keep):
    """``bench.py --free-mw``: pin every mass/width whose NAME is not in keep."""
    known = set(dict.fromkeys(f.var_registry.flat_names))
    assert not (set(keep) - known), sorted(set(keep) - known)
    f.set_fixed({n: float(f.defaults[n])
                 for n in dict.fromkeys(f.var_registry.flat_names)
                 if n.endswith(("_mass", "_width")) and n not in set(keep)
                 and n in f.defaults})


@functools.lru_cache(maxsize=1)
def _ctpwa_facts():
    """The CTPWA mask arithmetic, computed ONCE per process.

    Host-side and kernel-agnostic (the rule is ``gamma_matrix`` + the mask):
    the ten free parameters move 14 columns, 27 waves are cacheable and 7 are
    not, and 67 of the 88 denominator non-zeros sit in the cached block.
    """
    f = Fitter(CTPWA_CFG)
    f.apply_constrains()
    _pin_other_mw(f, CTPWA_FREE_MW)
    x = f.initial_values(seed=1)
    kc = f.kernel_config
    den = [str(n) for n in np.asarray(kc["den_names"]).ravel()]
    per = _moving_by_param(f, x)
    assert len(x) == 76, f"the ten-parameter column is x = 76, got {len(x)}"
    moving = sorted(set().union(*(per[n] for n in CTPWA_FREE_MW)))
    gam = np.asarray(kc["gamma_matrix"], np.float64)
    gam = gam.reshape(-1, gam.shape[-1])
    free = np.ones(len(den), np.int8)
    free[moving] = 0
    cols_of = [np.nonzero(gam[k])[0] for k in range(gam.shape[0])]
    fixed_wave = np.asarray([bool(free[c].all()) for c in cols_of], bool)
    n_nz = int(sum(c.size for c in cols_of))
    n_nz_fix = int(sum(cols_of[k].size for k in np.nonzero(fixed_wave)[0]))
    params = f.build_params(x)[0]
    return (kc, np.asarray(moving, np.int64), fixed_wave, n_nz, n_nz_fix,
            den, params)


def _ctpwa_present():
    return os.path.exists(CTPWA_CFG)


# ── gate 1: mask-on vs the fp64 reference ──────────────────────────────

#: MEASURED on the masked path over ``MASKED`` x ``rsize {1,17}`` x
#: ``norm {None, 1.3}``, worst ``_rel`` for each backend — the strict floors of
#: ``_flat_compute.FLOORS`` apply UNCHANGED, so no absolute companion is needed
#: (the ``_ident``/``_cp`` dynamic range is covered by the relative bound; the
#: local files' ``rel OR abs 1e-4`` escape would accept a mask bug on a
#: near-zero ``P``, and it is not needed here):
#:
#:   s0f: Q 9.2e-7, P 3.3e-6, dck 1.3e-6, dcoeff 7.5e-6, dnorm 2.9e-7
#:   s0g: Q 9.2e-7, P 3.3e-6, dck 1.3e-6, dcoeff 7.5e-6, dnorm 2.9e-7
#:
#: against floors 2e-5 / 3e-5 / 2e-5 / 1e-4 — a 9x..22x margin.

@pytest.mark.parametrize("name", SPECS_MASKED)
@pytest.mark.parametrize("cfg,cols", MASKED, ids=MASK_IDS)
@pytest.mark.parametrize("rs", [1, 17])
@pytest.mark.parametrize("norm", [None, 1.3])
def test_mask_on_matches_the_reference(name, cfg, cols, rs, norm):
    """Q/P/dck/dcoeff(free)/dnorm at the STRICT per-backend ``_rel`` floor.

    ``_gate`` is ``_flat_compute``'s floor for THIS backend — the measured
    relative bound, with no absolute escape: an absolute 1e-4 would accept a
    mask bug on an event whose ``P`` happens to be small.
    """
    f, kc, gpu = _masked(name, cfg, cols, rs=rs)
    ref = _references(kc, rs)
    params = _params(f)
    base = _in_grid(kc, 64, seed=0)
    data = _replicate(base, rs) if rs > 1 else base
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=norm)
        Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=norm)
        k = gpu.kernel
        assert k.n_fix > 0, "this case must actually cache something"
        assert k.n_fix + k.n_free == k.n_wave
        assert int(np.asarray(k.fixed_wave).sum()) == k.n_fix
        assert _gate(name, Pg, Pr, "p", cfg=cfg), "P"
        assert _gate(name, Qg, Qr, cfg=cfg), "Q"
        assert _gate(name, gg["ck"], gr["ck"], "ck", cfg=cfg), "dck"
        free_cols = np.setdiff1d(np.arange(k.n_basis),
                                 np.asarray(k.watched_cols, np.int64))
        if free_cols.size:
            assert _gate(name, gg["coeff"][free_cols], gr["coeff"][free_cols],
                         "coeff", cfg=cfg), "dcoeff (free columns)"
        if norm is not None:
            assert _gate(name, gpu.kernel._last_dnorm, ref._last_dnorm,
                         cfg=cfg), "dnorm"
    finally:
        ref.free()
        gpu.free()


# ── gate 1b: the off switch — no mask IS the parent ────────────────────

@pytest.mark.parametrize("name", SPECS_MASKED)
@pytest.mark.parametrize("cfg", [CFG_ONE, CFG_KM])
@pytest.mark.parametrize("rs", [1, 17])
@pytest.mark.parametrize("norm", [None, 1.3])
def test_with_no_mask_the_numbers_are_the_parent_s0e2(name, cfg, rs, norm):
    """Declaring nothing must give ``cuda_flat_pwa_gen_s0e2``'s numbers.

    A DIFFERENT oracle from the family floor: not "small compared with the
    reference", but "this is the parent's own computation".  ``P`` is written
    once per event, so it is EXACT; the accumulated quantities are
    ``atomicAdd``-ordered across blocks and are gated at :data:`PARENT_TOL`.
    """
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    base = _in_grid(kc, 64, seed=2)
    data = _replicate(base, rs) if rs > 1 else base
    parent = create_backend("cuda_flat_pwa_gen_s0e2", kc, model="pwa_gen",
                            resolution_size=rs)
    gpu = create_backend(name, kc, model="pwa_gen", resolution_size=rs)
    ha, hb = parent.load_data(data), gpu.load_data(data)
    try:
        assert gpu.kernel.n_fix == 0
        Qa, ga, Pa = parent.compute(params, ha, norm=norm)
        Qb, gb, Pb = gpu.compute(params, hb, norm=norm)
        assert np.array_equal(np.asarray(Pa), np.asarray(Pb)), (
            "P must be bit-identical to the parent's")
        for label, y, x in (("Q", Qb, Qa), ("dck", gb["ck"], ga["ck"]),
                            ("dcoeff", gb["coeff"], ga["coeff"])):
            assert _rel(y, x) <= PARENT_TOL, f"{label}: {_rel(y, x):.2e}"
        if norm is not None:
            assert _rel(gb["norm"], ga["norm"]) <= PARENT_TOL, "dnorm"
    finally:
        ha.free()
        hb.free()
        parent.free()
        gpu.free()


# ── gate 2: the exact-zero contract ────────────────────────────────────
@pytest.mark.parametrize("name", SPECS_MASKED)
@pytest.mark.parametrize("cfg,cols", MASKED, ids=MASK_IDS)
def test_a_cached_waves_columns_have_an_exactly_zero_coefficient_gradient(
        name, cfg, cols):
    """The accepted contract: a cached wave has no ``D``, hence no ``pv``.

    Exactly 0.0 — not "small" — at EVERY column a cached wave reads, and the
    non-cached columns are not all zero (otherwise the gate is vacuous).
    """
    f, kc, gpu = _masked(name, cfg, cols)
    k = gpu.kernel
    params = _params(f)
    data = _in_grid(kc, 48, seed=1)
    try:
        _Q, g, _P = gpu.compute(params, gpu.load_data(data), norm=1.3)
        cached = _cached_columns(k, _gamma_2d(kc, k.n_wave))
        assert cached.size, "this case must cache a wave"
        assert np.all(g["coeff"][cached] == 0.0), g["coeff"][cached]
        free = np.setdiff1d(np.arange(k.n_basis), cached)
        if free.size:
            assert np.any(g["coeff"][free] != 0.0), (
                "every free column's gradient vanished — the mask leaked")
    finally:
        gpu.free()


# ── gate 3: a partly fixed wave is NOT cached ──────────────────────────

@pytest.mark.parametrize("name", SPECS_MASKED)
def test_a_partly_fixed_wave_is_still_differentiated_and_not_cached(name):
    """``_ident``'s waves 2,3 read 5..9 and waves 0,1 read 0..4.

    Fixing 5..9 AND column 0 leaves waves 0,1 PARTLY fixed: they must not be
    cached (``n_fix`` is the same as for 5..9 alone), column 0 is declared but
    outside the cache's watch list, it keeps a gradient, and BOTH ``dcoeff``
    and ``dck`` still match the reference on the masked path.
    """
    f, kc, gpu = _masked(name, CFG_IDENT, [0, 5, 6, 7, 8, 9])
    k = gpu.kernel
    assert (k.n_fix, k.n_free) == (2, 2)
    assert list(np.asarray(k.fixed_waves, np.int64)) == [2, 3]
    assert 0 in set(np.asarray(k.declared_fixed, np.int64).tolist())
    assert 0 not in set(np.asarray(k.watched_cols, np.int64).tolist())
    # the extra declared column caches nothing extra: n_fix is unchanged
    _f2, _kc2, plain = _masked(name, CFG_IDENT, [5, 6, 7, 8, 9])
    try:
        assert plain.kernel.n_fix == k.n_fix
    finally:
        plain.free()
    ref = _references(kc)
    params = _params(f)
    data = _in_grid(kc, 64, seed=0)
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=1.3)
        assert _gate(name, Pg, Pr, "p", cfg=CFG_IDENT), "P"
        assert _gate(name, Qg, Qr, cfg=CFG_IDENT), "Q"
        assert _gate(name, gg["ck"], gr["ck"], "ck", cfg=CFG_IDENT), "dck"
        # column 0 belongs to a FREE wave, so it keeps a gradient
        assert gg["coeff"][0] != 0.0
        assert _gate(name, gg["coeff"][[0]], gr["coeff"][[0]], "coeff",
                     cfg=CFG_IDENT), "dcoeff at a declared-but-uncached column"
        free = np.setdiff1d(np.arange(k.n_basis),
                            np.asarray(k.watched_cols, np.int64))
        assert _gate(name, gg["coeff"][free], gr["coeff"][free], "coeff",
                     cfg=CFG_IDENT), "dcoeff (free columns)"
        # ... and it can MOVE without invalidating the cache (a declared column
        # no cached wave reads is not watched)
        moved = dict(params)
        c = np.asarray(params["coeff"], np.float64).copy()
        c[0] += 1e-9
        moved["coeff"] = c
        gpu.compute(moved, gpu.load_data(data), norm=1.3)
    finally:
        ref.free()
        gpu.free()


# ── gate 4: the invalidation guard ─────────────────────────────────────

@pytest.mark.parametrize("name", SPECS_MASKED)
def test_a_moved_fixed_value_raises_with_the_exact_message(name):
    """The guard runs BEFORE anything is launched, on every call.

    The message is the kernel's own ``FIXED_CHANGED`` (read from its module, so
    no text is duplicated here).  A rejected call leaves the handle untouched —
    the same compute still returns the same objective — and a FREE column may
    move.
    """
    f, kc, gpu = _masked(name, CFG_IDENT, [5, 6, 7, 8, 9])
    params = _params(f)
    h = gpu.load_data(_in_grid(kc, 24, seed=1))
    watched = np.asarray(gpu.kernel.watched_cols, np.int64)
    assert watched.size
    try:
        Q0 = gpu.compute(params, h, norm=1.3)[0]
        moved = dict(params)
        c = np.asarray(params["coeff"], np.float64).copy()
        c[watched[0]] += 1e-9
        moved["coeff"] = c
        with pytest.raises(ValueError) as ei:
            gpu.compute(moved, h, norm=1.3)
        assert str(ei.value) == _fixed_changed(gpu)
        # nothing was launched, nothing was left half-written
        assert _rel(gpu.compute(params, h, norm=1.3)[0], Q0) <= 1e-14
        # a FREE column moving is fine (no cache depends on it)
        free = np.setdiff1d(np.arange(c.size), watched)
        if free.size:
            moved2 = dict(params)
            c2 = np.asarray(params["coeff"], np.float64).copy()
            c2[free[0]] += 1e-9
            moved2["coeff"] = c2
            gpu.compute(moved2, h, norm=1.3)
    finally:
        h.free()
        gpu.free()


# ── gate 5: the fill at load, the guard from the first compute ─────────

@pytest.mark.parametrize("name", SPECS_MASKED)
def test_the_cache_is_filled_at_load_and_guarded_from_the_first_compute(name):
    """There is no snapshot: ``load_data`` fills from the DECLARED values.

    Every handle is READY the moment it is loaded (two live handles at once is
    the fitter's own state), and a FRESH backend's first compute already
    rejects a value that moved — the comparison is against the declaration,
    not against whatever the first call happened to pass.
    """
    f, kc, gpu = _masked(name, CFG_IDENT, [5, 6, 7, 8, 9])
    k = gpu.kernel
    assert k.has_fixed_values()
    h1 = gpu.load_data(_in_grid(kc, 16, seed=1))
    h2 = gpu.load_data(_in_grid(kc, 23, seed=2))
    try:
        assert h1.ptr != h2.ptr
        assert k.afix_ready(h1) and k.afix_ready(h2)
        # a second load does not un-ready the first
        assert k.afix_ready(h1)
    finally:
        h1.free()
        h2.free()
    fresh, kc2, gpu2 = _masked(name, CFG_IDENT, [5, 6, 7, 8, 9])
    params = _params(fresh)
    h = gpu2.load_data(_in_grid(kc2, 16, seed=1))
    try:
        c = np.asarray(params["coeff"], np.float64).copy()
        c[int(np.asarray(gpu2.kernel.watched_cols, np.int64)[0])] += 1e-9
        bad = dict(params)
        bad["coeff"] = c
        assert gpu2.kernel.afix_ready(h)
        with pytest.raises(ValueError) as ei:
            gpu2.compute(bad, h, norm=1.3)
        assert str(ei.value) == _fixed_changed(gpu2)
    finally:
        h.free()
        gpu2.free()


# ── gate 6: one declaration, at both levels ────────────────────────────

@pytest.mark.parametrize("name", SPECS_MASKED)
def test_a_mask_and_its_values_are_one_declaration_at_both_levels(name):
    f, kc, _gpu = _masked(name, CFG_IDENT, [5, 6, 7, 8, 9])
    params = _params(f)
    coeff = np.asarray(params["coeff"], np.float64)
    n_basis = int(np.asarray(kc["basis_table"]).shape[0])

    # ── the kernel constructor ─────────────────────────────────────────
    with pytest.raises(ValueError, match="declared WITH the values"):
        create_backend(name, kc, model="pwa_gen", fixed_cols=[5, 6])
    with pytest.raises(ValueError, match="no column is declared fixed"):
        create_backend(name, kc, model="pwa_gen",
                       fixed_values=_fv(kc, params, [5, 6]))
    with pytest.raises(ValueError,
                       match="do not cover the declaration") as ei:
        create_backend(name, kc, model="pwa_gen", fixed_cols=[5, 6, 7],
                       fixed_values={5: float(coeff[5])})
    msg = str(ei.value)
    for missing in (6, 7):
        assert str(missing) in msg, msg
    ones = np.zeros(n_basis, np.int8)
    ones[[5, 6, 7, 8, 9]] = 1
    with pytest.raises(ValueError, match="COLUMN INDICES"):
        create_backend(name, kc, model="pwa_gen", fixed_cols=ones,
                       fixed_values=np.asarray(coeff))

    # ── the fixed_coeff wrapper (the same rules) ────────────────────────
    w = _wrapper(name, kc)
    try:
        with pytest.raises(ValueError, match="needs the values of the columns"):
            w.set_fixed_cols([5, 6])
        with pytest.raises(ValueError, match="do not cover the declaration"):
            w.set_fixed_cols([5, 6, 7], {5: float(coeff[5])})
        with pytest.raises(ValueError, match="COLUMN INDICES"):
            w.set_fixed_cols(ones, np.asarray(coeff))
        with pytest.raises(ValueError, match="no columns"):
            w.set_fixed_cols(None, np.asarray(coeff))
        with pytest.raises(ValueError, match="no columns"):
            w.set_fixed_cols([], np.asarray(coeff))
        assert w.mask is None, "a rejected declaration must not install a mask"
        # a real declaration, then the documented way back to the parent path
        w.set_fixed_cols([5, 6, 7, 8, 9],
                         {int(c): float(coeff[int(c)]) for c in (5, 6, 7, 8, 9)})
        assert w.base.kernel.n_fix == 2
        w.set_fixed_cols(None)
        assert w.mask is None and w.values is None
        assert w.base.kernel.n_fix == 0
    finally:
        w.free()


# ── gate 7: the one-call derivation ───────────────────────────────────

@pytest.mark.parametrize("name", SPECS_MASKED)
@pytest.mark.parametrize("cfg,fix,how", [
    (CFG_ONE, ("k2_mass", "k2_width"), "transform"),
    (CFG_IDENT, ("k2_mass0", "k2_mass1", "k2_width0", "k2_width1"), "jacobian"),
], ids=["one-transform", "ident-jacobian"])
def test_set_fixed_coeffs_equals_the_explicit_columns_and_values(
        name, cfg, fix, how):
    """``set_fixed_coeffs(fitter)`` installs what the explicit form installs.

    The columns come from the constraint state; the VALUES are
    ``build_params(x)["coeff"][cols]`` — the declaration is both, so the two
    routes must agree exactly, mask and values.  Both derivations are covered:
    the graph walk (the plain BW basis of ``_one``, where pinning the
    mass/width fixes every column) and the exact Jacobian (``_ident``'s
    ``kmatrix_n`` weights, where the conservative walk masks nothing because
    the numerator couplings are declared inputs).
    """
    f = Fitter(cfg, backend={"name": WRAP, "base": name})
    f.apply_constrains()
    res = f.build_params(f.initial_values(seed=1))[1]
    f.set_fixed({n: float(res[n]) for n in fix})
    w = f.backend
    cols = f.model.fixed_coeff_columns(f.cm, x=f.initial_values(seed=1),
                                      how=how)
    assert len(cols), "the pinned masses/widths must fix some columns"
    try:
        w.set_fixed_coeffs(f, how=how)
        coeff = np.asarray(
            f.build_params(f.initial_values(seed=1))[0]["coeff"], np.float64)
        c = np.asarray(list(cols), np.int64)
        assert sorted(np.nonzero(w.mask)[0].tolist()) == sorted(c.tolist())
        assert np.array_equal(w.values[c], coeff[c])
        assert w.base.kernel.n_fix > 0
        w2 = _wrapper(name, f.kernel_config)
        try:
            w2.set_fixed_cols(c, {int(i): float(coeff[int(i)]) for i in c})
            assert np.array_equal(w.mask, w2.mask)
            assert np.array_equal(w.values, w2.values)
            assert w2.base.kernel.n_fix == w.base.kernel.n_fix
        finally:
            w2.free()
    finally:
        w.free()


# ── gate 8: the "caches nothing" diagnostic ────────────────────────────

@pytest.mark.parametrize("name", SPECS_MASKED)
def test_a_mask_that_caches_nothing_warns_once_and_a_partial_mask_is_silent(
        name):
    """``_ident``: fixing 5,6 leaves wave 2 reading 7,8,9 — no wave is fixed.

    That is a silent no-op on the cache, so it warns ONCE, naming the closest
    wave and every column it still reads — at the kernel AND (once, not twice)
    through the wrapper.  A PARTIAL mask (the production state) never nags.
    """
    f, kc, _gpu = _masked(name, CFG_IDENT, [5, 6, 7, 8, 9])
    params = _params(f)
    coeff = np.asarray(params["coeff"], np.float64)
    no_cache = [5, 6]
    with pytest.warns(MaskNotEngaged, match="will not engage") as rec:
        k = create_backend(name, kc, model="pwa_gen", fixed_cols=no_cache,
                           fixed_values=_fv(kc, params, no_cache))
    try:
        assert len(rec) == 1, [str(r.message) for r in rec]
        msg = str(rec[0].message)
        assert "wave 2" in msg, msg
        for col in ("k2_m4", "k2_rho", "k2_srho"):
            assert col in msg, msg
        assert k.kernel.n_fix == 0
    finally:
        k.free()
    w = _wrapper(name, kc)
    try:
        with pytest.warns(MaskNotEngaged) as rec:
            w.set_fixed_cols(no_cache,
                             {int(c): float(coeff[c]) for c in no_cache})
        assert len(rec) == 1, "the wrapper must not duplicate the base's warning"
        assert w.base.kernel.n_fix == 0
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            w.set_fixed_cols([5, 6, 7, 8, 9],
                             {int(c): float(coeff[c])
                              for c in (5, 6, 7, 8, 9)})
        assert w.base.kernel.n_fix == 2
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            w.set_fixed_cols(None)          # the parent path never warns
        assert w.mask is None
    finally:
        w.free()


# ── gate 9: the cache size, and the reload path ────────────────────────

@pytest.mark.parametrize("name", SPECS_MASKED)
@pytest.mark.parametrize("cfg,cols", [(CFG_IDENT, [5, 6, 7, 8, 9]),
                                      (CFG_ONE, [0, 1, 2])],
                         ids=["ident-2fixed", "one-1fixed"])
def test_the_fixed_cache_size_is_the_documented_formula_per_handle(
        name, cfg, cols):
    """``8·n_proj·n_fix`` bytes per event PER HANDLE — and none without a mask."""
    f, kc, gpu = _masked(name, cfg, cols)
    k = gpu.kernel
    try:
        assert k.n_fix == (2 if cfg == CFG_IDENT else 1)
        for ne in (0, 1, 64, 1000):
            assert k.fix_bytes(ne) == 8 * k.n_proj * k.n_fix * ne
        h1 = gpu.load_data(_in_grid(kc, 16, seed=1))
        h2 = gpu.load_data(_in_grid(kc, 101, seed=2))
        try:
            # each handle owns its own cache, sized for its own sample
            assert k.afix_ready(h1) and k.afix_ready(h2)
            assert k.fix_bytes(h1.ne) == 8 * k.n_proj * k.n_fix * h1.ne
            assert k.fix_bytes(h2.ne) == 8 * k.n_proj * k.n_fix * h2.ne
        finally:
            h1.free()
            h2.free()
        plain = create_backend(name, kc, model="pwa_gen")
        try:
            assert plain.kernel.n_fix == 0
            assert plain.kernel.fix_bytes(1000) == 0
        finally:
            plain.free()
    finally:
        gpu.free()


@pytest.mark.parametrize("name", SPECS_MASKED)
def test_changing_the_mask_reloads_each_retained_sample_once(name):
    """The wrapper's lifecycle over THIS kernel: rebuild per mask, reload per
    (sample, mask) — never an unchanged mask, never at ``set_fixed_cols``."""
    f, kc, _gpu = _masked(name, CFG_IDENT, [5, 6, 7, 8, 9])
    params = _params(f)
    coeff = np.asarray(params["coeff"], np.float64)
    mask_a, mask_b = [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]

    def set_(w, mask):
        w.set_fixed_cols(mask, {int(c): float(coeff[int(c)]) for c in mask})

    w = _wrapper(name, kc)
    try:
        h = w.load_data(_in_grid(kc, 32, seed=3))
        try:
            assert (w.creates, w.loads) == (1, 1)
            w.compute(params, h, norm=1.3)
            assert w.loads == 1, "an unchanged mask reloaded"
            set_(w, mask_a)
            assert (w.creates, w.loads) == (2, 1), (
                "the set must not touch a sample")
            w.compute(params, h, norm=1.3)
            assert w.loads == 2
            w.compute(params, h, norm=1.3)
            assert w.loads == 2, "no reload on an unchanged mask"
            set_(w, mask_b)
            assert (w.creates, w.loads) == (3, 2)
            w.compute(params, h, norm=1.3)
            assert w.loads == 3
        finally:
            h.free()
    finally:
        w.free()


# ── gate 10: masked chunk invariance ───────────────────────────────────

@pytest.mark.parametrize("name", SPECS_MASKED)
@pytest.mark.parametrize("rs", [1, 17])
def test_a_multi_chunk_call_equals_a_single_chunk_call(name, rs):
    """The fill is absolute-indexed and the group is never split.

    P is compared EXACTLY (one writer per event); the accumulated quantities at
    the re-association floor.
    """
    f, kc, _gpu = _masked(name, CFG_IDENT, [5, 6, 7, 8, 9], rs=rs)
    params = _params(f)
    data = _in_grid(kc, 500, seed=3)
    cols = [5, 6, 7, 8, 9]
    vals = _fv(kc, params, cols)

    def run(bs):
        be = create_backend(name, kc, model="pwa_gen", resolution_size=rs,
                            batch_size=bs, fixed_cols=cols, fixed_values=vals)
        try:
            h = be.load_data(data)
            Q, g, P = be.compute(params, h, norm=1.3)
            h.free()
            return Q, g["ck"], g["coeff"], np.asarray(P)
        finally:
            be.free()

    base = run(10 ** 9)
    for bs in (1, 7, 8, 17, 64, 5000):
        got = run(bs)
        assert np.array_equal(got[3], base[3]), (
            f"rs={rs} batch={bs}: P moved across the chunk boundary — the "
            "fill must be absolute-indexed")
        for label, a, b in (("Q", got[0], base[0]), ("dck", got[1], base[1]),
                            ("dcoeff", got[2], base[2])):
            assert _rel(a, b) <= 1e-14, (
                f"rs={rs} batch={bs}: {label} moved across the chunk boundary "
                f"(rel {_rel(a, b):.2e})")


# ── gate 11: a short final group on the masked path ────────────────────

@pytest.mark.parametrize("name", SPECS_MASKED)
@pytest.mark.parametrize("rs,ne", [(2, 64), (17, 101), (40, 101)])
def test_a_short_final_group_on_the_masked_path_matches_the_reference(
        name, rs, ne):
    """``ne % rsize != 0`` with a mixed mask: the last group is short."""
    f, kc, gpu = _masked(name, CFG_IDENT, [5, 6, 7, 8, 9], rs=rs)
    ref = _references(kc, rs)
    params = _params(f)
    data = _in_grid(kc, ne, seed=6)
    k = gpu.kernel
    try:
        assert (k.n_fix, k.n_free) == (2, 2)
        Q, g, P = gpu.compute(params, gpu.load_data(data), norm=1.3)
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        assert _gate(name, P, Pr, "p", cfg=CFG_IDENT), "P"
        assert _gate(name, Q, Qr, cfg=CFG_IDENT), "Q"
        assert _gate(name, g["ck"], gr["ck"], "ck", cfg=CFG_IDENT), "dck"
        free = np.setdiff1d(np.arange(k.n_basis),
                            np.asarray(k.watched_cols, np.int64))
        assert _gate(name, g["coeff"][free], gr["coeff"][free], "coeff",
                     cfg=CFG_IDENT), "dcoeff (free columns)"
        assert _gate(name, g["norm"], gr["norm"], cfg=CFG_IDENT), "dnorm"
    finally:
        gpu.free()
        ref.free()


# ── gate 11b: a group no block can hold ────────────────────────────────

@pytest.mark.parametrize("name", SPECS_MASKED)
def test_a_group_no_block_can_hold_computes_or_names_an_alternative(name):
    """``resolution_size`` 300 outgrows any tile (at most 256 events a block).

    A kernel that keeps no scratch between a group's events (``s0g``) must
    REFUSE such a group at construction, naming a backend that CAN walk it;
    one that keeps the staging (``s0f``) must compute it correctly.  Either
    answer is a contract — silently mis-closing a group is not.
    """
    rs = 300
    f = Fitter(CFG_IDENT)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    cols = [5, 6, 7, 8, 9]
    vals = _fv(kc, params, cols)
    data = _in_grid(kc, 34, seed=5)
    try:
        gpu = create_backend(name, kc, model="pwa_gen", resolution_size=rs,
                             fixed_cols=cols, fixed_values=vals)
    except ValueError as exc:
        named = [n for n in sorted(backends_for_model("pwa_gen"))
                 if n != name and n in str(exc)]
        assert named, (
            f"{name} refused a group it cannot hold without naming a working "
            f"alternative: {exc}")
        # the MOST SPECIFIC name in the message: ``cuda_flat_pwa_gen`` is a
        # substring of ``cuda_flat_pwa_gen_s0f`` and is not the alternative
        alt_name = max(named, key=len)
        alt = create_backend(alt_name, kc, model="pwa_gen",
                             resolution_size=rs, fixed_cols=cols,
                             fixed_values=vals)
        alt.free()              # the named alternative is usable on this shape
        return
    ref = _references(kc, rs)
    try:
        Q, g, P = gpu.compute(params, gpu.load_data(data), norm=1.3)
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        assert _gate(name, P, Pr, "p", cfg=CFG_IDENT), "P"
        assert _gate(name, Q, Qr, cfg=CFG_IDENT), "Q"
        assert _gate(name, g["ck"], gr["ck"], "ck", cfg=CFG_IDENT), "dck"
    finally:
        ref.free()
        gpu.free()


# ── gate 19: the multi-BW refusal (the shape guard) ────────────────────
#:
#: Placed beside gate 11b: both are REFUSALS that must name a usable
#: alternative.  The ``n_res1`` capability is the ``_flat_compute`` sweep's
#: skip; the gate itself lives here, with the leaf that owns the config
#: contract (``CONFIG_OWNER``), so no skip claims coverage elsewhere.

@pytest.mark.parametrize("name", SPECS_MASKED)
def test_a_multi_bw_chain_is_refused_naming_cuda_flat_pwa_gen2(name):
    """``n_res == 2`` has no one-BW-per-chain denominator, so refuse it loudly.

    Every backend here is tagged ``n_res1`` (``_flat_compute`` derives the tag
    from the registry, so a new masked leaf is covered by declaring it).  The
    multi-BW chain's amplitude is ``bw_{w,0}·bw_{w,1}`` — a product over the
    factor axis — which these kernels do not walk.  The refusal must happen at
    CONSTRUCTION, before any device work, and must name the documented
    fallback ``cuda_flat_pwa_gen2``; asserting only "raises ValueError" would
    let the recommendation be dropped while the gate stayed green.

    The fixture's shape is asserted FIRST: a gate written on a config that is
    not multi-BW would pass with the guard never reached.
    """
    assert "n_res1" in backend_tags(name, "pwa_gen"), (
        f"{name} is swept by the multi-BW gate without declaring `n_res1`")
    f = Fitter(CFG_MB)
    f.apply_constrains()
    kc = f.kernel_config
    assert int(f.model.n_res) == 2, (
        f"CFG_MB is not a multi-BW chain: model n_res = {f.model.n_res}")
    assert int(np.asarray(kc["num_table"]).shape[1]) == 2, (
        "CFG_MB's kernel config must carry two factors per chain, got "
        f"{np.asarray(kc['num_table']).shape}")
    with pytest.raises(ValueError, match="cuda_flat_pwa_gen2") as ei:
        create_backend(name, kc, model="pwa_gen")
    assert "ONE BW per chain" in str(ei.value), str(ei.value)


# ── gate 12: return_p ──────────────────────────────────────────────────

@pytest.mark.parametrize("name", SPECS_MASKED)
def test_return_p_false_does_not_change_the_objective(name):
    f, kc, gpu = _masked(name, CFG_IDENT, [5, 6, 7, 8, 9])
    params = _params(f)
    h = gpu.load_data(_in_grid(kc, 12, seed=2))
    try:
        Qa, ga, Pa = gpu.compute(params, h, norm=1.3, return_p=False)
        Qb, gb, Pb = gpu.compute(params, h, norm=1.3, return_p=True)
        assert Pa is None and np.asarray(Pb).size == 12
        assert _rel(Qa, Qb) <= 1e-14
        assert _rel(ga["ck"], gb["ck"]) <= 1e-14
        assert _rel(ga["coeff"], gb["coeff"]) <= 1e-14
    finally:
        h.free()
        gpu.free()


# ── gate 13: the empty sample ──────────────────────────────────────────

@pytest.mark.parametrize("name", SPECS_MASKED)
def test_an_empty_sample_is_the_identity_and_its_cache_is_ready(name):
    """``ne == 0`` is the identity, not the "no cache filled" stop."""
    f, kc, gpu = _masked(name, CFG_IDENT, [5, 6, 7, 8, 9])
    params = _params(f)
    empty = {k: np.asarray(v)[:0] for k, v in _in_grid(kc, 4).items()}
    h = gpu.load_data(empty)
    try:
        assert gpu.kernel.afix_ready(h), (
            "an empty sample has nothing to fill, so it must be READY")
        Q, g, P = gpu.compute(params, h, norm=1.3)
        assert Q == 0.0
        assert np.asarray(P).size == 0
        assert np.all(np.isfinite(g["ck"]))
        assert np.all(np.isfinite(g["coeff"]))
    finally:
        h.free()
        gpu.free()


# ── gate 14: the full fitter path ─────────────────────────────────────

@pytest.mark.parametrize("name", SPECS_MASKED)
def test_the_full_fitter_path_with_a_mask(name):
    """``Fitter.get_nll`` through the wrapper, with the phase-space AND the
    data handle live (each fills its own cache, over different samples), and
    the masked columns contributing EXACTLY nothing to the reduced gradient.
    """
    cfg = CFG_IDENT
    f0 = Fitter(cfg)
    f0.apply_constrains()
    res = f0.build_params(f0.initial_values(seed=1))[1]
    fix = {n: float(res[n]) for n in
           ("k2_mass0", "k2_mass1", "k2_width0", "k2_width1")}
    mask = list(range(10))
    out = {}
    for be in (REF, WRAP):
        spec = ({"name": WRAP, "base": name, "batch_size": 5000}
                if be == WRAP else {"name": be})
        f = Fitter(cfg, backend=spec)
        f.apply_constrains()
        f.set_fixed(fix)
        kc = f.kernel_config
        if be == WRAP:
            p0 = f.build_params(f.initial_values(seed=1))[0]
            coeff = np.asarray(p0["coeff"], np.float64)
            f.backend.set_fixed_cols(
                mask, {int(c): float(coeff[int(c)]) for c in mask})
        # TWO DIFFERENT samples: a cache shared between the phase-space and
        # the data handle would otherwise pass
        f.set_phsp(_in_grid(kc, 40, seed=4))
        f.set_data(_in_grid(kc, 53, seed=5))
        if be == WRAP:
            w = f.backend
            assert len(w._records) == 2, "both handles must stay live"
            for rec in list(w._records):
                assert rec.real is not None
                assert w.base.kernel.afix_ready(rec.real)
        x = f.initial_values(seed=1)
        out[be] = (f, f.get_nll(x))
    try:
        fw, (Qa, ga) = out[WRAP]
        fr, (Qr, gr) = out[REF]
        assert _gate(name, Qa, Qr, cfg=cfg), f"Q: {Qa!r} vs {Qr!r}"
        ga = np.asarray(ga, np.float64)
        gr = np.asarray(gr, np.float64)
        assert float(np.abs(ga - gr).max()
                     / max(np.abs(gr).max(), 1e-300)) < 1e-4, "x-gradient"
        # the masked columns are exactly zero BEFORE the reduction, and they
        # contribute exactly nothing to it: zeroing them again changes nothing
        x = fw.initial_values(seed=1)
        params, resolved = fw.build_params(x)
        _nll, raw = fw.get_nll_raw(params)
        masked = np.asarray(mask, np.int64)
        assert np.all(np.asarray(raw["coeff"])[masked] == 0.0), (
            np.asarray(raw["coeff"])[masked])
        zeroed = dict(raw)
        g = np.asarray(raw["coeff"], np.float64).copy()
        g[masked] = 0.0
        zeroed["coeff"] = g
        red = fw._flat_gradient(fw._kernel_builder.backward(raw, resolved),
                                resolved, x)
        red_zeroed = fw._flat_gradient(
            fw._kernel_builder.backward(zeroed, resolved), resolved, x)
        assert np.array_equal(red, red_zeroed)
        # ... and it is the gradient the fitter itself returned (a second call
        # is the same call up to the cross-block `atomicAdd` re-association)
        assert _rel(red, np.asarray(ga)) <= 1e-14
    finally:
        for _f, _ in out.values():
            _f.backend.free()


# ── gate 18: the split's scratch is the CONTEXT's, not a handle's ──────

#: The probe is ``cudaMemGetInfo``, and the driver hands out device memory in
#: 2 MiB granules (MEASURED on this box: one 4 800 000 B ``cudaMalloc`` moves
#: ``free_bytes`` by exactly 6 291 456 B).  The sample size below is chosen so
#: the legitimate fixed-wave cache is a WHOLE number of granules, otherwise the
#: probe's own rounding shows up as a residual: at ``ne = 100 000`` the 4.8 MB
#: cache rounds to 6 MiB and leaves +1 491 456 B for CORRECT code, which a
#: 1 MB bound would condemn.  ``fix_bytes(ne) % _ALLOC_QUANTUM == 0`` is
#: asserted so a mask whose cache does not land on a granule fails loudly here
#: instead of looking like a leak.
_ALLOC_QUANTUM = 2 * 1024 * 1024
#: what a per-handle buffer may add beyond the documented cache.  An
#: ``ne``-sized per-handle scratch is 7.3–16.8 MB at this sample size, so it is
#: 7–16x the bound; the legitimate ``us``/``amp`` caches cancel against the
#: parent's (same kernel code, same flat config, same sample).
_SCRATCH_BOUND = 1024 * 1024


def _parent_backend(name):
    """The backend a case-1 kernel was copied FROM (``_flat_isolation``)."""
    src = ISOLATION[name]["parent"]
    assert src, f"{name} declares no parent, so there is nothing to cancel"
    stem = src.removeprefix("kernels_").removesuffix(".cu")
    return "cuda_" + stem


#: ONE load, measured in a FRESH process.  A settled pool is not enough: the
#: measurement runs after thousands of other tests, whose freed blocks the
#: driver can hand out again, and a single reused 2 MiB block moves the delta by
#: a whole granule (MEASURED in the full suite: the parent read 25 165 824 B
#: instead of its 27 262 976 B and the residual was one granule).  A fresh
#: process has no earlier pool state, so the number is the allocation itself.
_DELTA_CHILD = r"""
import json, sys
sys.path.insert(0, {tests_dir!r})
import numpy as np
from tabpwa import Fitter, monitor
from tabpwa.backends import create_backend
from _flat_grid import _in_grid
backend, cfg, ne, seed, cols, masked = json.loads(sys.argv[1])
f = Fitter(cfg)
f.apply_constrains()
kc = f.kernel_config
kw = {{}}
if masked:
    p = f.build_params(f.initial_values(seed=1))[0]
    n = int(np.asarray(kc["basis_table"]).shape[0])
    c = np.asarray(p["coeff"], np.float64)
    v = np.zeros(n, np.float64)
    v[cols] = c[cols]
    kw = dict(fixed_cols=list(cols), fixed_values=v)
data = _in_grid(kc, ne, seed=seed)
be = create_backend(backend, kc, model="pwa_gen", **kw)
try:
    monitor.sync()
    base = monitor.free_bytes()
    h = be.load_data(data)
    monitor.sync()
    after = monitor.free_bytes()
    h.free()
finally:
    be.free()
print("DELTA", int(base - after))
"""


def _delta_in_fresh_process(backend, cfg, ne, seed, cols, masked):
    """``load_data``'s byte cost, measured in a child with an empty pool."""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(tests_dir)
    script = _DELTA_CHILD.format(tests_dir=tests_dir)
    spec = json.dumps([backend, cfg, int(ne), int(seed),
                       None if cols is None else [int(c) for c in cols],
                       bool(masked)])
    try:
        out = subprocess.run([sys.executable, "-c", script, spec], cwd=root,
                             capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        raise AssertionError(f"the memory-probe child could not run: {exc}")
    if out.returncode != 0:
        raise AssertionError(
            "the memory-probe child failed:\n" + out.stderr[-3000:])
    for line in reversed(out.stdout.splitlines()):
        if line.startswith("DELTA "):
            return int(line.split()[1])
    raise AssertionError(
        "the memory-probe child printed no delta:\n" + out.stdout[-3000:])


def _scratch_deltas(backend, parent, cfg, ne, cols):
    """``(delta masked, delta parent)``, each in its own fresh process."""
    dm = _delta_in_fresh_process(backend, cfg, ne, 5, cols, True)
    dp = _delta_in_fresh_process(parent, cfg, ne, 5, None, False)
    return dm, dp


@pytest.mark.parametrize("name", SPECS_MASKED)
def test_a_load_does_not_allocate_the_split_scratch(name):
    """A ``load_data`` must not add a per-handle copy of the split's scratch.

    The measured regression this names: the buffers were ``ne``-sized in every
    handle (336 MB each at 6 M events, 672 MB across the data and phase-space
    handles).  They are now ONE ``cap``-sized allocation per CONTEXT, made at
    construction, so loading a sample adds nothing proportional to it.

    The gate is DIFFERENTIAL and it is measured at a large sample, because the
    obvious form cannot fail.  MEASURED: at 16 events BOTH loads cost 0 B (the
    legitimate ``us``/``amp`` caches are ~314 B/event, i.e. a few KB, below what
    the probe resolves), so a per-handle ``ne``-sized scratch of 896–2 048 B
    there is invisible and only a cap-sized (≥ 2 MB) block could trip it.  At
    ``ne`` = 131 072 (chosen so ``fix_bytes`` is a whole number of 2 MiB
    granules):

        delta(masked) 33 554 432 B
        delta(parent) 27 262 976 B (the same ``us``/``amp`` caches)
        fix_bytes     6 291 456 B
        residual              0 B

    and a per-handle ``ne``-sized buffer — MEASURED with a temporary 56 B/event
    ``cudaMalloc`` in the kernel's ``load_data`` — raises the residual to
    8 388 608 B and the gate FAILS, which is what makes it a gate.  The
    subtraction of ``fix_bytes`` is what lets the legitimate cache stay: it is
    the one thing a masked load is allowed to add.

    The probe is :func:`tabpwa.monitor.free_bytes` (``libcudart`` via
    ``ctypes``), NOT a sampler: ``nvidia-smi`` reports whole MiB, and this is a
    few-MB allocation.  The oracle is DEVICE-WIDE, so another process on the card
    makes the window meaningless — that is the one skip, derived from a
    precondition rather than from the backend, and it is re-queried AFTER the
    measurement: a user set that changed mid-window makes the numbers
    unattributable, so the attempt is retried once and reported as invalid
    (not asserted) if it moves again.  A kernel that keeps NO scratch (``s0g``)
    has nothing to allocate; it is asserted to have no ``aux_bytes`` report and
    skipped with that reason, so the check is never silently vacuous for it.
    """
    if not monitor.has_device():
        pytest.skip("no CUDA device for the free-bytes probe")
    others = other_gpu_users()
    if others:
        pytest.skip("another process is using the GPU (pid %s); device-wide "
                    "free bytes cannot attribute allocations"
                    % ", ".join(str(p) for p in others))
    f, kc, gpu = _masked(name, CFG_IDENT, [5, 6, 7, 8, 9])
    k = gpu.kernel
    if not hasattr(k, "aux_bytes"):
        gpu.free()
        pytest.skip(f"{name}: no per-context scratch (no aux_bytes) — a load "
                    "has nothing to allocate")
    try:
        aux = k.aux_bytes(0)
        assert aux >= _ALLOC_QUANTUM, aux   # MBs, easily visible in bytes
        per_event = 8 * int(k.n_fix) * int(k.n_proj)
        assert per_event > 0, (k.n_fix, k.n_proj)
        ne = _ALLOC_QUANTUM // math.gcd(per_event, _ALLOC_QUANTUM)
        fix = k.fix_bytes(ne)
        assert fix == per_event * ne, (fix, per_event, ne)
        assert fix % _ALLOC_QUANTUM == 0, (fix, _ALLOC_QUANTUM)
    finally:
        gpu.free()
    parent = _parent_backend(name)
    args = (name, parent, CFG_IDENT, ne, [5, 6, 7, 8, 9])
    for attempt in (1, 2):
        before = other_gpu_users()
        if before:
            pytest.skip("another process joined the GPU (pid %s); device-wide "
                        "free bytes cannot attribute allocations"
                        % ", ".join(str(p) for p in before))
        dm, dp = _scratch_deltas(*args)
        if other_gpu_users() == before:
            break
        warnings.warn(f"{name}: the GPU user set changed during the bytes "
                      f"measurement (attempt {attempt}); retrying",
                      RuntimeWarning)
    else:
        pytest.skip(f"{name}: the GPU user set could not be held still across "
                    "two measurement attempts — the deltas are unattributable")
    residual = dm - dp - fix
    assert residual < _SCRATCH_BOUND, (
        f"{name}: loading {ne} events costs {dm:,} B where the parent "
        f"{parent} costs {dp:,} B; after the legitimate {fix:,} B fixed-wave "
        f"cache that leaves {residual:,} B of per-handle allocation (an "
        f"ne-sized scratch is {ne * 56:,}–{ne * 128:,} B here)")


# ── the CTPWA mask arithmetic — ONE gate, every masked backend ─────────

@pytest.mark.skipif(not _ctpwa_present(),
                    reason="the CTPWA benchmark config is not present")
@pytest.mark.parametrize("name", SPECS_MASKED)
def test_the_ctpwa_mask_arithmetic_is_the_same_on_every_masked_backend(name):
    """The production mask, recomputed from the FREE ``x``, never from names.

    ``bench.py --free-mw`` leaves the paper's ten mass/width names free and
    pins the rest.  Those ten move **14** columns, not 10: the two charge
    sectors have distinct column names but ONE aliased parameter, so
    ``K3st1780p_mass`` moves ``K3st1780p_s0``/``_width_t`` AND
    ``K3st1780m_s0``/``_width_t``.  Measured: 14 fixed columns, 27 fixed / 7
    free waves, 67 of 88 non-zeros in the cached block.
    """
    kc, moving, fixed_wave, n_nz, n_nz_fix, den, params = _ctpwa_facts()
    assert moving.tolist() == [5, 6, 7, 8, 9, 10, 35, 36, 37, 38,
                               53, 54, 55, 56]
    assert [den[i] for i in moving.tolist()] == [
        "phi1680_s0", "phi1680_width_t", "X1750_s0", "X1750_width_t",
        "omega2220_s0", "omega2220_width_t",
        "K3st1780p_s0", "K3st1780p_width_t",
        "K4st2045p_s0", "K4st2045p_width_t",
        "K3st1780m_s0", "K3st1780m_width_t",
        "K4st2045m_s0", "K4st2045m_width_t"]
    assert (int(fixed_wave.sum()), int((~fixed_wave).sum())) == (27, 7)
    assert np.nonzero(~fixed_wave)[0].tolist() == [2, 3, 4, 19, 20, 29, 30]
    assert (n_nz, n_nz_fix) == (88, 67)
    cols = [i for i in range(len(den)) if i not in set(moving.tolist())]
    be = create_backend(name, kc, model="pwa_gen", fixed_cols=cols,
                        fixed_values=_fv(kc, params, cols))
    try:
        k = be.kernel
        assert (k.n_fix, k.n_free) == (27, 7)
        assert np.array_equal(np.asarray(k.fixed_wave), fixed_wave)
        assert k.n_nz == n_nz - n_nz_fix
        assert k.n_nz_fix == n_nz_fix
    finally:
        be.free()


@pytest.mark.skipif(not _ctpwa_present(),
                    reason="the CTPWA benchmark config is not present")
@pytest.mark.parametrize("free,cols", [
    # the alias: the K+ mass drives BOTH charge sectors' s0 and width_t
    ("K3st1780p_mass", [35, 36, 53, 54]),
    ("K3st1780p_width", [36, 54]),
])
def test_an_aliased_parameter_moves_both_charge_sectors(free, cols):
    """The per-PARAMETER truth the union over the ten names cannot state.

    With only ``K3st1780p_mass`` free, the moving columns include the
    ``K3st1780m_*`` ones (53, 54) — their names do not share the prefix, so a
    name mask that fixes ``K3st1780p_s0`` alone would leave a moving column
    free while the cache believes it is frozen.
    """
    f = Fitter(CTPWA_CFG)
    f.apply_constrains()
    _pin_other_mw(f, (free,))
    x = f.initial_values(seed=1)
    den = [str(n) for n in np.asarray(f.kernel_config["den_names"]).ravel()]
    per = _moving_by_param(f, x)
    assert per[free] == cols, (free, per[free])
    assert [den[i] for i in per[free]] == [den[i] for i in cols]
    if free == "K3st1780p_mass":
        # exactly the trap: the p column is NOT enough
        assert 53 in per[free] and 54 in per[free]


# ── provenance: the case-1 row each caching kernel must have ───────────
#:
#: ``cuda_flat_pwa_gen_s0g`` is a case-1 copy of ``cuda_flat_pwa_gen_s0f``'s
#: source, and ``s0f`` is itself a copy of ``s0e2``'s.  That property — own
#: module/``.so``/symbols, parent byte-identical, every source digest — is ONE
#: shared table (``tests/_flat_isolation.py``, collected by
#: ``test_flat_isolation.py``); here only the COVERAGE is asserted, so a
#: ``fixed_coeff`` backend that arrives without a row fails at import.
assert set(SPECS_MASKED) <= set(ISOLATION), (
    "a masked backend must have a case-1 row in _flat_isolation: "
    f"{sorted(set(SPECS_MASKED) - set(ISOLATION))}")
