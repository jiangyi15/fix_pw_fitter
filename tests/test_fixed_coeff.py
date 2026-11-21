"""``fixed_coeff`` — the coefficient-mask lifecycle as a delegating backend.

The mask-aware kernels are immutable per mask, so changing the mask means a new
kernel.  This wrapper owns that: it starts on the parent path (mask-less), and
``set_fixed_cols`` frees the base and rebuilds it for the new mask, retaining
the loaded samples and re-loading each one lazily on its next compute.

Gates:
1. the wrapper delegates (its parent path equals a mask-less base; a set mask
   equals the base built directly with that mask);
2. the mask is NOT a constructor argument (``set_fixed_cols`` is the only way);
3. ``set_fixed_cols`` rebuilds the base once per call, and a retained sample is
   re-loaded once per (sample, mask) — never on an unchanged mask;
4. A -> B -> A matches fresh wrappers, and repeated cycles do not grow the
   device memory;
5. a failed rebuild rolls back to the previous base and mask;
6. the wrapper module stays kernel-agnostic (no kernel name/symbol/import);
7. ``set_fixed_cols(None)``, ``free()`` and the mask-resolution rules
   (aliases, the parameter map, the warning) behave as before;
8. ``set_fixed_coeffs(fitter)`` — the one-call derivation from the constraint
   state — installs exactly the fixed columns ``set_fixed_cols`` would for the
   columns the MODEL derives, produces the same NLL, takes ``x`` from the
   fitter for ``how="jacobian"``, and propagates a plain-``pwa`` model's own
   error.  It is the wrapper's only convenience: ``ComputeBackend`` stays
   fixed-column-agnostic, and the name is not ``set_fixed`` (that is the
   Fitter's PARAMETER setter);
9. the vocabulary itself: no user-facing name still says "mask" — the module,
   its two functions, the config-key constant and the wrapper method are the
   fixed-coefficient names, and ``mask_spec.py`` is gone.
"""
import gc

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import backends_for_model, create_backend
from tabpwa.backends._fixed_coeff import FixedCoeffBackend, mask_equal
from tabpwa.fixed_coeff_spec import (mask_from_cols, resolve_fixed_cols,
                                     resolve_mask_and_values)
import _cuda_mem
from test_cuda_flat_cache_pwa_gen import _in_grid

CFG_ONE = "tests/config_pwa_gen.yml"
CFG_KM = "tests/config_pwa_gen_km.yml"
CFG_IDENT = "tests/config_pwa_gen_ident.yml"
CFG_CP = "tests/config_pwa_gen_cp.yml"
KERNEL = "cuda_flat_pwa_gen_s0f"
WRAP = "fixed_coeff"

CACHE_FLOOR = 2e-5
CACHE_FLOOR_P = 3e-5
CACHE_FLOOR_COEFF = 1e-4
S0F_TOL = 1e-14


def _rel(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _floor_ok(a, b, rel_bound=CACHE_FLOOR, abs_bound=1e-4):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    worst = float(np.abs(a - b).max())
    return (worst / max(np.abs(b).max(), 1e-300)) < rel_bound or worst < abs_bound


def _as_mask(kc, cols):
    n = int(np.asarray(kc["basis_table"]).shape[0])
    m = np.zeros(n, np.int8)
    if cols is not None:
        m[np.asarray(list(cols), np.int64)] = 1
    return m


def _wrapper(kc, **kw):
    spec = {"name": WRAP, "base": KERNEL, **kw}
    return create_backend(spec, kc, model="pwa_gen")


def test_dropping_the_caller_reference_releases_the_record():
    """No ``free()``: the caller's handle is the LAST reference.

    The store is a ``WeakKeyDictionary``, so `del` + ``gc`` must remove the
    record AND run the handle's ``__del__`` (which frees the device copy).
    Reverting `_records` to a plain dict keeps every other leak test green —
    they all free by hand — so this is the gate for that fix.
    """
    import weakref
    f = Fitter(BENCH_CFG, backend={"name": WRAP, "base": KERNEL})
    f.apply_constrains()
    f.set_fixed({n: float(f.defaults[n])
                 for n in dict.fromkeys(f.var_registry.flat_names)
                 if ("_mass" in n or "_width" in n) and n not in FREE_TEN
                 and n in f.defaults})
    w = f.backend
    kc = f.kernel_config
    w.set_fixed_cols(f.model.fixed_coeff_columns(f.cm),
                     {int(c): float(f.build_params(
                         f.initial_values(seed=1))[0]["coeff"][int(c)])
                      for c in f.model.fixed_coeff_columns(f.cm)})
    sample = _in_grid(kc, 40, seed=4)
    sample_ref = weakref.ref(sample["mass"])     # a dict cannot be weakref'd
    h = w.load_data(sample)
    try:
        # two loaded samples, one compute: both records alive
        w.compute(f.build_params(f.initial_values(seed=1))[0], h,
                  norm=1.3)
        assert len(w._records) == 1
        real = h.real
        assert real is not None and not h.freed
        del sample, h                       # the caller drops BOTH
        gc.collect()
        assert len(w._records) == 0, "the record survived the caller"
        assert sample_ref() is None, "the sample dict is still reachable"
    finally:
        w.free()
    gc.collect()


def _fitter(cfg=CFG_IDENT):
    f = Fitter(cfg)
    f.apply_constrains()
    params, _ = f.build_params(f.initial_values(seed=1))
    return f, f.kernel_config, params, _in_grid(f.kernel_config, 64, seed=0)


MASK_A = [5, 6, 7, 8, 9]
MASK_B = [5, 6]


def _set(w, cols, params):
    """``set_fixed_cols`` in the only spelling the wrapper accepts: WITH the
    values of the columns it declares (the cache is filled at ``load_data``,
    so there is no coefficient read to fall back on)."""
    coeff = np.asarray(params["coeff"], np.float64)
    return w.set_fixed_cols(cols, {int(c): float(coeff[int(c)]) for c in cols})


def _set_fitter(f, cols):
    """The same, with the values taken at the fitter's own start point."""
    params = f.build_params(f.initial_values(seed=1))[0]
    return _set(f.backend, cols, params)


def _moving_by_param(f, x, step=1e-3):
    """``{free parameter: [coeff columns its perturbation moves]}``."""
    c0 = np.asarray(f.build_params(x)[0]["coeff"], np.float64)
    out = {}
    for i, name in enumerate(f.fwd_names()):
        h = step * max(abs(float(x[i])), 1.0)
        moved = set()
        for sgn in (1.0, -1.0):
            xp = np.asarray(x, np.float64).copy()
            xp[i] += sgn * h
            c = np.asarray(f.build_params(xp)[0]["coeff"], np.float64)
            moved.update(int(j) for j in np.nonzero(c != c0)[0])
        out[name] = sorted(moved)
    return out


# ── gate 1: delegation / parent path / registration ────────────────────

def test_registered_as_a_delegate():
    assert WRAP in backends_for_model("pwa_gen")
    assert "delegate" in _tags(WRAP)


def _tags(name):
    from tabpwa.backends import backend_tags
    return backend_tags(name, "pwa_gen")


def test_the_parent_path_matches_a_maskless_kernel():
    f, kc, params, data = _fitter()
    w = _wrapper(kc)
    k = create_backend(KERNEL, kc, model="pwa_gen")
    try:
        assert w.mask is None
        Qw, gw, Pw = w.compute(params, w.load_data(data), norm=1.3)
        Qk, gk, Pk = k.compute(params, k.load_data(data), norm=1.3)
        assert _floor_ok(Pw, Pk, CACHE_FLOOR_P)
        assert _floor_ok(Qw, Qk)
        assert _floor_ok(gw["ck"], gk["ck"])
    finally:
        w.free()
        k.free()


def test_set_fixed_cols_matches_the_kernel_built_with_the_mask():
    f, kc, params, data = _fitter()
    w = _wrapper(kc)
    k = create_backend(KERNEL, kc, model="pwa_gen", fixed_cols=MASK_A,
                       fixed_values=_fv(kc, params, MASK_A))
    try:
        _set(w, MASK_A, params)
        Qw, gw, Pw = w.compute(params, w.load_data(data), norm=1.3)
        Qk, gk, Pk = k.compute(params, k.load_data(data), norm=1.3)
        assert _floor_ok(Pw, Pk, CACHE_FLOOR_P)
        assert _floor_ok(Qw, Qk)
        assert _floor_ok(gw["ck"], gk["ck"])
        assert _floor_ok(gw["coeff"], gk["coeff"], CACHE_FLOOR_COEFF)
        assert w.base.kernel.n_fix == 2
        assert w.resolution_size == k.resolution_size
    finally:
        w.free()
        k.free()


# ── gate 2: the mask is not a constructor argument ─────────────────────

@pytest.mark.parametrize("kw", [
    {"mask": np.zeros(10, np.int8)},
    {"fixed_cols": [5]},
    {"fixed_names": ["k2_s0"]},
    {"fixed_param_map": {"m": ["k2_s0"]}},
])
def test_the_constructor_rejects_mask_arguments(kw):
    f = Fitter(CFG_IDENT)
    f.apply_constrains()
    with pytest.raises(ValueError, match="set_fixed_cols"):
        _wrapper(f.kernel_config, **kw)


def test_a_base_that_ignores_the_key_is_rebuilt_and_unaffected():
    """Any per-event base is valid: it ignores ``coeff_fixed_cols``.

    The wrapper still rebuilds (one reload, documented), and the numbers are
    plain ``s0e2`` — P exact, the atomically-ordered quantities at the floor.
    """
    f, kc, params, data = _fitter()
    w = _wrapper(kc, base="cuda_flat_pwa_gen_s0e2")
    ref = create_backend("cuda_flat_pwa_gen_s0e2", kc, model="pwa_gen")
    h = w.load_data(data)
    try:
        Q0, g0, P0 = w.compute(params, h, norm=1.3)
        assert (w.creates, w.loads) == (1, 1)
        _set(w, MASK_A, params)
        assert w.mask.sum() == len(MASK_A)          # recorded for introspection
        assert (w.creates, w.loads) == (2, 1), "the base is rebuilt, samples not"
        Q1, g1, P1 = w.compute(params, h, norm=1.3)
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        assert np.array_equal(P1, Pr)
        assert _rel(Q1, Qr) <= S0F_TOL
        assert _rel(g1["ck"], gr["ck"]) <= S0F_TOL
        # and the kernel config the caller passed was not mutated
        assert "coeff_fixed_cols" not in kc
    finally:
        h.free()
        w.free()
        ref.free()


def test_set_fixed_cols_none_restores_the_parent_path():
    f, kc, params, data = _fitter()
    w = _wrapper(kc)
    h = w.load_data(data)
    try:
        _set(w, MASK_A, params)
        w.compute(params, h, norm=1.3)
        w.set_fixed_cols()
        assert w.mask is None
        assert w.base.kernel.n_fix == 0
        assert w.creates == 3
    finally:
        h.free()
        w.free()


# ── gate 3: rebuild per mask# ── gate 3: rebuild per mask# ── gate 3: rebuild per mask, lazy reload per sample ───────────────────

def test_each_mask_rebuilds_the_base_once_and_reloads_per_sample():
    f, kc, params, data = _fitter()
    w = _wrapper(kc)
    h = w.load_data(data)
    try:
        assert (w.creates, w.loads) == (1, 1)
        w.compute(params, h, norm=1.3)
        assert (w.creates, w.loads) == (1, 1), "an unchanged mask did nothing"
        _set(w, MASK_A, params)
        assert (w.creates, w.loads) == (2, 1), "the set must not touch samples"
        w.compute(params, h, norm=1.3)
        assert (w.creates, w.loads) == (2, 2)
        w.compute(params, h, norm=1.3)
        assert (w.creates, w.loads) == (2, 2), "no reload on an unchanged mask"
        _set(w, MASK_B, params)
        assert (w.creates, w.loads) == (3, 2)
        w.compute(params, h, norm=1.3)
        assert (w.creates, w.loads) == (3, 3)
    finally:
        h.free()
        w.free()


def test_a_b_a_matches_fresh_wrappers_per_mask():
    f, kc, params, data = _fitter()
    w = _wrapper(kc)
    h = w.load_data(data)
    try:
        got = {}
        for mask in (MASK_A, MASK_B, MASK_A):
            _set(w, mask, params)
            got[tuple(mask)] = w.compute(params, h, norm=1.3)
        assert w.creates == 4 and w.loads == 4   # construction + 3 masks
        for mask in (MASK_A, MASK_B):
            fresh = _wrapper(kc)
            try:
                _set(fresh, mask, params)
                Q, g, P = fresh.compute(params, fresh.load_data(data),
                                        norm=1.3)
                Qa, ga, Pa = got[tuple(mask)]
                assert _floor_ok(Pa, P, CACHE_FLOOR_P)
                assert _floor_ok(Qa, Q)
                assert _floor_ok(ga["ck"], g["ck"])
                assert _floor_ok(ga["coeff"], g["coeff"], CACHE_FLOOR_COEFF)
            finally:
                fresh.free()
    finally:
        h.free()
        w.free()


def test_two_samples_reload_independently_through_get_nll():
    cfg = CFG_IDENT
    f0 = Fitter(cfg)
    f0.apply_constrains()
    res = f0.build_params(f0.initial_values(seed=1))[1]
    fix = {n: float(res[n]) for n in
           ("k2_mass0", "k2_mass1", "k2_width0", "k2_width1")}

    def build(with_mask):
        spec = {"name": WRAP, "base": KERNEL}
        f = Fitter(cfg, backend=spec)
        f.apply_constrains()
        f.set_fixed(fix)
        if with_mask:
            _set_fitter(f, list(range(10)))
        f.set_phsp(_in_grid(f.kernel_config, 40, seed=4))
        f.set_data(_in_grid(f.kernel_config, 53, seed=5))
        return f, f.initial_values(seed=1)

    f, x = build(False)
    try:
        f.get_nll(x)                     # parent path, two samples loaded
        assert f.backend.loads == 2
        _set_fitter(f, list(range(10)))
        Q, g = f.get_nll(x)              # both samples reload
        assert f.backend.loads == 4
        g = np.asarray(g)
        fresh, xf = build(True)
        try:
            Qf, gf = fresh.get_nll(xf)
            gf = np.asarray(gf)
            assert _floor_ok(Q, Qf, CACHE_FLOOR, 1e-5), (Q, Qf)
            assert float(np.abs(g - gf).max()
                         / max(np.abs(gf).max(), 1e-300)) < 1e-4
        finally:
            fresh.backend.free()
    finally:
        f.backend.free()


# ── gate 4: rollback / free / memory ──────────────────────────────────

def test_a_failed_rebuild_rolls_back_to_the_previous_base_and_mask():
    f, kc, params, data = _fitter()
    w = _wrapper(kc)
    h = w.load_data(data)
    try:
        _set(w, MASK_A, params)
        Qa, ga, Pa = w.compute(params, h, norm=1.3)
        base_a, mask_a = w.base, w.mask.copy()
        creates_a = w.creates
        orig = w._make_base

        def boom(mask, values=None):
            if mask is not None and int(mask.sum()) == 10:
                w.creates += 1          # the attempt (as _make_base would)
                raise RuntimeError("cannot build this mask")
            return orig(mask, values)
        w._make_base = boom
        with pytest.raises(RuntimeError, match="cannot build"):
            _set(w, list(range(10)), params)
        w._make_base = orig
        # previous base (rebuilt) and mask are in place
        assert np.array_equal(w.mask, mask_a)
        assert w.base is not base_a          # the rollback built a fresh base
        assert w.mask.sum() == 5
        assert w.creates == creates_a + 2    # the failed one + the rollback
        Qb, gb, Pb = w.compute(params, h, norm=1.3)
        assert _floor_ok(Pb, Pa, CACHE_FLOOR_P)
        assert _floor_ok(Qb, Qa)
        assert _floor_ok(gb["ck"], ga["ck"])
        # and a valid change still works
        _set(w, MASK_B, params)
        fresh = _wrapper(kc)
        try:
            _set(fresh, MASK_B, params)
            Qf = fresh.compute(params, fresh.load_data(data), norm=1.3)[0]
            assert _floor_ok(w.compute(params, h, norm=1.3)[0], Qf)
        finally:
            fresh.free()
    finally:
        h.free()
        w.free()


def test_set_fixed_cols_none_returns_to_the_parent_path_bit_identically():
    f, kc, params, data = _fitter()
    w = _wrapper(kc)
    ref = create_backend("cuda_flat_pwa_gen_s0e2", kc, model="pwa_gen")
    h = w.load_data(data)
    try:
        _set(w, MASK_A, params)
        w.compute(params, h, norm=1.3)
        w.set_fixed_cols()
        assert w.mask is None and w.base.kernel.n_fix == 0
        Qw, gw, Pw = w.compute(params, h, norm=1.3)
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        assert np.array_equal(Pw, Pr)
        assert _rel(Qw, Qr) <= S0F_TOL
        assert _rel(gw["ck"], gr["ck"]) <= S0F_TOL
        assert _rel(gw["coeff"], gr["coeff"]) <= S0F_TOL * 10
    finally:
        h.free()
        w.free()
        ref.free()


def test_free_is_idempotent_and_refuses_stale_handles():
    f, kc, params, data = _fitter()
    w = _wrapper(kc)
    h = w.load_data(data)
    _set(w, MASK_A, params)
    w.compute(params, h, norm=1.3)
    base = w.base
    w.free()
    assert w.base is None and h.real is None
    assert base.kernel is None            # the base itself was freed
    w.free()                              # idempotent
    with pytest.raises(ValueError, match="unknown or freed"):
        w.compute(params, h, norm=1.3)


def test_repeated_mask_changes_do_not_grow_device_memory():
    """No MONOTONIC growth across cycles — the shape, not an absolute bound.

    An absolute ``|delta free| < k MiB`` gate cannot work here: the probe reads
    the WHOLE device, and a display/compositor allocates and frees between
    samples (measured 10–40 MiB swings on an attached box while a direct
    in-process probe showed 0.0 MiB drift across 16 cycles).  So the gate is
    that the last cycles are FLAT, plus the wrapper's own record count.

    The trace is compared WITHIN a mask: the split's scratch (``A1``/``dA``/
    ``dscale``) is now ONE ``cap``-sized allocation per CONTEXT, so it exists
    exactly on the masks whose split can run and a mask change legitimately
    allocates/frees it with the context.  A leak would still drift across the
    cycles that share a mask, which is what this checks.

    The probe is ``libcudart`` through ``ctypes`` (``tests/_cuda_mem.py``), NOT
    ``torch``: importing torch costs every later raw-CUDA call in the process
    5.1x (measured 5.71 -> 29.38 ms per create/load/compute/free cycle).
    """
    lib = _cuda_mem.require()
    f, kc, params, data = _fitter()
    w = _wrapper(kc)
    mask_params = params
    MiB = 1024 * 1024
    vals = []

    def probe():
        """Collect OUR own garbage first, then read the device.

        A context freed by an earlier test file (or by the previous cycle) is
        only reclaimed when its Python object dies, so an uncollected record
        lands inside a sample as a jump that has nothing to do with this
        wrapper.  The gate is about the cycles, so the trace measures a settled
        process.
        """
        gc.collect()
        _cuda_mem.sync(lib)
        return _cuda_mem.free_bytes(lib)

    try:
        probe()                              # settle before the first sample
        for i in range(12):
            _set(w, MASK_A if i % 2 else MASK_B, mask_params)
            h = w.load_data(data)
            w.compute(params, h, norm=1.3)
            h.free()
            assert len(w._records) == 0
            vals.append(probe())
        # even cycles are MASK_B, odd are MASK_A (i % 2)
        for name, group in (("MASK_B", vals[::2]), ("MASK_A", vals[1::2])):
            tail = group[-3:]
            assert max(tail) - min(tail) < 2 * MiB, (name, vals)
        assert be_live(w) == (1, 0)          # one base, no live records
    finally:
        w.free()


def be_live(w):
    return (1 if w.base is not None else 0, len(w._records))


def test_a_freed_handle_releases_its_record_and_sample():
    """5 ``set_phsp``/``set_data`` cycles leave only the live records, and the
    samples are COLLECTABLE — not merely unreferenced by the wrapper.

    ``Fitter.set_phsp`` frees the previous phase-space holder and ``set_data``
    overwrites the previous data holder, so the previous pair is released here
    explicitly (the Fitter's own overwrite is not a free).  Every freed handle
    must pop its record: otherwise each cycle leaves a sample dict behind
    (GBs at the production size).
    """
    import weakref
    f = Fitter(CFG_IDENT, backend={"name": WRAP, "base": KERNEL})
    f.apply_constrains()
    kc = f.kernel_config
    base = _in_grid(kc, 64, seed=0)
    w = f.backend
    refs = []
    prev = None
    try:
        for i in range(5):
            sample = {k: np.array(v, copy=True) for k, v in base.items()}
            refs.append(weakref.ref(sample["mass"]))
            if prev is not None:
                prev.free()
            f.set_phsp(sample)
            f.set_data(sample)
            _set_fitter(f, MASK_A if i % 2 else MASK_B)
            f.get_nll(f.initial_values(seed=1))
            prev = f._data_holder
        # two live handles (phsp + data) — no accumulated records
        assert len(w._records) == 2, (len(w._records), i)
        f._phsp_holder.free()
        f._data_holder.free()
        assert len(w._records) == 0
        # the Fitter itself keeps the numpy dicts for the norm gradient;
        # clear those references so the weakrefs can go dead
        f._data_np = f._phsp_np = None
        sample = prev = None
        gc.collect()
        assert all(r() is None for r in refs), (
            "the retained samples are still referenced after free()")
    finally:
        w.free()


# ── gate 5: the resolution rules still apply ──────────────────────────

def test_resolve_fixed_cols_maps_names_to_columns():
    """The NAME helper: names -> column indices (the values still come from
    ``coeff``), with the "a name fixes every column carrying it" rule."""
    f = Fitter(CFG_IDENT)
    f.apply_constrains()
    kc = f.kernel_config
    den = [str(n) for n in np.asarray(kc["den_names"]).ravel()]
    names = list(dict.fromkeys(den[:5]))
    cols = np.asarray([i for i, n in enumerate(den) if n in set(names)],
                      np.int64)
    with pytest.warns(UserWarning, match="param_map"):
        got = resolve_fixed_cols(kc["den_names"], names=names)
    assert np.array_equal(got, cols)
    # explicit columns are the same declaration without the helper
    m = mask_from_cols(cols, int(np.asarray(kc["basis_table"]).shape[0]))
    assert m.sum() == len(cols)


def test_a_repeated_den_name_fixes_every_column_it_owns():
    f = Fitter(CFG_IDENT)
    f.apply_constrains()
    kc = f.kernel_config
    den = [str(n) for n in np.asarray(kc["den_names"]).ravel()]
    n = int(np.asarray(kc["basis_table"]).shape[0])
    with pytest.warns(UserWarning, match="param_map"):
        cols = resolve_fixed_cols(kc["den_names"], names=sorted(set(den)))
    assert cols.size == n == len(den)


def test_the_helper_needs_a_declaration_to_return_anything():
    f = Fitter(CFG_KM)
    f.apply_constrains()
    den = f.kernel_config["den_names"]
    assert resolve_fixed_cols(den).size == 0
    with pytest.raises(ValueError, match="unknown names"):
        resolve_fixed_cols(den, names=["no_such_column"])


@pytest.mark.skipif(not __import__("os").path.exists(
    "benchmarks/ctpwa_kketa/config_gen.yml"),
    reason="the CTPWA benchmark config is not present")
def test_an_alias_map_refuses_a_partial_declaration():
    """The parameter map turns an under-covered alias into an error, and a
    parameter name expands to every column it drives."""
    from benchmark_s0f import CTPWA_FREE_MW, _pin_other_mw
    cfg = "benchmarks/ctpwa_kketa/config_gen.yml"
    f = Fitter(cfg)
    f.apply_constrains()
    _pin_other_mw(f, CTPWA_FREE_MW)
    kc = f.kernel_config
    x = f.initial_values(seed=1)
    den = [str(n) for n in np.asarray(kc["den_names"]).ravel()]
    per = _moving_by_param(f, x)
    pmap = {n: [den[i] for i in per[n]] for n in CTPWA_FREE_MW}
    with pytest.raises(ValueError, match="only part of the columns"):
        resolve_fixed_cols(kc["den_names"], names=["K3st1780p_s0"],
                           param_map=pmap)
    cols = resolve_fixed_cols(kc["den_names"], names=["K3st1780p_mass"],
                              param_map=pmap)
    assert np.isin([35, 36, 53, 54], cols).all()
    with pytest.warns(UserWarning, match="param_map"):
        one = resolve_fixed_cols(kc["den_names"], names=["K3st1780p_s0"])
    assert one.size == 1 and one[0] == 35 and 53 not in one


# ── the ONE declaration: columns + values, at both levels ─────────────

def _fv(kc, params, cols):
    n_basis = int(np.asarray(kc["basis_table"]).shape[0])
    coeff = np.asarray(params["coeff"], np.float64)
    out = np.zeros(n_basis, np.float64)
    c = np.asarray(list(cols), np.int64).ravel()
    out[c] = coeff[c]
    return out


def test_a_mask_without_values_is_refused_at_both_levels():
    f, kc, params, _data = _fitter()
    with pytest.raises(ValueError, match="needs the values of the columns"):
        _wrapper(kc).set_fixed_cols(MASK_A)
    with pytest.raises(ValueError, match="declared WITH the values"):
        create_backend(KERNEL, kc, model="pwa_gen", fixed_cols=MASK_A)
    # ... and values without a mask
    with pytest.raises(ValueError, match="no column is declared fixed"):
        create_backend(KERNEL, kc, model="pwa_gen",
                       fixed_values=_fv(kc, params, MASK_A))
    with pytest.raises(ValueError, match="no column is declared fixed"):
        resolve_mask_and_values(None, params["coeff"], kc["basis_table"].shape[0])


def test_a_mask_that_caches_nothing_warns_once_with_the_wave_and_columns():
    """The `one`-column diagnostic: a non-empty mask that caches NO wave is a
    silent no-op, so it warns — naming the closest wave and the columns it
    still reads — exactly once, and never for a PARTIAL mask."""
    f, kc, params, data = _fitter()
    w = _wrapper(kc)
    try:
        # CFG_IDENT's waves 2,3 read {5..9}; fixing 5,6 leaves `one` (and 7,8,9)
        # free, so no wave is fully fixed
        with pytest.warns(UserWarning, match="will not engage") as rec:
            _set(w, [5, 6], params)
        assert len(rec) == 1, [str(r.message) for r in rec]
        msg = str(rec[0].message)
        assert "wave 2" in msg, msg
        # every column it still reads is named (CFG_IDENT's second wave)
        assert "k2_m4" in msg and "k2_rho" in msg, msg
        assert w.base.kernel.n_fix == 0
        # a PARTIAL mask (the 27/34 production state) must not nag
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _set(w, MASK_A, params)
        assert w.base.kernel.n_fix == 2
        # and the parent path never warns
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            w.set_fixed_cols(None)
    finally:
        w.free()


def test_the_cache_is_filled_at_load_and_guarded_from_the_first_compute():
    """The eager path: `load_data` fills `Afix`, so the first `compute` is
    already guarded (there is no snapshot to write)."""
    f, kc, params, data = _fitter()
    w = _wrapper(kc)
    try:
        _set(w, MASK_A, params)
        h = w.load_data(data)
        assert w.base.kernel.afix_ready(h.real), "the fill must run in load_data"
        assert w.base.kernel.has_fixed_values()
        # the first compute passes (the call's coeff IS the declared one) ...
        Q, g, P = w.compute(params, h, norm=1.3)
        assert Q == Q
        # ... and a mismatch raises on the FIRST call of a fresh wrapper
        w2 = _wrapper(kc)
        try:
            _set(w2, MASK_A, params)
            bad = dict(params)
            c2 = np.asarray(params["coeff"], np.float64).copy()
            c2[MASK_A[0]] += 1e-9
            bad["coeff"] = c2
            h2 = w2.load_data(data)
            with pytest.raises(ValueError, match="declared fixed changed"):
                w2.compute(bad, h2, norm=1.3)
            h2.free()
        finally:
            w2.free()
        h.free()
    finally:
        w.free()


def test_clear_the_mask_needs_no_values():
    f, kc, params, data = _fitter()
    w = _wrapper(kc)
    try:
        _set(w, MASK_A, params)
        assert w.mask is not None and w.values is not None
        w.set_fixed_cols(None)               # clearing: no values
        assert w.mask is None and w.values is None
        assert w.base.kernel.n_fix == 0
        assert w.base.kernel.aux_bytes(1000) == 0
        # [] is the same declaration
        _set(w, MASK_A, params)
        w.set_fixed_cols([])
        assert w.mask is None
        # values with no columns is an error
        with pytest.raises(ValueError, match="no columns"):
            w.set_fixed_cols(None, params["coeff"])
    finally:
        w.free()


# ── gate 6: the wrapper module stays kernel-agnostic ──────────────────

def test_the_wrapper_is_kernel_agnostic():
    import ast
    import tabpwa.backends._fixed_coeff as mod
    src = open(mod.__file__, encoding="utf-8").read()
    for needle in ("gps0f", "s0f", "_flat_pwa_gen"):
        assert needle not in src, f"the wrapper mentions {needle!r}"
    tree = ast.parse(src)
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module or "")
    assert not {m for m in mods
                if "flat_pwa" in m or "gps0" in m or "s0f" in m
                or m.startswith("tabpwa.cuda")}, mods
    # and the class is a real ComputeBackend delegate
    from tabpwa.backends.core import ComputeBackend
    assert issubclass(FixedCoeffBackend, ComputeBackend)


def test_values_must_cover_every_declared_column():
    """A ``{col: value}`` mapping is the declaration written twice."""
    f, kc, params, _data = _fitter()
    w = _wrapper(kc)
    coeff = np.asarray(params["coeff"], np.float64)
    try:
        partial = {MASK_A[0]: float(coeff[MASK_A[0]])}
        with pytest.raises(ValueError, match="do not cover the declaration"):
            w.set_fixed_cols(MASK_A, partial)
        # the full mapping is accepted
        w.set_fixed_cols(MASK_A, {int(c): float(coeff[int(c)]) for c in MASK_A})
    finally:
        w.free()


def test_a_0_or_1_vector_is_not_a_column_list():
    """``fixed_cols`` is COLUMN INDICES; a mask-shaped vector is refused."""
    f, kc, params, _data = _fitter()
    n_basis = int(np.asarray(kc["basis_table"]).shape[0])
    ones = np.zeros(n_basis, np.int8)
    ones[MASK_A] = 1
    with pytest.raises(ValueError, match="COLUMN INDICES"):
        create_backend(KERNEL, kc, model="pwa_gen", fixed_cols=ones,
                       fixed_values=np.asarray(params["coeff"]))
    with pytest.raises(ValueError, match="COLUMN INDICES"):
        create_backend(KERNEL, kc, model="pwa_gen",
                       fixed_cols=ones.astype(bool),
                       fixed_values=np.asarray(params["coeff"]))


def test_the_mask_needs_a_coefficient_basis():
    """A non-``pwa_gen`` model has no ``basis_table``: the error names it."""
    spec = {"name": WRAP, "base": "numpy_pwa"}
    f = Fitter("tests/config_pwa.yml", backend=spec)
    f.apply_constrains()
    with pytest.raises(ValueError, match="amp_model: pwa_gen"):
        f.backend.set_fixed_cols([0], [1.0])


def test_an_empty_declaration_with_values_raises_in_both_spellings():
    f, kc, params, _data = _fitter()
    w = _wrapper(kc)
    try:
        with pytest.raises(ValueError, match="no columns"):
            w.set_fixed_cols(None, params["coeff"])
        with pytest.raises(ValueError, match="no columns"):
            w.set_fixed_cols([], params["coeff"])
    finally:
        w.free()


def test_the_old_config_keys_are_ignored():
    """``coeff_fixed`` / ``coeff_fixed_cols`` are not mask sources any more.

    A key cannot carry the VALUES, so it was a second API: a config that still
    carries one must build the PARENT path (no cache), never a half-declared
    mask.
    """
    f = Fitter(CFG_IDENT)
    f.apply_constrains()
    kc = dict(f.kernel_config)
    kc["coeff_fixed_cols"] = MASK_A
    kc["coeff_fixed"] = ["one"]
    k = create_backend(KERNEL, kc, model="pwa_gen")
    try:
        assert k.kernel.n_fix == 0
        assert k.kernel.mask is None
    finally:
        k.free()


def test_the_wrapper_does_not_mutate_the_callers_config():
    f, kc, params, data = _fitter()
    before = dict(kc)
    w = _wrapper(kc)
    h = None
    try:
        _set(w, MASK_A, params)
        assert "coeff_fixed_cols" not in kc
        assert set(kc) == set(before)
        h = w.load_data(data)
        w.compute(params, h, norm=1.3)
    finally:
        if h is not None:
            h.free()
        w.free()


# ── gate 8: `set_fixed_coeffs(fitter)` — rule out / derive ───────────────

BENCH_CFG = "benchmarks/ctpwa_kketa/config_gen.yml"
FREE_TEN = ["phi1680_mass", "phi1680_width",
            "X1750_mass", "X1750_width",
            "omega2220_mass", "omega2220_width",
            "K3st1780p_mass", "K3st1780p_width",
            "K4st2045p_mass", "K4st2045p_width"]


def _bench_fitter(with_mask=False, mask=None):
    """The CTPWA benchmark with the paper's ten free mass/width parameters."""
    wback = {"name": WRAP, "base": KERNEL}
    f = Fitter(BENCH_CFG, backend=wback)
    f.apply_constrains()
    f.set_fixed({n: float(f.defaults[n])
                 for n in dict.fromkeys(f.var_registry.flat_names)
                 if ("_mass" in n or "_width" in n) and n not in FREE_TEN
                 and n in f.defaults})
    if with_mask:
        if mask is None:
            mask = f.model.fixed_coeff_columns(f.cm)
        _set_fitter(f, mask)
    f.set_phsp(_in_grid(f.kernel_config, 40, seed=4))
    f.set_data(_in_grid(f.kernel_config, 53, seed=5))
    return f


def test_set_fixed_coeffs_equals_the_explicit_columns():
    """One call from the fitter == ``set_fixed_cols`` with the same columns."""
    f = _bench_fitter()
    cols = f.model.fixed_coeff_columns(f.cm)
    assert cols and len(cols) < len(f.kernel_config["den_names"])

    f.backend.set_fixed_coeffs(f)
    assert f.backend.mask is not None
    assert sorted(np.nonzero(f.backend.mask)[0].tolist()) == sorted(cols)

    # an independent wrapper with the same columns installs the same mask
    w = _wrapper(f.kernel_config)
    try:
        _set(w, cols, f.build_params(f.initial_values(seed=1))[0])
        assert np.array_equal(w.mask, f.backend.mask)
    finally:
        w.free()

    # ... and produces the same NLL and gradient (the mask is a permutation)
    x = f.initial_values(seed=1)
    Qa, ga = f.get_nll(x)
    ref = _bench_fitter(with_mask=True, mask=cols)
    try:
        Qb, gb = ref.get_nll(x)
        assert _floor_ok(np.asarray(ga), np.asarray(gb), 1e-12, 1e-12)
        assert abs(float(Qa) - float(Qb)) <= 1e-12 * max(abs(float(Qb)), 1.0)
    finally:
        ref.backend.free()


def test_set_fixed_coeffs_jacobian_takes_x_from_the_fitter():
    f = _bench_fitter()
    f.backend.set_fixed_coeffs(f, how="jacobian")
    picked = sorted(np.nonzero(f.backend.mask)[0].tolist())

    explicit = f.model.fixed_coeff_columns(f.cm, x=f.initial_values(seed=1),
                                          how="jacobian")
    assert picked == sorted(explicit)

    # the same, stated with x given explicitly
    f.backend.set_fixed_cols(None)          # clears (needs no values)
    f.backend.set_fixed_coeffs(f, how="jacobian",
                            x=f.initial_values(seed=1))
    assert sorted(np.nonzero(f.backend.mask)[0].tolist()) == sorted(explicit)


def test_set_fixed_coeffs_propagates_the_model_error():
    _, kc, _, _ = _fitter()
    w = _wrapper(kc)
    try:
        f = Fitter("tests/config_pwa.yml")
        f.apply_constrains()
        with pytest.raises(TypeError, match="pwa_gen"):
            w.set_fixed_coeffs(f)
        assert w.mask is None, "a rejected derivation must not touch the mask"
        with pytest.raises(ValueError, match="(?s)transform.*jacobian"):
            w.set_fixed_coeffs(f, how="bogus")
        assert w.mask is None
    finally:
        w.free()


def test_the_convenience_is_the_wrappers_only_and_names_coefficients():
    """`ComputeBackend` stays mask-agnostic; the name is not `set_fixed`."""
    from tabpwa.backends.core import ComputeBackend
    assert not hasattr(ComputeBackend, "set_fixed")
    assert not hasattr(ComputeBackend, "set_fixed_coeffs")
    assert "set_fixed" not in vars(FixedCoeffBackend), (
        "`set_fixed` would be confused with the Fitter's PARAMETER setter")
    assert hasattr(FixedCoeffBackend, "set_fixed_coeffs")


# ── gate 9: the vocabulary names the thing ─────────────────────────────

def test_no_general_mask_vocabulary_remains():
    """The subject is "which coefficient columns are FIXED", not a mask.

    The user-facing names are `fixed_coeff_spec` / `resolve_fixed_cols` /
    `mask_from_cols` / `set_fixed_cols` / `set_fixed_coeffs`, and the wrapper is
    `backends/_fixed_coeff.py` with `FixedCoeffBackend`.  An ``int8[n_basis]``
    array keeps the word *mask* internally — that IS a bitmask — so this bans
    those public spellings only, and never inside this file (which spells them
    in order to forbid them).
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    me = pathlib.Path(__file__).resolve()
    banned = ("mask_spec", "FIXED_COEFF_KEY", "FIXED_VALUES_KEY",
              "coeff_fixed_cols", "coeff_fixed_values",
              "mask_from_indices", "MASK_CONFIG_KEY", "set_coeff_mask",
              "_masked_tile", "MaskedTileBackend", "_with_mask")

    offenders = []
    for sub in ("src", "tests"):
        for path in sorted((root / sub).rglob("*.py")):
            if path.resolve() == me:
                continue
            text = path.read_text(encoding="utf-8")
            offenders.extend(f"{path.relative_to(root)}: {word}"
                             for word in banned if word in text)
    assert offenders == [], "old vocabulary still referenced:\n" + "\n".join(
        offenders)

    assert not (root / "src" / "tabpwa" / "mask_spec.py").exists()
    assert (root / "src" / "tabpwa" / "fixed_coeff_spec.py").is_file()
    assert not (root / "src" / "tabpwa" / "backends" / "_masked_tile.py").exists()
    assert (root / "src" / "tabpwa" / "backends" / "_fixed_coeff.py").is_file()

    # the wrapper answers to the renamed method and to nothing else
    assert hasattr(FixedCoeffBackend, "set_fixed_coeffs")
    assert "set_coeff_mask" not in vars(FixedCoeffBackend)


def test_the_wrapper_module_and_class_are_registered_under_the_concept():
    """The import path and the class name say "fixed coefficient", not "tile"."""
    import tabpwa.backends._fixed_coeff as mod
    assert mod.FixedCoeffBackend is FixedCoeffBackend
    assert FixedCoeffBackend.__module__ == "tabpwa.backends._fixed_coeff"
    assert not hasattr(mod, "MaskedTileBackend")
