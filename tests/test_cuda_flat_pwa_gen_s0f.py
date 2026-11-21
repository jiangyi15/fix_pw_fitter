"""``cuda_flat_pwa_gen_s0f`` — the s0e fused tile with a FIXED-WAVE cache.

The design in one line: the host permutes the wave axis (fixed waves first),
builds ``gamma``/the CSR for the FREE waves only, and caches the fixed waves'
finished amplitude contribution ``Afix = Amp·conj(D)/|D|²`` per handle.  The
tile then runs its parent's phases over ``n_free`` plus two small fixed loops.

Gates (this kernel's own; the MASK contract every masked backend shares is
``tests/test_flat_masked.py``, selected by the ``fixed_coeff`` tag):

1. **registration + case-1** — ``pwa_gen`` only, own module/``.so``/symbols,
   one loaded library per process (no CUDA context duplication).
2. **mask-off is the parent** — ``n_fix == 0`` is the identity permutation,
   the full CSR and no cache, so it must agree with
   ``cuda_flat_pwa_gen_s0e2`` on Q/P/dck/dcoeff/dnorm, at ``rsize`` 1 and 17
   with ``norm=None``/``1.3``.  ``P`` is bit-identical; the other quantities
   are accumulated by ``atomicAdd`` across blocks and the PARENT itself is not
   bit-reproducible against itself at this sample size (measured in
   ``test_the_parent_is_not_bit_reproducible_so_the_bound_is_the_floor``), so
   they are gated at the measured 1e-14 re-association floor.
3. **finite differences** on ``ck`` and the free ``coeff`` columns, through
   BOTH fixed-block paths (``split_fixed`` on/off).
4. **the split's own levers** — F/B launch counts per chunk, the ``aux_bytes``
   context formula, the ``fix_bytes`` fused form, the fused branch vs F/T/B,
   the fused fallback when a group cannot fit one block, ``bwd_mode`` A/B.
5. **``chunk_plan``** and the host plan arithmetic (``wave_plan`` /
   ``split_waves`` / ``index_runs``, the CSR partition) in numpy, before the
   kernel is consulted.
6. **the CTPWA alias** — the per-PARAMETER column map (the union over the ten
   free parameters is the shared gate).
7. ``n_res != 1`` is rejected naming ``cuda_flat_pwa_gen2``; ``te``/``pick_te``
   never returns a tile ``create`` rejects.
"""

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import backends_for_model, create_backend
from tabpwa.cuda._flat_pwa_gen_s0f import (
    CUDAKernelFlatPwaGenS0F, index_runs, split_waves, wave_plan)
import _cuda_mem
from test_cuda_flat_cache_pwa_gen import _in_grid

CFG_ONE = "tests/config_pwa_gen.yml"
CFG_KM = "tests/config_pwa_gen_km.yml"
CFG_MB = "tests/config_pwa_gen_mb.yml"
CFG_IDENT = "tests/config_pwa_gen_ident.yml"
CFG_SAME = "tests/config_pwa_gen_same.yml"
CFG_CP = "tests/config_pwa_gen_cp.yml"
QUALIFYING = [CFG_ONE, CFG_KM, CFG_IDENT, CFG_SAME, CFG_CP]

# ── the fp32 cache floor (the family's; the cache set is unchanged) ─────
CACHE_FLOOR = 2e-5          # Q, dck, dnorm
CACHE_FLOOR_P = 3e-5        # per-event P
CACHE_FLOOR_COEFF = 1e-4    # dcoeff accumulates one rounding per (e,k,m)
CACHE_FLOOR_ABS = 1e-4
NH = 1e-6

#: s0f's only difference against s0e2 with no mask is that a handful of
#: index expressions are written through `n_fix + k` with `n_fix == 0`.  The
#: arithmetic is identical, but Q/dck/dcoeff/dnorm are `atomicAdd`-accumulated
#: across blocks and the PARENT is itself not bit-reproducible at 64 events
#: (see the dedicated test below) — so the gate is the measured 1e-14
#: re-association floor, ten orders below the fp32 cache floor.
S0F_TOL = 1e-14
S0F_TOL_COEFF = 1e-14

#: the ten mass/width names the CTPWA benchmark frees (benchmarks/ctpwa_kketa/
#: README.md §4).  `bench.py --free-mw ...` pins every OTHER mass/width.
CTPWA_CFG = "benchmarks/ctpwa_kketa/config_gen.yml"
CTPWA_FREE_MW = ("phi1680_mass", "phi1680_width", "X1750_mass", "X1750_width",
                 "omega2220_mass", "omega2220_width", "K3st1780p_mass",
                 "K3st1780p_width", "K4st2045p_mass", "K4st2045p_width")


def _rel(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _floor_ok(a, b, rel_bound=CACHE_FLOOR, abs_bound=CACHE_FLOOR_ABS):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    worst = float(np.abs(a - b).max())
    return (worst / max(np.abs(b).max(), 1e-300)) < rel_bound or worst < abs_bound


def _replicate(data, rs):
    out = {k: np.repeat(np.asarray(v), rs, axis=0) for k, v in data.items()}
    out["weight"] = np.full(len(np.asarray(data["weight"])) * rs, 1.0 / rs)
    return out


def _pin_other_mw(f, keep):
    """``bench.py --free-mw``: pin every mass/width whose NAME is not in keep."""
    known = set(dict.fromkeys(f.var_registry.flat_names))
    unknown = sorted(set(keep) - known)
    assert not unknown, unknown
    mw = {n: float(f.defaults[n])
          for n in dict.fromkeys(f.var_registry.flat_names)
          if n.endswith(("_mass", "_width")) and n not in set(keep)
          and n in f.defaults}
    f.set_fixed(mw)


def _moving_by_param(f, x, step=1e-3):
    """``{free parameter name: [coeff columns its perturbation moves]}``.

    Ground truth for the mask: it does NOT assume a name prefix, so an ALIASED
    parameter is reported through the columns it really moves.  This is the
    check a name-prefix mask fails: `K3st1780p_mass` moves the `K3st1780m_*`
    columns too, so fixing only `K3st1780p_s0` would cache a column that moves.
    """
    names = list(f.fwd_names())
    assert len(names) == len(x)
    c0 = np.asarray(f.build_params(x)[0]["coeff"], np.float64)
    out = {}
    for i, name in enumerate(names):
        h = step * max(abs(float(x[i])), 1.0)
        moved = set()
        for sgn in (1.0, -1.0):
            xp = np.asarray(x, np.float64).copy()
            xp[i] += sgn * h
            c = np.asarray(f.build_params(xp)[0]["coeff"], np.float64)
            moved.update(int(j) for j in np.nonzero(c != c0)[0])
        out[name] = sorted(moved)
    return out


def _ctpwa_fitter(free=CTPWA_FREE_MW):
    """The CTPWA model with everything but ``free`` pinned (bench.py's setup)."""
    f = Fitter(CTPWA_CFG)
    f.apply_constrains()
    _pin_other_mw(f, free)
    return f, f.initial_values(seed=1)


def _as_mask(kc, cols):
    """``int8 [n_basis]`` from column indices (the kernel's mask format)."""
    n_basis = int(np.asarray(kc["basis_table"]).shape[0])
    m = np.zeros(n_basis, np.int8)
    if cols is not None:
        m[np.asarray(list(cols), np.int64)] = 1
    return m


def _params(f, mask_filter=None):
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    return params


def _fv(kc, params, cols):
    """The full `coeff` vector at the declared columns.

    A mask is declared WITH its values (the ``Afix`` cache is filled at
    ``load_data``), so every kernel built for a mask in this fixture passes
    these.
    """
    n_basis = int(np.asarray(kc["basis_table"]).shape[0])
    coeff = np.asarray(params["coeff"], np.float64)
    out = np.zeros(n_basis, np.float64)
    c = np.asarray(list(cols), np.int64).ravel()
    out[c] = coeff[c]
    return out


# ── gate 1: registration and the case-1 copy ────────────────────────────

def test_registration():
    assert "cuda_flat_pwa_gen_s0f" in set(backends_for_model("pwa_gen"))
    assert "cuda_flat_pwa_gen_s0f" not in set(backends_for_model("pwa"))


def test_case1_own_module_library_and_symbols():
    """Own module, own ``.so``, own symbols; the parent is untouched."""
    import tabpwa.cuda._flat_pwa_gen_s0e2 as parent
    import tabpwa.cuda._flat_pwa_gen_s0f as new
    assert new is not parent
    f = Fitter(CFG_KM)
    f.apply_constrains()
    a = CUDAKernelFlatPwaGenS0F(f.kernel_config)
    b = parent.CUDAKernelFlatPwaGenS0E2(f.kernel_config)
    try:
        assert new._lib is not None and parent._lib is not None
        assert new._lib is not parent._lib
        assert a._lib is not b._lib
        assert hasattr(a._lib, "cuda_compute_flat_pwa_gen_s0f")
        assert not hasattr(a._lib, "cuda_compute_flat_pwa_gen_s0e2")
        assert hasattr(b._lib, "cuda_compute_flat_pwa_gen_s0e2")
        assert not hasattr(b._lib, "cuda_compute_flat_pwa_gen_s0f")
    finally:
        a.free()
        b.free()


def test_one_library_per_process():
    """Two kernels share the module's single ``.so`` — no second lib/context."""
    import tabpwa.cuda._flat_pwa_gen_s0f as new
    f = Fitter(CFG_KM)
    f.apply_constrains()
    a = CUDAKernelFlatPwaGenS0F(f.kernel_config)
    b = CUDAKernelFlatPwaGenS0F(f.kernel_config)
    try:
        assert a._lib is b._lib is new._lib
    finally:
        a.free()
        b.free()


# ── the host-side plan, in numpy, before the kernel is consulted ────────

def test_the_plan_is_the_identity_when_nothing_is_fixed():
    for cfg in QUALIFYING:
        f = Fitter(cfg)
        f.apply_constrains()
        kc = f.kernel_config
        gs_lo, gs_hi, nz_k, nz_m = index_runs(kc["gamma_matrix"],
                                              np.asarray(kc["basis_table"]).shape[0])
        n_wave = gs_lo.size
        perm, inv, fw = wave_plan(np.zeros(len(kc["den_names"]), np.int8),
                                  gs_lo, gs_hi, nz_m)
        assert not fw.any()
        assert np.array_equal(perm, np.arange(n_wave))
        assert np.array_equal(inv, np.arange(n_wave))
        sp = split_waves(gs_lo, gs_hi, nz_k, nz_m,
                         np.zeros_like(nz_m), np.zeros_like(nz_m), fw)
        assert sp["n_free"] == n_wave and sp["n_fix"] == 0
        assert np.array_equal(sp["gs_lo"], gs_lo)
        assert np.array_equal(sp["gs_hi"], gs_hi)
        assert np.array_equal(sp["nz_k"], nz_k)
        assert np.array_equal(sp["nz_m"], nz_m)
        assert sp["fix_pair"].size == 0


def test_the_free_and_fixed_csr_partition_the_full_one():
    """Every non-zero lands in exactly one block; the free wave indices are
    contiguous ``0 .. n_free-1``."""
    f = Fitter(CFG_IDENT)
    f.apply_constrains()
    kc = f.kernel_config
    den = list(kc["den_names"])
    gs_lo, gs_hi, nz_k, nz_m = index_runs(kc["gamma_matrix"], len(den))
    mask = np.zeros(len(den), np.int8)
    mask[5:] = 1
    perm, inv, fw = wave_plan(mask, gs_lo, gs_hi, nz_m)
    sp = split_waves(gs_lo, gs_hi, nz_k, nz_m,
                     np.zeros_like(nz_m), np.zeros_like(nz_m), fw)
    assert sp["n_fix"] + sp["n_free"] == gs_lo.size
    assert sp["nz_m"].size + sp["fix_pair"].size == nz_m.size
    assert np.array_equal(np.unique(sp["nz_k"]),
                          np.arange(sp["n_free"]))
    # a fixed wave is one whose every CSR column is declared fixed
    for k in np.nonzero(fw)[0]:
        assert mask[nz_m[gs_lo[k]:gs_hi[k]]].all()
    for k in np.nonzero(~fw)[0]:
        assert not mask[nz_m[gs_lo[k]:gs_hi[k]]].all()


# ── gate 2: mask-off is the parent, bit for bit where the parent is ─────

@pytest.mark.parametrize("cfg", [CFG_ONE, CFG_KM])
@pytest.mark.parametrize("rs", [1, 17])
@pytest.mark.parametrize("norm", [None, 1.3])
def test_mask_off_matches_s0e2(cfg, rs, norm):
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    data = (_in_grid(kc, 64, seed=2) if rs == 1
            else _replicate(_in_grid(kc, 17, seed=5), rs))
    a = create_backend("cuda_flat_pwa_gen_s0e2", kc, model="pwa_gen",
                       resolution_size=rs)
    b = create_backend("cuda_flat_pwa_gen_s0f", kc, model="pwa_gen",
                       resolution_size=rs)
    ha, hb = a.load_data(data), b.load_data(data)
    try:
        assert b.kernel.n_fix == 0
        # gate: with nothing fixed, kernels F and B are never launched and
        # the context's split scratch does not exist (the parent's path)
        assert b.kernel.aux_bytes(hb.ne) == 0
        Qa, ga, Pa = a.compute(params, ha, norm=norm)
        Qb, gb, Pb = b.compute(params, hb, norm=norm)
        assert b.kernel.split_counts() == (0, 0)
        # P has one writer per event -> exact; the rest is atomicAdd-ordered
        assert np.array_equal(Pa, Pb), "P must be bit-identical"
        for name, y, x in (("Q", Qb, Qa), ("dck", gb["ck"], ga["ck"]),
                           ("dcoeff", gb["coeff"], ga["coeff"])):
            assert _rel(y, x) <= (S0F_TOL_COEFF if name == "dcoeff"
                                  else S0F_TOL), f"{name}: {_rel(y, x):.2e}"
        if norm is not None:
            assert _rel(gb["norm"], ga["norm"]) <= S0F_TOL
    finally:
        ha.free()
        hb.free()
        a.free()
        b.free()


def test_the_parent_is_not_bit_reproducible_so_the_bound_is_the_floor():
    """Why gate 2 is 1e-14 and not ``np.array_equal`` on every quantity.

    ``Q``, ``dck`` and ``dcoeff`` are summed by ``atomicAdd`` across blocks, so
    two runs of the SAME parent kernel (different handles, same data) already
    differ by a few ulp.  Measured here, so the tolerance above is the floor
    and not a weakened gate.
    """
    f = Fitter(CFG_ONE)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    data = _in_grid(kc, 64, seed=2)
    a = create_backend("cuda_flat_pwa_gen_s0e2", kc, model="pwa_gen")
    b = create_backend("cuda_flat_pwa_gen_s0e2", kc, model="pwa_gen")
    try:
        ha, hb = a.load_data(data), b.load_data(data)
        differs = False
        for _ in range(6):
            Qa, ga, _ = a.compute(params, ha, norm=None)
            Qb, gb, _ = b.compute(params, hb, norm=None)
            differs |= (Qa != Qb or not np.array_equal(ga["ck"], gb["ck"])
                        or not np.array_equal(ga["coeff"], gb["coeff"]))
        assert differs, (
            "the parent became bit-reproducible — tighten gate 2 to "
            "np.array_equal on every quantity")
    finally:
        a.free()
        b.free()


# ── the masked fixtures: mask-on vs the reference, the zero contract, the
#    partly-fixed wave and the end-to-end path are the SHARED gates
#    (``tests/test_flat_masked.py``, selected by the ``fixed_coeff`` tag).
#    What stays here is the kernel's own: the split's launch counts and
#    buffers, its A/B levers, and the finite differences through both
#    fixed-block paths.

def _masked_pair(cfg, mask, rs=1, **kw):
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    ref = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=rs)
    m = (mask)
    gpu = create_backend("cuda_flat_pwa_gen_s0f", kc, model="pwa_gen",
                         resolution_size=rs, fixed_cols=m,
                         fixed_values=_fv(kc, _params(f), m), **kw)
    return f, kc, ref, gpu


# ── gate 5: finite differences through the masked path ─────────────────

def test_coeff_gradient_fd_on_the_free_columns():
    f, kc, ref, gpu = _masked_pair(CFG_IDENT, [5, 6, 7, 8, 9])
    params = _params(f)
    data = _in_grid(kc, 48, seed=3)
    h = gpu.load_data(data)
    try:
        _Q0, g0, _ = gpu.compute(params, h, norm=None)
        coeff = np.asarray(params["coeff"], np.float64).copy()
        free = np.setdiff1d(np.arange(coeff.size), gpu.kernel.watched_cols)
        for i in free:
            up, dn = coeff.copy(), coeff.copy()
            up[i] += NH
            dn[i] -= NH
            pu, pn = dict(params), dict(params)
            pu["coeff"], pn["coeff"] = up, dn
            fd = (gpu.compute(pu, h, norm=None)[0]
                  - gpu.compute(pn, h, norm=None)[0]) / (2 * NH)
            assert np.isclose(g0["coeff"][i], fd, rtol=1e-4, atol=1e-8), (
                f"row {i}: kernel {g0['coeff'][i]!r} vs FD {fd!r}")
        for i in gpu.kernel.watched_cols:
            assert g0["coeff"][i] == 0.0
    finally:
        gpu.free()
        ref.free()


@pytest.mark.parametrize("split", [True, False])
def test_ck_gradient_fd_through_the_masked_path(split):
    """``dck`` against central differences, for BOTH fixed-block paths.

    The split's fixed ``dck`` comes from kernel B (a GEMV over the tile's
    finished ``dA``), which is a NEW gradient path — the FD is what proves it,
    not the reference comparison.  ``split=False`` is the in-tile loop, so the
    two parametrisations also pin that the split did not change the gradient.
    """
    f, kc, ref, gpu = _masked_pair(CFG_IDENT, [5, 6, 7, 8, 9],
                                   split_fixed=split)
    params = _params(f)
    data = _in_grid(kc, 48, seed=4)
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
                dres = (2.0 * np.real(g0["ck"][i]) if part == 0
                        else -2.0 * np.imag(g0["ck"][i]))
                assert np.isclose(dres, fd, rtol=1e-4, atol=1e-8), (
                    f"ck[{i}] part {part}: {dres!r} vs {fd!r}")
    finally:
        gpu.free()
        ref.free()


def test_the_split_launches_f_and_b_only_when_something_is_fixed():
    """Gate: mask-off never runs F/B; mask-on runs them exactly once per call.

    And ``split_fixed=False`` keeps the fixed block in the tile: no F/B, no
    buffers (the benchmark lever, not the production path).
    """
    f, kc, ref, gpu = _masked_pair(CFG_IDENT, [5, 6, 7, 8, 9])
    params = _params(f)
    h = gpu.load_data(_in_grid(kc, 32, seed=1))
    try:
        assert gpu.kernel.n_fix == 2
        # the split's scratch is the CONTEXT's (cap-sized), not the handle's
        assert gpu.kernel.aux_bytes(h.ne) > 0
        gpu.compute(params, h, norm=1.3)
        assert gpu.kernel.split_counts() == (1, 1)
        gpu.compute(params, h, norm=1.3)
        assert gpu.kernel.split_counts() == (2, 2)
    finally:
        h.free()
        gpu.free()
        ref.free()
    # the in-tile path: same mask, no split
    m = ([5, 6, 7, 8, 9])
    g2 = create_backend("cuda_flat_pwa_gen_s0f", kc, model="pwa_gen",
                        fixed_cols=m, fixed_values=_fv(kc, params, m),
                        split_fixed=False)
    h2 = g2.load_data(_in_grid(kc, 32, seed=1))
    try:
        assert g2.kernel.n_fix == 2
        g2.compute(params, h2, norm=1.3)
        assert g2.kernel.aux_bytes(h2.ne) == 0
        assert g2.kernel.split_counts() == (0, 0)
    finally:
        h2.free()
        g2.free()
    # and with nothing fixed there is nothing to launch either way
    g3 = create_backend("cuda_flat_pwa_gen_s0f", kc, model="pwa_gen")
    h3 = g3.load_data(_in_grid(kc, 32, seed=1))
    try:
        assert g3.kernel.aux_bytes(h3.ne) == 0
        g3.compute(params, h3, norm=1.3)
        assert g3.kernel.split_counts() == (0, 0)
    finally:
        h3.free()
        g3.free()


def test_the_split_matches_the_in_tile_path():
    """Same mask, split on vs off: one objective, one gradient.

    The two differ only by where the fixed block is summed and by ``dA``'s
    float2 store, so the bound is the fp32 floor — not the 1e-14 ordering one.
    """
    f = Fitter(CFG_IDENT)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    data = _in_grid(kc, 64, seed=0)
    m = ([5, 6, 7, 8, 9])
    v = _fv(kc, params, m)
    a = create_backend("cuda_flat_pwa_gen_s0f", kc, model="pwa_gen",
                       fixed_cols=m, fixed_values=v, split_fixed=False)
    b = create_backend("cuda_flat_pwa_gen_s0f", kc, model="pwa_gen",
                       fixed_cols=m, fixed_values=v)
    try:
        Qa, ga, Pa = a.compute(params, a.load_data(data), norm=1.3)
        Qb, gb, Pb = b.compute(params, b.load_data(data), norm=1.3)
        assert _floor_ok(Pb, Pa, CACHE_FLOOR_P)
        assert _floor_ok(Qb, Qa)
        assert _floor_ok(gb["ck"], ga["ck"])
        assert _floor_ok(gb["coeff"], ga["coeff"], CACHE_FLOOR_COEFF)
    finally:
        a.free()
        b.free()


def test_the_aux_bytes_report_is_the_per_context_formula():
    """``cap·P·16 + cap·P·8 + cap·8`` — the CONTEXT's one allocation.

    It ignores ``ne`` (it is batch-sized) and tracks ``cap``: a smaller
    ``batch_size`` gives a smaller figure for the same sample.
    """
    f, kc, ref, gpu = _masked_pair(CFG_IDENT, [5, 6, 7, 8, 9])
    k = gpu.kernel
    try:
        formula = k.cap * k.n_proj * (16 + 8) + k.cap * 8
        for ne in (0, 1, 64, 10 ** 6):
            assert k.aux_bytes(ne) == formula, ne
        m = ([5, 6, 7, 8, 9])
        v = _fv(kc, _params(f), m)
        small = CUDAKernelFlatPwaGenS0F(
            kc, fixed_cols=m, fixed_values=v, batch_size=8)
        try:
            assert small.cap < k.cap
            assert small.aux_bytes(10 ** 6) == (
                small.cap * small.n_proj * (16 + 8) + small.cap * 8)
        finally:
            small.free()
        plain = CUDAKernelFlatPwaGenS0F(kc)
        try:
            assert plain.aux_bytes(1000) == 0
        finally:
            plain.free()
        off = CUDAKernelFlatPwaGenS0F(kc, fixed_cols=m, fixed_values=v,
                                      split_fixed=False)
        try:
            assert off.aux_bytes(1000) == 0
        finally:
            off.free()
    finally:
        gpu.free()
        ref.free()


def test_load_data_does_not_allocate_the_scratch():
    """The split's scratch belongs to the CONTEXT: a load adds no per-handle
    copy (its ``cap``-sized allocation happened at construction).

    The probe is ``libcudart`` through ``ctypes`` (``tests/_cuda_mem.py``), NOT
    ``torch``: importing torch costs every later raw-CUDA call in the process
    5.1x (measured 5.71 -> 29.38 ms per create/load/compute/free cycle).
    """
    lib = _cuda_mem.require()
    f, kc, ref, gpu = _masked_pair(CFG_IDENT, [5, 6, 7, 8, 9])
    k = gpu.kernel
    aux = k.aux_bytes(0)
    assert aux >= 2 * 1024 * 1024, aux      # MBs, easily visible
    data = _in_grid(kc, 16, seed=5)
    try:
        _cuda_mem.sync(lib)
        base = _cuda_mem.free_bytes(lib)
        h = gpu.load_data(data)             # tiny sample: us/amp are tiny
        _cuda_mem.sync(lib)
        after = _cuda_mem.free_bytes(lib)
        assert base - after < aux // 2, (base, after, aux)
        h.free()
    finally:
        gpu.free()
        ref.free()


def test_all_fixed_with_the_grouped_objective():
    """``n_free == 0`` on path B (rsize > te): the whole objective is F's A1,
    and the group scale reaches kernel B through ``dscale``."""
    f = Fitter(CFG_IDENT)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    rs = 17
    data = _in_grid(kc, 34, seed=5)
    ref = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=rs)
    m = list(range(10))
    v = _fv(kc, params, m)
    gpu = create_backend("cuda_flat_pwa_gen_s0f", kc, model="pwa_gen",
                         resolution_size=rs, fixed_cols=m, fixed_values=v)
    off = create_backend("cuda_flat_pwa_gen_s0f", kc, model="pwa_gen",
                         resolution_size=rs, fixed_cols=m, fixed_values=v,
                         split_fixed=False)
    try:
        assert gpu.kernel.n_free == 0 and gpu.kernel.n_fix == 4
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=1.3)
        Qo, go, Po = off.compute(params, off.load_data(data), norm=1.3)
        assert _floor_ok(Pg, Pr, CACHE_FLOOR_P)
        assert _floor_ok(Qg, Qr)
        assert _floor_ok(gg["ck"], gr["ck"])
        # the split and the in-tile path agree (dA's float2 store only)
        assert _floor_ok(Qg, Qo)
        assert _floor_ok(gg["ck"], go["ck"])
    finally:
        ref.free()
        gpu.free()
        off.free()


# ── the per-chunk F/T/B interleave ─────────────────────────────────────

@pytest.mark.parametrize("rs", [1, 2, 4, 17])
@pytest.mark.parametrize("bs", [1, 5, 7, 17, 1000, 50000])
def test_the_chunk_plan_never_splits_a_group(rs, bs):
    """Every chunk start / non-final length is a whole number of groups."""
    f = Fitter(CFG_KM)
    f.apply_constrains()
    k = create_backend("cuda_flat_pwa_gen_s0f", f.kernel_config,
                       model="pwa_gen", resolution_size=rs, batch_size=bs)
    try:
        for ne in (0, 1, 17, 1000, 9999, 50000, 123457):
            cap, n, aligned = k.kernel.chunk_plan(ne)
            assert cap > 0 and aligned == 1, (ne, cap, n)
            if rs > 1:
                assert cap % rs == 0, (bs, rs, cap)
            if ne > 0:
                assert n == (ne + cap - 1) // cap
    finally:
        k.free()


def test_f_and_b_run_once_per_chunk():
    """F(c) -> tile(c) -> B(c): the launch counts are one per chunk."""
    f = Fitter(CFG_IDENT)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    data = _in_grid(kc, 32, seed=4)
    mask = ([5, 6, 7, 8, 9])
    vals = _fv(kc, params, mask)
    one = create_backend("cuda_flat_pwa_gen_s0f", kc, model="pwa_gen",
                         fixed_cols=mask, fixed_values=vals, batch_size=10 ** 9)
    many = create_backend("cuda_flat_pwa_gen_s0f", kc, model="pwa_gen",
                          fixed_cols=mask, fixed_values=vals, batch_size=8)
    try:
        Q1, g1, P1 = one.compute(params, one.load_data(data), norm=1.3)
        assert one.kernel.split_counts() == (1, 1)
        Q2, g2, P2 = many.compute(params, many.load_data(data), norm=1.3)
        assert many.kernel.split_counts() == (4, 4)      # ceil(32/8)
        assert many.kernel.chunk_plan(32) == (8, 4, 1)
        # the interleave changes only the accumulation order across chunks
        assert _floor_ok(P2, P1, CACHE_FLOOR_P)
        assert _floor_ok(Q2, Q1)
        assert _floor_ok(g2["ck"], g1["ck"])
    finally:
        one.free()
        many.free()


# ── the all-fixed fused tile and the event-major B ─────────────────────

@pytest.mark.parametrize("rs", [1, 17, 40])
@pytest.mark.parametrize("norm", [None, 1.3])
def test_allfixed_fused_matches_the_F_T_B_path(rs, norm):
    """The fused branch vs the general F + tile + B path on the same inputs.

    With every wave fixed the tile builds A straight from Afix and back-
    propagates inside the same launch; Q/P/dck must agree at the fp32-cache
    floor, and F/B must NOT be launched (the counters prove it).
    """
    f = Fitter(CFG_IDENT)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    # 101 events: a SHORT final group at rsize 17 (5 groups + 16 events), so
    # the fused branch's group close is exercised against the reference too —
    # comparing fused against `fused=False` only would hide a shared bug
    data = _in_grid(kc, 101, seed=0)
    mask = list(range(10))          # all waves fixed
    vals = _fv(kc, params, mask)
    fused = create_backend("cuda_flat_pwa_gen_s0f", kc, model="pwa_gen",
                           fixed_cols=mask, fixed_values=vals, resolution_size=rs)
    old = create_backend("cuda_flat_pwa_gen_s0f", kc, model="pwa_gen",
                         fixed_cols=mask, fixed_values=vals, resolution_size=rs,
                         fused=False)
    ref = create_backend("numpy_flat", kc, model="pwa_gen",
                         resolution_size=rs)
    try:
        assert fused.kernel.n_free == 0
        Qf, gf, Pf = fused.compute(params, fused.load_data(data), norm=norm)
        Qo, go, Po = old.compute(params, old.load_data(data), norm=norm)
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=norm)
        assert _floor_ok(Pf, Pr, CACHE_FLOOR_P), "P vs numpy_flat"
        assert _floor_ok(Qf, Qr), "Q vs numpy_flat"
        assert _floor_ok(gf["ck"], gr["ck"]), "dck vs numpy_flat"
        assert fused.kernel.split_counts() == (0, 0), "F/B launched"
        assert fused.kernel.fused_count() >= 1
        assert _floor_ok(Pf, Po, CACHE_FLOOR_P)
        assert _floor_ok(Qf, Qo)
        assert _floor_ok(gf["ck"], go["ck"], CACHE_FLOOR_COEFF)
        if norm is not None:
            assert _floor_ok(gf["norm"], go["norm"])
        # the old path really did launch F and B
        assert old.kernel.split_counts()[0] >= 1
    finally:
        fused.free()
        old.free()
        ref.free()


def test_allfixed_fused_has_no_split_buffers():
    """`n_free == 0` allocates neither A1 nor dA (the fused tile needs none)."""
    f = Fitter(CFG_IDENT)
    f.apply_constrains()
    kc = f.kernel_config
    mask = list(range(10))
    k = create_backend("cuda_flat_pwa_gen_s0f", kc, model="pwa_gen",
                       fixed_cols=mask, fixed_values=_fv(kc, _params(f), mask))
    try:
        assert k.kernel.aux_bytes(1000) == 0
        assert k.kernel.fix_bytes(1000) == 8 * k.kernel.n_proj * 4 * 1000
    finally:
        k.free()


@pytest.mark.parametrize("rs", [17, 40])
def test_fused_falls_back_when_a_group_cannot_fit_one_block(rs):
    """`rsize > te` cannot hold a group's A/dA in one tile: the call runs the
    general F/T/B path (and still matches the reference)."""
    f = Fitter(CFG_IDENT)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    data = _in_grid(kc, 32, seed=2)
    mask = list(range(10))
    k = create_backend("cuda_flat_pwa_gen_s0f", kc, model="pwa_gen",
                       fixed_cols=mask, fixed_values=_fv(kc, params, mask),
                       resolution_size=rs, te=rs - 1)
    ref = create_backend("numpy_flat", kc, model="pwa_gen",
                         resolution_size=rs)
    try:
        assert k.kernel.te < rs
        Q, g, P = k.compute(params, k.load_data(data), norm=1.3)
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        assert k.kernel.fused_count() == 0
        assert k.kernel.split_counts()[0] >= 1      # F ran
        # the fallback really owns the split scratch (the allocation and the
        # report agree), and the fused path is the one that reports 0
        kk = k.kernel
        assert kk.aux_bytes(64) == kk.cap * kk.n_proj * (16 + 8) + kk.cap * 8
        assert kk.aux_bytes(64) > 0
        assert _floor_ok(P, Pr, CACHE_FLOOR_P)
        assert _floor_ok(Q, Qr)
        assert _floor_ok(g["ck"], gr["ck"])
    finally:
        k.free()
        ref.free()


def test_the_event_major_B_matches_the_wave_major_mapping():
    """Change 2: the new event-major `gps0f_bwd_fixed` vs the old mapping.

    Same mask (mixed), same data/params; only kernel B's traversal changes, so
    dck must agree at the fp32-cache floor.
    """
    f = Fitter(CFG_IDENT)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    data = _in_grid(kc, 64, seed=3)
    mask = ([5, 6, 7, 8, 9])
    vals = _fv(kc, params, mask)
    ev = create_backend("cuda_flat_pwa_gen_s0f", kc, model="pwa_gen",
                        fixed_cols=mask, fixed_values=vals, bwd_mode="event")
    wv = create_backend("cuda_flat_pwa_gen_s0f", kc, model="pwa_gen",
                        fixed_cols=mask, fixed_values=vals, bwd_mode="wave")
    try:
        assert ev.kernel.n_free > 0
        Qe, ge, Pe = ev.compute(params, ev.load_data(data), norm=1.3)
        Qw, gw, Pw = wv.compute(params, wv.load_data(data), norm=1.3)
        assert ev.kernel.split_counts()[1] >= 1
        assert wv.kernel.split_counts()[1] >= 1
        assert _floor_ok(Pe, Pw, CACHE_FLOOR_P)
        assert _floor_ok(Qe, Qw)
        assert _floor_ok(ge["ck"], gw["ck"], CACHE_FLOOR_COEFF)
        assert _floor_ok(ge["norm"], gw["norm"])
    finally:
        ev.free()
        wv.free()


def test_a_repeated_call_is_the_same_call():
    """A second compute with the same coeff reuses the handle state; the fill is
    not run again with a different meaning (the guard would have raised)."""
    f, kc, ref, gpu = _masked_pair(CFG_IDENT, [5, 6, 7, 8, 9])
    params = _params(f)
    data = _in_grid(kc, 40, seed=6)
    h = gpu.load_data(data)
    try:
        Q1, g1, P1 = gpu.compute(params, h, norm=1.3)
        Q2, g2, P2 = gpu.compute(params, h, norm=1.3)
        assert _rel(Q1, Q2) <= S0F_TOL
        assert np.array_equal(P1, P2)
        assert _rel(g1["ck"], g2["ck"]) <= S0F_TOL
        assert _rel(g1["coeff"], g2["coeff"]) <= S0F_TOL_COEFF
    finally:
        h.free()
        gpu.free()
        ref.free()


# ── the CTPWA mask arithmetic ──────────────────────────────────────────

#: the measured 27/7 + 67/88 arithmetic and the 14 moving columns are the
#: SHARED gate (``tests/test_flat_masked.py`` — one copy for every masked
#: backend).  What stays here is the per-PARAMETER truth of the alias, which
#: the union over the ten parameters cannot state.

@pytest.mark.skipif(not __import__("os").path.exists(CTPWA_CFG),
                    reason="the CTPWA benchmark config is not present")
@pytest.mark.parametrize("free,cols", [
    # the alias: the K+ mass drives BOTH charge sectors' s0 and width_t
    ("K3st1780p_mass", [35, 36, 53, 54]),
    ("K3st1780p_width", [36, 54]),
])
def test_an_aliased_parameter_moves_both_charge_sectors(free, cols):
    """The regression this whole mask discussion exists for.

    With only ``K3st1780p_mass`` free, the moving columns include the
    ``K3st1780m_*`` ones (53, 54) — their names do not share the prefix, so a
    name mask that fixes ``K3st1780p_s0`` alone would leave a moving column
    free while the cache believes it is frozen.
    """
    f, x = _ctpwa_fitter(free=(free,))
    den = [str(n) for n in np.asarray(f.kernel_config["den_names"]).ravel()]
    per = _moving_by_param(f, x)
    assert per[free] == cols, (free, per[free])
    assert [den[i] for i in per[free]] == [den[i] for i in cols]
    if free == "K3st1780p_mass":
        # exactly the trap: the p column is NOT enough
        assert 53 in per[free] and 54 in per[free]


@pytest.mark.skipif(not __import__("os").path.exists(CTPWA_CFG),
                    reason="the CTPWA benchmark config is not present")
def _wide_bug_kc(n_wave=80, n_basis=81, n_proj=2):
    """A shape whose tile shrinks below a whole group under the smem budget."""
    from benchmark_flat_vs_pwa import tile_flat
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = dict(tile_flat(f.kernel_config, n_wave, n_proj))
    nw = int(np.asarray(kc["fl_table"]).shape[0])
    nb = int(np.asarray(kc["basis_table"]).shape[1])
    kc["basis_table"] = np.ones((n_basis, nb), np.complex128)
    kc["basis_m_index"] = np.zeros(n_basis, np.int64)
    gam = np.zeros((nw, 1, n_basis))
    for k in range(nw):
        gam[k, 0, k % n_basis] = 1.0
    kc["gamma_matrix"] = gam
    kc["den_names"] = np.asarray([f"b{i}" for i in range(n_basis)])
    return kc


@pytest.mark.parametrize("rs", [2, 4])
def test_pick_te_never_returns_a_tile_create_rejects(rs):
    """The budget shrink must land on a whole group, not on `rsize - 1`.

    Reproduced on the pre-fix copy: ``n_wave 80, n_basis 81, n_proj 2`` at
    ``rsize 4`` shrank 8 -> 7 and ``create`` raised "context creation failed".
    The tile must construct AND compute the grouped objective correctly.
    """
    kc = _wide_bug_kc()
    m = ([0])
    rng = np.random.default_rng(3)
    params = {"ck": rng.normal(size=80) + 1j * rng.normal(size=80),
              "coeff": rng.uniform(0.3, 1.2, 81)}
    gpu = create_backend("cuda_flat_pwa_gen_s0f", kc, model="pwa_gen",
                         resolution_size=rs, fixed_cols=m,
                         fixed_values=_fv(kc, params, m))
    ref = create_backend("numpy_flat", kc, model="pwa_gen",
                         resolution_size=rs)
    try:
        k = gpu.kernel
        assert (k.n_wave, k.n_basis, k.n_proj) == (80, 81, 2)
        assert k.te >= 1
        if rs > 1 and k.te >= rs:
            assert k.te % rs == 0, (rs, k.te)
        data = _in_grid(kc, 64, seed=2)
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=1.3)
        assert _floor_ok(Pg, Pr, CACHE_FLOOR_P)
        assert _floor_ok(Qg, Qr)
        assert _floor_ok(gg["ck"], gr["ck"])
    finally:
        gpu.free()
        ref.free()


# ── gate 9: the guards and the contract edges ──────────────────────────

def test_multiple_bws_per_chain_is_rejected_with_the_general_backend_named():
    f = Fitter(CFG_MB)
    f.apply_constrains()
    with pytest.raises(ValueError, match="exactly ONE BW per chain"):
        CUDAKernelFlatPwaGenS0F(f.kernel_config)
    with pytest.raises(ValueError, match="cuda_flat_pwa_gen2"):
        create_backend("cuda_flat_pwa_gen_s0f", f.kernel_config,
                       model="pwa_gen")


#: the empty sample, ``return_p`` and the cache-size formula are the SHARED
#: gates (``tests/test_flat_masked.py``); the launch counts above are this
#: kernel's own.
