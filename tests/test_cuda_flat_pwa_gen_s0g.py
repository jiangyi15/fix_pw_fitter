"""``cuda_flat_pwa_gen_s0g`` — s0f's fixed-wave cache with the free block
un-staged, in ONE kernel per chunk.

Case-1 copy (kernel policy §1): own module, own ``.so``, own symbols, and
``kernels_flat_pwa_gen_s0f.cu`` byte-identical (pinned by hash below).

Gates (this kernel's own; the MASK contract every masked backend shares is
``tests/test_flat_masked.py``, selected by the ``fixed_coeff`` tag):

1. the copy is real (own module/lib/symbols, no cross-import of s0f, s0f's
   source unchanged) and its build-time helpers are verbatim copies;
2. mask OFF equals ``cuda_flat_pwa_gen_s0e2`` (the parent's path);
3. mask ON equals ``cuda_flat_pwa_gen_s0f`` — the case-1 cross-check, over
   ``rsize {1, 17, 40}`` x ``norm {None, 1.3}``, for a mixed mask, an
   all-fixed mask and the CP/identical-block fixtures (the ``numpy_flat``
   comparison itself is the shared gate);
4. FD on ``ck`` and on the FREE ``coeff`` columns, for no / mixed / all-fixed;
5. the launch plan: ONE tile launch per chunk, no F/B at all, a block never
   splits a group, an empty sample launches nothing, an oversized group is
   refused naming s0f, and ``te``/``tile_bytes`` are the whole-group budget;
6. the guard/edge set: multi-BW refused naming gen2, idempotent ``free()``,
   a repeated call is the same call.
"""
import hashlib
import os

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import backends_for_model, backend_tags, create_backend
from tabpwa.cuda._flat_pwa_gen_s0g import (
    CUDAKernelFlatPwaGenS0G, amp_slot_map, index_runs, split_waves,
    uslot_map, wave_plan)
from test_cuda_flat_cache_pwa_gen import _in_grid

CFG_ONE = "tests/config_pwa_gen.yml"
CFG_KM = "tests/config_pwa_gen_km.yml"
CFG_MB = "tests/config_pwa_gen_mb.yml"
CFG_IDENT = "tests/config_pwa_gen_ident.yml"
CFG_CP = "tests/config_pwa_gen_cp.yml"
KERNEL = "cuda_flat_pwa_gen_s0g"
QUALIFYING = (CFG_ONE, CFG_KM, CFG_IDENT, CFG_CP)

#: the fp32-cache floors the flat family is gated at (same numbers as the
#: sibling fixtures: the caches are float2, so ~1e-7 typical / 1e-5 worst)
CACHE_FLOOR = 2e-5
CACHE_FLOOR_P = 3e-5
CACHE_FLOOR_COEFF = 1e-4
CACHE_FLOOR_ABS = 1e-4

#: s0g vs s0f: same arithmetic, different summation order (A is accumulated
#: fixed-then-free per element, s0f sums F in a separate kernel).  Measured
#: <= 2.2e-15 on every quantity, so 1e-13 is the floor with margin — six
#: orders below the fp32 cache floor, i.e. the two are the same result.
S0G_TOL = 1e-13

NH = 1e-6

#: `kernels_flat_pwa_gen_s0f.cu` as it stands.  Any edit to s0f must fail HERE
#: first: s0g is a copy, never a reason to touch the parent.  This moved once,
#: deliberately: the unified mask API (values required, the fill at `load_data`,
#: no lazy/snapshot branch) is a change to the PARENT kernel itself.
S0F_CU_SHA256 = (
    "a108ba867b1ce1375066812c69b3d613293bdd4020dade35545650146d4e7744")


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


def _as_mask(kc, cols):
    n_basis = int(np.asarray(kc["basis_table"]).shape[0])
    m = np.zeros(n_basis, np.int8)
    if cols is not None:
        m[np.asarray(list(cols), np.int64)] = 1
    return m


def _params(f):
    return f.build_params(f.initial_values(seed=1))[0]


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


def _pair(cfg, cols=None, rs=1, **kw):
    """``(fitter, kc, numpy_flat, s0g)`` on one config/mask."""
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    ref = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=rs)
    extra = ({} if cols is None
             else {"fixed_cols": list(cols),
                   "fixed_values": _fv(kc, _params(f), cols)})
    gpu = create_backend(KERNEL, kc, model="pwa_gen", resolution_size=rs,
                         **{**extra, **kw})
    return f, kc, ref, gpu


# ── gate 1: a real case-1 copy ─────────────────────────────────────────

def test_registration():
    allowed = set(backends_for_model("pwa_gen"))
    assert KERNEL in allowed
    assert KERNEL not in set(backends_for_model("pwa"))
    assert "n_res1" in backend_tags(KERNEL, "pwa_gen")


def test_case1_own_module_library_and_symbols():
    """Own module, own ``.so``, own symbols — and no import of s0f."""
    import tabpwa.cuda._flat_pwa_gen_s0f as parent
    import tabpwa.cuda._flat_pwa_gen_s0g as new
    assert new is not parent
    import ast
    src = open(new.__file__.replace(".pyc", ".py")).read()
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(f"{node.module}.{a.name}" for a in node.names)
    assert not any("_flat_pwa_gen_s0f" in m for m in imported), (
        f"a copy must not import the kernel it copies: {sorted(imported)}")
    f = Fitter(CFG_KM)
    f.apply_constrains()
    a = CUDAKernelFlatPwaGenS0G(f.kernel_config)
    b = parent.CUDAKernelFlatPwaGenS0F(f.kernel_config)
    try:
        assert a._lib is not b._lib
        assert hasattr(a._lib, "cuda_compute_flat_pwa_gen_s0g")
        assert not hasattr(a._lib, "cuda_compute_flat_pwa_gen_s0f")
        assert hasattr(b._lib, "cuda_compute_flat_pwa_gen_s0f")
        assert not hasattr(b._lib, "cuda_compute_flat_pwa_gen_s0g")
        # the launch-plan reports of s0f's split do not exist here at all
        assert not hasattr(a, "split_counts")
        assert not hasattr(a, "aux_bytes")
        assert not hasattr(a, "fused_count")
        assert hasattr(a, "tile_count")
    finally:
        a.free()
        b.free()


def test_the_parent_kernel_source_is_unchanged():
    """Case 1 keeps ``kernels_flat_pwa_gen_s0f.cu`` byte-identical."""
    import tabpwa.cuda as pkg
    path = os.path.join(os.path.dirname(pkg.__file__),
                        "kernels_flat_pwa_gen_s0f.cu")
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert digest == S0F_CU_SHA256, (
        "kernels_flat_pwa_gen_s0f.cu changed — s0g is a copy, so a change "
        "here is a change to the parent, not to this kernel")


def test_the_build_time_helpers_are_verbatim_copies():
    """The helpers travel with the copy; a drift is a silent mask bug."""
    import tabpwa.cuda._flat_pwa_gen_s0f as parent
    for cfg in QUALIFYING:
        f = Fitter(cfg)
        f.apply_constrains()
        kc = f.kernel_config
        n_basis = int(np.asarray(kc["basis_table"]).shape[0])
        for x, y in zip(uslot_map(kc["basis_table"], kc["basis_m_index"]),
                        parent.uslot_map(kc["basis_table"],
                                         kc["basis_m_index"])):
            assert np.array_equal(x, y)
        a = index_runs(kc["gamma_matrix"], n_basis)
        b = parent.index_runs(kc["gamma_matrix"], n_basis)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)
        mask = np.zeros(n_basis, np.int8)
        mask[: max(1, n_basis // 3)] = 1
        pa = wave_plan(mask, a[0], a[1], a[3])
        pb = parent.wave_plan(mask, b[0], b[1], b[3])
        for x, y in zip(pa, pb):
            assert np.array_equal(x, y)
        # nz_u is the uslot of each column (the plan's own argument)
        nz_u = np.zeros(n_basis, np.int32)[np.asarray(a[3], np.int64)]
        nz_pair = (np.ascontiguousarray(nz_u).astype(np.int64) << 32) \
            | np.asarray(a[3], np.int64)
        xa = split_waves(a[0], a[1], a[2], a[3], nz_u,
                         np.ascontiguousarray(nz_pair), pa[2])
        xb = parent.split_waves(b[0], b[1], b[2], b[3], nz_u,
                                np.ascontiguousarray(nz_pair), pb[2])
        for key in xa:
            assert np.array_equal(xa[key], xb[key]), key


def test_amp_slot_map_is_a_verbatim_copy():
    import tabpwa.cuda._flat_pwa_gen_s0f as parent
    for cfg in QUALIFYING:
        f = Fitter(cfg)
        f.apply_constrains()
        kc = f.kernel_config
        nw = int(np.asarray(kc["fl_table"]).shape[0])
        P = max(1, int(np.asarray(kc["matrix_angle"]).shape[1]) // nw)
        args = (kc["matrix_angle"], kc["fl_table"], kc["fl_q_index"],
                kc["num_table"], kc["num_m_index"], nw, P)
        for x, y in zip(amp_slot_map(*args), parent.amp_slot_map(*args)):
            assert np.array_equal(x, y)


# ── gate 2: mask off is the parent ─────────────────────────────────────

@pytest.mark.parametrize("cfg", [CFG_ONE, CFG_KM])
@pytest.mark.parametrize("rs", [1, 17, 40])
@pytest.mark.parametrize("norm", [None, 1.3])
def test_mask_off_matches_s0e2(cfg, rs, norm):
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    base = _in_grid(kc, 64, seed=2)
    data = _replicate(base, rs) if rs > 1 else base
    a = create_backend("cuda_flat_pwa_gen_s0e2", kc, model="pwa_gen",
                       resolution_size=rs)
    b = create_backend(KERNEL, kc, model="pwa_gen", resolution_size=rs)
    ha, hb = a.load_data(data), b.load_data(data)
    try:
        assert b.kernel.n_fix == 0
        Qa, ga, Pa = a.compute(params, ha, norm=norm)
        Qb, gb, Pb = b.compute(params, hb, norm=norm)
        assert b.kernel.tile_count() == 1
        assert np.array_equal(Pa, Pb), "P must be bit-identical"
        for name, y, x in (("Q", Qb, Qa), ("dck", gb["ck"], ga["ck"]),
                           ("dcoeff", gb["coeff"], ga["coeff"])):
            assert _rel(y, x) <= S0G_TOL, f"{name}: {_rel(y, x):.2e}"
        if norm is not None:
            assert _rel(gb["norm"], ga["norm"]) <= S0G_TOL
    finally:
        ha.free()
        hb.free()
        a.free()
        b.free()


# ── gate 3: mask on equals s0f (and the reference) ─────────────────────

@pytest.mark.parametrize("rs", [1, 17, 40])
@pytest.mark.parametrize("norm", [None, 1.3])
@pytest.mark.parametrize("cfg,mask", [
    (CFG_IDENT, [5, 6, 7, 8, 9]),      # mixed: 2 fixed / 2 free
    (CFG_KM, list(range(5))),          # all fixed
    (CFG_ONE, list(range(3))),         # all fixed, one wave
    (CFG_CP, [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]),
])
def test_mask_on_matches_s0f_and_the_reference(cfg, mask, rs, norm):
    f, kc, ref, gpu = _pair(cfg, mask, rs=rs)
    params = _params(f)
    s0f = create_backend("cuda_flat_pwa_gen_s0f", kc, model="pwa_gen",
                         resolution_size=rs, fixed_cols=mask,
                         fixed_values=_fv(kc, params, mask))
    params = _params(f)
    base = _in_grid(kc, 64, seed=0)
    data = _replicate(base, rs) if rs > 1 else base
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=norm)
        Qf, gf, Pf = s0f.compute(params, s0f.load_data(data), norm=norm)
        Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=norm)
        assert gpu.kernel.n_fix > 0, "this case must actually cache something"
        # the fp32-cache floor against the independent reference
        assert _floor_ok(Pg, Pr, CACHE_FLOOR_P), "P vs numpy_flat"
        assert _floor_ok(Qg, Qr), "Q vs numpy_flat"
        assert _floor_ok(gg["ck"], gr["ck"]), "dck vs numpy_flat"
        free_cols = np.setdiff1d(np.arange(gpu.kernel.n_basis),
                                 gpu.kernel.watched_cols)
        if free_cols.size:
            assert _floor_ok(gg["coeff"][free_cols], gr["coeff"][free_cols],
                             CACHE_FLOOR_COEFF), "dcoeff vs numpy_flat"
        # and the tight gate against s0f: same arithmetic, another order
        assert _rel(Pg, Pf) <= S0G_TOL, "P vs s0f"
        assert _rel(Qg, Qf) <= S0G_TOL, "Q vs s0f"
        # s0f stages the finished `dA` through a float2 buffer for kernel B,
        # so a FIXED wave's dck carries one fp32 rounding that s0g (fp64
        # shared dA) does not — measured 1.8e-8, i.e. s0f's cache floor, and
        # s0g is the one closer to `numpy_flat`.  The FREE waves are fp64 in
        # both, so they keep the tight bound.
        fw = np.asarray(gpu.kernel.fixed_waves, np.int64)
        free_w = np.setdiff1d(np.arange(gpu.kernel.n_wave), fw)
        if fw.size:
            assert _rel(gg["ck"][fw], gf["ck"][fw]) <= 1e-6, "dck fixed vs s0f"
        if free_w.size:
            assert _rel(gg["ck"][free_w], gf["ck"][free_w]) <= S0G_TOL, (
                "dck free vs s0f")
        assert _rel(gg["coeff"], gf["coeff"]) <= S0G_TOL, "dcoeff vs s0f"
        if norm is not None:
            assert _rel(gg["norm"], gf["norm"]) <= S0G_TOL, "dnorm vs s0f"
    finally:
        ref.free()
        s0f.free()
        gpu.free()


# ── gate 4: finite differences ─────────────────────────────────────────

def test_coeff_gradient_fd_on_the_free_columns():
    f, kc, ref, gpu = _pair(CFG_IDENT, [5, 6, 7, 8, 9])
    params = _params(f)
    h = gpu.load_data(_in_grid(kc, 48, seed=3))
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
        h.free()
        gpu.free()
        ref.free()


@pytest.mark.parametrize("mask", [None, [5, 6, 7, 8, 9], list(range(10))])
def test_ck_gradient_fd_through_the_masked_path(mask):
    """``dck`` against central differences: no mask, mixed, all-fixed."""
    f, kc, ref, gpu = _pair(CFG_IDENT, mask)
    params = _params(f)
    h = gpu.load_data(_in_grid(kc, 48, seed=4))
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
        h.free()
        gpu.free()
        ref.free()


# ── the launch plan ────────────────────────────────────────────────────

@pytest.mark.parametrize("rs", [1, 17, 40])
@pytest.mark.parametrize("bs", [1, 7, 40, 1000, 50000])
def test_the_chunk_plan_never_splits_a_group(rs, bs):
    f = Fitter(CFG_IDENT)
    f.apply_constrains()
    k = CUDAKernelFlatPwaGenS0G(f.kernel_config, resolution_size=rs,
                                batch_size=bs)
    try:
        cap, n = k.chunk_plan(0)[0], None
        for ne in (0, 1, 17, 1000, 9999, 50000, 123457):
            cap, nch, aligned = k.chunk_plan(ne)
            assert cap > 0 and aligned == 1, (ne, cap, nch)
            if rs > 1:
                assert cap % rs == 0, (bs, rs, cap)
            if ne > 0:
                assert nch == (ne + cap - 1) // cap
        # and the block's own te is a whole number of groups
        assert k.te >= 1
        if rs > 1:
            assert k.te >= rs and k.te % rs == 0, (rs, k.te)
            assert k.groups_per_block() == k.te // rs
    finally:
        k.free()


def test_exactly_one_tile_launch_per_chunk():
    """There is no F and no B: one kernel per `cap` chunk, nothing else."""
    f = Fitter(CFG_IDENT)
    f.apply_constrains()
    kc = f.kernel_config
    params = _params(f)
    data = _in_grid(kc, 32, seed=4)
    cols = [5, 6, 7, 8, 9]
    vals = _fv(kc, params, cols)
    one = create_backend(KERNEL, kc, model="pwa_gen",
                         fixed_cols=cols, fixed_values=vals,
                         batch_size=10 ** 9)
    many = create_backend(KERNEL, kc, model="pwa_gen",
                          fixed_cols=cols, fixed_values=vals, batch_size=8)
    try:
        Q1, g1, P1 = one.compute(params, one.load_data(data), norm=1.3)
        assert one.kernel.tile_count() == 1
        Q2, g2, P2 = many.compute(params, many.load_data(data), norm=1.3)
        assert many.kernel.tile_count() == 4        # ceil(32/8)
        assert many.kernel.chunk_plan(32) == (8, 4, 1)
        # an EMPTY sample is the identity and launches nothing at all
        h0 = one.load_data({k: np.asarray(v)[:0] for k, v in data.items()})
        try:
            assert one.kernel.afix_ready(h0)
            one.compute(params, h0, norm=1.3)
            assert one.kernel.tile_count() == 1, "an empty sample launched a tile"
        finally:
            h0.free()
        assert _floor_ok(P2, P1, CACHE_FLOOR_P)
        assert _floor_ok(Q2, Q1)
        assert _floor_ok(g2["ck"], g1["ck"])
        # the chunked call is the same call, at the s0g/s0f reassociation floor
        s0f = create_backend("cuda_flat_pwa_gen_s0f", kc, model="pwa_gen",
                             fixed_cols=cols, fixed_values=vals,
                             batch_size=10 ** 9)
        Qf, gf, _ = s0f.compute(params, s0f.load_data(data), norm=1.3)
        assert _rel(Q2, Qf) <= S0G_TOL
        # s0f's fixed-block dck goes through a float2 `dA` buffer (see the
        # mask-on gate above); the free waves are fp64 in both
        fw = np.asarray(many.kernel.fixed_waves, np.int64)
        free_w = np.setdiff1d(np.arange(many.kernel.n_wave), fw)
        assert _rel(g2["ck"][fw], gf["ck"][fw]) <= 1e-6
        assert _rel(g2["ck"][free_w], gf["ck"][free_w]) <= S0G_TOL
        s0f.free()
    finally:
        one.free()
        many.free()


def test_a_group_too_large_for_one_block_is_refused_naming_s0f():
    """s0g closes a group inside a block; a group that cannot fit refuses."""
    f = Fitter(CFG_IDENT)
    f.apply_constrains()
    kc = f.kernel_config
    # 1e6-event groups can never fit a block (the tile is capped at 256 events)
    with pytest.raises(ValueError, match="cuda_flat_pwa_gen_s0f"):
        CUDAKernelFlatPwaGenS0G(kc, resolution_size=10 ** 6)
    with pytest.raises(ValueError, match="does not fit one block"):
        create_backend(KERNEL, kc, model="pwa_gen", resolution_size=10 ** 6)


def test_the_tile_size_shrinks_with_the_free_block():
    """`te` is chosen from the shared budget, never from `ne`."""
    f = Fitter(CFG_IDENT)
    f.apply_constrains()
    kc = f.kernel_config
    n_basis = int(np.asarray(kc["basis_table"]).shape[0])
    big = CUDAKernelFlatPwaGenS0G(kc)
    allfix = CUDAKernelFlatPwaGenS0G(
        kc, fixed_cols=list(range(n_basis)),
        fixed_values=_fv(kc, _params(f), list(range(n_basis))))
    try:
        assert big.te <= 256 and big.te > 32
        assert allfix.te >= big.te          # no free block -> more room
        assert allfix.te <= 256
        assert big.tile_bytes() == big.tile_bytes(te=big.te)
        assert big.tile_bytes(te=32) < big.tile_bytes()
        assert big.smem_bytes == big.tile_bytes()
        assert big.smem_bytes <= 48 * 1024
        # `rsize = 2` is the widest multi-group case: a block holds >= 100
        # groups (128 at te 256), which is what `te` must be a multiple of
        two = CUDAKernelFlatPwaGenS0G(kc, resolution_size=2)
        try:
            assert two.groups_per_block() >= 100, two.te
            assert two.groups_per_block() == two.te // 2
        finally:
            two.free()
    finally:
        big.free()
        allfix.free()


# ── gate 7: guards and edges ───────────────────────────────────────────

def test_multiple_bws_per_chain_is_rejected_with_the_general_backend_named():
    f = Fitter(CFG_MB)
    f.apply_constrains()
    with pytest.raises(ValueError, match="exactly ONE BW per chain"):
        CUDAKernelFlatPwaGenS0G(f.kernel_config)
    with pytest.raises(ValueError, match="cuda_flat_pwa_gen2"):
        create_backend(KERNEL, f.kernel_config, model="pwa_gen")


def test_free_is_idempotent():
    f = Fitter(CFG_IDENT)
    f.apply_constrains()
    kc = f.kernel_config
    cols = [5, 6, 7, 8, 9]
    k = CUDAKernelFlatPwaGenS0G(kc, fixed_cols=cols,
                                fixed_values=_fv(kc, _params(f), cols))
    h = k.load_data(_in_grid(kc, 8, seed=1))
    k.compute(_params(f), h, norm=1.3)
    h.free()
    h.free()                 # idempotent handle
    assert h.ptr is None
    k.free()
    k.free()                 # idempotent context
    assert k._ctx is None


def test_a_repeated_call_is_the_same_call():
    f, kc, ref, gpu = _pair(CFG_IDENT, [5, 6, 7, 8, 9])
    params = _params(f)
    h = gpu.load_data(_in_grid(kc, 64, seed=7))
    try:
        first = gpu.compute(params, h, norm=1.3)
        for _ in range(4):
            again = gpu.compute(params, h, norm=1.3)
            # the block reductions are `atomicAdd`-accumulated (as the
            # parent's are), so a repeat is the same call up to the last ulp
            assert _rel(again[0], first[0]) <= S0G_TOL
            assert _rel(again[1]["ck"], first[1]["ck"]) <= S0G_TOL
            assert _rel(again[1]["coeff"], first[1]["coeff"]) <= S0G_TOL
    finally:
        h.free()
        gpu.free()
        ref.free()


#: the CTPWA mask arithmetic (14 moving columns, 27 fixed / 7 free,
#: 67/88 non-zeros) is the SHARED gate
#: (``tests/test_flat_masked.py``) — one copy for every masked
#: backend, parametrised by the ``fixed_coeff`` tag.
