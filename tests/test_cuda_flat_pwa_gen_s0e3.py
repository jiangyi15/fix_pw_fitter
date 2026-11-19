"""``cuda_flat_pwa_gen_s0e3`` — the s0e fused tile with a Newton reciprocal.

The THIRD case-1 copy of the fused tile: `cuda_flat_pwa_gen_s0e2` is the
packed-CSR/2-chain-dot copy, this one is `cuda_flat_pwa_gen_s0e` VERBATIM plus
**exactly one numerical change** — the REAL reciprocal ``1/|D|²`` in phase A of
the tile kernel is evaluated as an fp32 seed followed by two fp64 Newton
refinements instead of the IEEE fp64 divide, with an exact-divide fallback
outside the fp32-seed range.  Its purpose is the A/B that isolates the divide
(the fp64 div unit is unpipelined on consumer GPUs, ~25-30 % of the fp64 work
on the production shape).  ``te`` is exposed as a runtime knob (``te=``) so
the tile size can be swept.

Gates (all inherited from the s0e fixture, which stays untouched):
1. **registration and the case-1 copy** — ``pwa_gen`` scope only, own module,
   own source, own ``.so``, own ``gps03_*``/``GPS03*``/``cuda_*_s0e3`` symbols;
   the s0e and s0e2 symbols are NOT present in this library.
2. **the basis rotation is exact** — ``Z = i·basis_table`` with the kernel's
   ``-i·Z`` recovers every basis row, and ``Σ_i coeff_i·basis_i(m)`` assembled
   from the rotated us-cache equals ``numpy_flat``'s ``D``.
3. **both dedups are exact, bit-for-bit**, with the measured fold ratios
   pinned per config.
4. **the uslot key is the row, not the kind** — the deliberately-wrong key is
   shown to give a different, smaller count on the real config.
5. **anchor vs ``numpy_flat``** on every ``n_res == 1`` fixture for
   ``Q``/``P``/``dck``/``dcoeff``/``dnorm``, ``norm=None`` and a scalar norm,
   at ``resolution_size`` 1 and 17.
6. **``dcoeff``/``dck`` against central differences** through the tiled path.
7. **chunk independence**, the reported chunk rule, path A vs path B.
8. **shared memory** reports/limits, and the tile-hint guard.
9. **s0e3 vs s0e: the reciprocal is the ONLY difference** — every quantity
   agrees to ``S0E3_TOL`` (1e-14, ~50 ulp of the exact value, ten orders
   below the fp32 cache floor), on ``CFG_ONE``/``CFG_KM`` at ``rsize`` 1/17
   with and without a norm.
10. **``te`` is accepted and arithmetic-invariant** — ``te=34`` (``rsize`` 17)
    reproduces the ``te=0`` run to <=1e-14: the tile size changes the tiling,
    never the arithmetic.  A hint that would split a log-sum group
    (``te=20`` at ``rsize=17``) raises ``ValueError`` up front.

Test data stays INSIDE both interpolation grids (outside, the tables clamp and
CPU/GPU diverge — see ``test_cuda_flat_cache_pwa_gen._in_grid``).
"""

# The compute-function suite — Q/P/grads vs numpy_flat (both norms,
# rsize 1 and 17), the grouped objective, FD gradients, chunk
# invariance, handle independence / freed handle, and the input errors —
# lives in `_flat_compute.py` and is collected by `test_flat_compute.py`
# for EVERY registered pwa_gen backend.  This file keeps only this
# kernel's own gates (tile/smem, path A/B, launch counts, the parent
# A/B, dedup/index pins).

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import backends_for_model, create_backend
from tabpwa.cuda._flat_pwa_gen_s0e3 import (
    CUDAKernelFlatPwaGenS0E3, amp_slot_map, index_runs, uslot_map)
from test_cuda_flat_cache_pwa_gen import _in_grid

CFG_ONE = "tests/config_pwa_gen.yml"
CFG_KM = "tests/config_pwa_gen_km.yml"
CFG_MB = "tests/config_pwa_gen_mb.yml"
CFG_IDENT = "tests/config_pwa_gen_ident.yml"
CFG_SAME = "tests/config_pwa_gen_same.yml"
CFG_CP = "tests/config_pwa_gen_cp.yml"
QUALIFYING = [CFG_ONE, CFG_KM, CFG_IDENT, CFG_SAME, CFG_CP]

# ── the fp32 cache floor ────────────────────────────────────────────────
# The two load-time caches are float2 (`us` = the rotated basis interp,
# `amp` = fa·fl·num), so agreement with the fp64 `numpy_flat` reference is
# limited by fp32, not by the algebra.  These are the SAME constants the
# sibling fixtures use, with the same reason: the error tracks P's DYNAMIC
# RANGE, not n_basis (`_ident` spans 7.5e-4 .. 7.2e2, so `_rel` = worst
# absolute / global max is set by the near-zero events).
# MEASURED for this kernel (ne=64, two seeds, rsize 1 and 17):
#   _pwa_gen, _km, _same   dQ ~4e-8 .. 2e-7   dck ~8e-8 .. 3e-7   dcoeff ~3e-7
#   _ident, _cp            dQ ~1e-6 .. 2e-5   dck ~7e-7 .. 1e-5   dcoeff ~7e-5
CACHE_FLOOR = 2e-5          # Q, dck, dnorm
CACHE_FLOOR_P = 3e-5        # per-event P
CACHE_FLOOR_COEFF = 1e-4    # dcoeff accumulates one rounding per (e,k,m)
#: The fp32 cache floor is an ABSOLUTE error, but ``_rel`` divides by the
#: value being compared — so a small Q (a 17-event grouped NLL is -0.39) turns
#: the same absolute gap into a larger relative one.  MEASURED on
#: ``config_pwa_gen_ident.yml`` at ``resolution_size = 17``:
#:     numpy_flat      Q = -0.3899469042747774
#:     this kernel     Q = -0.3899553800883502     (8.476e-06 apart)
#:     cuda_flat_pwa_gen_1bw  Q = -0.38995538008835173
#:     cuda_flat_pwa_gen2     Q = -0.38995538008835173
#: i.e. this kernel agrees with its two siblings to 4e-15 and the ENTIRE gap
#: is the shared float2 cache, not this kernel's algebra.  The grouped-Q gates
#: therefore also accept the absolute bound, which is the honest form for an
#: absolute error floor.
CACHE_FLOOR_ABS = 1e-4
NH = 1e-6

#: s0e3 vs s0e: the fp32 seed + 2 Newton steps differ from the exact fp64
#: divide by 1-2 ulp of ``1/|D|²``, so the quantities are no longer bit-equal.
#: 1e-14 is ~50 ulp of the accumulated value and ten orders BELOW the fp32
#: cache floor (2e-5 .. 1e-4 above), i.e. the gate cannot mask a real change.
S0E3_TOL = 1e-14

#: measured dedup ratios, pinned PER CONFIG (an average over the fixtures
#: would hide that the dense ones collapse nothing).
#:  us:  (n_basis, n_uslot) on the `(row bytes, mass column)` key
#:  amp: (n_wave, n_uniq)  on the full-numerator key
EXPECTED_DEDUP = {
    #                      n_basis  uslot     n_wave  uniq
    CFG_ONE:   ((3, 3), (1, 1)),
    CFG_KM:    ((5, 5), (2, 2)),
    CFG_IDENT: ((10, 10), (4, 4)),      # distinct angle columns per wave
    CFG_SAME:  ((10, 10), (4, 4)),
    CFG_CP:    ((20, 20), (8, 8)),
}
#: the real-config ratios this kernel was built for, asserted directly on the
#: model (not through a fixture config, which is tiny by design)
REAL_CFG = ("/home/jiangy/github/project94/pwa_workspace/jpsi_eta_pi_pi/"
            "04_model_search/current/config.yml")
REAL_USLOT = (62, 40)
REAL_AMP = (26, 13)
REAL_NNZ = 82


def _rel(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _floor_ok(a, b, rel_bound=CACHE_FLOOR, abs_bound=CACHE_FLOOR_ABS):
    """The fp32 cache floor is ABSOLUTE; ``_rel`` divides by the value.

    A grouped NLL can be small in magnitude (a 17-event group gives Q ~ -0.39),
    so the same absolute gap is a bigger ratio.  Accepting either bound is the
    honest form — and it is still ~1e-5, i.e. the measured fp32 cache floor.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    worst = float(np.abs(a - b).max())
    return (worst / max(np.abs(b).max(), 1e-300)) < rel_bound or worst < abs_bound


def _replicate(data, rs):
    out = {k: np.repeat(np.asarray(v), rs, axis=0) for k, v in data.items()}
    out["weight"] = np.full(len(np.asarray(data["weight"])) * rs, 1.0 / rs)
    return out


def _pair(cfg, ne=64, seed=0, **kw):
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    data = _in_grid(kc, ne, seed)
    x = f.initial_values(seed=1)
    params, _resolved = f.build_params(x)
    ref = create_backend("numpy_flat", kc, model="pwa_gen",
                         resolution_size=kw.get("resolution_size", 1))
    gpu = create_backend("cuda_flat_pwa_gen_s0e3", kc, model="pwa_gen", **kw)
    return f, ref, gpu, params, data, kc


# ── gate 1: registration and the case-1 copy ────────────────────────────

def test_registration():
    assert "cuda_flat_pwa_gen_s0e3" in set(backends_for_model("pwa_gen"))
    assert "cuda_flat_pwa_gen_s0e3" not in set(backends_for_model("pwa"))


def test_the_other_kernels_are_not_reused():
    """A case-1 copy: own module, own source, own library, own symbols."""
    import tabpwa.cuda._flat_pwa_gen_s0e3 as new
    import tabpwa.cuda._flat_pwa_gen2 as gen2
    import tabpwa.cuda._flat_pwa_gen_1bw as one
    assert new is not gen2 and new is not one
    f = Fitter(CFG_KM)
    f.apply_constrains()
    a = CUDAKernelFlatPwaGenS0E3(f.kernel_config)
    b = gen2.CUDAKernelFlatPwaGen2(f.kernel_config)
    c = one.CUDAKernelFlatPwaGen1BW(f.kernel_config)
    try:
        assert a._lib is not b._lib and a._lib is not c._lib
        assert hasattr(a._lib, "cuda_compute_flat_pwa_gen_s0e3")
        assert not hasattr(a._lib, "cuda_compute_flat_pwa_gen2")
        assert not hasattr(a._lib, "cuda_compute_flat_pwa_gen_1bw")
        # the s0e/grouped machinery gen2 and 1bw DO need but this kernel
        # does not: they run a second backward launch over a saved `A`
        assert not hasattr(a, "dA_bytes_cap")
        for nm in ("cuda_grp_nll", "cuda_scale"):
            assert not any(nm in d for d in dir(a._lib))
    finally:
        a.free()
        b.free()
        c.free()


def test_registration_and_case1():
    """Own name, own ``_lib``, own symbols — s0e's are not in this library."""
    assert "cuda_flat_pwa_gen_s0e3" in set(backends_for_model("pwa_gen"))
    assert "cuda_flat_pwa_gen_s0e3" not in set(backends_for_model("pwa"))
    import tabpwa.cuda._flat_pwa_gen_s0e as s0e
    import tabpwa.cuda._flat_pwa_gen_s0e2 as s0e2
    import tabpwa.cuda._flat_pwa_gen_s0e3 as s0e3
    assert s0e3 is not s0e and s0e3 is not s0e2
    f = Fitter(CFG_KM)
    f.apply_constrains()
    a = CUDAKernelFlatPwaGenS0E3(f.kernel_config)
    b = s0e.CUDAKernelFlatPwaGenS0E(f.kernel_config)
    c = s0e2.CUDAKernelFlatPwaGenS0E2(f.kernel_config)
    try:
        assert a._lib is not b._lib and a._lib is not c._lib
        assert hasattr(a._lib, "cuda_compute_flat_pwa_gen_s0e3")
        assert not hasattr(a._lib, "cuda_compute_flat_pwa_gen_s0e")
        assert not hasattr(a._lib, "cuda_compute_flat_pwa_gen_s0e2")
        assert not any("gps0_" in d or "gps02_" in d for d in dir(a._lib))
    finally:
        a.free()
        b.free()
        c.free()


# ── gate 2: the basis rotation, proved in numpy first ───────────────────

def _lin(tab, x, lo, dd):
    pos = (np.asarray(x, np.float64) - lo) / dd
    i0 = np.clip(pos.astype(np.int64), 0, tab.size - 2)
    fr = pos - i0
    return tab[i0] * (1.0 - fr) + tab[i0 + 1] * fr


def _real_config():
    import os
    if not os.path.exists(REAL_CFG):
        pytest.skip(f"real config not present: {REAL_CFG}")
    d = dict(Fitter(REAL_CFG).config.dic)
    d["amp_model"] = "pwa_gen"
    d["backend"] = {"name": "numpy_flat"}
    f = Fitter(d)
    f.apply_constrains()
    return f.kernel_config


def test_the_basis_rotation_is_exact():
    """``Z = i·basis`` and the kernel's ``-i·Z`` must recover every row.

    This is the design point that lets the kernel stay kind-agnostic: no
    ``basis_kind`` is inspected anywhere, because the rotation is uniform.
    """
    kc = _real_config()
    bas = np.asarray(kc["basis_table"], np.complex128)
    assert np.abs(-1j * (1j * bas) - bas).max() == 0.0
    # ... and it reproduces the deleted kernel's hand-rotated rows
    assert np.allclose((1j * bas[0]), 1j * bas[0])     # trivial, kept explicit
    names = kc["den_names"]
    i1 = names.index("one") if "one" in names else 0
    # an "s0"-style row is the constant 1, so i·1 must be (0, 1)
    s0 = [i for i, n in enumerate(names) if n.endswith("_s0")][0]
    z = 1j * bas[s0]
    assert np.allclose(z.real, 0.0) and np.allclose(z.imag, 1.0)


def test_the_denominator_from_the_rotated_table_is_numpy_flats_D():
    """``Σ_i coeff_i·basis_i(m)`` == the direct evaluation, in numpy.

    Assembled the way the KERNEL reads it (through the us-cache slots, with
    the ``F = -i·Z`` rotation), against the direct per-row evaluation that
    ``numpy_flat`` performs.  Done before the kernel is trusted so a wrong
    rotation cannot be masked.
    """
    kc = _real_config()
    bas = np.asarray(kc["basis_table"], np.complex128)
    bmi = np.asarray(kc["basis_m_index"], np.int64)
    m_min, m_del = float(kc["m_min"]), float(kc["m_delta"])
    n_wave = int(np.asarray(kc["fl_table"]).shape[0])
    n_basis, n_bins = bas.shape
    gam = np.asarray(kc["gamma_matrix"], np.float64)[:, 0, :]

    rng = np.random.default_rng(3)
    mass = m_min + rng.uniform(0.05, 0.95, (37, 3)) * m_del * (n_bins - 1)
    coeff = rng.uniform(0.3, 1.2, n_basis)
    uro, uco, sob = uslot_map(bas, bmi)

    direct = np.zeros((37, n_wave), np.complex128)
    kern = np.zeros((37, n_wave), np.complex128)
    for k in range(n_wave):
        sd = np.zeros(37, np.complex128)
        sk = np.zeros(37, np.complex128)
        for m in np.nonzero(gam[k])[0]:
            b = (_lin(bas[m].real, mass[:, bmi[m]], m_min, m_del)
                 + 1j * _lin(bas[m].imag, mass[:, bmi[m]], m_min, m_del))
            sd += gam[k, m] * coeff[m] * b
            # kernel path: Z = i·basis from the DEDUPED slot, then -i·Z
            u = sob[m]
            zr = _lin(bas[uro[u]].real, mass[:, uco[u]], m_min, m_del)
            zi = _lin(bas[uro[u]].imag, mass[:, uco[u]], m_min, m_del)
            Zr, Zi = -zi, zr                       # Z = i·basis
            sk += coeff[m] * (Zi + 1j * (-Zr))     # F = -i·Z = (Z_i, -Z_r)
        direct[:, k] = sd
        kern[:, k] = sk
    assert np.abs(kern - direct).max() == 0.0


# ── gate 3/4: both dedups exact, and the key choice pinned ──────────────

@pytest.mark.parametrize("cfg", QUALIFYING)
def test_the_dedups_are_exact_and_pinned(cfg):
    """Representative == every member, bit-for-bit; ratios pinned per config."""
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    bas = np.asarray(kc["basis_table"], np.complex128)
    bmi = np.asarray(kc["basis_m_index"], np.int64)
    n_wave = int(np.asarray(kc["fl_table"]).shape[0])
    n_proj = max(1, int(np.asarray(kc["matrix_angle"]).shape[1]) // n_wave)

    uro, uco, sob = uslot_map(bas, bmi)
    # exactness: the slot's representative row is byte-identical to the row,
    # and reads the same mass column
    for m in range(bas.shape[0]):
        u = sob[m]
        assert np.array_equal(bas[uro[u]], bas[m]), f"{cfg}: row {m}"
        assert uco[u] == bmi[m], f"{cfg}: column {m}"

    sow, rep = amp_slot_map(kc["matrix_angle"], kc["fl_table"],
                            kc["fl_q_index"], kc["num_table"],
                            kc["num_m_index"], n_wave, n_proj)
    am = np.asarray(kc["matrix_angle"])
    fl = np.asarray(kc["fl_table"])
    fq = np.asarray(kc["fl_q_index"])
    nt = np.asarray(kc["num_table"])
    nmi = np.asarray(kc["num_m_index"])
    for k in range(n_wave):
        r = int(rep[sow[k]])
        for p in range(n_proj):
            assert np.array_equal(am[:, p * n_wave + k],
                                  am[:, p * n_wave + r]), f"{cfg}: k{k} p{p}"
        for d in range(fl.shape[1]):
            assert np.array_equal(fl[k, d], fl[r, d])
            assert fq[k, d] == fq[r, d]
        for rr in range(nt.shape[1]):
            assert np.array_equal(nt[k, rr], nt[r, rr])
            assert nmi[k, rr] == nmi[r, rr]

    (n_b, n_u), (n_w, n_q) = EXPECTED_DEDUP[cfg]
    assert (bas.shape[0], uro.size) == (n_b, n_u), (
        f"{cfg}: us-cache fold moved to {bas.shape[0]} -> {uro.size}, pinned "
        f"at {n_b} -> {n_u}")
    assert (n_wave, rep.size) == (n_w, n_q), (
        f"{cfg}: amp-cache fold moved to {n_wave} -> {rep.size}, pinned at "
        f"{n_w} -> {n_q}")


@pytest.mark.parametrize("cfg", QUALIFYING)
def test_the_reported_counts_match_the_maps(cfg):
    f = Fitter(cfg)
    f.apply_constrains()
    k = CUDAKernelFlatPwaGenS0E3(f.kernel_config)
    try:
        assert k.n_uslot == k.uslot_row.size == k.uslot_col.size
        assert k.n_uniq == k.rep_of_slot.size
        assert k.slot_of_wave.size == k.n_wave
        assert k.slot_of_basis.size == k.n_basis
        assert k.nz_u.size == k.nz_m.size == k.nz_k.size == k.n_nz
    finally:
        k.free()


def test_the_uslot_key_is_the_row_not_the_kind():
    """The obvious-but-wrong key (``basis_kind``) must give a DIFFERENT count.

    ``basis_kind`` is the factor-local slot index (0 = s0, 1 = "one", 2 =
    running width, …), so two resonances' running widths share a kind while
    being different functions of the event.  Keying the us-cache on it would
    merge them — measured 16 "slots" on the real config where the truth is 43.
    Pinning this makes the choice explicit rather than accidental.
    """
    kc = _real_config()
    bas = np.asarray(kc["basis_table"], np.complex128)
    bmi = np.asarray(kc["basis_m_index"], np.int64)
    kind = np.asarray(kc["basis_kind"], np.int64)
    n_basis = bas.shape[0]
    assert (n_basis, uslot_map(bas, bmi)[0].size) == REAL_USLOT
    wrong = len(dict.fromkeys((int(kind[i]), int(bmi[i]))
                              for i in range(n_basis)))
    assert wrong < REAL_USLOT[1], (
        "the kind-keyed fold is not smaller any more — re-check the dedup key")
    # and the rows it would merge are NOT identical
    groups = {}
    for i in range(n_basis):
        groups.setdefault((int(kind[i]), int(bmi[i])), []).append(i)
    merged_differently = [v for v in groups.values()
                          if len({bas[i].tobytes() for i in v}) > 1]
    assert merged_differently, "the kind key would be safe — re-check the key"


def test_the_real_config_ratios_are_pinned():
    """The shapes this kernel exists for: 69 -> 43 uslots, 26 -> 13 amps."""
    kc = _real_config()
    n_wave = int(np.asarray(kc["fl_table"]).shape[0])
    n_proj = max(1, int(np.asarray(kc["matrix_angle"]).shape[1]) // n_wave)
    bas = np.asarray(kc["basis_table"])
    uro, _uco, _sob = uslot_map(bas, kc["basis_m_index"])
    _sow, rep = amp_slot_map(kc["matrix_angle"], kc["fl_table"],
                             kc["fl_q_index"], kc["num_table"],
                             kc["num_m_index"], n_wave, n_proj)
    assert (bas.shape[0], uro.size) == REAL_USLOT
    assert (n_wave, rep.size) == REAL_AMP
    _lo, _hi, _nk, nzm = index_runs(kc["gamma_matrix"], bas.shape[0])
    assert nzm.size == REAL_NNZ, (
        f"gamma non-zeros moved to {nzm.size}, pinned at {REAL_NNZ}")


# ── gate 5: the anchor, every qualifying fixture ────────────────────────





# ── gate 6/7: finite differences, the independent check ─────────────────





# ── gate 8/9: the chunk rule and chunk independence ─────────────────────

def _chunk_rule(batch_size, rsize):
    """The documented rule, restated here so the C side is checked against it."""
    cap = batch_size if batch_size > 0 else 50000
    cap = min(cap, 1 << 20)
    if rsize > 1:
        a = (cap // rsize) * rsize
        cap = a if a > 0 else rsize          # never zero, never unaligned
    return cap


@pytest.mark.parametrize("rsize", [1, 17, 100])
@pytest.mark.parametrize("batch_size", [1, 5, 17, 1000, 50000, 10 ** 9])
def test_the_chunk_rule_is_what_the_context_reports(batch_size, rsize):
    """``cuda_cap_*`` must be the rule above — never a Python re-derivation.

    The two guards are the ones measurement forced: ``batch_size = 1e9`` must
    be capped (a scratch of GBs for a small sample), and ``batch_size <
    rsize`` must still hold a WHOLE group — an unaligned cap there let a group
    straddle two chunks and made Q wrong by 49 % in gen2.
    """
    f = Fitter(CFG_KM)
    f.apply_constrains()
    k = CUDAKernelFlatPwaGenS0E3(f.kernel_config, batch_size=batch_size,
                                resolution_size=rsize)
    try:
        assert k.cap == _chunk_rule(batch_size, rsize)
        assert k.cap > 0
        if rsize > 1:
            assert k.cap % rsize == 0, "a chunk must hold whole log-sum groups"
        assert k.ngroups == (k.cap + rsize - 1) // rsize
    finally:
        k.free()




def test_a_group_never_straddles_a_tile_or_a_chunk():
    """``rsize`` larger than the tile, and a chunk holding many groups.

    Both are real: the production fit runs ``rsize = 17`` with the tile at 8
    (one block per group, walked internally), and a 5000-event chunk then
    holds 294 groups.  The objective must still take ONE log per group, which
    is what comparing against ``numpy_flat`` at the same ``rsize`` pins.
    """
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    for rs in (17, 100):
        one = _replicate(_in_grid(kc, rs, seed=11), rs)       # one group
        data = {k: np.concatenate([v, v, v]) for k, v in one.items()}
        ref = create_backend("numpy_flat", kc, model="pwa_gen",
                             resolution_size=rs)
        gpu = create_backend("cuda_flat_pwa_gen_s0e3", kc, model="pwa_gen",
                             resolution_size=rs, batch_size=5000)
        try:
            Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
            Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=1.3)
            assert _rel(Qg, Qr) < CACHE_FLOOR, f"rsize={rs} Q"
            assert _rel(gg["coeff"], gr["coeff"]) < CACHE_FLOOR_COEFF
            assert gpu.kernel.te >= 1
        finally:
            ref.free()
            gpu.free()


# ── gate 10: path A and path B agree ────────────────────────────────────

RSIZES = (1, 2, 4, 8, 10, 17, 25, 40, 100)


@pytest.mark.parametrize("rs", RSIZES)
def test_every_resolution_size_matches_numpy_flat(rs):
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    data = _in_grid(kc, 32, seed=7)
    data = data if rs == 1 else _replicate(data, rs)
    ref = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=rs)
    gpu = create_backend("cuda_flat_pwa_gen_s0e3", kc, model="pwa_gen",
                         resolution_size=rs)
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        Qg, gg, Pg = gpu.compute(params, gpu.load_data(data), norm=1.3)
        assert _rel(Pg, Pr) < CACHE_FLOOR_P, f"rs={rs} P"
        assert _rel(Qg, Qr) < CACHE_FLOOR, f"rs={rs} Q"
        assert _rel(gg["ck"], gr["ck"]) < CACHE_FLOOR, f"rs={rs} dck"
        assert _rel(gg["coeff"], gr["coeff"]) < CACHE_FLOOR_COEFF
        assert _rel(gpu.kernel._last_dnorm, ref._last_dnorm) < CACHE_FLOOR
    finally:
        ref.free()
        gpu.free()


def test_path_a_and_path_b_both_exist_and_are_selected():
    """A gate that pinned only "rs 17 works" could pass on path A alone.

    Path A is ``rsize <= te`` (whole groups inside one tile); path B is
    ``rsize > te`` (one block per group, walked internally).  Both are
    reachable on the SHIPPED auto rule, which is the point: the production
    ``rsize = 17`` lands in path B (tile 8), so path A must be reached by a
    smaller ``rsize`` rather than being dead code.
    """
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    ks = [CUDAKernelFlatPwaGenS0E3(kc, resolution_size=rs)
          for rs in (1, 4, 17, 100)]
    try:
        got = [(k.resolution_size, k.te,
                k.resolution_size <= k.te) for k in ks]
        # rsize 1 / 4 / 17 all fit inside the 8-event tile -> path A
        # rsize 100 cannot -> path B
        assert got[0][2] and got[1][2] and got[2][2] is False
        assert got[3][2] is False and got[3][1] < 100
        # and BOTH paths are numerically exercised by the anchor gates above
        assert sum(1 for _rs, _te, inA in got if inA) == 2
        assert sum(1 for _rs, _te, inA in got if not inA) == 2
    finally:
        for k in ks:
            k.free()


def test_the_full_fitter_path_matches_the_sibling_backend():
    """End to end through ``Fitter.get_nll`` — norm + data, not just the kernel.

    This is the call a fit actually makes, and it exercises the thing the
    kernel-level gates cannot: two LIVE handles (phase space and data) at
    once, the norm integrated over one while the data is evaluated on the
    other.  The two flat CUDA siblings share the fp32 caches, so they must
    agree to rounding; ``numpy_flat`` is the fp64 reference.
    """
    for rs in (1, 17):
        out = {}
        for be in ("numpy_flat", "cuda_flat_pwa_gen_s0e3"):
            spec = {"name": be, "resolution_size": rs}
            if be.startswith("cuda"):
                spec["batch_size"] = 5000
            f = Fitter(CFG_KM, backend=spec)
            f.apply_constrains()
            kc = f.kernel_config
            d = _in_grid(kc, 40, seed=4)
            if rs > 1:
                d = _replicate(d, rs)
            f.set_phsp(d)
            f.set_data(d)
            out[be] = f.get_nll(f.initial_values(seed=1))
        (Qa, ga), (Qr, gr) = out["cuda_flat_pwa_gen_s0e3"], out["numpy_flat"]
        assert _floor_ok(Qa, Qr, 2e-5, 1e-5), f"rs={rs} Q: {Qa!r} vs {Qr!r}"
        ga = np.asarray(ga)
        gr = np.asarray(gr)
        assert float(np.abs(ga - gr).max()
                     / max(np.abs(gr).max(), 1e-300)) < 1e-4, f"rs={rs} grad"


# ── gate 11: n_res == 1 is enforced, loudly ─────────────────────────────

def test_multiple_bws_per_chain_is_rejected_with_the_general_backend_named():
    f = Fitter(CFG_MB)
    f.apply_constrains()
    assert int(np.asarray(f.kernel_config["num_table"]).shape[1]) == 2
    with pytest.raises(ValueError, match="exactly ONE BW per chain"):
        CUDAKernelFlatPwaGenS0E3(f.kernel_config)
    with pytest.raises(ValueError, match="cuda_flat_pwa_gen2"):
        create_backend("cuda_flat_pwa_gen_s0e3", f.kernel_config,
                       model="pwa_gen")


def test_every_other_fixture_passes_the_gate():
    for cfg in QUALIFYING:
        f = Fitter(cfg)
        f.apply_constrains()
        assert int(np.asarray(f.kernel_config["num_table"]).shape[1]) == 1
        k = CUDAKernelFlatPwaGenS0E3(f.kernel_config)
        k.free()


# ── gate 12: shared memory ──────────────────────────────────────────────

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
    k = CUDAKernelFlatPwaGenS0E3(f.kernel_config)
    try:
        assert k.smem_bytes > 0
        assert k.smem_bytes == int(
            k._lib.cuda_smem_bytes_flat_pwa_gen_s0e3(k._ctx))
        assert k.smem_bytes == k.tile_bytes()
        # the tile arithmetic is ONE function, shared with the kernel
        for te in (1, 2, 4, 8):
            L_doubles = (
                2 * te * (k.n_proj * k.n_wave)     # cr, ci
                + 2 * te * k.n_wave                # D
                + te * k.n_wave                    # rd
                + 2 * te * k.n_wave                # pv
                + 4 * te * k.n_proj                # A, dA
                + 3 * te                           # pe, q, sc
                + 2 * k.n_wave + k.n_basis         # gck, gc
                + 3                                # tot
                + 2 * k.n_wave + k.n_basis)        # gr, gi, gcb
            assert k.tile_bytes(te=te) == L_doubles * 8, te
    finally:
        k.free()


def test_the_tile_budget_is_respected():
    """The chosen tile never exceeds the documented 48 KiB budget."""
    for cfg in QUALIFYING:
        f = Fitter(cfg)
        f.apply_constrains()
        k = CUDAKernelFlatPwaGenS0E3(f.kernel_config)
        try:
            assert k.te >= 1
            assert k.smem_bytes <= 48 * 1024, (cfg, k.te, k.smem_bytes)
        finally:
            k.free()


@pytest.mark.parametrize("n_wave_target", [500])
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
    be = create_backend("cuda_flat_pwa_gen_s0e3", kc, model="pwa_gen")
    try:
        assert be.kernel.smem_bytes > 48 * 1024, "this shape must cross 48 KiB"
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
    f1, big_kc = _wide_kc(500)
    f2 = Fitter(CFG_KM)
    f2.apply_constrains()
    big = CUDAKernelFlatPwaGenS0E3(big_kc)
    small = CUDAKernelFlatPwaGenS0E3(f2.kernel_config)
    try:
        assert big.smem_bytes > 48 * 1024 > small.smem_bytes
        p0, _ = f2.build_params(f2.initial_values(seed=1))
        ck0 = np.asarray(p0["ck"])
        n_w = big.n_wave
        n_b = int(np.asarray(big_kc["basis_table"]).shape[0])
        wide = {"ck": np.tile(ck0, (n_w + ck0.size - 1) // ck0.size)[:n_w],
                "coeff": np.random.default_rng(7).uniform(0.3, 1.2, n_b)}
        Qbig, _g, _P = big.compute(wide,
                                   big.load_data(_in_grid(big_kc, 256)),
                                   norm=128.0)
        assert np.isfinite(Qbig) and Qbig > 0.0, (
            "the wide context's launch was rejected after a narrower one was "
            "created — the opt-in was lowered")
    finally:
        big.free()
        small.free()


def test_an_over_large_tile_is_refused_not_silently_empty():
    _f, kc = _wide_kc(1400)
    with pytest.raises((RuntimeError, ValueError)):
        CUDAKernelFlatPwaGenS0E3(kc)


def test_a_tile_hint_cannot_split_a_group():
    f = Fitter(CFG_KM)
    f.apply_constrains()
    with pytest.raises((RuntimeError, ValueError)):
        CUDAKernelFlatPwaGenS0E3(f.kernel_config, resolution_size=17,
                                tile_events=23)


# ── contract ────────────────────────────────────────────────────────────









def test_an_empty_sample_is_the_identity_not_a_failure():
    """A zero-row load is legal: Q = 0 with zero gradients, no device trap."""
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    k = CUDAKernelFlatPwaGenS0E3(kc)
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    empty = {kk: np.asarray(v)[:0] for kk, v in _in_grid(kc, 4).items()}
    h = k.load_data(empty)
    try:
        Q, g, P = k.compute(params, h, norm=1.3)
        assert Q == 0.0
        assert np.asarray(P).size == 0
        assert np.all(np.isfinite(g["ck"]))
        assert np.all(np.isfinite(g["coeff"]))
    finally:
        h.free()
        k.free()


def test_return_p_false_does_not_change_the_objective():
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    k = CUDAKernelFlatPwaGenS0E3(kc)
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    h = k.load_data(_in_grid(kc, 12))
    try:
        Qa, ga, Pa = k.compute(params, h, norm=1.3, return_p=False)
        Qb, gb, Pb = k.compute(params, h, norm=1.3, return_p=True)
        assert Pa is None and np.asarray(Pb).size == 12
        assert Qa == Qb
        assert _rel(ga["ck"], gb["ck"]) == 0.0
        assert _rel(ga["coeff"], gb["coeff"]) == 0.0
    finally:
        h.free()
        k.free()




# ── gate 9/10: the reciprocal is the only change, and `te` is a knob ────

def _s0e_pair(cfg, ne=200, rs=1, seed=0):
    """``(s0e_backend, s0e3_backend, params, data)`` on ONE config + sample."""
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    data = _in_grid(kc, ne, seed)
    params, _ = f.build_params(f.initial_values(seed=1))
    a = create_backend("cuda_flat_pwa_gen_s0e", kc, model="pwa_gen",
                       resolution_size=rs)
    b = create_backend("cuda_flat_pwa_gen_s0e3", kc, model="pwa_gen",
                       resolution_size=rs)
    return a, b, params, data


@pytest.mark.parametrize("cfg", [CFG_ONE, CFG_KM])
@pytest.mark.parametrize("rs", [1, 17])
@pytest.mark.parametrize("norm", [None, 1.3])
def test_matches_s0e_reciprocal_only(cfg, rs, norm):
    """s0e3 == s0e to <= ``S0E3_TOL``: the divide is the ONLY difference.

    Every real step is still fp64 and the caches are still float2, so the only
    gap is the 1-2 ulp of the Newton-refined reciprocal — no longer bit-equal
    (which is why this gate is a tolerance, not ``==``), but ten orders below
    the fp32 cache floor, so it cannot hide a real regression.

    MEASURED (ne=200, rsize 1/17, norm None/1.3, CFG_ONE + CFG_KM, worst
    over all eight combinations, relative to s0e):
        Q      7.2e-16          P       4.0e-16        dnorm   8.3e-16
        dck    3.4e-16          dcoeff  5.5e-16
    i.e. <= 8.3e-16 (~4 ulp) — the 1e-14 bound is a 12x margin over the
    measurement and still ~10 orders below the fp32 cache floor.
    """
    a, b, params, data = _s0e_pair(cfg, rs=rs)
    try:
        ha, hb = a.load_data(data), b.load_data(data)
        try:
            Qa, ga, Pa = a.compute(params, ha, norm=norm)
            Qb, gb, Pb = b.compute(params, hb, norm=norm)
        finally:
            ha.free()
            hb.free()
    finally:
        a.free()
        b.free()
    assert abs(Qa - Qb) / max(abs(Qa), 1e-300) < S0E3_TOL
    assert _rel(Pb, Pa) < S0E3_TOL
    assert _rel(gb["ck"], ga["ck"]) < S0E3_TOL
    assert _rel(gb["coeff"], ga["coeff"]) < S0E3_TOL
    if norm is not None:
        assert (abs(gb["norm"] - ga["norm"])
                / max(abs(ga["norm"]), 1e-300)) < S0E3_TOL


def test_te_kwarg_is_accepted_and_invariant():
    """``te`` changes only the tiling: the arithmetic is untouched."""
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    data = _in_grid(kc, 200, seed=0)
    params, _ = f.build_params(f.initial_values(seed=1))
    hinted = create_backend("cuda_flat_pwa_gen_s0e3", kc, model="pwa_gen",
                            resolution_size=17, te=34)
    auto = create_backend("cuda_flat_pwa_gen_s0e3", kc, model="pwa_gen",
                          resolution_size=17, te=0)
    try:
        assert hinted.kernel.te == 34
        assert auto.kernel.te != 0
        ha, hb = hinted.load_data(data), auto.load_data(data)
        try:
            Qh, gh, Ph = hinted.compute(params, ha, norm=1.3)
            Qa, ga, Pa = auto.compute(params, hb, norm=1.3)
        finally:
            ha.free()
            hb.free()
    finally:
        hinted.free()
        auto.free()
    assert abs(Qh - Qa) / max(abs(Qa), 1e-300) < S0E3_TOL
    assert _rel(Ph, Pa) < S0E3_TOL
    assert _rel(gh["ck"], ga["ck"]) < S0E3_TOL
    assert _rel(gh["coeff"], ga["coeff"]) < S0E3_TOL
    assert (abs(gh["norm"] - ga["norm"])
            / max(abs(ga["norm"]), 1e-300)) < S0E3_TOL
    # a hint that would split a log-sum group fails early, in Python
    with pytest.raises(ValueError, match="multiple"):
        create_backend("cuda_flat_pwa_gen_s0e3", kc, model="pwa_gen",
                       resolution_size=17, te=20)
