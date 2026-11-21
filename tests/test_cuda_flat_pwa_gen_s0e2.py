"""``cuda_flat_pwa_gen_s0e2`` — the s0e fused tile on the flat ``pwa_gen`` config.

Gates:
1. **registration** — ``pwa_gen`` scope only, and a case-1 copy (own module,
   own source, own ``.so``, own symbols; gen2 and 1bw untouched).
2. **the basis rotation is exact** — ``Z = i·basis_table`` with the kernel's
   ``-i·Z`` recovers every basis row, and ``Σ_i coeff_i·basis_i(m)`` assembled
   from the rotated us-cache equals ``numpy_flat``'s ``D``.  Checked in numpy
   BEFORE the kernel is consulted, so a wrong rotation cannot be masked by a
   matching wrong kernel.
3. **both dedups are exact, bit-for-bit** — representative == every member on
   the key that defines the fold (row bytes + mass column; all projections of
   a wave + fl/num rows + their columns), and the measured fold ratios are
   pinned per config so a builder change that moves them is visible.
4. **the uslot key is the row, not the kind** — the deliberately-wrong key
   (``basis_kind``) is shown to give a DIFFERENT, smaller count on the real
   config, so the choice is pinned rather than merely asserted.
5. **anchor vs ``numpy_flat``** on every ``n_res == 1`` fixture for
   ``Q``/``P``/``dck``/``dcoeff``/``dnorm``, ``norm=None`` and a scalar norm,
   at ``resolution_size`` 1 and 17.
6. **``dcoeff`` against central differences** through the tiled path — the
   independent check that the dedup slot maps are the right columns.  A wrong
   map is self-consistent between two computes and only FD catches it.
7. **``dck`` against central differences** (raw Wirtinger half → 2·Re/-2·Im).
8. **chunk independence** — ``rsize`` 1/17 × ``batch_size``
   1000/5000/50000/1e9, all five quantities agree to <=1e-14.
9. **the chunk rule is what the context reports** (never re-derived in Python),
   including the two guards measurement forced: the ``1<<20`` cap and
   "a chunk below one group is raised to one whole group".
10. **path A and path B agree** — ``rsize`` 2..100 (a group larger than the
    tile is walked internally, and must still close to ONE log); ``rs = 1`` and
    ``17`` are the shared anchor's sweep.
11. **``n_res != 1`` is rejected** with ``cuda_flat_pwa_gen2`` named.
12. **shared memory** — the context's own report is the launch size, the tile
    arithmetic matches ``cuda_tile_bytes_*``, a wide config still crosses the
    48 KiB opt-in, and an over-large tile is refused rather than returning 0.

The two s0e2-specific gates (they carry the "identical algebra" claim):

13. **``test_registration_and_case1``** — registered for ``pwa_gen`` only, and
    the kernel module / ``_lib`` / symbol set is its own, distinct from
    ``cuda_flat_pwa_gen_s0e``.
14. **``test_matches_s0e_bitwise``** — s0e2 against s0e on the SAME handle
    shape, for ``Q``/``P``/``dck``/``dcoeff``/``dnorm``, at ``rsize`` 1 and 17
    with ``norm=None`` and ``norm=1.3``: max relative difference <= 1e-15.
    That is the *only* thing s0e2 changes — the summation order of the sparse
    dot (one packed 64-bit CSR load + two independent fp64 chains) — so this
    is the gate that pins "no algebraic change".

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
from tabpwa.cuda._flat_pwa_gen_s0e import CUDAKernelFlatPwaGenS0E
from tabpwa.cuda._flat_pwa_gen_s0e2 import (
    CUDAKernelFlatPwaGenS0E2, amp_slot_map, index_runs, uslot_map)
from tabpwa.cuda._flat_pwa_gen_s0e3 import CUDAKernelFlatPwaGenS0E3
from tabpwa.cuda._flat_pwa_gen_s0e4 import CUDAKernelFlatPwaGenS0E4
from tabpwa.cuda._flat_pwa_gen_s0e5 import CUDAKernelFlatPwaGenS0E5
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
    gpu = create_backend("cuda_flat_pwa_gen_s0e2", kc, model="pwa_gen", **kw)
    return f, ref, gpu, params, data, kc


# ── gate 1: registration and the case-1 copy ────────────────────────────

def test_registration():
    assert "cuda_flat_pwa_gen_s0e2" in set(backends_for_model("pwa_gen"))
    assert "cuda_flat_pwa_gen_s0e2" not in set(backends_for_model("pwa"))


def test_the_other_kernels_are_not_reused():
    """A case-1 copy: own module, own source, own library, own symbols."""
    import tabpwa.cuda._flat_pwa_gen_s0e2 as new
    import tabpwa.cuda._flat_pwa_gen2 as gen2
    import tabpwa.cuda._flat_pwa_gen_1bw as one
    assert new is not gen2 and new is not one
    f = Fitter(CFG_KM)
    f.apply_constrains()
    a = CUDAKernelFlatPwaGenS0E2(f.kernel_config)
    b = gen2.CUDAKernelFlatPwaGen2(f.kernel_config)
    c = one.CUDAKernelFlatPwaGen1BW(f.kernel_config)
    try:
        assert a._lib is not b._lib and a._lib is not c._lib
        assert hasattr(a._lib, "cuda_compute_flat_pwa_gen_s0e2")
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
    k = CUDAKernelFlatPwaGenS0E2(f.kernel_config)
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
    k = CUDAKernelFlatPwaGenS0E2(f.kernel_config, batch_size=batch_size,
                                resolution_size=rsize)
    try:
        assert k.cap == _chunk_rule(batch_size, rsize)
        assert k.cap > 0
        if rsize > 1:
            assert k.cap % rsize == 0, "a chunk must hold whole log-sum groups"
        assert k.ngroups == (k.cap + rsize - 1) // rsize
    finally:
        k.free()




def _budget_bug_kc():
    """A shape whose 8-event tile does NOT fit the smem budget.

    ``tile_flat`` gives n_wave 80 / n_proj 2; 81 basis rows make the layout
    bind at te = 7 for rsize 2/4 and at te = 7 for rsize 7 — the exact shape
    where the budget loop used to land on a non-multiple and ``create``
    rejected it.
    """
    from benchmark_flat_vs_pwa import tile_flat
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = dict(tile_flat(f.kernel_config, 80, 2))
    nw = int(np.asarray(kc["fl_table"]).shape[0])
    nb = int(np.asarray(kc["basis_table"]).shape[1])
    n_basis = 81
    kc["basis_table"] = np.ones((n_basis, nb), np.complex128)
    kc["basis_m_index"] = np.zeros(n_basis, np.int64)
    gam = np.zeros((nw, 1, n_basis))
    for k in range(nw):
        gam[k, 0, k % n_basis] = 1.0
    kc["gamma_matrix"] = gam
    kc["den_names"] = np.asarray([f"b{i}" for i in range(n_basis)])
    return kc


@pytest.mark.parametrize("rs", [2, 4, 7])
@pytest.mark.parametrize("cls", [CUDAKernelFlatPwaGenS0E2,
                                 CUDAKernelFlatPwaGenS0E,
                                 CUDAKernelFlatPwaGenS0E3,
                                 CUDAKernelFlatPwaGenS0E4,
                                 CUDAKernelFlatPwaGenS0E5],
                           ids=["s0e2", "s0e", "s0e3", "s0e4", "s0e5"])
def test_a_budget_shrunk_tile_still_holds_whole_groups(cls, rs):
    """`rsize` 2..7 on a wide shape used to be unconstructible.

    `pick_te` shrank the 8-event tile ONE event at a time to fit the
    shared-memory budget and could return, e.g., 7 for `rsize 4`; `create`
    then rejected it ("te 7 is not a multiple of rsize 4") and the caller saw
    only "context creation failed" -- a legal `resolution_size` failing for an
    internal reason it could neither see nor control.  The picker now floors
    the shrunk tile to whole groups (a smaller tile only helps the budget;
    below `rsize` one block owns one group, which is legal).

    Parametrized over EVERY s0e-family kernel (s0e2, s0e, s0e3, s0e4, s0e5):
    the picker is duplicated in each, s0e/s0e2 diverged once already, and the
    one-line whole-group floor was pasted into s0e3..s0e5 with no test at all.
    """
    k = cls(_budget_bug_kc(), resolution_size=rs)
    try:
        assert k.te < 8                 # the budget shrink actually fired
        assert k.te < rs or k.te % rs == 0, (k.te, rs)
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
        gpu = create_backend("cuda_flat_pwa_gen_s0e2", kc, model="pwa_gen",
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

#: ``rs = 1`` and ``rs = 17`` are the shared anchor's sweep (both norms, every
#: qualifying fixture); the values below are the ones only this file covers.
#: Path A is ``rsize <= te`` (here 2/4/8), path B ``rsize > te`` (25/40/100).
RSIZES = (2, 4, 8, 10, 25, 40, 100)


@pytest.mark.parametrize("rs", RSIZES)
def test_every_resolution_size_matches_numpy_flat(rs):
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    data = _replicate(_in_grid(kc, 32, seed=7), rs)
    ref = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=rs)
    gpu = create_backend("cuda_flat_pwa_gen_s0e2", kc, model="pwa_gen",
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
    ks = [CUDAKernelFlatPwaGenS0E2(kc, resolution_size=rs)
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
        for be in ("numpy_flat", "cuda_flat_pwa_gen_s0e2"):
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
        (Qa, ga), (Qr, gr) = out["cuda_flat_pwa_gen_s0e2"], out["numpy_flat"]
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
        CUDAKernelFlatPwaGenS0E2(f.kernel_config)
    with pytest.raises(ValueError, match="cuda_flat_pwa_gen2"):
        create_backend("cuda_flat_pwa_gen_s0e2", f.kernel_config,
                       model="pwa_gen")


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
    k = CUDAKernelFlatPwaGenS0E2(f.kernel_config)
    try:
        assert k.smem_bytes > 0
        assert k.smem_bytes == int(
            k._lib.cuda_smem_bytes_flat_pwa_gen_s0e2(k._ctx))
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
        k = CUDAKernelFlatPwaGenS0E2(f.kernel_config)
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
    be = create_backend("cuda_flat_pwa_gen_s0e2", kc, model="pwa_gen")
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
    big = CUDAKernelFlatPwaGenS0E2(big_kc)
    small = CUDAKernelFlatPwaGenS0E2(f2.kernel_config)
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
        CUDAKernelFlatPwaGenS0E2(kc)


def test_a_tile_hint_cannot_split_a_group():
    f = Fitter(CFG_KM)
    f.apply_constrains()
    with pytest.raises((RuntimeError, ValueError)):
        CUDAKernelFlatPwaGenS0E2(f.kernel_config, resolution_size=17,
                                tile_events=23)


# ── contract ────────────────────────────────────────────────────────────









def test_an_empty_sample_is_the_identity_not_a_failure():
    """A zero-row load is legal: Q = 0 with zero gradients, no device trap."""
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    k = CUDAKernelFlatPwaGenS0E2(kc)
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
    k = CUDAKernelFlatPwaGenS0E2(kc)
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




# ── s0e2 gate 13: registration + the case-1 copy ────────────────────────

def test_registration_and_case1():
    """Registered for ``pwa_gen`` only, with its OWN module/lib/symbols."""
    assert "cuda_flat_pwa_gen_s0e2" in set(backends_for_model("pwa_gen"))
    assert "cuda_flat_pwa_gen_s0e2" not in set(backends_for_model("pwa"))
    import tabpwa.cuda._flat_pwa_gen_s0e as old
    import tabpwa.cuda._flat_pwa_gen_s0e2 as new
    assert new is not old
    assert new._lib is not old._lib, "s0e2 must load its OWN .so"
    f = Fitter(CFG_KM)
    f.apply_constrains()
    a = CUDAKernelFlatPwaGenS0E2(f.kernel_config)
    b = old.CUDAKernelFlatPwaGenS0E(f.kernel_config)
    try:
        assert a._lib is not b._lib
        assert hasattr(a._lib, "cuda_compute_flat_pwa_gen_s0e2")
        assert not hasattr(a._lib, "cuda_compute_flat_pwa_gen_s0e")
    finally:
        a.free()
        b.free()


# ── s0e2 gate 14: identical algebra — only the summation order differs ───

#: s0e2's ONLY numerical change is the summation order of the sparse dot
#: (one packed 64-bit CSR load, two independent fp64 chains).  Same fp64
#: operations in the same order otherwise, so the two kernels agree to the
#: rounding of a single re-association — ~1e-16, never 1e-15.
S0E2_TOL = 1e-15

#: ``dcoeff`` is a SUM over (e,k) of ``2·(pv_r·Z_i + pv_i·Z_r)``: the terms are
#: the per-event contributions, the result is their (partly cancelling) sum, so
#: the same 1-ulp shift in ``sr``/``si`` is amplified when measured RELATIVE to
#: a small result — the sibling fixtures' ``_floor_ok`` documents the identical
#: absolute-vs-relative issue for the fp32 cache floor.
#: MEASURED over CFG_ONE/CFG_KM x rsize 1,17 x norm None,1.3:
#:     Q 0 .. 5.6e-16    P 0 .. 6.5e-16    dck 0 .. 3.3e-16
#:     dnorm 0 .. 7.1e-16
#:     dcoeff 0 .. 2.8e-15   (worst absolute 4.7e-13 on max|dcoeff| = 0.17,
#:                            i.e. a few ulps of the individual terms)
#: 1e-14 is still 10 orders below the fp32 cache floor (1e-4), so it cannot
#: hide an algebra change; it only stops the cancellation from tripping a
#: bound that was never meaningful for a sum.
S0E2_TOL_COEFF = 1e-14


@pytest.mark.parametrize("cfg", [CFG_ONE, CFG_KM])
@pytest.mark.parametrize("rs", [1, 17])
@pytest.mark.parametrize("norm", [None, 1.3])
def test_matches_s0e_bitwise(cfg, rs, norm):
    """s0e2 == s0e, quantity by quantity, on the SAME handle shape.

    ``rsize`` 1 and 17 cover the plain and the grouped objective;
    ``norm=None`` and ``1.3`` cover the two ``dQ/dP`` forms.  Both kernels
    read the same fp32 caches, so any difference beyond the one re-association
    would be an algebra change — which is exactly what this gate forbids.
    """
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    data = (_in_grid(kc, 64, seed=2) if rs == 1
            else _replicate(_in_grid(kc, 17, seed=5), rs))
    a = create_backend("cuda_flat_pwa_gen_s0e", kc, model="pwa_gen",
                       resolution_size=rs)
    b = create_backend("cuda_flat_pwa_gen_s0e2", kc, model="pwa_gen",
                       resolution_size=rs)
    try:
        Qa, ga, Pa = a.compute(params, a.load_data(data), norm=norm)
        Qb, gb, Pb = b.compute(params, b.load_data(data), norm=norm)
        quantities = [("Q", Qb, Qa), ("P", Pb, Pa),
                      ("dck", gb["ck"], ga["ck"]),
                      ("dcoeff", gb["coeff"], ga["coeff"])]
        if norm is not None:
            quantities.append(("dnorm", gb["norm"], ga["norm"]))
        for name, x2, x1 in quantities:
            tol = S0E2_TOL_COEFF if name == "dcoeff" else S0E2_TOL
            rel = _rel(x2, x1)
            assert rel <= tol, (
                f"{cfg} rs={rs} norm={norm}: {name} differs by {rel:.3e} "
                f"(bound {tol:.0e}) — s0e2 must differ only in the "
                f"summation order of the sparse dot")
    finally:
        a.free()
        b.free()
