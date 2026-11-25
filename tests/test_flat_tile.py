"""The s0e fused tile family — ONE config contract, five kernels as parameters.

``cuda_flat_pwa_gen_s0e{,2,3,4,5}`` are case-1 copies of one kernel on the
flat ``pwa_gen`` config.  This file is a CONFIG test: it says what a config must
make the family DO, and treats each kernel as a parameter of that contract.  It
asserts nothing about how a copy is built internally — no tile size, no shared
memory arithmetic, no launch/counter API, no symbol or library layout — because
none of that is the contract and each of them would turn a legitimate
re-implementation into a red test.

Gates:

1. **the numeric contract** — ``Q``/``P``/``dck``/``dcoeff``/``dnorm`` against
   the fp64 ``numpy_flat`` oracle, on every qualifying fixture, at
   ``resolution_size`` 1 and 17 with and without a scalar norm, at the fp32
   cache floor recorded per quantity below.
2. **the grouped objective over the whole ``rsize`` range** — 2..100 on one
   fixture, i.e. groups smaller than, equal to and much larger than any tile
   the family picks.
3. **a group larger than any tiling closes to ONE log** — a sample built as
   whole replicated groups, three groups deep, at ``rsize`` 17 and 100.
4. **an empty sample is the identity** — ``Q = 0`` and zero gradients, which is
   what the oracle returns too.
5. **``return_p=False`` does not move the objective** — ``P`` is omitted, the
   value and every gradient are unchanged.
6. **refusal is part of the contract** — a multi-BW chain (``n_res != 1``)
   raises and names ``cuda_flat_pwa_gen2``; ``resolution_size < 1`` raises.
7. **a config that must CONSTRUCT does construct and agree** — a wide shape
   whose tile is above the 48 KiB shared-memory opt-in (a rejection there used
   to be swallowed into ``Q = 0``), and ``rsize`` 2/4/7 on a shape that binds
   the tile budget.  The second is the ``pick_te`` regression: the picker used
   to shrink the 8-event tile one event at a time and could return a
   non-multiple of ``rsize``, so ``create`` rejected a legal
   ``resolution_size`` with an internal message the caller could neither see
   nor control.  Both are stated as "it constructs AND matches ``numpy_flat``",
   which is the observable form of the bug.
8. **the parent digest** (one policy gate, a table field) — each copy reproduces
   the copy it was made from: ``s0e2``/``s0e3``/``s0e4`` against ``s0e``,
   ``s0e5`` against ``s0e2``, at the tolerance that copy's own re-association
   justifies.  These are the claims the variant exists to make ("the packed CSR
   is a re-association", "the Newton reciprocal is the only change", "the
   pre-gather and the row split only reorder a sum", "FMA + the deduped
   denominator move the last bit"), so they are kept as a table entry rather
   than a file each.
9. **a schedule knob is arithmetically invisible** (``s0e4``'s ``split_nnz``) —
   the one scenario the fixture configs cannot reach (their gamma rows are 5
   non-zeros, below every threshold): on the 26-wave production shape the 9/10
   non-zero rows are actually split, and every threshold must agree with the
   parent.

Test data stays INSIDE both interpolation grids (outside, the tables clamp and
CPU/GPU diverge — see ``_flat_grid._in_grid``).

The compute-function suite itself — the grouped-objective identity, FD
gradients, chunk invariance, handle independence, the input errors — lives in
``_flat_compute.py`` and is collected by ``test_flat_compute.py`` for EVERY
registered ``pwa_gen`` backend; it is not repeated here.
"""

import os

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import backends_for_model, create_backend

from _flat_compute import CONFIG_OWNER, FLOORS, SPECS as LEAVES
from _flat_grid import CFG_KM, CFG_MB, CFG_ONE
from _flat_grid import QUALIFYING, _in_grid

CFG_ID = [c.split("/")[-1].removesuffix(".yml") for c in QUALIFYING]

#: the 26-wave PRODUCTION flat config (skipped when not on this machine).  Its
#: gamma rows are the reason `s0e4`'s split exists: nnz per wave
#: ``{1: 5, 3: 11, 9: 8, 10: 2}`` — the fixture configs top out at 5, i.e.
#: BELOW the default threshold, so they would never exercise a split row.
CFG_PROD = ("/home/jiangy/github/project94/pwa_workspace/jpsi_eta_pi_pi/"
            "04_model_search/run_423_kmatrix_n_20261119_141110/"
            "config_kn4_plot.yml")

# ── the fp32 cache floor ────────────────────────────────────────────────
# The load-time caches are float2 (`us` = the rotated basis interp, `amp` =
# fa·fl·num), so agreement with the fp64 `numpy_flat` reference is limited by
# fp32, not by the algebra.  These are the constants every sibling fixture used,
# with the same reason: the error tracks P's DYNAMIC RANGE, not n_basis
# (`_ident` spans 7.5e-4 .. 7.2e2, so `_rel` = worst absolute / global max is
# set by the near-zero events).
# MEASURED for the family (ne=64, two seeds, rsize 1 and 17):
#   _pwa_gen, _km, _same   dQ ~4e-8 .. 2e-7   dck ~8e-8 .. 3e-7   dcoeff ~3e-7
#   _ident, _cp            dQ ~1e-6 .. 2e-5   dck ~7e-7 .. 1e-5   dcoeff ~7e-5
CACHE_FLOOR = 2e-5          # Q, dck, dnorm
CACHE_FLOOR_P = 3e-5        # per-event P
CACHE_FLOOR_COEFF = 1e-4    # dcoeff accumulates one rounding per (e,k,m)
#: The floor is a RELATIVE bound and the comparison keeps it: every quantity
#: these gates compare is a value of the same observable, and the grouped NLL is
#: the only one that can be small (Q ~ -0.39 for a 17-event group), where
#: MEASURED on ``config_pwa_gen_ident.yml`` at ``resolution_size = 17``:
#:     numpy_flat      Q = -0.3899469042747774
#:     these kernels   Q = -0.3899553800883502     (8.476e-06 apart, 2.2e-5 rel)
#:     cuda_flat_pwa_gen_1bw / gen2  Q = -0.38995538008835173
#: i.e. the family agrees with its two siblings to 4e-15 and the ENTIRE gap is
#: the shared float2 cache, so 2e-5 is the cache's own floor, not a slack bound.


# ── helpers ─────────────────────────────────────────────────────────────

def _rel(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _ulps(a, b):
    """Worst difference in ulps OF THE OBSERVABLE (i.e. of its largest entry).

    Per-entry spacing would call any difference on a near-zero entry a huge
    number of ulps, which says nothing about the arithmetic: an FMA moves a
    value by a few ulps of its own magnitude, and a whole-array comparison is
    the meaningful one (a shape or algebra error is millions of ulps away).
    A complex array contributes its REAL part here; its imaginary part is
    covered by the complex-valued floor comparison.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        a, b = np.real(a), np.real(b)
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.size == 0:
        return 0.0
    scale = float(np.spacing(max(np.abs(b).max(), np.finfo(float).tiny)))
    return float(np.abs(a - b).max() / scale)


def _replicate(data, rs):
    out = {k: np.repeat(np.asarray(v), rs, axis=0) for k, v in data.items()}
    out["weight"] = np.full(len(np.asarray(data["weight"])) * rs, 1.0 / rs)
    return out


def _setup(cfg, ne=64, seed=0):
    """``(fitter, kernel_config, params, data)`` on one fixture + sample."""
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    return f, kc, params, _in_grid(kc, ne, seed)


def _oracle(kc, params, data, rs, norm):
    """The fp64 reference's ``(Q, grads, P, dnorm)`` on the same sample."""
    ref = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=rs)
    try:
        Q, g, P = ref.compute(params, ref.load_data(data), norm=norm)
        return Q, g, P, ref._last_dnorm
    finally:
        ref.free()


def _assert_at_the_floor(Qg, gg, Pg, dnorm_g, ref, label="",
                         P_bound=CACHE_FLOOR_P):
    """Every quantity against the oracle's ``ref``, at the recorded floor."""
    Qr, gr, Pr, dnorm_r = ref
    assert _rel(Pg, Pr) < P_bound, f"{label} P"
    assert _rel(Qg, Qr) < CACHE_FLOOR, f"{label} Q"
    assert _rel(gg["ck"], gr["ck"]) < CACHE_FLOOR, f"{label} dck"
    assert _rel(gg["coeff"], gr["coeff"]) < CACHE_FLOOR_COEFF, f"{label} dcoeff"
    if dnorm_r is not None:
        assert _rel(dnorm_g, dnorm_r) < CACHE_FLOOR, f"{label} dnorm"


# ── the wide shape (beyond the 48 KiB shared-memory opt-in) ──────────────

_WIDE = {}


def _wide_kc(n_wave_target):
    """``(fitter, kc)`` for a shape of ``n_wave_target`` waves and 2 projections.

    Above ~108 waves the shared tile needs the device's opt-in, and past the
    opt-in itself nothing can be tiled at all — the two ends this file gates
    (``test_the_widest_shape_computes_and_matches_the_reference`` and
    ``test_an_over_large_tile_is_refused``).
    """
    from benchmark_flat_vs_pwa import tile_flat
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = dict(tile_flat(f.kernel_config, n_wave_target, 2))
    kc["num_table"] = np.ones_like(np.asarray(kc["num_table"]))
    return f, kc


def _wide_case():
    """``(kc, params, data, oracle)`` on the 500-wave shape.

    Its 500 waves put the working set above the 48 KiB opt-in, so a launch that
    silently returned 0 (a rejected shared-memory request nobody checked) is
    exactly what this shape catches.  Built once: the reference and the config
    are read-only, and every kernel in the table is compared against the same
    numbers.
    """
    if "case" not in _WIDE:
        f, kc = _wide_kc(500)
        n_wave = int(np.asarray(kc["fl_table"]).shape[0])
        n_basis = int(np.asarray(kc["basis_table"]).shape[0])
        p0, _ = f.build_params(f.initial_values(seed=1))
        ck0 = np.asarray(p0["ck"])
        params = {"ck": np.tile(ck0, (n_wave + ck0.size - 1) // ck0.size)[:n_wave],
                  "coeff": np.random.default_rng(7).uniform(0.3, 1.2, n_basis)}
        data = _in_grid(kc, 64, 3)
        _WIDE["case"] = (kc, params, data,
                         _oracle(kc, params, data, 1, 1024.0))
    return _WIDE["case"]


# ── the budget-binding shape: the `pick_te` regression ──────────────────

_BUDGET = {}


def _budget_kc():
    """A shape whose 8-event tile does NOT fit the shared-memory budget.

    ``tile_flat`` gives n_wave 80 / n_proj 2; 81 basis rows make the layout bind
    below 8 events, which is the exact shape where the budget loop used to land
    on a non-multiple of ``rsize`` and ``create`` rejected it.
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


def _budget_case(rs):
    """``(kc, params, data, oracle)`` on the budget-binding shape."""
    if "case" not in _BUDGET:
        f = Fitter(CFG_KM)
        f.apply_constrains()
        kc = _budget_kc()
        n_wave = int(np.asarray(kc["fl_table"]).shape[0])
        n_basis = int(np.asarray(kc["basis_table"]).shape[0])
        p0, _ = f.build_params(f.initial_values(seed=1))
        ck0 = np.asarray(p0["ck"])
        params = {"ck": np.tile(ck0, (n_wave + ck0.size - 1) // ck0.size)[:n_wave],
                  "coeff": np.random.default_rng(7).uniform(0.3, 1.2, n_basis)}
        _BUDGET["case"] = (kc, params, _in_grid(kc, 64, 4), {})
    kc, params, data, refs = _BUDGET["case"]
    if rs not in refs:
        refs[rs] = _oracle(kc, params, data, rs, 1.3)
    return kc, params, data, refs[rs]


# ── the production shape: the one place a row is actually split ─────────

_PROD = {}


def _prod_case():
    """``(kc, params, data)`` on the 26-wave production flat config."""
    if "case" not in _PROD:
        if not os.path.exists(CFG_PROD):
            pytest.skip(f"production config not present: {CFG_PROD}")
        d = dict(Fitter(CFG_PROD).config.dic)
        d["amp_model"] = "pwa_gen"
        d["backend"] = {"name": "numpy_flat"}
        f = Fitter(d)
        f.apply_constrains()
        kc = f.kernel_config
        params, _ = f.build_params(f.initial_values(seed=1))
        _PROD["case"] = (kc, params, _in_grid(kc, 200, 0))
    return _PROD["case"]


# ── the kernels, and what genuinely differs between them ────────────────

class Kernel:
    """One case-1 copy: its registry name, the copy it must reproduce, and its
    own config-level gates.

    ``parent``      the copy whose numbers it must reproduce (None = the family
                    parent, whose oracle is ``numpy_flat`` itself)
    ``gates``       ``[(id, fn(kernel)), …]`` — the bespoke config gates
    """

    __slots__ = ("name", "short", "parent", "gates")

    def __init__(self, name, parent=None, gates=()):
        self.name = name
        self.short = name.removeprefix("cuda_flat_pwa_gen_")
        self.parent = parent
        self.gates = list(gates)

    def __repr__(self):
        return self.short


#: s0e2's ONLY numerical change is the summation order of the sparse dot (one
#: packed 64-bit CSR load, two independent fp64 chains) — but the GLOBAL Q and
#: dnorm are ``atomicAdd`` sums across blocks, so the two kernels cannot be
#: compared tighter than the launch-order noise of ONE kernel.  MEASURED on
#: this gate's fixture (CFG_KM, rsize 1, norm 1.3, 25 launches each):
#:     s0e2 vs s0e2 (SELF): Q up to 6.4e-16 (11/25 non-zero), dck 2.2e-16,
#:                          dcoeff 3.5e-17, dnorm 3.8e-16, P 0.0
#:     s0e2 vs s0e   (gate): Q up to 4.8e-16, P 8.7e-16, dck 2.0e-16,
#:                          dcoeff 4.2e-16, dnorm 5.6e-16
#: and a full-suite run hit Q = 1.13e-15 (2.5e-14 absolute), i.e. the old
#: 1e-15 bound sat INSIDE the launch noise and failed on a correct pair.
#: dcoeff is a SUM over (e,k) of ``2·(pv_r·Z_i + pv_i·Z_r)``: the terms are
#: per-event contributions and the result is their (partly cancelling) sum, so
#: the same 1-ulp shift is amplified when measured RELATIVE to a small result —
#: 2.8e-15 worst over CFG_ONE/CFG_KM x rsize 1,17 x norm None,1.3 (absolute
#: 4.7e-13 on max|dcoeff| = 0.17).  1e-14 matches s0e3/s0e4 and is still ~10
#: orders below the fp32 cache floor (1e-4), so it cannot hide an algebra
#: change; it only stops the atomics from tripping a bound that never held.
S0E2_TOL = 1e-14

#: s0e3 vs s0e: the fp32 seed + 2 Newton steps differ from the exact fp64
#: divide by 1-2 ulp of ``1/|D|²``, so the quantities are no longer bit-equal.
#: MEASURED (ne=200, rsize 1/17, norm None/1.3, CFG_ONE + CFG_KM, worst over all
#: eight combinations, relative to s0e): Q 7.2e-16, P 4.0e-16, dnorm 8.3e-16,
#: dck 3.4e-16, dcoeff 5.5e-16 — i.e. <= 8.3e-16 (~4 ulp), so 1e-14 is a 12x
#: margin and still ~10 orders below the cache floor.
S0E3_TOL = 1e-14

#: s0e4 vs s0e: the pre-gathered `cw` holds the SAME doubles and a split row sums
#: the same products in two halves, so the gap is a few ulp of the accumulated
#: value.  MEASURED (ne=200, rsize 1/17, norm None/1.3, CFG_ONE + CFG_KM):
#: Q 4.9e-16, P 0.0, dnorm 3.8e-16, dck 2.0e-16, dcoeff 3.6e-16; and on the
#: PRODUCTION config (where the 9/10-nnz rows ARE split; rs 1/17, norm 1.3,
#: split_nnz 0/3/6/9/130): Q 5.9e-16, P 4.0e-15, dnorm 9.8e-16, dck 8.2e-16,
#: dcoeff 3.4e-15 — 1e-14 is a 2.5x margin and still ~10 orders below the floor.
S0E4_TOL = 1e-14


def _compare_to_parent(a, b, params, data, norm, tol, tol_coeff, label=""):
    """Every quantity of ``b`` to ``a``'s value: ``a`` the parent, ``b`` the copy."""
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
    assert abs(Qa - Qb) / max(abs(Qa), 1e-300) < tol, f"{label} Q"
    assert _rel(Pb, Pa) < tol, f"{label} P"
    assert _rel(gb["ck"], ga["ck"]) < tol, f"{label} dck"
    assert _rel(gb["coeff"], ga["coeff"]) < tol_coeff, f"{label} dcoeff"
    if norm is not None:
        assert (abs(gb["norm"] - ga["norm"])
                / max(abs(ga["norm"]), 1e-300)) < tol, f"{label} dnorm"


# ── gate 8: the parent digest, one per copy ─────────────────────────────

def parent_gate_s0e2(k):
    """s0e2 == s0e, quantity by quantity, on the SAME handle shape.

    ``rsize`` 1 and 17 cover the plain and the grouped objective; ``norm=None``
    and ``1.3`` cover the two ``dQ/dP`` forms.  Both kernels read the same fp32
    caches, so any difference beyond the one re-association would be an algebra
    change — which is exactly what this gate forbids.
    """
    for cfg in (CFG_ONE, CFG_KM):
        f = Fitter(cfg)
        f.apply_constrains()
        kc = f.kernel_config
        params, _ = f.build_params(f.initial_values(seed=1))
        for rs in (1, 17):
            data = (_in_grid(kc, 64, seed=2) if rs == 1
                    else _replicate(_in_grid(kc, 17, seed=5), rs))
            for norm in (None, 1.3):
                a = create_backend(k.parent, kc, model="pwa_gen",
                                   resolution_size=rs)
                b = create_backend(k.name, kc, model="pwa_gen",
                                   resolution_size=rs)
                _compare_to_parent(a, b, params, data, norm,
                                   S0E2_TOL, S0E2_TOL,
                                   f"{cfg} rs={rs} norm={norm}")


def parent_gate_s0e3(k):
    """s0e3 == s0e to <= ``S0E3_TOL``: the reciprocal is the ONLY change.

    Every real step is still fp64 and the caches are still float2, so the only
    gap is the 1-2 ulp of the Newton-refined reciprocal — no longer bit-equal
    (which is why this gate is a tolerance, not ``==``), but ten orders below
    the fp32 cache floor, so it cannot hide a real regression.
    """
    for cfg in (CFG_ONE, CFG_KM):
        f = Fitter(cfg)
        f.apply_constrains()
        kc = f.kernel_config
        params, _ = f.build_params(f.initial_values(seed=1))
        data = _in_grid(kc, 200, seed=0)
        for rs in (1, 17):
            for norm in (None, 1.3):
                a = create_backend(k.parent, kc, model="pwa_gen",
                                   resolution_size=rs)
                b = create_backend(k.name, kc, model="pwa_gen",
                                   resolution_size=rs)
                _compare_to_parent(a, b, params, data, norm,
                                   S0E3_TOL, S0E3_TOL, f"{cfg} rs={rs} norm={norm}")


def parent_gate_s0e4(k):
    """s0e4 == s0e to <= ``S0E4_TOL``: only the summation order moves.

    ``cw`` holds the same ``double`` values as ``coeff[nz_m[i]]``, and a split
    row sums the same products in two halves, so nothing here can differ beyond
    a few ulp — ten orders below the fp32 cache floor.  ``P`` is BIT-equal on
    the fixtures because their rows are 5 non-zeros: nothing is split there.
    """
    for cfg in (CFG_ONE, CFG_KM):
        f = Fitter(cfg)
        f.apply_constrains()
        kc = f.kernel_config
        params, _ = f.build_params(f.initial_values(seed=1))
        data = _in_grid(kc, 200, seed=0)
        for rs in (1, 17):
            for norm in (None, 1.3):
                a = create_backend(k.parent, kc, model="pwa_gen",
                                   resolution_size=rs)
                b = create_backend(k.name, kc, model="pwa_gen",
                                   resolution_size=rs)
                _compare_to_parent(a, b, params, data, norm,
                                   S0E4_TOL, S0E4_TOL, f"{cfg} rs={rs} norm={norm}")


def parent_gate_s0e5(k):
    """s0e5 vs its parent: the parent's floor, and <= 64 ulp per entry.

    FMA re-association and the deduped denominator move a value by <= 1 ulp, so
    the parent's own measured floor is the honest family bound (``FLOORS`` is
    READ rather than restated, so a change to the family floor cannot leave this
    gate behind), and the variant's own claim is tighter still: every entry of
    every quantity within 64 ulp OF THE OBSERVABLE.  A shape or algebra error is
    millions of ulps away, so 64 ulp gates the change the variant actually makes
    (a rounding step or two) without pinning the exact bit pattern.
    """
    rel, p, coeff, abs_ok, _cfgs = FLOORS[k.parent]
    for cfg in QUALIFYING:
        f = Fitter(cfg)
        f.apply_constrains()
        kc = f.kernel_config
        params, _ = f.build_params(f.initial_values(seed=1))
        base = _in_grid(kc, 64, 0)
        for rs in (1, 17):
            data = base if rs == 1 else _replicate(base, rs)
            for norm in (None, 1.3):
                ref = create_backend(k.parent, kc, model="pwa_gen",
                                     resolution_size=rs)
                be = create_backend(k.name, kc, model="pwa_gen",
                                    resolution_size=rs)
                try:
                    Qr, gr, Pr = ref.compute(params, ref.load_data(data),
                                             norm=norm)
                    Qc, gc, Pc = be.compute(params, be.load_data(data),
                                            norm=norm)
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
                        assert _rel(be.kernel._last_dnorm,
                                    ref.kernel._last_dnorm) < rel
                finally:
                    ref.free()
                    be.free()
    # the tighter, per-entry form: the three fixtures this variant's own report
    # was measured on, both tilings and both objectives
    for cfg in QUALIFYING[:3]:
        f = Fitter(cfg)
        f.apply_constrains()
        kc = f.kernel_config
        params, _ = f.build_params(f.initial_values(seed=1))
        base = _in_grid(kc, 64, 0)
        for rs in (1, 17):
            data = base if rs == 1 else _replicate(base, rs)
            ref = create_backend(k.parent, kc, model="pwa_gen",
                                 resolution_size=rs)
            be = create_backend(k.name, kc, model="pwa_gen",
                                resolution_size=rs)
            try:
                for norm in (None, 1.3):
                    Qr, gr, Pr = ref.compute(params, ref.load_data(data),
                                             norm=norm)
                    Qc, gc, Pc = be.compute(params, be.load_data(data),
                                            norm=norm)
                    for label, a, b in (
                            ("Q", np.atleast_1d(Qc), np.atleast_1d(Qr)),
                            ("P", Pc, Pr), ("dck", gc["ck"], gr["ck"]),
                            ("dcoeff", gc["coeff"], gr["coeff"])):
                        assert _ulps(a, b) <= 64.0, (
                            f"{cfg} rs={rs} norm={norm} {label}: "
                            f"{_ulps(a, b):.1f} ulps of the observable")
                    if norm is not None:
                        assert _ulps(be.kernel._last_dnorm,
                                     ref.kernel._last_dnorm) <= 64.0
            finally:
                ref.free()
                be.free()


# ── gate 9: a schedule knob is arithmetically invisible ─────────────────

def config_gate_s0e4_split_nnz(k):
    """``split_nnz`` re-segments at CREATE time and never moves a number.

    The fixture rows (5 non-zeros) sit below every threshold, so the knob is
    swept on them AND on the 26-wave production shape, where the 9/10-non-zero
    rows are genuinely split into two halves summed by two threads.  The parent
    (``s0e``, which has no segmentation at all) is the reference; the assertion
    is numeric only.
    """
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    fixture = _in_grid(kc, 200, 0)
    _pkc, pparams, pdata = _prod_case()
    # the fixture rows are all 5 non-zeros, so both objectives are swept there
    # (a threshold of 3 splits them); on the production shape only the
    # normalised objective is — that is the shape the split was calibrated on
    # (MEASURED dcoeff 3.4e-15 there, and 1.1e-14 for the raw un-normalised sum,
    # whose per-event gradient is an order larger).
    for label, cfg_kc, cfg_params, data, thresholds, norms in (
            (CFG_KM, kc, params, fixture, (0, 3, 6, 9), (None, 1.3)),
            (CFG_PROD, _pkc, pparams, pdata, (0, 3, 6, 9, 130), (1.3,))):
        for rs in (1, 17):
            for norm in norms:
                base = create_backend(k.parent, cfg_kc, model="pwa_gen",
                                      resolution_size=rs)
                hb = base.load_data(data)
                try:
                    Q0, g0, P0 = base.compute(cfg_params, hb, norm=norm)
                finally:
                    hb.free()
                    base.free()
                for T in thresholds:
                    be = create_backend(k.name, cfg_kc, model="pwa_gen",
                                        resolution_size=rs, split_nnz=T)
                    try:
                        Q, g, P = be.compute(cfg_params, be.load_data(data),
                                             norm=norm)
                    finally:
                        be.free()
                    tag = f"{label} rs={rs} norm={norm} split_nnz={T}"
                    assert abs(Q - Q0) / max(abs(Q0), 1e-300) < S0E4_TOL, tag
                    assert _rel(P, P0) < S0E4_TOL, tag
                    assert _rel(g["ck"], g0["ck"]) < S0E4_TOL, tag
                    assert _rel(g["coeff"], g0["coeff"]) < S0E4_TOL, tag
                    if norm is not None:
                        assert (abs(g["norm"] - g0["norm"])
                                / max(abs(g0["norm"]), 1e-300)) < S0E4_TOL, tag


# ── the table ───────────────────────────────────────────────────────────

KERNELS = [
    Kernel("cuda_flat_pwa_gen_s0e"),
    Kernel("cuda_flat_pwa_gen_s0e2", parent="cuda_flat_pwa_gen_s0e",
           gates=[("s0e_parity", parent_gate_s0e2)]),
    Kernel("cuda_flat_pwa_gen_s0e3", parent="cuda_flat_pwa_gen_s0e",
           gates=[("s0e_parity", parent_gate_s0e3)]),
    Kernel("cuda_flat_pwa_gen_s0e4", parent="cuda_flat_pwa_gen_s0e",
           gates=[("s0e_parity", parent_gate_s0e4),
                  ("split_nnz", config_gate_s0e4_split_nnz)]),
    Kernel("cuda_flat_pwa_gen_s0e5", parent="cuda_flat_pwa_gen_s0e2",
           gates=[("s0e2_floor_parity", parent_gate_s0e5)]),
]
ID = [k.short for k in KERNELS]
BESPOKE = [(k, gid, fn) for k in KERNELS for gid, fn in k.gates]

#: the table above is hand-written, so it is checked against the OWNERSHIP rule
#: in ``_flat_compute`` — a leaf registered later is in neither, and fails here
#: rather than being silently uncovered by every config-contract file.
_ME = os.path.basename(__file__)
_OWNED = {n for n, f in CONFIG_OWNER.items() if f == _ME}
_TABLE = {k.name for k in KERNELS}
assert _TABLE == _OWNED, (
    f"{_ME}'s KERNELS is out of sync with _flat_compute.CONFIG_OWNER: "
    f"missing {sorted(_OWNED - _TABLE)}, stale {sorted(_TABLE - _OWNED)}")
assert set(CONFIG_OWNER) == set(LEAVES) | set(
    backends_for_model("pwa_gen", tag="coeff_pinned"))


# ── gate 0: which config may name the kernel ────────────────────────────

@pytest.mark.parametrize("k", KERNELS, ids=ID)
def test_the_model_scope_is_pwa_gen_only(k):
    """The family reads the FLAT config, so ``pwa_gen`` is its only model.

    A registration under a second model would make a flat kernel selectable for
    the plain-``pwa`` config, whose kernel config carries none of the keys this
    one reads.
    """
    assert k.name in set(backends_for_model("pwa_gen"))
    assert k.name not in set(backends_for_model("pwa"))


# ── gate 1: the numeric contract ────────────────────────────────────────

@pytest.mark.parametrize("k", KERNELS, ids=ID)
@pytest.mark.parametrize("cfg", QUALIFYING, ids=CFG_ID)
@pytest.mark.parametrize("norm", [None, 1.3])
@pytest.mark.parametrize("rs", [1, 17])
def test_numeric_contract(k, cfg, norm, rs):
    """``Q``/``P``/``grads`` against the fp64 oracle, at the recorded floor.

    Every qualifying fixture, both objectives (``norm=None`` is the raw sum,
    ``1.3`` the normalised NLL) and both the per-event and the grouped
    (``rsize = 17``) form.  Creating the backend by NAME is also what pins the
    registration: an unregistered kernel cannot be built at all.
    """
    _f, kc, params, base = _setup(cfg, ne=64, seed=0)
    data = _replicate(base, rs) if rs > 1 else base
    ref = _oracle(kc, params, data, rs, norm)
    be = create_backend(k.name, kc, model="pwa_gen", resolution_size=rs)
    try:
        Qg, gg, Pg = be.compute(params, be.load_data(data), norm=norm)
        dnorm = be.kernel._last_dnorm
    finally:
        be.free()
    _assert_at_the_floor(Qg, gg, Pg, dnorm, ref,
                         f"{cfg} rs={rs} norm={norm}")


# ── gate 2: the grouped objective over the whole rsize range ────────────

#: ``rs = 1`` and ``rs = 17`` are the numeric contract's sweep above; these are
#: the ones that pin the rest of the range — groups smaller than, equal to and
#: much larger than any tile the family picks (path A and path B are one
#: contract here: "a group closes to ONE log", whatever the tiling does).
RSIZES = (2, 4, 8, 10, 25, 40, 100)


@pytest.mark.parametrize("k", KERNELS, ids=ID)
@pytest.mark.parametrize("rs", RSIZES)
def test_grouped_objective_over_every_resolution_size(k, rs):
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    data = _replicate(_in_grid(kc, 32, seed=7), rs)
    ref = _oracle(kc, params, data, rs, 1.3)
    be = create_backend(k.name, kc, model="pwa_gen", resolution_size=rs)
    try:
        Qg, gg, Pg = be.compute(params, be.load_data(data), norm=1.3)
        dnorm = be.kernel._last_dnorm
    finally:
        be.free()
    _assert_at_the_floor(Qg, gg, Pg, dnorm, ref, f"rs={rs}")


# ── gate 3: one log per group, whatever the tiling ──────────────────────

@pytest.mark.parametrize("k", KERNELS, ids=ID)
def test_every_group_closes_to_one_log_at_a_large_rsize(k):
    """``rsize`` larger than the tiling, and a chunk holding many groups.

    Both are real: the production fit runs ``rsize = 17`` on a shape the family
    tiles 8 events at a time, and a 5000-event chunk then holds 294 groups.  The
    objective must still take ONE log per group, which is what comparing against
    ``numpy_flat`` at the same ``rsize`` pins — a group closed twice, or split
    across two tiles or two chunks, changes Q, not just its rounding.
    """
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    for rs in (17, 100):
        one = _replicate(_in_grid(kc, rs, seed=11), rs)       # ONE group
        data = {kk: np.concatenate([v, v, v]) for kk, v in one.items()}
        ref = _oracle(kc, params, data, rs, 1.3)
        be = create_backend(k.name, kc, model="pwa_gen", resolution_size=rs,
                            batch_size=5000)
        try:
            Qg, gg, _Pg = be.compute(params, be.load_data(data), norm=1.3)
            assert _rel(Qg, ref[0]) < CACHE_FLOOR, f"rsize={rs} Q"
            assert _rel(gg["coeff"], ref[1]["coeff"]) < CACHE_FLOOR_COEFF
        finally:
            be.free()


# ── gate 4/5: the two numeric edges of the contract ─────────────────────

@pytest.mark.parametrize("k", KERNELS, ids=ID)
def test_an_empty_sample_is_the_identity(k):
    """A zero-row load is legal: ``Q = 0`` with zero gradients, like the oracle.

    This is the degenerate end of the same contract, not an internal: the
    Fitter can hand a backend fewer events than expected (a selection that came
    back empty), and a device trap there would kill a fit for no reason.
    """
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    empty = {kk: np.asarray(v)[:0] for kk, v in _in_grid(kc, 4).items()}
    ref = _oracle(kc, params, empty, 1, 1.3)
    be = create_backend(k.name, kc, model="pwa_gen")
    try:
        Qg, gg, Pg = be.compute(params, be.load_data(empty), norm=1.3)
    finally:
        be.free()
    assert _rel(Qg, ref[0]) < CACHE_FLOOR
    assert np.asarray(Pg).size == np.asarray(ref[2]).size == 0
    assert np.allclose(gg["coeff"], ref[1]["coeff"])
    assert np.all(np.isfinite(gg["ck"]))
    assert np.all(np.isfinite(gg["coeff"]))


@pytest.mark.parametrize("k", KERNELS, ids=ID)
def test_return_p_false_does_not_change_the_objective(k):
    """``return_p=False`` skips the per-event output, not the arithmetic."""
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    be = create_backend(k.name, kc, model="pwa_gen")
    h = be.load_data(_in_grid(kc, 12))
    try:
        Qa, ga, Pa = be.compute(params, h, norm=1.3, return_p=False)
        Qb, gb, Pb = be.compute(params, h, norm=1.3, return_p=True)
        assert Pa is None and np.asarray(Pb).size == 12
        assert Qa == Qb
        assert _rel(ga["ck"], gb["ck"]) == 0.0
        assert _rel(ga["coeff"], gb["coeff"]) == 0.0
    finally:
        h.free()
        be.free()


# ── gate 6: refusal is part of the contract ─────────────────────────────

@pytest.mark.parametrize("k", KERNELS, ids=ID)
def test_a_multi_bw_chain_is_refused_naming_the_general_backend(k):
    """``n_res != 1`` is refused with the backend that CAN take it named."""
    f = Fitter(CFG_MB)
    f.apply_constrains()
    assert int(np.asarray(f.kernel_config["num_table"]).shape[1]) == 2
    with pytest.raises(ValueError, match="cuda_flat_pwa_gen2"):
        create_backend(k.name, f.kernel_config, model="pwa_gen")


@pytest.mark.parametrize("k", KERNELS, ids=ID)
def test_resolution_size_below_one_is_refused(k):
    f = Fitter(CFG_KM)
    f.apply_constrains()
    with pytest.raises(ValueError, match="resolution_size"):
        create_backend(k.name, f.kernel_config, resolution_size=0,
                       model="pwa_gen")


#: 1400 waves: the family's minimal (1-event) tile is 145 792 B, past this
#: box's 101 376 B opt-in, so NO tiling exists and ``create`` must raise.  470
#: waves is 49 072 B and does fit — that shape is the crossing case above, not
#: this one (MEASURED at te 1 on both).
OVER_LARGE_WAVES = 1400


@pytest.mark.parametrize("k", KERNELS, ids=ID)
def test_an_over_large_tile_is_refused_not_silently_empty(k):
    """A shape no tile fits is REFUSED — never launched with Q = 0 or P = 0.

    The failure this catches is a create that swallows the shared-memory
    rejection and returns a context whose launches all fail: every small fixture
    passes and a real wave count silently produces nothing.  It is the
    counterpart of ``test_the_widest_shape_computes_and_matches_the_reference``
    (which must COMPUTE at 500 waves), and both are stated as behaviour, not as
    a tile size.
    """
    _f, kc = _wide_kc(OVER_LARGE_WAVES)
    assert int(np.asarray(kc["fl_table"]).shape[0]) >= OVER_LARGE_WAVES
    with pytest.raises((RuntimeError, ValueError)):
        create_backend(k.name, kc, model="pwa_gen")


# ── the constructor knobs, which are public API ─────────────────────────
#
# `tile_events=`/`te=` shadows the tile the kernel would pick, and `split_nnz`
# re-segments a CSR row.  Both are DEBUG/BENCHMARK levers a caller can pass, so
# a value that would break the objective (a tile splitting a log-sum group, a
# negative threshold) must be refused up front — not silently floored into a
# different computation.
#
# `s0e`, `s0e2` and `s0e5` leave the tile check to the C create guard (the
# refusal surfaces as RuntimeError from `create`); `s0e3`/`s0e4` validate in
# Python, so their message is pinned by keyword.
REFUSED_KNOBS = [
    # s0e's spelling of the tile hint; 23 is not a multiple of rsize 17
    ("cuda_flat_pwa_gen_s0e", dict(resolution_size=17, tile_events=23), None),
    ("cuda_flat_pwa_gen_s0e2", dict(resolution_size=17, te=20), None),
    ("cuda_flat_pwa_gen_s0e3", dict(resolution_size=17, te=20), "multiple"),
    ("cuda_flat_pwa_gen_s0e3", dict(te=-1), "positive"),
    ("cuda_flat_pwa_gen_s0e4", dict(resolution_size=17, te=20), "multiple"),
    ("cuda_flat_pwa_gen_s0e4", dict(split_nnz=-1), "split_nnz"),
    ("cuda_flat_pwa_gen_s0e5", dict(resolution_size=17, te=20), None),
]
REFUSED_IDS = [f"{n.split('_')[-1]}-{'_'.join(k)}" for n, k, _m in REFUSED_KNOBS]


@pytest.mark.parametrize("name,kw,match", REFUSED_KNOBS, ids=REFUSED_IDS)
def test_a_constructor_knob_that_would_break_the_objective_is_refused(
        name, kw, match):
    f = Fitter(CFG_KM)
    f.apply_constrains()
    with pytest.raises((RuntimeError, ValueError)) as ei:
        create_backend(name, f.kernel_config, model="pwa_gen", **kw)
    if match is not None:
        assert match in str(ei.value), str(ei.value)
    else:
        # the C guard refuses at create; the wrapper must surface it rather
        # than return a context that cannot launch
        assert "context creation failed" in str(ei.value), str(ei.value)


#: a legal hint: two whole groups at `rsize` 17, well above the family's own
#: choice (8).  The tile size only re-associates the sums.
LEGAL_TE = 34


@pytest.mark.parametrize("k", KERNELS, ids=ID)
@pytest.mark.parametrize("norm", [None, 1.3])
def test_a_legal_tile_hint_is_arithmetically_invisible(k, norm):
    """A LEGAL tile hint changes the tiling only, never a number.

    MEASURED over the five kernels x both ``dQ/dP`` forms: worst relative gap
    5.5e-16 (``dcoeff``), i.e. a handful of ulp — ten orders below the fp32
    cache floor, so a real algebra change cannot hide under it.

    What this does NOT establish is that the hint reached the kernel: ``te`` is
    a tiling lever, not a contract, so the value it lands on is not asserted
    here (and an ignored hint would compute the same numbers, which is the
    harmless direction).  That an invalid hint is REFUSED is the contract, and
    that is pinned in ``test_construction_refuses_an_illegal_knob``.
    """
    f = Fitter(CFG_KM)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    data = _in_grid(kc, 200, seed=0)
    knob = "tile_events" if k.short == "s0e" else "te"
    hinted = create_backend(k.name, kc, model="pwa_gen", resolution_size=17,
                            **{knob: LEGAL_TE})
    auto = create_backend(k.name, kc, model="pwa_gen", resolution_size=17,
                          **{knob: 0})
    try:
        Qh, gh, Ph = hinted.compute(params, hinted.load_data(data), norm=norm)
        Qa, ga, Pa = auto.compute(params, auto.load_data(data), norm=norm)
        assert abs(Qh - Qa) / max(abs(Qa), 1e-300) < S0E3_TOL
        assert _rel(Ph, Pa) < S0E3_TOL
        assert _rel(gh["ck"], ga["ck"]) < S0E3_TOL
        assert _rel(gh["coeff"], ga["coeff"]) < S0E3_TOL
        if norm is not None:
            assert _rel(gh["norm"], ga["norm"]) < S0E3_TOL
    finally:
        hinted.free()
        auto.free()


# ── gate 7: a legal config CONSTRUCTS and agrees ────────────────────────

@pytest.mark.parametrize("k", KERNELS, ids=ID)
def test_the_widest_shape_computes_and_matches_the_reference(k):
    """A shape whose tile is above the 48 KiB opt-in must compute.

    The failure this catches is the one a rejected shared-memory request used
    to cause: a launch that never ran and a caller that saw ``Q = 0``.  The
    assertion is the CONTRACT — it computes and matches the oracle — not the
    tile size the implementation happens to pick.
    """
    kc, params, data, ref = _wide_case()
    be = create_backend(k.name, kc, model="pwa_gen")
    try:
        Qg, gg, Pg = be.compute(params, be.load_data(data), norm=1024.0)
        dnorm = be.kernel._last_dnorm
    finally:
        be.free()
    assert np.isfinite(Qg) and Qg > 0.0
    assert np.all(np.asarray(Pg) >= 0.0)
    _assert_at_the_floor(Qg, gg, Pg, dnorm, ref, "wide n_wave=500",
                         P_bound=CACHE_FLOOR)


@pytest.mark.parametrize("k", KERNELS, ids=ID)
@pytest.mark.parametrize("rs", [2, 4, 7])
def test_a_wide_shape_at_rsize_2_4_7_constructs_and_agrees(k, rs):
    """`rsize` 2..7 on a shape that binds the tile budget used to be illegal.

    The budget loop shrank the 8-event tile ONE event at a time and could return
    a non-multiple of ``rsize`` for a WIDE shape; ``create`` then rejected it and
    the caller saw only an internal "context creation failed" for a legal
    ``resolution_size`` it could neither see nor control.  The regression gate is
    that the config CONSTRUCTS and that its numbers are the oracle's.
    """
    kc, params, data, ref = _budget_case(rs)
    be = create_backend(k.name, kc, model="pwa_gen", resolution_size=rs)
    try:
        Qg, gg, Pg = be.compute(params, be.load_data(data), norm=1.3)
        dnorm = be.kernel._last_dnorm
    finally:
        be.free()
    assert np.isfinite(Qg) and np.all(np.isfinite(np.asarray(Pg)))
    _assert_at_the_floor(Qg, gg, Pg, dnorm, ref, f"budget rs={rs}")


# ── gate 8/9: the bespoke config gates, one callable per table entry ────

@pytest.mark.parametrize("k,gate_id,gate", BESPOKE,
                         ids=[f"{k.short}-{gid}" for k, gid, _fn in BESPOKE])
def test_bespoke_gate(k, gate_id, gate):
    """Each copy's own claim: the parent digest, or its schedule knob.

    A gate only one kernel has is a TABLE ENTRY for that kernel rather than a
    file of its own, so this stays one file and the entry is still named
    (``[s0e4-split_nnz]``).
    """
    gate(k)
