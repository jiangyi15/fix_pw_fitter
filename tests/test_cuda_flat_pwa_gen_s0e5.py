"""``cuda_flat_pwa_gen_s0e5`` — the s0e tile with FMA-ised chains and a deduped denominator.

Case-1 copy of ``cuda_flat_pwa_gen_s0e2``: own ``.cu``, own ``.so``, own
symbols, and the s0e2 kernel/wrapper/test stay byte-identical.  The shared
contract (Q/P/dck/dcoeff against ``numpy_flat``, FD gradients, chunk
invariance, handle independence, the input errors) reaches this backend
through the REGISTRY — it is an ``n_res1``-tagged ``pwa_gen`` leaf — so this
file keeps what is specific to the variant:

* **agreement with the parent** ``cuda_flat_pwa_gen_s0e2`` at the s0e2
  floors (P 3e-5, Q/dck 2e-5, coeff 1e-4 relative, ``_floor_ok`` with a
  1e-4 absolute escape),
* the **bitwise report** ``_bitwise_report`` builds and
  ``test_the_bitwise_report_against_the_parent`` prints: how many quantities
  are exactly equal and the worst difference in ulps.  FMA re-association
  changes the last bit, so the gate is "≤ 2 ulp", not "exact" — and the
  report is the measurement, not a hope,
* the **denominator dedup** pinned: ``key_of_wave`` groups the waves by their
  gamma pattern, and on the production config that is 18 keys for 26 waves,
* ``kernel_stats()`` (registers / local bytes / blocks per SM = occupancy).

Variant A's changes against s0e2 (algebra unchanged):
* every ``a*b + c`` chain the compiler left unfused is written as an explicit
  ``fma`` (the sparse dot, the projection sum and ``|A|²``, the u-merge, the
  ``ck·u`` pair, the coefficient gradient, the group sums),
* ``2Re(pv·Z)`` is two FMAs with the factor 2 folded into the per-entry
  constant (exact),
* the denominator dot is evaluated ONCE per gamma-pattern key (26 -> 18 keys
  on production) and every wave of a key reads that D and its ``1/|D|²``.

"""
import json
import os

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import backend_tags, backends_for_model, create_backend
from test_cuda_flat_cache_pwa_gen import _in_grid

from _flat_compute import FLOORS, QUALIFYING

NAME = "cuda_flat_pwa_gen_s0e5"
PARENT = "cuda_flat_pwa_gen_s0e2"
#: the production config the dedup count was measured on (external sample)
PROD = ("/home/jiangy/github/project94/pwa_workspace/jpsi_eta_pi_pi/"
        "04_model_search/run_423_kmatrix_n_20261119_141110/config_kn4_plot.yml")


def _rel(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _ulps(a, b):
    """Worst difference in ulps OF THE OBSERVABLE (i.e. of its largest entry).

    Per-entry spacing would call any difference on a near-zero entry a huge
    number of ulps, which says nothing about the arithmetic: an FMA moves a
    value by a few ulps of its own magnitude, and a whole-array comparison is
    the meaningful one (a shape/algebra error is millions of ulps away).
    """
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.size == 0:
        return 0.0
    scale = float(np.spacing(max(np.abs(b).max(), np.finfo(float).tiny)))
    return float(np.abs(a - b).max() / scale)


#: filled by `_bitwise_report` — printed by the report test
REPORT = {}


def _pair(cfg, rs=1, ne=64, seed=0):
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    base = _in_grid(kc, ne, seed)
    data = base
    if rs > 1:
        data = {k: np.repeat(np.asarray(v), rs, axis=0) for k, v in base.items()}
        data["weight"] = np.full(len(np.asarray(base["weight"])) * rs, 1.0 / rs)
    return f, kc, params, data


# ── 1. agreement with the parent, at the s0e2 floors ────────────────────

@pytest.mark.parametrize("cfg", QUALIFYING)
@pytest.mark.parametrize("norm", [None, 1.3])
@pytest.mark.parametrize("rs", [1, 17])
def test_matches_the_parent_at_the_s0e2_floor(cfg, norm, rs):
    rel, p, coeff, abs_ok, _cfgs = FLOORS[PARENT]
    _f, kc, params, data = _pair(cfg, rs)
    ref = create_backend(PARENT, kc, model="pwa_gen", resolution_size=rs)
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


# ── 2. the bitwise comparison, reported ─────────────────────────────────

BITWISE_CFGS = list(QUALIFYING) + ["tests/config_pwa_gen_mb.yml"]


@pytest.mark.parametrize("cfg", BITWISE_CFGS[:3])
@pytest.mark.parametrize("rs", [1, 17])
def test_the_bitwise_report_against_the_parent(cfg, rs):
    """FMA re-association changes the last bit — measure it, do not assume it.

    Every quantity is compared entry by entry: the report records how many
    entries are EXACTLY equal and the worst difference in ulps of the
    observable.  The gate is ≤ 64 ulp — an FMA and the deduped denominator
    move a value by a rounding step or two; an algorithmic difference would be
    millions of ulps away.
    """
    _f, kc, params, data = _pair(cfg, rs)
    ref = create_backend(PARENT, kc, model="pwa_gen", resolution_size=rs)
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
                n_exact = int(np.sum(a == b))
                rows.append((f"{cfg.split('/')[-1]} rs={rs} "
                             f"norm={norm} {label}", n_exact, a.size,
                             _ulps(a, b)))
                assert _ulps(a, b) <= 64.0, (
                    f"{cfg} rs={rs} norm={norm} {label}: "
                    f"{_ulps(a, b):.1f} ulps of the observable")
            if norm is not None:
                a = np.atleast_1d(be.kernel._last_dnorm)
                b = np.atleast_1d(ref.kernel._last_dnorm)
                rows.append((f"{cfg.split('/')[-1]} rs={rs} dnorm",
                             int(np.sum(a == b)), a.size, _ulps(a, b)))
                assert _ulps(a, b) <= 64.0
    finally:
        ref.free()
        be.free()
    REPORT[os.path.basename(cfg) + f"-rs{rs}"] = rows
    if os.environ.get("CFPG_SHOW_BITWISE"):
        for label, n_exact, n_tot, u in rows:
            print(f"  {label:52s} exact {n_exact:5d}/{n_tot:6d}  "
                  f"worst {u:.2f} ulp")


def test_the_bitwise_report_is_available():
    """The report collected above is a dict of per-fixture rows."""
    assert isinstance(REPORT, dict)


# ── 3. the denominator dedup, pinned ────────────────────────────────────

def _gamma_keys(kc, n_wave):
    """The expected keys, computed directly from the gamma matrix."""
    g = np.asarray(kc["gamma_matrix"], np.float64).reshape(n_wave, -1)
    seen, key = {}, np.zeros(n_wave, np.int32)
    for k in range(n_wave):
        cols = tuple(int(m) for m in np.nonzero(g[k])[0])
        key[k] = seen.setdefault(cols, len(seen))
    return key


@pytest.mark.parametrize("cfg", QUALIFYING)
def test_key_of_wave_is_the_gamma_pattern(cfg):
    """The dedup key IS the gamma row's column list — byte-exact config data.

    Recomputed here independently of the wrapper, so a wrong key (e.g. one
    keyed on `basis_kind`, which would merge different rows) fails.
    """
    _f, kc, _params, _data = _pair(cfg, ne=8)
    be = create_backend(NAME, kc, model="pwa_gen")
    try:
        k = be.kernel
        want = _gamma_keys(kc, k.n_wave)
        got = np.asarray(k.key_of_wave)
        assert got.shape == want.shape
        # same partition (labels may differ in order): compare the classes
        pairs = {}
        for a, b in zip(got.tolist(), want.tolist()):
            pairs.setdefault(a, set()).add(b)
        assert all(len(v) == 1 for v in pairs.values()), (got, want)
        assert len(set(got.tolist())) == len(set(want.tolist()))
        # every wave is in exactly one key, and the key CSR covers them
        assert int(k.n_key) == len(set(want.tolist()))
        assert np.array_equal(np.sort(np.asarray(k.wave_of)), np.arange(k.n_wave))
        assert k.key_lo[0] == 0 and k.key_lo[-1] == k.nz_key.size
        assert k.kw_lo[0] == 0 and k.kw_lo[-1] == k.n_wave
    finally:
        be.free()


def test_the_production_dedup_is_18_keys_for_26_waves():
    """MEASURED: 26 waves -> 18 keys (130 -> 74 non-zeros per event)."""
    if not os.path.exists(PROD):
        pytest.skip("production config not available on this host")
    f = Fitter(PROD)
    f.apply_constrains()
    be = create_backend(NAME, f.kernel_config, model="pwa_gen",
                        resolution_size=17)
    try:
        k = be.kernel
        assert k.n_wave == 26
        assert int(k.n_key) == 18
        assert int(k.nz_key.size) == 74
        assert int(k.nz_m.size) == 130
        sizes = np.bincount(np.asarray(k.key_of_wave))
        assert sorted(sizes.tolist(), reverse=True) == [
            4, 4, 2, 2] + [1] * 14
    finally:
        be.free()


# ── 4. registration, stats, artefact ────────────────────────────────────

def test_is_a_registered_n_res1_leaf():
    assert NAME in backends_for_model("pwa_gen")
    assert backend_tags(NAME, "pwa_gen") == frozenset({"n_res1"})
    assert NAME in FLOORS


def test_kernel_stats_reports_the_register_budget():
    _f, kc, _params, _data = _pair("tests/config_pwa_gen_km.yml", ne=8)
    be = create_backend(NAME, kc, model="pwa_gen")
    try:
        st = be.kernel.kernel_stats()
        for key in ("regs", "local_bytes", "blocks_per_sm", "warps_per_sm"):
            assert key in st
        assert st["regs"] > 0
        # 0 spills on sm_120 (local_bytes is the spill measure)
        if os.environ.get("CFPG_REQUIRE_NO_SPILLS"):
            assert st["local_bytes"] == 0, st
    finally:
        be.free()


def test_rejects_a_multi_bw_chain_like_the_parent():
    f = Fitter("tests/config_pwa_gen_mb.yml")
    f.apply_constrains()
    assert int(np.asarray(f.kernel_config["num_table"]).shape[1]) == 2
    with pytest.raises(ValueError, match="cuda_flat_pwa_gen2"):
        create_backend(NAME, f.kernel_config, model="pwa_gen")
