"""``cuda_flat_cache_pwa_gen`` — flat ``pwa_gen`` with cached ``F`` + dedup.

Gates:
1. registration — ``pwa_gen`` scope only;
2. the **backend-internal dedup**: rows fold on ``(coeff name, basis kind,
   mass column)``, ``n_basis`` shrinks, and the fold is lossless — the
   per-NAME coefficient gradient totals equal the full-form reference's,
   which is what the basis-weight transform consumes;
3. the **cache is correct**: ``F`` is computed at ``load_data`` and does not
   depend on ``ck``/``coeff``, so changing the parameters between two
   computes must give the same answer as a fresh handle;
4. ``dcoeff`` matches central differences of the kernel's own ``Q`` per NAME;
5. the shared config builder is untouched — a config still has all its rows,
   and ``dedup`` on it is a separate host pass that this backend does not
   require.

The anchor vs ``numpy_flat`` (``Q``/``P``/``ck``/``coeff``/``dnorm``, both
norms, rs 1 and 17), the wrong-length parameter guards and the deduped-config
rejection are covered once, registry-driven, in ``tests/_flat_compute.py``.

Test data stays INSIDE both interpolation grids (outside, the tables clamp
and CPU/GPU diverge).
"""
import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import backends_for_model, create_backend
from tabpwa.cuda._flat_cache_pwa_gen import CUDAKernelFlatCachePWAGen, dedup_basis
from tabpwa.pwa_gen_config import build_from_groups

CFG_KM = "tests/config_pwa_gen_km.yml"
CFG_MB = "tests/config_pwa_gen_mb.yml"

CACHE_FLOOR = 1e-6
NH = 1e-6


def _in_grid(kc, ne, seed=0):
    rng = np.random.RandomState(seed)
    n_basis = int(np.asarray(kc["basis_table"]).shape[1])
    n_fl = int(np.asarray(kc["fl_table"]).shape[-1])
    m_min, m_del = float(kc["m_min"]), float(kc["m_delta"])
    q_min, q_del = float(kc["q_min"]), float(kc["q_delta"])
    n_mass = int(np.max(kc["num_m_index"])) + 1
    n_q = int(np.max(kc["fl_q_index"])) + 1
    n_at = int(np.max(kc["angle_index"])) + 1
    nac = int(np.asarray(kc["angle_k"]).shape[1])
    return {
        "mass": m_min + 0.05 * m_del * n_basis
                + rng.rand(ne, n_mass) * (0.9 * m_del * n_basis),
        "q": q_min + 0.05 * q_del * n_fl
             + rng.rand(ne, n_q) * (0.9 * q_del * n_fl),
        "angle": rng.uniform(-np.pi, np.pi, (ne, n_at, nac)),
        "weight": 0.5 + rng.rand(ne),
        "bkg": 0.05 + 0.1 * rng.rand(ne),
    }


def _rel(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _pair(cfg, ne=64, seed=0):
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    data = _in_grid(kc, ne, seed)
    x = f.initial_values(seed=1)
    params, _resolved = f.build_params(x)
    ref = create_backend("numpy_flat", kc, model="pwa_gen")
    gpu = create_backend("cuda_flat_cache_pwa_gen", kc, model="pwa_gen")
    return ref, gpu, params, data, kc


def _by_name(names, vals, keep):
    """Per-NAME coefficient gradient totals from a full-form vector.

    ``keep`` is the surviving-row list; the cache reports one total per
    surviving row, so the comparison is per name, not per row.
    """
    out = {}
    for nm in dict.fromkeys(names):
        idxs = [i for i, n in enumerate(names) if n == nm]
        surv = [i for i in idxs if i in set(np.asarray(keep).tolist())]
        out[nm] = (sum(vals[i] for i in idxs), surv[0] if surv else None)
    return out


# ── gate 1 ───────────────────────────────────────────────────────────

def test_registration():
    allowed = set(backends_for_model("pwa_gen"))
    assert "cuda_flat_cache_pwa_gen" in allowed
    assert "cuda_flat_cache_pwa_gen" not in backends_for_model("pwa")


# ── gate 2: the backend-internal dedup ──────────────────────────────

def test_dedup_folds_on_the_structural_key():
    """The BUILDER does the fold; this backend's pass is now the identity.

    The rule is the same either way — ``(coeff name, basis kind, mass
    column)`` — and because it only changes ``basis_table`` rows and
    ``gamma_matrix`` columns, every consumer reads ``n_basis`` from a shape.
    """
    m = Fitter(CFG_MB).model
    c = build_from_groups(m, n_interp=200)
    n_built = np.asarray(c["basis_table"]).shape[0]
    # 4 waves x 8 rows folds to 2 factors x ... : fewer rows than uses
    n_uses = np.asarray(c["gamma_matrix"]).shape[0] * \
        np.asarray(c["gamma_matrix"]).shape[1]
    assert n_built <= n_uses
    # every (name, kind, mass column) triple appears exactly once
    keys = list(zip(c["den_names"], np.asarray(c["basis_kind"]).tolist(),
                    np.asarray(c["basis_m_index"]).tolist()))
    assert len(keys) == len(set(keys))
    # the local pass is the identity now
    keep, mapping = dedup_basis(c)
    assert np.array_equal(keep, np.arange(n_built))
    assert np.array_equal(mapping, np.arange(n_built))


def test_the_fold_lives_in_the_builder_not_the_backend():
    """One fold, one place: ``build_from_groups``.

    The backend used to fold privately; now it reads the folded config, so
    ``n_basis`` is simply the config's own row count and both GPU backends
    and ``numpy_flat`` agree without knowing who folded.
    """
    m = Fitter(CFG_MB).model
    c = build_from_groups(m, n_interp=200)
    n_built = np.asarray(c["basis_table"]).shape[0]
    gpu = CUDAKernelFlatCachePWAGen(c)
    try:
        assert gpu.n_basis == n_built
        assert gpu.n_basis_full == n_built          # identical now
        assert np.array_equal(gpu.basis_keep, np.arange(n_built))
        assert "fl_index" not in c and "num_index" not in c
    finally:
        gpu.free()


# ── gate 3: the folded gradient's per-name contract ─────────────────

@pytest.mark.parametrize("cfg", [CFG_KM, CFG_MB])
def test_coeff_gradient_is_one_entry_per_weight(cfg):
    """The gradient carries each weight's derivative exactly once.

    The kernel accumulates one total per surviving row and the folded copies
    are ZERO, so a plain sum over the vector gives the weight's total without
    over-counting — which is what the naming transform consumes.
    """
    ref, gpu, params, data, kc = _pair(cfg)
    try:
        _Qr, gr, _ = ref.compute(params, ref.load_data(data), norm=None)
        _Qc, gc, _ = gpu.compute(params, gpu.load_data(data), norm=None)
        names = kc["den_names"]
        keep = np.asarray(gpu.kernel.basis_keep)
        # the folded copies are zero
        mask = np.ones(len(names), bool)
        mask[keep] = False
        assert np.allclose(gc["coeff"][mask], 0.0), "folded copies not zero"
        # and the survivors carry the per-name total
        for nm in dict.fromkeys(names):
            tot = sum(gr["coeff"][i] for i, n in enumerate(names) if n == nm)
            surv = [i for i in range(len(names))
                    if names[i] == nm and not mask[i]]
            assert len(surv) == 1, (nm, surv)
            # float2 cache floor: dcoeff sums one rounding per (e,k,m) term
            assert abs(tot - gc["coeff"][surv[0]]) <= 1e-4 * max(abs(tot), 1e-12), \
                (nm, tot, gc["coeff"][surv[0]])
    finally:
        ref.free()
        gpu.free()


# ── gate 4: the cache ───────────────────────────────────────────────

def test_cache_is_parameter_independent():
    """F is filled at load_data, so two computes on one handle agree with
    two fresh handles.  A stale cache would show up as a mismatch here."""
    ref, gpu, params, data, kc = _pair(CFG_KM)
    h = gpu.load_data(data)
    try:
        Q1, g1, P1 = gpu.compute(params, h, norm=None)
        # change the parameters: the cache must still be valid
        p2 = dict(params)
        p2["ck"] = np.asarray(params["ck"]) * 1.7
        Q2, g2, P2 = gpu.compute(p2, h, norm=None)
        # a fresh handle must give the same second answer
        Q3, g3, P3 = gpu.compute(p2, gpu.load_data(data), norm=None)
        assert abs(Q2 - Q3) <= 1e-6 * abs(Q3), (Q2, Q3)
        assert _rel(P2, P3) < CACHE_FLOOR
        assert _rel(g2["ck"], g3["ck"]) < CACHE_FLOOR
    finally:
        ref.free()
        gpu.free()


# ── gate 5: finite differences per name ─────────────────────────────

def test_coeff_gradient_matches_fd_per_name():
    ref, gpu, params, data, kc = _pair(CFG_KM, ne=48)
    h = gpu.load_data(data)
    try:
        _Q0, g0, _ = gpu.compute(params, h, norm=None)
        names = kc["den_names"]
        keep = gpu.kernel.basis_keep
        coeff = np.asarray(params["coeff"]).copy()
        for nm in list(dict.fromkeys(names))[:4]:
            i = names.index(nm)
            cp, cm = dict(params), dict(params)
            a, b = coeff.copy(), coeff.copy()
            a[i] += NH
            b[i] -= NH
            cp["coeff"], cm["coeff"] = a, b
            fd = (gpu.compute(cp, h, norm=None)[0]
                  - gpu.compute(cm, h, norm=None)[0]) / (2 * NH)
            surv = [j for j in range(len(names))
                    if names[j] == nm and j in set(keep.tolist())][0]
            assert np.isclose(g0["coeff"][surv], fd, rtol=1e-4, atol=1e-8), (
                nm, g0["coeff"][surv], fd)
    finally:
        ref.free()
        gpu.free()


# ── gate: large n_wave must cross the shared-memory opt-in boundary ──
#
# The fixtures above use n_wave = 2, whose tile is ~20 KiB — far below the
# 48 KiB DEFAULT dynamic shared-memory limit.  A kernel that merely checks
# `smem_bytes <= device_optin` and never calls cudaFuncSetAttribute passes
# every one of those tests and then fails AT LAUNCH for a real wave count:
# the launch is rejected with "invalid argument", and because the macro only
# PRINTS (it does not raise) and the post-launch check is never reached, the
# call returns Q = 0 instead of erroring.  Measured on the pre-fix binary:
# n_wave 56 -> 13515.78 (tile 25.5 KiB, no opt-in needed), n_wave 110 ->
# 0.0 (tile 49.2 KiB, needs opt-in), silently and with exit status 0.
#
# The threshold on this shape is n_wave ~108 (an extra [FC_TE*n_wave]
# double for s_rd), so 110 and 200 both exercise it.


def _synthetic_wide(n_wave_target, n_events=2048):
    """A config wide enough to need the shared-memory opt-in."""
    from benchmark_flat_vs_pwa import FLAT_TEMPLATE, _events, tile_flat
    f = Fitter(FLAT_TEMPLATE)
    f.apply_constrains()
    n_comp = max(1, int(n_wave_target))
    kc = dict(tile_flat(f.kernel_config, n_comp, 2))
    kc["num_table"] = np.ones_like(np.asarray(kc["num_table"]))
    n_wave = int(np.asarray(kc["fl_table"]).shape[0])
    n_basis = int(np.asarray(kc["basis_table"]).shape[0])
    x = f.initial_values(seed=1)
    params, _ = f.build_params(x)
    ck0 = np.asarray(params["ck"])
    params = {
        "ck": np.tile(ck0, (n_wave + ck0.size - 1) // ck0.size)[:n_wave],
        "coeff": np.random.default_rng(7).uniform(0.3, 1.2, n_basis),
    }
    return kc, params, _events(kc, n_events)


@pytest.mark.parametrize("n_wave_target", [110, 200])
def test_wide_config_crosses_the_shared_memory_optin(n_wave_target):
    """A tile above 48 KiB must compute, not return 0."""
    kc, params, data = _synthetic_wide(n_wave_target)
    n_wave = int(np.asarray(kc["fl_table"]).shape[0])
    assert n_wave >= 108, "fixture is not wide enough to need the opt-in"

    be = create_backend("cuda_flat_cache_pwa_gen", kc, model="pwa_gen")
    try:
        Q, _g, P = be.compute(params, be.load_data(data), norm=1024.0)
    finally:
        be.free()
    assert np.isfinite(Q) and Q > 0.0, (
        f"n_wave={n_wave}: Q={Q!r} — the launch was rejected (shared-memory "
        f"opt-in missing?) and the error was swallowed")
    assert np.all(np.asarray(P) >= 0.0)


def test_an_over_large_tile_is_refused_not_silently_empty():
    """Past the DEVICE limit the create must fail loudly, not return 0."""
    kc, _params, _data = _synthetic_wide(470)      # ~176 KiB/block
    with pytest.raises((RuntimeError, ValueError)):
        CUDAKernelFlatCachePWAGen(kc)
