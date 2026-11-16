"""``cuda_flat_cache_pwa_gen`` — flat ``pwa_gen`` with cached ``F`` + dedup.

Gates:
1. registration — ``pwa_gen`` scope only;
2. the **backend-internal dedup**: rows fold on ``(coeff name, basis kind,
   mass column)``, ``n_basis`` shrinks, and the fold is lossless — the
   per-NAME coefficient gradient totals equal the full-form reference's,
   which is what the basis-weight transform consumes;
3. **anchor vs ``numpy_flat``** at ``norm=None`` and a scalar norm: ``Q`` /
   ``P`` / ``ck`` within the float2 amplitude-cache floor (1e-6, measured
   1e-8);
4. the **cache is correct**: ``F`` is computed at ``load_data`` and does not
   depend on ``ck``/``coeff``, so changing the parameters between two
   computes must give the same answer as a fresh handle;
5. ``dcoeff`` matches central differences of the kernel's own ``Q`` per NAME;
6. the shared config builder is untouched — a config still has all its rows,
   and ``dedup`` on it is a separate host pass that this backend does not
   require.

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
    m = Fitter(CFG_MB).model
    c = build_from_groups(m, n_interp=200)
    keep, mapping = dedup_basis(c)
    n_full = np.asarray(c["basis_table"]).shape[0]
    assert keep.size < n_full, "nothing folded"
    # every surviving key is unique
    keys = [(c["den_names"][i], int(c["basis_m_index"][i])) for i in keep]
    assert len(set(keys)) == len(keys)
    # and the map sends each original row onto a survivor
    assert set(mapping.tolist()) == set(range(keep.size))


def test_dedup_is_internal_to_the_backend():
    """The shared builder still emits every row — dedup did not move there."""
    m = Fitter(CFG_MB).model
    c = build_from_groups(m, n_interp=200)
    n_full = np.asarray(c["basis_table"]).shape[0]
    gpu = CUDAKernelFlatCachePWAGen(c)
    try:
        assert gpu.n_basis_full == n_full      # caller's vector is full-form
        assert gpu.n_basis < n_full            # internal fold
        # the config dict itself is untouched
        assert np.asarray(c["basis_table"]).shape[0] == n_full
        assert "fl_index" not in c and "num_index" not in c
    finally:
        gpu.free()


# ── gate 3: anchor ──────────────────────────────────────────────────

@pytest.mark.parametrize("cfg", [CFG_KM, CFG_MB])
@pytest.mark.parametrize("norm", [None, 1.3])
def test_anchor_matches_numpy_flat(cfg, norm):
    ref, gpu, params, data, kc = _pair(cfg)
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=norm)
        Qc, gc, Pc = gpu.compute(params, gpu.load_data(data), norm=norm)
        assert _rel(Pc, Pr) < CACHE_FLOOR, "P"
        assert _rel(Qc, Qr) < CACHE_FLOOR, "Q"
        assert _rel(gc["ck"], gr["ck"]) < CACHE_FLOOR, "ck"
        if norm is not None:
            assert _rel(gpu.kernel._last_dnorm, ref._last_dnorm) < CACHE_FLOOR
    finally:
        ref.free()
        gpu.free()


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
            assert abs(tot - gc["coeff"][surv[0]]) <= 1e-6 * max(abs(tot), 1e-12), \
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


# ── gate 6: contract ────────────────────────────────────────────────

def test_full_form_coeff_length_is_required():
    """The caller keeps the full-form vector; the backend gathers it."""
    ref, gpu, params, data, kc = _pair(CFG_KM, ne=16)
    h = gpu.load_data(data)
    try:
        bad = dict(params)
        bad["coeff"] = np.zeros(gpu.kernel.n_basis)   # the DEDUPED length
        with pytest.raises(ValueError, match="FULL n_basis"):
            gpu.compute(bad, h, norm=None)
    finally:
        ref.free()
        gpu.free()


def test_rejects_an_already_deduped_config():
    from tabpwa.pwa_gen_config import dedup
    m = Fitter(CFG_KM).model
    dd = dedup(build_from_groups(m, n_interp=200))
    with pytest.raises(ValueError, match="FULL"):
        CUDAKernelFlatCachePWAGen(dd)
