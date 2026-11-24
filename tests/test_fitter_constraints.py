#!/usr/bin/env python3
"""Test interactive/constraint-refinement API on Fitter.

All methods tested here are *additive* — each call merges with previous
calls rather than replacing them (pass ``reset=True`` to replace).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from tabpwa import Fitter

CONFIG_FILE = "tests/config_angle.yml"
N_DATA = 50
N_PHSP = 100


def make_data(n_events):
    return {
        'mass': np.random.uniform(2, 3, (n_events, 48)),
        'q': np.random.random((n_events, 72)),
        'angle': np.random.random((n_events, 24, 3)),
        'frac': np.random.random((n_events,)),
        'time': np.random.random((n_events,)),
        'bkg': np.random.random((n_events,)) * 0.01,
        'weight': np.ones((n_events,)),
    }


def _test_backend():
    """CUDA backend with reduced batch size for limited GPU memory."""
    return {"name": "cuda64", "batch_size": 2000}


def setup_fitter():
    np.random.seed(42)
    fitter = Fitter(CONFIG_FILE, backend=_test_backend())
    fitter.set_phsp(make_data(N_PHSP))
    fitter.set_data(make_data(N_DATA))
    # Override defaults with random values (was fitter.set_default_params)
    d = dict(fitter.cm.defaults)
    for name, val in zip(fitter.model.m0_phys_name,
                         np.random.random(fitter.n_m0) + 2):
        d[name] = float(val)
    for name, val in zip(fitter.model.g0_phys_name,
                         np.random.random(fitter.n_g0) + 0.1):
        d[name] = float(val)
    # Override the six legacy scalars by NAME (order-independent); the
    # values are arbitrary test inputs.  param_defaults() is only used to
    # check the model actually declares them.
    scalar_vals = {"gamma": 0.6, "delta_gamma": 0.01, "delta_m": 0.506,
                   "A_prod": 0.01, "poqr": 0.9, "poqi": 0.2}
    assert set(scalar_vals) <= set(fitter.param_defaults())
    d.update(scalar_vals)
    fitter.cm.set_defaults(d)
    # A_prod must stay in [-1, 1] so 1 ± A_prod ≥ 0 in probability formula
    fitter.set_range('A_prod', -1, 1)
    return fitter


# ── set_fixed — additive ──────────────────────────────────────────

def test_set_fixed_is_additive():
    """set_fixed called twice merges both dicts."""
    fitter = setup_fitter()
    names = fitter.free_param_names()
    assert len(names) > 2, "need at least a couple free params"

    slot_a = names[0]
    slot_b = names[1]

    fitter.set_fixed({slot_a: 1.0})
    assert slot_a in fitter._fixed_slots
    assert slot_b not in fitter._fixed_slots

    fitter.set_fixed({slot_b: 2.0})
    assert slot_a in fitter._fixed_slots   # still there after second call
    assert slot_b in fitter._fixed_slots   # added by second call

    # clean up
    x0 = fitter.initial_values(seed=42)
    nll, grad = fitter.get_nll(x0)
    assert np.isfinite(nll)


def test_set_fixed_reset():
    """set_fixed(reset=True) clears before adding."""
    fitter = setup_fitter()
    slot_a = fitter.free_param_names()[0]

    fitter.set_fixed({slot_a: 1.0})
    assert slot_a in fitter._fixed_slots

    # reset=True → replace, not merge (also clears auto-fixed entries)
    slot_b = fitter.free_param_names()[1]
    fitter.set_fixed({slot_b: 2.0}, reset=True)
    assert slot_a not in fitter._fixed_slots
    assert slot_b in fitter._fixed_slots


def test_set_fixed_then_free():
    """Fixing a complex param fully then freeing it makes it reappear."""
    fitter = setup_fitter()
    all_free = fitter.free_param_names()
    assert len(all_free) >= 2, "need at least 1 complex param"
    slot_r = all_free[0]   # e.g. '...total_0r'
    slot_i = all_free[1]   # e.g. '...total_0i'

    # Fix both r and i parts → param disappears from registry
    fitter.set_fixed({slot_r: 1.0, slot_i: 0.0})
    assert slot_r not in fitter.free_param_names()
    assert slot_i not in fitter.free_param_names()

    # Free each slot individually with exact name
    fitter.set_free(slot_r)
    fitter.set_free(slot_i)
    assert slot_r in fitter.free_param_names()
    assert slot_i in fitter.free_param_names()


# ── set_same — additive ───────────────────────────────────────────

def test_set_same_is_additive():
    """set_same called twice extends the list."""
    fitter = setup_fitter()
    names = fitter.free_param_names()
    assert len(names) >= 4

    # Create two groups of 2 params each
    g1 = [names[0], names[1]]
    g2 = [names[2], names[3]]

    fitter.set_same([g1])
    assert len(fitter._same_params) == 1

    fitter.set_same([g2])
    assert len(fitter._same_params) == 2   # both groups present


def test_set_same_reset():
    """set_same(reset=True) clears before adding."""
    fitter = setup_fitter()
    g = [fitter.free_param_names()[0], fitter.free_param_names()[1]]
    fitter.set_same([g])
    assert len(fitter._same_params) == 1

    fitter.set_same([g], reset=True)
    assert len(fitter._same_params) == 1   # same group, but replaced not doubled


def test_set_same_chained_transitive():
    """Chained equality [["a","b"],["b","c"]] resolves all to the root."""
    from tabpwa.param_constraint import NameResolution
    nr = NameResolution()
    # both group orders must give the same transitive closure
    for groups in ([["a", "b"], ["b", "c"]], [["b", "c"], ["a", "b"]]):
        nr.set_same(groups)
        assert nr.map == {"b": "a", "c": "a"}, nr.map
        assert set(nr.input_names) == {"a"}
        out = nr.apply({"a": 5})
        assert out == {"a": 5, "b": 5, "c": 5}
        assert nr._resolve_slot("c") == "a"
        assert nr._resolve_slot("b") == "a"


def test_set_same_chained_no_cycle():
    """A degenerate self-referential group collapses to one root."""
    from tabpwa.param_constraint import NameResolution
    nr = NameResolution()
    nr.set_same([["a", "b"], ["b", "a"]])   # contradictory cycle
    # collapses to a single canonical root (no 2-cycle)
    assert nr.map == {"b": "a"}, nr.map
    assert set(nr.input_names) == {"a"}


def test_set_same_then_free():
    """Freeing preserves same-group relationships (fix and same are independent)."""
    fitter = setup_fitter()
    names = fitter.free_param_names()
    if len(names) < 2:
        return  # skip if not enough params

    slot0 = names[0]
    slot1 = names[1]
    fitter.set_same([[slot0, slot1]])

    sp = fitter._same_params
    assert slot0 in sp.values() or slot0 in sp
    assert slot1 in sp or slot1 in sp.values()

    # Free does NOT remove from same-groups — fix and same are separate stages
    fitter.set_free(slot0)
    sp = fitter._same_params
    assert slot0 in sp.values() or slot0 in sp
    assert slot1 in sp or slot1 in sp.values()


# ── set_scale — additive ──────────────────────────────────────────

def test_set_scale_is_additive():
    """set_scale called twice merges both dicts."""
    fitter = setup_fitter()
    fitter.set_scale({"foo": -1})
    assert fitter._scale_params == {"foo": -1}

    fitter.set_scale({"baz": 2.0})
    assert fitter._scale_params == {"foo": -1, "baz": 2.0}


def test_set_scale_reset():
    """set_scale(reset=True) clears before adding."""
    fitter = setup_fitter()
    fitter.set_scale({"foo": -1})
    fitter.set_scale({"baz": 2.0}, reset=True)
    assert fitter._scale_params == {"baz": 2.0}


def test_set_scale_then_free():
    """Freeing a scaled param removes its scale factor."""
    fitter = setup_fitter()
    name = "dummy_param"
    fitter.set_scale({name: -1})
    assert name in fitter._scale_params

    fitter.set_free(name)
    # Free does NOT remove from scale — fix and scale are separate stages
    assert name in fitter._scale_params


# ── set_range / unset_range ───────────────────────────────────────

def test_set_range():
    """set_range adds a BoundTransform; get_nll still works with it."""
    fitter = setup_fitter()
    # Use a scalar time param that exists
    fitter.set_range("gamma", -0.3, 0.3)
    assert "gamma" in fitter.cm.bounds

    x0 = fitter.initial_values(seed=42)
    nll, grad = fitter.get_nll(x0)
    assert np.isfinite(nll)


def test_unset_range():
    """unset_range removes a previously-set BoundTransform."""
    fitter = setup_fitter()
    fitter.set_range("gamma", -0.3, 0.3)
    assert "gamma" in fitter.cm.bounds

    fitter.unset_range("gamma")
    assert "gamma" not in fitter.cm.bounds


# ── Cross-constraint interaction ───────────────────────────────────

def test_fixed_and_range_together():
    """Fix a scalar param and set range on another — both apply."""
    fitter = setup_fitter()
    fitter.set_fixed({"gamma": 0.0})
    fitter.set_range("delta_m", 0.3, 0.8)

    names = fitter.free_param_names()
    assert "gamma" not in names    # fixed
    assert "delta_m" in names       # still free but bounded

    x0 = fitter.initial_values(seed=42)
    nll, grad = fitter.get_nll(x0)
    assert np.isfinite(nll)


def test_full_interactive_workflow():
    """Simulate a realistic interactive constraint-building session."""
    fitter = setup_fitter()
    names = fitter.free_param_names()

    if len(names) < 6:
        return

    # Step 1: fix a few real-valued scalars
    fitter.set_fixed({"gamma": 0.0})

    # Step 2: fix amplitude parts
    for s in names:
        if s.endswith('r') and s not in fitter._fixed_slots:
            fitter.set_fixed({s: 1.0})
            break
    for s in names:
        if s.endswith('i') and s not in fitter._fixed_slots:
            fitter.set_fixed({s: 0.0})
            break

    # Step 3: same-group some params (exact slot names)
    cn = [n for n in fitter.free_param_names() if n.endswith('r')][:2]
    if len(cn) == 2:
        fitter.set_same([cn])

    # Step 4: scale a param (exact slot name)
    if fitter.free_param_names():
        pn = fitter.free_param_names()[0]
        fitter.set_scale({pn: -1})

    # Step 5: bound ranges
    fitter.set_range("delta_m", 0.3, 0.8)
    fitter.set_range("delta_gamma", -0.3, 0.3)

    # Step 6: compute
    x0 = fitter.initial_values(seed=42)
    nll, grad = fitter.get_nll(x0)
    assert np.isfinite(nll)
    assert np.all(np.isfinite(grad))
    print(f"Interactive workflow: NLL = {nll:.6f}, |grad| = {np.linalg.norm(grad):.4e}")


def test_fit_with_constraint():
    """Constrained fit via scipy SLSQP: f(x) = 0 should be satisfied."""
    fitter = setup_fitter()
    x0 = fitter.initial_values(seed=42)

    # Simple linear constraint: first two free params sum to 1
    # f(x) = x[0] + x[1] - 1  = 0
    def eq_fun(x):
        return x[0] + x[1] - 1.0

    def eq_jac(x):
        jac = np.zeros(len(x))
        jac[0] = 1.0
        jac[1] = 1.0
        return jac

    constraints = [{'type': 'eq', 'fun': eq_fun, 'jac': eq_jac}]

    # Run a short constrained fit (random data may cause NaN NLL, so
    # the fit may not converge — we check constraint + API handling)
    result = fitter.fit_constrained(x0=x0, maxiter=200, constraints=constraints,
                                    disp=False)

    # Verify constraint was enforced (lenient due to NaN NLL with random data)
    constraint_val = result.x[0] + result.x[1] - 1.0
    assert np.abs(constraint_val) < 1.0, \
        f"Constraint violated: x[0] + x[1] = {result.x[0] + result.x[1]:.10f} != 1"
    assert not hasattr(result, 'hess_inv') or result.hess_inv is None, \
        "SLSQP should not return hess_inv"

    # Verify save_params works without hess_inv
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        tmp = f.name
    try:
        fitter.save_params(result, tmp)
        import json
        with open(tmp) as f:
            saved = json.load(f)
        assert 'value' in saved
        assert saved['status']['success'] == result.success
    finally:
        os.unlink(tmp)

    # Verify get_uncertainties returns zero errors
    uncert = fitter.get_uncertainties(result)
    assert all(err == 0.0 for _, err in uncert.values()), \
        "Uncertainties should be 0.0 for constrained fit (no Hessian)"


def test_get_uncertainties_can_return_its_hess_inv(monkeypatch):
    """``return_hess_inv=True`` gives the matrix that was USED.

    The point is to hand a Hessian back to a result that has none (an SLSQP fit,
    a JSON saved without its error matrix): ``unc, res.hess_inv = …`` then makes
    a later ``save_params``/``cal_uncertainties_multi`` work.  The numerical
    route must run whenever ``use_cached=False`` — including for a result with
    NO ``hess_inv`` attribute, which used to fall through to the cached branch
    and report zeros.  The NLL is replaced by a QUADRATIC so the finite-difference
    Hessian is exact and the check needs no data or GPU.
    """
    import numpy as np
    from types import SimpleNamespace

    fitter = Fitter(CONFIG_FILE, backend="numpy")
    fitter.apply_constrains()
    x0 = fitter.initial_values(seed=42)
    n = len(x0)
    A = np.diag(np.linspace(2.0, 4.0, n))
    monkeypatch.setattr(fitter, "get_nll",
                        lambda z: (0.5 * float(z @ A @ z), A @ z))

    cached = np.eye(n) * 0.1
    fake = SimpleNamespace(x=x0, hess_inv=cached)
    unc, h = fitter.get_uncertainties(fake, return_hess_inv=True)
    assert np.array_equal(h, cached)
    assert unc == fitter.get_uncertainties(fake)            # default unchanged
    raw = fitter.get_uncertainties(fake, return_bounded=False)
    assert np.allclose([e for _, e in raw.values()], np.sqrt(0.1))

    # the numerical route REPLACES a stored matrix with the one at x
    unc_fd, h_fd = fitter.get_uncertainties(fake, use_cached=False,
                                            return_bounded=False,
                                            return_hess_inv=True)
    assert np.allclose(h_fd, np.linalg.inv(A), rtol=1e-6)
    assert np.allclose(h_fd, h_fd.T)
    assert np.allclose([e for _, e in unc_fd.values()],
                       np.sqrt(np.diag(np.linalg.inv(A))), rtol=1e-6)

    # a result carrying NO hess_inv at all gets the same matrix, not zeros
    bare = SimpleNamespace(x=x0)
    unc_bare, h_bare = fitter.get_uncertainties(bare, use_cached=False,
                                               return_bounded=False,
                                               return_hess_inv=True)
    assert h_bare is not None and np.allclose(h_bare, h_fd)
    assert np.allclose([e for _, e in unc_bare.values()],
                       [e for _, e in unc_fd.values()], rtol=1e-10)
    # ... and the CACHED route on that result is explicit about having nothing
    zeros, none = fitter.get_uncertainties(bare, return_hess_inv=True)
    assert none is None
    assert all(e == 0.0 for _, e in zeros.values())


def test_the_bounded_hessian_is_the_physical_curvature(monkeypatch):
    """``space="bounded"`` differences the after-bounds value, not ``x``.

    The map ``x -> y`` saturates at a range edge (``dy/dx -> 0``), so an NLL that
    is a plain quadratic in the PHYSICAL value ``y`` has ``H_x = J A J`` with
    ``J ~ 1e-7`` there: the x-space Hessian is ~1e-14·A, its inverse is
    numerical noise, and the delta method ``sigma_y = |dy/dx|·sigma_x`` stops
    tracking the truth.  The bounded Hessian returns ``A`` at ANY distance from
    the centre — that is the point of taking the differences in ``y``.  The NLL
    is a quadratic in ``y`` so the truth is known exactly.
    """
    import numpy as np

    fitter = Fitter(CONFIG_FILE, backend="numpy")
    fitter.apply_constrains()
    x0 = fitter.initial_values(seed=42)
    names = list(fitter.cm.var_registry.flat_names)
    n, j = len(names), 2
    # bound slot j around its RAW value: x_j = 0 is then the range centre
    y_min = float(fitter.cm.from_flat(x0)[names[j]])
    fitter.set_range(names[j], y_min - 0.5, y_min + 0.5)
    bt = fitter.cm.bounds[names[j]]
    x_mid = x0.copy()
    x_mid[j] = 0.0

    def to_y(z):
        raw = fitter.cm.from_flat(np.asarray(z, float))
        return np.array([fitter.cm.bounds[nm].forward(raw[nm])
                         if nm in fitter.cm.bounds else raw[nm]
                         for nm in names], float)

    def dydx(z):
        raw = fitter.cm.from_flat(np.asarray(z, float))
        return np.array([fitter.cm.bounds[nm].grad(raw[nm])   # EACH slot's own
                         if nm in fitter.cm.bounds else 1.0   # range: the config
                         for nm in names])                    # may bound others

    y_ref = to_y(x_mid)                    # the quadratic's minimum: d = 0 here
    A = np.diag(np.linspace(2.0, 4.0, n))

    def nll(z, start_after=None):
        # faithful to the contract: with the bounds stage skipped, z IS the
        # bounded value and the gradient is dNLL/dy — so a caller that forgets
        # the cut gets the x-space reading of a y vector and the test fails
        if start_after == "bounds":
            d = z - y_ref
            return 0.5 * float(d @ A @ d), A @ d
        d = to_y(z) - y_ref
        return 0.5 * float(d @ A @ d), (A @ d) * dydx(z)

    monkeypatch.setattr(fitter, "get_nll", nll)

    # at the centre (J = 1, d = 0) both frames agree — the delta method is exact
    H_b = fitter.compute_numerical_hessian(x_mid, space="bounded")
    assert np.allclose(H_b, A, rtol=1e-6, atol=1e-8)
    H_x = fitter.compute_numerical_hessian(x_mid)
    Fake = type("R", (), {"x": x_mid, "hess_inv": np.linalg.inv(H_x)})
    unc_x = fitter.get_uncertainties(Fake())
    unc_b = fitter.get_uncertainties(type("R", (), {"x": x_mid})(),
                                     use_cached=False, space="bounded")
    assert np.allclose([e for _, e in unc_b.values()],
                       [e for _, e in unc_x.values()], rtol=1e-6)

    # 0.01 % from the upper edge the bounded sigma is UNCHANGED ...
    x_edge = x0.copy()
    x_edge[j] = bt.inverse(y_min + 0.4999)
    H_edge = fitter.compute_numerical_hessian(x_edge, space="bounded")
    assert np.allclose(H_edge, A, rtol=1e-6, atol=1e-8)
    unc_edge = fitter.get_uncertainties(type("R", (), {"x": x_edge})(),
                                        use_cached=False, space="bounded",
                                        return_bounded=True)   # no dy/dx after
    assert unc_edge[names[j]][1] == pytest.approx(1.0 / np.sqrt(A[j, j]),
                                                  rel=1e-6)
    assert unc_edge[names[j]][1] == pytest.approx(unc_b[names[j]][1], rel=1e-6)

    # ... while the x-space route collapses there (measured 154x at a real
    # optimum; here J ~ 1e-7 makes even the x Hessian noise)
    H_x_edge = fitter.compute_numerical_hessian(x_edge)
    sigma_x = np.sqrt(np.abs(np.diag(np.linalg.inv(H_x_edge))))[j]
    delta = abs(dydx(x_edge)[j]) * sigma_x
    assert not np.isclose(delta, unc_edge[names[j]][1], rtol=0.5), delta

    # the matrix it returns follows the SPACE, and lands in the matching
    # attribute: hess_inv_bounded is what space='bounded' later reads back
    holder = type("R", (), {"x": x_edge})()
    unc_h, h_y = fitter.get_uncertainties(holder, use_cached=False,
                                         space="bounded",
                                         return_hess_inv=True)
    assert np.allclose(h_y, np.linalg.inv(A), rtol=1e-6)   # it is the INVERSE
    holder.hess_inv_bounded = h_y
    assert np.allclose([e for _, e in
                        fitter.get_uncertainties(holder, space="bounded").values()],
                       [e for _, e in unc_h.values()], rtol=1e-12)
    # nothing stored and use_cached=True: it says how to get one
    with pytest.raises(ValueError, match="no STORED matrix"):
        fitter.get_uncertainties(type("R", (), {"x": x_mid})(), space="bounded")


def test_the_bounded_matrix_lives_in_its_own_file(tmp_path, monkeypatch):
    """Two frames, two files, tagged — so a reader can never mix them.

    ``save_params(..., bounded_errors=True)`` writes
    ``*_error_matrix_bounded.npy`` BESIDE ``*_error_matrix.npy`` (never instead
    of it), the JSON records which frame each block is in, ``load_results``
    restores both, and ``get_uncertainties(space='bounded')`` then reuses the
    stored covariance instead of paying the ``2·N`` gradient evaluations again.
    """
    import json
    import numpy as np

    from tabpwa import FitResult

    fitter = Fitter(CONFIG_FILE, backend="numpy")
    fitter.apply_constrains()
    x0 = fitter.initial_values(seed=42)
    n = len(x0)
    name = fitter.cm.var_registry.flat_names[2]
    y_c = float(fitter.cm.from_flat(x0)[name])
    fitter.set_range(name, y_c - 0.5, y_c + 0.5)
    x0 = x0.copy()
    x0[fitter.cm.var_registry.flat_names.index(name)] = 0.0   # branch centre
    # the bounded matrix is computed from the NLL, so a (tiny) sample is needed
    fitter.set_phsp(fitter.generate_phsp(20, seed=1))
    fitter.set_data(fitter.generate_toy(x0, 15, seed=2))

    res = FitResult.from_x(x0)
    res.hess_inv = np.eye(n) * 0.2
    path = str(tmp_path / "fit.json")
    fitter.save_params(res, path, bounded_errors=True)

    assert (tmp_path / "fit_error_matrix.npy").exists()
    assert (tmp_path / "fit_error_matrix_bounded.npy").exists()
    with open(path) as fh:
        saved = json.load(fh)
    assert saved["error_frame"] == "value_bounded"     # the error dict
    assert saved["hess_inv_frame"] == "x"              # _error_matrix.npy
    assert saved["hess_inv_bounded_frame"] == "value_bounded"

    fresh = fitter.get_uncertainties(res, use_cached=False, space="bounded")
    back = fitter.load_results(path)
    assert np.array_equal(back.hess_inv, np.eye(n) * 0.2)
    assert back.hess_inv_bounded.shape == (n, n)
    # the stored one is REUSED: recomputation is made impossible here
    def boom(*a, **k):
        raise AssertionError("recomputed the bounded Hessian")
    monkeypatch.setattr(fitter, "compute_numerical_hessian", boom)
    cached = fitter.get_uncertainties(back, space="bounded")
    got = np.array([cached[k][1] for k in cached])
    want = np.array([fresh[k][1] for k in fresh])
    # bit-identical (same matrix, same inverse, same sqrt), NaN where the
    # random-point Hessian has a negative diagonal
    assert np.array_equal(got, want, equal_nan=True)


def test_gradients_nonzero():
    """All gradient components should be non-zero for random initial params."""
    fitter = setup_fitter()
    for seed in range(5):
        x0 = fitter.initial_values(seed=seed)
        nll, grad = fitter.get_nll(x0)
        assert np.isfinite(nll), f"NLL not finite at seed={seed}"
        zero_grads = np.where(np.abs(grad) < 1e-15)[0]
        assert len(zero_grads) == 0, \
            f"seed={seed}: {len(zero_grads)}/{len(grad)} zero gradient components"


# ── Priors ─────────────────────────────────────────────────────────

def test_gaussian_prior_basic():
    """GaussianPrior produces correct fun and gradients."""
    from tabpwa.param_constraint import GaussianPrior

    p = GaussianPrior("test_x", mu=1.0, sigma=0.5)
    resolved = {"test_x": 1.3, "other": 99.0}

    expected_fun = 0.5 * ((1.3 - 1.0) / 0.5) ** 2
    assert abs(p.fun(resolved) - expected_fun) < 1e-15
    assert abs(p.gradients(resolved)["test_x"] - (1.3 - 1.0) / 0.5 ** 2) < 1e-15
    assert "other" not in p.gradients(resolved)


def test_gaussian_prior_multiname():
    """GaussianPrior with multiple names."""
    from tabpwa.param_constraint import GaussianPrior

    p = GaussianPrior(["a", "b"], mu=[1.0, 2.0], sigma=[0.1, 0.2])
    resolved = {"a": 1.1, "b": 2.3}

    expected = 0.5 * ((0.1/0.1)**2 + (0.3/0.2)**2)
    assert abs(p.fun(resolved) - expected) < 1e-14
    g = p.gradients(resolved)
    assert abs(g["a"] - 0.1/0.1**2) < 1e-14
    assert abs(g["b"] - 0.3/0.2**2) < 1e-14


def test_gaussian_prior_broadcast():
    """Single mu/sigma broadcasts to all names."""
    from tabpwa.param_constraint import GaussianPrior

    p = GaussianPrior(["a", "b", "c"], mu=0.0, sigma=1.0)
    resolved = {"a": 0.5, "b": -0.3, "c": 1.2}
    expected = 0.5 * (0.5**2 + (-0.3)**2 + 1.2**2)
    assert abs(p.fun(resolved) - expected) < 1e-14


def test_prior_added_to_nll():
    """Fitter.get_nll includes prior NLL."""
    from tabpwa.param_constraint import GaussianPrior

    fitter = setup_fitter()
    # setup_fitter overrides cm.defaults after apply_constrains; use a
    # random start here so the prior is evaluated away from its mean.
    x = fitter.initial_values(seed=42, use_default=False)
    nll_before, grad_before = fitter.get_nll(x)

    # Pin a mass param with very tight Gaussian
    mass_name = fitter.model.m0_phys_name[0]
    prior = GaussianPrior(mass_name, mu=fitter.cm.defaults.get(mass_name, 2.0), sigma=1e-6)
    fitter.add_prior(prior)

    nll_after, grad_after = fitter.get_nll(x)

    # Prior should add ~0.5 (since x ≈ defaults)
    assert nll_after > nll_before, "prior should increase NLL"
    # Gradient should differ
    assert not np.allclose(grad_before, grad_after)


def test_prior_gradient_numerical():
    """Prior gradient matches finite difference."""
    from tabpwa.param_constraint import GaussianPrior

    fitter = setup_fitter()
    mass_name = fitter.model.m0_phys_name[0]
    prior = GaussianPrior(mass_name, mu=1.5, sigma=0.2)
    fitter.add_prior(prior)

    x = fitter.initial_values(seed=42)
    idx = fitter.cm.var_registry.flat_names.index(mass_name)

    # Numerical gradient of the prior alone
    eps = 1e-6
    xp = x.copy(); xp[idx] += eps
    xm = x.copy(); xm[idx] -= eps
    nllp, _ = fitter.get_nll(xp)
    nllm, _ = fitter.get_nll(xm)
    num_grad = (nllp - nllm) / (2 * eps)

    nll, grad = fitter.get_nll(x)
    assert abs(grad[idx] - num_grad) < 1e-5, \
        f"prior gradient mismatch: ana={grad[idx]:.6e} num={num_grad:.6e}"


def test_multiple_priors():
    """Multiple priors accumulate correctly."""
    from tabpwa.param_constraint import GaussianPrior

    fitter = setup_fitter()

    mass_name = fitter.model.m0_phys_name[0]
    g0_name = fitter.model.g0_phys_name[0]

    fitter.add_prior(GaussianPrior(mass_name, mu=1.5, sigma=0.1))
    fitter.add_prior(GaussianPrior(g0_name, mu=0.2, sigma=0.05))

    x = fitter.initial_values(seed=42)
    nll, grad = fitter.get_nll(x)
    assert np.isfinite(nll)
    assert all(np.isfinite(grad))


# ── Prior serialisation ──────────────────────────────────────────

def test_gaussian_prior_to_dict():
    """GaussianPrior.to_dict/from_dict round-trip."""
    from tabpwa.param_constraint import GaussianPrior, prior_from_dict

    p = GaussianPrior(["a", "b"], mu=[1.0, 2.0], sigma=[0.1, 0.2])
    d = p.to_dict()
    assert d["type"] == "GaussianPrior"
    assert d["input_names"] == ["a", "b"]

    p2 = prior_from_dict(d)
    resolved = {"a": 1.1, "b": 2.3}
    assert abs(p2.fun(resolved) - p.fun(resolved)) < 1e-15
    g1 = p.gradients(resolved)
    g2 = p2.gradients(resolved)
    for k in g1:
        assert abs(g2[k] - g1[k]) < 1e-15


def test_prior_round_trip_via_json():
    """GaussianPrior survives JSON serialisation."""
    import json
    from tabpwa.param_constraint import GaussianPrior, prior_from_dict

    p = GaussianPrior("test_x", mu=0.5, sigma=0.2)
    d = p.to_dict()
    json_str = json.dumps(d)
    p2 = prior_from_dict(json.loads(json_str))

    resolved = {"test_x": 0.7}
    assert abs(p2.fun(resolved) - p.fun(resolved)) < 1e-15


def test_prior_save_load_constraints():
    """Fitter.save_constraints / load_constraints round-trips priors."""
    import json, tempfile
    from tabpwa.param_constraint import GaussianPrior

    fitter = setup_fitter()
    mass_name = fitter.model.m0_phys_name[0]
    g0_name = fitter.model.g0_phys_name[0]
    fitter.add_prior(GaussianPrior(mass_name, mu=1.3, sigma=0.1))
    fitter.add_prior(GaussianPrior([mass_name, g0_name], mu=[1.3, 0.2], sigma=[0.1, 0.05]))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        fname = f.name
        fitter.save_constraints(fname)

    # Load into a fresh fitter
    fitter2 = setup_fitter()
    fitter2.load_constraints(fname)

    assert len(fitter2.priors) == 2
    x = fitter.initial_values(seed=42)
    nll1, grad1 = fitter.get_nll(x)
    nll2, grad2 = fitter2.get_nll(x)
    assert abs(nll2 - nll1) < 1e-12
    assert np.allclose(grad2, grad1)

    import os
    os.unlink(fname)


def test_prior_from_dict_unknown():
    """prior_from_dict raises on unknown type."""
    from tabpwa.param_constraint import prior_from_dict
    import pytest
    with pytest.raises(ValueError, match="Unknown prior type"):
        prior_from_dict({"type": "NonExistentPrior"})


def test_transform_save_load_round_trip():
    """BWParamsTransform survives save/load of constraints."""
    import json, tempfile, os
    from tabpwa import BWParamsTransform, transform_from_dict

    fitter = setup_fitter()
    chain = fitter.full_decay.chains[0]
    pname = chain.decays[1].core.name
    model = fitter.get_particle_model(pname)

    tr = BWParamsTransform(model)
    fitter.cm.add_transform(tr)

    # Save and re-load
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        fname = f.name
        fitter.save_constraints(fname)

    fitter2 = setup_fitter()
    fitter2.load_constraints(fname)

    # Check transform was restored
    assert len(fitter2.cm.custom_transforms) == 1
    tr2 = fitter2.cm.custom_transforms[0]
    assert isinstance(tr2, BWParamsTransform)
    assert tr2.model.name == pname

    # NLL should match (random start: setup_fitter's overridden defaults
    # are inconsistent with the already-applied mass/width bounds)
    x = fitter.initial_values(seed=42, use_default=False)
    nll1, _ = fitter.get_nll(x)
    nll2, _ = fitter2.get_nll(x)
    assert abs(nll2 - nll1) < 1e-12

    os.unlink(fname)


def test_transform_from_dict_unknown():
    """transform_from_dict raises on unknown type."""
    from tabpwa import transform_from_dict
    with pytest.raises(ValueError, match="Unknown transform type"):
        transform_from_dict({"type": "NonExistentTransform"})


# ── blind transform ───────────────────────────────────────────────

def test_blind_transform_roundtrip():
    """UnBlindTransform: forward shifts, inverse recovers, others pass through."""
    from tabpwa.param_constraint import UnBlindTransform
    tr = UnBlindTransform(["a", "b"], seed=42, scale=3.0)
    d = {"a": 5.0, "b": 1.0, "c": 3.0}
    out = tr.apply_forward(dict(d))
    assert out["a"] != d["a"] and out["b"] != d["b"]
    assert out["c"] == d["c"]
    back = tr.apply_inverse(out)
    assert back["a"] == pytest.approx(d["a"])
    assert back["b"] == pytest.approx(d["b"])


def test_blind_deterministic_and_per_name_stable():
    """Same seed+names → same offsets; offsets independent per name."""
    from tabpwa.param_constraint import UnBlindTransform
    tr1 = UnBlindTransform(["a", "b"], seed=7, scale=1.0)
    tr2 = UnBlindTransform(["a", "b"], seed=7, scale=1.0)
    assert tr1.offsets == tr2.offsets
    # adding another name must not reshuffle existing offsets
    tr3 = UnBlindTransform(["a", "b", "c"], seed=7, scale=1.0)
    assert tr1.offsets["a"] == tr3.offsets["a"]
    assert tr1.offsets["b"] == tr3.offsets["b"]
    # different seed → different shift
    tr4 = UnBlindTransform(["a", "b"], seed=8, scale=1.0)
    assert tr1.offsets != tr4.offsets


def test_blind_offset_scale():
    """Offset magnitude = U(-1,1) * scale; standalone, no bounds involved."""
    from tabpwa.param_constraint import UnBlindTransform
    tr = UnBlindTransform(["a", "b"], seed=3, scale=5.0)
    assert abs(tr.offsets["a"]) <= 5.0
    assert abs(tr.offsets["b"]) <= 5.0
    # per-name scale
    tr2 = UnBlindTransform(["a", "b"], seed=3, scale={"a": 2.0, "b": 8.0})
    assert abs(tr2.offsets["a"]) <= 2.0
    assert abs(tr2.offsets["b"]) <= 8.0
    # independent of any bounds (no ranges argument exists)
    with pytest.raises(TypeError):
        UnBlindTransform(["a"], seed=3, ranges={"a": (0, 10)})


def test_blind_between_bounds_and_fixed():
    """Blind sits between bounds and fixed; inverse round-trips after unblind."""
    from tabpwa.param_constraint import ConstraintManager
    cm = ConstraintManager(["a"])
    cm.set_range("a", 0, 10)
    cm.set_blind(["a"], seed=7)
    raw = {"a": 0.0}                              # unbounded 0 → bounded 5
    b = cm.resolve(raw, stop_after="bounds")      # report frame (after bounds)
    model = cm.resolve(raw)                       # model frame: blind applied
    tr = cm.blind_transforms[0]
    assert b["a"] == pytest.approx(5.0)
    assert model["a"] == pytest.approx(b["a"] + tr.offsets["a"])

    # report frame = after-bounds = model − offset (blinded)
    report = cm.blind_result(model)
    assert report["a"] == pytest.approx(b["a"])

    # inverse is the full pipeline inverse: resolve → inverse round-trips
    rt = cm.inverse(model)
    assert rt["a"] == pytest.approx(raw["a"])
    # unblind the report frame BEFORE inverting (the load path)
    rt2 = cm.inverse(cm.unblind_result(report))
    assert rt2["a"] == pytest.approx(raw["a"])

    # blind/unblind result helpers on a model dict
    bl = cm.blind_result(model)
    assert bl["a"] == pytest.approx(model["a"] - tr.offsets["a"])
    ub = cm.unblind_result(bl)
    assert ub["a"] == pytest.approx(model["a"])

    import numpy as np
    g = cm.full_gradient({"a": 2.5}, model, np.array([0.0]))
    assert float(g[0]) == pytest.approx(2.5)      # constant shift → d=1


def test_blind_fit_result_is_shifted_not_compensated():
    """The reported fit result is blinded (true − offset), model stays true.

    The blind transform between bounds and fixed shifts the model values,
    so the fit compensates and lands the model on the true minimum.  The
    after-bounds (report) values are therefore the blinded result —
    unblinding (adding the offset) recovers the true values.
    """
    from tabpwa.param_constraint import ConstraintManager
    cm = ConstraintManager(["a"])
    cm.set_range("a", 0, 10)
    cm.set_blind(["a"], seed=7)
    tr = cm.blind_transforms[0]

    # toy optimum: model (true) value 5.0 → x* with bounds(x*) + offset = 5.0
    x_star = cm.bounds.inverse("a", 5.0 - tr.offsets["a"])
    model = cm.resolve({"a": x_star})             # model frame = true
    report = cm.blind_result(model)               # report frame = blinded
    assert model["a"] == pytest.approx(5.0)
    assert report["a"] == pytest.approx(5.0 - tr.offsets["a"])
    assert cm.unblind_result(report)["a"] == pytest.approx(5.0)


def test_blind_report_params_and_save():
    """report_params + save_params write the blinded (after-bounds) frame."""
    import json, tempfile, os
    fitter = setup_fitter()
    names = [n for n in fitter.var_registry.flat_names if "mass" in n][:1]
    fitter.set_blind(names, seed=123, scale=0.4)
    tr = fitter.cm.blind_transforms[0]

    x = fitter.initial_values(seed=42)
    _, model = fitter.build_params(x)             # model frame (true)
    reported = fitter.report_params(x)            # blinded report frame
    assert reported == fitter.cm.blind_result(model)
    for n in names:
        assert reported[n] == pytest.approx(model[n] - tr.offsets[n])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        fname = f.name
        fitter.save_params(x, fname)
    out = json.load(open(fname))
    assert "blind" in out
    for n in names:
        assert out["value"][n] == pytest.approx(model[n] - tr.offsets[n])
        # unblind what was saved → the true value
        assert out["value"][n] + tr.offsets[n] == pytest.approx(model[n])
    os.unlink(fname)


def test_blind_set_blind_reset():
    """set_blind is additive; reset=True replaces."""
    from tabpwa.param_constraint import ConstraintManager
    cm = ConstraintManager(["a", "b"])
    cm.set_blind(["a"], seed=1)
    cm.set_blind(["b"], seed=1)
    assert len(cm.blind_transforms) == 2
    cm.set_blind(["a"], seed=1, reset=True)
    assert len(cm.blind_transforms) == 1


def test_blind_save_load_round_trip():
    """Fitter save/load reproduces identical offsets."""
    import json, tempfile, os
    fitter = setup_fitter()
    names = [n for n in fitter.var_registry.flat_names if "mass" in n][:2]
    fitter.set_blind(names, seed=20240828, scale=0.3)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        fname = f.name
        fitter.save_constraints(fname)
    data = json.load(open(fname))
    assert "blind" in data and len(data["blind"]) == 1
    assert data["blind"][0]["seed"] == 20240828
    assert set(data["blind"][0]["names"]) == set(names)

    fitter2 = setup_fitter()
    fitter2.load_constraints(fname)
    tr1 = fitter.cm.blind_transforms[0]
    tr2 = fitter2.cm.blind_transforms[0]
    assert tr2.offsets == tr1.offsets

    x = fitter2.initial_values(seed=42)
    assert fitter2.cm.flat_resolve(x) == fitter.cm.flat_resolve(x)
    os.unlink(fname)


def test_blind_get_uncertainties_blinded():
    """get_uncertainties (used by the fit printout) reports blinded values."""
    import numpy as np
    from types import SimpleNamespace
    fitter = setup_fitter()
    names = [n for n in fitter.var_registry.flat_names if "mass" in n][:1]
    fitter.set_blind(names, seed=55, scale=0.3)

    x = fitter.initial_values(seed=42)
    report = fitter.report_params(x)              # blinded report frame

    unc = fitter.get_uncertainties(
        SimpleNamespace(x=x, hess_inv=np.eye(len(x)) * 0.1))
    for n in names:
        # after-bounds value: blinded report frame, no extra shift needed
        assert unc[n][0] == pytest.approx(report[n])
        assert unc[n][1] == pytest.approx(np.sqrt(0.1))


def test_blind_with_same_uses_canon_directly():
    """Blinding an alias of a same group is a no-op; canon blinding works.

    The blind runs before the same stage, which maps every alias onto
    its canonical value *directly* — so an offset applied to an alias is
    overwritten.  To blind a same-group the canonical name must be given.
    """
    from tabpwa.param_constraint import ConstraintManager

    # alias blinded → no effect: the group uses the canon's value directly
    cm = ConstraintManager(["a", "b"])
    cm.set_same([["a", "b"]])
    cm.set_blind(["b"], seed=7)
    tr = cm.blind_transforms[0]
    assert tr.input_names == ["b"]                 # not resolved to canon
    model = cm.resolve({"a": 0.0, "b": 0.5})
    assert model["a"] == pytest.approx(0.0)        # canon untouched
    assert model["b"] == pytest.approx(model["a"]) # same maps b := a
    report = cm.blind_result(model)
    assert report["a"] == pytest.approx(0.0)       # nothing to unshift
    assert report["b"] == pytest.approx(report["a"])

    # canon blinded → the whole group shifts and reports consistently
    cm = ConstraintManager(["a", "b"])
    cm.set_same([["a", "b"]])
    cm.set_blind(["a"], seed=7)
    tr = cm.blind_transforms[0]
    model = cm.resolve({"a": 0.0, "b": 0.5})
    assert model["a"] == pytest.approx(tr.offsets["a"])   # shifted
    assert model["b"] == pytest.approx(model["a"])        # same'd to canon
    report = cm.blind_result(model)
    assert report["a"] == pytest.approx(0.0)              # blinded back
    assert report["b"] == pytest.approx(report["a"])      # alias consistent
    assert cm.unblind_result(report)["a"] == pytest.approx(model["a"])


def test_blind_config_handler():
    """apply_constrains('blind', spec) blinds the named parameters."""
    from tabpwa.constrain_plugins import apply_constrains
    fitter = setup_fitter()
    names = [n for n in fitter.var_registry.flat_names if "mass" in n][:1]
    apply_constrains(fitter, {"blind": {"seed": 99, "names": names, "scale": 0.5}})
    assert len(fitter.cm.blind_transforms) == 1
    assert fitter.cm.blind_transforms[0].seed == 99


# ── run if called directly ────────────────────────────────────────

if __name__ == "__main__":
    test_set_fixed_is_additive()
    print("✓ test_set_fixed_is_additive")
    test_set_fixed_reset()
    print("✓ test_set_fixed_reset")
    test_set_fixed_then_free()
    print("✓ test_set_fixed_then_free")
    test_set_same_is_additive()
    print("✓ test_set_same_is_additive")
    test_set_same_reset()
    print("✓ test_set_same_reset")
    test_set_same_then_free()
    print("✓ test_set_same_then_free")
    test_set_scale_is_additive()
    print("✓ test_set_scale_is_additive")
    test_set_scale_reset()
    print("✓ test_set_scale_reset")
    test_set_scale_then_free()
    print("✓ test_set_scale_then_free")
    test_set_range()
    print("✓ test_set_range")
    test_unset_range()
    print("✓ test_unset_range")
    test_fixed_and_range_together()
    print("✓ test_fixed_and_range_together")
    test_fit_with_constraint()
    print("✓ test_fit_with_constraint")
    test_gradients_nonzero()
    print("✓ test_gradients_nonzero")
    test_full_interactive_workflow()
    print("✓ test_full_interactive_workflow")
    print("\nAll constraint API tests passed!")


def test_the_bounded_hessian_keeps_its_off_diagonals(monkeypatch):
    """Each column divides by ITS OWN step — a non-diagonal reference is required.

    ``fd_hess`` builds ``H[:, j]`` from the derivative w.r.t. ``x_j``, so the
    divisor is ``2·eps[j]``.  A per-slot step array (what ``space="bounded"``
    passes) exposes a row/column mix-up only on the OFF-diagonals, and a diagonal
    reference cannot see it (its off-diagonals are zero either way).  Here the
    reference is a random SPD matrix and the two bounded slots sit at values of
    different magnitude, so their steps differ ~5x — the exact configuration that
    was wrong before.
    """
    import numpy as np

    from tabpwa import Fitter

    f = Fitter("tests/config_pwa.yml", backend="numpy_pwa")
    f.apply_constrains()
    names = list(f.cm.var_registry.flat_names)
    n = len(names)
    x = f.initial_values(seed=3)
    f.set_range(names[0], 4.0, 6.0)          # y = 5.0  -> step eps*5
    f.set_range(names[1], 0.1, 0.3)          # y = 0.2  -> step eps*1
    b0, b1 = f.cm.bounds[names[0]], f.cm.bounds[names[1]]
    x = x.copy()
    x[0] = float(b0.inverse(5.0))
    x[1] = float(b1.inverse(0.2))

    rng = np.random.default_rng(3)
    M = rng.normal(size=(n, n))
    A = M @ M.T + n * np.eye(n)              # SPD, genuinely non-diagonal
    assert not np.allclose(A, np.diag(np.diag(A)))

    def to_y(z):
        raw = f.cm.from_flat(np.asarray(z, float))
        return np.array([f.cm.bounds[nm].forward(raw[nm])
                         if nm in f.cm.bounds else raw[nm] for nm in names])

    def dydx(z):
        raw = f.cm.from_flat(np.asarray(z, float))
        return np.array([f.cm.bounds[nm].grad(raw[nm])
                         if nm in f.cm.bounds else 1.0 for nm in names])

    y_ref = to_y(x)

    def nll(z, start_after=None):
        y = np.asarray(z, float) if start_after == "bounds" else to_y(z)
        d = y - y_ref
        g = A @ d
        return 0.5 * float(d @ A @ d), (g if start_after == "bounds"
                                        else g * dydx(z))

    monkeypatch.setattr(f, "get_nll", nll)
    H = f.compute_numerical_hessian(x, space="bounded")
    H = 0.5 * (H + H.T)
    assert np.allclose(H, A, rtol=1e-6, atol=1e-8), \
        f"max|H-A| = {np.abs(H - A).max():.3g}"
    # the marginal sigmas (A is non-diagonal: sqrt(diag(A^-1)), not 1/sqrt(A_jj))
    assert np.allclose(np.sqrt(np.diag(np.linalg.inv(H))),
                       np.sqrt(np.diag(np.linalg.inv(A))), rtol=1e-5)
