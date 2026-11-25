"""Bound transforms: the uncertainty chain, the start off a wall, the frames."""
import warnings

import numpy as np
import pytest

from tabpwa.boundary import BoundTransform, START_GRAD
from tabpwa.param_constraint import ConstraintManager


def _cm(lo, hi, name="a"):
    cm = ConstraintManager([name])
    cm.bounds.set(name, lo, hi)
    return cm


# x values INSIDE the sine's branch ([-k*pi/2, +k*pi/2] = +-0.3927 for the
# range below); past the branch the map is flat by contract (next test).
@pytest.mark.parametrize("x", [0.0, 0.1, 0.2, -0.3, 0.39])
def test_bound_error_uses_unbounded_value(x):
    """sigma_y = |dy/dx| * sigma_x, with the Jacobian at the UNBOUNDED x.

    The incoming error is the optimizer-space (unbounded) one, exactly as
    ``sqrt(diag(hess_inv))`` returns it; ``apply_bound_dict`` must propagate
    before it overwrites the value with the bounded one.
    """
    cm = _cm(0.3, 0.8)
    bt = cm.bounds["a"]
    sigma_x = 0.1
    d = {"a": x}
    e = {"a": sigma_x}
    cm.apply_bound_dict(d, e)

    assert d["a"] == pytest.approx(float(bt(x)), rel=1e-14)
    assert e["a"] == pytest.approx(float(bt.trans_err(x, sigma_x)), rel=1e-14)
    # and the old (buggy) evaluation at the bounded value is a different number
    assert e["a"] != pytest.approx(float(bt.trans_err(d["a"], sigma_x)), rel=1e-6)


def test_past_the_edge_the_value_turns_around():
    """No clamp: the sine is periodic, so the edge is a stationary TURN, not a wall.

    ``x`` stays free (the optimizer coordinate is never limited), ``y`` reaches
    ``b`` at ``x = k*pi/2`` with ``dy/dx = 0`` and comes back down afterwards —
    which is why a fit pulled outwards converges to a proper, unique minimum AT
    the edge (``F'' = |g_y|/k > 0``) instead of being pinned by a flat region.
    ``inverse`` is the principal branch, so the physical value stays
    single-valued.
    """
    cm = _cm(0.3, 0.8)
    bt = cm.bounds["a"]
    edge = bt.k * np.pi / 2
    assert float(bt(edge)) == pytest.approx(bt.b, rel=1e-14)
    assert float(bt.grad(edge)) == pytest.approx(0.0, abs=1e-15)
    # past the edge the value comes back (period 2*k*pi) and the slope flips
    assert float(bt(1.05 * edge)) < bt.b
    assert float(bt.grad(1.05 * edge)) < 0.0
    assert float(bt(2 * edge)) == pytest.approx(bt.bias, rel=1e-14)
    # the inverse is the PRINCIPAL branch, so y -> x -> y round-trips
    assert float(bt.inverse(bt.b)) == pytest.approx(edge, rel=1e-12)
    assert float(bt.inverse(bt.a)) == pytest.approx(-edge, rel=1e-12)
    for y in (bt.a + 0.1 * (bt.b - bt.a), bt.bias, bt.b - 1e-9):
        assert float(bt(bt.inverse(y))) == pytest.approx(y, abs=1e-12)


def test_bound_error_at_centre_is_untransformed():
    """At x = 0 the Jacobian is 1, so sigma_y == sigma_x (old code: ~0.08x)."""
    cm = _cm(0.3, 0.8)
    d, e = {"a": 0.0}, {"a": 0.05}
    cm.apply_bound_dict(d, e)
    assert e["a"] == pytest.approx(0.05, rel=1e-14)


def test_bound_error_unbounded_names_untouched():
    """Names without a bound (or absent from the dict) keep their error."""
    cm = _cm(0.3, 0.8)
    d = {"a": 1.0, "b": 2.0}
    e = {"a": 0.1, "b": 0.2}
    cm.apply_bound_dict(d, e)
    assert e["b"] == pytest.approx(0.2)
    # an error for a name that is not in the value dict must not raise
    cm2 = _cm(0.3, 0.8)
    d2 = {}
    e2 = {"a": 0.1}
    cm2.apply_bound_dict(d2, e2)
    assert e2["a"] == pytest.approx(0.1)


def test_a_default_outside_its_range_warns_and_starts_off_the_wall():
    """A default outside its range warns AND starts just inside the edge.

    The exact inverse would put the start ON the wall, where ``dy/dx = 0``: the
    edge is a stationary point with ``F'' > 0``, so a minimizer reports
    convergence at iteration 0 — a WRONG answer when the optimum is interior.
    The start keeps ``|dy/dx| = START_GRAD`` instead, which costs 0.25 % of the
    range's width in ``y`` (the map is quadratically flat at the wall).
    """
    cm = ConstraintManager(["a", "b"])
    cm.set_defaults({"a": 0.9})          # outside the range below
    cm.set_range("a", 0.3, 0.8)
    with pytest.warns(RuntimeWarning, match="outside its range"):
        x = cm.initial_values(seed=1)
    bt = cm.bounds["a"]
    # finite, inside the range, strictly off the wall
    assert float(x[0]) > 0.0                       # the RIGHT wall (0.9 > 0.8)
    assert abs(float(bt.grad(x[0]))) == pytest.approx(START_GRAD, rel=1e-12)
    assert bt.a < float(bt(x[0])) < bt.b
    # the inset is tiny in y (0.25 % of the width), so the clipped value is kept
    assert bt.b - float(bt(x[0])) == pytest.approx(
        bt.k * (1.0 - np.sqrt(1.0 - START_GRAD ** 2)), rel=1e-12)
    assert bt.b - float(bt(x[0])) < 5e-3 * (bt.b - bt.a)
    # an in-range default is silent and EXACT (no nudge)
    cm2 = ConstraintManager(["a"])
    cm2.set_defaults({"a": 0.5})
    cm2.set_range("a", 0.3, 0.8)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x2 = cm2.initial_values(seed=1)
    assert float(x2[0]) == pytest.approx(
        float(cm2.bounds.inverse("a", 0.5)), rel=0, abs=0)


def test_a_start_off_the_wall_lets_the_fit_reach_an_interior_optimum():
    """The pi1(1600) case: range [1.4, 1.9], default 1.3544, optimum interior.

    Started exactly on the wall the objective has a zero gradient and BFGS
    stops at iteration 0, 0.2 away from the truth.  Started at the off-wall
    point it finds the interior optimum.
    """
    from scipy.optimize import minimize

    cm = ConstraintManager(["a"])
    cm.set_defaults({"a": 1.3544})
    cm.set_range("a", 1.4, 1.9)
    with pytest.warns(RuntimeWarning, match="outside its range"):
        x0 = cm.initial_values(seed=1)
    bt = cm.bounds["a"]
    y_target = 1.62

    def f(x):
        return (np.asarray(bt(x)).item() - y_target) ** 2

    def g(x):
        return 2.0 * (np.asarray(bt(x)).item() - y_target) * np.asarray(
            bt.grad(x)).item()

    wall = float(bt.inverse(1.3544))
    assert float(g(wall)) == pytest.approx(0.0, abs=1e-15)
    with warnings.catch_warnings():
        # scipy reports "precision loss" instead of converging: that IS the bug
        warnings.simplefilter("ignore")
        stuck = minimize(f, wall, jac=g, method="BFGS")
    assert abs(float(bt(stuck.x[0])) - y_target) > 0.15      # the hazard

    res = minimize(f, x0, jac=g, method="BFGS", options={"gtol": 1e-10})
    assert abs(float(bt(res.x[0])) - y_target) < 5e-3


def test_start_inside_keeps_an_in_range_default_exact():
    """A value with a healthy slope maps through the exact inverse, unchanged."""
    bt = BoundTransform(1.4, 1.9)
    for y in (1.45, 1.65, 1.85):
        assert float(bt.start_inside(y)) == float(bt.inverse(y))
    # a value inside the flat zone (0.1 % of the width off the wall) is pulled
    # to the same slope, on the same side
    near = 1.4 + 1e-3 * 0.5
    x = float(bt.start_inside(near))
    assert x < 0.0
    assert abs(float(bt.grad(x))) == pytest.approx(START_GRAD, rel=1e-12)
    # the wall itself still maps to the exact branch end (inverse untouched)
    assert float(bt.inverse(bt.a)) == pytest.approx(-bt.k * np.pi / 2, rel=1e-12)
    assert float(bt.grad(bt.inverse(bt.a))) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("lo,hi", [(0.0, None), (None, 2.0)])
def test_start_inside_on_a_half_line(lo, hi):
    """An out-of-range default on a half-line start 5.0e-3 inside, slope 0.1."""
    bt = BoundTransform(lo, hi)
    wall = bt.lo if lo is not None else bt.hi
    beyond = wall - 0.5 if lo is not None else wall + 0.5
    x = float(bt.start_inside(beyond))
    assert float(bt.grad(x)) == pytest.approx(START_GRAD, rel=1e-12)
    if lo is not None:
        assert x > 0.0 and float(bt(x)) == pytest.approx(
            bt.lo + np.sqrt(1.0 + x * x) - 1.0, rel=1e-12)
    else:
        assert x < 0.0 and float(bt(x)) == pytest.approx(
            bt.hi - np.sqrt(1.0 + x * x) + 1.0, rel=1e-12)
    assert abs(float(bt(x)) - wall) == pytest.approx(
        (1.0 - np.sqrt(1.0 - START_GRAD ** 2)) / np.sqrt(1.0 - START_GRAD ** 2),
        rel=1e-12)
    # in-range values are exact
    inside = 0.7 if lo is not None else 1.3
    assert float(bt.start_inside(inside)) == float(bt.inverse(inside))


# ── the pipeline's forward and backward cuts (start_after) ────────────

def test_resolve_start_after_bounds_is_the_second_half():
    """``stop_after='bounds'`` then ``start_after='bounds'`` == the full run.

    That identity is what makes the bounded frame a first-class variable: the
    value dict after the bounds stage can be fed straight back in and the rest of
    the pipeline (blind → fixed → same → scale → transforms → custom → defaults)
    is unchanged.
    """
    cm = ConstraintManager(["a", "b", "c"])
    cm.set_defaults({"d": 7.0})
    cm.set_range("a", 0.3, 0.8)
    cm.set_same([("b", "a")])
    cm.set_scale({"a": (2.0, 1.0)})
    raw = {"a": 0.2, "b": 0.0, "c": 1.5}
    full = cm.resolve(raw)
    half = cm.resolve(raw, stop_after="bounds")
    assert cm.resolve(half, start_after="bounds") == full
    # ... and the flat entry agrees too (a plain registry: the two halves are
    # ``flat_resolve`` and ``resolve``, since the former returns a dict)
    cm2 = ConstraintManager(["a", "b"])
    cm2.set_defaults({"c": 3.0})
    cm2.set_range("a", 0.3, 0.8)
    raw2 = np.array([0.2, 1.5])
    assert cm2.resolve(cm2.flat_resolve(raw2, stop_after="bounds"),
                       start_after="bounds") == cm2.flat_resolve(raw2)
    # an inverted window is a caller bug, not an empty dict
    with pytest.raises(ValueError, match="empty stage window"):
        cm.resolve(raw, start_after="scale", stop_after="bounds")


def test_the_bounded_gradient_is_the_x_gradient_over_the_jacobian():
    """``start_after='bounds'`` drops the bound Jacobian: ``dNLL/dy``."""
    from tabpwa import Fitter

    f = Fitter("tests/config_pwa.yml", backend="numpy_pwa")
    f.apply_constrains()
    x = f.initial_values(seed=3)
    name = f.cm.var_registry.flat_names[1]
    y0 = float(f.cm.from_flat(x)[name])
    f.set_range(name, y0 - 0.4, y0 + 0.4)
    bt = f.cm.bounds[name]
    i = f.cm.var_registry.flat_names.index(name)
    x = x.copy()
    x[i] = 0.0                              # the branch centre: well inside
    f.set_phsp(f.generate_phsp(40, seed=1))
    f.set_data(f.generate_toy(x, 20, seed=2))

    gx = f.get_nll(x)[1]
    y = f._bounded_coords(x)[0]
    gy = f.get_nll(y, start_after="bounds")[1]
    for k, nm in enumerate(f.cm.var_registry.flat_names):
        if nm in f.cm.bounds:
            jac = float(f.cm.bounds[nm].grad(f.cm.from_flat(x)[nm]))
            assert gy[k] == pytest.approx(gx[k] / jac, rel=1e-9, abs=1e-12)
        else:
            assert gy[k] == pytest.approx(gx[k], rel=1e-12, abs=1e-14)
    # the interrupt checkpoint is an x vector: a y call must not overwrite it
    before = np.asarray(f._last_xk).copy()
    f.get_nll(y, start_after="bounds")
    assert np.array_equal(f._last_xk, before)
    # and a y OUTSIDE the range is refused rather than extrapolated
    bad = np.array(y, float)
    bad[i] = bt.b + 0.1
    with pytest.raises(ValueError, match="outside its range"):
        f.get_nll(bad, start_after="bounds")


# ── one-sided (half-line) bounds ──────────────────────────────────────

def test_the_half_line_bound_is_unit_scaled_and_closed_at_the_wall():
    """``y = lo + sqrt(x²+1) − 1``: wall at the FINITE ``x = 0``, no scale.

    ``dy/dx`` is 0 at the wall and → 1 far from it, so ``|x| ~ 1`` is the
    quadratic region for EVERY parameter (a scale would move that per parameter,
    which is why there is none).  The open side is ``+inf`` so the callers that
    clip inside a range need no special case.
    """
    from tabpwa.boundary import BoundTransform

    bt = BoundTransform(0.0, None)          # kind='lower'
    assert bt.a == 0.0 and bt.b == float("inf")
    assert float(bt(0.0)) == pytest.approx(0.0)          # the wall
    assert float(bt(-2.0)) == pytest.approx(0.0)         # flat past it
    assert float(bt.grad(0.0)) == pytest.approx(0.0)
    assert float(bt.grad(50.0)) == pytest.approx(1.0, abs=1e-3)
    xs = np.array([0.0, 0.2, 1.0, 7.0])
    assert np.allclose(bt.inverse(bt(xs)), xs, rtol=1e-10, atol=1e-12)
    assert float(bt.inverse(0.0)) == pytest.approx(0.0)
    assert bt.to_dict() == {"low": 0.0, "high": None}

    up = BoundTransform(None, 2.0)          # kind='upper'
    assert up.a == float("-inf") and up.b == 2.0
    assert float(up(0.0)) == pytest.approx(2.0)
    assert float(up.grad(-50.0)) == pytest.approx(1.0, abs=1e-3)
    xs = np.array([0.0, -0.5, -3.0])
    assert np.allclose(up.inverse(up(xs)), xs, rtol=1e-10, atol=1e-12)


def test_a_half_bound_round_trips_and_curves_in_the_bounded_frame(monkeypatch):
    """Through the Fitter: init on the branch, JSON with ``null``, and the
    bounded Hessian using the one-sided clip (no upper wall)."""
    import json
    import tempfile
    import os
    from tabpwa import Fitter

    f = Fitter("tests/config_pwa.yml", backend="numpy_pwa")
    f.apply_constrains()
    names = list(f.cm.var_registry.flat_names)
    j = 1
    f.set_range(names[j], 0.0, None)                  # lower bound only
    assert f.cm.bounds[names[j]].kind == "lower"
    assert f.cm.bounds[names[j]].b == float("inf")
    assert f.cm.bounds.to_dict()[names[j]] == {"low": 0.0, "high": None}

    # the constrains round-trip keeps the open side open
    path = os.path.join(tempfile.mkdtemp(), "c.json")
    f.save_constraints(path)
    g = Fitter("tests/config_pwa.yml", backend="numpy_pwa")
    g.apply_constrains()
    g.load_constraints(path)
    assert g.cm.bounds[names[j]].kind == "lower"

    x0 = f.initial_values(seed=5)
    assert float(f.cm.from_flat(x0)[names[j]]) >= 0.0   # the branch draw
    f.set_phsp(f.generate_phsp(40, seed=1))
    f.set_data(f.generate_toy(x0, 20, seed=2))

    # a quadratic in the PHYSICAL value, with the pipeline's own cuts
    bt = f.cm.bounds[names[j]]
    A = np.diag(np.linspace(2.0, 4.0, len(x0)))

    def to_y(z):
        raw = f.cm.from_flat(np.asarray(z, float))
        return np.array([f.cm.bounds[nm].forward(raw[nm])
                         if nm in f.cm.bounds else raw[nm] for nm in names])

    def dydx(z):
        raw = f.cm.from_flat(np.asarray(z, float))
        return np.array([f.cm.bounds[nm].grad(raw[nm])
                         if nm in f.cm.bounds else 1.0 for nm in names])

    x_mid = x0.copy()
    x_mid[j] = 1.0                                     # y = 0.414, in the bulk
    y_ref = to_y(x_mid)

    def nll(z, start_after=None):
        y = z if start_after == "bounds" else to_y(z)
        d = y - y_ref
        g = A @ d
        return 0.5 * float(d @ A @ d), (g if start_after == "bounds"
                                        else g * dydx(z))

    monkeypatch.setattr(f, "get_nll", nll)
    H = f.compute_numerical_hessian(x_mid, space="bounded")
    assert np.allclose(H, A, rtol=1e-6, atol=1e-8)


def test_a_binding_range_is_reported_at_the_exact_edge():
    """A binding soft range is a WARNING, and the value IS the edge.

    The sine maps past-the-edge input to exactly the bound (no ``tan(pi/2)/c``
    precision problem), so the report is an equality — and no sigma exists for
    that slot, which is why ``fit`` says so instead of letting a
    ``success=True`` result be quoted with an error bar.
    """
    import warnings

    from tabpwa import Fitter

    f = Fitter("tests/config_pwa.yml", backend="numpy_pwa")
    f.apply_constrains()
    names = list(f.cm.var_registry.flat_names)
    x0 = f.initial_values(seed=1)
    f.set_phsp(f.generate_phsp(40, seed=1))
    f.set_data(f.generate_toy(x0, 30, seed=2))

    r0 = f.fit(x0, maxiter=400, disp=False)          # no ranges at all
    assert f.boundary_report(r0) == []

    j = int(np.argmax(np.abs(np.diag(r0.hess_inv))))
    s0 = float(np.sqrt(np.diag(r0.hess_inv))[j])
    yb = float(f.cm.flat_resolve(r0.x)[names[j]])
    lo, hi = yb - 6.0 * s0, yb - 0.5 * s0            # the free optimum sits outside
    f.set_range(names[j], lo, hi)
    bt = f.cm.bounds[names[j]]
    x_in = f.initial_values(seed=7)
    x_in[j] = bt.inverse(hi - 0.3 * (hi - lo))
    with pytest.warns(RuntimeWarning, match="sit ON a range edge"):
        r = f.fit(x_in, maxiter=400, disp=False)
    rep = f.boundary_report(r)
    assert [n for n, *_ in rep] == [names[j]]
    _n, val, rep_lo, rep_hi = rep[0]
    # the sine makes the edge a stationary TURN, so the fit converges TO it
    # within its own tolerance (measured y* - b ~ -2e-7), not bit-exactly
    assert rep_lo < val <= rep_hi
    assert rep_hi - val <= 1e-6 * (rep_hi - rep_lo)
    # a LOADED result never went through fit(): the sigma report warns too
    with pytest.warns(RuntimeWarning, match="sit ON a range edge"):
        f.get_uncertainties(r)

    # a range that does NOT bind stays silent
    f2 = Fitter("tests/config_pwa.yml", backend="numpy_pwa")
    f2.apply_constrains()
    f2.set_phsp(f.generate_phsp(40, seed=1))
    f2.set_data(f.generate_toy(x0, 30, seed=2))
    f2.set_range(names[j], yb - 20.0 * s0, yb + 20.0 * s0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r2 = f2.fit(x_in, maxiter=400, disp=False)
    assert not [w for w in caught if "range edge" in str(w.message)]
    assert f2.boundary_report(r2) == []
