"""Forward-mode dual scalars (:mod:`tabpwa.fwd`) — arithmetic vs finite differences."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tabpwa.fwd import Fwd  # noqa: E402

CONFIG = os.path.join(ROOT, "tests", "config_pwa.yml")


def _fd(fn, vals, i, h=1e-6):
    lo, hi = list(vals), list(vals)
    lo[i] -= h
    hi[i] += h
    out = (fn(*hi) - fn(*lo)) / (2 * h)
    return out.value if isinstance(out, Fwd) else out


def _seed(i, n, value=0.0):
    g = np.zeros(n)
    g[i] = 1.0
    return Fwd(value, g)


def test_arithmetic_matches_fd():
    vals = (2.3, 1.1)

    def fn(a, b):
        return ((a * 2.0 + b) / (a - b + 5.0)
                + np.sqrt(a + b) - 3.0 * b ** 2 + b / a)

    a, b = _seed(0, 2, vals[0]), _seed(1, 2, vals[1])
    e = ((a * 2.0 + b) / (a - b + 5.0)
         + (a + b).sqrt() - 3.0 * (b ** 2) + b / a)

    assert e.value == pytest.approx(fn(*vals), rel=1e-14)
    fd = np.array([_fd(fn, vals, 0), _fd(fn, vals, 1)])
    np.testing.assert_allclose(e.grad, fd, rtol=1e-8, atol=1e-10)


def test_scalar_ops_both_sides():
    x = _seed(0, 1, 4.0)
    assert (x + 2.0).value == 6.0 and (2.0 + x).value == 6.0
    assert (x - 1.0).value == 3.0 and (1.0 - x).value == -3.0
    assert (x * 2.0).grad.tolist() == [2.0] and (2.0 * x).grad.tolist() == [2.0]
    assert (x / 2.0).grad.tolist() == [0.5]
    assert (2.0 / x).grad.tolist() == [pytest.approx(-2.0 / 16.0)]
    assert (-x).value == -4.0 and (-x).grad.tolist() == [-1.0]
    assert x.sqrt().grad.tolist() == [pytest.approx(0.25)]
    assert x.log().grad.tolist() == [0.25]


def test_linearity_of_accumulation():
    """Adding contributions then scaling == scaling each then adding."""
    acc = Fwd(0.0, np.zeros(2))
    for v, gi in ((3.0, 0), (2.0, 1), (0.5, 0)):
        acc = acc + _seed(gi, 2, v)
    one = Fwd(5.5, np.array([2.0, 1.0]))
    assert acc.value == pytest.approx(one.value)
    np.testing.assert_allclose(acc.grad, one.grad)


def test_sigma_matches_quadratic_form():
    r = Fwd(0.25, np.array([2.0, -1.0, 0.5]))
    cov = np.array([[0.04, 0.01, 0.0],
                    [0.01, 0.09, 0.0],
                    [0.0, 0.0, 0.25]])
    assert r.sigma(cov) == pytest.approx(np.sqrt(r.grad @ cov @ r.grad))


def test_parameter_space_guard():
    a = Fwd(1.0, np.ones(2))
    b = Fwd(1.0, np.ones(1))
    for op in (lambda: a + b, lambda: a - b, lambda: a * b, lambda: a / b):
        with pytest.raises(ValueError, match="different parameter spaces"):
            op()


def test_ratio_gradient_is_the_quotient_rule():
    """The fraction case: R = num/den carries both gradients exactly."""
    num = Fwd(3.0, np.array([1.0, 0.0]))
    den = Fwd(12.0, np.array([0.0, 2.0]))
    r = num / den
    assert r.value == pytest.approx(0.25)
    np.testing.assert_allclose(r.grad, np.array([1 / 12, -3.0 * 2 / 144]))


# ── Fitter-side constructors (the fixed space is the fitter's) ─────────
def _fitter():
    from tabpwa import Fitter

    f = Fitter(CONFIG, backend="numpy_pwa")
    f.apply_constrains()
    return f


def test_fitter_names_and_seed():
    f = _fitter()
    names = f.fwd_names()
    assert names and names == list(f.free_param_names())

    x = f.fwd_parameter(names[0], value=3.0)
    assert x.value == 3.0
    assert x.grad.shape == (len(names),)
    assert x.grad[0] == 1.0 and x.grad.sum() == 1.0

    with pytest.raises(KeyError, match="not a free parameter"):
        f.fwd_parameter("no_such_parameter")


def test_fitter_fwd_of_maps_names_and_ignores_unknown():
    f = _fitter()
    names = f.fwd_names()
    y = f.fwd_of(5.0, {names[-1]: 2.0, "not_a_parameter": 9.0})
    expect = np.zeros(len(names))
    expect[-1] = 2.0
    np.testing.assert_allclose(y.grad, expect)

    # the fixed space makes dual arithmetic directly usable
    a = f.fwd_parameter(names[0], value=3.0)
    r = a / y
    assert r.value == pytest.approx(3.0 / 5.0)
    np.testing.assert_allclose(r.grad, [1.0 / 5.0] + [0.0] * (len(names) - 2)
                               + [-3.0 * 2.0 / 25.0])


def test_fitter_fwd_of_rejects_raw_kernel_blocks():
    """Raw kernel grads must be chained first — fwd_of says so, not silently 0."""
    f = _fitter()
    with pytest.raises(ValueError, match="not raw kernel blocks"):
        f.fwd_of(1.0, {"ck": np.ones(2)})


# ── FD Jacobian from the fitter -> Fwd rows ────────────────────────────
def test_fd_jacobian_matches_analytic():
    from tabpwa.fwd import rows

    f = _fitter()
    x = f.initial_values(seed=3)
    resolved = f.report_params(x)
    names = [n for n in f.fwd_names() if n in resolved]
    a, b = names[0], names[1] if len(names) > 1 else names[0]

    def fn(phys):
        va, vb = float(phys[a]), float(phys[b])
        return [va * va, 3.0 * vb, va * vb]

    vals, J = f.fd_jacobian(fn, x=x)
    scale = np.abs(J).max() if np.abs(J).max() > 0 else 1.0
    assert np.abs(J).max() > 0

    def analytic(row, pname):
        va, vb = float(resolved[a]), float(resolved[b])
        if row == 0:
            return 2.0 * va if pname == a else 0.0
        if row == 1:
            return 3.0 if pname == b else 0.0
        return vb if pname == a else (va if pname == b else 0.0)

    for r in range(3):
        for j, pname in enumerate(names):
            assert J[r, j] == pytest.approx(analytic(r, pname), rel=1e-5,
                                            abs=1e-8 * scale)

    # rows -> duals whose sigma is the quadratic form of that row
    cov = np.eye(len(names)) * 0.01
    for i, dual in enumerate(rows(vals, J)):
        assert dual.value == pytest.approx(vals[i])
        assert dual.sigma(cov) == pytest.approx(
            np.sqrt(J[i] @ cov @ J[i]))
    assert len(rows(vals, J)) == 3


def test_rows_rejects_bad_shape():
    from tabpwa.fwd import rows

    with pytest.raises(ValueError, match="does not match"):
        rows(np.zeros(3), np.zeros((2, 4)))


def test_fd_jacobian_rejects_complex_observable():
    """Fwd is real: a complex observable must be handed over as Re/Im/|y|^2."""
    f = _fitter()
    x = f.initial_values(seed=3)
    with pytest.raises(ValueError, match="observable is complex"):
        f.fd_jacobian(lambda p: np.array([1 + 2j]), x=x)


def test_fd_jacobian_passes_args_and_kwargs():
    """No lambda needed: extra arguments are forwarded to fn."""
    from tabpwa.fwd import rows

    f = _fitter()
    x = f.initial_values(seed=3)
    resolved = f.report_params(x)
    name = [n for n in f.fwd_names() if n in resolved][0]

    def fn(phys, grid, scale=1.0):
        return [scale * float(phys[name]) * g for g in grid]

    grid = [1.0, 2.0, 4.0]
    plain = f.fd_jacobian(lambda p: fn(p, grid, scale=2.0), x=x)
    direct = f.fd_jacobian(fn, (grid,), {"scale": 2.0}, x=x)

    np.testing.assert_allclose(direct[0], plain[0])
    np.testing.assert_allclose(direct[1], plain[1])
    np.testing.assert_allclose(rows(*direct)[1].value, 2.0 * resolved[name] * 2.0)


# ── elementary functions and polar <-> rectangular ─────────────────────
def test_trig_and_exp_match_fd():
    from tabpwa.fwd import cos, exp, sin

    vals = (0.7, -0.4)
    x, y = Fwd(vals[0], np.array([1.0, 0.0])), Fwd(vals[1], np.array([0.0, 1.0]))

    assert sin(x).value == pytest.approx(np.sin(vals[0]))
    np.testing.assert_allclose(sin(x).grad, [np.cos(vals[0]), 0.0])
    np.testing.assert_allclose(cos(x).grad, [-np.sin(vals[0]), 0.0])
    np.testing.assert_allclose(exp(y).grad, [0.0, np.exp(vals[1])])

    def fn(a, b):
        return np.sin(a) * np.cos(b) + np.exp(a - b)

    e = sin(x) * cos(y) + exp(x - y)
    fd = np.array([_fd(fn, vals, 0), _fd(fn, vals, 1)])
    np.testing.assert_allclose(e.grad, fd, rtol=1e-7, atol=1e-9)


def test_atan2_matches_fd():
    from tabpwa.fwd import atan2

    vals = (0.7, -0.4)                     # (x, y)
    x, y = Fwd(vals[0], np.array([1.0, 0.0])), Fwd(vals[1], np.array([0.0, 1.0]))
    t = atan2(y, x)
    assert t.value == pytest.approx(np.arctan2(vals[1], vals[0]))
    fd = np.array([_fd(lambda a, b: np.arctan2(b, a), vals, 0),
                   _fd(lambda a, b: np.arctan2(b, a), vals, 1)])
    np.testing.assert_allclose(t.grad, fd, rtol=1e-7, atol=1e-9)


def test_polar_rectangular_round_trip():
    from tabpwa.fwd import magnitude, polar, reim

    mag = Fwd(2.0, np.array([1.0, 0.0]))
    phi = Fwd(0.3, np.array([0.0, 1.0]))
    re, im = reim(mag, phi)
    assert re.value == pytest.approx(2.0 * np.cos(0.3))
    assert im.value == pytest.approx(2.0 * np.sin(0.3))
    # re^2 + im^2 recovers the magnitude squared, gradient included
    r2 = re * re + im * im
    assert r2.value == pytest.approx(4.0)
    np.testing.assert_allclose(r2.grad, [4.0, 0.0], atol=1e-12)
    # and back again
    m2, p2 = polar(re, im)
    assert m2.value == pytest.approx(2.0) and p2.value == pytest.approx(0.3)
    np.testing.assert_allclose(m2.grad, [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(p2.grad, [0.0, 1.0], atol=1e-12)
    assert magnitude(3.0, 4.0).value == pytest.approx(5.0)


def test_numbers_become_zero_tangent_duals():
    """Every argument is converted to Fwd; numbers get an empty tangent."""
    from tabpwa.fwd import atan2, cos, exp, reim, sin

    for out, expect in ((sin(0.7), np.sin(0.7)),
                        (cos(0.7), np.cos(0.7)),
                        (exp(0.7), np.exp(0.7)),
                        (atan2(1.0, 2.0), np.arctan2(1.0, 2.0))):
        assert isinstance(out, Fwd)
        assert out.grad.ndim == 0            # scalar tangent, numpy-broadcast
        assert out.value == pytest.approx(expect)

    re, im = reim(2.0, 0.3)
    assert re.grad.ndim == 0 and im.grad.ndim == 0
    assert re.value == pytest.approx(2.0 * np.cos(0.3))
    assert im.value == pytest.approx(2.0 * np.sin(0.3))


def test_abs_of_a_real_dual_and_of_a_pair():
    from tabpwa.fwd import abs as fabs
    from tabpwa.fwd import reim

    x = Fwd(-1.5, np.array([1.0, 0.0]))
    assert abs(x).value == 1.5                     # __abs__
    np.testing.assert_allclose(abs(x).grad, [-1.0, 0.0])
    assert fabs(x).value == 1.5
    assert fabs(-2.5).value == pytest.approx(2.5)  # plain number -> dual

    re, im = reim(Fwd(2.0, np.array([1.0, 0.0])), Fwd(0.3, np.array([0.0, 1.0])))
    mag = fabs(re, im)                             # complex modulus of the pair
    assert mag.value == pytest.approx(2.0)
    np.testing.assert_allclose(mag.grad, [1.0, 0.0], atol=1e-12)


def test_mod_matches_fd_and_numbers():
    from tabpwa.fwd import mod

    g, h = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    a, b = Fwd(7.3, g), Fwd(2.0, h)
    r = mod(a, b)
    assert r.value == pytest.approx(7.3 % 2.0)
    np.testing.assert_allclose(r.grad, [1.0, -3.0])          # -floor(a/b)

    fd = np.array([_fd(lambda x, y: x % y, (7.3, 2.0), 0),
                   _fd(lambda x, y: x % y, (7.3, 2.0), 1)])
    np.testing.assert_allclose(r.grad, fd, rtol=1e-6, atol=1e-8)

    np.testing.assert_allclose((a % 2.5).grad, [1.0, 0.0])
    rem = 10.0 % b
    assert rem.value == pytest.approx(0.0)
    np.testing.assert_allclose(rem.grad, [0.0, -5.0])        # -floor(10/2)
    assert mod(7.3, 2.0).value == pytest.approx(7.3 % 2.0)


def test_dual_decorator_for_user_functions():
    """`@dual` runs one body for numbers and for duals."""
    from tabpwa.fwd import dual

    @dual
    def lorentz(m, m0, w):
        return w * w / ((m * m - m0 * m0) ** 2 + w * w)

    @dual
    def pair(x):
        return (2.0 * x, x * x)

    # numbers in -> numbers out (no second implementation)
    def plain(m, m0, w):
        return w * w / ((m * m - m0 * m0) ** 2 + w * w)

    assert lorentz(0.77, 0.775, 0.15).value == pytest.approx(
        plain(0.77, 0.775, 0.15))
    a0, b0 = pair(3.0)
    assert (a0.value, b0.value) == (pytest.approx(6.0), pytest.approx(9.0))
    assert a0.grad.ndim == 0 and b0.grad.ndim == 0

    # one dual promotes every numeric argument into its space
    w = Fwd(0.15, np.array([0.0, 1.0]))
    r = lorentz(0.77, 0.775, w)
    assert isinstance(r, Fwd)
    fd = _fd(lambda ww: lorentz(0.77, 0.775, ww), (0.15,), 0)
    np.testing.assert_allclose(r.grad, [0.0, fd], rtol=1e-6, atol=1e-10)

    # tuple results are mapped elementwise
    a, b = pair(Fwd(3.0, np.array([1.0])))
    assert (a.value, b.value) == (6.0, 9.0)
    assert (a.grad[0], b.grad[0]) == (2.0, 6.0)


def test_covariance_and_correlation():
    from tabpwa.fwd import corr, corr_matrix, cov

    C = np.array([[0.04, 0.01], [0.01, 0.09]])
    a = Fwd(1.0, np.array([1.0, 0.0]))
    b = Fwd(2.0, np.array([1.0, 1.0]))

    assert a.cov(a, C) == pytest.approx(0.04)
    assert a.cov(b, C) == pytest.approx(0.05)
    assert b.cov(a, C) == pytest.approx(0.05)
    assert cov(a, b, C) == pytest.approx(0.05)

    sa, sb = np.sqrt(0.04), np.sqrt(0.15)   # [1,0] and [1,1] diagonals
    assert a.correlation(b, C) == pytest.approx(0.05 / (sa * sb))
    assert a.corr(b, C) == pytest.approx(a.correlation(b, C))
    assert corr(a, b, C) == pytest.approx(a.correlation(b, C))
    assert abs(a.correlation(b, C)) <= 1.0
    assert a.correlation(a, C) == pytest.approx(1.0)

    # a constant has no variance -> no correlation
    assert a.correlation(Fwd(3.0, 0.0), C) == 0.0
    assert Fwd(3.0, 0.0).correlation(a, C) == 0.0

    M = corr_matrix([a, b], C)
    assert M.shape == (2, 2)
    np.testing.assert_allclose(np.diag(M), [1.0, 1.0])
    assert M[0, 1] == pytest.approx(M[1, 0]) == pytest.approx(
        a.correlation(b, C))

    with pytest.raises(ValueError, match="different parameter spaces"):
        a.cov(Fwd(1.0, np.ones(3)), C)


# ── complex duals: a pair of real duals ───────────────────────────────
def test_complex_pair_algebra_matches_fd():
    from tabpwa.fwd import C

    t = Fwd(0.7, np.array([1.0]))
    z = (C(t, t * t) / C(2.0, -t)) + C(1.0, 1.0) * C(0.0, t).exp()

    def f(x):
        return (x + 1j * x * x) / (2 - 1j * x) + (1 + 1j) * np.exp(1j * x)

    ref = f(0.7)
    assert z.value == pytest.approx(ref)
    h = 1e-6
    assert z.re.grad[0] == pytest.approx(
        (f(0.7 + h).real - f(0.7 - h).real) / (2 * h), rel=1e-6)
    assert z.im.grad[0] == pytest.approx(
        (f(0.7 + h).imag - f(0.7 - h).imag) / (2 * h), rel=1e-6)

    # modulus / phase are real duals
    assert abs(z).value == pytest.approx(abs(ref))
    assert z.arg().value == pytest.approx(np.angle(ref))
    assert z.abs().sigma(np.eye(1)) >= 0.0

    # holomorphic round trips
    assert z.log().exp().value == pytest.approx(z.value)
    assert (z.sqrt() ** 2).value == pytest.approx(z.value)
    assert (z ** 2).value == pytest.approx(ref ** 2)

    # conjugation: z·conj(z) = |z|², purely real
    zz = z.conj() * z
    assert zz.value.imag == pytest.approx(0.0, abs=1e-12)
    assert zz.value.real == pytest.approx(abs(ref) ** 2)

    # constructors
    assert C.of(1 + 2j).value == pytest.approx(1 + 2j)
    c = C.from_polar(Fwd(2.0, np.array([1.0])), Fwd(0.3, np.array([0.0])))
    assert c.value == pytest.approx(2.0 * np.exp(0.3j))
    assert c.re.grad[0] == pytest.approx(np.cos(0.3))


def test_erf_gamma_loggamma_match_fd():
    from scipy.special import digamma, erf as s_erf, gamma as s_gamma
    from tabpwa.fwd import erf, gamma, loggamma

    x = Fwd(0.7, np.array([1.0, 0.0]))
    y = Fwd(2.5, np.array([0.0, 1.0]))

    assert erf(x).value == pytest.approx(s_erf(0.7))
    assert erf(x).grad[0] == pytest.approx(
        (2.0 / np.sqrt(np.pi)) * np.exp(-0.49))
    assert erf(x).grad[1] == 0.0

    assert gamma(y).value == pytest.approx(s_gamma(2.5))
    fd = _fd(lambda v: s_gamma(v), (2.5,), 0)
    assert gamma(y).grad[1] == pytest.approx(fd, rel=1e-6)

    assert gamma(5.0).value == pytest.approx(24.0)     # (n-1)!
    assert loggamma(y).value == pytest.approx(np.log(s_gamma(2.5)))
    assert loggamma(y).grad[1] == pytest.approx(digamma(2.5))
