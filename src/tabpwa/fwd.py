"""Forward-mode dual scalars over a fixed parameter space.

A :class:`Fwd` is a scalar `value` plus `grad`, the derivative of that
scalar with respect to a **fixed** list of parameters.  It exists so that a
quantity built from accumulated sums — a fraction, an asymmetry, a ratio of
yields — carries its exact gradient through ordinary arithmetic, without a
tape and without hand-written quotient rules.

There is no layout object: the parameter space *is* the gradient vector's
index space, fixed by the caller.  :class:`tabpwa.fitter.Fitter` owns it
(``fwd_names()`` / ``fwd_parameter(name)`` / ``fwd_of(value, grad_dict, x)``),
so every ``Fwd`` in one fit shares the same ordering; the operators check
the vector length, so mixing two spaces raises instead of broadcasting.
``fwd_of`` takes **one** point argument (``x``) and derives the resolved frame
from it, so a value and its gradient can never come from different points —
which is what makes a ratio or a covariance of duals meaningful.

Scope, deliberately narrow:

* **linear accumulation** — what the kernel already returns as
  ``(value, grads)``; the per-event stage only ever *adds*;
* **closed-form post-processing** on the accumulated scalars: ``+ - * /
  pow sqrt log``;
* *not* a general autodiff: no per-event arrays, no graph, no reverse pass,
  no re-evaluation of the model.  A quantity that needs the model at
  changed parameters is a kernel call (analytic gradient) or finite
  differences, not a ``Fwd``.

Cost is ``O(n_params)`` per operation and in practice numpy-overhead bound
(~1–3 µs per op), not per event.

Example::

    a = fitter.fwd_of(total_num, grad_num, x=x)   # value + x-space gradient
    b = fitter.fwd_of(total_den, grad_den, x=x)
    r = a / b                                     # value + exact gradient
    sigma = r.sigma(hess_inv)                     # sqrt(g C g^T)

``x`` is the flat free-parameter vector the kernel ran at: a ``Fwd``'s
``grad`` is a gradient over *that* space (``Fitter.fwd_names()`` is its
ordering), because that is the space a fit covariance describes.  The
kernel's raw gradient is by resolved name, so ``fwd_of`` applies the
constraint chain (``cm.full_gradient``) once, at the point it is given.
"""

import functools

import numpy as np

__all__ = ["Fwd", "rows", "sin", "cos", "exp", "atan2",
           "polar", "reim", "magnitude", "abs", "mod", "log", "sqrt",
           "erf", "gamma", "loggamma",
           "dual", "cov", "corr", "corr_matrix", "batch_sum",
           "data_split",
           "C"]

_NUM = (int, float, np.integer, np.floating)


def cov(a, b, cov_matrix):
    """Covariance of two duals (``a.cov(b, C)``)."""
    return a.cov(b, cov_matrix)


def corr(a, b, cov_matrix):
    """Correlation coefficient of two duals (``a.correlation(b, C)``)."""
    return a.correlation(b, cov_matrix)


def corr_matrix(duals, cov_matrix):
    """Full correlation matrix of a list of duals (a band's points, say)."""
    n = len(duals)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = duals[i].correlation(duals[j], cov_matrix)
    return out


def _tree_add_fwd(a, b):
    """Structural ``a + b`` for ``Fwd``/number leaves in dicts or sequences.

    Mirrors ``tfpwa.data.batch_sum``'s "first result seeds, the rest are
    added" — but for a pytree of duals.
    """
    if isinstance(a, (Fwd, _NUM)) or isinstance(b, (Fwd, _NUM)):
        return a + b
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            raise ValueError(f"batch_sum: structure differs: {sorted(a)} vs "
                             f"{sorted(b)}")
        return {k: _tree_add_fwd(a[k], b[k]) for k in a}
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise ValueError("batch_sum: sequences differ in length")
        items = [_tree_add_fwd(x, y) for x, y in zip(a, b)]
        return type(a)(*items) if hasattr(a, "_fields") else type(a)(items)
    # any other leaf pair (arrays, plain numbers) adds as itself — that is
    # what lets ``sum_stream`` carry values-only contributions, not just duals
    try:
        return a + b
    except TypeError:
        raise TypeError(f"batch_sum: cannot add {type(a).__name__} and "
                        f"{type(b).__name__}") from None


def _tree_len(tree):
    """Event count of a data pytree: the first **non-scalar** array leaf's
    ``shape[0]``.

    Scalar leaves (a per-dataset constant) carry no event axis, so they are
    skipped rather than indexed.
    """
    if isinstance(tree, dict):
        leaves = tree.values()
    elif isinstance(tree, (list, tuple)):
        leaves = tree
    else:
        leaves = (tree,)
    for leaf in leaves:
        if isinstance(leaf, (dict, list, tuple)):
            try:
                return _tree_len(leaf)
            except (IndexError, TypeError):
                continue
        arr = np.asarray(leaf)
        if arr.ndim == 0:
            continue                      # a constant, not an event axis
        return int(arr.shape[0])
    raise ValueError("_tree_len: no array leaf with an event axis")


def _tree_slice(tree, sl, axis=0):
    """Slice every array leaf of *tree* on *axis* (structure preserved)."""
    if isinstance(tree, dict):
        return {k: _tree_slice(v, sl, axis) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        items = [_tree_slice(v, sl, axis) for v in tree]
        return type(tree)(*items) if hasattr(tree, "_fields") else type(tree)(items)
    arr = np.asarray(tree)
    if arr.ndim == 0:
        return tree
    idx = [slice(None)] * arr.ndim
    idx[axis] = sl
    return arr[tuple(idx)]


def data_split(tree, batch, axis=0):
    """Yield contiguous chunks of *batch* events — ``tfpwa.data.data_split``.

    Walks a pytree of arrays and slices every array leaf on *axis*; scalar
    leaves pass through unchanged, and the structure is preserved.  The
    companion of :func:`batch_sum` (and what ``Fitter.sum_stream`` uses):

        for chunk in data_split(arrays, 50_000):
            ...
    """
    n = _tree_len(tree)
    if n == 0:
        return                      # empty sample: no chunks (not a bad batch)
    step = n if batch is None else int(batch)
    if step <= 0:
        raise ValueError(f"data_split: batch must be > 0, got {batch}")
    for lo in range(0, n, step):
        yield _tree_slice(tree, slice(lo, lo + step), axis)


def batch_sum(fn, chunks):
    """Accumulate ``fn(chunk)`` over *chunks* (the ``tfpwa.data.batch_sum``
    pattern: the first result seeds, every later one is added).

    Works for a single ``Fwd`` and for any pytree of them, so a chunked
    evaluation needs no reduction bookkeeping::

        total = batch_sum(lambda h: fwd_of(*compute(h), x=x), chunks)
    """
    total = None
    seen = False
    for chunk in chunks:
        out = fn(chunk)
        if not seen:
            total, seen = out, True
        else:
            total = _tree_add_fwd(total, out)
    if not seen:
        raise ValueError("batch_sum: no chunks")
    return total


def rows(values, jacobian):
    """One ``Fwd`` per observable from ``(values, J)``.

    ``jacobian`` has shape ``(n_observables, n_params)`` — exactly what
    :meth:`tabpwa.fitter.Fitter.fd_jacobian` returns — so a curve of values
    with their parameter gradients becomes a list of duals::

        y, J = f.fd_jacobian(fn, result)
        for dual in rows(y, J):
            print(dual.value, dual.sigma(cov))
    """
    values = np.atleast_1d(np.asarray(values, dtype=float))
    J = np.asarray(jacobian, dtype=float)
    if J.ndim != 2 or J.shape[0] != values.size:
        raise ValueError(
            f"rows: jacobian shape {J.shape} does not match {values.size} "
            "values")
    return [Fwd(values[i], J[i].copy()) for i in range(values.size)]


class Fwd:
    """Scalar with a value and a dense gradient over a fixed parameter list."""

    __slots__ = ("value", "grad")

    def __init__(self, value, grad):
        """``value`` plus the gradient over the **one** parameter space.

        The caller fixes the space (``Fitter.fwd_names()``) and every
        ``Fwd`` in an expression must use it — the operators check the
        gradient length, so a mixture of two spaces raises instead of
        broadcasting silently.
        """
        self.value = float(value)
        self.grad = np.asarray(grad, dtype=float)

    # -- helpers --------------------------------------------------------
    def _check(self, other):
        """Reject two *different* spaces; a scalar tangent broadcasts."""
        a, b = self.grad.shape, other.grad.shape
        if a != b and a != () and b != ():
            raise ValueError(
                "Fwd operands use different parameter spaces: "
                f"{self.grad.size} vs {other.grad.size} — build them from "
                "the same parameter list")
        return other

    def copy(self):
        return Fwd(self.value, self.grad.copy())

    def cov(self, other, cov):
        """Covariance of two duals: ``g_a · C · g_b``.

        Both must live in the same parameter space (a scalar tangent, i.e. a
        constant, has no covariance and gives 0).
        """
        other = self._check(other)
        if self.grad.ndim == 0 or other.grad.ndim == 0:
            return 0.0
        return float(self.grad @ np.asarray(cov, dtype=float) @ other.grad)

    def correlation(self, other, cov):
        """Correlation coefficient of two duals, ``cov / (sigma_a·sigma_b)``.

        0 when either side is a constant (zero variance).
        """
        c = self.cov(other, cov)
        sa, sb = self.sigma(cov), other.sigma(cov)
        if sa == 0.0 or sb == 0.0:
            return 0.0
        return c / (sa * sb)

    corr = correlation

    def sigma(self, cov):
        """Standard deviation ``sqrt(g C g^T)`` for a parameter covariance.

        A constant (scalar tangent, e.g. built from a plain number) has no
        uncertainty and returns 0.
        """
        g = self.grad
        if g.ndim == 0:
            return 0.0
        var = float(g @ np.asarray(cov, dtype=float) @ g)
        return float(np.sqrt(var if var > 0.0 else 0.0))

    # -- arithmetic -----------------------------------------------------
    def __add__(self, other):
        if isinstance(other, Fwd):
            other = self._check(other)
            return Fwd(self.value + other.value, self.grad + other.grad)
        if isinstance(other, _NUM):
            return Fwd(self.value + float(other), self.grad)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Fwd):
            other = self._check(other)
            return Fwd(self.value - other.value, self.grad - other.grad)
        if isinstance(other, _NUM):
            return Fwd(self.value - float(other), self.grad)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, _NUM):
            return Fwd(float(other) - self.value, -self.grad)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Fwd):
            other = self._check(other)
            return Fwd(self.value * other.value,
                       self.grad * other.value + other.grad * self.value)
        if isinstance(other, _NUM):
            c = float(other)
            return Fwd(self.value * c, self.grad * c)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Fwd):
            other = self._check(other)
            v = other.value
            return Fwd(self.value / v,
                       (self.grad * v - other.grad * self.value) / (v * v))
        if isinstance(other, _NUM):
            c = float(other)
            return Fwd(self.value / c, self.grad / c)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, _NUM):
            c = float(other)
            v = self.value
            return Fwd(c / v, -(c / (v * v)) * self.grad)
        return NotImplemented

    def __neg__(self):
        return Fwd(-self.value, -self.grad)

    def __mod__(self, other):
        """``self % other`` (Python/sympy floor semantics: sign of the divisor).

        The quotient's floor is locally constant, so the gradient is
        ``g_a - floor(a/b)·g_b`` — exact between the discontinuities.
        """
        if isinstance(other, Fwd):
            other = self._check(other)
            q = np.floor(self.value / other.value)
            return Fwd(self.value - other.value * q,
                       self.grad - q * other.grad)
        if isinstance(other, _NUM):
            c = float(other)
            if c == 0.0:
                raise ZeroDivisionError("Fwd % 0")
            return Fwd(self.value - c * np.floor(self.value / c), self.grad)
        return NotImplemented

    def __rmod__(self, other):
        if isinstance(other, _NUM):
            c = float(other)
            if self.value == 0.0:
                raise ZeroDivisionError("number % Fwd(0)")
            q = np.floor(c / self.value)
            return Fwd(c - self.value * q, -q * self.grad)
        return NotImplemented

    def __abs__(self):
        """``|value|``; the gradient picks up the sign (0 at the origin)."""
        v = self.value
        sign = 0.0 if v == 0.0 else (1.0 if v > 0.0 else -1.0)
        return Fwd(v if v >= 0.0 else -v, sign * self.grad)

    def __pos__(self):
        return self

    def __pow__(self, p):
        if isinstance(p, Fwd):
            raise TypeError("Fwd ** Fwd is not supported (fix the exponent)")
        if not isinstance(p, _NUM):
            return NotImplemented
        c = float(p)
        v = self.value
        return Fwd(v ** c, c * (v ** (c - 1.0)) * self.grad)

    # -- elementary functions -------------------------------------------
    def sqrt(self):
        r = np.sqrt(self.value)
        return Fwd(r, 0.5 / r * self.grad)

    def log(self):
        return Fwd(np.log(self.value), self.grad / self.value)

    def sin(self):
        return Fwd(np.sin(self.value), np.cos(self.value) * self.grad)

    def cos(self):
        return Fwd(np.cos(self.value), -np.sin(self.value) * self.grad)

    def exp(self):
        e = np.exp(self.value)
        return Fwd(e, e * self.grad)

    # -- display --------------------------------------------------------
    def __repr__(self):
        return f"Fwd(value={self.value:.12g}, n_grad={self.grad.size})"




# ── dispatch: numbers in, numbers out; any Fwd promotes them all ────────
def _like(x, ref):
    """*x* as a dual in *ref*'s space; other values pass through untouched."""
    if isinstance(x, Fwd):
        if ref is not None:
            ref._check(x)
        return x
    if isinstance(x, _NUM):
        return Fwd(float(x), 0.0)       # scalar tangent; numpy broadcasts it
    return x


def dual(fn):
    """Convert every argument to a :class:`Fwd` before calling *fn*.

    Numbers become duals in the first ``Fwd`` argument's parameter space (a
    zero-length tangent when there is none), so one body serves both the
    numeric and the dual call and the result is always a dual — take
    ``.value`` for the plain number.  Non-numeric arguments (``None``,
    strings, arrays) are passed through untouched.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        ref = next((a for a in list(args) + list(kwargs.values())
                    if isinstance(a, Fwd)), None)
        args = tuple(_like(a, ref) for a in args)
        kwargs = {k: _like(v, ref) for k, v in kwargs.items()}
        return fn(*args, **kwargs)
    return wrapper


# ── elementary functions on duals (numpy-style names) ──────────────────
@dual
def sin(x):
    return x.sin()


@dual
def cos(x):
    return x.cos()


@dual
def exp(x):
    return x.exp()


@dual
def log(x):
    return x.log()


@dual
def sqrt(x):
    return x.sqrt()


@dual
def erf(x):
    """Error function; ``d erf = (2/sqrt(pi))·exp(-x²)``."""
    from scipy.special import erf as _erf

    v = x.value
    return Fwd(float(_erf(v)),
               (2.0 / np.sqrt(np.pi)) * np.exp(-v * v) * x.grad)


@dual
def gamma(x):
    """Gamma function; ``d gamma = gamma(x)·digamma(x)``.

    Real duals only — ``scipy.special`` has no complex gamma (``loggamma``
    does, but that is a different branch choice).  Singular at the
    non-positive integers, like the function itself.
    """
    from scipy.special import digamma as _digamma
    from scipy.special import gamma as _gamma

    v = x.value
    g = float(_gamma(v))
    return Fwd(g, g * float(_digamma(v)) * x.grad)


@dual
def loggamma(x):
    """``log|gamma(x)|``; ``d loggamma = digamma(x)`` (stable for large x)."""
    from scipy.special import digamma as _digamma
    from scipy.special import gammaln as _gammaln

    v = x.value
    return Fwd(float(_gammaln(v)), float(_digamma(v)) * x.grad)


@dual
def atan2(y, x):
    """``atan2(y, x)``; ``d(theta) = (x·dy - y·dx) / (x^2 + y^2)``."""
    den = x.value * x.value + y.value * y.value
    return Fwd(np.arctan2(y.value, x.value),
               (x.value * y.grad - y.value * x.grad) / den)


@dual
def mod(a, b):
    """``a % b`` (floor semantics); the quotient's floor is locally constant."""
    q = np.floor(a.value / b.value)
    return Fwd(a.value - b.value * q, a.grad - q * b.grad)


@dual
def magnitude(re, im):
    """``sqrt(re^2 + im^2)`` of a dual pair."""
    mag = np.hypot(re.value, im.value)
    return Fwd(mag, (re.value * re.grad + im.value * im.grad) / mag)


@dual
def polar(re, im):
    """Rectangular -> polar: ``(magnitude, phase)`` duals of ``re + i*im``."""
    return (magnitude(re, im), atan2(im, re))


@dual
def reim(magnitude, phase):
    """Polar -> rectangular: ``(re, im)`` duals of ``m*e^{i*phi}``.

    ``re = m*cos(phi)``, ``im = m*sin(phi)`` — the pair carries both
    gradients, so ``re^2 + im^2`` recovers the magnitude squared.
    """
    c, s = np.cos(phase.value), np.sin(phase.value)
    re = Fwd(magnitude.value * c,
             c * magnitude.grad - magnitude.value * s * phase.grad)
    im = Fwd(magnitude.value * s,
             s * magnitude.grad + magnitude.value * c * phase.grad)
    return (re, im)


@dual
def abs(x, im=None):
    """``|x|`` — real modulus, or the complex modulus of the pair ``(x, im)``."""
    if im is None:
        return x.__abs__()
    return magnitude(x, im)


# ── complex duals: a pair of real duals, no separate statistics ────────
fwd_reim = reim
fwd_magnitude = magnitude
fwd_atan2 = atan2
fwd_exp = exp
fwd_log = log
def _c_real(x):
    """A real dual from a number or ``Fwd`` (shared by :class:`C`)."""
    if isinstance(x, Fwd):
        return x
    if isinstance(x, _NUM):
        return Fwd(float(x), 0.0)
    raise TypeError(f"expected a number or Fwd, got {type(x).__name__}")


def _as_c(x):
    if isinstance(x, C):
        return x
    if isinstance(x, Fwd) or isinstance(x, _NUM):
        return C(x, 0.0)
    raise TypeError(f"expected a number, Fwd or C, got {type(x).__name__}")


class C:
    """Complex dual: ``(re, im)`` as two real :class:`Fwd` values.

    The arithmetic is the usual complex algebra written on the pair, so every
    gradient stays **real** (the parameters are real) and no conjugate
    convention is needed.  There is deliberately no ``sigma``/``cov`` here:
    take them on ``.re`` / ``.im`` (or on ``abs()`` / ``arg()``), which is
    where the uncertainty of a complex quantity is actually defined.

    ``conj``, ``abs``, ``arg``, ``exp``, ``log``, ``sqrt`` and ``**`` are
    provided; ``abs`` returns a real ``Fwd`` (so ``.sigma(cov)`` works),
    ``arg`` too.
    """

    __slots__ = ("re", "im")

    def __init__(self, re=0.0, im=0.0):
        self.re = _c_real(re)
        self.im = _c_real(im)

    # -- constructors ---------------------------------------------------
    @classmethod
    def of(cls, z):
        """From a Python complex number (a constant: zero tangent)."""
        return cls(Fwd(float(np.real(z)), 0.0), Fwd(float(np.imag(z)), 0.0))

    @classmethod
    def from_polar(cls, magnitude, phase):
        """``m*e^{i*phi}`` from a real dual magnitude and phase."""
        re, im = fwd_reim(magnitude, phase)
        return cls(re, im)

    # -- accessors ------------------------------------------------------
    @property
    def value(self):
        return complex(self.re.value, self.im.value)

    def conj(self):
        return C(self.re, -self.im)

    def abs(self):
        """Magnitude as a real dual (``Fwd``)."""
        return fwd_magnitude(self.re, self.im)

    def arg(self):
        """Phase as a real dual (``Fwd``)."""
        return fwd_atan2(self.im, self.re)

    def exp(self):
        return C(*fwd_reim(fwd_exp(self.re), self.im))

    def log(self):
        return C(fwd_log(self.abs()), self.arg())

    def sqrt(self):
        return (self.log() * 0.5).exp()

    # -- arithmetic -----------------------------------------------------
    def __add__(self, other):
        o = _as_c(other)
        return C(self.re + o.re, self.im + o.im)

    __radd__ = __add__

    def __sub__(self, other):
        o = _as_c(other)
        return C(self.re - o.re, self.im - o.im)

    def __rsub__(self, other):
        return _as_c(other) - self

    def __neg__(self):
        return C(-self.re, -self.im)

    def __pos__(self):
        return self

    def __mul__(self, other):
        o = _as_c(other)
        return C(self.re * o.re - self.im * o.im,
                 self.re * o.im + self.im * o.re)

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = _as_c(other)
        n2 = o.re * o.re + o.im * o.im
        return C((self.re * o.re + self.im * o.im) / n2,
                 (self.im * o.re - self.re * o.im) / n2)

    def __rtruediv__(self, other):
        return _as_c(other) / self

    def __pow__(self, p):
        if isinstance(p, C):
            raise TypeError("C ** C is not supported (use exp(p*log(z)))")
        if not isinstance(p, (_NUM, Fwd)):
            return NotImplemented
        return (self.log() * p).exp()

    def __abs__(self):
        return self.abs()

    def __repr__(self):
        return f"C({self.value:.12g})"
