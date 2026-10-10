"""Numerics: finite-difference derivatives and sampled-function
inversion.

The common engine for consumers that need numerical derivatives or
interpolation in the flat (optimizer) parameter space:

- ``Fitter.compute_numerical_hessian`` (the uncertainties path),
- ``tabpwa.profile.fd_hessian`` (the profile-scan curvature),
- ``tabpwa.profile.ProfileResult.crossing`` (ΔQ inversion),
- gradient-vs-FD consistency checks (tests, the fitter self-test).

All functions take the callable/data directly — no Fitter dependency.
"""

import numpy as np


def fd_grad(value, x0, eps=1e-5, idx=None):
    """Central-difference gradient of a scalar function (a 3-point
    check per component — ``2·K`` value evaluations, K components).

    Args:
        value: callable ``x -> float``.
        x0: the expansion point.
        eps: the finite-difference step — a scalar, or a per-component
            array aligned with *idx* (e.g. ``1e-6·(1 + |x[idx]|)``).
        idx: the component indices to differentiate (None = all).

    Returns:
        The K gradient components, aligned with *idx*.
    """
    x0 = np.asarray(x0, float)
    idx = np.arange(len(x0)) if idx is None else np.atleast_1d(idx)
    eps = np.broadcast_to(np.asarray(eps, float), idx.shape)
    g = np.empty(len(idx))
    for c, j in enumerate(idx):
        h = float(eps[c])
        xp = x0.copy(); xp[j] += h
        xm = x0.copy(); xm[j] -= h
        g[c] = (value(xp) - value(xm)) / (2.0 * h)
    return g


def interp_inverse(xs, ys, level):
    """Invert a sampled function: the first ``x`` where ``ys`` crosses
    ``level`` (from below).

    The crossing inverts the monotone cubic (PCHIP) interpolant of the
    samples — no overshoot between nodes — solving
    ``PCHIP(x) = level`` with :func:`scipy.optimize.brentq` inside the
    bracketing interval.  The samples are scanned in the given order
    (x may increase or decrease — e.g. a side walked outward with
    decreasing parameter values); the bracket pair straddles the level
    in that order.  Falls back to the first sample when ``ys`` starts
    at/above the level.

    Args:
        xs: the sample positions in scan order (either direction).
        ys: the sample values.
        level: the value to invert at.

    Returns:
        The interpolated ``x``, or ``None`` when ``ys`` never reaches
        the level.
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    hit = np.nonzero(ys >= level)[0]
    if not len(hit):
        return None
    k = int(hit[0])
    if k == 0:
        return float(xs[0])
    if xs[k] < xs[k - 1]:                    # decreasing x → flip to increasing
        xs = xs[::-1].copy()
        ys = ys[::-1].copy()
        a, b = len(xs) - 1 - k, len(xs) - k  # the same straddling pair
    else:
        a, b = k - 1, k
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import brentq
    p = PchipInterpolator(xs, ys)
    return float(brentq(lambda x: float(p(x)) - level, xs[a], xs[b]))


def fd_hess(grad, x0, eps=1e-5):
    """Raw numerical Hessian at ``x0``: central differences on the
    gradient (2N evaluations — the gradient gives a whole Hessian
    column per pair), symmetrized.

    Args:
        grad: callable ``x -> grad_flat`` (a gradient function; pass
            ``lambda x: f.get_nll(x)[1]`` for a Fitter).
        x0: the expansion point.
        eps: the finite-difference step (absolute, one value for all
            parameters).

    Returns:
        The ``(N, N)`` symmetric Hessian matrix.
    """
    x0 = np.asarray(x0, float)
    n = len(x0)
    H = np.zeros((n, n))
    for j in range(n):
        xp = x0.copy(); xp[j] += eps
        xm = x0.copy(); xm[j] -= eps
        H[:, j] = (np.asarray(grad(xp)) - np.asarray(grad(xm))) / (2.0 * eps)
    return 0.5 * (H + H.T)                    # symmetrize
