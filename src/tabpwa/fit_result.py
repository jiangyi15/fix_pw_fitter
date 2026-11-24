"""Fit-result container.

``FitResult`` is the base class for everything that flows through
:meth:`tabpwa.fitter.Fitter.save_params` / :meth:`...load_results` — a
scipy-``OptimizeResult``-compatible attribute set (``x`` plus the optional
``fun`` / ``jac`` / ``success`` / ``message`` / ``hess_inv``).  Consumers
never need to special-case it: any object exposing those attributes
(scipy results included) remains accepted; this class just gives the
shape a name, sane defaults and constructors.
"""

import numpy as np


def as_matrix(m):
    """Densify an error matrix; ``None`` stays ``None``.

    A scipy result may carry a LINEAR OPERATOR rather than an array: BFGS
    returns the dense ``Hk``, but L-BFGS-B returns a ``LbfgsInvHessProduct``,
    which has NO ``toarray`` and makes ``np.asarray`` raise ``TypeError`` (so a
    fit with ``method="L-BFGS-B"`` died in ``FitResult`` after converging).
    Every consumer (``np.diag``, ``@``, ``np.save``) wants the matrix, so the
    conversion lives here once and ``FitResult.hess_inv`` is always an ndarray:
    a real operator is densified by its own ACTION on the identity (exact for
    the operator, ``N`` mat-vecs).
    """
    if m is None:
        return None
    toarray = getattr(m, "toarray", None)
    if callable(toarray):
        return np.asarray(toarray(), dtype=float)
    shape = getattr(m, "shape", None)
    if shape is not None and len(shape) == 2 and callable(getattr(m, "matmat", None)):
        return np.asarray(m.matmat(np.eye(int(shape[1]))), dtype=float)
    return np.asarray(m, dtype=float)


class FitResult:
    """Base container for fit results.

    Attributes:
        x: flat parameter vector (``None`` = empty result).
        fun: final NLL (``None`` when only a parameter point is known).
        jac: final gradient (``None`` when unavailable).
        success: convergence flag (``None`` when not a real fit).
        message: optimizer status message.
        hess_inv: inverse Hessian / error matrix (``None`` when unknown).

    Extra keyword arguments are stored verbatim as attributes, so
    subclasses and callers can attach derived quantities without
    subclassing.
    """

    def __init__(self, x=None, fun=None, jac=None, success=None,
                 message=None, hess_inv=None, **extra):
        self.x = None if x is None else np.asarray(x, dtype=float)
        self.fun = fun
        self.jac = jac
        self.success = success
        self.message = message
        self.hess_inv = as_matrix(hess_inv)
        for key, value in extra.items():
            setattr(self, key, value)

    # ── constructors ──────────────────────────────────────────────

    @classmethod
    def from_x(cls, x, **extra):
        """Result holding only a flat parameter vector (no fit status)."""
        return cls(x=x, message="No fit result", **extra)

    @classmethod
    def from_result(cls, res, **extra):
        """Copy from any attribute object (scipy ``OptimizeResult``, ...).

        Every field of *res* is preserved — scipy results are dict-backed
        (``nit``, ``status``, …), attribute objects are read via
        ``vars()`` — so nothing is lost in the conversion.  A ``FitResult``
        instance is returned as-is unless *extra* overrides fields.
        """
        if isinstance(res, cls) and not extra:
            return res
        if isinstance(res, dict):
            fields = dict(res)
        else:
            fields = dict(vars(res))
        for name in ("x", "fun", "jac", "success", "message", "hess_inv"):
            fields.setdefault(name, getattr(res, name, None))
        fields.update(extra)
        return cls(**fields)

    # ── status ────────────────────────────────────────────────────

    @property
    def is_empty(self):
        """True when no parameter vector is stored."""
        return self.x is None or len(self.x) == 0

    def __len__(self):
        return 0 if self.x is None else len(self.x)

    def __repr__(self):
        return (f"{type(self).__name__}(n={len(self)}, fun={self.fun}, "
                f"hess_inv={'yes' if self.hess_inv is not None else 'no'})")
