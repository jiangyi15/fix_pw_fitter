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
        self.hess_inv = (None if hess_inv is None
                         else np.asarray(hess_inv, dtype=float))
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
