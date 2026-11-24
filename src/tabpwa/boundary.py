"""
boundary — Variable bound transformations for optimization.

Maps the optimizer variable onto a bounded physical range with a SINE,
``y = bias + k*sin(x/k)`` on the branch ``|x| <= k*pi/2`` — the transform
MINUIT and every stepping minimizer of that family uses, and the one that keeps
the edge reachable at a FINITE ``x``.

The :class:`Boundary` class bundles a collection of named BoundTransforms
and provides dict-level apply / gradient-correction / save-load methods.
"""

import numpy as np


class BoundTransform:
    """ONE bound map, for a two-sided range or a one-sided (half-line) bound.

    The shape is chosen by which sides are given and stated by ``kind`` — but
    every consumer uses the same four methods (``forward``, ``grad``,
    ``inverse``, ``draw_start``), so no caller has to test the kind:

    ``kind == "range"`` (``lo`` and ``hi``)
        ``y = bias + k·sin(x/k)``, ``k = (hi-lo)/2``, ``bias = (lo+hi)/2``.
        ``dy/dx = cos(x/k)``, ``dy/dx(0) = 1``; the range is reached at the
        FINITE ``x = ±kπ/2`` and ``inverse`` is the PRINCIPAL branch.  ``x`` is
        NOT clamped: at the edge ``y' = 0`` and ``y'' = -1/k``, so for a fit
        pulled outwards (``g_y < 0``) ``F = NLL(y(x))`` has
        ``F'' = |g_y|/k > 0`` — a proper, unique, PD minimum AT the edge.  A
        clamp would instead pin everything past the edge to one value
        (``dNLL/dx = 0`` on a half-line, a non-unique minimiser, ``H_xx = 0``).

    ``kind == "lower"`` (``hi=None``)
        ``y = lo + sqrt(x²+1) - 1`` on the branch ``x ≥ 0``; ``dy/dx =
        x/sqrt(x²+1)`` (0 at the wall ``x = 0``, → 1 far from it, so a parameter
        that wants to be large is not compressed).  NO scale: a per-parameter
        scale would move the point where the map becomes linear, i.e. make the
        optimizer's step unit parameter-dependent, which ``gtol``/``eps`` assume
        is fixed.  ``kind == "upper"`` (``lo=None``) is the mirror
        ``y = hi - sqrt(x²+1) + 1`` on ``x ≤ 0``.

    The OPEN side is ``±inf`` in ``a``/``b``, so code that clips a value or a
    step inside the range needs no special case, and ``to_dict`` writes ``null``.
    A degenerate range (``lo == hi``) is a constant map with ``grad = 0``.
    """

    def __init__(self, lo, hi=None):
        if lo is None and hi is None:
            raise ValueError("bound: give lo, hi, or one of them (None = open)")
        self.lo = None if lo is None else float(lo)
        self.hi = None if hi is None else float(hi)
        self.k = self.bias = None
        if self.lo is not None and self.hi is not None:
            self.kind = "range"
            self.a = float(min(self.lo, self.hi))
            self.b = float(max(self.lo, self.hi))
            self.lo, self.hi = self.a, self.b
            self.k = (self.b - self.a) / 2.0
            self.bias = (self.b + self.a) / 2.0
        elif self.hi is None:
            self.kind = "lower"
            self.a, self.b = self.lo, float("inf")
        else:
            self.kind = "upper"
            self.a, self.b = float("-inf"), self.hi

    # ── the map ───────────────────────────────────────────────────────
    def forward(self, x):
        """Optimizer variable -> the bounded value (single-valued, smooth)."""
        if self.kind == "range":
            if self.k == 0.0:
                return (self.bias if np.isscalar(x)
                        else self.bias + 0.0 * np.asarray(x, float))
            return self.bias + self.k * np.sin(np.asarray(x, float) / self.k)
        if self.kind == "lower":
            t = np.maximum(x, 0.0)
            return self.lo + np.sqrt(t * t + 1.0) - 1.0
        t = np.maximum(-np.asarray(x, dtype=float), 0.0)
        return self.hi - np.sqrt(t * t + 1.0) + 1.0

    def __call__(self, x):
        return self.forward(x)

    def grad(self, x):
        """``dy/dx`` — signed (past a sine edge it is negative: the value turns)."""
        if self.kind == "range":
            if self.k == 0.0:
                return (0.0 if np.isscalar(x)
                        else 0.0 * np.asarray(x, float))
            return np.cos(np.asarray(x, float) / self.k)
        if self.kind == "lower":
            t = np.maximum(x, 0.0)
        else:
            t = np.maximum(-np.asarray(x, dtype=float), 0.0)
        return t / np.sqrt(t * t + 1.0)

    def jacobian(self, x):
        """Alias for grad (consistent naming)."""
        return self.grad(x)

    def inverse(self, y):
        """Bounded value -> the branch's optimizer variable (finite at the edge)."""
        if self.kind == "range":
            if self.k == 0.0:
                return 0.0 if np.isscalar(y) else 0.0 * np.asarray(y, float)
            t = (np.clip(y, self.a, self.b) - self.bias) / self.k
            return self.k * np.arcsin(np.clip(t, -1.0, 1.0))
        if self.kind == "lower":
            t = np.maximum(np.maximum(y, self.lo) - self.lo + 1.0, 1.0)
            return np.sqrt(t * t - 1.0)
        t = np.maximum(self.hi - np.minimum(y, self.hi) + 1.0, 1.0)
        return -np.sqrt(t * t - 1.0)

    def draw_start(self, u):
        """A random start ON the branch, from ``u`` uniform on ``[-1, 1]``.

        The draw is made in optimizer space; using it raw would put most starts
        on the far side of an edge (a zero gradient), so each shape maps it onto
        its own branch.  ONE call site, no kind test (see
        :meth:`ConstraintManager.initial_values`).
        """
        if self.kind == "range":
            return u * (self.k * np.pi / 2.0)
        if self.kind == "lower":
            return np.abs(u)
        return -np.abs(u)

    def trans_err(self, x, error):
        """Propagate an UNBOUNDED error: ``|dy/dx| * error`` at *x*."""
        return np.abs(self.grad(x)) * error

    def to_dict(self):
        """Serialize to a JSON-compatible dict (``None`` = an open side)."""
        return {"low": self.lo, "high": self.hi}


class Boundary:
    """Collection of named BoundTransform instances.

    Stores bounds by name (no existence checks at set time).
    Provides dict-level apply / gradient-correction / save-load.
    """

    def __init__(self):
        self._tfm = {}   # name → BoundTransform

    # ── modify ────────────────────────────────────────────────────

    def set(self, name, lo=None, hi=None):
        """Store a bound by name (no registry lookup).

        ``(lo, hi)`` both given → the two-sided sine; ``hi is None`` → a
        one-sided LOWER bound; ``lo is None`` → the upper mirror.  Both ``None``
        is not a bound and raises.
        """
        try:
            self._tfm[name] = BoundTransform(lo, hi)
        except ValueError as exc:
            raise ValueError(f"bound {name!r}: {exc}") from None

    def unset(self, name):
        """Remove a bound by name."""
        self._tfm.pop(name, None)

    def update(self, items):
        """Bulk set from {name: (lo, hi)} or {name: {low, high}}.

        Either entry may be ``None`` (JSON ``null``) for an OPEN side.
        """
        for name, spec in items.items():
            lo = spec[0] if isinstance(spec, (list, tuple)) else spec["low"]
            hi = spec[1] if isinstance(spec, (list, tuple)) else spec["high"]
            self.set(name, lo, hi)

    # ── apply ─────────────────────────────────────────────────────

    def apply(self, d):
        """Apply all bounds to a named dict *d* (in-place).

        Only transforms names that exist in *d* — silently skips
        names not present.
        """
        for name, bt in self._tfm.items():
            if name in d:
                d[name] = bt(d[name])
        return d

    def inverse(self, name, val):
        """Invert bound for *name*: map bounded *val* → unbounded.

        If *name* is not bounded, returns *val* unchanged.
        """
        bt = self._tfm.get(name)
        return bt.inverse(val) if bt else val

    def apply_from_unbounded(self, name, val, err=None):
        """Apply forward bound and optionally propagate error.

        Parameters
        ----------
        name : str
            Parameter name.
        val : float
            Unbounded value to transform.
        err : float or None
            Error on unbounded value.  If None, only the value is returned.

        Returns
        -------
        bounded_val : float
            If *name* is bounded: bt(val).  Otherwise: val.
        bounded_err : float or None
            If *err* is given: propagated error.  Otherwise: None.
            If *name* is not bounded, returns *err* unchanged.
        """
        bt = self._tfm.get(name)
        if bt is None:
            return (val, err) if err is not None else val
        bv = bt(val)
        if err is not None:
            return bv, bt.trans_err(val, err)
        return bv

    def apply_dict(self, d):
        """Apply forward bounds to all values in dict *d* (in-place).

        Only touches names that exist in *d* — silently skips others.
        """
        for name, bt in self._tfm.items():
            if name in d:
                d[name] = bt(d[name])
        return d

    # ── gradient ──────────────────────────────────────────────────

    def correct_gradient(self, grad_flat, x_flat, flat_names, raw_dict):
        """Apply the chain rule for bounded variables to *grad_flat*.

        Parameters
        ----------
        grad_flat : ndarray
            Gradient w.r.t. bounded variables (modified in-place).
        x_flat : ndarray
            Unbounded flat parameter vector.
        flat_names : list of str
            Names of parameters in the flat vector (same order as x_flat).
        raw_dict : dict
            **Unbounded** named-parameter dict (``from_flat(x)``): the
            Jacobian ``dy/dx`` is evaluated at the unbounded value.  (This
            docstring previously said "bounded", the same domain confusion
            that produced the ``apply_bound_dict`` uncertainty bug.)
        """
        for name, bt in self._tfm.items():
            try:
                si = flat_names.index(name)
                # Use bounded value from raw_dict for Jacobian
                val = raw_dict.get(name, x_flat[si])
                jac = bt.jacobian(val)
                # If multi-index (ck_slot), apply same jacobian to all
                ei = si + 1
                for idx in range(si, ei):
                    grad_flat[idx] *= jac
            except ValueError:
                pass  # name not in flat vector
        return grad_flat

    # ── serialise ─────────────────────────────────────────────────

    def to_dict(self):
        """Return {name: {"low": a, "high": b}} for JSON save."""
        return {name: bt.to_dict() for name, bt in self._tfm.items()}

    # ── introspection ─────────────────────────────────────────────

    def __contains__(self, name):
        return name in self._tfm

    def __getitem__(self, name):
        return self._tfm[name]

    def __len__(self):
        return len(self._tfm)

    def __iter__(self):
        return iter(self._tfm)

    def items(self):
        return self._tfm.items()

    def keys(self):
        return self._tfm.keys()

    def values(self):
        return self._tfm.values()

