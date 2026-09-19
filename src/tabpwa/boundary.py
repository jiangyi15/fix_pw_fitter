"""
boundary — Variable bound transformations for optimization.

Maps unbounded optimizer variables to bounded physical ranges using a
smooth, bijective arctan transform.

The :class:`Boundary` class bundles a collection of named BoundTransforms
and provides dict-level apply / gradient-correction / save-load methods.
"""

import numpy as np


class BoundTransform:
    """Transform unbounded variables to a bounded [a, b] range.
    
    Uses: y = bias + k*(2/π)*arctan(x*π/(2*k))
    where k = (b-a)/2, bias = (b+a)/2.
    
    This is BIJECTIVE over all real numbers (unlike sin which wraps).
    Forward: (-∞, +∞) → (a, b)  smooth, monotonic.
    Inverse: (a, b) → (-∞, +∞)  exact, no periodicity.
    Gradient at x=0 is 1 (same scaling as the original sin transform).
    
    trans_err propagates covariance through the transform.
    """

    def __init__(self, a, b):
        self.a = float(min(a, b))
        self.b = float(max(a, b))
        self.k = (self.b - self.a) / 2.0
        self.bias = (self.b + self.a) / 2.0
        self._c = np.pi / (2.0 * (self.b - self.a)) * 2.0  # π/(2*k)
        # Actually: c = π / (2*k)
        # But k = (b-a)/2, so 2*k = b-a
        # c = π / (b-a)

    def forward(self, x):
        """Unbounded x -> bounded y in (a, b). Bijective over all x."""
        c = np.pi / (self.b - self.a)
        return self.bias + (self.b - self.a) / np.pi * np.arctan(x * c)

    def __call__(self, x):
        return self.forward(x)

    def grad(self, x):
        """Gradient dy/dx at x (for chain rule in gradient backprop)."""
        c = np.pi / (self.b - self.a)
        return 1.0 / (1.0 + (x * c) ** 2)

    def jacobian(self, x):
        """Alias for grad (consistent naming)."""
        return self.grad(x)

    def inverse(self, y):
        """Bounded y in (a, b) -> unbounded x. Exact inverse."""
        c = np.pi / (self.b - self.a)
        yc = np.clip(y, self.a, self.b)
        return np.tan((yc - self.bias) * c) / c

    def trans_err(self, x, error):
        """Propagate error through the transform.
        
        Args:
            x: unbounded variable value.
            error: standard deviation (or error) on x.
        
        Returns:
            error on y (bounded value) = |grad(x)| * error
        """
        return np.abs(self.grad(x)) * error

    def to_dict(self):
        """Serialize to JSON-compatible dict."""
        return {"low": self.a, "high": self.b}


# Commonly used bounds for time-dependent amplitude parameters
TIME_PARAM_BOUNDS = {
    "delta_m":       [0.3, 0.8],
    "delta_gamma":  [-0.3, 0.3],
    "A_prod":       [-0.5, 0.5],
    "gamma":        [-0.3, 0.3],
}

# Fixed default values (used when parameters are not free)
TIME_PARAM_DEFAULTS = {
    "gamma":        0.0,
    "delta_m":      0.506,
    "delta_gamma":  0.0,
    "poqr":         1.0,
    "poqi":         0.0,
    "A_prod":       0.0,
}


# ── Boundary collection ────────────────────────────────────────────

class Boundary:
    """Collection of named BoundTransform instances.

    Stores bounds by name (no existence checks at set time).
    Provides dict-level apply / gradient-correction / save-load.
    """

    def __init__(self):
        self._tfm = {}   # name → BoundTransform

    # ── modify ────────────────────────────────────────────────────

    def set(self, name, lo, hi):
        """Store a bound by name (no registry lookup)."""
        self._tfm[name] = BoundTransform(lo, hi)

    def unset(self, name):
        """Remove a bound by name."""
        self._tfm.pop(name, None)

    def update(self, items):
        """Bulk set from {name: (lo, hi)} or {name: {low, high}}."""
        for name, spec in items.items():
            lo = spec[0] if isinstance(spec, (list, tuple)) else spec["low"]
            hi = spec[1] if isinstance(spec, (list, tuple)) else spec["high"]
            self._tfm[name] = BoundTransform(lo, hi)

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
            Bounded named-parameter dict (returned by ``apply()``).
            Used to get the bounded value for Jacobian evaluation.
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

