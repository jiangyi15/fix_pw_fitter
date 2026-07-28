"""
B-spline running width model — interior basis functions only.

Each B-spline basis function of order *p* spans ``p+1`` knot intervals.
Basis functions near the boundaries have incomplete support and are
absorbed into the nearest interior function.  Only **interior** basis
functions (with full support) are fitted::

    n_free = n_knots - 2 * order

    Γ(m) = Σ v_k · B_{k+order}(m)

For cubic (order=3), 7 knots → 1 free parameter, 9 knots → 3 free, etc.

The denominator::

    D(m) = m₀² − m² − i·m₀·Γ(m)

Each real basis coefficient has an imaginary counterpart for the mass
shift::

    Γ(m) = Σ (re_k + i·im_k) · B_k(m)

YAML usage::

    particle:
      sigma:
        mass: 0.5
        model: BSpline
        knots: [0.3, 0.5, 0.7, 1.0, 1.5, 2.5, 5.0]   # 7 knots, order 3 → 1 free
        order: 3
"""

import numpy as np
from .base import BaseModel, register_model
from ampfit.param_constraint import Transform


# ═══════════════════════════════════════════════════════════════════
# Clamped B-spline basis (Cox–de Boor)
# ═══════════════════════════════════════════════════════════════════

def bspline_basis_all(x, breakpoints, order=3):
    """Full clamped B-spline basis, all functions.

    Returns ``(basis_wide, n_all)`` where ``basis_wide`` has shape
    ``(len(x), n_all)`` with all ``n_all = len(breakpoints) + order - 1``
    basis functions (including edge functions with incomplete support).
    """
    x = np.asarray(x, dtype=float)
    bp = np.asarray(breakpoints, dtype=float)

    # Clamped knot vector
    t0 = np.full(order + 1, bp[0])
    t1 = np.full(order + 1, bp[-1])
    knots = np.concatenate([t0, bp[1:-1], t1])
    n_knots = len(knots)
    n_all = n_knots - order - 1  # total basis functions

    if n_all <= 0:
        raise ValueError(
            f"Need at least {order + 2} breakpoints for order {order}")

    # Order 0
    eps = 1e-12
    x_snap = np.where((x >= knots[-1]) & (x < knots[-1] + 10*eps),
                      knots[-1] - eps, x)
    basis = np.zeros((len(x_snap), n_all + order))
    for i in range(n_all + order):
        if i + 1 < n_knots:
            mask = (x_snap >= knots[i]) & (x_snap < knots[i + 1])
            basis[mask, i] = 1.0

    # Cox-de Boor recursion
    for p in range(1, order + 1):
        for i in range(n_all + order - p):
            denom_l = knots[i + p] - knots[i]
            left = np.zeros_like(x)
            if denom_l > 0:
                left = (x - knots[i]) / denom_l * basis[:, i]

            denom_r = knots[i + p + 1] - knots[i + 1]
            right = np.zeros_like(x)
            if denom_r > 0:
                right = (knots[i + p + 1] - x) / denom_r * basis[:, i + 1]

            basis[:, i] = left + right

    return basis[:, :n_all], n_all


def bspline_basis_interior(x, breakpoints, order=3):
    """Interior B-spline basis — ``n_free = max(0, n_knots - order - 1)``.

    The full clamped B-spline has ``n_all = n_knots + order - 1``
    functions.  We return only the interior ones (removing ``order``
    from each edge).  The sum goes to zero at the boundaries — edge
    functions are not fitted.

    For odd *order*, basis functions peak at knot positions.
    For even *order*, they peak **between** knots (bin centres).

    Returns ``(basis, n_free)``.
    """
    basis_wide, n_all = bspline_basis_all(x, breakpoints, order)
    n_free = max(0, n_all - 2 * order)  # = n_knots - order - 1

    if n_free <= 0:
        return np.zeros((len(np.asarray(x, dtype=float)), 0)), 0

    # Select interior columns: [order, n_all - order)
    return basis_wide[:, order:n_all - order].copy(), n_free

    return basis, n_free


# ═══════════════════════════════════════════════════════════════════
# Transform — fixes mass, knot variables pass through as free params
# ═══════════════════════════════════════════════════════════════════

class _SplineMassFixTransform(Transform):
    """Fixes mass; knot variables are untouched free params."""

    _has_inverse = False

    def __init__(self, mass_name, mass_default):
        super().__init__(input_names=[], output_names=[mass_name])
        self._fixed = {mass_name: float(mass_default)}

    def forward(self, d):
        result = dict(d)
        result.update(self._fixed)
        return result

    def backward(self, grad_out, d_in=None):
        return {}


# ═══════════════════════════════════════════════════════════════════
# BSplineGammaModel
# ═══════════════════════════════════════════════════════════════════

@register_model("BSpline")
class BSplineGammaModel(BaseModel):
    """B-spline running width — interior basis functions only.

    The running width is a sum over **interior** B-spline basis
    functions::

        n_free = n_knots - 2 * order

        Γ(m) = Σ (re_k + i·im_k) · B_k(m)

    Only basis functions with full support (spanning the interior)
    are fitted.  Edge functions are absorbed into the nearest
    interior function.

    Parameters (from YAML config):
        mass      — fixed pole mass (not fitted)
        knots     — list of breakpoint positions
        order     — spline order (default 3 = cubic)
        g_{2k}    — initial value for re_k (default 0.1)
        g_{2k+1}  — initial value for im_k (default 0.0)
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        knots = kwargs.get("knots", None)
        if knots is None:
            raise ValueError(f"BSpline model '{name}': 'knots' is required")
        if isinstance(knots, str):
            self.breakpoints = np.load(knots)
        else:
            self.breakpoints = np.asarray(knots, dtype=float)

        self.order = int(kwargs.get("order", 3))
        min_bp = self.order + 2
        if len(self.breakpoints) < min_bp:
            raise ValueError(
                f"BSpline model '{name}': need at least {min_bp} breakpoints "
                f"for order {self.order} (n_free = n_knots - order - 1)")
        self.n_free = len(self.breakpoints) - self.order - 1

    # ── gamma interface ──────────────────────────────────────────

    def get_gamma_count(self):
        return 2 * self.n_free

    def get_gamma_name(self):
        names = []
        for i in range(self.n_free):
            names.append(f"{self.name}_re_B_{i}")
            names.append(f"{self.name}_im_B_{i}")
        return names

    def gamma(self, m):
        """Return gamma components for interior basis functions.

        ``[B_0, i·B_0, B_1, i·B_1, ...]`` where B_k are the interior
        B-spline basis functions (with edge functions absorbed).
        """
        basis, _ = bspline_basis_interior(m, self.breakpoints, self.order)
        comps = []
        for i in range(basis.shape[1]):
            b = basis[:, i].astype(complex)
            comps.append(b)
            comps.append(1j * b)
        return comps

    # ── default parameters ───────────────────────────────────────

    def get_defaults(self):
        return {}

    # ── mass/width transform ─────────────────────────────────────

    def make_mass_width_transform(self):
        return _SplineMassFixTransform(
            f"{self.name}_mass",
            mass_default=float(self.kwargs.get("mass", 1.0)),
        )
