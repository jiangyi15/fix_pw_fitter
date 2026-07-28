"""
B-spline running width model.

Replaces the traditional Breit-Wigner running width with a sum over
B-spline basis functions::

    Γ(m) = Σ g_j · B_j(m)

    D(m) = m₀² - m² - i·m₀·Γ(m)

The mass m₀ is fixed to its config value (not fitted).  The spline
coefficients g_j are the fitted parameters, giving flexible control
over the lineshape without assuming a specific analytic form.

YAML usage::

    particle:
      sigma:
        mass: 0.5
        model: BSpline
        knots: [0.3, 0.5, 0.7, 1.0, 1.5, 2.5, 5.0]
        order: 3                  # cubic B-spline (default)
"""

import numpy as np
from .base import BaseModel, register_model
from ampfit.param_constraint import Transform


# ═══════════════════════════════════════════════════════════════════
# B-spline basis functions (Cox–de Boor recursion)
# ═══════════════════════════════════════════════════════════════════

def bspline_basis(x, breakpoints, order=3):
    """Evaluate all B-spline basis functions at points *x*.

    Constructs a clamped B-spline (first and last knots repeated
    ``order + 1`` times for proper boundary behaviour).

    Uses the Cox–de Boor recursion formula.

    Parameters
    ----------
    x : array-like
        Evaluation points.
    breakpoints : array-like
        Interior breakpoints defining the spline segments.  For *k*
        breakpoints and order *p*, the number of basis functions is
        ``k + p - 1``.
    order : int
        Polynomial order (1=linear, 2=quadratic, 3=cubic).

    Returns
    -------
    ndarray, shape (len(x), n_basis)
        ``basis[i, j]`` = value of the j-th basis function at x[i].
    """
    x = np.asarray(x, dtype=float)
    bp = np.asarray(breakpoints, dtype=float)

    # Clamped knot vector: repeat first/last breakpoint (order+1) times
    t0 = np.full(order + 1, bp[0])
    t1 = np.full(order + 1, bp[-1])
    knots = np.concatenate([t0, bp[1:-1], t1])
    n_knots = len(knots)
    n_basis = n_knots - order - 1

    if n_basis <= 0:
        raise ValueError(
            f"Need at least {order + 2} breakpoints for order {order}, "
            f"got {len(bp)}")
    # Also check min breakpoints for clamped spline
    min_bp = order + 1
    if len(bp) < min_bp:
        raise ValueError(
            f"Need at least {min_bp} breakpoints for order {order}, "
            f"got {len(bp)}")

    # Order 0 — handle right endpoint (x = knots[-1]) by snapping into last interval
    eps = 1e-12
    x_snap = np.where((x >= knots[-1]) & (x < knots[-1] + 10*eps),
                      knots[-1] - eps, x)
    basis = np.zeros((len(x_snap), n_basis + order))
    for i in range(n_basis + order):
        if i + 1 < n_knots:
            mask = (x_snap >= knots[i]) & (x_snap < knots[i + 1])
            basis[mask, i] = 1.0

    # Cox-de Boor recursion for orders 1..p
    for p in range(1, order + 1):
        for i in range(n_basis + order - p):
            denom_l = knots[i + p] - knots[i]
            left = np.zeros_like(x)
            if denom_l > 0:
                left = (x - knots[i]) / denom_l * basis[:, i]

            denom_r = knots[i + p + 1] - knots[i + 1]
            right = np.zeros_like(x)
            if denom_r > 0:
                right = (knots[i + p + 1] - x) / denom_r * basis[:, i + 1]

            basis[:, i] = left + right

    return basis[:, :n_basis]


# ═══════════════════════════════════════════════════════════════════
# Transform — fixes mass, gamma names pass through as free params
# ═══════════════════════════════════════════════════════════════════

class _SplineMassFixTransform(Transform):
    """Fixes mass to config value; gamma names are untouched free params.

    ``input_names = []``, ``output_names = [mass_name]``.
    """

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
    """Running width parameterised by clamped B-spline basis functions.

    The mass m₀ is fixed to the config value.  The running width is a
    complex sum over B-spline basis functions::

        Γ(m) = Σ_{k} (re_k + i·im_k) · B_k(m)

    where ``re_k`` and ``im_k`` are fitted coefficients and B_k(m) are
    the clamped B-spline basis functions.  The denominator becomes::

        D(m) = m₀² − m² − i·m₀·Γ(m)
             = (m₀² + m₀·Σ im_k·B_k(m)) − m² − i·m₀·Σ re_k·B_k(m)

    The ``re_k`` coefficients control the running **width**, the
    ``im_k`` coefficients control the running **mass shift**.

    Each basis function produces **two** gamma components::
        gamma_{2k}(m)   = B_k(m)      → scales with re_k
        gamma_{2k+1}(m) = i · B_k(m)  → scales with im_k

    Parameters (from YAML config):
        mass        — fixed pole mass (not fitted)
        knots       — list of breakpoint positions or path to .npy file.
                      For order *p* and *k* breakpoints you get
                      ``k + p - 1`` basis functions.
        order       — spline order (default 3 = cubic)
        g_{2k}      — initial value for re_k (default 0.1)
        g_{2k+1}    — initial value for im_k (default 0.0)
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        # Breakpoints (user-provided knot positions)
        knots = kwargs.get("knots", None)
        if knots is None:
            raise ValueError(f"BSpline model '{name}': 'knots' is required")
        if isinstance(knots, str):
            self.breakpoints = np.load(knots)
        else:
            self.breakpoints = np.asarray(knots, dtype=float)

        self.order = int(kwargs.get("order", 3))
        min_bp = self.order + 1
        if len(self.breakpoints) < min_bp:
            raise ValueError(
                f"BSpline model '{name}': need at least {min_bp} breakpoints "
                f"for order {self.order}, got {len(self.breakpoints)}")
        self.n_basis = len(self.breakpoints) + self.order - 1

    # ── gamma interface ──────────────────────────────────────────

    def get_gamma_count(self):
        """Two gamma components per basis function: B_k and i·B_k."""
        return 2 * self.n_basis

    def get_gamma_name(self):
        names = []
        for i in range(self.n_basis):
            names.append(f"{self.name}_re_B_{i}")
            names.append(f"{self.name}_im_B_{i}")
        return names

    def gamma(self, m):
        """Return gamma components: ``[B_0, i·B_0, B_1, i·B_1, ...]``.

        The real-part components give the running **width**.
        The imaginary-part components (pre-multiplied by ``i``) give
        the running **mass shift** via the kernel's ``Σ g_j·gamma_j``.
        """
        basis = bspline_basis(m, self.breakpoints, self.order)
        comps = []
        for i in range(self.n_basis):
            b = basis[:, i].astype(complex)
            comps.append(b)       # B_k → width  (g0 = re_k)
            comps.append(1j * b)  # i·B_k → mass shift (g0 = im_k)
        return comps

    # ── default parameters ───────────────────────────────────────

    def get_defaults(self):
        return {}  # mass is fixed by transform; gamma names are free variables

    # ── mass/width transform ─────────────────────────────────────

    def make_mass_width_transform(self):
        """Fix mass to config value; gamma names are free params."""
        return _SplineMassFixTransform(
            f"{self.name}_mass",
            mass_default=float(self.kwargs.get("mass", 1.0)),
        )
