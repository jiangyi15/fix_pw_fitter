"""
Cubic spline interpolation for particle models with a *k* parameter.

The fit parameter *k* controls a set of ``g_a(k)`` couplings that
blend *N* pre-computed gamma-table rows via cubic spline weights::

    Gamma_k(m) = Sum_a g_a(k) * Gamma_a(m)

where ``g_a(k)`` are the spline basis weights (all *N* grid points
contribute, unlike Catmull-Rom which only uses 4 adjacent points).

The spline uses "not-a-knot" boundary conditions and is built from
the :func:`spline_basis_matrix` pre-computed coefficient matrix.

YAML usage::

    particle:
      sigma:
        mass: 0.5
        model: ExpSpline           # spline-based version
        k: 1.0
        k_range: [0.1, 5.0]
        n_interp: 50
"""

import numpy as np
from .base import BaseModel, register_model
from ampfit.param_constraint import Transform


# ═══════════════════════════════════════════════════════════════════
# Spline basis matrix and helpers
# ═══════════════════════════════════════════════════════════════════

def spline_basis_matrix(xi, bc_type="not-a-knot"):
    """Build cubic spline coefficient matrix.

    Returns an array of shape ``(M, 4, N)`` where ``M = N-1`` intervals
    and the 4 coefficients per interval are ``(a, b, c, d)`` for the
    polynomial ``f(x) = a + b*x + c*x^2 + d*x^3``.

    Each ``h[i, :, j]`` gives the contribution of point value ``y[j]``
    to the coefficients of interval ``i``::

        a_i = sum_j h[i, 0, j] * y[j]
        b_i = sum_j h[i, 1, j] * y[j]
        c_i = sum_j h[i, 2, j] * y[j]
        d_i = sum_j h[i, 3, j] * y[j]

    Then for ``x`` in interval ``i``::

        f(x) = a_i + b_i*x + c_i*x^2 + d_i*x^3
             = sum_j w_j(x) * y[j]

    where ``w_j(x) = h[i,0,j] + h[i,1,j]*x + h[i,2,j]*x^2 + h[i,3,j]*x^3``.

    Parameters
    ----------
    xi : array-like of length N
        Grid points (must be strictly increasing).
    bc_type : str
        Boundary condition: ``"not-a-knot"`` (default) or ``"natural"``.

    Returns
    -------
    h_matrix : ndarray, shape (N-1, 4, N)
    """
    N = len(xi)
    xi = np.asarray(xi, dtype=np.float64)
    hi = np.diff(xi)  # N-1 intervals

    # ── Solve for second derivatives via tridiagonal system ──────
    # h_matrix[i] * c_i = y_matrix[i] * y_i
    h_mat = np.zeros((N, N))
    y_mat = np.zeros((N, N))

    if bc_type == "not-a-knot":
        h_mat[0, 0] = -hi[1]
        h_mat[0, 1] = hi[0] + hi[1]
        h_mat[0, 2] = -hi[0]
    elif bc_type == "natural":
        h_mat[0, 0] = 1.0
    else:
        raise ValueError(f"bc_type={bc_type!r} not in ('not-a-knot', 'natural')")

    for i in range(1, N - 1):
        h_mat[i, i - 1] = hi[i - 1]
        h_mat[i, i] = 2.0 * (hi[i - 1] + hi[i])
        h_mat[i, i + 1] = hi[i]
        y_mat[i, i - 1] = 6.0 / hi[i - 1]
        y_mat[i, i] = -6.0 * (1.0 / hi[i] + 1.0 / hi[i - 1])
        y_mat[i, i + 1] = 6.0 / hi[i]

    if bc_type == "not-a-knot":
        h_mat[-1, -3] = -hi[-1]
        h_mat[-1, -2] = hi[-1] + hi[-2]
        h_mat[-1, -1] = -hi[-2]
    elif bc_type == "natural":
        h_mat[-1, -1] = 1.0

    h_mat_inv = np.linalg.inv(h_mat)
    hy_mat = h_mat_inv @ y_mat  # (N, N): second-derivative solver

    # ── Convert to polynomial coefficients (a, b, c, d) ──────────
    # f(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3
    hi_col = hi[:, np.newaxis]  # (N-1, 1)
    I_mat = np.eye(N)

    c_i = hy_mat[:-1] / 2.0                    # (N-1, N)
    d_i = (hy_mat[1:] - hy_mat[:-1]) / (6.0 * hi_col)  # (N-1, N)
    b_i = (I_mat[1:] - I_mat[:-1]) / hi_col - c_i * hi_col - d_i * hi_col ** 2
    a_i = I_mat[:-1]

    # ── Convert to polynomial at x: a + b*x + c*x^2 + d*x^3 ─────
    x1 = xi[:-1, np.newaxis]  # (N-1, 1)
    x2 = x1 ** 2
    x3 = x2 * x1

    a = a_i - b_i * x1 + c_i * x2 - d_i * x3
    b = b_i - 2.0 * c_i * x1 + 3.0 * d_i * x2
    c = c_i - 3.0 * d_i * x1
    d = d_i

    h_matrix = np.stack([a, b, c, d], axis=1)  # (N-1, 4, N)
    return h_matrix


def spline_weights(k, h_matrix, k_grid):
    """Compute cubic spline weights at a given *k*.

    Returns
    -------
    g0 : ndarray, shape (N,)
        Weight ``w_j(k)`` for each grid point ``j``.
    idx : int
        Interval index containing *k* (clamped to [0, N-2]).
    """
    N = len(k_grid)
    k = float(k)
    k_min, k_max = float(k_grid[0]), float(k_grid[-1])
    delta_k = (k_max - k_min) / max(N - 1, 1)

    diff = (k - k_min) / delta_k
    idx = int(np.floor(diff))
    idx = max(0, min(idx, N - 2))

    # Coefficients for this interval
    a = h_matrix[idx, 0, :]  # (N,)
    b = h_matrix[idx, 1, :]
    c = h_matrix[idx, 2, :]
    d = h_matrix[idx, 3, :]

    g0 = a + k * (b + k * (c + d * k))
    return g0, idx


def spline_weight_deriv(k, h_matrix, k_grid):
    """Derivative ``dw_j(k)/dk`` for each grid point *j*.

    Returns
    -------
    dg_dk : ndarray, shape (N,)
    idx : int
        Interval index.
    """
    N = len(k_grid)
    k = float(k)
    k_min, k_max = float(k_grid[0]), float(k_grid[-1])
    delta_k = (k_max - k_min) / max(N - 1, 1)

    diff = (k - k_min) / delta_k
    idx = int(np.floor(diff))
    idx = max(0, min(idx, N - 2))

    b = h_matrix[idx, 1, :]  # (N,)
    c = h_matrix[idx, 2, :]
    d = h_matrix[idx, 3, :]

    dg_dk = b + k * (2.0 * c + 3.0 * d * k)
    return dg_dk, idx


# ═══════════════════════════════════════════════════════════════════
# Transform: k → spline weights {g_0(k), ..., g_{N-1}(k)}
# ═══════════════════════════════════════════════════════════════════

class KToSplineWeightsTransform(Transform):
    """Transform: k -> spline weights {g_0(k), ..., g_{N-1}(k)}.

    Unlike :class:`KToCRWeightsTransform` which uses 4-point Catmull-Rom,
    this uses full cubic spline interpolation.  All *N* grid points
    contribute at each *k* value.

    Parameters
    ----------
    k_name : str
        Input parameter (e.g. ``"sigma_k"``).
    mass_name : str
        Output mass name — fixed to YAML value.
    g0_names : list of str
        Output weight names (length *N*).
    mass_fixed : float
        Fixed mass value.
    k_min, k_max : float
        Range of *k*.
    bc_type : str
        Spline boundary condition (``"not-a-knot"`` or ``"natural"``).
    """

    _has_inverse = True

    def __init__(self, k_name, mass_name, g0_names,
                 mass_fixed=1.0,
                 k_min=0.1, k_max=5.0,
                 bc_type="not-a-knot"):
        out_names = [mass_name] + list(g0_names)
        super().__init__(input_names=[k_name],
                         output_names=out_names)
        self.k_name = k_name
        self.mass_name = mass_name
        self.g0_names = list(g0_names)
        self.n_k = len(g0_names)
        self.mass_fixed = float(mass_fixed)
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self._delta_k = (float(k_max) - float(k_min)) / max(self.n_k - 1, 1)

        # Pre-compute spline basis matrix
        k_grid = np.linspace(k_min, k_max, self.n_k)
        self._h_matrix = spline_basis_matrix(k_grid, bc_type)
        self._k_grid = k_grid

    def _weights(self, k):
        """Compute spline basis weights and index."""
        g0, idx = spline_weights(k, self._h_matrix, self._k_grid)
        return g0, idx

    def _weight_deriv(self, k):
        """Compute spline weight derivatives and index."""
        dg_dk, idx = spline_weight_deriv(k, self._h_matrix, self._k_grid)
        return dg_dk, idx

    def forward(self, d):
        k = float(d[self.k_name])
        g0, _ = self._weights(k)
        result = {self.mass_name: self.mass_fixed}
        for name, val in zip(self.g0_names, g0):
            result[name] = float(val)
        return result

    def backward(self, grad_out, d_in=None):
        k = float(d_in[self.k_name]) if d_in else 0.0
        dg_dk, _ = self._weight_deriv(k)

        dk = 0.0
        for j, name in enumerate(self.g0_names):
            dg = grad_out.get(name, 0.0)
            if dg != 0.0:
                dk += dg * dg_dk[j]

        return {self.k_name: dk}

    def inverse(self, d):
        """Approximate inverse via peak-weight position.

        If *k* is already in *d*, returns it directly.
        Otherwise finds *k* by locating the peak weight index and
        refining within the interval via ternary search.
        """
        if self.k_name in d:
            mass = float(d.get(self.mass_name, self.mass_fixed))
            return {self.k_name: float(d[self.k_name]),
                    self.mass_name: mass}

        target = np.array([float(d.get(n, 0.0)) for n in self.g0_names])
        peak = int(np.argmax(target))

        if peak <= 0:
            k0 = self.k_min
        elif peak >= self.n_k - 1:
            k0 = self.k_max
        else:
            lo = self.k_min + max(peak - 1, 0) * self._delta_k
            hi = self.k_min + min(peak + 2, self.n_k - 1) * self._delta_k

            def mse(k):
                g0, _ = self._weights(k)
                return float(np.sum((g0 - target) ** 2))

            for _ in range(20):
                m1 = (lo * 2 + hi) / 3.0
                m2 = (lo + hi * 2) / 3.0
                if mse(m1) < mse(m2):
                    hi = m2
                else:
                    lo = m1
            k0 = (lo + hi) / 2.0

        return {self.k_name: k0,
                self.mass_name: self.mass_fixed}


# ═══════════════════════════════════════════════════════════════════
# Base class for spline-interpolated k models
# ═══════════════════════════════════════════════════════════════════

class SplineKModel(BaseModel):
    """Base class for models with cubic-spline-interpolated *k* parameter.

    Subclasses MUST override :meth:`gamma_k` to define the gamma
    function at a specific k value.  The base class handles:

    - k as a fit parameter (``{name}_k`` in g0 array)
    - Spline interpolation over ``k_range`` with ``n_interp`` points
    - Gamma table pre-computation at each k_i (via :meth:`gamma`)
    - ``KToSplineWeightsTransform`` mapping k -> spline basis weights

    Parameters (from YAML config):
        mass        -- nominal mass (m_0)
        width       -- reference g_0 (default 1.0)
        k           -- initial/default k value (default 1.0)
        k_range     -- [k_min, k_max] (default [0.1, 5.0])
        n_interp    -- spline interpolation points (default 50)
        bc_type     -- spline boundary condition (default ``"not-a-knot"``)
    """

    def gamma_k(self, m, k):
        """Gamma(m) at a specific *k*.

        Subclasses MUST override this.

        Parameters
        ----------
        m : ndarray
            Invariant mass grid.
        k : float
            Current k value at this grid point.

        Returns
        -------
        ndarray
            Complex gamma(m) for this k.
        """
        raise NotImplementedError

    # -- k parameter handling ---------------------------------------

    def _k_range(self):
        kr = self.kwargs.get("k_range", [0.1, 5.0])
        return float(kr[0]), float(kr[1])

    def get_gamma_name(self):
        n_k = int(self.kwargs.get("n_interp", 50))
        return [f"{self.name}_gk{i}" for i in range(n_k)]

    def get_defaults(self):
        k0 = float(self.kwargs.get("k", 1.0))
        return {f"{self.name}_k": k0}

    def get_gamma_count(self):
        return int(self.kwargs.get("n_interp", 50))

    def make_mass_width_transform(self):
        n_k  = int(self.kwargs.get("n_interp", 50))
        k_min, k_max = self._k_range()
        m0   = float(self.kwargs.get("mass", 0.775))
        bc   = self.kwargs.get("bc_type", "not-a-knot")

        g0_names = [f"{self.name}_gk{i}" for i in range(n_k)]

        return KToSplineWeightsTransform(
            f"{self.name}_k",
            f"{self.name}_mass",
            g0_names,
            mass_fixed=m0,
            k_min=k_min, k_max=k_max,
            bc_type=bc,
        )

    def gamma(self, m):
        """Pre-computed Gamma(m) at each k-grid point."""
        n_k  = int(self.kwargs.get("n_interp", 50))
        k_min, k_max = self._k_range()

        k_grid = np.linspace(k_min, k_max, n_k)
        return [self.gamma_k(m, ki) for ki in k_grid]


# ═══════════════════════════════════════════════════════════════════
# Concrete models
# ═══════════════════════════════════════════════════════════════════

@register_model("ExpSpline")
class ExpSplineModel(SplineKModel):
    """Exponential lineshape with cubic-spline-interpolated *k*.

    Amplitude: ``A(m) = exp(-k*(m^2 - m_0^2))``.

    The spline interpolation ensures smooth weights across the
    full *k* range, avoiding the Catmull-Rom issue where the
    amplitude deviates from the true exponential when *k* is
    between grid points.

    Parameters (from YAML config):
        mass        -- nominal mass (m_0)
        width       -- reference g_0 (default 1.0)
        k           -- initial/default k value (default 1.0)
        k_range     -- [k_min, k_max] (default [0.1, 5.0])
        n_interp    -- spline interpolation points (default 50)
        bc_type     -- spline boundary condition (default ``\"not-a-knot\"``)
    """

    def gamma_k(self, m, k):
        r"""Gamma(m) for A(m) = exp(-k*(m^2 - m_0^2)).

        From::

            A(m) = 1/(m_0^2 - m^2 - i*m_0*g_0*gamma) = exp(-k*(m^2 - m_0^2))

        we solve::

            gamma(m) = (m_0^2 - m^2 - exp(k*(m^2 - m_0^2))) / (i*m_0*g_0)
        """
        m0 = float(self.kwargs.get("mass", 0.775))
        g0 = float(self.kwargs.get("width", 1.0))
        return (m0 ** 2 - m ** 2 - np.exp(k * (m ** 2 - m0 ** 2))) / (1j * m0 * g0)
