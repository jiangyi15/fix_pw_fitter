"""
SVD-reduced spline-k models.

Same physics as the spline-k models in ``spline_k_model.py`` — the fit
parameter *k* enters through cubic-spline weights ``w(k)`` that blend
gamma-table rows — but the gamma table is compressed with a truncated
SVD so the per-event amplitude runs over ``n_reduce`` terms instead of
``n_interp``::

    G(m)  = sum_j  w_j(k) * Gamma_j(m)                 # full, n_interp terms
          ~ sum_r  w'_r(k) * B_r(m)                    # reduced, n_reduce terms

    w'(k) = U_r^T w(k),      B_r = S_r Vh_r

The SVD is computed on a fine mass grid spanning ``[2*m_pi, m_b - m_pi]``
with ``n_mass_pts`` points (default 10000).

Numerical care:

* gamma is complex while the kernel couplings are real, so the SVD is
  applied to the real augmented matrix ``[Re(G) | Im(G)]`` — the
  projection ``U_r^T`` is then real and flows through the real-valued
  constraint pipeline unchanged.
* the exponential gamma grows like ``exp(k*m^2)`` (up to ~1e57), so a
  raw SVD is dominated by the explosive high-mass tail and cancels
  catastrophically elsewhere.  Three countermeasures are applied:
  the k-mean row (the baseline common to all k) is kept exactly as the
  first basis row (weight 1, via spline partition of unity), the
  residual rows (k) and columns (mass) are scaled to unit norm before
  the SVD ("balanced" SVD, with the scales folded back into the basis
  and projection), and |gamma| is capped at ``gamma_clip`` (default
  1e6).

``w' = U_r^T w(k)`` is *linear* in the spline weights, so gradients
``dL/dk`` propagate exactly through the reduction — the only
approximation is the SVD truncation itself.

NOTE: reduction accuracy depends strongly on the configuration.  For
the exponential model the SVD is excellent over physically relevant
mass ranges (e.g. ``m < 2 GeV``: |A|^2 error ~1e-5 at ``n_reduce=8``)
but poor over the full kinematic range ``[2*m_pi, m_b - m_pi]`` with
large ``k``, because the gamma family spans ~57 decades there.  Keep
``k_range`` moderate or restrict the mass range for accurate reduction.

YAML usage::

    particle:
      sigma:
        mass: 0.5
        model: ExpSplineSVD
        k: 1.0
        k_range: [0.1, 5.0]
        n_interp: 50        # spline k-grid resolution (pre-SVD basis)
        n_reduce: 8         # SVD components to keep
        n_mass_pts: 10000   # fine mass grid for the SVD
        gamma_clip: 1e6     # |gamma| cap before the SVD
        bc_type: not-a-knot
"""

import numpy as np
from .base import BaseModel, register_model
from ampfit.param_constraint import Transform
from ampfit.particle_model.spline_k_model import (
    SplineKModel,
    spline_basis_matrix, spline_weights, spline_weight_deriv,
)

# Mass range for the SVD fine grid: [2*m_pi, m_b - m_pi].
M_PION = 0.1396
M_B_MESON = 5.279


def balanced_mean_center_svd(G, clip=1e6, mean_center=True):
    """Clip, mean-center and balance a complex gamma row set, then SVD.

    Shared by the 1-parameter (``SVDSplineKModel``) and 2-parameter
    (``svd_2d_spline_k_model``) reductions:

    1. |gamma| is capped at *clip* (phase-preserving) so the explosive
       exp(k·m²) tail does not dominate the SVD.
    2. the *k-mean row* (baseline common to all parameter points) is
       subtracted, keeping the common offset out of the truncated basis.
    3. rows (parameters) and columns (mass) of the residual are scaled
       to unit norm ("balanced") so no single parameter or mass point
       dominates.

    Parameters
    ----------
    G : ndarray, shape (n_rows, n_mass)
        Complex gamma table, one row per parameter-grid point.
    clip : float
        |gamma| cap (0 disables).
    mean_center : bool
        Subtract the row-mean baseline before the SVD.

    Returns
    -------
    dict with ``U, S, Vh`` (SVD of the balanced residual),
    ``row_scale`` (n_rows,), ``col_scale`` (2·n_mass,),
    ``mean_row`` (1, 2·n_mass), ``n_mass``.
    """
    G = np.asarray(G, dtype=np.complex128)
    n_mass = G.shape[1]
    if clip > 0:
        mag = np.abs(G)
        big = mag > clip
        if np.any(big):
            G = G.copy()
            G[big] *= clip / mag[big]
    M = np.concatenate([G.real, G.imag], axis=1)          # (n_rows, 2n)
    mean_row = M.mean(axis=0, keepdims=True) if mean_center else None
    M_res = M if mean_row is None else M - mean_row
    row_scale = np.linalg.norm(M_res, axis=1, keepdims=True)
    col_scale = np.linalg.norm(M_res, axis=0, keepdims=True)
    row_scale = np.where(row_scale > 0, row_scale, 1.0)
    col_scale = np.where(col_scale > 0, col_scale, 1.0)
    U, S, Vh = np.linalg.svd(M_res / row_scale / col_scale,
                             full_matrices=False)
    return dict(U=U, S=S, Vh=Vh,
                row_scale=row_scale[:, 0], col_scale=col_scale[0],
                mean_row=mean_row, n_mass=n_mass)


def build_reduced_basis(svd, n_rows, n_reduce, mean_center=True):
    """Assemble basis rows and k-projection from an SVD dict.

    Returns ``(basis, projection, n_svd)``:

    * ``basis``     (n_reduce, 2·n_mass) — row 0 is the k-mean baseline
      when *mean_center* (exact, weight 1), remaining rows are the
      residual components ``(S_r·Vh_r)·Dc``.
    * ``projection`` (n_reduce, n_rows) — row 0 is ones (baseline
      weight = 1 by spline partition of unity), rows 1.. are
      ``U_rᵀ·Dr``.
    * ``n_svd``     number of residual SVD components kept.
    """
    n_svd = int(min(max(n_reduce - (1 if mean_center else 0), 1), n_rows))
    B_res = (svd["S"][:n_svd, None] * svd["Vh"][:n_svd]) * svd["col_scale"]
    P_res = (svd["U"][:, :n_svd].T * svd["row_scale"])
    if mean_center:
        assert svd["mean_row"] is not None
        basis = np.vstack([svd["mean_row"], B_res])
        projection = np.vstack([np.ones((1, n_rows)), P_res])
        return basis, projection, n_svd, 1 + n_svd
    return B_res, P_res, n_svd, n_svd


# ═══════════════════════════════════════════════════════════════════
# Transform: k → SVD-reduced weights {h_0(k), ..., h_{r-1}(k)}
# ═══════════════════════════════════════════════════════════════════

class KToSVDWeightsTransform(Transform):
    """Transform: k -> SVD-reduced gamma weights.

    Composes the cubic-spline weights ``w(k)`` (length *n_k*) with the
    SVD projection ``P = U_r^T`` (shape ``n_reduce × n_k``)::

        w'(k) = P @ w(k)

    The reduced weights multiply the SVD basis rows ``B_r`` to give the
    same gamma sum as the full spline model up to the SVD truncation.

    Parameters
    ----------
    k_name : str
        Input parameter (e.g. ``"sigma_k"``).
    mass_name : str
        Output mass name — fixed to YAML value.
    out_names : list of str
        Output reduced-weight names (length *n_reduce*).
    projection : ndarray, shape (n_reduce, n_k)
        Real SVD projection matrix ``U_r^T``.
    mass_fixed : float
        Fixed mass value.
    k_min, k_max : float
        Range of *k*.
    bc_type : str
        Spline boundary condition (``"not-a-knot"`` or ``"natural"``).
    """

    _has_inverse = True

    def __init__(self, k_name, mass_name, out_names,
                 projection,
                 mass_fixed=1.0,
                 k_min=0.1, k_max=5.0,
                 bc_type="not-a-knot"):
        out_names = list(out_names)
        super().__init__(input_names=[k_name],
                         output_names=[mass_name] + out_names)
        self.k_name = k_name
        self.mass_name = mass_name
        self.out_names = out_names
        self.n_out = len(out_names)
        self.mass_fixed = float(mass_fixed)
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self._projection = np.asarray(projection, dtype=np.float64)
        self.n_k = self._projection.shape[1]

        k_grid = np.linspace(k_min, k_max, self.n_k)
        self._h_matrix = spline_basis_matrix(k_grid, bc_type)
        self._k_grid = k_grid
        self._delta_k = (float(k_max) - float(k_min)) / max(self.n_k - 1, 1)

    def _weights(self, k):
        """Reduced spline weights ``w'(k) = P @ w(k)`` (n_reduce,)."""
        w, idx = spline_weights(k, self._h_matrix, self._k_grid)
        return self._projection @ w, idx

    def _weight_deriv(self, k):
        """Derivatives ``dw'(k)/dk = P @ dw(k)/dk`` (n_reduce,)."""
        dw, idx = spline_weight_deriv(k, self._h_matrix, self._k_grid)
        return self._projection @ dw, idx

    def forward(self, d):
        k = float(d[self.k_name])
        w, _ = self._weights(k)
        result = {self.mass_name: self.mass_fixed}
        for name, val in zip(self.out_names, w):
            result[name] = float(val)
        return result

    def backward(self, grad_out, d_in=None):
        """dk = sum_r dL/dw'_r * dw'_r/dk (exact, linear in projection)."""
        k = float(d_in[self.k_name]) if d_in else 0.0
        dwp, _ = self._weight_deriv(k)

        dk = 0.0
        for r, name in enumerate(self.out_names):
            dg = grad_out.get(name, 0.0)
            if dg != 0.0:
                dk += dg * dwp[r]
        return {self.k_name: dk}

    def inverse(self, d):
        """Recover *k* from reduced weights via ternary search."""
        if self.k_name in d:
            mass = float(d.get(self.mass_name, self.mass_fixed))
            return {self.k_name: float(d[self.k_name]),
                    self.mass_name: mass}

        target = np.array([float(d.get(n, 0.0)) for n in self.out_names])

        def mse(k):
            w, _ = self._weights(k)
            return float(np.sum((w - target) ** 2))

        lo, hi = self.k_min, self.k_max
        for _ in range(60):
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
# Base class for SVD-reduced spline-k models
# ═══════════════════════════════════════════════════════════════════

class SVDSplineKModel(SplineKModel):
    """Base class for SVD-compressed spline-interpolated *k* models.

    Subclasses MUST override :meth:`gamma_k` (as for :class:`SplineKModel`).
    The base class compresses the full gamma table via SVD::

        n_interp  rows →  n_reduce  basis rows
        w(k) (n_interp weights) →  w'(k) = U_r^T w(k)  (n_reduce weights)

    Parameters (from YAML config):
        mass        -- nominal mass (m_0)
        width       -- reference g_0 (default 1.0)
        k           -- initial/default k value (default 1.0)
        k_range     -- [k_min, k_max] (default [0.1, 5.0])
        n_interp    -- spline interpolation points (default 50)
        n_reduce    -- total number of gamma rows kept: 1 k-mean
                       baseline + ``n_reduce - 1`` SVD components
                       (default 8, clamped to n_interp + 1)
        n_mass_pts  -- fine mass grid for the SVD (default 10000),
                       spanning ``m_range``
        m_range     -- [lo, hi] mass range for the SVD grid (default
                       ``[2*m_pi, m_b - m_pi]``).  Restrict this to the
                       physically relevant region (e.g. ``[0.3, 2.0]``)
                       for accurate reduction of the exponential model.
        gamma_clip  -- cap |gamma| at this value before the SVD
                       (default 1e6; large enough to preserve the
                       exponential tail — the amplitude is already ~0
                       where gamma explodes, so the cap is safe)
        mean_center -- subtract the k-mean row (exact baseline, weight 1)
                       before the SVD (default True).  Uses 1 of the
                       ``n_reduce`` rows for the baseline; set False to
                       spend all rows on SVD components.
        bc_type     -- spline boundary condition (default ``"not-a-knot"``)
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        n_k = int(self.kwargs.get("n_interp", 50))
        n_reduce = int(self.kwargs.get("n_reduce", 8))
        n_mass_pts = int(self.kwargs.get("n_mass_pts", 10000))
        k_min, k_max = self._k_range()
        bc = self.kwargs.get("bc_type", "not-a-knot")

        mr = self.kwargs.get("m_range", None)
        if mr is None:
            m_min, m_max = 2.0 * M_PION, M_B_MESON - M_PION
        else:
            m_min, m_max = float(mr[0]), float(mr[1])
        self._m_range = (m_min, m_max)
        m_fine = np.linspace(m_min, m_max, n_mass_pts)
        k_grid = np.linspace(k_min, k_max, n_k)

        # Full complex gamma rows: (n_k, n_mass_pts)
        G = np.stack([self.gamma_k(m_fine, ki) for ki in k_grid])
        G = np.asarray(G, dtype=np.complex128)

        # ── Balanced, mean-centered SVD ────────────────────────────
        # The exponential gamma rows span ~1e57 (they grow like
        # exp(k·m²) at high m), so a raw SVD is dominated by the
        # explosive high-mass region and cancels catastrophically at
        # low mass.  See :func:`balanced_mean_center_svd` — the k-mean
        # baseline is kept exactly (weight 1), rows/columns balanced.
        mean_center = bool(self.kwargs.get("mean_center", True))
        gamma_clip = float(self.kwargs.get("gamma_clip", 1e6))
        svd = balanced_mean_center_svd(G, clip=gamma_clip,
                                       mean_center=mean_center)
        basis, projection, n_svd, n_reduce_tot = build_reduced_basis(
            svd, n_k, n_reduce, mean_center=mean_center)

        self.n_reduce = n_reduce_tot
        self._k_grid = k_grid
        self._m_fine = m_fine
        self._n_mass_pts = n_mass_pts
        self._basis = basis
        self._projection = projection
        self._bc = bc

    def gamma(self, m):
        """Interpolate the reduced SVD basis rows onto *m*."""
        m = np.asarray(m, dtype=float)
        n = self._n_mass_pts
        rows = []
        for i in range(self.n_reduce):
            row = self._basis[i]
            re = np.interp(m, self._m_fine, row[:n])
            im = np.interp(m, self._m_fine, row[n:])
            rows.append(re + 1j * im)
        return rows

    def get_gamma_count(self):
        return self.n_reduce

    def get_gamma_name(self):
        return [f"{self.name}_sr{i}" for i in range(self.n_reduce)]

    def make_mass_width_transform(self):
        k_min, k_max = self._k_range()
        m0 = float(self.kwargs.get("mass", 0.775))
        return KToSVDWeightsTransform(
            f"{self.name}_k",
            f"{self.name}_mass",
            self.get_gamma_name(),
            projection=self._projection,
            mass_fixed=m0,
            k_min=k_min, k_max=k_max,
            bc_type=self._bc,
        )


# ═══════════════════════════════════════════════════════════════════
# Concrete model
# ═══════════════════════════════════════════════════════════════════

@register_model("ExpSplineSVD")
class ExpSplineSVDModel(SVDSplineKModel):
    """Exponential lineshape with SVD-reduced spline-interpolated *k*.

    Same physics as :class:`ExpSplineModel` (``A(m) = exp(-k*(m^2 - m_0^2))``)
    but the ``n_interp`` gamma rows are compressed to ``n_reduce`` via
    the truncated SVD described in the module docstring.

    YAML usage::

        particle:
          sigma:
            mass: 0.5
            model: ExpSplineSVD
            k: 1.0
            k_range: [0.1, 5.0]
            n_interp: 50
            n_reduce: 8
            n_mass_pts: 10000
    """

    def gamma_k(self, m, k):
        r"""Gamma(m) for A(m) = exp(-k*(m^2 - m_0^2)).

        Same formula as ``ExpSplineModel``::

            gamma(m) = (m_0^2 - m^2 - exp(k*(m^2 - m_0^2))) / (i*m_0*g_0)
        """
        m0 = float(self.kwargs.get("mass", 0.775))
        g0 = float(self.kwargs.get("width", 1.0))
        return (m0 ** 2 - m ** 2 - np.exp(k * (m ** 2 - m0 ** 2))) / (1j * m0 * g0)
