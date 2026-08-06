"""
Two-parameter SVD-reduced spline model:  A(m) = exp(-(a + b*i)*(m^2 - m_0^2)).

The fit parameters *a* (decay rate) and *b* (oscillation rate) enter
through tensor-product cubic-spline weights over a fine ``a × b`` grid
(``n_a·n_b`` points)::

    w_{ij}(a,b) = w_i(a) · w_j(b)          (2D tensor spline)

    Gamma(m) = sum_{i,j} w_{ij}(a,b) · Gamma_{ij}(m)     # full
             ~ sum_r  w'_r(a,b) · B_r(m)                 # reduced

    w'(a,b) = P @ vec(w(a,b)),     B_r = S_r Vh_r

The gamma table is compressed with the same balanced, mean-centered,
clipped SVD as ``ExpSplineSVD`` (see
``ampfit.particle_model.svd_spline_k_model``).  The projection is
linear, so gradients ``dL/da``, ``dL/db`` propagate exactly.

YAML usage::

    particle:
      sigma:
        mass: 0.5
        model: Exp2DSplineSVD
        a: 1.0
        b: 0.5
        a_range: [0.1, 2.0]
        b_range: [0.1, 2.0]
        n_a: 12
        n_b: 12
        n_reduce: 18       # total rows = 1 baseline + n_reduce-1 SVD
        n_mass_pts: 10000
        m_range: [0.3, 2.0]
        gamma_clip: 1e6
"""

import numpy as np
from .base import BaseModel, register_model
from ampfit.param_constraint import Transform
from ampfit.particle_model.spline_k_model import (
    spline_basis_matrix, spline_weights, spline_weight_deriv,
)
from ampfit.particle_model.svd_spline_k_model import (
    M_PION, M_B_MESON,
    balanced_mean_center_svd, build_reduced_basis,
)


# ═══════════════════════════════════════════════════════════════════
# Transform: (a, b) → SVD-reduced weights
# ═══════════════════════════════════════════════════════════════════

class KToSVDWeights2DTransform(Transform):
    """Transform: (a, b) -> SVD-reduced gamma weights.

    Composes the 2D tensor-product cubic-spline weights ``w(a,b)``
    (length ``n_a·n_b``, rows ordered a-major) with the SVD projection
    ``P = [1; U_rᵀ·Dr]`` (shape ``n_reduce × n_a·n_b``)::

        w'(a,b) = P @ vec(w_a(a) ⊗ w_b(b))

    Row 0 of *P* is ones — the k-mean baseline weight is 1 by the
    spline partition of unity (both 1D weight sets sum to 1).

    Parameters
    ----------
    a_name, b_name : str
        Input parameter names (e.g. ``"sigma_a"``, ``"sigma_b"``).
    mass_name : str
        Output mass name — fixed to YAML value.
    out_names : list of str
        Output reduced-weight names (length *n_reduce*).
    projection : ndarray, shape (n_reduce, n_a·n_b)
        Real SVD projection.
    mass_fixed : float
        Fixed mass value.
    a_range, b_range : (float, float)
        Parameter ranges.
    n_a, n_b : int
        Grid resolutions per parameter.
    bc_type : str
        Spline boundary condition.
    """

    _has_inverse = True

    def __init__(self, a_name, b_name, mass_name, out_names,
                 projection,
                 mass_fixed=1.0,
                 a_range=(0.1, 2.0), b_range=(0.1, 2.0),
                 n_a=12, n_b=12,
                 bc_type="not-a-knot"):
        out_names = list(out_names)
        super().__init__(input_names=[a_name, b_name],
                         output_names=[mass_name] + out_names)
        self.a_name = a_name
        self.b_name = b_name
        self.mass_name = mass_name
        self.out_names = out_names
        self.mass_fixed = float(mass_fixed)
        self._projection = np.asarray(projection, dtype=np.float64)

        self.a_min, self.a_max = float(a_range[0]), float(a_range[1])
        self.b_min, self.b_max = float(b_range[0]), float(b_range[1])
        self.n_a = int(n_a)
        self.n_b = int(n_b)
        self.n_rows = self._projection.shape[1]
        assert self.n_rows == self.n_a * self.n_b, \
            "projection rows != n_a·n_b"

        self._a_grid = np.linspace(self.a_min, self.a_max, self.n_a)
        self._b_grid = np.linspace(self.b_min, self.b_max, self.n_b)
        self._h_a = spline_basis_matrix(self._a_grid, bc_type)
        self._h_b = spline_basis_matrix(self._b_grid, bc_type)

    # -- 2D tensor spline weights ------------------------------------

    def _weights(self, a, b):
        """Reduced weights w'(a,b) = P @ vec(w_a ⊗ w_b)."""
        wa, _ = spline_weights(a, self._h_a, self._a_grid)
        wb, _ = spline_weights(b, self._h_b, self._b_grid)
        w = np.outer(wa, wb).ravel()          # (n_a·n_b,), a-major
        return self._projection @ w

    def _weight_deriv(self, a, b):
        """(dw'/da, dw'/db) = (P @ vec(dwa⊗wb), P @ vec(wa⊗dwb))."""
        wa, _ = spline_weights(a, self._h_a, self._a_grid)
        wb, _ = spline_weights(b, self._h_b, self._b_grid)
        dwa, _ = spline_weight_deriv(a, self._h_a, self._a_grid)
        dwb, _ = spline_weight_deriv(b, self._h_b, self._b_grid)
        dA = self._projection @ np.outer(dwa, wb).ravel()
        dB = self._projection @ np.outer(wa, dwb).ravel()
        return dA, dB

    # -- Transform interface -----------------------------------------

    def forward(self, d):
        a = float(d[self.a_name])
        b = float(d[self.b_name])
        w = self._weights(a, b)
        result = {self.mass_name: self.mass_fixed}
        for name, val in zip(self.out_names, w):
            result[name] = float(val)
        return result

    def backward(self, grad_out, d_in=None):
        """da, db = sum_r dL/dw'_r · dw'_r/d(.) — exact, linear."""
        a = float(d_in[self.a_name]) if d_in else 0.0
        b = float(d_in[self.b_name]) if d_in else 0.0
        dA, dB = self._weight_deriv(a, b)

        da = db = 0.0
        for r, name in enumerate(self.out_names):
            dg = grad_out.get(name, 0.0)
            if dg != 0.0:
                da += dg * dA[r]
                db += dg * dB[r]
        return {self.a_name: da, self.b_name: db}

    def inverse(self, d):
        """Recover (a, b) from reduced weights.

        Coarse scan over the (a, b) grid followed by coordinate-wise
        ternary refinement.
        """
        if self.a_name in d and self.b_name in d:
            mass = float(d.get(self.mass_name, self.mass_fixed))
            return {self.a_name: float(d[self.a_name]),
                    self.b_name: float(d[self.b_name]),
                    self.mass_name: mass}

        target = np.array([float(d.get(n, 0.0)) for n in self.out_names])

        def mse(a, b):
            w = self._weights(a, b)
            return float(np.sum((w - target) ** 2))

        # Coarse scan
        best_a, best_b, best_mse = self.a_min, self.b_min, np.inf
        for a in np.linspace(self.a_min, self.a_max, 31):
            for b in np.linspace(self.b_min, self.b_max, 31):
                e = mse(a, b)
                if e < best_mse:
                    best_mse, best_a, best_b = e, a, b

        # Coordinate-wise ternary refinement
        lo_a, hi_a = self.a_min, self.a_max
        lo_b, hi_b = self.b_min, self.b_max
        a, b = best_a, best_b
        for _ in range(8):
            for _ in range(25):           # refine a (b fixed)
                m1 = (lo_a * 2 + hi_a) / 3.0
                m2 = (lo_a + hi_a * 2) / 3.0
                if mse(m1, b) < mse(m2, b):
                    hi_a = m2
                else:
                    lo_a = m1
            a = (lo_a + hi_a) / 2.0
            for _ in range(25):           # refine b (a fixed)
                m1 = (lo_b * 2 + hi_b) / 3.0
                m2 = (lo_b + hi_b * 2) / 3.0
                if mse(a, m1) < mse(a, m2):
                    hi_b = m2
                else:
                    lo_b = m1
            b = (lo_b + hi_b) / 2.0
            lo_a = max(lo_a, a - 0.2 * (self.a_max - self.a_min))
            hi_a = min(hi_a, a + 0.2 * (self.a_max - self.a_min))
            lo_b = max(lo_b, b - 0.2 * (self.b_max - self.b_min))
            hi_b = min(hi_b, b + 0.2 * (self.b_max - self.b_min))

        return {self.a_name: a, self.b_name: b,
                self.mass_name: self.mass_fixed}


# ═══════════════════════════════════════════════════════════════════
# Base class for 2D-parameter SVD spline models
# ═══════════════════════════════════════════════════════════════════

class SVDSplineKModel2D(BaseModel):
    """Base for SVD-compressed models with two spline-interpolated
    parameters (*a*, *b*).

    Subclasses MUST override :meth:`gamma_k` with signature
    ``gamma_k(self, m, a, b)``.  The base class:

    * builds the fine ``a × b`` gamma grid (``n_a·n_b`` rows),
    * compresses it via the balanced, mean-centered SVD to
      ``n_reduce`` basis rows,
    * provides the (a, b) → reduced-weights transform.

    Parameters (from YAML config):
        mass        -- nominal mass (m_0)
        width       -- reference g_0 (default 1.0)
        a, b        -- initial parameter values (default 1.0, 0.5)
        a_range, b_range -- parameter ranges (default [0.1, 2.0])
        n_a, n_b    -- grid resolutions (default 12, 12)
        n_reduce    -- total rows kept (default 18)
        n_mass_pts  -- fine mass grid for the SVD (default 10000)
        m_range     -- SVD mass range (default [2*m_pi, m_b - m_pi])
        gamma_clip  -- |gamma| cap before the SVD (default 1e6)
        mean_center -- keep the parameter-mean baseline exactly
                       (default True)
    """

    def amplitude_k(self, m, a, b):
        """Full complex physics amplitude A(m) at parameters (a, b).

        Subclasses MUST override this.  The base :meth:`gamma_k`
        derives the running-width gamma rows automatically.
        """
        raise NotImplementedError

    def gamma_k(self, m, a, b):
        """Gamma(m) at (a, b), derived from :meth:`amplitude_k`.

        Fixed corrected form (pure), no pure_exp option::

            gamma(m) = (m_0^2 - m^2 - 1/A(m)) / (i*m_0)
        """
        from .base import gamma_from_amplitude
        m0 = float(self.kwargs.get("mass", 0.775))
        A = self.amplitude_k(m, a, b)
        return gamma_from_amplitude(m, m0, A, 1.0, True)

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        n_a = int(self.kwargs.get("n_a", 12))
        n_b = int(self.kwargs.get("n_b", 12))
        n_reduce = int(self.kwargs.get("n_reduce", 18))
        n_mass_pts = int(self.kwargs.get("n_mass_pts", 10000))
        ar = self.kwargs.get("a_range", [0.1, 2.0])
        br = self.kwargs.get("b_range", [0.1, 2.0])
        self._a_range = (float(ar[0]), float(ar[1]))
        self._b_range = (float(br[0]), float(br[1]))
        bc = self.kwargs.get("bc_type", "not-a-knot")

        mr = self.kwargs.get("m_range", None)
        if mr is None:
            m_min, m_max = 2.0 * M_PION, M_B_MESON - M_PION
        else:
            m_min, m_max = float(mr[0]), float(mr[1])
        self._m_range = (m_min, m_max)
        m_fine = np.linspace(m_min, m_max, n_mass_pts)

        a_grid = np.linspace(*self._a_range, n_a)
        b_grid = np.linspace(*self._b_range, n_b)

        # gamma rows, a-major: (n_a·n_b, n_mass_pts)
        rows = [self.gamma_k(m_fine, a, b)
                for a in a_grid for b in b_grid]
        G = np.stack(rows).astype(np.complex128)

        mean_center = bool(self.kwargs.get("mean_center", True))
        gamma_clip = float(self.kwargs.get("gamma_clip", 1e6))
        svd = balanced_mean_center_svd(G, clip=gamma_clip,
                                       mean_center=mean_center)
        basis, projection, n_svd, n_reduce_tot = build_reduced_basis(
            svd, n_a * n_b, n_reduce, mean_center=mean_center)

        self.n_reduce = n_reduce_tot
        self.n_a = n_a
        self.n_b = n_b
        self._a_grid = a_grid
        self._b_grid = b_grid
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
            rows.append(np.interp(m, self._m_fine, row[:n])
                        + 1j * np.interp(m, self._m_fine, row[n:]))
        return rows

    def get_gamma_count(self):
        return self.n_reduce

    def get_gamma_name(self):
        return [f"{self.name}_sr{i}" for i in range(self.n_reduce)]

    def get_defaults(self):
        """Fit parameters: the two spline-interpolated rates *a*, *b*."""
        return {f"{self.name}_a": float(self.kwargs.get("a", 1.0)),
                f"{self.name}_b": float(self.kwargs.get("b", 0.5))}

    def make_mass_width_transform(self):
        m0 = float(self.kwargs.get("mass", 0.775))
        return KToSVDWeights2DTransform(
            f"{self.name}_a", f"{self.name}_b",
            f"{self.name}_mass",
            self.get_gamma_name(),
            projection=self._projection,
            mass_fixed=m0,
            a_range=self._a_range, b_range=self._b_range,
            n_a=self.n_a, n_b=self.n_b,
            bc_type=self._bc,
        )


# ═══════════════════════════════════════════════════════════════════
# Concrete model
# ═══════════════════════════════════════════════════════════════════

@register_model("Exp2DSplineSVD")
class Exp2DSplineSVDModel(SVDSplineKModel2D):
    """2-parameter exponential lineshape with SVD-reduced spline grid.

    Amplitude: ``A(m) = exp(-(a + b*i)*(m^2 - m_0^2))`` with *a*, *b*
    real fit parameters (decay rate, oscillation rate).

    YAML usage::

        particle:
          sigma:
            mass: 0.5
            model: Exp2DSplineSVD
            a: 1.0
            b: 0.5
            a_range: [0.1, 2.0]
            b_range: [0.1, 2.0]
            n_a: 12
            n_b: 12
            n_reduce: 18
    """

    def gamma_k(self, m, a, b):
        r"""gamma(m; a, b) for A(m) = exp(-(a+bi)(m^2 - m_0^2)).

        The gamma couplings are the reduced 2D spline weights (Σ = 1),
        so the rows are the total running width::

            gamma(m) = (m_0^2 - m^2 - exp((a+bi)(m^2-m_0^2))) / (i*m_0)

        which gives ``A(m) = exp(-(a+bi)(m^2-m_0^2))`` exactly for any
        configured width.
        """
        m0 = float(self.kwargs.get("mass", 0.775))
        return (m0 ** 2 - m ** 2
                - np.exp((a + 1j * b) * (m ** 2 - m0 ** 2))) / (1j * m0)
