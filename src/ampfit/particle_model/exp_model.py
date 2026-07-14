"""
Experimental exponential lineshape with CR k-interpolation.

The BW amplitude is parametrised as::

    A(m) = 1 / (m\u2080\u00b2 \u2212 m\u00b2 \u2212 i\u00b7m\u2080\u00b7\u03a3 g_a(k)\u00b7\u0393_a(m))
         = exp(-k\u00b7(m\u00b2 - m\u2080\u00b2))

    g_a(k_b) = \u03b4_{a,b}   (Kronecker delta at k-grid points)

Each :math:`\u0393_a(m) = (m\u2080\u00b2 \u2212 m\u00b2 \u2212 \exp(k_a\u00b7(m\u00b2 - m\u2080\u00b2)))/(i\u00b7m\u2080)`
gives the **exact** amplitude ``\exp(-k_a\u00b7(m\u00b2 - m\u2080\u00b2))`` at ``k = k_a``.

Between grid points, the Catmull-Rom weights ``g_a(k)`` smoothly
blend the *N* pre-computed gamma-table rows::

    \u03a3_a g_a(k)\u00b7\u0393_a(m) \u2248 \u0393_k(m)

The fit parameter *k* selects the CR weights via ``_ExpTransform``.

YAML usage::

    particle:
      sigma:
        mass: 0.5
        model: Exp
        k: 1.0               # initial k
        k_range: [0.1, 5.0]  # k min/max
        n_interp: 50         # CR interpolation points in k
"""

import numpy as np
from .base import BaseModel, register_model
from ampfit.param_constraint import Transform


def _cr_basis(t):
    """Catmull-Rom basis weights at t \u2208 [0, 1].

    Returns (w_{-1}, w_0, w_1, w_2) where::

        f(t) = w_{-1}\u00b7p_{-1} + w_0\u00b7p_0 + w_1\u00b7p_1 + w_2\u00b7p_2
    """
    t2 = t * t
    t3 = t2 * t
    return ((-t + 2.0*t2 - t3) / 2.0,
            (2.0 - 5.0*t2 + 3.0*t3) / 2.0,
            (t + 4.0*t2 - 3.0*t3) / 2.0,
            (-t2 + t3) / 2.0)


def _cr_basis_deriv(t):
    """Derivatives dw/dt of the CR basis weights."""
    t2 = t * t
    return ((-1.0 + 4.0*t - 3.0*t2) / 2.0,
            (-10.0*t + 9.0*t2) / 2.0,
            (1.0 + 8.0*t - 9.0*t2) / 2.0,
            (-2.0*t + 3.0*t2) / 2.0)


def _gamma_exp(m, k, m0, g0=1.0):
    r"""Exact gamma for A(m) = exp(-k\u00b7m\u00b2).

    A(m) = 1/(m\u2080\u00b2 \u2212 m\u00b2 \u2212 i\u00b7m\u2080\u00b7g\u2080\u00b7\u03b3) = exp(-k\u00b7m\u00b2)

    A(m) = exp(-k\u00b7(m\u00b2 - m\u2080\u00b2))  (peaks at 1 when m = m\u2080)

    \u03b3(m) = (m\u2080\u00b2 \u2212 m\u00b2 \u2212 exp(k\u00b7(m\u00b2 - m\u2080\u00b2))) / (i\u00b7m\u2080\u00b7g\u2080)
    """
    return (m0 ** 2 - m ** 2 - np.exp(k * (m ** 2 - m0 ** 2))) / (1j * m0 * g0)


class _ExpTransform(Transform):
    """Transform: k \u2192 CR weights {g_0(k), ..., g_{N-1}(k)}.

    The *N* output ``g_a(k)`` are the Catmull-Rom basis weights that
    select/blend the pre-computed gamma-table rows::

        g_a(k_b) = \u03b4_{a,b}    (exact at k-grid points)

    Parameters
    ----------
    k_name : str
        Input parameter (e.g. ``"sigma_k"``).
    mass_name : str
        Output mass name \u2014 fixed to YAML value.
    g0_names : list of str
        Output weight names (length *N*).
    mass_fixed : float
        Fixed mass value.
    k_min, k_max : float
        Range of *k*.
    """

    _has_inverse = True

    def __init__(self, k_name, mass_name, g0_names,
                 mass_fixed=1.0,
                 k_min=0.1, k_max=5.0):
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

    def _weights(self, k):
        """Compute CR basis weights.

        At grid points ``k = k_a``, the weights satisfy
        ``g_a(k_b) = δ_{a,b}`` via standard edge-duplication
        (same convention as the CUDA Catmull-Rom kernel).

        Returns (g0_array, xbin, t) for backward.
        """
        diff = (k - self.k_min) / self._delta_k
        xbin = int(np.floor(diff))
        xbin = max(0, min(xbin, self.n_k - 2))  # keep 4 neighbours with edge dupe
        t = max(0.0, min(diff - xbin, 1.0))
        w = _cr_basis(t)

        # Edge-duplicated indices (same convention as CUDA CR kernel)
        im1 = max(xbin - 1, 0)
        i2  = min(xbin + 2, self.n_k - 1)

        g0 = np.zeros(self.n_k, dtype=np.float64)
        g0[im1]   += w[0]
        g0[xbin]  += w[1]
        g0[xbin + 1] += w[2]
        g0[i2]    += w[3]
        return g0, xbin, t, im1, i2

    def forward(self, d):
        k = float(d[self.k_name])
        g0, _, _, _, _ = self._weights(k)
        result = {self.mass_name: self.mass_fixed}
        for name, val in zip(self.g0_names, g0):
            result[name] = float(val)
        return result

    def backward(self, grad_out, d_in=None):
        k = float(d_in[self.k_name]) if d_in else 0.0
        _, xbin, t, im1, i2 = self._weights(k)
        dw_dt = _cr_basis_deriv(t)
        idx_list = [im1, xbin, xbin + 1, i2]

        dk = 0.0
        for j, dw in enumerate(dw_dt):
            dg = grad_out.get(self.g0_names[idx_list[j]], 0.0)
            if dg != 0.0:
                dk += dg * dw
        dk /= self._delta_k  # multiply by dt/dk

        return {self.k_name: dk}

    def inverse(self, d):
        target = float(d.get(self.g0_names[0], 0.0))
        lo, hi = self.k_min, self.k_max
        for _ in range(25):
            mid = (lo + hi) / 2.0
            g0, _, _, _, _ = self._weights(mid)
            if g0[0] > target:
                lo = mid
            else:
                hi = mid
        return {self.k_name: (lo + hi) / 2.0,
                self.mass_name: self.mass_fixed}


@register_model("Exp")
class ExpModel(BaseModel):
    """Exponential lineshape with CR-interpolated k.

    The gamma table has *N* rows, one per k-grid point.  Each row
    ``a`` gives the exact ``\u0393_a(m)`` for ``A(m) = exp(-k_a\u00b7m\u00b2)``.
    The fit parameter *k* selects CR weights that blend these rows.

    Parameters (from YAML config):
        mass        \u2014 nominal mass (m\u2080, default 0.775)
        width       \u2014 reference g\u2080 (default 1.0)
        k           \u2014 initial/default k value (default 1.0)
        k_range     \u2014 [k_min, k_max] (default [0.1, 5.0])
        n_interp    \u2014 CR interpolation points (default 50)
    """

    def _k_range(self):
        kr = self.kwargs.get("k_range", [0.1, 5.0])
        return float(kr[0]), float(kr[1])

    def get_gamma_name(self):
        """Gamma-table row names: one per k-grid point."""
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

        g0_names = [f"{self.name}_gk{i}" for i in range(n_k)]

        return _ExpTransform(
            f"{self.name}_k",
            f"{self.name}_mass",
            g0_names,
            mass_fixed=m0,
            k_min=k_min, k_max=k_max,
        )

    def gamma(self, m):
        r"""Pre-computed \u0393_a(m) for each k-grid point.

        Each row gives the exact gamma for ``A(m) = exp(-k_a\u00b7m\u00b2)``::

            \u0393_a(m) = (m\u2080\u00b2 \u2212 m\u00b2 \u2212 exp(k_a\u00b7m\u00b2)) / (i\u00b7m\u2080\u00b7g\u2080)
        """
        n_k  = int(self.kwargs.get("n_interp", 50))
        k_min = float(self.kwargs.get("k_min", 0.1))
        k_max = float(self.kwargs.get("k_max", 5.0))
        m0   = float(self.kwargs.get("mass", 0.775))
        g0   = float(self.kwargs.get("width", 1.0))

        k_grid = np.linspace(k_min, k_max, n_k)
        return [_gamma_exp(m, ki, m0, g0) for ki in k_grid]
