"""
Base class for particle models with Catmull-Rom k-interpolation.

The fit parameter *k* controls a set of ``g_a(k)`` couplings that
blend *N* pre-computed gamma-table rows via CR basis weights:

    g_a(k_b) = delta_{a,b}   (exact at k-grid points)
    Gamma_k(m) = Sum_a g_a(k) * Gamma_a(m)

Subclass ``InterpKModel`` and override ``gamma_k(m, k)`` to define
the gamma function at a specific *k*.

YAML usage::

    particle:
      sigma:
        mass: 0.5
        model: Exp          # or any InterpKModel subclass
        k: 1.0              # initial k
        k_range: [0.1, 5.0] # k min/max
        n_interp: 50        # CR interpolation points in k
"""

import numpy as np
from .base import BaseModel
from ampfit.param_constraint import Transform


def cr_basis(t):
    """Catmull-Rom basis weights at t in [0, 1].

    Returns (w_{-1}, w_0, w_1, w_2) where::

        f(t) = w_{-1}*p_{-1} + w_0*p_0 + w_1*p_1 + w_2*p_2
    """
    t2 = t * t
    t3 = t2 * t
    return ((-t + 2.0*t2 - t3) / 2.0,
            (2.0 - 5.0*t2 + 3.0*t3) / 2.0,
            (t + 4.0*t2 - 3.0*t3) / 2.0,
            (-t2 + t3) / 2.0)


def cr_basis_deriv(t):
    """Derivatives dw/dt of the CR basis weights."""
    t2 = t * t
    return ((-1.0 + 4.0*t - 3.0*t2) / 2.0,
            (-10.0*t + 9.0*t2) / 2.0,
            (1.0 + 8.0*t - 9.0*t2) / 2.0,
            (-2.0*t + 3.0*t2) / 2.0)


class KToCRWeightsTransform(Transform):
    """Transform: k -> CR weights {g_0(k), ..., g_{N-1}(k)}.

    The *N* output ``g_a(k)`` are the Catmull-Rom basis weights that
    select/blend the pre-computed gamma-table rows::

        g_a(k_b) = delta_{a,b}    (exact at k-grid points)

    Parameters
    ----------
    k_name : str
        Input parameter (e.g. ``"sigma_k"``).
    mass_name : str
        Output mass name -- fixed to YAML value.
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
        ``g_a(k_b) = delta_{a,b}`` via standard edge-duplication
        (same convention as the CUDA Catmull-Rom kernel).

        Returns (g0_array, xbin, t, im1, i2).
        """
        diff = (k - self.k_min) / self._delta_k
        xbin = int(np.floor(diff))
        xbin = max(0, min(xbin, self.n_k - 2))
        t = max(0.0, min(diff - xbin, 1.0))
        w = cr_basis(t)

        im1 = max(xbin - 1, 0)
        i2  = min(xbin + 2, self.n_k - 1)

        g0 = np.zeros(self.n_k, dtype=np.float64)
        g0[im1]       += w[0]
        g0[xbin]      += w[1]
        g0[xbin + 1]  += w[2]
        g0[i2]        += w[3]
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
        dw_dt = cr_basis_deriv(t)
        idx_list = [im1, xbin, xbin + 1, i2]

        dk = 0.0
        for j, dw in enumerate(dw_dt):
            dg = grad_out.get(self.g0_names[idx_list[j]], 0.0)
            if dg != 0.0:
                dk += dg * dw
        dk /= self._delta_k

        return {self.k_name: dk}

    def inverse(self, d):
        """Approximate inverse via peak-weight position.

        Finds ``k`` by locating the index of the maximum weight
        and estimating the intra-bin fraction from the CR pattern.

        If the input parameter ``k`` is already present in *d*
        (e.g. when loading from ``values_from_dict``), it is
        returned directly — avoids reconstructing k from missing
        weight values.
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
            # Refine within bin [peak-1, peak+1] via ternary search
            lo = self.k_min + max(peak - 1, 0) * self._delta_k
            hi = self.k_min + min(peak + 2, self.n_k - 1) * self._delta_k

            def mse(k):
                g0, _, _, _, _ = self._weights(k)
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


class InterpKModel(BaseModel):
    """Base class for models with CR-interpolated k parameter.

    Subclasses MUST override :meth:`gamma_k` to define the gamma
    function at a specific k value.  The base class handles:

    - k as a fit parameter (``{name}_k`` in g0 array)
    - CR interpolation over ``k_range`` with ``n_interp`` points
    - Gamma table pre-computation at each k_i (via :meth:`gamma`)
    - ``KToCRWeightsTransform`` mapping k -> CR basis weights

    Parameters (from YAML config):
        mass        -- nominal mass (m_0)
        width       -- reference g_0 (default 1.0)
        k           -- initial/default k value (default 1.0)
        k_range     -- [k_min, k_max] (default [0.1, 5.0])
        n_interp    -- CR interpolation points (default 50)
    """

    def amplitude_k(self, m, k):
        """Full complex physics amplitude A(m) at parameter *k*.

        Subclasses MUST override this — it is the only physics a
        k-interpolated model provides.  The base :meth:`gamma_k`
        derives the running-width gamma rows automatically.

        Parameters
        ----------
        m : ndarray
            Invariant mass grid.
        k : float
            Current k value at this grid point.

        Returns
        -------
        ndarray
            Complex amplitude A(m) for this k.
        """
        raise NotImplementedError

    def gamma_k(self, m, k):
        """Gamma(m) at a specific *k*, derived from :meth:`amplitude_k`.

        With the kernel coupling structure ``Σ g_i·γ_i`` (g_i = the
        interpolation weights, Σ = 1), the row reproducing the physics
        amplitude is the total running width::

            gamma(m) = (m_0^2 - m^2 - 1/A(m)) / (i*m_0)
        """
        from .base import gamma_from_amplitude
        m0 = float(self.kwargs.get("mass", 0.775))
        A = self.amplitude_k(m, k)
        return gamma_from_amplitude(m, m0, A, 1.0, True)

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

        g0_names = [f"{self.name}_gk{i}" for i in range(n_k)]

        return KToCRWeightsTransform(
            f"{self.name}_k",
            f"{self.name}_mass",
            g0_names,
            mass_fixed=m0,
            k_min=k_min, k_max=k_max,
        )

    def gamma(self, m):
        """Pre-computed Gamma(m) at each k-grid point.

        Iterates over the k-grid and calls :meth:`gamma_k` for each.
        """
        n_k  = int(self.kwargs.get("n_interp", 50))
        k_min, k_max = self._k_range()
        m0   = float(self.kwargs.get("mass", 0.775))
        g0   = float(self.kwargs.get("width", 1.0))

        k_grid = np.linspace(k_min, k_max, n_k)
        return [self.gamma_k(m, ki) for ki in k_grid]
