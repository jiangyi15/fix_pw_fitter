"""Built-in particle model implementations.

Each model registers itself via the ``@register_model`` decorator
and is available through ``build_particle(model=...)``.
"""

import numpy as np
from .base import BaseModel, register_model
from ampfit.param_constraint import Transform


# ── Standard Breit-Wigner ────────────────────────────────────────

@register_model("BW")
class BWModel(BaseModel):
    """Standard relativisitic Breit-Wigner lineshape.

    ``gamma(m) = 1``  (constant width).
    """


# ── Fix mass/width transform (auto-fixes mass & width params) ────

class _FixMassWidthTransform(Transform):
    """Pass-through transform that auto-fixes mass and width parameters.

    ``input_names = []``, ``output_names = [mass_name, width_name]``.
    The transform carries the default values so the constraint manager
    can set them as fixed overrides instead of hardcoded 0.0.
    """

    _has_inverse = True

    def __init__(self, mass_name, width_name, mass_default=0.775, width_default=0.1):
        super().__init__(input_names=[], output_names=[mass_name, width_name])
        self._fixed = {mass_name: float(mass_default), width_name: float(width_default)}

    def forward(self, d):
        """Override mass and width with their fixed config values."""
        return {**d, **self._fixed}

    def backward(self, grad_out, d_in=None):
        return {}

    def inverse(self, d):
        return d


# ── Fixed-shape base ────────────────────────────────────────────

class FixedShapeModel(BaseModel):
    """Base for pre-computed fixed-shape models.

    Subclasses override ``fixed_shape(m)`` to return the **complex
    amplitude** :math:`A(m)` (i.e. the full lineshape value, typically
    the product of propagators).

    The base class converts this to ``gamma(m)`` automatically::

        gamma(m) = (m₀² − m² − 1/A(m)) / (i·m₀·g₀)

    so the kernel reproduces the desired shape.

    Mass and width are auto-fixed via Transform (not fitted).
    All shape parameters come from the YAML config.

    Example::

        @register_model("MyProduct")
        class MyProduct(FixedShapeModel):
            def fixed_shape(self, m):
                m0 = float(self.kwargs["mass"])
                g0 = float(self.kwargs.get("width", 0.1))
                # Amplitude = BW₁ · BW₂
                bw1 = 1 / (m0**2 - m**2 - 1j*m0*g0)
                bw2 = 1 / (1.2**2 - m**2 - 1j*1.2*0.05)
                return bw1 * bw2
    """

    def fixed_shape(self, m):
        """Return the complex amplitude A(m).

        Subclasses MUST override this.  Read parameters from
        ``self.kwargs`` (set from YAML config).
        """
        raise NotImplementedError

    def gamma(self, m):
        m0 = float(self.kwargs.get("mass", 0.775))
        g0 = float(self.kwargs.get("width", 0.1))
        A = self.fixed_shape(m)
        # Convert amplitude → gamma(m) for the kernel BW denominator
        #   A = 1/(m₀² − m² − i·m₀·g₀·γ)  →  γ = (m₀² − m² − 1/A) / (i·m₀·g₀)
        gamma_m = (m0**2 - m**2 - 1.0/A) / (1j * m0 * g0)
        return [gamma_m]

    def get_defaults(self):
        return {}  # no free params — all fixed via Transform

    def make_mass_width_transform(self):
        return _FixMassWidthTransform(
            f"{self.name}_mass", f"{self.name}_width",
            mass_default=float(self.kwargs.get("mass", 0.775)),
            width_default=float(self.kwargs.get("width", 0.1)),
        )


# ── Linear amplitude: A(m) = k·(m − m₀) ──────────────────────────

@register_model("linear")
class LinearShapeModel(FixedShapeModel):
    """Linear amplitude ``A(m) = k·(m − m₀)`` (fixed shape).

    A fixed-shape model whose amplitude is a linear function of the
    mass offset.  :class:`FixedShapeModel` converts it to the gamma row
    so the kernel's BW denominator reproduces the shape.  All
    parameters (mass, k) are fixed from the YAML config — nothing is
    fitted.

    YAML example::

        particle:
          my_res:
            mass: 1.0
            k: 0.3
            model: linear
    """

    def fixed_shape(self, m):
        m0 = float(self.kwargs.get("mass", 0.775))
        k = float(self.kwargs.get("k", 1.0))
        A = k * (np.asarray(m, dtype=float) - m0)
        # safe clip: the linear function crosses zero at m = m₀, where
        # 1/A would diverge in the gamma conversion — keep |A| ≥ a_min
        a_min = float(self.kwargs.get("a_min", 1e-6))
        return np.where(np.abs(A) < a_min, np.copysign(a_min, A), A)


# ── One / constant ───────────────────────────────────────────────

@register_model("one")
class OneModel(BaseModel):
    """Constant ``K``-matrix like parametrisation.

    Returns the Gamma value that makes
    ``1 = 1/(m0**2 - m**2 - i m0 g0 Gamma)``

    Mass and width are both fixed (``_FixMassWidthTransform``),
    so no physical defaults needed.
    """

    def get_defaults(self):
        return {}

    def gamma(self, m):
        m0 = self.kwargs["mass"]
        g0 = self.kwargs.get("width", 1.0)
        return [1j * (1 - m0**2 + m**2) / m0 / g0]

    def make_mass_width_transform(self):
        return _FixMassWidthTransform(
            f"{self.name}_mass", f"{self.name}_width",
            mass_default=self.kwargs.get("mass", 0.775),
            width_default=self.kwargs.get("width", 1.0),
        )


# ── Coupled-channel Flatté ───────────────────────────────────────

@register_model("FlatteC")
class FlatteCModel(BaseModel):
    """Flatté-like parametrisation (see ``flatte_model.py`` for full impl)."""

    def get_defaults(self):
        mass = float(self.kwargs.get("mass", 0.775))
        gammas = {f"{self.name}_g{i}": float(self.kwargs.get(f"g_{i}", 0.1))
                  for i in range(self.get_gamma_count())}
        return {f"{self.name}_mass": mass, **gammas}

    def get_gamma_count(self):
        return len(self.kwargs["mass_list"])

    def get_gamma_name(self):
        return [f"{self.name}_g{i}" for i in range(self.get_gamma_count())]

    def gamma(self, m):
        return [np.ones_like(m) + 0j] * self.get_gamma_count()


# ── Gounaris-Sakurai (rho-like) ──────────────────────────────────

@register_model("GS_rho")
class GSRhoModel(BaseModel):
    """Gounaris-Sakurai lineshape (see ``gs_rho_model.py`` for full impl)."""

    def get_defaults(self):
        return {f"{self.name}_mass": float(self.kwargs.get("mass", 0.775)),
                f"{self.name}_width": float(self.kwargs.get("width", 0.149))}

    def gamma(self, m):
        return [np.ones_like(m) + 0j] * self.get_gamma_count()


# ── Bugg lineshape ──────────────────────────────────────────────

@register_model("Bugg")
class BuggModel(BaseModel):
    """Bugg parametrisation (placeholder — see ``bugg_model.py`` for real impl)."""

    def gamma(self, m):
        return [np.ones_like(m) + 0j] * self.get_gamma_count()


# ── Width from external numpy file (linear interpolation) ────────

@register_model("width_linear_npy")
class WidthLinearNPYModel(BaseModel):
    """Energy-dependent width from a ``.npy`` file via linear interpolation.

    Based on the TFPWA ``WidthInterpLinearNpy`` model (``amp/interpolation.py``).

    The file must have columns ``[mass, Re(Pi), Im(Pi)]`` where Pi(m) is the
    complex self-energy.  The ``gamma(m)`` return value encodes::

        gamma(m) = Im(Pi(m)) + i · (Re(Pi(m₀)) − Re(Pi(m)))

    so that the framework computes::

        bw_dom = m₀² − m² − i·m₀·g₀·gamma(m)
               = m₀² − m² − m₀·g₀·(Re(Pi(m))−Re(Pi(m₀))) − i·m₀·g₀·Im(Pi(m))

    which matches the TFPWA ``convert_to_amp`` formula.

    YAML example::

        particle:
          f0(500):
            mass: 0.5
            width: 0.5
            model: width_linear_npy
            file: /path/to/width_table.npy
            width_scale: True   # optional: normalise Im(Pi(m₀)) to 1
    """

    def gamma(self, m):
        data = np.load(self.kwargs["file"])
        mi = data[:, 0]
        fi = data[:, 1] + 1j * data[:, 2]          # complex Pi(m)

        fm  = np.interp(m, mi, fi)                 # Pi(m)
        fm0 = np.interp(self.kwargs["mass"], mi, fi)  # Pi(m₀)

        # gamma = Im(Pi(m)) + i · (Re(Pi(m₀)) − Re(Pi(m)))
        g = np.imag(fm) + 1j * (np.real(fm0) - np.real(fm))

        if self.kwargs.get("width_scale", False) and np.imag(fm0) != 0:
            g = g / np.imag(fm0)

        return [g]


@register_model("GaussianBasis")
class GaussianBasisModel(FixedShapeModel):
    """Gaussian basis function for amplitude expansion.

    Amplitude is a Gaussian centered at *mu* with width *sigma*::

        A(m) = exp(-(m - μ)² / (2·σ²))

    YAML::

        particle:
          gauss_0:
            mu: 0.5
            sigma: 0.2
            model: GaussianBasis
    """

    def fixed_shape(self, m):
        mu = float(self.kwargs.get("mu", 0.775))
        sigma = float(self.kwargs.get("sigma", 0.1))
        return np.exp(-((np.asarray(m) - mu) ** 2) / (2.0 * sigma ** 2))


@register_model("BSplineBasis")
class BSplineBasisModel(FixedShapeModel):
    """B-spline basis function for amplitude expansion.

    Amplitude is a single B-spline basis function centered at *mu*
    with compact support proportional to *sigma* and *order*::

        A(m) ≈ B_k(m)   (k-th B-spline basis, centred at mu, peak = 1)

    Outside the support range, the amplitude is zero.
    *order* controls the smoothness (3 = cubic).

    YAML::

        particle:
          bspl_0:
            mu: 0.5
            sigma: 0.3
            order: 3
            model: BSplineBasis
    """

    def fixed_shape(self, m):
        mu = float(self.kwargs.get("mu", 0.775))
        sigma = float(self.kwargs.get("sigma", 0.1))
        order = int(self.kwargs.get("order", 3))

        # Knots: order+1 intervals on each side of mu, spacing sigma
        half = (order + 1) / 2.0
        knots = np.linspace(mu - half * sigma, mu + half * sigma, order + 2)

        # Clamped B-spline extended knot vector
        t0 = np.full(order + 1, knots[0])
        t1 = np.full(order + 1, knots[-1])
        ext = np.concatenate([t0, knots[1:-1], t1])
        n_all = len(ext) - order - 1  # = 2*order + 1
        k = n_all // 2  # center basis function index

        # Cox-de Boor for a single basis function B_{k,order}
        # Only need order-0 functions B_{k} through B_{k+order}
        B = np.zeros((order + 1, len(np.asarray(m))))
        # Order 0
        for i in range(order + 1):
            mask = (np.asarray(m) >= ext[k + i]) & (np.asarray(m) < ext[k + i + 1])
            B[i, mask] = 1.0
        # Recursion
        for p in range(1, order + 1):
            for i in range(order + 1 - p):
                denom_l = ext[k + i + p] - ext[k + i]
                if denom_l > 0:
                    left = (np.asarray(m) - ext[k + i]) / denom_l * B[i]
                else:
                    left = 0.0
                denom_r = ext[k + i + p + 1] - ext[k + i + 1]
                if denom_r > 0:
                    right = (ext[k + i + p + 1] - np.asarray(m)) / denom_r * B[i + 1]
                else:
                    right = 0.0
                B[i] = left + right

        return np.clip(B[0], 1e-15, None).astype(complex)
