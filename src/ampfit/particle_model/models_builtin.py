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

    def get_gamma_defaults(self):
        return [float(self.kwargs.get("width", 1.0))]

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

    def get_gamma_defaults(self):
        return [float(self.kwargs.get(f"g_{i}", 0.1)) for i in range(self.get_gamma_count())]

    def gamma(self, m):
        return [np.ones_like(m) + 0j] * self.get_gamma_count()


# ── Gounaris-Sakurai (rho-like) ──────────────────────────────────

@register_model("GS_rho")
class GSRhoModel(BaseModel):
    """Gounaris-Sakurai lineshape (see ``gs_rho_model.py`` for full impl)."""

    def get_defaults(self):
        return {f"{self.name}_mass": float(self.kwargs.get("mass", 0.775)),
                f"{self.name}_width": float(self.kwargs.get("width", 0.149))}

    def get_gamma_defaults(self):
        return [float(self.kwargs.get("width", 0.149))]

    def gamma(self, m):
        return [np.ones_like(m) + 0j] * self.get_gamma_count()


# ── Bugg lineshape (sigma/f0(500)) ──────────────────────────────

# ── Bugg lineshape ──────────────────────────────────────────────

@register_model("Bugg")
class BuggModel(BaseModel):
    """Bugg parametrisation (placeholder — see ``bugg_model.py`` for real impl)."""

    def get_gamma_defaults(self):
        return [float(self.kwargs.get("width", 0.1))]

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

    def get_gamma_defaults(self):
        return [float(self.kwargs.get("width", 0.1))]

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
