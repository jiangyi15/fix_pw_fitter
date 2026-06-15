"""Built-in particle model implementations.

Each model registers itself via the ``@register_model`` decorator
and is available through ``build_particle(model=...)``.
"""

import numpy as np
from .base import BaseModel, register_model


# ── Standard Breit-Wigner ────────────────────────────────────────

@register_model("BW")
class BWModel(BaseModel):
    """Standard relativisitic Breit-Wigner lineshape.

    ``gamma(m) = 1``  (constant width).
    """


# ── One / constant ───────────────────────────────────────────────

@register_model("one")
class OneModel(BaseModel):
    """Constant ``K``-matrix like parametrisation.

    Returns the Gamma value that makes
    ``1 = 1/(m0**2 - m**2 - i m0 g0 Gamma)``
    """

    def get_gamma_defaults(self):
        return [float(self.kwargs.get("width", 1.0))]

    def gamma(self, m):
        m0 = self.kwargs["mass"]
        g0 = self.kwargs.get("width", 1.0)
        return [1j * (1 - m0**2 + m**2) / m0 / g0]


# ── Coupled-channel Flatté ───────────────────────────────────────

@register_model("FlatteC")
class FlatteCModel(BaseModel):
    """Flatté-like parametrisation with multiple decay channels."""

    def get_gamma_count(self):
        return len(self.kwargs["mass_list"])

    def get_gamma_name(self):
        return [f"{self.name}_g{i}" for i in range(self.get_gamma_count())]

    def get_gamma_defaults(self):
        return [float(self.kwargs.get(f"g_{i}", 0.1))
                for i in range(self.get_gamma_count())]

    def gamma(self, m):
        return [np.ones_like(m) + 0j] * self.get_gamma_count()


# ── Gounaris-Sakurai (rho-like) ──────────────────────────────────

@register_model("GS_rho")
class GSRhoModel(BaseModel):
    """Gounaris-Sakurai lineshape (placeholder)."""

    def get_gamma_defaults(self):
        return [float(self.kwargs.get("width", 0.1))]

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
    """Energy-dependent width loaded from a ``.npy`` file.

    The file must have columns ``[mass, Re(gamma), Im(gamma)]``.
    Values are linearly interpolated and normalised to 1 at the
    resonance pole mass.

    YAML example::

        particle:
          f2(1270):
            mass: 1.275
            width: 0.185
            model: width_linear_npy
            file: /path/to/width_table.npy
    """

    def get_gamma_defaults(self):
        return [float(self.kwargs.get("width", 0.1))]

    def gamma(self, m):
        data = np.load(self.kwargs["file"])
        mi = data[:, 0]
        fi = data[:, 1] + 1j * data[:, 2]
        y = np.interp(m, mi, fi)
        y0 = np.interp(self.kwargs["mass"], mi, fi)
        return [y / y0]
