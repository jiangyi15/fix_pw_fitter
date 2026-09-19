"""
Two-pole rho-omega product lineshape with **optimizable** masses/widths.

Goal amplitude (constant-width product)::

    A(m) = 1 / (D_rho(m) . D_omega(m)),
    D_p(u) = m_p^2 - u - i . m_p . Gamma_p,   u = m^2

with all of ``m_rho, Gamma_rho, m_omega, Gamma_omega`` fitted.

This is realised as **one** ordinary kernel Breit-Wigner whose kernel mass
is ``m0 = m_rho`` and whose running width is a fixed 6-row gamma basis

    gamma(u) = [1, i, u, i.u, u^2, i.u^2]

contracted with 6 real couplings ``{name}_w0 .. {name}_w5``::

    g_bw(u) = c0 + c1.u + c2.u^2
    c0 = w0 + i.w1,  c1 = w2 + i.w3,  c2 = w4 + i.w5

Matching ``m0^2 - u - i.m0.g_bw(u) = D_rho.D_omega`` with
``a_p = m_p^2 - i.m_p.Gamma_p`` (so ``D_rho.D_omega = u^2 - (a_rho+a_omega)u
+ a_rho.a_omega``) gives::

    c2 =  i / m0
    c1 = -i.(a_rho + a_omega - 1) / m0
    c0 = -i.(m0^2 - a_rho.a_omega) / m0

The :class:`RhoOmegaWeightsTransform` maps the four physical parameters
(``mass, width, mass2, width2``) to the six real couplings; the kernel mass
``{name}_mass`` is *not* produced by the transform — it stays the fitted
kernel m0.  No kernel code is touched.

YAML usage::

    particle:
      rho_omega2:
        mass: 0.775
        width: 0.149
        mass2: 0.78266
        width2: 0.00868
        model: RhoOmega2
        float: [mass, width, mass2, width2]
"""

import numpy as np

from .base import BaseModel, register_model
from ampfit.param_constraint import Transform


class RhoOmegaWeightsTransform(Transform):
    """Transform: ``(m_rho, Gamma_rho, m_omega, Gamma_omega)`` -> 6 weights.

    Reads ``{p}_mass`` (kernel m0), ``{p}_width``, ``{p}_mass2`` and
    ``{p}_width2``; writes ``{p}_w0 .. {p}_w5`` (real).  ``{p}_mass`` is
    *not* an output — it stays the fitted kernel m0, whose explicit
    dependence is captured by the transform's backward pass and
    accumulated with the kernel m0 gradient.

    ``forward`` uses the closed form in the module docstring; ``backward``
    is the exact analytic derivative of the weights w.r.t. the four inputs.
    """

    _has_inverse = True

    def __init__(self, mass_name, width_name, mass2_name, width2_name):
        p = mass_name[:-len("_mass")] if mass_name.endswith("_mass") else mass_name
        out_names = [f"{p}_w{k}" for k in range(6)]
        super().__init__(
            input_names=[mass_name, width_name, mass2_name, width2_name],
            output_names=out_names)
        self.mass_name = mass_name
        self.width_name = width_name
        self.mass2_name = mass2_name
        self.width2_name = width2_name

    # -- helpers ------------------------------------------------------
    @staticmethod
    def _ab(m0, g1, m2, g2):
        """Complex pole factors ``a_p = m_p^2 - i.m_p.Gamma_p``."""
        return (m0 ** 2 - 1j * m0 * g1, m2 ** 2 - 1j * m2 * g2)

    @staticmethod
    def _coeffs(m0, a1, a2):
        """Polynomial coefficients ``(c0, c1, c2)`` of ``g_bw(u)``."""
        c2 = 1j / m0
        c1 = -1j * (a1 + a2 - 1.0) / m0
        c0 = -1j * (m0 ** 2 - a1 * a2) / m0
        return c0, c1, c2

    def forward(self, d):
        m0 = float(d[self.mass_name])
        g1 = float(d[self.width_name])
        m2 = float(d[self.mass2_name])
        g2 = float(d[self.width2_name])
        a1, a2 = self._ab(m0, g1, m2, g2)
        c0, c1, c2 = self._coeffs(m0, a1, a2)
        out = self.output_names
        return {out[0]: float(c0.real), out[1]: float(c0.imag),
                out[2]: float(c1.real), out[3]: float(c1.imag),
                out[4]: float(c2.real), out[5]: float(c2.imag)}

    def backward(self, grad_out, d_in=None):
        if not d_in:
            return {n: 0.0 for n in self.input_names}
        m0 = float(d_in[self.mass_name])
        g1 = float(d_in[self.width_name])
        m2 = float(d_in[self.mass2_name])
        g2 = float(d_in[self.width2_name])

        w = self.output_names
        gr = [float(grad_out.get(w[k], 0.0)) for k in range(6)]

        a1, a2 = self._ab(m0, g1, m2, g2)

        # partials of a_p w.r.t. the four inputs
        da1_dm0 = 2.0 * m0 - 1j * g1
        da1_dg1 = -1j * m0
        da2_dm2 = 2.0 * m2 - 1j * g2
        da2_dg2 = -1j * m2

        # c2 = i/m0
        dc2_dm0 = -1j / m0 ** 2

        # c1 = -i.(a1+a2-1)/m0
        dc1_dm0 = 1j * (a1 + a2 - 1.0) / m0 ** 2 + (-1j / m0) * da1_dm0
        dc1_dg1 = (-1j / m0) * da1_dg1
        dc1_dm2 = (-1j / m0) * da2_dm2
        dc1_dg2 = (-1j / m0) * da2_dg2

        # c0 = -i.m0 + i.a1.a2/m0
        dc0_dm0 = (-1j - 1j * a1 * a2 / m0 ** 2
                   + (1j * a2 / m0) * da1_dm0)
        dc0_dg1 = (1j * a2 / m0) * da1_dg1
        dc0_dm2 = (1j * a1 / m0) * da2_dm2
        dc0_dg2 = (1j * a1 / m0) * da2_dg2

        def combine(dc0, dc1, dc2):
            return (gr[0] * dc0.real + gr[1] * dc0.imag
                    + gr[2] * dc1.real + gr[3] * dc1.imag
                    + gr[4] * dc2.real + gr[5] * dc2.imag)

        zero = 0.0 + 0.0j
        return {
            self.mass_name: combine(dc0_dm0, dc1_dm0, dc2_dm0),
            self.width_name: combine(dc0_dg1, dc1_dg1, zero),
            self.mass2_name: combine(dc0_dm2, dc1_dm2, zero),
            self.width2_name: combine(dc0_dg2, dc1_dg2, zero),
        }

    def inverse(self, d):
        """Identity on the inputs when present (``interp_k_model`` style).

        The four physical inputs are pass-through keys in the resolved dict
        (``apply_forward`` preserves non-output keys), so the common case is
        simply returning them.  Without them there is no reliable way to
        recover the masses/widths from the weights alone — return an empty
        dict (callers then keep their existing values).
        """
        names = self.input_names
        if all(n in d for n in names):
            return {n: float(d[n]) for n in names}
        return {}


@register_model("RhoOmega2")
class RhoOmega2Model(BaseModel):
    """Two-pole rho-omega product lineshape with fitted masses/widths.

    ``gamma(m)`` returns the fixed 6-row polynomial basis
    ``[1, i, u, i.u, u^2, i.u^2]`` (``u = m^2``); the six real couplings
    are produced by :class:`RhoOmegaWeightsTransform` from the four
    physical parameters, and the kernel mass ``{name}_mass`` is the fitted
    rho mass.

    Parameters from YAML config:
        mass         -- rho mass (kernel m0), default 0.775
        width        -- rho width, default 0.149
        mass2        -- omega mass, default 0.78266
        width2       -- omega width, default 0.00868
    """

    _DEF_MASS = 0.775
    _DEF_WIDTH = 0.149
    _DEF_MASS2 = 0.78266
    _DEF_WIDTH2 = 0.00868

    def _mass2(self):
        return float(self.kwargs.get(
            "mass2", self.kwargs.get("omega_mass", self._DEF_MASS2)))

    def _width2(self):
        return float(self.kwargs.get(
            "width2", self.kwargs.get("omega_width", self._DEF_WIDTH2)))

    def get_gamma_count(self):
        return 6

    def get_gamma_name(self):
        return [f"{self.name}_w{k}" for k in range(6)]

    def gamma(self, m):
        """Fixed polynomial basis ``[1, i, u, i.u, u^2, i.u^2]``."""
        u = np.asarray(m, dtype=float) ** 2
        ones = np.ones_like(u, dtype=complex)
        return [ones,
                1j * ones,
                u.astype(complex),
                1j * u,
                (u ** 2).astype(complex),
                1j * (u ** 2)]

    def get_defaults(self):
        return {
            f"{self.name}_mass": float(self.kwargs.get("mass", self._DEF_MASS)),
            f"{self.name}_width": float(self.kwargs.get("width", self._DEF_WIDTH)),
            f"{self.name}_mass2": self._mass2(),
            f"{self.name}_width2": self._width2(),
        }

    def make_mass_width_transform(self):
        return RhoOmegaWeightsTransform(
            f"{self.name}_mass", f"{self.name}_width",
            f"{self.name}_mass2", f"{self.name}_width2")
