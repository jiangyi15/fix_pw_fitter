"""
Fitted rho-omega product lineshape with a **running-width BWR rho**.

Goal amplitude (the ``RhoOmega`` fixed shape, GS replaced by a simple
Breit-Wigner with running width, all parameters fitted)::

    A(m) = 1 / (D_rho(m) . D_omega(m)),
    D_rho   = m_rho^2 - u - i . m_rho . Gamma0 . gamma(u),   u = m^2
    D_omega = m_omega^2 - u - i . m_omega . Gamma_omega

with ``gamma(u)`` the running-width shape (normalised at the config
reference mass, no Gamma0 factor) and all of
``m_rho, Gamma0, m_omega, Gamma_omega`` fitted.

Realised as **one** ordinary kernel Breit-Wigner whose kernel mass is
``m0 = m_rho`` and whose fixed 8-row gamma basis

    [1, u, i, i.u, i.u^2, gamma(u), i.gamma(u), u.gamma(u)]

is contracted with 8 real couplings ``{name}_w0 .. {name}_w7``::

    bw_dom = m0^2 - u - i.m0. sum_k w_k gamma_k(u) = D_rho . D_omega

Closed form (W = m_omega.Gamma_omega, B = m_rho^2 - u, a = m_omega^2 - u)::

    sum w_k gamma_k(u) = (B - D_rho.D_omega) / (i.m_rho)
        = W.B/m_rho + Gamma0.gamma.a - i.[ B.(1-a)/m_rho + Gamma0.W.gamma ]

    w0 = +W.m_rho                          (row 1)
    w1 = -W/m_rho                          (row u)
    w2 = -m_rho.(1 - m_omega^2)            (row i)
    w3 = -(m_rho^2 - 1 + m_omega^2)/m_rho  (row i.u)
    w4 = +1/m_rho                          (row i.u^2)
    w5 = +Gamma0.m_omega^2                 (row gamma)
    w6 = -Gamma0.W                         (row i.gamma)
    w7 = -Gamma0                           (row u.gamma)

(as in ``RhoOmega2``, the "1" is the 1 GeV^2 constant of the B.(1-a)
term in natural units).  The :class:`RhoOmegaBWRWeightsTransform` maps
the four physical parameters to the 8 couplings; the kernel mass
``{name}_mass`` is *not* produced by the transform — it stays the
fitted kernel m0, whose explicit dependence is captured by the
transform's backward pass.  ``{name}_width`` means ``Gamma(M)`` — the
width at the config reference mass (the standard BWR fit convention).
No kernel code is touched.

YAML usage::

    particle:
      rho_omega_bwr:
        mass: 0.775
        width: 0.149
        mass2: 0.78266
        width2: 0.00868
        model: RhoOmegaBWR
        float: [mass, width, mass2, width2]
"""

import numpy as np

from .base import BaseModel, register_model
from .gs_rho_model import _gamma_run, _two_body_cm_mom
from tabpwa.param_constraint import Transform


class RhoOmegaBWRWeightsTransform(Transform):
    """Transform: ``(m_rho, Gamma0, m_omega, Gamma_omega)`` -> 8 weights.

    Reads ``{p}_mass`` (kernel m0), ``{p}_width`` (Gamma0), ``{p}_mass2``
    and ``{p}_width2``; writes ``{p}_w0 .. {p}_w7`` (real).  ``{p}_mass``
    is *not* an output — it stays the fitted kernel m0, whose explicit
    dependence is captured by the transform's backward pass and
    accumulated with the kernel m0 gradient.

    ``forward`` uses the closed form in the module docstring; ``backward``
    is the exact analytic derivative of the weights w.r.t. the four inputs.
    """

    _has_inverse = True

    def __init__(self, mass_name, width_name, mass2_name, width2_name):
        p = mass_name[:-len("_mass")] if mass_name.endswith("_mass") else mass_name
        out_names = [f"{p}_w{k}" for k in range(8)]
        super().__init__(
            input_names=[mass_name, width_name, mass2_name, width2_name],
            output_names=out_names)
        self.mass_name = mass_name
        self.width_name = width_name
        self.mass2_name = mass2_name
        self.width2_name = width2_name

    # -- helpers ------------------------------------------------------
    @staticmethod
    def _weights(m0, g1, m2, g2):
        """The 8 real couplings from the four physical parameters."""
        W = m2 * g2
        return (W * m0,
                -W / m0,
                -m0 * (1.0 - m2 ** 2),
                -(m0 ** 2 - 1.0 + m2 ** 2) / m0,
                1.0 / m0,
                g1 * m2 ** 2,
                -g1 * W,
                -g1)

    def forward(self, d):
        m0 = float(d[self.mass_name])
        g1 = float(d[self.width_name])
        m2 = float(d[self.mass2_name])
        g2 = float(d[self.width2_name])
        return dict(zip(self.output_names, self._weights(m0, g1, m2, g2)))

    def backward(self, grad_out, d_in=None):
        if not d_in:
            return {n: 0.0 for n in self.input_names}
        m0 = float(d_in[self.mass_name])
        g1 = float(d_in[self.width_name])
        m2 = float(d_in[self.mass2_name])
        g2 = float(d_in[self.width2_name])
        W = m2 * g2

        w = self.output_names
        gr = [float(grad_out.get(w[k], 0.0)) for k in range(8)]

        # partials of each weight w.r.t. (m0, g1, m2, g2)
        dw0 = (W, 0.0, g2 * m0, m2 * m0)
        dw1 = (W / m0 ** 2, 0.0, -g2 / m0, -m2 / m0)
        dw2 = (-(1.0 - m2 ** 2), 0.0, 2.0 * m0 * m2, 0.0)
        dw3 = (-1.0 - (1.0 - m2 ** 2) / m0 ** 2, 0.0, -2.0 * m2 / m0, 0.0)
        dw4 = (-1.0 / m0 ** 2, 0.0, 0.0, 0.0)
        dw5 = (0.0, m2 ** 2, 2.0 * g1 * m2, 0.0)
        dw6 = (0.0, -W, -g1 * g2, -g1 * m2)
        dw7 = (0.0, -1.0, 0.0, 0.0)

        out = [0.0, 0.0, 0.0, 0.0]
        for k, dw in enumerate((dw0, dw1, dw2, dw3, dw4, dw5, dw6, dw7)):
            gk = gr[k]
            if gk == 0.0:
                continue
            out[0] += gk * dw[0]
            out[1] += gk * dw[1]
            out[2] += gk * dw[2]
            out[3] += gk * dw[3]
        return dict(zip(self.input_names, out))

    def inverse(self, d):
        """Identity on the inputs when present (``RhoOmega2`` style)."""
        names = self.input_names
        if all(n in d for n in names):
            return {n: float(d[n]) for n in names}
        return {}


@register_model("RhoOmegaBWR")
class RhoOmegaBWRModel(BaseModel):
    """Running-width BWR rho x constant-width BW omega, fitted.

    ``gamma(m)`` returns the fixed 8-row basis
    ``[1, u, i, i.u, i.u^2, gamma(u), i.gamma(u), u.gamma(u)]``
    (``u = m^2``); the 8 real couplings are produced by
    :class:`RhoOmegaBWRWeightsTransform` from the four physical
    parameters, and the kernel mass ``{name}_mass`` is the fitted
    rho mass.

    Parameters from YAML config:
        mass         -- rho mass (kernel m0), default 0.775
        width        -- rho width Gamma0 = Gamma(M), default 0.149
        mass2        -- omega mass, default 0.78266
        width2       -- omega width, default 0.00868
        L            -- orbital angular momentum (default 1)
        d            -- Blatt-Weisskopf radius (default 3.0)
        daug2Mass    -- first daughter mass (default pi+, 0.13957039)
        daug3Mass    -- second daughter mass (default pi0, 0.1349768)
    """

    _DEF_MASS = 0.775
    _DEF_WIDTH = 0.149
    _DEF_MASS2 = 0.78266
    _DEF_WIDTH2 = 0.00868
    _def_m2 = 0.13957039
    _def_m3 = 0.1349768

    def _mass2(self):
        return float(self.kwargs.get(
            "mass2", self.kwargs.get("omega_mass", self._DEF_MASS2)))

    def _width2(self):
        return float(self.kwargs.get(
            "width2", self.kwargs.get("omega_width", self._DEF_WIDTH2)))

    def get_gamma_count(self):
        return 8

    def get_gamma_name(self):
        return [f"{self.name}_w{k}" for k in range(8)]

    def gamma(self, m):
        """The fixed 8-row basis; ``gamma(u)`` from the running width."""
        m_arr = np.asarray(m, dtype=float)
        u = m_arr ** 2
        ones = np.ones_like(u, dtype=complex)

        # running-width shape — q0 at the config reference mass
        # (the same convention as BWR / GS_rho: the fitted width is
        # Gamma at that reference)
        M = float(self.kwargs.get("M", self.kwargs.get("mass",
                                                       self._DEF_MASS)))
        L = int(self.kwargs.get("L", 1))
        d = float(self.kwargs.get("d", 3.0))
        dm2 = float(self.kwargs.get("daug2Mass", self._def_m2))
        dm3 = float(self.kwargs.get("daug3Mass", self._def_m3))
        if self._parent and self._parent._decays:
            d1 = float(self._parent._decays[0].outs[0].mass)
            d2 = float(self._parent._decays[0].outs[1].mass)
            q = _two_body_cm_mom(m_arr, d1, d2)
            q0 = _two_body_cm_mom(M, d1, d2)
        else:
            q = _two_body_cm_mom(m_arr, dm2, dm3)
            q0 = _two_body_cm_mom(M, dm2, dm3)
        gam = np.asarray(_gamma_run(m_arr, 1.0, q, q0, L, M, d),
                         dtype=float).astype(complex)

        return [ones,
                u.astype(complex),
                1j * ones,
                1j * u,
                1j * (u ** 2),
                gam,
                1j * gam,
                u * gam]

    def get_defaults(self):
        return {
            f"{self.name}_mass": float(self.kwargs.get("mass", self._DEF_MASS)),
            f"{self.name}_width": float(self.kwargs.get("width", self._DEF_WIDTH)),
            f"{self.name}_mass2": self._mass2(),
            f"{self.name}_width2": self._width2(),
        }

    def make_mass_width_transform(self):
        return RhoOmegaBWRWeightsTransform(
            f"{self.name}_mass", f"{self.name}_width",
            f"{self.name}_mass2", f"{self.name}_width2")
