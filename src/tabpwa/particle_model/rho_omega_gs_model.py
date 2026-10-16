"""
Fitted rho-omega product lineshape with a **Gounaris-Sakurai rho**.

The rho propagator uses the single complex GS unit (scale related to
``(q0, m0_ref)`` of the config reference)::

    D_rho   = m0^2 - u - i . m0 . Gamma0 . g(u),     u = m^2
    g(u)    = gamma(u) + i . fs(u) / m0_ref
    D_omega = m_omega^2 - u - i . m_omega . Gamma_omega

with ``gamma`` the running-width shape (``(q/q0)^(2L+1) . m0_ref/m . B'^2``)
and ``fs`` the GS mass-shift term — both computed at the config reference
(the fitted ``width`` is Gamma0 = Gamma at that reference) — and all of
``m_rho, Gamma0, m_omega, Gamma_omega`` fitted.  The whole complex unit
``g`` carries ONE fitted width: the fs and gamma pieces scale together
with Gamma0 (their reference factors are the constants
``m0_ref/m`` and ``M_ref^2/k0^3``; the ``1/m`` is loop structure, not a
reference factor).

The product is realised with the fixed 8-row gamma basis

    [1, u, i, i.u, i.u^2, g, u.g, i.g]

contracted with 8 real couplings ``{name}_w0 .. {name}_w7``::

    bw_dom = m0^2 - u - i.m0. sum_k w_k gamma_k(u) = D_rho . D_omega

Closed form (W = m_omega.Gamma_omega, B = m0^2 - u, a = m_omega^2 - u)::

    D_rho.D_omega = A.C - m0.Gamma0.g.(W + i.C),   A.C = B + B.(a-1)
    sum_k w_k gamma_k = B.(W + i.(a-1))/m0 + Gamma0.(C - i.W).g

row assignments::

    w0 = +W.m0                       (1)
    w1 = -W/m0                       (u)
    w2 = -m0.(1 - m2^2)              (i)
    w3 = -(m0^2 - 1 + m2^2)/m0       (i.u)
    w4 = +1/m0                       (i.u^2)
    w5 = +Gamma0.m2^2                (g)
    w6 = -Gamma0                     (u.g)
    w7 = -Gamma0.W                   (i.g)

with ``fs == 0`` (``g == gamma``) these are exactly the
``RhoOmegaBWR`` couplings.  The :class:`RhoOmegaGSWeightsTransform` maps
the four physical parameters to the 8 couplings; the kernel mass
``{name}_mass`` is *not* produced by the transform — it stays the fitted
kernel m0, whose explicit dependence is captured by the transform's
backward pass.  No kernel code is touched.

YAML usage::

    particle:
      rho_omega_gs:
        mass: 0.775
        width: 0.149
        mass2: 0.78266
        width2: 0.00868
        model: RhoOmegaGS
        float: [mass, width, mass2, width2]
"""

import numpy as np

from .base import BaseModel, register_model
from .gs_rho_model import _gamma_run, _two_body_cm_mom, _fs_fun
from tabpwa.param_constraint import Transform


class RhoOmegaGSWeightsTransform(Transform):
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
        return (W * m0,                          # 1
                -W / m0,                         # u
                -m0 * (1.0 - m2 ** 2),           # i
                -(m0 ** 2 - 1.0 + m2 ** 2) / m0,  # i.u
                1.0 / m0,                        # i.u^2
                g1 * m2 ** 2,                    # g
                -g1,                             # u.g
                -g1 * W)                         # i.g

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
        dw6 = (0.0, -1.0, 0.0, 0.0)
        dw7 = (0.0, -W, -g1 * g2, -g1 * m2)

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


@register_model("RhoOmegaGS")
class RhoOmegaGSModel(BaseModel):
    """GS rho x constant-width BW omega, fitted.

    ``gamma(m)`` returns the fixed 8-row basis
    ``[1, u, i, i.u, i.u^2, g, u.g, i.g]`` (``u = m^2``) with the single
    complex GS unit ``g = gamma + i.fs/m0_ref`` (the ``(q0, m0_ref)``
    scale of the config reference); the 8 real couplings are produced by
    :class:`RhoOmegaGSWeightsTransform` from the four physical
    parameters, and the kernel mass ``{name}_mass`` is the fitted rho
    mass.  At the reference point (``m0 = m0_ref``) the lineshape is
    exactly the fixed ``RhoOmega`` shape.

    Parameters from YAML config:
        mass         -- rho mass (kernel m0), default 0.775
        width        -- rho width Gamma0 = Gamma(M_ref), default 0.149
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
        """The fixed 8-row basis; ``g`` at the config reference mass."""
        m_arr = np.asarray(m, dtype=float)
        u = m_arr ** 2
        ones = np.ones_like(u, dtype=complex)

        # shape functions at the config reference mass (fixed basis —
        # the same convention as RhoOmegaBWR / GS_rho: the fitted width
        # is Gamma at that reference)
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
                         dtype=float)
        fs = np.asarray(_fs_fun(u, M ** 2, 1.0, dm2, dm3), dtype=float)
        g = gam + 1j * fs / M                      # the complex GS unit

        return [ones,
                u.astype(complex),
                1j * ones,
                1j * u,
                1j * (u ** 2),
                g,
                u * g,
                1j * g]

    def get_defaults(self):
        return {
            f"{self.name}_mass": float(self.kwargs.get("mass", self._DEF_MASS)),
            f"{self.name}_width": float(self.kwargs.get("width", self._DEF_WIDTH)),
            f"{self.name}_mass2": self._mass2(),
            f"{self.name}_width2": self._width2(),
        }

    def make_mass_width_transform(self):
        return RhoOmegaGSWeightsTransform(
            f"{self.name}_mass", f"{self.name}_width",
            f"{self.name}_mass2", f"{self.name}_width2")

    def rho_pole(self, mass=None, width=None):
        """Pole parameters ``(m_pole, Gamma_pole)`` of the *frozen-reference*
        GS rho propagator.

        Reads the fit parameters (defaults: the config ``mass``/``width``)
        and solves ``Re D_rho(m) = 0`` — the mass at which the propagator
        is purely imaginary — by fixed-point iteration on
        ``u = m0^2 + Gamma0*fs(u)/m0_ref`` (the shapes are the frozen
        reference ones, exactly what the kernel sees).  Returns the mass
        and the width at that point::

            D_rho(m_pole) = -i * m_pole * Gamma_pole

        Note this is the pole of the parameterised ``D_rho`` (shapes at
        the reference), not of a fully rebuilt GS.  (The omega pole is
        trivially ``(mass2, width2)`` — constant width.)
        """
        m0 = float(self.kwargs.get("mass", self._DEF_MASS)) if mass is None \
            else float(mass)
        g0 = float(self.kwargs.get("width", self._DEF_WIDTH)) if width is None \
            else float(width)
        M = float(self.kwargs.get("M", self.kwargs.get("mass", self._DEF_MASS)))
        L = int(self.kwargs.get("L", 1))
        d = float(self.kwargs.get("d", 3.0))
        dm2 = float(self.kwargs.get("daug2Mass", self._def_m2))
        dm3 = float(self.kwargs.get("daug3Mass", self._def_m3))
        if self._parent and self._parent._decays:
            d1 = float(self._parent._decays[0].outs[0].mass)
            d2 = float(self._parent._decays[0].outs[1].mass)
            q_of = lambda mm: _two_body_cm_mom(mm, d1, d2)
        else:
            q_of = lambda mm: _two_body_cm_mom(mm, dm2, dm3)
        q0 = q_of(M)
        gam = lambda mm: float(_gamma_run(mm, 1.0, q_of(mm), q0, L, M, d))
        fs = lambda mm: float(_fs_fun(mm ** 2, M ** 2, 1.0, dm2, dm3))

        # fixed point: Re D_rho = 0  <=>  u = m0^2 + (m0/m0_ref).g0*fs(u)
        u = m0 * m0
        for _ in range(500):
            u_new = m0 * m0 + (m0 / M) * g0 * fs(u)
            if abs(u_new - u) < 1e-14 * (1.0 + abs(u)):
                u = u_new
                break
            u = u_new
        else:
            raise ValueError("rho_pole: fixed point did not converge "
                             f"(m0={m0}, Gamma0={g0})")
        m_pole = float(np.sqrt(u))
        gamma_pole = m0 * g0 * gam(m_pole) / m_pole
        return m_pole, gamma_pole
