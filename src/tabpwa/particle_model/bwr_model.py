"""Relativistic Breit-Wigner with running width (BWR).

Based on the TFPWA reference implementation (``breit_wigner.py``).
The propagator denominator is::

    BW(m) = 1 / (m₀² - m² - i·m₀·Γ(m))

    Γ(m) = g₀ · (q/q₀)^{2L+1} · (m₀/m) · B'_L(q,q₀,d)²

where q is the breakup momentum and B'_L is the Blatt-Weisskopf
barrier factor for orbital angular momentum L.

YAML usage::

    particle:
      rho(770):
        mass: 0.775
        width: 0.149
        model: BWR
        L: 1
        daug2Mass: 0.13957039
        daug3Mass: 0.1349768
"""

import numpy as np
from ..param_constraint import Transform
from .base import BaseModel, register_model

# ── Helper functions (shared with GS_rho) ──────────────────────


def _two_body_cm_mom(m, m1, m2):
    """Real breakup momentum, 0 below threshold."""
    s = m ** 2
    m12s = (m1 + m2) ** 2
    m12d = (m1 - m2) ** 2
    p2 = (s - m12s) * (s - m12d)
    return np.sqrt(np.where(p2 > 0, p2, 0)) / (2 * m)


_BP_COEFF = {
    0: [1.0],
    1: [1.0, 1.0],
    2: [1.0, 3.0, 9.0],
    3: [1.0, 6.0, 45.0, 225.0],
    4: [1.0, 10.0, 135.0, 1575.0, 11025.0],
    5: [1.0, 15.0, 315.0, 6300.0, 99225.0, 893025.0],
}


def _bprime_poly(L, z):
    """Blatt-Weisskopf barrier polynomial at order *L*."""
    c = _BP_COEFF.get(L, [1.0])
    val = np.zeros_like(z)
    for ci in c:
        val = val * z + ci
    return val


def _bprime_poly_deriv(L, z):
    """d/dz of the Blatt-Weisskopf barrier polynomial at order *L*."""
    c = _BP_COEFF.get(L, [1.0])
    n = len(c)
    d = [c[i] * (n - 1 - i) for i in range(n - 1)]   # highest-first
    val = np.zeros_like(z)
    for ci in d:
        val = val * z + ci
    return val

def bwr_gamma_prime(m, model):
    """Analytic ``d gamma(m)/dm`` of the BWR running-width shape.

    Module-level function (not a particle-model method) — the transform
    layer is the only consumer.  ``model`` is a :class:`BWRModel`
    instance (or anything with the same ``kwargs`` / ``_parent``
    interface).  Returns 0 below threshold, where the shape is flat.

    With ``u = m^2``, ``q^2 = (u-a^2)(u-b^2)/(4u)``, ``n = 2L+1``::

        gamma  = (q/q0)^n * (M/m) * P(z0)/P(z),   z = (q d)^2
        d ln gamma/dm = n*q'/q - 1/m - (P'(z)/P(z)) * 2 q^2 d^2 * q'/q
        q'/q = (m^4 - a^2 b^2) / (m * p2),   p2 = (m^2-a^2)(m^2-b^2)
    """
    M  = float(model.kwargs.get("M",  model.kwargs.get("mass",  0.775)))
    L  = int(model.kwargs.get("L", 0))
    d  = float(model.kwargs.get("d", 3.0))
    if not (model._parent and model._parent._decays):
        raise ValueError(f"BWR model '{model.name}': no decay tree for q0")
    d1 = float(model._parent._decays[0].outs[0].mass)
    d2 = float(model._parent._decays[0].outs[1].mass)
    q0 = _two_body_cm_mom(M, d1, d2)
    a2 = (d1 + d2) ** 2
    b2 = (d1 - d2) ** 2

    m_arr = np.asarray(m, dtype=float)
    u = m_arr ** 2
    mm = np.sqrt(np.where(u > 0, u, 1.0))              # m
    q2 = (u - a2) * (u - b2) / (4.0 * np.where(u > 0, u, 1.0))
    above = q2 > 0
    q = np.where(above, np.sqrt(q2), 0.0)
    n = 2 * L + 1

    # q'/q = (m^4 - a^2 b^2) / (m * q^2 * 4 m^2) with q^2 = p2/(4m^2)
    #      = (m^4 - a^2 b^2) / (m * p2)
    qp_q = np.where(above, (u * u - a2 * b2)
                    / (mm * np.where(above, q2 * 4.0 * u, 1.0)), 0.0)

    z = (q * d) ** 2
    Pz = _bprime_poly(L, z)
    dPz = _bprime_poly_deriv(L, z)
    dz_dm = 2.0 * q * q * d * d * qp_q                 # dz/dm = 2 q d^2 q'

    gamma_m = np.where(above,
                       (q / np.where(q0 > 0, q0, 1.0)) ** n * (M / mm)
                       * _bprime_poly(L, (q0 * d) ** 2)
                       / np.where(Pz > 0, Pz, 1.0),
                       0.0)
    dlng = np.where(above,
                    n * qp_q - 1.0 / mm
                    - (dPz / np.where(Pz > 0, Pz, 1.0)) * dz_dm,
                    0.0)
    return gamma_m * dlng





# ── on-shell width parameter transform ──────────────────────────────

class WidthTransform(Transform):
    """Particle-special transform: the fitted width is the on-shell width.

    The kernel's running width is ``g0_kernel * gamma_table(m)``, with
    the gamma table built (and normalised) at the FIXED reference mass
    ``M`` of the particle entry.  A standard BWR fit therefore returns
    ``g0 = Gamma(M)`` — the width at the reference, not at the fitted
    pole.  With this transform the fitted ``{name}_width`` means
    ``Gamma(m0_fit)`` (the "normal BW" meaning) and the kernel-facing
    value is rescaled per iteration::

        g0_kernel = width_fit * gamma(M) / gamma(m0_fit)

    an exact reparametrisation of the standard BWR likelihood (the
    gamma shape is evaluated with the model's own ``gamma()``, the same
    function that built the table).

    ``backward`` propagates the width gradient back and adds the chain
    through the ratio's ``m0``-dependence (gamma' by centered finite
    differences, mirroring ``BWParamsTransform``'s style).

    This transform is what the ``BWRS`` model uses.
    """

    _has_inverse = True

    def __init__(self, model):
        self.model = model
        self.mass_name = f"{model.name}_mass"
        self.width_name = f"{model.name}_width"          # fit parameter
        self.out_name = f"{model.name}_width_t"          # kernel-facing
        import numpy as _np
        M = float(model.kwargs.get("M", model.kwargs.get("mass", 0.775)))
        self._gamma_M = float(_np.asarray(model.gamma(_np.array([M]))).reshape(-1)[0])
        super().__init__(input_names=[self.mass_name, self.width_name],
                         output_names=[self.out_name])

    def _ratio(self, m0):
        """gamma(M) / gamma(m0) at the resolved resonance mass."""
        import numpy as _np
        g_fit = float(_np.asarray(self.model.gamma(_np.array([float(m0)]))).reshape(-1)[0])
        if abs(g_fit) < 1e-300:
            raise ValueError(
                f"WidthTransform '{self.model.name}': gamma(m0) = 0 at "
                f"m0 = {m0!r} — width transform undefined")
        return self._gamma_M / g_fit

    def forward(self, d):
        m0 = d[self.mass_name]
        g = d[self.width_name]
        return {self.out_name: g * self._ratio(m0)}

    def inverse(self, d):
        """Kernel width -> fitted width: w_fit = w_kernel * gamma(m0)/gamma(M)."""
        import numpy as _np
        m0 = float(d[self.mass_name])
        w_kernel = float(d[self.out_name])
        g_fit = float(_np.asarray(
            self.model.gamma(_np.array([m0]))).reshape(-1)[0])
        if abs(g_fit) < 1e-300:
            raise ValueError(
                f"WidthTransform '{self.model.name}': gamma(m0) = 0 at "
                f"m0 = {m0!r}")
        return {self.mass_name: m0,
                self.width_name: w_kernel * g_fit / self._gamma_M}

    def backward(self, grad_out, d_in=None):
        d_in = d_in or {}
        m0 = float(d_in.get(self.mass_name, 0.0))
        g = float(d_in.get(self.width_name, 0.0))
        gw = grad_out.get(self.out_name, 0.0)
        if gw == 0.0:
            return {self.mass_name: 0.0, self.width_name: 0.0}
        import numpy as _np
        ratio = self._ratio(m0)                          # gamma(M)/gamma(m0)
        gprime = float(_np.asarray(
            bwr_gamma_prime(_np.array([m0]), self.model)).reshape(-1)[0])
        d_width = gw * ratio
        # d(ratio)/dm0 = -gamma_M * gamma'(m0) / gamma(m0)^2
        d_mass = -gw * g * ratio * ratio * gprime / self._gamma_M
        return {self.mass_name: d_mass, self.width_name: d_width}

    def to_dict(self):
        from tabpwa.particle_model.base import ALL_MODELS
        model_type = next((n for n, c in ALL_MODELS.items()
                           if isinstance(self.model, c)),
                          type(self.model).__name__)
        safe_kwargs = {k: v for k, v in self.model.kwargs.items()
                       if isinstance(v, (str, int, float, bool, list, dict))}
        return {"type": "WidthTransform",
                "particle_name": self.model.name,
                "model_type": model_type,
                "model_kwargs": safe_kwargs}

    @classmethod
    def from_dict(cls, d, **kwargs):
        from tabpwa.particle_model.base import build_particle
        model = kwargs.get("model")
        if model is None:
            model = build_particle(d["particle_name"], d["model_type"],
                                   **d.get("model_kwargs", {}))
        return cls(model)


# ── Model class ─────────────────────────────────────────────────


@register_model("BWR")
class BWRModel(BaseModel):
    """Relativistic Breit-Wigner with running width.

    The ``gamma(m)`` return value encodes the running width such that::

        bw_dom = m₀² - m² - i·m₀·g₀·gamma(m)
               = m₀² - m² - i·m₀·Γ(m)

    which matches the BWR denominator.

    Parameters from YAML config:
        mass          — pole mass m₀
        width         — nominal width g₀
        L             — orbital angular momentum (default 0)
        d             — impact parameter in GeV⁻¹ (default 3.0)
        daug2Mass     — first daughter mass (required if no decay tree)
        daug3Mass     — second daughter mass (required if no decay tree)
    """

    def get_defaults(self):
        return {f"{self.name}_mass": float(self.kwargs.get("mass", 0.775)),
                f"{self.name}_width": float(self.kwargs.get("width", 0.1))}



    def gamma(self, m):
        M  = float(self.kwargs.get("M",  self.kwargs.get("mass",  0.775)))
        L  = int(self.kwargs.get("L", 0))
        d  = float(self.kwargs.get("d", 3.0))

        # Breakup momenta — use decay tree masses (consistent with TFPWA)
        if self._parent and self._parent._decays:
            d1 = float(self._parent._decays[0].outs[0].mass)
            d2 = float(self._parent._decays[0].outs[1].mass)
            q  = _two_body_cm_mom(m, d1, d2)
            q0 = _two_body_cm_mom(M, d1, d2)
        else:
            raise ValueError(f"BWR model '{self.name}': no decay tree available for q0")

        # Running width factor: (q/q₀)^{2L+1} · (m₀/m) · B'_L(q,q₀,d)²
        qq0 = (q / q0) ** (2 * L + 1)
        mm0 = M / m
        bp2 = _bprime_poly(L, (q0 * d) ** 2) / _bprime_poly(L, (q * d) ** 2)

        gamma_m = qq0 * mm0 * bp2
        return [gamma_m]


# ── width-transform model variant ────────────────────────────────────

@register_model("BWRS")
class BWRSModel(BWRModel):
    """BWR with the fitted width meaning the on-shell width Gamma(m0):
    the running width evaluated at the resonance mass m0.

    Identical lineshape physics (:meth:`BWRModel.gamma`); the only
    difference is the parameter transform: the fitted ``{name}_width``
    is the width ``Gamma(m0_fit)`` (the running width evaluated at
    the resonance mass), and the
    kernel receives ``width * gamma(M) / gamma(m0_fit)`` — the same
    likelihood reparametrised so the fitted width is directly the
    physical width at m0 instead of the width at the fixed reference
    mass.  See :class:`WidthTransform`.
    """

    def get_gamma_name(self):
        # the kernel-facing width carries the transform suffix; the FIT
        # parameter stays ``{name}_width`` (the transform input)
        return [f"{self.name}_width_t"]

    def make_mass_width_transform(self):
        return WidthTransform(self)
