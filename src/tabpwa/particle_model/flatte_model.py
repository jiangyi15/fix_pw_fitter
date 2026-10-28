"""Flatté parametrisations for coupled-channel resonances.

Based on the TFPWA reference implementation (``amp/flatte.py``).
``FlatteC`` is the classic multi-channel Flatté; ``FlatteGen`` adds the
per-channel orbital angular momentum / Blatt-Weisskopf normalisation of
TFPWA's ``FlatteGen``.
The propagator denominator is::

    R(m) = 1 / (m₀² - m² - i·m₀·Σᵢ gᵢ · qᵢ(m) / m)

where qᵢ(m) is the complex breakup momentum for channel i::

    qᵢ =  √((m²-(mᵢ₁+mᵢ₂)²)(m²-(mᵢ₁-mᵢ₂)²)) / (2m)    (above threshold)
    qᵢ =  i·√(|(m²-(mᵢ₁+mᵢ₂)²)(m²-(mᵢ₁-mᵢ₂)²)|) / (2m)  (below threshold)

YAML usage::

    particle:
      f0(980):
        mass: 0.99
        model: FlatteC
        mass_list: [[0.13957, 0.13957], [0.49368, 0.49368]]
        g_0: 0.3
        g_1: 0.6
"""

import numpy as np

from ..bw_form_factor import barrier_ratio
from .base import BaseModel, register_model


def _breakup_momentum(m, m1, m2):
    """Complex breakup momentum for a two-body decay at mass *m*.

    Returns real above threshold, purely imaginary below threshold.
    """
    s = m ** 2
    mabp = m1 + m2
    mabm = m1 - m2
    p2 = (s - mabp * mabp) * (s - mabm * mabm) / (4.0 * s)
    return np.emath.sqrt(p2)


@register_model("FlatteC")
class FlatteCModel(BaseModel):
    """Flatté parametrisation with multiple coupled channels.

    The ``gamma(m)`` returns a list of ``qᵢ(m)/m`` (one per channel).
    The framework computes::

        g_bw = Σᵢ g₀ᵢ · gammaᵢ(m)
        bw_dom = m₀² - m² - i·m₀·g_bw

    which matches the Flatté denominator.

    Required YAML arg:
        mass_list: [[m11, m12], [m21, m22], ...]
            List of daughter-mass pairs, one per channel.

    Optional YAML args:
        g_i:  coupling constant for channel i (default 0.1)
    """

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
        mass_list = self.kwargs["mass_list"]
        out = []
        for ch in mass_list:
            m1, m2 = ch
            q = _breakup_momentum(m, m1, m2)
            out.append(q / m)
        return out


@register_model("FlatteGen")
class FlatteGenModel(BaseModel):
    r"""Generalised Flatté parametrisation (TFPWA ``ParticleFlateGen``).

    The framework computes ::

        g_bw = Σᵢ g₀ᵢ · gammaᵢ(m)
        bw_dom = m₀² - m² - i·m₀·g_bw

    so a channel's ``gamma`` row is (TFPWA ``amp/flatte.py``,
    ``ParticleFlateGen.get_amp``, divided by the kernel's outer ``m₀``)::

        gammaᵢ(m) = -im_sign · (qᵢ/m) · (m₀_ref/|qᵢ₀|)
                    · (|qᵢ|/|qᵢ₀|)^(2·lᵢ) · B'_{lᵢ}(|qᵢ|, |qᵢ₀|, d)²

    with the complex breakup momentum

        qᵢ = √((m²-(mᵢ₁+mᵢ₂)²)(m²-(mᵢ₁-mᵢ₂)²))/(2m)   (above threshold)
        qᵢ = i·√(|…|)/(2m)                              (below threshold)

    and ``qᵢ₀`` the same quantity evaluated at the **reference** mass
    ``m₀_ref`` (the config's ``mass:``, default 0.775).  ``FlatteGen``
    uses ``im_sign=-1`` — the same sign convention as ``FlatteC``.

    YAML usage::

        particle:
          a0(980):
            mass: 0.99
            model: FlatteGen
            mass_list: [[0.13957, 0.13957], [0.13957, 0.54786]]
            l_list: [0, 1]
            g_0: 0.3
            g_1: 0.6
            has_bprime: true       # optional (default true)
            d: 3.0                 # optional Blatt-Weisskopf radius

    Options (all optional; defaults in brackets):

    * ``l_list`` — orbital angular momentum per channel ``[0, 0, …]``;
      its length must equal ``len(mass_list)``.  Its ``2·l`` power of
      ``|qᵢ|/|qᵢ₀|`` makes the channel share the decay's angular
      dependence.
    * ``has_bprime`` — ``[True]`` include ``B'_{lᵢ}²``; ``False`` drops
      the barrier factor entirely.
    * ``cut_phsp`` — ``[False]`` set the channel term to zero below its
      threshold ``m < mᵢ₁+mᵢ₂`` (instead of the complex ``q`` branch).
    * ``no_q0`` — ``[False]`` drop the ``m₀_ref/|qᵢ₀|`` normalisation and
      use ``|qᵢ₀| = 1`` in the ``2·l`` power and the barrier.
    * ``no_m0`` — ``[False]`` replace the ``m₀_ref`` numerator of the
      ``qᵢ/m`` factor by 1 (TFPWA's ``m_c = 1/m``): the running-width
      term loses one power of ``m₀`` and the kernel's outer ``m₀`` no
      longer cancels.

    **Reference-mass caveat.**  In tabpwa the gamma rows are tabulated
    *once* at the config reference mass (``BaseModel.build_gamma_table``)
    and the kernel applies the *fitted* ``m₀`` only in the outer
    ``m₀² - m² - i·m₀·Σ…``.  TFPWA instead re-evaluates ``qᵢ₀`` and
    ``B'(…)`` at the current ``m₀`` on every call.  The two agree
    exactly when the fitted ``m₀`` equals the reference mass and drift
    apart as it is fitted away (the ``|qᵢ₀|`` / barrier normalisation
    stays pinned to the reference).  Fix ``{name}_mass`` — or accept the
    small difference — when a strict TFPWA match is required.

    ``Flatte2`` (TFPWA: ``gᵢ²`` in place of ``gᵢ``) is intentionally NOT
    provided: the kernel denominator is linear in the couplings
    (``Σ gᵢ·gammaᵢ``), so a squared coupling cannot be expressed through
    the shared gamma table — it would need a kernel change.
    """

    def __init__(self, name, im_sign=-1, l_list=None, has_bprime=True,
                 no_m0=False, no_q0=False, cut_phsp=False, **kwargs):
        super().__init__(name, **kwargs)
        mass_list = kwargs.get("mass_list")
        if not mass_list:
            raise ValueError(
                f"particle model '{name}' (FlatteGen) requires "
                f"mass_list: [[m11, m12], [m21, m22], ...] (one daughter "
                f"pair per coupled channel)")
        self.mass_list = mass_list
        if l_list is None:
            l_list = [0] * len(mass_list)
        l_list = list(l_list)
        if len(l_list) != len(mass_list):
            raise ValueError(
                f"particle model '{name}' (FlatteGen): l_list has "
                f"{len(l_list)} entries but mass_list has "
                f"{len(mass_list)} channel(s) — they must match")
        self.l_list = [int(l) for l in l_list]
        self.im_sign = im_sign
        self.has_bprime = bool(has_bprime)
        self.no_m0 = bool(no_m0)
        self.no_q0 = bool(no_q0)
        self.cut_phsp = bool(cut_phsp)
        # The options arrive as named arguments (before ``**kwargs`` reaches
        # BaseModel), so mirror the effective values into ``self.kwargs``:
        # BWParamsTransform.to_dict() serialises ``model.kwargs`` and
        # from_dict() rebuilds through build_particle(), which must see the
        # same l_list / flags (a lost l_list would silently change physics).
        self.kwargs.setdefault("im_sign", self.im_sign)
        self.kwargs.setdefault("l_list", list(self.l_list))
        for _key in ("has_bprime", "no_m0", "no_q0", "cut_phsp"):
            self.kwargs.setdefault(_key, getattr(self, _key))

    def get_defaults(self):
        mass = float(self.kwargs.get("mass", 0.775))
        gammas = {f"{self.name}_g{i}": float(self.kwargs.get(f"g_{i}", 0.1))
                  for i in range(self.get_gamma_count())}
        return {f"{self.name}_mass": mass, **gammas}

    def get_gamma_count(self):
        return len(self.mass_list)

    def get_gamma_name(self):
        return [f"{self.name}_g{i}" for i in range(self.get_gamma_count())]

    def gamma(self, m):
        m_arr = np.asarray(m, dtype=float)
        m0 = float(self.kwargs.get("mass", 0.775))
        d = float(self.kwargs.get("d", 3.0))
        # The kernel puts one m0 in front of Σ g·gamma; TFPWA's ``m_c``
        # carries another (m0, or 1 with no_m0) — divide it out here.
        m_scale = 1.0 / m0 if self.no_m0 else 1.0
        out = []
        for (m1, m2), l in zip(self.mass_list, self.l_list):
            q = _breakup_momentum(m_arr, m1, m2)
            q0 = _breakup_momentum(m0, m1, m2)
            aq = np.abs(q)
            if self.no_q0:
                aq0 = 1.0
                q0_scale = 1.0
            else:
                aq0 = np.abs(q0)
                q0_scale = m0 / aq0
            term = (q / m_arr) * m_scale * q0_scale
            if l != 0:
                term = term * (aq / aq0) ** (2 * l)
            if self.has_bprime:
                term = term * barrier_ratio(l, aq, aq0, d) ** 2
            if self.cut_phsp:
                term = np.where(m_arr < m1 + m2, 0.0 + 0.0j, term)
            out.append(-self.im_sign * term)
        return out

