"""Extra amplitude models for this benchmark, registered on import.

`config_pwacg.yml` needs line shapes this framework does not ship, because the
reference's are not the textbook ones.  They live here, in the benchmark, rather
than in `src/tabpwa/particle_model/`: they are *their* prescriptions, and a
benchmark that patches the library to match its reference is no longer measuring
the library.  `bench.py`, `make_data.py`, `mem.py` and `check_model.py` import
this module, so `model: PWACG_flatte980` in a config just works.

Both are copied from the reference's GENERATED module
(`rendered_scripts/fit_object_kk.py`, functions `flatte980` / `flatte1270`), not
from templates and not from the literature.

`config_pwacg.yml` needs line shapes this framework does not ship, because the
reference's are not the textbook ones.  Both are copied from the GENERATED
module (`rendered_scripts/fit_object_kk.py`, functions `flatte980` /
`flatte1270`), not from the templates and not from the physics literature:

    # theirs, verbatim
    def flatte980(m_, g_pipi, rg, Sbc):
        g_kk = rg * g_pipi
        rho_kk   = sqrt(abs(1 - 4*m_k*m_k / Sbc))
        rho_pipi = sqrt(abs(1 - 4*m_pi*m_pi / Sbc))
        return 1 / ((m_**2 - Sbc) - i*(g_pipi*rho_pipi + g_kk*rho_kk))

    def flatte1270(m_, w_, Sbc):
        rm, gr = m_*m_, m_*w_
        q2r = 0.25*rm - 0.0194792;  b2r = q2r*(q2r + 0.1825) + 0.033306
        g11270 = gr * b2r / power(q2r, 2.5)
        q2 = 0.25*Sbc - 0.0194792;  b2 = q2*(q2 + 0.1825) + 0.033306
        g1 = g11270 * power(q2, 2.5) / b2
        return gr / ((Sbc - rm) + i*g1)

Two things make these "not textbook": ``rho = sqrt(abs(...))`` is REAL on both
sides of threshold (a physical Flatté takes the complex breakup momentum, i.e.
imaginary below), and ``flatte1270`` is not a coupled-channel Flatté at all — it
is a running-width BW whose width carries their ``b2`` polynomial.

How they map onto this framework's denominator
``m0² − m² − i·m0·g0·γ(m)`` is derived, not guessed, and ``check_model.py``
verifies the result numerically against the formula above:

* ``flatte980`` — two channels, ``γ_i = ρ_i`` and couplings
  ``g_i = g_i^their / m0`` (our kernel multiplies by ``m0``);
* ``flatte1270`` — one channel, ``γ = (b2r/q2r^2.5)·q2^2.5/b2``, which is their
  ``g1/(m·w)``: their overall numerator ``m·w`` is a constant per chain and is
  absorbed into the (free) wave coupling.

Add a model here the same way: subclass `BaseModel`, decorate with
`register_model`, implement `gamma(m)`; nothing else changes.
"""
import numpy as np

from tabpwa.particle_model.base import BaseModel, register_model

#: their hard-coded daughter masses (GeV)
M_K = 0.493677
M_PI = 0.13957061
#: their `q2 = 0.25*Sbc - 0.0194792`, i.e. m_pi^2
M_PI2 = 0.0194792


@register_model("PWACG_flatte980")
class PwacgFlatte980(BaseModel):
    """Their `flatte980`: two real-ρ channels (ππ, K-K).

    ``γ(m) = [ √|1 − 4m_π²/m²| , √|1 − 4m_K²/m²| ]``

    The kernel then computes ``m0² − m² − i·m0·(g_0·γ_0 + g_1·γ_1)``, so the
    couplings that reproduce ``g_ππ/g_KK`` are ``g_i = g_i^their / m0``; the
    config carries those values as defaults.
    """

    def get_defaults(self):
        m0 = float(self.kwargs.get("mass", 0.9794812))
        return {f"{self.name}_mass": m0,
                f"{self.name}_g0": float(self.kwargs.get("g_0", 0.0127)),
                f"{self.name}_g1": float(self.kwargs.get("g_1", 0.1090))}

    def get_gamma_count(self):
        return 2

    def get_gamma_name(self):
        return [f"{self.name}_g0", f"{self.name}_g1"]

    def gamma(self, m):
        s = np.asarray(m, np.float64) ** 2
        return [np.sqrt(np.abs(1.0 - 4.0 * M_PI * M_PI / s)),
                np.sqrt(np.abs(1.0 - 4.0 * M_K * M_K / s))]


@register_model("PWACG_flatte1270")
class PwacgFlatte1270(BaseModel):
    """Their `flatte1270`: a running-width BW with their `b2` polynomial.

    ``γ(m) = (b2(m0)/q2(m0)^{2.5}) · q2(m)^{2.5} / b2(m)``, ``q2 = m²/4 − m_π²``,
    ``b2 = q2(q2 + 0.1825) + 0.033306`` — i.e. their ``g1/(m·w)``, with the
    constant numerator ``m·w`` left to the wave coupling.
    """

    def get_defaults(self):
        return {f"{self.name}_mass": float(self.kwargs.get("mass", 1.302)),
                f"{self.name}_width": float(self.kwargs.get("width", 0.189))}

    def gamma(self, m):
        m0 = float(self.kwargs.get("mass", 1.302))
        q2r = 0.25 * m0 * m0 - M_PI2
        b2r = q2r * (q2r + 0.1825) + 0.033306
        q2 = 0.25 * np.asarray(m, np.float64) ** 2 - M_PI2
        b2 = q2 * (q2 + 0.1825) + 0.033306
        with np.errstate(invalid="ignore", divide="ignore"):
            g = (b2r / q2r ** 2.5) * q2 ** 2.5 / b2
        # NOT verbatim here: below the ππ threshold q2 < 0 and a real q2**2.5 is
        # undefined, so their expression yields nan.  We map it to 0 — the
        # denominator then stays finite — because a nan amplitude would poison the
        # whole sample's NLL.  Above threshold (where the fit lives) the two agree
        # exactly; this branch is the one deliberate deviation from their code.
        return [np.nan_to_num(g, nan=0.0)]
