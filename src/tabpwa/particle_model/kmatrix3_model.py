"""Three-pole, one-channel K-matrix (``kmatrix3``).

The same construction as :mod:`tabpwa.particle_model.kmatrix2_model`, with one
more pole in the common denominator::

    K(s) = Σ_{i=1..3} g_i² / (m_i² − s)
    P(s) = Σ_{i=1..3} b_i  / (m_i² − s)
    A(s) = P(s) / (1 − i·ρ(s)·K(s))

Over the common denominator ``Π_i (m_i² − s)`` (cubic in ``s``)::

    den_basis = [ 1, s, s², s³,  −i·ρ, −i·s·ρ, −i·s²·ρ ]
    num_basis = [ 1, s, s² ]

    real rows   weight:  coefficients of  Π_i (m_i² − s)
    ρ rows      weight:  coefficients of  Σ_i g_i² Π_{j≠i} (m_j² − s)
    numerator   weight:  coefficients of  Σ_i b_i  Π_{j≠i} (m_j² − s)   (complex)

so the weights are the elementary symmetric polynomials of the pole masses
squared, and ``D = Π(m_i²−s) − iρ·Σ g_i² Π_{j≠i}(m_j²−s)`` exactly.  The
two-pole model is the ``g3 = b3 = 0`` special case (the ``(m3² − s)`` factor
cancels) and one pole is the plain Breit-Wigner — both are gated.

``ρ(m)`` is :func:`~tabpwa.particle_model.bwr_model.phase_space` —
``(q/m)·q^{2L}·B'²``, M-FREE — so the one-pole limit is ``BWRS`` exactly with
``bf² = m0·Γ0/ρ(m0)`` (the coupling carries the scale; no reference mass and
no ad-hoc threshold function anywhere).

It is a PURE :class:`~tabpwa.particle_model.base.BasisModel` instance: no
``gamma`` / ``get_gamma_name`` / ``make_mass_width_transform``, so the plain
``pwa`` kernel path refuses them (naming ``amp_model: pwa_gen``).
"""
import numpy as np

from .base import BasisModel, complex_slot_names, register_model
from .bwr_model import phase_space
from .gs_rho_model import gs_unit
from tabpwa.param_constraint import Transform, complex_tail

N_POLE = 3


def _prod_minus_s(roots, skip=None):
    """Ascending coefficients of ``Π_{i ≠ skip} (roots[i] − s)``."""
    c = np.array([1.0])
    for i, r in enumerate(roots):
        if i == skip:
            continue
        out = np.zeros(len(c) + 1)
        out[:len(c)] += r * c
        out[1:] -= c
        c = out
    return c


def kmatrix3_weights(m, g, b):
    """``(den_weights, num_weights)`` for the three-pole K-matrix.

    ``den_weights`` is real (unitarity); ``num_weights`` complex (production
    phases).  ``m``/``g`` are the pole masses and couplings, ``b`` the complex
    production couplings; all length ``N_POLE`` (unused poles contribute zero
    and factor out of the ratio).
    """
    roots = [float(x) ** 2 for x in m]
    pm = _prod_minus_s(roots)
    k_num = sum(float(g[i]) ** 2 * _prod_minus_s(roots, i) for i in range(len(m)))
    p_num = sum(complex(b[i]) * _prod_minus_s(roots, i) for i in range(len(m)))
    den = [float(pm[k]) for k in range(N_POLE + 1)]
    den += [float(k_num[k]) for k in range(N_POLE)]
    return den, [complex(x) for x in p_num]


class KMatrix3WeightsTransform(Transform):
    """Real slots → the 7 denominator + 6 numerator real weights.

    Inputs (all real): ``m1, m2, m3, g1, g2, g3`` and the three coupling
    magnitude/phase pairs ``b{1,2,3}{mag}, b{1,2,3}{phase}`` (suffixes from
    :func:`~tabpwa.param_constraint.complex_tail`).  The numerator heads are
    emitted in the ``ck`` pair form (``abs``, ``angle``), like
    ``KMatrix2WeightsTransform``.
    """

    _has_inverse = False

    def __init__(self, name, den_names, num_names, defaults, tail=None):
        self.tail = tuple(tail) if tail is not None else complex_tail()
        mag, phase = self.tail
        self.name = name
        keys = [f"m{i}" for i in range(1, N_POLE + 1)]
        keys += [f"g{i}" for i in range(1, N_POLE + 1)]
        for i in range(1, N_POLE + 1):
            keys += [f"b{i}{mag}", f"b{i}{phase}"]
        self._keys = tuple(keys)
        self.input_names = [f"{name}_{k}" for k in self._keys]
        self.output_names = (complex_slot_names(num_names, tail=self.tail)
                             + list(den_names))
        self._defaults = dict(defaults)

    def _inputs(self, d):
        d = d or {}
        return {key: float(d.get(in_name, self._defaults[in_name]))
                for key, in_name in zip(self._keys, self.input_names)}

    def _couplings(self, p):
        mag, phase = self.tail
        return [p[f"b{i}{mag}"] * np.exp(1j * p[f"b{i}{phase}"])
                for i in range(1, N_POLE + 1)]

    def forward(self, d):
        p = self._inputs(d)
        b = self._couplings(p)
        m = [p[f"m{i}"] for i in range(1, N_POLE + 1)]
        g = [p[f"g{i}"] for i in range(1, N_POLE + 1)]
        den, num = kmatrix3_weights(m, g, b)
        vals = []
        for n in num:                       # (mag, phase) per head, interleaved
            vals += [float(np.abs(n)), float(np.angle(n))]
        vals += [float(x) for x in den]
        return {out: val for out, val in zip(self.output_names, vals)}


class _KMatrix3Base(BasisModel):
    """Basis, names and transform of the three-pole K-matrix."""

    #: ``ρ(m)`` — the BW running-width shape, provided by the model
    def _rho(self, m):
        raise NotImplementedError

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.tail = complex_tail()

    # -- physical parameters ----------------------------------------

    def _coupling(self, key):
        mag, phase = self.tail
        z = complex(self.kwargs.get(key, 1.0))
        m = abs(z)
        p = float(np.angle(z))
        m = float(self.kwargs.get(f"{key}{mag}", m))
        p = float(self.kwargs.get(f"{key}{phase}", p))
        return m, p

    def _phys(self):
        mass = float(self.kwargs.get("mass", 0.775))
        width = float(self.kwargs.get("width", 0.1))
        mag, phase = self.tail
        out = {}
        for i in range(1, N_POLE + 1):
            out[f"m{i}"] = float(self.kwargs.get(f"m{i}", mass * (1 + 0.1 * (i - 1))))
            out[f"g{i}"] = float(self.kwargs.get(f"g{i}", width / N_POLE))
        for i in range(1, N_POLE + 1):
            m, p = self._coupling(f"b{i}")
            out[f"b{i}{mag}"] = m
            out[f"b{i}{phase}"] = p
        return out

    def get_defaults(self):
        d = {f"{self.name}_mass": float(self.kwargs.get("mass", 0.775)),
             f"{self.name}_width": float(self.kwargs.get("width", 0.1))}
        for key, val in self._phys().items():
            d[f"{self.name}_{key}"] = val
        return d

    # -- basis ------------------------------------------------------

    def den_basis(self, m):
        """``[1, s, s², s³, −iρ, −i·s·ρ, −i·s²·ρ]``, ``s = m²``."""
        m_arr = np.asarray(m, dtype=float)
        s = m_arr ** 2
        rho = np.asarray(self._rho(m_arr), dtype=complex)
        rows = [(s ** k).astype(complex) for k in range(N_POLE + 1)]
        rows += [-1j * (s ** k) * rho for k in range(N_POLE)]
        return rows

    def num_basis(self, m):
        m_arr = np.asarray(m, dtype=float)
        s = m_arr ** 2
        return [(s ** k).astype(complex) for k in range(N_POLE)]

    def get_den_name(self):
        real = [f"{self.name}_s0", f"{self.name}_poly",
                f"{self.name}_m4", f"{self.name}_m6"]
        rho = [f"{self.name}_rho", f"{self.name}_srho", f"{self.name}_s2rho"]
        return real + rho

    def get_num_name(self):
        return [f"{self.name}_n{i}" for i in range(N_POLE)]

    def make_basis_transform(self):
        defaults = {f"{self.name}_{k}": v for k, v in self._phys().items()}
        return KMatrix3WeightsTransform(
            self.name, self.get_den_name(), self.get_num_name(),
            defaults, tail=self.tail)


@register_model("kmatrix3")
class KMatrix3Model(_KMatrix3Base):
    """Three-pole K-matrix whose ``ρ`` is the BW running width (``BWRS``).

    One pole ⇒ ``BWRS`` exactly (``bf² = m0·Γ0``, ``ρ ≡ γ(m)``); the shape is
    :func:`~tabpwa.particle_model.bwr_model.phase_space`, i.e. the particle's
    own daughters, and carries no ``m0``/``g0`` constant (no reference mass at
    all): ``bf² = m0·Γ0/ρ(m0)``.
    """

    def _rho(self, m):
        kw = self.kwargs
        parent = self._parent
        if parent is None or not getattr(parent, "_decays", None):
            raise ValueError(
                f"kmatrix3 '{self.name}': needs the decay tree (daughters) "
                f"to evaluate the running-width shape")
        return phase_space(
            np.asarray(m, dtype=float), int(kw.get("L", self.default_L())),
            float(kw.get("d", 3.0)),
            float(parent._decays[0].outs[0].mass),
            float(parent._decays[0].outs[1].mass))


@register_model("kmatrix3_gs")
class KMatrix3GSModel(_KMatrix3Base):
    """Three-pole K-matrix whose ``ρ`` is the COMPLEX GS unit.

    ``ρ(m) = Γ(m)/g0 + i·fs(m²)/m0`` (:func:`~tabpwa.particle_model.
    gs_rho_model.gs_unit`) — the same complex shape ``GS_rho`` uses, so the
    denominator is dispersive: with ONE pole and ``bf² = m0·Γ0``::

        d = −i·ρ·bf² = g0·fs − i·m0·Γ   ⇒   f = β / (m0² − s + g0·fs − i·m0·Γ)

    i.e. exactly the GS propagator.  The extra poles are the K-matrix
    generalisation with the same dispersive phase space.

    A pure ``BasisModel``: no ``gamma`` / ``get_gamma_name`` /
    ``make_mass_width_transform``.  Daughters come from the decay tree when
    attached, else ``daug2Mass`` / ``daug3Mass`` (defaults π±, π0).
    """

    _L_FALLBACK = 1

    def _rho(self, m):
        kw = self.kwargs
        parent = self._parent
        if parent is not None and getattr(parent, "_decays", None):
            d2 = float(parent._decays[0].outs[0].mass)
            d3 = float(parent._decays[0].outs[1].mass)
        else:
            d2 = float(kw.get("daug2Mass", 0.13957039))
            d3 = float(kw.get("daug3Mass", 0.1349768))
        return gs_unit(np.asarray(m, dtype=float),
                       float(kw.get("mass", 0.775)),
                       int(kw.get("L", self.default_L())),
                       float(kw.get("d", 3.0)), d2, d3)
