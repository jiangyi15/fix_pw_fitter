"""Two-pole, one-channel K-matrix lineshape (``kmatrix2``).

Reference model for the :class:`~tabpwa.particle_model.base.BasisModel`
interface.  The amplitude is a rational function of ``s = m²``::

    ρ(s)  = sqrt(1 − (2·m_π)² / s)          # complex below threshold
    K(s)  = g1² / (m1² − s) + g2² / (m2² − s)
    P(s)  = b1  / (m1² − s) + b2  / (m2² − s)
    A(s)  = P(s) / (1 − i·ρ(s)·K(s))

Written over the common denominator ``(m1² − s)(m2² − s)`` this is the
basis ratio ``(Σ n_i N_i) / (Σ g_j G_j)`` with::

    den_basis = [ 1 , m² , m⁴ , −i·ρ(s) , +i·s·ρ(s) ]

    1       weight:  m1²m2²                                (kernel ``s0``)
    m²      weight:  −(m1² + m2²)                          (kernel ``poly``)
    m⁴      weight:  1
    −i·ρ    weight:  g1²m2² + g2²m1²
    +i·s·ρ  weight:  g1² + g2²

    num_basis = [ 1 , m² ]

    1       weight:  n0 = b1·m2² + b2·m1²
    m²      weight:  n1 = −(b1 + b2)

Denominator weights are **real** (unitarity); every ``i`` / ``−i`` factor
belongs to the basis row (the kernel ``coeff`` vector is a real
``double[]``).  The numerator weights may be complex (production phases).

Transform boundary: **real on both sides**
------------------------------------------
``param_constraint.Transform.fd_backward`` requires real scalar inputs and
produces a real VJP for a real parameter, so the transform speaks only
real slots — the same ``_r``/``_i`` convention as the ``ck`` parameters:

* inputs: ``m1, m2, g1, g2`` (real) and ``b1_r, b1_i, b2_r, b2_i``;
* outputs: the 5 real denominator weights and the real numerator slots::

      n0 = n0_r + i·n0_i ,  n0_r = b1r·m2² + b2r·m1² ,
                            n0_i = b1i·m2² + b2i·m1²
      n1 = n1_r + i·n1_i ,  n1_r = −(b1r + b2r) ,
                            n1_i = −(b1i + b2i)

:meth:`KMatrix2Model._num_weights` reassembles ``n0`` / ``n1`` from the
real slots internally; only the amplitude is complex.

YAML usage::

    particle:
      k2:
        mass: 0.8           # pole-1 nominal mass (kernel m0 / ordering)
        width: 0.3          # pole-1 nominal width
        model: kmatrix2
        m2: 1.35            # optional overrides
        g1: 0.4
        g2: 0.6
        b1_r: 1.0           # production couplings, real/imag slots
        b1_i: 0.0
        b2_r: 1.0
        b2_i: 0.0

``mass``/``width`` keep the particle found/ordered like any other
resonance; the actual pole parameters are ``m1, m2, g1, g2`` and the
complex ``b1 = b1_r + i·b1_i``, ``b2 = b2_r + i·b2_i`` (defaults
``m1 = mass``, ``m2 = 1.1·mass``, ``g1 = g2 = width/2``,
``b1 = b2 = 1``), so the model always builds.  ``b1`` / ``b2`` may also
be given directly as (possibly complex) YAML values; they are split into
the real slots.
"""

import numpy as np
from tabpwa.param_constraint import Transform

from .base import BasisModel, register_model

#: Default pion mass entering the two-body phase space ρ(s).
M_PI = 0.13957039


class KMatrix2WeightsTransform(Transform):
    """Real physical slots → the 5 denominator + 4 numerator real weights.

    Inputs (all real): ``m1, m2, g1, g2, b1_r, b1_i, b2_r, b2_i``.
    Outputs (all real): the 5 basis denominator weights, then
    ``n0_r, n0_i, n1_r, n1_i``.

    The forward is closed-form and returns plain Python floats (the
    outputs are polynomial in the inputs).  ``backward`` is inherited from
    :class:`~tabpwa.param_constraint.Transform` (the centered
    finite-difference VJP); with real inputs and real outputs it is a
    legitimate VJP for a real parameter space.
    """

    # the weights are a many-to-one image of the physical parameters —
    # no inverse.
    _has_inverse = False

    _KEYS = ("m1", "m2", "g1", "g2", "b1_r", "b1_i", "b2_r", "b2_i")

    def __init__(self, name, den_names, num_names, defaults):
        self.input_names = [f"{name}_{k}" for k in self._KEYS]
        self.output_names = list(den_names) + list(num_names)
        self._defaults = dict(defaults)
        self.name = name

    # -- helpers ----------------------------------------------------

    def _inputs(self, d):
        """Resolved eight real physical slots (defaults fill missing keys)."""
        d = d or {}
        return {key: float(d.get(in_name, self._defaults[in_name]))
                for key, in_name in zip(self._KEYS, self.input_names)}

    @staticmethod
    def _weights(m1, m2, g1, g2, b1_r, b1_i, b2_r, b2_i):
        """Closed-form real basis weights (denominator, then numerator)."""
        m1s, m2s = m1 * m1, m2 * m2
        g1s, g2s = g1 * g1, g2 * g2
        den = [
            m1s * m2s,                                 # 1       (kernel s0)
            -(m1s + m2s),                              # m²      (kernel poly)
            1.0,                                       # m⁴
            g1s * m2s + g2s * m1s,                     # −i·ρ row
            g1s + g2s,                                 # +i·s·ρ row
        ]
        num = [
            b1_r * m2s + b2_r * m1s,                   # n0_r
            b1_i * m2s + b2_i * m1s,                   # n0_i
            -(b1_r + b2_r),                            # n1_r
            -(b1_i + b2_i),                            # n1_i
        ]
        return den, num

    # -- transform --------------------------------------------------

    def forward(self, d):
        den, num = self._weights(**self._inputs(d))
        vals = [float(v) for v in den + num]
        return {out: val for out, val in zip(self.output_names, vals)}


@register_model("kmatrix2")
class KMatrix2Model(BasisModel):
    """Two-pole one-channel K-matrix shape (see module docstring)."""

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.m_pi = float(kwargs.get("m_pi", M_PI))

    # -- physical parameters ---------------------------------------

    def _phys(self):
        """Eight real physical slots, with defaults from ``mass`` / ``width``.

        ``b1`` / ``b2`` may be given directly (real or complex); explicit
        ``b1_r`` / ``b1_i`` / ``b2_r`` / ``b2_i`` keys override the split.
        """
        mass = float(self.kwargs.get("mass", 0.775))
        width = float(self.kwargs.get("width", 0.1))
        b1r, b1i = self._split_complex(self.kwargs.get("b1", 1.0))
        b2r, b2i = self._split_complex(self.kwargs.get("b2", 1.0))
        out = {
            "m1": float(self.kwargs.get("m1", mass)),
            "m2": float(self.kwargs.get("m2", mass * 1.1)),
            "g1": float(self.kwargs.get("g1", width / 2.0)),
            "g2": float(self.kwargs.get("g2", width / 2.0)),
            "b1_r": b1r, "b1_i": b1i,
            "b2_r": b2r, "b2_i": b2i,
        }
        for key in ("b1_r", "b1_i", "b2_r", "b2_i"):
            if key in self.kwargs:
                out[key] = float(self.kwargs[key])
        return out

    @staticmethod
    def _split_complex(val):
        """Split a real or complex value into real floats ``(re, im)``."""
        val = complex(val)
        return float(val.real), float(val.imag)

    def get_defaults(self):
        """Kernel mass/width + every real transform input."""
        d = {
            f"{self.name}_mass": float(self.kwargs.get("mass", 0.775)),
            f"{self.name}_width": float(self.kwargs.get("width", 0.1)),
        }
        for key, val in self._phys().items():
            d[f"{self.name}_{key}"] = val
        return d

    # -- basis ------------------------------------------------------

    def den_basis(self, m):
        """``[1, m², m⁴, −i·ρ(s), +i·s·ρ(s)]`` with ``s = m²``.

        Rows are complex; the ``±i`` factors belong here, not to the
        weights.
        """
        m_arr = np.asarray(m, dtype=float)
        s = m_arr ** 2
        rho = np.asarray(
            np.emath.sqrt(1.0 - (2.0 * self.m_pi) ** 2 / s), dtype=complex)
        return [
            np.ones_like(s, dtype=complex),
            s.astype(complex),
            (s ** 2).astype(complex),
            -1j * rho,
            +1j * s * rho,
        ]

    def num_basis(self, m):
        """``[1, m²]``."""
        m_arr = np.asarray(m, dtype=float)
        s = m_arr ** 2
        return [np.ones_like(s, dtype=complex), s.astype(complex)]

    def get_den_name(self):
        """Denominator weight names, in basis order (kernel slots first)."""
        return [
            f"{self.name}_s0",        # 1 weight
            f"{self.name}_poly",      # m² weight
            f"{self.name}_m4",        # m⁴ weight
            f"{self.name}_rho",       # −i·ρ weight
            f"{self.name}_srho",      # +i·s·ρ weight
        ]

    def get_num_name(self):
        """Numerator weight names — real ``_r``/``_i`` slots (``ck`` style)."""
        return [
            f"{self.name}_n0_r", f"{self.name}_n0_i",
            f"{self.name}_n1_r", f"{self.name}_n1_i",
        ]

    # -- weights ----------------------------------------------------

    def _resolved(self, params):
        """Params enriched with the nine real weights when they are absent.

        Weights already present in *params* win; otherwise they are
        computed from the eight real physical slots (explicit physical
        overrides honoured, model defaults fill the rest).
        """
        params = {} if params is None else params
        names = self.get_den_name() + self.get_num_name()
        if all(n in params for n in names):
            return params
        phys = self._phys()
        d = {f"{self.name}_{k}": v for k, v in phys.items()}
        prefix = self.name + "_"
        for key, val in params.items():
            if key.startswith(prefix) and key[len(prefix):] in phys:
                d[key] = val
        resolved = self.make_basis_transform().apply_forward(d)
        resolved.update({n: params[n] for n in names if n in params})
        return resolved

    def _den_weights(self, params):
        """Denominator weights — real (unitarity); coerced to float."""
        r = self._resolved(params)
        return [float(np.real(r[n])) for n in self.get_den_name()]

    def _num_weights(self, params):
        """Reassemble the complex numerator weights from the real slots."""
        r = self._resolved(params)
        n0r, n0i, n1r, n1i = (float(np.real(r[n]))
                              for n in self.get_num_name())
        return [complex(n0r, n0i), complex(n1r, n1i)]

    # -- transform --------------------------------------------------

    def make_basis_transform(self):
        """Map the eight real slots to the nine real basis weights."""
        defaults = {f"{self.name}_{k}": v for k, v in self._phys().items()}
        return KMatrix2WeightsTransform(
            self.name, self.get_den_name(), self.get_num_name(), defaults)
