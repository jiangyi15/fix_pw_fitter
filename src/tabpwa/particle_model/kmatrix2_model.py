"""Two-pole, one-channel K-matrix lineshape (``kmatrix2``).

Reference model for the :class:`~tabpwa.particle_model.base.BasisModel`
interface.  The amplitude is a rational function of ``s = m²``::

    ρ(s)  = sqrt(1 − (2·m_π)² / s)          # complex below threshold
    K(s)  = g1² / (m1² − s) + g2² / (m2² − s)
    P(s)  = b1  / (m1² − s) + b2  / (m2² − s)
    A(s)  = P(s) / (1 − i·ρ(s)·K(s))

**Not the conventional phase space**: this ``ρ`` is
``sqrt(1 − (2m_π)²/s)``, *not* the ``2k/m`` two-body phase space, so the
``g1/g2`` couplings are not directly comparable to literature values.

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

Production couplings: complex ``ck`` convention
-----------------------------------------------
``b1`` / ``b2`` are complex parameters stored as a **(magnitude, phase)
slot pair**, exactly like a complex ``ck`` parameter
(:func:`tabpwa.param_constraint.complex_tail` /
:meth:`CKProduct._get_val`)::

    b = mag · exp(i·phase)

The slot suffixes come from :func:`tabpwa.param_constraint.complex_tail`
— default ``("r", "i")`` (misleading names; they are *not*
real/imaginary parts), or e.g. ``("rho", "phi")`` under
``build_defaults.scope(complex_tail=(...))``.  The model's parameter
names are therefore::

    f"{name}_b1{mag}", f"{name}_b1{phase}", f"{name}_b2{mag}", f"{name}_b2{phase}"

with defaults ``mag = 1``, ``phase = 0`` (so ``b = 1``).  A YAML/kwarg
``b1: 1.0+0.3j`` is converted to ``(abs(b1), angle(b1))`` for the pair.

A **common phase** ``b_i → e^{iα}·b_i`` multiplies both numerator weights
by ``e^{iα}``, i.e. it is degenerate with the wave coupling ``ck`` — only
``arg(b2/b1)`` is physical, so in a fit one phase should be fixed (or only
the ratio interpreted).

Transform boundary: **real on both sides**
------------------------------------------
``param_constraint.Transform.fd_backward`` requires real scalar inputs and
produces a real VJP for a real parameter, so the transform speaks only
real slots:

* inputs (all real): ``m1, m2, g1, g2`` and the four coupling
  magnitude/phase slots above;
* outputs (all real): the expanded numerator slots followed by the 5
  real denominator weights.  The numerator weights use the **same
  (magnitude, phase) tail convention as the inputs** — they become ``ck``
  slots, so ``CKProduct.build_ck`` reads ``{name}{mag}`` /
  ``{name}{phase}`` and forms ``r·exp(iθ)``::

      head h = {name}_n0 / {name}_n1     (get_num_name(); one per basis row)
      slots  = h{mag}, h{phase}          (num_slot_names(); the transform outputs)
      values = abs(n_i), angle(n_i)

  with ``abs(0) = 0``, ``angle(0) = 0`` and the usual ``±π`` branch cut.
  Under ``complex_tail=("rho", "phi")`` the slots rename together with
  the ``b`` slots (``n0rho`` / ``n0phi`` ...), while the heads stay
  fixed.  A *real* numerator weight may be supplied as one parameter
  named exactly the head (no exponential applied).

The numerator weights are reassembled
``n_i = mag·exp(i·phase)`` from the real slots (or uses a bare head
directly); only the amplitude is complex.

YAML usage::

    particle:
      k2:
        mass: 0.8           # pole-1 nominal mass (kernel m0 / ordering)
        width: 0.3          # pole-1 nominal width
        model: kmatrix2
        m2: 1.35            # optional overrides
        g1: 0.4
        g2: 0.6
        b1: 1.0             # coupling magnitude/phase via complex_tail
        b2: 1.0

``mass``/``width`` keep the particle found/ordered like any other
resonance; the actual pole parameters are ``m1, m2, g1, g2`` and the
complex ``b1``, ``b2`` (defaults ``m1 = mass``, ``m2 = 1.1·mass``,
``g1 = g2 = width/2``, ``b1 = b2 = 1``), so the model always builds.
"""

import numpy as np

from tabpwa.param_constraint import Transform, complex_tail

from .base import BasisModel, complex_slot_names, register_model
from .bwr_model import BWRSModel

#: Default pion mass entering the two-body phase space ρ(s).
M_PI = 0.13957039


class KMatrix2WeightsTransform(Transform):
    """Real slots → the 5 denominator + 4 numerator real weights.

    Inputs (all real): ``m1, m2, g1, g2`` and the two coupling
    magnitude/phase pairs ``b1{mag}, b1{phase}, b2{mag}, b2{phase}``
    (suffixes from :func:`complex_tail`).  Internally
    ``b = mag·exp(i·phase)``.

    Outputs (all real): the four numerator slots first, then the 5 basis
    denominator weights.  The numerator heads (``{name}_n0``,
    ``{name}_n1``) are expanded to the **same (magnitude, phase) slots as a
    complex ``ck`` parameter**::

        n0{mag}, n0{phase}, n1{mag}, n1{phase}   (mag, phase = complex_tail())

    i.e. ``out[n0{mag}] = abs(n0)``, ``out[n0{phase}] = angle(n0)``.  The
    ``ck_map``/``CKProduct.build_ck`` machinery reads exactly this pair and
    forms ``r·exp(iθ)``, so the transform output dict can be fed straight
    to it.  ``abs(0) = 0`` and ``angle(0) = 0``; the phase has the usual
    ``±π`` branch cut (same as ``ck``).

    The forward returns plain Python floats.  ``backward`` is inherited
    from :class:`~tabpwa.param_constraint.Transform` (the centered
    finite-difference VJP); with real inputs and real outputs it is a
    legitimate VJP for a real parameter space (the polar map is smooth
    away from ``n = 0``).
    """

    # the weights are a many-to-one image of the physical parameters —
    # no inverse.
    _has_inverse = False

    def __init__(self, name, den_names, num_names, defaults, tail=None):
        self.tail = tuple(tail) if tail is not None else complex_tail()
        mag, phase = self.tail
        self.name = name
        self._keys = (
            "m1", "m2", "g1", "g2",
            f"b1{mag}", f"b1{phase}", f"b2{mag}", f"b2{phase}",
        )
        self.input_names = [f"{name}_{k}" for k in self._keys]
        # numerator heads → expanded real (magnitude, phase) output slots
        self.output_names = (complex_slot_names(num_names, tail=self.tail)
                             + list(den_names))
        self._defaults = dict(defaults)

    # -- helpers ----------------------------------------------------

    def _inputs(self, d):
        """Resolved eight real slots (defaults fill missing keys)."""
        d = d or {}
        return {key: float(d.get(in_name, self._defaults[in_name]))
                for key, in_name in zip(self._keys, self.input_names)}

    def _couplings(self, p):
        """``(b1, b2)`` complex from the magnitude/phase slots."""
        mag, phase = self.tail
        return (
            p[f"b1{mag}"] * np.exp(1j * p[f"b1{phase}"]),
            p[f"b2{mag}"] * np.exp(1j * p[f"b2{phase}"]),
        )

    @staticmethod
    def _weights(m1, m2, g1, g2, b1, b2):
        """Closed-form basis weights (denominator real, numerator polar).

        Numerator weights are ``n0 = b1·m2² + b2·m1²`` and
        ``n1 = −(b1+b2)``; they are emitted in the same (magnitude,
        phase) form as a complex ``ck`` slot: ``(abs(n), angle(n))``.
        """
        m1s, m2s = m1 * m1, m2 * m2
        g1s, g2s = g1 * g1, g2 * g2
        den = [
            m1s * m2s,                                 # 1       (kernel s0)
            -(m1s + m2s),                              # m²      (kernel poly)
            1.0,                                       # m⁴
            g1s * m2s + g2s * m1s,                     # −i·ρ row
            g1s + g2s,                                 # +i·s·ρ row
        ]
        n0 = b1 * m2s + b2 * m1s
        n1 = -(b1 + b2)
        num = [float(np.abs(n0)), float(np.angle(n0)),
               float(np.abs(n1)), float(np.angle(n1))]
        return den, num

    # -- transform --------------------------------------------------

    def forward(self, d):
        p = self._inputs(d)
        b1, b2 = self._couplings(p)
        den, num = self._weights(p["m1"], p["m2"], p["g1"], p["g2"], b1, b2)
        vals = [float(v) for v in num + den]
        return {out: val for out, val in zip(self.output_names, vals)}


@register_model("kmatrix2")
class KMatrix2Model(BasisModel):
    """Two-pole one-channel K-matrix shape (see module docstring)."""

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.m_pi = float(kwargs.get("m_pi", M_PI))
        # magnitude/phase suffixes for the complex production couplings
        self.tail = complex_tail()

    # -- physical parameters ---------------------------------------

    def _coupling(self, key):
        """``(magnitude, phase)`` for a production coupling ``b1`` / ``b2``.

        A complex kwarg ``b1`` is converted to ``(abs, angle)``; explicit
        tail-named kwargs (``b1{mag}`` / ``b1{phase}``) override the split.
        """
        mag, phase = self.tail
        z = complex(self.kwargs.get(key, 1.0))
        m = abs(z)
        p = float(np.angle(z))
        m = float(self.kwargs.get(f"{key}{mag}", m))
        p = float(self.kwargs.get(f"{key}{phase}", p))
        return m, p

    def _phys(self):
        """Eight real physical slots, with defaults from ``mass`` / ``width``."""
        mass = float(self.kwargs.get("mass", 0.775))
        width = float(self.kwargs.get("width", 0.1))
        mag, phase = self.tail
        out = {
            "m1": float(self.kwargs.get("m1", mass)),
            "m2": float(self.kwargs.get("m2", mass * 1.1)),
            "g1": float(self.kwargs.get("g1", width / 2.0)),
            "g2": float(self.kwargs.get("g2", width / 2.0)),
        }
        for key in ("b1", "b2"):
            m, p = self._coupling(key)
            out[f"{key}{mag}"] = m
            out[f"{key}{phase}"] = p
        return out

    def get_defaults(self):
        """Kernel mass/width + every real transform input (tail-named)."""
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
        """Numerator weight **head** names (one per ``num_basis`` row).

        The complex ``(magnitude, phase)`` slots are the tail-expanded
        names from :meth:`num_slot_names` (``ck`` convention); the head is
        also a valid single real-parameter name.
        """
        return [f"{self.name}_n0", f"{self.name}_n1"]

    # -- weights ----------------------------------------------------

    def _resolved(self, params):
        """Params enriched with the nine real weights when they are absent.

        Weights already present in *params* win; otherwise they are
        computed from the eight real physical slots (explicit physical
        overrides honoured, model defaults fill the rest).
        """
        params = {} if params is None else params
        names = self.get_den_name() + self.num_slot_names()
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

    # -- transform --------------------------------------------------

    def make_basis_transform(self):
        """Map the eight real slots to the nine real basis weights."""
        defaults = {f"{self.name}_{k}": v for k, v in self._phys().items()}
        return KMatrix2WeightsTransform(
            self.name, self.get_den_name(), self.get_num_name(),
            defaults, tail=self.tail)

@register_model("kmatrix2_bf")
class KMatrix2BFModel(KMatrix2Model, BWRSModel):
    """Two-pole K-matrix whose phase-space shape IS the BWRS running width.

    ``ρ(s)`` is the particle's own :meth:`~bwr_model.BWRModel.gamma`, the same
    ``(q/q0)^{2L+1}·(m0/m)·B'²`` the ``BWRS`` kernel uses — no ad-hoc
    ``sqrt(1 − (2m_π)²/s)`` and no extra constant: the shape is dimensionless
    and the couplings carry the dimensions.

    With ONE pole (``g2 = b2 = 0``) the amplitude is exactly ``BWRS``::

        f = β / (m0² − s − i·ρ·bf²) = β / (m0² − s − i·m0·Γ0·γ(m))

    for ``bf² = m0·Γ0`` — i.e. the K-matrix coupling is ``m0·Γ0`` and ``ρ`` is
    ``γ(m)``.  The second pole turns it into the K-matrix generalisation.

    ``get_gamma_name()`` is empty on purpose: there is no single running width
    to give the plain-``pwa`` kernel, so that path REFUSES this model (naming
    ``amp_model: pwa_gen``) instead of silently plotting a Breit-Wigner.
    """

    def den_basis(self, m):
        """``[1, m², m⁴, −i·γ(m), +i·s·γ(m)]`` — ``γ`` from the BWRS table."""
        m_arr = np.asarray(m, dtype=float)
        s = m_arr ** 2
        rho = np.asarray(self.gamma(m_arr)[0], dtype=complex)
        return [
            np.ones_like(s, dtype=complex),
            s.astype(complex),
            (s ** 2).astype(complex),
            -1j * rho,
            1j * s * rho,
        ]

    def get_gamma_name(self):
        """Empty: no plain-``pwa`` width exists for a two-pole K-matrix."""
        return []

    def make_mass_width_transform(self):
        """None: this model's mapping is its BASIS transform (K-matrix weights)."""
        return None
