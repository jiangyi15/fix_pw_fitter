"""``ExpCheb`` — the exponential lineshape as a Chebyshev/Bessel basis model.

The ``Exp`` lineshape is ``A(m) = exp(-k·(m² − m0²))``.  This model is a
:class:`~tabpwa.particle_model.base.BasisModel` for it, i.e. a ratio of two
linear bases (here ONE denominator row) with the fitted parameters entering
through the WEIGHTS only::

    u = s_c − s,          s_c = (m_lo² + m_hi²)/2,   half = (m_hi² − m_lo²)/2
    t = u / half                                                (rows, |t| ≤ 1)
    β = k·half,           I_n = modified Bessel function

    e^{k·u} = e^{k·mid}·[ I_0(β) + 2·Σ_{n≥1} I_n(β)·T_n(t) ]     (exact series)
    mid = m0² − s_c

    A(m) = e^{k(m0² − s_c)}·[ I_0(β) + 2 Σ_n I_n(β) T_n(t) ]
         = [ Σ_n w_n(k, m0)·T_n(t) ] / 1

so the rows are the Chebyshev polynomials ``T_n(t)`` — BOUNDED, smooth and
parameter-free, which is what the kernel's linear-interpolation tables need
(unlike an ``e^u·T_n`` row, whose growing envelope is what forces the
``gamma_clip`` of the ``ExpSpline``/SVD family).

Why this basis:

* the weights are closed form in the fitted parameters,
  ``w_n = e^{k(m0²−s_c)}·(1 or 2)·I_n(k·half)``, so `k` and `m0` are ANALYTIC
  (no k-grid, no spline weights, no SVD) and their derivatives are exact
  (``I_n'(β) = (I_{n−1}+I_{n+1})/2``);
* the truncation error is set by ``β`` alone and falls geometrically
  (``I_n(β)``): with 15 rows it is ~6e-10 over a moderate window, versus the
  Taylor-in-``k·u`` form which stalls (its argument is twice as large and the
  rows carry the whole dynamic range);
* the error is UNIFORM in the fitted ``m0`` — a drift of ``m0`` from the
  reference only changes the scalar prefactor ``e^{k(m0²−s_c)}``.

End to end the accuracy floor is the kernel's table interpolation of the rows,
not this truncation: ``~1.4e-3/n_interp²`` (2.8e-7 at the default
``n_interp=2000``).  Raise it with the build-time scope when needed::

    with build_defaults.scope(n_interp=4000):
        ...

Config::

    particle:
      sigma:
        mass: 0.5                 # m0 (fitted parameter, this is its default)
        model: ExpCheb
        k: 1.0                    # k (fitted parameter)
        n_cheb: 14                # Chebyshev rows = n_cheb + 1
        m_range: [0.28, 2.0]      # the window `t` is mapped onto
                                  # (default [2·M_PION, M_B − M_PION])

The fitted parameters are ``{name}_mass`` and ``{name}_k``; the single
denominator weight (1) is a transform OUTPUT, never fitted.
"""
import numpy as np
from scipy.special import iv

from tabpwa.param_constraint import Transform, complex_tail

from .base import BasisModel, complex_slot_names, register_model

M_PION = 0.1396
M_B_MESON = 5.279


def cheb_t(m_sq, s_c, half):
    """``t = (s_c − s)/half`` — the Chebyshev variable, in ``[-1, 1]``."""
    return (s_c - np.asarray(m_sq, dtype=float)) / half


class ExpChebWeights(Transform):
    """``(m0, k) → w_n = e^{k(m0²−s_c)}·(1 or 2)·I_n(k·half)`` (+ the row-1 weight).

    The numerator heads are emitted as the usual ``(magnitude, phase)`` pair
    (the weights are real and positive for ``k > 0``, so the phase is 0/π); the
    single denominator weight is the constant 1.

    The VJP is the inherited centred finite difference: there are only TWO
    inputs, so it costs four forward evaluations, and the analytic chain would
    have to invert the polar emission for no measurable gain.
    """

    _has_inverse = False

    def __init__(self, name, num_names, den_name, s_c, half,
                 mass_name=None, k_name=None, mass=None, k=None):
        mag, phase = self.tail = complex_tail()
        self.name = name
        self.n_cheb = len(num_names) - 1
        self.s_c = float(s_c)
        self.half = float(half)
        self.mass_name = mass_name or f"{name}_mass"
        self.k_name = k_name or f"{name}_k"
        self.input_names = [self.mass_name, self.k_name]
        self.output_names = complex_slot_names(num_names, tail=self.tail) \
            + [den_name]
        self._defaults = {self.mass_name: float(mass if mass is not None else 0.5),
                          self.k_name: float(k if k is not None else 1.0)}

    def _inputs(self, d):
        d = d or {}
        return (float(d.get(self.mass_name, self._defaults[self.mass_name])),
                float(d.get(self.k_name, self._defaults[self.k_name])))

    def _weights(self, m0, k):
        """The ``n_cheb + 1`` real numerator weights, in head order."""
        beta = k * self.half
        pre = np.exp(k * (m0 * m0 - self.s_c))
        w = [pre * iv(0, beta)]
        w += [2.0 * pre * iv(n, beta) for n in range(1, self.n_cheb + 1)]
        return w

    def forward(self, d):
        m0, k = self._inputs(d)
        vals = []
        for w in self._weights(m0, k):
            vals += [float(abs(w)), float(np.angle(w))]
        vals.append(1.0)                     # the single denominator row: 1
        return {out: val for out, val in zip(self.output_names, vals)}


@register_model("ExpCheb")
class ExpChebModel(BasisModel):
    """``exp(-k(m² − m0²))`` as ``Σ_n w_n(k, m0)·T_n(t) / 1`` (see the module)."""

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.mass0 = float(kwargs.get("mass", 0.5))
        self.k0 = float(kwargs.get("k", 1.0))
        self.n_cheb = int(kwargs.get("n_cheb", 14))
        if self.n_cheb < 1:
            raise ValueError(f"ExpCheb '{name}': n_cheb must be >= 1")
        mr = kwargs.get("m_range")
        if mr is None:
            m_min, m_max = 2.0 * M_PION, M_B_MESON - M_PION
        else:
            m_min, m_max = float(mr[0]), float(mr[1])
        if not m_min < m_max:
            raise ValueError(f"ExpCheb '{name}': m_range must be [lo, hi] with "
                             f"lo < hi, got [{m_min}, {m_max}]")
        self.m_range = (m_min, m_max)
        s_lo, s_hi = m_min ** 2, m_max ** 2
        self.s_c = 0.5 * (s_lo + s_hi)
        self.half = 0.5 * (s_hi - s_lo)

    # -- fitted parameters ------------------------------------------

    def get_defaults(self):
        return {f"{self.name}_mass": self.mass0,
                f"{self.name}_k": self.k0}

    # -- basis ------------------------------------------------------

    def num_basis(self, m):
        """``T_n(t)`` for n = 0 … n_cheb — bounded, parameter-free rows."""
        t = cheb_t(np.asarray(m, dtype=float) ** 2, self.s_c, self.half)
        return [np.polynomial.chebyshev.chebval(t, np.eye(self.n_cheb + 1)[n])
                .astype(complex) for n in range(self.n_cheb + 1)]

    def den_basis(self, m):
        """One row, the constant 1 (the amplitude is a pure numerator sum)."""
        s = np.asarray(m, dtype=float)
        return [np.ones(s.shape, dtype=complex)]

    def get_num_name(self):
        return [f"{self.name}_n{n}" for n in range(self.n_cheb + 1)]

    def get_den_name(self):
        return [f"{self.name}_one"]

    def make_basis_transform(self):
        return ExpChebWeights(
            self.name, self.get_num_name(), self.get_den_name()[0],
            self.s_c, self.half, mass=self.mass0, k=self.k0)
