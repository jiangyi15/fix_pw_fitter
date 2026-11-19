"""``kmatrix_n_ieps`` — the one-channel K-matrix with an ``m² − iε`` prescription.

A case-1 COPY of :mod:`~tabpwa.particle_model.kmatrix_n_model`:
``kmatrix_n`` / ``kmatrix_n_gs`` stay untouched.  This module puts the
infinitesimal ``iε`` where the physics wants it — inside EVERY pole factor of
the denominator::

    Π_i (m_i² − s + iε)

and keeps only the FIRST-ORDER term (``ε²`` and higher are dropped), which is
exact for every simple pole::

    Π_i (m_i² − s + iε) = Π(s) − iε·Π'(s) + O(ε²)
                        = Π(s) + iε·D(s) + O(ε²),
    D(s) ≡ Σ_i Π_{j≠i}(m_j² − s)          (degree N − 1)

so the extra basis rows are ``i·s^k`` (k = 0 … N−1) with the DERIVED weights

    d_k = −(k + 1)·w_{k+1}                (w = the real-row weights of Π)

— no new fit parameter, nothing to configure but ``ieps``::

    den rows  [1, s, …, s^N] + [−iρ·s^k] + [i·s^k]·(ieps·d_k)

and the NUMERATOR is shifted by the same rule — ``P(s) = Σ_i β_i Π_{j≠i}`` is
a product of the same factors, so ``P(s−iε) = P − iεP' + O(ε²)`` and

    n'_k = n_k − iε(k+1)·n_{k+1}          (no new head: the numerator weights
                                           are already complex — mag/phase)

i.e. the amplitude is ``A(s−iε)`` to first order in every polynomial factor
(the ``−iρ·U`` term is still evaluated at ``s``; shifting it too would need
ρ-row ε-weights).

Each pole then sits at ``s = m_i² + iε`` instead of on the real axis, so the
configurations that used to give ``inf`` give a finite amplitude:

* a pole mass below the two-body threshold (``ρ(m_i) = 0``),
* a zero width at an above-threshold pole (``g_i² = 0``).

It is a REGULATOR, not physics: the physical region is shifted by
``O(ε·|D| / |den|)``, so keep ``ieps`` far below ``|den|`` at the poles.

Two limits, by construction:

* a DEGENERATE pair (two equal pole masses) is a double root, where ``D = 0``
  as well: the linear-ε term cannot lift it (that needs the dropped ``ε²``),
  so the amplitude still diverges there — require distinct pole masses;
* ``gs_unit`` is ``nan`` below the two-body branch point (it is a loop
  function, not a phase space), so ``kmatrix_n_gs_ieps`` cannot regulate a
  below-threshold pole; above threshold it works like the BW shape.

``ieps = 0`` (the default) is exactly ``kmatrix_n``: no rows are added and
``get_den_name()`` is identical, so the prescription is strictly opt-in and
the two models agree bit for bit.

Config::

    particle:
      f2:
        model: kmatrix_n_ieps        # or kmatrix_n_gs_ieps (GS unit)
        mass_list: [0.775, 1.45]
        width_list: [0.149, 0.30]
        ieps: 1e-6                   # the iε of every pole factor (0 = off)

The ``ieps·d_k`` weights are transform OUTPUTS (like the exact ``beta0``
head), never fitted parameters.
"""
import numpy as np

from .base import register_model
from .kmatrix_n_model import (
    KMatrixNGSModel,
    KMatrixNModel,
    KMatrixNWeightsTransform,
    kmatrix_n_weights,
)


def ieps_name(name, k):
    """Name of the weight of the ``i·s^k`` regulator row."""
    return f"{name}_ieps{k}"


class KMatrixNIepsWeightsTransform(KMatrixNWeightsTransform):
    """The parent transform plus the ``iε·D`` rows.

    ``forward`` reuses the parent's weights and appends ``ieps·d_k`` with
    ``d_k = −(k+1)·w_{k+1}`` (the coefficients of ``D = −Π'``), so the extra
    weights are DERIVED from the pole masses — no new input slot.
    The extra rows have no parameter of their own, but their weights DO depend
    on the fitted masses (``d_k = −(k+1)w_{k+1}``), so ``backward`` routes
    their cotangents into the real-row slots they came from and lets the
    parent's analytic VJP do the rest.
    """

    def __init__(self, *args, ieps=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ieps = float(ieps)

    def _numerator(self, d):
        """The parent's complex numerator coefficients ``n_k`` (unshifted)."""
        p = self._inputs(d)
        n = self.n_pole
        mag, phase = self.tail
        m = [p[f"mass{i}"] for i in range(n)]
        w = [p[f"width{i}"] for i in range(n)]
        beta = [1.0 + 0.0j]
        beta += [p[f"beta{i}{mag}"] * np.exp(1j * p[f"beta{i}{phase}"])
                 for i in range(1, n)]
        g2 = self._coupling_sq(m, w)
        _, num = kmatrix_n_weights(m, list(np.asarray(g2, dtype=float)), beta)
        return [complex(x) for x in num], m, w, beta

    def _shifted(self, nk):
        """``n_k − iε(k+1)·n_{k+1}`` — the numerator at ``s − iε``."""
        n = self.n_pole
        return [nk[k] - 1j * self.ieps * (k + 1)
                * (nk[k + 1] if k + 1 < n else 0.0) for k in range(n)]

    def forward(self, d):
        out = super().forward(d)
        if not self.ieps:
            return out
        n = self.n_pole
        nk, _m, _w, _b = self._numerator(d)
        for k, z in enumerate(self._shifted(nk)):
            a = abs(z)
            out[self.output_names[2 * k]] = a
            out[self.output_names[2 * k + 1]] = float(np.angle(z)) if a else 0.0
        w_real = [float(out[m]) for m in self.output_names[2 * n:2 * n + n + 1]]
        for k in range(n):
            out[ieps_name(self.name, k)] = -self.ieps * (k + 1) * w_real[k + 1]
        return out

    def backward(self, grad_out, d_in=None):
        """Analytic VJP: pull the cotangents back through the two hidden maps.

        The subclass changes two sets of outputs without adding inputs, and
        both are LINEAR in the parent's own slots, so the cotangents are
        mapped back before the parent's adjoint runs::

            numerator   n'_k = n_k − iε(k+1)·n_{k+1}
                        ⇒  q_k = q'_k + iεk·q'_{k−1}        (q' from the
                            emitted |n'|, arg n')
            ε rows      e_k  = −ε(k+1)·w_{k+1}
                        ⇒  c_{w,k+1} −= ε(k+1)·c_{e,k}

        Both maps are exact (verified against FD to 1e-10 at ε = 1e-2, where
        an ``O(ε)`` defect would be glaring), so no ``forward`` is evaluated.
        """
        if not self.ieps:
            return super().backward(grad_out, d_in=d_in)
        d = self._last_input if d_in is None else d_in
        n = self.n_pole
        nk, _m, _w, _b = self._numerator(d)
        nk_s = self._shifted(nk)

        qp = []
        for k in range(n):
            cm = float(grad_out.get(self.output_names[2 * k], 0.0))
            cp = float(grad_out.get(self.output_names[2 * k + 1], 0.0))
            a = abs(nk_s[k])
            qp.append(0j if a == 0.0
                      else cm * nk_s[k] / a + 1j * cp * nk_s[k] / (a * a))
        q = [qp[k] + (1j * self.ieps * k * qp[k - 1] if k else 0j)
             for k in range(n)]

        # back onto the parent's (magnitude, phase) slots of the UNSHIFTED n
        g = dict(grad_out)
        for k in range(n):
            a = abs(nk[k])
            if a == 0.0:
                g[self.output_names[2 * k]] = 0.0
                g[self.output_names[2 * k + 1]] = 0.0
                continue
            u = nk[k] / a
            g[self.output_names[2 * k]] = float((np.conj(u) * q[k]).real)
            g[self.output_names[2 * k + 1]] = float(
                a * (np.conj(1j * u) * q[k]).real)

        # ε-row cotangents -> the real-row counts they were built from
        real = self.output_names[2 * n:2 * n + n + 1]
        for k in range(n):
            c = g.pop(ieps_name(self.name, k), 0.0)
            if c:
                g[real[k + 1]] = g.get(real[k + 1], 0.0) \
                    - self.ieps * (k + 1) * c
        return super().backward(g, d_in=d)


class _IepsBasis:
    """``+iε·D`` rows on top of a ``kmatrix_n`` basis (mixin, no parent edits)."""

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        ieps = float(kwargs.get("ieps", 0.0))
        if ieps < 0.0:
            raise ValueError(
                f"'{name}': ieps must be >= 0 (0 disables the regulator), "
                f"got {ieps!r}")
        self.ieps = ieps

    def den_basis(self, m):
        rows = super().den_basis(m)
        if self.ieps:
            s = np.asarray(m, dtype=float) ** 2
            rows += [(1j * s ** k).astype(complex) for k in range(self.n_pole)]
        return rows

    def get_den_name(self):
        names = super().get_den_name()
        if self.ieps:
            names = names + [ieps_name(self.name, k)
                             for k in range(self.n_pole)]
        return names

    def make_basis_transform(self):
        defaults = {f"{self.name}_{k}": v
                    for k, v in self._transform_defaults().items()}
        return KMatrixNIepsWeightsTransform(
            self.name, self.get_den_name(), self.get_num_name(), defaults,
            tail=self.tail, n_pole=self.n_pole,
            coupling_sq=self._coupling_sq,
            coupling_jac=self._coupling_sq_jac, ieps=self.ieps)


@register_model("kmatrix_n_ieps")
class KMatrixNIepsModel(_IepsBasis, KMatrixNModel):
    """``kmatrix_n`` + the ``iε`` pole prescription (``ρ = phase_space``)."""


@register_model("kmatrix_n_gs_ieps")
class KMatrixNGSIepsModel(_IepsBasis, KMatrixNGSModel):
    """``kmatrix_n_gs`` + the ``iε`` pole prescription (``ρ = gs_unit``)."""
