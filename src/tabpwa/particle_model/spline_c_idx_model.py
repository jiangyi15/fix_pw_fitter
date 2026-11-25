"""``spline_c_idx`` — a COMPLEX cubic spline carried by the NUMERATOR.

A :class:`~tabpwa.particle_model.base.BasisModel` (the ``pwa_gen`` basis
protocol), NOT a ``BaseModel``: it has no running width and no ``gamma``, so
the plain-``pwa`` kernel format cannot express it and refuses it naming
``amp_model: pwa_gen``.

The construction is TFPWA's ``Interp1DSplineIdx`` (``tf_pwa/amp/interpolation.py``,
registered ``spline_c_idx``), ported onto this tree's machinery:

* the ROWS are solved ONCE at construction by
  :func:`tabpwa.particle_model.spline_k_model.spline_basis_matrix` — the
  VERBATIM port of TFPWA's ``spline_xi_matrix`` (piecewise CUBIC monomials
  ``a + b·m + c·m² + d·m³`` per interval, one column per knot);
* ``bc_type`` is ``"not-a-knot"`` (TFPWA's default; ``"natural"`` is
  reachable and differs by up to 0.449 in a cardinal row — not cosmetic);
* ``with_bound`` (default ``False``) pins the first and last knot values to 0
  by DROPPING columns 0 and N−1, so ``n_free = N − 2``; ``True`` keeps all
  ``N`` columns (``n_free = N``) — column omission is exact because the
  matrix is linear in the knot values;
* the lineshape is ``A(m) = Σ_i c_i·B_i(m)`` with ``c_i ∈ ℂ`` and REAL rows,
  one complex coefficient per FITTED knot::

      B_i(points[j]) = δ_ij                      (the cardinal property)

  **``c`` stands for COMPLEX**: the knot VALUES are the fitted coefficients
  (``c_i = A(points[i])``), and the real and imaginary parts share the same
  row.  In tabpwa that is the ``complex_tail`` pair the ``ck`` machinery
  already carries (``CKProduct``'s ``r·e^{iθ}``), so there is NO extra
  transform and NO doubled row axis.  The heads are spelled
  ``{name}_point_{i}`` — TFPWA's spelling, so a constraint file transfers
  verbatim — and the slots are ``{name}_point_{i}r`` / ``{name}_point_{i}i``
  for the fitted knots only.

``pwa_gen`` expands ``num_basis`` into ONE FLAT ``ck`` SLOT PER ROW
(``AmplitudeModel.wave_expansion``), so a geometry wave owns ``n_free``
contiguous slots ``ck_{w,i} = g_ls(w)·c_i``.  The denominator is the single
constant row ``[1]`` (the shared fixed ``"one"`` — the ``SkipModel`` shape),
so ``|A|² ∝ |Σ_i c_i·B_i(m)|²`` and the normalisation is the fit's.

**Nothing but the coefficients is fitted.**  The knots are config constants,
so ``make_basis_transform() -> None`` (a coefficient IS the value at its
point — nothing to derive) and ``get_defaults() == {}``: the heads are plain
``ck`` parameters, which ``pwa_gen`` accepts WITHOUT a transform (a DERIVED
weight — ``kmatrix_n``'s β, ``ExpCheb``'s Bessel weights — still needs one).

GAUGE: TFPWA hard-pins one coefficient to ``1 + 0i`` (``fix_idx``, default
``N//2 − 1``; a NEGATIVE index also means the middle, and TFPWA's explicit
``null`` — "no pin" — has no counterpart here, so ``null`` keeps the default).  tabpwa's
default ``ck_redundancy`` constrain removes the same redundancy but pins the
FIRST coefficient of a wave context instead of the middle knot, so the counted
surface matches (one complex direction less) and the pinned POINT differs.
``fix_idx`` is therefore accepted and validated (and :meth:`fixed_values`
returns the slot dict), but the PIN itself is a constraint-layer job::

    f.set_free(f"{name}_point_{first}r"); f.set_free(f"{name}_point_{first}i")
    f.set_fixed(mdl.fixed_values())          # TFPWA's middle knot

Extrapolation — ``outside`` (default ``"spline"``, TFPWA's ``spline_c_idx``):
the interval index is CLIPPED to ``[0, N−2]`` and the end interval's CUBIC is
evaluated, i.e. cubic continuation, which can GROW outside the support (on an
18-knot production grid ``|A(0)|`` reaches ~2e2).  ``outside="zero"`` is
TFPWA's twin ``spline_c``: exactly 0 outside ``[p_0, p_{N−1})``.  The two
agree inside the support.

Config keys (``particle:`` beside ``mass``/``model``, like ``kmatrix_n``)::

    particle:
      k2:
        mass: 0.80
        model: spline_c_idx
        points: [0.30, 0.75, 1.20, 1.65, 2.10, 2.50]   # OR min_m/max_m/interp_N
        with_bound: false       # default: the end knots are pinned to 0
        bc_type: not-a-knot     # not-a-knot (default) | natural
        outside: spline         # spline (default) | zero
        fix_idx: null           # null/absent or a negative index -> N//2 - 1
        polar: true             # only the polar form maps to tabpwa's slots

``points`` (or ``min_m`` + ``max_m`` + ``interp_N``, a uniform grid) needs
``N > 2`` points (TFPWA's gate); the list must be finite, unique and
ascending, and ``bc_type: not-a-knot`` is singular at exactly ``N == 3``
(its two boundary rows coincide) — use ``natural`` there.  ``polar: false`` has NO tabpwa
counterpart (the ck slots are always magnitude/phase), so it is refused
rather than silently reinterpreted.  ``bc_type: clamped`` is TFPWA's
zero-slope variant and has no port here — refused by name.

TABULATION CAVEAT: TFPWA evaluates the cubic exactly per event; here
``num_basis`` is sampled on the kernel's ``gamma_grid`` (``n_interp``, default
2000) and the kernels re-interpolate it with Catmull-Rom, so the amplitude is
CR∘spline.  Keep ``points`` inside the grid window
(``m_min − 0.01 … m_max + 0.01``); raising ``n_interp`` through
``build_defaults.scope(n_interp=…)`` shrinks that extra error, which is
≈2.1e-7 at the default and falls as ``n_interp⁻³`` (``~1.6e3/n³``; measured
1.07e-4, 1.33e-5, 1.61e-6, 2.05e-7, 2.46e-8 at n = 250/500/1000/2000/4000).
"""

import numpy as np

from .base import BasisModel, register_model
from .spline_k_model import spline_basis_matrix

_BC_TYPES = ("not-a-knot", "natural")
_OUTSIDE = ("spline", "zero")


def _parse_points(name, kwargs):
    """The knot list: ``points`` or (``min_m``, ``max_m``, ``interp_N``)."""
    raw = kwargs.get("points")
    if raw is None:
        lo, hi, n = (kwargs.get("min_m"), kwargs.get("max_m"),
                     kwargs.get("interp_N"))
        if lo is None or hi is None or n is None:
            raise ValueError(
                f"spline_c_idx '{name}': the 'points' key is required — the "
                f"cubic spline's support points, ascending (e.g. "
                f"points: [0.3, 0.6, 0.9, 1.2]) — or give min_m, max_m and "
                f"interp_N for a uniform grid")
        lo, hi, n = float(lo), float(hi), int(n)
        if not (np.isfinite(lo) and np.isfinite(hi)) or not lo < hi:
            raise ValueError(
                f"spline_c_idx '{name}': min_m/max_m must be finite with "
                f"min_m < max_m, got [{lo}, {hi}]")
        p = np.linspace(lo, hi, n)
    else:
        try:
            p = np.asarray(raw, dtype=float).ravel()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"spline_c_idx '{name}': 'points' must be a list of numbers, "
                f"got {raw!r}") from exc
    if not np.all(np.isfinite(p)):
        bad = np.nonzero(~np.isfinite(p))[0].tolist()
        raise ValueError(
            f"spline_c_idx '{name}': 'points' must be finite; non-finite at "
            f"index {bad}")
    if p.size <= 2:
        raise ValueError(
            f"spline_c_idx '{name}': 'points' needs more than 2 entries for a "
            f"cubic spline, got N={p.size}")
    if not np.all(np.diff(p) > 0.0):
        bad = int(np.nonzero(np.diff(p) <= 0.0)[0][0])
        raise ValueError(
            f"spline_c_idx '{name}': 'points' must be strictly ascending "
            f"(duplicate or non-increasing at index {bad}: "
            f"{p[bad]:g} -> {p[bad + 1]:g})")
    return p


@register_model("spline_c_idx")
class SplineCIdx(BasisModel):
    """Complex cubic spline through ``points``, expanded in the numerator.

    TFPWA's ``spline_c_idx`` (see the module docstring for the construction,
    ``bc_type``/``with_bound``, the ``c`` = COMPLEX convention, the
    denominator ``[1]`` consequence, the gauge and the tabulation caveat).
    The row family is :meth:`_rows` (one method, solved once at construction).
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.points = _parse_points(name, kwargs)
        n = int(self.points.size)
        self.n_knots = n

        bc = str(kwargs.get("bc_type", "not-a-knot"))
        if bc == "clamped":
            raise ValueError(
                f"spline_c_idx '{name}': bc_type 'clamped' is TFPWA's "
                f"zero-slope variant and has no port in tabpwa; use "
                f"{_BC_TYPES}")
        if bc not in _BC_TYPES:
            raise ValueError(
                f"spline_c_idx '{name}': bc_type {bc!r} not in {_BC_TYPES}")
        self.bc_type = bc

        if n == 3 and bc == "not-a-knot":
            # TFPWA's two not-a-knot boundary rows coincide for three knots
            # ([-h1, h0+h1, -h0] twice), so the system is singular there.  In
            # TFPWA that surfaces as ``LinAlgError``; name the fix here.
            raise ValueError(
                f"spline_c_idx '{name}': bc_type 'not-a-knot' is singular for "
                f"N=3 (the two boundary rows coincide); use bc_type: natural, "
                f"or give more points")

        with_bound = kwargs.get("with_bound", False)
        if not isinstance(with_bound, bool):
            raise ValueError(
                f"spline_c_idx '{name}': with_bound must be a bool, got "
                f"{with_bound!r}")
        self.with_bound = with_bound
        # column omission: ``H`` is linear in the knot values, so dropping the
        # end columns IS pinning those values to 0 (TFPWA's n_free = N - 2)
        cols = np.arange(n) if self.with_bound else np.arange(1, n - 1)
        self._cols = cols
        self.n_free = int(cols.size)
        self._h = spline_basis_matrix(self.points, bc)[:, :, cols]

        outside = str(kwargs.get("outside", "spline"))
        if outside not in _OUTSIDE:
            raise ValueError(
                f"spline_c_idx '{name}': outside {outside!r} not in {_OUTSIDE}")
        self.outside = outside

        if not bool(kwargs.get("polar", True)):
            raise ValueError(
                f"spline_c_idx '{name}': polar: false has no tabpwa "
                f"counterpart — the ck slots are the (magnitude, phase) pair "
                f"(`complex_tail`); drop the key or set polar: true")

        # TFPWA's default: pin the middle knot (a DOCUMENTED index here; the
        # pin itself is the constraint layer's, see ``fixed_values``).
        # TFPWA's class default is ``-1`` and ANY negative index means "the
        # middle"; its explicit ``null`` means NO pin, which has no counterpart
        # here (the model cannot pin at all), so ``null`` keeps the default.
        fix = kwargs.get("fix_idx", None)
        if fix is None:
            fix = n // 2 - 1
        fix = int(fix)
        if fix < 0:
            fix = n // 2 - 1
        self.fix_idx = fix
        if not 0 <= self.fix_idx < self.n_free:
            raise ValueError(
                f"spline_c_idx '{name}': fix_idx={self.fix_idx} is outside the "
                f"fitted coefficient list (n_free = {self.n_free})")

        self._rows_cache = None

    # -- fitted parameters ------------------------------------------------

    def get_defaults(self):
        """No physical defaults: the ``c_i`` slots ARE the parameters.

        A coefficient is the spline's value at its knot, so it has no
        a-priori value — the fitter's random ``U(-π,π)`` start (the same rule
        ``g_ls`` gets) is the honest point.  This is what
        ``make_basis_transform() -> None`` implies.
        """
        return {}

    def fixed_values(self):
        """The ``fix_idx`` pin as slot names → values (``1 + 0i``).

        TFPWA pins ``point_value[fix_idx] = 1`` in the model.  tabpwa's
        transform-free model cannot produce a value (``make_basis_transform``
        is ``None`` by design), so this is the constraint-layer equivalent::

            f.set_free(f"{name}_point_{first}r"); f.set_fixed(mdl.fixed_values())

        The default ``fix_idx`` reproduces TFPWA's middle knot; tabpwa's own
        ``ck_redundancy`` constrain pins the first coefficient instead (same
        count, different point), which is why a caller who wants TFPWA's
        spelling must free the default pin first.
        """
        i = int(self._cols[self.fix_idx])
        return {f"{self.name}_point_{i}r": 1.0,
                f"{self.name}_point_{i}i": 0.0}

    # -- basis ------------------------------------------------------------

    def _rows(self, m):
        """The ``n_free`` REAL cardinal rows at ``m`` — THE row family."""
        x = np.asarray(m, dtype=float)
        flat = x.ravel()
        key = (x.shape, flat.tobytes(), self.outside)
        cached = self._rows_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        # interval index, clipped EXACTLY like TFPWA (``clip(idx, 0, N−2)``):
        # outside the support the end interval's cubic is evaluated
        idx = np.clip(np.searchsorted(self.points, flat, side="right") - 1,
                      0, self.points.size - 2)
        a, b, c, d = (self._h[idx, j] for j in range(4))   # (n, n_free) each
        m2 = flat[:, None]
        vals = a + m2 * (b + m2 * (c + d * m2))
        if self.outside == "zero":
            # TFPWA's `spline_c`: the half-open interval test, so the last
            # knot itself is OUTSIDE (as in `spline_x_matrix`)
            inside = (flat >= self.points[0]) & (flat < self.points[-1])
            vals = np.where(inside[:, None], vals, 0.0)
        out = vals.reshape(x.shape + (self.n_free,))
        rows = [out[..., j] for j in range(self.n_free)]
        self._rows_cache = (key, rows)
        return rows

    def num_basis(self, m):
        """``B_i(m)`` for the FITTED knots, in knot order — real rows.

        The rows are real (the cardinal cubic spline); the COMPLEX content of
        the lineshape is the per-row weight ``c_i``, so the row axis is not
        doubled into real/imag.
        """
        return [np.asarray(r, dtype=complex) for r in self._rows(m)]

    def den_basis(self, m):
        """One row, the constant 1 (the amplitude is a pure numerator sum)."""
        s = np.asarray(m, dtype=float)
        return [np.ones(s.shape, dtype=complex)]

    def get_num_name(self):
        """One complex coefficient head per FITTED knot: ``{name}_point_{i}``."""
        return [f"{self.name}_point_{i}" for i in self._cols]

    def get_den_name(self):
        """The SHARED constant ``"one"`` — fixed to 1 by the model layer.

        ``SkipModel``'s spelling: ``"one"`` is the one name the combined
        transform pins to 1 (``PWAGen.constant_weight_values``), and this
        model has no transform of its own to emit a per-name constant
        (``ExpCheb`` does, which is why it can spell ``{name}_one``).  A
        per-model name here would stay a FREE parameter and the denominator
        would be a random constant instead of 1.
        """
        return ["one"]

    def make_basis_transform(self):
        """``None`` — a coefficient IS the value at its knot.

        There is nothing to derive: ``c_i = A(points[i])`` is a physical
        number, not a function of other parameters, so the weights ARE the
        parameters (the ``BasisModel`` default contract).  A transform here
        would only re-emit its own inputs under other names.
        """
        return None