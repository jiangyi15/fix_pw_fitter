"""``kmatrix_n`` — a one-channel K-matrix with ANY number of poles.

One class, ``n_pole = len(masses)``, generalising :mod:`~tabpwa.particle_model.kmatrix2_model` and
:mod:`~tabpwa.particle_model.kmatrix3_model`::

    K(s) = Σ_{i=1..N} g_i² / (m_i² − s)
    P(s) = Σ_{i=1..N} b_i  / (m_i² − s)
    A(s) = P(s) / (1 − i·ρ(s)·K(s))

    den rows  [1, s, s², …, s^N,  −iρ, −iρs, …, −iρs^{N−1}]      (2N + 1)
    num rows  [1, s, …, s^{N−1}]                                  (N)

    real weights = coefficients of Π_i (m_i² − s)
    ρ weights    = coefficients of Σ_i g_i² Π_{j≠i} (m_j² − s)
    numerator    = coefficients of Σ_i β_i  Π_{j≠i} (m_j² − s)   (complex → ck pairs)

``ρ`` is the METHOD :meth:`KMatrixNModel._rho` — ``bwr_model.phase_space``,
``(q/m)·q^{2L}·B'²``, M-free.  It is the only knob of the family: the GS
variant is the subclass :class:`KMatrixNGSModel` (``model: kmatrix_n_gs``),
which overrides ``_rho`` with the complex unit ``gs_unit``.  Both are the
shapes the kernels use, so the one-pole limit is exact: ``BWRS`` with
``bf² = m0·Γ0/ρ(m0)``, or ``GS_rho`` with ``bf² = m0·Γ0``.

A pure :class:`~tabpwa.particle_model.base.BasisModel`: no ``gamma`` /
``get_gamma_name`` / ``make_mass_width_transform``; the plain ``pwa`` path
refuses it and ``pwa_gen`` reads the flat basis.

The pole content is a LIST of masses and widths — ``n_pole = len(masses)``::

    particle:
      f2:                    # (the particle NAME; ``rho`` is not a key here)
        model: kmatrix_n
        masses: [0.775, 1.45, 1.90]     # one entry per pole; sets n_pole
        widths: [0.149, 0.30, 0.20]     # one per pole; their fit-parameter values
        L: 1
        width_norm: true     # false: drop the /Re ρ(m_i) normalisation
        # GS variant: model: kmatrix_n_gs   (same keys, `_rho` overridden)

Each ``widths[i]`` is the width of pole *i* — the WIDTH CONVENTION is

    m·Γ(m) = m_i·Γ_i·ρ(m)/Re ρ(m_i)          (the −i coefficient; Γ(m_i) = Γ_i)

so the default coupling is ``g_i² = m_i·Γ_i/Re ρ(m_i)``: the ``−i`` already
sits in the basis row, hence ``Re`` at the pole only — ``ρ(m)`` itself stays
complex there (``Re ρ_gs`` is exactly 1, so the GS variant needs no coupling
override).  ``width_norm: false`` switches that normalisation OFF
(``g_i² = m_i·Γ_i``) for a phase space already normalised at the pole — the
declared width is then reproduced only where ``Re ρ(m_i) = 1``.
``masses: [m0]`` / ``widths: [Γ0]`` alone therefore reproduces ``BWRS`` /
``GS_rho`` with no manual coupling arithmetic.

The FITTED parameters are the pole MASS and the pole WIDTH —
``{name}_mass{i}``, ``{name}_width{i}``, ``{name}_beta{i}{mag/phase}`` — so the
numbers in ``masses`` / ``widths`` are their starting values; there is no
``g{i}`` parameter (the coupling is derived by the basis transform).
``n_pole``, ``mass`` and ``width`` remain as the scalar shorthand when no
lists are given (the reference ``M`` is then ``mass``, else ``masses[0]``).
"""
import numpy as np

from .base import BasisModel, complex_slot_names, register_model
from .bwr_model import phase_space
from .gs_rho_model import gs_unit
from tabpwa.param_constraint import Transform, complex_tail


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


def kmatrix_n_weights(m, g2, b):
    """``(den_weights, num_weights)`` for ``len(m)`` poles (real / complex).

    ``g2`` is the K-matrix coefficient itself — ``K = Σ g²/(m_i² − s)`` — not
    its square root: the ρ rows are the coefficients of
    ``Σ g_i² Π_{j≠i}(m_j² − s)``, and ``Σ g_i²`` is what a propagator
    consumes.  A width is turned into exactly this coefficient once, by the
    transform (``g² = m·Γ/Re ρ(m)``).
    """
    n = len(m)
    roots = [float(x) ** 2 for x in m]
    pm = _prod_minus_s(roots)
    k_num = sum(float(g2[i]) * _prod_minus_s(roots, i) for i in range(n))
    p_num = sum(complex(b[i]) * _prod_minus_s(roots, i) for i in range(n))
    den = [float(pm[k]) for k in range(n + 1)]
    den += [float(k_num[k]) for k in range(n)]
    return den, [complex(x) for x in p_num]


def den_names_for(name, n):
    """One name per basis row: ``n+1`` real then ``n`` ``rho`` rows.

    ``[s0, poly, m4, m6, …]`` for the real rows ``1, s, s², …`` and
    ``[rho, srho, s2rho, …]`` for ``−iρ·s^k``.  The real spelling follows
    ``kmatrix2``/``kmatrix3`` (``s0`` for ``s^0``, ``poly`` for ``s^1``, then
    ``m{2k}`` for ``s^k``), so the N=2 and N=3 lists coincide with the
    fixed-arity models' — the suffix is the ROW's index, not the weight's
    mass dimension.
    """
    real = [f"{name}_s0", f"{name}_poly"] + [f"{name}_m{2*k}" for k in range(2, n + 1)]
    rho = [f"{name}_rho", f"{name}_srho"] + [f"{name}_s{k}rho" for k in range(2, n)]
    return real + rho[:n]        # n=1 has ONE rho row


def num_names_for(name, n):
    return [f"{name}_n{i}" for i in range(n)]


class KMatrixNWeightsTransform(Transform):
    """Real slots → ``(2N+1)`` denominator + ``2N`` numerator real weights.

    The fitted per-pole parameters are the POLE MASS, the POLE WIDTH and
    the NUMERATOR coupling — ``mass1..massN``, ``width1..widthN``,
    ``beta1..betaN`` (polar: ``beta{i}{mag}``/``beta{i}{phase}``) — not the
    K-matrix coupling ``g``.  The basis rows need
    the coefficient ``g_i²`` (``K = Σ g_i²/(m_i² − s)``), so the transform
    converts each width ONCE, ``g_i² = m_i·w_i/Re ρ(m_i)`` (or ``m_i·w_i``
    under ``width_norm: false``) through the model's own
    :meth:`_coupling_sq`, and hands ``g²`` straight to
    :func:`kmatrix_n_weights`.  The numbers in a config's ``widths`` are
    therefore the starting values of the parameters.
    """

    _has_inverse = False

    def __init__(self, name, den_names, num_names, defaults, tail=None,
                 n_pole=None, coupling_sq=None):
        if coupling_sq is None:
            raise ValueError(
                "KMatrixNWeightsTransform needs the model's `coupling_sq` "
                "(width -> g²) rule")
        self.tail = tuple(tail) if tail is not None else complex_tail()
        self._coupling_sq = coupling_sq
        mag, phase = self.tail
        self.name = name
        self.n_pole = int(n_pole or len(num_names))
        keys = [f"mass{i}" for i in range(1, self.n_pole + 1)]
        keys += [f"width{i}" for i in range(1, self.n_pole + 1)]
        for i in range(1, self.n_pole + 1):
            keys += [f"beta{i}{mag}", f"beta{i}{phase}"]
        self._keys = tuple(keys)
        self.input_names = [f"{name}_{k}" for k in self._keys]
        self.output_names = (complex_slot_names(num_names, tail=self.tail)
                             + list(den_names))
        self._defaults = dict(defaults)

    def _inputs(self, d):
        d = d or {}
        return {key: float(d.get(in_name, self._defaults[in_name]))
                for key, in_name in zip(self._keys, self.input_names)}

    def forward(self, d):
        p = self._inputs(d)
        mag, phase = self.tail
        n = self.n_pole
        m = [p[f"mass{i}"] for i in range(1, n + 1)]
        w = [p[f"width{i}"] for i in range(1, n + 1)]
        g2 = np.asarray(self._coupling_sq(m, w), dtype=float)
        b = [p[f"beta{i}{mag}"] * np.exp(1j * p[f"beta{i}{phase}"])
             for i in range(1, n + 1)]
        den, num = kmatrix_n_weights(m, list(g2), b)
        vals = []
        for x in num:                       # (mag, phase) per head, interleaved
            vals += [float(np.abs(x)), float(np.angle(x))]
        vals += [float(x) for x in den]
        return {out: val for out, val in zip(self.output_names, vals)}


@register_model("kmatrix_n")
class KMatrixNModel(BasisModel):
    """One-channel K-matrix over a list of poles: ``masses`` / ``widths``.

    ``n_pole = len(masses)``; ``width_norm`` (``true`` by default) chooses
    whether a ``widths`` entry is turned into a coupling with the pole
    ``Re ρ(m_i)`` normalisation.  See the module docstring for the algebra.
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.tail = complex_tail()
        kw = self.kwargs
        masses = kw.get("masses")
        widths = kw.get("widths")
        if masses is not None:
            masses = [float(x) for x in masses]
            if not masses:
                raise ValueError(
                    f"kmatrix_n '{name}': masses must not be empty — one "
                    f"entry per pole")
            n = len(masses)
            if widths is not None and len(widths) != n:
                raise ValueError(
                    f"kmatrix_n '{name}': {len(widths)} widths for {n} masses "
                    f"— one width per pole")
            declared = kw.get("n_pole")
            if declared is not None and int(declared) != n:
                raise ValueError(
                    f"kmatrix_n '{name}': n_pole={int(declared)} but "
                    f"len(masses)={n} — the list IS the pole count")
        else:
            n = int(kw.get("n_pole", 2))
        self.n_pole = int(n)
        if self.n_pole < 1:
            raise ValueError(f"kmatrix_n '{name}': n_pole must be >= 1")
        norm = kw.get("width_norm", True)
        if not isinstance(norm, bool):
            raise ValueError(
                f"kmatrix_n '{name}': width_norm must be true (g² = m·Γ/Re ρ(m)) "
                f"or false (g² = m·Γ), got {norm!r}")
        self.width_norm = norm

    # -- physical parameters ----------------------------------------

    def _coupling(self, key):
        mag, phase = self.tail
        z = complex(self.kwargs.get(key, 1.0))
        m = abs(z)
        p = float(np.angle(z))
        m = float(self.kwargs.get(f"{key}{mag}", m))
        p = float(self.kwargs.get(f"{key}{phase}", p))
        return m, p

    def _reference_mass(self):
        """``M``: the scalar ``mass``, else the FIRST entry of ``masses``."""
        kw = self.kwargs
        if kw.get("mass") is not None:
            return float(kw["mass"])
        masses = kw.get("masses")
        if masses:
            return float(masses[0])
        return 0.775

    def _phys(self):
        mass = self._reference_mass()
        width = float(self.kwargs.get("width", 0.1))
        mag, phase = self.tail
        masses, widths = self.kwargs.get("masses"), self.kwargs.get("widths")
        out = {}
        for i in range(1, self.n_pole + 1):
            m = (float(masses[i - 1]) if masses is not None
                 and i <= len(masses) else mass * (1 + 0.1 * (i - 1)))
            gamma = (float(widths[i - 1]) if widths is not None
                     and i <= len(widths) else width / self.n_pole)
            out[f"mass{i}"] = m
            out[f"width{i}"] = gamma
        for i in range(1, self.n_pole + 1):
            m, p = self._coupling(f"beta{i}")
            out[f"beta{i}{mag}"] = m
            out[f"beta{i}{phase}"] = p
        return out

    def get_defaults(self):
        """The per-pole fit parameters — ``mass{i}`` / ``width{i}`` / ``beta{i}…``.

        The reference ``M`` and the scalar ``mass``/``width`` shorthand are
        BUILD-time (``_reference_mass``), not parameters: they feed the
        defaults above, so exposing them would add a free direction with no
        gradient.
        """
        return {f"{self.name}_{key}": val for key, val in self._phys().items()}

    # -- phase space ------------------------------------------------

    def _daughters(self):
        """``(m1, m2)`` of the decay's daughters: tree first, then config."""
        kw = self.kwargs
        parent = self._parent
        if parent is not None and getattr(parent, "_decays", None):
            return (float(parent._decays[0].outs[0].mass),
                    float(parent._decays[0].outs[1].mass))
        return (float(kw.get("daug2Mass", 0.13957039)),
                float(kw.get("daug3Mass", 0.1349768)))

    def _rho(self, m):
        """Phase space ``ρ(m)`` — THE overridable knob of this family."""
        kw = self.kwargs
        d2, d3 = self._daughters()
        return phase_space(np.asarray(m, dtype=float), int(kw.get("L", 0)),
                           float(kw.get("d", 3.0)), d2, d3)

    def _coupling_sq(self, m, gamma):
        """``g_i²`` for a pole of width ``gamma`` at mass ``m`` (arrays ok).

        THE WIDTH CONVENTION: the ``-i`` coefficient of the propagator is

            ``m·Γ(m) = m_i·Γ_i·ρ(m)/Re ρ(m_i)``

        — the FULL ``ρ(m)`` (complex for ``gs_unit``, where the imaginary
        part is the dispersive shift and the real part is the running
        width), normalised at the POLE mass by ``Re ρ(m_i)`` because the
        basis row already carries the ``-i``.  It is ``ρ(m)·g²`` for
        ``g² = m_i·Γ_i/Re ρ(m_i)``, so ``Γ(m_i) = Γ_i``; ``Re ρ_gs(m_i) = 1``
        exactly, which is why the GS subclass needs no coupling override.

        ``width_norm: false`` (``self.width_norm``) drops the division —
        ``g² = m·Γ`` — which is right for a unit ALREADY normalised at the
        pole (``Re ρ(m_i) = 1``, as ``gs_unit`` is) and is the escape hatch
        when the ``m0`` dependence of ``Re ρ(m0)`` is not wanted.  The
        declared width is then only reproduced where ``Re ρ(m_i) = 1``.

        Falls back to ``m·Γ`` when ``Re ρ(m_i)`` vanishes — a pole below
        threshold has ``Γ = 0`` and no width-preserving coupling exists.
        """
        m_arr = np.asarray(m, dtype=float)
        w_arr = np.asarray(gamma, dtype=float)
        g2 = m_arr * w_arr
        if not self.width_norm:
            return g2
        rho = np.asarray(self._rho(m_arr), dtype=complex).real
        pos = rho > 0
        return np.where(pos, g2 / np.where(pos, rho, 1.0), g2)

    # -- basis ------------------------------------------------------

    def den_basis(self, m):
        n = self.n_pole
        m_arr = np.asarray(m, dtype=float)
        s = m_arr ** 2
        rho = np.asarray(self._rho(m_arr), dtype=complex)
        rows = [(s ** k).astype(complex) for k in range(n + 1)]
        rows += [-1j * (s ** k) * rho for k in range(n)]
        return rows

    def num_basis(self, m):
        s = np.asarray(m, dtype=float) ** 2
        return [(s ** k).astype(complex) for k in range(self.n_pole)]

    def get_den_name(self):
        return den_names_for(self.name, self.n_pole)

    def get_num_name(self):
        return num_names_for(self.name, self.n_pole)

    def make_basis_transform(self):
        defaults = {f"{self.name}_{k}": v for k, v in self._phys().items()}
        return KMatrixNWeightsTransform(
            self.name, self.get_den_name(), self.get_num_name(),
            defaults, tail=self.tail, n_pole=self.n_pole,
            coupling_sq=self._coupling_sq)


@register_model("kmatrix_n_gs")
class KMatrixNGSModel(KMatrixNModel):
    """``kmatrix_n`` whose ``ρ`` is the COMPLEX GS unit (dispersive).

    ``ρ(m) = Γ(m)/g0 + i·fs(m²)/m0`` (:func:`~tabpwa.particle_model.
    gs_rho_model.gs_unit`) instead of the M-free ``phase_space`` — the same
    complex shape ``GS_rho`` uses.  With ONE pole and ``bf² = m0·Γ0``::

        d = −i·ρ·bf² = g0·fs − i·m0·Γ   ⇒   f = β / (m0² − s + g0·fs − i·m0·Γ)

    i.e. exactly the GS propagator; the extra poles are the K-matrix
    generalisation with the same dispersive phase space.

    ONLY ``_rho`` differs: the width-preserving coupling default follows it
    (``g² = m_i·Γ_i/Re ρ(m_i)``, and ``Re ρ_gs(m_i) = 1`` exactly, the unit
    already carrying ``1/M``), and the ``2N+1`` rows, the ``N`` numerator
    heads, the polynomial weights, the ``masses``/``widths`` pole list and the
    transform are the parent's.
    """

    def _rho(self, m):
        kw = self.kwargs
        d2, d3 = self._daughters()
        return gs_unit(np.asarray(m, dtype=float),
                       self._reference_mass(), int(kw.get("L", 1)),
                       float(kw.get("d", 3.0)), d2, d3)
