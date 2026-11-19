"""``kmatrix_n`` — a one-channel K-matrix with ANY number of poles.

One class, ``n_pole = len(mass_list)`` — the K-matrix family's only
implementation::

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

The pole content is a LIST — ``mass_list`` / ``width_list`` — so
``n_pole = len(mass_list)``::

    particle:
      f2:                    # (the particle NAME; ``rho`` is not a key here)
        model: kmatrix_n
        mass_list: [0.775, 1.45, 1.90]     # one entry per pole; sets n_pole
        width_list: [0.149, 0.30, 0.20]    # one per pole; their fit-parameter values
        L: 1                           # default: min(decay[0].get_ls_list())
        width_norm: true     # false: drop the /Re ρ(m_i) normalisation
        # GS variant: model: kmatrix_n_gs   (same keys, `_rho` overridden)

Each ``width_list[i]`` is the width of pole *i* — the WIDTH CONVENTION is

    m·Γ(m) = m_i·Γ_i·ρ(m)/Re ρ(m_i)          (the −i coefficient; Γ(m_i) = Γ_i)

so the default coupling is ``g_i² = m_i·Γ_i/Re ρ(m_i)``: the ``−i`` already
sits in the basis row, hence ``Re`` at the pole only — ``ρ(m)`` itself stays
complex there (``Re ρ_gs`` is exactly 1, so the GS variant needs no coupling
override).  ``width_norm: false`` switches that normalisation OFF
(``g_i² = m_i·Γ_i``) for a phase space already normalised at the pole — the
declared width is then reproduced only where ``Re ρ(m_i) = 1``.  It also makes
the basis transform's VJP ANALYTIC: every weight is then a polynomial in
``m_i²`` and ``Γ_i`` (linear in ``β``).  ``width_norm: true`` is analytic too —
its only non-polynomial piece, ``dRe ρ/dm``, is the hook :meth:`drho_dm`
(two centred ``ρ`` evaluations by default; override it with the exact
expression and the VJP sharpens with it) rather than a difference of the whole
transform — so ``KMatrixNWeightsTransform.backward`` is closed form in BOTH
modes and makes no extra ``forward`` calls.
``mass_list: [m0]`` / ``width_list: [Γ0]`` alone therefore reproduces ``BWRS`` /
``GS_rho`` with no manual coupling arithmetic.

The FITTED parameters are the pole MASS, the pole WIDTH and the beta pair —
``{name}_mass{i}``, ``{name}_width{i}``, ``{name}_beta{i}{mag/phase}`` with
``i = 0 … N−1`` for the first two and ``i = 1 … N−1`` for beta.  The
``mass_list`` / ``width_list`` numbers are the starting values of the first
two; ``beta`` has NO config key and NO default — it is a coupling, so it
starts random like ``g_ls`` and is set by ``--init`` or the constraint file.
``beta0 ≡ 1`` is the GAUGE head: the numerator's overall complex scale is
unobservable, so it is fixed and only the ``N−1`` relative weights are fitted
(with ``N = 1`` there is no numerator parameter: ``A = 1/den``).  There is no ``g{i}`` parameter (the coupling is derived by
the basis transform).
BOTH lists are required and equal length: ``n_pole`` is not a key, and there
is no scalar ``width`` shorthand — poles are DIFFERENT resonances, so there is
no total width to split.  ``mass`` (optional) is only the reference ``M`` of
the GS unit, defaulting to ``mass_list[0]``.
"""
import numpy as np

from .base import BasisModel, complex_slot_names, register_model
from .bwr_model import phase_space
from .gs_rho_model import gs_unit
from tabpwa.param_constraint import Transform, complex_tail


def _prod_minus_s(roots, skip=None):
    """Ascending coefficients of ``Π_{i ∉ skip} (roots[i] − s)``.

    ``skip`` is an index or a tuple of indices (the VJP needs the second
    derivative ``Π_{i ≠ j, i ≠ k}`` as well as the first).
    """
    skip = () if skip is None else \
        ((skip,) if isinstance(skip, (int, np.integer)) else tuple(skip))
    c = np.array([1.0])
    for i, r in enumerate(roots):
        if i in skip:
            continue
        out = np.zeros(len(c) + 1)
        out[:len(c)] += r * c
        out[1:] -= c
        c = out
    return c


def _polyval_asc(c, s):
    """Value of an ASCENDING-coefficient polynomial ``c`` at ``s``."""
    return np.polyval(np.asarray(c, dtype=complex)[::-1], s)


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
    the row index — ``s0`` for ``s^0``, ``poly`` for ``s^1``, then
    ``m{2k}`` for ``s^k`` — not the weight's mass dimension.
    """
    real = [f"{name}_s0", f"{name}_poly"] + [f"{name}_m{2*k}" for k in range(2, n + 1)]
    rho = [f"{name}_rho", f"{name}_srho"] + [f"{name}_s{k}rho" for k in range(2, n)]
    return real + rho[:n]        # n=1 has ONE rho row


def num_names_for(name, n):
    return [f"{name}_n{i}" for i in range(n)]


class KMatrixNWeightsTransform(Transform):
    """Real slots → ``(2N+1)`` denominator + ``2N`` numerator real weights.

    The fitted per-pole parameters are the POLE MASS, the POLE WIDTH and
    the NUMERATOR coupling — ``mass0..mass{N−1}``, ``width0..width{N−1}``,
    ``beta1..beta{N−1}`` (polar: ``beta{i}{mag}``/``beta{i}{phase}``) — not the
    K-matrix coupling ``g``.

    ``beta0`` is a FIXED head, ``1 + 0i``: the numerator's overall complex
    scale is unobservable (``P → |c|²P`` and ``norm → |c|²norm``, so ``P/norm``
    and the NLL are unchanged), so exposing it would only add two flat
    directions.  What remains is ``N−1`` relative complex weights, i.e. the
    same freedom as ``N`` Breit-Wigners with one overall reference fixed —
    and at ``N = 1`` there is no numerator parameter at all (the amplitude is
    ``1/den``, exactly ``BWRS`` / ``GS_rho``).  The basis rows need
    the coefficient ``g_i²`` (``K = Σ g_i²/(m_i² − s)``), so the transform
    converts each width ONCE, ``g_i² = m_i·w_i/Re ρ(m_i)`` (or ``m_i·w_i``
    under ``width_norm: false``) through the model's own
    :meth:`_coupling_sq`, and hands ``g²`` straight to
    :func:`kmatrix_n_weights`.  The numbers in a config's ``width_list`` are
    therefore the starting values of the parameters.
    """

    _has_inverse = False

    def __init__(self, name, den_names, num_names, defaults, tail=None,
                 n_pole=None, coupling_sq=None, coupling_jac=None):
        if coupling_sq is None:
            raise ValueError(
                "KMatrixNWeightsTransform needs the model's `coupling_sq` "
                "(width -> g²) rule")
        self.tail = tuple(tail) if tail is not None else complex_tail()
        self._coupling_sq = coupling_sq
        self._coupling_jac = coupling_jac
        mag, phase = self.tail
        self.name = name
        self.n_pole = int(n_pole or len(num_names))
        keys = [f"mass{i}" for i in range(self.n_pole)]
        keys += [f"width{i}" for i in range(self.n_pole)]
        # beta0 is the GAUGE head (fixed to 1): the numerator's overall scale
        # is unobservable (`P/norm` is invariant under it), so it is not a
        # parameter — only the N-1 relative weights are.
        for i in range(1, self.n_pole):
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
        m = [p[f"mass{i}"] for i in range(n)]
        w = [p[f"width{i}"] for i in range(n)]
        g2 = np.asarray(self._coupling_sq(m, w), dtype=float)
        b = [1.0 + 0.0j]                     # beta0: the fixed gauge head
        b += [p[f"beta{i}{mag}"] * np.exp(1j * p[f"beta{i}{phase}"])
              for i in range(1, n)]
        den, num = kmatrix_n_weights(m, list(g2), b)
        vals = []
        for x in num:                       # (mag, phase) per head, interleaved
            vals += [float(np.abs(x)), float(np.angle(x))]
        vals += [float(x) for x in den]
        return {out: val for out, val in zip(self.output_names, vals)}

    def backward(self, grad_out, d_in=None):
        """VJP of :meth:`forward` — ANALYTIC for a linear width rule.

        ``width_norm: false`` gives ``g_i² = m_i·Γ_i``, so every weight is a
        polynomial in ``m_i²`` and ``Γ_i`` (and linear in ``β``) and the
        adjoint below is exact::

            w(s) = Π_i (m_i² − s)                 ∂w/∂m_j = 2m_j Π_{l≠j}
            u(s) = Σ_i g2_i Π_{l≠i}               ∂u/∂Γ_j = (∂g2_j/∂Γ_j) Π_{l≠j}
                                                  ∂u/∂m_j = (∂g2_j/∂m_j) Π_{l≠j}
                                                            + 2m_j Σ_{i≠j} g2_i Π_{l≠i,j}
            n(s) = Σ_i β_i Π_{l≠i}                ∂n/∂m_j = 2m_j Σ_{i≠j} β_i Π_{l≠i,j}

        with ``Π_{l≠j} = _prod_minus_s(m², j)``.  The ``(magnitude, phase)``
        numerator slots are adjointed in the polar chart (a head with
        ``|n| = 0`` has no phase and contributes nothing).

        ``width_norm: true`` divides by ``Re ρ(m_i)``.  The only derivative
        that is not closed form there, ``dRe ρ/dm``, comes from two centred
        evaluations of ``ρ`` (``_coupling_sq_jac``), so this VJP is analytic in
        BOTH modes; a model that cannot supply partials at all still falls back
        to the inherited finite-difference VJP.
        """
        d = self._last_input if d_in is None else d_in
        p = self._inputs(d)
        n = self.n_pole
        mass = [p[f"mass{i}"] for i in range(n)]
        width = [p[f"width{i}"] for i in range(n)]
        g2, dg2_dm, dg2_dg = self._coupling_jac(mass, width)
        if dg2_dm is None:                     # non-polynomial rule -> FD
            return self.fd_backward(grad_out, d_in=d)

        mag, phase = self.tail
        beta = [1.0 + 0.0j]                    # beta0: the fixed gauge head
        beta += [p[f"beta{i}{mag}"] * np.exp(1j * p[f"beta{i}{phase}"])
                 for i in range(1, n)]

        g2 = np.asarray(g2, dtype=float)
        dg2_dm = np.asarray(dg2_dm, dtype=float)
        dg2_dg = np.asarray(dg2_dg, dtype=float)
        roots = [float(x) ** 2 for x in mass]
        P = [_prod_minus_s(roots, j) for j in range(n)]
        pair = {}

        def P2(i, j):
            """``Π_{l ≠ i, j}`` (the second-derivative factor)."""
            if (i, j) not in pair:
                pair[(i, j)] = _prod_minus_s(roots, (i, j))
            return pair[(i, j)]

        gm = np.zeros(n)
        gG = np.zeros(n)
        num_off = 2 * n                        # the numerator slots come first
        den_off = num_off
        rho_off = den_off + (n + 1)

        # real rows: w = Π_i (m_i² − s); ∂w/∂m_j = 2m_j Π_{l≠j} has degree
        # n − 1, so the s^n coefficient (k = n) is mass-independent
        for k in range(n):
            cw = float(grad_out.get(self.output_names[den_off + k], 0.0))
            if cw:
                for j in range(n):
                    gm[j] += cw * 2.0 * mass[j] * P[j][k]

        # rho rows: u = Σ_i g2_i Π_{l≠i}
        for k in range(n):
            cu = float(grad_out.get(self.output_names[rho_off + k], 0.0))
            if not cu:
                continue
            for j in range(n):
                # the cross term has degree n − 2, so it vanishes for k = n−1
                second = (sum(g2[i] * P2(i, j)[k] for i in range(n) if i != j)
                          if k <= n - 2 else 0.0)
                gm[j] += cu * (dg2_dm[j] * P[j][k]
                               + 2.0 * mass[j] * second)
                gG[j] += cu * dg2_dg[j] * P[j][k]

        # numerator heads: n_k = Σ_i β_i Π_{l≠i}, emitted as (|n_k|, arg n_k)
        nk = [sum(beta[i] * P[i][k] for i in range(n)) for k in range(n)]
        q = []
        for k in range(n):
            cm = float(grad_out.get(self.output_names[2 * k], 0.0))
            cp = float(grad_out.get(self.output_names[2 * k + 1], 0.0))
            a = abs(nk[k])
            if a == 0.0:
                q.append(0j)                   # no phase at a zero head
            else:
                q.append(cm * nk[k] / a + 1j * cp * nk[k] / (a * a))

        for k in range(n):
            if q[k] == 0:
                continue
            for j in range(n):
                if k > n - 2:
                    continue            # ∂n/∂m has degree n − 2
                second = sum(beta[i] * P2(i, j)[k]
                             for i in range(n) if i != j)
                gm[j] += (2.0 * mass[j] * second * q[k].conjugate()).real

        out = {}
        for j in range(n):
            out[f"mass{j}"] = float(gm[j])
            out[f"width{j}"] = float(gG[j])
        for j in range(1, n):
            Q = sum(q[k].conjugate() * P[j][k] for k in range(n))
            out[f"beta{j}{mag}"] = float((Q * np.exp(
                1j * p[f"beta{j}{phase}"])).real)
            out[f"beta{j}{phase}"] = float((1j * beta[j] * Q).real)
        return {name: out[key] for name, key in
                zip(self.input_names, self._keys)}


@register_model("kmatrix_n")
class KMatrixNModel(BasisModel):
    """One-channel K-matrix over a list of poles: ``mass_list`` / ``width_list``.

    ``n_pole = len(mass_list)``; ``width_norm`` (``true`` by default) chooses
    whether a ``width_list`` entry is turned into a coupling with the pole
    ``Re ρ(m_i)`` normalisation, and ``L`` defaults to
    :meth:`BasisModel.default_L` — the lowest partial wave of the particle's
    own decay.  See the module docstring for the algebra.
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.tail = complex_tail()
        kw = self.kwargs
        masses = kw.get("mass_list")
        widths = kw.get("width_list")
        if masses is None or widths is None:
            missing = [k for k, v in (("mass_list", masses),
                                      ("width_list", widths)) if v is None]
            raise ValueError(
                f"kmatrix_n '{name}': declare the poles with {missing} — "
                f"`mass_list: [...]` AND `width_list: [...]`, one entry per "
                f"pole (there is no scalar shorthand)")
        masses = [float(x) for x in masses]
        widths = [float(x) for x in widths]
        if not masses:
            raise ValueError(
                f"kmatrix_n '{name}': mass_list must not be empty — one entry "
                f"per pole")
        if len(widths) != len(masses):
            raise ValueError(
                f"kmatrix_n '{name}': {len(widths)} widths for {len(masses)} "
                f"masses — one width per pole")
        if "n_pole" in kw:
            raise ValueError(
                f"kmatrix_n '{name}': `n_pole` is not a key — the pole list "
                f"IS the pole count (len(mass_list) = {len(masses)})")
        self.n_pole = len(masses)
        norm = kw.get("width_norm", True)
        if not isinstance(norm, bool):
            raise ValueError(
                f"kmatrix_n '{name}': width_norm must be true (g² = m·Γ/Re ρ(m)) "
                f"or false (g² = m·Γ), got {norm!r}")
        self.width_norm = norm

    # -- physical parameters ----------------------------------------

    def _reference_mass(self):
        """``M``: the scalar ``mass``, else the FIRST entry of ``mass_list``."""
        kw = self.kwargs
        if kw.get("mass") is not None:
            return float(kw["mass"])
        masses = kw.get("mass_list")
        if masses:
            return float(masses[0])
        return 0.775

    def _pole_values(self):
        """``{mass{i}: m_i, width{i}: Γ_i}`` for ``i = 0 … N−1``."""
        masses, widths = self.kwargs["mass_list"], self.kwargs["width_list"]
        out = {}
        for i in range(self.n_pole):
            out[f"mass{i}"] = float(masses[i])
            out[f"width{i}"] = float(widths[i])
        return out

    def _transform_defaults(self):
        """Pole anchors + the β fallback.

        The β entries only fill a dict that LACKS them (a partial ``amp``
        call); in a fit the β slots are flat variables, so their value comes
        from ``x``.
        """
        out = self._pole_values()
        mag, phase = self.tail
        for i in range(1, self.n_pole):      # beta0 is the fixed gauge head
            out[f"beta{i}{mag}"] = 1.0
            out[f"beta{i}{phase}"] = 0.0
        return out

    def get_defaults(self):
        """The POLE anchors — ``mass{i}`` / ``width{i}``, ``i = 0 … N−1``.

        Only these enter ``cm.defaults``, and only they therefore start at
        their configured value in ``initial_values``: a pole mass/width is
        physics (narrow basin, label-sensitive — a random mass is nonsense).

        ``beta{i}`` (``i ≥ 1``) is deliberately absent: it is a COUPLING,
        linear in the amplitude and zero at the origin, so — like ``g_ls`` —
        it starts random (``U(-π, π)``) and is set by ``--init`` / the
        constraint file.  There is no ``beta{i}`` config key either, and
        ``beta0`` is not a parameter at all: it is the gauge head, fixed to
        ``1`` by the transform.
        """
        return {f"{self.name}_{k}": v for k, v in self._pole_values().items()}

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

    def _rho(self, m, sheet=1):
        """Phase space ``ρ(m)`` — THE overridable knob of this family.

        ``sheet`` is the SIGN OF THE BREAKUP-MOMENTUM SQUARE ROOT (int ``1``
        principal, ``−1`` the other branch): that is the whole of a Riemann
        sheet here, and ``ρ`` is odd in ``q``, so ``ρ(m, −1) = −ρ(m, 1)``.
        A COMPLEX ``m`` is the analytic continuation (the pole search
        evaluates this at a complex mass); a real ``m`` keeps the exact
        previous path, including the below-threshold clamp.
        """
        kw = self.kwargs
        d2, d3 = self._daughters()
        m_arr = np.asarray(m)
        if not np.iscomplexobj(m_arr):
            m_arr = m_arr.astype(float)
        return phase_space(m_arr,
                           int(kw.get("L", self.default_L())),
                           float(kw.get("d", 3.0)), d2, d3, sheet)

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

    #: Relative step of the DEFAULT ``dRe ρ/dm`` difference.
    #:
    #: Chosen from measured error against a Richardson-extrapolated truth
    #: (relative, worst over ``phase_space`` L=0/1 at m = 0.30 (near
    #: threshold), 0.775, 2.0 and ``gs_unit`` at 0.775)::
    #:
    #:     eps      fp64 rho        fp32-valued rho
    #:     1e-3     1.2e-4          1.2e-4
    #:     1e-4     1.2e-6          8.9e-4
    #:     1e-5     1.2e-8          5.7e-3      <- default
    #:     1e-6     1.5e-10         2.4e-2
    #:     1e-7     4.6e-9          1.0
    #:
    #: Our ``_rho`` is fp64, where 1e-6 is the optimum and 1e-5 is within
    #: 1.2e-8; 1e-7 is already past the round-off knee.  A ρ that is evaluated
    #: or tabulated in low precision moves the knee out by ~100x — set
    #: ``_RHO_FD_EPS = 1e-3`` there, or override :meth:`drho_dm` exactly.
    _RHO_FD_EPS = 1e-5

    def drho_dm(self, m):
        """``d Re ρ / dm`` — the ONE non-polynomial derivative in the model.

        Default: a centred difference of :meth:`_rho` (two cheap scalar
        evaluations, no ``forward`` of the basis transform).  This is the
        single hook for an exact expression — override it in a subclass or on
        an instance and the whole VJP sharpens with it, nothing else changes::

            model.drho_dm = lambda m: exact_drho(m)     # exact

        ``Re`` is taken BEFORE differencing: the coupling rule uses
        ``Re ρ(m)`` only, so the imaginary part carries no derivative.
        """
        m_arr = np.asarray(m, dtype=float)
        h = self._RHO_FD_EPS * np.maximum(1.0, np.abs(m_arr))
        rho_p = np.asarray(self._rho(m_arr + h), dtype=complex).real
        rho_m = np.asarray(self._rho(m_arr - h), dtype=complex).real
        return (rho_p - rho_m) / (2.0 * h)

    def _coupling_sq_jac(self, m, gamma):
        """``(g², ∂g²/∂m, ∂g²/∂Γ)`` — closed form apart from ``drho_dm``.

        ``width_norm: false`` (``g² = m·Γ``) is linear in both, exact.

        ``width_norm: true`` (``g² = m·Γ/Re ρ(m)``) is analytic in ``Γ``
        (``∂g²/∂Γ = m/Re ρ``); the mass slot needs ``dRe ρ/dm``, which comes
        from :meth:`drho_dm` (centred difference by default, exact if
        overridden).  Below threshold (``Re ρ ≤ 0``) the rule falls back to
        ``m·Γ`` and so do the partials.
        """
        m_arr = np.asarray(m, dtype=float)
        w_arr = np.asarray(gamma, dtype=float)
        g2 = self._coupling_sq(m_arr, w_arr)
        if not self.width_norm:
            return g2, w_arr, m_arr
        rho = np.asarray(self._rho(m_arr), dtype=complex).real
        pos = rho > 0
        drho = np.asarray(self.drho_dm(m_arr), dtype=float)
        safe = np.where(pos, rho, 1.0)
        dg2_dg = np.where(pos, m_arr / safe, m_arr)
        dg2_dm = np.where(pos, w_arr / safe - m_arr * w_arr * drho / safe ** 2,
                          w_arr)
        return g2, dg2_dm, dg2_dg

    # -- pole structure ---------------------------------------------

    @staticmethod
    def _sheet_sign(sheet):
        """``+1`` / ``−1`` — the phase-space square-root sign; an INT only."""
        if (isinstance(sheet, bool) or isinstance(sheet, float)
                or not isinstance(sheet, (int, np.integer))
                or sheet not in (1, -1)):
            raise ValueError(
                f"sheet must be the int 1 or −1 (the sign of the phase-space "
                f"square root q = ±sqrt(p²)/(2m)), got {sheet!r}")
        return int(sheet)

    def _pole_lists(self):
        """``(mass_list, width_list)`` as floats — the model's own values."""
        return ([float(x) for x in self.kwargs["mass_list"]],
                [float(x) for x in self.kwargs["width_list"]])

    def _den_polys(self):
        """``(P, Q)``: ascending coefficients of the two real polynomials.

        ``P(s) = Π_i (m_i² − s)`` (the real-row weights) and
        ``Q(s) = Σ_i g_i² Π_{j≠i} (m_j² − s)`` (the ``−iρ`` row weights),
        with ``g_i²`` from the model's own width rule — the same numbers the
        basis transform publishes, so the pole is the one the amplitude has.
        """
        mass, width = self._pole_lists()
        roots = [x * x for x in mass]
        g2 = np.asarray(self._coupling_sq(np.asarray(mass, dtype=float),
                                          np.asarray(width, dtype=float)),
                        dtype=float)
        P = _prod_minus_s(roots)
        Q = np.zeros(len(roots))
        for i in range(len(roots)):
            Q += g2[i] * _prod_minus_s(roots, i)
        return P.astype(complex), Q.astype(complex)

    def den(self, m, sheet=1):
        """``D(m) = P(m²) − i·ρ(m; sheet)·Q(m²)`` at a complex MASS ``m``.

        ``D = 0`` is the pole condition: it is the amplitude's denominator
        with the numerator cancelled (``β`` normalises the numerator, so it
        cannot move a zero).  ``m`` may be an array and a complex ``m`` takes
        the analytic continuation of ``ρ``.

        ``sheet`` is the int ``1`` / ``−1`` and it is nothing but the SIGN OF
        THE SQUARE ROOT in the phase space (``q = ±sqrt(p²)/(2m)``) — the
        branch that decides which Riemann sheet the amplitude sits on.  It
        enters through ``ρ`` alone, so a multi-channel model can put each
        channel on its own branch; for this one-channel family the two
        branches differ by ``q → −q``.
        """
        P, Q = self._den_polys()
        return self._den_value(m, P, Q, self._sheet_sign(sheet))

    def _den_value(self, m, P, Q, sheet):
        m_arr = np.asarray(m)
        if not np.iscomplexobj(m_arr):
            m_arr = m_arr.astype(float)
        s = m_arr ** 2
        rho = np.asarray(self._rho(m_arr, sheet))
        return _polyval_asc(P, s) - 1j * rho * _polyval_asc(Q, s)

    def _den_deriv(self, m, P, Q, sheet, eps=1e-6):
        """``dD/dm`` at a complex mass (``dρ/dm`` by a centred difference).

        ``ρ`` is the only non-polynomial piece; its derivative is a two-point
        difference in ``m`` — the pole positions it feeds are converged to
        ~1e-9, far below any physics scale, and the alternative (a complex
        step) would silently be wrong for a ``ρ`` that is not analytic.
        """
        m = complex(m)
        h = eps * max(1.0, abs(m))
        one = np.asarray([m])
        rho = complex(np.asarray(self._rho(one, sheet)).reshape(-1)[0])
        rho_p = complex(np.asarray(self._rho(one + h, sheet)).reshape(-1)[0])
        rho_m = complex(np.asarray(self._rho(one - h, sheet)).reshape(-1)[0])
        drho = (rho_p - rho_m) / (2.0 * h)
        s = m * m
        dP = np.polyder(np.asarray(P, dtype=complex)[::-1])
        dQ = np.polyder(np.asarray(Q, dtype=complex)[::-1])
        # d/ds = (1/2m)·d/dm for the ρ term, and P(m²), Q(m²) chain by 2m
        return (2.0 * m * np.polyval(dP, s)
                - 1j * (drho * _polyval_asc(Q, s)
                        + rho * 2.0 * m * np.polyval(dQ, s)))

    def solve_pole(self, inits=None, sheet=1, tol=1e-9, maxiter=100):
        """Complex pole positions ``m = √s`` where ``den(m) == 0``.

        A K-matrix pole is NOT the peak of a Breit-Wigner: ``get_bw_params``
        (a real running-width peak) does not exist for this model, and the
        pole is the zero of ``D(m) = P(m²) − i·sheet·ρ(m)·Q(m²)`` in the
        complex mass plane — the analytic continuation of the very ``ρ`` the
        amplitude uses.

        Args:
            inits: complex MASSES to start Newton from (``inits=[0.775-0.07j]``);
                ``None`` starts from each configured ``mass_list`` entry, which
                is the zero-width location of its pole.
            sheet: int ``1`` / ``−1`` — the sign of the phase-space square
                root, i.e. which branch of ``q = ±sqrt(p²)/(2m)`` (the Riemann
                sheet) the pole is searched on.
            tol: relative step at which the iteration stops (~1e-9: the
                centred-difference ``dρ/dm`` sets the floor).
            maxiter: Newton iterations per init.

        Returns:
            one dict per init, in init order::

                {"m": complex mass √s, "s": m²,
                 "mass": Re m, "width": −2·Im m,
                 "residual": |D(m)|, "converged": bool, "n_iter": int}

        ``width = −2·Im √s``, i.e. the usual ``√s = M − iΓ/2`` convention: a
        physical resonance has ``Im √s < 0`` (``sheet`` chooses the branch of
        the phase-space square root, not the sign of ``√s`` itself).  Each iteration is capped at
        half the current ``|m|`` so a Newton step cannot cross to another
        pole's basin; ``residual`` is the honest check on the answer.
        """
        sign = self._sheet_sign(sheet)
        P, Q = self._den_polys()
        if inits is None:
            starts = list(self._pole_lists()[0])
        else:
            starts = [complex(x) for x in np.atleast_1d(inits).tolist()]
        out = []
        for m0 in starts:
            m = complex(m0)
            n_iter, converged = 0, False
            for n_iter in range(1, int(maxiter) + 1):
                D = complex(np.asarray(
                    self._den_value(m, P, Q, sign)).reshape(-1)[0])
                if not np.isfinite(D):
                    break
                dD = self._den_deriv(m, P, Q, sign)
                if not np.isfinite(dD) or dD == 0:
                    break
                step = D / dD
                cap = 0.5 * max(abs(m), 1e-12)
                if abs(step) > cap:
                    step *= cap / abs(step)
                m -= step
                if abs(step) <= tol * max(1.0, abs(m)):
                    converged = True
                    break
            resid = float(abs(complex(
                np.asarray(self._den_value(m, P, Q, sign)).reshape(-1)[0])))
            out.append({"m": complex(m), "s": complex(m * m),
                        "mass": float(m.real), "width": float(-2.0 * m.imag),
                        "residual": resid, "converged": bool(converged),
                        "n_iter": int(n_iter)})
        return out

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
        defaults = {f"{self.name}_{k}": v
                    for k, v in self._transform_defaults().items()}
        return KMatrixNWeightsTransform(
            self.name, self.get_den_name(), self.get_num_name(),
            defaults, tail=self.tail, n_pole=self.n_pole,
            coupling_sq=self._coupling_sq,
            coupling_jac=self._coupling_sq_jac)


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
    heads, the polynomial weights, the ``mass_list``/``width_list`` pole list and the
    transform are the parent's.
    """

    _L_FALLBACK = 1

    def _rho(self, m, sheet=1):
        """``gs_unit`` at ``m`` — ``sheet`` is the sign of its square roots."""
        kw = self.kwargs
        d2, d3 = self._daughters()
        m_arr = np.asarray(m)
        if not np.iscomplexobj(m_arr):
            m_arr = m_arr.astype(float)
        return gs_unit(m_arr,
                       self._reference_mass(),
                       int(kw.get("L", self.default_L())),
                       float(kw.get("d", 3.0)), d2, d3, sheet)
