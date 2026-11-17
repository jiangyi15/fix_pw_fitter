"""Base classes and registration machinery for particle models."""

import warnings

import numpy as np
from tabpwa.param_constraint import Transform, complex_tail

ALL_MODELS = {}


def register_model(name):
    """Decorator: register a particle model class under `name`."""
    def _f(cls):
        ALL_MODELS[name] = cls
        return cls
    return _f


def build_particle(name, **kwargs):
    """Factory: instantiate a particle model by name.

    The ``model`` key in *kwargs* selects the registered class
    (default ``"BW"``).  Remaining kwargs are passed to the
    constructor.
    """
    from ..build_defaults import get as _build_get
    # default Blatt-Weisskopf radius from the global build context (models
    # that do not read ``d`` simply ignore it)
    kwargs.setdefault("d", _build_get("d", 3.0))
    model = kwargs.pop("model", "BW")
    return ALL_MODELS[model](name, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# Exponential-lineshape helpers
# ═══════════════════════════════════════════════════════════════════

def gamma_from_amplitude(m, m0, A, g0, pure_exp):
    """Running-width gamma rows for the BW form from a physics amplitude.

    The k-interpolated family (``InterpKModel``/``SplineKModel``) feeds
    gamma rows into the kernel denominator ``A = 1/(m_0^2-m^2-i*m_0*γ)``
    with couplings = the interpolation weights (Σ = 1).  Given the
    physics amplitude ``A(m)`` at a parameter point, the row that makes
    the BW reproduce it is the total running width::

        gamma = (m_0^2 - m^2 - 1/A) / (i*m_0)          [pure_exp=True]

    The legacy form (``pure_exp=False``, deprecated) kept a reference
    width in the denominator — only exact at width = 1::

        gamma = (m_0^2 - m^2 - 1/A) / (i*m_0*g_0)      [legacy]

    Parameters
    ----------
    m : ndarray
        Invariant mass grid.
    m0 : float
        Nominal mass.
    A : ndarray
        Physics amplitude (complex) at *m*.
    g0 : float
        Reference width (only used when ``pure_exp=False``).
    pure_exp : bool
        True (default): exact amplitude for any width.  False: legacy.
    """
    num = m0 ** 2 - m ** 2 - 1.0 / np.asarray(A)
    if pure_exp:
        return num / (1j * m0)
    return num / (1j * m0 * g0)


def warn_exp_non_pure(model, g0, pure):
    """Warn once per model when the ExpSpline uses width != 1 without
    ``pure_exp: true``.

    ExpSpline defaults to the legacy width-normalised gamma
    (``pure_exp`` unset = false, for backward compatibility), where a
    non-unit width reshapes the amplitude tail and the amplitude is not
    the documented ``A = exp(-k*(m^2-m_0^2))``.  Setting
    ``pure_exp: true`` gives the exact exponential for any width.
    Warns once per model instance.
    """
    if abs(float(g0) - 1.0) < 1e-9 or pure:
        return
    if getattr(model, "_pure_exp_warned", False):
        return
    model._pure_exp_warned = True
    warnings.warn(
        f"particle model '{model.name}' uses an exponential lineshape with "
        f"width={g0} != 1 but 'pure_exp: true' is not set: the amplitude is "
        r"NOT exp(-k(m^2-m_0^2))" f" (the width reshapes the tail). Set "
        f"'pure_exp: true' in the particle config for the "
        f"pure-exponential amplitude. NOTE: 'pure_exp: true' will become "
        f"the default in the next version.", UserWarning)


class BaseModel:
    """Base class for all particle resonance models.

    Subclasses must implement:
        gamma(self, m) -> list[ndarray]

    Subclasses may override:
        get_defaults() -> dict[str, float]  (all physical defaults)
        get_gamma_count() -> int            (default 1)
        get_gamma_name() -> list[str]       (default [`{name}_width`])
    """

    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self._parent = None

    def get_defaults(self):
        """All physical default values for this model (mass + gamma/width).

        Returns a dict mapping full parameter names to floats, e.g.::

            {"rhoA_mass": 0.775, "rhoA_width": 0.149}
            {"f0(980)_g0": 0.1, "f0(980)_g1": 0.1, "f0(980)_mass": 0.99}
        """
        mass = float(self.kwargs.get("mass", 0.775))
        width = float(self.kwargs.get("width", 0.1))
        return {f"{self.name}_mass": mass, f"{self.name}_width": width}

    def get_gamma_count(self) -> int:
        return 1

    def get_gamma_name(self):
        return [f"{self.name}_width"]

    def register_parent(self, particle):
        """Store the Particle that owns this model (for decay tree access)."""
        self._parent = particle

    def gamma(self, m):
        """Compute the gamma (width) function at masses *m*.

        Returns a list of complex ndarrays, one per gamma parameter.
        """
        return [np.ones_like(m) + 0j]

    def amplitude_raw(self, m, params):
        """Raw amplitude ``Σ n_i N_i / Σ g_j G_j`` (no transform applied).

        This is ONE implementation for every model: each resonance is a ratio
        of two fixed bases (:meth:`num_basis` / :meth:`den_basis`), and the
        weights are read by :meth:`_num_weights` / :meth:`_den_weights`.
        With the default basis that is exactly the BW denominator::

            A(m) = (Σ_i n_i·N_i(m)) / (m₀² - m² - i·m₀·Σ_j g_j·γ_j(m))

        so a plain model, a ``BasisModel`` and ``pwa_gen``'s kernel all share
        the same algebra instead of three copies that can drift.

        Args:
            m: mass array.
            params: dict of parameter names → values.

        Returns:
            Complex amplitude ``A(m)``.
        """
        m_arr = np.asarray(m, dtype=float)
        params = {} if params is None else params

        num = np.zeros_like(m_arr, dtype=complex)
        for val, row in zip(self._num_weights(params), self.num_basis(m_arr)):
            num = num + val * np.asarray(row, dtype=complex)

        den = np.zeros_like(m_arr, dtype=complex)
        for val, row in zip(self._den_weights(params), self.den_basis(m_arr)):
            den = den + val * np.asarray(row, dtype=complex)

        return num / den

    def amplitude(self, m, params=None):
        """Full amplitude A(m) with transform applied.

        Applies :meth:`make_mass_width_transform` to *params* first,
        so fixed values are filled in automatically.  ``amplitude(m, {})``
        works for models with transforms.

        For models without a transform, ``amplitude(m, params)`` is
        equivalent to ``amplitude_raw(m, params)``.

        Args:
            m: mass array.
            params: optional dict (``None`` / ``{}`` for defaults).

        Returns:
            Complex amplitude A(m).
        """
        if params is None:
            params = {}
        tr = self.make_mass_width_transform()
        if tr is not None:
            resolved = tr.apply_forward(dict(params))
        else:
            resolved = dict(params)
        return self.amplitude_raw(m, resolved)

    def make_mass_width_transform(self) -> Transform | None:
        """Create a Transform from physical parameters to real mass/width.

        Returns a :class:`~tabpwa.param_constraint.Transform` that reads
        physical mass and gamma values from the full parameter dict and
        replaces them with the model's real (derived) mass and width.

        The default returns ``None`` (physical = real, no transform).
        Override in subclasses with running widths (GS_rho, Bugg, etc.)
        to return a ``Transform`` with the appropriate computation.
        """
        return None

    # -- basis ------------------------------------------------------

    def num_basis(self, m):
        """Numerator basis rows ``N_i(m)`` (complex, parameter-free).

        Default: a single constant row.  When overriding with more than
        one row, also override :meth:`get_num_name` with matching names.
        """
        m_arr = np.asarray(m, dtype=float)
        return [np.ones_like(m_arr, dtype=complex)]

    def den_basis(self, m):
        """Denominator basis rows ``G_j(m)`` (complex, parameter-free).

        Default: the kernel's uniform coefficient basis::

            [ 1 , m² , −i·γ_1(m) , … , −i·γ_k(m) ]

        with the gamma rows from :meth:`gamma`.  Combined with the default
        weights ``[m0², −1, m0·g0_1, …]`` (see :meth:`get_den_name` and
        :meth:`_den_weights`) the sum is exactly the kernel denominator
        ``m0² − m² − i·m0·Σ_l g0_l·γ_l(m)``.
        """
        m_arr = np.asarray(m, dtype=float)
        # ``-m²`` here, so its weight is exactly the SHARED ``"one"`` = +1 and
        # the name reads as its value.  The pair gives the plain BW
        # denominator ``m0² - m² - i·m0·Σ g_j γ_j``.  Do NOT flip one side
        # without the other — that is what silently reversed every m² term.
        rows = [np.ones_like(m_arr, dtype=complex), -(m_arr ** 2) + 0j]
        rows.extend(-1j * g for g in self.gamma(m_arr))
        return rows

    # -- weight names ----------------------------------------------

    def get_num_name(self):
        """Head (base) names for the numerator rows ``n_i``.

        Default ``[]``: no free numerator weights, ``n = [1]``.  If
        non-empty, the names must match :meth:`num_basis` length.  These
        are *heads*: a complex weight is expanded by the consumer to the
        two real ``(magnitude, phase)`` slots named
        :meth:`num_slot_names` (the ``ck`` convention); a real weight may
        be given as one parameter named exactly the head.
        """
        return []

    def num_slot_names(self):
        """Expanded real ``(magnitude, phase)`` slot names for the heads.

        Uses :func:`complex_slot_names` with the active
        :func:`~tabpwa.param_constraint.complex_tail`; empty when
        :meth:`get_num_name` is empty.
        """
        return complex_slot_names(self.get_num_name())

    def get_den_name(self):
        """Weight names for the denominator rows ``g_j``.

        ``[f"{name}_s0", "one"] + get_gamma_name()`` — the kernel ``coeff``
        layout: the ``m0²`` slot, the m² coefficient, and the per-gamma
        ``mult_j`` weights.

        The m² weight is the SHARED literal ``"one"``, not ``{name}_poly``.
        Two reasons:

        * its value is the constant ``−1`` (the row is ``+m²``), identical for
          every resonance — naming it per resonance would make one physical
          constant look like N free parameters;
        * ``poly`` is ``kmatrix2``'s word for a *fitted* polynomial
          coefficient ``−(m1²+m2²)``, a different quantity with a different
          free/fixed status.  Reusing the name for a fixed ``−1`` conflated
          the two.
        """
        return [f"{self.name}_s0", "one"] + list(self.get_gamma_name())

    # -- weight extraction -----------------------------------------

    def _num_weights(self, params):
        """Numerator weights, with the documented missing-value defaults."""
        names = list(self.get_num_name())
        if not names:
            # no free numerator weights: n = [1] (one constant row)
            return [1.0]
        return [params.get(name, 0.0) for name in names]

    def _den_weights(self, params):
        """Denominator weights, derived as the kernel ``coeff`` vector.

        Layout of the default :meth:`get_den_name`:

        0. ``s0``  → ``params[s0_name]`` if given, else ``m0²``;
        1. ``poly`` → ``−1`` (fixed; the m² coefficient);
        2. gamma rows → ``m0 · params[g0_name]`` (the kernel ``mult_l``).

        Subclasses overriding :meth:`get_den_name` should override this
        too when their weight semantics differ.
        """
        m0 = float(params.get(f"{self.name}_mass",
                              self.kwargs.get("mass", 0.775)))
        names = list(self.get_den_name())
        gamma_names = set(self.get_gamma_name())
        out = []
        for i, name in enumerate(names):
            if i == 0:
                out.append(float(np.real(params.get(name, m0 ** 2))))
            elif name == "one":
                # the m² coefficient: the row is ``-m²``, so this is +1
                out.append(1.0)
            elif name in gamma_names:
                out.append(m0 * float(np.real(params.get(name, 0.0))))
            else:
                out.append(float(np.real(params.get(name, 0.0))))
        return out

    def make_basis_transform(self) -> "Transform | None":
        """Physical parameters → this model's own basis weights.

        Every model has a basis (see :meth:`num_basis` / :meth:`den_basis`),
        so every model gets a transform: its ``input_names`` are the physical
        parameters and its ``output_names`` are the weight names from
        :meth:`get_num_name` / :meth:`get_den_name`.  That is what makes
        ``pwa_gen`` fit *physics* rather than one free weight per basis row.

        Subclasses with a non-uniform basis override this (``kmatrix2``,
        ``RhoOmegaGS``); returning ``None`` opts out.
        """
        from tabpwa.pwa_gen_transform import DefaultBasisWeightsTransform
        return DefaultBasisWeightsTransform(self)
    def get_bw_params(self, params=None):
        """Breit-Wigner peak mass and width from the running gamma(m).

        ``gamma(m)`` is computed with the *original* model parameters
        (the NPY file's reference mass is unchanged), while the
        *overridden* m₀, g₀ are used in the denominator formula.
        This means the peak shifts when the fit mass differs from
        the NPY reference.

        Solves Re(m₀² - m² - i·m₀·Σ g_i·gamma_i(m)) = 0 for m,
        then returns the BW mass and width at that point.

        The gamma couplings are read from :meth:`get_gamma_name` (the
        authoritative per-model coupling list, which handles both
        single-channel models and multi-channel models like
        ``ck_matrix_v2`` whose couplings are not in
        :meth:`get_defaults`), with :meth:`get_defaults` as the
        per-name fallback.

        Args:
            params: optional dict overriding config values.
                    Accepts either bare names or ``{name}_``-prefixed
                    keys (as used by the fitter), e.g.::

                        {"mass": 1.3, "width": 0.15}
                        {"a2(1320)p_mass": 1.3, "a2(1320)p_width": 0.15}
                        {"g_0": 0.2}
                        {"f0(980)_g_0": 0.2}

        Returns:
            dict with keys ``"mass_bw"`` and ``"width_bw"``.
        """
        from scipy.optimize import root_scalar

        # Helper: read *key* from params (full name, e.g. ``rhoA_mass``).
        # Fall back to kwargs using bare key (``mass``) via prefix strip.
        def _p(key, fallback=None):
            if params and key in params:
                return float(params[key])
            bare = key
            if bare.startswith(self.name + "_"):
                bare = bare[len(self.name) + 1:]
            if bare in self.kwargs:
                return float(self.kwargs[bare])
            return fallback

        m0 = _p(f"{self.name}_mass", 0.775)
        defaults = self.get_defaults()
        gamma_names = self.get_gamma_name()
        g0_vals = [_p(n, float(defaults.get(n, 0.0))) for n in gamma_names]
        n_ch = len(g0_vals)

        def sum_gamma_im(m):
            g_list = self.gamma(np.array([float(m)]))
            return sum(float(g0_vals[i]) * float(g_list[i][0].imag) for i in range(n_ch))

        def f(m):
            return m0**2 - m**2 + m0 * sum_gamma_im(m)

        sol = root_scalar(f, x0=m0, x1=m0 * 1.1, method='secant', xtol=1e-8)
        if not sol.converged:
            raise RuntimeError(f"get_bw_params: root finding failed for {self.name}")

        mass_bw = float(sol.root)
        g_list = self.gamma(np.array([mass_bw]))
        width_bw = sum(float(g0_vals[i]) * float(g_list[i][0].real) for i in range(n_ch))
        return {"mass_bw": mass_bw, "width_bw": width_bw}


# ═══════════════════════════════════════════════════════════════════
# Rational lineshapes: ratio of two fixed bases
# ═══════════════════════════════════════════════════════════════════

def complex_slot_names(heads, tail=None):
    """Expand complex parameter *head* names to their real slot names.

    Mirrors the ``ck`` convention (:func:`tabpwa.param_constraint.complex_tail`
    / :meth:`CKProduct._get_val`): a complex weight/parameter named by its
    head ``h`` is stored as the two real slots ``h{mag}`` / ``h{phase}``
    (default ``hr`` / ``hi``; renamable to e.g. ``hrho`` / ``hphi`` via
    ``build_defaults.scope(complex_tail=(...))``).  The head itself remains
    the public name — consumers such as :class:`CKProduct` append the tail
    at lookup time.

    Args:
        heads: iterable of head (base) parameter names.
        tail: optional ``(magnitude_suffix, phase_suffix)``; defaults to
            the active :func:`~tabpwa.param_constraint.complex_tail`.

    Returns:
        list of expanded slot names, in ``head`` order, two per head.
    """
    mag, phase = tuple(tail) if tail is not None else complex_tail()
    out = []
    for head in heads:
        out.extend([f"{head}{mag}", f"{head}{phase}"])
    return out


class BasisModel(BaseModel):
    """Particle amplitude as a ratio of two *fixed* bases.

    A rational lineshape::

        A(m) = ( Σ_i n_i · N_i(m) ) / ( Σ_j g_j · G_j(m) )

    where the basis functions ``N_i`` / ``G_j`` are **parameter-free**
    (they depend only on the mass grid and on constants from the model
    config), and the weights ``n_i`` / ``g_j`` are the model parameters.

    This is the particle-layer interface only: it says nothing about how
    the weights reach the kernel.  A subclass that additionally encodes
    the basis through :meth:`gamma` (e.g. a polynomial gamma basis) can be
    used by the existing kernels; the plain basis interface is what the
    lineshape code / tests consume directly.

    Subclass hooks
    --------------
    ``num_basis(m)``
        Numerator basis rows ``N_i(m)`` (complex, parameter-free).
        Default ``[ones_like(m)]`` — a constant numerator.
    ``den_basis(m)``
        Denominator basis rows ``G_j(m)``.  The default is the **kernel's
        uniform coefficient basis**::

            [ 1 , m² , −i·γ_1(m) , … , −i·γ_k(m) ]

        with the gamma rows from :meth:`gamma`.  The matching weights are
        exactly the kernel ``coeff`` vector ``(s0, −1, mult_l)``::

            g = [ m0² , −1 , m0·g0_1 , … , m0·g0_k ]

        all **real** (the ``−i`` lives in the basis row).  Thus
        ``Σ_j g_j·G_j(m) = m0² − m² − i·m0·Σ_l g0_l·γ_l(m)`` — the
        kernel denominator, bit-identical to
        :meth:`BaseModel.amplitude_raw` for a plain BW.
    ``get_num_name()`` / ``get_den_name()``
        Parameter names of the weights, in basis order.  ``get_num_name``
        defaults to ``[]`` — *no* free numerator weights, i.e. ``n = [1]``
        (the numerator is just ``num_basis()[0]``).  ``get_den_name``
        defaults to
        ``[f"{name}_s0", f"{name}_poly"] + get_gamma_name()``: the first
        two are the kernel's built-in slots (``s0`` constant weight,
        ``poly`` = the m² coefficient, fixed to −1), the rest are the
        gamma rows.
    ``make_basis_transform()``
        Optional :class:`~tabpwa.param_constraint.Transform` mapping the
        model's *physical* parameters to the weights::

            input_names  == the physical parameters (mass, pole positions,
                            couplings, ...)
            output_names == get_num_name() + get_den_name()

        The transform's :meth:`~tabpwa.param_constraint.Transform.backward`
        (analytic, or the base-class finite-difference VJP) is the
        parameter-gradient chain the fitter/kernel will rely on.  Default
        ``None``: the weights are derived from the raw physical
        parameters by :meth:`amplitude_raw` (see below).

    Missing-weight defaults
    -----------------------
    :meth:`amplitude_raw` reads ``params[name]`` for every weight.  The
    default basis derives the kernel coefficients from the physical
    parameters instead::

        {name}_s0   = params[f"{name}_mass"]**2      (mass²; kernel s0)
        {name}_poly = −1                              (fixed)
        γ weight_l  = mass · params[g0_name_l]        (kernel mult_l)

    so ``amplitude_raw(m, {"r_mass": m0, "r_width": g0})`` is exactly the
    plain BW.  A numerator with ``get_num_name() == []`` is ``n = [1]``;
    with explicit numerator names a missing weight is 0.

    Type convention
    ---------------
    The basis rows ``N_i`` / ``G_j`` are **complex** ``complex128``
    functions; the weights are real/complex as follows: **denominator
    weights are real** (unitarity), while **numerator weights may be
    complex** (production phases) — they map to the kernel's real
    ``coeff`` / complex ``ck`` respectively.  Any ``i`` / ``−i`` factor
    belongs to a basis row, never to a denominator weight.  The amplitude
    is complex because the rows are.
    """



@register_model("basis")
class DefaultBasisModel(BasisModel):
    """The default :class:`BasisModel` basis — a registered, plain basis.

    ``num_basis = [1]`` and ``den_basis = [1, m², −i·γ]`` with the kernel
    coefficient weights ``[m0², −1, m0·g0]``, i.e. exactly the plain
    Breit–Wigner denominator of :class:`BaseModel` (bit-identical, see
    ``tests/test_particle_basis.py::test_basis_default_matches_bw_bitwise``).
    It exists so a config can select the *default* basis explicitly
    (``model: basis``) — the reference case of the ``pwa_gen`` model.
    """
