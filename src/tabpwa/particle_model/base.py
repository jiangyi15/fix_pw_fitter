"""Base classes and registration machinery for particle models."""

import warnings

import numpy as np
from tabpwa.param_constraint import Transform, complex_tail

ALL_MODELS = {}


def register_model(name):
    """Decorator: register a particle model class under `name`.

    The class must subclass :class:`BasisModel`: every consumer (the
    amplitude models, the builders, the backends) assumes the basis protocol
    is available, so a model outside the interface would fail later and
    further away.  `BaseModel` is the interface's default implementation.
    """
    def _f(cls):
        if not issubclass(cls, BasisModel):
            raise TypeError(
                f"particle model {cls.__name__} must subclass BasisModel "
                f"(or BaseModel, its default implementation)")
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


def _numerator_weights(names, params):
    """Numerator weights read from *params* by name.

    A head may be given directly (``params[head]``, real or complex) or as the
    ``ck`` pair ``head{mag}`` / ``head{phase}`` (``complex_tail()``) — what the
    model's transform writes.  No heads ⇒ the constant ``1``.
    """
    if not names:
        return [1.0]
    mag, phase = complex_tail()
    out = []
    for h in names:
        if h in params:
            out.append(params[h])
        else:
            out.append(float(params.get(f"{h}{mag}", 0.0))
                       * np.exp(1j * float(params.get(f"{h}{phase}", 0.0))))
    return out


class BasisModel:
    """Particle amplitude as a ratio of two *fixed* bases — THE INTERFACE.

    A rational lineshape::

        A(m) = ( Σ_i n_i · N_i(m) ) / ( Σ_j g_j · G_j(m) )

    ``N_i`` / ``G_j`` are **parameter-free** (mass grid + model-config
    constants); the weights ``n_i`` / ``g_j`` are the model parameters.  This
    class declares what a basis CONSUMER needs and nothing more: it has no
    concept of a running width, a gamma table, or the plain-``pwa`` kernel
    format.

    **Two parallel particle interfaces, with separate entry points.**
    :class:`BaseModel` subclasses this one, adds the running-width protocol
    (``gamma`` / ``get_gamma_name``) that the plain ``pwa`` family reads, and
    supplies this interface's DEFAULT implementation (its ``den_basis``
    composes ``[1, −m², −i·γ…]`` from that running width).  A model that states
    its basis directly — ``kmatrix2`` — subclasses **this** class and has no
    ``gamma`` and no ``get_gamma_name`` at all::

        BasisModel.amp_raw(m, params)     rows × weights, NO transform
        BasisModel.amp(m, params)         same (a basis has no transform)
        BaseModel.amplitude_raw(m, ...)   transform + basis
        BaseModel.amplitude(m, ...)       transform + basis
        BaseModel.amp(m, params)          → its own amplitude()

    Subclass hooks (required unless stated)
    --------------------------------------
    ``num_basis(m)``      numerator rows ``N_i(m)``; default one constant row.
    ``den_basis(m)``      denominator rows ``G_j(m)``.
    ``get_num_name()``    numerator names; default ``[]`` (``n = [1]``).
    ``get_den_name()``    denominator weight names.
    ``_numerator_weights(names, params)`` / the denominator weights read by
                          the weight vectors for a parameter dict.
    ``make_basis_transform()``
                          physical parameters → those names; default ``None``
                          ("the weights are the parameters").

    Rows are complex; denominator weights real (unitarity), numerator weights
    may be complex (production phases).  Any ``i``/``−i`` belongs to a row.
    """

    def __init__(self, name, **kwargs):
        # ``name``/``kwargs`` ARE basis inputs: ``get_den_name`` and
        # the weight names are built from ``name`` and read config
        # defaults out of ``kwargs``.
        self.name = name
        self.kwargs = kwargs
        self._parent = None

    def register_parent(self, particle):
        """Store the Particle that owns this model (for decay tree access)."""
        self._parent = particle

    def make_basis_transform(self):
        """Physical parameters → this model's basis weight names.

        Default ``None``: the basis weights ARE the raw parameters, so no
        transform is needed.  Overridden by :class:`BaseModel` (whose weights
        are ``m0²`` / ``m0·g``) and by models with their own mapping
        (``kmatrix2``).
        """
        return None

    def get_defaults(self):
        """Physical defaults for this model's parameters — ``{}`` by default.

        A basis model need not have physical parameters at all: when its basis
        weights ARE the parameters (``make_basis_transform() -> None``), any
        default would be an arbitrary starting point, and the fitter's
        ``initial_values`` starts un-defaulted names randomly instead.
        ``BaseModel`` overrides this with the mass/width defaults the plain
        ``pwa`` family reads.
        """
        return {}

    def num_basis(self, m):
        """Numerator basis rows ``N_i(m)`` (complex, parameter-free).

        Default: a single constant row.  When overriding with more than
        one row, also override :meth:`get_num_name` with matching names.
        """
        m_arr = np.asarray(m, dtype=float)
        return [np.ones_like(m_arr, dtype=complex)]

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

    def den_basis(self, m):
        """Denominator rows ``G_j(m)`` — REQUIRED (no generic default).

        A basis cannot invent its own denominator: the rows ARE the model's
        physics.  ``BaseModel`` supplies the plain-BW one; ``kmatrix2`` and
        the GS family supply theirs.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.den_basis: a basis must state its "
            f"denominator rows (subclass BasisModel, or use BaseModel for the "
            f"plain Breit-Wigner basis)")

    def get_den_name(self):
        """Denominator weight names, in basis order — REQUIRED."""
        raise NotImplementedError(
            f"{type(self).__name__}.get_den_name: one name per den_basis row")

    def amp_raw(self, m, params=None):
        """The basis ratio ``Σ n_i N_i / Σ g_j G_j`` at *params* as given.

        No transform and no defaults: the weights are read from *params* by
        name.  :meth:`amp` is this composed with
        :meth:`make_basis_transform`.
        """
        m_arr = np.asarray(m, dtype=float)
        params = params or {}
        num = np.zeros_like(m_arr, dtype=complex)
        for val, row in zip(_numerator_weights(self.get_num_name(), params),
                            self.num_basis(m_arr)):
            num = num + val * np.asarray(row, dtype=complex)
        # the denominator weights ARE the dict: the model's transform put
        # them there under their names (``one`` reads as its own value)
        weights = [float(np.real(params.get(n, 1.0 if n == "one" else 0.0)))
                   for n in self.get_den_name()]
        den = np.zeros_like(m_arr, dtype=complex)
        for val, row in zip(weights, self.den_basis(m_arr)):
            den = den + val * np.asarray(row, dtype=complex)
        return num / den

    def amp(self, m, params=None):
        """``amp_raw`` composed with the BASIS transform.

        The model's own ``make_basis_transform()`` turns the physical
        parameters (defaults filled first) into the basis weights; the ratio
        is then :meth:`amp_raw`, so the two entries never share a body.
        """
        d = {**self.get_defaults(), **(params or {})}
        tr = self.make_basis_transform()
        if tr is not None:
            d = tr.apply_forward(d)
        return self.amp_raw(m, d)

class BaseModel(BasisModel):
    """Default implementation of :class:`BasisModel` + the ``pwa`` interface.

    Adds the running-width protocol — ``gamma()`` / ``get_gamma_name()`` /
    ``get_gamma_count()`` / ``get_defaults()`` — that the plain ``pwa`` family
    reads, and supplies the DEFAULT basis on top of it::

        den_basis = [ 1 , −m² , −i·γ_1(m) , … , −i·γ_k(m) ]
        weights   = [ m0² ,  "one"=+1 , m0·g0_1 , … ]

    so ``Σ_j g_j·G_j = m0² − m² − i·m0·Σ_l g0_l·γ_l(m)`` — the plain
    Breit-Wigner denominator.  ``gamma`` is declared HERE, so ``den_basis``
    delegating to it is this implementation's own business, not the interface
    reaching into a sibling.

    Subclasses must implement:
        gamma(self, m) -> list[ndarray]

    Subclasses may override:
        get_defaults() -> dict[str, float]
        get_gamma_count() -> int            (default 1)
        get_gamma_name() -> list[str]       (default [`{name}_width`])
        …and any ``BasisModel`` hook (``kmatrix2`` overrides ``den_basis``).
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

    def get_mass_name(self):
        """Name of the mass parameter this model's weights are built from.

        The interface's ONE declaration of it: :meth:`get_bw_params`,
        :meth:`get_bw_param_names` and ``PlainBasisWeightsTransform`` all read this instead of each rebuilding
        ``f"{self.name}_mass"`` — a model that names its mass differently
        overrides it in one place.
        """
        return f"{self.name}_mass"

    def get_bw_param_names(self):
        """Candidate parameter names for :meth:`get_bw_params`.

        The physical mass plus the running-width slots.  A model with a width
        parameter that is not one of ``get_gamma_name()`` overrides this
        (``ck_matrix_v2`` adds its standalone width).
        """
        return [self.get_mass_name()] + list(self.get_gamma_name())

    def register_parent(self, particle):
        """Store the Particle that owns this model (for decay tree access)."""
        self._parent = particle

    def gamma(self, m):
        """Compute the gamma (width) function at masses *m*.

        Returns a list of complex ndarrays, one per gamma parameter.
        """
        return [np.ones_like(m) + 0j]





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

    def den_basis(self, m):
        """Denominator basis rows ``G_j(m)`` (complex, parameter-free).

        Default: the kernel's uniform coefficient basis::

            [ 1 , −m² , −i·γ_1(m) , … , −i·γ_k(m) ]

        with the gamma rows from :meth:`gamma`.  Combined with the model's
        weights ``[m0², +1, m0·g0_1, …]`` (its basis transform's outputs, see
        :meth:`get_den_name`) the sum is exactly the kernel denominator
        ``m0² − m² − i·m0·Σ_l g0_l·γ_l(m)``.  The minus sign lives in the ROW,
        so the shared ``"one"`` weight is ``+1``.
        """
        m_arr = np.asarray(m, dtype=float)
        # ``-m²`` here, so its weight is exactly the SHARED ``"one"`` = +1 and
        # the name reads as its value.  The pair gives the plain BW
        # denominator ``m0² - m² - i·m0·Σ g_j γ_j``.  Do NOT flip one side
        # without the other — that is what silently reversed every m² term.
        rows = [np.ones_like(m_arr, dtype=complex), -(m_arr ** 2) + 0j]
        rows.extend(-1j * g for g in self.gamma(m_arr))
        return rows

    def get_den_name(self):
        """Weight names for the denominator rows ``g_j``, in basis order.

        ``[f"{name}_s0", "one"] + [f"{g}_t" for g in get_gamma_name()]`` —
        the ``m0²`` slot, the shared m² coefficient, then one ``m0·g`` weight
        per coupling.  The names ARE the layout: ``s0`` is ``m0²`` and each
        ``{g}_t`` is ``m0·g`` (``PlainBasisWeightsTransform`` produces both),
        so a consumer that knows the model does not need them classified.

        The m² weight is the SHARED literal ``"one"``, not ``{name}_poly``:
        its value is the same ``+1`` for every resonance (the row is ``−m²``),
        and ``poly`` is ``kmatrix2``'s word for a *fitted* coefficient
        ``−(m1²+m2²)`` — a different quantity with a different free/fixed
        status.
        """
        return ([f"{self.name}_s0", "one"]
                + [f"{g}_t" for g in self.get_gamma_name()])

    def amplitude_raw(self, m, params):
        """A(m) from the GAMMA names, parameters as given.

        The denominator is ``m0² − m² − i·m0·Σ_l g_l·γ_l(m)``; no transform is
        applied and no defaults are filled — :meth:`amplitude` is this
        composed with :meth:`make_mass_width_transform`.
        """
        m_arr = np.asarray(m, dtype=float)
        params = params or {}
        m0 = float(params.get(self.get_mass_name(),
                              self.kwargs.get("mass", 0.775)))
        den = (m0 ** 2 - m_arr ** 2).astype(complex)
        for gname, grow in zip(self.get_gamma_name(), self.gamma(m_arr)):
            den = den - 1j * m0 * float(params.get(gname, 0.0)) * np.asarray(
                grow, dtype=complex)
        num = np.zeros_like(m_arr, dtype=complex)
        for val, row in zip(_numerator_weights(self.get_num_name(), params),
                            self.num_basis(m_arr)):
            num = num + val * np.asarray(row, dtype=complex)
        return num / den

    def amplitude(self, m, params=None):
        """``amplitude_raw`` composed with the MASS/WIDTH transform.

        The model's own stage produces the kernel-facing ``m0`` and the
        couplings from the physical parameters (defaults filled first), and
        the denominator is then stated through the gamma names.
        """
        d = {**self.get_defaults(), **(params or {})}
        tr = self.make_mass_width_transform()
        if tr is not None:
            d = tr.apply_forward(d)
        return self.amplitude_raw(m, d)

    def make_basis_transform(self):
        """Physical parameters → the PLAIN-BW basis weights.

        Derived from ``get_gamma_name()`` / ``gamma``, which only this
        class has — a model subclassing ``BasisModel`` alone supplies its
        own transform (or none).
        """
        return PlainBasisWeightsTransform(self)

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

        m0 = _p(self.get_mass_name(), 0.775)
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

class PlainBasisWeightsTransform(Transform):
    """Physical parameters → the weights of the PLAIN-BW basis.

    This is :class:`BaseModel`'s default, NOT the interface's: it is derived
    from ``get_gamma_name()`` / ``gamma()``, which only ``BaseModel`` has.  A
    model that states its basis directly (``kmatrix2``, and anything else
    subclassing :class:`BasisModel` alone) supplies its own transform instead —
    ``BasisModel.make_basis_transform`` defaults to ``None``.

    ``input_names`` are the model's PHYSICAL parameters, for every model: the
    mass, plus — when a coupling the basis reads is derived rather than fitted
    — the inputs of the stage that derives it (``RhoOmegaGS``: ``w0..w7`` from
    ``mass/width/mass2/width2``; ``BWRS``: ``{p}_width_t`` from mass and
    width).  A model whose couplings ARE its parameters (the default plain BW)
    declares them directly.

    This class IS the conversion for the fit: it computes the values and their
    VJP, so the fit never depends on a number-only accessor.

    The stage is held HERE (``self._mw``), so the transform is self-contained:
    it is the model's one answer to "physical parameters → basis weights", and
    no other transform has to be registered before it for it to be correct.

    A model whose basis weights are NOT ``m0 · coupling`` must override this
    with its own basis transform — ``kmatrix2`` does.

    The default basis is ``[1, −m², −i·γ_j(m)]`` with weights
    ``[m0², +1, m0·g_j]`` — the plain BW denominator.  A model that overrides
    ``den_basis`` **must** override this too.  ``kmatrix2`` does.  The base
    class cannot derive an arbitrary basis' weights from ``m0``/``g``.

    It maps the weight names positionally (``m0²`` then ``m0·{g}``), which is
    the ``[1, −m², −i·γ_j]`` layout.  A model with a DIFFERENT basis must
    override it — ``kmatrix2`` does (``KMatrix2WeightsTransform``).
    """

    _has_inverse = False

    def __init__(self, model):
        if not isinstance(model, BaseModel):
            # Cross-interface test of the allowed kind: this class is
            # ``BaseModel``'s default mapping and calls its running-width
            # stage.  A ``BasisModel`` must supply its own mapping (or None).
            raise TypeError(
                f"PlainBasisWeightsTransform is BaseModel's default basis "
                f"transform; {type(model).__name__} subclasses BasisModel and "
                f"must return its own from make_basis_transform()")
        self.model = model
        # ``"one"`` is EXCLUDED: it is a physical CONSTANT, not a function of
        # this model's parameters, and it is produced once for the whole fit by
        # a `FixedValueTransform` (see `AmplitudeModel.particle_transforms`).
        # Leaving it out here keeps "which model's transform defines this name"
        # unambiguous — two transforms declaring the same output would be
        # treated as alternatives and the later one would replace the earlier.
        # Everything comes from the model's own declarations: the names from
        # `get_den_name()` / `get_gamma_name()`, the mass from
        # `get_mass_name()`.  The names ARE the layout, so nothing is
        # classified: `forward` asks the model for the values and `backward`
        # states the derivative of that same conversion.
        self.output_names = [n for n in model.get_den_name() if n != "one"]
        self.mass_name = model.get_mass_name()
        couplings = list(model.get_gamma_name())

        # SELF-HELD: this transform maps PHYSICAL parameters to the basis
        # weights for EVERY model, so it carries the model's own stage when
        # one of the couplings the basis reads is derived rather than fitted
        # (``RhoOmegaGS``: ``w0..w7`` from mass/width/mass2/width2;
        # ``BWRS``: ``{p}_width_t`` from mass and width).  A model without
        # such a stage (the default plain BW) declares those couplings as
        # plain parameters instead — same mapping, one less object.
        #
        # Nothing is registered "before" this transform to feed it: that
        # ordering contract lived outside the transform and could be broken
        # by a registration change.  Calling the stage here also means the
        # inputs are always the PHYSICAL ones, which is what
        # ``make_basis_transform()`` promises.
        stage = model.make_mass_width_transform()
        produced = set(stage.output_names) if stage is not None else set()
        self._mw = stage if any(c in produced for c in couplings) else None

        inputs = []

        def _add(names):
            for n in names:
                if n not in inputs:
                    inputs.append(n)

        if self._mw is not None:
            _add(self._mw.input_names)
        # The mass is an input of THIS transform only when the model stage
        # does not produce it.  ``KToSVDWeightsTransform`` returns
        # ``{name}_mass`` alongside the couplings; declaring it as an input
        # too would put it on the fitted surface (a direction the stage then
        # overwrites) instead of leaving it, as in ``pwa``, a produced value.
        # A basis with no weight at all (``OneModel``) reads nothing.
        if self.output_names and self.mass_name not in produced:
            _add([self.mass_name])
        _add([c for c in couplings if c not in produced])
        self.input_names = inputs

    def _resolved(self, d):
        """The transform's inputs, with the model's own stage applied.

        A transform's ``forward`` returns ONLY its outputs, so the result is
        MERGED into ``d`` rather than replacing it — the mass is an input of
        this transform and must survive the stage (replacing ``d`` with the
        stage's output is how the fitted mass once fell back to the config
        value).
        """
        d = d or {}
        if self._mw is None:
            return d
        return {**d, **self._mw.forward(dict(d))}

    def _m0(self, d):
        """The fitted mass — the ONE place ``forward`` and ``backward`` read it.

        REQUIRED, with no fallback: the mass is a declared input, so a chain
        that failed to supply it has a wiring error.  Defaulting to the config
        value is not a rescue — it silently built ``s0`` and every ``m0·g``
        from a mass the fit was not using (measured ``dQ/dm = −6.9e4`` against
        a true ``−77.6``).  The same rule applies to every declared input:
        absent means raise, never zero.
        """
        return float(d[self.mass_name])

    def forward(self, d):
        """The model's own basis weights, selected to this transform's slots.

        This class is built BY the model (``BaseModel.make_basis_transform``),
        so it knows the model's conversion and writes it once, with a VJP: the
        weights are ``s0 = m0²`` and ``{g}_t = m0·g``, i.e. every
        ``get_den_name()`` slot except the shared ``"one"`` (a
        ``FixedValueTransform`` produces that once for the whole fit).

        Physical inputs in: the model's own stage is applied first when the
        couplings it reads are derived (see :meth:`_resolved`).
        """
        d = self._resolved(d)
        m0 = self._m0(d)
        # the model's conversion, written once: ``s0 = m0²`` and each
        # ``{g}_t = m0·g`` (``get_den_name`` says which name is which)
        out = {}
        for slot in self.output_names:
            if slot == f"{self.model.name}_s0":
                out[slot] = m0 * m0
            else:                       # ``{g}_t`` → coupling ``g``
                out[slot] = m0 * float(d[slot[:-len("_t")]])
        return out

    def backward(self, grad_out, d_in=None):
        """Exact VJP of ``s0 = m0²`` and ``{g}_t = m0·g``.

        Both maps are trivial, so the derivative is written down, not measured
        — the inherited ``fd_backward`` would spend ``2·len(input_names)``
        forwards per call and inject finite-difference noise into a chain the
        fitter differentiates every iteration.

        A DERIVED coupling is chained through the model's own stage
        (:meth:`~Transform.backward` of it), so the result is w.r.t. the
        physical parameters, exactly like :meth:`forward`.
        """
        d = self._resolved(d_in)
        grad = {name: 0.0 for name in self.input_names}
        coupling_grad = {}
        # The model's own names, used exactly as it declares them.
        if self.output_names:
            m0 = self._m0(d)
            # the model's exact names: ``{name}_s0`` carries ``m0²`` and each
            # ``{g}_t`` carries ``m0·g`` (this transform's conversion)
            g = float(grad_out.get(f"{self.model.name}_s0", 0.0))
            if g:
                grad[self.mass_name] = grad.get(self.mass_name, 0.0) + 2.0 * m0 * g
            for src in self.model.get_gamma_name():
                g = float(grad_out.get(f"{src}_t", 0.0))
                if g:
                    grad[self.mass_name] = (grad.get(self.mass_name, 0.0)
                                            + g * float(d[src]))
                    coupling_grad[src] = coupling_grad.get(src, 0.0) + g * m0
        if self._mw is not None:
            # route the derived couplings through the model's own VJP; what
            # it does not produce is a plain input and is copied below
            seeded = {n: coupling_grad.pop(n, 0.0)
                      for n in self._mw.output_names}
            for n, v in self._mw.backward(seeded, d_in=d).items():
                grad[n] = grad.get(n, 0.0) + v
        for src, v in coupling_grad.items():
            grad[src] = grad.get(src, 0.0) + v
        return grad

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



@register_model("basis")
class DefaultBasisModel(BaseModel):
    """The default :class:`BasisModel` basis — a registered, plain basis.

    ``num_basis = [1]`` and ``den_basis = [1, m², −i·γ]`` with the kernel
    coefficient weights ``[m0², −1, m0·g0]``, i.e. exactly the plain
    Breit–Wigner denominator of :class:`BaseModel` (bit-identical, see
    ``tests/test_particle_basis.py::test_basis_default_matches_bw_bitwise``).
    It exists so a config can select the *default* basis explicitly
    (``model: basis``) — the reference case of the ``pwa_gen`` model.
    """
