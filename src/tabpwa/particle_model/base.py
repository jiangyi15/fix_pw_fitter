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
    ``num_weights(params)`` / ``den_weights(params)``
                          the weight vectors for a parameter dict.
    ``make_basis_transform()``
                          physical parameters → those names; default ``None``
                          ("the weights are the parameters").

    Rows are complex; denominator weights real (unitarity), numerator weights
    may be complex (production phases).  Any ``i``/``−i`` belongs to a row.
    """

    def __init__(self, name, **kwargs):
        # ``name``/``kwargs`` ARE basis inputs: ``get_den_name`` and
        # ``den_weights`` build weight names from ``name`` and read config
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

    def num_weights(self, params):
        """Numerator weights, with the documented missing-value defaults."""
        names = list(self.get_num_name())
        if not names:
            # no free numerator weights: n = [1] (one constant row)
            return [1.0]
        return [params.get(name, 0.0) for name in names]

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

    def den_weights(self, params):
        """Denominator weight vector ``g_j`` for a parameter dict — REQUIRED."""
        raise NotImplementedError(
            f"{type(self).__name__}.den_weights: physical parameters -> g_j")

    def amp_raw(self, m, params=None):
        """``A(m)`` from the basis alone — rows × weights, NO transform.

        The basis-side evaluation entry point (``pwa_gen``, lineshape code).
        Separate from ``BaseModel.amplitude_raw`` so a basis-only model, which
        has no running width to resolve, is fully usable.
        """
        m_arr = np.asarray(m, dtype=float)
        params = {} if params is None else params
        num = np.zeros_like(m_arr, dtype=complex)
        for val, row in zip(self.num_weights(params), self.num_basis(m_arr)):
            num = num + val * np.asarray(row, dtype=complex)
        den = np.zeros_like(m_arr, dtype=complex)
        for val, row in zip(self.den_weights(params), self.den_basis(m_arr)):
            den = den + val * np.asarray(row, dtype=complex)
        return num / den

    def amp(self, m, params=None):
        """The basis entry point — ``amp_raw`` for a basis.

        A basis has no mass/width transform to apply, so this IS ``amp_raw``.
        Written as a real method, NOT ``amp = amp_raw``: a class-level alias
        binds the FUNCTION, so a subclass overriding ``amp_raw`` (the natural
        thing for a new basis) would keep the stale base implementation and
        silently return the wrong amplitude.  ``BaseModel`` overrides this with
        the transform-applying version.
        """
        return self.amp_raw(m, params)

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

            [ 1 , m² , −i·γ_1(m) , … , −i·γ_k(m) ]

        with the gamma rows from :meth:`gamma`.  Combined with the default
        weights ``[m0², −1, m0·g0_1, …]`` (see :meth:`get_den_name` and
        :meth:`den_weights`) the sum is exactly the kernel denominator
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
        return ([f"{self.name}_s0", "one"]
                + [f"{g}_t" for g in self.get_gamma_name()])

    def den_weights(self, params):
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
        out = []
        for i, name in enumerate(names):
            if i == 0:
                out.append(float(np.real(params.get(name, m0 ** 2))))
            elif name == "one":
                # the m² coefficient: the row is ``-m²``, so this is +1
                out.append(1.0)
            elif name.endswith("_t"):
                # a kernel-facing ``m0·g`` slot: the SOURCE name is the
                # physical coupling (``{g}`` without the ``_t``), so this
                # never collides with the fit parameter of the same base
                # name — see get_den_name().
                src = name[:-len("_t")]
                out.append(m0 * float(np.real(params.get(src, 0.0))))
            else:
                out.append(float(np.real(params.get(name, 0.0))))
        return out

    def amp(self, m, params=None):
        """The basis entry point, via THIS class's amplitude.

        ``BasisModel.amp`` is the no-transform evaluation; a model that
        has a mass/width transform must resolve the physical parameters
        first, so this routes through :meth:`amplitude`.
        """
        return self.amplitude(m, params)

    def amplitude_raw(self, m, params):
        """Raw amplitude ``Σ n_i N_i / Σ g_j G_j`` (no transform applied).

        This is ONE implementation for every model: each resonance is a ratio
        of two fixed bases (:meth:`num_basis` / :meth:`den_basis`), and the
        weights are read by :meth:`num_weights` / :meth:`den_weights`.
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
        for val, row in zip(self.num_weights(params), self.num_basis(m_arr)):
            num = num + val * np.asarray(row, dtype=complex)

        den = np.zeros_like(m_arr, dtype=complex)
        for val, row in zip(self.den_weights(params), self.den_basis(m_arr)):
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

class PlainBasisWeightsTransform(Transform):
    """Physical parameters → the weights of the PLAIN-BW basis.

    This is :class:`BaseModel`'s default, NOT the interface's: it is derived
    from ``get_gamma_name()`` / ``gamma()``, which only ``BaseModel`` has.  A
    model that states its basis directly (``kmatrix2``, and anything else
    subclassing :class:`BasisModel` alone) supplies its own transform instead —
    ``BasisModel.make_basis_transform`` defaults to ``None``.

    ``input_names`` are the physical ``{name}_mass`` plus the gamma couplings;
    ``output_names`` are exactly :meth:`get_den_name` (the kernel ``coeff``
    names), so the names always match the rows the builder emitted.

    The default basis is ``[1, m², -i·γ_j(m)]`` with weights
    ``[m0², -1, m0·g_j]`` — the plain BW denominator.  A model that overrides
    ``den_basis`` **must** override this too.  ``kmatrix2`` does.  The base
    class cannot derive an arbitrary basis' weights from ``m0``/``g``.

    KNOWN GAP (pre-existing, not introduced by the basis split):
    ``RhoOmegaGS`` has 10 basis rows but does NOT override this, so it inherits
    the 3-row mapping — ``ro_w0..ro_w7`` are read as transform INPUTS while the
    real physical parameters are ``ro_mass/width/mass2/width2``, and every
    weight resolves to 0 (denominator ``m0²−m²`` only).  A ``RhoOmegaGS`` +
    ``pwa_gen`` config is therefore wrong on this path today; it needs its own
    transform built on ``RhoOmegaGSWeightsTransform``.  Fixing that is separate
    work from this refactor.
    """

    _has_inverse = False

    def __init__(self, model):
        self.model = model
        name = model.name
        self.gamma_names = list(model.get_gamma_name())
        self.input_names = [f"{name}_mass"] + list(self.gamma_names)
        # ``"one"`` is a CONSTANT (the m² coefficient, always +1) — it is not a
        # function of the physical parameters, so it is not a transform output.
        # The constraint layer pins it (`set_fixed`), the same mechanism that
        # pins any other fixed parameter.
        self.output_names = [n for n in model.get_den_name() if n != "one"]

    def forward(self, d):
        """``m0²``, ``-1``, ``m0·g_j`` — the uniform basis weights."""
        d = d or {}
        m0 = float(d.get(f"{self.model.name}_mass",
                         self.model.kwargs.get("mass", 0.775)))
        # ``"one"`` is pinned by the constraint layer, not produced here.
        vals = [m0 * m0]
        # Each gamma row's weight is ``m0·g`` and is named ``{g}_t`` — the
        # kernel-facing slot, DISTINCT from the physical fit parameter ``{g}``.
        # Emitting the fit name here would make the transform overwrite its
        # own input (`apply_forward` replaces `output_names` in the resolved
        # dict), losing the physical coupling.
        vals += [m0 * float(d.get(g, 0.0)) for g in self.gamma_names]
        return dict(zip(self.output_names, vals))
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
