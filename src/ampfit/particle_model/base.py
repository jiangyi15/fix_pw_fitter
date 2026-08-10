"""Base classes and registration machinery for particle models."""

import warnings

import numpy as np
from ampfit.param_constraint import Transform

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
        """Raw amplitude A(m) without applying any transform.

        Evaluates the BW denominator formula::

            A(m) = 1 / (m₀² - m² - i·m₀·Σ g_j·γ_j(m))

        where ``m₀ = params[f\"{name}_mass\"]`` and ``g_j`` are read
        from *params* using :meth:`get_gamma_name`.

        Args:
            m: mass array.
            params: dict of parameter names → values (must include
                    ``{name}_mass`` and all gamma couplings).

        Returns:
            Complex amplitude A(m).
        """
        m_arr = np.asarray(m, dtype=float)
        m0 = float(params.get(f"{self.name}_mass", 0.775))
        g_list = self.gamma(m_arr)
        total = 0.0
        for name, g_val in zip(self.get_gamma_name(), g_list):
            g0 = float(params.get(name, 0.0))
            total += g0 * g_val
        return 1.0 / (m0**2 - m_arr**2 - 1j * m0 * total)

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

        Returns a :class:`~ampfit.param_constraint.Transform` that reads
        physical mass and gamma values from the full parameter dict and
        replaces them with the model's real (derived) mass and width.

        The default returns ``None`` (physical = real, no transform).
        Override in subclasses with running widths (GS_rho, Bugg, etc.)
        to return a ``Transform`` with the appropriate computation.
        """
        return None
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
