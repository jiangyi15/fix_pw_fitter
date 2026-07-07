"""Base classes and registration machinery for particle models."""

import numpy as np

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

    def get_gamma_count(self):
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

    def make_mass_width_transform(self):
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
        gamma_names = [k for k in defaults if k != f"{self.name}_mass"]
        g0_vals = [_p(n, float(defaults[n])) for n in gamma_names]
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
