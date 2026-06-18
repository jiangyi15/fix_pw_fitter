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
        get_gamma_count() -> int       (default 1)
        get_gamma_name() -> list[str]  (default [`{name}_width`])
        get_gamma_defaults() -> list   (default [width from kwargs])
    """

    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self._parent = None

    def get_gamma_count(self):
        return 1

    def get_gamma_name(self):
        return [f"{self.name}_width"]

    def get_gamma_defaults(self):
        """Default values for each gamma/width parameter.

        Returns list of floats matching get_gamma_name() length.
        """
        return [float(self.kwargs.get("width", 0.1))]

    def register_parent(self, particle):
        """Store the Particle that owns this model (for decay tree access)."""
        self._parent = particle

    def gamma(self, m):
        """Compute the gamma (width) function at masses *m*.

        Returns a list of complex ndarrays, one per gamma parameter.
        """
        return [np.ones_like(m) + 0j]
