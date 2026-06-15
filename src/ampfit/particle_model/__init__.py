"""
Particle model definitions for the amplitude analysis framework.

Each resonance lineshape is defined as a ``BaseModel`` subclass
registered via ``@register_model(name)``.  Models are instantiated
with ``build_particle(name, model="BW", **kwargs)``.

To add a custom model, create a new file in this directory that
imports from ``.base`` and uses ``@register_model``::

    # particle_model/my_model.py
    import numpy as np
    from .base import BaseModel, register_model

    @register_model("MyCustom")
    class MyCustomModel(BaseModel):
        def gamma(self, m):
            return [np.exp(-m**2) + 0j]

Then ensure the file is imported (e.g., in ``__init__.py``).
"""

from .base import BaseModel, ALL_MODELS, register_model, build_particle
from . import models_builtin  # noqa: F401 — trigger registration of built-in models

__all__ = ["BaseModel", "ALL_MODELS", "register_model", "build_particle"]
