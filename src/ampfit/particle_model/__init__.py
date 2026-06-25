"""
Particle model definitions for the amplitude analysis framework.

Each resonance lineshape is defined as a ``BaseModel`` subclass
registered via ``@register_model(name)``.  Models are instantiated
with ``build_particle(name, model="BW", **kwargs)``.

To add a custom model, create a new file in this directory that
imports from ``.base`` and uses ``@register_model``, then import it
in this ``__init__.py``::

    # particle_model/my_model.py
    import numpy as np
    from .base import BaseModel, register_model

    @register_model("MyCustom")
    class MyCustomModel(BaseModel):
        def gamma(self, m):
            return [np.exp(-m**2) + 0j]

    # particle_model/__init__.py
    from . import my_model   # noqa: F401
"""

from .base import BaseModel, ALL_MODELS, register_model, build_particle
from . import models_builtin   # noqa: F401 — register built-in models
from . import bugg_model       # noqa: F401 — register Bugg model
from . import gs_rho_model     # noqa: F401 — register GS_rho model
from . import flatte_model     # noqa: F401 — register FlatteC model
from . import bwr_model        # noqa: F401 — register BWR model
from . import ck_matrix_model  # noqa: F401 — register ck_matrix model
from . import ck_matrix_v2     # noqa: F401 — register ck_matrix_v2 model
__all__ = [
    "BaseModel", "ALL_MODELS", "register_model", "build_particle",
]
