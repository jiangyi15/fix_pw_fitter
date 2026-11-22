"""Register ``jax.config`` as a legacy importable module, for PWACG's sources.

PWACG's ``Tensor/CacheTensor.py`` (and ``dmultiprocess/multiprocess_interface.py``)
do

    from jax.config import config
    ...
    config.update("jax_enable_x64", True)

Modern JAX has no importable ``jax.config`` *module* — it is an attribute of the
``jax`` package holding the Config object — so that import raises.  The modern
equivalent of the pair is ``jax.config.update(...)``; this module makes the
legacy statement resolve without editing the vendored sources (the clone is kept
byte-identical, so it can be re-cloned and diffed).

Import this FIRST, before anything from PWACG.
"""
import sys
import types

import jax

_m = types.ModuleType("jax.config")
_m.config = jax.config
_m.__spec__ = None
sys.modules["jax.config"] = _m
