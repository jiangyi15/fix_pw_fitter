"""Config-selected backend: config: {backend: ...} is used by Fitter."""

import os
import tempfile

from ampfit import Fitter
from ampfit.config_loader import Config
from ampfit.backends.numpy_pwa_backend import NumpyPWABackend

BASE = os.path.join(os.path.dirname(__file__), "config_pwa.yml")


def _config_with(text):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "c.yml")
    with open(p, "w") as fh:
        fh.write(text)
    return p


def test_config_backend_spec_and_fitter_default():
    src = open(BASE).read()
    p = _config_with("config:\n    backend: numpy_pwa\n\n" + src)
    cfg = Config(p)
    assert cfg.backend_spec == "numpy_pwa"

    f = Fitter(p)                     # no explicit backend -> config value
    try:
        assert isinstance(f.backend, NumpyPWABackend)
    finally:
        f.backend.free()

    # explicit argument still wins over the config
    f2 = Fitter(p, backend="numpy_pwa")
    try:
        assert isinstance(f2.backend, NumpyPWABackend)
    finally:
        f2.backend.free()


def test_config_backend_absent_is_none():
    src = open(BASE).read()
    p = _config_with(src)
    assert Config(p).backend_spec is None


def test_create_backend_string_spec_keeps_kwargs():
    """kwargs must not be silently dropped for a string spec."""
    from ampfit.backends import create_backend, register_backend

    @register_backend("_kwarg_probe", amp_model=None)
    class _Probe:
        def __init__(self, kernel_config, value=None):
            self.value = value

    assert create_backend("_kwarg_probe", {}, value=7).value == 7
    assert create_backend({"name": "_kwarg_probe"}, {}, value=9).value == 9
