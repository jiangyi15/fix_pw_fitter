"""Config-declared plugins: ``plugin: [module, ...]`` imports before the model.

A plugin is a module whose import registers what the config names (a backend, an
amplitude model, a particle model).  The Fitter imports it right after loading
the raw config — before `build_amplitude_model` and before the backend spec is
resolved — so `backend: mpi_service` works without an `import` in the script.
"""

import sys

import pytest

from tabpwa import Fitter
from tabpwa.backends import backends_for_model
from tabpwa.config_loader import RawConfig
from tabpwa.plugins import load_plugins

CFG = "tests/config_pwa.yml"

_PLUGIN = '''
from tabpwa.backends.core import ComputeBackend, register_backend

LOADS = globals().get("LOADS", 0) + 1

@register_backend("{name}", tags=("probe",))
class DemoPluginBackend(ComputeBackend):
    """A registry marker; this test never constructs it."""
'''


def _cfg(**extra):
    dic = RawConfig(CFG).to_dict()
    dic.update(extra)
    return dic


def _plugin(tmp_path, monkeypatch, name):
    (tmp_path / f"{name}.py").write_text(_PLUGIN.format(name=name))
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


def test_a_config_imports_its_plugin_and_its_registrations_are_visible(
        tmp_path, monkeypatch):
    name = _plugin(tmp_path, monkeypatch, "tabpwa_demo_plugin_a")
    f = Fitter(_cfg(plugin=[name]), backend="numpy_pwa")
    assert name in sys.modules
    assert name in backends_for_model(f.model.name)      # registered before resolve
    assert sys.modules[name].LOADS == 1

    # importing twice is a no-op (sys.modules)
    assert load_plugins(name) == [name]
    assert sys.modules[name].LOADS == 1


def test_the_string_form_is_accepted(tmp_path, monkeypatch):
    name = _plugin(tmp_path, monkeypatch, "tabpwa_demo_plugin_b")
    f = Fitter(_cfg(plugin=name), backend="numpy_pwa")
    assert name in backends_for_model(f.model.name)


def test_a_missing_plugin_names_the_plugin_and_the_origin():
    with pytest.raises(ImportError, match="no_such_plugin_xyz"):
        Fitter(_cfg(plugin=["no_such_plugin_xyz"]), backend="numpy_pwa")
    with pytest.raises(ImportError, match="config my.yml"):
        load_plugins("no_such_plugin_xyz", where="config my.yml")


def test_a_config_without_plugins_is_unchanged():
    f = Fitter(_cfg(), backend="numpy_pwa")
    assert f.model.name == "pwa"

_MODEL_PLUGIN = '''
from tabpwa.amp_model import AmplitudeModel, register_amplitude_model
from tabpwa.backends import register_backend
from tabpwa.backends.core import ComputeBackend


@register_amplitude_model("{name}_model")
class DemoAmp(AmplitudeModel):
    name = "{name}_model"


@register_backend("demo_stub", model="{name}_model")
class DemoStub(ComputeBackend):
    def __init__(self, kernel_config, **kw):
        self.kernel_config = kernel_config
'''


def test_a_plugin_can_supply_the_AMPLITUDE_MODEL_the_config_names(
        tmp_path, monkeypatch):
    """The import runs BEFORE ``build_amplitude_model``, so the name resolves."""
    name = "tabpwa_demo_model_plugin"
    (tmp_path / f"{name}.py").write_text(_MODEL_PLUGIN.format(name=name))
    monkeypatch.syspath_prepend(str(tmp_path))
    f = Fitter(_cfg(plugin=[name], amp_model=f"{name}_model"), backend="demo_stub")
    assert f.model.name == f"{name}_model"
    assert type(f.model).__name__ == "DemoAmp"
    from tabpwa.amp_model import AMPLITUDE_MODELS
    assert f.model.name in AMPLITUDE_MODELS      # the import is what registered it


def test_a_plugin_that_raises_at_import_is_reported(tmp_path, monkeypatch):
    name = "tabpwa_bad_plugin"
    (tmp_path / f"{name}.py").write_text("raise RuntimeError('boom')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ImportError, match=name):
        Fitter(_cfg(plugin=[name]), backend="numpy_pwa")
    with pytest.raises(TypeError, match="module name or a list"):
        load_plugins(5)
