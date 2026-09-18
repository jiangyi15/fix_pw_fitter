"""BaseModel extraction: Config is a thin raw-input subclass."""

import ast
import inspect
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CONFIG = os.path.join(ROOT, "tests", "config_pwa.yml")


def _imports(module):
    mods = []
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
    return mods


def test_config_subclasses_base_model():
    from ampfit.base_model import BaseModel
    from ampfit.config_loader import Config

    assert issubclass(Config, BaseModel)
    assert isinstance(Config(CONFIG), BaseModel)


def test_base_model_builds_from_raw_dict_without_config():
    from ampfit.base_model import BaseModel
    from ampfit.config_loader import Config, load_config

    dic = load_config(CONFIG)
    base = BaseModel(dic, CONFIG)
    cfg = Config(CONFIG)
    assert base.topo_index == cfg.topo_index
    assert base.n_proj == cfg.n_proj
    assert base.angle_formula_mode == cfg.angle_formula_mode
    assert set(base.build_base_kernel_config()) == \
        set(cfg.build_base_kernel_config())
    # raw-input state lives on Config, not BaseModel
    assert not hasattr(base, "backend_spec")
    assert cfg.backend_spec == Config(CONFIG).backend_spec


def test_base_model_module_is_leaf():
    """base_model must not import config_loader / amp_model / backends."""
    from ampfit import base_model

    mods = _imports(base_model)
    assert not any("config_loader" in m for m in mods), mods
    assert not any("amp_model" in m for m in mods), mods
    assert not any("backends" in m for m in mods), mods


def test_base_model_carries_no_model_policy():
    """Model policy belongs to AmplitudeModel, not BaseModel."""
    from ampfit.base_model import BaseModel

    for attr in ("name", "params_transform_cls", "scalar_names",
                 "scalar_defaults", "data_layout", "build_params_transform"):
        assert not hasattr(BaseModel, attr), attr
