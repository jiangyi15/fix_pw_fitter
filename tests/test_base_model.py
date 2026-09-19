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


def test_config_is_a_legacy_alias_for_the_model():
    from tabpwa.amp_model import AmplitudeModel
    from tabpwa.config_loader import Config, RawConfig

    model = Config(CONFIG)                     # legacy alias
    assert isinstance(model, AmplitudeModel)
    assert model.name == "pwa"
    raw = RawConfig(CONFIG)
    assert not hasattr(raw, "topo_index")      # raw input has no physics
    assert model.topo_index is not None


def test_base_model_builds_from_raw_dict_without_config():
    from tabpwa.base_model import BaseModel
    from tabpwa.config_loader import Config, RawConfig, load_config

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
    assert RawConfig(CONFIG).backend_spec is None


def test_base_model_module_is_leaf():
    """base_model must not import config_loader / amp_model / backends."""
    from tabpwa import base_model

    mods = _imports(base_model)
    assert not any("config_loader" in m for m in mods), mods
    assert not any("amp_model" in m for m in mods), mods
    assert not any("backends" in m for m in mods), mods


def test_base_model_carries_no_model_policy():
    """Model policy belongs to AmplitudeModel, not BaseModel."""
    from tabpwa.base_model import BaseModel

    for attr in ("name", "params_transform_cls", "scalar_names",
                 "scalar_defaults", "data_layout", "build_params_transform"):
        assert not hasattr(BaseModel, attr), attr



def test_amplitude_model_is_a_base_model():
    from tabpwa.amp_model import AmplitudeModel, build_amplitude_model
    from tabpwa.base_model import BaseModel
    from tabpwa.config_loader import Config

    assert issubclass(AmplitudeModel, BaseModel)
    model = build_amplitude_model(Config(CONFIG))
    assert isinstance(model, BaseModel)
    # the policy object does not keep a reference to the raw Config
    assert not hasattr(model, "config")


def test_config_alias_and_factory_agree():
    from tabpwa.amp_model import build_amplitude_model
    from tabpwa.config_loader import Config

    a = Config(CONFIG)                      # legacy alias
    b = build_amplitude_model(CONFIG)       # explicit factory
    assert a.topo_index == b.topo_index
    assert a.n_proj == b.n_proj


def test_fitter_keeps_the_model():
    from tabpwa import Fitter
    from tabpwa.amp_model import AmplitudeModel
    from tabpwa.config_loader import RawConfig

    f = Fitter(CONFIG, backend="numpy_pwa")
    try:
        assert isinstance(f.config, RawConfig)       # raw input only
        assert isinstance(f.model, AmplitudeModel)   # the model
        assert f.all_comb == f.model.get_ck_map()
    finally:
        f.backend.free()


def test_display_split_tree_vs_params():
    """Decay/particle display on DecayTree; parameter labels on BaseModel."""
    from tabpwa.base_model import BaseModel
    from tabpwa.decay_tree import DecayTree

    for m in ("name_display_map", "display_decay", "display_chain"):
        assert hasattr(DecayTree, m), m
        assert not hasattr(BaseModel, m), m
    for m in ("display_g_ls", "display_g_lsbar", "display_a_total",
              "param_display"):
        assert hasattr(BaseModel, m), m
