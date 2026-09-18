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



def test_amplitude_model_is_a_base_model():
    from ampfit.amp_model import AmplitudeModel, build_amplitude_model
    from ampfit.base_model import BaseModel
    from ampfit.config_loader import Config

    assert issubclass(AmplitudeModel, BaseModel)
    model = build_amplitude_model(Config(CONFIG))
    assert isinstance(model, BaseModel)
    # the policy object does not keep a reference to the raw Config
    assert not hasattr(model, "config")


def test_model_shares_the_config_interpretation():
    from ampfit.amp_model import build_amplitude_model
    from ampfit.config_loader import Config

    cfg = Config(CONFIG)
    model = build_amplitude_model(cfg)
    # single interpretation: shared references, no duplicated derivation
    assert model.decay_tree is cfg.decay_tree
    assert model.unique_bw is cfg.unique_bw
    assert model.m0_phys_name is cfg.m0_phys_name
    # lazily-filled index data is visible through both objects
    model.build_kernel_config()
    assert len(cfg.m0_phys_name) == len(model.m0_phys_name) > 0
    assert len(cfg.unique_bw) == len(model.unique_bw) > 0


def test_fitter_keeps_the_model():
    from ampfit import Fitter
    from ampfit.amp_model import AmplitudeModel

    f = Fitter(CONFIG, backend="numpy_pwa")
    try:
        assert isinstance(f.model, AmplitudeModel)
        assert f.model.decay_tree is f.config.decay_tree
        assert f.model.m0_phys_name is f.config.m0_phys_name
        assert f.all_comb == f.model.get_ck_map()
    finally:
        f.backend.free()


def test_display_split_tree_vs_params():
    """Decay/particle display on DecayTree; parameter labels on BaseModel."""
    from ampfit.base_model import BaseModel
    from ampfit.decay_tree import DecayTree

    for m in ("name_display_map", "display_decay", "display_chain"):
        assert hasattr(DecayTree, m), m
        assert not hasattr(BaseModel, m), m
    for m in ("display_g_ls", "display_g_lsbar", "display_a_total",
              "param_display"):
        assert hasattr(BaseModel, m), m
