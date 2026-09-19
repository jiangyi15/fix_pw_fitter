"""AmplitudeModel: model-specific kernel config & parameter transform."""

import os
import tempfile

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.amp_model import (AmplitudeModel, PWA, FlavourTagMix,
                              build_amplitude_model, AMPLITUDE_MODELS)
from tabpwa.kernel_params import (PWAKernelParams, FlavourTagMixKernelParams)
from tabpwa.config_loader import Config

PWA_CFG = os.path.join(os.path.dirname(__file__), "config_pwa.yml")


def _with(text):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "c.yml")
    with open(p, "w") as fh:
        fh.write(text)
    return p


def test_default_model_is_pwa():
    cfg = Config(PWA_CFG)
    m = build_amplitude_model(cfg)
    assert isinstance(m, PWA) and m.name == "pwa"
    assert m.scalar_names == []
    assert isinstance(m.build_params_transform(), PWAKernelParams)
    assert cfg.angle_formula_mode == "helicity" and cfg.n_proj >= 1
    # the raw input holds no model instance / scalar mirror
    from tabpwa.config_loader import RawConfig
    raw = RawConfig(PWA_CFG)
    assert not hasattr(raw, "amplitude_model") and not hasattr(raw, "scalar_names")


def test_legacy_model_adds_scalars():
    path = _with("amp_model: p4_directly\n\n" + open(PWA_CFG).read())
    cfg = Config(path)
    m = build_amplitude_model(cfg)
    assert isinstance(m, FlavourTagMix) and m.name == "flavour_tag_mix"
    assert len(m.scalar_names) == 6
    assert isinstance(m.build_params_transform(), FlavourTagMixKernelParams)


def test_unknown_model_raises():
    path = _with("amp_model: no_such_model\n\n" + open(PWA_CFG).read())
    with pytest.raises(ValueError, match="unknown amp_model"):
        build_amplitude_model(Config(path))


def test_pwa_kernel_config_is_the_base_config():
    cfg = Config(PWA_CFG)
    base = cfg.build_base_kernel_config()
    out = cfg.build_all_index()
    assert set(out) == set(base)
    for k in base:
        a, b = out[k], base[k]
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b), k
        else:
            assert a == b, k


def test_pwa_params_have_no_scalar_key():
    f = Fitter(PWA_CFG, backend="numpy_pwa")
    try:
        x = f.initial_values(seed=0)
        params, _ = f.build_params(x)
        assert set(params) == {"ck", "m0", "g0"}
        assert f.param_defaults() == {}
        assert "scalar" not in f.param_names()
    finally:
        f.backend.free()


def test_legacy_params_have_scalar_key():
    f = Fitter("config_amp.yml", backend="numpy")
    try:
        x = f.initial_values(seed=0)
        params, _ = f.build_params(x)
        assert "scalar" in params and len(params["scalar"]) == 6
        assert len(f.param_defaults()) == 6
        assert "gamma" in f.param_names()
    finally:
        f.backend.free()


def test_registry_and_factory():
    assert "pwa" in AMPLITUDE_MODELS
    cfg = Config(PWA_CFG)
    assert isinstance(build_amplitude_model(cfg), AmplitudeModel)


def test_ck_index_helpers_respect_row_blocks():
    """_expand_to_blocks must use n_perm·n_cp, not a literal 8."""
    cfg = Config(PWA_CFG)
    n = len(cfg.get_ck_map())
    idx = cfg.get_ck_indices(["jpsi"])
    assert idx and max(idx) < n, (n, max(idx))
    assert cfg.build_all_index()["n_blocks"] == 1

    legacy = Config("config_angle.yml")
    n_leg = len(legacy.get_ck_map())
    assert legacy.build_all_index()["n_blocks"] == 8
    assert max(legacy.get_decay_ck_indices([("a1(1260)p", "rhoA")])) < n_leg


def test_backend_registry_gates_backends_per_model():
    """Names are registered per model; "default" is a per-model name."""
    from tabpwa.backends import backends_for_model, backend_class

    pwa = backends_for_model("pwa")
    assert {"integrated_pwa", "cuda_v4_pwa", "default"} <= pwa
    assert "integrated" not in pwa and "cuda64" not in pwa
    assert "cuda" not in pwa                       # legacy-only alias

    legacy = backends_for_model("flavour_tag_mix")
    assert {"integrated", "cuda64", "cuda", "default"} <= legacy
    assert "integrated_pwa" not in legacy

    # per-model default resolves to different classes
    assert backend_class("default", "pwa").__name__ == "CUDABackendV4PWA"
    assert backend_class("default", "flavour_tag_mix").__name__ == "CUDABackendV3"

    # selection-time rejection through the Fitter
    with pytest.raises(ValueError, match="not registered"):
        Fitter(PWA_CFG, backend="integrated")
    with pytest.raises(ValueError, match="not registered"):
        Fitter("config_angle.yml", backend="integrated_pwa")


def test_partial_scalar_defaults_keep_legacy_fallbacks():
    """A partial scalar_defaults override must not zero the other scalars."""
    base = open("config_angle.yml").read()
    path = _with(base + "\nscalar_defaults: {gamma: 0.123}\n")
    cfg = Config(path)
    d = build_amplitude_model(cfg).build_params_transform().param_defaults()
    assert d["gamma"] == pytest.approx(0.123)
    assert d["delta_m"] == pytest.approx(0.506)
    assert d["poqr"] == pytest.approx(1.0)
    assert set(d) == set(build_amplitude_model(cfg).scalar_names)


def test_fitter_uses_registered_default_backend(monkeypatch):
    """With no explicit/config backend, Fitter passes the name 'default'."""
    import tabpwa.backends as B
    seen = {}
    real = B.create_backend

    def fake(spec, kernel_config, *, model=None, **kw):
        seen["spec"], seen["model"] = spec, model
        return real(spec, kernel_config, model=model, **kw)

    monkeypatch.setattr(B, "create_backend", fake)
    f = Fitter(PWA_CFG)
    try:
        assert seen == {"spec": "default", "model": "pwa"}
        assert type(f.backend).__name__ == "CUDABackendV4PWA"
    finally:
        f.backend.free()
    f = Fitter("config_angle.yml")
    try:
        assert seen == {"spec": "default", "model": "flavour_tag_mix"}
        assert type(f.backend).__name__ == "CUDABackendV3"
    finally:
        f.backend.free()


def test_backend_registry_is_consistent():
    from tabpwa.backends import MODEL_BACKENDS, UNIVERSAL_BACKENDS, backend_class
    from tabpwa.amp_model import AMPLITUDE_MODELS
    canonical = {cls.name for cls in AMPLITUDE_MODELS.values()}
    assert set(MODEL_BACKENDS) <= canonical          # only real model names
    assert "default" in MODEL_BACKENDS["pwa"]
    assert "default" in MODEL_BACKENDS["flavour_tag_mix"]
    assert "shard" in UNIVERSAL_BACKENDS
    # a per-model name must not be resolvable without a model
    with pytest.raises(ValueError, match="several amplitude models"):
        backend_class("default")


def test_model_without_registered_backends_raises_clearly():
    from tabpwa.amp_model import AmplitudeModel, register_amplitude_model

    @register_amplitude_model("_no_backends")
    class NoBackends(AmplitudeModel):
        name = "_no_backends"

    path = _with("amp_model: _no_backends\n" + open(PWA_CFG).read())
    with pytest.raises(ValueError, match="not registered"):
        Fitter(path)


def test_empty_top_level_amp_model_falls_back_to_data():
    """amp_model: {} / [] / '' must not shadow data.amp_model."""
    base = open("config_angle.yml").read()
    for empty in ("{}", "[]", '""'):
        cfg = Config(_with(f"amp_model: {empty}\n" + base))
        assert build_amplitude_model(cfg).name == "flavour_tag_mix", empty


def test_explicit_top_level_amp_model_wins_over_data():
    base = open("config_angle.yml").read()      # data.amp_model = flavour_tag_mix
    cfg = Config(_with("amp_model: pwa\n" + base))
    assert build_amplitude_model(cfg).name == "pwa"


def test_param_names_match_legacy_flat_order():
    cfg = Config("config_amp.yml")
    bases = sorted({p for comb in cfg.get_ck_map()
                    for p in comb if isinstance(p, str)})
    legacy = list(dict.fromkeys(
        sorted([n + "r" for n in bases] + [n + "i" for n in bases])
        + list(cfg.m0_phys_name) + list(cfg.g0_phys_name)
        + list(build_amplitude_model(cfg).scalar_names)))
    assert build_amplitude_model(cfg).build_params_transform().param_names() == legacy


def test_param_names_are_unique_with_colliding_scalar():
    base = open("config_amp.yml").read()
    cfg = Config("config_amp.yml")
    ck_name = sorted({p for comb in cfg.get_ck_map()
                      for p in comb if isinstance(p, str)})[0] + "r"
    path = _with(base + f"\nscalar_names: [gamma, {ck_name}]\n")
    names = build_amplitude_model(Config(path)).build_params_transform().param_names()
    assert len(names) == len(set(names))


def test_integrated_backend_requires_legacy_block_structure():
    from tabpwa.backends.integrated_backend import IntegratedBackend

    kc_pwa = Config(PWA_CFG).build_all_index()      # n_blocks = 1
    with pytest.raises(ValueError, match="integrated_pwa"):
        IntegratedBackend(kc_pwa, base="numpy_pwa")

    kc = Config("config_angle.yml").build_all_index()
    be = IntegratedBackend(kc, base="numpy")
    assert be._n_blocks == 8 and be._n_perm == 4
    assert be._ng == be.kernel.n_wave // be._n_blocks
    assert len(be._groups_B0[0]) == be._n_perm      # 4 identical-particle copies


def test_invalid_backend_fails_before_kernel_config(monkeypatch):
    """Backend validation must precede the kernel-config build."""
    import tabpwa.config_loader as cl

    def _boom(self):
        raise AssertionError("build_all_index must not be called")

    monkeypatch.setattr(cl.BaseModel, "build_base_kernel_config", _boom)
    with pytest.raises(ValueError, match="not registered"):
        Fitter(PWA_CFG, backend="integrated")


def test_amp_model_module_does_not_import_backends():
    import tabpwa.amp_model as am
    src = open(am.__file__).read()
    assert "tabpwa.backends" not in src


def test_raw_config_has_no_model():
    """RawConfig is pure raw input; the model is built from it."""
    import tabpwa.config_loader as cl
    raw = cl.RawConfig(PWA_CFG)
    assert not hasattr(raw, "amplitude_model")
    assert not hasattr(raw, "scalar_names")
    assert cl.Config(PWA_CFG).name == "pwa"


def test_create_backend_injects_model_into_composer():
    """A backend whose __init__ takes ``model`` receives it (nested default)."""
    from tabpwa.backends import create_backend, register_backend

    @register_backend("_model_probe")
    class Probe:
        def __init__(self, kernel_config, model=None, tag=None):
            self.model, self.tag = model, tag

    p = create_backend({"name": "_model_probe", "tag": 7}, {}, model="pwa")
    assert p.model == "pwa" and p.tag == 7


def test_integrated_threads_model_to_nested_base(monkeypatch):
    import tabpwa.backends as B
    seen = {}
    real = B.create_backend

    def spy(spec, kernel_config, *, model=None, **kw):
        seen["spec"], seen["model"] = spec, model
        return real(spec, kernel_config, model=model, **kw)

    monkeypatch.setattr(B, "create_backend", spy)
    from tabpwa.backends.integrated_backend import IntegratedBackend
    kc = Config("config_angle.yml").build_all_index()
    IntegratedBackend(kc, base="numpy", model="flavour_tag_mix")
    assert seen == {"spec": "numpy", "model": "flavour_tag_mix"}


def test_shard_records_model_for_workers():
    from tabpwa.backends.shard_backend import ShardBackend
    kc = Config(PWA_CFG).build_all_index()
    be = ShardBackend(kc, backends=["default"], model="pwa")
    assert be._model == "pwa" and be._specs == [("default", None)]


def test_initial_values_use_default():
    """Defaulted params start at their default; use_default=False is random."""
    f = Fitter("config_angle.yml", backend="numpy")
    try:
        names = f.cm.var_registry.flat_names
        defaults = f.cm.defaults
        x = f.initial_values(seed=3)
        raw = f.cm.var_registry.build_initial(seed=3)   # same draw, no defaults

        assert any(n in defaults for n in names), "expected defaulted params"
        for i, n in enumerate(names):
            if n not in defaults:
                assert x[i] == raw[i], n               # ck etc. unchanged

        _, resolved = f.build_params(x)
        for n in defaults:
            if n in names:
                assert resolved[n] == pytest.approx(defaults[n], rel=1e-9), n

        # opt-out restores the all-random start
        x_rand = f.initial_values(seed=3, use_default=False)
        assert all(x_rand[i] == raw[i] for i in range(len(names)))
    finally:
        f.backend.free()


def test_model_builds_event_data_per_case():
    """PWA -> tree (+blocks, no frac/time); FlavourTagMix -> 24-row + frac/time."""
    import numpy as np
    from tabpwa.config_loader import Config
    from tabpwa.amp_model import build_amplitude_model

    def _mom(n, masses, seed=0):
        rs = np.random.RandomState(seed)
        p3 = rs.uniform(-0.4, 0.4, size=(n, len(masses), 3))
        out = np.empty((n, len(masses), 4))
        for j, m in enumerate(masses):
            E = np.sqrt(m ** 2 + (p3[:, j] ** 2).sum(-1))
            out[:, j] = np.concatenate([E[:, None], p3[:, j]], axis=-1)
        return out

    m = build_amplitude_model("tests/config_pwa.yml")     # path directly
    d = m.build_event_data(_mom(5, [0.13957, 0.13957, 0.54786]),
                           m.build_kernel_config())
    assert {"mass", "q", "angle", "weight"} <= set(d)
    assert "frac" not in d and "time" not in d

    ml = build_amplitude_model("config_amp.yml")          # path directly
    dl = ml.build_event_data(_mom(5, [0.13957] * 4),
                             ml.build_kernel_config())
    assert "frac" in dl and "time" in dl
    assert dl["angle"].shape[1] == 24


def test_build_amplitude_model_accepts_path_config_dict():
    from tabpwa.config_loader import Config, load_config
    from tabpwa.amp_model import build_amplitude_model

    m1 = build_amplitude_model("tests/config_pwa.yml")        # path
    m2 = build_amplitude_model(Config("tests/config_pwa.yml"))  # Config
    m3 = build_amplitude_model(load_config("tests/config_pwa.yml"))  # dict
    assert all(type(m).__name__ == "PWA" for m in (m1, m2, m3))
