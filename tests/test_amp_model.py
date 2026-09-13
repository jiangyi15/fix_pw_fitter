"""AmplitudeModel: model-specific kernel config & parameter transform."""

import os
import tempfile

import numpy as np
import pytest

from ampfit import Fitter
from ampfit.amp_model import (AmplitudeModel, PWA, FlavourTagMix,
                              build_amplitude_model, AMPLITUDE_MODELS)
from ampfit.kernel_params import (PWAKernelParams, FlavourTagMixKernelParams)
from ampfit.config_loader import Config

PWA_CFG = os.path.join(os.path.dirname(__file__), "config_pwa.yml")


def _with(text):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "c.yml")
    with open(p, "w") as fh:
        fh.write(text)
    return p


def test_default_model_is_pwa():
    cfg = Config(PWA_CFG)
    m = cfg.amplitude_model
    assert isinstance(m, PWA) and m.name == "pwa"
    assert m.scalar_names == []
    assert isinstance(m.build_params_transform(), PWAKernelParams)
    assert m.angle_formula == "helicity"
    assert cfg.scalar_names == [] and cfg.n_proj == m.n_proj


def test_legacy_model_adds_scalars():
    path = _with("amp_model: p4_directly\n\n" + open(PWA_CFG).read())
    cfg = Config(path)
    m = cfg.amplitude_model
    assert isinstance(m, FlavourTagMix) and m.name == "flavour_tag_mix"
    assert len(m.scalar_names) == 6
    assert isinstance(m.build_params_transform(), FlavourTagMixKernelParams)
    assert cfg.scalar_names == m.scalar_names


def test_unknown_model_raises():
    path = _with("amp_model: no_such_model\n\n" + open(PWA_CFG).read())
    with pytest.raises(ValueError, match="unknown amp_model"):
        Config(path)


def test_pwa_kernel_config_is_the_base_config():
    cfg = Config(PWA_CFG)
    base = cfg._build_base_kernel_config()
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
    """The model owns which backends are valid (integrated vs integrated_pwa)."""
    pwa = Config(PWA_CFG).amplitude_model
    assert pwa.supports_backend("integrated_pwa") and pwa.supports_backend("cuda_v4_pwa")
    assert not pwa.supports_backend("integrated")
    assert not pwa.supports_backend("cuda64")
    assert pwa.default_backend == "cuda_v4_pwa"

    legacy = Config("config_angle.yml").amplitude_model
    assert legacy.supports_backend("integrated") and legacy.supports_backend("cuda64")
    assert not legacy.supports_backend("integrated_pwa")
    assert legacy.default_backend == "cuda64"

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
    d = cfg.amplitude_model.build_params_transform().param_defaults()
    assert d["gamma"] == pytest.approx(0.123)
    assert d["delta_m"] == pytest.approx(0.506)
    assert d["poqr"] == pytest.approx(1.0)
    assert set(d) == set(cfg.scalar_names)


def test_fitter_resolves_model_default_backend(monkeypatch):
    import ampfit.backends as B
    seen = {}
    real = B.create_backend

    def fake(spec, kernel_config, **kw):
        seen["spec"] = spec
        return real(spec, kernel_config, **kw)

    monkeypatch.setattr(B, "create_backend", fake)
    f = Fitter(PWA_CFG)
    try:
        assert seen["spec"] == "cuda_v4_pwa"
    finally:
        f.backend.free()
    f = Fitter("config_angle.yml")
    try:
        assert seen["spec"] == "cuda64"
    finally:
        f.backend.free()


def test_backend_registry_is_consistent():
    from ampfit.backends import ALL_BACKENDS, BACKEND_MODELS
    from ampfit.amp_model import AMPLITUDE_MODELS
    canonical = {cls.name for cls in AMPLITUDE_MODELS.values()}
    assert set(BACKEND_MODELS) == set(ALL_BACKENDS)
    for models in BACKEND_MODELS.values():
        if models is not None:
            assert models <= canonical, models
