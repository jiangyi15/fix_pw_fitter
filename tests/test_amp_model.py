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
