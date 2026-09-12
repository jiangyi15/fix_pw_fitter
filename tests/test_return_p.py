"""return_p=False must not round-trip per-event P (and still give dnorm)."""

import numpy as np

from ampfit.config_loader import Config
from ampfit.backends import create_backend


def _data(ne, kc, seed=0):
    rs = np.random.RandomState(seed)
    nang = int(kc["angle_k"].shape[-1])
    return {"mass": 2.2 + 0.5 * rs.randn(ne, 1),
            "q": np.abs(rs.randn(ne, 2)) + 0.3,
            "angle": rs.uniform(-np.pi, np.pi, size=(ne, 1, nang)),
            "weight": np.ones(ne), "bkg": np.ones(ne)}


def _params(kc, seed=1):
    rs = np.random.RandomState(seed)
    n_proj = int(kc.get("n_proj", 1))
    N = kc["matrix_angle"].shape[1] // n_proj
    n_m0 = int(np.max(kc["m0_index"])) + 1
    n_g0 = int(np.max(kc["g0_index"])) + 1
    return {"ck": rs.randn(N) + 1j * rs.randn(N),
            "m0": 0.7 + 0.4 * np.abs(rs.randn(n_m0)),
            "g0": 0.1 + 0.3 * np.abs(rs.randn(n_g0))}


def test_numpy_pwa_return_p_false():
    cfg = Config("tests/config_pwa.yml")
    kc = cfg.build_all_index()
    params = _params(kc)
    be = create_backend("numpy_pwa", kc)
    try:
        hp = be.load_data(_data(400, kc, 2))
        hd = be.load_data(_data(300, kc, 0))
        norm = float(be.compute(params, hp, norm=None)[0])
        Q0, g0, P0 = be.compute(params, hd, norm=norm, return_p=False)
        assert P0 is None
        assert "norm" in g0 and np.isfinite(g0["norm"])
        Q1, _g1, P1 = be.compute(params, hd, norm=norm, return_p=True)
        assert P1 is not None and P1.shape[0] == 300
        assert abs(Q0 - Q1) < 1e-9
    finally:
        be.free()


def test_cuda_v4_pwa_return_p_false():
    cfg = Config("tests/config_pwa.yml")
    kc = cfg.build_all_index()
    params = _params(kc)
    try:
        be = create_backend("cuda_v4_pwa", kc)
    except RuntimeError as e:                 # no GPU
        import pytest
        pytest.skip(f"no CUDA device: {e}")
    try:
        hp = be.load_data(_data(400, kc, 2))
        hd = be.load_data(_data(300, kc, 0))
        norm = float(be.compute(params, hp, norm=None)[0])
        Q0, g0, P0 = be.compute(params, hd, norm=norm, return_p=False)
        assert P0 is None
        assert np.isfinite(g0["norm"])
        Q1, _g1, P1 = be.compute(params, hd, norm=norm, return_p=True)
        assert P1 is not None and P1.shape[0] == 300
        assert abs(Q0 - Q1) < 1e-9
    finally:
        be.free()
