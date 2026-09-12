"""integrated_pwa with a cuda_v5_pwa base: Gram norm == plain ΣwP, same NLL."""

import numpy as np
import pytest

from ampfit.config_loader import Config
from ampfit.backends import create_backend


def _data(ne, kc, seed):
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


def test_integrated_pwa_v5_matches_plain():
    cfg = Config("tests/config_pwa.yml")
    kc = cfg.build_all_index()
    params = _params(kc)
    data = _data(300, kc, 0)
    phsp = _data(500, kc, 2)

    try:
        plain = create_backend("cuda_v5_pwa", kc)
        intg = create_backend({"name": "integrated_pwa",
                               "base": "cuda_v5_pwa"}, kc)
    except RuntimeError as e:
        pytest.skip(f"no CUDA device: {e}")

    try:
        hp_p = plain.load_data(phsp)
        hd_p = plain.load_data(data)
        hp_i = intg.load_data(phsp)
        hd_i = intg.load_data(data)

        norm = float(plain.compute(params, hp_p, norm=None)[0])
        # integrated: norm=None, return_p=False -> fast Gram path
        norm_i = float(intg.compute(params, hp_i, norm=None,
                                    return_p=False)[0])
        assert abs(norm - norm_i) < 1e-6 * max(1.0, abs(norm))

        Qp, gp, _Pp = plain.compute(params, hd_p, norm=norm)
        Qi, gi, _Pi = intg.compute(params, hd_i, norm=norm)
        assert abs(Qp - Qi) < 1e-8 * max(1.0, abs(Qp))
        assert np.allclose(gi["ck"], gp["ck"], atol=1e-8)
        # integrated freezes m0/g0 grads (Gram pre-computed)
        assert np.allclose(gi["m0"], 0.0)
        assert np.allclose(gi["g0"], 0.0)
        assert abs(gi["norm"] - gp["norm"]) < 1e-6 * max(1.0,
                                                          abs(gp["norm"]))
    finally:
        intg.free()
        plain.free()
