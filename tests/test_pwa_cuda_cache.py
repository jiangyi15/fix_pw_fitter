"""GPU regression: cuda_v4_pwa_cache == cuda_v4_pwa at fixed m0/g0."""

import numpy as np
import pytest

from ampfit.config_loader import Config


@pytest.fixture(scope="module")
def pwa_small():
    cfg = Config("config_pwa.yml")
    kc = cfg.build_all_index()
    n_proj = int(kc.get("n_proj", 1))
    n_base = kc["matrix_angle"].shape[1] // n_proj
    n_m0 = int(np.max(kc["m0_index"])) + 1
    n_g0 = int(np.max(kc["g0_index"])) + 1
    rs = np.random.RandomState(0)
    ne = 300
    data = {"mass": 2.2 + 0.5 * rs.randn(ne, 1),
            "q": np.abs(rs.randn(ne, 2)) + 0.3,
            "angle": rs.uniform(-np.pi, np.pi,
                                size=(ne, 1, int(kc["angle_k"].shape[-1]))),
            "weight": np.ones(ne), "bkg": np.ones(ne)}
    ck = rs.randn(n_base) + 1j * rs.randn(n_base)
    m0 = 0.7 + 0.4 * np.abs(rs.randn(n_m0))
    g0 = 0.1 + 0.3 * np.abs(rs.randn(n_g0))
    return cfg, kc, data, {"ck": ck, "m0": m0, "g0": g0}, n_proj


def _make(kernel_cls, kc):
    try:
        return kernel_cls(kc, batch_size=128)
    except RuntimeError as e:
        pytest.skip(f"no CUDA device: {e}")


def test_v4_pwa_cache_matches_v4_pwa_fixed(pwa_small):
    from ampfit.cuda._v4_pwa import CUDAKernelV4PWA as KV4
    from ampfit.cuda._v4_pwa_cache import CUDAKernelV4PWACache as KC
    _cfg, kc, data, params, _p = pwa_small

    kv, kc_ = _make(KV4, kc), None
    try:
        kc_ = KC(kc, batch_size=128, m0=params["m0"], g0=params["g0"])
    except RuntimeError as e:
        pytest.skip(f"no CUDA device: {e}")
    try:
        hv = kv.load_data(data)
        hc = kc_.load_data(data)   # cache built at load with fixed m0/g0
        try:
            q_none = None
            for norm in (None, float(len(data["weight"]))):
                Qv, gv, Pv = kv.compute(params, hv, norm=norm)
                Qc, gc, Pc = kc_.compute(params, hc, norm=norm)
                assert Qv == pytest.approx(Qc, abs=1e-9)
                assert np.max(np.abs(Pv - Pc)) < 1e-9
                assert np.max(np.abs(gv["ck"] - gc["ck"])) < 1e-8
                assert np.allclose(gc["m0"], 0.0)
                assert np.allclose(gc["g0"], 0.0)
                if norm is None:
                    q_none = Qc
            # repeated call at unchanged m0/g0 hits the cache path
            Qr, _gr, _Pr = kc_.compute(params, hc, norm=None)
            assert Qr == pytest.approx(q_none, abs=1e-9)
        finally:
            hv.free()
            hc.free()
    finally:
        kv.free()
        kc_.free()
