"""ShardBackend end-to-end worker compute (numpy_pwa workers, no GPU)."""

import multiprocessing as _mp

import numpy as np

from ampfit.config_loader import Config
from ampfit.backends import create_backend

try:
    _mp.set_start_method("fork", force=True)
except (RuntimeError, ValueError):
    pass


def _arrays(ne, kc, seed):
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


def test_shard_numpy_matches_single():
    cfg = Config("config_pwa.yml")
    kc = cfg.build_all_index()
    params = _params(kc)
    data = _arrays(300, kc, 0)
    phsp = _arrays(600, kc, 2)

    single = create_backend("numpy_pwa", kc)
    shard = create_backend(
        {"name": "shard", "backends": ["numpy_pwa", "numpy_pwa"]}, kc)
    try:
        hs = single.load_data(data)
        hps = single.load_data(phsp)
        hd = shard.load_data(data)
        hpd = shard.load_data(phsp)

        norm_s = float(single.compute(params, hps, norm=None)[0])
        norm_d = float(shard.compute(params, hpd, norm=None)[0])
        assert abs(norm_s - norm_d) < 1e-6   # sums split across workers
        Q_s, g_s, P_s = single.compute(params, hs, norm=norm_s)
        Q_d, g_d, P_d = shard.compute(params, hd, norm=norm_s)

        assert abs(Q_s - Q_d) < 1e-9
        for key in ("ck", "m0", "g0", "norm"):
            assert np.allclose(np.asarray(g_d[key]), np.asarray(g_s[key]),
                               atol=1e-8), key
        # P rows are concatenated in worker (contiguous chunk) order
        assert np.allclose(P_d, P_s, atol=1e-8)

        # handle.free() drops the dataset but keeps the pool usable
        hd.free()
        hd2 = shard.load_data(data)
        Q2, _g2, _P2 = shard.compute(params, hd2, norm=norm_s)
        assert abs(Q2 - Q_s) < 1e-9
    finally:
        shard.free()
        single.free()
