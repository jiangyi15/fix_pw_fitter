"""cuda_v5_pwa extra edge cases: partial tail group, batch guard, g0 FD."""

import numpy as np
import pytest

from ampfit.config_loader import Config


def pwa_cfg():
    cfg = Config("config_pwa.yml")
    return cfg, cfg.build_all_index()


def _kernel(kc, batch_size=256, rsize=1):
    from ampfit.cuda._v5_pwa import CUDAKernelV5PWA as KV5
    try:
        return KV5(kc, batch_size=batch_size, resolution_size=rsize)
    except RuntimeError as e:
        pytest.skip(f"no CUDA device: {e}")


def _data(ne, seed=0, nang=3):
    rs = np.random.RandomState(seed)
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


def _group_nll(P, w, bkg, norm, rsize):
    """Reference: groups of size rsize; the LAST group may be partial."""
    n = len(P)
    n_groups = (n + rsize - 1) // rsize
    out = 0.0
    for g in range(n_groups):
        sl = slice(g * rsize, min((g + 1) * rsize, n))
        term = np.sum(w[sl] * (P[sl] / norm + bkg[sl]))
        out += -np.log(term)
    return out


def test_v5_partial_tail_group_matches_reference():
    cfg, kc = pwa_cfg()
    ne, rsize, norm = 300, 32, 5.0
    assert ne % rsize != 0                  # partial final group of 12
    k = _kernel(kc, batch_size=256, rsize=rsize)
    h = k.load_data(_data(ne, nang=int(kc["angle_k"].shape[-1])))
    try:
        params = _params(kc)
        Q, _g, P = k.compute(params, h, norm=norm)
        ref = _group_nll(P, np.ones(ne), np.ones(ne), norm, rsize)
        assert Q == pytest.approx(ref, rel=1e-9)
    finally:
        h.free()
        k.free()


def test_v5_batch_smaller_than_resolution_raises():
    cfg, kc = pwa_cfg()
    from ampfit.cuda._v5_pwa import CUDAKernelV5PWA as KV5
    try:
        with pytest.raises(ValueError):
            KV5(kc, batch_size=16, resolution_size=32)
    except RuntimeError as e:               # no GPU
        pytest.skip(f"no CUDA device: {e}")


def test_v5_g0_gradient_finite_difference():
    cfg, kc = pwa_cfg()
    ne, rsize = 120, 15
    k = _kernel(kc, batch_size=256, rsize=rsize)
    h = k.load_data(_data(ne, nang=int(kc["angle_k"].shape[-1])))
    try:
        params = _params(kc)
        norm = float(k.compute(params, h, norm=None)[0])
        _Q, g, _P = k.compute(params, h, norm=norm)

        def f(p):
            return k.compute(p, h, norm=norm)[0]

        eps = 1e-6
        idx = 1
        gp = dict(params, g0=params["g0"].copy())
        gm = dict(params, g0=params["g0"].copy())
        gp["g0"][idx] += eps
        gm["g0"][idx] -= eps
        num = (f(gp) - f(gm)) / (2 * eps)
        assert num == pytest.approx(g["g0"][idx], rel=1e-3, abs=1e-2)
    finally:
        h.free()
        k.free()
