#!/usr/bin/env python3
"""GPU test for the cuda_v3_ampcache backend (cached angular amplitudes).

The per-wave amplitude is a_w = ck_w·Amp_w(q,angles)/bw_p_w(m); Amp is a
pure-kinematics cache filled on the first compute() and reused, while the
BW propagator is recomputed every call (m0/g0 float).  These tests check
that Q/grads/P match the NumPy reference and that the cache stays valid
across calls with varying m0/g0.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from ampfit.config_loader import Config
from ampfit.backends import create_backend

CONFIG = "config_angle.yml"


def _require_cuda():
    try:
        from ampfit.cuda._v3_ampcache import CUDAKernelV3AmpCache
        lib = CUDAKernelV3AmpCache.__module__ and None
    except Exception as e:
        pytest.skip(f"cuda_v3_ampcache unavailable: {e}")
    try:
        k = create_backend("cuda_v3_ampcache", Config(CONFIG).build_all_index())
    except Exception as e:
        pytest.skip(f"no CUDA device / build failure: {e}")
    return k


def _params(config, rng):
    kc = config.build_all_index()
    n_ck = len(config.get_ck_map())
    return {
        'ck': rng.normal(size=n_ck) + 1j * rng.normal(size=n_ck),
        'm0': rng.random(int(np.max(kc["m0_index"])) + 1) + 2,
        'g0': rng.random(int(np.max(kc["g0_index"])) + 1) + 0.1,
        'scalar': [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }


def _data(n, seed):
    rng = np.random.default_rng(seed)
    return {
        'mass': rng.uniform(2, 3, (n, 48)),
        'q': rng.random((n, 72)),
        'angle': rng.random((n, 24, 3)),
        'frac': rng.random(n),
        'time': rng.random(n),
        'weight': np.ones(n),
        'bkg': rng.random(n) * 0.01,
    }


def test_cuda_v3_ampcache_matches_numpy():
    config = Config(CONFIG)
    kc = config.build_all_index()
    backend = _require_cuda()
    numpy_backend = create_backend("numpy", kc)

    for n in [1, 63, 300, 1500]:          # exercises batched + partial batches
        rng = np.random.default_rng(n)
        params = _params(config, rng)
        data = _data(n, seed=n)
        np_handle = numpy_backend.load_data(data)
        Q_np, grads_np, P_np = numpy_backend.compute(params, np_handle, norm=42.0)
        handle = backend.load_data(data)
        Q, grads, P = backend.compute(params, handle, norm=42.0)

        assert abs(Q - Q_np) < 1e-6 * max(1.0, abs(Q_np)), f"n={n} Q mismatch"
        assert np.max(np.abs(P - P_np)) < 1e-6, f"n={n} P mismatch"
        for key in ['ck', 'm0', 'g0']:
            rel = np.max(np.abs(grads[key] - grads_np[key])) / (
                np.max(np.abs(grads_np[key])) + 1e-30)
            assert rel < 1e-5, f"n={n} grad {key} rel {rel:.2e}"
        handle.free()

    backend.free()


def test_cuda_v3_ampcache_cache_reuse_with_varying_m0():
    """Cache is built once; later calls with different m0/g0/ck still match
    the reference (proves the BW part is recomputed per call)."""
    config = Config(CONFIG)
    kc = config.build_all_index()
    backend = _require_cuda()
    numpy_backend = create_backend("numpy", kc)

    data = _data(2000, seed=5)
    np_handle = numpy_backend.load_data(data)
    handle = backend.load_data(data)

    for i in range(3):
        rng = np.random.default_rng(100 + i)
        params = _params(config, rng)
        Q_np, grads_np, P_np = numpy_backend.compute(params, np_handle, norm=42.0)
        Q, grads, P = backend.compute(params, handle, norm=42.0)
        assert abs(Q - Q_np) < 1e-6 * max(1.0, abs(Q_np)), f"iter {i} Q mismatch"
        for key in ['ck', 'm0', 'g0']:
            rel = np.max(np.abs(grads[key] - grads_np[key])) / (
                np.max(np.abs(grads_np[key])) + 1e-30)
            assert rel < 1e-5, f"iter {i} grad {key} rel {rel:.2e}"

    # norm=None path (phsp-style) also uses the cache consistently
    params = _params(config, np.random.default_rng(7))
    Q_np, _, _ = numpy_backend.compute(params, np_handle, norm=None)
    Q, _, _ = backend.compute(params, handle, norm=None)
    assert abs(Q - Q_np) < 1e-6 * max(1.0, abs(Q_np))

    handle.free()
    backend.free()


if __name__ == "__main__":
    test_cuda_v3_ampcache_matches_numpy()
    test_cuda_v3_ampcache_cache_reuse_with_varying_m0()
    print("\nAll cuda_v3_ampcache GPU tests passed!")
