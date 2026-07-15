#!/usr/bin/env python3
"""Test merged-index CUDA backend against reference NumPy."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pytest
from ampfit.config_loader import Config
from ampfit.backends import create_backend

CONFIG_FILE = "config_angle.yml"


@pytest.fixture(scope="module")
def setup():
    config = Config(CONFIG_FILE)
    kernel_config = config.build_all_index()
    ck_map = config.get_ck_map()
    rng = np.random.default_rng()
    params = {
        'ck': rng.normal(size=len(ck_map)) + 1j * rng.normal(size=len(ck_map)),
        'm0': rng.random(int(np.max(kernel_config["m0_index"])) + 1) + 2,
        'g0': rng.random(int(np.max(kernel_config["g0_index"])) + 1) + 0.1,
        'scalar': [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }
    return kernel_config, params


@pytest.mark.parametrize("n", [32, 64])
def test_cuda_vs_numpy(n, setup):
    """Compare CUDA v3 against NumPy at various batch sizes."""
    kernel_config, params = setup
    rng = np.random.default_rng(42)
    data = {
        'mass': rng.uniform(2, 3, (n, 48)),
        'q': rng.random((n, 72)),
        'angle': rng.random((n, 24, 3)),
        'frac': rng.random((n,)),
        'time': rng.random((n,)),
        'bkg': rng.random((n,)) * 0.01,
        'weight': np.ones((n,)),
    }

    # NumPy reference
    np_be = create_backend("numpy", kernel_config)
    dh = np_be.load_data(data)
    Q_r, grads_r, P_r = np_be.compute(params, dh)

    # CUDA
    cuda_be = create_backend({"name": "cuda_v3", "batch_size": 2000}, kernel_config)
    dh_cu = cuda_be.load_data(data)
    Q_t, grads_t, P_t = cuda_be.compute(params, dh_cu)

    assert abs(Q_r - Q_t) < 1e-11, f"Q mismatch: {Q_r} vs {Q_t}"
    assert np.max(np.abs(P_r - P_t)) < 1e-11, "P mismatch"
    for key in ['ck', 'm0', 'g0', 'scalar']:
        g_ref = np.asarray(grads_r[key])
        g_test = np.asarray(grads_t[key])
        rel = np.max(np.abs(g_test - g_ref)) / (np.max(np.abs(g_ref)) + 1e-30)
        assert rel < 1e-11, f"grad_{key} mismatch: rel={rel:.2e}"

    dh_cu.free()
    cuda_be.free()
    print(f"  n={n:>4}: ✓ Q={Q_r:.6f}")
