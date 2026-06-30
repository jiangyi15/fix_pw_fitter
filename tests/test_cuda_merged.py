#!/usr/bin/env python3
"""Test merged-index CUDA backend against reference NumPy."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from ampfit.config_loader import Config
from ampfit.backends import create_backend

CONFIG_FILE = "config_angle.yml"

config = Config(CONFIG_FILE)
kernel_config = config.build_all_index()
ck_map = config.get_ck_map()

rng = np.random.default_rng()
params = {
    'ck': rng.normal(size=len(ck_map)) + 1j * rng.normal(size=len(ck_map)),
    'm0': rng.random(len(kernel_config["m0_index"])) + 2,
    'g0': rng.random(len(kernel_config["g0_index"])) + 0.1,
    'scalar': [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
}

for n in [64, 128, 256]:
    data = {
        'mass': rng.random((n, 48)),
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
    Q_r, grads_r, P_r = np_be.compute(params, dh, norm=None)

    # Merged CUDA
    cuda_be = create_backend("cuda_v3", kernel_config)
    dh_cu = cuda_be.load_data(data)
    Q_t, grads_t, P_t = cuda_be.compute(params, dh_cu)

    assert abs(Q_r - Q_t) < 1e-12, f"Q mismatch: {Q_r} vs {Q_t}"
    assert np.max(np.abs(P_r - P_t)) < 1e-12, "P mismatch"
    for key in ['ck', 'm0', 'g0', 'scalar']:
        g_ref = np.asarray(grads_r[key])
        g_test = np.asarray(grads_t[key])
        rel = np.max(np.abs(g_test - g_ref)) / (np.max(np.abs(g_ref)) + 1e-30)
        assert rel < 1e-12, f"grad_{key} mismatch: rel={rel:.2e}"
    print(f"  n={n:>4}: ✓ All gradients match")

    dh_cu.free()
    cuda_be.free()

print("✓ CUDA merged verified!")
