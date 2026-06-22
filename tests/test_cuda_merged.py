#!/usr/bin/env python3
"""Test CUDAMergedKernel against reference NumPy kernel."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from ampfit.config_loader import Config
from ampfit.numpy_kernel import NumpyKernel
from ampfit._cuda_merged import CUDAMergedKernel

CONFIG_FILE = "config_angle.yml"
N_EVENTS = 64

np.random.seed(42)
config = Config(CONFIG_FILE)
kernel_config = config.build_all_index()
ck_map = config.get_ck_map()

params = {
    'ck': np.random.randn(len(ck_map)) + 1j*np.random.randn(len(ck_map)),
    'm0': np.random.rand(len(config.m0_phys_name)) + 2,
    'g0': np.random.rand(len(config.g0_phys_name)) + 0.1,
    'scalar': [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
}

for n in [N_EVENTS, 128, 256]:
    np.random.seed(42)
    data = {
        'mass': np.random.random((n, 48)),
        'q': np.random.random((n, 72)),
        'angle': np.random.random((n, 24, 3)),
        'frac': np.random.random((n,)),
        'time': np.random.random((n,)),
        'bkg': np.random.random((n,)) * 0.01,
        'weight': np.ones((n,)),
    }

    ref = NumpyKernel(kernel_config)
    Q_r, grads_r, P_r = ref._compute(params, data)

    test = CUDAMergedKernel(kernel_config)
    dh = test.load_data(data)
    Q_t, grads_t, P_t = test.compute(params, dh)
    test.free()

    assert abs(Q_r - Q_t) < 1e-12, f"Q mismatch: {Q_r} vs {Q_t}"
    assert np.max(np.abs(P_r - P_t)) < 1e-12, f"P mismatch"
    for key in ['ck', 'm0', 'g0', 'scalar']:
        g_ref = np.asarray(grads_r[key])
        g_test = np.asarray(grads_t[key])
        rel = np.max(np.abs(g_test - g_ref)) / (np.max(np.abs(g_ref)) + 1e-30)
        assert rel < 1e-12, f"grad_{key} mismatch: rel={rel:.2e}"
    print(f"  n={n:>4}: ✓ All gradients match")

print("✓ CUDAMergedKernel verified!")
