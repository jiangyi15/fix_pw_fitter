#!/usr/bin/env python3
"""Full test suite for the CUDA kernel via ampfit package."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ampfit.config_loader import Config
from ampfit.numpy_kernel import NumpyKernel
from ampfit._cuda import CUDAKernel

CONFIG_FILE = "config_angle.yml"

def make_data(n_events):
    return {
        'mass': np.random.random((n_events, 48)),
        'q': np.random.random((n_events, 72)),
        'angle': np.random.random((n_events, 24, 3)),
        'frac': np.random.random((n_events,)),
        'time': np.random.random((n_events,)),
        'bkg': np.random.random((n_events,)) * 0.01,
        'weight': np.ones((n_events,)),
    }

def test_cuda_kernel_correctness():
    config = Config(CONFIG_FILE)
    kernel_config = config.build_all_index()
    ck_map = config.get_ck_map()
    
    params = {
        'ck': np.random.random(len(ck_map)) + 1j*np.random.random(len(ck_map)),
        'm0': np.random.random(len(config.m0_phys_name)) + 2,
        'g0': np.random.random(len(config.g0_phys_name)) + 0.1,
        'scalar': [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }
    data = make_data(100)
    
    # NumPy reference
    nk = NumpyKernel(kernel_config)
    Q_np, grads_np, P_np = nk._compute(params, data)
    
    # CUDA
    ck = CUDAKernel(kernel_config)
    dh = ck.load_data(data)
    Q_cu, grads_cu, P_cu = ck.compute(params, dh)
    
    assert abs(Q_np - Q_cu) < 1e-12, f"Q mismatch: {Q_np} vs {Q_cu}"
    assert np.max(np.abs(P_np - P_cu)) < 1e-12, "P mismatch"
    assert np.max(np.abs(grads_np['ck'] - grads_cu['ck'])) < 1e-12, "ck grad mismatch"
    assert np.max(np.abs(grads_np['m0'] - grads_cu['m0'])) < 1e-12, "m0 grad mismatch"
    assert np.max(np.abs(grads_np['g0'] - grads_cu['g0'])) < 1e-12, "g0 grad mismatch"
    
    dh.free()
    print("✓ CUDA kernel correctness test passed")

def test_multi_dataset():
    config = Config(CONFIG_FILE)
    kernel_config = config.build_all_index()
    ck_map = config.get_ck_map()
    
    params = {
        'ck': np.random.random(len(ck_map)) + 1j*np.random.random(len(ck_map)),
        'm0': np.random.random(len(config.m0_phys_name)) + 2,
        'g0': np.random.random(len(config.g0_phys_name)) + 0.1,
        'scalar': [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }
    
    ck = CUDAKernel(kernel_config)
    dh1 = ck.load_data(make_data(50))
    dh2 = ck.load_data(make_data(50))
    
    Q1, g1, _ = ck.compute(params, dh1)
    Q2, g2, _ = ck.compute(params, dh2)
    
    # Verify different datasets give different results
    assert abs(Q1 - Q2) > 0, "Different datasets should give different Q"
    
    # Verify each dataset is internally consistent
    # (re-computing same dataset gives same result)
    Q1b, _, _ = ck.compute(params, dh1)
    assert abs(Q1 - Q1b) < 1e-12, "Re-compute should give same Q"
    
    dh1.free(); dh2.free()
    print("✓ Multi-dataset test passed")

def test_with_norm():
    config = Config(CONFIG_FILE)
    kernel_config = config.build_all_index()
    ck_map = config.get_ck_map()
    
    params = {
        'ck': np.random.random(len(ck_map)) + 1j*np.random.random(len(ck_map)),
        'm0': np.random.random(len(config.m0_phys_name)) + 2,
        'g0': np.random.random(len(config.g0_phys_name)) + 0.1,
        'scalar': [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }
    
    data = make_data(50)
    phsp = make_data(100)
    phsp['bkg'] = np.zeros(100)
    
    ck = CUDAKernel(kernel_config)
    dh = ck.load_data(data)
    ph = ck.load_data(phsp)
    
    # Compute norm from phsp
    norm, ng, _ = ck.compute(params, ph, norm=None)
    norm = float(norm)
    
    # Compute NLL with norm - should not crash
    nll, grads, P = ck.compute(params, dh, norm=norm)
    assert np.isfinite(nll), "NLL should be finite"
    assert np.all(np.isfinite(grads['ck'])), "Gradients should be finite"
    
    dh.free(); ph.free()
    print("✓ Norm test passed")

if __name__ == "__main__":
    test_cuda_kernel_correctness()
    test_multi_dataset()
    test_with_norm()
    print("\n✓ All CUDA kernel tests passed!")
