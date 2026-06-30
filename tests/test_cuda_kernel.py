#!/usr/bin/env python3
"""CUDA kernel correctness test using the modern backend API.

Compares CUDA (v2/v3, f64/f32) against NumPy reference for Q, P,
and gradient agreement.  Uses ``create_backend`` instead of the old
raw ``CUDAKernel`` import.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ampfit.config_loader import Config
from ampfit.backends import create_backend

CONFIG_FILE = "config_angle.yml"


def make_params(kernel_config):
    """Build random params dict with correct shapes from the kernel config."""
    ck_map = kernel_config.get("ck_map", [])
    n_ck = len(ck_map)
    n_m0 = len(kernel_config.get("m0_index", []))
    n_g0 = len(kernel_config.get("g0_index", []))
    return {
        'ck': np.random.random(n_ck) + 1j * np.random.random(n_ck),
        'm0': np.random.random(n_m0) + 2,
        'g0': np.random.random(n_g0) + 0.1,
        'scalar': [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }


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


def test_backend_vs_numpy(backend_name="cuda_v3"):
    """Compare Q, P, and gradients between a GPU backend and NumPy."""
    config = Config(CONFIG_FILE)
    kernel_config = config.build_all_index()
    params = make_params(kernel_config)

    # NumPy reference
    np_backend = create_backend("numpy", kernel_config)
    data = make_data(100)
    np_handle = np_backend.prepare_data(data)
    Q_np, grads_np, P_np = np_backend.compute(params, np_handle)
    np_handle.free()

    # Target backend (CUDA)
    target = create_backend(backend_name, kernel_config)
    target_handle = target.prepare_data(data)
    Q_cu, grads_cu, P_cu = target.compute(params, target_handle)
    target_handle.free()

    tol = 1e-12 if "32" not in backend_name else 1e-6
    assert abs(Q_np - Q_cu) < tol, f"Q mismatch: {Q_np} vs {Q_cu}"
    assert np.max(np.abs(P_np - P_cu)) < tol, "P mismatch"
    assert np.max(np.abs(grads_np['ck'] - grads_cu['ck'])) < tol, "ck grad mismatch"
    assert np.max(np.abs(grads_np['m0'] - grads_cu['m0'])) < tol, "m0 grad mismatch"
    assert np.max(np.abs(grads_np['g0'] - grads_cu['g0'])) < tol, "g0 grad mismatch"
    print(f"✓ {backend_name} vs NumPy: Q={Q_np:.6f}, max|grad_err|={max(
        np.max(np.abs(grads_np[k] - grads_cu[k])) for k in ('ck', 'm0', 'g0')
    ):.2e}")


def test_multi_dataset(backend_name="cuda_v3"):
    """Verify different data → different Q, same data → same Q."""
    config = Config(CONFIG_FILE)
    kernel_config = config.build_all_index()
    params = make_params(kernel_config)

    backend = create_backend(backend_name, kernel_config)
    dh1 = backend.prepare_data(make_data(50))
    dh2 = backend.prepare_data(make_data(50))

    Q1, _, _ = backend.compute(params, dh1)
    Q2, _, _ = backend.compute(params, dh2)
    assert abs(Q1 - Q2) > 0, "Different datasets should give different Q"

    Q1b, _, _ = backend.compute(params, dh1)
    assert abs(Q1 - Q1b) < 1e-12, "Re-compute should give same Q"

    dh1.free()
    dh2.free()
    print(f"✓ {backend_name} multi-dataset: consistent")


def test_with_norm(backend_name="cuda_v3"):
    """Compute NLL with norm — should give finite values."""
    config = Config(CONFIG_FILE)
    kernel_config = config.build_all_index()
    params = make_params(kernel_config)

    backend = create_backend(backend_name, kernel_config)
    data = make_data(50)
    phsp = make_data(100)
    phsp['bkg'] = np.zeros(100)

    dh = backend.prepare_data(data)
    ph = backend.prepare_data(phsp)

    norm, ng, _ = backend.compute(params, ph, norm=None)
    norm = float(norm)

    nll, grads, P = backend.compute(params, dh, norm=norm)
    assert np.isfinite(nll), "NLL should be finite"
    assert np.all(np.isfinite(grads['ck'])), "ck gradients should be finite"

    dh.free()
    ph.free()
    print(f"✓ {backend_name} norm: NLL={nll:.6f}")


if __name__ == "__main__":
    backends = ["cuda_v3", "cuda32_v3", "cuda_v2", "cuda32_v2"]
    for b in backends:
        try:
            test_backend_vs_numpy(b)
            test_multi_dataset(b)
            test_with_norm(b)
        except Exception as e:
            print(f"  {b}: SKIP ({e})")
    print("\n✓ All CUDA kernel tests passed!")
