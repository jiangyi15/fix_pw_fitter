"""Test if kernel is deterministic"""
import numpy as np
from config_loader import Config
from numpy_kernel import NumpyKernel


def test_determinism():
    """Check if calling _compute twice gives same result"""
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    kernel = NumpyKernel(kernel_config)
    
    ck_map = config.get_ck_map()
    
    n_events = 500
    np.random.seed(42)
    data = {
        "mass": np.random.random((n_events, 2*3*8)),
        "q": np.random.random((n_events, 3*3*8)),
        "angle": np.random.random((n_events, 3*8, 3)),
        "frac": np.random.random((n_events,)),
        "time": np.random.random((n_events,)),
        "bkg": np.random.random((n_events,)) * 0.01,
        "weight": np.ones((n_events,)),
    }
    
    params = {
        "ck": np.random.random(len(ck_map)) + 1j*np.random.random(len(ck_map)),
        "m0": np.random.random(len(config.m0_phys_name)) + 2,
        "g0": np.random.random(len(config.g0_phys_name)) + 0.1,
        "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }
    
    # Call twice
    Q1, grads1, P1 = kernel._compute(params, data, norm=None)
    Q2, grads2, P2 = kernel._compute(params, data, norm=None)
    
    print("="*70)
    print("Testing determinism")
    print("="*70)
    print(f"Q1 = {Q1}")
    print(f"Q2 = {Q2}")
    print(f"Difference: {abs(Q1 - Q2):.2e}")
    print(f"P difference: {np.max(np.abs(P1 - P2)):.2e}")
    
    if np.allclose(P1, P2):
        print("✓ Deterministic!")
    else:
        print("✗ NOT deterministic!")

if __name__ == "__main__":
    test_determinism()
