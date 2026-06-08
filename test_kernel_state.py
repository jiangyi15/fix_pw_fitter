"""Test if kernel has mutable state"""
import numpy as np
from config_loader import Config
from numpy_kernel import NumpyKernel


def test_kernel_state():
    """Test if kernel modifies internal state"""
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    
    ck_map = config.get_ck_map()
    
    n_events = 100
    np.random.seed(42)
    data1 = {
        "mass": np.random.random((n_events, 2*3*8)),
        "q": np.random.random((n_events, 3*3*8)),
        "angle": np.random.random((n_events, 3*8, 3)),
        "frac": np.random.random((n_events,)),
        "time": np.random.random((n_events,)),
        "bkg": np.random.random((n_events,)) * 0.01,
        "weight": np.ones((n_events,)),
    }
    
    data2 = {
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
    
    # Test 1: Use SAME kernel for both calls
    print("="*70)
    print("Test 1: SAME kernel object")
    print("="*70)
    
    kernel_same = NumpyKernel(kernel_config)
    Q1, _, P1 = kernel_same._compute(params, data1, norm=None)
    Q2, _, P2 = kernel_same._compute(params, data1, norm=None)  # Same data!
    
    print(f"Call 1: Q = {Q1:.6f}")
    print(f"Call 2: Q = {Q2:.6f}")
    print(f"Difference: {abs(Q1 - Q2):.2e} {'✓' if abs(Q1-Q2) < 1e-10 else '✗'}")
    
    # Test 2: Use NEW kernel for second call
    print("\n" + "="*70)
    print("Test 2: NEW kernel object")
    print("="*70)
    
    kernel1 = NumpyKernel(kernel_config)
    Q1, _, P1 = kernel1._compute(params, data1, norm=None)
    
    kernel2 = NumpyKernel(kernel_config)  # NEW kernel!
    Q2, _, P2 = kernel2._compute(params, data1, norm=None)  # Same data!
    
    print(f"Kernel 1: Q = {Q1:.6f}")
    print(f"Kernel 2: Q = {Q2:.6f}")
    print(f"Difference: {abs(Q1 - Q2):.2e} {'✓' if abs(Q1-Q2) < 1e-10 else '✗'}")

if __name__ == "__main__":
    test_kernel_state()
