"""Test selective cache kernel with batching"""
import numpy as np
from config_loader import Config
from numpy_kernel_selective_cache import NumpyKernelSelectiveCache


def test():
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    kernel = NumpyKernelSelectiveCache(kernel_config)
    
    ck_map = config.get_ck_map()
    
    n_events = 500
    batch_size = 100
    
    np.random.seed(42)
    data_full = {
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
    
    # Process full
    Q_full, _, P_full = kernel._compute(params, data_full, norm=None)
    print(f"Full: Q = {Q_full:.6f}")
    
    # Process batches
    for i in range(5):
        start = i * batch_size
        end = start + batch_size
        
        batch_data = {k: v[start:end] for k, v in data_full.items()}
        
        Q_batch, _, P_batch = kernel._compute(params, batch_data, norm=None)
        P_slice = P_full[start:end]
        
        diff = np.max(np.abs(P_batch - P_slice))
        match = "✓" if diff < 1e-10 else "✗"
        
        print(f"Batch {i}: diff={diff:.2e} {match}")

if __name__ == "__main__":
    test()
