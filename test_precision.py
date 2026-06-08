"""Test if it's floating-point precision"""
import numpy as np
from config_loader import Config
from numpy_kernel import NumpyKernel


def test():
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    
    # Create SEPARATE kernels to avoid any shared state
    kernel_full = NumpyKernel(kernel_config)
    kernel_batch = NumpyKernel(kernel_config)
    
    ck_map = config.get_ck_map()
    
    n_events = 200  # Just 2 batches
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
    Q_full, _, P_full = kernel_full._compute(params, data_full, norm=None)
    print(f"Full: Q = {Q_full:.10f}, P[100:200] sum = {np.sum(P_full[100:200]):.10f}")
    
    # Process batch 1 alone (indices 100-200)
    batch_data = {k: v[100:200] for k, v in data_full.items()}
    Q_batch, _, P_batch = kernel_batch._compute(params, batch_data, norm=None)
    print(f"Batch: Q = {Q_batch:.10f}, P sum = {np.sum(P_batch):.10f}")
    
    print(f"\nDifference:")
    print(f"  Q: {abs(Q_full - Q_batch):.2e}")
    print(f"  P: {np.max(np.abs(P_full[100:200] - P_batch)):.2e}")
    
    # Check intermediate values
    print(f"\nData check:")
    print(f"  mass[100]: full={data_full['mass'][100, 0]:.10f}, batch={batch_data['mass'][0, 0]:.10f}")
    print(f"  Match: {np.allclose(data_full['mass'][100:200], batch_data['mass'])}")

if __name__ == "__main__":
    test()
