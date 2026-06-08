"""Test each batch individually"""
import numpy as np
from config_loader import Config
from numpy_kernel import NumpyKernel


def test_all_batches():
    """Compare each batch to corresponding slice from full"""
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    kernel = NumpyKernel(kernel_config)
    
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
    print("="*70)
    print(f"Full: Q = {Q_full:.6f}, P_sum = {np.sum(P_full):.6f}")
    print("="*70)
    
    # Process each batch and compare
    batch_P_sums = []
    for i in range(5):
        start = i * batch_size
        end = start + batch_size
        
        # Batch data
        batch_data = {
            'mass': data_full['mass'][start:end],
            'q': data_full['q'][start:end],
            'angle': data_full['angle'][start:end],
            'frac': data_full['frac'][start:end],
            'time': data_full['time'][start:end],
            'bkg': data_full['bkg'][start:end],
            'weight': data_full['weight'][start:end],
        }
        
        # Process batch
        Q_batch, _, P_batch = kernel._compute(params, batch_data, norm=None)
        
        # Compare to corresponding slice from full
        P_slice = P_full[start:end]
        
        diff = np.max(np.abs(P_batch - P_slice))
        match = "✓" if diff < 1e-10 else "✗"
        
        print(f"Batch {i}: P_batch_sum={np.sum(P_batch):.6f}, P_slice_sum={np.sum(P_slice):.6f}, diff={diff:.2e} {match}")
        
        batch_P_sums.append(np.sum(P_batch))
    
    total_batch_P = sum(batch_P_sums)
    print(f"\nSum of batch P: {total_batch_P:.6f}")
    print(f"P from full:    {np.sum(P_full):.6f}")
    print(f"Difference:     {abs(total_batch_P - np.sum(P_full)):.2e}")

if __name__ == "__main__":
    test_all_batches()
