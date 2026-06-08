"""Debug batch data extraction"""
import numpy as np
from config_loader import Config
from numpy_kernel import NumpyKernel


def debug_batch_extraction():
    """Check if batch data extraction is correct"""
    print("="*70)
    print("Debugging batch data extraction")
    print("="*70)
    
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
    
    # Process each batch and check intermediate results
    print("\nChecking intermediate values:")
    
    for batch_idx in range(5):
        start = batch_idx * batch_size
        end = start + batch_size
        
        batch_data = {
            'mass': data_full['mass'][start:end],
            'q': data_full['q'][start:end],
            'angle': data_full['angle'][start:end],
            'frac': data_full['frac'][start:end],
            'time': data_full['time'][start:end],
            'bkg': data_full['bkg'][start:end],
            'weight': data_full['weight'][start:end],
        }
        
        # Check data integrity
        print(f"\nBatch {batch_idx}:")
        for key in ['mass', 'frac', 'time']:
            batch_vals = batch_data[key]
            full_vals = data_full[key][start:end]
            diff = np.max(np.abs(batch_vals - full_vals))
            print(f"  {key}: max_diff = {diff:.2e}")
        
        # Process batch
        Q_batch, grads_batch, P_batch = kernel._compute(params, batch_data, norm=None)
        print(f"  Q = {Q_batch:.6f}, P_sum = {np.sum(P_batch):.6f}")
    
    # Now compare to processing same indices from full data
    print("\n" + "="*70)
    print("Comparing batch vs full data (same indices)")
    print("="*70)
    
    # Process full data
    Q_full, grads_full, P_full = kernel._compute(params, data_full, norm=None)
    
    # Compare batch 0 vs first 100 events from full
    batch_0_data = {
        'mass': data_full['mass'][0:100],
        'q': data_full['q'][0:100],
        'angle': data_full['angle'][0:100],
        'frac': data_full['frac'][0:100],
        'time': data_full['time'][0:100],
        'bkg': data_full['bkg'][0:100],
        'weight': data_full['weight'][0:100],
    }
    
    Q_batch0, grads_batch0, P_batch0 = kernel._compute(params, batch_0_data, norm=None)
    
    print(f"\nBatch 0:")
    print(f"  Q = {Q_batch0:.6f}")
    print(f"  P_sum = {np.sum(P_batch0):.6f}")
    
    print(f"\nFirst 100 from full:")
    print(f"  P[0:100] sum = {np.sum(P_full[0:100]):.6f}")
    print(f"  P[0:100] vs P_batch0: max_diff = {np.max(np.abs(P_full[0:100] - P_batch0)):.2e}")
    
    # Check if P values match
    if np.allclose(P_full[0:100], P_batch0):
        print("  ✓ P values match!")
    else:
        print("  ✗ P values DON'T match!")


if __name__ == "__main__":
    debug_batch_extraction()
