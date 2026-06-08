"""Debug why batching gives wrong results"""
import numpy as np
from config_loader import Config
from numpy_kernel import NumpyKernel


def debug_batching():
    """Manually test batching logic"""
    print("="*70)
    print("Debugging batching logic")
    print("="*70)
    
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    kernel = NumpyKernel(kernel_config)
    
    ck_map = config.get_ck_map()
    
    # Use 500 events, batch size 100
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
    
    # Test 1: Process all at once (norm=None)
    print("\n--- Test 1: Process all at once (norm=None) ---")
    Q_full, grads_full, P_full = kernel._compute(params, data_full, norm=None)
    print(f"Q = {Q_full}")
    print(f"P shape = {P_full.shape}")
    print(f"P sum = {np.sum(P_full)}")
    
    # Test 2: Process in batches (norm=None)
    print("\n--- Test 2: Process in batches (norm=None) ---")
    Q_batches = []
    P_batches = []
    grads_batches = []
    
    for i in range(0, n_events, batch_size):
        batch_data = {
            'mass': data_full['mass'][i:i+batch_size],
            'q': data_full['q'][i:i+batch_size],
            'angle': data_full['angle'][i:i+batch_size],
            'frac': data_full['frac'][i:i+batch_size],
            'time': data_full['time'][i:i+batch_size],
            'bkg': data_full['bkg'][i:i+batch_size],
            'weight': data_full['weight'][i:i+batch_size],
        }
        
        Q_batch, grads_batch, P_batch = kernel._compute(params, batch_data, norm=None)
        Q_batches.append(Q_batch)
        P_batches.append(P_batch)
        grads_batches.append(grads_batch)
        print(f"  Batch {i//batch_size}: Q={Q_batch:.6f}, P_sum={np.sum(P_batch):.6f}")
    
    Q_batched_total = sum(Q_batches)
    P_concat = np.concatenate(P_batches)
    
    print(f"\nQ_batched = {Q_batched_total}")
    print(f"P_concat sum = {np.sum(P_concat)}")
    
    print(f"\nDifference:")
    print(f"  Q: {abs(Q_full - Q_batched_total):.2e}")
    print(f"  P: {np.max(np.abs(P_full - P_concat)):.2e}")
    
    # Test 3: Process all at once (norm=1.0)
    print("\n--- Test 3: Process all at once (norm=1.0) ---")
    Q_full_norm, grads_full_norm, P_full_norm = kernel._compute(params, data_full, norm=1.0)
    print(f"Q = {Q_full_norm}")
    
    # Test 4: Process in batches (norm=1.0)
    print("\n--- Test 4: Process in batches (norm=1.0) ---")
    Q_batches_norm = []
    
    for i in range(0, n_events, batch_size):
        batch_data = {
            'mass': data_full['mass'][i:i+batch_size],
            'q': data_full['q'][i:i+batch_size],
            'angle': data_full['angle'][i:i+batch_size],
            'frac': data_full['frac'][i:i+batch_size],
            'time': data_full['time'][i:i+batch_size],
            'bkg': data_full['bkg'][i:i+batch_size],
            'weight': data_full['weight'][i:i+batch_size],
        }
        
        Q_batch, grads_batch, P_batch = kernel._compute(params, batch_data, norm=1.0)
        Q_batches_norm.append(Q_batch)
        print(f"  Batch {i//batch_size}: Q={Q_batch:.6f}")
    
    Q_batched_norm_total = sum(Q_batches_norm)
    
    print(f"\nQ_batched = {Q_batched_norm_total}")
    print(f"\nDifference: {abs(Q_full_norm - Q_batched_norm_total):.2e}")


if __name__ == "__main__":
    debug_batching()
