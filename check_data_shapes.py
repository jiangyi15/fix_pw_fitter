"""Check data shapes"""
import numpy as np
from config_loader import Config

config = Config("config_angle.yml")
kernel_config = config.build_all_index()

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

print("Data shapes:")
for key, val in data.items():
    print(f"  {key}: {val.shape}")

print("\nChecking batch extraction:")
batch_data = {k: v[0:100] for k, v in data.items()}

for key in data:
    orig_shape = data[key].shape
    batch_shape = batch_data[key].shape
    
    # Check if batch shape is correct
    expected_shape = (100,) + orig_shape[1:] if len(orig_shape) > 1 else (100,)
    
    if batch_shape == expected_shape:
        print(f"  {key}: {orig_shape} → {batch_shape} ✓")
    else:
        print(f"  {key}: {orig_shape} → {batch_shape} ✗ (expected {expected_shape})")
