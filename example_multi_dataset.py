"""
Example: Using GPUDataHolder for efficient multi-dataset computation

This demonstrates the new architecture where:
1. GPUConfig: Shared config arrays (indices, tables, matrices) - created once
2. GPUDataHolder: Per-dataset data + outputs - can create multiple instances
3. CUDAKernel: Uses GPUConfig, accepts different GPUDataHolders

Benefits:
- Zero data reload overhead when switching datasets
- Multiple datasets can coexist on GPU
- Clear ownership of GPU memory
- Ideal for iterative optimization with multiple datasets
"""

import numpy as np
from config_loader import Config
from cuda_kernel_cffi import CUDAKernel, GPUDataHolder

# Load configuration
config = Config("config_angle.yml")
kernel_config = config.build_all_index()

# Create kernel (owns shared config arrays)
kernel = CUDAKernel(kernel_config)
print("✓ Kernel created with shared config arrays")

# Create multiple datasets
print("\n" + "="*70)
print("Creating multiple datasets...")
print("="*70)

# Dataset 1: Small batch
np.random.seed(42)
data1 = {
    "mass": np.random.random((100, 48)),
    "q": np.random.random((100, 72)),
    "angle": np.random.random((100, 24, 3)),
    "frac": np.random.random((100,)),
    "time": np.random.random((100,)),
    "bkg": np.random.random((100,)) * 0.01,
    "weight": np.ones((100,)),
}

# Dataset 2: Medium batch
np.random.seed(123)
data2 = {
    "mass": np.random.random((500, 48)),
    "q": np.random.random((500, 72)),
    "angle": np.random.random((500, 24, 3)),
    "frac": np.random.random((500,)),
    "time": np.random.random((500,)),
    "bkg": np.random.random((500,)) * 0.01,
    "weight": np.ones((500,)),
}

# Dataset 3: Large batch
np.random.seed(456)
data3 = {
    "mass": np.random.random((1000, 48)),
    "q": np.random.random((1000, 72)),
    "angle": np.random.random((1000, 24, 3)),
    "frac": np.random.random((1000,)),
    "time": np.random.random((1000,)),
    "bkg": np.random.random((1000,)) * 0.01,
    "weight": np.ones((1000,)),
}

# Load all datasets to GPU
print("\nLoading datasets to GPU (one-time cost)...")
holder1 = kernel.create_data_holder(data1)
print(f"  Dataset 1: {data1['mass'].shape[0]} events loaded")

holder2 = kernel.create_data_holder(data2)
print(f"  Dataset 2: {data2['mass'].shape[0]} events loaded")

holder3 = kernel.create_data_holder(data3)
print(f"  Dataset 3: {data3['mass'].shape[0]} events loaded")

# Create parameters
ck_map = config.get_ck_map()
params = {
    "ck": np.random.random(len(ck_map)) + 1j*np.random.random(len(ck_map)),
    "m0": np.random.random(len(config.m0_phys_name)) + 2,
    "g0": np.random.random(len(config.g0_phys_name)) + 0.1,
    "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
}

# Compute with all datasets efficiently
print("\n" + "="*70)
print("Computing with all datasets (NO reload needed)...")
print("="*70)

import time

# Dataset 1
start = time.time()
Q1, grads1, P1 = kernel.compute(holder1, params)
time1 = time.time() - start
print(f"\nDataset 1: Q = {Q1:.6f}, Time = {time1*1000:.2f}ms")

# Dataset 2
start = time.time()
Q2, grads2, P2 = kernel.compute(holder2, params)
time2 = time.time() - start
print(f"Dataset 2: Q = {Q2:.6f}, Time = {time2*1000:.2f}ms")

# Dataset 3
start = time.time()
Q3, grads3, P3 = kernel.compute(holder3, params)
time3 = time.time() - start
print(f"Dataset 3: Q = {Q3:.6f}, Time = {time3*1000:.2f}ms")

# Switch back to dataset 1 (zero overhead!)
print("\n" + "="*70)
print("Switching back to Dataset 1 (instant - no reload)")
print("="*70)
start = time.time()
Q1_again, _, _ = kernel.compute(holder1, params)
time1_again = time.time() - start
print(f"Dataset 1 (recompute): Q = {Q1_again:.6f}, Time = {time1_again*1000:.2f}ms")
print(f"Match: {np.allclose(Q1, Q1_again)}")

# Demonstrate gradient switching
print("\n" + "="*70)
print("Computing gradients for different datasets")
print("="*70)

# Modify parameters
params_modified = params.copy()
params_modified["m0"] = params["m0"] + 0.01

# Compute gradients with different parameters on same dataset
Q1_mod, grads1_mod, _ = kernel.compute(holder1, params_modified)
print(f"\nDataset 1 with modified params:")
print(f"  Q changed from {Q1:.6f} to {Q1_mod:.6f}")
print(f"  ck gradient norm: {np.linalg.norm(grads1['ck']):.6f} → {np.linalg.norm(grads1_mod['ck']):.6f}")
print(f"  m0 gradient norm: {np.linalg.norm(grads1['m0']):.6f} → {np.linalg.norm(grads1_mod['m0']):.6f}")
print(f"  g0 gradient norm: {np.linalg.norm(grads1['g0']):.6f} → {np.linalg.norm(grads1_mod['g0']):.6f}")

# Same parameters on different dataset
Q2_same, grads2_same, _ = kernel.compute(holder2, params_modified)
print(f"\nDataset 2 with same modified params:")
print(f"  Q = {Q2_same:.6f}")
print(f"  ck gradient norm: {np.linalg.norm(grads2_same['ck']):.6f}")
print(f"  m0 gradient norm: {np.linalg.norm(grads2_same['m0']):.6f}")
print(f"  g0 gradient norm: {np.linalg.norm(grads2_same['g0']):.6f}")

# Memory management
print("\n" + "="*70)
print("Memory Management")
print("="*70)
print("\nEach GPUDataHolder owns its memory:")
print(f"  holder1: {holder1.n_events} events")
print(f"  holder2: {holder2.n_events} events")
print(f"  holder3: {holder3.n_events} events")
print("\nFree individual datasets when done:")
holder3.free()
print("  ✓ holder3 freed (holder1 & holder2 still usable)")

# Can still compute with holder1 and holder2
Q1_final, _, _ = kernel.compute(holder1, params)
print(f"  ✓ holder1 still works: Q = {Q1_final:.6f}")

# Free kernel (releases config arrays)
print("\nFree kernel (shared config arrays):")
kernel.free()
print("  ✓ Config arrays freed")
print("  Note: holder1 & holder2 still exist but kernel is destroyed")

print("\n" + "="*70)
print("✓ Multi-dataset computation complete!")
print("="*70)
print("\nKey benefits:")
print("  1. Zero reload overhead when switching datasets")
print("  2. Multiple datasets can coexist on GPU")
print("  3. Independent memory management")
print("  4. Ideal for iterative optimization")
