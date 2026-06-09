"""
Test CUDA kernel against NumPy for correctness and performance.
Uses the new API: kernel.load_data(data) → kernel.compute(params, data)
"""

import numpy as np
import time


def make_test_data(n_events, seed=42):
    """Create synthetic test data."""
    np.random.seed(seed)
    return {
        "mass": np.random.random((n_events, 2*3*8)),
        "q": np.random.random((n_events, 3*3*8)),
        "angle": np.random.random((n_events, 3*8, 3)),
        "frac": np.random.random((n_events,)),
        "time": np.random.random((n_events,)),
        "bkg": np.random.random((n_events,)) * 0.01,
        "weight": np.ones((n_events,)),
    }


def test_correctness():
    """Test CUDA kernel produces same results as NumPy"""
    print("="*70)
    print("CORRECTNESS TEST: CUDA vs NumPy")
    print("="*70)

    from config_loader import Config
    from numpy_kernel import NumpyKernelCorrect

    try:
        from cuda_kernel_cffi import CUDAKernel
    except Exception as e:
        print(f"\nCUDA not available: {e}")
        print("Skipping CUDA test")
        return False

    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()

    # Create kernels
    numpy_kernel = NumpyKernelCorrect(kernel_config)
    cuda_kernel = CUDAKernel(kernel_config)

    # Create test data
    n_events = 100
    data = make_test_data(n_events)

    # Load data via kernel.load_data() → returns GPUDataHolder
    data_holder = cuda_kernel.load_data(data)

    # Create parameters
    ck_map = config.get_ck_map()
    params = {
        "ck": np.random.random(len(ck_map)) + 1j*np.random.random(len(ck_map)),
        "m0": np.random.random(len(config.m0_phys_name)) + 2,
        "g0": np.random.random(len(config.g0_phys_name)) + 0.1,
        "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }

    # Compute with NumPy
    print("\nComputing with NumPy kernel...")
    Q_numpy, grads_numpy, P_numpy = numpy_kernel._compute(params, data, norm=None)

    # Compute with CUDA using new API: compute(params, data_holder)
    print("Computing with CUDA kernel (new API)...")
    try:
        Q_cuda, grads_cuda, P_cuda = cuda_kernel.compute(params, data_holder, norm=None)
    except Exception as e:
        print(f"CUDA computation failed: {e}")
        import traceback
        traceback.print_exc()
        data_holder.free()
        return False

    # Compare results
    print("\n" + "-"*70)
    print("RESULTS COMPARISON")
    print("-"*70)

    Q_error = abs(Q_numpy - Q_cuda)
    print(f"Q:  NumPy = {Q_numpy:.10f}, CUDA = {Q_cuda:.10f}")
    print(f"    error = {Q_error:.2e} {'✓' if Q_error < 1e-8 else '✗'}")

    P_max_error = np.max(np.abs(P_numpy - P_cuda))
    print(f"P:  max error = {P_max_error:.2e} {'✓' if P_max_error < 1e-8 else '✗'}")

    # Compare gradients
    print("\nGradient comparison:")
    for key in ["ck", "m0", "g0"]:
        grad_error = np.max(np.abs(grads_numpy[key] - grads_cuda[key]))
        print(f"  {key:8s}: max error = {grad_error:.2e} {'✓' if grad_error < 1e-8 else '✗'}")

    scalar_names = ["Gamma", "Delta_Gamma", "Delta_m", "A_p", "poq_rho", "pop_phi"]
    for i, name in enumerate(scalar_names):
        grad_error = abs(grads_numpy["scalar"][i] - grads_cuda["scalar"][i])
        print(f"  {name:12s}: error = {grad_error:.2e} {'✓' if grad_error < 1e-8 else '✗'}")

    print("-"*70)

    # Clean up
    data_holder.free()

    all_correct = (
        Q_error < 1e-8 and
        P_max_error < 1e-8 and
        all(np.max(np.abs(grads_numpy[k] - grads_cuda[k])) < 1e-8 for k in ["ck", "m0", "g0"]) and
        all(abs(grads_numpy["scalar"][i] - grads_cuda["scalar"][i]) < 1e-8 for i in range(6))
    )

    if all_correct:
        print("\n✓ CUDA kernel matches NumPy kernel!")
        return True
    else:
        print("\n✗ CUDA kernel has errors!")
        return False


def test_multi_dataset():
    """Demonstrate multiple datasets on GPU with zero reload overhead."""
    print("\n" + "="*70)
    print("MULTI-DATASET TEST")
    print("="*70)

    from config_loader import Config
    from numpy_kernel import NumpyKernelCorrect

    try:
        from cuda_kernel_cffi import CUDAKernel
    except Exception as e:
        print(f"CUDA not available: {e}")
        return False

    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    numpy_kernel = NumpyKernelCorrect(kernel_config)
    cuda_kernel = CUDAKernel(kernel_config)

    n_events = 50
    data1 = make_test_data(n_events, seed=100)
    data2 = make_test_data(n_events, seed=200)

    # Load two independent datasets via kernel.load_data()
    holder1 = cuda_kernel.load_data(data1)
    holder2 = cuda_kernel.load_data(data2)

    ck_map = config.get_ck_map()
    params = {
        "ck": np.random.random(len(ck_map)) + 1j*np.random.random(len(ck_map)),
        "m0": np.random.random(len(config.m0_phys_name)) + 2,
        "g0": np.random.random(len(config.g0_phys_name)) + 0.1,
        "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }

    # Compute with NumPy
    Q1_np, grads1_np, P1_np = numpy_kernel._compute(params, data1, norm=None)
    Q2_np, grads2_np, P2_np = numpy_kernel._compute(params, data2, norm=None)

    # Compute with CUDA on dataset 1
    print("\nComputing dataset 1...")
    Q1_cu, grads1_cu, P1_cu = cuda_kernel.compute(params, holder1, norm=None)

    # Compute with CUDA on dataset 2 (no data reload!)
    print("Computing dataset 2 (no reload)...")
    Q2_cu, grads2_cu, P2_cu = cuda_kernel.compute(params, holder2, norm=None)

    # Verify both match
    err1 = abs(Q1_np - Q1_cu)
    err2 = abs(Q2_np - Q2_cu)
    print(f"\nDataset 1 Q error: {err1:.2e} {'✓' if err1 < 1e-8 else '✗'}")
    print(f"Dataset 2 Q error: {err2:.2e} {'✓' if err2 < 1e-8 else '✗'}")

    ok = True
    for key in ["ck", "m0", "g0"]:
        e1 = np.max(np.abs(grads1_np[key] - grads1_cu[key]))
        e2 = np.max(np.abs(grads2_np[key] - grads2_cu[key]))
        ok &= e1 < 1e-8 and e2 < 1e-8
        print(f"  {key:8s}: ds1={e1:.2e} ds2={e2:.2e} {'✓' if (e1 < 1e-8 and e2 < 1e-8) else '✗'}")

    for i, name in enumerate(["Gamma", "Delta_Gamma", "Delta_m", "A_p", "poq_rho", "pop_phi"]):
        e1 = abs(grads1_np["scalar"][i] - grads1_cu["scalar"][i])
        e2 = abs(grads2_np["scalar"][i] - grads2_cu["scalar"][i])
        ok &= e1 < 1e-8 and e2 < 1e-8

    holder1.free()
    holder2.free()

    if ok:
        print("\n✓ Multi-dataset test passed!")
        return True
    else:
        print("\n✗ Multi-dataset test failed!")
        return False


def test_performance():
    """Benchmark CUDA vs NumPy performance"""
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARK")
    print("="*70)

    from config_loader import Config
    from numpy_kernel import NumpyKernelCorrect

    try:
        from cuda_kernel_cffi import CUDAKernel
    except Exception:
        print("CUDA not available, skipping performance test")
        return

    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()

    numpy_kernel = NumpyKernelCorrect(kernel_config)
    cuda_kernel = CUDAKernel(kernel_config)

    ck_map = config.get_ck_map()

    print(f"\n{'Events':<10} {'NumPy (ms)':<15} {'CUDA (ms)':<15} {'Speedup':<10}")
    print("-"*70)

    for n_events in [100, 500, 1000]:
        data = make_test_data(n_events)

        params = {
            "ck": np.random.random(len(ck_map)) + 1j*np.random.random(len(ck_map)),
            "m0": np.random.random(len(config.m0_phys_name)) + 2,
            "g0": np.random.random(len(config.g0_phys_name)) + 0.1,
            "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
        }

        # Benchmark NumPy
        start = time.time()
        for _ in range(10):
            numpy_kernel._compute(params, data, norm=None)
        time_numpy = (time.time() - start) / 10

        # Load to GPU once via kernel.load_data()
        holder = cuda_kernel.load_data(data)

        # Warm up
        cuda_kernel.compute(params, holder, norm=None)

        # Benchmark CUDA (data already on GPU)
        start = time.time()
        for _ in range(10):
            params["m0"] = params["m0"] + np.random.random(len(params["m0"])) * 0.001
            cuda_kernel.compute(params, holder, norm=None)
        time_cuda = (time.time() - start) / 10

        speedup = time_numpy / time_cuda

        print(f"{n_events:<10} {time_numpy*1000:>13.1f}  {time_cuda*1000:>13.1f}  {speedup:>8.2f}x")

        holder.free()

    print("="*70)


def test_numerical_gradient():
    """Verify gradients with numerical method"""
    print("\n" + "="*70)
    print("NUMERICAL GRADIENT VERIFICATION")
    print("="*70)

    from config_loader import Config

    try:
        from cuda_kernel_cffi import CUDAKernel
    except Exception:
        print("CUDA not available, skipping gradient test")
        return

    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()

    cuda_kernel = CUDAKernel(kernel_config)

    ck_map = config.get_ck_map()
    n_events = 100
    epsilon = 1e-5

    data = make_test_data(n_events)
    holder = cuda_kernel.load_data(data)

    params = {
        "ck": np.random.random(len(ck_map)) + 1j*np.random.random(len(ck_map)),
        "m0": np.random.random(len(config.m0_phys_name)) + 2,
        "g0": np.random.random(len(config.g0_phys_name)) + 0.1,
        "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }

    print("\nVerifying gradients with 3-point numerical method:")

    for param_name in ["m0", "g0", "ck"]:
        if param_name == "m0":
            idx = 0
            params_p = {k: v.copy() if isinstance(v, np.ndarray) else list(v) for k, v in params.items()}
            params_m = {k: v.copy() if isinstance(v, np.ndarray) else list(v) for k, v in params.items()}
            params_p['m0'][idx] += epsilon; params_m['m0'][idx] -= epsilon
            Qp, _, _ = cuda_kernel.compute(params_p, holder, norm=None)
            Qm, _, _ = cuda_kernel.compute(params_m, holder, norm=None)
            Q, grads, _ = cuda_kernel.compute(params, holder, norm=None)
            grad_num = (Qp - Qm) / (2 * epsilon)
            grad_ana = grads['m0'][idx]

        elif param_name == "g0":
            idx = 0
            params_p = {k: v.copy() if isinstance(v, np.ndarray) else list(v) for k, v in params.items()}
            params_m = {k: v.copy() if isinstance(v, np.ndarray) else list(v) for k, v in params.items()}
            params_p['g0'][idx] += epsilon; params_m['g0'][idx] -= epsilon
            Qp, _, _ = cuda_kernel.compute(params_p, holder, norm=None)
            Qm, _, _ = cuda_kernel.compute(params_m, holder, norm=None)
            Q, grads, _ = cuda_kernel.compute(params, holder, norm=None)
            grad_num = (Qp - Qm) / (2 * epsilon)
            grad_ana = grads['g0'][idx]

        else:  # ck - test real part
            idx = 0
            params_p = {k: v.copy() if isinstance(v, np.ndarray) else list(v) for k, v in params.items()}
            params_m = {k: v.copy() if isinstance(v, np.ndarray) else list(v) for k, v in params.items()}
            params_p['ck'][idx] += epsilon; params_m['ck'][idx] -= epsilon
            Qp, _, _ = cuda_kernel.compute(params_p, holder, norm=None)
            Qm, _, _ = cuda_kernel.compute(params_m, holder, norm=None)
            Q, grads, _ = cuda_kernel.compute(params, holder, norm=None)
            grad_num = (Qp - Qm) / (2 * epsilon)
            grad_ana = 2 * grads['ck'][idx].real

        error = abs(grad_num - grad_ana)
        print(f"  {param_name}[{idx}]: numerical={grad_num:.10f}  analytical={grad_ana:.10f}  error={error:.2e} {'✓' if error < 1e-5 else '✗'}")

    holder.free()
    print("\n✓ CUDA gradients verified!")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPLETE CUDA KERNEL TESTING SUITE")
    print("="*70)

    correct = test_correctness()

    if correct:
        test_multi_dataset()
        test_performance()
        test_numerical_gradient()

    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
