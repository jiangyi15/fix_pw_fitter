"""
Test CUDA kernel against NumPy for correctness and performance.
"""

import numpy as np
import time


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

    # Load data to GPU
    cuda_kernel.load_data(data)

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

    # Compute with CUDA
    print("Computing with CUDA kernel...")
    try:
        Q_cuda, grads_cuda, P_cuda = cuda_kernel.compute(params, norm=None)
    except Exception as e:
        print(f"CUDA computation failed: {e}")
        import traceback
        traceback.print_exc()
        cuda_kernel.free_data()
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
    cuda_kernel.free_data()

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

        # Benchmark NumPy
        start = time.time()
        for _ in range(10):
            numpy_kernel._compute(params, data, norm=None)
        time_numpy = (time.time() - start) / 10

        # Load data to GPU and benchmark CUDA
        cuda_kernel.load_data(data)

        # Warm up
        cuda_kernel.compute(params, norm=None)

        start = time.time()
        for _ in range(10):
            params["m0"] = params["m0"] + np.random.random(len(params["m0"])) * 0.001
            cuda_kernel.compute(params, norm=None)
        time_cuda = (time.time() - start) / 10

        speedup = time_numpy / time_cuda

        print(f"{n_events:<10} {time_numpy*1000:>13.1f}  {time_cuda*1000:>13.1f}  {speedup:>8.2f}x")

        cuda_kernel.free_data()

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

    cuda_kernel.load_data(data)

    params = {
        "ck": np.random.random(len(ck_map)) + 1j*np.random.random(len(ck_map)),
        "m0": np.random.random(len(config.m0_phys_name)) + 2,
        "g0": np.random.random(len(config.g0_phys_name)) + 0.1,
        "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }

    # Test m0 gradient
    print("\nVerifying m0[0] gradient with 3-point numerical method:")

    params_plus = {k: v.copy() if isinstance(v, np.ndarray) else list(v) for k, v in params.items()}
    params_minus = {k: v.copy() if isinstance(v, np.ndarray) else list(v) for k, v in params.items()}
    params_plus['m0'][0] += epsilon
    params_minus['m0'][0] -= epsilon

    Q_plus, _, _ = cuda_kernel.compute(params_plus, norm=None)
    Q_minus, _, _ = cuda_kernel.compute(params_minus, norm=None)
    Q, grads, _ = cuda_kernel.compute(params, norm=None)

    grad_num = (Q_plus - Q_minus) / (2 * epsilon)
    grad_ana = grads['m0'][0]

    error = abs(grad_num - grad_ana)
    print(f"  Numerical:  {grad_num:.10f}")
    print(f"  Analytical: {grad_ana:.10f}")
    print(f"  Error:      {error:.2e} {'✓' if error < 1e-5 else '✗'}")

    cuda_kernel.free_data()

    if error < 1e-5:
        print("\n✓ CUDA gradients verified!")
        return True
    else:
        print("\n✗ Gradient verification failed!")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPLETE CUDA KERNEL TESTING SUITE")
    print("="*70)

    # Test correctness
    correct = test_correctness()

    if correct:
        # Test performance
        test_performance()

        # Test gradients
        test_numerical_gradient()

    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
