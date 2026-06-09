"""
Debug CUDA kernel by comparing intermediate values with NumPy step by step.
"""

import numpy as np
from config_loader import Config
from numpy_kernel import NumpyKernelCorrect
from cuda_kernel_cffi import CUDAKernel


def debug_forward_pass():
    """Compare forward pass step by step"""
    print("="*70)
    print("DEBUG: Forward Pass Step-by-Step Comparison")
    print("="*70)
    
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    
    # Use small dataset for easier debugging
    n_events = 5  # Small for debugging
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
    
    ck_map = config.get_ck_map()
    params = {
        "ck": np.random.random(len(ck_map)) + 1j*np.random.random(len(ck_map)),
        "m0": np.random.random(len(config.m0_phys_name)) + 2,
        "g0": np.random.random(len(config.g0_phys_name)) + 0.1,
        "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }
    
    # Create NumPy kernel and compute
    numpy_kernel = NumpyKernelCorrect(kernel_config)
    print("\nComputing with NumPy...")
    Q_numpy, grads_numpy, P_numpy = numpy_kernel._compute(params, data, norm=None)
    
    # Create CUDA kernel
    cuda_kernel = CUDAKernel(kernel_config)
    cuda_kernel.load_data(data)
    
    print("\nComputing with CUDA...")
    Q_cuda, grads_cuda, P_cuda = cuda_kernel.compute(params, norm=None)
    
    # Compare results step by step
    print("\n" + "="*70)
    print("STEP 1: Final Results")
    print("="*70)
    print(f"Q:  NumPy = {Q_numpy:.10f}")
    print(f"    CUDA  = {Q_cuda:.10f}")
    print(f"    Error = {abs(Q_numpy - Q_cuda):.2e}")
    
    print(f"\nP[0]: NumPy = {P_numpy[0]:.10f}")
    print(f"      CUDA  = {P_cuda[0]:.10f}")
    print(f"      Error = {abs(P_numpy[0] - P_cuda[0]):.2e}")
    
    # Compare gradients
    print("\n" + "="*70)
    print("STEP 2: Gradient Comparison")
    print("="*70)
    
    print("\nck gradients:")
    print(f"  NumPy[0] = {grads_numpy['ck'][0]}")
    print(f"  CUDA[0]  = {grads_cuda['ck'][0]}")
    print(f"  Error    = {abs(grads_numpy['ck'][0] - grads_cuda['ck'][0]):.2e}")
    
    print("\nm0 gradients:")
    print(f"  NumPy[0] = {grads_numpy['m0'][0]:.10f}")
    print(f"  CUDA[0]  = {grads_cuda['m0'][0]:.10f}")
    print(f"  Error    = {abs(grads_numpy['m0'][0] - grads_cuda['m0'][0]):.2e}")
    
    print("\ng0 gradients:")
    print(f"  NumPy[0] = {grads_numpy['g0'][0]:.10f}")
    print(f"  CUDA[0]  = {grads_cuda['g0'][0]:.10f}")
    print(f"  Error    = {abs(grads_numpy['g0'][0] - grads_cuda['g0'][0]):.2e}")
    
    print("\nScalar gradients:")
    scalar_names = ["Gamma", "Delta_Gamma", "Delta_m", "A_p", "poq_rho", "pop_phi"]
    for i, name in enumerate(scalar_names):
        print(f"  {name:12s}: NumPy = {grads_numpy['scalar'][i]:.10f}, CUDA = {grads_cuda['scalar'][i]:.10f}, "
              f"Error = {abs(grads_numpy['scalar'][i] - grads_cuda['scalar'][i]):.2e}")
    
    # Now let's check intermediate values
    print("\n" + "="*70)
    print("STEP 3: Intermediate Value Checks")
    print("="*70)
    
    # We need to modify the kernels to output intermediate values
    # For now, let's check the parameters
    print("\nParameter shapes:")
    print(f"  ck: {params['ck'].shape}")
    print(f"  m0: {params['m0'].shape}")
    print(f"  g0: {params['g0'].shape}")
    
    print("\nConfig shapes:")
    print(f"  n_wave: {kernel_config['matrix_angle'].shape[1]}")
    print(f"  n_unique_bw (m0_index): {len(kernel_config['m0_index'])}")
    print(f"  n_gamma_rows: {kernel_config['matrix_gamma'].shape[0]}")
    print(f"  matrix_gamma: {kernel_config['matrix_gamma'].shape}")
    
    cuda_kernel.free_data()
    
    return abs(Q_numpy - Q_cuda) < 1e-6


def check_numpy_intermediates():
    """Print intermediate values from NumPy kernel for comparison"""
    print("\n" + "="*70)
    print("NUMPY INTERMEDIATE VALUES")
    print("="*70)
    
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    
    n_events = 2  # Very small for debugging
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
    
    ck_map = config.get_ck_map()
    params = {
        "ck": np.random.random(len(ck_map)) + 1j*np.random.random(len(ck_map)),
        "m0": np.random.random(len(config.m0_phys_name)) + 2,
        "g0": np.random.random(len(config.g0_phys_name)) + 0.1,
        "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }
    
    # Manually compute forward pass with print statements
    print("\nComputing forward pass manually with NumPy...")
    
    # Extract parameters
    ck = params["ck"]
    m0 = params["m0"]
    g0 = params["g0"]
    Gamma, Delta_Gamma, Delta_m, A_p, poq_rho, pop_phi = params["scalar"]
    
    # Extract data
    mass = data["mass"]
    momentum = data["q"]
    angle = data["angle"]
    frac = data["frac"]
    time = data["time"]
    weight = data["weight"]
    bkg = data.get("bkg", 0.0)
    
    # BW propagators
    g0_all = np.take(g0, kernel_config["g0_index"])
    print(f"\ng0_all shape: {g0_all.shape}")
    print(f"g0_all[0:5]: {g0_all[0:5]}")
    
    g0_m = np.take(mass, kernel_config["g0_mass_index"], axis=-1)
    print(f"\ng0_m shape: {g0_m.shape}")
    print(f"g0_m[0, 0:5]: {g0_m[0, 0:5]}")
    
    # Interpolation
    diff = (g0_m - kernel_config["gamma_min"]) / kernel_config["gamma_delta"]
    xbin = np.floor(diff).astype(np.intp)
    xbin = np.clip(xbin, 0, kernel_config["gamma_table"].shape[-1] - 2)
    delta = diff - xbin
    
    print(f"\ndiff[0, 0]: {diff[0, 0]}")
    print(f"xbin[0, 0]: {xbin[0, 0]}")
    print(f"delta[0, 0]: {delta[0, 0]}")
    
    idx = kernel_config["g0_index"] * kernel_config["gamma_table"].shape[-1] + xbin
    left = np.take(kernel_config["gamma_table"].ravel(), idx)
    right = np.take(kernel_config["gamma_table"].ravel(), idx + 1)
    g_interp = (right - left) * delta + left
    
    print(f"\ng_interp shape: {g_interp.shape}")
    print(f"g_interp[0, 0:5]: {g_interp[0, 0:5]}")
    
    g = g0_all * g_interp
    print(f"\ng shape: {g.shape}")
    print(f"g[0, 0:5]: {g[0, 0:5]}")
    
    g_bw = np.dot(g, kernel_config["matrix_gamma"])
    print(f"\ng_bw shape: {g_bw.shape}")
    print(f"g_bw[0, 0:5]: {g_bw[0, 0:5]}")
    
    # bw_dom
    m0_all = np.take(m0, kernel_config["m0_index"])
    m0_m = np.take(mass, kernel_config["mass_index"], axis=-1)
    bw_dom = m0_all**2 - m0_m**2 - 1j * m0_all * g_bw
    
    print(f"\nbw_dom shape: {bw_dom.shape}")
    print(f"bw_dom[0, 0]: {bw_dom[0, 0]}")
    
    # bw_p
    bw_dom_all = np.take(bw_dom, kernel_config["bw_order"], axis=-1)
    n_events_actual = bw_dom_all.shape[0]
    n_wave = kernel_config["matrix_angle"].shape[1]
    n_res = kernel_config["bw_order"].size // n_wave
    
    print(f"\nn_events_actual: {n_events_actual}")
    print(f"n_wave: {n_wave}")
    print(f"n_res: {n_res}")
    print(f"bw_order size: {kernel_config['bw_order'].size}")
    
    bw_dom_all_reshaped = bw_dom_all.reshape(n_events_actual, n_wave, n_res)
    bw_p = np.prod(bw_dom_all_reshaped, axis=-1)
    
    print(f"\nbw_p shape: {bw_p.shape}")
    print(f"bw_p[0, 0]: {bw_p[0, 0]}")
    print(f"bw_p[0, 1]: {bw_p[0, 1]}")
    
    # Check P calculation
    one_over_bw = 1.0 / bw_p
    print(f"\none_over_bw[0, 0]: {one_over_bw[0, 0]}")


if __name__ == "__main__":
    print("\nCUDA KERNEL DEBUG SESSION")
    print("="*70)
    
    # First check NumPy intermediates
    check_numpy_intermediates()
    
    # Then compare CUDA vs NumPy
    debug_forward_pass()
