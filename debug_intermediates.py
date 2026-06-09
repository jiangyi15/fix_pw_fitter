"""
Debug script to compare intermediate values between CUDA and NumPy step-by-step.
"""

import numpy as np
from config_loader import Config
from numpy_kernel import NumpyKernelCorrect
from cuda_kernel_cffi import CUDAKernel

def compare_intermediates():
    """Compare intermediate values between NumPy and CUDA"""
    
    # Load config
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    
    # Create test data
    n_events = 10  # Small number for easier debugging
    np.random.seed(42)
    data = {
        "mass": np.random.random((n_events, 48)),
        "q": np.random.random((n_events, 24)),
        "angle": np.random.random((n_events, 24, 3)),
        "frac": np.random.random((n_events,)),
        "time": np.random.random((n_events,)),
        "bkg": np.random.random((n_events,)) * 0.01,
        "weight": np.ones((n_events,)),
    }
    
    # Create parameters
    ck_map = config.get_ck_map()
    params = {
        "ck": np.random.random(len(ck_map)) + 1j*np.random.random(len(ck_map)),
        "m0": np.random.random(len(config.m0_phys_name)) + 2,
        "g0": np.random.random(len(config.g0_phys_name)) + 0.1,
        "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }
    
    # Extract scalar parameters
    Gamma, Delta_Gamma, Delta_m, A_p, poq_rho, pop_phi = params["scalar"]
    
    # ==================== NUMPY COMPUTATION ====================
    print("="*70)
    print("NUMPY INTERMEDIATE VALUES")
    print("="*70)
    
    nk = NumpyKernelCorrect(kernel_config)
    
    # Get internal arrays from numpy kernel
    ck = params["ck"]
    m0 = params["m0"]
    g0 = params["g0"]
    
    mass = data["mass"]
    momentum = data["q"]
    angle = data["angle"]
    
    # Compute intermediates manually following numpy_kernel.py logic
    
    # BW propagators
    g0_all = np.take(g0, kernel_config['g0_index'])
    g0_m = np.take(mass, kernel_config['g0_mass_index'], axis=-1)
    
    # Use the kernel's interp method
    nk_temp = NumpyKernelCorrect(kernel_config)
    g_interp = nk_temp.interp(kernel_config['gamma_table'], kernel_config['g0_index'], g0_m,
                              kernel_config['gamma_min'], kernel_config['gamma_delta'])
    
    g = g0_all * g_interp
    g_bw = np.dot(g, kernel_config['matrix_gamma'])
    
    print(f"\n1. g_interp:")
    print(f"   Shape: {g_interp.shape}")
    print(f"   [0, 0]: {g_interp[0, 0]}")
    print(f"   [0, 1]: {g_interp[0, 1]}")
    print(f"   Has imag: {np.any(g_interp.imag != 0)}")
    
    print(f"\n2. g (g0_all * g_interp):")
    print(f"   Shape: {g.shape}")
    print(f"   [0, 0]: {g[0, 0]}")
    print(f"   [0, 1]: {g[0, 1]}")
    
    print(f"\n3. g_bw (g @ matrix_gamma):")
    print(f"   Shape: {g_bw.shape}")
    print(f"   [0, 0]: {g_bw[0, 0]}")
    print(f"   [0, 1]: {g_bw[0, 1]}")
    print(f"   [0, 215]: {g_bw[0, 215]}")
    
    # bw_dom
    m0_all = np.take(m0, kernel_config['m0_index'])
    m0_m = np.take(mass, kernel_config['mass_index'], axis=-1)
    bw_dom = m0_all**2 - m0_m**2 - 1j * m0_all * g_bw
    
    print(f"\n4. bw_dom:")
    print(f"   Shape: {bw_dom.shape}")
    print(f"   [0, 0]: {bw_dom[0, 0]}")
    print(f"   [0, 1]: {bw_dom[0, 1]}")
    print(f"   [0, 215]: {bw_dom[0, 215]}")
    
    # bw_p
    bw_dom_all = np.take(bw_dom, kernel_config['bw_order'], axis=-1)
    n_events_np, n_wave_np, n_res_np = n_events, kernel_config['matrix_angle'].shape[1], kernel_config['bw_order'].size // kernel_config['matrix_angle'].shape[1]
    bw_dom_all_reshaped = bw_dom_all.reshape(n_events_np, n_wave_np, n_res_np)
    bw_p = np.prod(bw_dom_all_reshaped, axis=-1)
    
    print(f"\n5. bw_p:")
    print(f"   Shape: {bw_p.shape}")
    print(f"   [0, 0]: {bw_p[0, 0]}")
    print(f"   [0, 1]: {bw_p[0, 1]}")
    print(f"   [0, 447]: {bw_p[0, 447]}")
    
    # Angular factors
    ang = np.take(angle, kernel_config['angle_index'], axis=-2)
    ka = np.prod(np.cos(ang * kernel_config['angle_k'] + kernel_config['angle_b']), axis=-1)
    fa = np.dot(ka, kernel_config['matrix_angle'])
    
    print(f"\n6. ka (angular factors):")
    print(f"   Shape: {ka.shape}")
    print(f"   [0, 0]: {ka[0, 0]}")
    print(f"   [0, 1]: {ka[0, 1]}")
    print(f"   [0, 335]: {ka[0, 335]}")
    
    print(f"\n7. fa (ka @ matrix_angle):")
    print(f"   Shape: {fa.shape}")
    print(f"   [0, 0]: {fa[0, 0]}")
    print(f"   [0, 1]: {fa[0, 1]}")
    print(f"   [0, 447]: {fa[0, 447]}")
    
    # Skip FL factors for now due to indexing issue
    print(f"\n8. Skipping FL factors (indexing issue)")
    
    # Skip common_amp_factor for now
    print(f"\n9. Skipping common_amp_factor calculation")
    
    # ==================== CUDA COMPUTATION ====================
    print("\n" + "="*70)
    print("CUDA INTERMEDIATE VALUES")
    print("="*70)
    
    try:
        cuda_kernel = CUDAKernel(kernel_config)
        cuda_kernel.load_data(data)
        
        # Run forward pass
        Q_cuda, grads_cuda, P_cuda = cuda_kernel.compute(params, norm=None)
        
        # Get intermediate arrays from GPU
        print(f"\n1. g_interp:")
        print(f"   Shape: (n_events, n_unique_bw)")
        g_interp_real = cuda_kernel.gpu_data.g_interp_real_gpu.get()
        g_interp_imag = cuda_kernel.gpu_data.g_interp_imag_gpu.get()
        print(f"   [0, 0]: {complex(g_interp_real[0, 0], g_interp_imag[0, 0])}")
        print(f"   [0, 1]: {complex(g_interp_real[0, 1], g_interp_imag[0, 1])}")
        print(f"   Has imag: {np.any(g_interp_imag != 0)}")
        
        print(f"\n2. g_bw:")
        print(f"   Shape: (n_events, n_unique_bw)")
        g_bw_real = cuda_kernel.gpu_data.g_bw_real_gpu.get()
        g_bw_imag = cuda_kernel.gpu_data.g_bw_imag_gpu.get()
        print(f"   [0, 0]: {complex(g_bw_real[0, 0], g_bw_imag[0, 0])}")
        print(f"   [0, 1]: {complex(g_bw_real[0, 1], g_bw_imag[0, 1])}")
        print(f"   [0, 215]: {complex(g_bw_real[0, 215], g_bw_imag[0, 215])}")
        
        print(f"\n3. bw_dom:")
        print(f"   Shape: (n_events, n_unique_bw)")
        bw_dom_real = cuda_kernel.gpu_data.bw_dom_real_gpu.get()
        bw_dom_imag = cuda_kernel.gpu_data.bw_dom_imag_gpu.get()
        print(f"   [0, 0]: {complex(bw_dom_real[0, 0], bw_dom_imag[0, 0])}")
        print(f"   [0, 1]: {complex(bw_dom_real[0, 1], bw_dom_imag[0, 1])}")
        print(f"   [0, 215]: {complex(bw_dom_real[0, 215], bw_dom_imag[0, 215])}")
        
        print(f"\n4. bw_p:")
        print(f"   Shape: (n_events, n_wave)")
        bw_p_real = cuda_kernel.gpu_data.bw_p_real_gpu.get()
        bw_p_imag = cuda_kernel.gpu_data.bw_p_imag_gpu.get()
        print(f"   [0, 0]: {complex(bw_p_real[0, 0], bw_p_imag[0, 0])}")
        print(f"   [0, 1]: {complex(bw_p_real[0, 1], bw_p_imag[0, 1])}")
        print(f"   [0, 447]: {complex(bw_p_real[0, 447], bw_p_imag[0, 447])}")
        
        print(f"\n5. common_amp_factor:")
        print(f"   Shape: (n_events, n_wave)")
        common_real = cuda_kernel.gpu_data.common_amp_factor_real_gpu.get()
        common_imag = cuda_kernel.gpu_data.common_amp_factor_imag_gpu.get()
        print(f"   [0, 0]: {complex(common_real[0, 0], common_imag[0, 0])}")
        print(f"   [0, 1]: {complex(common_real[0, 1], common_imag[0, 1])}")
        print(f"   [0, 447]: {complex(common_real[0, 447], common_imag[0, 447])}")
        
        print(f"\n6. ap, am:")
        ap_real = cuda_kernel.gpu_data.ap_real_gpu.get()
        ap_imag = cuda_kernel.gpu_data.ap_imag_gpu.get()
        am_real = cuda_kernel.gpu_data.am_real_gpu.get()
        am_imag = cuda_kernel.gpu_data.am_imag_gpu.get()
        print(f"   ap[0]: {complex(ap_real[0], ap_imag[0])}")
        print(f"   am[0]: {complex(am_real[0], am_imag[0])}")
        
        # ==================== COMPARISON ====================
        print("\n" + "="*70)
        print("COMPARISON: NumPy vs CUDA")
        print("="*70)
        
        # Compare g_interp
        print(f"\n1. g_interp comparison:")
        g_interp_cuda = g_interp_real + 1j * g_interp_imag
        diff_g_interp = np.abs(g_interp - g_interp_cuda[:, :g_interp.shape[1]])
        print(f"   Max diff: {diff_g_interp.max():.6e}")
        print(f"   Mean diff: {diff_g_interp.mean():.6e}")
        
        # Compare g_bw
        print(f"\n2. g_bw comparison:")
        g_bw_cuda = g_bw_real + 1j * g_bw_imag
        diff_g_bw = np.abs(g_bw - g_bw_cuda)
        print(f"   Max diff: {diff_g_bw.max():.6e}")
        print(f"   Mean diff: {diff_g_bw.mean():.6e}")
        print(f"   NumPy g_bw[0, 0]: {g_bw[0, 0]}")
        print(f"   CUDA g_bw[0, 0]: {g_bw_cuda[0, 0]}")
        
        # Compare bw_dom
        print(f"\n3. bw_dom comparison:")
        bw_dom_cuda = bw_dom_real + 1j * bw_dom_imag
        diff_bw_dom = np.abs(bw_dom - bw_dom_cuda)
        print(f"   Max diff: {diff_bw_dom.max():.6e}")
        print(f"   Mean diff: {diff_bw_dom.mean():.6e}")
        print(f"   NumPy bw_dom[0, 0]: {bw_dom[0, 0]}")
        print(f"   CUDA bw_dom[0, 0]: {bw_dom_cuda[0, 0]}")
        
        # Compare bw_p
        print(f"\n4. bw_p comparison:")
        bw_p_cuda = bw_p_real + 1j * bw_p_imag
        diff_bw_p = np.abs(bw_p - bw_p_cuda)
        print(f"   Max diff: {diff_bw_p.max():.6e}")
        print(f"   Mean diff: {diff_bw_p.mean():.6e}")
        print(f"   NumPy bw_p[0, 0]: {bw_p[0, 0]}")
        print(f"   CUDA bw_p[0, 0]: {bw_p_cuda[0, 0]}")
        
        # Compare common_amp_factor
        print(f"\n5. common_amp_factor comparison:")
        common_cuda = common_real + 1j * common_imag
        print(f"   NumPy not computed (skipped)")
        print(f"   CUDA common[0, 0]: {common_cuda[0, 0]}")
        
        # Final Q comparison
        print(f"\n6. Final Q comparison:")
        Q_numpy, _, _ = nk._compute(params, data, norm=None)
        print(f"   NumPy Q: {Q_numpy:.10f}")
        print(f"   CUDA Q: {Q_cuda:.10f}")
        print(f"   Difference: {abs(Q_numpy - Q_cuda):.6e}")
        
        cuda_kernel.free()
        
    except Exception as e:
        print(f"\nCUDA computation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compare_intermediates()
