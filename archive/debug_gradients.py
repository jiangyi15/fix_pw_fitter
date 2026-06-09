"""
Debug gradient computation by comparing intermediate values
"""
import numpy as np
from config_loader import Config
from numpy_kernel_correct import NumpyKernelCorrect


def numerical_gradient_param(kernel, params, data, param_name, idx, epsilon=1e-5, imag=False):
    """Compute numerical gradient for a parameter"""
    params_plus = {k: v.copy() if isinstance(v, np.ndarray) else list(v) if isinstance(v, list) else v 
                   for k, v in params.items()}
    params_minus = {k: v.copy() if isinstance(v, np.ndarray) else list(v) if isinstance(v, list) else v 
                    for k, v in params.items()}
    
    if imag:
        params_plus[param_name][idx] += 1j * epsilon
        params_minus[param_name][idx] -= 1j * epsilon
    else:
        params_plus[param_name][idx] += epsilon
        params_minus[param_name][idx] -= epsilon
    
    Q_plus, _, _ = kernel._compute(params_plus, data, norm=None)
    Q_minus, _, _ = kernel._compute(params_minus, data, norm=None)
    
    return (Q_plus - Q_minus) / (2 * epsilon)


def debug_simple_case():
    """Debug with a simple case"""
    print("="*70)
    print("DEBUGGING GRADIENT COMPUTATION")
    print("="*70)
    
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    kernel = NumpyKernelCorrect(kernel_config)
    
    n_events = 10  # Very small for debugging
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
    
    # Compute analytical gradients
    Q, grads, P = kernel._compute(params, data, norm=None)
    
    print(f"\nForward pass Q = {Q:.6f}")
    
    # Test ck[0] gradient
    print("\n" + "="*70)
    print("DEBUGGING ck[0] GRADIENT")
    print("="*70)
    
    grad_ana_ck0 = grads['ck'][0]
    grad_num_ck0_real = numerical_gradient_param(kernel, params, data, 'ck', 0, epsilon=1e-7, imag=False)
    grad_num_ck0_imag = numerical_gradient_param(kernel, params, data, 'ck', 0, epsilon=1e-7, imag=True)
    
    print(f"Analytical: ∂Q/∂ck[0] = {grad_ana_ck0:.6f}")
    print(f"Numerical (real perturbation): {grad_num_ck0_real:.6f}")
    print(f"Numerical (imag perturbation): {grad_num_ck0_imag:.6f}")
    
    print(f"\nNumerical gradients represent:")
    print(f"  ∂Q/∂Re(ck[0]) = {grad_num_ck0_real:.6f}")
    print(f"  ∂Q/∂Im(ck[0]) = {grad_num_ck0_imag:.6f}")
    
    print(f"\nIf ∂Q/∂ck = {grad_ana_ck0:.6f}, then (using Wirtinger):")
    print(f"  ∂Q/∂Re(ck) = 2*Re(∂Q/∂ck) = {2*np.real(grad_ana_ck0):.6f}")
    print(f"  ∂Q/∂Im(ck) = -2*Im(∂Q/∂ck) = {-2*np.imag(grad_ana_ck0):.6f}")
    
    print(f"\nDiscrepancy factors:")
    print(f"  Real: {grad_num_ck0_real / (2*np.real(grad_ana_ck0)):.2f}x")
    print(f"  Imag: {grad_num_ck0_imag / (-2*np.imag(grad_ana_ck0)):.2f}x")
    
    # Test A_p gradient (should be correct)
    print("\n" + "="*70)
    print("DEBUGGING A_p GRADIENT (should be correct)")
    print("="*70)
    
    params_plus = {k: v.copy() if isinstance(v, np.ndarray) else list(v) if isinstance(v, list) else v 
                   for k, v in params.items()}
    params_minus = {k: v.copy() if isinstance(v, np.ndarray) else list(v) if isinstance(v, list) else v 
                    for k, v in params.items()}
    
    params_plus['scalar'][3] += 1e-7
    params_minus['scalar'][3] -= 1e-7
    
    Q_plus, _, _ = kernel._compute(params_plus, data, norm=None)
    Q_minus, _, _ = kernel._compute(params_minus, data, norm=None)
    grad_num_Ap = (Q_plus - Q_minus) / (2e-7)
    
    grad_ana_Ap = grads['scalar'][3]
    
    print(f"Analytical: {grad_ana_Ap:.6f}")
    print(f"Numerical: {grad_num_Ap:.6f}")
    print(f"Error: {abs(grad_ana_Ap - grad_num_Ap):.2e}")
    print(f"Status: {'✓ CORRECT' if abs(grad_ana_Ap - grad_num_Ap) < 1e-5 else '✗ WRONG'}")
    
    # Test m0[0] gradient
    print("\n" + "="*70)
    print("DEBUGGING m0[0] GRADIENT")
    print("="*70)
    
    grad_ana_m00 = grads['m0'][0]
    grad_num_m00 = numerical_gradient_param(kernel, params, data, 'm0', 0, epsilon=1e-7, imag=False)
    
    print(f"Analytical: {grad_ana_m00:.6f}")
    print(f"Numerical: {grad_num_m00:.6f}")
    print(f"Error: {abs(grad_ana_m00 - grad_num_m00):.2e}")
    print(f"Ratio: {grad_num_m00 / grad_ana_m00 if abs(grad_ana_m00) > 1e-10 else 'inf':.2f}x")


if __name__ == "__main__":
    debug_simple_case()
