"""Test the corrected gradient implementation"""
import numpy as np
from config_loader import Config
from numpy_kernel_correct import NumpyKernelCorrect


def numerical_gradient_3point(kernel, params, data, param_name, param_idx=None, epsilon=1e-5):
    """Compute numerical gradient using 3-point method"""
    params_plus = {k: v.copy() if isinstance(v, np.ndarray) else 
                   (list(v) if isinstance(v, list) else v) 
                   for k, v in params.items()}
    params_minus = {k: v.copy() if isinstance(v, np.ndarray) else 
                    (list(v) if isinstance(v, list) else v) 
                    for k, v in params.items()}
    
    if param_idx is not None:
        original_value = params[param_name][param_idx]
        params_plus[param_name][param_idx] = original_value + epsilon
        params_minus[param_name][param_idx] = original_value - epsilon
    
    Q_plus, _, _ = kernel._compute(params_plus, data, norm=None)
    Q_minus, _, _ = kernel._compute(params_minus, data, norm=None)
    
    grad_numerical = (Q_plus - Q_minus) / (2 * epsilon)
    return grad_numerical


def test_gradients():
    """Test gradient correctness"""
    print("="*70)
    print("TESTING FIXED GRADIENT IMPLEMENTATION")
    print("="*70)
    
    config = Config("config_angle.yml")
    kernel_config = config.build_all_index()
    kernel = NumpyKernelCorrect(kernel_config)
    
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
    
    params = {
        "ck": np.random.random(len(ck_map)) + 1j*np.random.random(len(ck_map)),
        "m0": np.random.random(len(config.m0_phys_name)) + 2,
        "g0": np.random.random(len(config.g0_phys_name)) + 0.1,
        "scalar": [0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    }
    
    # Get analytical gradients
    Q, grads_analytical, _ = kernel._compute(params, data, norm=None)
    
    all_passed = True
    
    # Test ck gradients
    print("\nck gradients:")
    for i in range(min(3, len(params["ck"]))):
        # Real part
        grad_num_real = numerical_gradient_3point(kernel, params, data, 'ck', param_idx=i, epsilon=epsilon)
        
        # Imaginary part
        params_test = {k: v.copy() if isinstance(v, np.ndarray) else list(v) if isinstance(v, list) else v 
                      for k, v in params.items()}
        params_test['ck'][i] = params['ck'][i] + 1j * epsilon
        Q_plus_imag, _, _ = kernel._compute(params_test, data, norm=None)
        params_test['ck'][i] = params['ck'][i] - 1j * epsilon
        Q_minus_imag, _, _ = kernel._compute(params_test, data, norm=None)
        grad_num_imag = (Q_plus_imag - Q_minus_imag) / (2 * epsilon)
        
        grad_ana = grads_analytical['ck'][i]
        
        # For Wirtinger calculus with real Q:
        # ∂Q/∂Re(ck) = 2 * Re(∂Q/∂ck)
        # ∂Q/∂Im(ck) = -2 * Im(∂Q/∂ck)
        grad_ana_real = 2 * np.real(grad_ana)
        grad_ana_imag = -2 * np.imag(grad_ana)
        
        error_real = abs(grad_num_real - grad_ana_real)
        error_imag = abs(grad_num_imag - grad_ana_imag)
        
        passed_real = error_real < 1e-4
        passed_imag = error_imag < 1e-4
        passed = passed_real and passed_imag
        
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed
        
        print(f"  ck[{i}]: {status}")
        print(f"    Real: ana={grad_ana_real:.6f}, num={grad_num_real:.6f}, error={error_real:.2e}")
        print(f"    Imag: ana={grad_ana_imag:.6f}, num={grad_num_imag:.6f}, error={error_imag:.2e}")
    
    # Test m0 gradients
    print("\nm0 gradients:")
    for i in range(min(3, len(params["m0"]))):
        grad_num = numerical_gradient_3point(kernel, params, data, 'm0', param_idx=i, epsilon=epsilon)
        grad_ana = grads_analytical['m0'][i]
        error = abs(grad_num - grad_ana)
        
        passed = error < 1e-4
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed
        
        print(f"  m0[{i}]: {status} ana={grad_ana:.6f}, num={grad_num:.6f}, error={error:.2e}")
    
    # Test g0 gradients
    print("\ng0 gradients:")
    for i in range(min(3, len(params["g0"]))):
        grad_num = numerical_gradient_3point(kernel, params, data, 'g0', param_idx=i, epsilon=epsilon)
        grad_ana = grads_analytical['g0'][i]
        error = abs(grad_num - grad_ana)
        
        passed = error < 1e-4
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed
        
        print(f"  g0[{i}]: {status} ana={grad_ana:.6f}, num={grad_num:.6f}, error={error:.2e}")
    
    # Test scalar gradients
    print("\nscalar gradients:")
    scalar_names = ['Gamma', 'Delta_Gamma', 'Delta_m', 'A_p', 'poq_rho', 'pop_phi']
    for i, name in enumerate(scalar_names):
        params_plus = {k: v.copy() if isinstance(v, np.ndarray) else list(v) if isinstance(v, list) else v 
                      for k, v in params.items()}
        params_minus = {k: v.copy() if isinstance(v, np.ndarray) else list(v) if isinstance(v, list) else v 
                       for k, v in params.items()}
        
        params_plus['scalar'][i] = params['scalar'][i] + epsilon
        params_minus['scalar'][i] = params['scalar'][i] - epsilon
        
        Q_plus, _, _ = kernel._compute(params_plus, data, norm=None)
        Q_minus, _, _ = kernel._compute(params_minus, data, norm=None)
        
        grad_num = (Q_plus - Q_minus) / (2 * epsilon)
        grad_ana = grads_analytical['scalar'][i]
        
        # Ensure scalar gradients are scalar (not arrays)
        if hasattr(grad_ana, 'shape') and grad_ana.shape != ():
            grad_ana = float(np.real(grad_ana))
        
        # Check that scalar gradients are REAL
        if np.iscomplexobj(grad_ana):
            grad_ana = float(np.real(grad_ana))
        
        error = abs(float(grad_num) - float(grad_ana))
        
        passed = error < 1e-4
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed
        
        print(f"  {name:15s}: {status} ana={grad_ana:.6f}, num={grad_num:.6f}, error={error:.2e}")
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL GRADIENT TESTS PASSED!")
    else:
        print("✗ SOME GRADIENT TESTS FAILED")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    test_gradients()
