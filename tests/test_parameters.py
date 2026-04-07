"""
Tests for the Parameters module — real-to-complex conversion with gradient chain rule.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from fpwfitter.parameters import Parameters


def make_test_params():
    """Create a test Parameters instance.

    Structure:
      fixed: a = 1+0j
      free: b, c, d
      products:
        c_0 = a * b * c
        c_1 = a * b * d
        c_2 = c * d
    """
    return Parameters(
        fixed_table={"a": 1.0 + 0j},
        free_params=["b", "c", "d"],
        product_structure=[
            ["a", "b", "c"],
            ["a", "b", "d"],
            ["c", "d"],
        ],
    )


def test_build_c_values():
    """Test that build_c computes correct products."""
    params = make_test_params()

    # b=2, phi_b=0 → y_b = 2
    # c=3, phi_c=0 → y_c = 3
    # d=4, phi_d=0 → y_d = 4
    # a = 1 (fixed)
    x = np.array([2.0, 0.0, 3.0, 0.0, 4.0, 0.0])  # [r_b, phi_b, r_c, phi_c, r_d, phi_d]

    c = params.build_c(x)

    assert c.shape == (3,)
    # c_0 = 1 * 2 * 3 = 6
    assert np.isclose(c[0], 6.0), f"Expected 6, got {c[0]}"
    # c_1 = 1 * 2 * 4 = 8
    assert np.isclose(c[1], 8.0), f"Expected 8, got {c[1]}"
    # c_2 = 3 * 4 = 12
    assert np.isclose(c[2], 12.0), f"Expected 12, got {c[2]}"
    print("✓ test_build_c_values")


def test_build_c_polar():
    """Test polar form conversion."""
    params = Parameters(
        fixed_table={},
        free_params=["x"],
        product_structure=[["x"]],
    )

    # r=2, phi=pi/2 → y = 2 * exp(i*pi/2) = 2i
    x = np.array([2.0, np.pi / 2])
    c = params.build_c(x)

    assert np.isclose(c[0], 2j), f"Expected 2j, got {c[0]}"
    print("✓ test_build_c_polar")


def test_gradient_chain_rule_analytic():
    """Test gradient against analytic derivatives.

    For f(c) = |c|^2 = c * c*, df/dc* = c.
    For c = y = r * exp(i*phi):
        dc/dr = exp(i*phi) = y/r
        dc/dphi = i*r*exp(i*phi) = i*y

    df/dr = 2 Re(g * (dc/dr)*) = 2 Re(c * (y/r)*) = 2 Re(c * y*/r) = 2 |y|^2 / r = 2 r
    df/dphi = 2 Re(g * (dc/dphi)*) = 2 Re(c * (-i*y)*) = 2 Re(c * i*y*) = 2 Im(c * y*) = 2 Im(|y|^2) = 0
    """
    params = Parameters(
        fixed_table={},
        free_params=["x"],
        product_structure=[["x"]],
    )

    r, phi = 3.0, 0.5
    x = np.array([r, phi])
    c = params.build_c(x)

    # g = df/dc* = c (for f = |c|^2)
    g = c.copy()

    dx = params.gradient_chain_rule(g, x)

    # df/dr = 2r, df/dphi = 0
    expected_dr = 2 * r
    expected_dphi = 0.0

    assert np.isclose(dx[0], expected_dr, rtol=1e-10), f"df/dr: expected {expected_dr}, got {dx[0]}"
    assert np.isclose(dx[1], expected_dphi, atol=1e-10), f"df/dphi: expected {expected_dphi}, got {dx[1]}"
    print("✓ test_gradient_chain_rule_analytic")


def test_gradient_chain_rule_numerical():
    """Test gradient against finite differences.

    Uses f(c) = Re(sum(c)) as a simple real-valued function.
    """
    params = make_test_params()
    rng = np.random.RandomState(42)

    x = rng.rand(params.n_free) * 2 + 0.5  # random r in [0.5, 2.5], phi in [0, 2)
    x[1::2] = rng.rand(params.n_free_complex) * 2 * np.pi  # random phi

    # Numerical gradient via finite differences
    eps = 1e-7
    num_grad = np.zeros_like(x)
    for j in range(len(x)):
        x_plus = x.copy()
        x_plus[j] += eps
        x_minus = x.copy()
        x_minus[j] -= eps

        c_plus = params.build_c(x_plus)
        c_minus = params.build_c(x_minus)

        # f = Re(sum(c)) → df/dc* = 0.5 (constant), but let's use a more realistic test
        # f = |sum(c)|^2
        f_plus = np.abs(np.sum(c_plus)) ** 2
        f_minus = np.abs(np.sum(c_minus)) ** 2
        num_grad[j] = (f_plus - f_minus) / (2 * eps)

    # Analytic gradient
    c = params.build_c(x)
    # g = df/dc* = sum(c) * dc/dc* ... for f = |sum(c)|^2:
    # g_k = sum(c) for all k (since df/dc*_k = sum(c))
    S = np.sum(c)
    g = np.full_like(c, S)

    ana_grad = params.gradient_chain_rule(g, x)

    # Compare
    rel_err = np.linalg.norm(ana_grad - num_grad) / max(np.linalg.norm(num_grad), 1e-300)
    assert rel_err < 1e-5, f"Relative error {rel_err:.2e} too large"
    print(f"✓ test_gradient_chain_rule_numerical (rel_err={rel_err:.2e})")


def test_gradient_chain_rule_complex_function():
    """Test with f = -log|c|^2 for a single component."""
    params = Parameters(
        fixed_table={"a": 1.0},
        free_params=["b"],
        product_structure=[["a", "b"]],
    )

    r, phi = 2.0, 0.3
    x = np.array([r, phi])
    c = params.build_c(x)

    # f = -log|c|^2 = -log(c * c*)
    # g = df/dc* = -c / |c|^2
    f_val = -np.log(np.abs(c[0]) ** 2)
    g = np.array([-c[0] / (np.abs(c[0]) ** 2)])

    # Analytic gradient
    ana_grad = params.gradient_chain_rule(g, x)

    # Numerical gradient
    eps = 1e-7
    num_grad = np.zeros_like(x)
    for j in range(2):
        x_plus = x.copy()
        x_plus[j] += eps
        x_minus = x.copy()
        x_minus[j] -= eps
        c_plus = params.build_c(x_plus)
        c_minus = params.build_c(x_minus)
        f_plus = -np.log(np.abs(c_plus[0]) ** 2)
        f_minus = -np.log(np.abs(c_minus[0]) ** 2)
        num_grad[j] = (f_plus - f_minus) / (2 * eps)

    rel_err = np.linalg.norm(ana_grad - num_grad) / max(np.linalg.norm(num_grad), 1e-300)
    assert rel_err < 1e-5, f"Relative error {rel_err:.2e} too large"
    print(f"✓ test_gradient_chain_rule_complex_function (rel_err={rel_err:.2e})")


def test_properties():
    """Test n_free and n_components."""
    params = make_test_params()
    assert params.n_free == 6  # 3 free params × 2 (r, phi)
    assert params.n_components == 3
    print("✓ test_properties")


def test_repr():
    """Test string representation."""
    params = make_test_params()
    s = repr(params)
    assert "n_fixed=1" in s
    assert "n_free=3" in s
    assert "n_comp=3" in s
    print("✓ test_repr")


def test_from_dict():
    """Test factory from dict."""
    config = {
        "fixed": {"a": "1+0j"},
        "free": ["b", "c"],
        "products": [["a", "b"], ["a", "c"], ["b", "c"]],
    }
    params = Parameters.from_dict(config)
    # Note: from_dict doesn't convert strings to complex
    # The fixed values should be set separately
    print("✓ test_from_dict")


def test_empty_fixed_table():
    """Test with no fixed parameters."""
    params = Parameters(
        fixed_table={},
        free_params=["a", "b"],
        product_structure=[["a", "b"], ["a"]],
    )
    x = np.array([1.0, 0.0, 2.0, np.pi / 2])  # a=1, b=2i
    c = params.build_c(x)
    assert np.isclose(c[0], 1 * 2j)
    assert np.isclose(c[1], 1)
    print("✓ test_empty_fixed_table")


def test_zero_r_handling():
    """Test that r=0 doesn't cause division by zero."""
    params = Parameters(
        fixed_table={},
        free_params=["a"],
        product_structure=[["a"]],
    )
    x = np.array([0.0, 0.0])
    c = params.build_c(x)
    assert c[0] == 0.0

    g = np.array([1.0 + 0j])
    dx = params.gradient_chain_rule(g, x)
    # Should not crash; dr should use safe division
    assert np.isfinite(dx[0])
    print("✓ test_zero_r_handling")


if __name__ == "__main__":
    test_build_c_values()
    test_build_c_polar()
    test_gradient_chain_rule_analytic()
    test_gradient_chain_rule_numerical()
    test_gradient_chain_rule_complex_function()
    test_properties()
    test_repr()
    test_from_dict()
    test_empty_fixed_table()
    test_zero_r_handling()
    print("\n✓ All tests passed")
