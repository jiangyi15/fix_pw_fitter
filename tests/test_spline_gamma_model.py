#!/usr/bin/env python3
"""Tests for the BSplineGammaModel (B-spline running width).

Run with::

    pytest tests/test_spline_gamma_model.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ampfit.particle_model import build_particle
from ampfit.particle_model.spline_gamma_model import (
    bspline_basis, BSplineGammaModel, _SplineMassFixTransform,
)


# ═══════════════════════════════════════════════════════════════════
# 1. B-spline basis functions
# ═══════════════════════════════════════════════════════════════════

class TestBSplineBasis:
    """Clamped B-spline basis evaluation."""

    def test_partition_of_unity(self):
        """B-spline basis sums to 1 everywhere in [lo, hi]."""
        bp = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        x = np.linspace(0, 4, 50)
        basis = bspline_basis(x, bp, order=3)
        total = basis.sum(axis=1)
        assert np.max(np.abs(total - 1.0)) < 1e-12

    def test_linear_order(self):
        """Order-1 B-spline gives linear interpolation."""
        bp = np.array([0.0, 2.0, 4.0])
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        basis = bspline_basis(x, bp, order=1)
        # 3 breakpoints, order 1 → 3+1-1 = 3 basis functions
        assert basis.shape[1] == 3
        # At x=0: only B_0 is 1
        assert abs(basis[0, 0] - 1.0) < 1e-12
        assert abs(basis[0, 1]) < 1e-12

    def test_outside_range(self):
        """Basis is zero outside the breakpoint range."""
        bp = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        x = np.array([-0.1, 5.1])
        basis = bspline_basis(x, bp, order=3)
        assert np.all(basis == 0.0)

    def test_few_breakpoints_raises(self):
        """Too few breakpoints for chosen order raises."""
        with pytest.raises(ValueError):
            bspline_basis(np.array([0.5]), [0.0, 1.0], order=2)  # need 3 for order 2


# ═══════════════════════════════════════════════════════════════════
# 2. _SplineMassFixTransform
# ═══════════════════════════════════════════════════════════════════

class TestSplineMassFixTransform:
    """Transform that fixes mass to config value."""

    def test_forward(self):
        tr = _SplineMassFixTransform("test_mass", mass_default=0.5)
        assert tr.input_names == []
        assert tr.output_names == ["test_mass"]

        d = {"test_re_B_0": 0.1, "other": 99.0}
        result = tr.apply_forward(d)
        assert result["test_mass"] == 0.5
        assert result["test_re_B_0"] == 0.1
        assert result["other"] == 99.0

    def test_backward_removes_mass(self):
        tr = _SplineMassFixTransform("test_mass", mass_default=0.5)

        # Apply forward first to store _last_input
        d = {"test_re_B_0": 0.1}
        tr.apply_forward(d)

        grad = {"test_re_B_0": 1.0, "test_mass": 2.0, "other": 3.0}
        grad_out = tr.apply_backward(grad)
        assert "test_mass" not in grad_out          # output-only → removed
        assert grad_out["test_re_B_0"] == 1.0        # untouched
        assert grad_out["other"] == 3.0              # untouched

    def test_different_name(self):
        """Works with non-default mass name."""
        tr = _SplineMassFixTransform("rhoA_mass", mass_default=1.23)
        result = tr.apply_forward({"x": 1.0})
        assert result["rhoA_mass"] == 1.23


# ═══════════════════════════════════════════════════════════════════
# 3. BSplineGammaModel
# ═══════════════════════════════════════════════════════════════════

class TestBSplineGammaModel:
    """Full BSplineGammaModel."""

    def test_n_basis(self):
        """Number of basis functions = breakpoints + order - 1."""
        m = build_particle("test", model="BSpline", mass=1.0,
                           knots=[0.0, 1.0, 2.0, 3.0])
        # 4 breakpoints, order 3 → 4+3-1 = 6
        assert m.n_basis == 6

    def test_gamma_count(self):
        """Two gamma components per basis function (re + im)."""
        m = build_particle("test", model="BSpline", mass=1.0,
                           knots=[0.0, 1.0, 2.0, 3.0])
        assert m.get_gamma_count() == 2 * m.n_basis

    def test_gamma_names(self):
        """Gamma names are {name}_re_B_{i} and {name}_im_B_{i}."""
        m = build_particle("test", model="BSpline", mass=1.0,
                           knots=[0.0, 1.0, 2.0, 3.0])
        names = m.get_gamma_name()
        assert names[0] == "test_re_B_0"
        assert names[1] == "test_im_B_0"
        assert names[2] == "test_re_B_1"
        assert len(names) == 2 * m.n_basis

    def test_gamma_structure(self):
        """gamma_{2k}=B_k, gamma_{2k+1}=i·B_k."""
        m = build_particle("test", model="BSpline", mass=1.0,
                           knots=[0.0, 1.0, 2.0, 3.0, 4.0])
        g = m.gamma(np.array([2.0]))
        for k in range(m.n_basis):
            Bk = g[2 * k][0].real      # real part of gamma_{2k}
            Bk_from_im = g[2 * k + 1][0].imag  # imag part of gamma_{2k+1}
            assert abs(Bk - Bk_from_im) < 1e-15, f"B_{k} mismatch"
            assert abs(g[2 * k][0].imag) < 1e-15       # gamma_{2k} is purely real
            assert abs(g[2 * k + 1][0].real) < 1e-15    # gamma_{2k+1} is purely imag

    def test_gamma_partition_of_unity(self):
        """Sum of all B_k should be 1 everywhere in range."""
        m = build_particle("test", model="BSpline", mass=1.0,
                           knots=[0.0, 1.0, 2.0, 3.0, 4.0])
        x = np.linspace(0, 4, 50)
        g = m.gamma(x)
        # B_k = real part of gamma_{2k}
        B_sum = sum(g[2 * k].real for k in range(m.n_basis))
        assert np.max(np.abs(B_sum - 1.0)) < 1e-12

    def test_defaults_empty(self):
        """get_defaults returns {} (mass fixed, gamma are free vars)."""
        m = build_particle("test", model="BSpline", mass=1.0,
                           knots=[0.0, 1.0, 2.0, 3.0])
        assert m.get_defaults() == {}

    def test_make_mass_width_transform(self):
        """Transform fixes mass."""
        m = build_particle("test", model="BSpline", mass=0.5,
                           knots=[0.0, 1.0, 2.0, 3.0])
        tr = m.make_mass_width_transform()
        assert tr.output_names == ["test_mass"]
        d = tr.apply_forward({"x": 1.0})
        assert d["test_mass"] == 0.5

    def test_complex_sum(self):
        """Σ g_j·gamma_j = Σ re_k·B_k + i·Σ im_k·B_k."""
        m = build_particle("test", model="BSpline", mass=1.0,
                           knots=[0.0, 1.0, 2.0, 3.0, 4.0])
        g = m.gamma(np.array([2.0]))

        # Pick coefficients: re = [0.5, -0.3, 0.2, ...], im = [0.1, 0.0, -0.05, ...]
        rng = np.random.default_rng(42)
        re = rng.uniform(-1, 1, m.n_basis)
        im = rng.uniform(-0.5, 0.5, m.n_basis)

        # Build g0 vector: [re_0, im_0, re_1, im_1, ...]
        g0 = np.column_stack([re, im]).ravel()

        total = sum(g0[j] * g[j][0] for j in range(2 * m.n_basis))

        # Expected: Σ re_k·B_k + i · Σ im_k·B_k
        Bk = np.array([g[2 * k][0].real for k in range(m.n_basis)])
        expected = re @ Bk + 1j * (im @ Bk)
        assert abs(total - expected) < 1e-12


# ═══════════════════════════════════════════════════════════════════
# 4. Construction edge cases
# ═══════════════════════════════════════════════════════════════════

class TestBSplineConstruction:
    """Model construction edge cases."""

    def test_no_knots_raises(self):
        """Missing 'knots' raises."""
        with pytest.raises(ValueError, match="knots"):
            build_particle("test", model="BSpline", mass=1.0)

    def test_too_few_knots_raises(self):
        """Too few knots for chosen order raises."""
        with pytest.raises(ValueError):
            build_particle("test", model="BSpline", mass=1.0,
                           knots=[0.0, 1.0], order=3)

    def test_order_2(self):
        """Order 2 (quadratic) works."""
        m = build_particle("test", model="BSpline", mass=1.0,
                           knots=[0.0, 1.0, 2.0, 3.0], order=2)
        assert m.n_basis == 5  # 4 + 2 - 1 = 5

        x = np.linspace(0, 3, 20)
        g = m.gamma(x)
        B_sum = sum(g[2 * k].real for k in range(m.n_basis))
        assert np.max(np.abs(B_sum - 1.0)) < 1e-12

    def test_order_1(self):
        """Order 1 (linear) works."""
        m = build_particle("test", model="BSpline", mass=1.0,
                           knots=[0.0, 1.0, 2.0, 3.0, 4.0], order=1)
        x = np.array([0.0, 2.0, 4.0])
        g = m.gamma(x)
        B_sum = sum(g[2 * k].real for k in range(m.n_basis))
        assert np.max(np.abs(B_sum - 1.0)) < 1e-12

    def test_registered(self):
        """Model is registered as 'BSpline'."""
        from ampfit.particle_model import ALL_MODELS
        assert "BSpline" in ALL_MODELS
