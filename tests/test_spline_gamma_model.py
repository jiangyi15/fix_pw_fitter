#!/usr/bin/env python3
"""Tests for BSplineGammaModel (B-spline running width — interior only).

Run with::

    pytest tests/test_spline_gamma_model.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ampfit.particle_model import build_particle, ALL_MODELS
from ampfit.particle_model.spline_gamma_model import (
    bspline_basis_all, bspline_basis_interior,
    BSplineGammaModel, _SplineMassFixTransform,
)


# ═══════════════════════════════════════════════════════════════════
# 1. B-spline basis
# ═══════════════════════════════════════════════════════════════════

class TestBSplineBasis:
    """Full and interior B-spline basis."""

    def test_full_partition_of_unity(self):
        """Full B-spline sums to 1 everywhere."""
        bp = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        x = np.linspace(0, 5, 100)
        basis, n = bspline_basis_all(x, bp, order=3)
        assert np.max(np.abs(basis.sum(axis=1) - 1.0)) < 1e-12

    def test_interior_partition_of_unity(self):
        """Interior B-spline counts correctly."""
        bp = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        x = np.linspace(0.001, 5.999, 100)
        basis, n = bspline_basis_interior(x, bp, order=3)
        assert n == 3  # 7 - 3 - 1 = 3

    def test_n_free_formula(self):
        """n_free = n_knots - order - 1."""
        for n_knots in range(5, 10):
            for order in [1, 2, 3]:
                if n_knots <= order + 1:
                    continue
                bp = np.linspace(0, 1, n_knots)
                _, n = bspline_basis_interior(bp, bp, order)
                assert n == n_knots - order - 1

    def test_peak_odd_order(self):
        """Odd-order (1, 3) peaks at knot positions."""
        bp = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        for order in [1, 3]:
            x = np.linspace(0, 6, 601)
            basis, n = bspline_basis_interior(x, bp, order)
            for k in range(n):
                peak = x[np.argmax(basis[:, k])]
                assert any(abs(peak - t) < 0.01 for t in bp), \
                    f"order {order}: peak {peak} not at knot"

    def test_peak_even_order(self):
        """Even-order (2) peaks between knots (bin centres)."""
        bp = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        x = np.linspace(0, 6, 601)
        basis, n = bspline_basis_interior(x, bp, order=2)
        for k in range(n):
            peak = x[np.argmax(basis[:, k])]
            midpoints = [(bp[i] + bp[i+1]) / 2 for i in range(len(bp)-1)]
            assert any(abs(peak - m) < 0.01 for m in midpoints), \
                f"peak {peak} not at bin centre"


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
        tr.apply_forward({"test_re_B_0": 0.1})
        grad = {"test_re_B_0": 1.0, "test_mass": 2.0, "other": 3.0}
        grad_out = tr.apply_backward(grad)
        assert "test_mass" not in grad_out
        assert grad_out["test_re_B_0"] == 1.0
        assert grad_out["other"] == 3.0


# ═══════════════════════════════════════════════════════════════════
# 3. BSplineGammaModel
# ═══════════════════════════════════════════════════════════════════

class TestBSplineGammaModel:
    """Full BSplineGammaModel."""

    BP7 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    BP5 = [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_n_free(self):
        """n_free = n_knots - order - 1."""
        m = build_particle("test", model="BSpline", mass=1.0, knots=self.BP7)
        assert m.n_free == 3  # 7 - 3 - 1

    def test_gamma_count(self):
        """Two gamma components per free basis (re + im)."""
        m = build_particle("test", model="BSpline", mass=1.0, knots=self.BP7)
        assert m.get_gamma_count() == 2 * m.n_free

    def test_gamma_names(self):
        """Names: {name}_re_B_{i} and {name}_im_B_{i}."""
        m = build_particle("test", model="BSpline", mass=1.0, knots=self.BP5)
        names = m.get_gamma_name()
        assert names[0] == "test_re_B_0"
        assert names[1] == "test_im_B_0"
        assert len(names) == 2 * m.n_free  # BP5: 5-3-1=1 → 2 gamma comps

    def test_gamma_structure(self):
        """gamma_{2k} real; gamma_{2k+1} pure imag = i * real part."""
        m = build_particle("test", model="BSpline", mass=1.0, knots=self.BP7)
        g = m.gamma(np.array([2.5]))
        for k in range(m.n_free):
            r = g[2 * k][0]
            i = g[2 * k + 1][0]
            assert abs(r.imag) < 1e-15
            assert abs(i.real) < 1e-15
            assert abs(r.real - i.imag) < 1e-15

    def test_defaults_empty(self):
        """get_defaults returns {}."""
        m = build_particle("test", model="BSpline", mass=1.0, knots=self.BP5)
        assert m.get_defaults() == {}

    def test_transform_fixes_mass(self):
        """Transform fixes mass."""
        m = build_particle("test", model="BSpline", mass=0.5, knots=self.BP5)
        tr = m.make_mass_width_transform()
        assert tr.output_names == ["test_mass"]
        d = tr.apply_forward({"x": 1.0})
        assert d["test_mass"] == 0.5

    def test_complex_sum(self):
        """Σ g_j·gamma_j = Σ re_k·B_k + i·Σ im_k·B_k."""
        m = build_particle("test", model="BSpline", mass=1.0, knots=self.BP7)
        g = m.gamma(np.array([2.5]))
        rng = np.random.default_rng(42)
        re = rng.uniform(-1, 1, m.n_free)
        im = rng.uniform(-0.5, 0.5, m.n_free)
        g0 = np.column_stack([re, im]).ravel()
        total = sum(g0[j] * g[j][0] for j in range(2 * m.n_free))
        Bk = np.array([g[2 * k][0].real for k in range(m.n_free)])
        expected = re @ Bk + 1j * (im @ Bk)
        assert abs(total - expected) < 1e-12


# ═══════════════════════════════════════════════════════════════════
# 4. Construction edge cases
# ═══════════════════════════════════════════════════════════════════

class TestConstruction:
    """Model construction edge cases."""

    def test_no_knots_raises(self):
        with pytest.raises(ValueError, match="knots"):
            build_particle("test", model="BSpline", mass=1.0)

    def test_too_few_knots_raises(self):
        with pytest.raises(ValueError):
            build_particle("test", model="BSpline", mass=1.0,
                           knots=[0.0, 1.0, 2.0], order=3)

    def test_registered(self):
        assert "BSpline" in ALL_MODELS

    def test_order_2(self):
        """Order 2 (quadratic) with bin-centre peaks."""
        m = build_particle("test", model="BSpline", mass=1.0,
                           knots=[0.0, 1.0, 2.0, 3.0, 4.0], order=2)
        assert m.n_free == 2  # 5 - 2 - 1 = 2
