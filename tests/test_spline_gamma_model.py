#!/usr/bin/env python3
"""Tests for BSplineGammaModel.

Run with::

    pytest tests/test_spline_gamma_model.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ampfit.particle_model import build_particle, ALL_MODELS
from ampfit.particle_model.spline_gamma_model import (
    bspline_basis_all,
    BSplineGammaModel, _SplineMassFixTransform,
)


class TestBasis:
    def test_pu(self):
        bp = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        basis, n = bspline_basis_all(np.linspace(0, 5, 100), bp, order=3)
        assert np.max(np.abs(basis.sum(axis=1) - 1.0)) < 1e-12


class TestMassFix:
    def test_forward(self):
        tr = _SplineMassFixTransform("test_mass", mass_default=0.5)
        r = tr.apply_forward({"x": 1.0})
        assert r["test_mass"] == 0.5 and r["x"] == 1.0

    def test_backward(self):
        tr = _SplineMassFixTransform("test_mass", mass_default=0.5)
        tr.apply_forward({"x": 1.0})
        g = tr.apply_backward({"test_mass": 2.0, "x": 3.0})
        assert "test_mass" not in g and g["x"] == 3.0


class TestModel:
    def test_n_free_knots(self):
        m = build_particle("test", model="BSpline", mass=1.0,
                           knots=[0, 1, 2, 3, 4, 5, 6])
        assert m.n_free == 3  # 7 - 3 - 1

    def test_n_free_range(self):
        m = build_particle("test", model="BSpline", mass=1.0,
                           x_range=[0, 5], n_free=8)
        assert m.n_free == 8

    def test_gamma_count(self):
        m = build_particle("test", model="BSpline", mass=1.0,
                           knots=[0, 1, 2, 3, 4, 5, 6])
        assert m.get_gamma_count() == 2 * m.n_free

    def test_gamma_names(self):
        m = build_particle("test", model="BSpline", mass=1.0,
                           knots=[0, 1, 2, 3, 4, 5, 6])
        assert m.get_gamma_name()[0] == "test_re_B_0"
        assert len(m.get_gamma_name()) == 6

    def test_structure(self):
        m = build_particle("test", model="BSpline", mass=1.0,
                           knots=[0, 1, 2, 3, 4, 5, 6])
        g = m.gamma(np.array([2.5]))
        for k in range(m.n_free):
            r, i = g[2*k][0], g[2*k+1][0]
            assert abs(r.imag) < 1e-15 and abs(i.real) < 1e-15

    def test_defaults(self):
        m = build_particle("test", model="BSpline", mass=1.0,
                           knots=[0, 1, 2, 3, 4, 5, 6])
        assert m.get_defaults() == {}

    def test_interior_active(self):
        """Interior functions are all active in the user's range."""
        for mode in ['knots', 'range']:
            if mode == 'knots':
                m = build_particle("test", model="BSpline", mass=1.0,
                                   knots=[0, 1, 2, 3, 4, 5, 6])
                x = np.linspace(0, 6, 100)
            else:
                m = build_particle("test", model="BSpline", mass=1.0,
                                   x_range=[0, 6], n_free=3, order=3)
                x = np.linspace(0, 6, 100)
            g = m.gamma(x)
            for k in range(m.n_free):
                assert np.any(np.abs(g[2*k].real) > 0.01), f"{mode}: B_{k} inactive"

    def test_complex_sum(self):
        m = build_particle("test", model="BSpline", mass=1.0,
                           knots=[0, 1, 2, 3, 4, 5, 6])
        g = m.gamma(np.array([2.5]))
        rng = np.random.default_rng(42)
        re = rng.uniform(-1, 1, m.n_free)
        im = rng.uniform(-0.5, 0.5, m.n_free)
        Bk = np.array([g[2*k][0].real for k in range(m.n_free)])
        expected = sum((re[k] + 1j*im[k]) * Bk[k] for k in range(m.n_free))
        g0 = np.column_stack([re, im]).ravel()
        direct = sum(g0[j] * g[j][0] for j in range(2*m.n_free))
        assert abs(direct - expected) < 1e-12


class TestConstruction:
    def test_no_spec(self):
        with pytest.raises(ValueError):
            build_particle("test", model="BSpline", mass=1.0)

    def test_range(self):
        m = build_particle("test", model="BSpline", mass=0.5,
                           x_range=[0, 5], n_free=8, order=3)
        assert m.n_free == 8
        assert len(m.breakpoints) == 12  # 8 + 3 + 1

    def test_registered(self):
        assert "BSpline" in ALL_MODELS

    def test_order_2(self):
        m = build_particle("test", model="BSpline", mass=1.0,
                           knots=[0, 1, 2, 3, 4], order=2)
        assert m.n_free == 2  # 5 - 2 - 1
