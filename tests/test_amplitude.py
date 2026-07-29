#!/usr/bin/env python3
"""Tests for BaseModel.amplitude() method.

Run with::

    pytest tests/test_amplitude.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ampfit.particle_model import build_particle


class TestAmplitudeBase:
    """amplitude() on models without transforms (explicit params needed)."""

    def test_bw_at_pole(self):
        """BW at pole: A = i/(m0·width)."""
        m = build_particle("r", model="BW", mass=0.775, width=0.149)
        A = m.amplitude(np.array([0.775]), {"r_mass": 0.775, "r_width": 0.149})
        expected = 1j / (0.775 * 0.149)
        assert abs(A[0] - expected) < 1e-12

    def test_bw_off_pole(self):
        """BW off pole matches formula."""
        m = build_particle("r", model="BW", mass=0.775, width=0.149)
        p = {"r_mass": 0.775, "r_width": 0.149}
        A = m.amplitude(np.array([0.5]), p)
        expected = 1.0 / (0.775**2 - 0.5**2 - 1j * 0.775 * 0.149)
        assert abs(A[0] - expected) < 1e-12

    def test_array_input(self):
        """amplitude works with array of masses."""
        m = build_particle("r", model="BW", mass=1.0, width=0.2)
        x = np.linspace(0.5, 1.5, 10)
        A = m.amplitude(x, {"r_mass": 1.0, "r_width": 0.2})
        assert len(A) == 10
        assert all(np.isfinite(A))


class TestAmplitudeFixedShape:
    """amplitude() on models with transforms (empty params works)."""

    def test_gaussian_basis_empty_params(self):
        """GaussianBasis: amplitude({}) returns correct Gaussian peak."""
        m = build_particle("g0", model="GaussianBasis", mu=0.5, sigma=0.2)
        A = m.amplitude(np.array([0.5]))
        assert abs(A[0] - 1.0) < 1e-12

    def test_gaussian_basis_shape(self):
        """GaussianBasis: amplitude tracks exp(-(m-mu)^2/(2*sigma^2))."""
        m = build_particle("g0", model="GaussianBasis", mu=0.5, sigma=0.2)
        x = np.array([0.3, 0.5, 0.7, 1.0])
        A = m.amplitude(x, {})
        expected = np.exp(-(x - 0.5)**2 / (2 * 0.2**2))
        assert np.allclose(A, expected)

    def test_transform_does_not_modify_input(self):
        """Transform applied to copy, not original dict."""
        m = build_particle("g0", model="GaussianBasis", mu=0.5, sigma=0.2)
        params = {"other": 99.0}
        m.amplitude(np.array([0.5]), params)
        assert params == {"other": 99.0}  # unchanged


class TestAmplitudeEdgeCases:
    """Edge cases."""

    def test_transformed_model_with_explicit_params(self):
        """Non-empty params override transform's fixed values."""
        m = build_particle("g0", model="GaussianBasis", mu=0.5, sigma=0.2)
        A = m.amplitude(np.array([0.5]), {"g0_mass": 2.0, "g0_width": 1.0})
        # gamma conversion compensates, so shape is still Gaussian
        assert abs(A[0] - 1.0) < 1e-6

    def test_bw_with_params(self):
        """Params correctly passed through for BW (no transform)."""
        m = build_particle("r", model="BW", mass=0.775, width=0.149)
        A = m.amplitude(np.array([0.775]), {"r_mass": 0.775, "r_width": 0.149})
        assert np.isfinite(A[0])
