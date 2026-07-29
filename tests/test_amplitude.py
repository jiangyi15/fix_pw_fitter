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


class TestAmplitudeRaw:
    """amplitude_raw(m, params) — no transform applied."""

    def test_bw_at_pole(self):
        m = build_particle("r", model="BW", mass=0.775, width=0.149)
        A = m.amplitude_raw(np.array([0.775]), {"r_mass": 0.775, "r_width": 0.149})
        expected = 1j / (0.775 * 0.149)
        assert abs(A[0] - expected) < 1e-12

    def test_bw_off_pole(self):
        m = build_particle("r", model="BW", mass=0.775, width=0.149)
        A = m.amplitude_raw(np.array([0.5]), {"r_mass": 0.775, "r_width": 0.149})
        expected = 1.0 / (0.775**2 - 0.5**2 - 1j * 0.775 * 0.149)
        assert abs(A[0] - expected) < 1e-12

    def test_array_input(self):
        m = build_particle("r", model="BW", mass=1.0, width=0.2)
        x = np.linspace(0.5, 1.5, 10)
        A = m.amplitude_raw(x, {"r_mass": 1.0, "r_width": 0.2})
        assert len(A) == 10 and all(np.isfinite(A))


class TestAmplitude:
    """amplitude(m, params=None) — transform applied."""

    def test_gaussian_empty_params(self):
        m = build_particle("g0", model="GaussianBasis", mu=0.5, sigma=0.2)
        A = m.amplitude(np.array([0.5]))
        assert abs(A[0] - 1.0) < 1e-12

    def test_gaussian_shape(self):
        m = build_particle("g0", model="GaussianBasis", mu=0.5, sigma=0.2)
        x = np.array([0.3, 0.5, 0.7, 1.0])
        A = m.amplitude(x, {})
        expected = np.exp(-(x - 0.5)**2 / (2 * 0.2**2))
        assert np.allclose(A, expected)

    def test_overrides_params(self):
        m = build_particle("g0", model="GaussianBasis", mu=0.5, sigma=0.2)
        A = m.amplitude(np.array([0.5]), {"g0_mass": 2.0})
        assert abs(A[0] - 1.0) < 1e-6

    def test_input_not_modified(self):
        m = build_particle("g0", model="GaussianBasis", mu=0.5, sigma=0.2)
        p = {"other": 99.0}
        m.amplitude(np.array([0.5]), p)
        assert p == {"other": 99.0}

    def test_none_vs_empty(self):
        m = build_particle("g0", model="GaussianBasis", mu=0.5, sigma=0.2)
        A1 = m.amplitude(np.array([0.5]), None)
        A2 = m.amplitude(np.array([0.5]), {})
        assert abs(A1[0] - A2[0]) < 1e-12

    def test_bw_with_transform_none(self):
        """BW has no transform — same as raw."""
        m = build_particle("r", model="BW", mass=0.775, width=0.149)
        A = m.amplitude(np.array([0.775]), {"r_mass": 0.775, "r_width": 0.149})
        assert np.isfinite(A[0])
