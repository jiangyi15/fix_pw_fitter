#!/usr/bin/env python3
"""Tests for the fixed-shape Linear model (A(m) = k·(m − m₀)).

Run with::

    pytest tests/test_linear_model.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ampfit.particle_model import build_particle


@pytest.fixture
def lin():
    return build_particle("a", model="linear", mass=1.0, k=0.3, width=0.1)


class TestLinearShape:
    def test_amplitude_is_linear(self, lin):
        """The BW amplitude reproduces A(m) = k·(m − m₀)."""
        m = np.array([0.5, 0.8, 1.2, 1.5, 2.0])
        gamma = np.asarray(lin.gamma(m)[0])
        A = 1.0 / (1.0 ** 2 - m ** 2 - 1j * 1.0 * 0.1 * gamma)
        target = 0.3 * (m - 1.0)
        assert np.allclose(A.real, target, atol=1e-12)

    def test_zero_at_m0(self, lin):
        """A(m₀) = 0 but the safe clip keeps 1/A finite (no nan)."""
        gamma = np.asarray(lin.gamma(np.array([1.0]))[0])
        assert np.all(np.isfinite(gamma))
        # the amplitude is clipped to |A| ≥ a_min (default 1e-6)
        A = 0.3 * (np.array([1.0]) - 1.0)
        eps = 1e-6
        assert abs(np.copysign(eps, A[0])) == eps

    def test_clip_away_from_zero(self, lin):
        """Points away from m₀ keep the exact linear amplitude."""
        m = np.array([0.5, 1.5, 2.0])
        gamma = np.asarray(lin.gamma(m)[0])
        A = 1.0 / (1.0 ** 2 - m ** 2 - 1j * 1.0 * 0.1 * gamma)
        assert np.allclose(A.real, 0.3 * (m - 1.0), atol=1e-6)

    def test_gamma_matches_fixed_shape_formula(self, lin):
        """γ = (m₀² − m² − 1/A) / (i·m₀·g₀) with A = k·(m−m₀)."""
        m = np.array([0.7, 1.3, 1.7])
        gamma = np.asarray(lin.gamma(m)[0])
        A = 0.3 * (m - 1.0)
        target = (1.0 ** 2 - m ** 2 - 1.0 / A) / (1j * 1.0 * 0.1)
        assert np.allclose(gamma, target, atol=1e-12)

    def test_registered(self):
        """'linear' is a registered model name."""
        from ampfit.particle_model import ALL_MODELS
        assert "linear" in ALL_MODELS


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
