#!/usr/bin/env python3
"""Tests for the B → 4π phase-space generator.

Run with::

    pytest tests/test_phasespace_b4pi.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ampfit.phasespace_b4pi import (
    generate_b4pi, two_body_momentum, sample_masses,
    _mass_weight, M_PION, M_B_MESON,
)


def _masses(ev):
    mom = ev["momenta"]
    E, px, py, pz = mom[:, :, 0], mom[:, :, 1], mom[:, :, 2], mom[:, :, 3]
    p4 = np.stack([E, px, py, pz], axis=-1)          # (n, 4, 4)
    n = len(mom)
    m2 = np.zeros((n, 4, 4))
    for a in range(4):
        for b in range(4):
            m2[:, a, b] = ((p4[:, a, 0] + p4[:, b, 0]) ** 2
                           - np.sum((p4[:, a, 1:] + p4[:, b, 1:]) ** 2, axis=1))
    return p4, np.sqrt(np.maximum(m2, 0))


class TestKinematics:
    @pytest.fixture
    def ev(self):
        return generate_b4pi(20000, seed=42)

    def test_mass_shell(self, ev):
        """Each pion is on the mass shell: p² = m_π²."""
        mom = ev["momenta"]
        m2s = (mom[:, :, 0] ** 2 - np.sum(mom[:, :, 1:] ** 2, axis=2))
        assert np.all(np.abs(m2s - M_PION ** 2) < 1e-10)

    def test_momentum_conservation(self, ev):
        """Σ p_i = (m_B, 0, 0, 0) in the B rest frame."""
        mom = ev["momenta"]
        tot = mom.sum(1)
        assert np.all(np.abs(tot[:, 0] - M_B_MESON) < 1e-10)
        assert np.all(np.abs(tot[:, 1:]) < 1e-10)

    def test_dipion_masses(self, ev):
        """m(π₁π₂) = m1 and m(π₃π₄) = m2 (the sampled variables)."""
        p4, m2 = _masses(ev)   # m2 = pair invariant masses
        assert np.all(np.abs(m2[:, 0, 1] - ev["m1"]) < 1e-8)
        assert np.all(np.abs(m2[:, 2, 3] - ev["m2"]) < 1e-8)

    def test_cross_pairs_physical(self, ev):
        """All pair masses above 2m_π and below m_B."""
        p4, m2 = _masses(ev)
        for a in range(4):
            for b in range(a + 1, 4):
                m = m2[:, a, b]
                assert np.all(m > (2 * M_PION) ** 2 - 1e-6)
                assert np.all(m < M_B_MESON ** 2 + 1e-6)


class TestSampling:
    def test_masses_symmetric(self):
        """The weight is symmetric in (m1, m2); samples must be too."""
        ev = generate_b4pi(200000, seed=7)
        assert abs(ev["m1"].mean() - ev["m2"].mean()) < 0.01

    def test_m1_matches_weight_marginal(self):
        """Sampled m1 marginal matches ∫ w(m1,m2) dm2."""
        ev = generate_b4pi(300000, seed=3)
        m1 = ev["m1"]
        mg = np.linspace(2 * M_PION, M_B_MESON - 2 * M_PION, 60)
        h, _ = np.histogram(m1, bins=mg, density=True)
        c = (mg[1:] + mg[:-1]) / 2

        def marginal(x):
            m2g = np.linspace(2 * M_PION, M_B_MESON - x, 200)
            w = two_body_momentum(M_B_MESON, x, m2g) \
                * two_body_momentum(x, M_PION, M_PION) \
                * two_body_momentum(m2g, M_PION, M_PION)
            return np.trapezoid(w, m2g)

        ref = np.array([marginal(x) for x in c])
        ref /= ref.sum() * (mg[1] - mg[0])
        # ignore the low-statistic low-m1 bins
        mask = ref > 0.02 * ref.max()
        rel = np.abs(h[mask] - ref[mask]) / ref[mask]
        assert np.median(rel) < 0.05, f"median rel err {np.median(rel)}"

    def test_flat_angles(self):
        """cos θ₁, cos θ₂ and φ are flat."""
        ev = generate_b4pi(100000, seed=9)
        for ct in (ev["cos_theta1"], ev["cos_theta2"]):
            assert abs(ct.mean()) < 0.02
            assert abs(ct.std() - 1 / np.sqrt(3)) < 0.02
        assert abs(ev["phi"].mean() - np.pi) < 0.05

    def test_seed_reproducible(self):
        a = generate_b4pi(1000, seed=123)
        b = generate_b4pi(1000, seed=123)
        assert np.array_equal(a["momenta"], b["momenta"])


class TestHelpers:
    def test_two_body_momentum_threshold(self):
        q = two_body_momentum(1.0, 0.4, 0.4)
        assert q > 0
        # exactly at threshold q = 0
        assert two_body_momentum(1.0, 0.5, 0.5) == pytest.approx(0.0, abs=1e-15)

    def test_ps_units(self):
        """Breakup momentum is physical (0 < q < M/2)."""
        q = two_body_momentum(5.279, 2.0, 2.0)
        assert 0 < q < 2.5


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
