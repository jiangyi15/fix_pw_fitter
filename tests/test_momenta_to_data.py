#!/usr/bin/env python3
"""Tests for the momentum → data npz converter (ρρ topology).

Run with::

    pytest tests/test_momenta_to_data.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ampfit.phasespace_b4pi import (generate_b4pi, two_body_momentum, M_PION,
                                    M_B_MESON)
from ampfit.momenta_to_data import (momenta_to_data, momenta_to_data_full,
                                    data_to_momentum, _boost, _boost3_op,
                                    _inv_mass_sq)


@pytest.fixture
def ev():
    return generate_b4pi(30000, seed=11)


@pytest.fixture
def data(ev):
    return momenta_to_data(ev["momenta"])


class TestTopologyRhoRho:
    def test_masses_match_generator(self, ev, data):
        """Row (block0, topo0) masses = the generator's m1, m2."""
        m = data["mass"][:, 0]
        assert np.all(np.abs(m[:, 0] - ev["m1"]) < 1e-8)
        assert np.all(np.abs(m[:, 1] - ev["m2"]) < 1e-8)

    def test_breakup_momenta(self, ev, data):
        """q values are the physical breakup momenta."""
        q = data["q"][:, 0]
        assert np.all(np.abs(q[:, 1] - two_body_momentum(ev["m1"], M_PION,
                                                         M_PION)) < 1e-8)
        assert np.all(np.abs(q[:, 2] - two_body_momentum(ev["m2"], M_PION,
                                                         M_PION)) < 1e-8)
        assert np.all(q[:, 0] >= 0)

    def test_phi_flat(self, data):
        """φ is flat in [−π, π] (the generator produced a flat φ)."""
        phi = data["angles"][:, 0, 0]
        assert phi.min() >= -np.pi - 1e-9 and phi.max() <= np.pi + 1e-9
        assert abs(phi.mean()) < 0.05
        assert abs(phi.std() - np.pi / np.sqrt(3)) < 0.05

    def test_cos_theta_flat(self, data):
        """cos θ₁, cos θ₂ are flat (the generator produced flat cosθ)."""
        for k in (1, 2):
            ct = np.cos(data["angles"][:, 0, k])
            assert abs(ct.mean()) < 0.05
            assert abs(ct.std() - 1 / np.sqrt(3)) < 0.05

    def test_blocks_populated(self, data):
        """All 8 blocks have a non-zero topo-0 row."""
        mass_t0 = data["mass"][:, ::3]      # topology-0 rows of each block
        assert np.all(np.any(mass_t0 != 0, axis=2))


class TestTopologyChain:
    """B → R₁(R₂(π⁺₁π⁻₁)π⁺₂)π⁻₂ and its mirror."""

    def test_masses(self, ev, data):
        """Chain masses = m(R₂) and m(R₁) (row 1)."""
        m = data["mass"][:, 1]
        # R₂ = (π⁺₁,π⁻₁) -> m1; R₁ = (π⁺₁,π⁻₁,π⁺₂) -> m123
        p = ev["momenta"]
        R2 = p[:, 0] + p[:, 1]
        R1 = R2 + p[:, 2]
        m_R2 = np.sqrt(np.maximum(R2[:, 0] ** 2 - np.sum(R2[:, 1:] ** 2, 1), 0))
        m_R1 = np.sqrt(np.maximum(R1[:, 0] ** 2 - np.sum(R1[:, 1:] ** 2, 1), 0))
        assert np.all(np.abs(m[:, 0] - m_R1) < 1e-8)
        assert np.all(np.abs(m[:, 1] - m_R2) < 1e-8)

    def test_chain_mirror_masses(self, ev, data):
        """Mirror chain (row 2): R₂ = (π⁺₁,π⁻₁), R₁ = (π⁺₁,π⁻₁,π⁻₂)."""
        m = data["mass"][:, 2]
        p = ev["momenta"]
        R2 = p[:, 0] + p[:, 1]
        R1 = R2 + p[:, 3]
        m_R2 = np.sqrt(np.maximum(R2[:, 0] ** 2 - np.sum(R2[:, 1:] ** 2, 1), 0))
        m_R1 = np.sqrt(np.maximum(R1[:, 0] ** 2 - np.sum(R1[:, 1:] ** 2, 1), 0))
        assert np.all(np.abs(m[:, 0] - m_R1) < 1e-8)
        assert np.all(np.abs(m[:, 1] - m_R2) < 1e-8)

    def test_phi_flat(self, data):
        """Chain φ is flat (phase-space azimuth)."""
        for t in (1, 2):
            phi = data["angles"][:, t, 0]
            assert abs(phi.mean()) < 0.05
            assert abs(phi.std() - np.pi / np.sqrt(3)) < 0.05

    def test_all_topologies_populated(self, data):
        """All 3 topologies × 8 blocks are filled."""
        assert np.all(np.any(data["mass"] != 0, axis=2))


class TestPermutations:
    def test_identical_perms_consistent(self, ev, data):
        """Swap-both permutation exchanges the two di-pion masses."""
        m0 = data["mass"][:, 0]      # block 0 (identity): (m12, m34)
        m3 = data["mass"][:, 9]      # block 3 (swap both π⁺,π⁻): (m34, m12)
        assert np.all(np.abs(m3[:, 0] - ev["m2"]) < 1e-8)
        assert np.all(np.abs(m3[:, 1] - ev["m1"]) < 1e-8)

    def test_cp_block_same_masses(self, ev, data):
        """CP block masses equal the non-CP ones (same event, same pairs)."""
        m0 = data["mass"][:, 0]
        m4 = data["mass"][:, 12]
        assert np.all(np.abs(m4 - m0) < 1e-8)


class TestHelpers:
    def test_boost_rest_frame(self):
        """Boosting a particle by its own velocity puts it at rest."""
        p = np.array([5.0, 1.0, 2.0, -0.5])
        v = p[1:] / p[0]
        pr = _boost(p, _boost3_op(v))
        assert abs(pr[1:]).max() < 1e-12
        assert pr[0] == pytest.approx(np.sqrt(_inv_mass_sq(p)), rel=1e-9)


class TestReference:
    """Compare against the reference data arrays (skipped if absent)."""

    REF_MOM = "/media/jiangy/JZAO/ana/time_dep_amp/test_4pi/create_data17/data_sig.npy"
    REF_NPZ = "/home/jiangy/github/project71/data/data_arrays.npz"

    @pytest.fixture
    def refs(self):
        if not (os.path.exists(self.REF_MOM) and os.path.exists(self.REF_NPZ)):
            pytest.skip("reference data files not available")
        mom = np.load(self.REF_MOM)
        ref = np.load(self.REF_NPZ)
        assert mom.shape[0] == ref["mass"].shape[0]
        return mom, ref

    def test_default_matches_full(self):
        """The default (CP-transform) momenta_to_data reproduces the
        reference full-computation within the same tolerances."""
        if not (os.path.exists(self.REF_MOM) and os.path.exists(self.REF_NPZ)):
            pytest.skip("reference data files not available")
        mom = np.load(self.REF_MOM)[:20000]
        a = momenta_to_data(mom)
        b = momenta_to_data_full(mom)
        assert np.max(np.abs(a["mass"] - b["mass"])) < 1e-8
        assert np.max(np.abs(a["q"] - b["q"])) < 1e-8
        d0 = np.abs(((a["angles"][:, :, 0] - b["angles"][:, :, 0] + np.pi)
                     % (2 * np.pi)) - np.pi)
        assert d0.max() < 1e-6
        assert np.max(np.abs(a["angles"][:, :, 1] - b["angles"][:, :, 1])) < 1e-6
        assert np.max(np.abs(a["angles"][:, :, 2] - b["angles"][:, :, 2])) < 1e-6

    def test_mass_q_match(self, refs):
        mom, ref = refs
        out = momenta_to_data(mom[:20000])
        assert np.max(np.abs(out["mass"] - ref["mass"][:20000])) < 1e-8
        assert np.max(np.abs(out["q"] - ref["q"][:20000])) < 1e-8

    def test_angles_match(self, refs):
        mom, ref = refs
        out = momenta_to_data(mom[:20000])
        oa, ra = out["angles"], ref["angles"][:20000]
        # wrapped azimuth difference (clean form)
        d0 = np.abs(((oa[:, :, 0] - ra[:, :, 0] + np.pi) % (2 * np.pi)) - np.pi)
        assert d0.max() < 1e-6
        assert np.max(np.abs(oa[:, :, 1] - ra[:, :, 1])) < 1e-6
        assert np.max(np.abs(oa[:, :, 2] - ra[:, :, 2])) < 1e-6


class TestDataToMomentum:
    """Reverse conversion: data → momenta reproduces the data exactly."""

    @pytest.fixture
    def roundtrip(self, ev):
        mom2 = data_to_momentum(momenta_to_data(ev["momenta"]))
        return momenta_to_data(mom2)

    def test_roundtrip_exact(self, data, roundtrip):
        """All 24 rows of mass/q/angles are reproduced."""
        assert np.abs(roundtrip["mass"] - data["mass"]).max() < 1e-9
        assert np.abs(roundtrip["q"] - data["q"]).max() < 1e-9
        d = np.abs(((roundtrip["angles"] - data["angles"] + np.pi)
                    % (2 * np.pi)) - np.pi)
        assert d.max() < 1e-9

    def test_reconstructed_momenta_physical(self, ev):
        """B at rest, pions on-shell, generator masses reproduced."""
        mom2 = data_to_momentum(momenta_to_data(ev["momenta"]))
        tot = mom2.sum(1)
        assert np.abs(tot[:, 0] - M_B_MESON).max() < 1e-9
        assert np.abs(tot[:, 1:]).max() < 1e-9
        m2s = mom2[:, :, 0] ** 2 - (mom2[:, :, 1:] ** 2).sum(2)
        assert np.abs(m2s - M_PION ** 2).max() < 1e-9
        m12 = np.sqrt((mom2[:, 0, 0] + mom2[:, 1, 0]) ** 2
                      - ((mom2[:, 0, 1:] + mom2[:, 1, 1:]) ** 2).sum(1))
        assert np.abs(m12 - ev["m1"]).max() < 1e-9

    def test_fixed_m3pi_roundtrip(self):
        """The fixed-m(πππ) generator also round-trips."""
        from ampfit.phasespace_b4pi import generate_b4pi_fixed_m3pi
        mom = generate_b4pi_fixed_m3pi(1.3, 20000, seed=3)["momenta"]
        data = momenta_to_data(mom)
        mom2 = data_to_momentum(data)
        data2 = momenta_to_data(mom2)
        assert np.abs(data2["mass"] - data["mass"]).max() < 1e-9
        assert np.abs(data2["q"] - data["q"]).max() < 1e-9


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
