#!/usr/bin/env python3
"""Tests for the momentum → data npz converter (ρρ topology).

Run with::

    pytest tests/test_momenta_to_data.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from tabpwa.phasespace_b4pi import (generate_b4pi, two_body_momentum, M_PION,
                                    M_B_MESON)
from tabpwa.momenta_to_data import (momenta_to_data, momenta_to_data_full,
                                    momenta_to_data_samesign,
                                    data_to_momentum, data_to_momentum_samesign,
                                    _boost, _boost3_op, _inv_mass_sq)


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
    """momenta_to_data (fast/CP path) vs momenta_to_data_full (reference
    computation) on seeded generated events — fully self-contained."""

    @pytest.fixture
    def mom(self, ev):
        return ev["momenta"][:20000]

    def test_default_matches_full(self, mom):
        """The default (CP-transform) momenta_to_data reproduces the
        reference full-computation within the same tolerances."""
        a = momenta_to_data(mom)
        b = momenta_to_data_full(mom)
        assert np.max(np.abs(a["mass"] - b["mass"])) < 1e-8
        assert np.max(np.abs(a["q"] - b["q"])) < 1e-8
        d0 = np.abs(((a["angles"][:, :, 0] - b["angles"][:, :, 0] + np.pi)
                     % (2 * np.pi)) - np.pi)
        assert d0.max() < 1e-6
        assert np.max(np.abs(a["angles"][:, :, 1] - b["angles"][:, :, 1])) < 1e-6
        assert np.max(np.abs(a["angles"][:, :, 2] - b["angles"][:, :, 2])) < 1e-6

    def test_mass_q_match(self, mom):
        out, ref = momenta_to_data(mom), momenta_to_data_full(mom)
        assert np.max(np.abs(out["mass"] - ref["mass"])) < 1e-8
        assert np.max(np.abs(out["q"] - ref["q"])) < 1e-8

    def test_angles_match(self, mom):
        out, ref = momenta_to_data(mom), momenta_to_data_full(mom)
        oa, ra = out["angles"], ref["angles"]
        # wrapped azimuth difference (clean form)
        d0 = np.abs(((oa[:, :, 0] - ra[:, :, 0] + np.pi) % (2 * np.pi)) - np.pi)
        assert d0.max() < 1e-6
        assert np.max(np.abs(oa[:, :, 1] - ra[:, :, 1])) < 1e-6
        assert np.max(np.abs(oa[:, :, 2] - ra[:, :, 2])) < 1e-6


def test_flavour_tag_event_data_routes_by_structure():
    """4π structure → the legacy 24-row fill (bit-equal to
    momenta_to_data); a no-identical-particles structure (K*K̄*-like:
    CP blocks only) → the generic tree fill with frac/time."""
    import copy

    import yaml
    from tabpwa.amp_model import build_amplitude_model
    from tabpwa.phasespace_b4pi import generate_b4pi

    ev = generate_b4pi(300, seed=11)
    mom = ev["momenta"]

    m = build_amplitude_model("tests/config_amp.yml")
    d = m.build_event_data(mom, m.build_kernel_config())
    ref = momenta_to_data(mom)
    assert m._legacy_td_structure() is True
    assert np.array_equal(d["mass"].reshape(300, 24, 2), ref["mass"])
    assert np.array_equal(d["q"].reshape(300, 24, 3), ref["q"])
    assert np.array_equal(d["angle"], ref["angles"])
    assert np.array_equal(d["frac"], ref["frac"])
    assert np.array_equal(d["time"], ref["time"])

    # K*K̄*-like: drop the identical-pair declaration (CP blocks only)
    cfg = yaml.safe_load(open("tests/config_amp.yml"))
    cfg["data"].pop("identical_particles", None)
    m2 = build_amplitude_model(cfg)
    assert m2._legacy_td_structure() is False
    kc2 = m2.build_kernel_config()
    d2 = m2.build_event_data(mom, kc2)
    # the data-row count the kernel derives from the config: the fill
    # and the config agree by construction (same tree enumeration)
    n_rows = int(np.max(kc2["angle_index"])) + 1
    assert d2["angle"].shape == (300, n_rows, 3)
    assert d2["mass"].shape == (300, 2 * n_rows)   # flat: n_rows × n_res
    assert np.allclose(d2["frac"], 0.5) and np.allclose(d2["time"], 0.0)

    # frac/time overrides + chunk equivalence (NaN-aware)
    d3 = m2.build_event_data(mom, kc2, frac=np.linspace(0, 1, 300),
                             time=np.ones(300))
    assert np.allclose(d3["frac"], np.linspace(0, 1, 300))
    assert np.allclose(d3["time"], 1.0)
    d4 = m2.build_event_data(mom, kc2, chunk=100)
    for k in d2:
        a, b = d4[k], d2[k]
        ok = np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" \
            else np.array_equal(a, b)
        assert ok, k

    # fit-level smoke: the CP-only structure runs through the numpy TD
    # backend (finite NLL)
    from tabpwa import Fitter

    f = Fitter(cfg, backend="numpy")
    f.apply_constrains()
    f.set_phsp(d2); f.set_data(d2)
    x0 = f.initial_values(seed=5)
    nll, grad = f.get_nll(x0)
    assert np.isfinite(nll) and np.all(np.isfinite(grad))
    del copy


class TestSameSignTopology:
    """B → (π⁺₁π⁺₂)(π⁻₁π⁻₂): same-charge pairing, cosθ ∈ [0, 1]."""

    @pytest.fixture
    def samesign(self, ev):
        return momenta_to_data_samesign(ev["momenta"])

    def test_pair_masses(self, ev, samesign):
        """m_pp/m_mm are the same-charge pair masses."""
        mom = ev["momenta"]
        m_pp = np.sqrt((mom[:, 0, 0] + mom[:, 2, 0]) ** 2
                       - ((mom[:, 0, 1:] + mom[:, 2, 1:]) ** 2).sum(1))
        m_mm = np.sqrt((mom[:, 1, 0] + mom[:, 3, 0]) ** 2
                       - ((mom[:, 1, 1:] + mom[:, 3, 1:]) ** 2).sum(1))
        assert np.abs(m_pp - samesign["m_pp"]).max() < 1e-9
        assert np.abs(m_mm - samesign["m_mm"]).max() < 1e-9

    def test_cos_theta_range(self, samesign):
        """The helicity cosines are restricted to [0, 1]."""
        assert samesign["theta_p"].min() >= -1e-12
        assert samesign["theta_p"].max() <= 1.0 + 1e-12
        assert samesign["theta_m"].min() >= -1e-12
        assert samesign["theta_m"].max() <= 1.0 + 1e-12

    def test_identical_particle_invariance(self, ev, samesign):
        """Swapping the two π⁺ (or the two π⁻) leaves the variables
        unchanged (identical-particle ambiguity fixed by cosθ ∈ [0,1]
        and the φ relabelling)."""
        mom = ev["momenta"]
        for i, j in [(0, 2), (1, 3), ((0, 2), (1, 3))]:
            s = mom.copy()
            if isinstance(i, tuple):
                s[:, i[0]], s[:, i[1]] = mom[:, i[1]].copy(), mom[:, i[0]].copy()
                s[:, j[0]], s[:, j[1]] = mom[:, j[1]].copy(), mom[:, j[0]].copy()
            else:
                s[:, i], s[:, j] = mom[:, j].copy(), mom[:, i].copy()
            d = momenta_to_data_samesign(s)
            assert np.abs(d["m_pp"] - samesign["m_pp"]).max() < 1e-9
            assert np.abs(d["m_mm"] - samesign["m_mm"]).max() < 1e-9
            assert np.abs(d["theta_p"] - samesign["theta_p"]).max() < 1e-9
            assert np.abs(d["theta_m"] - samesign["theta_m"]).max() < 1e-9
            dphi = np.abs(((d["phi"] - samesign["phi"] + np.pi)
                           % (2 * np.pi)) - np.pi)
            assert dphi.max() < 1e-9

    def test_samesign_roundtrip(self, ev, samesign):
        """data_to_momentum_samesign reverses it exactly (only the
        output order differs from build_momenta)."""
        mom2 = data_to_momentum_samesign(
            samesign["m_pp"], samesign["m_mm"],
            samesign["theta_p"], samesign["theta_m"], samesign["phi"])
        d2 = momenta_to_data_samesign(mom2)
        for k in ("m_pp", "m_mm", "theta_p", "theta_m"):
            assert np.abs(d2[k] - samesign[k]).max() < 1e-9
        dphi = np.abs(((d2["phi"] - samesign["phi"] + np.pi)
                       % (2 * np.pi)) - np.pi)
        assert dphi.max() < 1e-9
        # B at rest, pions on-shell
        tot = mom2.sum(1)
        assert np.abs(tot[:, 0] - M_B_MESON).max() < 1e-9
        assert np.abs(tot[:, 1:]).max() < 1e-9
        m2s = mom2[:, :, 0] ** 2 - (mom2[:, :, 1:] ** 2).sum(2)
        assert np.abs(m2s - M_PION ** 2).max() < 1e-9


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
        from tabpwa.phasespace_b4pi import generate_b4pi_fixed_m3pi
        mom = generate_b4pi_fixed_m3pi(1.3, 20000, seed=3)["momenta"]
        data = momenta_to_data(mom)
        mom2 = data_to_momentum(data)
        data2 = momenta_to_data(mom2)
        assert np.abs(data2["mass"] - data["mass"]).max() < 1e-9
        assert np.abs(data2["q"] - data["q"]).max() < 1e-9


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
