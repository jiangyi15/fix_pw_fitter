#!/usr/bin/env python3
"""Tests for the 2-parameter SVD-reduced spline model (Exp2DSplineSVD).

Run with::

    pytest tests/test_svd_2d_spline_k_model.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ampfit.particle_model import ALL_MODELS, build_particle
from ampfit.particle_model.svd_2d_spline_k_model import (
    KToSVDWeights2DTransform, SVDSplineKModel2D,
)


@pytest.fixture
def model():
    """12×12 grid, r=18: the recommended 2D reduction (~3e-4 off-grid)."""
    return build_particle(
        "sigma", mass=0.5, width=1.0, model="Exp2DSplineSVD",
        a=1.0, b=0.5, a_range=[0.1, 2.0], b_range=[0.1, 2.0],
        n_a=12, n_b=12, n_reduce=18, n_mass_pts=2000, m_range=[0.28, 2.0],
    )


def _true_amp(m, a, b):
    return np.exp(-(a + 1j * b) * (m ** 2 - 0.5 ** 2))


def _model_amp(model, m, a, b):
    tfm = model.make_mass_width_transform()
    out = tfm.forward({f"{model.name}_a": a, f"{model.name}_b": b})
    w = np.array([out[n] for n in model.get_gamma_name()])
    g = w @ np.array(model.gamma(np.asarray(m, dtype=float)))
    return 1.0 / (0.5 ** 2 - m ** 2 - 1j * 0.5 * g)


# ═══════════════════════════════════════════════════════════════════
# 1. Model structure
# ═══════════════════════════════════════════════════════════════════

class TestModelStructure:
    def test_registered(self):
        assert "Exp2DSplineSVD" in ALL_MODELS

    def test_param_handling(self, model):
        assert model.get_gamma_count() == model.n_reduce
        names = model.get_gamma_name()
        assert len(names) == model.n_reduce
        assert names[0] == "sigma_sr0"
        assert model.get_defaults() == {"sigma_a": 1.0, "sigma_b": 0.5}

    def test_transform_inputs(self, model):
        tfm = model.make_mass_width_transform()
        assert set(tfm.input_names) == {"sigma_a", "sigma_b"}
        assert "sigma_mass" in tfm.output_names

    def test_abstract_raises(self):
        with pytest.raises(NotImplementedError):
            SVDSplineKModel2D("bad", n_a=3, n_b=3, n_reduce=2,
                              n_mass_pts=50)

    def test_baseline_weight_one(self, model):
        """sr0 = 1 for any (a,b) — 2D partition of unity."""
        tfm = model.make_mass_width_transform()
        for a, b in [(0.3, 0.3), (1.0, 0.5), (1.8, 1.9)]:
            out = tfm.forward({"sigma_a": a, "sigma_b": b})
            assert out["sigma_sr0"] == pytest.approx(1.0, abs=1e-10)


# ═══════════════════════════════════════════════════════════════════
# 2. Transform Jacobian and inverse
# ═══════════════════════════════════════════════════════════════════

class TestTransform2D:
    def test_backward_fd_a_and_b(self, model):
        """dL/da, dL/db match central differences of forward."""
        tfm = model.make_mass_width_transform()
        rng = np.random.default_rng(5)
        grad = {n: float(rng.standard_normal())
                for n in model.get_gamma_name()}
        a0, b0 = 1.1, 0.8
        eps = 1e-5

        def loss(a, b):
            out = tfm.forward({"sigma_a": a, "sigma_b": b})
            return sum(out[n] * v for n, v in grad.items())

        back = tfm.backward(grad, d_in={"sigma_a": a0, "sigma_b": b0})
        da_fd = (loss(a0 + eps, b0) - loss(a0 - eps, b0)) / (2 * eps)
        db_fd = (loss(a0, b0 + eps) - loss(a0, b0 - eps)) / (2 * eps)
        assert back["sigma_a"] == pytest.approx(da_fd, rel=1e-4, abs=1e-4)
        assert back["sigma_b"] == pytest.approx(db_fd, rel=1e-4, abs=1e-4)

    def test_backward_baseline_zero(self, model):
        """Baseline (sr0) contributes nothing to da/db."""
        tfm = model.make_mass_width_transform()
        back = tfm.backward({"sigma_sr0": 1.0},
                            d_in={"sigma_a": 1.0, "sigma_b": 0.5})
        assert abs(back["sigma_a"]) < 1e-9
        assert abs(back["sigma_b"]) < 1e-9

    def test_inverse_roundtrip(self, model):
        """inverse recovers (a, b) from forward outputs."""
        tfm = model.make_mass_width_transform()
        for a, b in [(0.5, 0.4), (1.0, 0.5), (1.7, 1.6)]:
            out = tfm.forward({"sigma_a": a, "sigma_b": b})
            inv = tfm.inverse(out)
            assert abs(inv["sigma_a"] - a) < 0.05, \
                f"a roundtrip failed at a={a} (got {inv['sigma_a']})"
            assert abs(inv["sigma_b"] - b) < 0.05, \
                f"b roundtrip failed at b={b} (got {inv['sigma_b']})"

    def test_inverse_with_params(self, model):
        tfm = model.make_mass_width_transform()
        d = {"sigma_a": 1.2, "sigma_b": 0.7, "sigma_mass": 0.5}
        inv = tfm.inverse(d)
        assert inv["sigma_a"] == 1.2
        assert inv["sigma_b"] == 0.7


# ═══════════════════════════════════════════════════════════════════
# 3. Physics: reconstruction accuracy
# ═══════════════════════════════════════════════════════════════════

class TestReconstruction:
    def test_amplitude_accuracy(self, model):
        """Reduced 2D model reproduces exp(-(a+bi)(m²-m0²)) off-grid."""
        m = np.linspace(0.32, 1.9, 400)
        worst = 0.0
        for a, b in [(0.4, 0.3), (1.0, 0.5), (1.6, 1.4), (1.9, 1.9)]:
            A = _model_amp(model, m, a, b)
            A_true = _true_amp(m, a, b)
            rel = np.max(np.abs(A - A_true)) / np.max(np.abs(A_true))
            worst = max(worst, rel)
        assert worst < 1e-3, f"2D complex-amplitude rel err={worst:.2e}"

    def test_full_rank_exact_at_grid_points(self):
        """r = n_a·n_b + 1 reconstructs grid-point rows exactly."""
        m = build_particle("sigma", mass=0.5, model="Exp2DSplineSVD",
                           a=1.0, b=0.5, a_range=[0.1, 2.0],
                           b_range=[0.1, 2.0], n_a=4, n_b=4,
                           n_reduce=17, n_mass_pts=500, m_range=[0.28, 2.0])
        mtest = np.linspace(0.3, 1.9, 300)
        worst = 0.0
        for a in m._a_grid:
            for b in m._b_grid:
                A = _model_amp(m, mtest, a, b)
                A_true = _true_amp(mtest, a, b)
                worst = max(worst, np.max(np.abs(A - A_true))
                            / np.max(np.abs(A_true)))
        assert worst < 1e-4, f"full-rank grid-point err={worst:.2e}"

    def test_k_gradient_full_model_fd(self, model):
        """dA/da, dA/db through the full chain match finite differences."""
        m_test = np.array([0.6, 1.2])
        a0, b0 = 1.0, 0.5
        eps = 1e-5

        Ap = _model_amp(model, m_test, a0 + eps, b0)
        Am = _model_amp(model, m_test, a0 - eps, b0)
        dA_da_fd = (Ap - Am) / (2 * eps)
        Ap = _model_amp(model, m_test, a0, b0 + eps)
        Am = _model_amp(model, m_test, a0, b0 - eps)
        dA_db_fd = (Ap - Am) / (2 * eps)

        # analytic: dA/d(.) = dA/dgamma · sum_r B_r(m) · dw'_r/d(.)
        tfm = model.make_mass_width_transform()
        dA_dgamma, dB_dgamma = _dA_dgamma(model, m_test, a0, b0)
        dA, dB = tfm._weight_deriv(a0, b0)
        g = np.array(model.gamma(m_test))
        dA_da = dA_dgamma * (dA @ g)
        dA_db = dA_dgamma * (dB @ g)
        # note: gamma is purely imaginary; amplitude complex via a+bi
        assert np.max(np.abs(dA_da - dA_da_fd)) / np.max(
            np.abs(dA_da_fd)) < 1e-3
        assert np.max(np.abs(dA_db - dA_db_fd)) / np.max(
            np.abs(dA_db_fd)) < 1e-3


def _dA_dgamma(model, m, a, b):
    """dA/dgamma and dA/dgammabar via the denominator formula.

    A = 1/(m0²-m²-i·m0·gamma).  The model's gamma is a real-combination
    of complex basis rows, so dA/dgamma_i = A²·i·m0 (chain for the
    scalar gamma value), where gamma here means the total i·m0 factor is
    included.  We return dA/dgamma_scale with gamma_scale = gamma.
    """
    tfm = model.make_mass_width_transform()
    out = tfm.forward({f"{model.name}_a": a, f"{model.name}_b": b})
    w = np.array([out[n] for n in model.get_gamma_name()])
    g = np.array(model.gamma(np.asarray(m, dtype=float)))
    gamma = w @ g
    A = 1.0 / (0.5 ** 2 - m ** 2 - 1j * 0.5 * gamma)
    return (A ** 2) * (1j * 0.5), None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
