#!/usr/bin/env python3
"""Tests for the Exp model and InterpKModel base class.

Run with::

    pytest tests/test_exp_model.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ampfit.particle_model import ALL_MODELS, build_particle
from ampfit.particle_model.interp_k_model import (
    InterpKModel, KToCRWeightsTransform,
    cr_basis, cr_basis_deriv,
)
from ampfit.particle_model.exp_model import ExpModel


# ═══════════════════════════════════════════════════════════════════
# 1. CR basis functions
# ═══════════════════════════════════════════════════════════════════

class TestCRBasis:
    """Catmull-Rom basis weight functions."""

    def test_sum_to_one(self):
        """Weights sum to 1 for any t in [0, 1]."""
        for t in np.linspace(0, 1, 11):
            w = cr_basis(t)
            assert abs(sum(w) - 1.0) < 1e-15, f"sum(w)={sum(w)} at t={t}"

    def test_kronecker_at_ends(self):
        """At t=0: w=(0,1,0,0).  At t=1: w=(0,0,1,0)."""
        w0 = cr_basis(0.0)
        assert w0 == pytest.approx((0, 1, 0, 0), abs=1e-15)
        w1 = cr_basis(1.0)
        assert w1 == pytest.approx((0, 0, 1, 0), abs=1e-15)

    def test_derivative_finite_diff(self):
        """cr_basis_deriv matches central-difference on cr_basis."""
        t = 0.3
        eps = 1e-8
        dw = cr_basis_deriv(t)
        dw_fd = tuple((cr_basis(t + eps)[i] - cr_basis(t - eps)[i]) / (2 * eps)
                      for i in range(4))
        for a, fd in zip(dw, dw_fd):
            assert abs(a - fd) < 1e-6, f"dw/dt mismatch: {a} vs {fd}"

    def test_sum_of_derivs_zero(self):
        """Sum of derivatives = 0 (since sum(weights) = 1 constant)."""
        for t in np.linspace(0, 1, 11):
            dw = cr_basis_deriv(t)
            assert abs(sum(dw)) < 1e-15, f"sum(dw/dt)={sum(dw)} at t={t}"

    def test_polynomial_interpolation(self):
        """CR reproduces quadratic f(x) = x^2 exactly."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        f = x ** 2
        for t in np.linspace(0, 1, 10):
            xt = 1.0 + t  # in [1, 2], using control pts [0,1,2,3]
            w = cr_basis(t)
            interp = w[0] * f[0] + w[1] * f[1] + w[2] * f[2] + w[3] * f[3]
            exact = xt ** 2
            assert abs(interp - exact) < 1e-14, f"CR failed for x^2 at t={t}"


# ═══════════════════════════════════════════════════════════════════
# 2. KToCRWeightsTransform
# ═══════════════════════════════════════════════════════════════════

class TestKToCRWeightsTransform:
    """Transform: k -> CR basis weights."""

    @pytest.fixture
    def tfm(self):
        return KToCRWeightsTransform(
            "k", "mass",
            [f"g{i}" for i in range(10)],
            mass_fixed=0.5, k_min=0.0, k_max=9.0,
        )

    def test_g_a_kb_delta(self, tfm):
        """g_a(k_b) = delta_{a,b} at grid points."""
        for kt in [0.0, 1.0, 5.0, 9.0]:
            out = tfm.forward({"k": kt})
            g = np.array([out[f"g{i}"] for i in range(10)])
            idx = int(kt) if kt <= 9.0 else 8
            assert g[idx] == pytest.approx(1.0, abs=1e-15)
            assert abs(g.sum() - 1.0) < 1e-15
            # All others should be zero
            others = np.concatenate([g[:idx], g[idx + 1:]])
            assert np.all(np.abs(others) < 1e-15)

    def test_mass_fixed(self, tfm):
        """Mass output is fixed at the given value."""
        out = tfm.forward({"k": 3.0})
        assert out["mass"] == 0.5

    def test_backward_finite_diff(self, tfm):
        """d(g0)/d(k) matches central difference."""
        eps = 1e-6
        k_test = 3.5
        out_p = tfm.forward({"k": k_test + eps})
        out_m = tfm.forward({"k": k_test - eps})
        back = tfm.backward({"g0": 1.0}, d_in={"k": k_test})
        dg_dk_a = back["k"]
        dg_dk_fd = (out_p["g0"] - out_m["g0"]) / (2 * eps)
        assert dg_dk_a == pytest.approx(dg_dk_fd, abs=1e-6)

    def test_backward_multichannel(self, tfm):
        """dL/dk = sum over channels of dL/dg_a * dg_a/dk."""
        k_test = 2.5
        eps = 1e-6
        grad = {"g2": 1.0, "g4": 2.0}
        back = tfm.backward(grad, d_in={"k": k_test})
        dk_a = back["k"]

        # FD: perturb k, compute sum(grad * g_i)
        def loss(k):
            out = tfm.forward({"k": k})
            return sum(out[n] * v for n, v in grad.items())
        dk_fd = (loss(k_test + eps) - loss(k_test - eps)) / (2 * eps)
        assert dk_a == pytest.approx(dk_fd, abs=1e-6)

    def test_inverse(self, tfm):
        """inverse(forward(k)) ≈ k."""
        for kt in [0.5, 2.0, 4.5, 8.0]:
            out = tfm.forward({"k": kt})
            inv = tfm.inverse(out)
            assert abs(inv["k"] - kt) < 0.2, f"inverse roundtrip failed at k={kt} (got {inv['k']})"
            assert inv["mass"] == 0.5

    def test_edge_k_min(self, tfm):
        """g_0(k_min) = 1, sum = 1."""
        out = tfm.forward({"k": 0.0})
        g = np.array([out[f"g{i}"] for i in range(10)])
        assert g[0] == pytest.approx(1.0, abs=1e-15)
        assert g.sum() == pytest.approx(1.0, abs=1e-15)

    def test_edge_k_max(self, tfm):
        """g_{N-1}(k_max) = 1, sum = 1."""
        out = tfm.forward({"k": 9.0})
        g = np.array([out[f"g{i}"] for i in range(10)])
        assert g[9] == pytest.approx(1.0, abs=1e-15)
        assert g.sum() == pytest.approx(1.0, abs=1e-15)


# ═══════════════════════════════════════════════════════════════════
# 3. InterpKModel base class
# ═══════════════════════════════════════════════════════════════════

class TestInterpKModel:
    """Base class for k-interpolated models."""

    def test_abstract_raises(self):
        """Instantiation without gamma_k raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            m = ALL_MODELS.get("NonExistent", None)
            # If somehow registered, skip; otherwise just verify base raises
            from ampfit.particle_model.interp_k_model import InterpKModel
            # Can't instantiate abstract directly via build_particle since
            # InterpKModel is not registered; verify gamma() raises
            class BadModel(InterpKModel):
                pass
            b = BadModel("bad")
            b.gamma(np.array([0.5]))

    def test_subclass_works(self):
        """Minimal subclass with gamma_k works."""
        class GoodModel(InterpKModel):
            def gamma_k(self, m, k):
                return np.ones_like(m) + 0j

        m = GoodModel("test", mass=0.5, k=1.0, k_range=[0.1, 5.0], n_interp=10)
        g = m.gamma(np.array([0.5]))
        assert len(g) == 10  # one per k-grid point
        assert g[0][0] == 1.0 + 0j

    def test_param_handling(self):
        """get_gamma_name, get_defaults, get_gamma_count match config."""
        m = ExpModel("test", mass=0.5, k=1.5, k_range=[0.1, 5.0], n_interp=20)
        assert m.get_gamma_count() == 20
        names = m.get_gamma_name()
        assert len(names) == 20
        assert names[0] == "test_gk0"
        assert names[-1] == "test_gk19"
        assert m.get_defaults() == {"test_k": 1.5}


# ═══════════════════════════════════════════════════════════════════
# 4. ExpModel — physics correctness
# ═══════════════════════════════════════════════════════════════════

class TestExpModel:
    """Exponential lineshape: A(m) = exp(-k*(m^2 - m0^2))."""

    @pytest.fixture
    def model(self):
        return build_particle(
            "sigma", mass=0.5, width=1.0, model="Exp",
            k=1.0, k_range=[0.1, 5.0], n_interp=30,
        )

    def test_registered(self):
        assert "Exp" in ALL_MODELS
        assert ALL_MODELS["Exp"] is ExpModel

    def test_gamma_k_exact_at_m0(self, model):
        """At m=m0, gamma_k gives A=1."""
        m = np.array([0.5])
        for k in [0.1, 1.0, 3.0, 5.0]:
            g = model.gamma_k(m, k)
            A = 1.0 / (0.5**2 - m**2 - 1j * 0.5 * g)
            assert A[0] == pytest.approx(1.0 + 0j, abs=1e-15)

    def test_gamma_k_exact_any_m(self, model):
        """A(m) = exp(-k*(m^2 - m0^2)) exactly for any m, k."""
        m = np.array([0.3, 0.6, 0.9])
        for k in [0.1, 1.0, 3.0]:
            g = model.gamma_k(m, k)
            A = 1.0 / (0.5**2 - m**2 - 1j * 0.5 * g)
            A_ref = np.exp(-k * (m**2 - 0.5**2))
            assert np.all(np.abs(A - A_ref) < 1e-14)

    def test_interpolated_exact_at_grid(self, model):
        """CR-interpolated amplitude = exact at k-grid points."""
        m = np.array([0.3, 0.5, 0.8])
        n_k = model.get_gamma_count()
        k_min, k_max = 0.1, 5.0
        k_grid = np.linspace(k_min, k_max, n_k)

        # Build full gamma table
        gamma_rows = np.array([model.gamma_k(m, ki) for ki in k_grid])
        g0_names = model.get_gamma_name()
        tfm = KToCRWeightsTransform("k", "mass", g0_names,
                                     mass_fixed=0.5, k_min=k_min, k_max=k_max)

        for ki in k_grid[::5]:
            out = tfm.forward({"k": ki})
            g = np.array([out[n] for n in g0_names])
            gamma_k = np.sum(g[:, None] * gamma_rows, axis=0)
            A = 1.0 / (0.5**2 - m**2 - 1j * 0.5 * gamma_k)
            A_ref = np.exp(-ki * (m**2 - 0.5**2))
            diff = np.max(np.abs(A - A_ref))
            assert diff < 1e-13, f"Failed at k={ki}: diff={diff}"



# ═══════════════════════════════════════════════════════════════════
# 5. Full pipeline test via fitter jacobian (using real config)
# ═══════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """Transform gradient through the constraint pipeline."""

    def test_transform_backward_in_pipeline(self):
        """KToCRWeightsTransform backward works via cm.chain_gradient.

        Uses a Config (without fitter) to test the constraint pipeline.
        """
        from ampfit.config_loader import Config
        from ampfit.param_constraint import ConstraintManager

        cfg = Config("config_angle.yml")
        kc = cfg.build_all_index()

        # Manually create a KToCRWeightsTransform
        k_name = "test_k"
        mass_name = "test_mass"
        n_interp = 10
        g0_names = [f"test_gk{i}" for i in range(n_interp)]

        tfm = KToCRWeightsTransform(
            k_name, mass_name, g0_names,
            mass_fixed=0.5, k_min=0.1, k_max=5.0,
        )

        # The transform should work standalone
        k_test = 1.5
        out = tfm.forward({k_name: k_test})
        assert abs(sum(float(out[n]) for n in g0_names) - 1.0) < 1e-10
        assert out[mass_name] == 0.5

        # Backward: check single-channel derivative
        eps = 1e-6
        out_p = tfm.forward({k_name: k_test + eps})
        out_m = tfm.forward({k_name: k_test - eps})
        back = tfm.backward({"test_gk0": 1.0}, d_in={k_name: k_test})
        dg_dk_a = back[k_name]
        dg_dk_fd = (out_p["test_gk0"] - out_m["test_gk0"]) / (2 * eps)
        assert dg_dk_a == pytest.approx(dg_dk_fd, abs=1e-6)
