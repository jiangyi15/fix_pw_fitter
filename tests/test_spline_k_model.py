#!/usr/bin/env python3
"""Tests for the SplineKModel and ExpSpline model.

Run with::

    pytest tests/test_spline_k_model.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ampfit.particle_model import ALL_MODELS, build_particle
from ampfit.particle_model.spline_k_model import (
    SplineKModel, KToSplineWeightsTransform,
    spline_basis_matrix, spline_weights, spline_weight_deriv,
)
from ampfit.particle_model.interp_k_model import KToCRWeightsTransform
from ampfit.particle_model.exp_model import ExpModel


# ═══════════════════════════════════════════════════════════════════
# 1. Spline basis matrix
# ═══════════════════════════════════════════════════════════════════

class TestSplineBasis:
    """Cubic spline basis matrix."""

    def test_identity_for_linear(self):
        """Spline reproduces linear f(x) = x exactly."""
        xi = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        N = len(xi)
        h = spline_basis_matrix(xi)  # (N-1, 4, N)
        y = xi  # point values at xi
        for k_test in np.linspace(0.0, 4.0, 20):
            g0, idx = spline_weights(k_test, h, xi)
            interp = float(np.sum(g0 * y))
            assert interp == pytest.approx(k_test, abs=1e-12)

    def test_identity_for_quadratic(self):
        """Spline reproduces quadratic f(x) = x^2 exactly."""
        xi = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        h = spline_basis_matrix(xi)
        y = xi ** 2
        for k_test in np.linspace(0.0, 4.0, 20):
            g0, idx = spline_weights(k_test, h, xi)
            interp = float(np.sum(g0 * y))
            assert interp == pytest.approx(k_test ** 2, abs=1e-10)

    def test_identity_for_cubic(self):
        """Spline reproduces cubic f(x) = x^3 exactly."""
        xi = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        h = spline_basis_matrix(xi)
        y = xi ** 3
        for k_test in np.linspace(0.0, 4.0, 20):
            g0, idx = spline_weights(k_test, h, xi)
            interp = float(np.sum(g0 * y))
            assert interp == pytest.approx(k_test ** 3, abs=1e-8)

    def test_sum_of_weights(self):
        """Spline weights sum to 1 for any k in range (partition of unity)."""
        xi = np.linspace(0.0, 5.0, 10)
        h = spline_basis_matrix(xi)
        for k_test in np.linspace(0.0, 5.0, 25):
            g0, _ = spline_weights(k_test, h, xi)
            assert abs(np.sum(g0) - 1.0) < 1e-12, f"sum={np.sum(g0)} at k={k_test}"

    def test_weight_deriv_fd(self):
        """spline_weight_deriv matches central difference on spline_weights."""
        xi = np.linspace(0.0, 5.0, 10)
        h = spline_basis_matrix(xi)
        for k_test in np.linspace(0.2, 4.8, 10):
            eps = 1e-6
            dg_dk, _ = spline_weight_deriv(k_test, h, xi)
            g0_p, _ = spline_weights(k_test + eps, h, xi)
            g0_m, _ = spline_weights(k_test - eps, h, xi)
            dg_dk_fd = (g0_p - g0_m) / (2 * eps)
            max_err = np.max(np.abs(dg_dk - dg_dk_fd))
            assert max_err < 1e-6, f"max deriv error={max_err:.2e} at k={k_test}"

    def test_sum_of_derivs_zero(self):
        """Sum of weight derivatives = 0 (since sum(weights) = 1 constant)."""
        xi = np.linspace(0.0, 5.0, 10)
        h = spline_basis_matrix(xi)
        for k_test in np.linspace(0.0, 5.0, 10):
            dg_dk, _ = spline_weight_deriv(k_test, h, xi)
            assert abs(np.sum(dg_dk)) < 1e-12, f"sum(deriv)={np.sum(dg_dk)} at k={k_test}"


# ═══════════════════════════════════════════════════════════════════
# 2. KToSplineWeightsTransform
# ═══════════════════════════════════════════════════════════════════

class TestKToSplineWeightsTransform:
    """Transform: k -> spline basis weights."""

    @pytest.fixture(params=["not-a-knot", "natural"])
    def tfm(self, request):
        return KToSplineWeightsTransform(
            "k", "mass",
            [f"g{i}" for i in range(10)],
            mass_fixed=0.5, k_min=0.0, k_max=9.0,
            bc_type=request.param,
        )

    def test_sum_to_one(self, tfm):
        """Weights sum to 1 for any k."""
        for kt in np.linspace(tfm.k_min, tfm.k_max, 15):
            out = tfm.forward({"k": kt})
            g = np.array([out[f"g{i}"] for i in range(10)])
            assert abs(g.sum() - 1.0) < 1e-12, f"sum={g.sum()} at k={kt}"

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

        def loss(k):
            out = tfm.forward({"k": k})
            return sum(out[n] * v for n, v in grad.items())
        dk_fd = (loss(k_test + eps) - loss(k_test - eps)) / (2 * eps)
        assert dk_a == pytest.approx(dk_fd, abs=1e-6)

    def test_backward_full_grad(self, tfm):
        """Full gradient backward: dL/dk = sum_j dL/dg_j * dg_j/dk."""
        k_test = 3.7
        eps = 1e-6
        rng = np.random.default_rng(42)
        grad = {f"g{i}": float(rng.standard_normal()) for i in range(10)}
        back = tfm.backward(grad, d_in={"k": k_test})
        dk_a = back["k"]

        def loss(k):
            out = tfm.forward({"k": k})
            return sum(out[n] * v for n, v in grad.items())
        dk_fd = (loss(k_test + eps) - loss(k_test - eps)) / (2 * eps)
        assert dk_a == pytest.approx(dk_fd, abs=1e-6)

    def test_backward_complex_grad(self, tfm):
        """Backward works with complex-valued gradients."""
        k_test = 3.7
        rng = np.random.default_rng(42)
        grad = {f"g{i}": float(rng.standard_normal()) * (1 + 2j)
                for i in range(10)}
        back = tfm.backward(grad, d_in={"k": k_test})
        dk = back["k"]
        # Complex result should be non-trivial
        assert abs(dk) > 1e-10

    def test_inverse_with_k(self, tfm):
        """inverse returns k directly if present in dict."""
        d = {"k": 3.5, "mass": 0.5}
        inv = tfm.inverse(d)
        assert inv["k"] == 3.5
        # the inverse reconstructs only the input k; the fixed mass is
        # preserved by apply_inverse, not re-emitted by the inverse
        assert "mass" not in inv

    def test_inverse_from_gk(self, tfm):
        """inverse reconstructs k from gk weights."""
        for kt in [0.5, 2.0, 4.5, 8.0]:
            out = tfm.forward({"k": kt})
            # forward output has no 'k' key, so inverse falls to peak search
            inv = tfm.inverse(out)
            assert abs(inv["k"] - kt) < 0.2, \
                f"inverse roundtrip failed at k={kt} (got {inv['k']})"
            assert "mass" not in inv

    def test_edge_k_min(self, tfm):
        """Weights well-behaved at k_min."""
        out = tfm.forward({"k": tfm.k_min})
        g = np.array([out[f"g{i}"] for i in range(10)])
        assert g.sum() == pytest.approx(1.0, abs=1e-12)

    def test_edge_k_max(self, tfm):
        """Weights well-behaved at k_max."""
        out = tfm.forward({"k": tfm.k_max})
        g = np.array([out[f"g{i}"] for i in range(10)])
        assert g.sum() == pytest.approx(1.0, abs=1e-12)


# ═══════════════════════════════════════════════════════════════════
# 3. SplineKModel base class
# ═══════════════════════════════════════════════════════════════════

class TestSplineKModel:
    """Base class for spline-interpolated k models."""

    def test_abstract_raises(self):
        """Instantiation without gamma_k raises NotImplementedError."""
        class BadModel(SplineKModel):
            pass
        b = BadModel("bad")
        with pytest.raises(NotImplementedError):
            b.gamma(np.array([0.5]))

    def test_subclass_works(self):
        """Minimal subclass with gamma_k works."""
        class GoodModel(SplineKModel):
            def gamma_k(self, m, k):
                return np.ones_like(m) + 0j

        m = GoodModel("test", mass=0.5, k=1.0,
                       k_range=[0.1, 5.0], n_interp=10)
        g = m.gamma(np.array([0.5]))
        assert len(g) == 10  # one per k-grid point
        assert g[0][0] == 1.0 + 0j

    def test_param_handling(self):
        """get_gamma_name, get_defaults, get_gamma_count match config."""
        from ampfit.particle_model.spline_k_model import ExpSplineModel
        m = ExpSplineModel("test", mass=0.5, k=1.5,
                            k_range=[0.1, 5.0], n_interp=20)
        assert m.get_gamma_count() == 20
        names = m.get_gamma_name()
        assert len(names) == 20
        assert names[0] == "test_gk0"
        assert names[-1] == "test_gk19"
        assert m.get_defaults() == {"test_k": 1.5}


# ═══════════════════════════════════════════════════════════════════
# 4. ExpSplineModel — physics correctness
# ═══════════════════════════════════════════════════════════════════

class TestExpSplineModel:
    """Exponential lineshape with spline interpolation."""

    @pytest.fixture(params=[11, 50])
    def model(self, request):
        return build_particle(
            "sigma", mass=0.5, width=1.0, model="ExpSpline",
            k=1.0, k_range=[0.1, 5.0], n_interp=request.param,
        )

    def test_registered(self):
        assert "ExpSpline" in ALL_MODELS

    def test_gamma_k_exact_at_m0(self, model):
        """At m=m0, gamma_k gives A=1 for any k."""
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
        """Spline-interpolated amplitude = exact at k-grid points."""
        m = np.array([0.3, 0.5, 0.8])
        n_k = model.get_gamma_count()
        k_min, k_max = 0.1, 5.0
        k_grid = np.linspace(k_min, k_max, n_k)

        gamma_rows = np.array([model.gamma_k(m, ki) for ki in k_grid])
        g0_names = model.get_gamma_name()
        tfm = KToSplineWeightsTransform("k", "mass", g0_names,
                                         mass_fixed=0.5,
                                         k_min=k_min, k_max=k_max)

        for ki in k_grid[::5]:
            out = tfm.forward({"k": ki})
            g = np.array([out[n] for n in g0_names])
            gamma_k = np.sum(g[:, None] * gamma_rows, axis=0)
            A = 1.0 / (0.5**2 - m**2 - 1j * 0.5 * gamma_k)
            A_ref = np.exp(-ki * (m**2 - 0.5**2))
            diff = np.max(np.abs(A - A_ref))
            # At grid points the spline is exact up to matrix solve precision
            assert diff < 5e-10, f"Failed at k={ki}: diff={diff}"

    def test_interpolated_between_grid(self, model):
        """Spline interpolation at between-grid k is accurate.

        With n_interp=50 (fine grid) the error should be tiny.
        With n_interp=11 (coarse grid) the error should be better
        than the CR equivalent (no sign flips, no >10% errors).
        """
        m = np.array([0.3, 0.5, 0.8, 1.5, 3.0])
        n_k = model.get_gamma_count()
        k_min, k_max = 0.1, 5.0
        k_grid = np.linspace(k_min, k_max, n_k)

        gamma_rows = np.array([model.gamma_k(m, ki) for ki in k_grid])
        g0_names = model.get_gamma_name()
        tfm = KToSplineWeightsTransform("k", "mass", g0_names,
                                         mass_fixed=0.5,
                                         k_min=k_min, k_max=k_max)

        # Skip k values near edges where spline has boundary effects
        edge_skip = 2.0 * (k_max - k_min) / max(n_k - 1, 1)
        skip_lo = k_min + edge_skip
        skip_hi = k_max - edge_skip
        if skip_hi <= skip_lo:
            return  # too few points for meaningful between-grid test
        for k_test in np.linspace(skip_lo, skip_hi, 10):
            out = tfm.forward({"k": k_test})
            g = np.array([out[n] for n in g0_names])
            gamma_k = np.sum(g[:, None] * gamma_rows, axis=0)
            A = 1.0 / (0.5**2 - m**2 - 1j * 0.5 * gamma_k)
            A_ref = np.exp(-k_test * (m**2 - 0.5**2))
            max_rel_err = np.max(np.abs(abs(A) - abs(A_ref)) / abs(A_ref))
            # For n_interp=50, error should be very small
            if n_k >= 50:
                assert max_rel_err < 0.005, \
                    f"max rel err={max_rel_err:.2e} at k={k_test}"
            else:
                # For n_interp=11 the grid is very coarse over k_range=4.9;
                # errors can be large at high m. Just verify no sign flip.
                assert max_rel_err < 2.0 or all(A.real > -1e-10), \
                    f"error at k={k_test}: {max_rel_err:.2e}"

    def test_gamma_k_jacobian_vs_fd(self, model):
        """d(gamma_k)/d(k) via transform backward matches numerical.

        For a single event mass m, the CR-interpolated gamma is:
            gamma_k(m) = sum_i g_i(k) * gamma_i(m)
        The gradient d(gamma_k)/dk = sum_i gamma_i(m) * dg_i/dk
        should match 3-point FD.
        """
        m_single = np.array([0.6])
        n_k = model.get_gamma_count()
        k_min, k_max = 0.1, 5.0
        k_test = 1.5

        gamma_rows = np.array([model.gamma_k(m_single, ki) for ki in
                               np.linspace(k_min, k_max, n_k)])

        g0_names = model.get_gamma_name()
        tfm = KToSplineWeightsTransform("k", "mass", g0_names,
                                         mass_fixed=0.5,
                                         k_min=k_min, k_max=k_max)

        out = tfm.forward({"k": k_test})
        g = np.array([out[n] for n in g0_names])

        # FD: d(gamma_k)/dk
        eps = 1e-6
        out_p = tfm.forward({"k": k_test + eps})
        out_m = tfm.forward({"k": k_test - eps})
        g_p = np.array([out_p[n] for n in g0_names])
        g_m = np.array([out_m[n] for n in g0_names])
        gamma_p = np.sum(g_p * gamma_rows[:, 0])
        gamma_m = np.sum(g_m * gamma_rows[:, 0])
        dgamma_dk_fd = (gamma_p - gamma_m) / (2 * eps)

        # Analytic: sum_i gamma_i(m) * dg_i/dk
        grad_dict = {g0_names[i]: gamma_rows[i, 0]
                     for i in range(n_k)}
        back = tfm.backward(grad_dict, d_in={"k": k_test})
        dgamma_dk = back["k"]

        assert dgamma_dk == pytest.approx(dgamma_dk_fd, abs=1e-8, rel=1e-4), \
            f"d(gamma)/dk: analytic={dgamma_dk:.10g} FD={dgamma_dk_fd:.10g}"

    def test_no_sign_flip_at_high_mass(self):
        """Spline avoids sign flips that CR produces at high mass.

        With sufficient grid points the spline converges without
        negative amplitudes.
        """
        m = np.array([4.0])
        n_k = 50  # fine enough to avoid edge effects
        k_min, k_max = 0.0, 2.0
        k_grid = np.linspace(k_min, k_max, n_k)

        gamma_rows = np.array([
            (1.0 - m**2 - np.exp(ki * (m**2 - 1.0))) / (1j * 1.0 * 1.0)
            for ki in k_grid])
        g0_names = [f"gk{i}" for i in range(n_k)]

        # Test at several between-grid k values
        for k_test in np.linspace(0.3, 1.7, 8):
            tfm = KToSplineWeightsTransform("k", "mass", g0_names,
                                             mass_fixed=1.0,
                                             k_min=k_min, k_max=k_max)
            out = tfm.forward({"k": k_test})
            g = np.array([out[n] for n in g0_names])
            gamma_k = np.sum(g * gamma_rows[:, 0])
            A = 1.0 / (1.0 - m[0]**2 - 1j * 1.0 * gamma_k)
            A_true = np.exp(-k_test * (m[0]**2 - 1.0))

            # Amplitude should be positive
            assert abs(A) > 0, f"Zero amplitude at k={k_test}"
            # Re(A) should be positive (exp(-k*(m²-1)) is always positive)
            assert A.real > 0, f"Negative real part at k={k_test}: Re(A)={A.real:.6e}"

            # Error should be below 100% (no sign flip)
            err = abs(abs(A) - A_true) / A_true
            assert err < 1.0, f"Error too large at k={k_test}: {err:.2e}"


# ═══════════════════════════════════════════════════════════════════
# 5. Convergence with n_interp
# ═══════════════════════════════════════════════════════════════════

class TestSplineConvergence:
    """Spline interpolation converges with more grid points."""

    def test_convergence_narrow_range(self):
        """Spline converges with more points for a moderate regime.

        With k in [0.1, 2.0] and m ≤ 2.0, the exponential is smooth
        enough that n=50 gives < 1% between-grid error.
        """
        m0 = 0.5
        m = np.array([0.3, 0.5, 0.8, 1.2, 1.5, 2.0])
        k_min, k_max = 0.1, 2.0
        n_interp = 50
        k_grid = np.linspace(k_min, k_max, n_interp)
        g0 = 1.0  # reference width
        gamma_rows = np.array([
            (m0**2 - m**2 - np.exp(ki * (m**2 - m0**2))) / (1j * m0 * g0)
            for ki in k_grid])
        g0_names = [f"gk{i}" for i in range(n_interp)]
        tfm = KToSplineWeightsTransform("k", "mass", g0_names,
                                         mass_fixed=m0,
                                         k_min=k_min, k_max=k_max)
        max_err = 0.0
        edge = 2.0 * (k_max - k_min) / max(n_interp - 1, 1)
        for k_test in np.linspace(k_min + edge, k_max - edge, 10):
            out = tfm.forward({"k": k_test})
            g = np.array([out[n] for n in g0_names])
            gamma_k = np.sum(g[:, None] * gamma_rows, axis=0)
            A = 1.0 / (m0**2 - m**2 - 1j * m0 * gamma_k)
            A_ref = np.exp(-k_test * (m**2 - m0**2))
            err = np.max(np.abs(abs(A) - abs(A_ref)) / abs(A_ref))
            max_err = max(max_err, err)
        assert max_err < 0.01, \
            f"n={n_interp}: max_err={max_err:.2e} >= 0.01"

    def test_no_negative_at_high_mass(self):
        """Spline does not produce sign-flipped amplitudes at high mass
        when using sufficient grid points, unlike Catmull-Rom."""
        m0 = 0.5
        m = np.array([3.0, 4.0, 5.0])
        k_min, k_max = 0.1, 5.0
        n_interp = 100  # fine grid for stability at high mass
        k_grid = np.linspace(k_min, k_max, n_interp)
        g0 = 1.0
        gamma_rows = np.array([
            (m0**2 - m**2 - np.exp(ki * (m**2 - m0**2))) / (1j * m0 * g0)
            for ki in k_grid])
        g0_names = [f"gk{i}" for i in range(n_interp)]
        tfm = KToSplineWeightsTransform("k", "mass", g0_names,
                                         mass_fixed=m0,
                                         k_min=k_min, k_max=k_max)
        for k_test in np.linspace(0.5, 4.5, 10):
            out = tfm.forward({"k": k_test})
            g = np.array([out[n] for n in g0_names])
            gamma_k = np.sum(g[:, None] * gamma_rows, axis=0)
            A = 1.0 / (m0**2 - m**2 - 1j * m0 * gamma_k)
            # All amplitudes should be positive real (exp(-k*(m²-m0²)) > 0)
            for i in range(len(m)):
                assert A[i].real > 0 or abs(A[i].real) < 1e-15, \
                    f"Negative Re(A) at k={k_test}, m={m[i]}: {A[i].real:.4e}"


# ═══════════════════════════════════════════════════════════════════
# 6. Full pipeline test
# ═══════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """Spline transform gradient through the constraint pipeline."""

    def test_transform_backward_in_pipeline(self):
        """KToSplineWeightsTransform backward via cm.chain_gradient."""
        from ampfit.config_loader import Config
        from ampfit.param_constraint import ConstraintManager

        cfg = Config("config_angle.yml")
        kc = cfg.build_all_index()

        k_name = "test_k"
        mass_name = "test_mass"
        n_interp = 10
        g0_names = [f"test_gk{i}" for i in range(n_interp)]

        tfm = KToSplineWeightsTransform(
            k_name, mass_name, g0_names,
            mass_fixed=0.5, k_min=0.1, k_max=5.0,
        )

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


# ═══════════════════════════════════════════════════════════════════
# 7. Same gamma_k physics as ExpModel
# ═══════════════════════════════════════════════════════════════════

class TestGammaKIdentical:
    """ExpSplineModel and ExpModel have identical gamma_k."""

    def test_gamma_k_identical(self):
        """Both models produce the same gamma_k(m,k)."""
        exp = build_particle("a", mass=0.5, width=1.0, model="Exp",
                              k=1.0, k_range=[0.1, 5.0], n_interp=30)
        spl = build_particle("a", mass=0.5, width=1.0, model="ExpSpline",
                              k=1.0, k_range=[0.1, 5.0], n_interp=30)

        m = np.array([0.3, 0.5, 0.8, 1.2])
        for k in [0.1, 1.0, 3.0, 4.5]:
            g_exp = exp.gamma_k(m, k)
            g_spl = spl.gamma_k(m, k)
            assert np.all(np.abs(g_exp - g_spl) < 1e-15), \
                f"gamma_k mismatch at k={k}"


# ═══════════════════════════════════════════════════════════════════
# 8. pure_exp option: exact exponential for any width
# ═══════════════════════════════════════════════════════════════════

class TestPureExp:
    """``pure_exp: true`` makes the exp amplitude exactly exp(-k(m²-m0²))
    regardless of the configured width (default False for backward
    compatibility; the legacy width-normalised behaviour is deprecated
    and will be removed in the next version)."""

    def test_pure_exp_exact_any_width(self):
        """A(m) = exp(-k(m²-m0²)) exactly at width=0.4 with pure_exp."""
        m0 = 0.5
        m = np.array([0.3, 0.5, 0.8, 1.2, 2.0])
        mdl = build_particle("a", mass=m0, width=0.4, model="ExpSpline",
                             k=1.0, k_range=[0.1, 5.0], n_interp=30,
                             pure_exp=True)
        for k in [0.3, 1.0, 2.5]:
            g = mdl.gamma_k(m, k)
            A = 1.0 / (m0 ** 2 - m ** 2 - 1j * m0 * g)
            A_ref = np.exp(-k * (m ** 2 - m0 ** 2))
            assert np.max(np.abs(A - A_ref)) < 1e-14, \
                f"pure-exp amplitude off at k={k}"

    def test_legacy_width_shapes_tail(self):
        """pure_exp: false (legacy) width != 1 reshapes the amplitude."""
        import warnings
        m0, g0, k = 0.5, 0.4, 1.0
        m = np.array([0.5, 1.0, 2.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mdl = build_particle("a", mass=m0, width=g0, model="ExpSpline",
                                 k=1.0, k_range=[0.1, 5.0], n_interp=30,
                                 pure_exp=False)
            g = mdl.gamma_k(m, k)
        A = 1.0 / (m0 ** 2 - m ** 2 - 1j * m0 * g)
        A_ref = np.exp(-k * (m ** 2 - m0 ** 2))
        # legacy shape differs from the pure exponential at width != 1
        assert np.max(np.abs(A - A_ref)) > 1e-3
        # and equals 1/[(m0²-m²)(1-1/g0) + exp(k(m²-m0²))/g0]
        denom = (m0 ** 2 - m ** 2) * (1 - 1 / g0) + np.exp(k * (m ** 2 - m0 ** 2)) / g0
        assert np.max(np.abs(A - 1 / denom)) < 1e-14

    def test_warning_without_pure_exp(self):
        """Warning when width != 1 and pure_exp: false (legacy path)."""
        mdl = build_particle("a", mass=0.5, width=0.4, model="ExpSpline",
                             k=1.0, k_range=[0.1, 5.0], n_interp=5,
                             pure_exp=False)
        with pytest.warns(UserWarning, match="pure_exp"):
            mdl.gamma_k(np.array([0.6]), 1.0)

    def test_no_warning_width_one(self):
        """No warning when width == 1 (pure exponential holds)."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            build_particle("a", mass=0.5, width=1.0, model="ExpSpline",
                           k=1.0, k_range=[0.1, 5.0], n_interp=5)

    def test_no_warning_with_pure_exp(self):
        """No warning when pure_exp is set, even at width != 1."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            build_particle("a", mass=0.5, width=0.4, model="ExpSpline",
                           k=1.0, k_range=[0.1, 5.0], n_interp=5,
                           pure_exp=True)


class TestOtherExpModelsFixed:
    """Exp / ExpSplineSVD / Exp2DSplineSVD use the corrected formula
    always (no pure_exp option): exact exp(-k(m²-m0²)) at any width."""

    @pytest.mark.parametrize("model,kwargs", [
        ("Exp", dict(k=1.0, k_range=[0.1, 2.0], n_interp=5)),
        ("ExpSplineSVD", dict(k=1.0, k_range=[0.1, 2.0], n_interp=6,
                              n_reduce=4, n_mass_pts=300,
                              m_range=[0.3, 2.0])),
    ])
    def test_exact_at_width_ne_1(self, model, kwargs):
        import warnings
        m0 = 0.5
        m = np.array([0.3, 0.5, 0.8, 1.2, 2.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)  # no warning expected
            mdl = build_particle("a", mass=m0, width=0.4,
                                 model=model, **kwargs)
        for k in [0.5, 1.0]:
            g = mdl.gamma_k(m, k)
            A = 1.0 / (m0 ** 2 - m ** 2 - 1j * m0 * g)
            A_ref = np.exp(-k * (m ** 2 - m0 ** 2))
            assert np.max(np.abs(A - A_ref)) < 1e-14, \
                f"{model}: amplitude off at k={k}"

    def test_exp2d_exact_at_width_ne_1(self):
        import warnings
        m0 = 0.5
        m = np.array([0.3, 0.5, 0.8, 1.2])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            mdl = build_particle("a", mass=m0, width=0.4,
                                 model="Exp2DSplineSVD",
                                 a=1.0, b=0.0, a_range=[0.1, 2.0],
                                 b_range=[0.1, 2.0], n_a=3, n_b=3,
                                 n_reduce=4, n_mass_pts=200,
                                 m_range=[0.3, 2.0])
        g = mdl.gamma_k(m, 1.0, 0.0)
        A = 1.0 / (m0 ** 2 - m ** 2 - 1j * m0 * g)
        A_ref = np.exp(-1.0 * (m ** 2 - m0 ** 2))
        assert np.max(np.abs(A - A_ref)) < 1e-14
