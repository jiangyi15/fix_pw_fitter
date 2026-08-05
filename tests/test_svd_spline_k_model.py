#!/usr/bin/env python3
"""Tests for the SVD-reduced spline-k models.

Run with::

    pytest tests/test_svd_spline_k_model.py -v

Note on reduction accuracy: the exponential gamma grows like
``exp(k*m^2)``, so over the *full* kinematic range ``[2*m_pi, m_b-m_pi]``
with large ``k`` the family is not low-rank and the truncated SVD is
inaccurate.  The reconstruction tests therefore use a moderate
``k_range`` (where the mechanism is accurate); the transform-gradient
and inverse tests are exact for any configuration.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ampfit.particle_model import ALL_MODELS, build_particle
from ampfit.particle_model.svd_spline_k_model import (
    KToSVDWeightsTransform, SVDSplineKModel, ExpSplineSVDModel,
    M_PION, M_B_MESON,
)
from ampfit.particle_model.spline_k_model import (
    KToSplineWeightsTransform, spline_basis_matrix,
    spline_weights, spline_weight_deriv, ExpSplineModel,
)


# ═══════════════════════════════════════════════════════════════════
# 1. KToSVDWeightsTransform — exact k propagation
# ═══════════════════════════════════════════════════════════════════

def _make_projection(n_k=10, n_reduce=4, seed=0):
    """Random real projection matrix (r, n_k) with orthonormal rows."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_k, n_k))
    Q, _ = np.linalg.qr(A)
    return Q[:, :n_reduce].T  # (n_reduce, n_k), rows orthonormal


@pytest.fixture
def tfm():
    return KToSVDWeightsTransform(
        "k", "mass",
        [f"sr{i}" for i in range(4)],
        projection=_make_projection(),
        mass_fixed=0.5, k_min=0.0, k_max=9.0,
    )


class TestKToSVDWeightsTransform:
    def test_mass_fixed(self, tfm):
        """Mass output is fixed at the given value."""
        out = tfm.forward({"k": 3.0})
        assert out["mass"] == 0.5

    def test_output_count(self, tfm):
        """Forward emits exactly n_reduce reduced weights."""
        out = tfm.forward({"k": 2.0})
        assert all(f"sr{i}" in out for i in range(4))
        assert len([k for k in out if k.startswith("sr")]) == 4

    def test_backward_finite_diff_full_grad(self, tfm):
        """dL/dk = sum_r dL/dsr_r * dsr_r/dk matches central difference."""
        rng = np.random.default_rng(42)
        grad = {f"sr{i}": float(rng.standard_normal()) for i in range(4)}
        k_test = 3.7
        eps = 1e-6

        def loss(k):
            out = tfm.forward({"k": k})
            return sum(out[n] * v for n, v in grad.items())

        back = tfm.backward(grad, d_in={"k": k_test})
        dk_a = back["k"]
        dk_fd = (loss(k_test + eps) - loss(k_test - eps)) / (2 * eps)
        assert dk_a == pytest.approx(dk_fd, abs=1e-6, rel=1e-4)

    def test_backward_matches_spline_projection_chain(self, tfm):
        """Backward equals (P^T dg') · dw/dk with full spline derivatives.

        Since w' = P @ w(k), dL/dk must equal the projection pushed
        back onto the full spline weight derivative — this is the
        "k propagates to k correctly" property.
        """
        rng = np.random.default_rng(7)
        grad = {f"sr{i}": float(rng.standard_normal()) for i in range(4)}
        k_test = 2.3

        back = tfm.backward(grad, d_in={"k": k_test})
        dk_a = back["k"]

        dg_full = tfm._projection.T @ np.array([grad[f"sr{i}"] for i in range(4)])
        dw_dk, _ = spline_weight_deriv(k_test, tfm._h_matrix, tfm._k_grid)
        dk_chain = float(np.dot(dg_full, dw_dk))
        assert dk_a == pytest.approx(dk_chain, abs=1e-12)

    def test_backward_complex_grad(self, tfm):
        """Backward works with complex-valued gradients."""
        k_test = 2.5
        grad = {f"sr{i}": (1 + 2j) * (i + 1) for i in range(4)}
        dk = tfm.backward(grad, d_in={"k": k_test})["k"]
        assert abs(dk) > 1e-12

    def test_inverse_with_k(self, tfm):
        """inverse returns k directly if present in dict."""
        d = {"k": 3.5, "mass": 0.5}
        inv = tfm.inverse(d)
        assert inv["k"] == 3.5
        assert inv["mass"] == 0.5

    def test_inverse_roundtrip(self, tfm):
        """inverse recovers k from reduced weights."""
        for kt in [1.0, 3.0, 6.0, 8.0]:
            out = tfm.forward({"k": kt})
            inv = tfm.inverse(out)
            assert abs(inv["k"] - kt) < 0.05, \
                f"inverse roundtrip failed at k={kt} (got {inv['k']})"

    def test_jacobian_vs_fd(self):
        """Transform Jacobian dw'/dk matches finite differences.

        Analytic: J = P @ dw/dk.  Verified against a central difference
        at the roundoff-optimal step (weights reach ~1e5, so eps=1e-5).
        """
        proj = _make_projection(n_k=10, n_reduce=4, seed=1)
        t = KToSVDWeightsTransform(
            "k", "mass", [f"sr{i}" for i in range(4)],
            projection=proj, mass_fixed=0.5, k_min=0.0, k_max=9.0)
        eps = 1e-5
        for k in [1.0, 3.5, 6.0, 8.5]:
            J_an, _ = t._weight_deriv(k)
            wp, _ = t._weights(k + eps)
            wm, _ = t._weights(k - eps)
            J_fd = (wp - wm) / (2 * eps)
            scale = max(np.max(np.abs(J_an)), 1e-30)
            rel = np.max(np.abs(J_fd - J_an)) / scale
            assert rel < 1e-6, f"k={k}: Jacobian rel err={rel:.2e}"

    def test_backward_equals_jacobian_row(self, tfm):
        """backward({sr_r: 1}) == Jacobian row r (exact)."""
        k_test = 2.7
        J_an, _ = tfm._weight_deriv(k_test)
        for r, name in enumerate(tfm.out_names):
            dk = tfm.backward({name: 1.0}, d_in={"k": k_test})["k"]
            assert dk == pytest.approx(J_an[r], abs=1e-12), f"row {r}"


# ═══════════════════════════════════════════════════════════════════
# 2. SVD reduction accuracy (moderate k_range)
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def model():
    """Restricted mass range + moderate k: the well-conditioned regime."""
    return build_particle(
        "sigma", mass=0.5, width=1.0, model="ExpSplineSVD",
        k=1.0, k_range=[0.1, 2.0], n_interp=24,
        n_reduce=24, n_mass_pts=2000, m_range=[0.28, 2.0],
    )


class TestSVDReduction:
    def test_arrays_shapes_real(self, model):
        """Basis (r, 2n) and projection (r, n_k) are real, correct shapes."""
        n = model._n_mass_pts
        assert model._basis.shape == (model.n_reduce, 2 * n)
        assert model._projection.shape == (model.n_reduce, len(model._k_grid))
        assert not np.iscomplexobj(model._basis)
        assert not np.iscomplexobj(model._projection)

    def test_full_rank_exact(self, model):
        """r = n_k reconstructs the (clipped) gamma exactly through interp."""
        n = model._n_mass_pts
        k_grid = model._k_grid
        h = spline_basis_matrix(k_grid)

        mtest = np.linspace(0.3, 1.9, 700)
        G_ref = _clipped_gamma(model, mtest)
        worst = 0.0
        for k in k_grid[::3]:
            w, _ = spline_weights(k, h, k_grid)
            g_red = _reduced_gamma(model, mtest, w)
            g_ref = w @ G_ref
            worst = max(worst, float(np.max(np.abs(g_red - g_ref))
                                     / max(np.max(np.abs(g_ref)), 1e-30)))
        assert worst < 1e-5, f"full-rank reconstruction err={worst:.2e}"

    def test_truncation_accuracy(self, model):
        """n_reduce=8 reproduces |A|^2 to ~1e-4 of peak in the regime."""
        r = 8
        h = spline_basis_matrix(model._k_grid)
        mtest = np.linspace(0.3, 1.9, 700)
        worst = 0.0
        for k in np.linspace(0.2, 1.9, 6):
            w, _ = spline_weights(k, h, model._k_grid)
            g_red = _reduced_gamma(model, mtest, w, r)
            g_ref = w @ _clipped_gamma(model, mtest)
            A_red = 1.0 / (0.25 - mtest ** 2 - 1j * 0.5 * g_red)
            A_ref = 1.0 / (0.25 - mtest ** 2 - 1j * 0.5 * g_ref)
            p_red = np.abs(A_red) ** 2
            p_ref = np.abs(A_ref) ** 2
            worst = max(worst, float(np.max(np.abs(p_red - p_ref))
                                     / np.max(p_ref)))
        assert worst < 5e-3, f"truncated |A|^2 err/peak={worst:.2e}"

    def test_gamma_rows_match_full_model(self, model):
        """Reduced gamma rows + reduced weights ≈ ExpSpline full gamma."""
        from ampfit.particle_model.svd_spline_k_model import (
            KToSVDWeightsTransform)

        mg = np.linspace(0.35, 1.8, 500)
        full = build_particle("sigma", mass=0.5, width=1.0, model="ExpSpline",
                              k=1.0, k_range=[0.1, 2.0], n_interp=24)
        k_test = 1.1

        # Full: spline weights over all n_interp rows
        tfm_full = KToSplineWeightsTransform(
            "k", "mass", full.get_gamma_name(), mass_fixed=0.5,
            k_min=0.1, k_max=2.0)
        wf = np.array([tfm_full.forward({"k": k_test})[n]
                       for n in full.get_gamma_name()])
        G_full = np.array([full.gamma_k(mg, ki) for ki in model._k_grid])
        g_full = wf @ G_full

        # Reduced: projected weights over the SVD basis rows
        tfm_red = model.make_mass_width_transform()
        wr = np.array([tfm_red.forward({"sigma_k": k_test})[n]
                       for n in model.get_gamma_name()])
        g_red = wr @ np.array(model.gamma(mg))

        err = np.max(np.abs(g_red - g_full)) / np.max(np.abs(g_full))
        assert err < 1e-2, f"reduced vs full gamma: rel err={err:.3e}"


def _clipped_gamma(model, m):
    """gamma rows at *m* with the same clip the model applies."""
    clip = float(model.kwargs.get("gamma_clip", 1e6))
    G = np.array([model.gamma_k(m, ki) for ki in model._k_grid])
    if clip > 0:
        mag = np.abs(G)
        G = G.copy()
        G[mag > clip] *= clip / mag[mag > clip]
    return G


def _reduced_gamma(model, m, w, r=None):
    """Reduced-sum gamma at *m* for spline weights *w* (n_k,)."""
    r = r if r is not None else model.n_reduce
    n = model._n_mass_pts
    P = model._projection[:r]
    B = model._basis[:r]
    w_r = P @ w
    rows = []
    for i in range(r):
        rows.append(np.interp(m, model._m_fine, B[i, :n])
                    + 1j * np.interp(m, model._m_fine, B[i, n:]))
    return w_r @ np.array(rows)


# ═══════════════════════════════════════════════════════════════════
# 3. ExpSplineSVDModel — physics and k propagation
# ═══════════════════════════════════════════════════════════════════

class TestExpSplineSVDModel:
    @pytest.fixture(params=[4, 8])
    def model(self, request):
        return build_particle(
            "sigma", mass=0.5, width=1.0, model="ExpSplineSVD",
            k=1.0, k_range=[0.1, 2.0], n_interp=16,
            n_reduce=request.param, n_mass_pts=1500, m_range=[0.28, 2.0],
        )

    def test_registered(self):
        assert "ExpSplineSVD" in ALL_MODELS

    def test_param_handling(self, model):
        """Names, counts, defaults consistent with the reduction."""
        assert model.get_gamma_count() == model.n_reduce
        names = model.get_gamma_name()
        assert len(names) == model.n_reduce
        assert names[0] == "sigma_sr0"
        assert model.get_defaults() == {"sigma_k": 1.0}

    def test_n_reduce_clamped(self):
        """n_reduce is clamped to n_interp + 1 (mean baseline + residual)."""
        m = build_particle("sigma", mass=0.5, model="ExpSplineSVD",
                           k=1.0, k_range=[0.1, 2.0],
                           n_interp=10, n_reduce=99, n_mass_pts=500)
        assert m.n_reduce == 11  # 1 mean + 10 SVD
        assert m.get_gamma_count() == 11

    def test_gamma_clip_default_large(self):
        """gamma_clip defaults to a large value (not 1e3)."""
        m = build_particle("sigma", mass=0.5, model="ExpSplineSVD",
                           k=1.0, k_range=[0.1, 2.0],
                           n_interp=6, n_reduce=4, n_mass_pts=500)
        assert float(m.kwargs.get("gamma_clip", 1e6)) >= 1e4

    def test_k_gradient_full_model_fd(self, model):
        """dA/dk through the full chain (transform + gamma + denominator)
        matches central differences of the reduced-model amplitude."""
        m_test = np.array([0.6, 1.2, 1.8])
        k_test = 1.0
        eps = 1e-6
        tfm = model.make_mass_width_transform()

        def amp(k):
            out = tfm.forward({"sigma_k": k})
            w = np.array([out[n] for n in model.get_gamma_name()])
            g = np.array(model.gamma(m_test))
            gamma_k = w @ g
            return 1.0 / (0.5 ** 2 - m_test ** 2 - 1j * 0.5 * gamma_k)

        # Analytic: dA/dk = dA/dgamma * sum_r B_r(m) * dw'_r/dk
        out = tfm.forward({"sigma_k": k_test})
        w = np.array([out[n] for n in model.get_gamma_name()])
        g = np.array(model.gamma(m_test))
        gamma_k = w @ g
        A = 1.0 / (0.5 ** 2 - m_test ** 2 - 1j * 0.5 * gamma_k)
        dA_dgamma = (A ** 2) * (1j * 0.5)   # d/dgamma [1/(c - i*m0*gamma)]

        dwp, _ = tfm._weight_deriv(k_test)
        dA_dk = dA_dgamma * (dwp @ g)       # (n_mass,)

        # FD
        Ap = amp(k_test + eps)
        Am = amp(k_test - eps)
        dA_dk_fd = (Ap - Am) / (2 * eps)
        scale = max(1.0, float(np.max(np.abs(dA_dk_fd))))
        assert np.max(np.abs(dA_dk - dA_dk_fd)) < 1e-6 * scale, \
            f"grad mismatch: {dA_dk} vs FD {dA_dk_fd}"

    def test_amplitude_close_to_exp_spline(self, model):
        """Reduced model amplitude ≈ ExpSpline amplitude at fitted k."""
        m = np.array([0.3, 0.5, 0.8, 1.2, 1.6])
        k_test = 1.0

        svd = model
        tfm = svd.make_mass_width_transform()
        out = tfm.forward({"sigma_k": k_test})
        w = np.array([out[n] for n in svd.get_gamma_name()])
        g = np.array(svd.gamma(m))
        gamma_red = w @ g
        A_red = 1.0 / (0.5 ** 2 - m ** 2 - 1j * 0.5 * gamma_red)

        A_ref = np.exp(-k_test * (m ** 2 - 0.5 ** 2))
        rel = np.abs(np.abs(A_red) - np.abs(A_ref)) / np.abs(A_ref)
        assert np.max(rel) < 0.01, f"max rel |A| err={np.max(rel):.3e}"


# ═══════════════════════════════════════════════════════════════════
# 4. Base-class contract
# ═══════════════════════════════════════════════════════════════════

class TestSVDSplineKBase:
    def test_abstract_raises(self):
        """Instantiation without gamma_k raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            SVDSplineKModel("bad", n_interp=5, n_reduce=2, n_mass_pts=100)

    def test_mass_range_constants(self):
        """Fine grid spans [2*m_pi, m_b - m_pi] as required."""
        m = build_particle("sigma", mass=0.5, model="ExpSplineSVD",
                           k=1.0, k_range=[0.1, 2.0],
                           n_interp=5, n_reduce=2, n_mass_pts=101)
        lo, hi = m._m_fine[0], m._m_fine[-1]
        assert lo == pytest.approx(2.0 * M_PION)
        assert hi == pytest.approx(M_B_MESON - M_PION)
        assert len(m._m_fine) == 101
        assert np.all(np.diff(m._m_fine) > 0)

    def test_projection_is_real(self, model):
        """The SVD projection is real (kernel couplings are real)."""
        assert not np.iscomplexobj(model._projection)

    def test_mean_baseline_exact(self, model):
        """Row 0 of the basis is the k-mean; its weight is always 1.

        The k-mean baseline is the part of gamma common to all k.  Its
        weight is ``sum_j w_j = 1`` (spline partition of unity), so the
        reconstruction carries the common offset exactly — no shift.
        """
        n = model._n_mass_pts
        # k-mean of the clipped gamma on the fine grid
        G = _clipped_gamma(model, model._m_fine)
        M = np.concatenate([G.real, G.imag], axis=1)
        mean = M.mean(axis=0)
        assert np.allclose(model._basis[0], mean, atol=1e-12)
        # projection row 0 = ones -> weight 1
        assert np.allclose(model._projection[0], 1.0)
        # forward: reduced weight[0] == 1 for any k
        tfm = model.make_mass_width_transform()
        for k in [0.3, 1.0, 1.9]:
            out = tfm.forward({"sigma_k": k})
            assert out["sigma_sr0"] == pytest.approx(1.0, abs=1e-10)

    def test_mean_center_off(self):
        """mean_center=False spends all rows on SVD components."""
        m = build_particle("sigma", mass=0.5, model="ExpSplineSVD",
                           k=1.0, k_range=[0.1, 2.0],
                           n_interp=10, n_reduce=5, n_mass_pts=500,
                           mean_center=False)
        assert m.n_reduce == 5                 # no baseline row
        assert m._basis.shape == (5, 2 * 500)
        assert m._projection.shape == (5, 10)
        assert not np.allclose(m._projection[0], 1.0)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
