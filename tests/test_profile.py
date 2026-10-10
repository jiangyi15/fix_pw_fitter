"""Profile likelihood: quadratic prediction + Newton correction.

Two layers of validation:

1. ANALYTIC (fast, no GPU): a closed-form mock NLL
   ``q(x) = ½·dᵀH·d + κ·(wᵀd)³`` with ``d = x − x̂``.  For κ = 0 the
   profile is known exactly — ``Q_prof(Δ) = Δ²/(2·S_ii)`` with
   ``S = H⁻¹``, crossings at ``±√(2·ΔQ·S_ii)`` — so the scan machinery
   is verified to machine precision, including through the FD-Hessian
   path.  For κ > 0 the true profile drifts from the quadratic model
   and each corrected point is checked against a brute-force nuisance
   re-minimization.

2. TOY DATA (integration, GPU): the PWA toy generator
   (:func:`tabpwa.toy_pwa.toy_pwa_generate`) produces data generated
   FROM the model at truth parameters (accept-reject over phase
   space), so every parameter is constrained and the Hessian is well
   conditioned — the profile machinery assumes a minimum, which
   random-phsp data does not provide (the widths are unconstrained
   there and the Hessian has flat directions).  The corrected profile
   is validated against a brute-force profile at several offsets and
   the ΔQ = 1 interval is sanity-checked against the curvature
   estimate.
"""

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.profile import fd_hessian, profile_scan
from tabpwa.toy_pwa import toy_pwa_generate

CFG = "tests/config_pwa_bw.yml"
PARAM = "MI1m_mass"


class _PolyMock:
    """A closed-form NLL: ``q = ½·dᵀH·d + k4·(wᵀd)⁴ + k3·(wᵀd)³`` with
    ``d = x − x̂``.

    The quartic (stiffness) and cubic (skew) terms have ZERO curvature
    at d = 0, so ``S = H⁻¹`` is exactly the quadratic model the scan
    predicts with —     any deviation of ΔQ from ``Δ²/(2·S_ii)`` is pure
    non-quadraticity.  The quartic keeps q bounded below (a lone cubic
    is unbounded and a strong one lets the scan walk into −inf);
    k3 > 0 stiffens the +z side (z = wᵀd — which PARAMETER side that
    is depends on the random skew direction w).
    """

    def __init__(self, n=6, k3=0.0, k4=0.0, seed=1):
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((n, n))
        self.H = a @ a.T + n * np.eye(n)   # SPD, well conditioned
        self.S = np.linalg.inv(self.H)
        self.x_hat = rng.standard_normal(n)
        w = rng.standard_normal(n)
        self.w = w / np.linalg.norm(w)
        self.k3 = k3
        self.k4 = k4

    def get_nll(self, x):
        d = np.asarray(x, float) - self.x_hat
        z = self.w @ d
        q = 0.5 * d @ self.H @ d + self.k4 * z ** 4 + self.k3 * z ** 3
        g = self.H @ d + (4.0 * self.k4 * z ** 3
                          + 3.0 * self.k3 * z ** 2) * self.w
        return float(q), g


def test_profile_scan_quadratic_exact():
    m = _PolyMock(n=6, seed=1)
    idx = 2
    Sii = m.S[idx, idx]
    expect = np.sqrt(2.0 * Sii)          # the ΔQ = 1 crossing

    # exact h_inv: the prediction IS the profile path — every point
    # sits on the closed-form parabola
    prof = profile_scan(m, m.x_hat, m.S, idx, delta_q=1.0)
    for d, q in zip(prof.delta, prof.q):
        assert abs((q - prof.q_min) - d * d / (2.0 * Sii)) < 1e-9
    # crossings: linear interpolation between scan points on a convex
    # parabola underestimates by the chord sagitta (~3% at grow=1.6)
    assert abs(prof.cross_hi - expect) < 0.05 * expect
    assert abs(prof.cross_lo + expect) < 0.05 * expect
    lo, hi = prof.interval
    assert lo == pytest.approx(m.x_hat[idx] - expect, rel=0.05)
    assert hi == pytest.approx(m.x_hat[idx] + expect, rel=0.05)

    # the h_inv_fd path: central differences on the mock's gradient —
    # exact to roundoff for these polynomials, same crossings
    prof_fd = profile_scan(m, m.x_hat, None, idx, delta_q=1.0,
                           h_inv_fd=True)
    assert abs(prof_fd.cross_hi - expect) < 0.05 * expect
    assert abs(prof_fd.cross_lo + expect) < 0.05 * expect


def test_profile_scan_nonquadratic_matches_bruteforce():
    m = _PolyMock(n=6, k3=0.5, seed=3)
    idx = 1
    prof = profile_scan(m, m.x_hat, None, idx, delta_q=1.0,
                        max_points=8, h_inv_fd=True)
    assert prof.cross_hi is not None and prof.cross_lo is not None, \
        "the scan did not reach ΔQ = 1"

    from scipy.optimize import minimize as _minimize
    mask = np.ones(len(m.x_hat), bool)
    mask[idx] = False
    pd = np.asarray(prof.delta)

    def make_nui(delta):
        def nui(x_n):
            x = m.x_hat.copy()
            x[mask] = x_n
            x[idx] = m.x_hat[idx] + delta   # the offset, held fixed
            q, g = m.get_nll(x)
            return q, g[mask]
        return nui

    for delta in (0.5 * prof.cross_hi, prof.cross_hi,
                  1.5 * prof.cross_hi):
        k = int(np.argmin(np.abs(pd - delta)))
        # polish FROM the corrected point: can full BFGS improve on it?
        r = _minimize(make_nui(pd[k]), np.asarray(prof.x[k])[mask],
                      jac=True, method="BFGS",
                      options={"maxiter": 300, "gtol": 1e-10})
        assert abs(prof.q[k] - r.fun) < 1e-6, (
            f"Δ={pd[k]:.4f}: scan {prof.q[k]:.8f} vs brute {r.fun:.8f}")


def test_profile_scan_strongly_nonquadratic():
    """A skewed, stiff profile (ΔQ far from the quadratic model): the
    corrections must still track the TRUE profile even though the
    quadratic prediction is badly off."""
    m = _PolyMock(n=6, k3=50.0, k4=150.0, seed=3)
    idx = 1
    prof = profile_scan(m, m.x_hat, None, idx, delta_q=1.0,
                        max_points=12, h_inv_fd=True)
    assert prof.cross_hi is not None and prof.cross_lo is not None, \
        "the scan did not reach ΔQ = 1"

    # the mock IS strongly non-quadratic: S = H⁻¹ is the exact
    # quadratic part (zero curvature of k3/k4 terms at d = 0), so the
    # outermost ratios measure pure non-quadraticity — one side stiff
    # (ratio > 1), the other soft (ratio < 1).  Which side is which
    # depends on the random skew direction w.
    Sii = m.S[idx, idx]
    pd = np.asarray(prof.delta)
    ratios = (np.asarray(prof.q) - prof.q_min) / (pd * pd / (2.0 * Sii))
    ip = int(np.argmax(pd))
    im = int(np.argmin(pd))
    rp, rm = float(ratios[ip]), float(ratios[im])
    # the stiff side is monotone → its outermost point is the max; the
    # soft side peaks mid-scan (the quartic re-stiffens it far out) →
    # take the min over all points
    assert max(rp, rm) > 1.05, (
        f"no stiff side: ratios {rp:.3f}/{rm:.3f}")
    assert float(np.min(ratios)) < 0.85, (
        f"no soft side: min ratio {np.min(ratios):.3f}")
    # the stiff side rises faster → the narrower ΔQ = 1 interval
    if rp > rm:
        assert prof.cross_hi < abs(prof.cross_lo), (
            f"stiff + side but crossings not skewed: "
            f"{prof.cross_lo:+.4f}/{prof.cross_hi:+.4f}")
    else:
        assert prof.cross_hi > abs(prof.cross_lo), (
            f"stiff − side but crossings not skewed: "
            f"{prof.cross_lo:+.4f}/{prof.cross_hi:+.4f}")

    # and the corrected points still sit on the TRUE profile
    from scipy.optimize import minimize as _minimize
    mask = np.ones(len(m.x_hat), bool)
    mask[idx] = False
    for d in (prof.cross_hi, prof.cross_lo):
        k = int(np.argmin(np.abs(pd - d)))
        def nui(x_n, dd=pd[k]):
            x = m.x_hat.copy()
            x[mask] = x_n
            x[idx] = m.x_hat[idx] + dd   # the offset, held fixed
            q, g = m.get_nll(x)
            return q, g[mask]
        r = _minimize(nui, np.asarray(prof.x[k])[mask], jac=True,
                      method="BFGS", options={"maxiter": 300, "gtol": 1e-10})
        assert abs(prof.q[k] - r.fun) < 1e-6, (
            f"Δ={pd[k]:.4f}: scan {prof.q[k]:.8f} vs brute {r.fun:.8f}")


def _toy_fit(n_toy=3000, seed=73, maxiter=400):
    """Fit toy data generated from the model at truth parameters.

    Returns (fitter, fit_result, truth_x).
    """
    f = Fitter(CFG, backend="cuda_v4_pwa_s0c")
    f.apply_constrains()
    data, phsp, truth = toy_pwa_generate(f, n_toy, seed=seed, n_phsp=2000,
                                         batch_size=20000, verbose=False)
    f.set_phsp(phsp)
    f.set_data(data)

    # fit from a perturbed start (the truth generates the data)
    x_start = truth * (1.0 + 0.02 * np.sin(np.arange(len(truth)) + 1.0))
    res = f.fit(x_start, maxiter=maxiter, gtol=1e-5, disp=False)
    return f, res, truth


def test_profile_matches_bruteforce():
    f, res, _truth = _toy_fit()
    try:
        # scipy BFGS often reports "precision loss" on converged toy
        # fits; the gradient norm is the real convergence check
        x_hat = np.asarray(res.x)
        _, g_fit = f.get_nll(x_hat)
        assert np.linalg.norm(g_fit) < 1e-3, (
            f"fit did not converge: |grad| = {np.linalg.norm(g_fit):.2e}")
        names = f.cm.var_registry.flat_names
        idx = names.index(PARAM)
        # the FD Hessian (h_inv=None): the young BFGS hess_inv is too
        # coarse for an accurate prediction — the FD path (2N gradient
        # evals, once) makes the corrections converge in 1-2 steps
        prof = f.profile(res, PARAM, delta_q=1.0, max_points=10, h_inv_fd=True)
        assert prof.cross_hi is not None and prof.cross_lo is not None, \
            "the scan did not reach ΔQ = 1"

        # brute force: fix the parameter, re-minimize the NUISANCE
        # subspace (a full fit would move the scanned parameter back)
        from scipy.optimize import minimize as _minimize
        n = len(x_hat)
        mask = np.ones(n, bool)
        mask[idx] = False

        def make_nll_nuisance(delta):
            """The nuisance-subspace NLL with x_i FIXED at the offset."""
            def nll_nuisance(x_n):
                x = x_hat.copy()
                x[mask] = x_n
                x[idx] = x_hat[idx] + delta      # the offset, held fixed
                q, g = f.get_nll(x)
                return q, g[mask]
            return nll_nuisance

        # the profile point nearest each brute-force offset
        prof_delta = np.asarray(prof.delta)
        for delta in (0.5 * prof.cross_hi, prof.cross_hi,
                      1.5 * prof.cross_hi):
            k = int(np.argmin(np.abs(prof_delta - delta)))
            # start the polish FROM the corrected profile point (a fair
            # comparison: can full BFGS improve on the correction?)
            x_start_n = np.asarray(prof.x[k])[mask]
            r = _minimize(make_nll_nuisance(prof_delta[k]), x_start_n,
                          jac=True, method="BFGS",
                          options={"maxiter": 300, "gtol": 1e-8})
            q_prof = float(prof.q[k])
            assert q_prof <= r.fun + 1e-3, (
                f"Δ={prof_delta[k]:.4f}: corrected profile {q_prof:.5f} > "
                f"polished brute force {r.fun:.5f}")
            assert q_prof >= r.fun - 1e-2, (
                f"Δ={prof_delta[k]:.4f}: corrected profile {q_prof:.5f} far "
                f"below the polish {r.fun:.5f} — off the true profile")
    finally:
        f.free()


def test_profile_width_vs_bruteforce():
    """The width profile is genuinely non-parabolic (the running-width
    BW stiffens the + side): the ratio drifts off 1 with |Δ| and the
    corrected points still match the brute force there."""
    f, res, _truth = _toy_fit()
    try:
        x_hat = np.asarray(res.x)
        names = f.cm.var_registry.flat_names
        idx = names.index("MI1m_width")
        S_fd, _diag = fd_hessian(f, x_hat)
        Sii = S_fd[idx, idx]
        prof = f.profile(res, "MI1m_width", delta_q=1.0, max_points=12,
                         h_inv=S_fd)
        assert prof.cross_hi is not None and prof.cross_lo is not None, \
            "the scan did not reach ΔQ = 1"
        pd = np.asarray(prof.delta)
        ratios = (np.asarray(prof.q) - prof.q_min) / (pd * pd / (2.0 * Sii))
        ip = int(np.argmax(pd))
        assert ratios[ip] > 1.02, (
            f"width profile + side ratio {ratios[ip]:.3f} — expected the "
            "non-parabolic stiffening")

        from scipy.optimize import minimize as _minimize
        mask = np.ones(len(x_hat), bool)
        mask[idx] = False

        def make_nll_nuisance(delta):
            def nll_nuisance(x_n):
                x = x_hat.copy()
                x[mask] = x_n
                x[idx] = x_hat[idx] + delta   # the offset, held fixed
                q, g = f.get_nll(x)
                return q, g[mask]
            return nll_nuisance

        for delta in (prof.cross_hi, 1.5 * prof.cross_hi,
                      prof.cross_lo, 1.5 * prof.cross_lo):
            k = int(np.argmin(np.abs(pd - delta)))
            r = _minimize(make_nll_nuisance(pd[k]),
                          np.asarray(prof.x[k])[mask],
                          jac=True, method="BFGS",
                          options={"maxiter": 300, "gtol": 1e-8})
            assert abs(prof.q[k] - r.fun) < 1e-4, (
                f"Δ={pd[k]:.4f}: scan {prof.q[k]:.6f} vs brute {r.fun:.6f}")
    finally:
        f.free()


def test_profile_crossings_near_gaussian():
    f, res, _truth = _toy_fit()
    try:
        x_hat = np.asarray(res.x)
        names = f.cm.var_registry.flat_names
        idx = names.index(PARAM)
        # the SAME curvature the scan uses (the FD inverse Hessian):
        # comparing an FD-based crossing against the young BFGS
        # hess_inv mixes two curvature estimates and fails whenever
        # BFGS is still coarse
        S_fd, _diag = fd_hessian(f, x_hat)
        gauss = np.sqrt(S_fd[idx, idx])
        prof = f.profile(res, PARAM, delta_q=1.0, max_points=12,
                         h_inv=S_fd)
        lo, hi = prof.interval
        assert hi is not None and lo is not None
        # the true profile can be asymmetric and wider than the
        # quadratic estimate; stay within ~40% of it for this toy
        assert abs(hi - prof.x0[idx]) < 1.4 * gauss
        assert abs(prof.x0[idx] - lo) < 1.4 * gauss
    finally:
        f.free()
