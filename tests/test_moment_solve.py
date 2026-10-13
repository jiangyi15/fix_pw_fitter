"""End-to-end angular moment-method check on ``tests/config_pwa.yml``.

The config's populated chain is pure angular (all waves ``model: one``),
so the one-chain angular formula IS the amplitude.  Full process, no
likelihood fit, NO phase-space sample:

1. choose input ck, compute the angular intensity P(angles) and
   accept-reject TOY data on directly generated flat canonical angles;
2. ``solve_ckck`` builds the per-wave-pair weight functions w(x) from
   ``matrix_angle``, their analytic metric, and the data moments —
   and returns ck ck'* ABSOLUTE (the normalization is the counts).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.angular_metric import AngularMoments

CFG = "tests/config_pwa.yml"
N_DATA = 200_000


def test_moment_solve_recovers_ckck():
    f = Fitter(CFG, backend="numpy_pwa")
    f.apply_constrains()
    try:
        kc = f.kernel_config
        ma = np.asarray(kc["matrix_angle"], complex)   # (n_basis, P*N)
        k = np.asarray(kc["angle_k"], float)
        b = np.asarray(kc["angle_b"], float)
        kinds = [kk for _, kk in kc["variables"]]
        P = int(kc["n_proj_base"])
        N = ma.shape[1] // P

        # -- truth ck: the fixed wave + chosen free values ------------
        # MI1m and MI3m have PROPORTIONAL angular formulas in this
        # config (their matrix_angle columns are collinear) — the
        # angular moments cannot separate them; keep the truth out of
        # the degenerate direction (MI3m ck = 0)
        ck_free = {"jpsi->MI2p.etaMI2p->pip.pim_total_0r": 0.8,
                   "jpsi->MI3m.etaMI3m->pip.pim_total_0r": 0.0,
                   "jpsi->MI2p.etaMI2p->pip.pim_total_0i": 0.0,
                   "jpsi->MI3m.etaMI3m->pip.pim_total_0i": 0.0}
        x = f.initial_values(seed=42)
        names = f.cm.var_registry.flat_names
        for name, value in ck_free.items():
            if name in names:
                x[names.index(name)] = value
        _, resolved = f.build_params(x)
        ck = np.asarray(f._kernel_builder.forward(resolved)["ck"], complex)
        assert ck.size == N

        # -- angular amplitudes + intensity on flat angles ------------
        rng = np.random.default_rng(99)

        def flat(n):
            ang = np.empty((n, len(kinds)))
            for c, kind in enumerate(kinds):
                if kind == "phi":
                    ang[:, c] = rng.uniform(0.0, 2.0 * np.pi, n)
                else:
                    ang[:, c] = np.arccos(rng.uniform(-1.0, 1.0, n))
            return ang

        def fa_of(ang):
            z = np.prod(np.cos(ang[:, None, :] * k[None, :, :]
                               + b[None, :, :]), axis=-1)
            return (z @ ma).reshape(-1, P, N)

        def intensity(ang):
            return np.sum(np.abs(fa_of(ang) @ ck) ** 2, axis=1)

        cap = intensity(flat(1_000_000)).max()
        acc = []
        n_acc = 0
        while n_acc < N_DATA:
            ang = flat(300_000)
            pv = intensity(ang)
            if pv.max() >= cap:            # cap exceeded -> bias; redo
                cap = pv.max() * 1.05
                continue
            u = rng.uniform(0.0, cap, 300_000)
            keep = ang[u < pv]             # accept when u < P/cap * cap
            acc.append(keep)
            n_acc += len(keep)
        data_ang = np.concatenate(acc)[:N_DATA]

        # -- solve through the Fitter interface (no phsp) -------------
        am = f.angular_moments()
        kappa, waves = f.solve_ckck(data_ang)
        assert len(waves) == N
        kappa_true = np.outer(ck, ck.conj())

        # NOTE: this config's MI1m and MI3m have PROPORTIONAL angular
        # formulas (their matrix_angle columns are collinear), so the
        # pair coefficients are unique only modulo that degenerate
        # direction — the min-norm solve is not kappa itself.  The
        # physically meaningful check: the solved coefficients must
        # reproduce the true angular INTENSITY exactly.
        ang_chk = flat(200_000)
        p_true = am.intensity(kappa_true, ang_chk)
        p_solved = am.intensity(kappa, ang_chk)

        # the module's intensity convention must equal the DIRECT
        # projection sum (independent of the module internals)
        np.testing.assert_allclose(
            am.intensity(kappa_true, ang_chk),
            np.sum(np.abs(fa_of(ang_chk) @ ck) ** 2, axis=1),
            rtol=1e-9, atol=1e-12)
        # shape comparison: the solved kappa carries the counts
        # normalization (predicted counts == N_DATA) — an overall scale
        # the raw truth does not have; the recovered SHAPE must match
        pt = p_true / p_true.mean()
        ps = p_solved / p_solved.mean()
        rel = np.abs(ps - pt) / np.max(pt)
        print("\nwaves:", waves)
        print("intensity shape max |dP|/max(P):", np.max(rel))
        assert np.max(rel) < 0.02
        # the counts normalization: the predicted counts match the
        # observed sample size within its statistical fluctuation
        assert am.norm(kappa) == pytest.approx(N_DATA, rel=5e-3)
    finally:
        f.backend.free()


def test_efficiency_correction():
    """The efficiency-corrected moment solve: toys sampled from
    eps(x) . P(x) (the acceptance suppresses part of the angles), the
    AngularMoments built with the SAME eps expansion — the recovered
    ck ck'* must match the truth (shape) and the norm the counts."""
    f = Fitter(CFG, backend="numpy_pwa")
    f.apply_constrains()
    try:
        kc = f.kernel_config
        ma = np.asarray(kc["matrix_angle"], complex)
        k = np.asarray(kc["angle_k"], float)
        b = np.asarray(kc["angle_b"], float)
        kinds = [kk for _, kk in kc["variables"]]
        P = int(kc["n_proj_base"])
        N = ma.shape[1] // P

        # truth ck (real; out of the degenerate direction)
        ck_free = {"jpsi->MI2p.etaMI2p->pip.pim_total_0r": 0.8,
                   "jpsi->MI3m.etaMI3m->pip.pim_total_0r": 0.0,
                   "jpsi->MI2p.etaMI2p->pip.pim_total_0i": 0.0,
                   "jpsi->MI3m.etaMI3m->pip.pim_total_0i": 0.0}
        x = f.initial_values(seed=42)
        names = f.cm.var_registry.flat_names
        for name, v in ck_free.items():
            if name in names:
                x[names.index(name)] = v
        _, resolved = f.build_params(x)
        ck = np.asarray(f._kernel_builder.forward(resolved)["ck"], complex)

        am = f.angular_moments()          # flat: no eps
        # acceptance: eps = 1 + 0.3 Z_0 (evaluated per event); the
        # AngularMoments stays FLAT — the correction is the 1/eps
        # data weights
        rng = np.random.default_rng(77)

        def flat(n):
            ang = np.empty((n, len(kinds)))
            for c, kind in enumerate(kinds):
                ang[:, c] = (rng.uniform(0, 2 * np.pi, n) if kind == "phi"
                             else np.arccos(rng.uniform(-1, 1, n)))
            return ang

        def eps(ang):
            z = np.prod(np.cos(ang[:, None, :] * k[None, :, :]
                               + b[None, :, :]), axis=-1)
            return 1.0 + 0.3 * z[:, 0]

        def density(ang):
            return eps(ang) * am.intensity(np.outer(ck, ck.conj()), ang)

        cap = density(flat(1_000_000)).max()
        acc = []
        n = 0
        while n < N_DATA:
            ang = flat(300_000)
            pv = density(ang)
            if pv.max() >= cap:
                cap = pv.max() * 1.05
                continue
            u = rng.uniform(0, cap, 300_000)
            keep = ang[u < pv]
            acc.append(keep)
            n += len(keep)
        data_ang = np.concatenate(acc)[:N_DATA]

        # solve with the 1/eps data weights — the efficiency correction
        w = 1.0 / eps(data_ang)
        kappa, waves = f.solve_ckck(data_ang, weights=w)
        kappa_true = np.outer(ck, ck.conj())

        def p_of(kmat):
            return am.intensity(kmat, data_ang)

        pt = p_of(kappa_true) / p_of(kappa_true).mean()
        ps = p_of(kappa) / p_of(kappa).mean()
        rel = np.abs(ps - pt) / np.max(pt)
        print("\nefficiency-corrected intensity shape max |dP|/max(P):",
              np.max(rel))
        assert np.max(rel) < 0.02
        # the counts normalization with the eps-weighted integrals
        assert am.norm(kappa) == pytest.approx(N_DATA, rel=5e-3)
    finally:
        f.backend.free()


def test_weight_vectors_reduction():
    """The fixed solver reduces the basis to per-event weights whose
    sum IS the coefficient vector — downstream consumers need no
    linear algebra (no metric, no solve)."""
    f = Fitter(CFG, backend="numpy_pwa")
    f.apply_constrains()
    try:
        am = f.angular_moments()
        kinds = [kk for _, kk in f.kernel_config["variables"]]
        rng = np.random.default_rng(5)

        def flat(n):
            a = np.empty((n, len(kinds)))
            for c, kind in enumerate(kinds):
                a[:, c] = (rng.uniform(0, 2 * np.pi, n) if kind == "phi"
                           else np.arccos(rng.uniform(-1, 1, n)))
            return a

        ang = flat(50_000)
        wv = am.weight_vectors(ang)
        c_sum = wv.sum(axis=0)
        # the moments route (fixed solver applied to the sums) is the
        # same thing
        np.testing.assert_allclose(
            c_sum, am.solver @ am.moments(ang), rtol=1e-9, atol=1e-9)
        # and the Hermitian kappa matches solve() exactly
        np.testing.assert_allclose(am.kappa(c_sum), am.solve(ang),
                                   rtol=1e-9, atol=1e-9)
        # the per-event weights are reusable rows: a subsample's
        # coefficient vector is just a partial sum
        c_half = wv[:25_000].sum(axis=0)
        assert not np.allclose(c_half, c_sum)
        np.testing.assert_allclose(
            c_half, am.weight_vectors(ang[:25_000]).sum(axis=0),
            rtol=1e-12, atol=1e-12)
    finally:
        f.free()


def test_svd_weight_reduction():
    """SVD of the metric: the solver collapses to a rank-r sum of the
    scalar per-event observables, and their plain sums give the
    coefficients."""
    f = Fitter(CFG, backend="numpy_pwa")
    f.apply_constrains()
    try:
        am = f.angular_moments()
        kinds = [kk for _, kk in f.kernel_config["variables"]]
        rng = np.random.default_rng(11)

        def flat(n):
            a = np.empty((n, len(kinds)))
            for c, kind in enumerate(kinds):
                a[:, c] = (rng.uniform(0, 2 * np.pi, n) if kind == "phi"
                           else np.arccos(rng.uniform(-1, 1, n)))
            return a

        U, s2 = am.spectrum
        # the rank == the matrix rank of the metric
        assert am.rank == np.linalg.matrix_rank(am.wave_metric, tol=1e-8)
        assert am.rank < am.wave_metric.shape[0]     # the config is degenerate

        ang = flat(20_000)
        W = am.basis(ang)
        # (1) the rank-r sum identity: solver @ W(e) == sum_k u_k Phi_k/s2_k
        ex = 3
        direct = am.solver @ W[ex]
        via_svd = (U * (W[ex] @ U / s2)) @ np.ones(am.rank)
        np.testing.assert_allclose(direct, via_svd, rtol=1e-9, atol=1e-9)

        # (2) observables -> coefficients -> kappa == the direct solve
        gamma = am.modes(ang)
        assert gamma.shape == (am.rank,)
        c = am.coeffs_from_modes(gamma)
        np.testing.assert_allclose(c, am.weight_vectors(ang).sum(axis=0),
                                   rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(am.kappa(c), am.solve(ang),
                                   rtol=1e-8, atol=1e-8)

        # (3) per-event scalar weights: their sums ARE the observables
        ow = am.mode_weights(ang)
        np.testing.assert_allclose(ow.sum(axis=0), gamma, rtol=1e-12)
    finally:
        f.free()


def test_mode_errors_poisson():
    """Poisson(1) events: the across-toy scatter of the observables
    matches the sqrt-sum-of-squares error sigma = sqrt(sum w^2)."""
    f = Fitter(CFG, backend="numpy_pwa")
    f.apply_constrains()
    try:
        am = f.angular_moments()
        kinds = [kk for _, kk in f.kernel_config["variables"]]
        ck = np.array([1.0, 0.8, 0.0], complex)
        kappa_true = np.outer(ck, ck.conj())
        rng = np.random.default_rng(21)

        def flat(n):
            a = np.empty((n, len(kinds)))
            for c, kind in enumerate(kinds):
                a[:, c] = (rng.uniform(0, 2 * np.pi, n) if kind == "phi"
                           else np.arccos(rng.uniform(-1, 1, n)))
            return a

        def intensity(a):
            return am.intensity(kappa_true, a)

        cap = intensity(flat(500_000)).max()
        M, NTOY = 30, 15_000
        gammas, sigmas = [], []
        cs, we, k00, ke = [], [], [], []
        for _ in range(M):
            # POISSON event count: N ~ Poisson(NTOY) — the premise of
            # the sqrt-sum-of-squares error (a fixed-N toy would carry
            # the smaller multinomial variance)
            n_toy = rng.poisson(NTOY)
            acc, n = [], 0
            while n < n_toy:
                a = flat(30_000)
                pv = intensity(a)
                if pv.max() >= cap:
                    cap = pv.max() * 1.05
                    continue
                keep = a[rng.uniform(0, cap, 30_000) < pv]
                acc.append(keep)
                n += len(keep)
            toy = np.concatenate(acc)[:n_toy]
            gammas.append(am.modes(toy))
            sigmas.append(am.mode_errors(toy))
            cs.append(am.coeffs_from_modes(am.modes(toy)))
            we.append(am.weight_errors(toy))
            # the full-covariance route is identical to the direct
            # per-event weight errors (no diagonal approximation)
            np.testing.assert_allclose(
                am.coeff_errors(am.mode_covariance(toy)), we[-1],
                rtol=1e-8, atol=1e-8)
            k00.append(am.kappa(am.coeffs_from_modes(
                am.modes(toy)))[0, 0].real)
            ke.append(am.kappa_errors(toy)[0, 0])
        gammas = np.array(gammas)
        sigmas = np.array(sigmas)
        cs = np.array(cs)
        we = np.array(we)
        k00 = np.array(k00)
        ke = np.array(ke)
        ratio = gammas.std(axis=0) / sigmas.mean(axis=0)
        # the im-coefficient slots have identically zero weights here
        # (CP-conjugate projections) -> skip their 0/0 ratios
        nz = we.mean(axis=0) > 0
        r_c = cs.std(axis=0)[nz] / we.mean(axis=0)[nz]
        r_k = k00.std() / ke.mean()
        print("\nscatter/error  observables:", np.round(ratio, 3))
        print("scatter/error  coeffs     :", np.round(r_c, 3))
        print("scatter/error  kappa[0,0] :", round(float(r_k), 3))
        # M=30 -> ~13% on the scatter itself; allow a generous band
        assert np.all((ratio > 0.6) & (ratio < 1.7))
        assert np.all((r_c > 0.6) & (r_c < 1.7))
        assert 0.6 < r_k < 1.7
    finally:
        f.free()


def test_mode_forms_export():
    """The independent observable functions export as compact quadratic
    forms: Phi_k(x) = sum_ij Q_k[i,j] Z_i(x) Z_j(x) — reusable outside
    this module."""
    f = Fitter(CFG, backend="numpy_pwa")
    f.apply_constrains()
    try:
        am = f.angular_moments()
        Q = am.mode_forms()
        assert Q.shape == (am.rank, am.n_basis, am.n_basis)
        np.testing.assert_allclose(Q, np.transpose(Q, (0, 2, 1)),
                                   atol=1e-12)          # symmetric

        kinds = [kk for _, kk in f.kernel_config["variables"]]
        rng = np.random.default_rng(7)
        a = np.empty((5_000, len(kinds)))
        for c, kind in enumerate(kinds):
            a[:, c] = (rng.uniform(0, 2 * np.pi, 5_000) if kind == "phi"
                       else np.arccos(rng.uniform(-1, 1, 5_000)))

        # evaluate the exported basis standalone: Z and the quadratic forms
        Z = am._z(a)                                  # (n_ev, n_basis)
        phi = np.stack([((Z @ Q[k]) * Z).sum(axis=1)
                        for k in range(Q.shape[0])], axis=1)
        np.testing.assert_allclose(phi, am.mode_weights(a),
                                   rtol=1e-9, atol=1e-9)

        # the full standalone chain: basis -> sums -> recombination -> c
        A = am.mode_recombination
        assert A.shape == (len(am.wpairs), am.rank)
        c = A @ phi.sum(axis=0)
        np.testing.assert_allclose(c, am.weight_vectors(a).sum(axis=0),
                                   rtol=1e-9, atol=1e-9)
    finally:
        f.free()


def test_mode_kappa_map():
    """Every observable maps to an explicit ck ck'* combination: the
    Hermitian instrument K_k satisfies Tr(kappa @ K_k) == u_k . c for
    the coefficient vector c whose unstack is kappa."""
    f = Fitter(CFG, backend="numpy_pwa")
    f.apply_constrains()
    try:
        am = f.angular_moments()
        U = am.spectrum[0]
        K = am.mode_kappa()
        assert K.shape == (am.rank, am.n_wave, am.n_wave)
        rng = np.random.default_rng(13)
        c = rng.normal(size=len(am.wpairs))
        kappa = am.kappa(c)
        for k in range(am.rank):
            np.testing.assert_allclose(K[k], K[k].conj().T, atol=1e-12)
            got = np.trace(kappa @ K[k]).real
            np.testing.assert_allclose(got, U[:, k] @ c, rtol=1e-9,
                                       atol=1e-9)
    finally:
        f.free()


def test_physical_observables_report():
    """Observables recombined into physical ck_m ck_n* products: the
    identifiable items match the solved kappa exactly, the unmeasurable
    ones are flagged by the residual."""
    f = Fitter(CFG, backend="numpy_pwa")
    f.apply_constrains()
    try:
        am = f.angular_moments()
        kinds = [kk for _, kk in f.kernel_config["variables"]]
        ck = np.array([1.0, 0.8, 0.3], complex)
        rng = np.random.default_rng(31)

        def flat(n):
            a = np.empty((n, len(kinds)))
            for c, kind in enumerate(kinds):
                a[:, c] = (rng.uniform(0, 2 * np.pi, n) if kind == "phi"
                           else np.arccos(rng.uniform(-1, 1, n)))
            return a

        def intensity(a):
            return am.intensity(np.outer(ck, ck.conj()), a)

        cap = intensity(flat(500_000)).max()
        acc, n = [], 0
        while n < 100_000:
            a = flat(100_000)
            pv = intensity(a)
            if pv.max() >= cap:
                cap = pv.max() * 1.05
                continue
            keep = a[rng.uniform(0, cap, 100_000) < pv]
            acc.append(keep)
            n += len(keep)
        toy = np.concatenate(acc)[:100_000]

        items = [("abs", 0, 0), ("abs", 1, 1), ("abs", 2, 2),
                 ("re", 0, 1), ("re", 0, 2), ("re", 1, 2), ("im", 0, 1)]
        labels, vals, errs, resid = am.physical_observables(items, toy)
        kappa = am.solve(toy)

        for lab, v, e, r in zip(labels, vals, errs, resid):
            print(f"  {lab:16s} = {v:12.4f} +- {e:8.4f}  residual={r:.3f}")
        # the identifiable items (residual ~ 0) equal the kappa entries
        expected = [kappa[0, 0].real, kappa[1, 1].real, kappa[2, 2].real,
                    kappa[0, 1].real, kappa[0, 2].real, kappa[1, 2].real]
        for i, exp in enumerate(expected):
            if resid[i] < 1e-8:
                assert vals[i] == pytest.approx(exp, rel=1e-8)
        # truth ratios among FULLY identified items (Re(01), Re(12)):
        # Re(ck_1 ck_2*)/Re(ck_0 ck_1*) == (0.8*0.3)/(1*0.8) = 0.3
        assert resid[3] < 1e-8 and resid[5] < 1e-8
        assert vals[5] / vals[3] == pytest.approx(0.3, rel=0.10)
        # the imaginary products are unmeasurable in this config
        assert resid[6] == pytest.approx(1.0, abs=1e-6)
        assert errs[6] == pytest.approx(0.0, abs=1e-8)
    finally:
        f.free()


def test_physical_basis():
    """A basis of the observable space made of physical ck ck'* sums:
    rank elements, each fully identified (residual 0), each an explicit
    combination of the physical items."""
    f = Fitter(CFG, backend="numpy_pwa")
    f.apply_constrains()
    try:
        am = f.angular_moments()
        labels, coeffs, resid = am.physical_basis()
        assert len(labels) == am.rank          # counts, no leftover
        # `resid` is 0 for every returned basis element by construction;
        # verify the residual machinery independently instead: an Im
        # product is unmeasurable (~1), an identified diagonal is 0
        m = len(am.wpairs)
        _labs, _alphas, res = am.physical_functionals(
            [("im", 0, 1), ("abs", 0, 0)])
        _Ur = am.spectrum[0]
        _vim = np.zeros(m); _vim[am.wpairs.index((0, 1, "im"))] = 1.0
        _vab = np.zeros(m); _vab[am.wpairs.index((0, 0, "re"))] = 1.0
        np.testing.assert_allclose(
            res[0],
            np.linalg.norm(_vim - _Ur @ (_Ur.T @ _vim)) / np.linalg.norm(_vim),
            atol=1e-10)
        np.testing.assert_allclose(
            res[1],
            np.linalg.norm(_vab - _Ur @ (_Ur.T @ _vab)) / np.linalg.norm(_vab),
            atol=1e-10)
        assert res[0] > 0.9            # Im product: unmeasurable
        Ur = am.spectrum[0]
        P = Ur @ Ur.T
        for c in coeffs:                        # every element in-span
            np.testing.assert_allclose(c, P @ c, atol=1e-9)

        # measured values equal the observable recombinations, exactly
        kinds = [kk for _, kk in f.kernel_config["variables"]]
        rng = np.random.default_rng(41)

        def flat(n):
            a = np.empty((n, len(kinds)))
            for c, kind in enumerate(kinds):
                a[:, c] = (rng.uniform(0, 2 * np.pi, n) if kind == "phi"
                           else np.arccos(rng.uniform(-1, 1, n)))
            return a

        a = flat(30_000)
        labs2, vals, errs, r2 = am.physical_basis_observables(a)
        assert labs2 == labels
        alpha = coeffs @ Ur                         # basis -> gamma recombos
        np.testing.assert_allclose(
            vals, alpha @ am.modes(a), rtol=1e-8, atol=1e-8)
        assert np.all(errs > 0)
    finally:
        f.free()


def test_wave_groups():
    """wave_groups detects exactly the proportional (rank-1)
    combinations of the wave functions, verified against an independent
    computation on the matrix_angle columns."""
    f = Fitter(CFG, backend="numpy_pwa")
    f.apply_constrains()
    try:
        am = f.angular_moments()
        groups, ratios = am.wave_groups()
        ma = np.asarray(f.kernel_config["matrix_angle"], complex)
        P = am.n_proj
        cols = [np.concatenate([ma[:, p * am.n_wave + w] for p in range(P)])
                for w in range(am.n_wave)]

        def prop(i, j):
            if np.linalg.norm(cols[i]) < 1e-12 or np.linalg.norm(cols[j]) < 1e-12:
                return False
            M = np.stack([cols[i], cols[j]])
            sc = max(1.0, np.abs(M).max())
            return np.linalg.matrix_rank(M, tol=1e-9 * sc) <= 1

        # the grouping is the transitive closure of proportionality
        for g in groups:
            for i, j in zip(g, g[1:]):
                assert prop(g[0], j)
            # and members of different groups are NOT proportional
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                assert not prop(groups[a][0], groups[b][0])
        # ratios reproduce the columns
        for g, al in zip(groups, ratios):
            for m, alpha in zip(g, al):
                np.testing.assert_allclose(cols[m], alpha * cols[g[0]],
                                           atol=1e-9)
    finally:
        f.free()


def test_ck_basis_qr():
    """The ck-space rank-revealing QR (on the pair-moment Jacobian, no
    gamma basis): the independent ck directions versus the unmeasurable
    ones."""
    f = Fitter(CFG, backend="numpy_pwa")
    f.apply_constrains()
    try:
        am = f.angular_moments()
        J = am.ck_jacobian()
        assert J.shape == (len(am.wpairs), 2 * am.n_wave - 1)
        keep, drop, dR = am.ck_basis()
        assert len(keep) + len(drop) == 2 * am.n_wave - 1
        # this config: 4 independent ck directions, 1 unmeasurable
        assert len(keep) == 4
        assert len(drop) == 1
        assert int((dR > 1e-6).sum()) == 4
    finally:
        f.free()
