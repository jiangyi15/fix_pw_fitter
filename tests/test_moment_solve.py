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
