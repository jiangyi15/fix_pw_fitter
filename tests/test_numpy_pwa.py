#!/usr/bin/env python3
"""Tests for the pure-PWA NumPy kernel (numpy_pwa).

``numpy_pwa.NumpyPWA`` is the original ``NumpyKernel`` amplitude chain with
the time / D0-D0bar-mixing / scalar parts removed and the two flavour
blocks promoted to an incoherent projection sum:

    P(e) = |A_0(e)|² + |A_1(e)|².

Checks:
1. equals the original mixing forward at zero time & mixing up to the
   expected factor of 2 (P_pwa = 2·P_orig for frac = 0.5, A_p = 0),
   including ck/m0/g0 gradients (same factor);
2. analytic ck/m0/g0 gradients match central finite differences.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from ampfit.config_loader import Config
from ampfit.numpy_kernel import NumpyKernel
from ampfit.numpy_pwa import NumpyPWA


@pytest.fixture(scope="module")
def kc():
    return Config("config_angle.yml").build_all_index()


@pytest.fixture(scope="module")
def data(kc):
    rng = np.random.RandomState(0)
    ne = 24
    return {
        "mass": rng.uniform(2, 3, (ne, 48)),
        "q": rng.random((ne, 72)),
        "angle": rng.random((ne, 24, 3)),
        "bkg": rng.random(ne) * 0.01,
        "weight": np.ones(ne),
    }


@pytest.fixture(scope="module")
def params(kc):
    rng = np.random.RandomState(1)
    n_wave = kc["matrix_angle"].shape[1]
    n_m0 = int(np.max(kc["m0_index"])) + 1
    n_g0 = int(np.max(kc["g0_index"])) + 1
    return {
        "ck": rng.normal(size=n_wave) + 1j * rng.normal(size=n_wave),
        "m0": rng.uniform(0.5, 2.0, n_m0),
        "g0": rng.uniform(0.05, 0.4, n_g0),
    }


def test_pwa_matches_original_at_zero_time(kc, data, params):
    """PWA == original mixing model at time=0, frac=1/2, scalars off (×2)."""
    npw = NumpyPWA(kc)
    handle = npw.load_data(data)

    Q, grads, P = npw.compute(params, handle)

    nk = NumpyKernel(kc)
    zero_scalar = (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)   # poq_rho=1 (avoid 0/0)
    d0 = dict(data)
    d0["frac"] = np.full(data["mass"].shape[0], 0.5)
    d0["time"] = np.zeros(data["mass"].shape[0])
    Qo, grads_o, Po = nk._compute(
        {**params, "scalar": zero_scalar}, d0, norm=None)

    assert Q / Qo == pytest.approx(2.0, rel=1e-9)
    assert P is not None
    assert Po is not None
    assert np.allclose(P, 2.0 * Po, rtol=1e-9)
    for key in ("ck", "m0", "g0"):
        g = np.asarray(grads[key])
        go = np.asarray(grads_o[key])
        assert g / go == pytest.approx(2.0, rel=1e-6)


def test_gradients_match_finite_difference(kc, data, params):
    """Analytic ck/m0/g0 gradients == central finite differences."""
    npw = NumpyPWA(kc)
    handle = npw.load_data(data)
    Q, grads, P = npw.compute(params, handle)
    ck, m0, g0 = params["ck"], params["m0"], params["g0"]
    eps = 1e-5

    def qf(c, m, g):
        return npw.compute({"ck": c, "m0": m, "g0": g}, handle)[0]

    # ck: real/imag gradient convention of the original kernel
    for i in [0, 7, 63, 223, 224, 447]:
        c = ck.copy()
        c[i] += eps
        qp = qf(c, m0, g0)
        c[i] = ck[i] - eps
        qm = qf(c, m0, g0)
        fd_re = (qp - qm) / (2 * eps)
        assert fd_re == pytest.approx(2 * grads["ck"][i].real, rel=1e-5)

        c = ck.copy()
        c[i] += 1j * eps
        qp = qf(c, m0, g0)
        c[i] = ck[i] - 1j * eps
        qm = qf(c, m0, g0)
        fd_im = (qp - qm) / (2 * eps)
        assert fd_im == pytest.approx(-2 * grads["ck"][i].imag, rel=1e-5)

    for idx in [0, 1, 2]:
        m = m0.copy()
        m[idx] += eps
        qp = qf(ck, m, g0)
        m[idx] = m0[idx] - eps
        qm = qf(ck, m, g0)
        fd = (qp - qm) / (2 * eps)
        assert fd == pytest.approx(grads["m0"][idx], rel=1e-5)

        g = g0.copy()
        g[idx] += eps
        qp = qf(ck, m0, g)
        g[idx] = g0[idx] - eps
        qm = qf(ck, m0, g)
        fd = (qp - qm) / (2 * eps)
        assert fd == pytest.approx(grads["g0"][idx], rel=1e-5)
