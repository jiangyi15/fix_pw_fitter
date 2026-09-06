#!/usr/bin/env python3
"""Tests for the pure-PWA NumPy kernel (numpy_pwa).

``NumpyPWA`` is the original ``NumpyKernel`` amplitude chain (BW/gamma
running width, Blatt-Weisskopf fl, matrix_angle angular basis) with the
time / D0-D0bar-mixing / scalar parts removed and a generic projection sum

    n_wave = n_proj · N     (p-major entries, w = p·N + k)
    P(e)   = Σ_p |A_p(e)|²,  A_p(e) = Σ_k ck_k · a_{p,k}(e)

where all projections SHARE the single ck of length N = n_wave/n_proj.

``n_proj`` is read directly from the kernel config; ``cp_block=True``
doubles the projection count (CP/flavour-partner block counted as an
extra set of projections).

Checks:
1. n_proj from config + cp_block factor double the per-event P of a
   single block exactly (identical duplicated projection blocks);
2. analytic ck/m0/g0 gradients match central finite differences in both
   the plain and the duplicated layouts.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from ampfit.config_loader import Config, _projection_duplicate
from ampfit.numpy_pwa import NumpyPWA


@pytest.fixture(scope="module")
def kc():
    kc = Config("config_angle.yml").build_all_index()
    kc["n_proj"] = 1
    return kc


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
def m0g0(kc):
    rng = np.random.RandomState(1)
    n_m0 = int(np.max(kc["m0_index"])) + 1
    n_g0 = int(np.max(kc["g0_index"])) + 1
    return (rng.uniform(0.5, 2.0, n_m0), rng.uniform(0.05, 0.4, n_g0))


def _pwa(kc, cp_block=False):
    pwa = NumpyPWA(kc, cp_block=cp_block)
    assert pwa.n_proj == pwa.n_proj_base * (2 if cp_block else 1)
    return pwa


def test_single_block_finite_difference(kc, data, m0g0):
    """n_proj=1 (no duplication): P = |Σ_k ck a_k|²; grads match FD."""
    npw = _pwa(kc)
    h = npw.load_data(data)
    ck = (np.random.RandomState(2)
          .normal(size=npw.n_wave_base)
          + 1j * np.random.RandomState(2).normal(size=npw.n_wave_base))
    m0, g0 = m0g0
    Q, grads, P = npw.compute({"ck": ck, "m0": m0, "g0": g0}, h)
    assert P is not None and np.all(P > 0)

    eps = 1e-5

    def qf(c, m, g):
        return npw.compute({"ck": c, "m0": m, "g0": g}, h)[0]

    for i in [0, 63, 223, 447]:
        c = ck.copy()
        c[i] += eps
        qp = qf(c, m0, g0)
        c[i] = ck[i] - eps
        qm = qf(c, m0, g0)
        assert (qp - qm) / (2 * eps) == pytest.approx(
            2 * grads["ck"][i].real, rel=1e-5)

        c = ck.copy()
        c[i] += 1j * eps
        qp = qf(c, m0, g0)
        c[i] = ck[i] - 1j * eps
        qm = qf(c, m0, g0)
        assert (qp - qm) / (2 * eps) == pytest.approx(
            -2 * grads["ck"][i].imag, rel=1e-5)

    for idx in [0, 1]:
        m = m0.copy()
        m[idx] += eps
        qp = qf(ck, m, g0)
        m[idx] = m0[idx] - eps
        qm = qf(ck, m, g0)
        assert (qp - qm) / (2 * eps) == pytest.approx(
            grads["m0"][idx], rel=1e-5)

        g = g0.copy()
        g[idx] += eps
        qp = qf(ck, m0, g)
        g[idx] = g0[idx] - eps
        qm = qf(ck, m0, g)
        assert (qp - qm) / (2 * eps) == pytest.approx(
            grads["g0"][idx], rel=1e-5)


def test_projection_duplication_and_cp_block(kc, data, m0g0):
    """n_proj=P duplication multiplies P and grads by P; cp_block by 2P."""
    single = _pwa(kc)
    h1 = single.load_data(data)
    n1 = single.n_wave_base
    rng = np.random.RandomState(3)
    ck = rng.normal(size=n1) + 1j * rng.normal(size=n1)
    m0, g0 = m0g0
    Q1, g1, P1 = single.compute({"ck": ck, "m0": m0, "g0": g0}, h1)
    assert P1 is not None

    for P in (2, 3):
        kc2 = _projection_duplicate(kc, P)
        kc2["n_proj"] = P
        npw = _pwa(kc2)
        assert npw.n_wave_base == n1
        h2 = npw.load_data(data)
        Q, g, Pe = npw.compute({"ck": ck, "m0": m0, "g0": g0}, h2)
        assert Pe is not None
        assert np.allclose(Pe, P * P1, rtol=1e-9)  # block copies equal
        assert Q == pytest.approx(P * Q1, rel=1e-9)
        assert np.allclose(g["ck"], P * g1["ck"], rtol=1e-9)
        assert np.allclose(g["m0"], P * g1["m0"], rtol=1e-9)
        assert np.allclose(g["g0"], P * g1["g0"], rtol=1e-9)

        # cp_block doubles the projection count on top of n_proj: build the
        # p-major duplication for 2P copies but declare base n_proj = P.
        kc3 = _projection_duplicate(kc, 2 * P)
        kc3["n_proj"] = P
        npc = _pwa(kc3, cp_block=True)
        assert npc.n_proj == 2 * P and npc.n_wave_base == n1
        hc = npc.load_data(data)
        Qc, gc, Pc = npc.compute({"ck": ck, "m0": m0, "g0": g0}, hc)
        assert Pc is not None
        assert Qc == pytest.approx(2 * P * Q1, rel=1e-9)
        assert np.allclose(Pc, 2 * P * P1, rtol=1e-9)
        assert np.allclose(gc["ck"], 2 * P * g1["ck"], rtol=1e-9)
