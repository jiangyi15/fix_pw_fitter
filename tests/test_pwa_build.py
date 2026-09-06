#!/usr/bin/env python3
"""Tests for the pure-PWA kernel-config builder (pwa_build).

Covers the J/ψ → π⁺π⁻η model (config_pwa.yml): a top spin J=1 with two
spin projections (n_proj=2) sharing one ck, single flavour block
(no identical particles, no CP partner).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math

import numpy as np
import pytest

from ampfit.config_loader import Config
from ampfit.pwa_build import (build_pwa_kernel_config, pwa_event_data,
                              pwa_duplication_factors)
from ampfit.numpy_pwa import NumpyPWA
from ampfit.helicity_angle import (decay_chain_to_tree, tree_vertices,
                                   amplitude)


@pytest.fixture(scope="module")
def cfg():
    return Config("config_pwa.yml")


@pytest.fixture(scope="module")
def kc(cfg):
    return build_pwa_kernel_config(cfg)


@pytest.fixture(scope="module")
def data(kc):
    mom = np.load("data/phsp.npy")[:100]
    return pwa_event_data(Config("config_pwa.yml"), kc, mom)


def test_duplication_factors(cfg):
    assert pwa_duplication_factors(cfg)[:2] == (1, 1)


def test_builder_shapes(kc):
    assert kc["n_proj"] == 2                 # top jpsi helicity ±1
    assert kc["n_identical"] == 1 and kc["n_cp"] == 1
    N = kc["matrix_angle"].shape[1] // kc["n_proj"]
    assert len(kc["wave_names"]) == N
    # per-wave row arrays p-major duplicated (kernel contract):
    # entries = n_wave rows, each with n_res BW / n_decay FL entries
    n_wave = kc["matrix_angle"].shape[1]
    assert len(kc["bw_order"]) == n_wave          # n_res == 1 for jpsi
    assert len(kc["fl_order"]) == 2 * n_wave      # n_decay == 2


def test_matrix_matches_engine(kc):
    """p-major matrix_angle columns == numeric helicity amplitude."""
    cfg = Config("config_pwa.yml")
    vars_ = kc["variables"]
    rng = np.random.RandomState(7)
    waves = cfg.full_decay.get_partial_waves()
    N = len(waves)
    worst = 0.0
    for kk, (ls, ch) in enumerate(waves):
        tree = decay_chain_to_tree(ch)
        nv = len(tree_vertices(tree))
        for _ in range(20):
            angdict = {v: (rng.uniform(0, 2 * math.pi),
                           math.acos(rng.uniform(-1, 1))) for v in range(nv)}
            x = [angdict[v][0] if kind == 'phi' else angdict[v][1]
                 for (v, kind) in vars_]
            for p, lam in enumerate(kc["top_states"]):
                col = p * N + kk
                val = 0j
                for b in range(len(kc["angle_index"])):
                    fac = 1.0
                    for j in range(kc["angle_k"].shape[1]):
                        fac *= math.cos(kc["angle_k"][b, j] * x[j]
                                        + kc["angle_b"][b, j])
                    val += fac * kc["matrix_angle"][b, col]
                eng = amplitude(tree, tuple(ls), angdict, lam, (0, 0, 0))
                worst = max(worst, abs(val - eng))
    assert worst < 1e-12


def test_forward_positive_and_gradients(kc, data):
    npw = NumpyPWA(kc)
    n_m0 = int(np.max(kc["m0_index"])) + 1
    n_g0 = int(np.max(kc["g0_index"])) + 1
    rng = np.random.RandomState(0)
    ck = rng.normal(size=npw.n_wave_base) + 1j * rng.normal(size=npw.n_wave_base)
    m0 = rng.uniform(0.7, 1.5, n_m0)
    g0 = rng.uniform(0.05, 0.5, n_g0)
    Q, grads, P = npw.compute({"ck": ck, "m0": m0, "g0": g0},
                              npw.load_data(data))
    assert P is not None and np.all(P > 0)
    assert np.isfinite(Q) and Q > 0

    eps = 1e-7

    def qf(c, m, g):
        return npw.compute({"ck": c, "m0": m, "g0": g}, npw.load_data(data))[0]

    c = ck.copy()
    c[0] += eps
    qp = qf(c, m0, g0)
    c[0] = ck[0] - eps
    qm = qf(c, m0, g0)
    assert (qp - qm) / (2 * eps) == pytest.approx(
        2 * grads["ck"][0].real, rel=1e-3)

    m = m0.copy()
    m[0] += eps
    qp = qf(ck, m, g0)
    m[0] = m0[0] - eps
    qm = qf(ck, m, g0)
    assert (qp - qm) / (2 * eps) == pytest.approx(
        grads["m0"][0], rel=1e-3)

    g = g0.copy()
    g[0] += eps
    qp = qf(ck, m0, g)
    g[0] = g0[0] - eps
    qm = qf(ck, m0, g)
    assert (qp - qm) / (2 * eps) == pytest.approx(
        grads["g0"][0], rel=1e-3)
