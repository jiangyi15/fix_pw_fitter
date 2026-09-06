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


def test_declaration_driven_row_blocks(cfg):
    """Row-block factors come from identical/cp declarations in the config."""
    from ampfit.config_loader import row_block_factors

    assert row_block_factors(cfg.dic) == (1, 1, 1)     # config_pwa: none

    angle = Config("config_angle.yml")                 # legacy B->4pi
    assert row_block_factors(angle.dic) == (4, 2, 8)
    kc = angle.build_all_index()
    assert kc["n_blocks"] == 8 and kc["n_perm"] == 4 and kc["n_cp"] == 2
    assert kc["matrix_angle"].shape[1] == 448
    cm = angle.get_ck_map()
    assert len(cm) == 448
    n_ls = sum(1 for t in cm if "g_lsbar" in t[1])
    assert n_ls == 224                     # CP partner half carries g_lsbar


def test_build_all_index_routes_pure_pwa(cfg):
    """A single generic build_all_index emits the pure-PWA arrays (C==1)."""
    from ampfit.config_loader import row_block_factors

    assert row_block_factors(cfg.dic) == (1, 1, 1)
    c2 = Config("config_pwa.yml")
    kc = c2.build_all_index()                      # single generic entry point
    kc_ref = build_pwa_kernel_config(Config("config_pwa.yml"))
    for key in ("matrix_angle", "bw_order", "fl_order", "m0_index",
                "mass_index", "g0_index", "g0_mass_index", "fl_type",
                "fl_q_index", "angle_index", "angle_k", "angle_b",
                "matrix_gamma", "gamma_table", "fl_table"):
        assert np.allclose(kc[key], kc_ref[key], atol=1e-14)
    # Fitter-facing attributes synced for defaults/constraints
    assert c2.m0_phys_name == kc_ref["m0_names"]
    assert c2.g0_phys_name == kc_ref["g0_names"]
    assert len(kc["ck_map"]) == kc["matrix_angle"].shape[1] // kc["n_proj"]


def test_generate_pwa_phsp_conserves_four_momentum(cfg):
    """Flat phsp via two-body products + inverse boost chain: on shell."""
    from ampfit.pwa_build import generate_pwa_phsp

    chain = cfg.full_decay.get_partial_waves()[0][1]
    mom = generate_pwa_phsp(cfg, chain, 2000, seed=3)
    tot = mom.sum(axis=1)
    E = tot[:, 0]
    p3 = np.linalg.norm(tot[:, 1:], axis=1)
    M = cfg.dic["particle"][cfg.top]["mass"]
    assert np.abs(E - M).max() < 1e-6
    assert p3.max() < 1e-6
    mass_ok = [np.sqrt(np.clip(mom[:, i] ** 2 @ np.array([1, -1, -1, -1.]),
                               0, None)) for i in range(3)]
    for i, f in enumerate(cfg.finals):
        assert np.allclose(mass_ok[i], cfg.dic["particle"][f]["mass"],
                           atol=1e-6)


def test_load_all_data_prefix_momenta(tmp_path):
    """load_all_data converts data/phsp 4-momentum files via pwa_event_data."""
    import yaml
    from ampfit import Fitter
    from ampfit.pwa_build import generate_pwa_phsp

    cfg0 = Config("config_pwa.yml")
    chain = cfg0.full_decay.get_partial_waves()[0][1]
    mom = generate_pwa_phsp(cfg0, chain, 300, seed=7)

    d = yaml.safe_load(open("config_pwa.yml"))
    for pref in ("data", "phsp"):
        d["data"][pref] = f"{pref}.npy"
        d["data"][f"{pref}_weight"] = f"{pref}_w.npy"
    cfgp = tmp_path / "cfg.yml"
    cfgp.write_text(yaml.safe_dump(d))
    np.save(tmp_path / "data.npy", mom[:100])
    np.save(tmp_path / "data_w.npy", np.ones(100))
    np.save(tmp_path / "phsp.npy", mom)
    np.save(tmp_path / "phsp_w.npy", np.ones(300))

    f = Fitter(str(cfgp), backend="numpy_pwa")
    dn, pn = f.load_all_data()
    assert dn["mass"].shape == (100, 1)
    assert pn["mass"].shape == (300, 1)
    assert dn["angle"].shape == (100, 1, 4)     # canonical, any n_comps
    assert dn["q"].shape == (100, 2)
    # single-block ck reused via kc['ck_map']; NLL finite from defaults
    nll, grad = f.get_nll(f.initial_values())
    assert np.isfinite(nll)
    assert np.all(np.isfinite(np.asarray(grad)))
