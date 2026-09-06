#!/usr/bin/env python3
"""Tests for the integrated_pwa Gram-matrix norm.

∫dΦ |A|² ≈ Σ_e w_e Σ_p |Σ_k ck_k a_{p,k}(e)|²
          = Σ_{k,k'} ck_k·conj(ck_k')·D[k,k'],   D = Σ_e w_e Σ_p conj(a)·a
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from ampfit.config_loader import Config
from ampfit.pwa_build import build_pwa_kernel_config, pwa_event_data
from ampfit.integrated_pwa import IntegratedPWA


def _pwa_setup(n_events=1500):
    cfg = Config("config_pwa.yml")
    kc = build_pwa_kernel_config(cfg)
    data = pwa_event_data(cfg, kc, np.load("data/phsp.npy")[:n_events])
    return kc, data


def test_gram_norm_matches_direct_pwa():
    kc, data = _pwa_setup()
    ip = IntegratedPWA(kc)
    rng = np.random.RandomState(1)
    n_m0 = int(np.max(kc["m0_index"])) + 1
    n_g0 = int(np.max(kc["g0_index"])) + 1
    ck = rng.normal(size=ip.n_wave_base) + 1j * rng.normal(size=ip.n_wave_base)
    m0 = rng.uniform(0.7, 1.5, n_m0)
    g0 = rng.uniform(0.05, 0.5, n_g0)
    params = {"ck": ck, "m0": m0, "g0": g0}

    D = ip.gram(data, m0, g0)
    assert D.shape == (ip.n_wave_base, ip.n_wave_base)
    # Hermitian
    assert np.abs(D - D.conj().T).max() < 1e-10

    norm_g = ip.norm(ck, D)
    norm_d = ip.norm_from_data(data, params)
    assert norm_g == pytest.approx(norm_d, rel=1e-9)

    # dNorm/dRe(ck_k) = 2 Re((D @ ck)_k)
    eps = 1e-6
    c = ck.copy()
    c[0] += eps
    qp = ip.norm(c, D)
    c[0] = ck[0] - eps
    qm = ip.norm(c, D)
    assert (qp - qm) / (2 * eps) == pytest.approx(
        2 * np.real((D @ ck)[0]), rel=1e-6)


def test_gram_projection_count():
    """P=2 pure-PWA: D built from p-major entries reproduces direct P."""
    kc, data = _pwa_setup(n_events=800)
    assert kc["n_proj"] == 2
    ip = IntegratedPWA(kc)
    rng = np.random.RandomState(2)
    n_m0 = int(np.max(kc["m0_index"])) + 1
    n_g0 = int(np.max(kc["g0_index"])) + 1
    ck = rng.normal(size=ip.n_wave_base) + 1j * rng.normal(size=ip.n_wave_base)
    m0 = rng.uniform(0.7, 1.5, n_m0)
    g0 = rng.uniform(0.05, 0.5, n_g0)
    D = ip.gram(data, m0, g0)
    assert np.all(np.linalg.eigvalsh((D + D.conj().T) / 2) >= -1e-8)
    assert ip.norm(ck, D) == pytest.approx(
        ip.norm_from_data(data, {"ck": ck, "m0": m0, "g0": g0}), rel=1e-9)
