#!/usr/bin/env python3
"""Test the minimal-set angular-amplitude cache (cuda_v3_ampcache layout).

The per-wave amplitude factorizes as BW_w(m)·Amp_w(q, angles); the
angular part Amp_w = fa·fl is pure kinematics and cacheable.  These
tests check the wave→slot layout and that the cached recomputation is
numerically identical to the full NumpyKernel forward.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from ampfit.config_loader import Config
from ampfit.numpy_kernel import NumpyKernel
from ampfit.amp_cache import (build_amp_cache_layout, fill_amp_cache,
                              amp_factor_cached)

CONFIGS = ["config_angle.yml", "config_amp.yml"]


def _config(name):
    return Config(name).build_all_index()


def _data(n_events, seed=0):
    rng = np.random.RandomState(seed)
    return {
        "mass": rng.uniform(2, 3, (n_events, 48)),
        "q": rng.random((n_events, 72)),
        "angle": rng.random((n_events, 24, 3)),
        "frac": rng.random((n_events,)),
        "time": rng.random((n_events,)),
        "bkg": rng.random((n_events,)) * 0.01,
        "weight": np.ones((n_events,)),
    }


@pytest.mark.parametrize("config_name", CONFIGS)
def test_layout_minimal_set(config_name):
    """Cache size is well below n_wave and covers every wave."""
    c = _config(config_name)
    n_wave = c["matrix_angle"].shape[1]
    n_uniq, slot, rep = build_amp_cache_layout(c)
    assert n_uniq < n_wave                      # minimal set is smaller
    assert slot.shape == (n_wave,)
    assert rep.shape == (n_uniq,)
    assert len(np.unique(slot)) == n_uniq       # all slots used
    assert np.all(rep >= 0) and np.all(rep < n_wave)
    # every wave maps into a valid slot
    assert np.all((slot >= 0) & (slot < n_uniq))
    # representative of each slot maps back to that slot
    assert np.all(slot[rep] == np.arange(n_uniq, dtype=np.int32))


@pytest.mark.parametrize("config_name", CONFIGS)
def test_slot_members_share_signature(config_name):
    """Members of a slot are byte-identical in (col, fl-row) — the invariant
    that makes caching one value per slot exact."""
    c = _config(config_name)
    n_wave = c["matrix_angle"].shape[1]
    n_decay = c["fl_order"].size // n_wave
    n_uniq, slot, rep = build_amp_cache_layout(c)
    cols = c["matrix_angle"].T
    fl_rows = np.asarray(c["fl_order"]).reshape(n_wave, n_decay)
    sig = [(cols[w].tobytes(), fl_rows[w].tobytes()) for w in range(n_wave)]
    for s in range(n_uniq):
        members = np.where(slot == s)[0]
        base = sig[int(members[0])]
        assert all(sig[int(w)] == base for w in members[1:])


@pytest.mark.parametrize("config_name", CONFIGS)
def test_cached_forward_matches_reference(config_name):
    """Cached (Amp by slot / bw_p) == full NumpyKernel spatial factor."""
    c = _config(config_name)
    nk = NumpyKernel(c)
    n_uniq, slot, rep = build_amp_cache_layout(c)
    data = _data(150, seed=1)
    m0 = np.random.uniform(1.5, 3.0, int(np.max(c["m0_index"])) + 1)
    g0 = np.random.uniform(0.05, 0.5, int(np.max(c["g0_index"])) + 1)

    cache = fill_amp_cache(data, c, nk, rep)
    assert cache.shape == (150, n_uniq)
    assert np.all(np.isfinite(cache))

    ca_ref = nk._compute_common_amp_factor(data, m0=m0, g0=g0)
    ca_cached = amp_factor_cached(data, c, nk, m0, g0, slot, cache)
    # machine-precision agreement (measured rel ~2e-16); rtol-only so near a
    # BW pole the large |factor| does not falsely fail
    assert np.allclose(ca_cached, ca_ref, rtol=1e-10, atol=0)


def _forward_Q_from_ca(ca, ck, scalar, data, norm):
    """Q from a precomputed spatial factor ca (n_events, n_wave)."""
    n_wave = ca.shape[1]
    Gamma, Delta_Gamma, Delta_m, A_p, poq_rho, pop_phi = scalar
    frac, time, weight, bkg = (data["frac"], data["time"],
                               data["weight"], data["bkg"])
    a = ck * ca
    a_rs = a.reshape(-1, 2, n_wave // 2)
    ap = a_rs[:, 0, :].sum(-1)
    am = a_rs[:, 1, :].sum(-1)
    eL = np.exp(-1j * time * (-Delta_m / 2 - 1j * (Gamma + Delta_Gamma / 2) / 2))
    eH = np.exp(-1j * time * (+Delta_m / 2 - 1j * (Gamma - Delta_Gamma / 2) / 2))
    gp, gm = (eL + eH) / 2, (eL - eH) / 2
    poq = poq_rho * np.exp(1j * pop_phi)
    pap = gp * ap + gm * poq * am
    pam = (gm / poq) * ap + gp * am
    pb, pbbar = np.abs(pap) ** 2, np.abs(pam) ** 2
    P = frac * pb * (1 - A_p) + (1 - frac) * pbbar * (1 + A_p)
    return (-np.sum(weight * np.log(P / norm + bkg)) if norm is not None
            else np.sum(weight * P))


@pytest.mark.parametrize("config_name", CONFIGS)
def test_cached_gradients_match_reference(config_name):
    """m0/g0 finite-difference gradients of the cached forward match the
    analytic NumpyKernel gradients."""
    c = _config(config_name)
    nk = NumpyKernel(c)
    n_uniq, slot, rep = build_amp_cache_layout(c)
    data = _data(120, seed=2)
    n_m0 = int(np.max(c["m0_index"])) + 1
    n_g0 = int(np.max(c["g0_index"])) + 1
    rng = np.random.RandomState(3)          # deterministic: poles would flake
    m0 = rng.uniform(1.5, 3.0, n_m0)
    g0 = rng.uniform(0.05, 0.5, n_g0)
    ck = rng.randn(nk.n_wave) + 1j * rng.randn(nk.n_wave)
    norm = 42.0
    scalar = (0.6, 0.01, 0.506, 0.01, 0.9, 0.2)
    params = {"ck": ck, "m0": m0, "g0": g0, "scalar": scalar}

    Q_ref, grad_ref, P_ref = nk._compute(params, data, norm=norm)

    cache = fill_amp_cache(data, c, nk, rep)
    ca = amp_factor_cached(data, c, nk, m0, g0, slot, cache)
    assert np.allclose(_forward_Q_from_ca(ca, ck, scalar, data, norm),
                       Q_ref, rtol=1e-10, atol=0)

    # 3-point FD on the cached forward
    eps = 1e-6
    fd_m0 = np.zeros(n_m0)
    for i in range(n_m0):
        m0p, m0m = m0.copy(), m0.copy()
        m0p[i] += eps
        m0m[i] -= eps
        cap = amp_factor_cached(data, c, nk, m0p, g0, slot, cache)
        cam = amp_factor_cached(data, c, nk, m0m, g0, slot, cache)
        fd_m0[i] = (_forward_Q_from_ca(cap, ck, scalar, data, norm)
                    - _forward_Q_from_ca(cam, ck, scalar, data, norm)) / (2 * eps)
    fd_g0 = np.zeros(n_g0)
    for i in range(n_g0):
        g0p, g0m = g0.copy(), g0.copy()
        g0p[i] += eps
        g0m[i] -= eps
        cap = amp_factor_cached(data, c, nk, m0, g0p, slot, cache)
        cam = amp_factor_cached(data, c, nk, m0, g0m, slot, cache)
        fd_g0[i] = (_forward_Q_from_ca(cap, ck, scalar, data, norm)
                    - _forward_Q_from_ca(cam, ck, scalar, data, norm)) / (2 * eps)

    for name, fd, ref in (("m0", fd_m0, grad_ref["m0"]),
                          ("g0", fd_g0, grad_ref["g0"])):
        scale = np.abs(ref).max()
        assert scale > 0, f"{name}: zero reference gradient"
        keep = np.abs(ref) > 1e-3 * scale      # skip FD noise on ~0 grads
        assert np.allclose(fd[keep], ref[keep], rtol=1e-4, atol=1e-4 * scale)


if __name__ == "__main__":
    for cfg in CONFIGS:
        test_layout_minimal_set(cfg)
        test_slot_members_share_signature(cfg)
        test_cached_forward_matches_reference(cfg)
        test_cached_gradients_match_reference(cfg)
    print("\nAll amp_cache tests passed!")
