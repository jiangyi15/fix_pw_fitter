"""Tests for the pure-PWA partial-wave group plotter (plot_pwa_groups)."""

import numpy as np
import pytest

from ampfit.config_loader import Config
from ampfit.plot_pwa_groups import (
    discover_pwa_groups, pwa_mass_varfun, pwa_angle_varfun,
    angle_variable_labels, var_ranges)


@pytest.fixture(scope="module")
def cfg_pwa():
    return Config("config_pwa.yml")


def test_discover_policies_covers_all_base_waves(cfg_pwa):
    waves = list(cfg_pwa.full_decay.get_partial_waves())
    n_wave = len(waves)
    for by in ("chain", "resonance", "ls"):
        groups = discover_pwa_groups(cfg_pwa, by=by)
        # full ck length == n_wave (pure-PWA: no block duplication)
        flat = sorted(i for inds in groups.values() for i in inds)
        assert flat == list(range(n_wave)), by
        # every base wave belongs to exactly one group under chain/ls
        if by != "resonance":
            assert sum(len(v) for v in groups.values()) == n_wave


def test_discover_ls_tag_names(cfg_pwa):
    g = discover_pwa_groups(cfg_pwa, by="ls")
    tags = {k.split()[-1] for k in g}
    assert tags == {"PP", "DD", "FF"}      # MI1m/2p/3m: LS (1,1)…(3,3)


def test_discover_merge(cfg_pwa):
    g = discover_pwa_groups(cfg_pwa, by="ls", merge=[(".*", "ALL")])
    assert list(g) == ["ALL"]
    assert sorted(g["ALL"]) == list(range(3))


def test_angle_var_labels_phi_first():
    x = {"angle": np.zeros((5, 1, 4))}
    labels = angle_variable_labels(x)
    assert len(labels) == 4
    assert all("phi" in lb for lb in labels[:2])
    assert all("theta" in lb for lb in labels[2:])


def test_varfun_shapes_and_wrap():
    rng = np.random.RandomState(0)
    x = {"mass": rng.uniform(0.3, 4.0, size=(10, 1)),
         "angle": rng.uniform(-4, 4, size=(10, 2, 4))}
    m = pwa_mass_varfun(x)
    assert len(m) == 1 and m[0].shape == (10,)
    a = pwa_angle_varfun(x)
    assert len(a) == 8                      # 2 positions × 4 components
    # row-major layout: phi components are c=0,1 at every position
    wrapped_idx = [0, 1, 4, 5]
    for v in [a[i] for i in wrapped_idx]:   # phi block wrapped to [-pi, pi]
        assert np.all(np.abs(v) <= np.pi)
    assert not np.any(np.abs(a[2]) > 4)     # a theta component stays raw


def test_var_ranges_union():
    a = {"mass": np.array([[1.0], [3.0]])}
    b = {"mass": np.array([[0.5], [2.0]])}
    r = var_ranges(a, b, pwa_mass_varfun)
    assert r[0] == pytest.approx((0.5, 3.0))
