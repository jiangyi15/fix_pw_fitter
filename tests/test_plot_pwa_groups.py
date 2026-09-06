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


# ── ReadVar item classes (read_var.py) ────────────────────────────────

@pytest.fixture(scope="module")
def pwa_events():
    ne = 12
    rng = np.random.RandomState(11)
    return {"mass": np.arange(ne * 3, dtype=float).reshape(ne, 3) / 3 + 1,
            "angle": rng.uniform(-np.pi, np.pi, size=(ne, 3, 4)),
            "q": np.ones((ne, 3)), "weight": np.ones(ne)}


def test_readvar_item_classes(cfg_pwa):
    from ampfit.read_var import ReadVar, MassVar, AngleVar
    m = ReadVar(cfg_pwa, "pipeta")
    assert isinstance(m, MassVar) and m.kind == "mass"
    a = ReadVar(cfg_pwa, ("angle", "pipi/pip", "alpha"))
    assert isinstance(a, AngleVar) and a.kind == "angle"
    b = ReadVar(cfg_pwa, ("angle", "pipi", "cos(beta)"))
    assert isinstance(b, AngleVar) and b.angle_kind == "cos(beta)"
    with pytest.raises(ValueError):
        ReadVar(cfg_pwa, ("angle", "pipeta", "alpha"))   # no wave chain


def test_massvar_reads_column(cfg_pwa, pwa_events):
    from ampfit.read_var import MassVar
    m0 = MassVar(cfg_pwa, "pipi")
    m1 = MassVar(cfg_pwa, "pipeta")
    assert np.allclose(m0.read(pwa_events), pwa_events["mass"][:, 0])
    assert np.allclose(m1.read(pwa_events), pwa_events["mass"][:, 1])


def test_anglevar_vertex_mapping(cfg_pwa, pwa_events):
    from ampfit.read_var import AngleVar
    a = pwa_events["angle"]
    wrap = (a[:, 0, 0] + np.pi) % (2 * np.pi) - np.pi
    assert np.allclose(AngleVar(cfg_pwa, "pipi", "alpha").read(pwa_events),
                       wrap)
    wrap1 = (a[:, 0, 1] + np.pi) % (2 * np.pi) - np.pi
    assert np.allclose(
        AngleVar(cfg_pwa, "pipi/pip", "alpha").read(pwa_events), wrap1)
    assert np.allclose(
        AngleVar(cfg_pwa, "pipi", "cos(beta)").read(pwa_events),
        np.cos(a[:, 0, 2]))
    assert AngleVar(cfg_pwa, "pipi", "alpha").range == (-np.pi, np.pi)


def test_exprvar_elementwise(cfg_pwa, pwa_events):
    from ampfit.read_var import MassVar, ExprVar
    mv = {f"m_{n}": MassVar(cfg_pwa, n) for n in ("pipi", "pipeta")}
    e = ExprVar("max(m_pipi,m_pipeta) ** 2 - m_pipi", mv)
    got = e.read(pwa_events)
    want = np.maximum(pwa_events["mass"][:, 0],
                      pwa_events["mass"][:, 1]) ** 2 - pwa_events["mass"][:, 0]
    assert np.allclose(got, want)


def test_vars_from_config_classes(cfg_pwa):
    from ampfit.read_var import vars_from_config, MassVar, AngleVar
    items = dict(vars_from_config(cfg_pwa))
    assert isinstance(items["pipi"], MassVar)
    # config_pwa has only topology 0 wave-active -> pipeta angles skipped
    assert "pipi alpha" in items and "pipi/pip alpha" in items
    assert not any(k.startswith("pipeta ") for k in items)


def test_config_panels_from_readvars(cfg_pwa, pwa_events):
    from ampfit.plot_pwa_groups import config_panels
    p = config_panels(cfg_pwa, pwa_events, pwa_events)
    assert p["mass"]["keys"] == ["pipi", "pipeta", "pimeta"]
    # pipi + pipi/pip (alpha/cos) + the two extra_vars expressions
    assert len(p["angles"]["keys"]) == 6
    v = p["angles"]["varfun"](pwa_events)
    assert len(v) == 6 and all(x.shape == (12,) for x in v)
