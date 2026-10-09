"""Tests for the pure-PWA partial-wave group plotter (plot_pwa_groups)."""

import numpy as np
import pytest

from tabpwa.config_loader import Config
from tabpwa.plot_pwa_groups import (
    discover_pwa_groups, pwa_mass_varfun, pwa_angle_varfun,
    angle_variable_labels, var_ranges)


@pytest.fixture(scope="module")
def cfg_pwa():
    return Config("tests/config_pwa.yml")


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
    from tabpwa.read_var import ReadVar, MassVar, AngleVar
    m = ReadVar(cfg_pwa, "pipeta")
    assert isinstance(m, MassVar) and m.kind == "mass"
    a = ReadVar(cfg_pwa, ("angle", "pipi/pip", "alpha"))
    assert isinstance(a, AngleVar) and a.kind == "angle"
    b = ReadVar(cfg_pwa, ("angle", "pipi", "cos(beta)"))
    assert isinstance(b, AngleVar) and b.angle_kind == "cos(beta)"
    with pytest.raises(ValueError):
        ReadVar(cfg_pwa, ("angle", "pipeta", "alpha"))   # no wave chain


def test_massvar_reads_column(cfg_pwa, pwa_events):
    from tabpwa.read_var import MassVar
    m0 = MassVar(cfg_pwa, "pipi")
    m1 = MassVar(cfg_pwa, "pipeta")
    assert np.allclose(m0.read(pwa_events), pwa_events["mass"][:, 0])
    assert np.allclose(m1.read(pwa_events), pwa_events["mass"][:, 1])


def test_anglevar_vertex_mapping(cfg_pwa, pwa_events):
    from tabpwa.read_var import AngleVar
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
    from tabpwa.read_var import MassVar, ExprVar
    mv = {f"m_{n}": MassVar(cfg_pwa, n) for n in ("pipi", "pipeta")}
    e = ExprVar("max(m_pipi,m_pipeta) ** 2 - m_pipi", mv)
    got = e.read(pwa_events)
    want = np.maximum(pwa_events["mass"][:, 0],
                      pwa_events["mass"][:, 1]) ** 2 - pwa_events["mass"][:, 0]
    assert np.allclose(got, want)


def test_vars_from_config_classes(cfg_pwa):
    from tabpwa.read_var import vars_from_config, MassVar, AngleVar
    items = dict(vars_from_config(cfg_pwa))
    assert isinstance(items["pipi"], MassVar)
    # config_pwa has only topology 0 wave-active -> pipeta angles skipped
    assert "pipi alpha" in items and "pipi/pip alpha" in items
    assert not any(k.startswith("pipeta ") for k in items)


def test_config_panels_from_readvars(cfg_pwa, pwa_events):
    from tabpwa.plot_pwa_groups import config_panels
    p = config_panels(cfg_pwa, pwa_events, pwa_events)
    assert p["mass"]["keys"] == ["pipi", "pipeta", "pimeta"]
    # pipi + pipi/pip (alpha/cos) + the two extra_vars expressions
    assert len(p["angles"]["keys"]) == 6
    v = p["angles"]["varfun"](pwa_events)
    assert len(v) == 6 and all(x.shape == (12,) for x in v)


def test_apply_plot_entry_priority(cfg_pwa):
    """per-variable config bins/range override plot.config defaults."""
    from tabpwa.read_var import MassVar
    defaults = {"nbins": 200, "legend": False, "range": [0.0, 5.0]}
    v = MassVar(cfg_pwa, "pipi").apply_plot_entry(defaults)
    assert v.nbins == 200 and v.range == (0.0, 5.0)
    v2 = MassVar(cfg_pwa, "pipi").apply_plot_entry(
        {"bins": 25, "range": [0.5, 2.0]}, defaults)
    assert v2.nbins == 25 and v2.range == (0.5, 2.0)   # entry wins
    v3 = MassVar(cfg_pwa, "pipi").apply_plot_entry(
        {"bins": 40}, defaults)
    assert v3.nbins == 40 and v3.range == (0.0, 5.0)   # defaults retained


# ── analytic 3-body Dalitz boundary (plot.2Dplot add_dalitz_boundary) ───────

_M0, _M1, _M2, _M3 = (3.096813522559038, 0.139567, 0.139567, 0.5478621409593663)


def test_dalitz_boundary_math():
    from tabpwa.plot_pwa_groups import dalitz_boundary, kine_min_max
    s12, lo, hi = dalitz_boundary(_M0, _M1, _M2, _M3, N=64)
    assert s12[0] == pytest.approx((_M1 + _M2) ** 2)
    assert s12[-1] == pytest.approx((_M0 - _M3) ** 2)
    assert np.all(lo <= hi + 1e-12)
    # the region touches both corner lines at a single (degenerate) point
    assert lo[0] == pytest.approx(hi[0], abs=1e-9)
    assert lo[-1] == pytest.approx(hi[-1], abs=1e-6)
    # interior slice has a finite width and stays physical
    assert lo[32] < hi[32]
    assert np.all(lo > 0) and np.all(hi < (_M0 - _M1) ** 2 + 1e-9)
    # kine_min_max clamps out-of-range s12 to the boundary endpoints
    smin, smax = kine_min_max([0.0, s12[32], 1e6], _M0, _M1, _M2, _M3)
    lo_pt = kine_min_max([(_M1 + _M2) ** 2], _M0, _M1, _M2, _M3)
    hi_pt = kine_min_max([(_M0 - _M3) ** 2], _M0, _M1, _M2, _M3)
    assert smin[0] == pytest.approx(lo_pt[0][0])
    assert smax[0] == pytest.approx(lo_pt[1][0])
    assert smin[2] == pytest.approx(hi_pt[0][0])
    assert smax[2] == pytest.approx(hi_pt[1][0])


def test_dalitz_boundary_masses_rule(cfg_pwa):
    """[pipi, pipeta] = (pip,pim)+(pip,eta): B=pip shared, A=pim, C=eta."""
    from tabpwa.plot_pwa_groups import dalitz_boundary_masses
    m = dalitz_boundary_masses(cfg_pwa, "pipi", "pipeta")
    assert m == pytest.approx((_M0, _M1, _M2, _M3))
    # swapped order swaps A and C -> axes-transposed (same) region
    m2 = dalitz_boundary_masses(cfg_pwa, "pipeta", "pipi")
    assert m2 == pytest.approx((_M0, _M3, _M1, _M1))
    # the second config panel: (pip,eta)+(pim,eta), shared eta
    m3 = dalitz_boundary_masses(cfg_pwa, "pipeta", "pimeta")
    assert m3 == pytest.approx((_M0, _M1, _M3, _M1))
    with pytest.raises(ValueError, match="rule \\(A,B\\)\\+\\(B,C\\)"):
        dalitz_boundary_masses(cfg_pwa, "pipi", "pipi")


def test_plots_2d_from_config(cfg_pwa, pwa_events):
    from tabpwa.read_var import plots_2d_from_config
    panels = plots_2d_from_config(cfg_pwa)
    assert [k for k, _ in panels] == ["dalitz_1", "dalitz_2"]
    spec = dict(panels)["dalitz_1"]
    assert spec["x"].name == "m_pipi**2" and spec["y"].name == "m_pipeta**2"
    assert spec["boundary"] == ["pipi", "pipeta"]
    assert spec["binning"] == [[2, 2]] * 4   # explicit in config
    vx, vy = spec["x"].read(pwa_events), spec["y"].read(pwa_events)
    assert vx.shape == vy.shape == (12,)
    assert dict(panels)["dalitz_2"]["boundary"] == ["pipeta", "pimeta"]


def test_auto_binning_for_small_samples(cfg_pwa, pwa_events):
    """No adaptive_binning in the entry -> levels derived from N_data."""
    from tabpwa.amp_model import build_amplitude_model
    from tabpwa.plot_pwa_groups import config_2d_panels, auto_binning
    assert auto_binning(200) == [[2, 2]] * 2          # small sample
    assert auto_binning(20000) == [[2, 2]] * 4        # large sample (cap)
    dic = dict(cfg_pwa.dic)
    dic["plot"] = dict(cfg_pwa.dic["plot"])
    dic["plot"]["2Dplot"] = {
        "auto": {"x": "m_pipi**2", "y": "m_pipeta**2"}}   # no binning key
    cfg2 = build_amplitude_model(dic)
    panels = {p["key"]: p for p in config_2d_panels(cfg2, pwa_events)}
    assert panels["auto"]["binning"] == auto_binning(12)  # 12 toy events


def test_config_2d_panels_resolves_boundary(cfg_pwa, pwa_events):
    from tabpwa.plot_pwa_groups import config_2d_panels
    panels = {p["key"]: p for p in config_2d_panels(cfg_pwa, pwa_events)}
    assert set(panels) == {"dalitz_1", "dalitz_2"}
    assert panels["dalitz_1"]["boundary"] == pytest.approx(
        [_M0, _M1, _M2, _M3])
    v = panels["dalitz_1"]["varfun"](pwa_events)
    assert len(v) == 2 and all(a.shape == (12,) for a in v)
    assert panels["dalitz_1"]["scatter_style"] == {
        "c": "black", "s": 1, "alpha": 0.0}




# ── plot_2d drawing on a real Fitter (numpy_pwa, synthetic events) ─────────

@pytest.fixture(scope="module")
def real_plotter():
    """Real Fitter + real compute(); only the event sample is synthetic."""
    from tabpwa import Fitter, FitResult
    from tabpwa.pwa_build import build_tree_event_data, generate_pwa_phsp
    from tabpwa.plot_pw_groups import PWGroupPlotter

    f = Fitter("tests/config_pwa.yml", backend="numpy_pwa")
    f.apply_constrains()
    model, kc = f.model, f.kernel_config
    tree = f.decay_tree
    chain = tree.partial_waves()[0][1]
    byt = {tree.topo_index[ch.topo_id()]: ch
           for _, ch in tree.partial_waves()}
    phsp = build_tree_event_data(
        model, kc, byt, generate_pwa_phsp(model, chain, 1500, seed=41))
    data = build_tree_event_data(
        model, kc, byt, generate_pwa_phsp(model, chain, 1500, seed=42))
    f.set_phsp(phsp)
    f.set_data(data)
    r = FitResult.from_x(f.initial_values(seed=43))
    groups = {"ALL": list(range(len(model.full_decay.get_partial_waves())))}
    return PWGroupPlotter(f, r, groups).compute()


def test_plot_2d_boundary_draw(tmp_path, real_plotter):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    varfun = lambda d: [d["mass"][:, 0] ** 2, d["mass"][:, 1] ** 2]
    real_plotter.plot_2d(varfun, ["x", "y"], "bnd_test", binning=[[2, 2]],
                         output=str(tmp_path), boundary=[_M0, _M1, _M2, _M3])
    assert (tmp_path / "bnd_test.png").exists()

    # explicit ax + boundary resolved BY NAME through the real fitter.model
    fig, ax = plt.subplots()
    real_plotter.plot_2d(varfun, ["x", "y"], "bnd_ax", binning=[[2, 2]],
                         output=None, ax=ax, boundary=["pipi", "pipeta"])
    assert len(ax.lines) == 2
    assert ax.lines[0].get_xdata()[0] == pytest.approx((_M1 + _M2) ** 2)
    assert ax.lines[0].get_xdata()[-1] == pytest.approx((_M0 - _M3) ** 2)
    plt.close(fig)

    with pytest.raises(ValueError, match="2 particle names or 4 masses"):
        real_plotter.plot_2d(varfun, ["x", "y"], "bad", output=None,
                             boundary=[1.0, 2.0, 3.0])
    plt.close("all")


def test_plot_2d_plot_figs(tmp_path, real_plotter):
    """plot_figs selects the written figures (tf-pwa semantics)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    varfun = lambda d: [d["mass"][:, 0] ** 2, d["mass"][:, 1] ** 2]
    real_plotter.plot_2d(varfun, ["x", "y"], "pf", binning=[[2, 2]],
                         output=str(tmp_path),
                         plot_figs=["data_hist", "fitted", "pull"])
    assert (tmp_path / "pf.png").exists()              # pull
    assert (tmp_path / "pf_data_hist.png").exists()
    assert (tmp_path / "pf_fitted.png").exists()
    assert not (tmp_path / "pf_sideband_hist.png").exists()

    # pull not requested -> only the extras; sideband skipped (no bkg)
    real_plotter.plot_2d(varfun, ["x", "y"], "pf2", binning=[[2, 2]],
                         output=str(tmp_path),
                         plot_figs=["fitted", "sideband_hist"])
    assert (tmp_path / "pf2_fitted.png").exists()
    assert not (tmp_path / "pf2.png").exists()

    with pytest.raises(ValueError, match="unknown plot_figs entry"):
        real_plotter.plot_2d(varfun, ["x", "y"], "pf3", output=str(tmp_path),
                             plot_figs=["nonsense"])
    plt.close("all")


def test_dalitz_boundary_internal_node():
    """(A,B)+(B,C) at an internal node: m0 = the parent, not the top."""
    from types import SimpleNamespace
    from tabpwa.decay_tree import DecayTree
    from tabpwa.plot_pwa_groups import dalitz_boundary_masses
    # B0 -> D0 pi;  D0 -> (Rp -> pip pi0)(Rm -> pim pi0)
    spec = {
        "B0": ["D0", "pi"],
        "D0": [["Rp", "pim"], ["Rm", "pip"]],
        "Rp": ["pip", "pi0"],
        "Rm": ["pim", "pi0"],
    }
    tree = DecayTree(spec, {"$top": "B0",
                            "$finals": ["pip", "pim", "pi0", "pi"],
                            **{p: {"J": 0} for p in
                               ["B0", "D0", "Rp", "Rm", "pip", "pim",
                                "pi0", "pi"]}})
    cfg = SimpleNamespace(decay_tree=tree, dic={"particle": {
        "B0": {"mass": 5.2796}, "D0": {"mass": 1.8648},
        "pip": {"mass": 0.1396}, "pim": {"mass": 0.1396},
        "pi0": {"mass": 0.1350}, "pi": {"mass": 0.1396}}})
    # a/b are INTERNAL siblings: common parent D0, not the top
    m = dalitz_boundary_masses(cfg, "Rp", "Rm")
    assert m == pytest.approx((1.8648, 0.1396, 0.1350, 0.1396))
    # union {pip,pim,pi0} != finals — the internal-parent check allows it
    with pytest.raises(ValueError, match="share a parent"):
        dalitz_boundary_masses(cfg, "Rp", "pi")


def test_topo_name_from_index_roundtrip(cfg_pwa):
    """topo_name_from_index inverts topo_index_from_name (nodes -> name)."""
    tree = cfg_pwa.decay_tree
    for name in ("pipi", "pipeta", "pimeta"):
        slot = tree.topo_index_from_name(name)
        assert tree.topo_name_from_index(slot) == name
    with pytest.raises(KeyError, match="no topology slot"):
        tree.topo_name_from_index(99)


def test_topo_name_from_index_multicore():
    """Two-core topology -> list of core labels (structural order)."""
    from tabpwa.decay_tree import DecayTree
    spec = {
        "B0": [["rho1", "rho2"], ["rho3", "rho4"]],
        "rho1": ["pip1", "pim1"], "rho2": ["pip2", "pim2"],
        "rho3": ["pip1", "pim2"], "rho4": ["pip2", "pim1"],
    }
    tree = DecayTree(spec, {"$top": "B0", "$finals": [
        "pip1", "pim1", "pip2", "pim2"], **{
        p: {"J": 0} for p in
        ["B0", "rho1", "rho2", "rho3", "rho4", "pip1", "pim1", "pip2",
         "pim2"]}})
    slot = tree.topo_index_from_name(["rho1", "rho2"])
    names = tree.topo_name_from_index(slot)
    assert isinstance(names, list) and len(names) == 2
    # round-trip through the (sorted-group) canonical order
    assert tree.topo_index_from_name(names) == slot
    assert set(names) == {"rho1", "rho2"}


def test_plots_2d_display_axis_labels(cfg_pwa):
    """display "A vs B" splits into xlabel/ylabel; explicit keys win."""
    from tabpwa.amp_model import build_amplitude_model
    from tabpwa.read_var import plots_2d_from_config
    dic = dict(cfg_pwa.dic)
    dic["plot"] = dict(cfg_pwa.dic["plot"])
    dic["plot"]["2Dplot"] = {
        "p1": {"x": "m_pipi**2", "y": "m_pipeta**2",
               "display": "$A$ vs $B$"},
        "p2": {"x": "m_pipi**2", "y": "m_pipeta**2",
               "display": "$A$ vs $B$", "xlabel": "X", "ylabel": "Y"},
        "p3": {"x": "m_pipi**2", "y": "m_pipeta**2",
               "display": "no split marker"}}
    panels = dict(plots_2d_from_config(build_amplitude_model(dic)))
    assert panels["p1"]["labels"] == ["$A$", "$B$"]
    assert panels["p2"]["labels"] == ["X", "Y"]       # explicit wins
    assert panels["p3"]["labels"] == ["m_pipi**2", "m_pipeta**2"]
