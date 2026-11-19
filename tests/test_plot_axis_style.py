"""Axis style (``yscale`` / ``xscale`` / ``upper_ylim``) of the 1-D plots.

The config ``plot:`` section has always *parsed* ``yscale`` and
``upper_ylim`` (``read_var.BaseVar.apply_plot_entry``) without any consumer,
so the keys were silently dropped.  These tests pin the whole path:
drawing primitive → plotter → config item / config panel → axes.
"""

import copy
import os

import matplotlib
matplotlib.use("Agg")                     # headless, before pyplot

import numpy as np
import pytest

import tabpwa.plot_pw_draw as draw_mod
from tabpwa.plot_pw_draw import (draw_overlay_figure, draw_var_figure,
                                 panel_ylimits, per_variable)
from tabpwa.plot_pw_groups import PWGroupPlotter

N_MASS = 3          # config_pwa topologies: pipi, pipeta, pimeta
VARFUN = lambda d: [d["mass"][:, i] for i in range(d["mass"].shape[1])]  # noqa: E731


# ── synthetic plot data (no fitter needed) ─────────────────────────────

class _StubFitter:
    """The three attributes PWGroupPlotter.var_samples reads."""

    def __init__(self, data, phsp, purity=1.0):
        self._data_np = data
        self._phsp_np = phsp
        self._purity = purity


def _events(n, n_var, rs):
    """A falling mass spectrum + the kernel columns the plot vars read."""
    cols = [np.clip(0.3 + rs.exponential(0.45, n), 0.3, 3.0)
            for _ in range(n_var)]
    return {"mass": np.column_stack(cols),
            "angle": rs.uniform(-np.pi, np.pi, size=(n, n_var, 4)),
            "q": np.ones((n, n_var)), "weight": np.ones(n),
            "bkg": np.ones(n)}


def stub_plotter(n_data=4000, n_phsp=12000, n_var=1, seed=1, purity=0.9):
    """A computed PWGroupPlotter over synthetic events (no backend)."""
    rs = np.random.RandomState(seed)
    data = _events(n_data, n_var, rs)
    phsp = _events(n_phsp, n_var, rs)
    pl = PWGroupPlotter(_StubFitter(data, phsp, purity), None, {"all": [0]})
    # what compute() would have produced: one unit-weight group, a bkg
    # of `purity`-complement on the phsp rows
    pl._P_total = np.full(n_phsp, purity * n_data / n_phsp)
    pl._P_groups = [pl._P_total.copy()]
    pl._scale = 1.0
    pl._bkg_norm = float(n_phsp)
    pl._bkg_var = np.ones(n_phsp)
    pl._zorders = [5]
    return pl


@pytest.fixture
def captured(monkeypatch):
    """Capture figures instead of closing them in save_figure."""
    figs = []

    def _capture(fig, output, prefix, fmt="png"):
        os.makedirs(output, exist_ok=True)
        path = os.path.join(output, prefix + "." + fmt)
        fig.savefig(path)
        figs.append(fig)
        return path

    monkeypatch.setattr(draw_mod, "save_figure", _capture)
    yield figs
    import matplotlib.pyplot as plt
    for fig in figs:
        plt.close(fig)


def _draw(pl, tmp_path, prefix, **kw):
    """plot_var with one sub-variable per mass column of the stub."""
    n_var = pl.fitter._data_np["mass"].shape[1]
    return pl.plot_var(VARFUN, [f"m{i}" for i in range(n_var)],
                       0.3, 3.0, 0.05, prefix, output=str(tmp_path),
                       ranges=[(0.3, 3.0)] * n_var, **kw)


# ── the drawing primitives ─────────────────────────────────────────────

def test_per_variable_broadcast_and_validation():
    assert per_variable(None, 2) == [None, None]
    assert per_variable("log", 3, "yscale") == ["log"] * 3
    assert per_variable(50.0, 2, "upper_ylim") == [50.0, 50.0]
    assert per_variable(["log", None], 2, "yscale") == ["log", None]
    with pytest.raises(ValueError, match="yscale has 3 entries but this "
                                         "figure has 2 sub-variable"):
        per_variable(["log", "log", "log"], 2, "yscale")


def test_panel_ylimits_linear_vs_log():
    vals = [np.array([0.0, 4.0, 900.0])]
    assert panel_ylimits(vals, None, log=False) == (0.0, None)
    assert panel_ylimits(vals, 120.0, log=False) == (0.0, 120.0)
    lo, hi = panel_ylimits(vals, None, log=True)
    assert lo > 0 and lo < 4.0          # floored below the smallest +ve bin
    assert hi > 900.0                    # explicit top (log autoscale is
    #                                      unusable with zero-height bins)
    lo, hi = panel_ylimits(vals, 120.0, log=True)
    assert hi == 120.0 and 0 < lo < 120.0
    # a single tiny bin must not flatten the panel to 6 decades
    lo, _ = panel_ylimits([np.array([1e-9, 1000.0])], None, log=True)
    assert lo == pytest.approx(1000.0 * 1e-4)
    # no positive content at all still gives a drawable range
    lo, hi = panel_ylimits([np.zeros(3)], None, log=True)
    assert 0 < lo < hi


# ── plot_var (the user-facing entry) ───────────────────────────────────

def test_plot_var_log_y_keeps_pull_linear(tmp_path, captured):
    pl = stub_plotter()
    path = _draw(pl, tmp_path, "log", yscale="log", show_pull=True,
                 legend=True)
    assert os.path.exists(path)
    main, pull = captured[-1].axes[0], captured[-1].axes[1]
    assert main.get_yscale() == "log"
    assert pull.get_yscale() == "linear"             # pull never goes log
    assert main.get_ylim()[0] > 0                    # no (0, None) on log
    assert main.get_xlim() == (0.3, 3.0)
    assert pull.get_ylim()[0] < 0 < pull.get_ylim()[1]


@pytest.mark.parametrize("yscale", [None, "log"])
def test_upper_ylim_caps_the_top(tmp_path, captured, yscale):
    pl = stub_plotter()
    kw = {} if yscale is None else {"yscale": yscale}
    path = _draw(pl, tmp_path, f"cap_{yscale}", upper_ylim=120.0, **kw)
    assert os.path.exists(path)
    main = captured[-1].axes[0]
    assert main.get_ylim()[1] == pytest.approx(120.0)
    if yscale is None:
        assert main.get_ylim()[0] == 0.0             # linear floor unchanged
    else:
        assert main.get_yscale() == "log" and main.get_ylim()[0] > 0


def test_scale_free_call_keeps_zero_floor(tmp_path, captured):
    """Back-compat: no axis style ⇒ exactly today's (0, None)."""
    pl = stub_plotter()
    path = _draw(pl, tmp_path, "plain")
    assert os.path.exists(path)
    main = captured[-1].axes[0]
    assert main.get_yscale() == "linear"
    assert main.get_ylim()[0] == 0.0


def test_per_sub_variable_scales_and_limits(tmp_path, captured):
    pl = stub_plotter(n_var=2)
    path = _draw(pl, tmp_path, "mixed", yscale=["log", "linear"],
                 upper_ylim=[50.0, None])
    assert os.path.exists(path)
    a, b = captured[-1].axes[0], captured[-1].axes[1]
    assert a.get_yscale() == "log" and b.get_yscale() == "linear"
    assert a.get_ylim()[1] == pytest.approx(50.0)
    assert b.get_ylim()[0] == 0.0                    # unset entry ⇒ default


def test_wrong_length_sequence_raises(tmp_path):
    pl = stub_plotter(n_var=2)
    with pytest.raises(ValueError, match="yscale has 3 entries"):
        _draw(pl, tmp_path, "bad", yscale=["log", "log", "log"])


def test_overlay_figure_honours_axis_style(tmp_path, captured):
    pl = stub_plotter()
    smp = pl.var_samples(VARFUN)
    path = draw_overlay_figure([smp, smp], ["A", "B"], 0.3, 3.0, 0.05,
                               "ov", output=str(tmp_path), yscale="log",
                               upper_ylim=90.0)
    assert os.path.exists(path)
    ax = captured[-1].axes[0]
    assert ax.get_yscale() == "log"
    assert ax.get_ylim()[1] == pytest.approx(90.0)
    assert ax.get_ylim()[0] > 0
    with pytest.raises(ValueError, match="upper_ylim has 2 entries"):
        draw_overlay_figure([smp, smp], ["A", "B"], 0.3, 3.0, 0.05, "ov2",
                            output=str(tmp_path), upper_ylim=[1.0, 2.0])


def test_draw_var_figure_passes_through(tmp_path, captured):
    pl = stub_plotter()
    smp = pl.var_samples(VARFUN)
    path = draw_var_figure(smp, ["m"], 0.3, 3.0, 0.05, "dv",
                           output=str(tmp_path), show_pull=True,
                           yscale="log")
    assert os.path.exists(path)
    assert captured[-1].axes[0].get_yscale() == "log"
    assert captured[-1].axes[1].get_yscale() == "linear"


# ── the config path (the actual regression) ────────────────────────────

@pytest.fixture(scope="module")
def cfg_plot():
    """config_pwa with plot.config.yscale/upper_ylim + one per-var cap."""
    from tabpwa.amp_model import build_amplitude_model

    dic = copy.deepcopy(build_amplitude_model("tests/config_pwa.yml").dic)
    plot = dic["plot"]
    plot["config"] = {**plot.get("config", {}), "yscale": "log"}
    plot["mass"]["pipi"] = {**plot["mass"]["pipi"], "upper_ylim": 900.0}
    return build_amplitude_model(dic)


@pytest.fixture(scope="module")
def plot_events():
    return _events(200, N_MASS, np.random.RandomState(7))


def test_config_plot_items_carry_axis_style(cfg_plot, plot_events):
    from tabpwa.plot_pwa_groups import config_plot_items

    items = {it["key"]: it for it in
             config_plot_items(cfg_plot, plot_events, plot_events)}
    assert items["pipi"]["yscale"] == "log"          # plot.config default
    assert items["pipi"]["upper_ylim"] == 900.0      # per-variable entry
    assert items["pipeta"]["yscale"] == "log"
    assert items["pipeta"]["upper_ylim"] is None


def test_config_plot_item_reaches_the_axes(cfg_plot, plot_events, tmp_path,
                                           captured):
    """One figure per item, exactly as scripts/plot_pwa_groups.py draws it."""
    from tabpwa.plot_pwa_groups import config_plot_items

    items = {it["key"]: it for it in
             config_plot_items(cfg_plot, plot_events, plot_events)}
    it = items["pipi"]
    lo, hi = it["range"]
    pl = stub_plotter()
    pl.plot_var(it["varfun"], [it["label"]], lo, hi, it["width"], it["stem"],
                output=str(tmp_path), ranges=[it["range"]], show_pull=True,
                legend=it["legend"], yscale=it["yscale"],
                upper_ylim=it["upper_ylim"])
    main = captured[-1].axes[0]
    assert main.get_yscale() == "log"
    assert main.get_ylim()[1] == pytest.approx(900.0)


def test_config_panels_forward_per_variable_lists(cfg_plot, plot_events,
                                                  tmp_path, captured):
    from tabpwa.plot_pwa_groups import config_panels

    p = config_panels(cfg_plot, plot_events, plot_events)
    assert p["mass"]["keys"] == ["pipi", "pipeta", "pimeta"]
    assert p["mass"]["yscale"] == ["log", "log", "log"]
    assert p["mass"]["upper_ylim"] == [900.0, None, None]

    # the list is what plot_var receives for a multi-variable figure
    pl = stub_plotter(n_var=N_MASS)
    pl.plot_var(p["mass"]["varfun"], p["mass"]["labels"],
                p["mass"]["ranges"][0][0], p["mass"]["ranges"][0][1],
                p["mass"]["width"], "panel", output=str(tmp_path),
                ranges=p["mass"]["ranges"], yscale=p["mass"]["yscale"],
                upper_ylim=p["mass"]["upper_ylim"])
    axes = captured[-1].axes
    assert [ax.get_yscale() for ax in axes] == ["log"] * N_MASS
    assert axes[0].get_ylim()[1] == pytest.approx(900.0)
    assert axes[1].get_ylim()[1] > 0                 # auto top, still log
