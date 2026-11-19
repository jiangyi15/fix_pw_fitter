"""PWStackPlotter — merge plot data across datasets, then draw."""
import matplotlib
matplotlib.use("Agg")                     # headless

import numpy as np
import pytest
import yaml

from tabpwa import Fitter
from tabpwa.joint_fit import JointFitter
from tabpwa.plot_pw_stack import PWStackPlotter

CFG = "tests/config_pwa_bw.yml"


def _synth(fitter, n, seed):
    kc = fitter.kernel_config
    rs = np.random.RandomState(seed)
    n_at = int(np.max(kc["angle_index"])) + 1 if len(kc["angle_index"]) else 1
    nac = int(np.asarray(kc["angle_k"]).shape[-1])
    n_mass = int(np.max(kc["mass_index"])) + 1
    n_q = int(np.max(kc["fl_q_index"])) + 1
    gmin = float(kc["gamma_min"])
    gspan = float(kc["gamma_delta"]) * np.asarray(kc["gamma_table"]).shape[-1]
    return {"mass": (gmin + 0.05) + (0.9 * gspan - 0.1) * rs.rand(n, n_mass),
            "q": np.abs(rs.randn(n, n_q)) + 0.3,
            "angle": rs.uniform(-np.pi, np.pi, size=(n, n_at * nac)),
            "weight": np.ones(n), "bkg": np.ones(n)}


def _joint_two():
    joint = JointFitter()
    fa = joint.add_config(CFG, backend="numpy_pwa", load_data=False)
    fb = joint.add_config(CFG, backend="numpy_pwa", load_data=False)
    for f, seed in ((fa, 11), (fb, 12)):
        f.set_phsp(_synth(f, 120, seed + 100))
        f.set_data(_synth(f, 80, seed))
    return joint, fa, fb


VARFUN = lambda x: [x["mass"][:, 0]]      # noqa: E731 — one variable


def test_merged_samples_are_the_sum_of_the_parts():
    joint, fa, fb = _joint_two()
    x = joint.initial_values(seed=5)
    sp = PWStackPlotter([(fa, x, {}), (fb, x, {})], labels=["A", "B"]).compute()

    merged = sp.merged_samples(VARFUN)
    per = [p.var_samples(VARFUN) for p in sp.plotters]
    assert merged["meta"]["n_var"] == 1
    assert (len(merged["x_data"][0]) == len(per[0]["x_data"][0])
            + len(per[1]["x_data"][0]))

    # total weight sums add up per component
    for key in ("w_data", "w_model", "w_bkg"):
        assert (float(np.sum(merged[key])) == pytest.approx(
            sum(float(np.sum(s[key])) for s in per), rel=0, abs=1e-9))

    # merged histogram == sum of the per-dataset histograms (same bins)
    bins = np.linspace(0.4, 1.6, 31)
    for xk, wk in (("x_data", "w_data"), ("x_model", "w_model")):
        h_all = np.histogram(merged[xk][0], bins=bins,
                             weights=merged[wk])[0]
        h_sum = sum(np.histogram(s[xk][0], bins=bins, weights=s[wk])[0]
                    for s in per)
        assert np.allclose(h_all, h_sum, rtol=0, atol=1e-12)

    # one dataset → the merged samples are exactly that dataset's
    sp1 = PWStackPlotter([(fa, x, {})], labels=["A"]).compute()
    m1 = sp1.merged_samples(VARFUN)
    assert np.allclose(m1["w_data"], sp.plotters[0].var_samples(VARFUN)["w_data"])


def test_var_samples_accepts_per_event_phsp_weight_extra():
    """Regression: `phsp_weight_extra` as a per-event array must work in
    var_samples (documented input; crashed as a scalar cast after the
    plot_var refactor)."""
    joint, fa, fb = _joint_two()
    x = joint.initial_values(seed=9)
    extra = np.ones(len(fa._phsp_np["weight"]))

    sp = PWStackPlotter([(fa, x, {})], labels=["A"]).compute()
    h = sp.plotters[0]
    smp = h.var_samples(VARFUN, phsp_weight_extra=extra)   # must not raise
    plain = h.var_samples(VARFUN)

    assert all(np.array_equal(a, b) for a, b
               in zip(smp["x_data"], plain["x_data"]))
    assert all(np.array_equal(a, b) for a, b
               in zip(smp["x_model"], plain["x_model"]))
    assert np.allclose(smp["w_data"], plain["w_data"])
    assert np.allclose(smp["w_model"], plain["w_model"])   # extra ≡ ones
    assert len(smp["w_model"]) == len(plain["w_model"])


def test_stack_plotter_draws_stacked_and_overlay(tmp_path):
    joint, fa, fb = _joint_two()
    x = joint.initial_values(seed=6)
    sp = PWStackPlotter([(fa, x, {}), (fb, x, {})], labels=["A", "B"],
                        output=str(tmp_path))
    p1 = sp.plot_var(VARFUN, ["m(pipi)"], 0.4, 1.6, 0.04, "mpipi",
                     stacked=True, legend=True)
    p2 = sp.plot_var(VARFUN, ["m(pipi)"], 0.4, 1.6, 0.04, "mpipi_overlay",
                     stacked=False, legend=True)
    import os
    assert p1 == str(tmp_path / "mpipi.png") and os.path.exists(p1)
    assert p2 == str(tmp_path / "mpipi_overlay.png") and os.path.exists(p2)

    # pull variant smoke (χ² title path)
    sp.plot_var(VARFUN, ["m(pipi)"], 0.4, 1.6, 0.04, "mpipi_pull",
                stacked=True, show_pull=True)
    assert os.path.exists(str(tmp_path / "mpipi_pull.png"))


def test_stack_plotter_group_intersection_warns():
    joint, fa, fb = _joint_two()
    x = joint.initial_values(seed=7)
    # group present in only one dataset → dropped with a warning
    groups_a = {joint.fitters[0].all_comb[0][0]: [0]}
    sp = PWStackPlotter([(fa, x, groups_a), (fb, x, {})]).compute()
    with pytest.warns(RuntimeWarning, match="not present in every dataset"):
        sp.merged_samples(VARFUN)


@pytest.mark.parametrize("stacked", [True, False])
def test_stack_plotter_forwards_axis_style(tmp_path, monkeypatch, stacked):
    """yscale / upper_ylim reach the axes in both drawing modes."""
    import matplotlib.pyplot as plt
    import tabpwa.plot_pw_draw as draw_mod

    figs = []

    def _capture(fig, output, prefix, fmt="png"):
        import os
        os.makedirs(output, exist_ok=True)
        path = os.path.join(output, prefix + "." + fmt)
        fig.savefig(path)
        figs.append(fig)
        return path

    monkeypatch.setattr(draw_mod, "save_figure", _capture)

    joint, fa, fb = _joint_two()
    x = joint.initial_values(seed=8)
    sp = PWStackPlotter([(fa, x, {}), (fb, x, {})], labels=["A", "B"],
                        output=str(tmp_path))
    try:
        sp.plot_var(VARFUN, ["m(pipi)"], 0.4, 1.6, 0.04, "axis_style",
                    stacked=stacked, yscale="log", upper_ylim=110.0)
        ax = figs[-1].axes[0]
        assert ax.get_yscale() == "log"
        assert ax.get_ylim()[1] == pytest.approx(110.0)
        assert ax.get_ylim()[0] > 0
    finally:
        for fig in figs:
            plt.close(fig)


def test_stack_plotter_applies_one_cut_to_every_dataset(tmp_path):
    """`cut_function` on the stack: resolved PER DATASET, data and phsp."""
    joint, fa, fb = _joint_two()
    x = joint.initial_values(seed=8)
    sp = PWStackPlotter([(fa, x, {}), (fb, x, {})],
                        labels=["A", "B"]).compute()

    cut = lambda d: (d["mass"][:, 0] > d["mass"][:, 0].mean()).astype(float)
    merged = sp.merged_samples(VARFUN, cut_function=cut)
    parts = [pl.var_samples(VARFUN, *pl.resolved_extras(cut))
             for pl in sp.plotters]
    for key in ("w_data", "w_model", "w_bkg"):
        want = np.concatenate([s[key] for s in parts])
        assert np.array_equal(merged[key], want), key
    # the cut really bit, and differs per dataset
    assert not np.array_equal(parts[0]["w_data"], 
                              sp.plotters[0].var_samples(VARFUN)["w_data"])
    assert np.sum(parts[0]["w_data"]) > 0

    # stacked and overlaid both draw with the cut
    for stacked in (True, False):
        path = tmp_path / f"cut_{stacked}.png"
        sp.plot_var(VARFUN, ["m"], 0.2, 3.0, 0.1, f"cut_{stacked}",
                    stacked=stacked, cut_function=cut, output=str(tmp_path))
        assert path.exists()
