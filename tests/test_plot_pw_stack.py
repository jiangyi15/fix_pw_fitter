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
