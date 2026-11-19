"""PWGroupPlotter rec mode: group-summed weights invariants (s > 1)."""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from tabpwa import Fitter
from tabpwa.config_loader import Config
from tabpwa.pwa_build import pwa_event_data_tree
from tabpwa.plot_pw_groups import PWGroupPlotter
from tabpwa.plot_pwa_groups import discover_pwa_groups


def _momenta(n, seed):
    rs = np.random.RandomState(seed)
    p3 = rs.uniform(-0.4, 0.4, size=(n, 3, 3))
    m = [0.13957, 0.13957, 0.54786]
    out = np.empty((n, 3, 4))
    for j in range(3):
        E = np.sqrt(m[j] ** 2 + (p3[:, j] ** 2).sum(-1))
        out[:, j] = np.concatenate([E[:, None], p3[:, j]], axis=-1)
    return out


def _rec_plotter(n_orig=80, n_phsp_rec=120, copies=3):
    """A computed plotter in rec mode: smeared fitter rows, ORIGINAL plot rows."""
    cfg = Config("tests/config_pwa.yml")
    kc = cfg.build_all_index()
    pws = list(cfg.full_decay.get_partial_waves())
    byt = {cfg.topo_index[ch.topo_id()]: ch for _, ch in pws}

    data_rec = pwa_event_data_tree(cfg, kc, byt, _momenta(n_orig, 1))
    phsp_rec = pwa_event_data_tree(cfg, kc, byt, _momenta(n_phsp_rec, 2))
    # smeared phsp rows = event-major / copy-minor
    phsp_sm = {k: np.repeat(v, copies, axis=0)
               for k, v in phsp_rec.items()}

    f = Fitter("tests/config_pwa.yml", backend="numpy_pwa")
    f.set_phsp(phsp_sm)
    f.set_data(data_rec)
    x = f.initial_values(seed=0)
    groups = discover_pwa_groups(f.model, by="chain")

    plotter = PWGroupPlotter(f, OptimizeResult(x=x), groups)
    plotter.use_rec(data_rec_np=data_rec, phsp_rec_np=phsp_rec)
    return plotter.compute(), data_rec, phsp_rec


def test_rec_group_sum_invariants():
    plotter, data_rec, phsp_rec = _rec_plotter()
    f = plotter.fitter
    x = plotter.fit_result.x
    params, _ = f.build_params(x)

    # invariant 1: Σ_e W[e] == Σ_rows phsp_w·P over the full weight sample
    _Q, _g, P_raw = f.backend.compute(params, f._phsp_holder, norm=None)
    total = float(np.sum(f._phsp_np["weight"] * P_raw))
    assert abs(float(np.sum(plotter._P_total)) - total) < 1e-8 * total

    # invariant 2: model area == purity × Σ data_rec weights
    purity = f._purity if f._purity is not None else 1.0
    area = float(np.sum(plotter._P_total)) * plotter._scale
    target = float(np.sum(data_rec["weight"])) * purity
    assert abs(area - target) < 1e-8 * max(1.0, abs(target))


def test_cut_function_runs_on_the_rec_rows():
    """With ``use_rec`` the callable sees the ORIGINAL rows, not the smeared
    ones — so a content-based cut works unchanged, and a row-count-based
    weight is rejected instead of silently mis-aligning."""
    from tabpwa.plot_pwa_groups import pwa_mass_varfun

    plotter, data_rec, phsp_rec = _rec_plotter()
    n_orig = data_rec["weight"].size
    n_rec = phsp_rec["weight"].size
    f = plotter.fitter
    assert f._phsp_np["weight"].size == 3 * n_rec      # smeared fitter rows lose

    def cut(d):
        return (d["mass"][:, 0] > 0.6).astype(float)

    smp = plotter.var_samples(pwa_mass_varfun, *plotter.resolved_extras(cut))
    assert len(smp["x_data"][0]) == n_orig
    assert len(smp["w_model"]) == n_rec
    assert len(smp["w_bkg"]) == n_rec
    assert 0 < int(np.count_nonzero(cut(data_rec))) < n_orig   # the cut bites

    # same arrays as the explicit per-sample form, evaluated on the rec dicts
    ref = plotter.var_samples(
        pwa_mass_varfun,
        data_weight_extra=cut(plotter.data_var_np),
        phsp_weight_extra=cut(plotter.phsp_var_np))
    for key in ("w_data", "w_model", "w_bkg"):
        assert np.array_equal(smp[key], ref[key]), key

    # a function written against the SMEARED row count is caught
    with pytest.raises(ValueError, match="one weight per event"):
        plotter.resolved_extras(lambda d: np.ones(3))
