"""PWGroupPlotter rec mode: group-summed weights invariants (s > 1)."""

import numpy as np
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


def test_rec_group_sum_invariants():
    cfg = Config("tests/config_pwa.yml")
    kc = cfg.build_all_index()
    pws = list(cfg.full_decay.get_partial_waves())
    byt = {cfg.topo_index[ch.topo_id()]: ch for _, ch in pws}

    n_orig, copies = 80, 3
    data_rec = pwa_event_data_tree(cfg, kc, byt, _momenta(n_orig, 1))
    phsp_rec = pwa_event_data_tree(cfg, kc, byt, _momenta(120, 2))
    # smeared phsp rows = event-major / copy-minor
    phsp_sm = {k: np.repeat(v, copies, axis=0)
               for k, v in phsp_rec.items()}

    f = Fitter("tests/config_pwa.yml", backend="numpy_pwa")
    f.set_phsp(phsp_sm)
    f.set_data(data_rec)
    x = f.initial_values(seed=0)
    params, _ = f.build_params(x)
    groups = discover_pwa_groups(f.model, by="chain")

    plotter = PWGroupPlotter(f, OptimizeResult(x=x), groups)
    plotter.use_rec(data_rec_np=data_rec, phsp_rec_np=phsp_rec)
    plotter.compute()

    # invariant 1: Σ_e W[e] == Σ_rows phsp_w·P over the full weight sample
    _Q, _g, P_raw = f.backend.compute(params, f._phsp_holder, norm=None)
    total = float(np.sum(f._phsp_np["weight"] * P_raw))
    assert abs(float(np.sum(plotter._P_total)) - total) < 1e-8 * total

    # invariant 2: model area == purity × Σ data_rec weights
    purity = f._purity if f._purity is not None else 1.0
    area = float(np.sum(plotter._P_total)) * plotter._scale
    target = float(np.sum(data_rec["weight"])) * purity
    assert abs(area - target) < 1e-8 * max(1.0, abs(target))
