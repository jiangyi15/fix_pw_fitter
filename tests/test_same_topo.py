"""SameTopoJointFitter (broadcast + decay check) and load_fit dispatch."""
import copy

import numpy as np
import pytest
import yaml

from tabpwa import Fitter, load_fit
from tabpwa.joint_fit import JointFitter, SameTopoJointFitter

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


def _put(fitter, n, seed):
    fitter.set_phsp(_synth(fitter, n, seed + 100))
    fitter.set_data(_synth(fitter, n, seed))


def _template(**extra):
    cfg = yaml.safe_load(open(CFG))
    cfg = copy.deepcopy(cfg)
    cfg.update(extra)
    return cfg


def test_same_topo_accepts_same_decay_and_reports():
    cfg = _template(broadcast=[{"purity": 1.0, "label": "run1"},
                               {"purity": 0.9, "label": "run2"}])
    joint = SameTopoJointFitter(cfg, backend="numpy_pwa", load_data=False)
    assert isinstance(joint, JointFitter)
    assert joint.n_models == 2
    assert joint.dataset_labels() == ["0:run1", "1:run2"]
    assert joint.wave_report() == {"0:run1": len(joint.fitters[0].all_comb),
                                   "1:run2": len(joint.fitters[1].all_comb)}

    for f, seed in zip(joint.fitters, (1, 2)):
        _put(f, 60, seed)
    x = joint.initial_values(seed=3)
    total, _g = joint.get_nll(x)
    report = joint.nll_report()
    assert list(report) == joint.dataset_labels()
    assert sum(report.values()) == pytest.approx(total, rel=0, abs=1e-9)
    assert joint._replacements == []   # identical configs: no replacements


def test_load_fit_forwards_cm():
    from tabpwa.param_constraint import ConstraintManager

    shared = ConstraintManager([])
    plain = load_fit(CFG, backend="numpy_pwa", load_data=False, cm=shared)
    assert plain.cm is shared

    cfg = _template(broadcast=[{"purity": 1.0}, {"purity": 0.9}])
    joint = load_fit(cfg, backend="numpy_pwa", load_data=False, cm=shared)
    assert isinstance(joint, SameTopoJointFitter)
    assert joint.cm is shared
    assert joint._replacements == []


def test_same_topo_rejects_different_decay():
    # default forbidden=("decay",): the template is rejected by
    # ConfigBroadcast itself (the broadcast must not touch decay)
    cfg = _template(broadcast=[{"purity": 1.0},
                               {"decay": {"jpsi": [["pip", "pim"]]}}])
    with pytest.raises(ValueError, match="forbidden key"):
        SameTopoJointFitter(cfg, backend="numpy_pwa", load_data=False)

    # forbidden=() disables that check: the SameTopo safety net
    # (deep decay equality) fires instead
    cfg2 = _template(
        forbidden=(),
        broadcast=[{"purity": 1.0},
                   {"decay": {"jpsi": [["pip", "pim"]]}}])
    with pytest.raises(ValueError, match="different 'decay' section"):
        SameTopoJointFitter(cfg2, backend="numpy_pwa", load_data=False,
                            forbidden=())


def test_same_topo_single_config_ok():
    joint = SameTopoJointFitter(CFG, backend="numpy_pwa", load_data=False)
    assert joint.n_models == 1
    assert joint.dataset_labels() == ["0:config_pwa_bw.yml"]


def test_load_fit_dispatch():
    plain = load_fit(CFG, backend="numpy_pwa", load_data=False)
    assert isinstance(plain, Fitter)
    assert len(plain.cm.fixed_slots) > 0          # apply_constrains ran

    cfg = _template(broadcast=[{"purity": 1.0}, {"purity": 0.9}])
    joint = load_fit(cfg, backend="numpy_pwa", load_data=False)
    assert isinstance(joint, SameTopoJointFitter)
    assert joint.n_models == 2
    assert joint.fitters[1].config.dic.get("purity") == 0.9
