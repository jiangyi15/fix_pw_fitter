"""JointFitter — simultaneous multi-config fit on ONE shared x-space.

CPU only (``numpy_pwa`` backend): the feature under test is the shared
constraint manager and the summed objective, not the kernels.

Covered:
1. two copies of one config collapse to ONE registry (no doubled slots),
   and the joint NLL/grad is exactly 2x the single fitter's;
2. a shared parameter moves BOTH models (gradient chain over the shared
   slot agrees with finite differences);
3. ``fix_var`` on a shared name: union-merge, later value wins + WARN;
4. differing bound windows on a shared name: intersected + WARN.
"""
import copy

import numpy as np
import pytest
import yaml

from tabpwa import Fitter
from tabpwa.joint_fit import JointFitter

CFG = "tests/config_pwa_bw.yml"


def _synth(fitter, n, seed):
    """Kernel-shaped synthetic data inside the gamma-table grid."""
    kc = fitter.kernel_config
    rs = np.random.RandomState(seed)
    n_at = int(np.max(kc["angle_index"])) + 1 if len(kc["angle_index"]) else 1
    nac = int(np.asarray(kc["angle_k"]).shape[-1])
    n_mass = int(np.max(kc["mass_index"])) + 1
    n_q = int(np.max(kc["fl_q_index"])) + 1
    gmin = float(kc["gamma_min"])
    gspan = float(kc["gamma_delta"]) * (np.asarray(kc["gamma_table"]).shape[-1])
    return {"mass": (gmin + 0.05) + (0.9 * gspan - 0.1) * rs.rand(n, n_mass),
            "q": np.abs(rs.randn(n, n_q)) + 0.3,
            "angle": rs.uniform(-np.pi, np.pi, size=(n, n_at * nac)),
            "weight": np.ones(n), "bkg": np.ones(n)}


def _put(fitter, n, seed):
    fitter.set_phsp(_synth(fitter, n, seed + 100))
    fitter.set_data(_synth(fitter, n, seed))


def _write(tmp_path, name, dic):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(dic, sort_keys=False))
    return str(p)


def test_joint_two_copies_share_slots_and_double():
    single = Fitter(CFG, backend="numpy_pwa")
    single.apply_constrains()
    joint = JointFitter()
    fa = joint.add_config(CFG, backend="numpy_pwa", load_data=False)
    fb = joint.add_config(CFG, backend="numpy_pwa", load_data=False)

    # one shared registry — same names, same order, no doubled slots
    # (Fitter.param_names() is the raw builder list; the constrained
    # registry is cm.var_registry.flat_names)
    assert joint.param_names == single.cm.var_registry.flat_names
    assert len(joint.param_names) < len(single.param_names())
    # mass/width transforms are de-duplicated by their OUTPUT names: the
    # second config replaces, never stacks, the shared resonance's ones
    assert (len(joint.cm.mass_width_transforms)
            == len(single.cm.mass_width_transforms))

    _put(single, 120, seed=1)
    _put(fa, 120, seed=1)
    _put(fb, 120, seed=1)

    x = joint.initial_values(seed=3)
    n_single, g_single = single.get_nll(x)
    n_joint, g_joint = joint.get_nll(x)
    assert n_joint == pytest.approx(2.0 * n_single, rel=0, abs=1e-9)
    assert np.allclose(g_joint, 2.0 * g_single, rtol=1e-12, atol=1e-12)
    # report-frame round trip through the shared pipeline
    assert np.allclose(joint.values_from_dict({"value": joint.values_dict(x)}),
                       x, atol=1e-8)


def test_joint_shared_parameter_moves_both_models():
    joint = JointFitter()
    fa = joint.add_config(CFG, backend="numpy_pwa", load_data=False)
    fb = joint.add_config(CFG, backend="numpy_pwa", load_data=False)
    _put(fa, 100, seed=11)          # different datasets
    _put(fb, 100, seed=12)

    x = joint.initial_values(seed=4)
    name = next(n for n in joint.free_param_names() if n.endswith("_mass"))
    idx = joint.param_names.index(name)

    eps = 1e-6
    xp, xm = x.copy(), x.copy()
    xp[idx] += eps
    xm[idx] -= eps
    num = (joint.get_nll(xp)[0] - joint.get_nll(xm)[0]) / (2 * eps)
    _n, g = joint.get_nll(x)
    assert num == pytest.approx(g[idx], rel=1e-3, abs=1e-4)

    # the shared slot really is shared: BOTH models feel the shift
    dA = fa.get_nll(xp)[0] - fa.get_nll(xm)[0]
    dB = fb.get_nll(xp)[0] - fb.get_nll(xm)[0]
    assert abs(dA) > 1e-12 and abs(dB) > 1e-12


def test_joint_fixed_value_union_latest_wins(tmp_path):
    dic = yaml.safe_load(open(CFG))
    a = copy.deepcopy(dic)
    b = copy.deepcopy(dic)
    a.setdefault("constrains", {})["fix_var"] = {"MI2p_mass": 1.2}
    b.setdefault("constrains", {})["fix_var"] = {"MI2p_mass": 1.4}
    pa, pb = _write(tmp_path, "a.yml", a), _write(tmp_path, "b.yml", b)

    joint = JointFitter()
    joint.add_config(pa, backend="numpy_pwa", load_data=False)
    with pytest.warns(RuntimeWarning, match="shared fixed"):
        joint.add_config(pb, backend="numpy_pwa", load_data=False)

    assert joint.cm.fixed_slots["MI2p_mass"] == 1.4      # latest wins
    assert "MI2p_mass" not in joint.free_param_names()   # union-fixed


def test_joint_bounds_intersect(tmp_path):
    dic = yaml.safe_load(open(CFG))
    a = copy.deepcopy(dic)
    b = copy.deepcopy(dic)
    a.setdefault("constrains", {})["var_range"] = {"MI2p_mass": [1.1, 1.5]}
    b.setdefault("constrains", {})["var_range"] = {"MI2p_mass": [1.3, 1.7]}
    pa, pb = _write(tmp_path, "a.yml", a), _write(tmp_path, "b.yml", b)

    joint = JointFitter()
    joint.add_config(pa, backend="numpy_pwa", load_data=False)
    with pytest.warns(RuntimeWarning, match="intersected"):
        joint.add_config(pb, backend="numpy_pwa", load_data=False)

    spec = joint.cm.bounds.to_dict()["MI2p_mass"]
    assert (spec["low"], spec["high"]) == (1.3, 1.5)


def test_joint_fit_smoke():
    joint = JointFitter()
    fa = joint.add_config(CFG, backend="numpy_pwa", load_data=False)
    fb = joint.add_config(CFG, backend="numpy_pwa", load_data=False)
    _put(fa, 60, seed=21)
    _put(fb, 60, seed=22)

    x0 = joint.initial_values(seed=5)
    n0 = joint.get_nll(x0)[0]
    res = joint.fit(x0, maxiter=3, disp=False)
    assert np.isfinite(res.fun)
    assert len(res.x) == len(joint.free_param_names())
    assert res.fun <= n0 + 1e-6


def test_joint_init_from_configs():
    """JointFitter([...]) builds every config up front."""
    joint = JointFitter([CFG, (CFG, "numpy_pwa")], backend="numpy_pwa",
                        load_data=False)
    assert joint.n_models == 2
    for f in joint.fitters:
        _put(f, 60, seed=31)
    x = joint.initial_values(seed=2)
    n, g = joint.get_nll(x)
    assert np.isfinite(n)
    assert len(g) == len(joint.free_param_names())
    assert n == pytest.approx(2.0 * joint.fitters[0].get_nll(x)[0],
                              rel=1e-12, abs=1e-9)

    # spec-dict form and the single-config (non-list) form
    j2 = JointFitter([{"config": CFG}, CFG], backend="numpy_pwa",
                     load_data=False)
    assert j2.n_models == 2
    j3 = JointFitter(CFG, backend="numpy_pwa", load_data=False)
    assert j3.n_models == 1
    assert j3.param_names == joint.param_names


def test_joint_add_config_is_single_and_returns_fitter():
    """add_config builds exactly one Fitter (no broadcast logic inside)."""
    j = JointFitter(CFG, backend="numpy_pwa", load_data=False)
    assert j.n_models == 1
    assert isinstance(j.fitters[0], Fitter)

    cfg = yaml.safe_load(open(CFG))
    f = j.add_config(cfg, backend="numpy_pwa", load_data=False,
                     label="my-variant")
    assert isinstance(f, Fitter) and j.n_models == 2
    assert f._joint_label == "my-variant"


def test_joint_variants_added_from_outside_loop():
    """Config variants stay outside: expand, then add each config."""
    from tabpwa.config_broadcast import ConfigBroadcast, scan

    cfg = yaml.safe_load(open(CFG))
    cfg["broadcast"] = scan({"purity": [1.0, 0.9]})
    joint = JointFitter()
    for lbl, variant in ConfigBroadcast.items(cfg):
        joint.add_config(variant, backend="numpy_pwa", load_data=False,
                         label=lbl)
    assert joint.n_models == 2
    assert [f.config.dic.get("purity") for f in joint.fitters] == [1.0, 0.9]
    assert joint.fitters[1]._joint_label == "1:purity=0.9"


def test_joint_config_broadcast_block(tmp_path):
    """A `broadcast:` diff list = one model, one sub-Fitter per entry."""
    from tabpwa.config_broadcast import ConfigBroadcast

    proto = Fitter(CFG, backend="numpy_pwa")

    def write_npz(name, n, seed):
        d = _synth(proto, n, seed)
        np.savez(tmp_path / name, mass=d["mass"], q=d["q"],
                 angle=d["angle"], weight=d["weight"])
        return name

    write_npz("phsp.npz", 80, 1)          # shared phase space
    write_npz("a.npz", 50, 2)             # dataset A
    write_npz("b.npz", 70, 3)             # dataset B

    cfg = yaml.safe_load(open(CFG))
    cfg["data"] = {"dat_order": ["pip", "pim", "eta"],
                   "phsp_arr": "phsp.npz"}          # relative → resolved
    cfg["broadcast"] = [
        {"data": {"data_arr": "a.npz"}},
        {"data": {"data_arr": "b.npz"}, "purity": 0.93, "label": "runB"},
    ]
    path = _write(tmp_path, "multi.yml", cfg)

    joint = JointFitter()
    for lbl, variant in ConfigBroadcast.items(path):
        joint.add_config(variant, backend="numpy_pwa", label=lbl)

    assert joint.n_models == 2
    fa, fb = joint.fitters
    assert len(fa._data_np["weight"]) == 50        # per-entry data
    assert len(fb._data_np["weight"]) == 70
    assert fa._phsp_n == fb._phsp_n == 80          # shared phsp from base
    assert fa.param_names() == fb.param_names()    # one shared x-space
    assert fa._purity == 1.0 and fb._purity == 0.93
    assert fb._joint_label == "1:runB"             # from the entry's label

    x = joint.initial_values(seed=2)
    n, g = joint.get_nll(x)
    assert len(g) == len(joint.free_param_names())
    assert n == pytest.approx(fa.get_nll(x)[0] + fb.get_nll(x)[0],
                              rel=0, abs=1e-9)


def test_joint_broadcast_dotted_diff_and_empty_error(tmp_path):
    """Dotted-key diffs work; an empty broadcast list is an error."""
    from tabpwa.config_broadcast import ConfigBroadcast

    proto = Fitter(CFG, backend="numpy_pwa")

    def write_npz(name, n, seed):
        d = _synth(proto, n, seed)
        np.savez(tmp_path / name, mass=d["mass"], q=d["q"],
                 angle=d["angle"], weight=d["weight"])
        return str(tmp_path / name)

    cfg = yaml.safe_load(open(CFG))
    cfg["data"] = {"dat_order": ["pip", "pim", "eta"]}
    cfg["broadcast"] = [
        {"data.data_arr": write_npz("a.npz", 40, 2),
         "data.phsp_arr": write_npz("pa.npz", 60, 4)},
        {"data.data_arr": write_npz("b.npz", 50, 3),
         "data.phsp_arr": write_npz("pb.npz", 70, 5)},
    ]
    joint = JointFitter()
    for lbl, variant in ConfigBroadcast.items(cfg):
        joint.add_config(variant, backend="numpy_pwa", label=lbl)
    assert joint.n_models == 2
    assert len(joint.fitters[0]._data_np["weight"]) == 40
    assert len(joint.fitters[1]._data_np["weight"]) == 50
    assert joint.fitters[1]._phsp_n == 70

    bad = yaml.safe_load(open(CFG))
    bad["broadcast"] = []
    with pytest.raises(ValueError, match="empty 'broadcast'"):
        ConfigBroadcast(bad)
