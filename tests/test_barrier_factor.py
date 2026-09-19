"""Barrier-factor registry: class types, ``get_id()`` keys, per-decay config.

The barrier form is baked into the kernel config's ``fl_table`` (all
backends only interpolate it), so a pluggable registry + per-decay
``barrier`` / ``d`` kwargs are pure-Python.
"""
import copy

import numpy as np
import pytest

from ampfit.bw_form_factor import (BARRIER_MODELS, BarrierFactor,
                                   barrier_names, build_barrier, form_factor,
                                   register_barrier)

CFG = "tests/config_pwa.yml"


# ── registry / ids ────────────────────────────────────────────────

def test_default_types_and_ids():
    assert "bw" in barrier_names() and "exp" in barrier_names()
    a = build_barrier("exp", L=2, d=1.5)
    b = build_barrier("exp", L=2, d=1.5)
    c = build_barrier("exp", L=2, d=2.0)
    assert a == b and hash(a) == hash(b)      # same get_id -> dedups
    assert a != c                             # different d -> different id
    assert a.get_params() == {"type": "exp", "q0_ref": 1.0, "d": 1.5}
    assert a.get_id() == (2, (("d", 1.5), ("q0_ref", 1.0), ("type", "exp")))
    assert build_barrier("bw", L=1).get_params() == {
        "type": "bw", "q0_ref": 1.0, "d": 3.0}


def test_exp_formula_no_qL():
    q = np.array([0.0, 0.3, 0.7, 1.0])
    np.testing.assert_allclose(build_barrier("exp", L=2, d=1.5).factor(q),
                               np.exp(-0.5 * (q * 1.5) ** 2))
    # L-independent (no q^L) and scalar in -> float out
    s = build_barrier("exp", L=0, d=1.5)
    assert s.factor(0.4) == pytest.approx(float(np.exp(-0.5 * (0.4 * 1.5) ** 2)))
    assert isinstance(s.factor(0.4), float)


def test_bw_matches_legacy_form_factor():
    q = np.linspace(0.0, 3.0, 23)
    for L in (0, 1, 2, 3, 4):
        np.testing.assert_allclose(form_factor(L, q),
                                   build_barrier("bw", L=L).factor(q),
                                   rtol=0, atol=0)


def test_register_custom_type():
    @register_barrier("test_cube")
    class Cube(BarrierFactor):
        def factor(self, q):
            return np.asarray(q, dtype=float) ** 3

    try:
        assert "test_cube" in barrier_names()
        np.testing.assert_allclose(build_barrier("test_cube", L=1).factor([1., 2.]),
                                   [1., 8.])
    finally:
        BARRIER_MODELS.pop("test_cube", None)

    with pytest.raises(KeyError):
        build_barrier("does_not_exist")


# ── per-decay config ──────────────────────────────────────────────

def _override_model():
    from ampfit.amp_model import build_amplitude_model
    from ampfit.config_loader import load_config

    dic: dict = copy.deepcopy(load_config(CFG))
    dic["decay"]["jpsi"][0] = list(dic["decay"]["jpsi"][0]) + [
        {"barrier": "exp"}, {"d": 1.5}]
    return build_amplitude_model(dic)


def test_default_config_backward_compatible():
    from ampfit.amp_model import build_amplitude_model

    c = build_amplitude_model(CFG).build_kernel_config()
    assert c["fl_forms"] == ["bw"] * len(c["fl_forms"])
    assert c["fl_l"].tolist() == [1, 2, 3]
    assert (np.asarray(c["fl_d"]) == 3.0).all()
    # one row per unique L, exactly the legacy bw table
    q = c["fl_min"] + c["fl_delta"] * np.arange(c["fl_table"].shape[1])
    for i, L in enumerate(c["fl_l"]):
        np.testing.assert_allclose(c["fl_table"][i], form_factor(L, q),
                                   rtol=0, atol=0)


def test_per_decay_barrier_and_d():
    c = _override_model().build_kernel_config()
    assert set(c["fl_forms"]) == {"bw", "exp"}
    q = c["fl_min"] + c["fl_delta"] * np.arange(c["fl_table"].shape[1])
    for i, (kind, d, L) in enumerate(zip(c["fl_forms"], c["fl_d"], c["fl_l"])):
        np.testing.assert_allclose(c["fl_table"][i],
                                   form_factor(L, q, d=d, kind=kind),
                                   rtol=1e-12, atol=1e-14)
    exp_d = np.asarray(c["fl_d"])[[k == "exp" for k in c["fl_forms"]]]
    bw_d = np.asarray(c["fl_d"])[[k == "bw" for k in c["fl_forms"]]]
    assert (exp_d == 1.5).all()               # overridden radius
    assert (bw_d == 3.0).all()                # untouched sub-decays


def test_nested_barrier_spec_extra_params():
    from ampfit.amp_model import build_amplitude_model
    from ampfit.config_loader import load_config

    @register_barrier("test_gauss")
    class TestGauss(BarrierFactor):
        def __init__(self, L=0, q0_ref=1.0, d=3.0, alpha=1.0):
            super().__init__(L, q0_ref=q0_ref, d=d)
            self.alpha = float(alpha)

        def get_params(self):
            return {**super().get_params(), "alpha": self.alpha}

        def factor(self, q):
            return np.exp(-self.alpha * np.asarray(q, dtype=float) ** 2)

    dic: dict = copy.deepcopy(load_config(CFG))
    dic["decay"]["jpsi"][0] = list(dic["decay"]["jpsi"][0]) + [
        {"barrier": {"type": "test_gauss", "d": 1.5, "alpha": 2.0}}]
    try:
        c = build_amplitude_model(dic).build_kernel_config()
        i = c["fl_forms"].index("test_gauss")
        # extra param reached the class and the exact spec is recorded
        assert c["fl_specs"][i] == {
            "type": "test_gauss", "q0_ref": 1.0, "d": 1.5, "alpha": 2.0}
        q = c["fl_min"] + c["fl_delta"] * np.arange(c["fl_table"].shape[1])
        np.testing.assert_allclose(c["fl_table"][i],
                                   np.exp(-2.0 * q ** 2),
                                   rtol=1e-12, atol=1e-14)
        # a different alpha is a different id (would be another row)
        assert (build_barrier("test_gauss", L=int(c["fl_l"][i]),
                              d=1.5, alpha=3.0)
                != build_barrier("test_gauss", L=int(c["fl_l"][i]),
                                 d=1.5, alpha=2.0))
    finally:
        BARRIER_MODELS.pop("test_gauss", None)


# ── end-to-end: the per-decay barrier reaches the CUDA kernels ────

def test_numpy_vs_cuda_with_exp_barrier():
    from ampfit import Fitter
    from ampfit.config_loader import load_config
    from ampfit.pwa_build import generate_pwa_phsp, pwa_event_data_tree

    dic: dict = copy.deepcopy(load_config(CFG))
    dic["decay"]["jpsi"][0] = list(dic["decay"]["jpsi"][0]) + [
        {"barrier": "exp"}, {"d": 1.5}]

    f = Fitter(dic, backend="numpy_pwa")
    f.apply_constrains()
    kc, tree = f.kernel_config, f.decay_tree
    byt = {tree.topo_index[ch.topo_id()]: ch for _, ch in tree.partial_waves()}
    ch0 = tree.partial_waves()[0][1]
    ph = pwa_event_data_tree(f.model, kc, byt,
                             generate_pwa_phsp(f.model, ch0, 600, seed=11))
    da = pwa_event_data_tree(f.model, kc, byt,
                             generate_pwa_phsp(f.model, ch0, 250, seed=22))
    f.set_phsp(ph)
    f.set_data(da)
    params, _ = f.build_params(f.initial_values(seed=1))
    Qn, gn, Pn = f.backend.compute(params, f._phsp_holder, norm=None,
                                   return_p=True)
    f.free()

    try:
        g = Fitter(dic, backend="cuda_v4_pwa")
    except Exception as e:                        # no CUDA / plugin
        pytest.skip(f"cuda_v4_pwa unavailable: {e}")
    g.apply_constrains()
    g.set_phsp(ph)
    g.set_data(da)
    Qc, gc, Pc = g.backend.compute(params, g._phsp_holder, norm=None,
                                   return_p=True)
    g.free()

    assert Qn == pytest.approx(Qc, rel=1e-5)
    np.testing.assert_allclose(np.asarray(Pn), np.asarray(Pc),
                               rtol=1e-5, atol=1e-6)
    for key in ("ck", "m0", "g0"):
        np.testing.assert_allclose(np.asarray(gn[key]), np.asarray(gc[key]),
                                   rtol=1e-4, atol=1e-5, err_msg=key)
