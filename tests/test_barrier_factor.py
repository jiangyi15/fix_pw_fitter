"""Barrier-factor registry: class types, ``get_id()`` keys, per-decay config.

The barrier form is baked into the kernel config's ``fl_table`` (all
backends only interpolate it), so a pluggable registry + per-decay
``barrier`` / ``d`` kwargs are pure-Python.
"""
import copy

import numpy as np
import pytest

from tabpwa.bw_form_factor import (BARRIER_MODELS, BarrierFactor,
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
    from tabpwa.amp_model import build_amplitude_model
    from tabpwa.config_loader import load_config

    dic: dict = copy.deepcopy(load_config(CFG))
    dic["decay"]["jpsi"][0] = list(dic["decay"]["jpsi"][0]) + [
        {"barrier": "exp"}, {"d": 1.5}]
    return build_amplitude_model(dic)


def test_default_config_backward_compatible():
    from tabpwa.amp_model import build_amplitude_model

    m = build_amplitude_model(CFG)
    c = m.build_kernel_config()
    assert [b.name for b in m.fl_forms] == ["bw"] * len(m.fl_forms)
    assert [b.L for b in m.fl_forms] == [1, 2, 3]
    assert all(b.d == 3.0 for b in m.fl_forms)
    # barrier metadata is on the model, not the kernel config
    assert not ({"fl_forms", "fl_l", "fl_d", "fl_specs"} & set(c))
    # one row per unique L, exactly the legacy bw table
    q = c["fl_min"] + c["fl_delta"] * np.arange(c["fl_table"].shape[1])
    for i, b in enumerate(m.fl_forms):
        np.testing.assert_allclose(c["fl_table"][i], form_factor(b.L, q),
                                   rtol=0, atol=0)


def test_per_decay_barrier_and_d():
    m = _override_model()
    c = m.build_kernel_config()
    assert {b.name for b in m.fl_forms} == {"bw", "exp"}
    q = c["fl_min"] + c["fl_delta"] * np.arange(c["fl_table"].shape[1])
    for i, b in enumerate(m.fl_forms):
        np.testing.assert_allclose(c["fl_table"][i],
                                   form_factor(b.L, q, d=b.d, kind=b.name),
                                   rtol=1e-12, atol=1e-14)
    assert all(b.d == 1.5 for b in m.fl_forms if b.name == "exp")
    assert all(b.d == 3.0 for b in m.fl_forms if b.name == "bw")


def test_nested_barrier_spec_extra_params():
    from tabpwa.amp_model import build_amplitude_model
    from tabpwa.config_loader import load_config

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
        m = build_amplitude_model(dic)
        c = m.build_kernel_config()
        i = [b.name for b in m.fl_forms].index("test_gauss")
        # extra param reached the class and the exact spec is on the model
        assert m.fl_forms[i].get_params() == {
            "type": "test_gauss", "q0_ref": 1.0, "d": 1.5, "alpha": 2.0}
        q = c["fl_min"] + c["fl_delta"] * np.arange(c["fl_table"].shape[1])
        np.testing.assert_allclose(c["fl_table"][i],
                                   np.exp(-2.0 * q ** 2),
                                   rtol=1e-12, atol=1e-14)
        # a different alpha is a different id (would be another row)
        L = m.fl_forms[i].L
        assert (build_barrier("test_gauss", L=L, d=1.5, alpha=3.0)
                != build_barrier("test_gauss", L=L, d=1.5, alpha=2.0))
    finally:
        BARRIER_MODELS.pop("test_gauss", None)



# ── end-to-end: the per-decay barrier reaches the CUDA kernels ────

def test_numpy_vs_cuda_with_exp_barrier():
    from tabpwa import Fitter
    from tabpwa.config_loader import load_config
    from tabpwa.pwa_build import generate_pwa_phsp, pwa_event_data_tree

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
