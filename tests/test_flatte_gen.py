#!/usr/bin/env python3
"""``FlatteGen`` particle model (TFPWA ``FlatteGen`` port).

The kernel denominator is ``m0² - m² - i·m0·Σ_i g_i·gamma_i(m)`` with the
gamma rows precomputed on a mass grid.  ``FlatteGen`` implements TFPWA's
``ParticleFlateGen`` gamma

    gamma_i(m) = -im_sign · (q_i/m) · (m0_ref/|q_i0|)
                 · (|q_i|/|q_i0|)^(2 l_i) · B'_{l_i}(|q_i|,|q_i0|,d)²

(the kernel provides the outer ``m0``; TFPWA's ``m_c`` contributes the
``m0_ref`` above).

Checks:
1. the model's ``gamma`` rows, fed through ``bw_dom``, reproduce a pure
   numpy transcription of ``ParticleFlateGen.get_amp`` (default, l_list,
   has_bprime=False, cut_phsp=True, no_q0=True, no_m0=True, combinations);
2. registration / defaults / gamma names, and the ``l_list`` length check;
3. end-to-end: the ``gamma_table`` rows of a real PWA config match the
   direct evaluation and a ``numpy_pwa`` NLL+gradient is finite.
"""
import copy
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tabpwa.particle_model import build_particle                 # noqa: E402
from tabpwa.particle_model.flatte_model import FlatteGenModel  # noqa: E402

M0 = 0.99
MASS_LIST = [[0.139567, 0.139567], [0.139567, 0.547862]]
GS = (0.3, 0.6)          # g_0, g_1
D = 3.0                  # Blatt-Weisskopf radius (build_particle default)
# crosses both thresholds (0.279134 and 0.687429)
GRID = np.linspace(0.15, 1.4, 61)


# ── pure-numpy transcription of TFPWA (no tabpwa helpers) ─────────

def _ref_breakup(m, m1, m2):
    """TFPWA ``cal_monentum``: real q above threshold, i·|q| below."""
    s = np.asarray(m, dtype=float) ** 2
    p2 = (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2) / (4.0 * s)
    qa = np.sqrt(np.abs(p2))
    return np.where(p2 > 0, qa, 1j * qa)


_BP_COEFF = {  # tf_pwa.formula.Bprime_polynomial
    0: [1], 1: [1, 1], 2: [1, 3, 9], 3: [1, 6, 45, 225],
    4: [1, 10, 135, 1575, 11025], 5: [1, 15, 315, 6300, 99225, 893025],
}


def _ref_bprime_poly(l, z):
    c = _BP_COEFF[int(l)]
    return sum(c[::-1][i] * z ** i for i in range(int(l) + 1))


def _ref_bprime_q2_sq(l, q2, q02, d):
    """``Bprime_q2(l, |q|², |q0|², d) ** 2`` (as used by ParticleFlateGen)."""
    return (_ref_bprime_poly(l, np.asarray(q02) * d ** 2)
            / _ref_bprime_poly(l, np.asarray(q2) * d ** 2))


def _ref_rho(m, m0, mass_list, g, im_sign, l_list, has_bprime=True,
             cut_phsp=False, no_m0=False, no_q0=False, d=D):
    """``im_sign · Σ_i i·g_i·(width factor)`` — TFPWA ``get_amp``.

    Transcribed from ``ParticleFlateGen.get_amp``: the loop builds
    ``pi·i·g_i·m_c`` (``m_c = m0/m``, or ``1/m`` with ``no_m0``), then
    multiplies the ``m0/|q0|`` normalisation (skipped with ``no_q0``, where
    also ``|q0| = 1`` in the ``2l`` power and the barrier), the barrier
    ``Bprime_q2²`` and finally the ``m < ma+mb`` cut.
    """
    m = np.asarray(m, dtype=float)
    zeros = np.zeros_like(m)
    m_c = (1.0 / m) if no_m0 else (m0 / m)
    rhos = []
    for i, (mab, l) in enumerate(zip(mass_list, l_list)):
        ma, mb = mab
        pi = _ref_breakup(m, ma, mb)
        pi0 = _ref_breakup(m0, ma, mb)
        m_rho = pi * (1j * g[i] * m_c)
        if no_q0:
            pi0 = np.ones_like(pi0) + zeros
        else:
            m_rho = m_rho * (m0 / np.abs(pi0))
        if l != 0:
            m_rho = m_rho * (np.abs(pi) / np.abs(pi0)) ** (2 * l)
        if has_bprime:
            m_rho = m_rho * _ref_bprime_q2_sq(
                l, np.abs(pi) ** 2, np.abs(pi0) ** 2, d)
        if cut_phsp:
            m_rho = np.where(m < ma + mb, 0.0 + 0.0j, m_rho)
        rhos.append(m_rho)
    return im_sign * sum(rhos)


# ── helpers ───────────────────────────────────────────────────────

def _model(case=None, model="FlatteGen", **over):
    kw = dict(mass=M0, mass_list=MASS_LIST, g_0=GS[0], g_1=GS[1])
    kw.update(case or {})
    kw.update(over)
    return build_particle("f0", model=model, **kw)


def _bw_dom(model, m, gvals=GS):
    """Kernel denominator ``m0² - m² - i·m0·Σ g_i·gamma_i`` and its width."""
    m = np.asarray(m, dtype=float)
    m0 = float(model.kwargs.get("mass", 0.775))
    total = sum(g * np.asarray(gi)
                for g, gi in zip(gvals, model.gamma(m)))
    width = -1j * m0 * total
    return m0 ** 2 - m ** 2 + width, width


CASES = [
    {},                                            # default (all l = 0)
    {"l_list": [0, 1]},                            # explicit l_list
    {"l_list": [0, 1], "has_bprime": False},       # no barrier
    {"l_list": [0, 1], "cut_phsp": True},          # zero below threshold
    {"l_list": [0, 1], "no_q0": True},             # drop m0/|q0|, q0 = 1
    {"l_list": [0, 1], "no_m0": True},             # m_c = 1/m
    {"l_list": [0, 2]},                            # D-wave channel
    {"l_list": [0, 1], "no_m0": True, "no_q0": True},
    {"l_list": [0, 1], "no_q0": True, "cut_phsp": True},
    {"l_list": [1, 1], "has_bprime": False, "no_m0": True},
]


# ── 1. reference transcription ────────────────────────────────────

@pytest.mark.parametrize("case", CASES)
def test_gamma_matches_tfpwa_transcription(case):
    model = _model(case)
    l_list = case.get("l_list", [0, 0])
    ref_rho = _ref_rho(
        GRID, M0, MASS_LIST, GS, -1,
        l_list,
        has_bprime=case.get("has_bprime", True),
        cut_phsp=case.get("cut_phsp", False),
        no_m0=case.get("no_m0", False),
        no_q0=case.get("no_q0", False),
        d=D,
    )
    # the kernel's Δs is model-independent; the width part must match the
    # TFPWA rho exactly (the kernel's outer m0 is divided out in gamma)
    dom, width = _bw_dom(model, GRID)
    np.testing.assert_allclose(width, ref_rho, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(dom, M0 ** 2 - GRID ** 2 + ref_rho,
                               rtol=1e-12, atol=1e-13)

    # every gamma row is finite and the l=0 row is unaffected by has_bprime
    for row in model.gamma(GRID):
        assert np.all(np.isfinite(row))
    if l_list[0] == 0:
        g0 = model.gamma(GRID)[0]
        alt = _model({**case, "has_bprime": False}).gamma(GRID)[0]
        np.testing.assert_allclose(g0, alt, rtol=1e-12, atol=1e-14)


def test_default_is_flatteC_times_q0_normalisation():
    """l = 0, barrier = 1: gamma = FlatteC's q/m times m0/|q0| per channel."""
    gen = _model({}).gamma(GRID)
    c = build_particle("f0", model="FlatteC", mass=M0, mass_list=MASS_LIST,
                       g_0=GS[0], g_1=GS[1]).gamma(GRID)
    for a, b, (m1, m2) in zip(gen, c, MASS_LIST):
        q0 = np.abs(_ref_breakup(M0, m1, m2))
        np.testing.assert_allclose(a, b * (M0 / q0), rtol=1e-12, atol=1e-14)



# ── 3. registration / defaults / validation ───────────────────────

def test_registration_defaults_and_names():
    m = _model({"l_list": [0, 2]})
    assert isinstance(m, FlatteGenModel)
    assert m.get_gamma_count() == 2
    assert m.get_gamma_name() == ["f0_g0", "f0_g1"]
    assert m.get_defaults() == {"f0_mass": M0, "f0_g0": GS[0], "f0_g1": GS[1]}
    assert m.get_gamma_name() == [f"f0_g{i}" for i in range(2)]

    # unset g_i defaults to 0.1, mass to 0.775 (mirrors FlatteC)
    m2 = build_particle("f0", model="FlatteGen", mass_list=MASS_LIST)
    assert m2.get_defaults() == {"f0_mass": 0.775, "f0_g0": 0.1, "f0_g1": 0.1}
    assert FlatteGenModel("f0", mass_list=MASS_LIST).im_sign == -1


def test_transform_roundtrip_keeps_options():
    """BWParamsTransform serialises model.kwargs — l_list/flags must survive."""
    from tabpwa.param_constraint import BWParamsTransform

    m = _model({"l_list": [0, 2], "no_q0": True, "cut_phsp": True})
    d = BWParamsTransform(m).to_dict()
    assert d["model_type"] == "FlatteGen"
    back = BWParamsTransform.from_dict(d).model
    assert back.l_list == [0, 2]
    assert back.no_q0 is True and back.cut_phsp is True
    assert back.has_bprime is True and back.no_m0 is False
    for a, b in zip(m.gamma(GRID), back.gamma(GRID)):
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_invalid_l_list_raises():
    with pytest.raises(ValueError, match="l_list"):
        build_particle("f0", model="FlatteGen", mass=M0,
                       mass_list=MASS_LIST, l_list=[0])
    with pytest.raises(ValueError, match="mass_list"):
        build_particle("f0", model="FlatteGen", mass=M0)


# ── 4. end-to-end through a PWA config ────────────────────────────

# Follow the repo's model-test convention: patch a GENERIC PWA config in
# memory (no dedicated per-model fixture config).
CFG = os.path.join(ROOT, "tests", "config_pwa.yml")


def _cfg_dict(l_list=(0, 2)):
    from tabpwa.config_loader import load_config
    dic = copy.deepcopy(load_config(CFG))
    p = dic["particle"]["MI2p"]
    p["model"] = "FlatteGen"
    p["mass_list"] = [list(MASS_LIST[0]), list(MASS_LIST[1])]
    p["l_list"] = list(l_list)
    return dic


def _particle_model(model, name):
    for ch in model.full_decay.chains:
        for d in ch.decays[1:]:
            if d.core.name == name:
                return d.core._model
    raise KeyError(name)


def test_gamma_table_rows_match_direct_evaluation():
    from tabpwa import Fitter

    f = Fitter(_cfg_dict(), backend="numpy_pwa")
    try:
        sp = f.kernel_config
        names = list(f.model.g0_phys_name)
        assert {"MI2p_g0", "MI2p_g1"} <= set(names)
        pm = _particle_model(f.model, "MI2p")
        assert isinstance(pm, FlatteGenModel)
        assert pm.l_list == [0, 2]

        n_bins = sp["gamma_table"].shape[-1]
        grid = sp["gamma_min"] + sp["gamma_delta"] * np.arange(n_bins)
        assert grid[0] < MASS_LIST[0][0] + MASS_LIST[0][1] < grid[-1]

        direct = pm.gamma(grid)
        for i, gname in enumerate(pm.get_gamma_name()):
            row = sp["gamma_table"][names.index(gname)]
            assert np.all(np.isfinite(row))
            np.testing.assert_allclose(row, direct[i], rtol=1e-10,
                                       atol=1e-12, err_msg=gname)

        # the two rows differ (l=0 vs l=2) and match the TFPWA formula
        assert not np.allclose(direct[0], direct[1])
        m0_ref = float(pm.kwargs.get("mass", 0.775))
        ref = _ref_rho(grid, m0_ref, pm.mass_list, (1.0, 1.0), -1,
                       pm.l_list, d=3.0)
        np.testing.assert_allclose(
            -1j * m0_ref * sum(direct), ref, rtol=1e-10, atol=1e-12)
    finally:
        f.backend.free()


def test_numpy_pwa_nll_and_gradient_finite():
    from tabpwa import Fitter
    from tabpwa.pwa_build import generate_pwa_phsp, pwa_event_data_tree

    f = Fitter(_cfg_dict(), backend="numpy_pwa")
    f.apply_constrains()
    try:
        kc, tree = f.kernel_config, f.decay_tree
        byt = {tree.topo_index[ch.topo_id()]: ch
           for _, ch in tree.partial_waves()}
        ch0 = tree.partial_waves()[0][1]
        ph = pwa_event_data_tree(
            f.model, kc, byt, generate_pwa_phsp(f.model, ch0, 400, seed=11))
        da = pwa_event_data_tree(
            f.model, kc, byt, generate_pwa_phsp(f.model, ch0, 150, seed=22))
        f.set_phsp(ph)
        f.set_data(da)

        x = f.initial_values(seed=5)
        nll, grad = f.get_nll(x)
        assert np.isfinite(nll)
        assert grad.shape == x.shape
        assert np.all(np.isfinite(grad))
    finally:
        f.backend.free()
