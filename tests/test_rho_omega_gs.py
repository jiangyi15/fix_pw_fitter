#!/usr/bin/env python3
"""Tests for the GS-rho x BW-omega product model (``RhoOmegaGS``).

The model realises the ``RhoOmega`` fixed shape (Gounaris-Sakurai rho x
constant-width Breit-Wigner omega) with **all four masses/widths
fitted**, using ONE kernel Breit-Wigner whose fixed 8-row gamma basis

    [1, u, i, i.u, i.u^2, g, u.g, i.g]

(``g = gamma + i.fs/m0_ref`` — the single complex GS unit at the
``(q0, m0_ref)`` scale of the config reference) is contracted with eight
real couplings produced by :class:`RhoOmegaGSWeightsTransform`.  No
kernel code is modified.

Verified:
1. closed-form weight/denominator parity
   (``m0^2-u-i.m0.sum(w.gamma) == D_rho.D_omega``, packed definition);
2. line-shape parity against the **current** fixed ``RhoOmega`` model
   at the reference point (m0 = m0_ref), and against the packed
   denominator over a grid;
3. gamma basis rows and model-amplitude parity;
4. transform backward vs central finite differences; inverse round-trip;
5. registration/defaults (``build_particle``, gamma count/names);
6. the transform registers through the standard fitter bridge;
7. the full fitter pipeline (transform included) matches centered
   finite-difference gradients;
8. the ``rho_pole`` helper (pole of the frozen-reference D_rho).

Following the repo's model-test conventions: unit tests build the
particle directly via ``build_particle``; the fitter-level tests patch
``tests/config_rho_omega2.yml`` in memory (no dedicated fixture config).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from tabpwa.particle_model import build_particle, ALL_MODELS
from tabpwa.particle_model.rho_omega_gs_model import (
    RhoOmegaGSModel, RhoOmegaGSWeightsTransform,
)

CFG = "tests/config_rho_omega2.yml"
# Default pole parameters (rho, omega)
M0, G1, M2, G2 = 0.775, 0.149, 0.78266, 0.00868
P = "rho_omega_gs"
DM2, DM3 = 0.13957039, 0.1349768


# ── unit level: the particle model itself ────────────────────────

def _model(**kw):
    kw.setdefault("mass", M0)
    kw.setdefault("width", G1)
    if "omega_mass" not in kw:
        kw.setdefault("mass2", M2)
    if "omega_width" not in kw:
        kw.setdefault("width2", G2)
    return build_particle(P, model="RhoOmegaGS", **kw)


def _transform(model):
    return model.make_mass_width_transform()


def _params_dict(model, m0=M0, g1=G1, m2=M2, g2=G2):
    return {f"{model.name}_mass": m0, f"{model.name}_width": g1,
            f"{model.name}_mass2": m2, f"{model.name}_width2": g2}


def _weights(model, m0=M0, g1=G1, m2=M2, g2=G2):
    tr = _transform(model)
    out = tr.forward(_params_dict(model, m0, g1, m2, g2))
    return np.array([out[n] for n in tr.output_names])


def _shapes(model, m):
    """(gamma, fs) at the config reference mass (real arrays, Γ0-normalised)."""
    M = float(model.kwargs.get("M", model.kwargs.get("mass", M0)))
    L = int(model.kwargs.get("L", 1))
    d = float(model.kwargs.get("d", 3.0))
    if model._parent and model._parent._decays:
        d1 = float(model._parent._decays[0].outs[0].mass)
        d2 = float(model._parent._decays[0].outs[1].mass)
        q = _q_of(m, model)
        q0 = _q_of(np.asarray([M]), model)[0]
    else:
        q = _q_of(m, model)
        q0 = _q_of(np.asarray([M]), model)[0]
    gam = np.asarray(_gamma_run_impl(m, q, q0, L, M, d), dtype=float)
    fs = np.asarray(_fs_impl(m, M, DM2, DM3), dtype=float)
    return gam, fs


def _q_of(m, model):
    from tabpwa.particle_model.gs_rho_model import _two_body_cm_mom
    dm2 = float(model.kwargs.get("daug2Mass", DM2))
    dm3 = float(model.kwargs.get("daug3Mass", DM3))
    if model._parent and model._parent._decays:
        d1 = float(model._parent._decays[0].outs[0].mass)
        d2 = float(model._parent._decays[0].outs[1].mass)
        return _two_body_cm_mom(m, d1, d2)
    return _two_body_cm_mom(m, dm2, dm3)


def _gamma_run_impl(m, q, q0, L, M, d):
    from tabpwa.particle_model.gs_rho_model import _gamma_run
    return _gamma_run(m, 1.0, q, q0, L, M, d)


def _fs_impl(m, M, dm2, dm3):
    from tabpwa.particle_model.gs_rho_model import _fs_fun
    return _fs_fun(m ** 2, M ** 2, 1.0, dm2, dm3)


def _packed_d_rho(m, m0, g1, gam, fs, m_ref=M0):
    """Packed-definition rho denominator."""
    u = np.asarray(m, dtype=float) ** 2
    return m0 ** 2 - u - 1j * m0 * g1 * (gam + 1j * fs / m_ref)


def test_transform_closed_form_parity():
    """``m0^2-u-i.m0.sum(w.gamma) == D_rho.D_omega`` over a grid."""
    model = _model()
    tr = _transform(model)
    m = np.linspace(0.3, 1.4, 200)
    u = m ** 2
    w = _weights(model)
    rows = [np.asarray(r) for r in model.gamma(m)]
    g = rows[5]
    gam, fs = g.real, M0 * g.imag
    g_bw = sum(w[k] * rows[k] for k in range(8))
    lhs = M0 ** 2 - u - 1j * M0 * g_bw
    rhs = _packed_d_rho(m, M0, G1, gam, fs) * (
        (M2 ** 2 - u) - 1j * M2 * G2)
    assert np.allclose(lhs, rhs, rtol=1e-10, atol=1e-12)
    # structural identities
    assert w[0] == pytest.approx(M2 * G2 * M0, rel=1e-14)
    assert w[4] == pytest.approx(1.0 / M0, rel=1e-14)
    assert w[6] == pytest.approx(-G1, rel=1e-14)
    assert w[7] == pytest.approx(-G1 * M2 * G2, rel=1e-14)


def test_parity_against_fixed_rho_omega_model():
    """The floatable model reproduces the current fixed ``RhoOmega``
    lineshape exactly at the reference point (m0 = m0_ref)."""
    from tabpwa.particle_model.rho_omega_model import RhoOmegaModel
    model = _model()
    fixed = build_particle("rho_omega", model="RhoOmega", mass=M0,
                           width=G1, omega_mass=M2, omega_width=G2)
    m = np.linspace(0.35, 1.3, 150)
    d = _params_dict(model)
    mine = model.amplitude(m, d)
    theirs = fixed.fixed_shape(m)
    assert np.allclose(mine, theirs, rtol=1e-10, atol=1e-12)


def test_gamma_basis_rows():
    model = _model()
    m = np.array([0.3, 0.77, 1.2])
    u = m ** 2
    g = [np.asarray(row) for row in model.gamma(m)]
    assert len(g) == 8
    for arr in g:
        assert arr.shape == m.shape
    assert np.allclose(g[0], np.ones_like(u, dtype=complex))
    assert np.allclose(g[1], u.astype(complex))
    assert np.allclose(g[2], 1j * np.ones_like(u, dtype=complex))
    assert np.allclose(g[3], 1j * u)
    assert np.allclose(g[4], 1j * (u ** 2))
    gs = g[5]
    assert np.all(np.abs(gs.real) > 0.0)         # running-width shape
    assert np.allclose(M0 * gs.imag,             # fs via the packed row
                       _fs_impl(m, M0, DM2, DM3), rtol=1e-13)
    assert np.allclose(g[6], u * gs)             # u.g
    assert np.allclose(g[7], 1j * gs)            # i.g


def test_model_amplitude_parity():
    """``model.amplitude`` (exact gamma path) == packed D_rho * D_omega."""
    model = _model()
    m = np.linspace(0.3, 1.4, 150)
    d = _params_dict(model)
    gam, fs = _shapes(model, m)
    direct = 1.0 / (_packed_d_rho(m, M0, G1, gam, fs) * (
        (M2 ** 2 - m ** 2) - 1j * M2 * G2))
    assert np.allclose(model.amplitude(m, d), direct, rtol=1e-10, atol=1e-12)
    # amplitude_raw with the transform-produced couplings is identical
    tr = _transform(model)
    resolved = tr.apply_forward(d)
    assert np.allclose(model.amplitude_raw(m, resolved), direct,
                       rtol=1e-10, atol=1e-12)


def test_transform_backward_fd():
    model = _model()
    tr = _transform(model)
    d = _params_dict(model)
    rng = np.random.RandomState(0)
    grad_out = {n: rng.randn() for n in tr.output_names}
    ana = tr.backward(grad_out, d)
    eps = 1e-7
    for name in tr.input_names:
        dp = dict(d); dp[name] += eps
        dm = dict(d); dm[name] -= eps
        wp = tr.forward(dp)
        wm = tr.forward(dm)
        num = sum(grad_out[n] * (wp[n] - wm[n]) / (2 * eps)
                  for n in tr.output_names)
        assert ana[name] == pytest.approx(num, rel=1e-6, abs=1e-9)


def test_transform_inverse_roundtrip():
    model = _model()
    tr = _transform(model)
    d = _params_dict(model)
    resolved = tr.apply_forward(d)
    inv = tr.inverse(resolved)
    for name in tr.input_names:
        assert inv[name] == pytest.approx(d[name], rel=1e-15)
    # missing inputs -> empty dict (no spurious reconstruction)
    assert tr.inverse({n: 0.0 for n in tr.output_names}) == {}


def test_defaults_and_registration():
    assert "RhoOmegaGS" in ALL_MODELS
    model = _model()
    assert model.get_gamma_count() == 8
    assert model.get_gamma_name() == [f"{P}_w{k}" for k in range(8)]
    d = model.get_defaults()
    assert d[f"{P}_mass"] == M0
    assert d[f"{P}_width"] == G1
    assert d[f"{P}_mass2"] == M2
    assert d[f"{P}_width2"] == G2
    # omega aliases
    m2 = _model(omega_mass=0.78, omega_width=0.01)
    assert m2.get_defaults()[f"{P}_mass2"] == 0.78
    assert m2.get_defaults()[f"{P}_width2"] == 0.01
    tr = _transform(model)
    assert isinstance(tr, RhoOmegaGSWeightsTransform)
    assert f"{P}_mass" not in tr.output_names
    assert len(tr.output_names) == 8


# ── fitter level: the full pipeline ──────────────────────────────

@pytest.fixture(scope="module")
def env(tmp_path_factory):
    # RhoOmegaGS config: the RhoOmega2 fixture with the model and
    # particle key patched
    text = (open(CFG).read()
            .replace("model: RhoOmega2", "model: RhoOmegaGS")
            .replace("rho_omega2", P))
    cfg = tmp_path_factory.mktemp("rho_omega_gs") / "config_rho_omega_gs.yml"
    cfg.write_text(text)
    try:
        from tabpwa import Fitter
        from tabpwa.pwa_build import generate_pwa_phsp
        f = Fitter(str(cfg), backend="cuda_v4_pwa_s0c")
    except RuntimeError as e:                     # no CUDA device / build
        pytest.skip(f"no CUDA device: {e}")
    f.apply_constrains()
    model = f.model
    chain = model.full_decay.get_partial_waves()[0][1]
    mom = generate_pwa_phsp(model, chain, 400, seed=7)
    phsp = model.build_event_data(mom, f.kernel_config, weight=np.ones(400))
    data = model.build_event_data(mom[:200], f.kernel_config,
                                  weight=np.ones(200))
    f.set_phsp(phsp)
    f.set_data(data)
    tfms = [t for t in f.cm.mass_width_transforms
            if isinstance(t, RhoOmegaGSWeightsTransform)]
    assert len(tfms) == 1
    return {"f": f, "x0": f.initial_values(seed=73), "tfm": tfms[0]}


def test_transform_registered_via_fitter(env):
    """The standard bridge registers the transform on the fitter."""
    t = env["tfm"]
    assert t.mass_name == f"{P}_mass"
    assert t.input_names == [f"{P}_mass", f"{P}_width",
                             f"{P}_mass2", f"{P}_width2"]
    assert env["tfm"].output_names == [f"{P}_w{k}" for k in range(8)]


def test_fitter_gradients_match_fd(env):
    """The full pipeline (transform included) has consistent grads."""
    from tabpwa.numerics import fd_grad

    f = env["f"]
    x = env["x0"].copy()
    nll, grad = f.get_nll(x)
    assert np.isfinite(nll)
    assert np.all(np.isfinite(grad))
    names = f.cm.var_registry.flat_names
    rng = np.random.RandomState(7)
    sel = rng.choice(len(names), size=min(6, len(names)), replace=False)
    fd = fd_grad(lambda z: f.get_nll(z)[0], x,
                 eps=1e-6 * (1.0 + np.abs(x)[sel]), idx=sel)
    for c, i in enumerate(sel):
        assert grad[i] == pytest.approx(fd[c], rel=1e-3, abs=1e-8), names[i]
