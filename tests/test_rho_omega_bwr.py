#!/usr/bin/env python3
"""Tests for the BWR-rho x BW-omega product model (``RhoOmegaBWR``).

The model realises

    A(m) = 1 / (D_rho(m) . D_omega(m)),
    D_rho   = m_rho^2 - u - i . m_rho . Gamma0 . gamma(u),   u = m^2
    D_omega = m_omega^2 - u - i . m_omega . Gamma_omega

with ``gamma(u)`` the running-width shape (the ``RhoOmega`` lineshape
with GS replaced by a simple BWR) and all four masses/widths fitted,
using ONE kernel Breit-Wigner whose fixed 8-row gamma basis

    [1, u, i, i.u, i.u^2, gamma(u), i.gamma(u), u.gamma(u)]

is contracted with eight real couplings produced by
:class:`RhoOmegaBWRWeightsTransform`.  No kernel code is modified.

Verified:
1. closed-form weight/denominator parity
   (``m0^2-u-i.m0.sum(w.gamma) == D_rho.D_omega``);
2. gamma basis rows and model-amplitude parity against the direct product;
3. transform backward vs central finite differences; inverse round-trip;
4. registration/defaults (``build_particle``, gamma count/names);
5. the transform registers through the standard fitter bridge;
6. the full fitter pipeline (transform included) matches centered
   finite-difference gradients.

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
from tabpwa.particle_model.rho_omega_bwr_model import (
    RhoOmegaBWRModel, RhoOmegaBWRWeightsTransform,
)

CFG = "tests/config_rho_omega2.yml"
# Default pole parameters (rho, omega)
M0, G1, M2, G2 = 0.775, 0.149, 0.78266, 0.00868
P = "rho_omega_bwr"


# ── unit level: the particle model itself ────────────────────────

def _model(**kw):
    kw.setdefault("mass", M0)
    kw.setdefault("width", G1)
    if "omega_mass" not in kw:
        kw.setdefault("mass2", M2)
    if "omega_width" not in kw:
        kw.setdefault("width2", G2)
    return build_particle(P, model="RhoOmegaBWR", **kw)


def _transform(model):
    return model.make_mass_width_transform()


def _params_dict(model, m0=M0, g1=G1, m2=M2, g2=G2):
    return {f"{model.name}_mass": m0, f"{model.name}_width": g1,
            f"{model.name}_mass2": m2, f"{model.name}_width2": g2}


def _weights(model, m0=M0, g1=G1, m2=M2, g2=G2):
    tr = _transform(model)
    out = tr.forward(_params_dict(model, m0, g1, m2, g2))
    return np.array([out[n] for n in tr.output_names])


def _direct_denominator(m, m0, g1, m2, g2, gam):
    """D_rho . D_omega with the given running-width shape *gam*."""
    u = np.asarray(m, dtype=float) ** 2
    d_rho = m0 ** 2 - u - 1j * m0 * g1 * gam
    d_om = m2 ** 2 - u - 1j * m2 * g2
    return d_rho * d_om


def test_transform_closed_form_parity():
    """``m0^2-u-i.m0.sum(w.gamma) == D_rho.D_omega`` over a grid."""
    model = _model()
    tr = _transform(model)
    m = np.linspace(0.3, 1.4, 200)
    u = m ** 2
    w = _weights(model)
    rows = [np.asarray(r) for r in model.gamma(m)]
    gam = rows[5].real
    g_bw = sum(w[k] * rows[k] for k in range(8))
    lhs = M0 ** 2 - u - 1j * M0 * g_bw
    rhs = _direct_denominator(m, M0, G1, M2, G2, gam)
    assert np.allclose(lhs, rhs, rtol=1e-10, atol=1e-12)
    # structural identities: w4 = 1/m0, w7 = -Gamma0
    assert w[4] == pytest.approx(1.0 / M0, rel=1e-14)
    assert w[7] == pytest.approx(-G1, rel=1e-14)


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
    gam = g[5]
    assert np.allclose(gam.imag, 0.0)            # real shape
    assert np.all(gam.real > 0.0)                # positive running width
    assert np.allclose(g[6], 1j * gam)           # i.gamma
    assert np.allclose(g[7], u * gam)            # u.gamma


def test_model_amplitude_parity():
    """``model.amplitude`` (exact gamma path) == direct 1/(D_rho D_omega)."""
    model = _model()
    m = np.linspace(0.3, 1.4, 150)
    d = _params_dict(model)
    gam = np.asarray(model.gamma(m))[5].real
    direct = 1.0 / _direct_denominator(m, M0, G1, M2, G2, gam)
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
    assert "RhoOmegaBWR" in ALL_MODELS
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
    assert isinstance(tr, RhoOmegaBWRWeightsTransform)
    assert f"{P}_mass" not in tr.output_names
    assert len(tr.output_names) == 8


# ── fitter level: the full pipeline ──────────────────────────────

@pytest.fixture(scope="module")
def env(tmp_path_factory):
    # RhoOmegaBWR config: the RhoOmega2 fixture with the model and
    # particle key patched
    text = (open(CFG).read()
            .replace("model: RhoOmega2", "model: RhoOmegaBWR")
            .replace("rho_omega2", P))
    cfg = tmp_path_factory.mktemp("rho_omega_bwr") / "config_rho_omega_bwr.yml"
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
            if isinstance(t, RhoOmegaBWRWeightsTransform)]
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
