#!/usr/bin/env python3
"""Tests for the two-pole rho-omega product model (``RhoOmega2``).

The model realises

    A(m) = 1 / (D_rho(m) . D_omega(m)),
    D_p(u) = m_p^2 - u - i . m_p . Gamma_p,   u = m^2

with all four masses/widths fitted, using ONE kernel Breit-Wigner whose
fixed 6-row gamma basis ``[1, i, u, i.u, u^2, i.u^2]`` is contracted with
six real couplings produced by :class:`RhoOmegaWeightsTransform`.  No
kernel code is modified.

Checks:
1. closed-form weight/denominator parity (``m0^2-u-i.m0.g_bw == D_rho.D_omega``);
2. gamma basis rows and model-amplitude parity against the direct product;
3. transform backward vs central finite differences;
4. kernel-path lineshape parity (Catmull-Rom interpolation error reported);
5. full ``Fitter`` gradients (numpy_pwa and cuda_v4_pwa) vs 3-point FD;
6. cuda_v4_pwa selection/NLL sanity and numpy-vs-cuda NLL parity.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ampfit.particle_model import build_particle, ALL_MODELS
from ampfit.particle_model.rho_omega2_model import (
    RhoOmega2Model, RhoOmegaWeightsTransform,
)
from ampfit.config_loader import Config
from ampfit.amp_model import build_amplitude_model
from ampfit.numpy_kernel import NumpyKernel

CFG = "tests/config_rho_omega2.yml"
# Default pole parameters (rho, omega)
M0, G1, M2, G2 = 0.775, 0.149, 0.78266, 0.00868
P = "rho_omega2"


# ── helpers ──────────────────────────────────────────────────────────

def _model(**kw):
    kw.setdefault("mass", M0)
    kw.setdefault("width", G1)
    if "omega_mass" not in kw:
        kw.setdefault("mass2", M2)
    if "omega_width" not in kw:
        kw.setdefault("width2", G2)
    return build_particle(P, model="RhoOmega2", **kw)


def _transform(model):
    return model.make_mass_width_transform()


def _params_dict(model, m0=M0, g1=G1, m2=M2, g2=G2):
    return {f"{model.name}_mass": m0, f"{model.name}_width": g1,
            f"{model.name}_mass2": m2, f"{model.name}_width2": g2}


def _weights(model, m0=M0, g1=G1, m2=M2, g2=G2):
    tr = _transform(model)
    out = tr.forward(_params_dict(model, m0, g1, m2, g2))
    return np.array([out[n] for n in tr.output_names])


def _direct_denominator(m, m0, g1, m2, g2):
    u = np.asarray(m, dtype=float) ** 2
    a1 = m0 ** 2 - 1j * m0 * g1
    a2 = m2 ** 2 - 1j * m2 * g2
    return (a1 - u) * (a2 - u)


def _find_model(ampl_model):
    for chain in ampl_model.full_decay.chains:
        for decay in chain.decays[1:]:
            if decay.core.name == P:
                return decay.core._model
    raise AssertionError(f"particle {P!r} not found in decay tree")


def _build_fitter(backend):
    from ampfit import Fitter
    from ampfit.pwa_build import generate_pwa_phsp

    f = Fitter(CFG, backend=backend)
    cfg = f.model
    chain = cfg.full_decay.get_partial_waves()[0][1]
    mom = generate_pwa_phsp(cfg, chain, 400, seed=7)
    phsp = cfg.build_event_data(mom, f.kernel_config, weight=np.ones(400))
    data = cfg.build_event_data(mom[:200], f.kernel_config, weight=np.ones(200))
    f.set_phsp(phsp)
    f.set_data(data)
    return f


def _free_four(f):
    for attr in ("mass", "width", "mass2", "width2"):
        f.set_free(f"{P}_{attr}")


def _fd_grads(f, x, names, step=1e-6):
    """Central-difference NLL gradient at the free params in *names*."""
    idx = {n: i for i, n in enumerate(f.free_param_names())}
    out = {}
    for n in names:
        i = idx[n]
        xp = x.copy(); xp[i] += step
        xm = x.copy(); xm[i] -= step
        out[n] = (f.get_nll(xp)[0] - f.get_nll(xm)[0]) / (2.0 * step)
    return out


# ── closed form / basis ──────────────────────────────────────────────

def test_transform_closed_form_parity():
    """``m0^2 - u - i.m0.g_bw(u) == D_rho(u).D_omega(u)`` over a grid."""
    model = _model()
    tr = _transform(model)
    m = np.linspace(0.3, 1.4, 200)
    u = m ** 2
    w = _weights(model)
    gamma = model.gamma(m)
    g_bw = sum(w[k] * gamma[k] for k in range(6))
    lhs = M0 ** 2 - u - 1j * M0 * g_bw
    rhs = _direct_denominator(m, M0, G1, M2, G2)
    assert np.allclose(lhs, rhs, rtol=1e-10, atol=1e-12)
    # w4 = Re(i/m0) = 0 exactly, w5 = Im(i/m0) = 1/m0
    assert abs(w[4]) < 1e-15
    assert w[5] == pytest.approx(1.0 / M0, rel=1e-14)


def test_gamma_basis_rows():
    model = _model()
    m = np.array([0.3, 0.77, 1.2])
    u = m ** 2
    g = model.gamma(m)
    assert len(g) == 6
    for arr in g:
        assert arr.shape == m.shape
    assert np.allclose(g[0], np.ones_like(u, dtype=complex))
    assert np.allclose(g[1], 1j * np.ones_like(u, dtype=complex))
    assert np.allclose(g[2], u.astype(complex))
    assert np.allclose(g[3], 1j * u)
    assert np.allclose(g[4], (u ** 2).astype(complex))
    assert np.allclose(g[5], 1j * (u ** 2))


def test_model_amplitude_parity():
    """``model.amplitude`` (exact gamma path) == direct 1/(D_rho D_omega)."""
    model = _model()
    m = np.linspace(0.3, 1.4, 150)
    d = _params_dict(model)
    direct = 1.0 / _direct_denominator(m, M0, G1, M2, G2)
    assert np.allclose(model.amplitude(m, d), direct, rtol=1e-10, atol=1e-12)
    # amplitude_raw with the transform-produced couplings is identical
    tr = _transform(model)
    resolved = tr.apply_forward(d)
    assert np.allclose(model.amplitude_raw(m, resolved), direct,
                       rtol=1e-10, atol=1e-12)


# ── transform backward / inverse / defaults ──────────────────────────

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
    assert "RhoOmega2" in ALL_MODELS
    model = _model()
    assert model.get_gamma_count() == 6
    assert model.get_gamma_name() == [f"{P}_w{k}" for k in range(6)]
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
    assert isinstance(tr, RhoOmegaWeightsTransform)
    assert f"{P}_mass" not in tr.output_names
    assert len(tr.output_names) == 6


# ── kernel-path lineshape (interpolation error) ──────────────────────

def test_kernel_lineshape_interpolation():
    """Kernel Catmull-Rom gamma path reproduces ``1/(D_rho D_omega)``."""
    cfg = Config(CFG)
    ampl = build_amplitude_model(cfg)
    kc = ampl.build_kernel_config()
    nk = NumpyKernel(kc)

    model = _find_model(ampl)
    tr = _transform(model)
    w = tr.forward(_params_dict(model))
    wv = np.array([w[n] for n in tr.output_names])
    rows = np.array([kc["g0_names"].index(n) for n in tr.output_names])

    m = np.linspace(0.4, 1.2, 400)
    types = np.broadcast_to(rows[:, None], (6, len(m)))
    x = np.broadcast_to(m, (6, len(m)))
    g_interp = nk.interp_catmull_rom(
        kc["gamma_table"], types, x, kc["gamma_min"], kc["gamma_delta"])
    g_bw = (wv[:, None] * g_interp).sum(axis=0)
    u = m ** 2
    bw_kernel = M0 ** 2 - u - 1j * M0 * g_bw
    exact = _direct_denominator(m, M0, G1, M2, G2)

    rel = np.abs(bw_kernel - exact) / np.abs(exact)
    # Catmull-Rom table interpolation: sub-1e-5 on the physical mass range
    assert np.max(rel) < 1e-4
    print(f"\nRhoOmega2 kernel gamma-table max rel. lineshape error = "
          f"{np.max(rel):.3e}")


# ── Fitter gradients (numpy_pwa reference) ───────────────────────────

def test_fitter_gradients_numpy_pwa():
    f = _build_fitter("numpy_pwa")
    try:
        _free_four(f)
        free = f.free_param_names()
        for attr in ("mass", "width", "mass2", "width2"):
            assert f"{P}_{attr}" in free
        x = f.initial_values(seed=3)
        nll, grad = f.get_nll(x)
        assert np.isfinite(nll)
        assert np.all(np.isfinite(grad))

        names = [f"{P}_{a}" for a in ("mass", "width", "mass2", "width2")]
        fd = _fd_grads(f, x, names)
        idx = {n: i for i, n in enumerate(f.free_param_names())}
        for n in names:
            assert grad[idx[n]] == pytest.approx(fd[n], rel=1e-3)
    finally:
        f.backend.free()


# ── cuda_v4_pwa ──────────────────────────────────────────────────────

def _cuda_fitter_or_skip():
    from ampfit.backends import backends_for_model
    if "cuda_v4_pwa" not in backends_for_model("pwa"):
        pytest.skip("cuda_v4_pwa not registered for the pwa model")
    try:
        f = _build_fitter("cuda_v4_pwa")
    except RuntimeError as e:                     # no CUDA device / build
        pytest.skip(f"no CUDA device: {e}")
    return f


def test_cuda_v4_pwa_selected_and_nll():
    f = _cuda_fitter_or_skip()
    try:
        from ampfit.backends import backends_for_model
        assert "cuda_v4_pwa" in backends_for_model(f.model.name)
        _free_four(f)
        x = f.initial_values(seed=4)
        nll, grad = f.get_nll(x)
        assert np.isfinite(nll)
        assert np.all(np.isfinite(grad))

        # changing mass2 must change the NLL
        idx = {n: i for i, n in enumerate(f.free_param_names())}
        i2 = idx[f"{P}_mass2"]
        x2 = x.copy(); x2[i2] += 0.01
        nll2, _ = f.get_nll(x2)
        assert abs(nll2 - nll) > 1e-6
    finally:
        f.backend.free()


def test_cuda_v4_pwa_gradients_fd():
    f = _cuda_fitter_or_skip()
    try:
        _free_four(f)
        x = f.initial_values(seed=6)
        nll, grad = f.get_nll(x)
        names = [f"{P}_{a}" for a in ("mass", "width", "mass2", "width2")]
        fd = _fd_grads(f, x, names)
        idx = {n: i for i, n in enumerate(f.free_param_names())}
        for n in names:
            assert grad[idx[n]] == pytest.approx(fd[n], rel=1e-3)
    finally:
        f.backend.free()


def test_numpy_vs_cuda_nll_parity():
    fb = _cuda_fitter_or_skip()            # skips cleanly without a GPU
    try:
        fa = _build_fitter("numpy_pwa")
        try:
            x = fa.initial_values(seed=5)
            na, ga = fa.get_nll(x)
            nb, gb = fb.get_nll(x)
            assert na == pytest.approx(nb, rel=1e-6)
            assert np.allclose(ga, gb, rtol=1e-4, atol=1e-6)
        finally:
            fa.backend.free()
    finally:
        fb.backend.free()
