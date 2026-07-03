#!/usr/bin/env python3
"""Tests for recently added/modified APIs: LinearTransform, fmt_meas,
model get_defaults, get_decay_ck_indices(wave_idx), save_params(x_flat),
ConstraintManager resolve/inverse/chain_gradient, and Fitter.fit().

Run with::

    pytest tests/test_new_apis.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json, io, tempfile

from ampfit.param_constraint import (
    LinearTransform, ScaleTransform, ConstraintManager,
)
from ampfit.utils import fmt_meas
from ampfit.config_loader import Config
from ampfit import Fitter


# ═══════════════════════════════════════════════════════════════════
# 1. LinearTransform (replaces ScaleTransform)
# ═══════════════════════════════════════════════════════════════════

def test_lineartransform_forward():
    """Basic scale (bias=0) matches old behavior."""
    t = LinearTransform('x', 2.0)
    assert t.forward({'x': 3.0}) == {'x': 6.0}
    assert t.forward({'x': 0.0}) == {'x': 0.0}
    assert t.forward({'x': -1.5}) == {'x': -3.0}


def test_lineartransform_forward_with_bias():
    """Scale + bias: y = a * x + b."""
    t = LinearTransform('x', 2.0, 5.0)
    assert t.forward({'x': 3.0}) == {'x': 11.0}   # 2*3 + 5
    assert t.forward({'x': 0.0}) == {'x': 5.0}    # 2*0 + 5
    assert t.forward({'x': -1.0}) == {'x': 3.0}   # 2*(-1) + 5


def test_lineartransform_forward_missing_key():
    """If key not in dict, dict passes through unchanged."""
    t = LinearTransform('x', 2.0, 5.0)
    assert t.forward({'y': 10.0}) == {'y': 10.0}


def test_lineartransform_backward():
    """Backward: d(y)/d(x) = a. Bias doesn't appear."""
    t = LinearTransform('x', 2.0, 5.0)
    assert t.backward({'x': 4.0}) == {'x': 8.0}


def test_lineartransform_backward_missing():
    """Missing key passes through unchanged."""
    t = LinearTransform('x', 2.0)
    assert t.backward({'y': 4.0}) == {'y': 4.0}


def test_lineartransform_inverse():
    """Inverse: x = (y - b) / a."""
    t = LinearTransform('x', 2.0, 5.0)
    assert t.inverse({'x': 11.0}) == {'x': 3.0}
    assert t.inverse({'x': 5.0}) == {'x': 0.0}


def test_lineartransform_inverse_zero_factor():
    """If factor==0, inverse is a no-op (protect division by zero)."""
    t = LinearTransform('x', 0.0, 5.0)
    assert t.inverse({'x': 10.0}) == {'x': 10.0}


def test_lineartransform_roundtrip():
    """forward(inverse(d)) == d and inverse(forward(d)) == d."""
    t = LinearTransform('x', 2.0, 5.0)
    d = {'x': 3.5}
    assert t.inverse(t.forward(d)) == d
    assert t.forward(t.inverse(d)) == d
    # Also test with bias=0
    t2 = LinearTransform('x', -1.0)
    assert t2.inverse(t2.forward(d)) == d
    assert t2.forward(t2.inverse(d)) == d


def test_lineartransform_backward_roundtrip():
    """backward is consistent: grad = a * grad."""
    t = LinearTransform('x', 3.0, 2.0)
    g = {'x': 5.0, 'y': 2.0}
    bg = t.backward(g)
    assert bg['x'] == 15.0   # 5 * 3
    assert bg['y'] == 2.0    # unchanged


def test_scale_transform_alias():
    """ScaleTransform is an alias for LinearTransform."""
    assert ScaleTransform is LinearTransform
    s = ScaleTransform('x', -1.0)
    assert s.forward({'x': 5.0}) == {'x': -5.0}
    assert s.inverse({'x': -5.0}) == {'x': 5.0}


# ═══════════════════════════════════════════════════════════════════
# 2. utils.fmt_meas
# ═══════════════════════════════════════════════════════════════════

def test_fmt_meas_no_error():
    """e=0 → just value, no ±."""
    assert fmt_meas(5.0, 0.0) == r'$5.00$'
    assert fmt_meas(5.0, 0.0, pct=True) == r'$5.0$\%'


def test_fmt_meas_none():
    """v=None → $-$."""
    assert fmt_meas(None, 0.0) == r'$-$'


def test_fmt_meas_threshold_2dp():
    """e < 0.355 → 2 decimal places."""
    result = fmt_meas(60.19, 0.056)
    assert '60.19' in result and '0.06' in result


def test_fmt_meas_threshold_1dp():
    """0.355 ≤ e < 0.950 → 1 decimal place."""
    result = fmt_meas(1.30, 0.60)
    assert result.count('.') == 2  # one for value, one for error


def test_fmt_meas_threshold_0dp():
    """e ≥ 0.950 → 0 decimal places."""
    result = fmt_meas(18.76, 4.29, pct=True)
    assert r'$19\pm4\%$' in result or r'$19\pm4$' in result


def test_fmt_meas_pct_suffix():
    """pct=True appends percent sign."""
    result = fmt_meas(5.0, 0.3, pct=True)
    assert result.endswith(r'\%$')


# ═══════════════════════════════════════════════════════════════════
# 3. Particle model get_defaults
# ═══════════════════════════════════════════════════════════════════

def test_model_get_defaults_bw():
    """BW model has mass+width defaults."""
    from ampfit.particle_model.models_builtin import BWModel
    m = BWModel('test', mass=1.0, width=0.1)
    d = m.get_defaults()
    assert isinstance(d, dict)
    assert 'test_mass' in d and d['test_mass'] == 1.0
    assert 'test_width' in d and d['test_width'] == 0.1


def test_model_get_defaults_bwr():
    """BWR model has mass+width defaults."""
    from ampfit.particle_model.bwr_model import BWRModel
    m = BWRModel('test', mass=1.0, width=0.1, L=1, daug2Mass=0.14, daug3Mass=0.14)
    d = m.get_defaults()
    assert 'test_mass' in d
    assert 'test_width' in d


def test_model_get_defaults_gs():
    """GS_rho model has mass+width defaults."""
    from ampfit.particle_model.gs_rho_model import GSRhoModel
    m = GSRhoModel('test', mass=1.0, width=0.1)
    d = m.get_defaults()
    assert 'test_mass' in d
    assert 'test_width' in d


def test_model_get_defaults_flattec():
    """FlatteC model has mass and _g0.._gN couplings."""
    from ampfit.particle_model.models_builtin import FlatteCModel
    m = FlatteCModel('f0_980', mass=0.965, mass_list=[(0.1, 0.1), (0.14, 0.14)])
    d = m.get_defaults()
    assert 'f0_980_mass' in d
    assert 'f0_980_g0' in d
    assert 'f0_980_g1' in d


def test_model_get_defaults_one():
    """OneModel returns {} (params fixed by transform)."""
    from ampfit.particle_model.models_builtin import OneModel
    m = OneModel('NR0', mass=0.475, width=0.55)
    assert m.get_defaults() == {}


def test_model_get_defaults_fixed_shape():
    """FixedShapeModel subclasses return {} (params fixed by transform)."""
    from ampfit.particle_model.rho_omega_model import RhoOmegaModel
    m = RhoOmegaModel('test', mass=0.775, width=0.149)
    assert m.get_defaults() == {}




def test_defaults_via_fitter():
    """Fitter.defaults includes mass and width from models."""
    fitter = Fitter('config_amp.yml', backend='numpy')
    defaults = fitter.defaults
    # Check a known mass is present
    assert 'a1(1260)p_mass' in defaults
    assert 'a1(1260)p_width' in defaults
    # Flatte couplings
    assert 'f0(980)_g0' in defaults
    assert 'f0(980)_g1' in defaults
    # CK matrix: width only (gamma names computed by transform)
    assert 'pi2(1670)p_width' in defaults
    assert 'pi2(1670)p_mass' in defaults
    # OneModel NR0: not in defaults (transform handles it)
    assert 'NR0_mass' not in defaults
    assert 'NR0_width' not in defaults


# ═══════════════════════════════════════════════════════════════════
# 4. Config.get_decay_ck_indices with wave_idx
# ═══════════════════════════════════════════════════════════════════

def test_get_decay_ck_indices():
    """Basic pair resolution returns non-empty indices."""
    cfg = Config('config_amp.yml')
    idx = cfg.get_decay_ck_indices([('a1(1260)p', 'rhoA')])
    assert len(idx) > 0
    assert all(isinstance(i, int) for i in idx)


def test_get_decay_ck_indices_wave_idx():
    """wave_idx separates S-wave (0) from D-wave (1) for a1→ρπ."""
    cfg = Config('config_amp.yml')
    s_wave = cfg.get_decay_ck_indices([('a1(1260)p', 'rhoA')], wave_idx=0)
    d_wave = cfg.get_decay_ck_indices([('a1(1260)p', 'rhoA')], wave_idx=1)
    all_wave = cfg.get_decay_ck_indices([('a1(1260)p', 'rhoA')])
    # S-wave + D-wave = all (disjoint union)
    assert len(s_wave) == len(d_wave)
    assert len(s_wave) + len(d_wave) == len(all_wave)
    assert sorted(s_wave + d_wave) == sorted(all_wave)


def test_get_decay_ck_indices_nonexistent():
    """Non-existent decay pair returns empty list."""
    cfg = Config('config_amp.yml')
    idx = cfg.get_decay_ck_indices([('a1(1260)p', 'NONEXISTENT')])
    assert idx == []


def test_get_decay_ck_indices_multi_pair():
    """Multiple pairs resolve independently and are unioned."""
    cfg = Config('config_amp.yml')
    single_a = cfg.get_decay_ck_indices([('pi2(1670)p', 'f0(500)')])
    single_b = cfg.get_decay_ck_indices([('pi2(1670)p', 'f0(980)')])
    # This must be called one pair at a time (AND semantics per call)
    assert len(single_a) > 0
    assert len(single_b) > 0


# ═══════════════════════════════════════════════════════════════════
# 5. ConstraintManager resolve / inverse / chain_gradient
# ═══════════════════════════════════════════════════════════════════

def _make_cm():
    """Create a minimal ConstraintManager for unit tests."""
    cm = ConstraintManager(
        all_comb=[('p0', 'p1')],
        all_names=['x', 'y', 'z']
    )
    return cm


def test_cm_resolve_basic():
    """resolve passes through unchanged dict."""
    cm = _make_cm()
    d = cm.resolve({'x': 1.0, 'y': 2.0, 'z': 3.0})
    assert d['x'] == 1.0
    assert d['y'] == 2.0


def test_cm_inverse_basic():
    """inverse returns resolve's input."""
    cm = _make_cm()
    d = {'x': 1.0, 'y': 2.0}
    resolved = cm.resolve(d)
    back = cm.inverse(resolved)
    for k in d:
        assert abs(back[k] - d[k]) < 1e-12, f"{k} mismatch"


def test_cm_set_scale_roundtrip():
    """set_scale → resolve → inverse recovers original."""
    cm = _make_cm()
    cm.set_scale({'x': -1.0})
    d = {'x': 5.0, 'y': 2.0, 'z': 3.0}
    resolved = cm.resolve(d)
    assert resolved['x'] == -5.0  # scaled
    back = cm.inverse(resolved)
    assert back['x'] == 5.0       # inverse recovers


def test_cm_set_scale_with_bias():
    """set_scale with (factor, bias) tuple."""
    cm = _make_cm()
    cm.set_scale({'x': (2.0, 5.0)})
    d = {'x': 3.0, 'y': 2.0}
    resolved = cm.resolve(d)
    assert resolved['x'] == 11.0  # 2*3 + 5
    back = cm.inverse(resolved)
    assert abs(back['x'] - 3.0) < 1e-12


def test_cm_chain_gradient():
    """chain_gradient reverses scale transform."""
    cm = _make_cm()
    cm.set_scale({'x': -1.0})
    d = {'x': 5.0, 'y': 2.0, 'z': 3.0}
    resolved = cm.resolve(d)
    grad_in = {'x': 1.0, 'y': 0.0}
    grad_out = cm.chain_gradient(grad_in, resolved, d)
    assert grad_out['x'] == -1.0  # d(-x)/dx = -1


def test_cm_set_scale_then_free():
    """set_scale then set_free removes the transform."""
    cm = _make_cm()
    cm.set_scale({'x': 2.0})
    assert 'x' in cm.scale_params
    cm.set_free('x')
    assert 'x' not in cm.scale_params


# ═══════════════════════════════════════════════════════════════════
# 6. Fitter.save_params with flat x vector
# ═══════════════════════════════════════════════════════════════════

def _make_fitter():
    """Create a fitter with minimal data for testing."""
    fitter = Fitter('config_amp.yml', backend='numpy')
    n_d, n_p = 10, 20
    data = {
        'mass': np.random.uniform(2, 3, (n_d, 48)),
        'q': np.random.random((n_d, 72)),
        'angle': np.random.random((n_d, 24, 3)),
        'frac': np.random.random((n_d,)),
        'time': np.random.random((n_d,)),
        'bkg': np.random.random((n_d,)) * 0.01,
        'weight': np.ones((n_d,)),
    }
    phsp = {k: np.random.random((n_p,) + v.shape[1:]) if v.ndim > 1
            else np.random.random(n_p)
            for k, v in data.items()}
    fitter.set_phsp(phsp)
    fitter.set_data(data)
    return fitter


def test_save_params_with_x_vector():
    """save_params accepts flat x vector and produces valid JSON."""
    fitter = _make_fitter()
    x = fitter.initial_values()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        path = f.name
    try:
        fitter.save_params(x, path)
        with open(path) as f:
            data = json.load(f)
        assert 'value' in data
        assert len(data['value']) > 0
        # Error dict may be empty (no hess_inv)
        assert 'error' in data
        # Status should be present with nan NLL
        assert 'status' in data
        assert np.isnan(data['status']['NLL'])
    finally:
        os.unlink(path)


def test_save_params_roundtrip():
    """save_params(x) → load via values_from_dict → same NLL."""
    fitter = _make_fitter()
    x0 = fitter.initial_values()
    nll0, _ = fitter.get_nll(x0)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        path = f.name
    try:
        fitter.save_params(x0, path)
        with open(path) as f:
            data = json.load(f)
        x1 = fitter.values_from_dict(data)
        nll1, _ = fitter.get_nll(x1)
        assert abs(nll1 - nll0) < 1e-6, f"NLL mismatch: {nll0} vs {nll1}"
    finally:
        os.unlink(path)


# ═══════════════════════════════════════════════════════════════════
# 7. Fitter — basic fit integration
# ═══════════════════════════════════════════════════════════════════

def test_fitter_fit_basic():
    """fit() runs and returns a result with expected keys."""
    fitter = _make_fitter()
    x0 = fitter.initial_values()
    result = fitter.fit(x0, maxiter=3, disp=False)
    assert hasattr(result, 'x')
    assert hasattr(result, 'fun')
    assert hasattr(result, 'hess_inv')
    assert len(result.x) == len(x0)


def test_fitter_fit_nll_decreases():
    """fit() reduces NLL from initial value."""
    fitter = _make_fitter()
    x0 = fitter.initial_values()
    nll0, _ = fitter.get_nll(x0)
    result = fitter.fit(x0, maxiter=5, disp=False)
    nll_final = result.fun
    assert nll_final <= nll0 + 1e-6, f"NLL increased: {nll0} → {nll_final}"


def test_fitter_fit_with_constraints():
    """fit() works when constraints (scale, same) are set — may not converge."""
    fitter = _make_fitter()
    free = fitter.free_param_names()
    if len(free) > 2:
        fitter.set_same([[free[0], free[1]]])
    x0 = fitter.initial_values()
    result = fitter.fit(x0, maxiter=3, disp=False)
    # Should return a result with expected attributes even if not converged
    assert hasattr(result, 'x')
    assert hasattr(result, 'fun')
    assert len(result.x) == len(x0)


# ═══════════════════════════════════════════════════════════════════
# 8. Fitter._last_xk
# ═══════════════════════════════════════════════════════════════════

def test_last_xk_initially_none():
    """_last_xk is None before any get_nll call."""
    fitter = _make_fitter()
    assert fitter._last_xk is None


def test_last_xk_after_get_nll():
    """_last_xk stores x from last get_nll call."""
    fitter = _make_fitter()
    x0 = fitter.initial_values()
    _, _ = fitter.get_nll(x0)
    assert fitter._last_xk is not None
    assert len(fitter._last_xk) == len(x0)
    assert np.allclose(fitter._last_xk, x0)


def test_last_xk_updates():
    """_last_xk is updated on each get_nll call."""
    fitter = _make_fitter()
    x0 = fitter.initial_values()
    _, _ = fitter.get_nll(x0)
    x1 = x0 + 0.1
    _, _ = fitter.get_nll(x1)
    assert np.allclose(fitter._last_xk, x1)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
