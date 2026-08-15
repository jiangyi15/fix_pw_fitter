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
import pytest

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
    """cm.defaults includes mass and width from models."""
    fitter = Fitter('config_amp.yml', backend='numpy')
    defaults = fitter.cm.defaults
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
    """set_scale then set_free preserves the transform (fix/scale are independent)."""
    cm = _make_cm()
    cm.set_scale({'x': 2.0})
    assert 'x' in cm.scale_params
    cm.set_free('x')
    assert 'x' in cm.scale_params


# ═══════════════════════════════════════════════════════════════════
# 5b. ConstraintManager resolve/flat_resolve stop_after
# ═══════════════════════════════════════════════════════════════════

def _make_cm_with_constraints():
    """ConstraintManager with defaults, bounds, fixed, same, scale set up."""
    cm = ConstraintManager(
        all_names=['x', 'y', 'z', 'w']
    )
    cm.set_defaults({'x': 10.0, 'y': 20.0, 'z': 30.0})
    cm.set_range('x', -5, 5)
    cm.set_fixed({'w': 99.0})
    cm.set_same([('y', 'z')])      # z aliases to y
    cm.set_scale({'y': (2.0, 1.0)})  # y → 2*y + 1
    return cm


def test_cm_resolve_stop_after_transforms_defaults_excluded():
    """stop_after='transforms' returns pipeline output, no defaults."""
    cm = _make_cm_with_constraints()
    d = cm.resolve({'x': 1.0, 'y': 2.0}, stop_after='transforms')
    assert 'x' in d
    assert 'y' in d
    assert 'z' in d  # aliased


def test_cm_resolve_stop_after_bounds():
    """stop_after='bounds' applies bounds but no further transforms."""
    cm = _make_cm_with_constraints()
    d = cm.resolve({'x': 10.0, 'y': 2.0}, stop_after='bounds')
    # x has range [-5, 5]; bounds smoothly maps unbounded → bounded
    assert d['x'] > -5 and d['x'] < 5, f"x={d['x']} should be within [-5,5]"
    assert d['y'] == 2.0  # no scale yet
    assert 'z' not in d   # not in input, not aliased yet


def test_cm_resolve_stop_after_fixed():
    """stop_after='fixed' injects fixed values."""
    cm = _make_cm_with_constraints()
    d = cm.resolve({'x': 1.0, 'y': 2.0}, stop_after='fixed')
    assert d['w'] == 99.0        # fixed injected
    assert d['y'] == 2.0          # no scale yet
    assert 'z' not in d           # not aliased yet


def test_cm_resolve_stop_after_same():
    """stop_after='same' resolves aliases."""
    cm = _make_cm_with_constraints()
    d = cm.resolve({'x': 1.0, 'y': 2.0}, stop_after='same')
    assert d['z'] == d['y']       # z aliased to y
    assert d['y'] == 2.0          # no scale yet


def test_cm_resolve_stop_after_scale():
    """stop_after='scale' applies scale but not mass_width."""
    cm = _make_cm_with_constraints()
    d = cm.resolve({'x': 1.0, 'y': 2.0}, stop_after='scale')
    assert abs(d['y'] - 5.0) < 1e-12  # 2*2 + 1 = 5


def test_cm_resolve_stop_after_transforms():
    """stop_after='transforms' returns all transforms, no defaults."""
    cm = _make_cm_with_constraints()
    d = cm.resolve({'x': 1.0, 'y': 2.0}, stop_after='transforms')
    assert 'x' in d
    assert 'y' in d
    assert 'z' in d           # aliased
    assert 'w' in d           # fixed
    # Should NOT have default-only params that weren't in input
    # (there are none in this test — all names are either input or derived)


def test_cm_resolve_full_includes_defaults():
    """Full resolve merges defaults for params not in pipeline output."""
    cm = _make_cm_with_constraints()
    # 'x' is not in input, but should get default
    d = cm.resolve({'y': 2.0})
    assert d['x'] == 10.0   # from defaults
    assert d['w'] == 99.0   # from fixed
    # z is aliased from y at 'same' stage (before scale), so z != y after scale
    assert d['z'] == 2.0
    assert abs(d['y'] - 5.0) < 1e-12  # y scaled: 2*2 + 1 = 5


def test_cm_flat_resolve_stop_after_transforms():
    """flat_resolve with stop_after='transforms' via Fitter."""
    from ampfit.fitter import Fitter
    f = Fitter('config_angle.yml', backend='numpy')
    x = f.initial_values(seed=42)
    d = f.cm.flat_resolve(x, stop_after='transforms')
    assert isinstance(d, dict)


def test_cm_flat_resolve_stop_after_bounds():
    """flat_resolve with stop_after='bounds'."""
    from ampfit.fitter import Fitter
    f = Fitter('config_angle.yml', backend='numpy')
    x = f.initial_values(seed=42)
    d = f.cm.flat_resolve(x, stop_after='bounds')
    # Keys should be raw ck names, not yet resolved
    assert any('_0r' in k or '_0i' in k for k in d)


def test_cm_stop_after_roundtrip():
    """resolve(stop_after=...) + defaults should match full resolve."""
    cm = _make_cm_with_constraints()
    full = cm.resolve({'x': 1.0, 'y': 2.0})
    partial = cm.resolve({'x': 1.0, 'y': 2.0}, stop_after='transforms')
    # Merge defaults manually
    merged = dict(cm.defaults)
    merged.update(partial)
    for k in full:
        assert abs(full[k] - merged[k]) < 1e-12, f"{k} mismatch"


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
    # A_prod must stay in [-1, 1] so 1 ± A_prod ≥ 0 in probability formula
    fitter.set_range('A_prod', -1, 1)
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


# ═══════════════════════════════════════════════════════════════════
# 6. CK matrix v2 — polar convention + ref_file (standalone, no config)
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def ck_matrix_test_data(tmp_path_factory):
    """Create synthetic gamma/partial/order files for CK matrix model tests."""
    tmp = tmp_path_factory.mktemp("ck_data")
    n_ck = 3
    n_x = 50

    # Gamma table: x values + M_00 data (minimal — only needed for gamma_scale)
    x = np.linspace(0.5, 2.0, n_x)
    # Need 4 columns: x, real part of M_00, imag part, something for gamma
    gamma_data = np.zeros((n_x, 4))
    gamma_data[:, 0] = x
    gamma_data[:, 1] = 1.0 / (1.0 + (x - 1.0)**2)  # simple BW-like shape
    np.save(str(tmp / "gamma.npy"), gamma_data)

    # Partial matrix M_ab: (n_x, n_ck, n_ck) complex
    M = np.zeros((n_x, n_ck, n_ck), dtype=np.complex128)
    for i in range(n_x):
        for a in range(n_ck):
            M[i, a, a] = 1.0 + 0.1j * (a + 1)  # diag terms
            for b in range(a + 1, n_ck):
                M[i, a, b] = 0.1 * (a + 1) / (1.0 + (x[i] - 1.0)**2)
                M[i, b, a] = M[i, a, b].conj()
    np.save(str(tmp / "partial.npy"), M)

    # Order file
    order_names = [f"test_g_ls_{a}r" for a in range(n_ck)]
    with open(str(tmp / "order.json"), "w") as f:
        json.dump(order_names, f)

    return {
        "gamma_file": str(tmp / "gamma.npy"),
        "partial_file": str(tmp / "partial.npy"),
        "order_file": str(tmp / "order.json"),
        "n_ck": n_ck,
    }


def _ck_model_kwargs(ck_matrix_test_data, extra=None):
    kw = dict(ck_matrix_test_data)
    kw.pop("n_ck")
    if extra:
        kw.update(extra)
    return kw


def test_ck_matrix_v2_polar_forward(ck_matrix_test_data):
    """CK matrix v2: forward with magnitude/phase convention (r*exp(j*θ))."""
    from ampfit.particle_model.ck_matrix_v2 import CKMatrixModelV2
    kw = _ck_model_kwargs(ck_matrix_test_data, {
        "mass": 1.0, "width": 0.1,
        "ck": [2.0, 0.5, 1.5, -0.3, 3.0, 1.2],
    })
    m = CKMatrixModelV2("test", **kw)
    tr = m.make_mass_width_transform()
    d = {"test_width": 0.1}
    for a in range(tr.n_ck):
        d[tr.order_names[a]] = tr.ck_r0[a]
        d[tr.order_names[a].rstrip("r") + "i"] = tr.ck_i0[a]
    fwd = tr.forward(d)
    for name in tr.gamma_names:
        assert np.isfinite(fwd[name]), f"Non-finite gamma: {name}"


def test_ck_matrix_v2_polar_gradients(ck_matrix_test_data):
    """CK matrix v2: polar gradients match numerical (r≠1, θ≠0)."""
    from ampfit.particle_model.ck_matrix_v2 import CKMatrixModelV2
    kw = _ck_model_kwargs(ck_matrix_test_data, {
        "mass": 1.0, "width": 0.1,
        "ck": [2.0, 0.5, 1.5, -0.3, 3.0, 1.2],
    })
    m = CKMatrixModelV2("test", **kw)
    tr = m.make_mass_width_transform()

    gamma_names = set(tr.gamma_names)
    loss_fn = lambda dd: sum(v for k, v in tr.forward(dd).items() if k in gamma_names)
    d = {"test_width": 0.1}
    for a in range(tr.n_ck):
        d[tr.order_names[a]] = tr.ck_r0[a]
        d[tr.order_names[a].rstrip("r") + "i"] = tr.ck_i0[a]

    grad = tr.backward({n: 1.0 for n in tr.gamma_names}, d)
    eps = 1e-6

    for a in range(tr.n_ck):
        name = tr.order_names[a]
        iname = name.rstrip("r") + "i"
        r0, t0 = tr.ck_r0[a], tr.ck_i0[a]
        num_r = (loss_fn({**d, name: r0+eps}) - loss_fn({**d, name: r0-eps})) / (2*eps)
        num_t = (loss_fn({**d, iname: t0+eps}) - loss_fn({**d, iname: t0-eps})) / (2*eps)
        ana_r = grad.get(name, 0)
        ana_t = grad.get(iname, 0)
        assert abs(num_r - ana_r) < 1e-4, f"Mag gradient mismatch for {name}: {abs(num_r-ana_r):.2e}"
        assert abs(num_t - ana_t) < 1e-4, f"Phase gradient mismatch for {iname}: {abs(num_t-ana_t):.2e}"


def test_ck_matrix_v2_sqrt_division(ck_matrix_test_data):
    """ck_matrix_v2_sqrt divides the M table by √s = m."""
    from ampfit.particle_model.ck_matrix_v2 import (CKMatrixModelV2,
                                                    CKMatrixModelV2Sqrt)
    kw = _ck_model_kwargs(ck_matrix_test_data, {"mass": 1.0, "width": 0.1})
    base = CKMatrixModelV2("test", **kw)
    sq = CKMatrixModelV2Sqrt("test", **kw)

    # the sqrt model's table = base table / m (row-wise, zero-pad aligned)
    m_arr = np.maximum(np.abs(base.x_table), 1e-9)
    assert np.allclose(sq.M_table * m_arr[:, None, None],
                       base.M_table)
    # ... and the gamma functions at mass m carry the 1/m factor:
    # both are normalised at m₀ = 1.0 (scales agree to interpolation
    # precision), so γ_sqrt(m) ≈ γ_base(m) / m
    m = np.linspace(0.6, 1.9, 50)
    for gi_b, gi_s in zip(base.gamma(m), sq.gamma(m)):
        gi_b = np.asarray(gi_b).real
        gi_s = np.asarray(gi_s).real
        assert np.allclose(gi_s, gi_b / m, rtol=5e-3), \
            "sqrt-model gamma should carry the 1/m factor"


def test_ck_matrix_v2_ref_file(ck_matrix_test_data, tmp_path):
    """CK matrix v2: ref_file loads g_ls from reference JSON."""
    from ampfit.particle_model.ck_matrix_v2 import CKMatrixModelV2
    # Create a reference JSON with known g_ls values
    ref = {
        "test_width": 0.15,
        "test_g_ls_0r": 3.0, "test_g_ls_0i": 0.8,
        "test_g_ls_1r": 2.5, "test_g_ls_1i": -0.5,
        "test_g_ls_2r": 1.2, "test_g_ls_2i": 0.3,
    }
    ref_path = tmp_path / "ref.json"
    with open(ref_path, "w") as f:
        json.dump(ref, f)

    kw = _ck_model_kwargs(ck_matrix_test_data, {
        "mass": 1.0, "width": 0.1,
        "ref_file": str(ref_path),
    })
    m = CKMatrixModelV2("test", **kw)
    tr = m.make_mass_width_transform()

    # g_ls not in input_names (reference mode)
    gls_in = [n for n in tr.input_names if "g_ls" in n]
    assert len(gls_in) == 0, f"g_ls should not be in input_names, found {len(gls_in)}"
    assert any("width" in n for n in tr.input_names), "width should be in input_names"

    # _ref_ck populated with reference values
    assert m._ref_ck is not None
    assert len(m._ref_ck) == ck_matrix_test_data["n_ck"]
    assert abs(abs(m._ref_ck[0]) - 3.0) < 1e-6
    assert abs(np.angle(m._ref_ck[0]) - 0.8) < 1e-6

    # get_bw_params works
    bw = m.get_bw_params()
    assert "mass_bw" in bw and "width_bw" in bw

def test_ck_matrix_v2_jacobian(ck_matrix_test_data):
    """CK matrix v2: Jacobian d(gamma)/d(r) matches numerical (signed r)."""
    from ampfit.particle_model.ck_matrix_v2 import CKMatrixModelV2
    import numpy as np
    kw = _ck_model_kwargs(ck_matrix_test_data, {
        "mass": 1.0, "width": 0.1,
        "ck": [2.0, -1.5, 0.5, -0.3, -2.0, 1.2],  # include negative magnitudes
    })
    m = CKMatrixModelV2("test", **kw)
    tr = m.make_mass_width_transform()

    # Build input dict with signed r values (some negative)
    d = {"test_width": 0.1}
    for a in range(tr.n_ck):
        d[tr.order_names[a]] = tr.ck_r0[a]
        d[tr.order_names[a].rstrip("r") + "i"] = tr.ck_i0[a]

    eps = 1e-6
    max_rel = 0.0

    for a in range(tr.n_ck):
        name = tr.order_names[a]
        r0 = tr.ck_r0[a]

        # Numerical Jacobian by perturbing r_a
        dp = dict(d); dp[name] = r0 + eps
        dm = dict(d); dm[name] = r0 - eps
        fwd_p = tr.forward(dp)
        fwd_m = tr.forward(dm)

        # Analytical Jacobian from backward with unit gradient for each gamma
        for gi, gn in enumerate(tr.gamma_names):
            grad_out = {g: 1.0 if g == gn else 0.0 for g in tr.gamma_names}
            bw = tr.backward(grad_out, d)
            ana = bw.get(name, 0.0)  # d(gamma_i)/d(r_a) = backprop with unit gradient

            num = (fwd_p.get(gn, 0) - fwd_m.get(gn, 0)) / (2 * eps)
            if abs(num) < 1e-15:
                continue

            rel = abs(ana - num) / max(abs(num), 1e-30)
            if rel > max_rel:
                max_rel = rel
            assert rel < 0.02, (
                f"Jacobian mismatch for {gn} w.r.t. {name}: "
                f"ana={ana:.6e} num={num:.6e} rel={rel:.4e}")

    assert max_rel < 0.02, f"Max Jacobian relative error: {max_rel:.4e}"


# ── Display name tests ─────────────────────────────────────────

def test_particle_display_default():
    """Particle gets LaTeX display from fmt_particle."""
    from ampfit.config_loader import Particle
    cases = [
        ("a1(1260)p",  r"$a_1(1260)^+$"),
        ("a1(1260)m",  r"$a_1(1260)^-$"),
        ("f0(980)",    r"$f_0(980)$"),
        ("rhoA",       r"$\rho$"),
        ("rhoB",       r"$\rho$"),
        ("rho_omega",  r"$\rho/\omega$"),
        ("NR0",        r"$\text{NR}$"),
    ]
    for name, expected in cases:
        p = Particle(name)
        assert p.display == expected, f"{name}: got {p.display!r}, expected {expected!r}"


def test_particle_display_custom():
    """YAML ``display`` key is stored as-is (no forced $$)."""
    from ampfit.config_loader import Particle
    p = Particle("my_res", display=r"\mathrm{MyRes}")
    assert p.display == r"\mathrm{MyRes}"


def test_config_name_display_map():
    """name_display_map covers all particles in decay chains."""
    from ampfit.config_loader import Config
    c = Config("config_angle.yml")
    m = c.name_display_map()
    assert isinstance(m, dict)
    assert len(m) > 5
    # Every chain particle is included
    for chain in c.full_decay.chains:
        for decay in chain.decays[1:]:
            assert decay.core.name in m, f"{decay.core.name} missing from map"


def test_config_display_decay():
    """display_decay wraps particle names in LaTeX."""
    from ampfit.config_loader import Config
    c = Config("config_angle.yml")
    for chain in c.full_decay.chains:
        for decay in chain.decays[1:]:
            d = c.display_decay(decay)
            assert d.startswith("$") or "$" in d, f"decay display missing LaTeX: {d}"
            assert r"\to" in d, f"decay display missing \\to: {d}"
            break
        break


def test_config_display_g_ls():
    """display_g_ls returns one LaTeX label per LS combination."""
    from ampfit.config_loader import Config
    c = Config("config_angle.yml")
    for chain in c.full_decay.chains:
        for decay in chain.decays[1:]:
            labels = c.display_g_ls(decay)
            n_ls = len(decay.get_ls_list())
            assert len(labels) == n_ls, (
                f"expected {n_ls} labels, got {len(labels)}")
            for lab in labels:
                assert lab.startswith("$g^{")
            break
        break


def test_config_display_a_total():
    """display_a_total returns a LaTeX string."""
    from ampfit.config_loader import Config
    c = Config("config_angle.yml")
    for chain in c.full_decay.chains:
        d = c.display_a_total(chain)
        assert "$a_{\\mathrm{total}}" in d, f"unexpected a_total: {d}"
        break


def test_config_param_display_mass():
    """param_display formats mass parameters."""
    from ampfit.config_loader import Config
    c = Config("config_angle.yml")
    assert c.param_display("rhoA_mass") == r"$m_{\rho}$"
    assert r"_mass" not in c.param_display("f0(980)_mass")


def test_config_param_display_width():
    """param_display formats width parameters."""
    from ampfit.config_loader import Config
    c = Config("config_angle.yml")
    d = c.param_display("rhoA_width")
    assert d.startswith(r"$\Gamma")
    assert r"\rho" in d


def test_config_param_display_g_ls():
    """param_display formats g_ls magnitude and phase."""
    from ampfit.config_loader import Config
    c = Config("config_angle.yml")
    # Find a g_ls name from the config
    name_r = None
    name_i = None
    for chain in c.full_decay.chains:
        for decay in chain.decays[1:]:
            ds = str(decay).replace("+", ".")
            for idx in range(len(decay.get_ls_list())):
                name_r = f"{ds}_g_ls_{idx}r"
                name_i = f"{ds}_g_ls_{idx}i"
                break
            break
        break
    if name_r:
        dr = c.param_display(name_r)
        assert dr.startswith("$|")
        assert "g" in dr
    if name_i:
        di = c.param_display(name_i)
        assert di.startswith(r"$\arg(")


def test_config_param_display_scalar():
    """param_display formats scalar names."""
    from ampfit.config_loader import Config
    c = Config("config_angle.yml")
    assert c.param_display("gamma") == r"$\Gamma$"
    assert c.param_display("delta_gamma") == r"$\Delta\Gamma$"
    assert c.param_display("A_prod") == r"$A_{\mathrm{prod}}$"


def test_config_param_display_unknown():
    """param_display escapes underscores for unknown names."""
    from ampfit.config_loader import Config
    c = Config("config_angle.yml")
    d = c.param_display("some_unknown_param")
    assert r"\_" in d, f"underscore not escaped: {d}"
    assert r"\mathrm" in d, f"missing \\mathrm: {d}"


def test_discover_groups_display_labels():
    """discover_groups keys are particle display names.

    Charge-merged waves get ``^{\pm}``; a1(1260)+ and a1(1260)- stay
    separate (exact display names).  d1d2 groups join with `` + ``.
    """
    from ampfit.config_loader import Config
    from ampfit.plot_pw_groups import discover_groups

    c = Config("config_angle.yml")
    g = discover_groups(c)

    # Charge-merged keys use the ± superscript, not a p/m variant
    assert "$a_2(1320)^{\\pm}$" in g, f"merged a2 missing: {sorted(g)}"
    assert "$\\pi(1300)^{\\pm}$" in g
    assert "$\\pi_1(1600)^{\\pm}$" in g

    # a1(1260)+ / a1(1260)- remain separate with correct charge signs
    assert "$a_1(1260)^+$" in g
    assert "$a_1(1260)^-$" in g

    # d1d2 mode: display-name pairs joined with " + "
    assert "$\\rho$ + $\\rho$" in g
    assert "$f_0(980)$ + $\\rho$" in g

    # All keys are display names (contain LaTeX $), and every group
    # expands over all 8 CP blocks (ck count divisible by 8)
    for k, v in g.items():
        assert "$" in k, f"non-display label: {k!r}"
        assert len(v) % 8 == 0, f"group {k} not CP-expanded"


def test_discover_groups_merge_preserves_labels():
    """merge patterns apply to internal names; synthetic labels kept."""
    from ampfit.config_loader import Config
    from ampfit.plot_pw_groups import discover_groups

    c = Config("config_angle.yml")
    # No MI0 particles here → merge is a no-op, labels unchanged
    g = discover_groups(c, merge=[("^MI0\\d", "MI0")])
    assert "$a_1(1260)^+$" in g
    assert "$\\rho$ + $\\rho$" in g


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
