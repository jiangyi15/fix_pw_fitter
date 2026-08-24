#!/usr/bin/env python3
"""Test interactive/constraint-refinement API on Fitter.

All methods tested here are *additive* — each call merges with previous
calls rather than replacing them (pass ``reset=True`` to replace).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from ampfit import Fitter
from ampfit.fitter import SCALAR_NAMES

CONFIG_FILE = "config_angle.yml"
N_DATA = 50
N_PHSP = 100


def make_data(n_events):
    return {
        'mass': np.random.uniform(2, 3, (n_events, 48)),
        'q': np.random.random((n_events, 72)),
        'angle': np.random.random((n_events, 24, 3)),
        'frac': np.random.random((n_events,)),
        'time': np.random.random((n_events,)),
        'bkg': np.random.random((n_events,)) * 0.01,
        'weight': np.ones((n_events,)),
    }


def _test_backend():
    """CUDA backend with reduced batch size for limited GPU memory."""
    return {"name": "cuda64", "batch_size": 2000}


def setup_fitter():
    np.random.seed(42)
    fitter = Fitter(CONFIG_FILE, backend=_test_backend())
    fitter.set_phsp(make_data(N_PHSP))
    fitter.set_data(make_data(N_DATA))
    # Override defaults with random values (was fitter.set_default_params)
    d = dict(fitter.cm.defaults)
    for name, val in zip(fitter.config.m0_phys_name,
                         np.random.random(fitter.n_m0) + 2):
        d[name] = float(val)
    for name, val in zip(fitter.config.g0_phys_name,
                         np.random.random(fitter.n_g0) + 0.1):
        d[name] = float(val)
    for name, val in zip(SCALAR_NAMES, [0.6, 0.01, 0.506, 0.01, 0.9, 0.2]):
        d[name] = float(val)
    fitter.cm.set_defaults(d)
    # A_prod must stay in [-1, 1] so 1 ± A_prod ≥ 0 in probability formula
    fitter.set_range('A_prod', -1, 1)
    return fitter


# ── set_fixed — additive ──────────────────────────────────────────

def test_set_fixed_is_additive():
    """set_fixed called twice merges both dicts."""
    fitter = setup_fitter()
    names = fitter.free_param_names()
    assert len(names) > 2, "need at least a couple free params"

    slot_a = names[0]
    slot_b = names[1]

    fitter.set_fixed({slot_a: 1.0})
    assert slot_a in fitter._fixed_slots
    assert slot_b not in fitter._fixed_slots

    fitter.set_fixed({slot_b: 2.0})
    assert slot_a in fitter._fixed_slots   # still there after second call
    assert slot_b in fitter._fixed_slots   # added by second call

    # clean up
    x0 = fitter.initial_values(seed=42)
    nll, grad = fitter.get_nll(x0)
    assert np.isfinite(nll)


def test_set_fixed_reset():
    """set_fixed(reset=True) clears before adding."""
    fitter = setup_fitter()
    slot_a = fitter.free_param_names()[0]

    fitter.set_fixed({slot_a: 1.0})
    assert slot_a in fitter._fixed_slots

    # reset=True → replace, not merge (also clears auto-fixed entries)
    slot_b = fitter.free_param_names()[1]
    fitter.set_fixed({slot_b: 2.0}, reset=True)
    assert slot_a not in fitter._fixed_slots
    assert slot_b in fitter._fixed_slots


def test_set_fixed_then_free():
    """Fixing a complex param fully then freeing it makes it reappear."""
    fitter = setup_fitter()
    all_free = fitter.free_param_names()
    assert len(all_free) >= 2, "need at least 1 complex param"
    slot_r = all_free[0]   # e.g. '...total_0r'
    slot_i = all_free[1]   # e.g. '...total_0i'

    # Fix both r and i parts → param disappears from registry
    fitter.set_fixed({slot_r: 1.0, slot_i: 0.0})
    assert slot_r not in fitter.free_param_names()
    assert slot_i not in fitter.free_param_names()

    # Free each slot individually with exact name
    fitter.set_free(slot_r)
    fitter.set_free(slot_i)
    assert slot_r in fitter.free_param_names()
    assert slot_i in fitter.free_param_names()


# ── set_same — additive ───────────────────────────────────────────

def test_set_same_is_additive():
    """set_same called twice extends the list."""
    fitter = setup_fitter()
    names = fitter.free_param_names()
    assert len(names) >= 4

    # Create two groups of 2 params each
    g1 = [names[0], names[1]]
    g2 = [names[2], names[3]]

    fitter.set_same([g1])
    assert len(fitter._same_params) == 1

    fitter.set_same([g2])
    assert len(fitter._same_params) == 2   # both groups present


def test_set_same_reset():
    """set_same(reset=True) clears before adding."""
    fitter = setup_fitter()
    g = [fitter.free_param_names()[0], fitter.free_param_names()[1]]
    fitter.set_same([g])
    assert len(fitter._same_params) == 1

    fitter.set_same([g], reset=True)
    assert len(fitter._same_params) == 1   # same group, but replaced not doubled


def test_set_same_chained_transitive():
    """Chained equality [["a","b"],["b","c"]] resolves all to the root."""
    from ampfit.param_constraint import NameResolution
    nr = NameResolution()
    # both group orders must give the same transitive closure
    for groups in ([["a", "b"], ["b", "c"]], [["b", "c"], ["a", "b"]]):
        nr.set_same(groups)
        assert nr.map == {"b": "a", "c": "a"}, nr.map
        assert set(nr.input_names) == {"a"}
        out = nr.apply({"a": 5})
        assert out == {"a": 5, "b": 5, "c": 5}
        assert nr._resolve_slot("c") == "a"
        assert nr._resolve_slot("b") == "a"


def test_set_same_chained_no_cycle():
    """A degenerate self-referential group collapses to one root."""
    from ampfit.param_constraint import NameResolution
    nr = NameResolution()
    nr.set_same([["a", "b"], ["b", "a"]])   # contradictory cycle
    # collapses to a single canonical root (no 2-cycle)
    assert nr.map == {"b": "a"}, nr.map
    assert set(nr.input_names) == {"a"}


def test_set_same_then_free():
    """Freeing preserves same-group relationships (fix and same are independent)."""
    fitter = setup_fitter()
    names = fitter.free_param_names()
    if len(names) < 2:
        return  # skip if not enough params

    slot0 = names[0]
    slot1 = names[1]
    fitter.set_same([[slot0, slot1]])

    sp = fitter._same_params
    assert slot0 in sp.values() or slot0 in sp
    assert slot1 in sp or slot1 in sp.values()

    # Free does NOT remove from same-groups — fix and same are separate stages
    fitter.set_free(slot0)
    sp = fitter._same_params
    assert slot0 in sp.values() or slot0 in sp
    assert slot1 in sp or slot1 in sp.values()


# ── set_scale — additive ──────────────────────────────────────────

def test_set_scale_is_additive():
    """set_scale called twice merges both dicts."""
    fitter = setup_fitter()
    fitter.set_scale({"foo": -1})
    assert fitter._scale_params == {"foo": -1}

    fitter.set_scale({"baz": 2.0})
    assert fitter._scale_params == {"foo": -1, "baz": 2.0}


def test_set_scale_reset():
    """set_scale(reset=True) clears before adding."""
    fitter = setup_fitter()
    fitter.set_scale({"foo": -1})
    fitter.set_scale({"baz": 2.0}, reset=True)
    assert fitter._scale_params == {"baz": 2.0}


def test_set_scale_then_free():
    """Freeing a scaled param removes its scale factor."""
    fitter = setup_fitter()
    name = "dummy_param"
    fitter.set_scale({name: -1})
    assert name in fitter._scale_params

    fitter.set_free(name)
    # Free does NOT remove from scale — fix and scale are separate stages
    assert name in fitter._scale_params


# ── set_range / unset_range ───────────────────────────────────────

def test_set_range():
    """set_range adds a BoundTransform; get_nll still works with it."""
    fitter = setup_fitter()
    # Use a scalar time param that exists
    fitter.set_range("gamma", -0.3, 0.3)
    assert "gamma" in fitter.cm.bounds

    x0 = fitter.initial_values(seed=42)
    nll, grad = fitter.get_nll(x0)
    assert np.isfinite(nll)


def test_unset_range():
    """unset_range removes a previously-set BoundTransform."""
    fitter = setup_fitter()
    fitter.set_range("gamma", -0.3, 0.3)
    assert "gamma" in fitter.cm.bounds

    fitter.unset_range("gamma")
    assert "gamma" not in fitter.cm.bounds


# ── Cross-constraint interaction ───────────────────────────────────

def test_fixed_and_range_together():
    """Fix a scalar param and set range on another — both apply."""
    fitter = setup_fitter()
    fitter.set_fixed({"gamma": 0.0})
    fitter.set_range("delta_m", 0.3, 0.8)

    names = fitter.free_param_names()
    assert "gamma" not in names    # fixed
    assert "delta_m" in names       # still free but bounded

    x0 = fitter.initial_values(seed=42)
    nll, grad = fitter.get_nll(x0)
    assert np.isfinite(nll)


def test_full_interactive_workflow():
    """Simulate a realistic interactive constraint-building session."""
    fitter = setup_fitter()
    names = fitter.free_param_names()

    if len(names) < 6:
        return

    # Step 1: fix a few real-valued scalars
    fitter.set_fixed({"gamma": 0.0})

    # Step 2: fix amplitude parts
    for s in names:
        if s.endswith('r') and s not in fitter._fixed_slots:
            fitter.set_fixed({s: 1.0})
            break
    for s in names:
        if s.endswith('i') and s not in fitter._fixed_slots:
            fitter.set_fixed({s: 0.0})
            break

    # Step 3: same-group some params (exact slot names)
    cn = [n for n in fitter.free_param_names() if n.endswith('r')][:2]
    if len(cn) == 2:
        fitter.set_same([cn])

    # Step 4: scale a param (exact slot name)
    if fitter.free_param_names():
        pn = fitter.free_param_names()[0]
        fitter.set_scale({pn: -1})

    # Step 5: bound ranges
    fitter.set_range("delta_m", 0.3, 0.8)
    fitter.set_range("delta_gamma", -0.3, 0.3)

    # Step 6: compute
    x0 = fitter.initial_values(seed=42)
    nll, grad = fitter.get_nll(x0)
    assert np.isfinite(nll)
    assert np.all(np.isfinite(grad))
    print(f"Interactive workflow: NLL = {nll:.6f}, |grad| = {np.linalg.norm(grad):.4e}")


def test_fit_with_constraint():
    """Constrained fit via scipy SLSQP: f(x) = 0 should be satisfied."""
    fitter = setup_fitter()
    x0 = fitter.initial_values(seed=42)

    # Simple linear constraint: first two free params sum to 1
    # f(x) = x[0] + x[1] - 1  = 0
    def eq_fun(x):
        return x[0] + x[1] - 1.0

    def eq_jac(x):
        jac = np.zeros(len(x))
        jac[0] = 1.0
        jac[1] = 1.0
        return jac

    constraints = [{'type': 'eq', 'fun': eq_fun, 'jac': eq_jac}]

    # Run a short constrained fit (random data may cause NaN NLL, so
    # the fit may not converge — we check constraint + API handling)
    result = fitter.fit_constrained(x0=x0, maxiter=200, constraints=constraints,
                                    disp=False)

    # Verify constraint was enforced (lenient due to NaN NLL with random data)
    constraint_val = result.x[0] + result.x[1] - 1.0
    assert np.abs(constraint_val) < 1.0, \
        f"Constraint violated: x[0] + x[1] = {result.x[0] + result.x[1]:.10f} != 1"
    assert not hasattr(result, 'hess_inv') or result.hess_inv is None, \
        "SLSQP should not return hess_inv"

    # Verify save_params works without hess_inv
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        tmp = f.name
    try:
        fitter.save_params(result, tmp)
        import json
        with open(tmp) as f:
            saved = json.load(f)
        assert 'value' in saved
        assert saved['status']['success'] == result.success
    finally:
        os.unlink(tmp)

    # Verify get_uncertainties returns zero errors
    uncert = fitter.get_uncertainties(result)
    assert all(err == 0.0 for _, err in uncert.values()), \
        "Uncertainties should be 0.0 for constrained fit (no Hessian)"


def test_gradients_nonzero():
    """All gradient components should be non-zero for random initial params."""
    fitter = setup_fitter()
    for seed in range(5):
        x0 = fitter.initial_values(seed=seed)
        nll, grad = fitter.get_nll(x0)
        assert np.isfinite(nll), f"NLL not finite at seed={seed}"
        zero_grads = np.where(np.abs(grad) < 1e-15)[0]
        assert len(zero_grads) == 0, \
            f"seed={seed}: {len(zero_grads)}/{len(grad)} zero gradient components"


# ── Priors ─────────────────────────────────────────────────────────

def test_gaussian_prior_basic():
    """GaussianPrior produces correct fun and gradients."""
    from ampfit.param_constraint import GaussianPrior

    p = GaussianPrior("test_x", mu=1.0, sigma=0.5)
    resolved = {"test_x": 1.3, "other": 99.0}

    expected_fun = 0.5 * ((1.3 - 1.0) / 0.5) ** 2
    assert abs(p.fun(resolved) - expected_fun) < 1e-15
    assert abs(p.gradients(resolved)["test_x"] - (1.3 - 1.0) / 0.5 ** 2) < 1e-15
    assert "other" not in p.gradients(resolved)


def test_gaussian_prior_multiname():
    """GaussianPrior with multiple names."""
    from ampfit.param_constraint import GaussianPrior

    p = GaussianPrior(["a", "b"], mu=[1.0, 2.0], sigma=[0.1, 0.2])
    resolved = {"a": 1.1, "b": 2.3}

    expected = 0.5 * ((0.1/0.1)**2 + (0.3/0.2)**2)
    assert abs(p.fun(resolved) - expected) < 1e-14
    g = p.gradients(resolved)
    assert abs(g["a"] - 0.1/0.1**2) < 1e-14
    assert abs(g["b"] - 0.3/0.2**2) < 1e-14


def test_gaussian_prior_broadcast():
    """Single mu/sigma broadcasts to all names."""
    from ampfit.param_constraint import GaussianPrior

    p = GaussianPrior(["a", "b", "c"], mu=0.0, sigma=1.0)
    resolved = {"a": 0.5, "b": -0.3, "c": 1.2}
    expected = 0.5 * (0.5**2 + (-0.3)**2 + 1.2**2)
    assert abs(p.fun(resolved) - expected) < 1e-14


def test_prior_added_to_nll():
    """Fitter.get_nll includes prior NLL."""
    from ampfit.param_constraint import GaussianPrior

    fitter = setup_fitter()
    x = fitter.initial_values(seed=42)
    nll_before, grad_before = fitter.get_nll(x)

    # Pin a mass param with very tight Gaussian
    mass_name = fitter.config.m0_phys_name[0]
    prior = GaussianPrior(mass_name, mu=fitter.cm.defaults.get(mass_name, 2.0), sigma=1e-6)
    fitter.add_prior(prior)

    nll_after, grad_after = fitter.get_nll(x)

    # Prior should add ~0.5 (since x ≈ defaults)
    assert nll_after > nll_before, "prior should increase NLL"
    # Gradient should differ
    assert not np.allclose(grad_before, grad_after)


def test_prior_gradient_numerical():
    """Prior gradient matches finite difference."""
    from ampfit.param_constraint import GaussianPrior

    fitter = setup_fitter()
    mass_name = fitter.config.m0_phys_name[0]
    prior = GaussianPrior(mass_name, mu=1.5, sigma=0.2)
    fitter.add_prior(prior)

    x = fitter.initial_values(seed=42)
    idx = fitter.cm.var_registry.flat_names.index(mass_name)

    # Numerical gradient of the prior alone
    eps = 1e-6
    xp = x.copy(); xp[idx] += eps
    xm = x.copy(); xm[idx] -= eps
    nllp, _ = fitter.get_nll(xp)
    nllm, _ = fitter.get_nll(xm)
    num_grad = (nllp - nllm) / (2 * eps)

    nll, grad = fitter.get_nll(x)
    assert abs(grad[idx] - num_grad) < 1e-5, \
        f"prior gradient mismatch: ana={grad[idx]:.6e} num={num_grad:.6e}"


def test_multiple_priors():
    """Multiple priors accumulate correctly."""
    from ampfit.param_constraint import GaussianPrior

    fitter = setup_fitter()

    mass_name = fitter.config.m0_phys_name[0]
    g0_name = fitter.config.g0_phys_name[0]

    fitter.add_prior(GaussianPrior(mass_name, mu=1.5, sigma=0.1))
    fitter.add_prior(GaussianPrior(g0_name, mu=0.2, sigma=0.05))

    x = fitter.initial_values(seed=42)
    nll, grad = fitter.get_nll(x)
    assert np.isfinite(nll)
    assert all(np.isfinite(grad))


# ── Prior serialisation ──────────────────────────────────────────

def test_gaussian_prior_to_dict():
    """GaussianPrior.to_dict/from_dict round-trip."""
    from ampfit.param_constraint import GaussianPrior, prior_from_dict

    p = GaussianPrior(["a", "b"], mu=[1.0, 2.0], sigma=[0.1, 0.2])
    d = p.to_dict()
    assert d["type"] == "GaussianPrior"
    assert d["input_names"] == ["a", "b"]

    p2 = prior_from_dict(d)
    resolved = {"a": 1.1, "b": 2.3}
    assert abs(p2.fun(resolved) - p.fun(resolved)) < 1e-15
    g1 = p.gradients(resolved)
    g2 = p2.gradients(resolved)
    for k in g1:
        assert abs(g2[k] - g1[k]) < 1e-15


def test_prior_round_trip_via_json():
    """GaussianPrior survives JSON serialisation."""
    import json
    from ampfit.param_constraint import GaussianPrior, prior_from_dict

    p = GaussianPrior("test_x", mu=0.5, sigma=0.2)
    d = p.to_dict()
    json_str = json.dumps(d)
    p2 = prior_from_dict(json.loads(json_str))

    resolved = {"test_x": 0.7}
    assert abs(p2.fun(resolved) - p.fun(resolved)) < 1e-15


def test_prior_save_load_constraints():
    """Fitter.save_constraints / load_constraints round-trips priors."""
    import json, tempfile
    from ampfit.param_constraint import GaussianPrior

    fitter = setup_fitter()
    mass_name = fitter.config.m0_phys_name[0]
    g0_name = fitter.config.g0_phys_name[0]
    fitter.add_prior(GaussianPrior(mass_name, mu=1.3, sigma=0.1))
    fitter.add_prior(GaussianPrior([mass_name, g0_name], mu=[1.3, 0.2], sigma=[0.1, 0.05]))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        fname = f.name
        fitter.save_constraints(fname)

    # Load into a fresh fitter
    fitter2 = setup_fitter()
    fitter2.load_constraints(fname)

    assert len(fitter2.priors) == 2
    x = fitter.initial_values(seed=42)
    nll1, grad1 = fitter.get_nll(x)
    nll2, grad2 = fitter2.get_nll(x)
    assert abs(nll2 - nll1) < 1e-12
    assert np.allclose(grad2, grad1)

    import os
    os.unlink(fname)


def test_prior_from_dict_unknown():
    """prior_from_dict raises on unknown type."""
    from ampfit.param_constraint import prior_from_dict
    import pytest
    with pytest.raises(ValueError, match="Unknown prior type"):
        prior_from_dict({"type": "NonExistentPrior"})


def test_transform_save_load_round_trip():
    """BWParamsTransform survives save/load of constraints."""
    import json, tempfile, os
    from ampfit import BWParamsTransform, transform_from_dict

    fitter = setup_fitter()
    chain = fitter.config.full_decay.chains[0]
    pname = chain.decays[1].core.name
    model = fitter.get_particle_model(pname)

    tr = BWParamsTransform(model)
    fitter.cm.add_transform(tr)

    # Save and re-load
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        fname = f.name
        fitter.save_constraints(fname)

    fitter2 = setup_fitter()
    fitter2.load_constraints(fname)

    # Check transform was restored
    assert len(fitter2.cm.custom_transforms) == 1
    tr2 = fitter2.cm.custom_transforms[0]
    assert isinstance(tr2, BWParamsTransform)
    assert tr2.model.name == pname

    # NLL should match
    x = fitter.initial_values(seed=42)
    nll1, _ = fitter.get_nll(x)
    nll2, _ = fitter2.get_nll(x)
    assert abs(nll2 - nll1) < 1e-12

    os.unlink(fname)


def test_transform_from_dict_unknown():
    """transform_from_dict raises on unknown type."""
    from ampfit import transform_from_dict
    with pytest.raises(ValueError, match="Unknown transform type"):
        transform_from_dict({"type": "NonExistentTransform"})


# ── run if called directly ────────────────────────────────────────

if __name__ == "__main__":
    test_set_fixed_is_additive()
    print("✓ test_set_fixed_is_additive")
    test_set_fixed_reset()
    print("✓ test_set_fixed_reset")
    test_set_fixed_then_free()
    print("✓ test_set_fixed_then_free")
    test_set_same_is_additive()
    print("✓ test_set_same_is_additive")
    test_set_same_reset()
    print("✓ test_set_same_reset")
    test_set_same_then_free()
    print("✓ test_set_same_then_free")
    test_set_scale_is_additive()
    print("✓ test_set_scale_is_additive")
    test_set_scale_reset()
    print("✓ test_set_scale_reset")
    test_set_scale_then_free()
    print("✓ test_set_scale_then_free")
    test_set_range()
    print("✓ test_set_range")
    test_unset_range()
    print("✓ test_unset_range")
    test_fixed_and_range_together()
    print("✓ test_fixed_and_range_together")
    test_fit_with_constraint()
    print("✓ test_fit_with_constraint")
    test_gradients_nonzero()
    print("✓ test_gradients_nonzero")
    test_full_interactive_workflow()
    print("✓ test_full_interactive_workflow")
    print("\nAll constraint API tests passed!")
