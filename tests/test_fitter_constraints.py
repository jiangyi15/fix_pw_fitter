#!/usr/bin/env python3
"""Test interactive/constraint-refinement API on Fitter.

All methods tested here are *additive* — each call merges with previous
calls rather than replacing them (pass ``reset=True`` to replace).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from ampfit import Fitter

CONFIG_FILE = "config_angle.yml"
N_DATA = 100
N_PHSP = 200


def make_data(n_events):
    return {
        'mass': np.random.random((n_events, 48)),
        'q': np.random.random((n_events, 72)),
        'angle': np.random.random((n_events, 24, 3)),
        'frac': np.random.random((n_events,)),
        'time': np.random.random((n_events,)),
        'bkg': np.random.random((n_events,)) * 0.01,
        'weight': np.ones((n_events,)),
    }


def setup_fitter():
    fitter = Fitter(CONFIG_FILE)
    fitter.set_phsp(make_data(N_PHSP))
    fitter.set_data(make_data(N_DATA))
    fitter.set_default_params(
        m0=np.random.random(fitter.n_m0) + 2,
        g0=np.random.random(fitter.n_g0) + 0.1,
        scalar=[0.6, 0.01, 0.506, 0.01, 0.9, 0.2],
    )
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


def test_set_same_then_free():
    """Freeing a member of a same-group removes it from the group."""
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

    fitter.set_free(slot0)
    sp = fitter._same_params
    assert slot0 not in sp.values() and slot0 not in sp
    assert slot1 not in sp and slot1 not in sp.values()


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
    assert name not in fitter._scale_params


# ── set_range / unset_range ───────────────────────────────────────

def test_set_range():
    """set_range adds a BoundTransform; get_nll still works with it."""
    fitter = setup_fitter()
    # Use a scalar time param that exists
    fitter.set_range("gamma", -0.3, 0.3)
    assert len(fitter._bound_transforms) >= 1

    x0 = fitter.initial_values(seed=42)
    nll, grad = fitter.get_nll(x0)
    assert np.isfinite(nll)


def test_unset_range():
    """unset_range removes a previously-set BoundTransform."""
    fitter = setup_fitter()
    fitter.set_range("gamma", -0.3, 0.3)
    assert len(fitter._bound_transforms) >= 1

    fitter.unset_range("gamma")
    idxs_with_gamma = [i for i, n in enumerate(fitter._var_registry.flat_names)
                       if n == "gamma"]
    for i in idxs_with_gamma:
        assert i not in fitter._bound_transforms


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
    test_gradients_nonzero()
    print("✓ test_gradients_nonzero")
    test_full_interactive_workflow()
    print("✓ test_full_interactive_workflow")
    print("\nAll constraint API tests passed!")
