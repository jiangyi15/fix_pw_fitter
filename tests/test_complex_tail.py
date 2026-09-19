"""Configurable complex tail: suffixes naming a complex ck's mag/phase slots.

The default stays ``("r", "i")``; ``scope(complex_tail=("rho", "phi"))``
selects alternative suffixes.  The pair is captured at build time (the
kernel-params transform), so a fit reuses the cached transform.
"""
import os

import numpy as np

from tabpwa.amp_model import build_amplitude_model
from tabpwa.build_defaults import DEFAULTS, scope
from tabpwa.param_constraint import CKProduct, complex_tail, phase_name

CFG = os.path.join(os.path.dirname(__file__), "config_pwa.yml")


def _bases(model):
    return sorted({p for comb in model.get_ck_map()
                   for p in comb if isinstance(p, str)})


# ── helpers ───────────────────────────────────────────────────────

def test_helpers():
    assert phase_name("xr") == "xi"
    assert phase_name("xrho", tail=("rho", "phi")) == "xphi"
    assert phase_name("plain", tail=("rho", "phi")) == "plain"
    assert complex_tail() == ("r", "i")
    assert DEFAULTS["complex_tail"] == ("r", "i")


# ── default tail ──────────────────────────────────────────────────

def test_default_tail_param_names():
    model = build_amplitude_model(CFG)
    names = model.build_params_transform().param_names()
    assert complex_tail() == ("r", "i")
    assert any(n.endswith("r") for n in names)
    assert any(n.endswith("i") for n in names)
    # every magnitude slot pairs with a phase slot
    for n in names:
        if n.endswith("r") and n[:-1] + "r" == n:
            assert phase_name(n) in names


# ── scoped tail ───────────────────────────────────────────────────

def test_scoped_tail_param_names():
    with scope(complex_tail=("rho", "phi")):
        assert complex_tail() == ("rho", "phi")
        model = build_amplitude_model(CFG)
        names = model.build_params_transform().param_names()
        assert any(n.endswith("rho") for n in names)
        assert any(n.endswith("phi") for n in names)
        for n in names:
            if n.endswith("rho"):
                assert phase_name(n, ("rho", "phi")) in names
    assert complex_tail() == ("r", "i")


def test_fitter_captures_tail_at_build():
    """The tail is baked into the Fitter's cached transform, not re-read."""
    from tabpwa import Fitter

    with scope(complex_tail=("rho", "phi")):
        fitter = Fitter(CFG, backend="numpy_pwa")
    names = fitter.param_names()
    assert any(n.endswith("rho") for n in names)
    assert any(n.endswith("phi") for n in names)
    # global default restored, but the fitter keeps its build-time tail
    assert complex_tail() == ("r", "i")


# ── forward / backward under the rho/phi tail ─────────────────────

def test_forward_backward_rho_phi():
    model = build_amplitude_model(CFG)
    tail = ("rho", "phi")
    pc = CKProduct(model.get_ck_map(), tail=tail)

    rng = np.random.RandomState(0)
    slot = {}
    for b in _bases(model):
        slot[b + "rho"] = rng.uniform(0.5, 2.0)
        slot[b + "phi"] = rng.uniform(-np.pi, np.pi)

    ck = pc.build_ck(slot)
    assert ck.shape == (pc.n_wave,)
    assert np.all(np.isfinite(ck.real)) and np.all(np.isfinite(ck.imag))

    grad_ck = rng.randn(pc.n_wave) + 1j * rng.randn(pc.n_wave)
    grads = pc.backprop_grad(slot, grad_ck)
    assert grads
    assert all(k.endswith(("rho", "phi")) for k in grads)
    assert any(k.endswith("rho") for k in grads)
    assert any(k.endswith("phi") for k in grads)
