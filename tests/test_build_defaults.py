"""Global build context: global attributes + temporary scopes, isolation.

Config files stay purely physics — build-time defaults (table sampling,
barrier radius / type) are read from the context.
"""
import copy

import numpy as np
import pytest

import tabpwa
from tabpwa.build_defaults import scope, get

CFG = "tests/config_pwa.yml"


# ── the context itself ────────────────────────────────────────────

def test_builtin_defaults_and_get():
    assert get("n_interp") == 2000
    assert get("d") == 3.0
    assert get("barrier") == "bw"
    assert get("missing", 7) == 7


def test_context_scopes_and_nests():
    with scope(n_interp=123):
        assert get("n_interp") == 123
        with scope(d=1.5):
            assert get("n_interp") == 123 and get("d") == 1.5
        assert get("d") == 3.0            # inner override released
    assert get("n_interp") == 2000        # outer override released


def test_back_to_global_after_with():
    assert get("n_interp") == 2000 and get("d") == 3.0
    with scope(n_interp=4000, d=1.5):
        assert get("n_interp") == 4000 and get("d") == 1.5
    # outside the scope -> back to the global defaults
    assert get("n_interp") == 2000 and get("d") == 3.0


def test_context_is_isolated_across_threads():
    import threading

    seen = {}

    def worker():
        seen["before"] = get("d")
        with scope(d=9.0):
            seen["inside"] = get("d")
        seen["after"] = get("d")

    with scope(d=1.5):
        t = threading.Thread(target=worker)
        t.start()
        t.join()
    assert seen == {"before": 3.0, "inside": 9.0, "after": 3.0}


# ── build integration ─────────────────────────────────────────────

def test_n_interp_controls_table_sampling():
    from tabpwa.amp_model import build_amplitude_model
    from tabpwa.config_loader import load_config

    dic: dict = copy.deepcopy(load_config(CFG))
    with scope(n_interp=1234):
        c = build_amplitude_model(dic).build_kernel_config()
    assert c["fl_table"].shape[1] == 1234
    assert c["gamma_table"].shape[1] == 1234
    # without the override -> built-in default
    c2 = build_amplitude_model(dic).build_kernel_config()
    assert c2["fl_table"].shape[1] == 2000


def test_context_d_hits_decays_and_particles():
    from tabpwa.amp_model import build_amplitude_model
    from tabpwa.config_loader import load_config

    dic: dict = copy.deepcopy(load_config(CFG))
    with scope(d=1.5):
        m = build_amplitude_model(dic)
        c = m.build_kernel_config()
    assert all(b.d == 1.5 for b in m.fl_forms)          # decay vertices
    models = [getattr(d.core, "_model", None)
              for ch in m.decay_tree.full.chains for d in ch.decays]
    assert all(mm.kwargs.get("d") == 1.5 for mm in models if mm is not None)


def test_context_barrier_type():
    from tabpwa.amp_model import build_amplitude_model
    from tabpwa.config_loader import load_config

    dic: dict = copy.deepcopy(load_config(CFG))
    with scope(barrier="exp"):
        m = build_amplitude_model(dic)
        m.build_kernel_config()
    assert {b.name for b in m.fl_forms} == {"exp"}


def test_context_d_overridden_per_decay_and_particle():
    from tabpwa.amp_model import build_amplitude_model
    from tabpwa.config_loader import load_config

    dic: dict = copy.deepcopy(load_config(CFG))
    dic["decay"]["jpsi"][0] = list(dic["decay"]["jpsi"][0]) + [{"d": 2.0}]
    with scope(d=1.5):
        m = build_amplitude_model(dic)
        m.build_kernel_config()
    ds = {b.d for b in m.fl_forms}
    assert 2.0 in ds and 1.5 in ds                       # explicit wins

    from tabpwa.particle_model import build_particle
    with scope(d=1.5):
        mod = build_particle("R", model="GS_rho", mass=0.775, width=0.149,
                             L=1, d=0.9)
    assert mod.kwargs["d"] == 0.9                        # explicit wins


def test_config_defaults_loaded_as_temporary_context():
    from tabpwa.amp_model import build_amplitude_model
    from tabpwa.config_loader import load_config

    dic: dict = copy.deepcopy(load_config(CFG))
    dic["defaults"] = {"d": 1.5, "n_interp": 400, "barrier": "exp"}
    m = build_amplitude_model(dic)
    c = m.build_kernel_config()
    assert all(b.d == 1.5 for b in m.fl_forms)
    assert c["fl_table"].shape[1] == 400
    assert {b.name for b in m.fl_forms} == {"exp"}
    models = [getattr(d.core, "_model", None)
              for ch in m.decay_tree.full.chains for d in ch.decays]
    assert all(mm.kwargs.get("d") == 1.5 for mm in models if mm is not None)
    # temporary: not leaked into the global context ...
    assert get("d") == 3.0
    assert get("n_interp") == 2000
    # ... and a fresh model without defaults is unaffected
    m2 = build_amplitude_model(copy.deepcopy(load_config(CFG)))
    c2 = m2.build_kernel_config()
    assert all(b.d == 3.0 for b in m2.fl_forms)
    assert c2["fl_table"].shape[1] == 2000


def test_config_defaults_override_global_context():
    from tabpwa.amp_model import build_amplitude_model
    from tabpwa.config_loader import load_config

    dic: dict = copy.deepcopy(load_config(CFG))
    dic["defaults"] = {"d": 1.5}
    with scope(d=2.5):
        m = build_amplitude_model(dic)
        c = m.build_kernel_config()
        assert all(b.d == 1.5 for b in m.fl_forms)   # config wins in its build
        assert get("d") == 2.5                        # outer context untouched


def test_context_d_reaches_particle_model():
    from tabpwa.particle_model import build_particle

    kw = dict(model="GS_rho", mass=0.775, width=0.149, L=1,
              daug2Mass=0.13957039, daug3Mass=0.1349768)
    with scope(d=1.5):
        mod = build_particle("R", **kw)
    with scope(d=3.0):
        ref = build_particle("R", **kw)
    assert mod.kwargs["d"] == 1.5
    m = np.array(0.8)
    assert not np.allclose(mod.gamma(m), ref.gamma(m))
