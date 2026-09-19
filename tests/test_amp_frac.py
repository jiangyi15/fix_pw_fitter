"""AmplitudeFractions: kernel-param handling is delegated to the transform."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CONFIG = os.path.join(ROOT, "tests", "config_pwa.yml")


class _StubResult:
    """Minimal OptimizeResult stand-in (only ``x`` / ``hess_inv`` are read)."""

    def __init__(self, x):
        self.x = x
        self.hess_inv = None


def _make():
    from tabpwa import Fitter
    from tabpwa.amp_frac import AmplitudeFractions

    f = Fitter(CONFIG, backend="numpy_pwa")
    f.apply_constrains()
    x = f.initial_values(seed=3)
    af = AmplitudeFractions(f, _StubResult(x), data=object())
    return f, af


def test_default_param_names_are_the_free_variables():
    f, af = _make()
    try:
        names = af._default_param_names()
        assert names == list(f.cm.var_registry.flat_names)
        # m0/g0 must be included iff they are actually free.  This config
        # fixes them, so the old code's unconditional re-listing of
        # m0_phys_name/g0_phys_name would have (wrongly) propagated them.
        m0g0 = set(f.model.m0_phys_name) | set(f.model.g0_phys_name)
        assert m0g0.isdisjoint(names)
    finally:
        f.backend.free()


def test_phys_to_params_matches_the_transform_baseline():
    f, af = _make()
    try:
        base = af._phys_to_params({})
        assert set(base) >= {"ck", "m0", "g0"}
        assert np.allclose(base["ck"], af._params["ck"])
        assert np.allclose(base["m0"], af._params["m0"])
        assert np.allclose(base["g0"], af._params["g0"])
    finally:
        f.backend.free()


def test_phys_to_params_writes_m0_slot():
    f, af = _make()
    try:
        names = f.model.m0_phys_name
        name = names[0]
        idx = names.index(name)
        base = af._phys_to_params({})
        p = af._phys_to_params({name: 1.234})
        assert p["m0"][idx] == 1.234
        assert np.allclose(np.delete(p["m0"], idx), np.delete(base["m0"], idx))
    finally:
        f.backend.free()


def test_phys_to_params_now_applies_ck_overrides():
    """ck overrides used to be silently ignored; they must now take effect."""
    f, af = _make()
    try:
        m0g0 = set(f.model.m0_phys_name) | set(f.model.g0_phys_name)
        ck_names = [n for n in af._resolved
                    if n not in m0g0 and n.endswith("r")]
        assert ck_names, "expected some ck parameter names"
        name = ck_names[0]
        base = af._phys_to_params({})
        p = af._phys_to_params({name: float(af._resolved[name]) + 0.75})
        assert not np.allclose(p["ck"], base["ck"])
    finally:
        f.backend.free()


def test_compute_masks_ck_gradient_and_passes_other_groups_through(monkeypatch):
    f, af = _make()
    try:
        fake = {
            "ck": np.ones(af._n_ck, dtype=complex),
            "m0": np.zeros(len(f.model.m0_phys_name)),
            "g0": np.zeros(len(f.model.g0_phys_name)),
        }
        monkeypatch.setattr(f.backend, "compute",
                            lambda p, d, norm=None: (2.0, dict(fake), None))

        seen = {}
        real_backward = f._kernel_builder.backward

        def spy(grads, resolved):
            seen["keys"] = set(grads)
            seen["ck"] = np.asarray(grads["ck"]).copy()
            return real_backward(grads, resolved)

        monkeypatch.setattr(f._kernel_builder, "backward", spy)

        total, _ = af._compute_total_and_grad(af._params, mask=[0])
        assert total == 2.0
        # it must not invent parameter groups (e.g. a fake "scalar")
        assert seen["keys"] == {"ck", "m0", "g0"}
        # inside the mask kept, outside the mask zeroed
        assert seen["ck"][0] == 1.0
        assert all(seen["ck"][i] == 0.0 for i in range(1, af._n_ck))
    finally:
        f.backend.free()
