#!/usr/bin/env python3
"""Tests for the toy |A|² event generator.

Run with::

    pytest tests/test_toy_generator.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.toy_generator import (
    TAU_B0_PDG, TIME_MAX, GAMMA_B0_PDG, _build_params, draw_extra,
    toy_generate,
)


class TestHelpers:
    def test_draw_extra_is_a_numpy_call(self):
        """``random:`` IS ``np.random.<dist>(size=n, **kwargs)``."""
        rng = np.random.default_rng(6)
        u = draw_extra(5000, rng, {"dist": "uniform", "low": 2.0, "high": 3.0})
        assert u.min() >= 2.0 and u.max() <= 3.0
        c = draw_extra(20000, rng, {"dist": "choice", "a": [0.0, 1.0]})
        assert set(np.unique(c)) == {0.0, 1.0}
        assert 0.45 < c.mean() < 0.55
        n = draw_extra(5000, rng, {"dist": "standard_normal"})
        assert abs(n.mean()) < 0.1 and abs(n.std() - 1.0) < 0.1
        assert np.all(draw_extra(4, rng, None, default=7.0) == 7.0)

    def test_draw_extra_window_and_lifetime(self):
        """The two laws the configs use: U(0, TIME_MAX) and the PDG exponential."""
        rng = np.random.default_rng(4)
        t = draw_extra(200_000, rng,
                       {"dist": "uniform", "low": 0.0, "high": TIME_MAX})
        assert t.min() >= 0.0 and t.max() <= TIME_MAX
        assert abs(t.mean() - TIME_MAX / 2) < 0.05
        e = draw_extra(50_000, rng,
                       {"dist": "exponential", "scale": TAU_B0_PDG})
        assert e.min() >= 0.0
        assert abs(e.mean() - TAU_B0_PDG) < 0.1 * TAU_B0_PDG

    def test_draw_extra_refusals(self):
        rng = np.random.default_rng(7)
        with pytest.raises(ValueError, match="RandomState method"):
            draw_extra(4, rng, {"dist": "not_a_distribution"})
        with pytest.raises(ValueError, match="rejected by numpy"):
            draw_extra(4, rng, {"dist": "uniform", "nope": 1})
        with pytest.raises(ValueError, match="'dist' key"):
            draw_extra(4, rng, {"low": 0.0, "high": 1.0})

    def test_build_params_nonzero(self):
        """Defaults produce a non-trivial amplitude (ck != 0)."""
        f = Fitter("tests/config_angle.yml", backend="numpy")
        params = _build_params(f)
        assert np.max(np.abs(params["ck"])) > 0


class TestToyGenerate:
    def test_toy_pipeline(self):
        """End-to-end: phase space -> data -> |A|² -> rejection."""
        f = Fitter("tests/config_angle.yml", backend="numpy")
        ev = toy_generate(f, 30, seed=5, compute_batch=3000)
        assert len(ev["momenta"]) == 30
        d = ev["data"]
        assert d["frac"].min() >= 0 and d["frac"].max() <= 1
        assert d["time"].min() >= 0
        assert np.all(np.isfinite(ev["P"]))
        # mass/q/angles have the expected shapes
        assert d["mass"].shape == (30, 48)
        assert d["q"].shape == (30, 72)
        assert d["angle"].shape == (30, 24, 3)

    def test_reproducible(self):
        f = Fitter("tests/config_angle.yml", backend="numpy")
        a = toy_generate(f, 30, seed=9, compute_batch=3000)
        f2 = Fitter("tests/config_angle.yml", backend="numpy")
        b = toy_generate(f2, 30, seed=9, compute_batch=3000)
        assert np.array_equal(a["momenta"], b["momenta"])
        assert np.array_equal(a["P"], b["P"])


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
