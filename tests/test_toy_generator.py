#!/usr/bin/env python3
"""Tests for the toy |A|² event generator.

Run with::

    pytest tests/test_toy_generator.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ampfit import Fitter
from ampfit.toy_generator import (
    toy_generate, _build_params, _make_frac_time,
    TAU_B0_PDG, GAMMA_B0_PDG,
)


class TestHelpers:
    def test_frac_random_binary(self):
        """frac='random' is a binary {0, 1} tag."""
        rng = np.random.default_rng(1)
        frac, _ = _make_frac_time(20000, "random", None, rng)
        assert set(np.unique(frac)) <= {0.0, 1.0}
        assert 0.45 < frac.mean() < 0.55

    def test_time_exp(self):
        """time='exp' is exponential with the PDG B0 width."""
        rng = np.random.default_rng(2)
        _, time = _make_frac_time(50000, None, "exp", rng)
        assert time.min() >= 0
        assert abs(time.mean() - TAU_B0_PDG) < 0.1 * TAU_B0_PDG

    def test_scalar_passthrough(self):
        rng = np.random.default_rng(3)
        frac, time = _make_frac_time(10, 0.3, 2.0, rng)
        assert np.all(frac == 0.3)
        assert np.all(time == 2.0)

    def test_build_params_nonzero(self):
        """Defaults produce a non-trivial amplitude (ck != 0)."""
        f = Fitter("config_angle.yml", backend="numpy")
        params = _build_params(f)
        assert np.max(np.abs(params["ck"])) > 0


class TestToyGenerate:
    def test_toy_pipeline(self):
        """End-to-end: phase space -> data -> |A|² -> rejection."""
        f = Fitter("config_angle.yml", backend="numpy")
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
        f = Fitter("config_angle.yml", backend="numpy")
        a = toy_generate(f, 30, seed=9, compute_batch=3000)
        f2 = Fitter("config_angle.yml", backend="numpy")
        b = toy_generate(f2, 30, seed=9, compute_batch=3000)
        assert np.array_equal(a["momenta"], b["momenta"])
        assert np.array_equal(a["P"], b["P"])


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
