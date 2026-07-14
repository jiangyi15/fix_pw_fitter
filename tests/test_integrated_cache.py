#!/usr/bin/env python3
"""Tests for IntegratedBackend Gram matrix cache file.

Run with::

    pytest tests/test_integrated_cache.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ampfit.config_loader import Config
from ampfit.backends.integrated_backend import IntegratedBackend
from ampfit.fitter import Fitter

CONFIG_FILE = "config_angle.yml"


@pytest.fixture(scope="module")
def kernel_config():
    return Config(CONFIG_FILE).build_all_index()


@pytest.fixture(scope="module")
def params():
    f = Fitter(CONFIG_FILE, backend="numpy")
    x = f.initial_values(seed=42)
    _, _, _, _ = f._build_params(x)
    # Rebuild with new fitter to avoid GPU context issues
    f2 = Fitter(CONFIG_FILE, backend="numpy")
    x2 = f2.initial_values(seed=42)
    p, _, _, _ = f2._build_params(x2)
    return p


@pytest.fixture
def phsp():
    rng = np.random.default_rng(42)
    n = 200
    return {
        "mass": rng.uniform(0.3, 5.1, (n, 48)).astype(np.float64),
        "q": rng.uniform(0, 1, (n, 72)).astype(np.float64),
        "angle": rng.uniform(-np.pi, np.pi, (n, 24, 3)).astype(np.float64),
        "frac": rng.random(n).astype(np.float64),
        "time": rng.random(n).astype(np.float64),
        "weight": np.ones(n, dtype=np.float64),
    }


# ═══════════════════════════════════════════════════════════════════
# 1. Constructor accepts cache_file
# ═══════════════════════════════════════════════════════════════════

class TestConstructor:
    """IntegratedBackend cache_file parameter."""

    def test_cache_file_none_default(self, kernel_config):
        """Default cache_file is None."""
        be = IntegratedBackend(kernel_config, base="numpy")
        assert be._cache_file is None
        be.free()

    def test_cache_file_string(self, kernel_config):
        """String cache_file is stored."""
        be = IntegratedBackend(kernel_config, base="numpy",
                               cache_file="/tmp/gram.npz")
        assert be._cache_file == "/tmp/gram.npz"
        be.free()

    def test_cache_file_none_explicit(self, kernel_config):
        """Explicit None cache_file."""
        be = IntegratedBackend(kernel_config, base="numpy", cache_file=None)
        assert be._cache_file is None
        be.free()

    def test_set_cache_file(self, kernel_config):
        """set_cache_file() updates the path."""
        be = IntegratedBackend(kernel_config, base="numpy")
        assert be._cache_file is None
        be.set_cache_file("/tmp/gram.npz")
        assert be._cache_file == "/tmp/gram.npz"
        be.set_cache_file(None)
        assert be._cache_file is None
        be.set_cache_file("/tmp/other.npz")
        assert be._cache_file == "/tmp/other.npz"
        be.free()


# ═══════════════════════════════════════════════════════════════════
# 2. Cache saves and loads Gram matrices
# ═══════════════════════════════════════════════════════════════════

class TestCacheRoundtrip:
    """Gram matrix cache file roundtrip."""

    @pytest.fixture
    def cache_path(self, tmp_path):
        return str(tmp_path / "gram_cache.npz")

    def test_save_and_load(self, kernel_config, params, phsp, cache_path):
        """First call saves, second call loads — results match exactly."""
        # First: compute + save
        be1 = IntegratedBackend(kernel_config, base="numpy",
                                cache_file=cache_path)
        pb1 = be1.load_data(phsp)
        norm1, _, _ = be1.compute(params, pb1, norm=None, return_p=False)
        be1.free()
        assert os.path.exists(cache_path)

        # Second: load from cache
        be2 = IntegratedBackend(kernel_config, base="numpy",
                                cache_file=cache_path)
        pb2 = be2.load_data(phsp)
        norm2, _, _ = be2.compute(params, pb2, norm=None, return_p=False)
        be2.free()
        assert norm1 == pytest.approx(norm2, abs=1e-10)

    def test_cache_m0_mismatch_invalidates(self, kernel_config, params, phsp, cache_path):
        """When m0 changes, cache is not used (recomputes and overwrites)."""
        # Save with original m0
        be1 = IntegratedBackend(kernel_config, base="numpy",
                                cache_file=cache_path)
        pb1 = be1.load_data(phsp)
        norm1, _, _ = be1.compute(params, pb1, norm=None, return_p=False)
        be1.free()

        # Change m0 — should recompute
        params2 = params.copy()
        params2["m0"] = params["m0"] * 1.01
        be2 = IntegratedBackend(kernel_config, base="numpy",
                                cache_file=cache_path)
        pb2 = be2.load_data(phsp)
        norm2, _, _ = be2.compute(params2, pb2, norm=None, return_p=False)
        be2.free()
        assert abs(norm1 - norm2) > 0.1  # different results

        # Original m0 again — should recompute (cache was overwritten)
        be3 = IntegratedBackend(kernel_config, base="numpy",
                                cache_file=cache_path)
        pb3 = be3.load_data(phsp)
        norm3, _, _ = be3.compute(params, pb3, norm=None, return_p=False)
        be3.free()
        assert norm1 == pytest.approx(norm3, abs=1e-10)

    def test_cache_g0_mismatch_invalidates(self, kernel_config, params, phsp, cache_path):
        """When g0 changes, cache is not used."""
        be1 = IntegratedBackend(kernel_config, base="numpy",
                                cache_file=cache_path)
        pb1 = be1.load_data(phsp)
        norm1, _, _ = be1.compute(params, pb1, norm=None, return_p=False)
        be1.free()

        # Change g0
        params2 = params.copy()
        params2["g0"] = params["g0"] * 0.9
        be2 = IntegratedBackend(kernel_config, base="numpy",
                                cache_file=cache_path)
        pb2 = be2.load_data(phsp)
        norm2, _, _ = be2.compute(params2, pb2, norm=None, return_p=False)
        be2.free()
        assert abs(norm1 - norm2) > 0.1

    def test_cache_missing_file(self, kernel_config, params, phsp, cache_path):
        """Missing cache file gracefully falls back to compute."""
        # cache_path doesn't exist yet
        be = IntegratedBackend(kernel_config, base="numpy",
                               cache_file=cache_path)
        pb = be.load_data(phsp)
        norm, _, _ = be.compute(params, pb, norm=None, return_p=False)
        be.free()
        assert os.path.exists(cache_path)  # now saved
        assert norm > 0

    def test_set_cache_file_after_construction(self, kernel_config, params, phsp, cache_path):
        """set_cache_file() after construction works for saving."""
        be = IntegratedBackend(kernel_config, base="numpy")
        be.set_cache_file(cache_path)
        pb = be.load_data(phsp)
        norm, _, _ = be.compute(params, pb, norm=None, return_p=False)
        be.free()
        assert os.path.exists(cache_path)

    def test_set_cache_file_for_loading(self, kernel_config, params, phsp, cache_path):
        """set_cache_file() before second run works for loading."""
        # First: save
        be1 = IntegratedBackend(kernel_config, base="numpy",
                                cache_file=cache_path)
        pb1 = be1.load_data(phsp)
        norm1, _, _ = be1.compute(params, pb1, norm=None, return_p=False)
        be1.free()

        # Second: load via set_cache_file
        be2 = IntegratedBackend(kernel_config, base="numpy")
        be2.set_cache_file(cache_path)
        pb2 = be2.load_data(phsp)
        norm2, _, _ = be2.compute(params, pb2, norm=None, return_p=False)
        be2.free()
        assert norm1 == pytest.approx(norm2, abs=1e-10)

    def test_no_cache_file_computes_normally(self, kernel_config, params, phsp):
        """Without cache_file, ensure_gram works normally (no file I/O)."""
        be = IntegratedBackend(kernel_config, base="numpy")
        assert be._cache_file is None
        pb = be.load_data(phsp)
        norm, _, _ = be.compute(params, pb, norm=None, return_p=False)
        be.free()
        assert norm > 0

    def test_cache_hermitian_preserved(self, kernel_config, params, phsp, cache_path):
        """Cached Gram matrices remain Hermitian after load."""
        be = IntegratedBackend(kernel_config, base="numpy",
                                cache_file=cache_path)
        pb = be.load_data(phsp)
        be.compute(params, pb, norm=None, return_p=False)
        # Check stored Gram matrices
        Mpp = pb.Mpp
        Mmm = pb.Mmm
        assert np.allclose(Mpp, Mpp.conj().T, atol=1e-14), "Mpp not Hermitian"
        assert np.allclose(Mmm, Mmm.conj().T, atol=1e-14), "Mmm not Hermitian"
        be.free()

        # Reload from cache and recheck
        be2 = IntegratedBackend(kernel_config, base="numpy",
                                cache_file=cache_path)
        pb2 = be2.load_data(phsp)
        be2.compute(params, pb2, norm=None, return_p=False)
        assert np.allclose(pb2.Mpp, pb2.Mpp.conj().T, atol=1e-14)
        assert np.allclose(pb2.Mmm, pb2.Mmm.conj().T, atol=1e-14)
        be2.free()

    def test_corrupted_cache_handled_gracefully(self, kernel_config, params, phsp, cache_path):
        """Corrupted cache file falls back to compute."""
        # Save valid cache
        be1 = IntegratedBackend(kernel_config, base="numpy",
                                cache_file=cache_path)
        pb1 = be1.load_data(phsp)
        norm1, _, _ = be1.compute(params, pb1, norm=None, return_p=False)
        be1.free()

        # Corrupt the file by writing garbage
        with open(cache_path, "w") as f:
            f.write("not a valid npz file")

        # Should fall back to compute, not crash
        be2 = IntegratedBackend(kernel_config, base="numpy",
                                cache_file=cache_path)
        pb2 = be2.load_data(phsp)
        norm2, _, _ = be2.compute(params, pb2, norm=None, return_p=False)
        be2.free()
        assert norm1 == pytest.approx(norm2, abs=1e-10)
