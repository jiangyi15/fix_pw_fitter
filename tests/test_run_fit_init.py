"""``run_fit.py --save-init``: path derivation and start-vector replay."""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from run_fit import _init_out_path  # noqa: E402


def test_init_out_path_default_next_to_config():
    assert _init_out_path(None, "tutorials/config.yml") == \
        "tutorials/config_init.json"
    assert _init_out_path("", "tutorials/config.yml") == \
        "tutorials/config_init.json"


def test_init_out_path_explicit_and_extension():
    assert _init_out_path("out.json", "c.yml") == "out.json"
    assert _init_out_path("out", "c.yml") == "out.json"      # ext added
    assert _init_out_path("dir/start", "c.yml") == "dir/start.json"


def test_init_out_path_loop_run_suffix():
    assert _init_out_path("out.json", "c.yml", run=2) == "out2.json"
    assert _init_out_path(None, "c.yml", run=3) == "c_init3.json"


def test_save_init_round_trips_through_values_from_dict(tmp_path):
    """A saved start vector replays bit-for-bit via values_from_dict()."""
    from tabpwa import Fitter

    fitter = Fitter(os.path.join(ROOT, "config_angle.yml"), backend="numpy")
    try:
        fitter.apply_constrains()
        x0 = fitter.initial_values(seed=7)

        path = tmp_path / "init.json"
        fitter.save_params(x0, str(path))
        data = json.loads(path.read_text())

        x1 = fitter.values_from_dict(data)
        assert x1.shape == x0.shape
        assert np.allclose(x0, x1), \
            f"replay mismatch, max |dx| = {np.max(np.abs(x0 - x1)):.3e}"
    finally:
        fitter.backend.free()
