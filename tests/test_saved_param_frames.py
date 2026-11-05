"""save_params: the saved error section carries its own value frame.

The uncertainty is propagated through the bound transform only (see
``Fitter.save_params``), so it belongs to the after-bounds value, not to the
resolved one.  ``value_bounded`` records exactly that value, so an ``error``
is never paired with a number from a different transform stage.
"""
import json

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from tabpwa import Fitter

CFG = "tests/config_angle.yml"
SIGMA_X = 0.01


def _fitter_with_bound_and_scale():
    """A fitter where one free parameter has BOTH a bound and a scale.

    A_prod gets range [-1, 1] (bias 0, so a raw x maps into the interval)
    and a linear scale 2, so the resolved value is 2x the after-bounds one —
    the two frames are guaranteed to differ.
    """
    f = Fitter(CFG, backend="numpy")
    f.set_range("A_prod", -1.0, 1.0)
    f.set_scale({"A_prod": 2.0})
    return f


def _save(tmp_path, f, x):
    res = OptimizeResult(x=x, hess_inv=np.eye(len(x)) * SIGMA_X ** 2,
                         fun=1.0, jac=np.zeros(len(x)), success=True,
                         message="")
    path = tmp_path / "params.json"
    f.save_params(res, str(path))
    return json.loads(path.read_text())


def test_value_bounded_pairs_with_error(tmp_path):
    f = _fitter_with_bound_and_scale()
    x = f.initial_values(seed=1)
    i = f.cm.var_registry.flat_names.index("A_prod")
    x[i] = 0.7                                  # off-centre: Jacobian != 1
    bt = f.cm.bounds["A_prod"]

    d = _save(tmp_path, f, x)
    assert "value_bounded" in d

    bounded = float(bt(x[i]))
    assert d["value_bounded"]["A_prod"] == pytest.approx(bounded, rel=1e-12)
    # the error belongs to the after-bounds value ...
    assert d["error"]["A_prod"] == pytest.approx(
        float(bt.trans_err(x[i], SIGMA_X)), rel=1e-12)
    # ... while "value" is the resolved frame (scale 2 applied)
    assert d["value"]["A_prod"] == pytest.approx(2.0 * bounded, rel=1e-10)
    # the two frames really differ, i.e. the old file paired mismatched numbers
    assert abs(d["value"]["A_prod"] - d["value_bounded"]["A_prod"]) > 1e-3


def test_value_bounded_round_trips_through_loader(tmp_path):
    """The extra section must not disturb values_from_dict / --init."""
    f = _fitter_with_bound_and_scale()
    x = f.initial_values(seed=1)
    i = f.cm.var_registry.flat_names.index("A_prod")
    x[i] = 0.7

    d = _save(tmp_path, f, x)
    x_back = f.values_from_dict(d)
    assert np.allclose(x_back, x, rtol=0, atol=1e-9)
