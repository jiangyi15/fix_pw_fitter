"""Bound-transform uncertainty propagation (apply_bound_dict)."""
import numpy as np
import pytest

from tabpwa.param_constraint import ConstraintManager


def _cm(lo, hi, name="a"):
    cm = ConstraintManager([name])
    cm.bounds.set(name, lo, hi)
    return cm


@pytest.mark.parametrize("x", [0.0, 0.5, 1.0, 2.0, 4.0, -3.0])
def test_bound_error_uses_unbounded_value(x):
    """sigma_y = |dy/dx| * sigma_x, with the Jacobian at the UNBOUNDED x.

    The incoming error is the optimizer-space (unbounded) one, exactly as
    ``sqrt(diag(hess_inv))`` returns it; ``apply_bound_dict`` must propagate
    before it overwrites the value with the bounded one.
    """
    cm = _cm(0.3, 0.8)
    bt = cm.bounds["a"]
    sigma_x = 0.1
    d = {"a": x}
    e = {"a": sigma_x}
    cm.apply_bound_dict(d, e)

    assert d["a"] == pytest.approx(float(bt(x)), rel=1e-14)
    assert e["a"] == pytest.approx(float(bt.trans_err(x, sigma_x)), rel=1e-14)
    # and the old (buggy) evaluation at the bounded value is a different number
    assert e["a"] != pytest.approx(float(bt.trans_err(d["a"], sigma_x)), rel=1e-6)


def test_bound_error_at_centre_is_untransformed():
    """At x = 0 the Jacobian is 1, so sigma_y == sigma_x (old code: ~0.08x)."""
    cm = _cm(0.3, 0.8)
    d, e = {"a": 0.0}, {"a": 0.05}
    cm.apply_bound_dict(d, e)
    assert e["a"] == pytest.approx(0.05, rel=1e-14)


def test_bound_error_unbounded_names_untouched():
    """Names without a bound (or absent from the dict) keep their error."""
    cm = _cm(0.3, 0.8)
    d = {"a": 1.0, "b": 2.0}
    e = {"a": 0.1, "b": 0.2}
    cm.apply_bound_dict(d, e)
    assert e["b"] == pytest.approx(0.2)
    # an error for a name that is not in the value dict must not raise
    cm2 = _cm(0.3, 0.8)
    d2 = {}
    e2 = {"a": 0.1}
    cm2.apply_bound_dict(d2, e2)
    assert e2["a"] == pytest.approx(0.1)
