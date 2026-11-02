"""SympyTransform: write ``calc`` once; forward/backward are generated."""

import numpy as np
import pytest
import sympy as sp

from tabpwa.param_constraint import SympyTransform


class SinSquareSym(SympyTransform):
    def __init__(self, prefix="jpsi->R(1)_x", scale=2.0):
        self.input_names = [prefix]
        self.output_names = ["a", "b"]
        self.scale = scale
        self.calc_calls = 0

    def calc(self, s):
        self.calc_calls += 1
        x = s[self.input_names[0]]
        return {"a": self.scale * sp.sin(x), "b": x ** 2}


def _analytic(dx, x=0.4):
    return 2.0 * np.cos(x) * dx["a"] + 2.0 * x * dx["b"]


def test_forward_and_backward_are_exact():
    tr = SinSquareSym()
    x = 0.4
    out = tr.forward({tr.input_names[0]: x})
    assert out["a"] == pytest.approx(2.0 * np.sin(x))
    assert out["b"] == pytest.approx(x ** 2)

    dx = {"a": 1.3, "b": -0.7}
    g = tr.backward(dx, {tr.input_names[0]: x})
    assert g[tr.input_names[0]] == pytest.approx(_analytic(dx), rel=1e-12)

    # exact derivative matches the numerical fallback
    fd = tr.fd_backward(dx, {tr.input_names[0]: x})
    assert g[tr.input_names[0]] == pytest.approx(
        fd[tr.input_names[0]], rel=1e-7, abs=1e-9)


def test_calc_is_compiled_once():
    tr = SinSquareSym()
    tr.forward({tr.input_names[0]: 0.2})
    tr.forward({tr.input_names[0]: 0.3})
    tr.backward({"a": 1.0, "b": 1.0}, {tr.input_names[0]: 0.4})
    assert tr.calc_calls == 1


def test_captured_constants_and_dynamic_names():
    """Names with '->', '(' and ')' (particle-prefixed) are supported."""
    tr = SinSquareSym(prefix="B->K(1)_m", scale=3.0)
    out = tr.forward({"B->K(1)_m": 1.0})
    assert out["a"] == pytest.approx(3.0 * np.sin(1.0))
    assert out["b"] == pytest.approx(1.0)


def test_missing_and_extra_outputs_raise():
    class Missing(SympyTransform):
        def __init__(self):
            self.input_names = ["x"]
            self.output_names = ["y"]
        def calc(self, s):
            return {}

    with pytest.raises(ValueError, match="missing outputs"):
        Missing().forward({"x": 1.0})

    class Extra(SympyTransform):
        def __init__(self):
            self.input_names = ["x"]
            self.output_names = ["y"]
        def calc(self, s):
            return {"y": s["x"], "z": s["x"]}

    with pytest.raises(ValueError, match="undeclared outputs"):
        Extra().forward({"x": 1.0})


def test_inverse_optional():
    tr = SinSquareSym()
    d = {tr.input_names[0]: 0.4, "a": 1.0}
    assert tr.apply_inverse(dict(d)) == d


def test_non_dict_calc_rejected():
    class Bad(SympyTransform):
        def __init__(self):
            self.input_names = ["x"]
            self.output_names = ["y"]
        def calc(self, s):
            return s["x"]

    with pytest.raises(TypeError, match="dict"):
        Bad().forward({"x": 1.0})


def test_class_level_names_rejected():
    with pytest.raises(TypeError, match="instance attribute"):
        class Bad(SympyTransform):       # noqa: F841
            input_names = ["x"]
            output_names = ["y"]
            def calc(self, s):
                return {"y": s["x"]}


def test_apply_forward_backward_chain():
    tr = SinSquareSym()
    name = tr.input_names[0]
    tr.apply_forward({name: 0.4, "other": 2.0})
    back = tr.apply_backward({"a": 1.3, "b": -0.7})
    assert back[name] == pytest.approx(_analytic({"a": 1.3, "b": -0.7}),
                                       rel=1e-12)
    assert "a" not in back and "b" not in back
