"""Forward-only transforms: the base class supplies a finite-difference VJP.

A user-defined mass/width transform implements ``forward`` (plus the
input/output declarations); ``Transform.backward`` defaults to centered
finite differences, and ``inverse`` stays optional.  An analytic
``backward`` override still wins.
"""

import numpy as np
import pytest

from tabpwa.param_constraint import Transform


class SinSquare(Transform):
    def __init__(self, scale=2.0):
        super().__init__(input_names=["x"], output_names=["a", "b"])
        self.scale = scale            # captured constant, not an input

    def forward(self, d):
        x = float(d["x"])
        return {"a": self.scale * np.sin(x), "b": x ** 2}


def _analytic(dx, x=0.4):
    return 2.0 * np.cos(x) * dx["a"] + 2.0 * x * dx["b"]


def test_forward_only_backward_matches_analytic():
    tr = SinSquare()
    dx = {"a": 1.3, "b": -0.7}
    g = tr.backward(dx, {"x": 0.4})
    assert set(g) == {"x"}
    assert g["x"] == pytest.approx(_analytic(dx), rel=1e-8)


def test_apply_backward_and_inverse_optional():
    tr = SinSquare()
    tr.apply_forward({"x": 0.4, "other": 7.0})
    back = tr.apply_backward({"a": 1.3, "b": -0.7})
    assert back["x"] == pytest.approx(_analytic({"a": 1.3, "b": -0.7}),
                                      rel=1e-8)
    assert "a" not in back and "b" not in back      # derived outputs removed
    # no inverse implemented: apply_inverse leaves the dict untouched
    out = tr.apply_inverse({"x": 0.4, "a": 1.0})
    assert out["x"] == 0.4 and out["a"] == 1.0


def test_zero_point_is_finite():
    """The FD backward is finite where a ck/r-style formula would be 0/0."""
    g = SinSquare().backward({"a": 1.0, "b": 1.0}, {"x": 0.0})
    assert np.isfinite(g["x"])
    assert g["x"] == pytest.approx(2.0, rel=1e-7)      # 2·cos(0) + 0


def test_forward_receives_only_declared_inputs():
    class Hit(Transform):
        def __init__(self):
            super().__init__(input_names=["x"], output_names=["y"])
            self.seen = None

        def forward(self, d):
            self.seen = set(d)
            return {"y": float(d["x"]) * 3.0}

    tr = Hit()
    tr.apply_forward({"x": 1.0, "junk": 2.0})
    assert tr.seen == {"x"}
    assert tr.backward({"y": 2.0}, tr._last_input)["x"] == pytest.approx(6.0)


def test_analytic_backward_override_wins():
    class Analytic(SinSquare):
        def backward(self, grad_out, d_in=None):
            return {"x": 123.0}

    assert Analytic().backward({"a": 1.0, "b": 1.0}, {"x": 0.4})["x"] == 123.0


def test_fd_rejects_complex_inputs():
    with pytest.raises(ValueError, match="complex"):
        SinSquare().fd_backward({"a": 1.0, "b": 1.0}, {"x": 1.0 + 0j})


# ── FD vs an existing analytic mass-width transform ─────────────────

def test_fd_matches_analytic_model_transform():
    from tabpwa import Fitter

    f = Fitter("tests/config_rho_omega2.yml", backend="numpy_pwa")
    try:
        tr = next(t for t in f.cm.mass_width_transforms
                  if type(t).__name__ == "RhoOmegaWeightsTransform")
        d_in = {n: f.cm.defaults[n] for n in tr.input_names}
        g_out = {n: 1.3 for n in tr.output_names}
        ana = tr.backward(g_out, d_in)
        fd = tr.fd_backward(g_out, d_in)
        for n in tr.input_names:
            assert fd[n] == pytest.approx(ana[n], rel=1e-7, abs=1e-9), n
    finally:
        f.free()
