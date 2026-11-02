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
    """Explicit-name form (``super().__init__``)."""

    def __init__(self, scale=2.0):
        super().__init__(input_names=["x"], output_names=["a", "b"])
        self.scale = scale            # captured constant, not an input

    def forward(self, d):
        x = float(d["x"])
        return {"a": self.scale * np.sin(x), "b": x ** 2}


class DynamicNames(Transform):
    """Instance-assignment form: no ``super().__init__``, per-particle names."""

    def __init__(self, prefix="x", scale=2.0):
        self.input_names = [prefix]
        self.output_names = ["a", "b"]
        self.scale = scale

    def forward(self, d):
        x = float(d[self.input_names[0]])
        return {"a": self.scale * np.sin(x), "b": x ** 2}


def _analytic(dx, x=0.4):
    return 2.0 * np.cos(x) * dx["a"] + 2.0 * x * dx["b"]


def test_forward_only_backward_matches_analytic():
    tr = SinSquare()
    dx = {"a": 1.3, "b": -0.7}
    g = tr.backward(dx, {"x": 0.4})
    assert set(g) == {"x"}
    assert g["x"] == pytest.approx(_analytic(dx), rel=1e-8)


def test_instance_names_without_super_init():
    tr = DynamicNames()
    assert tr.input_names == ["x"] and tr.output_names == ["a", "b"]
    dx = {"a": 1.3, "b": -0.7}
    assert tr.backward(dx, {"x": 0.4})["x"] == pytest.approx(
        _analytic(dx), rel=1e-8)
    # the registration/apply paths read the instance names
    tr.apply_forward({"x": 0.4, "junk": 1.0})
    back = tr.apply_backward(dict(dx))
    assert back["x"] == pytest.approx(_analytic(dx), rel=1e-8)


def test_names_must_be_declared_before_use():
    class Undeclared(Transform):
        def forward(self, d):
            return {}

    tr = Undeclared()
    with pytest.raises(NotImplementedError, match="input_names"):
        tr.input_names
    with pytest.raises(NotImplementedError, match="output_names"):
        tr.output_names
    with pytest.raises(NotImplementedError, match="names must be declared"):
        tr.apply_forward({"x": 1.0})


def test_class_level_name_declarations_rejected():
    """Particle-prefixed names cannot be class attributes."""
    with pytest.raises(TypeError, match="instance attribute"):
        class Bad(Transform):          # noqa: F841
            input_names = ["x"]
            output_names = ["y"]
            def forward(self, d):
                return {}


def test_constructor_names_still_work():
    """The base __init__ shim still accepts explicit names."""

    class SuperDeclared(Transform):
        def __init__(self, **kw):
            super().__init__(**kw)

        def forward(self, d):
            return {"y": 1.0}

    tr = SuperDeclared(input_names=["z"], output_names=["q"])
    assert tr.input_names == ["z"] and tr.output_names == ["q"]


def test_direct_init_assignment_without_super():
    """A custom __init__ may assign the names directly; no super() needed."""

    class DirectInit(Transform):
        def __init__(self, name, scale):
            self.input_names = [name]
            self.output_names = ["y"]
            self.scale = scale

        def forward(self, d):
            return {"y": self.scale * float(d[self.input_names[0]]) ** 2}

    tr = DirectInit("z", 3.0)
    assert tr.input_names == ["z"] and tr.output_names == ["y"]
    g = tr.backward({"y": 1.0}, {"z": 2.0})
    assert g["z"] == pytest.approx(12.0, rel=1e-8)   # 3·2z


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
