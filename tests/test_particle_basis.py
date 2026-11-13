"""Tests for the rational-lineshape basis interface (``BasisModel``) and the
``kmatrix2`` reference model.

Particle layer only: these tests exercise the amplitude algebra, the
real-slot physical-parameter → weight transform, and the VJP the kernel
gradient chain will rely on.  No kernel / backend / fitter involvement.
"""

import numpy as np
import pytest

from tabpwa.particle_model import ALL_MODELS, build_particle
from tabpwa.particle_model.base import BaseModel, BasisModel
from tabpwa.particle_model.kmatrix2_model import (
    KMatrix2Model, KMatrix2WeightsTransform, M_PI as KM_PI,
)


# ── helpers ──────────────────────────────────────────────────────────

CASES = [
    dict(m1=0.80, m2=1.35, g1=0.35, g2=0.55, b1=0.9, b2=0.4),
    dict(m1=0.65, m2=1.55, g1=0.20, g2=0.90, b1=1.3, b2=-0.6),
    dict(m1=0.95, m2=1.15, g1=0.60, g2=0.10, b1=0.2, b2=0.8),
]


def _phys_params(name, m1, m2, g1, g2, b1, b2):
    """Build the 8 real input slots (complex b split into _r/_i)."""
    b1, b2 = complex(b1), complex(b2)
    return {
        f"{name}_m1": m1, f"{name}_m2": m2,
        f"{name}_g1": g1, f"{name}_g2": g2,
        f"{name}_b1_r": b1.real, f"{name}_b1_i": b1.imag,
        f"{name}_b2_r": b2.real, f"{name}_b2_i": b2.imag,
    }


def _mass_grid(p1, p2, n=241, lo=0.25, hi=1.75):
    """Grid spanning below threshold and both poles (poles excluded)."""
    m = np.linspace(lo, hi, n)
    return m[(np.abs(m - p1) > 1e-4) & (np.abs(m - p2) > 1e-4)]


def _direct_km2(m, case, m_pi=KM_PI):
    """Independent direct evaluation of ``P / (1 - i·ρ·K)``."""
    s = np.asarray(m, dtype=float) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.asarray(np.emath.sqrt(1.0 - (2.0 * m_pi) ** 2 / s),
                         dtype=complex)
        k = (case["g1"] ** 2 / (case["m1"] ** 2 - s)
             + case["g2"] ** 2 / (case["m2"] ** 2 - s))
        p = (case["b1"] / (case["m1"] ** 2 - s)
             + case["b2"] / (case["m2"] ** 2 - s))
        return p / (1.0 - 1j * rho * k)


def _km2_model(name="k2", mass=0.80, width=0.30):
    return build_particle(name, model="kmatrix2", mass=mass, width=width)


# ── 1. BasisModel defaults reproduce the plain BW bit-for-bit ────────

def test_basis_default_matches_bw_bitwise():
    m = np.linspace(0.30, 1.20, 57)
    params = {"r_mass": 0.775, "r_width": 0.149}

    bw = build_particle("r", model="BW", mass=0.775, width=0.149)
    basis = BasisModel("r", mass=0.775, width=0.149)

    ref = bw.amplitude_raw(m, params)
    got = basis.amplitude_raw(m, params)
    assert np.array_equal(ref.view(np.float64), got.view(np.float64))

    # the basis decomposition is the kernel's coeff basis:
    # [1, m², −i·γ] with weights [m0², −1, m0·g0]
    den = basis.den_basis(m)
    assert len(den) == len(basis.get_den_name()) == 3
    assert basis.get_den_name() == ["r_s0", "r_poly", "r_width"]
    assert np.array_equal(den[0], np.ones_like(m, dtype=complex))
    assert np.array_equal(den[1], m ** 2 + 0j)
    assert np.array_equal(den[2], -1j * np.ones_like(m))

    # no basis transform: amplitude() == amplitude_raw()
    assert np.array_equal(
        basis.amplitude(m, params), basis.amplitude_raw(m, params))

    # build_particle("BW", ...) is untouched and still works
    bw2 = build_particle("r", model="BW", mass=0.700, width=0.100)
    assert isinstance(bw2, BaseModel)
    assert "BW" in ALL_MODELS
    assert bw2.name == "r"


def test_missing_weights_use_documented_defaults():
    """Default weights: s0 = mass², poly = −1, gamma rows = mass·g0 = 0."""
    mdl = BasisModel("r", mass=0.775, width=0.149)
    m = np.array([0.5, 0.9])
    got = mdl.amplitude_raw(m, {})
    ref = 1.0 / (0.775 ** 2 - m ** 2)
    assert np.allclose(got, ref, rtol=0, atol=0)


def test_denominator_weights_real_rows_complex():
    """Denominator weights are real; all i factors live in the basis rows."""
    # -- default BasisModel: kernel coeff weights [m0², −1, m0·g0] --
    mdl = BasisModel("r", mass=0.775, width=0.149)
    m = np.linspace(0.30, 1.20, 23)
    rows = mdl.den_basis(m)
    assert all(np.iscomplexobj(r) for r in rows)
    assert any(np.any(np.imag(r) != 0) for r in rows)
    g = mdl._den_weights({"r_mass": 0.775, "r_width": 0.149})
    assert np.all(np.imag(g) == 0)
    assert np.all(np.imag(mdl._num_weights({})) == 0)

    # -- KMatrix2 at several random real parameter points --
    km = _km2_model()
    m = _mass_grid(0.8, 1.35)
    rng = np.random.default_rng(11)
    for _ in range(5):
        case = dict(
            m1=rng.uniform(0.60, 1.00), m2=rng.uniform(1.10, 1.60),
            g1=rng.uniform(0.05, 0.80), g2=rng.uniform(0.05, 0.80),
            b1=rng.uniform(-1.0, 1.0), b2=rng.uniform(-1.0, 1.0),
        )
        params = _phys_params("k2", **case)
        out = km.make_basis_transform().apply_forward(dict(params))
        # every transform output is a real float (fd_backward contract)
        assert all(np.isrealobj(v) for v in out.values())
        gk = [out[name] for name in km.get_den_name()]
        assert np.all(np.imag(gk) == 0), gk
        # real production couplings → real reassembled numerator weights
        nw = km._num_weights(params)
        assert all(np.imag(v) == 0 for v in nw)

    rows = km.den_basis(m)
    assert all(np.iscomplexobj(r) for r in rows)
    assert any(np.any(np.imag(r) != 0) for r in rows)
    assert all(np.iscomplexobj(r) for r in km.num_basis(m))


def test_complex_numerator_weights_supported():
    """Complex ``b_i`` flow through the real ``_r``/``_i`` slots end-to-end."""
    case = dict(m1=0.80, m2=1.35, g1=0.35, g2=0.55,
                b1=1.0 + 0.3j, b2=0.7 - 0.2j)
    mdl = _km2_model()
    m = _mass_grid(case["m1"], case["m2"])
    params = _phys_params("k2", **case)

    tr = mdl.make_basis_transform()
    out = tr.apply_forward(dict(params))
    # all transform outputs remain real floats
    assert all(np.isrealobj(v) for v in out.values())
    n0r, n0i, n1r, n1i = (out[n] for n in mdl.get_num_name())
    m1s, m2s = case["m1"] ** 2, case["m2"] ** 2
    assert n0i == pytest.approx(0.3 * m2s - 0.2 * m1s, rel=1e-14, abs=1e-15)
    assert n1i == pytest.approx(-(0.3 - 0.2), rel=1e-14, abs=1e-15)
    # denominator weights stay real even with complex b_i
    assert np.all(np.imag([out[name] for name in mdl.get_den_name()]) == 0)

    # reassembled complex numerator weights match the direct algebra
    nw = mdl._num_weights(params)
    assert nw[0] == pytest.approx(case["b1"] * m2s + case["b2"] * m1s,
                                  rel=1e-14, abs=1e-15)
    assert nw[1] == pytest.approx(-(case["b1"] + case["b2"]),
                                  rel=1e-14, abs=1e-15)

    got = mdl.amplitude(m, params)
    ref = _direct_km2(m, case, m_pi=mdl.m_pi)
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-14)

    # amplitude_raw also handles the complex numerator path
    raw = mdl.amplitude_raw(m, params)
    assert np.allclose(raw, ref, rtol=1e-12, atol=1e-14)


# ── 2. kmatrix2 amplitude == the direct formula ──────────────────────

@pytest.mark.parametrize("case", CASES)
def test_kmatrix2_matches_direct_formula(case):
    mdl = _km2_model()
    m = _mass_grid(case["m1"], case["m2"])
    params = _phys_params("k2", **case)

    got = mdl.amplitude(m, params)
    ref = _direct_km2(m, case, m_pi=mdl.m_pi)
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-14)

    # grid actually crosses both poles and the ππ threshold
    assert m.min() < 2 * mdl.m_pi < m.max()
    assert m.min() < case["m1"] < m.max()
    assert m.min() < case["m2"] < m.max()


# ── 3. basis reconstruction == amplitude_raw ─────────────────────────

def test_basis_route_equals_raw_amplitude():
    case = CASES[0]
    mdl = _km2_model()
    m = _mass_grid(case["m1"], case["m2"])
    params = _phys_params("k2", **case)

    tr = mdl.make_basis_transform()
    resolved = tr.apply_forward(dict(params))

    num_rows = mdl.num_basis(m)
    den_rows = mdl.den_basis(m)
    assert len(num_rows) == 2
    assert len(mdl.get_num_name()) == 4      # real _r/_i slots
    assert len(den_rows) == len(mdl.get_den_name()) == 5
    assert all(np.iscomplexobj(r) for r in num_rows + den_rows)
    assert all(np.isrealobj(resolved[n]) for n in mdl.get_num_name())
    assert np.all(np.imag([resolved[n] for n in mdl.get_den_name()]) == 0)

    num = np.zeros_like(m, dtype=complex)
    for weight, row in zip(mdl._num_weights(resolved), num_rows):
        num = num + weight * row
    den = np.zeros_like(m, dtype=complex)
    for name, row in zip(mdl.get_den_name(), den_rows):
        den = den + resolved[name] * row
    route = num / den

    raw = mdl.amplitude_raw(m, resolved)
    assert np.allclose(route, raw, rtol=1e-12, atol=1e-14)
    assert np.allclose(mdl.amplitude(m, params), raw, rtol=1e-12, atol=1e-14)


def test_amplitude_uses_transform_defaults_for_empty_params():
    mdl = _km2_model()
    m = _mass_grid(0.8, 0.88)
    params = {f"k2_{k}": v for k, v in mdl._phys().items()}
    assert np.allclose(mdl.amplitude(m, {}), mdl.amplitude(m, params),
                       rtol=1e-14, atol=0)


# ── 4. transform forward: exact weights + name alignment ─────────────

def test_transform_forward_weights_and_names():
    mdl = _km2_model()
    tr = mdl.make_basis_transform()
    assert isinstance(tr, KMatrix2WeightsTransform)
    assert tr.output_names == mdl.get_den_name() + mdl.get_num_name()
    assert tr.input_names == [
        "k2_m1", "k2_m2", "k2_g1", "k2_g2",
        "k2_b1_r", "k2_b1_i", "k2_b2_r", "k2_b2_i",
    ]

    case = CASES[1]
    params = _phys_params("k2", **case)
    out = tr.apply_forward(dict(params))

    m1s, m2s = case["m1"] ** 2, case["m2"] ** 2
    g1s, g2s = case["g1"] ** 2, case["g2"] ** 2
    b1r, b1i = complex(case["b1"]).real, complex(case["b1"]).imag
    b2r, b2i = complex(case["b2"]).real, complex(case["b2"]).imag
    expected = {
        "k2_s0": m1s * m2s,                       # 1 row (kernel s0)
        "k2_poly": -(m1s + m2s),                  # m² row (kernel poly)
        "k2_m4": 1.0,                             # m⁴ row
        "k2_rho": g1s * m2s + g2s * m1s,          # −i·ρ row
        "k2_srho": g1s + g2s,                     # +i·s·ρ row
        "k2_n0_r": b1r * m2s + b2r * m1s,         # numerator real part
        "k2_n0_i": b1i * m2s + b2i * m1s,         # numerator imaginary part
        "k2_n1_r": -(b1r + b2r),
        "k2_n1_i": -(b1i + b2i),
    }
    assert set(expected) == set(tr.output_names)
    for name, val in expected.items():
        assert out[name] == pytest.approx(val, rel=1e-14, abs=1e-15)
        # every output is a real float (i factors live in the basis rows)
        assert np.imag(out[name]) == 0
        assert np.isrealobj(out[name])
    assert np.iscomplexobj(list(out.values())) is False


# ── 5. VJP of the real input slots (FD reference) ────────────────────

def test_transform_backward_matches_finite_difference():
    """fd_backward runs on real slots and matches FD with real cotangents."""
    mdl = _km2_model()
    params = _phys_params("k2", **CASES[2])
    tr = mdl.make_basis_transform()
    out = tr.apply_forward(dict(params))
    assert all(np.isrealobj(v) for v in out.values())

    rng = np.random.default_rng(7)
    cot = {name: float(rng.normal()) for name in tr.output_names}

    vjp = tr.backward(cot, d_in=dict(params))
    assert set(vjp) == set(tr.input_names)

    def scalar(x):
        o = tr.forward(x)
        return sum(cot[name] * o[name] for name in tr.output_names)

    for name in tr.input_names:
        x = float(params[name])
        h = 1e-6 * max(1.0, abs(x))
        p_plus = dict(params)
        p_minus = dict(params)
        p_plus[name] = x + h
        p_minus[name] = x - h
        fd = (scalar(p_plus) - scalar(p_minus)) / (2.0 * h)
        rel = abs(vjp[name] - fd) / max(abs(fd), 1e-12)
        assert rel < 1e-5, (name, vjp[name], fd)

    # the b1_i / b2_i slots are genuinely perturbed (nonzero derivatives)
    assert abs(vjp["k2_b1_i"]) > 0
    assert abs(vjp["k2_b2_i"]) > 0


def test_amplitude_gradient_chain_with_real_cotangents():
    """The real-slot VJP reproduces the amplitude gradient chain."""
    case = CASES[0]
    mdl = _km2_model()
    m = _mass_grid(case["m1"], case["m2"], n=65)
    params = _phys_params("k2", **case)

    tr = mdl.make_basis_transform()
    resolved = tr.apply_forward(dict(params))

    num_rows = mdl.num_basis(m)
    den_rows = mdl.den_basis(m)
    num = np.zeros_like(m, dtype=complex)
    for weight, row in zip(mdl._num_weights(resolved), num_rows):
        num = num + weight * row
    den = np.zeros_like(m, dtype=complex)
    for name, row in zip(mdl.get_den_name(), den_rows):
        den = den + resolved[name] * row

    rng = np.random.default_rng(3)
    c = rng.normal(size=m.size) + 1j * rng.normal(size=m.size)

    # real objective Q = Re(Σ_e c_e·A_e); real cotangents for every slot
    n0, n1 = mdl.get_num_name()[0:2], mdl.get_num_name()[2:4]
    d_a_d_n0 = [row / den for row in num_rows]
    d_a_d_n1 = [row / den for row in num_rows]
    cot = {}
    cot[n0[0]] = float(np.real(np.sum(c * d_a_d_n0[0])))
    cot[n0[1]] = float(np.real(np.sum(c * (1j * d_a_d_n0[0]))))
    cot[n1[0]] = float(np.real(np.sum(c * d_a_d_n1[1])))
    cot[n1[1]] = float(np.real(np.sum(c * (1j * d_a_d_n1[1]))))
    for name, row in zip(mdl.get_den_name(), den_rows):
        cot[name] = float(np.real(np.sum(c * (-num * row / den ** 2))))

    vjp = tr.backward(cot, d_in=dict(params))

    def q(x):
        return float(np.real(np.sum(c * mdl.amplitude(m, x))))

    for name in tr.input_names:
        x = float(params[name])
        h = 1e-6 * max(1.0, abs(x))
        p_plus = dict(params)
        p_minus = dict(params)
        p_plus[name] = x + h
        p_minus[name] = x - h
        fd = (q(p_plus) - q(p_minus)) / (2.0 * h)
        rel = abs(vjp[name] - fd) / max(abs(fd), 1e-12)
        assert rel < 1e-5, (name, vjp[name], fd)


# ── 6. registration & defaults ───────────────────────────────────────

def test_kmatrix2_registration_and_defaults():
    assert "kmatrix2" in ALL_MODELS
    assert ALL_MODELS["kmatrix2"] is KMatrix2Model

    mdl = _km2_model()
    assert isinstance(mdl, KMatrix2Model)

    defaults = mdl.get_defaults()
    tr = mdl.make_basis_transform()
    for name in tr.input_names:
        assert name in defaults

    # defaults derived from mass / width (pole-1 nominal values)
    assert defaults["k2_mass"] == pytest.approx(0.80)
    assert defaults["k2_width"] == pytest.approx(0.30)
    assert defaults["k2_m1"] == pytest.approx(0.80)
    assert defaults["k2_m2"] == pytest.approx(0.88)
    assert defaults["k2_g1"] == pytest.approx(0.15)
    assert defaults["k2_g2"] == pytest.approx(0.15)
    # real numerator slots (default b_i = 1 + 0i)
    assert defaults["k2_b1_r"] == pytest.approx(1.0)
    assert defaults["k2_b1_i"] == pytest.approx(0.0)
    assert defaults["k2_b2_r"] == pytest.approx(1.0)
    assert defaults["k2_b2_i"] == pytest.approx(0.0)

    # explicit kwargs, including a complex b1 split into real slots
    mdl2 = build_particle("k2", model="kmatrix2", mass=0.8, width=0.3,
                          m2=1.9, g1=0.4, b1=1.0 + 0.3j, b2_r=-0.2,
                          b2_i=0.1)
    d2 = mdl2.get_defaults()
    assert d2["k2_m2"] == pytest.approx(1.9)
    assert d2["k2_g1"] == pytest.approx(0.4)
    assert d2["k2_g2"] == pytest.approx(0.15)
    assert d2["k2_b1_r"] == pytest.approx(1.0)
    assert d2["k2_b1_i"] == pytest.approx(0.3)
    assert d2["k2_b2_r"] == pytest.approx(-0.2)
    assert d2["k2_b2_i"] == pytest.approx(0.1)
