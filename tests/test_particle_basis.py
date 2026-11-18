"""Tests for the rational-lineshape basis interface (``BasisModel``) and the
``kmatrix2`` reference model.

Particle layer only: these tests exercise the amplitude algebra, the
real-slot physical-parameter → weight transform (magnitude/phase ``b``
slots, ``ck`` convention), and the VJP the kernel gradient chain will
rely on.  No kernel / backend / fitter involvement.
"""

import numpy as np
import pytest

from tabpwa.build_defaults import scope
from tabpwa.param_constraint import CKProduct, complex_tail
from tabpwa.particle_model import ALL_MODELS, build_particle
from tabpwa.particle_model.base import (
    BaseModel, BasisModel, DefaultBasisModel)
from tabpwa.particle_model.kmatrix2_model import (
    KMatrix2Model, KMatrix2WeightsTransform, M_PI as KM_PI,
)


# ── helpers ──────────────────────────────────────────────────────────

CASES = [
    dict(m1=0.80, m2=1.35, g1=0.35, g2=0.55, b1=0.9, b2=0.4),
    dict(m1=0.65, m2=1.55, g1=0.20, g2=0.90, b1=1.3, b2=-0.6),
    dict(m1=0.95, m2=1.15, g1=0.60, g2=0.10, b1=0.2, b2=0.8),
]

# non-trivial production phases (b = mag·exp(i·phase))
PHASE_CASE = dict(m1=0.80, m2=1.35, g1=0.35, g2=0.55,
                  b1=0.8 * np.exp(0.4j), b2=1.2 * np.exp(-1.1j))


def _phys_params(name, m1, m2, g1, g2, b1, b2, tail=None):
    """Build the 8 real input slots (complex b → magnitude/phase pair).

    The slot suffixes follow the active ``complex_tail()`` unless given.
    """
    mag, phase = tuple(tail) if tail is not None else complex_tail()
    b1, b2 = complex(b1), complex(b2)
    return {
        f"{name}_m1": m1, f"{name}_m2": m2,
        f"{name}_g1": g1, f"{name}_g2": g2,
        f"{name}_b1{mag}": abs(b1), f"{name}_b1{phase}": float(np.angle(b1)),
        f"{name}_b2{mag}": abs(b2), f"{name}_b2{phase}": float(np.angle(b2)),
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
        p = (complex(case["b1"]) / (case["m1"] ** 2 - s)
             + complex(case["b2"]) / (case["m2"] ** 2 - s))
        return p / (1.0 - 1j * rho * k)


def _km2_model(name="k2", mass=0.80, width=0.30):
    return build_particle(name, model="kmatrix2", mass=mass, width=width)


# ── 1. BasisModel defaults reproduce the plain BW bit-for-bit ────────

def test_basis_default_matches_bw_bitwise():
    m = np.linspace(0.30, 1.20, 57)
    params = {"r_mass": 0.775, "r_width": 0.149}

    bw = build_particle("r", model="BW", mass=0.775, width=0.149)
    basis = DefaultBasisModel("r", mass=0.775, width=0.149)

    ref = bw.amplitude_raw(m, params)
    got = basis.amp_raw(m, params)
    assert np.array_equal(ref.view(np.float64), got.view(np.float64))

    # the basis decomposition is the kernel's coeff basis:
    # [1, m², −i·γ] with weights [m0², −1, m0·g0]
    den = basis.den_basis(m)
    assert len(den) == len(basis.get_den_name()) == 3
    # the m² weight is the SHARED literal "one" (value -1), not "r_poly":
    # `poly` is kmatrix2's FITTED polynomial coefficient -(m1^2+m2^2), a
    # different quantity with a different free/fixed status.
    # the kernel-facing m0*g slot carries the ``_t`` suffix so it can never
    # collide with the physical fit parameter of the same base name
    assert basis.get_den_name() == ["r_s0", "one", "r_width_t"]
    assert np.array_equal(den[0], np.ones_like(m, dtype=complex))
    # the row is -m² so its weight ``"one"`` is +1 (name reads as its value)
    assert np.array_equal(den[1], -(m ** 2) + 0j)
    assert np.array_equal(den[2], -1j * np.ones_like(m))

    # no basis transform: amplitude() == amplitude_raw()
    assert np.array_equal(
        basis.amp(m, params), basis.amp_raw(m, params))

    # build_particle("BW", ...) is untouched and still works
    bw2 = build_particle("r", model="BW", mass=0.700, width=0.100)
    assert isinstance(bw2, BaseModel)
    assert "BW" in ALL_MODELS
    assert bw2.name == "r"


def test_missing_weights_use_documented_defaults():
    """Default weights: s0 = mass², poly = −1, gamma rows = mass·g0 = 0."""
    mdl = DefaultBasisModel("r", mass=0.775, width=0.149)
    m = np.array([0.5, 0.9])
    got = mdl.amp_raw(m, {})
    ref = 1.0 / (0.775 ** 2 - m ** 2)
    assert np.allclose(got, ref, rtol=0, atol=0)


def test_denominator_weights_real_rows_complex():
    """Denominator weights are real; all i factors live in the basis rows."""
    # -- default BasisModel: kernel coeff weights [m0², −1, m0·g0] --
    mdl = DefaultBasisModel("r", mass=0.775, width=0.149)
    m = np.linspace(0.30, 1.20, 23)
    rows = mdl.den_basis(m)
    assert all(np.iscomplexobj(r) for r in rows)
    assert any(np.any(np.imag(r) != 0) for r in rows)
    g = mdl.den_weights({"r_mass": 0.775, "r_width": 0.149})
    assert np.all(np.imag(g) == 0)
    assert np.all(np.imag(mdl.num_weights({})) == 0)

    # -- KMatrix2 at several random magnitude/phase points --
    km = _km2_model()
    m = _mass_grid(0.8, 1.35)
    rng = np.random.default_rng(11)
    for _ in range(5):
        case = dict(
            m1=rng.uniform(0.60, 1.00), m2=rng.uniform(1.10, 1.60),
            g1=rng.uniform(0.05, 0.80), g2=rng.uniform(0.05, 0.80),
            b1=rng.uniform(0.10, 1.00) * np.exp(1j * rng.uniform(-np.pi, np.pi)),
            b2=rng.uniform(0.10, 1.00) * np.exp(1j * rng.uniform(-np.pi, np.pi)),
        )
        params = _phys_params("k2", **case)
        out = km.make_basis_transform().apply_forward(dict(params))
        # every transform output is a real float (fd_backward contract)
        assert all(np.isrealobj(v) for v in out.values())
        gk = [out[name] for name in km.get_den_name()]
        assert np.all(np.imag(gk) == 0), gk
        # numerator outputs are real Cartesian/polar slots
        assert all(np.isrealobj(out[name]) for name in km.num_slot_names())

    rows = km.den_basis(m)
    assert all(np.iscomplexobj(r) for r in rows)
    assert any(np.any(np.imag(r) != 0) for r in rows)
    assert all(np.iscomplexobj(r) for r in km.num_basis(m))


def test_phases_flow_through_numerator_path():
    """Magnitude/phase ``b`` slots reproduce complex production couplings."""
    case = PHASE_CASE
    mdl = _km2_model()
    m = _mass_grid(case["m1"], case["m2"])
    params = _phys_params("k2", **case)

    tr = mdl.make_basis_transform()
    out = tr.apply_forward(dict(params))
    assert all(np.isrealobj(v) for v in out.values())

    b1, b2 = complex(case["b1"]), complex(case["b2"])
    m1s, m2s = case["m1"] ** 2, case["m2"] ** 2
    n0 = b1 * m2s + b2 * m1s
    n1 = -(b1 + b2)
    n0m, n0p, n1m, n1p = (out[name] for name in mdl.num_slot_names())
    # numerator outputs use the (magnitude, phase) tail convention
    assert n0m == pytest.approx(abs(n0), rel=1e-14, abs=1e-15)
    assert n0p == pytest.approx(np.angle(n0), rel=1e-14, abs=1e-15)
    assert n1m == pytest.approx(abs(n1), rel=1e-14, abs=1e-15)
    assert n1p == pytest.approx(np.angle(n1), rel=1e-14, abs=1e-15)
    # denominator weights stay real
    assert np.all(np.imag([out[name] for name in mdl.get_den_name()]) == 0)

    # reassembled complex numerator weights
    nw = mdl.num_weights(params)
    assert nw[0] == pytest.approx(n0, rel=1e-14, abs=1e-15)
    assert nw[1] == pytest.approx(n1, rel=1e-14, abs=1e-15)

    ref = _direct_km2(m, case, m_pi=mdl.m_pi)
    assert np.allclose(mdl.amp(m, params), ref, rtol=1e-12, atol=1e-14)
    assert np.allclose(mdl.amp_raw(m, params), ref,
                       rtol=1e-12, atol=1e-14)


def test_transform_outputs_feed_ckproduct():
    """Numerator outputs are ``ck`` slots: ``build_ck(out)`` reproduces n_i.

    This is the ``ck_map`` contract: the transform output dict can be
    handed straight to ``CKProduct.build_ck``, which reads the
    ``(magnitude, phase)`` pair and forms ``r·exp(iθ)``.
    """
    case = PHASE_CASE
    mdl = _km2_model()
    m = _mass_grid(case["m1"], case["m2"])
    params = _phys_params("k2", **case)

    tr = mdl.make_basis_transform()
    out = tr.apply_forward(dict(params))

    tail = mdl.tail
    heads = mdl.get_num_name()
    assert heads == [f"{mdl.name}_n0", f"{mdl.name}_n1"]
    # one CKProduct over the head names: the tail is applied by the
    # consumer, which looks up the expanded slots the transform produced
    w0, w1 = CKProduct([[heads[0]], [heads[1]]], tail=tail).build_ck(out)

    b1, b2 = complex(case["b1"]), complex(case["b2"])
    m1s, m2s = case["m1"] ** 2, case["m2"] ** 2
    n0 = b1 * m2s + b2 * m1s
    n1 = -(b1 + b2)
    assert w0 == pytest.approx(n0, rel=1e-14, abs=1e-15)
    assert w1 == pytest.approx(n1, rel=1e-14, abs=1e-15)

    # full chain: weights → build_ck → basis ratio → amplitude
    num_rows = mdl.num_basis(m)
    den_rows = mdl.den_basis(m)
    den = np.zeros_like(m, dtype=complex)
    for name, row in zip(mdl.get_den_name(), den_rows):
        den = den + out[name] * row
    amp = (w0 * num_rows[0] + w1 * num_rows[1]) / den

    ref = _direct_km2(m, case, m_pi=mdl.m_pi)
    assert np.allclose(amp, ref, rtol=1e-12, atol=1e-14)
    assert np.allclose(mdl.amp(m, params), amp, rtol=1e-12, atol=1e-14)


def test_bare_head_real_numerator_weight():
    """A real numerator weight may be given as one parameter named the head."""
    mdl = _km2_model()
    m = _mass_grid(0.8, 1.35)
    params = {"k2_n0": 2.5, "k2_n1": -0.7}

    nw = mdl.num_weights(params)
    assert nw[0] == pytest.approx(2.5, rel=0, abs=0)
    assert nw[1] == pytest.approx(-0.7, rel=0, abs=0)

    # amplitude with the bare-head path vs the manual basis ratio
    resolved = mdl._resolved(params)
    den = np.zeros_like(m, dtype=complex)
    for name, row in zip(mdl.get_den_name(), mdl.den_basis(m)):
        den = den + resolved[name] * row
    num_rows = mdl.num_basis(m)
    amp = (2.5 * num_rows[0] - 0.7 * num_rows[1]) / den
    assert np.allclose(mdl.amp_raw(m, params), amp, rtol=0, atol=0)


# ── 2. kmatrix2 amplitude == the direct formula ──────────────────────

@pytest.mark.parametrize("case", CASES)
def test_kmatrix2_matches_direct_formula(case):
    mdl = _km2_model()
    m = _mass_grid(case["m1"], case["m2"])
    params = _phys_params("k2", **case)

    got = mdl.amp(m, params)
    ref = _direct_km2(m, case, m_pi=mdl.m_pi)
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-14)

    # grid actually crosses both poles and the ππ threshold
    assert m.min() < 2 * mdl.m_pi < m.max()
    assert m.min() < case["m1"] < m.max()
    assert m.min() < case["m2"] < m.max()


# ── 3. basis reconstruction == amplitude_raw ─────────────────────────

def test_basis_route_equals_raw_amplitude():
    case = PHASE_CASE
    mdl = _km2_model()
    m = _mass_grid(case["m1"], case["m2"])
    params = _phys_params("k2", **case)

    tr = mdl.make_basis_transform()
    resolved = tr.apply_forward(dict(params))

    num_rows = mdl.num_basis(m)
    den_rows = mdl.den_basis(m)
    assert len(num_rows) == 2
    assert len(mdl.get_num_name()) == 2      # one head per basis row
    assert len(mdl.num_slot_names()) == 4    # expanded (mag, phase) slots
    assert len(den_rows) == len(mdl.get_den_name()) == 5
    assert all(np.iscomplexobj(r) for r in num_rows + den_rows)
    assert all(np.isrealobj(resolved[n]) for n in mdl.num_slot_names())
    assert np.all(np.imag([resolved[n] for n in mdl.get_den_name()]) == 0)

    num = np.zeros_like(m, dtype=complex)
    for weight, row in zip(mdl.num_weights(resolved), num_rows):
        num = num + weight * row
    den = np.zeros_like(m, dtype=complex)
    for name, row in zip(mdl.get_den_name(), den_rows):
        den = den + resolved[name] * row
    route = num / den

    raw = mdl.amp_raw(m, resolved)
    assert np.allclose(route, raw, rtol=1e-12, atol=1e-14)
    assert np.allclose(mdl.amp(m, params), raw, rtol=1e-12, atol=1e-14)


def test_amplitude_uses_transform_defaults_for_empty_params():
    mdl = _km2_model()
    m = _mass_grid(0.8, 0.88)
    params = {f"k2_{k}": v for k, v in mdl._phys().items()}
    assert np.allclose(mdl.amp(m, {}), mdl.amp(m, params),
                       rtol=1e-14, atol=0)


# ── 4. transform forward: exact weights + name alignment ─────────────

def test_transform_forward_weights_and_names():
    mdl = _km2_model()
    tr = mdl.make_basis_transform()
    assert isinstance(tr, KMatrix2WeightsTransform)
    # default complex_tail() = ("r", "i"): magnitude/phase input slots
    assert tr.input_names == [
        "k2_m1", "k2_m2", "k2_g1", "k2_g2",
        "k2_b1r", "k2_b1i", "k2_b2r", "k2_b2i",
    ]
    assert mdl.get_num_name() == ["k2_n0", "k2_n1"]     # heads (one per row)
    assert mdl.num_slot_names() == [
        "k2_n0r", "k2_n0i", "k2_n1r", "k2_n1i",
    ]
    assert tr.output_names == mdl.num_slot_names() + mdl.get_den_name()

    case = PHASE_CASE
    params = _phys_params("k2", **case)
    out = tr.apply_forward(dict(params))

    m1s, m2s = case["m1"] ** 2, case["m2"] ** 2
    g1s, g2s = case["g1"] ** 2, case["g2"] ** 2
    b1, b2 = complex(case["b1"]), complex(case["b2"])
    n0 = b1 * m2s + b2 * m1s
    n1 = -(b1 + b2)
    expected = {
        "k2_s0": m1s * m2s,                       # 1 row (kernel s0)
        "k2_poly": -(m1s + m2s),                  # m² row (kernel poly)
        "k2_m4": 1.0,                             # m⁴ row
        "k2_rho": g1s * m2s + g2s * m1s,          # −i·ρ row
        "k2_srho": g1s + g2s,                     # +i·s·ρ row
        "k2_n0r": abs(n0),                        # numerator magnitude
        "k2_n0i": np.angle(n0),                   # numerator phase
        "k2_n1r": abs(n1),
        "k2_n1i": np.angle(n1),
    }
    assert set(expected) == set(tr.output_names)
    for name, val in expected.items():
        assert out[name] == pytest.approx(val, rel=1e-13, abs=1e-15)
        # every output is a real float (i factors live in the basis rows)
        assert np.imag(out[name]) == 0
        assert np.isrealobj(out[name])


# ── 5. VJP of the real input slots (FD reference) ────────────────────

def test_transform_backward_matches_finite_difference():
    """fd_backward runs on real slots (incl. phases) and matches FD."""
    mdl = _km2_model()
    params = _phys_params("k2", **PHASE_CASE)
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

    # the phase slots are genuinely differentiated (nonzero derivatives)
    assert abs(vjp["k2_b1i"]) > 0
    assert abs(vjp["k2_b2i"]) > 0


def test_amplitude_gradient_chain_with_real_cotangents():
    """The real-slot VJP reproduces the amplitude gradient chain."""
    case = PHASE_CASE
    mdl = _km2_model()
    m = _mass_grid(case["m1"], case["m2"], n=65)
    params = _phys_params("k2", **case)

    tr = mdl.make_basis_transform()
    resolved = tr.apply_forward(dict(params))

    num_rows = mdl.num_basis(m)
    den_rows = mdl.den_basis(m)
    num = np.zeros_like(m, dtype=complex)
    for weight, row in zip(mdl.num_weights(resolved), num_rows):
        num = num + weight * row
    den = np.zeros_like(m, dtype=complex)
    for name, row in zip(mdl.get_den_name(), den_rows):
        den = den + resolved[name] * row

    rng = np.random.default_rng(3)
    c = rng.normal(size=m.size) + 1j * rng.normal(size=m.size)

    # real objective Q = Re(Σ_e c_e·A_e); real cotangents for every slot.
    # The numerator outputs are polar (mag, phase) of n_i, and
    # Q = Re(S_i·n_i) with S_i = Σ c·N_i/D, so
    #   ∂Q/∂mag = Re(S·e^{iθ}),   ∂Q/∂phase = −mag·Im(S·e^{iθ})
    n0m, n0p, n1m, n1p = mdl.num_slot_names()
    cot = {}
    for row, mag_name, phase_name in (
            (num_rows[0], n0m, n0p),
            (num_rows[1], n1m, n1p)):
        s = np.sum(c * row / den)
        mag = float(resolved[mag_name])
        theta = float(resolved[phase_name])
        cot[mag_name] = float(np.real(s * np.exp(1j * theta)))
        cot[phase_name] = float(-mag * np.imag(s * np.exp(1j * theta)))
    for name, row in zip(mdl.get_den_name(), den_rows):
        cot[name] = float(np.real(np.sum(c * (-num * row / den ** 2))))

    vjp = tr.backward(cot, d_in=dict(params))

    def q(x):
        return float(np.real(np.sum(c * mdl.amp(m, x))))

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


# ── 6. registration, tail scoping & phase degeneracy ─────────────────

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
    # magnitude/phase slots (default b = 1·e^{i0} = 1)
    assert defaults["k2_b1r"] == pytest.approx(1.0)    # magnitude
    assert defaults["k2_b1i"] == pytest.approx(0.0)    # phase
    assert defaults["k2_b2r"] == pytest.approx(1.0)
    assert defaults["k2_b2i"] == pytest.approx(0.0)

    # complex kwarg converted to (|b|, arg b) slots
    mdl2 = build_particle("k2", model="kmatrix2", mass=0.8, width=0.3,
                          m2=1.9, g1=0.4,
                          b1=0.8 * np.exp(0.4j), b2=1.2 * np.exp(-1.1j))
    d2 = mdl2.get_defaults()
    assert d2["k2_m2"] == pytest.approx(1.9)
    assert d2["k2_g1"] == pytest.approx(0.4)
    assert d2["k2_g2"] == pytest.approx(0.15)
    assert d2["k2_b1r"] == pytest.approx(0.8)
    assert d2["k2_b1i"] == pytest.approx(0.4)
    assert d2["k2_b2r"] == pytest.approx(1.2)
    assert d2["k2_b2i"] == pytest.approx(-1.1)


def test_complex_tail_scope_renames_coupling_slots():
    case = PHASE_CASE
    m = _mass_grid(case["m1"], case["m2"])

    with scope(complex_tail=("rho", "phi")):
        mdl = _km2_model()
        assert mdl.tail == ("rho", "phi")
        tr = mdl.make_basis_transform()
        assert tr.input_names == [
            "k2_m1", "k2_m2", "k2_g1", "k2_g2",
            "k2_b1rho", "k2_b1phi", "k2_b2rho", "k2_b2phi",
        ]
        # numerator heads are tail-independent; the expanded slots rename
        assert mdl.get_num_name() == ["k2_n0", "k2_n1"]
        assert mdl.num_slot_names() == [
            "k2_n0rho", "k2_n0phi", "k2_n1rho", "k2_n1phi",
        ]
        assert tr.output_names == mdl.num_slot_names() + mdl.get_den_name()
        defaults = mdl.get_defaults()
        for name in tr.input_names:
            assert name in defaults
        params = _phys_params("k2", **case)     # follows complex_tail()
        assert "k2_b1rho" in params and "k2_b1phi" in params
        got = mdl.amp(m, params)

    # same physical b, default-named slots → same amplitude
    ref_mdl = _km2_model()
    ref = ref_mdl.amp(m, _phys_params("k2", **case))
    assert np.allclose(got, ref, rtol=1e-13, atol=1e-15)


def test_common_phase_is_degenerate_with_wave_coupling():
    """b_i → e^{iα} b_i scales the numerator ⇒ |A| unchanged."""
    case = PHASE_CASE
    alpha = 0.37
    shifted = dict(case,
                   b1=case["b1"] * np.exp(1j * alpha),
                   b2=case["b2"] * np.exp(1j * alpha))
    # arg(b2/b1) is invariant under the common phase
    assert np.angle(complex(shifted["b2"]) / complex(shifted["b1"])) \
        == pytest.approx(np.angle(complex(case["b2"]) / complex(case["b1"])))

    mdl = _km2_model()
    m = _mass_grid(case["m1"], case["m2"])
    a1 = mdl.amp(m, _phys_params("k2", **case))
    a2 = mdl.amp(m, _phys_params("k2", **shifted))
    assert np.allclose(np.abs(a1), np.abs(a2), rtol=1e-12, atol=1e-14)
    # the shifted amplitude is the original times the common phase
    assert np.allclose(a2, np.exp(1j * alpha) * a1,
                       rtol=1e-12, atol=1e-14)
