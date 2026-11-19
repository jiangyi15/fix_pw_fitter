"""Tests for the rational-lineshape basis interface (``BasisModel``) and the
``kmatrix_n`` reference model.

Particle layer only: these tests exercise the amplitude algebra, the real-slot
physical-parameter → basis-weight transform (pole masses/widths plus the
numerator ``beta`` magnitude/phase slots, the ``ck`` convention), and the VJP
the kernel gradient chain will rely on.  No kernel / backend / fitter
involvement.
"""

import numpy as np
import pytest

from tabpwa.build_defaults import scope
from tabpwa.param_constraint import CKProduct, complex_tail
from tabpwa.particle_model import ALL_MODELS, build_particle
from tabpwa.particle_model.base import (
    BaseModel, BasisModel, DefaultBasisModel)
from tabpwa.particle_model.bwr_model import phase_space
from tabpwa.particle_model.kmatrix_n_model import KMatrixNModel


# ── helpers ──────────────────────────────────────────────────────────

#: the two-pole reference content, and a non-trivial production coupling
#: (``beta1 = mag·exp(i·phase)``; ``beta0 ≡ 1`` is the fixed gauge head)
MASSES = [0.80, 1.35]
WIDTHS = [0.35, 0.55]
BETA = 0.8 * np.exp(0.4j)

#: the p-wave phase space the ``kmatrix_n`` gates use, and the daughter
#: masses its no-tree fallback reads
L_KN = 1
D_KN = 3.0
D2, D3 = 0.13957039, 0.1349768


def _kn_model(name="k", masses=MASSES, widths=WIDTHS, **kw):
    return build_particle(name, model="kmatrix_n",
                          mass_list=list(masses), width_list=list(widths),
                          L=kw.pop("L", L_KN), **kw)


def _kn_params(name, masses, widths, beta=None, tail=None):
    """The real input slots of a (two-)pole model.

    ``mass{i}``/``width{i}`` for every pole; ``beta1{mag,phase}`` only when
    there is a second pole — ``beta0`` is fixed to ``1 + 0i`` and is not a
    parameter.  The suffixes follow the active ``complex_tail()`` unless given.
    """
    mag, phase = tuple(tail) if tail is not None else complex_tail()
    d = {}
    for i, (mm, ww) in enumerate(zip(masses, widths)):
        d[f"{name}_mass{i}"] = float(mm)
        d[f"{name}_width{i}"] = float(ww)
    if len(masses) > 1:
        b = complex(beta)
        d[f"{name}_beta1{mag}"] = abs(b)
        d[f"{name}_beta1{phase}"] = float(np.angle(b))
    return d


def _num_weights(mdl, params):
    """Complex numerator weights as the model reads them (head or ck pair)."""
    mag, ph = complex_tail()
    out = []
    for h in mdl.get_num_name():
        if h in params:
            out.append(complex(params[h]))
        else:
            out.append(float(params.get(f"{h}{mag}", 0.0))
                       * np.exp(1j * float(params.get(f"{h}{ph}", 0.0))))
    return out or [1.0]


def _mass_grid(*poles, n=241, lo=0.30, hi=1.70):
    """Grid spanning below threshold and both poles (poles excluded)."""
    m = np.linspace(lo, hi, n)
    for p in poles:
        m = m[np.abs(m - p) > 1e-4]
    return m


def _direct_kn(m, masses, widths, beta=None, L=L_KN, d=D_KN, d2=D2, d3=D3):
    """Independent direct evaluation of ``P / (1 − i·ρ·K)``.

    ``K = Σ g_i²/(m_i² − s)``, ``P = Σ b_i/(m_i² − s)`` with the fixed gauge
    head ``b_0 = 1`` and ``g_i² = m_i·Γ_i/ρ(m_i)`` (the model's width
    convention).  The ``-i`` lives in the basis row, so only ``ρ`` at the pole
    is real there.
    """
    n = len(masses)
    assert n <= 2, "this reference helper covers the gates' 1- and 2-pole cases"
    s = np.asarray(m, dtype=float) ** 2
    rho = np.asarray(phase_space(np.asarray(m, dtype=float), L, d, d2, d3),
                     dtype=float)
    rho_pole = np.asarray(phase_space(np.asarray(masses, dtype=float),
                                      L, d, d2, d3), dtype=float)
    g2 = [masses[i] * widths[i] / rho_pole[i] for i in range(n)]
    b = [1.0 + 0.0j] + ([complex(beta)] if n > 1 else [])
    k = sum(g2[i] / (masses[i] ** 2 - s) for i in range(n))
    p = sum(b[i] / (masses[i] ** 2 - s) for i in range(n))
    return p / (1.0 - 1j * rho * k)


# ── 1. BasisModel defaults reproduce the plain BW bit-for-bit ────────

def test_basis_default_matches_bw_bitwise():
    m = np.linspace(0.30, 1.20, 57)
    params = {"r_mass": 0.775, "r_width": 0.149}

    bw = build_particle("r", model="BW", mass=0.775, width=0.149)
    basis = DefaultBasisModel("r", mass=0.775, width=0.149)

    ref = bw.amplitude_raw(m, params)
    got = basis.amp(m, params)
    assert np.array_equal(ref.view(np.float64), got.view(np.float64))

    # the basis decomposition is the kernel's coeff basis:
    # [1, m², −i·γ] with weights [m0², −1, m0·g0]
    den = basis.den_basis(m)
    assert len(den) == len(basis.get_den_name()) == 3
    # the m² weight is the SHARED literal "one" (value -1), not "r_poly":
    # `poly` is the K-matrix family's FITTED polynomial coefficient
    # -(m1^2+m2^2) — a different quantity with a different free/fixed status.
    # the kernel-facing m0*g slot carries the ``_t`` suffix so it can never
    # collide with the physical fit parameter of the same base name
    assert basis.get_den_name() == ["r_s0", "one", "r_width_t"]
    assert np.array_equal(den[0], np.ones_like(m, dtype=complex))
    # the row is -m² so its weight ``"one"`` is +1 (name reads as its value)
    assert np.array_equal(den[1], -(m ** 2) + 0j)
    assert np.array_equal(den[2], -1j * np.ones_like(m))

    # the two interfaces agree: basis route (amp, its transform) and
    # running-width route (amplitude, its mass/width transform)
    assert np.allclose(basis.amp(m, params), basis.amplitude(m, params),
                       rtol=0, atol=0)

    # build_particle("BW", ...) is untouched and still works
    bw2 = build_particle("r", model="BW", mass=0.700, width=0.100)
    assert isinstance(bw2, BaseModel)
    assert "BW" in ALL_MODELS
    assert bw2.name == "r"


def test_missing_weights_use_the_model_defaults():
    """``amp(m, {})`` is the model at its configured defaults.

    The denominator names come from the model's own transform, which fills
    the configured mass/width; an explicitly supplied input still wins.
    """
    mdl = DefaultBasisModel("r", mass=0.775, width=0.149)
    m = np.array([0.5, 0.9])
    got = mdl.amp(m, {})
    ref = 1.0 / (0.775 ** 2 - m ** 2 - 1j * 0.775 * 0.149)
    assert np.allclose(got, ref, rtol=0, atol=0)

    # an explicit input overrides the default (here: a zero width)
    got0 = mdl.amp(m, {"r_width": 0.0})
    assert np.allclose(got0, 1.0 / (0.775 ** 2 - m ** 2), rtol=0, atol=0)


def test_amplitude_uses_the_gamma_names_not_the_basis():
    """``amplitude`` states the denominator with ``gamma()``.

    Same value as the basis route, but computed from ``get_gamma_name()`` /
    ``gamma()`` — the interface ``BaseModel`` owns — so the two entries do not
    share an implementation that could drift.
    """
    mdl = DefaultBasisModel("r", mass=0.775, width=0.149)
    m = np.array([0.5, 0.9])
    params = {"r_mass": 0.775, "r_width": 0.149}
    ref = 1.0 / (0.775 ** 2 - m ** 2 - 1j * 0.775 * 0.149)   # gamma(m) = 1
    assert np.allclose(mdl.amplitude(m, params), ref, rtol=0, atol=0)
    assert np.allclose(mdl.amplitude(m, params), mdl.amplitude_raw(m, params),
                       rtol=0, atol=0)


def test_denominator_weights_real_rows_complex():
    """Denominator weights are real; all i factors live in the basis rows."""
    # -- default BasisModel: kernel coeff weights [m0², −1, m0·g0] --
    mdl = DefaultBasisModel("r", mass=0.775, width=0.149)
    m = np.linspace(0.30, 1.20, 23)
    rows = mdl.den_basis(m)
    assert all(np.iscomplexobj(r) for r in rows)
    assert any(np.any(np.imag(r) != 0) for r in rows)
    g = mdl.make_basis_transform().forward({"r_mass": 0.775, "r_width": 0.149})
    assert all(np.isrealobj(v) for v in g.values())
    assert np.all(np.imag(_num_weights(mdl, {})) == 0)

    # -- kmatrix_n at several random pole / beta points --
    km = _kn_model()
    rng = np.random.default_rng(11)
    for _ in range(5):
        masses = [rng.uniform(0.60, 1.00), rng.uniform(1.10, 1.60)]
        widths = [rng.uniform(0.05, 0.80), rng.uniform(0.05, 0.80)]
        beta = rng.uniform(0.10, 1.00) * np.exp(
            1j * rng.uniform(-np.pi, np.pi))
        params = _kn_params("k", masses, widths, beta)
        out = km.make_basis_transform().apply_forward(dict(params))
        # every transform output is a real float (fd_backward contract)
        assert all(np.isrealobj(v) for v in out.values())
        gk = [out[name] for name in km.get_den_name()]
        assert np.all(np.imag(gk) == 0), gk
        # numerator outputs are real Cartesian/polar slots
        assert all(np.isrealobj(out[name]) for name in km.num_slot_names())

    m = _mass_grid(*MASSES)
    rows = km.den_basis(m)
    assert all(np.iscomplexobj(r) for r in rows)
    assert any(np.any(np.imag(r) != 0) for r in rows)
    assert all(np.iscomplexobj(r) for r in km.num_basis(m))


def test_phases_flow_through_numerator_path():
    """The ``beta`` (magnitude, phase) pair reproduces the complex coupling."""
    mdl = _kn_model()
    m = _mass_grid(*MASSES)
    params = _kn_params("k", MASSES, WIDTHS, BETA)

    tr = mdl.make_basis_transform()
    out = tr.apply_forward(dict(params))
    assert all(np.isrealobj(v) for v in out.values())

    b1 = complex(BETA)
    m0s, m1s = MASSES[0] ** 2, MASSES[1] ** 2
    # numerator coefficients: beta0 = 1 fixed, beta1 the fitted coupling
    n0 = m1s + b1 * m0s
    n1 = -(1.0 + b1)
    n0m, n0p, n1m, n1p = mdl.num_slot_names()
    # numerator outputs use the (magnitude, phase) tail convention
    assert out[n0m] == pytest.approx(abs(n0), rel=1e-14, abs=1e-15)
    assert out[n0p] == pytest.approx(np.angle(n0), rel=1e-14, abs=1e-15)
    assert out[n1m] == pytest.approx(abs(n1), rel=1e-14, abs=1e-15)
    assert out[n1p] == pytest.approx(np.angle(n1), rel=1e-14, abs=1e-15)
    # denominator weights stay real
    assert np.all(np.imag([out[name] for name in mdl.get_den_name()]) == 0)

    # the complex numerator weights, read from the transform's slots
    nw = _num_weights(mdl, out)
    assert nw[0] == pytest.approx(n0, rel=1e-14, abs=1e-15)
    assert nw[1] == pytest.approx(n1, rel=1e-14, abs=1e-15)

    ref = _direct_kn(m, MASSES, WIDTHS, BETA)
    assert np.allclose(mdl.amp(m, params), ref, rtol=1e-12, atol=1e-14)
    # amp_raw takes the weights as given, i.e. the transform's output
    assert np.allclose(mdl.amp_raw(m, out), ref, rtol=1e-12, atol=1e-14)


def test_transform_outputs_feed_ckproduct():
    """Numerator outputs are ``ck`` slots: ``build_ck(out)`` reproduces n_i.

    This is the ``ck_map`` contract: the transform output dict can be
    handed straight to ``CKProduct.build_ck``, which reads the
    ``(magnitude, phase)`` pair and forms ``r·exp(iθ)``.
    """
    mdl = _kn_model()
    m = _mass_grid(*MASSES)
    params = _kn_params("k", MASSES, WIDTHS, BETA)

    tr = mdl.make_basis_transform()
    out = tr.apply_forward(dict(params))

    tail = mdl.tail
    heads = mdl.get_num_name()
    assert heads == [f"{mdl.name}_n0", f"{mdl.name}_n1"]
    # one CKProduct over the head names: the tail is applied by the
    # consumer, which looks up the expanded slots the transform produced
    w0, w1 = CKProduct([[heads[0]], [heads[1]]], tail=tail).build_ck(out)

    b1 = complex(BETA)
    m0s, m1s = MASSES[0] ** 2, MASSES[1] ** 2
    n0 = m1s + b1 * m0s
    n1 = -(1.0 + b1)
    assert w0 == pytest.approx(n0, rel=1e-14, abs=1e-15)
    assert w1 == pytest.approx(n1, rel=1e-14, abs=1e-15)

    # full chain: weights → build_ck → basis ratio → amplitude
    num_rows = mdl.num_basis(m)
    den_rows = mdl.den_basis(m)
    den = np.zeros_like(m, dtype=complex)
    for name, row in zip(mdl.get_den_name(), den_rows):
        den = den + out[name] * row
    amp = (w0 * num_rows[0] + w1 * num_rows[1]) / den

    ref = _direct_kn(m, MASSES, WIDTHS, BETA)
    assert np.allclose(amp, ref, rtol=1e-12, atol=1e-14)
    assert np.allclose(mdl.amp(m, params), amp, rtol=1e-12, atol=1e-14)


def test_bare_head_real_numerator_weight():
    """A real numerator weight may be given as one parameter named the head."""
    mdl = _kn_model()
    m = _mass_grid(*MASSES)
    params = {"k_n0": 2.5, "k_n1": -0.7}

    nw = _num_weights(mdl, params)
    assert nw[0] == pytest.approx(2.5, rel=0, abs=0)
    assert nw[1] == pytest.approx(-0.7, rel=0, abs=0)

    # amplitude with the bare-head path vs the manual basis ratio
    resolved = mdl.make_basis_transform().apply_forward(dict(params))
    den = np.zeros_like(m, dtype=complex)
    for name, row in zip(mdl.get_den_name(), mdl.den_basis(m)):
        den = den + resolved[name] * row
    num_rows = mdl.num_basis(m)
    amp = (2.5 * num_rows[0] - 0.7 * num_rows[1]) / den
    # amp_raw reads the weights as given: keep the bare head, drop the
    # expanded pair the transform would produce
    given = dict(resolved)
    for n in mdl.num_slot_names():
        given.pop(n, None)
    given.update(params)                 # keep the caller's bare heads
    assert np.allclose(mdl.amp_raw(m, given), amp, rtol=0, atol=0)


# ── 2. basis reconstruction == amplitude_raw ─────────────────────────

def test_basis_route_equals_raw_amplitude():
    mdl = _kn_model()
    m = _mass_grid(*MASSES)
    params = _kn_params("k", MASSES, WIDTHS, BETA)

    tr = mdl.make_basis_transform()
    resolved = tr.apply_forward(dict(params))

    num_rows = mdl.num_basis(m)
    den_rows = mdl.den_basis(m)
    # two poles: 2 numerator rows, 2N+1 = 5 denominator rows
    assert len(num_rows) == 2
    assert len(mdl.get_num_name()) == 2      # one head per basis row
    assert len(mdl.num_slot_names()) == 4    # expanded (mag, phase) slots
    assert len(den_rows) == len(mdl.get_den_name()) == 5
    assert all(np.iscomplexobj(r) for r in num_rows + den_rows)
    assert all(np.isrealobj(resolved[n]) for n in mdl.num_slot_names())
    assert np.all(np.imag([resolved[n] for n in mdl.get_den_name()]) == 0)

    num = np.zeros_like(m, dtype=complex)
    for weight, row in zip(_num_weights(mdl, resolved), num_rows):
        num = num + weight * row
    den = np.zeros_like(m, dtype=complex)
    for name, row in zip(mdl.get_den_name(), den_rows):
        den = den + resolved[name] * row
    route = num / den

    raw = mdl.amp_raw(m, resolved)
    assert np.allclose(route, raw, rtol=1e-12, atol=1e-14)
    assert np.allclose(mdl.amp(m, params), raw, rtol=1e-12, atol=1e-14)


def test_amplitude_uses_transform_defaults_for_empty_params():
    """A dict without the ``beta`` slots resolves to the transform fallback."""
    mdl = _kn_model()
    m = _mass_grid(*MASSES, n=241)[::8]
    params = mdl.get_defaults()
    assert np.allclose(mdl.amp(m, {}), mdl.amp(m, params),
                       rtol=1e-14, atol=0)
    # the fallback is beta1 = 1·e^{i0}, i.e. the same point written out
    full = {**params, "k_beta1r": 1.0, "k_beta1i": 0.0}
    assert np.allclose(mdl.amp(m, {}), mdl.amp(m, full), rtol=1e-14, atol=0)


# ── 3. transform forward: exact weights + name alignment ─────────────

def test_transform_forward_weights_and_names():
    mdl = _kn_model()
    tr = mdl.make_basis_transform()
    # default complex_tail() = ("r", "i"): (magnitude, phase) input slots
    assert tr.input_names == [
        "k_mass0", "k_mass1", "k_width0", "k_width1",
        "k_beta1r", "k_beta1i",
    ]
    assert mdl.get_num_name() == ["k_n0", "k_n1"]       # heads (one per row)
    assert mdl.num_slot_names() == [
        "k_n0r", "k_n0i", "k_n1r", "k_n1i",
    ]
    assert tr.output_names == mdl.num_slot_names() + mdl.get_den_name()

    params = _kn_params("k", MASSES, WIDTHS, BETA)
    out = tr.apply_forward(dict(params))

    m0s, m1s = MASSES[0] ** 2, MASSES[1] ** 2
    rho_pole = phase_space(np.asarray(MASSES), L_KN, D_KN, D2, D3)
    g2 = [MASSES[i] * WIDTHS[i] / rho_pole[i] for i in range(2)]
    b1 = complex(BETA)
    # numerator coefficients of (b0·(m1²−s) + b1·(m0²−s)); b0 = 1
    n0 = m1s + b1 * m0s
    n1 = -(1.0 + b1)
    expected = {
        "k_n0r": abs(n0),                          # numerator magnitude
        "k_n0i": np.angle(n0),                     # numerator phase
        "k_n1r": abs(n1),
        "k_n1i": np.angle(n1),
        "k_s0": m0s * m1s,                         # Π(m_i²) → s^0 row
        "k_poly": -(m0s + m1s),                    # s^1 row
        "k_m4": 1.0,                               # s² row
        "k_rho": g2[0] * m1s + g2[1] * m0s,        # −i·ρ row
        "k_srho": -(g2[0] + g2[1]),                # −i·s·ρ row
    }
    assert set(expected) == set(tr.output_names)
    for name, val in expected.items():
        assert out[name] == pytest.approx(val, rel=1e-13, abs=1e-15)
        # every output is a real float (i factors live in the basis rows)
        assert np.imag(out[name]) == 0
        assert np.isrealobj(out[name])


# ── 4. VJP of the real input slots (FD reference) ────────────────────

def test_kmatrix_n_transform_backward_matches_finite_difference():
    """fd_backward runs on real slots (incl. the beta phase) and matches FD."""
    mdl = _kn_model()
    params = _kn_params("k", MASSES, WIDTHS, BETA)
    tr = mdl.make_basis_transform()
    out = tr.apply_forward(dict(params))
    assert all(np.isrealobj(out[n]) for n in tr.output_names)

    rng = np.random.default_rng(7)
    cot = {name: float(rng.normal()) for name in tr.output_names}

    vjp = tr.backward(cot, d_in=dict(params))
    assert set(vjp) == set(tr.input_names)

    def scalar(x):
        o = tr.forward(x)
        return sum(cot[name] * o[name] for name in tr.output_names)

    for name in tr.input_names:
        x = float(params[name])
        # a reference step an order above ``fd_eps`` so the two centered
        # differences are independent evaluations of the same derivative
        h = 1e-5 * max(1.0, abs(x))
        fd = (scalar({**params, name: x + h})
              - scalar({**params, name: x - h})) / (2.0 * h)
        assert abs(vjp[name] - fd) <= 1e-8 * max(abs(fd), 1.0), (
            name, vjp[name], fd)

    # apply_backward: outputs leave the dict, inputs accumulate
    grad = {**cot, "k_mass0": 0.0}
    tr.apply_forward(dict(params))
    tr.apply_backward(grad)
    for name in tr.output_names:
        assert name not in grad
    for name in tr.input_names:
        assert grad[name] == pytest.approx(vjp[name], rel=1e-12, abs=1e-12)

    # the phase slot is genuinely differentiated (nonzero derivative)
    assert abs(vjp["k_beta1i"]) > 0


def test_amplitude_gradient_chain_with_real_cotangents():
    """The real-slot VJP reproduces the amplitude gradient chain."""
    mdl = _kn_model()
    m = _mass_grid(*MASSES, n=65)
    params = _kn_params("k", MASSES, WIDTHS, BETA)

    tr = mdl.make_basis_transform()
    resolved = tr.apply_forward(dict(params))

    num_rows = mdl.num_basis(m)
    den_rows = mdl.den_basis(m)
    num = np.zeros_like(m, dtype=complex)
    for weight, row in zip(_num_weights(mdl, resolved), num_rows):
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
        # an order above ``fd_eps``: the reference must not repeat the
        # transform's own step, or the two finite differences would share
        # their truncation error instead of cross-checking the chain
        h = 1e-5 * max(1.0, abs(x))
        p_plus = dict(params)
        p_minus = dict(params)
        p_plus[name] = x + h
        p_minus[name] = x - h
        fd = (q(p_plus) - q(p_minus)) / (2.0 * h)
        rel = abs(vjp[name] - fd) / max(abs(fd), 1e-12)
        assert rel < 1e-5, (name, vjp[name], fd)


# ── 5. registration, tail scoping ────────────────────────────────────

def test_kmatrix_n_registration_and_defaults():
    assert "kmatrix_n" in ALL_MODELS
    assert ALL_MODELS["kmatrix_n"] is KMatrixNModel

    mdl = _kn_model()
    assert isinstance(mdl, KMatrixNModel)

    # ONLY the pole anchors have defaults: a mass/width is physics, while a
    # beta is a COUPLING (random start, like g_ls) and beta0 is the fixed
    # gauge head — so neither appears here
    defaults = mdl.get_defaults()
    assert set(defaults) == {"k_mass0", "k_mass1", "k_width0", "k_width1"}
    assert defaults["k_mass0"] == pytest.approx(0.80)
    assert defaults["k_mass1"] == pytest.approx(1.35)
    assert defaults["k_width0"] == pytest.approx(0.35)
    assert defaults["k_width1"] == pytest.approx(0.55)

    tr = mdl.make_basis_transform()
    assert set(tr.input_names) == set(defaults) | {"k_beta1r", "k_beta1i"}
    # ... so the pole anchors alone are a valid amplitude (beta falls back)
    m = _mass_grid(*MASSES)
    amp0 = mdl.amp(m, {})
    assert np.all(np.isfinite(amp0))
    assert np.allclose(amp0, mdl.amp(m, defaults), rtol=0, atol=0)


def test_complex_tail_scope_renames_coupling_slots():
    m = _mass_grid(*MASSES)

    with scope(complex_tail=("rho", "phi")):
        mdl = _kn_model()
        assert mdl.tail == ("rho", "phi")
        tr = mdl.make_basis_transform()
        assert tr.input_names == [
            "k_mass0", "k_mass1", "k_width0", "k_width1",
            "k_beta1rho", "k_beta1phi",
        ]
        # numerator heads are tail-independent; the expanded slots rename
        assert mdl.get_num_name() == ["k_n0", "k_n1"]
        assert mdl.num_slot_names() == [
            "k_n0rho", "k_n0phi", "k_n1rho", "k_n1phi",
        ]
        assert tr.output_names == mdl.num_slot_names() + mdl.get_den_name()
        params = _kn_params("k", MASSES, WIDTHS, BETA)   # follows complex_tail()
        assert "k_beta1rho" in params and "k_beta1phi" in params
        # the renamed slots are transform inputs, and only they are NOT
        # pole anchors (the beta is deliberately not a default)
        assert set(tr.input_names) - set(mdl.get_defaults()) == {
            "k_beta1rho", "k_beta1phi"}
        got = mdl.amp(m, params)

    # same physical beta, default-named slots → same amplitude
    ref_mdl = _kn_model()
    ref = ref_mdl.amp(m, _kn_params("k", MASSES, WIDTHS, BETA))
    assert np.allclose(got, ref, rtol=1e-13, atol=1e-15)


# ── the interface split, pinned ─────────────────────────────────────

def test_basis_model_is_the_interface_and_knows_no_mass_or_width():
    """``BasisModel`` must not carry the pwa protocol.

    The two interfaces are parallel: ``gamma`` / ``get_gamma_name`` /
    ``get_gamma_count`` / ``get_bw_params`` / ``make_mass_width_transform``
    belong to ``BaseModel``, which also supplies the default basis.  A
    basis-only model (``kmatrix_n``) therefore has none of them.
    """
    PWA = ("gamma", "get_gamma_name", "get_gamma_count", "get_bw_params",
           "make_mass_width_transform")
    for name in PWA:
        assert not hasattr(BasisModel, name), (
            f"BasisModel must not define {name!r} — that is the pwa interface")
    # ... and kmatrix_n, the basis-only model, lacks them too
    km = build_particle("k", model="kmatrix_n",
                        mass_list=list(MASSES), width_list=list(WIDTHS), L=L_KN)
    assert isinstance(km, BasisModel) and not isinstance(km, BaseModel)
    for name in PWA:
        assert not hasattr(km, name), f"kmatrix_n must not have {name!r}"


def test_amp_is_a_real_method_not_a_stale_alias():
    """Overriding ``amp_raw`` must change ``amp``.

    ``amp = amp_raw`` at class level binds the FUNCTION, so a subclass
    override would leave ``amp`` returning the base body — a silent
    wrong-physics answer rather than an error.
    """
    class Override(BasisModel):
        def num_basis(self, m):
            return [np.ones_like(np.asarray(m, float), dtype=complex)]

        def den_basis(self, m):
            return [np.ones_like(np.asarray(m, float), dtype=complex)]

        def get_den_name(self):
            return ["one"]

        def amp_raw(self, m, params=None):
            return np.full(np.asarray(m, float).shape, 42.0 + 0j)

    m = np.array([0.5, 0.9])
    got = Override("S")
    assert np.allclose(np.asarray(got.amp_raw(m)), 42.0)
    assert np.allclose(np.asarray(got.amp(m)), 42.0), (
        "amp did not follow the amp_raw override — it is a stale alias")


def test_rho_omega_gs_basis_transform_is_self_contained():
    """Physical parameters → basis weights, for EVERY model.

    ``RhoOmegaGS`` derives its 8 couplings ``w0..w7`` from four physical
    parameters, and the basis weights are ``m0·w_k``.  The basis transform
    holds that stage itself (``self._mw``) and declares the PHYSICAL inputs,
    so no second transform has to be registered before it — the mapping is
    complete on its own, which is what ``make_basis_transform()`` promises.

    Two failures this pins, both measured:
    * declaring the couplings as inputs instead (the stage's OUTPUTS) made the
      transform correct only if the producer happened to be registered first;
    * replacing ``d`` with the stage's return value dropped ``{name}_mass``
      (a transform returns only its outputs), so ``m0`` fell back to the
      config value — dQ/dm = -6.9e4 against a true -77.6.
    """
    d = {"ro_mass": 0.775, "ro_width": 0.149,
         "ro_mass2": 0.78266, "ro_width2": 0.00868}
    mdl = build_particle("ro", model="RhoOmegaGS",
                         mass=d["ro_mass"], width=d["ro_width"],
                         mass2=d["ro_mass2"], width2=d["ro_width2"])
    tr = mdl.make_basis_transform()
    assert list(tr.input_names) == list(d), (
        "the basis transform must declare the PHYSICAL inputs")

    out = tr.apply_forward(dict(d))
    w = [float(out[f"ro_w{k}_t"]) for k in range(8)]
    assert any(abs(x) > 1e-12 for x in w), "every basis weight resolved to 0"

    # the FITTED mass enters s0 and the m0·g products
    moved = {**d, "ro_mass": d["ro_mass"] * 1.01}
    out2 = tr.apply_forward(moved)
    assert out2["ro_s0"] != out["ro_s0"]
    assert [out2[f"ro_w{k}_t"] for k in range(8)] != [
        out[f"ro_w{k}_t"] for k in range(8)]


def test_rho_omega_gs_pwa_and_basis_paths_agree():
    """Same physics, two interfaces: the lineshapes must be identical."""
    d = {"ro_mass": 0.775, "ro_width": 0.149,
         "ro_mass2": 0.78266, "ro_width2": 0.00868}
    mdl = build_particle("ro", model="RhoOmegaGS",
                         mass=d["ro_mass"], width=d["ro_width"],
                         mass2=d["ro_mass2"], width2=d["ro_width2"])
    m = np.array([0.60, 0.775, 1.00, 1.40])
    a_pwa = np.asarray(mdl.amplitude(m, dict(d)), dtype=complex)
    a_bas = np.asarray(mdl.amp(m, dict(d)), dtype=complex)
    assert np.allclose(a_pwa, a_bas, rtol=1e-12, atol=1e-14), (
        f"pwa and basis paths disagree: {a_pwa} vs {a_bas}")


def test_plain_basis_transform_backward_is_analytic_and_exact():
    """The default basis transform carries its own VJP, and it is exact.

    ``s0 = m0²`` and ``{g}_t = m0·g`` are trivial to differentiate, so the
    derivative is written down.  Inheriting ``fd_backward`` would cost
    ``2·len(input_names)`` forwards per call (18 on ``RhoOmegaGS``) and inject
    finite-difference noise into a chain the fitter differentiates every
    iteration — which is also why an exact VJP can be checked here against
    central differences at 1e-9 rather than at FD accuracy.
    """
    from tabpwa.param_constraint import Transform

    d = {"ro_mass": 0.775, "ro_width": 0.149,
         "ro_mass2": 0.78266, "ro_width2": 0.00868}
    mdl = build_particle("ro", model="RhoOmegaGS",
                         mass=d["ro_mass"], width=d["ro_width"],
                         mass2=d["ro_mass2"], width2=d["ro_width2"])
    tr = mdl.make_basis_transform()
    assert type(tr).backward is not Transform.backward, (
        "the default basis transform still uses the finite-difference VJP")

    phys = mdl.make_mass_width_transform().apply_forward(dict(d))
    d_in = {k: float(phys[k]) for k in tr.input_names}
    rng = np.random.default_rng(3)
    cot = {n: float(rng.normal()) for n in tr.output_names}
    vjp = tr.backward(cot, d_in=d_in)
    assert set(vjp) == set(tr.input_names)

    def scalar(dd):
        o = tr.forward(dd)
        return sum(cot[n] * o[n] for n in tr.output_names)

    for name in tr.input_names:
        h = 1e-7 * max(1.0, abs(d_in[name]))
        fd = (scalar({**d_in, name: d_in[name] + h})
              - scalar({**d_in, name: d_in[name] - h})) / (2.0 * h)
        assert abs(vjp[name] - fd) <= 1e-7 * max(abs(fd), 1.0), (
            name, vjp[name], fd)

    # the couplings are NOT inputs: their gradient is routed through the
    # model's own stage, so it lands on the physical parameters
    assert not (set(vjp) & {f"ro_w{k}" for k in range(8)})
    assert vjp["ro_mass"] != 0.0
    assert any(vjp[n] != 0.0 for n in ("ro_width", "ro_mass2", "ro_width2"))


def test_plain_basis_transform_requires_its_declared_inputs():
    """A missing declared input is an ERROR, never a zero weight.

    Defaulting a coupling to 0 is how ``RhoOmegaGS``'s basis once collapsed to
    ``m0² − m²`` — up to 1.8x off, with nothing raised anywhere.  The transform
    declares what it reads, so an absent name is a wiring error: ``apply_forward``
    passes only declared inputs, which makes the KeyError name the culprit.
    """
    d = {"ro_mass": 0.775, "ro_width": 0.149,
         "ro_mass2": 0.78266, "ro_width2": 0.00868}
    mdl = build_particle("ro", model="RhoOmegaGS",
                         mass=d["ro_mass"], width=d["ro_width"],
                         mass2=d["ro_mass2"], width2=d["ro_width2"])
    tr = mdl.make_basis_transform()
    phys = dict(d)                      # physical parameters ARE the inputs

    # every declared input present -> the weights exist
    out = tr.apply_forward(dict(phys))
    assert any(abs(out[f"ro_w{k}_t"]) > 1e-12 for k in range(8))

    # a missing PHYSICAL input is an error, not a silently-zero weight
    with pytest.raises(KeyError):
        tr.apply_forward({k: v for k, v in phys.items() if k != "ro_mass"})
    with pytest.raises(KeyError):
        tr.apply_forward({k: v for k, v in phys.items() if k != "ro_width"})
