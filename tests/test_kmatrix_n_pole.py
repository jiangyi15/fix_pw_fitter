"""``kmatrix_n.solve_pole`` — the complex pole ``√s`` where ``den == 0``.

``get_bw_params`` (a real running-width peak) does not exist for a basis
model, and a K-matrix pole is not a peak anyway: it is the zero of
``D(m) = P(m²) − i·sheet·ρ(m)·Q(m²)`` continued into the complex mass plane.
These gates pin

* ``den`` against the basis rows the kernels use (one denominator, two
  spellings),
* the real-input path of ``ρ`` being untouched by the complex one,
* the analytic oracle ``s = m₀² − i·m₀Γ₀`` for a CONSTANT ``ρ`` (both
  sheets, and a complex ``ρ``),
* the zero-width case, where the poles are exactly ``mass_list``,
* a realistic ππ P-wave pole and the N=3 multi-pole case,
* ``kmatrix_n_gs`` against the ``GS_rho`` propagator it degenerates to,
* ``sheet`` being an INT (``1`` physical / ``−1`` second).
"""
import numpy as np
import pytest

from tabpwa.particle_model import build_particle


def _model(model="kmatrix_n", **kw):
    kw.setdefault("L", 1)
    return build_particle("x", model=model, **kw)


def _one(**kw):
    kw.setdefault("mass_list", [0.775])
    kw.setdefault("width_list", [0.149])
    return _model(**kw)


# ── the denominator is the basis denominator ──────────────────────

def test_den_equals_the_basis_rows():
    """``den(m)`` is Σ (den weight)·(den row) — the amplitude's own den."""
    mdl = _model(mass_list=[0.775, 1.45], width_list=[0.149, 0.30])
    tr = mdl.make_basis_transform()
    out = tr.apply_forward({})                       # config defaults
    names = mdl.get_den_name()
    weights = [out[n] for n in names]
    for m in (0.30, 0.775, 1.0, 1.45, 2.5):
        rows = [r[0] for r in mdl.den_basis(np.array([m]))]
        assert mdl.den(m) == pytest.approx(sum(w * r for w, r in
                                               zip(weights, rows)), rel=1e-13)


def test_den_shape_at_a_real_point_is_not_zero():
    """A sanity anchor: away from the pole the denominator is O(Γ)."""
    mdl = _one()
    assert abs(mdl.den(0.775)) > 1e-3


# ── complex ρ: the real path is untouched, the continuation is smooth ──

def test_real_phase_space_path_is_unchanged():
    """Real input keeps the below-threshold clamp, bit for bit."""
    from tabpwa.particle_model.bwr_model import phase_space

    m = np.array([0.20, 0.279, 0.50, 0.775, 2.0])
    a = b = 0.13957039
    s = m ** 2
    p2 = (s - (a + b) ** 2) * (s - (a - b) ** 2)
    q = np.sqrt(np.where(p2 > 0, p2, 0)) / (2 * m)
    want = (q / m) * q ** 2 / (1.0 + (q * 3.0) ** 2)
    got = phase_space(m, 1, 3.0, a, b)
    assert not np.iscomplexobj(got)
    assert np.array_equal(got, want)


# golden values of gs_unit on the REAL axis, taken from the pre-change code:
# the complex branch must not move them (bit-for-bit on this machine, and to
# 1e-14 here so a different libm does not make the gate flaky)
GS_UNIT_GOLDEN = [
    (0.3, 0.025342366437234125, 0.32851573433452463),
    (0.5, 0.4653772571820728, 0.11293122982539094),
    (0.775, 1.0, 0.0),
    (1.5, 1.6115302735985695, 0.9794830234849764),
    (2.5, 1.8305479766522232, 6.422628393811259),
]


def test_gs_unit_real_path_is_unchanged():
    from tabpwa.particle_model.gs_rho_model import gs_unit

    for m, re_, im_ in GS_UNIT_GOLDEN:
        got = gs_unit(np.array([m]), 0.775, 1, 3.0, 0.13957039, 0.1349768)[0]
        assert got.real == pytest.approx(re_, rel=1e-14)
        assert got.imag == pytest.approx(im_, rel=1e-14 if im_ else 0, abs=0)


def test_complex_rho_reduces_to_the_real_value_above_threshold():
    mdl = _one()
    for m in (0.40, 0.775, 1.5):
        real = mdl._rho(np.array([m]))[0]
        near = mdl._rho(np.array([complex(m, 1e-12)]))[0]
        assert near == pytest.approx(real, rel=0, abs=1e-9)


def test_sheet_is_the_square_root_sign():
    """The sheet flips ``q = ±sqrt(p²)/(2m)``: ``ρ(m, −1) = −ρ(m, 1)``.

    ``ρ`` is odd in ``q`` (``q^{2L+1}`` over an even barrier), so the other
    square-root branch is ``−ρ`` — and therefore
    ``den(m, 1) + den(m, −1) = 2·P(m²)``.
    """
    masses = [0.775, 1.45]
    mdl = _model(mass_list=masses, width_list=[0.149, 0.30])
    for m in (0.6, complex(0.775, -0.04), complex(1.3, 0.1)):
        r1 = mdl._rho(np.array([m]), 1)[0]
        rm1 = mdl._rho(np.array([m]), -1)[0]
        assert rm1 == pytest.approx(-r1, rel=1e-14)
        two_P = 2.0 * np.prod([m0 ** 2 - m ** 2 for m0 in masses])
        assert mdl.den(m, 1) + mdl.den(m, -1) == pytest.approx(two_P, rel=1e-13)

    # and the complex ρ really is complex at a complex mass
    assert not np.isrealobj(mdl._rho(np.array([complex(0.775, -0.05)]))[0])


# ── exact oracles ────────────────────────────────────────────────

def test_zero_width_poles_are_exactly_the_mass_list():
    masses = [0.775, 1.45, 1.90]
    mdl = _model(mass_list=masses, width_list=[0.0, 0.0, 0.0])
    poles = mdl.solve_pole()
    assert len(poles) == 3
    for p, m0 in zip(poles, masses):
        assert p["m"] == pytest.approx(complex(m0, 0.0), abs=1e-12)
        assert p["width"] == pytest.approx(0.0, abs=1e-12)
        assert p["converged"] and p["residual"] < 1e-12


def test_constant_rho_is_analytic():
    """With a constant ρ the pole is exactly ``s = m₀² − i·m₀Γ₀``.

    A ρ that does not depend on ``q`` is sheet-BLIND (the sheet is the sign of
    the square root, not an algebraic sign on ρ), so this oracle is the
    physical branch only — the real phase space covers the sheet flip.
    """
    m0, g0 = 0.775, 0.149
    mdl = _one(mass_list=[m0], width_list=[g0])
    mdl._rho = lambda m, sheet=1: np.full(np.shape(m), 1.0, dtype=complex)
    p = mdl.solve_pole()[0]
    assert p["s"] == pytest.approx(complex(m0 ** 2, -m0 * g0), abs=1e-10)
    assert p["m"].imag < 0
    assert p["converged"]
    # ... and the sheet flag cannot move it: nothing in ρ depends on q
    assert mdl.solve_pole(sheet=-1)[0]["s"] == pytest.approx(p["s"], abs=1e-10)


def test_complex_constant_rho_shifts_the_pole_really():
    """A complex ρ = c_r + i·c_i moves Re s by (c_i/c_r)·m₀Γ₀ (still exact)."""
    m0, g0 = 0.775, 0.149
    cr, ci = 1.3, 0.4
    mdl = _one(mass_list=[m0], width_list=[g0])
    mdl._rho = lambda m, sheet=1: np.full(np.shape(m), complex(cr, ci))
    p = mdl.solve_pole()[0]
    want = complex(m0 ** 2 + (ci / cr) * m0 * g0, -m0 * g0)
    assert p["s"] == pytest.approx(want, abs=1e-10)


# ── realistic poles ──────────────────────────────────────────────

def test_pipi_p_wave_pole():
    mdl = _one()
    p = mdl.solve_pole()[0]
    assert p["converged"]
    assert p["m"].imag < 0                      # physical sheet: M − iΓ/2
    assert abs(mdl.den(p["m"])) < 1e-13
    assert p["mass"] == pytest.approx(0.775, abs=0.02)
    assert p["width"] == pytest.approx(0.149, abs=0.02)


def test_second_sheet_is_the_mirror_for_a_real_rho():
    mdl = _one()
    p1 = mdl.solve_pole(sheet=1)[0]
    p2 = mdl.solve_pole(sheet=-1)[0]
    assert p2["m"] == pytest.approx(p1["m"].conjugate(), abs=1e-9)
    assert p2["m"].imag > 0


def test_three_poles_are_all_solved():
    masses = [0.775, 1.45, 1.90]
    mdl = _model(mass_list=masses, width_list=[0.149, 0.30, 0.20])
    poles = mdl.solve_pole()
    assert len(poles) == 3
    for p, m0 in zip(poles, masses):
        assert p["converged"]
        assert abs(mdl.den(p["m"])) < 1e-12
        assert p["mass"] == pytest.approx(m0, abs=0.05)
        assert p["m"].imag < 0


def test_inits_select_the_iteration_starts():
    """``inits`` is the list of complex masses Newton starts from."""
    mdl = _model(mass_list=[0.775, 1.45], width_list=[0.149, 0.30])
    one = mdl.solve_pole(inits=[0.775 - 0.07j])
    assert len(one) == 1 and abs(mdl.den(one[0]["m"])) < 1e-12
    # starting at the second pole's mass finds the second pole
    second = mdl.solve_pole(inits=1.45)[0]
    assert second["mass"] == pytest.approx(1.45, abs=0.05)


def test_sheet_must_be_an_int():
    mdl = _one()
    for bad in ("I", True, 2, 0.0, -1.0):
        with pytest.raises(ValueError, match="sheet must be the int"):
            mdl.solve_pole(sheet=bad)


# ── GS variant ───────────────────────────────────────────────────

def test_gs_unit_accepts_a_complex_mass():
    from tabpwa.particle_model.gs_rho_model import gs_unit

    m = complex(0.775, -0.05)
    val = gs_unit(np.array([m]), 0.775, 1, 3.0, 0.13957039, 0.1349768)
    assert np.iscomplexobj(val) and np.all(np.isfinite(val))
    # the sheet is the square-root sign here too: q -> -q flips the running
    # width AND moves fs through the log branch, so it is NOT exactly -rho
    other = gs_unit(np.array([m]), 0.775, 1, 3.0, 0.13957039, 0.1349768,
                    sheet=-1)
    assert np.isfinite(other).all() and other != pytest.approx(-val, rel=1e-3)
    # continuous with the real-axis value at m = M (Re ρ = 1 exactly there)
    real = gs_unit(np.array([0.775]), 0.775, 1, 3.0, 0.13957039, 0.1349768)
    assert real[0].real == pytest.approx(1.0, abs=1e-12)


def test_gs_one_pole_is_the_gs_propagator():
    """``kmatrix_n_gs`` N=1 with ``width_norm: false`` IS ``GS_rho``.

    ``D_kmatrix(m) = M² − m² − i·M·g0·gamma_GS(m)`` for every complex ``m`` —
    the coupling ``g² = M·Γ0`` and the unit carrying ``1/M`` is exactly the
    GS denominator, so the pole found here is the GS pole.
    """
    from tabpwa.particle_model import build_particle as bp

    gs = bp("x", model="GS_rho", mass=0.775, width=0.149, L=1)
    km = _model("kmatrix_n_gs", mass_list=[0.775], width_list=[0.149],
                width_norm=False)
    for m in (0.70 + 0j, complex(0.775, -0.05), complex(1.2, -0.2)):
        gam = gs.gamma(np.array([m]))[0]
        want = 0.775 ** 2 - m ** 2 - 1j * 0.775 * 0.149 * gam
        assert km.den(m) == pytest.approx(want, rel=1e-12, abs=1e-14)


def test_gs_pole_is_finite_and_converged():
    mdl = _model("kmatrix_n_gs", mass_list=[0.775], width_list=[0.149],
                 width_norm=False)
    p = mdl.solve_pole()[0]
    assert p["converged"] and abs(mdl.den(p["m"])) < 1e-13
    assert p["m"].imag < 0
    assert p["mass"] == pytest.approx(0.775, abs=0.02)
    assert p["width"] == pytest.approx(0.149, abs=0.02)


def test_ieps_reports_the_unregularised_pole():
    """``ieps`` is a real-axis regulator, not part of the pole condition."""
    parent = _one()
    reg = _one(model="kmatrix_n_ieps", ieps=1e-6)
    p0 = parent.solve_pole()[0]
    p1 = reg.solve_pole()[0]
    assert p1["m"] == pytest.approx(p0["m"], abs=1e-6)
