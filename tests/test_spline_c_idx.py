"""``spline_c_idx`` — the COMPLEX cubic spline carried by the NUMERATOR.

Owned by this file rather than ``tests/test_pwa_gen.py``: that file owns the
``pwa_gen`` BACKEND/config contract and is parametrised over the shipped
fixtures, while a particle model's maths belongs beside its siblings
(``tests/test_exp_cheb.py``, ``tests/test_spline_k_model.py``).  The pwa_gen
integration of this model (the expansion count, the tabulated rows, the NLL
gradient) is gated here too, on ``tests/config_pwa_gen_cidx.yml``.

The construction is TFPWA's ``Interp1DSplineIdx`` (``spline_c_idx``): rows from
``spline_xi_matrix`` (tabpwa's verbatim port,
``spline_k_model.spline_basis_matrix``), ``bc_type="not-a-knot"``,
``with_bound`` dropping the end knots, complex knot values, cubic continuation
outside by default (``outside="spline"``; ``"zero"`` is TFPWA's ``spline_c``).

Gates
-----
A. rows: the port against ``scipy.CubicSpline(bc_type=...)`` per unit vector
   (1e-12), an INDEPENDENT tridiagonal solve of the same equations (1e-10),
   partition of unity and the cardinal property at ``points``;
B. ``with_bound``: ``n_free = N−2`` (the end coefficients absent from the
   slots, the end knots at 0) vs ``n_free = N`` (the end knots at ``c_end``);
C. extrapolation: ``outside="spline"`` reproduces the end interval's cubic
   outside the support, ``"zero"`` returns exactly 0, both agree inside;
D. the complex coefficients: TFPWA's slot spelling
   ``{name}_point_{i}r`` / ``{name}_point_{i}i``, one complex coefficient per
   fitted knot, real and imaginary parts moving independently;
E. the refusals (bad/missing points, bad ``bc_type``/``outside``/``fix_idx``,
   ``polar: false``, the plain-``pwa`` path naming ``amp_model: pwa_gen``);
F. pwa_gen: ``len(ck) == n_wave·n_free``, one contiguous block per geometry
   wave, ``num_table`` carrying row ``i``, ``P``/``Q`` from the model's rows,
   and the NLL gradient vs central differences.  (``drop_unit_factors`` and
   its numerator guard are owned by ``tests/test_skip_model.py``.)
"""
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from tabpwa import Fitter
from tabpwa.config_loader import RawConfig
from tabpwa.particle_model import ALL_MODELS
from tabpwa.particle_model.base import BaseModel, BasisModel, build_particle

CFG = "tests/config_pwa_gen_cidx.yml"
POINTS = [0.30, 0.75, 1.20, 1.65, 2.10, 2.50]


def _model(points=POINTS, name="k2", **kw):
    return build_particle(name, model="spline_c_idx", points=list(points), **kw)


def _cfg(path=CFG, **over):
    d = RawConfig(path).to_dict()
    d.update(over)
    return d


def _synth(kc, n, seed, lo=0.6, hi=1.7):
    """Kernel-shaped synthetic events (the flat vocabulary)."""
    r = np.random.RandomState(seed)
    n_mass = int(np.max(kc["num_m_index"])) + 1
    n_q = int(np.max(kc["fl_q_index"])) + 1
    n_at = int(np.max(kc["angle_index"])) + 1
    nac = np.asarray(kc["angle_k"]).shape[1]
    return {
        "mass": lo + (hi - lo) * r.rand(n, n_mass),
        "q": np.abs(r.randn(n, n_q)) + 0.3,
        "angle": r.uniform(-np.pi, np.pi, size=(n, n_at, nac)),
        "weight": 0.5 + r.rand(n),
        "bkg": np.abs(r.randn(n)) * 0.1 + 0.05,
    }


def _loaded(cfg=None, n_phsp=240, n_data=120):
    f = Fitter(cfg if cfg is not None else _cfg(), backend="numpy_flat")
    f.apply_constrains()
    kc = f.kernel_config
    f.set_phsp(_synth(kc, n_phsp, 1))
    f.set_data(_synth(kc, n_data, 2))
    return f


def _xi_rows(points, y, x, bc="not-a-knot"):
    """TFPWA's spline equations transcribed once more, here in the test.

    ``S_i'(x_i) = S_{i-1}'(x_i)`` plus the two boundary conditions
    (``not-a-knot``: third derivative continuous across the second knot;
    ``natural``: S'' = 0 at both ends), solved for the interval second
    derivatives, then the interval cubic in GLOBAL coordinates.  This is a
    TRANSCRIPTION of ``spline_basis_matrix``'s own equations, not an
    independent derivation — it pins that the port solves what it says it
    solves.  The independent check is the ``scipy.CubicSpline`` comparison.
    """
    p = np.asarray(points, float)
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    n = p.size
    h = np.diff(p)
    H = np.zeros((n, n))
    Y = np.zeros((n, n))
    if bc == "not-a-knot":
        H[0, 0], H[0, 1], H[0, 2] = -h[1], h[0] + h[1], -h[0]
    elif bc == "natural":
        H[0, 0] = 1.0
    else:
        raise ValueError(bc)
    for i in range(1, n - 1):
        H[i, i - 1] = h[i - 1]
        H[i, i] = 2.0 * (h[i - 1] + h[i])
        H[i, i + 1] = h[i]
        Y[i, i - 1] = 6.0 / h[i - 1]
        Y[i, i] = -6.0 * (1.0 / h[i] + 1.0 / h[i - 1])
        Y[i, i + 1] = 6.0 / h[i]
    if bc == "not-a-knot":
        H[-1, -3], H[-1, -2], H[-1, -1] = -h[-1], h[-1] + h[-2], -h[-2]
    else:
        H[-1, -1] = 1.0
    hy = np.linalg.solve(H, Y)
    ci = hy[:-1] / 2.0
    di = (hy[1:] - hy[:-1]) / (6.0 * h[:, None])
    eye = np.eye(n)
    bi = (eye[1:] - eye[:-1]) / h[:, None] - ci * h[:, None] \
        - di * h[:, None] ** 2
    ai = eye[:-1]
    x1 = p[:-1, None]
    A = ai - bi * x1 + ci * x1 ** 2 - di * x1 ** 3
    B = bi - 2.0 * ci * x1 + 3.0 * di * x1 ** 2
    C = ci - 3.0 * di * x1
    Hm = np.stack([A, B, C, di], axis=1)          # (n-1, 4, n)
    idx = np.clip(np.searchsorted(p, x, side="right") - 1, 0, n - 2)
    vals = np.zeros(x.size)
    for j in range(4):
        vals += (x ** j) * (Hm[idx, j] @ y)
    return vals


# ── A: the row family ───────────────────────────────────────────────

def test_registered_as_a_basis_model_with_no_running_width():
    assert "spline_c_idx" in ALL_MODELS
    m = _model()
    assert isinstance(m, BasisModel) and not isinstance(m, BaseModel)
    for attr in ("gamma", "get_gamma_name", "get_mass_name",
                 "make_mass_width_transform"):
        assert not hasattr(m, attr), attr


@pytest.mark.parametrize("bc", ["not-a-knot", "natural"])
def test_rows_match_scipy_for_the_configured_bc(bc):
    """The port's rows == ``scipy.CubicSpline(bc_type=…)`` per unit vector."""
    m = _model(bc_type=bc, with_bound=True)
    n = len(POINTS)
    grid = np.linspace(POINTS[0], POINTS[-1], 401)
    rows = m.num_basis(grid)
    assert len(rows) == n
    for i in range(n):
        ref = CubicSpline(POINTS, np.eye(n)[i], bc_type=bc)(grid)
        assert np.max(np.abs(np.asarray(rows[i]).real - ref)) < 1e-12, (bc, i)
        # the rows are REAL (the complex part is the weight, not the row)
        assert np.max(np.abs(np.asarray(rows[i]).imag)) == 0.0


@pytest.mark.parametrize("bc", ["not-a-knot", "natural"])
def test_rows_match_an_independent_solve_of_tfpwa_equations(bc):
    m = _model(bc_type=bc, with_bound=True)
    n = len(POINTS)
    grid = np.linspace(POINTS[0], POINTS[-1], 301)
    for i in range(n):
        ref = _xi_rows(POINTS, np.eye(n)[i], grid, bc=bc)
        assert np.max(np.abs(np.asarray(m.num_basis(grid)[i]).real - ref)) \
            < 1e-10, (bc, i)


def test_not_a_knot_is_not_cosmetic():
    """The bc really changes the rows (`natural` is not a relabelling)."""
    a = np.asarray([r.real for r in _model(with_bound=True).num_basis(
        np.linspace(POINTS[0], POINTS[-1], 101))])
    b = np.asarray([r.real for r in _model(bc_type="natural",
                                          with_bound=True).num_basis(
        np.linspace(POINTS[0], POINTS[-1], 101))])
    assert np.max(np.abs(a - b)) > 1e-6


def test_partition_of_unity_and_the_cardinal_property():
    m = _model(with_bound=True)
    n = len(POINTS)
    grid = np.linspace(POINTS[0], POINTS[-1], 401)
    rows = np.asarray([np.asarray(r).real for r in m.num_basis(grid)])
    assert np.max(np.abs(rows.sum(axis=0) - 1.0)) < 1e-12
    at = np.asarray([np.asarray(r).real for r in m.num_basis(POINTS)])
    assert np.max(np.abs(at - np.eye(n))) < 1e-12


def test_all_coefficients_one_is_a_constant_amplitude():
    """``with_bound=True`` + all ``c_i = 1`` ⇒ ``A ≡ 1`` (partition of unity).

    The production-shape case too: 18 knots across the two-body window.
    """
    pts = list(np.linspace(0.30, 2.50, 18))
    m = build_particle("k2", model="spline_c_idx", points=pts, with_bound=True)
    assert m.n_free == 18
    params = {f"k2_point_{i}": 1.0 + 0.0j for i in range(18)}
    grid = np.linspace(pts[0], pts[-1], 501)
    got = np.asarray(m.amp(grid, params))
    # MEASURED 4.6e-12 over 18 terms on a 501-point grid (5.8e-12 on 4001 — a
    # solve-conditioning artefact; the port agrees with TFPWA to 9.1e-13 per
    # row), 6.7e-16 at the 6-knot fixture below
    assert np.max(np.abs(got.real - 1.0)) < 1e-10
    assert np.max(np.abs(got.imag)) == 0.0
    # ... and the cardinality at the knots on that wider grid
    at = np.asarray([np.asarray(r).real for r in m.num_basis(pts)])
    assert np.max(np.abs(at - np.eye(18))) < 1e-10


# ── B: with_bound ───────────────────────────────────────────────────

def test_with_bound_controls_the_end_knots():
    n = len(POINTS)
    c = np.array([0.7 - 0.2j, 1.1 + 0.4j, -0.3 + 0.9j,
                  -1.2 - 0.1j, 0.5 + 0.5j, 0.2 - 0.8j])

    free = _model(with_bound=False)
    bound = _model(with_bound=True)
    assert (free.n_free, bound.n_free) == (n - 2, n)
    # the end coefficients are ABSENT from the free model's slots ...
    assert free.get_num_name() == [f"k2_point_{i}" for i in range(1, n - 1)]
    assert bound.get_num_name() == [f"k2_point_{i}" for i in range(n)]
    # ... so the end knots evaluate to 0 (they are pinned to zero) ...
    p_free = {f"k2_point_{i}": c[i] for i in range(1, n - 1)}
    got = np.asarray(free.amp([POINTS[0], POINTS[-1]], p_free))
    # MEASURED 3.5e-14 (the dropped columns leave a numerical null there)
    assert np.max(np.abs(got)) < 1e-12
    # ... while ``with_bound`` keeps them as the fitted values
    p_all = {f"k2_point_{i}": c[i] for i in range(n)}
    got = np.asarray(bound.amp([POINTS[0], POINTS[-1]], p_all))
    assert abs(complex(got[0]) - c[0]) < 1e-12
    assert abs(complex(got[-1]) - c[-1]) < 1e-12


# ── C: extrapolation ────────────────────────────────────────────────

def test_outside_spline_is_the_end_interval_cubic_and_zero_is_zero():
    """``spline`` (TFPWA ``spline_c_idx``) continues cubically; ``zero`` is 0.

    MEASURED on this 6-knot set with a random coefficient vector: the cubic
    continuation equals scipy's ``extrapolate=True`` value to 1.2e-13 (a
    genuine cubic tail, not a constant), and ``|A|`` at h/2, h, 4h outside the
    support is 4.8 / 10.5 / 142 below and 5.9 / 15.1 / 236 above — it GROWS,
    which is why ``outside="zero"`` exists.  Inside the support the two rules
    agree exactly (0.0).
    """
    n = len(POINTS)
    rng = np.random.RandomState(3)
    c = rng.randn(n) + 1j * rng.randn(n)
    m_s = _model(with_bound=True, outside="spline")
    m_z = _model(with_bound=True, outside="zero")
    p = {f"k2_point_{i}": c[i] for i in range(n)}
    h = POINTS[1] - POINTS[0]
    xs = np.concatenate([POINTS[0] - np.array([0.5, 1.0, 4.0]) * h,
                         POINTS[-1] + np.array([0.5, 1.0, 4.0]) * h])
    got_s = np.asarray(m_s.amp(xs, p))
    got_z = np.asarray(m_z.amp(xs, p))
    ref = CubicSpline(POINTS, c, bc_type="not-a-knot")(xs)   # cubic tail
    assert np.max(np.abs(got_s - ref)) < 1e-10
    assert np.max(np.abs(got_z)) == 0.0
    # inside, both outside rules are the same amplitude
    grid = np.linspace(POINTS[0], POINTS[-1] - 1e-9, 201)
    assert np.max(np.abs(np.asarray(m_s.amp(grid, p))
                         - np.asarray(m_z.amp(grid, p)))) < 1e-12


# ── D: complex coefficients, slot spelling ──────────────────────────

def test_slots_are_tfpwa_point_names_one_complex_coefficient_each():
    m = _model(with_bound=True)
    n = len(POINTS)
    heads = m.get_num_name()
    assert heads == [f"k2_point_{i}" for i in range(n)]
    assert m.num_slot_names() == [f"k2_point_{i}{t}"
                                  for i in range(n) for t in "ri"]
    # no doubled row axis: N rows, 2 real slots per row, in knot order
    assert len(m.num_basis(np.zeros(2))) == n
    assert len(m.num_slot_names()) == 2 * n
    # the CKProduct convention: a head absent from the dict is built from its
    # (r, i) slots, so a caller may pass either form
    z = 0.4 * np.exp(0.9j)
    a = m.amp(np.array([POINTS[1]]), {"k2_point_1": z})
    b = m.amp(np.array([POINTS[1]]), {"k2_point_1r": 0.4,
                                      "k2_point_1i": 0.9})
    assert abs(complex(a[0]) - z) < 1e-12
    assert abs(complex(b[0]) - z) < 1e-12


def test_amp_at_points_returns_the_complex_coefficients():
    """``A(points[i]) == c_i`` — the cardinal property, per component."""
    m = _model(with_bound=True)
    n = len(POINTS)
    c = np.array([1.0 + 0.0j, -0.4 + 0.6j, 0.25 - 1.1j,
                  2.0 + 0.3j, -0.05 - 0.7j, 0.9 + 0.9j])
    params = {f"k2_point_{i}": c[i] for i in range(n)}
    got = np.asarray(m.amp(POINTS, params))
    assert np.max(np.abs(got.real - c.real)) < 1e-10
    assert np.max(np.abs(got.imag - c.imag)) < 1e-10
    # real and imaginary parts are INDEPENDENT directions
    p2 = dict(params)
    p2["k2_point_2"] = c[2] + 1j * 0.5
    got2 = np.asarray(m.amp(POINTS, p2))
    assert abs(got2[2] - (c[2] + 1j * 0.5)) < 1e-10
    assert np.allclose(got2.real, got.real)
    assert np.allclose(np.delete(got2.imag, 2), np.delete(got.imag, 2))


def test_denominator_is_the_single_constant_row():
    m = _model()
    grid = np.linspace(0.4, 2.0, 13)
    den = m.den_basis(grid)
    assert len(den) == 1
    assert np.allclose(den[0], 1.0) and np.allclose(den[0].imag, 0.0)
    assert m.get_den_name() == ["one"]
    # no shape parameter is fitted: the coefficients ARE the parameters
    assert m.get_defaults() == {}
    assert m.make_basis_transform() is None


def test_fix_idx_names_the_tfpwa_middle_knot_and_is_a_constraint_job():
    m = _model(with_bound=True)
    n = len(POINTS)
    assert m.fix_idx == n // 2 - 1                 # TFPWA's default
    assert m.fixed_values() == {f"k2_point_{n // 2 - 1}r": 1.0,
                                f"k2_point_{n // 2 - 1}i": 0.0}
    # ``None`` resolves to the same default, and TFPWA's -1 (or any negative
    # index) also means "the middle"; TFPWA's explicit ``null`` ("no pin") has
    # no counterpart, so here ``None`` keeps the default.  The model cannot
    # PIN anything (no transform) — the constraint layer does.
    assert _model(fix_idx=None).fix_idx == n // 2 - 1
    assert _model(fix_idx=-1).fix_idx == n // 2 - 1
    assert _model(fix_idx=-7).fix_idx == n // 2 - 1
    # with_bound=False indexes the FITTED list, so the knot is one higher
    assert _model(with_bound=False).fixed_values() == {"k2_point_3r": 1.0,
                                                       "k2_point_3i": 0.0}


# ── E: refusals ─────────────────────────────────────────────────────

@pytest.mark.parametrize("kw, match", [
    ({}, "the 'points' key is required"),
    ({"points": [0.3, 0.6]}, "more than 2"),
    ({"points": [0.3, 0.6, 0.6, 1.0]}, "strictly ascending"),
    ({"points": [0.3, 0.9, 0.6, 1.2]}, "strictly ascending"),
    ({"points": [0.3, 0.6, np.inf, 1.2]}, "must be finite"),
    ({"points": [0.3, 0.6, np.nan, 1.2]}, "must be finite"),
    ({"points": "not a list"}, "must be a list of numbers"),
    ({"points": POINTS, "bc_type": "clamped"}, "clamped"),
    ({"points": POINTS, "bc_type": "cubic"}, "bc_type"),
    ({"points": POINTS, "outside": "constant"}, "outside"),
    ({"points": POINTS, "polar": False}, "polar: false"),
    ({"points": POINTS, "fix_idx": 99}, "fix_idx"),
    ({"points": POINTS, "with_bound": "true"}, "with_bound must be a bool"),
    ({"min_m": 0.3, "max_m": 0.3, "interp_N": 4}, "min_m < max_m"),
])
def test_bad_configuration_is_refused_with_its_own_message(kw, match):
    with pytest.raises(ValueError, match=match):
        build_particle("k2", model="spline_c_idx", **kw)


def test_points_or_a_uniform_grid_and_the_n_three_gate():
    a = _model(points=[0.3, 0.6, 0.9, 1.2])
    b = build_particle("k2", model="spline_c_idx",
                       min_m=0.3, max_m=1.2, interp_N=4)
    assert np.allclose(a.points, b.points)
    # N > 2 (TFPWA's assert): three points are accepted ...
    m3 = build_particle("k2", model="spline_c_idx", points=[0.3, 0.9, 1.5],
                        with_bound=True, bc_type="natural")
    assert m3.n_free == 3
    assert np.max(np.abs(np.asarray([r.real for r in m3.num_basis(m3.points)])
                         - np.eye(3))) < 1e-12
    # ... but not-a-knot IS singular for three knots (TFPWA: LinAlgError); the
    # model names the fix instead
    with pytest.raises(ValueError, match="singular for N=3"):
        build_particle("k2", model="spline_c_idx", points=[0.3, 0.9, 1.5])


def test_plain_pwa_refuses_it_naming_the_basis_path():
    with pytest.raises(ValueError, match="amp_model: pwa_gen"):
        Fitter(_cfg(amp_model="pwa"))
    f = Fitter(_cfg())
    assert "basis_table" in f.kernel_config


# ── F: pwa_gen integration ──────────────────────────────────────────

def _flat(f):
    m = f.model
    kc = f.kernel_config
    return m, kc, m._pg_particle_models()["k2"]


def test_the_numerator_expands_into_n_free_slots_per_wave():
    f = Fitter(_cfg(), backend="numpy_flat")
    m, kc, mdl = _flat(f)
    n_free = mdl.n_free
    assert n_free == len(POINTS) - 2          # the fixture uses with_bound: False
    geom = m.full_decay.get_partial_waves()
    ck_map = m.get_ck_map()
    assert len(geom) == 1                     # the fixture is one partial wave
    assert len(ck_map) == len(geom) * n_free
    exp = m.wave_expansion()
    assert [b for b, _o, _c in exp] == list(range(len(geom)))
    assert [c for _b, _o, c in exp] == [n_free] * len(geom)
    off = 0
    for _b, o, c in exp:
        assert o == off
        off += c
    assert off == len(ck_map)
    # the head is the LAST tuple entry: one per fitted knot, in knot order
    heads = mdl.get_num_name()
    for i, comb in enumerate(ck_map):
        assert comb[-1] == heads[i]

    # the tabulated rows are the model's rows, one per expanded wave
    nt = np.asarray(kc["num_table"])
    assert nt.shape[0] == len(ck_map)
    grid = kc["m_min"] + kc["m_delta"] * np.arange(nt.shape[-1])
    rows = mdl.num_basis(grid)
    for k in range(nt.shape[0]):
        assert np.max(np.abs(nt[k, 0] - rows[k])) < 1e-12, k
    # the denominator is ONE row (the constant), not the spline
    assert kc["den_names"] == ["one"]
    assert np.asarray(kc["basis_table"]).shape[0] == 1
    # every expanded wave reads the SAME (daughter) mass column
    assert len(np.unique(np.asarray(kc["num_m_index"]))) == 1
    # the knots sit inside the kernel's table window
    m_lo, m_hi = kc["m_min"], kc["m_min"] + kc["m_delta"] * (nt.shape[-1] - 1)
    assert mdl.points[0] > m_lo and mdl.points[-1] < m_hi


def test_with_bound_true_is_the_full_n_free_on_the_fixture():
    d = _cfg()
    d["particle"]["k2"]["with_bound"] = True
    f = Fitter(d, backend="numpy_flat")
    m, kc, mdl = _flat(f)
    assert mdl.n_free == len(POINTS)
    assert len(m.get_ck_map()) == mdl.n_free
    assert np.asarray(kc["num_table"]).shape[0] == mdl.n_free


def test_the_fit_surface_names_the_complex_coefficients():
    f = _loaded()
    kc = f.kernel_config
    mdl = f.model._pg_particle_models()["k2"]
    tr = f.model.params_transform_cls(f.model, kc)
    names = set(tr.param_names())
    for i in range(1, len(POINTS) - 1):
        assert {f"k2_point_{i}r", f"k2_point_{i}i"} <= names
    # ``ck_redundancy`` pins ONE reference coefficient per wave context (the
    # FIRST fitted knot), leaving n_free - 1 complex coefficients
    free = set(f.free_param_names())
    first = mdl.get_num_name()[0]
    assert f"{first}r" not in free and f"{first}i" not in free
    assert len(free) == 2 * (mdl.n_free - 1)
    assert all(nm.startswith("k2_point_") for nm in free)


def test_P_and_Q_reproduce_by_hand_from_the_rows():
    """``P`` from the MODEL's rows (not the tabulated table).

    ``fl``/``fa`` come from the config (they are the kernel's own inputs); the
    numerator is the model evaluated at that event's daughter mass, so the
    residual is the table's Catmull-Rom representation of a cubic — the only
    approximation left in this wiring.
    """
    from tabpwa.backends.numpy_flat_backend import NumpyFlatBackend

    f = _loaded()
    m, kc, mdl = _flat(f)
    x = f.initial_values(seed=1)
    params, _res = f.build_params(x)
    ck = np.asarray(params["ck"])
    n_wave = np.asarray(kc["fl_table"]).shape[0]
    n_proj = np.asarray(kc["matrix_angle"]).shape[1] // n_wave

    data = f._data_np
    Q_k, _g, P_k = f.backend.compute(params, f._data_holder, norm=None)

    ne = data["mass"].shape[0]
    fl_tab = np.asarray(kc["fl_table"])
    fl_q = np.asarray(kc["fl_q_index"])
    num_m = np.asarray(kc["num_m_index"])
    ak, ab = np.asarray(kc["angle_k"]), np.asarray(kc["angle_b"])
    ai, am = np.asarray(kc["angle_index"]), np.asarray(kc["matrix_angle"])
    ka = np.zeros((ne, ak.shape[0]), complex)
    for a in range(ak.shape[0]):
        ka[:, a] = np.prod(np.cos(data["angle"][:, ai[a], :] * ak[a] + ab[a]),
                           axis=-1)
    fa = (ka @ am).reshape(ne, n_proj, n_wave).transpose(0, 2, 1)
    P_hand = np.zeros(ne)
    for e in range(ne):
        A = np.zeros(n_proj, complex)
        for k in range(n_wave):
            # the barrier product over the chain's decay vertices
            fl = 1.0
            for d in range(fl_q.shape[1]):
                q = data["q"][e, fl_q[k, d]]
                fl = fl * NumpyFlatBackend._cr(fl_tab[k, d], q,
                                               kc["q_min"], kc["q_delta"])
            # the MODEL's row at the event mass, NOT num_table
            row = np.asarray(mdl.num_basis(
                np.array([data["mass"][e, num_m[k, 0]]])))[k, 0]
            for p in range(n_proj):
                A[p] += ck[k] * fa[e, k, p] * fl * row
        P_hand[e] = float(np.sum(np.abs(A) ** 2))
    scale = max(np.max(P_hand), 1e-300)
    # MEASURED 2.5e-9: the table's Catmull-Rom read of the cubic (the wiring
    # itself, evaluated on ``num_table`` instead of the model, is 2.9e-16)
    assert np.max(np.abs(P_k - P_hand)) / scale < 1e-8
    # ... and Q is the same weighted sum of that P (``norm=None``: the
    # unnormalised sum, which is what the backend returns for a norm-less call)
    w = data["weight"]
    Q_hand = float(np.sum(w * P_hand))
    assert abs(Q_hand - Q_k) / max(abs(Q_k), 1e-300) < 1e-8


def test_nll_gradient_matches_fd_on_numpy_flat():
    """Every free coefficient slot (real AND imaginary) matches FD.

    The knots are NOT fitted — the ``c_i`` are — so this is the whole
    parameter surface.
    """
    f = _loaded()
    x = f.initial_values(seed=1)
    nll, grad = f.get_nll(x)
    assert np.isfinite(nll) and np.all(np.isfinite(grad))
    names = f.free_param_names()
    assert names and all(nm.startswith("k2_point_") for nm in names)
    h = 1e-6
    for i, nm in enumerate(names):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        fd = (f.get_nll(xp)[0] - f.get_nll(xm)[0]) / (2 * h)
        # MEASURED max |grad - fd| = 2.0e-8 over the 6 slots
        assert abs(grad[i] - fd) <= 1e-6 * abs(grad[i]) + 1e-9, (
            nm, grad[i], fd)
    assert np.max(np.abs(grad)) > 0.0


CUDA_LEAVES = ["cuda_flat_pwa_gen_s0e2", "cuda_flat_pwa_gen_ampcache"]


@pytest.mark.parametrize("name", CUDA_LEAVES)
def test_the_cuda_leaves_agree_with_numpy_flat_on_the_fixture(name):
    """The spline config reaches the GPU leaves at the fp32 floor.

    Nothing else pins that a NUMERATOR-expanded config (the denominator is the
    single fixed ``one``, the free coefficients sit in ``ck``) is accepted by
    the CUDA leaves and numerically the same there.  MEASURED (64 events,
    seed 2, norm 1.3): P 5.0e-8, Q 3.3e-10, dck 3.8e-9 for BOTH leaves.
    ``ampcache`` reports ``grads["coeff"] == 0`` by contract (every basis
    weight is pinned), so the coefficient block is compared only for an
    unpinned leaf.
    """
    from tabpwa.backends import backend_tags, create_backend
    from tabpwa.backends.auto_backend import cuda_state
    if not cuda_state()["has_cuda"]:
        pytest.skip("CUDA unusable here")
    from _flat_grid import _in_grid

    f = Fitter(_cfg(), backend="numpy_flat")
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    data = _in_grid(kc, 64, 2)
    ref = create_backend("numpy_flat", kc, model="pwa_gen")
    got = create_backend(name, kc, model="pwa_gen")
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        Q, g, P = got.compute(params, got.load_data(data), norm=1.3)
    finally:
        ref.free()
        got.free()
    assert np.max(np.abs(np.asarray(P) - np.asarray(Pr))) \
        / max(np.max(np.abs(Pr)), 1e-300) < 1e-6
    assert abs(Q - Qr) / max(abs(Qr), 1e-300) < 1e-8
    assert np.max(np.abs(np.asarray(g["ck"]) - np.asarray(gr["ck"]))) \
        / max(np.max(np.abs(np.asarray(gr["ck"]))), 1e-300) < 1e-7
    if "coeff_pinned" in backend_tags(name, "pwa_gen"):
        assert np.all(np.asarray(g["coeff"]) == 0.0)
    else:
        assert np.max(np.abs(np.asarray(g["coeff"])
                             - np.asarray(gr["coeff"]))) < 1e-6