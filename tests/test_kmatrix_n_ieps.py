"""``kmatrix_n_ieps`` / ``kmatrix_n_gs_ieps`` — the ``m² − iε`` prescription.

ε sits inside EVERY pole factor, ``Π_i(m_i² − s + iε)``, linearised in ε
(``ε²`` and higher dropped, exact for simple poles):

    Π_i(m_i² − s + iε) = Π(s) + iε·D(s) + O(ε²),   D ≡ Σ_i Π_{j≠i}(m_j² − s)

so the extra rows are ``i·s^k`` (k = 0 … N−1) with the DERIVED weights
``d_k = −(k+1)·w_{k+1}`` (w = the real-row weights of Π).

Gates: ``ieps = 0`` is bit-identical to ``kmatrix_n``; the rows and their
weights are the derivative of Π; a below-threshold pole and a zero-width pole
become finite; a DEGENERATE pair remains a double root (D = 0 there, so the
linear term cannot lift it); the weights are transform outputs, never fitted;
and the VJP still chains through them.
"""
import numpy as np
import pytest

from tabpwa.particle_model import ALL_MODELS, build_particle
from tabpwa.particle_model.kmatrix_n_ieps_model import ieps_name


class _Out:
    mass = 0.13957039


class _Dec:
    outs = [_Out(), _Out()]


class _Parent:
    _decays = [_Dec()]


TWO = ([0.775, 1.45], [0.149, 0.30])


def _build(model, masses, widths, ieps=0.0):
    mdl = build_particle("k", model=model, mass_list=masses,
                         width_list=widths, L=1, ieps=ieps)
    mdl.register_parent(_Parent())
    return mdl


def test_models_are_registered():
    assert "kmatrix_n_ieps" in ALL_MODELS
    assert "kmatrix_n_gs_ieps" in ALL_MODELS


@pytest.mark.parametrize("base", ["kmatrix_n", "kmatrix_n_gs"])
def test_ieps_zero_is_the_unregulated_model_bit_for_bit(base):
    masses, widths = TWO
    grid = np.linspace(0.30, 3.0, 301)
    plain = build_particle("k", model=base, mass_list=masses,
                           width_list=widths, L=1)
    plain.register_parent(_Parent())
    reg = _build(base + "_ieps", masses, widths, ieps=0.0)
    assert reg.get_den_name() == plain.get_den_name()
    assert np.array_equal(reg.amp(grid, reg.get_defaults()),
                          plain.amp(grid, plain.get_defaults()))


def test_extra_rows_are_ieps_times_the_derivative_of_pi():
    """rows `i·s^k` (k < N) with weights `-(k+1)·w_{k+1}` = ε·(−Π')."""
    n, eps = "k", 1e-6
    masses, widths = TWO
    mdl = _build("kmatrix_n_ieps", masses, widths, ieps=eps)
    N = mdl.n_pole
    names = mdl.get_den_name()
    assert names[2 * N + 1:] == [ieps_name(n, k) for k in range(N)]
    assert len(names) == 2 * N + 1 + N

    grid = np.array([0.40, 0.90, 1.60])
    s = grid ** 2
    rows = mdl.den_basis(grid)
    for k in range(N):
        assert np.array_equal(rows[2 * N + 1 + k],
                              (1j * s ** k).astype(complex))

    out = mdl.make_basis_transform().apply_forward(mdl.get_defaults())
    w = [out[nm] for nm in names[:N + 1]]              # real rows of Π(s)
    for k in range(N):
        assert out[ieps_name(n, k)] == pytest.approx(-eps * (k + 1) * w[k + 1])


def test_rows_share_the_P_K_format():
    """num == den[:N]; ρ rows = −iρ·those; ε rows = i·those — ONE row set."""
    n = "k"
    masses, widths = TWO
    mdl = _build("kmatrix_n_ieps", masses, widths, ieps=1e-6)
    N = mdl.n_pole
    grid = np.array([0.50, 0.95, 1.30])
    den, num = mdl.den_basis(grid), mdl.num_basis(grid)
    rho = np.asarray(mdl._rho(grid), dtype=complex)
    for k in range(N):
        assert np.array_equal(num[k], den[k])
        assert np.allclose(den[N + 1 + k], -1j * rho * num[k], rtol=0, atol=0)
        assert np.array_equal(den[2 * N + 1 + k], 1j * num[k])


@pytest.mark.parametrize("n_pole", [1, 2])
def test_amplitude_matches_the_direct_form_with_ieps(n_pole):
    """`A == num(s−iε) / (Π(s−iε) − iρU)` — BOTH polynomials shifted."""
    from tabpwa.particle_model.kmatrix_n_model import _prod_minus_s
    n, eps = "k", 1e-6
    m = [0.70 + 0.40 * i for i in range(n_pole)]
    w = [0.15 + 0.05 * i for i in range(n_pole)]
    mdl = _build("kmatrix_n_ieps", m, w, ieps=eps)

    grid = np.array([0.50, 0.95, 1.30])
    s = grid ** 2
    roots = [x * x for x in m]

    def prod(skip=()):
        """Π_{j ∉ skip} (m_j² − s), evaluated at s."""
        skip = (skip,) if isinstance(skip, int) else tuple(skip)
        out = np.ones_like(s, dtype=complex)
        for j, r in enumerate(roots):
            if j not in skip:
                out = out * (r - s)
        return out

    # Π + iε·(−Π′), from the coefficient rule w_k − iε(k+1)w_{k+1}
    w_par = np.asarray(_prod_minus_s(roots), dtype=float)
    Pi = sum(w_par[k] * s ** k for k in range(n_pole + 1))
    Pi_eps = sum((w_par[k] - 1j * eps * (k + 1)
                  * (w_par[k + 1] if k + 1 <= n_pole else 0.0)) * s ** k
                 for k in range(n_pole + 1))

    # Σ_i Π_{j≠i} + iε·Σ_i Σ_{l≠i} Π_{j≠i,l}   (β0 = 1, β_{i≥1} = 1)
    num = sum(prod(i) for i in range(n_pole))
    num_eps = sum(prod(i) + 1j * eps * sum(prod((i, l))
                                          for l in range(n_pole) if l != i)
                  for i in range(n_pole))

    g2 = np.asarray(mdl._coupling_sq(m, w), dtype=float)
    U = sum(g2[i] * prod(i) for i in range(n_pole))
    rho = np.asarray(mdl._rho(grid), dtype=complex)

    expected = num_eps / (Pi_eps - 1j * rho * U)
    assert np.allclose(mdl.amp(grid, mdl.get_defaults()), expected,
                       rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("case,masses,widths", [
    ("below threshold", [0.20], [0.05]),
    ("zero width", [0.775], [0.0]),
])
def test_simple_real_axis_poles_become_finite(case, masses, widths):
    plain = _build("kmatrix_n", masses, widths)
    reg = _build("kmatrix_n_ieps", masses, widths, ieps=1e-6)
    at = np.array([masses[0]])
    with np.errstate(all="ignore"):
        assert not np.isfinite(plain.amp(at, plain.get_defaults())[0])
    A = reg.amp(at, reg.get_defaults())[0]
    assert np.isfinite(A)
    assert abs(A) > 1.0                 # still a spike: |num| / (ε·|D|)
    # ... bounded, and finite at the neighbouring point as well
    assert np.isfinite(reg.amp(at + 1e-3, reg.get_defaults())[0])


def test_degenerate_poles_are_not_regulated():
    """A double root has D = Π' = 0: the linear-ε term cannot lift it."""
    reg = _build("kmatrix_n_ieps", [0.775, 0.775], [0.149, 0.15], ieps=1e-6)
    with np.errstate(all="ignore"):
        A = reg.amp(np.array([0.775]), reg.get_defaults())[0]
    assert not np.isfinite(A)


def test_gs_variant_regulates_a_zero_width_pole():
    """`gs_unit` is defined above threshold only — the zero-width case."""
    plain = _build("kmatrix_n_gs", [0.775], [0.0])
    reg = _build("kmatrix_n_gs_ieps", [0.775], [0.0], ieps=1e-6)
    with np.errstate(all="ignore"):
        assert not np.isfinite(plain.amp(np.array([0.775]),
                                         plain.get_defaults())[0])
    A = reg.amp(np.array([0.775]), reg.get_defaults())[0]
    assert np.isfinite(A) and abs(A) > 1.0


def test_gs_below_threshold_is_outside_the_shape():
    """Not a denominator zero: `gs_unit` itself is nan there — no regulator."""
    reg = _build("kmatrix_n_gs_ieps", [0.20], [0.05], ieps=1e-6)
    with np.errstate(all="ignore"):
        assert not np.isfinite(np.asarray(reg._rho(np.array([0.20])),
                                          dtype=complex)[0])
        assert not np.isfinite(reg.amp(np.array([0.20]), reg.get_defaults())[0])


def test_weights_are_fixed_not_parameters():
    from tabpwa import Fitter
    from tabpwa.config_loader import RawConfig

    n = "k2"
    d = RawConfig("tests/config_pwa_gen.yml").to_dict()
    d["particle"][n].update({"J": 1, "P": -1, "model": "kmatrix_n_ieps",
                             "L": 1, "mass_list": [0.8, 1.35],
                             "width_list": [0.15, 0.2], "ieps": 1e-6})
    d["backend"] = {"name": "numpy_flat", "resolution_size": 1}
    f = Fitter(d)
    f.apply_constrains()

    names = f.kernel_config["den_names"]
    assert names[-2:] == [ieps_name(n, 0), ieps_name(n, 1)]
    assert len(names) == 2 * 2 + 1 + 2
    assert np.asarray(f.kernel_config["basis_table"]).shape[0] == len(names)
    flat = list(f.cm.var_registry.flat_names)
    for k in range(2):
        assert ieps_name(n, k) not in flat
        assert ieps_name(n, k) not in f.cm.defaults


@pytest.mark.parametrize("ieps", [0.0, 1e-6, 1e-2])
def test_vjp_is_analytic_and_exact_in_both_modes(ieps):
    """No forward calls either way; exact against Richardson.

    ``ieps = 1e-2`` is the decisive case: an ``O(ε)`` defect in the cotangent
    map for the hidden numerator shift would show up at 1e-2 relative here,
    not hidden below a 1e-6 regulator.
    """
    n = "k"
    mdl = _build("kmatrix_n_ieps", [0.60, 1.20], [0.149, 0.30], ieps=ieps)
    tr = mdl.make_basis_transform()
    params = {f"{n}_mass0": 0.60, f"{n}_mass1": 1.20,
              f"{n}_width0": 0.149, f"{n}_width1": 0.30,
              f"{n}_beta1r": 0.7, f"{n}_beta1i": -0.4}
    tr.apply_forward(dict(params))
    cot = {nm: float(np.random.default_rng(3).normal())
           for nm in tr.output_names}

    calls = []
    original = tr.forward
    tr.forward = lambda d: (calls.append(1), original(d))[1]
    try:
        vjp = tr.backward(cot, d_in=dict(params))
    finally:
        tr.forward = original
    assert calls == []                   # analytic: no forward evaluations

    def scalar(x):
        o = original(x)
        return sum(cot[k] * o[k] for k in tr.output_names)

    for name, val in vjp.items():
        x = params[name]

        def fd(h):
            return (scalar({**params, name: x + h})
                    - scalar({**params, name: x - h})) / (2.0 * h)

        richardson = (4.0 * fd(5e-6) - fd(1e-5)) / 3.0
        assert val == pytest.approx(richardson, rel=1e-7,
                                    abs=1e-8), name


def test_physical_region_is_biased_only_at_ieps_over_den():
    grid = np.linspace(0.30, 3.0, 601)
    plain = _build("kmatrix_n", [0.775], [0.149])
    reg = _build("kmatrix_n_ieps", [0.775], [0.149], ieps=1e-6)
    A0 = plain.amp(grid, plain.get_defaults())
    A1 = reg.amp(grid, reg.get_defaults())
    # |den| ~ 0.1155 at the rho pole -> the shift is of order ieps/|den| ~ 9e-6
    assert np.max(np.abs(A1 - A0) / np.abs(A0)) < 2e-5


def test_negative_ieps_is_rejected():
    with pytest.raises(ValueError, match="ieps must be >= 0"):
        build_particle("k", model="kmatrix_n_ieps", mass_list=[0.775],
                       width_list=[0.149], ieps=-1e-6)
