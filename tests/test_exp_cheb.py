"""``ExpCheb`` — the exponential lineshape as a Chebyshev/Bessel basis model.

    A(m) = exp(-k(m² − m0²)) = Σ_n w_n(k, m0)·T_n(t) / 1
    t = (s_c − s)/half,   w_n = e^{k(m0²−s_c)}·(1 or 2)·I_n(k·half)

Gates: one denominator row (=1), bounded parameter-free rows, the amplitude
against the exact exponential (truncation ~1e-9 at 11 rows, ~1e-13 at 15),
UNIFORMITY in the fitted m0 (only the scalar prefactor moves), the weights
being transform outputs rather than parameters, and the pwa_gen flat config
building with the tabulated Chebyshev rows.
"""
import numpy as np
import pytest
from scipy.special import iv

from tabpwa.particle_model import ALL_MODELS, build_particle

M0, K = 0.5, 1.0
MR = [0.28, 2.0]


def _model(n_cheb=14, **kw):
    return build_particle("sigma", model="ExpCheb", mass=M0, k=K,
                          n_cheb=n_cheb, m_range=MR, **kw)


def test_registered_and_shapes():
    assert "ExpCheb" in ALL_MODELS
    mdl = _model(n_cheb=10)
    m = np.array([0.30, 0.80, 1.50])
    assert mdl.get_den_name() == ["sigma_one"]
    assert len(mdl.get_num_name()) == 11
    assert np.array_equal(mdl.den_basis(m)[0], np.ones(3, dtype=complex))
    rows = mdl.num_basis(m)
    assert len(rows) == 11
    # T_0 = 1, and every row is bounded by 1 on the window
    assert np.allclose(rows[0], 1.0)
    grid = np.linspace(*MR, 500)
    assert max(np.max(np.abs(r)) for r in mdl.num_basis(grid)) <= 1.0 + 1e-12


def test_amplitude_matches_the_exponential():
    m = np.linspace(*MR, 2000)
    exact = np.exp(-K * (m ** 2 - M0 ** 2))
    peak = np.abs(exact).max()
    for n_cheb, bound in ((10, 1e-7), (14, 1e-11)):
        A = _model(n_cheb).amp(m, _model(n_cheb).get_defaults())
        assert np.max(np.abs(A - exact)) / peak < bound, n_cheb


def test_truncation_falls_with_the_row_count():
    m = np.linspace(*MR, 1500)
    exact = np.exp(-K * (m ** 2 - M0 ** 2))
    peak = np.abs(exact).max()
    err = []
    for n_cheb in (4, 8, 12):
        mdl = _model(n_cheb)
        err.append(np.max(np.abs(mdl.amp(m, mdl.get_defaults()) - exact)) / peak)
    assert err[0] > err[1] > err[2]


def test_uniform_in_k_and_in_the_fitted_mass():
    m = np.linspace(*MR, 1500)
    mdl = _model()
    for k in (0.1, 1.0, 1.9):
        d = {**mdl.get_defaults(), "sigma_k": k}
        e = np.exp(-k * (m ** 2 - M0 ** 2))
        assert np.max(np.abs(mdl.amp(m, d) - e)) / np.abs(e).max() < 1e-8
    # a drift of m0 only moves the scalar prefactor -> same truncation error
    for m0 in (0.40, 0.60):
        d = {**mdl.get_defaults(), "sigma_mass": m0}
        e = np.exp(-K * (m ** 2 - m0 ** 2))
        assert np.max(np.abs(mdl.amp(m, d) - e)) / np.abs(e).max() < 1e-11


def test_weights_are_the_bessel_series():
    mdl = _model(n_cheb=8)
    tr = mdl.make_basis_transform()
    out = tr.apply_forward(mdl.get_defaults())
    beta = K * mdl.half
    pre = np.exp(K * (M0 ** 2 - mdl.s_c))
    w = [pre * iv(0, beta)] + [2 * pre * iv(n, beta) for n in range(1, 9)]
    for n, wn in enumerate(w):
        mag, ph = out[f"sigma_n{n}r"], out[f"sigma_n{n}i"]
        assert mag * np.exp(1j * ph) == pytest.approx(wn, rel=1e-13)
    assert out["sigma_one"] == 1.0


def test_vjp_is_finite_and_nonzero():
    mdl = _model()
    tr = mdl.make_basis_transform()
    params = dict(mdl.get_defaults())
    tr.apply_forward(params)
    cot = {n: 1.0 for n in tr.output_names}
    vjp = tr.backward(cot, d_in=params)
    assert set(vjp) == {"sigma_mass", "sigma_k"}
    assert all(np.isfinite(v) for v in vjp.values())
    assert abs(vjp["sigma_k"]) > 0 and abs(vjp["sigma_mass"]) > 0


def test_pwa_gen_flat_config_builds_with_its_rows():
    from tabpwa import Fitter
    from tabpwa.config_loader import RawConfig

    n, n_cheb = "k2", 10
    d = RawConfig("tests/config_pwa_gen.yml").to_dict()
    d["particle"][n].update({"J": 1, "P": -1, "model": "ExpCheb",
                             "mass": 0.8, "k": 1.0, "n_cheb": n_cheb,
                             "m_range": [0.3, 2.0]})
    d["backend"] = {"name": "numpy_flat", "resolution_size": 1}
    f = Fitter(d)
    f.apply_constrains()
    kc = f.kernel_config

    assert list(kc["den_names"]) == [f"{n}_one"]
    num = np.asarray(kc["num_table"])
    assert num.shape[0] == n_cheb + 1
    assert np.allclose(np.asarray(num[0], dtype=complex), 1.0)
    # the tabulated rows are the Chebyshev ones
    mdl = f.model.particle_model(n)
    grid = np.asarray(kc.get("num_grid", []), dtype=float)
    if grid.size == num.shape[1]:
        rows = mdl.num_basis(grid)
        for a, b in zip(num, rows):
            assert np.allclose(np.asarray(a, dtype=complex), b, rtol=1e-10,
                               atol=1e-12)

    flat = list(f.cm.var_registry.flat_names)
    assert f"{n}_mass" in flat and f"{n}_k" in flat
    assert f"{n}_one" not in flat                    # a fixed output, not a param
    for i in range(n_cheb + 1):
        assert f"{n}_n{i}r" not in flat
