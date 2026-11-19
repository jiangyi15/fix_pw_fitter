"""``kmatrix_n`` — one-channel K-matrix with any number of poles.

Gates: the pole content is the ``mass_list`` / ``width_list`` pair
(``n_pole = len(mass_list)``), that pair alone reproduces ``BWRS`` / ``GS_rho``
at one pole (the width -> coupling rule is the width-preserving one), the
FITTED parameters are the pole masses and WIDTHS (there is no ``g``), the
amplitude equals the hand-written ``P/(1 − iρK)`` for N = 1…4, the phase
space is an overridable METHOD (the GS variant is a subclass), and
``pwa_gen`` builds the ``2N+1``-row flat basis.
"""
import numpy as np
import pytest

from tabpwa.particle_model import build_particle
from tabpwa.particle_model.bwr_model import phase_space


class _Out:
    mass = 0.13957039


class _Dec:
    outs = [_Out(), _Out()]


class _Parent:
    _decays = [_Dec()]


def _resolve(n, m, w, b):
    """Resolved dict for a width-parameter model.

    ``mass{i}`` / ``width{i}`` for every pole, ``beta{i}`` for ``i >= 1``
    only: ``beta0`` is the fixed gauge head, not a parameter.
    """
    out = {f"{n}_mass{i}": float(mi) for i, mi in enumerate(m)}
    out.update({f"{n}_width{i}": float(wi) for i, wi in enumerate(w)})
    for i, bi in enumerate(b):
        if i == 0:
            continue
        out[f"{n}_beta{i}r"] = float(abs(bi))
        out[f"{n}_beta{i}i"] = float(np.angle(bi))
    return out


def test_pole_count_is_len_mass_list():
    """`mass_list` IS the pole count; `width_list` is one entry per pole."""
    n = "k"
    m = [0.775, 1.45, 1.90, 2.40]
    w = [0.149, 0.30, 0.20, 0.15]
    mdl = build_particle(n, model="kmatrix_n", mass_list=m, width_list=w, L=1)
    assert mdl.n_pole == 4
    assert len(mdl.get_den_name()) == 2 * 4 + 1
    assert len(mdl.get_num_name()) == 4

    d = mdl.get_defaults()
    for i, (mi, wi) in enumerate(zip(m, w)):
        assert d[f"{n}_mass{i}"] == pytest.approx(mi)
        assert d[f"{n}_width{i}"] == pytest.approx(wi)   # the widths ARE the params

    # ... and the transform turns each width into g² = m·Γ/ρ(m)
    # (no tree attached here -> the config's default daughters, π± and π0)
    rho = phase_space(np.array(m), 1, 3.0, _Out.mass, 0.1349768)
    g2 = mdl._coupling_sq(np.array(m), np.array(w))
    assert np.allclose(g2, np.array(m) * np.array(w) / rho, rtol=1e-15)

    with pytest.raises(ValueError, match="one width per pole"):
        build_particle(n, model="kmatrix_n", mass_list=m, width_list=w[:-1])
    with pytest.raises(ValueError, match="n_pole` is not a key"):
        build_particle(n, model="kmatrix_n", mass_list=m, width_list=w, n_pole=3)
    with pytest.raises(ValueError, match="mass_list must not be empty"):
        build_particle(n, model="kmatrix_n", mass_list=[], width_list=[])
    # no scalar shorthand: both lists are required
    with pytest.raises(ValueError, match=r"declare the poles with \['width_list'\]"):
        build_particle(n, model="kmatrix_n", mass_list=m)
    with pytest.raises(ValueError, match=r"declare the poles with \['mass_list'\]"):
        build_particle(n, model="kmatrix_n", width_list=w)


class _DecLS:
    """A decay that knows its partial waves (like `Decay.get_ls_list`)."""

    def __init__(self, ls):
        self.outs = [_Out(), _Out()]
        self._ls = list(ls)

    def get_ls_list(self):
        return list(self._ls)


class _ParentLS:
    def __init__(self, ls):
        self._decays = [_DecLS(ls)]


@pytest.mark.parametrize("model", ["kmatrix_n", "kmatrix_n_gs"])
def test_default_L_comes_from_the_decay(model):
    """`L` defaults to min(decay[0].get_ls_list()), the smallest l."""
    n, m0, G0 = "rho", 0.775, 0.149
    mdl = build_particle(n, model=model, mass_list=[m0], width_list=[G0])
    mdl.register_parent(_ParentLS([(1, 1), (3, 1)]))
    assert mdl.default_L() == 1
    grid = np.array([0.60, 0.775, 1.00])
    if model == "kmatrix_n":
        expect = phase_space(grid, 1, 3.0, _Out.mass, _Out.mass)
    else:
        from tabpwa.particle_model.gs_rho_model import gs_unit
        expect = gs_unit(grid, m0, 1, 3.0, _Out.mass, _Out.mass)
    assert np.allclose(mdl._rho(grid), expect, rtol=1e-15)

    # an s-wave decay picks l = 0; an explicit `L` always wins
    mdl.register_parent(_ParentLS([(0, 0), (2, 0)]))
    assert mdl.default_L() == 0
    mdl_explicit = build_particle(n, model=model, mass_list=[m0], width_list=[G0], L=2)
    mdl_explicit.register_parent(_ParentLS([(0, 0), (2, 0)]))
    assert np.allclose(mdl_explicit._rho(grid),
                       phase_space(grid, 2, 3.0, _Out.mass, _Out.mass)
                       if model == "kmatrix_n"
                       else gs_unit(grid, m0, 2, 3.0, _Out.mass, _Out.mass),
                       rtol=1e-15)

    # no decay tree: the class fallback (0 for the BW shape, 1 for GS)
    bare = build_particle(n, model=model, mass_list=[m0], width_list=[G0])
    assert bare.default_L() == (0 if model == "kmatrix_n" else 1)


def test_default_L_is_shared_by_the_other_models():
    """`BasisModel.default_L` is one rule for the whole family (`BWRS` here)."""
    from tabpwa.particle_model.bwr_model import running_width

    n, m0, G0 = "rho", 0.775, 0.149
    mdl = build_particle(n, model="BWRS", mass=m0, width=G0)
    assert mdl.default_L() == 0                        # no tree attached
    mdl.register_parent(_ParentLS([(1, 1), (3, 1)]))
    assert mdl.default_L() == 1
    grid = np.array([0.60, 0.775, 1.00])
    assert np.allclose(mdl.gamma(grid)[0],
                       running_width(grid, m0, 1, 3.0, _Out.mass, _Out.mass))

    # an explicit `L` still wins over the tree
    ex = build_particle(n, model="BWRS", mass=m0, width=G0, L=2)
    ex.register_parent(_ParentLS([(0, 0), (2, 0)]))
    assert ex.default_L() == 0
    assert np.allclose(ex.gamma(grid)[0],
                       running_width(grid, m0, 2, 3.0, _Out.mass, _Out.mass))


def test_beta0_is_the_gauge_head_and_beta1_starts_random():
    """`beta0` is fixed to 1; `beta{i>=1}` are couplings with no default."""
    from tabpwa import Fitter
    from tabpwa.config_loader import RawConfig

    n = "k2"
    mdl = build_particle(n, model="kmatrix_n", mass_list=[0.8, 1.35],
                         width_list=[0.3, 0.25], beta0=1.0)   # key is inert
    assert f"{n}_beta0r" not in mdl.get_defaults()
    assert f"{n}_beta1r" not in mdl.get_defaults()
    assert f"{n}_mass0" in mdl.get_defaults()
    assert f"{n}_width0" in mdl.get_defaults()

    d = RawConfig("tests/config_pwa_gen.yml").to_dict()
    d["particle"][n].update({"model": "kmatrix_n",
                             "mass_list": [0.8, 1.35],
                             "width_list": [0.3, 0.25], "L": 1})
    d["backend"] = {"name": "numpy_flat", "resolution_size": 1}
    f = Fitter(d)
    f.apply_constrains()
    flat = list(f.cm.var_registry.flat_names)
    assert f"{n}_beta1r" in flat                     # a free relative weight
    assert f"{n}_beta0r" not in flat                 # the gauge head, not a param
    assert f"{n}_beta1r" not in f.cm.defaults        # no start value
    res = f.cm.flat_resolve(f.initial_values(seed=7))
    assert res[f"{n}_mass0"] == pytest.approx(0.8)           # anchor kept
    assert res[f"{n}_width0"] == pytest.approx(0.3)
    assert res[f"{n}_beta1r"] != pytest.approx(1.0)          # random draw

    # N = 1 has NO numerator parameter at all (A = 1/den)
    one = build_particle(n, model="kmatrix_n", mass_list=[0.8],
                         width_list=[0.3])
    assert one.make_basis_transform().input_names == [
        f"{n}_mass0", f"{n}_width0"]
    # ... and with beta0 fixed, N poles carry 2N + 2(N-1) free parameters,
    # the same count as N Breit-Wigners with one overall reference
    assert len([x for x in flat if x.startswith(n + "_")]) == 2 * 2 + 2


def test_drho_dm_step_choice_is_pinned():
    """`_RHO_FD_EPS = 1e-5`: fp64-exact to 1e-6, and 1e-3 is the low-precision
    escape hatch (it is measurably better when the *values* are fp32)."""
    n = "k"
    masses = [0.30, 0.60, 0.775, 2.00]

    def d(mdl, m, h, f32=False):
        a = np.asarray(mdl._rho(np.array([m + h])), dtype=complex).real[0]
        b = np.asarray(mdl._rho(np.array([m - h])), dtype=complex).real[0]
        if f32:
            a, b = np.float32(a), np.float32(b)
        return (float(a) - float(b)) / (2.0 * h)

    def truth(mdl, m):                       # Richardson on a coarse pair
        h = 1e-4
        return (4.0 * d(mdl, m, h / 2) - d(mdl, m, h)) / 3.0

    worst64 = worst32 = worst32_coarse = 0.0
    for model in ("kmatrix_n", "kmatrix_n_gs"):
        for mass in masses:
            mdl = build_particle(n, model=model, mass_list=[mass],
                                 width_list=[0.15], L=1)
            mdl.register_parent(_Parent())
            assert mdl._RHO_FD_EPS == 1e-5
            t = truth(mdl, mass)
            h = mdl._RHO_FD_EPS * max(1.0, mass)
            worst64 = max(worst64, abs(d(mdl, mass, h) - t) / abs(t))
            worst32 = max(worst32, abs(d(mdl, mass, h, True) - t) / abs(t))
            worst32_coarse = max(
                worst32_coarse,
                abs(d(mdl, mass, 1e-3 * max(1.0, mass), True) - t) / abs(t))

    assert worst64 <= 1e-6                   # fp64: no practical error
    assert worst32_coarse <= 1e-3            # coarse step survives fp32 values
    assert worst32_coarse * 10.0 < worst32   # ... and beats the fine step there


def test_drho_dm_hook_defaults_to_fd_and_takes_an_exact_override():
    """`drho_dm` is the one hook: FD by default, exact if the user supplies it."""
    n = "k"
    mdl = build_particle(n, model="kmatrix_n", mass_list=[0.70, 1.10],
                         width_list=[0.15, 0.20], L=1)      # width_norm: true
    mdl.register_parent(_Parent())
    m = np.array([0.70, 1.10])
    w = np.array([0.15, 0.20])

    # default: a centred difference of Re rho (an independent step here)
    h = 1e-6
    ref = (np.asarray(mdl._rho(m + h), dtype=complex).real
           - np.asarray(mdl._rho(m - h), dtype=complex).real) / (2.0 * h)
    assert np.allclose(mdl.drho_dm(m), ref, rtol=1e-6)

    # an EXACT override: pretend rho = m², so dRe rho/dm = 2m
    mdl._rho = lambda x: np.asarray(x, dtype=float) ** 2
    mdl.drho_dm = lambda x: 2.0 * np.asarray(x, dtype=float)
    g2, dg2_dm, dg2_dg = mdl._coupling_sq_jac(m, w)
    assert np.allclose(g2, w / m)                 # m·Γ/m²
    assert np.allclose(dg2_dm, -w / m ** 2)       # exact, no FD error
    assert np.allclose(dg2_dg, 1.0 / m)

    # ... and the full VJP is then exact against a difference of `forward`
    tr = mdl.make_basis_transform()
    params = {f"{n}_mass0": 0.70, f"{n}_mass1": 1.10,
              f"{n}_width0": 0.15, f"{n}_width1": 0.20,
              f"{n}_beta1r": 0.7, f"{n}_beta1i": -0.4}
    tr.apply_forward(dict(params))
    cot = {nm: float(i + 1) * (1.0 if i % 2 == 0 else -1.0)
           for i, nm in enumerate(tr.output_names)}
    vjp = tr.backward(cot, d_in=dict(params))

    def scalar(x):
        o = tr.forward(x)
        return sum(cot[k] * o[k] for k in tr.output_names)

    for name, val in vjp.items():
        x = params[name]
        step = 1e-5 * max(1.0, abs(x))
        fd = (scalar({**params, name: x + step})
              - scalar({**params, name: x - step})) / (2.0 * step)
        assert val == pytest.approx(fd, rel=1e-6, abs=1e-9), name


@pytest.mark.parametrize("n_pole", [1, 2, 3, 4])
def test_width_norm_false_has_an_analytic_vjp(n_pole):
    """`g² = m·Γ` makes every weight polynomial ⇒ closed-form VJP, no FD."""
    n = "k"
    m = [0.70 + 0.40 * i for i in range(n_pole)]
    w = [0.15 + 0.05 * i for i in range(n_pole)]
    mdl = build_particle(n, model="kmatrix_n", mass_list=m, width_list=w,
                         width_norm=False, L=1)
    mdl.register_parent(_Parent())
    tr = mdl.make_basis_transform()
    params = {f"{n}_mass{i}": m[i] for i in range(n_pole)}
    params.update({f"{n}_width{i}": w[i] for i in range(n_pole)})
    for i in range(1, n_pole):
        params[f"{n}_beta{i}r"] = 0.5 + 0.2 * i
        params[f"{n}_beta{i}i"] = -0.3 * i
    tr.apply_forward(dict(params))

    cot = {nm: float(i + 1) * (1.0 if i % 2 == 0 else -1.0)
           for i, nm in enumerate(tr.output_names)}
    calls = []
    original = tr.forward
    tr.forward = lambda d: (calls.append(1), original(d))[1]
    try:
        vjp = tr.backward(cot, d_in=dict(params))
    finally:
        tr.forward = original
    assert calls == []            # analytic: no forward evaluations at all

    def scalar(x):
        o = original(x)
        return sum(cot[k] * o[k] for k in tr.output_names)

    for name, val in vjp.items():
        x = params[name]
        h = 1e-5 * max(1.0, abs(x))          # an order above fd_eps
        fd = (scalar({**params, name: x + h})
              - scalar({**params, name: x - h})) / (2.0 * h)
        assert val == pytest.approx(fd, rel=1e-7, abs=1e-8), name


def test_width_norm_true_vjp_is_analytic_apart_from_rho():
    """`g² = m·Γ/Re ρ(m)` — analytic VJP; only `dRe ρ/dm` is a difference.

    The ρ derivative is TWO cheap evaluations of ``_rho``, not the 2·n_input
    differences of the whole transform, so ``backward`` still calls no
    ``forward``.
    """
    n = "k"
    mdl = build_particle(n, model="kmatrix_n", mass_list=[0.70, 1.10],
                         width_list=[0.15, 0.20], L=1)      # width_norm: true
    mdl.register_parent(_Parent())
    m = np.array([0.70, 1.10])
    w = np.array([0.15, 0.20])

    _, dg2_dm, dg2_dg = mdl._coupling_sq_jac(m, w)
    ref = (np.asarray(mdl._coupling_sq(m + 1e-7, w), dtype=float)
           - np.asarray(mdl._coupling_sq(m - 1e-7, w), dtype=float)) / 2e-7
    assert np.allclose(dg2_dm, ref, rtol=1e-6)
    rho = np.asarray(mdl._rho(m), dtype=complex).real
    assert np.allclose(dg2_dg, m / rho, rtol=1e-12)

    tr = mdl.make_basis_transform()
    params = {f"{n}_mass0": 0.70, f"{n}_mass1": 1.10,
              f"{n}_width0": 0.15, f"{n}_width1": 0.20,
              f"{n}_beta1r": 0.7, f"{n}_beta1i": -0.4}
    tr.apply_forward(dict(params))
    cot = {nm: float(i + 1) * (1.0 if i % 2 == 0 else -1.0)
           for i, nm in enumerate(tr.output_names)}

    calls = []
    original = tr.forward
    tr.forward = lambda d: (calls.append(1), original(d))[1]
    try:
        vjp = tr.backward(cot, d_in=dict(params))
    finally:
        tr.forward = original
    assert calls == []                     # no full-transform differences

    def scalar(x):
        o = original(x)
        return sum(cot[k] * o[k] for k in tr.output_names)

    for name, val in vjp.items():
        x = params[name]
        h = 1e-5 * max(1.0, abs(x))        # an order above fd_eps
        fd = (scalar({**params, name: x + h})
              - scalar({**params, name: x - h})) / (2.0 * h)
        assert val == pytest.approx(fd, rel=1e-6, abs=1e-8), name


@pytest.mark.parametrize("n_pole", [1, 2, 3, 4, 5])
def test_den_names_have_one_entry_per_basis_row(n_pole):
    """`get_den_name()` / `get_num_name()` must match the basis LENGTH."""
    n = "k"
    mdl = build_particle(n, model="kmatrix_n",
                         mass_list=[0.7 + 0.2 * i for i in range(n_pole)],
                         width_list=[0.15] * n_pole)
    mdl.register_parent(_Parent())
    grid = np.array([0.80])
    assert len(mdl.den_basis(grid)) == 2 * n_pole + 1
    assert len(mdl.get_den_name()) == 2 * n_pole + 1
    assert len(mdl.num_basis(grid)) == len(mdl.get_num_name()) == n_pole

    out = mdl.make_basis_transform().apply_forward(mdl.get_defaults())
    for nm in mdl.get_den_name() + mdl.get_num_name():
        assert nm + "r" in out or nm + "i" in out or nm in out, nm

    # the names must be distinct (a repeat silently merges two rows)
    assert len(set(mdl.get_den_name())) == 2 * n_pole + 1
    assert len(set(mdl.get_num_name())) == n_pole


@pytest.mark.parametrize("n_pole", [1, 2, 3])
def test_numerator_and_denominator_share_one_row_set(n_pole):
    """P and K ride on the SAME rows: `num_basis == den_basis[:N]`, and the ρ
    rows are those rows times `−iρ` — one basis, three coefficient choices."""
    n = "k"
    m = [0.70 + 0.40 * i for i in range(n_pole)]
    w = [0.15 + 0.05 * i for i in range(n_pole)]
    mdl = build_particle(n, model="kmatrix_n", mass_list=m, width_list=w, L=1)
    mdl.register_parent(_Parent())
    grid = np.array([0.50, 0.95, 1.30])
    den = mdl.den_basis(grid)
    num = mdl.num_basis(grid)
    rho = np.asarray(mdl._rho(grid), dtype=complex)
    assert len(num) == n_pole and len(den) == 2 * n_pole + 1
    for k in range(n_pole):
        assert np.array_equal(num[k], den[k])          # P's rows ARE K's real rows
        assert np.allclose(den[n_pole + 1 + k], -1j * rho * num[k],
                           rtol=0, atol=0)              # ρ rows = −iρ × same rows


@pytest.mark.parametrize("n_pole", [1, 2, 3, 4])
def test_amplitude_matches_the_direct_kmatrix(n_pole):
    """`amp` == P/(1 − iρK) written out by hand (no transform machinery).

    ``beta0 = 1`` (the gauge head) and ``beta{i>=1} = 1`` (the transform's
    fallback), so P = Σ_i 1/(m_i² − s) and K = Σ_i g_i²/(m_i² − s).
    """
    n = "k"
    m = [0.70 + 0.40 * i for i in range(n_pole)]
    w = [0.15 + 0.05 * i for i in range(n_pole)]
    mdl = build_particle(n, model="kmatrix_n", mass_list=m, width_list=w, L=1)
    mdl.register_parent(_Parent())

    grid = np.array([0.50, 0.95, 1.30])
    s = grid ** 2
    rho = np.asarray(mdl._rho(grid), dtype=complex)
    g2 = np.asarray(mdl._coupling_sq(m, w), dtype=float)
    K = sum(g2[i] / (m[i] ** 2 - s) for i in range(n_pole))
    P = sum(1.0 / (m[i] ** 2 - s) for i in range(n_pole))
    expected = P / (1.0 - 1j * rho * K)

    assert np.allclose(mdl.amp(grid, mdl.get_defaults()), expected,
                       rtol=1e-13, atol=1e-13)


def test_parameters_are_masses_and_widths():
    """No `g` parameter: the coupling is derived from the width."""
    n = "k"
    mdl = build_particle(n, model="kmatrix_n", mass_list=[0.775, 1.45],
                         width_list=[0.149, 0.30], L=1)
    keys = set(mdl.get_defaults())
    assert f"{n}_g1" not in keys
    assert f"{n}_width0" in keys and f"{n}_mass0" in keys
    # the scalar reference pair is build-time only — not a fit parameter
    assert f"{n}_mass" not in keys and f"{n}_width" not in keys
    names = mdl.make_basis_transform().input_names
    assert set(names) == {
        f"{n}_mass0", f"{n}_mass1", f"{n}_width0", f"{n}_width1",
        f"{n}_beta1r", f"{n}_beta1i",          # beta0 is the fixed gauge head
    }
    assert f"{n}_beta0r" not in names



def test_masses_and_widths_alone_reproduce_the_scalar_models():
    """One pole: the list defaults ARE BWRS / GS_rho — no coupling arithmetic."""
    n, mass, width, L = "rho", 0.775, 0.149, 1
    bw = build_particle(n, model="BWRS", mass=mass, width=width, L=L)
    gs = build_particle(n, model="GS_rho", mass=mass, width=width, L=L)
    k_bw = build_particle(n, model="kmatrix_n", mass_list=[mass],
                          width_list=[width], L=L)
    k_gs = build_particle(n, model="kmatrix_n_gs", mass_list=[mass],
                          width_list=[width], L=L)
    for mdl in (bw, gs, k_bw, k_gs):
        mdl.register_parent(_Parent())

    grid = np.array([0.60, 0.775, 1.00, 1.50, 2.00])
    a_bw = bw.amplitude(grid, {f"{n}_mass": mass, f"{n}_width": width})
    a_gs = gs.amplitude(grid, {f"{n}_mass": mass, f"{n}_width": width})
    assert np.allclose(k_bw.amp(grid, k_bw.get_defaults()), a_bw,
                       rtol=1e-12, atol=1e-14)
    assert np.allclose(k_gs.amp(grid, k_gs.get_defaults()), a_gs,
                       rtol=1e-12, atol=1e-14)


def test_width_norm_switches_the_pole_normalisation():
    """`width_norm: false` drops the `/Re ρ(m_i)`: g² = m·Γ, no m0 dependence."""
    n, m0, G0, L = "rho", 0.775, 0.149, 1
    rho0 = phase_space(np.array([m0]), L, 3.0, _Out.mass, _Out.mass)[0]

    mdl = build_particle(n, model="kmatrix_n", mass_list=[m0], width_list=[G0], L=L)
    off = build_particle(n, model="kmatrix_n", mass_list=[m0], width_list=[G0], L=L,
                         width_norm=False)
    for m_ in (mdl, off):
        m_.register_parent(_Parent())
    assert mdl._coupling_sq([m0], [G0])[0] == pytest.approx(m0 * G0 / rho0)
    assert off._coupling_sq([m0], [G0])[0] == pytest.approx(m0 * G0)
    assert off.width_norm is False and mdl.width_norm is True
    # the width parameters themselves are identical under both settings
    assert off.get_defaults()[f"{n}_width0"] == pytest.approx(G0)

    # with the normalisation off, the −i coefficient is m0·Γ0·ρ(m): the
    # declared width comes back scaled by Re ρ(m0) at the pole
    m = np.array([0.60, 0.775, 1.00])
    D = 1.0 / off.amp(m, off.get_defaults())
    mG = 1j * (D - (m0 ** 2 - m ** 2))
    assert np.allclose(mG, m0 * G0 * np.asarray(off._rho(m), complex),
                       rtol=1e-13, atol=0)
    assert mG[1].real / m0 == pytest.approx(G0 * rho0, rel=1e-13)

    # a unit already normalised at the pole is unaffected by the switch
    gs_on = build_particle(n, model="kmatrix_n_gs", mass_list=[m0], width_list=[G0],
                           L=L)
    gs_off = build_particle(n, model="kmatrix_n_gs", mass_list=[m0], width_list=[G0],
                            L=L, width_norm=False)
    for m_ in (gs_on, gs_off):
        m_.register_parent(_Parent())
    assert gs_on._coupling_sq([m0], [G0])[0] == pytest.approx(G0 * m0)
    assert gs_off._coupling_sq([m0], [G0])[0] == pytest.approx(G0 * m0)

    for bad in ("rho", "none", 0, 1):
        with pytest.raises(ValueError, match="width_norm must be true"):
            build_particle(n, model="kmatrix_n", mass_list=[m0], width_list=[G0],
                           width_norm=bad)


@pytest.mark.parametrize("model", ["kmatrix_n", "kmatrix_n_gs"])
def test_width_convention_is_re_rho_at_the_pole(model):
    """A `width_list` entry is the width AT ITS POLE.

    The ``-i`` coefficient of the propagator is
    ``m·Γ(m) = m0·Γ0·ρ(m)/Re ρ(m0)`` — the full ``ρ(m)`` (complex for
    ``gs_unit``), with only the pole normalisation real — so
    ``Γ(m0) = Γ0``; the coupling is ``g² = m0·Γ0/Re ρ(m0)``.  Checked end to
    end through the amplitude (one pole, ``b1 = 1`` ⇒ ``1/A`` IS the
    denominator).
    """
    n, m0, G0, L = "rho", 0.775, 0.149, 1
    mdl = build_particle(n, model=model, mass_list=[m0], width_list=[G0], L=L)
    mdl.register_parent(_Parent())
    p = mdl.get_defaults()

    rho0 = np.asarray(mdl._rho(np.array([m0])), dtype=complex)[0].real
    assert p[f"{n}_width0"] == pytest.approx(G0)
    assert mdl._coupling_sq([m0], [G0])[0] == pytest.approx(m0 * G0 / rho0)

    m = np.array([0.35, 0.60, 0.775, 1.00, 1.50, 2.50])
    D = 1.0 / mdl.amp(m, p)                   # one pole: D = m0² - s - i·ρ·g²
    mG = 1j * (D - (m0 ** 2 - m ** 2))        # = ρ(m)·g², the -i coefficient
    rho = np.asarray(mdl._rho(m), dtype=complex)
    assert np.allclose(mG, m0 * G0 * rho / rho0, rtol=1e-13, atol=0)
    if np.any(rho.imag):
        assert np.abs(mG.imag).max() > 0      # GS: the dispersive piece is real
    i0 = int(np.argmin(np.abs(m - m0)))
    assert mG[i0].real == pytest.approx(m0 * G0, rel=1e-13)   # Γ(m0) = Γ0


def test_weights_are_the_polynomial_coefficients():
    n = "k"
    m = [0.80, 1.35, 1.80, 2.40]
    g = [0.35, 0.55, 0.30, 0.20]
    b = [0.9 + 0.0j, 0.4 + 0.0j, 0.2 + 0.0j, 0.1 - 0.2j]
    mdl = build_particle(n, model="kmatrix_n", mass_list=m,
                         width_list=[0.1] * len(m))
    mdl.register_parent(_Parent())
    grid = np.array([0.60, 0.80, 1.20])
    s = grid ** 2
    rho = np.asarray(phase_space(grid, 0, 3.0, _Out.mass, _Out.mass), complex)
    roots = [x * x for x in m]

    def prod(skip=None):
        out = np.ones_like(s, dtype=complex)
        for i, r in enumerate(roots):
            if i != skip:
                out = out * (r - s)
        return out

    den_direct = prod() - 1j * rho * sum(g[i] ** 2 * prod(i) for i in range(4))
    # beta0 = 1 (the gauge head); b[1:] are the fitted relative weights
    num_direct = prod(0) + sum(b[i] * prod(i) for i in range(1, 4))

    # the width that produces the intended g² at each pole
    w = np.array(g) ** 2 * np.asarray(mdl._rho(np.array(m)), complex).real \
        / np.array(m)
    out = mdl.make_basis_transform().apply_forward(
        {**{f"{k}": v for k, v in mdl._transform_defaults().items()},
         **_resolve(n, m, w, b)})
    w_den = [float(out[nm]) for nm in mdl.get_den_name()]
    den = sum(val * row for val, row in zip(w_den, mdl.den_basis(grid)))
    num = sum(out[f"{h}r"] * np.exp(1j * out[f"{h}i"]) * row
              for h, row in zip(mdl.get_num_name(), mdl.num_basis(grid)))
    assert np.allclose(den, den_direct, rtol=1e-14, atol=1e-14)
    # the numerator sums the beta0 head (O(m^6)) with three smaller ones, so
    # the cancellation costs a couple of ulp at the m^0 coefficient
    assert np.allclose(num, num_direct, rtol=1e-13, atol=1e-13)


def test_rho_is_a_method_not_an_option():
    """The phase space is the overridable method; the GS variant is a subclass."""
    from tabpwa.particle_model.gs_rho_model import gs_unit

    n, mass, width, L = "rho", 0.775, 0.149, 1
    mass_list = [mass, 1.50, 2.00]
    width_list = [width, 0.40, 0.25]
    bw = build_particle(n, model="kmatrix_n", mass_list=mass_list, width_list=width_list, L=L)
    gs = build_particle(n, model="kmatrix_n_gs", mass_list=mass_list, width_list=width_list,
                        L=L)
    for mdl in (bw, gs):
        mdl.register_parent(_Parent())

    grid = np.array([0.60, 0.775, 1.00, 1.50])
    d2 = _Out.mass
    assert np.allclose(bw._rho(grid),
                       phase_space(grid, L, 3.0, d2, d2), rtol=1e-15)
    assert np.allclose(gs._rho(grid), gs_unit(grid, mass, L, 3.0, d2, d2),
                       rtol=1e-15)
    assert not np.allclose(bw._rho(grid), gs._rho(grid))

    # identical arity and names: the subclass changes only the shape
    assert bw.get_den_name() == gs.get_den_name()
    assert bw.get_num_name() == gs.get_num_name()
    assert bw.n_pole == gs.n_pole == 3
    # `rho:` is not a config knob any more
    assert not hasattr(bw, "rho_kind")
    # ... and the width -> coupling rule follows `_rho`
    rho0 = phase_space(np.array([mass]), L, 3.0, d2, d2)[0]
    assert bw._coupling_sq([mass], [width])[0] == pytest.approx(
        mass * width / rho0)
    assert gs._coupling_sq([mass], [width])[0] == pytest.approx(mass * width)


def test_builds_in_pwa_gen():
    from tabpwa import Fitter
    from tabpwa.config_loader import RawConfig

    km = build_particle("k2", model="kmatrix_n", mass_list=[0.8, 1.35, 1.8, 2.4],
                        width_list=[0.3, 0.3, 0.2, 0.15], L=1)
    assert len(km.get_den_name()) == 9
    for name in ("gamma", "get_gamma_name", "make_mass_width_transform"):
        assert not hasattr(km, name), name

    # N = 1 AND N = 3: the den name list must equal the basis length in the
    # kernel config too (the single-pole case has ONE rho row)
    for model, mass_list, width_list, n_den in (
            ("kmatrix_n", [0.8], [0.3], 3),
            ("kmatrix_n_gs", [0.8], [0.3], 3),
            ("kmatrix_n", [0.8, 1.35, 1.8], [0.3, 0.3, 0.2], 7),
            ("kmatrix_n_gs", [0.8, 1.35, 1.8], [0.3, 0.3, 0.2], 7)):
        d = RawConfig("tests/config_pwa_gen.yml").to_dict()
        d["particle"]["k2"].update({"model": model, "mass_list": mass_list,
                                    "width_list": width_list, "L": 1})
        d["backend"] = {"name": "numpy_flat", "resolution_size": 1}
        f = Fitter(d)
        f.apply_constrains()
        assert len(f.kernel_config["den_names"]) == n_den, (model, len(mass_list))
        assert any(k.endswith("_width0") for k in f.model.particle_defaults()), model
