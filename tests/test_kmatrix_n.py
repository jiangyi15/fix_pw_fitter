"""``kmatrix_n`` — one-channel K-matrix with any number of poles.

Gates: the pole content is the ``mass_list`` / ``width_list`` pair
(``n_pole = len(mass_list)``), that pair alone reproduces ``BWRS`` / ``GS_rho``
at one pole (the width -> coupling rule is the width-preserving one), the
FITTED parameters are the pole masses and WIDTHS (there is no ``g``), the
generic class matches the fixed-arity models at N = 2 and N = 3, the phase
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
    """Resolved dict for a width-parameter model (``mass{i}`` / ``width{i}``)."""
    out = {f"{n}_mass{i}": float(mi) for i, mi in enumerate(m)}
    out.update({f"{n}_width{i}": float(wi) for i, wi in enumerate(w)})
    for i, bi in enumerate(b):
        out[f"{n}_beta{i}r"] = float(abs(bi))
        out[f"{n}_beta{i}i"] = float(np.angle(bi))
    return out


def _resolve_g(n, m, g, b):
    """Resolved dict for the fixed-arity models (``m{i}`` / ``g{i}`` / ``b{i}``)."""
    out = {f"{n}_m{i}": float(mi) for i, mi in enumerate(m, start=1)}
    out.update({f"{n}_g{i}": float(gi) for i, gi in enumerate(g, start=1)})
    for i, bi in enumerate(b, start=1):
        out[f"{n}_b{i}r"] = float(abs(bi))
        out[f"{n}_b{i}i"] = float(np.angle(bi))
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


@pytest.mark.parametrize("model,fallback", [
    ("BWRS", 0), ("kmatrix2_bf", 0), ("kmatrix3", 0), ("kmatrix3_gs", 1)])
def test_default_L_is_shared_by_the_other_models(model, fallback):
    """`BasisModel.default_L` is one rule for the whole family."""
    n, m0, G0 = "rho", 0.775, 0.149
    mdl = build_particle(n, model=model, mass=m0, width=G0)
    assert mdl.default_L() == fallback                    # no tree attached
    mdl.register_parent(_ParentLS([(1, 1), (3, 1)]))
    assert mdl.default_L() == 1

    grid = np.array([0.60, 0.775, 1.00])
    if model == "BWRS":
        from tabpwa.particle_model.bwr_model import running_width
        assert np.allclose(mdl.gamma(grid)[0],
                           running_width(grid, m0, 1, 3.0,
                                         _Out.mass, _Out.mass))
    elif model == "kmatrix3_gs":
        from tabpwa.particle_model.gs_rho_model import gs_unit
        assert np.allclose(mdl._rho(grid),
                           gs_unit(grid, m0, 1, 3.0, _Out.mass, _Out.mass))
    else:
        assert np.allclose(mdl._rho(grid),
                           phase_space(grid, 1, 3.0, _Out.mass, _Out.mass))

    # an explicit `L` still wins over the tree
    ex = build_particle(n, model=model, mass=m0, width=G0, L=2)
    ex.register_parent(_ParentLS([(0, 0), (2, 0)]))
    assert ex.default_L() == 0
    if model == "BWRS":
        from tabpwa.particle_model.bwr_model import running_width
        assert np.allclose(ex.gamma(grid)[0],
                           running_width(grid, m0, 2, 3.0,
                                         _Out.mass, _Out.mass))
    elif model == "kmatrix3_gs":
        from tabpwa.particle_model.gs_rho_model import gs_unit
        assert np.allclose(ex._rho(grid),
                           gs_unit(grid, m0, 2, 3.0, _Out.mass, _Out.mass))
    else:
        assert np.allclose(ex._rho(grid),
                           phase_space(grid, 2, 3.0, _Out.mass, _Out.mass))


def test_beta_has_no_config_key_or_default():
    """`beta{i}` is a coupling: no config key, no default — random like g_ls."""
    from tabpwa import Fitter
    from tabpwa.config_loader import RawConfig

    n = "k2"
    mdl = build_particle(n, model="kmatrix_n", mass_list=[0.8, 1.35],
                         width_list=[0.3, 0.25], beta0=1.0)   # key is inert
    assert f"{n}_beta0r" not in mdl.get_defaults()
    assert f"{n}_mass0" in mdl.get_defaults()
    assert f"{n}_width0" in mdl.get_defaults()

    d = RawConfig("tests/config_pwa_gen.yml").to_dict()
    d["particle"][n].update({"model": "kmatrix_n",
                             "mass_list": [0.8, 1.35],
                             "width_list": [0.3, 0.25], "L": 1})
    d["backend"] = {"name": "numpy_flat", "resolution_size": 1}
    f = Fitter(d)
    f.apply_constrains()
    assert f"{n}_beta0r" in f.cm.var_registry.flat_names     # IS a parameter
    assert f"{n}_beta0r" not in f.cm.defaults                # with no start value
    res = f.cm.flat_resolve(f.initial_values(seed=7))
    assert res[f"{n}_mass0"] == pytest.approx(0.8)           # anchor kept
    assert res[f"{n}_width0"] == pytest.approx(0.3)
    assert res[f"{n}_beta0r"] != pytest.approx(1.0)          # random draw


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


@pytest.mark.parametrize("n_pole", [2, 3])
def test_den_names_match_the_fixed_arity_models(n_pole):
    """Same names as `kmatrix2` / `kmatrix3` at the same arity."""
    n = "k"
    ref = "kmatrix2" if n_pole == 2 else "kmatrix3"
    a = build_particle(n, model="kmatrix_n",
                       mass_list=[0.7 + 0.2 * i for i in range(n_pole)],
                       width_list=[0.15] * n_pole).get_den_name()
    b = build_particle(n, model=ref).get_den_name()
    assert a == b


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
    assert set(mdl.make_basis_transform().input_names) == {
        f"{n}_mass0", f"{n}_mass1", f"{n}_width0", f"{n}_width1",
        f"{n}_beta0r", f"{n}_beta0i", f"{n}_beta1r", f"{n}_beta1i",
    }



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


@pytest.mark.parametrize("n_pole", [2, 3])
def test_matches_the_fixed_arity_models(n_pole):
    n, mass, L = "k2", 0.80, 1
    m = [0.80, 1.35, 1.80][:n_pole]
    w = [0.30, 0.30, 0.25][:n_pole]
    b = [0.9 + 0.0j, 0.4 + 0.0j, 0.2 + 0.0j][:n_pole]
    ref_model = "kmatrix2_bf" if n_pole == 2 else "kmatrix3"
    ref = build_particle(n, model=ref_model, mass=mass, width=0.3, L=L)
    gen = build_particle(n, model="kmatrix_n", mass_list=m, width_list=w, L=L)
    for mdl in (ref, gen):
        mdl.register_parent(_Parent())
    assert gen.n_pole == n_pole
    grid = np.array([0.60, 0.80, 1.00, 1.50, 2.0])
    # the fixed-arity models take the coupling; ours takes the width — feed
    # both the SAME g² through the width -> coupling rule
    g = np.sqrt(gen._coupling_sq(m, w))
    assert np.allclose(gen.amp(grid, _resolve(n, m, w, b)),
                       ref.amp(grid, _resolve_g(n, m, g, b)),
                       rtol=1e-12, atol=1e-14)


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
    num_direct = sum(b[i] * prod(i) for i in range(4))

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
    assert np.allclose(num, num_direct, rtol=1e-14, atol=1e-14)


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
