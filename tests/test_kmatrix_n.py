"""``kmatrix_n`` — one-channel K-matrix with any number of poles.

Gates: the pole content is a LIST of masses and widths (``n_pole =
len(masses)``), that list alone reproduces ``BWRS`` / ``GS_rho`` at one pole
(the width -> coupling rule is the width-preserving one), the FITTED
parameters are the pole masses and WIDTHS (there is no ``g`` parameter), the
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
    out = {f"{n}_mass{i}": float(mi) for i, mi in enumerate(m, start=1)}
    out.update({f"{n}_width{i}": float(wi) for i, wi in enumerate(w, start=1)})
    for i, bi in enumerate(b, start=1):
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


def test_pole_count_is_len_masses():
    """`masses` IS the pole count; `widths` is one entry per pole."""
    n = "k"
    m = [0.775, 1.45, 1.90, 2.40]
    w = [0.149, 0.30, 0.20, 0.15]
    mdl = build_particle(n, model="kmatrix_n", masses=m, widths=w, L=1)
    assert mdl.n_pole == 4
    assert len(mdl.get_den_name()) == 2 * 4 + 1
    assert len(mdl.get_num_name()) == 4

    d = mdl.get_defaults()
    for i, (mi, wi) in enumerate(zip(m, w), start=1):
        assert d[f"{n}_mass{i}"] == pytest.approx(mi)
        assert d[f"{n}_width{i}"] == pytest.approx(wi)   # widths ARE the params

    # ... and the transform turns each width into g² = m·Γ/ρ(m)
    # (no tree attached here -> the config's default daughters, π± and π0)
    rho = phase_space(np.array(m), 1, 3.0, _Out.mass, 0.1349768)
    g2 = mdl._coupling_sq(np.array(m), np.array(w))
    assert np.allclose(g2, np.array(m) * np.array(w) / rho, rtol=1e-15)

    with pytest.raises(ValueError, match="one width per pole"):
        build_particle(n, model="kmatrix_n", masses=m, widths=w[:-1])
    with pytest.raises(ValueError, match="the list IS the pole count"):
        build_particle(n, model="kmatrix_n", masses=m, n_pole=3)
    with pytest.raises(ValueError, match="masses must not be empty"):
        build_particle(n, model="kmatrix_n", masses=[])


@pytest.mark.parametrize("n_pole", [1, 2, 3, 4, 5])
def test_den_names_have_one_entry_per_basis_row(n_pole):
    """`get_den_name()` / `get_num_name()` must match the basis LENGTH."""
    n = "k"
    mdl = build_particle(n, model="kmatrix_n", n_pole=n_pole)
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
    a = build_particle(n, model="kmatrix_n", n_pole=n_pole).get_den_name()
    b = build_particle(n, model=ref).get_den_name()
    assert a == b


def test_parameters_are_masses_and_widths():
    """No `g` parameter: the coupling is derived from the width."""
    n = "k"
    mdl = build_particle(n, model="kmatrix_n", masses=[0.775, 1.45],
                         widths=[0.149, 0.30], L=1)
    keys = set(mdl.get_defaults())
    assert f"{n}_g1" not in keys
    assert f"{n}_width1" in keys and f"{n}_mass1" in keys
    # the scalar reference pair is build-time only — not a fit parameter
    assert f"{n}_mass" not in keys and f"{n}_width" not in keys
    assert set(mdl.make_basis_transform().input_names) == {
        f"{n}_mass1", f"{n}_mass2", f"{n}_width1", f"{n}_width2",
        f"{n}_beta1r", f"{n}_beta1i", f"{n}_beta2r", f"{n}_beta2i",
    }



def test_masses_and_widths_alone_reproduce_the_scalar_models():
    """One pole: the list defaults ARE BWRS / GS_rho — no coupling arithmetic."""
    n, mass, width, L = "rho", 0.775, 0.149, 1
    bw = build_particle(n, model="BWRS", mass=mass, width=width, L=L)
    gs = build_particle(n, model="GS_rho", mass=mass, width=width, L=L)
    k_bw = build_particle(n, model="kmatrix_n", masses=[mass],
                          widths=[width], L=L)
    k_gs = build_particle(n, model="kmatrix_n_gs", masses=[mass],
                          widths=[width], L=L)
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

    mdl = build_particle(n, model="kmatrix_n", masses=[m0], widths=[G0], L=L)
    off = build_particle(n, model="kmatrix_n", masses=[m0], widths=[G0], L=L,
                         width_norm=False)
    for m_ in (mdl, off):
        m_.register_parent(_Parent())
    assert mdl._coupling_sq([m0], [G0])[0] == pytest.approx(m0 * G0 / rho0)
    assert off._coupling_sq([m0], [G0])[0] == pytest.approx(m0 * G0)
    assert off.width_norm is False and mdl.width_norm is True
    # the width parameters themselves are identical under both settings
    assert off.get_defaults()[f"{n}_width1"] == pytest.approx(G0)

    # with the normalisation off, the −i coefficient is m0·Γ0·ρ(m): the
    # declared width comes back scaled by Re ρ(m0) at the pole
    m = np.array([0.60, 0.775, 1.00])
    D = 1.0 / off.amp(m, off.get_defaults())
    mG = 1j * (D - (m0 ** 2 - m ** 2))
    assert np.allclose(mG, m0 * G0 * np.asarray(off._rho(m), complex),
                       rtol=1e-13, atol=0)
    assert mG[1].real / m0 == pytest.approx(G0 * rho0, rel=1e-13)

    # a unit already normalised at the pole is unaffected by the switch
    gs_on = build_particle(n, model="kmatrix_n_gs", masses=[m0], widths=[G0],
                           L=L)
    gs_off = build_particle(n, model="kmatrix_n_gs", masses=[m0], widths=[G0],
                            L=L, width_norm=False)
    for m_ in (gs_on, gs_off):
        m_.register_parent(_Parent())
    assert gs_on._coupling_sq([m0], [G0])[0] == pytest.approx(G0 * m0)
    assert gs_off._coupling_sq([m0], [G0])[0] == pytest.approx(G0 * m0)

    for bad in ("rho", "none", 0, 1):
        with pytest.raises(ValueError, match="width_norm must be true"):
            build_particle(n, model="kmatrix_n", masses=[m0], widths=[G0],
                           width_norm=bad)


@pytest.mark.parametrize("model", ["kmatrix_n", "kmatrix_n_gs"])
def test_width_convention_is_re_rho_at_the_pole(model):
    """A `widths` entry is the width AT ITS POLE.

    The ``-i`` coefficient of the propagator is
    ``m·Γ(m) = m0·Γ0·ρ(m)/Re ρ(m0)`` — the full ``ρ(m)`` (complex for
    ``gs_unit``), with only the pole normalisation real — so
    ``Γ(m0) = Γ0``; the coupling is ``g² = m0·Γ0/Re ρ(m0)``.  Checked end to
    end through the amplitude (one pole, ``b1 = 1`` ⇒ ``1/A`` IS the
    denominator).
    """
    n, m0, G0, L = "rho", 0.775, 0.149, 1
    mdl = build_particle(n, model=model, masses=[m0], widths=[G0], L=L)
    mdl.register_parent(_Parent())
    p = mdl.get_defaults()

    rho0 = np.asarray(mdl._rho(np.array([m0])), dtype=complex)[0].real
    assert p[f"{n}_width1"] == pytest.approx(G0)
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
    gen = build_particle(n, model="kmatrix_n", masses=m, widths=w, L=L)
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
    mdl = build_particle(n, model="kmatrix_n", masses=m,
                         widths=[0.1] * len(m))
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
        {**{f"{k}": v for k, v in mdl._phys().items()}, **_resolve(n, m, w, b)})
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
    masses = [mass, 1.50, 2.00]
    widths = [width, 0.40, 0.25]
    bw = build_particle(n, model="kmatrix_n", masses=masses, widths=widths, L=L)
    gs = build_particle(n, model="kmatrix_n_gs", masses=masses, widths=widths,
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

    km = build_particle("k2", model="kmatrix_n", masses=[0.8, 1.35, 1.8, 2.4],
                        widths=[0.3, 0.3, 0.2, 0.15], L=1)
    assert len(km.get_den_name()) == 9
    for name in ("gamma", "get_gamma_name", "make_mass_width_transform"):
        assert not hasattr(km, name), name

    # N = 1 AND N = 3: the den name list must equal the basis length in the
    # kernel config too (the single-pole case has ONE rho row)
    for model, masses, widths, n_den in (
            ("kmatrix_n", [0.8], [0.3], 3),
            ("kmatrix_n_gs", [0.8], [0.3], 3),
            ("kmatrix_n", [0.8, 1.35, 1.8], [0.3, 0.3, 0.2], 7),
            ("kmatrix_n_gs", [0.8, 1.35, 1.8], [0.3, 0.3, 0.2], 7)):
        d = RawConfig("tests/config_pwa_gen.yml").to_dict()
        d["particle"]["k2"].update({"model": model, "masses": masses,
                                    "widths": widths, "L": 1})
        d["backend"] = {"name": "numpy_flat", "resolution_size": 1}
        f = Fitter(d)
        f.apply_constrains()
        assert len(f.kernel_config["den_names"]) == n_den, (model, len(masses))
        assert any(k.endswith("_width1") for k in f.model.particle_defaults()), model
