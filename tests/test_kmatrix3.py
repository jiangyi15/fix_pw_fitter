"""``kmatrix3`` — the three-pole K-matrix driven by the BW running width.

Gates: the polynomial weights against the direct algebraic form, the reduction
to ``kmatrix2_bf`` when the third pole is switched off, the one-pole limit
equal to ``BWRS``, and the interface (basis-only: the plain ``pwa`` path
refuses it, ``pwa_gen`` builds a 7-row basis).
"""
import numpy as np

from tabpwa.particle_model import build_particle
from tabpwa.particle_model.bwr_model import phase_space
from tabpwa.particle_model.kmatrix3_model import kmatrix3_weights


class _Out:
    mass = 0.13957039


class _Dec:
    outs = [_Out(), _Out()]


class _Parent:
    """Minimal decay-tree stub: the two pions both shapes need."""

    _decays = [_Dec()]


def _phys(n, m, g, b):
    out = {}
    for i, (mi, gi) in enumerate(zip(m, g), start=1):
        out[f"{n}_m{i}"] = float(mi)
        out[f"{n}_g{i}"] = float(gi)
    for i, bi in enumerate(b, start=1):
        out[f"{n}_b{i}r"] = float(abs(bi))
        out[f"{n}_b{i}i"] = float(np.angle(bi))
    return out


def test_weights_are_the_elementary_symmetric_polynomials():
    """``Σ weights·rows`` equals ``Π(m²−s) − iρ·Σ g² Π_{j≠i}`` directly."""
    n, mass, L = "k2", 0.80, 1
    m = [0.80, 1.35, 1.80]
    g = [0.35, 0.55, 0.25]
    b = [0.9 + 0.0j, 0.4 + 0.0j, 0.2 + 0.0j]
    mdl = build_particle(n, model="kmatrix3", mass=mass, width=0.3, L=L)
    mdl.register_parent(_Parent())
    params = _phys(n, m, g, b)

    grid = np.array([0.60, 0.80, 1.00])
    s = grid ** 2
    rho = np.asarray(phase_space(grid, L, 3.0, _Out.mass, _Out.mass),
                     dtype=complex)
    roots = [x * x for x in m]

    def prod(skip=None):
        out = np.ones_like(s, dtype=complex)
        for i, r in enumerate(roots):
            if i != skip:
                out = out * (r - s)
        return out

    den_direct = prod() - 1j * rho * sum(g[i] ** 2 * prod(i) for i in range(3))
    num_direct = sum(b[i] * prod(i) for i in range(3))

    out = mdl.make_basis_transform().apply_forward(
        {**{f"{n}_{k}": v for k, v in mdl._phys().items()}, **params})
    w = [float(out[name]) for name in mdl.get_den_name()]
    den_basis = np.zeros_like(s, dtype=complex)
    for val, row in zip(w, mdl.den_basis(grid)):
        den_basis = den_basis + val * row
    num = np.zeros_like(s, dtype=complex)
    for h, row in zip(mdl.get_num_name(), mdl.num_basis(grid)):
        num = num + out[f"{h}r"] * np.exp(1j * out[f"{h}i"]) * row

    assert np.allclose(den_basis, den_direct, rtol=1e-14, atol=1e-14)
    assert np.allclose(num, num_direct, rtol=1e-14, atol=1e-14)
    # the ratio agrees to the round-off of D (a grid point sits near Re D = 0)
    assert np.allclose(num / den_basis, num_direct / den_direct,
                       rtol=1e-12, atol=1e-12)


def test_three_poles_reduce_to_two():
    """``g3 = b3 = 0`` factors ``(m3²−s)`` out — the ratio IS ``kmatrix2_bf``."""
    n, mass, L = "k2", 0.80, 1
    m = [0.80, 1.35]
    g = [0.35, 0.55]
    b = [0.9 + 0.0j, 0.4 + 0.0j]
    km2 = build_particle(n, model="kmatrix2_bf", mass=mass, width=0.3, L=L)
    km3 = build_particle(n, model="kmatrix3", mass=mass, width=0.3, L=L)
    for mdl in (km2, km3):
        mdl.register_parent(_Parent())

    grid = np.array([0.60, 0.80, 1.00])
    a2 = km2.amp(grid, _phys(n, m, g, b))
    a3 = km3.amp(grid, _phys(n, m + [1.80], g + [0.0], b + [0.0j]))
    assert np.allclose(a3, a2, rtol=1e-12, atol=1e-14)


def test_one_pole_is_bwrs():
    """One pole ⇒ the amplitude is ``BWRS`` exactly (``bf² = m0·Γ0``)."""
    n, mass, width, L = "rho", 0.775, 0.149, 1
    bw = build_particle(n, model="BWRS", mass=mass, width=width, L=L)
    k3 = build_particle(n, model="kmatrix3", mass=mass, width=width, L=L)
    for mdl in (bw, k3):
        mdl.register_parent(_Parent())

    grid = np.array([0.60, 0.775, 1.00])
    a_bw = bw.amplitude(grid, {f"{n}_mass": mass, f"{n}_width": width})
    rho_m0 = phase_space(np.array([mass]), L, 3.0, _Out.mass, _Out.mass)[0]
    p = _phys(n, [mass, 2.0, 2.5],
              [np.sqrt(mass * width / rho_m0), 0.0, 0.0],
              [1.0 + 0.0j, 0.0j, 0.0j])
    assert np.allclose(k3.amp(grid, p), a_bw, rtol=1e-12, atol=1e-14)

    # ... and the extra poles genuinely change the shape
    p2 = _phys(n, [mass, 1.45, 1.90],
               [np.sqrt(mass * width / rho_m0), 0.40, 0.30],
               [1.0 + 0.0j, 0.50 + 0.0j, 0.30 + 0.0j])
    assert not np.allclose(k3.amp(grid, p2), a_bw, rtol=1e-4, atol=0)


def test_kmatrix3_is_basis_only_and_pwa_gen_builds():
    """A pure ``BasisModel``: no running-width interface, 7-row flat basis."""
    from tabpwa import Fitter
    from tabpwa.config_loader import RawConfig

    km = build_particle("k2", model="kmatrix3", mass=0.8, width=0.3)
    for name in ("gamma", "get_gamma_name", "make_mass_width_transform"):
        assert not hasattr(km, name), name

    d = RawConfig("tests/config_pwa_gen.yml").to_dict()
    d["particle"]["k2"].update({"model": "kmatrix3", "L": 1})
    d["backend"] = {"name": "numpy_flat", "resolution_size": 1}
    f = Fitter(d)
    f.apply_constrains()
    assert f.kernel_config["den_names"] == [
        "k2_s0", "k2_poly", "k2_m4", "k2_m6", "k2_rho", "k2_srho", "k2_s2rho"]

    d2 = dict(d)
    d2["amp_model"] = "pwa"
    d2["backend"] = {"name": "numpy_pwa"}
    try:
        Fitter(d2).apply_constrains()
    except ValueError as exc:
        assert "basis-only" in str(exc)
    else:                                   # pragma: no cover
        raise AssertionError("the plain-pwa path accepted a basis-only model")


def test_kmatrix3_gs_one_pole_is_gs_rho():
    """``kmatrix3_gs``'s ρ is the COMPLEX GS unit: one pole ⇒ ``GS_rho``.

    ``−i·ρ·bf² = g0·fs − i·m0·Γ`` for ``bf² = m0·Γ0``, i.e. the GS propagator
    ``m0² − s + g0·fs − i·m0·Γ`` exactly (a dispersive, not just absorptive,
    phase space).
    """
    n, mass, width, L = "rho", 0.775, 0.149, 1
    gs = build_particle(n, model="GS_rho", mass=mass, width=width, L=L)
    k3 = build_particle(n, model="kmatrix3_gs", mass=mass, width=width, L=L)
    for mdl in (gs, k3):
        mdl.register_parent(_Parent())

    grid = np.array([0.60, 0.775, 1.00])
    a_gs = gs.amplitude(grid, {f"{n}_mass": mass, f"{n}_width": width})
    p = _phys(n, [mass, 2.0, 2.5], [np.sqrt(mass * width), 0.0, 0.0],
              [1.0 + 0.0j, 0.0j, 0.0j])
    assert np.allclose(k3.amp(grid, p), a_gs, rtol=1e-12, atol=1e-14)

    # extra poles change the shape
    p2 = _phys(n, [mass, 1.45, 1.90], [np.sqrt(mass * width), 0.40, 0.30],
               [1.0 + 0.0j, 0.50 + 0.0j, 0.30 + 0.0j])
    assert not np.allclose(k3.amp(grid, p2), a_gs, rtol=1e-4, atol=0)

    for name in ("gamma", "get_gamma_name", "make_mass_width_transform"):
        assert not hasattr(k3, name), name


def test_kmatrix3_gs_builds_in_pwa_gen():
    """The GS variant is a 7-row basis like the others."""
    from tabpwa import Fitter
    from tabpwa.config_loader import RawConfig

    d = RawConfig("tests/config_pwa_gen.yml").to_dict()
    d["particle"]["k2"].update({"model": "kmatrix3_gs", "L": 1})
    d["backend"] = {"name": "numpy_flat", "resolution_size": 1}
    f = Fitter(d)
    f.apply_constrains()
    assert f.kernel_config["den_names"] == [
        "k2_s0", "k2_poly", "k2_m4", "k2_m6", "k2_rho", "k2_srho", "k2_s2rho"]
