"""Half-integer spins: `1/2 -> 1/2 0 0` end to end, and the vertex that cannot exist.

The three finals carry spins 1/2, 0, 0 (``tests/config_pwa_gen_spin_half.yml``):
every vertex is a spin-1/2 core whose daughters differ by one half, so

* the LS enumeration steps in halves and the ORBITAL ``l`` stays an integer
  (``s = 1/2``, ``l = 1`` from the parity product -1);
* the helicity list of the top and of ``N`` is ``[-1/2, +1/2]``, so the
  projection count is ``2 x 2 = 4`` (not ``2J_top + 1 = 2``);
* ``l`` is what the angle tables index, and they are built for
  ``n_proj = 4`` columns per wave.

It also pins the LITERAL reading — ``res(1/2) -> a(0) + b(0)`` — as a REFUSAL:
``s = 0`` with ``|ja - s| = 1/2`` is not an integer, so the vertex has no
allowed ``(l, s)`` at all.  Before the refusal it reached
``build_from_groups`` and died with ``ValueError: need at least one array to
stack``.
"""
import numpy as np
import pytest
from fractions import Fraction

from tabpwa import Fitter
from tabpwa.backends import backends_for_model, create_backend
from tabpwa.helicity_angle import to_spin

import _flat_grid as fg

CFG = "tests/config_pwa_gen_spin_half.yml"

#: The literal `1/2 -> 1/2 0 0` with a spin-0 pair at the lower vertex.
IMPOSSIBLE = {
    "amp_model": "pwa_gen",
    "data": {"dat_order": ["pip", "pim", "eta"]},
    "decay": {"top": [["res", "eta"]], "res": ["pip", "pim"]},
    "particle": {
        "$top": "top", "$finals": ["pip", "pim", "eta"], "res": ["k2"],
        "top": {"J": "1/2", "P": -1, "mass": 1.5},
        "pip": {"J": 0, "P": -1, "mass": 0.139567},
        "pim": {"J": 0, "P": -1, "mass": 0.139567},
        "eta": {"J": 0, "P": -1, "mass": 0.547862},
        "k2": {"J": "1/2", "P": -1, "mass": 0.80, "width": 0.30,
               "model": "basis"},
    },
}


def _fit():
    f = Fitter(CFG)
    f.apply_constrains()
    return f


def test_the_half_integer_ls_enumeration_is_halves_for_s_and_integer_for_l():
    """``s`` is a half-integer, ``l`` is an integer (an orbital momentum).

    The pre-existing code did ``int(2*J)`` on the raw config value, so a
    fractional spin either raised (``ValueError: invalid literal for int``) or
    truncated ``|ja - s|`` down instead of up.
    """
    f = _fit()
    for ch in f.full_decay.chains:
        for dec in ch.decays:
            ls = dec.get_ls_list()
            assert ls, f"{dec} has no (l, s)"
            for l, s in ls:
                assert isinstance(l, int) and l >= 0, (dec, l, s)
                assert Fraction(s).denominator in (1, 2), (dec, l, s)
    # this fixture: both vertices are s = 1/2 with l = 1 (the parity product
    # is -1, so l must be odd)
    waves = list(f.full_decay.get_partial_waves())
    assert len(waves) == 1
    # wave = ((ls at the top vertex, ls at the lower vertex), chain)
    assert waves[0][0] == ((1, Fraction(1, 2)), (1, Fraction(1, 2)))


def test_the_projection_count_is_the_product_over_top_and_finals():
    """P = (2J_top+1) x PROD(2J_f+1) = 2 x 2 = 4, and n_proj follows it.

    ``matrix_angle`` is ``(n_angle_basis, n_proj * n_wave)`` p-major, so P is
    also the number of columns per wave — a config can neither state nor
    override it on this path.
    """
    f = _fit()
    kc = f.kernel_config
    assert len(f.model._helicity_top_states()) == 2
    assert [tuple(str(x) for x in s) for s in
            f.model._helicity_external_states()] == [
        ("-1/2", "(Fraction(-1, 2), Fraction(0, 1), Fraction(0, 1))"),
        ("-1/2", "(Fraction(1, 2), Fraction(0, 1), Fraction(0, 1))"),
        ("1/2", "(Fraction(-1, 2), Fraction(0, 1), Fraction(0, 1))"),
        ("1/2", "(Fraction(1, 2), Fraction(0, 1), Fraction(0, 1))"),
    ]
    n_wave = len(f.model.get_ck_map())
    assert kc["n_proj"] == 4
    assert np.asarray(kc["matrix_angle"]).shape == (16, 4 * n_wave)


def test_a_half_integer_resonance_decaying_to_two_scalars_is_refused():
    """``1/2 -> 0 0`` has no ``(l, s)``: ``s = 0`` needs ``l = 1/2``.

    Refused BY NAME, at the vertex, instead of surfacing as numpy's
    ``need at least one array to stack`` from ``build_from_groups``.
    """
    with pytest.raises(ValueError, match="no allowed \\(l, s\\) combination"):
        Fitter(IMPOSSIBLE)
    with pytest.raises(ValueError, match="l=1/2"):
        Fitter(IMPOSSIBLE)


@pytest.mark.parametrize("leaf", ["numpy_flat", "cuda_flat_pwa_gen_s0e2"])
def test_the_half_integer_config_computes(leaf):
    """The numbers exist and the CUDA leaf agrees with the reference.

    The tolerance is the family's fp32-cache floor: this config is 1 wave with
    one projection axis of 4, so it is a plain compute on both sides.
    """
    if leaf not in set(backends_for_model("pwa_gen")):
        pytest.skip(f"{leaf} is not registered on this install")
    f = _fit()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    data = fg._in_grid(kc, 64, seed=2)
    a = create_backend("numpy_flat", kc, model="pwa_gen", resolution_size=1)
    b = create_backend(leaf, kc, model="pwa_gen", resolution_size=1)
    try:
        ha, hb = a.load_data(data), b.load_data(data)
        try:
            Qa, ga, Pa = a.compute(params, ha, norm=1.3)
            Qb, gb, Pb = b.compute(params, hb, norm=1.3)
        finally:
            for h in (ha, hb):
                if hasattr(h, "free"):
                    h.free()
    finally:
        a.free()
        b.free()
    assert np.isfinite(Qb)
    assert abs(Qa - Qb) / max(abs(Qa), 1e-300) < 1e-6
    rel = np.max(np.abs(Pa - Pb)) / max(np.max(np.abs(Pa)), 1e-300)
    assert rel < 1e-4
    for key in ga:
        if ga[key] is not None and key in gb:
            d = np.max(np.abs(np.asarray(ga[key]) - np.asarray(gb[key])))
            assert d / max(np.max(np.abs(np.asarray(ga[key]))), 1e-300) < 1e-4


def test_to_spin_spells_the_spins_the_config_uses():
    """`"1/2"`, `0.5` and `Fraction(1, 2)` are the same spin (the reader is one)."""
    assert to_spin("1/2") == to_spin(0.5) == to_spin(Fraction(1, 2))
    assert to_spin(1) == Fraction(1, 1)