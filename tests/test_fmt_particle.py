"""fmt_particle: charge suffixes must not mangle names ending in _p/_m/_b."""

import pytest

from ampfit.utils import fmt_particle


@pytest.mark.parametrize("name,expected", [
    ("a1(1260)p", r"$a_1(1260)^+$"),
    ("a1(1260)m", r"$a_1(1260)^-$"),
    ("f0(980)", r"$f_0(980)$"),
    ("rhoA", r"$\rho$"),
    ("pip", r"$\pi^{+}$"),
    ("pim", r"$\pi^{-}$"),
    # single-char proton must stay a proton
    ("p", r"$p$"),
    # names ending in _p / _m / _b are NOT charge suffixes (no dangling _)
    ("X_p", r"$X_p$"),
    ("X_m", r"$X_m$"),
    ("X_b", r"$X_b$"),
])
def test_fmt_particle(name, expected):
    assert fmt_particle(name) == expected
    # full=False drops the delimiters
    assert fmt_particle(name, full=False) == expected[1:-1]
