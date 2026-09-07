"""alignment_D_parts: conjugate Wigner-D over three alignment angles.

Checks that the monomial decomposition matches the numerical Wigner-D
and that α=β=γ=0 gives the identity (δ_{m,m′}), i.e. adding the alignment
factor with zero angles reproduces the un-aligned amplitude.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cmath
import math

import numpy as np
import pytest

from ampfit.helicity_angle import (alignment_D_parts, helicity_values,
                                   wigner_d, wigner_D_conj, to_spin)


def _eval_parts(parts, x):
    v = 0.0j
    for coef, (kind, f) in parts:
        t = math.cos(f * x) if kind == "c" else math.sin(f * x)
        v += coef * t
    return v


def _D_mono(j, m, mp, alpha, beta, gamma):
    a, b, g = alignment_D_parts(j, m, mp)
    return _eval_parts(a, alpha) * _eval_parts(b, beta) * _eval_parts(g, gamma)


def _D_num(j, m, mp, alpha, beta, gamma):
    """D^{j*}_{m,mp}(α,β,γ) = e^{+imα} d^j_{m,mp}(β) e^{+i·mp·γ}."""
    return wigner_D_conj(j, m, mp, alpha, beta) * cmath.exp(1j * float(mp) * gamma)


@pytest.mark.parametrize("J", [0, 1, 1.5, 2, 2.5])
def test_zero_angles_identity(J):
    # m range
    ms = helicity_values(J)
    for m in ms:
        for mp in ms:
            val = _D_mono(J, m, mp, 0.0, 0.0, 0.0)
            expect = 1.0 if m == mp else 0.0
            assert abs(val - expect) < 1e-9, (J, m, mp, val)


@pytest.mark.parametrize("J", [1, 1.5, 2])
def test_matches_numeric_at_random_angles(J):
    rng = np.random.RandomState(0)
    ms = helicity_values(J)
    for _ in range(20):
        m = ms[rng.randint(len(ms))]
        mp = ms[rng.randint(len(ms))]
        alpha, beta, gamma = rng.uniform(0, 2 * np.pi, 3)
        mono = _D_mono(J, m, mp, alpha, beta, gamma)
        num = _D_num(J, m, mp, alpha, beta, gamma)
        assert abs(mono - num) < 1e-6, (J, m, mp, alpha, beta, gamma)
