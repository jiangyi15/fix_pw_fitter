"""Helicity amplitude T_{λ,LS}(φ,θ) for decay chains.

Everything is in direct Fourier basis: ``coeff · Π cos(k·var/2) · Π sin(k·var/2)``
where ``coeff`` is a sympy expression (may include ``I`` for imaginary parts).
"""

from __future__ import annotations

import math
import sympy as sp
from dataclasses import dataclass, field
from itertools import product as iproduct


I = sp.I


# ============================================================================
#  Fourier factor
# ============================================================================

@dataclass
class Factor:
    name: str      # e.g. "theta_0", "phi_1"
    func: str      # "cos" or "sin"
    k: int         # multiplier of var/2


@dataclass
class AmpTerm:
    coeff: sp.Expr = sp.Integer(0)
    factors: list[Factor] = field(default_factory=list)


HelicityKey = str
LSKey = str


# ============================================================================
#  Half-angle → Fourier  (binomial expansion of sin^p·cos^q)
# ============================================================================

def _expand_half_angle(sin_pow: int, cos_pow: int):
    """sin(θ/2)^n · cos(θ/2)^m → {("cos"|"sin", k): sp.Rational}."""
    from math import comb
    result: dict[tuple[str, int], sp.Rational] = {}
    denom = 2 ** (cos_pow + sin_pow)
    sm4 = sin_pow % 4
    for k1 in range(cos_pow + 1):
        for k2 in range(sin_pow + 1):
            c = comb(cos_pow, k1) * comb(sin_pow, k2)
            if c == 0: continue
            if k2 % 2 == 1: c = -c
            n = cos_pow + sin_pow - 2 * k1 - 2 * k2
            k = abs(n)
            r_factor = [1, 0, -1, 0][sm4]
            i_factor = [0, 1, 0, -1][sm4]
            if k == 0:
                if r_factor:
                    result[("cos", 0)] = result.get(("cos", 0), sp.Integer(0)) + sp.Rational(c * r_factor, denom)
                continue
            if r_factor:
                result[("cos", k)] = result.get(("cos", k), sp.Integer(0)) + sp.Rational(c * r_factor, denom)
            if i_factor:
                coeff = sp.Rational(c * i_factor, denom)
                if n < 0: coeff = -coeff
                result[("sin", k)] = result.get(("sin", k), sp.Integer(0)) + coeff
    return result


# ============================================================================
#  Wigner-d weights → Fourier basis immediately
# ============================================================================

def _group_like_terms(terms: list[AmpTerm]) -> list[AmpTerm]:
    """Sum coefficients of AmpTerms that share the same factor basis."""
    from collections import defaultdict
    groups: dict[tuple, sp.Expr] = defaultdict(lambda: sp.Integer(0))
    for t in terms:
        key = tuple(sorted((f.name, f.func, f.k) for f in t.factors))
        groups[key] += t.coeff
    return [AmpTerm(coeff=c, factors=[Factor(n, f, k) for n, f, k in k])
            for k, c in groups.items() if c != 0]


def _wd_fourier(J: float, m1: float, m2: float, var_idx: int):
    """Wigner-d^J_{m1,m2}(θ) → list of AmpTerm in cos(kθ/2)/sin(kθ/2) basis."""
    twoJ = round(2 * J)
    jpm1 = round(J + m1); jmm1 = round(J - m1)
    jpm2 = round(J + m2); jmm2 = round(J - m2)
    num_num = math.factorial(jpm1) * math.factorial(jmm1) * math.factorial(jpm2) * math.factorial(jmm2)

    terms = []
    for L in range(twoJ + 1):
        k = (L + m2 - m1) / 2
        if abs(k - round(k)) > 1e-10: continue
        k = round(k)
        if k < max(0, m2 - m1) or k > min(jmm1, jpm2): continue

        sign = 1 if (round(m1 - m2) + k) % 2 == 0 else -1
        denom = 1
        if jmm1 - k >= 0: denom *= math.factorial(jmm1 - k)
        if jpm2 - k >= 0: denom *= math.factorial(jpm2 - k)
        if round(m1 - m2) + k >= 0: denom *= math.factorial(round(m1 - m2) + k)
        if k >= 0: denom *= math.factorial(k)

        p, r = 1, num_num
        i = 2
        while i * i <= r:
            while r % (i * i) == 0:
                r //= (i * i); p *= i
            i += 1
        g = math.gcd(p, denom)
        p //= g; denom //= g
        wd_coeff = sp.Rational(sign * p, denom) * sp.sqrt(r)

        sp_pow, cp_pow = L, twoJ - L
        # expand to Fourier immediately
        for (func, kk), frac in _expand_half_angle(sp_pow, cp_pow).items():
            if frac == 0: continue
            factors = []
            if kk > 0:
                factors.append(Factor(f"theta_{var_idx}", func, kk))
            terms.append(AmpTerm(coeff=wd_coeff * frac, factors=factors))
    return _group_like_terms(terms)


# ============================================================================
#  Vertex amplitude
# ============================================================================

def _vertex_terms(Ja, Jb, Jc, la, lb, lc, L, S, theta_idx, phi_idx):
    """Build AmpTerms for one vertex with given (L,S) and helicities."""
    from sympy.physics.wigner import clebsch_gordan

    delta = lb - lc
    if abs(delta) > Ja + 1e-10:
        return []

    cg1 = clebsch_gordan(Jb, Jc, S, lb, -lc, delta)
    if cg1 == 0:
        return []
    cg2 = clebsch_gordan(float(L), S, Ja, 0.0, delta, delta)
    if cg2 == 0:
        return []
    ls_factor = sp.sqrt(sp.Rational(2 * L + 1, 2 * round(Ja) + 1))
    base = sp.nsimplify(cg1 * cg2 * ls_factor)

    wd_terms = _wd_fourier(Ja, la, delta, theta_idx)

    # φ factors
    phi_terms: list[AmpTerm] = []
    if abs(la) < 1e-10:
        phi_terms.append(AmpTerm(coeff=sp.Integer(1)))
    else:
        abs_la = int(abs(la) * 2)
        # real part: cos(|la|·φ)
        phi_terms.append(AmpTerm(coeff=sp.Integer(1),
            factors=[Factor(f"phi_{phi_idx}", "cos", abs_la)]))
        # imag part: i · sign(la) · sin(|la|·φ)
        sin_sign = -1 if la < 0 else 1
        phi_terms.append(AmpTerm(coeff=I * sin_sign,
            factors=[Factor(f"phi_{phi_idx}", "sin", abs_la)]))

    # Combine Wigner-d × φ factors
    result = []
    for wt in wd_terms:
        for pt in phi_terms:
            result.append(AmpTerm(
                coeff=base * wt.coeff * pt.coeff,
                factors=wt.factors + pt.factors,
            ))
    return _group_like_terms(result)
