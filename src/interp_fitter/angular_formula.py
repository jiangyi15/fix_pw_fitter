"""
Angular formula computation for decay chains.

Builds the angular amplitude from (L, S) assignments using Wigner-d
half-angle Fourier expansions and CG coefficients.

Following the same structured pipeline as the JS reference:
  power-form (sin^p·cos^q) → half-angle expansion → Fourier basis
"""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import product as iproduct
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
#  Exact rational + sqrt number  (simplified Surd)
# ---------------------------------------------------------------------------

class Surd:
    """``a·sqrt(r) / d``  with integers a, r, d and gcd-reduced."""
    def __init__(self, a: int, r: int = 1, d: int = 1):
        if d < 0:
            a, d = -a, -d
        g = math.gcd(abs(a), d) if hasattr(math, 'gcd') else _gcd(abs(a), d)
        self.a = a // g
        self.r = r
        self.d = d // g

    def __repr__(self):
        return f"Surd({self.a}, {self.r}, {self.d})"

    @staticmethod
    def ONE():
        return Surd(1, 1, 1)

    @staticmethod
    def ZERO():
        return Surd(0, 1, 1)

    def is_zero(self):
        return self.a == 0

    def copy(self):
        return Surd(self.a, self.r, self.d)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Surd(self.a * other, self.r, self.d)
        return Surd(self.a * other.a, self.r * other.r, self.d * other.d)

    def __neg__(self):
        return Surd(-self.a, self.r, self.d)

    def __add__(self, other):
        if self.r != other.r:
            raise ValueError("Different radicands")
        return Surd(self.a * other.d + other.a * self.d, self.r, self.d * other.d)

    def __eq__(self, other):
        return self.a == other.a and self.r == other.r and self.d == other.d


# ---------------------------------------------------------------------------
#  Structured term
# ---------------------------------------------------------------------------

@dataclass
class Term:
    """A single angular term in power form.

    ``coeff · sin(θ/2)^sin_pow · cos(θ/2)^cos_pow · trig(k·φ)``
    where trig is ``cos`` or ``sin``.
    """
    coeff: Surd
    im: bool = False                  # imaginary amplitude flag
    theta_pairs: list[tuple[int, int, int]] = field(default_factory=list)
    #   (var_idx, sin_pow, cos_pow)  for each theta variable
    phi_pairs: list[tuple[int, str, int]] = field(default_factory=list)
    #   (var_idx, func, k)  func ∈ {"cos", "sin"}


# ---------------------------------------------------------------------------
#  Half-angle expansion  (sin^p·cos^q → cos(kθ) / sin(kθ))
# ---------------------------------------------------------------------------

def _binomial(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)

def expand_half_angle(sp: int, cp: int):
    """Expand sin(θ/2)^sp · cos(θ/2)^cp into ``{cos(kθ): coeff, sin(kθ): coeff}``.

    Uses Euler formulas:
      sin(θ/2) = (e^{iθ/2} - e^{-iθ/2}) / 2i
      cos(θ/2) = (e^{iθ/2} + e^{-iθ/2}) / 2

    Returns dict mapping ``("cos", k)`` or ``("sin", k)`` to ``Surd`` coefficient.
    """
    if sp == 0 and cp == 0:
        return {("cos", 0): Surd.ONE()}

    # Build (e^{iθ/2} + e^{-iθ/2})^cp · (e^{iθ/2} - e^{-iθ/2})^sp / (2^cp · (2i)^sp)
    from math import comb
    result: dict[tuple[str, int], Surd] = {}

    for k1 in range(cp + 1):
        for k2 in range(sp + 1):
            exp = (cp - 2 * k1) + (sp - 2 * k2)  # exponent of e^{iθ/2}
            coeff = comb(cp, k1) * comb(sp, k2)
            if coeff == 0:
                continue
            if k2 % 2 == 1:
                coeff = -coeff  # (-1)^k2 from (e^{i·θ/2} - e^{-i·θ/2}) expansion

            # Convert e^{i·exp·θ/2} = cos(exp·θ/2) + i·sin(exp·θ/2)
            # We need cos(kθ) / sin(kθ) where k = |exp|
            k = abs(exp)
            if k == 0:
                key = ("cos", 0)
                result[key] = result.get(key, Surd.ZERO()) + Surd(coeff, 1, 1)
            elif exp > 0:
                # cos(k·θ/2) + i·sin(k·θ/2)
                # cos(k·θ/2)
                result[("cos", k)] = result.get(("cos", k), Surd.ZERO()) + Surd(coeff, 1, 1)
                # sin(k·θ/2)  (imaginary part)
                result[("sin", k)] = result.get(("sin", k), Surd.ZERO()) + Surd(coeff, 1, 1)
            else:
                # cos(k·θ/2) - i·sin(k·θ/2)
                result[("cos", k)] = result.get(("cos", k), Surd.ZERO()) + Surd(coeff, 1, 1)
                result[("sin", k)] = result.get(("sin", k), Surd.ZERO()) + Surd(-coeff, 1, 1)

    # Divide by 2^cp · (2i)^sp
    denom = (2 ** cp) * ((2j) ** sp)  # complex denominator
    # For the real amplitude, we need to handle this properly
    # (2i)^sp = 2^sp · i^sp
    # i^sp = 1 if sp%4==0, i if sp%4==1, -1 if sp%4==2, -i if sp%4==3

    # Actually, let's use the real expansion directly.
    # The term is sin(θ/2)^sp · cos(θ/2)^cp
    # We expand it as a sum of cos(kθ) and sin(kθ) terms
    # Using Chebyshev polynomials: sin^sp·cos^cp → Σ a_k·cos(kθ) + b_k·sin(kθ)

    return _expand_sin_cos_half(sp, cp)


def _expand_sin_cos_half(sp: int, cp: int):
    """Direct real expansion of sin(θ/2)^sp · cos(θ/2)^cp."""
    from math import comb
    result: dict[tuple[str, int], Surd] = {}

    # Use: sin(θ/2) = (e^{iθ/2} - e^{-iθ/2})/(2i)
    #       cos(θ/2) = (e^{iθ/2} + e^{-iθ/2})/2
    # Product: (e^{iθ/2} + e^{-iθ/2})^cp · (e^{iθ/2} - e^{-iθ/2})^sp / (2^cp · (2i)^sp)
    #
    # Expand numerator:
    # Σ_{k1=0}^{cp} Σ_{k2=0}^{sp} C(cp,k1)·C(sp,k2)·(-1)^{k2}·e^{i·(cp-2k1+sp-2k2)·θ/2}
    #
    # Let n = cp + sp - 2k1 - 2k2  (exponent of e^{iθ/2})
    # e^{i·n·θ/2} = cos(n·θ/2) + i·sin(n·θ/2)
    #
    # Now separate into real/imag parts.

    total_den = Fraction(1, (2 ** cp) * (2 ** sp) * (1 if sp % 2 == 0 else 2))
    # Actually the 2i factor... Let me simplify differently.
    # (2i)^sp = 2^sp * i^sp
    # i^0=1, i^1=i, i^2=-1, i^3=-i
    # So denominator = 2^cp * 2^sp * i^sp = 2^{cp+sp} * i^sp

    pow2 = 2 ** (cp + sp)

    for k1 in range(cp + 1):
        for k2 in range(sp + 1):
            c = comb(cp, k1) * comb(sp, k2)
            if c == 0:
                continue
            if k2 % 2 == 1:
                c = -c  # (-1)^k2

            n = cp + sp - 2 * k1 - 2 * k2  # exponent of e^{iθ/2}

            # e^{i·n·θ/2} = cos(n·θ/2) + i·sin(n·θ/2)
            # Divide by 2^{cp+sp}·i^{sp}
            # i^{sp} factor:
            #   i^0=1, i^1=i, i^2=-1, i^3=-i, i^4=1, ...
            sp_mod = sp % 4
            # cos term gets divided by i^{sp}
            # sin term gets divided by i^{sp-1} (since sin comes from i*sin part)

            k = abs(n)  # Fourier mode
            if k == 0:
                coeff = Surd(c, 1, pow2)
                if sp_mod == 2:
                    coeff = -coeff  # i^2 = -1
                elif sp_mod in (1, 3):
                    continue  # purely imaginary → no real part
                s = result.get(("cos", 0), Surd.ZERO())
                try:
                    s = s + coeff
                except:
                    pass
                result[("cos", 0)] = s
                continue

            # Real part: cos(k·θ/2) or cos(n·θ/2)
            # After dividing by i^{sp}
            sign_real = 1
            if sp_mod == 0:
                sign_real = 1
            elif sp_mod == 2:
                sign_real = -1
            else:
                continue  # cos term is purely imaginary after division → no real contribution

            if n > 0:
                key = ("cos", k)
            else:
                key = ("cos", k)  # cos(n·θ/2) = cos(|n|·θ/2) (even)

            coeff = Surd(c * sign_real, 1, pow2)
            result[key] = result.get(key, Surd.ZERO()) + coeff

            # Imag part: sin(k·θ/2)
            sign_imag = 1
            if sp_mod == 0:
                sign_imag = 1  # i·sin · 1 = i·sin
            elif sp_mod == 2:
                sign_imag = -1  # i·sin · (-1) = -i·sin
            elif sp_mod == 1:
                sign_imag = 1  # i·sin · i = -sin → but this is imaginary... hmm
                continue
            elif sp_mod == 3:
                continue

            # For the real-valued amplitude, the imaginary part from e^{iθ/2}
            # becomes a sine term: sin(n·θ/2) = sign(n)·sin(|n|·θ/2)
            if n > 0:
                key = ("sin", k)
            else:
                # sin(n·θ/2) = -sin(|n|·θ/2)
                key = ("sin", k)

            coeff = Surd(c * sign_imag, 1, pow2)
            if n < 0:
                coeff = -coeff
            result[key] = result.get(key, Surd.ZERO()) + coeff

    return result


# ---------------------------------------------------------------------------
#  Product-to-sum for phi
# ---------------------------------------------------------------------------

def phi_product_to_sum(func1: str, k1: int, func2: str, k2: int):
    """Combine two phi factors of the same variable using product-to-sum.

    Returns list of ``(coeff_numer, func, k)``.
    """
    if (func1, k1) == ("cos", 0) or func1 == "1":
        return [("1", func2, k2)]
    if (func2, k2) == ("cos", 0) or func2 == "1":
        return [("1", func1, k1)]
    if func1 == "cos" and func2 == "cos":
        return [("1/2", "cos", k1 + k2), ("1/2", "cos", abs(k1 - k2))]
    if func1 == "sin" and func2 == "sin":
        return [("1/2", "cos", abs(k1 - k2)), ("-1/2", "cos", k1 + k2)]
    if func1 == "cos" and func2 == "sin":
        return [("1/2", "sin", k1 + k2), ("-1/2" if k1 >= k2 else "1/2", "sin", abs(k1 - k2))]
    if func1 == "sin" and func2 == "cos":
        return [("1/2", "sin", k1 + k2), ("1/2" if k1 >= k2 else "-1/2", "sin", abs(k1 - k2))]
    return []


# ---------------------------------------------------------------------------
#  Wigner-d half-angle weights
# ---------------------------------------------------------------------------

def wigner_d_weights(J: float, m1: float, m2: float):
    """Return list of ``(coeff_Surd, sin_pow, cos_pow)`` for d^J_{m1,m2}(θ)."""
    twoJ = round(2 * J)
    jpm1 = round(J + m1)
    jmm1 = round(J - m1)
    jpm2 = round(J + m2)
    jmm2 = round(J - m2)

    # Numerator under sqrt: (j+m1)!(j-m1)!(j+m2)!(j-m2)!
    from math import factorial
    num_num = factorial(jpm1) * factorial(jmm1) * factorial(jpm2) * factorial(jmm2)

    weights = []
    for L in range(twoJ + 1):
        k = (L + m2 - m1) / 2
        if abs(k - round(k)) > 1e-10:
            continue
        k = round(k)
        if k < max(0, m2 - m1):
            continue
        if k > min(jmm1, jpm2):
            continue

        sign = 1 if (round(m1 - m2) + k) % 2 == 0 else -1

        denom = 1
        if jmm1 - k >= 0:
            denom *= factorial(jmm1 - k)
        if jpm2 - k >= 0:
            denom *= factorial(jpm2 - k)
        if round(m1 - m2) + k >= 0:
            denom *= factorial(round(m1 - m2) + k)
        if k >= 0:
            denom *= factorial(k)

        # Extract perfect squares
        p, r = 1, num_num
        i = 2
        while i * i <= r:
            while r % (i * i) == 0:
                r //= (i * i)
                p *= i
            i += 1

        g = math.gcd(p, denom) if hasattr(math, 'gcd') else _gcd(p, denom)
        p //= g
        denom //= g

        weights.append((Surd(sign * p, r, denom), L, twoJ - L))

    return weights


# ---------------------------------------------------------------------------
#  AngularFormula — builds the full angular expression for a DecayChain
# ---------------------------------------------------------------------------

@dataclass
class AngularFormula:
    """Angular amplitude for a decay chain with specific (L,S) assignments.

    Stores terms in Fourier basis: ``coeff · cos(k·θ/2) · sin(k'·θ/2) · cos(l·φ)``
    """
    terms: list[dict] = field(default_factory=list)
    # Each term: {coeff_Surd, im, factors: [(var_name, func, k), ...]}


# ============================================================================
#  CHEBYSHEV-BASED HALF-ANGLE EXPANSION (cleaner)
# ============================================================================

def _cheb_expand(sp: int, cp: int):
    """Expand sin(θ/2)^sp · cos(θ/2)^cp into cos(k·θ/2) / sin(k·θ/2).

    Returns list of ``((func, k), Fraction)``.
    k is the multiplier of θ/2: term = ``func(k·θ/2)``.
    """
    from math import comb
    result: dict[tuple[str, int], Fraction] = {}
    denom = 2 ** (cp + sp)
    sp_mod4 = sp % 4

    for k1 in range(cp + 1):
        for k2 in range(sp + 1):
            c = comb(cp, k1) * comb(sp, k2)
            if c == 0:
                continue
            if k2 % 2 == 1:
                c = -c

            n = cp + sp - 2 * k1 - 2 * k2
            k = abs(n)

            r_factor = [1, 0, -1, 0][sp_mod4]
            i_factor = [0, 1, 0, -1][sp_mod4]

            if k == 0:
                if r_factor != 0:
                    result[("cos", 0)] = result.get(("cos", 0), Fraction(0, 1)) + Fraction(c * r_factor, denom)
                continue

            if r_factor != 0:
                result[("cos", k)] = result.get(("cos", k), Fraction(0, 1)) + Fraction(c * r_factor, denom)

            if i_factor != 0:
                coeff = Fraction(c * i_factor, denom)
                if n < 0:
                    coeff = -coeff
                result[("sin", k)] = result.get(("sin", k), Fraction(0, 1)) + coeff

    return list(result.items())


def expand_half_angle_cheb(sp: int, cp: int):
    """Clean expansion of sin(θ/2)^sp · cos(θ/2)^cp into cos(kθ)/sin(kθ)."""
    return _cheb_expand(sp, cp)


# ---------------------------------------------------------------------------
#  Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # test half-angle expansion
    print("sin(θ/2)^2:")
    for (func, k), (_, num, den) in expand_half_angle_cheb(2, 0):
        coeff = Fraction(num, den)
        print(f"  {coeff} * {func}({k}θ)")

    print("\ncos(θ/2)^2:")
    for (func, k), coeff in expand_half_angle_cheb(0, 2):
        print(f"  {coeff} * {func}({k}θ)")

    print("\nsin(θ/2)^1·cos(θ/2)^1:")
    for (func, k), coeff in expand_half_angle_cheb(1, 1):
        print(f"  {coeff} * {func}({k}θ)")
