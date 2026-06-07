"""
Angular formula computation for decay chains.

Pipeline: DecayChain + (L,S) assignments → vertex amplitudes →
cascade combination → half-angle Fourier expansion → final formula.

Uses exact rational arithmetic throughout (Fractions + factorials).
"""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import product as iproduct
from dataclasses import dataclass, field
from typing import Any


# ============================================================================
#  Structured factor types
# ============================================================================

@dataclass
class Factor:
    """``cos(k·var/2)`` or ``sin(k·var/2)`` for a given vertex variable."""
    var_idx: int
    kind: str      # "theta" or "phi"
    func: str      # "cos" or "sin"
    k: int         # multiplier of var/2


@dataclass
class FourierTerm:
    """A single Fourier term or power-form intermediate.

    In **power form** (before expansion):
        ``theta_power`` contains ``(var_idx, sin_pow, cos_pow)`` tuples.
    In **Fourier form** (after :func:`expand_to_fourier`):
        ``factors`` contains :class:`Factor` objects, at most one per variable.
    """
    coeff: 'Any' = 0  # sympy expression (exact) or float
    im: bool = False
    factors: list[Factor] = field(default_factory=list)
    theta_power: list[tuple[int, int, int]] = field(default_factory=list)
    #  (var_idx, sin_pow, cos_pow)  — used during cascade combine


# ============================================================================
#  Exact rational — just use Fraction for simplicity
#  (sqrt coefficients come from CG / Wigner-d factorials)
# ============================================================================

def _fact(n: int) -> int:
    return math.factorial(n)

def _comb(n: int, k: int) -> int:
    return math.comb(n, k)


# ============================================================================
#  CG coefficient:  <j1 m1 j2 m2 | JM>
#  Using Racah formula with exact integer arithmetic
# ============================================================================

def cg_coeff(j1: float, m1: float, j2: float, m2: float,
             J: float, M: float):
    """Clebsch-Gordan coefficient ``⟨j1 m1 j2 m2 | J M⟩``.

    Returns a ``sympy`` expression (exact symbolic).
    Raises ``ImportError`` if sympy is not installed.
    """
    from sympy.physics.wigner import wigner_3j
    import sympy as sp
    w3 = wigner_3j(sp.Rational(j1), sp.Rational(j2), sp.Rational(J),
                   sp.Rational(m1), sp.Rational(m2), sp.Rational(-M))
    phase = (-1) ** (sp.Rational(j1 - j2 + M))
    return sp.sqrt(2 * J + 1) * phase * w3


# ============================================================================
#  Wigner-d half-angle weights  (exact, from factorial ratios)
# ============================================================================

def wigner_d_weights(J: float, m1: float, m2: float):
    """Return ``[(coeff_float, sin_pow, cos_pow), ...]`` for d^J_{m1,m2}(θ)."""
    twoJ = round(2 * J)
    jpm1 = round(J + m1); jmm1 = round(J - m1)
    jpm2 = round(J + m2); jmm2 = round(J - m2)

    num_num = _fact(jpm1) * _fact(jmm1) * _fact(jpm2) * _fact(jmm2)

    weights = []
    for L in range(twoJ + 1):
        k = (L + m2 - m1) / 2
        if abs(k - round(k)) > 1e-10:
            continue
        k = round(k)
        if k < max(0, m2 - m1) or k > min(jmm1, jpm2):
            continue

        sign = 1 if (round(m1 - m2) + k) % 2 == 0 else -1
        denom = 1
        if jmm1 - k >= 0: denom *= _fact(jmm1 - k)
        if jpm2 - k >= 0: denom *= _fact(jpm2 - k)
        if round(m1 - m2) + k >= 0: denom *= _fact(round(m1 - m2) + k)
        if k >= 0: denom *= _fact(k)

        p, r = 1, num_num
        i = 2
        while i * i <= r:
            while r % (i * i) == 0:
                r //= (i * i)
                p *= i
            i += 1

        g = math.gcd(p, denom)
        p //= g; denom //= g

        import sympy as _sp3
        coeff = _sp3.Rational(sign * p, denom) * _sp3.sqrt(r)
        weights.append((coeff, L, twoJ - L))

    return weights


# ============================================================================
#  Half-angle Fourier expansion
# ============================================================================

def expand_half_angle(sin_pow: int, cos_pow: int):
    """Expand sin(θ/2)^n · cos(θ/2)^m into ``{("cos"/"sin", k): sympy Rational}``."""
    import sympy as _sp
    result: dict[tuple[str, int], _sp.Rational] = {}
    denom = 2 ** (cos_pow + sin_pow)
    sm4 = sin_pow % 4

    for k1 in range(cos_pow + 1):
        for k2 in range(sin_pow + 1):
            c = _comb(cos_pow, k1) * _comb(sin_pow, k2)
            if c == 0: continue
            if k2 % 2 == 1: c = -c

            n = cos_pow + sin_pow - 2 * k1 - 2 * k2
            k = abs(n)
            r_factor = [1, 0, -1, 0][sm4]
            i_factor = [0, 1, 0, -1][sm4]

            if k == 0:
                if r_factor != 0:
                    result[("cos", 0)] = result.get(("cos", 0), _sp.Integer(0)) + _sp.Rational(c * r_factor, denom)
                continue

            if r_factor != 0:
                result[("cos", k)] = result.get(("cos", k), _sp.Integer(0)) + _sp.Rational(c * r_factor, denom)

            if i_factor != 0:
                coeff = _sp.Rational(c * i_factor, denom)
                if n < 0:
                    coeff = -coeff
                result[("sin", k)] = result.get(("sin", k), _sp.Integer(0)) + coeff

    return result


# ============================================================================
#  Vertex amplitude from (L, S)
# ============================================================================

def vertex_amplitude(Ja: float, Jb: float, Jc: float,
                     la: float, lb: float, lc: float,
                     L: int, S: float, theta_idx: int, phi_idx: int):
    """Build angular terms for one decay vertex with given (L,S).

    Returns list of ``(coeff_Fraction, sqrt_r, im, theta_terms, phi_terms)``
    where:
      - theta_terms: list of (theta_idx, sin_pow, cos_pow)
      - phi_terms:   list of (phi_idx, func, k)  func ∈ {"cos", "sin"}
    """
    delta = lb - lc
    if abs(delta) > Ja + 1e-10:
        return []

    cg1 = cg_coeff(Jb, lb, Jc, -lc, S, delta)
    if abs(cg1) < 1e-15:
        return []

    cg2 = cg_coeff(float(L), 0.0, S, delta, Ja, delta)
    if abs(cg2) < 1e-15:
        return []

    import sympy as _sp4
    ls_factor = _sp4.sqrt(
        _sp4.Rational(2 * L + 1, 2 * round(Ja) + 1))
    base = cg1 * cg2 * ls_factor
    wd = wigner_d_weights(Ja, la, delta)

    terms = []
    for wd_c, sp, cp in wd:
        theta_terms = [(theta_idx, sp, cp)]
        phi_terms = []
        if abs(la) < 1e-10:
            phi_terms = [(phi_idx, "cos", 0)]
        else:
            abs_la = int(abs(la) * 2)
            phi_terms = [(phi_idx, "cos", abs_la)]

        terms.append(FourierTerm(
            coeff=base * wd_c, im=False,
            theta_power=theta_terms,
            factors=[Factor(idx, "phi", f, k) for idx, f, k in phi_terms],
        ))

    return terms


# ============================================================================
#  Cascade combination  (simplified — helicity sum)
# ============================================================================

def combine_vertices(vertex_terms_list):
    """Multiply amplitudes through the decay cascade."""
    if len(vertex_terms_list) == 1:
        return vertex_terms_list[0]

    v0 = vertex_terms_list[0]
    v1 = vertex_terms_list[1]
    combined = []
    for t0 in v0:
        for t1 in v1:
            combined.append(FourierTerm(
                coeff=t0.coeff * t1.coeff,
                im=t0.im ^ t1.im,
                theta_power=t0.theta_power + t1.theta_power,
                factors=t0.factors + t1.factors,
            ))
    return combined


# ============================================================================
#  Expand to Fourier basis (theta + phi)
# ============================================================================

def expand_to_fourier(terms: list[FourierTerm]) -> list[FourierTerm]:
    """Convert power-form ``FourierTerm`` objects to Fourier basis.

    Each input term has theta in ``(idx, sp, cp)`` power form.
    Output terms have ``factors`` as :class:`Factor` objects, with one factor
    per variable at most (product-to-sum applied).
    """
    result: list[FourierTerm] = []

    for term in terms:
        import sympy as _sp2
        expansions: list[tuple[list[Factor], _sp2.Expr]] \
            = [([], _sp2.Integer(1))]

        # ── Expand theta power factors ──
        for var_idx, sp, cp in term.theta_power:
            half_exp = expand_half_angle(sp, cp)
            new_exp = []
            for factors, c in expansions:
                for (func, k), frac in half_exp.items():
                    if frac == 0:
                        continue
                    new_factors = factors + [Factor(var_idx, "theta", func, k)]
                    new_exp.append((new_factors, c * _sp2.Rational(frac.numerator, frac.denominator)))
            expansions = new_exp

        # ── Phi product-to-sum ──
        phi_by_idx: dict[int, list[Factor]] = {}
        for f in term.factors:
            phi_by_idx.setdefault(f.var_idx, []).append(f)

        for idx, phis in phi_by_idx.items():
            products: list[tuple[str, int, _sp2.Expr]] = \
                [(phis[0].func, phis[0].k, _sp2.Integer(1))]
            for pf in phis[1:]:
                new_prods = []
                for func1, k1, c1 in products:
                    for res in _phi_product(func1, k1, pf.func, pf.k):
                        pfunc, pk, frac_str = res
                        new_prods.append((pfunc, pk, c1 * _sp2.Rational(frac_str)))
                products = new_prods
            new_exp = []
            for factors, c in expansions:
                for pfunc, pk, pc in products:
                    if pfunc != "1":
                        new_factors = factors + [Factor(idx, "phi", pfunc, pk)]
                    else:
                        new_factors = factors
                    new_exp.append((new_factors, c * pc))
            expansions = new_exp

        # ── Build output FourierTerms ──
        for factors, c in expansions:
            if c == 0:
                continue
            result.append(FourierTerm(
                coeff=term.coeff * c,
                im=term.im,
                factors=factors,
            ))

    return result


def _phi_product(f1: str, k1: int, f2: str, k2: int):
    """Product-to-sum for two phi factors of the same variable."""
    if f1 == "1": return [("1", 0, "1")]
    if f2 == "1": return []
    if f1 == "cos" and f2 == "cos":
        return [("cos", k1 + k2, "1/2"), ("cos", abs(k1 - k2), "1/2")]
    if f1 == "sin" and f2 == "sin":
        return [("cos", abs(k1 - k2), "1/2"), ("cos", k1 + k2, "-1/2")]
    if f1 == "cos" and f2 == "sin":
        return [("sin", k1 + k2, "1/2"), ("sin", abs(k1 - k2), "-1/2" if k1 >= k2 else "1/2")]
    if f1 == "sin" and f2 == "cos":
        return [("sin", k1 + k2, "1/2"), ("sin", abs(k1 - k2), "1/2" if k1 >= k2 else "-1/2")]
    return []


# ============================================================================
#  Main API: compute angular formula for a DecayChain
# ============================================================================

def compute_angular_formula(decay_chain, ls_assignment: list[tuple[int, float]]):
    """Compute the angular formula for a DecayChain with (L,S) assignments.

    Parameters
    ----------
    decay_chain : DecayChain
        The decay chain (from config_builder).
    ls_assignment : list of (int, float)
        One (L, S) pair per two-body decay in the chain, in order.

    Returns
    -------
    dict with keys:
      - ``fourier_terms``: list of expanded Fourier terms
      - ``n_theta``: number of theta variables
      - ``n_phi``: number of phi variables
      - ``source``: "cg_fourier"
    """
    # Get only two-body decays
    decays = [d for d in decay_chain.decays if len(d.children) == 2]

    if len(decays) != len(ls_assignment):
        raise ValueError(
            f"Expected {len(decays)} (L,S) pairs, got {len(ls_assignment)}"
        )

    all_terms = []
    for vi, (decay, (L, S)) in enumerate(zip(decays, ls_assignment)):
        pp = decay.parent_particle
        c1p = decay.child_particles[0]
        c2p = decay.child_particles[1]

        Ja = pp.props.get("J", 0)
        Jb = c1p.props.get("J", 0)
        Jc = c2p.props.get("J", 0)

        # Helicity values: -J to +J in steps of 1
        helicities = {}
        for name, J in [("a", Ja), ("b", Jb), ("c", Jc)]:
            h = []
            h_val = -J
            while h_val <= J + 1e-10:
                h.append(h_val)
                h_val += 1.0
            helicities[name] = h

        # Sum over helicities
        vertex_terms = []
        for la in helicities["a"]:
            for lb in helicities["b"]:
                for lc in helicities["c"]:
                    terms = vertex_amplitude(Ja, Jb, Jc, la, lb, lc, L, S, vi, vi)
                    vertex_terms.extend(terms)

        all_terms.append(vertex_terms)

    # Combine cascadingly
    combined = combine_vertices(all_terms)

    # Expand to Fourier basis
    fourier = expand_to_fourier(combined)

    return {
        "fourier_terms": fourier,
        "n_theta": len(decays),
        "n_phi": len(decays),
    }


# ============================================================================
#  Quick test
# ============================================================================

if __name__ == "__main__":
    from fractions import Fraction
    print("=== Half-angle expansions ===")
    for sp in range(3):
        for cp in range(3):
            r = expand_half_angle(sp, cp)
            parts = [f"{coeff}*{func}({k}·θ/2)" for (func,k), coeff in r.items() if coeff != 0]
            print(f"  sin^{sp}·cos^{cp} = {' + '.join(parts) if parts else '1'}")

    print("\n=== Wigner-d weights for J=1/2, m1=+1/2, m2=+1/2 ===")
    for coeff, r, sp, cp in wigner_d_weights(0.5, 0.5, 0.5):
        print(f"  {coeff}·√{r} · sin^{sp}·cos^{cp}")

    print("\n=== Simple vertex: J=0 → 0+0, L=0, S=0 ===")
    terms = vertex_amplitude(0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0)
    for t in terms:
        print(f"  coeff={t[0]}, sqrt_r={t[1]}, theta={t[3]}, phi={t[4]}")
