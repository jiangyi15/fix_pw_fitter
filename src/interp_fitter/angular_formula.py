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
    """``cos(k·var/2)`` or ``sin(k·var/2)`` for an angle variable.

    ``name`` identifies the variable (e.g. ``"theta_0"``, ``"phi_1"``).
    """
    name: str      # variable name, e.g. "theta_0", "phi_1"
    func: str      # "cos" or "sin"
    k: int         # multiplier of var/2


@dataclass
class FourierTerm:
    """A single Fourier term: ``coeff · Π Factor``."""
    coeff: 'Any' = 0
    im: bool = False
    factors: list[Factor] = field(default_factory=list)
    helicities: dict[str, float] = field(default_factory=dict)
    # e.g. {"la": 0.5, "lb": 0.0, "lc": -0.5} for the vertex


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

def cg_coeff(j1, m1, j2, m2, J, M):
    """Clebsch-Gordan coefficient ``⟨j1 m1 j2 m2 | J M⟩``.

    Returns an exact sympy expression (no floats).
    All arguments are converted via ``sp.Rational`` for exactness.
    """
    from sympy.physics.wigner import wigner_3j
    import sympy as sp
    def _r(x): return sp.Rational(str(x))
    w3 = wigner_3j(_r(j1), _r(j2), _r(J), _r(m1), _r(m2), _r(-M))
    if w3 == 0:
        return sp.Integer(0)
    phase = (-1) ** _r(int(j1 - j2 + M))
    result = sp.sqrt(_r(int(2 * J + 1))) * phase * w3
    return sp.nsimplify(result)


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
        _sp4.Rational(int(2 * L + 1), int(2 * round(Ja) + 1)))
    base = cg1 * cg2 * ls_factor
    wd = wigner_d_weights(Ja, la, delta)

    import sympy as _sp5
    terms = []
    for wd_c, sp, cp in wd:
        # Expand sin^sp·cos^cp → Fourier basis immediately
        half_exp = expand_half_angle(sp, cp)
        # Each half_exp entry: (("cos"/"sin", k), sympy_Rational)
        for (func, k), frac in half_exp.items():
            if frac == 0:
                continue
            theta_name = f"theta_{theta_idx}"
            theta_factors = [Factor(theta_name, func, k)] if k > 0 else []

            hel = {"la": la, "lb": lb, "lc": lc}
            if abs(la) < 1e-10:
                terms.append(FourierTerm(
                    coeff=base * wd_c * frac, im=False,
                    factors=theta_factors, helicities=hel,
                ))
            else:
                abs_la = int(abs(la) * 2)
                sin_sign = -1 if la < 0 else 1
                terms.append(FourierTerm(
                    coeff=base * wd_c * frac, im=False,
                    factors=theta_factors + [Factor(f"phi_{phi_idx}", "cos", abs_la)],
                    helicities=hel,
                ))
                terms.append(FourierTerm(
                    coeff=base * wd_c * frac * sin_sign, im=True,
                    factors=theta_factors + [Factor(f"phi_{phi_idx}", "sin", abs_la)],
                    helicities=hel,
                ))

    return terms


# ============================================================================
#  Cascade combination  (simplified — helicity sum)
# ============================================================================

def combine_vertices(vertex_terms_list):
    """Multiply amplitudes through the decay cascade.

    Each consecutive pair is matched by ``lb(v_i) == la(v_{i+1})``
    (the first daughter of the parent vertex is the parent of the child vertex).
    """
    if len(vertex_terms_list) == 1:
        return vertex_terms_list[0]

    v0 = vertex_terms_list[0]
    v1 = vertex_terms_list[1]
    combined = []
    for t0 in v0:
        lb0 = t0.helicities.get("lb")
        for t1 in v1:
            la1 = t1.helicities.get("la")
            if lb0 is not None and la1 is not None and abs(lb0 - la1) > 1e-10:
                continue  # helicity mismatch — skip
            coeff = t0.coeff * t1.coeff
            if t0.im and t1.im:
                coeff = -coeff  # i² = -1
            combined.append(FourierTerm(
                coeff=coeff,
                im=t0.im ^ t1.im,
                factors=t0.factors + t1.factors,
                helicities=t0.helicities | t1.helicities,
            ))
    return combined


# ============================================================================
#  Expand to Fourier basis (theta + phi)
# ============================================================================




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

    # Sum like terms (same trig basis) across helicities
    # Separately for real (im=False) and imaginary (im=True) parts
    from collections import defaultdict
    summed: dict[tuple, dict[bool, Any]] = defaultdict(lambda: {False: 0, True: 0})
    import sympy as _sp6
    for ft in combined:
        key = tuple(sorted((f.name, f.func, f.k) for f in ft.factors))
        summed[key][ft.im] = summed[key].get(ft.im, _sp6.Integer(0)) + ft.coeff

    fourier_terms = []
    for key, parts in summed.items():
        for im_flag, coeff in parts.items():
            if coeff == 0:
                continue
            factors = [Factor(name, func, k) for name, func, k in key]
            fourier_terms.append(FourierTerm(coeff=coeff, im=im_flag, factors=factors))

    return {
        "fourier_terms": fourier_terms,
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
