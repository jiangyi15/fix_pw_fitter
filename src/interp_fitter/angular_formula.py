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
#  Helpers
# ============================================================================

def _helicities(J: float):
    """Return list of helicity values for spin J (step 1)."""
    vals = []
    h = -J
    while h <= J + 1e-10:
        vals.append(h)
        h += 1.0
    return vals


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
    helicities: list[dict[str, float]] = field(default_factory=list)


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
    """Sum coefficients of AmpTerms that share the same factor basis.

    Helicities from like terms are merged (list concatenation).
    """
    from collections import defaultdict
    groups: dict[tuple, tuple[sp.Expr, list]] = {}
    for t in terms:
        key = tuple(sorted((f.name, f.func, f.k) for f in t.factors))
        if key in groups:
            c, hel = groups[key]
            groups[key] = (c + t.coeff, hel + t.helicities)
        else:
            groups[key] = (t.coeff, list(t.helicities))
    return [AmpTerm(coeff=c, factors=[Factor(n, f, k) for n, f, k in k],
                    helicities=hel)
            for k, (c, hel) in groups.items() if c != 0]


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

    hel_dict = {"la": la, "lb": lb, "lc": lc}
    result = []
    for wt in wd_terms:
        for pt in phi_terms:
            result.append(AmpTerm(
                coeff=base * wt.coeff * pt.coeff,
                factors=wt.factors + pt.factors,
                helicities=[hel_dict],
            ))
    return _group_like_terms(result)


# ============================================================================
#  Cascade combination
# ============================================================================

def compute_amplitude(decay_chain, ls_assignment: list[tuple[int, float]]):
    """Compute T_{λ,LS} for a DecayChain with given (L,S) assignment.

    Returns ``{helicity_key: {ls_key: [AmpTerm, …]}}``.

    Parameters
    ----------
    decay_chain : DecayChain
        From ``config_builder``.
    ls_assignment : list of (int, float)
        One (L, S) per two-body decay, in DFS order.

    Returns
    -------
    dict
        ``{ "λ_B,λ_π,…": { "L₁,S₁;L₂,…": [AmpTerm, …] } }``
    """
    # Get two-body decays in order
    decays = [d for d in decay_chain.decays if len(d.children) == 2]
    if len(decays) != len(ls_assignment):
        raise ValueError(
            f"Expected {len(decays)} (L,S) pairs, got {len(ls_assignment)}")

    # Build per-vertex helicity terms
    vertex_data = []
    for vi, (decay, (Lv, Sv)) in enumerate(zip(decays, ls_assignment)):
        pp, c1p, c2p = decay.parent_particle, decay.child_particles[0], decay.child_particles[1]
        Ja, Jb, Jc = pp.props.get("J", 0), c1p.props.get("J", 0), c2p.props.get("J", 0)
        hel_terms: dict[str, list[AmpTerm]] = {}
        for la in _helicities(Ja):
            for lb in _helicities(Jb):
                for lc in _helicities(Jc):
                    terms = _vertex_terms(Ja, Jb, Jc, la, lb, lc, Lv, Sv, vi, vi)
                    if terms:
                        hel_terms[f"{la},{lb},{lc}"] = terms
        vertex_data.append(hel_terms)

    # Tree structure: v1 ← lb(v0), v2 ← lc(v0), rest linear
    def _parent_of(vi):
        if vi == 1:
            return (0, "lb")
        if vi == 2:
            return (0, "lc")
        if vi > 2:
            return (vi - 1, "lb")
        return None

    def _get_hel(terms, pv, key):
        for t in terms:
            for h in t.helicities:
                if key in h:
                    return h[key]
        return None

    def _cascade(v_idx, prev_terms, prev_hkey):
        if v_idx >= len(vertex_data):
            return {prev_hkey: [("", prev_terms)]}

        result = {}
        parent = _parent_of(v_idx)
        need_hel = _get_hel(prev_terms, parent[0], parent[1]) if parent and prev_terms else None

        for hel_key, terms in vertex_data[v_idx].items():
            la = float(hel_key.split(",")[0])
            if need_hel is not None and abs(la - need_hel) > 1e-10:
                continue

            if prev_terms is None:
                new_terms = terms
                # first vertex: key from la (external)
                new_hkey = hel_key.split(",")[0]
            else:
                new_terms = _group_like_terms([
                    AmpTerm(coeff=pt.coeff * ct.coeff,
                            factors=pt.factors + ct.factors,
                            helicities=pt.helicities + ct.helicities)
                    for pt in prev_terms for ct in terms
                ])
                # add lb,lc of this vertex (external daughters)
                parts = hel_key.split(",")
                new_hkey = prev_hkey + "," + parts[1] + "," + parts[2]

            sub = _cascade(v_idx + 1, new_terms, new_hkey)
            for hk, tl_list in sub.items():
                for lsk, tl in tl_list:
                    result.setdefault(hk, []).append((lsk, tl))
        return result

    raw = _cascade(0, None, "")

    # Sum over intermediate helicities (same external key → combine)
    summed: dict[str, dict[str, list[AmpTerm]]] = {}
    for hk, entries in raw.items():
        for lsk, tl in entries:
            summed.setdefault(hk, {}).setdefault(lsk, []).extend(tl)
    for hk in summed:
        for lsk in summed[hk]:
            summed[hk][lsk] = _group_like_terms(summed[hk][lsk])

    # ── PhiCombine: for J=0 root with ≥3 decays ──
    if len(decays) >= 3:
        root_J = decays[0].parent_particle.props.get("J", 0) if decays[0].parent_particle else None
        if root_J == 0:
            combined = {}
            for hk, ld in summed.items():
                new_ld = {}
                for lsk, terms in ld.items():
                    new_terms = []
                    for t in terms:
                        new_factors = []
                        kill = False
                        for f in t.factors:
                            if f.name == "phi_1":
                                if f.func == "sin":
                                    kill = True
                                    break
                            elif f.name == "phi_2":
                                new_factors.append(Factor("chi", f.func, f.k))
                            else:
                                new_factors.append(f)
                        if not kill:
                            new_terms.append(AmpTerm(coeff=t.coeff, factors=new_factors))
                    if new_terms:
                        new_ld[lsk] = _group_like_terms(new_terms)
                if new_ld:
                    combined[hk] = new_ld
            return combined

    return summed
