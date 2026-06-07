"""Print the angular formula for the README config example."""
from collections import defaultdict
import sympy as sp
from interp_fitter.config_builder import parse_physics
from interp_fitter.angular_formula import compute_angular_formula

physics = {
    "decay": {
        "A": [["R", "C", {"p_break": True}], ["Y", "D", {"p_break": True}]],
        "R": [["B", "D", {"p_break": True}]],
        "Y": ["B", "C"],
    },
    "particle": {
        "$top": "A", "$finals": ["B", "C", "D"],
        "A": {"J": 0}, "R": ["R1", "R2"],
        "Y": {"J": 1}, "R1": {"J": 0}, "R2": {"J": 1},
        "B": {"J": 0}, "C": {"J": 0}, "D": {"J": 0},
    },
}

m = parse_physics(physics)
for ci, dc in enumerate(m.decay_chains):
    names = [f"{d.parent}->{'+'.join(d.children)}" for d in dc.decays]
    print(f"Chain {ci}: {'  '.join(names)}")
    for ls in dc.ls_combinations():
        f = compute_angular_formula(dc, list(ls))
        groups = defaultdict(list)
        for ft in f["fourier_terms"]:
            th = tuple((x.var_idx, x.func, x.k) for x in ft.factors if x.k)
            groups[(th,)].append(ft.coeff)
        parts = []
        for (th,), coeffs in groups.items():
            total = sum(coeffs, sp.Integer(0))
            if total == 0:
                continue
            trigs = []
            for idx, func, kk in th:
                n = kk // 2
                if n == 0:
                    trigs.append("1")
                elif kk % 2 == 0:
                    trigs.append(f"{func}({n}·α_{idx})" if n > 1 else f"{func}(α_{idx})")
                else:
                    trigs.append(f"{func}({kk}·α_{idx}/2)")
            trig = " · ".join(t for t in trigs if t)
            c = sp.nsimplify(total)
            parts.append(f"{c}  ×  {trig}" if trig else f"{c}")
        print(f"  LS = {ls}")
        for p in parts:
            print(f"    {p}")
        print()
