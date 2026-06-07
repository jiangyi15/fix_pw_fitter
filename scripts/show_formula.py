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
        "A": {"J": 0, "P": -1}, "R": ["R1", "R2"],
        "Y": {"J": 1, "P": -1}, "R1": {"J": 0, "P": 1}, "R2": {"J": 1, "P": -1},
        "B": {"J": 0, "P": -1}, "C": {"J": 0, "P": -1}, "D": {"J": 0, "P": -1},
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
            th = tuple((x.name, x.func, x.k) for x in ft.factors if x.k)
            groups[(th,)].append(ft.coeff)
        real_parts = []
        imag_parts = []
        for ft in f["fourier_terms"]:
            trigs = []
            for x in ft.factors:
                n = x.k // 2
                if n == 0:
                    trigs.append("1")
                elif x.k % 2 == 0:
                    trigs.append(f"{x.func}({n}·{x.name})" if n > 1 else f"{x.func}({x.name})")
                else:
                    trigs.append(f"{x.func}({x.k}·{x.name}/2)")
            trig = " · ".join(t for t in trigs if t)
            coeff = ft.coeff
            if isinstance(coeff, sp.Float) and abs(float(coeff)) < 1e-12:
                continue
            if ft.im:
                imag_parts.append(f"{coeff}  ×  {trig}" if trig else f"{coeff}")
            else:
                real_parts.append(f"{coeff}  ×  {trig}" if trig else f"{coeff}")
        print(f"  LS = {ls}")
        if real_parts:
            print("    Re:  " + "  +  ".join(real_parts))
        if imag_parts:
            print("    Im:  " + "  +  ".join(imag_parts))
        print()
