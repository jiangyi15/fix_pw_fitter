"""Print the angular formula for the README config example."""
from interp_fitter.config_builder import parse_physics
from interp_fitter.angular_formula import compute_angular_formula
import sympy as sp

physics = {
    "decay": {
        "A": [["R", "C", {"p_break": True}], ["Y", "D", {"p_break": True}]],
        "R": [["B", "D", {"p_break": True}]],
        "Y": ["B", "C"],
    },
    "particle": {
        "$top": "A", "$finals": ["B", "C", "D"],
        "A": {"J": 0, "P": -1},
        "R": ["R1", "R2"],
        "Y": {"J": 1, "P": -1},
        "R1": {"J": 0, "P": 1}, "R2": {"J": 1, "P": -1},
        "B": {"J": 0, "P": -1}, "C": {"J": 0, "P": -1}, "D": {"J": 0, "P": -1},
    },
}

m = parse_physics(physics)
for ci, dc in enumerate(m.decay_chains):
    names = [f"{d.parent}->{'+'.join(d.children)}" for d in dc.decays]
    print(f"Chain {ci}: {'  '.join(names)}")
    for ls in dc.ls_combinations():
        formula = compute_angular_formula(dc, list(ls))
        print(f"  LS {ls}")
        for ft in formula["fourier_terms"]:
            t = [x for x in ft.factors if x.kind == "theta" and x.k]
            p = [x for x in ft.factors if x.kind == "phi" and x.k]
            trig = " * ".join(
                [f"{x.func}({''if x.k==1 else str(x.k)+'*'}{x.kind}_{x.var_idx}/2)" for x in t] +
                [f"{x.func}({''if x.k==1 else str(x.k)+'*'}{x.kind}_{x.var_idx}/2)" for x in p]
            )
            coeff = sp.nsimplify(ft.coeff)
            line = str(coeff)
            if trig:
                line += "  |  " + trig
            print(f"    {line}")
        print()
