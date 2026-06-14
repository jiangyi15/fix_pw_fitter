import sympy
import math

_s = "sin"
_c = "cos"

cache_formula = {
((('pim1', 'pip1'), ('pim2', 'pip2')), ((0, 0), (1, 0), (1, 0))): [
{"coeffs":  1/math.sqrt(3), "k":  [1,1,1],  "b": [_c, _s, _s]},
{"coeffs": -1/math.sqrt(3), "k": [0,1,1], "b":  [_c, _c, _c]},
],
((('pim1', 'pip1'), ('pim2', 'pip2')), ((1, 1), (1, 0), (1, 0))): [
{"coeffs": -1/math.sqrt(2)*1j, "k": [1,1,1], "b":  [_s, _s, _s]},
],
((('pim1', 'pip1'), ('pim2', 'pip2')), ((2, 2), (1, 0), (1, 0))): [
{"coeffs":  1/math.sqrt(6), "k":  [1,1,1],  "b": [_c, _s, _s]},
{"coeffs": 2/math.sqrt(6), "k":  [0,1,1], "b":  [_c, _c, _c]},
],
((('pim1', 'pip1'), ('pim2', 'pip2')), ((1, 1), (0, 0), (1, 0))): [
{"coeffs": -1, "k":  [0,0,1], "b":  [_c, _c, _c]},
],
((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((2, 2), (2, 1), (1, 0))): [
{"coeffs": math.sqrt(6)/4*1j, "k":  [1,2,1], "b":  [_s, _s, _s]},
],
((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((1, 1), (1, 1), (1, 0))): [
{"coeffs": -math.sqrt(2)/2*1j, "k":  [1,1,1], "b":  [_s, _s, _s]},
],
((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((1, 1), (0, 1), (1, 0))): [
{"coeffs":  1/math.sqrt(3), "k":  [1,1,1],  "b": [_c, _s, _s]},
{"coeffs": -1/math.sqrt(3), "k": [0,1,1], "b":  [_c, _c, _c]},
],
((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((1, 1), (2, 1), (1, 0))): [
{"coeffs":  1/math.sqrt(6), "k":  [1,1,1],  "b": [_c, _s, _s]},
{"coeffs": 2/math.sqrt(6), "k":  [0,1,1], "b":  [_c, _c, _c]},
],
((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((1, 1), (1, 0), (0, 0))): [
{"coeffs": -1, "k":  [0,1,0], "b":  [_c, _c, _c]},
],
((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((1, 1), (1, 2), (2, 0))): [
{"coeffs": -3*math.sqrt(10)/20, "k":  [1,1,2], "b":  [_c, _s, _s]},
{"coeffs": math.sqrt(10)/20, "k":  [0,1,0], "b":  [_c, _c, _c]},
{"coeffs": 3*math.sqrt(20)/20, "k":  [0,1,2], "b":  [_c, _c, _c]},
],
((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((1, 1), (3, 2), (2, 0))): [
{"coeffs": -math.sqrt(15)/10, "k":  [1,1,2], "b":  [_c, _s, _s]},
{"coeffs": -math.sqrt(15)/20, "k":  [0,1,0], "b":  [_c, _c, _c]},
{"coeffs": -3*math.sqrt(15)/20, "k":  [0,1,2], "b":  [_c, _c, _c]},
],
((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((2, 2), (0, 2), (2, 0))): [
{"coeffs": 3*math.sqrt(5)/80, "k":  [2,0,0], "b":  [_c, _c, _c]},
{"coeffs": -3*math.sqrt(5)/80, "k":  [2,0,2], "b":  [_c, _c, _c]},
{"coeffs": -3*math.sqrt(5)/80, "k":  [2,2,0], "b":  [_c, _c, _c]},
{"coeffs": 3*math.sqrt(5)/80, "k":  [2,2,2], "b":  [_c, _c, _c]},
{"coeffs": -3*math.sqrt(5)/20, "k":  [1,2,2], "b":  [_c, _s, _s]},
{"coeffs": math.sqrt(5)/80, "k":  [0,0,0], "b":  [_c, _c, _c]},
{"coeffs": 3*math.sqrt(5)/80, "k":  [0,0,2], "b":  [_c, _c, _c]},
{"coeffs": 3*math.sqrt(5)/80, "k":  [0,2,0], "b":  [_c, _c, _c]},
{"coeffs": 9*math.sqrt(5)/80, "k":  [0,2,2], "b":  [_c, _c, _c]},
],
((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((2, 2), (2, 2), (2, 0))): [
{"coeffs": 3*math.sqrt(14)/112, "k":  [2,0,0], "b":  [_c, _c, _c]},
{"coeffs": -3*math.sqrt(14)/112, "k":  [2,0,2], "b":  [_c, _c, _c]},
{"coeffs": -3*math.sqrt(14)/112, "k":  [2,2,0], "b":  [_c, _c, _c]},
{"coeffs": 3*math.sqrt(14)/112, "k":  [2,2,2], "b":  [_c, _c, _c]},
{"coeffs": 3*math.sqrt(14)/56, "k":  [1,2,2], "b":  [_c, _s, _s]},
{"coeffs": -math.sqrt(14)/112, "k":  [0,0,0], "b":  [_c, _c, _c]},
{"coeffs": -3*math.sqrt(14)/112, "k":  [0,2,0], "b":  [_c, _c, _c]},
{"coeffs": -3*math.sqrt(14)/112, "k":  [0,0,2], "b":  [_c, _c, _c]},
{"coeffs": -9*math.sqrt(14)/112, "k":  [0,2,2], "b":  [_c, _c, _c]},
],
((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((2, 2), (4, 2), (2, 0))): [
{"coeffs": 3*math.sqrt(70)/1120, "k":  [2,0,0], "b":  [_c, _c, _c]},
{"coeffs": -3*math.sqrt(70)/1120, "k":  [2,0,2], "b":  [_c, _c, _c]},
{"coeffs": -3*math.sqrt(70)/1120, "k":  [2,2,0], "b":  [_c, _c, _c]},
{"coeffs": 3*math.sqrt(70)/1120, "k":  [2,2,2], "b":  [_c, _c, _c]},
{"coeffs": 3*math.sqrt(70)/70, "k":  [1,2,2], "b":  [_c, _s, _s]},
{"coeffs": 3*math.sqrt(70)/560, "k":  [0,0,0], "b":  [_c, _c, _c]},
{"coeffs": 9*math.sqrt(70)/560, "k":  [0,2,0], "b":  [_c, _c, _c]},
{"coeffs": 9*math.sqrt(70)/560, "k":  [0,0,2], "b":  [_c, _c, _c]},
{"coeffs": 27*math.sqrt(70)/560, "k":  [0,2,2], "b":  [_c, _c, _c]},
],
((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((0, 0), (0, 0), (0, 0))): [
{"coeffs": 1, "k":  [0,0,0], "b":  [_c, _c, _c]},
],
((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((0, 0), (1, 1), (1, 0))): [
{"coeffs": -1, "k":  [0,0,1], "b":  [_c, _c, _c]},
],
((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((2, 2), (2, 0), (0, 0))): [
{"coeffs": 1/4, "k":  [0,0,0], "b":  [_c, _c, _c]},
{"coeffs": 3/4, "k":  [0,2,0], "b":  [_c, _c, _c]},
],
((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((2, 2), (1, 1), (1, 0))): [
    {"coeffs": -3*math.sqrt(10)/20, "k":  [1,2,1], "b":  [_c, _s, _s]},
    {"coeffs": math.sqrt(10)/20, "k":  [0,0,1], "b":  [_c, _c, _c]},
    {"coeffs": 3*math.sqrt(10)/20, "k":  [0,2,1], "b":  [_c, _c, _c]},
],
((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((2, 2), (3, 1), (1, 0))): [
{"coeffs": -math.sqrt(15)/10, "k":  [1,2,1], "b":  [_c, _s, _s]},
{"coeffs": -math.sqrt(15)/20, "k":  [0,0,1], "b":  [_c, _c, _c]},
{"coeffs": -3*math.sqrt(15)/20, "k":  [0,2,1], "b":  [_c, _c, _c]},
]
}

for (k, v) in list(cache_formula.items()):
    if k[0] == (('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')):
        cache_formula[((('pim1', 'pim2', 'pip1'), ('pim1', 'pip1')), k[1])] = v


def get_angle_formula(decaychain, ls):
    topo_id = decaychain.topo_id()
    if (topo_id, ls) in cache_formula:
        return cache_formula[(topo_id, ls)]
    else:
        raise IndexError(f"{(topo_id, ls)}, not found")
