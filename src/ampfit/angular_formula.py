import math

_s = "sin"
_c = "cos"

cache_formula = {
    ((('pim1', 'pip1'), ('pim2', 'pip2')), ((0,0), (0,0), (0,0))): [
        {"coeffs": 1, "k": [0, 0, 0], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim2', 'pip2')), ((1,1), (0,0), (1,0))): [
        {"coeffs": -1, "k": [0, 0, 1], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim2', 'pip2')), ((2,2), (0,0), (2,0))): [
        {"coeffs": 1/4, "k": [0, 0, 0], "b": [_c, _c, _c]},
        {"coeffs": 3/4, "k": [0, 0, 2], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim2', 'pip2')), ((1,1), (1,0), (0,0))): [
        {"coeffs": -1, "k": [0, 1, 0], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim2', 'pip2')), ((0,0), (1,0), (1,0))): [
        {"coeffs": 1/3*math.sqrt(3), "k": [1, 1, 1], "b": [_c, _s, _s]},
        {"coeffs": -1/3*math.sqrt(3), "k": [0, 1, 1], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim2', 'pip2')), ((1,1), (1,0), (1,0))): [
        {"coeffs": (-1/2*math.sqrt(2))*1j, "k": [1, 1, 1], "b": [_s, _s, _s]},
    ],
    ((('pim1', 'pip1'), ('pim2', 'pip2')), ((2,2), (1,0), (1,0))): [
        {"coeffs": 1/6*math.sqrt(6), "k": [1, 1, 1], "b": [_c, _s, _s]},
        {"coeffs": 1/3*math.sqrt(6), "k": [0, 1, 1], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim2', 'pip2')), ((1,1), (1,0), (2,0))): [
        {"coeffs": -3/20*math.sqrt(10), "k": [1, 1, 2], "b": [_c, _s, _s]},
        {"coeffs": 1/20*math.sqrt(10), "k": [0, 1, 0], "b": [_c, _c, _c]},
        {"coeffs": 3/20*math.sqrt(10), "k": [0, 1, 2], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim2', 'pip2')), ((2,2), (1,0), (2,0))): [
        {"coeffs": (1/4*math.sqrt(6))*1j, "k": [1, 1, 2], "b": [_s, _s, _s]},
    ],
    ((('pim1', 'pip1'), ('pim2', 'pip2')), ((3,3), (1,0), (2,0))): [
        {"coeffs": -1/10*math.sqrt(15), "k": [1, 1, 2], "b": [_c, _s, _s]},
        {"coeffs": -1/20*math.sqrt(15), "k": [0, 1, 0], "b": [_c, _c, _c]},
        {"coeffs": -3/20*math.sqrt(15), "k": [0, 1, 2], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim2', 'pip2')), ((2,2), (2,0), (0,0))): [
        {"coeffs": 1/4, "k": [0, 0, 0], "b": [_c, _c, _c]},
        {"coeffs": 3/4, "k": [0, 2, 0], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim2', 'pip2')), ((1,1), (2,0), (1,0))): [
        {"coeffs": -3/20*math.sqrt(10), "k": [1, 2, 1], "b": [_c, _s, _s]},
        {"coeffs": 1/20*math.sqrt(10), "k": [0, 0, 1], "b": [_c, _c, _c]},
        {"coeffs": 3/20*math.sqrt(10), "k": [0, 2, 1], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim2', 'pip2')), ((2,2), (2,0), (1,0))): [
        {"coeffs": (1/4*math.sqrt(6))*1j, "k": [1, 2, 1], "b": [_s, _s, _s]},
    ],
    ((('pim1', 'pip1'), ('pim2', 'pip2')), ((3,3), (2,0), (1,0))): [
        {"coeffs": -1/10*math.sqrt(15), "k": [1, 2, 1], "b": [_c, _s, _s]},
        {"coeffs": -1/20*math.sqrt(15), "k": [0, 0, 1], "b": [_c, _c, _c]},
        {"coeffs": -3/20*math.sqrt(15), "k": [0, 2, 1], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim2', 'pip2')), ((0,0), (2,0), (2,0))): [
        {"coeffs": 3/80*math.sqrt(5), "k": [2, 0, 0], "b": [_c, _c, _c]},
        {"coeffs": -3/80*math.sqrt(5), "k": [2, 0, 2], "b": [_c, _c, _c]},
        {"coeffs": -3/80*math.sqrt(5), "k": [2, 2, 0], "b": [_c, _c, _c]},
        {"coeffs": 3/80*math.sqrt(5), "k": [2, 2, 2], "b": [_c, _c, _c]},
        {"coeffs": -3/20*math.sqrt(5), "k": [1, 2, 2], "b": [_c, _s, _s]},
        {"coeffs": 1/80*math.sqrt(5), "k": [0, 0, 0], "b": [_c, _c, _c]},
        {"coeffs": 3/80*math.sqrt(5), "k": [0, 0, 2], "b": [_c, _c, _c]},
        {"coeffs": 3/80*math.sqrt(5), "k": [0, 2, 0], "b": [_c, _c, _c]},
        {"coeffs": 9/80*math.sqrt(5), "k": [0, 2, 2], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim2', 'pip2')), ((1,1), (2,0), (2,0))): [
        {"coeffs": (-3/80*math.sqrt(10))*1j, "k": [2, 0, 0], "b": [_s, _c, _c]},
        {"coeffs": (3/80*math.sqrt(10))*1j, "k": [2, 0, 2], "b": [_s, _c, _c]},
        {"coeffs": (3/80*math.sqrt(10))*1j, "k": [2, 2, 0], "b": [_s, _c, _c]},
        {"coeffs": (-3/80*math.sqrt(10))*1j, "k": [2, 2, 2], "b": [_s, _c, _c]},
        {"coeffs": (3/40*math.sqrt(10))*1j, "k": [1, 2, 2], "b": [_s, _s, _s]},
    ],
    ((('pim1', 'pip1'), ('pim2', 'pip2')), ((2,2), (2,0), (2,0))): [
        {"coeffs": 3/112*math.sqrt(14), "k": [2, 0, 0], "b": [_c, _c, _c]},
        {"coeffs": -3/112*math.sqrt(14), "k": [2, 0, 2], "b": [_c, _c, _c]},
        {"coeffs": -3/112*math.sqrt(14), "k": [2, 2, 0], "b": [_c, _c, _c]},
        {"coeffs": 3/112*math.sqrt(14), "k": [2, 2, 2], "b": [_c, _c, _c]},
        {"coeffs": 3/56*math.sqrt(14), "k": [1, 2, 2], "b": [_c, _s, _s]},
        {"coeffs": -1/112*math.sqrt(14), "k": [0, 0, 0], "b": [_c, _c, _c]},
        {"coeffs": -3/112*math.sqrt(14), "k": [0, 0, 2], "b": [_c, _c, _c]},
        {"coeffs": -3/112*math.sqrt(14), "k": [0, 2, 0], "b": [_c, _c, _c]},
        {"coeffs": -9/112*math.sqrt(14), "k": [0, 2, 2], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim2', 'pip2')), ((3,3), (2,0), (2,0))): [
        {"coeffs": (-3/160*math.sqrt(10))*1j, "k": [2, 0, 0], "b": [_s, _c, _c]},
        {"coeffs": (3/160*math.sqrt(10))*1j, "k": [2, 0, 2], "b": [_s, _c, _c]},
        {"coeffs": (3/160*math.sqrt(10))*1j, "k": [2, 2, 0], "b": [_s, _c, _c]},
        {"coeffs": (-3/160*math.sqrt(10))*1j, "k": [2, 2, 2], "b": [_s, _c, _c]},
        {"coeffs": (-3/20*math.sqrt(10))*1j, "k": [1, 2, 2], "b": [_s, _s, _s]},
    ],
    ((('pim1', 'pip1'), ('pim2', 'pip2')), ((4,4), (2,0), (2,0))): [
        {"coeffs": 3/1120*math.sqrt(70), "k": [2, 0, 0], "b": [_c, _c, _c]},
        {"coeffs": -3/1120*math.sqrt(70), "k": [2, 0, 2], "b": [_c, _c, _c]},
        {"coeffs": -3/1120*math.sqrt(70), "k": [2, 2, 0], "b": [_c, _c, _c]},
        {"coeffs": 3/1120*math.sqrt(70), "k": [2, 2, 2], "b": [_c, _c, _c]},
        {"coeffs": 3/70*math.sqrt(70), "k": [1, 2, 2], "b": [_c, _s, _s]},
        {"coeffs": 3/560*math.sqrt(70), "k": [0, 0, 0], "b": [_c, _c, _c]},
        {"coeffs": 9/560*math.sqrt(70), "k": [0, 0, 2], "b": [_c, _c, _c]},
        {"coeffs": 9/560*math.sqrt(70), "k": [0, 2, 0], "b": [_c, _c, _c]},
        {"coeffs": 27/560*math.sqrt(70), "k": [0, 2, 2], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((0,0), (0,0), (0,0))): [
        {"coeffs": 1, "k": [0, 0, 0], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((0,0), (1,1), (1,0))): [
        {"coeffs": -1, "k": [0, 0, 1], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((0,0), (2,2), (2,0))): [
        {"coeffs": 1/4, "k": [0, 0, 0], "b": [_c, _c, _c]},
        {"coeffs": 3/4, "k": [0, 0, 2], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((1,1), (1,0), (0,0))): [
        {"coeffs": -1, "k": [0, 1, 0], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((1,1), (0,1), (1,0))): [
        {"coeffs": 1/3*math.sqrt(3), "k": [1, 1, 1], "b": [_c, _s, _s]},
        {"coeffs": -1/3*math.sqrt(3), "k": [0, 1, 1], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((1,1), (1,1), (1,0))): [
        {"coeffs": (-1/2*math.sqrt(2))*1j, "k": [1, 1, 1], "b": [_s, _s, _s]},
    ],
    ((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((1,1), (2,1), (1,0))): [
        {"coeffs": 1/6*math.sqrt(6), "k": [1, 1, 1], "b": [_c, _s, _s]},
        {"coeffs": 1/3*math.sqrt(6), "k": [0, 1, 1], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((1,1), (1,2), (2,0))): [
        {"coeffs": -3/20*math.sqrt(10), "k": [1, 1, 2], "b": [_c, _s, _s]},
        {"coeffs": 1/20*math.sqrt(10), "k": [0, 1, 0], "b": [_c, _c, _c]},
        {"coeffs": 3/20*math.sqrt(10), "k": [0, 1, 2], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((1,1), (2,2), (2,0))): [
        {"coeffs": (1/4*math.sqrt(6))*1j, "k": [1, 1, 2], "b": [_s, _s, _s]},
    ],
    ((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((1,1), (3,2), (2,0))): [
        {"coeffs": -1/10*math.sqrt(15), "k": [1, 1, 2], "b": [_c, _s, _s]},
        {"coeffs": -1/20*math.sqrt(15), "k": [0, 1, 0], "b": [_c, _c, _c]},
        {"coeffs": -3/20*math.sqrt(15), "k": [0, 1, 2], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((2,2), (2,0), (0,0))): [
        {"coeffs": 1/4, "k": [0, 0, 0], "b": [_c, _c, _c]},
        {"coeffs": 3/4, "k": [0, 2, 0], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((2,2), (1,1), (1,0))): [
        {"coeffs": -3/20*math.sqrt(10), "k": [1, 2, 1], "b": [_c, _s, _s]},
        {"coeffs": 1/20*math.sqrt(10), "k": [0, 0, 1], "b": [_c, _c, _c]},
        {"coeffs": 3/20*math.sqrt(10), "k": [0, 2, 1], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((2,2), (2,1), (1,0))): [
        {"coeffs": (1/4*math.sqrt(6))*1j, "k": [1, 2, 1], "b": [_s, _s, _s]},
    ],
    ((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((2,2), (3,1), (1,0))): [
        {"coeffs": -1/10*math.sqrt(15), "k": [1, 2, 1], "b": [_c, _s, _s]},
        {"coeffs": -1/20*math.sqrt(15), "k": [0, 0, 1], "b": [_c, _c, _c]},
        {"coeffs": -3/20*math.sqrt(15), "k": [0, 2, 1], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((2,2), (0,2), (2,0))): [
        {"coeffs": 3/80*math.sqrt(5), "k": [2, 0, 0], "b": [_c, _c, _c]},
        {"coeffs": -3/80*math.sqrt(5), "k": [2, 0, 2], "b": [_c, _c, _c]},
        {"coeffs": -3/80*math.sqrt(5), "k": [2, 2, 0], "b": [_c, _c, _c]},
        {"coeffs": 3/80*math.sqrt(5), "k": [2, 2, 2], "b": [_c, _c, _c]},
        {"coeffs": -3/20*math.sqrt(5), "k": [1, 2, 2], "b": [_c, _s, _s]},
        {"coeffs": 1/80*math.sqrt(5), "k": [0, 0, 0], "b": [_c, _c, _c]},
        {"coeffs": 3/80*math.sqrt(5), "k": [0, 0, 2], "b": [_c, _c, _c]},
        {"coeffs": 3/80*math.sqrt(5), "k": [0, 2, 0], "b": [_c, _c, _c]},
        {"coeffs": 9/80*math.sqrt(5), "k": [0, 2, 2], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((2,2), (1,2), (2,0))): [
        {"coeffs": (-3/80*math.sqrt(10))*1j, "k": [2, 0, 0], "b": [_s, _c, _c]},
        {"coeffs": (3/80*math.sqrt(10))*1j, "k": [2, 0, 2], "b": [_s, _c, _c]},
        {"coeffs": (3/80*math.sqrt(10))*1j, "k": [2, 2, 0], "b": [_s, _c, _c]},
        {"coeffs": (-3/80*math.sqrt(10))*1j, "k": [2, 2, 2], "b": [_s, _c, _c]},
        {"coeffs": (3/40*math.sqrt(10))*1j, "k": [1, 2, 2], "b": [_s, _s, _s]},
    ],
    ((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((2,2), (2,2), (2,0))): [
        {"coeffs": 3/112*math.sqrt(14), "k": [2, 0, 0], "b": [_c, _c, _c]},
        {"coeffs": -3/112*math.sqrt(14), "k": [2, 0, 2], "b": [_c, _c, _c]},
        {"coeffs": -3/112*math.sqrt(14), "k": [2, 2, 0], "b": [_c, _c, _c]},
        {"coeffs": 3/112*math.sqrt(14), "k": [2, 2, 2], "b": [_c, _c, _c]},
        {"coeffs": 3/56*math.sqrt(14), "k": [1, 2, 2], "b": [_c, _s, _s]},
        {"coeffs": -1/112*math.sqrt(14), "k": [0, 0, 0], "b": [_c, _c, _c]},
        {"coeffs": -3/112*math.sqrt(14), "k": [0, 0, 2], "b": [_c, _c, _c]},
        {"coeffs": -3/112*math.sqrt(14), "k": [0, 2, 0], "b": [_c, _c, _c]},
        {"coeffs": -9/112*math.sqrt(14), "k": [0, 2, 2], "b": [_c, _c, _c]},
    ],
    ((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((2,2), (3,2), (2,0))): [
        {"coeffs": (-3/160*math.sqrt(10))*1j, "k": [2, 0, 0], "b": [_s, _c, _c]},
        {"coeffs": (3/160*math.sqrt(10))*1j, "k": [2, 0, 2], "b": [_s, _c, _c]},
        {"coeffs": (3/160*math.sqrt(10))*1j, "k": [2, 2, 0], "b": [_s, _c, _c]},
        {"coeffs": (-3/160*math.sqrt(10))*1j, "k": [2, 2, 2], "b": [_s, _c, _c]},
        {"coeffs": (-3/20*math.sqrt(10))*1j, "k": [1, 2, 2], "b": [_s, _s, _s]},
    ],
    ((('pim1', 'pip1'), ('pim1', 'pip1', 'pip2')), ((2,2), (4,2), (2,0))): [
        {"coeffs": 3/1120*math.sqrt(70), "k": [2, 0, 0], "b": [_c, _c, _c]},
        {"coeffs": -3/1120*math.sqrt(70), "k": [2, 0, 2], "b": [_c, _c, _c]},
        {"coeffs": -3/1120*math.sqrt(70), "k": [2, 2, 0], "b": [_c, _c, _c]},
        {"coeffs": 3/1120*math.sqrt(70), "k": [2, 2, 2], "b": [_c, _c, _c]},
        {"coeffs": 3/70*math.sqrt(70), "k": [1, 2, 2], "b": [_c, _s, _s]},
        {"coeffs": 3/560*math.sqrt(70), "k": [0, 0, 0], "b": [_c, _c, _c]},
        {"coeffs": 9/560*math.sqrt(70), "k": [0, 0, 2], "b": [_c, _c, _c]},
        {"coeffs": 9/560*math.sqrt(70), "k": [0, 2, 0], "b": [_c, _c, _c]},
        {"coeffs": 27/560*math.sqrt(70), "k": [0, 2, 2], "b": [_c, _c, _c]},
    ],
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
