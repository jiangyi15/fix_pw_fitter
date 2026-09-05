"""Tests for the numeric helicity-angle engine (ampfit.helicity_angle).

Covers:
  * Clebsch–Gordan against closed-form values and orthonormality
  * Wigner small-d unitarity
  * exact monomial expansion == numeric amplitude for integer and
    half-integer decay chains
"""
import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from ampfit.helicity_angle import (cg, wigner_d, amplitude,
                                   amplitude_monomials, tree_info,
                                   wave_ls_lists)


def _eval_mono(var_order, mono, angles):
    tot = 0j
    for key, coef in mono.items():
        val = 1.0
        for (var_id, v, which), (kind, f) in zip(var_order, key):
            x = angles[v][1] if which == 'theta' else angles[v][0]
            val *= math.cos(f * x) if kind == 'c' else math.sin(f * x)
        tot += coef * val
    return tot


def test_cg_closed_forms():
    assert cg(2.5, 0.5, 0, 0, 2.5, 0.5) == pytest.approx(1.0)
    assert cg(0, 0, 1, 0, 1, 0) == pytest.approx(1.0)
    assert cg(0.5, 0.5, 0.5, -0.5, 1, 0) == pytest.approx(1 / math.sqrt(2))
    assert cg(1, 1, 1, 0, 2, 1) == pytest.approx(math.sqrt(0.5))
    assert cg(1, 1, 1, -1, 2, 0) == pytest.approx(math.sqrt(1 / 6))
    assert cg(1, 0, 1, 0, 2, 0) == pytest.approx(math.sqrt(2 / 3))
    # forbidden by m1+m2 != M or by triangle rule
    assert cg(0.5, 0.5, 0.5, 0.5, 0, 0) == 0.0
    assert cg(0.5, 0.5, 1, 0, 0, 0) == 0.0
    # physically invalid projections (integer spin with half-integer m)
    assert cg(0, 0, 2, -1.5, 2, -1.5) == 0.0
    assert cg(2.5, 1, 0, 0, 2.5, 1) == 0.0


def test_cg_orthonormality():
    rng = np.random.default_rng(3)
    for j1v, j2v in [(0.5, 1.0), (1.0, 1.5), (2.0, 1.0)]:
        j1, j2 = Fraction(j1v), Fraction(j2v)

        def pick(j):
            vals = [Fraction(k, 2) for k in range(-int(2 * j), int(2 * j) + 1, 2)]
            return vals[rng.integers(0, len(vals))]

        m1, m2 = pick(j1), pick(j2)
        tot = 0.0
        for J2 in range(abs(int(2 * j1) - int(2 * j2)),
                       int(2 * j1) + int(2 * j2) + 1, 2):
            J = Fraction(J2, 2)
            for m3 in [Fraction(k, 2) for k in range(-J2, J2 + 1, 2)]:
                c = cg(j1, m1, j2, m2, J, m3)
                tot += c * c
        assert tot == pytest.approx(1.0, abs=1e-9)


def test_wigner_d_reference_values():
    # Reference values at θ = 0.8 from sympy's wigner_d (our convention).
    th = 0.8
    refs = [
        (0.5, 0.5, 0.5, 0.9210609940028851),
        (0.5, -0.5, 0.5, -0.3894183423086505),
        (1.0, 1.0, -1.0, 0.1516466453264173),
        (1.0, 1.0, 0.0, 0.5072473564005260),
        (1.0, 0.0, 0.0, 0.6967067093471655),
        (1.5, 1.5, -1.5, 0.0590539852396813),
        (1.5, 0.5, -0.5, 0.6016747288982572),
        (2.0, 2.0, 0.0, 0.3151267091442942),
        (2.0, 0.0, 0.0, 0.2281003582740334),
        (2.5, 1.5, -1.5, 0.1914392471849580),
    ]
    for J, m, mp, expected in refs:
        assert wigner_d(J, m, mp, th) == pytest.approx(expected, abs=1e-11)


def _check_wave(node, wave, lamtop, lamleaves, n=60, tol=1e-9):
    _, mono = amplitude_monomials(node, wave, lamtop, lamleaves)
    var_order = [(2 * v, v, 'phi') for v in range(len(tree_info(node)[0]))]
    var_order += [(2 * v + 1, v, 'theta') for v in range(len(tree_info(node)[0]))]
    var_order.sort()
    rng = np.random.default_rng(5)
    worst = 0.0
    nv = len(tree_info(node)[0])
    for _ in range(n):
        ang = {v: (rng.uniform(0, 2 * math.pi), rng.uniform(0, math.pi))
               for v in range(nv)}
        a = amplitude(node, wave, ang, lamtop, lamleaves)
        b = _eval_mono(var_order, mono, ang)
        worst = max(worst, abs(a - b))
    assert worst < tol, (node, wave, worst)


def test_monomial_matches_numeric_integer_cascade():
    node = (0, [(1, [0, 0]), (1, [0, 0])])
    lsets = wave_ls_lists(node)
    for wave in itertools.product(*lsets):
        _check_wave(node, wave, 0, (0, 0, 0, 0))


def test_monomial_matches_numeric_half_integer():
    node = (0.5, [0.5, 0])
    lsets = wave_ls_lists(node)
    for wave in itertools.product(*lsets):
        for lamtop, lamleaf in [(0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]:
            _check_wave(node, wave, lamtop, (lamleaf, 0), n=40)


def test_monomial_small_sparse():
    # jpsi-like: (1, [(Jres,[0,0]), 0]) — few nonzero monomials per wave.
    for Jres in (1, 2):
        node = (1, [(Jres, [0, 0]), 0])
        lsets = wave_ls_lists(node)
        for wave in itertools.product(*lsets):
            _, mono = amplitude_monomials(node, wave, 1, (0, 0, 0))
            assert 0 < len(mono) <= 16


def test_reproduces_predefined_B_to_4pi_formulas():
    """The generator reproduces the predefined angular_formula.cache_formula
    entries (B→ρρ→4π-like, sub-decays with s=0) up to a permutation of the
    angle variables (per-cache-entry variable order differs only by ordering).

    Representative entry: ls ((0,0),(1,0),(1,0)) —
      √3/3·(cos(Φ) sinθ1 sinθ2 − cosθ1 cosθ2),   Φ = φ1 + φ2.
    """
    from ampfit.angular_formula import cache_formula
    key = (('pim1', 'pip1'), ('pim2', 'pip2')), ((0, 0), (1, 0), (1, 0))
    entry = cache_formula[key]

    def eval_cache(x):
        s = 0j
        for t in entry:
            f = 1.0
            for kk, bb, xx in zip(t['k'], t['b'], x):
                f *= math.cos(kk * xx) if bb == 'cos' else math.sin(kk * xx)
            s += complex(t['coeffs']) * f
        return s

    tree = (0, [(1, [0, 0]), (1, [0, 0])])
    wave = ((0, 0), (1, 0), (1, 0))
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(40):
        th1 = rng.uniform(0.2, math.pi - 0.2)
        th2 = rng.uniform(0.2, math.pi - 0.2)
        Phi = rng.uniform(0, 2 * math.pi)
        # cache variables (from the scan) are x = (Φ, θ1, θ2); the entry's
        # ordering differs from our per-vertex (φ_v, θ_v) storage, which is
        # exactly the "only the order of the angles differs" statement.
        c = eval_cache((Phi, th1, th2))
        phi1 = float(rng.uniform(0, 2 * math.pi))
        angles = {0: (0.0, 0.0), 1: (phi1, th1), 2: (Phi - phi1, th2)}
        a = amplitude(tree, wave, angles, 0, (0, 0, 0, 0))
        worst = max(worst, abs(a - c))
    assert worst < 1e-10


def test_gauge_top0_matches_cache_phi_theta_layout():
    """Top J=0 gauge rule: drop the first three per-vertex angles
    (φ0, θ0, φ1); the remaining layout is [φ2?, θ1, θ2] which matches the
    old cache's (Φ, θ1, θ2) ordering with the surviving azimuth as Φ."""
    from ampfit.angular_formula import cache_formula
    from ampfit.helicity_angle import gauge_fix_top0

    tree = (0, [(1, [0, 0]), (1, [0, 0])])
    topo = (('pim1', 'pip1'), ('pim2', 'pip2'))

    def eval_mono(red, mono, angles):
        tot = 0j
        for key, coef in mono.items():
            val = 1.0
            for (vtx, kind), (k, f) in zip(red, key):
                x = angles[vtx][0] if kind == 'phi' else angles[vtx][1]
                val *= math.cos(f * x) if k == 'c' else math.sin(f * x)
            tot += coef * val
        return tot

    def eval_cache(entry, x):
        s = 0j
        for t in entry:
            f = 1.0
            for kk, bb, xx in zip(t['k'], t['b'], x):
                f *= math.cos(kk * xx) if bb == 'cos' else math.sin(kk * xx)
            s += complex(t['coeffs']) * f
        return s

    rng = np.random.default_rng(53)
    worst = 0.0
    for ls in [((0, 0), (1, 0), (1, 0)),
               ((1, 1), (1, 0), (1, 0)),
               ((2, 2), (1, 0), (1, 0))]:
        entry = cache_formula[(topo, ls)]
        _, mono = amplitude_monomials(tree, ls, 0, (0, 0, 0, 0))
        red, mono_g = gauge_fix_top0(mono, 3)
        assert red == [(2, 'phi'), (1, 'theta'), (2, 'theta')]
        for _ in range(40):
            th1 = rng.uniform(0.2, math.pi - 0.2)
            th2 = rng.uniform(0.2, math.pi - 0.2)
            ph2 = rng.uniform(0, 2 * math.pi)
            angles = {0: (0.0, 0.0), 1: (0.0, th1), 2: (ph2, th2)}
            c = eval_cache(entry, (ph2, th1, th2))
            o = amplitude(tree, ls, angles, 0, (0, 0, 0, 0))
            g = eval_mono(red, mono_g, angles)
            worst = max(worst, abs(o - c), abs(o - g))
    assert worst < 1e-12


def test_chain_angular_table_reproduces_cache():
    """DecayChain → canonical angular table; the gauge-fixed [φ,θ,θ] table
    reproduces the predefined B→ρρ cache rows numerically."""
    from ampfit.config_loader import Config
    from ampfit.helicity_angle import (chain_angular_table, evaluate_table)
    from ampfit.angular_formula import cache_formula

    cfg = Config('config_amp.yml')
    ch = [cc for cc in cfg.full_decay.chains if 'rhoA' in str(cc)][0]
    tbl = chain_angular_table(ch)
    assert tbl['variables'] == [(2, 'phi'), (1, 'theta'), (2, 'theta')]
    assert tbl['n_projections'] == 1
    assert tbl['n_waves'] == 3

    topo = (('pim1', 'pip1'), ('pim2', 'pip2'))

    def eval_cache(entry, x):
        s = 0j
        for t in entry:
            f = 1.0
            for kk, bb, xx in zip(t['k'], t['b'], x):
                f *= math.cos(kk * xx) if bb == 'cos' else math.sin(kk * xx)
            s += complex(t['coeffs']) * f
        return s

    rng = np.random.default_rng(17)
    worst = 0.0
    for wi, row in enumerate(tbl['waves']):
        ls = tuple(row['ls'])
        if (topo, ls) not in cache_formula:
            continue
        entry = cache_formula[(topo, ls)]
        for _ in range(30):
            th1 = rng.uniform(0.2, math.pi - 0.2)
            th2 = rng.uniform(0.2, math.pi - 0.2)
            ph2 = rng.uniform(0, 2 * math.pi)
            vals = [ph2, th1, th2]
            Amat = evaluate_table(tbl, vals)[wi]
            c = eval_cache(entry, (ph2, th1, th2))
            worst = max(worst, abs(Amat - c))
    assert worst < 1e-12


def test_chain_angular_table_self_consistent():
    from ampfit.config_loader import Config
    from ampfit.helicity_angle import (chain_angular_table, evaluate_table,
                                       decay_chain_to_tree, amplitude)

    cfg = Config('config_amp.yml')
    ch = [cc for cc in cfg.full_decay.chains if 'rhoA' in str(cc)][0]
    tbl = chain_angular_table(ch)
    tree = decay_chain_to_tree(ch)
    rng = np.random.default_rng(7)
    worst = 0.0
    for wi, row in enumerate(tbl['waves']):
        lt, lls = row['proj']
        for _ in range(15):
            ang = {v: (0.0, 0.0) for v in range(tbl['n_vertices'])}
            ph2 = rng.uniform(0, 2 * math.pi)
            ang[1] = (0.0, rng.uniform(0.2, math.pi - 0.2))
            ang[2] = (ph2, rng.uniform(0.2, math.pi - 0.2))
            A = amplitude(tree, row['ls'], ang, lt, lls)
            vals = [(ang[v][0] if k == 'phi' else ang[v][1])
                    for (v, k) in tbl['variables']]
            Amat = evaluate_table(tbl, vals)[wi]
            worst = max(worst, abs(A - Amat))
    assert worst < 1e-12


def test_angular_model_combines_all_chains():
    """config_amp: all chains merge into one phi-first gauge layout with a
    global basis; every entry reproduces amplitude() numerically."""
    from ampfit.config_loader import Config
    from ampfit.helicity_angle import (angular_model, evaluate_model,
                                       decay_chain_to_tree, amplitude)

    cfg = Config('config_amp.yml')
    chains = list(cfg.full_decay.chains)
    models = angular_model(chains)
    assert len(models) == 1
    key = next(iter(models))
    model = models[key]
    assert model['variables'] == [(2, 'phi'), (1, 'theta'), (2, 'theta')]
    assert model['n_chains'] == len(chains)
    assert len(model['entries']) == 62
    assert len(model['basis']) == 19

    rng = np.random.default_rng(71)
    worst = 0.0
    for _ in range(40):
        e = model['entries'][int(rng.integers(len(model['entries'])))]
        ch = chains[e['chain']]
        tree = decay_chain_to_tree(ch)
        ang = {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0)}
        th1 = rng.uniform(0.2, math.pi - 0.2)
        th2 = rng.uniform(0.2, math.pi - 0.2)
        ph2 = rng.uniform(0, 2 * math.pi)
        ang[1] = (0.0, th1)
        ang[2] = (ph2, th2)
        A = amplitude(tree, e['ls'], ang, e['proj'][0], e['proj'][1])
        vals = [(ang[v][0] if k == 'phi' else ang[v][1])
                for (v, k) in model['variables']]
        idx = model['entries'].index(e)
        Amat = evaluate_model(model, vals)[idx]
        worst = max(worst, abs(A - Amat))
    assert worst < 1e-12


def test_angle_formula_mode_option():
    """Config option angle_formula: 'helicity' (default) / 'cache'."""
    import copy
    import yaml
    from ampfit.config_loader import Config, load_config

    base = yaml.safe_load(open('config_amp.yml'))
    assert Config(copy.deepcopy(base)).angle_formula_mode == 'helicity'
    d = copy.deepcopy(base)
    d['angle_formula'] = 'cache'
    assert Config(d).angle_formula_mode == 'cache'
    d = copy.deepcopy(base)
    d['angle_formula'] = 'nope'
    with pytest.raises(ValueError):
        Config(d)


def test_compare_to_cache_all_matched():
    from ampfit.config_loader import Config
    from ampfit.helicity_angle import compare_to_cache

    cfg = Config('config_amp.yml')
    rows, summary = compare_to_cache(list(cfg.full_decay.chains),
                                     verbose=False)
    assert summary['rows_found'] == 62
    assert summary['unmatched'] == 0
