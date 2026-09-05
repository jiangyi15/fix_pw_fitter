"""Tests for the momentum→per-vertex angle module (momenta_to_angles).

Validated properties for random physical B→ρρ→4π events:
  * on-shell / inverse of the Lorentz boost
  * in every vertex rest frame the two daughters are back-to-back
  * global 3-D rotation of the event leaves θ₁, θ₂ and φ₁+φ₂ invariant
    (the second daughter's axis is opposite, so azimuths enter as a sum)
"""
import math

import numpy as np
import pytest

from ampfit.config_loader import Config
from ampfit.helicity_angle import decay_chain_leaves
from ampfit.momenta_to_angles import (_boost_4vector as boost,
                                      build_node_momenta,
                                      decay_angles_from_momenta)


def _rho_chain():
    cfg = Config('config_amp.yml')
    return [cc for cc in cfg.full_decay.chains if 'rhoA' in str(cc)][0]


def _generate_event(chain, seed):
    rng = np.random.default_rng(seed)
    masses = {}
    for d in chain.decays:
        masses[d.core.name] = float(d.core.mass)
        for o in d.outs:
            masses.setdefault(o.name, float(o.mass))

    def two(M, m1, m2):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        p = float(np.sqrt(max(((M ** 2 - (m1 + m2) ** 2)
                               * (M ** 2 - (m1 - m2) ** 2)), 0.0))) / (2 * M)
        return (np.array([math.hypot(p, m1), *(p * u).tolist()]),
                np.array([math.hypot(p, m2), *(-p * u).tolist()]))

    def outs_of(name):
        for d in chain.decays:
            if d.core.name == name:
                return [o.name for o in d.outs]
        return []

    cm = {chain.top: np.array([masses[chain.top], 0., 0., 0.])}

    def rec(name):
        outs = outs_of(name)
        if not outs:
            return
        beta = cm[name][1:] / cm[name][0]
        a, b = two(masses[name], masses[outs[0]], masses[outs[1]])
        # rest→CM: inverse of the module boost (boost with −β)
        cm[outs[0]] = boost(a, -beta)
        cm[outs[1]] = boost(b, -beta)
        rec(outs[0])
        rec(outs[1])

    rec(chain.top)
    return {o.name: cm[o.name] for o in decay_chain_leaves(chain)}


def _wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def test_boost_on_shell_and_inverse():
    rng = np.random.default_rng(3)
    for _ in range(20):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        p = float(rng.uniform(0.1, 1.0))
        q = np.array([math.hypot(p, 0.5), *(p * u).tolist()])
        beta = np.array([0.4, -0.3, 0.8])
        qb = boost(q, beta)
        assert (q[0] ** 2 - q[1:] @ q[1:]) == pytest.approx(
            qb[0] ** 2 - qb[1:] @ qb[1:], abs=1e-9)
        assert np.max(np.abs(boost(qb, -beta) - q)) < 1e-9


def test_back_to_back_and_rotation_invariance():
    chain = _rho_chain()
    worst_bb = 0.0
    worst_th = 0.0
    worst_phi_sum = 0.0
    for seed in range(8):
        fin = _generate_event(chain, seed)
        mom = build_node_momenta(chain, fin)
        # daughters back-to-back in each rest frame
        for d in chain.decays:
            E = mom[d.core.name][0]
            beta = mom[d.core.name][1:] / E
            q = [boost(mom[o.name], beta) for o in d.outs]
            worst_bb = max(worst_bb, float(np.linalg.norm(q[0][1:] + q[1][1:])))
        a1 = decay_angles_from_momenta(chain, fin)
        # rotate all final momenta by a random 3-D rotation
        rng = np.random.default_rng(1000 + seed)
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        ang = rng.uniform(0, 2 * math.pi)
        ca, sa = math.cos(ang), math.sin(ang)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]],
                      [-ax[1], ax[0], 0.]])
        Rm = lambda v: v + sa * (K @ v) + (1 - ca) * (K @ (K @ v))
        fin2 = {k: np.array([v[0], *Rm(v[1:])]) for k, v in fin.items()}
        a2 = decay_angles_from_momenta(chain, fin2)
        # θ of the two sub-decays and φ1+φ2 are rotation invariant
        worst_th = max(worst_th,
                       abs(_wrap(a1[1][1] - a2[1][1])),
                       abs(_wrap(a1[2][1] - a2[2][1])))
        worst_phi_sum = max(
            worst_phi_sum,
            abs(_wrap((a1[1][0] + a1[2][0]) - (a2[1][0] + a2[2][0]))))
    assert worst_bb < 1e-9
    assert worst_th < 1e-9
    assert worst_phi_sum < 1e-9


def test_inverse_angles_to_momenta_roundtrip():
    """angles → 4-momenta → angles reproduces the Euler pairs."""
    from ampfit.momenta_to_angles import angles_to_momenta

    chain = _rho_chain()
    nv = len(chain.decays)
    worst = 0.0
    for seed in range(25):
        rng = np.random.default_rng(seed)
        angs = [(rng.uniform(-math.pi, math.pi),
                 rng.uniform(0.15, math.pi - 0.15)) for _ in range(nv)]
        fin = angles_to_momenta(chain, angs)
        # momentum conservation (top at rest)
        tot = np.zeros(4)
        for v in fin.values():
            tot += v
        assert np.linalg.norm(tot[1:]) < 1e-9
        back = decay_angles_from_momenta(chain, fin)
        for a, b in zip(angs, back):
            worst = max(worst, abs(_wrap(a[0] - b[0])), abs(a[1] - b[1]))
    assert worst < 1e-9


def test_aligns_with_original_momenta_to_data():
    """On the repo's own B→4π events the module reproduces the original
    momenta_to_data ρρ row-0 kinematics:
        th1 = θ₁, th2 = θ₂,  φ = wrap(φ₁+φ₂+π)."""
    from ampfit.phasespace_b4pi import generate_b4pi
    from ampfit.momenta_to_data import momenta_to_data

    chain = _rho_chain()
    labels = ['pip1', 'pim1', 'pip2', 'pim2']
    ev = generate_b4pi(300, seed=11)
    data = momenta_to_data(ev['momenta'])
    orig = data['angles'][:, 0]                 # (φ, θ1, θ2)
    worst = [0.0, 0.0, 0.0]
    for i in range(ev['momenta'].shape[0]):
        fin = {lab: ev['momenta'][i][j] for j, lab in enumerate(labels)}
        angs = decay_angles_from_momenta(chain, fin)
        worst[0] = max(worst[0], abs(orig[i, 1] - angs[1][1]))
        worst[1] = max(worst[1], abs(orig[i, 2] - angs[2][1]))
        worst[2] = max(worst[2], abs(_wrap(orig[i, 0]
                                           - (angs[1][0] + angs[2][0] + math.pi))))
    assert worst[0] < 1e-9
    assert worst[1] < 1e-9
    assert worst[2] < 1e-9
