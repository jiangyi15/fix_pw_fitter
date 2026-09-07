#!/usr/bin/env python3
"""Validation runner for the spinful-final alignment pipeline.

Self-contained checks on a two-topology Lambda_c -> (p pim) ... cascade with
a spin-1/2 FINAL proton (p):

  1. The analytic angle/|p| event fill and the momentum-based path produce
     identical per-chain alignment columns (max |d| < 1e-8).
  2. Cross-checks against REFERENCE NUMBERS produced by tf-pwa runs (fixed
     3 events embedded below).  tf-pwa stores its aligned euler as
     (gamma, beta, alpha) for the physical z-y-z rotation
     R = Rz(alpha) Ry(beta) Rz(gamma); after mapping to our standard
     (alpha, beta, gamma) = tf(gamma, beta, alpha) our columns agree to
     ~1e-14 (embedded tolerance 1e-8).  No tf-pwa source is used here -
     only the recorded results.
  3. Physics invariants of the alignment: per-state amplitudes depend
     strongly on the alignment angles, while the total intensity
     P = sum_p |A_p|^2 is exactly invariant when the final spin is fully
     summed (unitarity); each single chain alone is invariant.

Usage: python tests/validate_aligned_pwa.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from ampfit.config_loader import Config
from ampfit.pwa_build import pwa_event_data_tree
from ampfit.momenta_to_angles import aligned_euler_from_momenta

CONFIG = """data:
    dat_order: [p, pim, pip, eta]
    data: /tmp/does-not-exist.npy
    phsp: /tmp/does-not-exist.npy

decay:
    Lc:
    - [Sigmapi, eta]
    - [pieta, Lambdap]
    Sigmapi: [Lambdap, pip]
    pieta: [pip, eta]
    Lambdap: [p, pim]

particle:
    $top: Lc
    $finals: [p, pim, pip, eta]
    Sigmapi: [ Sig1385p ]
    pieta: [ a098 ]
    Lc:
        J: 0.5
        P: +1
        spins: [-0.5, 0.5]
        mass: 2.28646
    Lambdap:
        J: 0.5
        P: +1
        spins: [-0.5, 0.5]
        mass: 1.11568
    p:
        J: 0.5
        P: +1
        spins: [-0.5, 0.5]
        mass: 0.938272
    pip:
        J: 0
        P: -1
        mass: 0.13957
    pim:
        J: 0
        P: -1
        mass: 0.13957
    eta:
        J: 0
        P: -1
        mass: 0.54786
    Sig1385p:
        J: 1.5
        P: +1
        mass: 1.3828
        width: 0.037
        model: BW
    a098:
        J: 0
        P: +1
        mass: 0.98
        width: 0.075
        model: BW
"""

# 3 CM events used for the tf-pwa cross-check (final order p,pim,pip,eta)
_MOM = np.array([
    # event 0
    [[0.975915130554, 0.136441738423, -0.205592431764, 0.105694844142],
     [0.207906460120, -0.008078163353, -0.088856646547, 0.125636582653],
     [0.311915179704, -0.197036469893, -0.184533589060, 0.070251537143],
     [0.790723229621, 0.068672894823, 0.478982667371, -0.301582963938]],
    # event 1
    [[0.988775313472, 0.160839256047, 0.121609348649, 0.238042379803],
     [0.155779023817, -0.051523958907, 0.006219368068, -0.045759375877],
     [0.351182433090, 0.159987810134, 0.063492450711, 0.272437012395],
     [0.790723229621, -0.269303107273, -0.191321167427, -0.464720016322]],
    # event 2
    [[0.957735139356, 0.019049874280, 0.144670350717, 0.124939357638],
     [0.191177858908, 0.083807551805, -0.021923653791, 0.097799982114],
     [0.346823772114, -0.007927365948, 0.043047320740, 0.314469440566],
     [0.790723229621, -0.094930060136, -0.165794017667, -0.537208780318]],
])

# tf-pwa aligned euler as RECORDED (columns = gamma, beta, alpha of the
# physical rotation).  shape (3 events, 2 chains, 3).
TF_EULER = np.array([
    [[-3.6970996651, 2.6253571620, 2.2266220639],
     [1.4973808790, 2.6253571620, -4.0565632433]],
    [[0.0740583972, 0.7475220317, 2.9534003296],
     [-1.8872822242, 0.7475220317, -3.3297849776]],
    [[0.0609297940, 2.2283227226, -5.7200135562],
     [2.3701488278, 2.2283227226, 0.5631717510]],
])


def run(tmp='/tmp/validate_aligned_pwa.yml'):
    open(tmp, 'w').write(CONFIG)
    cfg = Config(tmp)
    kc = cfg.build_all_index()
    chains, by_tid = [], {}
    for _, dc in cfg.full_decay.get_partial_waves():
        tid = cfg.topo_index[dc.topo_id()]
        by_tid[tid] = dc
        chains.append(dc)

    n_decay = cfg.n_decay
    n_topo = cfg.n_topo
    mom = _MOM.copy()
    X = pwa_event_data_tree(cfg, kc, by_tid, mom, spinful_names=['p'])['angle']

    ok_all = True

    def check(name, cond, extra=''):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

    check('angle buffer shape (n_events, n_topo, 9)',
          X.shape == (mom.shape[0], n_topo, 2 * n_decay + 3))

    # 1) two ampfit paths agree per chain
    ref = aligned_euler_from_momenta(
        chains, {nm: mom[:, i] for i, nm in enumerate(
            ['p', 'pim', 'pip', 'eta'])}, ['p'])
    worst = max(float(np.abs(X[:, tid, 6:9] - ref['p'][:, chains.index(by_tid[tid]), :]).max())
                for tid in by_tid)
    check('aligned paths identical (analytic vs momentum)',
          worst < 1e-8, f"max|d|={worst:.2e}")

    # 2) match the recorded tf-pwa numbers (mapped to our convention:
    #    ampfit (alpha,beta,gamma) = tf (gamma, beta, alpha))
    exp = TF_EULER[..., [2, 1, 0]]          # (3, 2, 3)
    got = np.stack([X[:, 0, 6:9], X[:, 1, 6:9]], axis=1)  # (3, 2, 3)
    diff = float(np.abs(got - exp).max())
    check('aligned euler == recorded tf-pwa results (mapped convention)',
          diff < 1e-8, f"max|d|={diff:.2e}")

    # 3) physics invariants on a larger seeded batch
    rng = np.random.default_rng(11)
    pp = rng.uniform(0.02, 0.3, (60, 4, 3))
    pp = pp - pp.sum(axis=1)[:, None, :] / 4
    mm = {'p': 0.938272, 'pim': 0.13957, 'pip': 0.13957, 'eta': 0.54786}
    big = np.empty((60, 4, 4))
    for j, nm in enumerate(['p', 'pim', 'pip', 'eta']):
        pj = pp[:, j]
        E = np.sqrt(mm[nm] ** 2 + (pj * pj).sum(-1))
        big[:, j] = np.stack([E, pj[:, 0], pj[:, 1], pj[:, 2]], -1)
    Xb = pwa_event_data_tree(cfg, kc, by_tid, big, spinful_names=['p'])['angle']
    M, AK, AB = kc['matrix_angle'], kc['angle_k'], kc['angle_b']
    P = int(kc['n_proj'])
    N = M.shape[1] // P

    def wave_amp(xrows, col):
        A = np.zeros((xrows.shape[0], P), complex)
        for b in range(M.shape[0]):
            fac = np.prod(np.cos(AK[b, :] * xrows + AB[b, :]), axis=-1)
            for q in range(P):
                A[:, q] += fac * M[b, q * N + col]
        return A

    x0, x1 = Xb[:, 0].copy(), Xb[:, 1].copy()
    z0, z1 = x0.copy(), x1.copy()
    z0[:, 6:9] = 0.0
    z1[:, 6:9] = 0.0
    A0r, A1r = wave_amp(x0, 0), wave_amp(x1, 1)
    A0z, A1z = wave_amp(z0, 0), wave_amp(z1, 1)

    rel = float(np.abs(np.sum(np.abs(A0r + A1r) ** 2, axis=-1)
                       - np.sum(np.abs(A0z + A1z) ** 2, axis=-1)).max())
    check('total P invariant under full final-spin sum', rel < 1e-9,
          f"max|d|={rel:.2e}")

    spread = float(np.abs((A0r + A1r) - (A0z + A1z)).max())
    check('per-state amplitudes strongly alignment-dependent',
          spread > 1e-2, f"max|dA_state|={spread:.3f}")

    for label, Ar, Az in (('chain0', A0r, A0z), ('chain1', A1r, A1z)):
        d = float(np.abs(np.sum(np.abs(Ar) ** 2, axis=-1)
                         - np.sum(np.abs(Az) ** 2, axis=-1)).max())
        check(f'{label} single-chain |A|^2 unitary-invariant', d < 1e-9,
              f"max|d|={d:.2e}")

    print(f"\nRESULT: {'ALL PASS' if ok_all else 'FAILURES PRESENT'}")
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(run())
