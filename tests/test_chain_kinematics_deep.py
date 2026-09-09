"""Regression: chain_kinematics round-trips exactly at any depth.

canonical_of_chain -> reconstruct_from_canonical -> canonical_of_chain
must be the identity (masses ~1e-15, every vertex angle ~1e-15) for a
physical THREE-vertex cascade, where the older single-boost reconstruction
failed at the deepest vertex.
"""

import os
import tempfile

import numpy as np

from ampfit.config_loader import Config
from ampfit.momenta_to_angles import angles_to_momenta
from ampfit.chain_kinematics import (
    canonical_of_chain, reconstruct_from_canonical, chain_meta)

LC_CONFIG = """data_order:
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
    Lc: {J: 0.5, P: +1, spins: [-0.5, 0.5], mass: 2.28646}
    Lambdap: {J: 0.5, P: +1, spins: [-0.5, 0.5], mass: 1.11568}
    p: {J: 0.5, P: +1, spins: [-0.5, 0.5], mass: 0.938272}
    pip: {J: 0, P: -1, mass: 0.13957}
    pim: {J: 0, P: -1, mass: 0.13957}
    eta: {J: 0, P: -1, mass: 0.54786}
    Sig1385p: {J: 1.5, P: +1, mass: 1.3828, width: 0.037, model: BW}
    a098: {J: 0, P: +1, mass: 0.98, width: 0.075, model: BW}
"""

FINALS = ["p", "pim", "pip", "eta"]


def _config():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "lc.yml")
    with open(p, "w") as fh:
        fh.write(LC_CONFIG)
    return Config(p)


def _physical_events(chain, n=12, seed=7):
    """On-shell Lambda_c CM events via the scalar two-body inverse."""
    rng = np.random.default_rng(seed)
    nv = len(chain.decays)
    arr = np.empty((n, len(FINALS), 4))
    for e in range(n):
        angs = [(rng.uniform(-np.pi, np.pi),
                 rng.uniform(0.05, np.pi - 0.05)) for _ in range(nv)]
        fm = angles_to_momenta(chain, angs)
        arr[e] = np.stack([fm[o] for o in FINALS], axis=0)
    return arr


def test_deep_chain_round_trip_exact():
    cfg = _config()
    chain = [dc for _, dc in cfg.full_decay.get_partial_waves()][0]
    assert len(chain.decays) == 3          # Lc -> Sig1385p -> Lambdap
    arr = _physical_events(chain)
    M, phi, theta = canonical_of_chain(cfg, chain, arr)
    meta = chain_meta(cfg, chain)

    arr2 = reconstruct_from_canonical(meta, M, phi, theta)
    M2, phi2, theta2 = canonical_of_chain(cfg, chain, arr2)

    assert np.abs(M - M2).max() < 1e-9
    assert np.abs(phi - phi2).max() < 1e-9      # every vertex, incl deepest
    assert np.abs(theta - theta2).max() < 1e-9
    assert np.abs(arr - arr2).max() < 1e-9
