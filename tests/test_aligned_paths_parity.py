"""Regression: the two ampfit aligned-euler paths agree with each other.

For a Lambda_c -> p pim pip eta multi-topology process (spinful final "p")
we compare, on physical CM events, the alignment euler angles produced by

  * the momentum path ``aligned_euler_from_momenta`` (per-chain SU(2) frames
    built from 4-momenta), and
  * the angle/|p| path ``pwa_event_data_tree`` alignment slice (frames built
    from precomputed per-vertex angles and two-body |p| only).

tf-pwa parity note: both ampfit paths reproduce tf-pwa's per-particle SU(2)
*frames* (r_matrix/b_matrix) to machine precision (~1e-14).  The *aligned*
rotation R = ref_r . inv(chain_r) is therefore expected to agree with tf-pwa
at the rotation level too.  The only naming difference is that tf-pwa's
``SU2M.get_euler_angle`` reports the z-y-z triple with alpha<->gamma swapped
(its columns are (gamma, beta, alpha) relative to our Rz(a)Ry(b)Rz(g)
convention); the physical rotations agree.  This test only needs pure ampfit
(no tf-pwa dependency) and is fast.
"""
import numpy as np

from ampfit.config_loader import Config
from ampfit.momenta_to_angles import (
    aligned_euler_from_momenta, angles_to_momenta,
)
from ampfit.pwa_build import pwa_event_data_tree

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


def _cfg(tmp_path):
    cfg = tmp_path / "lc_p_pim_pip_eta.yml"
    cfg.write_text(LC_CONFIG)
    c = Config(str(cfg))
    assert c.n_decay == 3 and c.n_topo == 2
    return c


def _chains(cfg):
    return [dc for _, dc in cfg.full_decay.get_partial_waves()]


def _cm_events(chain, n_events=6, seed=20260907):
    """Physical on-shell Lambda_c -> p pim pip eta CM events (E = m_Lc).

    Built by inverting per-vertex Euler angles for *chain* (top at rest at
    its nominal mass), so every final is on its mass shell and total CM
    energy equals the top mass - the same kinematics tf-pwa parity uses.
    """
    rng = np.random.default_rng(seed)
    nv = len(chain.decays)
    finals = ["p", "pim", "pip", "eta"]
    mom = {nm: np.zeros((n_events, 4)) for nm in finals}
    for e in range(n_events):
        angs = [(rng.uniform(-np.pi, np.pi), rng.uniform(0.05, np.pi - 0.05))
                for _ in range(nv)]
        fm = angles_to_momenta(chain, angs)
        for nm in finals:
            mom[nm][e] = fm[nm]
    # sanity: on-shell + CM
    tot = np.stack(list(mom.values())).sum(axis=0)
    assert np.max(np.abs(tot[:, 1:])) < 1e-9
    assert abs(float(np.mean(tot[:, 0])) - 2.28646) < 1e-9
    return mom


def test_aligned_euler_paths_agree(tmp_path):
    """aligned_euler_from_momenta == pwa_event_data_tree alignment slice.

    Row t (tid) of the angle buffer maps to chain list-index ``topo_index``;
    column offset of the alignment slice is 2*n_decay (six base vertex cols +
    three euler cols for the one spinful final 'p').
    """
    c = _cfg(tmp_path)
    chains = _chains(c)
    assert len(chains) == 2
    mom = _cm_events(chains[0])

    # --- momentum API ---
    out = aligned_euler_from_momenta(chains, mom, ["p"],
                                     final_rest=False,
                                     align_ref="center_mass")
    assert out is not None
    em = out["p"]                                   # (n, n_chains, 3)

    # --- analytic tree fill ---
    kc = c.build_all_index()
    byt = {}
    for _, dc in c.full_decay.get_partial_waves():
        byt[c.topo_index[dc.topo_id()]] = dc
    moms = np.stack([mom[nm] for nm in c.finals], axis=1)
    dd = pwa_event_data_tree(c, kc, byt, moms, spinful_names=["p"])
    ang = dd["angle"]
    n = moms.shape[0]
    # (n_events, n_topo, 2*n_decay + 3*len(spinful_names)) = (n, 2, 9)
    assert ang.shape == (n, c.n_topo, 2 * c.n_decay + 3)
    al = ang[:, :, 2 * c.n_decay:2 * c.n_decay + 3]     # (n, n_topo, 3)

    # alignment slice (per topology row) equals momentum-API (per chain index)
    for i, dc in enumerate(chains):
        tid = c.topo_index[dc.topo_id()]
        err = np.max(np.abs(em[:, i, :] - al[:, tid, :]))
        assert err < 1e-8, (f"path mismatch chain {i} tid {tid}: {err:.3e}")

    # both chains carry a real alignment (center_mass reference is non-trivial)
    assert np.any(np.abs(al[:, 0, :]) > 1e-6)
    assert np.any(np.abs(al[:, 1, :]) > 1e-6)


def test_aligned_euler_paths_agree_final_rest(tmp_path):
    """Same agreement with final_rest=True (reference boost included)."""
    c = _cfg(tmp_path)
    chains = _chains(c)
    mom = _cm_events(chains[1], n_events=5, seed=7)

    res = aligned_euler_from_momenta(chains, mom, ["p"],
                                     final_rest=True,
                                     align_ref="center_mass")
    assert res is not None
    out = res["p"]

    kc = c.build_all_index()
    byt = {c.topo_index[dc.topo_id()]: dc
           for _, dc in c.full_decay.get_partial_waves()}
    moms = np.stack([mom[nm] for nm in c.finals], axis=1)
    dd = pwa_event_data_tree(c, kc, byt, moms, spinful_names=["p"])
    ang = dd["angle"]
    assert ang.shape == (5, c.n_topo, 2 * c.n_decay + 3)
    al = ang[:, :, 2 * c.n_decay:2 * c.n_decay + 3]
    for i, dc in enumerate(chains):
        tid = c.topo_index[dc.topo_id()]
        assert np.max(np.abs(out[:, i, :] - al[:, tid, :])) < 1e-8
