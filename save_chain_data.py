#!/usr/bin/env python3
"""Save single-chain stand-alone data (masses + angles) from a 4-momentum file.

Given an event 4-momentum file (N, n_finals, 4) in ``cfg.finals`` order,
this boosts every event to its own centre-of-mass frame (same as
``pwa_event_data`` / TFPWA ``center_mass=True``) and writes the
intermediate masses and per-vertex angles of ONE decay chain into a .npz
that can be consumed standalone (no kernel config needed).

Usage:
    python save_chain_data.py \\
        --config config.yml \\
        --chain 1            # topology slot (0-based, first active chain)
        --data ../data/data_momenta.npy \\
        --out single_chain_data.npz

Output keys:
    momenta_cm   (n, n_finals, 4)   CM-boosted final momenta (cfg.finals)
    mass_top     (n,)               per-event sqrt(s)
    mass_sub     (n,)               invariant mass of the sub-decay pair
    q_top, q_sub (n,)               two-body breakup momenta
    phi, theta   (n, n_vertices)    per-vertex Euler angles (decays order)
    variables    (n_vars, 2) int    canonical variable list [(vertex, phi=0|theta=1), ...]
    chain        str                the selected chain
    finals       list[str]          final-particle order of the columns
"""
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from ampfit.config_loader import Config
from ampfit.helicity_angle import decay_chain_leaves
from ampfit.helicity_angle import canonical_variables, tree_vertices
from ampfit.momenta_to_angles import decay_angles_vectorized
from ampfit.pwa_build import _boost_vec, _two_body_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True,
                    help="4-momentum .npy (N, n_finals, 4), cfg.finals order")
    ap.add_argument("--chain", type=int, default=0,
                    help="topology slot (0-based, first partial-wave chain "
                         "of that topology)")
    ap.add_argument("--out", default="single_chain_data.npz")
    args = ap.parse_args()

    cfg = Config(args.config)
    kc = cfg.build_all_index()

    # first partial-wave chain belonging to topology slot args.chain
    chain = None
    for ls, ch in cfg.full_decay.get_partial_waves():
        if cfg.topo_index.get(ch.topo_id()) == args.chain:
            chain = ch
            break
    if chain is None:
        raise SystemExit(f"no partial-wave chain on topology slot "
                         f"{args.chain} (n_topo={cfg.n_topo})")

    mom = np.load(args.data)
    if mom.ndim != 3 or mom.shape[-1] != 4:
        raise SystemExit(f"expected (N, n_finals, 4), got {mom.shape}")
    n = mom.shape[0]

    # CM boost (identical to pwa_event_data / TFPWA)
    tot = mom.sum(axis=1)
    beta = -(tot[:, 1:] / tot[:, 0:1])
    mom_cm = np.stack([_boost_vec(mom[:, j], beta) for j in range(mom.shape[1])],
                      axis=1)
    mass_top = np.sqrt(np.clip(tot ** 2 @ np.array([1, -1, -1, -1.]), 0, None))

    # reorder finals-ordered columns into this chain's leaf order
    names = [o.name for o in decay_chain_leaves(chain)]
    perm = [cfg.finals.index(nm) for nm in names]
    mom_chain = mom_cm[:, perm]

    # masses / breakups of the sub system (decays[1])
    sub = [o.name for o in chain.decays[1].outs]
    i_a = names.index(sub[0])
    i_b = names.index(sub[1])
    pa, pb = mom_chain[:, i_a], mom_chain[:, i_b]
    m2 = lambda v: np.clip(v ** 2 @ np.array([1, -1, -1, -1.]), 0, None)
    m_a = float(cfg.dic["particle"][sub[0]]["mass"])
    m_b = float(cfg.dic["particle"][sub[1]]["mass"])
    bachelor = [nm for nm in names if nm not in sub][0]
    m_c = float(cfg.dic["particle"][bachelor]["mass"])
    mass_sub = np.sqrt(m2(pa + pb))
    q_top = _two_body_q(np.maximum(mass_top, mass_sub + m_c), mass_sub, m_c)
    q_sub = _two_body_q(mass_sub, m_a, m_b)

    # per-vertex Euler angles
    phi, theta = decay_angles_vectorized(chain, mom_chain)
    nv = phi.shape[1]
    variables = canonical_variables(nv, top_j0=False)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out,
             momenta_cm=mom_cm, mass_top=mass_top, mass_sub=mass_sub,
             q_top=q_top, q_sub=q_sub,
             phi=phi, theta=theta,
             variables=np.array([[v, 0 if k == 'phi' else 1]
                                 for v, k in variables]),
             chain=str(chain), finals=cfg.finals)
    print(f"wrote {args.out}: {n} events, {nv} vertices")
    print(f"chain: {chain}")
    print(f"finals (columns): {cfg.finals}")
    print(f"mass_top [{mass_top.min():.4f},{mass_top.max():.4f}]  "
          f"mass_sub [{mass_sub.min():.4f},{mass_sub.max():.4f}]")


if __name__ == "__main__":
    main()
