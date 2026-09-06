#!/usr/bin/env python3
"""Save single-chain stand-alone mass & angles (.npy) from a 4-momentum file.

Input  : event 4-momentum file (N, n_finals, 4) in ``cfg.finals`` order.
Output : per chosen chain two plain .npy files

    {out}_mass.npy    (N, n_mass)  intermediate (sub-system) invariant
                                   masses of the chain's decays[1:]
    {out}_angles.npy  (N, n_vars)  canonical per-vertex Euler angles,
                                   phi block first then theta block,
                                   each ordered by vertex — the layout the
                                   kernel/amplitude evaluation consumes

Events are boosted to their own centre-of-mass frame first (identical to
``pwa_event_data`` / TFPWA ``center_mass=True``), so the output is frame
independent.

The chain is selected by its structural resonance name as declared under
``decay:``, e.g. ``pipi`` / ``pipeta`` / ``pimeta`` (the pairing tag used
in the decay entries); the first partial-wave chain of that pairing is
used.

Usage:
    python save_chain_data.py --config config.yml --chain pipeta \
        --data ../data/data_momenta.npy --out pipeta
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from ampfit.config_loader import Config
from ampfit.helicity_angle import decay_chain_leaves
from ampfit.helicity_angle import canonical_variables, tree_vertices
from ampfit.momenta_to_angles import decay_angles_vectorized
from ampfit.pwa_build import _boost_vec


def _core_leaves(cfg, name):
    """Final leaves beneath a structural (decay-section) particle *name*."""
    d = cfg.dic["decay"]
    finals = cfg.finals

    def outs_of(n):
        entry = d.get(n)
        if entry is None:
            return []
        if not isinstance(entry, list):
            entry = [entry]
        outs = []
        for item in entry:
            if isinstance(item, str):
                outs.append(item)
            elif isinstance(item, (list, tuple)):
                outs += [k for k in item if isinstance(k, str)]
        return outs

    def rec(n):
        if n in finals:
            return [n]
        leaves = []
        for o in outs_of(n):
            leaves += rec(o)
        return leaves

    return rec(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True,
                    help="4-momentum .npy (N, n_finals, 4), cfg.finals order")
    ap.add_argument("--chain", required=True,
                    help="structural resonance name in decay:, e.g. pipeta; "
                         "or an integer topology slot")
    ap.add_argument("--out", default=None,
                    help="output file prefix (default = --chain value)")
    args = ap.parse_args()

    cfg = Config(args.config)
    kc = cfg.build_all_index()

    # resolve chain id -> topology slot (by structural name leaves)
    tid = None
    try:
        tid = int(args.chain)
    except ValueError:
        want = sorted(_core_leaves(cfg, args.chain))
        for key, idx in cfg.topo_index.items():
            leaves = sorted(sum((list(grp) for grp in key), []))
            if leaves == want:
                tid = idx
                break
        if tid is None:
            raise SystemExit(
                f"chain name {args.chain!r} leaves {want} not found in "
                f"topologies {cfg.topo_index}")

    chain = None
    for ls, ch in cfg.full_decay.get_partial_waves():
        if cfg.topo_index.get(ch.topo_id()) == tid:
            chain = ch
            break
    if chain is None:
        raise SystemExit(f"no partial-wave chain on topology {tid} "
                         f"(n_topo={cfg.n_topo})")

    mom = np.load(args.data)
    if mom.ndim != 3 or mom.shape[-1] != 4:
        raise SystemExit(f"expected (N, n_finals, 4), got {mom.shape}")
    n = mom.shape[0]

    # CM boost (identical to pwa_event_data / TFPWA)
    tot = mom.sum(axis=1)
    beta = -(tot[:, 1:] / tot[:, 0:1])
    mom_cm = np.stack([_boost_vec(mom[:, j], beta)
                       for j in range(mom.shape[1])], axis=1)

    names = [o.name for o in decay_chain_leaves(chain)]
    perm = [cfg.finals.index(nm) for nm in names]
    mom_chain = mom_cm[:, perm]

    # intermediate (sub-system) invariant masses of decays[1:]
    g = np.array([1, -1, -1, -1.])
    masses = []
    for dec in chain.decays[1:]:
        out = [o.name for o in dec.outs]
        p = sum(mom_chain[:, names.index(o)] for o in out)
        masses.append(np.sqrt(np.clip(p ** 2 @ g, 0.0, None)))
    mass = np.stack(masses, axis=-1) if masses else np.empty((n, 0))

    # canonical per-vertex angles (phi block then theta block, by vertex)
    phi, theta = decay_angles_vectorized(chain, mom_chain)
    nv = phi.shape[1]
    vars_ = canonical_variables(nv, top_j0=False)
    ang = np.empty((n, len(vars_)))
    for j, (v, kind) in enumerate(vars_):
        ang[:, j] = phi[:, v] if kind == 'phi' else theta[:, v]

    prefix = args.out or str(args.chain)
    np.save(prefix + "_mass.npy", mass)
    np.save(prefix + "_angles.npy", ang)
    print(f"wrote {prefix}_mass.npy    {mass.shape}")
    print(f"wrote {prefix}_angles.npy  {ang.shape}   ({nv} vertices, "
          f"phi-first)")
    print(f"chain: {chain}")
    print(f"finals (columns): {cfg.finals}")


if __name__ == "__main__":
    main()
