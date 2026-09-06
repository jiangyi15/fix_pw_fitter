#!/usr/bin/env python3
"""Save single-chain stand-alone mass & angles (.npy) from a 4-momentum file.

Input  : event 4-momentum file (N, n_finals, 4) in ``cfg.finals`` order.
Output : per chosen chain ONE plain .npy file, all variables stacked

    {out}.npy  (N, n_mass + n_vars)
        first columns:  intermediate (sub-system) invariant masses of the
                        chain's decays[1:]
        last columns:   canonical per-vertex Euler angles, phi block first
                        then theta block, each ordered by vertex

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
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from ampfit.config_loader import Config
from ampfit.helicity_angle import decay_chain_leaves
from ampfit.helicity_angle import canonical_variables, tree_vertices
from ampfit.momenta_to_angles import decay_angles_vectorized
from ampfit.pwa_build import _boost_vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True,
                    help="4-momentum .npy (N, n_finals, 4), cfg.finals order")
    ap.add_argument("--chain", required=True,
                    help="structural resonance name(s) in decay: — a single "
                         "label (pipeta), a comma/JSON list of the internal "
                         "cores of one chain ([rhoA,rhoB] for any number of "
                         "decays), or an integer topology slot")
    ap.add_argument("--out", default=None,
                    help="output file prefix (default = --chain value)")
    args = ap.parse_args()

    cfg = Config(args.config)
    kc = cfg.build_all_index()

    # resolve chain selector -> topology slot (int, single name, or list)
    sel = args.chain
    try:
        tid = int(sel)
    except ValueError:
        if sel.strip().startswith("["):
            sel = json.loads(sel)
        elif "," in sel:
            sel = [x.strip() for x in sel.split(",")]
        tid = cfg.topo_index_from_name(sel)

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

    # single output: stack ALL variables into one array
    #   columns = [intermediate masses ..., phi(v0..), theta(v0..)]
    arr = np.concatenate([mass, ang], axis=-1)
    prefix = args.out or str(args.chain)
    np.save(prefix + ".npy", arr)
    print(f"wrote {prefix}.npy    {arr.shape}   "
          f"[{mass.shape[1]} mass + {ang.shape[1]} angles, phi-first]")
    print(f"chain: {chain}")
    print(f"finals (columns): {cfg.finals}")


if __name__ == "__main__":
    main()
