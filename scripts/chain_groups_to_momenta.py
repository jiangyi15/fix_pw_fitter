"""script 3/3 — rebuild final 4-momenta from the smeared chain group file.

Inverse of ``scripts/save_chain_data.py``: reads the plain group .npy
written by ``scripts/smear_chain_groups.py`` — same column layout as the
chain-data file, over ``n_events × copies`` rows:

    [resonance masses (n_res)  |  φ(v0..)  |  θ(v0..)]

rebuilds every (masses, angles) set back into final-state 4-momenta with
the vectorised two-body inverse boost chain and writes

    (n_events × copies, n_finals, 4)

event-major / copy-minor — exactly the layout for ``cuda_v5_pwa`` with
``resolution_size=copies`` (each group = the resolution cloud of one original
event).  The decay-tree structure and the fixed top mass are re-derived
from ``--config`` + ``--chain`` (the same selector save_chain_data.py /
smear_chain_groups.py use).

Usage:
    python scripts/smear_chain_groups.py --config config.yml --chain pipeta \\
        --chain-data chain_pipeta.npy --copies 20 \\
        --sigma-mass sigma_a2p.npy --out groups_pipeta
    python scripts/chain_groups_to_momenta.py --config config.yml \\
        --chain pipeta --groups groups_pipeta.npy \\
        --out data_momenta_groups.npy
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ampfit.config_loader import Config                       # noqa: E402
from ampfit.chain_kinematics import (                         # noqa: E402
    chain_meta, reconstruct_from_canonical)


def resolve_chain(cfg, sel):
    """Same selector semantics as ``save_chain_data.py``."""
    try:
        tid = int(sel)
    except ValueError:
        if sel.strip().startswith("["):
            sel = json.loads(sel)
        elif "," in sel:
            sel = [x.strip() for x in sel.split(",")]
        tid = cfg.topo_index_from_name(sel)
    for _ls, ch in cfg.full_decay.get_partial_waves():
        if cfg.topo_index.get(ch.topo_id()) == tid:
            return tid, ch
    raise SystemExit(f"no partial-wave chain on topology {tid}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--groups", required=True,
                    help=".npy from smear_chain_groups.py")
    ap.add_argument("--out", default="data_momenta_groups.npy")
    args = ap.parse_args()

    cfg = Config(args.config)
    tid, chain = resolve_chain(cfg, args.chain)
    nv = len(chain.decays)
    n_res = nv - 1
    meta = chain_meta(cfg, chain)
    top_mass = float(
        cfg.dic["particle"][chain.decays[0].core.name]["mass"])

    arr = np.load(args.groups)
    want = n_res + 2 * nv
    if arr.ndim != 2 or arr.shape[1] != want:
        raise SystemExit(f"--groups must be (N, {n_res} + {2 * nv} "
                         f"= {want}), got {arr.shape}")
    N = arr.shape[0]

    M = np.empty((N, nv))
    M[:, 0] = top_mass
    M[:, 1:] = arr[:, :n_res]
    phi = arr[:, n_res:n_res + nv]
    theta = arr[:, n_res + nv:]

    mom = reconstruct_from_canonical(meta, M, phi, theta)
    np.save(args.out, mom)
    # carry the smear-step group-normalised weight companion to the final name
    src_w = os.path.splitext(args.groups)[0] + "_weight.npy"
    if os.path.exists(src_w):
        dst_w = os.path.splitext(args.out)[0] + "_weight.npy"
        np.save(dst_w, np.load(src_w))
        print(f"weights: {dst_w} (carried from smear step)")
    print(f"tid {tid}: {N} group events -> {args.out} {mom.shape}")
    print(f"chain top: {chain.decays[0].core.name}; "
          f"finals (columns): {meta['finals']}")


if __name__ == "__main__":
    main()
