#!/usr/bin/env python3
"""Convert pure-PWA 4-momentum data to the kernel-ready ``{prefix}_arr`` npz.

The event arrays (mass/q/angle/weight/bkg) are built with the same
tree-based converter the runtime loader uses (``pwa_event_data_tree``), so
the npz can be referenced from the config as ``data_arr`` / ``phsp_arr`` /
``data_rec_arr`` / ``phsp_rec_arr`` ... and loaded by
``Fitter.load_dataset(prefix)`` — both loader paths agree by construction.

Optional per-event sidecars override the converter defaults:

* ``--weight  file.npy``   per-event weights (default ones)
* ``--bkg     file.npy``   per-event background values (default ones)
* ``--dat-order a,b,c``    column order of the momenta file (default:
                           the order ``pwa_event_data_tree`` expects, i.e.
                           config ``finals``)

Usage::

    python scripts/convert_pwa_data_npz.py --config config.yml \\
        --momenta data_momenta_orig.npy --out data_rec_arr.npz
    python scripts/convert_pwa_data_npz.py --config config.yml \\
        --momenta ../data2/mc_momenta.npy --out phsp_arr.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from ampfit.config_loader import load_config
from ampfit.decay_tree import DecayTree
from ampfit.pwa_build import block_orders, build_tree_event_data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--momenta", required=True,
                    help="4-momentum .npy (n, n_finals, 4)")
    ap.add_argument("--weight", default=None,
                    help="optional per-event weight .npy (n,)")
    ap.add_argument("--bkg", default=None,
                    help="optional per-event background .npy (n,)")
    ap.add_argument("--dat-order", default=None,
                    help="comma list of final names in the momenta columns "
                         "(default config finals order)")
    ap.add_argument("--out", required=True,
                    help="output .npz (use e.g. data_arr.npz)")
    args = ap.parse_args()

    dic = load_config(args.config)
    tree = DecayTree(dic["decay"], dic["particle"], dic.get("data"))
    pws = tree.partial_waves()
    byt = {tree.topo_index[ch.topo_id()]: ch for _, ch in pws}
    blocks = block_orders(tree.finals, dic.get("data") or {})

    mom = np.load(args.momenta)
    if mom.ndim != 3 or mom.shape[-1] != 4:
        raise SystemExit(f"--momenta must be (n, n_finals, 4), got {mom.shape}")

    order = (args.dat_order.split(",") if args.dat_order
             else list(tree.finals))
    if len(order) != mom.shape[1]:
        raise SystemExit(f"{len(order)} dat-order names but "
                         f"{mom.shape[1]} momenta columns")
    perm = [list(order).index(f) for f in tree.finals]
    if perm != list(range(len(perm))):
        mom = mom[:, perm]

    arr = build_tree_event_data(tree, None, byt, mom, blocks=blocks)
    if args.weight:
        w = np.load(args.weight).astype(float).ravel()
        if w.shape[0] != mom.shape[0]:
            raise SystemExit("--weight length != events")
        arr["weight"] = w
    if args.bkg:
        b = np.load(args.bkg).astype(float).ravel()
        if b.shape[0] != mom.shape[0]:
            raise SystemExit("--bkg length != events")
        arr["bkg"] = b

    np.savez(args.out, **arr)
    print(f"wrote {args.out}  ({mom.shape[0]} events) keys="
          f"{sorted(arr)}")


if __name__ == "__main__":
    main()
