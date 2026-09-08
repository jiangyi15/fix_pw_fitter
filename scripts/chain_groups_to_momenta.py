"""script 3/3 — rebuild final 4-momenta from the smeared chain group npz.

Inverse of ``save_chain_data.py``: reads the group npz written by
``scripts/smear_chain_groups.py`` (``mass/phi/theta`` over
``n_events × copies`` rows + the serialised decay-tree ``meta``), rebuilds
every (masses, angles) set back into final-state 4-momenta with the
vectorised two-body inverse boost chain and writes

    (n_events × copies, n_finals, 4)

event-major / copy-minor — exactly the layout for ``cuda_v5_pwa`` with
``nll_batch=copies`` (each group = the resolution cloud of one original
event).

The npz is self-contained (the decay structure and the fixed top mass are
inside ``meta``), so no config is needed at this step.

Usage:
    python scripts/smear_chain_groups.py --config config.yml --chain pipeta \\
        --chain-data chain_pipeta.npy --copies 20 \\
        --sigma-mass sigma_a2p.npy --out groups_pipeta.npz
    python scripts/chain_groups_to_momenta.py --groups groups_pipeta.npz \\
        --out data_momenta_groups.npy
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ampfit.chain_kinematics import reconstruct_from_canonical   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groups", required=True,
                    help="npz from smear_chain_groups.py")
    ap.add_argument("--out", default="data_momenta_groups.npy")
    args = ap.parse_args()

    z = np.load(args.groups)
    mass, phi, theta = z["mass"], z["phi"], z["theta"]
    meta = json.loads(str(z["meta"]))
    N, n_res = mass.shape
    nv = phi.shape[1]
    if n_res != nv - 1 or theta.shape != phi.shape:
        raise SystemExit(f"inconsistent group arrays: mass{N, n_res}, "
                         f"phi{phi.shape}, theta{theta.shape}")

    # vertex masses: fixed top + the smeared resonance columns
    M = np.empty((N, nv))
    M[:, 0] = float(meta["top_mass"])
    M[:, 1:] = mass

    mom = reconstruct_from_canonical(meta, M, phi, theta)
    np.save(args.out, mom)
    copies = int(meta.get("copies", 0))
    print(f"{N} events ({N // copies if copies else '?'} groups) "
          f"-> {args.out} {mom.shape}")
    top = meta["decays"][0][0]
    print(f"chain top: {top}; finals (columns): {meta['finals']}")


if __name__ == "__main__":
    main()
