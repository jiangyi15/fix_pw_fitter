"""script 2/2 — smear the chain data (save_chain_data.py) into groups of
resolution-smeared replicas and rebuild 4-momenta.

Pipeline (two scripts):

    1. ``scripts/save_chain_data.py``  (already exists) turns one topology
       chain of the config into the flat per-event chain data array
           [resonance masses of decays[1:]  |  φ(v0..)  |  θ(v0..)],
       where φ/θ are the canonical per-vertex Euler angles of that chain and
       the resonance masses are the subsystem invariant masses.

    2. THIS script reads that file (plus the same config/chain selector) and
       makes, for every original event, ``--copies`` smeared replicas:

       * resonance masses ``M[:, 1:]`` are perturbed with per-event Gaussian
         noise ``N(0, sigma_mass[e])`` and reflected into
         ``[Σ rest masses of the vertex leaves, m_parent − m_sibling]``
         (use ``--reflect-mass 0`` to keep only the lower threshold);
       * vertex angles are optionally perturbed with ``--sigma-phi`` /
         ``--sigma-theta`` (per-event arrays, reflected into [−π, π]/[0, π]);
       * the top mass is fixed to the config mass of the top particle (events
         of the ππη J/ψ sample sit exactly at the top rest mass).

       Each smeared (masses, angles) set is turned BACK into final-state
       4-momenta by the inverse two-body boost chain, giving

           (n_events × copies, n_finals, 4)

       with the copies of one event stored contiguously — exactly the layout
       for ``cuda_v5_pwa`` with ``nll_batch=copies`` (each group = one
       resolution cloud of one original event).

Example:
    python scripts/save_chain_data.py --config config.yml --chain pipeta \\
        --data ../data2/data_momenta.npy --out chain_pipeta

    python scripts/smear_chain_to_momenta.py \\
        --config config.yml --chain pipeta \\
        --chain-data chain_pipeta.npy --copies 20 \\
        --sigma-mass sigma_a2p.npy         # (n,) per-event mass resolution
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
from ampfit.helicity_angle import decay_chain_leaves          # noqa: E402


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


def _load_optional(path, n, want, label):
    """Per-event sigma array → (n, want) or zeros."""
    if path is None:
        return np.zeros((n, want))
    a = np.load(path).astype(float)
    if a.ndim == 1 and want == 1:
        a = a[:, None]
    if a.shape[0] != n:
        raise SystemExit(f"{label}: {a.shape[0]} events, expected {n}")
    return a[:, :want]


def _subtree_rest_sum(meta, core):
    decays = {d[0]: d[1] for d in meta["decays"]}
    stack, acc = [core], []
    while stack:
        x = stack.pop()
        if x in decays:
            stack += list(decays[x])
        else:
            acc.append(meta["rest"][x])
    return float(sum(acc))


def _reflect(x, lo, hi):
    """Reflect *x* into [lo, hi] (hi may be +inf → one-sided at lo)."""
    x = np.asarray(x, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    if np.all(np.isinf(hi)):
        return lo + np.abs(x - lo)
    d = np.maximum(hi - lo, 1e-300)
    r = (x - lo) % (2.0 * d)
    return np.minimum(np.maximum(lo + np.where(r <= d, r, 2.0 * d - r), lo),
                      hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--chain", required=True,
                    help="chain selector of save_chain_data.py (pairing "
                         "label / core list / integer tid)")
    ap.add_argument("--chain-data", required=True,
                    help=".npy from save_chain_data.py (n, n_res + 2·nv)")
    ap.add_argument("--copies", type=int, default=20)
    ap.add_argument("--sigma-mass", help="(n,) or (n, n_res) per-event mass σ")
    ap.add_argument("--sigma-phi", help="(n,) or (n, nv) per-event φ σ")
    ap.add_argument("--sigma-theta", help="(n,) or (n, nv) per-event θ σ")
    ap.add_argument("--reflect-mass", type=int, default=1,
                    help="0 → only clamp the lower kinematic threshold")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="data_momenta_groups.npy")
    args = ap.parse_args()

    cfg = Config(args.config)
    tid, chain = resolve_chain(cfg, args.chain)
    nv = len(chain.decays)
    n_res = nv - 1
    meta = chain_meta(cfg, chain)

    arr = np.load(args.chain_data)
    want = n_res + 2 * nv
    if arr.ndim != 2 or arr.shape[1] != want:
        raise SystemExit(f"--chain-data must be (n, {n_res} mass + "
                         f"{2 * nv} angles = {want}), got {arr.shape}")
    n = arr.shape[0]
    copies = max(1, args.copies)

    # canonical variables: [resonance masses (decays[1:]), φ block, θ block]
    M0 = np.empty((n, nv))
    M0[:, 0] = float(cfg.dic["particle"][chain.decays[0].core.name]["mass"])
    M0[:, 1:] = arr[:, :n_res]
    phi0 = arr[:, n_res:n_res + nv]
    theta0 = arr[:, n_res + nv:]
    if args.reflect_mass not in (0, 1):
        raise SystemExit("--reflect-mass must be 0 or 1")

    sig_m = _load_optional(args.sigma_mass, n, n_res, "--sigma-mass")
    sig_phi = _load_optional(args.sigma_phi, n, nv, "--sigma-phi")
    sig_th = _load_optional(args.sigma_theta, n, nv, "--sigma-theta")

    N = n * copies
    rng = np.random.RandomState(args.seed)

    # event-major, copy-minor expansion (rows e·copies+j ← (e, j))
    M = np.repeat(M0, copies, axis=0).copy()
    phi = np.repeat(phi0, copies, axis=0).copy()
    theta = np.repeat(theta0, copies, axis=0).copy()

    # smeared resonance masses (decay vertex index v = 1..nv-1, col v-1)
    # bounds: [Σ rest of the vertex leaves, m_parent − m_sibling]
    core_idx = {d[0]: i for i, d in enumerate(meta["decays"])}
    child_low = [_subtree_rest_sum(meta, d[0]) for d in meta["decays"]]
    for v in range(1, nv):
        sig = sig_m[:, v - 1]
        if np.max(sig) <= 0:
            continue
        core = meta["decays"][v][0]
        parent_i = next(p for p, outs in enumerate(meta["decays"])
                        if core in outs[1])
        sib = [o for o in meta["decays"][parent_i][1] if o != core][0]
        mu = M0[:, v]
        if parent_i == 0:
            Mpar = np.full(n, M0[0, 0])          # fixed top mass
        else:
            Mpar = M[::copies, parent_i]         # parent row of each event
        sib_m = (np.full(n, meta["rest"][sib]) if sib not in core_idx
                 else M0[:, core_idx[sib]])
        lo = np.full(n, child_low[v])
        hi = Mpar - sib_m if args.reflect_mass else np.full(n, np.inf)

        drawn = mu[:, None] + rng.normal(size=(n, copies)) * sig[:, None]
        M[:, v] = _reflect(drawn, lo[:, None], hi[:, None]).ravel()

    # optional angle smearing
    if np.max(sig_phi) > 0:
        noise = rng.normal(size=(n, copies, nv)) * sig_phi[:, None, :]
        phi[:] = _reflect(phi0[:, None, :] + noise, -np.pi, np.pi) \
            .reshape(N, nv)
    if np.max(sig_th) > 0:
        noise = rng.normal(size=(n, copies, nv)) * sig_th[:, None, :]
        theta[:] = _reflect(theta0[:, None, :] + noise, 0.0, np.pi) \
            .reshape(N, nv)

    mom = reconstruct_from_canonical(meta, M, phi, theta)
    np.save(args.out, mom)
    print(f"tid {tid} ({chain.decays[0].core.name}): "
          f"{n} events x {copies} copies -> {args.out} {mom.shape}")

    if n_res >= 1:
        mm = M[:, 1].reshape(n, copies)
        print("resonance mass original / group mean / group std (first 5):",
              np.round(M0[:5, 1], 5).tolist(),
              np.round(mm.mean(axis=1)[:5], 5).tolist(),
              np.round(mm.std(axis=1)[:5], 5).tolist())


if __name__ == "__main__":
    main()
