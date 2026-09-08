"""script 2/3 — smear one chain-data event set into resolution group copies.

Pipeline (three scripts):

    1. ``scripts/save_chain_data.py``      (existing)  momenta -> flat
       per-event chain data (n, n_res + 2·nv):
           [resonance masses of decays[1:]  |  φ(v0..)  |  θ(v0..)].

    2. THIS script  (smear)                for every original event makes
       ``--copies`` smeared replicas of its canonical variables:
           * resonance masses are perturbed with per-event Gaussian noise
             N(0, sigma_mass[e]) and reflected into [Σ rest masses of the
             vertex leaves, m_parent − m_sibling]  (--reflect-mass 0 keeps
             only the lower kinematic threshold);
           * vertex angles optionally perturbed with per-event sigma_phi /
             sigma_theta (reflected into [−π, π] / [0, π]);
           * the top mass is fixed to the config mass of the top particle.
       The smeared copies are written event-major, copy-minor as a plain
       .npy array with EXACTLY the same column layout as save_chain_data:

           (n_events × copies, n_res + 2·nv)

    3. ``scripts/chain_groups_to_momenta.py``  (rebuild)  smeared canonical
       sets -> final 4-momenta (n×copies, n_finals, 4), the group layout
       for ``cuda_v5_pwa`` with ``resolution_size=copies``.

Example:
    python scripts/save_chain_data.py --config config.yml --chain pipeta \\
        --data ../data2/data_momenta.npy --out chain_pipeta
    python scripts/smear_chain_groups.py --config config.yml \\
        --chain pipeta --chain-data chain_pipeta.npy --copies 20 \\
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
from ampfit.chain_kinematics import chain_meta                # noqa: E402


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
    """Reflect *x* into [lo, hi] (hi = +inf → one-sided reflection at lo)."""
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
    ap.add_argument("--chain", required=True)
    ap.add_argument("--chain-data", required=True,
                    help=".npy from save_chain_data.py")
    ap.add_argument("--copies", type=int, default=20)
    ap.add_argument("--sigma-mass", help="(n,) or (n, n_res) per-event mass σ")
    ap.add_argument("--sigma-phi", help="(n,) or (n, nv) per-event φ σ")
    ap.add_argument("--sigma-theta", help="(n,) or (n, nv) per-event θ σ")
    ap.add_argument("--reflect-mass", type=int, default=1,
                    help="0 → only clamp the lower kinematic threshold")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="groups_pipeta",
                    help="output prefix (a .npy file is written)")
    args = ap.parse_args()

    cfg = Config(args.config)
    tid, chain = resolve_chain(cfg, args.chain)
    nv = len(chain.decays)
    n_res = nv - 1
    meta = chain_meta(cfg, chain)
    top_mass = float(
        cfg.dic["particle"][chain.decays[0].core.name]["mass"])

    arr = np.load(args.chain_data)
    want = n_res + 2 * nv
    if arr.ndim != 2 or arr.shape[1] != want:
        raise SystemExit(f"--chain-data must be (n, {n_res} + {2 * nv} "
                         f"= {want}), got {arr.shape}")
    n = arr.shape[0]
    copies = max(1, args.copies)
    if args.reflect_mass not in (0, 1):
        raise SystemExit("--reflect-mass must be 0 or 1")

    sig_m = _load_optional(args.sigma_mass, n, n_res, "--sigma-mass")
    sig_phi = _load_optional(args.sigma_phi, n, nv, "--sigma-phi")
    sig_th = _load_optional(args.sigma_theta, n, nv, "--sigma-theta")

    rng = np.random.RandomState(args.seed)

    # original canonical variables: [res masses, φ(v..), θ(v..)]
    mass0 = arr[:, :n_res].copy()
    phi0 = arr[:, n_res:n_res + nv].copy()
    theta0 = arr[:, n_res + nv:].copy()

    # event-major, copy-minor expansion
    N = n * copies
    mass = np.repeat(mass0, copies, axis=0)
    phi = np.repeat(phi0, copies, axis=0)
    theta = np.repeat(theta0, copies, axis=0)

    # resonance masses (decay vertex v = 1..nv-1 -> column v-1); top fixed.
    core_idx = {d[0]: i for i, d in enumerate(meta["decays"])}
    child_low = [_subtree_rest_sum(meta, d[0]) for d in meta["decays"]]
    M0full = np.empty((n, nv))
    M0full[:, 0] = top_mass
    M0full[:, 1:] = mass0
    for v in range(1, nv):
        sig = sig_m[:, v - 1]
        if np.max(sig) <= 0:
            continue
        core = meta["decays"][v][0]
        parent_i = next(p for p, outs in enumerate(meta["decays"])
                        if core in outs[1])
        sib = [o for o in meta["decays"][parent_i][1] if o != core][0]
        mu = M0full[:, v]
        if parent_i == 0:
            Mpar = np.full(n, top_mass)
        else:
            Mpar = M0full[:, parent_i]        # parent row of each event
        sib_m = (np.full(n, meta["rest"][sib]) if sib not in core_idx
                 else M0full[:, core_idx[sib]])
        lo = np.full(n, child_low[v])
        hi = Mpar - sib_m if args.reflect_mass else np.full(n, np.inf)

        drawn = mu[:, None] + rng.normal(size=(n, copies)) * sig[:, None]
        mass[:, v - 1] = _reflect(drawn, lo[:, None], hi[:, None]).ravel()

    if np.max(sig_phi) > 0:
        noise = rng.normal(size=(n, copies, nv)) * sig_phi[:, None, :]
        phi[:] = _reflect(phi0[:, None, :] + noise, -np.pi, np.pi) \
            .reshape(N, nv)
    if np.max(sig_th) > 0:
        noise = rng.normal(size=(n, copies, nv)) * sig_th[:, None, :]
        theta[:] = _reflect(theta0[:, None, :] + noise, 0.0, np.pi) \
            .reshape(N, nv)

    out = args.out if args.out.endswith(".npy") else args.out + ".npy"
    np.save(out, np.concatenate([mass, phi, theta], axis=-1))
    print(f"tid {tid} ({chain.decays[0].core.name}): "
          f"{n} events x {copies} copies -> {out}  {N} x {want}")
    if n_res >= 1:
        mm = mass[:, 0].reshape(n, copies)
        print("resonance mass original / group mean / group std (first 5):",
              np.round(mass0[:5, 0], 5).tolist(),
              np.round(mm.mean(axis=1)[:5], 5).tolist(),
              np.round(mm.std(axis=1)[:5], 5).tolist())


if __name__ == "__main__":
    main()
