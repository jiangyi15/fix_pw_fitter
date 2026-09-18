#!/usr/bin/env python3
"""Generate pure-PWA toy data for the generic run_fit driver.

Writes kernel-format ``.npz`` files (data + phsp) plus an ``init.json``
start point, so run_fit needs no --toy flag:

    python scripts/gen_toy_pwa.py --config config_pwa.yml
    python run_fit.py --config config_pwa.yml --backend numpy_pwa \\
        --data data_pwa.npz --phsp phsp_pwa.npz --init init_pwa.json --fit

Flat phase space is generated with the config masses (two-body product +
inverse boost chain); toy events are drawn without replacement from an
independent proposal sample weighted by the model density at the Fitter's
parameter point.

The density uses the **Fitter parameter layer** (constraints/defaults via
``initial_values``/``build_params``) and an actual **Backend**
(``--backend``, default ``numpy_pwa``) — never the raw kernel directly.

``--seed`` is a master seed: the sub-seeds are ``seed`` (phsp), ``seed+1``
(proposal), ``seed+2`` (parameter point) and ``seed+3`` (event selection).
Omit it and a random master seed is drawn and printed, so a run can be
reproduced with ``--seed <printed>``.
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

import numpy as np

from ampfit import Fitter
from ampfit.config_loader import row_block_factors
from ampfit.pwa_build import pwa_event_data_tree, generate_pwa_phsp


def _npz(data):
    """Kernel-format npz dict — only the keys the PWA path actually needs
    (mass/q/angle/weight).  time/frac/bkg_raw are legacy-mixing inputs and
    are omitted to save disk + load memory; the loader fills defaults."""
    return {
        "mass": data["mass"],
        "q": data["q"],
        "angle": data["angle"],
        "weight": data["weight"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_pwa.yml")
    ap.add_argument("--backend", default="numpy_pwa",
                    help='compute backend (name or YAML dict); the toy '
                         'generator is a CPU utility, default "numpy_pwa"')
    ap.add_argument("--seed", type=int, default=None,
                    help="master seed; sub-seeds are seed, +1 (phsp/proposal), "
                         "+2 (params), +3 (event selection).  Omitted -> draw "
                         "a random seed and print it")
    ap.add_argument("--nph", type=int, default=8000)
    ap.add_argument("--nprop", type=int, default=40000)
    ap.add_argument("--ndata", type=int, default=4000)
    ap.add_argument("--out-data", default="data_pwa.npz")
    ap.add_argument("--out-phsp", default="phsp_pwa.npz")
    ap.add_argument("--out-init", default="init_pwa.json")
    args = ap.parse_args()

    # master seed: random when omitted (printed so the run is reproducible)
    if args.seed is None:
        s = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        s %= 2**31 - 1
        print(f"seed = {s}   (reproduce with --seed {s})")
    else:
        s = int(args.seed)

    backend_spec = args.backend
    if isinstance(backend_spec, str) and backend_spec.strip().startswith("{"):
        import yaml
        backend_spec = yaml.safe_load(backend_spec)

    # Fitter is the composition root: it owns the parameter layer (model +
    # constraints) AND the configured backend for the density computation.
    f = Fitter(args.config, backend=backend_spec)
    if row_block_factors(f.config.dic)[2] != 1:
        raise SystemExit("gen_toy_pwa is pure-PWA only (n_blocks != 1: "
                         "identical/cp particles declared)")
    f.apply_constrains()
    model, kc = f.model, f.kernel_config
    tree = f.decay_tree

    chain = tree.partial_waves()[0][1]
    byt = {tree.topo_index[ch.topo_id()]: ch
           for _, ch in tree.partial_waves()}

    # ── independent flat samples ───────────────────────────────────────────
    phsp = pwa_event_data_tree(model, kc, byt,
                               generate_pwa_phsp(model, chain, args.nph,
                                                 seed=s))
    prop = pwa_event_data_tree(model, kc, byt,
                               generate_pwa_phsp(model, chain, args.nprop,
                                                 seed=s + 1))

    # Parameter point from the Fitter parameter layer (constraints/defaults):
    # ck is random (symmetric start), m0/g0 come from the config defaults.
    x0 = f.initial_values(seed=s + 2)
    params, resolved = f.build_params(x0)

    # Density via the configured Backend (norm=None -> unnormalised P/e).
    backend = f.backend
    _, _, P0 = backend.compute(params, backend.load_data(prop),
                               norm=None, return_p=True)
    if P0 is None:
        raise SystemExit("backend compute returned no P")
    prob = np.clip(np.asarray(P0, dtype=float), 0, None)
    prob /= prob.sum()

    rng = np.random.RandomState(s + 3)
    idx = rng.choice(args.nprop, size=args.ndata, replace=False, p=prob)
    data = {kk: vv[idx] for kk, vv in prop.items()}

    np.savez(args.out_data, **_npz(data))
    np.savez(args.out_phsp, **_npz(phsp))
    print(f"wrote {args.out_data} ({args.ndata} events), "
          f"{args.out_phsp} ({args.nph} events)")

    # ── start point for run_fit --init, same source as the generated data ─
    f.save_params(x0, args.out_init)
    print(f"wrote {args.out_init} (start point = generating point)")


if __name__ == "__main__":
    main()
