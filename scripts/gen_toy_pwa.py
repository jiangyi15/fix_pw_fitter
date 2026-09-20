#!/usr/bin/env python3
"""Generate pure-PWA toy data for the generic run_fit driver.

Writes a kernel-format ``.npz`` data file plus an ``init.json`` start point
(and, only when ``--nph > 0``, a flat phase-space norm ``.npz``):

    python scripts/gen_toy_pwa.py --config config_pwa.yml        # data + init
    python run_fit.py --config config_pwa.yml --backend numpy_pwa \\
        --data data_pwa.npz --init init_pwa.json --fit

    python scripts/gen_toy_pwa.py --config config_pwa.yml --nph 40000   # + phsp

Flat phase space is generated with the config masses (two-body product +
inverse boost chain).  Accept with probability ``P(x)/guess``, where the
guess of the maximum ``P`` starts at ``maxp-scale · max(batch P)`` and is
raised the same way only when a batch exceeds it; the already-accepted events
are then retrospectively **thinned** with probability
``old_guess/new_guess``.  The loop stops as soon as ``--ndata`` are accepted
(no fixed proposal pool).

The density uses the **Fitter parameter layer** (constraints/defaults via
``initial_values``/``build_params``) and an actual **Backend**
(``--backend``, default ``numpy_pwa``) — never the raw kernel directly.

The generating parameter point is either drawn from the Fitter parameter
layer (``initial_values``: ck random, m0/g0 at their config defaults) or read
from a saved parameter file with ``--params`` (``save_params`` JSON, e.g. a fit
result).  In both cases the **same** point is written to ``init.json``, so
``run_fit --init init.json`` starts exactly at the truth.

Flat events are generated and consumed in batches (``--batch-size``), so peak
memory is O(batch + ndata).  The loop stops as soon as ``--ndata`` are
accepted; ``--max-count`` (default 1000, ``0`` = unlimited) only guards
against a runaway loop.

``--seed`` is a master seed: the sub-seeds are ``seed`` (phsp), ``seed+1``
(proposal), ``seed+2`` (parameter point, unused with ``--params``) and
``seed+3`` (event selection).  Omit it and a random master seed is drawn and
printed, so a run can be reproduced with ``--seed <printed>``.
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

import numpy as np

from tabpwa import Fitter
from tabpwa.pwa_build import build_tree_event_data, generate_pwa_phsp


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
    ap.add_argument("--params", default=None,
                    help="JSON (save_params format) giving the generating "
                         "parameter point, e.g. a fit result; default: the "
                         "Fitter's random initial_values")
    ap.add_argument("--seed", type=int, default=None,
                    help="master seed; sub-seeds are seed, +1 (phsp/proposal), "
                         "+2 (params), +3 (event selection).  Omitted -> draw "
                         "a random seed and print it")
    ap.add_argument("--nph", type=int, default=0,
                    help="flat phase-space (norm) events to write to "
                         "--out-phsp; 0 = skip (supply data.phsp in the "
                         "config, or generate it separately)")
    ap.add_argument("--ndata", type=int, default=4000)
    ap.add_argument("--max-count", type=int, default=1000,
                    help="guard on the number of loops (0 = unlimited); the "
                         "loop stops as soon as --ndata are accepted")
    ap.add_argument("--batch-size", type=int, default=20000,
                    help="flat phase-space events generated per loop")
    ap.add_argument("--maxp-scale", type=float, default=1.1,
                    help="guess max P = scale * max(batch P)")
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

    if args.ndata < 1:
        raise SystemExit("--ndata must be >= 1")
    batch = max(1, int(args.batch_size))
    # --max-count only guards against a runaway loop (0 = unlimited); the
    # loop stops as soon as --ndata events are accepted.
    max_loops = int(args.max_count) if args.max_count > 0 else None

    # Fitter is the composition root: it owns the parameter layer (model +
    # constraints) AND the configured backend for the density computation.
    f = Fitter(args.config, backend=backend_spec)
    f.apply_constrains()
    model, kc = f.model, f.kernel_config
    tree = f.decay_tree

    chain = tree.partial_waves()[0][1]
    byt = {tree.topo_index[ch.topo_id()]: ch
           for _, ch in tree.partial_waves()}

    # ── normalisation sample (optional; independent of the proposal stream)
    phsp = None
    if args.nph > 0:
        phsp = build_tree_event_data(
            model, kc, byt,
            generate_pwa_phsp(model, chain, args.nph, seed=s))

    # Parameter point: from --params (save_params JSON), else from the Fitter
    # parameter layer (ck random symmetric start, m0/g0 at config defaults).
    if args.params:
        import json
        with open(args.params) as fh:
            x0 = f.values_from_dict(json.load(fh))
        print(f"params      : {args.params}")
    else:
        x0 = f.initial_values(seed=s + 2)
    params, resolved = f.build_params(x0)

    # ── adaptive accept–reject over a streamed flat phase space ───────────
    # Guess the maximum P and accept x with probability P(x)/guess.  The
    # guess starts at scale*max(batch P) and is raised (to scale*max(batch P))
    # only when a batch exceeds it; the accepted events are then
    # retrospectively THINNED with probability old_guess/new_guess.  The loop
    # stops as soon as --ndata are accepted; --max-count only guards a
    # runaway loop.
    backend = f.backend
    rng = np.random.RandomState(s + 3)
    store = ("mass", "q", "angle", "weight")
    cols = {k: [] for k in store}
    have = tried = loops = 0
    guess = None                               # guess of max P
    limit = max_loops if max_loops is not None else float("inf")
    scale = float(args.maxp_scale)
    print(f"sampling {args.ndata} events: <= {batch}/loop, "
          f"max loops {max_loops if max_loops else 'inf'}, scale {scale:g}")
    while have < args.ndata and loops < limit:
        ev = build_tree_event_data(
            model, kc, byt,
            generate_pwa_phsp(model, chain, batch, seed=s + 1 + loops))
        _, _, P = backend.compute(params, backend.load_data(ev),
                                  norm=None, return_p=True)
        if P is None:
            raise SystemExit("backend compute returned no P")
        P = np.asarray(P, dtype=float)
        bmax = float(P.max()) if P.size else 0.0

        if guess is None:
            guess = scale * bmax
            print(f"  loop {loops}: guess max P = {guess:.6g}")
        elif bmax > guess:                        # exceeded -> raise + thin
            new = scale * bmax
            before = have
            if have:                              # keep each with old/new
                mask = rng.random(have) < (guess / new)
                for k in store:
                    cols[k] = [np.concatenate(cols[k])[mask]]
                have = int(mask.sum())
            guess = new
            print(f"  loop {loops}: guess max P -> {guess:.6g} "
                  f"(kept {have}/{before} = old/new {0 if not before else (have / before):.2f})")

        keep = ((rng.random(batch) * guess <= P) if guess and guess > 0
                else np.zeros(batch, dtype=bool))
        tried += batch
        nk = int(keep.sum())
        if nk:
            idx = np.flatnonzero(keep)[:args.ndata - have]
            for k in store:
                cols[k].append(ev[k][idx])
            have += len(idx)
        loops += 1
    if have < args.ndata:
        raise SystemExit(
            f"accepted only {have} of {args.ndata} events in {loops} loops "
            f"({tried} tried); raise --max-count or --batch-size")
    data = {k: np.concatenate(cols[k]) for k in store}
    print(f"accepted {have} events in {loops} loops "
          f"({have}/{tried} = {have / max(1, tried):.1%}, "
          f"guess max P = {guess:.6g})")

    np.savez(args.out_data, **_npz(data))
    if phsp is not None:
        np.savez(args.out_phsp, **_npz(phsp))
        print(f"wrote {args.out_data} ({have} events), "
              f"{args.out_phsp} ({args.nph} events)")
    else:
        print(f"wrote {args.out_data} ({have} events); "
              f"no phsp (--nph 0)")

    # ── start point for run_fit --init, same source as the generated data ─
    f.save_params(x0, args.out_init)
    print(f"wrote {args.out_init} (start point = generating point)")


if __name__ == "__main__":
    main()
