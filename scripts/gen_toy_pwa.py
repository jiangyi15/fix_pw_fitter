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
accepted; ``--max-count`` (default 0 = no cap) is an optional hard cap, and the
runaway guard is ``--stall-loops`` (consecutive loops that accept NOTHING), so a
peaked amplitude is never stopped for being slow.  ``--cache-dir DIR`` checkpoints
every accepted batch and resumes from it (pass the same ``--seed``), which is how
a long generation survives a wall-clock kill.

``--seed`` is a master seed: the sub-seeds are ``seed+1+loop`` (proposal
batches), ``seed+2`` (the parameter point, unused with ``--params``) and
``seed+3`` (event selection); the norm sample uses ``seed``.  Omit it and a random master seed is drawn and
printed, so a run can be reproduced with ``--seed <printed>``.

The written file is a PURE-SIGNAL sample with no ``bkg`` column (the loader
fills the uniform shape), so fit it at purity 1.  For a signal+background toy,
or for a background-shaped ``bkg`` column, call the library instead:
``Fitter.generate_toy(fbg=…)`` / ``Fitter.generate_toy_bg(bg=…)`` and
``Fitter.generate_phsp(bkg=…)`` with the same function.
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

import numpy as np

from tabpwa import Fitter
from tabpwa.phasespace import fix_beam
from tabpwa.toy_pwa import toy_pwa_generate


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
    ap.add_argument("--max-count", type=int, default=0,
                    help="OPTIONAL hard cap on the number of proposal loops "
                         "(0 = no cap); the loop stops as soon as --ndata are "
                         "accepted")
    ap.add_argument("--stall-loops", type=int, default=1000,
                    help="runaway guard: raise after this many CONSECUTIVE "
                         "loops that accept nothing (a peaked amplitude runs as "
                         "long as it keeps accepting)")
    ap.add_argument("--cache-dir", default=None,
                    help="checkpoint the accepted pool here and RESUME from it "
                         "on the next call (needs an explicit --seed); a "
                         "wall-clock kill or a crash then loses nothing")
    ap.add_argument("--verbose", action="store_true",
                    help="print a line for EVERY loop (the ceiling moves); "
                         "progress lines are automatic on a terminal")
    ap.add_argument("--batch-size", type=int, default=20000,
                    help="flat phase-space events generated per loop")
    ap.add_argument("--maxp-scale", type=float, default=1.1,
                    help="guess max P = scale * max(batch P)")
    ap.add_argument("--maxp", type=float, default=None,
                    help="initial ceiling on P, e.g. the 'max P' printed by a "
                         "previous run at this parameter point; it is still "
                         "raised if a batch exceeds it")
    ap.add_argument("--top-p", nargs=3, type=float, default=None,
                    metavar=("PX", "PY", "PZ"),
                    help="the top's momentum in GeV/c (e.g. 0.034 0 0 for the "
                         "BESIII crossing angle); default: the rest frame")
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

    beam = fix_beam(*args.top_p) if args.top_p else None
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
    # Parameter point: from --params (save_params JSON), else from the Fitter
    # parameter layer (ck random symmetric start, m0/g0 at config defaults).
    if args.params:
        import json
        with open(args.params) as fh:
            # free values -> x AND fixed values -> the constraint layer, so the
            # toy truth matches the source fit including its fixed parameters
            x0 = f.load_params(json.load(fh))
        print(f"params      : {args.params}")
    else:
        x0 = f.initial_values(seed=s + 2)

    # ── the accept-reject loop lives in toy_pwa.toy_pwa_generate ──────────
    # (one implementation; the script only parses, calls and writes)
    data, stats = toy_pwa_generate(
        f, args.ndata, seed=s, batch_size=batch,
        maxp_scale=args.maxp_scale, max_loops=max_loops,
        stall_loops=args.stall_loops, x0=x0, top_p=beam, maxp=args.maxp,
        verbose=args.verbose, cache_dir=args.cache_dir)

    # ── the norm sample is a SEPARATE call (the toy needs only P and max P) ──
    phsp = f.generate_phsp(args.nph, seed=s, top_p=beam) if args.nph > 0 else None

    np.savez(args.out_data, **_npz(data))
    if phsp is not None:
        np.savez(args.out_phsp, **_npz(phsp))
        print(f"wrote {args.out_data} ({len(data['mass'])} events), "
              f"{args.out_phsp} ({len(phsp['mass'])} events)")
    else:
        print(f"wrote {args.out_data} ({len(data['mass'])} events); "
              f"no phsp (--nph 0)")

    # ── start point for run_fit --init, same source as the generated data ─
    f.save_params(stats.truth, args.out_init)
    print(f"wrote {args.out_init} (start point = generating point)")
    print(f"max P       : {stats.maxp:.6g}   "
          f"(reuse with --maxp {stats.maxp:.6g})")


if __name__ == "__main__":
    main()
