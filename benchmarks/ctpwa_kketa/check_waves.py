#!/usr/bin/env python3
"""One-hot scan: is every partial wave alive in the generated sample?

    python check_waves.py                        # config_gen.yml, 2000 events
    python check_waves.py --config config.yml --n 500

For each wave it zeroes every coupling except that one and sums the per-event
intensity.  A **dead** wave returns exactly 0.  That is the failure mode this
guards: a decay chain whose top vertex resolves to the wrong leaf (e.g.
`jpsi -> Kp + R(Kp eta)`, where K+ then appears twice) gives `q = 0` for every
event of that chain, and the waves built on it vanish — while the sample is
still finite, still inside every grid, and still times plausibly.

The flat config (`pwa_gen`) has one coefficient row per partial wave, so it is
the natural place to count them; it is also the same sample the `pwa` config
reads, so a live scan here covers both.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from tabpwa import Fitter            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config_gen.yml"))
    ap.add_argument("--backend", default=None,
                    help="default: the reference backend of the config's model")
    ap.add_argument("--n", type=int, default=2000, help="data events to scan")
    ap.add_argument("--tol", type=float, default=1e-12,
                    help="a wave is dead below tol x the strongest wave")
    a = ap.parse_args()

    f = Fitter(a.config, backend=a.backend) if a.backend else Fitter(a.config)
    f.set_phsp(f.load_dataset("phsp", max_events=200))
    f.set_data(f.load_dataset("data", max_events=a.n))
    f.apply_constrains()
    x0 = f.initial_values(seed=1)
    params, _resolved = f.build_params(x0)

    key = "ck" if "ck" in params else "coeff"
    base = np.array(params[key])
    handle = getattr(f, "_data_holder", None) or f.backend.load_data(
        f.load_dataset("data", max_events=a.n))

    tot = np.empty(base.shape[0])
    for w in range(base.shape[0]):
        p = dict(params)
        c = np.zeros_like(base)
        c[w] = 1.0
        p[key] = c
        _Q, _g, P = f.backend.compute(p, handle, norm=1.0, return_p=True)
        tot[w] = float(np.sum(np.abs(P)))

    scale = tot.max() if tot.size and tot.max() > 0 else 1.0
    alive = tot > a.tol * scale
    print(f"{os.path.basename(a.config)}: {tot.size} waves, {int(alive.sum())} alive "
          f"(min |P| sum {tot.min():.3e}, max {scale:.3e}, {a.n} events)")
    dead = np.flatnonzero(~alive)
    if dead.size:
        raise SystemExit(f"DEAD waves: {dead.tolist()} — check the decay topology "
                         f"(a charge sector has collapsed)")


if __name__ == "__main__":
    main()
