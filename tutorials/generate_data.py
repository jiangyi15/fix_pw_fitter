#!/usr/bin/env python3
"""Generate a small pure-PWA toy + phase-space sample for the tutorial.

Wraps ``scripts/gen_toy_pwa.py`` with tutorial-local defaults:

    data_arr.npz   (ndata toy events, kernel arrays)
    phsp_arr.npz   (nph phase-space events, kernel arrays)
    init_pwa.json  (constraint-driven start point for run_fit --init)

Run it once, then ``./fit.sh`` (or run_fit directly) fits the toy.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ndata", type=int, default=800)
    ap.add_argument("--nph", type=int, default=2000)
    ap.add_argument("--nprop", type=int, default=8000)
    args = ap.parse_args()

    cmd = [sys.executable, os.path.join(ROOT, "scripts", "gen_toy_pwa.py"),
           "--config", os.path.join(HERE, "config.yml"),
           "--ndata", str(args.ndata), "--nph", str(args.nph),
           "--nprop", str(args.nprop),
           "--out-data", os.path.join(HERE, "data_arr.npz"),
           "--out-phsp", os.path.join(HERE, "phsp_arr.npz"),
           "--out-init", os.path.join(HERE, "init_pwa.json")]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
