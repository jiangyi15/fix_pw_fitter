"""Build and save an ONNX model with realistic dimensions for benchmarking.

Usage:
    python scripts/export_example_onnx.py [--waves 200] [--events 1000] [--output model.onnx]
"""

import argparse
import os
import numpy as np
import sys
sys.path.insert(0, "src")

from interp_fitter.onnx_model import build_onnx_model, export_to_onnx


def make_benchmark_config(nwaves=200, n_m0=50, n_g0=50,
                          n_gamma=200, n_bw=400, nres=2,
                          n_fl=50, ndec=2,
                          nbasis=50, n_ang=50, n_per=3):
    """Create a config dict with given dimensions."""

    n_int = 200  # interpolation table size

    config = {
        # -- tables --
        "gamma_table": np.ones((n_gamma, n_int), dtype=complex),
        "fl_table":    np.ones((n_fl, n_int), dtype=float),

        # -- mapping matrices --
        "matrix_gamma": np.ones((n_m0, n_gamma), dtype=float),
        # (nbasis, nwaves) — must match nwaves
        "matrix_ang": np.ones((nbasis, nwaves), dtype=complex),

        # -- gamma indexers --
        "gamma_type":  np.zeros(n_gamma, dtype=int),
        "gamma_index": np.zeros(n_gamma, dtype=int),
        "gamma_min":   0.0,
        "gamma_delta": 0.01,
        "g0_index":    np.zeros(n_gamma, dtype=int),

        # -- BW indexers --
        "m0_index":       np.zeros(n_bw, dtype=int),
        "bw_index":       np.zeros(n_bw, dtype=int),
        "bw_gamma_index": np.zeros(n_bw, dtype=int),
        "bw_order":       np.tile(np.arange(n_bw), nwaves * nres // n_bw + 1)[:nwaves * nres],

        # -- form factor indexers --
        "q_index":  np.zeros(n_fl, dtype=int),
        "fl_type":  np.zeros(n_fl, dtype=int),
        "fl_min":   0.0,
        "fl_delta": 0.01,
        "fl_order": np.zeros(nwaves * ndec, dtype=int),

        # -- angular indexers --
        "angle_index": np.arange(n_ang, dtype=int),
        "angle_k":     np.ones(n_ang, dtype=float),
        "angle_b":     np.zeros(n_ang, dtype=float),
        "ang_order":   np.tile(np.arange(n_ang), (nbasis, n_per))[:, :n_per],
    }

    # Ensure bw_order is within bounds
    config["bw_order"] = np.random.default_rng(0).integers(0, n_bw,
                                                           size=nwaves * nres).astype(np.int64)

    return config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--waves", type=int, default=200)
    p.add_argument("--events", type=int, default=1000)
    p.add_argument("--output", default="kernel.onnx")
    args = p.parse_args()

    print(f"Building config: nwaves={args.waves}, nbasis=50, n_bw=400, n_m0=50, n_fl=50")
    config = make_benchmark_config(nwaves=args.waves)

    base, ext = os.path.splitext(args.output)

    for variant, label in [("with_norm", True), ("no_norm", False)]:
        out = f"{base}_{variant}{ext}"
        print(f"Building {variant} …")
        model = build_onnx_model(config, with_norm=label)
        with open(out, "wb") as f:
            f.write(model.SerializeToString())
        size_mb = os.path.getsize(out) / 1e6
        print(f"  → {out}  ({len(model.graph.node)} nodes, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
