#!/usr/bin/env python3
"""Benchmark ``pwa`` vs ``pwa_gen`` on the CPU backends (gate E).

Shape template: ``tests/config_pwa_gen.yml`` (one partial wave) tiled to the
198-wave shape (``n_proj=2``, ``N=99``).  Three kernel configs:

* ``pwa``            — the plain projection-sum PWA (198 waves);
* ``pwa_gen N≡1``    — the same wave count plus the numerator machinery with
                       an all-zero ``n_index`` (must cost ~the same);
* ``pwa_gen N=[1,m²]`` — the numerator rows doubled (396 waves) with a
                       two-row numerator table (``1``, ``m²``) and the
                       generic coefficient basis (``den_row``/``den_table``).

Measured: ``compute(norm=<scalar>, return_p=True)`` (data NLL + gradients),
i.e. **kernel-only**.  Caveat for reading the table: ``pwa_gen``'s
``params["coeff"]`` is assembled by the fitter-side ``PWAGenKernelParams``
*before* ``compute`` and is therefore NOT in the timed region, while the
plain backends derive ``s0``/``mult`` from ``m0``/``g0`` *inside* ``compute``.
The rows are kernel-vs-kernel, not end-to-end; for the user-facing cost see
``tests/benchmark_pwa_gen_s0e.py`` (``Fitter.get_nll``) and the ``N≡1``
bit-identity gate (same kernel config -> same code path).
the realistic per-iteration call.  Physics is irrelevant (random params).

Usage (repo root)::

    python tests/benchmark_pwa_gen.py                 # 65k / 262k
    python tests/benchmark_pwa_gen.py --sizes 10000,100000 --trials 3
"""
import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from benchmark_pwa_kernels import (PRESETS, _bench, _rate, _time, tile_case)
from tabpwa.amp_model import build_amplitude_model
from tabpwa.backends import create_backend

TEMPLATE = "tests/config_pwa_gen.yml"


def _plain_case(n_comp, n_proj):
    """Plain PWA shape template: ``n_wave = n_proj · N``."""
    base = build_amplitude_model(TEMPLATE).build_kernel_config()
    N0 = int(base["matrix_angle"].shape[1]) // int(base.get("n_proj", 1) or 1)
    k = max(1, int(round(n_comp / N0)))
    return tile_case(base, n_proj, k), N0 * k


def _gen_case(kc):
    """Tiled config + the pwa_gen numerator / coefficient basis.

    Every wave gets a numerator row (``0``/``1`` alternating) and the generic
    denominator basis ``[1, −m², m⁴]`` — the same slot count as the plain
    ``[1, −m², −iγ]`` preset, so only the wave count and the numerator
    interpolation are new.
    """
    out = copy.copy(kc)
    n_wave = int(kc["matrix_angle"].shape[1])
    n_bw = len(np.asarray(kc["mass_index"]))
    grid_n = int(np.asarray(kc["gamma_table"]).shape[-1])
    grid = (float(kc["gamma_min"])
            + float(kc["gamma_delta"]) * np.arange(grid_n))
    out["n_table"] = np.stack([np.ones(grid_n, complex), grid ** 2 + 0j])
    out["n_min"] = float(kc["gamma_min"])
    out["n_delta"] = float(kc["gamma_delta"])
    out["n_index"] = np.tile(np.array([0, 1], np.int32), n_wave // 2)
    out["n_mass_index"] = np.asarray(kc["mass_index"])[
        np.asarray(kc["bw_order"]).reshape(-1)].astype(np.int32)
    out["den_row"] = np.tile(np.array([[0, 1, 2]], np.int32), (n_bw, 1))
    out["den_table"] = (grid ** 4 + 0j)[None, :]
    out["den_min"] = float(kc["gamma_min"])
    out["den_delta"] = float(kc["gamma_delta"])
    return out


def _events(kc, ne, rng):
    gmin = float(kc["gamma_min"])
    gspan = float(kc["gamma_delta"]) * np.asarray(kc["gamma_table"]).shape[-1]
    n_mass = int(np.max(kc["mass_index"])) + 1
    n_mom = int(np.max(kc["fl_q_index"])) + 1
    n_ang = (int(np.max(kc["angle_index"])) + 1
             if len(kc["angle_index"]) > 0 else 0)
    n_ac = int(np.asarray(kc["angle_k"]).shape[-1])
    return {
        "mass": rng.uniform(gmin + 0.05 * gspan, gmin + 0.95 * gspan,
                            (ne, n_mass)),
        "q": rng.uniform(0.0, 1.0, (ne, n_mom)),
        "angle": rng.uniform(-np.pi, np.pi, (ne, n_ang, n_ac)),
        "weight": np.ones(ne),
        "bkg": np.full(ne, 0.01),
    }


def _params(kc, rng, generic):
    n_wave = int(kc["matrix_angle"].shape[1])
    P = int(kc.get("n_proj", 1) or 1)
    n_res = np.asarray(kc["bw_order"]).size // n_wave
    params = {
        "ck": rng.normal(size=n_wave // P) + 1j * rng.normal(size=n_wave // P),
        "m0": rng.uniform(2.0, 3.0, int(np.max(kc["m0_index"])) + 1),
        "g0": rng.uniform(0.05, 0.2, int(np.max(kc["g0_index"])) + 1),
    }
    if generic:
        params["coeff"] = rng.uniform(
            0.2, 1.5, (n_wave * n_res, np.asarray(kc["den_row"]).shape[1]))
    return params


def run(sizes, n_comp, n_proj, warmup, trials, backends):
    kc_plain, N = _plain_case(n_comp, n_proj)
    kc_gen1 = _gen_case(kc_plain)
    kc_gen1["n_index"] = np.zeros(np.asarray(kc_gen1["n_index"]).shape,
                                  np.int32)               # N ≡ 1
    kc_tiled, _ = _plain_case(2 * n_comp, n_proj)
    kc_gen2 = _gen_case(kc_tiled)

    cases = [
        ("pwa", kc_plain, False),
        ("pwa 2x waves", kc_tiled, False),
        ("pwa_gen N=1", kc_gen1, True),
        ("pwa_gen N=[1,m^2]", kc_gen2, True),
    ]
    print("  kernel-only (generated coeff for pwa_gen is assembled outside "
          "the timed call; see the module docstring)")
    print(f"wave shape: n_wave = {kc_plain['matrix_angle'].shape[1]} (pwa, "
          f"N={N}, n_proj={n_proj});  "
          f"{kc_gen2['matrix_angle'].shape[1]} (N=[1,m^2])")
    header = f"{'n_events':>9}" + "".join(f" | {c:>24}" for c, _, _ in cases)
    for label, spec in backends:
        print("=" * len(header))
        print(f"  data NLL + gradients — backend {label}")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for ne in sizes:
            rng = np.random.default_rng(11)
            cells = []
            for _name, kc, generic in cases:
                data = _events(kc, ne, rng)
                params = _params(kc, rng, generic)
                be = create_backend(spec, kc, model="pwa_gen"
                                    if generic else "pwa")
                dh = be.load_data(data)
                t = _bench(lambda: be.compute(params, dh, norm=float(ne) * .5,
                                              return_p=True), warmup, trials)
                cells.append(f"{_rate(ne, t)}  {_time(t)}")
                del dh, be
            print(f"{ne:>9}" + "".join(f" | {c:>24}" for c in cells))
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=None,
                    help="comma separated event counts (default: the large preset)")
    ap.add_argument("--preset", default="large", choices=sorted(PRESETS))
    ap.add_argument("--n-comp", type=int, default=99,
                    help="N (shared-ck length per projection)")
    ap.add_argument("--n-proj", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--backends", default="cpu_pwa_s0c,numpy_pwa")
    args = ap.parse_args()

    sizes = ([int(x) for x in args.sizes.split(",")] if args.sizes
             else PRESETS[args.preset][:2])
    labels = {"cpu_pwa_s0c": "cpu_pwa_s0c", "cpu_pwa_gen": "cpu_pwa_gen",
              "cpu_v5_pwa": "cpu_v5_pwa", "numpy_pwa": "numpy_pwa"}
    backends = [(labels[b], b) for b in args.backends.split(",") if b]
    run(sizes, args.n_comp, args.n_proj, args.warmup, args.trials, backends)


if __name__ == "__main__":
    main()
