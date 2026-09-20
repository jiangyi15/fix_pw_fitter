#!/usr/bin/env python3
"""Benchmark the pure-PWA kernel family on a synthetic kernel config.

No full amplitude model is built: a small real kernel config is used only as a
*shape template*, then its wave axis is tiled to a chosen number of components
``n_comp`` (the shared-ck length ``N``).  The physics is irrelevant — every
array keeps the shape the kernels expect — so this reproduces a realistic
fixed-shape workload (compare ``project37``'s `F_{ijk}` / n_comp).

Backends: ``numpy_pwa`` (reference), ``cuda_v4_pwa``, ``cuda_v4_pwa_cache``,
``cuda32_v4_pwa_cache`` (fp32), ``cuda_v5_pwa`` and ``integrated_pwa``.

Sections:

* ``P`` / ``NLL+grad`` — per backend, same random event data.
  ``P`` = ``compute(norm=None, return_p=True)`` (per-event amplitude only);
  ``NLL+grad`` = ``compute(norm=<scalar>, return_p=True)`` (data NLL + full
  ck/m0/g0 gradient, background included).  The norm is a scalar input here
  (precomputed) — the same convention as ``project37``'s precomputed ``M``.
* ``full NLL`` — ``integrated_pwa(base=<amp cache>)``: the Gram norm over the
  phsp bundle (``compute(phsp, norm=None)``; ``D`` is pre-integrated once) plus
  ``compute(data, norm=norm)``, reported as norm / data / full.

Cells show SI-scaled throughput and per-call wall time.  Unavailable backends
(no CUDA/plugin) are reported as ``SKIP``.  ``numpy_pwa`` (CPU) is skipped by
default — add ``--include-cpu`` to include it.

Usage (from the repo root)::

    python tests/benchmark_pwa_kernels.py                      # n_comp=96, small
    python tests/benchmark_pwa_kernels.py --n-comp 100 --n-proj 2 \
        --preset large --trials 3
    python tests/benchmark_pwa_kernels.py --n-comp 100 --n-proj 2 --preset large \
        --compute-sizes 100000,1000000 --warmup 1 --trials 5

``--n-comp`` is ``N`` — the shared-ck length / components **per projection**
(``n_wave = n_proj · N`` total); ``--n-proj`` sets the projection count to
match a reference (e.g. 2).  The actual ``N`` is ``N0 · round(n_comp / N0)``
and both ``N`` and the total are printed.
"""
import argparse
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tabpwa.amp_model import build_amplitude_model
from tabpwa.backends import create_backend

PRESETS = {
    "small": [1024, 4096, 10000],
    "large": [65536, 262144, 1048576],
}

BACKENDS = [
    ("numpy_pwa",           "numpy_pwa"),
    ("cuda_v4_pwa",         "cuda_v4_pwa"),
    ("cuda_v4_pwa_cache",   "cuda_v4_pwa_cache"),
    ("cuda_v4_pwa_cache32", "cuda32_v4_pwa_cache"),
    ("cuda_v5_pwa",         "cuda_v5_pwa"),
    ("integrated_pwa",      {"name": "integrated_pwa", "base": "cuda_v4_pwa"}),
]

# The CPU reference is skipped by default (slow at 10^5-10^6 and not the
# like-for-like CPU of a precomputed-F matvec); re-enable with --include-cpu.
CPU_BACKENDS = {"numpy_pwa"}


# ── synthetic kernel config ────────────────────────────────────────

def _fit_proj(a, axis, n_proj):
    """Repeat/truncate the projection axis to exactly *n_proj*."""
    P0 = a.shape[axis]
    if n_proj == P0:
        return a
    reps = int(np.ceil(n_proj / P0))
    return np.repeat(a, reps, axis=axis)[
        tuple(slice(None) if i != axis else slice(0, n_proj)
              for i in range(a.ndim))]


def tile_case(kc, n_proj, k):
    """Set projections to *n_proj* and tile the wave axis k times.

    Result: ``n_wave = n_proj · N`` with ``N = N0·k`` (shared-ck length).
    Projections are duplicated/truncated (the template's projection columns
    are identical copies by design), then each projection's ``N0`` waves are
    tiled k times — only per-wave arrays grow, so the index tables and the
    unique-parameter arrays (m0/g0/fl/angle) stay valid.
    """
    if int(kc.get("n_blocks", 1)) != 1:
        raise ValueError("tile_case expects a single block (n_blocks == 1); "
                         "use a config without identical/CP declarations")
    n_proj, k = int(n_proj), int(k)
    if n_proj < 1 or k < 1:
        raise ValueError("n_proj and k must be >= 1")

    ma = np.asarray(kc["matrix_angle"])
    B, n_wave = ma.shape
    P0 = int(kc.get("n_proj", 1) or 1)
    N = n_wave // P0
    n_res = np.asarray(kc["bw_order"]).size // n_wave
    n_decay = np.asarray(kc["fl_order"]).size // n_wave

    ma = _fit_proj(ma.reshape(B, P0, N), 1, n_proj)          # (B, n_proj, N)
    bw = _fit_proj(np.asarray(kc["bw_order"]).reshape(P0, N, n_res), 0, n_proj)
    flo = _fit_proj(np.asarray(kc["fl_order"]).reshape(P0, N, n_decay), 0, n_proj)

    out = copy.copy(kc)
    out["n_proj"] = n_proj
    out["matrix_angle"] = np.tile(ma, (1, 1, k)).reshape(B, n_proj * N * k)
    out["bw_order"] = np.tile(bw, (1, k, 1)).reshape(-1)
    out["fl_order"] = np.tile(flo, (1, k, 1)).reshape(-1)

    ck_map = list(kc.get("ck_map", []))
    if len(ck_map) == N:
        out["ck_map"] = ck_map * k
    return out


def synthetic_case(config, n_comp, n_proj, n_events, n_phsp=0, seed=123):
    """(kc, data, params, norm, info) with ``n_comp = N`` shared-ck length.

    ``N`` is the number of components **per projection** (the shared-ck
    length); the total per-event wave entries are ``n_wave = n_proj · N``.
    Projections are matched to *n_proj* (None -> the template's) and the
    wave axis is tiled so the actual ``N`` is ``N0 · round(n_comp / N0)``.
    """
    base = build_amplitude_model(config).build_kernel_config()
    P0 = int(base.get("n_proj", 1) or 1)
    N0 = int(base["matrix_angle"].shape[1]) // P0
    P = int(n_proj) if n_proj else P0
    k = max(1, int(round(n_comp / N0)))
    kc = tile_case(base, P, k)

    rng = np.random.default_rng(seed)
    n_mass = int(np.max(kc["mass_index"])) + 1
    n_mom = int(np.max(kc["fl_q_index"])) + 1
    n_ang = (int(np.max(kc["angle_index"])) + 1
             if len(kc["angle_index"]) > 0 else 0)
    n_comp_ang = int(kc["angle_k"].shape[-1])
    def _events(ne):
        return {
            "mass": rng.uniform(2.0, 3.0, (ne, n_mass)),
            "q": rng.uniform(0.0, 1.0, (ne, n_mom)),
            "angle": rng.uniform(-np.pi, np.pi, (ne, n_ang, n_comp_ang)),
            "weight": np.ones(ne),
            "bkg": np.full(ne, 0.01),
        }

    data = _events(n_events)
    phsp = _events(n_phsp) if n_phsp > 0 else None
    N = int(N0 * k)
    params = {
        "ck": rng.normal(size=N) + 1j * rng.normal(size=N),
        "m0": rng.uniform(2.0, 3.0, int(np.max(kc["m0_index"])) + 1),
        "g0": rng.uniform(0.05, 0.2, int(np.max(kc["g0_index"])) + 1),
    }
    norm = float(n_events) * 0.5          # a plausible precomputed norm scalar
    info = {"N0": N0, "N": N, "k": k, "n_proj": P,
            "n_wave": int(kc["matrix_angle"].shape[1])}
    return kc, data, phsp, params, norm, info


# ── timing helpers ─────────────────────────────────────────────────

def _bench(fn, warmup, trials):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(trials):
        fn()
    return (time.perf_counter() - t0) / trials


def _rate(events, seconds):
    r = events / seconds
    for div, suffix in ((1e9, "G/s"), (1e6, "M/s"), (1e3, "k/s")):
        if r >= div:
            return f"{r / div:>7.3g} {suffix}"
    return f"{r:>7.3g} /s"


def _time(seconds):
    if seconds >= 1.0:
        return f"{seconds:>7.3g} s "
    return f"{seconds * 1e3:>7.3g} ms"


def _table(title, cols, sizes, run):
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)
    header = f"{'n_events':>9}" + "".join(f" | {c:>22}" for c in cols)
    print(header)
    print("-" * len(header))
    for n in sizes:
        cells = run(n)
        print(f"{n:>9}" + "".join(f" | {c:>22}" for c in cells))
    print()


def bench(kc, data_max, params, norm, sizes, warmup, trials, backends):
    cols = [label for label, _ in backends]

    def make_run(use_norm):
        def run(n):
            data = {k: v[:n] for k, v in data_max.items()}
            cells = []
            for _, spec in backends:
                try:
                    be = create_backend(spec, kc, model="pwa")
                    dh = be.load_data(data)
                    nrm = norm if use_norm else None
                    t = _bench(lambda: be.compute(params, dh, norm=nrm,
                                                  return_p=True), warmup, trials)
                    del dh, be
                    cells.append(f"{_rate(n, t)}  {_time(t)}")
                except Exception:
                    cells.append(f"{'SKIP':>22}")
            return cells
        return run

    _table("P only   compute(norm=None, return_p=True)   [throughput · ms/call]",
           cols, sizes, make_run(False))
    _table("NLL+grad compute(norm=<scalar>, return_p=True)   "
           "[throughput · ms/call]", cols, sizes, make_run(True))


def bench_full_nll(kc, data_max, phsp_max, params, sizes, ratio, warmup,
                   trials, base):
    """Full NLL: integrated_pwa Gram norm over phsp + data NLL via the base."""
    spec = {"name": "integrated_pwa", "base": base}
    print("=" * 78)
    print(f"  full NLL   integrated_pwa(base={base})   "
          f"(data:phsp = 1:{ratio})   [ms/call]")
    print("=" * 78)
    header = (f"{'n_data':>9} | {'n_phsp':>9} | {'norm(phsp)':>11} | "
              f"{'nll(data)':>11} | {'full NLL':>11} | {'data thrpt':>12}")
    print(header)
    print("-" * len(header))
    for n in sizes:
        data = {k: v[:n] for k, v in data_max.items()}
        nph = ratio * n
        phsp = {k: v[:nph] for k, v in phsp_max.items()}
        try:
            be = create_backend(spec, kc, model="pwa")
            ph = be.load_data(phsp)
            dh = be.load_data(data)

            def step_norm():
                return be.compute(params, ph, norm=None, return_p=False)[0]

            nrm = step_norm()                     # build Gram (one-time)
            t_norm = _bench(step_norm, warmup, trials)
            t_data = _bench(lambda: be.compute(params, dh, norm=nrm,
                                               return_p=True), warmup, trials)

            def step_full():
                v = be.compute(params, ph, norm=None, return_p=False)[0]
                be.compute(params, dh, norm=v, return_p=True)

            t_full = _bench(step_full, warmup, trials)
            be.free()
            print(f"{n:>9} | {nph:>9} | {t_norm * 1e3:>8.3g} ms | "
                  f"{t_data * 1e3:>8.3g} ms | {t_full * 1e3:>8.3g} ms | "
                  f"{_rate(n, t_full):>12}")
        except Exception:
            print(f"{n:>9} | {nph:>9} | {'SKIP':>11} | "
                  f"{'SKIP':>11} | {'SKIP':>11} | {'SKIP':>12}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tests/config_pwa.yml",
                    help="shape template (needs n_blocks == 1)")
    ap.add_argument("--n-comp", type=int, default=96,
                    help="target N = shared-ck length / components per "
                         "projection (total n_wave = n_proj * N)")
    ap.add_argument("--n-proj", type=int, default=None,
                    help="number of projections (default: the template's)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="small")
    ap.add_argument("--compute-sizes", default=None,
                    help="comma list, overrides --preset")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--include-cpu", action="store_true",
                    help="also benchmark numpy_pwa (skipped by default; it is "
                         "slow at 10^5-10^6 and not the like-for-like CPU)")
    ap.add_argument("--ratio", type=int, default=1,
                    help="phsp/data ratio for the full-NLL section")
    ap.add_argument("--integrated-base", default="cuda_v4_pwa_cache",
                    help="data-NLL backend for integrated_pwa (default: the "
                         "amp-cached backend)")
    ap.add_argument("--skip-full-nll", action="store_true",
                    help="skip the integrated_pwa norm(phsp)+NLL(data) section")
    ap.add_argument("--skip-compute", action="store_true",
                    help="skip the two raw-compute tables (keep full NLL)")
    args = ap.parse_args()

    backends = [(label, spec) for label, spec in BACKENDS
                if args.include_cpu or label not in CPU_BACKENDS]

    sizes = ([int(x) for x in args.compute_sizes.split(",") if x]
             if args.compute_sizes else list(PRESETS[args.preset]))
    n_max = max(sizes)

    print(f"template = {args.config}   building synthetic case "
          f"(n_events {n_max}, n_comp {args.n_comp}, "
          f"n_proj {args.n_proj}) ...")
    n_phsp = 0 if args.skip_full_nll else args.ratio * n_max
    kc, data_max, phsp_max, params, norm, info = synthetic_case(
        args.config, args.n_comp, args.n_proj, n_max, n_phsp)
    print(f"template N0 = {info['N0']}, tiled x{info['k']} -> "
          f"n_comp N = {info['N']} (shared ck), n_proj = {info['n_proj']}, "
          f"n_wave (total) = {info['n_wave']}, "
          f"data cols = {data_max['mass'].shape[1]}/"
          f"{data_max['q'].shape[1]}/"
          f"{data_max['angle'].shape[1] * data_max['angle'].shape[2]}\n")

    if not args.skip_compute:
        bench(kc, data_max, params, norm, sizes, args.warmup, args.trials,
              backends)
    if phsp_max is not None:
        bench_full_nll(kc, data_max, phsp_max, params, sizes, args.ratio,
                       args.warmup, args.trials, args.integrated_base)


if __name__ == "__main__":
    main()
