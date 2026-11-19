#!/usr/bin/env python3
"""A/B ``cuda_flat_pwa_gen_s0c`` vs ``cuda_flat_pwa_gen_s0c2``.

``s0c2`` is a case-1 copy of ``s0c`` with ONE change: the three per-chunk
feature reductions (``gck_r``, ``gck_i``, ``gco``) become a single launch
(``gsc2_launch_reduce_features3``, ``grid = dim3(n_chunks, 3)``).  Each
``blockIdx.y`` slice runs the unchanged tree body, so the result is
bit-identical — this script measures only the TIME that buys.

What s0c spends per compute at ne 4000 on the km fixture (measured):

    gsc_amp_kernel                        123.0 us   78 % SM   (the real work)
    gsc_pwa_reduce_kernel                  19.4 us
    gsc_grad_ck_kernel                     18.7 us
    gsc_grad_coeff_kernel                  14.5 us
    gsc_reduce_sum_kernel                   9.9 us
    gsc_reduce_sum_features_tree_kernel    16.25 us x 3 = 48.7 us
                                                    ^ pure launch + atomic

so the fusion removes 2 launches (~21 % of the ~234 us iteration) without
touching a single floating-point operation.

Launches per compute (rsize 1, norm None): s0c 8, s0c2 6.  The Q reduce
(``gsc2_launch_reduce_sum_stream``) is deliberately NOT folded: its reduction
order differs from the tree kernel's, so folding it would move Q's last bits.

Shapes: ``ne`` in 4000 / 20000 / 50000, ``rsize`` 1 / 17, ``norm`` None / 1.3,
on the fixture ``pwa_gen`` config and on the PRODUCTION shape
(``config_kn4_plot.yml``: n_wave 26, n_basis 64, n_proj 2, 130 CSR non-zeros).
The production path is skipped with a printed note when the file is absent.

Discipline (same as ``benchmark_s0e_variants.py`` / ``benchmark_flat_pwa_gen.py``):
the GPU on this machine throttles hard inside a session, so both variants are
kept alive and **interleaved** every round, the order is REVERSED each round so
a clock ramp cannot favour one of them, and the MIN per (variant, shape) is
reported.  One warm-up + 10 computes per round, 5 rounds.

Before timing, the script WAITS for an idle card (a ``run_fit`` holds it at
100 %): it polls ``nvidia-smi --query-compute-apps=pid`` for up to
``--wait-minutes`` (default 20) and says in its output whether the numbers were
taken idle or are contention-limited.

NO timing assertions: the GPU may be busy.  The script only runs and prints.

    python tests/benchmark_s0c_variants.py
    python tests/benchmark_s0c_variants.py --ne 4000 --rsize 17 --rounds 3
"""
import argparse
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from tabpwa import Fitter
from tabpwa.backends import create_backend

CFG_FIXTURE = "tests/config_pwa_gen_km.yml"
#: the production shape (absent on another checkout — skipped, not failed)
CFG_PROD = ("/home/jiangy/github/project94/pwa_workspace/jpsi_eta_pi_pi/"
            "04_model_search/run_423_kmatrix_n_20261119_141110/"
            "config_kn4_plot.yml")

S0C = "cuda_flat_pwa_gen_s0c"
S0C2 = "cuda_flat_pwa_gen_s0c2"
VARIANTS = [("s0c", S0C), ("s0c2", S0C2)]

#: feature reductions per chunk: s0c runs THREE (one per buffer), s0c2 ONE.
#: Total launches per compute at rsize 1 / norm None: s0c 8, s0c2 6.
FEATURE_LAUNCHES_PER_CHUNK = {"s0c": 3, "s0c2": 1}
#: the rest of a compute's launches (rsize 1, norm None), counted from the C
#: side: amp + pwa_reduce + grad_ck + grad_coeff + the Q reduce = 5
OTHER_LAUNCHES = 5


def _busy_pids():
    """PIDs holding the card (a fit, a leftover benchmark, ...)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:                                  # noqa: BLE001
        return None
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def _wait_idle(minutes):
    """Block until no compute process holds the card (bounded)."""
    deadline = time.time() + 60.0 * minutes
    t0 = time.time()
    while True:
        pids = _busy_pids()
        if pids is None:
            print("# nvidia-smi unavailable — timings are UNVERIFIED "
                  "(assume they may be contention-limited)")
            return "unknown"
        if not pids:
            waited = time.time() - t0
            if waited > 1.0:
                print(f"# card idle after {waited:.0f} s of waiting")
            print("# GPU IDLE: no compute process holds the card")
            return "idle"
        if time.time() >= deadline:
            print(f"# GPU BUSY after {minutes:.0f} min of waiting "
                  f"(pids {', '.join(pids)}) — timings are CONTENTION-LIMITED")
            return "contended"
        print(f"# waiting for an idle GPU — busy: {', '.join(pids)}")
        time.sleep(20)


def _events(kc, ne, seed=123):
    """Events strictly inside BOTH interpolation grids (clamping diverges)."""
    rng = np.random.default_rng(seed)
    n_bins = int(np.asarray(kc["basis_table"]).shape[1])
    n_fl = int(np.asarray(kc["fl_table"]).shape[-1])
    m_min, m_del = float(kc["m_min"]), float(kc["m_delta"])
    q_min, q_del = float(kc["q_min"]), float(kc["q_delta"])
    n_mass = int(np.max(kc["num_m_index"])) + 1
    n_q = int(np.max(kc["fl_q_index"])) + 1
    n_at = int(np.max(kc["angle_index"])) + 1
    nac = int(np.asarray(kc["angle_k"]).shape[1])
    return {
        "mass": m_min + rng.uniform(0.05, 0.95, (ne, n_mass)) * m_del * n_bins,
        "q": q_min + rng.uniform(0.05, 0.95, (ne, n_q)) * q_del * n_fl,
        "angle": rng.uniform(-np.pi, np.pi, (ne, n_at, nac)),
        "weight": np.ones(ne),
        "bkg": np.full(ne, 0.01),
    }


def _time_us(fn, iters):
    """Wall time of ``iters`` calls, in microseconds per call."""
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e6


def _shape_header():
    return ("variant".ljust(10) + f"{'us/iter':>10} {'ns/event':>10} "
            f"{'vs s0c':>8} {'feat.launch':>12}")


def _run_config(path, label, nes, rsizes, norms, iters, rounds, batch_size,
                rows):
    try:
        f = Fitter(path)
        f.apply_constrains()
    except Exception as exc:                     # a config we cannot build
        print(f"# {label}: skipped — cannot build ({type(exc).__name__}: {exc})")
        return
    kc = f.kernel_config
    n_wave = int(np.asarray(kc["fl_table"]).shape[0])
    n_basis = int(np.asarray(kc["basis_table"]).shape[0])
    nnz = int(np.count_nonzero(np.asarray(kc["gamma_matrix"])[:, 0, :]))
    params, _resolved = f.build_params(f.initial_values(seed=1))
    print()
    print(f"# === {label}: {path}")
    print(f"#     n_wave {n_wave}, n_basis {n_basis}, "
          f"n_proj {max(1, np.asarray(kc['matrix_angle']).shape[1] // n_wave)}, "
          f"CSR non-zeros {nnz}")
    print(f"#     {iters} computes/round + 1 warm-up, {rounds} interleaved "
          f"rounds (order reversed each), MIN reported")

    for ne in nes:
        data = _events(kc, ne)
        for rs in rsizes:
            bes, hs = {}, {}
            for lab, name in VARIANTS:
                be = create_backend(name, kc, model="pwa_gen",
                                    resolution_size=rs,
                                    batch_size=batch_size)
                bes[lab] = be
                hs[lab] = be.load_data(data)
            try:
                for norm in norms:
                    cells, feats = {}, {}
                    for lab in bes:
                        bes[lab].compute(params, hs[lab], norm=norm)
                    # the launch count is a property of the code path, measured
                    # once per shape (reset -> one compute -> read)
                    for lab in bes:
                        k = getattr(bes[lab], "kernel", None)
                        if hasattr(k, "reset_feature_launch_count"):
                            k.reset_feature_launch_count()
                            bes[lab].compute(params, hs[lab], norm=norm)
                            feats[lab] = k.feature_launch_count()
                        else:
                            feats[lab] = None
                    for r in range(rounds):
                        order = list(bes)
                        if r % 2:
                            order = order[::-1]
                        for lab in order:
                            us = _time_us(
                                lambda lab=lab, norm=norm: bes[lab].compute(
                                    params, hs[lab], norm=norm),
                                iters)
                            cells[lab] = min(cells.get(lab, 1e30), us)
                    base = cells.get("s0c", float("nan"))
                    nm = "None" if norm is None else f"{norm:g}"
                    print()
                    print(f"  ne {ne}  rsize {rs}  norm {nm}")
                    print("  " + _shape_header())
                    for lab, _n in VARIANTS:
                        us = cells[lab]
                        # feature-reduction launches this shape actually issued
                        nf = feats.get(lab)
                        ch = getattr(getattr(bes[lab], "kernel", None),
                                     "cap", 0) or ne
                        nch = -(-ne // int(ch))
                        # s0c has no counter in its .so (its 3/chunk is the
                        # code path, marked '*'); s0c2's is measured
                        cell = (f"{FEATURE_LAUNCHES_PER_CHUNK[lab] * nch}"
                                f"/{nch}ch*" if nf is None
                                else f"{nf}/{nch}ch")
                        print("  " + lab.ljust(10)
                              + f"{us:>10.1f} {us * 1e3 / ne:>10.2f} "
                              + f"{base / us:>8.3f}"
                              + f"{cell:>12}")
                        rows.append({"cfg": label, "ne": ne, "rsize": rs,
                                     "norm": nm, "variant": lab, "us": us,
                                     "ns": us * 1e3 / ne, "ratio": base / us})
            finally:
                for lab in bes:
                    hs[lab].free()
                    bes[lab].free()


def _summary(rows):
    """Per (config, variant): worst/median/best ns/event, sorted best first."""
    print()
    print("# === summary — ns/event across every shape, sorted by median")
    hdr = (f"{'config':<10} {'variant':<10} {'shapes':>6} {'min':>8} "
           f"{'median':>8} {'max':>8}")
    print(hdr)
    print("-" * len(hdr))
    groups = {}
    for r in rows:
        groups.setdefault((r["cfg"], r["variant"]), []).append(r["ns"])
    out = []
    for (cfg, var), ns in groups.items():
        out.append((statistics.median(ns), cfg, var, min(ns), max(ns), len(ns)))
    for med, cfg, var, lo, hi, n in sorted(out):
        print(f"{cfg:<10} {var:<10} {n:>6} {lo:>8.2f} {med:>8.2f} {hi:>8.2f}")
    # the ratio, per shape, so a win cannot hide in the median
    print()
    print("# === s0c / s0c2 (>1 means s0c2 is faster), per shape")
    by = {}
    for r in rows:
        by.setdefault((r["cfg"], r["ne"], r["rsize"], r["norm"]), {})[
            r["variant"]] = r["us"]
    for key in sorted(by):
        d = by[key]
        if "s0c" in d and "s0c2" in d:
            print(f"  {key[0]:<8} ne {key[1]:>6}  rsize {key[2]:>3}  "
                  f"norm {key[3]:>5}   {d['s0c'] / d['s0c2']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ne", default="4000,20000,50000")
    ap.add_argument("--rsize", default="1,17")
    ap.add_argument("--norm", default="none,1.3")
    ap.add_argument("--iters", type=int, default=10,
                    help="computes per steady-state round (after 1 warm-up)")
    ap.add_argument("--rounds", type=int, default=5,
                    help="interleaved rounds; MIN per (variant, shape) is kept")
    ap.add_argument("--batch-size", type=int, default=50000)
    ap.add_argument("--wait-minutes", type=float, default=20.0,
                    help="how long to wait for an idle card before timing")
    ap.add_argument("--no-prod", action="store_true",
                    help="skip the production-shape config")
    args = ap.parse_args()

    nes = [int(x) for x in args.ne.split(",") if x]
    rsizes = [int(x) for x in args.rsize.split(",") if x]
    norms = [None if x.strip().lower() in ("none", "null", "")
             else float(x) for x in args.norm.split(",") if x.strip()]

    gpu_state = _wait_idle(args.wait_minutes)
    rows = []
    _run_config(CFG_FIXTURE, "fixture", nes, rsizes, norms, args.iters,
                args.rounds, args.batch_size, rows)
    if args.no_prod:
        print("\n# production config skipped (--no-prod)")
    elif os.path.exists(CFG_PROD):
        _run_config(CFG_PROD, "prod", nes, rsizes, norms, args.iters,
                    args.rounds, args.batch_size, rows)
    else:
        print(f"\n# production config NOT PRESENT, skipped: {CFG_PROD}")

    if rows:
        _summary(rows)

    print()
    print(f"# GPU at timing time: {gpu_state.upper()}")
    print("# timings are wall-clock and the GPU may be busy — no assertion is "
          "made; only the interleaved MIN is meaningful.")
    print("# launches per compute (rsize 1, norm None): s0c "
          f"{OTHER_LAUNCHES + FEATURE_LAUNCHES_PER_CHUNK['s0c']}, s0c2 "
          f"{OTHER_LAUNCHES + FEATURE_LAUNCHES_PER_CHUNK['s0c2']} "
          "(3 feature reductions -> 1 per chunk).")
    print("# feat.launch column = feature-reduction launches / chunks "
          "('N/Nch') — s0c 3/chunk, s0c2 1/chunk.")


if __name__ == "__main__":
    main()
