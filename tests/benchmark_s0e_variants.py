#!/usr/bin/env python3
"""A/B/C the s0e tile family — base, packed-CSR (s0e2), Newton reciprocal (s0e3).

`s0e3` is a case-1 copy of `s0e` with **exactly one numerical change**: the
REAL reciprocal ``1/|D|²`` of the tile kernel's phase A is an fp32 seed plus
two fp64 Newton refinements instead of the IEEE fp64 divide, with an
exact-divide fallback outside the fp32-seed range.  The divide is ~25-30 % of
the fp64 work on the production shape (26 fp64 divides, 130 dot FMAs and 104
common values per event, and fp64 div is unpipelined on consumer GPUs), so
this script is the measurement that says whether the swap pays.

Variants measured (all on the SAME config, sample and parameters):

    s0e                 the base fused tile
    s0e2                the packed-CSR + 2-chain-dot copy
    s0e3                the Newton reciprocal, ``te = 0`` (kernel's choice)
    s0e3 te=rsize       the same, tile pinned to one log-sum group
    s0e3 te=2*rsize     the same, two groups per tile

Shapes: ``ne`` in 4000 / 20000 / 50000, ``rsize`` in 1 / 17 (per-event and
grouped objective), ``norm`` None / 1.3 — on the fixture ``pwa_gen`` config
and on the PRODUCTION shape (``config_kn4_plot.yml``: n_wave 26, n_basis 64,
n_proj 2, 130 CSR non-zeros); the production path is skipped with a printed
note when the file is absent.

Discipline (same as ``benchmark_s0e_ilp.py`` / ``benchmark_flat_pwa_gen.py``):
the GPU on this machine throttles hard inside a session, so every variant is
kept alive and **interleaved** every round, the order is REVERSED each round
so a clock ramp cannot favour one variant, and the MIN per (variant, shape)
is reported.  One warm-up + 10 computes per round, 5 rounds.

NO timing assertions: the GPU may be busy.  The script only runs and prints.

    python tests/benchmark_s0e_variants.py
    python tests/benchmark_s0e_variants.py --ne 4000 --rsize 17 --rounds 3

Profiling the tile kernels (each lives in its own ``.so`` with its own symbol,
so a ``regex:`` filter picks the right one):

    ncu --target-processes all --kernel-name regex:gps0_tile_kernel \\
        --set full -o /tmp/opencode/s0e_tile python tests/benchmark_s0e_variants.py
    ncu --target-processes all --kernel-name regex:gps02_tile_kernel \\
        --set full -o /tmp/opencode/s0e2_tile python tests/benchmark_s0e_variants.py
    ncu --target-processes all --kernel-name regex:gps03_tile_kernel \\
        --set full -o /tmp/opencode/s0e3_tile python tests/benchmark_s0e_variants.py
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from tabpwa import Fitter
from tabpwa.backends import create_backend

CFG_FIXTURE = "tests/config_pwa_gen_km.yml"
#: the shape s0e3 was designed against — absent on another checkout/machine
CFG_PROD = ("/home/jiangy/github/project94/pwa_workspace/jpsi_eta_pi_pi/"
            "04_model_search/run_423_kmatrix_n_20261119_141110/"
            "config_kn4_plot.yml")

#: (label, backend name) — the ``te`` variants are expanded per ``rsize``
VARIANTS = (
    ("s0e", "cuda_flat_pwa_gen_s0e"),
    ("s0e2", "cuda_flat_pwa_gen_s0e2"),
    ("s0e3", "cuda_flat_pwa_gen_s0e3"),
)
S0E3 = "cuda_flat_pwa_gen_s0e3"
#: the tile kernel symbol of each read-only variant, for the ``ncu`` commands
KERNELS = {"cuda_flat_pwa_gen_s0e": "gps0_tile_kernel",
           "cuda_flat_pwa_gen_s0e2": "gps02_tile_kernel",
           "cuda_flat_pwa_gen_s0e3": "gps03_tile_kernel"}


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


def _label(name):
    return name.replace("cuda_flat_pwa_gen_", "")


def _variants_for(rs):
    """``[(label, backend name, kwargs)]`` — the s0e3 ``te`` sweep included."""
    out = [(lab, name, {}) for lab, name in VARIANTS]
    out.append((f"s0e3 te={rs}", S0E3, {"te": rs}))
    out.append((f"s0e3 te={2 * rs}", S0E3, {"te": 2 * rs}))
    return out


def _shape_header(variants):
    return ("variant".ljust(14) + f"{'us/iter':>10} {'ns/event':>10} "
            f"{'vs s0e':>8}")


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
            variants = _variants_for(rs)
            bes, hs = {}, {}
            for lab, name, kw in variants:
                kw = dict(kw)
                if kw.get("te") is not None and rs > 1 and kw["te"] >= rs \
                        and kw["te"] % rs:
                    continue
                be = create_backend(name, kc, model="pwa_gen",
                                    resolution_size=rs,
                                    batch_size=batch_size, **kw)
                bes[lab] = be
                hs[lab] = be.load_data(data)
                # compiled resource usage — the thing the launch-bounds / -D
                # knob trades against; `local > 0` means the kernel SPILLS.
                stats = getattr(getattr(be, "kernel", None), "kernel_stats",
                                None)
                if stats is not None:
                    try:
                        st = stats()
                        print(f"# {lab:14s} regs {st['regs']:3d}  "
                              f"local {st['local_bytes']:4d} B  "
                              f"blocks/SM {st['blocks_per_sm']}  "
                              f"warps/SM {st['warps_per_sm']}"
                              + ("   SPILLS" if st["local_bytes"] else ""))
                    except Exception as exc:            # noqa: BLE001
                        print(f"# {lab:14s} kernel_stats failed: {exc}")
            try:
                for norm in norms:
                    cells = {}
                    for lab, _n, _kw in variants:
                        if lab not in bes:
                            continue
                        bes[lab].compute(params, hs[lab], norm=norm)
                    for r in range(rounds):
                        order = [lab for lab, _n, _kw in variants
                                 if lab in bes]
                        if r % 2:
                            order = order[::-1]
                        for lab in order:
                            us = _time_us(
                                lambda lab=lab, norm=norm: bes[lab].compute(
                                    params, hs[lab], norm=norm),
                                iters)
                            cells[lab] = min(cells.get(lab, 1e30), us)
                    base = cells.get("s0e", float("nan"))
                    nm = "None" if norm is None else f"{norm:g}"
                    print()
                    print(f"  ne {ne}  rsize {rs}  norm {nm}")
                    print("  " + _shape_header(variants))
                    for lab, _n, _kw in variants:
                        if lab not in cells:
                            continue
                        us = cells[lab]
                        print("  " + lab.ljust(14)
                              + f"{us:>10.1f} {us * 1e3 / ne:>10.2f} "
                              + f"{base / us:>8.3f}")
                        rows.append({"cfg": label, "ne": ne, "rsize": rs,
                                     "norm": nm, "variant": lab, "us": us,
                                     "ns": us * 1e3 / ne,
                                     "ratio": base / us})
            finally:
                for lab in bes:
                    hs[lab].free()
                    bes[lab].free()


def _summary(rows):
    """Per (config, variant): worst/median/best ns/event, sorted best first."""
    print()
    print("# === summary — ns/event across every shape, sorted by median")
    hdr = (f"{'config':<10} {'variant':<14} {'shapes':>6} {'min':>8} "
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
        print(f"{cfg:<10} {var:<14} {n:>6} {lo:>8.2f} {med:>8.2f} {hi:>8.2f}")


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
    ap.add_argument("--no-prod", action="store_true",
                    help="skip the production-shape config")
    args = ap.parse_args()

    nes = [int(x) for x in args.ne.split(",") if x]
    rsizes = [int(x) for x in args.rsize.split(",") if x]
    norms = [None if x.strip().lower() in ("none", "null", "")
             else float(x) for x in args.norm.split(",") if x.strip()]

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
    print("# timings are wall-clock and the GPU may be busy — no assertion is "
          "made; only the interleaved MIN is meaningful.")
    print()
    print("# profile the tile kernels (one .so each, distinct gps0* symbols):")
    for name, kern in KERNELS.items():
        tag = _label(name)
        print(f"  ncu --target-processes all --kernel-name regex:{kern} "
              f"--set full -o /tmp/opencode/{tag}_tile "
              f"python tests/benchmark_s0e_variants.py")
    print("#   (gps03_tile_kernel serves both s0e3 rows — te only changes the "
          "grid, not the kernel)")


if __name__ == "__main__":
    main()
