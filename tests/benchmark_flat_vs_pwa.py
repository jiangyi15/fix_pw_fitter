#!/usr/bin/env python3
"""Compare the flat `pwa_gen` kernels with the plain-`pwa` CUDA family at the
shape where the two are the SAME model.

A `pwa_gen` config with ``n_index ≡ 0`` and the default denominator basis is
exactly the plain ``pwa`` config (same F, same D, same P) — that is gate A of
the pwa_gen fixture.  So at that setting the fair opponents are the pure-PWA
kernels, and ``cuda_v5_pwa_s0e`` is the closest: one BW per wave with the
s0 pole parameterisation, which is what the flat config's default basis
``[1, −m², −iγ]`` reduces to.

Shape convention matches ``tests/benchmark_pwa_kernels.py``: ``--n-comp`` is
N, the shared-ck length **per projection** (``n_wave = n_proj · N``).

Everything is measured interleaved in one process (this GPU throttles
180..2730 MHz within a session, so sequential numbers are meaningless).

    python tests/benchmark_flat_vs_pwa.py
    python tests/benchmark_flat_vs_pwa.py --n-comp 99 --n-proj 2 --sizes 65536,262144
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from tabpwa import Fitter
from tabpwa.backends import create_backend

FLAT_TEMPLATE = "tests/config_pwa_gen_km.yml"
PWA_TEMPLATE = "tests/config_pwa.yml"


# ── flat config tiling (same convention as benchmark_pwa_kernels) ──────
def _fit_proj(a, axis, n_proj):
    P0 = a.shape[axis]
    if n_proj == P0:
        return a
    reps = int(np.ceil(n_proj / P0))
    sl = tuple(slice(None) if i != axis else slice(0, n_proj)
               for i in range(a.ndim))
    return np.repeat(a, reps, axis=axis)[sl]


def tile_flat(kc, n_comp, n_proj):
    """Tile the flat config to ``n_wave = n_proj · N``, ``N = N0·k``."""
    out = dict(kc)
    N0 = int(np.asarray(kc["fl_table"]).shape[0])
    k = max(1, int(round(n_comp / N0)))
    am = np.asarray(kc["angle_matrix"])
    out["fl_table"] = np.tile(np.asarray(kc["fl_table"]), (k, 1, 1))
    out["num_table"] = np.tile(np.asarray(kc["num_table"]), (k, 1, 1))
    out["fl_q_index"] = np.tile(np.asarray(kc["fl_q_index"]), (k, 1))
    out["num_m_index"] = np.tile(np.asarray(kc["num_m_index"]), (k, 1))
    out["angle_matrix"] = np.tile(_fit_proj(am, 1, int(n_proj)), (k, 1, 1))
    out["gamma_matrix"] = np.tile(np.asarray(kc["gamma_matrix"]), (k, 1, 1))
    return out


def _events(kc, ne, seed=123):
    rng = np.random.default_rng(seed)
    n_basis = int(np.asarray(kc["basis_table"]).shape[1])
    n_fl = int(np.asarray(kc["fl_table"]).shape[-1])
    m_min, m_del = float(kc["m_min"]), float(kc["m_delta"])
    q_min, q_del = float(kc["q_min"]), float(kc["q_delta"])
    n_mass = int(np.max(kc["num_m_index"])) + 1
    n_q = int(np.max(kc["fl_q_index"])) + 1
    n_at = int(np.max(kc["angle_index"])) + 1
    nac = int(np.asarray(kc["angle_k"]).shape[1])
    return {
        "mass": m_min + rng.uniform(0.05, 0.95, (ne, n_mass)) * m_del * n_basis,
        "q": q_min + rng.uniform(0.05, 0.95, (ne, n_q)) * q_del * n_fl,
        "angle": rng.uniform(-np.pi, np.pi, (ne, n_at, nac)),
        "weight": np.ones(ne),
        "bkg": np.full(ne, 0.01),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-comp", type=int, default=99)
    ap.add_argument("--n-proj", type=int, default=2)
    ap.add_argument("--sizes", default="65536,262144")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    flat_backends = ("cuda_flat_pwa_gen", "cuda_flat_cache_pwa_gen")
    pwa_backends = ("cuda_v5_pwa_s0e", "cuda_v5_pwa_s0c", "cuda_v4_pwa_cache")

    # ── the flat case, forced to the plain-pwa model (n ≡ 1) ──────────
    ff = Fitter(FLAT_TEMPLATE)
    ff.apply_constrains()
    fkc = tile_flat(ff.kernel_config, args.n_comp, args.n_proj)
    fkc = dict(fkc)
    fkc["num_table"] = np.ones_like(np.asarray(fkc["num_table"]))
    fx = ff.initial_values(seed=1)
    fparams, _ = ff.build_params(fx)
    n_wave = int(np.asarray(fkc["fl_table"]).shape[0])
    n_basis = int(np.asarray(fkc["basis_table"]).shape[0])
    ck0 = np.asarray(fparams["ck"])
    fparams = {"ck": np.tile(ck0, (n_wave + ck0.size - 1) // ck0.size)[:n_wave],
               "coeff": np.random.default_rng(7).uniform(0.3, 1.2, n_basis)}

    # ── the plain-pwa case at the same shape ─────────────────────────
    from benchmark_pwa_kernels import synthetic_case
    pkc, pdata, _phsp, pparams, pnorm, info = synthetic_case(
        PWA_TEMPLATE, args.n_comp, args.n_proj, 256)

    print("=" * 78)
    print(f"flat pwa_gen (n ≡ 1): n_wave={n_wave} "
          f"n_proj={np.asarray(fkc['angle_matrix']).shape[1]} n_basis={n_basis}")
    print(f"plain pwa          : n_wave={info['n_wave']} "
          f"n_proj={info['n_proj']} N={info['N']}")
    print("=" * 78)

    header = (f"{'n_events':>10}" + "".join(f" | {n:>22}" for n in
                                            flat_backends + pwa_backends))
    print(header)
    print("-" * len(header))
    for ne in [int(s) for s in args.sizes.split(",") if s]:
        fdata = _events(fkc, ne)
        pd = {k: (v[:ne] if isinstance(v, np.ndarray) and v.shape[0] == 256
                  else v) for k, v in pdata.items()}
        if pd["mass"].shape[0] != ne:
            pd = synthetic_case(PWA_TEMPLATE, args.n_comp, args.n_proj, ne)[1]
        fnorm = float(ne) * 0.5
        pn = float(ne) * 0.5

        insts = {}
        for name in flat_backends:
            try:
                k = create_backend(name, fkc, model="pwa_gen")
                insts[name] = (k, k.load_data(fdata), fparams, fnorm)
            except Exception as e:                            # noqa: BLE001
                print(f"  {name} SKIP {type(e).__name__}: {str(e)[:50]}")
        for name in pwa_backends:
            try:
                k = create_backend(name, pkc, model="pwa")
                insts[name] = (k, k.load_data(pd), pparams, pn)
            except Exception as e:                            # noqa: BLE001
                print(f"  {name} SKIP {type(e).__name__}: {str(e)[:50]}")
        if not insts:
            continue
        times = {n: [] for n in insts}
        for _ in range(args.warmup):
            for k, h, pp, nn in insts.values():
                k.compute(pp, h, norm=nn)
        for _ in range(args.trials):
            for name, (k, h, pp, nn) in insts.items():
                t0 = time.perf_counter()
                k.compute(pp, h, norm=nn)
                times[name].append((time.perf_counter() - t0) * 1e3)
        cells = []
        for name in flat_backends + pwa_backends:
            if name not in times:
                cells.append(f"{'SKIP':>22}")
                continue
            t = min(times[name])
            cells.append(f"{t:9.2f} ms {ne / t / 1e3:6.1f}M")
        print(f"{ne:>10}" + "".join(f" | {c:>22}" for c in cells))
        for k, h, _pp, _nn in insts.values():
            k.free()


if __name__ == "__main__":
    main()
