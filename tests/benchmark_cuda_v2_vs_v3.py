#!/usr/bin/env python
"""
Speed comparison: CUDA v2 (linear interpolation) vs v3 (Catmull-Rom).

Builds both kernels, runs them on the same data with increasing batch sizes,
and prints timing + throughput.

Usage:
    cd /media/jiangy/JZAO/github/qwen_code/project71
    python tests/benchmark_cuda_v2_vs_v3.py
"""

import sys, os, time, json, numpy as np

# Ensure ampfit is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ampfit import Fitter
from ampfit.backends import CUDABackendV2, CUDABackendV3

# ── Config ──────────────────────────────────────────────────────────
CONFIG = "config_test.yml"
PARAMS_FILE = "tfpwa_actual_params.json"
N_WARMUP = 5
N_TRIALS = 20
BATCH_SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# ── Load fitter & kernel config ─────────────────────────────────────
print("=" * 65)
print("  CUDA v2 (linear) vs v3 (Catmull-Rom) — speed comparison")
print("=" * 65)

fitter_ref = Fitter(CONFIG, backend="numpy")
kc = fitter_ref.kernel_config

with open(PARAMS_FILE) as f:
    tfpwa_params = json.load(f)

slot_dict = {}
for key, val in tfpwa_params.items():
    if "_g_ls" in key or "_g_lsbar" in key or "_total_" in key:
        slot_dict[key] = float(val)

ck = fitter_ref.pc.build_ck(slot_dict)
m0 = fitter_ref.default_m0
g0 = fitter_ref.default_g0
scalar = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

# ── Load data ───────────────────────────────────────────────────────
data = np.load("data/data_arrays.npz")
N_total = data["mass"].shape[0]
print(f"  Total events available: {N_total}")

params_dict = {"ck": ck, "m0": m0, "g0": g0, "scalar": scalar}


def make_data_dict(n):
    return {
        "mass": data["mass"][:n].reshape(n, -1),
        "q": data["q"][:n].reshape(n, -1),
        "angle": data["angles"][:n],
        "frac": data["frac"][:n],
        "time": data["time"][:n],
        "weight": data["weight"][:n],
        "bkg": data["bkg_raw"][:n],
    }


# ── Benchmark ───────────────────────────────────────────────────────
def time_backend(BackendClass, name):
    print(f"\n  --- {name} ---")
    backend = BackendClass(kc, batch_size=0)
    results = []
    for bs in BATCH_SIZES:
        if bs > N_total:
            continue
        d = make_data_dict(bs)
        dh = backend.load_data(d)

        # Warmup
        for _ in range(N_WARMUP):
            backend.compute(params_dict, dh)

        # Timed runs
        times = []
        for _ in range(N_TRIALS):
            t0 = time.perf_counter()
            backend.compute(params_dict, dh)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

        times.sort()
        median_ms = times[len(times) // 2]
        throughput = bs / (median_ms / 1000)
        results.append((bs, median_ms, throughput))
        print(f"    {bs:>6d} events: {median_ms:>8.2f} ms  {throughput:>10.0f} ev/s")
        dh.free()

    backend.free()
    return results


res_v2 = time_backend(CUDABackendV2, "v2 linear")
res_v3 = time_backend(CUDABackendV3, "v3 Catmull-Rom")

# ── Summary table ───────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SUMMARY")
print("=" * 65)
print(f"  {'Batch':>6} | {'v2 (ms)':>10} | {'v3 (ms)':>10} | {'v2 ev/s':>10} | {'v3 ev/s':>10} | {'ratio':>7}")
print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}")
for r2, r3 in zip(res_v2, res_v3):
    assert r2[0] == r3[0]
    bs = r2[0]
    m2, t2 = r2[1], r2[2]
    m3, t3 = r3[1], r3[2]
    ratio = m2 / m3 if m3 > 0 else 0
    tag = "  ✅ v3 faster" if m3 < m2 else "  ⚠️ v2 faster"
    print(f"  {bs:>6d} | {m2:>10.2f} | {m3:>10.2f} | {t2:>10.0f} | {t3:>10.0f} | {ratio:>6.3f}x{tag}")
