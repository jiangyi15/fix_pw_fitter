#!/usr/bin/env python3
"""Comprehensive benchmark comparing all compute backends.

Backends tested:
  1. NumpyBackend (f64, CPU)     — reference
  2. CUDABackend  (f64, GPU)     — CUDA double precision
  3. CUDABackend  (f32, GPU)     — CUDA single precision ("cuda32")
  4. ONNXBackend  (f32, CPU)     — ONNX Runtime CPU, weight-masked for any batch size
  5. ONNXBackend  (f32, CUDA)    — ONNX Runtime GPU, weight-masked for any batch size

ONNX models are built in-memory from kernel_config at a moderate batch size.
Variable batch sizes are handled internally by ONNXBackend.compute() which
splits data into fixed-size chunks with weight=0 masking on partial batches.

Metrics:
  - Execution time (ms)
  - Throughput (events/s)
  - Numerical accuracy vs NumPy f64 reference
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

# Set LD_LIBRARY_PATH for CUDA runtime libs (cuDNN etc.)
_cuda_lib_path = "/usr/local/lib/ollama/mlx_cuda_v13"
if _cuda_lib_path not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = (
        f"{_cuda_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )

from ampfit.config_loader import Config
from ampfit.backends import CUDABackend, NumpyBackend, ONNXBackend

# ── Configuration ─────────────────────────────────────────────
CONFIG_FILE = "config_angle.yml"
ONNX_MODEL = "pwa_forward.onnx"

# Auto-detect ONNX batch size from the model file
import onnxruntime as _ort
_onnx_sess = _ort.InferenceSession(ONNX_MODEL, providers=["CPUExecutionProvider"])
ONNX_MAX_BATCH = [i.shape[0] for i in _onnx_sess.get_inputs() if i.name == "mass"][0]

BATCH_SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

WARMUP = 5
TRIALS = 20

np.random.seed(42)


def make_params(n_ck=448, n_m0=20, n_g0=23):
    return {
        "ck": np.random.randn(n_ck).astype(np.complex128)
              + 1j * np.random.randn(n_ck).astype(np.complex128),
        "m0": np.random.rand(n_m0).astype(np.float64) + 2.0,
        "g0": np.random.rand(n_g0).astype(np.float64) + 0.1,
        "scalar": np.array([0.6, 0.01, 0.506, 0.01, 0.9, 0.2], dtype=np.float64),
    }


def make_data(n_events):
    """Generate synthetic data for *exactly* n_events."""
    return {
        "mass": np.random.rand(n_events, 48).astype(np.float64),
        "q": np.random.rand(n_events, 72).astype(np.float64),
        "angle": np.random.rand(n_events, 24, 3).astype(np.float64),
        "frac": np.random.rand(n_events).astype(np.float64),
        "time": np.random.rand(n_events).astype(np.float64),
        "weight": np.ones(n_events, dtype=np.float64),
        "bkg": np.random.rand(n_events).astype(np.float64) * 0.01,
    }


def init_backend(name, kernel_config):
    if name == "numpy":
        return NumpyBackend(kernel_config)
    elif name == "cuda_f64":
        return CUDABackend(kernel_config, dtype="float64")
    elif name == "cuda_f32":
        return CUDABackend(kernel_config, dtype="float32")
    elif name == "onnx_cpu":
        return ONNXBackend(
            kernel_config=kernel_config,
            batch_size=ONNX_MAX_BATCH,
            providers=["CPUExecutionProvider"],
        )
    elif name == "onnx_cuda":
        return ONNXBackend(
            kernel_config=kernel_config,
            batch_size=ONNX_MAX_BATCH,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    else:
        raise ValueError(f"Unknown backend: {name}")


def convert_params(params, backend_name):
    p = dict(params)
    if backend_name.startswith("cuda_f32") or backend_name.startswith("onnx"):
        p["ck"] = p["ck"].astype(np.complex64)
        p["m0"] = p["m0"].astype(np.float32)
        p["g0"] = p["g0"].astype(np.float32)
        p["scalar"] = p["scalar"].astype(np.float32)
    return p


def convert_data(data, backend_name):
    d = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            if backend_name.startswith("cuda_f32") or backend_name.startswith("onnx"):
                d[k] = v.astype(np.float32)
            else:
                d[k] = v.copy()
        else:
            d[k] = v
    return d


def benchmark_backend(name, kernel_config, n_events, trials=TRIALS, warmup=WARMUP):
    backend = init_backend(name, kernel_config)

    # Generate base data (ONNXBackend's compute handles batching internally)
    raw_params = make_params()
    raw_data = make_data(n_events)
    data = convert_data(raw_data, name)
    params = convert_params(raw_params, name)

    dh = backend.load_data(data)

    # Warmup
    for _ in range(warmup):
        backend.compute(params, dh, norm=None)

    # Timed runs
    times = []
    q_vals = []
    for _ in range(trials):
        t0 = time.perf_counter()
        Q, grads, P = backend.compute(params, dh, norm=None)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        q_vals.append(float(Q))

    backend.free()

    median_t = np.median(times)
    throughput = n_events / median_t

    return {
        "backend": name,
        "n_events": n_events,
        "time_ms": median_t * 1000,
        "throughput": throughput,
        "Q_mean": np.mean(q_vals),
        "Q_std": np.std(q_vals),
        "grad_keys": list(grads.keys()),
    }


def accuracy_check(name, kernel_config, n_events, ref_backend="numpy"):
    """Compare backend output against NumPy f64 reference."""
    ref = init_backend(ref_backend, kernel_config)
    backend = init_backend(name, kernel_config)

    raw_params = make_params()
    raw_data = make_data(n_events)

    # Reference (numpy f64)
    ref_data = {k: v.copy() for k, v in raw_data.items()}
    ref_dh = ref.load_data(ref_data)
    Q_ref, grads_ref, P_ref = ref.compute(raw_params, ref_dh, norm=None)
    ref.free()

    # Test backend (ONNXBackend handles batching internally)
    data = convert_data(raw_data, name)
    params = convert_params(raw_params, name)
    dh = backend.load_data(data)
    Q, grads, P = backend.compute(params, dh, norm=None)
    backend.free()

    results = {}
    results["Q_rel_diff"] = abs(float(Q) - float(Q_ref)) / (abs(float(Q_ref)) + 1e-30)

    for key in ["ck", "m0", "g0", "scalar"]:
        if key in grads and key in grads_ref:
            g_test = np.asarray(grads[key])
            g_ref_ = np.asarray(grads_ref[key])
            if np.iscomplexobj(g_test) or np.iscomplexobj(g_ref_):
                denom = np.max(np.abs(g_ref_)) + 1e-30
                results[f"grad_{key}_max_rel"] = float(
                    np.max(np.abs(g_test.astype(np.complex128)
                                 - g_ref_.astype(np.complex128))) / denom
                )
            else:
                denom = np.max(np.abs(g_ref_)) + 1e-30
                results[f"grad_{key}_max_rel"] = float(
                    np.max(np.abs(g_test.astype(np.float64)
                                 - g_ref_.astype(np.float64))) / denom
                )

    if P is not None and P_ref is not None:
        p_test = np.asarray(P).astype(np.float64)
        p_ref_ = np.asarray(P_ref).astype(np.float64)
        results["P_max_abs_diff"] = float(np.max(np.abs(p_test - p_ref_)))

    return results


def main():
    print("=" * 75)
    print("  Backend Performance & Accuracy Benchmark")
    print(f"  ONNX model: fixed batch {ONNX_MAX_BATCH}, weight-masked for variable sizes")
    print("=" * 75)

    config = Config(CONFIG_FILE)
    kernel_config = config.build_all_index()

    BACKENDS = [
        ("numpy",     "NumPy f64 CPU"),
        ("cuda_f64",  "CUDA f64 GPU"),
        ("cuda_f32",  "CUDA f32 GPU"),
        ("onnx_cpu",  "ONNX f32 CPU"),
        ("onnx_cuda", "ONNX f32 CUDA"),
    ]

    # ─────────────────────────────────────────────
    # 1. Performance benchmark
    # ─────────────────────────────────────────────
    print("\n" + "-" * 75)
    print("  PERFORMANCE")
    print("-" * 75)

    header = f"{'Backend':<16} | {'Events':>7} | {'Time (ms)':>10} | {'Throughput':>12} | {'Q':>12}"
    print(header)
    print("-" * 75)

    all_results = []
    for be_name, be_label in BACKENDS:
        for n in BATCH_SIZES:
            r = benchmark_backend(be_name, kernel_config, n)
            all_results.append(r)
            tp_str = f"{r['throughput']:,.0f} ev/s"
            print(
                f"{be_label:<16} | {r['n_events']:>7} | {r['time_ms']:>10.3f} | {tp_str:>12} | {r['Q_mean']:>12.4f}"
            )

    # ─────────────────────────────────────────────
    # 2. Throughput scaling table
    # ─────────────────────────────────────────────
    print("\n" + "-" * 75)
    print("  THROUGHPUT SCALING (events/second)")
    print("-" * 75)

    from collections import defaultdict

    by_backend = defaultdict(dict)
    for r in all_results:
        by_backend[r["backend"]][r["n_events"]] = r["throughput"]

    all_sizes = sorted(set(r["n_events"] for r in all_results))
    header = f"{'Backend':<16}"
    for s in all_sizes:
        header += f" | {s:>7}"
    print(header)
    print("-" * (16 + 10 * len(all_sizes)))

    for be_name, be_label in BACKENDS:
        line = f"{be_label:<16}"
        for s in all_sizes:
            val = by_backend[be_name].get(s)
            line += f" | {val:>7,.0f}" if val is not None else f" | {'N/A':>7}"
        print(line)

    # ─────────────────────────────────────────────
    # 3. Accuracy vs NumPy f64 reference
    # ─────────────────────────────────────────────
    print("\n" + "-" * 75)
    print("  ACCURACY vs NumPy f64 (reference)")
    print("-" * 75)

    acc_header = (
        f"{'Backend':<16} | {'Q rel diff':>12} | {'grad_ck rel':>12} "
        f"| {'grad_m0 rel':>12} | {'grad_g0 rel':>12} "
        f"| {'grad_scalar rel':>12} | {'P abs diff':>12}"
    )
    print(acc_header)
    print("-" * 75)

    for be_name, be_label in BACKENDS:
        if be_name == "numpy":
            print(
                f"{be_label:<16} | {'0.0':>12} | {'0.0':>12} "
                f"| {'0.0':>12} | {'0.0':>12} | {'0.0':>12} | {'0.0':>12}"
            )
            continue

        n = 256  # compare at modest size
        acc = accuracy_check(be_name, kernel_config, n)
        print(
            f"{be_label:<16}"
            f" | {acc.get('Q_rel_diff', float('nan')):>12.2e}"
            f" | {acc.get('grad_ck_max_rel', float('nan')):>12.2e}"
            f" | {acc.get('grad_m0_max_rel', float('nan')):>12.2e}"
            f" | {acc.get('grad_g0_max_rel', float('nan')):>12.2e}"
            f" | {acc.get('grad_scalar_max_rel', float('nan')):>12.2e}"
            f" | {acc.get('P_max_abs_diff', float('nan')):>12.2e}"
        )

    # ─────────────────────────────────────────────
    # 4. Summary at n=256
    # ─────────────────────────────────────────────
    print("\n" + "-" * 75)
    print("  SUMMARY at n=256")
    print("-" * 75)
    for r in all_results:
        if r["n_events"] == 256:
            tp_str = f"{r['throughput']:,.0f}"
            print(f"  {r['backend']:<12} | {r['time_ms']:>8.3f} ms | {tp_str:>12} ev/s")

    # ─────────────────────────────────────────────
    # 5. Speedup vs NumPy (at each batch size)
    # ─────────────────────────────────────────────
    print("\n" + "-" * 75)
    print("  SPEEDUP vs NumPy f64 (×)")
    print("-" * 75)

    # Collect time_ms for each backend & batch size
    time_map = defaultdict(dict)
    for r in all_results:
        time_map[r["backend"]][r["n_events"]] = r["time_ms"]

    su_header = f"{'Backend':<16}"
    for s in all_sizes:
        su_header += f" | {s:>6}"
    print(su_header)
    print("-" * (16 + 8 * len(all_sizes)))

    for be_name, be_label in BACKENDS:
        line = f"{be_label:<16}"
        for s in all_sizes:
            if be_name == "numpy":
                line += f" | {'1.00×':>6}"
            else:
                np_t = time_map["numpy"].get(s)
                this_t = time_map[be_name].get(s)
                if np_t and this_t:
                    speedup = np_t / this_t
                    line += f" | {speedup:>5.1f}×"
                else:
                    line += f" | {'N/A':>6}"
        print(line)

    print("\n" + "=" * 75)
    print("  Benchmark complete!")
    print("=" * 75)


if __name__ == "__main__":
    main()
