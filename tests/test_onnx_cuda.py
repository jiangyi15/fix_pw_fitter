#!/usr/bin/env python3
"""Test ONNX Runtime inference with CUDA execution provider.

Validates:
  1. CUDAExecutionProvider is listed and selectable
  2. Model loads and runs inference on GPU
  3. Numeric agreement between CPU and CUDA execution providers
  4. Performance comparison (CPU vs CUDA)
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import onnxruntime as ort


# Build ONNX model from config (matches how onnx_backend.py does it).
CONFIG_FILE = "config_angle.yml"
BATCH_SIZE = 512

import warnings
from ampfit.config_loader import Config
from ampfit._onnx_builder import PWAONNXBuilder
config = Config(CONFIG_FILE)
kc = config.build_all_index()
builder = PWAONNXBuilder(kc)
_model = builder.build(batch_size=BATCH_SIZE, norm_model=False)
_model_norm = builder.build(batch_size=BATCH_SIZE, norm_model=True)
del config, kc, builder

WARMUP = 3
TRIALS = 10


def make_data(n_events, n_ck=448, n_m0=20, n_g0=23, n_scalar=6):
    """Generate synthetic data matching the ONNX model's expected shapes."""
    return {
        "ck_real": np.random.randn(n_ck).astype(np.float32),
        "ck_imag": np.random.randn(n_ck).astype(np.float32),
        "m0": np.random.rand(n_m0).astype(np.float32) + 2.0,
        "g0": np.random.rand(n_g0).astype(np.float32) + 0.1,
        "mass": np.random.rand(n_events, 48).astype(np.float32),
        "q": np.random.rand(n_events, 72).astype(np.float32),
        "angle": np.random.rand(n_events, 24, 3).astype(np.float32),
        "frac": np.random.rand(n_events).astype(np.float32),
        "time": np.random.rand(n_events).astype(np.float32),
        "weight": np.ones(n_events, dtype=np.float32),
        "bkg": np.random.rand(n_events).astype(np.float32) * 0.01,
        "norm": np.array([1.0], dtype=np.float32),
        "Gamma": np.array([0.5], dtype=np.float32),
        "Delta_Gamma": np.array([0.2], dtype=np.float32),
        "Delta_m": np.array([0.3], dtype=np.float32),
        "A_prod": np.array([0.4], dtype=np.float32),
        "poq_rho": np.array([0.6], dtype=np.float32),
        "pop_phi": np.array([0.7], dtype=np.float32),
    }


def test_providers_available():
    """CUDA and CPU providers must be available."""
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    assert "CUDAExecutionProvider" in providers, (
        f"CUDAExecutionProvider not found. Got: {providers}"
    )
    assert "CPUExecutionProvider" in providers
    print("✓ CUDA and CPU execution providers are available")


def _sess(providers):
    return ort.InferenceSession(_model.SerializeToString(), providers=providers)
def _sess_norm(providers):
    return ort.InferenceSession(_model_norm.SerializeToString(), providers=providers)

def test_model_loads():
    """Model loads successfully with CUDA provider."""
    sess = _sess(["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]
    print(f"Model inputs ({len(input_names)}): {input_names}")
    print(f"Model outputs ({len(output_names)}): {output_names}")
    assert "Q" in output_names
    assert "P" in output_names
    print("✓ Model loaded successfully with CUDA provider")


def test_cpu_cuda_numerical_agreement():
    """CPU and CUDA inference must produce identical results within tolerance."""
    cpu_sess = _sess(["CPUExecutionProvider"])
    cuda_sess = _sess(["CUDAExecutionProvider", "CPUExecutionProvider"])

    n = BATCH_SIZE
    data = make_data(n)
    feed = {k: v for k, v in data.items()}

    cpu_outs = cpu_sess.run(None, feed)
    cuda_outs = cuda_sess.run(None, feed)

    cpu_dict = dict(zip([o.name for o in cpu_sess.get_outputs()], cpu_outs))
    cuda_dict = dict(zip([o.name for o in cuda_sess.get_outputs()], cuda_outs))

    for name in cpu_dict:
        cpu_val = np.asarray(cpu_dict[name])
        cuda_val = np.asarray(cuda_dict[name])
        abs_diff = np.abs(cpu_val - cuda_val)
        max_abs = np.max(abs_diff)
        # Use relative tolerance for scalars (float32 reduction differences),
        # absolute tolerance for element-wise outputs
        if cpu_val.ndim == 0:
            rel_diff = max_abs / (np.abs(float(cpu_val)) + 1e-10)
            assert rel_diff < 1e-3, (
                f"Output '{name}' relative mismatch: {rel_diff:.2e} (abs: {max_abs:.2e})"
            )
        else:
            assert max_abs < 1e-1, (
                f"Output '{name}' mismatch: max diff = {max_abs:.2e}"
            )
    print(f"  n={n}: ✓ CPU ≃ CUDA (rel < 0.1%, abs < 0.1)")


def test_cuda_provider_actually_used():
    """Verify the CUDA execution provider is actually selected (not CPU fallback)."""
    sess = _sess(["CUDAExecutionProvider", "CPUExecutionProvider"])
    active_provider = sess.get_providers()[0]
    print(f"  Active provider: {active_provider}")
    assert "CUDA" in active_provider, (
        f"CUDA provider is not active. Got: {active_provider}"
    )
    print("✓ CUDA provider is active")

    # Run a simple inference to verify it works
    data = make_data(BATCH_SIZE)
    feed = {k: v for k, v in data.items()}
    outs = sess.run(None, feed)
    Q = float(outs[0])
    assert np.isfinite(Q), "Q is not finite"
    print(f"  Q = {Q:.6f} ✓")


def test_cpu_vs_cuda_performance():
    """Rough performance comparison between CPU and CUDA providers."""

    def benchmark(provider, n, warmup=WARMUP, trials=TRIALS):
        sess = _sess([provider])
        data = make_data(n)
        feed = {k: v for k, v in data.items()}
        # Warmup
        for _ in range(warmup):
            sess.run(None, feed)
        # Timed runs
        times = []
        for _ in range(trials):
            t0 = time.perf_counter()
            sess.run(None, feed)
            times.append(time.perf_counter() - t0)
        return np.median(times)

    # The model has a fixed batch size, so we can only benchmark at that size.
    # For multi-size benchmarking, rebuild with --batch-size N.
    print(f"\nBenchmarking at model's fixed batch size ({BATCH_SIZE}):")
    print(f"{'Provider':>10} | {'Time (ms)':>10}")
    print("-" * 25)
    times = {}
    for provider_name, provider_key in [("CPU", "CPUExecutionProvider"),
                                         ("CUDA", "CUDAExecutionProvider")]:
        t = benchmark(provider_key, BATCH_SIZE)
        times[provider_key] = t
        print(f"{provider_name:>10} | {t*1000:>10.3f}")
    speedup = times["CPUExecutionProvider"] / times["CUDAExecutionProvider"]
    print(f"\n  CPU / CUDA speedup: {speedup:.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("ONNX Runtime CUDA Test Suite")
    print(f"onnxruntime version: {ort.__version__}")
    print("=" * 60)

    print("\n1. Testing provider availability...")
    test_providers_available()

    print("\n2. Testing model load with CUDA...")
    test_model_loads()

    print("\n3. Testing CUDA provider is actually active...")
    test_cuda_provider_actually_used()

    print("\n4. Testing numerical agreement CPU vs CUDA...")
    test_cpu_cuda_numerical_agreement()

    print("\n5. Performance benchmark...")
    test_cpu_vs_cuda_performance()

    print("\n" + "=" * 60)
    print("✓ All ONNX Runtime CUDA tests passed!")
    print("=" * 60)
