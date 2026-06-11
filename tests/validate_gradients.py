#!/usr/bin/env python3
"""Validate ALL backend gradients via 3-point numerical differentiation.

Tests NumPy, CUDA f64, CUDA f32, and ONNX (with dual model) gradients
against a 3-point central-difference numerical reference.
"""

import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_cuda_lib_path = "/usr/local/lib/ollama/mlx_cuda_v13"
if _cuda_lib_path not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = f"{_cuda_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

from ampfit.config_loader import Config
from ampfit.backends import NumpyBackend, ONNXBackend, CUDABackend

np.random.seed(42)
CONFIG_FILE = "config_angle.yml"
N_EVENTS = 64  # small for speed


def make_params():
    return {
        "ck": np.random.randn(448).astype(np.complex128)
              + 1j * np.random.randn(448).astype(np.complex128),
        "m0": np.random.rand(20).astype(np.float64) + 2.0,
        "g0": np.random.rand(23).astype(np.float64) + 0.1,
        "scalar": np.array([0.6, 0.01, 0.506, 0.01, 0.9, 0.2], dtype=np.float64),
    }


def make_data(n):
    return {
        "mass": np.random.rand(n, 48).astype(np.float64),
        "q": np.random.rand(n, 72).astype(np.float64),
        "angle": np.random.rand(n, 24, 3).astype(np.float64),
        "frac": np.random.rand(n).astype(np.float64),
        "time": np.random.rand(n).astype(np.float64),
        "weight": np.ones(n, dtype=np.float64),
        "bkg": np.random.rand(n).astype(np.float64) * 0.01,
    }


# ── Numerical gradients (3-point central difference) ───────────

def numerical_grad_Q(fn, params, eps=1e-6):
    """Compute Wirtinger gradient ∂Q/∂param via 3-point central difference.

    For complex ck: returns ∂Q/∂ck = 0.5·(∂Q/∂Re(ck) − j·∂Q/∂Im(ck)).
    For real params: returns ∂Q/∂param directly.
    """
    grads = {}

    ck = params["ck"].copy()
    grad_ck = np.zeros_like(ck, dtype=np.complex128)
    for i in range(len(ck)):
        ck[i] += eps
        Qp = fn({**params, "ck": ck})
        ck[i] -= 2 * eps
        Qm = fn({**params, "ck": ck})
        ck[i] += eps
        dQ_dRe = (Qp - Qm) / (2 * eps)

        ck[i] += 1j * eps
        Qp = fn({**params, "ck": ck})
        ck[i] -= 2j * eps
        Qm = fn({**params, "ck": ck})
        ck[i] += 1j * eps
        dQ_dIm = (Qp - Qm) / (2 * eps)

        grad_ck[i] = 0.5 * (dQ_dRe - 1j * dQ_dIm)
    grads["ck"] = grad_ck

    for key, arr in [("m0", params["m0"]), ("g0", params["g0"]),
                     ("scalar", params["scalar"])]:
        grad = np.zeros_like(arr)
        for i in range(len(arr)):
            arr[i] += eps
            Qp = fn({**params, key: arr})
            arr[i] -= 2 * eps
            Qm = fn({**params, key: arr})
            arr[i] += eps
            grad[i] = (Qp - Qm) / (2 * eps)
        grads[key] = grad

    return grads


def report(name, grads, ref, tol=1e-3):
    ok = True
    for key in ["ck", "m0", "g0", "scalar"]:
        g = np.asarray(grads[key])
        r = np.asarray(ref[key])
        denom = np.max(np.abs(r)) + 1e-30
        if key == "ck":
            abs_diff = np.abs(g.astype(np.complex128) - r.astype(np.complex128))
        else:
            abs_diff = np.abs(g.astype(np.float64) - r.astype(np.float64))
        max_rel = float(np.max(abs_diff) / denom)
        max_abs = float(np.max(abs_diff))
        passed = max_rel < tol or max_abs < 1e-8
        if not passed:
            ok = False
        print(f"  {name:<10} {key:<10} rel={max_rel:.2e}  abs={max_abs:.2e}  {'✓' if passed else '✗ FAIL'}")
    return ok


def main():
    print("=" * 70)
    print("  Gradient Validation — 3-Point Numerical Reference")
    print("=" * 70)

    config = Config(CONFIG_FILE)
    kernel_config = config.build_all_index()
    params = make_params()
    data = make_data(N_EVENTS)

    # ── Build backends ──
    backends = [
        ("NumPy",   NumpyBackend(kernel_config)),
        ("CUDAb64", CUDABackend(kernel_config, dtype="float64")),
        ("CUDAb32", CUDABackend(kernel_config, dtype="float32")),
        ("ONNX",    ONNXBackend("pwa_forward.onnx",
                                norm_model_path="pwa_forward_norm.onnx",
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])),
    ]

    # ════════════════════════════════════════════════════════════
    # norm=None (Q = sum(P*weight))
    # ════════════════════════════════════════════════════════════
    print("\n── norm=None (Q = sum(P·weight)) ──")

    def Q_norm_fn(p):
        nk = NumpyBackend(kernel_config)
        Q, _, _ = nk.compute(p, nk.load_data(data), norm=None)
        nk.free()
        return float(Q)

    print("  Computing numerical reference (3-point)...")
    num_grads = numerical_grad_Q(Q_norm_fn, params, eps=1e-6)
    print("  Done.")

    results_norm = {}
    for label, be in backends:
        Q, grads, _ = be.compute(params, be.load_data(data), norm=None)
        results_norm[label] = grads
        be.free()

    print(f"\n{'Gradients vs Numerical Reference (norm=None)':^60}")
    print("-" * 65)
    for label, _ in backends:
        report(label, results_norm[label], num_grads)

    # ════════════════════════════════════════════════════════════
    # norm=1000 (NLL)
    # ════════════════════════════════════════════════════════════
    NORM_VAL = 1000.0
    print(f"\n── norm={NORM_VAL} (NLL) ──")

    def Q_nll_fn(p):
        nk = NumpyBackend(kernel_config)
        Q, _, _ = nk.compute(p, nk.load_data(data), norm=NORM_VAL)
        nk.free()
        return float(Q)

    print("  Computing numerical reference (3-point)...")
    num_grads_nll = numerical_grad_Q(Q_nll_fn, params, eps=1e-6)
    print("  Done.")

    results_nll = {}
    for label, be in backends:
        be2 = NumpyBackend(kernel_config) if label == "NumPy" else \
              CUDABackend(kernel_config, dtype="float64") if label == "CUDAb64" else \
              CUDABackend(kernel_config, dtype="float32") if label == "CUDAb32" else \
              ONNXBackend("pwa_forward.onnx",
                          norm_model_path="pwa_forward_norm.onnx",
                          providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        Q, grads, _ = be2.compute(params, be2.load_data(data), norm=NORM_VAL)
        results_nll[label] = grads
        be2.free()

    print(f"\n{'Gradients vs Numerical Reference (NLL)':^60}")
    print("-" * 65)
    for label, _ in backends:
        report(label, results_nll[label], num_grads_nll)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    all_ok = True
    for mode, results in [("norm=None", results_norm), ("norm=NLL ", results_nll)]:
        refs = num_grads if "norm=None" in mode else num_grads_nll
        for label, _ in backends:
            ok = report(label, results[label], refs)
            print(f"    {mode} {label:<10}: {'✓ ALL PASS' if ok else '✗ FAIL'}")
            if not ok:
                all_ok = False
    print("=" * 70)
    print(f"  Overall: {'✓ ALL GRADIENTS CORRECT' if all_ok else '✗ SOME FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
