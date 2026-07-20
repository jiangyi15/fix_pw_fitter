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
from ampfit.backends import create_backend
from ampfit import Fitter

CONFIG_FILE = "config_angle.yml"
N_EVENTS = 16  # small enough for f32 precision, large enough for stable gradients


def make_physical_params():
    """Build physically meaningful kernel params via the fitter pipeline.

    Random params cause NaN in v3 Catmull-Rom interpolation (extrapolation
    outside the physical interpolation table), so we use Fitter.initial_values
    and _build_params to get physically consistent values.
    """
    f = Fitter(CONFIG_FILE, backend="numpy")
    x = f.initial_values(seed=42)
    params, _ = f._build_params(x)
    return params


def make_data(n, seed=12345):
    rng = np.random.default_rng(seed)
    return {
        "mass": (rng.random((n, 48)) * 4.8 + 0.3).astype(np.float64),  # [0.3, 5.1] GeV — physical range
        "q": rng.random((n, 72)).astype(np.float64),
        "angle": rng.random((n, 24, 3)).astype(np.float64),
        "frac": rng.random(n).astype(np.float64),
        "time": rng.random(n).astype(np.float64),
        "weight": np.ones(n, dtype=np.float64),
        "bkg": rng.random(n).astype(np.float64) * 0.01,
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
    # ONNX uses linear interpolation (vs Catmull-Rom for others) → larger tolerance
    if "ONNX" in name:
        tol = 5e-3
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
    params = make_physical_params()
    data = make_data(N_EVENTS)

    # ════════════════════════════════════════════════════════════
    # norm=None (Q = sum(P*weight))
    # ════════════════════════════════════════════════════════════
    print("\n── norm=None (Q = sum(P·weight)) ──")

    nk = create_backend("numpy", kernel_config)
    dh = nk.load_data(data)

    def Q_norm_fn(p):
        Q, _, _ = nk.compute(p, dh, norm=None)
        return float(Q)

    print("  Computing numerical reference (3-point)...")
    num_grads = numerical_grad_Q(Q_norm_fn, params, eps=1e-6)
    print("  Done.")
    nk.free()

    # Test each backend independently — GPU backends pre-allocate
    # scratch buffers (batch_size=50000, ≈1.7GB/context) so creating
    # multiple at once can exhaust GPU memory.
    backend_configs = [
        ("NumPy",    "numpy"),
        ("CUDAv2",   "cuda_v2"),
        ("CUDA32v2", "cuda32_v2"),
        ("CUDAv3",   "cuda_v3"),
        ("CUDA32v3", "cuda32_v3"),
        ("ONNXcpu",  "onnx_cpu"),
    ]

    results_norm = {}
    for label, be_name in backend_configs:
        be = create_backend(be_name, kernel_config)
        Q, grads, _ = be.compute(params, be.load_data(data), norm=None)
        results_norm[label] = grads
        be.free()

    print(f"\n{'Gradients vs Numerical Reference (norm=None)':^60}")
    print("-" * 65)
    for label, _ in backend_configs:
        report(label, results_norm[label], num_grads)
    if hasattr(dh, 'free'): dh.free()

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    all_ok = True
    for label, _ in backend_configs:
        ok = report(label, results_norm[label], num_grads)
        print(f"    norm=None {label:<10}: {'✓ ALL PASS' if ok else '✗ FAIL'}")
        if not ok:
            all_ok = False
    print("=" * 70)
    print(f"  Overall: {'✓ ALL GRADIENTS CORRECT' if all_ok else '✗ SOME FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
