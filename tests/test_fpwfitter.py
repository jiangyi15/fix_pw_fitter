"""
Comprehensive tests for fpwfitter — CUDA (FpwFitter) and NumPy (NumpyFitter).

Usage:
    python tests/test_fpwfitter.py          # run all tests
    python tests/test_fpwfitter.py test_nproj  # run specific test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import time
import tempfile
import os

from fpwfitter import FpwFitter, NumpyFitter, compute_M


# ===========================================================================
# Helpers
# ===========================================================================

def make_data(n_data, n_mc, n_proj, n_comp, seed=42):
    """Generate deterministic random test data."""
    rng = np.random.RandomState(seed)
    F_data = rng.randn(n_data, n_proj, n_comp) + 1j * rng.randn(n_data, n_proj, n_comp)
    F_mc   = rng.randn(n_mc,   n_proj, n_comp) + 1j * rng.randn(n_mc,   n_proj, n_comp)
    w_data = np.abs(rng.randn(n_data))
    w_mc   = np.abs(rng.randn(n_mc))
    B_data = np.abs(rng.randn(n_data))
    B_mc   = np.abs(rng.randn(n_mc))
    c      = rng.randn(n_comp) + 1j * rng.randn(n_comp)
    return F_data, F_mc, w_data, w_mc, B_data, B_mc, c


def numpy_ref(F_data, w_data, B_data, M, N_b, c, purity):
    """Pure NumPy reference NLL + gradient (efficient einsum)."""
    A  = np.einsum('ijk,k->ij', F_data, c)
    S  = np.sum(np.abs(A) ** 2, axis=1)
    Ns = np.real(np.vdot(c, M @ c))
    if Ns < 1e-300: Ns = 1e-300
    if N_b < 1e-300: N_b = 1e-300
    P  = S / Ns * purity + B_data / N_b * (1 - purity)
    P  = np.maximum(P, 1e-300)
    nll = -np.dot(w_data, np.log(P))

    ratio = w_data / P
    G     = A * ratio[:, None]
    g_data = np.einsum('ijk,ij->k', np.conj(F_data), G)
    dNs_dc = M @ c
    S_corr = np.sum(w_data * S / P)
    grad = -purity / Ns * g_data + purity / Ns**2 * dNs_dc * S_corr
    return nll, grad, Ns, P


def check(fitter, c, purity, nll_ref, grad_ref, tol_nll=1e-8, tol_grad=1e-6,
          label=""):
    """Validate fitter output against reference."""
    nll, grad = fitter.evaluate(c)
    nll_err = abs(nll - nll_ref) / max(abs(nll_ref), 1e-300)
    # Use absolute error when gradient is small
    grad_norm_ref = np.linalg.norm(grad_ref)
    if grad_norm_ref < 1e-8:
        grad_err = np.linalg.norm(grad - grad_ref)
    else:
        grad_err = np.linalg.norm(grad - grad_ref) / grad_norm_ref

    ok_nll   = nll_err < tol_nll
    ok_grad  = grad_err < tol_grad
    ok_nan   = not np.isnan(nll) and not np.any(np.isnan(grad))

    status = "✓" if (ok_nll and ok_grad and ok_nan) else "✗"
    name = type(fitter).__name__
    print(f"  {status} {name:14s}  NLL_err={nll_err:.2e}  "
          f"grad_err={grad_err:.2e}  NaN={'yes' if not ok_nan else 'no'}"
          f"  {label}")
    return ok_nll and ok_grad and ok_nan


# ===========================================================================
# Tests
# ===========================================================================

def test_basic():
    """Small case: both backends match NumPy reference."""
    print("\n=== test_basic ===")
    F_data, F_mc, w_data, w_mc, B_data, B_mc, c = make_data(1000, 5000, 2, 10)
    purity = 0.8

    M, N_b = compute_M(F_mc, w_mc, B_mc)
    nll_ref, grad_ref, Ns_ref, _ = numpy_ref(F_data, w_data, B_data, M, N_b, c, purity)

    ok = True
    ok &= check(FpwFitter.from_M(F_data, w_data, B_data, M, N_b, purity),
                c, purity, nll_ref, grad_ref)
    ok &= check(NumpyFitter.from_M(F_data, w_data, B_data, M, N_b, purity),
                c, purity, nll_ref, grad_ref)
    print(f"  {'✓ ALL PASSED' if ok else '✗ FAILED'}")
    return ok


def test_nproj():
    """Test n_proj = 1, 2, 3, 4."""
    print("\n=== test_nproj ===")
    n_data, n_mc, n_comp = 10000, 50000, 20
    purity = 0.8
    ok = True

    for n_proj in [1, 2, 3, 4]:
        F_data, F_mc, w_data, w_mc, B_data, B_mc, c = make_data(
            n_data, n_mc, n_proj, n_comp, seed=n_proj)
        M, N_b = compute_M(F_mc, w_mc, B_mc)
        nll_ref, grad_ref, _, _ = numpy_ref(F_data, w_data, B_data, M, N_b, c, purity)

        ok &= check(FpwFitter.from_M(F_data, w_data, B_data, M, N_b, purity),
                    c, purity, nll_ref, grad_ref, label=f"j={n_proj}")
    print(f"  {'✓ ALL PASSED' if ok else '✗ FAILED'}")
    return ok


def test_ncomp():
    """Test n_comp = 1, 5, 50, 100."""
    print("\n=== test_ncomp ===")
    n_data, n_mc, n_proj = 5000, 25000, 2
    purity = 0.7
    ok = True

    for n_comp in [1, 5, 50, 100]:
        F_data, F_mc, w_data, w_mc, B_data, B_mc, c = make_data(
            n_data, n_mc, n_proj, n_comp, seed=n_comp)
        M, N_b = compute_M(F_mc, w_mc, B_mc)
        nll_ref, grad_ref, _, _ = numpy_ref(F_data, w_data, B_data, M, N_b, c, purity)

        ok &= check(FpwFitter.from_M(F_data, w_data, B_data, M, N_b, purity),
                    c, purity, nll_ref, grad_ref, label=f"k={n_comp}")
        ok &= check(NumpyFitter.from_M(F_data, w_data, B_data, M, N_b, purity),
                    c, purity, nll_ref, grad_ref, label=f"k={n_comp}")
    print(f"  {'✓ ALL PASSED' if ok else '✗ FAILED'}")
    return ok


def test_purity():
    """Test purity = 0.0, 0.5, 1.0 (edge cases)."""
    print("\n=== test_purity ===")
    F_data, F_mc, w_data, w_mc, B_data, B_mc, c = make_data(5000, 25000, 2, 20)
    ok = True

    for purity in [0.0, 0.5, 1.0]:
        M, N_b = compute_M(F_mc, w_mc, B_mc)
        nll_ref, grad_ref, _, _ = numpy_ref(F_data, w_data, B_data, M, N_b, c, purity)

        ok &= check(FpwFitter.from_M(F_data, w_data, B_data, M, N_b, purity),
                    c, purity, nll_ref, grad_ref, label=f"p={purity}")
    print(f"  {'✓ ALL PASSED' if ok else '✗ FAILED'}")
    return ok


def test_return_P():
    """Validate return_P=True matches reference."""
    print("\n=== test_return_P ===")
    F_data, F_mc, w_data, w_mc, B_data, B_mc, c = make_data(5000, 25000, 2, 20)
    M, N_b = compute_M(F_mc, w_mc, B_mc)
    _, _, Ns_ref, P_ref = numpy_ref(F_data, w_data, B_data, M, N_b, c, 0.8)

    ok = True
    for cls in [FpwFitter, NumpyFitter]:
        f = cls.from_M(F_data, w_data, B_data, M, N_b, 0.8)
        _, _, P = f.evaluate(c, return_P=True)
        match = np.allclose(P, P_ref, rtol=1e-10)
        print(f"  {'✓' if match else '✗'} {cls.__name__:14s}  P match: {match}")
        ok &= match
    print(f"  {'✓ ALL PASSED' if ok else '✗ FAILED'}")
    return ok


def test_save_load_M():
    """Save M from one fitter, load into another, verify results."""
    print("\n=== test_save_load_M ===")
    F_data, F_mc, w_data, w_mc, B_data, B_mc, c = make_data(5000, 25000, 2, 20)
    M, N_b = compute_M(F_mc, w_mc, B_mc)

    f1 = FpwFitter.from_M(F_data, w_data, B_data, M, N_b, 0.8)
    nll1, g1 = f1.evaluate(c)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "M.npz")
        f1.save_M(path)

        # Load into NumPy fitter (cross-backend)
        f2 = NumpyFitter.load_M(path, F_data, w_data, B_data, 0.8)
        nll2, g2 = f2.evaluate(c)

    ok_nll = abs(nll2 - nll1) < 1e-8
    ok_g   = np.allclose(g2, g1, rtol=1e-8)
    print(f"  {'✓' if ok_nll else '✗'} Cross-backend NLL match: {ok_nll}")
    print(f"  {'✓' if ok_g else '✗'} Cross-backend grad match: {ok_g}")
    print(f"  {'✓ ALL PASSED' if (ok_nll and ok_g) else '✗ FAILED'}")
    return ok_nll and ok_g


def test_get_M():
    """Verify get_M() returns the same M that was passed in."""
    print("\n=== test_get_M ===")
    F_data, F_mc, w_data, w_mc, B_data, B_mc, c = make_data(1000, 5000, 2, 10)
    M, N_b = compute_M(F_mc, w_mc, B_mc)
    f = FpwFitter.from_M(F_data, w_data, B_data, M, N_b, 0.8)
    M_out, Nb_out = f.get_M()
    ok = np.allclose(M_out, M) and abs(Nb_out - N_b) < 1e-12
    print(f"  {'✓' if ok else '✗'} M match: {np.allclose(M_out, M)}, "
          f"N_b match: {abs(Nb_out - N_b) < 1e-12}")
    print(f"  {'✓ ALL PASSED' if ok else '✗ FAILED'}")
    return ok


def test_properties():
    """Verify N_s, N_b, n_comp properties."""
    print("\n=== test_properties ===")
    F_data, F_mc, w_data, w_mc, B_data, B_mc, c = make_data(1000, 5000, 2, 15)
    M, N_b = compute_M(F_mc, w_mc, B_mc)

    f_gpu = FpwFitter.from_M(F_data, w_data, B_data, M, N_b, 0.8)
    f_cpu = NumpyFitter.from_M(F_data, w_data, B_data, M, N_b, 0.8)

    ok = True
    for f, name in [(f_gpu, "CUDA"), (f_cpu, "NumPy")]:
        ok_n = f.n_comp == 15
        ok_b = abs(f.N_b - N_b) < 1e-10
        print(f"  {'✓' if (ok_n and ok_b) else '✗'} {name:4s}  "
              f"n_comp={f.n_comp}, N_b={f.N_b:.4f}")
        ok &= ok_n and ok_b

    # N_s set after evaluate
    f_gpu.evaluate(c)
    _, _, Ns_ref, _ = numpy_ref(F_data, w_data, B_data, M, N_b, c, 0.8)
    ok_s = abs(f_gpu.N_s - Ns_ref) / Ns_ref < 1e-8
    print(f"  {'✓' if ok_s else '✗'} N_s after eval: "
          f"CUDA={f_gpu.N_s:.4f}  ref={Ns_ref:.4f}")
    ok &= ok_s
    print(f"  {'✓ ALL PASSED' if ok else '✗ FAILED'}")
    return ok


def test_reproducibility():
    """Multiple evaluates should give same result."""
    print("\n=== test_reproducibility ===")
    F_data, F_mc, w_data, w_mc, B_data, B_mc, c = make_data(5000, 25000, 2, 20)
    M, N_b = compute_M(F_mc, w_mc, B_mc)
    f = FpwFitter.from_M(F_data, w_data, B_data, M, N_b, 0.8)

    results = [f.evaluate(c) for _ in range(3)]
    nlls = [r[0] for r in results]
    grads = [r[1] for r in results]

    # NLL may vary at machine-epsilon due to atomicAdd ordering
    nll_spread = max(nlls) - min(nlls)
    grad_max   = max(np.linalg.norm(g - grads[0]) for g in grads)

    ok_nll = nll_spread / nlls[0] < 1e-12
    ok_g   = grad_max < 1e-10
    print(f"  {'✓' if ok_nll else '✗'} NLL spread: {nll_spread:.2e} "
          f"(rel: {nll_spread/nlls[0]:.2e})")
    print(f"  {'✓' if ok_g else '✗'} Grad max diff: {grad_max:.2e}")
    print(f"  {'✓ ALL PASSED' if (ok_nll and ok_g) else '✗ FAILED'}")
    return ok_nll and ok_g


def test_multiple_evals():
    """Evaluate with different c values."""
    print("\n=== test_multiple_evals ===")
    F_data, F_mc, w_data, w_mc, B_data, B_mc, _ = make_data(5000, 25000, 2, 20)
    M, N_b = compute_M(F_mc, w_mc, B_mc)
    f = FpwFitter.from_M(F_data, w_data, B_data, M, N_b, 0.8)
    rng = np.random.RandomState(99)
    ok = True
    for i in range(5):
        c = rng.randn(20) + 1j * rng.randn(20)
        nll_ref, g_ref, _, _ = numpy_ref(F_data, w_data, B_data, M, N_b, c, 0.8)
        ok &= check(f, c, 0.8, nll_ref, g_ref, label=f"eval#{i+1}")
    print(f"  {'✓ ALL PASSED' if ok else '✗ FAILED'}")
    return ok


def test_compute_M():
    """Verify compute_M against explicit NumPy loop."""
    print("\n=== test_compute_M ===")
    rng = np.random.RandomState(7)
    n_mc, n_proj, n_comp = 20000, 3, 15
    F_mc = rng.randn(n_mc, n_proj, n_comp) + 1j * rng.randn(n_mc, n_proj, n_comp)
    w_mc = np.abs(rng.randn(n_mc))
    B_mc = np.abs(rng.randn(n_mc))

    M, N_b = compute_M(F_mc, w_mc, B_mc)

    # Reference
    M_ref = np.zeros((n_comp, n_comp), dtype=np.complex128)
    for j in range(n_proj):
        F_j = F_mc[:, j, :]
        M_ref += np.dot(np.conj(F_j).T, w_mc[:, None] * F_j)
    N_b_ref = np.dot(w_mc, B_mc)

    ok_M = np.allclose(M, M_ref, rtol=1e-12)
    ok_b = abs(N_b - N_b_ref) < 1e-8
    print(f"  {'✓' if ok_M else '✗'} M match: {np.allclose(M, M_ref, rtol=1e-12)}")
    print(f"  {'✓' if ok_b else '✗'} N_b match: {abs(N_b - N_b_ref) < 1e-8}")
    print(f"  {'✓ ALL PASSED' if (ok_M and ok_b) else '✗ FAILED'}")
    return ok_M and ok_b


def test_compute_M_chunked():
    """Verify chunked compute_M gives same result as single chunk."""
    print("\n=== test_compute_M_chunked ===")
    rng = np.random.RandomState(11)
    n_mc, n_proj, n_comp = 100000, 2, 30
    F_mc = rng.randn(n_mc, n_proj, n_comp) + 1j * rng.randn(n_mc, n_proj, n_comp)
    w_mc = np.abs(rng.randn(n_mc))
    B_mc = np.abs(rng.randn(n_mc))

    M1, Nb1 = compute_M(F_mc, w_mc, B_mc, chunk_size=100000)
    M2, Nb2 = compute_M(F_mc, w_mc, B_mc, chunk_size=5000)
    M3, Nb3 = compute_M(F_mc, w_mc, B_mc, chunk_size=1000)

    ok = (np.allclose(M1, M2) and np.allclose(M2, M3)
          and abs(Nb1 - Nb2) < 1e-6 and abs(Nb2 - Nb3) < 1e-6)
    print(f"  {'✓' if ok else '✗'} chunk=100k vs 5k vs 1k all match")
    print(f"  {'✓ ALL PASSED' if ok else '✗ FAILED'}")
    return ok


def test_from_mc():
    """Test from_mc convenience constructor (end-to-end)."""
    print("\n=== test_from_mc ===")
    F_data, F_mc, w_data, w_mc, B_data, B_mc, c = make_data(5000, 25000, 2, 20)
    purity = 0.8

    f_gpu = FpwFitter.from_mc(F_data, F_mc, w_data, w_mc, B_data, B_mc, purity)
    f_cpu = NumpyFitter.from_mc(F_data, F_mc, w_data, w_mc, B_data, B_mc, purity)

    M, N_b = compute_M(F_mc, w_mc, B_mc)
    nll_ref, g_ref, _, _ = numpy_ref(F_data, w_data, B_data, M, N_b, c, purity)

    ok = True
    ok &= check(f_gpu, c, purity, nll_ref, g_ref, label="GPU from_mc")
    ok &= check(f_cpu, c, purity, nll_ref, g_ref, label="CPU from_mc")
    print(f"  {'✓ ALL PASSED' if ok else '✗ FAILED'}")
    return ok


def test_speedup():
    """Measure GPU vs CPU speedup."""
    print("\n=== test_speedup ===")
    F_data, F_mc, w_data, w_mc, B_data, B_mc, c = make_data(
        100_000, 500_000, 2, 50)
    M, N_b = compute_M(F_mc, w_mc, B_mc)

    f_gpu = FpwFitter.from_M(F_data, w_data, B_data, M, N_b, 0.8)
    f_cpu = NumpyFitter.from_M(F_data, w_data, B_data, M, N_b, 0.8)

    # Warmup
    f_gpu.evaluate(c)
    f_cpu.evaluate(c)

    t0 = time.perf_counter()
    for _ in range(5):
        f_cpu.evaluate(c)
    t_cpu = (time.perf_counter() - t0) / 5

    t0 = time.perf_counter()
    for _ in range(5):
        f_gpu.evaluate(c)
    t_gpu = (time.perf_counter() - t0) / 5

    speedup = t_cpu / t_gpu
    print(f"  CPU: {t_cpu*1000:.1f} ms  GPU: {t_gpu*1000:.1f} ms  "
          f"speedup: {speedup:.1f}×")
    ok = speedup > 5.0  # should be at least 5×
    print(f"  {'✓' if ok else '✗'} speedup > 5×")
    print(f"  {'✓ ALL PASSED' if ok else '✗ FAILED'}")
    return ok


# ===========================================================================
# Runner
# ===========================================================================

_ALL_TESTS = [v for k, v in sorted(globals().items())
              if k.startswith('test_') and callable(v)]


def run_all():
    results = {}
    for fn in _ALL_TESTS:
        try:
            results[fn.__name__] = fn()
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            results[fn.__name__] = False

    print("\n" + "=" * 60)
    passed = sum(results.values())
    total  = len(results)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n{passed}/{total} tests passed")
    return all(results.values())


if __name__ == "__main__":
    # If a specific test name is given as argv, run only that
    if len(sys.argv) > 1:
        name = sys.argv[1]
        fn = globals().get(name)
        if fn and callable(fn):
            fn()
        else:
            print(f"Test '{name}' not found. Available:")
            for t in _ALL_TESTS:
                print(f"  {t.__name__}")
    else:
        success = run_all()
        sys.exit(0 if success else 1)
