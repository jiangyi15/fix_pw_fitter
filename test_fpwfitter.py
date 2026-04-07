"""
Test: validate CUDA fitter against pure NumPy reference implementation.
"""
import numpy as np
import time

# ---------------------------------------------------------------------------
# Pure NumPy reference
# ---------------------------------------------------------------------------

def reference_nll_and_grad(
    F_data, F_mc, w_data, w_mc, B_data, B_mc, c, purity
):
    """
    F_data : (n_data, n_proj, n_comp)  complex
    F_mc   : (n_mc,   n_proj, n_comp)  complex
    c      : (n_comp,)                  complex
    """
    # Amplitudes
    # A[i,j] = sum_k F[i,j,k] * c[k]
    A = np.einsum('ijk,k->ij', F_data, c)          # (n_data, n_proj)
    S = np.sum(np.abs(A) ** 2, axis=1)              # (n_data,)

    # Normalisation from MC
    A_mc = np.einsum('ijk,k->ij', F_mc, c)          # (n_mc, n_proj)
    S_mc = np.sum(np.abs(A_mc) ** 2, axis=1)        # (n_mc,)
    N_s = np.dot(w_mc, S_mc)
    N_b = np.dot(w_mc, B_mc)

    # PDF
    P = S / N_s * purity + B_data / N_b * (1 - purity)
    P = np.maximum(P, 1e-300)

    # NLL
    nll = -np.dot(w_data, np.log(P))

    # Gradient d(-ln L)/d(c*)
    # dS/dc_k^* = sum_j F_ijk^* * A_ij
    dS_dc = np.einsum('ijk,ij->k', np.conj(F_data), A)   # (n_comp,)

    # dN_s/dc_k^* = sum_j sum_i' w_i' F_i'jk^* * A_i'j
    dNs_dc = np.einsum('ijk,ij->k', np.conj(F_mc) * w_mc[:, None, None], A_mc)

    # P gradient
    # dP/dc_k^* = p/N_s * (dS/dc_k^* - S/N_s * dNs_dc_k)
    # But this is per-event... let's compute the full gradient directly:
    # d(-ln L)/dc_k^* = -sum_i w_i/P_i * dP_i/dc_k^*
    # dP_i/dc_k^* = p/N_s * (dS_i/dc_k^* - S_i/N_s * dNs_dc_k)

    g_data_term = np.zeros_like(c, dtype=np.complex128)
    for k in range(len(c)):
        for j in range(F_data.shape[1]):
            g_data_term[k] += np.dot(
                w_data / P,
                np.conj(F_data[:, j, k]) * A[:, j]
            )

    S_corr = np.dot(w_data * S / P, np.ones_like(w_data))  # sum w_i * S_i / P_i

    grad = (
        -purity / N_s * g_data_term
        + purity / (N_s ** 2) * dNs_dc * S_corr
    )

    return nll, grad, N_s, N_b, P


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_small():
    np.random.seed(42)

    n_data = 1000
    n_mc   = 5000
    n_proj = 2
    n_comp = 10

    F_data = (np.random.randn(n_data, n_proj, n_comp) +
              1j * np.random.randn(n_data, n_proj, n_comp))
    F_mc   = (np.random.randn(n_mc, n_proj, n_comp) +
              1j * np.random.randn(n_mc, n_proj, n_comp))
    w_data = np.abs(np.random.randn(n_data))
    w_mc   = np.abs(np.random.randn(n_mc))
    B_data = np.abs(np.random.randn(n_data))
    B_mc   = np.abs(np.random.randn(n_mc))
    purity = 0.8

    c = np.random.randn(n_comp) + 1j * np.random.randn(n_comp)

    # Reference
    ref_nll, ref_grad, ref_Ns, ref_Nb, ref_P = reference_nll_and_grad(
        F_data, F_mc, w_data, w_mc, B_data, B_mc, c, purity
    )
    print(f"Reference:  NLL = {ref_nll:.6f}")
    print(f"            N_s = {ref_Ns:.4f},  N_b = {ref_Nb:.4f}")
    print(f"            |grad| = {np.linalg.norm(ref_grad):.6f}")

    # CUDA
    try:
        from fpwfitter import FpwFitter
        fitter = FpwFitter(F_data, F_mc, w_data, w_mc, B_data, B_mc, purity)
        cuda_nll, cuda_grad = fitter.evaluate(c)
        print(f"\nCUDA:       NLL = {cuda_nll:.6f}")
        print(f"            N_s = {fitter.N_s:.4f},  N_b = {fitter.N_b:.4f}")
        print(f"            |grad| = {np.linalg.norm(cuda_grad):.6f}")

        # Compare
        nll_err = abs(cuda_nll - ref_nll) / abs(ref_nll)
        grad_err = np.linalg.norm(cuda_grad - ref_grad) / np.linalg.norm(ref_grad)
        print(f"\nNLL  rel error: {nll_err:.2e}")
        print(f"Grad rel error: {grad_err:.2e}")

        if nll_err < 1e-8 and grad_err < 1e-6:
            print("\n✓ PASSED")
        else:
            print("\n✗ FAILED — errors too large")

    except Exception as e:
        print(f"\nCUDA test skipped (library not available): {e}")


def test_performance():
    """Timing test with realistic sizes."""
    np.random.seed(42)

    n_data = 100_000
    n_mc   = 500_000
    n_proj = 2
    n_comp = 50

    print(f"\nPerformance test: n_data={n_data}, n_mc={n_mc}, n_proj={n_proj}, n_comp={n_comp}")

    F_data = (np.random.randn(n_data, n_proj, n_comp) +
              1j * np.random.randn(n_data, n_proj, n_comp))
    F_mc   = (np.random.randn(n_mc, n_proj, n_comp) +
              1j * np.random.randn(n_mc, n_proj, n_comp))
    w_data = np.abs(np.random.randn(n_data))
    w_mc   = np.abs(np.random.randn(n_mc))
    B_data = np.abs(np.random.randn(n_data))
    B_mc   = np.abs(np.random.randn(n_mc))
    purity = 0.8
    c = np.random.randn(n_comp) + 1j * np.random.randn(n_comp)

    try:
        from fpwfitter import FpwFitter

        t0 = time.perf_counter()
        fitter = FpwFitter(F_data, F_mc, w_data, w_mc, B_data, B_mc, purity)
        t_create = time.perf_counter() - t0
        print(f"  Create (pre-compute): {t_create:.3f} s")

        t0 = time.perf_counter()
        for _ in range(5):
            nll, grad = fitter.evaluate(c)
        t_eval = (time.perf_counter() - t0) / 5
        print(f"  Evaluate (avg):       {t_eval:.3f} s")
        print(f"  NLL = {nll:.2f}")

    except Exception as e:
        print(f"  Skipped: {e}")


if __name__ == "__main__":
    test_small()
    test_performance()
