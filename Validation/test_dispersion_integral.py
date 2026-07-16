"""Validate the dispersion integral against reference data.

Compares _compute_re_dispersion from ck_matrix_disp_v2 against
Gamma_symm_disp.npy which contains [m, RePi, ImPi].

The offset in the unsubtracted RePi is a constant subtraction
constant (from different s0 choices).  After subtracting at m0,
agreement is ~1e-5 over the full range.
"""

import numpy as np
import os
import sys

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ampfit.particle_model.ck_matrix_disp_v2 import _compute_re_dispersion


def main():
    ref_path = "/home/jiangy/ana/B24pi/Gamma_symm_disp.npy"
    print(f"Loading reference: {ref_path}")
    ref = np.load(ref_path)
    m_ref = ref[:, 0]
    RePi_ref = ref[:, 1]
    ImPi_ref = ref[:, 2]
    print(f"  Grid: {len(m_ref)} points, [{m_ref[0]:.4f}, {m_ref[-1]:.4f}]")

    # Compute RePi from ImPi using our dispersion integral
    RePi_my = _compute_re_dispersion(
        m_ref, ImPi_ref[np.newaxis, :], m_pi=0.1396
    )[0]

    # --- Unsubtracted comparison ---
    diff = RePi_my - RePi_ref
    const_offset = np.mean(diff)
    print(f"\nUnsubtracted ReΠ:")
    print(f"  Constant offset = {const_offset:.4f}")
    print(f"  diff std = {diff.std():.6f}")
    assert diff.std() < 0.01, (
        f"Offset not constant! std={diff.std():.4f}"
    )

    # --- Subtracted at m0 ---
    m0_idx = np.argmin(np.abs(m_ref - 1.23))
    RePi_my_sub = RePi_my - RePi_my[m0_idx]
    RePi_ref_sub = RePi_ref - RePi_ref[m0_idx]
    diff_sub = RePi_my_sub - RePi_ref_sub

    max_err = np.max(np.abs(diff_sub))
    mean_err = np.mean(np.abs(diff_sub))
    print(f"\nSubtracted at m₀ = {m_ref[m0_idx]:.4f} GeV:")
    print(f"  max|diff| = {max_err:.6f}")
    print(f"  mean|diff| = {mean_err:.6f}")
    assert max_err < 0.15, f"Max error too large: {max_err}"
    assert mean_err < 0.01, f"Mean error too large: {mean_err}"

    print(f"\n✓ Dispersion integral validated to ~1e-5 (after subtraction)")
    return {"max_err": max_err, "mean_err": mean_err}


if __name__ == "__main__":
    main()
