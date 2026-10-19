import json
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest


def _generate_lineshape_tables():
    """Generate the synthetic lineshape tables referenced by the
    repo-root test configs config_angle.yml / config_amp.yml
    (tests/lineshapes/) locally — the configs used to point at external
    fit tables under /home/jiangy/ana that no longer exist.  Tables are
    deterministic; physics content is irrelevant for the engine tests.

    width_linear_npy ``file:``   → (N, 3) [m, Im Pi, Re Pi]
    ck_matrix_v2 ``gamma_file``  → same file (only column 0 is read)
    ck_matrix_v2 ``partial_file``→ (N, n_ck, n_ck) complex
    ck_matrix_v2 ``order_file``  → JSON list of n_ck channel names
    """
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lineshapes")
    if os.path.exists(os.path.join(d, "pi2(1670)p_partial_order.json")):
        return
    os.makedirs(d, exist_ok=True)

    m = np.linspace(0.25, 2.7, 2000)

    specs = {  # basename: (bump center, bump width, scale)
        "Gamma_symm_disp.npy":  (0.775, 0.45, 1.0),
        "a2(1320)p_gamma.npy":  (1.318, 0.30, 0.9),
        "pi2(1670)p_gamma.npy": (1.716, 0.35, 0.8),
        "pi1(1600)p_gamma.npy": (1.600, 0.35, 0.7),
        "a1(1640)p_gamma.npy":  (1.640, 0.30, 0.6),
        "pi1300p_gamma.npy":    (1.300, 0.40, 0.5),
        "pi1600p_gamma.npy":    (1.600, 0.45, 0.4),
    }
    for name, (m0, w, s) in specs.items():
        im = s * np.exp(-((m - m0) / w) ** 2) + 0.05   # Im Pi: bump + floor
        re = 0.2 * np.cos(2.5 * (m - m0)) * np.exp(-((m - m0) / 1.2) ** 2)
        np.save(os.path.join(d, name), np.column_stack([m, im, re]))

    # ck_matrix_v2 triplets for pi2(1670)p
    n_ck = 2
    bump = np.exp(-((m - 1.716) / 0.35) ** 2)
    coef = np.array([[0.8 + 0.1j, 0.3 - 0.05j],
                     [0.3 + 0.05j, 0.6 - 0.08j]])
    M = np.zeros((len(m), n_ck, n_ck), dtype=np.complex128)
    for i in range(n_ck):
        for j in range(n_ck):
            M[:, i, j] = coef[i, j] * (0.4 + bump) * np.exp(1j * 0.3 * (m - 1.716))
    np.save(os.path.join(d, "pi2(1670)p_partial.npy"), M)
    with open(os.path.join(d, "pi2(1670)p_partial_order.json"), "w") as f:
        json.dump(["ch_1", "ch_2"], f)


def pytest_configure(config):
    # Runs before test collection, so module-level Config(...) calls in
    # test files see the generated files.
    _generate_lineshape_tables()
