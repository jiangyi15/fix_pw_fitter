"""Parity tests: cuda_v4_pwa_s0e (single fused tile kernel) vs the family.

s0e fuses s0d's whole per-iteration path into ONE kernel: a block owns
``S0E_TE`` events and keeps BW, 1/BW, common, common/BW, A_p and dQ/dA_p
in shared memory (no batching, one launch, grid = ceil(ne/S0E_TE)).  The
math and the basis are unchanged, so:

* Fitter-level NLL + flat gradient matches ``cuda_v4_pwa_s0c`` at the
  ~1e-13 class (separate compilation + a different summation order: the
  tile kernel reduces per block with atomics instead of per-event
  partials through global memory).
* Kernel-level Q / P / ck / m0 / g0 / dnorm match the fp64 CPU sibling
  ``cpu_pwa_s0c`` at the fp32-amplitude-cache noise floor (~1e-7) for
  event counts on both sides of the tile boundary (1, 7, 8, 9, 33, 400)
  — the last tile is partial, so this is the tiling gate.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

from test_v4_pwa_1bw import _make_env, _run
from test_cpu_pwa_s0c import _kernel_config, _synth, _params

from tabpwa.backends import create_backend, backends_for_model

CONFIGS = ["tests/config_pwa.yml", "tests/config_pwa_bw.yml",
           "tests/config_pwa_bwr.yml", "tests/config_pwa_flatte.yml"]


def _close(a, b, rtol=1e-4, atol_frac=1e-6):
    """allclose with a tolerance scaled to the vector magnitude.

    The CUDA path caches the angular amplitude as float2 (inherited from
    the ampcache family) while the CPU sibling is fp64, so cancellation-
    dominated components differ relatively (but not absolutely).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return np.allclose(a, b, rtol=rtol,
                       atol=atol_frac * max(np.max(np.abs(b)), 1e-30))


def test_registration():
    assert "cuda_v4_pwa_s0e" in backends_for_model("pwa")


@pytest.mark.parametrize("config", CONFIGS)
def test_s0e_matches_s0c(config):
    """NLL + the flat gradient vector match s0c within family tolerance."""
    f, data, phsp = _make_env(config)
    x0 = f.initial_values(seed=63)
    nll_c, grads_c = _run(config, "cuda_v4_pwa_s0c", data, phsp, x0)
    nll_e, grads_e = _run(config, "cuda_v4_pwa_s0e", data, phsp, x0)
    assert float(nll_e) == pytest.approx(float(nll_c), rel=1e-10, abs=1e-10)
    assert grads_e.shape == grads_c.shape
    assert np.allclose(grads_e, grads_c, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("ne", [1, 7, 8, 9, 33, 400])
@pytest.mark.parametrize("norm", [None, 1.7])
def test_tiling_matches_cpu_reference(ne, norm):
    """Q/P/gradients/dnorm vs the fp64 CPU sibling, incl. partial tiles."""
    kc = _kernel_config()
    cpu = create_backend("cpu_pwa_s0c", kc, model="pwa")
    gpu = create_backend("cuda_v4_pwa_s0e", kc, model="pwa")
    k = cpu.kernel
    data = _synth(kc, ne, seed=5)
    params = _params(k, seed=7, scale=0.5)

    Q1, g1, P1 = cpu.compute(params, cpu.load_data(data), norm=norm)
    Q2, g2, P2 = gpu.compute(params, gpu.load_data(data), norm=norm)

    assert _close(Q2, Q1)
    assert P2.shape == (ne,)
    assert _close(P2, P1)
    for key in ("ck", "m0", "g0"):
        assert _close(g2[key], g1[key]), key
    if norm is not None:
        assert _close(gpu.kernel._last_dnorm, cpu.kernel._last_dnorm,
                      rtol=1e-6, atol_frac=1e-8)
    else:
        assert gpu.kernel._last_dnorm is None

    cpu.free()
    gpu.free()


def test_return_p_false():
    """return_p=False skips only the P copy-out (Q/grads unchanged)."""
    kc = _kernel_config()
    gpu = create_backend("cuda_v4_pwa_s0e", kc, model="pwa")
    k = gpu.kernel
    data = _synth(kc, 64, seed=2)
    params = _params(k)
    h = gpu.load_data(data)

    Q_full, g_full, P = gpu.compute(params, h, norm=1.3, return_p=True)
    Q_fast, g_fast, P_none = gpu.compute(params, h, norm=1.3, return_p=False)
    assert P is not None and P.shape == (64,)
    assert P_none is None
    # the fused kernel accumulates Q/gradients with per-block atomics, so
    # two calls can differ in the last bits (same class as the existing
    # s0b/s0c/s0d accumulator family)
    assert Q_fast == pytest.approx(Q_full, rel=1e-12)
    for key in ("ck", "m0", "g0"):
        assert _close(g_fast[key], g_full[key], rtol=1e-12, atol_frac=1e-14), key
    gpu.free()
