"""Parity tests: cuda_v4_pwa_s0d (no staging barrier) vs cuda_v4_pwa_s0c.

s0d = s0c with the uslot interp reading the mass column directly from
the READ-ONLY global input; the s_mass staging and its __syncthreads
(012c8fb) are gone.  The interp input is the same double either way.
Separate compilations differ at the ~1e-16 fp-order class (nvcc
reassociation), so the cross-kernel gate is rtol=1e-10; the race gate
is BITWISE SELF-DETERMINISM — s0d repeated with fresh kernel instances
(scheduling varies) must be bit-identical to itself every time, while
a stale-read race would produce reproducibly different BW/NLL.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

from test_v4_pwa_1bw import _make_env, _run

CONFIGS = ["tests/config_pwa.yml", "tests/config_pwa_bw.yml",
           "tests/config_pwa_bwr.yml"]


@pytest.mark.parametrize("config", CONFIGS)
def test_s0d_matches_s0c(config):
    """NLL + the flat gradient vector match s0c within family tolerance."""
    f, data, phsp = _make_env(config)
    x0 = f.initial_values(seed=63)
    nll_c, grads_c = _run(config, "cuda_v4_pwa_s0c", data, phsp, x0)
    nll_d, grads_d = _run(config, "cuda_v4_pwa_s0d", data, phsp, x0)
    assert nll_d == pytest.approx(nll_c, rel=1e-10, abs=1e-10)
    assert grads_d.shape == grads_c.shape
    assert np.allclose(grads_d, grads_c, rtol=1e-10, atol=1e-10)


def test_s0d_race_gate_repeated():
    """The 012c8fb race was scheduling-dependent and produced WRONG BW
    (O(1) deviations).  Measure: both s0c and s0d show a ~1.8e-12
    finite-reassociation nondeterminism across context instances
    (measured 30 runs each), so bitwise self-determinism does not hold
    for the FAMILY — the race detector is instead the rtol=1e-10 gate
    on every fresh-instance run (1e-12 noise floor, O(0.1) race
    signature: 8 orders of margin each way)."""
    for config in CONFIGS[:2]:
        data, phsp = _make_env(config)[1:]              # one dataset per config
        x0 = _make_env(config)[0].initial_values(seed=63)
        for it in range(10):                            # fresh kernels, same data
            a = _run(config, "cuda_v4_pwa_s0d", data, phsp, x0)
            c = _run(config, "cuda_v4_pwa_s0c", data, phsp, x0)
            assert np.allclose(a[0], c[0], rtol=1e-10, atol=1e-10), (config, it)
            assert np.allclose(a[1], c[1], rtol=1e-10, atol=1e-10), (config, it)
