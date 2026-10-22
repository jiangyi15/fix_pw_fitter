"""Parity tests: cuda_v5_pwa_s0d (s0c u-cache variant) vs cuda_v5_pwa_s0c.

s0d = s0c with the uslot gamma interp filled ONCE per data load
(iteration-invariant inputs); the per-iteration kernel is a pure matvec
(no interp, no shared staging, no barrier).  Same grouped log-sum NLL
and interface as s0c.  Gates: s0d == s0c at rsize=1 (per-event group —
compare against the v4 s0c pipeline) and at rsize>1 (group kernels),
with the grad path and _last_dnorm checked.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

from test_pwa_v5_s0c import _make, pwa_small


def test_v5_s0d_rsize1_equals_v5_s0c(pwa_small):
    from tabpwa.cuda._v4_pwa_s0c import CUDAKernelV4PWAS0C as KV4S0C
    from tabpwa.cuda._v5_pwa_s0c import CUDAKernelV5PWAS0C as KV5S0C
    from tabpwa.cuda._v5_pwa_s0d import CUDAKernelV5PWAS0D as KV5S0D
    _cfg, kc, data, phsp, params, _p = pwa_small

    k4, kc5 = _make(KV4S0C, kc), _make(KV5S0C, kc, resolution_size=1)
    kd = _make(KV5S0D, kc, resolution_size=1)
    hs = {n: m.load_data(data) for n, m in
          (("v4", k4), ("s0c", kc5), ("s0d", kd))}
    hp = k4.load_data(phsp)
    try:
        norm = k4.compute(params, hp, norm=None)[0]
        Q4, g4, P4 = k4.compute(params, hs["v4"], norm=norm)
        Qc, gc, Pc = kc5.compute(params, hs["s0c"], norm=norm)
        Qd, gd, Pd = kd.compute(params, hs["s0d"], norm=norm)
        # s0d identical to s0c (same interp doubles, same order)
        assert Qd == Qc
        assert np.array_equal(Pd, Pc)
        for k in gc:
            assert np.array_equal(gd[k], gc[k]), k
        assert kd._last_dnorm == kc5._last_dnorm
        # and both track the v4 per-event pipeline within family tolerance
        assert Qd == pytest.approx(Q4, abs=1e-9)
        assert np.max(np.abs(gd["ck"] - g4["ck"])) < 1e-9
    finally:
        for h in hs.values():
            h.free()
        hp.free()
        k4.free(); kc5.free(); kd.free()


def test_v5_s0d_grouped_rsize(pwa_small):
    """Grouped log-sum at rsize>1: s0d identical to s0c."""
    from tabpwa.cuda._v5_pwa_s0c import CUDAKernelV5PWAS0C as KV5S0C
    from tabpwa.cuda._v5_pwa_s0d import CUDAKernelV5PWAS0D as KV5S0D
    _cfg, kc, data, phsp, params, _p = pwa_small
    resolution_size = 12

    kc5 = _make(KV5S0C, kc, resolution_size=resolution_size)
    kd = _make(KV5S0D, kc, resolution_size=resolution_size)
    h5, hd = kc5.load_data(data), kd.load_data(data)
    hp = kc5.load_data(phsp)
    try:
        norm = float(kc5.compute(params, hp, norm=None)[0])
        Qc, gc, Pc = kc5.compute(params, h5, norm=norm)
        Qd, gd, Pd = kd.compute(params, hd, norm=norm)
        assert Qd == Qc
        assert np.array_equal(Pd, Pc)
        if isinstance(gc, dict):
            for k in gc:
                assert np.array_equal(gd[k], gc[k]), k
        else:
            assert np.array_equal(gd, gc)
        assert kd._last_dnorm == kc5._last_dnorm
    finally:
        h5.free(); hd.free(); hp.free()
        kc5.free(); kd.free()
