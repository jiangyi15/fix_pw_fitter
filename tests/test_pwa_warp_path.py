"""Warp-path (``n_base > 32``) parity for the s0c/s0d/s0e family.

``pwa_amp_reduce_kernel``'s ``n_base > 32`` branch used to run the
per-projection ``dQ/dA_p`` reduction INSIDE ``if (lane == 0)``: lane 0
alone walked ``k = 0 (mod 32)`` and the ``__shfl_down_sync`` tree ran
with 31 inactive lanes (undefined behaviour).  Every N > 32 gradient was
garbage, and P/Q were wrong too.  The repaired path is pinned here
against the fp64 CPU sibling ``cpu_pwa_s0c`` (verified to 3.5e-16
against ``numpy_pwa``) at the benchmark shape ``n_wave = 198`` (N = 99).

The benchmark generator now draws masses inside the config's own
invariant-mass grid.  The u-cache basis encodes ``m²`` as an extended
gamma-table row, and its interp CLAMPS outside that grid — the old
hardcoded ``uniform(2.0, 3.0)`` therefore produced out-of-grid events
(~2 % of them) with up-to-100x P outliers: a harness artefact, not
kernel behaviour.  ``test_benchmark_generator_stays_inside_grid`` is the
regression gate for it.

The same shape is what makes ``n_base > 32``: with the small test configs
(``n_wave = 6``, N = 3) the family's thread path runs instead.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

from benchmark_pwa_kernels import synthetic_case

from tabpwa.cpu._pwa_s0c import CPUKernelPWAS0C
from tabpwa.cuda._v4_pwa_s0b import CUDAKernelV4PWAS0B
from tabpwa.cuda._v4_pwa_s0c import CUDAKernelV4PWAS0C
from tabpwa.cuda._v4_pwa_s0d import CUDAKernelV4PWAS0D
from tabpwa.cuda._v4_pwa_s0e import CUDAKernelV4PWAS0E
from tabpwa.cuda._v5_pwa_s0c import CUDAKernelV5PWAS0C
from tabpwa.cuda._v5_pwa_s0cw import CUDAKernelV5PWAS0CW
from tabpwa.cuda._v5_pwa_s0d import CUDAKernelV5PWAS0D

CFG = "tests/config_pwa.yml"
# benchmark shape: N = n_wave/n_proj = 99 > 32 -> the warp branch
N_COMP, N_PROJ, NEV = 100, 2, 400

# every kernel that carries pwa_amp_reduce_kernel's warp branch (the v5
# entries run the per-event NLL at resolution_size=1, i.e. the v4 math)
BACKENDS = {
    "cuda_v4_pwa_s0b": CUDAKernelV4PWAS0B,
    "cuda_v4_pwa_s0c": CUDAKernelV4PWAS0C,
    "cuda_v4_pwa_s0d": CUDAKernelV4PWAS0D,
    "cuda_v4_pwa_s0e": CUDAKernelV4PWAS0E,
    "cuda_v5_pwa_s0c": CUDAKernelV5PWAS0C,
    "cuda_v5_pwa_s0cw": CUDAKernelV5PWAS0CW,
    "cuda_v5_pwa_s0d": CUDAKernelV5PWAS0D,
}


def _grid(kc):
    """(lo, hi) of the config's invariant-mass grid."""
    lo = float(kc["gamma_min"])
    span = float(kc["gamma_delta"]) * np.asarray(kc["gamma_table"]).shape[-1]
    return lo, lo + span


def _case(nev=NEV, n_comp=N_COMP, n_proj=N_PROJ):
    return synthetic_case(CFG, n_comp, n_proj, nev)


def _rel(a, b):
    """max|a-b| / max|b| — a magnitude-scaled relative difference.

    The CUDA path stores the angular amplitude as float2 (amp-cache
    design) while the CPU sibling is fp64, so cancellation-dominated
    components differ relatively but not absolutely.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


# ── harness gate: the benchmark data must sit inside the mass grid ──

@pytest.mark.parametrize("n_comp,n_proj", [(100, 2), (100, 1), (30, 2)])
def test_benchmark_generator_stays_inside_grid(n_comp, n_proj):
    kc, data, phsp, params, norm, info = synthetic_case(
        CFG, n_comp, n_proj, 500, n_phsp=200)
    lo, hi = _grid(kc)
    for name, d in (("data", data), ("phsp", phsp)):
        m = np.asarray(d["mass"])
        assert m.min() >= lo, (name, m.min(), lo)
        assert m.max() <= hi, (name, m.max(), hi)


# ── the repaired warp path: full parity vs the fp64 CPU sibling ─────

@pytest.mark.parametrize("backend", sorted(BACKENDS))
@pytest.mark.parametrize("norm", [None, 1.3])
def test_warp_path_matches_cpu(backend, norm):
    kc, data, _, params, _, info = _case()
    assert info["N"] > 32                      # the warp branch is in use
    cpu = CPUKernelPWAS0C(kc)
    k = BACKENDS[backend](kc)
    Qc, gc, Pc = cpu.compute(params, cpu.load_data(data), norm=norm)
    Q, g, P = k.compute(params, k.load_data(data), norm=norm)

    assert P.shape == (NEV,)
    for key in ("ck", "m0", "g0"):
        assert np.all(np.isfinite(np.asarray(g[key]))), key
    assert np.all(np.isfinite(P))

    assert _rel(Q, Qc) < 1e-5, ("Q", _rel(Q, Qc))
    assert _rel(P, Pc) < 1e-6, ("P", _rel(P, Pc))
    for key in ("ck", "m0", "g0"):
        assert _rel(g[key], gc[key]) < 1e-4, (key, _rel(g[key], gc[key]))
    if norm is not None:
        assert k._last_dnorm is not None
        assert _rel(k._last_dnorm, cpu._last_dnorm) < 1e-4
    else:
        # v4 kernels report None for an unnormalised call; the v5 log-sum
        # family always stores the device accumulator (0.0 without a norm)
        assert k._last_dnorm in (None, 0.0)


def test_out_of_grid_masses_stay_finite():
    """Outside the grid the m² basis clamps to the edge node (by design).

    No parity is asserted there — the point is that the repaired warp
    path never produces NaN/inf for such events either.
    """
    kc, data, _, params, _, _ = _case()
    lo, hi = _grid(kc)
    mass = np.array(data["mass"], copy=True)
    mass[:7] = hi * 1.05                       # push a few events out
    data = {**data, "mass": mass}
    for name, K in BACKENDS.items():
        k = K(kc)
        Q, g, P = k.compute(params, k.load_data(data), norm=1.3)
        assert np.isfinite(Q), name
        assert np.all(np.isfinite(P)), name
        for key in ("ck", "m0", "g0"):
            assert np.all(np.isfinite(np.asarray(g[key]))), (name, key)
        k.free()
