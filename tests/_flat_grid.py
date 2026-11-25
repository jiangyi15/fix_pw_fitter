"""Test-side helper shared by the flat ``pwa_gen`` suites: sample + configs.

One definition, so every suite that loads a synthetic sample on the flat config
uses the SAME numbers — a second copy of ``_in_grid`` would drift and shift
every measured floor with it.

``_in_grid`` is the ``cuda_flat_cache_pwa_gen`` recipe moved here —
behaviourally identical and bit-for-bit reproduced (the same ``RandomState``
calls in the same order, so a sample depends only on ``(kc, ne, seed)``).  It
was REFORMATTED on the move, so the claim is pinned rather than asserted: see
``test_the_shared_sample_is_bit_for_bit_the_moved_recipe`` in
``test_flat_dedup.py``, which reproduces the pre-move body and compares every
array with ``np.array_equal``.  Events are strictly INSIDE both interpolation
grids, because outside them the tables clamp and the CPU/GPU paths diverge (a
data problem that looks exactly like a kernel bug).

The ``CFG_*`` paths are the flat config fixtures; ``QUALIFYING`` is the ones
with exactly one BW per chain (a ``_mb`` chain has two, so the ``n_res1``
kernels refuse it).
"""
from functools import lru_cache

import numpy as np

CFG_ONE = "tests/config_pwa_gen.yml"
CFG_KM = "tests/config_pwa_gen_km.yml"
CFG_MB = "tests/config_pwa_gen_mb.yml"
CFG_IDENT = "tests/config_pwa_gen_ident.yml"
CFG_SAME = "tests/config_pwa_gen_same.yml"
CFG_CP = "tests/config_pwa_gen_cp.yml"
#: every fixture with exactly one BW per chain (a `_mb` chain has two)
QUALIFYING = [CFG_ONE, CFG_KM, CFG_IDENT, CFG_SAME, CFG_CP]


def _in_grid(kc, ne, seed=0):
    """Events strictly inside BOTH interpolation grids.

    Outside, the tables clamp and the CPU/GPU paths diverge — a data problem
    that looks exactly like a kernel bug.
    """
    rng = np.random.RandomState(seed)
    n_basis = int(np.asarray(kc["basis_table"]).shape[1])
    n_fl = int(np.asarray(kc["fl_table"]).shape[-1])
    m_min = float(kc["m_min"])
    m_del = float(kc["m_delta"])
    q_min = float(kc["q_min"])
    q_del = float(kc["q_delta"])
    n_mass = int(np.max(kc["num_m_index"])) + 1
    n_q = int(np.max(kc["fl_q_index"])) + 1
    n_at = int(np.max(kc["angle_index"])) + 1
    nac = int(np.asarray(kc["angle_k"]).shape[1])
    return {
        "mass": m_min + 0.05 * m_del * n_basis
                + rng.rand(ne, n_mass) * (0.9 * m_del * n_basis),
        "q": q_min + 0.05 * q_del * n_fl
             + rng.rand(ne, n_q) * (0.9 * q_del * n_fl),
        "angle": rng.uniform(-np.pi, np.pi, (ne, n_at, nac)),
        "weight": 0.5 + rng.rand(ne),
        "bkg": 0.05 + 0.1 * rng.rand(ne),
    }


@lru_cache(maxsize=None)
def _anchor_inputs_cached(cfg):
    from tabpwa import Fitter

    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    params, _ = f.build_params(f.initial_values(seed=1))
    return kc, params


def anchor_inputs(cfg):
    """``(kc, params)`` for a fixture — built ONCE per session, not per case.

    Both are pure functions of the config (``apply_constrains`` then
    ``build_params(initial_values(seed=1))``), while the harness rebuilt them in
    every parametrised cell: ``Fitter(cfg)`` MEASURES 5-27 ms per build — the
    per-(wave, factor, row) Python bookkeeping, NOT the ``n_interp`` grid
    (measured flat from 4000 down to 500 points) — repeated 4-8x per fixture
    and again by every flat file.  ``n_interp`` is therefore not the lever; this
    cache is.

    The ``Fitter`` is dropped (tests that need live data build their own), the
    kc is shared (no backend writes into it), and the params are shallow-copied
    per call so a test that mutates one cannot poison the next case.
    """
    kc, params = _anchor_inputs_cached(cfg)
    return kc, {k: (v.copy() if hasattr(v, "copy") else v)
                for k, v in params.items()}
