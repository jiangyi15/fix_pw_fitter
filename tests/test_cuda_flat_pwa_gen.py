"""``cuda_flat_pwa_gen`` — the flat ``pwa_gen`` config on the GPU.

A case-1 copy: own ``.cu``/``.so``/wrapper/symbols, reading the config
:func:`tabpwa.pwa_gen_config.build_from_groups` emits (the full, non-deduped
form).

Gates:
1. registration — ``pwa_gen`` scope only, and the default stays ``numpy_flat``;
2. an all-zero coefficient vector is the empty model: no crash, no garbage
   number, every ``P`` NaN (``D = 0`` exactly).

Everything else — the anchor vs ``numpy_flat`` on ``_km``/``_mb`` (both norms,
rs 1 and 17), the finite-difference gates, the deduped-config rejection and the
parameter-length guards — is covered once, registry-driven, in
``tests/_flat_compute.py``.

Test data stays INSIDE both interpolation grids — outside, the tables clamp
and CPU/GPU diverge (the repo's documented gotcha, and the first thing that
looked like a kernel bug here).
"""
import numpy as np

from tabpwa import Fitter
from tabpwa.backends import backends_for_model, create_backend

CFG_KM = "tests/config_pwa_gen_km.yml"


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


def _pair(cfg, ne=64, seed=0):
    """(numpy_flat backend, cuda backend, params, handle factory, kc)."""
    f = Fitter(cfg)
    f.apply_constrains()
    kc = f.kernel_config
    data = _in_grid(kc, ne, seed)
    x = f.initial_values(seed=1)
    params, _resolved = f.build_params(x)
    ref = create_backend("numpy_flat", kc, model="pwa_gen")
    gpu = create_backend("cuda_flat_pwa_gen", kc, model="pwa_gen")
    return ref, gpu, params, data, kc


# ── gate 1: registration / scope ─────────────────────────────────────

def test_registration():
    allowed = set(backends_for_model("pwa_gen"))
    assert "cuda_flat_pwa_gen" in allowed
    assert "numpy_flat" in allowed
    assert "cuda_flat_pwa_gen" not in backends_for_model("pwa")


def test_default_is_still_the_cpu_reference():
    f = Fitter(CFG_KM)
    assert type(f.backend).__name__ == "NumpyFlatBackend"


# ── gate 2: the parameter surface ────────────────────────────────────

def test_zero_coefficients_zero_the_amplitude():
    """All-zero basis weights ⇒ D = 0 ⇒ no amplitude; must not crash or NaN."""
    ref, gpu, params, data, _kc = _pair(CFG_KM, ne=16)
    h = gpu.load_data(data)
    try:
        p = dict(params)
        p["coeff"] = np.zeros_like(np.asarray(params["coeff"]))
        Q, g, P = gpu.compute(p, h, norm=None)
        # D = 0 exactly, so the amplitude is undefined: the kernel must not
        # crash, and every output must be NaN rather than a garbage number
        # (0/0).  A caller that fixes every weight at zero has an empty model.
        assert not np.any(np.isfinite(P))
    finally:
        ref.free()
        gpu.free()
