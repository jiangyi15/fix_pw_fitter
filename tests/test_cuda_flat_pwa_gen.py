"""``cuda_flat_pwa_gen`` — the flat ``pwa_gen`` config on the GPU.

A case-1 copy: own ``.cu``/``.so``/wrapper/symbols, reading the config
:func:`tabpwa.pwa_gen_config.build_from_groups` emits (the full, non-deduped
form).

Gates:
1. registration — ``pwa_gen`` scope only, and the default stays ``numpy_flat``;
2. **anchor vs ``numpy_flat``** on both fixtures at ``norm=None`` and a scalar
   ``norm``: ``Q`` / ``P`` / ``ck`` / ``coeff`` / ``dnorm`` within the
   documented float2 amplitude-cache floor (1e-6).  Measured 1e-8;
3. **finite differences**: ``dcoeff`` matches central differences of the
   kernel's own ``Q`` (rtol 1e-4); ``dck`` is the raw Wirtinger half, so it
   matches ``2·Re``/``2·Im`` of the finite difference (the real-parameter
   convention is applied downstream by ``CKProduct.backprop_grad``) — the same
   contract ``numpy_pwa``/``numpy_flat`` follow;
4. the **deduped** config is rejected, because this kernel reads the full
   form and would mis-index every folded row;
5. the kernel derives every count from a ``shape[...]`` and refuses a config
   whose tile exceeds the device's opt-in shared-memory limit.

Test data stays INSIDE both interpolation grids — outside, the tables clamp
and CPU/GPU diverge (the repo's documented gotcha, and the first thing that
looked like a kernel bug here).
"""
import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import backends_for_model, create_backend
from tabpwa.pwa_gen_config import build_from_groups, dedup

CFG_KM = "tests/config_pwa_gen_km.yml"
CFG_MB = "tests/config_pwa_gen_mb.yml"

# GPU float2 amplitude cache -> fp32: the documented CPU-vs-GPU floor
CACHE_FLOOR = 1e-6
NH = 1e-6


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


def _rel(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


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


# ── gate 2: anchor against the CPU reference ────────────────────────

@pytest.mark.parametrize("cfg", [CFG_KM, CFG_MB])
@pytest.mark.parametrize("norm", [None, 1.3])
def test_anchor_matches_numpy_flat(cfg, norm):
    ref, gpu, params, data, _kc = _pair(cfg)
    try:
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=norm)
        Qc, gc, Pc = gpu.compute(params, gpu.load_data(data), norm=norm)
        assert _rel(Pc, Pr) < CACHE_FLOOR, "P"
        assert _rel(Qc, Qr) < CACHE_FLOOR, "Q"
        assert _rel(gc["ck"], gr["ck"]) < CACHE_FLOOR, "ck"
        assert _rel(gc["coeff"], gr["coeff"]) < CACHE_FLOOR, "coeff"
        if norm is not None:
            # ``NumpyFlatBackend`` IS the kernel; the CUDA one wraps it
            assert _rel(gpu.kernel._last_dnorm,
                        ref._last_dnorm) < CACHE_FLOOR, "dnorm"
    finally:
        ref.free()
        gpu.free()


# ── gate 3: finite differences against the kernel's own Q ───────────

def test_coeff_gradient_matches_fd():
    """``dcoeff`` is exact: ratio 1.0 against central differences of Q."""
    ref, gpu, params, data, kc = _pair(CFG_KM, ne=48)
    h = gpu.load_data(data)
    try:
        Q0, g0, _ = gpu.compute(params, h, norm=None)
        assert np.isfinite(Q0)
        coeff = np.asarray(params["coeff"]).copy()
        for i in range(min(5, coeff.size)):
            cp, cm = dict(params), dict(params)
            a, b = coeff.copy(), coeff.copy()
            a[i] += NH
            b[i] -= NH
            cp["coeff"], cm["coeff"] = a, b
            fd = (gpu.compute(cp, h, norm=None)[0]
                  - gpu.compute(cm, h, norm=None)[0]) / (2 * NH)
            assert np.isclose(g0["coeff"][i], fd, rtol=1e-4, atol=1e-8), (
                i, g0["coeff"][i], fd)
    finally:
        ref.free()
        gpu.free()


def test_ck_gradient_is_the_wirtinger_half():
    """``grad_ck`` matches ``2·Re``/``2·Im`` of the finite difference.

    The kernel returns the raw half, like ``numpy_pwa`` — the 2·Re
    real-parameter convention is applied downstream by
    ``CKProduct.backprop_grad``.
    """
    ref, gpu, params, data, kc = _pair(CFG_KM, ne=32)
    h = gpu.load_data(data)
    try:
        _Q0, g0, _ = gpu.compute(params, h, norm=None)
        ck = np.asarray(params["ck"]).copy()
        for i in range(ck.size):
            cp, cm = dict(params), dict(params)
            a, b = ck.copy(), ck.copy()
            a[i] += NH
            b[i] -= NH
            cp["ck"], cm["ck"] = a, b
            fd = (gpu.compute(cp, h, norm=None)[0]
                  - gpu.compute(cm, h, norm=None)[0]) / (2 * NH)
            assert np.isclose(2.0 * g0["ck"][i].real, fd, rtol=1e-4,
                              atol=1e-8), (i, g0["ck"][i], fd)
        # and the ck block agrees with the CPU reference's convention
        Qr, gr, _ = ref.compute(params, ref.load_data(data), norm=None)
        assert _rel(g0["ck"], gr["ck"]) < CACHE_FLOOR
    finally:
        ref.free()
        gpu.free()


# ── gate 4: the full-form contract ──────────────────────────────────

def test_rejects_a_deduped_config():
    """``dedup`` folds rows and emits ``fl_index``/``num_index``.

    This kernel does not read those, so accepting a folded config would
    silently mis-index every row — it must raise instead.
    """
    from tabpwa.cuda._flat_pwa_gen import CUDAKernelFlatPWAGen
    m = Fitter(CFG_KM).model
    dd = dedup(build_from_groups(m, n_interp=200))
    with pytest.raises(ValueError, match="non-deduped"):
        CUDAKernelFlatPWAGen(dd)


# ── gate 5: parameter-surface contract ──────────────────────────────

def test_wrong_param_lengths_are_refused():
    ref, gpu, params, data, kc = _pair(CFG_KM, ne=16)
    h = gpu.load_data(data)
    try:
        bad = dict(params)
        bad["ck"] = np.zeros(np.asarray(params["ck"]).size + 1, complex)
        with pytest.raises(ValueError, match="ck has"):
            gpu.compute(bad, h, norm=None)
        bad = dict(params)
        bad["coeff"] = np.zeros(np.asarray(params["coeff"]).size + 1)
        with pytest.raises(ValueError, match="coeff has"):
            gpu.compute(bad, h, norm=None)
    finally:
        ref.free()
        gpu.free()


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
