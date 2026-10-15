"""GPU regression: cuda_v4_pwa_pslot — projection-sum PWA with PER-SLOT ck.

Run alone — shared-GPU multi-file segfault, pre-existing.

The kernel is the cuda_v4_pwa sibling whose only semantic delta is that the
coupling array ``ck`` is PER-SLOT: it has length ``n_wave = n_proj·N``
(entry ``(p, t)`` reads ``ck[p·N + t]``) instead of the shared length ``N``,
and the ck-gradient writes each slot with NO cross-projection accumulation.

Independent-reference design (does NOT import the Lane-A polarised classes):
the reference is the committed ``NumpyPWA`` kernel, loaded with the same
data / m0 / g0, plus an inline finite-difference check of a single slot.

Config layout: the polarised model's kernel config already carries
``n_proj = P`` projections; ``_projection_duplicate(kc, P)`` p-major
duplicates it once more, giving ``n_wave = P²·N`` entries with the same
``n_proj = P`` (so the pslot kernel sees ``N_exp = P·N`` slots per
projection and ``ck`` length ``P²·N``).

Exact equivalence used below (the naive "un-expanded / sum-(m,j)" reading is
NOT exact — the two duplications give different projection multiplicities):
with ``ck2 = np.tile(ck, P²)`` the pslot intensity equals ``NumpyPWA`` on the
**expanded** config with the same-shared ``ck`` replicated ``P`` times
(``np.tile(ck, P)``); the per-slot ck-gradient, summed over the outer
projection axis (``reshape(P, P, N).sum(axis=0)``), reproduces the shared-ck
gradient exactly.
"""

import numpy as np
import pytest

from tabpwa.amp_model import build_amplitude_model
from tabpwa.base_model import _projection_duplicate
from tabpwa.config_loader import RawConfig


@pytest.fixture(scope="module")
def pslot_fixture():
    rc = RawConfig("tests/config_pwa_bw.yml")
    rc.dic["amp_model"] = "polarised_pwa"
    model = build_amplitude_model(rc.dic)
    kc = model.build_kernel_config()          # n_proj = P projections
    P = int(kc["n_proj"])
    N = kc["matrix_angle"].shape[1] // P      # base waves per projection
    kc_exp = _projection_duplicate(kc, P)     # n_wave = P²·N, n_proj stays P

    n_m0 = int(np.max(kc_exp["m0_index"])) + 1
    n_g0 = int(np.max(kc_exp["g0_index"])) + 1
    n_mass = int(np.max(kc_exp["mass_index"])) + 1
    n_mom = int(np.max(kc_exp["fl_q_index"])) + 1
    n_at = int(np.max(kc_exp["angle_index"])) + 1
    n_comp = int(kc_exp["angle_k"].shape[-1])

    rs = np.random.RandomState(0)
    ne = 60
    data = {
        "mass": 2.0 + 0.6 * rs.rand(ne, n_mass),
        "q": 0.3 + np.abs(rs.randn(ne, n_mom)),
        "angle": rs.uniform(-np.pi, np.pi, (ne, n_at, n_comp)),
        "weight": np.ones(ne),
        "bkg": np.ones(ne),
    }
    ck = rs.randn(N) + 1j * rs.randn(N)
    m0 = 0.5 + 0.4 * np.abs(rs.randn(n_m0))
    g0 = 0.1 + 0.3 * np.abs(rs.randn(n_g0))
    return kc_exp, kc, P, N, data, {"ck": ck, "m0": m0, "g0": g0}


def _make(kernel_cls, kc):
    try:
        return kernel_cls(kc, batch_size=16)
    except (RuntimeError, OSError) as e:
        pytest.skip(f"no usable CUDA device/library: {e}")


def test_pslot_matches_analytic_reference(pslot_fixture):
    from tabpwa.cuda._v4_pwa_pslot import CUDAKernelV4Pslot
    from tabpwa.numpy_pwa import NumpyPWA

    kc_exp, kc, P, N, data, params = pslot_fixture
    ck, m0, g0 = params["ck"], params["m0"], params["g0"]

    # ── per-slot replication: ck2[slot] carries the shared ck ──────────
    ck2 = np.tile(ck, P * P)                 # length P²·N == n_wave
    assert len(ck2) == kc_exp["matrix_angle"].shape[1]

    k = _make(CUDAKernelV4Pslot, kc_exp)
    try:
        h = k.load_data(data)
        Q, g, Pe = k.compute({"ck": ck2, "m0": m0, "g0": g0}, h, norm=None)
        assert Pe is not None
        assert len(g["ck"]) == kc_exp["matrix_angle"].shape[1]
        assert len(g["m0"]) == len(m0)
        assert len(g["g0"]) == len(g0)
    finally:
        h.free()
        k.free()

    # ── reference: NumpyPWA on the expanded config, shared tile(ck, P) ──
    npw = NumpyPWA(kc_exp)
    h2 = npw.load_data(data)
    Qr, gr, _Pr = npw.compute({"ck": np.tile(ck, P), "m0": m0, "g0": g0},
                              h2, norm=None)

    # fp64 CUDA vs the fp64 numpy reference; the fp32 angular cache gives
    # ~1e-7 relative noise (same convention as test_pwa_cuda_cache.py).
    assert Q == pytest.approx(Qr, rel=1e-6)
    scale_m0 = max(1e-12, np.max(np.abs(gr["m0"])))
    scale_g0 = max(1e-12, np.max(np.abs(gr["g0"])))
    assert np.allclose(g["m0"], gr["m0"], rtol=1e-5, atol=1e-6 * scale_m0)
    assert np.allclose(g["g0"], gr["g0"], rtol=1e-5, atol=1e-6 * scale_g0)

    # per-slot ck-gradient: the shared-ck gradient is the sum over the outer
    # projection axis of the per-slot partials (no cross-p accumulation).
    g_sum = g["ck"].reshape(P, P, N).sum(axis=0).reshape(-1)
    scale_ck = max(1e-12, np.max(np.abs(gr["ck"])))
    assert np.allclose(g_sum, gr["ck"], rtol=1e-5, atol=1e-6 * scale_ck)

    # ── exact reduction to the shared-ck (UN-expanded) model ────────────
    # Embed the single shared ck block-diagonally in the per-slot layout:
    # ck2[m, j, k] = ck[k]·δ_{mj}.  Then A_p = Σ_k ck[k]·a_{p,k} and the
    # expanded pslot intensity equals NumpyPWA on the un-expanded config.
    ck_diag = np.zeros((P, P, N), dtype=complex)
    for i in range(P):
        ck_diag[i, i, :] = ck
    ck_diag = ck_diag.reshape(-1)

    k2 = _make(CUDAKernelV4Pslot, kc_exp)
    try:
        h3 = k2.load_data(data)
        Qd, gd, _Pd = k2.compute({"ck": ck_diag, "m0": m0, "g0": g0},
                                 h3, norm=None)
    finally:
        h3.free()
        k2.free()

    npw_u = NumpyPWA(kc)                     # un-expanded, shared ck
    h4 = npw_u.load_data(data)
    Qu, gu, _Pu = npw_u.compute({"ck": ck, "m0": m0, "g0": g0},
                                h4, norm=None)

    assert Qd == pytest.approx(Qu, rel=1e-6)
    assert np.allclose(gd["m0"], gu["m0"], rtol=1e-5,
                       atol=1e-6 * max(1e-12, np.max(np.abs(gu["m0"]))))
    assert np.allclose(gd["g0"], gu["g0"], rtol=1e-5,
                       atol=1e-6 * max(1e-12, np.max(np.abs(gu["g0"]))))
    g_diag = gd["ck"].reshape(P, P, N)[np.arange(P), np.arange(P), :].sum(0)
    assert np.allclose(g_diag, gu["ck"], rtol=1e-5,
                       atol=1e-6 * max(1e-12, np.max(np.abs(gu["ck"]))))


def test_pslot_per_slot_response_finite_difference(pslot_fixture):
    from tabpwa.cuda._v4_pwa_pslot import CUDAKernelV4Pslot

    kc_exp, _kc, P, N, data, params = pslot_fixture
    ck, m0, g0 = params["ck"], params["m0"], params["g0"]
    ck2 = np.tile(ck, P * P)

    k = _make(CUDAKernelV4Pslot, kc_exp)
    try:
        h = k.load_data(data)

        def qf(v):
            return k.compute({"ck": v, "m0": m0, "g0": g0}, h, norm=None)[0]

        Q0, g, _P = k.compute({"ck": ck2, "m0": m0, "g0": g0}, h, norm=None)

        # a slot in the second projection block (off the shared-replica
        # diagonal) — its ck-gradient is nonzero, so the slot is genuinely
        # per-projection (a shared-ck kernel would not expose it at all).
        s = 1 * (P * N) + 0 * N + 1
        assert g["ck"][s] != 0.0

        eps = 1e-6

        v = ck2.copy()
        v[s] += eps
        qp = qf(v)
        v[s] -= 2 * eps
        qm = qf(v)
        assert qp != Q0 or qm != Q0          # perturbing one slot moves NLL
        fd_re = (qp - qm) / (2 * eps)
        # Wirtinger: dQ/dRe(ck) = 2·Re(grad_ck)
        assert fd_re == pytest.approx(2 * g["ck"][s].real, rel=1e-4)

        v = ck2.copy()
        v[s] += 1j * eps
        qp = qf(v)
        v[s] -= 2j * eps
        qm = qf(v)
        fd_im = (qp - qm) / (2 * eps)
        # Wirtinger: dQ/dIm(ck) = -2·Im(grad_ck)
        assert fd_im == pytest.approx(-2 * g["ck"][s].imag, rel=1e-4)
    finally:
        h.free()
        k.free()


def test_pslot_ck_length_validation(pslot_fixture):
    from tabpwa.cuda._v4_pwa_pslot import CUDAKernelV4Pslot

    kc_exp, _kc, _P, N, data, params = pslot_fixture
    k = _make(CUDAKernelV4Pslot, kc_exp)
    try:
        h = k.load_data(data)
        # a shared-length (N) ck is rejected: pslot requires n_wave = P·N
        with pytest.raises(ValueError, match="cuda_v4_pwa_pslot"):
            k.compute({"ck": params["ck"], "m0": params["m0"],
                       "g0": params["g0"]}, h, norm=None)
    finally:
        h.free()
        k.free()


def test_polarised_wrapper_over_cuda_pslot_matches_direct():
    """End-to-end: the `polarised` wrapper with base=cuda_v4_pwa_pslot
    (expansion + per-slot ck''=ck·L + Cholesky-adjoint spin grads) must
    reproduce the direct rho-contraction path (NLL + full flat gradient,
    including the spin-parameter entries) at nonzero polarisation."""
    import yaml

    from tabpwa import Fitter

    cfg = yaml.safe_load(open("tests/config_pwa_bw.yml"))
    cfg["amp_model"] = "polarised_pwa"

    try:
        f_dir = Fitter(dict(cfg), backend="polarised")
        f_gpu = Fitter(dict(cfg),
                       backend={"name": "polarised",
                                "base": "cuda_v4_pwa_pslot"})
    except (RuntimeError, OSError) as e:
        pytest.skip(f"no usable CUDA device/library: {e}")

    def _synth(fitter, n, seed):
        kc = fitter.kernel_config
        rs = np.random.RandomState(seed)
        n_at = int(np.max(kc["angle_index"])) + 1 if len(kc["angle_index"]) else 1
        nac = int(np.asarray(kc["angle_k"]).shape[-1])
        n_mass = int(np.max(kc["mass_index"])) + 1
        n_q = int(np.max(kc["fl_q_index"])) + 1
        gmin = float(kc["gamma_min"])
        gspan = float(kc["gamma_delta"]) * np.asarray(kc["gamma_table"]).shape[-1]
        # canonical pipeline layout (pwa_build.build_tree_event_data):
        # angle (n, n_topo, n_base) — the CUDA loader requires this shape
        return {"mass": (gmin + 0.05) + (0.9 * gspan - 0.1) * rs.rand(n, n_mass),
                "q": np.abs(rs.randn(n, n_q)) + 0.3,
                "angle": rs.uniform(-np.pi, np.pi, size=(n, n_at, nac)),
                "weight": np.ones(n), "bkg": np.ones(n)}

    for f in (f_dir, f_gpu):
        f.set_phsp(_synth(f, 80, 101))
        f.set_data(_synth(f, 80, 1))

    x = f_dir.initial_values(seed=3)
    free = f_dir.free_param_names()
    for name in ("jpsi_px", "jpsi_py", "jpsi_pz"):   # nonzero polarisation
        x[free.index(name)] = 0.2

    n1, g1 = f_dir.get_nll(x)
    n2, g2 = f_gpu.get_nll(x)

    # the CUDA kernel's fp32 angular cache gives ~1e-7 relative noise
    # (same convention as the rest of this file)
    assert n2 == pytest.approx(n1, rel=1e-6)
    scale = max(1e-12, np.max(np.abs(g1)))
    assert np.allclose(g2, g1, rtol=1e-4, atol=1e-6 * scale)
