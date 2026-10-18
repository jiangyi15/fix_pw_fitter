"""GPU regression: cuda_v5_pwa_s0cw — s0c structure + GROUP-WEIGHTED log NLL.

Run alone — shared-GPU multi-process flakiness.

``cuda_v5_pwa_s0cw`` is the ``cuda_v5_pwa_s0c`` sibling whose grouped
log-sum NLL applies the group TOTAL WEIGHT ``W_g = Σ_{e∈g} w_e``:

    Q          = -Σ_g  W_g · log( (Pb_g/norm + Bb_g) / W_g )
    dNLL/dnorm =  Σ_g  W_g · Pb_g / (norm · (Pb_g + norm·Bb_g))
    c_g        = -W_g / (Pb_g + norm·Bb_g)      (per-event grads: w_e·c_g)

with Pb_g = Σ_{e∈g} w_e·P_e, Bb_g = Σ_{e∈g} w_e·bkg_e.

Verified:
1. group-NORMALISED weights (w = 1/resolution_size per row, Σ w = 1 per
   group ⇒ W_g = 1): the s0cw Q, grads and dnorm are EXACTLY the s0c's
   (the regression anchor);
2. non-normalised weights: Q matches the numpy weighted-log reference
   and the ck/m0/g0 gradients match central finite differences;
3. resolution_size = 1 (unit weights) == s0c at res 1;
4. the same compute twice is bitwise deterministic.
"""

import numpy as np
import pytest

from tabpwa.config_loader import Config


@pytest.fixture(scope="module")
def pwa_small():
    cfg = Config("tests/config_pwa_bw.yml")
    kc = cfg.build_all_index()
    n_proj = int(kc.get("n_proj", 1))
    n_base = kc["matrix_angle"].shape[1] // n_proj
    n_m0 = int(np.max(kc["m0_index"])) + 1
    n_g0 = int(np.max(kc["g0_index"])) + 1
    rs = np.random.RandomState(0)
    ne = 300
    # masses INSIDE the gamma-table grid [gamma_min, gamma_min+bins*delta]
    # (the s0c m² basis is a table row — physical data always is; raw
    # random masses beyond the grid would clamp at the boundary)
    m_lo, m_hi = float(kc["gamma_min"]) + 0.05, \
        float(kc["gamma_min"]) + float(kc["gamma_delta"]) * \
        (gt_bins := kc["gamma_table"].shape[-1]) - 0.05
    data = {"mass": m_lo + (m_hi - m_lo) * rs.rand(ne, 1),
            "q": np.abs(rs.randn(ne, 2)) + 0.3,
            "angle": rs.uniform(-np.pi, np.pi,
                                size=(ne, 1, int(kc["angle_k"].shape[-1]))),
            "weight": np.ones(ne), "bkg": np.ones(ne)}
    phsp = {"mass": m_lo + (m_hi - m_lo) * rs.rand(ne + 50, 1),
            "q": np.abs(rs.randn(ne + 50, 2)) + 0.3,
            "angle": rs.uniform(-np.pi, np.pi, size=(
                ne + 50, 1, int(kc["angle_k"].shape[-1]))),
            "weight": np.ones(ne + 50), "bkg": np.zeros(ne + 50)}
    ck = rs.randn(n_base) + 1j * rs.randn(n_base)
    m0 = 0.7 + 0.4 * np.abs(rs.randn(n_m0))
    g0 = 0.1 + 0.3 * np.abs(rs.randn(n_g0))
    return cfg, kc, data, phsp, {"ck": ck, "m0": m0, "g0": g0}, n_proj


def _make(kernel_cls, kc, **kw):
    try:
        return kernel_cls(kc, batch_size=256, **kw)
    except (RuntimeError, OSError) as e:
        pytest.skip(f"no CUDA device/library: {e}")


def _numpy_weighted_log_nll(P, w, bkg, norm, resolution_size):
    """Reference: Q = -Σ_g W_g·log((Pb_g/norm + Bb_g)/W_g),
    Pb_g = Σ_{e∈g} w·P, Bb_g = Σ_{e∈g} w·bkg, W_g = Σ_{e∈g} w."""
    ng = len(P) // resolution_size
    r = resolution_size
    Pb = (w * P).reshape(ng, r).sum(axis=1)
    Bb = (w * bkg).reshape(ng, r).sum(axis=1)
    W = w.reshape(ng, r).sum(axis=1)
    return -float(np.sum(W * np.log((Pb / norm + Bb) / W)))


def test_s0cw_equals_s0c_with_normalised_weights(pwa_small):
    """Regression anchor: group-normalised weights (Σ w = 1 per group)
    make the s0cw NLL/gradients/dnorm EXACTLY the s0c's."""
    from tabpwa.cuda._v5_pwa_s0c import CUDAKernelV5PWAS0C as KS0C
    from tabpwa.cuda._v5_pwa_s0cw import CUDAKernelV5PWAS0CW as KS0CW
    _cfg, kc, data, phsp, params, _p = pwa_small
    resolution_size = 10
    ne = data["mass"].shape[0]
    assert ne % resolution_size == 0

    norm_weights = np.full(ne, 1.0 / resolution_size)
    data_w = dict(data, weight=norm_weights)

    k_s0c = _make(KS0C, kc, resolution_size=resolution_size)
    k_cw = _make(KS0CW, kc, resolution_size=resolution_size)
    h_s0c, h_cw = k_s0c.load_data(data_w), k_cw.load_data(data_w)
    try:
        Q0, g0_, P0 = k_s0c.compute(params, h_s0c, norm=None)
        Q1, g1_, P1 = k_cw.compute(params, h_cw, norm=None)
        # phsp path (norm=None) is untouched/linear: identical Q and P
        assert Q1 == pytest.approx(Q0, rel=1e-10)
        assert np.max(np.abs(P1 - P0)) < 1e-10 * np.max(np.abs(P0))

        # group NLL path (norm divides the group sums; W_g = 1 exactly)
        Qs, gs, _ = k_s0c.compute(params, h_s0c, norm=Q0)
        Qw, gw, _ = k_cw.compute(params, h_cw, norm=Q0)
        assert Qw == pytest.approx(Qs, rel=1e-10)
        for key in ("ck", "m0", "g0"):
            scale = max(1e-12, np.max(np.abs(gs[key])))
            assert np.allclose(gw[key], gs[key], rtol=1e-10,
                               atol=1e-12 * scale), key
        # d(NLL)/dnorm identical too
        assert k_cw._last_dnorm == pytest.approx(k_s0c._last_dnorm,
                                                 rel=1e-10)
    finally:
        h_s0c.free(); h_cw.free()
        k_s0c.free(); k_cw.free()


def test_s0cw_matches_numpy_weighted_log(pwa_small):
    """Non-normalised weights (w = 1 flat, res=10 ⇒ W_g = 10): the NLL
    matches the numpy weighted-log reference; the ck/m0/g0 gradients
    match central finite differences of that NLL."""
    from tabpwa.cuda._v5_pwa_s0cw import CUDAKernelV5PWAS0CW as KS0CW
    _cfg, kc, data, phsp, params, _p = pwa_small
    resolution_size = 10
    ne = data["mass"].shape[0]
    assert ne % resolution_size == 0

    flat = np.ones(ne)
    data_w = dict(data, weight=flat)

    k = _make(KS0CW, kc, resolution_size=resolution_size)
    h = k.load_data(data_w)
    hp = k.load_data(phsp)          # 350 = 35 groups of 10: no partial group
    try:
        norm = float(k.compute(params, hp, norm=None)[0])
        Q, g, P = k.compute(params, h, norm=norm)
        dn = k._last_dnorm          # capture BEFORE the FD perturbed calls

        ref = _numpy_weighted_log_nll(P, flat, data["bkg"], norm,
                                      resolution_size)
        assert Q == pytest.approx(ref, rel=1e-9)

        # central finite differences against the same weighted-log NLL
        eps = 1e-6

        def nll(p):
            return k.compute(p, h, norm=norm)[0]

        # one ck real part — convention: dQ/dRe = 2·Re(grad_ck)
        idx = 1
        c = params["ck"].copy()
        c[idx] += eps
        qp = nll(dict(params, ck=c))
        c = params["ck"].copy()
        c[idx] -= eps
        qm = nll(dict(params, ck=c))
        num = (qp - qm) / (2 * eps)
        assert num == pytest.approx(2 * np.real(g["ck"][idx]),
                                    rel=1e-4, abs=1e-5)

        # one m0 (the s0/mult chain to m0/g0 happens in the wrapper)
        m = params["m0"].copy()
        m[2] += eps
        qp = nll(dict(params, m0=m))
        m = params["m0"].copy()
        m[2] -= eps
        qm = nll(dict(params, m0=m))
        num = (qp - qm) / (2 * eps)
        assert num == pytest.approx(g["m0"][2], rel=1e-4, abs=1e-2)

        # one g0
        gg = params["g0"].copy()
        gg[1] += eps
        qp = nll(dict(params, g0=gg))
        gg = params["g0"].copy()
        gg[1] -= eps
        qm = nll(dict(params, g0=gg))
        num = (qp - qm) / (2 * eps)
        assert num == pytest.approx(g["g0"][1], rel=1e-4, abs=1e-2)

        # d(NLL)/dnorm of the weighted log: Σ_g W_g·Pb_g/(norm·(Pb+norm·Bb))
        ng = ne // resolution_size
        Pb_g = (flat * P).reshape(ng, resolution_size).sum(axis=1)
        Bb_g = (flat * data["bkg"]).reshape(ng, resolution_size).sum(axis=1)
        W_g = float(resolution_size)          # Σ w per group (w = 1 flat)
        dn_ref = float(np.sum(W_g * Pb_g / (norm * (Pb_g + norm * Bb_g))))
        assert dn == pytest.approx(dn_ref, rel=1e-8)
    finally:
        h.free(); hp.free()
        k.free()


def test_s0cw_res1_equals_s0c_res1(pwa_small):
    """resolution_size = 1 (unit weights): the weighted log reduces to the
    per-group form with W_g = 1 — identical NLL to s0c."""
    from tabpwa.cuda._v5_pwa_s0c import CUDAKernelV5PWAS0C as KS0C
    from tabpwa.cuda._v5_pwa_s0cw import CUDAKernelV5PWAS0CW as KS0CW
    _cfg, kc, data, phsp, params, _p = pwa_small

    k_s0c = _make(KS0C, kc, resolution_size=1)
    k_cw = _make(KS0CW, kc, resolution_size=1)
    h_s0c, h_cw = k_s0c.load_data(data), k_cw.load_data(data)
    hp_s0c = k_s0c.load_data(phsp)
    try:
        norm = float(k_s0c.compute(params, hp_s0c, norm=None)[0])
        Qs, gs, _ = k_s0c.compute(params, h_s0c, norm=norm)
        Qw, gw, _ = k_cw.compute(params, h_cw, norm=norm)
        assert Qw == pytest.approx(Qs, abs=1e-9)
        assert np.max(np.abs(gw["ck"] - gs["ck"])) < 1e-9
        assert np.max(np.abs(gw["m0"] - gs["m0"])) < 1e-9
        assert np.max(np.abs(gw["g0"] - gs["g0"])) < 1e-9
    finally:
        h_s0c.free(); h_cw.free(); hp_s0c.free()
        k_s0c.free(); k_cw.free()


def test_s0cw_zero_weight_group_inert(pwa_small):
    """A fully zero-weight group contributes nothing to Q or dnorm
    (the kernel skips W_g = 0 groups instead of evaluating 0·log(0))."""
    from tabpwa.cuda._v5_pwa_s0cw import CUDAKernelV5PWAS0CW as KS0CW
    _cfg, kc, data, phsp, params, _p = pwa_small
    resolution_size = 10
    ne = data["mass"].shape[0]
    ng = ne // resolution_size

    w = np.ones(ne)
    w[:resolution_size] = 0.0            # group 0 has zero total weight
    data_w = dict(data, weight=w)

    k = _make(KS0CW, kc, resolution_size=resolution_size)
    h = k.load_data(data_w)
    hp = k.load_data(phsp)
    try:
        norm = float(k.compute(params, hp, norm=None)[0])
        Q, _g, P = k.compute(params, h, norm=norm)
        dn = k._last_dnorm

        Pb_g = (w * P).reshape(ng, resolution_size).sum(axis=1)
        Bb_g = (w * data["bkg"]).reshape(ng, resolution_size).sum(axis=1)
        W_g = w.reshape(ng, resolution_size).sum(axis=1)
        live = W_g > 0
        ref = -float(np.sum(
            W_g[live] * np.log((Pb_g[live] / norm + Bb_g[live])
                               / W_g[live])))
        assert Q == pytest.approx(ref, rel=1e-9)
        dn_ref = float(np.sum(
            W_g[live] * Pb_g[live] / (norm * (Pb_g[live]
                                              + norm * Bb_g[live]))))
        assert dn == pytest.approx(dn_ref, rel=1e-8)
        assert np.isfinite(Q) and np.isfinite(dn)
    finally:
        h.free(); hp.free()
        k.free()


def test_s0cw_deterministic(pwa_small):
    """The same compute twice gives the identical Q/grads (bitwise)."""
    from tabpwa.cuda._v5_pwa_s0cw import CUDAKernelV5PWAS0CW as KS0CW
    _cfg, kc, data, _phsp, params, _p = pwa_small

    k = _make(KS0CW, kc, resolution_size=10)
    h = k.load_data(data)
    try:
        norm = 42.0   # arbitrary fixed norm: exercises the group path
        Q1, g1, P1 = k.compute(params, h, norm=norm)
        Q2, g2, P2 = k.compute(params, h, norm=norm)
        assert Q1 == Q2
        for key in ("ck", "m0", "g0"):
            assert np.array_equal(g1[key], g2[key]), key
        assert np.array_equal(P1, P2)
        assert k._last_dnorm == k._last_dnorm
    finally:
        h.free()
        k.free()
