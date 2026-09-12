"""GPU regression: cuda_v5_pwa log-sum-group NLL.

v5 is the v4 projection-sum PWA with ONE difference: in the data-NLL branch
(``norm`` is not None) the per-event ``-log`` is replaced by one log per
``resolution_size``-sized group of events,

    Q = -Σ_groups log Σ_{e∈g} w_e·(P_e/norm + bkg_e)

The phase-space path (``norm=None``) stays the linear Σ w·P, and the
per-event returned P stays P_sig = Σ_p |A_p|².  ``resolution_size=1`` with unit
weights reproduces v4 exactly; the gradients (including ``d(NLL)/d(norm)``,
exposed through ``_last_dnorm``) follow the per-group chain rule.
"""

import numpy as np
import pytest

from ampfit.config_loader import Config


@pytest.fixture(scope="module")
def pwa_small():
    cfg = Config("tests/config_pwa.yml")
    kc = cfg.build_all_index()
    n_proj = int(kc.get("n_proj", 1))
    n_base = kc["matrix_angle"].shape[1] // n_proj
    n_m0 = int(np.max(kc["m0_index"])) + 1
    n_g0 = int(np.max(kc["g0_index"])) + 1
    rs = np.random.RandomState(0)
    ne = 300
    data = {"mass": 2.2 + 0.5 * rs.randn(ne, 1),
            "q": np.abs(rs.randn(ne, 2)) + 0.3,
            "angle": rs.uniform(-np.pi, np.pi,
                                size=(ne, 1, int(kc["angle_k"].shape[-1]))),
            "weight": np.ones(ne), "bkg": np.ones(ne)}
    phsp = {"mass": 2.2 + 0.5 * rs.randn(ne + 50, 1),
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
    except RuntimeError as e:
        pytest.skip(f"no CUDA device: {e}")


def _numpy_group_nll(P, w, bkg, norm, resolution_size):
    """Reference: Q = -Σ_g log Σ_{e∈g} w_e·(P_e/norm + bkg_e)."""
    ng = len(P) // resolution_size
    Pb = (w * P / norm + w * bkg).reshape(ng, resolution_size).sum(axis=1)
    return -float(np.sum(np.log(Pb)))


def test_v5_batch1_equals_v4(pwa_small):
    from ampfit.cuda._v4_pwa import CUDAKernelV4PWA as KV4
    from ampfit.cuda._v5_pwa import CUDAKernelV5PWA as KV5
    _cfg, kc, data, phsp, params, _p = pwa_small

    kv, k5 = _make(KV4, kc), _make(KV5, kc, resolution_size=1)
    hv, h5 = kv.load_data(data), k5.load_data(data)
    hp = kv.load_data(phsp)
    try:
        norm = kv.compute(params, hp, norm=None)[0]
        Qv, gv, Pv = kv.compute(params, hv, norm=norm)
        Q5, g5, P5 = k5.compute(params, h5, norm=norm)
        assert Qv == pytest.approx(Q5, abs=1e-9)
        assert np.max(np.abs(Pv - P5)) == 0.0
        assert np.max(np.abs(gv["ck"] - g5["ck"])) < 1e-9
        assert np.max(np.abs(gv["m0"] - g5["m0"])) < 1e-9
        assert np.max(np.abs(gv["g0"] - g5["g0"])) < 1e-9
        # d(NLL)/d(norm): v4 per-event closed form == v5 single-event group
        w, b = np.ones(len(Pv)), np.ones(len(Pv))
        dn_ref = np.sum(w * Pv / (norm * (Pv + b * norm)))
        assert dn_ref == pytest.approx(k5._last_dnorm, abs=1e-6)
    finally:
        for h in (hv, h5, hp):
            h.free()
        kv.free(); k5.free()


def test_v5_group_matches_numpy_reference_and_phsp_identical(pwa_small):
    from ampfit.cuda._v4_pwa import CUDAKernelV4PWA as KV4
    from ampfit.cuda._v5_pwa import CUDAKernelV5PWA as KV5
    _cfg, kc, data, phsp, params, _p = pwa_small
    resolution_size = 12

    kv, k5 = _make(KV4, kc), _make(KV5, kc, resolution_size=resolution_size)
    h5 = k5.load_data(data)
    hp5, hpv = k5.load_data(phsp), kv.load_data(phsp)
    try:
        # phase-space path must be unchanged from v4 (linear Σ w·P)
        Qn5, gn5, _ = k5.compute(params, hp5, norm=None)
        Qnv, gnv, _ = kv.compute(params, hpv, norm=None)
        assert Qn5 == pytest.approx(Qnv, abs=1e-9)
        assert np.max(np.abs(gn5["ck"] - gnv["ck"])) < 1e-9

        # group NLL against the plain numpy reference over returned P
        norm = float(Qnv)
        Q, _g, P = k5.compute(params, h5, norm=norm)
        ref = _numpy_group_nll(P, data["weight"], data["bkg"], norm,
                               resolution_size)
        assert Q == pytest.approx(ref, abs=1e-6)
    finally:
        h5.free(); hp5.free(); hpv.free()
        kv.free(); k5.free()


def test_v5_group_grads_match_finite_difference(pwa_small):
    from ampfit.cuda._v5_pwa import CUDAKernelV5PWA as KV5
    _cfg, kc, data, phsp, params, _p = pwa_small
    resolution_size = 15
    ne = data["mass"].shape[0]
    assert ne % resolution_size == 0

    k5 = _make(KV5, kc, resolution_size=resolution_size)
    h5 = k5.load_data(data)
    hp = k5.load_data(phsp)
    try:
        norm = float(k5.compute(params, hp, norm=None)[0])
        _Q, g, _P = k5.compute(params, h5, norm=norm)

        def nll(p, norm):
            return k5.compute(p, h5, norm=norm)[0]

        eps = 1e-6
        # dnorm
        dn = (nll(params, norm + eps) - nll(params, norm - eps)) / (2 * eps)
        assert dn == pytest.approx(k5._last_dnorm, rel=1e-4, abs=1e-6)

        # one ck real part — convention: dQ/dRe = 2·Re(grad_ck)
        idx = 0
        c = params["ck"].copy()
        c[idx] += eps
        cp = dict(params, ck=c)
        c = params["ck"].copy()
        c[idx] -= eps
        cm = dict(params, ck=c)
        num = (nll(cp, norm) - nll(cm, norm)) / (2 * eps)
        assert num == pytest.approx(2 * np.real(g["ck"][idx]), rel=1e-4,
                                    abs=1e-5)

        # one m0 (real parameter gradient returned directly)
        m = params["m0"].copy()
        m[2] += eps
        num = (nll(dict(params, m0=m), norm) - nll(params, norm)) / eps
        assert num == pytest.approx(g["m0"][2], rel=1e-4, abs=1e-2)
    finally:
        h5.free(); hp.free()
        k5.free()


def test_v5_weights_inside_group_sum(pwa_small):
    """Non-unit weights enter the group sum (Σ w·(P/norm+bkg)), not a
    per-event log factor — resolution_size=1 with w≠1 therefore differs from v4."""
    from ampfit.cuda._v4_pwa import CUDAKernelV4PWA as KV4
    from ampfit.cuda._v5_pwa import CUDAKernelV5PWA as KV5
    _cfg, kc, data, _phsp, params, _p = pwa_small

    wdata = dict(data, weight=0.5 + 0.5 * np.abs(
        np.random.RandomState(3).randn(data["mass"].shape[0])))
    kv, k5 = _make(KV4, kc), _make(KV5, kc, resolution_size=1)
    hv, h5 = kv.load_data(wdata), k5.load_data(wdata)
    try:
        norm = 10.0
        Qv, _g, P = kv.compute(params, hv, norm=norm)
        Q5, _g5, _ = k5.compute(params, h5, norm=norm)
        # v4 = -Σ w·log(P/norm+bkg); v5(group1) = -Σ log(w·(P/norm+bkg))
        ref5 = -float(np.sum(np.log(
            wdata["weight"] * (P / norm + wdata["bkg"]))))
        assert Q5 == pytest.approx(ref5, abs=1e-6)
        assert abs(Qv - Q5) > 1e-3
    finally:
        hv.free(); h5.free()
        kv.free(); k5.free()
