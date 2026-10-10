"""GPU regression: cuda_v5_pwa_s0 — direct pole parameters + log-sum NLL.

``cuda_v5_pwa_s0`` combines the ``cuda_v4_pwa_s0`` kernel structure (one BW
per wave, outside-transformed parameters ``s0 = m0²`` / ``mult = m0·g0``,
device outputs dQ/ds0 + dQ/dmult, host chains to m0/g0) with the cuda_v5
grouped log-sum data NLL (one ``-log`` per ``resolution_size``-sized group;
``dNLL/dnorm`` accumulated per group on the C path).

Verified:
1. ``resolution_size=1`` with unit weights reproduces ``cuda_v4_pwa_s0``
   exactly (NLL, P, ck/m0/g0 gradients);
2. at ``resolution_size > 1`` the NLL matches the generic ``cuda_v5_pwa``
   (same likelihood, different BW parametrisation) and the numpy group
   reference over the returned P;
3. the gradients (incl. ``d(NLL)/dnorm``) match finite differences.
"""

import numpy as np
import pytest

from tabpwa.config_loader import Config


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


def test_v5_s0_batch1_equals_v4_s0(pwa_small):
    from tabpwa.cuda._v4_pwa_s0 import CUDAKernelV4PWAS0 as KV4S0
    from tabpwa.cuda._v5_pwa_s0 import CUDAKernelV5PWAS0 as KV5S0
    _cfg, kc, data, phsp, params, _p = pwa_small

    k4, k5 = _make(KV4S0, kc), _make(KV5S0, kc, resolution_size=1)
    h4, h5 = k4.load_data(data), k5.load_data(data)
    hp = k4.load_data(phsp)
    try:
        norm = k4.compute(params, hp, norm=None)[0]
        Q4, g4, P4 = k4.compute(params, h4, norm=norm)
        Q5, g5, P5 = k5.compute(params, h5, norm=norm)
        assert Q4 == pytest.approx(Q5, abs=1e-9)
        assert np.max(np.abs(P4 - P5)) == 0.0
        assert np.max(np.abs(g4["ck"] - g5["ck"])) < 1e-9
        assert np.max(np.abs(g4["m0"] - g5["m0"])) < 1e-9
        assert np.max(np.abs(g4["g0"] - g5["g0"])) < 1e-9
        # d(NLL)/d(norm): single-event groups == the v4 per-event form
        w, b = np.ones(len(P4)), np.ones(len(P4))
        dn_ref = np.sum(w * P4 / (norm * (P4 + b * norm)))
        assert dn_ref == pytest.approx(k5._last_dnorm, abs=1e-6)
    finally:
        for h in (h4, h5, hp):
            h.free()
        k4.free(); k5.free()


def test_v5_s0_group_matches_v5_and_numpy(pwa_small):
    from tabpwa.cuda._v5_pwa import CUDAKernelV5PWA as KV5
    from tabpwa.cuda._v5_pwa_s0 import CUDAKernelV5PWAS0 as KV5S0
    _cfg, kc, data, phsp, params, _p = pwa_small
    resolution_size = 12

    kv, k5 = _make(KV5, kc, resolution_size=resolution_size), \
        _make(KV5S0, kc, resolution_size=resolution_size)
    hv, h5 = kv.load_data(data), k5.load_data(data)
    # phsp count is intentionally not a multiple of resolution_size here
    with pytest.warns(RuntimeWarning, match="not a multiple"):
        hp5 = k5.load_data(phsp)
    hpv = kv.load_data(phsp)
    try:
        # phase-space path: linear Σ w·P, same as the generic v5
        Qn5, gn5, _ = k5.compute(params, hp5, norm=None)
        Qnv, gnv, _ = kv.compute(params, hpv, norm=None)
        # the s0 BW form (mult = m0*g0 precomputed) rounds differently
        # from the generic m0*g_interp path — same model, <=1 ulp
        assert Qn5 == pytest.approx(Qnv, rel=1e-11)
        assert np.max(np.abs(gn5["ck"] - gnv["ck"])) < 1e-8

        # group NLL: same likelihood as the generic v5 (same m0/g0 params)
        norm = float(Qnv)
        Q, _g, P = k5.compute(params, h5, norm=norm)
        Qv, _gv, _Pv = kv.compute(params, hv, norm=norm)
        assert Q == pytest.approx(Qv, rel=1e-10)
        ref = _numpy_group_nll(P, data["weight"], data["bkg"], norm,
                               resolution_size)
        assert Q == pytest.approx(ref, abs=1e-6)
    finally:
        h5.free(); hp5.free(); hv.free(); hpv.free()
        kv.free(); k5.free()


def test_v5_s0_group_grads_match_finite_difference(pwa_small):
    from tabpwa.cuda._v5_pwa_s0 import CUDAKernelV5PWAS0 as KV5S0
    _cfg, kc, data, phsp, params, _p = pwa_small
    resolution_size = 15
    ne = data["mass"].shape[0]
    assert ne % resolution_size == 0

    k5 = _make(KV5S0, kc, resolution_size=resolution_size)
    h5 = k5.load_data(data)
    with pytest.warns(RuntimeWarning, match="not a multiple"):
        hp = k5.load_data(phsp)          # phsp is not a multiple of rsize
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

        # one m0 (real parameter gradient returned directly — the s0/mult
        # chain to m0/g0 happens in the wrapper)
        m = params["m0"].copy()
        m[2] += eps
        num = (nll(dict(params, m0=m), norm) - nll(params, norm)) / eps
        assert num == pytest.approx(g["m0"][2], rel=1e-4, abs=1e-2)

        # one g0
        gg = params["g0"].copy()
        gg[1] += eps
        num = (nll(dict(params, g0=gg), norm) - nll(params, norm)) / eps
        assert num == pytest.approx(g["g0"][1], rel=1e-4, abs=1e-2)
    finally:
        h5.free(); hp.free()
        k5.free()
