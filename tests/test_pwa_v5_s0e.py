"""cuda_v5_pwa_s0e — fused tile kernel + grouped log-sum NLL.

The fused single-kernel tile of ``cuda_v4_pwa_s0e`` (one block owns whole
resolution groups; BW / 1/BW / common / common⁄BW / A_p / dQ⁄dA_p stay in
shared memory; one launch, no batching) with the cuda_v5 grouped log-sum
data NLL:

    q_e = w_e·(P_e/norm + bkg_e),  S_g = Σ_{e∈g} q_e
    Q = -Σ_g log(S_g),  dQ/dP_e = -(w_e/norm)/S_g
    dQ/dnorm = Σ_g (1/S_g)·(Σ_{e∈g} w_e·P_e)/norm²

Gates:
1. ``resolution_size=1`` with UNIT weights reproduces ``cuda_v4_pwa_s0e``
   (Q / P / ck / m0 / g0 / dnorm), both norm modes.
2. ``resolution_size ∈ {1, 2, 8, 100}`` matches ``cuda_v5_pwa_s0c`` and
   ``cuda_v5_pwa_s0d`` at the SAME group size (they share the NLL), for
   event counts both divisible and NOT divisible by the group size — the
   latter is the partial-last-group path; 100 > S0E_TE also exercises the
   one-block-per-group / shared-accumulator path.
3. The NLL matches an independent numpy group reference computed from the
   returned per-event P.
4. Finite-difference gradients through ``Fitter.get_nll`` (kernel + norm
   chain + dNLL/dnorm) at ``resolution_size ∈ {2, 8}``.
5. ``n_res > 1`` is rejected like the sibling one-BW kernels.
"""

import os
import sys
import warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

from test_cpu_pwa_s0c import _kernel_config, _synth, _params

from tabpwa import Fitter
from tabpwa.backends import create_backend, backends_for_model
from tabpwa.cuda._v4_pwa_s0e import CUDAKernelV4PWAS0E
from tabpwa.cuda._v5_pwa_s0c import CUDAKernelV5PWAS0C
from tabpwa.cuda._v5_pwa_s0d import CUDAKernelV5PWAS0D
from tabpwa.cuda._v5_pwa_s0e import CUDAKernelV5PWAS0E

CFG = "tests/config_pwa.yml"
NEV = 400
NH = 1e-6


def _unit(data):
    """Unit per-event weights (the v4/v5-with-rsize-1 anchor)."""
    return {**data, "weight": np.ones(len(data["weight"]))}


def _rel(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _numpy_group_nll(P, w, bkg, norm, rs):
    """Independent reference: Q = -Σ_g log Σ_{e∈g} w_e·(P_e/norm + bkg_e)."""
    ng = len(P) // rs
    gp = (w * P / norm + w * bkg).reshape(ng, rs).sum(axis=1)
    return -float(np.sum(np.log(gp)))


def test_registration():
    assert "cuda_v5_pwa_s0e" in backends_for_model("pwa")
    b = create_backend("cuda_v5_pwa_s0e", _kernel_config(), model="pwa",
                       resolution_size=8)
    assert b.kernel.resolution_size == 8
    assert b.dnorm_on_gpu is True
    b.free()


@pytest.mark.parametrize("norm", [None, 1.3])
def test_rsize1_unit_weights_matches_v4_s0e(norm):
    """rsize=1 + unit weights == the per-event v4 fused kernel."""
    kc = _kernel_config()
    data = _unit(_synth(kc, NEV, seed=5))

    k4 = CUDAKernelV4PWAS0E(kc)
    k5 = CUDAKernelV5PWAS0E(kc, resolution_size=1)
    params = _params(k5, seed=7, scale=0.5)
    Q4, g4, P4 = k4.compute(params, k4.load_data(data), norm=norm)
    Q5, g5, P5 = k5.compute(params, k5.load_data(data), norm=norm)

    assert _rel(Q5, Q4) < 1e-9
    assert _rel(P5, P4) < 1e-9
    for key in ("ck", "m0", "g0"):
        assert _rel(g5[key], g4[key]) < 1e-9, key
    if norm is not None:
        assert _rel(k5._last_dnorm, k4._last_dnorm) < 1e-9
    k4.free()
    k5.free()


@pytest.mark.parametrize("ne", [NEV, 417])          # 417: partial last group
@pytest.mark.parametrize("rs", [1, 2, 3, 7, 8, 100])   # 100 > S0E_TE; 3/7: rs·(TE/rs) < TE
@pytest.mark.parametrize("norm", [None, 1.3])
def test_matches_v5_family_at_same_rsize(rs, ne, norm):
    """Same grouped NLL as cuda_v5_pwa_s0c / _s0d at the same rsize."""
    kc = _kernel_config()
    data = _synth(kc, ne, seed=11)
    k = CUDAKernelV5PWAS0E(kc, resolution_size=rs)
    params = _params(k, seed=3, scale=0.4)

    if ne % rs:
        # partial last group: the v5 family's load-time warning
        with pytest.warns(RuntimeWarning):
            h = k.load_data(data)
    else:
        h = k.load_data(data)
    Q, g, P = k.compute(params, h, norm=norm)
    assert P.shape == (ne,)
    for key in ("ck", "m0", "g0"):
        assert np.all(np.isfinite(np.asarray(g[key]))), key

    for name, K in (("s0c", CUDAKernelV5PWAS0C), ("s0d", CUDAKernelV5PWAS0D)):
        ref = K(kc, resolution_size=rs)
        with warnings.catch_warnings():
            # the reference kernels warn about the same partial group
            warnings.simplefilter("ignore", RuntimeWarning)
            Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=norm)
        assert _rel(Q, Qr) < 1e-6, (name, "Q", _rel(Q, Qr))
        assert _rel(P, Pr) < 1e-6, (name, "P", _rel(P, Pr))
        for key in ("ck", "m0", "g0"):
            assert _rel(g[key], gr[key]) < 1e-6, (name, key)
        if norm is not None:
            assert _rel(k._last_dnorm, ref._last_dnorm) < 1e-6, name
        ref.free()
    k.free()


def test_nll_matches_numpy_group_reference():
    """Q equals the independent group sum over the returned per-event P."""
    kc = _kernel_config()
    data = _synth(kc, 256, seed=13)
    rs, norm = 8, 1.3
    k = CUDAKernelV5PWAS0E(kc, resolution_size=rs)
    params = _params(k, seed=9, scale=0.5)
    Q, _g, P = k.compute(params, k.load_data(data), norm=norm)
    ref = _numpy_group_nll(P, np.asarray(data["weight"]),
                           np.asarray(data["bkg"]), norm, rs)
    assert abs(Q - ref) <= 1e-10 * abs(ref)
    # and dNLL/dnorm against a finite difference of the same reference
    h = 1e-6
    Qp, _gp, Pp = k.compute(params, k.load_data(data), norm=norm + h)
    Qm, _gm, Pm = k.compute(params, k.load_data(data), norm=norm - h)
    ref_p = _numpy_group_nll(Pp, np.asarray(data["weight"]),
                             np.asarray(data["bkg"]), norm + h, rs)
    ref_m = _numpy_group_nll(Pm, np.asarray(data["weight"]),
                             np.asarray(data["bkg"]), norm - h, rs)
    fd = (ref_p - ref_m) / (2 * h)
    assert abs(k._last_dnorm - fd) <= 1e-6 * max(abs(fd), 1.0)
    k.free()


@pytest.mark.parametrize("rs", [2, 8])
def test_fitter_fd_gradient(rs):
    """End-to-end FD through Fitter.get_nll (kernel + norm chain + dnorm)."""
    kc = _kernel_config()
    f = Fitter(CFG, backend={"name": "cuda_v5_pwa_s0e", "resolution_size": rs})
    assert f.backend.kernel.resolution_size == rs
    f.set_phsp(_synth(kc, 128, seed=29))
    f.set_data(_synth(kc, 96, seed=23))

    x = f.initial_values(seed=5)
    nll0, gr0 = f.get_nll(x)
    assert np.isfinite(nll0)
    assert np.all(np.isfinite(gr0))
    assert np.isfinite(f.backend.kernel._last_dnorm)

    names = f.free_param_names()
    for i in range(0, len(x), max(1, len(x) // 6)):
        xp = x.copy(); xp[i] += NH
        xm = x.copy(); xm[i] -= NH
        fd = (f.get_nll(xp)[0] - f.get_nll(xm)[0]) / (2 * NH)
        assert abs(fd - gr0[i]) <= 1e-4 * max(abs(fd), 1.0), names[i]
    f.backend.free()


def test_rejects_multi_bw_chains():
    """n_res > 1 (two resonances in a chain) must fail fast (one-BW kernel)."""
    kc = dict(Fitter(CFG, backend="cuda_v4_pwa").kernel_config)
    kc["bw_order"] = np.repeat(np.asarray(kc["bw_order"]).reshape(-1), 2)
    with pytest.raises(ValueError, match="exactly ONE BW"):
        CUDAKernelV5PWAS0E(kc, resolution_size=1)

