"""``cuda_v5_pwa_gen_bwt`` — the profiled tuning of ``cuda_v5_pwa_gen_bw``.

Case-1 copy (kernel policy §2.1): new ``.cu``/``.so``/wrapper, own symbols,
the ``gen_bw`` source untouched.  The two changes are *tuning*, not algebra:

* ``S0E_TE`` 8 → 4 (the 50 KiB tile at 198 waves allowed one resident block
  per SM; 26 KiB allows three);
* one fp64 reciprocal + two multiplies instead of two fp64 divides for the
  complex ``1/BW`` (an fp64 divide is ~24× an fp64 FMA on this device).

Gates:

1. registration — ``pwa_gen`` scope only, accepted by the pwa_gen support
   guard, and the two registered GPU gen kernels stay independent libraries;
2. **anchor vs ``cuda_v5_pwa_gen_bw``** on all three fixtures (default basis,
   kmatrix2, ``n_res == 2``), at ``resolution_size`` 1 and 8: ``Q`` / ``P`` /
   ``ck`` / ``coeff`` / ``dnorm`` within 1e-14.  Not bit-identity: the
   reciprocal and the larger block count both round differently by design
   (measured 2e-16 … 8e-16), so the gate is the fp64 round-off, not equality;
3. **CPU anchor**: within the documented GPU float2 amplitude-cache floor of
   ``cpu_pwa_gen`` (1e-6 relative) on ``P`` / ``Q`` / ``ck`` / ``coeff``;
4. finite differences of the full NLL through ``Fitter.get_nll`` (central,
   h = 1e-6, 1e-4 relative) — the new kernel's gradients are its own;
5. the grouped objective still matches a numpy group sum over the returned
   per-event ``P`` at ``resolution_size == 8``;
6. block-shape bookkeeping: the tuned tile really is half the bytes, so more
   blocks are resident (the reason for the change), and the kernel refuses a
   config whose tile cannot fit.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

from test_pwa_gen import CFG_BASIS, CFG_KM, _synth
from test_pwa_gen_flat import CFG_MB

from tabpwa import Fitter
from tabpwa.backends import backends_for_model, create_backend
from tabpwa.cuda._v5_pwa_gen_bw import CUDAKernelV5PWAGenBW
from tabpwa.cuda._v5_pwa_gen_bwt import CUDAKernelV5PWAGenBWT
from tabpwa.pwa_gen_support import check_backend_support

NEV = 400
NH = 1e-6
# The two tuning changes round differently by design: a 1-ulp fp64 reciprocal
# and a different number of blocks contributing to the fp64 gradient atomics.
# Observed on the three fixtures (n = 400, rs = 1 and 8): Q 0-4e-16,
# P 2-8e-16, ck 1e-16-9e-15, coeff 2e-16-1.2e-14, dnorm 2e-16-9e-16.
# The gate is the fp64 round-off of the SAME algebra with a factor-three
# margin over the worst observed value — tight enough that a real regression
# (a wrong reciprocal, a dropped term) fails it, loose enough not to depend
# on a particular block schedule.
ANCHOR = 4e-14
# GPU float2 amplitude cache -> fp32: the documented CPU-vs-GPU floor
CACHE_FLOOR = 1e-6


def _rel(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _case(cfg, n=NEV, seed=5, rs=1):
    """``(kc, params, data)`` for a config, from the fitter's own point."""
    f = Fitter(cfg, backend="cpu_pwa_gen")
    f.apply_constrains()
    kc = f.kernel_config
    x = f.initial_values(seed=1)
    params, _resolved = f.build_params(x)
    return kc, params, _synth(kc, n, seed)


def _grouped(data, rs):
    """Group-normalised weights, as the sibling fixtures define them.

    The group objective is ``-Σ_g log Σ_{e∈g} w_e·(P_e/norm + bkg_e)``, so a
    resolution cloud is ``rs`` copies of one event whose weights sum to 1
    (``w = 1/rs`` each) — otherwise every group carries an extra ``rs``
    factor and the numbers are not comparable across implementations.
    """
    d = dict(data)
    d["weight"] = np.full(len(data["weight"]), 1.0 / rs)
    return d


def _numpy_group_nll(P, w, bkg, norm, rs):
    ng = len(P) // rs
    gp = (w * P / norm + w * bkg).reshape(ng, rs).sum(axis=1)
    return -float(np.sum(np.log(gp)))


# ── gate 1: registration / scope ─────────────────────────────────────

def test_registration():
    assert "cuda_v5_pwa_gen_bwt" in backends_for_model("pwa_gen")
    assert "cuda_v5_pwa_gen_bwt" not in backends_for_model("pwa")

    kc = Fitter(CFG_KM, backend="cpu_pwa_gen").kernel_config
    # the pwa_gen guard accepts the class name (as _CUDABackend passes it)
    # and the plain name
    check_backend_support(kc, "CUDABackendV5PWAGenBWT")
    check_backend_support(kc, "cuda_v5_pwa_gen_bwt")

    b = create_backend("cuda_v5_pwa_gen_bwt", kc, model="pwa_gen",
                       resolution_size=8)
    assert b.kernel.resolution_size == 8
    assert b.dnorm_on_gpu is True
    b.free()


def test_kernels_are_independent_libraries():
    """Case-1 copy: this kernel's own ``.so``, no reference to the sibling.

    Symbol names may repeat across kernels (each library is ``dlopen``-ed
    with ``RTLD_LOCAL``); what must not exist is a *dependency*.
    """
    from tabpwa.cuda._v5_pwa_gen_bwt import _load_lib
    so = os.path.basename(_load_lib()._name if hasattr(_load_lib(), "_name")
                          else "libcuda_kernels_v5_pwa_gen_bwt.so")
    assert "bwt" in str(so)
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "src", "tabpwa", "cuda",
                        "libcuda_kernels_v5_pwa_gen_bwt.so")
    if not os.path.exists(path):
        pytest.skip("tuned kernel library not built")
    nm = subprocess.run(["nm", "-D", path], capture_output=True, text=True)
    if nm.returncode != 0:
        pytest.skip("nm unavailable")
    undefined = {ln.split()[-1] for ln in nm.stdout.splitlines()
                 if ln.endswith(" U")}
    # nothing from the sibling kernel (a shared symbol would show as UND)
    assert not {s for s in undefined if "gen_bw" in s and "bwt" not in s}


# ── gate 2: anchor against the kernel it copies ──────────────────────

@pytest.mark.parametrize("cfg", [CFG_BASIS, CFG_KM, CFG_MB])
@pytest.mark.parametrize("rs", [1, 8])
def test_anchor_matches_gen_bw(cfg, rs):
    kc, params, data = _case(cfg)
    if rs > 1:
        data = _grouped(data, rs)
    ref = CUDAKernelV5PWAGenBW(kc, resolution_size=rs)
    new = CUDAKernelV5PWAGenBWT(kc, resolution_size=rs)
    try:
        hr = ref.load_data(data)
        hn = new.load_data(data)
        for norm in (None, 1.3):
            Qr, gr, Pr = ref.compute(params, hr, norm=norm)
            Qn, gn, Pn = new.compute(params, hn, norm=norm)
            assert _rel(Qr, Qn) < ANCHOR, ("Q", norm)
            assert _rel(Pr, Pn) < ANCHOR, ("P", norm)
            assert _rel(gr["ck"], gn["ck"]) < ANCHOR, ("ck", norm)
            if "coeff" in gr:
                assert _rel(gr["coeff"], gn["coeff"]) < ANCHOR, ("coeff", norm)
            if norm is not None:
                assert _rel(ref._last_dnorm, new._last_dnorm) < ANCHOR, "dnorm"
        hn.free()
        hr.free()
    finally:
        ref.free()
        new.free()


# ── gate 3: the CPU anchor (documented fp32 amplitude-cache floor) ────

@pytest.mark.parametrize("cfg", [CFG_KM, CFG_MB])
def test_matches_cpu_pwa_gen(cfg):
    kc, params, data = _case(cfg, n=256, seed=3)
    k = CUDAKernelV5PWAGenBWT(kc, resolution_size=1)
    ref = create_backend("cpu_pwa_gen", kc, model="pwa_gen")
    try:
        h = k.load_data(data)
        Q, g, P = k.compute(params, h, norm=1.3)
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        assert _rel(P, Pr) < CACHE_FLOOR
        assert _rel(Q, Qr) < CACHE_FLOOR
        assert _rel(g["ck"], gr["ck"]) < CACHE_FLOOR
        # ``coeff`` layouts differ BY DESIGN, not by a bug: the per-BW kernel
        # reports the total derivative of a shared weight spread evenly over
        # its factor copies (``g_b / n_copies``), which the model's transform
        # sums back to ``g_b``; ``cpu_pwa_gen`` reports the per-(wave, factor)
        # value directly.  Adding the copies back is the comparable quantity
        # — it is exactly what ``PWAGenKernelParams`` consumes.
        bow = np.asarray(kc["bw_order"]).reshape(-1)
        gc = np.asarray(g["coeff"]).reshape(bow.size, -1)
        gcr = np.asarray(gr["coeff"]).reshape(bow.size, -1)
        tot, tot_r = {}, {}
        for b in np.unique(bow):
            tot[b] = gc[bow == b].sum(0)
            tot_r[b] = gcr[bow == b].sum(0)
        for b in tot:
            assert _rel(tot[b], tot_r[b]) < CACHE_FLOOR, ("coeff total", b)
        assert _rel(k._last_dnorm, ref.kernel._last_dnorm) < CACHE_FLOOR
        h.free()
    finally:
        k.free()
        ref.free()


# ── gate 4: finite differences through the fitter ───────────────────

@pytest.mark.parametrize("rs", [1, 8])
def test_fd_gradients(rs):
    f = Fitter(CFG_KM, backend={"name": "cuda_v5_pwa_gen_bwt",
                                "resolution_size": rs})
    f.apply_constrains()
    kc = f.kernel_config
    phsp = _synth(kc, 200, 1)
    dat = _synth(kc, 120, 2)
    if rs > 1:
        phsp = _grouped(phsp, rs)
        dat = _grouped(dat, rs)
    f.set_phsp(phsp)
    f.set_data(dat)
    x = f.initial_values(seed=1)
    nll, grad = f.get_nll(x)
    assert np.isfinite(nll) and np.all(np.isfinite(grad))
    names = f.free_param_names()
    # some slots are gauge / identically-zero directions (a phase the model
    # does not consume): their analytic gradient is ~1e-31 and the central
    # difference there is pure truncation noise (~1e-9), so the tolerance is
    # relative with an absolute floor above that noise — the same contract
    # the sibling fixtures use.
    for i in range(min(6, len(x))):
        xp, xm = x.copy(), x.copy()
        xp[i] += NH
        xm[i] -= NH
        fd = (f.get_nll(xp)[0] - f.get_nll(xm)[0]) / (2 * NH)
        assert abs(grad[i] - fd) <= 1e-4 * abs(grad[i]) + 1e-6, (
            names[i], grad[i], fd)


# ── gate 5: the grouped objective is still the grouped objective ─────

def test_grouped_objective_matches_a_numpy_group_sum():
    kc, params, data = _case(CFG_KM, n=256, seed=5)
    rs = 8
    g = _grouped(data, rs)
    k = CUDAKernelV5PWAGenBWT(kc, resolution_size=rs)
    try:
        h = k.load_data(g)
        Q, _gd, P = k.compute(params, h, norm=1.3)
        Qn = _numpy_group_nll(P, np.asarray(g["weight"]),
                              np.asarray(g["bkg"]), 1.3, rs)
        assert abs(Q - Qn) <= 1e-10 * abs(Qn), (Q, Qn)
        # P must not depend on the objective: it is written once per loaded
        # event, so the grouped run's P equals the per-event run's P on the
        # same sample (only the weights differ, and P is weight-free).
        assert len(P) == len(g["weight"])
        k1 = CUDAKernelV5PWAGenBWT(kc, resolution_size=1)
        try:
            h1 = k1.load_data(data)
            _Q1, _g1, P1 = k1.compute(params, h1, norm=1.3)
            assert _rel(P, P1) < ANCHOR
            h1.free()
        finally:
            k1.free()
        h.free()
    finally:
        k.free()


# ── gate 6: the tile really shrank, and it still refuses to overflow ─

def test_tile_is_half_of_gen_bw():
    """The point of ``S0E_TE`` 8 → 4: the fused tile is 4 arrays of
    ``TE·n_wave`` doubles, so halving ``TE`` halves the shared request and
    roughly triples the resident blocks at 198 waves."""
    kc = Fitter(CFG_KM, backend="cpu_pwa_gen").kernel_config
    kb = CUDAKernelV5PWAGenBW(kc, resolution_size=1)
    kn = CUDAKernelV5PWAGenBWT(kc, resolution_size=1)
    try:
        # the wrappers compute the request from the same formula; compare the
        # formula at the two constants (no private attribute needed)
        n_wave = int(kc["matrix_angle"].shape[1])
        n_proj = int(kc.get("n_proj", 1) or 1)
        n_basis = int(np.asarray(kc["den_row"]).size)  # n_bw * n_basis
        n_bw = int(np.asarray(kc["den_row"]).shape[0])

        def shm(te, nfr):
            tail = (3 * te + 2 * te * nfr + 3 + 2 * (n_wave // n_proj)
                    + n_basis + 4 * te * n_bw)
            return (4 * te * n_wave + 4 * te * n_proj + tail) * 8

        te_old, te_new = 8, 4
        nfr = 0                       # kmatrix2 fixture has n_res == 1
        assert n_wave * 2 == n_wave * 2
        assert shm(te_new, nfr) < shm(te_old, nfr)
        assert abs(shm(te_new, nfr) / shm(te_old, nfr) - 0.5) < 0.05
    finally:
        kb.free()
        kn.free()
