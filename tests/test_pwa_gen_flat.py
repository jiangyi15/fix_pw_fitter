"""``cuda_v5_pwa_gen_flat`` — the ``pwa_gen`` fused tile kernel with a
**flat, unpadded, per-(wave, factor) contiguous** (CSR-style) denominator
basis, for any ``n_res >= 1`` Breit–Wigner factors per wave:

    bw_{w,r}(m) = Σ_{idx=off[f]}^{off[f+1]-1} coeff[idx]·T[g_idx[idx]](m)
    BW_w(m)     = Π_r bw_{w,r}(m),                     f = w·n_res + r
    ∂Q/∂coeff[idx] = Σ_e (∂Q/∂BW_w)·Π_{r'≠r} bw_{w,r'}·T[g_idx[idx]](m_e)

Everything else (fused tile, grouped log-sum NLL with the group total
weight, the context-owned combined basis table, the numerator folded into
the amp-cache fill) is the ``cuda_v5_pwa_gen_s0e`` machinery; only the
addressing changed, so the host contract (padded ``params["coeff"]`` in,
padded ``grads["coeff"]`` out) is unchanged.

Gates:
1. registration — ``pwa_gen`` scope only, and the ``pwa_gen`` support guard
   accepts the new backend;
2. **equal widths** (default-basis ``tests/config_pwa_gen.yml``,
   ``n_index ≡ 0``): ``n_basis_tot == n_fact·n_basis`` and the numbers are
   those of ``cuda_v5_pwa_gen_s0e``.  ``P`` is written per event and is
   bit-identical; ``Q`` / ``dnorm`` / the gradient blocks are reduced with
   float64 atomics in BOTH kernels, which is not even bit-reproducible
   between two calls of the *same* kernel (verified), so they are gated at
   the atomic floor (rel ≤ 1e-12, measured ≤ 5e-16);
3. **kmatrix2** (``tests/config_pwa_gen_km.yml``, 5 rows, numerator active):
   same ``Q`` / ``P`` / ``ck`` / ``coeff`` as ``cuda_v5_pwa_gen_s0e``;
4. **ragged** (a ``kmatrix2`` next to a plain BW — 5 rows vs 3): the point
   of the kernel.  ``n_basis_tot < n_fact·n_basis``, and the live
   coefficient columns match the padded kernel at the atomic floor.  The
   padding columns are structurally absent here (flat gradient exactly 0)
   while the padded kernel reports the derivative w.r.t. a coefficient that
   is fixed at zero — the caller ignores those columns, so only the live
   columns are compared;
5. finite-difference check of ``dcoeff`` on the ragged config, through the
   padded params contract (``rtol`` 1e-4, central differences, h=1e-6);
6. ``Fitter(cfg, backend="cuda_v5_pwa_gen_flat")`` smoke: finite NLL and
   gradients matching ``cuda_v5_pwa_gen_s0e``;
7. kernel independence: ``nm -D`` shows only this kernel's own entry points
   and no reference to another kernel's symbols.

Multi-BW gates (``n_res == 2``, ``tests/config_pwa_gen_mb.yml`` — a
``jpsi -> pipi(kmatrix2) + sig(BW) + eta`` chain, so every wave's
denominator is a PRODUCT of two factors and the factor widths are ragged):

8. shape / layout: ``n_res == 2``, ``n_fact == 2·n_wave``, per-factor width
   set ``{3, 5}``, ``n_basis_tot < n_fact·n_basis``, and ``cpu_pwa_gen``
   runs the same fixture;
9. ``resolution_size = 1``: ``P`` / ``Q`` / ``ck`` / ``coeff`` / ``dnorm``
   vs ``cpu_pwa_gen`` (the reference any-``n_res`` implementation) and vs
   ``numpy_pwa`` — the fp32 amplitude-cache floor;
10. ``resolution_size = 8`` with group-normalised weights (``w = 1/8``, so
    ``W_g == 1`` and the group objective equals the CPU one): ``Q`` /
    ``ck`` / ``coeff`` / ``dnorm`` vs ``cpu_v5_pwa`` at the same group size,
    ``P`` vs ``cpu_pwa_gen``, and ``Q`` vs an independent numpy group sum
    over the returned ``P``;
11. FD of ``dcoeff`` at ``n_res == 2`` over slots of BOTH factors;
12. factor independence: rescaling the two factors of a wave reciprocally
    (``c`` and ``1/c``) leaves the forward invariant (the product, not the
    sum) while each factor's own coefficient gradient rescales by ``1/c``
    and ``c`` respectively.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

from test_pwa_gen import CFG_BASIS, CFG_KM, _mixed_cfg, _synth

from tabpwa import Fitter
from tabpwa.backends import backends_for_model, create_backend
from tabpwa.cuda._v5_pwa_gen_flat import CUDAKernelV5PWAGenFlat
from tabpwa.cuda._v5_pwa_gen_s0e import CUDAKernelV5PWAGenS0E
from tabpwa.pwa_gen_support import check_backend_support

CFG_PWA = "tests/config_pwa.yml"
CFG_MB = "tests/config_pwa_gen_mb.yml"
NEV = 400
NH = 1e-6
ATOMIC = 1e-12          # fp64 global-atomic reduction floor


def _rel(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _rand_params(k, seed=7):
    r = np.random.RandomState(seed)
    N = k.n_wave_base
    return {"ck": 0.5 * (r.randn(N) + 1j * r.randn(N)),
            "m0": np.linspace(0.7, 1.5, k.n_m0_params),
            "g0": np.linspace(0.1, 0.5, k.n_g0_params)}


# ── gate 1: registration / scope ─────────────────────────────────────

def test_registration():
    assert "cuda_v5_pwa_gen_flat" in backends_for_model("pwa_gen")
    assert "cuda_v5_pwa_gen_flat" not in backends_for_model("pwa")
    # the pwa_gen capability guard accepts it (class name, as _CUDABackend
    # passes it) and a plain pwa_gen config needs no guard at all
    check_backend_support(Fitter(CFG_KM, backend="numpy_pwa").kernel_config,
                          "CUDABackendV5PWAGenFlat")
    check_backend_support(Fitter(CFG_KM, backend="numpy_pwa").kernel_config,
                          "cuda_v5_pwa_gen_flat")

    kc = Fitter(CFG_BASIS, backend="numpy_pwa").kernel_config
    b = create_backend("cuda_v5_pwa_gen_flat", kc, model="pwa_gen",
                       resolution_size=8)
    assert b.kernel.resolution_size == 8
    assert b.dnorm_on_gpu is True
    b.free()


def test_multi_bw_chain_shape_and_cpu_backend():
    """The two-resonance fixture: ``n_res == 2``, ragged per-factor widths.

    ``jpsi -> pipi(kmatrix2) + sig(BW) + eta`` gives every wave the product
    ``bw_{w,0}·bw_{w,1}``: the k-matrix factor carries 5 basis rows, the
    plain BW 3, so the flat layout drops 2 slots per second factor.
    """
    f = Fitter(CFG_MB, backend="numpy_pwa")
    kc = f.kernel_config
    n_wave = kc["matrix_angle"].shape[1]
    assert np.asarray(kc["bw_order"]).size // n_wave == 2      # n_res == 2
    assert kc["den_row"].shape[1] == 5
    assert np.any(np.asarray(kc["den_row"]) == -1)            # ragged

    kf = CUDAKernelV5PWAGenFlat(kc, resolution_size=1)
    assert kf.n_res == 2
    assert kf.n_fact == 2 * kf.n_wave
    assert kf.off.size == kf.n_fact + 1
    assert int(kf.off[0]) == 0 and int(kf.off[-1]) == kf.n_basis_tot
    assert set(np.unique(kf.n_basis_w)) == {3, 5}
    assert kf.n_basis_tot < kf.n_fact * kf.n_basis
    assert kf.wave_of_slot.size == kf.n_basis_tot
    assert np.all(kf.wave_of_slot
                  == np.repeat(np.arange(kf.n_fact) // kf.n_res,
                               kf.n_basis_w))

    # the flat block is the packed live columns, per (wave, factor)
    padded = np.arange(1.0, kf.n_fact * kf.n_basis + 1.0).reshape(
        kf.n_fact, kf.n_basis)
    flat = kf._pack(padded)
    for fct in range(kf.n_fact):
        b0, b1 = int(kf.off[fct]), int(kf.off[fct + 1])
        assert np.array_equal(flat[b0:b1], padded[fct, :b1 - b0])

    # the reference any-n_res implementation builds and runs this config
    cpu = create_backend("cpu_pwa_gen", kc, model="pwa_gen")
    x = f.initial_values(seed=1)
    params, _res = f.build_params(x)
    Q, _g, P = cpu.compute(params, cpu.load_data(_synth(kc, 64, 5)),
                           norm=1.3)
    assert np.isfinite(Q) and np.all(np.isfinite(P))
    cpu.free()
    kf.free()


# ── multi-BW gates: n_res == 2 against the CPU oracles ───────────────

def _mb():
    f = Fitter(CFG_MB, backend="numpy_pwa")
    kc = f.kernel_config
    x = f.initial_values(seed=1)
    params, _res = f.build_params(x)
    return f, kc, params


def _grouped(data, rs):
    """Group-normalised weights: ``W_g == 1``, so the CUDA group objective
    (``-Σ_g W_g·log(S_g/W_g)``) equals the CPU one (``-Σ_g log S_g``)."""
    d = dict(data)
    d["weight"] = np.full(len(data["weight"]), 1.0 / rs)
    return d


def _numpy_group_nll(P, w, bkg, norm, rs):
    ng = len(P) // rs
    gp = (w * P / norm + w * bkg).reshape(ng, rs).sum(axis=1)
    return -float(np.sum(np.log(gp)))


def test_multi_bw_matches_cpu_pwa_gen():
    """rs=1: ``cpu_pwa_gen`` (any n_res) and ``numpy_pwa`` are the oracles."""
    _f, kc, params = _mb()
    data = _synth(kc, NEV, 5)

    kf = CUDAKernelV5PWAGenFlat(kc, resolution_size=1)
    Q, g, P = kf.compute(params, kf.load_data(data), norm=1.3)

    for name in ("cpu_pwa_gen", "numpy_pwa"):
        ref = create_backend(name, kc, model="pwa_gen")
        Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
        assert _rel(P, Pr) < 1e-6, (name, "P")
        assert abs(Q - Qr) <= 1e-6 * abs(Qr), (name, "Q", Q, Qr)
        assert _rel(g["ck"], gr["ck"]) < 1e-5, (name, "ck")
        if name == "cpu_pwa_gen":
            # the per-(wave, factor) reference
            assert _rel(g["coeff"], gr["coeff"]) < 1e-5, (name, "coeff")
        else:
            # ``numpy_pwa`` attributes a shared BW's gradient to its first
            # copy only (the model's transform sums over the copies), so
            # compare the per-BW total, which is what the fitter consumes.
            gc = np.asarray(g["coeff"]).reshape(kf.n_fact, -1)
            grc = np.asarray(gr["coeff"]).reshape(kf.n_fact, -1)
            bow = np.asarray(kc["bw_order"]).reshape(-1)
            base = bow % len(np.unique(bow))
            for b in np.unique(base):
                m = base == b
                assert _rel(gc[m].sum(0), grc[m].sum(0)) < 1e-5, (name, b)
        assert abs(kf._last_dnorm - ref.kernel._last_dnorm) \
            <= 1e-6 * abs(ref.kernel._last_dnorm), (name, "dnorm")
        ref.free()
    kf.free()


def test_multi_bw_grouped_nll_at_rsize_8():
    """rs=8: the grouped log-sum, vs ``cpu_v5_pwa`` and a numpy group sum."""
    _f, kc, params = _mb()
    rs = 8
    data = _grouped(_synth(kc, 256, 5), rs)

    kf = CUDAKernelV5PWAGenFlat(kc, resolution_size=rs)
    Q, g, P = kf.compute(params, kf.load_data(data), norm=1.3)

    # independent group reference from the returned per-event P
    w = np.asarray(data["weight"])
    Qn = _numpy_group_nll(P, w, np.asarray(data["bkg"]), 1.3, rs)
    assert abs(Q - Qn) <= 1e-10 * abs(Qn), (Q, Qn)

    ref = create_backend("cpu_v5_pwa", kc, model="pwa_gen",
                         resolution_size=rs)
    Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
    assert abs(Q - Qr) <= 1e-6 * abs(Qr), (Q, Qr)
    assert _rel(P, Pr) < 1e-6
    assert _rel(g["ck"], gr["ck"]) < 1e-5
    assert _rel(g["coeff"], gr["coeff"]) < 1e-5
    assert abs(kf._last_dnorm - ref.kernel._last_dnorm) \
        <= 1e-6 * abs(ref.kernel._last_dnorm)

    # P is objective-independent: also the per-event cpu_pwa_gen value
    cpu = create_backend("cpu_pwa_gen", kc, model="pwa_gen")
    _Qc, _gc, Pc = cpu.compute(params, cpu.load_data(data), norm=1.3)
    assert _rel(P, Pc) < 1e-6
    cpu.free()
    ref.free()
    kf.free()


def test_fd_dcoeff_multi_bw():
    """FD of ``dcoeff`` at ``n_res == 2``, over slots of BOTH factors."""
    _f, kc, params = _mb()
    data = _synth(kc, 80, 5)

    kf = CUDAKernelV5PWAGenFlat(kc, resolution_size=1)
    h = kf.load_data(data)
    Q0, gf, _P = kf.compute(params, h, norm=1.3)
    assert np.isfinite(Q0)
    coeff = kf._pack(params["coeff"])
    gflat = kf._pack(gf["coeff"])
    nw = kf.n_wave

    def _Q(cflat):
        p = dict(params)
        p["coeff"] = kf._unpack(cflat)
        return kf.compute(p, h, norm=1.3, return_p=False)[0]

    # a few slots of the first factor and of the second, per wave
    idx = []
    for wv in range(nw):
        for r in range(kf.n_res):
            b0 = int(kf.off[wv * kf.n_res + r])
            idx.extend([b0, b0 + 1])
    for i in idx:
        cp = coeff.copy()
        cp[i] += NH
        cm = coeff.copy()
        cm[i] -= NH
        fd = (_Q(cp) - _Q(cm)) / (2 * NH)
        assert np.isclose(gflat[i], fd, rtol=1e-4, atol=1e-6), (
            f"flat slot {i}: {gflat[i]} vs FD {fd}")
    h.free()
    kf.free()


def test_multi_bw_factor_product_not_sum():
    """Reciprocal rescaling of a wave's two factors: the forward is
    invariant (the denominators are MULTIPLIED), while each factor's own
    coefficient gradient rescales by ``1/c`` (resp. ``c``).

    ``coeff -> coeff/c`` gives ``bw/c``, so ``∂Q/∂coeff' = c·∂Q/∂coeff``;
    with ``c`` on factor 0 and ``1/c`` on factor 1 the product
    ``BW = (c·bw_0)·(bw_1/c)`` is unchanged, which no sum could satisfy.
    """
    _f, kc, params = _mb()
    data = _synth(kc, 120, 5)
    kf = CUDAKernelV5PWAGenFlat(kc, resolution_size=1)
    h = kf.load_data(data)

    c = 2.5
    scale = np.ones(kf.n_fact)
    for wv in range(kf.n_wave):
        scale[wv * kf.n_res] = 1.0 / c          # factor 0: coeff *= 1/c
        scale[wv * kf.n_res + 1] = c            # factor 1: coeff *= c
    p2 = dict(params)
    p2["coeff"] = params["coeff"] * scale[:, None]

    Q1, g1, P1 = kf.compute(params, h, norm=1.3)
    Q2, g2, P2 = kf.compute(p2, h, norm=1.3)
    # the product is invariant up to the rounding of c·bw_0 resp. bw_1/c
    assert np.allclose(P1, P2, rtol=1e-12, atol=0.0)
    assert abs(Q1 - Q2) <= 1e-12 * abs(Q1)

    # coeff' = coeff·s  =>  ∂Q/∂coeff' = ∂Q/∂coeff / s  (per factor):
    # factor 0 (s = 1/c) grows by c, factor 1 (s = c) shrinks by 1/c.
    expect = kf._pack(g1["coeff"]) / np.repeat(scale, kf.n_basis_w)
    got = kf._pack(g2["coeff"])
    assert _rel(got, expect) < 1e-10
    h.free()
    kf.free()


def test_fitter_multi_bw_fd():
    """End-to-end FD at ``n_res == 2`` (kernel + norm chain + dnorm + the
    basis transform), through the same ``Fitter`` the fitter uses."""
    f = Fitter(CFG_MB, backend="cuda_v5_pwa_gen_flat")
    kc = f.kernel_config
    f.set_phsp(_synth(kc, 200, 1))
    f.set_data(_synth(kc, 120, 2))
    x = f.initial_values(seed=1)
    nll, grad = f.get_nll(x)
    assert np.isfinite(nll) and np.all(np.isfinite(grad))

    names = f.free_param_names()
    h = 1e-6
    for i in range(0, len(x), max(1, len(x) // 6)):
        xp = x.copy()
        xp[i] += h
        xm = x.copy()
        xm[i] -= h
        fd = (f.get_nll(xp)[0] - f.get_nll(xm)[0]) / (2 * h)
        assert abs(fd - grad[i]) <= 1e-4 * max(abs(fd), 1.0), names[i]
    f.backend.free()


# ── gate 2: equal widths == the padded pwa_gen kernel ────────────────

@pytest.mark.parametrize("rs", [1, 8])
def test_equal_width_anchor(rs):
    kc = Fitter(CFG_BASIS, backend="numpy_pwa").kernel_config
    assert "den_row" not in kc and "n_table" not in kc      # default basis
    data = _synth(kc, NEV, 5)

    kf = CUDAKernelV5PWAGenFlat(kc, resolution_size=rs)
    kg = CUDAKernelV5PWAGenS0E(kc, resolution_size=rs)
    # every wave carries the same number of slots: nothing to strip
    assert kf.n_basis_tot == kf.n_wave * kf.n_basis
    assert np.all(kf.n_basis_w == kf.n_basis)

    params = _rand_params(kf)
    Qf, gf, Pf = kf.compute(params, kf.load_data(data), norm=1.3)
    Qg, gg, Pg = kg.compute(params, kg.load_data(data), norm=1.3)

    # P is per-event and must be bit-identical (same arithmetic everywhere)
    assert np.array_equal(Pf, Pg)
    # Q / dnorm / gradients go through float64 atomics in both kernels
    assert _rel(Qf, Qg) < ATOMIC
    assert _rel(kf._last_dnorm, kg._last_dnorm) < ATOMIC
    for key in ("ck", "m0", "g0"):
        assert _rel(gf[key], gg[key]) < ATOMIC, key


# ── gate 3: kmatrix2 (5 rows, numerator active) ──────────────────────

@pytest.mark.parametrize("rs", [1, 8])
def test_kmatrix2_anchor(rs):
    f = Fitter(CFG_KM, backend="numpy_pwa")
    kc = f.kernel_config
    data = _synth(kc, NEV, 5)
    x = f.initial_values(seed=1)
    params, _resolved = f.build_params(x)

    kf = CUDAKernelV5PWAGenFlat(kc, resolution_size=rs)
    kg = CUDAKernelV5PWAGenS0E(kc, resolution_size=rs)
    # 1, −m² and the three kmatrix2 rows — rectangular, so no padding
    assert kf.n_basis == 5
    assert kf.n_basis_tot == kf.n_wave * kf.n_basis

    Qf, gf, Pf = kf.compute(params, kf.load_data(data), norm=1.3)
    Qg, gg, Pg = kg.compute(params, kg.load_data(data), norm=1.3)

    assert np.array_equal(Pf, Pg)
    assert _rel(Qf, Qg) < ATOMIC
    assert _rel(kf._last_dnorm, kg._last_dnorm) < ATOMIC
    assert _rel(gf["ck"], gg["ck"]) < ATOMIC
    assert gf["coeff"].shape == params["coeff"].shape
    assert _rel(gf["coeff"], gg["coeff"]) < ATOMIC


# ── gate 4: ragged basis widths (the point of the kernel) ────────────

def _ragged():
    """``kmatrix2`` (5 rows) next to a plain BW (3 rows) in one ``pwa_gen``."""
    f = Fitter(_mixed_cfg(), backend="numpy_pwa")
    kc = f.kernel_config
    assert "den_row" in kc and kc["den_row"].shape[1] == 5
    assert np.any(np.asarray(kc["den_row"]) == -1)          # padded encoding
    return f, kc


@pytest.mark.parametrize("rs", [1, 8])
def test_ragged_anchor(rs):
    f, kc = _ragged()
    data = _synth(kc, NEV, 5)
    x = f.initial_values(seed=1)
    params, _resolved = f.build_params(x)

    kf = CUDAKernelV5PWAGenFlat(kc, resolution_size=rs)
    kg = CUDAKernelV5PWAGenS0E(kc, resolution_size=rs)
    # the padded layout spends 5 slots per wave; the flat one only the
    # wave's own width (5 for the k-matrix, 3 for the plain BW)
    assert kf.n_basis_tot < kf.n_wave * kf.n_basis
    assert kf.n_basis_tot == int(np.sum(kf.n_basis_w))
    assert set(np.unique(kf.n_basis_w)) == {3, 5}

    Qf, gf, Pf = kf.compute(params, kf.load_data(data), norm=1.3)
    Qg, gg, Pg = kg.compute(params, kg.load_data(data), norm=1.3)

    assert np.array_equal(Pf, Pg)
    assert _rel(Qf, Qg) < ATOMIC
    assert _rel(kf._last_dnorm, kg._last_dnorm) < ATOMIC
    assert _rel(gf["ck"], gg["ck"]) < ATOMIC

    # the live columns are the wave's own range (flat -> padded index)
    live = kf._flat_src
    live_mask = np.zeros(kf.n_wave * kf.n_basis, bool)
    live_mask[live] = True
    gflat = gf["coeff"].reshape(-1)
    gpad = gg["coeff"].reshape(-1)
    assert _rel(gflat[live], gpad[live]) < ATOMIC
    # no phantom padding slot: the flat gradient is exactly zero there
    assert np.all(gflat[~live_mask] == 0.0)
    # ... while the padded kernel reports the derivative w.r.t. the
    # zero-fixed padding coefficients (ignored by PWAGenKernelParams)
    assert np.max(np.abs(gpad[~live_mask])) > 1.0


def test_ragged_flat_coefficients_are_unpadded():
    """The flat block is the packed live columns, in padded column order."""
    _f, kc = _ragged()
    kf = CUDAKernelV5PWAGenFlat(kc, resolution_size=1)
    padded = np.arange(1.0, kf.n_wave * kf.n_basis + 1.0).reshape(
        kf.n_wave, kf.n_basis)
    flat = kf._pack(padded)
    assert flat.size == kf.n_basis_tot
    # per wave, the first n_basis_w columns, contiguous
    for w in range(kf.n_wave):
        b0, b1 = int(kf.off[w]), int(kf.off[w + 1])
        assert np.array_equal(flat[b0:b1], padded[w, :b1 - b0])
    # round trip: the unpacked padding columns are exactly zero
    back = kf._unpack(flat)
    for w in range(kf.n_wave):
        wd = int(kf.n_basis_w[w])
        assert np.array_equal(back[w, :wd], padded[w, :wd])
        assert np.all(back[w, wd:] == 0.0)


# ── gate 5: finite differences of dcoeff on the ragged config ────────

def test_fd_dcoeff_ragged():
    f, kc = _ragged()
    data = _synth(kc, NEV, 5)
    x = f.initial_values(seed=1)
    params, _resolved = f.build_params(x)

    kf = CUDAKernelV5PWAGenFlat(kc, resolution_size=1)
    h = kf.load_data(data)
    Q0, gf, _P = kf.compute(params, h, norm=1.3)
    assert np.isfinite(Q0)
    coeff = kf._pack(params["coeff"])
    gflat = kf._pack(gf["coeff"])

    def _Q(cflat):
        p = dict(params)
        p["coeff"] = kf._unpack(cflat)
        return kf.compute(p, h, norm=1.3, return_p=False)[0]

    idx = np.random.RandomState(0).choice(kf.n_basis_tot, 8, replace=False)
    for i in idx:
        cp = coeff.copy()
        cp[i] += NH
        cm = coeff.copy()
        cm[i] -= NH
        fd = (_Q(cp) - _Q(cm)) / (2 * NH)
        assert np.isclose(gflat[i], fd, rtol=1e-4, atol=1e-6), (
            f"flat slot {i}: {gflat[i]} vs FD {fd}")
    h.free()


# ── gate 6: Fitter-level smoke ───────────────────────────────────────

def _loaded(cfg, backend, n_phsp=300, n_data=150):
    fit = Fitter(cfg, backend=backend)
    kc = fit.kernel_config
    fit.set_phsp(_synth(kc, n_phsp, 1))
    fit.set_data(_synth(kc, n_data, 2))
    return fit


def test_fitter_smoke_matches_gen_s0e():
    fg = _loaded(CFG_KM, "cuda_v5_pwa_gen_flat")
    fr = _loaded(CFG_KM, "cuda_v5_pwa_gen_s0e")
    x = fg.initial_values(seed=1)
    assert np.array_equal(x, fr.initial_values(seed=1))

    nll_f, grad_f = fg.get_nll(x)
    nll_r, grad_r = fr.get_nll(x)
    assert np.isfinite(nll_f)
    assert np.all(np.isfinite(grad_f))
    assert nll_f == pytest.approx(nll_r, rel=ATOMIC)
    assert _rel(grad_f, grad_r) < ATOMIC


def test_fitter_equal_width_config_matches_gen_s0e():
    """The equal-width anchor through the whole NLL chain (kernel + norm)."""
    fg = _loaded(CFG_BASIS, "cuda_v5_pwa_gen_flat")
    fr = _loaded(CFG_BASIS, "cuda_v5_pwa_gen_s0e")
    x = fg.initial_values(seed=2)
    nll_f, grad_f = fg.get_nll(x)
    nll_r, grad_r = fr.get_nll(x)
    assert np.isfinite(nll_f)
    assert nll_f == pytest.approx(nll_r, rel=ATOMIC)
    assert _rel(grad_f, grad_r) < ATOMIC


# ── gate 7: kernel independence ──────────────────────────────────────

def test_exports_and_no_foreign_symbols():
    from tabpwa.cuda.build import ensure
    lib = ensure("kernels_v5_pwa_gen_flat.cu",
                 "libcuda_kernels_v5_pwa_gen_flat.so")
    assert lib and os.path.exists(lib)

    defined = subprocess.run(["nm", "-D", "--defined-only", lib],
                             capture_output=True, text=True).stdout
    names = {ln.split()[-1] for ln in defined.splitlines() if ln.strip()}
    for want in ("cuda_create_context_v5_gen_flat", "cuda_load_data_v5",
                 "cuda_free_context_v5", "cuda_free_data_v5",
                 "cuda_gram_matrix_v5", "cuda_compute_v5_gen_flat",
                 "cuda_get_device_count", "cuda_get_device_name"):
        assert want in names, want
    # the padded sibling's create/compute must NOT be exported here
    assert "cuda_create_context_genv5s0e" not in names
    assert "cuda_compute_v5_coeff" not in names

    undef = subprocess.run(["nm", "-D", "--undefined-only", lib],
                           capture_output=True, text=True).stdout
    for ln in undef.splitlines():
        sym = ln.split()[-1]
        assert "cuda_create_context_" not in sym, sym
        assert "cuda_compute_v5" not in sym, sym
        assert "genv5s0e" not in sym and "pwa_flat" not in sym, sym
