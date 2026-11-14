"""``cuda_v5_pwa_gen_bw`` — the ``pwa_gen`` fused tile kernel with the
denominator basis block indexed **per unique base Breit–Wigner** (case-1 copy
of ``cuda_v5_pwa_gen_flat``):

    bw_b(m)   = Σ_{idx=off_bw[b]}^{off_bw[b+1]-1} coeff[idx]·T[g_idx[idx]](m)
    f         = w·n_res + r,     b = bw_of_factor[f]   (== bw_order[f])
    BW_w(m)   = Π_r bw_{b(w,r)}(m)
    ∂Q/∂coeff[idx] = Σ_{(w,r): b(w,r)=b} Σ_e (∂Q/∂BW_w)
                       · Π_{r'≠r} bw_{b(w,r')}(m_e) · T[g_idx[idx]](m_e)

Everything else (fused tile, grouped log-sum NLL with the group total weight,
the context-owned combined basis table, the numerator folded into the
amp-cache fill) is the ``cuda_v5_pwa_gen_flat`` / ``cuda_v5_pwa_gen_s0e``
machinery; only the addressing of the coefficient block changed, so the host
contract (padded ``params["coeff"]`` in, padded ``grads["coeff"]`` out) is
unchanged.

Gates:
1. registration — ``pwa_gen`` scope only, and the ``pwa_gen`` support guard
   accepts the new backend;
2. **cross-variant equality** with ``cuda_v5_pwa_gen_flat`` on the default
   basis (``tests/config_pwa_gen.yml``), ``kmatrix2``
   (``tests/config_pwa_gen_km.yml``) and the two-BW
   (``tests/config_pwa_gen_mb.yml``, ``n_res == 2``) fixtures at
   ``resolution_size ∈ {1, 8}``: ``P`` bit-identical, ``Q`` / ``dnorm`` /
   ``ck`` / the per-BW ``coeff`` total at the fp64-atomic floor (rel ≤ 1e-12,
   measured ≤ 7e-16);
3. block size — the per-BW block is ``n_bw·n_basis`` (equal-width) and
   strictly smaller than the flat variant's whenever ``n_wave > n_bw``
   (ratios reported for all three fixtures);
4. ragged + multi-BW correctness — ``n_res == 2`` against ``cpu_pwa_gen``
   (the reference any-``n_res`` implementation) and the ``rs = 8`` grouped
   log-sum against ``cpu_v5_pwa`` + an independent numpy group sum;
5. finite differences of ``dcoeff`` at ``n_res == 2`` (central, h=1e-6,
   rtol 1e-4), through the padded params contract;
6. factor independence — reciprocal rescaling of a wave's two factors leaves
   the forward invariant (the denominators are MULTIPLIED) while each BW's
   own coefficient gradient rescales reciprocally;
7. ``Fitter(cfg, backend="cuda_v5_pwa_gen_bw")`` smoke + NLL/gradient
   equality with ``cuda_v5_pwa_gen_flat`` (this pins the padded-gradient
   expansion: the per-BW derivative is spread over its factor copies as an
   equal share, so the unchanged host copy-sum recovers the total — a
   literal broadcast would be off by the copy count);
8. kernel independence: ``nm -D`` shows only this kernel's own entry points
   and no reference to another kernel's symbols.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

from test_pwa_gen import CFG_BASIS, CFG_KM, _synth

from tabpwa import Fitter
from tabpwa.backends import backends_for_model, create_backend
from tabpwa.cuda._v5_pwa_gen_bw import CUDAKernelV5PWAGenBW
from tabpwa.cuda._v5_pwa_gen_flat import CUDAKernelV5PWAGenFlat
from tabpwa.pwa_gen_support import check_backend_support

CFG_MB = "tests/config_pwa_gen_mb.yml"
NEV = 400
NH = 1e-6
ATOMIC = 1e-12          # fp64 global-atomic reduction floor


def _rel(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def _rand_params(k):
    r = np.random.RandomState(7)
    N = k.n_wave_base
    return {"ck": 0.5 * (r.randn(N) + 1j * r.randn(N)),
            "m0": np.linspace(0.7, 1.5, k.n_m0_params),
            "g0": np.linspace(0.1, 0.5, k.n_g0_params)}


def _params_and_data(cfg, n=NEV, seed=5):
    """``(kc, params, data)``: random m0/g0 params for the default basis, the
    fitter's own basis weights for a generic (``den_row``) config."""
    f = Fitter(cfg, backend="numpy_pwa")
    kc = f.kernel_config
    k0 = CUDAKernelV5PWAGenBW(kc, resolution_size=1)
    try:
        if "den_row" in kc:
            x = f.initial_values(seed=1)
            params, _resolved = f.build_params(x)
        else:
            params = _rand_params(k0)
    finally:
        k0.free()
    return kc, params, _synth(kc, n, seed)


def _copy_totals(g, n_fact, n_bw, bw_of_factor):
    """Per-BW totals of a padded ``(n_fact, -1)`` coefficient gradient."""
    gp = np.asarray(g).reshape(n_fact, -1)
    return np.array([gp[bw_of_factor == b].sum(axis=0) for b in range(n_bw)])


# ── gate 1: registration / scope ─────────────────────────────────────

def test_registration():
    assert "cuda_v5_pwa_gen_bw" in backends_for_model("pwa_gen")
    assert "cuda_v5_pwa_gen_bw" not in backends_for_model("pwa")
    kc = Fitter(CFG_KM, backend="numpy_pwa").kernel_config
    check_backend_support(kc, "CUDABackendV5PWAGenBW")
    check_backend_support(kc, "cuda_v5_pwa_gen_bw")

    b = create_backend("cuda_v5_pwa_gen_bw", kc, model="pwa_gen",
                       resolution_size=8)
    assert b.kernel.resolution_size == 8
    assert b.dnorm_on_gpu is True
    b.free()


# ── gate 2: cross-variant equality with cuda_v5_pwa_gen_flat ──────────

@pytest.mark.parametrize("cfg", [CFG_BASIS, CFG_KM, CFG_MB])
@pytest.mark.parametrize("rs", [1, 8, 16])
def test_matches_flat_variant(cfg, rs):
    """``rs = 16`` exercises the ``rs > S0E_TE`` group-accumulator path."""
    kc, params, data = _params_and_data(cfg)
    kb = CUDAKernelV5PWAGenBW(kc, resolution_size=rs)
    kf = CUDAKernelV5PWAGenFlat(kc, resolution_size=rs)

    Qb, gb, Pb = kb.compute(params, kb.load_data(data), norm=1.3)
    Qf, gf, Pf = kf.compute(params, kf.load_data(data), norm=1.3)

    # P is written per event and must be bit-identical (same arithmetic)
    assert np.array_equal(Pb, Pf)
    # Q / dnorm / gradients go through float64 atomics in both kernels
    assert _rel(Qb, Qf) < ATOMIC
    assert _rel(kb._last_dnorm, kf._last_dnorm) < ATOMIC
    assert _rel(gb["ck"], gf["ck"]) < ATOMIC

    if "coeff" in gb:
        # the block layouts differ (per BW vs per (wave, factor)), so the
        # comparable quantity is the per-BW total — exactly what the model's
        # unchanged copy-sum consumes.
        bow = kb.bw_of_factor
        assert gb["coeff"].shape == params["coeff"].shape      # padded
        assert _rel(_copy_totals(gb["coeff"], kb.n_fact, kb.n_bw, bow),
                    _copy_totals(gf["coeff"], kb.n_fact, kb.n_bw, bow)) \
            < ATOMIC
    else:
        assert _rel(gb["m0"], gf["m0"]) < ATOMIC
        assert _rel(gb["g0"], gf["g0"]) < ATOMIC

    kb.free()
    kf.free()


# ── gate 3: block size (the point of the kernel) ─────────────────────

def test_block_size():
    """``n_basis_tot == n_bw·n_basis`` (equal-width) and strictly smaller
    than the flat sibling's block whenever ``n_wave > n_bw``."""
    ratios = {}
    for cfg, equal_width in ((CFG_BASIS, True), (CFG_KM, True),
                             (CFG_MB, False)):
        kc = Fitter(cfg, backend="numpy_pwa").kernel_config
        kb = CUDAKernelV5PWAGenBW(kc, resolution_size=1)
        kf = CUDAKernelV5PWAGenFlat(kc, resolution_size=1)

        n_wave = int(kc["matrix_angle"].shape[1])
        assert kb.n_wave == n_wave
        assert kb.n_bw == len(np.unique(np.asarray(kc["bw_order"])))
        assert kb.n_bw < kb.n_wave                       # weights are shared
        assert kb.off_bw.size == kb.n_bw + 1
        assert int(kb.off_bw[0]) == 0
        assert int(kb.off_bw[-1]) == kb.n_basis_tot
        assert kb.g_idx.size == kb.n_basis_tot
        assert kb.flist.size == kb.n_fact
        assert np.array_equal(np.sort(kb.flist), np.arange(kb.n_fact))
        assert np.array_equal(kb.bw_of_factor,
                              np.asarray(kc["bw_order"]).astype(np.int32))

        # per-BW block: no duplicated weights anywhere
        assert kb.n_basis_tot == int(np.sum(kb.n_basis_w))
        assert kb.n_basis_w.size == kb.n_bw
        if equal_width:
            assert np.all(kb.n_basis_w == kb.n_basis)
            assert kb.n_basis_tot == kb.n_bw * kb.n_basis
        else:
            assert kb.n_basis_tot <= kb.n_bw * kb.n_basis

        # strictly smaller than the flat layout (it repeats per wave/factor)
        assert kf.n_basis_tot == int(np.sum(kf.n_basis_w))
        assert kb.n_basis_tot < kf.n_basis_tot
        ratios[os.path.basename(cfg)] = (kb.n_basis_tot, kf.n_basis_tot)
        kb.free()
        kf.free()

    assert set(ratios) == {"config_pwa_gen.yml", "config_pwa_gen_km.yml",
                          "config_pwa_gen_mb.yml"}
    print({k: f"{v[0]}/{v[1]}" for k, v in ratios.items()})


# ── gate 4: ragged + multi-BW correctness (CPU oracles) ───────────────

def _mb():
    return _params_and_data(CFG_MB)


def _grouped(data, rs):
    d = dict(data)
    d["weight"] = np.full(len(data["weight"]), 1.0 / rs)
    return d


def _numpy_group_nll(P, w, bkg, norm, rs):
    ng = len(P) // rs
    gp = (w * P / norm + w * bkg).reshape(ng, rs).sum(axis=1)
    return -float(np.sum(np.log(gp)))


def test_multi_bw_matches_cpu_pwa_gen():
    """``n_res == 2`` (two shared BWs, ragged 5/3 widths) vs the CPU
    reference: ``P`` 1e-6, ``Q`` 1e-6, per-BW ``coeff`` 1e-5."""
    kc, params, data = _mb()
    kb = CUDAKernelV5PWAGenBW(kc, resolution_size=1)
    assert kb.n_res == 2
    assert kb.n_bw == 2
    assert set(np.unique(kb.n_basis_w)) == {3, 5}
    Q, g, P = kb.compute(params, kb.load_data(data), norm=1.3)

    ref = create_backend("cpu_pwa_gen", kc, model="pwa_gen")
    Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
    assert _rel(P, Pr) < 1e-6
    assert abs(Q - Qr) <= 1e-6 * abs(Qr)
    assert _rel(g["ck"], gr["ck"]) < 1e-5
    bow = np.asarray(kc["bw_order"]).reshape(-1)
    assert _rel(_copy_totals(g["coeff"], kb.n_fact, kb.n_bw, bow),
                _copy_totals(gr["coeff"], kb.n_fact, kb.n_bw, bow)) < 1e-5
    assert abs(kb._last_dnorm - ref.kernel._last_dnorm) \
        <= 1e-6 * abs(ref.kernel._last_dnorm)
    ref.free()
    kb.free()


def test_multi_bw_grouped_nll_at_rsize_8():
    """``rs = 8``: the grouped log-sum vs ``cpu_v5_pwa`` and a numpy sum."""
    kc, params, _d = _mb()
    rs = 8
    data = _grouped(_synth(kc, 256, 5), rs)

    kb = CUDAKernelV5PWAGenBW(kc, resolution_size=rs)
    Q, g, P = kb.compute(params, kb.load_data(data), norm=1.3)

    w = np.asarray(data["weight"])
    Qn = _numpy_group_nll(P, w, np.asarray(data["bkg"]), 1.3, rs)
    assert abs(Q - Qn) <= 1e-10 * abs(Qn)

    ref = create_backend("cpu_v5_pwa", kc, model="pwa_gen", resolution_size=rs)
    Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
    assert abs(Q - Qr) <= 1e-6 * abs(Qr)
    assert _rel(P, Pr) < 1e-6
    assert _rel(g["ck"], gr["ck"]) < 1e-5
    bow = np.asarray(kc["bw_order"]).reshape(-1)
    assert _rel(_copy_totals(g["coeff"], kb.n_fact, kb.n_bw, bow),
                _copy_totals(gr["coeff"], kb.n_fact, kb.n_bw, bow)) < 1e-5
    assert abs(kb._last_dnorm - ref.kernel._last_dnorm) \
        <= 1e-6 * abs(ref.kernel._last_dnorm)
    ref.free()
    kb.free()


# ── gate 5: finite differences of dcoeff at n_res == 2 ───────────────

def test_fd_dcoeff_multi_bw():
    """Central differences over slots of BOTH factors.

    The padded ``(n_fact, n_basis)`` block holds the same weights in every
    copy of a BW, so perturbing the representative copy of a BW is the same
    perturbation as perturbing the shared weight; the matching gradient is
    the BW's total (the copy sum).
    """
    kc, params, data = _mb()
    kb = CUDAKernelV5PWAGenBW(kc, resolution_size=1)
    h = kb.load_data(data)
    Q0, gf, _P = kb.compute(params, h, norm=1.3)
    assert np.isfinite(Q0)
    coeff = np.asarray(params["coeff"], np.float64).copy()
    gbw = _copy_totals(gf["coeff"], kb.n_fact, kb.n_bw,
                       np.asarray(kc["bw_order"]).reshape(-1))

    def _Q(cpad):
        p = dict(params)
        p["coeff"] = cpad
        return kb.compute(p, h, norm=1.3, return_p=False)[0]

    idx = []
    for b in range(kb.n_bw):
        f = int(kb.flist[int(kb.foff[b])])          # representative copy
        for j in range(int(kb.n_basis_w[b])):
            idx.append((f, j))
    for f, j in idx:
        cp = coeff.copy()
        cp[f, j] += NH
        cm = coeff.copy()
        cm[f, j] -= NH
        fd = (_Q(cp) - _Q(cm)) / (2 * NH)
        b = int(kb.bw_of_factor[f])
        assert np.isclose(gbw[b][j], fd, rtol=1e-4, atol=1e-6), (
            f"BW {b} slot {j}: {gbw[b][j]} vs FD {fd}")
    h.free()
    kb.free()


# ── gate 6: factor independence (product, not sum) ───────────────────

def test_multi_bw_factor_product_not_sum():
    """Reciprocal rescaling of the two BWs: the forward is invariant (the
    denominators are MULTIPLIED), each BW's gradient rescales reciprocally.

    ``coeff -> coeff/c`` on BW 0 and ``coeff·c`` on BW 1 gives
    ``∂Q/∂coeff' = c·∂Q/∂coeff`` resp. ``∂Q/∂coeff/c`` while
    ``BW = (bw_0/c)·(bw_1·c)`` is unchanged — impossible for a sum.
    """
    kc, params, data = _mb()
    kb = CUDAKernelV5PWAGenBW(kc, resolution_size=1)
    h = kb.load_data(data)
    assert kb.n_bw == 2

    c = 2.5
    scale = np.ones(kb.n_bw)
    scale[0] = 1.0 / c          # BW 0: coeff *= 1/c
    scale[1] = c                # BW 1: coeff *= c
    p2 = dict(params)
    p2["coeff"] = np.asarray(params["coeff"]) * scale[kb.bw_of_factor][:, None]

    Q1, g1, P1 = kb.compute(params, h, norm=1.3)
    Q2, g2, P2 = kb.compute(p2, h, norm=1.3)
    assert np.allclose(P1, P2, rtol=1e-12, atol=0.0)
    assert abs(Q1 - Q2) <= 1e-12 * abs(Q1)

    bow = kb.bw_of_factor
    t1 = _copy_totals(g1["coeff"], kb.n_fact, kb.n_bw, bow)
    t2 = _copy_totals(g2["coeff"], kb.n_fact, kb.n_bw, bow)
    assert _rel(t2, t1 / scale[:, None]) < 1e-10
    h.free()
    kb.free()


# ── gate 7: Fitter-level smoke + equality with the flat sibling ───────

def _loaded(cfg, backend, n_phsp=300, n_data=150):
    fit = Fitter(cfg, backend=backend)
    kc = fit.kernel_config
    fit.set_phsp(_synth(kc, n_phsp, 1))
    fit.set_data(_synth(kc, n_data, 2))
    return fit


@pytest.mark.parametrize("cfg", [CFG_BASIS, CFG_KM, CFG_MB])
def test_fitter_smoke_matches_flat(cfg):
    fb = _loaded(cfg, "cuda_v5_pwa_gen_bw")
    ff = _loaded(cfg, "cuda_v5_pwa_gen_flat")
    x = fb.initial_values(seed=1)
    assert np.array_equal(x, ff.initial_values(seed=1))

    nll_b, grad_b = fb.get_nll(x)
    nll_f, grad_f = ff.get_nll(x)
    assert np.isfinite(nll_b)
    assert np.all(np.isfinite(grad_b))
    assert nll_b == pytest.approx(nll_f, rel=ATOMIC)
    # the whole chain, including the padded-coefficient copy-sum
    assert _rel(grad_b, grad_f) < 1e-12, (cfg, _rel(grad_b, grad_f))
    fb.backend.free()
    ff.backend.free()


def test_fitter_fd_multi_bw():
    """End-to-end FD at ``n_res == 2`` (kernel + norm chain + basis
    transform) through the same ``Fitter`` the fitter uses."""
    f = Fitter(CFG_MB, backend="cuda_v5_pwa_gen_bw")
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


# ── gate 8: kernel independence ──────────────────────────────────────

def test_gram_path_matches_flat():
    """The s0 preset's Gram matrix (same per-BW basis, same amplitudes)."""
    kc, _params, _data = _params_and_data(CFG_BASIS, n=64, seed=3)
    kb = CUDAKernelV5PWAGenBW(kc, resolution_size=1)
    kf = CUDAKernelV5PWAGenFlat(kc, resolution_size=1)
    ph = kb.load_data(_synth(kc, 64, 3))
    m0 = np.linspace(0.7, 1.5, kb.n_m0_params)
    g0 = np.linspace(0.1, 0.5, kb.n_g0_params)
    D = kb.compute_gram(ph, m0, g0)
    Df = kf.compute_gram(kf.load_data(_synth(kc, 64, 3)), m0, g0)
    assert np.all(np.isfinite(D))
    assert np.allclose(D, D.conj().T)
    assert _rel(D, Df) < ATOMIC
    # the generic basis owns no m0/g0, so the Gram path raises
    kc_km, _p, _d = _params_and_data(CFG_KM, n=16)
    kg = CUDAKernelV5PWAGenBW(kc_km, resolution_size=1)
    with pytest.raises(ValueError, match="Gram path needs an explicit"):
        kg.compute_gram(kg.load_data(_synth(kc_km, 16, 3)), m0, g0)
    kg.free()
    kb.free()
    kf.free()


def test_exports_and_no_foreign_symbols():
    from tabpwa.cuda.build import ensure
    lib = ensure("kernels_v5_pwa_gen_bw.cu", "libcuda_kernels_v5_pwa_gen_bw.so")
    assert lib and os.path.exists(lib)

    defined = subprocess.run(["nm", "-D", "--defined-only", lib],
                             capture_output=True, text=True).stdout
    names = {ln.split()[-1] for ln in defined.splitlines() if ln.strip()}
    for want in ("cuda_create_context_v5_gen_bw", "cuda_load_data_v5",
                 "cuda_free_context_v5", "cuda_free_data_v5",
                 "cuda_gram_matrix_v5", "cuda_compute_v5_gen_bw",
                 "cuda_get_device_count", "cuda_get_device_name"):
        assert want in names, want
    # the siblings' create/compute must NOT be exported here
    assert "cuda_create_context_genv5s0e" not in names
    assert "cuda_compute_v5_coeff" not in names
    assert "cuda_create_context_v5_gen_flat" not in names
    assert "cuda_compute_v5_gen_flat" not in names

    undef = subprocess.run(["nm", "-D", "--undefined-only", lib],
                           capture_output=True, text=True).stdout
    for ln in undef.splitlines():
        sym = ln.split()[-1]
        assert "cuda_create_context_" not in sym, sym
        assert "cuda_compute_v5" not in sym, sym
        assert "genv5s0e" not in sym and "gen_flat" not in sym, sym
