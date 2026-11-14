"""``cuda_v5_pwa_gen_s0e`` — fused tile + group-total-weight log-sum on
the ``pwa_gen`` basis.

The s0e fused single-kernel tile (one BW per chain, explicit ``coeff`` /
``g_coeff``) with the two ``pwa_gen`` additions — the kernel's own combined
basis table (raw gamma rows + pre-rotated "1" / "m²" + the ``den_table``
rows rotated to ``Z = i·F``) and the numerator ``N_{n_index[w]}(m_e)``
folded into the amp-cache fill — plus the **group-total-weight** data NLL
of ``cuda_v5_pwa_s0cw``:

    S_g = Σ_{e∈g} w_e·(P_e/norm + bkg_e),     W_g = Σ_{e∈g} w_e
    Q   = -Σ_g W_g·log(S_g/W_g)

Gates:
0. registration (``pwa_gen`` scope) + the kernel-scope guard accepts it;
1. plain-path regression: ``resolution_size=1`` with **unit** weights is
   bit-identical to ``cuda_v5_pwa_s0e`` (Q/P at fp64 atomic round-off,
   gradients at 1e-12); the ``norm=None`` path is unchanged for ANY weights;
2. ``resolution_size=1`` with **non-unit** weights is the per-event NLL:
   GPU vs ``cpu_pwa_gen`` within the fp32 amp-cache floor.  The old s0c
   form (``cuda_v5_pwa_s0e``) gives ``-Σ log(w·pdf)`` instead and fails here;
3. ``resolution_size ∈ {2, 8}`` with non-unit weights reproduces the host
   ``Σ_g W_g·log(S_g/W_g)`` built from the kernel's own ``P``;
4. a zero-weight group is inert (same host formula with the group skipped,
   finite gradients — no ``log(0)`` / NaN);
5. ``kmatrix2`` (``tests/config_pwa_gen_km.yml``) P / ``coeff`` gradients
   against the CPU ``pwa_gen`` kernel; FD through ``Fitter.get_nll`` for
   ``m1, g1, b1r, b2r`` at ``resolution_size ∈ {1, 8}``;
6. kernel independence: the library exports only the ``_v5`` lifecycle entry
   points, its own ``cuda_create_context_genv5s0e`` (pwa_gen basis tables) and
   ``cuda_compute_v5_coeff`` (the 13-argument s0e/gens compute — the 14-argument
   ``cuda_compute_v5_s0mult`` of s0c/s0d/s0cw is a different signature), and
   loads next to ``libcuda_kernels_v5_pwa_s0e.so`` in one process.
"""

import ctypes
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

from test_pwa_gen import _loaded, _synth

from tabpwa import Fitter
from tabpwa.backends import backends_for_model, create_backend
from tabpwa.cuda._v5_pwa_gen_s0e import CUDAKernelV5PWAGenS0E
from tabpwa.cuda._v5_pwa_s0e import CUDAKernelV5PWAS0E
from tabpwa.pwa_gen_support import check_backend_support

CFG_BASIS = "tests/config_pwa_gen.yml"
CFG_KM = "tests/config_pwa_gen_km.yml"
CFG_PWA = "tests/config_pwa.yml"
NEV = 400
NH = 1e-6


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


def _host_Q(P, weight, bkg, norm, rs):
    """``Q = Σ_g W_g·log(S_g/W_g)`` from the kernel's own per-event P.

    Zero-weight groups (``W_g == 0``) are skipped — that is the kernel's
    rule, so the host side must skip them too.
    """
    P = np.asarray(P, np.float64)
    w = np.asarray(weight, np.float64)
    b = np.asarray(bkg, np.float64)
    Q = 0.0
    for a in range(0, P.size, rs):
        sl = slice(a, a + rs)
        S = float(np.sum(w[sl] * (P[sl] / norm + b[sl])))
        W = float(np.sum(w[sl]))
        if W > 0.0:
            Q -= W * np.log(S / W)
    return Q


# ── gate 0: registration ─────────────────────────────────────────────

def test_registration():
    assert "cuda_v5_pwa_gen_s0e" in backends_for_model("pwa_gen")
    assert "cuda_v5_pwa_gen_s0e" not in backends_for_model("pwa")
    kc = Fitter(CFG_BASIS, backend="numpy_pwa").kernel_config
    # the kernel-scope guard accepts this backend (and only this one)
    check_backend_support(Fitter(CFG_KM, backend="numpy_pwa").kernel_config,
                          "CUDABackendV5PWAGenS0E")
    check_backend_support(kc, "CUDABackendV5PWAGenS0E")
    # exactly one pwa_gen GPU log-sum kernel: no sibling / alias remains
    gen = [b for b in backends_for_model("pwa_gen") if "gen_s0e" in b]
    assert gen == ["cuda_v5_pwa_gen_s0e"], gen


def test_rejects_multi_bw_chains():
    kc = dict(Fitter(CFG_PWA, backend="cuda_v4_pwa").kernel_config)
    kc["bw_order"] = np.repeat(np.asarray(kc["bw_order"]).reshape(-1), 2)
    with pytest.raises(ValueError, match="exactly ONE BW"):
        CUDAKernelV5PWAGenS0E(kc, resolution_size=1)


# ── gate 1: unit weights at rsize=1 is cuda_v5_pwa_s0e, bit for bit ──

def test_gate_a_default_basis_unit_weights_is_s0e_bitwise():
    f = Fitter(CFG_BASIS, backend="numpy_pwa")
    kc = f.kernel_config
    assert "n_table" not in kc and "den_row" not in kc
    data = _synth(kc, NEV, 5)
    data["weight"] = np.ones_like(data["weight"])       # W_g == 1

    ref = CUDAKernelV5PWAS0E(kc, resolution_size=1)
    gen = CUDAKernelV5PWAGenS0E(kc, resolution_size=1)
    params = _rand_params(gen)
    Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
    Qg, gg, Pg = gen.compute(params, gen.load_data(data), norm=1.3)

    # Q is reduced with atomics in *both* kernels (the reference is itself
    # non-deterministic at ~1 ulp run to run), so compare it to fp64
    # round-off; P and every gradient block must be bit-identical.
    assert Qg == pytest.approx(Qr, rel=1e-14)
    assert np.array_equal(Pg, Pr)
    # ck/m0/g0 are accumulated with atomics in both kernels, so compare to
    # fp64 round-off; P is written per event and must be bit-identical.
    for key in ("ck", "m0", "g0"):
        assert _rel(gg[key], gr[key]) < 1e-12, key
    assert gen._last_dnorm == pytest.approx(ref._last_dnorm, rel=1e-14)
    assert gen.n_uniq == ref.n_uniq and gen.n_basis == ref.n_basis
    ref.free()
    gen.free()


def test_gate_a_norm_none_path_is_unchanged():
    """The ``use_norm == 0`` path is per-event linear — any weights."""
    kc = Fitter(CFG_BASIS, backend="numpy_pwa").kernel_config
    data = _synth(kc, NEV, 9)          # non-unit weights on purpose
    ref = CUDAKernelV5PWAS0E(kc, resolution_size=8)
    gen = CUDAKernelV5PWAGenS0E(kc, resolution_size=8)
    params = _rand_params(gen, seed=11)
    Qr, _gr, Pr = ref.compute(params, ref.load_data(data), norm=None)
    Qg, _gg, Pg = gen.compute(params, gen.load_data(data), norm=None)
    # Q = Σ w·P — summed on the host from the SAME P in both cases
    assert Qg == pytest.approx(Qr, rel=1e-14)
    assert np.array_equal(Pg, Pr)
    ref.free()
    gen.free()


# ── gate 2: rsize=1 with NON-unit weights is the per-event NLL ───────

def test_gate_b_rsize1_non_unit_weights_matches_per_event_cpu():
    """The point of the change: ``-w·log(w·pdf/w) == -w·log pdf``.

    ``cpu_pwa_gen`` is the per-event reference.  The old s0c-form kernel
    (``cuda_v5_pwa_s0e``, same config, same params) computes
    ``-Σ log(w·pdf)`` and is off by ``Σ log w`` — reported below so the
    discrimination is visible.
    """
    kc = Fitter(CFG_BASIS, backend="numpy_pwa").kernel_config
    data = _synth(kc, NEV, 5)
    w = np.asarray(data["weight"])
    assert not np.allclose(w, 1.0)                    # non-unit on purpose
    norm = 1.3
    params = _rand_params_pwa_gen(kc, seed=7)

    gen = CUDAKernelV5PWAGenS0E(kc, resolution_size=1)
    Qg, gg, Pg = gen.compute(params, gen.load_data(data), norm=norm)

    ref = create_backend("cpu_pwa_gen", kc, model="pwa_gen")
    hr = ref.load_data(data)
    Qr, gr, Pr = ref.compute(params, hr, norm=norm)
    assert _rel(Pg, Pr) < 1e-6
    assert _rel(Qg, Qr) < 1e-6, (Qg, Qr)
    for key in ("ck", "m0", "g0"):
        assert _rel(gg[key], gr[key]) < 1e-5, key
    ref.free()

    # the old objective on the identical inputs
    old = CUDAKernelV5PWAS0E(kc, resolution_size=1)
    Qo, _go, Po = old.compute(params, old.load_data(data), norm=norm)
    assert _rel(Po, Pg) < 1e-6                        # same P, only the NLL
    old.free()
    # before: |Qo − Qcpu| / |Qcpu| is O(1) (the Σ log w offset);
    # after:  |Qg − Qcpu| / |Qcpu| < 1e-6.
    before = abs(Qo - Qr) / abs(Qr)
    after = abs(Qg - Qr) / abs(Qr)
    assert before > 1e-3, (before, after)
    assert after < 1e-6, (before, after)


def test_gate_b_full_nll_vs_per_event_cpu():
    """Full norm-chained NLL (purity constant included) at rsize=1."""
    fg = _loaded(CFG_KM, {"name": "cuda_v5_pwa_gen_s0e",
                          "resolution_size": 1}, n_phsp=300, n_data=150)
    kc = fg.kernel_config
    assert "den_row" in kc and "n_table" in kc
    assert np.any(np.asarray(kc["n_index"]) == 1)

    x = fg.initial_values(seed=1)
    Qg, gg = fg.get_nll(x)
    assert np.isfinite(Qg) and np.all(np.isfinite(gg))

    fr = _loaded(CFG_KM, "cpu_pwa_gen", n_phsp=300, n_data=150)
    xr = fr.initial_values(seed=1)
    Qc, gc = fr.get_nll(xr)
    assert np.allclose(x, xr)
    assert abs(Qg - Qc) / abs(Qc) < 1e-6, (Qg, Qc)
    assert _rel(gg, gc) < 1e-5


# ── gate 3: the W_g formula at rsize > 1 ────────────────────────────

@pytest.mark.parametrize("rs", [2, 8])
def test_gate_c_host_w_formula(rs):
    """``Q = Σ_g W_g·log(S_g/W_g)`` rebuilt from the kernel's own P.

    1e-9 relative is the fp64 atomic accumulation order across blocks (the
    host sums sequentially, the kernel atomically) — the math is identical.
    """
    kc = Fitter(CFG_BASIS, backend="numpy_pwa").kernel_config
    data = _synth(kc, NEV, 3)
    assert not np.allclose(data["weight"], 1.0)
    norm = 0.87
    k = CUDAKernelV5PWAGenS0E(kc, resolution_size=rs)
    Q, _g, P = k.compute(_rand_params_pwa_gen(kc, seed=4), k.load_data(data),
                         norm=norm, return_p=True)
    Qh = _host_Q(P, data["weight"], data["bkg"], norm, rs)
    assert abs(Q - Qh) / abs(Qh) < 1e-9, (Q, Qh)
    k.free()

    # the s0c form is NOT the same number here (non-unit weights): the
    # objective really is rsize-dependent.
    S_list = [np.sum(data["weight"][a:a + rs]
                     * (P[a:a + rs] / norm + data["bkg"][a:a + rs]))
              for a in range(0, P.size, rs)]
    Qs0c = float(-np.sum(np.log(S_list)))
    assert abs(Q - Qs0c) / abs(Qh) > 1e-3


# ── gate 4: a zero-weight group is inert ────────────────────────────

@pytest.mark.parametrize("rs", [1, 8])
def test_gate_d_zero_weight_group(rs):
    kc = Fitter(CFG_BASIS, backend="numpy_pwa").kernel_config
    data = _synth(kc, NEV, 3)
    w = np.array(data["weight"])
    w[:rs] = 0.0                       # one COMPLETE group with W_g == 0
    data["weight"] = w
    norm = 0.87
    k = CUDAKernelV5PWAGenS0E(kc, resolution_size=rs)
    Q, g, P = k.compute(_rand_params_pwa_gen(kc, seed=4), k.load_data(data),
                        norm=norm, return_p=True)

    # the same host formula skips W_g == 0 (what the kernel must do)
    assert np.isfinite(Q)
    Qh = _host_Q(P, data["weight"], data["bkg"], norm, rs)
    assert abs(Q - Qh) / abs(Qh) < 1e-9, (Q, Qh)
    # and it is NOT the "with group" value (which is -W·log(0) = +inf)
    for key, val in g.items():
        assert np.all(np.isfinite(val)), key
    assert np.isfinite(k._last_dnorm)
    k.free()


# ── gate 5: kmatrix2 end to end + FD ────────────────────────────────

def test_gate_e_kmatrix2_direct_vs_cpu():
    fg = _loaded(CFG_KM, {"name": "cuda_v5_pwa_gen_s0e",
                          "resolution_size": 1}, n_phsp=300, n_data=150)
    kc = fg.kernel_config
    x = fg.initial_values(seed=1)

    p = fg.build_params(x)[0]
    Qn, gn, Pn = fg.backend.compute(p, fg._data_holder, norm=None)
    ref = create_backend("cpu_pwa_s0c", kc, model="pwa_gen")
    hr = ref.load_data(fg._data_np)
    Qr, gr, Pr = ref.compute(p, hr, norm=None)
    assert _rel(Pn, Pr) < 1e-6
    assert _rel(Qn, Qr) < 1e-6
    for key in ("ck", "coeff"):
        assert _rel(gn[key], gr[key]) < 1e-5, key
    ref.free()


@pytest.mark.parametrize("rs", [1, 8])
def test_gate_f_fd_gradients(rs):
    f = _loaded(CFG_KM, {"name": "cuda_v5_pwa_gen_s0e",
                         "resolution_size": rs}, n_phsp=200, n_data=120)
    x = f.initial_values(seed=5)
    nll, grad = f.get_nll(x)
    assert np.isfinite(nll) and np.all(np.isfinite(grad))
    assert np.isfinite(f.backend.kernel._last_dnorm)
    names = f.free_param_names()
    for nm in ("k2_m1", "k2_g1", "k2_b1r", "k2_b2r"):
        i = names.index(nm)
        xp = x.copy()
        xp[i] += NH
        xm = x.copy()
        xm[i] -= NH
        fd = (f.get_nll(xp)[0] - f.get_nll(xm)[0]) / (2 * NH)
        assert abs(grad[i] - fd) <= 1e-4 * max(abs(fd), 1.0), (
            nm, grad[i], fd)


def _rand_params_pwa_gen(kc, seed=7):
    """Kernel params for a default-basis config (m0/g0 path)."""
    r = np.random.RandomState(seed)
    n_wave = int(np.asarray(kc["matrix_angle"]).shape[1])
    n_proj = int(kc.get("n_proj", 1) or 1)
    return {"ck": 0.5 * (r.randn(n_wave // n_proj)
                         + 1j * r.randn(n_wave // n_proj)),
            "m0": np.linspace(0.7, 1.5, int(np.max(kc["m0_index"])) + 1),
            "g0": np.linspace(0.1, 0.5, int(np.max(kc["g0_index"])) + 1)}


# ── gate 6: standalone kernel unit ──────────────────────────────────

def _lib_paths():
    from tabpwa.cuda.build import ensure
    return (ensure("kernels_v5_pwa_gen_s0e.cu",
                   "libcuda_kernels_v5_pwa_gen_s0e.so"),
            ensure("kernels_v5_pwa_s0e.cu", "libcuda_kernels_v5_pwa_s0e.so"))


def test_gate_g_symbols_are_private():
    gen_lib, s0e_lib = _lib_paths()
    assert gen_lib and s0e_lib
    out = subprocess.run(["nm", "-D", "--defined-only", gen_lib],
                         capture_output=True, text=True, check=True).stdout
    exported = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        typ, name = parts[-2], parts[-1]
        # "u" = unique globals of the CUDA/thrust runtime headers; not ours
        if typ == "u":
            continue
        exported.append(name)
    assert exported, "no defined symbols found"
    # the C entry points: the shared v5 family names, plus the two 13-argument
    # specialities — the gen kernel's own create (pwa_gen basis/numerator
    # tables) and the s0e compute (one flat [n_wave·n_basis] gradient block
    # instead of ogs0 + ogmult)
    allowed = {
        "cuda_create_context_genv5s0e",
        "cuda_free_context_v5",
        "cuda_load_data_v5",
        "cuda_free_data_v5",
        "cuda_compute_v5_coeff",
        "cuda_gram_matrix_v5",
        "cuda_get_device_count",
        "cuda_get_device_name",
    }
    bad = [n for n in exported if n.startswith("cuda_") and n not in allowed]
    assert bad == [], f"leaked cuda symbols: {bad}"
    assert {n for n in exported if n.startswith("cuda_")} == allowed
    # everything else must be a *_gen helper (or a cuda::std instantiation)
    cxx = [n for n in exported if not n.startswith("cuda_")]
    demangled = subprocess.run(["c++filt"] + cxx, capture_output=True,
                               text=True, check=True).stdout.splitlines()
    for dem in demangled:
        if "cuda::std" in dem:
            continue
        base = dem.split("(")[0]
        assert base.endswith("_gen") or "_gen_" in base, (
            f"unexpected exported helper: {dem}")
    assert cxx, "no helper symbols found"


def test_gate_g_loads_next_to_s0e():
    gen_lib, s0e_lib = _lib_paths()
    a = ctypes.CDLL(s0e_lib)
    b = ctypes.CDLL(gen_lib)
    # the gen copy keeps its own basis-table create; every other entry point
    # uses the shared v5 family names (same signature, separate libraries)
    assert hasattr(b, "cuda_create_context_genv5s0e")
    assert hasattr(a, "cuda_create_context_v5")
    assert not hasattr(a, "cuda_create_context_genv5s0e")
    for nm in ("cuda_compute_v5_coeff", "cuda_get_device_count"):
        assert hasattr(a, nm) and hasattr(b, nm)
    # the 14-argument family computes (m0g0 / s0mult) are NOT exported here
    assert not hasattr(a, "cuda_compute_v5_s0mult")
    assert not hasattr(b, "cuda_compute_v5_s0mult")
    assert not hasattr(a, "cuda_compute_v5_m0g0")
    assert not hasattr(b, "cuda_compute_v5_m0g0")
    # no undefined reference to another kernel's code: every cuda_* undefined
    # symbol must come from the CUDA runtime, not from the sibling library
    out = subprocess.run(["nm", "-D", "--undefined-only", gen_lib],
                         capture_output=True, text=True, check=True).stdout
    undef = [ln.split()[-1] for ln in out.splitlines() if ln.split()]
    assert [n for n in undef if n.startswith("cuda_compute_")] == []


# ── grouped log-sum bookkeeping carried over from s0e ───────────────

def test_partial_last_group_warns():
    kc = Fitter(CFG_BASIS, backend="numpy_pwa").kernel_config
    k = CUDAKernelV5PWAGenS0E(kc, resolution_size=8)
    data = _synth(kc, 5, 3)          # 5 not a multiple of 8
    with pytest.warns(RuntimeWarning):
        k.load_data(data)
    k.free()


def test_generic_does_not_upload_bogus_gamma_columns():
    """The generic path must not invent ``gamma_col_idx`` from matrix_gamma.

    The device never reads m0/g0/gamma-row indices when the basis is
    explicit; uploading ``argmax(matrix_gamma, axis=1)`` (and the unused
    g0 arrays) is both meaningless and misleading.
    """
    kc = Fitter(CFG_KM, backend="numpy_pwa").kernel_config
    k = CUDAKernelV5PWAGenS0E(kc, resolution_size=1)
    try:
        assert k._generic
        assert k._gamma_col_idx.size == 0
        assert k._g0_index_w.size == 0
        assert k._g0_mass_index_w.size == 0
        assert k.row_start.size == 0
    finally:
        k.free()
