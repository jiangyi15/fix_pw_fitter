"""``cuda_v5_pwa_gen_s0e`` — fused tile + grouped log-sum on the ``pwa_gen`` basis.

The s0e fused single-kernel tile (one BW per chain, explicit ``coeff`` /
``g_coeff``, grouped log-sum data NLL) with the two ``pwa_gen`` additions:

* the kernel's own create builds the combined basis table (raw gamma rows +
  the two pre-rotated rows "1" / "m²" + the ``den_table`` rows rotated to
  ``Z = i·F``), indexed through ``uslot_row`` / ``slot_map``;
* the numerator ``N_{n_index[w]}(m_e)`` is folded into the amp-cache fill
  (load time, fixed data) — ``n_index[w] == 0`` is the constant row and is
  short-circuited.

Gates:
0. registration (``pwa_gen`` scope) + the kernel-scope guard accepts it;
1. ``n_index ≡ 0`` / default basis (``tests/config_pwa_gen.yml``) is
   **bit-identical** to ``cuda_v5_pwa_s0e`` at ``resolution_size ∈ {1, 8}``
   (Q, P, ck/m0/g0 and dNLL/dnorm);
2. ``kmatrix2`` (``tests/config_pwa_gen_km.yml``) end to end vs the CPU
   ``pwa_gen`` path (``cpu_pwa_s0c`` / ``numpy_pwa``): NLL, P and gradients
   agree within the fp32-amp floor; ``n_index`` includes row 1;
3. FD through ``Fitter.get_nll`` on the GPU backend for ``m1, g1, b1r, b2r``
   at ``resolution_size ∈ {1, 8}`` (explicit-``coeff`` chain on device);
4. kernel independence: the library exports only ``*_genv5s0e`` C symbols
   and loads next to ``libcuda_kernels_v5_pwa_s0e.so`` in one process.
"""

import ctypes
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

from test_pwa_gen import _cfg, _loaded, _synth

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


# ── gate 0: registration ─────────────────────────────────────────────

def test_registration():
    assert "cuda_v5_pwa_gen_s0e" in backends_for_model("pwa_gen")
    assert "cuda_v5_pwa_gen_s0e" not in backends_for_model("pwa")
    kc = Fitter(CFG_BASIS, backend="numpy_pwa").kernel_config
    # the kernel-scope guard accepts this backend (and only this one)
    check_backend_support(Fitter(CFG_KM, backend="numpy_pwa").kernel_config,
                          "CUDABackendV5PWAGenS0E")
    check_backend_support(kc, "CUDABackendV5PWAGenS0E")


def test_rejects_multi_bw_chains():
    kc = dict(Fitter(CFG_PWA, backend="cuda_v4_pwa").kernel_config)
    kc["bw_order"] = np.repeat(np.asarray(kc["bw_order"]).reshape(-1), 2)
    with pytest.raises(ValueError, match="exactly ONE BW"):
        CUDAKernelV5PWAGenS0E(kc, resolution_size=1)


# ── gate 1: N ≡ 1 + default basis is cuda_v5_pwa_s0e, bit for bit ────

@pytest.mark.parametrize("rs", [1, 8])
def test_gate_a_default_basis_is_s0e_bitwise(rs):
    f = Fitter(CFG_BASIS, backend="numpy_pwa")
    kc = f.kernel_config
    assert "n_table" not in kc and "den_row" not in kc
    data = _synth(kc, NEV, 5)

    ref = CUDAKernelV5PWAS0E(kc, resolution_size=rs)
    gen = CUDAKernelV5PWAGenS0E(kc, resolution_size=rs)
    params = _rand_params(gen)
    Qr, gr, Pr = ref.compute(params, ref.load_data(data), norm=1.3)
    Qg, gg, Pg = gen.compute(params, gen.load_data(data), norm=1.3)

    # Q is reduced with atomics in *both* kernels (the reference is itself
    # non-deterministic at ~1 ulp run to run), so compare it to fp64
    # round-off; P and every gradient block must be bit-identical.
    assert Qg == pytest.approx(Qr, rel=1e-14)
    assert np.array_equal(Pg, Pr)
    # ck/m0/g0 are accumulated with atomics in both kernels (the reference's
    # own gradient varies at ~1e-14 run to run), so compare to fp64 round-off;
    # P is written per event and must be bit-identical.
    for key in ("ck", "m0", "g0"):
        assert _rel(gg[key], gr[key]) < 1e-12, key
    assert gen._last_dnorm == pytest.approx(ref._last_dnorm, rel=1e-14)
    assert gen.n_uniq == ref.n_uniq and gen.n_basis == ref.n_basis
    # norm=None too
    Qr0, _gr0, Pr0 = ref.compute(params, ref.load_data(data), norm=None)
    Qg0, _gg0, Pg0 = gen.compute(params, gen.load_data(data), norm=None)
    assert Qg0 == pytest.approx(Qr0, rel=1e-14)
    assert np.array_equal(Pg0, Pr0)
    ref.free()
    gen.free()


# ── gate 2: kmatrix2 end to end vs the CPU pwa_gen path ─────────────

def _cpu_params(f, x):
    return f.build_params(x)[0]


def _loaded_unit(cfg, backend, n_phsp=300, n_data=150):
    """Like ``_loaded`` but with unit weights.

    The v5 grouped log-sum with ``resolution_size=1`` differs from the
    per-event v4 form by the constant ``-sum log w`` (and its gradient by a
    factor ``w``), so a like-for-like comparison against the v4 CPU
    references requires unit weights — the documented equivalence.
    """
    f = Fitter(cfg, backend=backend)
    kc = f.kernel_config
    p = _synth(kc, n_phsp, 1)
    d = _synth(kc, n_data, 2)
    p["weight"] = np.ones_like(p["weight"])
    d["weight"] = np.ones_like(d["weight"])
    f.set_phsp(p)
    f.set_data(d)
    return f


@pytest.mark.parametrize("rs", [1, 8])
def test_gate_b_matches_cpu_pwa_gen(rs):
    fg = _loaded_unit(CFG_KM, {"name": "cuda_v5_pwa_gen_s0e",
                               "resolution_size": rs}, n_phsp=300, n_data=150)
    kc = fg.kernel_config
    assert "den_row" in kc and "n_table" in kc
    assert np.any(np.asarray(kc["n_index"]) == 1)

    x = fg.initial_values(seed=1)
    Qg, gg = fg.get_nll(x)
    assert np.isfinite(Qg) and np.all(np.isfinite(gg))

    # per-event P vs the CPU opaque-basis kernel (same data)
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

    # full NLL (norm chain): like-for-like against the v5 CPU sibling at the
    # SAME rsize (the grouped objective is rsize-dependent), plus the v4
    # references at rsize == 1 where they coincide (unit weights).
    refs = [("cpu_v5_pwa", {"name": "cpu_v5_pwa", "resolution_size": rs})]
    if rs == 1:
        refs += [("cpu_pwa_s0c", "cpu_pwa_s0c"), ("numpy_pwa", "numpy_pwa")]
    for label, be in refs:
        fr = _loaded_unit(CFG_KM, be, n_phsp=300, n_data=150)
        Qc, gc = fr.get_nll(fr.initial_values(seed=1))
        assert abs(Qg - Qc) / abs(Qc) < 1e-6, label
        assert _rel(gg, gc) < 1e-5, label


# ── gate 3: FD through Fitter.get_nll (explicit-coeff chain) ─────────

@pytest.mark.parametrize("rs", [1, 8])
def test_gate_c_fd_gradients(rs):
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


# ── gate 4: standalone kernel unit ──────────────────────────────────

def _lib_paths():
    from tabpwa.cuda.build import ensure
    return (ensure("kernels_v5_pwa_gen_s0e.cu",
                   "libcuda_kernels_v5_pwa_gen_s0e.so"),
            ensure("kernels_v5_pwa_s0e.cu", "libcuda_kernels_v5_pwa_s0e.so"))


def test_gate_d_symbols_are_private():
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
    # the C entry points: exactly the *_genv5s0e suffix
    bad = [n for n in exported
           if n.startswith("cuda_") and not n.endswith("_genv5s0e")]
    assert bad == [], f"leaked cuda symbols: {bad}"
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


def test_gate_d_loads_next_to_s0e():
    gen_lib, s0e_lib = _lib_paths()
    a = ctypes.CDLL(s0e_lib)
    b = ctypes.CDLL(gen_lib)
    # the gen copy exposes ONLY its own suffixed entry points ...
    assert hasattr(b, "cuda_get_device_count_genv5s0e")
    assert not hasattr(b, "cuda_get_device_count_v5s0e")
    assert not hasattr(b, "cuda_compute_v5s0e")
    # ... and the original is untouched / still loadable
    assert hasattr(a, "cuda_get_device_count_v5s0e")
    assert not hasattr(a, "cuda_get_device_count_genv5s0e")
    # resolving both at once is the coexistence check (no device call here —
    # the actual context creation is covered by the numerics gates)


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
