"""cpu_v5_pwa — CPU (C + OpenMP) general pure-PWA with the v5 group NLL.

``cpu_v5_pwa`` is the CPU counterpart of ``cuda_v5_pwa``: the general
(any ``n_res``) projection-sum PWA of ``cpu_pwa_gen`` with ONE difference —
in the data-NLL branch events are grouped in contiguous blocks of
``resolution_size`` and one log is taken per group,

    q_e  = w_e·(P_e/norm + bkg_e) ,  S_g = Σ_{e∈g} q_e
    Q    = −Σ_g log(S_g)
    dQ/dP_e  = −(w_e/norm)/S_g
    dQ/dnorm = Σ_g (1/S_g)·(Σ_{e∈g} w_e·P_e)/norm²

The phase-space path (``norm=None``) is the gen per-event linear Σ w·P and
the returned per-event ``P`` stays ``P_sig = Σ_p |A_p|²``.

Correctness anchors:

* ``resolution_size <= 1`` (in practice ``1``): **bit-identical** to the
  v4 kernel (``cpu_pwa_gen`` / archived ``kernels_cpu_pwa_gen_v1.c``) —
  Q / P / ck / m0 / g0 / dnorm / ``_last_dnorm`` — for unit AND non-unit
  weights, both norm modes, on ``config_pwa.yml``, the tiled 198-wave
  layout and the ``n_res == 2`` fixture.  A one-event *group* would
  instead give ``−log(w_e·(P_e/norm+b_e))`` = v4 − ``log(w_e)``, i.e. off
  by the constant ``−Σ_e log(w_e)`` with a different ``dQ/dP`` — which is
  exactly why that size never uses the group formula.
* ``resolution_size ∈ {2, 8}``: the GPU sibling ``cuda_v5_pwa`` on the
  same data/config (fp32 amp-cache floor; the GPU also groups at the same
  ``resolution_size``), plus ``n_res == 2`` at ``rs = 8``.
* ``resolution_size`` = 64 (== SOA_TILE) and 65/100/1000 (group spans
  chunks): a naive numpy group reference (and the GPU) — the
  linearity/const_g branch.
* Gradients through ``Fitter.get_nll`` at ``resolution_size ∈ {2, 8}``.
"""
import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import create_backend, backends_for_model, backend_class

from test_cpu_pwa_gen import _multires_dic
from test_cpu_pwa_s0c import _params, _synth

CFG = "tests/config_pwa.yml"


# ── helpers ────────────────────────────────────────────────────────

def _kernel_config(cfg=CFG):
    return Fitter(cfg, backend="numpy_pwa").kernel_config


def _gen(kc):
    return create_backend("cpu_pwa_gen", kc, model="pwa")


def _v5(kc, rs=1):
    return create_backend("cpu_v5_pwa", kc, model="pwa", resolution_size=rs)


def _cuda(kc, rs):
    try:
        return create_backend("cuda_v5_pwa", kc, model="pwa",
                              resolution_size=rs)
    except Exception as e:                                # noqa: BLE001
        pytest.skip(f"cuda_v5_pwa unavailable: {e}")


def _unit(data):
    return dict(data, weight=np.ones_like(data["weight"]))


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(a))),
                                             1e-300))


def _naive_group_nll(P, w, b, norm, rs):
    """Reference Q and dQ/dnorm for contiguous groups of ``rs`` events."""
    ng = (len(P) + rs - 1) // rs
    S = np.empty(ng)
    swp = np.empty(ng)
    for g in range(ng):
        sl = slice(g * rs, min((g + 1) * rs, len(P)))
        S[g] = float(np.sum(w[sl] * (P[sl] / norm + b[sl])))
        swp[g] = float(np.sum(w[sl] * P[sl]))
    Q = -float(np.sum(np.log(S)))
    dnorm = float(np.sum(swp / (norm * norm * S)))
    return Q, dnorm


# ── registration / shape ───────────────────────────────────────────

def test_registration_and_alias():
    avail = backends_for_model("pwa")
    assert "cpu_v5_pwa" in avail
    assert "cpu_v5_pwa_gen" in avail          # stacked alias
    assert "cpu_pwa_gen" in avail and "cpu_v4_pwa" in avail
    assert backend_class("cpu_v5_pwa", "pwa") \
        is backend_class("cpu_v5_pwa_gen", "pwa")
    assert backend_class("cpu_pwa_gen", "pwa") \
        is backend_class("cpu_v4_pwa", "pwa")


def test_one_kernel_one_library_for_both_names():
    """``cpu_pwa_gen`` is a thin alias: same class hierarchy, same .so."""
    import os
    from tabpwa.cpu import _pwa_gen, _pwa_v5
    from tabpwa.cpu.build import ensure

    kc = _kernel_config()
    assert issubclass(_pwa_gen.CPUKernelPWAGen, _pwa_v5.CPUKernelPWAV5)
    g = _pwa_gen.CPUKernelPWAGen(kc)
    assert g.resolution_size == 1
    # the alias really runs the v5 symbols (one kernel), not a v4 build
    assert hasattr(g._lib, "cpu_pwa_v5_compute")
    assert not hasattr(g._lib, "cpu_pwa_gen_compute")

    p = ensure("kernels_cpu_pwa_v5.c", "libcpu_kernels_cpu_pwa_v5.so")
    assert p and p.endswith("libcpu_kernels_cpu_pwa_v5.so")
    cpu_dir = os.path.dirname(p)
    # the superseded v4 kernel is archived: no live source, no built lib
    assert not os.path.exists(
        os.path.join(cpu_dir, "kernels_cpu_pwa_gen.c"))
    assert os.path.exists(os.path.join(
        cpu_dir, "archive", "kernels_cpu_pwa_gen_v1.c"))
    # (a stale libcpu_kernels_cpu_pwa_gen.so from an older checkout may
    #  linger as a build artefact — nothing builds or loads it any more)
    g.free()


def test_shapes_and_resolution_size():
    kc = _kernel_config()
    b = _v5(kc, rs=8)
    k = b.kernel
    assert k.resolution_size == 8
    assert k.n_res == 1
    assert k.n_wave == kc["matrix_angle"].shape[1]
    assert k.n_proj == int(kc["n_proj"]) == 2
    assert k.n_wave_base * k.n_proj == k.n_wave
    assert k.n_fact == k.n_wave
    assert k.dnorm_on_gpu is True
    b.free()
    assert _v5(kc, rs=1).kernel.resolution_size == 1


# ── rs <= 1: the v4 per-event path, bit-identical to cpu_pwa_gen ──

@pytest.mark.parametrize("norm", [None, 1.7])
def test_rs1_bit_identical_to_gen(norm):
    kc = _kernel_config()
    gen, v5 = _gen(kc), _v5(kc, rs=1)
    data = _unit(_synth(kc, 400, seed=11))
    params = _params(gen.kernel, seed=7, scale=0.5)

    Q1, g1, P1 = gen.compute(params, gen.load_data(data), norm=norm)
    Q2, g2, P2 = v5.compute(params, v5.load_data(data), norm=norm)

    assert np.float64(Q1) == np.float64(Q2)
    assert np.array_equal(P1, P2)
    for key in ("ck", "m0", "g0"):
        assert np.array_equal(np.asarray(g1[key]), np.asarray(g2[key])), key
    if norm is not None:
        assert gen.kernel._last_dnorm == v5.kernel._last_dnorm
    else:
        assert v5.kernel._last_dnorm is None
    gen.free(); v5.free()


@pytest.mark.parametrize("norm", [None, 200.0])
def test_rs1_bit_identical_tiled_layout(norm):
    """198 waves: ``synthetic_case("config_pwa.yml", 100, 2, 400)``."""
    from benchmark_pwa_kernels import synthetic_case

    kc, data, _phsp, params, _norm, info = synthetic_case(CFG, 100, 2, 400)
    assert info["n_wave"] == 198 and info["N"] == 99
    data = _unit(data)
    gen, v5 = _gen(kc), _v5(kc, rs=1)

    Q1, g1, P1 = gen.compute(params, gen.load_data(data), norm=norm)
    Q2, g2, P2 = v5.compute(params, v5.load_data(data), norm=norm)
    assert np.float64(Q1) == np.float64(Q2)
    assert np.array_equal(P1, P2)
    for key in ("ck", "m0", "g0"):
        assert np.array_equal(np.asarray(g1[key]), np.asarray(g2[key])), key
    if norm is not None:
        assert gen.kernel._last_dnorm == v5.kernel._last_dnorm
    gen.free(); v5.free()


def test_rs1_nonunit_weights_identical_to_v4():
    """``rs = 1`` is the v4 per-event NLL for ANY weights — never the group
    formula.

    A one-event group would give ``−log(w_e·(P_e/norm + b_e))``, which is
    the v4 term ``−w_e·log(…)`` only at ``w_e = 1``; for non-unit weights
    the two differ by the constant ``−Σ_e log(w_e)`` (and ``dQ/dP`` by a
    factor ``w_e``).  ``cpu_v5_pwa(rsize=1)`` therefore takes the exact v4
    path and must reproduce ``cpu_pwa_gen`` bit for bit at non-unit
    weights too — that is what lets ``cpu_pwa_gen`` be a thin alias.
    """
    kc = _kernel_config()
    gen, v5 = _gen(kc), _v5(kc, rs=1)
    data = _synth(kc, 400, seed=11)              # non-unit weights
    assert not np.all(data["weight"] == 1.0)
    params = _params(gen.kernel, seed=7, scale=0.5)

    for norm in (None, 1.7):
        Q1, g1, P1 = gen.compute(params, gen.load_data(data), norm=norm)
        Q2, g2, P2 = v5.compute(params, v5.load_data(data), norm=norm)
        assert np.float64(Q1) == np.float64(Q2)
        assert np.array_equal(P1, P2)
        for key in ("ck", "m0", "g0"):
            assert np.array_equal(np.asarray(g1[key]), np.asarray(g2[key])), key
        if norm is not None:
            assert gen.kernel._last_dnorm == v5.kernel._last_dnorm
            # documentation of the design decision (pure numpy):
            #   group-of-one  = −Σ log(w·(P/norm+b)) = unit-weight v4 − Σ log w
            # while the kernel's rsize=1 value IS the weighted per-event v4
            w, b, P = data["weight"], data["bkg"], P1
            q_unit = -float(np.sum(np.log(P / norm + b)))     # v4 at w = 1
            q_grp1 = -float(np.sum(np.log(w * (P / norm + b))))
            shift = -float(np.sum(np.log(w)))
            assert q_grp1 == pytest.approx(q_unit + shift, abs=1e-10)
            assert Q1 == pytest.approx(
                -float(np.sum(w * np.log(P / norm + b))), abs=1e-10)
    gen.free(); v5.free()


# ── phase-space path: rs-independent, bit-identical to gen ─────────

@pytest.mark.filterwarnings("ignore:.*not a multiple.*:RuntimeWarning")
@pytest.mark.parametrize("rs", [2, 8, 100, 1000])
def test_phsp_path_identical_to_gen_for_any_rsize(rs):
    """``norm=None`` is the linear Σ w·P: groups do not exist there."""
    kc = _kernel_config()
    gen, v5 = _gen(kc), _v5(kc, rs=rs)
    data = _synth(kc, 400, seed=11)
    params = _params(gen.kernel, seed=7, scale=0.5)

    Q1, g1, P1 = gen.compute(params, gen.load_data(data), norm=None)
    Q2, g2, P2 = v5.compute(params, v5.load_data(data), norm=None)
    assert np.float64(Q1) == np.float64(Q2)
    assert np.array_equal(P1, P2)
    for key in ("ck", "m0", "g0"):
        assert np.array_equal(np.asarray(g1[key]), np.asarray(g2[key])), key
    assert v5.kernel._last_dnorm is None
    gen.free(); v5.free()


# ── group NLL: naive numpy reference (incl. rs > SOA_TILE) ─────────

@pytest.mark.parametrize("rs", [2, 8, 64, 65, 100, 1000])
def test_group_nll_matches_numpy_reference(rs):
    """Covers both chunk edges: rs == SOA_TILE (64, one group per chunk) and
    rs > SOA_TILE (65/100/1000, group spans chunks)."""
    kc = _kernel_config()
    gen, v5 = _gen(kc), _v5(kc, rs=rs)
    data = _synth(kc, 400, seed=11)              # 400 not a multiple of 1000
    params = _params(gen.kernel, seed=7, scale=0.5)

    if 400 % rs:
        with pytest.warns(RuntimeWarning, match="not a multiple"):
            h = v5.load_data(data)
    else:
        h = v5.load_data(data)
    norm = float(gen.compute(params, gen.load_data(data), norm=None)[0])
    Q, _g, P = v5.compute(params, h, norm=norm)

    w, b = data["weight"], data["bkg"]
    Qref, dnref = _naive_group_nll(P, w, b, norm, rs)
    assert Q == pytest.approx(Qref, abs=1e-10 * max(abs(Qref), 1.0))
    assert v5.kernel._last_dnorm == pytest.approx(
        dnref, rel=1e-13)
    gen.free(); v5.free()


def test_partial_last_group():
    """A final partial group logs only its own events (like the GPU)."""
    kc = _kernel_config()
    v5 = _v5(kc, rs=16)
    data = _synth(kc, 200, seed=5)               # 12 full groups + 8 events
    params = _params(v5.kernel, seed=3, scale=0.5)
    norm = 2.0
    with pytest.warns(RuntimeWarning, match="not a multiple"):
        h = v5.load_data(data)
    Q, _g, P = v5.compute(params, h, norm=norm)
    Qref, _dn = _naive_group_nll(P, data["weight"], data["bkg"], norm, 16)
    assert Q == pytest.approx(Qref, abs=1e-11 * max(abs(Qref), 1.0))
    v5.free()


# ── group NLL: GPU sibling ────────────────────────────────────────

@pytest.mark.parametrize("rs", [2, 8])
def test_group_matches_cuda_v5_pwa(rs):
    """Same data/config/resolution_size as the general GPU v5 kernel.

    ``cuda_v5_pwa`` also groups at ``resolution_size``, so the comparison
    is direct; the floor is the fp32 amplitude cache (~5e-8 on P).
    """
    kc = _kernel_config()
    data = _synth(kc, 416, seed=11)
    cud = _cuda(kc, rs)
    v5 = _v5(kc, rs=rs)
    params = _params(v5.kernel, seed=7, scale=0.5)

    hc = cud.load_data(data)
    norm = float(cud.compute(params, hc, norm=None)[0])
    Qc, gc, Pc = cud.compute(params, hc, norm=norm)
    Q5, g5, P5 = v5.compute(params, v5.load_data(data), norm=norm)

    assert abs(Qc - Q5) <= 1e-6 * abs(Qc)
    assert _rel(Pc, P5) <= 1e-6
    for key in ("ck", "m0", "g0"):
        assert _rel(gc[key], g5[key]) <= 1e-6, key
    assert abs(cud.kernel._last_dnorm - v5.kernel._last_dnorm) \
        <= 1e-6 * abs(cud.kernel._last_dnorm)
    cud.free(); v5.free()


def test_multi_res_group_matches_cuda_v5_pwa():
    """n_res == 2, rs = 8 — the general (multi-BW) model with grouping."""
    kc = Fitter(_multires_dic(), backend="numpy_pwa").kernel_config
    data = _synth(kc, 480, seed=11)
    cud = _cuda(kc, 8)
    v5 = _v5(kc, rs=8)
    assert v5.kernel.n_res == 2
    params = _params(v5.kernel, seed=7)

    hc = cud.load_data(data)
    norm = float(cud.compute(params, hc, norm=None)[0])
    Qc, gc, Pc = cud.compute(params, hc, norm=norm)
    Q5, g5, P5 = v5.compute(params, v5.load_data(data), norm=norm)
    assert abs(Qc - Q5) <= 1e-6 * abs(Qc)
    assert _rel(Pc, P5) <= 1e-6
    for key in ("ck", "m0", "g0"):
        assert _rel(gc[key], g5[key]) <= 1e-6, key
    assert abs(cud.kernel._last_dnorm - v5.kernel._last_dnorm) \
        <= 1e-6 * abs(cud.kernel._last_dnorm)
    cud.free(); v5.free()


def test_rs1_semantics_vs_gpu_sibling():
    """Documents the ONE intentional CPU/GPU difference at rsize == 1.

    ``cuda_v5_pwa`` applies the group formula even for a one-event group
    (``−log(w·(P/norm+b))``); ``cpu_v5_pwa(rsize=1)`` is the v4 per-event
    objective (``−w·log(…)``) so that ``cpu_pwa_gen`` can alias it.  With
    unit weights the two coincide; with non-unit weights they differ by
    exactly ``−Σ log w`` in Q (and by ``w`` in dQ/dP).
    """
    kc = _kernel_config()
    data = _synth(kc, 416, seed=11)
    assert not np.all(data["weight"] == 1.0)
    cud = _cuda(kc, 1)
    v5 = _v5(kc, rs=1)
    params = _params(v5.kernel, seed=7, scale=0.5)

    hc = cud.load_data(data)
    norm = float(cud.compute(params, hc, norm=None)[0])
    Qc, _gc, Pc = cud.compute(params, hc, norm=norm)
    Q5, _g5, P5 = v5.compute(params, v5.load_data(data), norm=norm)
    assert _rel(Pc, P5) <= 1e-6                       # same P
    w, b = data["weight"], data["bkg"]
    q_grp1 = -float(np.sum(np.log(w * (P5 / norm + b))))
    assert Qc == pytest.approx(q_grp1, abs=1e-6 * abs(q_grp1))
    assert Q5 == pytest.approx(-float(np.sum(w * np.log(P5 / norm + b))),
                               abs=1e-10)

    # unit weights -> v4 == group-of-one, so the siblings agree
    ud = _unit(data)
    hc_u = cud.load_data(ud)
    Qc_u = cud.compute(params, hc_u, norm=norm)[0]
    Q5_u = v5.compute(params, v5.load_data(ud), norm=norm)[0]
    assert abs(Qc_u - Q5_u) <= 1e-6 * abs(Qc_u)
    cud.free(); v5.free()


# ── end-to-end Fitter FD (kernel + norm chain + dnorm) ────────────

@pytest.mark.parametrize("rs", [2, 8])
def test_fitter_fd_gradient(rs):
    kc = _kernel_config()
    # build a Fitter on the template config with the v5 backend
    f = Fitter(CFG, backend={"name": "cpu_v5_pwa", "resolution_size": rs})
    assert f.backend.kernel.resolution_size == rs
    data = _synth(kc, 96, seed=23)
    phsp = _synth(kc, 128, seed=29)
    f.set_phsp(phsp)
    f.set_data(data)

    x = f.initial_values(seed=5)
    nll0, gr0 = f.get_nll(x)
    assert np.isfinite(nll0)
    assert np.all(np.isfinite(gr0))
    assert np.isfinite(f.backend.kernel._last_dnorm)

    names = f.free_param_names()
    for i in range(0, len(x), max(1, len(x) // 6)):
        hh = 1e-6
        xp = x.copy(); xp[i] += hh
        xm = x.copy(); xm[i] -= hh
        fd = (f.get_nll(xp)[0] - f.get_nll(xm)[0]) / (2 * hh)
        assert abs(fd - gr0[i]) <= 1e-4 * max(abs(fd), 1.0), names[i]
    f.backend.free()


# ── misc ──────────────────────────────────────────────────────────

def test_return_p_false():
    kc = _kernel_config()
    gen, v5 = _gen(kc), _v5(kc, rs=8)
    data = _synth(kc, 128, seed=2)
    params = _params(gen.kernel, seed=5)
    h = v5.load_data(data)

    Q_full, g_full, P = v5.compute(params, h, norm=1.3, return_p=True)
    Q_fast, g_fast, P_none = v5.compute(params, h, norm=1.3, return_p=False)
    assert P is not None and P.shape == (128,)
    assert P_none is None
    assert Q_fast == Q_full
    for key in ("ck", "m0", "g0"):
        assert np.array_equal(g_fast[key], g_full[key])
    gen.free(); v5.free()
