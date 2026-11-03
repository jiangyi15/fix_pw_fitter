"""cpu_pwa_gen — CPU (C + OpenMP) general pure-PWA backend (any n_res).

CPU counterpart of ``cuda_v4_pwa``: the projection-sum PWA with any number
of Breit–Wigner factors per chain.  Correctness anchors:

* ``n_res == 1``: **bit-identical** to the ``cpu_pwa_s0c`` CPU sibling
  (Q / P / ck / m0 / g0 / dnorm), on the fixed-shape ``config_pwa.yml``
  model and on the tiled ``synthetic_case`` layout (198 waves).
* ``n_res == 2``: FP64 agreement with the ``numpy_pwa`` reference
  (Q / P / ck / m0 / g0 / dnorm) — the only differences come from the
  BLAS reduction order in the reference's projection matvec — plus the
  ``cuda_v4_pwa`` general GPU sibling (fp32 amp cache → ~1e-7 floor),
  central-difference gradients of ``compute`` (Wirtinger convention), and
  an end-to-end ``Fitter`` FD check (kernel + norm chain + dnorm).
"""
import copy
import time

import numpy as np
import pytest
import yaml

from tabpwa import Fitter
from tabpwa.backends import create_backend, backends_for_model

from test_cpu_pwa_s0c import _dims, _params, _synth

CFG = "tests/config_pwa.yml"
CFG_BW = "tests/config_pwa_bw.yml"
NH = 1e-6
TOL64 = 1e-14          # fp64 CPU path vs reference


# ── helpers ────────────────────────────────────────────────────────

def _kernel_config(cfg=CFG):
    return Fitter(cfg, backend="numpy_pwa").kernel_config


def _multires_dic():
    """In-memory 2-resonance config (``n_res == 2``).

    ``tests/config_pwa_bw.yml`` with the top decay widened to three
    daughters so that TWO of them are sub-decay resonances:

        jpsi -> pipi + sigma2 + eta ,  pipi -> pip + pim ,
        sigma2 -> kp + km

    (all five finals distinct, so nothing depends on a symmetrisation
    declaration).  The multi-resonance config file is owned elsewhere —
    this fixture is built from the single-resonance template instead.
    ``n_res == 2`` is asserted by every user of this helper.
    """
    with open(CFG_BW) as fh:
        dic = yaml.safe_load(fh)
    d = copy.deepcopy(dic)
    d["decay"] = {
        "jpsi": [["pipi", "sigma2", "eta"]],
        "pipi": ["pip", "pim"],
        "sigma2": ["kp", "km"],
    }
    part = d["particle"]
    part["$finals"] = ["pip", "pim", "eta", "kp", "km"]
    part["pipi"] = ["MI1m"]                  # J=1, P=-1 -> pip+pim S-wave ok
    part["kp"] = {"J": 0, "P": -1, "mass": 0.493677}
    part["km"] = {"J": 0, "P": -1, "mass": 0.493677}
    part["sigma2"] = {"J": 0, "P": 1, "mass": 0.5, "width": 0.12}
    # geometry only — the events are supplied through set_data/set_phsp
    d["data"] = {"dat_order": ["pip", "pim", "eta", "kp", "km"],
                 "random_z": False, "center_mmass": True}
    return d


def _gen(kc):
    return create_backend("cpu_pwa_gen", kc, model="pwa")


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(a))),
                                             1e-300))


# ── registration / shape ───────────────────────────────────────────

def test_registration_and_alias():
    avail = backends_for_model("pwa")
    assert "cpu_pwa_gen" in avail
    assert "cpu_v4_pwa" in avail            # stacked CPU sibling alias
    from tabpwa.backends import backend_class
    assert backend_class("cpu_pwa_gen") is backend_class("cpu_v4_pwa")


def test_shapes_single_res():
    kc = _kernel_config()
    b = _gen(kc)
    k = b.kernel
    assert k.n_res == 1
    assert k.n_wave == kc["matrix_angle"].shape[1]
    assert k.n_proj == int(kc["n_proj"]) == 2
    assert k.n_wave_base * k.n_proj == k.n_wave
    assert k.n_fact == k.n_wave                    # one factor per wave
    assert k.slot_map.size == k.n_fact * k.n_basis
    b.free()


def test_shapes_multi_res():
    kc = Fitter(_multires_dic(), backend="numpy_pwa").kernel_config
    b = _gen(kc)
    k = b.kernel
    assert k.n_res == 2
    assert k.n_fact == 2 * k.n_wave
    assert k.slot_map.size == k.n_fact * k.n_basis
    # every (wave, factor) has its own gamma-row range
    assert k.row_start.shape == (k.n_fact,)
    assert k._g0_index_f.shape == (k.n_gamma_rows,)
    b.free()


# ── n_res == 1: bit-identical to cpu_pwa_s0c ───────────────────────

@pytest.mark.parametrize("norm", [None, 1.7])
def test_single_res_bit_identical_to_s0c(norm):
    kc = _kernel_config()
    gen, s0c = _gen(kc), create_backend("cpu_pwa_s0c", kc, model="pwa")
    data = _synth(kc, 400, seed=11)
    params = _params(gen.kernel, seed=7, scale=0.5)

    Q1, g1, P1 = gen.compute(params, gen.load_data(data), norm=norm)
    Q2, g2, P2 = s0c.compute(params, s0c.load_data(data), norm=norm)

    assert np.float64(Q1) == np.float64(Q2)
    assert np.array_equal(P1, P2)
    for key in ("ck", "m0", "g0"):
        assert np.array_equal(np.asarray(g1[key]), np.asarray(g2[key])), key
    if norm is not None:
        assert gen.kernel._last_dnorm == s0c.kernel._last_dnorm
    else:
        assert gen.kernel._last_dnorm is None

    gen.free()
    s0c.free()


def test_single_res_bit_identical_tiled_layout():
    """198 waves (``synthetic_case("config_pwa.yml", 100, 2, 400)``)."""
    from benchmark_pwa_kernels import synthetic_case

    kc, data, _phsp, params, norm, info = synthetic_case(CFG, 100, 2, 400)
    assert info["n_wave"] == 198 and info["N"] == 99
    gen, s0c = _gen(kc), create_backend("cpu_pwa_s0c", kc, model="pwa")
    assert gen.kernel.n_res == 1 and gen.kernel.n_res == s0c.kernel.n_res
    assert gen.kernel.n_basis == s0c.kernel.n_basis

    for nm in (None, norm):
        Q1, g1, P1 = gen.compute(params, gen.load_data(data), norm=nm)
        Q2, g2, P2 = s0c.compute(params, s0c.load_data(data), norm=nm)
        assert np.float64(Q1) == np.float64(Q2)
        assert np.array_equal(P1, P2)
        for key in ("ck", "m0", "g0"):
            assert np.array_equal(np.asarray(g1[key]),
                                  np.asarray(g2[key])), key
        if nm is not None:
            assert gen.kernel._last_dnorm == s0c.kernel._last_dnorm

    gen.free()
    s0c.free()


def test_return_p_false_multi_res():
    kc = Fitter(_multires_dic(), backend="numpy_pwa").kernel_config
    cpu = _gen(kc)
    k = cpu.kernel
    data = _synth(kc, 64, seed=2)
    params = _params(k, seed=5)
    h = cpu.load_data(data)

    Q_full, g_full, P = cpu.compute(params, h, norm=1.3, return_p=True)
    Q_fast, g_fast, P_none = cpu.compute(params, h, norm=1.3, return_p=False)
    assert P is not None and P.shape == (64,)
    assert P_none is None
    assert Q_fast == Q_full
    for key in ("ck", "m0", "g0"):
        assert np.array_equal(g_fast[key], g_full[key])
    cpu.free()


# ── n_res == 2: numpy_pwa oracle (fp64 CPU paths) ──────────────────

@pytest.mark.parametrize("norm", [None, 1.7])
def test_multi_res_matches_numpy_pwa(norm):
    kc = Fitter(_multires_dic(), backend="numpy_pwa").kernel_config
    gen = _gen(kc)
    ref = create_backend("numpy_pwa", kc, model="pwa")
    assert gen.kernel.n_res == 2
    data = _synth(kc, 500, seed=11)
    params = _params(gen.kernel, seed=7, scale=0.5)

    Q1, g1, P1 = gen.compute(params, gen.load_data(data), norm=norm)
    Q2, g2, P2 = ref.compute(params, ref.load_data(data), norm=norm)

    assert abs(Q1 - Q2) <= TOL64 * abs(Q2)
    assert _rel(P1, P2) <= TOL64
    for key in ("ck", "m0", "g0"):
        assert _rel(g1[key], g2[key]) <= TOL64, key
    if norm is not None:
        assert abs(gen.kernel._last_dnorm - ref.kernel._last_dnorm) \
            <= TOL64 * abs(ref.kernel._last_dnorm)
    else:
        assert gen.kernel._last_dnorm is None

    gen.free()


def test_multi_res_against_cuda_v4_pwa():
    """The general GPU sibling at n_res=2 (fp32 amp cache floor).

    Also documents the reference: ``cuda_v4_pwa`` itself vs ``numpy_pwa``
    on the same config — the GPU and the CPU backend must agree with each
    other to the same level as each agrees with the reference.
    """
    kc = Fitter(_multires_dic(), backend="numpy_pwa").kernel_config
    try:
        cud = create_backend("cuda_v4_pwa", kc, model="pwa")
    except Exception as e:                                # noqa: BLE001
        pytest.skip(f"cuda_v4_pwa unavailable: {e}")
    gen = _gen(kc)
    ref = create_backend("numpy_pwa", kc, model="pwa")
    data = _synth(kc, 500, seed=11)
    params = _params(gen.kernel, seed=7, scale=0.5)

    Qn, gn, Pn = ref.compute(params, ref.load_data(data), norm=1.7)
    Qc, gc, Pc = cud.compute(params, cud.load_data(data), norm=1.7)
    Qg, gg, Pg = gen.compute(params, gen.load_data(data), norm=1.7)

    # the general GPU kernel vs the FP64 oracle: fp32 amp cache floor
    assert abs(Qc - Qn) <= 1e-6 * abs(Qn)
    assert _rel(Pc, Pn) <= 1e-5
    for key in ("ck", "m0", "g0"):
        assert _rel(gc[key], gn[key]) <= 1e-5, key

    # cpu_pwa_gen vs the GPU sibling: same floor
    assert abs(Qg - Qc) <= 1e-6 * abs(Qc)
    assert _rel(Pg, Pc) <= 1e-5
    for key in ("ck", "m0", "g0"):
        assert _rel(gg[key], gc[key]) <= 1e-5, key

    gen.free()
    cud.free()


# ── n_res == 2: finite-difference gradients of the kernel ──────────

@pytest.mark.parametrize("norm", [None, 1.7])
def test_multi_res_fd_gradients(norm):
    kc = Fitter(_multires_dic(), backend="numpy_pwa").kernel_config
    cpu = _gen(kc)
    k = cpu.kernel
    data = _synth(kc, 60, seed=17)
    h = cpu.load_data(data)
    p = _params(k, seed=3, scale=0.4)

    def f(pp):
        return cpu.compute(pp, h, norm=norm)[0]

    Q, g, _ = cpu.compute(p, h, norm=norm)
    assert np.isfinite(Q)

    # m0 / g0 (real parameters)
    for key in ("m0", "g0"):
        for i in range(len(p[key])):
            q = np.array(p[key], float)
            q[i] += NH
            a = f({**p, key: q})
            q[i] -= 2 * NH
            b = f({**p, key: q})
            fd = (a - b) / (2 * NH)
            assert abs(fd - g[key][i]) <= 1e-4 * max(abs(fd), 1.0), (key, i)

    # ck: Wirtinger convention dQ/dRe = 2 Re(g), dQ/dIm = -2 Im(g)
    for i in range(k.n_wave_base):
        q = np.array(p["ck"], complex)
        q[i] += NH
        a = f({**p, "ck": q})
        q[i] -= 2 * NH
        b = f({**p, "ck": q})
        fd_re = (a - b) / (2 * NH)
        assert abs(fd_re - 2 * g["ck"][i].real) <= 1e-4 * max(abs(fd_re), 1.0)

        q = np.array(p["ck"], complex)
        q[i] += 1j * NH
        a = f({**p, "ck": q})
        q[i] -= 2j * NH
        b = f({**p, "ck": q})
        fd_im = (a - b) / (2 * NH)
        assert abs(fd_im + 2 * g["ck"][i].imag) <= 1e-4 * max(abs(fd_im), 1.0)

    cpu.free()


# ── n_res == 2: end-to-end Fitter FD (kernel + norm chain + dnorm) ──

def test_fitter_fd_gradient():
    from tabpwa.amp_model import build_amplitude_model

    dic = _multires_dic()
    kc = build_amplitude_model(dic).build_kernel_config()
    f = Fitter(dic, backend="cpu_pwa_gen")
    assert f.model.n_res == 2
    data = _synth(kc, 80, seed=23)
    phsp = _synth(kc, 120, seed=29)
    f.set_phsp(phsp)
    f.set_data(data)
    x = f.initial_values(seed=5)
    nll0, gr0 = f.get_nll(x)
    assert np.isfinite(nll0)
    assert np.all(np.isfinite(gr0))

    names = f.free_param_names()
    for i in range(0, len(x), max(1, len(x) // 5)):
        hh = 1e-6
        xp = x.copy(); xp[i] += hh
        xm = x.copy(); xm[i] -= hh
        fd = (f.get_nll(xp)[0] - f.get_nll(xm)[0]) / (2 * hh)
        assert abs(fd - gr0[i]) <= 1e-4 * max(abs(fd), 1.0), names[i]
    f.backend.free()


# ── informational benchmark (no failure on a slow number) ──────────

def test_benchmark_throughput(capsys):
    from benchmark_pwa_kernels import synthetic_case

    kc, data, _phsp, params, norm, _info = synthetic_case(CFG, 100, 2, 65536)
    rows = []
    for name in ("cpu_pwa_gen", "cpu_pwa_s0c"):
        b = create_backend(name, kc, model="pwa")
        h = b.load_data(data)
        b.compute(params, h, norm=norm)                   # warm
        best = min(_timeit(lambda: b.compute(params, h, norm=norm))
                   for _ in range(3))
        rows.append((name, best))
        h.free()
        b.free()
    with capsys.disabled():
        for name, dt in rows:
            print(f"{name}: n=65536  {dt*1e3:8.2f} ms  "
                  f"{65536/dt/1e6:7.2f} Mev/s")
    assert all(dt > 0 for _, dt in rows)


def _timeit(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def test_flat_angle_input_matches_3d():
    kc = Fitter(_multires_dic(), backend="numpy_pwa").kernel_config
    cpu = _gen(kc)
    data = _synth(kc, 24, seed=37)
    flat = {**data, "angle": data["angle"].reshape(24, -1)}
    p = _params(cpu.kernel, seed=11)
    q3, g3, _ = cpu.compute(p, cpu.load_data(data), norm=None)
    q2, g2, _ = cpu.compute(p, cpu.load_data(flat), norm=None)
    assert q2 == pytest.approx(q3, rel=1e-12)
    assert np.allclose(g2["ck"], g3["ck"], rtol=1e-12, atol=1e-14)
    cpu.free()
