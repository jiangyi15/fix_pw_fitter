"""cpu_pwa_s0c — CPU (C + OpenMP) pure-PWA backend with s0c pole params.

The C kernel works in ``s0 = m0²`` / ``mult = m0·g0`` (like
``cuda_v4_pwa_s0c``) and the wrapper chains the gradients back to m0/g0.
Correctness anchors:

* FP64 agreement with the ``numpy_pwa`` reference (Q / P / ck / m0 / g0 /
  dnorm) — the only differences come from the BLAS reduction order in the
  reference's projection matvec.
* central-difference gradients of ``compute`` (Wirtinger convention:
  ``∂Q/∂Re ck = 2 Re(g)``, ``∂Q/∂Im ck = −2 Im(g)``).
* an end-to-end ``Fitter`` FD check (kernel + norm chain + dnorm).
"""
import time

import numpy as np
import pytest

from tabpwa import Fitter
from tabpwa.backends import create_backend, backends_for_model

CFG = "tests/config_pwa.yml"
NH = 1e-6


# ── helpers ────────────────────────────────────────────────────────

def _kernel_config():
    return Fitter(CFG, backend="numpy_pwa").kernel_config


def _dims(kc):
    n_at = (int(np.max(kc["angle_index"])) + 1
            if len(kc["angle_index"]) else 1)
    nac = int(np.asarray(kc["angle_k"]).shape[-1])
    n_mass = int(np.max(kc["mass_index"])) + 1
    n_q = int(np.max(kc["fl_q_index"])) + 1
    return n_at, nac, n_mass, n_q


def _synth(kc, n, seed=3):
    """Kernel-shaped synthetic event data (mass/q/angle/weight/bkg)."""
    n_at, nac, n_mass, n_q = _dims(kc)
    rs = np.random.RandomState(seed)
    gmin = float(kc["gamma_min"])
    gspan = float(kc["gamma_delta"]) * np.asarray(kc["gamma_table"]).shape[-1]
    return {
        "mass": (gmin + 0.05) + (0.9 * gspan - 0.1) * rs.rand(n, n_mass),
        "q": np.abs(rs.randn(n, n_q)) + 0.3,
        "angle": rs.uniform(-np.pi, np.pi, size=(n, n_at, nac)),
        "weight": 0.5 + rs.rand(n),
        "bkg": np.abs(rs.randn(n)) * 0.1 + 0.05,
    }


def _params(k, seed=7, scale=0.5):
    rs = np.random.RandomState(seed)
    n0 = k.n_m0_params
    g0 = np.linspace(0.06, 0.30, n0) * (1.0 + 0.1 * rs.randn(n0))
    m0 = np.linspace(0.45, 1.35, n0)
    return {
        "ck": scale * (rs.randn(k.n_wave_base) + 1j * rs.randn(k.n_wave_base)),
        "m0": m0,
        "g0": g0,
    }


# ── registration / shape ───────────────────────────────────────────

def test_registration_and_shapes():
    assert "cpu_pwa_s0c" in backends_for_model("pwa")
    kc = _kernel_config()
    b = create_backend("cpu_pwa_s0c", kc, model="pwa")
    k = b.kernel
    assert k.n_wave == kc["matrix_angle"].shape[1]
    assert k.n_proj == int(kc["n_proj"]) == 2          # n_proj from config
    assert k.n_wave_base * k.n_proj == k.n_wave
    assert k.n_res == 1
    assert k.row_start.shape == (k.n_wave,)
    assert k._g0_index_w.shape == (k.n_gamma_rows,)
    b.free()


# ── agreement with numpy_pwa ───────────────────────────────────────

@pytest.mark.parametrize("norm", [None, 1.7])
def test_matches_numpy_pwa(norm):
    kc = _kernel_config()
    cpu = create_backend("cpu_pwa_s0c", kc, model="pwa")
    ref = create_backend("numpy_pwa", kc, model="pwa")
    k = cpu.kernel
    data = _synth(kc, 400, seed=11)
    params = _params(k)

    Q1, g1, P1 = cpu.compute(params, cpu.load_data(data), norm=norm)
    Q2, g2, P2 = ref.compute(params, ref.load_data(data), norm=norm)

    assert np.allclose(Q1, Q2, rtol=1e-10, atol=1e-8)
    assert np.allclose(P1, P2, rtol=1e-10, atol=1e-6)
    for key in ("ck", "m0", "g0"):
        assert np.allclose(g1[key], g2[key], rtol=1e-10, atol=1e-6), key
    if norm is not None:
        assert np.allclose(k._last_dnorm, ref.kernel._last_dnorm, rtol=1e-10)
    else:
        assert k._last_dnorm is None

    cpu.free()


def test_return_p_false():
    kc = _kernel_config()
    cpu = create_backend("cpu_pwa_s0c", kc, model="pwa")
    k = cpu.kernel
    data = _synth(kc, 64, seed=2)
    params = _params(k)
    h = cpu.load_data(data)

    Q_full, g_full, P = cpu.compute(params, h, norm=1.3, return_p=True)
    Q_fast, g_fast, P_none = cpu.compute(params, h, norm=1.3, return_p=False)
    assert P is not None and P.shape == (64,)
    assert P_none is None
    assert Q_fast == Q_full
    for key in ("ck", "m0", "g0"):
        assert np.array_equal(g_fast[key], g_full[key])
    cpu.free()


# ── finite-difference gradients of the kernel ──────────────────────

@pytest.mark.parametrize("norm", [None, 1.7])
def test_fd_gradients(norm):
    kc = _kernel_config()
    cpu = create_backend("cpu_pwa_s0c", kc, model="pwa")
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


# ── end-to-end Fitter FD (kernel + norm chain + dnorm) ─────────────

def test_fitter_fd_gradient():
    kc = _kernel_config()
    f = Fitter(CFG, backend="cpu_pwa_s0c")
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
        h = 1e-6
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        fd = (f.get_nll(xp)[0] - f.get_nll(xm)[0]) / (2 * h)
        assert abs(fd - gr0[i]) <= 1e-4 * max(abs(fd), 1.0), names[i]
    f.backend.free()


# ── benchmark (informational: no failure on a slow number) ─────────

def test_benchmark_throughput(capsys):
    kc = _kernel_config()
    cpu = create_backend("cpu_pwa_s0c", kc, model="pwa")
    k = cpu.kernel
    params = _params(k)

    cuda = None
    try:
        cuda = create_backend("cuda_v4_pwa_s0c", kc, model="pwa")
    except Exception:                                     # noqa: BLE001
        cuda = None

    rows = []
    for n in (65536, 262144):
        data = _synth(kc, n, seed=31)
        h = cpu.load_data(data)
        cpu.compute(params, h, norm=1.0)                  # warm
        t0 = time.perf_counter()
        cpu.compute(params, h, norm=1.0)
        dt = time.perf_counter() - t0

        cuda_dt = None
        if cuda is not None:
            hc = cuda.load_data(data)
            cuda.compute(params, hc, norm=1.0)            # warm
            t1 = time.perf_counter()
            cuda.compute(params, hc, norm=1.0)
            cuda_dt = time.perf_counter() - t1
            hc.free()

        rows.append((n, dt, cuda_dt))
        ratio = (dt / cuda_dt) if cuda_dt else float("nan")
        with capsys.disabled():
            print(f"cpu_pwa_s0c: n={n:>7d}  {dt*1e3:8.2f} ms  "
                  f"{n/dt/1e6:7.2f} Mev/s"
                  + (f"   cuda {cuda_dt*1e3:7.2f} ms  ratio {ratio:5.2f}x"
                     if cuda_dt else "   (cuda unavailable)"))
        h.free()

    # informational only — correctness is asserted elsewhere
    assert all(dt > 0 for _, dt, _ in rows)
    cpu.free()
    if cuda is not None:
        cuda.free()


def test_flat_angle_input_matches_3d():
    """A flattened (ne, n_at*n_ac) angle input is reshaped correctly."""
    kc = _kernel_config()
    cpu = create_backend("cpu_pwa_s0c", kc, model="pwa")
    k = cpu.kernel
    data = _synth(kc, 24, seed=37)
    flat = {**data, "angle": data["angle"].reshape(24, -1)}
    p = _params(k, seed=11)
    q3, g3, _ = cpu.compute(p, cpu.load_data(data), norm=None)
    q2, g2, _ = cpu.compute(p, cpu.load_data(flat), norm=None)
    assert q2 == pytest.approx(q3, rel=1e-12)
    assert np.allclose(g2["ck"], g3["ck"], rtol=1e-12, atol=1e-14)
    cpu.free()


# ── opaque basis: a simple ONE-slot basis (nothing to do with s0) ────

def test_simple_basis_single_slot():
    """``n_basis = 1``: one basis function per wave, coefficient not s0.

    The C kernel must not assume anything about a coefficient, so a model
    whose denominator is ``coeff_w·F_w(e)`` (one basis function per wave,
    no s0, no −i rotation inside the kernel) has to run through the same
    code path.  Checked against a direct NumPy evaluation and FD gradients
    of both ``coeff`` and ``ck``.
    """
    from tabpwa.cpu import _pwa_s0c as mod

    lib, ffi = mod._load_lib(), mod._ffi
    ne, nw, P, nf = 37, 8, 2, 1        # nf = 1 -> one basis slot per wave
    N = nw // P
    rs = np.random.RandomState(41)
    F = rs.randn(ne, nw) + 1j * rs.randn(ne, nw)     # basis functions
    amp = rs.randn(ne, nw) + 1j * rs.randn(ne, nw)
    ck = 0.3 * (rs.randn(N) + 1j * rs.randn(N))
    coeff = 0.5 + rs.rand(ne if False else nw * nf)     # arbitrary numbers
    wgt = 0.5 + rs.rand(ne)
    bkg = 0.05 + 0.1 * rs.rand(ne)
    slot_map = np.arange(nw * nf, dtype=np.int32)       # identity: u == slot
    norm = 1.3

    keep = []

    def _db(a):
        arr = np.ascontiguousarray(a, np.float64)
        buf = ffi.from_buffer(arr)
        keep.append((arr, buf))
        return ffi.cast("double*", buf)

    def _ib(a):
        arr = np.ascontiguousarray(a, np.int32)
        buf = ffi.from_buffer(arr)
        keep.append((arr, buf))
        return ffi.cast("int*", buf)

    # tables are event-contiguous: [wave][event] / [slot][event]
    h = lib.cpu_pwa_s0c_load(
        _db(amp.real.T), _db(amp.imag.T), _db(F.real.T), _db(F.imag.T),
        _ib(slot_map), nw, _db(wgt), _db(bkg), ne,
        nw, N, P, nf)

    def run(co, cc=ck, use_norm=True):
        gc = np.zeros(nw * nf, np.float64)
        gcr = np.zeros(N, np.float64)
        gci = np.zeros(N, np.float64)
        dn = ffi.new("double*")
        q = lib.cpu_pwa_s0c_compute(
            h, _db(co), _db(cc.real), _db(cc.imag),
            norm if use_norm else 0.0, 1 if use_norm else 0,
            _db(gc), _db(gcr), _db(gci), ffi.NULL, dn)
        return q, gc, gcr + 1j * gci, float(dn[0])

    # ── reference: BW = coeff_w·F_w ─────────────────────────────────
    def ref_q(co):
        bw = co[None, :] * F
        common = amp / bw
        A = common.reshape(ne, P, N) @ ck
        Pv = np.sum(np.abs(A) ** 2, axis=1)
        return float(np.sum(wgt * np.log(Pv / norm + bkg) * -1.0))

    Q, gco, gck, _ = run(coeff)
    assert Q == pytest.approx(ref_q(coeff), rel=1e-12)

    # FD on the plain coefficients (no s0, no m0/g0 anywhere)
    for i in range(nw * nf):
        c = coeff.copy()
        c[i] += NH
        a = run(c)[0]
        c[i] -= 2 * NH
        b = run(c)[0]
        fd = (a - b) / (2 * NH)
        assert abs(fd - gco[i]) <= 1e-4 * max(abs(fd), 1.0), i

    # FD on ck (Wirtinger convention)
    for k in range(N):
        c = np.array(ck, complex)
        c[k] += NH
        a = run(coeff, c)[0]
        c[k] -= 2 * NH
        b = run(coeff, c)[0]
        fd = (a - b) / (2 * NH)
        assert abs(fd - 2 * gck[k].real) <= 1e-4 * max(abs(fd), 1.0)

    lib.cpu_pwa_s0c_free(h)
    keep.clear()
