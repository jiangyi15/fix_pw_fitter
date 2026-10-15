"""GPU regression: cuda_polarised — projection-sum PWA + spin-density matrix.

Run alone — shared-GPU multi-file segfault, pre-existing.

The standalone ``cuda_polarised`` kernel mirrors ``NumpyPolarisedPWA``
(``numpy_pwa.py``, the FD-green reference): same ampcache/BW/m0/g0 chains
and shared-ck forward as ``cuda_v4_pwa`` (n_wave = n_proj·N, ck length N,
NO config duplication), but the reduce contracts the per-projection
amplitudes with the spin-density matrix

    P(e) = Re Σ_{p,p'} rho[p,p']·A_p·conj(A_p')

with ``rho = I + Σ_l scalar[l]·G_l`` built HOST-side per call from the
polarisation parameters (the generator basis is a context constant).  The
scalar gradients are computed IN the reduce kernel and returned finished —
no raw dQ/dρ anywhere in the returned dicts.

Checks:
1. CUDA backend == NumpyPolarisedPWA (NLL + ck/m0/g0/scalar gradients) at
   nonzero polarisation;
2. scalar = 0  ⇒  exactly the plain ``NumpyPWA`` projection sum;
3. central finite difference of one polarisation parameter (jpsi_px)
   through the backend matches grads["scalar"];
4. full Fitter end-to-end vs the numpy backend (NLL + flat gradient).
"""

import numpy as np
import pytest
import yaml

from tabpwa.amp_model import build_amplitude_model
from tabpwa.config_loader import RawConfig


@pytest.fixture(scope="module")
def pol_fixture():
    rc = RawConfig("tests/config_pwa_bw.yml")
    rc.dic["amp_model"] = "polarised_pwa"
    model = build_amplitude_model(rc.dic)
    kc = model.build_kernel_config()          # ORIGINAL config: n_proj = P,
    P = int(kc["n_proj"])                     # n_wave = P·N — no expansion
    N = kc["matrix_angle"].shape[1] // P

    n_m0 = int(np.max(kc["m0_index"])) + 1
    n_g0 = int(np.max(kc["g0_index"])) + 1
    n_mass = int(np.max(kc["mass_index"])) + 1
    n_mom = int(np.max(kc["fl_q_index"])) + 1
    n_at = int(np.max(kc["angle_index"])) + 1   # canonical 3D angle layout
    n_comp = int(np.asarray(kc["angle_k"]).shape[-1])

    rs = np.random.RandomState(0)
    ne = 80
    data = {
        "mass": 2.0 + 0.6 * rs.rand(ne, n_mass),
        "q": 0.3 + np.abs(rs.randn(ne, n_mom)),
        "angle": rs.uniform(-np.pi, np.pi, (ne, n_at, n_comp)),
        "weight": np.ones(ne),
        "bkg": np.ones(ne),
    }
    ck = rs.randn(N) + 1j * rs.randn(N)
    m0 = 0.5 + 0.4 * np.abs(rs.randn(n_m0))
    g0 = 0.1 + 0.3 * np.abs(rs.randn(n_g0))
    scalar = np.array([0.2, -0.1, 0.15])      # nonzero polarisation
    return kc, P, N, data, {"ck": ck, "m0": m0, "g0": g0, "scalar": scalar}


def _make_backend(kc):
    from tabpwa.backends.cuda_backends import CUDAPolarisedBackend
    try:
        return CUDAPolarisedBackend(kc, batch_size=16)
    except (RuntimeError, OSError) as e:
        pytest.skip(f"no usable CUDA device/library: {e}")


def test_polarised_matches_numpy_reference(pol_fixture):
    from tabpwa.numpy_pwa import NumpyPolarisedPWA

    kc, P, N, data, params = pol_fixture

    be = _make_backend(kc)
    try:
        h = be.load_data(data)
        Q, g, Pe = be.compute(params, h, norm=None)
        assert Pe is not None
        assert len(g["ck"]) == N
        assert len(g["scalar"]) == P * P - 1
        # no raw dQ/drho may leak out of the backend
        assert "drho" not in g
    finally:
        h.free()
        be.free()

    npw = NumpyPolarisedPWA(kc)
    h2 = npw.load_data(data)
    Qr, gr, _Pr = npw.compute(params, h2, norm=None)

    # fp64 CUDA vs the fp64 numpy reference; the fp32 angular cache gives
    # ~1e-7 relative noise (same convention as the other GPU tests here).
    assert Q == pytest.approx(Qr, rel=1e-6)
    assert np.allclose(g["ck"], gr["ck"], rtol=1e-5,
                       atol=1e-6 * max(1e-12, np.max(np.abs(gr["ck"]))))
    assert np.allclose(g["m0"], gr["m0"], rtol=1e-5,
                       atol=1e-6 * max(1e-12, np.max(np.abs(gr["m0"]))))
    assert np.allclose(g["g0"], gr["g0"], rtol=1e-5,
                       atol=1e-6 * max(1e-12, np.max(np.abs(gr["g0"]))))
    scale_s = max(1e-12, np.max(np.abs(gr["scalar"])))
    assert np.allclose(g["scalar"], gr["scalar"], rtol=1e-5,
                       atol=1e-6 * scale_s)


def test_zero_spin_matches_pwa(pol_fixture):
    from tabpwa.numpy_pwa import NumpyPWA

    kc, _P, N, data, params = pol_fixture

    be = _make_backend(kc)
    try:
        h = be.load_data(data)
        zero = dict(params, scalar=np.zeros(len(params["scalar"])))
        Q, g, Pe = be.compute(zero, h, norm=None)
        assert np.all(np.isfinite(g["scalar"]))
    finally:
        h.free()
        be.free()

    npz = NumpyPWA(kc)
    h2 = npz.load_data(data)
    Qz, gz, _Pz = npz.compute({"ck": params["ck"], "m0": params["m0"],
                               "g0": params["g0"]}, h2, norm=None)
    assert Q == pytest.approx(Qz, rel=1e-6)


def test_scalar_fd(pol_fixture):
    """Central FD of one scalar entry (jpsi_px) through the backend."""
    kc, _P, _N, data, params = pol_fixture

    be = _make_backend(kc)
    try:
        h = be.load_data(data)

        def qf(scalar):
            p = dict(params, scalar=scalar)
            return be.compute(p, h, norm=None)[0]

        scalar = np.asarray(params["scalar"], dtype=float).copy()
        Q0, g, _Pe = be.compute(dict(params), h, norm=None)

        eps = 1e-6
        s = scalar.copy()
        s[0] += eps                       # jpsi_px is the first generator
        qp = qf(s)
        s[0] -= 2 * eps
        qm = qf(s)
        fd = (qp - qm) / (2 * eps)
        assert qp != Q0 or qm != Q0       # the NLL moves with jpsi_px
        assert fd == pytest.approx(g["scalar"][0], rel=1e-4)
    finally:
        h.free()
        be.free()


def test_fitter_end_to_end():
    """Fitter(backend="cuda_polarised") == Fitter(backend="numpy_polarised")
    end-to-end: NLL + full flat gradient (including the spin entries) at
    nonzero polarisation, canonical 3D-angle synthetic data."""
    from tabpwa import Fitter

    cfg = yaml.safe_load(open("tests/config_pwa_bw.yml"))
    cfg["amp_model"] = "polarised_pwa"

    try:
        f_np = Fitter(dict(cfg), backend="numpy_polarised")
        f_gpu = Fitter(dict(cfg), backend="cuda_polarised")
    except (RuntimeError, OSError) as e:
        pytest.skip(f"no usable CUDA device/library: {e}")

    def _synth(fitter, n, seed):
        kc = fitter.kernel_config
        rs = np.random.RandomState(seed)
        n_at = (int(np.max(kc["angle_index"])) + 1
                if len(kc["angle_index"]) else 1)
        nac = int(np.asarray(kc["angle_k"]).shape[-1])
        n_mass = int(np.max(kc["mass_index"])) + 1
        n_q = int(np.max(kc["fl_q_index"])) + 1
        gmin = float(kc["gamma_min"])
        gspan = (float(kc["gamma_delta"])
                 * np.asarray(kc["gamma_table"]).shape[-1])
        # canonical pipeline layout (pwa_build.build_tree_event_data):
        # angle (n, n_topo, n_base) — the CUDA loader requires this shape
        return {"mass": (gmin + 0.05) + (0.9 * gspan - 0.1) * rs.rand(n, n_mass),
                "q": np.abs(rs.randn(n, n_q)) + 0.3,
                "angle": rs.uniform(-np.pi, np.pi, size=(n, n_at, nac)),
                "weight": np.ones(n), "bkg": np.ones(n)}

    for f in (f_np, f_gpu):
        f.set_phsp(_synth(f, 80, 101))
        f.set_data(_synth(f, 80, 1))

    x = f_np.initial_values(seed=3)
    free = f_np.free_param_names()
    for name in ("jpsi_px", "jpsi_py", "jpsi_pz"):   # nonzero polarisation
        x[free.index(name)] = 0.2

    n1, g1 = f_np.get_nll(x)
    n2, g2 = f_gpu.get_nll(x)

    # the CUDA kernel's fp32 angular cache gives ~1e-7 relative noise
    assert n2 == pytest.approx(n1, rel=1e-6)
    scale = max(1e-12, np.max(np.abs(g1)))
    assert np.allclose(g2, g1, rtol=1e-4, atol=1e-6 * scale)
