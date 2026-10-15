"""polarised_pwa — spin-density model: registration, pwa identity, gradients."""
import numpy as np
import pytest
import yaml

from tabpwa import Fitter, load_fit
from tabpwa.spin_density import generators, polarisation_names, rho_matrix

CFG = "tests/config_pwa_bw.yml"


def _cfg(model="polarised_pwa"):
    cfg = yaml.safe_load(open(CFG))
    cfg["amp_model"] = model
    return cfg


def _synth(fitter, n, seed):
    kc = fitter.kernel_config
    rs = np.random.RandomState(seed)
    n_at = int(np.max(kc["angle_index"])) + 1 if len(kc["angle_index"]) else 1
    nac = int(np.asarray(kc["angle_k"]).shape[-1])
    n_mass = int(np.max(kc["mass_index"])) + 1
    n_q = int(np.max(kc["fl_q_index"])) + 1
    gmin = float(kc["gamma_min"])
    gspan = float(kc["gamma_delta"]) * np.asarray(kc["gamma_table"]).shape[-1]
    return {"mass": (gmin + 0.05) + (0.9 * gspan - 0.1) * rs.rand(n, n_mass),
            "q": np.abs(rs.randn(n, n_q)) + 0.3,
            "angle": rs.uniform(-np.pi, np.pi, size=(n, n_at * nac)),
            "weight": np.ones(n), "bkg": np.ones(n)}


def _put(fitter, n, seed):
    fitter.set_phsp(_synth(fitter, n, seed + 100))
    fitter.set_data(_synth(fitter, n, seed))


def test_model_and_params_registered():
    f = Fitter(_cfg(), backend="numpy_polarised")
    assert f.model.name == "polarised_pwa"
    names = f.model.scalar_names
    assert names[:3] == ["jpsi_px", "jpsi_py", "jpsi_pz"]   # top prefix
    assert len(names) == f.model._n_j() ** 2 - 1            # trace-P basis
    free = f.free_param_names()
    for n in names:
        assert n in free                                    # ordinary params


def test_zero_polarisation_matches_pwa():
    f_pwa = Fitter(CFG, backend="numpy_pwa")
    f_pol = Fitter(_cfg(), backend="numpy_polarised")
    _put(f_pwa, 80, 1)
    _put(f_pol, 80, 1)

    xA = f_pwa.initial_values(seed=1)
    xB = f_pol.initial_values(seed=1)[:len(xA)].copy()   # same prefix order
    nA, gA = f_pwa.get_nll(xA)
    nB, gB = f_pol.get_nll(xB)
    assert nB == pytest.approx(nA, rel=1e-12)
    assert np.allclose(gB[:len(xA)], gA, atol=1e-9)
    # spin gradients are NOT zero at rho = I: they measure the intensity
    # sensitivity to polarisation (helicity interference) — the physics
    # this model adds on top of the plain pwa
    assert np.any(np.abs(gB[len(xA):]) > 0.0)


def test_spin_gradients_match_fd():
    f = Fitter(_cfg(), backend="numpy_polarised")
    _put(f, 100, 2)
    x = f.initial_values(seed=2)
    free = f.free_param_names()
    n, g = f.get_nll(x)
    eps = 1e-6
    for name in ("jpsi_px", "jpsi_py", "jpsi_pz"):
        idx = free.index(name)
        xp, xm = x.copy(), x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        num = (f.get_nll(xp)[0] - f.get_nll(xm)[0]) / (2 * eps)
        assert num == pytest.approx(g[idx], rel=1e-4, abs=1e-6), name


def test_default_backend_is_cuda():
    """polarised_pwa's default backend is the CUDA one — never a silent
    numpy fallback (the numpy reference stays explicit-only)."""
    from tabpwa.backends.cuda_backends import CUDAPolarisedBackend
    try:
        f = Fitter(_cfg())
    except (RuntimeError, OSError) as e:
        pytest.skip(f"no usable CUDA device/library: {e}")
    assert isinstance(f.backend, CUDAPolarisedBackend)


def test_polarisation_scan_identifiability():
    """The NLL responds to the polarisation: with distinct data/phsp event
    sets the px direction is identifiable (non-degenerate scan)."""
    cfg = _cfg()
    f_true = Fitter(cfg, backend="numpy_polarised")
    _put(f_true, 400, 1)
    x0 = f_true.initial_values(seed=5)
    free = f_true.free_param_names()
    ipx = free.index("jpsi_px")

    nlls = []
    for px in np.linspace(0.0, 0.8, 9):
        x = x0.copy()
        x[ipx] = px
        nlls.append(f_true.get_nll(x)[0])
    # the params are visible: the NLL actually responds
    assert np.ptp(nlls) > 1.0


def test_load_fit_dispatch_polarised(tmp_path):
    f = load_fit(_cfg(), backend="numpy_polarised", load_data=False)
    assert isinstance(f, Fitter)
    assert f.model.name == "polarised_pwa"


def test_generators_traceless_hermitian():
    for P in (2, 3):
        gen = generators(P)
        assert len(gen) == P * P - 1
        for g in gen:
            assert np.allclose(g, g.conj().T)            # Hermitian
            assert abs(np.trace(g).real) < 1e-12         # traceless
        assert rho_matrix(np.zeros(P * P - 1), P) == pytest.approx(
            np.eye(P))
