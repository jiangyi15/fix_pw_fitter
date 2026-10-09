"""Tests for the FitResult base class (tabpwa.fit_result)."""

import json

import numpy as np
import pytest

from tabpwa import Fitter, FitResult


def test_from_x_defaults():
    r = FitResult.from_x([1.0, 2.0, 3.0])
    assert isinstance(r, FitResult)
    assert np.allclose(r.x, [1.0, 2.0, 3.0])
    assert r.fun is None and r.jac is None and r.hess_inv is None
    assert r.success is None and r.message == "No fit result"
    assert not r.is_empty and len(r) == 3
    assert "n=3" in repr(r) and "hess_inv=no" in repr(r)


def test_empty_result():
    r = FitResult()
    assert r.x is None and r.is_empty and len(r) == 0
    assert FitResult.from_x([]).is_empty


def test_extra_kwargs_stored_verbatim():
    r = FitResult.from_x([1.0], nll=3.5, chi2=1.2)
    assert r.nll == 3.5 and r.chi2 == 1.2


def test_from_result_copies_scipy_like_object():
    class Res:  # scipy OptimizeResult stand-in
        x = np.array([1.0, 2.0])
        fun = 0.5
        jac = np.array([1e-9, 1e-9])
        success = True
        message = "CONVERGENCE"
        hess_inv = np.eye(2)

    r = FitResult.from_result(Res())
    assert np.allclose(r.x, Res.x) and r.fun == 0.5
    assert r.success is True and r.message == "CONVERGENCE"
    assert np.allclose(r.hess_inv, np.eye(2))
    # a FitResult is passed through unchanged (unless extra overrides)
    assert FitResult.from_result(r) is r
    r2 = FitResult.from_result(r, fun=9.9)
    assert r2 is not r and r2.fun == 9.9 and np.allclose(r2.x, r.x)


def test_load_results_returns_fitresult(tmp_path):
    """load_results reconstructs a FitResult; save_params accepts it."""
    f = Fitter("tests/config_pwa.yml", backend="numpy_pwa")
    x0 = f.initial_values(seed=1)
    path = str(tmp_path / "res.json")

    f.save_params(x0, path)                       # flat-x save
    r = f.load_results(path)
    assert isinstance(r, FitResult)
    assert np.allclose(r.x, x0) and r.hess_inv is None

    # full FitResult (partial fields) round-trips identically to flat x
    path2 = str(tmp_path / "res2.json")
    f.save_params(FitResult.from_x(x0, fun=12.5), path2)
    a = json.load(open(path))
    b = json.load(open(path2))
    assert a["value"] == b["value"]
    assert b["status"]["NLL"] == pytest.approx(12.5)

    r2 = f.load_results(path2)
    assert isinstance(r2, FitResult)
    assert np.allclose(r2.x, x0)



def test_from_result_preserves_scipy_fields():
    """scipy OptimizeResult is dict-backed: extra fields survive."""
    from scipy.optimize import OptimizeResult
    res = OptimizeResult(x=[1.0, 2.0], fun=2.0, success=True, nit=7,
                         status=0, message="ok")
    r = FitResult.from_result(res)
    assert isinstance(r, FitResult)
    assert r.nit == 7 and r.status == 0 and r.message == "ok"
    assert np.allclose(r.x, [1.0, 2.0]) and r.fun == 2.0


def test_fit_returns_fitresult(tmp_path):
    """Fitter.fit() wraps the scipy result into a FitResult (fields kept)."""
    from tabpwa.pwa_build import build_tree_event_data, generate_pwa_phsp
    f = Fitter("tests/config_pwa.yml", backend="numpy_pwa")
    f.apply_constrains()
    model, kc = f.model, f.kernel_config
    tree = f.decay_tree
    chain = tree.partial_waves()[0][1]
    byt = {tree.topo_index[ch.topo_id()]: ch
           for _, ch in tree.partial_waves()}
    phsp = build_tree_event_data(
        model, kc, byt, generate_pwa_phsp(model, chain, 400, seed=51))
    data = build_tree_event_data(
        model, kc, byt, generate_pwa_phsp(model, chain, 400, seed=52))
    f.set_phsp(phsp)
    f.set_data(data)

    res = f.fit(f.initial_values(seed=53), maxiter=2, disp=False)
    assert isinstance(res, FitResult)
    assert res.x is not None and res.fun is not None
    assert res.hess_inv is not None            # BFGS inverse Hessian
    assert res.success is not None
    assert res.nit is not None                 # scipy extras preserved

    path = str(tmp_path / "fit.json")
    f.save_params(res, path)                   # saves like a scipy result
    r = f.load_results(path)
    assert isinstance(r, FitResult)
    assert np.allclose(r.x, res.x)
