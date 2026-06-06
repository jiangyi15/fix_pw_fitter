import numpy as np
import onnx
import onnxruntime as ort

from interp_fitter.kernel import Kernel
from interp_fitter.onnx_model import build_onnx_model


def _mini_config():
    n_int = 20
    return {
        "gamma_table": np.ones((1, n_int), dtype=complex),
        "fl_table": np.ones((1, n_int), dtype=float),
        "matrix_gamma": np.ones((1, 1), dtype=float),
        "matrix_ang": np.ones((1, 2), dtype=complex),
        "gamma_type":  np.array([0]),
        "gamma_index": np.array([0]),
        "gamma_min":   0.0,
        "gamma_delta": 0.1,
        "g0_index": np.array([0]),
        "m0_index":       np.array([0, 0]),
        "bw_index":       np.array([0, 0]),
        "bw_gamma_index": np.array([0, 0]),
        "bw_order":       np.array([0, 1]),
        "q_index":  np.array([0]),
        "fl_type":  np.array([0]),
        "fl_min":   0.0,
        "fl_delta": 0.1,
        "fl_order": np.array([0, 0]),
        "angle_index": np.array([0]),
        "angle_k":     np.array([1.0]),
        "angle_b":     np.array([0.0]),
        "ang_order":   np.array([[0]]),
    }


def test_model_build_and_check():
    config = _mini_config()
    model = build_onnx_model(config)
    onnx.checker.check_model(model)
    assert len(model.graph.node) > 0


def test_onnx_matches_numpy():
    config = _mini_config()
    nevt = 4

    # --- numpy reference ---
    k = Kernel(config)
    data = {
        "mass": np.ones((nevt, 1)) * 0.5,
        "q": np.ones((nevt, 1)) * 0.5,
        "angle": np.ones((nevt, 1)) * 0.5,
        "time": np.ones(nevt) * 0.1,
        "weight": np.ones(nevt),
        "frac": np.zeros(nevt),
        "bkg": np.zeros(nevt),
    }
    params = {
        "ck": np.array([1.0 + 0.3j, 0.7 - 0.2j]),
        "m0": np.array([1.5]),
        "g0": np.array([0.1]),
        "time_params": np.array([0.1, 0.02, 0.5, 0.8, 0.3, 0.0]),
    }
    P_ref, Q_ref, _ = k.compute(params, data, norm=1.0)

    # --- ONNX ---
    model = build_onnx_model(config)
    sess = ort.InferenceSession(model.SerializeToString())

    inputs = {
        "ck_re": params["ck"].real.astype(np.float32),
        "ck_im": params["ck"].imag.astype(np.float32),
        "m0": params["m0"].astype(np.float32),
        "g0": params["g0"].astype(np.float32),
        "time_params": params["time_params"].astype(np.float32),
        "mass": data["mass"].astype(np.float32),
        "q": data["q"].astype(np.float32),
        "angle": data["angle"].astype(np.float32),
        "time": data["time"].astype(np.float32),
        "weight": data["weight"].astype(np.float32),
        "frac": np.zeros(nevt, dtype=np.float32),
        "bkg": np.zeros(nevt, dtype=np.float32),
        "norm": np.array(1.0, dtype=np.float32),
    }
    P_onnx, Q_onnx = sess.run(["P", "Q"], inputs)

    np.testing.assert_allclose(P_onnx, P_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(Q_onnx, Q_ref, rtol=1e-5, atol=1e-5)


def test_onnx_no_norm():
    """Without norm, numpy Q = sum(w*P) and ONNX uses norm=1 internally."""
    config = _mini_config()
    nevt = 4

    k = Kernel(config)
    data = {
        "mass": np.ones((nevt, 1)) * 0.5,
        "q": np.ones((nevt, 1)) * 0.5,
        "angle": np.ones((nevt, 1)) * 0.5,
        "time": np.ones(nevt) * 0.1,
        "weight": np.ones(nevt),
        "frac": np.zeros(nevt),
        "bkg": np.zeros(nevt),
    }
    params = {
        "ck": np.array([1.0 + 0.3j, 0.7 - 0.2j]),
        "m0": np.array([1.5]),
        "g0": np.array([0.1]),
        "time_params": np.array([0.1, 0.02, 0.5, 0.8, 0.3, 0.0]),
    }
    P_ref, Q_ref, _ = k.compute(params, data, norm=None)

    model = build_onnx_model(config, with_norm=False)
    sess = ort.InferenceSession(model.SerializeToString())

    inputs = {
        "ck_re": params["ck"].real.astype(np.float32),
        "ck_im": params["ck"].imag.astype(np.float32),
        "m0": params["m0"].astype(np.float32),
        "g0": params["g0"].astype(np.float32),
        "time_params": params["time_params"].astype(np.float32),
        "mass": data["mass"].astype(np.float32),
        "q": data["q"].astype(np.float32),
        "angle": data["angle"].astype(np.float32),
        "time": data["time"].astype(np.float32),
        "weight": data["weight"].astype(np.float32),
        "frac": np.zeros(nevt, dtype=np.float32),
        "bkg": np.zeros(nevt, dtype=np.float32),
    }
    P_onnx, Q_onnx = sess.run(["P", "Q"], inputs)
    np.testing.assert_allclose(P_onnx, P_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(Q_onnx, Q_ref, rtol=1e-5, atol=1e-5)


def test_save_and_reload(tmp_path):
    config = _mini_config()
    model = build_onnx_model(config)
    path = tmp_path / "kernel.onnx"
    onnx.save(model, str(path))
    loaded = onnx.load(str(path))
    onnx.checker.check_model(loaded)
