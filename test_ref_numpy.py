import numpy as np
import pytest

from ref_numpy import Kernel


def make_mini_kernel():
    """Build a minimal Kernel with simple preset values.

    Architecture:
      nwaves=2, n_m0=1, n_g0=1, n_gamma=1, n_bw=2, n_fl=1,
      n_ang=1, nbasis=1, nres=1, ndecays=1
    """
    k = Kernel.__new__(Kernel)
    nevt = 10
    n_int = 20

    # Interpolation tables
    k.gamma_table = np.ones((1, n_int), dtype=complex)
    k.gamma_type = np.array([0])
    k.gamma_min = 0.0
    k.gamma_delta = 0.1

    k.fl_table = np.ones((1, n_int), dtype=float)
    k.fl_type = np.array([0])
    k.fl_min = 0.0
    k.fl_delta = 0.1

    # Mapping matrices
    k.matrix_gamma = np.ones((1, 1), dtype=float)       # (n_m0=1, n_gamma=1)
    k.matrix_ang = np.ones((1, 2), dtype=complex)       # (nbasis=1, nwaves=2)

    # Indexers
    k.g0_index = np.array([0])                           # (n_gamma=1,)
    k.gamma_index = np.array([0])                        # (n_gamma=1,)
    k.m0_index = np.array([0, 0])                        # (n_bw=2,)
    k.bw_index = np.array([0, 0])                        # (n_bw=2,)
    k.bw_gamma_index = np.array([0, 0])                  # (n_bw=2,)
    k.bw_order = np.array([0, 1])                        # (nwaves*nres=2,)
    k.q_index = np.array([0])                            # (n_fl=1,)
    k.fl_order = np.array([0, 0])                        # (nwaves*ndecays=2,)
    k.angle_index = np.array([0])                        # (n_ang=1,)
    k.angle_k = np.array([1.0])                          # (n_ang=1,)
    k.angle_b = np.array([0.0])                          # (n_ang=1,)
    k.ang_order = np.array([[0]])                        # (nbasis=1, n_per=1)

    return k, nevt


def simple_data(nevt):
    return {
        "mass": np.ones((nevt, 2)) * 0.5,       # need ndim_mass >= max(gamma_index,bw_index)+1
        "q": np.ones((nevt, 1)) * 0.5,
        "angle": np.ones((nevt, 1)) * 0.5,
        "time": np.ones(nevt) * 0.1,
        "weight": np.ones(nevt),
        "frac": 0.0,
    }


def simple_params():
    return {
        "ck": np.array([1.0 + 0.3j, 0.7 - 0.2j]),
        "m0": np.array([1.5]),
        "g0": np.array([0.1]),
        "time_params": np.array([0.1, 0.02, 0.5, 0.8, 0.3, 0.0]),
    }


# ============================================================
#  Interpolation
# ============================================================

class TestInterpolation:
    def test_exact_bin(self):
        table = np.array([[0.0, 1.0, 4.0, 9.0]], dtype=float)
        x = np.array([0.0, 0.3, 0.6])
        types = np.zeros(3, dtype=int)
        result = Kernel.interp(x, table, types, 0.0, 0.3)
        expected = np.array([0.0, 1.0, 4.0])
        np.testing.assert_allclose(result, expected)

    def test_mid_bin(self):
        table = np.array([[0.0, 2.0, 4.0]], dtype=float)
        x = np.array([0.15])
        types = np.zeros(1, dtype=int)
        result = Kernel.interp(x, table, types, 0.0, 0.3)
        # xbin=0, frac=0.5 -> left=0.0, right=2.0 -> 1.0
        np.testing.assert_allclose(result, [1.0])

    def test_multiple_types(self):
        table = np.array([
            [0.0, 10.0, 20.0],
            [100.0, 110.0, 120.0],
        ], dtype=float)
        x = np.array([0.15, 0.45])
        types = np.array([0, 1])
        result = Kernel.interp(x, table, types, 0.0, 0.3)
        # type 0: xbin=0, frac=0.5 -> left=0.0, right=10.0 -> 5.0
        # type 1: xbin=1, frac=0.5 -> left=110.0, right=120.0 -> 115.0
        np.testing.assert_allclose(result, [5.0, 115.0])

    def test_complex_table(self):
        table = np.array([[1.0+2.0j, 3.0+4.0j, 5.0+6.0j]], dtype=complex)
        x = np.array([0.15])
        types = np.zeros(1, dtype=int)
        result = Kernel.interp(x, table, types, 0.0, 0.3)
        expected = (table[0, 1] + table[0, 0]) / 2
        np.testing.assert_allclose(result, [expected])


# ============================================================
#  Forward
# ============================================================

class TestForward:
    def test_basic_no_norm(self):
        k, nevt = make_mini_kernel()
        data = simple_data(nevt)
        params = simple_params()
        P, Q, grads = k.compute(params, data)

        assert P.shape == (nevt,)
        assert np.all(P >= 0)
        assert Q > 0

    def test_with_norm(self):
        k, nevt = make_mini_kernel()
        data = simple_data(nevt)
        params = simple_params()
        norm = 5.0
        P, Q, grads = k.compute(params, data, norm=norm)

        assert "norm" in grads
        assert P.shape == (nevt,)

    def test_no_norm_derivative_dP(self):
        """Without norm Q = Σ w·P, gradient check dQ/dP = weight."""
        k, nevt = make_mini_kernel()
        data = simple_data(nevt)
        params = simple_params()
        P, Q, grads = k.compute(params, data)
        np.testing.assert_allclose(grads["m0"].shape, (1,))
        np.testing.assert_allclose(grads["g0"].shape, (1,))


# ============================================================
#  Gradient — numerical verification
# ============================================================

class TestGradientsNumerical:
    eps = 1e-6

    def _q_for_params(self, k, data, params, norm):
        _, Q, _ = k.compute(params, data, norm=norm)
        return Q

    def _num_grad_real(self, k, data, params, key, idx, norm=None, eps=1e-6):
        """Central-difference ∂Q/∂params[key][idx] for a real parameter."""
        p0 = params[key][idx]
        params_up = params.copy()
        params_down = params.copy()
        params_up[key] = params[key].copy()
        params_down[key] = params[key].copy()
        params_up[key][idx] = p0 + eps
        params_down[key][idx] = p0 - eps
        Q_up = self._q_for_params(k, data, params_up, norm)
        Q_down = self._q_for_params(k, data, params_down, norm)
        return (Q_up - Q_down) / (2 * eps)

    def _num_grad_complex(self, k, data, params, key, idx, norm=None, eps=1e-6):
        """Finite-difference ∂Q/∂Re and ∂Q/∂Im for complex param, return complex gradient."""
        c0 = params[key][idx]
        g_re = self._num_grad_real(k, data, params, key, idx, norm, eps)
        # For imag part we need to perturb Im(ck) by epsilon
        p_imag = params.copy()
        p_imag[key] = params[key].copy()
        p_imag[key][idx] = c0 + 1j * eps
        Q_up = self._q_for_params(k, data, p_imag, norm)
        p_imag[key][idx] = c0 - 1j * eps
        Q_down = self._q_for_params(k, data, p_imag, norm)
        g_im = (Q_up - Q_down) / (2 * eps)
        return g_re + 1j * g_im

    def _num_grad_time(self, k, data, params, tp_idx, norm=None, eps=1e-6):
        p_up = params.copy()
        p_down = params.copy()
        p_up["time_params"] = params["time_params"].copy()
        p_down["time_params"] = params["time_params"].copy()
        p_up["time_params"][tp_idx] += eps
        p_down["time_params"][tp_idx] -= eps
        Q_up = self._q_for_params(k, data, p_up, norm)
        Q_down = self._q_for_params(k, data, p_down, norm)
        return (Q_up - Q_down) / (2 * eps)

    def _num_grad_norm(self, k, data, params, norm, eps=1e-6):
        Q_up = self._q_for_params(k, data, params, norm + eps)
        Q_down = self._q_for_params(k, data, params, norm - eps)
        return (Q_up - Q_down) / (2 * eps)

    # --- Tests: no norm ---

    def test_ck_no_norm(self):
        k, nevt = make_mini_kernel()
        data = simple_data(nevt)
        params = simple_params()
        _, _, grads = k.compute(params, data)
        for i in range(len(params["ck"])):
            num = self._num_grad_complex(k, data, params, "ck", i)
            # grads["ck"][i] should be ∂Q/∂Re + i*∂Q/∂Im
            np.testing.assert_allclose(grads["ck"][i].real, num.real, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(grads["ck"][i].imag, num.imag, rtol=1e-5, atol=1e-5)

    def test_m0_no_norm(self):
        k, nevt = make_mini_kernel()
        data = simple_data(nevt)
        params = simple_params()
        _, _, grads = k.compute(params, data)
        for i in range(len(params["m0"])):
            num = self._num_grad_real(k, data, params, "m0", i)
            np.testing.assert_allclose(grads["m0"][i], num, rtol=1e-5, atol=1e-5)

    def test_g0_no_norm(self):
        k, nevt = make_mini_kernel()
        data = simple_data(nevt)
        params = simple_params()
        _, _, grads = k.compute(params, data)
        for i in range(len(params["g0"])):
            num = self._num_grad_real(k, data, params, "g0", i)
            np.testing.assert_allclose(grads["g0"][i], num, rtol=1e-5, atol=1e-5)

    def test_time_params_no_norm(self):
        k, nevt = make_mini_kernel()
        data = simple_data(nevt)
        params = simple_params()
        _, _, grads = k.compute(params, data)
        names = ["gamma", "dg", "dm", "poqr", "poqi", "ap"]
        for i in range(6):
            num = self._num_grad_time(k, data, params, i)
            np.testing.assert_allclose(grads["time_params"][i], num,
                                       rtol=1e-5, atol=1e-5,
                                       err_msg=f"Mismatch for time_params[{names[i]}]")

    # --- Tests: with norm ---

    def test_ck_with_norm(self):
        k, nevt = make_mini_kernel()
        data = simple_data(nevt)
        params = simple_params()
        norm = 5.0
        _, _, grads = k.compute(params, data, norm=norm)
        for i in range(len(params["ck"])):
            num = self._num_grad_complex(k, data, params, "ck", i, norm=norm)
            np.testing.assert_allclose(grads["ck"][i].real, num.real, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(grads["ck"][i].imag, num.imag, rtol=1e-5, atol=1e-5)

    def test_m0_with_norm(self):
        k, nevt = make_mini_kernel()
        data = simple_data(nevt)
        params = simple_params()
        norm = 5.0
        _, _, grads = k.compute(params, data, norm=norm)
        for i in range(len(params["m0"])):
            num = self._num_grad_real(k, data, params, "m0", i, norm=norm)
            np.testing.assert_allclose(grads["m0"][i], num, rtol=1e-5, atol=1e-5)

    def test_g0_with_norm(self):
        k, nevt = make_mini_kernel()
        data = simple_data(nevt)
        params = simple_params()
        norm = 5.0
        _, _, grads = k.compute(params, data, norm=norm)
        for i in range(len(params["g0"])):
            num = self._num_grad_real(k, data, params, "g0", i, norm=norm)
            np.testing.assert_allclose(grads["g0"][i], num, rtol=1e-5, atol=1e-5)

    def test_norm_gradient(self):
        k, nevt = make_mini_kernel()
        data = simple_data(nevt)
        params = simple_params()
        norm = 5.0
        _, _, grads = k.compute(params, data, norm=norm)
        num = self._num_grad_norm(k, data, params, norm)
        np.testing.assert_allclose(grads["norm"], num, rtol=1e-5, atol=1e-5)

    def test_time_params_with_norm(self):
        k, nevt = make_mini_kernel()
        data = simple_data(nevt)
        params = simple_params()
        norm = 5.0
        _, _, grads = k.compute(params, data, norm=norm)
        for i in range(6):
            num = self._num_grad_time(k, data, params, i, norm=norm)
            np.testing.assert_allclose(grads["time_params"][i], num,
                                       rtol=1e-5, atol=1e-5)

    # --- More realistic config ---

    def test_complex_config(self):
        """Test with non-trivial BW, gamma, form factor, and angular structure."""
        k = Kernel.__new__(Kernel)
        nevt = 30
        n_int = 50

        k.gamma_min = 0.0
        k.gamma_delta = 0.05
        k.fl_min = 0.0
        k.fl_delta = 0.05

        np.random.seed(7)
        k.gamma_table = np.random.randn(2, n_int) + 1j * np.random.randn(2, n_int)
        k.fl_table = np.random.rand(1, n_int)
        k.gamma_type = np.array([0, 0])                # (n_gamma=2,)
        k.fl_type = np.array([0])                       # (n_fl=1,)

        k.matrix_gamma = np.array([[0.8, 0.2],
                                   [0.3, 0.7]], dtype=float)  # (n_m0=2, n_gamma=2)
        # nwaves must be EVEN for reshape(nevt, 2, -1).sum(-1)
        k.matrix_ang = np.array([[1.0, 0.0, 0.5, 0.1],
                                 [0.2, 1.0, 0.0, 0.3]], dtype=complex)  # (nbasis=2, nwaves=4)

        k.g0_index = np.array([0, 1])                   # (n_gamma=2,)
        k.gamma_index = np.array([0, 0])                # (n_gamma=2,)

        k.m0_index = np.array([0, 0, 1, 1])             # (n_bw=4,)
        k.bw_index = np.array([0, 0, 1, 1])             # (n_bw=4,)
        k.bw_gamma_index = np.array([0, 0, 1, 1])       # (n_bw=4,)

        # nwaves=4, nres=1 → bw_order length = 4
        k.bw_order = np.array([0, 1, 2, 3])              # (nwaves*nres=4,)

        k.q_index = np.array([0])                        # (n_fl=1,)
        # nwaves=4, ndecays=1 → fl_order length = 4
        k.fl_order = np.array([0, 0, 0, 0])              # (nwaves*ndecays=4,)

        k.angle_index = np.array([0, 1])
        k.angle_k = np.array([1.0, 2.0])
        k.angle_b = np.array([0.0, 0.5])
        k.ang_order = np.array([[0], [1]])               # (nbasis=2, n_per=1)

        data = {
            "mass": np.column_stack([
                np.random.uniform(0.1, 2.0, nevt),
                np.random.uniform(0.1, 2.0, nevt),
            ]),
            "q": np.random.uniform(0.1, 2.0, (nevt, 1)),
            "angle": np.column_stack([
                np.random.uniform(-np.pi, np.pi, nevt),
                np.random.uniform(-np.pi, np.pi, nevt),
            ]),
            "time": np.random.uniform(0.0, 10.0, nevt),
            "weight": np.ones(nevt),
            "frac": 0.3,
        }

        params = {
            "ck": np.array([0.8 + 0.5j, 0.3 - 0.1j, 0.6 + 0.2j, 0.1 - 0.4j]),
            "m0": np.array([1.2, 1.8]),
            "g0": np.array([0.15, 0.10]),
            "time_params": np.array([0.05, 0.01, 0.3, 0.9, 0.2, 0.05]),
        }

        norm = 3.0

        P, Q_no, _ = k.compute(params, data)
        assert P.shape == (nevt,)
        assert np.all(P >= 0)

        P2, Q_no2, _ = k.compute(params, data)
        np.testing.assert_allclose(P, P2)
        np.testing.assert_allclose(Q_no, Q_no2)

        _, Q_norm, _ = k.compute(params, data, norm=norm)
        assert Q_norm != Q_no

    def test_gradient_symmetry_real(self):
        """Check that dQ/d(Re ck) and dQ/d(Im ck) give the correct
        directional derivative."""
        k, nevt = make_mini_kernel()
        data = simple_data(nevt)
        params = simple_params()

        _, Q0, grads = k.compute(params, data)

        eps = 1e-6
        for i in range(len(params["ck"])):
            # Perturb along real axis
            p_r = params.copy()
            p_r["ck"] = params["ck"].copy()
            p_r["ck"][i] += eps
            _, Qr, _ = k.compute(p_r, data)
            dQ_dRe_num = (Qr - Q0) / eps
            np.testing.assert_allclose(grads["ck"][i].real, dQ_dRe_num,
                                       rtol=1e-5, atol=1e-5)

            # Perturb along imag axis
            p_i = params.copy()
            p_i["ck"] = params["ck"].copy()
            p_i["ck"][i] += 1j * eps
            _, Qi, _ = k.compute(p_i, data)
            dQ_dIm_num = (Qi - Q0) / eps
            np.testing.assert_allclose(grads["ck"][i].imag, dQ_dIm_num,
                                       rtol=1e-5, atol=1e-5)

    def test_no_norm_consistent(self):
        """Without norm, Q = Σ w·P, so dQ/dP = weight."""
        k, nevt = make_mini_kernel()
        data = simple_data(nevt)
        params = simple_params()
        P, Q, grads = k.compute(params, data)
        np.testing.assert_allclose(Q, np.sum(data["weight"] * P))

    def test_norm_consistent(self):
        """With norm, Q = -Σ w·log(P/norm + bkg)."""
        k, nevt = make_mini_kernel()
        data = simple_data(nevt)
        params = simple_params()
        norm = 5.0
        P, Q, _ = k.compute(params, data, norm=norm)
        expected = -np.sum(data["weight"] * np.log(P / norm + data.get("bkg", 0)))
        np.testing.assert_allclose(Q, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
