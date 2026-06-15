"""ONNX Runtime backend — float32 CPU/GPU via onnxruntime."""
import numpy as np
from .core import ComputeBackend, register_backend


@register_backend("onnx")
@register_backend("onnx_cpu")
@register_backend("onnx_cuda")
class ONNXBackend(ComputeBackend):
    """ONNX Runtime backend (float32, CPU/GPU via onnxruntime)."""
    dtype = np.float32
    _SCALAR_NAMES = ["Gamma", "Delta_Gamma", "Delta_m", "A_prod", "poq_rho", "pop_phi"]
    _GRAD_MAP = {
        "grad_ck_real": "ck", "grad_ck_imag": "ck",
        "grad_m0": "m0", "grad_g0": "g0", "grad_scalar": "scalar",
    }

    def __init__(self, kernel_config=None, model_path=None,
                 norm_model_path="pwa_forward_norm.onnx",
                 batch_size=8192, providers=None):
        import onnxruntime as ort
        if providers is None:
            providers = ['CPUExecutionProvider']
        if kernel_config is not None:
            from ampfit._onnx_builder import PWAONNXBuilder
            builder = PWAONNXBuilder(kernel_config)
            forward_model = builder.build(batch_size=batch_size, norm_model=False)
            norm_model = builder.build(batch_size=batch_size, norm_model=True)
            self.sess = ort.InferenceSession(forward_model.SerializeToString(), providers=providers)
            self.sess_norm = ort.InferenceSession(norm_model.SerializeToString(), providers=providers)
        else:
            if model_path is None:
                raise ValueError("model_path required when kernel_config not provided")
            self.sess = ort.InferenceSession(model_path, providers=providers)
            self.sess_norm = ort.InferenceSession(norm_model_path, providers=providers)
        self._input_names = [i.name for i in self.sess.get_inputs()]
        self._output_names = [o.name for o in self.sess.get_outputs()]
        self._input_names_norm = [i.name for i in self.sess_norm.get_inputs()]
        self._onnx_batch = batch_size if kernel_config is not None else 0
        for inp in self.sess.get_inputs():
            if inp.name == "mass": self._onnx_batch = inp.shape[0]; break
        if self._onnx_batch is None or self._onnx_batch == 0:
            raise RuntimeError("Could not infer ONNX batch size")

    def load_data(self, data_np):
        return {k: (v.astype(np.float32) if isinstance(v, np.ndarray) else v)
                for k, v in data_np.items()}

    def _build_feed(self, input_names, data_slice, params, norm):
        feed = {}
        for name in input_names:
            if name == "ck_real":
                feed[name] = np.asarray(np.real(params.get("ck", 0)), dtype=np.float32)
            elif name == "ck_imag":
                feed[name] = np.asarray(np.imag(params.get("ck", 0)), dtype=np.float32)
            elif name in self._SCALAR_NAMES:
                scalar = params.get("scalar", np.zeros(6, dtype=np.float32))
                feed[name] = np.asarray([scalar[self._SCALAR_NAMES.index(name)]], dtype=np.float32)
            elif name in params:
                feed[name] = np.asarray(params[name], dtype=np.float32)
            elif name in data_slice:
                feed[name] = np.asarray(data_slice[name], dtype=np.float32)
            elif name == "norm":
                feed[name] = np.array([norm if norm is not None else 1.0], dtype=np.float32)
        return feed

    def _run_batch(self, sess, output_names, feed):
        return dict(zip(output_names, sess.run(output_names, feed)))

    def _extract_grads(self, result):
        grads = {}
        if "grad_ck_real" in result and "grad_ck_imag" in result:
            grads["ck"] = (result["grad_ck_real"].astype(np.complex64)
                           + 1j * result["grad_ck_imag"].astype(np.complex64))
        for onnx_name, key in self._GRAD_MAP.items():
            if onnx_name in result and key != "ck":
                grads[key] = result[onnx_name]
        return grads

    @staticmethod
    def _is_event_array(key, array, n_total):
        return isinstance(array, np.ndarray) and array.ndim >= 1 and array.shape[0] == n_total

    def compute(self, params, data_handle, norm=None):
        n_total = data_handle["mass"].shape[0]
        bs = self._onnx_batch; n_batches = (n_total + bs - 1) // bs
        sess = self.sess_norm if norm is None else self.sess
        input_names = self._input_names_norm if norm is None else self._input_names
        output_names = self._output_names
        total_Q = 0.0; total_grads = None; all_P = []
        for i in range(n_batches):
            start = i * bs; end = min(start + bs, n_total); n_valid = end - start
            batch_data = {}
            for k, v in data_handle.items():
                if self._is_event_array(k, v, n_total):
                    if n_valid == bs:
                        batch_data[k] = v[start:end]
                    else:
                        full = np.zeros((bs,) + v.shape[1:], dtype=v.dtype)
                        full[:n_valid] = v[start:end]
                        if k == "weight": full[n_valid:] = 0.0
                        batch_data[k] = full
                else:
                    batch_data[k] = v
            feed = self._build_feed(input_names, batch_data, params, norm)
            result = self._run_batch(sess, output_names, feed)
            P_batch = result.get("P", np.zeros(n_valid, dtype=np.float32))
            if norm is None:
                w = batch_data.get("weight", np.ones(bs, dtype=np.float32))
                total_Q += float(np.sum(w[:n_valid] * P_batch[:n_valid]))
            else:
                total_Q += float(result.get("Q", 0.0))
            grads = self._extract_grads(result)
            if total_grads is None:
                total_grads = {k: v.copy() for k, v in grads.items()}
            else:
                for k in grads: total_grads[k] += grads[k]
            all_P.append(P_batch[:n_valid])
        P_all = all_P[0] if len(all_P) == 1 else np.concatenate(all_P, axis=0)
        return total_Q, total_grads, P_all
