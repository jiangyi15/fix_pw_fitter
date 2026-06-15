"""
Backend implementations for ampfit.

Each class is registered with ``@register_backend(name)`` and available
through ``create_backend(name, kernel_config, **kwargs)``.
"""
import numpy as np
from .core import ComputeBackend, register_backend, DataHandle


# ── NumPy backend ───────────────────────────────────────────────

@register_backend("numpy")
class NumpyBackend(ComputeBackend):
    """Pure NumPy computation (f64, CPU)."""
    def __init__(self, kernel_config):
        from ampfit.numpy_kernel import NumpyKernelCorrect
        self.kernel = NumpyKernelCorrect(kernel_config)

    def load_data(self, data_np):
        return data_np  # numpy dict is its own handle

    def compute(self, params, data_handle, norm=None):
        return self.kernel._compute(params, data_handle, norm=norm)


# ── CUDA base (original, f64/f32 via GPUArray) ─────────────────

@register_backend("cuda")
@register_backend("cuda64")
class CUDABackend(ComputeBackend):
    """CUDA-accelerated backend (float64)."""
    def __init__(self, kernel_config, dtype="float64"):
        self._dtype = np.float64 if dtype == "float64" else np.float32
        if dtype == "float32":
            from ampfit._cuda_f32 import CUDAKernel32 as _K
        else:
            from ampfit._cuda import CUDAKernel as _K
        self.kernel = _K(kernel_config)
        self._phsp_buffer = None
        self._phsp_scratch = None
        self._phsp_n = 0
        self._phsp_batch_size = 50000

    @property
    def dtype(self): return self._dtype

    def load_data(self, data_np):
        data = {k: (v.astype(self._dtype) if isinstance(v, np.ndarray) else v)
                for k, v in data_np.items()}
        return self.kernel.load_data(data)

    def compute(self, params, data_handle, norm=None):
        p = {}
        for k, v in params.items():
            if k == "ck":
                dt = np.complex64 if self._dtype == np.float32 else np.complex128
                p[k] = v.astype(dt)
            elif isinstance(v, np.ndarray):
                p[k] = v.astype(self._dtype)
            else:
                p[k] = v
        return self.kernel.compute(p, data_handle, norm=norm)

    def prepare_phsp_batched(self, phsp_np, n_events):
        is_f32 = self._dtype == np.float32
        if is_f32:
            from ampfit._cuda_f32 import GPUDataBuffer32 as GPUDataBuffer, GPUDataHolder32 as GPUDataHolder
        else:
            from ampfit._cuda import GPUDataBuffer, GPUDataHolder
        gc = self.kernel.gpu_config; lib = self.kernel.lib
        phsp = dict(phsp_np)
        w = phsp.get("weight", np.ones(phsp["mass"].shape[0]))
        ws = np.sum(w)
        if ws > 0: phsp["weight"] = (w / ws).astype(self._dtype)
        b = phsp.get("bkg", np.zeros(phsp["mass"].shape[0]))
        if np.isscalar(b): b = np.full(phsp["mass"].shape[0], b, dtype=self._dtype)
        ne = n_events
        self._phsp_buffer = GPUDataBuffer(lib, [
            ("mass", ((ne, phsp["mass"].shape[1]), self._dtype)),
            ("q", ((ne, phsp["q"].shape[1]), self._dtype)),
            ("angle", ((phsp["angle"].size,), self._dtype)),
            ("frac", ((ne,), self._dtype)), ("time", ((ne,), self._dtype)),
            ("weight", ((ne,), self._dtype)), ("bkg", ((ne,), self._dtype)),
        ])
        for key in ["mass", "q", "frac", "time", "weight"]:
            self._phsp_buffer.set(key, phsp[key].astype(self._dtype))
        self._phsp_buffer.set("angle", phsp["angle"].flatten().astype(self._dtype))
        self._phsp_buffer.set("bkg", b.astype(self._dtype))
        self._phsp_scratch = GPUDataHolder(lib, gc.n_wave, gc.n_unique_bw, gc.n_gamma_rows)
        self._phsp_scratch.n_mass = phsp["mass"].shape[1]
        self._phsp_scratch.n_momentum = phsp["q"].shape[1]
        self._phsp_scratch.alloc_intermediates(self._phsp_batch_size)
        self._phsp_n = ne

    def compute_norm_batched(self, params):
        if self._phsp_buffer is None:
            raise RuntimeError("No batched phsp prepared")
        total_norm = 0.0; total_grads = None
        bs = self._phsp_batch_size
        n_batches = (self._phsp_n + bs - 1) // bs
        for b in range(n_batches):
            start = b * bs; end = min(start + bs, self._phsp_n)
            self._phsp_scratch.attach_input_slice(
                self._phsp_buffer, start, end,
                self._phsp_scratch.n_mass, self._phsp_scratch.n_momentum)
            n_b, g_b, _ = self.kernel.compute(params, self._phsp_scratch, norm=None)
            total_norm += float(n_b)
            if total_grads is None:
                total_grads = {k: v.copy() for k, v in g_b.items()}
            else:
                for k in g_b: total_grads[k] += g_b[k]
        return total_norm, total_grads

    def free_phsp_batched(self):
        if self._phsp_scratch is not None: self._phsp_scratch.free(); self._phsp_scratch = None
        if self._phsp_buffer is not None: self._phsp_buffer.free(); self._phsp_buffer = None

    def free(self):
        self.free_phsp_batched(); self.kernel.free()


# ── CUDA v2 (C-managed memory) ─────────────────────────────────

@register_backend("cuda_v2")
@register_backend("cuda64_v2")
class CUDABackendV2(ComputeBackend):
    """v2 CUDA backend — C-managed memory, no GPUArray/GPUConfig."""
    def __init__(self, kernel_config, batch_size=50000):
        from ampfit._cuda_v2 import CUDAKernelV2 as _K
        self.kernel = _K(kernel_config, batch_size=batch_size)
        self._phsp_np = None; self._phsp_scratch = None; self._phsp_n = 0

    def load_data(self, data_np): return self.kernel.load_data(data_np)
    def compute(self, params, data_handle, norm=None):
        return self.kernel.compute(params, data_handle, norm=norm)

    def prepare_phsp_batched(self, phsp_np, n_events):
        self._phsp_np = phsp_np; self._phsp_n = n_events
        self._phsp_scratch = self.kernel.load_data(phsp_np)

    def compute_norm_batched(self, params):
        if self._phsp_scratch is None: raise RuntimeError("No phsp loaded")
        n_b, g_b, _ = self.kernel.compute(params, self._phsp_scratch, norm=None)
        return n_b, g_b

    def free_phsp_batched(self):
        self._phsp_np = None
        if self._phsp_scratch is not None: self._phsp_scratch.free(); self._phsp_scratch = None

    def free(self): self.free_phsp_batched(); self.kernel.free()


@register_backend("cuda32_v2")
class CUDABackendV2F32(ComputeBackend):
    """v2 CUDA backend — float32 version for 2-4x faster computation."""
    dtype = np.float32
    def __init__(self, kernel_config, batch_size=50000):
        from ampfit._cuda_v2_f32 import CUDAKernelV2F32 as _K
        self.kernel = _K(kernel_config, batch_size=batch_size)
        self._phsp_np = None; self._phsp_scratch = None; self._phsp_n = 0

    def load_data(self, data_np): return self.kernel.load_data(data_np)
    def compute(self, params, data_handle, norm=None):
        return self.kernel.compute(params, data_handle, norm=norm)

    def prepare_phsp_batched(self, phsp_np, n_events):
        self._phsp_np = phsp_np; self._phsp_n = n_events
        self._phsp_scratch = self.kernel.load_data(phsp_np)

    def compute_norm_batched(self, params):
        if self._phsp_scratch is None: raise RuntimeError("No phsp loaded")
        n_b, g_b, _ = self.kernel.compute(params, self._phsp_scratch, norm=None)
        return n_b, g_b

    def free_phsp_batched(self):
        self._phsp_np = None
        if self._phsp_scratch is not None: self._phsp_scratch.free(); self._phsp_scratch = None

    def free(self): self.free_phsp_batched(); self.kernel.free()


# ── CUDA v3 (Catmull-Rom) ──────────────────────────────────────

@register_backend("cuda_v3")
@register_backend("cuda64_v3")
class CUDABackendV3(ComputeBackend):
    """v3 CUDA backend — Catmull-Rom interpolation."""
    def __init__(self, kernel_config, batch_size=50000):
        from ampfit._cuda_v3 import CUDAKernelV3 as _K
        self.kernel = _K(kernel_config, batch_size=batch_size)
        self._phsp_np = None; self._phsp_scratch = None; self._phsp_n = 0

    def load_data(self, data_np): return self.kernel.load_data(data_np)
    def compute(self, params, data_handle, norm=None):
        return self.kernel.compute(params, data_handle, norm=norm)

    def prepare_phsp_batched(self, phsp_np, n_events):
        self._phsp_np = phsp_np; self._phsp_n = n_events
        self._phsp_scratch = self.kernel.load_data(phsp_np)

    def compute_norm_batched(self, params):
        if self._phsp_scratch is None: raise RuntimeError("No phsp loaded")
        n_b, g_b, _ = self.kernel.compute(params, self._phsp_scratch, norm=None)
        return n_b, g_b

    def free_phsp_batched(self):
        self._phsp_np = None
        if self._phsp_scratch is not None: self._phsp_scratch.free(); self._phsp_scratch = None

    def free(self): self.free_phsp_batched(); self.kernel.free()


@register_backend("cuda32_v3")
class CUDABackendV3F32(ComputeBackend):
    """v3 CUDA backend — float32 with Catmull-Rom interpolation."""
    dtype = np.float32
    def __init__(self, kernel_config, batch_size=50000):
        from ampfit._cuda_v3_f32 import CUDAKernelV3F32 as _K
        self.kernel = _K(kernel_config, batch_size=batch_size)
        self._phsp_np = None; self._phsp_scratch = None; self._phsp_n = 0

    def load_data(self, data_np): return self.kernel.load_data(data_np)
    def compute(self, params, data_handle, norm=None):
        return self.kernel.compute(params, data_handle, norm=norm)

    def prepare_phsp_batched(self, phsp_np, n_events):
        self._phsp_np = phsp_np; self._phsp_n = n_events
        self._phsp_scratch = self.kernel.load_data(phsp_np)

    def compute_norm_batched(self, params):
        if self._phsp_scratch is None: raise RuntimeError("No phsp loaded")
        n_b, g_b, _ = self.kernel.compute(params, self._phsp_scratch, norm=None)
        return n_b, g_b

    def free_phsp_batched(self):
        self._phsp_np = None
        if self._phsp_scratch is not None: self._phsp_scratch.free(); self._phsp_scratch = None

    def free(self): self.free_phsp_batched(); self.kernel.free()


# ── ONNX Runtime backend ───────────────────────────────────────

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
        if providers is None: providers = ['CPUExecutionProvider']
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
