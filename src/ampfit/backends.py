"""
Compute backends for ampfit Fitter.

Each backend wraps a kernel implementation with a uniform interface:
  - load_data(data_dict) → data_handle
  - compute(params, data_handle, norm) → (Q, grads, P)
  - free()

Usage:
    from ampfit.backends import CUDABackend, NumpyBackend, ONNXBackend
    fitter = Fitter("config.yml", backend=CUDABackend("float64"))
"""
import numpy as np


class DataHandle:
    """Opaque handle for data loaded on a backend."""
    pass


# ── Base class ──────────────────────────────────────────────────

class ComputeBackend:
    """Abstract compute backend.

    Subclasses must implement:
      load_data(self, data_np) -> DataHandle
      compute(self, params, data_handle, norm) -> (Q, grads_dict, P)
      free(self)
    """
    dtype = np.float64

    def load_data(self, data_np):
        raise NotImplementedError

    def compute(self, params, data_handle, norm=None):
        raise NotImplementedError

    def free(self):
        pass


# ── NumPy backend ───────────────────────────────────────────────

class NumpyBackend(ComputeBackend):
    """Pure NumPy computation (f64, CPU)."""

    def __init__(self, kernel_config):
        from ampfit.numpy_kernel import NumpyKernelCorrect
        self.kernel = NumpyKernelCorrect(kernel_config)

    def load_data(self, data_np):
        return data_np  # numpy dict is its own handle

    def compute(self, params, data_handle, norm=None):
        return self.kernel._compute(params, data_handle, norm=norm)


# ── CUDA backends ───────────────────────────────────────────────

class CUDABackend(ComputeBackend):
    """CUDA-accelerated backend (float64 by default)."""

    def __init__(self, kernel_config, dtype="float64"):
        self.dtype = np.float64 if dtype == "float64" else np.float32
        if dtype == "float32":
            from ampfit._cuda_f32 import CUDAKernel32 as _K
        else:
            from ampfit._cuda import CUDAKernel as _K
        self._kernel_class = _K
        self.kernel = _K(kernel_config)
        self._phsp_buffer = None
        self._phsp_scratch = None
        self._phsp_n = 0
        self._phsp_batch_size = 50000

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, val):
        self._dtype = val

    def load_data(self, data_np):
        # Cast to backend dtype
        data = {k: (v.astype(self._dtype) if isinstance(v, np.ndarray) else v)
                for k, v in data_np.items()}
        return self.kernel.load_data(data)

    def compute(self, params, data_handle, norm=None):
        # Cast params to backend dtype
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
        """Set up batched phsp computation for large datasets."""
        from ampfit._cuda import GPUDataBuffer, GPUDataHolder
        gc = self.kernel.gpu_config
        lib = self.kernel.lib

        # Normalize weights
        phsp = dict(phsp_np)
        w = phsp.get("weight", np.ones(phsp["mass"].shape[0]))
        ws = np.sum(w)
        if ws > 0:
            phsp["weight"] = (w / ws).astype(self._dtype)
        b = phsp.get("bkg", np.zeros(phsp["mass"].shape[0]))
        if np.isscalar(b):
            b = np.full(phsp["mass"].shape[0], b, dtype=self._dtype)

        ne = n_events
        self._phsp_buffer = GPUDataBuffer(lib, [
            ("mass",   ((ne, phsp["mass"].shape[1]), self._dtype)),
            ("q",      ((ne, phsp["q"].shape[1]), self._dtype)),
            ("angle",  ((phsp["angle"].size,), self._dtype)),
            ("frac",   ((ne,), self._dtype)),
            ("time",   ((ne,), self._dtype)),
            ("weight", ((ne,), self._dtype)),
            ("bkg",    ((ne,), self._dtype)),
        ])
        for key in ["mass", "q", "frac", "time", "weight"]:
            self._phsp_buffer.set(key, phsp[key].astype(self._dtype))
        self._phsp_buffer.set("angle", phsp["angle"].flatten().astype(self._dtype))
        self._phsp_buffer.set("bkg", b.astype(self._dtype))

        self._phsp_scratch = GPUDataHolder(lib, gc.n_wave, gc.n_unique_bw, gc.n_gamma_rows)
        self._phsp_scratch.alloc_intermediates(self._phsp_batch_size)
        self._phsp_n = ne

    def compute_norm_batched(self, params):
        """Compute norm over batched phsp."""
        if self._phsp_buffer is None:
            raise RuntimeError("No batched phsp prepared")
        total_norm = 0.0
        total_grads = None
        bs = self._phsp_batch_size
        n_batches = (self._phsp_n + bs - 1) // bs
        gc = self.kernel.gpu_config

        for b in range(n_batches):
            start = b * bs
            end = min(start + bs, self._phsp_n)
            self._phsp_scratch.attach_input_slice(
                self._phsp_buffer, start, end,
                None, None)
            n_b, g_b, _ = self.kernel.compute(params, self._phsp_scratch, norm=None)
            total_norm += float(n_b)
            if total_grads is None:
                total_grads = {k: v.copy() for k, v in g_b.items()}
            else:
                for k in g_b:
                    total_grads[k] += g_b[k]
        return total_norm, total_grads

    def free_phsp_batched(self):
        if self._phsp_scratch is not None:
            self._phsp_scratch.free()
            self._phsp_scratch = None
        if self._phsp_buffer is not None:
            self._phsp_buffer.free()
            self._phsp_buffer = None

    def free(self):
        self.free_phsp_batched()
        self.kernel.free()


# ── ONNX Runtime backend ───────────────────────────────────────

class ONNXBackend(ComputeBackend):
    """ONNX Runtime backend (float32, CPU/GPU via onnxruntime)."""

    def __init__(self, model_path="pwa_forward.onnx", providers=None):
        import onnxruntime as ort
        if providers is None:
            providers = ['CPUExecutionProvider']
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self._input_names = [i.name for i in self.sess.get_inputs()]
        self._output_names = [o.name for o in self.sess.get_outputs()]

    def load_data(self, data_np):
        return data_np

    def compute(self, params, data_handle, norm=None):
        feed = {}
        for name in self._input_names:
            if name in params:
                v = params[name]
                feed[name] = np.asarray(v, dtype=np.float32)
            elif name in data_handle:
                v = data_handle[name]
                feed[name] = np.asarray(v, dtype=np.float32)
            elif name == "norm":
                feed[name] = np.array(norm or 1.0, dtype=np.float32)
        outs = self.sess.run(self._output_names, feed)
        result = dict(zip(self._output_names, outs))
        Q = result.get("Q", 0.0)
        P = result.get("P", np.zeros(0))
        # Build grads dict from available outputs
        grads = {}
        for gname in ["grad_ck_real", "grad_ck_imag", "grad_m0",
                       "grad_g0", "grad_scalar"]:
            if gname in result:
                grads[gname] = result[gname]
        if "grad_ck_real" in grads and "grad_ck_imag" in grads:
            grads["ck"] = (grads["grad_ck_real"].astype(np.complex64)
                           + 1j * grads["grad_ck_imag"].astype(np.complex64))
        return Q, grads, P
