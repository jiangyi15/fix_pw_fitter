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
import onnx


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
    """ONNX Runtime backend (float32, CPU/GPU via onnxruntime).

    Uses **two** ONNX models under the hood:

    * ``pwa_forward_norm.onnx`` — the *norm model* — computes
      ``Q = sum(P * weight)`` with correct norm gradients.  Used when the
      caller passes ``norm=None``.
    * ``pwa_forward.onnx`` — the *forward model* — computes the full negative
      log-likelihood ``Q = -Σ w·log(P/norm + bkg)`` with NLL gradients.  Used
      when the caller passes a numeric ``norm`` value.

    Handles arbitrarily large datasets by splitting into fixed-size batches
    matching the ONNX model's static batch dimension (8192 by default).  The
    final (partial) batch is padded with zeros and masked via ``weight=0``,
    which correctly contributes nothing to the loss or gradients.
    """

    _SCALAR_NAMES = ["Gamma", "Delta_Gamma", "Delta_m",
                     "A_prod", "poq_rho", "pop_phi"]

    _GRAD_MAP = {
        "grad_ck_real": "ck",
        "grad_ck_imag": "ck",
        "grad_m0": "m0",
        "grad_g0": "g0",
        "grad_scalar": "scalar",
    }

    def __init__(self, kernel_config=None, model_path=None,
                 norm_model_path="pwa_forward_norm.onnx",
                 batch_size=8192, providers=None):
        """
        Parameters
        ----------
        kernel_config : dict, optional
            Kernel configuration dict (from Config.build_all_index()).
            If provided, builds both models in memory instead of loading
            from file.
        model_path : str, optional
            Path to the forward ONNX model (NLL + NLL gradients).
            Required when kernel_config is not provided.
        norm_model_path : str
            Path to the norm ONNX model (sum(P*weight) + norm gradients).
        batch_size : int
            Batch size for in-memory model building (default 8192).
        providers : list of str, optional
            ONNX Runtime execution providers (default: CPU only).
        """
        import onnxruntime as ort
        if providers is None:
            providers = ['CPUExecutionProvider']

        if kernel_config is not None:
            # Build both models in memory from kernel_config
            from ampfit._onnx_builder import PWAONNXBuilder
            builder = PWAONNXBuilder(kernel_config)

            forward_model = builder.build(batch_size=batch_size,
                                          norm_model=False)
            norm_model = builder.build(batch_size=batch_size,
                                       norm_model=True)

            self.sess = ort.InferenceSession(
                forward_model.SerializeToString(), providers=providers)
            self.sess_norm = ort.InferenceSession(
                norm_model.SerializeToString(), providers=providers)
        else:
            # Backward compat: load from file paths
            if model_path is None:
                raise ValueError(
                    "model_path is required when kernel_config is not provided"
                )
            self.sess = ort.InferenceSession(model_path, providers=providers)
            self.sess_norm = ort.InferenceSession(norm_model_path,
                                                  providers=providers)

        self._input_names = [i.name for i in self.sess.get_inputs()]
        self._output_names = [o.name for o in self.sess.get_outputs()]
        self._input_names_norm = [i.name for i in self.sess_norm.get_inputs()]

        # Infer batch size from either model (they should agree)
        self._onnx_batch = batch_size if kernel_config is not None else 0
        for inp in self.sess.get_inputs():
            if inp.name == "mass":
                self._onnx_batch = inp.shape[0]
                break
        if self._onnx_batch is None or self._onnx_batch == 0:
            raise RuntimeError(
                "Could not infer ONNX batch size from model inputs"
            )

    def load_data(self, data_np):
        """Store data, casting event arrays to float32 for the ONNX model."""
        return {k: (v.astype(np.float32) if isinstance(v, np.ndarray) else v)
                for k, v in data_np.items()}

    # ── helpers ─────────────────────────────────────────────────

    def _build_feed(self, input_names, data_slice, params, norm):
        """Build an ONNX feed dict for one batch.

        Parameters
        ----------
        input_names : list of str
            Expected input names (from the session being used).
        data_slice : dict
            Sliced event data for this batch (already float32 from load_data).
        params : dict
            Fit parameters (ck, m0, g0, scalar).
        norm : float or None
            Normalisation value (only passed when the model expects it).
        """
        feed = {}
        for name in input_names:
            if name == "ck_real":
                feed[name] = np.asarray(np.real(params.get("ck", 0)),
                                        dtype=np.float32)
            elif name == "ck_imag":
                feed[name] = np.asarray(np.imag(params.get("ck", 0)),
                                        dtype=np.float32)
            elif name in self._SCALAR_NAMES:
                scalar = params.get("scalar", np.zeros(6, dtype=np.float32))
                idx = self._SCALAR_NAMES.index(name)
                feed[name] = np.asarray([scalar[idx]], dtype=np.float32)
            elif name in params:
                feed[name] = np.asarray(params[name], dtype=np.float32)
            elif name in data_slice:
                feed[name] = np.asarray(data_slice[name], dtype=np.float32)
            elif name == "norm":
                feed[name] = np.array([norm if norm is not None else 1.0],
                                      dtype=np.float32)
        return feed

    def _run_batch(self, sess, output_names, feed):
        outs = sess.run(output_names, feed)
        return dict(zip(output_names, outs))

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
        return (isinstance(array, np.ndarray)
                and array.ndim >= 1
                and array.shape[0] == n_total)

    # ── main compute ────────────────────────────────────────────

    def compute(self, params, data_handle, norm=None):
        n_total = data_handle["mass"].shape[0]
        bs = self._onnx_batch
        n_batches = (n_total + bs - 1) // bs

        # Select which model session to use
        if norm is None:
            # Norm model: Q = sum(P*weight), proper norm gradients
            sess = self.sess_norm
            input_names = self._input_names_norm
            output_names = self._output_names  # same outputs
        else:
            # Forward model: NLL + NLL gradients
            sess = self.sess
            input_names = self._input_names
            output_names = self._output_names

        total_Q = 0.0
        total_grads = None
        all_P = []

        for i in range(n_batches):
            start = i * bs
            end = min(start + bs, n_total)
            n_valid = end - start

            # Slice / pad event-level arrays
            batch_data = {}
            for k, v in data_handle.items():
                if self._is_event_array(k, v, n_total):
                    if n_valid == bs:
                        batch_data[k] = v[start:end]
                    else:
                        full = np.zeros((bs,) + v.shape[1:], dtype=v.dtype)
                        full[:n_valid] = v[start:end]
                        if k == "weight":
                            full[n_valid:] = 0.0
                        batch_data[k] = full
                else:
                    batch_data[k] = v

            feed = self._build_feed(input_names, batch_data, params, norm)
            result = self._run_batch(sess, output_names, feed)
            P_batch = result.get("P", np.zeros(n_valid, dtype=np.float32))

            if norm is None:
                # Norm model Q = sum(P * weight)
                w = batch_data.get("weight", np.ones(bs, dtype=np.float32))
                total_Q += float(np.sum(w[:n_valid] * P_batch[:n_valid]))
            else:
                # Forward model Q = NLL (already summed by the model)
                total_Q += float(result.get("Q", 0.0))

            # Accumulate gradients
            grads = self._extract_grads(result)
            if total_grads is None:
                total_grads = {k: v.copy() for k, v in grads.items()}
            else:
                for k in grads:
                    total_grads[k] = total_grads[k] + grads[k]

            all_P.append(P_batch[:n_valid])

        if len(all_P) == 1:
            P_all = all_P[0]
        else:
            P_all = np.concatenate(all_P, axis=0)

        return total_Q, total_grads, P_all
