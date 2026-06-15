"""CUDA backends — GPU-accelerated computation (f64 and f32)."""
import numpy as np
from .core import ComputeBackend, register_backend


@register_backend("cuda")
@register_backend("cuda64")
class CUDABackend(ComputeBackend):
    """CUDA-accelerated backend (float64, original GPUArray-style)."""
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
        if self._phsp_buffer is None: raise RuntimeError("No batched phsp prepared")
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

    def free(self): self.free_phsp_batched(); self.kernel.free()


# ── v2 backends (C-managed memory, linear interpolation) ──────

@register_backend("cuda_v2")
@register_backend("cuda64_v2")
class CUDABackendV2(ComputeBackend):
    """v2 CUDA backend — C-managed memory."""
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
    """v2 CUDA backend — float32."""
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


# ── v3 backends (Catmull-Rom interpolation) ───────────────────

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
    """v3 CUDA backend — float32 with Catmull-Rom."""
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
